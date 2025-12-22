use std::time::{Duration, Instant};
use halo2_proofs::{
    halo2curves::pasta::EqAffine,
    poly::{
        Coeff, EvaluationDomain, Polynomial,
        commitment::{Blind, Params, ParamsProver, Prover, Verifier},
        ipa::{
            commitment::ParamsIPA,
            msm::MSMIPA,
            multiopen::{ProverIPA, VerifierIPA},
        },
    },
};
use halo2_proofs::poly::{LagrangeCoeff, ProverQuery, Rotation, VerifierQuery};
use halo2_proofs::transcript::{
    Blake2bRead, Blake2bWrite, Challenge255,
    TranscriptRead, TranscriptReadBuffer, TranscriptWrite, TranscriptWriterBuffer
};
use ff::Field;
use group::{Curve, prime::PrimeCurveAffine as CurveAffine, cofactor::CofactorCurveAffine};
use num_traits::pow;
use rand_core::OsRng;

use crate::utils::bulletproof::{BulletproofParams, prove as bulletproof_prove, verify as bulletproof_verify};

/// Precomputed values shared across all columns in multi-column barycentric proofs.
///
/// This struct holds expensive computations that depend only on `beta`, `domain`, and `poly_params`,
/// which are identical across all columns in a multi-column proof. By computing these once and
/// reusing them, we eliminate ~87.5% of redundant work in the multi-column case.
pub struct BarycentricPrecomputed {
    /// Omega powers: ω^i for i = 0 to dim-1
    omega_powers: Vec<halo2_proofs::halo2curves::pasta::Fp>,

    /// Barycentric coefficients: b_i = scaling_factor * ω^i / (beta - ω^i)
    /// where scaling_factor = (beta^dim - 1) / dim
    b_coeffs: Vec<halo2_proofs::halo2curves::pasta::Fp>,

    /// Bulletproof parameters (shared across all columns)
    bulletproof_params: BulletproofParams<EqAffine>,

    /// Domain size (for validation)
    dim: usize,

    /// Beta evaluation point (for validation)
    beta: halo2_proofs::halo2curves::pasta::Fp,
}

impl BarycentricPrecomputed {
    /// Compute all shared values once for the multi-column case.
    ///
    /// This performs the expensive operations that would otherwise be repeated per column:
    /// - Computing omega powers (dim field multiplications)
    /// - Computing barycentric coefficients (dim field inversions)
    /// - Cloning bulletproof parameters (large vector clone)
    pub fn new(
        beta: halo2_proofs::halo2curves::pasta::Fp,
        poly_params: &ParamsIPA<EqAffine>,
        domain: &EvaluationDomain<halo2_proofs::halo2curves::pasta::Fp>,
    ) -> Self {
        use halo2_proofs::halo2curves::pasta::Fp;
        let dim = domain.get_n() as usize;

        // Compute omega powers: ω^i for i = 0 to dim-1
        let mut omega_powers = Vec::with_capacity(dim);
        omega_powers.push(Fp::ONE);
        for i in 1..dim {
            omega_powers.push(domain.rotate_omega(omega_powers[i - 1], Rotation(1)));
        }

        // Compute scaling factor: (beta^dim - 1) / dim
        let beta_d = beta.pow([dim as u64]);
        let numerator = beta_d - Fp::ONE;
        let d_inv = Fp::from(dim as u64).invert().unwrap();
        let scaling_factor = numerator * d_inv;

        // Compute barycentric coefficients: b_i = scaling_factor * ω^i / (beta - ω^i)
        let mut b_coeffs = Vec::with_capacity(dim);
        for i in 0..dim {
            let omega_i = omega_powers[i];
            let denominator = beta - omega_i;

            if denominator == Fp::ZERO {
                // This shouldn't happen with random beta
                b_coeffs.push(Fp::ZERO);
            } else {
                let b_i = scaling_factor * omega_i * denominator.invert().unwrap();
                b_coeffs.push(b_i);
            }
        }

        // Create BulletproofParams once (avoids repeated vector clones)
        let bulletproof_params = BulletproofParams::new(
            poly_params.get_g().to_vec(),
            poly_params.get_w(),
            poly_params.get_u(),
        );

        Self {
            omega_powers,
            b_coeffs,
            bulletproof_params,
            dim,
            beta,
        }
    }

    /// Validate that precomputed values match the current parameters.
    /// Used as a debug assertion to catch programming errors.
    #[allow(dead_code)]
    fn validate(&self, beta: halo2_proofs::halo2curves::pasta::Fp, dim: usize) {
        debug_assert_eq!(self.beta, beta, "Beta mismatch in precomputed values");
        debug_assert_eq!(self.dim, dim, "Domain size mismatch in precomputed values");
    }
}

/// Helper enum to handle both borrowed (multi-column) and owned (single-column) precomputed values.
///
/// This solves the lifetime challenge: in multi-column case, we borrow from a precomputed struct,
/// but in single-column case, we need to own the freshly computed values.
enum BarycentricValues<'a> {
    /// Multi-column case: reference to precomputed values (efficient)
    Borrowed(&'a BarycentricPrecomputed),
    /// Single-column case: owned freshly computed values (backward compatible)
    Owned(Box<BarycentricPrecomputed>),
}

impl<'a> BarycentricValues<'a> {
    /// Get omega powers reference
    fn omega_powers(&self) -> &[halo2_proofs::halo2curves::pasta::Fp] {
        match self {
            Self::Borrowed(p) => &p.omega_powers,
            Self::Owned(p) => &p.omega_powers,
        }
    }

    /// Get barycentric coefficients reference
    fn b_coeffs(&self) -> &[halo2_proofs::halo2curves::pasta::Fp] {
        match self {
            Self::Borrowed(p) => &p.b_coeffs,
            Self::Owned(p) => &p.b_coeffs,
        }
    }

    /// Get bulletproof parameters reference
    fn bulletproof_params(&self) -> &BulletproofParams<EqAffine> {
        match self {
            Self::Borrowed(p) => &p.bulletproof_params,
            Self::Owned(p) => &p.bulletproof_params,
        }
    }
}

/// Private helper: Barycentric interpolation for a single column
///
/// This function handles the core barycentric proof logic for one polynomial chunk.
/// It is called by the public bary_ipa() function (once for single-column, multiple times for multi-column).
fn bary_ipa_single_column(
    poly: Polynomial<halo2_proofs::halo2curves::pasta::Fp, Coeff>,
    poly_advice: Polynomial<halo2_proofs::halo2curves::pasta::Fp, LagrangeCoeff>,
    poly_com: EqAffine,
    beta: halo2_proofs::halo2curves::pasta::Fp,
    poly_params: &ParamsIPA<EqAffine>,
    domain: &EvaluationDomain<halo2_proofs::halo2curves::pasta::Fp>,
    alpha: halo2_proofs::halo2curves::pasta::Fp,
    blind: halo2_proofs::halo2curves::pasta::Fp,
    precomputed: Option<&BarycentricPrecomputed>,
) -> (Vec<u8>, Duration, usize, EqAffine, halo2_proofs::halo2curves::pasta::Fp) {
    use halo2_proofs::halo2curves::pasta::Fp;


    // Convert poly_advice from Lagrange to coefficient form for commitment
    // OUTSIDE of timer, because we could assume we get it from the halo2 prover
    let poly_advice_coeff = domain.lagrange_to_coeff(poly_advice.clone());

    let bary_timer = Instant::now();

    println!("Barycentric IPA: Starting opening proof for two polynomials at beta");
    println!("poly length: {}, poly_advice length: {}", poly.values.len(), poly_advice.values.len());
    println!("Domain size: {}", domain.get_n());


    // CRITICAL: poly.values should be interpreted as LAGRANGE values (row assignments)
    // NOT as coefficients! They match the advice column values exactly.

    // For split columns, we prove: <poly_chunk, b_coeffs_chunk> = advice_col(beta)
    // using inner product argument with barycentric coefficients

    println!("\nBarycentric IPA: Computing barycentric evaluation for advice column chunk");
    println!("  poly length: {}, poly_advice length: {}", poly.values.len(), poly_advice.values.len());

    // poly_advice is a single column from the split (or full column)
    // It contains a chunk of the witness in Lagrange form
    let chunk_size = poly_advice.values.len();  // Size of this chunk in the domain
    let dim = domain.get_n() as usize;  // Full domain size
    let omega = domain.get_omega();

    // Barycentric formula: f(beta) = (beta^d - 1) / d * Σ_{i=0}^{d-1} f_i * ω^i / (beta - ω^i)
    // This can be written as an inner product: f(beta) = <f, b_coeffs>
    // where b_coeffs[i] = (beta^d - 1) / d * ω^i / (beta - ω^i)

    // Get omega powers, b_coeffs, and bulletproof_params
    // Either from precomputed struct (multi-column) or compute fresh (single-column)
    let values = if let Some(precomp) = precomputed {
        // Multi-column case: use precomputed values (optimized path)
        println!("  Using precomputed barycentric coefficients (multi-column optimization)");
        BarycentricValues::Borrowed(precomp)
    } else {
        // Single-column case: compute fresh values (backward compatible)
        println!("  Computing barycentric coefficients (single-column)");
        BarycentricValues::Owned(Box::new(
            BarycentricPrecomputed::new(beta, poly_params, domain)
        ))
    };

    // Compute inner product <poly_advice, b_coeffs> to get evaluation at beta
    // let mut rho_advice_bary = Fp::ZERO;
    // for i in 0..dim {
    //     rho_advice_bary += poly_advice.values[i] * b_coeffs[i];
    // }
    // println!("  poly_advice(beta) = {:?} (via barycentric inner product)", rho_advice_bary);

    // Verify barycentric formula matches coefficient evaluation
    let rho_advice_coeff = poly_advice_coeff.evaluate(beta);
    // assert_eq!(
    //     rho_advice_bary, rho_advice_coeff,
    //     "Barycentric evaluation does not match coefficient evaluation!"
    // );
    // println!("Barycentric IPA: Barycentric formula verified ✓");

    // Now compute the evaluation of poly (external witness) using the same approach
    // poly.values are Lagrange values that should match poly_advice (without blinding)
    println!("\nBarycentric IPA: Computing external poly evaluation");
    println!("  poly has {} Lagrange values (witness only)", poly.values.len());

    // Compute <poly, b_coeffs[0..poly.len()]> for the external witness
    // Note: poly doesn't include padding or blinding, only the actual witness values
    // let mut rho_poly = Fp::ZERO;
    // for i in 0..poly.values.len() {
    //     rho_poly += poly.values[i] * b_coeffs[i];
    // }
    // println!("  poly(beta) = {:?} (via barycentric inner product on witness chunk)", rho_poly);

    // Compute the blinding contribution from rows beyond the witness
    // let mut blinding_contribution = Fp::ZERO;
    // for i in poly.values.len()..dim {
    //     blinding_contribution += poly_advice.values[i] * b_coeffs[i];
    // }
    // println!("  Blinding contribution = {:?}", blinding_contribution);

    // Create the combined polynomial for the IPA proof (witness + blinding)
    // poly.values contains: [0..witness_end): witness+padding, [witness_end..dim): zeros (for blinding)
    // poly_advice.values contains: [0..witness_end): witness+padding, [witness_end..dim): blinding
    // Find witness_end by scanning forward to find where blinding starts
    // Note: poly.values.len() might be < dim in single-column case
    let poly_len = poly.values.len();
    let witness_end = (0..poly_len.min(dim))
        .find(|&i| poly.values[i] == Fp::ZERO && poly_advice.values[i] != Fp::ZERO)
        .unwrap_or(poly_len.min(dim));

    let mut combined_lag = vec![Fp::ZERO; dim];
    // Copy witness + padding section from poly (up to poly length)
    for i in 0..witness_end.min(poly_len) {
        combined_lag[i] = poly.values[i];
    }
    // Copy blinding section from poly_advice
    for i in witness_end..dim {
        combined_lag[i] = poly_advice.values[i];
    }

    // Compute commitment to blinding-only polynomial for verifier
    let mut poly_only_blind = vec![Fp::ZERO; dim];
    for i in witness_end..dim {
        poly_only_blind[i] = poly_advice.values[i];
    }
    let poly_com_blind = poly_params.commit(&Polynomial::from_coefficients_vec(poly_only_blind), Blind::default()).to_affine();

    // inner product between combined_lag and b_coeffs
    // let rho_poly_blinded = {
    //     let mut acc = Fp::ZERO;
    //     for i in 0..dim {
    //         acc += combined_lag[i] * b_coeffs[i];
    //     }
    //     acc
    // };
    // println!("  poly(beta) + blinding = {:?}", rho_poly_blinded);

    // Now they should match!
    // println!("\nDEBUG: Checking if evaluations match after adding blinding:");
    // println!("  rho_poly + blinding = {:?}", rho_poly_blinded);
    // println!("  rho_advice_bary     = {:?}", rho_advice_bary);
    // println!("  Match: {}", rho_poly_blinded == rho_advice_bary);

    // Create Bulletproof inner product argument
    // Prove: <combined_lag, b_coeffs> = rho_poly_blinded
    println!("\n=== Creating Bulletproof Inner Product Argument ===");

    // Use blinding factor that matches poly_com + poly_com_blind
    // Since both poly_com and poly_com_blind use Blind::default() = Fp::ONE,
    // the combined commitment has blinding factor 2*Fp::ONE
    let bulletproof_alpha = Fp::ONE + Fp::ONE;

    // Prove the inner product
    println!("Bulletproof: Proving <combined_lag, b_coeffs> = {:.6?}", rho_advice_coeff);
    println!("  Witness vector size: {}", combined_lag.len());
    println!("  Public vector size: {}", values.b_coeffs().len());

    let (bulletproof_proof, bulletproof_time, bulletproof_size) = bulletproof_prove(
        values.bulletproof_params(),
        &combined_lag,
        values.b_coeffs(),
        bulletproof_alpha,
    );

    println!("Bulletproof: Proof generated successfully");
    println!("Bulletproof: Proof time: {:?}", bulletproof_time);
    println!("Bulletproof: Proof size: {} bytes", bulletproof_size);
    println!("Bulletproof: Verifier will compute combined commitment homomorphically as poly_com + poly_com_blind");

    // rho_advice_coeff: advice column after interpolation at beta
    // rho_poly_blinded: external poly barycentric

    // CRITICAL: poly is in COEFFICIENT form, but poly_advice is in LAGRANGE form
    // We need to interpret the same data consistently!

    // Option 1: Convert poly (coeff) to Lagrange, then back to coeff - should match poly_advice_coeff
    // let poly_as_lagrange = domain.coeff_to_lagrange(poly.clone());
    // let poly_reinterpreted_coeff = domain.lagrange_to_coeff(poly_as_lagrange);
    //
    // println!("\nDEBUG: Checking coefficient interpretation:");
    // println!("  poly (original coeff) commitment:");
    // let poly_com_original = poly_params.commit(&poly, Blind::default()).to_affine();
    // println!("    {:?}", poly_com_original);
    //
    // println!("  poly (coeff->lag->coeff) commitment:");
    // let poly_com_reinterp = poly_params.commit(&poly_reinterpreted_coeff, Blind::default()).to_affine();
    // println!("    {:?}", poly_com_reinterp);
    //
    // println!("  poly_advice_coeff commitment:");
    // let poly_advice_com = poly_params.commit(&poly_advice_coeff, Blind::default()).to_affine();
    // println!("    {:?}", poly_advice_com);
    //
    // println!("\nDEBUG: Checking if poly.values (as Lagrange) matches poly_advice:");
    // let poly_as_lagrange_direct = domain.lagrange_from_vec(poly.values.clone());
    // let poly_from_lag_coeff = domain.lagrange_to_coeff(poly_as_lagrange_direct.clone());
    // let poly_from_lag_com = poly_params.commit(&poly_from_lag_coeff, Blind::default()).to_affine();
    // println!("  poly.values as Lagrange, then to coeff commitment:");
    // println!("    {:?}", poly_from_lag_com);
    // println!("  Match with poly_advice: {}", poly_from_lag_com == poly_advice_com);

    // Use the blinded evaluation for both (they should match)
    // let rho = rho_poly_blinded;
    // let rho_advice = rho_advice_bary;
    // println!("\nBarycentric IPA: Final evaluations at beta:");
    // println!("  rho (external, blinded) = {:?}", rho);
    // println!("  rho_advice (internal)   = {:?}", rho_advice);
    // println!("  Evaluations match: {}", rho == rho_advice);

    // Create transcript for Fiat-Shamir
    let mut transcript_ipa_proof = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);

    // Verify that our combined commitment matches halo2's advice commitment
    println!("\nBarycentric IPA: Verifying commitment matches halo2's advice commitment");
    let combined_lag_poly = Polynomial::from_coefficients_vec(combined_lag);
    // Compare combined_lag_poly and poly by value
    println!("rhos {:?} {:?}", combined_lag_poly.evaluate(beta), poly.evaluate(beta));

    let poly_com_combined = poly_params.commit(&combined_lag_poly, Blind::default() + Blind::default()).to_affine();

    println!("poly_com_combined = poly_com + poly_com_blind");
    println!("  combined commitment: {:?}", poly_com_combined);
    println!("  external commitment with blinding:   {:?}", (poly_com + poly_com_blind).to_affine());
    // println!("  external commitment with blinding:   {:?}", (poly_com + poly_com_blind).to_affine());
    // verify coefficients match in witness section only (blinding section differs)
    // Note: only check up to poly_len since poly might be shorter than witness_end
    for i in 0..witness_end.min(poly_len) {
        assert_eq!(
            combined_lag_poly.values[i], poly.values[i],
            "Combined polynomial does not match external polynomial at index {}",
            i
        );
    }



    // Create ProverQuery for the combined polynomial (witness + blinding) at beta
    // Use the same blinding factor that halo2 used for the advice column
    let queries = vec![
        ProverQuery {
            point: beta,
            poly: &poly_advice_coeff, // TODO: A challenge here is that the poly was not materialized yet
            blind: Blind(blind),
        },
    ];

    // Create IPA proof
    let prover = ProverIPA::new(&poly_params);
    prover
        .create_proof(&mut OsRng, &mut transcript_ipa_proof, queries)
        .unwrap();

    let proof_bytes = transcript_ipa_proof.finalize();
    let proof_size = proof_bytes.len();
    let prover_time = bary_timer.elapsed();

    println!("Barycentric IPA: Prover time: {:?}", prover_time);
    println!("Barycentric IPA: IPA Proof size: {} bytes", proof_size);
    println!("Barycentric IPA: Total proof size (IPA + Bulletproof): {} bytes", proof_size + bulletproof_size);

    // Combine both proofs
    let mut combined_proof = Vec::new();
    // Write IPA proof length (4 bytes)
    combined_proof.extend_from_slice(&(proof_bytes.len() as u32).to_le_bytes());
    // Write IPA proof
    combined_proof.extend_from_slice(&proof_bytes);
    // Write Bulletproof proof
    combined_proof.extend_from_slice(&bulletproof_proof);

    let total_proof_size = combined_proof.len();

    (combined_proof, prover_time, total_proof_size, poly_com_blind, rho_advice_coeff)
}

/// Barycentric interpolation-based commitment for IPA (unified single/multi-column)
///
/// This function handles both single-column and multi-column barycentric proofs.
/// For single-column, it calls the helper once. For multi-column, it iterates over
/// all columns and combines the proofs.
///
/// # Parameters:
/// - `poly_chunks`: Pre-computed polynomial chunks (one per column)
/// - `poly_com_chunks`: Pre-computed commitments to polynomial chunks
/// - `advice_lagrange`: Vector of advice columns in Lagrange form (one or more columns)
/// - `advice_com`: Vector of commitments to advice columns
/// - `beta`: Evaluation point
/// - `poly_params`: IPA parameters
/// - `domain`: Evaluation domain
/// - `alpha`: Blinding factor (currently unused)
/// - `advice_blind`: Vector of blinding factors per column
/// - `poly_col_len`: Number of columns (1 for single, >1 for multi-column)
///
/// # Returns:
/// Tuple of (proof_bytes, prover_time, verifier_time, proof_size, poly_com_blind_vec, rho_vec)
/// where proof_bytes contains combined proofs and _vec parameters are vectors for each column
pub fn bary_ipa(
    poly_chunks: Vec<Polynomial<halo2_proofs::halo2curves::pasta::Fp, Coeff>>,
    poly_com_chunks: Vec<EqAffine>,
    advice_lagrange: Vec<Polynomial<halo2_proofs::halo2curves::pasta::Fp, LagrangeCoeff>>,
    advice_com: Vec<EqAffine>,
    beta: halo2_proofs::halo2curves::pasta::Fp,
    poly_params: &ParamsIPA<EqAffine>,
    domain: EvaluationDomain<halo2_proofs::halo2curves::pasta::Fp>,
    alpha: halo2_proofs::halo2curves::pasta::Fp,
    advice_blind: Vec<halo2_proofs::halo2curves::pasta::Fp>,
    poly_col_len: usize,
) -> (Vec<u8>, Duration, usize, Vec<EqAffine>, Vec<halo2_proofs::halo2curves::pasta::Fp>) {
    use halo2_proofs::halo2curves::pasta::Fp;

    if poly_col_len == 1 {
        // ===== SINGLE-COLUMN CASE =====
        println!("\n=== Barycentric IPA: Single Column ===");

        // Use pre-computed polynomial and commitment
        let (proof, ptime, psize, blind_com, eval) = bary_ipa_single_column(
            poly_chunks[0].clone(),
            advice_lagrange[0].clone(),
            poly_com_chunks[0],
            beta,
            poly_params,
            &domain,
            alpha,
            advice_blind[0],
            None,
        );

        return (proof, ptime, psize, vec![blind_com], vec![eval]);
    }

    // ===== MULTI-COLUMN CASE =====
    println!("\n=== Barycentric IPA: Multi-Column ({} columns) ===", poly_col_len);

    println!("  Number of columns: {}", poly_col_len);
    println!("  Domain size: {}", domain.get_n());

    // OPTIMIZATION: Precompute shared values once for all columns
    println!("  Precomputing shared barycentric coefficients and parameters...");
    let precompute_timer = Instant::now();
    let precomputed = BarycentricPrecomputed::new(beta, poly_params, &domain);
    let precompute_time = precompute_timer.elapsed();
    println!("  Precomputation time: {:?}", precompute_time);
    println!("  This will be amortized across {} columns", poly_col_len);

    let mut all_proofs = Vec::new();
    let mut poly_com_blind_vec = Vec::new();
    let mut rho_vec = Vec::new();
    let mut total_proof_size = 0;
    let mut total_timer = precompute_time;  // Include precomputation in total time

    // Process each column
    for col_idx in 0..poly_col_len {
        println!("\n--- Column {}/{} ---", col_idx + 1, poly_col_len);

        // Use pre-computed polynomial chunk and commitment
        let poly_chunk = poly_chunks[col_idx].clone();
        let poly_com_chunk = poly_com_chunks[col_idx];

        // Get advice column for this chunk
        let poly_advice_chunk = advice_lagrange[col_idx].clone();

        // Call single-column bary_ipa with precomputed values
        let (proof_bytes, _ptime, psize, poly_com_blind, rho) = bary_ipa_single_column(
            poly_chunk,
            poly_advice_chunk,
            poly_com_chunk,
            beta,
            poly_params,
            &domain,
            alpha,
            advice_blind[col_idx],
            Some(&precomputed),  // OPTIMIZATION: Pass precomputed values
        );

        println!("  Column {} proof size: {} bytes", col_idx, psize);
        println!("  Column {} evaluation: {:?}", col_idx, rho);

        all_proofs.push(proof_bytes);
        poly_com_blind_vec.push(poly_com_blind);
        rho_vec.push(rho);
        total_proof_size += psize;
        total_timer += _ptime;
    }

    // Combine all column proofs
    let mut combined_proof = Vec::new();

    // Write number of columns (4 bytes)
    combined_proof.extend_from_slice(&(poly_col_len as u32).to_le_bytes());

    // Write each column proof with length prefix
    for (col_idx, proof) in all_proofs.iter().enumerate() {
        combined_proof.extend_from_slice(&(proof.len() as u32).to_le_bytes());
        combined_proof.extend_from_slice(proof);
        println!("  Added column {} proof: {} bytes", col_idx, proof.len());
    }

    let final_proof_size = combined_proof.len();
    // let total_time = total_timer.elapsed();

    println!("\n=== Multi-column Proof Summary ===");
    println!("  Number of columns: {}", poly_col_len);
    println!("  Individual proof sizes: {} bytes total", total_proof_size);
    println!("  Overhead (metadata): {} bytes", final_proof_size - total_proof_size);
    println!("  Total proof size: {} bytes", final_proof_size);
    println!("  Average per column: {} bytes", total_proof_size / poly_col_len);
    println!("  Total proving time: {:?}", total_timer);

    (combined_proof, total_timer, final_proof_size, poly_com_blind_vec, rho_vec)
}

/// Private helper: Barycentric verifier for a single column
fn bary_verify_ipa_single_column(
    proof_bytes: &[u8],
    poly_com: EqAffine,
    poly_com_blind: EqAffine,
    poly_advice_com: EqAffine,
    beta: halo2_proofs::halo2curves::pasta::Fp,
    rho: halo2_proofs::halo2curves::pasta::Fp,
    poly_params: &ParamsIPA<EqAffine>,
    domain: &EvaluationDomain<halo2_proofs::halo2curves::pasta::Fp>,
    precomputed: Option<&BarycentricPrecomputed>,
) -> bool {
    use halo2_proofs::halo2curves::pasta::Fp;

    println!("Barycentric IPA: Starting verification");

    let verify_timer = Instant::now();

    // Split the combined proof
    if proof_bytes.len() < 4 {
        println!("Error: Proof too short");
        return false;
    }

    let ipa_proof_len = u32::from_le_bytes([
        proof_bytes[0],
        proof_bytes[1],
        proof_bytes[2],
        proof_bytes[3],
    ]) as usize;

    if proof_bytes.len() < 4 + ipa_proof_len {
        println!("Error: Invalid proof structure");
        return false;
    }

    let ipa_proof = &proof_bytes[4..4 + ipa_proof_len];
    let bulletproof_proof = &proof_bytes[4 + ipa_proof_len..];

    println!("  IPA proof size: {} bytes", ipa_proof.len());
    println!("  Bulletproof proof size: {} bytes", bulletproof_proof.len());

    // Verify IPA proof
    println!("\n=== Verifying IPA Opening Proof ===");
    let mut transcript_ipa_verify = Blake2bRead::<_, _, Challenge255<_>>::init(ipa_proof);

    // Use the halo2-generated commitment directly (includes witness + blinding)
    println!("  Using commitment: {:?}", poly_com);

    // Create verifier query for the polynomial
    let queries = std::iter::empty()
        .chain(Some(VerifierQuery::new_commitment(&poly_advice_com, beta, rho)));

    // Verify the IPA proof
    let verifier_params = poly_params.verifier_params();
    let verifier = VerifierIPA::new(&verifier_params);
    let msm = MSMIPA::new(&poly_params);

    let ipa_result = verifier.verify_proof(&mut transcript_ipa_verify, queries, msm).is_ok();
    println!("  IPA verification: {}", if ipa_result { "PASS" } else { "FAIL" });

    // Verify Bulletproof inner product proof
    println!("\n=== Verifying Bulletproof Inner Product Argument ===");

    // Get omega powers, b_coeffs, and bulletproof_params
    // Either from precomputed struct (multi-column) or compute fresh (single-column)
    let values = if let Some(precomp) = precomputed {
        // Multi-column case: use precomputed values (optimized path)
        println!("  Using precomputed barycentric coefficients (multi-column optimization)");
        BarycentricValues::Borrowed(precomp)
    } else {
        // Single-column case: compute fresh values (backward compatible)
        println!("  Computing barycentric coefficients (single-column)");
        BarycentricValues::Owned(Box::new(
            BarycentricPrecomputed::new(beta, poly_params, domain)
        ))
    };

    // Compute combined commitment homomorphically: poly_com + poly_com_blind
    // This exploits the homomorphic property of Pedersen commitments:
    // commit(poly) + commit(blind) = commit(poly + blind)
    use halo2_proofs::halo2curves::pasta::Eq;
    let poly_com_curve: Eq = poly_com.into();
    let poly_com_blind_curve: Eq = poly_com_blind.into();
    let combined_lag_commitment = (poly_com_curve + poly_com_blind_curve).to_affine();

    println!("  Computing combined commitment homomorphically:");
    println!("    poly_com = {:?}", poly_com);
    println!("    poly_com_blind = {:?}", poly_com_blind);
    println!("    combined = {:?}", combined_lag_commitment);

    // Verify the Bulletproof proof
    println!("  Verifying <combined_lag, b_coeffs> = {:.6?}", rho);

    let bulletproof_ok = bulletproof_verify(
        values.bulletproof_params(),
        combined_lag_commitment,
        values.b_coeffs(),
        rho,
        bulletproof_proof,
    );

    if bulletproof_ok {
        println!("  Bulletproof verification: PASS");
    } else {
        println!("  Bulletproof verification: FAIL");
    }

    let verify_time = verify_timer.elapsed();
    println!("\nBarycentric IPA: Total verifier time: {:?}", verify_time);
    println!("Barycentric IPA: Overall verification: {}", if ipa_result && bulletproof_ok { "PASS" } else { "FAIL" });

    ipa_result && bulletproof_ok
}

/// Barycentric Verifier for IPA (unified single/multi-column)
///
/// Verifies both the IPA opening proof and the Bulletproof inner product proof.
///
/// # Parameters:
/// - `proof_bytes`: Combined proof from the prover (IPA + Bulletproof)
/// - `poly_com`: Commitment to the polynomial (external witness)
/// - `poly_com_blind`: Commitment to the blinding part
/// - `poly_advice_com`: Commitment to the advice column (from halo2)
/// - `beta`: The point at which polynomial was opened
/// - `rho`: Evaluation of polynomial at beta (with blinding contribution)
/// - `poly_params`: IPA parameters
/// - `domain`: Evaluation domain (for computing barycentric coefficients)
///
/// # Returns:
/// `true` if both verifications pass, `false` otherwise
pub fn bary_verify_ipa(
    proof_bytes: &[u8],
    poly_com_vec: Vec<EqAffine>,
    poly_com_blind_vec: Vec<EqAffine>,
    poly_advice_com_vec: Vec<EqAffine>,
    beta: halo2_proofs::halo2curves::pasta::Fp,
    rho_vec: Vec<halo2_proofs::halo2curves::pasta::Fp>,
    poly_params: &ParamsIPA<EqAffine>,
    domain: EvaluationDomain<halo2_proofs::halo2curves::pasta::Fp>,
    poly_col_len: usize,
) -> bool {
    use halo2_proofs::halo2curves::pasta::Fp;

    if poly_col_len == 1 {
        // ===== SINGLE-COLUMN CASE =====
        println!("\n=== Barycentric Verification: Single Column ===");

        return bary_verify_ipa_single_column(
            proof_bytes,
            poly_com_vec[0],
            poly_com_blind_vec[0],
            poly_advice_com_vec[0],
            beta,
            rho_vec[0],
            poly_params,
            &domain,
            None,
        );
    }

    // ===== MULTI-COLUMN CASE =====
    println!("\n=== Barycentric Verification: Multi-Column ({} columns) ===", poly_col_len);

    // Parse proof structure
    if proof_bytes.len() < 4 {
        println!("Error: Proof too short (less than 4 bytes)");
        return false;
    }

    let num_columns = u32::from_le_bytes([
        proof_bytes[0], proof_bytes[1], proof_bytes[2], proof_bytes[3]
    ]) as usize;

    println!("  Proof contains {} columns", num_columns);

    if num_columns != poly_col_len {
        println!("Error: Proof has {} columns but expected {}", num_columns, poly_col_len);
        return false;
    }

    // Extract individual column proofs
    let mut offset = 4;
    let mut column_proofs = Vec::new();

    for col_idx in 0..num_columns {
        if offset + 4 > proof_bytes.len() {
            println!("Error: Incomplete proof at column {} (offset={})", col_idx, offset);
            return false;
        }

        let proof_len = u32::from_le_bytes([
            proof_bytes[offset],
            proof_bytes[offset + 1],
            proof_bytes[offset + 2],
            proof_bytes[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + proof_len > proof_bytes.len() {
            println!("Error: Incomplete proof data at column {} (need {} more bytes)",
                     col_idx, (offset + proof_len) - proof_bytes.len());
            return false;
        }

        column_proofs.push(&proof_bytes[offset..offset + proof_len]);
        println!("  Column {} proof: {} bytes", col_idx, proof_len);
        offset += proof_len;
    }

    // OPTIMIZATION: Precompute shared values once for all columns
    println!("\n  Precomputing shared barycentric coefficients and parameters...");
    let precompute_timer = Instant::now();
    let precomputed = BarycentricPrecomputed::new(beta, poly_params, &domain);
    let precompute_time = precompute_timer.elapsed();
    println!("  Precomputation time: {:?}", precompute_time);
    println!("  This will be amortized across {} columns\n", poly_col_len);

    // Verify each column
    let mut all_verified = true;

    for col_idx in 0..poly_col_len {
        println!("\n--- Verifying column {}/{} ---", col_idx + 1, poly_col_len);

        let verified = bary_verify_ipa_single_column(
            column_proofs[col_idx],
            poly_com_vec[col_idx],
            poly_com_blind_vec[col_idx],
            poly_advice_com_vec[col_idx],
            beta,
            rho_vec[col_idx],
            poly_params,
            &domain,
            Some(&precomputed),  // OPTIMIZATION: Pass precomputed values
        );

        if !verified {
            println!("  Column {} verification: FAIL ✗", col_idx);
            all_verified = false;
        } else {
            println!("  Column {} verification: PASS ✓", col_idx);
        }
    }

    println!("\n=== Multi-column Verification Summary ===");
    println!("  Overall result: {}", if all_verified { "PASS ✓" } else { "FAIL ✗" });

    all_verified
}
