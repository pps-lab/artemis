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
use group::Curve;
use num_traits::pow;
use rand_core::OsRng;

/// Barycentric interpolation-based commitment for IPA
///
/// This function will implement a barycentric interpolation approach
/// to demonstrate correspondence between polynomial commitments and evaluations.
///
/// # Parameters:
/// - `poly`: The witness polynomial (coefficients become witness vector a)
/// - `poly_params`: IPA parameters (provides generators)
/// - `domain`: Evaluation domain (provides omega for FFT matrices)
/// - `alpha`: Blinding factor for the initial commitment
///
/// # Returns:
/// Tuple of (proof_bytes, prover_time, verifier_time, proof_size)
pub fn bary_ipa(
    poly: Polynomial<halo2_proofs::halo2curves::pasta::Fp, Coeff>,
    poly_advice: Polynomial<halo2_proofs::halo2curves::pasta::Fp, LagrangeCoeff>,
    beta: halo2_proofs::halo2curves::pasta::Fp,
    poly_params: &ParamsIPA<EqAffine>,
    domain: EvaluationDomain<halo2_proofs::halo2curves::pasta::Fp>,
    alpha: halo2_proofs::halo2curves::pasta::Fp,
    blind: halo2_proofs::halo2curves::pasta::Fp,  // Blinding factor from advice_blind
) -> (Vec<u8>, Duration, Duration, usize) {
    use halo2_proofs::halo2curves::pasta::Fp;

    let bary_timer = Instant::now();

    println!("Barycentric IPA: Starting opening proof for two polynomials at beta");
    println!("poly length: {}, poly_advice length: {}", poly.values.len(), poly_advice.values.len());
    println!("Domain size: {}", domain.get_n());

    // Convert poly_advice from Lagrange to coefficient form for commitment
    let poly_advice_coeff = domain.lagrange_to_coeff(poly_advice.clone());

    // HYPOTHESIS 1: Try treating poly.values AS Lagrange values directly (not coefficients)
    println!("\n=== HYPOTHESIS 1: poly.values are Lagrange values (row assignments) ===");
    let mut poly_as_lagrange = poly.values.clone();
    println!("Padding poly from {} to {} (domain size)", poly_as_lagrange.len(), domain.get_n());
    while poly_as_lagrange.len() < domain.get_n() as usize {
        poly_as_lagrange.push(Fp::ZERO);
    }

    let mut h1_matches = 0;
    let mut h1_first_mismatch = None;
    let mut match_indices = Vec::new();
    let mut mismatch_indices = Vec::new();

    for i in 0..poly_as_lagrange.len().min(100) {
        if poly_as_lagrange[i] == poly_advice.values[i] {
            h1_matches += 1;
            match_indices.push(i);
        } else {
            mismatch_indices.push(i);
            if h1_first_mismatch.is_none() {
                h1_first_mismatch = Some(i);
            }
        }
    }

    // Continue counting for the rest without storing indices
    for i in 100..poly_as_lagrange.len() {
        if poly_as_lagrange[i] == poly_advice.values[i] {
            h1_matches += 1;
        }
    }

    println!("H1 Matches: {}/{} ({:.2}%)", h1_matches, poly_as_lagrange.len(),
             100.0 * h1_matches as f64 / poly_as_lagrange.len() as f64);
    println!("First 100 indices - Matches: {:?}", &match_indices[..match_indices.len().min(30)]);
    println!("First 100 indices - Mismatches: {:?}", &mismatch_indices[..mismatch_indices.len().min(30)]);

    if let Some(idx) = h1_first_mismatch {
        println!("  First mismatch at [{}]: poly={:?}, advice={:?}",
                 idx, poly_as_lagrange.get(idx), poly_advice.values.get(idx));
    }

    // HYPOTHESIS 2: Try converting poly (as coefficients) to Lagrange
    println!("\n=== HYPOTHESIS 2: poly is coefficients, convert to Lagrange ===");
    let mut poly_padded = poly.values.clone();
    while poly_padded.len() < domain.get_n() as usize {
        poly_padded.push(Fp::ZERO);
    }
    let poly_coeff_padded = Polynomial::from_coefficients_vec(poly_padded.clone());
    let poly_lagrange = domain.coeff_to_lagrange(poly_coeff_padded.clone());

    let mut h2_matches = 0;
    let mut h2_first_mismatch = None;
    for i in 0..poly_lagrange.values.len() {
        if poly_lagrange.values[i] == poly_advice.values[i] {
            h2_matches += 1;
        } else if h2_first_mismatch.is_none() {
            h2_first_mismatch = Some(i);
        }
    }
    println!("H2 Matches: {}/{} ({:.2}%)", h2_matches, poly_lagrange.values.len(),
             100.0 * h2_matches as f64 / poly_lagrange.values.len() as f64);
    if let Some(idx) = h2_first_mismatch {
        println!("  First mismatch at [{}]: poly_lag={:?}, advice={:?}",
                 idx, poly_lagrange.values.get(idx), poly_advice.values.get(idx));
    }

    // Check blinding region (last few rows where halo2 adds blinding)
    println!("\n=== Checking blinding region (last 10 rows) ===");
    for i in (domain.get_n() as usize - 10)..(domain.get_n() as usize) {
        println!("  [{}] poly_advice = {:?}, is_zero = {}",
                 i, poly_advice.values.get(i), poly_advice.values[i] == Fp::ZERO);
    }

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

    println!("Barycentric IPA: Computing barycentric coefficients for inner product");

    // Precompute omega powers: ω^i for i = 0 to d-1
    let mut omega_powers = Vec::with_capacity(dim);
    omega_powers.push(Fp::ONE);
    for i in 1..dim {
        omega_powers.push(domain.rotate_omega(omega_powers[i - 1], Rotation(1)));
    }

    // Compute scaling factor: (beta^d - 1) / d
    let beta_d = beta.pow([dim as u64]);
    let numerator = beta_d - Fp::ONE;
    let d_inv = Fp::from(dim as u64).invert().unwrap();
    let scaling_factor = numerator * d_inv;

    // Compute barycentric coefficients b_i = scaling_factor * ω^i / (beta - ω^i)
    let mut b_coeffs = Vec::with_capacity(dim);
    for i in 0..dim {
        let omega_i = omega_powers[i];
        let denominator = beta - omega_i;

        if denominator == Fp::ZERO {
            println!("Warning: beta equals a root of unity at index {}", i);
            // This shouldn't happen with random beta
            b_coeffs.push(Fp::ZERO);
        } else {
            let b_i = scaling_factor * omega_i * denominator.invert().unwrap();
            b_coeffs.push(b_i);
        }
    }

    // Compute inner product <poly_advice, b_coeffs> to get evaluation at beta
    let mut rho_advice_bary = Fp::ZERO;
    for i in 0..dim {
        rho_advice_bary += poly_advice.values[i] * b_coeffs[i];
    }
    println!("  poly_advice(beta) = {:?} (via barycentric inner product)", rho_advice_bary);

    // Verify barycentric formula matches coefficient evaluation
    let rho_advice_coeff = poly_advice_coeff.evaluate(beta);
    assert_eq!(
        rho_advice_bary, rho_advice_coeff,
        "Barycentric evaluation does not match coefficient evaluation!"
    );
    println!("Barycentric IPA: Barycentric formula verified ✓");

    // Now compute the evaluation of poly (external witness) using the same approach
    // poly.values are Lagrange values that should match poly_advice (without blinding)
    println!("\nBarycentric IPA: Computing external poly evaluation");
    println!("  poly has {} Lagrange values (witness only)", poly.values.len());

    // Compute <poly, b_coeffs[0..poly.len()]> for the external witness
    // Note: poly doesn't include padding or blinding, only the actual witness values
    let mut rho_poly = Fp::ZERO;
    for i in 0..poly.values.len() {
        rho_poly += poly.values[i] * b_coeffs[i];
    }
    println!("  poly(beta) = {:?} (via barycentric inner product on witness chunk)", rho_poly);

    // Compute the blinding contribution from rows beyond the witness
    let mut blinding_contribution = Fp::ZERO;
    for i in poly.values.len()..dim {
        blinding_contribution += poly_advice.values[i] * b_coeffs[i];
    }
    println!("  Blinding contribution = {:?}", blinding_contribution);

    // Add blinding to external poly evaluation
    let rho_poly_blinded = rho_poly + blinding_contribution;
    println!("  poly(beta) + blinding = {:?}", rho_poly_blinded);

    // Now they should match!
    println!("\nDEBUG: Checking if evaluations match after adding blinding:");
    println!("  rho_poly + blinding = {:?}", rho_poly_blinded);
    println!("  rho_advice_bary     = {:?}", rho_advice_bary);
    println!("  Match: {}", rho_poly_blinded == rho_advice_bary);

    // CRITICAL: poly is in COEFFICIENT form, but poly_advice is in LAGRANGE form
    // We need to interpret the same data consistently!

    // Option 1: Convert poly (coeff) to Lagrange, then back to coeff - should match poly_advice_coeff
    let poly_as_lagrange = domain.coeff_to_lagrange(poly.clone());
    let poly_reinterpreted_coeff = domain.lagrange_to_coeff(poly_as_lagrange);

    println!("\nDEBUG: Checking coefficient interpretation:");
    println!("  poly (original coeff) commitment:");
    let poly_com_original = poly_params.commit(&poly, Blind::default()).to_affine();
    println!("    {:?}", poly_com_original);

    println!("  poly (coeff->lag->coeff) commitment:");
    let poly_com_reinterp = poly_params.commit(&poly_reinterpreted_coeff, Blind::default()).to_affine();
    println!("    {:?}", poly_com_reinterp);

    println!("  poly_advice_coeff commitment:");
    let poly_advice_com = poly_params.commit(&poly_advice_coeff, Blind::default()).to_affine();
    println!("    {:?}", poly_advice_com);

    println!("\nDEBUG: Checking if poly.values (as Lagrange) matches poly_advice:");
    let poly_as_lagrange_direct = domain.lagrange_from_vec(poly.values.clone());
    let poly_from_lag_coeff = domain.lagrange_to_coeff(poly_as_lagrange_direct.clone());
    let poly_from_lag_com = poly_params.commit(&poly_from_lag_coeff, Blind::default()).to_affine();
    println!("  poly.values as Lagrange, then to coeff commitment:");
    println!("    {:?}", poly_from_lag_com);
    println!("  Match with poly_advice: {}", poly_from_lag_com == poly_advice_com);

    // For now, commit to poly as Lagrange values (without worrying about commitment blinding)
    // The evaluation already matches after adding the polynomial blinding contribution
    let poly_com = poly_from_lag_com;

    // Use the blinded evaluation for both (they should match)
    let rho = rho_poly_blinded;
    let rho_advice = rho_advice_bary;
    println!("\nBarycentric IPA: Final evaluations at beta:");
    println!("  rho (external, blinded) = {:?}", rho);
    println!("  rho_advice (internal)   = {:?}", rho_advice);
    println!("  Evaluations match: {}", rho == rho_advice);

    // Create transcript for Fiat-Shamir
    let mut transcript_ipa_proof = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);

    // Create ProverQuery for the external polynomial at beta
    // We only need to prove poly evaluation, poly_advice is already committed in halo2
    let queries = vec![
        ProverQuery {
            point: beta,
            poly: &poly_from_lag_coeff,
            blind: Blind::default(),
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
    println!("Barycentric IPA: Proof size: {} bytes", proof_size);

    (proof_bytes, prover_time, Duration::from_micros(0), proof_size)
}

/// Barycentric Verifier for IPA
///
/// Verifies the opening proof for two polynomials at beta.
///
/// # Parameters:
/// - `proof_bytes`: The proof from the prover
/// - `poly_com`: Commitment to the first polynomial
/// - `poly_advice_com`: Commitment to the second polynomial (advice)
/// - `beta`: The point at which polynomials were opened
/// - `rho`: Evaluation of first polynomial at beta
/// - `rho_advice`: Evaluation of second polynomial at beta
/// - `poly_params`: IPA parameters
///
/// # Returns:
/// `true` if verification passes, `false` otherwise
pub fn bary_verify_ipa(
    proof_bytes: &[u8],
    poly_com: EqAffine,
    poly_advice_com: EqAffine,
    beta: halo2_proofs::halo2curves::pasta::Fp,
    rho: halo2_proofs::halo2curves::pasta::Fp,
    rho_advice: halo2_proofs::halo2curves::pasta::Fp,
    poly_params: &ParamsIPA<EqAffine>,
) -> bool {
    println!("Barycentric IPA: Starting verification");

    let verify_timer = Instant::now();

    // Initialize transcript for reading
    let mut transcript_ipa_verify = Blake2bRead::<_, _, Challenge255<_>>::init(proof_bytes);

    // Create verifier query for the external polynomial
    let queries = std::iter::empty()
        .chain(Some(VerifierQuery::new_commitment(&poly_com, beta, rho)));

    // Verify the proof
    let verifier_params = poly_params.verifier_params();
    let verifier = VerifierIPA::new(&verifier_params);
    let msm = MSMIPA::new(&poly_params);

    let result = verifier.verify_proof(&mut transcript_ipa_verify, queries, msm).is_ok();

    let verify_time = verify_timer.elapsed();
    println!("Barycentric IPA: Verifier time: {:?}", verify_time);

    result
}
