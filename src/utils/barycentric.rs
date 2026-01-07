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
use group::{Curve, cofactor::CofactorCurveAffine};
use num_traits::pow;
use rand_core::OsRng;

use crate::utils::bulletproof::{BulletproofParams, prove as bulletproof_prove, verify as bulletproof_verify};

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
/// Tuple of (proof_bytes, prover_time, verifier_time, proof_size, poly_com_blind, rho)
/// where proof_bytes contains both IPA and Bulletproof proofs
pub fn bary_ipa(
    poly: Polynomial<halo2_proofs::halo2curves::pasta::Fp, Coeff>,
    poly_advice: Polynomial<halo2_proofs::halo2curves::pasta::Fp, LagrangeCoeff>,
    poly_com: EqAffine,
    beta: halo2_proofs::halo2curves::pasta::Fp,
    poly_params: &ParamsIPA<EqAffine>,
    domain: EvaluationDomain<halo2_proofs::halo2curves::pasta::Fp>,
    alpha: halo2_proofs::halo2curves::pasta::Fp,
    blind: halo2_proofs::halo2curves::pasta::Fp,  // Blinding factor from advice_blind
) -> (Vec<u8>, Duration, Duration, usize, EqAffine, halo2_proofs::halo2curves::pasta::Fp) {
    use halo2_proofs::halo2curves::pasta::Fp;

    let bary_timer = Instant::now();

    println!("Barycentric IPA: Starting opening proof for two polynomials at beta");
    println!("poly length: {}, poly_advice length: {}", poly.values.len(), poly_advice.values.len());
    println!("Domain size: {}", domain.get_n());

    // Convert poly_advice from Lagrange to coefficient form for commitment
    let poly_advice_coeff = domain.lagrange_to_coeff(poly_advice.clone());

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

    // Create the combined polynomial for the IPA proof (witness + blinding)
    let mut combined_lag = vec![Fp::ZERO; dim];
    for i in 0..poly.values.len() {
        combined_lag[i] = poly.values[i];
    }
    for i in poly.values.len()..dim {
        combined_lag[i] = poly_advice.values[i];
    }
    // inner product between combined_lag and b_coeffs
    let rho_poly_blinded = {
        let mut acc = Fp::ZERO;
        for i in 0..dim {
            acc += combined_lag[i] * b_coeffs[i];
        }
        acc
    };
    println!("  poly(beta) + blinding = {:?}", rho_poly_blinded);

    // Now they should match!
    println!("\nDEBUG: Checking if evaluations match after adding blinding:");
    println!("  rho_poly + blinding = {:?}", rho_poly_blinded);
    println!("  rho_advice_bary     = {:?}", rho_advice_bary);
    println!("  Match: {}", rho_poly_blinded == rho_advice_bary);

    // Create Bulletproof inner product argument
    // Prove: <combined_lag, b_coeffs> = rho_poly_blinded
    println!("\n=== Creating Bulletproof Inner Product Argument ===");

    // Create BulletproofParams from IPA parameters
    let bulletproof_params = BulletproofParams::new(
        poly_params.get_g().to_vec(),
        poly_params.get_w(),
        poly_params.get_u(),
    );

    // Use blinding factor that matches poly_com + poly_com_blind
    // Since both poly_com and poly_com_blind use Blind::default() = Fp::ONE,
    // the combined commitment has blinding factor 2*Fp::ONE
    let bulletproof_alpha = Fp::ONE + Fp::ONE;

    // Prove the inner product
    println!("Bulletproof: Proving <combined_lag, b_coeffs> = {:.6?}", rho_poly_blinded);
    println!("  Witness vector size: {}", combined_lag.len());
    println!("  Public vector size: {}", b_coeffs.len());

    let (bulletproof_proof, bulletproof_time, bulletproof_size) = bulletproof_prove(
        &bulletproof_params,
        &combined_lag,
        &b_coeffs,
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

    // Use the blinded evaluation for both (they should match)
    let rho = rho_poly_blinded;
    let rho_advice = rho_advice_bary;
    println!("\nBarycentric IPA: Final evaluations at beta:");
    println!("  rho (external, blinded) = {:?}", rho);
    println!("  rho_advice (internal)   = {:?}", rho_advice);
    println!("  Evaluations match: {}", rho == rho_advice);

    // Create transcript for Fiat-Shamir
    let mut transcript_ipa_proof = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);

    // Create the combined polynomial for the IPA proof (witness + blinding)
    let mut poly_only_blind = vec![Fp::ZERO; dim];
    for i in poly.values.len()..dim {
        poly_only_blind[i] = poly_advice.values[i];
    }
    let poly_com_blind = poly_params.commit(&Polynomial::from_coefficients_vec(poly_only_blind), Blind::default()).to_affine();

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
    // verify coefficients match
    for i in 0..poly.values.len() {
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

    (combined_proof, prover_time, Duration::from_micros(0), total_proof_size, poly_com_blind, rho_poly_blinded)
}

/// Barycentric Verifier for IPA with Bulletproof Inner Product Argument
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
    poly_com: EqAffine,
    poly_com_blind: EqAffine,
    poly_advice_com: EqAffine,
    beta: halo2_proofs::halo2curves::pasta::Fp,
    rho: halo2_proofs::halo2curves::pasta::Fp,
    poly_params: &ParamsIPA<EqAffine>,
    domain: EvaluationDomain<halo2_proofs::halo2curves::pasta::Fp>,
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

    // Reconstruct barycentric coefficients (same as prover)
    let dim = domain.get_n() as usize;
    let omega = domain.get_omega();

    // Precompute omega powers
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

    // Compute barycentric coefficients
    let mut b_coeffs = Vec::with_capacity(dim);
    for i in 0..dim {
        let omega_i = omega_powers[i];
        let denominator = beta - omega_i;

        if denominator == Fp::ZERO {
            println!("Warning: beta equals a root of unity at index {}", i);
            b_coeffs.push(Fp::ZERO);
        } else {
            let b_i = scaling_factor * omega_i * denominator.invert().unwrap();
            b_coeffs.push(b_i);
        }
    }

    // Create BulletproofParams from IPA parameters
    let bulletproof_params = BulletproofParams::new(
        poly_params.get_g().to_vec(),
        poly_params.get_w(),
        poly_params.get_u(),
    );

    // Compute combined commitment homomorphically: poly_com + poly_com_blind
    // This exploits the homomorphic property of Pedersen commitments:
    // commit(poly) + commit(blind) = commit(poly + blind)
    let combined_lag_commitment = (poly_com.to_curve() + poly_com_blind.to_curve()).to_affine();

    println!("  Computing combined commitment homomorphically:");
    println!("    poly_com = {:?}", poly_com);
    println!("    poly_com_blind = {:?}", poly_com_blind);
    println!("    combined = {:?}", combined_lag_commitment);

    // Verify the Bulletproof proof
    println!("  Verifying <combined_lag, b_coeffs> = {:.6?}", rho);

    let bulletproof_ok = bulletproof_verify(
        &bulletproof_params,
        combined_lag_commitment,
        &b_coeffs,
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
