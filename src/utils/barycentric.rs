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
) -> (Vec<u8>, Duration, Duration, usize) {
    use halo2_proofs::halo2curves::pasta::Fp;

    let bary_timer = Instant::now();

    println!("Barycentric IPA: Starting opening proof for two polynomials at beta");

    // Convert poly_advice from Lagrange to coefficient form for commitment
    let poly_advice_coeff = domain.lagrange_to_coeff(poly_advice.clone());

    for i in 0..poly.values.len() {
        println!("poly_advice val {} {:?} {:?}", i, poly_advice.values.get(poly_advice.values.len() - i), poly.values.get(i));
    }

    // poly advice coef are the coefficient form poly of the advice cols, f(x) = a_i(x)
    // poly coef the coeffs = the witness
    // we want: evaluate 'the poly' by interpreting the coefficients as evaluations, and evaluate the same
    let dim = poly_advice.values.len();
    let omega = domain.get_omega();

    // Evaluate poly_advice at beta using barycentric formula for roots of unity:
    // f(z) = (z^d - 1) / d * Σ_{i=0}^{d-1} f_i * ω^i / (z - ω^i)
    println!("Barycentric IPA: Using barycentric formula for poly_advice evaluation");

    // Precompute omega powers: ω^i for i = 0 to d-1
    let mut omega_powers = Vec::with_capacity(dim);
    omega_powers.push(Fp::ONE);
    for i in 1..dim {
        omega_powers.push(domain.rotate_omega(omega_powers[i - 1], Rotation(1)));
    }

    // Compute the sum: Σ_{i=0}^{d-1} f_i * ω^i / (beta - ω^i)
    let mut sum = Fp::ZERO;
    for i in 0..dim {
        let f_i = poly_advice.values[i];
        let omega_i = omega_powers[i];
        let denominator = beta - omega_i;

        // Handle case where beta equals a root of unity (should not happen in practice)
        if denominator == Fp::ZERO {
            println!("Warning: beta equals a root of unity at index {}", i);
            sum = f_i;
            break;
        }

        let term = f_i * omega_i * denominator.invert().unwrap();
        sum += term;
    }

    // Compute (beta^d - 1) / d
    let beta_d = beta.pow([dim as u64]);
    let numerator = beta_d - Fp::ONE;
    let d_inv = Fp::from(dim as u64).invert().unwrap();
    let scaling_factor = numerator * d_inv;

    // Final barycentric evaluation
    let rho_advice_bary = scaling_factor * sum;

    // Verify barycentric formula matches coefficient evaluation
    let rho_advice_coeff = poly_advice_coeff.evaluate(beta);
    assert_eq!(
        rho_advice_bary, rho_advice_coeff,
        "Barycentric evaluation does not match coefficient evaluation!"
    );
    println!("Barycentric IPA: Barycentric formula verified ✓");

    // Commit to both polynomials
    let poly_com = poly_params.commit(&poly, Blind::default()).to_affine();
    let poly_advice_com = poly_params.commit(&poly_advice_coeff, Blind::default()).to_affine();
    println!("Barycentric IPA: Computed commitments");

    // Evaluate both polynomials at beta
    let rho = poly.evaluate(beta);
    let rho_advice = rho_advice_bary;
    println!("Barycentric IPA: Evaluations at beta: rho={:?}, rho_advice={:?}", rho, rho_advice);

    // Create transcript for Fiat-Shamir
    let mut transcript_ipa_proof = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);

    // Create ProverQuery for both polynomials at beta
    let queries = vec![
        ProverQuery {
            point: beta,
            poly: &poly,
            blind: Blind::default(),
        },
        ProverQuery {
            point: beta,
            poly: &poly_advice_coeff,
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

    // Create verifier queries for both polynomials
    let queries = std::iter::empty()
        .chain(Some(VerifierQuery::new_commitment(&poly_com, beta, rho)))
        .chain(Some(VerifierQuery::new_commitment(&poly_advice_com, beta, rho_advice)));

    // Verify the proof
    let verifier_params = poly_params.verifier_params();
    let verifier = VerifierIPA::new(&verifier_params);
    let msm = MSMIPA::new(&poly_params);

    let result = verifier.verify_proof(&mut transcript_ipa_verify, queries, msm).is_ok();

    let verify_time = verify_timer.elapsed();
    println!("Barycentric IPA: Verifier time: {:?}", verify_time);

    result
}
