// Bulletproof-style Inner Product Argument
//
// Generic implementation of the Bulletproof inner product argument protocol.
// Proves knowledge of a committed vector `a` such that <a, b> = c for public vector `b`.
//
// Based on the Bulletproofs paper: https://eprint.iacr.org/2017/1066.pdf
//
// ## Protocol Variant
//
// This is a SINGLE-VECTOR variant where only `a` is committed (private) and `b` is public.
// This differs from the standard Bulletproofs where both vectors are committed.
//
// **Prover (each round):**
// - L = <a_L, G_R> + c_L*U + alpha_L*H  where c_L = <a_L, b_R>
// - R = <a_R, G_L> + c_R*U + alpha_R*H  where c_R = <a_R, b_L>
// - Fold: a' = a_L*e + a_R*e^(-1), b' = b_L*e^(-1) + b_R*e, G' = G_L*e^(-1) + G_R*e
//
// **Verifier:**
// - Start with: P = C + c*U  (where C = <a,G> + alpha*H, c = <a,b>)
// - Fold: P' = L*e^2 + P + R*e^(-2)
// - Final check: P' == a_final*G_final + (a_final*b_final)*U + alpha_final*H
//
// This variant is more efficient when one vector is public (as in barycentric evaluation).
//
// ## Verification
//
// **Mathematical Correctness:**
// The recursive folding preserves the inner product relation. After one round:
//   P' = L*e^2 + P + R*e^(-2)
//      = <a', G'> + <a', b'>*U + alpha'*H
// where:
//   a' = a_L*e + a_R*e^(-1)
//   b' = b_L*e^(-1) + b_R*e
//   G' = G_L*e^(-1) + G_R*e
//   alpha' = alpha + alpha_L*e^2 + alpha_R*e^(-2)
//
// This can be verified by expanding <a', G'> and <a', b'>:
//   <a', G'> = <a_L*e + a_R*e^(-1), G_L*e^(-1) + G_R*e>
//            = <a_L, G_L> + <a_L, G_R>*e^2 + <a_R, G_L>*e^(-2) + <a_R, G_R>
//   <a', b'> = <a_L, b_L> + <a_L, b_R>*e^2 + <a_R, b_L>*e^(-2) + <a_R, b_R>
//
// After log₂(n) rounds, we get vectors of size 1, and the final check verifies:
//   P_final == a_final*G_final + (a_final*b_final)*U + alpha_final*H
//
// **Comparison with Standard Bulletproofs:**
// In the standard two-vector variant (both a and b committed):
// - L = <a_L, G_R> + <b_R, H_L> + <a_L, b_R>*U + alpha_L*W
// - R = <a_R, G_L> + <b_L, H_R> + <a_R, b_L>*U + alpha_R*W
//
// In our single-vector variant (b is public):
// - L = <a_L, G_R> + <a_L, b_R>*U + alpha_L*H
// - R = <a_R, G_L> + <a_R, b_L>*U + alpha_R*H
//
// The <b_R, H_L> and <b_L, H_R> terms are eliminated since b is public.
//
// **Reference Implementation:**
// Verified against zkcrypto/bulletproofs:
// https://github.com/zkcrypto/bulletproofs/blob/main/src/inner_product_proof.rs
//
// **Test Results:**
// All 6 tests pass, including:
// - Basic correctness with and without blinding
// - Wrong witness rejection
// - Wrong inner product rejection
// - Zero witness handling
// - Various vector sizes (n = 16, 32, 64, 128)
//
// Proof sizes: 64*log₂(n) + 64 bytes
// - n=16:  320 bytes (4 rounds)
// - n=32:  384 bytes (5 rounds)
// - n=64:  448 bytes (6 rounds)
// - n=128: 512 bytes (7 rounds)

use std::time::{Duration, Instant};
use halo2_proofs::{
    arithmetic::best_multiexp,
    halo2curves::CurveAffine,
    poly::ipa::commitment::ParamsIPA,
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, Transcript,
        TranscriptRead, TranscriptReadBuffer, TranscriptWrite, TranscriptWriterBuffer
    },
};
use ff::Field;
use group::Curve;
use rand_core::OsRng;

/// Parameters for the Bulletproof inner product argument
///
/// Contains the generator points and auxiliary points needed for the protocol.
/// The vector size `n` must be a power of 2.
pub struct BulletproofParams<C: CurveAffine> {
    /// Vector of generators G[0..n]
    pub g_vec: Vec<C>,
    /// Blinding generator H (for hiding the witness)
    pub h: C,
    /// Auxiliary generator U (for the inner product term)
    pub u: C,
    /// Vector size (must be power of 2)
    pub n: usize,
}

impl<C: CurveAffine> BulletproofParams<C> {
    /// Create new Bulletproof parameters
    ///
    /// # Arguments
    /// * `g_vec` - Vector of generator points (length must be power of 2)
    /// * `h` - Blinding generator
    /// * `u` - Auxiliary generator for inner product term
    ///
    /// # Panics
    /// Panics if `g_vec.len()` is not a power of 2
    pub fn new(g_vec: Vec<C>, h: C, u: C) -> Self {
        let n = g_vec.len();
        assert!(n.is_power_of_two(), "Vector size must be a power of 2");
        assert!(n > 0, "Vector size must be positive");

        Self { g_vec, h, u, n }
    }

}

// Note: ParamsIPA fields (w, u) are private to halo2_proofs crate.
// When integrating with barycentric code, generators will need to be
// extracted in a different way or passed explicitly.

/// Compute inner product of two scalar vectors
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Inner product <a, b> = sum(a[i] * b[i])
fn inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    a.iter()
        .zip(b.iter())
        .map(|(&a_i, &b_i)| a_i * b_i)
        .fold(F::ZERO, |acc, x| acc + x)
}

/// Fold two scalar vectors using challenge
///
/// Computes: result[i] = left[i] * challenge + right[i] * challenge_inv
///
/// # Arguments
/// * `left` - Left half vector
/// * `right` - Right half vector
/// * `challenge` - Challenge scalar
/// * `challenge_inv` - Inverse of challenge
fn fold_scalars<F: Field>(left: &[F], right: &[F], challenge: F, challenge_inv: F) -> Vec<F> {
    assert_eq!(left.len(), right.len(), "Vectors must have same length");

    left.iter()
        .zip(right.iter())
        .map(|(&l, &r)| l * challenge + r * challenge_inv)
        .collect()
}

/// Fold two point vectors using challenge
///
/// Computes: result[i] = left[i] * challenge_inv + right[i] * challenge
/// (Note: inverted compared to scalars for the inner product to be preserved)
///
/// Uses best_multiexp for efficient batch computation (2-element multiexp per fold)
///
/// # Arguments
/// * `left` - Left half of generator vector
/// * `right` - Right half of generator vector
/// * `challenge` - Challenge scalar
/// * `challenge_inv` - Inverse of challenge
fn fold_points<C: CurveAffine>(
    left: &[C],
    right: &[C],
    challenge: C::Scalar,
    challenge_inv: C::Scalar,
) -> Vec<C> {
    assert_eq!(left.len(), right.len(), "Vectors must have same length");

    // For each point, compute G'[i] = G_L[i] * e^(-1) + G_R[i] * e using multiexp
    let n = left.len();
    let mut result = Vec::with_capacity(n);

    // Optimization: Reuse scalar and point arrays instead of allocating every iteration
    let mut scalars = [challenge_inv, challenge];
    let mut points = [left[0], right[0]];

    for i in 0..n {
        points[0] = left[i];
        points[1] = right[i];
        let folded = best_multiexp(&scalars, &points).to_affine();
        result.push(folded);
    }

    result
}

/// Bulletproof inner product argument - Prover
///
/// Proves knowledge of witness vector `a` such that:
/// - C = <a, G> + alpha * H (commitment)
/// - <a, b> = c (inner product relation)
///
/// # Arguments
/// * `params` - Bulletproof parameters
/// * `a` - Secret witness vector (length must match params.n)
/// * `b` - Public vector (known to verifier)
/// * `alpha` - Blinding factor for commitment
///
/// # Returns
/// `(proof_bytes, prover_time, proof_size)`
pub fn prove<C: CurveAffine>(
    params: &BulletproofParams<C>,
    a: &[C::Scalar],
    b: &[C::Scalar],
    alpha: C::Scalar,
) -> (Vec<u8>, Duration, usize)
where
    C::Scalar: ff::PrimeField + ff::FromUniformBytes<64>,
{
    let timer = Instant::now();
    let n = params.n;

    // Validate inputs
    assert_eq!(a.len(), n, "Witness vector must match parameter size");
    assert_eq!(b.len(), n, "Public vector must match parameter size");
    assert!(n.is_power_of_two(), "Vector size must be power of 2");

    // Initialize transcript
    let mut transcript = Blake2bWrite::<Vec<u8>, C, Challenge255<C>>::init(vec![]);

    // Initialize working vectors (clone to avoid mutating inputs)
    let mut a_vec = a.to_vec();
    let mut b_vec = b.to_vec();
    let mut g_vec = params.g_vec.clone();

    // Track accumulated blinding factor
    let mut alpha_acc = alpha;

    // Number of folding rounds = log2(n)
    let num_rounds = (n as f64).log2() as usize;

    println!("Bulletproof Prover: n={}, rounds={}", n, num_rounds);

    // Preallocate buffers for L/R commitments (reused across all rounds)
    let max_half_size = n / 2;
    let mut l_scalars = Vec::with_capacity(max_half_size + 2);
    let mut l_points = Vec::with_capacity(max_half_size + 2);
    let mut r_scalars = Vec::with_capacity(max_half_size + 2);
    let mut r_points = Vec::with_capacity(max_half_size + 2);

    // Recursive folding
    for round in 0..num_rounds {
        let half = a_vec.len() / 2;

        println!("  Round {}: vector size = {}", round, a_vec.len());

        // Split vectors in half
        let (a_l, a_r) = a_vec.split_at(half);
        let (b_l, b_r) = b_vec.split_at(half);
        let (g_l, g_r) = g_vec.split_at(half);

        // Compute cross inner products
        let c_l = inner_product(a_l, b_r);
        let c_r = inner_product(a_r, b_l);

        // Generate random blinding factors
        let alpha_l = C::Scalar::random(OsRng);
        let alpha_r = C::Scalar::random(OsRng);

        // Compute L commitment: <a_L, G_R> + c_L * U + alpha_L * H
        // Reuse preallocated buffers
        l_scalars.clear();
        l_scalars.extend_from_slice(a_l);
        l_scalars.push(c_l);
        l_scalars.push(alpha_l);

        l_points.clear();
        l_points.extend_from_slice(g_r);
        l_points.push(params.u);
        l_points.push(params.h);

        let l_commitment = best_multiexp(&l_scalars, &l_points).to_affine();

        // Compute R commitment: <a_R, G_L> + c_R * U + alpha_R * H
        // Reuse preallocated buffers
        r_scalars.clear();
        r_scalars.extend_from_slice(a_r);
        r_scalars.push(c_r);
        r_scalars.push(alpha_r);

        r_points.clear();
        r_points.extend_from_slice(g_l);
        r_points.push(params.u);
        r_points.push(params.h);

        let r_commitment = best_multiexp(&r_scalars, &r_points).to_affine();

        // Write L, R to transcript
        transcript.write_point(l_commitment).expect("Failed to write L to transcript");
        transcript.write_point(r_commitment).expect("Failed to write R to transcript");

        // Get challenge from transcript (Fiat-Shamir)
        let challenge: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
        let challenge_inv = challenge.invert().unwrap();

        // Fold vectors
        a_vec = fold_scalars(a_l, a_r, challenge, challenge_inv);
        b_vec = fold_scalars(b_l, b_r, challenge_inv, challenge);
        g_vec = fold_points(g_l, g_r, challenge, challenge_inv);

        // Update accumulated blinding factor: alpha += alpha_L * e^2 + alpha_R * e^{-2}
        alpha_acc = alpha_acc + alpha_l * challenge.square() + alpha_r * challenge_inv.square();
    }

    // Final values (vectors now have size 1)
    assert_eq!(a_vec.len(), 1, "Final vector should have size 1");
    let a_final = a_vec[0];
    let alpha_final = alpha_acc;

    println!("  Final: a={:?}, alpha={:?}", a_final, alpha_final);

    // Write final values to transcript
    transcript.write_scalar(a_final).expect("Failed to write a_final");
    transcript.write_scalar(alpha_final).expect("Failed to write alpha_final");

    let proof_bytes = transcript.finalize();
    let proof_size = proof_bytes.len();
    let prover_time = timer.elapsed();

    println!("Bulletproof Prover: Proof size = {} bytes, time = {:?}", proof_size, prover_time);

    (proof_bytes, prover_time, proof_size)
}

/// Bulletproof inner product argument - Verifier
///
/// Verifies a proof that commitment C opens to witness a satisfying <a, b> = c.
///
/// # Arguments
/// * `params` - Bulletproof parameters (same as prover)
/// * `commitment` - Commitment C = <a, G> + alpha * H
/// * `b` - Public vector (same as prover)
/// * `c` - Claimed inner product result
/// * `proof` - Proof bytes from prover
///
/// # Returns
/// `true` if proof verifies, `false` otherwise
pub fn verify<C: CurveAffine>(
    params: &BulletproofParams<C>,
    commitment: C,
    b: &[C::Scalar],
    c: C::Scalar,
    proof: &[u8],
) -> bool
where
    C::Scalar: ff::PrimeField + ff::FromUniformBytes<64>,
{
    let timer = Instant::now();
    let n = params.n;

    // Validate inputs
    assert_eq!(b.len(), n, "Public vector must match parameter size");

    // Initialize read transcript
    let mut transcript = Blake2bRead::<&[u8], C, Challenge255<C>>::init(proof);

    // Initialize verification state
    // CRITICAL: Start with commitment adjusted by claimed inner product
    // P = C + c * U (where c is the claimed inner product <a, b>)
    // Use multiexp for efficient computation
    let scalars = vec![C::Scalar::ONE, c];
    let points = vec![commitment, params.u];
    let mut commitment_acc = best_multiexp(&scalars, &points).to_affine();
    let mut b_vec = b.to_vec();
    let mut g_vec = params.g_vec.clone();

    // Number of rounds
    let num_rounds = (n as f64).log2() as usize;

    println!("Bulletproof Verifier: n={}, rounds={}", n, num_rounds);

    // Preallocate buffers for commitment folding (reused across all rounds)
    let mut commitment_fold_scalars = vec![C::Scalar::ZERO; 3];
    let mut commitment_fold_points = vec![commitment; 3];

    // Process each round
    for round in 0..num_rounds {
        let half = b_vec.len() / 2;

        println!("  Round {}: vector size = {}", round, b_vec.len());

        // Read L, R from proof
        let l_commitment = transcript.read_point().expect("Failed to read L");
        let r_commitment = transcript.read_point().expect("Failed to read R");

        // Get challenge (must match prover's transcript)
        let challenge: C::Scalar = *transcript.squeeze_challenge_scalar::<()>();
        let challenge_inv = challenge.invert().unwrap();

        // Fold commitment: C' = L * e^2 + C + R * e^{-2}
        // Use preallocated buffers for efficient batch computation
        let challenge_sq = challenge.square();
        let challenge_inv_sq = challenge_inv.square();
        commitment_fold_scalars[0] = challenge_sq;
        commitment_fold_scalars[1] = C::Scalar::ONE;
        commitment_fold_scalars[2] = challenge_inv_sq;
        commitment_fold_points[0] = l_commitment;
        commitment_fold_points[1] = commitment_acc;
        commitment_fold_points[2] = r_commitment;
        commitment_acc = best_multiexp(&commitment_fold_scalars, &commitment_fold_points).to_affine();

        // Fold b vector
        let (b_l, b_r) = b_vec.split_at(half);
        b_vec = fold_scalars(b_l, b_r, challenge_inv, challenge);

        // Fold generator vector
        let (g_l, g_r) = g_vec.split_at(half);
        g_vec = fold_points(g_l, g_r, challenge, challenge_inv);
    }

    // Read final values
    let a_final = transcript.read_scalar().expect("Failed to read a_final");
    let alpha_final = transcript.read_scalar().expect("Failed to read alpha_final");

    println!("  Final: a={:?}, alpha={:?}", a_final, alpha_final);

    // Verify final equation: C' == a_final * G'[0] + (a_final * b'[0]) * U + alpha_final * H
    assert_eq!(g_vec.len(), 1, "Final generator vector should have size 1");
    assert_eq!(b_vec.len(), 1, "Final b vector should have size 1");

    let g_final = g_vec[0];
    let b_final = b_vec[0];

    // Compute expected commitment
    let expected_commitment = best_multiexp(
        &[a_final, a_final * b_final, alpha_final],
        &[g_final, params.u, params.h],
    ).to_affine();

    let verified = commitment_acc == expected_commitment;

    let verify_time = timer.elapsed();
    println!("Bulletproof Verifier: Result = {}, time = {:?}", verified, verify_time);

    verified
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::halo2curves::pasta::{EqAffine, Fp};
    use halo2_proofs::poly::ipa::commitment::ParamsIPA;
    use halo2_proofs::poly::commitment::ParamsProver;
    use ff::Field;

    /// Helper to create test parameters
    fn create_test_params(n: usize) -> BulletproofParams<EqAffine> {
        use group::Group;
        use halo2_proofs::halo2curves::pasta::Eq;

        // Create random generators for testing
        // In real usage, these would come from a trusted setup
        let mut g_vec = Vec::with_capacity(n);
        for _ in 0..n {
            g_vec.push(Eq::random(OsRng).to_affine());
        }

        let h = Eq::random(OsRng).to_affine();  // Blinding generator
        let u = Eq::random(OsRng).to_affine();  // Auxiliary generator

        BulletproofParams::new(g_vec, h, u)
    }

    /// Helper to create commitment C = <a, G> + alpha * H
    fn commit(
        params: &BulletproofParams<EqAffine>,
        a: &[Fp],
        alpha: Fp,
    ) -> EqAffine {
        assert_eq!(a.len(), params.n);

        let mut scalars = a.to_vec();
        scalars.push(alpha);

        let mut points = params.g_vec.clone();
        points.push(params.h);

        best_multiexp(&scalars, &points).to_affine()
    }

    #[test]
    fn test_bulletproof_no_blinding() {
        println!("\n=== Test: No Blinding (Debug) ===");

        let n = 16;
        let params = create_test_params(n);

        // Create random witness and public vector
        let a: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
        let b: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
        let alpha = Fp::ZERO;  // No blinding

        // Compute inner product
        let c = inner_product(&a, &b);
        println!("  Inner product c = {:?}", c);

        // Create commitment (no blinding)
        let commitment = commit(&params, &a, alpha);

        // Generate proof
        let (proof, prover_time, proof_size) = prove(&params, &a, &b, alpha);

        println!("  Prover time: {:?}, Proof size: {} bytes", prover_time, proof_size);

        // Verify proof
        let verified = verify(&params, commitment, &b, c, &proof);

        println!("  Verified: {}", verified);
        assert!(verified, "Proof should verify for correct witness (no blinding)");
    }

    #[test]
    fn test_bulletproof_basic() {
        println!("\n=== Test: Basic Correctness ===");

        let n = 16;
        let params = create_test_params(n);

        // Create random witness and public vector
        let a: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
        let b: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
        let alpha = Fp::random(OsRng);

        // Compute inner product
        let c = inner_product(&a, &b);

        // Create commitment
        let commitment = commit(&params, &a, alpha);

        // Generate proof
        let (proof, prover_time, proof_size) = prove(&params, &a, &b, alpha);

        println!("Prover time: {:?}, Proof size: {} bytes", prover_time, proof_size);

        // Verify proof
        let verified = verify(&params, commitment, &b, c, &proof);

        assert!(verified, "Proof should verify for correct witness");
    }

    #[test]
    fn test_bulletproof_wrong_witness() {
        println!("\n=== Test: Wrong Witness Fails ===");

        let n = 16;
        let params = create_test_params(n);

        // Create witness a and public b
        let a: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
        let b: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
        let alpha = Fp::random(OsRng);

        // Compute correct inner product
        let c = inner_product(&a, &b);

        // Generate proof with witness a
        let (proof, _, _) = prove(&params, &a, &b, alpha);

        // Create commitment to different witness a'
        let a_prime: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
        let wrong_commitment = commit(&params, &a_prime, alpha);

        // Verification should fail
        let verified = verify(&params, wrong_commitment, &b, c, &proof);

        assert!(!verified, "Proof should fail for wrong witness");
    }

    #[test]
    fn test_bulletproof_wrong_inner_product() {
        println!("\n=== Test: Wrong Inner Product Fails ===");

        let n = 16;
        let params = create_test_params(n);

        let a: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
        let b: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
        let alpha = Fp::random(OsRng);

        // Compute correct inner product
        let c = inner_product(&a, &b);
        let wrong_c = c + Fp::ONE;  // Incorrect claimed result

        let commitment = commit(&params, &a, alpha);

        // Generate proof
        let (proof, _, _) = prove(&params, &a, &b, alpha);

        // Verify with wrong inner product
        let verified = verify(&params, commitment, &b, wrong_c, &proof);

        assert!(!verified, "Proof should fail for wrong inner product claim");
    }

    #[test]
    fn test_bulletproof_various_sizes() {
        println!("\n=== Test: Various Vector Sizes ===");

        for k in 4..8 {
            let n = 1 << k;  // 16, 32, 64, 128
            println!("\n  Testing n = {}", n);

            let params = create_test_params(n);

            let a: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
            let b: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
            let alpha = Fp::random(OsRng);

            let c = inner_product(&a, &b);
            let commitment = commit(&params, &a, alpha);

            let (proof, prover_time, proof_size) = prove(&params, &a, &b, alpha);
            let verified = verify(&params, commitment, &b, c, &proof);

            println!("    n={}: proof_size={} bytes, prover_time={:?}, verified={}",
                     n, proof_size, prover_time, verified);

            assert!(verified, "Proof should verify for n={}", n);
        }
    }

    #[test]
    fn test_bulletproof_zero_witness() {
        println!("\n=== Test: Zero Witness ===");

        let n = 16;
        let params = create_test_params(n);

        // All-zero witness
        let a = vec![Fp::ZERO; n];
        let b: Vec<Fp> = (0..n).map(|_| Fp::random(OsRng)).collect();
        let alpha = Fp::random(OsRng);

        // Inner product should be zero
        let c = inner_product(&a, &b);
        assert_eq!(c, Fp::ZERO);

        let commitment = commit(&params, &a, alpha);

        let (proof, _, _) = prove(&params, &a, &b, alpha);
        let verified = verify(&params, commitment, &b, c, &proof);

        assert!(verified, "Proof should verify for zero witness");
    }
}
