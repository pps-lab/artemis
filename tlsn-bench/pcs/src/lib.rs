//! External commitment schemes for the Artemis link on a TLSNotary-style
//! predicate proof.
//!
//! The committed vector is the packed TLS payload `u` (one BN254 scalar per 31
//! bytes) followed by one mask coefficient `mu`. Both schemes commit to the
//! polynomial `p(X) = sum_i a_i X^i` and open it at a point `z`.
//!
//! * [`ipa`]: Pedersen vector commitment over BN254 G1 with a zero-knowledge
//!   inner-product opening. Transparent (generators come from hash-to-curve),
//!   no pairing. This is the scheme the ProveKit instantiation uses.
//! * [`kzg`]: hiding KZG10 over BN254. The benchmark setup samples the
//!   trapdoor locally, which is fine for timing and unusable for anything else.
//! * [`lego`]: LegoSNARK CP_link (Kiltz-Wee QA-NIZK) between a Pedersen
//!   commitment and KZG commitments to wire columns, for the Barretenberg
//!   cost estimate (`lego_bench`).

pub mod ipa;
pub mod kzg;
pub mod lego;
mod transcript;

pub use ark_bn254::{Fr, G1Affine, G1Projective};

/// Evaluate `sum_i coeffs[i] * z^i`.
pub fn eval_poly(coeffs: &[Fr], z: Fr) -> Fr {
    coeffs
        .iter()
        .rev()
        .fold(Fr::from(0u64), |acc, c| acc * z + c)
}

/// Pack bytes into field elements, 31 bytes per element, little endian. The
/// circuit side uses the same packing (`PACK` = 31).
pub fn pack_bytes(bytes: &[u8]) -> Vec<Fr> {
    use ark_ff::PrimeField;
    bytes
        .chunks(31)
        .map(Fr::from_le_bytes_mod_order)
        .collect()
}
