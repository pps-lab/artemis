//! Hiding KZG10 over BN254 (degree-1 hiding polynomial, as in Marlin's KZG).
//!
//! The setup below samples the trapdoor locally. Use it for timing only.

use {
    ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Affine, G2Projective},
    ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM},
    ark_ff::{One, UniformRand, Zero},
    ark_serialize::{CanonicalDeserialize, CanonicalSerialize},
    ark_std::rand::RngCore,
    rayon::prelude::*,
};

#[derive(Clone, Debug)]
pub struct Srs {
    /// `tau^i * G`.
    pub g:      Vec<G1Affine>,
    /// `H`, `tau * H`.
    pub h:      [G1Affine; 2],
    pub g2:     G2Affine,
    pub tau_g2: G2Affine,
}

#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Proof {
    pub w:           G1Affine,
    /// Evaluation of the hiding polynomial at the opening point.
    pub hiding_eval: Fr,
}

impl Proof {
    pub fn size_bytes(&self) -> usize {
        self.compressed_size()
    }
}

impl Srs {
    /// Insecure setup (known trapdoor) for polynomials with up to `n`
    /// coefficients.
    pub fn insecure_setup<R: RngCore>(n: usize, rng: &mut R) -> Self {
        let tau = Fr::rand(rng);
        let mut pows = Vec::with_capacity(n);
        let mut p = Fr::one();
        for _ in 0..n {
            pows.push(p);
            p *= tau;
        }
        let g1 = G1Projective::generator();
        let g: Vec<G1Projective> = pows.par_iter().map(|p| g1 * p).collect();
        let h = g1 * Fr::rand(rng);
        let g2 = G2Projective::generator();
        Self {
            g: G1Projective::normalize_batch(&g),
            h: [h.into_affine(), (h * tau).into_affine()],
            g2: g2.into_affine(),
            tau_g2: (g2 * tau).into_affine(),
        }
    }

    /// Commit with hiding polynomial `blind[0] + blind[1] X`.
    pub fn commit(&self, coeffs: &[Fr], blind: [Fr; 2]) -> G1Projective {
        assert!(coeffs.len() <= self.g.len());
        G1Projective::msm_unchecked(&self.g[..coeffs.len()], coeffs)
            + self.h[0] * blind[0]
            + self.h[1] * blind[1]
    }

    pub fn open(&self, coeffs: &[Fr], blind: [Fr; 2], z: Fr) -> Proof {
        // Synthetic division of p(X) - p(z) by (X - z).
        let mut q = vec![Fr::zero(); coeffs.len().saturating_sub(1)];
        let mut carry = Fr::zero();
        for i in (1..coeffs.len()).rev() {
            carry = coeffs[i] + carry * z;
            q[i - 1] = carry;
        }
        // The hiding polynomial has degree 1, so its quotient is blind[1].
        let w = G1Projective::msm_unchecked(&self.g[..q.len()], &q) + self.h[0] * blind[1];
        Proof {
            w:           w.into_affine(),
            hiding_eval: blind[0] + blind[1] * z,
        }
    }

    pub fn verify(&self, commitment: &G1Affine, z: Fr, v: Fr, proof: &Proof) -> bool {
        let lhs = commitment.into_group()
            - G1Projective::generator() * v
            - self.h[0] * proof.hiding_eval;
        let rhs_g2 = self.tau_g2.into_group() - self.g2 * z;
        Bn254::multi_pairing([lhs.into_affine(), (-proof.w.into_group()).into_affine()], [
            self.g2,
            rhs_g2.into_affine(),
        ])
        .0
        .is_one()
    }
}

#[cfg(test)]
mod tests {
    use {super::*, crate::eval_poly, ark_std::test_rng};

    #[test]
    fn roundtrip_and_reject() {
        let mut rng = test_rng();
        let srs = Srs::insecure_setup(40, &mut rng);
        let coeffs: Vec<Fr> = (0..37).map(|_| Fr::rand(&mut rng)).collect();
        let blind = [Fr::rand(&mut rng), Fr::rand(&mut rng)];
        let c = srs.commit(&coeffs, blind).into_affine();
        let z = Fr::rand(&mut rng);
        let v = eval_poly(&coeffs, z);
        let proof = srs.open(&coeffs, blind, z);
        assert!(srs.verify(&c, z, v, &proof));
        assert!(!srs.verify(&c, z, v + Fr::one(), &proof));
        assert!(!srs.verify(&c, z + Fr::one(), v, &proof));
    }
}
