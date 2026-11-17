//! LegoSNARK CP_link (Campanelli, Fiore, Querol, CCS'19) between a Pedersen
//! commitment and KZG commitments to wire columns: the Kiltz-Wee QA-NIZK for
//! linear subspaces.
//!
//! A wire column of UltraHonk is committed as `W = sum_s w_s * B_s`, where
//! `B_s = [tau^row(s)]_1` and `s` runs over the cells of the column that can
//! be non-zero (the active trace rows and the ZK masking rows). The external
//! commitment is `c = r * h_0 + sum_j u_j * h_{1+j}`. Each committed element is
//! a public linear combination of wire cells: cell `(wire, slot)` contributes
//! `coeff * h_{1+ped}` ([`Link`]). With `x = (r, w^(1), ..., w^(ell))`,
//!
//! ```text
//!   (c, W_1, ..., W_ell)^T = M x,
//! ```
//!
//! where row 0 of `M` holds `h_0` under `r` and `coeff * h_{1+ped}` under
//! every linked cell, and row `i` holds `B_s` under cell `s` of wire `i`.
//!
//! * Setup: `k <- F^{ell+1}`, `a <- F`; `[P]_1 = [M^T k]_1` (one G1 element
//!   per column of `M`), `[C_i]_2 = [a k_i]_2`, `[a]_2`.
//! * Prove: `pi = [x^T P]_1`, one MSM over `r` and every non-zero cell of
//!   every linked wire.
//! * Verify: `e(pi, [a]_2) = e(c, [C_0]_2) * prod_i e(W_i, [C_i]_2)`, one
//!   multi-pairing of `ell + 2` pairings.
//!
//! The setup is specific to the circuit (the position of the linked cells) and
//! samples the trapdoor `k` locally: timing only, like the KZG setup of this
//! crate. It needs no trapdoor of the KZG reference string, so it runs on
//! Barretenberg's real points.

use {
    ark_bn254::{Bn254, Fr, G1Affine, G1Projective, G2Projective},
    ark_ec::{
        pairing::Pairing, AffineRepr, CurveGroup, PrimeGroup, ScalarMul, VariableBaseMSM,
    },
    ark_ff::{BigInteger, BigInteger256, One, UniformRand},
    ark_std::rand::RngCore,
    rayon::prelude::*,
};

type G2Prepared = <Bn254 as Pairing>::G2Prepared;

/// Cell `slot` of linked wire `wire` contributes `coeff * h[1 + ped]` to the
/// external commitment.
#[derive(Clone, Copy, Debug)]
pub struct Link {
    pub wire:  usize,
    pub slot:  usize,
    pub ped:   usize,
    pub coeff: Fr,
}

pub struct Crs {
    /// Number of linked wire commitments.
    pub ell:   usize,
    /// Cells per linked wire.
    pub slots: usize,
    /// Pedersen key: `h[0]` for the randomness, `h[1 + j]` for element `j`.
    pub h:     Vec<G1Affine>,
    /// `[M^T k]_1`: `p[0]` for `r`, `p[1 + wire * slots + slot]` for a cell.
    pub p:     Vec<G1Affine>,
    /// `[a k_i]_2`; index 0 belongs to the Pedersen commitment.
    pub c:     Vec<G2Prepared>,
    /// `[a]_2`.
    pub a:     G2Prepared,
}

impl Crs {
    /// `bases[s]` is the KZG point of cell `s` (the same for every wire).
    pub fn setup<R: RngCore>(
        bases: &[G1Affine],
        ell: usize,
        elements: usize,
        links: &[Link],
        rng: &mut R,
    ) -> Self {
        assert!(ell >= 1);
        let slots = bases.len();
        let g1 = G1Projective::generator();
        let h_scalars: Vec<Fr> = (0..=elements).map(|_| Fr::rand(rng)).collect();
        let h = g1.batch_mul(&h_scalars);

        let k: Vec<Fr> = (0..=ell).map(|_| Fr::rand(rng)).collect();
        let a = Fr::rand(rng);

        // Row 0 of M under the wire cells.
        let mut row0: Vec<Option<(u32, Fr)>> = vec![None; ell * slots];
        for l in links {
            assert!(l.wire < ell && l.slot < slots && l.ped < elements);
            let cell = &mut row0[l.wire * slots + l.slot];
            assert!(cell.is_none(), "two links on one cell");
            *cell = Some((l.ped as u32, l.coeff));
        }
        let mut p: Vec<G1Projective> = Vec::with_capacity(1 + ell * slots);
        p.push(h[0] * k[0]);
        p.par_extend(row0.par_iter().enumerate().map(|(cell, link)| {
            let mut pt = bases[cell % slots] * k[1 + cell / slots];
            if let Some((ped, coeff)) = link {
                pt += h[1 + *ped as usize] * (k[0] * coeff);
            }
            pt
        }));
        let p = G1Projective::normalize_batch(&p);

        let g2 = G2Projective::generator();
        let prep = |s: Fr| G2Prepared::from((g2 * s).into_affine());
        Self {
            ell,
            slots,
            h,
            p,
            c: k.iter().map(|ki| prep(a * ki)).collect(),
            a: prep(a),
        }
    }

    /// Size of the prover key `[P]_1` and of the Pedersen key in bytes
    /// (uncompressed affine points, 64 bytes each).
    pub fn key_bytes(&self) -> (usize, usize) {
        (64 * self.p.len(), 64 * self.h.len())
    }

    /// `c = r * h_0 + sum_j u_j * h_{1+j}`.
    pub fn commit(&self, r: Fr, elements: &[BigInteger256]) -> G1Projective {
        assert_eq!(elements.len() + 1, self.h.len());
        self.h[0] * r + G1Projective::msm_bigint(&self.h[1..], elements)
    }

    /// `pi = [x^T P]_1`. Zero cells are skipped; arkworks' MSM handles small
    /// scalars (bytes, range-check limbs) in separate, cheaper buckets.
    pub fn prove(&self, r: Fr, wires: &[&[BigInteger256]]) -> G1Projective {
        assert_eq!(wires.len(), self.ell);
        let mut pi = self.p[0] * r;
        for (i, wire) in wires.iter().enumerate() {
            assert_eq!(wire.len(), self.slots);
            let keys = &self.p[1 + i * self.slots..1 + (i + 1) * self.slots];
            let (bases, scalars): (Vec<G1Affine>, Vec<BigInteger256>) = wire
                .par_iter()
                .zip(keys)
                .filter(|(s, _)| !s.is_zero())
                .map(|(s, b)| (*b, *s))
                .unzip();
            pi += G1Projective::msm_bigint(&bases, &scalars);
        }
        pi
    }

    pub fn verify(&self, pi: &G1Affine, c: &G1Affine, wire_coms: &[G1Affine]) -> bool {
        assert_eq!(wire_coms.len(), self.ell);
        let neg_pi = (-pi.into_group()).into_affine();
        let mut g1 = vec![neg_pi, *c];
        g1.extend_from_slice(wire_coms);
        let mut g2 = vec![self.a.clone()];
        g2.extend(self.c.iter().cloned());
        Bn254::multi_pairing(g1, g2).0.is_one()
    }
}

/// KZG commitment to a wire column: `sum_s w_s * bases[s]`.
pub fn commit_wire(bases: &[G1Affine], wire: &[BigInteger256]) -> G1Projective {
    assert_eq!(bases.len(), wire.len());
    let (b, s): (Vec<G1Affine>, Vec<BigInteger256>) = wire
        .par_iter()
        .zip(bases)
        .filter(|(s, _)| !s.is_zero())
        .map(|(s, b)| (*b, *s))
        .unzip();
    G1Projective::msm_bigint(&b, &s)
}

/// `[tau^i]_1` for a locally sampled `tau` (tests).
pub fn insecure_kzg_bases<R: RngCore>(n: usize, rng: &mut R) -> Vec<G1Affine> {
    let tau = Fr::rand(rng);
    let mut pows = Vec::with_capacity(n);
    let mut p = Fr::one();
    for _ in 0..n {
        pows.push(p);
        p *= tau;
    }
    G1Projective::generator().batch_mul(&pows)
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        ark_ff::{Field, PrimeField, Zero},
        ark_std::test_rng,
    };

    /// Two wires of 16 cells. Six bytes sit in cells of both wires; the
    /// commitment is to the two packed elements `b0 + 256 b1 + 256^2 b2`.
    fn fixture() -> (Vec<G1Affine>, Vec<Link>, [Vec<Fr>; 2], Vec<Fr>) {
        let mut rng = test_rng();
        let bases = insecure_kzg_bases(16, &mut rng);
        let bytes: Vec<u64> = vec![7, 250, 31, 0, 99, 180];
        let cells = [(0, 3), (1, 5), (0, 9), (1, 0), (1, 14), (0, 1)];
        let mut wires = [
            (0..16).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>(),
            (0..16).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>(),
        ];
        wires[0][7] = Fr::zero();
        let mut links = vec![];
        for (j, ((wire, slot), b)) in cells.iter().zip(&bytes).enumerate() {
            wires[*wire][*slot] = Fr::from(*b);
            links.push(Link {
                wire:  *wire,
                slot:  *slot,
                ped:   j / 3,
                coeff: Fr::from(256u64).pow([(j % 3) as u64]),
            });
        }
        let elements = bytes
            .chunks(3)
            .map(|c| Fr::from(c[0] + 256 * c[1] + 65536 * c[2]))
            .collect();
        (bases, links, wires, elements)
    }

    fn big(v: &[Fr]) -> Vec<BigInteger256> {
        v.iter().map(|x| x.into_bigint()).collect()
    }

    #[test]
    fn honest_proof_verifies() {
        let mut rng = test_rng();
        let (bases, links, wires, elements) = fixture();
        let crs = Crs::setup(&bases, 2, 2, &links, &mut rng);
        let r = Fr::rand(&mut rng);
        let c = crs.commit(r, &big(&elements)).into_affine();
        let (w0, w1) = (big(&wires[0]), big(&wires[1]));
        let coms = [
            commit_wire(&bases, &w0).into_affine(),
            commit_wire(&bases, &w1).into_affine(),
        ];
        let pi = crs.prove(r, &[&w0, &w1]).into_affine();
        assert!(crs.verify(&pi, &c, &coms));
        // A different proof element, commitment or wire order is rejected.
        let other = (pi.into_group() + G1Projective::generator()).into_affine();
        assert!(!crs.verify(&other, &c, &coms));
        assert!(!crs.verify(&pi, &coms[0], &coms));
        assert!(!crs.verify(&pi, &c, &[coms[1], coms[0]]));
    }

    #[test]
    fn other_payload_cell_is_rejected() {
        let mut rng = test_rng();
        let (bases, links, mut wires, elements) = fixture();
        let crs = Crs::setup(&bases, 2, 2, &links, &mut rng);
        let r = Fr::rand(&mut rng);
        let c = crs.commit(r, &big(&elements)).into_affine();
        // The circuit runs on another byte in a linked cell.
        wires[1][5] += Fr::one();
        let (w0, w1) = (big(&wires[0]), big(&wires[1]));
        let coms = [
            commit_wire(&bases, &w0).into_affine(),
            commit_wire(&bases, &w1).into_affine(),
        ];
        // The honest prover algorithm on the inconsistent witness fails ...
        let pi = crs.prove(r, &[&w0, &w1]).into_affine();
        assert!(!crs.verify(&pi, &c, &coms));
        // ... and so does the proof for the committed payload.
        wires[1][5] -= Fr::one();
        let (v0, v1) = (big(&wires[0]), big(&wires[1]));
        let pi = crs.prove(r, &[&v0, &v1]).into_affine();
        assert!(!crs.verify(&pi, &c, &coms));
        // A change in a cell that is not linked keeps the link valid.
        wires[0][12] += Fr::one();
        let u0 = big(&wires[0]);
        let coms = [
            commit_wire(&bases, &u0).into_affine(),
            commit_wire(&bases, &v1).into_affine(),
        ];
        let pi = crs.prove(r, &[&u0, &v1]).into_affine();
        assert!(crs.verify(&pi, &c, &coms));
    }
}
