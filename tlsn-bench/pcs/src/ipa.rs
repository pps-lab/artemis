//! Pedersen vector commitment over BN254 G1 with a zero-knowledge
//! inner-product evaluation argument (Bulletproofs folding, blinded `L`/`R`,
//! and a final Schnorr proof for the folded opening).
//!
//! Commitment: `C = <a, G> + r * H`. It is additively homomorphic, so a
//! commitment `c` to the payload and a commitment `c_mu` to the mask add up to
//! a commitment to the masked polynomial.
//!
//! Opening at `z` with value `v = sum_i a_i z^i`: `2 log2(n) + 1` group
//! elements and 2 scalars. The verifier does one MSM of size `n`.

use {
    crate::transcript::Transcript,
    ark_bn254::{Fq, Fr, G1Affine, G1Projective},
    ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM},
    ark_ff::{Field, One, PrimeField, UniformRand, Zero},
    ark_serialize::{CanonicalDeserialize, CanonicalSerialize},
    ark_std::rand::RngCore,
    rayon::prelude::*,
    sha2::{Digest, Sha256},
};

/// Public parameters: `n` independent generators, a blinding generator and a
/// generator for the inner-product value. All come from hash-to-curve.
#[derive(Clone, Debug)]
pub struct CommitKey {
    pub g: Vec<G1Affine>,
    pub h: G1Affine,
    pub u: G1Affine,
}

#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Proof {
    pub l:  Vec<G1Affine>,
    pub r:  Vec<G1Affine>,
    pub t:  G1Affine,
    pub z1: Fr,
    pub z2: Fr,
}

impl Proof {
    /// Serialized size with compressed points.
    pub fn size_bytes(&self) -> usize {
        // The two `Vec` length prefixes are not part of the proof: the
        // verifier knows log2(n).
        self.compressed_size() - 16
    }
}

/// Try-and-increment hash to BN254 G1 (cofactor 1).
fn hash_to_curve(domain: &[u8], index: u64) -> G1Affine {
    let mut ctr = 0u32;
    loop {
        let mut h = Sha256::new();
        h.update(b"tlsn-bench-pcs/ipa/v1");
        h.update(domain);
        h.update(index.to_le_bytes());
        h.update(ctr.to_le_bytes());
        let x = Fq::from_le_bytes_mod_order(&h.finalize());
        if let Some(p) = G1Affine::get_point_from_x_unchecked(x, false) {
            if !p.is_zero() {
                return p;
            }
        }
        ctr += 1;
    }
}

impl CommitKey {
    /// Deterministic transparent setup for vectors of up to `n` elements
    /// (`n` is rounded up to a power of two).
    pub fn setup(n: usize) -> Self {
        let n = n.max(2).next_power_of_two();
        let g = (0..n as u64)
            .into_par_iter()
            .map(|i| hash_to_curve(b"g", i))
            .collect();
        Self {
            g,
            h: hash_to_curve(b"h", 0),
            u: hash_to_curve(b"u", 0),
        }
    }

    pub fn n(&self) -> usize {
        self.g.len()
    }

    /// `<coeffs, G[offset..]> + blind * H`.
    pub fn commit_at(&self, offset: usize, coeffs: &[Fr], blind: Fr) -> G1Projective {
        assert!(offset + coeffs.len() <= self.n(), "vector exceeds the key");
        G1Projective::msm_unchecked(&self.g[offset..offset + coeffs.len()], coeffs) + self.h * blind
    }

    pub fn commit(&self, coeffs: &[Fr], blind: Fr) -> G1Projective {
        self.commit_at(0, coeffs, blind)
    }
}

fn inner(a: &[Fr], b: &[Fr]) -> Fr {
    a.par_iter().zip(b).map(|(x, y)| *x * y).sum()
}

fn start_transcript(n: usize, c: &G1Affine, z: Fr, v: Fr) -> (Transcript, Fr) {
    let mut t = Transcript::new(b"tlsn-bench-pcs/ipa-open/v1");
    t.absorb(b"n", &(n as u64));
    t.absorb(b"c", c);
    t.absorb(b"z", &z);
    t.absorb(b"v", &v);
    let xi = t.challenge(b"xi");
    (t, xi)
}

/// Prove that the polynomial committed in `commitment` (with blinding
/// `blind`) evaluates to `v` at `z`.
pub fn open<R: RngCore>(
    ck: &CommitKey,
    coeffs: &[Fr],
    blind: Fr,
    commitment: &G1Affine,
    z: Fr,
    v: Fr,
    rng: &mut R,
) -> Proof {
    let n = ck.n();
    assert!(coeffs.len() <= n);
    let mut a = coeffs.to_vec();
    a.resize(n, Fr::zero());
    let mut b = Vec::with_capacity(n);
    let mut pow = Fr::one();
    for _ in 0..n {
        b.push(pow);
        pow *= z;
    }
    let mut g = ck.g.clone();
    let mut blind = blind;

    let (mut t, xi) = start_transcript(n, commitment, z, v);
    let u = (ck.u * xi).into_affine();

    let rounds = n.trailing_zeros() as usize;
    let (mut ls, mut rs) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds));
    while a.len() > 1 {
        let half = a.len() / 2;
        let (a_lo, a_hi) = a.split_at(half);
        let (b_lo, b_hi) = b.split_at(half);
        let (g_lo, g_hi) = g.split_at(half);
        let (bl, br) = (Fr::rand(rng), Fr::rand(rng));
        let (l, r) = rayon::join(
            || G1Projective::msm_unchecked(g_hi, a_lo) + u * inner(a_lo, b_hi) + ck.h * bl,
            || G1Projective::msm_unchecked(g_lo, a_hi) + u * inner(a_hi, b_lo) + ck.h * br,
        );
        let lr = G1Projective::normalize_batch(&[l, r]);
        t.absorb(b"l", &lr[0]);
        t.absorb(b"r", &lr[1]);
        let x = t.challenge(b"x");
        let x_inv = x.inverse().expect("challenge is non-zero");

        let a_next: Vec<Fr> = a_lo
            .par_iter()
            .zip(a_hi)
            .map(|(lo, hi)| *lo + x_inv * hi)
            .collect();
        let b_next: Vec<Fr> = b_lo
            .par_iter()
            .zip(b_hi)
            .map(|(lo, hi)| *lo + x * hi)
            .collect();
        let g_next: Vec<G1Projective> = g_lo
            .par_iter()
            .zip(g_hi)
            .map(|(lo, hi)| *hi * x + lo)
            .collect();
        blind += x * bl + x_inv * br;
        a = a_next;
        b = b_next;
        g = G1Projective::normalize_batch(&g_next);
        ls.push(lr[0]);
        rs.push(lr[1]);
    }

    // Schnorr proof of knowledge of (a, blind) with
    // P_final = a * (G_f + b_f * U) + blind * H.
    let base = g[0] + u * b[0];
    let (d, s) = (Fr::rand(rng), Fr::rand(rng));
    let tt = (base * d + ck.h * s).into_affine();
    t.absorb(b"t", &tt);
    let c = t.challenge(b"c");
    Proof {
        l:  ls,
        r:  rs,
        t:  tt,
        z1: d + c * a[0],
        z2: s + c * blind,
    }
}

/// Check an opening proof. Returns `true` iff it verifies.
pub fn verify(ck: &CommitKey, commitment: &G1Affine, z: Fr, v: Fr, proof: &Proof) -> bool {
    let n = ck.n();
    let rounds = n.trailing_zeros() as usize;
    if proof.l.len() != rounds || proof.r.len() != rounds {
        return false;
    }
    let (mut t, xi) = start_transcript(n, commitment, z, v);

    let mut xs = Vec::with_capacity(rounds);
    for (l, r) in proof.l.iter().zip(&proof.r) {
        t.absorb(b"l", l);
        t.absorb(b"r", r);
        let x = t.challenge(b"x");
        if x.is_zero() {
            return false;
        }
        xs.push(x);
    }
    t.absorb(b"t", &proof.t);
    let c = t.challenge(b"c");

    // s_i = prod over rounds j of x_j^{bit}, where round 0 splits on the top
    // bit of i.
    let mut s = vec![Fr::one()];
    for x in &xs {
        let mut next = Vec::with_capacity(s.len() * 2);
        next.extend(s.iter().copied());
        next.extend(s.iter().map(|e| *e * x));
        // Round j splits the current vector in halves; the first challenge
        // must end up on the top bit, so later challenges go to lower bits.
        s = next;
    }
    // `s` built this way puts x_0 on the lowest bit; reorder by bit reversal.
    let s: Vec<Fr> = (0..n)
        .into_par_iter()
        .map(|i| s[(i.reverse_bits() >> (usize::BITS as usize - rounds)) as usize])
        .collect();

    // b_f = prod_j (1 + x_j * z^(n / 2^(j+1))).
    let mut z_pows = Vec::with_capacity(rounds);
    let mut zp = z;
    for _ in 0..rounds {
        z_pows.push(zp);
        zp = zp.square();
    }
    let b_f: Fr = xs
        .iter()
        .enumerate()
        .map(|(j, x)| Fr::one() + *x * z_pows[rounds - 1 - j])
        .product();

    // z1 * (G_f + b_f * xi * U) + z2 * H
    //   == T + c * (C + v * xi * U + sum_j x_j L_j + x_j^-1 R_j)
    let mut x_invs = xs.clone();
    ark_ff::batch_inversion(&mut x_invs);
    let scalars_g: Vec<Fr> = s.par_iter().map(|e| *e * proof.z1).collect();
    let lhs_g = G1Projective::msm_unchecked(&ck.g, &scalars_g);

    let mut bases = vec![ck.u, ck.h, proof.t, *commitment];
    let mut scalars = vec![xi * (proof.z1 * b_f - c * v), proof.z2, -Fr::one(), -c];
    for ((l, r), (x, x_inv)) in proof.l.iter().zip(&proof.r).zip(xs.iter().zip(&x_invs)) {
        bases.push(*l);
        scalars.push(-c * x);
        bases.push(*r);
        scalars.push(-c * x_inv);
    }
    (lhs_g + G1Projective::msm_unchecked(&bases, &scalars)).is_zero()
}

#[cfg(test)]
mod tests {
    use {super::*, crate::eval_poly, ark_std::test_rng};

    fn setup(len: usize) -> (CommitKey, Vec<Fr>, Fr, G1Affine) {
        let mut rng = test_rng();
        let ck = CommitKey::setup(len);
        let coeffs: Vec<Fr> = (0..len).map(|_| Fr::rand(&mut rng)).collect();
        let blind = Fr::rand(&mut rng);
        let c = ck.commit(&coeffs, blind).into_affine();
        (ck, coeffs, blind, c)
    }

    #[test]
    fn roundtrip() {
        for len in [1, 2, 3, 8, 37, 64] {
            let mut rng = test_rng();
            let (ck, coeffs, blind, c) = setup(len);
            let z = Fr::rand(&mut rng);
            let v = eval_poly(&coeffs, z);
            let proof = open(&ck, &coeffs, blind, &c, z, v, &mut rng);
            assert!(verify(&ck, &c, z, v, &proof), "len {len}");
            assert_eq!(proof.size_bytes(), (2 * proof.l.len() + 1) * 32 + 64);
        }
    }

    #[test]
    fn rejects_wrong_value_point_commitment() {
        let mut rng = test_rng();
        let (ck, coeffs, blind, c) = setup(37);
        let z = Fr::rand(&mut rng);
        let v = eval_poly(&coeffs, z);
        let proof = open(&ck, &coeffs, blind, &c, z, v, &mut rng);
        assert!(!verify(&ck, &c, z, v + Fr::one(), &proof));
        assert!(!verify(&ck, &c, z + Fr::one(), v, &proof));
        let other = (c + ck.g[0]).into_affine();
        assert!(!verify(&ck, &other, z, v, &proof));
        // A prover that lies about the value cannot produce a valid proof by
        // running the honest algorithm.
        let bad = open(&ck, &coeffs, blind, &c, z, v + Fr::one(), &mut rng);
        assert!(!verify(&ck, &c, z, v + Fr::one(), &bad));
        let mut tampered = proof.clone();
        tampered.z1 += Fr::one();
        assert!(!verify(&ck, &c, z, v, &tampered));
    }

    #[test]
    fn homomorphic_mask() {
        // c commits to the payload, c_mu to the mask at position n.
        let mut rng = test_rng();
        let ck = CommitKey::setup(17);
        let payload: Vec<Fr> = (0..16).map(|_| Fr::rand(&mut rng)).collect();
        let (mu, r, r_mu) = (Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng));
        let c = ck.commit(&payload, r);
        let c_mu = ck.commit_at(16, &[mu], r_mu);
        let mut masked = payload.clone();
        masked.push(mu);
        let sum = (c + c_mu).into_affine();
        let z = Fr::rand(&mut rng);
        let v = eval_poly(&masked, z);
        let proof = open(&ck, &masked, r + r_mu, &sum, z, v, &mut rng);
        assert!(verify(&ck, &sum, z, v, &proof));
    }
}
