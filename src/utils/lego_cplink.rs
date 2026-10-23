//! LegoSNARK-style CP_link (Campanelli, Fiore, Querol, CCS'19) for halo2 advice-column
//! commitments under KZG.
//!
//! LegoSNARK's CP_link links a Pedersen commitment `c_w = r*h_0 + sum_i w_i*h_i` to the
//! "witness commitment" of a SNARK whose commitment is itself a Pedersen-like (linear)
//! commitment to the witness. For halo2 + KZG this is exactly the case: the commitment to an
//! advice column is `sum_j a_j * [L_j(tau)]_1` in the Lagrange basis, i.e. a Pedersen
//! vector commitment with generators `[L_j(tau)]_1` (KZG commitments in halo2 carry no extra
//! blinding term; hiding comes from the random blinding rows at the end of each column).
//!
//! CP_link is the Kiltz-Wee QA-NIZK for linear subspaces applied to the linear map
//!
//!   (c_w, c'_1, ..., c'_ell)^T = M * (r, a_1, ..., a_ell)^T
//!
//! where `a_i` are the full contents (all `n = 2^k` rows) of the `ell` advice columns that hold
//! the committed weights, `c'_i` their halo2/KZG commitments, and `M` is the `(ell+1) x t`
//! matrix (t = 1 + ell*n) with
//!   * row 0:   `h_0` under `r`, `h_{idx}` under the cell that holds weight `idx`, 0 elsewhere,
//!   * row i:   `[L_j(tau)]_1` under cell `j` of column `i`, 0 elsewhere.
//!
//! Setup: `k <- Z_p^{ell+1}`, `a <- Z_p`;  `[P]_1 = [M^T k]_1` (t G1 elements),
//!        `[C]_2 = [a*k]_2` (ell+1 G2 elements), `[a]_2`.
//! Prove: `pi = [x^T P]_1` (one MSM of size t).
//! Verify: `e(pi, [a]_2) == e(c_w, [C_0]_2) * prod_i e(c'_i, [C_i]_2)` (ell+2 pairings).
//! Soundness is the usual Kiltz-Wee argument (KerMDH in G2); zero-knowledge holds since
//! `pi = sum_i k_i * y_i` can be simulated from the statement with the trapdoor `k`.
//!
//! The scheme needs a relation-specific (but circuit-independent) trusted setup: `[P]_1` depends
//! on the KZG SRS, the number of rows/columns and the position of the weights, not on the model.
use core::fmt::Debug;
use std::time::Instant;

use ff::Field;
use group::{prime::PrimeCurveAffine, Curve, Group};
use halo2_proofs::{
  arithmetic::best_multiexp,
  helpers::SerdeCurveAffine,
  poly::{commitment::Params, kzg::commitment::ParamsKZG, LagrangeCoeff, Polynomial},
};
use halo2curves::pairing::{Engine, MillerLoopResult, MultiMillerLoop};
use rand_core::OsRng;
use rayon::prelude::*;

/// Common reference string of CP_link.
pub struct LegoCpLinkCrs<E: MultiMillerLoop> {
  /// Number of linked advice columns.
  pub ell: usize,
  /// Number of rows per column (2^k).
  pub n: usize,
  /// Rows of each column that hold weights (weight `idx` sits at column `idx / rows_per_col`,
  /// row `idx % rows_per_col`).
  pub rows_per_col: usize,
  /// Number of committed weights.
  pub num_weights: usize,
  /// Pedersen generators for the weight commitment: `h[0]` for the randomness, `h[1 + idx]` for
  /// weight `idx`.
  pub h: Vec<E::G1Affine>,
  /// `[P]_1 = [M^T k]_1`; `p[0]` belongs to `r`, `p[1 + i*n + j]` to cell `j` of column `i`.
  pub p: Vec<E::G1Affine>,
  /// `[C_i]_2 = [a k_i]_2` for `i = 0..=ell` (index 0 belongs to the Pedersen commitment).
  pub c: Vec<E::G2Prepared>,
  /// `[a]_2`.
  pub a: E::G2Prepared,
}

fn batch_to_affine<E: Engine>(proj: Vec<E::G1>) -> Vec<E::G1Affine> {
  let mut aff = vec![E::G1Affine::identity(); proj.len()];
  proj
    .par_chunks(1 << 14)
    .zip(aff.par_chunks_mut(1 << 14))
    .for_each(|(p, q)| E::G1::batch_normalize(p, q));
  aff
}

/// Generates the CP_link CRS for `ell` advice columns of `params.n()` rows, where the
/// `num_weights` weights occupy the first `rows_per_col` rows of each column (column-major).
pub fn lego_cplink_setup<E: MultiMillerLoop<G1Affine: SerdeCurveAffine, G2Affine: SerdeCurveAffine> + Debug>(
  params: &ParamsKZG<E>,
  ell: usize,
  rows_per_col: usize,
  num_weights: usize,
) -> LegoCpLinkCrs<E> {
  let n = params.n() as usize;
  assert!(ell >= 1, "CP_link needs at least one linked column");
  assert!(rows_per_col <= n, "rows_per_col ({rows_per_col}) exceeds the column size ({n})");
  assert!(rows_per_col * ell >= num_weights, "weights do not fit into {ell} x {rows_per_col} cells");
  let g_lagrange = params.g_lagrange();
  assert_eq!(g_lagrange.len(), n);

  let timer = Instant::now();
  // Pedersen commitment key for the weights (independent generators).
  let h_proj: Vec<E::G1> = (0..=num_weights).into_par_iter().map(|_| E::G1::random(OsRng)).collect();
  let h = batch_to_affine::<E>(h_proj);
  println!("CP_link setup: {} Pedersen generators in {:?}", h.len(), timer.elapsed());

  // QA-NIZK trapdoor.
  let k: Vec<E::Scalar> = (0..=ell).map(|_| E::Scalar::random(OsRng)).collect();
  let a = E::Scalar::random(OsRng);

  // [P]_1 = [M^T k]_1.
  let timer = Instant::now();
  let mut p_proj: Vec<E::G1> = Vec::with_capacity(1 + ell * n);
  p_proj.push(h[0] * k[0]);
  p_proj.par_extend((0..ell * n).into_par_iter().map(|t| {
    let i = t / n;
    let j = t % n;
    let mut pt = g_lagrange[j] * k[i + 1];
    if j < rows_per_col {
      let idx = i * rows_per_col + j;
      if idx < num_weights {
        pt += h[1 + idx] * k[0];
      }
    }
    pt
  }));
  let p = batch_to_affine::<E>(p_proj);
  println!("CP_link setup: [P]_1 ({} G1 elements) in {:?}", p.len(), timer.elapsed());

  let g2 = params.g2();
  let c = k.iter().map(|ki| E::G2Prepared::from((g2 * (a * ki)).to_affine())).collect();
  let a = E::G2Prepared::from((g2 * a).to_affine());

  LegoCpLinkCrs { ell, n, rows_per_col, num_weights, h, p, c, a }
}

/// Pedersen commitment to the weights: `r*h_0 + sum_idx w_idx * h_{1+idx}`.
pub fn lego_pedersen_commit<E: MultiMillerLoop>(crs: &LegoCpLinkCrs<E>, r: E::Scalar, weights: &[E::Scalar]) -> E::G1 {
  assert_eq!(weights.len(), crs.num_weights);
  let mut scalars = Vec::with_capacity(weights.len() + 1);
  scalars.push(r);
  scalars.extend_from_slice(weights);
  best_multiexp(&scalars, &crs.h[..scalars.len()])
}

/// CP_link proof: `pi = [x^T P]_1` with `x = (r, a_1, ..., a_ell)`.
pub fn lego_cplink_prove<E: MultiMillerLoop>(
  crs: &LegoCpLinkCrs<E>,
  r: E::Scalar,
  advice_cols: &[Polynomial<E::Scalar, LagrangeCoeff>],
) -> E::G1 {
  assert_eq!(advice_cols.len(), crs.ell, "expected {} linked advice columns", crs.ell);
  let mut pi = crs.p[0] * r;
  for (i, col) in advice_cols.iter().enumerate() {
    assert_eq!(col.len(), crs.n);
    let scalars: Vec<E::Scalar> = col.iter().copied().collect();
    pi += best_multiexp(&scalars, &crs.p[1 + i * crs.n..1 + (i + 1) * crs.n]);
  }
  pi
}

/// CP_link verification: `e(pi, [a]_2) == e(c_w, [C_0]_2) * prod_i e(c'_i, [C_i]_2)`.
pub fn lego_cplink_verify<E: MultiMillerLoop>(
  crs: &LegoCpLinkCrs<E>,
  pi: E::G1Affine,
  weight_com: E::G1Affine,
  advice_coms: &[E::G1Affine],
) -> bool {
  assert_eq!(advice_coms.len(), crs.ell);
  let neg_pi = -pi;
  let mut terms: Vec<(&E::G1Affine, &E::G2Prepared)> = Vec::with_capacity(crs.ell + 2);
  terms.push((&neg_pi, &crs.a));
  terms.push((&weight_com, &crs.c[0]));
  for (y, c) in advice_coms.iter().zip(crs.c[1..].iter()) {
    terms.push((y, c));
  }
  E::multi_miller_loop(&terms).final_exponentiation() == E::Gt::identity()
}
