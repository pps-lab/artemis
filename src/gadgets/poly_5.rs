use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use super::gadget::{GadgetConfig, GadgetType};

type PolyConfig = GadgetConfig;

/// Artemis (Horner) in-circuit evaluation of the weight polynomial `p(X) = sum_i w_i X^i` at the
/// Fiat-Shamir challenge `beta`.
///
/// Layout: the `ell` weight columns hold the weights row-major (weight `i` sits in column `i % ell`,
/// row `i / ell`), padded with zeros to a full last row. The accumulator column `acc` holds at row
/// `r` the evaluation of the tail of the polynomial:
///
///   acc_r = sum_j w_{r*ell + j} * beta^j + beta^ell * acc_{r+1},   acc_R = 0,
///
/// so `acc_0 = p(beta)`. The gate reads the weight cells in place (no copy constraints) and the
/// accumulator at `Rotation::next`; the anchor `acc_R = 0` is copy-constrained to the zero constant.
/// The region is assigned in every synthesize pass (keygen included) so that the selector exists.
pub struct Poly5Chip<F: PrimeField> {
  config: Rc<PolyConfig>,
  beta: Value<F>,
  coeffs: Vec<F>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> Poly5Chip<F> {
  /// `coeffs` are the weights in natural order, padded with zeros to a multiple of `poly_ell`.
  pub fn construct(config: Rc<PolyConfig>, beta: Value<F>, coeffs: Vec<F>) -> Self {
    Self { config, beta, coeffs, _marker: PhantomData }
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.selector();
    let columns = gadget_config.columns_poly.clone();
    let ell = columns.len() - 1;
    let acc = columns[ell];

    meta.create_gate("Artemis Horner gate", |meta| {
      let s = meta.query_selector(selector);
      let beta: Expression<F> = gadget_config.beta.expr();
      // 1, beta, ..., beta^ell
      let mut beta_pows = vec![Expression::Constant(F::ONE)];
      for _ in 0..ell {
        let last = beta_pows.last().unwrap().clone();
        beta_pows.push(last * beta.clone());
      }
      let out = meta.query_advice(acc, Rotation::cur());
      let next = meta.query_advice(acc, Rotation::next());
      let mut res = next * beta_pows[ell].clone();
      for j in 0..ell {
        res = res + meta.query_advice(columns[j], Rotation::cur()) * beta_pows[j].clone();
      }
      vec![s * (out - res)]
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Poly5, vec![selector]);

    GadgetConfig { selectors, ..gadget_config }
  }

  /// Assigns the accumulator column (rows `0..=R`, with the zero anchor at row `R`) and returns the
  /// cell at row 0, which holds `p(beta)`.
  pub fn assign(&self, mut layouter: impl Layouter<F>, zero: &AssignedCell<F, F>) -> Result<AssignedCell<F, F>, Error> {
    let cols = &self.config.columns_poly;
    let ell = cols.len() - 1;
    let acc = cols[ell];
    assert!(ell >= 1 && self.coeffs.len() % ell == 0, "coefficients must be padded to a multiple of poly_ell");
    let rows = self.coeffs.len() / ell;

    let beta_pows: Vec<Value<F>> = (0..=ell).map(|i| self.beta.map(|b| b.pow([i as u64]))).collect();
    let mut vals = vec![Value::known(F::ZERO); rows + 1];
    for r in (0..rows).rev() {
      let mut e = vals[r + 1] * beta_pows[ell];
      for j in 0..ell {
        e = e + beta_pows[j] * Value::known(self.coeffs[r * ell + j]);
      }
      vals[r] = e;
    }

    let selector = if self.config.use_selectors {
      Some(self.config.selectors.get(&GadgetType::Poly5).unwrap()[0])
    } else {
      None
    };

    layouter.assign_region(
      || "Artemis Horner",
      |mut region| {
        let mut rho = None;
        for r in 0..rows {
          if let Some(s) = selector {
            s.enable(&mut region, r)?;
          }
          let cell = region.assign_advice(|| "acc", acc, r, || vals[r])?;
          if r == 0 {
            rho = Some(cell);
          }
        }
        let anchor = region.assign_advice(|| "acc anchor", acc, rows, || Value::known(F::ZERO))?;
        region.constrain_equal(anchor.cell(), zero.cell())?;
        Ok(rho.unwrap())
      },
    )
  }
}
