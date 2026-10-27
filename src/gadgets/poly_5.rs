use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
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
    // poly_merged: the accumulator shares its region with the weights, which are assigned before the
    // constants, so the zero anchor `acc_R = 0` is enforced by a second selector instead of a copy
    // constraint to the zero constant.
    let anchor_selector = if gadget_config.poly_merged { Some(meta.selector()) } else { None };
    let columns = gadget_config.columns_poly.clone();
    let ell = columns.len() - 1;
    let acc = columns[ell];

    meta.create_gate("Artemis Horner gate", |meta| {
      let s = meta.query_selector(selector);
      let s_anchor = anchor_selector.map(|a| meta.query_selector(a));
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
      let mut constraints = vec![s * (out.clone() - res)];
      if let Some(s_anchor) = s_anchor {
        constraints.push(s_anchor * out);
      }
      constraints
    });

    let mut selectors = gadget_config.selectors;
    let mut poly_selectors = vec![selector];
    if let Some(a) = anchor_selector {
      poly_selectors.push(a);
    }
    selectors.insert(GadgetType::Poly5, poly_selectors);

    GadgetConfig { selectors, ..gadget_config }
  }

  /// Accumulator values `acc_0..=acc_R` (`acc_R = 0`) for the padded coefficient vector.
  fn accumulator_values(&self) -> Vec<Value<F>> {
    let ell = self.config.columns_poly.len() - 1;
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
    vals
  }

  /// `poly_merged`: assigns the accumulator (rows `0..=R`) inside the region that holds the weights
  /// row-major in the general advice columns, so the gate reads weights and accumulator at the same
  /// rows by construction. The anchor `acc_R = 0` is enforced by the anchor selector. Returns the
  /// cell at row 0 (`p(beta)`).
  pub fn assign_in_region(&self, region: &mut Region<'_, F>) -> Result<AssignedCell<F, F>, Error> {
    let cols = &self.config.columns_poly;
    let acc = cols[cols.len() - 1];
    let mut vals = self.accumulator_values();
    let rows = vals.len() - 1;
    // Test aid: POLY_TAMPER=1 perturbs acc_0, which must make the Horner gate (MockProver) and the
    // CP-link check (acc_0 == p(beta)) fail.
    if std::env::var("POLY_TAMPER").map(|v| v == "1").unwrap_or(false) {
      vals[0] = vals[0].map(|v| v + F::ONE);
    }
    let sels = self.config.selectors.get(&GadgetType::Poly5).unwrap();
    assert!(sels.len() == 2 && self.config.use_selectors, "poly_merged needs the Horner and anchor selectors");
    let mut rho = None;
    for r in 0..rows {
      sels[0].enable(region, r)?;
      let cell = region.assign_advice(|| "acc", acc, r, || vals[r])?;
      if r == 0 {
        rho = Some(cell);
      }
    }
    sels[1].enable(region, rows)?;
    region.assign_advice(|| "acc anchor", acc, rows, || Value::known(F::ZERO))?;
    Ok(rho.unwrap())
  }

  /// Assigns the accumulator column (rows `0..=R`, with the zero anchor at row `R`) and returns the
  /// cell at row 0, which holds `p(beta)`.
  pub fn assign(&self, mut layouter: impl Layouter<F>, zero: &AssignedCell<F, F>) -> Result<AssignedCell<F, F>, Error> {
    let cols = &self.config.columns_poly;
    let acc = cols[cols.len() - 1];
    let vals = self.accumulator_values();
    let rows = vals.len() - 1;

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
