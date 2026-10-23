use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error, Expression},
  poly::Rotation,
};

use super::gadget::{GadgetConfig, GadgetType};

type PolyConfig = GadgetConfig;

/// Artemis (inner product) in-circuit evaluation of the weight polynomial at the challenge `beta`:
/// the powers of beta are materialised in `ell` advice columns and the evaluation is the inner
/// product of the weight cells with them.
///
/// Columns (`columns_poly`): `[0, ell)` weights (row-major, weight `i` at column `i % ell`, row
/// `i / ell`), `[ell, 2 ell)` powers of beta (`B_{r,j} = beta^{r*ell + j}`), `2 ell` accumulator.
/// Gates (selector `s_row`, rows `0..R`):
///   acc_r = sum_j w_{r,j} * B_{r,j} + acc_{r+1}          (acc_R = 0, copy-constrained to zero)
///   B_{r,j} = B_{r,j-1} * beta        for j >= 1
/// and (selector `s_chain`, rows `1..R`):
///   B_{r,0} = B_{r-1,ell-1} * beta,
/// with `B_{0,0}` copy-constrained to the one constant. Hence `acc_0 = p(beta)`.
pub struct Poly2Chip<F: PrimeField> {
  config: Rc<PolyConfig>,
  beta: Value<F>,
  coeffs: Vec<F>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> Poly2Chip<F> {
  /// `coeffs` are the weights in natural order, padded with zeros to a multiple of `poly_ell`.
  pub fn construct(config: Rc<PolyConfig>, beta: Value<F>, coeffs: Vec<F>) -> Self {
    Self { config, beta, coeffs, _marker: PhantomData }
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let s_row = meta.selector();
    let s_chain = meta.selector();
    let columns = gadget_config.columns_poly.clone();
    let ell = (columns.len() - 1) / 2;
    let coeff_cols = columns[0..ell].to_vec();
    let beta_cols = columns[ell..2 * ell].to_vec();
    let acc = columns[2 * ell];

    meta.create_gate("Artemis inner-product gate", |meta| {
      let s = meta.query_selector(s_row);
      let beta: Expression<F> = gadget_config.beta.expr();
      let out = meta.query_advice(acc, Rotation::cur());
      let next = meta.query_advice(acc, Rotation::next());
      let mut res = next;
      for j in 0..ell {
        res = res + meta.query_advice(coeff_cols[j], Rotation::cur()) * meta.query_advice(beta_cols[j], Rotation::cur());
      }
      let mut constraints = vec![s.clone() * (out - res)];
      for j in 1..ell {
        let cur = meta.query_advice(beta_cols[j], Rotation::cur());
        let prev = meta.query_advice(beta_cols[j - 1], Rotation::cur());
        constraints.push(s.clone() * (cur - prev * beta.clone()));
      }
      constraints
    });

    meta.create_gate("Artemis beta chain", |meta| {
      let s = meta.query_selector(s_chain);
      let beta: Expression<F> = gadget_config.beta.expr();
      let cur = meta.query_advice(beta_cols[0], Rotation::cur());
      let prev = meta.query_advice(beta_cols[ell - 1], Rotation::prev());
      vec![s * (cur - prev * beta)]
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Poly2, vec![s_row, s_chain]);

    GadgetConfig { selectors, ..gadget_config }
  }

  /// Assigns the beta-power columns and the accumulator column and returns the cell at row 0 of the
  /// accumulator, which holds `p(beta)`.
  pub fn assign(
    &self,
    mut layouter: impl Layouter<F>,
    zero: &AssignedCell<F, F>,
    one: &AssignedCell<F, F>,
  ) -> Result<AssignedCell<F, F>, Error> {
    let cols = &self.config.columns_poly;
    let ell = (cols.len() - 1) / 2;
    let beta_cols = cols[ell..2 * ell].to_vec();
    let acc = cols[2 * ell];
    assert!(ell >= 1 && self.coeffs.len() % ell == 0, "coefficients must be padded to a multiple of poly_ell");
    let rows = self.coeffs.len() / ell;

    let mut pows = Vec::with_capacity(rows * ell);
    let mut p = Value::known(F::ONE);
    for _ in 0..rows * ell {
      pows.push(p);
      p = p * self.beta;
    }
    let mut vals = vec![Value::known(F::ZERO); rows + 1];
    for r in (0..rows).rev() {
      let mut e = vals[r + 1];
      for j in 0..ell {
        e = e + pows[r * ell + j] * Value::known(self.coeffs[r * ell + j]);
      }
      vals[r] = e;
    }

    let selectors = if self.config.use_selectors {
      let s = self.config.selectors.get(&GadgetType::Poly2).unwrap();
      Some((s[0], s[1]))
    } else {
      None
    };

    layouter.assign_region(
      || "Artemis inner product",
      |mut region| {
        let mut rho = None;
        for r in 0..rows {
          if let Some((s_row, s_chain)) = selectors {
            s_row.enable(&mut region, r)?;
            if r > 0 {
              s_chain.enable(&mut region, r)?;
            }
          }
          for j in 0..ell {
            let cell = region.assign_advice(|| "beta pow", beta_cols[j], r, || pows[r * ell + j])?;
            if r == 0 && j == 0 {
              region.constrain_equal(cell.cell(), one.cell())?;
            }
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
