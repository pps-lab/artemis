use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::{ff::PrimeField, impl_add_binop_specify_output},
  plonk::{Advice, Column, ConstraintSystem, Error, Expression},
  poly::Rotation,
};
use rmp_serde::config;

use crate::gadgets::{adder::AdderChip, dot_prod::DotProductChip};

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type PolyConfig = GadgetConfig;

pub struct PolyChip<F: PrimeField> {
  config: Rc<PolyConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> PolyChip<F> {
  pub fn construct(config: Rc<PolyConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn get_a_columns(config: &GadgetConfig) -> Vec<Column<Advice>> {
    //println!("Poly len: {:?}", config.columns_poly.len());
    config.columns_poly[..(config.columns_poly.len() - 1)/2].into()
  }

  pub fn get_b_columns(config: &GadgetConfig) -> Vec<Column<Advice>> {
    config.columns_poly[(config.columns_poly.len() - 1)/2..].into()
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.selector();
    let columns = &gadget_config.columns;

    meta.create_gate("poly gate", |meta| {
      let s = meta.query_selector(selector);
      let gate_a = meta.query_advice(gadget_config.columns_poly[0], Rotation::cur());
      let gate_b = meta.query_advice(gadget_config.columns_poly[1], Rotation::cur());
      let gate_b_prev = meta.query_advice(gadget_config.columns_poly[1], Rotation::prev());
      let beta = meta.query_instance(gadget_config.columns_public[0], Rotation(0));
      vec![s * (gate_b - (gate_a + gate_b_prev * beta))]
    });

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Poly, vec![selector]);

    GadgetConfig {
      columns: gadget_config.columns,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for PolyChip<F> {
  fn name(&self) -> String {
    "dot product".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    self.config.columns.len()
  }

  fn num_inputs_per_row(&self) -> usize {
    (self.config.columns.len() - 1) / 2
  }

  fn num_outputs_per_row(&self) -> usize {
    1
  }
  // f1(X)i0(X) - f2(X) = 0
  // The caller is expected to pad the inputs
  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert_eq!(vec_inputs.len(), 2);

    let inp = &vec_inputs[0];
    //println!("Weights len: {}", inp.len());
    let weights = &vec_inputs[1];
    assert_eq!(inp.len(), weights.len());
    assert_eq!(inp.len(), self.num_inputs_per_row());

    let zero = &single_inputs[0];

    if self.config.use_selectors {
      let selector = self.config.selectors.get(&GadgetType::DotProduct).unwrap()[0];
      selector.enable(region, row_offset).unwrap();
    }

    let inp_cols = DotProductChip::<F>::get_input_columns(&self.config);
    inp
      .iter()
      .enumerate()
      .map(|(i, cell)| cell.copy_advice(|| "", region, inp_cols[i], row_offset))
      .collect::<Result<Vec<_>, _>>()
      .unwrap();

    let weight_cols = DotProductChip::<F>::get_weight_columns(&self.config);
    weights
      .iter()
      .enumerate()
      .map(|(i, cell)| cell.copy_advice(|| "", region, weight_cols[i], row_offset))
      .collect::<Result<Vec<_>, _>>()
      .unwrap();

    // All columns need to be assigned
    if self.config.columns.len() % 2 == 0 {
      zero
        .copy_advice(
          || "",
          region,
          self.config.columns[self.config.columns.len() - 2],
          row_offset,
        )
        .unwrap();
    }

    let e = inp
      .iter()
      .zip(weights.iter())
      .map(|(a, b)| a.value().map(|x: &F| *x) * b.value())
      .reduce(|a, b| a + b)
      .unwrap();

    let res = region
      .assign_advice(
        || "",
        self.config.columns_poly[self.config.columns_poly.len() - 1],
        row_offset,
        || e,
      )
      .unwrap();

    Ok(vec![res])
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert_eq!(vec_inputs.len(), 2);
    assert_eq!(single_inputs.len(), 1);
    let zero = &single_inputs[0];

    let mut a_i = vec_inputs[0].clone();
    let mut b_i = vec_inputs[1].clone();
    let res = layouter
    .assign_region(
      || "dot prod rows",
      |mut region| {
        let a_cols = a_i
        .iter()
        .enumerate()
        .map(|(i, cell)| cell.copy_advice(|| "", &mut region, PolyChip::<F>::get_a_columns(&self.config)[0], i))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
        let a_cols = PolyChip::<F>::get_a_columns(&self.config);
        let b_cols = b_i
        .iter()
        .enumerate()
        .map(|(i, cell)| cell.copy_advice(|| "", &mut region, PolyChip::<F>::get_b_columns(&self.config)[0], i))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
        //let b_cols = PolyChip::<F>::get_a_columns(&self.config);
        Ok(vec![b_cols[0].clone()])
      }
    );
    Ok(res.unwrap())
  }
}
