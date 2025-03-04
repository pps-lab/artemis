use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{Advice, Assigned, Column, ConstraintSystem, Error, Expression, Instance},
  poly::Rotation,
};
use rmp_serde::config;
use serde_json::map::VacantEntry;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type PolyConfig = GadgetConfig;

pub struct Poly4Chip<F: PrimeField> {
  config: Rc<PolyConfig>,
  _marker: PhantomData<F>,
  coeffs: Vec<F>,
  betas: Vec<Value<F>>,
  alphas: Vec<F>,
}

impl<F: PrimeField> Poly4Chip<F> {
  pub fn construct(config: Rc<PolyConfig>, betas: Vec<Value<F>>, alphas: Vec<F>, coeffs: Vec<F>) -> Self {
    Self {
      config,
      _marker: PhantomData,
      coeffs,
      betas,
      alphas,
    }
  }

  // pub fn get_beta_columns(columns: &Vec<Column<Advice>>) -> Column<Advice> {
  //   //let num_inputs = (columns.len() - 2) / 2;
  //   columns[columns.len() - 3]
  // }

  pub fn get_coeff_columns(columns: &Vec<Column<Advice>>) -> Vec<Column<Advice>> {
    let num_inputs = columns.len() - 2;
    columns[0..num_inputs].to_vec()
  }

  // pub fn get_alpha_columns(columns: &Vec<Column<Advice>>) -> Vec<Column<Advice>> {
  //   let num_inputs = (columns.len() - 3) / 2;
  //   columns[num_inputs..num_inputs * 2].to_vec()
  // }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.selector();
    let columns = &gadget_config.columns_poly;

    meta.create_gate("Poly gate", |meta| {
      let s = meta.query_selector(selector);

      let beta: Expression<F> = gadget_config.beta.expr();
      let one = Expression::Constant(F::ONE);
      let mut betas = vec![];
      let mut curr = one;
      for column in 0..gadget_config.columns_poly.len() - 2 {
        betas.push(curr.clone());
        curr = curr * beta.clone();
      }
      betas = betas.into_iter().rev().collect();
      let last_beta = betas.last().unwrap().clone() * beta.clone();
      println!("Betas: {:?}", (betas.clone(), betas.len()));

      let gate_coeffs =  Poly4Chip::<F>::get_coeff_columns(columns)
        .iter()
        .map(|col| meta.query_advice(*col, Rotation::cur()))
        .collect::<Vec<_>>();

      // let gate_alphas =  Poly3Chip::<F>::get_alpha_columns(columns)
      //   .iter()
      //   .map(|col| meta.query_advice(*col, Rotation::cur()))
      //   .collect::<Vec<_>>();
        // .iter()
        // .map(|col| meta.query_advice(*col, Rotation::cur()))
        // .collect::<Vec<_>>();

      let bias = meta.query_advice(columns[columns.len() - 2], Rotation::cur());
      let gate_output = meta.query_advice(columns[columns.len() - 1], Rotation::cur());
      let res = gate_coeffs
        .iter()
        .zip(betas)
        .map(|(a, b)| a.clone() * b.clone())
        .fold(Expression::Constant(F::ZERO), |a, b| a + b.clone());
      //let res = res + bias;
      // let res = gate_inp
      //   .iter()
      //   .zip(gate_coeffs)
      //   .map(|(a, b)| a.clone() * b.clone())
      //   .fold(Expression::Constant(F::ZERO), |a, b| a + b);
      let res = res + bias * last_beta;

      vec![s * (res.clone() - res)]
    });
    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::DotProductBias, vec![selector]);
    //println!("Selectors: {:?}", selectors);

    GadgetConfig {
      columns: gadget_config.columns,
      selectors,
      ..gadget_config
    }
  }
}

impl<F: PrimeField> Gadget<F> for Poly4Chip<F> {
  fn name(&self) -> String {
    "Poly".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    self.config.columns.len()
  }

  fn num_inputs_per_row(&self) -> usize {
    self.betas.len() - 1
  }

  fn num_outputs_per_row(&self) -> usize {
    1
  }

  // The caller is expected to pad the inputs
  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let cols = &self.config.columns_poly;
    let coeffs_vec = &vec_inputs[0];
    let mut coeffs = vec![];
    for j in 0..self.num_inputs_per_row() {
      coeffs.push(self.coeffs[row_offset * self.num_inputs_per_row() + j]);
    }

    assert_eq!(coeffs.len(), self.num_inputs_per_row());

    //let zero = &single_inputs[0];
    let bias = &single_inputs[0];

    if self.config.use_selectors {
      let selector = self
        .config
        .selectors
        .get(&GadgetType::DotProductBias)
        .unwrap()[0];
      selector.enable(region, row_offset).unwrap();
    }

    let coeff_cols = Poly4Chip::<F>::get_coeff_columns(&cols);

    let last_beta = self.betas[0];

    bias.copy_advice(
      || "",
      region,
      cols[cols.len() - 2],
      row_offset,
    )?;

    let coeff_cells: Vec<_> = (0..coeffs_vec.len())
      .map(|i | {
        region.assign_advice(|| "" , coeff_cols[i], row_offset, || Value::known(coeffs[i])).unwrap()
      })
      .collect();


    let mut e = coeffs.clone().into_iter().zip(self.betas[1..self.betas.len()].to_vec()) 
      // .zip(alphas.iter())
      .map(| (a , b)| Value::known(a) * b)
      .reduce(|a, b| a + b)
      .unwrap();

    e = e + last_beta * bias.value().map(|x| *x); //beta_vec.iter()
      // .zip(coeffs.iter())
      // .map(|(a, b)|  Value::known(*a) * b.value())
      // .reduce(|a, b| a + b)
      // .unwrap();
    //let e = e + bias.value().map(|x: &F| *x);
    //println!("res {:?}", e);

    let res = region
      .assign_advice(
        || "",
        cols[cols.len() - 1],
        row_offset,
        || e,
      )
      .unwrap();
    //println!("beta vec val: {:?}", beta_vec);
    Ok(vec![res])
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert!(single_inputs.len() <= 2);
    let cols = &self.config.columns_poly;
    let zero = layouter.assign_region(|| " ", |mut region| {
      region.assign_advice(|| "", cols[cols.len() - 2], 0, || Value::known(F::ZERO))
    }).unwrap();
    let bias = zero;
    let coeffs = vec_inputs[0].clone();

    let output = layouter
      .assign_region(
        || "Poly rows",
        |mut region| {
          println!("Called once");
          let mut cur_bias = bias.clone();
          println!("Last idx: {:?}", coeffs.len() / self.num_inputs_per_row());
          for i in 0..coeffs.len() / self.num_inputs_per_row() {
            let mut weight_vec = vec![];
            for j in 0..self.num_inputs_per_row() {
              weight_vec.push(coeffs[i * self.num_inputs_per_row() + j]);
            }
            cur_bias = self
              .op_row_region(&mut region, i, &vec![weight_vec], &vec![&cur_bias])
              .unwrap()[0]
              .clone();
          }
          Ok(cur_bias)
        },
      )
      .unwrap();

    Ok(vec![output])
  }
}
