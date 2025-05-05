use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region, Value},
  halo2curves::ff::PrimeField,
  plonk::{Advice, Column, ConstraintSystem, Error, Expression, Instance},
  poly::Rotation,
};
use serde::de::value;
use serde_json::map::VacantEntry;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

type PolyConfig = GadgetConfig;

pub struct Poly2Chip<F: PrimeField> {
  config: Rc<PolyConfig>,
  _marker: PhantomData<F>,
  betas: Vec<Value<F>>,
  coeffs: Vec<F>,
}

impl<F: PrimeField> Poly2Chip<F> {
  pub fn construct(config: Rc<PolyConfig>, betas: Vec<Value<F>>, coeffs: Vec<F>) -> Self {
    Self {
      config,
      _marker: PhantomData,
      betas,
      coeffs
    }
  }

  pub fn get_input_columns(columns: &Vec<Column<Advice>>) -> Vec<Column<Advice>> {
    let num_inputs = (columns.len() - 2) / 2;
    columns[0..num_inputs].to_vec()
  }

  pub fn get_coeff_columns(columns: &Vec<Column<Advice>>) -> Vec<Column<Advice>> {
    let num_inputs = (columns.len() - 2) / 2;
    columns[num_inputs..num_inputs * 2].to_vec()
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let selector = meta.selector();
    let columns = &gadget_config.columns_poly;

    meta.create_gate("Poly gate", |meta| {
      let s = meta.query_selector(selector);
      // let gate_inp = Poly2Chip::<F>::get_input_columns(columns)
      //   .iter()
      //   .map(|col| meta.query_advice(*col, Rotation::cur()))
      //   .collect::<Vec<_>>();
      let gate_inp = Poly2Chip::<F>::get_input_columns(columns)
        .iter()
        .map(|col| meta.query_advice(*col, Rotation::cur()))
        .collect::<Vec<_>>();

      let gate_coeffs = Poly2Chip::<F>::get_coeff_columns(columns)
        .iter()
        .map(|col| meta.query_advice(*col, Rotation::cur()))
        .collect::<Vec<_>>();

      let bias = meta.query_advice(columns[columns.len() - 2], Rotation::cur());
      let gate_output = meta.query_advice(columns[columns.len() - 1], Rotation::cur());

      let res = gate_inp
        .iter()
        .zip(gate_coeffs)
        .map(|(a, b)| a.clone() * b.clone())
        .reduce(|a, b| a + b).unwrap();
      let res = res + bias;

      vec![s * (res.clone() - res)]
    });

    let beta_cols = Poly2Chip::<F>::get_input_columns(columns);
    meta.create_gate("first beta gate", |meta| {
      let s = meta.query_selector(selector);
      // let gate_inp = Poly2Chip::<F>::get_input_columns(columns)
      //   .iter()
      //   .map(|col| meta.query_advice(*col, Rotation::cur()))
      //   .collect::<Vec<_>>();
      let beta: Expression<F> = gadget_config.beta.expr();

      let prev_beta = meta.query_advice(beta_cols[beta_cols.len() - 1], Rotation::prev());
      let curr_beta = meta.query_advice(beta_cols[0], Rotation::cur());

      vec![s * (curr_beta - prev_beta * beta)]
    });

    for idx in 1..beta_cols.len() {
      meta.create_gate("beta gate", |meta| {
        let s = meta.query_selector(selector);
        // let gate_inp = Poly2Chip::<F>::get_input_columns(columns)
        //   .iter()
        //   .map(|col| meta.query_advice(*col, Rotation::cur()))
        //   .collect::<Vec<_>>();
        let beta: Expression<F> = gadget_config.beta.expr();
  
        let prev_beta = meta.query_advice(beta_cols[idx - 1], Rotation::cur());
        let curr_beta = meta.query_advice(beta_cols[idx], Rotation::cur());
  
        vec![s * (curr_beta - prev_beta * beta)]
      });
    }

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

impl<F: PrimeField> Gadget<F> for Poly2Chip<F> {
  fn name(&self) -> String {
    "Poly".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    self.config.columns.len()
  }

  fn num_inputs_per_row(&self) -> usize {
    (self.config.columns_poly.len() - 2) / 2
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
    //assert_eq!(vec_inputs.len(), 2);

    //let inp = &vec_inputs[0];
    let coeffs = &vec_inputs[0];

    let mut coeffs = vec![];
    for j in 0..self.num_inputs_per_row() {
      //coeffs.push(self.coeffs[row_offset + (coeffs.len() / self.num_inputs_per_row()) * j]);
      coeffs.push(self.coeffs[row_offset * self.num_inputs_per_row() + j]);
    }

    //assert_eq!(inp.len(), coeffs.len());
    assert_eq!(coeffs.len(), self.num_inputs_per_row());

    let bias = single_inputs[1];

    if self.config.use_selectors {
      let selector = self
        .config
        .selectors
        .get(&GadgetType::DotProductBias)
        .unwrap()[0];
      selector.enable(region, row_offset).unwrap();
    }

    let mut beta_vec = vec![];
    for j in 0..self.num_inputs_per_row() {
      //beta_vec.push(self.betas[row_offset + (coeffs.len() / self.num_inputs_per_row()) * j]);
      beta_vec.push(self.betas[row_offset * self.num_inputs_per_row() + j]);
    }

    //let inp_cols =   //Poly2Chip::<F>::get_input_columns(&cols);
    // inp
    //   .iter()
    //   .enumerate()
    //   .map(|(i, cell)| cell.copy_advice(|| "", region, inp_cols[i], row_offset))
    //   .collect::<Result<Vec<_>, _>>()
    //   .unwrap();

    //let coeff_cols = Poly2Chip::<F>::get_coeff_columns(&cols);
    // coeffs
    //   .iter()
    //   .enumerate()
    //   .map(|(i, cell)| cell.copy_advice(|| "", region, coeff_cols[i], row_offset))
    //   .collect::<Result<Vec<_>, _>>()
    //   .unwrap();

    bias.copy_advice(
      || "",
      region,
      cols[cols.len() - 2],
      row_offset,
    )?;
    let cols_inp = Poly2Chip::<F>::get_input_columns(cols);

    let _betas: Vec<_> = beta_vec.iter().enumerate().map(|(idx, x)| {
      region
      .assign_advice(
        || "",
        cols_inp[idx],
        row_offset,
        || *x,
      )
    }).collect();

    let e = beta_vec.iter()
      .zip(coeffs.iter())
      .map(|(a, b)|  *a * Value::known(b))
      .reduce(|a, b| a + b)
      .unwrap();
    let e = e + bias.value().map(|x: &F| *x);

    if row_offset == 0 || row_offset == 1 {
      println!("coeffs: {:?}, {:?}, e: {:?}, bias: {:?}, beta_vec: {:?}", coeffs, vec_inputs[0], e, bias, beta_vec);
    }

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
    //assert_eq!(vec_inputs.len(), 2);
    assert!(single_inputs.len() <= 2);
    //let zero = single_inputs[0];
    //let mut inputs = vec_inputs;
    let cols = &self.config.columns_poly;
    let zero = layouter.assign_region(|| " ", |mut region| {
      region.assign_advice(|| "", cols[cols.len() - 2], 0, || Value::known(F::ZERO))
    }).unwrap();
    let bias = &zero;
    let output = layouter
      .assign_region(
        || "Poly rows",
        |mut region| {
          let mut cur_bias = bias.clone();
          let mut coeffs = vec_inputs[0].clone();
  
          while coeffs.len() % self.num_inputs_per_row() != 0 {
            //inputs.push(&zero);
            coeffs.push(&zero);
          }
          for i in 0..coeffs.len() / self.num_inputs_per_row() {
            let mut weights = vec![];
            for j in 0..self.num_inputs_per_row() {
              //weights.push(coeffs[(i + coeffs.len()) / self.num_inputs_per_row() * j]);
              weights.push(coeffs[i * self.num_inputs_per_row() + j]);
            }
            //coeffs[i * self.num_inputs_per_row()..(i + 1) * self.num_inputs_per_row()].to_vec();
            
            cur_bias = self
              .op_row_region(&mut region, i, &vec![weights], &vec![&zero, &cur_bias])
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
