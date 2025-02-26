use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::ff::PrimeField,
  plonk::Error,
};

use crate::{gadgets::gadget::GadgetConfig, utils::helpers::broadcast};

use super::layer::{AssignedTensor, CellRc, LayerConfig};

pub mod add;
pub mod div_var;
pub mod mul;
pub mod sub;

pub trait Arithmetic<F: PrimeField> {
  fn get_inp_size(layer_config: &LayerConfig) -> i64 {
    let inp_size: usize = layer_config
      .inp_shapes
      .iter()
      .map(|x| x.iter().product())
      .max()
      .unwrap();

    inp_size as i64
  }

  fn gadget_forward(
    &self,
    layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    constants: &Vec<&AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error>;

  fn arithmetic_forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<(Vec<CellRc<F>>, Vec<usize>), Error> {
    assert_eq!(tensors.len(), 2);
    // println!("tensors: {:?} {:?}", tensors[0].shape(), tensors[1].shape());
    let (inp1, inp2) = broadcast(&tensors[0], &tensors[1]);
    let out_shape = inp1.shape();
    assert_eq!(inp1.shape(), inp2.shape());

    let zero = constants.get(&0).unwrap().as_ref();

    let inp1_vec = inp1.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let inp2_vec = inp2.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
    let vec_inputs = vec![inp1_vec, inp2_vec];
    let constants = vec![zero];
    let out = self.gadget_forward(
      layouter.namespace(|| ""),
      &vec_inputs,
      &constants,
      gadget_config.clone(),
    )?;
    let out = out.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>();
    Ok((out, out_shape.to_vec()))
  }
}
