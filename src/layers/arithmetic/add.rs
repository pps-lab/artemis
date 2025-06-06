use std::{collections::HashMap, rc::Rc, vec};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter},
  halo2curves::ff::PrimeField,
  plonk::Error,
};
use ndarray::{Array, IxDyn};

use crate::{
  gadgets::{
    add_pairs::AddPairsChip,
    gadget::{Gadget, GadgetConfig, GadgetType},
    nonlinear::relu::ReluChip,
  },
  layers::layer::{ActivationType, AssignedTensor, CellRc, GadgetConsumer},
};

use super::{
  super::layer::{Layer, LayerConfig},
  Arithmetic,
};

#[derive(Clone, Debug)]
pub struct AddChip {}

impl AddChip {
  fn get_activation(&self, layer_params: &Vec<i64>) -> ActivationType {
    let activation = layer_params[0];
    match activation {
      0 => ActivationType::None,
      1 => ActivationType::Relu,
      _ => panic!("Unsupported activation type for add"),
    }
  }
}

impl<F: PrimeField> Arithmetic<F> for AddChip {
  fn gadget_forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    constants: &Vec<&AssignedCell<F, F>>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    let add_pairs_chip = AddPairsChip::<F>::construct(gadget_config);
    let out = add_pairs_chip.forward(layouter.namespace(|| "add chip"), &vec_inputs, constants)?;
    Ok(out)
  }
}

impl<F: PrimeField> Layer<F> for AddChip {
  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    tensors: &Vec<AssignedTensor<F>>,
    constants: &HashMap<i64, CellRc<F>>,
    gadget_config: Rc<GadgetConfig>,
    layer_config: &LayerConfig,
  ) -> Result<Vec<AssignedTensor<F>>, Error> {
    let activation = self.get_activation(&layer_config.layer_params);

    // Do the addition
    let (out, out_shape) = self.arithmetic_forward(
      layouter.namespace(|| ""),
      tensors,
      constants,
      gadget_config.clone(),
    )?;

    // Do the fused activation
    let out = if activation == ActivationType::Relu {
      let zero = constants.get(&0).unwrap();
      let single_inps = vec![zero.as_ref()];

      let out = out.iter().map(|x| x.as_ref()).collect::<Vec<_>>();

      let relu_chip = ReluChip::<F>::construct(gadget_config);
      let out = relu_chip.forward(layouter.namespace(|| "relu"), &vec![out], &single_inps)?;
      let out = out.into_iter().map(|x| Rc::new(x)).collect::<Vec<_>>();
      out
    } else if activation == ActivationType::None {
      out
    } else {
      panic!("Unsupported activation type for add");
    };

    let out = Array::from_shape_vec(IxDyn(out_shape.as_slice()), out).unwrap();

    Ok(vec![out])
  }

  fn num_rows(&self, layer_config: &LayerConfig, num_cols: i64) -> i64 {
    let inp_size = <AddChip as Arithmetic<F>>::get_inp_size(layer_config);

    // TODO: times two? check add_pairs...
    let num_add_per_row = num_cols / 3;
    let mut num_rows = (inp_size as i64).div_ceil(num_add_per_row);

    let activation = self.get_activation(&layer_config.layer_params);
    match activation {
      ActivationType::Relu => {
        let num_relu_per_row = num_cols / 2;
        num_rows += (inp_size as i64).div_ceil(num_relu_per_row);
      }
      ActivationType::None => (),
      _ => panic!("Unsupported activation type for add"),
    }
    //println!("#####Add num_rows: {}", num_rows);
    num_rows
  }
}

impl GadgetConsumer for AddChip {
  fn used_gadgets(&self, layer_config: &LayerConfig) -> Vec<crate::gadgets::gadget::GadgetType> {
    let activation = self.get_activation(&layer_config.layer_params);
    let mut outp = vec![GadgetType::AddPairs];

    match activation {
      ActivationType::Relu => outp.push(GadgetType::Relu),
      ActivationType::None => (),
      _ => panic!("Unsupported activation type for add"),
    }
    outp
  }
}
