use std::{
  collections::{BTreeMap, BTreeSet, HashMap},
  marker::PhantomData,
  rc::Rc,
  sync::{Arc, Mutex}, time::Instant,
};
use bitvec::index::BitIdx;
use ecc::{integer::rns::Rns, maingate::{AssignedValue, MainGate, MainGateConfig, RangeChip, RangeConfig, RangeInstructions, RegionCtx}, AssignedPoint, BaseFieldEccChip, EccConfig};
use halo2_proofs::{
  circuit::{self, AssignedCell, Layouter, SimpleFloorPlanner, Value},
  halo2curves::ff::{FromUniformBytes, PrimeField},
  plonk::{Advice, Circuit, Column, ConstraintSystem, Error, FirstPhase, Instance, Phase, SecondPhase, ThirdPhase}, poly::{self, Coeff, Polynomial},
};
use halo2curves::{bn256::Bn256, pairing::Engine, CurveAffine};
use lazy_static::lazy_static;
use ndarray::{Array, IxDyn};
use num_bigint::BigUint;
use rand_core::OsRng;
use rmp_serde::config;
use ff::Field;
use ecc::maingate::MainGateInstructions;
use group::{Curve, Group};
use crate::{
  commitments::{
    commit::Commit,
    packer::PackerChip,
    poseidon_commit::{PoseidonCommitChip, L, RATE, WIDTH},
  },
  gadgets::{
    add_pairs::AddPairsChip, adder::AdderChip, bias_div_round_relu6::BiasDivRoundRelu6Chip, dot_prod::DotProductChip, dot_prod_bias::DotProductBiasChip, gadget::{Gadget, GadgetConfig, GadgetType}, input_lookup::InputLookupChip, max::MaxChip, mul_pairs::MulPairsChip, nonlinear::{cos::CosGadgetChip, exp::ExpGadgetChip, logistic::LogisticGadgetChip, pow::PowGadgetChip, relu::ReluChip, relu_decompose::ReluDecomposeChip, rsqrt::RsqrtGadgetChip, sin::SinGadgetChip, sqrt::SqrtGadgetChip, tanh::TanhGadgetChip}, poly::PolyChip, poly_2::Poly2Chip, poly_3::Poly3Chip, poly_4::Poly4Chip, sqrt_big::SqrtBigChip, square::SquareGadgetChip, squared_diff::SquaredDiffGadgetChip, sub_pairs::SubPairsChip, update::UpdateGadgetChip, var_div::VarDivRoundChip, var_div_big::VarDivRoundBigChip, var_div_big3::VarDivRoundBig3Chip
  },
  layers::{
    arithmetic::{add::AddChip, div_var::DivVarChip, mul::MulChip, sub::SubChip},
    avg_pool_2d::AvgPool2DChip,
    batch_mat_mul::BatchMatMulChip,
    conv2d::Conv2DChip,
    cos::CosChip,
    dag::{DAGLayerChip, DAGLayerConfig},
    fc::fully_connected::{FullyConnectedChip, FullyConnectedConfig},
    layer::{AssignedTensor, CellRc, GadgetConsumer, LayerConfig, LayerType},
    logistic::LogisticChip,
    max_pool_2d::MaxPool2DChip,
    mean::MeanChip,
    noop::NoopChip,
    pow::PowChip,
    rsqrt::RsqrtChip,
    shape::{
      broadcast::BroadcastChip, concatenation::ConcatenationChip, mask_neg_inf::MaskNegInfChip,
      pack::PackChip, pad::PadChip, permute::PermuteChip, reshape::ReshapeChip,
      resize_nn::ResizeNNChip, rotate::RotateChip, slice::SliceChip, split::SplitChip,
      transpose::TransposeChip,
    },
    sin::SinChip,
    softmax::SoftmaxChip,
    sqrt::SqrtChip,
    square::SquareChip,
    squared_diff::SquaredDiffChip,
    tanh::TanhChip,
    update::UpdateChip,
  },
  utils::{
    helpers::{convert_to_bigint, RAND_START_IDX},
    loader::{load_model_msgpack, ModelMsgpack},
  },
};
type Bn256G1 =  <Bn256 as Engine>::G1Affine;
lazy_static! {
  pub static ref GADGET_CONFIG: Mutex<GadgetConfig> = Mutex::new(GadgetConfig::default());
  pub static ref PUBLIC_VALS: Mutex<Vec<BigUint>> = Mutex::new(vec![]);
}

#[derive(Clone, Debug, Default)]
pub struct ModelCircuit<C: CurveAffine<ScalarExt: PrimeField>> {
  pub used_gadgets: Arc<BTreeSet<GadgetType>>,
  pub dag_config: DAGLayerConfig,
  pub tensors: BTreeMap<i64, Array<C::ScalarExt, IxDyn>>,
  pub commit_before: Vec<Vec<i64>>,
  pub commit_after: Vec<Vec<i64>>,
  pub k: usize,
  pub beta_pows: Vec<C::ScalarExt>, 
  pub bits_per_elem: usize,
  pub inp_idxes: Vec<i64>,
  pub num_random: i64,
}

#[derive(Clone, Debug)]
pub struct ModelConfig<F: PrimeField + Ord + FromUniformBytes<64>> {
  pub gadget_config: Rc<GadgetConfig>,
  pub public_col: Column<Instance>,
  pub hasher: Option<PoseidonCommitChip<F, WIDTH, RATE, L>>,
  pub _marker: PhantomData<F>,
  pub main_gate_config: MainGateConfig,
  pub range_config: RangeConfig,
}

impl<F: PrimeField + Ord + FromUniformBytes<64>>  ModelConfig<F> {
  fn ecc_chip_config(&self) -> EccConfig {
    EccConfig::new(self.range_config.clone(), self.main_gate_config.clone())
  }

  fn config_range<N: PrimeField>(
    &self,
    layouter: &mut impl Layouter<N>,
  ) -> Result<(), Error> {
      let range_chip = RangeChip::<N>::new(self.range_config.clone());
      range_chip.load_table(layouter)?;

      Ok(())
  }
}

impl<C: CurveAffine<ScalarExt: PrimeField + Ord + FromUniformBytes<64>>> ModelCircuit<C> {
  pub fn assign_tensors_map(
    &self,
    mut layouter: impl Layouter<C::ScalarExt>,
    columns: &Vec<Column<Advice>>,
    tensors: &BTreeMap<i64, Array<C::ScalarExt, IxDyn>>,
  ) -> Result<(BTreeMap<i64, AssignedTensor<C::ScalarExt>>, Vec<Rc<AssignedCell<C::ScalarExt, C::ScalarExt>>>, Vec<C::ScalarExt>), Error> {
    let (tensors, flat, flat_f) = layouter.assign_region(
      || "asssignment",
      |mut region| {
        let mut assigned_tensors = BTreeMap::new();
        //println!("witness columns len: {:?}", columns.len());
          // let mut cell_idx = 0;
          // for (tensor_idx, tensor) in tensors.iter() {
          //   let mut flat = vec![];
          //   for val in tensor.iter() {
          //     let row_idx = cell_idx;
          //     let cell = region
          //       .assign_advice(
          //         || "assignment",
          //         columns_witness[0],
          //         row_idx,
          //         || Value::known(*val),
          //       )
          //       .unwrap();
          //     flat.push(Rc::new(cell));
          //     cell_idx += 1;
          //   }
          //   let tensor = Array::from_shape_vec(tensor.shape(), flat).unwrap();
          //   assigned_tensors.insert(*tensor_idx, tensor);
          //   // let tensor = Array::from_shape_vec(tensor.shape(), flat).unwrap();
          //   // assigned_tensors.insert(*tensor_idx, tensor);
          // }
        let mut cell_idx = 0;
        let mut big_flat = vec![];
        let mut flat_f = vec![];
        for (tensor_idx, tensor) in tensors.iter() {
          let mut flat = vec![];
          for val in tensor.iter() {
            let row_idx = cell_idx / columns.len();
            let col_idx = cell_idx % columns.len();
            let cell = region
              .assign_advice(
                || "assignment",
                columns[col_idx],
                row_idx,
                || Value::known(*val),
              )
              .unwrap();
            flat.push(Rc::new(cell.clone()));
            flat_f.push(*val);
            big_flat.push(Rc::new(cell));
            cell_idx += 1;
          }
          let tensor = Array::from_shape_vec(tensor.shape(), flat).unwrap();
          assigned_tensors.insert(*tensor_idx, tensor);
        }

        Ok((assigned_tensors, big_flat, flat_f))
      },
    )?;

    Ok((tensors, flat, flat_f))
  }

  pub fn tensor_map_to_vec(
    &self,
    tensor_map: &BTreeMap<i64, Array<CellRc<C::ScalarExt>, IxDyn>>,
  ) -> Vec<AssignedTensor<C::ScalarExt>> {
    let smallest_tensor = tensor_map
      .iter()
      .min_by_key(|(_, tensor)| tensor.len())
      .unwrap()
      .1;
    let max_tensor_key = tensor_map
      .iter()
      .max_by_key(|(key, _)| *key)
      .unwrap()
      .0
      .clone();
    let mut tensors = vec![];
    for i in 0..max_tensor_key + 1 {
      let tensor = tensor_map.get(&i).unwrap_or(smallest_tensor);
      tensors.push(tensor.clone());
    }

    tensors
  }

  pub fn assign_tensors_vec(
    &self,
    mut layouter: impl Layouter<C::ScalarExt>,
    columns: &Vec<Column<Advice>>,
    tensors: &BTreeMap<i64, Array<C::ScalarExt, IxDyn>>,
  ) -> Result<(Vec<AssignedTensor<C::ScalarExt>>, Vec<Rc<AssignedCell<C::ScalarExt, C::ScalarExt>>>, Vec<C::ScalarExt>), Error>  {
    let (tensor_map, flat, flat_f) = self
      .assign_tensors_map(
        layouter.namespace(|| "assign_tensors_map"),
        columns,
        tensors,
      )
      .unwrap();
    //println!("{:?}", flat);
    Ok((self.tensor_map_to_vec(&tensor_map), flat, flat_f))
  }

  pub fn assign_constants(
    &self,
    mut layouter: impl Layouter<C::ScalarExt>,
    gadget_config: Rc<GadgetConfig>,
  ) -> Result<HashMap<i64, CellRc<C::ScalarExt>>, Error> {
    let sf = gadget_config.scale_factor;
    let min_val = gadget_config.min_val;
    let max_val = gadget_config.max_val;

    let constants = layouter.assign_region(
      || "constants",
      |mut region| {
        let mut constants: HashMap<i64, CellRc<C::ScalarExt>> = HashMap::new();

        let vals = vec![0 as i64, 1, sf as i64, min_val, max_val];
        let shift_val_i64 = -min_val * 2; // FIXME
        let shift_val_f = C::ScalarExt::from(shift_val_i64 as u64);
        for (i, val) in vals.iter().enumerate() {
          let cell = region.assign_fixed(
            || format!("constant_{}", i),
            gadget_config.fixed_columns[0],
            i,
            || Value::known(C::ScalarExt::from((val + shift_val_i64) as u64) - shift_val_f),
          )?;
          constants.insert(*val, Rc::new(cell));
        }

        // TODO: I've made some very bad life decisions
        // TOOD: this needs to be a random oracle
        let r_base = C::ScalarExt::from(0x123456789abcdef);
        let mut r = r_base.clone();
        //println!("Num randoms: {}", self.num_random);
        for i in 0..self.num_random {
          let rand = region.assign_fixed(
            || format!("rand_{}", i),
            gadget_config.fixed_columns[0],
            constants.len(),
            || Value::known(r),
          )?;
          r = r * r_base;
          constants.insert(RAND_START_IDX + (i as i64), Rc::new(rand));
        }

        Ok(constants)
      },
    )?;
    Ok(constants)
  }

  // TODO: for some horrifying reason, assigning to fixed columns causes everything to blow up
  // Currently get around this by assigning to advice columns
  // This is secure because of the equality checks but EXTREMELY STUPID
  pub fn assign_constants2(
    &self,
    mut layouter: impl Layouter<C::ScalarExt>,
    gadget_config: Rc<GadgetConfig>,
    fixed_constants: &HashMap<i64, CellRc<C::ScalarExt>>,
  ) -> Result<HashMap<i64, CellRc<C::ScalarExt>>, Error> {
    let sf = gadget_config.scale_factor;
    let min_val = gadget_config.min_val;
    let max_val = gadget_config.max_val;

    let constants = layouter.assign_region(
      || "constants",
      |mut region| {
        let mut constants: HashMap<i64, CellRc<C::ScalarExt>> = HashMap::new();

        let vals = vec![0 as i64, 1, sf as i64, min_val, max_val];
        let shift_val_i64 = -min_val * 2; // FIXME
        let shift_val_f = C::ScalarExt::from(shift_val_i64 as u64);
        for (i, val) in vals.iter().enumerate() {
          let assignment_idx = i as usize;
          let row_idx = assignment_idx / gadget_config.columns.len();
          let col_idx = assignment_idx % gadget_config.columns.len();
          let cell = region.assign_advice(
            || format!("constant_{}", i),
            gadget_config.columns[col_idx],
            row_idx,
            || Value::known(C::ScalarExt::from((val + shift_val_i64) as u64) - shift_val_f),
          )?;
          constants.insert(*val, Rc::new(cell));
        }

        // TODO: I've made some very bad life decisions
        // TOOD: this needs to be a random oracle
        let r_base = C::ScalarExt::from(0x123456789abcdef);
        let mut r = r_base.clone();
        for i in 0..self.num_random {
          let assignment_idx = constants.len();
          let row_idx = assignment_idx / gadget_config.columns.len();
          let col_idx = assignment_idx % gadget_config.columns.len();
          let rand = region.assign_advice(
            || format!("rand_{}", i),
            gadget_config.columns[col_idx],
            row_idx,
            || Value::known(r),
          )?;
          r = r * r_base;
          constants.insert(RAND_START_IDX + (i as i64), Rc::new(rand));
        }

        for (k, v) in fixed_constants.iter() {
          let v2 = constants.get(k).unwrap();
          region.constrain_equal(v.cell(), v2.cell()).unwrap();
        }
        Ok(constants)
      },
    )?;
    Ok(constants)
  }

  pub fn generate_from_file(config_file: &str, inp_file: &str, witness_column: bool, chunks: usize, k_ipt: usize, c_ipt: usize, pedersen: bool) -> ModelCircuit<C> {
    let config = load_model_msgpack(config_file, inp_file, witness_column);
    Self::generate_from_msgpack(config, true, witness_column, chunks, k_ipt, c_ipt, pedersen)
  }

  pub fn generate_from_msgpack(config: ModelMsgpack, panic_empty_tensor: bool, poly_commit: bool, chunks: usize, k_ipt: usize, c_ipt: usize, pedersen: bool) -> ModelCircuit<C> {
    let to_field = |x: i64| {
      let bias = 1 << 31;
      let x_pos = x + bias;
      C::ScalarExt::from(x_pos as u64) - C::ScalarExt::from(bias as u64)
    };
    let match_layer = |x: &str| match x {
      "AveragePool2D" => LayerType::AvgPool2D,
      "Add" => LayerType::Add,
      "BatchMatMul" => LayerType::BatchMatMul,
      "Broadcast" => LayerType::Broadcast,
      "Concatenation" => LayerType::Concatenation,
      "Conv2D" => LayerType::Conv2D,
      "Cos" => LayerType::Cos,
      "Div" => LayerType::DivFixed, // TODO: rename to DivFixed
      "DivVar" => LayerType::DivVar,
      "FullyConnected" => LayerType::FullyConnected,
      "Logistic" => LayerType::Logistic,
      "MaskNegInf" => LayerType::MaskNegInf,
      "MaxPool2D" => LayerType::MaxPool2D,
      "Mean" => LayerType::Mean,
      "Mul" => LayerType::Mul,
      "Noop" => LayerType::Noop,
      "Pack" => LayerType::Pack,
      "Pad" => LayerType::Pad,
      "Pow" => LayerType::Pow,
      "Permute" => LayerType::Permute,
      "Reshape" => LayerType::Reshape,
      "ResizeNearestNeighbor" => LayerType::ResizeNN,
      "Rotate" => LayerType::Rotate,
      "Rsqrt" => LayerType::Rsqrt,
      "Sin" => LayerType::Sin,
      "Slice" => LayerType::Slice,
      "Softmax" => LayerType::Softmax,
      "Split" => LayerType::Split,
      "Sqrt" => LayerType::Sqrt,
      "Square" => LayerType::Square,
      "SquaredDifference" => LayerType::SquaredDifference,
      "Sub" => LayerType::Sub,
      "Tanh" => LayerType::Tanh,
      "Transpose" => LayerType::Transpose,
      "Update" => LayerType::Update,
      _ => panic!("unknown op: {}", x),
    };

    let mut tensors = BTreeMap::new();
    for flat in config.tensors {
      let value_flat = flat.data.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
      let shape = flat.shape.iter().map(|x| *x as usize).collect::<Vec<_>>();
      let num_el: usize = shape.iter().product();
      if panic_empty_tensor && num_el != value_flat.len() {
        panic!("tensor shape and data length mismatch");
      }
      let tensor = if num_el == value_flat.len() {
        Array::from_shape_vec(IxDyn(&shape), value_flat).unwrap()
      } else {
        // Fill with zeros
        Array::from_shape_vec(IxDyn(&shape), vec![C::ScalarExt::ZERO; num_el]).unwrap()
      };

      tensors.insert(flat.idx, tensor);
    }

    let i64_to_usize = |x: &Vec<i64>| x.iter().map(|x| *x as usize).collect::<Vec<_>>();

    let mut used_gadgets = BTreeSet::new();

    let dag_config = {
      let ops = config
        .layers
        .iter()
        .map(|layer| {
          let layer_type = match_layer(&layer.layer_type);
          let layer_config = LayerConfig {
            layer_type,
            layer_params: layer.params.clone(),
            inp_shapes: layer.inp_shapes.iter().map(|x| i64_to_usize(x)).collect(),
            out_shapes: layer.out_shapes.iter().map(|x| i64_to_usize(x)).collect(),
            mask: layer.mask.clone(),
            implementation_idx: layer.implementation.unwrap_or(0),
          };
          let layer_gadgets = match layer_type {
            LayerType::Add => Box::new(AddChip {}) as Box<dyn GadgetConsumer>,
            LayerType::AvgPool2D => Box::new(AvgPool2DChip {}) as Box<dyn GadgetConsumer>,
            LayerType::BatchMatMul => Box::new(BatchMatMulChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Broadcast => Box::new(BroadcastChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Concatenation => Box::new(ConcatenationChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Cos => Box::new(CosChip {}) as Box<dyn GadgetConsumer>,
            LayerType::DivFixed => Box::new(ConcatenationChip {}) as Box<dyn GadgetConsumer>,
            LayerType::DivVar => Box::new(DivVarChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Conv2D => Box::new(Conv2DChip {
              config: LayerConfig::default(),
              _marker: PhantomData::<C::ScalarExt>,
            }) as Box<dyn GadgetConsumer>,
            LayerType::FullyConnected => Box::new(FullyConnectedChip {
              config: FullyConnectedConfig { normalize: true },
              _marker: PhantomData::<C::ScalarExt>,
            }) as Box<dyn GadgetConsumer>,
            LayerType::Logistic => Box::new(LogisticChip {}) as Box<dyn GadgetConsumer>,
            LayerType::MaskNegInf => Box::new(MaskNegInfChip {}) as Box<dyn GadgetConsumer>,
            LayerType::MaxPool2D => Box::new(MaxPool2DChip {
              marker: PhantomData::<C::ScalarExt>,
            }) as Box<dyn GadgetConsumer>,
            LayerType::Mean => Box::new(MeanChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Mul => Box::new(MulChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Noop => Box::new(NoopChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Pack => Box::new(PackChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Pad => Box::new(PadChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Pow => Box::new(PowChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Permute => Box::new(PermuteChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Reshape => Box::new(ReshapeChip {}) as Box<dyn GadgetConsumer>,
            LayerType::ResizeNN => Box::new(ResizeNNChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Rotate => Box::new(RotateChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Rsqrt => Box::new(RsqrtChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Sin => Box::new(SinChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Slice => Box::new(SliceChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Softmax => Box::new(SoftmaxChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Split => Box::new(SplitChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Sqrt => Box::new(SqrtChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Square => Box::new(SquareChip {}) as Box<dyn GadgetConsumer>,
            LayerType::SquaredDifference => Box::new(SquaredDiffChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Sub => Box::new(SubChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Tanh => Box::new(TanhChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Transpose => Box::new(TransposeChip {}) as Box<dyn GadgetConsumer>,
            LayerType::Update => Box::new(UpdateChip {}) as Box<dyn GadgetConsumer>,
          }
          .used_gadgets(&layer_config);
          for gadget in layer_gadgets {
            used_gadgets.insert(gadget);
          }
          layer_config
        })
        .collect::<Vec<_>>();
      let inp_idxes = config
        .layers
        .iter()
        .map(|layer| i64_to_usize(&layer.inp_idxes))
        .collect::<Vec<_>>();
      let out_idxes = config
        .layers
        .iter()
        .map(|layer| i64_to_usize(&layer.out_idxes))
        .collect::<Vec<_>>();
      let final_out_idxes = config
        .out_idxes
        .iter()
        .map(|x| *x as usize)
        .collect::<Vec<_>>();
      DAGLayerConfig {
        inp_idxes,
        out_idxes,
        ops,
        final_out_idxes,
      }
    };
    let mut params_len = 0;
    if poly_commit {
      for (idx, tensor) in tensors.iter() {
        for val in tensor {
          params_len += 1;
        }
      }
  
      while params_len % chunks != 0 {
        params_len += 1;
      }
    }


    // The input lookup is always used
    used_gadgets.insert(GadgetType::InputLookup);
    let used_gadgets = Arc::new(used_gadgets);
    let gadget = &GADGET_CONFIG;
    let cloned_gadget = gadget.lock().unwrap().clone();

    *gadget.lock().unwrap() = GadgetConfig {
      poly_commit,
      pedersen,
      scale_factor: config.global_sf as u64,
      shift_min_val: -(config.global_sf * config.global_sf * (1 << 17)),
      div_outp_min_val: -(1 << (k_ipt as usize - 1)),
      min_val: -(1 << (k_ipt as usize - 1)),
      max_val: (1 << (k_ipt as usize - 1)) - 10,
      k: k_ipt as usize, // additional k for encoding the witness in 1 row
      num_rows: (1 << k_ipt as usize) - 10 + 1,
      num_cols:  c_ipt as usize,
      used_gadgets: used_gadgets.clone(),
      commit_before: config.commit_before.clone().unwrap_or(vec![]),
      commit_after: config.commit_after.clone().unwrap_or(vec![]),
      use_selectors: config.use_selectors.unwrap_or(true),
      num_bits_per_elem: config.bits_per_elem.unwrap_or(config.k),
      poly_ell: chunks,
      ..cloned_gadget
    };
    println!("!Tensors: {:?}", tensors.keys());
    println!("!commit_before: {:?}", config.commit_before);
    ModelCircuit {
      tensors,
      dag_config,
      used_gadgets,
      beta_pows: vec![C::ScalarExt::ZERO; params_len],
      k: k_ipt as usize,
      bits_per_elem: config.bits_per_elem.unwrap_or(k_ipt as i64) as usize,
      inp_idxes: config.inp_idxes.clone(),
      commit_after: config.commit_after.unwrap_or(vec![]),
      commit_before: config.commit_before.unwrap_or(vec![]),
      num_random: config.num_random.unwrap_or(0),
    }
  }

  pub fn assign_and_commit(
    &self,
    mut layouter: impl Layouter<C::ScalarExt>,
    constants: &HashMap<i64, CellRc<C::ScalarExt>>,
    config: &ModelConfig<C::ScalarExt>,
    tensors: &BTreeMap<i64, Array<C::ScalarExt, IxDyn>>,
  ) -> (BTreeMap<i64, AssignedTensor<C::ScalarExt>>, CellRc<C::ScalarExt>) {
    let num_bits = self.bits_per_elem;
    let packer_config = PackerChip::<C::ScalarExt>::construct(num_bits, config.gadget_config.as_ref());
    let packer_chip = PackerChip::<C::ScalarExt> {
      config: packer_config,
    };
    let (tensor_map, packed) = packer_chip
      .assign_and_pack(
        layouter.namespace(|| "packer"),
        config.gadget_config.clone(),
        constants,
        tensors,
      )
      .unwrap();

    let zero = constants.get(&0).unwrap().clone();
    let commit_chip = config.hasher.clone().unwrap();
    //// TODO: ADD A new commitment type here !:))
    let commitments = commit_chip
      .commit(
        layouter.namespace(|| "commit"),
        config.gadget_config.clone(),
        constants,
        &packed,
        zero.clone(),
      )
      .unwrap();
    assert_eq!(commitments.len(), 1);

    (tensor_map, commitments[0].clone())
  }

  pub fn copy_and_commit(
    &self,
    mut layouter: impl Layouter<C::ScalarExt>,
    constants: &HashMap<i64, CellRc<C::ScalarExt>>,
    config: &ModelConfig<C::ScalarExt>,
    tensors: &BTreeMap<i64, AssignedTensor<C::ScalarExt>>,
  ) -> CellRc<C::ScalarExt> {
    let num_bits = self.bits_per_elem;
    let packer_config = PackerChip::<C::ScalarExt>::construct(num_bits, config.gadget_config.as_ref());
    let packer_chip = PackerChip::<C::ScalarExt> {
      config: packer_config,
    };
    let packed = packer_chip
      .copy_and_pack(
        layouter.namespace(|| "packer"),
        config.gadget_config.clone(),
        constants,
        tensors,
      )
      .unwrap();

    let zero = constants.get(&0).unwrap().clone();
    let commit_chip = config.hasher.clone().unwrap();

    let commitments = commit_chip
      .commit(
        layouter.namespace(|| "commit"),
        config.gadget_config.clone(),
        constants,
        &packed,
        zero.clone(),
      )
      .unwrap();
    assert_eq!(commitments.len(), 1);

    commitments[0].clone()
  }
}

impl<C: CurveAffine<ScalarExt: PrimeField + Ord + FromUniformBytes<64>>> Circuit<C::ScalarExt> for ModelCircuit<C> {
  type Config = ModelConfig<C::ScalarExt>;
  type FloorPlanner = SimpleFloorPlanner;
  type Params = ();

  fn without_witnesses(&self) -> Self {
    todo!()
  }

  fn configure(meta: &mut ConstraintSystem<C::ScalarExt>) -> Self::Config {
    let mut gadget_config = crate::model::GADGET_CONFIG.lock().unwrap().clone();
    gadget_config.columns_poly = (0..(gadget_config.poly_ell))
    .map(|_| {
      let col = meta.advice_column();
      println!("Assigned column: {:?}", col);
      col
    })
    .collect::<Vec<_>>();

    let columns: Vec<Column<Advice>>;
    //let beta_pows = (0..gadget_config.poly_ell + 1).map(|i| beta.pow([i as u64])).rev().collect::<Vec<_>>();
    if gadget_config.poly_commit {
      let beta = meta.challenge_usable_after(FirstPhase);
      gadget_config.columns_poly.push(meta.advice_column_in(SecondPhase));
      gadget_config.columns_poly.push(meta.advice_column_in(SecondPhase));
      gadget_config.beta = beta;
      //gadget_config.columns_poly.push(meta.advice_column());
      
      // gadget_config.columns_poly_public = (0..gadget_config.poly_ell)
      // .map(|_| meta.instance_column())
      // .collect::<Vec<_>>();
      gadget_config = Poly4Chip::configure(meta, gadget_config);
      columns = (0..gadget_config.num_cols)
      .map(|_| meta.advice_column_in(SecondPhase))
      .collect::<Vec<_>>();
    } else {
      columns = (0..gadget_config.num_cols)
      .map(|_| meta.advice_column_in(FirstPhase))
      .collect::<Vec<_>>();
    }
    for col in gadget_config.columns_witness.iter() {
      meta.enable_equality(*col);
    }
    for col in gadget_config.columns_poly.iter() {
      meta.enable_equality(*col);
    }

    //println!("num columns: {}", gadget_config.num_cols);

    let maingate_cols = columns[0..5].to_vec();
    for col in columns.iter() {
      meta.enable_equality(*col);
    }
  
    gadget_config.columns = columns;

    let public_col = meta.instance_column();

    meta.enable_equality(public_col);
    gadget_config.columns_public = vec![public_col];

    gadget_config.fixed_columns = vec![meta.fixed_column()];
    meta.enable_equality(gadget_config.fixed_columns[0]);

    // The input lookup is always loaded
    gadget_config = InputLookupChip::<C::ScalarExt>::configure(meta, gadget_config);

    let used_gadgets = gadget_config.used_gadgets.clone();
    for gadget_type in used_gadgets.iter() {
      gadget_config = match gadget_type {
        GadgetType::AddPairs => AddPairsChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Adder => AdderChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::BiasDivRoundRelu6 => BiasDivRoundRelu6Chip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::BiasDivFloorRelu6 => panic!(),
        GadgetType::Cos => CosGadgetChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::DotProduct => DotProductChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::DotProductBias => DotProductBiasChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Exp => ExpGadgetChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Logistic => LogisticGadgetChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Max => MaxChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::MulPairs => MulPairsChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Pow => PowGadgetChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Relu => ReluChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::ReluDecompose => ReluDecomposeChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Rsqrt => RsqrtGadgetChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Sin => SinGadgetChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Sqrt => SqrtGadgetChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::SqrtBig => SqrtBigChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Square => SquareGadgetChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::SquaredDiff => SquaredDiffGadgetChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::SubPairs => SubPairsChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Tanh => TanhGadgetChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::VarDivRound => VarDivRoundChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::VarDivRoundBig => VarDivRoundBigChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::VarDivRoundBig3 => VarDivRoundBig3Chip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::InputLookup => gadget_config, // This is always loaded
        GadgetType::Update => UpdateGadgetChip::<C::ScalarExt>::configure(meta, gadget_config),
        GadgetType::Packer => panic!(),
        GadgetType::Poly => PolyChip::configure(meta, gadget_config),
        //GadgetType::Poly2 => Poly2Chip::configure(meta, gadget_config),
      };
    }
    println!("commit_before: {:?}", gadget_config.commit_before);
    let hasher = if gadget_config.commit_before.len() + gadget_config.commit_after.len() > 0 {
      let packer_config =
        PackerChip::<C::ScalarExt>::construct(gadget_config.num_bits_per_elem as usize, &gadget_config);
      gadget_config = PackerChip::<C::ScalarExt>::configure(meta, packer_config, gadget_config);

      // TODO
      let input = gadget_config.columns[0..L].try_into().unwrap();
      let state = gadget_config.columns[L..L + WIDTH].try_into().unwrap();
      let partial_sbox = gadget_config.columns[L + WIDTH].into();
      Some(PoseidonCommitChip::<C::ScalarExt, WIDTH, RATE, L>::configure(
        meta,
        input,
        state,
        partial_sbox,
      ))
    } else {
      None
    };

    let mut main_gate_config= MainGateConfig::default();
    let mut range_config= RangeConfig::default();

    if gadget_config.pedersen {
      let rns = Rns::<C::Base, C::ScalarExt, 4, 68>::construct();
      //let rns = Rns::<C::Base, C::ScalarExt, 4, 64>::construct();
      main_gate_config = MainGate::<C::ScalarExt>::configure_reuse(meta, gadget_config.columns_poly.clone(), gadget_config.columns_public[0]);
      //main_gate_config = MainGate::<C::ScalarExt>::configure(meta, gadget_config.columns_public[0]);
      let overflow_bit_lens = rns.overflow_lengths();
      let composition_bit_lens = vec![17];
      println!("Overflow bit lens: {:?}", overflow_bit_lens);

      range_config = RangeChip::<C::ScalarExt>::configure(
          meta,
          &main_gate_config,
          composition_bit_lens,
          overflow_bit_lens,
      );
    }
    println!("Phases: {:?}", meta.advice_column_phase());

    ModelConfig {
      gadget_config: gadget_config.into(),
      public_col,
      hasher,
      _marker: PhantomData,
      main_gate_config,
      range_config,
    }
  }

  fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<C::ScalarExt>) -> Result<(), Error> {
    let timer = Instant::now();
    // Assign tables
    let gadget_rc: Rc<GadgetConfig> = config.gadget_config.clone().into();
    for gadget in self.used_gadgets.iter() {
      match gadget {
        GadgetType::AddPairs => {
          let chip = AddPairsChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "add pairs lookup"))?;
        }
        GadgetType::Adder => {
          let chip = AdderChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "adder lookup"))?;
        }
        GadgetType::BiasDivRoundRelu6 => {
          let chip = BiasDivRoundRelu6Chip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "bias div round relu6 lookup"))?;
        }
        GadgetType::Cos => {
          let chip = CosGadgetChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "cos lookup"))?;
        }
        GadgetType::DotProduct => {
          let chip = DotProductChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "dot product lookup"))?;
        }
        GadgetType::DotProductBias => {
          let chip = DotProductBiasChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "dot product bias lookup"))?;
        }
        GadgetType::VarDivRound => {
          let chip = VarDivRoundChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "var div lookup"))?;
        }
        GadgetType::Pow => {
          let chip = PowGadgetChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "pow lookup"))?;
        }
        GadgetType::Relu => {
          let chip = ReluChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "relu lookup"))?;
        }
        GadgetType::ReluDecompose => {
          let chip = ReluDecomposeChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "relu decompose lookup"))?;
        }
        GadgetType::Rsqrt => {
          let chip = RsqrtGadgetChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "rsqrt lookup"))?;
        }
        GadgetType::Sin => {
          let chip = SinGadgetChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "sin lookup"))?;
        }
        GadgetType::Sqrt => {
          let chip = SqrtGadgetChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "sqrt lookup"))?;
        }
        GadgetType::Tanh => {
          let chip = TanhGadgetChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "tanh lookup"))?;
        }
        GadgetType::Exp => {
          let chip = ExpGadgetChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "exp lookup"))?;
        }
        GadgetType::Logistic => {
          let chip = LogisticGadgetChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "logistic lookup"))?;
        }
        GadgetType::InputLookup => {
          let chip = InputLookupChip::<C::ScalarExt>::construct(gadget_rc.clone());
          chip.load_lookups(layouter.namespace(|| "input lookup"))?;
        }
        GadgetType::VarDivRoundBig => {}
        GadgetType::VarDivRoundBig3 => {}
        GadgetType::Max => {}
        GadgetType::MulPairs => {}
        GadgetType::SqrtBig => {}
        GadgetType::Square => {}
        GadgetType::SquaredDiff => {}
        GadgetType::SubPairs => {}
        GadgetType::Update => {}
        _ => panic!("unsupported gadget {:?}", gadget),
      }
    }

    // Assign weights and constants
    let constants_base = self
      .assign_constants(
        layouter.namespace(|| "constants"),
        config.gadget_config.clone(),
      )
      .unwrap();
    // Some halo2 cancer
    let constants = self
      .assign_constants2(
        layouter.namespace(|| "constants 2"),
        config.gadget_config.clone(),
        &constants_base,
      )
      .unwrap();

    let mut commitments = vec![];
    let (tensors, flat, mut flat_f) = if self.commit_before.len() > 0 {
      // Commit to the tensors before the DAG
      //println!("commit to the tensors :)))");
      let mut tensor_map = BTreeMap::new();
      let mut ignore_idxes: Vec<i64> = vec![];
      // for commit_idxes in self.commit_before.iter() {
        //println!("Indexes:{:?})))", commit_idxes);
        let to_commit = BTreeMap::from_iter(
          self.commit_before[0]
            .iter()
            .map(|idx| (*idx, self.tensors.get(idx).expect(&format!("Idx: {:?}", idx)).clone())),
        );
        
        println!("num tensors: {:?}", self.tensors.iter().size_hint());
        // for idx in commit_idxes.iter() {
        //   println!("Tensor: {:?}", self.tensors.get(idx).unwrap().clone());
        // }
        let (mut committed_tensors, commitment) = self.assign_and_commit(
          layouter.namespace(|| "commit"),
          &constants,
          &config,
          &to_commit,
        );
        commitments.push(commitment);
        tensor_map.append(&mut committed_tensors);
        ignore_idxes.extend(self.commit_before[0].iter());
      // }

      // Assign the remainder of the tensors
      let mut assign_map = BTreeMap::new();
      for (idx, tensor) in self.tensors.iter() {
        if ignore_idxes.contains(idx) {
          continue;
        }
        assign_map.insert(*idx, tensor.clone());
      }
      let (mut remainder_tensor_map, flat, flat_f) = self
        .assign_tensors_map(
          layouter.namespace(|| "assignment"),
          &config.gadget_config.columns,
          &assign_map,
        )
        .unwrap();

      // Merge the two maps
      tensor_map.append(&mut remainder_tensor_map);

      // Return the tensors
      (self.tensor_map_to_vec(&tensor_map), vec![], vec![])
    } else if config.gadget_config.poly_commit {
      println!("Called once");
      self
      .assign_tensors_vec(
        layouter.namespace(|| "assignment"),
        &config.gadget_config.columns,
        &self.tensors,
      )
      .unwrap()
    } else {
      self
        .assign_tensors_vec(
          layouter.namespace(|| "assignment"),
          &config.gadget_config.columns,
          &self.tensors,
        )
        .unwrap()
    };  

    let mut rho = vec![];
    let mut poly_coeffs = vec![];
    for val in flat {
      poly_coeffs.push(val.clone());
    }

    let mut new_public_vals = vec![];

    if config.gadget_config.poly_commit {
      //poly_vals = vec![poly_betas, poly_coeffs];
      //let new_betas = poly_vals[0].iter().map(|x| x.as_ref()).collect();
      let zero = constants.get(&0).unwrap();
      let mut new_coeffs = poly_coeffs.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
      let beta = layouter.get_challenge(config.gadget_config.beta);
      new_public_vals.push(convert_to_bigint(beta.map(|x| x.to_owned())));
      //let poly: Polynomial<C::Scalar, Coeff> = Polynomial::from_coefficients_vec(poly_coeffs.clone());
      let beta_pows = (0..config.gadget_config.poly_ell + 1).map(|i| beta.map(|x| x.pow([i as u64]))).rev().collect::<Vec<_>>();
      while new_coeffs.len() % (self.beta_pows.len() - 1) != 0 {
        new_coeffs.push(&zero);
        flat_f.push(C::ScalarExt::ZERO)
      }

      new_coeffs = new_coeffs.clone().into_iter().rev().collect::<Vec<_>>();
      flat_f = flat_f.clone().into_iter().rev().collect::<Vec<_>>();
      let poly_com_chip = Poly4Chip::<C::ScalarExt>::construct(gadget_rc.clone(), beta_pows.clone(), vec![C::ScalarExt::ONE; config.gadget_config.poly_ell], flat_f);
      //new_public_vals.push(beta);
      rho = poly_com_chip.forward(layouter.namespace(|| "poly commit"), vec![new_coeffs.clone()].as_ref(), vec![zero.as_ref()].as_ref()).unwrap();
      println!("RHO Circ: {:?}", rho);
    }

    if config.gadget_config.pedersen {
      let ecc_chip_config = config.ecc_chip_config();
      let mut ecc_chip =
          BaseFieldEccChip::<C, 4, 68>::new(ecc_chip_config);
      // let mut ecc_chip =
      //     BaseFieldEccChip::<C, 4, 64>::new(ecc_chip_config);
      let main_gate = MainGate::<C::Scalar>::new(config.main_gate_config.clone());
      let aux_generator = <C as CurveAffine>::CurveExt::random(OsRng).to_affine();
      layouter.assign_region(
          || "assign aux values",
          |region| {
              let offset = 0;
              let ctx = &mut RegionCtx::new(region, offset);
              //println!("ONE");
              ecc_chip.assign_aux_generator(ctx, Value::known(aux_generator))?;
              ecc_chip.assign_aux(ctx, 2, 1)?;
              ecc_chip.get_mul_aux(2, 1)?;
              Ok(())
          },
      )?;
      layouter.assign_region(
          || "region 0",
          |region| {
              let offset = 0;
              let ctx = &mut RegionCtx::new(region, offset);

              let base = C::CurveExt::random(OsRng);
              let s = C::Scalar::random(OsRng);
              let result = base * s;

              let base = ecc_chip.assign_point(ctx, Value::known(base.into()))?;
              let s = main_gate.assign_value(ctx, Value::known(s))?;
              let result_0 = ecc_chip.assign_point(ctx, Value::known(result.into()))?;

              let result_1 = ecc_chip.mul(ctx, &base, &s, 2)?;
              ecc_chip.assert_equal(ctx, &result_0, &result_1)?;

              Ok(())
          },
      )?;
      config.config_range(&mut layouter)?;
    }

    // Perform the dag
    let dag_chip = DAGLayerChip::<C::ScalarExt>::construct(self.dag_config.clone());
    let (final_tensor_map, result) = dag_chip.forward(
      layouter.namespace(|| "dag"),
      &tensors,
      &constants,
      config.gadget_config.clone(),
      &LayerConfig::default(),
    )?;

    if self.commit_after.len() > 0 {
      for commit_idxes in self.commit_after.iter() {
        let to_commit = BTreeMap::from_iter(commit_idxes.iter().map(|idx| {
          (
            *idx,
            final_tensor_map.get(&(*idx as usize)).unwrap().clone(),
          )
        }));
        let commitment = self.copy_and_commit(
          layouter.namespace(|| "commit"),
          &constants,
          &config,
          &to_commit,
        );
        commitments.push(commitment);
      }
    }

    let mut pub_layouter = layouter.namespace(|| "public");
    let mut total_idx = 0;
    if config.gadget_config.poly_commit {
      total_idx += 1;
    }
    // if config.gadget_config.poly_commit {
    //   for poly_res in rho {
    //     pub_layouter
    //     .constrain_instance(poly_res.cell(), config.public_col, total_idx)
    //     .unwrap();
    //     let val = convert_to_bigint(poly_res.value().map(|x| x.to_owned()));
    //     new_public_vals.push(val);
    //     total_idx += 1;
    //   }
    //   println!("Poly vals len: {}", poly_coeffs.len());
    // }

    for cell in commitments.iter() {
      pub_layouter
        .constrain_instance(cell.as_ref().cell(), config.public_col, total_idx)
        .unwrap();
      let val = convert_to_bigint(cell.value().map(|x| x.to_owned()));
      new_public_vals.push(val);
      total_idx += 1;
    }
    let curr_idx = total_idx;
    for tensor in result {
      for cell in tensor.iter() {
        pub_layouter
          .constrain_instance(cell.as_ref().cell(), config.public_col, total_idx)
          .unwrap();
        let val = convert_to_bigint(cell.value().map(|x| x.to_owned()));
        new_public_vals.push(val);
        total_idx += 1;
      }
    }
   //println!("Res size: {}", total_idx - curr_idx);
    *PUBLIC_VALS.lock().unwrap() = new_public_vals;
    println!("Synthesis time: {:?}", timer.elapsed());
    Ok(())
  }
}
