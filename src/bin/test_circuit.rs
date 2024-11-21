use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr, halo2curves::bn256::G1Affine};
use halo2curves::bn256::Bn256;
use zkml::{
  model::ModelCircuit,
  utils::{
    helpers::get_public_values,
    loader::{load_model_msgpack, ModelMsgpack},
  },
};

fn main() {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let inp_fname = std::env::args().nth(2).expect("input file path");
  let witness_column = std::env::args().nth(3).expect("witness_col").parse().unwrap();
  let poly_chunks: usize = std::env::args().nth(3).expect("poly chunks").parse().unwrap();
  let config: ModelMsgpack = load_model_msgpack(&config_fname, &inp_fname, witness_column);
  let circuit = ModelCircuit::<G1Affine>::generate_from_file(&config_fname, &inp_fname, witness_column, 0, 17, 10, false);

  let _prover = MockProver::run(config.k.try_into().unwrap(), &circuit, vec![vec![]]).unwrap();
  let public_vals = get_public_values();

  let prover = MockProver::run(config.k.try_into().unwrap(), &circuit, vec![public_vals]).unwrap();
  assert_eq!(prover.verify(), Ok(()));
  println!("Proof verified!")
}
