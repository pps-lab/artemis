use halo2_proofs::{dev::MockProver, halo2curves::bn256::Fr, poly::{self, commitment::Params}};
use halo2curves::bn256::Bn256;
use zkml::{
  model::ModelCircuit,
  utils::{
    helpers::get_public_values, loader::{load_model_msgpack, ModelMsgpack}, proving_ipa::get_ipa_params, proving_kzg::get_kzg_params
  },
};

fn main() {
  let num_params: u32 = std::env::args().nth(1).expect("num of params").parse().unwrap();
  let kzg_or_ipa = std::env::args().nth(2).expect("kzg or ipa");
  if kzg_or_ipa == "kzg" {
    let poly_params = get_kzg_params::<Bn256>("params_kzg", num_params as u32);
    println!("Params len: {}", poly_params.k());
  } else {
    let poly_params = get_ipa_params("params_ipa", num_params as u32);
    println!("Params len: {}", poly_params.k());
  }
}
