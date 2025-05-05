use halo2_proofs::halo2curves::{pasta::{Fp, EqAffine}};
use halo2curves::{bls12381::Bls12, bn256::Bn256, pairing::Engine};
use zkml::{
  model::ModelCircuit,
  utils::{proving_ipa::time_circuit_ipa, proving_kzg::time_circuit_kzg},
};
fn main() {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let inp_fname = std::env::args().nth(2).expect("input file path");
  let kzg_or_ipa = std::env::args().nth(3).expect("kzg or ipa");
  let commit = std::env::args().nth(4).expect("witness col").parse().unwrap();
  let chunks: usize = std::env::args().nth(5).expect("chunks for poly circuit").parse().unwrap();
  let k_ipt: usize = std::env::args().nth(6).expect("num of rows").parse().unwrap();
  let c_ipt: usize = std::env::args().nth(7).expect("num of columns").parse().unwrap();
  let cp_link = std::env::args().nth(8).expect("cplink col").parse().unwrap();
  let pedersen: bool = std::env::args().nth(9).expect("pedersen").parse().unwrap();
  let num_runs = std::env::args().nth(10).expect("number of verifier runs").parse().unwrap();
  let directory = std::env::args().nth(11).expect("directory name").parse().unwrap();

  if kzg_or_ipa.get(..3) == Some("kzg") {
    if kzg_or_ipa.get(4..) == Some("bls") {
      let circuit = ModelCircuit::<<Bls12 as Engine>::G1Affine>::generate_from_file(&config_fname, &inp_fname, commit, chunks, k_ipt, c_ipt, pedersen);
      time_circuit_kzg::<Bls12>(circuit, commit, pedersen, chunks, cp_link, num_runs, directory, c_ipt);
    } else {
      let circuit = ModelCircuit::<<Bn256 as Engine>::G1Affine>::generate_from_file(&config_fname, &inp_fname, commit, chunks, k_ipt, c_ipt, pedersen);
      time_circuit_kzg::<Bn256>(circuit, commit, pedersen,chunks, cp_link, num_runs, directory, c_ipt);
    }
  } else {
    let circuit = ModelCircuit::<EqAffine>::generate_from_file(&config_fname, &inp_fname, commit, chunks, k_ipt, c_ipt, pedersen);
    //let k = circuit.k;
    time_circuit_ipa(circuit, commit, chunks, num_runs, directory);
  }
}
