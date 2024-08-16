use halo2_proofs::halo2curves::{bn256::Fr, pasta::Fp};
use zkml::{
  model::ModelCircuit,
  utils::{proving_ipa::time_circuit_ipa, proving_kzg::time_circuit_kzg},
};
use graphviz_rust::*;
use std::fs;
fn main() {
  let config_fname = std::env::args().nth(1).expect("config file path");
  let inp_fname = std::env::args().nth(2).expect("input file path");
  let kzg_or_ipa = std::env::args().nth(3).expect("kzg or ipa");
  let witness_column_str = std::env::args().nth(4).expect("witness col");
  let fname = std::env::args().nth(6).expect("graphic file name");
  let ell: usize = std::env::args().nth(5).expect("num of cols").parse().unwrap();
  let k_ipt: usize = std::env::args().nth(6).expect("num of rows").parse().unwrap();
  println!("{}", witness_column_str);
  let witness_column = witness_column_str.parse().unwrap();

  println!("Poly ell: {:?}", ell);

  if kzg_or_ipa != "kzg" && kzg_or_ipa != "ipa" {
    panic!("Must specify kzg or ipa");
  }

  if kzg_or_ipa == "kzg" {
    let circuit = ModelCircuit::<Fr>::generate_from_file(&config_fname, &inp_fname, witness_column, ell, k_ipt);
    let k = circuit.k;
    println!("K: {}", k);
    let dot_string = halo2_proofs::dev::circuit_dot_graph(&circuit);
    use plotters::prelude::*;
    let root = BitMapBackend::new(&fname, (1000, 3000)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let root = root
        .titled("Example Circuit Layout", ("sans-serif", 60))
        .unwrap();

    halo2_proofs::dev::CircuitLayout::default().render(k as u32, &circuit, &root).unwrap();
    time_circuit_kzg(circuit);
  } else {
    let circuit = ModelCircuit::<Fp>::generate_from_file(&config_fname, &inp_fname, false, 0, 17);
    time_circuit_ipa(circuit);
  }
}
