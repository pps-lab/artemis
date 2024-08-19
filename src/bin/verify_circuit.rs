use halo2_proofs::halo2curves::bn256::Fr;
use zkml::{
  model::ModelCircuit,
  utils::{loader::load_config_msgpack, proving_kzg::verify_circuit_kzg},
};

fn main() {

  let config_fname = std::env::args().nth(1).expect("config file path");
  let vkey_fname = std::env::args().nth(2).expect("verification key file path");
  let proof_fname = std::env::args().nth(3).expect("proof file path");
  let public_vals_fname = std::env::args().nth(4).expect("public values file path");
  let kzg_or_ipa = std::env::args().nth(5).expect("kzg or ipa");
  let witness_column_str = std::env::args().nth(6).expect("witness col").parse().unwrap();
  let fname = std::env::args().nth(7).expect("graphic file name");
  println!("HELLOOO: {}", config_fname);
  if kzg_or_ipa != "kzg" && kzg_or_ipa != "ipa" {
    panic!("Must specify kzg or ipa");
  }

  if kzg_or_ipa == "kzg" {
    let config = load_config_msgpack(&config_fname, witness_column_str);
    let circuit = ModelCircuit::<Fr>::generate_from_msgpack(config, false, witness_column_str, 0, 17, 10);
    let k = circuit.k;
    let dot_string = halo2_proofs::dev::circuit_dot_graph(&circuit);
    use plotters::prelude::*;
    let root = SVGBackend::new(&fname, (1000, 3000)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let root = root
        .titled("Example Circuit Layout", ("sans-serif", 60))
        .unwrap();

    halo2_proofs::dev::CircuitLayout::default().render(k as u32, &circuit, &root).unwrap();
    println!("Loaded configuration");
    verify_circuit_kzg(circuit, &vkey_fname, &proof_fname, &public_vals_fname);
  } else {
    // Serialization of the verification key doesn't seem to be supported for IPA
    panic!("Not implemented");
  }
}
