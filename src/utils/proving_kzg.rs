use std::{
  fs::File,
  io::{BufReader, Write},
  path::Path,
  time::Instant,
};

use ff::Field;
use halo2_proofs::{
  arithmetic::{eval_polynomial, kate_division}, circuit, dev::MockProver, halo2curves::bn256::{Bn256, Fr, G1Affine}, plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Error, VerifyingKey}, poly::{commitment::{Blind, Params, ParamsProver, Prover, Verifier}, kzg::{commitment::{KZGCommitmentScheme, ParamsKZG}, msm::DualMSM, multiopen::{ProverSHPLONK, VerifierSHPLONK}, strategy::{AccumulatorStrategy, SingleStrategy}}, Coeff, Polynomial, ProverQuery, VerificationStrategy, VerifierQuery}, transcript::{
    self, Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge, Transcript, TranscriptRead, TranscriptReadBuffer, TranscriptWrite, TranscriptWriterBuffer
  }, SerdeFormat
};
use halo2curves::{bn256::pairing, pairing::Engine, group::Curve};
use rand_core::OsRng;

use crate::{model::ModelCircuit, utils::helpers::{get_public_values, poly_divmod}};

pub fn get_kzg_params(params_dir: &str, degree: u32) -> ParamsKZG<Bn256> {
  let rng = rand::thread_rng();
  let path = format!("{}/{}.params", params_dir, degree);
  let params_path = Path::new(&path);
  if File::open(&params_path).is_err() {
    let params = ParamsKZG::<Bn256>::setup(degree, rng);
    let mut buf = Vec::new();

    params.write(&mut buf).expect("Failed to write params");
    let mut file = File::create(&params_path).expect("Failed to create params file");
    file
      .write_all(&buf[..])
      .expect("Failed to write params to file");
  }

  let mut params_fs = File::open(&params_path).expect("couldn't load params");
  let params = ParamsKZG::<Bn256>::read(&mut params_fs).expect("Failed to read params");
  params
}

pub fn serialize(data: &Vec<u8>, path: &str) -> u64 {
  let mut file = File::create(path).unwrap();
  file.write_all(data).unwrap();
  file.metadata().unwrap().len()
}

pub fn verify_kzg(
  params: &ParamsKZG<Bn256>,
  vk: &VerifyingKey<G1Affine>,
  strategy: SingleStrategy<Bn256>,
  public_vals: &Vec<Fr>,
  mut transcript: Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
) {
  assert!(
    verify_proof::<
      KZGCommitmentScheme<Bn256>,
      VerifierSHPLONK<'_, Bn256>,
      Challenge255<G1Affine>,
      Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
      halo2_proofs::poly::kzg::strategy::SingleStrategy<'_, Bn256>,
    >(&params, &vk, strategy, &[&[&public_vals]], &mut transcript)
    .is_ok(),
    "proof did not verify"
  );
}

pub fn time_circuit_kzg(circuit: ModelCircuit<Fr>) {
  let rng = rand::thread_rng();
  let start = Instant::now();

  let degree = circuit.k as u32;
  let params = get_kzg_params("./params_kzg", degree);

  let circuit_duration = start.elapsed();
  println!(
    "Time elapsed in params construction: {:?}",
    circuit_duration
  );
  let mut circuit = circuit.clone();
  let beta = Fr::random(&mut OsRng);
  //let beta = Fr::ONE;
  let mut tensor_len = 0usize;
  let mut poly_coeff = vec![];
  for (tensor_idx, tensor) in circuit.tensors.clone() {
    for val in tensor.clone() {
      tensor_len += 1;
      poly_coeff.push(val);
    }
    //println!("Tensor: {:?}, idx: {}", tensor, tensor_idx);
  }
  println!("Poly coeff len: {}", poly_coeff.len());
  let beta_pows = (0..poly_coeff.len()).map(|i| beta.pow([i as u64])).collect::<Vec<_>>();
  circuit.beta_pows = beta_pows.clone();
  let poly: Polynomial<Fr, Coeff> = Polynomial::from_coefficients_vec(poly_coeff);
  let rho = poly.evaluate(beta);
  let blind = Blind::default();

  let vk_circuit = circuit.clone();
  let vk = keygen_vk(&params, &vk_circuit).unwrap();
  drop(vk_circuit);
  let vk_duration = start.elapsed();
  println!(
    "Time elapsed in generating vkey: {:?}",
    vk_duration - circuit_duration
  );

  let vkey_size = serialize(&vk.to_bytes(SerdeFormat::RawBytes), "vkey");
  println!("vkey size: {} bytes", vkey_size);

  let pk_circuit = circuit.clone();
  let pk = keygen_pk(&params, vk, &pk_circuit).unwrap();
  let pk_duration = start.elapsed();
  println!(
    "Time elapsed in generating pkey: {:?}",
    pk_duration - vk_duration
  );
  drop(pk_circuit);

  let pkey_size = serialize(&pk.to_bytes(SerdeFormat::RawBytes), "pkey");
  println!("pkey size: {} bytes", pkey_size);

  let fill_duration = start.elapsed();
  let proof_circuit = circuit.clone();
  let _prover = MockProver::run(degree, &proof_circuit, vec![vec![]]).unwrap();

  println!(
    "Time elapsed in filling circuit: {:?}",
    fill_duration - pk_duration
  );

  let mut public_vals = get_public_values();
  let mut pub_val_idx = 0;
  for beta in beta_pows {
    public_vals[pub_val_idx] = beta;
    pub_val_idx += 1;
  }
  public_vals[pub_val_idx] = rho + Fr::ONE;


  // Convert public vals to serializable format
  let public_vals_u8: Vec<u8> = public_vals
    .iter()
    .map(|v: &Fr| v.to_bytes().to_vec())
    .flatten()
    .collect();
  let public_vals_u8_size = serialize(&public_vals_u8, "public_vals");
  println!("Public vals size: {} bytes", public_vals_u8_size);

  let proof_duration_start = start.elapsed();
  let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
  create_proof::<
    KZGCommitmentScheme<Bn256>,
    ProverSHPLONK<'_, Bn256>,
    Challenge255<G1Affine>,
    _,
    Blake2bWrite<Vec<u8>, G1Affine, Challenge255<G1Affine>>,
    ModelCircuit<Fr>,
  >(
    &params,
    &pk,
    &[proof_circuit],
    &[&[&public_vals]],
    rng,
    &mut transcript,
  )
  .unwrap();
  let proof = transcript.finalize();
  let proof_duration = start.elapsed();
  println!("Proving time: {:?}", proof_duration - proof_duration_start);

  let proof_size = serialize(&proof, "proof");
  let proof = std::fs::read("proof").unwrap();

  println!("Proof size: {} bytes", proof_size);

  let strategy = SingleStrategy::new(&params);
  let transcript_read = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

  println!("public vals len: {:?}", public_vals.len());
  println!("Rho: {:?}", rho);
  verify_kzg(
    &params,
    &pk.get_vk(),
    strategy,
    &public_vals,
    transcript_read,
  );
  let verify_duration = start.elapsed();
  println!("Verifying time: {:?}", verify_duration - proof_duration);
  
  // KZG Commit proof
  let kzg_proof_timer = Instant::now();

  println!("Tensor len: {}", tensor_len);


  let mut transcript_kzg_proof = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);

  let poly_com: G1Affine = params.commit(&poly, blind).to_affine();

  let queries = [
    ProverQuery {
        point: beta,
        poly: &poly,
        blind,
    }
  ].to_vec();
  //let (q, r) = poly_divmod::<Fr>(&poly, &Polynomial::from_coefficients_vec(vec![-rho, Fr::one()]));
  //let (q, r) = (poly - &Polynomial::from_coefficients_vec(vec![rho])).divide_with_q_and_r(&Polynomial::from_coefficients_vec(vec![-beta, Fr::ONE])).unwrap();
  //assert!(r.is_zero());
  let prover = ProverSHPLONK::new(&params);
  prover
      .create_proof(&mut OsRng, &mut transcript_kzg_proof, queries)
      .unwrap();

  let proof_kzg = transcript_kzg_proof.finalize();

  //let pi = params.commit(&q, blind);
  println!("KZG proof time: {:?}", kzg_proof_timer.elapsed());
  //KZG Commit vfy
  let kzg_vfy_timer = Instant::now();
  // let lhs = pairing(&(poly_com - (params.get_g()[0] * rho)).into(), &params.g2());
  // let rhs = pairing(&pi.into(), &(&params.s_g2() - &params.g2() * beta).into());
  // assert_eq!(lhs, rhs);
  let verifier_params = params.verifier_params();
  let verifier = VerifierSHPLONK::new(&verifier_params);
  let mut transcript_kzg_verify = Blake2bRead::<_, _, Challenge255<_>>::init(proof_kzg.as_slice());

  let queries = std::iter::empty()
      .chain(Some(VerifierQuery::new_commitment(&poly_com, beta, rho)));

  let msm = DualMSM::new(&params);
  assert!(verifier.verify_proof(&mut transcript_kzg_verify, queries, msm).is_ok());

  println!("KZG vfy time: {:?}", kzg_vfy_timer.elapsed());
}

// Standalone verification
pub fn verify_circuit_kzg(
  circuit: ModelCircuit<Fr>,
  vkey_fname: &str,
  proof_fname: &str,
  public_vals_fname: &str,
) {
  let degree = circuit.k as u32;
  let params = get_kzg_params("./params_kzg", degree);
  println!("Loaded the parameters");

  let vk = VerifyingKey::read::<BufReader<File>, ModelCircuit<Fr>>(
    &mut BufReader::new(File::open(vkey_fname).unwrap()),
    SerdeFormat::RawBytes,
    (),
  )
  .unwrap();
  println!("Loaded vkey");

  let proof = std::fs::read(proof_fname).unwrap();

  let public_vals_u8 = std::fs::read(&public_vals_fname).unwrap();
  let public_vals: Vec<Fr> = public_vals_u8
    .chunks(32)
    .map(|chunk| Fr::from_bytes(chunk.try_into().expect("conversion failed")).unwrap())
    .collect();

  let strategy = SingleStrategy::new(&params);
  let transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);

  let start = Instant::now();
  let verify_start = start.elapsed();
  verify_kzg(&params, &vk, strategy, &public_vals, transcript);
  let verify_duration = start.elapsed();
  println!("Verifying time: {:?}", verify_duration - verify_start);
  println!("Proof verified!")
}
