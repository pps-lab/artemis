use std::{
  fs::File,
  io::{BufReader, Write},
  path::Path,
  time::Instant,
};

use halo2_proofs::{
  arithmetic::eval_polynomial, circuit, dev::MockProver, halo2curves::pasta::{EqAffine, Fp}, plonk::{create_proof, keygen_pk, keygen_vk, verify_proof}, poly::{
    commitment::{Blind, Params, ParamsProver, Prover, MSM}, ipa::{
      self, commitment::{IPACommitmentScheme, ParamsIPA}, msm::MSMIPA, multiopen::ProverIPA, strategy::SingleStrategy
    }, Coeff, Polynomial, VerificationStrategy
  }, transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, Transcript, TranscriptRead, TranscriptReadBuffer, TranscriptWrite, TranscriptWriterBuffer
  }
};
use halo2curves::{bn256, pasta::EpAffine};
use ff::Field;
use crate::{model::ModelCircuit, utils::helpers::get_public_values};

pub fn get_ipa_params(params_dir: &str, degree: u32) -> ParamsIPA<EqAffine> {
  let path = format!("{}/{}.params", params_dir, degree);
  let params_path = Path::new(&path);
  if File::open(&params_path).is_err() {
    let params: ParamsIPA<EqAffine> = ParamsIPA::new(degree);
    let mut buf = Vec::new();

    params.write(&mut buf).expect("Failed to write params");
    let mut file = File::create(&params_path).expect("Failed to create params file");
    file
      .write_all(&buf[..])
      .expect("Failed to write params to file");
  }

  let params_fs = File::open(&params_path).expect("couldn't load params");
  let params: ParamsIPA<EqAffine> =
    Params::read::<_>(&mut BufReader::new(params_fs)).expect("Failed to read params");
  params
}

pub fn time_circuit_ipa(circuit: ModelCircuit<Fp>, commit_poly: bool, poly_col_len: usize) {
  let mut rng = &mut rand::thread_rng();
  let start = Instant::now();

  let degree = circuit.k as u32;
  let mut circuit = circuit.clone();
  let empty_circuit = circuit.clone();
  let mut proof_circuit = circuit.clone();

  let params = get_ipa_params("./params_ipa", degree);

  let circuit_duration = start.elapsed();
  println!(
    "Time elapsed in params construction: {:?}",
    circuit_duration
  );

  let vk = keygen_vk(&params, &empty_circuit).unwrap();
  let vk_duration = start.elapsed();
  println!(
    "Time elapsed in generating vkey: {:?}",
    vk_duration - circuit_duration
  );

  let pk = keygen_pk(&params, vk, &empty_circuit).unwrap();
  let pk_duration = start.elapsed();
  println!(
    "Time elapsed in generating pkey: {:?}",
    pk_duration - vk_duration
  );
  drop(empty_circuit);

  let fill_duration = start.elapsed();
  let _prover = MockProver::run(degree, &proof_circuit, vec![vec![];  poly_col_len + 1]).unwrap();
  println!(
    "Time elapsed in filling circuit: {:?}",
    fill_duration - pk_duration
  );

  // IPA Commit proof
  let ipa_proof_timer = Instant::now();
  //let poly_coeff = vec![Fp::one(); 1 << params.k()];
  let mut tensor_len = 0usize;
  let mut poly_coeff = vec![];
  for (tensor_idx, tensor) in circuit.tensors.clone() {
    for val in tensor.clone() {
      tensor_len += 1;
      poly_coeff.push(val);
    }
    //println!("Tensor: {:?}, idx: {}", tensor, tensor_idx);
  }

  let mut transcript =
      Blake2bWrite::<Vec<u8>, EqAffine, Challenge255<EqAffine>>::init(vec![]);
  let beta = transcript.squeeze_challenge_scalar::<()>();
  let mut beta_pows = (0..poly_coeff.len()).map(|i| beta.pow([i as u64])).collect::<Vec<_>>();
  poly_coeff.extend(vec![Fp::ZERO; 2usize.pow(circuit.k as u32) - poly_coeff.len()]);

  println!("Poly coeff len: {}", poly_coeff.len());
  let poly: Polynomial<Fp, Coeff> = Polynomial::from_coefficients_vec(poly_coeff.clone());
  let blind = Blind::default();
  let poly_com: EqAffine = params.commit(&poly, blind).into();
  transcript.write_point(poly_com).unwrap();
  if commit_poly {
      while beta_pows.len() % poly_col_len != 0 {
      //inputs.push(&zero);
      beta_pows.push(Fp::ZERO);
    }
  }

  proof_circuit.beta_pows = beta_pows.clone();
  //println!("Circ beta pows: {:?}", circuit.beta_pows);
  // Evaluate the polynomial
  let rho = eval_polynomial(&poly, *beta);
  transcript.write_scalar(rho).unwrap();

  let (proof, ch_prover) = {
      ipa::commitment::create_proof(&params, rng, &mut transcript, &poly, blind, *beta).unwrap();
      let ch_prover = transcript.squeeze_challenge();
      (transcript.finalize(), ch_prover)
  };
  println!("IPA commit proof time: {:?}", ipa_proof_timer.elapsed());
  // Verify the opening proof
  let ipa_vfy_timer = Instant::now();
  let mut transcript =
      Blake2bRead::<&[u8], EqAffine, Challenge255<EqAffine>>::init(&proof[..]);
  let beta_prime = transcript.squeeze_challenge_scalar::<()>();
  assert_eq!(*beta, *beta_prime);
  let p_prime = transcript.read_point().unwrap();
  assert_eq!(poly_com, p_prime);
  let rho_prime = transcript.read_scalar().unwrap();
  assert_eq!(rho, rho_prime);

  let mut commitment_msm = MSMIPA::new(&params);
  commitment_msm.append_term(Fp::one(), poly_com.into());

  let guard = ipa::commitment::verify_proof(&params, commitment_msm, &mut transcript, *beta, rho).unwrap();
  let ch_verifier = transcript.squeeze_challenge();
  assert_eq!(*ch_prover, *ch_verifier);
          // Test guard behavior prior to checking another proof
  {
    // Test use_challenges()
    let msm_challenges = guard.clone().use_challenges();
    assert!(msm_challenges.check());

    // Test use_g()
    let g = guard.compute_g();
    let (msm_g, _accumulator) = guard.clone().use_g(g);
    assert!(msm_g.check());
  }
  println!("IPA commit vfy time: {:?}", ipa_vfy_timer.elapsed());

  let mut public_vals = vec![vec![]; poly_col_len + 1];
  public_vals[poly_col_len] = get_public_values();
  //let mut betas = vec![vec![]; poly_col_len];
  if commit_poly {
    for i in 0..poly_col_len {
      for j in 0..beta_pows.len() / poly_col_len {
        public_vals[i].push(beta_pows[i + j * poly_col_len] )
      }
    }
    let mut pub_val_idx = 0;
    // for beta in beta_pows {
    //   public_vals[pub_val_idx] = beta;
    //   pub_val_idx += 1;
    // }
    public_vals[poly_col_len][pub_val_idx] = rho;
  }
  let public_vals_slice = public_vals.iter().map(|x| x.as_slice()).collect::<Vec<_>>();

  let proof_duration_start = start.elapsed();
  let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
  create_proof::<IPACommitmentScheme<EqAffine>, ProverIPA<EqAffine>, _, _, _, _>(
    &params,
    &pk,
    &[proof_circuit],
    &[&public_vals_slice.as_slice()],
    &mut rand::thread_rng(),
    &mut transcript,
  )
  .unwrap();
  let proof = transcript.finalize();
  let proof_duration = start.elapsed();
  println!("Proving time: {:?}", proof_duration - proof_duration_start);

  let proof_size = {
    let mut folder = std::path::PathBuf::new();
    folder.push("proof");
    let mut fd = std::fs::File::create(folder.as_path()).unwrap();
    folder.pop();
    fd.write_all(&proof).unwrap();
    fd.metadata().unwrap().len()
  };
  println!("Proof size: {} bytes", proof_size);

  let strategy = SingleStrategy::new(&params);
  let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
  assert!(
    verify_proof(
      &params,
      pk.get_vk(),
      strategy,
      &[public_vals_slice.as_slice()],
      &mut transcript
    )
    .is_ok(),
    "proof did not verify"
  );
  let verify_duration = start.elapsed();
  println!("Verifying time: {:?}", verify_duration - proof_duration);


}
