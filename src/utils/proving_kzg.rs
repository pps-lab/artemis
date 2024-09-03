use core::num;
use std::{
  fs::File, io::{BufReader, Write}, path, path::Path, time::{Duration, Instant}
};
use crate::utils::helpers::{cplink1, cplink1_lite, cplink2, powers, setup, vanishing_on_set, verify1, verify1_lite};
use bitvec::order::verify;
use ff::{Field, WithSmallOrderMulGroup};
use group::cofactor::CofactorCurveAffine;
use halo2_proofs::{
  arithmetic::{eval_polynomial, kate_division}, circuit, dev::MockProver, halo2curves::bn256::{Bn256, Fr, G1Affine}, plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Error, VerifyingKey}, poly::{self, commitment::{Blind, Params, ParamsProver, Prover, Verifier}, kzg::{commitment::{KZGCommitmentScheme, ParamsKZG}, msm::DualMSM, multiopen::{ProverSHPLONK, VerifierSHPLONK}, strategy::{AccumulatorStrategy, SingleStrategy}}, Coeff, EvaluationDomain, Polynomial, ProverQuery, VerificationStrategy, VerifierQuery}, transcript::{
    self, Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge, Transcript, TranscriptRead, TranscriptReadBuffer, TranscriptWrite, TranscriptWriterBuffer
  }, SerdeFormat
};
use halo2curves::{bn256::pairing, pairing::Engine, group::Curve};
use rand::thread_rng;
use rand_core::OsRng;

use csv::Writer;
use rmp_serde::config;
use crate::{model::{ModelCircuit, GADGET_CONFIG}, utils::helpers::{get_public_values, poly_divmod}};

pub fn get_kzg_params(params_dir: &str, degree: u32) -> ParamsKZG<Bn256> {
  let rng = rand::thread_rng();
  let path = format!("{}/{}.params", params_dir, degree);
  println!("Path: {}", path);
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
  public_vals: &[&[Fr]],
  mut transcript: Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
) {
  assert!(
    verify_proof::<
      KZGCommitmentScheme<Bn256>,
      VerifierSHPLONK<'_, Bn256>,
      Challenge255<G1Affine>,
      Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
      halo2_proofs::poly::kzg::strategy::SingleStrategy<'_, Bn256>,
    >(&params, &vk, strategy, &[&public_vals], &mut transcript)
    .is_ok(),
    "proof did not verify"
  );
}

pub fn time_circuit_kzg(circuit: ModelCircuit<Fr>, commit_poly: bool, poly_col_len: usize, cp_link: bool, num_runs: usize, directory: String, c: usize) {
  let rng = rand::thread_rng();
  //println!("Num of total columns: {}, advice: {}, instance: {}, fixed: {}", total_columns, cs.num_advice_columns, cs.num_instance_columns, cs.num_fixed_columns);
  let start = Instant::now();

  let degree = circuit.k as u32;
  let params = get_kzg_params(format!("{}/params_kzg", directory).as_str(), degree);

  let circuit_duration = start.elapsed();
  println!(
    "Time elapsed in params construction: {:?}",
    circuit_duration
  );
  let mut circuit = circuit.clone();

  //let alpha = Fr::random(rand::thread_rng());
  let alpha = Fr::ONE;
  let beta = Fr::random(rand::thread_rng());
  //let beta = Fr::from(2);
  //let beta = Fr::one();

  let mut tensor_len = 0usize;
  let mut poly_coeff = Vec::with_capacity(2usize.pow((circuit.k + poly_col_len) as u32));
  for (tensor_idx, tensor) in circuit.tensors.clone() {
    for val in tensor.clone() {
      poly_coeff.push(val);
      if tensor_len == 0 {
        println!("Value: {:?}", val);
      }
      tensor_len += 1;
    }
    //println!("Tensor: {:?}, idx: {}", tensor, tensor_idx);
  }
  println!("Poly coeff 0: {:?}", poly_coeff[0]);
  let poly: Polynomial<Fr, Coeff> = Polynomial::from_coefficients_vec(poly_coeff.clone());
  let mut polys = vec![];
  if poly_col_len > 0 {
    polys = (0..poly_col_len).map(|x| {
      let poly: Polynomial<Fr, Coeff> = Polynomial::from_coefficients_vec(poly_coeff[(x * poly_coeff.len() / poly_col_len)..((x + 1) * poly_coeff.len() / poly_col_len)].to_vec()) * alpha.pow([x as u64]);
      poly
    }).collect::<Vec<_>>();
  }

  if commit_poly {
    let beta_pows = (0..poly_col_len + 1).map(|i| beta.pow([i as u64])).rev().collect::<Vec<_>>();

    // while beta_pows.len() % poly_col_len != 0 {
    //   //inputs.push(&zero);
    //   beta_pows.push(Fr::ZERO);
    // }
    while poly_coeff.len() % poly_col_len != 0  {
      poly_coeff.push(Fr::ZERO);
    }
  
    circuit.beta_pows = beta_pows.clone();
  }

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

  let pk_circuit = circuit.clone();
  let pk = keygen_pk(&params, vk, &pk_circuit).unwrap();
  let pk_duration = start.elapsed();
  println!(
    "Time elapsed in generating pkey: {:?}",
    pk_duration - vk_duration
  );
  drop(pk_circuit);

  let fill_duration = start.elapsed();
  let proof_circuit = circuit.clone();

  let _prover = MockProver::run(degree, &proof_circuit, vec![vec![]]).unwrap();
  println!(
    "Time elapsed in filling circuit: {:?}",
    fill_duration - pk_duration
  );
  let mut public_vals = vec![vec![]];
  //let mut betas = vec![vec![]; poly_col_len];

  public_vals[0] = get_public_values();
  if commit_poly {
    public_vals[0][0] = rho;
  }

  let public_vals_slice = public_vals.iter().map(|x| x.as_slice()).collect::<Vec<_>>();

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
    &[public_vals_slice.as_slice()],
    rng,
    &mut transcript,
  )
  .unwrap();
  let proof = transcript.finalize();
  let proof_duration = start.elapsed();
  let mut proving_time = proof_duration - proof_duration_start;
  //println!("Proving time: {:?}", proof_duration - proof_duration_start);
  let mut proof_size = proof.len();
  println!("Proof size: {}", proof.len());
  let strategy = SingleStrategy::new(&params);
  let transcript_read = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
  let mut verifying_time = vec![];
  for i in 0..num_runs {
    let vfy_timer = Instant::now();
    println!("public vals len: {:?}", public_vals.iter().map(|vec| vec.len()).fold(0, |a, b| a + b));
    println!("Rho: {:?}", rho);
    verify_kzg(
      &params,
      &pk.get_vk(),
      strategy.clone(),
      &public_vals_slice,
      transcript_read.clone(),
    );
    let verify_duration = vfy_timer.elapsed();
    //println!("Verifying time: {:?}", verify_duration - proof_duration);
    verifying_time.push(verify_duration);
  }

  //CPLink proof: 
  if cp_link {
    if poly_col_len < 1  {
      // slow
      let col_size = circuit.k;
      let witness_size =  poly_coeff.len();
      let l = c;
      let size = witness_size / l;
      let (HH, thetas, zs, z_v, z_last, zhats, z_coms, zhat_coms) = setup(col_size as u32, witness_size, l, &params);
      let vals = (0..l).map(|y| poly_coeff[y*size..(y+1)*size].to_vec()).collect::<Vec<_>>();
      let coeffs = vals.iter().map(|x| HH.lagrange_from_vec(x.clone())).collect::<Vec<_>>();
      let us = coeffs.iter().map(|x| HH.lagrange_to_coeff(x.clone())).collect::<Vec<_>>();
      let u_coms = us.iter().map(|u| params.commit_g1(u)).collect();
      let z_v_com = params.commit_g1(&z_v);
      let poly = HH.lagrange_to_coeff(HH.lagrange_from_vec(poly_coeff));
      let (uprimes, cp2_prove, cp2_vfy) = cplink2(thetas, HH.clone(), us, z_v, z_v_com, u_coms, params.clone());
      let (chats, ds, cprimes, wcom, bigc, d, x, zz, cp1_prove) = cplink1(uprimes, zs, zhats, poly, params.clone(), z_last, HH);
      proving_time += cp1_prove + cp2_prove;
      for i in 0..num_runs {
        let cp1_vfy = verify1(&cprimes, &chats, &ds, &params,&z_coms, &zhat_coms, wcom, bigc, d, x, zz);
        let vfy_total = cp1_vfy + cp2_vfy;
        verifying_time[i] += vfy_total;
      }
    } else {
      //fast
      let domain = EvaluationDomain::<Fr>::new(1, params.k());
      let domain_vals = (0..poly_coeff.len() / poly_col_len).map(|i| domain.get_omega().pow([i as u64])).collect::<Vec<_>>();
      // let domain_vals = (0..poly_col_len).map(|i| domain_vals[(i * poly_coeff.len() / poly_col_len)..((i + 1) * poly_coeff.len() / poly_col_len)].to_vec()).collect::<Vec<_>>() ;
      let z = vanishing_on_set(&domain_vals);
      //let z = Polynomial::from_coefficients_vec(vec![Fr::one()]);
      let z_com = params.commit_g2(&z);
      //println!("poly sum: {:?}", poly_sum);
      //println!("z: {:?}", z);
      for poly in &polys {
        //let (q, mut uhat) = poly_divmod(poly, &z);   
        let uhat = poly.clone();
        let chat = params.commit_g1(&uhat);
        let polycom = params.commit_g1(&poly);
        println!("Three");
        let (chat, d_small, cprime, wcom, bigc, d, x, zz, cplink_time) = cplink1_lite(&poly, &chat, &polycom, &z, &poly, &params, &domain);
        proving_time += cplink_time;
        for i in 0..num_runs {
          verifying_time[i] += verify1_lite(cprime, chat, d_small, params.clone(), z_com, wcom, bigc, d, x, zz);
        }
      }
    }
  }

  // KZG Commit proof

  if commit_poly {
    let mut poly_params = params;
    if (poly_col_len) > 1  {
      poly_params = get_kzg_params(format!("{}/params_kzg", directory).as_str(), degree + (poly_col_len - 1).ilog2() + 1 as u32);
    }

    let kzg_proof_timer = Instant::now();
    println!("Tensor len: {}", tensor_len);
  
    let mut transcript_kzg_proof = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
    let poly_com = poly_params.commit(&poly, blind).to_affine();
    // let poly_coms: Vec<G1Affine> = polys.iter().map(|poly| params.commit(poly, blind).to_affine()).collect::<Vec<_>>();
    // let poly_com_sum = poly_coms.iter().fold(G1Affine::identity(), |a, b| (a + b).into());
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
    let prover = ProverSHPLONK::new(&poly_params);
    prover
        .create_proof(&mut OsRng, &mut transcript_kzg_proof, queries)
        .unwrap();
  
    let proof_kzg = transcript_kzg_proof.finalize();
    proof_size += proof_kzg.len();
    //let pi = params.commit(&q, blind);
    println!("KZG proof time: {:?}", kzg_proof_timer.elapsed());
    proving_time += kzg_proof_timer.elapsed();
    //KZG Commit vfy
    for i in 0..num_runs {
      let kzg_vfy_timer = Instant::now();
      // let lhs = pairing(&(poly_com - (params.get_g()[0] * rho)).into(), &params.g2());
      // let rhs = pairing(&pi.into(), &(&params.s_g2() - &params.g2() * beta).into());
      // assert_eq!(lhs, rhs);
      let verifier_params = poly_params.verifier_params();
      let verifier = VerifierSHPLONK::new(&verifier_params);
      let mut transcript_kzg_verify = Blake2bRead::<_, _, Challenge255<_>>::init(proof_kzg.as_slice());
    
      let queries = std::iter::empty()
          .chain(Some(VerifierQuery::new_commitment(&poly_com, beta, rho)));
    
      let msm = DualMSM::new(&poly_params);
      assert!(verifier.verify_proof(&mut transcript_kzg_verify, queries, msm).is_ok());
      let vfy_time = kzg_vfy_timer.elapsed();
      verifying_time[i] += vfy_time;
      println!("KZG vfy time: {:?}", vfy_time);
    }
  }

  println!("Proving time: {:?}", proving_time);
  println!("Verifying time: {:?}", verifying_time);

  // Create a file to write the CSV to
  let file = File::create("results/output.csv").unwrap();

  // Create a CSV writer
  let mut wtr = Writer::from_writer(file);

  // Write the header row
  wtr.write_record(&["Prover time", "Verifier time", "Proof size"]).unwrap();
  // Write some rows of data
  let proving_time_str = format!("{:?}", proving_time);
  let verifying_time_str = format!("{:?}", verifying_time);
  wtr.write_record(&[proving_time_str, verifying_time_str, proof_size.to_string()]).unwrap();
  // Flush and finish writing
  wtr.flush().unwrap();

  println!("CSV file created successfully.");
}

// Standalone verification
pub fn verify_circuit_kzg(
  circuit: ModelCircuit<Fr>,
  vkey_fname: &str,
  proof_fname: &str,
  public_vals_fname: &str,
) {
  let degree = circuit.k as u32;
  let params = get_kzg_params("~/params_kzg", degree);
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
  verify_kzg(&params, &vk, strategy, &[public_vals.as_slice()], transcript);
  let verify_duration = start.elapsed();
  println!("Verifying time: {:?}", verify_duration - verify_start);
  println!("Proof verified!")
}
