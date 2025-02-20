use core::num;
use std::{
  fmt::Debug, fs::File, io::{BufReader, Write}, path::{self, Path}, time::{Duration, Instant}
};
use crate::utils::helpers::{cplink1, cplink1_lite, cplink2, powers, setup, vanishing_on_set, verify1, verify1_lite};
use bitvec::{domain::Domain, order::verify};
use ff::{Field, FromUniformBytes, WithSmallOrderMulGroup};
use group::{Group, prime::PrimeCurveAffine};
use halo2_proofs::{
  arithmetic::{eval_polynomial, kate_division}, circuit, dev::MockProver, helpers::SerdeCurveAffine, plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Error, VerifyingKey}, poly::{self, commitment::{Blind, Params, ParamsProver, Prover, Verifier}, kzg::{commitment::{KZGCommitmentScheme, ParamsKZG}, msm::DualMSM, multiopen::{ProverSHPLONK, VerifierSHPLONK}, strategy::{AccumulatorStrategy, SingleStrategy}}, Coeff, EvaluationDomain, LagrangeCoeff, Polynomial, ProverQuery, Rotation, VerificationStrategy, VerifierQuery}, transcript::{
    self, Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge, Transcript, TranscriptRead, TranscriptReadBuffer, TranscriptWrite, TranscriptWriterBuffer
  }, SerdeFormat
};
use halo2curves::{bls12381::Bls12, bn256::pairing, group::Curve, pairing::{Engine, MultiMillerLoop}, serde::SerdeObject};
use rand::thread_rng;
use rand_core::OsRng;

use csv::Writer;
use rmp_serde::config;
use crate::{model::{ModelCircuit, GADGET_CONFIG}, utils::helpers::{get_public_values, poly_divmod}};
//use halo2curves::{bn256, ::{EpAffine, Eq}};

pub fn get_kzg_params<E: Engine<G1Affine: SerdeCurveAffine, G2Affine: SerdeCurveAffine> + Debug>(params_dir: &str, degree: u32) -> ParamsKZG<E> {
  let rng = rand::thread_rng();
  let curve_type = E::type_of();
  let path = format!("{}_{curve_type}/{}.params", params_dir, degree);
  println!("Path: {}", path);
  let params_path = Path::new(&path);
  //let params = ParamsKZG::<E>::setup(degree, rng);
  if File::open(&params_path).is_err() {
    let params = ParamsKZG::<E>::setup(degree, rng);
    let mut buf = Vec::new();

    params.write(&mut buf).expect("Failed to write params");
    let mut file = File::create(&params_path).expect("Failed to create params file");
    file
      .write_all(&buf[..])
      .expect("Failed to write params to file");
  }

  let mut params_fs = File::open(&params_path).expect("couldn't load params");
  let params = ParamsKZG::<E>::read(&mut params_fs).expect("Failed to read params");
  params
}

pub fn serialize(data: &Vec<u8>, path: &str) -> u64 {
  let mut file = File::create(path).unwrap();
  file.write_all(data).unwrap();
  file.metadata().unwrap().len()
}

pub fn verify_kzg<
    E: Engine<
      G1Affine: SerdeCurveAffine,
      G2Affine: SerdeCurveAffine,
      Scalar: FromUniformBytes<64> + Ord + WithSmallOrderMulGroup<3>,
    > + Debug + MultiMillerLoop
  >(
  params: &ParamsKZG<E>,
  vk: &VerifyingKey<E::G1Affine>,
  strategy: SingleStrategy<E>,
  public_vals: &[&[E::Scalar]],
  mut transcript: Blake2bRead<&[u8], E::G1Affine, Challenge255<E::G1Affine>>,
) {
  assert!(
    verify_proof::<
      KZGCommitmentScheme<E>,
      VerifierSHPLONK<'_, E>,
      Challenge255<E::G1Affine>,
      Blake2bRead<&[u8], E::G1Affine, Challenge255<E::G1Affine>>,
      halo2_proofs::poly::kzg::strategy::SingleStrategy<'_, E>,
    >(&params, &vk, strategy, &[&public_vals], &mut transcript)
    .is_ok(),
    "proof did not verify"
  );
}

pub fn time_circuit_kzg<
  E: Engine<
    G1Affine: SerdeCurveAffine,
    G2Affine: SerdeCurveAffine,
    Scalar: FromUniformBytes<64> + Ord + WithSmallOrderMulGroup<3>,
  > + Debug + MultiMillerLoop
>(circuit: ModelCircuit<E::G1Affine>, commit_poly: bool, poly_col_len: usize, cp_link: bool, num_runs: usize, directory: String, c: usize) {
  println!("Type of Engine: {:?}", E::type_of());
  println!("Usize size: {:?}", std::mem::size_of::<usize>());

  let sigma = true;
  let rng = rand::thread_rng();
  //println!("Num of total columns: {}, advice: {}, instance: {}, fixed: {}", total_columns, cs.num_advice_columns, cs.num_instance_columns, cs.num_fixed_columns);
  let start = Instant::now();

  let degree = circuit.k as u32;
  let params = get_kzg_params(format!("{}/params_kzg", directory).as_str(), degree);
  let mut poly_params = params.clone();
  if commit_poly {
    if (poly_col_len) > 1  {
      poly_params = get_kzg_params(format!("{}/params_kzg", directory).as_str(), degree + (poly_col_len - 1).ilog2() + 1 as u32);
    }
  }
  let circuit_duration = start.elapsed();
  println!(
    "Time elapsed in params construction: {:?}",
    circuit_duration
  );
  let mut circuit = circuit.clone();

  let alpha = E::Scalar::ONE;
  let beta = E::Scalar::random(rand::thread_rng());

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
  let poly_coeff_len = poly_coeff.len();
  let poly: Polynomial<E::Scalar, Coeff> = Polynomial::from_coefficients_vec(poly_coeff.clone());
  let mut polys = vec![];
  if poly_col_len > 0 {
    polys = (0..poly_col_len).map(|x| {
      let poly: Polynomial<E::Scalar, Coeff> = Polynomial::from_coefficients_vec(poly_coeff[(x * poly_coeff.len() / poly_col_len)..((x + 1) * poly_coeff.len() / poly_col_len)].to_vec()) * alpha.pow([x as u64]);
      poly
    }).collect::<Vec<_>>();
  }

  if commit_poly {
    let beta_pows = (0..poly_col_len + 1).map(|i| beta.pow([i as u64])).rev().collect::<Vec<_>>();

    // while beta_pows.len() % poly_col_len != 0 {
    //   //inputs.push(&zero);
    //   beta_pows.push(E::Scalar::ZERO);
    // }
    while poly_coeff.len() % poly_col_len != 0  {
      poly_coeff.push(E::Scalar::ZERO);
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
  let mut public_vals = vec![vec![]];
  // if commit_poly {
  //   public_vals.push(vec![]);
  // }
  let _prover = MockProver::run(degree, &proof_circuit, public_vals.clone()).unwrap();
  println!(
    "Time elapsed in filling circuit: {:?}",
    fill_duration - pk_duration
  );

  //let mut betas = vec![vec![]; poly_col_len];

  public_vals[0] = get_public_values();
  // if commit_poly {
  //   public_vals[0][0] = rho;
  // }

  let public_vals_slice = public_vals.iter().map(|x| x.as_slice()).collect::<Vec<_>>();

  let proof_duration_start = start.elapsed();
  let mut transcript = Blake2bWrite::<_, E::G1Affine, Challenge255<_>>::init(vec![]);
  // Extract advice polys from proof generation
  let mut advice_lagrange: Vec<Polynomial<E::Scalar, LagrangeCoeff>> = vec![];
  let mut advice_blind: Vec<E::Scalar> = vec![];
  create_proof::<
    KZGCommitmentScheme<E>,
    ProverSHPLONK<'_, E>,
    Challenge255<E::G1Affine>,
    _,
    Blake2bWrite<Vec<u8>, E::G1Affine, Challenge255<E::G1Affine>>,
    ModelCircuit<E::G1Affine>,
  >(
    &params,
    &pk,
    &[proof_circuit.clone()],
    &[public_vals_slice.as_slice()],
    rng.clone(),
    &mut transcript,
    &mut advice_lagrange,
    &mut advice_blind,
  )
  .unwrap();

  let proof = transcript.finalize();
  let proof_duration = start.elapsed();
  let mut proving_time = proof_duration - proof_duration_start;
  println!("Proving time: {:?}", proving_time);
  let mut proof_size = proof.len();
  println!("Proof size: {}", proof.len());
  let strategy = SingleStrategy::new(&params);
  let transcript_read = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
  let mut proof_transcript = transcript_read.clone();
  let mut verifying_time = vec![];
  let public_valss: Vec<E::Scalar> = get_public_values();
  // if commit_poly {
  //   public_vals[0][0] = rho;
  // }
  for _ in 0..num_runs {
    let vfy_timer = Instant::now();
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

  let mut advice_com = vec![];
  for _ in 0..advice_lagrange.len() {
    advice_com.push(proof_transcript.read_point().unwrap().to_curve());
  }
  // /// test some commitment stuff
  // println!("Advice lagrange: {:?}", advice_lagrange.len());
  // println!("C: {:?}", c);
  // println!("Advice blind: {:?}", advice_blind);

  // let mut advice_mu = Polynomial::<E::Scalar, LagrangeCoeff>::from_coefficients_vec(vec![E::Scalar::ZERO]);//advice_lagrange[0].clone();
  // let blind1 = E::Scalar::ZERO;
  // println!("Advice 1 coeff 0: {:?}", E::Scalar::ZERO);
  // let mu_poly = Polynomial::<E::Scalar, LagrangeCoeff>::from_coefficients_vec(vec![E::Scalar::ZERO]);
  // let mu_commit = params.commit_lagrange(&mu_poly, Blind(E::Scalar::ZERO));
  // println!("Mu 1 poly: {:?}", mu_poly);
  // advice_mu[0] = E::Scalar::ZERO;
  // let no_mu_commit = params.commit_lagrange(&advice_mu, Blind(E::Scalar::ZERO));
  // let mut proof_transcript = transcript_read.clone();
  // let mut advice_com = vec![];
  // for _ in 0..advice_lagrange.len() {
  //   advice_com.push(proof_transcript.read_point().unwrap().to_curve());
  // }
  // //assert!(mu_commit + no_mu_commit == E::Scalar::ZERO);


  // let g_lagrange = E::G1Affine::identity();//params.g_lagrange();
  // let g_lag_mu = g_lagrange;//[0];

  // // external com
  // let mu = mu_poly[0];
  // let b = E::Scalar::random(rng.clone());
  // let G = E::G1::random(rng.clone());
  // let H = E::G1::random(rng.clone());
  // let C_hat = G * mu + H * b;

  // //internal com
  // let mu_circ = mu.clone();
  // let a = E::Scalar::ZERO;
  // let g = g_lag_mu.clone();
  // let h = E::G1::random(rng.clone());
  // let C = mu_commit;
  // println!("g : {:?}, mu_circ: {:?}", g, mu_circ);
  // //assert!(C == g * mu_circ, "C: {:?}, g * mu_circ: {:?}", C, g * mu_circ);
  // //Sigma protocol (https://eprint.iacr.org/2021/934.pdf Fig.2): 
  // if sigma {
  //   let timer = Instant::now();
  //   //1 
  //   let G_tilde  = G.clone(); 
  //   //2 
  //   let r = E::Scalar::random(rng.clone());
  //   let delta = E::Scalar::random(rng.clone());
  //   let gamma = E::Scalar::random(rng.clone());
  //   let A = g * r + h * delta; 
  //   let A_hat = G_tilde * r + H * gamma;
  //   //3 
  //   let e = E::Scalar::random(rng.clone());
  //   //4
  //   let z = r + e * mu; 
  //   let omega = delta + e * a;
  //   let Omega = gamma + e * b; 
  //   //5 
  //   let lhs1 = g * z + h * omega; // = g * (r + e * mu) + h * (delta)
  //   let rhs1 = A + C * e; // = g * (r + (mu * e)) + h * delta 
  //   assert!(lhs1 == rhs1);
    
  //   let lhs2 = G_tilde * z + H * Omega;
  //   let rhs2 = A_hat + C_hat * e;
  //   assert!(lhs2 == rhs2);

  // // Vanishing proof 
  //   let domain = EvaluationDomain::<E::Scalar>::new(1, params.k());
  //   let v_h_com = Polynomial::<E::Scalar, Coeff>::from_coefficients_vec(vec![-E::Scalar::ONE, E::Scalar::ONE]);
  //   println!("v_h Poly: {:?}, eval: {:?}", v_h_com, v_h_com.evaluate(E::Scalar::ZERO));
  //   let com_poly = Polynomial::<E::Scalar, LagrangeCoeff>::from_coefficients_vec(vec![mu]);
  //   let mid_poly = Polynomial::<E::Scalar, LagrangeCoeff>::from_coefficients_vec(vec![vec![E::Scalar::ZERO], vec![E::Scalar::ONE; circuit.k - 1]].concat());
  //   let com_poly_coeff = domain.lagrange_to_coeff(com_poly);
  //   let mid_poly_coeff = domain.lagrange_to_coeff(mid_poly);
  //   let (q, r) = mid_poly_coeff.divide_with_q_and_r(&v_h_com).unwrap();
  //   println!("R: {:?}", r);
  //   println!("Sigma proof time: {:?}", timer.elapsed());
  // }

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
      //let u_coms = us.iter().map(|u| params.commit_g1(u)).collect();
      let u_coms = vec![E::G1::identity(); c];
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
      let domain = EvaluationDomain::<E::Scalar>::new(1, params.k());
      let domain_vals = (0..poly_coeff.len() / poly_col_len).map(|i| domain.get_omega().pow([i as u64])).collect::<Vec<_>>();
      // let domain_vals = (0..poly_col_len).map(|i| domain_vals[(i * poly_coeff.len() / poly_col_len)..((i + 1) * poly_coeff.len() / poly_col_len)].to_vec()).collect::<Vec<_>>() ;
      let z = vanishing_on_set(&domain_vals);
      //let z = Polynomial::from_coefficients_vec(vec![E::Scalar::one()]);
      let z_com = params.commit_g2(&z);
      //println!("poly sum: {:?}", poly_sum);
      //println!("z: {:?}", z);
      for i in 0..polys.len() {
        //let (q, mut uhat) = poly_divmod(poly, &z);   
        let uhat = polys[i].clone();
        //let chat = params.commit_g1(&uhat);
        let chat = E::G1::identity();///advice_com[i];
        let polycom = params.commit_g1(&polys[i]);
        println!("Three");
        let (chat, d_small, cprime, wcom, bigc, d, x, zz, cplink_time) = cplink1_lite(&polys[i], &chat, &polycom, &z, &poly, &params, &domain);
        proving_time += cplink_time;
        for i in 0..num_runs {
          verifying_time[i] += verify1_lite(cprime, chat, d_small, params.clone(), z_com, wcom, bigc, d, x, zz);
        }
      }
    }
  }

  // KZG Commit proof

  if commit_poly {
    let idx = (poly_coeff_len + poly_col_len - 1) / poly_col_len - 1;
    let beta = public_valss[0];
    let rho_advice = advice_lagrange[poly_col_len + 1][idx];
    let rho = poly.evaluate(beta);
    println!("(Rho, Beta): {:?}", (rho, beta));
    println!("public vals len: {:?}", public_vals.iter().map(|vec| vec.len()).fold(0, |a, b| a + b));
    
    assert!(rho == rho_advice, "rho: {:?}, rho_advice: {:?}", rho, rho_advice);

    let kzg_proof_timer = Instant::now();
    println!("Tensor len: {}", tensor_len);
  
    let mut transcript_kzg_proof = Blake2bWrite::<_, E::G1Affine, Challenge255<_>>::init(vec![]);
    let poly_com = poly_params.commit(&poly, blind).to_affine();
    // let poly_coms: Vec<E::G1Affine> = polys.iter().map(|poly| params.commit(poly, blind).to_affine()).collect::<Vec<_>>();
    // let poly_com_sum = poly_coms.iter().fold(E::G1Affine::identity(), |a, b| (a + b).into());
    let queries = [
      ProverQuery {
          point: beta,
          poly: &poly,
          blind,
      }
    ].to_vec();
    //let (q, r) = poly_divmod::<E::Scalar>(&poly, &Polynomial::from_coefficients_vec(vec![-rho, E::Scalar::one()]));
    //let (q, r) = (poly - &Polynomial::from_coefficients_vec(vec![rho])).divide_with_q_and_r(&Polynomial::from_coefficients_vec(vec![-beta, E::Scalar::ONE])).unwrap();
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

    let kzg_proof_timer = Instant::now();
    println!("Tensor len: {}", tensor_len);
    
    // Advice rho proof
    let mut transcript_kzg_proof = Blake2bWrite::<_, E::G1Affine, Challenge255<_>>::init(vec![]);
    let domain = pk.get_vk().get_domain();
    let poly_com_advice = advice_com[poly_col_len + 1].to_affine();
    let poly_advice = &domain.lagrange_to_coeff(advice_lagrange[poly_col_len + 1].clone());

    let queries = [
      ProverQuery {
          point: domain.rotate_omega(E::Scalar::ONE, Rotation((idx) as i32)),
          poly: &poly_advice,
          blind: Blind::default(),
      }
    ].to_vec();
    //let (q, r) = poly_divmod::<E::Scalar>(&poly, &Polynomial::from_coefficients_vec(vec![-rho, E::Scalar::one()]));
    //let (q, r) = (poly - &Polynomial::from_coefficients_vec(vec![rho])).divide_with_q_and_r(&Polynomial::from_coefficients_vec(vec![-beta, E::Scalar::ONE])).unwrap();
    //assert!(r.is_zero());
    let prover = ProverSHPLONK::new(&poly_params);
    prover
        .create_proof(&mut OsRng, &mut transcript_kzg_proof, queries)
        .unwrap();
  
    let proof_kzg_advice = transcript_kzg_proof.finalize();
    proof_size += proof_kzg.len();
    // //let pi = params.commit(&q, blind);
    println!("KZG proof time: {:?}", kzg_proof_timer.elapsed());
    proving_time += kzg_proof_timer.elapsed();
    //KZG Commit vfy
    println!("Proving time: {:?}", proving_time);
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
      
      // For advice
      let kzg_vfy_timer = Instant::now();
      // let lhs = pairing(&(poly_com - (params.get_g()[0] * rho)).into(), &params.g2());
      // let rhs = pairing(&pi.into(), &(&params.s_g2() - &params.g2() * beta).into());
      // assert_eq!(lhs, rhs);
      let verifier_params = poly_params.verifier_params();
      let verifier = VerifierSHPLONK::new(&verifier_params);
      let mut transcript_kzg_verify = Blake2bRead::<_, _, Challenge255<_>>::init(proof_kzg_advice.as_slice());
    
      let queries = std::iter::empty()
          .chain(Some(VerifierQuery::new_commitment(&poly_com_advice, domain.rotate_omega(E::Scalar::ONE, Rotation((idx) as i32)), rho)));
    
      let msm = DualMSM::new(&poly_params);
      assert!(verifier.verify_proof(&mut transcript_kzg_verify, queries, msm).is_ok());
      let vfy_time = kzg_vfy_timer.elapsed();
      verifying_time[i] += vfy_time;
      println!("KZG vfy time: {:?}", vfy_time);
    }
  }
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

//Standalone verification
pub fn verify_circuit_kzg<
  E: Engine<
    G1Affine: SerdeCurveAffine,
    G2Affine: SerdeCurveAffine,
    Scalar: FromUniformBytes<64> + Ord + WithSmallOrderMulGroup<3> + SerdeObject,
  > + Debug + MultiMillerLoop
>(
  circuit: ModelCircuit<E::G1Affine>,
  vkey_fname: &str,
  proof_fname: &str,
  public_vals_fname: &str,
) {
  let degree = circuit.k as u32;
  let params = get_kzg_params::<E>("~/params_kzg", degree);
  println!("Loaded the parameters");

  let vk = VerifyingKey::read::<BufReader<File>, ModelCircuit<E::G1Affine>>(
    &mut BufReader::new(File::open(vkey_fname).unwrap()),
    SerdeFormat::RawBytes,
    (),
  )
  .unwrap();
  println!("Loaded vkey");

  let proof = std::fs::read(proof_fname).unwrap();

  let public_vals_u8 = std::fs::read(&public_vals_fname).unwrap();
  let public_vals: Vec<E::Scalar> = public_vals_u8
    .chunks(32)
    .map(|chunk| E::Scalar::from_uniform_bytes(chunk.try_into().expect("conversion failed")))
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
