use core::num;
use std::{
  cmp::min, fmt::Debug, fs::File, io::{BufReader, Write}, path::{self, Path}, time::{Duration, Instant}
};
use crate::utils::helpers::{cplink1, cplink1_lite, cplink2, powers, setup, vanishing_on_set, verify1, verify1_lite};
use crate::utils::lego_cplink::{lego_cplink_prove, lego_cplink_setup, lego_cplink_verify, lego_pedersen_commit};
use bitvec::{domain::Domain, order::verify};
use ff::{Field, FromUniformBytes, WithSmallOrderMulGroup};
use group::{Group, GroupEncoding, prime::PrimeCurveAffine};
use halo2_proofs::{
  arithmetic::{eval_polynomial, kate_division}, circuit, dev::MockProver, helpers::SerdeCurveAffine, plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Error, VerifyingKey}, poly::{self, commitment::{Blind, Params, ParamsProver, Prover, Verifier}, kzg::{commitment::{KZGCommitmentScheme, ParamsKZG}, msm::DualMSM, multiopen::{ProverGWC, ProverSHPLONK, VerifierGWC, VerifierSHPLONK}, strategy::{AccumulatorStrategy, SingleStrategy}}, Coeff, EvaluationDomain, LagrangeCoeff, Polynomial, ProverQuery, Rotation, VerificationStrategy, VerifierQuery}, transcript::{
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

/// Index of advice column `col` among the advice commitments in the proof transcript: halo2 commits
/// phase by phase, and within a phase in column order.
pub fn advice_transcript_index(phases: &[u8], col: usize) -> usize {
  let phase = phases[col];
  phases.iter().enumerate().filter(|(i, p)| **p < phase || (**p == phase && *i < col)).count()
}

/// Verifies a single-query KZG (GWC) opening proof: `com` opens to `eval` at `point`.
/// Unlike a bare `Verifier::verify_proof`, this runs the final pairing check (`SingleStrategy`).
pub fn verify_kzg_opening<
    E: Engine<
      G1Affine: SerdeCurveAffine,
      G2Affine: SerdeCurveAffine,
      Scalar: FromUniformBytes<64> + Ord + WithSmallOrderMulGroup<3>,
    > + Debug + MultiMillerLoop
  >(
  params: &ParamsKZG<E>,
  proof: &[u8],
  com: &E::G1Affine,
  point: E::Scalar,
  eval: E::Scalar,
) -> bool {
  let verifier = VerifierGWC::new(params);
  let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(proof);
  let queries = std::iter::once(VerifierQuery::new_commitment(com, point, eval));
  let strategy = SingleStrategy::new(params);
  <SingleStrategy<'_, E> as VerificationStrategy<'_, KZGCommitmentScheme<E>, VerifierGWC<'_, E>>>::process(
    strategy,
    |msm| verifier.verify_proof(&mut transcript, queries, msm).map_err(|_| Error::Opening),
  )
  .is_ok()
}

pub fn time_circuit_kzg<
  E: Engine<
    G1Affine: SerdeCurveAffine,
    G2Affine: SerdeCurveAffine,
    Scalar: FromUniformBytes<64> + Ord + WithSmallOrderMulGroup<3> + SerdeObject,
  > + Debug + MultiMillerLoop
>(circuit: ModelCircuit<E::G1Affine>, commit_poly: bool, pedersen: bool, poly_col_len: usize, cp_link: bool, num_runs: usize, directory: String, c: usize, slow: bool, lego: bool) {
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
      if tensor_len > 100 && tensor_len < 150 {println!("val: {:?}", val)}
      poly_coeff.push(val);
      if tensor_len == 0 {
        println!("Value: {:?}", val);
      }
      tensor_len += 1;
    }
    //println!("Tensor: {:?}, idx: {}", tensor, tensor_idx);
  }

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

  let proof_circuit = circuit.clone();
  let mut public_vals = vec![vec![]];
  // MockProver::run is only used to synthesize once and populate PUBLIC_VALS.
  let _prover = MockProver::run(degree, &proof_circuit, public_vals.clone()).unwrap();
  let fill_duration = start.elapsed();
  println!(
    "Time elapsed in filling circuit (MockProver): {:?}",
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
  let transcript_read: Blake2bRead<&[u8], <E as Engine>::G1Affine, Challenge255<<E as Engine>::G1Affine>> = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
  let mut proof_transcript = transcript_read.clone();
  let mut verifying_time = vec![Duration::new(0, 0); num_runs];
  let public_valss: Vec<E::Scalar> = get_public_values();
  // if commit_poly {
  //   public_vals[0][0] = rho;
  // }

  let mut advice_com = vec![];
  for _ in 0..advice_lagrange.len() {
    advice_com.push(proof_transcript.read_point().unwrap().to_curve());
  }
  //let slow = false;
  // LegoSNARK CP_link (see utils/lego_cplink.rs). Same circuit layout as Apollo: the weights are
  // copied into the first `poly_col_len` advice columns, `size` rows each (column-major).
  if lego {
    let ell = poly_col_len;
    assert!(ell >= 1, "cp_link_lego needs poly_cols >= 1 (the columns holding the weights)");
    let n = params.n() as usize;
    let size = (poly_coeff.len() + ell - 1) / ell;
    println!("LegoSNARK CP_link: {} weights in {} columns x {} rows (2^{} rows per column)", poly_coeff.len(), ell, size, circuit.k);

    // Sanity: the linked advice columns really hold the weights at the expected positions, and the
    // commitments read from the transcript are the commitments to those columns.
    for (idx, w) in poly_coeff.iter().enumerate() {
      assert_eq!(advice_lagrange[idx / size][idx % size], *w, "weight {} not found in advice column {} row {}", idx, idx / size, idx % size);
    }
    for i in 0..ell {
      assert_eq!(params.commit_lagrange(&advice_lagrange[i], Blind::default()), advice_com[i], "advice commitment {} does not match the transcript", i);
    }

    let setup_timer = Instant::now();
    let crs = lego_cplink_setup::<E>(&params, ell, size, poly_coeff.len());
    println!("LegoSNARK CP_link setup time (not counted): {:?}", setup_timer.elapsed());

    // Pedersen commitment to the weights. Counted towards proving time, like the KZG commitment to
    // the weights in the Artemis (`poly`) path.
    let r = E::Scalar::random(rand::thread_rng());
    let commit_timer = Instant::now();
    let weight_com = lego_pedersen_commit::<E>(&crs, r, &poly_coeff).to_affine();
    let commit_time = commit_timer.elapsed();
    println!("LegoSNARK CP_link weight commitment time: {:?}", commit_time);
    proving_time += commit_time;

    let prove_timer = Instant::now();
    let pi = lego_cplink_prove::<E>(&crs, r, &advice_lagrange[..ell]).to_affine();
    let prove_time = prove_timer.elapsed();
    println!("LegoSNARK CP_link proof time: {:?}", prove_time);
    proving_time += prove_time;
    proof_size += pi.to_bytes().as_ref().len(); // compressed, like the halo2 transcript

    let advice_coms_affine: Vec<E::G1Affine> = advice_com[..ell].iter().map(|c| c.to_affine()).collect();
    for i in 0..num_runs {
      let vfy_timer = Instant::now();
      assert!(lego_cplink_verify::<E>(&crs, pi, weight_com, &advice_coms_affine), "LegoSNARK CP_link proof did not verify");
      let vfy_time = vfy_timer.elapsed();
      println!("LegoSNARK CP_link verify time: {:?}", vfy_time);
      verifying_time[i] += vfy_time;
    }
  } else if cp_link {
    if poly_col_len < 1 && slow {
      // slow
      println!("Slow CPLINK");
      let col_size = circuit.k;
      let witness_size =  poly_coeff.len();
      let l = c;
      let size = (witness_size + l - 1) / l;
      let (HH, thetas, zs, z_v, z_last, zhats, z_coms, zhat_coms) = setup(col_size as u32, witness_size, l, &params);
      let vals = (0..l).map(|y| if y < l - 1 {poly_coeff[y*size..(y+1)*size].to_vec()} else {poly_coeff[y*size..].to_vec()}).collect::<Vec<_>>();
      //let vals = (0..l).map(|y| (0..size).map(|x| poly_coeff[y + l * x]).collect::<Vec<_>>()).collect::<Vec<_>>();
      let coeffs = vals.iter().map(|x| HH.lagrange_from_vec(x.clone())).collect::<Vec<_>>();
      //let us = coeffs.iter().map(|x| HH.lagrange_to_coeff(x.clone())).collect::<Vec<_>>();
      //let us = advice_lagrange.clone();
      let polys: Vec<_> = advice_lagrange.iter().map(|poly |HH.lagrange_to_coeff(poly.clone())).collect();
      //println!("Sum: {:?}", sum);
      //let u_coms = us.iter().map(|u| params.commit_g1(u)).collect();
      //let u_coms = advice_com.clone();
      //let u_coms = vec![E::G1::identity(); c];
      let z_v_com = params.commit_g2(&z_v);
      let u = HH.lagrange_to_coeff(HH.lagrange_from_vec(poly_coeff.clone()));
      let u_com = params.commit_g1(&u);
      let (uprimes, uprime_coms, cp2_prove, cp2_vfy, cp2_proof_size) = cplink2(thetas, HH.clone(), u, z_v.clone(), z_v_com, u_com, params.clone());
      proof_size += cp2_proof_size;
      proving_time += cp2_prove;
      for i in 0..num_runs {
        verifying_time[i] += cp2_vfy;
      }
      for i in 0..uprimes.len() {
        //let (q, mut uhat) = poly_divmod(poly, &z);   
        let poly_coeff = HH.lagrange_to_coeff(advice_lagrange[i].clone());//polys[i].clone();

       //let chat = E::G1::identity();///advice_com[i];
        let polycom = params.commit_g1(&poly_coeff);
        let (chat, d_small, cprime, wcom, bigc, d, x, zz, cplink_time) = cplink1_lite(&uprimes[i], &uprime_coms[i], &polycom, &z_v, &poly_coeff, &params, &HH);
        proving_time += cplink_time;
        // Proof elements per column (compressed): c', d_small, d. chat (weight commitment) and bigc (advice
        // commitment) are statement elements; the zeroed ZK elements (wcom, zz) are not sent.
        proof_size += 3 * cprime.to_affine().to_bytes().as_ref().len();
        for i in 0..num_runs {
          verifying_time[i] += verify1_lite(cprime, chat, d_small, &params, z_v_com, bigc, d);
        }
      }
      // let (chats, ds, cprimes, wcom, bigc, d, x, zz, cp1_prove) = cplink1(uprimes, zs, zhats, u, params.clone(), z_last, HH);
      // proving_time += cp1_prove + cp2_prove;
      // for i in 0..num_runs {
      //   let cp1_vfy = verify1(&cprimes, &chats, &ds, &params,&z_coms, &zhat_coms, wcom, bigc, d, x, zz);
      //   let vfy_total = cp1_vfy + cp2_vfy;
      //   verifying_time[i] += vfy_total;
      // }
    } else if poly_col_len >= 1 {
      println!("Apollo, cols: {:?}, rows: {:?}", c, circuit.k);
      //fast
      let domain = EvaluationDomain::<E::Scalar>::new(1, params.k());
      let size = (poly_coeff.len() + poly_col_len - 1) / poly_col_len;
      println!("Size: {:?}", size);
      let domain_vals = powers(domain.get_omega()).take(size as usize).collect::<Vec<_>>();
      // let domain_vals = (0..poly_col_len).map(|i| domain_vals[(i * poly_coeff.len() / poly_col_len)..((i + 1) * poly_coeff.len() / poly_col_len)].to_vec()).collect::<Vec<_>>() ;
      let z = vanishing_on_set(&domain_vals);
      //let z = Polynomial::from_coefficients_vec(vec![E::Scalar::one()]);
      let z_com = params.commit_g2(&z);
      //println!("poly sum: {:?}", poly_sum);
      //println!("z: {:?}", z);
      let polys_lagrange = (0..poly_col_len).map(|x| {
        let poly: Polynomial<E::Scalar, LagrangeCoeff> = if x < poly_col_len - 1 {
          domain.lagrange_from_vec(poly_coeff[(x * size)..((x + 1) * size)].to_vec())
        } else {
          domain.lagrange_from_vec(poly_coeff[(x * size)..].to_vec())
        };
        poly
      }).collect::<Vec<_>>();

      let uhats: Vec<_> = polys_lagrange.iter().map(|poly| poly_divmod(&domain.lagrange_to_coeff(poly.clone()), &z).1).collect();
      let chats: Vec<_> = uhats.iter().map(|uhat| params.commit_g1(&uhat)).collect();
      for i in 0..polys_lagrange.len() {
        //let (q, mut uhat) = poly_divmod(poly, &z);   
        let poly_coeff = domain.lagrange_to_coeff(advice_lagrange[i].clone());//polys[i].clone();

       //let chat = E::G1::identity();///advice_com[i];
        // The advice commitment the proof links to is the one in the halo2 transcript (first-phase
        // columns 0..poly_col_len come first in the transcript).
        let polycom = advice_com[i];
        debug_assert_eq!(params.commit_g1(&poly_coeff), polycom);
        let (chat, d_small, cprime, wcom, bigc, d, x, zz, cplink_time) = cplink1_lite(&uhats[i], &chats[i], &polycom, &z, &poly_coeff, &params, &domain);
        proving_time += cplink_time;
        // Proof elements per column (compressed): c', d_small, d. chat (weight commitment) and bigc (advice
        // commitment) are statement elements; the zeroed ZK elements (wcom, zz) are not sent.
        proof_size += 3 * cprime.to_affine().to_bytes().as_ref().len();
        for i in 0..num_runs {
          verifying_time[i] += verify1_lite(cprime, chat, d_small, &params, z_com, bigc, d);
        }
      }
    } else {
      println!("Fast CPLINK");
      let col_size = circuit.k;
      let witness_size =  poly_coeff.len();
      let size = (witness_size + c - 1) / c;
      let chunks_per_column = std::cmp::min((1 << circuit.k) / size, c);
      println!("Chunks per column: {:?}", chunks_per_column);
      let l = chunks_per_column;
      let (HH, thetas, zs, z_v, z_last, zhats, z_coms, zhat_coms) = setup(col_size as u32, size * chunks_per_column, l, &params);
      let poly_col_len = (poly_coeff.len() + (1 << circuit.k) - 1) / (1 << circuit.k);
      let poly_coeff = poly_coeff[..min(poly_col_len * chunks_per_column * size, poly_coeff.len())].to_vec();
      let vals = (0..c).map(|y| if y < c - 1 {poly_coeff[y*size..(y+1)*size].to_vec()} else {poly_coeff[y*size..].to_vec()}).collect::<Vec<_>>();
      //let poly_col_len = 3;
      //let vals = (0..l).map(|y| (0..size).map(|x| poly_coeff[y + l * x]).collect::<Vec<_>>()).collect::<Vec<_>>();
      //let coeffs = vals.iter().map(|x| HH.lagrange_from_vec(x.clone())).collect::<Vec<_>>();
      //let us = coeffs.iter().map(|x| HH.lagrange_to_coeff(x.clone())).collect::<Vec<_>>();
      //let us = advice_lagrange.clone();
      //let polys: Vec<_> = advice_lagrange.iter().map(|poly |HH.lagrange_to_coeff(poly.clone())).collect();
      //println!("Sum: {:?}", sum);
      //let u_coms = us.iter().map(|u| params.commit_g1(u)).collect();
      //let u_coms = advice_com.clone();
      //let u_coms = vec![E::G1::identity(); c];

      let z_v_com = params.commit_g2(&z_v);
      println!("poly_cols {:?}, chunks per column: {:?}", poly_col_len, chunks_per_column);
      for j in 0..poly_col_len {
        println!("j: {}, chunks_per_col: {}, j*c: {}", j, chunks_per_column, j*chunks_per_column);
        let evals =  if j < poly_col_len - 1 {
          vals[j*chunks_per_column..(j+1)*chunks_per_column].iter().flatten().map(|x| *x).collect::<Vec<E::Scalar>>()
        } else {
          vals[j*chunks_per_column..].iter().flatten().map(|x| *x).collect::<Vec<E::Scalar>>()
        };
        let chunks_left = std::cmp::min(c - (j) * chunks_per_column, chunks_per_column);
        println!("Chunks left {:?}", chunks_left);
        let u = HH.lagrange_to_coeff(HH.lagrange_from_vec(evals.clone()));
        let u_com = params.commit_g1(&u);
        let (uprimes, uprime_coms, cp2_prove, cp2_vfy, cp2_proof_size) = cplink2(thetas[..chunks_left].to_vec().clone(), HH.clone(), u, z_v.clone(), z_v_com, u_com, params.clone());
        proof_size += cp2_proof_size;
        proving_time += cp2_prove;
        for i in 0..num_runs {
          verifying_time[i] += cp2_vfy;
        }
        for i in 0..uprimes.len() {
          //let (q, mut uhat) = poly_divmod(poly, &z);   
          let witness_poly_coeff = HH.lagrange_to_coeff(advice_lagrange[i + j*chunks_per_column].clone());//polys[i].clone();
  
         //let chat = E::G1::identity();///advice_com[i];
          let polycom = params.commit_g1(&witness_poly_coeff);
          let (chat, d_small, cprime, wcom, bigc, d, x, zz, cplink_time) = cplink1_lite(&uprimes[i], &uprime_coms[i], &polycom, &z_v, &witness_poly_coeff, &params, &HH);
          proving_time += cplink_time;
          // Proof elements per column (compressed): c', d_small, d. chat (weight commitment) and bigc (advice
          // commitment) are statement elements; the zeroed ZK elements (wcom, zz) are not sent.
          proof_size += 3 * cprime.to_affine().to_bytes().as_ref().len();
          for i in 0..num_runs {
            verifying_time[i] += verify1_lite(cprime, chat, d_small, &params, z_v_com, bigc, d);
          }
        }
      }

      //fast
      // println!("Fast CPLink");
      // let domain = EvaluationDomain::<E::Scalar>::new(1, params.k());
      // let size = (poly_coeff.len() + c - 1) / c;
      // let domain_vals = powers(domain.get_omega()).take(size as usize).collect::<Vec<_>>();
      // // let domain_vals = (0..poly_col_len).map(|i| domain_vals[(i * poly_coeff.len() / poly_col_len)..((i + 1) * poly_coeff.len() / poly_col_len)].to_vec()).collect::<Vec<_>>() ;
      // let z = vanishing_on_set(&domain_vals);
      // //let z = Polynomial::from_coefficients_vec(vec![E::Scalar::one()]);
      // let z_com = params.commit_g2(&z);
      // //println!("poly sum: {:?}", poly_sum);
      // //println!("z: {:?}", z);
      // let polys_lagrange = (0..c).map(|x| {
      //   let poly: Polynomial<E::Scalar, LagrangeCoeff> = if x < c - 1 {
      //     domain.lagrange_from_vec(poly_coeff[(x * size)..((x + 1) * size)].to_vec())
      //   } else {
      //     domain.lagrange_from_vec(poly_coeff[(x * size)..].to_vec())
      //   };
      //   poly
      // }).collect::<Vec<_>>();

      // let uhats: Vec<_> = polys_lagrange.iter().map(|poly| poly_divmod(&domain.lagrange_to_coeff(poly.clone()), &z).1).collect();
      // let chats: Vec<_> = uhats.iter().map(|uhat| params.commit_g1(&uhat)).collect();
      // for i in 0..c {
      //   //let (q, mut uhat) = poly_divmod(poly, &z);   
      //   let poly_coeff = domain.lagrange_to_coeff(advice_lagrange[i].clone());//polys[i].clone();

      //  //let chat = E::G1::identity();///advice_com[i];
      //   let polycom = params.commit_g1(&poly_coeff);
      //   let (chat, d_small, cprime, wcom, bigc, d, x, zz, cplink_time) = cplink1_lite(&uhats[i], &chats[i], &polycom, &z, &poly_coeff, &params, &domain);
      //   proving_time += cplink_time;
      //   proof_size += cprime.to_affine().to_raw_bytes().len() + chat.to_affine().to_raw_bytes().len() + d_small.to_affine().to_raw_bytes().len();
      //   proof_size += wcom.to_affine().to_raw_bytes().len() + bigc.to_affine().to_raw_bytes().len() + d.to_affine().to_raw_bytes().len();
      //   proof_size += zz.to_raw_bytes().len();  
      //   for i in 0..num_runs {
      //     verifying_time[i] += verify1_lite(cprime, chat, d_small, &params, z_com, bigc, d);
      //   }
      // }
    }
  }

  // Artemis (`poly`): two KZG openings link the committed weight polynomial p to the in-circuit value:
  //  (1) p opens to rho at the Fiat-Shamir challenge beta;
  //  (2) the Poly5/Poly2 accumulator column opens to rho at omega^row_idx (the last Horner row).
  // Both are single-query GWC openings (one MSM each) and are verified with a real pairing check.
  if commit_poly {
    let col_idx = if pedersen {poly_col_len * 2} else {poly_col_len};
    let row_idx = (poly_coeff.len() + poly_col_len - 1) / poly_col_len - 1;
    let beta = public_valss[0];

    let rho_advice = advice_lagrange[col_idx][row_idx];
    let rho = poly.evaluate(beta);
    println!("(Rho, Beta): {:?}, row_idx: {:?}, col_idx: {:?}", (rho, beta), row_idx, col_idx);
    assert!(rho == rho_advice, "rho: {:?}, rho_advice: {:?}", rho, rho_advice);

    // (1) Weight polynomial at beta. The commitment is the "commit" phase of the CP-SNARK; it is timed
    // separately and counted towards the prover time (same convention as cp_link_lego).
    let commit_timer = Instant::now();
    let poly_com = poly_params.commit(&poly, blind).to_affine();
    let commit_time = commit_timer.elapsed();
    println!("Weight commitment time: {:?}", commit_time);
    proving_time += commit_time;

    let kzg_proof_timer = Instant::now();
    let mut transcript_kzg_proof = Blake2bWrite::<_, E::G1Affine, Challenge255<_>>::init(vec![]);
    ProverGWC::new(&poly_params)
      .create_proof(&mut OsRng, &mut transcript_kzg_proof, [ProverQuery { point: beta, poly: &poly, blind }])
      .unwrap();
    let proof_kzg = transcript_kzg_proof.finalize();
    let kzg_proof_time = kzg_proof_timer.elapsed();
    println!("KZG proof time (weights at beta): {:?}", kzg_proof_time);
    proving_time += kzg_proof_time;
    proof_size += proof_kzg.len();

    // (2) Accumulator column at omega^row_idx (degree 2^k, so the circuit SRS suffices).
    let kzg_proof_timer = Instant::now();
    let domain = pk.get_vk().get_domain();
    let open_point = domain.rotate_omega(E::Scalar::ONE, Rotation(row_idx as i32));
    // halo2 writes the advice commitments to the transcript phase by phase (column order within a
    // phase), so the transcript index of the SecondPhase accumulator column differs from its column index.
    let com_idx = advice_transcript_index(&pk.get_vk().cs().advice_column_phase(), col_idx);
    let poly_com_advice = advice_com[com_idx].to_affine();
    assert_eq!(params.commit_lagrange(&advice_lagrange[col_idx], Blind::default()).to_affine(), poly_com_advice,
      "transcript commitment {} is not the commitment to advice column {}", com_idx, col_idx);
    let poly_advice = domain.lagrange_to_coeff(advice_lagrange[col_idx].clone());
    let mut transcript_kzg_proof = Blake2bWrite::<_, E::G1Affine, Challenge255<_>>::init(vec![]);
    ProverGWC::new(&params)
      .create_proof(&mut OsRng, &mut transcript_kzg_proof, [ProverQuery { point: open_point, poly: &poly_advice, blind: Blind::default() }])
      .unwrap();
    let proof_kzg_advice = transcript_kzg_proof.finalize();
    let kzg_proof_time = kzg_proof_timer.elapsed();
    println!("KZG proof time (advice cell): {:?}", kzg_proof_time);
    proving_time += kzg_proof_time;
    proof_size += proof_kzg_advice.len();
    println!("Proving time: {:?}", proving_time);

    for i in 0..num_runs {
      let kzg_vfy_timer = Instant::now();
      assert!(verify_kzg_opening::<E>(&poly_params, &proof_kzg, &poly_com, beta, rho), "weight-polynomial opening did not verify");
      let vfy_time = kzg_vfy_timer.elapsed();
      verifying_time[i] += vfy_time;
      println!("KZG vfy time (weights at beta): {:?}", vfy_time);

      let kzg_vfy_timer = Instant::now();
      assert!(verify_kzg_opening::<E>(&params, &proof_kzg_advice, &poly_com_advice, open_point, rho), "advice-cell opening did not verify");
      let vfy_time = kzg_vfy_timer.elapsed();
      verifying_time[i] += vfy_time;
      println!("KZG vfy time (advice cell): {:?}", vfy_time);
    }
  }

  for i in 0..num_runs {
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
    verifying_time[i] += verify_duration;
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
