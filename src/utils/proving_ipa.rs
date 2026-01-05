use core::num;
use std::{
  fs::File,
  io::{BufReader, Write},
  path::Path,
  time::Instant,
};

use csv::Writer;
use group::{cofactor::CofactorCurveAffine, Group};
use halo2_proofs::{
  arithmetic::eval_polynomial, circuit, dev::MockProver, halo2curves::pasta::{EqAffine, Fp}, plonk::{create_proof, keygen_pk, keygen_vk, verify_proof}, poly::{
    commitment::{Blind, Params, ParamsProver, Prover, Verifier, MSM}, ipa::{
      self, commitment::{IPACommitmentScheme, ParamsIPA}, msm::MSMIPA, multiopen::{ProverIPA, VerifierIPA}, strategy::SingleStrategy
    }, Coeff, Polynomial, ProverQuery, Rotation, VerificationStrategy, VerifierQuery
  }, transcript::{
    Blake2bRead, Blake2bWrite, Challenge255, Transcript, TranscriptRead, TranscriptReadBuffer, TranscriptWrite, TranscriptWriterBuffer
  }
};
use halo2curves::{bn256, pasta::{EpAffine, Eq}};
use ff::Field;
use group::Curve;
use rand::thread_rng;
use rand_core::OsRng;
use crate::{
  model::ModelCircuit,
  utils::{
    barycentric::{bary_ipa, bary_verify_ipa},
    helpers::{get_public_values, zkfft_commit_ipa, zkfft_verify_ipa}
  }
};

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

pub fn time_circuit_ipa(circuit: ModelCircuit<EqAffine>, commit_poly: bool, poly_col_len: usize,  num_runs: usize, directory: String, pedersen: bool, zkfft: bool, barycentric: bool) {
  let mut rng = &mut rand::thread_rng();
  let start = Instant::now();

  let degree = circuit.k as u32;
  let mut circuit = circuit.clone();

  let mut tensor_len = 0usize;
  let mut poly_coeff = Vec::with_capacity(2usize.pow((circuit.k + poly_col_len) as u32));
  for (tensor_idx, tensor) in circuit.tensors.clone() {
    for val in tensor.clone() {
      tensor_len += 1;
      poly_coeff.push(val);
    }
    //println!("Tensor: {:?}, idx: {}", tensor, tensor_idx);
  }
  println!("First poly coeff len: {:?}", poly_coeff.len());
  let mut transcript_proof =
      Blake2bWrite::<Vec<u8>, EqAffine, Challenge255<EqAffine>>::init(vec![]);
  let beta = Fp::random(rng);//transcript_proof.squeeze_challenge_scalar::<()>();
  //let alpha = Fp::random(&mut thread_rng());
  let alpha = Fp::ONE;

  if commit_poly {
    let beta_pows = (0..poly_col_len + 1).map(|i| beta.pow([i as u64])).rev().collect::<Vec<_>>();

    while poly_coeff.len() % poly_col_len != 0  {
      poly_coeff.push(Fp::ZERO);
    }
    //circuit.beta_pows = beta_pows.clone();

    circuit.beta_pows = beta_pows.clone();
  }

  // let empty_circuit = circuit.clone();
  // let mut proof_circuit = circuit.clone();

  let params = get_ipa_params(format!("{}/params_ipa", directory).as_str(), degree);

  let circuit_duration = start.elapsed();
  println!(
    "Time elapsed in params construction: {:?}",
    circuit_duration
  );
  
  let vk = keygen_vk(&params, &circuit).unwrap();
  let vk_duration = start.elapsed();
  println!(
    "Time elapsed in generating vkey: {:?}",
    vk_duration - circuit_duration
  );

  let pk = keygen_pk(&params, vk, &circuit).unwrap();
  let pk_duration = start.elapsed();
  println!(
    "Time elapsed in generating pkey: {:?}",
    pk_duration - vk_duration
  );
  //drop(empty_circuit);


  let fill_duration = start.elapsed();
  let _prover = MockProver::run(degree, &circuit, vec![vec![]]).unwrap();
  println!(
    "Time elapsed in filling circuit: {:?}",
    fill_duration - pk_duration
  );

  //let poly_coeff = vec![Fp::one(); 1 << params.k()];
  //println!("poly coeff len: {}", poly_coeff.len());
  //println!("sum: {}", circuit.k as u32 + poly_col_len as u32);

  let mut poly_params = params.clone();
  if poly_col_len > 1 {
    poly_params = get_ipa_params(format!("{}/params_ipa", directory).as_str(), degree + (poly_col_len - 1).ilog2() + 1 as u32);
  }

  // while poly_coeff.len() < 2usize.pow(poly_params.k() as u32) {
  //   poly_coeff.push(Fp::ZERO);
  // }
  println!("Second poly coeff len: {:?}", poly_coeff.len());
  //poly_coeff.extend(vec![Fp::ZERO; 2usize.pow(circuit.k as u32 + poly_col_len as u32) - poly_coeff.len()]);

  let poly: Polynomial<Fp, Coeff> = Polynomial::from_coefficients_vec(poly_coeff.clone());
  let mut polys = vec![Polynomial::from_coefficients_vec(poly_coeff.clone())];
  if poly_col_len > 0 {
    polys = (0..poly_col_len).map(|x| {
      let poly: Polynomial<Fp, Coeff> = Polynomial::from_coefficients_vec(poly_coeff[(x * poly_coeff.len() / poly_col_len)..((x + 1) * poly_coeff.len() / poly_col_len)].to_vec()) * alpha.pow([x as u64]);
      poly
    }).collect::<Vec<_>>();
  }

  let rho = eval_polynomial(&poly, beta);

  let mut public_vals = vec![vec![]];
  //let mut betas = vec![vec![]; poly_col_len];

  public_vals[0] = get_public_values();
  if commit_poly { 
    public_vals[0][0] = rho;
  }

  let public_vals_slice = public_vals.iter().map(|x| x.as_slice()).collect::<Vec<_>>();

  let proof_duration_start = start.elapsed();
  let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
  let mut advice_lagrange = vec![];
  let mut advice_blind = vec![];
  create_proof::<IPACommitmentScheme<EqAffine>, ProverIPA<EqAffine>, _, _, _, _>(
    &params,
    &pk,
    &[circuit],
    &[&public_vals_slice.as_slice()],
    &mut rand::thread_rng(),
    &mut transcript,
    &mut advice_lagrange,
    &mut advice_blind,
  )
  .unwrap();

  let proof = transcript.finalize();
  let proof_duration = start.elapsed();

  println!("DEBUG: advice_blind length: {}", advice_blind.len());
  for i in 0..advice_blind.len().min(3) {
    println!("DEBUG: advice_blind[{}] = {:?}", i, advice_blind[i]);
  }

  let mut advice_com = vec![];
  let transcript_read: Blake2bRead<&[u8], EqAffine, Challenge255<EqAffine>> = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
  let mut proof_transcript = transcript_read.clone();
  for _ in 0..advice_lagrange.len() {
    advice_com.push(proof_transcript.read_point().unwrap().to_curve());
  }

  let public_valss: Vec<Fp> = get_public_values();
  println!("Proving time: {:?}", proof_duration - proof_duration_start);
  let mut proving_time = proof_duration - proof_duration_start;
  println!("Proof size: {} bytes", proof.len());
  let mut proof_size = proof.len();
  let mut verifying_time = vec![];
  for i in 0..num_runs {
    let verification = Instant::now();
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    let strategy = SingleStrategy::new(&params);
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
    let verify_duration = verification.elapsed();
    verifying_time.push(verify_duration);
    //println!("Verifying time: {:?}", verify_duration);
  }

  if commit_poly {
    let col_idx = if pedersen {poly_col_len * 2} else {poly_col_len};
    let row_idx = (poly_coeff.len() + poly_col_len - 1) / poly_col_len - 1;
    println!("poly coeff len: {:?}, poly col len: {:?}", poly_coeff.len(), poly_col_len);
    let beta = public_valss[0];
    
    let rho_advice = advice_lagrange[col_idx][row_idx];
    let rho = poly.evaluate(beta);
    println!("(Rho, Beta): {:?}, row_idx: {:?}, col_idx: {:?}", (rho, beta), row_idx, col_idx);
    println!("public vals len: {:?}", public_vals.iter().map(|vec| vec.len()).fold(0, |a, b| a + b));
    
    assert!(rho == rho_advice, "rho: {:?}, rho_advice: {:?}", rho, rho_advice);

    let ipa_proof_timer = Instant::now();
    println!("Tensor len: {}", tensor_len);
  
    let mut transcript_ipa_proof = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
    let poly_com = poly_params.commit(&poly, Blind::default()).to_affine();
    // let poly_coms: Vec<E::G1Affine> = polys.iter().map(|poly| params.commit(poly, blind).to_affine()).collect::<Vec<_>>();
    // let poly_com_sum = poly_coms.iter().fold(E::G1Affine::identity(), |a, b| (a + b).into());
    let queries = [
      ProverQuery {
          point: beta,
          poly: &poly,
          blind: Blind::default(),
      }
    ].to_vec();
    //let (q, r) = poly_divmod::<E::Scalar>(&poly, &Polynomial::from_coefficients_vec(vec![-rho, E::Scalar::one()]));
    //let (q, r) = (poly - &Polynomial::from_coefficients_vec(vec![rho])).divide_with_q_and_r(&Polynomial::from_coefficients_vec(vec![-beta, E::Scalar::ONE])).unwrap();
    //assert!(r.is_zero());
    let prover = ProverIPA::new(&poly_params);
    prover
        .create_proof(&mut OsRng, &mut transcript_ipa_proof, queries)
        .unwrap();
  
    let proof_ipa = transcript_ipa_proof.finalize();
    proof_size += proof_ipa.len();
    //let pi = params.commit(&q, blind);
    println!("IPA proof time: {:?}", ipa_proof_timer.elapsed());
    proving_time += ipa_proof_timer.elapsed();

    let ipa_proof_timer = Instant::now();
    println!("Tensor len: {}", tensor_len);
    
    // Advice rho proof
    let mut transcript_ipa_proof = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
    let domain = pk.get_vk().get_domain();
    let poly_com_advice = advice_com[poly_col_len + 1].to_affine();
    let poly_advice = &domain.lagrange_to_coeff(advice_lagrange[poly_col_len + 1].clone());

    let queries = [
      ProverQuery {
          point: domain.rotate_omega(Fp::ONE, Rotation((row_idx) as i32)),
          poly: &poly_advice,
          blind: Blind::default(),
      }
    ].to_vec();
    //let (q, r) = poly_divmod::<E::Scalar>(&poly, &Polynomial::from_coefficients_vec(vec![-rho, E::Scalar::one()]));
    //let (q, r) = (poly - &Polynomial::from_coefficients_vec(vec![rho])).divide_with_q_and_r(&Polynomial::from_coefficients_vec(vec![-beta, E::Scalar::ONE])).unwrap();
    //assert!(r.is_zero());
    let prover = ProverIPA::new(&poly_params);
    prover
        .create_proof(&mut OsRng, &mut transcript_ipa_proof, queries)
        .unwrap();
  
    let proof_ipa_advice = transcript_ipa_proof.finalize();
    proof_size += proof_ipa.len();
    // //let pi = params.commit(&q, blind);
    println!("IPA proof time: {:?}", ipa_proof_timer.elapsed());
    proving_time += ipa_proof_timer.elapsed();
    //IPA Commit vfy
    println!("Proving time: {:?}", proving_time);
    for i in 0..num_runs {
      let ipa_vfy_timer = Instant::now();
      // let lhs = pairing(&(poly_com - (params.get_g()[0] * rho)).into(), &params.g2());
      // let rhs = pairing(&pi.into(), &(&params.s_g2() - &params.g2() * beta).into());
      // assert_eq!(lhs, rhs);
      let verifier_params = poly_params.verifier_params();
      let verifier = VerifierIPA::new(&verifier_params);
      let mut transcript_ipa_verify = Blake2bRead::<_, _, Challenge255<_>>::init(proof_ipa.as_slice());
    
      let queries = std::iter::empty()
          .chain(Some(VerifierQuery::new_commitment(&poly_com, beta, rho)));
    
      let msm = MSMIPA::new(&poly_params);
      assert!(verifier.verify_proof(&mut transcript_ipa_verify, queries, msm).is_ok());
      let vfy_time = ipa_vfy_timer.elapsed();
      verifying_time[i] += vfy_time;
      println!("IPA vfy time: {:?}", vfy_time);
      
      // For advice
      let ipa_vfy_timer = Instant::now();
      // let lhs = pairing(&(poly_com - (params.get_g()[0] * rho)).into(), &params.g2());
      // let rhs = pairing(&pi.into(), &(&params.s_g2() - &params.g2() * beta).into());
      // assert_eq!(lhs, rhs);
      let verifier_params = poly_params.verifier_params();
      let verifier = VerifierIPA::new(&verifier_params);
      let mut transcript_ipa_verify = Blake2bRead::<_, _, Challenge255<_>>::init(proof_ipa_advice.as_slice());
    
      let queries = std::iter::empty()
          .chain(Some(VerifierQuery::new_commitment(&poly_com_advice, domain.rotate_omega(Fp::ONE, Rotation((row_idx) as i32)), rho)));
    
      let msm = MSMIPA::new(&poly_params);
      assert!(verifier.verify_proof(&mut transcript_ipa_verify, queries, msm).is_ok());
      let vfy_time = ipa_vfy_timer.elapsed();
      verifying_time[i] += vfy_time;
      println!("IPA vfy time: {:?}", vfy_time);
    }
  }

  // zkFFT commitment (if enabled)
  if zkfft {
    println!("Running zkFFT commitment...");
    println!("Warning! Takes a lot of memory");

    // Get domain for omega
    let domain = pk.get_vk().get_domain();

    // Prepare witness and generators for computing initial commitment P
    let omega = domain.get_omega();
    let poly_advice_coeff = domain.lagrange_to_coeff(advice_lagrange[poly_col_len + 1].clone());
    let mut a: Vec<Fp> = poly_advice_coeff.values.clone();
    let n = a.len();
    let next_pow2 = n.next_power_of_two();
    if n < next_pow2 {
        a.resize(next_pow2, Fp::ZERO);
    }
    let n = a.len();
    let k = n;

    // Precompute omega^i for all i
    let mut omega_powers = Vec::with_capacity(k);
    omega_powers.push(Fp::ONE);
    for i in 1..k {
        omega_powers.push(omega_powers[i - 1] * omega);
    }

    println!("Precomputing");

    // Build b[i][j] = omega^(i*j) using iterative multiplication
    let mut b = Vec::with_capacity(k);
    for i in 0..k {
        let mut b_i = Vec::with_capacity(n);
        let omega_i = omega_powers[i];
        let mut current = Fp::ONE;
        for _j in 0..n {
            b_i.push(current);
            current *= omega_i;
        }
        b.push(b_i);
    }

    // Get generators (same as prover)
    let generators_g: Vec<EqAffine> = poly_params.get_g()
        .iter().take(n).copied().collect();
    let generator_g = generators_g.clone();
    let generator_h = poly_params.get_g()[n];

    // Generate alpha (blinding factor)
    let alpha = Fp::random(OsRng);

    // Compute initial commitment P = sum(a[i]*g[i]) + sum(inner_products[i]*g'[i]) + alpha*h
    println!("Computing initial commitment P...");

    // Compute inner products: inner_products[i] = sum(a[j] * b[i][j])
    let mut inner_products = Vec::with_capacity(k);
    for i in 0..k {
        let mut ip = Fp::ZERO;
        for j in 0..n {
            ip += a[j] * b[i][j];
        }
        inner_products.push(ip);
    }

    // Build multiexp terms for P
    use halo2_proofs::arithmetic::best_multiexp;
    let mut P_terms: Vec<(Fp, EqAffine)> = Vec::new();

    // Add a[i] * generators_g[i] terms
    for i in 0..n {
        P_terms.push((a[i], generators_g[i]));
    }

    // Add inner_products[i] * generator_g[i] terms
    for i in 0..k {
        P_terms.push((inner_products[i], generator_g[i]));
    }

    // Add alpha * generator_h term
    P_terms.push((alpha, generator_h));

    // Compute P using multiexp
    let (coeffs, bases): (Vec<_>, Vec<_>) = P_terms.iter().cloned().unzip();
    let commitment_P: EqAffine = best_multiexp(&coeffs, &bases).into();

    println!("Initial commitment P computed");

    // Run prover with alpha
    let (zkfft_proof_bytes, zkfft_ptime, _zkfft_vtime, zkfft_size) =
        zkfft_commit_ipa(poly_advice_coeff.clone(), &poly_params, domain.clone(), alpha);

    proving_time += zkfft_ptime;
    proof_size += zkfft_size;

    println!("zkFFT: Proof size: {} bytes", zkfft_size);

    // Run verifier
    println!("Running zkFFT verifier...");
    let verify_start = Instant::now();

    let verified = zkfft_verify_ipa(
        &zkfft_proof_bytes,
        b,
        generators_g,
        generator_g,
        generator_h,
        commitment_P,
    );

    let vfy_time = verify_start.elapsed();

    if verified {
        println!("zkFFT: Verification PASSED ✓");
        // Update verifying time for all runs
        for i in 0..num_runs {
            verifying_time[i] += vfy_time;
        }
    } else {
        println!("zkFFT: Verification FAILED ✗");
        panic!("zkFFT verification failed!");
    }
  }

  // Barycentric commitment (if enabled)
  if barycentric {
    println!("Running barycentric commitment...");
    println!("DEBUG: Total advice columns: {}", advice_lagrange.len());
    println!("DEBUG: poly_col_len = {}, col_idx would be = {}", poly_col_len, if pedersen {poly_col_len * 2} else {poly_col_len});

    // Print sizes of first few advice columns
    for i in 0..advice_lagrange.len().min(5) {
      println!("DEBUG: advice_lagrange[{}].len() = {}", i, advice_lagrange[i].values.len());
    }

    // Get domain for omega
    let domain = pk.get_vk().get_domain();

    // Try to find which column contains the witness by checking matches
    println!("\nDEBUG: Checking which advice column matches poly best:");
    println!("DEBUG: poly_col_len = {}, so witness should be split across first {} columns", poly_col_len, poly_col_len);
    let mut poly_as_lagrange_test = poly.values.clone();
    while poly_as_lagrange_test.len() < domain.get_n() as usize {
        poly_as_lagrange_test.push(Fp::ZERO);
    }

    let mut best_col = 0;
    let mut best_matches = 0;

    for col_idx in 0..advice_lagrange.len() {
        let mut matches = 0;
        for i in 0..poly.values.len() {
            if poly_as_lagrange_test[i] == advice_lagrange[col_idx].values[i] {
                matches += 1;
            }
        }
        println!("  Column {}: {}/{} matches ({:.2}%)", col_idx, matches, poly.values.len(),
                 100.0 * matches as f64 / poly.values.len() as f64);

        if matches > best_matches {
            best_matches = matches;
            best_col = col_idx;
        }
    }

    println!("\nDEBUG: Best matching column is {} with {}/{} matches", best_col, best_matches, poly.values.len());

    // Check if witness is split across multiple columns (columns_poly[0..poly_col_len])
    println!("\nDEBUG: Checking if witness is split across first {} columns:", poly_col_len);
    for col_idx in 0..poly_col_len.min(advice_lagrange.len()) {
        let chunk_size = (poly.values.len() + poly_col_len - 1) / poly_col_len;
        let start_idx = col_idx * chunk_size;
        let end_idx = (start_idx + chunk_size).min(poly.values.len());

        let mut chunk_matches = 0;
        for i in 0..(end_idx - start_idx) {
            let poly_idx = start_idx + i;
            if poly_idx < poly.values.len() && poly_as_lagrange_test[poly_idx] == advice_lagrange[col_idx].values[i] {
                chunk_matches += 1;
            }
        }
        println!("  Column {} (chunk {}): {}/{} matches", col_idx, col_idx, chunk_matches, end_idx - start_idx);
    }

    // Reconstruct the full witness polynomial from split columns
    println!("\nDEBUG: Reconstructing full witness from columns 0..{}", poly_col_len);
    let chunk_size = (poly.values.len() + poly_col_len - 1) / poly_col_len;

    // Check if blinding is consistent across columns
    println!("DEBUG: Checking blinding consistency across columns:");
    for row in (domain.get_n() as usize - 6)..(domain.get_n() as usize) {
        print!("  Row {}: ", row);
        for col in 0..poly_col_len {
            print!("col{}={:?} ", col, advice_lagrange[col].values[row] == Fp::ZERO);
        }
        println!();
    }

    let mut reconstructed_lagrange = vec![Fp::ZERO; domain.get_n() as usize];
    for col_idx in 0..poly_col_len {
        let start_idx = col_idx * chunk_size;
        let end_idx = (start_idx + chunk_size).min(poly.values.len());

        for i in 0..(end_idx - start_idx) {
            let poly_idx = start_idx + i;
            reconstructed_lagrange[poly_idx] = advice_lagrange[col_idx].values[i];
        }
    }

    // Copy blinding values from first column
    for i in poly.values.len()..(domain.get_n() as usize) {
        reconstructed_lagrange[i] = advice_lagrange[0].values[i];
    }

    let poly_advice_lagrange = domain.lagrange_from_vec(reconstructed_lagrange);

    println!("DEBUG: Reconstructed witness polynomial from {} columns", poly_col_len);
    println!("DEBUG: Verifying reconstruction matches poly:");
    let mut recon_matches = 0;
    for i in 0..poly.values.len() {
        if poly_as_lagrange_test[i] == poly_advice_lagrange.values[i] {
            recon_matches += 1;
        }
    }
    println!("  Reconstruction: {}/{} matches ({:.2}%)", recon_matches, poly.values.len(),
             100.0 * recon_matches as f64 / poly.values.len() as f64);

    // Run prover with both polynomials at beta
    // Pass the blinding factor from the first column (or reconstructed column)
    let blind_value = advice_blind[0];  // For single column case, use first blind
    println!("DEBUG: Using blind value: {:?}", blind_value);

    let (bary_proof_bytes, bary_ptime, _bary_vtime, bary_size) =
        bary_ipa(poly.clone(), poly_advice_lagrange, beta, &poly_params, domain.clone(), alpha, blind_value);

    proving_time += bary_ptime;
    proof_size += bary_size;

    println!("Barycentric: Proof size: {} bytes", bary_size);

    // For verification, we need:
    // 1. Commitment to the external polynomial (as Lagrange values)
    // 2. Evaluation at beta (which includes blinding contribution)

    // Compute poly as Lagrange and commit
    let poly_as_lagrange_direct = domain.lagrange_from_vec(poly.values.clone());
    let poly_from_lag_coeff = domain.lagrange_to_coeff(poly_as_lagrange_direct.clone());
    let poly_com = poly_params.commit(&poly_from_lag_coeff, Blind::default()).to_affine();

    // Compute evaluation: <poly, b_coeffs> + blinding_contribution
    // This was already computed in bary_ipa as rho_poly_blinded
    // We need to recompute it here for the verifier

    // Precompute omega powers for barycentric coefficients
    let mut omega_powers = Vec::with_capacity(domain.get_n() as usize);
    omega_powers.push(Fp::ONE);
    for i in 1..(domain.get_n() as usize) {
        omega_powers.push(domain.rotate_omega(omega_powers[i - 1], Rotation(1)));
    }

    // Compute barycentric coefficients
    let beta_d = beta.pow([domain.get_n() as u64]);
    let numerator = beta_d - Fp::ONE;
    let d_inv = Fp::from(domain.get_n() as u64).invert().unwrap();
    let scaling_factor = numerator * d_inv;

    let mut b_coeffs = Vec::with_capacity(domain.get_n() as usize);
    for i in 0..(domain.get_n() as usize) {
        let omega_i = omega_powers[i];
        let denominator = beta - omega_i;
        if denominator == Fp::ZERO {
            b_coeffs.push(Fp::ZERO);
        } else {
            let b_i = scaling_factor * omega_i * denominator.invert().unwrap();
            b_coeffs.push(b_i);
        }
    }

    // Compute <poly, b_coeffs>
    let mut rho = Fp::ZERO;
    for i in 0..poly.values.len() {
        rho += poly.values[i] * b_coeffs[i];
    }

    // Add blinding contribution from reconstructed poly_advice
    let chunk_size = (poly.values.len() + poly_col_len - 1) / poly_col_len;
    let mut reconstructed_lagrange = vec![Fp::ZERO; domain.get_n() as usize];
    for col_idx in 0..poly_col_len {
        let start_idx = col_idx * chunk_size;
        let end_idx = (start_idx + chunk_size).min(poly.values.len());
        for i in 0..(end_idx - start_idx) {
            let poly_idx = start_idx + i;
            reconstructed_lagrange[poly_idx] = advice_lagrange[col_idx].values[i];
        }
    }
    for i in poly.values.len()..(domain.get_n() as usize) {
        reconstructed_lagrange[i] = advice_lagrange[0].values[i];
    }

    let mut blinding_contribution = Fp::ZERO;
    for i in poly.values.len()..(domain.get_n() as usize) {
        blinding_contribution += reconstructed_lagrange[i] * b_coeffs[i];
    }

    rho += blinding_contribution;

    // Run verifier
    println!("Running barycentric verifier...");
    for i in 0..num_runs {
      let verify_start = Instant::now();

      let verified = bary_verify_ipa(
          &bary_proof_bytes,
          poly_com,
          beta,
          rho,
          &poly_params,
      );

      let vfy_time = verify_start.elapsed();
      verifying_time[i] += vfy_time;
      println!("Barycentric IPA vfy time: {:?}", vfy_time);

      if !verified {
          println!("Barycentric: Verification FAILED ✗");
          panic!("Barycentric verification failed!");
      }
    }
    println!("Barycentric: Verification PASSED ✓");
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
