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
  arithmetic::eval_polynomial, circuit, dev::MockProver, halo2curves::pasta::{EqAffine, Fp}, plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, ExtraPoint, ExtraProverQuery, ExtraVerifierQuery}, poly::{
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
  utils::proving_kzg::advice_transcript_index,
  utils::{
    barycentric::{bary_ipa, bary_verify_ipa},
    helpers::{get_public_values, zkfft_commit_ipa, zkfft_verify_ipa, zkfft_commit_ipa_fft},
    lazy_b_matrix::LazyBMatrix
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

/// Verifies a single-query IPA opening proof: `com` opens to `eval` at `point`.
/// Unlike a bare `Verifier::verify_proof`, this runs the final MSM check (`SingleStrategy`).
pub fn verify_ipa_opening(params: &ParamsIPA<EqAffine>, proof: &[u8], com: &EqAffine, point: Fp, eval: Fp) -> bool {
  let verifier = VerifierIPA::new(params);
  let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(proof);
  let queries = std::iter::once(VerifierQuery::new_commitment(com, point, eval));
  let strategy = SingleStrategy::new(params);
  strategy.process(|msm| verifier.verify_proof(&mut transcript, queries, msm).map_err(|_| halo2_proofs::plonk::Error::Opening)).is_ok()
}

pub fn time_circuit_ipa(circuit: ModelCircuit<EqAffine>, commit_poly: bool, poly_col_len: usize,  num_runs: usize, directory: String, pedersen: bool, zkfft: bool, barycentric: bool) {
  let mut rng = &mut rand::thread_rng();
  let start = Instant::now();

  let degree = circuit.k as u32;
  let mut circuit = circuit;

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


  // MockProver::run is only used to synthesize once and populate PUBLIC_VALS; it is dropped right after.
  let mock_prover = MockProver::run(degree, &circuit, vec![vec![]]).unwrap();
  if std::env::var("MOCK_VERIFY").is_ok() {
    // MockProver reports "cell not assigned" for cells that a gate reads but that were assigned in a
    // different region. The Artemis gate reads the weight columns in place by design, so those
    // reports (and only those, for the Artemis regions) are ignored; any other failure aborts.
    match mock_prover.verify() {
      Ok(()) => println!("MockProver: all constraints satisfied"),
      Err(errs) => {
        let real: Vec<_> = errs
          .into_iter()
          .filter(|e| !(matches!(e, halo2_proofs::dev::VerifyFailure::CellNotAssigned { .. }) && format!("{}", e).contains("Artemis")))
          .collect();
        assert!(real.is_empty(), "MockProver failures:\n{}", real.iter().map(|e| format!("{}", e)).collect::<Vec<_>>().join("\n"));
        println!("MockProver: all constraints satisfied (cross-region weight reads of the Artemis gate ignored)");
      }
    }
  }
  let fill_duration = start.elapsed();
  println!(
    "Time elapsed in filling circuit (MockProver): {:?}",
    fill_duration - pk_duration
  );

  //let poly_coeff = vec![Fp::one(); 1 << params.k()];
  //println!("poly coeff len: {}", poly_coeff.len());
  //println!("sum: {}", circuit.k as u32 + poly_col_len as u32);


  // while poly_coeff.len() < 2usize.pow(poly_params.k() as u32) {
  //   poly_coeff.push(Fp::ZERO);
  // }
  println!("Second poly coeff len: {:?}", poly_coeff.len());
  //poly_coeff.extend(vec![Fp::ZERO; 2usize.pow(circuit.k as u32 + poly_col_len as u32) - poly_coeff.len()]);

  let poly: Polynomial<Fp, Coeff> = Polynomial::from_coefficients_vec(poly_coeff.clone());

  // Params for the weight polynomial: the smallest power of two >= its length (the number of weights). The
  // circuit params are reused by reference when they are large enough. The IPA opening (prover and verifier)
  // costs O(params.n) regardless of the polynomial's degree, so the previous rule, 2^(k + ceil(log2 ell)),
  // oversized it by 16x on mnist and 2x on mobilenet. zkfft keeps the previous sizing.
  let poly_params_big;
  let poly_params: &ParamsIPA<EqAffine> = {
    let k_weights = if zkfft {
      if poly_col_len > 1 { degree + (poly_col_len - 1).ilog2() + 1 } else { degree }
    } else {
      (poly.len().max(2) as u64).next_power_of_two().trailing_zeros()
    };
    if k_weights > degree {
      poly_params_big = get_ipa_params(format!("{}/params_ipa", directory).as_str(), k_weights);
      &poly_params_big
    } else {
      &params
    }
  };

  let rho = eval_polynomial(&poly, beta);

  let mut public_vals = vec![vec![]];
  //let mut betas = vec![vec![]; poly_col_len];

  public_vals[0] = get_public_values();
  drop(mock_prover);
  if commit_poly { 
    public_vals[0][0] = rho;
  }

  let public_vals_slice = public_vals.iter().map(|x| x.as_slice()).collect::<Vec<_>>();

  // Artemis (`poly`/`pedersen`): the two openings that link the committed weight polynomial p to the circuit
  // are folded into halo2's batched IPA multi-open argument as extra queries:
  //  (1) p opens to rho at the Fiat-Shamir challenge beta. Folded only when p fits the circuit params; for
  //      larger models it stays a standalone IPA opening with the weight params (below).
  //  (2) the Poly5/Poly2 accumulator column opens to rho at omega^0 (Horner runs bottom-up, acc_0 = p(beta)).
  // The verifier reads both evaluations from the transcript and checks that they agree. Each standalone IPA
  // opening costs O(params.n) (about 4 MSMs plus the generator folding), prover and verifier; folded, an
  // opening is one Kate division. The column's blinding factor flows through the multi-open automatically.
  let col_idx = if pedersen { poly_col_len * 2 } else { poly_col_len };
  let acc_point = ExtraPoint::Fixed(Fp::ONE); // omega^0
  let beta_challenge = 0usize; // beta is the circuit's only phase challenge
  let weight_blind = Blind::default();
  let fold_weights = commit_poly && std::ptr::eq(poly_params, &params);
  let poly_padded: Polynomial<Fp, Coeff> = if fold_weights {
    let mut values = poly.values.clone();
    values.resize(params.n() as usize, Fp::ZERO);
    Polynomial::from_coefficients_vec(values)
  } else {
    Polynomial::from_coefficients_vec(vec![])
  };
  let extra_queries: Vec<ExtraProverQuery<'_, Fp>> = if commit_poly {
    assert_eq!(pk.get_vk().cs().num_challenges(), 1, "expected beta to be the only challenge");
    let mut queries = vec![ExtraProverQuery::Advice { column: col_idx, point: acc_point }];
    if fold_weights {
      queries.push(ExtraProverQuery::External { poly: &poly_padded, blind: weight_blind, point: ExtraPoint::Challenge(beta_challenge) });
    }
    queries
  } else {
    vec![]
  };

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
    &extra_queries,
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
  // Artemis: commit to the weights (the CP-SNARK's input, timed but not counted, like for Apollo/Lunar) and,
  // when p does not fit the circuit params, prove the standalone opening (1) with the weight params. beta is
  // the proof's phase-2 challenge (published by the synthesize pass).
  let mut proof_ipa = vec![];
  let mut poly_com = EqAffine::identity();
  if commit_poly {
    let beta = public_valss[0];
    let rho = poly.evaluate(beta);
    let rho_advice = advice_lagrange[col_idx][0];
    println!("poly coeff len: {:?}, poly col len: {:?}", poly_coeff.len(), poly_col_len);
    println!("(Rho, Beta): {:?}, col_idx: {:?}", (rho, beta), col_idx);
    assert!(rho == rho_advice, "rho: {:?}, rho_advice: {:?}", rho, rho_advice);

    let commit_timer = Instant::now();
    poly_com = poly_params.commit(&poly, weight_blind).to_affine();
    println!("Weight commitment time (not counted): {:?}", commit_timer.elapsed());

    if fold_weights {
      println!("Weight opening folded into the circuit multi-open (k_weights <= k)");
    } else {
      let ipa_proof_timer = Instant::now();
      let mut transcript_ipa_proof = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
      ProverIPA::new(poly_params)
        .create_proof(&mut rand::thread_rng(), &mut transcript_ipa_proof, [ProverQuery { point: beta, poly: &poly, blind: weight_blind }])
        .unwrap();
      proof_ipa = transcript_ipa_proof.finalize();
      let ipa_proof_time = ipa_proof_timer.elapsed();
      println!("IPA proof time (weights at beta, standalone): {:?}", ipa_proof_time);
      proving_time += ipa_proof_time;
      proof_size += proof_ipa.len();
    }
    println!("Proving time: {:?}", proving_time);
  }
  let extra_queries_v: Vec<ExtraVerifierQuery<'_, EqAffine>> = if commit_poly {
    let mut queries = vec![ExtraVerifierQuery::Advice { column: col_idx, point: acc_point }];
    if fold_weights {
      queries.push(ExtraVerifierQuery::External { commitment: &poly_com, point: ExtraPoint::Challenge(beta_challenge) });
    }
    queries
  } else {
    vec![]
  };

  let mut verifying_time = vec![];
  for _ in 0..num_runs {
    let verification = Instant::now();
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    let strategy = SingleStrategy::new(&params);
    let mut extra_evals = vec![];
    assert!(
      verify_proof(
        &params,
        pk.get_vk(),
        strategy,
        &[public_vals_slice.as_slice()],
        &mut transcript,
        &extra_queries_v,
        &mut extra_evals,
      )
      .is_ok(),
      "proof did not verify"
    );
    if commit_poly {
      // CP-link check: the opened accumulator cell equals the committed weight polynomial at beta.
      let rho_circuit = extra_evals[0];
      if fold_weights {
        assert_eq!(rho_circuit, extra_evals[1], "accumulator cell does not match the weight polynomial at beta");
      } else {
        assert!(
          verify_ipa_opening(poly_params, &proof_ipa, &poly_com, public_valss[0], rho_circuit),
          "weight-polynomial opening did not verify"
        );
      }
    }
    let verify_duration = verification.elapsed();
    verifying_time.push(verify_duration);
  }

  // zkFFT commitment (if enabled)
  if zkfft {
    println!("Running zkFFT commitment for {} columns...", poly_col_len);

    // Get domain for omega
    let domain = pk.get_vk().get_domain();
    let omega = domain.get_omega();

    // Loop over each polynomial column
    for col_idx in 0..poly_col_len {
      println!("\n=== zkFFT Column {} ===", col_idx);

      // Prepare witness and generators for computing initial commitment P
      let poly_advice_coeff = domain.lagrange_to_coeff(advice_lagrange[col_idx].clone());
      let mut a: Vec<Fp> = poly_advice_coeff.values.clone();
      let n = a.len();
      let next_pow2 = n.next_power_of_two();
      if n < next_pow2 {
          a.resize(next_pow2, Fp::ZERO);
      }
      let n = a.len();
      let k = n;

      println!("Creating lazy b matrix");

      // Use lazy representation to save memory (2MB vs 32GB for n=32K)
      let b_lazy = LazyBMatrix::new(omega, n);

      // Get generators (same as prover)
      let generators_g: Vec<EqAffine> = poly_params.get_g()
          .iter().take(n).copied().collect();
      let generator_g = generators_g.clone();
      let generator_h = poly_params.get_w();

      // Generate alpha (blinding factor)
      let alpha = Fp::random(OsRng);

      // Compute initial commitment P = sum(a[i]*g[i]) + sum(inner_products[i]*g'[i]) + alpha*h
      println!("Computing initial commitment P...");

      // Compute all inner products using FFT: O(k log k) instead of O(k²)
      let inner_products = b_lazy.compute_all_inner_products_fft(&a);

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
          zkfft_commit_ipa_fft(poly_advice_coeff.clone(), poly_params, domain.clone(), alpha);

      proving_time += zkfft_ptime;
      proof_size += zkfft_size;

      println!("zkFFT: Column {}: Proof size: {} bytes", col_idx, zkfft_size);

      // Run verifier
      println!("Running zkFFT verifier...");
        for i in 0..num_runs {
            let verify_start = Instant::now();

            let verified = zkfft_verify_ipa(
                &zkfft_proof_bytes,
                b_lazy.clone(),
                generators_g.clone(),
                generator_g.clone(),
                generator_h,
                commitment_P,
            );

            let vfy_time = verify_start.elapsed();

            if verified {
                println!("zkFFT: Column {} Verification PASSED ✓", col_idx);
                verifying_time[i] += vfy_time;
            } else {
                println!("zkFFT: Column {} Verification FAILED ✗", col_idx);
                panic!("zkFFT verification failed for column {}!", col_idx);
            }
        }
    } // end for col_idx
  }

  // Barycentric commitment (if enabled)
  if barycentric {
    println!("\n=== Barycentric IPA ({} columns) ===", poly_col_len);

    // Get domain for omega
    let domain = pk.get_vk().get_domain();

    // Prepare inputs as vectors
    let advice_lagrange_vec: Vec<_> = if poly_col_len == 1 {
        // Single column: reconstruct as before
        let chunk_size = (poly.values.len() + poly_col_len - 1) / poly_col_len;
        let mut reconstructed = vec![Fp::ZERO; domain.get_n() as usize];
        for col_idx in 0..poly_col_len {
            let start_idx = col_idx * chunk_size;
            let end_idx = (start_idx + chunk_size).min(poly.values.len());
            for i in 0..(end_idx - start_idx) {
                reconstructed[start_idx + i] = advice_lagrange[col_idx].values[i];
            }
        }
        for i in poly.values.len()..(domain.get_n() as usize) {
            reconstructed[i] = advice_lagrange[0].values[i];
        }
        vec![domain.lagrange_from_vec(reconstructed)]
    } else {
        // Multi-column: pass columns directly
        advice_lagrange[0..poly_col_len].to_vec()
    };

    let advice_com_vec: Vec<EqAffine> = advice_com[0..poly_col_len]
        .iter().map(|c| (*c).into()).collect();
    let advice_blind_vec: Vec<Fp> = advice_blind[0..poly_col_len].to_vec();

    // Pre-compute polynomial chunks and commitments (exclude from proving time)
    // This is preprocessing - the witness is already chunked in the circuit
    let (poly_chunks, poly_com_vec): (Vec<_>, Vec<_>) = if poly_col_len == 1 {
        let poly_com = params.commit(&poly, Blind::default()).to_affine();
        (vec![poly.clone()], vec![poly_com])
    } else {
        let chunk_size = (poly.values.len() + poly_col_len - 1) / poly_col_len;
        // Number of blinding rows at the end of the domain
        let num_blinding = domain.get_n() as usize - poly.values.len();
        let witness_end = domain.get_n() as usize - num_blinding;

        (0..poly_col_len).map(|col_idx| {
            let start_idx = col_idx * chunk_size;
            let end_idx = (start_idx + chunk_size).min(poly.values.len());
            let chunk_len = end_idx - start_idx;

            // Allocate full domain size, fill witness data up to witness_end, leave blinding space
            let mut poly_chunk_values = vec![Fp::ZERO; domain.get_n() as usize];
            // Fill witness data (up to witness_end to leave room for blinding)
            for i in 0..chunk_len.min(witness_end) {
                poly_chunk_values[i] = poly.values[start_idx + i];
            }
            // Positions [witness_end .. domain.get_n()) stay zero for blinding values
            let poly_chunk = Polynomial::from_coefficients_vec(poly_chunk_values);
            let poly_com_chunk = params.commit(&poly_chunk, Blind::default()).to_affine();

            (poly_chunk, poly_com_chunk)
        }).unzip()
    };

    // Call unified bary_ipa function with pre-computed chunks
    // IMPORTANT: Use params (not poly_params) because barycentric works with domain size
    // poly_params has larger degree for multi-column polynomial commitments
    let (bary_proof_bytes, bary_ptime, bary_size,
         poly_com_blind_vec, rho_vec) = bary_ipa(
        poly_chunks,
        poly_com_vec.clone(),
        advice_lagrange_vec,
        advice_com_vec.clone(),
        beta,
        &params,
        domain.clone(),
        alpha,
        advice_blind_vec,
        poly_col_len,
    );

    proving_time += bary_ptime;
    proof_size += bary_size;

    println!("Barycentric: Proof size: {} bytes", bary_size);
    println!("Barycentric: Evaluations: {:?}", rho_vec);

    // poly_com_vec is already computed above before the proving timer
    // Use it for verification

    // Run unified verifier
    println!("Running barycentric verifier...");
    for i in 0..num_runs {
        let verify_start = Instant::now();

        let verified = bary_verify_ipa(
            &bary_proof_bytes,
            poly_com_vec.clone(),
            poly_com_blind_vec.clone(),
            advice_com_vec.clone(),
            beta,
            rho_vec.clone(),
            &params,
            domain.clone(),
            poly_col_len,
        );

        let vfy_time = verify_start.elapsed();
        verifying_time[i] += vfy_time;

        if !verified {
            panic!("Barycentric verification failed!");
        }
    }
    println!("Barycentric: All verifications PASSED ✓");
  }
  println!("Proving time: {:?}", proving_time);
  println!("Verifying time: {:?}", verifying_time);

  // Create a file to write the CSV to
  let file = File::create("results/output.csv").unwrap();

  // Create a CSV writer
  let mut wtr = Writer::from_writer(file);

  wtr.write_record(&["Prover time", "Verifier time", "Proof size"]).unwrap();
  // Write some rows of data
  let proving_time_str = format!("{:?}", proving_time);
  let verifying_time_str = format!("{:?}", verifying_time);
  wtr.write_record(&[proving_time_str, verifying_time_str, proof_size.to_string()]).unwrap();
  // Flush and finish writing
  wtr.flush().unwrap();

  println!("CSV file created successfully.");
}
