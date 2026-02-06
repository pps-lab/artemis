#![allow(non_camel_case_types)]
use commitment::CommitmentScheme;
use commitment::Params;
use commitment::ParamsProver;
use commitment::Verifier;
use commitment::MSM;
use ff::FromUniformBytes;
use ff::WithSmallOrderMulGroup;
use halo2_proofs::arithmetic::eval_polynomial;
use halo2_proofs::arithmetic::kate_division;
use halo2_proofs::helpers::SerdeCurveAffine;
use halo2_proofs::transcript::Blake2bWrite;
use halo2_proofs::transcript::{Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer, TranscriptRead, TranscriptWrite, Transcript, EncodedChallenge};
use halo2_proofs::poly::commitment::{Prover, Blind};
use halo2_proofs::transcript::Blake2bRead;
use halo2curves::bn256::Bn256;
use halo2curves::bn256::Fr;
use group::{Curve, Group};
use halo2_proofs::poly::*;
use halo2_proofs::plonk::Error;
use halo2_proofs::plonk::*;
use halo2curves::bn256::Gt;
use halo2curves::bn256::G1;
use group::prime::PrimeCurveAffine;
use halo2curves::pairing::{Engine, MultiMillerLoop};
use group::ff::Field;
use halo2curves::pasta::pallas::Scalar;
use kzg::commitment::KZGCommitmentScheme;
use kzg::commitment::ParamsKZG;
use kzg::msm::DualMSM;
use kzg::msm::MSMKZG;
use kzg::msm::MSMKZG2;
use kzg::multiopen::ProverSHPLONK;
use kzg::multiopen::VerifierSHPLONK;
use kzg::strategy::{AccumulatorStrategy};
use halo2_proofs::poly::VerificationStrategy;
use num_traits::Pow;
use rand::Rng;
use rand::RngCore;
use rand_core::OsRng;
use zkml::gadgets::square;
use zkml::utils::proving_kzg::get_kzg_params;
use std::ops::MulAssign;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::time::Instant;
use std::env::{self, args, Args};
use std::fs;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use core::fmt::Debug;
use zkml::utils::helpers::poly_divmod;
use rayon::slice::ParallelSlice;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

#[derive(Clone, Debug)]
pub struct CS2_PP<P: Engine> {
    // putting all fields as public. May need to be changed later
    pub deg_bound: usize,
    pub g1: P::G1,
    pub g2: P::G2,
    pub pws_g1: Vec<P::G1>,
    pub pws_g2: Vec<P::G2>,
    pub alpha_g2: P::G2,
    pub pws_rando: Vec<P::G1>,
}

pub fn eval_on_pnt_in_grp<G: Curve, E: Engine<G1 = G> + Debug>(
    p: &Polynomial<E::Scalar, Coeff>,
    pnt_pws_in_grp: &Vec<G>,
) -> G {
    //let mut g_vector_affine;
    //G::batch_normalize(pnt_pws_in_grp.as_mut(), g_vector_affine);
    // from: multi_scalar_mul following update in ark-ec
    let mut msm: MSMKZG<E> = MSMKZG::new();
    for (base, scalar) in pnt_pws_in_grp.iter().zip(p.values.clone()) {
        msm.append_term(scalar, *base); 
    }
    let comm = msm.eval();

    comm
}

pub fn eval_on_pnt_in_grp2<G: Curve, E: Engine<G2 = G> + Debug>(
    p: &Polynomial<E::Scalar, Coeff>,
    pnt_pws_in_grp: &Vec<G>,
) -> G {
    //let mut g_vector_affine;
    //G::batch_normalize(pnt_pws_in_grp.as_mut(), g_vector_affine);
    // from: multi_scalar_mul following update in ark-ec
    let mut msm: MSMKZG2<E> = MSMKZG2::new();
    for (base, scalar) in pnt_pws_in_grp.iter().zip(p.values.clone()) {
        msm.append_term(scalar, *base); 
    }
    let comm = msm.eval();

    comm
}

pub fn powers<F: Field>(base: F) -> impl Iterator<Item = F> {
    std::iter::successors(Some(F::ONE), move |power| Some(base * power))
}

pub fn gen_pows<Fr: WithSmallOrderMulGroup<3>>(alpha: Fr, t: usize) -> Vec<Fr> {
    let mut powers_of_alpha = Vec::<Fr>::with_capacity(t + 1);
    powers_of_alpha.push(Fr::ONE);
    let mut power = alpha;

    for _ in 0..t {
        powers_of_alpha.push(power);
        power *= &alpha;
    }
    powers_of_alpha
}

fn fast_product_parallel<F: WithSmallOrderMulGroup<3>>(polys: &[Polynomial<F, Coeff>]) -> Polynomial<F, Coeff> {
    if polys.is_empty() {
        return Polynomial::from_coefficients_vec(vec![F::ONE]);
    }

    // Group polynomials into chunks and compute products within each chunk
    let chunk_size = (polys.len() as f64).sqrt() as usize + 1;
    let chunk_products: Vec<_> = polys
        .par_chunks(chunk_size)
        .map(|chunk| chunk.iter().fold(Polynomial::from_coefficients_vec(vec![F::ONE]), |a, b| a * b))
        .collect();

    // Combine chunk products
    chunk_products.into_par_iter().reduce(
        || Polynomial::from_coefficients_vec(vec![F::ONE]),
        |a, b| a * &b
    )
}

fn fast_product<F: WithSmallOrderMulGroup<3>>(polys: &[Polynomial<F, Coeff>]) -> Polynomial<F, Coeff> {
    if polys.is_empty() {
        return Polynomial::from_coefficients_vec(vec![F::ONE]);
    }
    if polys.len() == 1 {
        return polys[0].clone();
    }
    let mid = polys.len() / 2;
    let left = fast_product(&polys[..mid]);
    let right = fast_product(&polys[mid..]);
    left* &right
}

pub fn vanishing_on_set<F: Field + WithSmallOrderMulGroup<3>>(set: &[F]) -> Polynomial<F, Coeff> {
    let mut vanishing = Polynomial::<F, Coeff>{
        values: vec![F::ONE],
        _marker: PhantomData,
    };

    let mut iter = 0;
    let mut polys = vec![];
    for point in set {
        vanishing = vanishing.naive_mul(&Polynomial::<F, Coeff>{
            values: vec![-*point, F::ONE],
            _marker: PhantomData,
        });
        if iter % 1000 == 0 {
            polys.push(vanishing);
            vanishing = Polynomial::<F, Coeff>::from_coefficients_vec(vec![F::ONE]);
        }
        iter += 1;
    }
    polys.push(vanishing);
    //vanishing = polys.iter().fold(Polynomial::<F, Coeff>::from_coefficients_vec(vec![F::ONE]), |a, b| a * b);
    vanishing = fast_product_parallel(&polys);
    vanishing
}

fn setup_pp<P: Engine<Scalar: WithSmallOrderMulGroup<3>>, R: Rng + Clone>(rng: &mut R, t: usize) -> CS2_PP<P> {
    let alpha = P::Scalar::random(rng.clone());
    let powers_of_alpha = gen_pows(alpha, t);
    let g1 = P::G1::random(rng.clone());
    let pws_g1 = vec![g1; t].iter().zip(powers_of_alpha.clone()).map(|(point, scalar)| point.mul(scalar)).collect::<Vec<_>>();

    let g2 = P::G2::random(rng.clone());
    let pws_g2 = vec![g2; t].iter().zip(powers_of_alpha.clone()).map(|(point, scalar)| point.mul(scalar)).collect::<Vec<_>>();

    let mut alpha_g2 = g2.clone();
    alpha_g2.mul_assign(alpha);
    let rando = P::Scalar::random(rng);
    let powers_of_alpha_rando: Vec<_> = powers_of_alpha.iter().map(|x| g1.mul(rando.mul(x))).collect();
    CS2_PP {
        deg_bound: t,
        g1, 
        g2, 
        pws_g1, 
        pws_g2, 
        alpha_g2, 
        pws_rando: powers_of_alpha_rando
    }
}

// Define necessary traits and structures based on ff

fn interpolate<F: WithSmallOrderMulGroup<3>> (vals: Vec<F>, domain: EvaluationDomain<F>) -> Polynomial<F, Coeff> {
    domain.lagrange_to_coeff(domain.lagrange_from_vec(vals))
}

fn keygen1<E: Engine<Scalar: WithSmallOrderMulGroup<3>>>(t: u32) -> CS2_PP<E> {
    let mut rng = OsRng;
    setup_pp(&mut rng, t as usize)
}

fn keygen2<E: Engine<Scalar: WithSmallOrderMulGroup<3>>>(t: u32) -> CS2_PP<E> {
    keygen1(t)
}

fn setup<E: Engine<Scalar: WithSmallOrderMulGroup<3>, G1Affine: SerdeCurveAffine, G2Affine: SerdeCurveAffine> + Debug>
(
    col_size: u32,
    witness_size: usize,
    l: usize,
    params: ParamsKZG<E>
) -> (
    //CS2_PP<E>,
    ParamsKZG<E>,
    EvaluationDomain<E::Scalar>,
    Vec<E::Scalar>,
    Vec<Polynomial<E::Scalar, Coeff>>,
    Polynomial<E::Scalar, Coeff>,
    Polynomial<E::Scalar, Coeff>,
    Vec<Polynomial<E::Scalar, Coeff>>,
    Vec<E::G2>,
    Vec<E::G2>
) {
    let setuptimer = Instant::now();

    let HH = EvaluationDomain::<E::Scalar>::new(1, col_size);
    //let HH = Radix2EvaluationDomain::new(col_size).unwrap();
    let n = 2u32.pow(col_size);
    let rng = OsRng;
    let size = witness_size / l;
    println!("Witness size: {}, Columns size {}, Columns {}", size, col_size, l);
    //let ck = keygen2::<E>(n * 2);
    let HH_vals = powers(HH.get_omega()).take(n as usize).collect::<Vec<_>>();
    let thetas = (0..l).map(|x| HH_vals[x * size]).collect::<Vec<_>>();
    let params_time = setuptimer.elapsed();
    println!("SETUP: Parameters time: {:?}", params_time);

    let mut vanishing = vec![E::Scalar::ZERO; HH.get_n() as usize + 1];
    vanishing[0] = -E::Scalar::ONE;
    vanishing[HH.get_n() as usize] = E::Scalar::ONE;
    let vanishing_poly = HH.coeff_from_vec(vanishing);

    let zs = (0..l).map(|x| vanishing_on_set(&HH_vals[x * size..(x + 1) * size].to_vec())).collect::<Vec<_>>();
    let z_v = zs[0].clone();
    let z_last = vanishing_on_set(&HH_vals[l * size..].to_vec());

    //println!("Vanishing poly: {:?}", vanishing_poly);
    //let (z_whole, r_whole) = vanishing_poly.divide_with_q_and_r(&Polynomial::from(z_last.clone())).unwrap();
    let (z_whole, r_whole) = poly_divmod(&vanishing_poly, &z_last);  
    //let z_whole = z_last.clone();
    assert!(r_whole.is_zero(), "value: {:?}", r_whole);

    let mut zhats = (0..l).map(|i| {
        let mut vals = vec![E::Scalar::ZERO; i * size];
        let ones = vec![E::Scalar::ONE; size];
        vals.extend(ones);
        let poly = HH.lagrange_to_coeff(HH.lagrange_from_vec(vals));
        //println!("Poly: {:?}", poly);
        poly
    }).collect::<Vec<_>>();

    let polynomial_time = setuptimer.elapsed();
    println!("SETUP: Polynomials time: {:?}", polynomial_time - params_time);
    //let z_coms = zs.iter().map(|z| eval_on_pnt_in_grp2::<E::G2, E>(&z, &ck.pws_g2)).collect::<Vec<_>>();
    let z_coms = zs.iter().map(|z| params.commit_g2(&z)).collect::<Vec<_>>();
    zhats.push(z_whole);
    //let zhat_coms = zhats.iter().map(|z| eval_on_pnt_in_grp2::<E::G2, E>(&z, &ck.pws_g2)).collect::<Vec<_>>();
    let zhat_coms = zhats.iter().map(|z| params.commit_g2(z)).collect::<Vec<_>>();
    let commitments_time = setuptimer.elapsed();
    println!("SETUP: Commitments time: {:?}", commitments_time - polynomial_time);
    println!("SETUP: Total time: {:?}", setuptimer.elapsed());
    (params, HH, thetas, zs, z_v, z_last, zhats, z_coms, zhat_coms)
}

fn cplink1<E: Engine<Scalar: WithSmallOrderMulGroup<3> + Ord, G1Affine: SerdeCurveAffine, G2Affine: SerdeCurveAffine> + Debug>(
    uprimes: Vec<Polynomial<E::Scalar, Coeff>>,
    zs: Vec<Polynomial<E::Scalar, Coeff>>,
    zhats: Vec<Polynomial<E::Scalar, Coeff>>,
    poly: Polynomial<E::Scalar, Coeff>,
    params: ParamsKZG<E>,
    //ck: CS2_PP<E>,
    z_last: Polynomial<E::Scalar, Coeff>,
    HH: EvaluationDomain<E::Scalar>,
) -> (
    Vec<E::G1>,
    Vec<E::G1>,
    Vec<E::G1>,
    E::G1,
    E::G1,
    E::G1,
    E::Scalar,
    E::Scalar
) {
    let cplink1_timer = Instant::now();
    let qus_timer = Instant::now();
    let l = uprimes.len();
    let HH_vals: Vec<E::Scalar> = powers(HH.get_omega()).take(10).collect();

    let rng = OsRng;
    let (uhats, qs) = {
        let mut uvec = vec![];
        let mut qvec = vec![];
        for i in 0..l {
            //let (q, r) = uprimes[i].clone().divide_with_q_and_r(&zs[i].clone()).unwrap();
            let (q, r) = poly_divmod(&uprimes[i], &zs[i]);    
            uvec.push(r);
            qvec.push(q);

        }
        (uvec, qvec)
    };

    let bigqu_timer = Instant::now();
    let sum = {
        let mut sum = Polynomial::<E::Scalar, Coeff>::zero();
        for i in 0..l {
            sum = sum + &(zhats[i].clone() * &uhats[i].clone());
        }
        sum
    };

    let mut vanishing = vec![E::Scalar::ZERO; HH.get_n() as usize + 1];
    vanishing[0] = -E::Scalar::ONE;
    vanishing[HH.get_n() as usize] = E::Scalar::ONE;
    let vanishing_poly = HH.coeff_from_vec(vanishing);

    let q = poly.clone() - &sum;
    //let (q, r) = q.divide_with_q_and_r(&zhats[l]).unwrap();
    let (q, r) = poly_divmod(&q, &zhats[l]);
    assert!(r.is_zero(), "R value: {:?}, Q value: {:?}\n, ", r, q);


    let gammas = (0..l).map(|_| E::Scalar::ZERO).collect::<Vec<_>>();
    let os = (0..l).map(|_| E::Scalar::ZERO).collect::<Vec<_>>();
    let oprimes = (0..l).map(|i| zs[i].clone() * &Polynomial::from_coefficients_vec(vec![gammas[i]]) + &Polynomial::from_coefficients_vec(vec![os[i]])).collect::<Vec<_>>();

    let commit_timer = Instant::now();
    let bigc = params.commit_g1(&poly);
    //let bigc = eval_on_pnt_in_grp::<E::G1, E>(&poly, &ck.pws_g1);

    let chats = (0..l).map(|i| {
        //let ucom = eval_on_pnt_in_grp::<E::G1, E>(&uprimes[i], &ck.pws_g1);
        let ucom = params.commit_g1(&uprimes[i]);
        ucom
    }).collect::<Vec<_>>();

    let cprimes = (0..l).map(|i| {
        //let uhatcom = eval_on_pnt_in_grp::<E::G1, E>(&uhats[i], &ck.pws_g1);
        let uhatcom = params.commit_g1(&uhats[i]);
        //let oprimecom = eval_on_pnt_in_grp::<E::G1, E>(&oprimes[i], &ck.pws_rando);
        let oprimecom = params.commit_g1(&oprimes[i]);
        uhatcom + oprimecom
    }).collect::<Vec<_>>();

    let ds = (0..l).map(|i| {
        //let qcom = eval_on_pnt_in_grp::<E::G1, E>(&qs[i], &ck.pws_g1);
        let qcom = params.commit_g1(&qs[i]);
        //let gammacom = eval_on_pnt_in_grp::<E::G1, E>(&Polynomial::from_coefficients_vec(vec![gammas[i]]), &ck.pws_rando);
        let gammacom = params.commit_g1(&Polynomial::from_coefficients_vec(vec![gammas[i]]));
        qcom - gammacom
    }).collect::<Vec<_>>();

    let beta = Polynomial::<E::Scalar, Coeff>::zero();
    //let d = eval_on_pnt_in_grp::<E::G1, E>(&q, &ck.pws_g1) + eval_on_pnt_in_grp::<E::G1, E>(&beta, &ck.pws_rando);
    let d = params.commit_g1(&q) + params.commit_g1(&beta);
    //println!("Commitment time: {:?}", commit_timer.elapsed());

    let x = E::Scalar::random(&mut OsRng);
    let mut osum = Polynomial::<E::Scalar, Coeff>::zero();
    for i in 0..l {
        osum = osum + &(Polynomial::from_coefficients_vec(vec![os[i]]) * &zhats[i]);
    }
    let otilde = Polynomial::zero() - &(beta * &zhats[l] + &osum);
    let zz = otilde.evaluate(x);
    let (w, rem) = (otilde - &Polynomial::from_coefficients_vec(vec![zz])).divide_with_q_and_r(&Polynomial::from_coefficients_vec(vec![-x, E::Scalar::ONE])).unwrap();
    //let wcom = eval_on_pnt_in_grp::<E::G1, E>(&w, &ck.pws_rando);
    let wcom = params.commit_g1(&w);
    println!("CPLINK1: Prover time: {:?}", cplink1_timer.elapsed());
    (chats, ds, cprimes, wcom, bigc, d, x, zz)
}

fn cplink1_lite<E: Engine<Scalar: WithSmallOrderMulGroup<3> + Ord, G1Affine: SerdeCurveAffine, G2Affine: SerdeCurveAffine> + Debug> (
    u: Polynomial<E::Scalar, Coeff>,
    chat: E::G1,
    bigc: E::G1,
    z: Polynomial<E::Scalar, Coeff>,
    poly: Polynomial<E::Scalar, Coeff>,
    params: ParamsKZG<E>,
    //ck: CS2_PP<E>,
    HH: EvaluationDomain<E::Scalar>,
) -> (
    E::G1,
    E::G1,
    E::G1,
    E::G1,
    E::G1,
    E::G1,
    E::Scalar,
    E::Scalar
) {
    // let chat = params.commit_g1(&u);
    // let bigc = params.commit_g1(&poly);
    let cplink1_timer = Instant::now();
    let qus_timer = Instant::now();
    let HH_vals: Vec<E::Scalar> = powers(HH.get_omega()).take(10).collect();

    let rng = OsRng;
    let (uhat, q_small) = {
        //let (q, r) = uprimes[i].clone().divide_with_q_and_r(&zs[i].clone()).unwrap();
        let (q, r) = poly_divmod(&u, &z);    
        (r, q)
    };

    //let bigqu_timer = Instant::now();
    let sum = uhat.clone();

    let mut vanishing = vec![E::Scalar::ZERO; HH.get_n() as usize + 1];
    vanishing[0] = -E::Scalar::ONE;
    vanishing[HH.get_n() as usize] = E::Scalar::ONE;
    //let vanishing_poly = HH.coeff_from_vec(vanishing);

    let q = poly.clone() - &sum;
    //let (q, r) = q.divide_with_q_and_r(&zhats[l]).unwrap();
    let (q, r) = poly_divmod(&q, &z);
    assert!(r.is_zero(), "R value: {:?}, Q value: {:?}\n, ", r, q);

    let gamma = E::Scalar::ZERO;
    let o = E::Scalar::ZERO;
    let oprime = Polynomial::from_coefficients_vec(vec![E::Scalar::ZERO]);

    let cprime = {
        //let uhatcom = eval_on_pnt_in_grp::<E::G1, E>(&uhats[i], &ck.pws_g1);
        let uhatcom = params.commit_g1(&uhat);
        //let oprimecom = eval_on_pnt_in_grp::<E::G1, E>(&oprimes[i], &ck.pws_rando);
        let oprimecom = params.commit_g1(&oprime);
        uhatcom + oprimecom
    };

    let d_small = {
        //let qcom = eval_on_pnt_in_grp::<E::G1, E>(&qs[i], &ck.pws_g1);
        let qcom = params.commit_g1(&q_small);
        //let gammacom = eval_on_pnt_in_grp::<E::G1, E>(&Polynomial::from_coefficients_vec(vec![gammas[i]]), &ck.pws_rando);
        let gammacom = params.commit_g1(&Polynomial::from_coefficients_vec(vec![gamma]));
        qcom - gammacom
    };

    let beta = Polynomial::<E::Scalar, Coeff>::zero();
    //let d = eval_on_pnt_in_grp::<E::G1, E>(&q, &ck.pws_g1) + eval_on_pnt_in_grp::<E::G1, E>(&beta, &ck.pws_rando);
    let d = params.commit_g1(&q) + params.commit_g1(&beta);
    //println!("Commitment time: {:?}", commit_timer.elapsed());

    let x = E::Scalar::random(&mut OsRng);
    let mut osum = Polynomial::<E::Scalar, Coeff>::zero();
    osum = osum + &(Polynomial::from_coefficients_vec(vec![o]));

    let otilde = Polynomial::zero() - &(beta + &osum);
    let zz = otilde.evaluate(x);
    let (w, rem) = (otilde - &Polynomial::from_coefficients_vec(vec![zz])).divide_with_q_and_r(&Polynomial::from_coefficients_vec(vec![-x, E::Scalar::ONE])).unwrap();
    //let wcom = eval_on_pnt_in_grp::<E::G1, E>(&w, &ck.pws_rando);
    let wcom = params.commit_g1(&w);
    println!("CPLINK1: Prover time: {:?}", cplink1_timer.elapsed());
    (chat, d_small, cprime, wcom, bigc, d, x, zz)
}

fn verify1_lite<P: Engine + Debug> (
    cprime: P::G1,
    chat: P::G1,
    d_small: P::G1,
    params: ParamsKZG<P>,
    z_com: P::G2,
    wcom: P::G1,
    bigc: P::G1,
    d: P::G1,
    x: P::Scalar,
    z: P::Scalar,
) {
    let verifier_timer = Instant::now();
    // First CPlink1 check:
    let term1 = P::pairing(&chat.to_affine(), &params.g2());
    let term2 = P::pairing(&d_small.to_affine(), &z_com.to_affine());
    let term3 = P::pairing(&cprime.to_affine(), &params.g2());
    assert_eq!(term1, term2 + term3);

    // Second CPLink1 check:
    let term1 = P::pairing(&wcom.to_affine(), &params.commit_g2(&Polynomial::from_coefficients_vec(vec![-x, P::Scalar::ONE])).to_affine());
    let term2 = P::pairing(&bigc.to_affine(), &params.g2());
    let term3 = P::pairing(&d.to_affine(), &z_com.to_affine());

    let term4 = P::pairing(&cprime.to_affine(), &params.g2());

    let term5 = P::pairing(&params.commit_g1(&Polynomial::from_coefficients_vec(vec![z])).to_affine(), &params.s_g2());

    let lhs = term1 + term3 + term4 + term5;
    let rhs = term2;
    assert_eq!(lhs, rhs);
    println!("CPLINK1: Verifier time: {:?}", verifier_timer.elapsed());
}

fn cplink2<E: Engine<Scalar: WithSmallOrderMulGroup<3> + Ord + FromUniformBytes<64>, G1Affine: SerdeCurveAffine, G2Affine: SerdeCurveAffine> + Debug + MultiMillerLoop>(
    thetas: Vec<E::Scalar>, 
    HH: EvaluationDomain<E::Scalar>, 
    us: Vec<Polynomial<E::Scalar, Coeff>>, 
    z_v: Polynomial<E::Scalar, Coeff>,
    z_v_com: E::G1,
    u_coms: Vec<E::G1>,
    //ck: CS2_PP<E>,
    params: ParamsKZG<E>,
) -> Vec<Polynomial<<E as Engine>::Scalar, Coeff>> {
    let cplink2_timer = Instant::now();
    let rng = &mut OsRng;
    let l = thetas.len();
    let betas = (0..l).map(|_| Polynomial::<E::Scalar, Coeff>::from_coefficients_vec(vec![E::Scalar::random(rng.clone()); 3])).collect::<Vec<_>>();

    // theta shifts
    let theta_pows = thetas.iter().map(|theta| {
        let mut pows = vec![E::Scalar::ONE];
        for i in 1..HH.get_n() as usize {
            pows.push(pows[i - 1] * theta);
        }
        pows
    }).collect::<Vec<_>>();
    //println!("Theta pows: {:?}", theta_pows);

    let urands: Vec<Polynomial<E::Scalar, Coeff>> = us.iter().zip(betas).map(|(u, beta)| u.clone() + &(beta * &z_v)).collect::<Vec<_>>();
    let uprimes: Vec<Polynomial<E::Scalar, Coeff>> = urands.iter().zip(theta_pows.clone()).map(|(u, pows)| {
        let coeffs = u.clone().values;
        let shifted_coeffs = coeffs.iter().zip(pows.clone()).map(|(coeff, pow)| coeff.mul(pow.invert().unwrap())).collect::<Vec<_>>();
        let uprime = Polynomial::from_coefficients_vec(shifted_coeffs);
        uprime
    }).collect::<Vec<_>>();

    let hs: Vec<Polynomial<E::Scalar, Coeff>> = us.iter().zip(uprimes.clone()).zip(theta_pows).map(|((u, uprime), pow)| {
        let coeffs = uprime.values;
        let shifted_coeffs = coeffs.iter().zip(pow).map(|(coeff, pow)| *coeff * pow).collect::<Vec<_>>();
        let uprime_shift = Polynomial::from_coefficients_vec(shifted_coeffs);
        //let (h, r) = (u.clone() - &uprime_shift).divide_with_q_and_r(&z_v).unwrap();
        let (h, r) = poly_divmod(&(u.clone() - &uprime_shift), &z_v);
        //assert!(r.is_zero());
        h
    }).collect::<Vec<_>>();

    let h_coms = hs.iter().map(|h| params.commit_g1(&h)).collect::<Vec<_>>();
    let zv_com = params.commit_g1(&z_v);
    let u_coms = us.iter().map(|u| params.commit_g1(&u)).collect::<Vec<_>>();
    let uprime_coms = uprimes.iter().map(|uprime| params.commit_g1(&uprime).to_affine()).collect::<Vec<_>>();
    // eval hjs on rho 
    // eval zv on rho
    // eval ujs on rho
    // eval ujprimes on rho*theta 
    let rho = E::Scalar::random(rng.clone());
    let h_evals = hs.iter().map(|h| h.evaluate(rho)).collect::<Vec<_>>();
    let zv_eval = z_v.evaluate(rho);
    let u_evals = us.iter().map(|u| u.evaluate(rho)).collect::<Vec<_>>();
    let rho_thetas = thetas.iter().map(|theta| rho * theta).collect::<Vec<_>>();
    let uprime_evals = uprimes.iter().zip(rho_thetas.clone()).map(|(uprime, rho_theta)| uprime.evaluate(rho_theta)).collect::<Vec<_>>();
    let rho_evals = [h_evals.clone(), vec![zv_eval], u_evals.clone()].concat();
    let polys = [hs.clone(), vec![z_v], us].concat();
    let poly_coms = [h_coms, vec![zv_com], u_coms].concat();
    let mut poly_coms_affine =  vec![E::G1Affine::identity(); poly_coms.len()];
    E::G1::batch_normalize(poly_coms.as_slice(), &mut poly_coms_affine);

    let randos = polys.iter().map(|_| E::Scalar::random(rng.clone())).collect::<Vec<_>>();

    //Proofs at rho
    let (proof_1, evals_1) = create_proof::<
        KZGCommitmentScheme<E>, 
        ProverSHPLONK<E>, _, 
        Blake2bWrite<_, _, 
        Challenge255<_>>
    >(&params, polys.clone(), poly_coms_affine.clone(), vec![rho; polys.len()]);

    //Proofs at rho * theta
    let (proof_2, evals) = create_proof::<
        KZGCommitmentScheme<E>, 
        ProverSHPLONK<E>, _, 
        Blake2bWrite<_, _, 
        Challenge255<_>>
    >(&params, uprimes.clone(), uprime_coms.clone(), rho_thetas.clone());
    
    let prover_time = cplink2_timer.elapsed();
    println!("CPLINK2: Prover time: {:?}", prover_time);

    let verifier_params = params.verifier_params();

    verify::<
        KZGCommitmentScheme<E>,
        VerifierSHPLONK<_>,
        _,
        Blake2bRead<_, _, Challenge255<_>>,
        AccumulatorStrategy<_>,
    >(verifier_params, &proof_1[..], poly_coms_affine, vec![rho; polys.len()], rho_evals);


    verify::<
        KZGCommitmentScheme<E>,
        VerifierSHPLONK<_>,
        _,
        Blake2bRead<_, _, Challenge255<_>>,
        AccumulatorStrategy<_>,
    >(verifier_params, &proof_2[..], uprime_coms, rho_thetas, uprime_evals);

    let uprime_evals = uprimes.iter().zip(thetas).map(|(uprime, theta)| uprime.evaluate(rho*theta)).collect::<Vec<_>>();
    for i in 0..hs.len() {
        let lhs = h_evals[i] * zv_eval;
        let rhs: E::Scalar = u_evals[i] - uprime_evals[i];
        assert_eq!(lhs, rhs);
    }

    let verifier_time = cplink2_timer.elapsed() - prover_time;
    println!("CPLINK2: Verifier time: {:?}", verifier_time);

    //println!("CPLINK2: Total time: {:?}", cplink2_timer.elapsed());
    uprimes
}

// fn verify<E: Engine>() {}
pub fn div_by_vanishing<F: Field>(poly: Polynomial<F, Coeff>, roots: &[F]) -> Vec<F> {
    let poly = roots
        .iter()
        .fold(poly.values, |poly, point| kate_division(&poly, *point));

    poly
}

fn verify1<P: Engine + Debug> (
    cprimes: Vec<P::G1>,
    chats: Vec<P::G1>,
    ds: Vec<P::G1>,
    //ck: CS2_PP<P>,
    params: ParamsKZG<P>,
    z_coms: Vec<P::G2>,
    zhat_coms: Vec<P::G2>,
    wcom: P::G1,
    bigc: P::G1,
    d: P::G1,
    x: P::Scalar,
    z: P::Scalar,
) {
    let verifier_timer = Instant::now();
    let l = cprimes.len();
    // First CPlink1 check:
    for i in 0..l {
        let term1 = P::pairing(&chats[i].to_affine(), &params.g2());
        let term2 = P::pairing(&ds[i].to_affine(), &z_coms[i].to_affine());
        let term3 = P::pairing(&cprimes[i].to_affine(), &params.g2());
        assert_eq!(term1, term2 + term3);
    }

    // Second CPLink1 check:
    let term1 = P::pairing(&wcom.to_affine(), &params.commit_g2(&Polynomial::from_coefficients_vec(vec![-x, P::Scalar::ONE])).to_affine());
    let term2 = P::pairing(&bigc.to_affine(), &params.g2());
    let term3 = P::pairing(&d.to_affine(), &zhat_coms[l].to_affine());
    let term4 = {
        let mut sum = <P>::Gt::identity();
        for i in 0..l {
            sum = sum + P::pairing(&cprimes[i].to_affine(), &zhat_coms[i].to_affine());
        }
        sum
    };
    let term5 = P::pairing(&params.commit_g1(&Polynomial::from_coefficients_vec(vec![z])).to_affine(), &params.s_g2());

    let lhs = term1 + term3 + term4 + term5;
    let rhs = term2;
    assert_eq!(lhs, rhs);
    println!("CPLINK1: Verifier time: {:?}", verifier_timer.elapsed());
}

fn cplink (witness_size: usize, col_size: usize, l: usize, params: ParamsKZG<Bn256>) {
    type F = <Bn256 as Engine>::Scalar;
    type P = Bn256;
    let rng = &mut OsRng;
    let size = witness_size / l;
    let (params, HH, thetas, zs, z_v, z_last, zhats, z_coms, zhat_coms) = setup::<P>(col_size as u32, witness_size, l, params);
    // Prover
    // input polys 
    let prover_timer = Instant::now();
    let poly_timer = Instant::now();
    let vals = (0..l).map(|y| (y*size..(y+1)*size).map(|x| F::from(x as u64)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let coeffs = vals.iter().map(|x| HH.lagrange_from_vec(x.clone())).collect::<Vec<_>>();
    let us = coeffs.iter().map(|x| HH.lagrange_to_coeff(x.clone())).collect::<Vec<_>>();
    let poly_vals = vals.into_iter().flatten().collect::<Vec<_>>();
    let poly = HH.lagrange_to_coeff(HH.lagrange_from_vec(poly_vals));
    // println!("Construct polys: {:?}", poly_timer.elapsed());
    // CPLINK2 
    let uprimes = cplink2::<P>(thetas, HH.clone(), us, z_v, <Bn256 as Engine>::G1::identity(), vec![<Bn256 as Engine>::G1::identity(); 1], params.clone());
    // At this point we have uprimes which match poly 
    // CPLINK1 (uhats are uprimes in paper)
    let (chats, ds, cprimes, wcom, bigc, d, x, zz) = cplink1(uprimes, zs, zhats, poly, params.clone(), z_last, HH);
    //println!("TOTAL: Prover time: {:?}", prover_timer.elapsed());
    // Verify CPlink1
    verify1(cprimes, chats, ds, params, z_coms, zhat_coms, wcom, bigc, d, x, zz);
}

fn cplink_lite (witness_size: usize, col_size: usize, l: usize, params: ParamsKZG<Bn256>) {
    type F = <Bn256 as Engine>::Scalar;
    type P = Bn256;
    let rng = &mut OsRng;
    let size = witness_size / l;
    let (params, HH, thetas, zs, z_v, z_last, zhats, z_coms, zhat_coms) = setup::<P>(col_size as u32, witness_size, l, params);
    // Prover
    // input polys 
    let prover_timer = Instant::now();
    let poly_timer = Instant::now();
    let vals = (0..l).map(|y| (y*size..(y+1)*size).map(|x| F::from(x as u64)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let coeffs = vals.iter().map(|x| HH.lagrange_from_vec(x.clone())).collect::<Vec<_>>();
    let us = coeffs.iter().map(|x| HH.lagrange_to_coeff(x.clone())).collect::<Vec<_>>();
    let poly_vals = vals.into_iter().flatten().collect::<Vec<_>>();
    let poly = HH.lagrange_to_coeff(HH.lagrange_from_vec(poly_vals));
    // println!("Construct polys: {:?}", poly_timer.elapsed());
    // CPLINK2 
    //let uprimes = cplink2::<P>(thetas, HH.clone(), us, z_v, <Bn256 as Engine>::G1::identity(), vec![<Bn256 as Engine>::G1::identity(); 1], params.clone());
    // At this point we have uprimes which match poly 
    // CPLINK1 (uhats are uprimes in paper)
    let u = us[0].clone();
    let z = zs[0].clone();
    let chat = params.commit_g1(&u);
    let bigc = params.commit_g1(&poly);
    let (chats, d_small, cprime, wcom, bigc, d, x, zz) = cplink1_lite(u, chat, bigc, z, poly, params.clone(), HH);
    //println!("TOTAL: Prover time: {:?}", prover_timer.elapsed());
    // Verify CPlink1
    verify1_lite(cprime, chats, d_small, params, z_coms[0], wcom, bigc, d, x, zz);
}
fn verify<
        'a,
        'params,
        Scheme: CommitmentScheme,
        V: Verifier<'params, Scheme>,
        E: EncodedChallenge<Scheme::Curve>,
        T: TranscriptReadBuffer<&'a [u8], Scheme::Curve, E>,
        Strategy: VerificationStrategy<'params, Scheme, V, Output = Strategy>,
    >(
        params: &'params Scheme::ParamsVerifier,
        proof: &'a [u8],
        poly_coms: Vec<Scheme::Curve>,
        points: Vec<Scheme::Scalar>,
        evals: Vec<Scheme::Scalar>,
    ) {
        let verifier = V::new(params);

        let mut transcript = T::init(proof);

        let queries = poly_coms.iter().zip(evals).zip(points).map(|((poly_com, eval), point)| VerifierQuery::new_commitment(poly_com, point, eval)).collect::<Vec<_>>();

        let queries = queries.clone();

        {
            let strategy = Strategy::new(params);
            let strategy = strategy
                .process(|msm_accumulator: <V as Verifier<'params, Scheme>>::MSMAccumulator | {
                    verifier
                        .verify_proof(&mut transcript, queries.clone(), msm_accumulator)
                        .map_err(|_| Error::Opening)
                })
                .unwrap();
            //println!("Strat: {:?}", strategy.finalize());
            assert!(strategy.finalize());
        }
    }

fn create_proof<
    'params,
    Scheme: CommitmentScheme,
    P: Prover<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    T: TranscriptWriterBuffer<Vec<u8>, Scheme::Curve, E>,
>(
    params: &'params Scheme::ParamsProver,
    polys: Vec<Polynomial<Scheme::Scalar, Coeff>>,
    poly_coms: Vec<Scheme::Curve>,
    points: Vec<Scheme::Scalar>,
) -> (Vec<u8>, Vec<Scheme::Scalar>)
where
    Scheme::Scalar: WithSmallOrderMulGroup<3>,
{

    let mut transcript = T::init(vec![]);
    let evals = polys.iter().zip(points.clone()).map(|(poly, point)| eval_polynomial(poly, point)).collect::<Vec<_>>();
    let queries = polys.iter().zip(points).map(|(poly, point)| 
        ProverQuery{
        point, 
        poly, 
        blind: Blind::default()
    }).collect::<Vec<_>>();
    
    let prover = P::new(params);
    prover
        .create_proof(&mut OsRng, &mut transcript, queries)
        .unwrap();

    (transcript.finalize(), evals)
}

#[test]
fn test_stuff() {
    type F = <Bn256 as Engine>::Scalar;
    // let HH = EvaluationDomain::<F>::new(1, 10);
    // let omega = HH.get_omega();
    // let omega_pows = powers(omega).take(32).collect::<Vec<_>>();

    // let vals = (0..4).map(|x| F::from(x)).collect::<Vec<_>>();
    // let poly = HH.lagrange_from_vec(vals);
    // let poly_coeff = HH.lagrange_to_coeff(poly.clone());
    // let squared = poly_coeff.clone() * &poly_coeff;
    // for i in omega_pows[0..6].iter() {
    //     println!("Eval: {:?}", squared.evaluate(*i));
    // }
    // let vanishing1 = HH.coeff_from_vec(vec![-omega_pows[0], F::ONE]);
    // let vanishing2 = HH.coeff_from_vec(vec![-omega_pows[1], F::ONE]);
    // let vanishing_both = vanishing1.clone() * &vanishing2;

    // Prove commitment evals
    const K: u32 = 10;

    let params = ParamsKZG::<Bn256>::new(K);
    let mut rng = &mut OsRng;
    let polynomials = (0..4).map(|i| Polynomial::from_coefficients_vec((0..10).map(|j| F::from(10 * i + j)).collect())).collect::<Vec<_>>();
   //let polynomials = vec![Polynomial::ranvec![F::random(&mut rng); 10]); 4];
    for polynomial in polynomials.clone() {
        println!("Polynomial: {:?}", polynomial);
    }
    let poly_coms = polynomials.iter().map(|polynomial| params.commit_g1(polynomial).to_affine()).collect::<Vec<_>>();
    let point = F::random(&mut OsRng);

    let (proof, evals) = create_proof::<
        KZGCommitmentScheme<Bn256>,
        ProverSHPLONK<_>,
        _,
        Blake2bWrite<_, _, Challenge255<_>>,
    >(&params, polynomials.clone(), poly_coms.clone(), vec![point; polynomials.len()]);

    let verifier_params = params.verifier_params();

    verify::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<_>,
        _,
        Blake2bRead<_, _, Challenge255<_>>,
        AccumulatorStrategy<_>,
    >(verifier_params, &proof[..], poly_coms, vec![point; polynomials.len()], evals);
}
fn main() {
    let args: Vec<String> = env::args().collect();
    let pow: usize = args[1].parse().unwrap(); 
    let witness_size: usize = args[2].parse().unwrap(); 
    assert!(witness_size < 2usize.pow(pow as u32));
    let l: usize = args[3].parse().unwrap(); 
    let params = get_kzg_params("./params_kzg", pow as u32);
    cplink_lite(witness_size, pow, l, params);
}
#[test]
fn test_cplink() {
    let args: Vec<String> = env::args().collect();
    let pow: usize = args[4].parse().unwrap(); 
    let witness_size: usize = args[5].parse().unwrap(); 
    assert!(witness_size < 2usize.pow(pow as u32));
    let l: usize = args[6].parse().unwrap(); 
    //let witness_size = 10usize.pow(3);
    //let col_size = witness_size.next_power_of_two();
    //let l = 10;
    let params =  get_kzg_params("./params_kzg", pow as u32);
    cplink(witness_size, pow, l, params);
}

#[test]
fn test_cplink_lite() {
    let args: Vec<String> = env::args().collect();
    let pow: usize = 15;
    let witness_size: usize = 10000 ;
    assert!(witness_size < 2usize.pow(pow as u32));
    let l: usize = 1;
    //let witness_size = 10usize.pow(3);
    //let col_size = witness_size.next_power_of_two();
    //let l = 10;
    let params =  get_kzg_params("./params_kzg", pow as u32);
    cplink_lite(witness_size, pow, l, params);
}