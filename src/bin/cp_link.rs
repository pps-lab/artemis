#![allow(non_camel_case_types)]
use commitment::MSM;
use ff::WithSmallOrderMulGroup;
use halo2_proofs::arithmetic::kate_division;
use halo2_proofs::arithmetic::lagrange_interpolate;
use halo2curves::bn256::Bn256;
use halo2curves::bn256::Fr;
use group::{Curve, Group};
use halo2_proofs::poly::*;
use halo2_proofs::plonk::*;
use halo2curves::bn256::Gt;
use halo2curves::bn256::G1;
use halo2curves::pairing::{Engine, MultiMillerLoop};
use group::ff::Field;
use halo2curves::pasta::pallas::Scalar;
use kzg::msm::MSMKZG;
use kzg::msm::MSMKZG2;
use num_traits::Pow;
use rand::Rng;
use rand::RngCore;
use rand_core::OsRng;
use zkml::gadgets::square;
use std::ops::MulAssign;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::time::Instant;
use std::env::{self, args, Args};
use std::fs;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use core::fmt::Debug;


/// Multiplies a polynomial by a scalar.
fn poly_scalar_mul<F: Field>(a: F, p: &Polynomial<F, Coeff>) -> Polynomial<F, Coeff> {
    p.clone() * a
}

/// Adds two polynomials.
fn poly_add<F: Field>(u: &Polynomial<F, Coeff>, v: &Polynomial<F, Coeff>) -> Polynomial<F, Coeff> {
    u.clone() + v
}

/// Subtracts polynomial v from polynomial u.
fn poly_sub<F: Field>(u: &Polynomial<F, Coeff>, v: &Polynomial<F, Coeff>) -> Polynomial<F, Coeff> {
    u.clone() - v
}

/// Multiplies two polynomials using FFT.
fn poly_mul<F: Field + WithSmallOrderMulGroup<3>>(u: &Polynomial<F, Coeff>, v: &Polynomial<F, Coeff>) -> Polynomial<F, Coeff> {
    u.clone() * v
}

fn poly_deg<F: Field>(p: &Polynomial<F, Coeff>) -> usize {
    let norm = p.normalize();
    norm.values.len() - 1
}

/// Scales a polynomial by multiplying/dividing by x^n.
fn poly_scale<F: Field>(p: &Polynomial<F, Coeff>, n: isize) -> Polynomial<F, Coeff> {
    if n >= 0 {
        let mut coeffs = vec![F::ZERO; n as usize];
        coeffs.extend(p.values.clone());
        Polynomial::from_coefficients_vec(coeffs)
    } else {
        let n_abs = n.abs() as usize;
        Polynomial::from_coefficients_vec(p.values.iter().skip(n_abs).cloned().collect())
    }
}

fn poly_recip<F: Field + WithSmallOrderMulGroup<3>>(p: &Polynomial<F, Coeff>) -> Polynomial<F, Coeff> {
    let k = poly_deg(p) + 1;
    assert!(k > 0 && !(*p.values.last().unwrap() == F::ZERO) && (k & (k - 1)) == 0, "k must be a power of 2: {}", k);

    if k == 1 {
        return Polynomial::from_coefficients_vec(vec![p.values[0].invert().unwrap()]);
    }

    let q = poly_recip(&Polynomial::from_coefficients_vec(p.values[k/2..].to_vec()));
    let r = poly_sub(
        &poly_scale(&poly_scalar_mul(F::ONE + F::ONE, &q), 3 * k as isize / 2 - 2),
        &poly_mul(&poly_mul(&q, &q), p),
    );

    poly_scale(&r, -(k as isize) + 2)
}
/// Fast polynomial division u(x) / v(x).
fn poly_divmod<F: Field + WithSmallOrderMulGroup<3>>(u: &Polynomial<F, Coeff>, v: &Polynomial<F, Coeff>) -> (Polynomial<F, Coeff>, Polynomial<F, Coeff>) {
    let m = poly_deg(u);
    let n = poly_deg(v);

    // Ensure deg(v) is one less than some power of 2
    let nd = (n + 1).next_power_of_two() - 1 - n;
    let ue = poly_scale(u, nd as isize);
    let ve = poly_scale(v, nd as isize);

    let s = poly_recip(&ve);
    let q = poly_scale(&poly_mul(&ue, &s), -2 * poly_deg(&ve) as isize);

    // Handle case when m > 2n
    let q = if m > 2 * n {
        let t = poly_sub(&poly_scale(&Polynomial::from_coefficients_vec(vec![F::ONE]), 2 * poly_deg(&ve) as isize), &poly_mul(&s, &ve));
        let (q2, _) = poly_divmod(&poly_scale(&poly_mul(&ue, &t), -2 * poly_deg(&ve) as isize), &ve);
        poly_add(&q, &q2)
    } else {
        q
    };

    // Remainder, r = u - v * q
    let r = poly_sub(u, &poly_mul(v, &q));

    (q, r)
}

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

fn powers<F: Field>(base: F) -> impl Iterator<Item = F> {
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

pub fn vanishing_on_set<F: Field + WithSmallOrderMulGroup<3>>(set: &[F]) -> Polynomial<F, Coeff> {
    let mut vanishing = Polynomial::<F, Coeff>{
        values: vec![F::ONE],
        _marker: PhantomData,
    };//from_coefficients_slice(&[F::ONE]);

    let mut iter = 0;
    for point in set {
        let timer = Instant::now();
        vanishing = vanishing.naive_mul(&Polynomial::<F, Coeff>{
            values: vec![-*point, F::ONE],
            _marker: PhantomData,
        });
        if iter % 500 == 0 {
            //println!("One mult time : {:?}", timer.elapsed());
        }
        iter += 1;
    }
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

fn setup<E: Engine<Scalar: WithSmallOrderMulGroup<3>> + Debug>(
    col_size: u32,
    witness_size: usize,
    l: usize
) -> (
    CS2_PP<E>,
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
    println!("Size: {}, n: {}, col_size {}", size, n, col_size);
    let ck = keygen2::<E>(n * 2);
    let HH_vals = powers(HH.get_omega()).take(n as usize).collect::<Vec<_>>();
    let thetas = (0..l).map(|x| HH_vals[x * size]).collect::<Vec<_>>();

    println!("Generating stuff: {:?}", setuptimer.elapsed());
    let mut vanishing = vec![E::Scalar::ZERO; HH.get_n() as usize + 1];
    vanishing[0] = -E::Scalar::ONE;
    vanishing[HH.get_n() as usize] = E::Scalar::ONE;
    let vanishing_poly = HH.coeff_from_vec(vanishing);

    let zs = (0..l).map(|x| vanishing_on_set(&HH_vals[x * size..(x + 1) * size].to_vec())).collect::<Vec<_>>();
    let z_v = zs[0].clone();
    let z_last = vanishing_on_set(&HH_vals[l * size..].to_vec());
    for val in HH_vals[l * size..].iter() {
        //println!("Eval: {:?}", z_last.evaluate(*val));
    }
    for val in HH_vals {
        //println!("Eval vanishing: {:?}", vanishing_poly.evaluate(val));
    }
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
    println!("Polys time: {:?}", setuptimer.elapsed());
    let z_coms = zs.iter().map(|z| eval_on_pnt_in_grp2::<E::G2, E>(&z, &ck.pws_g2)).collect::<Vec<_>>();
    zhats.push(z_whole);
    let zhat_coms = zhats.iter().map(|z| eval_on_pnt_in_grp2::<E::G2, E>(&z, &ck.pws_g2)).collect::<Vec<_>>();

    println!("Setup time: {:?}", setuptimer.elapsed());
    (ck, HH, thetas, zs, z_v, z_last, zhats, z_coms, zhat_coms)
}

fn cplink1<E: Engine<Scalar: WithSmallOrderMulGroup<3>> + Debug>(
    uprimes: Vec<Polynomial<E::Scalar, Coeff>>,
    zs: Vec<Polynomial<E::Scalar, Coeff>>,
    zhats: Vec<Polynomial<E::Scalar, Coeff>>,
    poly: Polynomial<E::Scalar, Coeff>,
    ck: CS2_PP<E>,
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
    let bigc = eval_on_pnt_in_grp::<E::G1, E>(&poly, &ck.pws_g1);

    let chats = (0..l).map(|i| {
        let ucom = eval_on_pnt_in_grp::<E::G1, E>(&uprimes[i], &ck.pws_g1);
        ucom
    }).collect::<Vec<_>>();

    let cprimes = (0..l).map(|i| {
        let uhatcom = eval_on_pnt_in_grp::<E::G1, E>(&uhats[i], &ck.pws_g1);
        let oprimecom = eval_on_pnt_in_grp::<E::G1, E>(&oprimes[i], &ck.pws_rando);
        uhatcom + oprimecom
    }).collect::<Vec<_>>();

    let ds = (0..l).map(|i| {
        let qcom = eval_on_pnt_in_grp::<E::G1, E>(&qs[i], &ck.pws_g1);
        let gammacom = eval_on_pnt_in_grp::<E::G1, E>(&Polynomial::from_coefficients_vec(vec![gammas[i]]), &ck.pws_rando);
        qcom - gammacom
    }).collect::<Vec<_>>();

    let beta = Polynomial::<E::Scalar, Coeff>::zero();
    let d = eval_on_pnt_in_grp::<E::G1, E>(&q, &ck.pws_g1) + eval_on_pnt_in_grp::<E::G1, E>(&beta, &ck.pws_rando);
    println!("Commitment time: {:?}", commit_timer.elapsed());

    let x = E::Scalar::random(&mut OsRng);
    let mut osum = Polynomial::<E::Scalar, Coeff>::zero();
    for i in 0..l {
        osum = osum + &(Polynomial::from_coefficients_vec(vec![os[i]]) * &zhats[i]);
    }
    let otilde = Polynomial::zero() - &(beta * &zhats[l] + &osum);
    let z = otilde.evaluate(x);
    let (w, rem) = (otilde - &Polynomial::from_coefficients_vec(vec![z])).divide_with_q_and_r(&Polynomial::from_coefficients_vec(vec![-x, E::Scalar::ONE])).unwrap();
    let wcom = eval_on_pnt_in_grp::<E::G1, E>(&w, &ck.pws_rando);
    println!("CPLink1 time: {:?}", cplink1_timer.elapsed());
    (chats, ds, cprimes, wcom, bigc, d, x, z)
}

fn cplink2<E: Engine<Scalar: WithSmallOrderMulGroup<3>> + Debug>(
    thetas: Vec<E::Scalar>, 
    HH: EvaluationDomain<E::Scalar>, 
    us: Vec<Polynomial<E::Scalar, Coeff>>, 
    z_v: Polynomial<E::Scalar, Coeff>,
    z_v_com: E::G1,
    u_coms: Vec<E::G1>,
    ck: CS2_PP<E>,
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

    let h_coms = hs.iter().map(|h| eval_on_pnt_in_grp::<E::G1, E>(h, &ck.pws_g1)).collect::<Vec<_>>();

    // eval hjs on rho 
    // eval zv on rho
    // eval ujs on rho
    // eval ujprimes on rho*theta 
    let rho = E::Scalar::random(rng.clone());
    let h_evals = hs.iter().map(|h| h.evaluate(rho)).collect::<Vec<_>>();
    let zv_eval = z_v.evaluate(rho);
    let u_evals = us.iter().map(|u| u.evaluate(rho)).collect::<Vec<_>>();
    let rho_evals = [h_evals, vec![zv_eval], u_evals].concat();
    let polys = [hs, vec![z_v], us.clone()].concat();
    let poly_coms = [h_coms, vec![z_v_com], u_coms].concat();
    let randos = polys.iter().map(|_| E::Scalar::random(rng.clone())).collect::<Vec<_>>();
    // let (proof, b_star, c_star) = <CS2 as CPEvals<P, CS2>>::multiple_evals_on_same_point(
    //     &ck,
    //     (rho, rho_evals),
    // polys,
    //     poly_coms,
    //     randos,
    //     //rng
    // )
    // .unwrap();

    let uprime_evals = uprimes.iter().zip(thetas).map(|(uprime, theta)| uprime.evaluate(rho*theta)).collect::<Vec<_>>();

    println!("CPlink 2 time: {:?}", cplink2_timer.elapsed());
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
    ck: CS2_PP<P>,
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
        let term1 = P::pairing(&chats[i].to_affine(), &ck.g2.to_affine());
        let term2 = P::pairing(&ds[i].to_affine(), &z_coms[i].to_affine());
        let term3 = P::pairing(&cprimes[i].to_affine(), &ck.g2.to_affine());
        assert_eq!(term1, term2 + term3);
    }

    // Second CPLink1 check:
    let term1 = P::pairing(&wcom.to_affine(), &eval_on_pnt_in_grp2::<P::G2, P>(&Polynomial::from_coefficients_vec(vec![-x, P::Scalar::ONE]), &ck.pws_g2).to_affine());
    let term2 = P::pairing(&bigc.to_affine(), &ck.g2.to_affine());
    let term3 = P::pairing(&d.to_affine(), &zhat_coms[l].to_affine());
    let term4 = {
        let mut sum = <P>::Gt::identity();
        for i in 0..l {
            sum = sum + P::pairing(&cprimes[i].to_affine(), &zhat_coms[i].to_affine());
        }
        sum
    };
    let term5 = P::pairing(&eval_on_pnt_in_grp::<P::G1, P>(&Polynomial::from_coefficients_vec(vec![z]), &ck.pws_g1).to_affine(), &ck.alpha_g2.to_affine());

    let lhs = term1 + term3 + term4 + term5;
    let rhs = term2;
    assert_eq!(lhs, rhs);
    println!("Verifier time: {:?}", verifier_timer.elapsed());
}

fn cplink (witness_size: usize, col_size: usize, l: usize) {
    type F = <Bn256 as Engine>::Scalar;
    type P = Bn256;
    let rng = &mut OsRng;
    let size = witness_size / l;
    let (ck, HH, thetas, zs, z_v, z_last, zhats, z_coms, zhat_coms) = setup::<P>(col_size as u32, witness_size, l);
    // Prover
    // input polys 
    let prover_timer = Instant::now();
    let poly_timer = Instant::now();
    let vals = (0..l).map(|y| (y*size..(y+1)*size).map(|x| F::from(x as u64)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let coeffs = vals.iter().map(|x| HH.lagrange_from_vec(x.clone())).collect::<Vec<_>>();
    let us = coeffs.iter().map(|x| HH.lagrange_to_coeff(x.clone())).collect::<Vec<_>>();
    let poly_vals = vals.into_iter().flatten().collect::<Vec<_>>();
    let poly = HH.lagrange_to_coeff(HH.lagrange_from_vec(poly_vals));
    println!("Construct polys: {:?}", poly_timer.elapsed());
    // CPLINK2 
    let uprimes = cplink2::<P>(thetas, HH.clone(), us, z_v, <Bn256 as Engine>::G1::identity(), vec![<Bn256 as Engine>::G1::identity(); 1], ck.clone());
    // At this point we have uprimes which match poly 
    // CPLINK1 (uhats are uprimes in paper)
    let (chats, ds, cprimes, wcom, bigc, d, x, z) = cplink1(uprimes, zs, zhats, poly, ck.clone(), z_last, HH);
    println!("Prover time: {:?}", prover_timer.elapsed());
    // Verify CPlink1
    verify1(cprimes, chats, ds, ck, z_coms, zhat_coms, wcom, bigc, d, x, z);
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
    cplink(witness_size, pow, l);
}
#[test]
fn test_stuff() {
    type F = <Bn256 as Engine>::Scalar;
    let HH = EvaluationDomain::<F>::new(1, 4);
    let omega = HH.get_omega();
    let omega_pows = powers(omega).take(32).collect::<Vec<_>>();

    let vals = (0..4).map(|x| F::from(x)).collect::<Vec<_>>();
    let poly = HH.lagrange_from_vec(vals);
    let poly_coeff = HH.lagrange_to_coeff(poly.clone());
    let squared = poly_coeff.clone() * &poly_coeff;
    for i in omega_pows[0..6].iter() {
        println!("Eval: {:?}", squared.evaluate(*i));
    }
    let vanishing1 = HH.coeff_from_vec(vec![-omega_pows[0], F::ONE]);
    let vanishing2 = HH.coeff_from_vec(vec![-omega_pows[1], F::ONE]);
    let vanishing_both = vanishing1.clone() * &vanishing2;
}
fn main() {
    let args: Vec<String> = env::args().collect();
    let pow: usize = args[4].parse().unwrap(); 
    let witness_size: usize = args[5].parse().unwrap(); 
    assert!(witness_size < 2usize.pow(pow as u32));
    let l: usize = args[6].parse().unwrap(); 
    //let witness_size = 10usize.pow(3);
    //let col_size = witness_size.next_power_of_two();
    //let l = 10;
    cplink(witness_size, witness_size, l);
}
