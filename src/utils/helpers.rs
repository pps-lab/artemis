use std::time::Duration;
use std::{marker::PhantomData, time::Instant};
use std::fmt::Debug;
use ff::{Field, WithSmallOrderMulGroup};
use group::Curve;
use halo2_proofs::{
  circuit::{AssignedCell, Value}, halo2curves::ff::PrimeField, helpers::SerdeCurveAffine, poly::{kzg::commitment::ParamsKZG, Coeff, EvaluationDomain, Polynomial}
};
use halo2curves::pairing::Engine;
use ndarray::{Array, IxDyn};
use num_bigint::BigUint;
use rand_core::OsRng;
use rayon::{iter::{IntoParallelIterator, ParallelIterator}, slice::ParallelSlice};

use crate::{gadgets::gadget::convert_to_u128, model::PUBLIC_VALS};

// TODO: this is very bad
pub const RAND_START_IDX: i64 = i64::MIN;
pub const NUM_RANDOMS: i64 = 20001;

// Conversion / printing functions
pub fn convert_to_bigint<F: PrimeField>(x: Value<F>) -> BigUint {
  let mut big = Default::default();
  x.map(|x| {
    big = BigUint::from_bytes_le(x.to_repr().as_ref());
  });
  big
}

pub fn convert_pos_int<F: PrimeField>(x: Value<F>) -> i128 {
  let bias = 1 << 60;
  let x_pos = x + Value::known(F::from(bias as u64));
  let mut outp: i128 = 0;
  x_pos.map(|x| {
    let x_pos = convert_to_u128(&x);
    let tmp = x_pos as i128 - bias;
    outp = tmp;
  });
  return outp;
}

pub fn print_pos_int<F: PrimeField>(prefix: &str, x: Value<F>, scale_factor: u64) {
  let tmp = convert_pos_int(x);
  let tmp_float = tmp as f64 / scale_factor as f64;
  //println!("{} x: {} ({})", prefix, tmp, tmp_float);
}

pub fn print_assigned_arr<F: PrimeField>(
  prefix: &str,
  arr: &Vec<&AssignedCell<F, F>>,
  scale_factor: u64,
) {
  for (idx, x) in arr.iter().enumerate() {
    print_pos_int(
      &format!("{}[{}]", prefix, idx),
      x.value().map(|x: &F| x.to_owned()),
      scale_factor,
    );
  }
}

pub fn cplink1_lite<E: Engine<Scalar: WithSmallOrderMulGroup<3> + Ord, G1Affine: SerdeCurveAffine, G2Affine: SerdeCurveAffine> + Debug> (
  u: &Polynomial<E::Scalar, Coeff>,
  chat: &E::G1,
  bigc: &E::G1,
  z: &Polynomial<E::Scalar, Coeff>,
  poly: &Polynomial<E::Scalar, Coeff>,
  params: &ParamsKZG<E>,
  //ck: CS2_PP<E>,
  HH: &EvaluationDomain<E::Scalar>,
) -> (
  E::G1,
  E::G1,
  E::G1,
  E::G1,
  E::G1,
  E::G1,
  E::Scalar,
  E::Scalar,
  Duration
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
      // u = q * z + r
      // chat = d_small * z_com + cprime
      println!("Diff: {:?}", (q.clone() *z + &r - u).is_zero() );
      (r, q)
  };

  //println!("Uhat: {:?}", uhat);
  println!("Diff: {:?}", poly_divmod(&(uhat.clone() - poly), &z).1.is_zero());

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
  let proving_time = cplink1_timer.elapsed();
  println!("CPLINK1: Prover time: {:?}", cplink1_timer.elapsed());
  (*chat, d_small, cprime, wcom, *bigc, d, x, zz, proving_time)
}

pub fn verify1_lite<P: Engine + Debug> (
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
) -> Duration {
  let verifier_timer = Instant::now();
  // First CPlink1 check:
  // chat = d_small * z + cprime
  // 
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
  verifier_timer.elapsed()
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

pub fn powers<F: Field>(base: F) -> impl Iterator<Item = F> {
  std::iter::successors(Some(F::ONE), move |power| Some(base * power))
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
// Get the public values
pub fn get_public_values<F: PrimeField>() -> Vec<F> {
  let mut public_vals = vec![];
  for val in PUBLIC_VALS.lock().unwrap().iter() {
    let val = F::from_str_vartime(&val.to_str_radix(10));
    public_vals.push(val.unwrap());
  }
  public_vals
}

// Broadcast
fn shape_dominates(s1: &[usize], s2: &[usize]) -> bool {
  if s1.len() != s2.len() {
    return false;
  }

  for (x1, x2) in s1.iter().zip(s2.iter()) {
    if x1 < x2 {
      return false;
    }
  }

  true
}

// Precondition: s1.len() < s2.len()
fn intermediate_shape(s1: &[usize], s2: &[usize]) -> Vec<usize> {
  let mut res = vec![1; s2.len() - s1.len()];
  for s in s1.iter() {
    res.push(*s);
  }
  res
}

fn final_shape(s1: &[usize], s2: &[usize]) -> Vec<usize> {
  let mut res = vec![];
  for (x1, x2) in s1.iter().zip(s2.iter()) {
    res.push(std::cmp::max(*x1, *x2));
  }
  res
}

pub fn broadcast<G: Clone>(
  x1: &Array<G, IxDyn>,
  x2: &Array<G, IxDyn>,
) -> (Array<G, IxDyn>, Array<G, IxDyn>) {
  if x1.shape() == x2.shape() {
    return (x1.clone(), x2.clone());
  }

  if x1.ndim() == x2.ndim() {
    let s1 = x1.shape();
    let s2 = x2.shape();
    if shape_dominates(s1, s2) {
      return (x1.clone(), x2.broadcast(s1).unwrap().into_owned());
    } else if shape_dominates(x2.shape(), x1.shape()) {
      return (x1.broadcast(s2).unwrap().into_owned(), x2.clone());
    }
  }

  let (tmp1, tmp2) = if x1.ndim() < x2.ndim() {
    (x1, x2)
  } else {
    (x2, x1)
  };

  // tmp1.ndim() < tmp2.ndim()
  let s1 = tmp1.shape();
  let s2 = tmp2.shape();
  let s = intermediate_shape(s1, s2);
  let final_shape = final_shape(s2, s.as_slice());

  let tmp1 = tmp1.broadcast(s.clone()).unwrap().into_owned();
  let tmp1 = tmp1.broadcast(final_shape.as_slice()).unwrap().into_owned();
  let tmp2 = tmp2.broadcast(final_shape.as_slice()).unwrap().into_owned();
  // println!("x1: {:?} x2: {:?}", x1.shape(), x2.shape());
  // println!("s1: {:?} s2: {:?} s: {:?}", s1, s2, s);
  // println!("tmp1 shape: {:?}", tmp1.shape());
  // println!("tmp2 shape: {:?}", tmp2.shape());

  if x1.ndim() < x2.ndim() {
    return (tmp1, tmp2);
  } else {
    return (tmp2, tmp1);
  }
}

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
pub fn poly_divmod<F: Field + WithSmallOrderMulGroup<3>>(u: &Polynomial<F, Coeff>, v: &Polynomial<F, Coeff>) -> (Polynomial<F, Coeff>, Polynomial<F, Coeff>) {

  let m = poly_deg(u);
  let n = poly_deg(v);
  if m < n {
    return (Polynomial::zero(), u.clone())
  }
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
