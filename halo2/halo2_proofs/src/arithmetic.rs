//! This module provides common utilities, traits and structures for group,
//! field and polynomial arithmetic.

use super::multicore;
pub use ff::Field;
use group::{
    ff::{BatchInvert, PrimeField},
    Curve, Group, GroupOpsOwned, ScalarMulOwned,
};

pub use halo2curves::{CurveAffine, CurveExt};

use crate::{
    fft::{
        self, parallel,
        recursive::{self, FFTData},
    },
    plonk::{get_duration, get_time, log_info},
    poly::EvaluationDomain,
};
use std::{env, mem};

/// This represents an element of a group with basic operations that can be
/// performed. This allows an FFT implementation (for example) to operate
/// generically over either a field or elliptic curve group.
pub trait FftGroup<Scalar: Field>:
    Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>
{
}

impl<T, Scalar> FftGroup<Scalar> for T
where
    Scalar: Field,
    T: Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>,
{
}

/// TEMP
pub static mut MULTIEXP_TOTAL_TIME: usize = 0;

fn multiexp_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C], acc: &mut C::Curve) {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();

    let c = if bases.len() < 4 {
        1
    } else if bases.len() < 32 {
        3
    } else {
        (f64::from(bases.len() as u32)).ln().ceil() as usize
    };

    fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
        let skip_bits = segment * c;
        let skip_bytes = skip_bits / 8;

        if skip_bytes >= 32 {
            return 0;
        }

        let mut v = [0; 8];
        for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
            *v = *o;
        }

        let mut tmp = u64::from_le_bytes(v);
        tmp >>= skip_bits - (skip_bytes * 8);
        tmp = tmp % (1 << c);

        tmp as usize
    }

    let segments = (256 / c) + 1;

    for current_segment in (0..segments).rev() {
        for _ in 0..c {
            *acc = acc.double();
        }

        #[derive(Clone, Copy)]
        enum Bucket<C: CurveAffine> {
            None,
            Affine(C),
            Projective(C::Curve),
        }

        impl<C: CurveAffine> Bucket<C> {
            fn add_assign(&mut self, other: &C) {
                *self = match *self {
                    Bucket::None => Bucket::Affine(*other),
                    Bucket::Affine(a) => Bucket::Projective(a + *other),
                    Bucket::Projective(mut a) => {
                        a += *other;
                        Bucket::Projective(a)
                    }
                }
            }

            fn add(self, mut other: C::Curve) -> C::Curve {
                match self {
                    Bucket::None => other,
                    Bucket::Affine(a) => {
                        other += a;
                        other
                    }
                    Bucket::Projective(a) => other + &a,
                }
            }
        }

        let mut buckets: Vec<Bucket<C>> = vec![Bucket::None; (1 << c) - 1];

        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
            let coeff = get_at::<C::Scalar>(current_segment, c, coeff);
            if coeff != 0 {
                buckets[coeff - 1].add_assign(base);
            }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = C::Curve::identity();
        for exp in buckets.into_iter().rev() {
            running_sum = exp.add(running_sum);
            *acc = *acc + &running_sum;
        }
    }
}

/// Performs a small multi-exponentiation operation.
/// Uses the double-and-add algorithm with doublings shared across points.
pub fn small_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();
    let mut acc = C::Curve::identity();

    // for byte idx
    for byte_idx in (0..32).rev() {
        // for bit idx
        for bit_idx in (0..8).rev() {
            acc = acc.double();
            // for each coeff
            for coeff_idx in 0..coeffs.len() {
                let byte = coeffs[coeff_idx].as_ref()[byte_idx];
                if ((byte >> bit_idx) & 1) != 0 {
                    acc += bases[coeff_idx];
                }
            }
        }
    }

    acc
}

/// Pippenger MSM parallelised over point chunks only (the pre-Oct-2026 `best_multiexp`): used
/// for small inputs and as the reference in tests.
pub fn multiexp_chunked<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    assert_eq!(coeffs.len(), bases.len());
    let num_threads = multicore::current_num_threads();
    let res = if coeffs.len() > num_threads {
        let chunk = coeffs.len() / num_threads;
        let num_chunks = coeffs.chunks(chunk).len();
        let mut results = vec![C::Curve::identity(); num_chunks];
        multicore::scope(|scope| {
            let chunk = coeffs.len() / num_threads;

            for ((coeffs, bases), acc) in coeffs
                .chunks(chunk)
                .zip(bases.chunks(chunk))
                .zip(results.iter_mut())
            {
                scope.spawn(move |_| {
                    multiexp_serial(coeffs, bases, acc);
                });
            }
        });
        results.iter().fold(C::Curve::identity(), |a, b| a + b)
    } else {
        let mut acc = C::Curve::identity();
        multiexp_serial(coeffs, &bases, &mut acc);
        acc
    };

    res
}


// ---------------------------------------------------------------------------------------------
// Pippenger MSM with Booth (signed-digit) windows and batched affine bucket additions, ported
// from halo2curves 0.10 (`msm_best`, "cyclone" MSM) on 27 Oct 2026. Differences from upstream:
// the work is split over windows *and* point chunks so that machines with more threads than
// windows (e.g. 32 vCPUs vs. 19 windows at 2^19) are kept busy, and the bases are never cloned.
// ---------------------------------------------------------------------------------------------

/// Number of affine bucket additions sharing one field inversion. Upstream uses 64 with a
/// linear scan for the collision check; with the per-bucket flag below the scan is O(1) and 256
/// was the sweet spot on an M-series Mac (2^19: 168 ms at 64, 146 ms at 256, 150 ms at 1024).
const MSM_BATCH_SIZE: usize = 256;

fn get_booth_index(window_index: usize, window_size: usize, el: &[u8]) -> i32 {
    // Booth encoding: slice `window_size + 1` bits with a one-bit overlap between windows and a
    // zero bit appended at the least significant end, so that digits lie in
    // [-2^(w-1), 2^(w-1)] and the bucket count halves without preprocessing the scalars.
    let skip_bits = (window_index * window_size).saturating_sub(1);
    let skip_bytes = skip_bits / 8;

    let mut v: [u8; 4] = [0; 4];
    for (dst, src) in v.iter_mut().zip(el.iter().skip(skip_bytes)) {
        *dst = *src
    }
    let mut tmp = u32::from_le_bytes(v);

    if window_index == 0 {
        tmp <<= 1;
    }

    tmp >>= skip_bits - (skip_bytes * 8);
    tmp &= (1 << (window_size + 1)) - 1;

    let sign = tmp & (1 << window_size) == 0;

    tmp = (tmp + 1) >> 1;

    if sign {
        tmp as i32
    } else {
        -(((!(tmp - 1)) & ((1 << window_size) - 1)) as i32)
    }
}

#[derive(Debug, Clone, Copy)]
struct MsmAffine<C: CurveAffine> {
    x: C::Base,
    y: C::Base,
}

impl<C: CurveAffine> MsmAffine<C> {
    fn from(point: &C) -> Self {
        let coords = point.coordinates().unwrap();
        Self {
            x: *coords.x(),
            y: *coords.y(),
        }
    }

    fn neg(&self) -> Self {
        Self {
            x: self.x,
            y: -self.y,
        }
    }

    fn eval(&self) -> C {
        C::from_xy(self.x, self.y).unwrap()
    }
}

#[derive(Debug, Clone)]
enum BucketAffine<C: CurveAffine> {
    None,
    Point(MsmAffine<C>),
}

#[derive(Debug, Clone)]
enum BucketProjective<C: CurveAffine> {
    None,
    Point(C::Curve),
}

impl<C: CurveAffine> BucketProjective<C> {
    fn add_assign(&mut self, point: &C, sign: bool) {
        *self = match *self {
            BucketProjective::None => BucketProjective::Point({
                if sign {
                    point.to_curve()
                } else {
                    -point.to_curve()
                }
            }),
            BucketProjective::Point(a) => {
                if sign {
                    Self::Point(a + point)
                } else {
                    Self::Point(a - point)
                }
            }
        }
    }

    fn add(&self, other: &BucketAffine<C>) -> C::Curve {
        match (self, other) {
            (Self::Point(this), BucketAffine::Point(other)) => *this + other.eval(),
            (Self::Point(this), BucketAffine::None) => *this,
            (Self::None, BucketAffine::Point(other)) => other.eval().to_curve(),
            (Self::None, BucketAffine::None) => C::Curve::identity(),
        }
    }
}

impl<C: CurveAffine> BucketAffine<C> {
    fn assign(&mut self, point: &MsmAffine<C>, sign: bool) -> bool {
        match *self {
            Self::None => {
                *self = Self::Point(if sign { *point } else { point.neg() });
                true
            }
            Self::Point(_) => false,
        }
    }

    fn x(&self) -> C::Base {
        match self {
            Self::None => panic!("::x None"),
            Self::Point(a) => a.x,
        }
    }

    fn y(&self) -> C::Base {
        match self {
            Self::None => panic!("::y None"),
            Self::Point(a) => a.y,
        }
    }

    fn is_inf(&self) -> bool {
        matches!(self, Self::None)
    }

    fn set_x(&mut self, x: &C::Base) {
        match self {
            Self::None => panic!("::set_x None"),
            Self::Point(ref mut a) => a.x = *x,
        }
    }

    fn set_y(&mut self, y: &C::Base) {
        match self {
            Self::None => panic!("::set_y None"),
            Self::Point(ref mut a) => a.y = *y,
        }
    }

    fn set_inf(&mut self) {
        *self = Self::None;
    }
}

#[derive(Debug, Clone, Default)]
struct SchedulePoint {
    base_idx: usize,
    buck_idx: usize,
    sign: bool,
}

/// Adds `points[i].base` to bucket `points[i].buck_idx` for all scheduled points with a single
/// field inversion (Montgomery's trick on the slopes).
fn msm_batch_add<C: CurveAffine>(
    size: usize,
    buckets: &mut [BucketAffine<C>],
    points: &[SchedulePoint],
    bases: &[MsmAffine<C>],
) {
    let mut t = vec![C::Base::ZERO; size]; // numerators of the slopes, times the running product
    let mut z = vec![C::Base::ZERO; size]; // denominators of the slopes
    let mut acc = C::Base::ONE;

    for ((sp, t), z) in points.iter().zip(t.iter_mut()).zip(z.iter_mut()) {
        let SchedulePoint {
            base_idx,
            buck_idx,
            sign,
        } = sp;
        if buckets[*buck_idx].is_inf() {
            // bases are never the identity (checked by the caller)
            continue;
        }

        if buckets[*buck_idx].x() == bases[*base_idx].x {
            // same x: doubling when the y's agree with the sign, P + (-P) otherwise
            if (buckets[*buck_idx].y() == bases[*base_idx].y) ^ !*sign {
                let x_squared = bases[*base_idx].x.square();
                *z = buckets[*buck_idx].y() + buckets[*buck_idx].y(); // 2y
                *t = acc * (x_squared + x_squared + x_squared); // acc * 3x^2
                acc *= *z;
                continue;
            }
            buckets[*buck_idx].set_inf();
            continue;
        }
        *z = buckets[*buck_idx].x() - bases[*base_idx].x; // x2 - x1
        if *sign {
            *t = acc * (buckets[*buck_idx].y() - bases[*base_idx].y);
        } else {
            *t = acc * (buckets[*buck_idx].y() + bases[*base_idx].y);
        }
        acc *= *z;
    }

    acc = acc
        .invert()
        .expect("msm_batch_add: zero denominator in a batched affine addition");

    for ((sp, t), z) in points.iter().zip(t.iter()).zip(z.iter()).rev() {
        let SchedulePoint {
            base_idx,
            buck_idx,
            sign,
        } = sp;
        if buckets[*buck_idx].is_inf() {
            continue;
        }
        let lambda = acc * t;
        acc *= z;
        let x = lambda.square() - (buckets[*buck_idx].x() + bases[*base_idx].x);
        if *sign {
            buckets[*buck_idx].set_y(&((lambda * (bases[*base_idx].x - x)) - bases[*base_idx].y));
        } else {
            buckets[*buck_idx].set_y(&((lambda * (bases[*base_idx].x - x)) + bases[*base_idx].y));
        }
        buckets[*buck_idx].set_x(&x);
    }
}

struct Schedule<C: CurveAffine> {
    buckets: Vec<BucketAffine<C>>,
    /// one flag per bucket: is the bucket already waiting in the current batch?
    scheduled: Vec<bool>,
    set: Vec<SchedulePoint>,
    ptr: usize,
}

impl<C: CurveAffine> Schedule<C> {
    fn new(c: usize, batch_size: usize) -> Self {
        Self {
            buckets: vec![BucketAffine::None; 1 << (c - 1)],
            scheduled: vec![false; 1 << (c - 1)],
            set: vec![SchedulePoint::default(); batch_size],
            ptr: 0,
        }
    }

    #[inline]
    fn contains(&self, buck_idx: usize) -> bool {
        self.scheduled[buck_idx]
    }

    fn execute(&mut self, bases: &[MsmAffine<C>]) {
        if self.ptr != 0 {
            msm_batch_add(self.ptr, &mut self.buckets, &self.set, bases);
            for sp in &self.set[..self.ptr] {
                self.scheduled[sp.buck_idx] = false;
            }
            self.ptr = 0;
        }
    }

    fn add(&mut self, bases: &[MsmAffine<C>], base_idx: usize, buck_idx: usize, sign: bool) {
        if !self.buckets[buck_idx].assign(&bases[base_idx], sign) {
            self.set[self.ptr] = SchedulePoint {
                base_idx,
                buck_idx,
                sign,
            };
            self.scheduled[buck_idx] = true;
            self.ptr += 1;
        }
        if self.ptr == self.set.len() {
            self.execute(bases);
        }
    }
}

/// Window size for an MSM of `n` points.
fn msm_window_size(n: usize) -> usize {
    if n < 4 {
        1
    } else if n < 32 {
        3
    } else {
        (n as f64).ln().ceil() as usize
    }
}

/// Bucket accumulation of one window over one chunk of points: the bucket sums are reduced to
/// the window's contribution (not yet shifted).
fn msm_window_chunk<C: CurveAffine>(
    w: usize,
    c: usize,
    coeffs: &[<C::Scalar as PrimeField>::Repr],
    bases: &[C],
    bases_local: &[MsmAffine<C>],
) -> C::Curve {
    // buckets for points that could not be scheduled (projective fallback)
    let mut j_bucks = vec![BucketProjective::<C>::None; 1 << (c - 1)];
    // affine buckets filled through the batched additions
    let mut sched = Schedule::new(c, MSM_BATCH_SIZE);

    for (base_idx, coeff) in coeffs.iter().enumerate() {
        let buck_idx = get_booth_index(w, c, coeff.as_ref());
        if buck_idx != 0 {
            let sign = buck_idx.is_positive();
            let buck_idx = buck_idx.unsigned_abs() as usize - 1;
            if sched.contains(buck_idx) {
                // the bucket is already waiting in the batch: fall back to a projective add
                j_bucks[buck_idx].add_assign(&bases[base_idx], sign);
            } else {
                sched.add(bases_local, base_idx, buck_idx, sign);
            }
        }
    }
    sched.execute(bases_local);

    // summation by parts: 3a + 2b + 1c = a + (a + b) + ((a + b) + c)
    let mut running_sum = C::Curve::identity();
    let mut acc = C::Curve::identity();
    for (j_buck, a_buck) in j_bucks.iter().zip(sched.buckets.iter()).rev() {
        running_sum += j_buck.add(a_buck);
        acc += running_sum;
    }
    acc
}

/// Batched-affine Pippenger MSM (see the module comment above). `coeffs` and `bases` must have
/// the same length and no base may be the identity.
pub fn multiexp_batch_affine<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    use rayon::prelude::*;

    assert_eq!(coeffs.len(), bases.len());
    let n = coeffs.len();
    let c = msm_window_size(n);

    let coeffs: Vec<_> = coeffs.par_iter().map(|a| a.to_repr()).collect();
    let bases_local: Vec<_> = bases.par_iter().map(MsmAffine::from).collect();

    let number_of_windows = C::Scalar::NUM_BITS as usize / c + 1;

    // Split the points into chunks as well when there are more threads than windows, keeping
    // every chunk large compared with the bucket count so the reduction stays negligible.
    let num_threads = multicore::current_num_threads();
    let mut num_chunks = (num_threads + number_of_windows - 1) / number_of_windows;
    while num_chunks > 1 && n / num_chunks < (1 << (c + 1)) {
        num_chunks -= 1;
    }
    let chunk_len = (n + num_chunks - 1) / num_chunks;

    let tasks: Vec<(usize, usize)> = (0..number_of_windows)
        .rev()
        .flat_map(|w| (0..num_chunks).map(move |ch| (w, ch)))
        .collect();
    let partials: Vec<C::Curve> = tasks
        .par_iter()
        .map(|&(w, ch)| {
            let start = ch * chunk_len;
            let end = (start + chunk_len).min(n);
            msm_window_chunk::<C>(
                w,
                c,
                &coeffs[start..end],
                &bases[start..end],
                &bases_local[start..end],
            )
        })
        .collect();

    // Combine the windows: acc = sum_w 2^(c*w) * window_w, evaluated with Horner from the top.
    let mut acc = C::Curve::identity();
    for w in (0..number_of_windows).rev() {
        for _ in 0..c {
            acc = acc.double();
        }
        let task_base = (number_of_windows - 1 - w) * num_chunks;
        for ch in 0..num_chunks {
            acc += partials[task_base + ch];
        }
    }
    acc
}

/// Performs a multi-exponentiation operation.
///
/// Only the first `coeffs.len()` bases are used (`bases` may be longer); the bases are never
/// copied. Large inputs go through the batched-affine Pippenger (`multiexp_batch_affine`), small
/// ones through the chunked Pippenger (`multiexp_chunked`).
pub fn best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    assert!(coeffs.len() <= bases.len());
    let bases = &bases[..coeffs.len()];

    log_info(format!("msm: {}", coeffs.len()));

    let start = get_time();
    let res = if msm_window_size(coeffs.len()) < 10 {
        multiexp_chunked(coeffs, bases)
    } else {
        multiexp_batch_affine(coeffs, bases)
    };

    let duration = get_duration(start);
    #[allow(unsafe_code)]
    unsafe {
        crate::arithmetic::MULTIEXP_TOTAL_TIME += duration;
    }

    res
}

/// Dispatcher
pub fn best_fft<Scalar: Field, G: FftGroup<Scalar>>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    fft::fft(a, omega, log_n, data, inverse);
}

/// Convert coefficient bases group elements to lagrange basis by inverse FFT.
pub fn g_to_lagrange<C: CurveAffine>(g_projective: Vec<C::Curve>, k: u32) -> Vec<C> {
    let n_inv = C::Scalar::TWO_INV.pow_vartime(&[k as u64, 0, 0, 0]);
    let omega = C::Scalar::ROOT_OF_UNITY;
    let mut omega_inv = C::Scalar::ROOT_OF_UNITY_INV;
    for _ in k..C::Scalar::S {
        omega_inv = omega_inv.square();
    }

    let mut g_lagrange_projective = g_projective;
    let n = g_lagrange_projective.len();
    let fft_data = FFTData::new(n, omega, omega_inv);

    // inverse FFT (the 1/n scaling follows below); `true` selects the omega_inv twiddles
    best_fft(&mut g_lagrange_projective, omega_inv, k, &fft_data, true);
    parallelize(&mut g_lagrange_projective, |g, _| {
        for g in g.iter_mut() {
            *g *= n_inv;
        }
    });

    let mut g_lagrange = vec![C::identity(); 1 << k];
    parallelize(&mut g_lagrange, |g_lagrange, starts| {
        C::Curve::batch_normalize(
            &g_lagrange_projective[starts..(starts + g_lagrange.len())],
            g_lagrange,
        );
    });

    g_lagrange
}

/// This evaluates a provided polynomial (in coefficient form) at `point`.
pub fn eval_polynomial<F: Field>(poly: &[F], point: F) -> F {
    fn evaluate<F: Field>(poly: &[F], point: F) -> F {
        poly.iter()
            .rev()
            .fold(F::ZERO, |acc, coeff| acc * point + coeff)
    }
    let n = poly.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(poly, point)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::ZERO; num_threads];
        multicore::scope(|scope| {
            for (chunk_idx, (out, poly)) in
                parts.chunks_mut(1).zip(poly.chunks(chunk_size)).enumerate()
            {
                scope.spawn(move |_| {
                    let start = chunk_idx * chunk_size;
                    out[0] = evaluate(poly, point) * point.pow_vartime(&[start as u64, 0, 0, 0]);
                });
            }
        });
        parts.iter().fold(F::ZERO, |acc, coeff| acc + coeff)
    }
}

/// This computes the inner product of two vectors `a` and `b`.
///
/// This function will panic if the two vectors are not the same size.
pub fn compute_inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    // TODO: parallelize?
    assert_eq!(a.len(), b.len());

    let mut acc = F::ZERO;
    for (a, b) in a.iter().zip(b.iter()) {
        acc += (*a) * (*b);
    }

    acc
}

/// Divides polynomial `a` in `X` by `X - b` with
/// no remainder.
pub fn kate_division<'a, F: Field, I: IntoIterator<Item = &'a F>>(a: I, mut b: F) -> Vec<F>
where
    I::IntoIter: DoubleEndedIterator + ExactSizeIterator,
{
    b = -b;
    let a = a.into_iter();

    let mut q = vec![F::ZERO; a.len() - 1];

    let mut tmp = F::ZERO;
    for (q, r) in q.iter_mut().rev().zip(a.rev()) {
        let mut lead_coeff = *r;
        lead_coeff.sub_assign(&tmp);
        *q = lead_coeff;
        tmp = lead_coeff;
        tmp.mul_assign(&b);
    }

    q
}

/// This simple utility function will parallelize an operation that is to be
/// performed over a mutable slice.
pub fn parallelize<T: Send, F: Fn(&mut [T], usize) + Send + Sync + Clone>(v: &mut [T], f: F) {
    let n = v.len();
    let num_threads = multicore::current_num_threads();
    let mut chunk = (n as usize) / num_threads;
    if chunk < num_threads {
        chunk = 1;
    }

    multicore::scope(|scope| {
        for (chunk_num, v) in v.chunks_mut(chunk).enumerate() {
            let f = f.clone();
            scope.spawn(move |_| {
                let start = chunk_num * chunk;
                f(v, start);
            });
        }
    });
}

/// Compute the binary logarithm floored.
pub fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

/// Returns coefficients of an n - 1 degree polynomial given a set of n points
/// and their evaluations. This function will panic if two values in `points`
/// are the same.
pub fn lagrange_interpolate<F: Field>(points: &[F], evals: &[F]) -> Vec<F> {
    assert_eq!(points.len(), evals.len());
    if points.len() == 1 {
        // Constant polynomial
        vec![evals[0]]
    } else {
        let mut denoms = Vec::with_capacity(points.len());
        for (j, x_j) in points.iter().enumerate() {
            let mut denom = Vec::with_capacity(points.len() - 1);
            for x_k in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
            {
                denom.push(*x_j - x_k);
            }
            denoms.push(denom);
        }
        // Compute (x_j - x_k)^(-1) for each j != i
        denoms.iter_mut().flat_map(|v| v.iter_mut()).batch_invert();

        let mut final_poly = vec![F::ZERO; points.len()];
        for (j, (denoms, eval)) in denoms.into_iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(points.len());
            let mut product = Vec::with_capacity(points.len() - 1);
            tmp.push(F::ONE);
            for (x_k, denom) in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms.into_iter())
            {
                product.resize(tmp.len() + 1, F::ZERO);
                for ((a, b), product) in tmp
                    .iter()
                    .chain(std::iter::once(&F::ZERO))
                    .zip(std::iter::once(&F::ZERO).chain(tmp.iter()))
                    .zip(product.iter_mut())
                {
                    *product = *a * (-denom * x_k) + *b * denom;
                }
                std::mem::swap(&mut tmp, &mut product);
            }
            assert_eq!(tmp.len(), points.len());
            assert_eq!(product.len(), points.len() - 1);
            for (final_coeff, interpolation_coeff) in final_poly.iter_mut().zip(tmp.into_iter()) {
                *final_coeff += interpolation_coeff * eval;
            }
        }
        final_poly
    }
}

pub(crate) fn evaluate_vanishing_polynomial<F: Field>(roots: &[F], z: F) -> F {
    fn evaluate<F: Field>(roots: &[F], z: F) -> F {
        roots.iter().fold(F::ONE, |acc, point| (z - point) * acc)
    }
    let n = roots.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(roots, z)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::ONE; num_threads];
        multicore::scope(|scope| {
            for (out, roots) in parts.chunks_mut(1).zip(roots.chunks(chunk_size)) {
                scope.spawn(move |_| out[0] = evaluate(roots, z));
            }
        });
        parts.iter().fold(F::ONE, |acc, part| acc * part)
    }
}

pub(crate) fn powers<F: Field>(base: F) -> impl Iterator<Item = F> {
    std::iter::successors(Some(F::ONE), move |power| Some(base * power))
}

/// Reverse `l` LSBs of bitvector `n`
pub fn bitreverse(mut n: usize, l: usize) -> usize {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    r
}

#[cfg(test)]
use crate::plonk::{start_measure, stop_measure};
use rand_core::OsRng;

#[cfg(test)]
use crate::halo2curves::pasta::Fp;

#[test]
fn test_lagrange_interpolate() {
    let rng = OsRng;

    let points = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();
    let evals = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();

    for coeffs in 0..5 {
        let points = &points[0..coeffs];
        let evals = &evals[0..coeffs];

        let poly = lagrange_interpolate(points, evals);
        assert_eq!(poly.len(), points.len());

        for (point, eval) in points.iter().zip(evals) {
            assert_eq!(eval_polynomial(&poly, *point), *eval);
        }
    }
}

