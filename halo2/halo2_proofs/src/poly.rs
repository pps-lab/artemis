//! Contains utilities for performing arithmetic over univariate polynomials in
//! various forms, including computing commitments to them and provably opening
//! the committed polynomials at arbitrary points.

use crate::arithmetic::parallelize;
use crate::helpers::SerdePrimeField;
use crate::plonk::Assigned;
use crate::{multicore, SerdeFormat};

use ark_std::Zero;
use ff::{PrimeField, WithSmallOrderMulGroup};
use group::ff::{BatchInvert, Field};
use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, RangeFrom, RangeFull, Sub};

/// Generic commitment scheme structures
pub mod commitment;
mod domain;
mod query;
mod strategy;

/// Inner product argument commitment scheme
pub mod ipa;

/// KZG commitment scheme
pub mod kzg;

#[cfg(test)]
mod multiopen_test;

pub use domain::*;
pub use query::{ProverQuery, VerifierQuery};
pub use strategy::{Guard, VerificationStrategy};

/// This is an error that could occur during proving or circuit synthesis.
// TODO: these errors need to be cleaned up
#[derive(Debug)]
pub enum Error {
    /// OpeningProof is not well-formed
    OpeningError,
    /// Caller needs to re-sample a point
    SamplingError,
}

/// The basis over which a polynomial is described.
pub trait Basis: Copy + Debug + Send + Sync {}

/// The polynomial is defined as coefficients
#[derive(Clone, Copy, Debug)]
pub struct Coeff;
impl Basis for Coeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials
#[derive(Clone, Copy, Debug)]
pub struct LagrangeCoeff;
impl Basis for LagrangeCoeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials in
/// an extended size domain which supports multiplication
#[derive(Clone, Copy, Debug)]
pub struct ExtendedLagrangeCoeff;
impl Basis for ExtendedLagrangeCoeff {}

/// Represents a univariate polynomial defined over a field and a particular
/// basis.
#[derive(Clone, Debug)]
pub struct Polynomial<F, B> {
    pub values: Vec<F>,
    pub _marker: PhantomData<B>,
}

impl<F, B> Index<usize> for Polynomial<F, B> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<usize> for Polynomial<F, B> {
    fn index_mut(&mut self, index: usize) -> &mut F {
        self.values.index_mut(index)
    }
}

impl<F, B> Index<RangeFrom<usize>> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: RangeFrom<usize>) -> &[F] {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<RangeFrom<usize>> for Polynomial<F, B> {
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [F] {
        self.values.index_mut(index)
    }
}

impl<F, B> Index<RangeFull> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: RangeFull) -> &[F] {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<RangeFull> for Polynomial<F, B> {
    fn index_mut(&mut self, index: RangeFull) -> &mut [F] {
        self.values.index_mut(index)
    }
}

impl <F: Field, B: Clone> Polynomial<F, B> {
    pub fn pad(&self, n: usize) -> Self {
        let mut poly = self.clone();
        poly.values.extend(vec![F::ZERO; n]);
        poly      
    }
}

impl <F: Field> Polynomial<F, Coeff> {

    pub fn naive_mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            Polynomial::zero()
        } else {
            let mut result = vec![F::ZERO; self.values.len() + other.values.len() - 1];
            for (i, self_coeff) in self.values.iter().enumerate() {
                for (j, other_coeff) in other.values.iter().enumerate() {
                    result[i + j] += &(*self_coeff * other_coeff);
                }
            }
            Polynomial::from_coefficients_vec(result)
        }
    }

    pub fn normalize(&self) -> Self {
        let mut poly = self.clone();
        while let Some(true) = poly.values.last().map(|c| *c == F::ZERO && poly.values.len() > 1) {
            poly.values.pop();
        }
        poly
    }

    pub fn zero() -> Self {
        Self { values: vec![F::ZERO], _marker: PhantomData }
    }

    pub fn is_zero(&self) -> bool {
        self.values.is_empty() || self.values.iter().all(|coeff| *coeff == F::ZERO)
    }
    pub fn evaluate(&self, point: F) -> F {
        fn eval<F: Field>(poly: &[F], point: F) -> F {
            poly.iter()
                .rev()
                .fold(F::ZERO, |acc, coeff| acc * point + coeff)
        }
        let poly = self.values.clone();
        let n = poly.len();
        let num_threads = multicore::current_num_threads();
        if n * 2 < num_threads {
            eval(&poly, point)
        } else {
            let chunk_size = (n + num_threads - 1) / num_threads;
            let mut parts = vec![F::ZERO; num_threads];
            multicore::scope(|scope| {
                for (chunk_idx, (out, poly)) in
                    parts.chunks_mut(1).zip(poly.chunks(chunk_size)).enumerate()
                {
                    scope.spawn(move |_| {
                        let start = chunk_idx * chunk_size;
                        out[0] = eval(poly, point) * point.pow_vartime(&[start as u64, 0, 0, 0]);
                    });
                }
            });
            parts.iter().fold(F::ZERO, |acc, coeff| acc + coeff)
        }
    }

    // pub fn from_coefficients_vec(coeffs: Vec<F>) -> Self{
    //     Self { values: coeffs, _marker: PhantomData }
    // }

    pub fn divide_with_q_and_r(
        &self,
        divisor: &Self,
    ) -> Option<(Polynomial<F, Coeff>, Polynomial<F, Coeff>)> {
        let mut remainder: Polynomial<F, Coeff> = self.clone().into();
        while let Some(true) = remainder.values.last().map(|c| *c == F::ZERO) {
            remainder.values.pop();
        }

        let mut divisor_mut = divisor.clone();
        // Can unwrap here because we know self is not zero.
        while let Some(true) = divisor_mut.values.last().map(|c| *c == F::ZERO) {
            divisor_mut.values.pop();
        }

        if remainder.is_zero() {
            Some((Polynomial::zero(), Polynomial::zero()))
        } else if divisor_mut.is_zero() {
            panic!("Dividing by zero polynomial")
        } else if remainder.values.len() < divisor_mut.values.len() {
            Some((Polynomial::zero(), self.clone().into()))
        } else {
            // Now we know that self.degree() >= divisor.degree();
            let mut quotient = vec![F::ZERO; self.values.len() - divisor_mut.values.len() + 1];
            let mut remainder: Polynomial<F, Coeff> = self.clone().into();
            let divisor_leading_inv = divisor_mut.values.last().unwrap().invert().unwrap();
            while !remainder.is_zero() && remainder.values.len() >= divisor_mut.values.len() {
                let cur_q_coeff = *remainder.values.last().unwrap() * divisor_leading_inv;
                let cur_q_degree = remainder.values.len() - divisor_mut.values.len();
                quotient[cur_q_degree] = cur_q_coeff;

                for (i, div_coeff) in divisor_mut.iter().enumerate() {
                    remainder[cur_q_degree + i] -= &(cur_q_coeff * div_coeff);
                }
                while let Some(true) = remainder.values.last().map(|c| *c == F::ZERO) {
                    remainder.values.pop();
                }
            }
            Some((Polynomial::from_coefficients_vec(quotient), remainder))
        }
    }
}

impl<F, B> Deref for Polynomial<F, B> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        &self.values[..]
    }
}

impl<F, B> DerefMut for Polynomial<F, B> {
    fn deref_mut(&mut self) -> &mut [F] {
        &mut self.values[..]
    }
}

impl<F, B> Polynomial<F, B> {
    
    /// Iterate over the values, which are either in coefficient or evaluation
    /// form depending on the basis `B`.
    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.values.iter()
    }

    /// Iterate over the values mutably, which are either in coefficient or
    /// evaluation form depending on the basis `B`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
        self.values.iter_mut()
    }

    /// Gets the size of this polynomial in terms of the number of
    /// coefficients used to describe it.
    pub fn num_coeffs(&self) -> usize {
        self.values.len()
    }

    pub fn from_coefficients_vec(coeffs: Vec<F>) -> Self{
        Self { values: coeffs, _marker: PhantomData }
    }
}

impl<F: SerdePrimeField, B> Polynomial<F, B> {
    /// Reads polynomial from buffer using `SerdePrimeField::read`.  
    pub(crate) fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        let mut poly_len = [0u8; 4];
        reader.read_exact(&mut poly_len)?;
        let poly_len = u32::from_be_bytes(poly_len);

        (0..poly_len)
            .map(|_| F::read(reader, format))
            .collect::<io::Result<Vec<_>>>()
            .map(|values| Self {
                values,
                _marker: PhantomData,
            })
    }

    /// Writes polynomial to buffer using `SerdePrimeField::write`.  
    pub(crate) fn write<W: io::Write>(
        &self,
        writer: &mut W,
        format: SerdeFormat,
    ) -> io::Result<()> {
        writer.write_all(&(self.values.len() as u32).to_be_bytes())?;
        for value in self.values.iter() {
            value.write(writer, format)?;
        }
        Ok(())
    }
}

pub(crate) fn batch_invert_assigned<F: Field>(
    assigned: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
) -> Vec<Polynomial<F, LagrangeCoeff>> {
    let mut assigned_denominators: Vec<_> = assigned
        .iter()
        .map(|f| {
            f.iter()
                .map(|value| value.denominator())
                .collect::<Vec<_>>()
        })
        .collect();

    assigned_denominators
        .iter_mut()
        .flat_map(|f| {
            f.iter_mut()
                // If the denominator is trivial, we can skip it, reducing the
                // size of the batch inversion.
                .filter_map(|d| d.as_mut())
        })
        .batch_invert();

    assigned
        .iter()
        .zip(assigned_denominators.into_iter())
        .map(|(poly, inv_denoms)| poly.invert(inv_denoms.into_iter().map(|d| d.unwrap_or(F::ONE))))
        .collect()
}

impl<F: Field> Polynomial<Assigned<F>, LagrangeCoeff> {
    pub(crate) fn invert(
        &self,
        inv_denoms: impl Iterator<Item = F> + ExactSizeIterator,
    ) -> Polynomial<F, LagrangeCoeff> {
        assert_eq!(inv_denoms.len(), self.values.len());
        Polynomial {
            values: self
                .values
                .iter()
                .zip(inv_denoms.into_iter())
                .map(|(a, inv_den)| a.numerator() * inv_den)
                .collect(),
            _marker: self._marker,
        }
    }
}

impl<'a, F: Field, B: Basis> Add<&'a Polynomial<F, B>> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn add(mut self, rhs: &'a Polynomial<F, B>) -> Polynomial<F, B> {
        let mut rhs = rhs.clone();
        if self.values.len() > rhs.values.len() {
            rhs = rhs.pad(self.values.len() - rhs.values.len());
        } else {
            self = self.pad(rhs.values.len() - self.values.len());
        }
        parallelize(&mut self.values, |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.values[start..].iter()) {
                *lhs += *rhs;
            }
        });
        self
    }
}

impl<'a, F: Field, B: Basis> Sub<&'a Polynomial<F, B>> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn sub(mut self, rhs: &'a Polynomial<F, B>) -> Polynomial<F, B> {
        let mut rhs = rhs.clone();
        if self.values.len() > rhs.values.len() {
            rhs = rhs.pad(self.values.len() - rhs.values.len());
        } else {
            self = self.pad(rhs.values.len() - self.values.len());
        }
        if rhs.values.len() == 1 {
            self = Sub::<F>::sub(&self, rhs.values[0]);
        }
        else {
            parallelize(&mut self.values, |lhs, start| {
                for (lhs, rhs) in lhs.iter_mut().zip(rhs.values[start..].iter()) {
                    *lhs -= *rhs;
                }
            });
        }
        self
    }
}

impl<F: Field> Polynomial<F, LagrangeCoeff> {
    /// Rotates the values in a Lagrange basis polynomial by `Rotation`
    pub fn rotate(&self, rotation: Rotation) -> Polynomial<F, LagrangeCoeff> {
        let mut values = self.values.clone();
        if rotation.0 < 0 {
            values.rotate_right((-rotation.0) as usize);
        } else {
            values.rotate_left(rotation.0 as usize);
        }
        Polynomial {
            values,
            _marker: PhantomData,
        }
    }
}

impl<F: Field, B: Basis> Mul<F> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn mul(mut self, rhs: F) -> Polynomial<F, B> {
        if rhs == F::ZERO {
            return Polynomial {
                values: vec![F::ZERO; self.len()],
                _marker: PhantomData,
            };
        }
        if rhs == F::ONE {
            return self;
        }

        parallelize(&mut self.values, |lhs, _| {
            for lhs in lhs.iter_mut() {
                *lhs *= rhs;
            }
        });

        self
    }
}

impl<'a, F: Field, B: Basis> Sub<F> for &'a Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn sub(self, rhs: F) -> Polynomial<F, B> {
        let mut res = self.clone();
        res.values[0] -= rhs;
        res
    }
}

impl<'a, F: Field + WithSmallOrderMulGroup<3>> Mul<&'a Polynomial<F, Coeff>> for Polynomial<F, Coeff> {
    type Output = Polynomial<F, Coeff>;

    // fn mul(self, rhs: &'a Polynomial<F, Coeff>) -> Polynomial<F, Coeff> {
    //     if self.is_zero() || rhs.is_zero() {
    //         Polynomial::zero()
    //     } else {
    //         let mut result = vec![F::ZERO; self.values.len() + rhs.values.len() + 1];
    //         for (i, self_coeff) in self.values.iter().enumerate() {
    //             for (j, other_coeff) in rhs.values.iter().enumerate() {
    //                 result[i + j] += &(*self_coeff * other_coeff);
    //             }
    //         }
    //         Polynomial::from_coefficients_vec(result)
    //     }
    // }

    #[inline]
    fn mul(self, other: &'a Polynomial<F, Coeff>) -> Polynomial<F, Coeff> {
        if self.is_zero() || other.is_zero() {
            Polynomial::zero()
        } else {
            let rhs = self.normalize();
            let lhs = other.normalize();
            let k = (rhs.values.len() + lhs.values.len() - 1).next_power_of_two().ilog2();
            //println!("k: {}, vals len 1: {}, vals len 2: {}", k, self.values.len(), other.values.len());
            let domain = EvaluationDomain::new(1, k);
            let self_evals = domain.coeff_to_lagrange(lhs);
            let other_evals = domain.coeff_to_lagrange(rhs.clone());
            let res_evals = self_evals.clone() * &other_evals;
            let res_coeffs = domain.lagrange_to_coeff(res_evals);
            res_coeffs.normalize()
        }
    }
}

impl<'a, F: Field> Mul<&'a Polynomial<F, LagrangeCoeff>> for Polynomial<F, LagrangeCoeff> {
    type Output = Polynomial<F, LagrangeCoeff>;

    #[inline]
    fn mul(self, other: &'a Polynomial<F, LagrangeCoeff>) -> Polynomial<F, LagrangeCoeff> {
        let result = self.values.iter().zip(other.values.clone()).map(|(a, b)| *a * b).collect::<Vec<_>>();
        Self::from_coefficients_vec(result)
    }
}


/// Describes the relative rotation of a vector. Negative numbers represent
/// reverse (leftmost) rotations and positive numbers represent forward (rightmost)
/// rotations. Zero represents no rotation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rotation(pub i32);

impl Rotation {
    /// The current location in the evaluation domain
    pub fn cur() -> Rotation {
        Rotation(0)
    }

    /// The previous location in the evaluation domain
    pub fn prev() -> Rotation {
        Rotation(-1)
    }

    /// The next location in the evaluation domain
    pub fn next() -> Rotation {
        Rotation(1)
    }
}
