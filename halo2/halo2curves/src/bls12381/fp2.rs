//! This module implements arithmetic over the quadratic extension field Fp2.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::cmp::Ordering;
use std::convert::TryInto;
use ff::{Field, PrimeField, WithSmallOrderMulGroup};
use rand_core::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

use crate::bls12381::fp::Fp;
use crate::impl_sum_prod;

use super::fp::FpRepr;

#[derive(Copy, Clone)]
pub struct Fp2 {
    pub c0: Fp,
    pub c1: Fp,
}

impl fmt::Debug for Fp2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} + {:?}*u", self.c0, self.c1)
    }
}

impl Default for Fp2 {
    fn default() -> Self {
        Fp2::zero()
    }
}
impl Ord for Fp2 {
    #[inline(always)]
    fn cmp(&self, other: &Fp2) -> Ordering {
        match self.c1.cmp(&other.c1) {
            Ordering::Greater => Ordering::Greater,
            Ordering::Less => Ordering::Less,
            Ordering::Equal => self.c0.cmp(&other.c0),
        }
    }
}

impl PartialOrd for Fp2 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Fp2) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "zeroize")]
impl zeroize::DefaultIsZeroes for Fp2 {}

impl From<Fp> for Fp2 {
    fn from(f: Fp) -> Fp2 {
        Fp2 {
            c0: f,
            c1: Fp::zero(),
        }
    }
}

impl ConstantTimeEq for Fp2 {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.c0.ct_eq(&other.c0) & self.c1.ct_eq(&other.c1)
    }
}

impl Eq for Fp2 {}
impl PartialEq for Fp2 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        bool::from(self.ct_eq(other))
    }
}

impl ConditionallySelectable for Fp2 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Fp2 {
            c0: Fp::conditional_select(&a.c0, &b.c0, choice),
            c1: Fp::conditional_select(&a.c1, &b.c1, choice),
        }
    }
}

impl<'a> Neg for &'a Fp2 {
    type Output = Fp2;

    #[inline]
    fn neg(self) -> Fp2 {
        self.neg()
    }
}

impl Neg for Fp2 {
    type Output = Fp2;

    #[inline]
    fn neg(self) -> Fp2 {
        -&self
    }
}

impl<'a, 'b> Sub<&'b Fp2> for &'a Fp2 {
    type Output = Fp2;

    #[inline]
    fn sub(self, rhs: &'b Fp2) -> Fp2 {
        self.sub(rhs)
    }
}

impl<'a, 'b> Add<&'b Fp2> for &'a Fp2 {
    type Output = Fp2;

    #[inline]
    fn add(self, rhs: &'b Fp2) -> Fp2 {
        self.add(rhs)
    }
}

impl<'a, 'b> Mul<&'b Fp2> for &'a Fp2 {
    type Output = Fp2;

    #[inline]
    fn mul(self, rhs: &'b Fp2) -> Fp2 {
        self.mul(rhs)
    }
}

impl_binops_additive!(Fp2, Fp2);
impl_binops_multiplicative!(Fp2, Fp2);
const ZETA: Fp2 = Fp2 {
    c0: Fp::from_raw([
        0x2E01FFFFFFFEFFFE,
        0xDE17D813620A0002,
        0xDDB3A93BE6F89688,
        0xBA69C6076A0F77EA,
        0x5F19672FDF76CE51,
        0
        ]),
    c1: Fp::zero(),
};
const MODULUS: [u64; 6] = [
    0xb9fe_ffff_ffff_aaab,
    0x1eab_fffe_b153_ffff,
    0x6730_d2a0_f6b0_f624,
    0x6477_4b84_f385_12bf,
    0x4b1b_a7b6_434b_acd7,
    0x1a01_11ea_397f_e69a,
];

const T: Fp = Fp::from_raw([
    0xdcff7fffffffd555,
    0xf55ffff58a9ffff,
    0xb39869507b587b12,
    0xb23ba5c279c2895f,
    0x258dd3db21a5d66b,
    0xd0088f51cbff34d,
]);

#[rustfmt::skip]
const T_MINUS_ONE_DIV_TWO: Fp = Fp::from_raw([
    0xee7fbfffffffeaaa,
    0x7aaffffac54ffff,
    0xd9cc34a83dac3d89,
    0xd91dd2e13ce144af,
    0x92c6e9ed90d2eb35,
    0x680447a8e5ff9a6,
]);


const MODULUS_BITS: u32 = 381;

impl crate::serde::SerdeObject for Fp2 {
    fn from_raw_bytes_unchecked(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), 96);
        let [c0, c1] = [0, 48].map(|i| Fp::from_raw_bytes_unchecked(&bytes[i..i + 48]));
        Self{c0, c1}
    }
    fn from_raw_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 96 {
            return None;
        }
        let [c0, c1] = [0, 48].map(|i| Fp::from_raw_bytes(&bytes[i..i + 48]));
        c0.zip(c1).map(|(c0, c1)| Self { c0, c1 })
    }
    fn to_raw_bytes(&self) -> Vec<u8> {
        let mut res = Vec::with_capacity(96);
        for limb in self.c0.0.iter().chain(self.c1.0.iter()) {
            res.extend_from_slice(&limb.to_le_bytes());
        }
        res
    }
    fn read_raw_unchecked<R: std::io::Read>(reader: &mut R) -> Self {
        let [c0, c1] = [(); 2].map(|_| Fp::read_raw_unchecked(reader));
        Self { c0, c1 }
    }
    fn read_raw<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let c0 = Fp::read_raw(reader)?;
        let c1 = Fp::read_raw(reader)?;
        Ok(Self { c0, c1 })
    }

    fn write_raw<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.c0.write_raw(writer)?;
        self.c1.write_raw(writer)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Fp2Repr([u8; 96]);
impl Default for Fp2Repr {
    fn default() -> Self { Fp2Repr([0; 96]) }
}
impl AsMut<[u8]> for Fp2Repr {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}
impl AsRef<[u8]> for Fp2Repr {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
impl Field for Fp2 {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    fn random(mut rng: impl RngCore) -> Self {
        Fp2 {
            c0: Fp::random(&mut rng),
            c1: Fp::random(&mut rng),
        }
    }

    fn is_zero(&self) -> Choice {
        self.c0.is_zero() & self.c1.is_zero()
    }

    fn square(&self) -> Self {
        self.square()
    }

    fn double(&self) -> Self {
        self.double()
    }

    fn sqrt(&self) -> CtOption<Self> {
        // Algorithm 9, https://eprint.iacr.org/2012/685.pdf

        if self.is_zero().into() {
            CtOption::new(Self::ZERO, Choice::from(1))
        } else {
            // a1 = self^((q - 3) / 4)
            // 0xc19139cb84c680a6e14116da060561765e05aa45a1c72a34f082305b61f3f51
            let u: [u64; 6] = T_MINUS_ONE_DIV_TWO.0;
            let mut a1 = self.pow(u);
            let mut alpha = a1;

            alpha.square_assign();
            alpha.mul_assign(self);
            let mut a0 = alpha;
            a0.frobenius_map();
            a0.mul_assign(&alpha);

            let neg1 = Fp2 {
                c0: Fp::ROOT_OF_UNITY,
                c1: Fp::zero(),
            };

            if a0 == neg1 {
                CtOption::new(a0, Choice::from(0))
            } else {
                a1.mul_assign(self);

                if alpha == neg1 {
                    a1.mul_assign(&Fp2 {
                        c0: Fp::zero(),
                        c1: Fp::one(),
                    });
                } else {
                    alpha += &Fp2::ONE;
                    // alpha = alpha^((q - 1) / 2)
                    // 0x183227397098d014dc2822db40c0ac2ecbc0b548b438e5469e10460b6c3e7ea3
                    let u: [u64; 6] = T.0;
                    alpha = alpha.pow(u);
                    a1.mul_assign(&alpha);
                }
                CtOption::new(a1, Choice::from(1))
            }
        }
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        ff::helpers::sqrt_ratio_generic(num, div)
    }

    fn invert(&self) -> CtOption<Self> {
        self.invert()
    }
}

impl From<u64> for Fp2 {
    fn from(val: u64) -> Self {
        Fp2 {
            c0: Fp::from(val),
            c1: Fp::zero(),
        }
    }
}

impl PrimeField for Fp2 {
    type Repr = Fp2Repr;

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        let c0 = Fp::from_bytes(&repr.0[..48].try_into().unwrap());
        let c1 = Fp::from_bytes(&repr.0[48..].try_into().unwrap());
        // Disallow overflow representation
        CtOption::new(Fp2{c0:c0.unwrap(), c1:c1.unwrap()}, Choice::from(1))
    }
    fn to_repr(&self) -> Self::Repr {
        Fp2Repr(self.to_bytes())
    }

    fn is_odd(&self) -> Choice {
        Choice::from(self.to_bytes()[0] & 1)
    }

    const MODULUS: &'static str =
        "0x1A0111EA397FE69A4B1BA7B6434BACD764774B84F38512BF6730D2A0F6B0F6241EABFFFEB153FFFFB9FEFFFFFFFFAAAB";
    const NUM_BITS: u32 = MODULUS_BITS;
    const CAPACITY: u32 = Self::NUM_BITS - 1;
    const MULTIPLICATIVE_GENERATOR: Self = Fp2 {
        c0: Fp::MULTIPLICATIVE_GENERATOR,
        c1: Fp::zero(),
    };
    const S: u32 = 0u32;
    const ROOT_OF_UNITY: Self = Fp2::zero();
    const ROOT_OF_UNITY_INV: Self = Fp2 {
        c0: Fp::zero(),
        c1: Fp::zero(),
    };
    const DELTA: Self = Fp2 {
        c0: Fp::zero(),
        c1: Fp::zero(),
    };
    const TWO_INV: Self = Fp2 {
        c0: Fp::TWO_INV,
        c1: Fp::zero(),
    };
}

impl WithSmallOrderMulGroup<3> for Fp2 {
    const ZETA: Self = ZETA;
}
// impl_binops_additive!(Fp2, Fp2);
// impl_binops_multiplicative!(Fp2, Fp2);
impl_sum_prod!(Fp2);
impl Fp2 {
    pub fn to_bytes(&self) -> [u8; 96] {
        let mut res = [0u8; 96];
        let c0_bytes = self.c0.to_bytes();
        let c1_bytes = self.c1.to_bytes();
        res[0..48].copy_from_slice(&c0_bytes[..]);
        res[48..96].copy_from_slice(&c1_bytes[..]);
        res
    }
    #[inline]
    pub const fn zero() -> Fp2 {
        Fp2 {
            c0: Fp::zero(),
            c1: Fp::zero(),
        }
    }

    #[inline]
    pub const fn one() -> Fp2 {
        Fp2 {
            c0: Fp::one(),
            c1: Fp::zero(),
        }
    }

    pub fn is_zero(&self) -> Choice {
        self.c0.is_zero() & self.c1.is_zero()
    }

    pub(crate) fn random(mut rng: impl RngCore) -> Fp2 {
        Fp2 {
            c0: Fp::random(&mut rng),
            c1: Fp::random(&mut rng),
        }
    }

    /// Raises this element to p.
    #[inline(always)]
    pub fn frobenius_map(&self) -> Self {
        // This is always just a conjugation. If you're curious why, here's
        // an article about it: https://alicebob.cryptoland.net/the-frobenius-endomorphism-with-finite-fields/
        self.conjugate()
    }

    #[inline(always)]
    pub fn conjugate(&self) -> Self {
        Fp2 {
            c0: self.c0,
            c1: -self.c1,
        }
    }

    #[inline(always)]
    pub fn mul_by_nonresidue(&self) -> Fp2 {
        // Multiply a + bu by u + 1, getting
        // au + a + bu^2 + bu
        // and because u^2 = -1, we get
        // (a - b) + (a + b)u

        Fp2 {
            c0: self.c0 - self.c1,
            c1: self.c0 + self.c1,
        }
    }

    /// Returns whether or not this element is strictly lexicographically
    /// larger than its negation.
    #[inline]
    pub fn lexicographically_largest(&self) -> Choice {
        // If this element's c1 coefficient is lexicographically largest
        // then it is lexicographically largest. Otherwise, in the event
        // the c1 coefficient is zero and the c0 coefficient is
        // lexicographically largest, then this element is lexicographically
        // largest.

        self.c1.lexicographically_largest()
            | (self.c1.is_zero() & self.c0.lexicographically_largest())
    }

    pub const fn square(&self) -> Fp2 {
        // Complex squaring:
        //
        // v0  = c0 * c1
        // c0' = (c0 + c1) * (c0 + \beta*c1) - v0 - \beta * v0
        // c1' = 2 * v0
        //
        // In BLS12-381's F_{p^2}, our \beta is -1 so we
        // can modify this formula:
        //
        // c0' = (c0 + c1) * (c0 - c1)
        // c1' = 2 * c0 * c1

        let a = (&self.c0).add(&self.c1);
        let b = (&self.c0).sub(&self.c1);
        let c = (&self.c0).add(&self.c0);

        Fp2 {
            c0: (&a).mul(&b),
            c1: (&c).mul(&self.c1),
        }
    }

    pub fn mul(&self, rhs: &Fp2) -> Fp2 {
        // F_{p^2} x F_{p^2} multiplication implemented with operand scanning (schoolbook)
        // computes the result as:
        //
        //   a·b = (a_0 b_0 + a_1 b_1 β) + (a_0 b_1 + a_1 b_0)i
        //
        // In BLS12-381's F_{p^2}, our β is -1, so the resulting F_{p^2} element is:
        //
        //   c_0 = a_0 b_0 - a_1 b_1
        //   c_1 = a_0 b_1 + a_1 b_0
        //
        // Each of these is a "sum of products", which we can compute efficiently.

        Fp2 {
            c0: Fp::sum_of_products([self.c0, -self.c1], [rhs.c0, rhs.c1]),
            c1: Fp::sum_of_products([self.c0, self.c1], [rhs.c1, rhs.c0]),
        }
    }

    pub fn mul_assign(&mut self, other: &Self) {
        let mut t1 = self.c0 * other.c0;
        let mut t0 = self.c0 + self.c1;
        let t2 = self.c1 * other.c1;
        self.c1 = other.c0 + other.c1;
        self.c0 = t1 - t2;
        t1 += t2;
        t0 *= self.c1;
        self.c1 = t0 - t1;
    }

    pub fn square_assign(&mut self) {
        let ab = self.c0 * self.c1;
        let c0c1 = self.c0 + self.c1;
        let mut c0 = -self.c1;
        c0 += self.c0;
        c0 *= c0c1;
        c0 -= ab;
        self.c1 = ab.double();
        self.c0 = c0 + ab;
    }


    pub const fn add(&self, rhs: &Fp2) -> Fp2 {
        Fp2 {
            c0: (&self.c0).add(&rhs.c0),
            c1: (&self.c1).add(&rhs.c1),
        }
    }

    pub const fn sub(&self, rhs: &Fp2) -> Fp2 {
        Fp2 {
            c0: (&self.c0).sub(&rhs.c0),
            c1: (&self.c1).sub(&rhs.c1),
        }
    }

    pub const fn neg(&self) -> Fp2 {
        Fp2 {
            c0: (&self.c0).neg(),
            c1: (&self.c1).neg(),
        }
    }

    pub fn sqrt(&self) -> CtOption<Self> {
        // Algorithm 9, https://eprint.iacr.org/2012/685.pdf
        // with constant time modifications.

        CtOption::new(Fp2::zero(), self.is_zero()).or_else(|| {
            // a1 = self^((p - 3) / 4)
            let a1 = self.pow_vartime(&[
                0xee7f_bfff_ffff_eaaa,
                0x07aa_ffff_ac54_ffff,
                0xd9cc_34a8_3dac_3d89,
                0xd91d_d2e1_3ce1_44af,
                0x92c6_e9ed_90d2_eb35,
                0x0680_447a_8e5f_f9a6,
            ]);

            // alpha = a1^2 * self = self^((p - 3) / 2 + 1) = self^((p - 1) / 2)
            let alpha: Fp2 = a1.square() * self;

            // x0 = self^((p + 1) / 4)
            let x0: Fp2 = a1 * self;

            // In the event that alpha = -1, the element is order p - 1 and so
            // we're just trying to get the square of an element of the subfield
            // Fp. This is given by x0 * u, since u = sqrt(-1). Since the element
            // x0 = a + bu has b = 0, the solution is therefore au.
            CtOption::new(
                Fp2 {
                    c0: -x0.c1,
                    c1: x0.c0,
                },
                alpha.ct_eq(&(&Fp2::one()).neg()),
            )
            // Otherwise, the correct solution is (1 + alpha)^((q - 1) // 2) * x0
            .or_else(|| {
                CtOption::new(
                    (alpha + Fp2::one()).pow_vartime(&[
                        0xdcff_7fff_ffff_d555,
                        0x0f55_ffff_58a9_ffff,
                        0xb398_6950_7b58_7b12,
                        0xb23b_a5c2_79c2_895f,
                        0x258d_d3db_21a5_d66b,
                        0x0d00_88f5_1cbf_f34d,
                    ]) * x0,
                    Choice::from(1),
                )
            })
            // Only return the result if it's really the square root (and so
            // self is actually quadratic nonresidue)
            .and_then(|sqrt| CtOption::new(sqrt, sqrt.square().ct_eq(self)))
        })
    }

    /// Computes the multiplicative inverse of this field
    /// element, returning None in the case that this element
    /// is zero.
    pub fn invert(&self) -> CtOption<Self> {
        // We wish to find the multiplicative inverse of a nonzero
        // element a + bu in Fp2. We leverage an identity
        //
        // (a + bu)(a - bu) = a^2 + b^2
        //
        // which holds because u^2 = -1. This can be rewritten as
        //
        // (a + bu)(a - bu)/(a^2 + b^2) = 1
        //
        // because a^2 + b^2 = 0 has no nonzero solutions for (a, b).
        // This gives that (a - bu)/(a^2 + b^2) is the inverse
        // of (a + bu). Importantly, this can be computing using
        // only a single inversion in Fp.

        (self.c0.square() + self.c1.square()).invert().map(|t| Fp2 {
            c0: self.c0 * t,
            c1: self.c1 * -t,
        })
    }

    /// Although this is labeled "vartime", it is only
    /// variable time with respect to the exponent. It
    /// is also not exposed in the public API.
    pub fn pow_vartime(&self, by: &[u64; 6]) -> Self {
        let mut res = Self::one();
        for e in by.iter().rev() {
            for i in (0..64).rev() {
                res = res.square();

                if ((*e >> i) & 1) == 1 {
                    res *= self;
                }
            }
        }
        res
    }

    pub const fn size() -> usize {
        96
    }
    /// Vartime exponentiation for larger exponents, only
    /// used in testing and not exposed through the public API.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn pow_vartime_extended(&self, by: &[u64]) -> Self {
        let mut res = Self::one();
        for e in by.iter().rev() {
            for i in (0..64).rev() {
                res = res.square();

                if ((*e >> i) & 1) == 1 {
                    res *= self;
                }
            }
        }
        res
    }
}

#[test]
fn test_conditional_selection() {
    let a = Fp2 {
        c0: Fp::from_raw_unchecked([1, 2, 3, 4, 5, 6]),
        c1: Fp::from_raw_unchecked([7, 8, 9, 10, 11, 12]),
    };
    let b = Fp2 {
        c0: Fp::from_raw_unchecked([13, 14, 15, 16, 17, 18]),
        c1: Fp::from_raw_unchecked([19, 20, 21, 22, 23, 24]),
    };

    assert_eq!(
        ConditionallySelectable::conditional_select(&a, &b, Choice::from(0u8)),
        a
    );
    assert_eq!(
        ConditionallySelectable::conditional_select(&a, &b, Choice::from(1u8)),
        b
    );
}

#[test]
fn test_equality() {
    fn is_equal(a: &Fp2, b: &Fp2) -> bool {
        let eq = a == b;
        let ct_eq = a.ct_eq(b);

        assert_eq!(eq, bool::from(ct_eq));

        eq
    }

    assert!(is_equal(
        &Fp2 {
            c0: Fp::from_raw_unchecked([1, 2, 3, 4, 5, 6]),
            c1: Fp::from_raw_unchecked([7, 8, 9, 10, 11, 12]),
        },
        &Fp2 {
            c0: Fp::from_raw_unchecked([1, 2, 3, 4, 5, 6]),
            c1: Fp::from_raw_unchecked([7, 8, 9, 10, 11, 12]),
        }
    ));

    assert!(!is_equal(
        &Fp2 {
            c0: Fp::from_raw_unchecked([2, 2, 3, 4, 5, 6]),
            c1: Fp::from_raw_unchecked([7, 8, 9, 10, 11, 12]),
        },
        &Fp2 {
            c0: Fp::from_raw_unchecked([1, 2, 3, 4, 5, 6]),
            c1: Fp::from_raw_unchecked([7, 8, 9, 10, 11, 12]),
        }
    ));

    assert!(!is_equal(
        &Fp2 {
            c0: Fp::from_raw_unchecked([1, 2, 3, 4, 5, 6]),
            c1: Fp::from_raw_unchecked([2, 8, 9, 10, 11, 12]),
        },
        &Fp2 {
            c0: Fp::from_raw_unchecked([1, 2, 3, 4, 5, 6]),
            c1: Fp::from_raw_unchecked([7, 8, 9, 10, 11, 12]),
        }
    ));
}

#[test]
fn test_squaring() {
    let a = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xc9a2_1831_63ee_70d4,
            0xbc37_70a7_196b_5c91,
            0xa247_f8c1_304c_5f44,
            0xb01f_c2a3_726c_80b5,
            0xe1d2_93e5_bbd9_19c9,
            0x04b7_8e80_020e_f2ca,
        ]),
        c1: Fp::from_raw_unchecked([
            0x952e_a446_0462_618f,
            0x238d_5edd_f025_c62f,
            0xf6c9_4b01_2ea9_2e72,
            0x03ce_24ea_c1c9_3808,
            0x0559_50f9_45da_483c,
            0x010a_768d_0df4_eabc,
        ]),
    };
    let b = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xa1e0_9175_a4d2_c1fe,
            0x8b33_acfc_204e_ff12,
            0xe244_15a1_1b45_6e42,
            0x61d9_96b1_b6ee_1936,
            0x1164_dbe8_667c_853c,
            0x0788_557a_cc7d_9c79,
        ]),
        c1: Fp::from_raw_unchecked([
            0xda6a_87cc_6f48_fa36,
            0x0fc7_b488_277c_1903,
            0x9445_ac4a_dc44_8187,
            0x0261_6d5b_c909_9209,
            0xdbed_4677_2db5_8d48,
            0x11b9_4d50_76c7_b7b1,
        ]),
    };

    assert_eq!(a.square(), b);
}

#[test]
fn test_multiplication() {
    let a = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xc9a2_1831_63ee_70d4,
            0xbc37_70a7_196b_5c91,
            0xa247_f8c1_304c_5f44,
            0xb01f_c2a3_726c_80b5,
            0xe1d2_93e5_bbd9_19c9,
            0x04b7_8e80_020e_f2ca,
        ]),
        c1: Fp::from_raw_unchecked([
            0x952e_a446_0462_618f,
            0x238d_5edd_f025_c62f,
            0xf6c9_4b01_2ea9_2e72,
            0x03ce_24ea_c1c9_3808,
            0x0559_50f9_45da_483c,
            0x010a_768d_0df4_eabc,
        ]),
    };
    let b = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xa1e0_9175_a4d2_c1fe,
            0x8b33_acfc_204e_ff12,
            0xe244_15a1_1b45_6e42,
            0x61d9_96b1_b6ee_1936,
            0x1164_dbe8_667c_853c,
            0x0788_557a_cc7d_9c79,
        ]),
        c1: Fp::from_raw_unchecked([
            0xda6a_87cc_6f48_fa36,
            0x0fc7_b488_277c_1903,
            0x9445_ac4a_dc44_8187,
            0x0261_6d5b_c909_9209,
            0xdbed_4677_2db5_8d48,
            0x11b9_4d50_76c7_b7b1,
        ]),
    };
    let c = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xf597_483e_27b4_e0f7,
            0x610f_badf_811d_ae5f,
            0x8432_af91_7714_327a,
            0x6a9a_9603_cf88_f09e,
            0xf05a_7bf8_bad0_eb01,
            0x0954_9131_c003_ffae,
        ]),
        c1: Fp::from_raw_unchecked([
            0x963b_02d0_f93d_37cd,
            0xc95c_e1cd_b30a_73d4,
            0x3087_25fa_3126_f9b8,
            0x56da_3c16_7fab_0d50,
            0x6b50_86b5_f4b6_d6af,
            0x09c3_9f06_2f18_e9f2,
        ]),
    };

    assert_eq!(a * b, c);
}

#[test]
fn test_addition() {
    let a = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xc9a2_1831_63ee_70d4,
            0xbc37_70a7_196b_5c91,
            0xa247_f8c1_304c_5f44,
            0xb01f_c2a3_726c_80b5,
            0xe1d2_93e5_bbd9_19c9,
            0x04b7_8e80_020e_f2ca,
        ]),
        c1: Fp::from_raw_unchecked([
            0x952e_a446_0462_618f,
            0x238d_5edd_f025_c62f,
            0xf6c9_4b01_2ea9_2e72,
            0x03ce_24ea_c1c9_3808,
            0x0559_50f9_45da_483c,
            0x010a_768d_0df4_eabc,
        ]),
    };
    let b = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xa1e0_9175_a4d2_c1fe,
            0x8b33_acfc_204e_ff12,
            0xe244_15a1_1b45_6e42,
            0x61d9_96b1_b6ee_1936,
            0x1164_dbe8_667c_853c,
            0x0788_557a_cc7d_9c79,
        ]),
        c1: Fp::from_raw_unchecked([
            0xda6a_87cc_6f48_fa36,
            0x0fc7_b488_277c_1903,
            0x9445_ac4a_dc44_8187,
            0x0261_6d5b_c909_9209,
            0xdbed_4677_2db5_8d48,
            0x11b9_4d50_76c7_b7b1,
        ]),
    };
    let c = Fp2 {
        c0: Fp::from_raw_unchecked([
            0x6b82_a9a7_08c1_32d2,
            0x476b_1da3_39ba_5ba4,
            0x848c_0e62_4b91_cd87,
            0x11f9_5955_295a_99ec,
            0xf337_6fce_2255_9f06,
            0x0c3f_e3fa_ce8c_8f43,
        ]),
        c1: Fp::from_raw_unchecked([
            0x6f99_2c12_73ab_5bc5,
            0x3355_1366_17a1_df33,
            0x8b0e_f74c_0aed_aff9,
            0x062f_9246_8ad2_ca12,
            0xe146_9770_738f_d584,
            0x12c3_c3dd_84bc_a26d,
        ]),
    };

    assert_eq!(a + b, c);
}

#[test]
fn test_subtraction() {
    let a = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xc9a2_1831_63ee_70d4,
            0xbc37_70a7_196b_5c91,
            0xa247_f8c1_304c_5f44,
            0xb01f_c2a3_726c_80b5,
            0xe1d2_93e5_bbd9_19c9,
            0x04b7_8e80_020e_f2ca,
        ]),
        c1: Fp::from_raw_unchecked([
            0x952e_a446_0462_618f,
            0x238d_5edd_f025_c62f,
            0xf6c9_4b01_2ea9_2e72,
            0x03ce_24ea_c1c9_3808,
            0x0559_50f9_45da_483c,
            0x010a_768d_0df4_eabc,
        ]),
    };
    let b = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xa1e0_9175_a4d2_c1fe,
            0x8b33_acfc_204e_ff12,
            0xe244_15a1_1b45_6e42,
            0x61d9_96b1_b6ee_1936,
            0x1164_dbe8_667c_853c,
            0x0788_557a_cc7d_9c79,
        ]),
        c1: Fp::from_raw_unchecked([
            0xda6a_87cc_6f48_fa36,
            0x0fc7_b488_277c_1903,
            0x9445_ac4a_dc44_8187,
            0x0261_6d5b_c909_9209,
            0xdbed_4677_2db5_8d48,
            0x11b9_4d50_76c7_b7b1,
        ]),
    };
    let c = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xe1c0_86bb_bf1b_5981,
            0x4faf_c3a9_aa70_5d7e,
            0x2734_b5c1_0bb7_e726,
            0xb2bd_7776_af03_7a3e,
            0x1b89_5fb3_98a8_4164,
            0x1730_4aef_6f11_3cec,
        ]),
        c1: Fp::from_raw_unchecked([
            0x74c3_1c79_9519_1204,
            0x3271_aa54_79fd_ad2b,
            0xc9b4_7157_4915_a30f,
            0x65e4_0313_ec44_b8be,
            0x7487_b238_5b70_67cb,
            0x0952_3b26_d0ad_19a4,
        ]),
    };

    assert_eq!(a - b, c);
}

#[test]
fn test_negation() {
    let a = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xc9a2_1831_63ee_70d4,
            0xbc37_70a7_196b_5c91,
            0xa247_f8c1_304c_5f44,
            0xb01f_c2a3_726c_80b5,
            0xe1d2_93e5_bbd9_19c9,
            0x04b7_8e80_020e_f2ca,
        ]),
        c1: Fp::from_raw_unchecked([
            0x952e_a446_0462_618f,
            0x238d_5edd_f025_c62f,
            0xf6c9_4b01_2ea9_2e72,
            0x03ce_24ea_c1c9_3808,
            0x0559_50f9_45da_483c,
            0x010a_768d_0df4_eabc,
        ]),
    };
    let b = Fp2 {
        c0: Fp::from_raw_unchecked([
            0xf05c_e7ce_9c11_39d7,
            0x6274_8f57_97e8_a36d,
            0xc4e8_d9df_c664_96df,
            0xb457_88e1_8118_9209,
            0x6949_13d0_8772_930d,
            0x1549_836a_3770_f3cf,
        ]),
        c1: Fp::from_raw_unchecked([
            0x24d0_5bb9_fb9d_491c,
            0xfb1e_a120_c12e_39d0,
            0x7067_879f_c807_c7b1,
            0x60a9_269a_31bb_dab6,
            0x45c2_56bc_fd71_649b,
            0x18f6_9b5d_2b8a_fbde,
        ]),
    };

    assert_eq!(-a, b);
}

#[test]
fn test_sqrt() {
    // a = 1488924004771393321054797166853618474668089414631333405711627789629391903630694737978065425271543178763948256226639*u + 784063022264861764559335808165825052288770346101304131934508881646553551234697082295473567906267937225174620141295
    let a = Fp2 {
        c0: Fp::from_raw_unchecked([
            0x2bee_d146_27d7_f9e9,
            0xb661_4e06_660e_5dce,
            0x06c4_cc7c_2f91_d42c,
            0x996d_7847_4b7a_63cc,
            0xebae_bc4c_820d_574e,
            0x1886_5e12_d93f_d845,
        ]),
        c1: Fp::from_raw_unchecked([
            0x7d82_8664_baf4_f566,
            0xd17e_6639_96ec_7339,
            0x679e_ad55_cb40_78d0,
            0xfe3b_2260_e001_ec28,
            0x3059_93d0_43d9_1b68,
            0x0626_f03c_0489_b72d,
        ]),
    };

    assert_eq!(a.sqrt().unwrap().square(), a);

    // b = 5, which is a generator of the p - 1 order
    // multiplicative subgroup
    let b = Fp2 {
        c0: Fp::from_raw_unchecked([
            0x6631_0000_0010_5545,
            0x2114_0040_0eec_000d,
            0x3fa7_af30_c820_e316,
            0xc52a_8b8d_6387_695d,
            0x9fb4_e61d_1e83_eac5,
            0x005c_b922_afe8_4dc7,
        ]),
        c1: Fp::zero(),
    };

    assert_eq!(b.sqrt().unwrap().square(), b);

    // c = 25, which is a generator of the (p - 1) / 2 order
    // multiplicative subgroup
    let c = Fp2 {
        c0: Fp::from_raw_unchecked([
            0x44f6_0000_0051_ffae,
            0x86b8_0141_9948_0043,
            0xd715_9952_f1f3_794a,
            0x755d_6e3d_fe1f_fc12,
            0xd36c_d6db_5547_e905,
            0x02f8_c8ec_bf18_67bb,
        ]),
        c1: Fp::zero(),
    };

    assert_eq!(c.sqrt().unwrap().square(), c);

    // 2155129644831861015726826462986972654175647013268275306775721078997042729172900466542651176384766902407257452753362*u + 2796889544896299244102912275102369318775038861758288697415827248356648685135290329705805931514906495247464901062529
    // is nonsquare.
    assert!(bool::from(
        Fp2 {
            c0: Fp::from_raw_unchecked([
                0xc5fa_1bc8_fd00_d7f6,
                0x3830_ca45_4606_003b,
                0x2b28_7f11_04b1_02da,
                0xa7fb_30f2_8230_f23e,
                0x339c_db9e_e953_dbf0,
                0x0d78_ec51_d989_fc57,
            ]),
            c1: Fp::from_raw_unchecked([
                0x27ec_4898_cf87_f613,
                0x9de1_394e_1abb_05a5,
                0x0947_f85d_c170_fc14,
                0x586f_bc69_6b61_14b7,
                0x2b34_75a4_077d_7169,
                0x13e1_c895_cc4b_6c22,
            ])
        }
        .sqrt()
        .is_none()
    ));
}

#[test]
fn test_inversion() {
    let a = Fp2 {
        c0: Fp::from_raw_unchecked([
            0x1128_ecad_6754_9455,
            0x9e7a_1cff_3a4e_a1a8,
            0xeb20_8d51_e08b_cf27,
            0xe98a_d408_11f5_fc2b,
            0x736c_3a59_232d_511d,
            0x10ac_d42d_29cf_cbb6,
        ]),
        c1: Fp::from_raw_unchecked([
            0xd328_e37c_c2f5_8d41,
            0x948d_f085_8a60_5869,
            0x6032_f9d5_6f93_a573,
            0x2be4_83ef_3fff_dc87,
            0x30ef_61f8_8f48_3c2a,
            0x1333_f55a_3572_5be0,
        ]),
    };

    let b = Fp2 {
        c0: Fp::from_raw_unchecked([
            0x0581_a133_3d4f_48a6,
            0x5824_2f6e_f074_8500,
            0x0292_c955_349e_6da5,
            0xba37_721d_dd95_fcd0,
            0x70d1_6790_3aa5_dfc5,
            0x1189_5e11_8b58_a9d5,
        ]),
        c1: Fp::from_raw_unchecked([
            0x0eda_09d2_d7a8_5d17,
            0x8808_e137_a7d1_a2cf,
            0x43ae_2625_c1ff_21db,
            0xf85a_c9fd_f7a7_4c64,
            0x8fcc_dda5_b8da_9738,
            0x08e8_4f0c_b32c_d17d,
        ]),
    };

    assert_eq!(a.invert().unwrap(), b);

    assert!(bool::from(Fp2::zero().invert().is_none()));
}

#[test]
fn test_lexicographic_largest() {
    assert!(!bool::from(Fp2::zero().lexicographically_largest()));
    assert!(!bool::from(Fp2::one().lexicographically_largest()));
    assert!(bool::from(
        Fp2 {
            c0: Fp::from_raw_unchecked([
                0x1128_ecad_6754_9455,
                0x9e7a_1cff_3a4e_a1a8,
                0xeb20_8d51_e08b_cf27,
                0xe98a_d408_11f5_fc2b,
                0x736c_3a59_232d_511d,
                0x10ac_d42d_29cf_cbb6,
            ]),
            c1: Fp::from_raw_unchecked([
                0xd328_e37c_c2f5_8d41,
                0x948d_f085_8a60_5869,
                0x6032_f9d5_6f93_a573,
                0x2be4_83ef_3fff_dc87,
                0x30ef_61f8_8f48_3c2a,
                0x1333_f55a_3572_5be0,
            ]),
        }
        .lexicographically_largest()
    ));
    assert!(!bool::from(
        Fp2 {
            c0: -Fp::from_raw_unchecked([
                0x1128_ecad_6754_9455,
                0x9e7a_1cff_3a4e_a1a8,
                0xeb20_8d51_e08b_cf27,
                0xe98a_d408_11f5_fc2b,
                0x736c_3a59_232d_511d,
                0x10ac_d42d_29cf_cbb6,
            ]),
            c1: -Fp::from_raw_unchecked([
                0xd328_e37c_c2f5_8d41,
                0x948d_f085_8a60_5869,
                0x6032_f9d5_6f93_a573,
                0x2be4_83ef_3fff_dc87,
                0x30ef_61f8_8f48_3c2a,
                0x1333_f55a_3572_5be0,
            ]),
        }
        .lexicographically_largest()
    ));
    assert!(!bool::from(
        Fp2 {
            c0: Fp::from_raw_unchecked([
                0x1128_ecad_6754_9455,
                0x9e7a_1cff_3a4e_a1a8,
                0xeb20_8d51_e08b_cf27,
                0xe98a_d408_11f5_fc2b,
                0x736c_3a59_232d_511d,
                0x10ac_d42d_29cf_cbb6,
            ]),
            c1: Fp::zero(),
        }
        .lexicographically_largest()
    ));
    assert!(bool::from(
        Fp2 {
            c0: -Fp::from_raw_unchecked([
                0x1128_ecad_6754_9455,
                0x9e7a_1cff_3a4e_a1a8,
                0xeb20_8d51_e08b_cf27,
                0xe98a_d408_11f5_fc2b,
                0x736c_3a59_232d_511d,
                0x10ac_d42d_29cf_cbb6,
            ]),
            c1: Fp::zero(),
        }
        .lexicographically_largest()
    ));
}

#[cfg(feature = "zeroize")]
#[test]
fn test_zeroize() {
    use zeroize::Zeroize;

    let mut a = Fp2::one();
    a.zeroize();
    assert!(bool::from(a.is_zero()));
}

#[cfg(test)]
use rand::SeedableRng;
#[cfg(test)]
use rand_xorshift::XorShiftRng;

// #[test]
// fn test_ser() {
//     let mut rng = XorShiftRng::from_seed([
//         0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
//         0xe5,
//     ]);

//     let a0 = Fp2::random(&mut rng);
//     let a_bytes = a0.to_bytes();
//     let a1 = Fp2::from_bytes(&a_bytes).unwrap();
//     assert_eq!(a0, a1);
// }

#[test]
fn test_fq2_ordering() {
    let mut a = Fp2 {
        c0: Fp::zero(),
        c1: Fp::zero(),
    };

    let mut b = a;

    assert!(a.cmp(&b) == Ordering::Equal);
    b.c0 += &Fp::one();
    assert!(a.cmp(&b) == Ordering::Less);
    a.c0 += &Fp::one();
    assert!(a.cmp(&b) == Ordering::Equal);
    b.c1 += &Fp::one();
    assert!(a.cmp(&b) == Ordering::Less);
    a.c0 += &Fp::one();
    assert!(a.cmp(&b) == Ordering::Less);
    a.c1 += &Fp::one();
    assert!(a.cmp(&b) == Ordering::Greater);
    b.c0 += &Fp::one();
    assert!(a.cmp(&b) == Ordering::Equal);
}

#[test]
fn test_fp2_basics() {
    assert_eq!(
        Fp2 {
            c0: Fp::zero(),
            c1: Fp::zero(),
        },
        Fp2::ZERO
    );
    assert_eq!(
        Fp2 {
            c0: Fp::one(),
            c1: Fp::zero(),
        },
        Fp2::ONE
    );
    assert_eq!(Fp2::ZERO.is_zero().unwrap_u8(), 1);
    assert_eq!(Fp2::ONE.is_zero().unwrap_u8(), 0);
    assert_eq!(
        Fp2 {
            c0: Fp::zero(),
            c1: Fp::one(),
        }
        .is_zero()
        .unwrap_u8(),
        0
    );
}

#[test]
fn test_fp2_squaring() {
    let mut a = Fp2 {
        c0: Fp::one(),
        c1: Fp::one(),
    }; // u + 1
    a.square_assign();
    assert_eq!(
        a,
        Fp2 {
            c0: Fp::zero(),
            c1: Fp::one() + Fp::one(),
        }
    ); // 2u

    let mut a = Fp2 {
        c0: Fp::zero(),
        c1: Fp::one(),
    }; // u
    a.square_assign();
    assert_eq!(a, {
        let neg1 = -Fp::one();
        Fp2 {
            c0: neg1,
            c1: Fp::zero(),
        }
    }); // -1
}

// #[test]
// fn test_fp2_mul_nonresidue() {
//     let mut rng = XorShiftRng::from_seed([
//         0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
//         0xe5,
//     ]);
//     let nine = Fp::one().double().double().double() + Fp::one();
//     let nqr = Fp2 {
//         c0: nine,
//         c1: Fp::one(),
//     };

//     for _ in 0..1000 {
//         let mut a = Fp2::random(&mut rng);
//         let mut b = a;
//         a.mul_by_nonresidue();
//         b.mul_assign(&nqr);

//         assert_eq!(a, b);
//     }
// }

// #[test]
// fn test_fp2_legendre() {
//     assert_eq!(LegendreSymbol::Zero, Fp2::ZERO.legendre());
//     // i^2 = -1
//     let mut m1 = Fp2::ONE;
//     m1 = m1.neg();
//     assert_eq!(LegendreSymbol::QuadraticResidue, m1.legendre());
//     m1.mul_by_nonresidue();
//     assert_eq!(LegendreSymbol::QuadraticNonResidue, m1.legendre());
// }

#[test]
fn test_sqrt_2() {
    let mut rng = XorShiftRng::from_seed([
        0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
        0xe5,
    ]);

    for _ in 0..10000 {
        let a = Fp2::random(&mut rng);
        // if a.legendre() == LegendreSymbol::QuadraticNonResidue {
        //     assert!(bool::from(a.sqrt().is_none()));
        // }
    }

    for _ in 0..10000 {
        let a = Fp2::random(&mut rng);
        let mut b = a;
        b.square_assign();
        //assert_eq!(b.legendre(), LegendreSymbol::QuadraticResidue);

        let b = b.sqrt().unwrap();
        let mut negb = b;
        negb = negb.neg();

        assert!(a == b || a == negb);
    }

    let mut c = Fp2::ONE;
    for _ in 0..10000 {
        let mut b = c;
        b.square_assign();
        //assert_eq!(b.legendre(), LegendreSymbol::QuadraticResidue);

        b = b.sqrt().unwrap();

        if b != c {
            b = b.neg();
        }

        assert_eq!(b, c);

        c += &Fp2::ONE;
    }
}

// #[test]
// fn test_frobenius() {
//     let mut rng = XorShiftRng::from_seed([
//         0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06, 0xbc,
//         0xe5,
//     ]);

//     for _ in 0..100 {
//         for i in 0..14 {
//             let mut a = Fp2::random(&mut rng);
//             let mut b = a;

//             for _ in 0..i {
//                 a = a.pow([
//                     0x3c208c16d87cfd47,
//                     0x97816a916871ca8d,
//                     0xb85045b68181585d,
//                     0x30644e72e131a029,
//                 ]);
//             }
//             b.frobenius_map();

//             assert_eq!(a, b);
//         }
//     }
// }

#[test]
fn test_field() {
    crate::tests::field::random_field_tests::<Fp2>("Fp2".to_string());
}

#[test]
fn test_serialization() {
    crate::tests::field::random_serialization_test::<Fp2>("Fp2".to_string());
    #[cfg(feature = "derive_serde")]
    crate::tests::field::random_serde_test::<Fp2>("Fp2".to_string());
}
