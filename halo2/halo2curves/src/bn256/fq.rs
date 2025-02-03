#[cfg(feature = "asm")]
use crate::bn256::assembly::field_arithmetic_asm;
#[cfg(not(feature = "asm"))]
use crate::{field_arithmetic, field_specific};

use crate::arithmetic::{adc, mac, sbb};
use crate::bn256::LegendreSymbol;
use crate::ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use crate::{
    field_bits, field_common, impl_add_binop_specify_output, impl_binops_additive,
    impl_binops_additive_specify_output, impl_binops_multiplicative,
    impl_binops_multiplicative_mixed, impl_sub_binop_specify_output, impl_sum_prod,
};
use core::convert::TryInto;
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

#[cfg(feature = "derive_serde")]
use serde::{Deserialize, Serialize};

/// This represents an element of $\mathbb{F}_q$ where
///
/// `p = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47`
///
/// is the base field of the BN254 curve.
// The internal representation of this type is four 64-bit unsigned
// integers in little-endian order. `Fq` values are always in
// Montgomery form; i.e., Fq(a) = aR mod q, with R = 2^256.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "derive_serde", derive(Serialize, Deserialize))]
pub struct Fq(pub(crate) [u64; 4]);

/// Constant representing the modulus
/// q = 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
const MODULUS: Fq = Fq([
    0x3c208c16d87cfd47,
    0x97816a916871ca8d,
    0xb85045b68181585d,
    0x30644e72e131a029,
]);

/// The modulus as u32 limbs.
#[cfg(not(target_pointer_width = "64"))]
const MODULUS_LIMBS_32: [u32; 8] = [
    0xd87c_fd47,
    0x3c20_8c16,
    0x6871_ca8d,
    0x9781_6a91,
    0x8181_585d,
    0xb850_45b6,
    0xe131_a029,
    0x3064_4e72,
];

/// INV = -(q^{-1} mod 2^64) mod 2^64
const INV: u64 = 0x87d20782e4866389;

/// R = 2^256 mod q
const R: Fq = Fq([
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
]);

/// R^2 = 2^512 mod q
const R2: Fq = Fq([
    0xf32cfc5b538afa89,
    0xb5e71911d44501fb,
    0x47ab1eff0a417ff6,
    0x06d89f71cab8351f,
]);

/// R^3 = 2^768 mod q
const R3: Fq = Fq([
    0xb1cd6dafda1530df,
    0x62f210e6a7283db6,
    0xef7f0b0c0ada0afb,
    0x20fd6e902d592544,
]);

pub const NEGATIVE_ONE: Fq = Fq([
    0x68c3488912edefaa,
    0x8d087f6872aabf4f,
    0x51e1a24709081231,
    0x2259d6b14729c0fa,
]);

const MODULUS_STR: &str = "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47";

/// Obtained with:
/// `sage: GF(0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47).primitive_element()`
const MULTIPLICATIVE_GENERATOR: Fq = Fq::from_raw([0x03, 0x0, 0x0, 0x0]);

const TWO_INV: Fq = Fq::from_raw([
    0x9e10460b6c3e7ea4,
    0xcbc0b548b438e546,
    0xdc2822db40c0ac2e,
    0x183227397098d014,
]);

// TODO: Can we simply put 0 here::
const ROOT_OF_UNITY: Fq = Fq::zero();

// Unused constant for base field
const ROOT_OF_UNITY_INV: Fq = Fq::zero();

// Unused constant for base field
const DELTA: Fq = Fq::zero();

/// `ZETA^3 = 1 mod r` where `ZETA^2 != 1 mod r`
const ZETA: Fq = Fq::from_raw([
    0x5763473177fffffeu64,
    0xd4f263f1acdb5c4fu64,
    0x59e26bcea0d48bacu64,
    0x0u64,
]);

impl_binops_additive!(Fq, Fq);
impl_binops_multiplicative!(Fq, Fq);
field_common!(
    Fq,
    MODULUS,
    INV,
    MODULUS_STR,
    TWO_INV,
    ROOT_OF_UNITY_INV,
    DELTA,
    ZETA,
    R,
    R2,
    R3
);
impl_sum_prod!(Fq);

#[cfg(not(feature = "asm"))]
field_arithmetic!(Fq, MODULUS, INV, sparse);
#[cfg(feature = "asm")]
field_arithmetic_asm!(Fq, MODULUS, INV);

#[cfg(target_pointer_width = "64")]
field_bits!(Fq, MODULUS);
#[cfg(not(target_pointer_width = "64"))]
field_bits!(Fq, MODULUS, MODULUS_LIMBS_32);

impl Fq {
    pub const fn size() -> usize {
        32
    }

    pub fn legendre(&self) -> LegendreSymbol {
        // s = self^((modulus - 1) // 2)
        // 0x183227397098d014dc2822db40c0ac2ecbc0b548b438e5469e10460b6c3e7ea3
        let s = &[
            0x9e10460b6c3e7ea3u64,
            0xcbc0b548b438e546u64,
            0xdc2822db40c0ac2eu64,
            0x183227397098d014u64,
        ];
        let s = self.pow(s);
        if s == Self::zero() {
            LegendreSymbol::Zero
        } else if s == Self::one() {
            LegendreSymbol::QuadraticResidue
        } else {
            LegendreSymbol::QuadraticNonResidue
        }
    }
}

impl ff::Field for Fq {
    const ZERO: Self = Self::zero();
    const ONE: Self = Self::one();

    fn random(mut rng: impl RngCore) -> Self {
        let mut random_bytes = [0; 64];
        rng.fill_bytes(&mut random_bytes[..]);

        Self::from_uniform_bytes(&random_bytes)
    }

    fn double(&self) -> Self {
        self.double()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        self.square()
    }

    /// Computes the square root of this element, if it exists.
    fn sqrt(&self) -> CtOption<Self> {
        let tmp = self.pow([
            0x4f082305b61f3f52,
            0x65e05aa45a1c72a3,
            0x6e14116da0605617,
            0x0c19139cb84c680a,
        ]);

        CtOption::new(tmp, tmp.square().ct_eq(self))
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        ff::helpers::sqrt_ratio_generic(num, div)
    }

    /// Returns the multiplicative inverse of the
    /// element. If it is zero, the method fails.
    fn invert(&self) -> CtOption<Self> {
        self.invert()
    }
}

impl ff::PrimeField for Fq {
    type Repr = [u8; 32];

    const NUM_BITS: u32 = 254;
    const CAPACITY: u32 = 253;
    const MODULUS: &'static str = MODULUS_STR;
    const MULTIPLICATIVE_GENERATOR: Self = MULTIPLICATIVE_GENERATOR;
    const ROOT_OF_UNITY: Self = ROOT_OF_UNITY;
    const ROOT_OF_UNITY_INV: Self = ROOT_OF_UNITY_INV;
    const TWO_INV: Self = TWO_INV;
    const DELTA: Self = DELTA;
    const S: u32 = 0;

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        let mut tmp = Fq([0, 0, 0, 0]);

        tmp.0[0] = u64::from_le_bytes(repr[0..8].try_into().unwrap());
        tmp.0[1] = u64::from_le_bytes(repr[8..16].try_into().unwrap());
        tmp.0[2] = u64::from_le_bytes(repr[16..24].try_into().unwrap());
        tmp.0[3] = u64::from_le_bytes(repr[24..32].try_into().unwrap());

        // Try to subtract the modulus
        let (_, borrow) = sbb(tmp.0[0], MODULUS.0[0], 0);
        let (_, borrow) = sbb(tmp.0[1], MODULUS.0[1], borrow);
        let (_, borrow) = sbb(tmp.0[2], MODULUS.0[2], borrow);
        let (_, borrow) = sbb(tmp.0[3], MODULUS.0[3], borrow);

        // If the element is smaller than MODULUS then the
        // subtraction will underflow, producing a borrow value
        // of 0xffff...ffff. Otherwise, it'll be zero.
        let is_some = (borrow as u8) & 1;

        // Convert to Montgomery form by computing
        // (a.R^0 * R^2) / R = a.R
        tmp *= &R2;

        CtOption::new(tmp, Choice::from(is_some))
    }

    fn to_repr(&self) -> Self::Repr {
        // Turn into canonical form by computing
        // (a.R) / R = a
        let tmp =
            Self::montgomery_reduce(&[self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0]);

        let mut res = [0; 32];
        res[0..8].copy_from_slice(&tmp.0[0].to_le_bytes());
        res[8..16].copy_from_slice(&tmp.0[1].to_le_bytes());
        res[16..24].copy_from_slice(&tmp.0[2].to_le_bytes());
        res[24..32].copy_from_slice(&tmp.0[3].to_le_bytes());

        res
    }

    fn is_odd(&self) -> Choice {
        Choice::from(self.to_repr()[0] & 1)
    }
}

impl FromUniformBytes<64> for Fq {
    /// Converts a 512-bit little endian integer into
    /// an `Fq` by reducing by the modulus.
    fn from_uniform_bytes(bytes: &[u8; 64]) -> Self {
        Self::from_u512([
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
        ])
    }
}

impl WithSmallOrderMulGroup<3> for Fq {
    const ZETA: Self = ZETA;
}

#[cfg(test)]
mod test {
    use super::*;
    use ff::Field;
    use rand_core::OsRng;

    #[test]
    fn test_sqrt_fq() {
        let v = (Fq::TWO_INV).square().sqrt().unwrap();
        assert!(v == Fq::TWO_INV || (-v) == Fq::TWO_INV);

        for _ in 0..10000 {
            let a = Fq::random(OsRng);
            let mut b = a;
            b = b.square();
            assert_eq!(b.legendre(), LegendreSymbol::QuadraticResidue);

            let b = b.sqrt().unwrap();
            let mut negb = b;
            negb = negb.neg();

            assert!(a == b || a == negb);
        }

        let mut c = Fq::one();
        for _ in 0..10000 {
            let mut b = c;
            b = b.square();
            assert_eq!(b.legendre(), LegendreSymbol::QuadraticResidue);

            b = b.sqrt().unwrap();

            if b != c {
                b = b.neg();
            }

            assert_eq!(b, c);

            c += &Fq::one();
        }
    }

    #[test]
    fn test_from_u512() {
        assert_eq!(
            Fq::from_raw([
                0x1f8905a172affa8a,
                0xde45ad177dcf3306,
                0xaaa7987907d73ae2,
                0x24d349431d468e30,
            ]),
            Fq::from_u512([
                0xaaaaaaaaaaaaaaaa,
                0xaaaaaaaaaaaaaaaa,
                0xaaaaaaaaaaaaaaaa,
                0xaaaaaaaaaaaaaaaa,
                0xaaaaaaaaaaaaaaaa,
                0xaaaaaaaaaaaaaaaa,
                0xaaaaaaaaaaaaaaaa,
                0xaaaaaaaaaaaaaaaa
            ])
        );
    }

    #[test]
    fn test_field() {
        crate::tests::field::random_field_tests::<Fq>("fq".to_string());
    }

    #[test]
    #[cfg(feature = "bits")]
    fn test_bits() {
        crate::tests::field::random_bits_tests::<Fq>("fq".to_string());
    }

    #[test]
    fn test_zeta() {
        let zeta = Fq::ZETA;
        println!("Modulus: {:?}", MODULUS);
        println!("One: {:?}", R);
        println!("TWO : {:?}, TWO_INV : {:?}", TWO_INV.invert().unwrap(), TWO_INV);
        println!("Zeta: {:?}", ZETA);
        println!("-1: {:?}, \n -2: {:?}", -Fq::from(1), -Fq::from(2));
        assert_eq!(-Fq::one(), zeta * zeta * zeta);
    }

    #[test]
    fn test_serialization() {
        crate::tests::field::random_serialization_test::<Fq>("fq".to_string());
        //#[cfg(feature = "derive_serde")]
        crate::tests::field::random_serde_test::<Fq>("fq".to_string());
    }
}
