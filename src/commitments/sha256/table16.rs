use halo2_proofs::halo2curves::ff::PrimeField;
use std::convert::TryInto;
use std::marker::PhantomData;

use super::Sha256Instructions;
use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, Region, Value},
    plonk::{Advice, Any, Assigned, Column, ConstraintSystem, Error},
};

mod compression;
mod gates;
mod message_schedule;
mod spread_table;
mod util;

use compression::*;
pub use compression::State;
use gates::*;
use message_schedule::*;
use spread_table::*;
use util::*;

const ROUNDS: usize = 64;
pub const STATE: usize = 8;

#[allow(clippy::unreadable_literal)]
pub(crate) const ROUND_CONSTANTS: [u32; ROUNDS] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub const IV: [u32; STATE] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

#[derive(Clone, Copy, Debug, Default)]
/// A word in a `Table16` message block.
// TODO: Make the internals of this struct private.
pub struct BlockWord(pub Value<u32>);

#[derive(Clone, Debug)]
/// Little-endian bits (up to 64 bits)
pub struct Bits<const LEN: usize>([bool; LEN]);

impl<const LEN: usize> Bits<LEN> {
    fn spread<const SPREAD: usize>(&self) -> [bool; SPREAD] {
        spread_bits(self.0)
    }
}

impl<const LEN: usize> std::ops::Deref for Bits<LEN> {
    type Target = [bool; LEN];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const LEN: usize> From<[bool; LEN]> for Bits<LEN> {
    fn from(bits: [bool; LEN]) -> Self {
        Self(bits)
    }
}

impl<const LEN: usize> From<&Bits<LEN>> for [bool; LEN] {
    fn from(bits: &Bits<LEN>) -> Self {
        bits.0
    }
}

impl<F: PrimeField, const LEN: usize> From<&Bits<LEN>> for Assigned<F> {
    fn from(bits: &Bits<LEN>) -> Assigned<F> {
        assert!(LEN <= 64);
        F::from(lebs2ip(&bits.0)).into()
    }
}

impl From<&Bits<16>> for u16 {
    fn from(bits: &Bits<16>) -> u16 {
        lebs2ip(&bits.0) as u16
    }
}

impl From<u16> for Bits<16> {
    fn from(int: u16) -> Bits<16> {
        Bits(i2lebsp::<16>(int.into()))
    }
}

impl From<&Bits<32>> for u32 {
    fn from(bits: &Bits<32>) -> u32 {
        lebs2ip(&bits.0) as u32
    }
}

impl From<u32> for Bits<32> {
    fn from(int: u32) -> Bits<32> {
        Bits(i2lebsp::<32>(int.into()))
    }
}

#[derive(Clone, Debug)]
pub struct AssignedBits<F: PrimeField, const LEN: usize>(AssignedCell<Bits<LEN>, F>);

impl<F: PrimeField, const LEN: usize> std::ops::Deref for AssignedBits<F, LEN> {
    type Target = AssignedCell<Bits<LEN>, F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: PrimeField, const LEN: usize> AssignedBits<F, LEN> {
    fn assign_bits<A, AR, T: TryInto<[bool; LEN]> + std::fmt::Debug + Clone>(
        region: &mut Region<'_, F>,
        annotation: A,
        column: impl Into<Column<Any>>,
        offset: usize,
        value: Value<T>,
    ) -> Result<Self, Error>
    where
        A: Fn() -> AR,
        AR: Into<String>,
        <T as TryInto<[bool; LEN]>>::Error: std::fmt::Debug,
    {
        let value: Value<[bool; LEN]> = value.map(|v| v.try_into().unwrap());
        let value: Value<Bits<LEN>> = value.map(|v| v.into());

        let column: Column<Any> = column.into();
        match column.column_type() {
            Any::Advice(_) => {
                region.assign_advice(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            Any::Fixed => {
                region.assign_fixed(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            _ => panic!("Cannot assign to instance column"),
        }
        .map(AssignedBits)
    }
}

impl<F: PrimeField> AssignedBits<F, 16> {
    fn value_u16(&self) -> Value<u16> {
        self.value().map(|v| v.into())
    }

    fn assign<A, AR>(
        region: &mut Region<'_, F>,
        annotation: A,
        column: impl Into<Column<Any>>,
        offset: usize,
        value: Value<u16>,
    ) -> Result<Self, Error>
    where
        A: Fn() -> AR,
        AR: Into<String>,
    {
        let column: Column<Any> = column.into();
        let value: Value<Bits<16>> = value.map(|v| v.into());
        match column.column_type() {
            Any::Advice(_) => {
                region.assign_advice(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            Any::Fixed => {
                region.assign_fixed(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            _ => panic!("Cannot assign to instance column"),
        }
        .map(AssignedBits)
    }
}

impl<F: PrimeField> AssignedBits<F, 32> {
    fn value_u32(&self) -> Value<u32> {
        self.value().map(|v| v.into())
    }

    fn assign<A, AR>(
        region: &mut Region<'_, F>,
        annotation: A,
        column: impl Into<Column<Any>>,
        offset: usize,
        value: Value<u32>,
    ) -> Result<Self, Error>
    where
        A: Fn() -> AR,
        AR: Into<String>,
    {
        let column: Column<Any> = column.into();
        let value: Value<Bits<32>> = value.map(|v| v.into());
        match column.column_type() {
            Any::Advice(_) => {
                region.assign_advice(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            Any::Fixed => {
                region.assign_fixed(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            _ => panic!("Cannot assign to instance column"),
        }
        .map(AssignedBits)
    }
}

/// Configuration for a [`Table16Chip<F>`].
#[derive(Clone, Debug)]
pub struct Table16Config {
    lookup: SpreadTableConfig,
    message_schedule: MessageScheduleConfig,
    compression: CompressionConfig,
}

/// A chip that implements SHA-256 with a maximum lookup table size of $2^16$.
#[derive(Clone, Debug)]
pub struct Table16Chip<F: PrimeField> {
    config: Table16Config,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Chip<F> for Table16Chip<F> {
    type Config = Table16Config;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: PrimeField> Table16Chip<F> {
    /// Reconstructs this chip from the given config.
    pub fn construct(config: <Self as Chip<F>>::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    /// Configures a circuit to include this chip.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
    ) -> <Self as Chip<F>>::Config {
        // Columns required by this chip:
        let message_schedule = meta.advice_column();
        let extras = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // - Three advice columns to interact with the lookup table.
        let input_tag = meta.advice_column();
        let input_dense = meta.advice_column();
        let input_spread = meta.advice_column();

        let lookup = SpreadTableChip::configure(meta, input_tag, input_dense, input_spread);
        let lookup_inputs = lookup.input.clone();

        // Rename these here for ease of matching the gates to the specification.
        let _a_0 = lookup_inputs.tag;
        let a_1 = lookup_inputs.dense;
        let a_2 = lookup_inputs.spread;
        let a_3 = extras[0];
        let a_4 = extras[1];
        let a_5 = message_schedule;
        let a_6 = extras[2];
        let a_7 = extras[3];
        let a_8 = extras[4];
        let _a_9 = extras[5];

        // Add all advice columns to permutation
        for column in [a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8].iter() {
            meta.enable_equality(*column);
        }

        let compression =
            CompressionConfig::configure(meta, lookup_inputs.clone(), message_schedule, extras);

        let message_schedule =
            MessageScheduleConfig::configure(meta, lookup_inputs, message_schedule, extras);

        Table16Config {
            lookup,
            message_schedule,
            compression,
        }
    }

    /// Loads the lookup table required by this chip into the circuit.
    pub fn load(
        config: Table16Config,
        layouter: &mut impl Layouter<F>,
    ) -> Result<(), Error> {
        SpreadTableChip::load(config.lookup, layouter)
    }
}

impl<F: PrimeField> Sha256Instructions<F> for Table16Chip<F> {
    type State = State<F>;
    type BlockWord = BlockWord;

    fn initialization_vector(
        &self,
        layouter: &mut impl Layouter<F>,
    ) -> Result<State<F>, Error> {
        self.config().compression.initialize_with_iv(layouter, IV)
    }

    fn initialization(
        &self,
        layouter: &mut impl Layouter<F>,
        init_state: &Self::State,
    ) -> Result<Self::State, Error> {
        self.config()
            .compression
            .initialize_with_state(layouter, init_state.clone())
    }

    // Given an initialized state and an input message block, compress the
    // message block and return the final state.
    fn compress(
        &self,
        layouter: &mut impl Layouter<F>,
        initialized_state: &Self::State,
        input: [Self::BlockWord; super::BLOCK_SIZE],
    ) -> Result<Self::State, Error> {
        let config = self.config();
        let (_, w_halves) = config.message_schedule.process(layouter, input)?;
        config
            .compression
            .compress(layouter, initialized_state.clone(), w_halves)
    }

    fn digest(
        &self,
        layouter: &mut impl Layouter<F>,
        state: &Self::State,
    ) -> Result<[Self::BlockWord; super::DIGEST_SIZE], Error> {
        // Copy the dense forms of the state variable chunks down to this gate.
        // Reconstruct the 32-bit dense words.
        let cells = self.config().compression.digest(layouter, state.clone())?;
        let mut out = [BlockWord(Value::known(0)); super::DIGEST_SIZE];
        for (o, c) in out.iter_mut().zip(cells.iter()) {
            *o = BlockWord(c.value().map(|v| v.to_repr().as_ref()[..4].iter().rev().fold(0u32, |acc, b| (acc << 8) | *b as u32)));
        }
        Ok(out)
    }
}

impl<F: PrimeField> Table16Chip<F> {
    /// Like `Sha256Instructions::compress`, but also returns the cells holding the 16 input
    /// words W_0..W_15 of the message schedule, so that callers can copy-constrain them.
    pub fn compress_with_inputs(
        &self,
        layouter: &mut impl Layouter<F>,
        initialized_state: &State<F>,
        input: [BlockWord; super::BLOCK_SIZE],
    ) -> Result<(State<F>, Vec<AssignedCell<Bits<32>, F>>), Error> {
        let config = self.config();
        let (w, w_halves) = config.message_schedule.process(layouter, input)?;
        let inputs = w[..super::BLOCK_SIZE].iter().map(|w| (**w).0.clone()).collect();
        let state = config
            .compression
            .compress(layouter, initialized_state.clone(), w_halves)?;
        Ok((state, inputs))
    }

    /// Initial state from arbitrary 32-bit words (e.g. a chaining value); the words are decomposed
    /// and range-checked, the caller copy-constrains their dense halves.
    pub fn initialization_from_values(
        &self,
        layouter: &mut impl Layouter<F>,
        words: [Value<u32>; STATE],
    ) -> Result<State<F>, Error> {
        self.config().compression.initialize_with_values(layouter, words)
    }

    /// The digest as eight assigned 32-bit words (big-endian word order of SHA-256).
    /// NOTE: upstream Table16 omits the feed-forward; this is the state after the 64 rounds.
    pub fn digest_cells(
        &self,
        layouter: &mut impl Layouter<F>,
        state: &State<F>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        self.config().compression.digest(layouter, state.clone())
    }
}

/// Common assignment patterns used by Table16 regions.
trait Table16Assignment {
    /// Assign cells for general spread computation used in sigma, ch, ch_neg, maj gates
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn assign_spread_outputs<F: PrimeField>(
        &self,
        region: &mut Region<'_, F>,
        lookup: &SpreadInputs,
        a_3: Column<Advice>,
        row: usize,
        r_0_even: Value<[bool; 16]>,
        r_0_odd: Value<[bool; 16]>,
        r_1_even: Value<[bool; 16]>,
        r_1_odd: Value<[bool; 16]>,
    ) -> Result<
        (
            (AssignedBits<F, 16>, AssignedBits<F, 16>),
            (AssignedBits<F, 16>, AssignedBits<F, 16>),
        ),
        Error,
    > {
        // Lookup R_0^{even}, R_0^{odd}, R_1^{even}, R_1^{odd}
        let r_0_even = SpreadVar::with_lookup(
            region,
            lookup,
            row - 1,
            r_0_even.map(SpreadWord::<16, 32>::new),
        )?;
        let r_0_odd =
            SpreadVar::with_lookup(region, lookup, row, r_0_odd.map(SpreadWord::<16, 32>::new))?;
        let r_1_even = SpreadVar::with_lookup(
            region,
            lookup,
            row + 1,
            r_1_even.map(SpreadWord::<16, 32>::new),
        )?;
        let r_1_odd = SpreadVar::with_lookup(
            region,
            lookup,
            row + 2,
            r_1_odd.map(SpreadWord::<16, 32>::new),
        )?;

        // Assign and copy R_1^{odd}
        r_1_odd
            .spread
            .copy_advice(|| "Assign and copy R_1^{odd}", region, a_3, row)?;

        Ok((
            (r_0_even.dense, r_1_even.dense),
            (r_0_odd.dense, r_1_odd.dense),
        ))
    }

    /// Assign outputs of sigma gates
    #[allow(clippy::too_many_arguments)]
    fn assign_sigma_outputs<F: PrimeField>(
        &self,
        region: &mut Region<'_, F>,
        lookup: &SpreadInputs,
        a_3: Column<Advice>,
        row: usize,
        r_0_even: Value<[bool; 16]>,
        r_0_odd: Value<[bool; 16]>,
        r_1_even: Value<[bool; 16]>,
        r_1_odd: Value<[bool; 16]>,
    ) -> Result<(AssignedBits<F, 16>, AssignedBits<F, 16>), Error> {
        let (even, _odd) = self.assign_spread_outputs(
            region, lookup, a_3, row, r_0_even, r_0_odd, r_1_even, r_1_odd,
        )?;

        Ok(even)
    }
}

