use halo2_proofs::halo2curves::ff::PrimeField;
use super::{util::*, AssignedBits};
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Chip, Layouter, Region, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, TableColumn},
    poly::Rotation,
};
use std::convert::TryInto;
use std::marker::PhantomData;

const BITS_7: usize = 1 << 7;
const BITS_10: usize = 1 << 10;
const BITS_11: usize = 1 << 11;
const BITS_13: usize = 1 << 13;
const BITS_14: usize = 1 << 14;

/// An input word into a lookup, containing (tag, dense, spread)
#[derive(Copy, Clone, Debug)]
pub(super) struct SpreadWord<const DENSE: usize, const SPREAD: usize> {
    pub tag: u8,
    pub dense: [bool; DENSE],
    pub spread: [bool; SPREAD],
}

/// Helper function that returns tag of 16-bit input
pub fn get_tag(input: u16) -> u8 {
    let input = input as usize;
    if input < BITS_7 {
        0
    } else if input < BITS_10 {
        1
    } else if input < BITS_11 {
        2
    } else if input < BITS_13 {
        3
    } else if input < BITS_14 {
        4
    } else {
        5
    }
}

impl<const DENSE: usize, const SPREAD: usize> SpreadWord<DENSE, SPREAD> {
    pub(super) fn new(dense: [bool; DENSE]) -> Self {
        assert!(DENSE <= 16);
        SpreadWord {
            tag: get_tag(lebs2ip(&dense) as u16),
            dense,
            spread: spread_bits(dense),
        }
    }

    pub(super) fn try_new<T: TryInto<[bool; DENSE]> + std::fmt::Debug>(dense: T) -> Self
    where
        <T as TryInto<[bool; DENSE]>>::Error: std::fmt::Debug,
    {
        assert!(DENSE <= 16);
        let dense: [bool; DENSE] = dense.try_into().unwrap();
        SpreadWord {
            tag: get_tag(lebs2ip(&dense) as u16),
            dense,
            spread: spread_bits(dense),
        }
    }
}

/// A variable stored in advice columns corresponding to a row of [`SpreadTableConfig`].
#[derive(Clone, Debug)]
pub(super) struct SpreadVar<F: PrimeField, const DENSE: usize, const SPREAD: usize> {
    pub tag: Value<u8>,
    pub dense: AssignedBits<F, DENSE>,
    pub spread: AssignedBits<F, SPREAD>,
}

impl<F: PrimeField, const DENSE: usize, const SPREAD: usize> SpreadVar<F, DENSE, SPREAD> {
    pub(super) fn with_lookup(
        region: &mut Region<'_, F>,
        cols: &SpreadInputs,
        row: usize,
        word: Value<SpreadWord<DENSE, SPREAD>>,
    ) -> Result<Self, Error> {
        let tag = word.map(|word| word.tag);
        let dense_val = word.map(|word| word.dense);
        let spread_val = word.map(|word| word.spread);

        region.assign_advice(
            || "tag",
            cols.tag,
            row,
            || tag.map(|tag| F::from(tag as u64)),
        )?;

        let dense =
            AssignedBits::<F, DENSE>::assign_bits(region, || "dense", cols.dense, row, dense_val)?;

        let spread =
            AssignedBits::<F, SPREAD>::assign_bits(region, || "spread", cols.spread, row, spread_val)?;

        Ok(SpreadVar { tag, dense, spread })
    }

    pub(super) fn without_lookup(
        region: &mut Region<'_, F>,
        dense_col: Column<Advice>,
        dense_row: usize,
        spread_col: Column<Advice>,
        spread_row: usize,
        word: Value<SpreadWord<DENSE, SPREAD>>,
    ) -> Result<Self, Error> {
        let tag = word.map(|word| word.tag);
        let dense_val = word.map(|word| word.dense);
        let spread_val = word.map(|word| word.spread);

        let dense = AssignedBits::<F, DENSE>::assign_bits(
            region,
            || "dense",
            dense_col,
            dense_row,
            dense_val,
        )?;

        let spread = AssignedBits::<F, SPREAD>::assign_bits(
            region,
            || "spread",
            spread_col,
            spread_row,
            spread_val,
        )?;

        Ok(SpreadVar { tag, dense, spread })
    }
}

#[derive(Clone, Debug)]
pub(super) struct SpreadInputs {
    pub(super) tag: Column<Advice>,
    pub(super) dense: Column<Advice>,
    pub(super) spread: Column<Advice>,
}

#[derive(Clone, Debug)]
pub(super) struct SpreadTable {
    pub(super) tag: TableColumn,
    pub(super) dense: TableColumn,
    pub(super) spread: TableColumn,
}

#[derive(Clone, Debug)]
pub(super) struct SpreadTableConfig {
    pub input: SpreadInputs,
    pub table: SpreadTable,
}

#[derive(Clone, Debug)]
pub(super) struct SpreadTableChip<F: Field> {
    config: SpreadTableConfig,
    _marker: PhantomData<F>,
}

impl<F: Field> Chip<F> for SpreadTableChip<F> {
    type Config = SpreadTableConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: PrimeField> SpreadTableChip<F> {
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        input_tag: Column<Advice>,
        input_dense: Column<Advice>,
        input_spread: Column<Advice>,
    ) -> <Self as Chip<F>>::Config {
        let table_tag = meta.lookup_table_column();
        let table_dense = meta.lookup_table_column();
        let table_spread = meta.lookup_table_column();

        meta.lookup("lookup", |meta| {
            let tag_cur = meta.query_advice(input_tag, Rotation::cur());
            let dense_cur = meta.query_advice(input_dense, Rotation::cur());
            let spread_cur = meta.query_advice(input_spread, Rotation::cur());

            vec![
                (tag_cur, table_tag),
                (dense_cur, table_dense),
                (spread_cur, table_spread),
            ]
        });

        SpreadTableConfig {
            input: SpreadInputs {
                tag: input_tag,
                dense: input_dense,
                spread: input_spread,
            },
            table: SpreadTable {
                tag: table_tag,
                dense: table_dense,
                spread: table_spread,
            },
        }
    }

    pub fn load(
        config: SpreadTableConfig,
        layouter: &mut impl Layouter<F>,
    ) -> Result<<Self as Chip<F>>::Loaded, Error> {
        layouter.assign_table(
            || "spread table",
            |mut table| {
                // We generate the row values lazily (we only need them during keygen).
                let mut rows = SpreadTableConfig::generate::<F>();

                for index in 0..(1 << 16) {
                    let mut row = None;
                    table.assign_cell(
                        || "tag",
                        config.table.tag,
                        index,
                        || {
                            row = rows.next();
                            Value::known(row.map(|(tag, _, _)| tag).unwrap())
                        },
                    )?;
                    table.assign_cell(
                        || "dense",
                        config.table.dense,
                        index,
                        || Value::known(row.map(|(_, dense, _)| dense).unwrap()),
                    )?;
                    table.assign_cell(
                        || "spread",
                        config.table.spread,
                        index,
                        || Value::known(row.map(|(_, _, spread)| spread).unwrap()),
                    )?;
                }

                Ok(())
            },
        )
    }
}

impl SpreadTableConfig {
    fn generate<F: PrimeField>() -> impl Iterator<Item = (F, F, F)> {
        (1..=(1 << 16)).scan((F::ZERO, F::ZERO, F::ZERO), |(tag, dense, spread), i| {
            // We computed this table row in the previous iteration.
            let res = (*tag, *dense, *spread);

            // i holds the zero-indexed row number for the next table row.
            match i {
                BITS_7 | BITS_10 | BITS_11 | BITS_13 | BITS_14 => *tag += F::ONE,
                _ => (),
            }
            *dense += F::ONE;
            if i & 1 == 0 {
                // On even-numbered rows we recompute the spread.
                *spread = F::ZERO;
                for b in 0..16 {
                    if (i >> b) & 1 != 0 {
                        *spread += F::from(1 << (2 * b));
                    }
                }
            } else {
                // On odd-numbered rows we add one.
                *spread += F::ONE;
            }

            Some(res)
        })
    }
}

