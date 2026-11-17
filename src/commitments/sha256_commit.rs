//! SHA-256 commitment to the model weights (baseline next to Poseidon).
//!
//! The committed values are the packer outputs (`PackerChip` with one element per packed value), i.e.
//! each weight shifted to `w + 2^(b-1) ∈ [0, 2^b)` and range-checked, serialised as one big-endian 32-bit
//! word. The commitment is `SHA-256(words || r)` with standard SHA-256 padding (0x80 bit, zeros,
//! 64-bit message length), computed with the halo2_gadgets Table16 chip (made field-generic in
//! `commitments::sha256`). The packed cells are copy-constrained to the message-schedule words
//! W_0..W_15 of their block, and the padding words to constants.
//!
//! One Table16 instance needs ~2,100 rows per 64-byte block. To keep the circuit at the size the
//! model needs (2^k rows), `num_chips` instances on disjoint advice columns are configured and block
//! i is compressed by instance i mod num_chips; the chaining state is copied between instances, so the
//! result is the ordinary sequential SHA-256 hash. The digest (eight 32-bit words) is returned as
//! eight cells that become public values.
//!
//! As in the upstream gadget, the IV is assigned as a witness (not pinned to constants); this does not
//! change the cost.
use std::{collections::HashMap, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Value},
  halo2curves::ff::PrimeField,
  plonk::{Advice, Column, ConstraintSystem, Error, Expression, Fixed, Selector, TableColumn},
  poly::Rotation,
};

use super::{
  commit::Commit,
  sha256::{BlockWord, Table16Chip, Table16Config, BLOCK_SIZE, IV, STATE},
};
use crate::{gadgets::gadget::GadgetConfig, layers::layer::CellRc};

/// Rows of one Table16 block: message schedule (547) + initialisation + 64 rounds (1536) + slack.
/// (The feed-forward rows live in the shared `ff_cols`.)
pub const SHA256_ROWS_PER_BLOCK: usize = 2112;

/// Number of 64-byte blocks needed to hash `num_words` 32-bit words with SHA-256 padding.
pub fn sha256_num_blocks(num_words: usize) -> usize {
  // message words, then 0x80000000, zeros, and two length words
  (num_words + 1 + 2).div_ceil(BLOCK_SIZE)
}

#[derive(Clone, Debug)]
pub struct Sha256CommitChip {
  pub configs: Vec<Table16Config>,
  pub constant_col: Column<Fixed>,
  pub link_col: Column<Advice>,
  /// feed-forward: o = h + s mod 2^32; columns h, s_lo, s_hi, o, o_lo, o_hi, carry
  pub ff_cols: [Column<Advice>; 7],
  pub q_ff: Selector,
  pub range16: TableColumn,
}

impl Sha256CommitChip {
  pub fn configure<F: PrimeField>(meta: &mut ConstraintSystem<F>, num_chips: usize) -> Self {
    assert!(num_chips >= 1);
    let configs = (0..num_chips).map(|_| Table16Chip::<F>::configure(meta)).collect();
    let constant_col = meta.fixed_column();
    meta.enable_constant(constant_col);
    let link_col = meta.advice_column();
    meta.enable_equality(link_col);

    let ff_cols: [Column<Advice>; 7] = std::array::from_fn(|_| meta.advice_column());
    for c in ff_cols.iter() {
      meta.enable_equality(*c);
    }
    let q_ff = meta.complex_selector();
    let range16 = meta.lookup_table_column();
    let [h, s_lo, s_hi, o, o_lo, o_hi, carry] = ff_cols;
    meta.create_gate("sha256 feed-forward", |meta| {
      let q = meta.query_selector(q_ff);
      let h = meta.query_advice(h, Rotation::cur());
      let s_lo = meta.query_advice(s_lo, Rotation::cur());
      let s_hi = meta.query_advice(s_hi, Rotation::cur());
      let o = meta.query_advice(o, Rotation::cur());
      let o_lo = meta.query_advice(o_lo, Rotation::cur());
      let o_hi = meta.query_advice(o_hi, Rotation::cur());
      let carry = meta.query_advice(carry, Rotation::cur());
      let c16 = Expression::Constant(F::from(1u64 << 16));
      let c32 = Expression::Constant(F::from(1u64 << 32));
      vec![
        q.clone() * (o.clone() - o_lo - c16.clone() * o_hi),
        q.clone() * (o + carry.clone() * c32 - h - s_lo - c16 * s_hi),
        q * carry.clone() * (Expression::Constant(F::ONE) - carry),
      ]
    });
    for col in [o_lo, o_hi] {
      meta.lookup("sha256 feed-forward range", |meta| {
        let q = meta.query_selector(q_ff);
        let v = meta.query_advice(col, Rotation::cur());
        vec![(q * v, range16)]
      });
    }
    Sha256CommitChip { configs, constant_col, link_col, ff_cols, q_ff, range16 }
  }

  /// One feed-forward row per state word: o_j = h_j + s_j mod 2^32. `h` holds either the previous
  /// block's output cells or None for the constant IV. Returns the (o, o_lo, o_hi) cells.
  fn feed_forward<F: PrimeField>(
    &self,
    layouter: &mut impl Layouter<F>,
    h: &Option<Vec<AssignedCell<F, F>>>,
    s: &[(halo2_proofs::circuit::Cell, halo2_proofs::circuit::Cell, Value<u32>)],
  ) -> Result<Vec<[AssignedCell<F, F>; 3]>, Error> {
    let [h_col, s_lo_col, s_hi_col, o_col, o_lo_col, o_hi_col, carry_col] = self.ff_cols;
    layouter.assign_region(
      || "sha256 feed-forward",
      |mut region| {
        let mut out = vec![];
        for j in 0..STATE {
          self.q_ff.enable(&mut region, j)?;
          let h_cell = match h {
            Some(prev) => prev[j].copy_advice(|| "h", &mut region, h_col, j)?,
            None => region.assign_advice_from_constant(|| "iv", h_col, j, F::from(IV[j] as u64))?,
          };
          let h_val = h_cell.value().map(|v| Self::word_value(v));
          let (s_lo_cell, s_hi_cell, s_val) = &s[j];
          let s_lo = region.assign_advice(|| "s_lo", s_lo_col, j, || s_val.map(|v| F::from((v & 0xffff) as u64)))?;
          let s_hi = region.assign_advice(|| "s_hi", s_hi_col, j, || s_val.map(|v| F::from((v >> 16) as u64)))?;
          region.constrain_equal(s_lo.cell(), *s_lo_cell)?;
          region.constrain_equal(s_hi.cell(), *s_hi_cell)?;
          let sum = h_val.zip(*s_val).map(|(a, b)| a as u64 + b as u64);
          let o_val = sum.map(|x| x & 0xffff_ffff);
          let o = region.assign_advice(|| "o", o_col, j, || o_val.map(F::from))?;
          let o_lo = region.assign_advice(|| "o_lo", o_lo_col, j, || o_val.map(|v| F::from(v & 0xffff)))?;
          let o_hi = region.assign_advice(|| "o_hi", o_hi_col, j, || o_val.map(|v| F::from(v >> 16)))?;
          region.assign_advice(|| "carry", carry_col, j, || sum.map(|x| F::from(x >> 32)))?;
          out.push([o, o_lo, o_hi]);
        }
        Ok(out)
      },
    )
  }

  /// Loads the spread lookup tables (one per instance); call once per synthesis.
  pub fn load_tables<F: PrimeField>(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
    for c in self.configs.iter() {
      Table16Chip::<F>::load(c.clone(), layouter)?;
    }
    layouter.assign_table(
      || "range16",
      |mut table| {
        for i in 0..(1usize << 16) {
          table.assign_cell(|| "r", self.range16, i, || Value::known(F::from(i as u64)))?;
        }
        Ok(())
      },
    )
  }

  fn word_value<F: PrimeField>(v: &F) -> u32 {
    let repr = v.to_repr();
    let bytes = repr.as_ref();
    // little-endian representation; the value must fit in 32 bits
    debug_assert!(bytes[4..].iter().all(|b| *b == 0), "SHA-256 input word does not fit in 32 bits");
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
  }
}

impl<F: PrimeField> Commit<F> for Sha256CommitChip {
  fn commit(
    &self,
    mut layouter: impl Layouter<F>,
    _gadget_config: Rc<GadgetConfig>,
    _constants: &HashMap<i64, CellRc<F>>,
    values: &Vec<CellRc<F>>,
    blinding: CellRc<F>,
  ) -> Result<Vec<CellRc<F>>, Error> {
    let chips: Vec<Table16Chip<F>> = self.configs.iter().map(|c| Table16Chip::construct(c.clone())).collect();

    // Message: the packed values and the blinding word, then the SHA-256 padding.
    let msg: Vec<CellRc<F>> = values.iter().cloned().chain(std::iter::once(blinding)).collect();
    let msg_bits = (msg.len() as u64) * 32;
    let num_blocks = sha256_num_blocks(msg.len());
    let total_words = num_blocks * BLOCK_SIZE;
    let mut padding = vec![0u32; total_words - msg.len()];
    padding[0] = 0x8000_0000;
    let n = padding.len();
    padding[n - 2] = (msg_bits >> 32) as u32;
    padding[n - 1] = msg_bits as u32;

    let words: Vec<BlockWord> = msg
      .iter()
      .map(|c| BlockWord(c.value().map(|v| Self::word_value(v))))
      .chain(padding.iter().map(|p| BlockWord(Value::known(*p))))
      .collect();

    // Chaining value H (None = the constant IV) and its 16-bit halves for the next initialisation.
    let mut h: Option<Vec<AssignedCell<F, F>>> = None;
    let mut h_halves: Vec<(AssignedCell<F, F>, AssignedCell<F, F>)> = vec![];
    let mut h_vals: [Value<u32>; STATE] = IV.map(Value::known);
    let mut input_cells = Vec::with_capacity(total_words);
    let mut iv_links = vec![];
    for (b, block) in words.chunks(BLOCK_SIZE).enumerate() {
      let chip = &chips[b % chips.len()];
      let init = chip.initialization_from_values(&mut layouter, h_vals)?;
      let init_cells = init.dense_word_cells();
      if b == 0 {
        iv_links = init_cells.clone();
      } else {
        layouter.assign_region(
          || "sha256 chain",
          |mut region| {
            for ((lo, hi, _), (o_lo, o_hi)) in init_cells.iter().zip(h_halves.iter()) {
              region.constrain_equal(*lo, o_lo.cell())?;
              region.constrain_equal(*hi, o_hi.cell())?;
            }
            Ok(())
          },
        )?;
      }
      let (out, inputs) = chip.compress_with_inputs(&mut layouter, &init, block.try_into().unwrap())?;
      input_cells.extend(inputs);
      let ff = self.feed_forward(&mut layouter, &h, &out.dense_word_cells())?;
      h_vals = std::array::from_fn(|j| ff[j][0].value().map(|v| Self::word_value(v)));
      h_halves = ff.iter().map(|[_, lo, hi]| (lo.clone(), hi.clone())).collect();
      h = Some(ff.into_iter().map(|[o, _, _]| o).collect());
    }
    let digest = h.unwrap();

    // Link the message words to the packed cells and the padding words to constants.
    layouter.assign_region(
      || "sha256 link",
      |mut region| {
        for (cell, w) in msg.iter().zip(input_cells.iter()) {
          region.constrain_equal(cell.cell(), w.cell())?;
        }
        let mut row = 0;
        for (p, w) in padding.iter().zip(input_cells[msg.len()..].iter()) {
          let c = region.assign_advice_from_constant(|| "pad", self.link_col, row, F::from(*p as u64))?;
          region.constrain_equal(c.cell(), w.cell())?;
          row += 1;
        }
        // the first block starts from the constant IV
        for ((lo, hi, _), iv) in iv_links.iter().zip(IV.iter()) {
          let c = region.assign_advice_from_constant(|| "iv_lo", self.link_col, row, F::from((iv & 0xffff) as u64))?;
          region.constrain_equal(c.cell(), *lo)?;
          let c = region.assign_advice_from_constant(|| "iv_hi", self.link_col, row + 1, F::from((iv >> 16) as u64))?;
          region.constrain_equal(c.cell(), *hi)?;
          row += 2;
        }
        Ok(())
      },
    )?;

    Ok(digest.into_iter().map(Rc::new).collect())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use halo2_proofs::{
    circuit::SimpleFloorPlanner,
    dev::MockProver,
    halo2curves::bn256::Fr,
    plonk::Circuit,
  };

  #[derive(Clone)]
  struct TestCircuit {
    words: Vec<u32>,
  }

  #[derive(Clone)]
  struct TestConfig {
    chip: Sha256CommitChip,
    col: Column<Advice>,
  }

  impl Circuit<Fr> for TestCircuit {
    type Config = TestConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
      self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> TestConfig {
      let col = meta.advice_column();
      meta.enable_equality(col);
      TestConfig { chip: Sha256CommitChip::configure(meta, NUM_CHIPS.with(|n| *n.borrow())), col }
    }

    fn synthesize(&self, config: TestConfig, mut layouter: impl Layouter<Fr>) -> Result<(), Error> {
      config.chip.load_tables(&mut layouter)?;
      let cells = layouter.assign_region(
        || "inputs",
        |mut region| {
          let mut out = vec![];
          for (i, w) in self.words.iter().enumerate() {
            out.push(Rc::new(region.assign_advice(|| "w", config.col, i, || Value::known(Fr::from(*w as u64)))?));
          }
          out.push(Rc::new(region.assign_advice(|| "r", config.col, self.words.len(), || Value::known(Fr::from(0u64)))?));
          Ok(out)
        },
      )?;
      let (vals, blinding) = (cells[..cells.len() - 1].to_vec(), cells[cells.len() - 1].clone());
      let digest = config.chip.commit(
        layouter.namespace(|| "sha"),
        Rc::new(GadgetConfig::default()),
        &HashMap::new(),
        &vals,
        blinding,
      )?;
      let mut bytes = vec![];
      for d in digest.iter() {
        d.value().map(|v| bytes.extend_from_slice(&Sha256CommitChip::word_value(v).to_be_bytes()));
      }
      let expected = {
        use sha2::Digest;
        let mut msg = vec![];
        for w in self.words.iter().chain(std::iter::once(&0u32)) {
          msg.extend_from_slice(&w.to_be_bytes());
        }
        sha2::Sha256::digest(&msg).to_vec()
      };
      assert_eq!(bytes, expected, "in-circuit digest != sha2 digest");
      Ok(())
    }
  }

  thread_local! { static NUM_CHIPS: std::cell::RefCell<usize> = std::cell::RefCell::new(1); }

  fn run(num_words: usize, num_chips: usize) {
    NUM_CHIPS.with(|n| *n.borrow_mut() = num_chips);
    let words: Vec<u32> = (0..num_words as u32).map(|i| i.wrapping_mul(2654435761) >> 13).collect();
    let circuit = TestCircuit { words };
    let prover = MockProver::<Fr>::run(17, &circuit, vec![]).unwrap();
    prover.assert_satisfied();
  }

  #[test]
  fn sha256_commit_matches_sha2_one_chip() {
    run(5, 1); // single block
    run(40, 1); // three blocks
  }

  #[test]
  fn sha256_commit_matches_sha2_three_chips() {
    run(100, 3); // seven blocks round-robin over three instances
  }
}

