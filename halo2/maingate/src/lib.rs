//! `maingate` defines basic instructions for a starndart like PLONK gate and
//! implments a 5 width gate with two multiplication and one rotation
//! customisation

#![deny(missing_debug_implementations)]
#![deny(missing_docs)]

use halo2_proofs::circuit::AssignedCell;

#[macro_use]
mod instructions;
mod main_gate;
mod range;

pub use halo2_proofs::utils::*;
pub use instructions::{CombinationOptionCommon, MainGateInstructions, Term};
pub use main_gate::*;
pub use range::*;

#[cfg(test)]
use halo2_proofs::halo2curves;
#[cfg(test)]
pub use halo2_proofs::utils::mock_prover_verify;

/// AssignedValue
pub type AssignedValue<F> = AssignedCell<F, F>;
/// AssignedCondition
pub type AssignedCondition<F> = AssignedCell<F, F>;
