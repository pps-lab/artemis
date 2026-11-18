//! Artemis: the circuit holds only the predicate. The ProveKit Artemis hook
//! (`artemis-bench prepare --artemis`) links `payload` and `mask` to an
//! external Pedersen commitment by adding the Horner constraints at R1CS
//! level; the Noir source does not compute a commitment.

global N: u32 = 1024;

fn main(payload: [u8; N], mask: Field) {
    // Empty predicate, as in `nocom`.
    let _ = payload;
    let _ = mask;
}
