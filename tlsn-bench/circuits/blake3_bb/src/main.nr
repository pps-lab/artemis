//! BLAKE3 commitment recomputed in the circuit through the BLAKE3 black box
//! (Barretenberg).

global N: u32 = 1024;

fn main(committed_hash: pub [u8; 32], payload: [u8; N]) {
    assert(std::hash::blake3(payload) == committed_hash);
}
