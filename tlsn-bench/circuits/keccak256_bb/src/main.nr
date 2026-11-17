//! Keccak-256 commitment recomputed in the circuit through the keccakf1600
//! black box (Barretenberg).

global N: u32 = 1024;

fn main(committed_hash: pub [u8; 32], payload: [u8; N]) {
    assert(keccak256::keccak256(payload, N) == committed_hash);
}
