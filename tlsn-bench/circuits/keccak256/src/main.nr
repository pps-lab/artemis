//! Keccak-256 commitment (a TLSNotary hash option) recomputed in the circuit,
//! as explicit Noir code (ProveKit has no Keccak black box).

global N: u32 = 1024;

fn main(committed_hash: pub [u8; 32], payload: [u8; N]) {
    assert(keccak_noir::keccak256(payload, N) == committed_hash);
}
