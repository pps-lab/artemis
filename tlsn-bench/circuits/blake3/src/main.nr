//! BLAKE3 commitment (a TLSNotary hash option) recomputed in the circuit, as
//! explicit Noir code (ProveKit has no BLAKE3 black box).

global N: u32 = 1024;
global CHUNKS: u32 = if N == 0 { 1 } else { (N + 1023) / 1024 };

fn main(committed_hash: pub [u8; 32], payload: [u8; N]) {
    assert(blake3_noir::blake3::<N, CHUNKS>(payload) == committed_hash);
}
