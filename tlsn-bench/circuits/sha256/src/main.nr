//! What TLSNotary ships: recompute the SHA-256 commitment in the circuit.

global N: u32 = 1024;

fn main(committed_hash: pub [u8; 32], payload: [u8; N]) {
    let computed = sha256::sha256_var(payload, N);
    assert(computed == committed_hash);
}
