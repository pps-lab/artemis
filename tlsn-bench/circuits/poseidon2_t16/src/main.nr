//! zk-friendly recomputation: Poseidon2 sponge (TACEO width 16, rate 15) over
//! the payload packed 31 bytes per field element. The bytes are packed in the
//! circuit, so this row starts from the same range-checked bytes as SHA-256.

use poseidon2::bn254::permutation::t16;

global N: u32 = 1024;
global PACK: u32 = 31;
global CHUNKS: u32 = (N + PACK - 1) / PACK;
global RATE: u32 = 15;

fn pack(payload: [u8; N]) -> [Field; CHUNKS] {
    let mut out: [Field; CHUNKS] = [0; CHUNKS];
    for i in 0..CHUNKS {
        let mut acc: Field = 0;
        let mut weight: Field = 1;
        for j in 0..PACK {
            let idx = i * PACK + j;
            if idx < N {
                acc += weight * (payload[idx] as Field);
                weight *= 256;
            }
        }
        out[i] = acc;
    }
    out
}

fn main(payload: [u8; N]) -> pub Field {
    let packed = pack(payload);
    let mut state: [Field; 16] = [0; 16];
    // Fixed-length domain separation in the capacity element.
    state[RATE] = N as Field;
    for i in 0..CHUNKS {
        let slot = i % RATE;
        state[slot] += packed[i];
        if slot == RATE - 1 {
            state = t16(state);
        }
    }
    if CHUNKS % RATE != 0 {
        state = t16(state);
    }
    state[0]
}
