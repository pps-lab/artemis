//! zk-friendly recomputation with the backend's native Poseidon2 permutation
//! (width 4, rate 3): Barretenberg has custom gates for it, ProveKit lowers
//! the black box directly to R1CS. Same in-circuit packing as `poseidon2_t16`.

global N: u32 = 1024;
global PACK: u32 = 31;
global CHUNKS: u32 = (N + PACK - 1) / PACK;
global RATE: u32 = 3;

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
    let mut state: [Field; 4] = [0; 4];
    state[RATE] = N as Field;
    for i in 0..CHUNKS {
        let slot = i % RATE;
        state[slot] += packed[i];
        if slot == RATE - 1 {
            state = std::hash::poseidon2_permutation(state);
        }
    }
    if CHUNKS % RATE != 0 {
        state = std::hash::poseidon2_permutation(state);
    }
    state[0]
}
