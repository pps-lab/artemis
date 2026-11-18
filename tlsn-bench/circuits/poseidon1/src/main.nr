//! zk-friendly recomputation with Poseidon (v1) in the configuration of the zkML Poseidon baseline
//! (src/commitments/poseidon_commit.rs: halo2 Pow5Chip, P128Pow5T3): width 3, rate 2, x^5 S-box,
//! 8 full rounds, sponge over the committed values followed by a field blinder. The permutation is
//! noir-lang/poseidon's BN254 `x5_3` (circomlib constants, 57 partial rounds; halo2's P128Pow5T3 has
//! 56). The payload is packed 31 bytes per field element in the circuit, as in the Artemis rows.

use poseidon::poseidon::bn254::perm::x5_3;

global N: u32 = 1024;
global PACK: u32 = 31;
global CHUNKS: u32 = (N + PACK - 1) / PACK;
global RATE: u32 = 2;

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

fn main(payload: [u8; N], blinder: Field) -> pub Field {
    let packed = pack(payload);
    // Rate elements 0..1, capacity element 2 holds the fixed message length (as halo2's ConstantLength).
    let mut state: [Field; 3] = [0; 3];
    state[RATE] = (CHUNKS + 1) as Field;
    for i in 0..CHUNKS + 1 {
        let slot = i % RATE;
        state[slot] += if i < CHUNKS { packed[i] } else { blinder };
        if slot == RATE - 1 {
            state = x5_3(state);
        }
    }
    if (CHUNKS + 1) % RATE != 0 {
        state = x5_3(state);
    }
    state[0]
}
