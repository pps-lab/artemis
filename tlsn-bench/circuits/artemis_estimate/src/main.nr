//! COST ESTIMATE ONLY, NOT SOUND. The Horner link written in Noir with the
//! evaluation point as a public input. A prover who chooses `eval_point`
//! after seeing it can cheat; a sound link needs the point to be a verifier
//! challenge drawn after the witness is committed, which Barretenberg does
//! not expose. Used to estimate what the link would cost on UltraHonk.

global N: u32 = 1024;
global PACK: u32 = 31;
global CHUNKS: u32 = (N + PACK - 1) / PACK;

fn main(eval_point: pub Field, claimed_eval: pub Field, payload: [u8; N], mask: Field) {
    let mut acc: Field = mask;
    for c in 0..CHUNKS {
        let i = CHUNKS - 1 - c;
        let mut packed: Field = 0;
        let mut weight: Field = 1;
        for j in 0..PACK {
            let idx = i * PACK + j;
            if idx < N {
                packed += weight * (payload[idx] as Field);
                weight *= 256;
            }
        }
        acc = acc * eval_point + packed;
    }
    assert(acc * eval_point == claimed_eval);
}
