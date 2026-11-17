//! Variant of the LegoSNARK cost estimate that packs inside the circuit: the
//! payload bytes are packed 31 per field element and every packed element is
//! a witness of its own (`packed`), so that CP_link can link the cells that
//! hold the packed elements instead of the byte cells. Like `nocom`, the
//! circuit binds nothing by itself; the link is made outside on the wire
//! commitments (see README.md).

global N: u32 = 1024;
global PACK: u32 = 31;
global CHUNKS: u32 = (N + PACK - 1) / PACK;

fn main(payload: [u8; N], packed: [Field; CHUNKS]) {
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
        assert(acc == packed[i]);
    }
}
