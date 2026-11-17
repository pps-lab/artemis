// BLAKE3 (hash mode, 32-byte output) as explicit Noir code over u32 words.
//
// ProveKit does not lower Noir's BLAKE3 black box to R1CS, so the ProveKit
// rows need the compression function spelled out. The input length N and the
// chunk count C = max(1, ceil(N / 1024)) are compile-time, so the chunk and
// tree structure is static.

global IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
global MSG_PERMUTATION: [u32; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

global CHUNK_START: u32 = 1;
global CHUNK_END: u32 = 2;
global PARENT: u32 = 4;
global ROOT: u32 = 8;

fn rotr(x: u32, n: u32) -> u32 {
    // The two shifted parts occupy disjoint bit ranges.
    (x >> n) + (x << (32 - n))
}

fn add3(a: u32, b: u32, c: u32) -> u32 {
    ((a as u64) + (b as u64) + (c as u64)) as u32
}

fn add2(a: u32, b: u32) -> u32 {
    ((a as u64) + (b as u64)) as u32
}

fn g(s: [u32; 16], a: u32, b: u32, c: u32, d: u32, mx: u32, my: u32) -> [u32; 16] {
    let mut s = s;
    s[a] = add3(s[a], s[b], mx);
    s[d] = rotr(s[d] ^ s[a], 16);
    s[c] = add2(s[c], s[d]);
    s[b] = rotr(s[b] ^ s[c], 12);
    s[a] = add3(s[a], s[b], my);
    s[d] = rotr(s[d] ^ s[a], 8);
    s[c] = add2(s[c], s[d]);
    s[b] = rotr(s[b] ^ s[c], 7);
    s
}

/// Compression function, truncated to the 8-word chaining value. The counter
/// is below 2^32 for every input this circuit can take.
fn compress(cv: [u32; 8], block: [u32; 16], counter: u32, block_len: u32, flags: u32) -> [u32; 8] {
    let mut s: [u32; 16] = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7], IV[0], IV[1], IV[2], IV[3], counter,
        0, block_len, flags,
    ];
    let mut m = block;
    for round in 0..7 {
        s = g(s, 0, 4, 8, 12, m[0], m[1]);
        s = g(s, 1, 5, 9, 13, m[2], m[3]);
        s = g(s, 2, 6, 10, 14, m[4], m[5]);
        s = g(s, 3, 7, 11, 15, m[6], m[7]);
        s = g(s, 0, 5, 10, 15, m[8], m[9]);
        s = g(s, 1, 6, 11, 12, m[10], m[11]);
        s = g(s, 2, 7, 8, 13, m[12], m[13]);
        s = g(s, 3, 4, 9, 14, m[14], m[15]);
        if round < 6 {
            let mut p: [u32; 16] = [0; 16];
            for i in 0..16 {
                p[i] = m[MSG_PERMUTATION[i]];
            }
            m = p;
        }
    }
    let mut out: [u32; 8] = [0; 8];
    for i in 0..8 {
        out[i] = s[i] ^ s[i + 8];
    }
    out
}

/// BLAKE3 of `msg`. `C` must be max(1, ceil(N / 1024)).
pub fn blake3<let N: u32, let C: u32>(msg: [u8; N]) -> [u8; 32] {
    let chunks = if N == 0 { 1 } else { (N + 1023) / 1024 };
    assert(C == chunks);

    let mut cvs: [[u32; 8]; C] = [[0; 8]; C];
    for c in 0..C {
        let start = c * 1024;
        let len = if N - start < 1024 { N - start } else { 1024 };
        let blocks = if len == 0 { 1 } else { (len + 63) / 64 };
        let mut cv = IV;
        for b in 0..16 {
            if b < blocks {
                let base = start + b * 64;
                let mut words: [u32; 16] = [0; 16];
                for w in 0..16 {
                    let mut word: u32 = 0;
                    let mut scale: u32 = 1;
                    for j in 0..4 {
                        let idx = base + w * 4 + j;
                        if idx < N {
                            word += (msg[idx] as u32) * scale;
                        }
                        if j < 3 {
                            scale *= 256;
                        }
                    }
                    words[w] = word;
                }
                let block_len = if len - b * 64 < 64 { len - b * 64 } else { 64 };
                let mut flags = 0;
                if b == 0 {
                    flags += CHUNK_START;
                }
                if b == blocks - 1 {
                    flags += CHUNK_END;
                    if C == 1 {
                        flags += ROOT;
                    }
                }
                cv = compress(cv, words, c, block_len, flags);
            }
        }
        cvs[c] = cv;
    }

    // Merge pairwise, carrying an odd last node up; this is the BLAKE3 tree.
    let mut len = C;
    for _level in 0..32 {
        if len > 1 {
            let half = len / 2;
            let flags = if len == 2 { PARENT + ROOT } else { PARENT };
            for i in 0..C / 2 {
                if i < half {
                    let l = cvs[2 * i];
                    let r = cvs[2 * i + 1];
                    let block: [u32; 16] = [
                        l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], r[0], r[1], r[2], r[3], r[4],
                        r[5], r[6], r[7],
                    ];
                    cvs[i] = compress(IV, block, 0, 64, flags);
                }
            }
            if len % 2 == 1 {
                cvs[half] = cvs[len - 1];
            }
            len = half + len % 2;
        }
    }

    let mut out: [u8; 32] = [0; 32];
    for i in 0..8 {
        let bytes: [u8; 4] = (cvs[0][i] as Field).to_le_bytes();
        for j in 0..4 {
            out[4 * i + j] = bytes[j];
        }
    }
    out
}
