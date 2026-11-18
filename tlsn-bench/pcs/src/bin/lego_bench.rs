//! LegoSNARK CP_link on the wire columns of a real Barretenberg proof.
//!
//! ```text
//! lego_bench --dump <dir> --proof <file> --crs <bn254_g1.dat> --payload <file>
//!            --mode <bytes|packed|vars> [--first-var <v> --count <k>] [--runs 5]
//! ```
//!
//! `<dir>` holds `wires.bin` and `cells.bin`, written by the patched `bb`
//! (`BB_DUMP_WIRES`, see `barretenberg-dump.patch`) while it produced
//! `<file>`. The tool
//!
//! 1. finds the cells that hold the committed variables and the smallest set
//!    of wires that covers them (the linked wires),
//! 2. recomputes all four wire commitments from the dumped values with
//!    Barretenberg's reference string and looks them up in the proof,
//! 3. runs the CP_link setup, the Pedersen commitment, the prover and the
//!    verifier on those values and on the commitments read from the proof.
//!
//! Modes: `bytes` commits to the payload bytes one by one, `packed` to the
//! payload packed 31 bytes per element (the packing is part of the linear map,
//! the linked cells are still the byte cells), `vars` to the builder variables
//! `first-var .. first-var + count` (a circuit that packs in the circuit).
//! Prints one JSON object.

use {
    ark_bn254::{Fq, Fr, G1Affine},
    ark_ec::{AffineRepr, CurveGroup},
    ark_ff::{BigInteger, BigInteger256, Field, PrimeField, UniformRand, Zero},
    rayon::prelude::*,
    std::{collections::HashMap, fs, time::Instant},
    tlsn_bench_pcs::lego::{commit_wire, Crs, Link},
};

const WIRES: [&str; 4] = ["w_l", "w_r", "w_o", "w_4"];
const PACK: usize = 31;
const UNUSED: u32 = u32::MAX;

struct Dump {
    dyadic: usize,
    active: usize,
    masked: usize,
    /// Commitments computed by `bb` (x, y).
    coms:   Vec<(BigInteger256, BigInteger256)>,
    /// Per wire: the active rows followed by the masking rows.
    wires:  Vec<Vec<BigInteger256>>,
    /// Row-major real variable index of every cell of the active rows.
    cells:  Vec<u32>,
    /// Real variable index of every builder variable.
    real:   Vec<u32>,
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn big_at(b: &[u8], off: usize) -> BigInteger256 {
    BigInteger256::new([
        u64_at(b, off),
        u64_at(b, off + 8),
        u64_at(b, off + 16),
        u64_at(b, off + 24),
    ])
}

fn load_dump(dir: &str) -> Dump {
    let b = fs::read(format!("{dir}/wires.bin")).expect("wires.bin");
    assert_eq!(&b[..8], b"BBWIRES1");
    let (dyadic, active, masked) = (
        u64_at(&b, 8) as usize,
        u64_at(&b, 16) as usize,
        u64_at(&b, 24) as usize,
    );
    let mut off = 32;
    let mut coms = vec![];
    let mut tails = vec![];
    for _ in 0..4 {
        coms.push((big_at(&b, off), big_at(&b, off + 32)));
        off += 64;
        tails.push((0..masked).map(|j| big_at(&b, off + 32 * j)).collect::<Vec<_>>());
        off += 32 * masked;
    }
    let mut wires = vec![];
    for tail in tails {
        let mut w: Vec<BigInteger256> = (0..active)
            .into_par_iter()
            .map(|i| big_at(&b, off + 32 * i))
            .collect();
        w.extend(tail);
        wires.push(w);
        off += 32 * active;
    }
    assert_eq!(off, b.len());

    let c = fs::read(format!("{dir}/cells.bin")).expect("cells.bin");
    assert_eq!(&c[..8], b"BBCELLS1");
    let (rows, vars) = (u64_at(&c, 8) as usize, u64_at(&c, 16) as usize);
    assert_eq!(rows, active);
    assert_eq!(c.len(), 24 + 4 * (4 * rows + vars));
    let u32s = |off: usize, n: usize| -> Vec<u32> {
        c[off..off + 4 * n]
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
            .collect()
    };
    Dump {
        dyadic,
        active,
        masked,
        coms,
        wires,
        cells: u32s(24, 4 * rows),
        real: u32s(24 + 16 * rows, vars),
    }
}

/// Points `first .. first + n` of Barretenberg's `bn254_g1.dat` (`[tau^i]_1`,
/// affine, 32-byte big-endian coordinates).
fn load_crs(path: &str, ranges: &[(usize, usize)]) -> Vec<G1Affine> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = fs::File::open(path).expect("crs");
    let mut out = vec![];
    for &(first, n) in ranges {
        let mut buf = vec![0u8; 64 * n];
        f.seek(SeekFrom::Start(64 * first as u64)).unwrap();
        f.read_exact(&mut buf).expect("crs too short");
        let pts: Vec<G1Affine> = buf
            .par_chunks_exact(64)
            .map(|c| {
                let x = Fq::from_be_bytes_mod_order(&c[..32]);
                let y = Fq::from_be_bytes_mod_order(&c[32..]);
                G1Affine::new_unchecked(x, y)
            })
            .collect();
        out.extend(pts);
    }
    assert!(out[0].is_on_curve() && out[out.len() - 1].is_on_curve());
    out
}

/// A G1 point as Barretenberg writes it into an UltraHonk proof: four field
/// elements `x_lo, x_hi, y_lo, y_hi` with 136-bit low limbs, big endian.
fn proof_encoding(p: &G1Affine) -> Vec<u8> {
    let mut out = vec![];
    for coord in [p.x, p.y] {
        let le = coord.into_bigint().to_bytes_le();
        let (mut lo, mut hi) = ([0u8; 32], [0u8; 32]);
        lo[..17].copy_from_slice(&le[..17]);
        hi[..15].copy_from_slice(&le[17..]);
        lo.reverse();
        hi.reverse();
        out.extend(lo);
        out.extend(hi);
    }
    out
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn list(xs: &[f64]) -> String {
    format!("[{}]", xs.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(","))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let need = |n: &str| arg(&args, n).unwrap_or_else(|| panic!("missing {n}"));
    let dump = load_dump(&need("--dump"));
    let proof = fs::read(need("--proof")).expect("proof");
    let payload = fs::read(need("--payload")).expect("payload");
    let mode = need("--mode");
    let runs: usize = arg(&args, "--runs").map_or(5, |r| r.parse().unwrap());
    let n = payload.len();
    let slots = dump.active + dump.masked;

    // Committed variables: (builder variable, Pedersen index, coefficient).
    let committed: Vec<(usize, usize, Fr)> = match mode.as_str() {
        "bytes" => (0..n).map(|j| (j, j, Fr::from(1u64))).collect(),
        "packed" => (0..n)
            .map(|j| (j, j / PACK, Fr::from(256u64).pow([(j % PACK) as u64])))
            .collect(),
        "vars" => {
            let first: usize = need("--first-var").parse().unwrap();
            let count: usize = need("--count").parse().unwrap();
            (0..count).map(|i| (first + i, i, Fr::from(1u64))).collect()
        }
        other => panic!("unknown mode {other}"),
    };
    let elements = committed.iter().map(|c| c.1).max().unwrap() + 1;

    // Where the committed variables sit: per variable the wires that hold it
    // and its first row in each of them.
    let mut index: HashMap<u32, usize> = HashMap::new();
    for (i, (var, ..)) in committed.iter().enumerate() {
        let prev = index.insert(dump.real[*var], i);
        assert!(prev.is_none(), "two committed variables in one copy class");
    }
    let mut first_row = vec![[usize::MAX; 4]; committed.len()];
    let mut occurrences = [0usize; 4];
    for (cell, var) in dump.cells.iter().enumerate() {
        if *var == UNUSED {
            continue;
        }
        if let Some(&i) = index.get(var) {
            let (row, wire) = (cell / 4, cell % 4);
            occurrences[wire] += 1;
            if first_row[i][wire] == usize::MAX {
                first_row[i][wire] = row;
            }
        }
    }
    let masks: Vec<u8> = first_row
        .iter()
        .map(|r| (0..4).fold(0u8, |m, w| m | (((r[w] != usize::MAX) as u8) << w)))
        .collect();
    assert!(masks.iter().all(|m| *m != 0), "a committed variable is in no wire cell");

    // Cell statistics per wire (masking rows included).
    let class = |s: &BigInteger256| -> usize {
        match s.num_bits() {
            0 => 0,
            1..=8 => 1,
            9..=16 => 2,
            17..=32 => 3,
            33..=64 => 4,
            _ => 5,
        }
    };
    let stats: Vec<[usize; 6]> = dump
        .wires
        .iter()
        .map(|w| {
            let mut s = [0usize; 6];
            w.iter().for_each(|x| s[class(x)] += 1);
            s
        })
        .collect();
    let nonzero = |w: usize| slots - stats[w][0];

    // Smallest set of wires that holds every committed variable; ties go to
    // the set with the fewest non-zero cells.
    let subset = (1u8..16)
        .filter(|s| masks.iter().all(|m| m & s != 0))
        .min_by_key(|s| {
            let cells: usize = (0..4).filter(|w| s >> w & 1 == 1).map(nonzero).sum();
            (s.count_ones(), cells)
        })
        .unwrap();
    let linked: Vec<usize> = (0..4).filter(|w| subset >> w & 1 == 1).collect();
    let ell = linked.len();

    // One canonical cell per committed variable, in the first linked wire
    // that holds it.
    let mut links = Vec::with_capacity(committed.len());
    let mut values = vec![Fr::zero(); elements];
    for (i, (var, ped, coeff)) in committed.iter().enumerate() {
        let (slot_wire, wire) = linked
            .iter()
            .enumerate()
            .find(|(_, w)| first_row[i][**w] != usize::MAX)
            .map(|(s, w)| (s, *w))
            .unwrap();
        let row = first_row[i][wire];
        let value = Fr::from_bigint(dump.wires[wire][row]).unwrap();
        if mode != "vars" {
            assert_eq!(value, Fr::from(payload[*var] as u64), "cell of byte {var} holds another value");
        }
        values[*ped] += *coeff * value;
        links.push(Link { wire: slot_wire, slot: row, ped: *ped, coeff: *coeff });
    }
    if mode == "packed" {
        let direct = tlsn_bench_pcs::pack_bytes(&payload);
        assert_eq!(values, direct);
    }
    let values: Vec<BigInteger256> = values.iter().map(|v| v.into_bigint()).collect();

    // Realism check: the dumped values, committed with Barretenberg's points,
    // give the commitments that `bb` computed and wrote into the proof.
    let bases = load_crs(&need("--crs"), &[(0, dump.active), (dump.dyadic - dump.masked, dump.masked)]);
    let t = Instant::now();
    let coms: Vec<G1Affine> = dump
        .wires
        .iter()
        .map(|w| commit_wire(&bases, w).into_affine())
        .collect();
    let wire_commit_s = t.elapsed().as_secs_f64() / 4.0;
    let mut proof_offsets = vec![];
    for (w, com) in coms.iter().enumerate() {
        let (x, y) = com.xy().expect("wire commitment at infinity");
        assert_eq!((x.into_bigint(), y.into_bigint()), dump.coms[w], "{}: recomputed commitment differs from bb's", WIRES[w]);
        let off = find(&proof, &proof_encoding(com))
            .unwrap_or_else(|| panic!("{}: commitment not found in the proof", WIRES[w]));
        proof_offsets.push(off);
    }

    // CP_link.
    let mut rng = rand::thread_rng();
    let t = Instant::now();
    let crs = Crs::setup(&bases, ell, elements, &links, &mut rng);
    let setup_s = t.elapsed().as_secs_f64();
    let (key_p, key_h) = crs.key_bytes();
    let r = Fr::rand(&mut rng);
    let linked_wires: Vec<&[BigInteger256]> = linked.iter().map(|w| dump.wires[*w].as_slice()).collect();
    let linked_coms: Vec<G1Affine> = linked.iter().map(|w| coms[*w]).collect();

    let (mut commit_s, mut prove_s, mut verify_s) = (vec![], vec![], vec![]);
    let mut c = G1Affine::identity();
    let mut pi = G1Affine::identity();
    for _ in 0..runs {
        let t = Instant::now();
        c = crs.commit(r, &values).into_affine();
        commit_s.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        pi = crs.prove(r, &linked_wires).into_affine();
        prove_s.push(t.elapsed().as_secs_f64());
    }
    for _ in 0..5 {
        let t = Instant::now();
        let ok = crs.verify(&pi, &c, &linked_coms);
        verify_s.push(t.elapsed().as_secs_f64());
        assert!(ok, "CP_link proof rejected");
    }
    // A commitment to another payload is rejected.
    let other = (c.into_group() + crs.h[1]).into_affine();
    assert!(!crs.verify(&pi, &other, &linked_coms));

    let names = |ws: &[usize]| ws.iter().map(|w| format!("\"{}\"", WIRES[*w])).collect::<Vec<_>>().join(",");
    let per_wire = |f: &dyn Fn(usize) -> String| (0..4).map(f).collect::<Vec<_>>().join(",");
    let msm_cells: usize = linked.iter().map(|w| nonzero(*w)).sum();
    let small: usize = linked.iter().map(|w| stats[*w][1] + stats[*w][2]).sum();
    println!(
        "{{\"mode\":\"{mode}\",\"bytes\":{n},\"elements\":{elements},\"dyadic_size\":{},\"active_rows\":{},\
         \"masked_rows\":{},\"linked_wires\":[{}],\"pairings\":{},\"nonzero_cells\":[{}],\
         \"cells_le_8_bits\":[{}],\"cells_le_16_bits\":[{}],\"cells_le_32_bits\":[{}],\"cells_le_64_bits\":[{}],\
         \"cells_full_size\":[{}],\"committed_occurrences\":[{}],\"committed_only_in_linked\":{},\
         \"msm_cells\":{msm_cells},\"msm_small_share\":{:.6},\"wire_commitments_match_bb\":true,\
         \"wire_commitment_proof_offsets\":[{}],\"wire_commit_s\":{wire_commit_s},\"setup_s\":{setup_s},\
         \"key_p_bytes\":{key_p},\"key_pedersen_bytes\":{key_h},\"commit_s\":{},\"prove_s\":{},\"verify_s\":{},\
         \"commit_runs\":{},\"prove_runs\":{},\"verify_runs\":{},\"proof_bytes\":32}}",
        dump.dyadic,
        dump.active,
        dump.masked,
        names(&linked),
        ell + 2,
        per_wire(&|w| nonzero(w).to_string()),
        per_wire(&|w| stats[w][1].to_string()),
        per_wire(&|w| (stats[w][1] + stats[w][2]).to_string()),
        per_wire(&|w| (stats[w][1] + stats[w][2] + stats[w][3]).to_string()),
        per_wire(&|w| (stats[w][1] + stats[w][2] + stats[w][3] + stats[w][4]).to_string()),
        per_wire(&|w| stats[w][5].to_string()),
        per_wire(&|w| occurrences[w].to_string()),
        masks.iter().filter(|m| **m & !subset == 0).count(),
        small as f64 / msm_cells as f64,
        proof_offsets.iter().map(|o| o.to_string()).collect::<Vec<_>>().join(","),
        median(&commit_s),
        median(&prove_s),
        median(&verify_s),
        list(&commit_s),
        list(&prove_s),
        list(&verify_s),
    );
}
