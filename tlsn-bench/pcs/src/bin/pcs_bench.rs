//! Time the external commitment and opening on their own.
//!
//! `pcs_bench <ipa|kzg> <elements> [reps]` prints one JSON object. Setup is
//! not timed. `elements` counts the packed payload plus the mask.

use {
    ark_bn254::Fr,
    ark_ec::CurveGroup,
    ark_ff::UniformRand,
    std::time::Instant,
    tlsn_bench_pcs::{eval_poly, ipa, kzg},
};

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: pcs_bench <ipa|kzg> <elements> [reps]");
        std::process::exit(2);
    }
    let scheme = args[1].as_str();
    let n: usize = args[2].parse().expect("elements");
    let reps: usize = args.get(3).map_or(5, |r| r.parse().expect("reps"));
    let mut rng = rand::thread_rng();
    let coeffs: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let z = Fr::rand(&mut rng);
    let v = eval_poly(&coeffs, z);
    let (mut commit, mut open, mut verify) = (vec![], vec![], vec![]);
    let proof_bytes;
    match scheme {
        "ipa" => {
            let ck = ipa::CommitKey::setup(n);
            let blind = Fr::rand(&mut rng);
            let mut size = 0;
            for _ in 0..reps {
                let t = Instant::now();
                let c = ck.commit(&coeffs, blind).into_affine();
                commit.push(t.elapsed().as_secs_f64());
                let t = Instant::now();
                let proof = ipa::open(&ck, &coeffs, blind, &c, z, v, &mut rng);
                open.push(t.elapsed().as_secs_f64());
                let t = Instant::now();
                assert!(ipa::verify(&ck, &c, z, v, &proof));
                verify.push(t.elapsed().as_secs_f64());
                size = proof.size_bytes();
            }
            proof_bytes = size;
        }
        "kzg" => {
            let srs = kzg::Srs::insecure_setup(n, &mut rng);
            let blind = [Fr::rand(&mut rng), Fr::rand(&mut rng)];
            let mut size = 0;
            for _ in 0..reps {
                let t = Instant::now();
                let c = srs.commit(&coeffs, blind).into_affine();
                commit.push(t.elapsed().as_secs_f64());
                let t = Instant::now();
                let proof = srs.open(&coeffs, blind, z);
                open.push(t.elapsed().as_secs_f64());
                let t = Instant::now();
                assert!(srs.verify(&c, z, v, &proof));
                verify.push(t.elapsed().as_secs_f64());
                size = proof.size_bytes();
            }
            proof_bytes = size;
        }
        other => panic!("unknown scheme {other}"),
    }
    println!(
        "{{\"scheme\":\"{scheme}\",\"elements\":{n},\"commit_s\":{},\"open_s\":{},\"verify_s\":{},\"proof_bytes\":{proof_bytes}}}",
        median(commit),
        median(open),
        median(verify)
    );
}
