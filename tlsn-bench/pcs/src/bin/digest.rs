//! `digest <keccak256|blake3>`: hash stdin, print hex. Used by bench.py to
//! compute the expected commitments of the Keccak-256 and BLAKE3 rows.
use std::io::Read;

use sha3::{Digest, Keccak256};

fn main() {
    let alg = std::env::args().nth(1).expect("usage: digest <keccak256|blake3>");
    let mut data = Vec::new();
    std::io::stdin().read_to_end(&mut data).expect("read stdin");
    let out: Vec<u8> = match alg.as_str() {
        "keccak256" => Keccak256::digest(&data).to_vec(),
        "blake3" => blake3::hash(&data).as_bytes().to_vec(),
        other => panic!("unknown hash {other}"),
    };
    println!("{}", out.iter().map(|b| format!("{b:02x}")).collect::<String>());
}
