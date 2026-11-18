//! TLSNotary age check with the commitment BLAKE3(dob || blinder) (black box) recomputed in the circuit.

use date::Date;

fn main(
    proof_date: pub Date,
    committed_hash: pub [u8; 32],
    payload: [u8; 10], // "1985-03-12"
    blinder: [u8; 16],
) {
    let mut input: [u8; 26] = [0; 26];
    for i in 0..10 {
        input[i] = payload[i];
    }
    for i in 0..16 {
        input[10 + i] = blinder[i];
    }
    assert(std::hash::blake3(input) == committed_hash);
    assert(is_18(payload, proof_date));
}

fn is_18(dob: [u8; 10], proof_date: Date) -> bool {
    let s: str<8> =
        [dob[0], dob[1], dob[2], dob[3], dob[5], dob[6], dob[8], dob[9]].as_str_unchecked();
    Date::from_str_long_year(s).add_years(18).lt(proof_date)
}
