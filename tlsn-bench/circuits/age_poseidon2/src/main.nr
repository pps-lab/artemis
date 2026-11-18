//! Age check with a Poseidon2 commitment (native width-4 permutation) to the
//! packed date of birth and a field blinder.

use date::Date;

fn main(proof_date: pub Date, payload: [u8; 10], blinder: Field) -> pub Field {
    let mut packed: Field = 0;
    let mut weight: Field = 1;
    for i in 0..10 {
        packed += weight * (payload[i] as Field);
        weight *= 256;
    }
    assert(is_18(payload, proof_date));
    std::hash::poseidon2_permutation([packed, blinder, 0, 10])[0]
}

fn is_18(dob: [u8; 10], proof_date: Date) -> bool {
    // "YYYY-MM-DD" -> "YYYYMMDD"
    let s: str<8> =
        [dob[0], dob[1], dob[2], dob[3], dob[5], dob[6], dob[8], dob[9]].as_str_unchecked();
    Date::from_str_long_year(s).add_years(18).lt(proof_date)
}

