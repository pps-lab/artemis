//! Age check without any commitment (lower bound).

use date::Date;

fn main(proof_date: pub Date, payload: [u8; 10]) {
    assert(is_18(payload, proof_date));
}

fn is_18(dob: [u8; 10], proof_date: Date) -> bool {
    // "YYYY-MM-DD" -> "YYYYMMDD"
    let s: str<8> =
        [dob[0], dob[1], dob[2], dob[3], dob[5], dob[6], dob[8], dob[9]].as_str_unchecked();
    Date::from_str_long_year(s).add_years(18).lt(proof_date)
}

