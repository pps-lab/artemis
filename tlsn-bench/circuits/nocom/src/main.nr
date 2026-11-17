//! Lower bound: the payload bytes are range-checked private inputs and
//! nothing binds them to a commitment.

global N: u32 = 1024;

fn main(payload: [u8; N]) {
    // Empty predicate. The `u8` parameter type range-checks every byte.
    let _ = payload;
}
