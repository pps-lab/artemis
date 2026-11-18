use {
    ark_bn254::Fr,
    ark_ff::PrimeField,
    ark_serialize::CanonicalSerialize,
    sha2::{Digest, Sha256},
};

/// Minimal Fiat-Shamir transcript: a running SHA-256 chain.
pub(crate) struct Transcript {
    state: [u8; 32],
}

impl Transcript {
    pub(crate) fn new(label: &[u8]) -> Self {
        Self {
            state: Sha256::digest(label).into(),
        }
    }

    pub(crate) fn absorb<T: CanonicalSerialize>(&mut self, label: &[u8], value: &T) {
        let mut bytes = Vec::with_capacity(value.compressed_size());
        value
            .serialize_compressed(&mut bytes)
            .expect("serialization into a Vec cannot fail");
        let mut h = Sha256::new();
        h.update(self.state);
        h.update((label.len() as u64).to_le_bytes());
        h.update(label);
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
        self.state = h.finalize().into();
    }

    /// A challenge with 2^-125 statistical distance from uniform (two blocks
    /// reduced modulo the 254-bit order).
    pub(crate) fn challenge(&mut self, label: &[u8]) -> Fr {
        let mut wide = [0u8; 64];
        for (i, chunk) in wide.chunks_mut(32).enumerate() {
            let mut h = Sha256::new();
            h.update(self.state);
            h.update(b"challenge");
            h.update(label);
            h.update([i as u8]);
            chunk.copy_from_slice(&h.finalize());
        }
        let mut h = Sha256::new();
        h.update(self.state);
        h.update(b"ratchet");
        h.update(label);
        self.state = h.finalize().into();
        Fr::from_le_bytes_mod_order(&wide)
    }
}
