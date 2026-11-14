#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use ff::Field;
use criterion::{BenchmarkId, Criterion};
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::{
    poly::kzg::commitment::ParamsKZG,
    halo2curves::bn256::{Fr, Bn256},
    arithmetic::{best_multiexp, multiexp_chunked},
};
use rand_core::OsRng;

pub fn bench_kzg_multiexp(c: &mut Criterion) {
    let mut group = c.benchmark_group("kzg_msm");
    group.sample_size(10);
    let max_k: u32 = std::env::var("MSM_MAX_K").ok().and_then(|v| v.parse().ok()).unwrap_or(19);
    let min_k: u32 = std::env::var("MSM_MIN_K").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    let params = ParamsKZG::<Bn256>::new(max_k);
    let bases = params.get_g();
    let scalars: Vec<Fr> = (0..(1usize << max_k)).map(|_| Fr::random(OsRng)).collect();

    for k in min_k..=max_k {
        let size = 1usize << k;
        group.bench_function(BenchmarkId::new("batch_affine", k), |b| {
            b.iter(|| best_multiexp(&scalars[..size], &bases[..size]))
        });
        group.bench_function(BenchmarkId::new("chunked", k), |b| {
            b.iter(|| multiexp_chunked(&scalars[..size], &bases[..size]))
        });
    }
}

criterion_group!{
    name=benches;
    config=Criterion::default();
    targets=bench_kzg_multiexp
  }
criterion_main!(benches);
