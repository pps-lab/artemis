#[macro_use]
extern crate criterion;
extern crate rand_core;
extern crate group;
use group::ff::Field;
use criterion::{BenchmarkId, Criterion};

use halo2_proofs::{
  halo2curves::bn256::Fr,
  arithmetic::best_fft,
  poly::EvaluationDomain
};
use rand_core::OsRng;

pub fn bench_kzg_fft(c: &mut Criterion) {
  let j = 5;
  let mut group = c.benchmark_group("kzg_fft");
  group.sample_size(10);
  let max_k: u32 = std::env::var("FFT_MAX_K").ok().and_then(|v| v.parse().ok()).unwrap_or(19);
  let min_k: u32 = std::env::var("FFT_MIN_K").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
  let label = std::env::var("FFT").unwrap_or_else(|_| "parallel".to_string());
  for k in min_k..=max_k {
    let domain = EvaluationDomain::new(j,k);
    let omega = domain.get_omega();
    let l = 1<<k;
    let data = domain.get_fft_data(l);
    group.bench_function(BenchmarkId::new(&label, k), |b| {
        let mut a = (0..(1 << k)).map(|_| Fr::random(OsRng)).collect::<Vec<_>>();
        b.iter(|| {
            best_fft(&mut a, omega, k as u32, data, false);
        });
    });
  }
}

criterion_group!{
  name=benches;
  config=Criterion::default();
  targets=bench_kzg_fft
}
criterion_main!(benches);
