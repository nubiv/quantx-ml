use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use polars::prelude::*;

mod utils;

#[cfg(target_arch = "aarch64")]
use utils::neon_mul;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use utils::{avx2_mul, sse_mul};
use utils::{gen_input, scalar_mul};

fn bench_all(c: &mut Criterion) {
    let mut g = c.benchmark_group("scalar_vs_simd_vs_polars");

    for &n in &[1usize << 12, 1usize << 16, 1usize << 20] {
        let (a, b) = gen_input(n);
        g.throughput(Throughput::Elements(n as u64));

        // scalar
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |ben, &_n| {
            ben.iter(|| {
                let out = scalar_mul(black_box(&a), black_box(&b));
                black_box(out);
            })
        });

        // SIMD per-arch
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            g.bench_with_input(BenchmarkId::new("simd_sse", n), &n, |ben, &_n| unsafe {
                ben.iter(|| {
                    let out = sse_mul(black_box(&a), black_box(&b));
                    black_box(out);
                })
            });

            if is_x86_feature_detected!("avx2") {
                g.bench_with_input(BenchmarkId::new("simd_avx2", n), &n, |ben, &_n| unsafe {
                    ben.iter(|| {
                        let out = avx2_mul(black_box(&a), black_box(&b));
                        black_box(out);
                    })
                });
            }
        }

        #[cfg(target_arch = "aarch64")]
        g.bench_with_input(BenchmarkId::new("simd_neon", n), &n, |ben, &_n| unsafe {
            ben.iter(|| {
                let out = neon_mul(black_box(&a), black_box(&b));
                black_box(out);
            })
        });

        // Polars (same for both arches)
        let a_s = Series::new("a".into(), a.clone());
        let b_s = Series::new("b".into(), b.clone());
        g.bench_with_input(BenchmarkId::new("polars_series_mul", n), &n, |ben, &_n| {
            ben.iter(|| {
                let out = (a_s.f32().unwrap() * b_s.f32().unwrap()).into_series();
                let _ = black_box(out);
            })
        });

        let df = DataFrame::new(vec![a_s.clone().into(), b_s.clone().into()]).unwrap();
        let lf = df.lazy().select([col("a") * col("b")]);
        g.bench_with_input(BenchmarkId::new("polars_lazy_select", n), &n, |ben, &_n| {
            ben.iter(|| {
                let out = lf.clone().collect().unwrap();
                black_box(out);
            })
        });
    }

    g.finish();
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
