use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{Matrix2x3, SMatrix};
use similarity_least_squares;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("three points", |b| {
        b.iter(|| {
            similarity_least_squares::from_smatrices(
                black_box(Matrix2x3::new(1.0, 1.0, -2.0, 0.0, 2.0, -0.5)),
                black_box(Matrix2x3::new(-1.5, -1.5, -3.0, -2.0, -1.0, -2.25)),
            )
            .unwrap();
        })
    });

    c.bench_function("five points", |b| {
        b.iter(|| {
            similarity_least_squares::from_smatrices(
                black_box(Matrix2x3::new(1.0, 1.0, -2.0, 0.0, 2.0, -0.5)),
                black_box(Matrix2x3::new(-1.5, -1.5, -3.0, -2.0, -1.0, -2.25)),
            )
            .unwrap();
        })
    });

    c.bench_function("seven points", |b| {
        b.iter(|| {
            similarity_least_squares::from_smatrices(
                black_box(SMatrix::<f64, 2, 7>::from_vec(vec![
                    0.0, 0.0, 1.0, 1.0, -2.0, 0.0, 2.0, -0.5, -1.0, -1.0, 1.0, 0.0, 0.0, 1.0,
                ])),
                black_box(SMatrix::<f64, 2, 7>::from_vec(vec![
                    0.0, -2.0, 1.5, -0.5, -3.0, -2.0, 3.0, -2.75, -1.5, -3.5, 1.5, -2.0, 0.0, -0.5,
                ])),
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
