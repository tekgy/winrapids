use super::*;

#[test]
fn basic() {
    assert_eq!(mean_arithmetic(&[1.0, 2.0, 3.0]), 2.0);
}

#[test]
fn single() {
    assert_eq!(mean_arithmetic(&[42.0]), 42.0);
}

#[test]
fn empty_is_nan() {
    assert!(mean_arithmetic(&[]).is_nan());
}

#[test]
fn nan_propagates() {
    assert!(mean_arithmetic(&[1.0, f64::NAN, 3.0]).is_nan());
}

#[test]
fn negative_values() {
    assert_eq!(mean_arithmetic(&[-2.0, -1.0, 0.0, 1.0, 2.0]), 0.0);
}

#[test]
fn accumulator_matches_batch() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let batch = mean_arithmetic(&data);
    let mut acc = MeanAccumulator::new();
    for &x in &data { acc.push(x); }
    assert!((acc.value() - batch).abs() < 1e-14);
}

#[test]
fn accumulator_merge() {
    let mut a = MeanAccumulator::new();
    let mut b = MeanAccumulator::new();
    for &x in &[1.0, 2.0, 3.0] { a.push(x); }
    for &x in &[4.0, 5.0] { b.push(x); }
    let merged = a.merge(&b);
    assert!((merged.value() - 3.0).abs() < 1e-14);
}

#[test]
fn accumulator_empty() {
    assert!(MeanAccumulator::new().value().is_nan());
}
