use super::*;

#[test]
fn speed_example() {
    // 60 mph out, 40 mph back → harmonic mean = 48
    assert!((mean_harmonic(&[60.0, 40.0]) - 48.0).abs() < 1e-12);
}

#[test]
fn all_equal() {
    assert!((mean_harmonic(&[5.0, 5.0, 5.0]) - 5.0).abs() < 1e-14);
}

#[test]
fn le_geometric() {
    // Power mean inequality: harmonic ≤ geometric ≤ arithmetic
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let harm = mean_harmonic(&data);
    let geo = crate::mean_geometric::mean_geometric(&data);
    let arith = crate::mean_arithmetic::mean_arithmetic(&data);
    assert!(harm <= geo + 1e-14, "harmonic {harm} > geometric {geo}");
    assert!(geo <= arith + 1e-14, "geometric {geo} > arithmetic {arith}");
}

#[test]
fn negative_is_nan() {
    assert!(mean_harmonic(&[1.0, -1.0]).is_nan());
}

#[test]
fn zero_is_nan() {
    assert!(mean_harmonic(&[1.0, 0.0]).is_nan());
}

#[test]
fn empty_is_nan() {
    assert!(mean_harmonic(&[]).is_nan());
}
