use super::*;

#[test]
fn basic() {
    // RMS of 3, 4 = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
    let result = mean_quadratic(&[3.0, 4.0]);
    assert!((result - (12.5_f64).sqrt()).abs() < 1e-14);
}

#[test]
fn ge_arithmetic() {
    // Power mean inequality: quadratic ≥ arithmetic
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let quad = mean_quadratic(&data);
    let arith = crate::mean_arithmetic::mean_arithmetic(&data);
    assert!(quad >= arith - 1e-14, "quadratic {quad} < arithmetic {arith}");
}

#[test]
fn all_equal() {
    assert!((mean_quadratic(&[3.0, 3.0, 3.0]) - 3.0).abs() < 1e-14);
}

#[test]
fn negative_values_magnitude() {
    // RMS treats sign as irrelevant
    assert!((mean_quadratic(&[-3.0, 3.0]) - 3.0).abs() < 1e-14);
}

#[test]
fn empty_is_nan() {
    assert!(mean_quadratic(&[]).is_nan());
}
