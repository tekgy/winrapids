use super::*;

#[test]
fn basic() {
    // geometric mean of 2, 8 = sqrt(16) = 4
    assert!((mean_geometric(&[2.0, 8.0]) - 4.0).abs() < 1e-14);
}

#[test]
fn all_equal() {
    assert!((mean_geometric(&[5.0, 5.0, 5.0]) - 5.0).abs() < 1e-14);
}

#[test]
fn le_arithmetic() {
    // Power mean inequality: geometric ≤ arithmetic for all data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let geo = mean_geometric(&data);
    let arith = crate::mean_arithmetic::mean_arithmetic(&data);
    assert!(geo <= arith + 1e-14);
}

#[test]
fn negative_is_nan() {
    assert!(mean_geometric(&[1.0, -1.0, 3.0]).is_nan());
}

#[test]
fn zero_is_nan() {
    assert!(mean_geometric(&[1.0, 0.0, 3.0]).is_nan());
}

#[test]
fn empty_is_nan() {
    assert!(mean_geometric(&[]).is_nan());
}

#[test]
fn single() {
    assert!((mean_geometric(&[7.0]) - 7.0).abs() < 1e-14);
}
