use super::*;

#[test]
fn equal_weights_is_arithmetic() {
    let data = vec![1.0, 2.0, 3.0];
    let weights = vec![1.0, 1.0, 1.0];
    assert!((mean_weighted(&data, &weights) - 2.0).abs() < 1e-14);
}

#[test]
fn weighted_toward_heavy() {
    let data = vec![1.0, 10.0];
    let weights = vec![1.0, 9.0];
    // (1*1 + 10*9) / (1+9) = 91/10 = 9.1
    assert!((mean_weighted(&data, &weights) - 9.1).abs() < 1e-14);
}

#[test]
fn zero_weight_ignored() {
    let data = vec![1.0, 1000.0, 3.0];
    let weights = vec![1.0, 0.0, 1.0];
    assert!((mean_weighted(&data, &weights) - 2.0).abs() < 1e-14);
}

#[test]
fn all_zero_weights_is_nan() {
    assert!(mean_weighted(&[1.0, 2.0], &[0.0, 0.0]).is_nan());
}

#[test]
fn mismatched_lengths_is_nan() {
    assert!(mean_weighted(&[1.0, 2.0], &[1.0]).is_nan());
}

#[test]
fn empty_is_nan() {
    assert!(mean_weighted(&[], &[]).is_nan());
}
