use super::*;

#[test]
fn alpha_zero_is_arithmetic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let trimmed = mean_trimmed(&data, 0.0);
    let arith = crate::mean_arithmetic::mean_arithmetic(&data);
    assert!((trimmed - arith).abs() < 1e-14);
}

#[test]
fn trims_extremes() {
    // With outliers: [1, 2, 3, 4, 100], alpha=0.2 trims 1 from each end → mean(2,3,4) = 3
    let data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
    assert!((mean_trimmed(&data, 0.2) - 3.0).abs() < 1e-14);
}

#[test]
fn robust_to_outliers() {
    let clean = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let dirty = vec![1.0, 2.0, 3.0, 4.0, 1000.0];
    let clean_trim = mean_trimmed(&clean, 0.2);
    let dirty_trim = mean_trimmed(&dirty, 0.2);
    // After trimming, both should give mean(2,3,4) = 3
    assert!((clean_trim - dirty_trim).abs() < 1e-14);
}

#[test]
fn empty_is_nan() {
    assert!(mean_trimmed(&[], 0.1).is_nan());
}

#[test]
fn bad_alpha_is_nan() {
    assert!(mean_trimmed(&[1.0, 2.0], -0.1).is_nan());
    assert!(mean_trimmed(&[1.0, 2.0], 0.5).is_nan());
    assert!(mean_trimmed(&[1.0, 2.0], f64::NAN).is_nan());
}
