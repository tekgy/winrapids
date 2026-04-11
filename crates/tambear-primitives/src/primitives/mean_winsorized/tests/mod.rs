use super::*;

#[test]
fn alpha_zero_is_arithmetic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let w = mean_winsorized(&data, 0.0);
    let a = crate::mean_arithmetic::mean_arithmetic(&data);
    assert!((w - a).abs() < 1e-14);
}

#[test]
fn clamps_extremes() {
    // [1, 2, 3, 4, 100], alpha=0.2 → k=1 → clamp to [2, 2, 3, 4, 4] → mean = 3.0
    let data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
    assert!((mean_winsorized(&data, 0.2) - 3.0).abs() < 1e-14);
}

#[test]
fn preserves_n() {
    // Unlike trimmed, winsorized always uses all n elements
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    // alpha=0.2, k=1: [2,2,3,4,4] → mean = 15/5 = 3.0
    assert!((mean_winsorized(&data, 0.2) - 3.0).abs() < 1e-14);
}

#[test]
fn empty_is_nan() {
    assert!(mean_winsorized(&[], 0.1).is_nan());
}
