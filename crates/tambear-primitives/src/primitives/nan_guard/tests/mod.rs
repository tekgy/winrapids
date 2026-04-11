use super::*;

#[test]
fn nan_min_propagates() {
    assert!(nan_min(1.0, f64::NAN).is_nan());
    assert!(nan_min(f64::NAN, 1.0).is_nan());
    assert!(nan_min(f64::NAN, f64::NAN).is_nan());
}

#[test]
fn nan_min_normal() {
    assert_eq!(nan_min(1.0, 2.0), 1.0);
    assert_eq!(nan_min(-1.0, 1.0), -1.0);
}

#[test]
fn nan_max_propagates() {
    assert!(nan_max(1.0, f64::NAN).is_nan());
    assert!(nan_max(f64::NAN, 1.0).is_nan());
}

#[test]
fn nan_max_normal() {
    assert_eq!(nan_max(1.0, 2.0), 2.0);
}

#[test]
fn sorted_total_puts_nan_last() {
    let data = vec![3.0, f64::NAN, 1.0, 2.0];
    let s = sorted_total(&data);
    assert_eq!(s[0], 1.0);
    assert_eq!(s[1], 2.0);
    assert_eq!(s[2], 3.0);
    assert!(s[3].is_nan());
}

#[test]
fn sorted_finite_excludes_nan() {
    let data = vec![3.0, f64::NAN, 1.0, f64::NAN];
    let s = sorted_finite(&data);
    assert_eq!(s, vec![1.0, 3.0]);
}
