use super::*;

#[test]
fn basic_three_values() {
    // log(exp(1) + exp(2) + exp(3)) = log(e + e² + e³) ≈ 3.40761
    let result = log_sum_exp(&[1.0, 2.0, 3.0]);
    let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
    assert!((result - expected).abs() < 1e-14);
}

#[test]
fn single_element() {
    assert_eq!(log_sum_exp(&[42.0]), 42.0);
}

#[test]
fn empty_is_neg_infinity() {
    assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
}

#[test]
fn all_neg_infinity() {
    assert_eq!(log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]), f64::NEG_INFINITY);
}

#[test]
fn nan_propagates() {
    assert!(log_sum_exp(&[1.0, f64::NAN, 3.0]).is_nan());
}

#[test]
fn large_values_no_overflow() {
    // exp(1000) overflows f64, but lse should handle it
    let result = log_sum_exp(&[1000.0, 1001.0]);
    let expected = 1001.0 + (1.0 + (-1.0_f64).exp()).ln();
    assert!((result - expected).abs() < 1e-12);
}

#[test]
fn small_values_no_underflow() {
    // exp(-1000) underflows to 0, but lse should give -999.something
    let result = log_sum_exp(&[-1000.0, -999.0]);
    let expected = -999.0 + (1.0 + (-1.0_f64).exp()).ln();
    assert!((result - expected).abs() < 1e-12);
}

// --- Pairwise ---

#[test]
fn pair_identity_left() {
    assert_eq!(log_sum_exp_pair(f64::NEG_INFINITY, 5.0), 5.0);
}

#[test]
fn pair_identity_right() {
    assert_eq!(log_sum_exp_pair(5.0, f64::NEG_INFINITY), 5.0);
}

#[test]
fn pair_associative() {
    let a = 1.0;
    let b = 2.0;
    let c = 3.0;
    let lhs = log_sum_exp_pair(log_sum_exp_pair(a, b), c);
    let rhs = log_sum_exp_pair(a, log_sum_exp_pair(b, c));
    assert!((lhs - rhs).abs() < 1e-14, "associativity: {lhs} != {rhs}");
}

#[test]
fn pair_commutative() {
    let a = 1.5;
    let b = 3.7;
    assert_eq!(log_sum_exp_pair(a, b), log_sum_exp_pair(b, a));
}

#[test]
fn pair_nan_propagates() {
    assert!(log_sum_exp_pair(f64::NAN, 1.0).is_nan());
    assert!(log_sum_exp_pair(1.0, f64::NAN).is_nan());
}
