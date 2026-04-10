//! Workup parity: `nonparametric::pearson_r`
//!
//! Supporting test suite for
//! `docs/research/atomic-industrialization/pearson_r.md`
//!
//! Each test maps to one case in the workup. If any test fails, the workup
//! must be updated — either the implementation was corrected (add a row to
//! the version history) or the expected value was wrong.
//!
//! Oracle values computed via mpmath at 50 decimal digits (see workup
//! Appendix B for the reproduction script).

use tambear::nonparametric::pearson_r;

// ─── Section 5.2: oracle cases ────────────────────────────────────────────────

/// Case 1 (workup §5.2): monotone increasing → r = +1.0
#[test]
fn pearson_r_monotone_increasing_is_one() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let got = pearson_r(&x, &y);
    assert!(
        (got - 1.0).abs() < 1e-15,
        "expected 1.0, got {}", got
    );
}

/// Case 2 (workup §5.2): monotone decreasing → r = −1.0
#[test]
fn pearson_r_monotone_decreasing_is_minus_one() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [5.0, 4.0, 3.0, 2.0, 1.0];
    let got = pearson_r(&x, &y);
    assert!(
        (got - (-1.0)).abs() < 1e-15,
        "expected -1.0, got {}", got
    );
}

/// Case 3 (workup §5.2): partial correlation — exact rational value.
///
/// x = [1,2,3,4,5], y = [2,1,4,3,5]
/// mpmath 50dp: 0.8 (exact)
#[test]
fn pearson_r_partial_matches_oracle() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [2.0, 1.0, 4.0, 3.0, 5.0];
    let got = pearson_r(&x, &y);
    assert!(
        (got - 0.8).abs() < 1e-15,
        "expected 0.8, got {}", got
    );
}

/// Case 4 (workup §5.2): Rodgers-Nicewander (1988) textbook dataset.
///
/// x = [2,4,4,4,5,5,7,9], y = [1,2,3,4,4,5,6,7]
/// mpmath 50dp: 0.93541434669348534639593718307913732543900495194468
/// scipy 1.x:   0.9354143466934852 (≤ 1 ULP difference)
#[test]
fn pearson_r_textbook_matches_oracle() {
    let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let y = [1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0];
    let got = pearson_r(&x, &y);
    let expected = 0.9354143466934853_f64; // mpmath 50dp rounded to nearest f64
    assert!(
        (got - expected).abs() < 1e-14,
        "textbook case: expected {}, got {}", expected, got
    );
}

/// Case 5 (workup §5.2): constant x → NaN.
#[test]
fn pearson_r_constant_x_is_nan() {
    let x = [3.0, 3.0, 3.0, 3.0, 3.0];
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let got = pearson_r(&x, &y);
    assert!(got.is_nan(), "constant x: expected NaN, got {}", got);
}

/// Case 5b: constant y → NaN.
#[test]
fn pearson_r_constant_y_is_nan() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [7.0, 7.0, 7.0, 7.0, 7.0];
    let got = pearson_r(&x, &y);
    assert!(got.is_nan(), "constant y: expected NaN, got {}", got);
}

/// Case 6 (workup §5.2): n=20 random (numpy.random.seed(42)).
///
/// mpmath 50dp: -0.15729399538437135121...
/// scipy 1.x:   -0.1572939953843713
#[test]
fn pearson_r_random_n20_matches_oracle() {
    let x = [
         0.4967141530112327, -0.13826430117118466,  0.6476885381006925,
         1.5230298564080254, -0.23415337472333597, -0.23413695694918055,
         1.5792128155073915,  0.7674347291529088,  -0.4694743859349521,
         0.5425600435859647, -0.46341769281246226, -0.46572975357025687,
         0.24196227156603412,-1.913280244657798,   -1.7249178325130328,
        -0.5622875292409727, -1.0128311203344238,   0.3142473325952739,
        -0.9080240755212109, -1.4123037013352915,
    ];
    let y = [
         1.465648768921554,  -0.22577630048653566,  0.06752820468792384,
        -1.4247481862134568, -0.5443827245251827,   0.11092258970986608,
        -1.1509935774223028,  0.37569801834567196, -0.600638689918805,
        -0.2916937497932768, -0.6017066122293969,   1.8522781845089378,
        -0.013497224737933921,-1.0577109289559004,  0.822544912103189,
        -1.2208436499710222,  0.2088635950047554,  -1.9596701238797756,
        -1.3281860488984305,  0.19686123586912352,
    ];
    let got = pearson_r(&x, &y);
    let expected = -0.15729399538437135_f64;
    assert!(
        (got - expected).abs() < 1e-14,
        "n=20 seed42: expected {}, got {}", expected, got
    );
}

/// Case 7 (workup §5.2): near-constant x stress test.
///
/// x = [1 + 1e-8*i for i in 1..=10], y = [1,3,2,5,4,7,6,8,9,10]
/// mpmath 50dp: 0.96363636384678606936...
/// scipy 1.x:   0.963636363846786 (≤ 1 ULP)
///
/// Tests that the two-pass algorithm is numerically stable when x has
/// very small variance relative to its mean (cancellation-sensitive input).
#[test]
fn pearson_r_near_constant_x_stress() {
    let eps = 1e-8_f64;
    let x: Vec<f64> = (1..=10).map(|i| 1.0 + eps * i as f64).collect();
    let y = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0, 9.0, 10.0];
    let got = pearson_r(&x, &y);
    let expected = 0.9636363638467861_f64; // mpmath 50dp nearest f64
    assert!(
        (got - expected).abs() < 1e-13,
        "near-const stress: expected {}, got {} (diff={})",
        expected, got, (got - expected).abs()
    );
}

/// Case 8 (workup §5.2): large-magnitude shift — validates two-pass stability.
///
/// x = [1e10+1, 1e10+2, 1e10+3, 1e10+4, 1e10+5], y = [1,2,3,4,5]
/// mpmath 50dp: 1.0 (exact)
///
/// The one-pass formula (n·Σx² − (Σx)²) catastrophically cancels on this
/// input. The two-pass algorithm is correct. This test validates that
/// tambear's implementation is immune to this failure mode.
#[test]
fn pearson_r_large_shift_two_pass_stable() {
    let shift = 1e10_f64;
    let x: Vec<f64> = (1..=5).map(|i| shift + i as f64).collect();
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let got = pearson_r(&x, &y);
    assert!(
        (got - 1.0).abs() < 1e-13,
        "large-shift case: expected 1.0, got {} (one-pass would give NaN/wrong)", got
    );
}

// ─── Section 4: edge cases ───────────────────────────────────────────────────

/// n=0 → NaN.
#[test]
fn pearson_r_empty_is_nan() {
    let got = pearson_r(&[], &[]);
    assert!(got.is_nan(), "empty input: expected NaN, got {}", got);
}

/// n=1 → NaN (single point has zero variance).
#[test]
fn pearson_r_n1_is_nan() {
    let got = pearson_r(&[3.14], &[2.72]);
    assert!(got.is_nan(), "n=1: expected NaN, got {}", got);
}

// ─── Section 8: invariants ───────────────────────────────────────────────────

/// Invariant (workup §8.1): r(x, y) = r(y, x). Symmetry.
#[test]
fn pearson_r_symmetric() {
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let y = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0];
    let rxy = pearson_r(&x, &y);
    let ryx = pearson_r(&y, &x);
    assert!(
        (rxy - ryx).abs() < 1e-15,
        "symmetry: r(x,y)={}, r(y,x)={}", rxy, ryx
    );
}

/// Invariant (workup §8.2): r(x+c, y) = r(x, y). Shift invariance.
#[test]
fn pearson_r_shift_invariant_x() {
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let y = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0];
    let r0 = pearson_r(&x, &y);
    let x_shifted: Vec<f64> = x.iter().map(|v| v + 1e9).collect();
    let r1 = pearson_r(&x_shifted, &y);
    assert!(
        (r0 - r1).abs() < 1e-13,
        "shift invariance: r(x,y)={}, r(x+1e9,y)={}", r0, r1
    );
}

/// Invariant (workup §8.2): r(x+c, y) = r(x, y). Shift invariance in y.
#[test]
fn pearson_r_shift_invariant_y() {
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let y = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0];
    let r0 = pearson_r(&x, &y);
    let y_shifted: Vec<f64> = y.iter().map(|v| v - 5e8).collect();
    let r1 = pearson_r(&x, &y_shifted);
    assert!(
        (r0 - r1).abs() < 1e-13,
        "shift invariance y: r(x,y)={}, r(x,y-5e8)={}", r0, r1
    );
}

/// Invariant (workup §8.3): r(αx, y) = r(x, y) for α > 0. Positive scale.
#[test]
fn pearson_r_scale_invariant_positive() {
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let y = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0];
    let r0 = pearson_r(&x, &y);
    let x_scaled: Vec<f64> = x.iter().map(|v| v * 1234.5).collect();
    let r1 = pearson_r(&x_scaled, &y);
    assert!(
        (r0 - r1).abs() < 1e-15,
        "scale invariance (positive): r(x,y)={}, r(1234.5x,y)={}", r0, r1
    );
}

/// Invariant (workup §8.4): r(−x, y) = −r(x, y). Sign flip under negation.
#[test]
fn pearson_r_sign_flip_under_negation() {
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let y = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0];
    let r0 = pearson_r(&x, &y);
    let x_neg: Vec<f64> = x.iter().map(|v| -v).collect();
    let r1 = pearson_r(&x_neg, &y);
    assert!(
        (r0 + r1).abs() < 1e-15,
        "sign flip: r(x,y) + r(-x,y) = {} (expected 0)", r0 + r1
    );
}

/// Invariant (workup §8.5): r ∈ [−1, 1] for all inputs.
///
/// Tested over 50 random cases (Xoshiro256, seed 99999).
#[test]
fn pearson_r_bounded_minus_one_to_one() {
    use tambear::rng::{Xoshiro256, TamRng};
    let mut rng = Xoshiro256::new(99999);
    for trial in 0..50 {
        let x: Vec<f64> = (0..30).map(|_| rng.next_f64() * 100.0 - 50.0).collect();
        let y: Vec<f64> = (0..30).map(|_| rng.next_f64() * 100.0 - 50.0).collect();
        let r = pearson_r(&x, &y);
        if !r.is_nan() {
            assert!(
                r >= -1.0 - 1e-12 && r <= 1.0 + 1e-12,
                "trial {}: r = {} out of bounds", trial, r
            );
        }
    }
}

/// Invariant: r(x, x) = 1.0 for any non-constant x.
#[test]
fn pearson_r_self_correlation_is_one() {
    let x = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
    let got = pearson_r(&x, &x);
    assert!(
        (got - 1.0).abs() < 1e-14,
        "r(x,x): expected 1.0, got {}", got
    );
}
