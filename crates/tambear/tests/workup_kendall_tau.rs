//! Workup parity: `kendall_tau`
//!
//! Supporting test suite for
//! `docs/research/workups/nonparametric/kendall_tau.md`
//!
//! Each test below maps to one case in the workup. If any test fails, the
//! workup must be updated — either the implementation was corrected (add a
//! row to the version history) or the expected value was wrong.

use tambear::nonparametric::kendall_tau;

/// Case 1 (workup §5.2): monotone increasing → τ = 1.
#[test]
fn kendall_tau_monotone_is_one() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [1.0, 2.0, 3.0, 4.0, 5.0];
    let got = kendall_tau(&x, &y);
    assert!((got - 1.0).abs() < 1e-15, "got {}", got);
}

/// Case 2 (workup §5.2): monotone decreasing → τ = −1.
#[test]
fn kendall_tau_antimonotone_is_minus_one() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [5.0, 4.0, 3.0, 2.0, 1.0];
    let got = kendall_tau(&x, &y);
    assert!((got - (-1.0)).abs() < 1e-15, "got {}", got);
}

/// Case 3 (workup §5.2): light ties
/// mpmath truth: 0.9486832980505138
/// scipy 1.17.1: 0.9486832980505138
#[test]
fn kendall_tau_light_ties_matches_mpmath() {
    let x = [1.0, 2.0, 2.0, 3.0, 4.0];
    let y = [1.0, 3.0, 2.0, 4.0, 5.0];
    let got = kendall_tau(&x, &y);
    let expected = 0.9486832980505138;
    assert!((got - expected).abs() < 1e-14, "got {}, expected {}", got, expected);
}

/// Case 4 (workup §5.2): heavy ties
/// mpmath truth: 0.4166666666666667
/// scipy 1.17.1: 0.4166666666666667
#[test]
fn kendall_tau_heavy_ties_matches_mpmath() {
    let x = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
    let y = [1.0, 2.0, 1.0, 3.0, 2.0, 3.0];
    let got = kendall_tau(&x, &y);
    let expected = 0.41666666666666663;
    assert!((got - expected).abs() < 1e-14, "got {}, expected {}", got, expected);
}

/// Case 5 (workup §5.2): random n=20 (seed=42 via numpy.random)
///
/// mpmath truth (50 dp): -0.042105263157894736
/// scipy 1.17.1:         -0.042105263157894736
#[test]
fn kendall_tau_random_n20_matches_mpmath() {
    // Precomputed via `numpy.random.seed(42); randn(20)`
    let x = [
        0.4967141530112327, -0.13826430117118466, 0.6476885381006925,
        1.5230298564080254, -0.23415337472333597, -0.23413695694918055,
        1.5792128155073915, 0.7674347291529088, -0.4694743859349521,
        0.5425600435859647, -0.46341769281246226, -0.46572975357025687,
        0.24196227156603412, -1.913280244657798, -1.7249178325130328,
        -0.5622875292409727, -1.0128311203344238, 0.3142473325952739,
        -0.9080240755212109, -1.4123037013352915,
    ];
    let y = [
        1.465648768921554, -0.22577630048653566, 0.06752820468792384,
        -1.4247481862134568, -0.5443827245251827, 0.11092258970986608,
        -1.1509935774223028, 0.37569801834567196, -0.600638689918805,
        -0.2916937497932768, -0.6017066122293969, 1.8522781845089378,
        -0.013497224737933921, -1.0577109289559004, 0.822544912103189,
        -1.2208436499710222, 0.2088635950047554, -1.9596701238797756,
        -1.3281860488984305, 0.19686123586912352,
    ];
    let got = kendall_tau(&x, &y);
    let expected = -0.042105263157894736;
    assert!((got - expected).abs() < 1e-14, "got {}, expected {}", got, expected);
}

/// Case 6 (workup §4): constant input in one dimension → NaN
#[test]
fn kendall_tau_constant_x_is_nan() {
    let x = [1.0, 1.0, 1.0, 1.0];
    let y = [1.0, 2.0, 3.0, 4.0];
    let got = kendall_tau(&x, &y);
    assert!(got.is_nan(), "constant x: expected NaN, got {}", got);
}

/// Case 7 (workup §4): n < 2 → NaN (documented sentinel).
#[test]
fn kendall_tau_too_small_is_nan() {
    assert!(kendall_tau(&[], &[]).is_nan());
    assert!(kendall_tau(&[1.0], &[2.0]).is_nan());
}

/// Invariant (workup §8): τ(x, y) = τ(y, x). Symmetry.
#[test]
fn kendall_tau_symmetric() {
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let y = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0];
    let a = kendall_tau(&x, &y);
    let b = kendall_tau(&y, &x);
    assert!((a - b).abs() < 1e-15, "τ(x,y)={}, τ(y,x)={}", a, b);
}

/// Invariant (workup §8): τ(x+c, y) = τ(x, y). Shift invariance in x.
#[test]
fn kendall_tau_shift_invariant_x() {
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let y = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0];
    let a = kendall_tau(&x, &y);
    let x_shifted: Vec<f64> = x.iter().map(|v| v + 100.0).collect();
    let b = kendall_tau(&x_shifted, &y);
    assert!((a - b).abs() < 1e-15, "shift-invariance failed: {} vs {}", a, b);
}

/// Invariant (workup §8): τ(x*c, y) = τ(x, y) for c > 0. Scale invariance.
#[test]
fn kendall_tau_scale_invariant_positive() {
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let y = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0];
    let a = kendall_tau(&x, &y);
    let x_scaled: Vec<f64> = x.iter().map(|v| v * 42.0).collect();
    let b = kendall_tau(&x_scaled, &y);
    assert!((a - b).abs() < 1e-15, "scale-invariance failed: {} vs {}", a, b);
}

/// Invariant (workup §8): τ(−x, y) = −τ(x, y). Sign flip under negation of x.
#[test]
fn kendall_tau_sign_flip_under_negation() {
    let x = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let y = [2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0];
    let a = kendall_tau(&x, &y);
    let x_neg: Vec<f64> = x.iter().map(|v| -v).collect();
    let b = kendall_tau(&x_neg, &y);
    assert!((a + b).abs() < 1e-15, "negation failed: {} + {} = {}", a, b, a + b);
}

/// Range bound (workup §8): τ ∈ [−1, 1] for any input.
#[test]
fn kendall_tau_bounded() {
    use tambear::rng::{Xoshiro256, TamRng};
    let mut rng = Xoshiro256::new(12345);
    for _ in 0..50 {
        let x: Vec<f64> = (0..30).map(|_| rng.next_f64()).collect();
        let y: Vec<f64> = (0..30).map(|_| rng.next_f64()).collect();
        let t = kendall_tau(&x, &y);
        if !t.is_nan() {
            assert!(
                t >= -1.0 - 1e-14 && t <= 1.0 + 1e-14,
                "τ = {} out of bounds", t
            );
        }
    }
}
