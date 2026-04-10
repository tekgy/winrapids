//! Workup parity: `special_functions::erfc`
//!
//! Supporting test suite for
//! `docs/research/atomic-industrialization/erfc.md`
//!
//! Bug history:
//! - Original boundary |x|<0.5: CF not converged in 200 iters at x=0.5 → 8e-9 error.
//!   Fix: extended Taylor to |x|<1.5 (2026-04-10).
//! - Boundary |x|<1.5: Taylor accumulates 82 ULP error at x=1.386 (e.g. Phi(-1.96)=62 ULP).
//!   Fix: reduced Taylor boundary to |x|<1.0. CF converges in ≤188 iters above x=1.0.
//! Current accuracy: ≤5 ULP (Taylor), ≤21 ULP (CF), max observed 14 ULP at x=-6.
//!
//! Oracle values computed via mpmath at 50 decimal digits.
//! All relative errors verified against the mpmath reference.

use tambear::special_functions::erfc;

// ─── Section 5.1: oracle cases — correct region ───────────────────────────────

/// Case 1: erfc(0) = 1 exactly.
#[test]
fn erfc_zero_is_one() {
    assert_eq!(erfc(0.0), 1.0);
}

/// Case 2: Taylor region x=0.1
/// mpmath 50dp: 0.887537083981715...
#[test]
fn erfc_point1_matches_oracle() {
    let got = erfc(0.1);
    let expected = 0.887537083981715_f64;
    let rel_err = (got - expected).abs() / expected;
    assert!(
        rel_err < 1e-14,
        "x=0.1: got={}, expected={}, rel_err={:.2e}", got, expected, rel_err
    );
}

/// Case 3: Taylor region x=0.25
/// mpmath 50dp: 0.7236736098317631
#[test]
fn erfc_quarter_matches_oracle() {
    let got = erfc(0.25);
    let expected = 0.7236736098317631_f64;
    let rel_err = (got - expected).abs() / expected;
    assert!(
        rel_err < 1e-14,
        "x=0.25: got={}, expected={}, rel_err={:.2e}", got, expected, rel_err
    );
}

/// Case 4: Taylor region x=0.3
/// mpmath 50dp: 0.6713732405408726
#[test]
fn erfc_0p3_matches_oracle() {
    let got = erfc(0.3);
    let expected = 0.6713732405408726_f64;
    let rel_err = (got - expected).abs() / expected;
    assert!(
        rel_err < 1e-14,
        "x=0.3: got={}, expected={}, rel_err={:.2e}", got, expected, rel_err
    );
}

/// Case 7: x=1.0 — CF region, barely within machine precision.
/// mpmath 50dp: 0.15729920705028513
#[test]
fn erfc_one_matches_oracle() {
    let got = erfc(1.0);
    let expected = 0.15729920705028513_f64;
    let rel_err = (got - expected).abs() / expected;
    assert!(
        rel_err < 5e-15,
        "x=1.0: got={}, expected={}, rel_err={:.2e}", got, expected, rel_err
    );
}

/// Case 8: x=1.5 — CF region, good accuracy.
/// mpmath 50dp: 0.033894853524689274
#[test]
fn erfc_1p5_matches_oracle() {
    let got = erfc(1.5);
    let expected = 0.033894853524689274_f64;
    let rel_err = (got - expected).abs() / expected;
    assert!(
        rel_err < 2e-14,
        "x=1.5: got={}, expected={}, rel_err={:.2e}", got, expected, rel_err
    );
}

/// Case 9: x=2.0
/// mpmath 50dp: 0.004677734981047266
#[test]
fn erfc_two_matches_oracle() {
    let got = erfc(2.0);
    let expected = 0.004677734981047266_f64;
    let rel_err = (got - expected).abs() / expected;
    assert!(
        rel_err < 2e-14,
        "x=2.0: got={}, expected={}, rel_err={:.2e}", got, expected, rel_err
    );
}

/// Case 10: x=3.0
/// mpmath 50dp: 2.209049699858544e-5
#[test]
fn erfc_three_matches_oracle() {
    let got = erfc(3.0);
    let expected = 2.209049699858544e-5_f64;
    let rel_err = (got - expected).abs() / expected;
    assert!(
        rel_err < 2e-14,
        "x=3.0: got={}, expected={}, rel_err={:.2e}", got, expected, rel_err
    );
}

/// Case 11: x=5.0 — deep tail.
/// mpmath 50dp: 1.537459794428035e-12
#[test]
fn erfc_five_matches_oracle() {
    let got = erfc(5.0);
    let expected = 1.537459794428035e-12_f64;
    let rel_err = (got - expected).abs() / expected;
    assert!(
        rel_err < 1e-12,
        "x=5.0: got={:.15e}, expected={:.15e}, rel_err={:.2e}",
        got, expected, rel_err
    );
}

/// Case 12: x=-1.0 — negative input.
/// mpmath 50dp: 1.8427007929497148
#[test]
fn erfc_neg_one_matches_oracle() {
    let got = erfc(-1.0);
    let expected = 1.8427007929497148_f64;
    let rel_err = (got - expected).abs() / expected;
    assert!(
        rel_err < 5e-15,
        "x=-1.0: got={}, expected={}, rel_err={:.2e}", got, expected, rel_err
    );
}

/// Case 13: x=-3.0 — negative, near 2.
/// mpmath 50dp: 1.9999779095030015
#[test]
fn erfc_neg_three_matches_oracle() {
    let got = erfc(-3.0);
    let expected = 1.9999779095030015_f64;
    let rel_err = (got - expected).abs() / expected;
    assert!(
        rel_err < 2e-14,
        "x=-3.0: got={}, expected={}, rel_err={:.2e}", got, expected, rel_err
    );
}

// ─── Section 4: edge cases ───────────────────────────────────────────────────

/// NaN input → NaN.
#[test]
fn erfc_nan_input_is_nan() {
    assert!(erfc(f64::NAN).is_nan(), "erfc(NaN) should be NaN");
}

/// x > 27 → 0.0 (subnormal flush).
#[test]
fn erfc_large_positive_flushes_to_zero() {
    let got = erfc(28.0);
    assert!(
        got == 0.0 || got < 1e-300,
        "erfc(28) should be ~0, got {}", got
    );
}

/// x < -27 → 2.0 (reflected flush).
#[test]
fn erfc_large_negative_flushes_to_two() {
    let got = erfc(-28.0);
    assert!(
        (got - 2.0).abs() < 1e-14,
        "erfc(-28) should be ~2, got {}", got
    );
}

// ─── Section 8: invariants ───────────────────────────────────────────────────

/// Invariant: erfc(-x) = 2 - erfc(x) for all finite x.
#[test]
fn erfc_reflection_symmetry() {
    let test_xs = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
    for &x in &test_xs {
        let pos = erfc(x);
        let neg = erfc(-x);
        let sum = pos + neg;
        assert!(
            (sum - 2.0).abs() < 1e-13,
            "symmetry: erfc({}) + erfc(-{}) = {} (expected 2.0, diff={})",
            x, x, sum, (sum - 2.0).abs()
        );
    }
}

/// Range: erfc(x) ∈ [0, 2] for all finite inputs.
///
/// Note: the function returns exactly 0.0 for x > 27 and exactly 2.0 for
/// x < -27 (subnormal flush). For inputs in [-27, 27] the value is in (0, 2).
#[test]
fn erfc_range_is_zero_to_two() {
    use tambear::rng::{Xoshiro256, TamRng};
    let mut rng = Xoshiro256::new(77777);
    for _ in 0..100 {
        let x = (rng.next_f64() - 0.5) * 20.0; // range [-10, 10]
        let e = erfc(x);
        assert!(
            e >= 0.0 && e <= 2.0,
            "erfc({}) = {} out of range [0, 2]", x, e
        );
    }
}

// ─── BUG DOCUMENTATION (workup §5.2 / §10) ───────────────────────────────────

/// Case 5: x=0.5 — formerly broken (CF failed to converge), now fixed.
///
/// **Historical note (workup §5.2)**: Before 2026-04-10, tambear returned
/// 0.47950011817484517 (rel_err=8.37e-9). Root cause: the CF was switched on
/// at |x|=0.5 but required ~1000 iterations to converge there; the 200-iteration
/// budget gave only ~8e-9 accuracy.
///
/// **Fix history**: Taylor boundary 0.5→1.5 (2026-04-10 first fix), then 1.5→1.0 (correct fix).
/// At |x|<1.0, Taylor ≤5 ULP. At |x|≥1.0, CF ≤21 ULP.
#[test]
fn erfc_bug_boundary_x_0p5_accuracy_degraded() {
    let got = erfc(0.5);
    let oracle = 0.4795001221869535_f64; // mpmath 50dp
    let rel_err = (got - oracle).abs() / oracle;

    // FIXED: now within < 1e-15, not 8.37e-9 as before.
    assert!(
        rel_err < 2e-15,
        "erfc(0.5): rel_err={:.2e} (oracle={}, got={}) — expected < 2e-15 after fix",
        rel_err, oracle, got
    );
}

/// Case 6: x=0.7 — formerly had 4e-12 relative error, now fixed.
/// Expected (mpmath): 0.32219880616258156
#[test]
fn erfc_bug_boundary_x_0p7_accuracy_degraded() {
    let got = erfc(0.7);
    let oracle = 0.32219880616258156_f64; // mpmath 50dp
    let rel_err = (got - oracle).abs() / oracle;

    // FIXED: now within < 1e-15.
    assert!(
        rel_err < 2e-15,
        "erfc(0.7): rel_err={:.2e} (oracle={}, got={}) — expected < 2e-15 after fix",
        rel_err, oracle, got
    );
}

/// x=1.0 — boundary between Taylor (|x|<1.0) and CF (|x|≥1.0) regions.
/// CF converges in 188 iterations here, giving ~8 ULP accuracy.
#[test]
fn erfc_x1p0_is_near_machine_precision() {
    let got = erfc(1.0);
    let oracle = 0.15729920705028513_f64;
    let rel_err = (got - oracle).abs() / oracle;
    assert!(
        rel_err < 2e-14,
        "erfc(1.0): rel_err={:.2e}, expected < 2e-14 (CF region, ≤21 ULP)", rel_err
    );
}

/// x=1.386 (erfc argument for normal_cdf(-1.96)) — formerly 82 ULP with Taylor,
/// now ≤10 ULP via CF. This is the critical value for 95% confidence intervals.
#[test]
fn erfc_x1p386_ncdf_critical_value() {
    // erfc(1.96/sqrt(2)) = erfc(1.38592929...)
    let arg = 1.96_f64 / std::f64::consts::SQRT_2;
    let got = erfc(arg);
    let oracle = 0.04999579029644084_f64; // mpmath 50dp
    let rel_err = (got - oracle).abs() / oracle;
    assert!(
        rel_err < 5e-15,
        "erfc(1.386): rel_err={:.2e}, expected < 5e-15 (critical for normal_cdf(-1.96))",
        rel_err
    );
}

/// x=1.2 — formerly in [1.0, 1.5) Taylor range where error was up to 82 ULP.
/// Now in CF region, should be ≤21 ULP.
#[test]
fn erfc_x1p2_cf_region_accuracy() {
    let got = erfc(1.2);
    let oracle = 0.08968602177036462_f64; // mpmath 50dp
    let rel_err = (got - oracle).abs() / oracle;
    assert!(
        rel_err < 2e-14,
        "erfc(1.2): rel_err={:.2e}, expected < 2e-14 (CF region)", rel_err
    );
}
