//! Workup parity: `special_functions::erfc`
//!
//! Supporting test suite for
//! `docs/research/atomic-industrialization/erfc.md`
//!
//! **IMPORTANT**: This test suite documents a confirmed accuracy bug.
//! Tests prefixed `erfc_bug_*` document the current broken behavior.
//! Tests prefixed `erfc_*` assert correct behavior.
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
/// **Fix**: extended Taylor series region from |x| < 0.5 to |x| < 1.5.
/// Now: rel_err < 1e-15. Verified against mpmath 50dp.
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

/// COMPARISON: x=1.0 — CF barely within machine precision (1.4e-15).
/// This is the region where the bug transitions to acceptable accuracy.
#[test]
fn erfc_x1p0_is_near_machine_precision() {
    let got = erfc(1.0);
    let oracle = 0.15729920705028513_f64;
    let rel_err = (got - oracle).abs() / oracle;
    // At x=1.0, the CF converges (just barely) to ~1 ULP
    assert!(
        rel_err < 3e-15,
        "erfc(1.0): rel_err={:.2e}, expected < 3e-15", rel_err
    );
}
