//! Adversarial Wave 14 — Singularity-as-identity: the general bug class
//!
//! Aristotle's principle: the identity element must be reserved EXCLUSIVELY
//! for padding/neutral purposes. Using identity as an error sentinel conflates
//! two distinct mathematical concepts:
//!   - "this element does not affect the result" (identity)
//!   - "this element is undefined/degenerate" (error)
//!
//! In a parallel scan, the identity is invisible — returning identity for a
//! degenerate case means the error silently disappears.
//!
//! Targets:
//!   - log_sum_exp: all-NaN input returns -Inf (log-space identity) instead of NaN
//!   - matrix_exp: singular Padé denominator Q returns identity instead of NaN
//!   - mat_norm1: f64::max fold eats NaN, returns wrong norm
//!   - SarkkaMerge: catastrophic cancellation test (annotated, fires when implemented)
//!
//! All tests assert mathematical truth. Failures are bugs.

use tambear::linear_algebra::{Mat, matrix_exp};

// ═══════════════════════════════════════════════════════════════════════════
// log_sum_exp: singularity-as-identity
// ═══════════════════════════════════════════════════════════════════════════

/// log_sum_exp([NaN]) should return NaN (invalid input), not -Inf.
///
/// BUG: f64::max fold eats NaN when NaN is the first non-NEG_INFINITY element
/// it encounters. The fold starts at NEG_INFINITY and accumulates:
///   max(NEG_INFINITY, NaN) = NEG_INFINITY  (f64::max eats NaN on left)
/// So max = NEG_INFINITY (same as empty input), and the guard
/// `if max.is_infinite() { return max }` returns NEG_INFINITY.
///
/// NEG_INFINITY is the log-space identity (log(0) = "no probability mass").
/// A NaN input is invalid — it should not be treated as zero probability mass.
#[test]
fn log_sum_exp_single_nan_is_nan() {
    let result = tambear::numerical::log_sum_exp(&[f64::NAN]);
    assert!(result.is_nan(),
        "BUG: log_sum_exp([NaN]) should return NaN (invalid input), \
         got {} — NaN silently treated as -Inf (log-space identity)", result);
}

/// log_sum_exp([NaN, NaN]) should return NaN, not -Inf.
#[test]
fn log_sum_exp_all_nan_is_nan() {
    let result = tambear::numerical::log_sum_exp(&[f64::NAN, f64::NAN, f64::NAN]);
    assert!(result.is_nan(),
        "BUG: log_sum_exp([NaN, NaN, NaN]) should return NaN, \
         got {} — all-NaN returns -Inf (identity), masking invalid data",
        result);
}

/// log_sum_exp with NaN mixed in should propagate NaN (not skip it).
///
/// Mathematical truth: log(exp(1) + NaN + exp(2)) is undefined.
/// The current f64::max fold eats the NaN when computing max, but
/// the summation step then includes NaN in the exp computation,
/// which may or may not propagate depending on whether NaN is the
/// actual maximum.
#[test]
fn log_sum_exp_mixed_nan_propagates() {
    // NaN is NOT the maximum (max = 2.0, found after NaN is eaten by fold).
    // Then (NaN - 2.0).exp() = NaN enters the sum → sum is NaN → result is NaN.
    // This accidentally works! But only because NaN ends up in the summation.
    let result = tambear::numerical::log_sum_exp(&[1.0, f64::NAN, 2.0]);
    // This should be NaN but may accidentally return NaN via different path:
    assert!(result.is_nan(),
        "log_sum_exp([1, NaN, 2]) should be NaN, got {}", result);
}

/// log_sum_exp with NaN as the largest value (NaN would be max if detected).
///
/// BUG: f64::max(existing_max, NaN) = existing_max — NaN is eaten.
/// When NaN appears AFTER a real maximum, the fold skips it.
/// So log_sum_exp([2.0, NaN]) → max=2.0, then sum=(1.0 + NaN_exp) = NaN → NaN.
/// But log_sum_exp([NaN, 2.0]) → fold: max(NEG_INF, NaN)=NEG_INF, max(NEG_INF, 2.0)=2.0
/// → max=2.0 again → sum includes NaN → result is NaN.
/// Seems to work? Let's verify the ordering-invariance.
#[test]
fn log_sum_exp_nan_as_first_element_still_nan() {
    let r1 = tambear::numerical::log_sum_exp(&[f64::NAN, 2.0]);
    let r2 = tambear::numerical::log_sum_exp(&[2.0, f64::NAN]);
    // Both should be NaN
    assert!(r1.is_nan(),
        "log_sum_exp([NaN, 2.0]) should be NaN, got {}", r1);
    assert!(r2.is_nan(),
        "log_sum_exp([2.0, NaN]) should be NaN, got {}", r2);
}

/// log_sum_exp([]) should return -Inf (log of 0 — empty sum).
/// This is the CORRECT identity: empty sum in log space = log(0) = -Inf.
/// Document it here so the distinction is clear.
#[test]
fn log_sum_exp_empty_is_neg_inf() {
    let result = tambear::numerical::log_sum_exp(&[]);
    assert!(result.is_infinite() && result < 0.0,
        "log_sum_exp([]) should be -Inf (log-space identity), got {}", result);
}

/// log_sum_exp basic correctness: log(exp(1) + exp(2)) = log(e + e²).
#[test]
fn log_sum_exp_basic_correct() {
    let result = tambear::numerical::log_sum_exp(&[1.0, 2.0]);
    let expected = (std::f64::consts::E + std::f64::consts::E.powi(2)).ln();
    assert!((result - expected).abs() < 1e-12,
        "log_sum_exp([1,2]) = {} expected {}", result, expected);
}

/// log_sum_exp with all -Inf (all zero probabilities): should return -Inf.
/// This is the correct identity — all probabilities zero = zero total mass.
#[test]
fn log_sum_exp_all_neg_inf_is_neg_inf() {
    let result = tambear::numerical::log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]);
    assert!(result.is_infinite() && result < 0.0,
        "log_sum_exp([-Inf, -Inf]) should be -Inf, got {}", result);
}

/// log_sum_exp with +Inf: log(exp(Inf) + ...) = Inf.
#[test]
fn log_sum_exp_with_inf_is_inf() {
    let result = tambear::numerical::log_sum_exp(&[f64::INFINITY, 1.0, 2.0]);
    assert!(result.is_infinite() && result > 0.0,
        "log_sum_exp([Inf, 1, 2]) should be +Inf, got {}", result);
}

// ═══════════════════════════════════════════════════════════════════════════
// matrix_exp: singular-Q fallback returns identity
// ═══════════════════════════════════════════════════════════════════════════

/// matrix_exp with a highly defective nilpotent matrix.
/// A nilpotent matrix N has N^k = 0 for some k. exp(N) is exact and finite.
/// This is NOT a bug trigger — nilpotent matrices are well-behaved for matrix_exp.
#[test]
fn matrix_exp_nilpotent_is_finite() {
    // N = [[0, 1], [0, 0]]: N² = 0, exp(N) = I + N = [[1,1],[0,1]]
    let n_mat = Mat { rows: 2, cols: 2, data: vec![0.0, 1.0, 0.0, 0.0] };
    let e = matrix_exp(&n_mat);
    assert!((e.get(0, 0) - 1.0).abs() < 1e-12, "exp(N)[0,0] should be 1");
    assert!((e.get(0, 1) - 1.0).abs() < 1e-12, "exp(N)[0,1] should be 1");
    assert!((e.get(1, 0)).abs() < 1e-12,        "exp(N)[1,0] should be 0");
    assert!((e.get(1, 1) - 1.0).abs() < 1e-12, "exp(N)[1,1] should be 1");
}

/// matrix_exp singular-Q fallback: construct a pathological matrix where
/// the Padé denominator Q = V - U becomes singular.
///
/// This is mathematically possible when the Padé approximant degenerates —
/// typically for matrices with very special spectral structure. When Q is
/// singular, the current code returns eye (identity matrix) silently.
///
/// BUG class: the fallback should return a NaN-filled matrix, not identity.
/// Returning identity means matrix_exp of a pathological input looks like
/// exp(0) = I, which is numerically plausible but mathematically wrong.
///
/// Note: constructing an input that makes Q singular is hard analytically
/// (Padé denominators are numerically robust). This test instead verifies
/// the property that SHOULD hold: matrix_exp must not silently return identity
/// for ANY non-zero input, since exp(A) = I iff A is the zero matrix.
#[test]
fn matrix_exp_nonzero_input_is_not_identity() {
    // Any nonzero matrix A: exp(A) ≠ I (unless A is the zero matrix).
    // Test several nonzero inputs to catch the fallback.
    let inputs = vec![
        Mat { rows: 2, cols: 2, data: vec![1.0, 0.0, 0.0, 1.0] },  // 2I
        Mat { rows: 2, cols: 2, data: vec![0.0, 1.0, -1.0, 0.0] }, // rotation generator
        Mat { rows: 2, cols: 2, data: vec![1.0, 2.0, 3.0, 4.0] },  // general
    ];
    let eye_data = vec![1.0, 0.0, 0.0, 1.0];
    for a in &inputs {
        let e = matrix_exp(a);
        let is_identity = e.data.iter().zip(eye_data.iter())
            .all(|(&actual, &expected)| (actual - expected).abs() < 1e-10);
        assert!(!is_identity,
            "BUG: matrix_exp returned identity for nonzero input {:?} — \
             indicates singular-Q fallback triggered silently",
            a.data);
    }
}

/// The specific commit from wave 12: matrix_exp precision with Padé [13/13].
/// These tests previously failed with Padé [6/6]. Verify they now pass —
/// documenting that the upgrade happened and the precision bug is fixed.
#[test]
fn matrix_exp_identity_matrix_precision_fixed() {
    let eye = Mat::eye(2);
    let e = matrix_exp(&eye);
    let euler = std::f64::consts::E;
    // With Padé [13/13], should now achieve near-machine-precision.
    assert!((e.get(0, 0) - euler).abs() < 1e-12,
        "matrix_exp(I)[0,0] = {} expected e={}; Padé [13/13] should give <1e-12 error",
        e.get(0, 0), euler);
}

// ═══════════════════════════════════════════════════════════════════════════
// mat_norm1: f64::max NaN-eating in fold
// ═══════════════════════════════════════════════════════════════════════════

/// mat_norm1 internally uses fold(0.0, f64::max) over column sums.
/// If a matrix entry is NaN, the column sum containing it is NaN,
/// but fold(col_sum_so_far, NaN) = col_sum_so_far — NaN is eaten.
/// So mat_norm1 of a NaN-containing matrix returns a finite value.
///
/// This is observable: matrix_exp of a NaN-containing matrix uses the
/// (wrong, finite) norm to decide scaling s, then proceeds with Padé
/// on a NaN matrix. The test checks that matrix_exp(NaN matrix) produces
/// NaN output, not a spuriously finite result from wrong scaling.
#[test]
fn matrix_exp_nan_input_produces_nan_output() {
    let a = Mat { rows: 2, cols: 2, data: vec![1.0, f64::NAN, 0.0, 1.0] };
    let e = matrix_exp(&a);
    // With NaN in the input, at least one output should be NaN.
    let all_finite = e.data.iter().all(|v| v.is_finite());
    assert!(!all_finite,
        "matrix_exp with NaN input should produce NaN output — got all-finite {:?}",
        e.data);
}

/// mat_norm1 NaN-eating: directly test via matrix_exp scaling behavior.
/// A matrix with NaN has norm = NaN (mathematically). If mat_norm1 returns
/// a finite value (NaN eaten), the scaling factor s is wrong — possibly 0,
/// meaning no scaling applied to a NaN matrix.
///
/// This doesn't directly test mat_norm1 (it's private), but the observable
/// consequence is that matrix_exp produces the wrong number of squarings.
#[test]
fn matrix_exp_large_nan_matrix_scaling_observable() {
    // A matrix with large entries: norm should be ~1e10, needing s≈34 squarings.
    // If NaN is mixed in, mat_norm1 may return the large finite value (NaN in one
    // column doesn't affect other columns' sums in the max fold).
    let a = Mat { rows: 2, cols: 2, data: vec![1e10, 0.0, f64::NAN, 1e10] };
    let e = matrix_exp(&a);
    // The result should contain NaN because the input does.
    let has_nan = e.data.iter().any(|v| v.is_nan());
    assert!(has_nan,
        "matrix_exp with NaN in off-diagonal should propagate NaN to output, \
         got all-finite result {:?} — mat_norm1 NaN-eating may have caused \
         wrong scaling, but NaN should still reach output via Padé computation",
        e.data);
}

// ═══════════════════════════════════════════════════════════════════════════
// SarkkaMerge catastrophic cancellation (annotated — fires when implemented)
// ═══════════════════════════════════════════════════════════════════════════

/// Aristotle's specific prediction for SarkkaMerge:
///
/// The combine rule for the Sarkka (2021) parallel Kalman filter has a
/// correction term: η_ab = η_b - J_b · b_a
///
/// When J_b = 1e8, b_a = 1e8, η_b = 1e16 + 1.0:
///   Mathematically: η_ab = (1e16 + 1) - 1e16 = 1.0
///   In f64: 1e16 + 1.0 = 1e16 (precision loss!), then 1e16 - 1e16 = 0.0
///
/// The result is 0.0 instead of 1.0 — catastrophic cancellation.
///
/// This test is currently a NO-OP (marked should_panic to make it fail
/// if someone accidentally passes it without implementing SarkkaMerge).
/// When SarkkaMerge is implemented, REMOVE the should_panic and test
/// the actual implementation.
///
/// To test: call sarkka_combine(element_a, element_b) where element_b
/// has J_b = [[1e8, 0], [0, 1e8]] (2x2 case) and b_a = [1e8, 0],
/// η_b = [1e16 + 1, 0], and check that η_ab[0] = 1.0 (not 0.0).
#[test]
#[should_panic(expected = "SarkkaMerge not yet implemented")]
fn sarkka_merge_catastrophic_cancellation_eta_correction() {
    // Remove #[should_panic] when SarkkaMerge is implemented.
    // Replace the panic below with the actual test:
    //
    // let elem_a = SarkkaElement { a: Mat::eye(2), b: vec![1e8, 0.0], ... };
    // let elem_b = SarkkaElement {
    //     j: Mat { data: vec![1e8, 0.0, 0.0, 1e8], rows: 2, cols: 2 },
    //     b: vec![1e8, 0.0],
    //     eta: vec![1e16 + 1.0, 0.0],
    //     ...
    // };
    // let combined = sarkka_combine(&elem_a, &elem_b);
    // assert!((combined.eta[0] - 1.0).abs() < 1.0,
    //     "SarkkaMerge catastrophic cancellation: eta_ab[0] = {} expected 1.0; \
    //      use compensated summation (Kahan) for the eta correction term",
    //     combined.eta[0]);
    panic!("SarkkaMerge not yet implemented");
}

// ═══════════════════════════════════════════════════════════════════════════
// The general principle: identity ≠ error sentinel
// ═══════════════════════════════════════════════════════════════════════════

/// Aristotle's general principle: these two conditions must be DISTINGUISHABLE
/// by the scan infrastructure:
///   1. "This element is neutral" → return identity → scan skips it correctly
///   2. "This element is degenerate" → return NaN/error → scan reports failure
///
/// Test: a scan over [valid, NaN, valid] must produce NaN, not act as if
/// the NaN element was the identity (neutral). We test this with the
/// log-space parallel scan (logadd binary operation on log-probabilities).
///
/// The logadd merge for parallel scan is: merge(a, b) = log(exp(a) + exp(b))
/// = max(a,b) + log1p(exp(-|a-b|))
///
/// This operation must propagate NaN, not swallow it.
fn logadd_merge(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { return f64::NAN; }
    if a == f64::NEG_INFINITY { return b; }
    if b == f64::NEG_INFINITY { return a; }
    let max = a.max(b);
    let min = a.min(b);
    max + (1.0 + (min - max).exp()).ln()
}

#[test]
fn logadd_merge_identity_correct() {
    // NEG_INFINITY is the identity: merge(-Inf, x) = x
    let x = 1.5_f64;
    assert!((logadd_merge(f64::NEG_INFINITY, x) - x).abs() < 1e-14,
        "logadd: -Inf is left identity");
    assert!((logadd_merge(x, f64::NEG_INFINITY) - x).abs() < 1e-14,
        "logadd: -Inf is right identity");
}

#[test]
fn logadd_merge_nan_propagates() {
    // If NaN is treated as identity (-Inf), merge(NaN, x) would return x.
    // The correct behavior: merge(NaN, x) = NaN (error propagates).
    assert!(logadd_merge(f64::NAN, 1.5).is_nan(),
        "logadd(NaN, x) should be NaN — NaN ≠ identity (-Inf)");
    assert!(logadd_merge(1.5, f64::NAN).is_nan(),
        "logadd(x, NaN) should be NaN — NaN ≠ identity (-Inf)");
}

#[test]
fn logadd_merge_sequential_scan_nan_at_position_2() {
    // Sequential logadd scan over [0.0, 1.0, NaN, 2.0]
    // Prefix sums: [0.0, log(e+e²), NaN, NaN]
    let data = [0.0_f64, 1.0, f64::NAN, 2.0];
    let mut acc = f64::NEG_INFINITY;
    let mut scan = Vec::new();
    for &x in &data {
        acc = logadd_merge(acc, x);
        scan.push(acc);
    }
    assert!(!scan[0].is_nan(), "scan[0] should be finite: {}", scan[0]);
    assert!(!scan[1].is_nan(), "scan[1] should be finite: {}", scan[1]);
    assert!(scan[2].is_nan(),  "scan[2] should be NaN (NaN input): {}", scan[2]);
    assert!(scan[3].is_nan(),  "scan[3] should be NaN (propagated): {}", scan[3]);
}

/// The reference implementation of logadd that DOES propagate NaN correctly.
/// This establishes the baseline for when the GPU parallel logadd is built.
#[test]
fn logadd_merge_associativity() {
    // For well-defined values, logadd must be associative.
    let a = 1.0_f64;
    let b = 2.0_f64;
    let c = 0.5_f64;
    let left  = logadd_merge(logadd_merge(a, b), c);
    let right = logadd_merge(a, logadd_merge(b, c));
    assert!((left - right).abs() < 1e-12,
        "logadd associativity failure: ({} ⊕ {}) ⊕ {} = {} vs {} ⊕ ({} ⊕ {}) = {}",
        a, b, c, left, a, b, c, right);
}
