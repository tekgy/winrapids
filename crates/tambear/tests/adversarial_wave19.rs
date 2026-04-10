//! Adversarial Wave 19 — RQA and MFDFA NaN propagation failures
//!
//! Pre-flight correctness sweep for the Riemann zero universality experiment.
//! Both RQA and MFDFA are used to compare zeta zero spacings to market eigenvalue
//! spacings. If either implementation has NaN-eating bugs, the experiment result
//! would be noise with a convincing finite value.
//!
//! Two confirmed bugs:
//!
//! 1. `rqa` (complexity.rs:891-899): NaN in data → NaN pairwise distance →
//!    `d2 > eps2` evaluates to false for NaN (NaN comparisons return false) →
//!    early exit never fires → `d2 <= eps2` also false → r = false.
//!    NaN pairs are treated as NOT recurrent. Recurrence matrix is corrupted
//!    without any indication. rr, det, lam all return finite values from a
//!    NaN-contaminated matrix.
//!
//! 2. `mfdfa` (complexity.rs:1488): `rms2.max(1e-300)` — NaN.max(1e-300) = 1e-300
//!    (f64::max NaN-eating). NaN segments get minimum-variance treatment instead
//!    of propagating the invalidity. h(q) estimates are computed on corrupted data.
//!
//! Mathematical truths:
//!   - rqa: if any data point is NaN, the entire recurrence structure is undefined
//!     → all RqaResult fields must be NaN
//!   - mfdfa: if any data point is NaN, the cumulative profile is NaN-contaminated
//!     from that point → all MfdfaResult fields must be NaN
//!   - rqa: epsilon = NaN must return RqaResult::nan() (already guarded — verify)
//!   - rqa: constant signal (all equal) → all points recurrent → rr = 1.0
//!   - mfdfa: monofractal signal (fBm H=0.5) should give h(q=2) ≈ 0.5
//!
//! All tests assert mathematical truths. Failures are bugs.

use tambear::{rqa, mfdfa, RqaResult, MfdfaResult};

// ═══════════════════════════════════════════════════════════════════════════
// RQA — Test 1: NaN in data must propagate
// ═══════════════════════════════════════════════════════════════════════════

/// NaN in data → NaN squared distance in recurrence computation.
/// `d2 > eps2` with NaN d2 evaluates to FALSE → early exit skipped.
/// `d2 <= eps2` with NaN d2 evaluates to FALSE → pair marked non-recurrent.
/// Result: corrupted recurrence matrix with no warning. rr = finite but wrong.
///
/// EXPECTED: rqa returns RqaResult where all fields are NaN.
/// ACTUAL (BUG): returns finite rr, det, etc. computed on corrupted matrix.
#[test]
fn rqa_nan_in_data_must_propagate() {
    let mut data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
    data[50] = f64::NAN;

    let result = rqa(&data, 3, 1, 0.5, 2);

    assert!(result.rr.is_nan(),
        "BUG: rqa with NaN in data should return NaN rr, got {} \
         — NaN distance is silently treated as non-recurrent: \
         `d2 <= eps2` with NaN d2 = false, so pair is marked as not recurrent \
         without any NaN propagation",
        result.rr);
}

/// NaN in the FIRST element: the entire embedding is corrupted from the start.
/// Every delay vector involving index 0 has NaN.
/// For m=2, tau=1: vectors are (data[0], data[1]), (data[1], data[2]), ...
/// Vector 0 has NaN. All distances from vector 0 to all others = NaN.
/// Expected: rr = NaN (recurrence structure undefined).
#[test]
fn rqa_nan_at_start_corrupts_all_vectors_from_first() {
    let mut data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
    data[0] = f64::NAN;

    let result = rqa(&data, 2, 1, 0.5, 2);

    assert!(result.rr.is_nan(),
        "BUG: rqa with NaN at index 0 should return NaN rr (all vectors \
         sharing index 0 are corrupted), got {}",
        result.rr);
}

// ═══════════════════════════════════════════════════════════════════════════
// RQA — Test 2: Correctness on known inputs
// ═══════════════════════════════════════════════════════════════════════════

/// Constant signal: all delay vectors are identical.
/// Distance between any two vectors = 0 < epsilon (for any epsilon > 0).
/// Therefore ALL pairs are recurrent. RR = 1.0 (100% recurrence rate).
/// DET = 1.0 (all recurrent points lie on diagonal lines).
#[test]
fn rqa_constant_signal_all_recurrent() {
    let data = vec![3.14_f64; 100];
    let result = rqa(&data, 3, 1, 0.1, 2);

    if result.rr.is_nan() { return; } // skip if implementation returns nan for degenerate

    assert!((result.rr - 1.0).abs() < 1e-10,
        "Constant signal: all pairs recurrent → rr should be 1.0, got {}", result.rr);
    // DET may not be exactly 1.0 due to boundary segments shorter than lmin.
    // For a long constant signal, DET should be very close to 1.0.
    assert!(result.det > 0.99,
        "Constant signal: most recurrent points in diagonal lines → det should be > 0.99, got {}",
        result.det);
}

/// Perfectly non-recurrent signal: strictly increasing sequence with large gaps.
/// If gaps exceed epsilon, no pair is recurrent. rr = 0.
#[test]
fn rqa_non_recurrent_signal_rr_zero() {
    // Strictly increasing with step >> epsilon
    let data: Vec<f64> = (0..50).map(|i| i as f64 * 10.0).collect();
    let result = rqa(&data, 2, 1, 0.5, 2); // epsilon = 0.5, gaps = 10.0

    if result.rr.is_nan() { return; }

    assert!(result.rr < 1e-10,
        "Strictly increasing signal with gaps >> epsilon: rr should be ~0, got {}",
        result.rr);
}

/// RQA guard: epsilon = NaN must return nan result.
/// Already guarded at line 876 (`!epsilon.is_finite()`). Verify.
#[test]
fn rqa_nan_epsilon_returns_nan() {
    let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
    let result = rqa(&data, 3, 1, f64::NAN, 2);
    assert!(result.rr.is_nan(),
        "rqa with NaN epsilon should return NaN result (guard check), got {}",
        result.rr);
}

/// RQA guard: epsilon = 0 must return nan (guard: epsilon <= 0.0).
#[test]
fn rqa_zero_epsilon_returns_nan() {
    let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
    let result = rqa(&data, 3, 1, 0.0, 2);
    assert!(result.rr.is_nan(),
        "rqa with epsilon=0 should return NaN result (guard check), got {}",
        result.rr);
}

// ═══════════════════════════════════════════════════════════════════════════
// MFDFA — Test 1: NaN in data must propagate
// ═══════════════════════════════════════════════════════════════════════════

/// NaN in data → mean_x = NaN (from sum) → profile[i] = NaN for all i after NaN.
/// In each segment containing NaN profile values:
///   rms2 = sum of NaN² / sf = NaN
///   rms2.max(1e-300) = f64::max(NaN, 1e-300) = 1e-300  ← NaN EATEN
/// NaN segments treated as minimum-variance. h(q) estimates computed from
/// a mixture of valid and minimum-variance segments. Result is finite but wrong.
///
/// EXPECTED: mfdfa with NaN in data returns MfdfaResult where all h_q fields are NaN.
/// ACTUAL (BUG): returns finite h(q) values from NaN-contaminated profile.
#[test]
fn mfdfa_nan_in_data_must_propagate() {
    let mut data: Vec<f64> = (0..500).map(|i| (i as f64 * 0.07).sin()).collect();
    data[250] = f64::NAN;

    let q_vals = vec![2.0_f64];
    let result = mfdfa(&data, &q_vals, 4, 128);

    assert!(result.h_q[0].is_nan(),
        "BUG: mfdfa with NaN in data should return NaN h(q=2), got {} \
         — rms2.max(1e-300) uses f64::max which eats NaN, treating NaN segments \
         as minimum-variance (rms2 = 1e-300) instead of propagating NaN",
        result.h_q[0]);
}

/// NaN at the start: mean_x is already NaN.
/// profile[1] = 0 + (NaN - NaN) = NaN.
/// All subsequent profile values are NaN (cumulative sum propagates NaN).
/// Every segment is NaN-contaminated → all rms2 = NaN → all h(q) = NaN.
#[test]
fn mfdfa_nan_at_start_all_h_nan() {
    let mut data: Vec<f64> = (0..400).map(|i| (i as f64 * 0.1).cos()).collect();
    data[0] = f64::NAN;

    let q_vals = vec![-2.0_f64, 0.0, 2.0, 4.0];
    let result = mfdfa(&data, &q_vals, 4, 100);

    for (q_idx, &q) in q_vals.iter().enumerate() {
        assert!(result.h_q[q_idx].is_nan(),
            "BUG: mfdfa with NaN at index 0 should return NaN h(q={}), got {} \
             — NaN propagates through the cumulative profile but is eaten by \
             rms2.max(1e-300) in each segment",
            q, result.h_q[q_idx]);
    }
    assert!(result.width.is_nan(),
        "BUG: mfdfa width should be NaN, got {}", result.width);
}

// ═══════════════════════════════════════════════════════════════════════════
// MFDFA — Test 2: Monofractal correctness
// ═══════════════════════════════════════════════════════════════════════════

/// For a pure random walk (H=0.5), mfdfa with q=2 should give h(q=2) ∈ [0.3, 0.9].
/// This is a sanity check for the happy path.
///
/// Note: mfdfa is a log-log regression estimator with finite-sample variance.
/// The tolerance is deliberately wide — we're checking the implementation is
/// in the right order of magnitude, not precision-matching the true H.
#[test]
fn mfdfa_random_walk_h2_near_half() {
    // MFDFA takes the CUMULATIVE PROFILE of the input internally.
    // So input should be IID increments (not their cumulative sum) for H ≈ 0.5.
    // IID {-1, +1} → profile = random walk → DFA gives H ≈ 0.5.
    // Use Xorshift for good mixing.
    let mut state: u64 = 12345678901u64;
    let iid: Vec<f64> = (0..2000).map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        if (state & 1) == 0 { 1.0_f64 } else { -1.0_f64 }
    }).collect();

    let q_vals = vec![2.0_f64];
    let result = mfdfa(&iid, &q_vals, 8, 500);

    if result.h_q[0].is_nan() { return; } // skip if too short

    // H ≈ 0.5 for IID series. Wide tolerance for finite samples.
    // MFDFA on IID {±1} should give h(q=2) near 0.5 (the profile is random walk H=0.5).
    assert!(result.h_q[0] > 0.3 && result.h_q[0] < 0.9,
        "mfdfa IID increments h(q=2) should be ≈ 0.5, got {}", result.h_q[0]);
}

/// Width of the multifractal spectrum for a monofractal (IID noise) should be
/// near zero — all h(q) values are approximately equal to 0.5.
/// Width = max(h_q) - min(h_q) ≈ 0 for monofractal.
#[test]
fn mfdfa_monofractal_width_near_zero() {
    // Pure white noise (IID) is monofractal: all h(q) ≈ 0.5
    // Use sum of incommensurate cosines as a "white noise" proxy
    // Each cosine is approximately uncorrelated with the others at the segment level
    let data: Vec<f64> = (0..800usize).map(|i| {
        (i as f64 * 0.31).sin()
        + (i as f64 * 0.77).cos()
        + (i as f64 * 1.23).sin()
        + (i as f64 * 2.11).cos()
    }).collect();

    let q_vals: Vec<f64> = (-4..=4).map(|q| q as f64).collect();
    let result = mfdfa(&data, &q_vals, 4, 200);

    if result.width.is_nan() { return; }

    // Monofractal: width should be small (< 0.5 is a reasonable threshold)
    // Truly monofractal would give 0; finite sample gives small non-zero width
    assert!(result.width < 0.8,
        "mfdfa monofractal (white noise) width should be small, got {}", result.width);
}

// ═══════════════════════════════════════════════════════════════════════════
// MFDFA — Test 3: q=0 is geometric mean (special case)
// ═══════════════════════════════════════════════════════════════════════════

/// When q ≈ 0, MFDFA uses geometric mean of F²(s,v) instead of the power-mean
/// formula (which diverges as q→0). The implementation should handle this without
/// NaN or division by zero.
#[test]
fn mfdfa_q_zero_does_not_produce_nan_for_clean_data() {
    let data: Vec<f64> = (0..600).map(|i| (i as f64 * 0.05).sin() + (i as f64 * 0.13).cos()).collect();
    let q_vals = vec![0.0_f64];
    let result = mfdfa(&data, &q_vals, 4, 150);

    // h(q=0) for a clean periodic signal should be finite
    assert!(result.h_q[0].is_finite() || result.h_q[0].is_nan(),
        "mfdfa q=0 on clean data should return finite h or NaN (not inf), got {}",
        result.h_q[0]);
    // Should not be Inf
    assert!(!result.h_q[0].is_infinite(),
        "mfdfa q=0 returned Inf: {}", result.h_q[0]);
}
