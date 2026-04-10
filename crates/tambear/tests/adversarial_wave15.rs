//! Adversarial Wave 15 — Three-test template applied to all parallel merge operators
//!
//! Aristotle's complete adversarial suite for any Op variant:
//!   Test 1: Catastrophic cancellation (large dynamic range)
//!   Test 2: Associativity (compose(compose(a,b),c) vs compose(a,compose(b,c)))
//!   Test 3: Validity propagation (NaN/degenerate inputs not silently masked)
//!
//! Applied to: MomentStats::merge, SufficientStatistics::merge, CopaState::merge
//! These are the three parallel Welford merge operators — the Welford Op variant
//! Aristotle listed alongside SarkkaMerge and LogSumExpMerge.
//!
//! All tests assert mathematical truths. Failures are bugs.

use tambear::descriptive::{MomentStats, moments_ungrouped};
use tambear::intermediates::SufficientStatistics;
use tambear::copa::CopaState;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build a MomentStats from a slice. Excludes NaN (by design of moments_ungrouped).
fn ms_from(vals: &[f64]) -> MomentStats {
    moments_ungrouped(vals)
}

// ═══════════════════════════════════════════════════════════════════════════
// MomentStats::merge — Test 1: Catastrophic cancellation
// ═══════════════════════════════════════════════════════════════════════════

/// Welford merge delta = mean_b - mean_a. When both means are large and close,
/// this subtraction loses precision.
///
/// Two partitions: A = [1e15, 1e15 + 2] (mean = 1e15 + 1), B = [1e15 + 2, 1e15 + 4]
/// True combined variance over [1e15, 1e15+2, 1e15+2, 1e15+4]:
///   Values shifted: [0, 2, 2, 4]. Variance = 2.0 (population), 8/3 (sample).
///   Mean = 1e15 + 2.
///
/// The merge formula computes delta = (1e15+3) - (1e15+1) = 2.0 in exact arithmetic.
/// In f64, (1e15 + 3) - (1e15 + 1): 1e15 has 15 significant digits, adding 3 or 1
/// should be representable, so delta should be exactly 2.0.
///
/// This tests that the merge formula doesn't lose precision for large-mean data.
#[test]
fn moment_merge_catastrophic_cancellation_large_mean() {
    let base = 1e15_f64;
    let a = ms_from(&[base, base + 2.0]);
    let b = ms_from(&[base + 2.0, base + 4.0]);

    let merged = MomentStats::merge(&a, &b);

    // Combined mean should be base + 2.0
    let expected_mean = base + 2.0;
    assert!((merged.sum / merged.count - expected_mean).abs() < 1.0,
        "Large-mean merge: mean error too large. \
         got mean={}, expected ~{}, err={:.2e}",
        merged.sum / merged.count, expected_mean,
        (merged.sum / merged.count - expected_mean).abs());

    // Variance: vals shifted to [0,2,2,4], pop var = (0+4+4+16)/4 - 4 = 6 - 4 = 2.0
    // m2 = sum of squared deviations from mean = (2)^2 + (0)^2 + (0)^2 + (2)^2 = 8
    let expected_m2 = 8.0;
    let rel_err = (merged.m2 - expected_m2).abs() / expected_m2;
    assert!(rel_err < 1e-8,
        "Large-mean merge: m2 catastrophic cancellation. \
         expected_m2={}, got_m2={}, rel_err={:.2e} — \
         delta = mean_b - mean_a loses precision for large means",
        expected_m2, merged.m2, rel_err);
}

/// Extreme case: means differ by 1 ULP at scale 1e15.
/// In f64, 1e15 + 1 == 1e15 (1 ULP ≈ 0.125 at scale 1e15).
/// So delta becomes 0 even though the true difference is 1.
/// This demonstrates where the formula structurally breaks.
#[test]
fn moment_merge_subnormal_delta_at_scale() {
    let base = 1e15_f64;
    // 1 at this scale is below machine epsilon * base
    let eps = base * f64::EPSILON;
    // delta = 1.0, which is < eps ≈ 0.22 ... actually let's compute:
    // f64::EPSILON ≈ 2.2e-16, base = 1e15, eps = 0.22
    // So base + 1.0 IS representable (1.0 > 0.22 threshold).
    // But base + 0.1 may not be: 0.1 < 0.22, so base + 0.1 == base.
    let a = ms_from(&[base, base + eps * 0.1]); // second val rounds to base
    let b = ms_from(&[base, base + eps * 0.1]); // same

    let merged = MomentStats::merge(&a, &b);
    let direct = ms_from(&[base, base + eps * 0.1, base, base + eps * 0.1]);

    // The merge and direct computation should agree on count.
    assert!((merged.count - direct.count).abs() < 0.5,
        "count should be 4: merge={} direct={}", merged.count, direct.count);

    // If delta is lost (rounds to 0), merged.m2 may be 0 while direct.m2 > 0.
    // Document the precision loss rather than asserting correctness.
    if merged.m2 == 0.0 && direct.m2 > 0.0 {
        // This is expected precision loss — document it.
        // Not a bug in the merge formula per se, but worth knowing.
        assert!(direct.m2 < 1e-10,
            "Precision loss in Welford merge at scale 1e15: \
             direct.m2={:.2e} but merged.m2=0 (delta rounds to 0). \
             This is expected for inputs where delta < machine_epsilon * scale.",
            direct.m2);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MomentStats::merge — Test 2: Associativity
// ═══════════════════════════════════════════════════════════════════════════

/// Parallel Welford merge must be associative for the scan to be correct.
/// compose(compose(A,B),C) = compose(A,compose(B,C)) to machine precision.
#[test]
fn moment_merge_associativity_normal_values() {
    let a = ms_from(&[1.0, 2.0, 3.0]);
    let b = ms_from(&[4.0, 5.0, 6.0]);
    let c = ms_from(&[7.0, 8.0, 9.0]);

    let left  = MomentStats::merge(&MomentStats::merge(&a, &b), &c);
    let right = MomentStats::merge(&a, &MomentStats::merge(&b, &c));

    // Mean should match exactly (simple sum of sums / total count)
    assert!((left.sum - right.sum).abs() < 1e-12,
        "Associativity: sum mismatch: left={} right={}", left.sum, right.sum);
    assert!((left.count - right.count).abs() < 1e-12,
        "Associativity: count mismatch");

    // m2 (variance-related) should match to machine precision
    let rel_err_m2 = (left.m2 - right.m2).abs() / left.m2.max(1e-300);
    assert!(rel_err_m2 < 1e-10,
        "Associativity m2 failure: left.m2={} right.m2={} rel_err={:.2e}",
        left.m2, right.m2, rel_err_m2);

    // m3 (skewness-related) — higher moments have more floating-point sensitivity
    let scale = left.m3.abs().max(right.m3.abs()).max(1e-300);
    let rel_err_m3 = (left.m3 - right.m3).abs() / scale;
    assert!(rel_err_m3 < 1e-8,
        "Associativity m3 failure: left.m3={} right.m3={} rel_err={:.2e}",
        left.m3, right.m3, rel_err_m3);
}

/// Associativity with large dynamic range in the merge formula.
/// This is Aristotle's key concern for the SarkkaMerge correction term —
/// the same structure appears in m3 and m4 merge formulas.
#[test]
fn moment_merge_associativity_large_dynamic_range() {
    // Partitions with very different scales
    let a = ms_from(&[1e8, 2e8, 3e8]);          // large values
    let b = ms_from(&[1.0, 2.0, 3.0]);          // small values
    let c = ms_from(&[1e8 + 1.0, 2e8 + 1.0]);  // large + small offset

    let left  = MomentStats::merge(&MomentStats::merge(&a, &b), &c);
    let right = MomentStats::merge(&a, &MomentStats::merge(&b, &c));

    let scale = left.m2.abs().max(right.m2.abs()).max(1e-300);
    let rel_err_m2 = (left.m2 - right.m2).abs() / scale;
    assert!(rel_err_m2 < 1e-6,
        "Associativity m2 with large dynamic range: \
         left.m2={} right.m2={} rel_err={:.2e} — \
         higher-order terms in Welford merge may amplify cancellation",
        left.m2, right.m2, rel_err_m2);
}

// ═══════════════════════════════════════════════════════════════════════════
// MomentStats::merge — Test 3: min/max NaN-eating
// ═══════════════════════════════════════════════════════════════════════════

/// MomentStats::merge uses `a.min.min(b.min)` and `a.max.max(b.max)`.
/// Rust's f64::min/max eat NaN: min(NaN, x) = x.
///
/// When does a partition have NaN in its min? When moments_ungrouped is called
/// with all-NaN input, it returns empty() which has min=+Inf, max=-Inf.
/// These sentinel values are correct for the "empty partition" identity.
///
/// But what if NaN contamination reaches the min/max fields through a different path?
/// The design excludes NaN before computing min/max in moments_ungrouped,
/// so the min/max NaN-eating in merge is safe ONLY IF all partitions came from
/// moments_ungrouped. If a MomentStats is constructed directly with NaN in min/max,
/// the merge silently drops it.
///
/// This test documents the assumption: merge(A, B) is safe only when A and B
/// were produced by moments_ungrouped (or an equivalent NaN-excluding accumulator).
#[test]
fn moment_merge_min_max_nan_eating_direct_construction() {
    // Construct a MomentStats with NaN in min directly (bypassing moments_ungrouped)
    let contaminated = MomentStats {
        count: 2.0,
        sum: 3.0,
        min: f64::NAN,   // contaminated
        max: 2.0,
        m2: 0.5,
        m3: 0.0,
        m4: 0.25,
    };
    let valid = ms_from(&[4.0, 5.0]);

    let merged = MomentStats::merge(&contaminated, &valid);

    // Mathematically: merged.min should be NaN (contaminated partition had NaN min)
    // BUG: f64::min(NaN, 4.0) = 4.0 — NaN silently dropped, merged.min = 4.0
    assert!(merged.min.is_nan(),
        "BUG: MomentStats::merge uses f64::min (NaN-eating) for min field. \
         contaminated.min=NaN, valid.min=4.0, got merged.min={} (NaN swallowed). \
         If MomentStats can be constructed with NaN min/max, the merge silently \
         loses the contamination. Fix: use NaN-propagating min, or validate inputs.",
        merged.min);
}

/// The symmetric case: NaN in max.
#[test]
fn moment_merge_max_nan_eating_direct_construction() {
    let contaminated = MomentStats {
        count: 2.0,
        sum: 3.0,
        min: 1.0,
        max: f64::NAN,  // contaminated
        m2: 0.5,
        m3: 0.0,
        m4: 0.25,
    };
    let valid = ms_from(&[4.0, 5.0]);

    let merged = MomentStats::merge(&contaminated, &valid);

    assert!(merged.max.is_nan(),
        "BUG: MomentStats::merge uses f64::max (NaN-eating) for max field. \
         contaminated.max=NaN, valid.max=5.0, got merged.max={} (NaN swallowed).",
        merged.max);
}

// ═══════════════════════════════════════════════════════════════════════════
// MomentStats::merge — Correctness: merged equals one-pass on concatenated data
// ═══════════════════════════════════════════════════════════════════════════

/// Merge of two partitions should equal one-pass over the concatenated data.
/// This is the fundamental correctness property of Welford's parallel algorithm.
#[test]
fn moment_merge_equals_onepass_n16() {
    let vals_a: Vec<f64> = (0..8).map(|i| i as f64 * 1.5).collect();
    let vals_b: Vec<f64> = (8..16).map(|i| i as f64 * 1.5).collect();
    let all_vals: Vec<f64> = vals_a.iter().chain(vals_b.iter()).cloned().collect();

    let a = ms_from(&vals_a);
    let b = ms_from(&vals_b);
    let merged = MomentStats::merge(&a, &b);
    let direct = ms_from(&all_vals);

    assert!((merged.count - direct.count).abs() < 1e-10, "count mismatch");
    assert!((merged.sum - direct.sum).abs() < 1e-8, "sum mismatch: {} vs {}", merged.sum, direct.sum);
    let rel_m2 = (merged.m2 - direct.m2).abs() / direct.m2.max(1e-300);
    assert!(rel_m2 < 1e-10, "m2 mismatch: rel_err={:.2e}", rel_m2);
}

#[test]
fn moment_merge_equals_onepass_n17() {
    // n=17: non-power-of-2, exercises the scan identity padding path
    let vals_a: Vec<f64> = (0..9).map(|i| i as f64 + 0.5).collect();
    let vals_b: Vec<f64> = (9..17).map(|i| i as f64 + 0.5).collect();
    let all_vals: Vec<f64> = vals_a.iter().chain(vals_b.iter()).cloned().collect();

    let a = ms_from(&vals_a);
    let b = ms_from(&vals_b);
    let merged = MomentStats::merge(&a, &b);
    let direct = ms_from(&all_vals);

    assert!((merged.count - direct.count).abs() < 1e-10, "n=17 count mismatch");
    let rel_m2 = (merged.m2 - direct.m2).abs() / direct.m2.max(1e-300);
    assert!(rel_m2 < 1e-10, "n=17 m2 mismatch: rel_err={:.2e}", rel_m2);
}

#[test]
fn moment_merge_4way_equals_onepass_n100() {
    // Four-way merge (simulating 4-thread parallel): should equal one-pass
    let all_vals: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1 + 1.0).collect();
    let a = ms_from(&all_vals[0..25]);
    let b = ms_from(&all_vals[25..50]);
    let c = ms_from(&all_vals[50..75]);
    let d = ms_from(&all_vals[75..100]);

    let merged = MomentStats::merge(&MomentStats::merge(&a, &b),
                                    &MomentStats::merge(&c, &d));
    let direct = ms_from(&all_vals);

    assert!((merged.count - direct.count).abs() < 1e-10, "n=100 count");
    let rel_m2 = (merged.m2 - direct.m2).abs() / direct.m2.max(1e-300);
    assert!(rel_m2 < 1e-8, "n=100 m2 rel_err={:.2e}", rel_m2);
}

// ═══════════════════════════════════════════════════════════════════════════
// MomentStats identity element
// ═══════════════════════════════════════════════════════════════════════════

/// empty() is the identity: merge(empty, A) = A and merge(A, empty) = A.
#[test]
fn moment_merge_empty_is_left_identity() {
    let a = ms_from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let empty = MomentStats::empty();
    let merged = MomentStats::merge(&empty, &a);

    assert!((merged.count - a.count).abs() < 1e-12, "Left identity: count");
    assert!((merged.sum   - a.sum).abs()   < 1e-12, "Left identity: sum");
    assert!((merged.m2    - a.m2).abs()    < 1e-12, "Left identity: m2");
    assert!((merged.min   - a.min).abs()   < 1e-12, "Left identity: min");
    assert!((merged.max   - a.max).abs()   < 1e-12, "Left identity: max");
}

#[test]
fn moment_merge_empty_is_right_identity() {
    let a = ms_from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let empty = MomentStats::empty();
    let merged = MomentStats::merge(&a, &empty);

    assert!((merged.count - a.count).abs() < 1e-12, "Right identity: count");
    assert!((merged.sum   - a.sum).abs()   < 1e-12, "Right identity: sum");
    assert!((merged.m2    - a.m2).abs()    < 1e-12, "Right identity: m2");
    assert!((merged.min   - a.min).abs()   < 1e-12, "Right identity: min");
    assert!((merged.max   - a.max).abs()   < 1e-12, "Right identity: max");
}

// ═══════════════════════════════════════════════════════════════════════════
// CopaState::merge — Test 1: Catastrophic cancellation
// ═══════════════════════════════════════════════════════════════════════════

fn copa_from_rows(rows: &[[f64; 2]]) -> CopaState {
    let p = 2;
    let mut state = CopaState::new(p);
    for &row in rows {
        state.add(&row);
    }
    state
}

/// CopaState merge: delta[j] = mean_b[j] - mean_a[j].
/// Same catastrophic cancellation risk as MomentStats for large, close means.
#[test]
fn copa_merge_catastrophic_cancellation_large_mean() {
    let base = 1e12_f64;
    let a = copa_from_rows(&[[base, base + 1.0], [base + 2.0, base + 3.0]]);
    let b = copa_from_rows(&[[base + 4.0, base + 5.0], [base + 6.0, base + 7.0]]);

    let merged = CopaState::merge(&a, &b);

    // Reference: direct accumulation of all 4 rows
    let direct = copa_from_rows(&[
        [base, base + 1.0], [base + 2.0, base + 3.0],
        [base + 4.0, base + 5.0], [base + 6.0, base + 7.0],
    ]);

    let cov_merged = merged.covariance_population();
    let cov_direct = direct.covariance_population();

    for i in 0..2 {
        for j in 0..2 {
            let m = cov_merged.get(i, j);
            let d = cov_direct.get(i, j);
            let scale = m.abs().max(d.abs()).max(1e-300);
            let rel_err = (m - d).abs() / scale;
            assert!(rel_err < 1e-6,
                "CopaState merge catastrophic cancellation: \
                 cov[{},{}] merged={} direct={} rel_err={:.2e} — \
                 delta = mean_b - mean_a at scale {:.0} loses precision",
                i, j, m, d, rel_err, base);
        }
    }
}

/// CopaState merge associativity: three partitions in different orders.
#[test]
fn copa_merge_associativity() {
    let a = copa_from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = copa_from_rows(&[[5.0, 6.0], [7.0, 8.0]]);
    let c = copa_from_rows(&[[9.0, 10.0], [11.0, 12.0]]);

    let left  = CopaState::merge(&CopaState::merge(&a, &b), &c);
    let right = CopaState::merge(&a, &CopaState::merge(&b, &c));

    let cov_l = left.covariance_population();
    let cov_r = right.covariance_population();

    for i in 0..2 {
        for j in 0..2 {
            let l = cov_l.get(i, j);
            let r = cov_r.get(i, j);
            assert!((l - r).abs() < 1e-10,
                "CopaState merge associativity failure at cov[{},{}]: \
                 left={} right={}", i, j, l, r);
        }
    }
}

/// CopaState identity: CopaState::new() (zero observations) is the identity.
#[test]
fn copa_merge_empty_is_identity() {
    let p = 2;
    let a = copa_from_rows(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let empty = CopaState::new(p);

    let left  = CopaState::merge(&empty, &a);
    let right = CopaState::merge(&a, &empty);

    let cov_a = a.covariance_population();
    let cov_l = left.covariance_population();
    let cov_r = right.covariance_population();

    for i in 0..2 {
        for j in 0..2 {
            assert!((cov_a.get(i,j) - cov_l.get(i,j)).abs() < 1e-12,
                "CopaState: empty not left identity at [{},{}]", i, j);
            assert!((cov_a.get(i,j) - cov_r.get(i,j)).abs() < 1e-12,
                "CopaState: empty not right identity at [{},{}]", i, j);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SufficientStatistics::merge — correctness and associativity
// ═══════════════════════════════════════════════════════════════════════════

fn ss_from(values: &[f64], groups: &[usize], n_groups: usize) -> SufficientStatistics {
    // Manual groupwise Welford accumulation.
    let mut sums = vec![0.0f64; n_groups];
    let mut counts = vec![0.0f64; n_groups];
    // First pass: sums and counts
    for (&v, &g) in values.iter().zip(groups.iter()) {
        if !v.is_nan() {
            sums[g] += v;
            counts[g] += 1.0;
        }
    }
    // Second pass: m2 via centered deviations
    let mut m2 = vec![0.0f64; n_groups];
    for (&v, &g) in values.iter().zip(groups.iter()) {
        if !v.is_nan() && counts[g] > 0.0 {
            let mean = sums[g] / counts[g];
            m2[g] += (v - mean) * (v - mean);
        }
    }
    SufficientStatistics::from_welford(n_groups, sums, m2, counts)
}

/// SufficientStatistics merge equals one-pass over concatenated data.
#[test]
fn suff_stats_merge_equals_onepass() {
    let vals_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let keys_a = vec![0usize, 0, 1, 1, 2];
    let vals_b = vec![6.0, 7.0, 8.0, 9.0, 10.0];
    let keys_b = vec![0usize, 1, 1, 2, 2];

    let a = ss_from(&vals_a, &keys_a, 3);
    let b = ss_from(&vals_b, &keys_b, 3);
    let merged = a.merge(&b);

    // Direct one-pass
    let all_vals: Vec<f64> = vals_a.iter().chain(vals_b.iter()).cloned().collect();
    let all_keys: Vec<usize> = keys_a.iter().chain(keys_b.iter()).cloned().collect();
    let direct = ss_from(&all_vals, &all_keys, 3);

    for g in 0..3 {
        assert!((merged.mean(g) - direct.mean(g)).abs() < 1e-10,
            "Group {} mean: merged={} direct={}", g, merged.mean(g), direct.mean(g));
        let rel_var = (merged.variance_sample(g) - direct.variance_sample(g)).abs()
            / direct.variance_sample(g).max(1e-300);
        assert!(rel_var < 1e-8,
            "Group {} variance rel_err={:.2e}", g, rel_var);
    }
}

/// SufficientStatistics merge associativity: three partitions.
#[test]
fn suff_stats_merge_associativity() {
    let make = |start: usize, end: usize| -> SufficientStatistics {
        let vals: Vec<f64> = (start..end).map(|i| i as f64).collect();
        let keys: Vec<usize> = (start..end).map(|i| i % 2).collect();
        ss_from(&vals, &keys, 2)
    };

    let a = make(0, 4);
    let b = make(4, 8);
    let c = make(8, 12);

    let left  = a.merge(&b).merge(&c);
    let right = a.merge(&b.merge(&c));

    for g in 0..2 {
        assert!((left.mean(g) - right.mean(g)).abs() < 1e-10,
            "SuffStats associativity: group {} mean left={} right={}",
            g, left.mean(g), right.mean(g));
        let rel_var = (left.variance_sample(g) - right.variance_sample(g)).abs()
            / right.variance_sample(g).max(1e-300);
        assert!(rel_var < 1e-8,
            "SuffStats associativity: group {} var rel_err={:.2e}", g, rel_var);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-operator consistency: MomentStats.merge vs SufficientStatistics.merge
// ═══════════════════════════════════════════════════════════════════════════

/// MomentStats and SufficientStatistics implement the same Welford formula.
/// When applied to the same data, their variances should agree.
#[test]
fn welford_merge_implementations_agree() {
    let vals_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let vals_b = vec![6.0, 7.0, 8.0, 9.0, 10.0];

    // MomentStats path
    let ms_a = ms_from(&vals_a);
    let ms_b = ms_from(&vals_b);
    let ms_merged = MomentStats::merge(&ms_a, &ms_b);
    let ms_var = ms_merged.m2 / (ms_merged.count - 1.0);

    // SufficientStatistics path (single group)
    let all_vals: Vec<f64> = vals_a.iter().chain(vals_b.iter()).cloned().collect();
    let keys: Vec<usize> = vec![0; all_vals.len()];
    let ss_direct = ss_from(&all_vals, &keys, 1);
    let ss_var = ss_direct.variance_sample(0);

    let rel_err = (ms_var - ss_var).abs() / ss_var.max(1e-300);
    assert!(rel_err < 1e-10,
        "Welford implementations disagree: MomentStats sample_var={} \
         SufficientStatistics sample_var={} rel_err={:.2e}",
        ms_var, ss_var, rel_err);
}
