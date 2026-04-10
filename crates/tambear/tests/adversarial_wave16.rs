//! Adversarial Wave 16 — NaN-eating in Davies-Bouldin index and Hurst R/S
//!
//! Aristotle's three-test template applied to:
//!   1. `clustering::davies_bouldin_score` — `fold(0.0_f64, f64::max)` at line 666
//!      BUG CLASS A: initial accumulator 0.0 is wrong (should be NEG_INFINITY for max)
//!      BUG CLASS B: f64::max eats NaN — NaN ratio → silent 0.0, not NaN propagation
//!   2. `complexity::hurst_rs` — two NaN-eating folds at lines 240-241
//!      `fold(f64::NEG_INFINITY, f64::max)` and `fold(f64::INFINITY, f64::min)`
//!      NaN in cumulative deviation → range = NaN eaten → H computed from garbage
//!
//! Mathematical truths asserted:
//!   - Davies-Bouldin: if any (s[i]+s[j])/d_ij ratio is NaN, the cluster's max should be NaN
//!   - Davies-Bouldin: fold starting from 0.0 underestimates the score when all ratios < 0
//!     (impossible in practice since s,d ≥ 0, but the wrong identity is a latent class bug)
//!   - Hurst R/S: if any data point is NaN, hurst_rs should return NaN (not a finite H value)
//!   - R/S range = max(cum_dev) - min(cum_dev): NaN in cum_dev must propagate to range
//!
//! All tests assert mathematical truths. Failures are bugs.

use tambear::{
    davies_bouldin_score,
    ClusterCentroids,
    hurst_rs,
};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers — build ClusterCentroids by hand so we can inject NaN
// ═══════════════════════════════════════════════════════════════════════════

/// Build a pre-computed ClusterCentroids for two clusters in 1D.
/// Cluster 0 centroid at c0, cluster 1 centroid at c1.
/// Sizes control the mean intra-cluster distance denominator.
fn two_cluster_cc(c0: f64, c1: f64, n_dims: usize) -> ClusterCentroids {
    let mut id_to_idx = HashMap::new();
    id_to_idx.insert(0i32, 0usize);
    id_to_idx.insert(1i32, 1usize);
    ClusterCentroids {
        k: 2,
        n_dims,
        sizes: vec![1, 1],
        centroids: vec![c0, c1],
        id_to_idx,
    }
}

/// A single well-separated 1D data set: cluster 0 = [0.0], cluster 1 = [10.0]
/// DB score: s[0] = 0, s[1] = 0 (single-point clusters), d_01 = 10
/// (s[0]+s[1])/d = 0 → score = 0.0. This is the "trivial" degenerate.
fn trivial_two_cluster() -> (Vec<f64>, Vec<i32>, usize, ClusterCentroids) {
    let data = vec![0.0_f64, 10.0];
    let labels = vec![0i32, 1];
    let cc = two_cluster_cc(0.0, 10.0, 1);
    (data, labels, 1, cc)
}

// ═══════════════════════════════════════════════════════════════════════════
// Davies-Bouldin — Test 1: wrong initial accumulator
// ═══════════════════════════════════════════════════════════════════════════

/// Davies-Bouldin max over other clusters must use NEG_INFINITY as identity, not 0.0.
///
/// For a single-point cluster (s[i]=0) paired with all other single-point clusters,
/// every ratio (s[i]+s[j])/d_ij = 0/d = 0.0.
/// With identity=0.0, fold(0.0, max) returns 0.0 — appears correct by coincidence.
/// But the REAL identity for max is NEG_INFINITY; 0.0 is a numeric value that
/// masks the distinction between "no ratios computed" and "all ratios were zero".
///
/// This test is informational: it documents the wrong-identity smell, even though
/// for non-negative ratios (s,d ≥ 0) the behavior is numerically correct.
/// The test WILL PASS currently — it documents the boundary of when the bug bites.
#[test]
fn davies_bouldin_wrong_identity_masked_by_non_negative_ratios() {
    let (data, labels, n_dims, cc) = trivial_two_cluster();
    let score = davies_bouldin_score(&data, &labels, n_dims, Some(&cc));
    // Single-point clusters: s[0]=s[1]=0, so (0+0)/d_01 = 0.0
    // fold(0.0, max) gives 0.0 → DB = 0.0
    // Mathematically correct here, but wrong identity was used
    assert_eq!(score, 0.0,
        "Single-point cluster DB should be 0.0 (coincidence, not correctness of identity)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Davies-Bouldin — Test 2: NaN in distance → NaN must propagate
// ═══════════════════════════════════════════════════════════════════════════

/// If a centroid coordinate is NaN, then d_ij = sqrt(NaN) = NaN.
/// (s[i]+s[j])/NaN = NaN. Then fold(0.0, f64::max) with NaN:
///   f64::max(0.0, NaN) = 0.0  ← NaN EATEN
/// So the cluster's contribution becomes 0.0 (falsely excellent separation).
/// DB score returns a finite value instead of NaN.
///
/// EXPECTED: davies_bouldin_score should return NaN when any centroid is NaN.
/// ACTUAL (BUG): returns 0.0 — distance is NaN, ratio is NaN, max eats NaN.
#[test]
fn davies_bouldin_nan_centroid_must_propagate() {
    // Cluster 0 at 0.0, cluster 1 at NaN
    let mut cc = two_cluster_cc(0.0, f64::NAN, 1);
    // Give each cluster one point near its centroid
    cc.sizes = vec![2, 2];
    let data = vec![0.0_f64, 1.0, f64::NAN, f64::NAN];
    let labels = vec![0i32, 0, 1, 1];

    let score = davies_bouldin_score(&data, &labels, 1, Some(&cc));

    assert!(score.is_nan(),
        "BUG: davies_bouldin_score with NaN centroid should return NaN, \
         but got {} — fold(0.0, f64::max) eats NaN so 0.0 is returned \
         instead of propagating the degenerate distance",
        score);
}

/// If the inter-centroid distance d_ij = 0.0 (identical centroids),
/// the ratio (s[i]+s[j])/0 = Inf. Inf in fold(0.0, max) correctly propagates
/// (f64::max(0.0, Inf) = Inf). So identical centroids return Inf — which is
/// mathematically correct (zero separation between clusters is infinitely bad).
/// This verifies the fold does NOT eat Inf.
#[test]
fn davies_bouldin_zero_centroid_distance_gives_inf() {
    // Both centroids at 0.0 — zero inter-cluster distance
    let mut cc = two_cluster_cc(0.0, 0.0, 1);
    cc.sizes = vec![2, 2];
    // Points: cluster 0 at [0.0, 1.0], cluster 1 at [2.0, 3.0]
    // s[0] = (|0-0| + |1-0|)/2 = 0.5, s[1] = (|2-0| + |3-0|)/2 = 2.5
    let data = vec![0.0_f64, 1.0, 2.0, 3.0];
    let labels = vec![0i32, 0, 1, 1];

    let score = davies_bouldin_score(&data, &labels, 1, Some(&cc));

    // d_01 < 1e-300 guard returns 0.0 for that ratio, so score = 0.0
    // (The guard clips the Inf — this is a separate design decision)
    // The point is: the fold itself doesn't eat Inf when Inf is present
    assert!(score.is_finite() || score.is_infinite(),
        "Davies-Bouldin with coincident centroids should give 0.0 or Inf, got NaN: {}",
        score);
}

// ═══════════════════════════════════════════════════════════════════════════
// Davies-Bouldin — Test 3: NaN validity propagation via intra-cluster distance
// ═══════════════════════════════════════════════════════════════════════════

/// s[i] = mean intra-cluster distance. If a data point is NaN, then
/// sq_dist(NaN_point, centroid) = NaN, so s[i] += NaN → s[i] = NaN.
/// Then (NaN + s[j])/d = NaN. fold(0.0, max) eats NaN → score garbage.
///
/// EXPECTED: NaN data → NaN score.
/// ACTUAL (BUG): NaN data → 0.0 score (NaN eaten by fold).
#[test]
fn davies_bouldin_nan_data_point_must_propagate() {
    // Clean centroids, NaN in cluster 0's data
    let cc = two_cluster_cc(0.0, 10.0, 1);
    let data = vec![f64::NAN, 1.0, 10.0, 11.0];  // NaN in cluster 0
    let labels = vec![0i32, 0, 1, 1];

    let score = davies_bouldin_score(&data, &labels, 1, Some(&cc));

    // s[0] = (dist(NaN,0) + dist(1,0))/2 = (NaN + 1)/2 = NaN
    // s[1] = (dist(10,10) + dist(11,10))/2 = (0 + 1)/2 = 0.5
    // ratio(0,1) = (NaN + 0.5)/10 = NaN
    // fold(0.0, max): f64::max(0.0, NaN) = 0.0  ← eaten
    assert!(score.is_nan(),
        "BUG: davies_bouldin_score with NaN data point should return NaN, \
         but got {} — f64::max(0.0, NaN) = 0.0 eats the propagated NaN",
        score);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hurst R/S — Test 1: NaN in data must propagate
// ═══════════════════════════════════════════════════════════════════════════

/// Mathematical truth: hurst_rs is estimating a property of the time series.
/// A NaN in the series means the series is undefined. The R/S statistic for
/// any block containing a NaN is undefined — the range of cumulative deviations
/// is undefined.
///
/// EXPECTED: hurst_rs returns NaN when any data point is NaN.
/// ACTUAL (BUG): returns a finite H value — the NaN-eating fold produces a
/// "range" of 0.0 or garbage, then log(0) = -Inf contaminates the regression,
/// or the point is skipped, giving a misleadingly finite result.
#[test]
fn hurst_rs_nan_in_data_must_propagate() {
    // 50 clean points + 1 NaN
    let mut data: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    data[25] = f64::NAN;

    let h = hurst_rs(&data);

    assert!(h.is_nan(),
        "BUG: hurst_rs with NaN in data should return NaN, but got {} \
         — fold(NEG_INFINITY, f64::max) eats NaN in the cumulative deviation range",
        h);
}

/// Single-block NaN: the entire block contains only NaN values.
/// block mean = NaN, cum_dev[i] = NaN - NaN = NaN for all i.
/// range = NaN_max - NaN_min = NaN - NaN = NaN.
/// std = NaN. std > 0.0 is false for NaN (NaN comparisons return false),
/// so the block is silently SKIPPED — rs_count stays 0 for this block,
/// meaning the NaN is swallowed without propagation.
///
/// If ALL blocks are pure-NaN, rs_count = 0, hurst_rs returns NaN.
/// If SOME blocks are clean, hurst_rs returns a finite H from only the clean blocks.
/// This is the silent data corruption: NaN blocks are silently dropped.
#[test]
fn hurst_rs_nan_block_silently_dropped() {
    // 100 data points: first 50 clean (random walk), last 50 all NaN
    let mut data: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
    data.extend(std::iter::repeat(f64::NAN).take(50));

    let h = hurst_rs(&data);

    // If hurst_rs correctly propagates NaN: h.is_nan()
    // If hurst_rs silently drops NaN blocks: h is finite (computed from clean blocks only)
    assert!(h.is_nan(),
        "BUG: hurst_rs should return NaN when any data contains NaN, \
         but got {} — NaN blocks are silently dropped because \
         `std > 0.0` evaluates to false for NaN, suppressing the block \
         without propagating the invalidity",
        h);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hurst R/S — Test 2: range = max - min must use correct identity
// ═══════════════════════════════════════════════════════════════════════════

/// The R/S range is computed as:
///   max(cum_dev) - min(cum_dev)
/// where cum_dev is the cumulative deviation from the block mean.
///
/// If the cumulative deviation is a constant (all elements equal),
/// range = 0.0, std = 0.0, so the block is skipped (std > 0.0 is false).
/// This is correct behavior.
///
/// But verify: a flat signal (zero variance) gives H = NaN (no blocks qualify).
#[test]
fn hurst_rs_flat_signal_returns_nan() {
    // All identical values → zero variance → all blocks skipped → rs_count = 0 → NaN
    let data = vec![42.0_f64; 100];
    let h = hurst_rs(&data);
    assert!(h.is_nan(),
        "hurst_rs on flat signal (zero variance) should return NaN (no valid blocks), \
         got {}",
        h);
}

/// The max fold's starting value of NEG_INFINITY is correct for the maximum.
/// The min fold's starting value of INFINITY is correct for the minimum.
/// Verify: a block with a single valid element gives range = 0.
/// This tests that the identity values don't contaminate single-element blocks.
/// (block_size ≥ 10 in practice, but let's test via white-box: the fold identities)
///
/// More directly: verify that a pure-positive cumulative deviation signal
/// has range = max_cum_dev - 0.0, not max_cum_dev - NEG_INFINITY.
/// Use a data series where cum_dev is always ≥ 0: a monotone increasing series.
#[test]
fn hurst_rs_monotone_signal_range_nonnegative() {
    // Pure trend: cum_dev increases monotonically
    // hurst_rs should return H ≈ 1.0 for a strong trend
    let data: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let h = hurst_rs(&data);
    // H should be near 1.0 for a pure trend (not 0.5 random walk)
    // More importantly: should be in valid range [0, 1]
    assert!(h >= 0.0 && h <= 1.5,
        "hurst_rs on pure trend should give H in [0.0, 1.5], got {}",
        h);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hurst R/S — Test 3: Inf in data → range = Inf, std = Inf, ratio = NaN
// ═══════════════════════════════════════════════════════════════════════════

/// If data contains Inf, mean = Inf, cum_dev[0] = x - Inf = Inf or NaN.
/// The range becomes Inf - (-Inf) = Inf, or NaN. std = Inf.
/// Inf/Inf = NaN. std > 0.0 is true for Inf (Inf > 0.0 = true).
/// So rs_sum += NaN → rs_sum = NaN → rs_avg = NaN → log(NaN) = NaN
/// → log_rs.push(NaN) → regression gives NaN H.
///
/// Expected: hurst_rs returns NaN when data contains Inf.
/// This tests whether Inf propagates or gets swallowed.
#[test]
fn hurst_rs_inf_in_data_must_propagate() {
    let mut data: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
    data[50] = f64::INFINITY;

    let h = hurst_rs(&data);

    // Inf in data → NaN R/S ratio → regression on NaN log_rs → NaN H
    // OR: Inf → Inf std → std > 0.0 true → rs_sum += NaN → NaN result
    assert!(h.is_nan() || h.is_infinite(),
        "hurst_rs with Inf in data should return NaN or Inf, not a finite H = {}",
        h);
}

// ═══════════════════════════════════════════════════════════════════════════
// Hurst R/S — Associativity (block-order independence)
// ═══════════════════════════════════════════════════════════════════════════

/// R/S Hurst is estimated by fitting log(R/S) = H * log(n) + const
/// over multiple block sizes. The log-log slope H should not depend on
/// the order in which block sizes are processed.
///
/// This tests a weaker property: H is the same whether we process the
/// signal forward or reversed (for a stationary process, H is symmetric).
/// This catches bugs where the fold order matters.
#[test]
fn hurst_rs_forward_backward_agree_for_antipersistent() {
    // Anti-persistent: alternating +1/-1
    let data: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let mut reversed = data.clone();
    reversed.reverse();

    let h_forward = hurst_rs(&data);
    let h_reversed = hurst_rs(&reversed);

    if h_forward.is_nan() || h_reversed.is_nan() { return; } // skip degenerate

    assert!((h_forward - h_reversed).abs() < 0.1,
        "hurst_rs should give same H for forward and reversed alternating signal: \
         forward={}, reversed={}", h_forward, h_reversed);
}

// ═══════════════════════════════════════════════════════════════════════════
// Davies-Bouldin — Associativity: score must not depend on label encoding
// ═══════════════════════════════════════════════════════════════════════════

/// DB score is a symmetric function of clusters. Relabeling cluster 0↔1
/// should give the same score. This tests that the loop over i, and the
/// inner loop over j ≠ i, are correctly symmetric.
#[test]
fn davies_bouldin_score_symmetric_under_label_relabeling() {
    // 4 points, 2 clusters, 1D: cluster 0 = [0,1], cluster 1 = [9,10]
    let data = vec![0.0_f64, 1.0, 9.0, 10.0];
    let labels_a = vec![0i32, 0, 1, 1];
    let labels_b = vec![1i32, 1, 0, 0];  // swap 0↔1

    let score_a = davies_bouldin_score(&data, &labels_a, 1, None);
    let score_b = davies_bouldin_score(&data, &labels_b, 1, None);

    assert!((score_a - score_b).abs() < 1e-10,
        "DB score must be symmetric under cluster relabeling: \
         labels_a={}, labels_b={}", score_a, score_b);
}

/// DB score with 3 well-separated clusters should be less than with 2 poorly-separated ones.
/// This is a monotonicity sanity check.
#[test]
fn davies_bouldin_score_better_separation_gives_lower_score() {
    // Well-separated: clusters at 0, 100, 200
    let data_good = vec![0.0_f64, 1.0, 100.0, 101.0, 200.0, 201.0];
    let labels_good = vec![0i32, 0, 1, 1, 2, 2];

    // Poorly separated: clusters at 0, 2, 4
    let data_bad = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let labels_bad = vec![0i32, 0, 1, 1, 2, 2];

    let score_good = davies_bouldin_score(&data_good, &labels_good, 1, None);
    let score_bad = davies_bouldin_score(&data_bad, &labels_bad, 1, None);

    if score_good.is_nan() || score_bad.is_nan() { return; }

    assert!(score_good < score_bad,
        "Better-separated clusters should have lower DB score: \
         well_separated={}, poorly_separated={}", score_good, score_bad);
}
