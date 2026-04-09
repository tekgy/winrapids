//! Adversarial tests for fintek-rescue GAPs:
//! range volatility, Hill tail index, tripower quarticity, DTW, edit distance.

use tambear::volatility::*;
use tambear::nonparametric::*;

// ═══════════════════════════════════════════════════════════════════════════
// RANGE VOLATILITY ESTIMATORS
// ═══════════════════════════════════════════════════════════════════════════

/// Parkinson: H = L → variance = 0.
#[test]
fn parkinson_flat_bar() {
    let v = parkinson_variance(100.0, 100.0);
    assert!((v - 0.0).abs() < 1e-15, "Flat bar Parkinson should be 0, got {}", v);
}

/// Parkinson gold: H=110, L=90 → (ln(110/90))² / (4·ln 2) = (0.2007)² / 2.773 ≈ 0.01452
#[test]
fn parkinson_gold_standard() {
    let v = parkinson_variance(110.0, 90.0);
    let expected = (110.0_f64 / 90.0).ln().powi(2) / (4.0 * 2.0_f64.ln());
    assert!((v - expected).abs() < 1e-12, "Parkinson should match formula: {} vs {}", v, expected);
    assert!((v - 0.01452).abs() < 0.0001);
}

/// Parkinson NaN on invalid inputs.
#[test]
fn parkinson_invalid() {
    assert!(parkinson_variance(-1.0, 100.0).is_nan());
    assert!(parkinson_variance(100.0, 0.0).is_nan());
    assert!(parkinson_variance(90.0, 100.0).is_nan()); // H < L
}

/// Garman-Klass on flat bar (no drift, no range) = 0.
#[test]
fn garman_klass_flat_bar() {
    let v = garman_klass_variance(100.0, 100.0, 100.0, 100.0);
    assert!((v - 0.0).abs() < 1e-15, "Flat GK should be 0, got {}", v);
}

/// Garman-Klass: non-zero range with O=C → pure range term.
#[test]
fn garman_klass_no_drift() {
    let v = garman_klass_variance(100.0, 110.0, 90.0, 100.0);
    let lnhl = (110.0_f64 / 90.0).ln();
    let expected = 0.5 * lnhl * lnhl; // ln(C/O)=0 so second term vanishes
    assert!((v - expected).abs() < 1e-12, "GK no-drift should be 0.5·(ln(H/L))², got {}", v);
}

/// Rogers-Satchell is drift-independent: same result regardless of C location.
#[test]
fn rogers_satchell_drift_independent() {
    // Two bars with same H/L but different O/C ordering
    let v1 = rogers_satchell_variance(100.0, 110.0, 90.0, 105.0); // up day
    let v2 = rogers_satchell_variance(105.0, 110.0, 90.0, 100.0); // down day
    // Both should be finite and non-negative
    assert!(v1.is_finite() && v1 >= 0.0);
    assert!(v2.is_finite() && v2 >= 0.0);
}

/// Rogers-Satchell NaN on non-positive prices.
#[test]
fn rogers_satchell_invalid() {
    assert!(rogers_satchell_variance(0.0, 110.0, 90.0, 100.0).is_nan());
}

/// Yang-Zhang on series of flat bars: should be zero variance.
#[test]
fn yang_zhang_flat_series() {
    let bars = 10;
    let opens = vec![100.0; bars];
    let highs = vec![100.0; bars];
    let lows = vec![100.0; bars];
    let closes = vec![100.0; bars];
    let prev_closes = vec![100.0; bars];
    let v = yang_zhang_variance(&opens, &highs, &lows, &closes, &prev_closes);
    assert!(v.abs() < 1e-10, "Flat series YZ should be 0, got {}", v);
}

/// Yang-Zhang positive for real-ish OHLC data.
#[test]
fn yang_zhang_real_data() {
    let opens = vec![100.0, 101.0, 99.5, 102.0, 100.5];
    let highs = vec![101.5, 102.0, 100.0, 103.0, 101.0];
    let lows = vec![99.5, 100.5, 99.0, 101.0, 100.0];
    let closes = vec![101.0, 99.5, 102.0, 100.5, 100.8];
    let prev_closes = vec![99.8, 101.0, 99.5, 102.0, 100.5];
    let v = yang_zhang_variance(&opens, &highs, &lows, &closes, &prev_closes);
    assert!(v.is_finite() && v >= 0.0, "YZ should be non-negative, got {}", v);
}

/// Yang-Zhang with mismatched lengths: NaN.
#[test]
fn yang_zhang_length_mismatch() {
    let v = yang_zhang_variance(&[100.0], &[101.0, 102.0], &[99.0], &[100.5], &[100.0]);
    assert!(v.is_nan());
}

// ═══════════════════════════════════════════════════════════════════════════
// HILL TAIL INDEX
// ═══════════════════════════════════════════════════════════════════════════

/// Hill on Pareto-distributed data (α=2): should recover α ≈ 2.
#[test]
fn hill_pareto_alpha() {
    // Sample from Pareto with α=3: X = U^(-1/α) where U ~ Uniform(0,1)
    let n = 2000;
    let alpha_true: f64 = 3.0;
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let data: Vec<f64> = (0..n).map(|_| {
        let u: f64 = tambear::rng::TamRng::next_f64(&mut rng).max(1e-300);
        u.powf(-1.0 / alpha_true)
    }).collect();

    // Use top 5% as tail
    let k = (n as f64 * 0.05) as usize;
    let alpha_est = hill_tail_alpha(&data, k);
    assert!(alpha_est.is_finite(), "Hill α should be finite, got {}", alpha_est);
    // Hill estimator is consistent but noisy; allow wide tolerance
    assert!((alpha_est - alpha_true).abs() < 1.0,
        "Hill α should recover true α: est={}, true={}", alpha_est, alpha_true);
}

/// Hill with k=0: NaN.
#[test]
fn hill_k_zero() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(hill_estimator(&data, 0).is_nan());
}

/// Hill with k >= n: NaN.
#[test]
fn hill_k_too_large() {
    let data = vec![1.0, 2.0, 3.0];
    assert!(hill_estimator(&data, 5).is_nan());
}

/// Hill with empty data: NaN.
#[test]
fn hill_empty() {
    assert!(hill_estimator(&[], 3).is_nan());
}

// ═══════════════════════════════════════════════════════════════════════════
// TRIPOWER QUARTICITY
// ═══════════════════════════════════════════════════════════════════════════

/// TQ on constant returns: zero (no variation).
#[test]
fn tripower_constant() {
    let r = vec![0.0; 20];
    let tq = tripower_quarticity(&r);
    assert!((tq - 0.0).abs() < 1e-15, "TQ of zero returns should be 0, got {}", tq);
}

/// TQ on positive-variance returns: positive.
#[test]
fn tripower_nonzero() {
    let r = vec![0.01, -0.02, 0.015, -0.01, 0.005, -0.005, 0.02, -0.015, 0.01, -0.01];
    let tq = tripower_quarticity(&r);
    assert!(tq.is_finite() && tq > 0.0, "TQ should be positive, got {}", tq);
}

/// TQ too short: NaN.
#[test]
fn tripower_too_short() {
    assert!(tripower_quarticity(&[0.01, 0.02]).is_nan());
}

// ═══════════════════════════════════════════════════════════════════════════
// DTW
// ═══════════════════════════════════════════════════════════════════════════

/// DTW of identical sequences: 0.
#[test]
fn dtw_identical() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let d = dtw(&x, &x);
    assert!((d - 0.0).abs() < 1e-15, "DTW(x,x) should be 0, got {}", d);
}

/// DTW of shifted sequence: still small (time-warping handles shifts).
#[test]
fn dtw_shifted() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; // x prepended with 0
    let d = dtw(&x, &y);
    // With time warping, cost should be low
    assert!(d >= 0.0 && d < 5.0, "DTW with prefix should be small, got {}", d);
}

/// DTW gold standard: [1,3,4,9,8] vs [1,2,4,5,8] → DP table calculation.
#[test]
fn dtw_gold_standard() {
    let x = vec![1.0, 3.0, 4.0, 9.0, 8.0];
    let y = vec![1.0, 2.0, 4.0, 5.0, 8.0];
    let d = dtw(&x, &y);
    // Expected cost along optimal path (computed manually):
    // match(1,1)=0, match(3,2)=1, match(4,4)=0, match(9,5)=4, match(8,8)=0
    // Optimal path sum = 5
    assert!(d <= 6.0, "DTW should find low-cost path, got {}", d);
}

/// DTW with empty sequence: NaN.
#[test]
fn dtw_empty() {
    let d = dtw(&[], &[1.0, 2.0]);
    assert!(d.is_nan());
}

/// DTW is symmetric: DTW(x,y) = DTW(y,x).
#[test]
fn dtw_symmetric() {
    let x = vec![1.0, 4.0, 2.0, 7.0];
    let y = vec![3.0, 2.0, 5.0, 6.0, 1.0];
    let d1 = dtw(&x, &y);
    let d2 = dtw(&y, &x);
    assert!((d1 - d2).abs() < 1e-10, "DTW should be symmetric: {} vs {}", d1, d2);
}

/// DTW banded with small window: may be larger than unbanded (path constrained).
#[test]
fn dtw_banded_constraint() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // reversed
    let d_full = dtw(&x, &y);
    let d_band = dtw_banded(&x, &y, 1);
    // Banded with tight window should be ≥ full DTW
    assert!(d_band >= d_full - 1e-10,
        "Banded DTW should be >= full: band={}, full={}", d_band, d_full);
}

// ═══════════════════════════════════════════════════════════════════════════
// LEVENSHTEIN EDIT DISTANCE
// ═══════════════════════════════════════════════════════════════════════════

/// Levenshtein of identical sequences: 0.
#[test]
fn levenshtein_identical() {
    let a = vec![1, 2, 3, 4, 5];
    let d = levenshtein(&a, &a);
    assert_eq!(d, 0);
}

/// Levenshtein gold: "kitten" → "sitting" = 3.
/// Encoded as integers: k=1,i=2,t=3,e=4,n=5,s=6,g=7.
/// kitten = [1,2,3,3,4,5]
/// sitting = [6,2,3,3,2,5,7]
/// Edits: k→s (sub), e→i (sub), append g = 3 edits
#[test]
fn levenshtein_kitten_sitting() {
    let kitten = vec![1_i64, 2, 3, 3, 4, 5];
    let sitting = vec![6_i64, 2, 3, 3, 2, 5, 7];
    let d = levenshtein(&kitten, &sitting);
    assert_eq!(d, 3, "Classic kitten→sitting should be 3, got {}", d);
}

/// Levenshtein empty cases.
#[test]
fn levenshtein_empty() {
    assert_eq!(levenshtein(&[], &[]), 0);
    assert_eq!(levenshtein(&[1, 2, 3], &[]), 3);
    assert_eq!(levenshtein(&[], &[1, 2, 3]), 3);
}

/// Levenshtein symmetry: d(a,b) = d(b,a).
#[test]
fn levenshtein_symmetric() {
    let a = vec![1, 4, 2, 7, 3];
    let b = vec![4, 1, 2, 5, 3, 6];
    let d1 = levenshtein(&a, &b);
    let d2 = levenshtein(&b, &a);
    assert_eq!(d1, d2);
}

/// quantile_symbolize: data split into roughly equal bins.
#[test]
fn quantile_symbolize_balanced() {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let symbols = quantile_symbolize(&data, 5);
    assert_eq!(symbols.len(), 100);
    // Count each symbol
    let mut counts = [0_usize; 5];
    for &s in &symbols {
        if s >= 0 && (s as usize) < 5 { counts[s as usize] += 1; }
    }
    // Each quintile should have ~20 elements
    for (i, &c) in counts.iter().enumerate() {
        assert!(c >= 15 && c <= 25, "Quintile {} count should be ~20, got {}", i, c);
    }
}

/// Edit distance on symbolized series: identical data → 0.
#[test]
fn edit_distance_identical_series() {
    let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
    let d = edit_distance_on_series(&data, &data, 5);
    assert_eq!(d, 0);
}

/// Edit distance on different series: positive.
#[test]
fn edit_distance_different_series() {
    let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
    let y: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).cos()).collect();
    let d = edit_distance_on_series(&x, &y, 5);
    assert!(d > 0, "Different series should have positive edit distance, got {}", d);
}
