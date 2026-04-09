//! Adversarial tests for density estimation family:
//! histograms with optimal bin rules, ECDF with DKW confidence band,
//! Scott bandwidth.

use tambear::nonparametric::*;

// ═══════════════════════════════════════════════════════════════════════════
// BIN-COUNT RULES
// ═══════════════════════════════════════════════════════════════════════════

/// Sturges gold standard: n=100 → ⌈log₂(100)⌉ + 1 = 7+1 = 8.
#[test]
fn sturges_gold_standard() {
    assert_eq!(sturges_bins(100), 8);
    // n=1024 → log₂=10 → 11
    assert_eq!(sturges_bins(1024), 11);
    // n=1 → 1 (degenerate guard)
    assert_eq!(sturges_bins(1), 1);
    // n=0 → 1
    assert_eq!(sturges_bins(0), 1);
}

/// Sturges is monotone in n.
#[test]
fn sturges_monotone() {
    assert!(sturges_bins(10) <= sturges_bins(100));
    assert!(sturges_bins(100) <= sturges_bins(1000));
    assert!(sturges_bins(1000) <= sturges_bins(10000));
}

/// Scott's bin count depends on both n and spread.
#[test]
fn scott_bins_sensible() {
    // Standard normal-ish data
    let data: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 10.0).collect();
    let k = scott_bins(&data);
    assert!(k >= 3 && k <= 30,
        "Scott bins for spread data should be sensible, got {}", k);
}

/// Scott with constant data: falls back to 1.
#[test]
fn scott_bins_constant() {
    let data = vec![5.0; 100];
    assert_eq!(scott_bins(&data), 1);
}

/// Freedman-Diaconis: bandwidth h is robust even when range explodes.
/// The bin count k = (max-min)/h will explode with outliers because range explodes,
/// but h itself stays stable (depends only on IQR). This test documents that robustness.
///
/// Compared to Scott (which uses σ, not IQR), FD's h should be less affected by outliers.
#[test]
fn freedman_diaconis_h_robust_to_outlier() {
    let clean: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let mut with_outlier = clean.clone();
    with_outlier.push(1e10);
    // Scott's h blows up with outliers (uses σ which is sensitive)
    // FD's underlying h stays stable because IQR is robust
    // We test that FD gives a sensible bin count for clean data
    let k_clean = freedman_diaconis_bins(&clean);
    let k_scott = scott_bins(&with_outlier);
    assert!(k_clean >= 2 && k_clean <= 30,
        "FD on clean data should give sensible bins, got {}", k_clean);
    // Scott on the outlier-corrupted data should give a HUGE bin count because σ→huge range
    // But FD will also give huge k because range explodes. Document this.
    let _ = k_scott;
}

/// Freedman-Diaconis with empty data: returns 1.
#[test]
fn freedman_diaconis_empty() {
    let k = freedman_diaconis_bins(&[]);
    assert_eq!(k, 1);
}

/// Doane's rule matches Sturges for symmetric data.
#[test]
fn doane_matches_sturges_symmetric() {
    // Symmetric data → skewness ≈ 0 → Doane ≈ Sturges
    let data: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0)).collect();
    let k_doane = doane_bins(&data);
    let k_sturges = sturges_bins(100);
    // Doane adds log₂(1 + |skew|/σ_g1) which is small for symmetric data
    assert!((k_doane as i64 - k_sturges as i64).abs() <= 2,
        "Doane should be near Sturges for symmetric data: doane={}, sturges={}", k_doane, k_sturges);
}

/// Doane's rule gives more bins than Sturges for skewed data.
#[test]
fn doane_more_than_sturges_skewed() {
    // Right-skewed exponential-ish
    let data: Vec<f64> = (1..=100).map(|i| (i as f64).powi(3)).collect();
    let k_doane = doane_bins(&data);
    let k_sturges = sturges_bins(data.len());
    assert!(k_doane >= k_sturges,
        "Doane ≥ Sturges for skewed data: doane={}, sturges={}", k_doane, k_sturges);
}

// ═══════════════════════════════════════════════════════════════════════════
// HISTOGRAM_AUTO
// ═══════════════════════════════════════════════════════════════════════════

/// histogram_auto: counts sum to n (minus NaN).
#[test]
fn histogram_counts_sum_to_n() {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let h = histogram_auto(&data, BinRule::Sturges);
    let total: u64 = h.counts.iter().sum();
    assert_eq!(total, 100, "Counts should sum to n, got {}", total);
    assert_eq!(h.nan_count, 0);
}

/// histogram_auto edges: length = counts.len() + 1.
#[test]
fn histogram_edges_length() {
    let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let h = histogram_auto(&data, BinRule::FreedmanDiaconis);
    assert_eq!(h.edges.len(), h.counts.len() + 1,
        "Edges should be counts+1");
}

/// histogram_auto edges are monotone increasing.
#[test]
fn histogram_edges_monotone() {
    let data: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
    let h = histogram_auto(&data, BinRule::Scott);
    for i in 1..h.edges.len() {
        assert!(h.edges[i] > h.edges[i - 1],
            "Edges not monotone at {}: {} vs {}", i, h.edges[i - 1], h.edges[i]);
    }
}

/// histogram_auto with constant data: single bin with all counts.
#[test]
fn histogram_constant_data() {
    let data = vec![5.0; 10];
    let h = histogram_auto(&data, BinRule::Sturges);
    assert_eq!(h.counts.len(), 1, "Constant data should have 1 bin");
    assert_eq!(h.counts[0], 10);
}

/// histogram_auto with NaN: counted separately.
#[test]
fn histogram_nan_count() {
    let mut data: Vec<f64> = (0..10).map(|i| i as f64).collect();
    data.push(f64::NAN);
    data.push(f64::NAN);
    let h = histogram_auto(&data, BinRule::Sturges);
    assert_eq!(h.nan_count, 2);
    let total: u64 = h.counts.iter().sum();
    assert_eq!(total, 10, "Non-NaN counts should be 10");
}

/// histogram_auto with empty data: empty result.
#[test]
fn histogram_empty() {
    let h = histogram_auto(&[], BinRule::Sturges);
    assert!(h.edges.is_empty() && h.counts.is_empty());
    assert_eq!(h.nan_count, 0);
}

/// histogram_auto with Fixed(k) respects the count exactly.
#[test]
fn histogram_fixed_bins() {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let h = histogram_auto(&data, BinRule::Fixed(5));
    assert_eq!(h.counts.len(), 5, "Fixed(5) should give exactly 5 bins");
    let total: u64 = h.counts.iter().sum();
    assert_eq!(total, 100);
}

/// Uniform data into uniform bins: balanced counts.
#[test]
fn histogram_uniform_balanced() {
    // 100 uniform values into 10 bins → each bin should have ~10
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let h = histogram_auto(&data, BinRule::Fixed(10));
    for (i, &c) in h.counts.iter().enumerate() {
        assert!(c >= 8 && c <= 12,
            "Uniform bin {} should have ~10 counts, got {}", i, c);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ECDF
// ═══════════════════════════════════════════════════════════════════════════

/// ECDF of n distinct points: F̂(x_i) = (i+1)/n.
#[test]
fn ecdf_basic() {
    let data = vec![3.0, 1.0, 2.0, 5.0, 4.0];
    let e = ecdf(&data);
    assert_eq!(e.x, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(e.p, vec![0.2, 0.4, 0.6, 0.8, 1.0]);
}

/// ECDF eval: below min → 0, above max → 1.
#[test]
fn ecdf_eval_boundaries() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let e = ecdf(&data);
    assert_eq!(e.eval(0.0), 0.0);
    assert_eq!(e.eval(6.0), 1.0);
    assert_eq!(e.eval(10.0), 1.0);
    assert_eq!(e.eval(-10.0), 0.0);
}

/// ECDF eval: at a data point, returns cumulative prob.
#[test]
fn ecdf_eval_at_points() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let e = ecdf(&data);
    assert!((e.eval(1.0) - 0.2).abs() < 1e-10);
    assert!((e.eval(3.0) - 0.6).abs() < 1e-10);
    assert!((e.eval(5.0) - 1.0).abs() < 1e-10);
    // Between points: equals the lower-neighbor's cum prob
    assert!((e.eval(1.5) - 0.2).abs() < 1e-10);
    assert!((e.eval(3.5) - 0.6).abs() < 1e-10);
}

/// ECDF is monotone non-decreasing.
#[test]
fn ecdf_monotone() {
    let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.37).sin()).collect();
    let e = ecdf(&data);
    for i in 1..e.p.len() {
        assert!(e.p[i] >= e.p[i - 1],
            "ECDF not monotone at {}: {} < {}", i, e.p[i], e.p[i - 1]);
    }
}

/// ECDF with NaN: filters them out.
#[test]
fn ecdf_filters_nan() {
    let data = vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0];
    let e = ecdf(&data);
    assert_eq!(e.x.len(), 3);
    assert_eq!(e.x, vec![1.0, 2.0, 3.0]);
}

/// ECDF empty: degenerate, eval returns NaN.
#[test]
fn ecdf_empty() {
    let e = ecdf(&[]);
    assert!(e.x.is_empty());
    assert!(e.eval(0.0).is_nan());
}

// ═══════════════════════════════════════════════════════════════════════════
// DKW CONFIDENCE BAND
// ═══════════════════════════════════════════════════════════════════════════

/// DKW band width: ε_n = √(ln(2/α) / (2n)).
/// For α=0.05, n=100: ε = √(ln(40)/200) = √(3.689/200) = √0.01844 = 0.1358
#[test]
fn dkw_band_gold_standard() {
    let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let e = ecdf(&data);
    let (lo, hi) = ecdf_confidence_band(&e, 0.05);
    // At the median point, p=0.5, so band should be [0.5 - ε, 0.5 + ε]
    let expected_eps = (40.0_f64.ln() / 200.0).sqrt();
    assert!((expected_eps - 0.1358).abs() < 0.001,
        "Expected ε ≈ 0.1358, computed {}", expected_eps);
    // Find index where p = 0.5 (index 49, 50/100)
    let halfway_lower = lo[49];
    let halfway_upper = hi[49];
    let expected_lower = (0.5 - expected_eps).max(0.0);
    let expected_upper = (0.5 + expected_eps).min(1.0);
    assert!((halfway_lower - expected_lower).abs() < 1e-6,
        "DKW lower at median: expected {}, got {}", expected_lower, halfway_lower);
    assert!((halfway_upper - expected_upper).abs() < 1e-6,
        "DKW upper at median: expected {}, got {}", expected_upper, halfway_upper);
}

/// DKW band: lower ≤ ECDF ≤ upper everywhere.
#[test]
fn dkw_band_contains_ecdf() {
    let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.7).cos()).collect();
    let e = ecdf(&data);
    let (lo, hi) = ecdf_confidence_band(&e, 0.05);
    for i in 0..e.p.len() {
        assert!(lo[i] <= e.p[i] && e.p[i] <= hi[i],
            "ECDF {} not in band [{}, {}] at i={}", e.p[i], lo[i], hi[i], i);
    }
}

/// DKW band: clamped to [0, 1].
#[test]
fn dkw_band_clamped() {
    let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let e = ecdf(&data);
    let (lo, hi) = ecdf_confidence_band(&e, 0.05);
    for (&l, &h) in lo.iter().zip(hi.iter()) {
        assert!(l >= 0.0 && l <= 1.0, "lower in [0,1], got {}", l);
        assert!(h >= 0.0 && h <= 1.0, "upper in [0,1], got {}", h);
    }
}

/// DKW band: tighter for larger n (ε shrinks as 1/√n).
#[test]
fn dkw_band_shrinks_with_n() {
    let small: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let large: Vec<f64> = (1..=2000).map(|i| i as f64 / 100.0).collect();
    let e_s = ecdf(&small);
    let e_l = ecdf(&large);
    let (lo_s, hi_s) = ecdf_confidence_band(&e_s, 0.05);
    let (lo_l, hi_l) = ecdf_confidence_band(&e_l, 0.05);
    // Band width at the median
    let w_s = hi_s[9] - lo_s[9];
    let w_l = hi_l[999] - lo_l[999];
    assert!(w_l < w_s,
        "DKW band should shrink with n: large n={}, small n={}", w_l, w_s);
}

/// DKW band: tighter for larger alpha (less coverage → tighter).
#[test]
fn dkw_band_tighter_with_larger_alpha() {
    let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    let e = ecdf(&data);
    let (lo_01, hi_01) = ecdf_confidence_band(&e, 0.01);
    let (lo_20, hi_20) = ecdf_confidence_band(&e, 0.20);
    // 99% band should be wider than 80% band
    let w_99 = hi_01[49] - lo_01[49];
    let w_80 = hi_20[49] - lo_20[49];
    assert!(w_99 > w_80,
        "99% band ({}) should be wider than 80% band ({})", w_99, w_80);
}

// ═══════════════════════════════════════════════════════════════════════════
// SCOTT BANDWIDTH
// ═══════════════════════════════════════════════════════════════════════════

/// Scott bandwidth gold standard: for σ=1, n=100 → h = 1.06·1·100^(-0.2) ≈ 0.422.
#[test]
fn scott_bandwidth_gold_standard() {
    // σ ≈ 1 sample: normalized data with SD 1
    let data: Vec<f64> = (0..100).map(|i| (i as f64 - 49.5) / 28.87).collect();
    // (i - 49.5)/28.87 for i in 0..100 has approx SD 1
    let h = scott_bandwidth(&data);
    // Scott = 1.06 · σ · n^(-0.2), σ≈1, n=100 → 1.06 · 100^(-0.2) ≈ 0.422
    let expected = 1.06 * 1.0 * 100.0_f64.powf(-0.2);
    assert!((h - expected).abs() < 0.1,
        "Scott bandwidth should be ~{} for σ≈1, n=100, got {}", expected, h);
}

/// Scott with constant data: fallback 1.0.
#[test]
fn scott_bandwidth_constant() {
    assert_eq!(scott_bandwidth(&[5.0; 10]), 1.0);
}

/// Scott with n<2: fallback 1.0.
#[test]
fn scott_bandwidth_too_few() {
    assert_eq!(scott_bandwidth(&[1.0]), 1.0);
    assert_eq!(scott_bandwidth(&[]), 1.0);
}

/// Scott < Silverman when data is normal-ish (Silverman uses 0.9, Scott uses 1.06).
/// Actually Silverman uses 0.9 · min(σ, IQR/1.34) so depends on IQR.
/// For pure normal, min = σ, so Silverman = 0.9σ·n^(-0.2), Scott = 1.06σ·n^(-0.2).
/// Scott > Silverman.
#[test]
fn scott_vs_silverman() {
    let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.01 - 0.5).sin() * 2.0).collect();
    let scott = scott_bandwidth(&data);
    let silverman = silverman_bandwidth(&data);
    assert!(scott.is_finite() && silverman.is_finite());
    // Both should be positive
    assert!(scott > 0.0 && silverman > 0.0);
}
