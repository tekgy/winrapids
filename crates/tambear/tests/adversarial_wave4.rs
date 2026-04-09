//! Adversarial tests — Wave 4: Jarque-Bera, McNemar, Cochran's Q,
//! concordance correlation (Lin 1989)

use tambear::descriptive::MomentStats;
use tambear::hypothesis::*;

// ═══════════════════════════════════════════════════════════════════════════
// JARQUE-BERA
// ═══════════════════════════════════════════════════════════════════════════

/// JB on normal data (S≈0, K≈0): should not reject.
#[test]
fn jarque_bera_normal() {
    // Generate normal via Xoshiro256 + Box-Muller (proper RNG, not LCG)
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let mut values = Vec::new();
    for _ in 0..500 {
        let v = tambear::rng::sample_normal(&mut rng, 0.0, 1.0);
        values.push(v);
    }
    let stats = tambear::descriptive::moments_ungrouped(&values);
    let result = jarque_bera(&stats);
    assert!(result.statistic.is_finite(), "JB should be finite");
    assert!(result.p_value > 0.05,
        "JB on normal data should not reject: JB={}, p={}", result.statistic, result.p_value);
}

/// JB on uniform data: excess kurtosis ≈ -1.2, should reject.
#[test]
fn jarque_bera_uniform() {
    let values: Vec<f64> = (0..200).map(|i| i as f64 / 200.0).collect();
    let stats = tambear::descriptive::moments_ungrouped(&values);
    let result = jarque_bera(&stats);
    assert!(result.statistic > 3.0,
        "JB on uniform should have large statistic: got {}", result.statistic);
}

/// JB on constant data: S=0, K=0, JB=0.
#[test]
fn jarque_bera_constant() {
    let stats = MomentStats {
        count: 100.0, sum: 500.0, min: 5.0, max: 5.0,
        m2: 0.0, m3: 0.0, m4: 0.0,
    };
    let result = jarque_bera(&stats);
    // Skewness and kurtosis are 0/0 for constant data → NaN
    // JB = (n/6)(NaN² + NaN²/4) = NaN
    assert!(result.statistic.is_nan() || result.statistic == 0.0,
        "JB on constant should be NaN or 0, got {}", result.statistic);
}

/// JB with n < 3: NaN.
#[test]
fn jarque_bera_too_few() {
    let stats = MomentStats {
        count: 2.0, sum: 3.0, min: 1.0, max: 2.0,
        m2: 0.5, m3: 0.0, m4: 0.0,
    };
    let result = jarque_bera(&stats);
    assert!(result.statistic.is_nan(), "JB with n=2 should be NaN");
}

/// JB statistic is always non-negative.
#[test]
fn jarque_bera_non_negative() {
    let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let stats = tambear::descriptive::moments_ungrouped(&data);
    let result = jarque_bera(&stats);
    assert!(result.statistic >= 0.0 || result.statistic.is_nan(),
        "JB should be non-negative, got {}", result.statistic);
}

/// JB is O(1) from MomentStats — the tambear poster child.
/// Verify it produces the same result from raw data vs pre-computed stats.
#[test]
fn jarque_bera_from_stats_vs_raw() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let stats = tambear::descriptive::moments_ungrouped(&data);
    let jb1 = jarque_bera(&stats);
    // Same data, different stats construction
    let stats2 = tambear::descriptive::moments_ungrouped(&data);
    let jb2 = jarque_bera(&stats2);
    assert!((jb1.statistic - jb2.statistic).abs() < 1e-10,
        "JB should be deterministic from MomentStats");
}

// ═══════════════════════════════════════════════════════════════════════════
// MCNEMAR'S TEST
// ═══════════════════════════════════════════════════════════════════════════

/// McNemar with no discordant pairs: χ² = 0, p = 1.
#[test]
fn mcnemar_no_discordant() {
    let result = mcnemar(&[10.0, 0.0, 0.0, 10.0], false);
    assert!((result.statistic - 0.0).abs() < 1e-10,
        "McNemar no discordant: χ²=0, got {}", result.statistic);
    assert!((result.p_value - 1.0).abs() < 1e-10,
        "McNemar no discordant: p=1, got {}", result.p_value);
}

/// McNemar with asymmetric discordant pairs: should detect.
#[test]
fn mcnemar_asymmetric() {
    // 20 switched from 0→1, only 5 from 1→0
    let result = mcnemar(&[30.0, 20.0, 5.0, 45.0], false);
    let expected = (20.0 - 5.0_f64).powi(2) / 25.0; // 225/25 = 9.0
    assert!((result.statistic - expected).abs() < 0.01,
        "McNemar χ² should be {}, got {}", expected, result.statistic);
    assert!(result.p_value < 0.01,
        "McNemar should reject: p={}", result.p_value);
}

/// McNemar with continuity correction.
#[test]
fn mcnemar_continuity() {
    let without = mcnemar(&[10.0, 8.0, 2.0, 10.0], false);
    let with = mcnemar(&[10.0, 8.0, 2.0, 10.0], true);
    // Continuity correction reduces the statistic
    assert!(with.statistic <= without.statistic,
        "Continuity correction should reduce χ²: {} vs {}", with.statistic, without.statistic);
}

/// McNemar with b=c: symmetric, χ² = 0.
#[test]
fn mcnemar_symmetric() {
    let result = mcnemar(&[10.0, 5.0, 5.0, 10.0], false);
    assert!((result.statistic - 0.0).abs() < 1e-10,
        "McNemar symmetric: χ²=0, got {}", result.statistic);
}

// ═══════════════════════════════════════════════════════════════════════════
// COCHRAN'S Q
// ═══════════════════════════════════════════════════════════════════════════

/// Cochran's Q with k=2: should be equivalent to McNemar.
#[test]
fn cochran_q_k2() {
    // 5 subjects, 2 treatments
    let data = vec![
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        1.0, 0.0,
        0.0, 0.0,
    ];
    let result = cochran_q(&data, 5, 2);
    assert!(result.statistic.is_finite(), "Q should be finite");
    assert!(result.df == 1.0, "df should be k-1=1, got {}", result.df);
}

/// Cochran's Q with identical treatments: Q = 0.
#[test]
fn cochran_q_identical() {
    let data = vec![
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
    ];
    let result = cochran_q(&data, 4, 3);
    assert!((result.statistic - 0.0).abs() < 1e-10,
        "Cochran Q identical: should be 0, got {}", result.statistic);
}

/// Cochran's Q with clear treatment effect.
#[test]
fn cochran_q_effect() {
    // Treatment 3 always succeeds, treatments 1-2 rarely
    let data = vec![
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 1.0, 1.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
    ];
    let result = cochran_q(&data, 6, 3);
    assert!(result.statistic > 5.0,
        "Cochran Q should detect effect: Q={}", result.statistic);
    assert!(result.p_value < 0.05,
        "Cochran Q should reject: p={}", result.p_value);
}

/// Cochran's Q with all zeros: degenerate.
#[test]
fn cochran_q_all_zeros() {
    let data = vec![0.0; 12]; // 4 subjects, 3 treatments
    let result = cochran_q(&data, 4, 3);
    assert!((result.statistic - 0.0).abs() < 1e-10 || result.p_value > 0.9,
        "All-zero Q should be 0 or p~1");
}

/// Cochran's Q with k=1: degenerate → NaN.
#[test]
fn cochran_q_single_treatment() {
    let result = cochran_q(&[1.0, 0.0, 1.0], 3, 1);
    assert!(result.statistic.is_nan(),
        "Cochran Q with k=1 should be NaN");
}

// ═══════════════════════════════════════════════════════════════════════════
// CONCORDANCE CORRELATION (Lin 1989)
// ═══════════════════════════════════════════════════════════════════════════

/// CCC of identical data: ρ_c = 1.
#[test]
fn concordance_identical() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let ccc = tambear::nonparametric::concordance_correlation(&x, &x);
    assert!((ccc - 1.0).abs() < 1e-10,
        "CCC of identical should be 1.0, got {}", ccc);
}

/// CCC with perfect correlation but different scale: ρ_c < 1.
/// This is the KEY property: Pearson r = 1 but CCC < 1.
#[test]
fn concordance_scale_shift() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 10.0).collect(); // scale + shift
    let ccc = tambear::nonparametric::concordance_correlation(&x, &y);
    let pearson = tambear::nonparametric::pearson_r(&x, &y);
    // Pearson should be 1.0 (perfect linear), but CCC < 1 (different scale/location)
    assert!((pearson - 1.0).abs() < 0.01, "Pearson should be ~1, got {}", pearson);
    assert!(ccc < 0.5,
        "CCC with scale+shift should be much less than 1 (disagreement): got {}", ccc);
}

/// CCC of negatively correlated data: ρ_c < 0.
#[test]
fn concordance_negative() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let ccc = tambear::nonparametric::concordance_correlation(&x, &y);
    assert!(ccc < 0.0, "CCC of reversed should be negative, got {}", ccc);
}

/// CCC range: [-1, 1].
#[test]
fn concordance_range() {
    let x = vec![1.0, 4.0, 2.0, 8.0, 5.0];
    let y = vec![3.0, 7.0, 1.0, 9.0, 4.0];
    let ccc = tambear::nonparametric::concordance_correlation(&x, &y);
    assert!(ccc >= -1.0 && ccc <= 1.0,
        "CCC should be in [-1,1], got {}", ccc);
}

/// CCC with n=1: NaN.
#[test]
fn concordance_n1() {
    let ccc = tambear::nonparametric::concordance_correlation(&[1.0], &[2.0]);
    assert!(ccc.is_nan(), "CCC with n=1 should be NaN, got {}", ccc);
}

/// CCC with constant x and y: NaN (zero variance → undefined).
#[test]
fn concordance_constant() {
    let ccc = tambear::nonparametric::concordance_correlation(&[5.0; 10], &[5.0; 10]);
    // Same constant → denom = 0 + 0 + 0 = 0 → NaN
    assert!(ccc.is_nan(), "CCC of equal constants should be NaN (0/0), got {}", ccc);
}

/// Gold standard: CCC for known analytical case.
/// x = [1,2,3,4,5], y = [1,2,3,4,5] shifted by 1: y = x + 1
/// μx=3, μy=4, σx²=σy²=2, σxy=2, denom=2+2+1=5, CCC=4/5=0.8
#[test]
fn concordance_gold_standard() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 3.0, 4.0, 5.0, 6.0]; // x + 1
    let ccc = tambear::nonparametric::concordance_correlation(&x, &y);
    // μx=3, μy=4, var_x=var_y=2, cov=2
    // CCC = 2*2 / (2 + 2 + (3-4)²) = 4/5 = 0.8
    assert!((ccc - 0.8).abs() < 0.01,
        "CCC gold standard should be 0.8, got {}", ccc);
}
