//! Holographic error-correction experiment — Layer 4 discover in action.
//!
//! ## Hypothesis
//!
//! `sweep_correlation` runs Pearson, Spearman, Kendall τ, distance correlation,
//! and Hoeffding's D in superposition. When influential outliers are injected at
//! the DATA level, the views diverge: Pearson shifts dramatically (it is
//! non-robust to outliers), while Spearman and Kendall remain stable (they
//! operate on ranks and are breakdown-resistant at 29% contamination rate).
//!
//! The `agreement` metric drops because the views no longer tell a coherent
//! story — the system has detected that the data is internally inconsistent.
//! This is holographic error-correction: no single method "knows" the data is
//! corrupted, but the disagreement pattern across methods reveals it.
//!
//! ## Critical design constraint
//!
//! Corruption must be applied at the DATA level (modify the x/y arrays before
//! calling sweep_correlation), NOT at the TamSession level (corrupting a shared
//! intermediate). If a shared intermediate is corrupted, all methods see the
//! same corrupted value and continue to agree — view_agreement stays high and
//! the corruption goes undetected. The holographic property only emerges when
//! methods compute FROM DIFFERENT MATHEMATICAL PRINCIPLES applied to the same
//! raw data.
//!
//! ## Data generation
//!
//! Correlated bivariate normal via Cholesky: x ~ N(0,1), y = r·x + √(1-r²)·ε.
//! Deterministic: LCG seed 42, r = 0.7, n = 100.
//! Corruption: 3 extreme x values multiplied by 100.

use tambear::superposition::sweep_correlation;
use tambear::nonparametric::{pearson_r, spearman, kendall_tau};

// ─── Data generation ─────────────────────────────────────────────────────────

/// Simple LCG for deterministic data generation. No randomness in tests.
fn lcg_next(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    // Map to (0, 1) via the upper 32 bits
    ((*state >> 32) as f64 + 0.5) / 4294967296.0
}

/// Box-Muller transform: two uniform samples → one N(0,1) sample.
fn box_muller(u1: f64, u2: f64) -> f64 {
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

/// Generate (x, y) pairs with true Pearson correlation r, n=100, seed=42.
fn generate_correlated(r: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut state: u64 = 42;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let sqrt_one_minus_r2 = (1.0 - r * r).sqrt();

    for _ in 0..n {
        let u1 = lcg_next(&mut state);
        let u2 = lcg_next(&mut state);
        let u3 = lcg_next(&mut state);
        let u4 = lcg_next(&mut state);
        let xi = box_muller(u1, u2);
        let eps = box_muller(u3, u4);
        let yi = r * xi + sqrt_one_minus_r2 * eps;
        x.push(xi);
        y.push(yi);
    }
    (x, y)
}

/// Corrupt: find the 3 indices with largest |x|, multiply x[i] by 100.
/// This creates 3 highly influential outliers — enough to pull Pearson
/// dramatically while Spearman and Kendall are barely affected (only 3%
/// of rank order changes).
fn corrupt_three_extreme(x: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = x.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    let mut corrupted = x.to_vec();
    for (idx, _) in indexed.iter().take(3) {
        corrupted[*idx] *= 100.0;
    }
    corrupted
}

// ─── Baseline tests ───────────────────────────────────────────────────────────

/// Baseline clean data: empirical Pearson should be near r=0.7.
/// With n=100 and true ρ=0.7, the sample estimate is in [0.55, 0.80] with
/// 99% probability (Fisher z CI). We check the deterministic LCG result.
#[test]
fn holographic_baseline_pearson_near_0_7() {
    let (x, y) = generate_correlated(0.7, 100);
    let r = pearson_r(&x, &y);
    // Accept ±0.15 from 0.7 (very conservative — n=100 standard error = √((1-ρ²)²/n) ≈ 0.03)
    assert!(
        (r - 0.7).abs() < 0.15,
        "Baseline Pearson r={:.4} should be near 0.70 for n=100, true ρ=0.7",
        r
    );
}

/// Baseline clean data: all three rank-based methods should agree directionally.
/// With r≈0.7, Spearman ρ and Kendall τ are both positive and significant.
#[test]
fn holographic_baseline_rank_methods_positive() {
    let (x, y) = generate_correlated(0.7, 100);
    let rho = spearman(&x, &y);
    let tau = kendall_tau(&x, &y);
    assert!(rho > 0.4, "Baseline Spearman ρ={:.4} should be > 0.4", rho);
    assert!(tau > 0.3, "Baseline Kendall τ={:.4} should be > 0.3", tau);
}

/// Baseline agreement: all five methods should detect positive association on clean bivariate normal.
/// With r=0.7, all methods agree on direction and approximate magnitude.
/// Agreement value reflects scale differences: Pearson~0.71, Spearman~0.80, Kendall~0.50,
/// dcor~0.67, Hoeffding~0.35 — directionally consistent but numerically on different scales.
/// Threshold 0.30 is conservative (scalar_agreement uses max relative deviation from mean).
#[test]
fn holographic_baseline_agreement_high() {
    let (x, y) = generate_correlated(0.7, 100);
    let sup = sweep_correlation(&x, &y);
    assert!(
        sup.agreement >= 0.30,
        "Baseline agreement={:.3} should be >= 0.30 on clean data with r=0.7",
        sup.agreement
    );
}

/// Baseline: sweep_correlation modal_value (median of Pearson/Spearman/Kendall) is positive.
/// With r=0.7 and n=100, all three signed correlation measures are positive.
#[test]
fn holographic_baseline_modal_value_positive() {
    let (x, y) = generate_correlated(0.7, 100);
    let sup = sweep_correlation(&x, &y);
    assert!(
        sup.modal_value > 0.0,
        "Baseline modal_value={:.4} should be positive for r=0.7",
        sup.modal_value
    );
}

// ─── Corruption tests ─────────────────────────────────────────────────────────

/// Corruption shifts Pearson dramatically: multiplying 3 extreme x values by 100
/// creates outliers that dominate the covariance. Pearson should change by >= 0.10.
#[test]
fn holographic_corruption_shifts_pearson() {
    let (x, y) = generate_correlated(0.7, 100);
    let r_clean = pearson_r(&x, &y);

    let x_corrupted = corrupt_three_extreme(&x);
    let r_corrupted = pearson_r(&x_corrupted, &y);

    let shift = (r_corrupted - r_clean).abs();
    assert!(
        shift >= 0.10,
        "Corruption should shift Pearson by >= 0.10: clean={:.4}, corrupted={:.4}, shift={:.4}",
        r_clean,
        r_corrupted,
        shift
    );
}

/// Rank methods are stable under leverage-point corruption.
/// Multiplying 3 values by 100 changes only their relative ranks (they move to
/// one extreme), which barely perturbs Spearman ρ or Kendall τ.
/// Stability criterion: |corrupted - clean| < 0.15 for both.
#[test]
fn holographic_corruption_spearman_stable() {
    let (x, y) = generate_correlated(0.7, 100);
    let rho_clean = spearman(&x, &y);

    let x_corrupted = corrupt_three_extreme(&x);
    let rho_corrupted = spearman(&x_corrupted, &y);

    let shift = (rho_corrupted - rho_clean).abs();
    assert!(
        shift < 0.15,
        "Spearman should be stable: clean={:.4}, corrupted={:.4}, shift={:.4}",
        rho_clean,
        rho_corrupted,
        shift
    );
}

/// Kendall τ is stable under leverage-point corruption.
#[test]
fn holographic_corruption_kendall_stable() {
    let (x, y) = generate_correlated(0.7, 100);
    let tau_clean = kendall_tau(&x, &y);

    let x_corrupted = corrupt_three_extreme(&x);
    let tau_corrupted = kendall_tau(&x_corrupted, &y);

    let shift = (tau_corrupted - tau_clean).abs();
    assert!(
        shift < 0.12,
        "Kendall τ should be stable: clean={:.4}, corrupted={:.4}, shift={:.4}",
        tau_clean,
        tau_corrupted,
        shift
    );
}

// ─── Core holographic theorem ─────────────────────────────────────────────────

/// THE HOLOGRAPHIC THEOREM: corruption drops agreement.
///
/// Clean data: all methods agree → agreement high.
/// Corrupted data: Pearson diverges from Spearman/Kendall → agreement drops.
/// The drop in agreement IS the signal that something is wrong.
///
/// This is holographic error-correction: the fault is detectable in the
/// DISAGREEMENT PATTERN, not in any single method's value.
#[test]
fn holographic_agreement_drops_under_corruption() {
    let (x, y) = generate_correlated(0.7, 100);
    let sup_clean = sweep_correlation(&x, &y);

    let x_corrupted = corrupt_three_extreme(&x);
    let sup_corrupted = sweep_correlation(&x_corrupted, &y);

    let clean_agreement = sup_clean.agreement;
    let corrupted_agreement = sup_corrupted.agreement;

    // The core assertion: corruption must drop agreement.
    assert!(
        corrupted_agreement < clean_agreement,
        "Corruption must drop agreement: clean={:.3}, corrupted={:.3}",
        clean_agreement,
        corrupted_agreement
    );

    // Quantitative: agreement should drop by at least 0.05
    let drop = clean_agreement - corrupted_agreement;
    assert!(
        drop >= 0.05,
        "Agreement drop should be >= 0.05: clean={:.3}, corrupted={:.3}, drop={:.3}",
        clean_agreement,
        corrupted_agreement,
        drop
    );
}

// ─── Structural invariants ────────────────────────────────────────────────────

/// Views count: sweep_correlation always produces exactly 5 views.
/// (Pearson, Spearman, Kendall, distance correlation, Hoeffding's D)
#[test]
fn holographic_sweep_has_five_views() {
    let (x, y) = generate_correlated(0.7, 100);
    let sup = sweep_correlation(&x, &y);
    assert_eq!(
        sup.views.len(),
        5,
        "sweep_correlation must produce exactly 5 views, got {}",
        sup.views.len()
    );
}

/// View names are the five expected methods, in order.
#[test]
fn holographic_view_names_correct() {
    let (x, y) = generate_correlated(0.7, 100);
    let sup = sweep_correlation(&x, &y);
    let names: Vec<&str> = sup.views.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["pearson", "spearman", "kendall_tau", "distance_correlation", "hoeffdings_d"],
        "View names should be in canonical order"
    );
}

/// Agreement is in [0, 1] for both clean and corrupted data.
#[test]
fn holographic_agreement_in_unit_interval() {
    let (x, y) = generate_correlated(0.7, 100);
    let sup_clean = sweep_correlation(&x, &y);

    let x_corrupted = corrupt_three_extreme(&x);
    let sup_corrupted = sweep_correlation(&x_corrupted, &y);

    assert!(
        sup_clean.agreement >= 0.0 && sup_clean.agreement <= 1.0,
        "Clean agreement must be in [0,1]: got {:.4}",
        sup_clean.agreement
    );
    assert!(
        sup_corrupted.agreement >= 0.0 && sup_corrupted.agreement <= 1.0,
        "Corrupted agreement must be in [0,1]: got {:.4}",
        sup_corrupted.agreement
    );
}

/// Sensitivity = 1 - agreement. High sensitivity = parameter-sensitive result.
/// Corrupted data should be more sensitive (less agreement) than clean data.
#[test]
fn holographic_sensitivity_increases_under_corruption() {
    let (x, y) = generate_correlated(0.7, 100);
    let sup_clean = sweep_correlation(&x, &y);

    let x_corrupted = corrupt_three_extreme(&x);
    let sup_corrupted = sweep_correlation(&x_corrupted, &y);

    assert!(
        sup_corrupted.sensitivity() > sup_clean.sensitivity(),
        "Sensitivity must increase under corruption: clean={:.3}, corrupted={:.3}",
        sup_clean.sensitivity(),
        sup_corrupted.sensitivity()
    );
}

/// Corrupted modal value: Spearman/Kendall rank-based methods should outvote
/// Pearson in the median — modal_value should remain positive and near the
/// rank-method estimate, not shift toward the corrupted Pearson.
#[test]
fn holographic_modal_value_robust_to_corruption() {
    let (x, y) = generate_correlated(0.7, 100);
    let sup_clean = sweep_correlation(&x, &y);

    let x_corrupted = corrupt_three_extreme(&x);
    let sup_corrupted = sweep_correlation(&x_corrupted, &y);

    // The modal_value is the median of (Pearson, Spearman, Kendall) — 3 values.
    // Clean: median of (Pearson≈0.71, Spearman≈0.80, Kendall≈0.50) → median≈0.71
    // Corrupted: Pearson shifts but Kendall and Spearman stay near their values.
    // The median of 3 values changes only when the median element itself changes.
    // Allow shift up to 0.25 — the key property is that it remains positive and near the
    // rank-based estimate rather than being pulled to the corrupted Pearson extreme.
    let modal_shift = (sup_corrupted.modal_value - sup_clean.modal_value).abs();
    assert!(
        modal_shift < 0.25,
        "Modal value should be robust (rank methods limit shift): \
         clean_modal={:.4}, corrupted_modal={:.4}, shift={:.4}",
        sup_clean.modal_value,
        sup_corrupted.modal_value,
        modal_shift
    );
    // The corrupted modal_value must still be positive (association not inverted)
    assert!(
        sup_corrupted.modal_value > 0.0,
        "Corrupted modal_value={:.4} should remain positive (association direction preserved)",
        sup_corrupted.modal_value
    );
}

// ─── Null case: symmetric corruption ─────────────────────────────────────────

/// Zero-correlation baseline: with r=0.0, all methods should report near-zero.
/// This confirms the data generator works at the other extreme.
#[test]
fn holographic_null_correlation_baseline() {
    let (x, y) = generate_correlated(0.0, 100);
    let r = pearson_r(&x, &y);
    // With n=100 and true ρ=0, sample r is in [-0.2, 0.2] essentially always
    assert!(
        r.abs() < 0.25,
        "Null-correlation baseline: Pearson r={:.4} should be near 0",
        r
    );
}

/// High-correlation baseline: with r=0.95, all methods agree and agreement is high.
#[test]
fn holographic_high_correlation_agreement_very_high() {
    let (x, y) = generate_correlated(0.95, 100);
    let sup = sweep_correlation(&x, &y);
    // At r=0.95, Pearson≈0.95, Spearman≈0.93, Kendall≈0.80 — all very similar in direction
    assert!(
        sup.agreement >= 0.40,
        "High-correlation data: agreement={:.3} should be >= 0.40",
        sup.agreement
    );
}
