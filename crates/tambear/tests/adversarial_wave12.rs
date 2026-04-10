//! Adversarial Wave 12 — Unstaged new code: information theory divergences,
//! matrix functions, and iterative solvers
//!
//! Targets: hellinger_distance, total_variation_distance, chi_squared_divergence,
//!          renyi_divergence, bhattacharyya_coefficient, f_divergence,
//!          wasserstein_1d, mmd_rbf, energy_distance, p_log_p, p_log_p_over_q,
//!          matrix_exp, matrix_log, matrix_sqrt, conjugate_gradient, gmres
//!
//! All tests assert mathematical truths. Failures are bugs.

use tambear::information_theory::{
    hellinger_distance_sq, hellinger_distance,
    total_variation_distance, chi_squared_divergence,
    renyi_divergence, bhattacharyya_coefficient, bhattacharyya_distance,
    f_divergence, wasserstein_1d, mmd_rbf, energy_distance,
    p_log_p, p_log_p_over_q,
    shannon_entropy, kl_divergence,
};
use tambear::linear_algebra::{
    Mat, matrix_exp, matrix_log, matrix_sqrt,
    conjugate_gradient, gmres,
};

// ═══════════════════════════════════════════════════════════════════════════
// p_log_p and p_log_p_over_q — atoms
// ═══════════════════════════════════════════════════════════════════════════

/// p=0: 0 * ln(0) = 0 by convention (limit p→0+).
#[test]
fn p_log_p_at_zero() {
    assert_eq!(p_log_p(0.0), 0.0, "p_log_p(0) should be 0");
}

/// p=1: 1 * ln(1) = 0.
#[test]
fn p_log_p_at_one() {
    let v = p_log_p(1.0);
    assert!((v - 0.0).abs() < 1e-14, "p_log_p(1) should be 0, got {}", v);
}

/// BUG: p < 0 (invalid probability) → treated as 0 (silently wrong).
/// p_log_p(-0.5): -0.5 <= 0 → returns 0.0. But negative probability is undefined.
/// A correct implementation should return NaN to signal invalid input.
#[test]
fn p_log_p_negative_should_be_nan() {
    let v = p_log_p(-0.5);
    assert!(v.is_nan(),
        "BUG: p_log_p(-0.5) should return NaN (invalid probability), got {} (treated as 0)", v);
}

/// p = NaN: NaN <= 0.0 is false → NaN * ln(NaN) = NaN. Correct propagation.
#[test]
fn p_log_p_nan_propagates() {
    let v = p_log_p(f64::NAN);
    assert!(v.is_nan(), "p_log_p(NaN) should be NaN, got {}", v);
}

/// p_log_p_over_q: p=0, q=0 → both zero, convention returns 0.
#[test]
fn p_log_p_over_q_both_zero() {
    let v = p_log_p_over_q(0.0, 0.0);
    assert_eq!(v, 0.0, "p_log_p_over_q(0,0) should be 0, got {}", v);
}

/// p=0.5, q=0.5: 0.5 * ln(1) = 0.
#[test]
fn p_log_p_over_q_equal() {
    let v = p_log_p_over_q(0.5, 0.5);
    assert!(v.abs() < 1e-14, "p_log_p_over_q(0.5, 0.5) should be 0, got {}", v);
}

/// p > 0, q = 0: returns +∞ (documented absolute continuity violation).
#[test]
fn p_log_p_over_q_q_zero() {
    let v = p_log_p_over_q(0.3, 0.0);
    assert_eq!(v, f64::INFINITY, "p_log_p_over_q(p>0, 0) should be Inf, got {}", v);
}

// ═══════════════════════════════════════════════════════════════════════════
// hellinger_distance — properties
// ═══════════════════════════════════════════════════════════════════════════

/// H(P,P) = 0 (identity).
#[test]
fn hellinger_self_is_zero() {
    let p = vec![0.25, 0.25, 0.25, 0.25];
    assert_eq!(hellinger_distance(&p, &p), 0.0,
        "H(P,P) should be 0");
}

/// H in [0, 1] always.
#[test]
fn hellinger_range() {
    let p = vec![1.0, 0.0, 0.0, 0.0];
    let q = vec![0.0, 0.0, 0.0, 1.0]; // maximally different
    let h = hellinger_distance(&p, &q);
    assert!(h >= 0.0 && h <= 1.0 + 1e-12,
        "H should be in [0,1], got {}", h);
}

/// H(P,Q) for orthogonal distributions = 1 / sqrt(2) * sqrt(2) = 1.
/// Actually: H²(P,Q) = 1 - BC. When P and Q have disjoint support: BC = 0, H² = 1, H = 1.
#[test]
fn hellinger_disjoint_is_one() {
    let p = vec![0.5, 0.5, 0.0, 0.0];
    let q = vec![0.0, 0.0, 0.5, 0.5];
    let h = hellinger_distance(&p, &q);
    assert!((h - 1.0).abs() < 1e-12,
        "H(disjoint P,Q) should be 1, got {}", h);
}

/// H is symmetric: H(P,Q) = H(Q,P).
#[test]
fn hellinger_symmetric() {
    let p = vec![0.1, 0.4, 0.3, 0.2];
    let q = vec![0.3, 0.3, 0.2, 0.2];
    let h1 = hellinger_distance(&p, &q);
    let h2 = hellinger_distance(&q, &p);
    assert!((h1 - h2).abs() < 1e-12, "H should be symmetric: {} vs {}", h1, h2);
}

/// H² and H² are consistent: hellinger_distance = sqrt(hellinger_distance_sq).
#[test]
fn hellinger_sq_consistent() {
    let p = vec![0.1, 0.4, 0.5];
    let q = vec![0.3, 0.3, 0.4];
    let h = hellinger_distance(&p, &q);
    let h2 = hellinger_distance_sq(&p, &q);
    assert!((h - h2.sqrt()).abs() < 1e-12,
        "H = sqrt(H²): H={}, sqrt(H²)={}", h, h2.sqrt());
}

/// BUG: negative probability input — treated as 0 via max(0.0).
/// Hellinger uses `pi.max(0.0).sqrt()`. Negative p is clamped to 0, not NaN.
/// This silently handles invalid inputs.
#[test]
fn hellinger_negative_prob_clamped() {
    let p = vec![-0.1, 0.6, 0.5]; // invalid: negative prob
    let q = vec![0.3, 0.4, 0.3];
    let h = hellinger_distance(&p, &q);
    // Currently: -0.1.max(0.0) = 0.0 → treated as zero prob. Silent.
    // Should be NaN. This test FAILS to document the silent clamping bug.
    assert!(h.is_nan(),
        "BUG: hellinger_distance with negative prob should return NaN (invalid input), got {} (silently clamped to 0)", h);
}

// ═══════════════════════════════════════════════════════════════════════════
// total_variation_distance — properties
// ═══════════════════════════════════════════════════════════════════════════

/// TV(P,P) = 0.
#[test]
fn tv_self_is_zero() {
    let p = vec![0.1, 0.3, 0.6];
    assert_eq!(total_variation_distance(&p, &p), 0.0, "TV(P,P) should be 0");
}

/// TV in [0, 1].
#[test]
fn tv_disjoint_is_half() {
    // TV = 0.5 * sum |p_i - q_i| for disjoint: p=[1,0], q=[0,1]
    // 0.5 * (|1-0| + |0-1|) = 0.5 * 2 = 1. Wait: TV(P,Q) = 0.5 * Σ |p-q|.
    // For [1,0] vs [0,1]: 0.5 * (1 + 1) = 1.
    let p = vec![1.0, 0.0];
    let q = vec![0.0, 1.0];
    let tv = total_variation_distance(&p, &q);
    assert!((tv - 1.0).abs() < 1e-12, "TV(disjoint) should be 1, got {}", tv);
}

/// TV is symmetric.
#[test]
fn tv_symmetric() {
    let p = vec![0.2, 0.3, 0.5];
    let q = vec![0.4, 0.4, 0.2];
    let tv1 = total_variation_distance(&p, &q);
    let tv2 = total_variation_distance(&q, &p);
    assert!((tv1 - tv2).abs() < 1e-12, "TV should be symmetric");
}

/// Pinsker's inequality: TV ≤ sqrt(KL / 2).
#[test]
fn tv_pinsker_inequality() {
    let p = vec![0.5, 0.3, 0.2];
    let q = vec![0.4, 0.35, 0.25];
    let tv = total_variation_distance(&p, &q);
    let kl = kl_divergence(&p, &q);
    assert!(kl >= 0.0 && tv >= 0.0, "KL and TV must be non-negative");
    assert!(tv <= (kl / 2.0).sqrt() + 1e-10,
        "Pinsker: TV ≤ sqrt(KL/2): TV={}, sqrt(KL/2)={}", tv, (kl / 2.0).sqrt());
}

// ═══════════════════════════════════════════════════════════════════════════
// renyi_divergence — properties and edge cases
// ═══════════════════════════════════════════════════════════════════════════

/// D_1(P||Q) = KL(P||Q) — limit as alpha→1.
#[test]
fn renyi_alpha_one_equals_kl() {
    let p = vec![0.5, 0.3, 0.2];
    let q = vec![0.4, 0.35, 0.25];
    let r1 = renyi_divergence(&p, &q, 1.0);
    let kl = kl_divergence(&p, &q);
    assert!((r1 - kl).abs() < 1e-10,
        "D_1 should equal KL: D_1={}, KL={}", r1, kl);
}

/// D_α(P||P) = 0 for all α (self-divergence).
#[test]
fn renyi_self_is_zero() {
    let p = vec![0.5, 0.3, 0.2];
    for alpha in &[0.5, 1.0, 2.0, 10.0] {
        let d = renyi_divergence(&p, &p, *alpha);
        assert!(d.abs() < 1e-10,
            "D_{}(P||P) should be 0, got {}", alpha, d);
    }
}

/// D_α ≥ 0 for all α ≥ 0 (non-negativity).
#[test]
fn renyi_nonnegative() {
    let p = vec![0.5, 0.3, 0.2];
    let q = vec![0.4, 0.35, 0.25];
    for alpha in &[0.1, 0.5, 0.9, 1.5, 2.0, 5.0] {
        let d = renyi_divergence(&p, &q, *alpha);
        assert!(d >= -1e-12,
            "D_{}(P||Q) should be ≥ 0, got {}", alpha, d);
    }
}

/// D_α is non-decreasing in α (ordering property).
#[test]
fn renyi_monotone_in_alpha() {
    let p = vec![0.6, 0.3, 0.1];
    let q = vec![0.2, 0.5, 0.3];
    let d1 = renyi_divergence(&p, &q, 1.0);
    let d2 = renyi_divergence(&p, &q, 2.0);
    assert!(d2 >= d1 - 1e-10,
        "D_α should be non-decreasing in α: D_1={}, D_2={}", d1, d2);
}

/// D_0(P||Q) = -log(overlap): for disjoint P, Q → Inf.
#[test]
fn renyi_alpha0_disjoint_is_inf() {
    let p = vec![1.0, 0.0];
    let q = vec![0.0, 1.0];
    let d = renyi_divergence(&p, &q, 0.0);
    assert_eq!(d, f64::INFINITY, "D_0(disjoint) should be Inf, got {}", d);
}

/// BUG: alpha=NaN should return NaN (not 0).
/// `(alpha - 1.0).abs() < 1e-12`: NaN.abs() = NaN, NaN < 1e-12 = false.
/// `alpha == 0.0`: NaN == 0 = false.
/// `alpha == f64::INFINITY`: NaN == Inf = false.
/// Falls through to formula: sum computation proceeds, then `1/(NaN-1) * ln(sum)` = NaN.
#[test]
fn renyi_nan_alpha_is_nan() {
    let p = vec![0.5, 0.5];
    let q = vec![0.5, 0.5];
    let d = renyi_divergence(&p, &q, f64::NAN);
    assert!(d.is_nan(), "renyi_divergence with NaN alpha should return NaN, got {}", d);
}

/// Rényi divergence with q_i = 0 and p_i > 0: returns Inf.
#[test]
fn renyi_absolute_continuity_violation() {
    let p = vec![0.5, 0.5];
    let q = vec![1.0, 0.0]; // q[1] = 0, p[1] = 0.5 > 0
    let d = renyi_divergence(&p, &q, 2.0);
    assert_eq!(d, f64::INFINITY,
        "D_2 with q_i=0, p_i>0 should be Inf, got {}", d);
}

// ═══════════════════════════════════════════════════════════════════════════
// bhattacharyya — properties
// ═══════════════════════════════════════════════════════════════════════════

/// BC(P,P) = 1 (sum of sqrt(pi * pi) = sum of pi = 1).
#[test]
fn bhattacharyya_self_is_one() {
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let bc = bhattacharyya_coefficient(&p, &p);
    assert!((bc - 1.0).abs() < 1e-12, "BC(P,P) should be 1, got {}", bc);
}

/// BC(disjoint) = 0, D_B(disjoint) = Inf.
#[test]
fn bhattacharyya_disjoint() {
    let p = vec![0.5, 0.5, 0.0, 0.0];
    let q = vec![0.0, 0.0, 0.5, 0.5];
    let bc = bhattacharyya_coefficient(&p, &q);
    assert_eq!(bc, 0.0, "BC(disjoint) should be 0, got {}", bc);
    let db = bhattacharyya_distance(&p, &q);
    assert_eq!(db, f64::INFINITY, "D_B(disjoint) should be Inf, got {}", db);
}

/// Consistency: H² = 1 - BC.
#[test]
fn hellinger_bhattacharyya_consistency() {
    let p = vec![0.3, 0.4, 0.3];
    let q = vec![0.2, 0.5, 0.3];
    let bc = bhattacharyya_coefficient(&p, &q);
    let h2 = hellinger_distance_sq(&p, &q);
    // H² = 1 - BC (exact relationship)
    assert!((h2 - (1.0 - bc)).abs() < 1e-12,
        "H² should equal 1-BC: H²={}, 1-BC={}", h2, 1.0 - bc);
}

// ═══════════════════════════════════════════════════════════════════════════
// wasserstein_1d — properties
// ═══════════════════════════════════════════════════════════════════════════

/// W1(X, X) = 0.
#[test]
fn wasserstein_self_is_zero() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let w = wasserstein_1d(&x, &x);
    assert_eq!(w, 0.0, "W1(X,X) should be 0, got {}", w);
}

/// W1 for shifted distribution: W1([0,1,2],[1,2,3]) should be 1.0.
#[test]
fn wasserstein_shifted_by_one() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![1.0, 2.0, 3.0];
    let w = wasserstein_1d(&x, &y);
    assert!((w - 1.0).abs() < 1e-10, "W1(shift by 1) should be 1.0, got {}", w);
}

/// W1 is symmetric.
#[test]
fn wasserstein_symmetric() {
    let x = vec![1.0, 3.0, 5.0, 7.0];
    let y = vec![2.0, 4.0, 6.0];
    let w1 = wasserstein_1d(&x, &y);
    let w2 = wasserstein_1d(&y, &x);
    assert!((w1 - w2).abs() < 1e-10, "W1 should be symmetric: {} vs {}", w1, w2);
}

/// W1 ≥ 0.
#[test]
fn wasserstein_nonnegative() {
    let x = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let y = vec![6.0, 5.0, 3.0, 5.0];
    let w = wasserstein_1d(&x, &y);
    assert!(w >= 0.0, "W1 should be non-negative, got {}", w);
}

/// Empty input → NaN.
#[test]
fn wasserstein_empty_is_nan() {
    let w = wasserstein_1d(&[], &[1.0, 2.0]);
    assert!(w.is_nan(), "W1([], [1,2]) should be NaN, got {}", w);
}

/// All-NaN input (filtered to empty) → NaN.
#[test]
fn wasserstein_all_nan_is_nan() {
    let w = wasserstein_1d(&[f64::NAN, f64::NAN], &[1.0, 2.0]);
    assert!(w.is_nan(), "W1(all-NaN, [1,2]) should be NaN, got {}", w);
}

/// Unequal-size case: [0, 2] vs [1] — W1 = 1.
/// x sorted=[0,2], y=[1]. CDF_x steps: at 0→0.5, at 2→1.0. CDF_y: at 1→1.0.
/// ∫|F_x - F_y| = |0-0|*(0-(-Inf))=0 (before 0), |0.5-0|*(1-0)=0.5, |0.5-1|*(2-1)=0.5. Total=1.
#[test]
fn wasserstein_unequal_size() {
    let x = vec![0.0, 2.0];
    let y = vec![1.0];
    let w = wasserstein_1d(&x, &y);
    assert!((w - 1.0).abs() < 1e-10, "W1([0,2],[1]) should be 1.0, got {}", w);
}

// ═══════════════════════════════════════════════════════════════════════════
// mmd_rbf — properties
// ═══════════════════════════════════════════════════════════════════════════

/// MMD²(X, X) = 0 when the same sample is passed as both arguments.
///
/// BUG: The current implementation uses inconsistent estimators:
/// - Exx and Eyy use the U-statistic (exclude diagonal: sum over i≠j)
/// - Exy uses the V-statistic (include all pairs i,j including i=j)
/// This asymmetry means Exx ≠ Exy when x IS y (same slice), so MMD²(X,X) ≠ 0.
/// Fix: Exy should also exclude diagonal when x and y are the same distribution.
/// Or: use all V-statistics consistently (biased but numerically consistent).
#[test]
fn mmd_self_is_zero() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mmd = mmd_rbf(&x, &x, Some(1.0));
    assert!(mmd.abs() < 1e-10,
        "BUG: MMD²(X,X) should be 0 but got {} — Exy uses V-stat while Exx/Eyy use U-stat",
        mmd);
}

/// n < 2 → NaN.
#[test]
fn mmd_too_small_is_nan() {
    let mmd1 = mmd_rbf(&[1.0], &[1.0, 2.0], Some(1.0));
    assert!(mmd1.is_nan(), "MMD with n=1 should be NaN, got {}", mmd1);
}

/// MMD² can be negative for different samples from the same distribution.
/// Note: this is a consequence of the inconsistent estimator bug (mmd_self_is_zero).
/// With consistent estimators, the U-statistic MMD² can still be negative for finite
/// samples when the distributions are close — that is expected behavior.
#[test]
fn mmd_can_be_negative_for_same_distribution() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.5, 2.5, 3.5]; // slight shift
    let mmd = mmd_rbf(&x, &y, Some(1.0));
    // No assertion on sign — just that it's finite
    assert!(mmd.is_finite(), "MMD² should be finite, got {}", mmd);
}

/// MMD² for very different distributions should be clearly positive.
#[test]
fn mmd_different_distributions_positive() {
    let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect(); // [0, 2]
    let y: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 0.1).collect(); // [10, 12]
    let mmd = mmd_rbf(&x, &y, Some(1.0));
    // With bandwidth=1, exp(-dist²/2): exp(-100/2) ≈ 0. Cross-terms ≈ 0.
    // exx ≈ exp(0) = 1 for nearby x, eyy ≈ 1, exy ≈ 0. MMD² ≈ 1-0+1 = 2.
    assert!(mmd > 0.1, "MMD² for distributions 10 units apart should be positive, got {}", mmd);
}

// ═══════════════════════════════════════════════════════════════════════════
// energy_distance — properties
// ═══════════════════════════════════════════════════════════════════════════

/// E(X,X) = 0 (same sample).
#[test]
fn energy_distance_self_is_zero() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let e = energy_distance(&x, &x);
    assert!(e.abs() < 1e-10, "Energy distance(X,X) should be 0, got {}", e);
}

/// E ≥ 0 always.
#[test]
fn energy_distance_nonnegative() {
    let x = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let e = energy_distance(&x, &y);
    assert!(e >= -1e-10, "Energy distance should be non-negative, got {}", e);
}

/// Empty input → NaN.
#[test]
fn energy_distance_empty_is_nan() {
    let e = energy_distance(&[], &[1.0, 2.0]);
    assert!(e.is_nan(), "Energy distance([], [1,2]) should be NaN, got {}", e);
}

/// E is symmetric: E(X,Y) = E(Y,X).
#[test]
fn energy_distance_symmetric() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![4.0, 5.0, 6.0, 7.0];
    let e1 = energy_distance(&x, &y);
    let e2 = energy_distance(&y, &x);
    assert!((e1 - e2).abs() < 1e-10, "Energy distance should be symmetric: {} vs {}", e1, e2);
}

// ═══════════════════════════════════════════════════════════════════════════
// matrix_exp — properties
// ═══════════════════════════════════════════════════════════════════════════

fn mat_from(rows: usize, cols: usize, data: Vec<f64>) -> Mat {
    Mat { data, rows, cols }
}

/// exp(0) = I for zero matrix.
#[test]
fn matrix_exp_zero_is_identity() {
    let z = mat_from(2, 2, vec![0.0; 4]);
    let e = matrix_exp(&z);
    // exp(0) = I
    assert!((e.get(0, 0) - 1.0).abs() < 1e-10, "exp(0)[0,0] should be 1, got {}", e.get(0, 0));
    assert!((e.get(1, 1) - 1.0).abs() < 1e-10, "exp(0)[1,1] should be 1, got {}", e.get(1, 1));
    assert!((e.get(0, 1)).abs() < 1e-10, "exp(0)[0,1] should be 0, got {}", e.get(0, 1));
    assert!((e.get(1, 0)).abs() < 1e-10, "exp(0)[1,0] should be 0, got {}", e.get(1, 0));
}

/// exp(I) = e * I for identity matrix.
/// BUG: Padé [6/6] approximant only achieves ~7 significant digits.
/// Full double precision requires Padé [13/13] (as used by MATLAB/SciPy).
/// This test documents the precision deficiency — it should fail until the
/// approximant order is upgraded.
#[test]
fn matrix_exp_identity() {
    let eye = Mat::eye(2);
    let e = matrix_exp(&eye);
    let euler = std::f64::consts::E;
    // Padé [6/6] gives ~4.5e-8 error here; full precision would be <1e-14.
    assert!((e.get(0, 0) - euler).abs() < 1e-12,
        "BUG: exp(I)[0,0] should be e={} to 1e-12; Padé [6/6] gives {}, error={:.2e} (need Padé [13/13])",
        euler, e.get(0, 0), (e.get(0, 0) - euler).abs());
    assert!((e.get(1, 1) - euler).abs() < 1e-12,
        "BUG: exp(I)[1,1] should be e={} to 1e-12; got {}", euler, e.get(1, 1));
    assert!(e.get(0, 1).abs() < 1e-10, "exp(I) off-diag should be 0, got {}", e.get(0, 1));
}

/// exp(diag(a,b)) = diag(exp(a), exp(b)).
/// BUG: Padé [6/6] approximant insufficient — only ~7 significant digits.
#[test]
fn matrix_exp_diagonal() {
    let a = mat_from(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
    let e = matrix_exp(&a);
    // Full double precision would give error < 1e-14.
    assert!((e.get(0, 0) - 1.0_f64.exp()).abs() < 1e-12,
        "BUG: exp(diag)[0,0] should be e^1={}, got {}, error={:.2e} (Padé [6/6] too low order)",
        1.0_f64.exp(), e.get(0, 0), (e.get(0, 0) - 1.0_f64.exp()).abs());
    assert!((e.get(1, 1) - 2.0_f64.exp()).abs() < 1e-12,
        "BUG: exp(diag)[1,1] should be e^2={}, got {}", 2.0_f64.exp(), e.get(1, 1));
    assert!(e.get(0, 1).abs() < 1e-10, "exp(diag) off-diag should be 0");
    assert!(e.get(1, 0).abs() < 1e-10, "exp(diag) off-diag should be 0");
}

/// det(exp(A)) = exp(trace(A)) — Jacobi's formula.
#[test]
fn matrix_exp_det_equals_exp_trace() {
    // 2x2: det = ad - bc. For exp(A): det(exp(A)) = exp(tr(A)).
    let a = mat_from(2, 2, vec![1.0, 0.5, -0.5, 2.0]);
    let e = matrix_exp(&a);
    let det_e = e.get(0, 0) * e.get(1, 1) - e.get(0, 1) * e.get(1, 0);
    let trace_a: f64 = 1.0 + 2.0; // tr(A) = 3
    let expected_det = trace_a.exp();
    assert!((det_e - expected_det).abs() < 1e-6,
        "det(exp(A)) should be exp(tr(A))={}, got {}", expected_det, det_e);
}

/// exp(A) for 1x1 matrix: scalar exp.
/// BUG: Padé [6/6] gives ~1.9e-6 error for input 2.5.
/// The 1x1 case should reduce to scalar exp() — no Padé needed.
#[test]
fn matrix_exp_1x1() {
    let a = mat_from(1, 1, vec![2.5]);
    let e = matrix_exp(&a);
    let expected = 2.5_f64.exp();
    // BUG: even 1x1 matrix goes through Padé approximation instead of scalar exp()
    // Error is ~1.9e-6 — should be <1e-14 (direct scalar exp).
    assert!((e.get(0, 0) - expected).abs() < 1e-12,
        "BUG: matrix_exp([[2.5]]) should be e^2.5={}, got {}, error={:.2e} (1x1 should use scalar exp)",
        expected, e.get(0, 0), (e.get(0, 0) - expected).abs());
}

/// BUG: exp(NaN matrix) — norm1 = NaN, log2(NaN) = NaN, ceil(NaN) as u32 can panic or overflow.
/// Testing that it doesn't panic and returns NaN.
#[test]
fn matrix_exp_nan_no_panic() {
    let a = mat_from(2, 2, vec![f64::NAN, 0.0, 0.0, 1.0]);
    let result = std::panic::catch_unwind(|| matrix_exp(&a));
    assert!(result.is_ok(), "matrix_exp with NaN should not panic");
    if let Ok(e) = result {
        // All entries should be NaN (NaN propagates through scaling and Padé)
        let any_nan = (0..2).flat_map(|i| (0..2).map(move |j| (i, j)))
            .any(|(i, j)| e.get(i, j).is_nan());
        assert!(any_nan || e.get(0, 0).is_finite(),
            "matrix_exp with NaN input: result should contain NaN or be handled gracefully");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// conjugate_gradient — correctness and edge cases
// ═══════════════════════════════════════════════════════════════════════════

/// CG on 2x2 SPD: exact solution in ≤ 2 iterations.
#[test]
fn cg_spd_exact() {
    // A = [[4, 1], [1, 3]], b = [1, 2]. Solution: [1/11, 7/11].
    let a = mat_from(2, 2, vec![4.0, 1.0, 1.0, 3.0]);
    let b = vec![1.0, 2.0];
    let result = conjugate_gradient(&a, &b, None, Some(1e-10), Some(100));
    assert!(result.converged, "CG should converge on 2x2 SPD system");
    // Solution: A⁻¹ b = [1/11, 7/11] ≈ [0.0909, 0.6364]
    assert!((result.x[0] - 1.0 / 11.0).abs() < 1e-8,
        "CG x[0] should be 1/11, got {}", result.x[0]);
    assert!((result.x[1] - 7.0 / 11.0).abs() < 1e-8,
        "CG x[1] should be 7/11, got {}", result.x[1]);
}

/// CG on non-SPD matrix — may diverge.
/// The test documents that CG on non-symmetric matrix doesn't guarantee convergence.
#[test]
fn cg_non_symmetric_residual_check() {
    // Non-symmetric A: CG is not guaranteed to converge.
    let a = mat_from(2, 2, vec![4.0, 3.0, 1.0, 2.0]); // not symmetric
    let b = vec![1.0, 1.0];
    let result = conjugate_gradient(&a, &b, None, Some(1e-10), Some(50));
    // We only check that it doesn't panic and returns a finite result
    assert!(result.residual_norm.is_finite(),
        "CG on non-symmetric A should return finite residual, got {}", result.residual_norm);
    assert!(result.x.iter().all(|v| v.is_finite()),
        "CG on non-symmetric A should return finite x, got {:?}", result.x);
}

/// CG with identity matrix A=I: solution is b in 1 step.
#[test]
fn cg_identity_matrix() {
    let a = Mat::eye(3);
    let b = vec![2.0, 3.0, 4.0];
    let result = conjugate_gradient(&a, &b, None, Some(1e-10), Some(10));
    assert!(result.converged, "CG on identity should converge instantly");
    for i in 0..3 {
        assert!((result.x[i] - b[i]).abs() < 1e-10,
            "CG on I: x[{}] should be {}, got {}", i, b[i], result.x[i]);
    }
}

/// BUG: CG with zero RHS b=[0,...]: solution x=[0]. p_ap check fires if A has 0 columns.
/// Actually: b=[0,0,0] → r = b - A*x0 = 0 - 0 = 0. r_dot = 0. alpha = 0/p_ap.
/// If p_ap != 0 (it won't be — p=r=[0,0,0], ap = A*[0,0,0] = [0,0,0], p_ap = 0).
/// p_ap < 1e-300 → break immediately. residual_norm = 0/b_norm.
/// b_norm = 0, so b_norm.max(1e-300) = 1e-300. r_dot_new.sqrt()/1e-300 ≈ 0.
/// Should converge with x=[0,0,0] and residual=0.
#[test]
fn cg_zero_rhs() {
    let a = mat_from(2, 2, vec![4.0, 1.0, 1.0, 3.0]);
    let b = vec![0.0, 0.0];
    let result = conjugate_gradient(&a, &b, None, Some(1e-10), Some(10));
    assert!(result.x.iter().all(|v| v.abs() < 1e-10),
        "CG with b=0 should give x=0, got {:?}", result.x);
}

/// CG with NaN in A: should not panic, returns NaN x.
#[test]
fn cg_nan_matrix_no_panic() {
    let a = mat_from(2, 2, vec![f64::NAN, 0.0, 0.0, 3.0]);
    let b = vec![1.0, 2.0];
    let result = std::panic::catch_unwind(|| conjugate_gradient(&a, &b, None, None, Some(5)));
    assert!(result.is_ok(), "CG with NaN in A should not panic");
}

// ═══════════════════════════════════════════════════════════════════════════
// gmres — correctness
// ═══════════════════════════════════════════════════════════════════════════

/// GMRES on a simple non-symmetric 2x2 system.
#[test]
fn gmres_non_symmetric_system() {
    // A = [[2, 1], [0, 3]], b = [3, 6].
    // Row 2: 3*x1 = 6 → x1 = 2. Row 1: 2*x0 + x1 = 3 → x0 = 0.5.
    let a = mat_from(2, 2, vec![2.0, 1.0, 0.0, 3.0]);
    let b = vec![3.0, 6.0];
    let result = gmres(&a, &b, None, Some(1e-10), Some(100), Some(10));
    assert!(result.converged || result.residual_norm < 1e-8,
        "GMRES should solve simple triangular system, residual={}", result.residual_norm);
    assert!((result.x[0] - 0.5).abs() < 1e-6,
        "GMRES x[0] should be 0.5, got {}", result.x[0]);
    assert!((result.x[1] - 2.0).abs() < 1e-6,
        "GMRES x[1] should be 2, got {}", result.x[1]);
}

/// GMRES on identity: instant convergence.
#[test]
fn gmres_identity() {
    let a = Mat::eye(3);
    let b = vec![1.0, 2.0, 3.0];
    let result = gmres(&a, &b, None, Some(1e-10), Some(20), Some(10));
    for i in 0..3 {
        assert!((result.x[i] - b[i]).abs() < 1e-8,
            "GMRES on I: x[{}] should be {}, got {}", i, b[i], result.x[i]);
    }
}

/// GMRES: n=1 (trivial case A=[a], b=[b], x=b/a).
#[test]
fn gmres_1x1() {
    let a = mat_from(1, 1, vec![3.0]);
    let b = vec![6.0];
    let result = gmres(&a, &b, None, Some(1e-10), Some(10), Some(5));
    assert!((result.x[0] - 2.0).abs() < 1e-8,
        "GMRES 1x1: x should be 2, got {}", result.x[0]);
}
