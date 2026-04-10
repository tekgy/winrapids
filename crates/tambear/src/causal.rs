//! # Family 35 — Causal Inference
//!
//! PSM, IPW, DiD, RDD, doubly robust, mediation, sensitivity.
//!
//! ## Architecture
//!
//! Every method composes existing primitives — no new kernels:
//! - Propensity score = logistic regression (F10)
//! - Matching = nearest-neighbor on propensity scores
//! - IPW = weighted mean
//! - DiD = interaction regression
//! - RDD = local polynomial regression (WLS)
//!
//! Kingdom A: all reduce to regression/matching compositions.

use crate::linear_algebra::{Mat, cholesky, cholesky_solve, simple_linear_regression};

// ═══════════════════════════════════════════════════════════════════════════
// Propensity Score
// ═══════════════════════════════════════════════════════════════════════════

/// Estimate propensity scores via logistic regression (IRLS).
/// `x`: n×d covariate matrix, `treatment`: binary (0/1) of length n.
/// Returns P(T=1|X) for each observation.
pub fn propensity_scores(x: &Mat, treatment: &[f64]) -> Vec<f64> {
    let n = x.rows;
    let d = x.cols;
    assert_eq!(treatment.len(), n);

    // IRLS for logistic regression: β ← (X'WX)⁻¹ X'Wz
    let mut beta = vec![0.0; d + 1]; // +1 for intercept
    let max_iter = 25;

    for _ in 0..max_iter {
        let mut mu = vec![0.0; n];
        for i in 0..n {
            let mut z = beta[0]; // intercept
            for j in 0..d { z += beta[j + 1] * x.get(i, j); }
            mu[i] = sigmoid(z);
        }

        // Weight = μ(1-μ), working response z = η + (y-μ)/w
        // Build augmented normal equations: (X̃'WX̃)β = X̃'Wz̃
        let p = d + 1;
        let mut xtwx = vec![0.0; p * p];
        let mut xtwz = vec![0.0; p];

        for i in 0..n {
            let w = (mu[i] * (1.0 - mu[i])).max(1e-12);
            let eta = beta[0] + (1..p).map(|j| beta[j] * x.get(i, j - 1)).sum::<f64>();
            let z_i = eta + (treatment[i] - mu[i]) / w;

            // X̃[i] = [1, x_i1, ..., x_id]
            for j in 0..p {
                let xj = if j == 0 { 1.0 } else { x.get(i, j - 1) };
                xtwz[j] += w * xj * z_i;
                for k in j..p {
                    let xk = if k == 0 { 1.0 } else { x.get(i, k - 1) };
                    let v = w * xj * xk;
                    xtwx[j * p + k] += v;
                    if k != j { xtwx[k * p + j] += v; }
                }
            }
        }

        let a = Mat::from_vec(p, p, xtwx);
        if let Some(l) = cholesky(&a) {
            beta = cholesky_solve(&l, &xtwz);
        } else {
            break; // singular, keep current estimate
        }
    }

    // Compute final probabilities
    (0..n).map(|i| {
        let z = beta[0] + (0..d).map(|j| beta[j + 1] * x.get(i, j)).sum::<f64>();
        sigmoid(z)
    }).collect()
}

// Thin re-export so local callers don't need the full path.
// The canonical implementation lives in neural.rs (special_functions delegates there too).
#[inline]
fn sigmoid(z: f64) -> f64 { crate::neural::sigmoid(z) }

// ═══════════════════════════════════════════════════════════════════════════
// Propensity Score Matching
// ═══════════════════════════════════════════════════════════════════════════

/// Result of propensity score matching.
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Matched pairs: (treated_idx, control_idx).
    pub pairs: Vec<(usize, usize)>,
    /// ATT estimate.
    pub att: f64,
    /// Standard error of ATT (assuming independent matched pairs).
    pub se: f64,
}

/// 1:1 nearest-neighbor propensity score matching with optional caliper.
/// Returns ATT = E[Y(1) - Y(0) | T=1].
pub fn psm_match(
    propensity: &[f64],
    treatment: &[f64],
    outcome: &[f64],
    caliper: Option<f64>,
) -> MatchResult {
    let n = propensity.len();
    assert_eq!(treatment.len(), n);
    assert_eq!(outcome.len(), n);

    let treated: Vec<usize> = (0..n).filter(|&i| treatment[i] > 0.5).collect();
    let control: Vec<usize> = (0..n).filter(|&i| treatment[i] <= 0.5).collect();

    let mut pairs = Vec::new();
    let mut used = vec![false; n];
    let mut diffs = Vec::new();

    for &t in &treated {
        let mut best_c = None;
        let mut best_dist = f64::MAX;
        for &c in &control {
            if used[c] { continue; }
            let dist = (propensity[t] - propensity[c]).abs();
            if let Some(cal) = caliper {
                if dist > cal { continue; }
            }
            if dist < best_dist { best_dist = dist; best_c = Some(c); }
        }
        if let Some(c) = best_c {
            used[c] = true;
            pairs.push((t, c));
            diffs.push(outcome[t] - outcome[c]);
        }
    }

    let m = diffs.len() as f64;
    let att = if m > 0.0 { diffs.iter().sum::<f64>() / m } else { f64::NAN };
    let se = if m > 1.0 {
        let var: f64 = diffs.iter().map(|d| (d - att).powi(2)).sum::<f64>() / (m - 1.0);
        (var / m).sqrt()
    } else { f64::NAN };

    MatchResult { pairs, att, se }
}

// ═══════════════════════════════════════════════════════════════════════════
// Inverse Probability Weighting (IPW)
// ═══════════════════════════════════════════════════════════════════════════

/// IPW treatment effect result.
#[derive(Debug, Clone)]
pub struct IpwResult {
    /// ATE estimate.
    pub ate: f64,
    /// ATT estimate.
    pub att: f64,
}

/// Inverse Probability Weighting with stabilized weights and trimming.
/// Trim observations with propensity < trim_lo or > trim_hi.
pub fn ipw(
    propensity: &[f64],
    treatment: &[f64],
    outcome: &[f64],
    trim_lo: f64,
    trim_hi: f64,
) -> IpwResult {
    let n = propensity.len();
    let p_treat: f64 = treatment.iter().sum::<f64>() / n as f64;

    let mut sum_t = 0.0;
    let mut w_t = 0.0;
    let mut sum_c = 0.0;
    let mut w_c = 0.0;

    for i in 0..n {
        let e = propensity[i].clamp(trim_lo, trim_hi);
        if treatment[i] > 0.5 {
            let w = p_treat / e;
            sum_t += w * outcome[i];
            w_t += w;
        } else {
            let w = (1.0 - p_treat) / (1.0 - e);
            sum_c += w * outcome[i];
            w_c += w;
        }
    }

    let ate = if w_t > 0.0 && w_c > 0.0 { sum_t / w_t - sum_c / w_c } else { f64::NAN };

    // ATT: weight only control group
    let mut att_sum_c = 0.0;
    let mut att_w_c = 0.0;
    let mut att_sum_t = 0.0;
    let mut att_n_t = 0.0;
    for i in 0..n {
        let e = propensity[i].clamp(trim_lo, trim_hi);
        if treatment[i] > 0.5 {
            att_sum_t += outcome[i];
            att_n_t += 1.0;
        } else {
            let w = e / (1.0 - e);
            att_sum_c += w * outcome[i];
            att_w_c += w;
        }
    }
    let att = if att_n_t > 0.0 && att_w_c > 0.0 {
        att_sum_t / att_n_t - att_sum_c / att_w_c
    } else { f64::NAN };

    IpwResult { ate, att }
}

// ═══════════════════════════════════════════════════════════════════════════
// Difference-in-Differences (DiD)
// ═══════════════════════════════════════════════════════════════════════════

/// DiD result.
#[derive(Debug, Clone)]
pub struct DidResult {
    /// DiD treatment effect estimate.
    pub effect: f64,
    /// Standard error.
    pub se: f64,
    /// t-statistic.
    pub t_stat: f64,
    /// p-value.
    pub p_value: f64,
}

/// Simple 2×2 Difference-in-Differences.
/// `y`: outcomes, `treat`: treatment group (0/1), `post`: post-period (0/1).
pub fn did(y: &[f64], treat: &[f64], post: &[f64]) -> DidResult {
    let n = y.len();
    assert_eq!(treat.len(), n);
    assert_eq!(post.len(), n);

    // Four cell means
    let mut sums = [0.0f64; 4]; // [ctrl_pre, ctrl_post, treat_pre, treat_post]
    let mut counts = [0.0f64; 4];
    for i in 0..n {
        let t = if treat[i] > 0.5 { 2 } else { 0 };
        let p = if post[i] > 0.5 { 1 } else { 0 };
        sums[t + p] += y[i];
        counts[t + p] += 1.0;
    }
    let means: Vec<f64> = (0..4).map(|i| {
        if counts[i] > 0.0 { sums[i] / counts[i] } else { f64::NAN }
    }).collect();

    // DiD = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)
    let effect = (means[3] - means[2]) - (means[1] - means[0]);

    // SE via pooled residual variance
    let mut ss_resid = 0.0;
    for i in 0..n {
        let t = if treat[i] > 0.5 { 2 } else { 0 };
        let p = if post[i] > 0.5 { 1 } else { 0 };
        let resid = y[i] - means[t + p];
        ss_resid += resid * resid;
    }
    let df = n as f64 - 4.0;
    let mse = if df > 0.0 { ss_resid / df } else { f64::NAN };
    let se = (mse * (1.0 / counts[0] + 1.0 / counts[1] + 1.0 / counts[2] + 1.0 / counts[3])).sqrt();
    let t_stat = if se > 0.0 { effect / se } else { f64::NAN };
    let p_value = crate::special_functions::t_two_tail_p(t_stat, df);

    DidResult { effect, se, t_stat, p_value }
}

// ═══════════════════════════════════════════════════════════════════════════
// Regression Discontinuity Design (RDD)
// ═══════════════════════════════════════════════════════════════════════════

/// RDD result.
#[derive(Debug, Clone)]
pub struct RddResult {
    /// Estimated discontinuity (treatment effect at cutoff).
    pub effect: f64,
    /// Intercept left of cutoff.
    pub intercept_left: f64,
    /// Intercept right of cutoff.
    pub intercept_right: f64,
    /// Slope left.
    pub slope_left: f64,
    /// Slope right.
    pub slope_right: f64,
}

/// Sharp RDD via local linear regression on each side of cutoff.
/// `running`: running variable, `outcome`: outcome variable, `cutoff`: threshold.
/// `bandwidth`: use only points within `bandwidth` of `cutoff`.
pub fn rdd_sharp(
    running: &[f64],
    outcome: &[f64],
    cutoff: f64,
    bandwidth: f64,
) -> RddResult {
    let n = running.len();
    assert_eq!(outcome.len(), n);

    // Split into left (below cutoff) and right (at/above cutoff) within bandwidth
    let mut left_x = Vec::new();
    let mut left_y = Vec::new();
    let mut right_x = Vec::new();
    let mut right_y = Vec::new();

    for i in 0..n {
        let d = running[i] - cutoff;
        if d.abs() > bandwidth { continue; }
        if running[i] < cutoff {
            left_x.push(d);
            left_y.push(outcome[i]);
        } else {
            right_x.push(d);
            right_y.push(outcome[i]);
        }
    }

    let left_fit = simple_linear_regression(&left_x, &left_y);
    let right_fit = simple_linear_regression(&right_x, &right_y);
    let (a_l, b_l) = (left_fit.intercept, left_fit.slope);
    let (a_r, b_r) = (right_fit.intercept, right_fit.slope);

    RddResult {
        effect: a_r - a_l,
        intercept_left: a_l,
        intercept_right: a_r,
        slope_left: b_l,
        slope_right: b_r,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sensitivity Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// E-value: minimum confounding strength to explain away the effect.
/// `rr` = observed risk ratio (point estimate).
pub fn e_value(rr: f64) -> f64 {
    let rr = rr.abs().max(1.0); // ensure ≥ 1
    rr + (rr * (rr - 1.0)).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Doubly Robust (AIPW)
// ═══════════════════════════════════════════════════════════════════════════

/// Augmented IPW (doubly robust) ATE estimate.
/// Consistent if EITHER propensity OR outcome model is correct.
/// `mu1`, `mu0`: predicted outcomes under treatment/control (from outcome model).
pub fn doubly_robust_ate(
    propensity: &[f64],
    treatment: &[f64],
    outcome: &[f64],
    mu1: &[f64],
    mu0: &[f64],
) -> f64 {
    let n = propensity.len();
    let mut sum = 0.0;
    for i in 0..n {
        let e = propensity[i].clamp(0.01, 0.99);
        let t = treatment[i];
        let dr = (mu1[i] - mu0[i])
            + t * (outcome[i] - mu1[i]) / e
            - (1.0 - t) * (outcome[i] - mu0[i]) / (1.0 - e);
        sum += dr;
    }
    sum / n as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: {a} vs {b} (diff={})", (a - b).abs());
    }

    // ── Propensity Score ────────────────────────────────────────────────

    #[test]
    fn propensity_scores_separable() {
        // x > 0 → treated with high probability
        let x = Mat::from_rows(&[
            &[-2.0], &[-1.5], &[-1.0], &[-0.5], &[0.0],
            &[0.5], &[1.0], &[1.5], &[2.0], &[2.5],
        ]);
        let treatment = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ps = propensity_scores(&x, &treatment);
        // Treated units should have higher propensity
        let mean_treated: f64 = (5..10).map(|i| ps[i]).sum::<f64>() / 5.0;
        let mean_control: f64 = (0..5).map(|i| ps[i]).sum::<f64>() / 5.0;
        assert!(mean_treated > mean_control, "Treated should have higher propensity");
    }

    // ── PSM ─────────────────────────────────────────────────────────────

    #[test]
    fn psm_known_effect() {
        // Treatment effect = 5.0 exactly
        let propensity = vec![0.3, 0.3, 0.4, 0.4, 0.7, 0.7, 0.6, 0.6];
        let treatment = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let outcome = vec![10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 18.0];
        let res = psm_match(&propensity, &treatment, &outcome, None);
        assert_eq!(res.pairs.len(), 4, "Should match all treated");
        // ATT should be ~5.0
        assert!((res.att - 5.0).abs() < 2.0, "ATT={} should be ~5.0", res.att);
    }

    #[test]
    fn psm_caliper_excludes() {
        // Caliper so tight no matches possible
        let propensity = vec![0.1, 0.9, 0.15, 0.85];
        let treatment = vec![0.0, 1.0, 0.0, 1.0];
        let outcome = vec![1.0, 2.0, 3.0, 4.0];
        let res = psm_match(&propensity, &treatment, &outcome, Some(0.01));
        assert_eq!(res.pairs.len(), 0, "Tight caliper should exclude all");
    }

    // ── IPW ─────────────────────────────────────────────────────────────

    #[test]
    fn ipw_constant_propensity() {
        // If e(x) = 0.5 for all, IPW = simple difference in means
        let propensity = vec![0.5; 8];
        let treatment = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let outcome = vec![10.0, 11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 18.0];
        let res = ipw(&propensity, &treatment, &outcome, 0.01, 0.99);
        let diff_means = 16.5 - 11.5;
        close(res.ate, diff_means, 0.1, "IPW ATE with constant propensity");
    }

    // ── DiD ─────────────────────────────────────────────────────────────

    #[test]
    fn did_known_effect() {
        // Parallel trends + treatment effect of 3.0
        let y = vec![
            10.0, 11.0, // ctrl_pre
            12.0, 13.0, // ctrl_post (trend = +2)
            20.0, 21.0, // treat_pre
            25.0, 26.0, // treat_post (trend = +2 + effect 3.0)
        ];
        let treat = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let post = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let res = did(&y, &treat, &post);
        close(res.effect, 3.0, 1e-10, "DiD effect");
    }

    #[test]
    fn did_no_effect() {
        // Parallel trends, no treatment
        let y = vec![10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0];
        let treat = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let post = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let res = did(&y, &treat, &post);
        close(res.effect, 0.0, 1e-10, "DiD no effect");
        assert!(res.p_value > 0.5, "Should not reject H₀");
    }

    // ── RDD ─────────────────────────────────────────────────────────────

    #[test]
    fn rdd_sharp_known_jump() {
        // y = x + 3·I(x ≥ 0), so discontinuity = 3.0
        let running: Vec<f64> = (-20..=20).map(|i| i as f64 * 0.5).collect();
        let outcome: Vec<f64> = running.iter().map(|&x| {
            x + if x >= 0.0 { 3.0 } else { 0.0 }
        }).collect();
        let res = rdd_sharp(&running, &outcome, 0.0, 5.0);
        close(res.effect, 3.0, 0.5, "RDD effect");
    }

    // ── E-value ─────────────────────────────────────────────────────────

    #[test]
    fn e_value_known() {
        // RR = 2.0: E-value = 2 + √(2·1) = 2 + √2 ≈ 3.414
        let ev = e_value(2.0);
        close(ev, 2.0 + 2.0_f64.sqrt(), 1e-10, "E-value for RR=2");
    }

    #[test]
    fn e_value_no_effect() {
        // RR = 1.0: E-value = 1 + √0 = 1
        close(e_value(1.0), 1.0, 1e-10, "E-value for null");
    }

    // ── Doubly Robust ───────────────────────────────────────────────────

    #[test]
    fn doubly_robust_known_effect() {
        // Perfect propensity model + true effect = 5
        let propensity = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let treatment = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let outcome = vec![10.0, 11.0, 12.0, 15.0, 16.0, 17.0];
        let mu1 = vec![15.0, 16.0, 17.0, 15.0, 16.0, 17.0]; // E[Y(1)|X]
        let mu0 = vec![10.0, 11.0, 12.0, 10.0, 11.0, 12.0]; // E[Y(0)|X]
        let ate = doubly_robust_ate(&propensity, &treatment, &outcome, &mu1, &mu0);
        close(ate, 5.0, 0.1, "DR ATE");
    }
}
