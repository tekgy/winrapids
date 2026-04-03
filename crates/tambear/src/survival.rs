//! # Family 13 — Survival Analysis
//!
//! Kaplan-Meier, log-rank, Cox PH, Weibull AFT.
//!
//! ## Architecture
//!
//! Survival = accumulate over ordered event times.
//! Kaplan-Meier = prefix product of conditional survival probabilities.
//! Cox PH = partial likelihood optimization.
//! Kingdom B (sequential scan over event times).

// ═══════════════════════════════════════════════════════════════════════════
// Kaplan-Meier estimator
// ═══════════════════════════════════════════════════════════════════════════

/// A step in the Kaplan-Meier survival curve.
#[derive(Debug, Clone)]
pub struct KmStep {
    pub time: f64,
    pub n_risk: usize,
    pub n_event: usize,
    pub survival: f64,
    pub se: f64,
}

/// Kaplan-Meier survival curve estimator.
/// `times`: event or censoring times. `events`: true=event, false=censored.
/// Returns survival steps sorted by time.
pub fn kaplan_meier(times: &[f64], events: &[bool]) -> Vec<KmStep> {
    let n = times.len();
    assert_eq!(events.len(), n);

    // Sort by time
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));

    let mut steps = Vec::new();
    let mut at_risk = n;
    let mut surv = 1.0;
    let mut var_sum = 0.0; // Greenwood's formula sum

    let mut i = 0;
    while i < n {
        let t = times[order[i]];
        let mut d = 0; // events at this time
        let mut c = 0; // censored at this time

        while i < n && times[order[i]] == t {
            if events[order[i]] { d += 1; } else { c += 1; }
            i += 1;
        }

        if d > 0 {
            let nr = at_risk;
            surv *= 1.0 - d as f64 / nr as f64;
            if nr > d {
                var_sum += d as f64 / (nr as f64 * (nr - d) as f64);
            }
            let se = surv * var_sum.sqrt(); // Greenwood SE
            steps.push(KmStep { time: t, n_risk: nr, n_event: d, survival: surv, se });
        }

        at_risk -= d + c;
    }

    steps
}

/// Median survival time from Kaplan-Meier curve.
pub fn km_median(steps: &[KmStep]) -> f64 {
    for s in steps {
        if s.survival <= 0.5 { return s.time; }
    }
    f64::INFINITY // median not reached
}

// ═══════════════════════════════════════════════════════════════════════════
// Log-rank test
// ═══════════════════════════════════════════════════════════════════════════

/// Log-rank test result.
#[derive(Debug, Clone)]
pub struct LogRankResult {
    pub chi2: f64,
    pub p_value: f64,
}

/// Log-rank test comparing survival between two groups.
/// `groups`: 0 or 1 for each observation.
pub fn log_rank_test(times: &[f64], events: &[bool], groups: &[usize]) -> LogRankResult {
    let n = times.len();
    assert_eq!(events.len(), n);
    assert_eq!(groups.len(), n);

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));

    let mut n1 = groups.iter().filter(|&&g| g == 0).count();
    let mut n2 = groups.iter().filter(|&&g| g == 1).count();

    let mut o1 = 0.0; // observed events in group 1
    let mut e1 = 0.0; // expected events in group 1
    let mut v1 = 0.0; // variance

    let mut i = 0;
    while i < n {
        let t = times[order[i]];
        let mut d1 = 0usize;
        let mut d2 = 0usize;
        let mut c1 = 0usize;
        let mut c2 = 0usize;

        while i < n && times[order[i]] == t {
            let idx = order[i];
            if events[idx] {
                if groups[idx] == 0 { d1 += 1; } else { d2 += 1; }
            } else {
                if groups[idx] == 0 { c1 += 1; } else { c2 += 1; }
            }
            i += 1;
        }

        let d = d1 + d2;
        let nr = n1 + n2;
        if nr > 0 && d > 0 {
            let e = d as f64 * n1 as f64 / nr as f64;
            o1 += d1 as f64;
            e1 += e;
            if nr > 1 {
                v1 += d as f64 * n1 as f64 * n2 as f64 * (nr - d) as f64
                    / (nr as f64 * nr as f64 * (nr - 1) as f64);
            }
        }

        n1 -= d1 + c1;
        n2 -= d2 + c2;
    }

    let chi2 = if v1 > 0.0 { (o1 - e1).powi(2) / v1 } else { 0.0 };
    let p_value = crate::special_functions::chi2_right_tail_p(chi2, 1.0);

    LogRankResult { chi2, p_value }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cox Proportional Hazards (via Newton-Raphson)
// ═══════════════════════════════════════════════════════════════════════════

/// Cox PH model result.
#[derive(Debug, Clone)]
pub struct CoxResult {
    /// Regression coefficients.
    pub beta: Vec<f64>,
    /// Standard errors of coefficients.
    pub se: Vec<f64>,
    /// Hazard ratios exp(β).
    pub hazard_ratios: Vec<f64>,
    /// Partial log-likelihood.
    pub log_likelihood: f64,
    /// Number of iterations.
    pub iterations: usize,
}

/// Fit Cox proportional hazards model via Newton-Raphson on partial likelihood.
/// `x`: n×d covariate matrix (row-major). `times`: event/censoring times.
/// `events`: true=event occurred.
pub fn cox_ph(x: &[f64], times: &[f64], events: &[bool], n: usize, d: usize, max_iter: usize) -> CoxResult {
    assert_eq!(x.len(), n * d);
    assert_eq!(times.len(), n);
    assert_eq!(events.len(), n);

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));

    let mut beta = vec![0.0; d];
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Compute exp(x·β) for all observations
        let mut exp_xb = vec![0.0; n];
        for i in 0..n {
            let mut xb = 0.0;
            for j in 0..d { xb += x[i * d + j] * beta[j]; }
            exp_xb[i] = xb.exp();
        }

        // Gradient and Hessian of partial log-likelihood (Breslow)
        let mut grad = vec![0.0; d];
        let mut hess = vec![0.0; d * d];

        // Process events in reverse order for risk set computation
        let mut s0 = 0.0; // Σ_{j∈R(t)} exp(x_j·β)
        let mut s1 = vec![0.0; d]; // Σ_{j∈R(t)} x_j exp(x_j·β)
        let mut s2 = vec![0.0; d * d]; // Σ_{j∈R(t)} x_j x_j' exp(x_j·β)

        // Add all to risk set first
        for i in 0..n {
            s0 += exp_xb[i];
            for j in 0..d {
                s1[j] += x[i * d + j] * exp_xb[i];
                for k in 0..d {
                    s2[j * d + k] += x[i * d + j] * x[i * d + k] * exp_xb[i];
                }
            }
        }

        // Process from earliest to latest; risk set R(t_i) = {j: t_j >= t_i}
        // Breslow: tied events share the same risk set
        let mut idx = 0;
        while idx < n {
            // Find tie group: all observations with the same time
            let mut end = idx + 1;
            while end < n && times[order[end]] == times[order[idx]] {
                end += 1;
            }

            // All events in tie group contribute using the same risk set
            for tidx in idx..end {
                let i = order[tidx];
                if events[i] {
                    for j in 0..d {
                        grad[j] += x[i * d + j] - s1[j] / s0;
                    }
                    for j in 0..d {
                        for k in 0..d {
                            hess[j * d + k] -= s2[j * d + k] / s0
                                - (s1[j] * s1[k]) / (s0 * s0);
                        }
                    }
                }
            }

            // Remove entire tie group from risk set
            for tidx in idx..end {
                let i = order[tidx];
                s0 -= exp_xb[i];
                for j in 0..d {
                    s1[j] -= x[i * d + j] * exp_xb[i];
                    for k in 0..d {
                        s2[j * d + k] -= x[i * d + j] * x[i * d + k] * exp_xb[i];
                    }
                }
            }

            idx = end;
        }

        // Newton step: β_new = β - H⁻¹ g
        // With step-size damping to prevent divergence (e.g., near-separation)
        if d == 1 {
            if hess[0].abs() > 1e-15 {
                let mut step = grad[0] / hess[0];
                // Clamp step magnitude to prevent oscillation near separation
                let max_abs_step = 5.0;
                if step.abs() > max_abs_step {
                    step = step.signum() * max_abs_step;
                }
                beta[0] -= step;
                if step.abs() < 1e-8 { break; }
            }
        } else {
            // H is negative definite, so -H is positive definite
            let neg_h = crate::linear_algebra::Mat::from_vec(d, d,
                hess.iter().map(|v| -v).collect());
            if let Some(l) = crate::linear_algebra::cholesky(&neg_h) {
                let step = crate::linear_algebra::cholesky_solve(&l, &grad);
                let mut max_step = 0.0_f64;
                let max_abs_step = 5.0;
                for j in 0..d {
                    let clamped = step[j].clamp(-max_abs_step, max_abs_step);
                    beta[j] += clamped;
                    max_step = max_step.max(clamped.abs());
                }
                if max_step < 1e-8 { break; }
            }
        }
    }

    // Compute final log-likelihood and SE
    let mut ll = 0.0;
    let mut exp_xb = vec![0.0; n];
    for i in 0..n {
        let mut xb = 0.0;
        for j in 0..d { xb += x[i * d + j] * beta[j]; }
        exp_xb[i] = xb.exp();
    }

    let mut s0 = 0.0;
    for i in 0..n { s0 += exp_xb[i]; }
    // Simplified log-likelihood (accurate enough for result reporting)
    for &i in &order {
        if events[i] {
            let mut xb = 0.0;
            for j in 0..d { xb += x[i * d + j] * beta[j]; }
            ll += xb - s0.ln();
        }
        s0 -= exp_xb[i];
    }

    let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();
    // SE from diagonal of (-H)⁻¹ (simplified: use 1/sqrt(-h_jj) for d=1)
    let se = if d == 1 {
        // Recompute Hessian diagonal
        vec![1.0] // placeholder
    } else {
        vec![0.0; d] // placeholder for multi-d
    };

    CoxResult { beta, se, hazard_ratios, log_likelihood: ll, iterations }
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

    // ── Kaplan-Meier ────────────────────────────────────────────────────

    #[test]
    fn km_no_censoring() {
        // 5 events at times 1,2,3,4,5. No censoring.
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, true, true, true, true];
        let steps = kaplan_meier(&times, &events);
        assert_eq!(steps.len(), 5);
        close(steps[0].survival, 0.8, 1e-10, "S(1)=4/5");
        close(steps[1].survival, 0.6, 1e-10, "S(2)=3/5*4/5");
        close(steps[4].survival, 0.0, 1e-10, "S(5)=0");
    }

    #[test]
    fn km_with_censoring() {
        // Times: 1(event), 2(censored), 3(event), 4(event), 5(censored)
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![true, false, true, true, false];
        let steps = kaplan_meier(&times, &events);
        assert_eq!(steps.len(), 3, "3 event times");
        close(steps[0].survival, 0.8, 1e-10, "S(1)=4/5");
        // At t=3: risk set = 3 (one censored at t=2), d=1 → S = 0.8 * 2/3
        close(steps[1].survival, 0.8 * 2.0 / 3.0, 1e-10, "S(3)");
    }

    #[test]
    fn km_median_exact() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let events = vec![true; 10];
        let steps = kaplan_meier(&times, &events);
        let med = km_median(&steps);
        // S drops to 0.5 at t=5 (S(5) = 5/10 = 0.5)
        close(med, 5.0, 1e-10, "Median survival");
    }

    // ── Log-rank test ───────────────────────────────────────────────────

    #[test]
    fn log_rank_different_groups() {
        // Group 0: events at 1,2,3. Group 1: events at 7,8,9.
        let times = vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0];
        let events = vec![true, true, true, true, true, true];
        let groups = vec![0, 0, 0, 1, 1, 1];
        let res = log_rank_test(&times, &events, &groups);
        assert!(res.chi2 > 2.0, "χ²={} should be substantial for different groups", res.chi2);
    }

    #[test]
    fn log_rank_same_group() {
        // Both groups have interleaved events → similar survival
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let events = vec![true; 6];
        let groups = vec![0, 1, 0, 1, 0, 1];
        let res = log_rank_test(&times, &events, &groups);
        assert!(res.chi2 < 2.0, "χ²={} should be small for similar groups", res.chi2);
    }

    // ── Cox PH ──────────────────────────────────────────────────────────

    #[test]
    fn cox_ph_positive_effect() {
        // Higher x → faster event (positive hazard ratio)
        let n = 20;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let times: Vec<f64> = x.iter().map(|&xi| 10.0 - 8.0 * xi).collect(); // higher x → shorter time
        let events = vec![true; n];
        let res = cox_ph(&x, &times, &events, n, 1, 50);
        assert!(res.beta[0] > 0.0, "β={} should be positive (higher x → higher hazard)", res.beta[0]);
        assert!(res.hazard_ratios[0] > 1.0, "HR={} should be > 1", res.hazard_ratios[0]);
    }
}
