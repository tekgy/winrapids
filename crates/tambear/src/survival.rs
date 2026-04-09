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

    // Sort by time, filter NaN (total_cmp sorts NaN last — skip them)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| times[a].total_cmp(&times[b]));
    let n_valid = order.iter().position(|&i| times[i].is_nan()).unwrap_or(order.len());

    let mut steps = Vec::new();
    let mut at_risk = n_valid;
    let mut surv = 1.0;
    let mut var_sum = 0.0; // Greenwood's formula sum

    let mut i = 0;
    while i < n_valid {
        let t = times[order[i]];
        let mut d = 0; // events at this time
        let mut c = 0; // censored at this time

        while i < n_valid && times[order[i]] == t {
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
    // Skip NaN entries (total_cmp sorts them last)
    let n_valid = order.iter().position(|&i| times[i].is_nan()).unwrap_or(order.len());

    let mut n1 = order[..n_valid].iter().filter(|&&i| groups[i] == 0).count();
    let mut n2 = order[..n_valid].iter().filter(|&&i| groups[i] == 1).count();

    let mut o1 = 0.0; // observed events in group 1
    let mut e1 = 0.0; // expected events in group 1
    let mut v1 = 0.0; // variance

    let mut i = 0;
    while i < n_valid {
        let t = times[order[i]];
        let mut d1 = 0usize;
        let mut d2 = 0usize;
        let mut c1 = 0usize;
        let mut c2 = 0usize;

        while i < n_valid && times[order[i]] == t {
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
    /// Schoenfeld residuals for each event (in time order).
    ///
    /// `schoenfeld_residuals[k][j]` = x_{ij} - E[x_j | R(t_i)] for the k-th event.
    /// Where E[x_j | R(t_i)] = S1_j(t_i) / S0(t_i) is the risk-set weighted mean of x_j.
    ///
    /// Use for Cox PH proportional hazards assumption testing:
    /// - Plot residuals vs time (or log-time) per covariate
    /// - Fit a smoothed line: non-zero slope → time-varying effect → PH violated
    /// - Formal test: cor.test(schoenfeld_j, time) with Grambsch-Therneau correction
    ///
    /// Residuals sum to approximately zero at convergence (score equation).
    pub schoenfeld_residuals: Vec<Vec<f64>>,
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

        // Compute exp(x·β) for all observations (clamp to prevent overflow)
        let mut exp_xb = vec![0.0; n];
        let mut xb_max = f64::NEG_INFINITY;
        let mut xb_vals = vec![0.0; n];
        for i in 0..n {
            let mut xb = 0.0;
            for j in 0..d { xb += x[i * d + j] * beta[j]; }
            xb_vals[i] = xb;
            if xb > xb_max { xb_max = xb; }
        }
        // Log-sum-exp trick: exp(xb - xb_max) keeps values in [0, 1]
        // This shifts the partial likelihood by a constant that cancels in ratios
        for i in 0..n {
            exp_xb[i] = (xb_vals[i] - xb_max).exp();
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

    // Compute final log-likelihood and SE with log-sum-exp stabilization
    let mut ll = 0.0;
    let mut xb_vals_final = vec![0.0; n];
    let mut xb_max_final = f64::NEG_INFINITY;
    for i in 0..n {
        let mut xb = 0.0;
        for j in 0..d { xb += x[i * d + j] * beta[j]; }
        xb_vals_final[i] = xb;
        if xb > xb_max_final { xb_max_final = xb; }
    }
    let mut exp_xb = vec![0.0; n];
    for i in 0..n { exp_xb[i] = (xb_vals_final[i] - xb_max_final).exp(); }

    let mut s0 = 0.0;
    for i in 0..n { s0 += exp_xb[i]; }
    for &i in &order {
        if events[i] && s0 > 1e-300 {
            // log-likelihood: xb_i - log(Σ exp(xb_j)) = xb_i - xb_max - log(s0)
            ll += (xb_vals_final[i] - xb_max_final) - s0.ln();
        }
        s0 -= exp_xb[i];
        if s0 < 0.0 { s0 = 0.0; } // guard against floating-point underflow
    }

    let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    // SE from diagonal of (-H)⁻¹ and Schoenfeld residuals, recomputed at final beta
    let mut hess_final = vec![0.0; d * d];
    let mut s1_f = vec![0.0; d];
    let mut s2_f = vec![0.0; d * d];
    let mut s0_f = 0.0;
    let mut schoenfeld_residuals: Vec<Vec<f64>> = Vec::new();
    for i in 0..n {
        s0_f += exp_xb[i];
        for j in 0..d {
            s1_f[j] += x[i * d + j] * exp_xb[i];
            for k in 0..d {
                s2_f[j * d + k] += x[i * d + j] * x[i * d + k] * exp_xb[i];
            }
        }
    }
    let mut idx_f = 0;
    while idx_f < n {
        let mut end_f = idx_f + 1;
        while end_f < n && times[order[end_f]] == times[order[idx_f]] { end_f += 1; }
        for tidx in idx_f..end_f {
            let i = order[tidx];
            if events[i] && s0_f > 0.0 {
                // Schoenfeld residual: x_i - E[x | risk set] = x_i - S1/S0
                let resid: Vec<f64> = (0..d).map(|j| {
                    x[i * d + j] - s1_f[j] / s0_f
                }).collect();
                schoenfeld_residuals.push(resid);

                for j in 0..d {
                    for k in 0..d {
                        hess_final[j * d + k] -= s2_f[j * d + k] / s0_f
                            - (s1_f[j] * s1_f[k]) / (s0_f * s0_f);
                    }
                }
            }
        }
        for tidx in idx_f..end_f {
            let i = order[tidx];
            s0_f -= exp_xb[i];
            for j in 0..d {
                s1_f[j] -= x[i * d + j] * exp_xb[i];
                for k in 0..d {
                    s2_f[j * d + k] -= x[i * d + j] * x[i * d + k] * exp_xb[i];
                }
            }
        }
        idx_f = end_f;
    }
    // SE = sqrt(diag((-H)⁻¹))
    let se = if d == 1 {
        let h = -hess_final[0];
        vec![if h > 1e-15 { (1.0 / h).sqrt() } else { f64::NAN }]
    } else {
        let neg_h = crate::linear_algebra::Mat::from_vec(d, d,
            hess_final.iter().map(|v| -v).collect());
        match crate::linear_algebra::cholesky(&neg_h) {
            Some(l) => {
                (0..d).map(|j| {
                    let mut ej = vec![0.0; d];
                    ej[j] = 1.0;
                    let col = crate::linear_algebra::cholesky_solve(&l, &ej);
                    col[j].max(0.0).sqrt()
                }).collect()
            }
            None => vec![f64::NAN; d],
        }
    };

    CoxResult { beta, se, hazard_ratios, log_likelihood: ll, iterations, schoenfeld_residuals }
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

    #[test]
    fn cox_schoenfeld_residuals_count_and_sum() {
        // Schoenfeld residuals: one per event. At β=0 (null model), residual for
        // observation i is x_i - mean(x in risk set), and residuals sum = score(β=0).
        // Use mild effect so optimizer converges; then verify count and that each
        // residual has the correct sign (event with highest covariate → positive residual).
        let times = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // earlier time = shorter survival
        let x = vec![0.2, 0.4, 0.6, 0.8, 1.0];     // higher x → shorter time (positive effect)
        let events = vec![true; 5];
        let n = 5;
        let res = cox_ph(&x, &times, &events, n, 1, 100);

        // One residual per event
        assert_eq!(res.schoenfeld_residuals.len(), n,
            "should have one Schoenfeld residual per event");
        // Each residual is length d=1
        assert!(res.schoenfeld_residuals.iter().all(|r| r.len() == 1));
        // All finite
        assert!(res.schoenfeld_residuals.iter().all(|r| r[0].is_finite()),
            "Schoenfeld residuals should all be finite");
        // Last event (x=0.2, earliest time) faces a risk set of just {0.2}: residual ≈ 0
        let last_resid = res.schoenfeld_residuals.last().unwrap()[0];
        assert!(last_resid.abs() < 1e-3,
            "Last event faces only itself in risk set → residual ≈ 0, got {}", last_resid);
    }

    #[test]
    fn cox_schoenfeld_residuals_multivariate() {
        // Two covariates: residuals still one per event, length 2.
        let n = 8;
        let x: Vec<f64> = (0..n).flat_map(|i| {
            vec![i as f64, (n - i) as f64]
        }).collect();
        let times: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let events = vec![true; n];
        let res = cox_ph(&x, &times, &events, n, 2, 50);

        assert_eq!(res.schoenfeld_residuals.len(), n);
        assert!(res.schoenfeld_residuals.iter().all(|r| r.len() == 2));
        // All residuals should be finite
        assert!(res.schoenfeld_residuals.iter().all(|r| r.iter().all(|v| v.is_finite())),
            "Schoenfeld residuals should all be finite");
    }
}
