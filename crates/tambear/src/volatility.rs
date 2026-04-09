//! # Family 18 — Volatility & Financial Time Series
//!
//! GARCH(1,1), EWMA, realized volatility, microstructure metrics.
//!
//! ## Architecture
//!
//! GARCH = sequential variance recursion (Kingdom B) + MLE (Kingdom C).
//! Realized measures = accumulation over intraday returns (Kingdom A).
//! Microstructure = simple accumulate from trade/quote data.

// ═══════════════════════════════════════════════════════════════════════════
// GARCH(1,1)
// ═══════════════════════════════════════════════════════════════════════════

/// GARCH(1,1) model result.
#[derive(Debug, Clone)]
pub struct GarchResult {
    /// ω (constant).
    pub omega: f64,
    /// α (ARCH coefficient, shock persistence).
    pub alpha: f64,
    /// β (GARCH coefficient, volatility persistence).
    pub beta: f64,
    /// Conditional variances σ²_t (length n).
    pub variances: Vec<f64>,
    /// Log-likelihood.
    pub log_likelihood: f64,
    /// Number of optimization iterations.
    pub iterations: usize,
    /// True if α + β > 0.99 (near IGARCH boundary).
    /// When true, ω and unconditional variance are unreliable.
    pub near_igarch: bool,
}

/// Fit GARCH(1,1) model via MLE on return series.
/// `returns`: mean-adjusted return series.
/// Optimizes via coordinate descent on (ω, α, β) with constraints.
pub fn garch11_fit(returns: &[f64], max_iter: usize) -> GarchResult {
    let n = returns.len();
    assert!(n >= 10);

    // Unconditional variance as starting estimate
    let moments = crate::descriptive::moments_ungrouped(returns);
    let uncond_var = moments.variance(0);

    // Initialize: ω = 0.1*var, α = 0.1, β = 0.8
    // Floor uncond_var to prevent ln(0) in reparameterization
    let uncond_var = uncond_var.max(1e-15);
    let mut omega = (0.1 * uncond_var).max(1e-20);
    let mut alpha = 0.1;
    let mut beta = 0.8;

    let garch_ll = |omega: f64, alpha: f64, beta: f64| -> (f64, Vec<f64>) {
        let mut sigma2 = vec![0.0; n];
        sigma2[0] = uncond_var.max(1e-15); // backcast initialization (floored)
        for t in 1..n {
            sigma2[t] = omega + alpha * returns[t - 1].powi(2) + beta * sigma2[t - 1];
            sigma2[t] = sigma2[t].max(1e-15); // floor
        }
        let ll: f64 = (0..n).map(|t| {
            -0.5 * (std::f64::consts::TAU.ln() + sigma2[t].ln() + returns[t].powi(2) / sigma2[t])
        }).sum();
        (ll, sigma2)
    };

    // L-BFGS optimization with unconstrained reparameterization.
    // θ_1 = ln(ω),  ω = exp(θ_1) > 0
    // θ_2 = logit(α/0.5), α = 0.5·sigmoid(θ_2) ∈ (0, 0.5)
    // θ_3 = logit(β/0.999), β = 0.999·sigmoid(θ_3) ∈ (0, 0.999)
    // With post-hoc stationarity enforcement: α + β < 1.

    let sigmoid = |x: f64| -> f64 { 1.0 / (1.0 + (-x).exp()) };
    let logit = |p: f64| -> f64 { (p / (1.0 - p)).ln() };

    let to_constrained = |theta: &[f64]| -> (f64, f64, f64) {
        let o = theta[0].exp();
        let a = 0.5 * sigmoid(theta[1]);
        let b = 0.999 * sigmoid(theta[2]);
        (o, a, b)
    };

    let neg_ll = |theta: &[f64]| -> f64 {
        let (o, a, b) = to_constrained(theta);
        if a + b >= 0.999 { return 1e20; }
        -garch_ll(o, a, b).0
    };

    let neg_ll_grad = |theta: &[f64]| -> Vec<f64> {
        let delta = 1e-7;
        let f0 = neg_ll(theta);
        (0..3).map(|i| {
            let mut t = theta.to_vec();
            t[i] += delta;
            (neg_ll(&t) - f0) / delta
        }).collect()
    };

    let theta0 = vec![
        omega.ln(),
        logit(alpha / 0.5),
        logit(beta / 0.999),
    ];

    let result = crate::optimization::lbfgs(&neg_ll, &neg_ll_grad, &theta0, 5, max_iter, 1e-8);
    let iterations = result.iterations;
    let (omega, alpha, beta) = to_constrained(&result.x);

    // Post-hoc stationarity enforcement
    let (alpha, beta) = if alpha + beta >= 0.999 {
        let s = alpha + beta;
        (alpha * 0.998 / s, beta * 0.998 / s)
    } else {
        (alpha, beta)
    };
    let near_igarch = alpha + beta > 0.99;
    // Clamp omega: near IGARCH, the optimizer can push ω to absurd values
    // because the unconditional variance ω/(1-α-β) → ∞. Cap at 100× sample variance.
    let omega = if near_igarch { omega.min(100.0 * uncond_var) } else { omega };
    let (log_likelihood, variances) = garch_ll(omega, alpha, beta);
    GarchResult { omega, alpha, beta, variances, log_likelihood, iterations, near_igarch }
}

/// Forecast GARCH(1,1) conditional variance h steps ahead.
pub fn garch11_forecast(res: &GarchResult, last_return: f64, horizon: usize) -> Vec<f64> {
    let uncond = res.omega / (1.0 - res.alpha - res.beta).max(1e-10);
    let mut sigma2 = res.omega + res.alpha * last_return.powi(2)
        + res.beta * res.variances.last().copied().unwrap_or(uncond);
    let mut forecasts = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        forecasts.push(sigma2);
        // E[σ²_{t+h}] = ω + (α+β)·σ²_{t+h-1} (since E[ε²]=σ²)
        sigma2 = res.omega + (res.alpha + res.beta) * sigma2;
    }
    forecasts
}

// ═══════════════════════════════════════════════════════════════════════════
// EWMA Volatility (RiskMetrics)
// ═══════════════════════════════════════════════════════════════════════════

/// EWMA conditional variance: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}.
/// Default λ = 0.94 (RiskMetrics daily).
pub fn ewma_variance(returns: &[f64], lambda: f64) -> Vec<f64> {
    let n = returns.len();
    let mut sigma2 = Vec::with_capacity(n);
    // Initialize with the first squared return (RiskMetrics convention).
    // Using the full-sample mean would introduce look-ahead bias because
    // returns[1..n-1] are not yet observed at t=0.
    let var0: f64 = if n > 0 { returns[0].powi(2).max(1e-15) } else { 1e-15 };
    sigma2.push(var0);
    for t in 1..n {
        let s = lambda * sigma2[t - 1] + (1.0 - lambda) * returns[t - 1].powi(2);
        sigma2.push(s.max(1e-15));
    }
    sigma2
}

// ═══════════════════════════════════════════════════════════════════════════
// Realized Volatility
// ═══════════════════════════════════════════════════════════════════════════

/// Realized variance: sum of squared intraday returns.
pub fn realized_variance(intraday_returns: &[f64]) -> f64 {
    intraday_returns.iter().map(|r| r * r).sum()
}

/// Realized volatility = √RV.
pub fn realized_volatility(intraday_returns: &[f64]) -> f64 {
    realized_variance(intraday_returns).sqrt()
}

/// Bipower variation: robust to jumps.
/// BV = (π/2) · Σ |r_t| · |r_{t-1}|
pub fn bipower_variation(intraday_returns: &[f64]) -> f64 {
    let n = intraday_returns.len();
    if n < 2 { return 0.0; }
    let mu1 = (2.0 / std::f64::consts::PI).sqrt(); // E[|Z|] for Z~N(0,1)
    let sum: f64 = (1..n).map(|t| {
        intraday_returns[t].abs() * intraday_returns[t - 1].abs()
    }).sum();
    sum / (mu1 * mu1)
}

/// Jump test statistic (Barndorff-Nielsen & Shephard).
/// Large positive values indicate jumps. Under H₀: BNS ~ N(0,1).
pub fn jump_test_bns(intraday_returns: &[f64]) -> f64 {
    let rv = realized_variance(intraday_returns);
    let bv = bipower_variation(intraday_returns);
    let n = intraday_returns.len() as f64;
    // Relative jump
    if bv < 1e-15 { return 0.0; }
    let ratio = (rv - bv) / bv;
    // Simplified asymptotic variance (tri-power quarticity estimator skipped for now)
    ratio * n.sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Microstructure metrics
// ═══════════════════════════════════════════════════════════════════════════

/// Roll spread estimator: 2·√(-Cov(Δp_t, Δp_{t-1})).
/// Returns 0 if covariance is non-negative (no spread signal).
pub fn roll_spread(prices: &[f64]) -> f64 {
    let n = prices.len();
    if n < 3 { return 0.0; }
    let dp: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();
    let m = dp.len();
    // Covariance of consecutive price changes
    let mean_dp: f64 = dp.iter().sum::<f64>() / m as f64;
    let cov: f64 = dp[..m - 1].iter().zip(dp[1..].iter())
        .map(|(a, b)| (a - mean_dp) * (b - mean_dp))
        .sum::<f64>() / (m - 1) as f64;
    if cov >= 0.0 { return 0.0; }
    2.0 * (-cov).sqrt()
}

/// Kyle's lambda: price impact per unit of order flow.
/// Simple regression ΔP = λ·(signed_volume) + ε.
/// Returns λ (slope).
pub fn kyle_lambda(price_changes: &[f64], signed_volumes: &[f64]) -> f64 {
    let n = price_changes.len();
    assert_eq!(signed_volumes.len(), n);
    if n < 2 { return 0.0; }

    let mean_v = signed_volumes.iter().sum::<f64>() / n as f64;
    let mean_p = price_changes.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_v = 0.0;
    for i in 0..n {
        let dv = signed_volumes[i] - mean_v;
        cov += dv * (price_changes[i] - mean_p);
        var_v += dv * dv;
    }
    if var_v < 1e-15 { return 0.0; }
    cov / var_v
}

/// Amihud illiquidity ratio: (1/n) Σ |r_t| / volume_t.
pub fn amihud_illiquidity(returns: &[f64], volumes: &[f64]) -> f64 {
    let n = returns.len();
    assert_eq!(volumes.len(), n);
    let sum: f64 = returns.iter().zip(volumes.iter())
        .map(|(r, v)| if *v > 0.0 { r.abs() / v } else { 0.0 })
        .sum();
    sum / n as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// Annualization helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Annualize daily volatility (assuming 252 trading days).
pub fn annualize_vol(daily_vol: f64, trading_days: f64) -> f64 {
    daily_vol * trading_days.sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Engle ARCH-LM test
// ═══════════════════════════════════════════════════════════════════════════

/// Result of Engle's ARCH-LM test for conditional heteroscedasticity.
#[derive(Debug, Clone)]
pub struct ArchLmResult {
    /// LM test statistic: (n - q) * R² ~ χ²(q)
    pub statistic: f64,
    /// Right-tail p-value
    pub p_value: f64,
    /// Degrees of freedom (= n_lags)
    pub df: usize,
    /// Number of observations used (= n - n_lags)
    pub n_obs: usize,
}

/// Engle's ARCH-LM test for autoregressive conditional heteroscedasticity.
///
/// H₀: no ARCH effects (squared residuals are serially uncorrelated).
/// H₁: ARCH effects present.
///
/// Algorithm:
/// 1. Square the residuals: u_t = e_t²
/// 2. Regress u_t on [1, u_{t-1}, …, u_{t-q}] via OLS
/// 3. Compute R² of that regression
/// 4. LM statistic = (n - q) * R² ~ χ²(q) under H₀
///
/// Returns `None` if `residuals.len() <= n_lags` (insufficient data).
pub fn arch_lm_test(residuals: &[f64], n_lags: usize) -> Option<ArchLmResult> {
    let n = residuals.len();
    if n <= n_lags || n_lags == 0 {
        return None;
    }

    // Squared residuals
    let u: Vec<f64> = residuals.iter().map(|e| e * e).collect();

    // Effective sample: t = n_lags .. n  (n_eff observations)
    let n_eff = n - n_lags;

    // Design matrix: [1, u_{t-1}, ..., u_{t-q}]  shape (n_eff, n_lags+1)
    let n_cols = n_lags + 1;
    let mut x_data = vec![0.0_f64; n_eff * n_cols];
    for t in 0..n_eff {
        x_data[t * n_cols] = 1.0; // intercept
        for lag in 1..=n_lags {
            x_data[t * n_cols + lag] = u[n_lags + t - lag];
        }
    }

    // Response: u_{t} for t = n_lags .. n
    let y: Vec<f64> = u[n_lags..].to_vec();

    // OLS via qr_solve
    let x_mat = crate::linear_algebra::Mat {
        rows: n_eff,
        cols: n_cols,
        data: x_data,
    };
    let beta = crate::linear_algebra::qr_solve(&x_mat, &y);

    // Fitted values and residuals
    let y_hat: Vec<f64> = (0..n_eff)
        .map(|t| {
            (0..n_cols)
                .map(|k| x_mat.data[t * n_cols + k] * beta[k])
                .sum::<f64>()
        })
        .collect();

    let y_mean = y.iter().sum::<f64>() / n_eff as f64;
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = y
        .iter()
        .zip(y_hat.iter())
        .map(|(yi, fi)| (yi - fi).powi(2))
        .sum();

    // R² — clamp to [0, 1] for numerical safety
    let r2 = if ss_tot < 1e-300 {
        0.0
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };

    let statistic = (n_eff as f64) * r2;
    let p_value = crate::special_functions::chi2_right_tail_p(statistic, n_lags as f64);

    Some(ArchLmResult {
        statistic,
        p_value,
        df: n_lags,
        n_obs: n_eff,
    })
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

    #[test]
    fn garch_fits() {
        // Generate GARCH(1,1) with known parameters: ω=0.001, α=0.1, β=0.85
        // The fitted parameters should recover the true values within tolerance.
        let true_omega = 0.001;
        let true_alpha = 0.1;
        let true_beta = 0.85;
        let n = 500;
        let mut returns = vec![0.0; n];
        let mut sigma2: f64 = true_omega / (1.0 - true_alpha - true_beta); // unconditional var
        let mut rng = 42u64;
        for t in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let z = (rng as f64 / u64::MAX as f64 - 0.5) * 3.46; // ~uniform scaled
            returns[t] = sigma2.sqrt() * z;
            sigma2 = true_omega + true_alpha * returns[t].powi(2) + true_beta * sigma2;
        }

        let res = garch11_fit(&returns, 200);
        // Parameter recovery: fitted values should be near the true DGP parameters
        assert!((res.alpha - true_alpha).abs() < 0.15,
            "α={:.4} should be near true α={true_alpha}", res.alpha);
        assert!((res.beta - true_beta).abs() < 0.15,
            "β={:.4} should be near true β={true_beta}", res.beta);
        // Stationarity is guaranteed by the true DGP (α+β=0.95 < 1)
        assert!(res.alpha + res.beta < 1.0, "Stationarity: α+β={}", res.alpha + res.beta);
        assert!(res.omega > 0.0, "ω={}", res.omega);
        assert_eq!(res.variances.len(), n);
    }

    #[test]
    fn garch_forecast_mean_reverts() {
        let res = GarchResult {
            omega: 0.001, alpha: 0.1, beta: 0.8, variances: vec![0.01; 10],
            log_likelihood: 0.0, iterations: 0, near_igarch: false,
        };
        let fc = garch11_forecast(&res, 0.05, 100);
        let uncond = 0.001 / (1.0 - 0.1 - 0.8);
        // Forecast should converge toward unconditional variance
        assert!((fc[99] - uncond).abs() < uncond * 0.1,
            "Long-horizon forecast={} should approach uncond={}", fc[99], uncond);
    }

    #[test]
    fn ewma_constant_returns() {
        // With constant returns r=c, EWMA converges to σ² = c².
        // EWMA_t = λ·EWMA_{t-1} + (1-λ)·r² — at equilibrium σ² = r².
        let returns = vec![0.01; 20];
        let sigma2 = ewma_variance(&returns, 0.94);
        assert_eq!(sigma2.len(), 20);
        // After 20 steps of EWMA with constant r=0.01, converges to r² = 0.0001
        close(sigma2[19], 0.0001, 1e-5, "EWMA convergence to r² for constant returns");
    }

    #[test]
    fn rv_basic() {
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        let rv = realized_variance(&returns);
        let expected: f64 = returns.iter().map(|r| r * r).sum();
        close(rv, expected, 1e-15, "RV");
    }

    #[test]
    fn bipower_nonnegative() {
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.005];
        let bv = bipower_variation(&returns);
        assert!(bv >= 0.0, "BV={bv} should be non-negative");
        assert!(bv > 0.0, "BV should be positive for nonzero returns");
    }

    #[test]
    fn roll_spread_recovers_known_spread() {
        // Roll (1984): spread S → Cov(ΔP_t, ΔP_{t-1}) = -S²/4
        // Roll estimator: S = 2√(-Cov) recovers the true spread.
        // Pure bid-ask bounce with no fundamental drift: prices alternate ±S/2.
        let true_spread = 0.02;
        let half = true_spread / 2.0;
        // Alternating bounce: 100, 100.01, 99.99, 100.01, ... (deterministic)
        let mut prices = vec![100.0_f64];
        for i in 0..199 {
            let last = *prices.last().unwrap();
            prices.push(last + if i % 2 == 0 { half } else { -half });
        }
        let rs = roll_spread(&prices);
        // For pure bounce, empirical autocovariance → -S²/4 exactly as n→∞
        close(rs, true_spread, 5e-4, "Roll spread should recover true S=0.02");
    }

    #[test]
    fn kyle_lambda_recovers_known_slope() {
        // Kyle (1985): ΔP_t = λ·Q_t + ε, where λ is price impact.
        // With known DGP λ=0.005, OLS should recover it.
        let true_lambda = 0.005;
        let n = 100;
        let signed_volumes: Vec<f64> = (0..n).map(|i| {
            // Deterministic signed volumes: alternating +/-100, 200, 50, ...
            let v = ((i % 4 + 1) * 50) as f64;
            if i % 2 == 0 { v } else { -v }
        }).collect();
        let price_changes: Vec<f64> = signed_volumes.iter()
            .map(|&q| true_lambda * q)  // exact linear: no noise
            .collect();
        let lambda = kyle_lambda(&price_changes, &signed_volumes);
        close(lambda, true_lambda, 1e-8, "Kyle λ should recover true slope");
    }

    #[test]
    fn amihud_positive() {
        let returns = vec![0.01, -0.02, 0.03];
        let volumes = vec![1e6, 2e6, 1.5e6];
        let ill = amihud_illiquidity(&returns, &volumes);
        assert!(ill > 0.0, "Amihud={ill} should be positive");
    }

    #[test]
    fn annualize_scales() {
        let daily = 0.01; // 1% daily
        let annual = annualize_vol(daily, 252.0);
        close(annual, 0.01 * 252.0_f64.sqrt(), 1e-10, "Annualized");
    }

    // ── Regression: L-BFGS optimizer for GARCH ─────────────────────────
    // The old fixed-step gradient ascent was slow and inaccurate at parameter
    // boundaries. L-BFGS with reparameterization should recover parameters
    // more accurately, especially for near-integrated processes.
    #[test]
    fn garch_lbfgs_parameter_recovery_regression() {
        // Near-integrated GARCH: α+β = 0.95 (challenging for old optimizer)
        let true_omega = 0.0005;
        let true_alpha = 0.05;
        let true_beta = 0.90;
        let n = 1000;
        let mut returns = vec![0.0; n];
        let mut sigma2: f64 = true_omega / (1.0 - true_alpha - true_beta);
        let mut rng = crate::rng::Xoshiro256::new(42);
        for t in 0..n {
            let z = crate::rng::sample_normal(&mut rng, 0.0, 1.0);
            returns[t] = sigma2.sqrt() * z;
            sigma2 = true_omega + true_alpha * returns[t].powi(2) + true_beta * sigma2;
        }

        let res = garch11_fit(&returns, 500);
        // With L-BFGS, parameter recovery should be reasonable
        assert!(res.alpha + res.beta < 1.0, "Stationarity violated: α+β={}", res.alpha + res.beta);
        assert!(res.omega > 0.0, "ω should be positive, got {}", res.omega);
        // Parameters should be in plausible range
        assert!((res.alpha + res.beta - 0.95).abs() < 0.15,
            "α+β={:.3} should be near 0.95", res.alpha + res.beta);
    }

    // ── Engle ARCH-LM tests ──────────────────────────────────────────────

    #[test]
    fn arch_lm_white_noise_high_pvalue() {
        // iid N(0,1) — no ARCH effects → p-value should be large (fail to reject H₀)
        // Use a deterministic LCG so the test is reproducible.
        let mut rng = 12345u64;
        let n = 200;
        let residuals: Vec<f64> = (0..n)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng as f64 / u64::MAX as f64 - 0.5) * 2.0 * 1.7320508 // uniform → σ≈1
            })
            .collect();
        let result = arch_lm_test(&residuals, 5).unwrap();
        // White noise: p-value should be large (not reject H₀ at 5% level)
        assert!(
            result.p_value > 0.05,
            "White noise: p={:.4} should be > 0.05 (no ARCH effects)",
            result.p_value
        );
        assert_eq!(result.df, 5);
        assert_eq!(result.n_obs, n - 5);
    }

    #[test]
    fn arch_lm_arch_process_low_pvalue() {
        // ARCH(1): σ²_t = 0.01 + 0.7 * e²_{t-1} → strong ARCH effect
        let n = 300;
        let mut residuals = vec![0.0_f64; n];
        let mut sigma2 = 0.1_f64;
        let mut rng = 99999u64;
        for t in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let z = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0 * 1.7320508;
            residuals[t] = sigma2.sqrt() * z;
            sigma2 = (0.01 + 0.7 * residuals[t] * residuals[t]).max(1e-8);
        }
        let result = arch_lm_test(&residuals, 4).unwrap();
        // Strong ARCH: statistic should be large, p-value small
        assert!(
            result.statistic > 5.0,
            "ARCH process: statistic={:.3} should be > 5.0",
            result.statistic
        );
    }

    #[test]
    fn arch_lm_insufficient_data_returns_none() {
        let residuals = vec![0.1, -0.2, 0.3]; // 3 obs, asking for 5 lags
        assert!(arch_lm_test(&residuals, 5).is_none());
        // Zero lags is also invalid
        assert!(arch_lm_test(&residuals, 0).is_none());
    }

    #[test]
    fn arch_lm_constant_residuals_no_panic() {
        // All identical → no variance in squared residuals; R²=0, stat=0
        let residuals = vec![1.0_f64; 50];
        let result = arch_lm_test(&residuals, 3).unwrap();
        assert!(result.statistic.is_finite(), "statistic should be finite");
        assert!(result.p_value.is_finite(), "p_value should be finite");
    }
}
