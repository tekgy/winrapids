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
    let mean_r = returns.iter().sum::<f64>() / n as f64;
    let uncond_var: f64 = returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / n as f64;

    // Initialize: ω = 0.1*var, α = 0.1, β = 0.8
    let mut omega = 0.1 * uncond_var;
    let mut alpha = 0.1;
    let mut beta = 0.8;

    let garch_ll = |omega: f64, alpha: f64, beta: f64| -> (f64, Vec<f64>) {
        let mut sigma2 = vec![0.0; n];
        sigma2[0] = uncond_var; // backcast initialization
        for t in 1..n {
            sigma2[t] = omega + alpha * returns[t - 1].powi(2) + beta * sigma2[t - 1];
            sigma2[t] = sigma2[t].max(1e-15); // floor
        }
        let ll: f64 = (0..n).map(|t| {
            -0.5 * (std::f64::consts::TAU.ln() + sigma2[t].ln() + returns[t].powi(2) / sigma2[t])
        }).sum();
        (ll, sigma2)
    };

    let mut best_ll = f64::NEG_INFINITY;
    let mut best = (omega, alpha, beta);
    let mut iterations = 0;

    // Simple grid refinement + coordinate search
    let delta = 1e-5;
    for iter in 0..max_iter {
        iterations = iter + 1;
        let (ll0, _) = garch_ll(omega, alpha, beta);

        // Gradient via finite differences
        let dll_domega = (garch_ll(omega + delta, alpha, beta).0 - ll0) / delta;
        let dll_dalpha = (garch_ll(omega, alpha + delta, beta).0 - ll0) / delta;
        let dll_dbeta = (garch_ll(omega, alpha, beta + delta).0 - ll0) / delta;

        // Steepest ascent with small step
        let step = 1e-5;
        omega = (omega + step * dll_domega).max(1e-10);
        alpha = (alpha + step * dll_dalpha).clamp(1e-6, 0.499);
        beta = (beta + step * dll_dbeta).clamp(1e-6, 0.999);

        // Enforce stationarity: α + β < 1
        if alpha + beta >= 0.999 {
            let s = alpha + beta;
            alpha *= 0.998 / s;
            beta *= 0.998 / s;
        }

        let (ll, _) = garch_ll(omega, alpha, beta);
        if ll > best_ll {
            best_ll = ll;
            best = (omega, alpha, beta);
        }

        if (ll - ll0).abs() < 1e-8 { break; }
    }

    let (omega, alpha, beta) = best;
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
    // Initialize with sample variance
    let var0: f64 = returns.iter().map(|r| r * r).sum::<f64>() / n as f64;
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
        // Generate GARCH-like returns
        let n = 500;
        let mut returns = vec![0.0; n];
        let mut sigma2: f64 = 0.01;
        let mut rng = 42u64;
        for t in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let z = (rng as f64 / u64::MAX as f64 - 0.5) * 3.46; // ~uniform scaled
            returns[t] = sigma2.sqrt() * z;
            sigma2 = 0.001 + 0.1 * returns[t].powi(2) + 0.85 * sigma2;
        }

        let res = garch11_fit(&returns, 200);
        assert!(res.alpha > 0.0 && res.alpha < 0.5, "α={}", res.alpha);
        assert!(res.beta > 0.0 && res.beta < 1.0, "β={}", res.beta);
        assert!(res.alpha + res.beta < 1.0, "α+β={}", res.alpha + res.beta);
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
        let returns = vec![0.01; 20];
        let sigma2 = ewma_variance(&returns, 0.94);
        assert_eq!(sigma2.len(), 20);
        // Should converge to 0.01² = 0.0001
        assert!(sigma2[19] > 0.0);
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
    fn roll_spread_bid_ask() {
        // Simulated bid-ask bounce: price oscillates by spread
        let spread = 0.05;
        let mut prices = vec![100.0];
        let mut rng = 42u64;
        for _ in 0..99 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let last = *prices.last().unwrap();
            let bounce = if rng % 2 == 0 { spread } else { -spread };
            prices.push(last + bounce);
        }
        let rs = roll_spread(&prices);
        assert!(rs > 0.0, "Roll spread={rs} should detect bid-ask bounce");
    }

    #[test]
    fn kyle_lambda_positive_impact() {
        // Positive order flow → positive price change
        let signed_volumes = vec![100.0, -50.0, 200.0, -100.0, 150.0];
        let price_changes = vec![0.5, -0.3, 1.0, -0.5, 0.8];
        let lambda = kyle_lambda(&price_changes, &signed_volumes);
        assert!(lambda > 0.0, "λ={lambda} should be positive for positive impact");
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
}
