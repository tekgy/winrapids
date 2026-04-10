//! # Family 18 — Volatility & Financial Time Series
//!
//! GARCH(1,1), EWMA, realized volatility, microstructure metrics.
//!
//! ## Architecture
//!
//! GARCH filter = affine prefix scan (Kingdom A), NOT Kingdom B.
//! σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1} is the affine map f_t(x) = β·x + (ω + α·r²_{t-1}).
//! The coefficient b_t = ω + α·r²_{t-1} depends on the DATA r_{t-1}, not the current state σ²_t.
//! Affine maps compose: (f_s ∘ f_t)(x) = (a_s·a_t)·x + (a_s·b_t + b_s). Semigroup law holds.
//! This is a companion-matrix prefix scan — same structure as any linear recurrence.
//! The outer parameter optimization (ω, α, β search) is Kingdom C.
//!
//! Prior label "Kingdom B" was wrong. The filter is Kingdom A; only the MLE loop is Kingdom C.
//!
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
// GARCH Variants: EGARCH, GJR-GARCH, TGARCH
// ═══════════════════════════════════════════════════════════════════════════

/// EGARCH(1,1) model result (Nelson 1991).
///
/// Variance recursion in log-space: no positivity constraints on parameters.
/// ln σ²_t = ω + β·ln σ²_{t-1} + α·g(z_{t-1})
/// where g(z) = z - E[|z|] + γ·z  (asymmetric term)
/// and E[|z|] = √(2/π) ≈ 0.7979 for N(0,1).
#[derive(Debug, Clone)]
pub struct EgarchResult {
    /// ω (log-variance constant).
    pub omega: f64,
    /// α (innovation magnitude coefficient).
    pub alpha: f64,
    /// γ (leverage/asymmetry coefficient).
    /// Negative γ → bad news amplifies volatility more than good news.
    pub gamma: f64,
    /// β (log-variance persistence).
    pub beta: f64,
    /// Log conditional variances ln σ²_t (length n).
    pub log_variances: Vec<f64>,
    /// Conditional variances σ²_t = exp(ln σ²_t) (length n).
    pub variances: Vec<f64>,
    /// Log-likelihood.
    pub log_likelihood: f64,
    /// Optimization iterations.
    pub iterations: usize,
}

impl EgarchResult {
    pub fn nan(n: usize) -> Self {
        Self {
            omega: f64::NAN, alpha: f64::NAN, gamma: f64::NAN, beta: f64::NAN,
            log_variances: vec![f64::NAN; n], variances: vec![f64::NAN; n],
            log_likelihood: f64::NAN, iterations: 0,
        }
    }
}

/// Fit EGARCH(1,1) on a return series.
///
/// EGARCH captures leverage: stock returns exhibit negative correlation between
/// past returns and current volatility. γ < 0 models this asymmetry.
///
/// Returns are assumed mean-zero (mean-adjusted before calling).
pub fn egarch11_fit(returns: &[f64], max_iter: usize) -> EgarchResult {
    let n = returns.len();
    if n < 10 { return EgarchResult::nan(n); }

    let moments = crate::descriptive::moments_ungrouped(returns);
    let uncond_var = moments.variance(0).max(1e-15);

    // E[|z|] for standard normal
    const E_ABS_Z: f64 = 0.7978845608028654; // sqrt(2/π)

    let egarch_ll = |omega: f64, alpha: f64, gamma: f64, beta: f64| -> (f64, Vec<f64>, Vec<f64>) {
        let mut lsig2 = vec![0.0_f64; n];
        lsig2[0] = uncond_var.ln(); // backcast
        for t in 1..n {
            let z = returns[t - 1] / lsig2[t - 1].exp().sqrt().max(1e-15);
            let g = z.abs() - E_ABS_Z + gamma * z;
            lsig2[t] = omega + beta * lsig2[t - 1] + alpha * g;
        }
        let sigma2: Vec<f64> = lsig2.iter().map(|&ls| ls.exp().max(1e-300)).collect();
        let ll: f64 = (0..n).map(|t| {
            -0.5 * (std::f64::consts::TAU.ln() + lsig2[t] + returns[t].powi(2) / sigma2[t])
        }).sum();
        (ll, lsig2, sigma2)
    };

    // Reparameterization for unconstrained optimization:
    // ω is free (no positivity needed — it's in log space already)
    // α is free
    // γ is free
    // β = 0.999·tanh(θ_β)  (constrained to (-0.999, 0.999) for stationarity)
    let sigmoid = |x: f64| -> f64 { 1.0 / (1.0 + (-x).exp()) };

    let to_constrained = |theta: &[f64]| -> (f64, f64, f64, f64) {
        let o = theta[0];
        let a = theta[1];
        let g = theta[2];
        let b = 0.999 * theta[3].tanh();
        (o, a, g, b)
    };

    let neg_ll = |theta: &[f64]| -> f64 {
        let (o, a, g, b) = to_constrained(theta);
        -egarch_ll(o, a, g, b).0
    };

    let neg_ll_grad = |theta: &[f64]| -> Vec<f64> {
        let delta = 1e-7;
        let f0 = neg_ll(theta);
        (0..4).map(|i| {
            let mut t = theta.to_vec();
            t[i] += delta;
            (neg_ll(&t) - f0) / delta
        }).collect()
    };

    // Initial values: ω ≈ ln(σ²)·(1-β), α=0.1, γ=-0.05, β=0.9
    let beta0 = 0.9_f64;
    let theta0 = vec![
        uncond_var.ln() * (1.0 - beta0),
        0.1,
        -0.05,
        beta0.atanh() / 0.999,
    ];

    let result = crate::optimization::lbfgs(&neg_ll, &neg_ll_grad, &theta0, 5, max_iter, 1e-8);
    let (omega, alpha, gamma, beta) = to_constrained(&result.x);
    let (log_likelihood, log_variances, variances) = egarch_ll(omega, alpha, gamma, beta);

    // Suppress unused warning on sigmoid (only used in logistic variants)
    let _ = sigmoid;

    EgarchResult {
        omega, alpha, gamma, beta,
        log_variances, variances, log_likelihood,
        iterations: result.iterations,
    }
}

/// GJR-GARCH(1,1) model result (Glosten, Jagannathan, Runkle 1993).
///
/// Also called threshold GARCH. Extends GARCH(1,1) with an asymmetric
/// shock term for negative returns:
///
/// σ²_t = ω + α·r²_{t-1} + γ·r²_{t-1}·𝟙[r_{t-1} < 0] + β·σ²_{t-1}
///
/// With γ > 0, negative shocks amplify volatility more than positive shocks.
/// Stationarity: α + γ/2 + β < 1.
#[derive(Debug, Clone)]
pub struct GjrGarchResult {
    /// ω (constant).
    pub omega: f64,
    /// α (ARCH coefficient).
    pub alpha: f64,
    /// γ (leverage/asymmetry coefficient for negative shocks).
    pub gamma: f64,
    /// β (GARCH coefficient).
    pub beta: f64,
    /// Conditional variances σ²_t (length n).
    pub variances: Vec<f64>,
    /// Log-likelihood.
    pub log_likelihood: f64,
    /// Optimization iterations.
    pub iterations: usize,
    /// True if α + γ/2 + β > 0.99 (near non-stationarity).
    pub near_unit_root: bool,
}

impl GjrGarchResult {
    pub fn nan(n: usize) -> Self {
        Self {
            omega: f64::NAN, alpha: f64::NAN, gamma: f64::NAN, beta: f64::NAN,
            variances: vec![f64::NAN; n], log_likelihood: f64::NAN,
            iterations: 0, near_unit_root: false,
        }
    }
}

/// Fit GJR-GARCH(1,1) on a return series.
///
/// GJR-GARCH is widely preferred over symmetric GARCH for equity returns
/// because it explicitly models the leverage effect.
pub fn gjr_garch11_fit(returns: &[f64], max_iter: usize) -> GjrGarchResult {
    let n = returns.len();
    if n < 10 { return GjrGarchResult::nan(n); }

    let moments = crate::descriptive::moments_ungrouped(returns);
    let uncond_var = moments.variance(0).max(1e-15);

    let gjr_ll = |omega: f64, alpha: f64, gamma: f64, beta: f64| -> (f64, Vec<f64>) {
        let mut sigma2 = vec![0.0_f64; n];
        sigma2[0] = uncond_var.max(1e-15);
        for t in 1..n {
            let r_prev = returns[t - 1];
            let indicator = if r_prev < 0.0 { 1.0 } else { 0.0 };
            sigma2[t] = omega + alpha * r_prev.powi(2) + gamma * r_prev.powi(2) * indicator
                + beta * sigma2[t - 1];
            sigma2[t] = sigma2[t].max(1e-15);
        }
        let ll: f64 = (0..n).map(|t| {
            -0.5 * (std::f64::consts::TAU.ln() + sigma2[t].ln() + returns[t].powi(2) / sigma2[t])
        }).sum();
        (ll, sigma2)
    };

    // Reparameterization:
    // θ_0 = ln(ω),    ω = exp(θ_0) > 0
    // θ_1 = ln(α),    α = exp(θ_1) > 0   (typically small positive)
    // θ_2 = ln(γ),    γ = exp(θ_2) > 0   (leverage effect positive)
    // θ_3 = logit(β/0.999),  β = 0.999·sigmoid(θ_3) ∈ (0, 0.999)
    let sigmoid = |x: f64| -> f64 { 1.0 / (1.0 + (-x).exp()) };

    let to_constrained = |theta: &[f64]| -> (f64, f64, f64, f64) {
        let o = theta[0].exp();
        // α and γ: ensure α > 0 and γ > 0 (leverage amplifies, not dampens)
        let a = theta[1].exp().min(0.49); // cap α at 0.49 to keep room for γ
        let g = theta[2].exp().min(0.49);
        let b = 0.999 * sigmoid(theta[3]);
        (o, a, g, b)
    };

    let neg_ll = |theta: &[f64]| -> f64 {
        let (o, a, g, b) = to_constrained(theta);
        // Stationarity: α + γ/2 + β < 1  (E[effective ARCH coeff] = α + γ/2)
        if a + g / 2.0 + b >= 0.999 { return 1e20; }
        -gjr_ll(o, a, g, b).0
    };

    let neg_ll_grad = |theta: &[f64]| -> Vec<f64> {
        let delta = 1e-7;
        let f0 = neg_ll(theta);
        (0..4).map(|i| {
            let mut t = theta.to_vec();
            t[i] += delta;
            (neg_ll(&t) - f0) / delta
        }).collect()
    };

    let logit = |p: f64| -> f64 { (p / (1.0 - p)).ln() };
    let theta0 = vec![
        (0.1 * uncond_var).max(1e-20).ln(),  // ω₀ = 10% of sample variance
        (-3.0_f64),                            // α₀ ≈ exp(-3) ≈ 0.05
        (-4.0_f64),                            // γ₀ ≈ exp(-4) ≈ 0.018
        logit(0.8 / 0.999),                   // β₀ = 0.8
    ];

    let result = crate::optimization::lbfgs(&neg_ll, &neg_ll_grad, &theta0, 5, max_iter, 1e-8);
    let (omega, alpha, gamma, beta) = to_constrained(&result.x);

    // Post-hoc stationarity clamp
    let persist = alpha + gamma / 2.0 + beta;
    let (alpha, gamma, beta) = if persist >= 0.999 {
        let scale = 0.998 / persist;
        (alpha * scale, gamma * scale, beta * scale)
    } else {
        (alpha, gamma, beta)
    };
    let near_unit_root = alpha + gamma / 2.0 + beta > 0.99;
    let (log_likelihood, variances) = gjr_ll(omega, alpha, gamma, beta);

    // Suppress unused warning
    let _ = sigmoid;

    GjrGarchResult {
        omega, alpha, gamma, beta,
        variances, log_likelihood,
        iterations: result.iterations,
        near_unit_root,
    }
}

/// TGARCH(1,1) model result (Zakoian 1994 / Rabemananjara-Zakoian).
///
/// Models conditional standard deviation (not variance) with asymmetry:
///
/// σ_t = ω + α⁺·r⁺_{t-1} + α⁻·r⁻_{t-1} + β·σ_{t-1}
///
/// where r⁺ = max(r, 0), r⁻ = max(-r, 0).
///
/// Advantages over GJR-GARCH: directly models σ (not σ²), which avoids
/// the Jensen inequality distortion in volatility forecasting.
#[derive(Debug, Clone)]
pub struct TgarchResult {
    /// ω (constant in σ recursion).
    pub omega: f64,
    /// α⁺ (positive shock coefficient).
    pub alpha_pos: f64,
    /// α⁻ (negative shock coefficient; > α⁺ for leverage effect).
    pub alpha_neg: f64,
    /// β (conditional volatility persistence).
    pub beta: f64,
    /// Conditional standard deviations σ_t (length n).
    pub sigma: Vec<f64>,
    /// Conditional variances σ²_t (length n).
    pub variances: Vec<f64>,
    /// Log-likelihood.
    pub log_likelihood: f64,
    /// Optimization iterations.
    pub iterations: usize,
    /// True if α⁺ E[r⁺] + α⁻ E[r⁻] + β ≥ 0.99 (approximate persistence).
    pub near_unit_root: bool,
}

impl TgarchResult {
    pub fn nan(n: usize) -> Self {
        Self {
            omega: f64::NAN, alpha_pos: f64::NAN, alpha_neg: f64::NAN, beta: f64::NAN,
            sigma: vec![f64::NAN; n], variances: vec![f64::NAN; n],
            log_likelihood: f64::NAN, iterations: 0, near_unit_root: false,
        }
    }
}

/// Fit TGARCH(1,1) on a return series.
///
/// Stationarity condition (Nelson 1990): E[ln(α⁺·ε⁺ + α⁻·ε⁻ + β)²] < 0,
/// which is approximately α⁺²/2 + α⁻²/2 + β² < 1. We use β + (α⁺+α⁻)·E[|ε|] < 1
/// with E[|ε|] = √(2/π) as a practical constraint.
pub fn tgarch11_fit(returns: &[f64], max_iter: usize) -> TgarchResult {
    let n = returns.len();
    if n < 10 { return TgarchResult::nan(n); }

    let moments = crate::descriptive::moments_ungrouped(returns);
    let uncond_var = moments.variance(0).max(1e-15);
    let uncond_sigma = uncond_var.sqrt();

    // E[|z|] for N(0,1) — used in stationarity bound
    const E_ABS_Z: f64 = 0.7978845608028654; // sqrt(2/π)

    let tgarch_ll = |omega: f64, alpha_pos: f64, alpha_neg: f64, beta: f64| -> (f64, Vec<f64>) {
        let mut sigma = vec![0.0_f64; n];
        sigma[0] = uncond_sigma.max(1e-15);
        for t in 1..n {
            let r = returns[t - 1];
            let rpos = r.max(0.0);
            let rneg = (-r).max(0.0);
            sigma[t] = omega + alpha_pos * rpos + alpha_neg * rneg + beta * sigma[t - 1];
            sigma[t] = sigma[t].max(1e-15);
        }
        let ll: f64 = (0..n).map(|t| {
            let s2 = sigma[t].powi(2);
            -0.5 * (std::f64::consts::TAU.ln() + s2.ln() + returns[t].powi(2) / s2)
        }).sum();
        (ll, sigma)
    };

    let sigmoid = |x: f64| -> f64 { 1.0 / (1.0 + (-x).exp()) };

    // Reparameterization:
    // θ_0 = ln(ω),   ω = exp(θ_0) > 0
    // θ_1 = ln(α⁺),  α⁺ = exp(θ_1) > 0
    // θ_2 = ln(α⁻),  α⁻ = exp(θ_2) > 0
    // θ_3 = logit(β/0.999),  β = 0.999·sigmoid(θ_3) ∈ (0, 0.999)
    let to_constrained = |theta: &[f64]| -> (f64, f64, f64, f64) {
        let o = theta[0].exp();
        let ap = theta[1].exp().min(0.4);
        let an = theta[2].exp().min(0.4);
        let b = 0.999 * sigmoid(theta[3]);
        (o, ap, an, b)
    };

    let neg_ll = |theta: &[f64]| -> f64 {
        let (o, ap, an, b) = to_constrained(theta);
        // Stationarity: (α⁺ + α⁻)·E[|ε|] + β < 1
        if (ap + an) * E_ABS_Z + b >= 0.999 { return 1e20; }
        -tgarch_ll(o, ap, an, b).0
    };

    let neg_ll_grad = |theta: &[f64]| -> Vec<f64> {
        let delta = 1e-7;
        let f0 = neg_ll(theta);
        (0..4).map(|i| {
            let mut t = theta.to_vec();
            t[i] += delta;
            (neg_ll(&t) - f0) / delta
        }).collect()
    };

    let logit = |p: f64| -> f64 { (p / (1.0 - p)).ln() };
    let theta0 = vec![
        (0.1 * uncond_sigma).max(1e-20).ln(), // ω₀ = 10% of sample σ
        (-3.0_f64),                             // α⁺₀ ≈ exp(-3) ≈ 0.05
        (-2.5_f64),                             // α⁻₀ ≈ exp(-2.5) ≈ 0.08 (slightly higher for leverage)
        logit(0.8 / 0.999),                    // β₀ = 0.8
    ];

    let result = crate::optimization::lbfgs(&neg_ll, &neg_ll_grad, &theta0, 5, max_iter, 1e-8);
    let (omega, alpha_pos, alpha_neg, beta) = to_constrained(&result.x);

    // Post-hoc stationarity clamp
    let persist = (alpha_pos + alpha_neg) * E_ABS_Z + beta;
    let (alpha_pos, alpha_neg, beta) = if persist >= 0.999 {
        let scale = 0.998 / persist;
        (alpha_pos * scale, alpha_neg * scale, beta * scale)
    } else {
        (alpha_pos, alpha_neg, beta)
    };
    let near_unit_root = (alpha_pos + alpha_neg) * E_ABS_Z + beta > 0.99;
    let (log_likelihood, sigma) = tgarch_ll(omega, alpha_pos, alpha_neg, beta);
    let variances: Vec<f64> = sigma.iter().map(|&s| s * s).collect();

    // Suppress unused warning
    let _ = sigmoid;

    TgarchResult {
        omega, alpha_pos, alpha_neg, beta,
        sigma, variances, log_likelihood,
        iterations: result.iterations,
        near_unit_root,
    }
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
// Range volatility estimators (OHLC-based)
// ═══════════════════════════════════════════════════════════════════════════

/// Parkinson (1980) range-based variance estimator.
///
/// σ²_P = (1/(4·ln 2)) · (ln(high/low))²
///
/// ~5x more efficient than close-to-close variance. Assumes no drift.
/// Input: open, high, low, close for a single period.
pub fn parkinson_variance(high: f64, low: f64) -> f64 {
    if high <= 0.0 || low <= 0.0 || high < low { return f64::NAN; }
    let lnhl = (high / low).ln();
    lnhl * lnhl / (4.0 * 2.0_f64.ln())
}

/// Garman-Klass (1980) OHLC variance estimator.
///
/// σ²_GK = 0.5·(ln(H/L))² - (2·ln 2 - 1)·(ln(C/O))²
///
/// ~7.4x more efficient than close-to-close. Assumes no drift.
pub fn garman_klass_variance(open: f64, high: f64, low: f64, close: f64) -> f64 {
    if open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0 { return f64::NAN; }
    let lnhl = (high / low).ln();
    let lnco = (close / open).ln();
    0.5 * lnhl * lnhl - (2.0 * 2.0_f64.ln() - 1.0) * lnco * lnco
}

/// Rogers-Satchell (1991) drift-independent variance estimator.
///
/// σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)
///
/// Unlike Parkinson and Garman-Klass, handles non-zero drift.
pub fn rogers_satchell_variance(open: f64, high: f64, low: f64, close: f64) -> f64 {
    if open <= 0.0 || high <= 0.0 || low <= 0.0 || close <= 0.0 { return f64::NAN; }
    let lnho = (high / open).ln();
    let lnhc = (high / close).ln();
    let lnlo = (low / open).ln();
    let lnlc = (low / close).ln();
    lnhc * lnho + lnlc * lnlo
}

/// Yang-Zhang (2000) drift-independent variance estimator.
///
/// σ²_YZ = σ²_O + k·σ²_C + (1-k)·σ²_RS
///
/// where σ²_O is overnight variance (prev_close → open), σ²_C is close-to-close,
/// and k is chosen to minimize estimator variance.
///
/// Most efficient OHLC estimator; requires prev_close for overnight returns.
/// This function computes YZ across a series of bars (windows).
///
/// Returns the YZ variance estimate for the full sample.
pub fn yang_zhang_variance(
    opens: &[f64], highs: &[f64], lows: &[f64], closes: &[f64], prev_closes: &[f64],
) -> f64 {
    let n = opens.len();
    if n < 2 || highs.len() != n || lows.len() != n || closes.len() != n || prev_closes.len() != n {
        return f64::NAN;
    }

    // Overnight returns: ln(O_t / C_{t-1})
    // Close-to-close returns: ln(C_t / C_{t-1})
    let mut over_returns = Vec::with_capacity(n);
    let mut cc_returns = Vec::with_capacity(n);
    let mut rs_sum = 0.0;
    for i in 0..n {
        if prev_closes[i] <= 0.0 || opens[i] <= 0.0 || closes[i] <= 0.0 {
            return f64::NAN;
        }
        over_returns.push((opens[i] / prev_closes[i]).ln());
        cc_returns.push((closes[i] / prev_closes[i]).ln());
        rs_sum += rogers_satchell_variance(opens[i], highs[i], lows[i], closes[i]);
    }
    let nf = n as f64;
    let over_mean = over_returns.iter().sum::<f64>() / nf;
    let cc_mean = cc_returns.iter().sum::<f64>() / nf;
    let sigma2_o: f64 = over_returns.iter().map(|x| (x - over_mean).powi(2)).sum::<f64>() / (nf - 1.0);
    let sigma2_c: f64 = cc_returns.iter().map(|x| (x - cc_mean).powi(2)).sum::<f64>() / (nf - 1.0);
    let sigma2_rs = rs_sum / nf;

    // Yang-Zhang k (optimal weighting): k = 0.34 / (1.34 + (n+1)/(n-1))
    let k = 0.34 / (1.34 + (nf + 1.0) / (nf - 1.0));
    sigma2_o + k * sigma2_c + (1.0 - k) * sigma2_rs
}

// ═══════════════════════════════════════════════════════════════════════════
// Hill tail index estimator
// ═══════════════════════════════════════════════════════════════════════════

/// Hill (1975) tail index estimator.
///
/// For heavy-tailed data, fits a Pareto tail. Returns 1/α where α is
/// the tail exponent. Larger → lighter tail.
///
/// `data`: sample (will be sorted internally).
/// `k`: number of order statistics to use (tail depth). Typically 0.05n to 0.1n.
///
/// Formula: ξ̂ = (1/k) · Σ_{i=1}^k ln(X_{(n-i+1)} / X_{(n-k)})
///
/// where X_{(i)} are order statistics in ascending order.
pub fn hill_estimator(data: &[f64], k: usize) -> f64 {
    let n = data.len();
    if n == 0 || k == 0 || k >= n { return f64::NAN; }
    // Use absolute values for two-sided tails
    let mut sorted: Vec<f64> = data.iter().map(|x| x.abs()).filter(|x| x.is_finite()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let ns = sorted.len();
    if k >= ns { return f64::NAN; }
    // The threshold is the (ns - k)-th order statistic (k values above it)
    let threshold = sorted[ns - k - 1];
    if threshold <= 0.0 { return f64::NAN; }
    // Sum ln(X_i / threshold) for the top k values
    let mut sum = 0.0_f64;
    for i in 0..k {
        let x = sorted[ns - 1 - i];
        if x > 0.0 {
            sum += (x / threshold).ln();
        }
    }
    sum / k as f64 // xi_hat = 1 / alpha
}

/// Hill tail index (alpha) from the estimator.
///
/// Returns α = 1/ξ̂. For financial returns, α ∈ [2, 5] is common.
/// α < 2 implies infinite variance.
pub fn hill_tail_alpha(data: &[f64], k: usize) -> f64 {
    let xi = hill_estimator(data, k);
    if xi > 1e-15 { 1.0 / xi } else { f64::NAN }
}

/// Tripower Quarticity (TQ): fourth-moment analogue of bipower variation.
///
/// TQ = n · μ_{4/3}^{-3} · Σ |r_{t-2}|^{4/3} · |r_{t-1}|^{4/3} · |r_t|^{4/3}
///
/// where μ_{4/3} = 2^{2/3} · Γ(7/6) / Γ(1/2). Numerically, μ_{4/3} ≈ 0.8309,
/// so μ_{4/3}^3 ≈ 0.5736 and 1/μ_{4/3}^3 ≈ 1.7434.
///
/// Used as the variance-of-BV in the Barndorff-Nielsen-Shephard jump test.
pub fn tripower_quarticity(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 3 { return f64::NAN; }
    const MU_43_CUBED: f64 = 0.5736;
    let mut sum = 0.0_f64;
    for t in 2..n {
        let a = returns[t - 2].abs().powf(4.0 / 3.0);
        let b = returns[t - 1].abs().powf(4.0 / 3.0);
        let c = returns[t].abs().powf(4.0 / 3.0);
        sum += a * b * c;
    }
    (n as f64 / MU_43_CUBED) * sum
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
// VPIN — Volume-synchronized Probability of Informed Trading
// ═══════════════════════════════════════════════════════════════════════════

/// VPIN result (Easley, Lopez de Prado, O'Hara 2012).
#[derive(Debug, Clone)]
pub struct VpinResult {
    /// VPIN values per volume bucket.
    pub vpin: Vec<f64>,
    /// Number of volume buckets formed.
    pub n_buckets: usize,
}

/// Compute VPIN via Bulk Volume Classification (BVC).
///
/// BVC classifies each trade's volume as buy/sell using the CDF of
/// standardized price changes: V_buy = V * Φ(ΔP / σ_ΔP), V_sell = V - V_buy.
///
/// Volume is partitioned into equal-sized buckets of `bucket_volume`.
/// VPIN per bucket = |V_buy - V_sell| / V_bucket, averaged over `n_avg`
/// trailing buckets.
///
/// # Parameters
/// - `prices`: trade prices (length n)
/// - `volumes`: trade volumes (length n, must be positive)
/// - `bucket_volume`: target volume per bucket (e.g., daily_volume / 50)
/// - `n_avg`: number of trailing buckets to average (e.g., 50)
///
/// Returns `VpinResult` with one VPIN value per bucket starting from bucket `n_avg`.
pub fn vpin_bvc(prices: &[f64], volumes: &[f64], bucket_volume: f64, n_avg: usize) -> VpinResult {
    let n = prices.len();
    if n < 2 || volumes.len() != n || bucket_volume <= 0.0 || n_avg == 0 {
        return VpinResult { vpin: vec![], n_buckets: 0 };
    }

    // Price changes and their standard deviation
    let dp: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();
    let mean_dp = dp.iter().sum::<f64>() / dp.len() as f64;
    let var_dp = dp.iter().map(|d| (d - mean_dp).powi(2)).sum::<f64>() / dp.len().max(1) as f64;
    let sigma = var_dp.sqrt().max(1e-15);

    // BVC: classify each trade (starting from index 1) as buy/sell fraction
    // using Φ(ΔP / σ)
    let mut buy_vol_accum = 0.0_f64;
    let mut total_vol_accum = 0.0_f64;
    let mut bucket_order_imbalances: Vec<f64> = Vec::new();

    for i in 0..dp.len() {
        let z = dp[i] / sigma;
        let phi_z = crate::special_functions::normal_cdf(z); // Φ(z)
        let v = volumes[i + 1]; // trade at index i+1
        let v_buy = v * phi_z;
        let v_sell = v - v_buy;

        buy_vol_accum += v_buy;
        total_vol_accum += v;

        // Check if bucket is full
        while total_vol_accum >= bucket_volume {
            let overflow = total_vol_accum - bucket_volume;
            // Fraction of this trade that belongs to the current bucket
            let frac_in = if v > 1e-15 { 1.0 - overflow / v } else { 1.0 };
            let adj_buy = buy_vol_accum - v_buy * (1.0 - frac_in.clamp(0.0, 1.0));
            let adj_sell = bucket_volume - adj_buy;
            let oi = (adj_buy - adj_sell).abs() / bucket_volume;
            bucket_order_imbalances.push(oi);

            // Carry remainder to next bucket
            buy_vol_accum = v_buy * (1.0 - frac_in.clamp(0.0, 1.0));
            total_vol_accum = overflow;
        }
    }

    let nb = bucket_order_imbalances.len();
    if nb < n_avg {
        return VpinResult { vpin: vec![], n_buckets: nb };
    }

    // Rolling average of order imbalance over n_avg buckets
    let mut vpin_vals = Vec::with_capacity(nb - n_avg + 1);
    let mut sum: f64 = bucket_order_imbalances[..n_avg].iter().sum();
    vpin_vals.push(sum / n_avg as f64);
    for i in n_avg..nb {
        sum += bucket_order_imbalances[i] - bucket_order_imbalances[i - n_avg];
        vpin_vals.push(sum / n_avg as f64);
    }

    VpinResult { vpin: vpin_vals, n_buckets: nb }
}

// ═══════════════════════════════════════════════════════════════════════════
// Visibility Graphs (NVG / HVG)
// ═══════════════════════════════════════════════════════════════════════════

/// Natural Visibility Graph degree sequence (Lacasa et al. 2008).
///
/// Two samples (t_a, y_a) and (t_b, y_b) are connected if all intermediate
/// points (t_c, y_c) satisfy: y_c < y_a + (y_b - y_a) * (t_c - t_a) / (t_b - t_a).
///
/// Returns the degree of each node (length = data.len()).
pub fn nvg_degree(data: &[f64]) -> Vec<u32> {
    let n = data.len();
    if n == 0 { return vec![]; }
    let mut degree = vec![0u32; n];

    for a in 0..n {
        for b in (a + 1)..n {
            // Check visibility: all c in (a+1..b) must satisfy the criterion
            let ya = data[a];
            let yb = data[b];
            let span = (b - a) as f64;
            let visible = (a + 1..b).all(|c| {
                let t_frac = (c - a) as f64 / span;
                data[c] < ya + (yb - ya) * t_frac
            });
            if visible {
                degree[a] += 1;
                degree[b] += 1;
            }
        }
    }
    degree
}

/// Horizontal Visibility Graph degree sequence (Luque et al. 2009).
///
/// Two samples (t_a, y_a) and (t_b, y_b) are connected if all intermediate
/// points satisfy: y_c < min(y_a, y_b).
///
/// Returns the degree of each node (length = data.len()).
pub fn hvg_degree(data: &[f64]) -> Vec<u32> {
    let n = data.len();
    if n == 0 { return vec![]; }
    let mut degree = vec![0u32; n];

    for a in 0..n {
        for b in (a + 1)..n {
            let threshold = data[a].min(data[b]);
            let visible = (a + 1..b).all(|c| data[c] < threshold);
            if visible {
                degree[a] += 1;
                degree[b] += 1;
            }
        }
    }
    degree
}

/// Mean degree of the natural visibility graph.
///
/// For random iid series, E[degree] → 4 (Lacasa et al. 2008).
/// For periodic series, degree distribution has characteristic peaks.
pub fn nvg_mean_degree(data: &[f64]) -> f64 {
    let deg = nvg_degree(data);
    if deg.is_empty() { return f64::NAN; }
    deg.iter().map(|&d| d as f64).sum::<f64>() / deg.len() as f64
}

/// Mean degree of the horizontal visibility graph.
///
/// For random iid series, E[degree] = 4 exactly.
pub fn hvg_mean_degree(data: &[f64]) -> f64 {
    let deg = hvg_degree(data);
    if deg.is_empty() { return f64::NAN; }
    deg.iter().map(|&d| d as f64).sum::<f64>() / deg.len() as f64
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

    // ── EGARCH tests ──────────────────────────────────────────────────────

    fn make_garch_returns(n: usize, omega: f64, alpha: f64, beta: f64, seed: u64) -> Vec<f64> {
        let mut returns = vec![0.0; n];
        let mut sigma2 = omega / (1.0 - alpha - beta).max(1e-10);
        let mut rng = crate::rng::Xoshiro256::new(seed);
        for t in 0..n {
            let z = crate::rng::sample_normal(&mut rng, 0.0, 1.0);
            returns[t] = sigma2.sqrt() * z;
            sigma2 = omega + alpha * returns[t].powi(2) + beta * sigma2;
        }
        returns
    }

    #[test]
    fn egarch_fits_and_stationary() {
        let returns = make_garch_returns(500, 0.001, 0.1, 0.85, 42);
        let r = egarch11_fit(&returns, 200);
        assert!(r.omega.is_finite(), "ω should be finite, got {}", r.omega);
        assert!(r.alpha.is_finite(), "α should be finite, got {}", r.alpha);
        assert!(r.beta.is_finite(), "β should be finite, got {}", r.beta);
        // EGARCH stationarity: |β| < 1
        assert!(r.beta.abs() < 1.0, "EGARCH |β|={:.4} must be < 1", r.beta);
        assert_eq!(r.log_variances.len(), 500);
        assert_eq!(r.variances.len(), 500);
        // All variances must be positive
        for &v in &r.variances {
            assert!(v > 0.0 && v.is_finite(), "variance should be positive finite, got {v}");
        }
    }

    #[test]
    fn egarch_log_likelihood_finite() {
        let returns = make_garch_returns(200, 0.001, 0.1, 0.85, 7);
        let r = egarch11_fit(&returns, 100);
        // LL can be positive for scaled Gaussians (density > 1 is valid)
        assert!(r.log_likelihood.is_finite(), "LL should be finite, got {}", r.log_likelihood);
    }

    #[test]
    fn egarch_too_short_returns_nan() {
        let r = egarch11_fit(&[0.01, -0.02, 0.03], 100);
        assert!(r.omega.is_nan());
    }

    // ── GJR-GARCH tests ───────────────────────────────────────────────────

    #[test]
    fn gjr_garch_fits_and_stationary() {
        let returns = make_garch_returns(500, 0.001, 0.08, 0.85, 99);
        let r = gjr_garch11_fit(&returns, 200);
        assert!(r.omega.is_finite(), "ω={}", r.omega);
        assert!(r.alpha >= 0.0 && r.alpha.is_finite(), "α={}", r.alpha);
        assert!(r.gamma >= 0.0 && r.gamma.is_finite(), "γ={}", r.gamma);
        assert!(r.beta >= 0.0 && r.beta.is_finite(), "β={}", r.beta);
        // Stationarity: α + γ/2 + β < 1
        let persist = r.alpha + r.gamma / 2.0 + r.beta;
        assert!(persist < 1.0, "GJR stationarity persist={:.4} must be < 1", persist);
        assert_eq!(r.variances.len(), 500);
        for &v in &r.variances { assert!(v > 0.0, "variance should be positive, got {v}"); }
    }

    #[test]
    fn gjr_garch_leverage_positive() {
        // On symmetric GARCH data, γ should still be non-negative
        // (the optimizer won't push it negative — the constraint floor is 0)
        let returns = make_garch_returns(300, 0.001, 0.1, 0.8, 55);
        let r = gjr_garch11_fit(&returns, 200);
        assert!(r.gamma >= 0.0, "leverage γ={:.4} should be non-negative", r.gamma);
    }

    #[test]
    fn gjr_garch_log_likelihood_finite() {
        let returns = make_garch_returns(200, 0.001, 0.1, 0.8, 11);
        let r = gjr_garch11_fit(&returns, 100);
        // LL can be positive for scaled Gaussians (density > 1 is valid)
        assert!(r.log_likelihood.is_finite(), "LL={}", r.log_likelihood);
    }

    #[test]
    fn gjr_garch_too_short_returns_nan() {
        let r = gjr_garch11_fit(&[0.01, -0.02], 100);
        assert!(r.omega.is_nan());
    }

    // ── TGARCH tests ──────────────────────────────────────────────────────

    #[test]
    fn tgarch_fits_and_stationary() {
        let returns = make_garch_returns(500, 0.001, 0.08, 0.85, 31);
        let r = tgarch11_fit(&returns, 200);
        assert!(r.omega.is_finite() && r.omega > 0.0, "ω={}", r.omega);
        assert!(r.alpha_pos >= 0.0 && r.alpha_pos.is_finite(), "α⁺={}", r.alpha_pos);
        assert!(r.alpha_neg >= 0.0 && r.alpha_neg.is_finite(), "α⁻={}", r.alpha_neg);
        assert!(r.beta >= 0.0 && r.beta.is_finite(), "β={}", r.beta);
        assert_eq!(r.sigma.len(), 500);
        assert_eq!(r.variances.len(), 500);
        for &s in &r.sigma { assert!(s > 0.0, "sigma should be positive, got {s}"); }
        // Stationarity: (α⁺ + α⁻)·E[|z|] + β < 1
        const E_ABS_Z: f64 = 0.7978845608028654;
        let persist = (r.alpha_pos + r.alpha_neg) * E_ABS_Z + r.beta;
        assert!(persist < 1.0, "TGARCH persist={:.4} must be < 1", persist);
    }

    #[test]
    fn tgarch_sigma_squared_equals_variances() {
        let returns = make_garch_returns(200, 0.001, 0.1, 0.8, 17);
        let r = tgarch11_fit(&returns, 100);
        for (i, (&s, &v)) in r.sigma.iter().zip(r.variances.iter()).enumerate() {
            assert!((s * s - v).abs() < 1e-10, "t={i}: σ²={} but variances={}. diff={}", s*s, v, (s*s-v).abs());
        }
    }

    #[test]
    fn tgarch_log_likelihood_finite() {
        let returns = make_garch_returns(200, 0.001, 0.1, 0.8, 23);
        let r = tgarch11_fit(&returns, 100);
        // LL can be positive for scaled Gaussians (density > 1 is valid)
        assert!(r.log_likelihood.is_finite(), "LL={}", r.log_likelihood);
    }

    #[test]
    fn tgarch_too_short_returns_nan() {
        let r = tgarch11_fit(&[0.01], 100);
        assert!(r.omega.is_nan());
    }
}
