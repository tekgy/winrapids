//! # Family 17 — Time Series Models
//!
//! ARMA, ARIMA, exponential smoothing, unit root tests, Granger causality.
//!
//! ## Architecture
//!
//! AR/MA fitting = Yule-Walker (Kingdom A: Toeplitz solve from autocorrelation).
//! ARIMA = difference + ARMA. Exponential smoothing = sequential scan (Kingdom B).
//! Unit root tests = regression-based (ADF reuses F10 OLS).

// ═══════════════════════════════════════════════════════════════════════════
// AR model (Yule-Walker)
// ═══════════════════════════════════════════════════════════════════════════

/// AR(p) model result.
#[derive(Debug, Clone)]
pub struct ArResult {
    /// AR coefficients φ₁, ..., φₚ.
    pub coefficients: Vec<f64>,
    /// Innovation variance σ².
    pub sigma2: f64,
    /// AIC.
    pub aic: f64,
}

/// Fit AR(p) model via Yule-Walker equations.
/// Uses Levinson-Durbin recursion (O(p²), numerically stable).
pub fn ar_fit(data: &[f64], p: usize) -> ArResult {
    let n = data.len();
    assert!(n > p + 1);

    let moments = crate::descriptive::moments_ungrouped(data);
    let mean = moments.mean();
    let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();

    // Autocorrelation r(0), r(1), ..., r(p)
    let mut r = vec![0.0; p + 1];
    for lag in 0..=p {
        for t in lag..n {
            r[lag] += centered[t] * centered[t - lag];
        }
        r[lag] /= n as f64;
    }

    // Levinson-Durbin
    let mut phi = vec![0.0; p];
    let mut sigma2 = r[0];
    if sigma2 < 1e-15 {
        // Constant series — zero variance, no AR structure
        let ll = -0.5 * n as f64 * (2.0 * std::f64::consts::PI * 1e-15_f64).ln() - n as f64 / 2.0;
        return ArResult { coefficients: phi, sigma2: 0.0, aic: -2.0 * ll + 2.0 * (p + 1) as f64 };
    }
    if p == 0 {
        let ll = -0.5 * n as f64 * (2.0 * std::f64::consts::PI * sigma2).ln() - n as f64 / 2.0;
        return ArResult { coefficients: phi, sigma2, aic: -2.0 * ll + 2.0 };
    }

    let mut prev = vec![0.0; p];

    for k in 0..p {
        // Reflection coefficient
        let mut num = r[k + 1];
        for j in 0..k { num -= prev[j] * r[k - j]; }
        let kappa = num / sigma2;

        // Update coefficients
        phi[k] = kappa;
        for j in 0..k { phi[j] = prev[j] - kappa * prev[k - 1 - j]; }

        sigma2 *= 1.0 - kappa * kappa;
        prev[..=k].copy_from_slice(&phi[..=k]);
    }

    let ll = -0.5 * n as f64 * (2.0 * std::f64::consts::PI * sigma2).ln() - n as f64 / 2.0;
    let aic = -2.0 * ll + 2.0 * (p + 1) as f64;

    ArResult { coefficients: phi, sigma2, aic }
}

/// Predict next values using AR model.
pub fn ar_predict(data: &[f64], ar: &ArResult, horizon: usize) -> Vec<f64> {
    let mean = crate::descriptive::moments_ungrouped(data).mean();
    let p = ar.coefficients.len();
    let mut buf: Vec<f64> = data.iter().map(|x| x - mean).collect();
    let mut preds = Vec::with_capacity(horizon);

    for _ in 0..horizon {
        let n = buf.len();
        let mut val = 0.0;
        for j in 0..p {
            if n > j { val += ar.coefficients[j] * buf[n - 1 - j]; }
        }
        buf.push(val);
        preds.push(val + mean);
    }
    preds
}

// ═══════════════════════════════════════════════════════════════════════════
// Differencing (for ARIMA)
// ═══════════════════════════════════════════════════════════════════════════

/// Difference a time series d times.
pub fn difference(data: &[f64], d: usize) -> Vec<f64> {
    let mut result = data.to_vec();
    for _ in 0..d {
        result = result.windows(2).map(|w| w[1] - w[0]).collect();
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Exponential Smoothing
// ═══════════════════════════════════════════════════════════════════════════

/// Simple exponential smoothing result.
#[derive(Debug, Clone)]
pub struct SesResult {
    /// Smoothed values.
    pub fitted: Vec<f64>,
    /// Forecast (next value).
    pub forecast: f64,
    /// Smoothing parameter α.
    pub alpha: f64,
}

/// Simple Exponential Smoothing. `alpha` ∈ (0, 1).
pub fn simple_exponential_smoothing(data: &[f64], alpha: f64) -> SesResult {
    let n = data.len();
    assert!(n > 0);
    let mut fitted = Vec::with_capacity(n);
    let mut level = data[0];
    fitted.push(level);

    for i in 1..n {
        level = alpha * data[i] + (1.0 - alpha) * level;
        fitted.push(level);
    }

    SesResult { fitted, forecast: level, alpha }
}

/// Holt's linear trend method (double exponential smoothing).
pub fn holt_linear(data: &[f64], alpha: f64, beta: f64, horizon: usize) -> Vec<f64> {
    let n = data.len();
    assert!(n >= 2);
    let mut level = data[0];
    let mut trend = data[1] - data[0];

    for i in 1..n {
        let new_level = alpha * data[i] + (1.0 - alpha) * (level + trend);
        let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
        level = new_level;
        trend = new_trend;
    }

    (1..=horizon).map(|h| level + h as f64 * trend).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// ADF unit root test (simplified)
// ═══════════════════════════════════════════════════════════════════════════

/// ADF test result.
#[derive(Debug, Clone)]
pub struct AdfResult {
    /// ADF test statistic.
    pub statistic: f64,
    /// Number of lags used.
    pub n_lags: usize,
    /// Critical values at 1%, 5%, 10% (asymptotic MacKinnon approximations).
    pub critical_1pct: f64,
    pub critical_5pct: f64,
    pub critical_10pct: f64,
}

/// Augmented Dickey-Fuller test for unit root.
/// Tests H₀: unit root (non-stationary) vs H₁: stationary.
pub fn adf_test(data: &[f64], n_lags: usize) -> AdfResult {
    let n = data.len();
    if n <= n_lags + 2 {
        return AdfResult {
            statistic: f64::NAN,
            n_lags,
            critical_1pct: -3.43,
            critical_5pct: -2.86,
            critical_10pct: -2.57,
        };
    }

    let dy: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
    let m = dy.len();

    // Regression: Δy_t = α + γ·y_{t-1} + Σ δ_j·Δy_{t-j} + ε_t
    // γ = 0 under H₀ (unit root)
    let p = 2 + n_lags; // intercept + y_{t-1} + lagged differences
    let nobs = m - n_lags;
    let mut x = vec![0.0; nobs * p];
    let mut y_reg = vec![0.0; nobs];

    for t in n_lags..m {
        let row = t - n_lags;
        y_reg[row] = dy[t];
        x[row * p] = 1.0; // intercept
        x[row * p + 1] = data[t]; // y_{t-1} (note: data[t] corresponds to y before difference at t)
        for j in 0..n_lags {
            x[row * p + 2 + j] = dy[t - 1 - j];
        }
    }

    // OLS: (X'X)⁻¹ X'y
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for i in 0..nobs {
        for j in 0..p {
            xty[j] += x[i * p + j] * y_reg[i];
            for k in 0..p {
                xtx[j * p + k] += x[i * p + j] * x[i * p + k];
            }
        }
    }

    if nobs <= p {
        return AdfResult {
            statistic: f64::NAN,
            n_lags,
            critical_1pct: -3.43,
            critical_5pct: -2.86,
            critical_10pct: -2.57,
        };
    }

    let a = crate::linear_algebra::Mat::from_vec(p, p, xtx);
    let l = match crate::linear_algebra::cholesky(&a) {
        Some(l) => l,
        None => {
            return AdfResult {
                statistic: f64::NAN,
                n_lags,
                critical_1pct: -3.43,
                critical_5pct: -2.86,
                critical_10pct: -2.57,
            };
        }
    };
    let beta = crate::linear_algebra::cholesky_solve(&l, &xty);

    // Standard error of γ (coefficient on y_{t-1})
    let mut ss_resid = 0.0;
    for i in 0..nobs {
        let fitted: f64 = (0..p).map(|j| x[i * p + j] * beta[j]).sum();
        ss_resid += (y_reg[i] - fitted).powi(2);
    }
    let mse = ss_resid / (nobs - p) as f64;

    // Diagonal of (X'X)⁻¹
    let mut ej = vec![0.0; p];
    ej[1] = 1.0;
    let inv_col = crate::linear_algebra::cholesky_solve(&l, &ej);
    let se_gamma = (mse * inv_col[1]).sqrt();

    let statistic = beta[1] / se_gamma;

    // MacKinnon critical values (constant + trend, asymptotic)
    AdfResult {
        statistic,
        n_lags,
        critical_1pct: -3.43,
        critical_5pct: -2.86,
        critical_10pct: -2.57,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Autocorrelation / Partial autocorrelation
// ═══════════════════════════════════════════════════════════════════════════

/// Sample autocorrelation function for lags 0..max_lag.
pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let max_lag = max_lag.min(n.saturating_sub(1));
    let moments = crate::descriptive::moments_ungrouped(data);
    let mean = moments.mean();
    let var = moments.variance(0);
    if var < 1e-15 { return vec![1.0; max_lag + 1]; }

    (0..=max_lag).map(|lag| {
        let r: f64 = data[..n - lag].iter()
            .zip(data[lag..].iter())
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum::<f64>() / (n as f64 * var);
        r
    }).collect()
}

/// Partial autocorrelation function via Levinson-Durbin.
pub fn pacf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let moments = crate::descriptive::moments_ungrouped(data);
    let mean = moments.mean();
    let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();

    let mut r = vec![0.0; max_lag + 1];
    let var = moments.variance(0);
    for lag in 0..=max_lag {
        r[lag] = centered[..n - lag].iter()
            .zip(centered[lag..].iter())
            .map(|(a, b)| a * b)
            .sum::<f64>() / (n as f64 * var);
    }

    let mut pacf_vals = vec![0.0; max_lag + 1];
    pacf_vals[0] = 1.0;
    if max_lag == 0 { return pacf_vals; }

    let mut phi = vec![0.0; max_lag];
    let mut sigma2 = 1.0; // r[lag] is normalized autocorrelation (ρ), so ρ[0] = 1

    for k in 0..max_lag {
        let mut num = r[k + 1];
        for j in 0..k { num -= phi[j] * r[k - j]; }
        let kappa = num / sigma2;
        pacf_vals[k + 1] = kappa;

        let prev = phi[..k].to_vec();
        phi[k] = kappa;
        for j in 0..k { phi[j] = prev[j] - kappa * prev[k - 1 - j]; }
        sigma2 *= 1.0 - kappa * kappa;
    }

    pacf_vals
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
    fn ar1_recovery() {
        // Generate AR(1) with φ = 0.8
        let n = 1000;
        let phi_true = 0.8;
        let mut data = vec![0.0; n];
        let mut rng = 42u64;
        for t in 1..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
            data[t] = phi_true * data[t - 1] + noise;
        }
        let res = ar_fit(&data, 1);
        assert!((res.coefficients[0] - phi_true).abs() < 0.15,
            "AR(1) coeff={} should be ~{}", res.coefficients[0], phi_true);
    }

    #[test]
    fn ar_predict_constant() {
        let data = vec![5.0; 20];
        let ar = ar_fit(&data, 1);
        let preds = ar_predict(&data, &ar, 5);
        for &p in &preds {
            assert!((p - 5.0).abs() < 0.1, "Constant series should predict constant, got {p}");
        }
    }

    #[test]
    fn difference_once() {
        let data = vec![1.0, 3.0, 6.0, 10.0];
        let diff = difference(&data, 1);
        close(diff[0], 2.0, 1e-10, "diff[0]");
        close(diff[1], 3.0, 1e-10, "diff[1]");
        close(diff[2], 4.0, 1e-10, "diff[2]");
    }

    #[test]
    fn difference_twice() {
        let data = vec![1.0, 3.0, 6.0, 10.0];
        let diff = difference(&data, 2);
        close(diff[0], 1.0, 1e-10, "ddiff[0]"); // 3-2=1
        close(diff[1], 1.0, 1e-10, "ddiff[1]"); // 4-3=1
    }

    #[test]
    fn ses_constant() {
        let data = vec![5.0; 10];
        let res = simple_exponential_smoothing(&data, 0.3);
        close(res.forecast, 5.0, 1e-10, "SES forecast for constant");
    }

    #[test]
    fn holt_linear_trend() {
        // Linear trend: y = 2t
        let data: Vec<f64> = (0..20).map(|i| 2.0 * i as f64).collect();
        let preds = holt_linear(&data, 0.8, 0.2, 3);
        // Should extrapolate the trend
        assert!(preds[0] > 38.0 && preds[0] < 42.0, "Holt pred[0]={}", preds[0]);
    }

    #[test]
    fn acf_white_noise() {
        // Pseudo-white noise → ACF near 0 for lag > 0
        let mut data = vec![0.0; 500];
        let mut rng = 42u64;
        for v in data.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = rng as f64 / u64::MAX as f64;
        }
        let ac = acf(&data, 10);
        close(ac[0], 1.0, 1e-10, "ACF(0)");
        for lag in 1..=10 {
            assert!(ac[lag].abs() < 0.2, "ACF({lag})={} should be ~0 for white noise", ac[lag]);
        }
    }

    #[test]
    fn adf_stationary_rejects() {
        // Stationary AR(1) with small φ → should reject unit root
        let n = 200;
        let mut data = vec![0.0; n];
        let mut rng = 42u64;
        for t in 1..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
            data[t] = 0.3 * data[t - 1] + noise;
        }
        let res = adf_test(&data, 1);
        assert!(res.statistic < res.critical_5pct,
            "ADF stat={} should be < {} for stationary series", res.statistic, res.critical_5pct);
    }
}
