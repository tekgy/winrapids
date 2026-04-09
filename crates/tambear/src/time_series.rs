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

/// Fit AR(p) via Burg's method (Burg 1968).
///
/// Unlike Yule-Walker (which uses sample autocorrelations), Burg recursively
/// minimizes the sum of forward and backward prediction errors. This:
/// - Guarantees stability (all roots inside the unit circle)
/// - Produces less biased estimates for short series
/// - Does not require windowing (no implicit data extension)
///
/// The Burg algorithm is the Levinson-Durbin recursion applied to the
/// forward/backward reflection coefficient directly from data residuals,
/// avoiding the autocorrelation estimate.
///
/// Reference: J.P. Burg, "A New Analysis Technique for Time Series Data" (1968).
pub fn ar_burg_fit(data: &[f64], p: usize) -> ArResult {
    let n = data.len();
    if n < p + 2 || p == 0 {
        return ArResult {
            coefficients: vec![0.0; p],
            sigma2: if n > 0 {
                crate::descriptive::moments_ungrouped(data).variance(0)
            } else { f64::NAN },
            aic: f64::NAN,
        };
    }

    // Center the data
    let mean = crate::descriptive::moments_ungrouped(data).mean();
    let x: Vec<f64> = data.iter().map(|v| v - mean).collect();

    // f = forward prediction errors, b = backward prediction errors
    let mut f = x.clone();
    let mut b = x.clone();

    // Initial variance estimate: sample variance
    let mut sigma2 = x.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if sigma2 < 1e-300 {
        return ArResult {
            coefficients: vec![0.0; p],
            sigma2: 0.0,
            aic: f64::NAN,
        };
    }

    // Final AR coefficients (accumulated via Levinson recursion)
    let mut phi = vec![0.0; p];

    for k in 0..p {
        // Compute reflection coefficient from forward/backward errors at stage k.
        // Indices: f[k+1..n] and b[k..n-1] (both have length n - k - 1)
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in (k + 1)..n {
            let fi = f[i];
            let bi = b[i - 1];
            num += fi * bi;
            den += fi * fi + bi * bi;
        }
        if den < 1e-300 {
            // No more signal to fit; remaining coefficients stay zero.
            break;
        }
        let kappa = 2.0 * num / den;

        // Levinson update on phi
        let mut new_phi = vec![0.0; p];
        for j in 0..k { new_phi[j] = phi[j] - kappa * phi[k - 1 - j]; }
        new_phi[k] = kappa;
        for j in 0..=k { phi[j] = new_phi[j]; }

        // Update forward/backward error arrays for next stage
        let mut new_f = f.clone();
        let mut new_b = b.clone();
        for i in (k + 1)..n {
            new_f[i] = f[i] - kappa * b[i - 1];
            new_b[i] = b[i - 1] - kappa * f[i];
        }
        f = new_f;
        b = new_b;

        // Update variance
        sigma2 *= 1.0 - kappa * kappa;
        if sigma2 < 1e-300 { sigma2 = 1e-300; break; }
    }

    // Log-likelihood and AIC (Gaussian innovations)
    let nf = n as f64;
    let ll = -0.5 * nf * (2.0 * std::f64::consts::PI * sigma2).ln() - nf / 2.0;
    let aic = -2.0 * ll + 2.0 * (p + 1) as f64;

    ArResult { coefficients: phi, sigma2, aic }
}

/// Burg AR power spectral density evaluated at a given normalized frequency.
///
/// For AR(p) model with coefficients φ and innovation variance σ²:
/// PSD(f) = σ² / |1 - Σ φ_k exp(-i·2π·f·k)|²
///
/// `f`: normalized frequency in [0, 0.5] (0 = DC, 0.5 = Nyquist).
pub fn ar_psd_at(ar: &ArResult, f: f64) -> f64 {
    let omega = 2.0 * std::f64::consts::PI * f;
    let mut re = 1.0_f64;
    let mut im = 0.0_f64;
    for (k, &phi_k) in ar.coefficients.iter().enumerate() {
        let arg = omega * (k + 1) as f64;
        re -= phi_k * arg.cos();
        im += phi_k * arg.sin();
    }
    let denom_mag_sq = re * re + im * im;
    if denom_mag_sq < 1e-300 { return f64::INFINITY; }
    ar.sigma2 / denom_mag_sq
}

/// Evaluate Burg AR PSD over a uniform frequency grid [0, 0.5].
///
/// Returns (frequencies, psd_values) with `n_freqs` evenly-spaced points.
pub fn ar_psd(ar: &ArResult, n_freqs: usize) -> (Vec<f64>, Vec<f64>) {
    if n_freqs == 0 { return (vec![], vec![]); }
    let mut freqs = Vec::with_capacity(n_freqs);
    let mut psd = Vec::with_capacity(n_freqs);
    for i in 0..n_freqs {
        let f = 0.5 * i as f64 / (n_freqs - 1).max(1) as f64;
        freqs.push(f);
        psd.push(ar_psd_at(ar, f));
    }
    (freqs, psd)
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
// CUSUM and binary segmentation changepoint detection
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a CUSUM changepoint scan.
#[derive(Debug, Clone)]
pub struct CusumResult {
    /// Location of maximum |CUSUM| statistic (candidate changepoint index).
    pub argmax: usize,
    /// Maximum |CUSUM| value.
    pub max_abs_cusum: f64,
    /// CUSUM series (length n).
    pub cusum: Vec<f64>,
}

/// Classical CUSUM on the mean: S_k = Σ_{t=1}^{k} (x_t - x̄).
///
/// At a changepoint, the CUSUM deviates maximally from zero. `argmax` of |S|
/// is the most likely changepoint location.
pub fn cusum_mean(data: &[f64]) -> CusumResult {
    let n = data.len();
    if n == 0 {
        return CusumResult { argmax: 0, max_abs_cusum: 0.0, cusum: vec![] };
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut cusum = Vec::with_capacity(n);
    let mut running = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut argmax = 0;
    for (i, &x) in data.iter().enumerate() {
        running += x - mean;
        cusum.push(running);
        let a = running.abs();
        if a > max_abs { max_abs = a; argmax = i; }
    }
    CusumResult { argmax, max_abs_cusum: max_abs, cusum }
}

/// Binary segmentation changepoint detection via CUSUM.
///
/// Recursively splits the series at the maximum CUSUM point, accepting the
/// split if the max |CUSUM| exceeds `threshold`. Returns sorted changepoint
/// indices.
///
/// `data`: time series.
/// `threshold`: minimum max|CUSUM| to accept a split (e.g., 2·σ·√n).
/// `min_segment_size`: minimum size of any returned segment.
/// `max_changepoints`: cap on returned changepoints (to bound recursion).
pub fn cusum_binary_segmentation(
    data: &[f64],
    threshold: f64,
    min_segment_size: usize,
    max_changepoints: usize,
) -> Vec<usize> {
    let mut cps: Vec<usize> = Vec::new();
    if data.len() < 2 * min_segment_size || max_changepoints == 0 {
        return cps;
    }

    // Stack of (start, end) intervals to process
    let mut stack: Vec<(usize, usize)> = vec![(0, data.len())];
    while let Some((s, e)) = stack.pop() {
        if e - s < 2 * min_segment_size { continue; }
        if cps.len() >= max_changepoints { break; }
        let segment = &data[s..e];
        let result = cusum_mean(segment);
        if result.max_abs_cusum < threshold { continue; }
        // Candidate changepoint is at argmax + 1 (split after that index)
        let cp_rel = result.argmax + 1;
        if cp_rel < min_segment_size || (e - s - cp_rel) < min_segment_size {
            continue;
        }
        let cp_abs = s + cp_rel;
        cps.push(cp_abs);
        // Recurse on both halves
        stack.push((s, cp_abs));
        stack.push((cp_abs, e));
    }
    cps.sort_unstable();
    cps
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
        let (c1, c5, c10) = mackinnon_adf_critical_values(n);
        return AdfResult {
            statistic: f64::NAN,
            n_lags,
            critical_1pct: c1,
            critical_5pct: c5,
            critical_10pct: c10,
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
        let (c1, c5, c10) = mackinnon_adf_critical_values(nobs);
        return AdfResult {
            statistic: f64::NAN,
            n_lags,
            critical_1pct: c1,
            critical_5pct: c5,
            critical_10pct: c10,
        };
    }

    let a = crate::linear_algebra::Mat::from_vec(p, p, xtx);
    let l = match crate::linear_algebra::cholesky(&a) {
        Some(l) => l,
        None => {
            let (c1, c5, c10) = mackinnon_adf_critical_values(nobs);
            return AdfResult {
                statistic: f64::NAN,
                n_lags,
                critical_1pct: c1,
                critical_5pct: c5,
                critical_10pct: c10,
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

    // MacKinnon (2010) finite-sample critical values for "constant" model.
    // Response surface: cv(n) = β_∞ + β_1/n + β_2/n²
    // Coefficients from MacKinnon (2010) Table 1, case "c" (constant, no trend).
    let (c1, c5, c10) = mackinnon_adf_critical_values(nobs);
    AdfResult {
        statistic,
        n_lags,
        critical_1pct: c1,
        critical_5pct: c5,
        critical_10pct: c10,
    }
}

/// MacKinnon (2010) finite-sample critical values for ADF "constant" model.
///
/// Response surface: cv(n) = β_∞ + β_1/n + β_2/n²
/// Coefficients from MacKinnon (2010, Journal of Applied Econometrics) Table 1.
fn mackinnon_adf_critical_values(n: usize) -> (f64, f64, f64) {
    let nf = n as f64;
    let inv_n = 1.0 / nf;
    let inv_n2 = inv_n * inv_n;

    // Coefficients for case "c" (constant, no trend), 1 regressor:
    //   1%:  -3.4336  + (-5.999) /T + (-29.25)/T²
    //   5%:  -2.8621  + (-2.738) /T + (-8.36) /T²
    //  10%:  -2.5671  + (-1.438) /T + (-4.48) /T²
    let c1  = -3.4336 + (-5.999) * inv_n + (-29.25) * inv_n2;
    let c5  = -2.8621 + (-2.738) * inv_n + (-8.36) * inv_n2;
    let c10 = -2.5671 + (-1.438) * inv_n + (-4.48) * inv_n2;

    (c1, c5, c10)
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
// Ljung-Box portmanteau test (Ljung & Box 1978)
// ═══════════════════════════════════════════════════════════════════════════

/// Ljung-Box test result.
#[derive(Debug, Clone)]
pub struct LjungBoxResult {
    pub statistic: f64,
    pub p_value: f64,
    pub df: usize,
    pub n_lags: usize,
}

/// Ljung-Box portmanteau test for autocorrelation.
///
/// Q(h) = n(n+2) Σ_{k=1}^h ρ̂²_k / (n-k) ~ χ²(h - fitted_params) under H₀.
///
/// `fitted_params`: number of ARMA parameters (p+q). Use 0 for raw series.
pub fn ljung_box(data: &[f64], n_lags: usize, fitted_params: usize) -> LjungBoxResult {
    let n = data.len();
    let nf = n as f64;
    let rho = acf(data, n_lags);

    let q: f64 = (1..=n_lags.min(rho.len() - 1)).map(|k| {
        rho[k] * rho[k] / (nf - k as f64)
    }).sum::<f64>() * nf * (nf + 2.0);

    let df = n_lags.saturating_sub(fitted_params).max(1);
    let p_value = crate::special_functions::chi2_right_tail_p(q, df as f64);

    LjungBoxResult { statistic: q, p_value, df, n_lags }
}

// ═══════════════════════════════════════════════════════════════════════════
// Durbin-Watson test (Durbin & Watson 1950)
// ═══════════════════════════════════════════════════════════════════════════

/// Durbin-Watson test result.
#[derive(Debug, Clone)]
pub struct DurbinWatsonResult {
    /// d statistic. Range [0, 4]. d ≈ 2 → no autocorrelation.
    pub statistic: f64,
    /// Estimated AR(1) coefficient: ρ̂ ≈ 1 - d/2.
    pub rho_hat: f64,
}

/// Durbin-Watson statistic for serial autocorrelation in residuals.
///
/// d = Σ(ê_t - ê_{t-1})² / Σ ê_t². Interpretation:
/// - d ≈ 2: no autocorrelation
/// - d < 1.5: positive autocorrelation (rule of thumb)
/// - d > 2.5: negative autocorrelation (rule of thumb)
pub fn durbin_watson(residuals: &[f64]) -> DurbinWatsonResult {
    let n = residuals.len();
    if n < 2 {
        return DurbinWatsonResult { statistic: 2.0, rho_hat: 0.0 };
    }
    let num: f64 = (1..n).map(|t| (residuals[t] - residuals[t - 1]).powi(2)).sum();
    let den: f64 = residuals.iter().map(|e| e * e).sum();
    if den < 1e-300 {
        return DurbinWatsonResult { statistic: 2.0, rho_hat: 0.0 };
    }
    let d = num / den;
    let rho_hat = 1.0 - d / 2.0;
    DurbinWatsonResult { statistic: d, rho_hat }
}

// ═══════════════════════════════════════════════════════════════════════════
// KPSS test for stationarity (Kwiatkowski et al. 1992)
// ═══════════════════════════════════════════════════════════════════════════

/// KPSS test result.
#[derive(Debug, Clone)]
pub struct KpssResult {
    pub statistic: f64,
    pub critical_1pct: f64,
    pub critical_5pct: f64,
    pub critical_10pct: f64,
    pub n_lags: usize,
}

/// KPSS test for stationarity.
///
/// H₀: series is stationary. H₁: unit root (non-stationary).
/// Note: opposite of ADF! Use both together for robust assessment.
///
/// `trend`: false = level stationarity (constant only), true = trend stationarity.
/// `n_lags`: truncation lag for Newey-West. None = automatic (4·(T/100)^0.25).
pub fn kpss_test(data: &[f64], trend: bool, n_lags: Option<usize>) -> KpssResult {
    let n = data.len();
    let nf = n as f64;

    if n < 4 {
        return KpssResult {
            statistic: f64::NAN, critical_1pct: f64::NAN,
            critical_5pct: f64::NAN, critical_10pct: f64::NAN, n_lags: 0,
        };
    }

    // OLS residuals: demean (level) or detrend (trend)
    let residuals: Vec<f64> = if trend {
        // Regress y on (1, t): y = a + b·t + e
        let t_mean = (nf - 1.0) / 2.0;
        let y_mean = data.iter().sum::<f64>() / nf;
        let mut stt = 0.0;
        let mut sty = 0.0;
        for i in 0..n {
            let ti = i as f64 - t_mean;
            stt += ti * ti;
            sty += ti * (data[i] - y_mean);
        }
        let b = sty / stt;
        let a = y_mean - b * t_mean;
        (0..n).map(|i| data[i] - a - b * i as f64).collect()
    } else {
        let mean = data.iter().sum::<f64>() / nf;
        data.iter().map(|x| x - mean).collect()
    };

    // Partial sums
    let mut s = vec![0.0; n];
    s[0] = residuals[0];
    for t in 1..n { s[t] = s[t - 1] + residuals[t]; }

    // Long-run variance (Newey-West with Bartlett kernel)
    let lag = n_lags.unwrap_or((4.0 * (nf / 100.0).powf(0.25)).floor() as usize);
    let mut sigma2 = residuals.iter().map(|e| e * e).sum::<f64>() / nf;
    for j in 1..=lag.min(n - 1) {
        let w = 1.0 - j as f64 / (lag as f64 + 1.0);
        let cross: f64 = (j..n).map(|t| residuals[t] * residuals[t - j]).sum();
        sigma2 += 2.0 * w * cross / nf;
    }
    if sigma2 < 1e-300 {
        sigma2 = 1e-300;
    }

    // Test statistic
    let eta = s.iter().map(|si| si * si).sum::<f64>() / (nf * nf * sigma2);

    // Critical values (KPSS 1992 Table 1)
    let (cv10, cv5, cv1) = if trend {
        (0.119, 0.146, 0.216)
    } else {
        (0.347, 0.463, 0.739)
    };

    KpssResult { statistic: eta, critical_1pct: cv1, critical_5pct: cv5, critical_10pct: cv10, n_lags: lag }
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

    // ── Regression: finite-sample ADF critical values ──────────────────
    // Old code used fixed asymptotic values {-3.43, -2.86, -2.57}.
    // MacKinnon response surface gives more negative values for small n.
    #[test]
    fn adf_finite_sample_critical_values_regression() {
        // For small n, critical values should be more negative than asymptotic
        let small_data: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
        let res_small = adf_test(&small_data, 1);

        // Asymptotic values: -3.43, -2.86, -2.57
        // For n=30, finite-sample correction makes them more negative
        assert!(res_small.critical_1pct < -3.43,
            "1% CV for n=30 should be more negative than -3.43, got {}", res_small.critical_1pct);
        assert!(res_small.critical_5pct < -2.86,
            "5% CV for n=30 should be more negative than -2.86, got {}", res_small.critical_5pct);
        assert!(res_small.critical_10pct < -2.57,
            "10% CV for n=30 should be more negative than -2.57, got {}", res_small.critical_10pct);

        // For large n, should converge toward asymptotic values
        let large_data: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.01).sin()).collect();
        let res_large = adf_test(&large_data, 1);
        assert!((res_large.critical_1pct - (-3.4336)).abs() < 0.01,
            "1% CV for large n should be near -3.4336, got {}", res_large.critical_1pct);
        assert!((res_large.critical_5pct - (-2.8621)).abs() < 0.01,
            "5% CV for large n should be near -2.8621, got {}", res_large.critical_5pct);
    }

    // ── Ljung-Box test ────────────────────────────────────────────────────

    #[test]
    fn ljung_box_white_noise() {
        // Genuine iid noise from Xoshiro256 — no autocorrelation
        let mut rng = crate::rng::Xoshiro256::new(42);
        let wn: Vec<f64> = (0..200).map(|_| crate::rng::TamRng::next_f64(&mut rng) - 0.5).collect();
        let r = ljung_box(&wn, 10, 0);
        // With 200 observations and 10 lags of genuine white noise, Q ~ chi2(10).
        // Expected Q ≈ 10; p should be comfortably > 0.01 for seed=42.
        assert!(r.p_value > 0.01, "white noise: ljung-box p should be > 0.01, got {}", r.p_value);
        assert!(r.statistic >= 0.0);
    }

    #[test]
    fn ljung_box_autocorrelated() {
        // AR(1) with high rho — should reject H₀
        let mut data = vec![0.0f64; 100];
        for i in 1..100 { data[i] = 0.9 * data[i-1] + (i as f64 * 0.01).sin() * 0.1; }
        let r = ljung_box(&data, 10, 0);
        assert!(r.p_value < 0.001, "highly autocorrelated: ljung-box p={}", r.p_value);
    }

    // ── Durbin-Watson test ────────────────────────────────────────────────

    #[test]
    fn durbin_watson_no_autocorrelation() {
        // Residuals with no serial correlation — DW should be near 2.0.
        // Use iid noise from Xoshiro256 for genuine independence.
        let mut rng = crate::rng::Xoshiro256::new(99);
        let resid: Vec<f64> = (0..50).map(|_| crate::rng::TamRng::next_f64(&mut rng) - 0.5).collect();
        let r = durbin_watson(&resid);
        assert!((r.statistic - 2.0).abs() < 1.0, "iid noise: DW should be near 2.0, got {}", r.statistic);
    }

    #[test]
    fn durbin_watson_positive_autocorrelation() {
        // Monotone residuals (perfect positive autocorrelation) → DW near 0
        let resid: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let r = durbin_watson(&resid);
        assert!(r.statistic < 0.5, "monotone residuals: DW should be near 0, got {}", r.statistic);
    }

    #[test]
    fn durbin_watson_negative_autocorrelation() {
        // Alternating residuals → DW near 4.0
        let resid: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let r = durbin_watson(&resid);
        assert!(r.statistic > 3.5, "alternating residuals: DW should be near 4.0, got {}", r.statistic);
    }

    // ── KPSS test ─────────────────────────────────────────────────────────

    #[test]
    fn kpss_stationary_series() {
        // Stationary: sin wave with no trend — should fail to reject (p > 0.05)
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let r = kpss_test(&data, false, None);
        // KPSS H₀ = stationary; high p → fail to reject
        // Statistic below 10% critical value (0.347) means stationary
        assert!(r.statistic < r.critical_10pct,
            "stationary series: KPSS stat={:.3} should be < 10% cv={:.3}",
            r.statistic, r.critical_10pct);
    }

    #[test]
    fn kpss_nonstationary_random_walk() {
        // Random walk with large iid steps — clearly non-stationary
        let mut rng = crate::rng::Xoshiro256::new(7);
        let mut data = vec![0.0f64; 200];
        for i in 1..200 {
            // Steps from N(0,1) approximated as sum of 12 uniform - 6
            let step: f64 = (0..12).map(|_| crate::rng::TamRng::next_f64(&mut rng)).sum::<f64>() - 6.0;
            data[i] = data[i-1] + step;
        }
        let r = kpss_test(&data, false, None);
        // A random walk with 200 steps should almost always exceed 1% critical value
        assert!(r.statistic > r.critical_5pct,
            "random walk: KPSS stat={:.3} should exceed 5% cv={:.3}",
            r.statistic, r.critical_5pct);
    }
}
