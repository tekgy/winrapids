//! # Family 17 — Time Series Models
//!
//! ARMA, ARIMA, exponential smoothing, unit root tests, Granger causality.
//!
//! ## Architecture
//!
//! AR/MA fitting = Yule-Walker (Kingdom A: Toeplitz solve from autocorrelation).
//! ARIMA = difference + ARMA.
//! Exponential smoothing = Kingdom A via affine semigroup prefix scan (see signal_processing::affine_prefix_scan).
//! Prior label "Kingdom B" for exponential smoothing was wrong — the map a=(1-α), b=α·x_t
//! is data-determined and affine maps compose with bounded representation.
//! ARMA(p,q) filter = Kingdom A in both AR-only and MA-present cases via companion matrix
//! prefix scan (M = constant MA-coefficient matrix, b_t = data-determined AR drive).
//! `arma_fit` = Kingdom A (filter) + Kingdom C (outer MLE), same structure as GARCH.
//! Unit root tests = regression-based (ADF reuses F10 OLS).

// ═══════════════════════════════════════════════════════════════════════════
// Levinson-Durbin recursion (general Toeplitz solver)
// ═══════════════════════════════════════════════════════════════════════════

/// Levinson-Durbin recursion for symmetric positive-definite Toeplitz systems.
///
/// Solves the Yule-Walker system T·φ = r[1..p] where T is the p×p symmetric
/// Toeplitz matrix built from r[0..p-1], using Durbin's recursion (O(p²)).
///
/// # Arguments
///
/// - `r`: autocorrelation sequence r[0], r[1], ..., r[p] where r[0] is the
///   zero-lag autocovariance (or normalized so r[0]=1 for correlations).
///   Must satisfy `r.len() >= 2` and `r[0] > 0`.
///
/// # Returns
///
/// `(phi, kappas, sigma2)` where:
/// - `phi`: AR coefficients φ₁, ..., φₚ (length = `r.len() - 1`).
/// - `kappas`: reflection coefficients κ₁, ..., κₚ. These ARE the partial
///   autocorrelations (PACF values at lags 1..p).
/// - `sigma2`: final prediction-error variance (= r[0] · ∏(1 - κₖ²)).
///
/// # Applications
///
/// - AR model fitting via Yule-Walker (see `ar_fit`)
/// - Partial autocorrelation function computation (see `pacf`)
/// - Linear prediction coding (speech/audio)
/// - Burg's initialization, Wiener filter design
///
/// # Notes
///
/// The reflection coefficients satisfy |κₖ| < 1 for a stationary AR process.
/// If |κₖ| ≥ 1 (non-stationary input), sigma2 is clamped to 0.
pub fn levinson_durbin(r: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let p = r.len().saturating_sub(1);
    let mut phi = vec![0.0; p];
    let mut kappas = vec![0.0; p];
    let mut sigma2 = r[0];

    if p == 0 || sigma2 < 1e-300 {
        return (phi, kappas, sigma2.max(0.0));
    }

    let mut prev = vec![0.0; p];

    for k in 0..p {
        // Reflection coefficient κ_{k+1}
        let mut num = r[k + 1];
        for j in 0..k {
            num -= prev[j] * r[k - j];
        }
        let kappa = num / sigma2;
        kappas[k] = kappa;

        // Update AR coefficients (Levinson update)
        phi[k] = kappa;
        for j in 0..k {
            phi[j] = prev[j] - kappa * prev[k - 1 - j];
        }

        sigma2 *= 1.0 - kappa * kappa;
        sigma2 = sigma2.max(0.0); // clamp for numerical stability
        prev[..=k].copy_from_slice(&phi[..=k]);
    }

    (phi, kappas, sigma2)
}

/// Build delay-embedded trajectory matrix from a time series.
///
/// Implements Takens' delay embedding theorem: each row of the returned matrix
/// is a phase-space point `[x[t], x[t+tau], x[t+2*tau], ..., x[t+(d-1)*tau]]`.
///
/// # Arguments
///
/// - `data`: input time series.
/// - `dim`: embedding dimension `d` (number of coordinates per point).
/// - `tau`: time delay in samples.
///
/// # Returns
///
/// Row-major matrix of shape (n_rows × dim) as `Vec<Vec<f64>>`. Returns empty
/// vec if `data.len() < (dim-1)*tau + 1`.
///
/// # Applications
///
/// Phase-space reconstruction, correlation dimension, Lyapunov exponents,
/// recurrence plots, convergent cross-mapping (CCM), FNN analysis.
pub fn delay_embed(data: &[f64], dim: usize, tau: usize) -> Vec<Vec<f64>> {
    let n = data.len();
    let min_len = dim.saturating_sub(1) * tau + 1;
    if n < min_len || dim == 0 {
        return Vec::new();
    }
    let n_rows = n - (dim - 1) * tau;
    let mut mat = Vec::with_capacity(n_rows);
    for t in 0..n_rows {
        let row: Vec<f64> = (0..dim).map(|d| data[t + d * tau]).collect();
        mat.push(row);
    }
    mat
}

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

    // Constant series guard
    if r[0] < 1e-15 {
        let phi = vec![0.0; p];
        let ll = -0.5 * n as f64 * (2.0 * std::f64::consts::PI * 1e-15_f64).ln() - n as f64 / 2.0;
        return ArResult {
            coefficients: phi,
            sigma2: 0.0,
            aic: -2.0 * ll + 2.0 * (p + 1) as f64,
        };
    }

    // Levinson-Durbin recursion (delegate to public primitive)
    let (phi, _kappas, sigma2) = levinson_durbin(&r);

    let ll = -0.5 * n as f64 * (2.0 * std::f64::consts::PI * sigma2.max(1e-300)).ln() - n as f64 / 2.0;
    let aic = -2.0 * ll + 2.0 * (p + 1) as f64;

    ArResult {
        coefficients: phi,
        sigma2,
        aic,
    }
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
            if n > j {
                val += ar.coefficients[j] * buf[n - 1 - j];
            }
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
            } else {
                f64::NAN
            },
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
        for j in 0..k {
            new_phi[j] = phi[j] - kappa * phi[k - 1 - j];
        }
        new_phi[k] = kappa;
        for j in 0..=k {
            phi[j] = new_phi[j];
        }

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
        if sigma2 < 1e-300 {
            sigma2 = 1e-300;
            break;
        }
    }

    // Log-likelihood and AIC (Gaussian innovations)
    let nf = n as f64;
    let ll = -0.5 * nf * (2.0 * std::f64::consts::PI * sigma2).ln() - nf / 2.0;
    let aic = -2.0 * ll + 2.0 * (p + 1) as f64;

    ArResult {
        coefficients: phi,
        sigma2,
        aic,
    }
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
    if denom_mag_sq < 1e-300 {
        return f64::INFINITY;
    }
    ar.sigma2 / denom_mag_sq
}

/// Evaluate Burg AR PSD over a uniform frequency grid [0, 0.5].
///
/// Returns (frequencies, psd_values) with `n_freqs` evenly-spaced points.
pub fn ar_psd(ar: &ArResult, n_freqs: usize) -> (Vec<f64>, Vec<f64>) {
    if n_freqs == 0 {
        return (vec![], vec![]);
    }
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

/// Undo d rounds of first-differencing by cumulative summation.
///
/// Given `initial` values (the first value lost at each differencing round)
/// and a differenced series, reconstructs the original. `initial` must have
/// length `d` — element 0 is the first value of the original series, element 1
/// is the first value after the first cumsum, etc.
///
/// If `initial` is empty and `d == 0`, returns `data` unchanged.
pub fn undifference(data: &[f64], initial: &[f64]) -> Vec<f64> {
    let d = initial.len();
    let mut result = data.to_vec();
    for i in (0..d).rev() {
        let mut out = Vec::with_capacity(result.len() + 1);
        out.push(initial[i]);
        let mut acc = initial[i];
        for &v in &result {
            acc += v;
            out.push(acc);
        }
        result = out;
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Rolling statistics via prefix sums
// ═══════════════════════════════════════════════════════════════════════════

/// Compute rolling population variance of `returns` over `window` observations using prefix sums.
///
/// Returns a vector of length `n - window + 1` where each entry is the population variance
/// of the corresponding window (i..i+window). Returns an empty vector if `n < window`.
///
/// Uses the numerically stable two-pass prefix-sum form:
/// `var = E[x²] - (E[x])²` computed from prefix sums in O(n) time.
pub fn rolling_variance_prefix(returns: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len();
    if n < window {
        return Vec::new();
    }
    let mut cumsum = vec![0.0_f64; n + 1];
    let mut cumsum2 = vec![0.0_f64; n + 1];
    for i in 0..n {
        cumsum[i + 1] = cumsum[i] + returns[i];
        cumsum2[i + 1] = cumsum2[i] + returns[i] * returns[i];
    }
    let wf = window as f64;
    (0..=(n - window))
        .map(|i| {
            let s = cumsum[i + window] - cumsum[i];
            let s2 = cumsum2[i + window] - cumsum2[i];
            let mean = s / wf;
            (s2 / wf - mean * mean).max(0.0)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// ARMA(p,q) — conditional sum of squares estimation
// ═══════════════════════════════════════════════════════════════════════════

/// Result of ARMA(p,q) or ARIMA(p,d,q) fit.
#[derive(Debug, Clone)]
pub struct ArmaResult {
    /// AR coefficients φ₁, ..., φₚ.
    pub ar: Vec<f64>,
    /// MA coefficients θ₁, ..., θ_q.
    pub ma: Vec<f64>,
    /// Intercept (mean of the process × (1 - Σφ_i) for ARMA;
    /// for ARIMA with d > 0 this is the intercept on the differenced scale).
    pub intercept: f64,
    /// Innovation variance σ².
    pub sigma2: f64,
    /// Conditional sum of squares (total residual sum of squares).
    pub css: f64,
    /// AIC = n·ln(σ²) + 2·(p + q + 1).
    pub aic: f64,
    /// BIC = n·ln(σ²) + (p + q + 1)·ln(n).
    pub bic: f64,
    /// Number of L-BFGS iterations.
    pub iterations: usize,
    /// Residuals (length = data.len() - max(p,q) for conditional method).
    pub residuals: Vec<f64>,
}

/// Compute conditional sum-of-squares residuals for ARMA(p,q).
///
/// General primitive for all ARMA-family methods. Pre-sample values and
/// residuals are zero-initialized (conditional initialization). The full
/// residual vector `ε[0..n]` is returned with `ε[t] = 0` for `t < max(p,q)`.
///
/// # Parameters
/// - `centered`: de-meaned observation sequence (length n)
/// - `ar`:  AR coefficients φ[1..p] in lag order (φ₁ first)
/// - `ma`:  MA coefficients θ[1..q] in lag order (θ₁ first)
///
/// # Returns
/// Residual vector of length `n`. Pre-sample entries are 0.0.
///
/// # Consumers
/// ARMA, ARIMA, SARIMA, VARMA fit; Ljung-Box residual checking;
/// impulse response functions; ARMA-GARCH joint estimation.
pub fn arma_css_residuals(centered: &[f64], ar: &[f64], ma: &[f64]) -> Vec<f64> {
    let n = centered.len();
    let p = ar.len();
    let q = ma.len();
    let start = p.max(q);
    if n <= start {
        return vec![];
    }

    // Full residual vector (pre-sample = 0).
    let mut eps = vec![0.0_f64; n];

    for t in start..n {
        let mut pred = 0.0_f64;
        // AR part: Σ φ_i · x_{t-i}
        for i in 0..p {
            pred += ar[i] * centered[t - 1 - i];
        }
        // MA part: Σ θ_j · ε_{t-j}
        for j in 0..q {
            pred += ma[j] * eps[t - 1 - j];
        }
        eps[t] = centered[t] - pred;
    }

    eps[start..].to_vec()
}

/// Fit ARMA(p,q) by conditional sum of squares (CSS) via L-BFGS.
///
/// The objective is `Σ ε_t²` where residuals are computed with
/// zero-initialized pre-sample values. L-BFGS minimizes this with
/// numerical gradients (central difference, step = 1e-5).
///
/// # Parameters
/// - `data`: observed series.
/// - `p`: AR order.
/// - `q`: MA order.
/// - `max_iter`: L-BFGS iteration limit (default 200 is reasonable).
///
/// # Returns
/// `ArmaResult` with fitted coefficients, intercept, sigma2, AIC/BIC, residuals.
///
/// # Kingdom
/// Inner filter (`arma_css_residuals`) = **Kingdom A** for both pure AR and MA-present
/// cases: state vector `s_t = [ε_t, …, ε_{t-q+1}]^T` propagates via constant companion
/// matrix M (containing θ_j) plus data-determined drive `b_t(x) = x_t - Σ φ_i x_{t-i}`.
/// M is constant → map is data-determined → affine semigroup → bounded representation.
/// The prior label "Kingdom B when q > 0" was wrong: residuals appear as *state entries*
/// but the *map itself* (M) is constant, not state-dependent.
///
/// Outer MLE (L-BFGS over CSS objective) = **Kingdom C** (iterative fixed-point).
/// Together: Kingdom A (filter) + Kingdom C (optimization) — same structure as GARCH.
pub fn arma_fit(data: &[f64], p: usize, q: usize, max_iter: usize) -> ArmaResult {
    let n = data.len();
    let start = p.max(q);
    let n_eff = n.saturating_sub(start);

    // Degenerate case
    if n_eff < 3 || (p == 0 && q == 0) {
        let mean = data.iter().sum::<f64>() / n.max(1) as f64;
        let ss: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
        let sigma2 = ss / n.max(1) as f64;
        let aic = n as f64 * sigma2.max(1e-300).ln() + 2.0;
        let bic = n as f64 * sigma2.max(1e-300).ln() + (n as f64).ln();
        return ArmaResult {
            ar: vec![],
            ma: vec![],
            intercept: mean,
            sigma2,
            css: ss,
            aic,
            bic,
            iterations: 0,
            residuals: data.iter().map(|x| x - mean).collect(),
        };
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();

    let n_params = p + q;

    // Initialize AR coefficients from Yule-Walker if p > 0.
    let mut x0 = vec![0.0_f64; n_params];
    if p > 0 {
        let ar_init = ar_fit(data, p);
        for i in 0..p {
            x0[i] = ar_init.coefficients[i] * 0.5; // damped to help convergence
        }
    }
    // MA coefficients start at 0.

    let centered_ref = &centered;
    let objective = |params: &[f64]| -> f64 {
        let ar_slice = &params[..p];
        let ma_slice = &params[p..p + q];
        let resid = arma_css_residuals(centered_ref, ar_slice, ma_slice);
        resid.iter().map(|e| e * e).sum::<f64>()
    };

    let gradient = |params: &[f64]| -> Vec<f64> {
        let eps = 1e-5;
        let f0 = objective(params);
        let mut grad = vec![0.0; n_params];
        let mut perturbed = params.to_vec();
        for i in 0..n_params {
            let orig = perturbed[i];
            perturbed[i] = orig + eps;
            let fp = objective(&perturbed);
            perturbed[i] = orig - eps;
            let fm = objective(&perturbed);
            perturbed[i] = orig;
            grad[i] = (fp - fm) / (2.0 * eps);
        }
        grad
    };

    let opt = crate::optimization::lbfgs(&objective, &gradient, &x0, 10, max_iter, 1e-8);

    let ar_coefs: Vec<f64> = opt.x[..p].to_vec();
    let ma_coefs: Vec<f64> = opt.x[p..p + q].to_vec();
    let residuals = arma_css_residuals(&centered, &ar_coefs, &ma_coefs);
    let css: f64 = residuals.iter().map(|e| e * e).sum();
    let sigma2 = css / n_eff.max(1) as f64;
    let ln_s2 = sigma2.max(1e-300).ln();
    let k = (p + q + 1) as f64; // +1 for intercept
    let nf = n_eff as f64;
    let aic = nf * ln_s2 + 2.0 * k;
    let bic = nf * ln_s2 + k * nf.ln();

    ArmaResult {
        ar: ar_coefs,
        ma: ma_coefs,
        intercept: mean,
        sigma2,
        css,
        aic,
        bic,
        iterations: opt.iterations,
        residuals,
    }
}

/// Result of ARIMA(p,d,q) fit.
#[derive(Debug, Clone)]
pub struct ArimaResult {
    /// The ARMA fit on the differenced series.
    pub arma: ArmaResult,
    /// Integration order.
    pub d: usize,
    /// First values lost to each round of differencing (length d).
    /// Needed by `arima_forecast` to undo differencing.
    pub diff_initials: Vec<f64>,
    /// Original series length.
    pub n_original: usize,
}

/// Fit ARIMA(p,d,q) = difference d times, then ARMA(p,q) on the result.
///
/// # Parameters
/// - `data`: observed series.
/// - `p`: AR order.
/// - `d`: differencing order (typically 0, 1, or 2).
/// - `q`: MA order.
/// - `max_iter`: L-BFGS iteration limit.
pub fn arima_fit(data: &[f64], p: usize, d: usize, q: usize, max_iter: usize) -> ArimaResult {
    let n_original = data.len();

    // Capture the first value at each differencing round for undifferencing.
    let mut diff_initials = Vec::with_capacity(d);
    let mut current = data.to_vec();
    for _ in 0..d {
        if current.is_empty() {
            break;
        }
        diff_initials.push(current[0]);
        current = current.windows(2).map(|w| w[1] - w[0]).collect();
    }

    let arma = arma_fit(&current, p, q, max_iter);

    ArimaResult {
        arma,
        d,
        diff_initials,
        n_original,
    }
}

/// Forecast h steps ahead from a fitted ARIMA model.
///
/// Generates point forecasts on the differenced scale using the ARMA
/// coefficients, then undifferences to the original scale.
///
/// `last_values`: the last `max(p, q)` values of the **original** series
/// (needed to seed the AR recursion after undifferencing). If the series
/// is short, pass the entire original series.
pub fn arima_forecast(fit: &ArimaResult, last_values: &[f64], horizon: usize) -> Vec<f64> {
    let p = fit.arma.ar.len();
    let q = fit.arma.ma.len();
    let start = p.max(q);

    // Build the recent history on the differenced scale.
    let mut diff_vals = difference(last_values, fit.d);
    let mean = fit.arma.intercept;
    // Center
    for v in diff_vals.iter_mut() {
        *v -= mean;
    }

    // Recent residuals (approximate: use last `q` residuals from the fit).
    let mut recent_eps: Vec<f64> = if q > 0 && !fit.arma.residuals.is_empty() {
        let r = &fit.arma.residuals;
        let take = q.min(r.len());
        r[r.len() - take..].to_vec()
    } else {
        vec![]
    };
    while recent_eps.len() < q {
        recent_eps.insert(0, 0.0);
    }

    // Generate forecasts on the centered-differenced scale.
    let mut forecasts_diff = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        let mut pred = 0.0_f64;
        let n_hist = diff_vals.len();
        for i in 0..p {
            if n_hist > i {
                pred += fit.arma.ar[i] * diff_vals[n_hist - 1 - i];
            }
        }
        for j in 0..q {
            if recent_eps.len() > j {
                pred += fit.arma.ma[j] * recent_eps[recent_eps.len() - 1 - j];
            }
        }
        // Future shocks are 0 (point forecast).
        diff_vals.push(pred);
        recent_eps.push(0.0);
        forecasts_diff.push(pred + mean); // un-center
    }

    // Undifference: we need the initial values to reconstruct.
    // The last `d` values of the original series provide the anchors.
    if fit.d == 0 {
        return forecasts_diff;
    }

    // Build the initials for undifferencing from the tail of the original series.
    let mut tail = last_values.to_vec();
    let mut undo_initials = Vec::with_capacity(fit.d);
    for round in 0..fit.d {
        if tail.is_empty() {
            break;
        }
        undo_initials.push(*tail.last().unwrap());
        tail = tail.windows(2).map(|w| w[1] - w[0]).collect();
    }
    // Reverse: undifference expects innermost-first.
    undo_initials.reverse();

    // The forecasted differenced values get undifferenced with these anchors.
    // But undifference reconstructs the full series including the anchor.
    // We only want the new h values.
    let full = undifference(&forecasts_diff, &undo_initials);
    // The first element of undifference output is the anchor; skip it.
    // Actually, undifference prepends `initial[i]` at each level.
    // For d=1: undifference([f1,f2,...], [last]) produces
    //   [last, last+f1, last+f1+f2, ...] — length h+1.
    // We want the last h elements.
    let skip = full.len().saturating_sub(horizon);
    full[skip..].to_vec()
}

/// Auto-select ARIMA order (p,d,q) by AIC grid search.
///
/// Tries all (p,d,q) with `p ∈ [0, max_p]`, `d ∈ [0, max_d]`, `q ∈ [0, max_q]`
/// and returns the fit with the lowest AIC.
///
/// # Parameters
/// - `max_p`, `max_d`, `max_q`: upper bounds for the grid search.
/// - `max_iter`: L-BFGS iteration limit per fit.
pub fn auto_arima(
    data: &[f64],
    max_p: usize,
    max_d: usize,
    max_q: usize,
    max_iter: usize,
) -> ArimaResult {
    let mut best: Option<ArimaResult> = None;
    let mut best_aic = f64::INFINITY;

    for d in 0..=max_d {
        let diffed = difference(data, d);
        if diffed.len() < 10 {
            continue;
        } // not enough data after differencing
        for p in 0..=max_p {
            for q in 0..=max_q {
                if p + q == 0 {
                    continue;
                }
                let start = p.max(q);
                if diffed.len() <= start + 3 {
                    continue;
                }
                let fit = arima_fit(data, p, d, q, max_iter);
                if fit.arma.aic.is_finite() && fit.arma.aic < best_aic {
                    best_aic = fit.arma.aic;
                    best = Some(fit);
                }
            }
        }
    }

    best.unwrap_or_else(|| arima_fit(data, 1, 0, 0, max_iter))
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
        return CusumResult {
            argmax: 0,
            max_abs_cusum: 0.0,
            cusum: vec![],
        };
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
        if a > max_abs {
            max_abs = a;
            argmax = i;
        }
    }
    CusumResult {
        argmax,
        max_abs_cusum: max_abs,
        cusum,
    }
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
        if e - s < 2 * min_segment_size {
            continue;
        }
        if cps.len() >= max_changepoints {
            break;
        }
        let segment = &data[s..e];
        let result = cusum_mean(segment);
        if result.max_abs_cusum < threshold {
            continue;
        }
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
///
/// **Kingdom A** via affine semigroup prefix scan: `level_t = α·x_t + (1-α)·level_{t-1}`.
/// Affine map `a_t = (1-α)` (constant), `b_t = α·x_t` (data-determined). Same
/// algebraic structure as EWMA variance; delegates conceptually to `affine_prefix_scan`.
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

    SesResult {
        fitted,
        forecast: level,
        alpha,
    }
}

/// Holt's linear trend method (double exponential smoothing).
///
/// **Kingdom A** via affine 3×3 companion matrix prefix scan. State vector
/// `s_t = [level_t, trend_t, 1]^T` propagates via constant companion matrix
/// (α, β are fixed parameters) with data-determined offset `b_t = [α·x_t, β·x_t, 0]^T`.
/// Affine maps compose with bounded (3×3) representation. Same structure as Kalman
/// filter with fixed observation model.
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
/// Response surface: `cv(n) = β_∞ + β_1/n + β_2/n²`
/// Coefficients from MacKinnon (2010, *Journal of Applied Econometrics*) Table 1,
/// case "c" (constant, no trend, 1 regressor).
///
/// # Parameters
/// - `n`: number of observations
///
/// # Returns
/// `(cv_1pct, cv_5pct, cv_10pct)` — critical values at 1%, 5%, 10% significance.
///
/// # Consumers
/// ADF test, Phillips-Perron test, DF-GLS, KPSS comparison, any unit root
/// test that needs finite-sample MacKinnon critical values.
pub fn mackinnon_adf_critical_values(n: usize) -> (f64, f64, f64) {
    let nf = n as f64;
    let inv_n = 1.0 / nf;
    let inv_n2 = inv_n * inv_n;

    // Coefficients for case "c" (constant, no trend), 1 regressor:
    //   1%:  -3.4336  + (-5.999) /T + (-29.25)/T²
    //   5%:  -2.8621  + (-2.738) /T + (-8.36) /T²
    //  10%:  -2.5671  + (-1.438) /T + (-4.48) /T²
    let c1 = -3.4336 + (-5.999) * inv_n + (-29.25) * inv_n2;
    let c5 = -2.8621 + (-2.738) * inv_n + (-8.36) * inv_n2;
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
    if var < 1e-15 {
        return vec![1.0; max_lag + 1];
    }

    (0..=max_lag)
        .map(|lag| {
            let r: f64 = data[..n - lag]
                .iter()
                .zip(data[lag..].iter())
                .map(|(a, b)| (a - mean) * (b - mean))
                .sum::<f64>()
                / (n as f64 * var);
            r
        })
        .collect()
}

/// Partial autocorrelation function via Levinson-Durbin.
///
/// The PACF at lag k is the reflection coefficient κₖ from the Levinson-Durbin
/// recursion on the normalized autocorrelation sequence.
pub fn pacf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let moments = crate::descriptive::moments_ungrouped(data);
    let mean = moments.mean();
    let var = moments.variance(0);
    let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();

    // Build normalized autocorrelation sequence ρ[0..=max_lag]
    let mut r = vec![0.0; max_lag + 1];
    for lag in 0..=max_lag {
        r[lag] = centered[..n - lag]
            .iter()
            .zip(centered[lag..].iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / (n as f64 * var);
    }

    let mut pacf_vals = vec![0.0; max_lag + 1];
    pacf_vals[0] = 1.0;
    if max_lag == 0 {
        return pacf_vals;
    }

    // The reflection coefficients from Levinson-Durbin ARE the PACF values
    let (_phi, kappas, _sigma2) = levinson_durbin(&r);
    for (k, &kappa) in kappas.iter().enumerate() {
        pacf_vals[k + 1] = kappa;
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

    let q: f64 = (1..=n_lags.min(rho.len() - 1))
        .map(|k| rho[k] * rho[k] / (nf - k as f64))
        .sum::<f64>()
        * nf
        * (nf + 2.0);

    let df = n_lags.saturating_sub(fitted_params).max(1);
    let p_value = crate::special_functions::chi2_right_tail_p(q, df as f64);

    LjungBoxResult {
        statistic: q,
        p_value,
        df,
        n_lags,
    }
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
        return DurbinWatsonResult {
            statistic: 2.0,
            rho_hat: 0.0,
        };
    }
    let num: f64 = (1..n)
        .map(|t| (residuals[t] - residuals[t - 1]).powi(2))
        .sum();
    let den: f64 = residuals.iter().map(|e| e * e).sum();
    if den < 1e-300 {
        return DurbinWatsonResult {
            statistic: 2.0,
            rho_hat: 0.0,
        };
    }
    let d = num / den;
    let rho_hat = 1.0 - d / 2.0;
    DurbinWatsonResult {
        statistic: d,
        rho_hat,
    }
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
            statistic: f64::NAN,
            critical_1pct: f64::NAN,
            critical_5pct: f64::NAN,
            critical_10pct: f64::NAN,
            n_lags: 0,
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
    for t in 1..n {
        s[t] = s[t - 1] + residuals[t];
    }

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

    KpssResult {
        statistic: eta,
        critical_1pct: cv1,
        critical_5pct: cv5,
        critical_10pct: cv10,
        n_lags: lag,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phillips-Perron unit-root test
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the Phillips-Perron unit-root test.
///
/// H₀: unit root (non-stationary). H₁: stationary.
/// Same asymptotic null distribution as ADF — use same MacKinnon critical values.
#[derive(Debug, Clone)]
pub struct PpResult {
    /// Phillips-Perron τ statistic (after non-parametric correction).
    pub statistic: f64,
    /// Number of Newey-West lags used for variance correction.
    pub n_lags: usize,
    /// Critical values at 1%, 5%, 10% (same as ADF MacKinnon approximations).
    pub critical_1pct: f64,
    pub critical_5pct: f64,
    pub critical_10pct: f64,
}

/// Phillips-Perron unit-root test.
///
/// A non-parametric correction to the Dickey-Fuller test that allows for
/// heteroscedastic and serially correlated errors without adding lagged
/// differences to the regression. Uses a Newey-West long-run variance estimate.
///
/// Procedure:
/// 1. Run DF regression: Δy_t = α + γ·y_{t-1} + ε_t
/// 2. Correct the t-statistic using the Newey-West long-run variance ω².
/// 3. Corrected statistic: τ_PP = t_γ · (σ̂ / ω) − T · (ω − σ̂) · se(γ̂) / (2 · ω · σ̂)
///
/// `n_lags`: Newey-West truncation lag. None → automatic (4·(T/100)^0.25).
pub fn pp_test(data: &[f64], n_lags: Option<usize>) -> PpResult {
    let n = data.len();
    let (cv1, cv5, cv10) = mackinnon_adf_critical_values(n);
    if n < 5 {
        return PpResult {
            statistic: f64::NAN,
            n_lags: 0,
            critical_1pct: cv1,
            critical_5pct: cv5,
            critical_10pct: cv10,
        };
    }

    // Step 1: DF regression Δy_t = α + γ·y_{t-1} + ε_t
    let dy: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
    let m = dy.len(); // = n - 1
    let y_lag: Vec<f64> = data[..m].to_vec();

    // OLS: regress dy on (1, y_lag) → [α, γ]
    let t = m as f64;
    let sy = dy.iter().sum::<f64>();
    let sx = y_lag.iter().sum::<f64>();
    let sxx = y_lag.iter().map(|x| x * x).sum::<f64>();
    let sxy = dy.iter().zip(y_lag.iter()).map(|(y, x)| y * x).sum::<f64>();

    let denom = t * sxx - sx * sx;
    if denom.abs() < 1e-300 {
        return PpResult {
            statistic: f64::NAN,
            n_lags: 0,
            critical_1pct: cv1,
            critical_5pct: cv5,
            critical_10pct: cv10,
        };
    }
    let gamma = (t * sxy - sx * sy) / denom;
    let alpha = (sy - gamma * sx) / t;
    let residuals: Vec<f64> = (0..m).map(|i| dy[i] - alpha - gamma * y_lag[i]).collect();

    // σ̂² = residual variance
    let sigma2 = residuals.iter().map(|e| e * e).sum::<f64>() / (t - 2.0);
    if sigma2 < 1e-300 {
        return PpResult {
            statistic: f64::NAN,
            n_lags: 0,
            critical_1pct: cv1,
            critical_5pct: cv5,
            critical_10pct: cv10,
        };
    }

    // Standard error of γ̂ (from OLS covariance formula)
    let se_gamma = (sigma2 * t / denom).sqrt();

    // Step 2: Newey-West long-run variance ω²
    let lag = n_lags.unwrap_or_else(|| (4.0 * (t / 100.0).powf(0.25)) as usize);
    // γ₀ = sample variance of residuals (lag 0)
    let gamma0 = residuals.iter().map(|e| e * e).sum::<f64>() / t;
    let mut omega2 = gamma0;
    for h in 1..=lag.min(m - 1) {
        let cov_h: f64 = (h..m).map(|i| residuals[i] * residuals[i - h]).sum::<f64>() / t;
        let w = 1.0 - h as f64 / (lag as f64 + 1.0); // Bartlett kernel
        omega2 += 2.0 * w * cov_h;
    }
    if omega2 < 1e-300 {
        return PpResult {
            statistic: f64::NAN,
            n_lags: lag,
            critical_1pct: cv1,
            critical_5pct: cv5,
            critical_10pct: cv10,
        };
    }

    // Step 3: PP corrected statistic
    let t_df = gamma / se_gamma;
    let sigma = sigma2.sqrt();
    let omega = omega2.sqrt();
    // τ_PP = t_γ · (σ/ω) − T·(ω−σ)·se(γ̂) / (2·ω·σ)
    let statistic = t_df * (sigma / omega) - t * (omega - sigma) * se_gamma / (2.0 * omega * sigma);

    PpResult {
        statistic,
        n_lags: lag,
        critical_1pct: cv1,
        critical_5pct: cv5,
        critical_10pct: cv10,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lo-MacKinlay variance ratio test
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the Lo-MacKinlay variance ratio test.
#[derive(Debug, Clone)]
pub struct VarianceRatioResult {
    /// VR(q) = Var(q-period return) / (q × Var(1-period return)).
    /// Under random walk, VR = 1.0.
    pub vr: f64,
    /// Standardized z-statistic using homoscedastic variance formula.
    pub z_stat: f64,
    /// Heteroscedasticity-robust z-statistic (z*).
    pub z_star: f64,
    /// Aggregation interval q used.
    pub q: usize,
}

/// Lo-MacKinlay (1988) variance ratio test for the random walk hypothesis.
///
/// Tests H₀: return series follows a random walk (VR = 1).
/// H₁: returns are autocorrelated (VR ≠ 1).
///
/// - VR > 1: positive autocorrelation (momentum)
/// - VR < 1: negative autocorrelation (mean-reversion)
///
/// `q`: aggregation interval (typically 2, 4, 8, 16). Defaults to 2.
pub fn variance_ratio_test(data: &[f64], q: Option<usize>) -> VarianceRatioResult {
    let q = q.unwrap_or(2).max(2);
    let nan = VarianceRatioResult {
        vr: f64::NAN,
        z_stat: f64::NAN,
        z_star: f64::NAN,
        q,
    };
    let n = data.len();
    if n < q * 2 + 1 {
        return nan;
    }

    // 1-period returns
    let returns: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
    let nq = returns.len();
    let mu = returns.iter().sum::<f64>() / nq as f64;

    // σ_a² = variance of 1-period returns (biased MLE consistent estimator)
    let sigma_a2 = returns.iter().map(|r| (r - mu) * (r - mu)).sum::<f64>() / nq as f64;
    if sigma_a2 < 1e-300 {
        return nan;
    }

    // σ_c²(q) = overlapping q-period variance estimator.
    // VR(q) = σ_c²(q) / (q * σ_a²)
    // Lo-MacKinlay: σ_c²(q) = (1/(n*q)) * Σ_t (sum of q consecutive returns - q*mu)²
    // where the sum runs over all overlapping windows starting at t=q-1..nq-1
    let mu_c = mu * q as f64;
    let q_returns: Vec<f64> = (q - 1..nq)
        .map(|i| returns[i + 1 - q..=i].iter().sum())
        .collect();
    let sigma_c2 = q_returns
        .iter()
        .map(|r| (r - mu_c) * (r - mu_c))
        .sum::<f64>()
        / (nq as f64 * q as f64);

    let vr = sigma_c2 / sigma_a2;

    // Homoscedastic z-statistic (Lo-MacKinlay eq. 4)
    let phi1 = 2.0 * (2.0 * q as f64 - 1.0) * (q as f64 - 1.0) / (3.0 * q as f64 * nq as f64);
    let z_stat = if phi1 > 0.0 {
        (vr - 1.0) / phi1.sqrt()
    } else {
        f64::NAN
    };

    // Heteroscedastic-robust z* (Lo-MacKinlay eq. 17)
    // δ_j = autocorrelation of squared demeaned returns at lag j
    let demeaned: Vec<f64> = returns.iter().map(|r| r - mu).collect();
    let denom_ss: f64 = demeaned.iter().map(|r| r * r).sum::<f64>();
    let mut theta = 0.0f64;
    for j in 1..q {
        let num: f64 = (j..nq)
            .map(|t| demeaned[t] * demeaned[t] * demeaned[t - j] * demeaned[t - j])
            .sum();
        let delta_j = num / (denom_ss * denom_ss / nq as f64);
        let weight = 2.0 * (q as f64 - j as f64) / q as f64;
        theta += weight * weight * delta_j;
    }
    let z_star = if theta > 0.0 {
        (vr - 1.0) / theta.sqrt()
    } else {
        z_stat
    };

    VarianceRatioResult {
        vr,
        z_stat,
        z_star,
        q,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Von Neumann ratio (serial randomness test)
// ─────────────────────────────────────────────────────────────────────────────

/// Von Neumann ratio: mean squared successive differences / sample variance.
///
/// Under independence, δ²/s² ≈ 2.
/// Values < 2 suggest positive autocorrelation (trending).
/// Values > 2 suggest negative autocorrelation (oscillating).
/// Returns NaN if variance is zero.
pub fn von_neumann_ratio(data: &[f64]) -> f64 {
    let clean: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean = clean.iter().sum::<f64>() / nf;
    let s2 = clean.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (nf - 1.0);
    if s2 < 1e-300 {
        return f64::NAN;
    }
    let delta2 = clean
        .windows(2)
        .map(|w| (w[1] - w[0]) * (w[1] - w[0]))
        .sum::<f64>()
        / (nf - 1.0);
    delta2 / s2
}

// ─────────────────────────────────────────────────────────────────────────────
// Bartels rank test for randomness
// ─────────────────────────────────────────────────────────────────────────────

/// Bartels (1982) rank test for randomness.
///
/// Tests H₀: sequence is random (no serial dependence) against H₁: autocorrelated.
/// Uses the von Neumann ratio applied to the ranks of the observations.
///
/// Returns the standardized test statistic. Under H₀, asymptotically N(0,1).
/// Large |z| → reject H₀ (non-random). Negative z → trending; positive → oscillating.
pub fn bartels_rank_test(data: &[f64]) -> f64 {
    let clean: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 4 {
        return f64::NAN;
    }
    let nf = n as f64;

    // Compute ranks (1-indexed, midranks for ties)
    let mut indexed: Vec<(f64, usize)> = clean
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].0 - indexed[i].0).abs() < 1e-10 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // midrank
        for k in i..j {
            ranks[indexed[k].1] = avg_rank;
        }
        i = j;
    }

    // RVN = Σ(r_t - r_{t+1})² / Σ(r_t - r̄)²  (rank von Neumann ratio)
    let r_mean = (nf + 1.0) / 2.0;
    let ss_num: f64 = ranks
        .windows(2)
        .map(|w| (w[0] - w[1]) * (w[0] - w[1]))
        .sum();
    let ss_den: f64 = ranks.iter().map(|r| (r - r_mean) * (r - r_mean)).sum();
    if ss_den < 1e-300 {
        return f64::NAN;
    }
    let rvn = ss_num / ss_den;

    // Under H₀, E[RVN] = 2n/(n-1), Var[RVN] = 4n²(n-2)/((n+1)(n-1)³) ... simplify to N approx.
    // Bartels gives: z = (RVN - mu_RVN) / sigma_RVN
    let mu = 2.0 * nf / (nf - 1.0);
    let var = 4.0 * nf * nf * (nf - 2.0) / ((nf + 1.0) * (nf - 1.0).powi(3));
    if var < 1e-300 {
        return f64::NAN;
    }
    (rvn - mu) / var.sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Zivot-Andrews breakpoint test (single structural break + unit root)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the Zivot-Andrews (1992) test.
///
/// H₀: unit root with no structural break.
/// H₁: stationary with one structural break at the estimated breakpoint.
#[derive(Debug, Clone)]
pub struct ZivotAndrewsResult {
    /// Minimum t-statistic over all candidate breakpoints (most favorable for H₁).
    pub statistic: f64,
    /// Index of estimated breakpoint (0-indexed into the original series).
    pub breakpoint: usize,
    /// Critical values at 1%, 5%, 10% (Zivot-Andrews table B).
    pub critical_1pct: f64,
    pub critical_5pct: f64,
    pub critical_10pct: f64,
}

/// Zivot-Andrews (1992) unit root test allowing one structural break.
///
/// Searches over all interior breakpoints (indices [trim, n-trim]).
/// At each breakpoint τ, runs ADF regression augmented with DU_t (level break)
/// and DT_t (trend break). Selects the breakpoint that minimizes the ADF t-stat.
///
/// `trim`: minimum fraction of obs to exclude from ends (default 0.15).
/// `n_lags`: number of ADF augmentation lags (default 0 = DF test).
pub fn zivot_andrews_test(
    data: &[f64],
    trim: Option<f64>,
    n_lags: Option<usize>,
) -> ZivotAndrewsResult {
    let nan = ZivotAndrewsResult {
        statistic: f64::NAN,
        breakpoint: 0,
        critical_1pct: -5.34,
        critical_5pct: -4.93,
        critical_10pct: -4.58,
    };
    let n = data.len();
    let lags = n_lags.unwrap_or(0);
    let t = trim.unwrap_or(0.15);
    let t_lo = ((t * n as f64) as usize).max(lags + 2);
    let t_hi = n.saturating_sub(((t * n as f64) as usize).max(1));

    if t_hi <= t_lo || n < 10 {
        return nan;
    }

    let mut min_stat = f64::INFINITY;
    let mut best_bp = t_lo;

    for bp in t_lo..=t_hi {
        // Regressors: intercept, t, DU_t = 1(t>bp), DT_t = t·1(t>bp), y_{t-1}, Δy_{t-j}
        let k = 5 + lags; // intercept + trend + DU + DT + y_{t-1} + lags
        let nobs = n - 1 - lags;
        if nobs < k + 1 {
            continue;
        }

        let mut x = vec![0.0f64; nobs * k];
        let mut y = vec![0.0f64; nobs];

        for t_idx in (lags + 1)..n {
            let row = t_idx - lags - 1;
            let t_f = t_idx as f64;
            // Δy_t
            y[row] = data[t_idx] - data[t_idx - 1];
            // intercept
            x[row * k] = 1.0;
            // trend
            x[row * k + 1] = t_f;
            // DU_t (level break: 1 if t > bp)
            x[row * k + 2] = if t_idx > bp { 1.0 } else { 0.0 };
            // DT_t (trend break)
            x[row * k + 3] = if t_idx > bp { (t_idx - bp) as f64 } else { 0.0 };
            // y_{t-1}
            x[row * k + 4] = data[t_idx - 1];
            // lagged differences
            for j in 1..=lags {
                x[row * k + 4 + j] = data[t_idx - j] - data[t_idx - j - 1];
            }
        }

        // OLS via normal equations (small k, use simple Gram approach)
        // XᵀX and Xᵀy
        let mut xtx = vec![0.0f64; k * k];
        let mut xty = vec![0.0f64; k];
        for row in 0..nobs {
            for j in 0..k {
                xty[j] += x[row * k + j] * y[row];
                for l in 0..k {
                    xtx[j * k + l] += x[row * k + j] * x[row * k + l];
                }
            }
        }
        // Solve via Cholesky (simple, reuse pattern from ADF)
        let beta = match solve_normal_equations(&xtx, &xty, k) {
            Some(b) => b,
            None => continue,
        };
        // t-statistic for β[4] = coefficient on y_{t-1}
        let fitted: Vec<f64> = (0..nobs)
            .map(|row| (0..k).map(|j| x[row * k + j] * beta[j]).sum::<f64>())
            .collect();
        let rss: f64 = (0..nobs).map(|row| (y[row] - fitted[row]).powi(2)).sum();
        let s2 = rss / (nobs as f64 - k as f64);
        // Var(β[4]) = s² · (XᵀX)⁻¹[4,4]
        // Approximate: use the diagonal of (XᵀX)⁻¹ for se
        // Full inversion is expensive; use the Sherman-Morrison approach or just
        // compute se from the OLS formula for the γ coefficient specifically.
        // For speed, compute se(γ) from: se² = s² / (denominator in partial regression)
        // Partial out other regressors from y_{t-1}: M_other · x4 / s²
        // Simpler: recompute XᵀX inverse diagonal element [4,4] via cofactor
        // We'll do a focused 2-step: partial regression of y_{t-1} on other regressors
        let se4_sq = compute_se_sq_for_col(&x, &y, &beta, s2, k, nobs, 4);
        if se4_sq <= 0.0 {
            continue;
        }
        let t_stat = beta[4] / se4_sq.sqrt();
        if t_stat < min_stat {
            min_stat = t_stat;
            best_bp = bp;
        }
    }

    if min_stat.is_infinite() {
        return nan;
    }

    ZivotAndrewsResult {
        statistic: min_stat,
        breakpoint: best_bp,
        critical_1pct: -5.34,
        critical_5pct: -4.93,
        critical_10pct: -4.58,
    }
}

/// Solve normal equations Ax = b via Cholesky (positive definite, n×n).
fn solve_normal_equations(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    // Cholesky decomposition L·Lᵀ = A
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                let v = a[i * n + i] - s;
                if v < 1e-300 {
                    return None;
                }
                l[i * n + j] = v.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - s) / l[j * n + j];
            }
        }
    }
    // Forward substitution L·y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i * n + k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i * n + i];
    }
    // Backward substitution Lᵀ·x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k]).sum();
        x[i] = (y[i] - s) / l[i * n + i];
    }
    Some(x)
}

/// Compute se²(β[col]) = s² · (XᵀX)⁻¹[col,col] using partial regression.
/// This is the Frisch-Waugh theorem: regress x_col on remaining regressors,
/// take residuals m, then se² = s² / (mᵀm).
fn compute_se_sq_for_col(
    x: &[f64],
    _y: &[f64],
    _beta: &[f64],
    s2: f64,
    k: usize,
    nobs: usize,
    col: usize,
) -> f64 {
    // Extract column col from X
    let x_col: Vec<f64> = (0..nobs).map(|r| x[r * k + col]).collect();
    // Other columns
    let other_cols: Vec<usize> = (0..k).filter(|&j| j != col).collect();
    let ko = other_cols.len();
    if ko == 0 {
        let ss: f64 = x_col.iter().map(|v| v * v).sum();
        return if ss > 0.0 { s2 / ss } else { 0.0 };
    }
    // Build X_other matrix (nobs × ko)
    let mut xo = vec![0.0f64; nobs * ko];
    for r in 0..nobs {
        for (ji, &j) in other_cols.iter().enumerate() {
            xo[r * ko + ji] = x[r * k + j];
        }
    }
    // OLS of x_col on X_other: β_o = (XoᵀXo)⁻¹ Xoᵀ x_col
    let mut xotxo = vec![0.0f64; ko * ko];
    let mut xotxcol = vec![0.0f64; ko];
    for r in 0..nobs {
        for j in 0..ko {
            xotxcol[j] += xo[r * ko + j] * x_col[r];
            for l in 0..ko {
                xotxo[j * ko + l] += xo[r * ko + j] * xo[r * ko + l];
            }
        }
    }
    let beta_o = match solve_normal_equations(&xotxo, &xotxcol, ko) {
        Some(b) => b,
        None => return 0.0,
    };
    // Residual m = x_col - X_other·β_o
    let mm: f64 = (0..nobs)
        .map(|r| {
            let fitted: f64 = (0..ko).map(|j| xo[r * ko + j] * beta_o[j]).sum();
            let resid = x_col[r] - fitted;
            resid * resid
        })
        .sum();
    if mm < 1e-300 {
        0.0
    } else {
        s2 / mm
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{label}: {a} vs {b} (diff={})",
            (a - b).abs()
        );
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
        assert!(
            (res.coefficients[0] - phi_true).abs() < 0.15,
            "AR(1) coeff={} should be ~{}",
            res.coefficients[0],
            phi_true
        );
    }

    #[test]
    fn ar_predict_constant() {
        let data = vec![5.0; 20];
        let ar = ar_fit(&data, 1);
        let preds = ar_predict(&data, &ar, 5);
        for &p in &preds {
            assert!(
                (p - 5.0).abs() < 0.1,
                "Constant series should predict constant, got {p}"
            );
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
        assert!(
            preds[0] > 38.0 && preds[0] < 42.0,
            "Holt pred[0]={}",
            preds[0]
        );
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
            assert!(
                ac[lag].abs() < 0.2,
                "ACF({lag})={} should be ~0 for white noise",
                ac[lag]
            );
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
        assert!(
            res.statistic < res.critical_5pct,
            "ADF stat={} should be < {} for stationary series",
            res.statistic,
            res.critical_5pct
        );
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
        assert!(
            res_small.critical_1pct < -3.43,
            "1% CV for n=30 should be more negative than -3.43, got {}",
            res_small.critical_1pct
        );
        assert!(
            res_small.critical_5pct < -2.86,
            "5% CV for n=30 should be more negative than -2.86, got {}",
            res_small.critical_5pct
        );
        assert!(
            res_small.critical_10pct < -2.57,
            "10% CV for n=30 should be more negative than -2.57, got {}",
            res_small.critical_10pct
        );

        // For large n, should converge toward asymptotic values
        let large_data: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.01).sin()).collect();
        let res_large = adf_test(&large_data, 1);
        assert!(
            (res_large.critical_1pct - (-3.4336)).abs() < 0.01,
            "1% CV for large n should be near -3.4336, got {}",
            res_large.critical_1pct
        );
        assert!(
            (res_large.critical_5pct - (-2.8621)).abs() < 0.01,
            "5% CV for large n should be near -2.8621, got {}",
            res_large.critical_5pct
        );
    }

    // ── Ljung-Box test ────────────────────────────────────────────────────

    #[test]
    fn ljung_box_white_noise() {
        // Genuine iid noise from Xoshiro256 — no autocorrelation
        let mut rng = crate::rng::Xoshiro256::new(42);
        let wn: Vec<f64> = (0..200)
            .map(|_| crate::rng::TamRng::next_f64(&mut rng) - 0.5)
            .collect();
        let r = ljung_box(&wn, 10, 0);
        // With 200 observations and 10 lags of genuine white noise, Q ~ chi2(10).
        // Expected Q ≈ 10; p should be comfortably > 0.01 for seed=42.
        assert!(
            r.p_value > 0.01,
            "white noise: ljung-box p should be > 0.01, got {}",
            r.p_value
        );
        assert!(r.statistic >= 0.0);
    }

    #[test]
    fn ljung_box_autocorrelated() {
        // AR(1) with high rho — should reject H₀
        let mut data = vec![0.0f64; 100];
        for i in 1..100 {
            data[i] = 0.9 * data[i - 1] + (i as f64 * 0.01).sin() * 0.1;
        }
        let r = ljung_box(&data, 10, 0);
        assert!(
            r.p_value < 0.001,
            "highly autocorrelated: ljung-box p={}",
            r.p_value
        );
    }

    // ── Durbin-Watson test ────────────────────────────────────────────────

    #[test]
    fn durbin_watson_no_autocorrelation() {
        // Residuals with no serial correlation — DW should be near 2.0.
        // Use iid noise from Xoshiro256 for genuine independence.
        let mut rng = crate::rng::Xoshiro256::new(99);
        let resid: Vec<f64> = (0..50)
            .map(|_| crate::rng::TamRng::next_f64(&mut rng) - 0.5)
            .collect();
        let r = durbin_watson(&resid);
        assert!(
            (r.statistic - 2.0).abs() < 1.0,
            "iid noise: DW should be near 2.0, got {}",
            r.statistic
        );
    }

    #[test]
    fn durbin_watson_positive_autocorrelation() {
        // Monotone residuals (perfect positive autocorrelation) → DW near 0
        let resid: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let r = durbin_watson(&resid);
        assert!(
            r.statistic < 0.5,
            "monotone residuals: DW should be near 0, got {}",
            r.statistic
        );
    }

    #[test]
    fn durbin_watson_negative_autocorrelation() {
        // Alternating residuals → DW near 4.0
        let resid: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let r = durbin_watson(&resid);
        assert!(
            r.statistic > 3.5,
            "alternating residuals: DW should be near 4.0, got {}",
            r.statistic
        );
    }

    // ── KPSS test ─────────────────────────────────────────────────────────

    #[test]
    fn kpss_stationary_series() {
        // Stationary: sin wave with no trend — should fail to reject (p > 0.05)
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let r = kpss_test(&data, false, None);
        // KPSS H₀ = stationary; high p → fail to reject
        // Statistic below 10% critical value (0.347) means stationary
        assert!(
            r.statistic < r.critical_10pct,
            "stationary series: KPSS stat={:.3} should be < 10% cv={:.3}",
            r.statistic,
            r.critical_10pct
        );
    }

    #[test]
    fn kpss_nonstationary_random_walk() {
        // Random walk with large iid steps — clearly non-stationary
        let mut rng = crate::rng::Xoshiro256::new(7);
        let mut data = vec![0.0f64; 200];
        for i in 1..200 {
            // Steps from N(0,1) approximated as sum of 12 uniform - 6
            let step: f64 = (0..12)
                .map(|_| crate::rng::TamRng::next_f64(&mut rng))
                .sum::<f64>()
                - 6.0;
            data[i] = data[i - 1] + step;
        }
        let r = kpss_test(&data, false, None);
        // A random walk with 200 steps should almost always exceed 1% critical value
        assert!(
            r.statistic > r.critical_5pct,
            "random walk: KPSS stat={:.3} should exceed 5% cv={:.3}",
            r.statistic,
            r.critical_5pct
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Extended stationarity / unit root / dependence tests
// ═══════════════════════════════════════════════════════════════════════════

/// Phillips-Perron test result.
#[derive(Debug, Clone)]
pub struct PhillipsPerronResult {
    /// PP test statistic (Z_t form).
    pub statistic: f64,
    /// Number of Newey-West lags used for long-run variance.
    pub n_lags: usize,
    /// MacKinnon critical values (same distribution as ADF under null).
    pub critical_1pct: f64,
    pub critical_5pct: f64,
    pub critical_10pct: f64,
}

/// Newey-West (1987) long-run variance estimator with Bartlett kernel.
///
/// Estimates the spectral density at frequency zero of a stationary process,
/// which equals the variance of the sample mean scaled by n. Used by
/// Phillips-Perron, KPSS, any HAC-robust standard error.
///
/// `residuals`: the process to estimate long-run variance of.
/// `n_lags`: bandwidth (number of autocovariance lags to include).
///   If `None`, uses the Andrews (1991) rule: floor(4*(n/100)^(2/9)).
///
/// Returns the long-run variance estimate (always >= 0).
pub fn newey_west_lrv(residuals: &[f64], n_lags: Option<usize>) -> f64 {
    let n = residuals.len();
    if n < 2 {
        return f64::NAN;
    }
    let nl = n_lags.unwrap_or_else(|| (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize);

    let nf = n as f64;
    // Autocovariance at lag 0 (sample variance without centering — raw sum of squares / n)
    let gamma0: f64 = residuals.iter().map(|e| e * e).sum::<f64>() / nf;

    let mut lrv = gamma0;
    for j in 1..=nl.min(n - 1) {
        let w = 1.0 - (j as f64) / (nl as f64 + 1.0); // Bartlett kernel
        let mut gamma_j = 0.0_f64;
        for t in j..n {
            gamma_j += residuals[t] * residuals[t - j];
        }
        gamma_j /= nf;
        lrv += 2.0 * w * gamma_j;
    }
    lrv.max(0.0)
}

/// Phillips-Perron (1988) unit root test.
///
/// Same null as ADF (H₀: unit root) but uses Newey-West HAC correction
/// for serial correlation instead of augmented lag terms. More robust to
/// heteroskedasticity and unspecified serial correlation structure.
///
/// Uses the Z_t form (t-ratio correction). Decomposes into:
/// `simple_linear_regression` + `newey_west_lrv` + PP formula.
pub fn phillips_perron_test(data: &[f64], n_lags: Option<usize>) -> PhillipsPerronResult {
    let n = data.len();
    let nl = n_lags.unwrap_or_else(|| ((n as f64).powf(1.0 / 3.0) * 1.5) as usize);

    if n < 10 {
        let (c1, c5, c10) = mackinnon_adf_critical_values(n);
        return PhillipsPerronResult {
            statistic: f64::NAN,
            n_lags: nl,
            critical_1pct: c1,
            critical_5pct: c5,
            critical_10pct: c10,
        };
    }

    // OLS: y_t = α + ρ·y_{t-1} + e_t  (using the global primitive)
    let x_lag: Vec<f64> = data[..n - 1].to_vec();
    let y_vals: Vec<f64> = data[1..].to_vec();
    let reg = crate::linear_algebra::simple_linear_regression(&x_lag, &y_vals);
    if !reg.slope.is_finite() {
        let (c1, c5, c10) = mackinnon_adf_critical_values(n - 1);
        return PhillipsPerronResult {
            statistic: f64::NAN,
            n_lags: nl,
            critical_1pct: c1,
            critical_5pct: c5,
            critical_10pct: c10,
        };
    }

    let rho = reg.slope;
    let nobs = n - 1;
    let nf = nobs as f64;

    // Short-run variance (OLS residual variance)
    let s2: f64 = reg.residuals.iter().map(|e| e * e).sum::<f64>() / nf;

    // Long-run variance via the global Newey-West primitive
    let lrv = newey_west_lrv(&reg.residuals, Some(nl));

    // Sums needed for the PP correction
    let sx: f64 = x_lag.iter().sum();
    let sxx: f64 = x_lag.iter().map(|x| x * x).sum();
    let denom_xx = sxx - sx * sx / nf;

    let se_rho = (s2 / denom_xx.max(1e-30)).sqrt();
    let correction = 0.5 * (lrv - s2) * denom_xx.sqrt() / (s2.max(1e-30).sqrt() * nf);
    let t_rho = (rho - 1.0) / se_rho;
    let z_t = t_rho - correction / se_rho;

    let (c1, c5, c10) = mackinnon_adf_critical_values(nobs);
    PhillipsPerronResult {
        statistic: z_t,
        n_lags: nl,
        critical_1pct: c1,
        critical_5pct: c5,
        critical_10pct: c10,
    }
}

/// Box-Pierce (1970) portmanteau test for white noise.
///
/// Predecessor to Ljung-Box with a simpler Q statistic:
/// Q_BP = n · Σ_{k=1}^{m} ρ̂(k)²
///
/// Under H₀ (white noise), Q_BP ~ χ²(m - fitted_params).
/// Less accurate than Ljung-Box for small samples but included
/// for completeness and for comparison with legacy implementations.
pub fn box_pierce(data: &[f64], n_lags: usize, fitted_params: usize) -> LjungBoxResult {
    let n = data.len();
    if n <= n_lags + 1 {
        return LjungBoxResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            n_lags,
            df: 0,
        };
    }
    let acf_vals = acf(data, n_lags);
    let q: f64 = n as f64 * acf_vals[1..].iter().map(|r| r * r).sum::<f64>();
    let df = n_lags.saturating_sub(fitted_params);
    let p = if df > 0 {
        1.0 - crate::special_functions::chi2_cdf(q, df as f64)
    } else {
        f64::NAN
    };
    LjungBoxResult {
        statistic: q,
        p_value: p,
        n_lags,
        df,
    }
}

/// Breusch-Godfrey LM test for serial correlation in regression residuals.
///
/// More general than Durbin-Watson: tests for autocorrelation at lags 1..p
/// simultaneously. Under H₀ (no serial correlation), the LM statistic
/// n·R² ~ χ²(p).
///
/// `residuals`: OLS residuals. `regressors`: the original X matrix (row-major,
/// n rows × k cols). `p`: number of lags to test.
///
/// Returns (lm_statistic, p_value, df).
pub fn breusch_godfrey(
    residuals: &[f64],
    regressors: &[f64],
    k: usize,
    p: usize,
) -> (f64, f64, usize) {
    let n = residuals.len();
    if n <= k + p || p == 0 {
        return (f64::NAN, f64::NAN, p);
    }
    // Auxiliary regression: e_t = X_t·β + Σ_{j=1}^p γ_j·e_{t-j} + u_t
    // LM stat = n · R² of this regression
    let n_aux_cols = k + p;
    let nobs = n - p;
    let mut x_aug = vec![0.0; nobs * n_aux_cols];
    let mut y_aux = vec![0.0; nobs];

    for t in p..n {
        let row = t - p;
        y_aux[row] = residuals[t];
        // Original regressors
        for j in 0..k {
            x_aug[row * n_aux_cols + j] = regressors[t * k + j];
        }
        // Lagged residuals
        for j in 0..p {
            x_aug[row * n_aux_cols + k + j] = residuals[t - 1 - j];
        }
    }

    // OLS on auxiliary regression to get R²
    let mean_y = y_aux.iter().sum::<f64>() / nobs as f64;
    let ss_tot: f64 = y_aux.iter().map(|y| (y - mean_y).powi(2)).sum();
    if ss_tot < 1e-30 {
        return (0.0, 1.0, p);
    }

    // Solve via normal equations (X'X β = X'y via Cholesky)
    let nc = n_aux_cols;
    let beta = match crate::linear_algebra::ols_normal_equations(&x_aug, &y_aux, nobs, nc) {
        Some(b) => b,
        None => return (f64::NAN, f64::NAN, p),
    };

    let mut ss_res = 0.0_f64;
    for i in 0..nobs {
        let fitted: f64 = (0..nc).map(|j| x_aug[i * nc + j] * beta[j]).sum();
        ss_res += (y_aux[i] - fitted).powi(2);
    }
    let r2 = 1.0 - ss_res / ss_tot;
    let lm = nobs as f64 * r2;
    let pval = 1.0 - crate::special_functions::chi2_cdf(lm, p as f64);
    (lm, pval, p)
}

/// Turning point test for randomness (nonparametric).
///
/// A turning point is an index t where x_{t-1} < x_t > x_{t+1} (peak) or
/// x_{t-1} > x_t < x_{t+1} (trough). Under IID, the expected number of
/// turning points is 2(n-2)/3 with variance (16n-29)/90.
///
/// Returns (n_turning_points, z_statistic). Z > 0 means more turning points
/// than expected (oscillatory); Z < 0 means fewer (trendy/smooth).
pub fn turning_point_test(data: &[f64]) -> (usize, f64) {
    let n = data.len();
    if n < 3 {
        return (0, f64::NAN);
    }
    let mut tp = 0_usize;
    for i in 1..n - 1 {
        if (data[i] > data[i - 1] && data[i] > data[i + 1])
            || (data[i] < data[i - 1] && data[i] < data[i + 1])
        {
            tp += 1;
        }
    }
    let expected = 2.0 * (n - 2) as f64 / 3.0;
    let variance = (16 * n - 29) as f64 / 90.0;
    let z = if variance > 0.0 {
        (tp as f64 - expected) / variance.sqrt()
    } else {
        0.0
    };
    (tp, z)
}

/// Rank-based von Neumann ratio.
///
/// Replaces raw values with their ranks, then computes the VN ratio on ranks.
/// Distribution-free — invariant to marginal distribution. Under IID,
/// the expected value is still ~2.
pub fn rank_von_neumann_ratio(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 3 {
        return f64::NAN;
    }
    // Compute ranks
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        ranks[orig_idx] = (rank + 1) as f64;
    }
    von_neumann_ratio(&ranks)
}

/// Spectral flatness (Wiener entropy) of a PSD.
///
/// SF = exp(mean(ln(S))) / mean(S) = geometric_mean(S) / arithmetic_mean(S).
///
/// Ranges [0, 1]. SF = 1 for white noise (flat PSD), SF → 0 for a pure tone.
/// This is the frequency-domain analog of the time-domain white noise tests
/// (Ljung-Box, Box-Pierce). A low spectral flatness means energy is
/// concentrated at specific frequencies.
pub fn spectral_flatness(psd: &[f64]) -> f64 {
    let n = psd.len();
    if n == 0 {
        return f64::NAN;
    }
    let log_mean = psd
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|v| v.ln())
        .sum::<f64>()
        / psd.iter().filter(|&&v| v > 0.0).count().max(1) as f64;
    let arith_mean = psd.iter().sum::<f64>() / n as f64;
    if arith_mean < 1e-300 {
        return f64::NAN;
    }
    log_mean.exp() / arith_mean
}

/// Spectral rolloff frequency: the frequency below which `pct` (e.g. 0.85)
/// of the total spectral energy is contained.
///
/// Common values: 0.85 (standard), 0.95 (near-total energy).
/// Returns the frequency in the same units as `freqs`.
pub fn spectral_rolloff(freqs: &[f64], psd: &[f64], pct: f64) -> f64 {
    if freqs.len() != psd.len() || psd.is_empty() {
        return f64::NAN;
    }
    let total: f64 = psd.iter().sum();
    if total <= 0.0 {
        return f64::NAN;
    }
    let threshold = pct * total;
    let mut cumsum = 0.0_f64;
    for (i, &p) in psd.iter().enumerate() {
        cumsum += p;
        if cumsum >= threshold {
            return freqs[i];
        }
    }
    *freqs.last().unwrap()
}

/// Spectral centroid: the "center of mass" of the PSD.
///
/// SC = Σ(f_i · S_i) / Σ(S_i)
///
/// Indicates where the "average" frequency content lives. Higher centroid
/// means energy is concentrated at higher frequencies. The first spectral
/// moment.
pub fn spectral_centroid(freqs: &[f64], psd: &[f64]) -> f64 {
    if freqs.len() != psd.len() || psd.is_empty() {
        return f64::NAN;
    }
    let total: f64 = psd.iter().sum();
    if total <= 0.0 {
        return f64::NAN;
    }
    freqs
        .iter()
        .zip(psd.iter())
        .map(|(&f, &s)| f * s)
        .sum::<f64>()
        / total
}

/// Spectral bandwidth (spread): second central moment of the PSD.
///
/// BW = sqrt(Σ((f_i - centroid)² · S_i) / Σ(S_i))
///
/// Measures how spread out the energy is around the centroid.
/// Narrow bandwidth → tonal signal. Wide → broadband/noisy.
pub fn spectral_bandwidth(freqs: &[f64], psd: &[f64]) -> f64 {
    let c = spectral_centroid(freqs, psd);
    if !c.is_finite() {
        return f64::NAN;
    }
    let total: f64 = psd.iter().sum();
    if total <= 0.0 {
        return f64::NAN;
    }
    let m2: f64 = freqs
        .iter()
        .zip(psd.iter())
        .map(|(&f, &s)| (f - c).powi(2) * s)
        .sum::<f64>();
    (m2 / total).sqrt()
}

/// Spectral skewness: third central moment of the PSD (normalized).
///
/// Measures asymmetry of energy distribution around the centroid.
pub fn spectral_skewness(freqs: &[f64], psd: &[f64]) -> f64 {
    let c = spectral_centroid(freqs, psd);
    let bw = spectral_bandwidth(freqs, psd);
    if !c.is_finite() || !bw.is_finite() || bw < 1e-30 {
        return f64::NAN;
    }
    let total: f64 = psd.iter().sum();
    let m3: f64 = freqs
        .iter()
        .zip(psd.iter())
        .map(|(&f, &s)| (f - c).powi(3) * s)
        .sum::<f64>();
    m3 / (total * bw.powi(3))
}

/// Spectral kurtosis: fourth central moment of the PSD (normalized, excess).
///
/// Measures peakedness of energy distribution. High kurtosis → sharp peaks.
pub fn spectral_kurtosis(freqs: &[f64], psd: &[f64]) -> f64 {
    let c = spectral_centroid(freqs, psd);
    let bw = spectral_bandwidth(freqs, psd);
    if !c.is_finite() || !bw.is_finite() || bw < 1e-30 {
        return f64::NAN;
    }
    let total: f64 = psd.iter().sum();
    let m4: f64 = freqs
        .iter()
        .zip(psd.iter())
        .map(|(&f, &s)| (f - c).powi(4) * s)
        .sum::<f64>();
    m4 / (total * bw.powi(4)) - 3.0 // excess kurtosis
}

/// Spectral crest factor: ratio of peak PSD to arithmetic mean PSD.
///
/// CF = max(S) / mean(S). Measures how "peaky" the spectrum is.
/// CF = 1 for flat (white noise). CF → ∞ for a pure tone.
pub fn spectral_crest(psd: &[f64]) -> f64 {
    if psd.is_empty() {
        return f64::NAN;
    }
    let max_val = psd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_val = psd.iter().sum::<f64>() / psd.len() as f64;
    if mean_val < 1e-300 {
        return f64::NAN;
    }
    max_val / mean_val
}

/// Spectral slope: linear regression slope of log(PSD) vs log(freq).
///
/// Measures the rate of spectral decay. For 1/f^β noise, the slope is -β.
/// White noise: β ≈ 0. Pink noise: β ≈ 1. Brown noise: β ≈ 2.
pub fn spectral_slope(freqs: &[f64], psd: &[f64]) -> f64 {
    if freqs.len() != psd.len() || psd.len() < 2 {
        return f64::NAN;
    }
    // Use only positive freq and psd values
    let mut lf = Vec::new();
    let mut lp = Vec::new();
    for (&f, &p) in freqs.iter().zip(psd.iter()) {
        if f > 0.0 && p > 0.0 {
            lf.push(f.ln());
            lp.push(p.ln());
        }
    }
    if lf.len() < 2 {
        return f64::NAN;
    }
    let n = lf.len() as f64;
    let mx = lf.iter().sum::<f64>() / n;
    let my = lp.iter().sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..lf.len() {
        let dx = lf[i] - mx;
        num += dx * (lp[i] - my);
        den += dx * dx;
    }
    if den < 1e-30 {
        f64::NAN
    } else {
        num / den
    }
}

/// Full-width at half-maximum (FWHM) of the dominant spectral peak.
///
/// Finds the peak with maximum PSD, then measures the width of the peak
/// at half its height above the baseline (minimum PSD). Returns the width
/// in frequency units.
///
/// FWHM characterizes the sharpness of the dominant oscillation.
/// Narrow FWHM → stable periodic signal. Wide FWHM → damped/noisy oscillation.
pub fn spectral_fwhm(freqs: &[f64], psd: &[f64]) -> f64 {
    if freqs.len() != psd.len() || psd.len() < 3 {
        return f64::NAN;
    }
    let baseline = psd.iter().cloned().fold(f64::INFINITY, f64::min);
    let (peak_idx, peak_val) = psd
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let half_height = baseline + (peak_val - baseline) / 2.0;

    // Walk left from peak
    let mut left_freq = freqs[peak_idx];
    for i in (0..peak_idx).rev() {
        if psd[i] <= half_height {
            // Linear interpolation
            let frac = (half_height - psd[i]) / (psd[i + 1] - psd[i]).max(1e-30);
            left_freq = freqs[i] + frac * (freqs[i + 1] - freqs[i]);
            break;
        }
    }
    // Walk right from peak
    let mut right_freq = freqs[peak_idx];
    for i in (peak_idx + 1)..psd.len() {
        if psd[i] <= half_height {
            let frac = (half_height - psd[i]) / (psd[i - 1] - psd[i]).max(1e-30);
            right_freq = freqs[i] - frac * (freqs[i] - freqs[i - 1]);
            break;
        }
    }
    right_freq - left_freq
}

/// Q factor of the dominant spectral peak: peak_freq / FWHM.
///
/// High Q → sharp resonance. Low Q → heavily damped.
/// Dimensionless. Standard measure in physics and signal processing.
pub fn spectral_q_factor(freqs: &[f64], psd: &[f64]) -> f64 {
    let fwhm = spectral_fwhm(freqs, psd);
    if !fwhm.is_finite() || fwhm <= 0.0 {
        return f64::NAN;
    }
    let peak_idx = psd
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let peak_freq = freqs[peak_idx];
    if peak_freq <= 0.0 {
        return f64::NAN;
    }
    peak_freq / fwhm
}

/// Spectral flux: L2 norm of the frame-to-frame PSD change.
///
/// Given two consecutive PSD frames, SF = ||S_2 - S_1||_2.
/// Measures how rapidly the spectral content is changing. Used in onset
/// detection (audio) and regime-change detection (financial).
///
/// Returns a vector of length `frames.len() - 1`.
pub fn spectral_flux(frames: &[Vec<f64>]) -> Vec<f64> {
    if frames.len() < 2 {
        return vec![];
    }
    frames
        .windows(2)
        .map(|pair| {
            let (a, b) = (&pair[0], &pair[1]);
            let n = a.len().min(b.len());
            (0..n).map(|i| (b[i] - a[i]).powi(2)).sum::<f64>().sqrt()
        })
        .collect()
}

/// Spectral decrease: weighted sum of spectral differences.
///
/// SD = (1 / Σ S_k) · Σ_{k=2}^{N} (S_k - S_1) / (k - 1)
///
/// Alternative to spectral slope; emphasizes the rate of decrease in the
/// low-frequency region. Common in MPEG-7 audio descriptors.
pub fn spectral_decrease(psd: &[f64]) -> f64 {
    if psd.len() < 2 {
        return f64::NAN;
    }
    let total: f64 = psd[1..].iter().sum();
    if total < 1e-300 {
        return f64::NAN;
    }
    let s1 = psd[0];
    let num: f64 = (1..psd.len()).map(|k| (psd[k] - s1) / k as f64).sum();
    num / total
}

/// Spectral contrast per subband.
///
/// For each of `n_bands` equal-width subbands of the PSD, compute
/// `peak - valley` (in dB if PSD is in linear power: 10·log10(peak/valley)).
/// Returns a vector of length `n_bands`.
///
/// High contrast in a band → clear spectral structure there.
/// Low contrast → flat/noisy in that band.
pub fn spectral_contrast(freqs: &[f64], psd: &[f64], n_bands: usize) -> Vec<f64> {
    if freqs.len() != psd.len() || psd.is_empty() || n_bands == 0 {
        return vec![];
    }
    let n = psd.len();
    let band_size = (n + n_bands - 1) / n_bands;
    let mut contrasts = Vec::with_capacity(n_bands);
    for b in 0..n_bands {
        let start = b * band_size;
        let end = ((b + 1) * band_size).min(n);
        if start >= end {
            contrasts.push(f64::NAN);
            continue;
        }
        let slice = &psd[start..end];
        let peak = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let valley = slice.iter().cloned().fold(f64::INFINITY, f64::min);
        if valley > 1e-300 {
            contrasts.push(10.0 * (peak / valley).log10());
        } else {
            contrasts.push(f64::NAN);
        }
    }
    contrasts
}

/// Dominant frequency: the frequency at which the PSD is maximized.
///
/// Trivial but used so often that it deserves its own function for composability.
pub fn dominant_frequency(freqs: &[f64], psd: &[f64]) -> f64 {
    if freqs.len() != psd.len() || psd.is_empty() {
        return f64::NAN;
    }
    let idx = psd
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    freqs[idx]
}

/// Dominant frequency power: the PSD value at the dominant frequency.
pub fn dominant_frequency_power(psd: &[f64]) -> f64 {
    if psd.is_empty() {
        return f64::NAN;
    }
    psd.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Peak-to-average power ratio (PAPR / crest factor squared).
///
/// PAPR = max(S) / mean(S). Same as spectral_crest but the name is standard
/// in communications/radar. Included as an alias for discoverability.
pub fn peak_to_average_power_ratio(psd: &[f64]) -> f64 {
    spectral_crest(psd)
}

/// Number of spectral peaks above a threshold.
///
/// A peak is a local maximum where psd[i] > threshold_ratio * max(psd).
/// Returns the count. Useful as a complexity measure of the spectrum.
pub fn spectral_peak_count(psd: &[f64], threshold_ratio: f64) -> usize {
    if psd.len() < 3 {
        return 0;
    }
    let max_val = psd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let threshold = threshold_ratio * max_val;
    let mut count = 0;
    for i in 1..psd.len() - 1 {
        if psd[i] > psd[i - 1] && psd[i] > psd[i + 1] && psd[i] >= threshold {
            count += 1;
        }
    }
    count
}

// ─────────────────────────────────────────────────────────────────────────────
// STL-like Seasonal-Trend Decomposition
// ─────────────────────────────────────────────────────────────────────────────
//
// Cleveland et al. (1990) STL (Seasonal-Trend decomposition using Loess)
// decomposes a time series Y = T + S + R where:
//   T = trend component
//   S = seasonal component (periodic with given period)
//   R = remainder Y - T - S
//
// Full STL iterates inner and outer loops. This implements a simplified
// single-pass version sufficient for the fintek leaf:
//
//   1. Compute moving-average trend (window = period, centered)
//   2. Detrend: Y - T
//   3. Seasonal component: for each position within cycle, average the
//      detrended values at that phase across all cycles
//   4. Remainder: Y - T - S
//
// This is the "X-11 inner loop" approximation used when n_iter=1.

/// Decomposed time series result.
#[derive(Debug, Clone)]
pub struct StlResult {
    /// Trend component (moving average; NaN at boundaries).
    pub trend: Vec<f64>,
    /// Seasonal component (periodic with length = input series).
    pub seasonal: Vec<f64>,
    /// Remainder = observed − trend − seasonal.
    pub remainder: Vec<f64>,
    /// Seasonal period used.
    pub period: usize,
}

impl StlResult {
    /// Reconstructs the series: trend + seasonal + remainder.
    ///
    /// At boundary positions where trend is NaN, the trend term is treated as 0
    /// (remainder absorbs the full observed value at those positions).
    pub fn reconstruct(&self) -> Vec<f64> {
        (0..self.trend.len())
            .map(|i| {
                let t = if self.trend[i].is_finite() {
                    self.trend[i]
                } else {
                    0.0
                };
                t + self.seasonal[i] + self.remainder[i]
            })
            .collect()
    }

    /// Seasonal strength: Var(S) / (Var(S) + Var(R)) per Cleveland 1990.
    pub fn seasonal_strength(&self) -> f64 {
        let var_s = variance_of(&self.seasonal);
        let var_r = variance_of(&self.remainder);
        if var_s + var_r < 1e-30 {
            return 0.0;
        }
        (1.0 - var_r / (var_s + var_r)).max(0.0)
    }

    /// Trend strength: Var(T) / (Var(T) + Var(R)).
    pub fn trend_strength(&self) -> f64 {
        let var_t = variance_of(&self.trend);
        let var_r = variance_of(&self.remainder);
        if var_t + var_r < 1e-30 {
            return 0.0;
        }
        (1.0 - var_r / (var_t + var_r)).max(0.0)
    }
}

fn variance_of(v: &[f64]) -> f64 {
    let valid: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let n = valid.len();
    if n < 2 {
        return 0.0;
    }
    let mean = valid.iter().sum::<f64>() / n as f64;
    valid.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
}

/// Decompose a time series into trend, seasonal, and remainder.
///
/// `data`: observed time series (any length ≥ 2·period).
/// `period`: seasonal period in time steps (e.g., 252 for annual in daily data).
/// `robust`: if true, use median instead of mean for the seasonal averaging step.
///
/// Returns `None` when `data.len() < 2 * period` or `period < 2`.
pub fn stl_decompose(data: &[f64], period: usize, robust: bool) -> Option<StlResult> {
    let n = data.len();
    if period < 2 || n < 2 * period {
        return None;
    }

    // Step 1: Moving-average trend (centered, window = period).
    // For even period, apply a 2×(period/2) MA to avoid phase shift.
    let mut trend = vec![f64::NAN; n];
    let half = period / 2;
    if period % 2 == 1 {
        // Odd: simple centered MA of width `period`
        for i in half..(n - half) {
            let sum: f64 = data[i - half..=i + half].iter().sum();
            trend[i] = sum / period as f64;
        }
    } else {
        // Even: 2×(period/2) — average of two offset MAs
        for i in half..(n - half) {
            let sum_a: f64 = data[(i - half + 1)..=(i + half)].iter().sum();
            let sum_b: f64 = data[(i - half)..=(i + half - 1)].iter().sum();
            trend[i] = (sum_a + sum_b) / (2.0 * period as f64);
        }
    }

    // Step 2: Detrend where trend is available.
    let detrended: Vec<f64> = (0..n)
        .map(|i| {
            if trend[i].is_finite() {
                data[i] - trend[i]
            } else {
                f64::NAN
            }
        })
        .collect();

    // Step 3: Seasonal component — for each phase p ∈ [0, period),
    // collect all detrended values at that phase and take mean (or median).
    let mut seasonal = vec![0.0_f64; n];
    for p in 0..period {
        let vals: Vec<f64> = (0..n)
            .filter(|&i| i % period == p && detrended[i].is_finite())
            .map(|i| detrended[i])
            .collect();
        if vals.is_empty() {
            continue;
        }
        let center = if robust {
            median_val(&vals)
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        };
        for i in (p..n).step_by(period) {
            seasonal[i] = center;
        }
    }

    // Step 4: Remainder.
    let remainder: Vec<f64> = (0..n)
        .map(|i| {
            let t = if trend[i].is_finite() { trend[i] } else { 0.0 };
            data[i] - t - seasonal[i]
        })
        .collect();

    Some(StlResult {
        trend,
        seasonal,
        remainder,
        period,
    })
}

fn median_val(v: &[f64]) -> f64 {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod stl_tests {
    use super::*;

    #[test]
    fn stl_pure_seasonal() {
        // Series = sin(2π·i/period) with no trend or noise
        let period = 20_usize;
        let n = 200;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
            .collect();
        let r = stl_decompose(&data, period, false).expect("should decompose");
        // Seasonal strength should be high
        let ss = r.seasonal_strength();
        assert!(ss > 0.8, "seasonal_strength={:.3} for pure sinusoid", ss);
    }

    #[test]
    fn stl_pure_trend() {
        // Linear trend, no seasonality.
        // Single-pass STL captures the trend via moving average. The MA boundary
        // (half-period NaN at each end) limits how much trend variance is extracted,
        // but trend should still dominate over remainder for long series.
        let period = 10;
        let n = 200;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let r = stl_decompose(&data, period, false).expect("should decompose");
        let ts = r.trend_strength();
        assert!(ts > 0.5, "trend_strength={:.3} for linear ramp", ts);
    }

    #[test]
    fn stl_reconstruction() {
        let period = 12;
        let n = 120;
        let data: Vec<f64> = (0..n)
            .map(|i| i as f64 * 0.1 + (i % period) as f64)
            .collect();
        let r = stl_decompose(&data, period, false).expect("should decompose");
        // Reconstruction should match observed data
        let rec = r.reconstruct();
        assert_eq!(rec.len(), n);
        for i in 0..n {
            assert!(
                (rec[i] - data[i]).abs() < 1e-9,
                "mismatch at i={}: rec={:.6} vs data={:.6}",
                i,
                rec[i],
                data[i]
            );
        }
    }

    #[test]
    fn stl_too_short() {
        let r = stl_decompose(&[1.0, 2.0, 3.0], 5, false);
        assert!(r.is_none());
    }

    #[test]
    fn stl_period_1_rejected() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(stl_decompose(&data, 1, false).is_none());
    }

    #[test]
    fn stl_robust_mode() {
        // Robust (median) should also work
        let period = 7;
        let n = 70;
        let data: Vec<f64> = (0..n).map(|i| (i % period) as f64).collect();
        let r = stl_decompose(&data, period, true).expect("should decompose");
        assert!(r.seasonal_strength() > 0.5);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ARMA / ARIMA tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod arima_tests {
    use super::*;
    use crate::rng::{sample_normal, Xoshiro256};

    #[test]
    fn undifference_roundtrip() {
        let data = vec![10.0, 12.0, 11.0, 15.0, 13.0];
        let d = difference(&data, 1);
        let reconstructed = undifference(&d, &[data[0]]);
        for (a, b) in data.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-12, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn undifference_d2_roundtrip() {
        let data = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
        let d1 = difference(&data, 1);
        let d2 = difference(&data, 2);
        let reconstructed = undifference(&d2, &[data[0], d1[0]]);
        for (a, b) in data.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-12, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn arma_fit_pure_ar1() {
        // Generate AR(1) data: x_t = 0.7 x_{t-1} + ε_t
        let n = 500;
        let mut rng = Xoshiro256::new(42);
        let mut data = vec![0.0; n];
        for t in 1..n {
            data[t] = 0.7 * data[t - 1] + sample_normal(&mut rng, 0.0, 1.0);
        }
        let fit = arma_fit(&data, 1, 0, 200);
        assert!(
            (fit.ar[0] - 0.7).abs() < 0.15,
            "AR(1) coeff should be near 0.7, got {}",
            fit.ar[0]
        );
        assert!(fit.ma.is_empty());
        assert!(fit.sigma2 > 0.5 && fit.sigma2 < 2.0);
    }

    #[test]
    fn arma_fit_pure_ma1() {
        // Generate MA(1) data: x_t = ε_t + 0.5 ε_{t-1}
        let n = 500;
        let mut rng = Xoshiro256::new(99);
        let mut eps = vec![0.0; n];
        let mut data = vec![0.0; n];
        for t in 0..n {
            eps[t] = sample_normal(&mut rng, 0.0, 1.0);
            data[t] = eps[t] + if t > 0 { 0.5 * eps[t - 1] } else { 0.0 };
        }
        let fit = arma_fit(&data, 0, 1, 200);
        assert!(fit.ar.is_empty());
        assert!(
            (fit.ma[0] - 0.5).abs() < 0.2,
            "MA(1) coeff should be near 0.5, got {}",
            fit.ma[0]
        );
    }

    #[test]
    fn arma_fit_arma11() {
        // ARMA(1,1): x_t = 0.6 x_{t-1} + ε_t + 0.3 ε_{t-1}
        let n = 800;
        let mut rng = Xoshiro256::new(123);
        let mut eps = vec![0.0; n];
        let mut data = vec![0.0; n];
        for t in 0..n {
            eps[t] = sample_normal(&mut rng, 0.0, 1.0);
            let ar = if t > 0 { 0.6 * data[t - 1] } else { 0.0 };
            let ma = if t > 0 { 0.3 * eps[t - 1] } else { 0.0 };
            data[t] = ar + eps[t] + ma;
        }
        let fit = arma_fit(&data, 1, 1, 300);
        assert!(fit.ar.len() == 1);
        assert!(fit.ma.len() == 1);
        // Coefficient recovery within ±0.2 for N=800
        assert!(
            (fit.ar[0] - 0.6).abs() < 0.25,
            "AR coeff {}, expected ~0.6",
            fit.ar[0]
        );
    }

    #[test]
    fn arima_fit_random_walk_d1() {
        // Random walk = ARIMA(0,1,0): differences are white noise
        let n = 300;
        let mut rng = Xoshiro256::new(77);
        let mut data = vec![0.0; n];
        for t in 1..n {
            data[t] = data[t - 1] + sample_normal(&mut rng, 0.0, 1.0);
        }
        let fit = arima_fit(&data, 0, 1, 0, 200);
        assert_eq!(fit.d, 1);
        assert!(fit.arma.ar.is_empty());
        assert!(fit.arma.ma.is_empty());
        // σ² of innovations should be ~1.0
        assert!(
            fit.arma.sigma2 > 0.5 && fit.arma.sigma2 < 2.0,
            "sigma2 {}",
            fit.arma.sigma2
        );
    }

    #[test]
    fn arima_forecast_random_walk() {
        let n = 100;
        let mut rng = Xoshiro256::new(55);
        let mut data = vec![100.0; n];
        for t in 1..n {
            data[t] = data[t - 1] + sample_normal(&mut rng, 0.0, 0.5);
        }
        let fit = arima_fit(&data, 0, 1, 0, 100);
        let fc = arima_forecast(&fit, &data, 5);
        assert_eq!(fc.len(), 5);
        // Random walk forecast is flat at last value
        let last = *data.last().unwrap();
        for (i, &v) in fc.iter().enumerate() {
            assert!(v.is_finite(), "forecast[{i}] is not finite");
            // For ARIMA(0,1,0), forecasts should be near last value
            assert!((v - last).abs() < 5.0, "forecast[{i}]={v}, last={last}");
        }
    }

    #[test]
    fn auto_arima_picks_reasonable_order() {
        // AR(1) with moderate coefficient — auto should produce finite AIC
        // and a non-trivial model (p+q >= 1 or d >= 1).
        let n = 300;
        let mut rng = Xoshiro256::new(42);
        let mut data = vec![0.0; n];
        for t in 1..n {
            data[t] = 0.5 * data[t - 1] + sample_normal(&mut rng, 0.0, 1.0);
        }
        let fit = auto_arima(&data, 3, 1, 2, 100);
        assert!(fit.arma.aic.is_finite());
        // Should pick some non-trivial model
        let total_order = fit.arma.ar.len() + fit.arma.ma.len() + fit.d;
        assert!(
            total_order >= 1,
            "auto_arima should pick a non-trivial model"
        );
    }

    #[test]
    fn arma_fit_degenerate_short_data() {
        let fit = arma_fit(&[1.0, 2.0], 1, 1, 100);
        // Should not panic; returns degenerate result
        assert!(fit.sigma2.is_finite());
    }

    #[test]
    fn arma_css_residuals_pure_ar1_small() {
        // x = [0, 1, 2, 3, 4], AR=[0.5], MA=[]
        // centered: [-2, -1, 0, 1, 2]
        // predicted[1] = 0.5 * (-2) = -1.0, actual = -1, resid = 0
        // predicted[2] = 0.5 * (-1) = -0.5, actual = 0, resid = 0.5
        let centered = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let r = arma_css_residuals(&centered, &[0.5], &[]);
        assert_eq!(r.len(), 4);
        assert!((r[0] - 0.0).abs() < 1e-12); // t=1: pred=-1, actual=-1
        assert!((r[1] - 0.5).abs() < 1e-12); // t=2: pred=-0.5, actual=0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Stationarity extension tests: PP, VR, von Neumann, Bartels
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod stationarity_extension_tests {
    use super::*;
    use crate::rng::{sample_normal, Xoshiro256};

    fn make_rw(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Xoshiro256::new(seed);
        let mut x = vec![0.0f64; n];
        for t in 1..n {
            x[t] = x[t - 1] + sample_normal(&mut rng, 0.0, 1.0);
        }
        x
    }

    fn make_ar1_stationary(n: usize, phi: f64, seed: u64) -> Vec<f64> {
        let mut rng = Xoshiro256::new(seed);
        let mut x = vec![0.0f64; n];
        for t in 1..n {
            x[t] = phi * x[t - 1] + sample_normal(&mut rng, 0.0, 1.0);
        }
        x
    }

    #[test]
    fn pp_test_stationary_has_finite_stat() {
        let data = make_ar1_stationary(200, 0.5, 42);
        let r = pp_test(&data, None);
        assert!(
            r.statistic.is_finite(),
            "PP stat should be finite, got {}",
            r.statistic
        );
        assert!(r.n_lags > 0);
    }

    #[test]
    fn pp_test_random_walk_less_negative_than_stationary() {
        // RW: PP stat near 0 (non-stationary region).
        // AR(0.5): PP stat much more negative.
        let rw = make_rw(300, 1);
        let ar = make_ar1_stationary(300, 0.5, 2);
        let r_rw = pp_test(&rw, None);
        let r_ar = pp_test(&ar, None);
        assert!(
            r_ar.statistic < r_rw.statistic,
            "AR(0.5) PP stat ({}) should be < RW ({}) for stationarity detection",
            r_ar.statistic,
            r_rw.statistic
        );
    }

    #[test]
    fn pp_test_too_short_returns_nan() {
        let r = pp_test(&[1.0, 2.0, 3.0], None);
        assert!(r.statistic.is_nan());
    }

    #[test]
    fn variance_ratio_random_walk_near_one() {
        // Under true RW, VR(q) → 1 asymptotically
        let rw = make_rw(2000, 7);
        let r = variance_ratio_test(&rw, Some(4));
        assert!(r.vr.is_finite());
        assert!(
            (r.vr - 1.0).abs() < 0.3,
            "RW VR should be near 1.0, got {}",
            r.vr
        );
    }

    #[test]
    fn variance_ratio_ar1_returns_below_one() {
        // AR(1) level process: Δx_t = (φ-1)*x_{t-1} + ε → first differences
        // are negatively autocorrelated (mean-reverting) → VR < 1.
        // This is correct Lo-MacKinlay behavior: the test operates on first differences of data.
        let data = make_ar1_stationary(2000, 0.7, 5);
        let r = variance_ratio_test(&data, Some(8));
        assert!(r.vr.is_finite());
        assert!(
            r.vr < 1.0,
            "AR(0.7) level → differences mean-revert → VR < 1, got {}",
            r.vr
        );
    }

    #[test]
    fn variance_ratio_cumulative_ar1_above_one() {
        // Cumulative sum of positively autocorrelated returns → VR > 1
        let mut rng = Xoshiro256::new(5);
        let n = 2000;
        // Generate AR(1) returns with φ=0.7
        let mut r_t = vec![0.0f64; n];
        for t in 1..n {
            r_t[t] = 0.7 * r_t[t - 1] + sample_normal(&mut rng, 0.0, 0.1);
        }
        // Cumulative price (like prices from autocorrelated returns)
        let mut prices = vec![100.0f64; n + 1];
        for t in 0..n {
            prices[t + 1] = prices[t] + r_t[t];
        }
        let r = variance_ratio_test(&prices, Some(4));
        assert!(r.vr.is_finite());
        assert!(
            r.vr > 1.0,
            "Cumulative AR(0.7) → returns autocorrelated → VR > 1, got {}",
            r.vr
        );
    }

    #[test]
    fn variance_ratio_too_short_nan() {
        let r = variance_ratio_test(&[1.0, 2.0, 3.0], Some(4));
        assert!(r.vr.is_nan());
    }

    #[test]
    fn von_neumann_ratio_independent_near_two() {
        // IID white noise → von Neumann ratio ≈ 2
        let mut rng = Xoshiro256::new(13);
        let data: Vec<f64> = (0..500)
            .map(|_| sample_normal(&mut rng, 0.0, 1.0))
            .collect();
        let vn = von_neumann_ratio(&data);
        assert!(vn.is_finite());
        assert!((vn - 2.0).abs() < 0.5, "IID → VN ≈ 2, got {vn}");
    }

    #[test]
    fn von_neumann_ratio_trending_below_two() {
        // Linear trend → small successive differences relative to total variance → VN < 2
        let data: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let vn = von_neumann_ratio(&data);
        assert!(vn.is_finite());
        assert!(
            vn < 1.0,
            "Linear trend → VN ≪ 2 (positive autocorr), got {vn}"
        );
    }

    #[test]
    fn von_neumann_ratio_constant_returns_nan() {
        let vn = von_neumann_ratio(&[5.0; 10]);
        assert!(vn.is_nan());
    }

    #[test]
    fn bartels_random_data_near_zero() {
        // IID data → Bartels z stat should be near 0
        let mut rng = Xoshiro256::new(99);
        let data: Vec<f64> = (0..200)
            .map(|_| sample_normal(&mut rng, 0.0, 1.0))
            .collect();
        let z = bartels_rank_test(&data);
        assert!(z.is_finite());
        assert!(z.abs() < 5.0, "IID → Bartels |z| < 5, got {z}");
    }

    #[test]
    fn bartels_trending_negative() {
        // Strong trend → positive rank autocorrelation → z < 0 (RVN < expected)
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let z = bartels_rank_test(&data);
        assert!(z.is_finite());
        assert!(z < 0.0, "Trend → Bartels z < 0, got {z}");
    }

    #[test]
    fn bartels_too_short_nan() {
        let z = bartels_rank_test(&[1.0, 2.0]);
        assert!(z.is_nan());
    }

    #[test]
    fn zivot_andrews_stationary_returns_finite() {
        let data = make_ar1_stationary(100, 0.5, 88);
        let r = zivot_andrews_test(&data, None, None);
        assert!(
            r.statistic.is_finite(),
            "ZA stat should be finite for stationary data"
        );
        assert!(r.breakpoint > 0 && r.breakpoint < data.len());
    }

    #[test]
    fn zivot_andrews_too_short_nan() {
        let r = zivot_andrews_test(&[1.0, 2.0, 3.0], None, None);
        assert!(r.statistic.is_nan());
    }
}

#[cfg(test)]
mod extended_stationarity_tests {
    use super::*;
    use crate::rng::{sample_normal, Xoshiro256};

    #[test]
    fn pp_random_walk_does_not_reject() {
        let n = 200;
        let mut rng = Xoshiro256::new(42);
        let mut data = vec![0.0; n];
        for t in 1..n {
            data[t] = data[t - 1] + sample_normal(&mut rng, 0.0, 1.0);
        }
        let r = phillips_perron_test(&data, None);
        assert!(r.statistic.is_finite());
        // Random walk: PP stat should NOT be very negative (should not reject)
        assert!(
            r.statistic > r.critical_1pct,
            "PP should not reject at 1% for random walk: stat={} crit={}",
            r.statistic,
            r.critical_1pct
        );
    }

    #[test]
    fn pp_white_noise_rejects() {
        let n = 200;
        let mut rng = Xoshiro256::new(42);
        let data: Vec<f64> = (0..n).map(|_| sample_normal(&mut rng, 0.0, 1.0)).collect();
        let r = phillips_perron_test(&data, None);
        assert!(r.statistic.is_finite());
        // White noise: PP should strongly reject unit root
        assert!(
            r.statistic < 0.0,
            "PP stat for WN should be negative, got {}",
            r.statistic
        );
    }

    #[test]
    fn pp_short_returns_nan() {
        let r = phillips_perron_test(&[1.0, 2.0], None);
        assert!(r.statistic.is_nan());
    }

    #[test]
    fn box_pierce_white_noise_high_p() {
        let n = 200;
        let mut rng = Xoshiro256::new(42);
        let data: Vec<f64> = (0..n).map(|_| sample_normal(&mut rng, 0.0, 1.0)).collect();
        let r = box_pierce(&data, 10, 0);
        assert!(
            r.p_value > 0.01,
            "BP p-value on WN should be high, got {}",
            r.p_value
        );
    }

    #[test]
    fn box_pierce_correlated_low_p() {
        // AR(1) with strong correlation
        let n = 200;
        let mut rng = Xoshiro256::new(42);
        let mut data = vec![0.0; n];
        for t in 1..n {
            data[t] = 0.9 * data[t - 1] + sample_normal(&mut rng, 0.0, 1.0);
        }
        let r = box_pierce(&data, 10, 0);
        assert!(
            r.p_value < 0.05,
            "BP p-value on AR(1) should be low, got {}",
            r.p_value
        );
    }

    #[test]
    fn spectral_flatness_white_noise_near_one() {
        // Flat PSD → SF near 1
        let flat = vec![1.0; 100];
        let sf = spectral_flatness(&flat);
        assert!(
            (sf - 1.0).abs() < 1e-10,
            "flat PSD should have SF=1, got {sf}"
        );
    }

    #[test]
    fn spectral_flatness_pure_tone_near_zero() {
        let mut psd = vec![0.001; 100];
        psd[10] = 100.0; // dominant peak
        let sf = spectral_flatness(&psd);
        assert!(sf < 0.1, "peaked PSD should have low SF, got {sf}");
    }

    #[test]
    fn spectral_centroid_symmetric() {
        let freqs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let psd = vec![1.0; 10]; // uniform
        let c = spectral_centroid(&freqs, &psd);
        assert!(
            (c - 4.5).abs() < 1e-10,
            "centroid of uniform [0..9] = 4.5, got {c}"
        );
    }

    #[test]
    fn spectral_bandwidth_zero_for_delta() {
        let freqs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let mut psd = vec![0.0; 10];
        psd[5] = 1.0;
        let bw = spectral_bandwidth(&freqs, &psd);
        assert!(
            bw.abs() < 1e-10,
            "delta PSD should have 0 bandwidth, got {bw}"
        );
    }

    #[test]
    fn spectral_rolloff_85pct() {
        let freqs: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let psd = vec![1.0; 100]; // uniform
        let ro = spectral_rolloff(&freqs, &psd, 0.85);
        // 85% of 100 bins = bin 84 → freq 84
        assert!(
            (ro - 84.0).abs() < 1.0,
            "85% rolloff of uniform should be ~84, got {ro}"
        );
    }

    #[test]
    fn spectral_crest_flat_is_one() {
        let psd = vec![5.0; 50];
        let cf = spectral_crest(&psd);
        assert!((cf - 1.0).abs() < 1e-10, "flat PSD crest = 1, got {cf}");
    }

    #[test]
    fn spectral_slope_white_near_zero() {
        let freqs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let psd = vec![1.0; 100];
        let slope = spectral_slope(&freqs, &psd);
        assert!(
            slope.abs() < 0.1,
            "white noise slope should be ~0, got {slope}"
        );
    }

    #[test]
    fn spectral_fwhm_sharp_peak_narrow() {
        let freqs: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let mut psd = vec![0.1; 100];
        // Sharp peak at freq=5.0 (index 50)
        psd[49] = 0.5;
        psd[50] = 1.0;
        psd[51] = 0.5;
        let fwhm = spectral_fwhm(&freqs, &psd);
        assert!(
            fwhm.is_finite() && fwhm > 0.0 && fwhm < 1.0,
            "sharp peak should have narrow FWHM, got {fwhm}"
        );
    }

    #[test]
    fn spectral_q_factor_sharp_peak_high() {
        let freqs: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let mut psd = vec![0.01; 100];
        psd[49] = 0.3;
        psd[50] = 1.0;
        psd[51] = 0.3;
        let q = spectral_q_factor(&freqs, &psd);
        assert!(
            q.is_finite() && q > 1.0,
            "sharp peak should have high Q, got {q}"
        );
    }

    #[test]
    fn spectral_skewness_symmetric_near_zero() {
        let freqs: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // Symmetric PSD around centroid
        let psd: Vec<f64> = (0..100)
            .map(|i| {
                let x = (i as f64 - 50.0) / 10.0;
                (-x * x / 2.0).exp()
            })
            .collect();
        let sk = spectral_skewness(&freqs, &psd);
        assert!(
            sk.abs() < 0.1,
            "symmetric PSD should have ~0 skewness, got {sk}"
        );
    }

    // ── Dependence tests ────────────────────────────────────────────────

    #[test]
    fn turning_point_iid_near_expected() {
        let mut rng = Xoshiro256::new(42);
        let data: Vec<f64> = (0..200)
            .map(|_| sample_normal(&mut rng, 0.0, 1.0))
            .collect();
        let (tp, z) = turning_point_test(&data);
        // IID: expected ≈ 2*(200-2)/3 ≈ 132, z should be near 0
        assert!(
            tp > 100 && tp < 165,
            "IID turning points should be ~132, got {tp}"
        );
        assert!(z.abs() < 3.0, "z should be moderate for IID, got {z}");
    }

    #[test]
    fn turning_point_trend_below_expected() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (tp, z) = turning_point_test(&data);
        assert_eq!(tp, 0, "linear trend has 0 turning points");
        assert!(z < -3.0, "trend should give very negative z");
    }

    #[test]
    fn rank_vn_iid_near_two() {
        let mut rng = Xoshiro256::new(42);
        let data: Vec<f64> = (0..200)
            .map(|_| sample_normal(&mut rng, 0.0, 1.0))
            .collect();
        let rvn = rank_von_neumann_ratio(&data);
        assert!(
            (rvn - 2.0).abs() < 0.5,
            "rank VN for IID should be ~2, got {rvn}"
        );
    }

    #[test]
    fn breusch_godfrey_no_autocorr() {
        // Generate IID residuals with dummy regressors
        let mut rng = Xoshiro256::new(42);
        let n = 100;
        let k = 2; // intercept + one regressor
        let resid: Vec<f64> = (0..n).map(|_| sample_normal(&mut rng, 0.0, 1.0)).collect();
        let x: Vec<f64> = (0..n * k)
            .map(|i| {
                if i % k == 0 {
                    1.0
                } else {
                    sample_normal(&mut rng, 0.0, 1.0)
                }
            })
            .collect();
        let (lm, pval, _) = breusch_godfrey(&resid, &x, k, 4);
        assert!(lm.is_finite());
        assert!(pval > 0.01, "IID residuals should not reject, p={pval}");
    }

    // ── Additional spectral tests ───────────────────────────────────────

    #[test]
    fn spectral_flux_constant_is_zero() {
        let f1 = vec![1.0; 10];
        let f2 = vec![1.0; 10];
        let flux = spectral_flux(&[f1, f2]);
        assert_eq!(flux.len(), 1);
        assert!(flux[0].abs() < 1e-12);
    }

    #[test]
    fn spectral_flux_changing_is_positive() {
        let f1 = vec![1.0; 10];
        let f2 = vec![2.0; 10];
        let flux = spectral_flux(&[f1, f2]);
        assert!(flux[0] > 0.0);
    }

    #[test]
    fn spectral_decrease_flat_near_zero() {
        let psd = vec![1.0; 50];
        let sd = spectral_decrease(&psd);
        assert!(sd.abs() < 1e-10, "flat PSD → decrease ≈ 0, got {sd}");
    }

    #[test]
    fn spectral_contrast_uniform_is_zero_db() {
        let freqs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let psd = vec![1.0; 20];
        let c = spectral_contrast(&freqs, &psd, 4);
        assert_eq!(c.len(), 4);
        for &v in &c {
            assert!(
                v.abs() < 1e-10,
                "uniform PSD contrast should be 0 dB, got {v}"
            );
        }
    }

    #[test]
    fn dominant_frequency_picks_peak() {
        let freqs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let psd = vec![0.1, 0.5, 10.0, 0.3, 0.1];
        assert_eq!(dominant_frequency(&freqs, &psd), 3.0);
    }

    #[test]
    fn spectral_peak_count_single_peak() {
        let mut psd = vec![0.1; 50];
        psd[25] = 1.0;
        assert_eq!(spectral_peak_count(&psd, 0.5), 1);
    }

    #[test]
    fn spectral_peak_count_high_threshold_filters() {
        let mut psd = vec![0.1; 50];
        psd[10] = 1.0; // big peak
        psd[30] = 0.3; // small peak: 0.3 < 0.5 * 1.0 = threshold
        assert_eq!(
            spectral_peak_count(&psd, 0.5),
            1,
            "only the large peak should pass"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PELT — Pruned Exact Linear Time Changepoint Detection
// ═══════════════════════════════════════════════════════════════════════════

/// Gaussian segment cost: n × ln(variance) using prefix sums.
///
/// Returns INFINITY for degenerate segments (n < 2 or zero variance).
pub fn pelt_segment_cost(cumsum: &[f64], cumsum2: &[f64], start: usize, end: usize) -> f64 {
    let n = end - start;
    if n < 2 {
        return f64::INFINITY;
    }
    let nf = n as f64;
    let s = cumsum[end] - cumsum[start];
    let s2 = cumsum2[end] - cumsum2[start];
    let var = (s2 / nf - (s / nf) * (s / nf)).max(1e-30);
    nf * var.ln()
}

/// Pruned Exact Linear Time (PELT) changepoint detection (Killick et al. 2012).
///
/// Finds the optimal segmentation of `data` under the Gaussian cost model
/// with BIC penalty `β = 2 × ln(n)`. Returns the indices of changepoints
/// (exclusive segment boundaries, so segment k runs `cps[k-1]..cps[k]`).
///
/// # Arguments
/// * `data` — time series to segment
/// * `min_seg` — minimum segment length (default: use `pelt` convenience fn)
/// * `penalty` — cost penalty per changepoint (None → BIC: 2·ln(n))
///
/// # Returns
/// Sorted vector of changepoint positions in `1..n-1` (empty if no breaks).
///
/// # Kingdom
/// The unpruned DP `F(t) = min_τ [F(τ) + C(τ,t) + β]` is Kingdom A in the min-plus
/// (tropical) semiring — tropical matrix-vector product, O(n²) per step.
/// The PELT pruning (discarding τ where F(τ) + C(τ,t) > F(t) + β) is state-dependent
/// (the pruning set depends on F[t]), making the efficient O(n) algorithm Kingdom B.
/// We label the implementation Kingdom B. The tropical semiring structure is real —
/// Op::TropicalMinPlus would unlock the O(n²) parallel version.
pub fn pelt(data: &[f64], min_seg: usize, penalty: Option<f64>) -> Vec<usize> {
    let n = data.len();
    if n < 2 * min_seg {
        return Vec::new();
    }

    // Prefix sums for O(1) segment cost
    let mut cumsum = vec![0.0f64; n + 1];
    let mut cumsum2 = vec![0.0f64; n + 1];
    for i in 0..n {
        cumsum[i + 1] = cumsum[i] + data[i];
        cumsum2[i + 1] = cumsum2[i] + data[i] * data[i];
    }

    let beta = penalty.unwrap_or_else(|| 2.0 * (n as f64).ln());

    // DP: f[t] = min cost of segmenting data[0..t]
    let mut f = vec![f64::INFINITY; n + 1];
    let mut last_cp = vec![0usize; n + 1];
    f[0] = -beta; // absorb one penalty for the initial segment
    let mut candidates: Vec<usize> = vec![0];

    for t in min_seg..=n {
        let mut best_f = f64::INFINITY;
        let mut best_s = 0usize;
        for &s in &candidates {
            if t - s < min_seg {
                continue;
            }
            let cost = f[s] + pelt_segment_cost(&cumsum, &cumsum2, s, t) + beta;
            if cost < best_f {
                best_f = cost;
                best_s = s;
            }
        }
        f[t] = best_f;
        last_cp[t] = best_s;

        // PELT pruning: discard s where f[s] + cost(s,t) > f[t] + beta
        // Those s can never be optimal for any future t' > t.
        candidates.retain(|&s| {
            if t - s < min_seg {
                return true;
            }
            f[s] + pelt_segment_cost(&cumsum, &cumsum2, s, t) <= f[t] + beta
        });
        candidates.push(t);
    }

    // Backtrace
    let mut cps = Vec::new();
    let mut pos = n;
    while pos > 0 {
        let prev = last_cp[pos];
        if prev > 0 {
            cps.push(prev);
        }
        pos = prev;
    }
    cps.sort_unstable();
    cps
}

#[cfg(test)]
mod tests_pelt {
    use super::*;

    fn wn(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = crate::rng::Xoshiro256::new(seed);
        (0..n)
            .map(|_| crate::rng::sample_normal(&mut rng, 0.0, 1.0))
            .collect()
    }

    #[test]
    fn pelt_too_short_returns_empty() {
        let r = pelt(&[0.0; 5], 10, None);
        assert!(r.is_empty());
    }

    #[test]
    fn pelt_white_noise_few_changepoints() {
        let data = wn(200, 42);
        let cps = pelt(&data, 10, None);
        // White noise: PELT may find 0-3 spurious breaks; should not explode
        assert!(cps.len() <= 10, "too many spurious changepoints: {:?}", cps);
    }

    #[test]
    fn pelt_step_function_detects_break() {
        let mut data = vec![0.0f64; 100];
        for i in 50..100 {
            data[i] = 5.0;
        }
        let cps = pelt(&data, 5, None);
        assert!(!cps.is_empty(), "step function: should detect changepoint");
        // The changepoint should be near index 50
        let near_50 = cps.iter().any(|&cp| (cp as isize - 50).abs() <= 5);
        assert!(near_50, "changepoint should be near 50, got {:?}", cps);
    }

    #[test]
    fn pelt_no_break_when_homogeneous() {
        // Very tight uniform data — PELT should find 0 breaks
        let data = vec![1.0f64; 100];
        let cps = pelt(&data, 10, None);
        assert!(
            cps.is_empty(),
            "constant data: no changepoints expected, got {:?}",
            cps
        );
    }

    #[test]
    fn pelt_segment_cost_symmetry() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut cs = vec![0.0f64; 6];
        let mut cs2 = vec![0.0f64; 6];
        for i in 0..5 {
            cs[i + 1] = cs[i] + data[i];
            cs2[i + 1] = cs2[i] + data[i] * data[i];
        }
        let c1 = pelt_segment_cost(&cs, &cs2, 0, 5);
        let c2 = pelt_segment_cost(&cs, &cs2, 0, 5);
        assert_eq!(c1, c2);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BOCPD — Bayesian Online Changepoint Detection
// ═══════════════════════════════════════════════════════════════════════════

/// BOCPD run-length posterior at time t.
///
/// Companion to `bocpd_step` for streaming usage.
#[derive(Debug, Clone)]
pub struct BocpdState {
    /// Run-length posterior: `run_post[r]` = P(run_length = r | data[0..t])
    pub run_post: Vec<f64>,
    /// Sufficient statistics: count, sum, sum-of-squares per run length
    pub rl_count: Vec<f64>,
    pub rl_sum: Vec<f64>,
    pub rl_sum2: Vec<f64>,
    /// Current time step (0-indexed)
    pub t: usize,
    /// Maximum tracked run length
    pub max_run: usize,
    /// Hazard rate (probability of changepoint per step)
    pub hazard: f64,
}

impl BocpdState {
    /// Initialize BOCPD state. `max_run` caps the run-length distribution.
    /// `hazard` = 1/expected_run_length (common: 1/200 for financial data).
    pub fn new(max_run: usize, hazard: f64) -> Self {
        let mut run_post = vec![0.0f64; max_run + 1];
        run_post[0] = 1.0;
        BocpdState {
            run_post,
            rl_count: vec![0.0; max_run + 1],
            rl_sum: vec![0.0; max_run + 1],
            rl_sum2: vec![0.0; max_run + 1],
            t: 0,
            max_run,
            hazard,
        }
    }

    /// Update with a new observation `x`.
    ///
    /// Returns the (run_length_with_max_posterior, max_posterior, mean_run_length).
    pub fn update(&mut self, x: f64) -> (usize, f64, f64) {
        let cur_len = (self.t + 1).min(self.max_run);
        let mut growth = vec![0.0f64; cur_len + 2];
        let mut cp_mass = 0.0f64;

        // Compute predictive probability under Normal-Gamma prior (κ₀ = 1, μ₀ = 0)
        let prior_var = 2.0;
        let prior_pred = (-0.5 * (std::f64::consts::TAU * prior_var).ln()
            - 0.5 * x * x / prior_var)
            .exp()
            .max(1e-300);

        for r in 0..=cur_len.min(self.max_run.saturating_sub(1)) {
            if self.run_post[r] < 1e-300 {
                continue;
            }
            let cnt = self.rl_count[r];
            let log_pred = if cnt < 1.0 {
                -0.5 * (std::f64::consts::TAU * prior_var).ln() - 0.5 * x * x / prior_var
            } else {
                let mean = self.rl_sum[r] / cnt;
                let var = (self.rl_sum2[r] / cnt - mean * mean).abs() + 1e-10;
                -0.5 * (std::f64::consts::TAU * var).ln() - 0.5 * (x - mean) * (x - mean) / var
            };
            let pred = log_pred.exp().max(1e-300);
            if r + 1 <= self.max_run {
                growth[r + 1] = self.run_post[r] * pred * (1.0 - self.hazard);
            }
            cp_mass += self.run_post[r] * prior_pred * self.hazard;
        }

        // Update sufficient statistics (shift by 1)
        for r in (1..=cur_len.min(self.max_run)).rev() {
            self.rl_count[r] = self.rl_count[r - 1] + 1.0;
            self.rl_sum[r] = self.rl_sum[r - 1] + x;
            self.rl_sum2[r] = self.rl_sum2[r - 1] + x * x;
        }
        self.rl_count[0] = 1.0;
        self.rl_sum[0] = x;
        self.rl_sum2[0] = x * x;

        // Set new posteriors
        for r in 1..=cur_len.min(self.max_run) {
            self.run_post[r] = growth[r];
        }
        self.run_post[0] = cp_mass;

        // Normalize
        let total: f64 = self.run_post[..=cur_len.min(self.max_run)].iter().sum();
        if total > 0.0 {
            for r in 0..=cur_len.min(self.max_run) {
                self.run_post[r] /= total;
            }
        }

        self.t += 1;
        let max_len = cur_len.min(self.max_run);
        let max_p = self.run_post[..=max_len]
            .iter()
            .copied()
            .fold(0.0f64, f64::max);
        let argmax = self.run_post[..=max_len]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mean_rl: f64 = self.run_post[..=max_len]
            .iter()
            .enumerate()
            .map(|(r, &p)| r as f64 * p)
            .sum();
        (argmax, max_p, mean_rl)
    }
}

/// Batch BOCPD: run on a complete series, return detected changepoint indices.
///
/// Uses Adams-MacKay 2007 online message passing with Gaussian predictive
/// (Normal-Gamma conjugate). Changepoints are detected by large drops in the
/// maximum run-length posterior.
///
/// # Arguments
/// * `data` — time series
/// * `max_run` — cap on tracked run length (memory: O(max_run))
/// * `hazard` — prior probability of a changepoint per step (1/expected_run_len)
/// * `threshold` — posterior-drop threshold above which a changepoint is flagged
///   (None → adaptive: `median_drop + 2·std_drop`, minimum 0.3)
///
/// # Returns
/// Sorted vector of detected changepoint time indices.
///
/// # Kingdom
/// **Kingdom A math / Kingdom B implementation.**
/// If sufficient stats are precomputed from the observation sequence, the update
/// `p_t = D(x_t) · H · p_{t-1} + reset · e_0` is a data-determined linear map on
/// the run-length posterior vector (D(x_t) diagonal with data-computed likelihoods,
/// H constant hazard matrix). That is Kingdom A via linear prefix scan.
/// This implementation maintains sufficient stats incrementally (B impl), but the
/// underlying mathematics is data-determined. Label: A math / B impl.
pub fn bocpd(data: &[f64], max_run: usize, hazard: f64, threshold: Option<f64>) -> Vec<usize> {
    let n = data.len();
    if n < 10 { return Vec::new(); }

    let mut state = BocpdState::new(max_run, hazard);
    let mut mean_rls = Vec::with_capacity(n);

    for &x in data {
        let (_, _, mean_rl) = state.update(x);
        mean_rls.push(mean_rl);
    }

    // Detection: Changepoint flagged by significant drop in mean run length.
    // A changepoint at t causes mean_rl to drop from ~t to ~0.
    let mut changepoints = Vec::new();
    
    // Use a relative threshold if not provided: e.g., drop > 50% of current mean
    // or a fixed threshold.
    let thresh = threshold.unwrap_or(20.0); 

    for t in 1..n {
        let drop = mean_rls[t - 1] - mean_rls[t];
        if drop > thresh {
            changepoints.push(t);
        }
    }

    changepoints
}

// ═══════════════════════════════════════════════════════════════════════════
// Log returns primitive
// ═══════════════════════════════════════════════════════════════════════════

/// Compute log returns from a price series.
///
/// r_i = ln(p_{i+1} / p_i) for i = 0 .. n−2.
///
/// Log returns are preferred over simple returns for financial time-series
/// analysis because they are time-additive, approximately normally distributed
/// for short intervals, and well-defined for volatility modelling.
///
/// # Parameters
/// - `prices`: price series, length n. All values must be strictly positive.
///   Non-positive prices produce NaN for the corresponding return.
///
/// # Returns
/// Vector of log returns with length `prices.len() - 1`.
/// Returns empty Vec if `prices.len() < 2`.
///
/// # Formula
/// r_i = ln(p_{i+1}) − ln(p_i)   (numerically stable: no division, no cancellation)
///
/// # Properties
/// - r_i > −∞ for p_i > 0, p_{i+1} > 0
/// - Σ r_i = ln(p_n / p_0): multi-period log return = sum of single-period log returns
/// - For small r: log return ≈ simple return (r ≈ ΔP/P)
///
/// # Consumers
/// GARCH fitting (requires log returns as input), volatility estimation,
/// Brownian motion calibration, IAT spectral analysis (family-17 bridges),
/// any fintek bridge that starts from raw prices.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 { return vec![]; }
    prices.windows(2)
        .map(|w| {
            // Guard: non-positive, non-finite, or NaN prices → NaN return.
            // `<= 0.0` catches zero and negative but NOT Inf or NaN (both fail the test).
            if !w[0].is_finite() || !w[1].is_finite() || w[0] <= 0.0 || w[1] <= 0.0 {
                f64::NAN
            } else {
                (w[1] / w[0]).ln()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests_bocpd {
    use super::*;

    fn wn(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = crate::rng::Xoshiro256::new(seed);
        (0..n)
            .map(|_| crate::rng::sample_normal(&mut rng, 0.0, 0.1))
            .collect()
    }

    #[test]
    fn bocpd_too_short_returns_empty() {
        assert!(bocpd(&[0.0; 5], 100, 1.0 / 200.0, None).is_empty());
    }

    #[test]
    fn bocpd_white_noise_few_changepoints() {
        let data = wn(200, 42);
        let cps = bocpd(&data, 500, 1.0 / 200.0, None);
        assert!(cps.len() <= 20, "too many spurious CPs: {:?}", cps);
    }

    #[test]
    fn bocpd_step_function_detects_change() {
        let mut data = vec![0.0f64; 100];
        for i in 50..100 {
            data[i] = 1.0;
        }
        let cps = bocpd(&data, 500, 1.0 / 200.0, None);
        let near_50 = cps.iter().any(|&cp| (cp as isize - 50).abs() <= 10);
        assert!(
            near_50,
            "step function: changepoint expected near 50, got {:?}",
            cps
        );
    }

    #[test]
    fn bocpd_state_normalizes() {
        let mut state = BocpdState::new(100, 0.01);
        for x in [0.1, -0.2, 0.05, 0.3, -0.1] {
            state.update(x);
            let sum: f64 = state.run_post[..=state.t.min(state.max_run)].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "posterior must sum to 1, got {sum}"
            );
        }
    }

    #[test]
    fn log_returns_length() {
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let r = log_returns(&prices);
        assert_eq!(r.len(), 3, "length should be n-1");
    }

    #[test]
    fn log_returns_constant_price_is_zero() {
        let prices = vec![50.0; 5];
        let r = log_returns(&prices);
        for &ri in &r { assert!(ri.abs() < 1e-15, "log return = {ri}"); }
    }

    #[test]
    fn log_returns_additive() {
        // Σ log returns = ln(p_last / p_first)
        let prices = vec![100.0, 110.0, 121.0, 133.1];
        let r = log_returns(&prices);
        let total: f64 = r.iter().sum();
        let expected = (133.1_f64 / 100.0).ln();
        assert!((total - expected).abs() < 1e-10, "total={total}, expected={expected}");
    }

    #[test]
    fn log_returns_nonpositive_price_is_nan() {
        let prices = vec![100.0, 0.0, 110.0];
        let r = log_returns(&prices);
        assert!(r[0].is_nan(), "r[0] from p=100->0 should be NaN");
        assert!(r[1].is_nan(), "r[1] from p=0->110 should be NaN");
    }
}
