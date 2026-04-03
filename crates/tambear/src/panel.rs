//! # Family 12 — Panel Data & Econometrics
//!
//! FE, RE, FD, two-way FE, Hausman, 2SLS, DiD.
//!
//! ## Architecture
//!
//! FE = demeaned OLS (F06 per-group centering + F10 regression).
//! RE = LME random intercept (F11).
//! FD = Affine lag scan + OLS.
//! Hausman = χ² test (F07).
//! 2SLS = two-stage GramMatrix solve (F10 × 2).

use crate::linear_algebra::{Mat, lstsq, inv, pinv};
use crate::mixed_effects::lme_random_intercept;

// ═══════════════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════════════

/// Fixed effects panel regression result.
#[derive(Debug, Clone)]
pub struct FeResult {
    /// Slope coefficients (excludes intercept — absorbed by demeaning).
    pub beta: Vec<f64>,
    /// Clustered standard errors (one per coefficient).
    pub se_clustered: Vec<f64>,
    /// Within R² (on demeaned data).
    pub r2_within: f64,
    /// Degrees of freedom: NT - N - K.
    pub df: usize,
}

/// First-difference panel regression result.
#[derive(Debug, Clone)]
pub struct FdResult {
    /// Slope coefficients on differenced data.
    pub beta: Vec<f64>,
    /// R² on differenced data.
    pub r2: f64,
}

/// Hausman test result (FE vs RE).
#[derive(Debug, Clone)]
pub struct HausmanResult {
    /// Test statistic (χ²).
    pub chi2: f64,
    /// Degrees of freedom (number of coefficients compared).
    pub df: usize,
    /// p-value.
    pub p_value: f64,
}

/// Two-stage least squares result.
#[derive(Debug, Clone)]
pub struct TwoSlsResult {
    /// Second-stage coefficients.
    pub beta: Vec<f64>,
    /// First-stage F-statistic (weak instrument diagnostic).
    pub first_stage_f: f64,
}

/// Difference-in-differences result.
#[derive(Debug, Clone)]
pub struct DidResult {
    /// Average treatment effect on the treated.
    pub att: f64,
    /// Standard error of ATT.
    pub se: f64,
    /// t-statistic.
    pub t_stat: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Fixed Effects — within estimator
// ═══════════════════════════════════════════════════════════════════════════

/// Fixed effects panel regression (within estimator).
///
/// `x`: row-major n×d covariate matrix.
/// `y`: response (length n).
/// `units`: unit index for each observation (0-indexed).
///
/// Returns demeaned OLS coefficients with clustered standard errors.
pub fn panel_fe(x: &[f64], y: &[f64], n: usize, d: usize, units: &[usize]) -> FeResult {
    assert_eq!(x.len(), n * d);
    assert_eq!(y.len(), n);
    assert_eq!(units.len(), n);

    let n_units = *units.iter().max().unwrap_or(&0) + 1;

    // Compute per-unit means (F06 centering)
    let mut y_sums = vec![0.0; n_units];
    let mut x_sums = vec![0.0; n_units * d];
    let mut counts = vec![0usize; n_units];
    for i in 0..n {
        let u = units[i];
        counts[u] += 1;
        y_sums[u] += y[i];
        for j in 0..d { x_sums[u * d + j] += x[i * d + j]; }
    }

    // Demean
    let mut y_dm = vec![0.0; n];
    let mut x_dm = vec![0.0; n * d];
    for i in 0..n {
        let u = units[i];
        let c = counts[u] as f64;
        y_dm[i] = y[i] - y_sums[u] / c;
        for j in 0..d {
            x_dm[i * d + j] = x[i * d + j] - x_sums[u * d + j] / c;
        }
    }

    // OLS on demeaned data: β = (X̃'X̃)⁻¹ X̃'ỹ
    let x_mat = Mat::from_vec(n, d, x_dm.clone());
    let beta = lstsq(&x_mat, &y_dm);

    // Residuals + R²
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let y_dm_mean = y_dm.iter().sum::<f64>() / n as f64;
    let mut residuals = vec![0.0; n];
    for i in 0..n {
        let mut fitted = 0.0;
        for j in 0..d { fitted += beta[j] * x_dm[i * d + j]; }
        residuals[i] = y_dm[i] - fitted;
        ss_res += residuals[i] * residuals[i];
        let dev = y_dm[i] - y_dm_mean;
        ss_tot += dev * dev;
    }
    let r2_within = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

    // Clustered standard errors (sandwich: (X̃'X̃)⁻¹ B (X̃'X̃)⁻¹)
    // B = Σ_u (X̃_u' û_u)(X̃_u' û_u)'
    let mut xtx = vec![0.0; d * d];
    for i in 0..n {
        for j in 0..d {
            for k in 0..d {
                xtx[j * d + k] += x_dm[i * d + j] * x_dm[i * d + k];
            }
        }
    }
    let xtx_mat = Mat::from_vec(d, d, xtx);
    let xtx_inv = match inv(&xtx_mat) {
        Some(m) => m,
        None => {
            // Singular X'X: return NaN SEs instead of garbage
            let df = n.saturating_sub(n_units).saturating_sub(d);
            return FeResult {
                beta, se_clustered: vec![f64::NAN; d], r2_within, df,
            };
        }
    };

    // Meat of the sandwich
    let mut meat = vec![0.0; d * d];
    for u in 0..n_units {
        let mut xu_resid = vec![0.0; d]; // X̃_u' û_u
        for i in 0..n {
            if units[i] == u {
                for j in 0..d {
                    xu_resid[j] += x_dm[i * d + j] * residuals[i];
                }
            }
        }
        for j in 0..d {
            for k in 0..d {
                meat[j * d + k] += xu_resid[j] * xu_resid[k];
            }
        }
    }

    // Small-sample correction: N/(N-1) * (NT-1)/(NT-K)
    // Guard: with 1 cluster, N/(N-1) = 1/0 → Inf. Clustered SEs are
    // undefined for a single cluster; fall back to uncorrected sandwich.
    let nt = n as f64;
    let nu = n_units as f64;
    let correction = if n_units <= 1 {
        1.0
    } else {
        (nu / (nu - 1.0)) * ((nt - 1.0) / (nt - d as f64))
    };

    // V = (X'X)⁻¹ M (X'X)⁻¹ * correction
    let meat_mat = Mat::from_vec(d, d, meat);
    let mut vcov = vec![0.0; d * d];
    for j in 0..d {
        for k in 0..d {
            for l in 0..d {
                let im = xtx_inv.data[j * d + l]; // (X'X)⁻¹[j,l]
                for m in 0..d {
                    vcov[j * d + k] += im * meat_mat.data[l * d + m] * xtx_inv.data[m * d + k];
                }
            }
        }
    }

    let se_clustered: Vec<f64> = (0..d).map(|j| (vcov[j * d + j] * correction).abs().sqrt()).collect();
    let df = n.saturating_sub(n_units).saturating_sub(d);

    FeResult { beta, se_clustered, r2_within, df }
}

// ═══════════════════════════════════════════════════════════════════════════
// Random Effects — delegates to F11 LME
// ═══════════════════════════════════════════════════════════════════════════

/// Random effects panel regression. Thin wrapper around F11 `lme_random_intercept`.
pub fn panel_re(
    x: &[f64], y: &[f64], n: usize, d: usize,
    units: &[usize], max_iter: usize, tol: f64,
) -> crate::mixed_effects::LmeResult {
    lme_random_intercept(x, y, n, d, units, max_iter, tol)
}

// ═══════════════════════════════════════════════════════════════════════════
// First-Difference estimator (Kingdom B — affine lag scan)
// ═══════════════════════════════════════════════════════════════════════════

/// First-difference panel estimator.
///
/// Assumes data is sorted by (unit, time). Differences within each unit.
pub fn panel_fd(x: &[f64], y: &[f64], n: usize, d: usize, units: &[usize]) -> FdResult {
    assert_eq!(x.len(), n * d);
    assert_eq!(y.len(), n);
    assert_eq!(units.len(), n);

    // First-difference within each unit
    let mut dy = Vec::new();
    let mut dx = Vec::new();
    for i in 1..n {
        if units[i] == units[i - 1] {
            dy.push(y[i] - y[i - 1]);
            for j in 0..d {
                dx.push(x[i * d + j] - x[(i - 1) * d + j]);
            }
        }
    }

    let n_diff = dy.len();
    if n_diff <= d {
        return FdResult { beta: vec![0.0; d], r2: 0.0 };
    }

    let dx_mat = Mat::from_vec(n_diff, d, dx);
    let beta = lstsq(&dx_mat, &dy);

    let dy_mean = dy.iter().sum::<f64>() / n_diff as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..n_diff {
        let mut fitted = 0.0;
        for j in 0..d { fitted += beta[j] * dx_mat.data[i * d + j]; }
        ss_res += (dy[i] - fitted).powi(2);
        ss_tot += (dy[i] - dy_mean).powi(2);
    }
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

    FdResult { beta, r2 }
}

// ═══════════════════════════════════════════════════════════════════════════
// Two-Way Fixed Effects (double demeaning)
// ═══════════════════════════════════════════════════════════════════════════

/// Two-way FE: demean by both unit and time period, add back grand mean.
pub fn panel_twfe(
    x: &[f64], y: &[f64], n: usize, d: usize,
    units: &[usize], times: &[usize],
) -> FeResult {
    let n_units = *units.iter().max().unwrap_or(&0) + 1;
    let n_times = *times.iter().max().unwrap_or(&0) + 1;

    // Unit means
    let mut y_u = vec![0.0; n_units];
    let mut x_u = vec![0.0; n_units * d];
    let mut c_u = vec![0usize; n_units];
    // Time means
    let mut y_t = vec![0.0; n_times];
    let mut x_t = vec![0.0; n_times * d];
    let mut c_t = vec![0usize; n_times];
    // Grand mean
    let mut y_grand = 0.0;
    let mut x_grand = vec![0.0; d];

    for i in 0..n {
        let u = units[i];
        let t = times[i];
        c_u[u] += 1;
        c_t[t] += 1;
        y_u[u] += y[i];
        y_t[t] += y[i];
        y_grand += y[i];
        for j in 0..d {
            x_u[u * d + j] += x[i * d + j];
            x_t[t * d + j] += x[i * d + j];
            x_grand[j] += x[i * d + j];
        }
    }
    let nf = n as f64;
    y_grand /= nf;
    for j in 0..d { x_grand[j] /= nf; }

    // Double-demean: ỹ = y - ȳ_u - ȳ_t + ȳ
    let mut y_dm = vec![0.0; n];
    let mut x_dm = vec![0.0; n * d];
    for i in 0..n {
        let u = units[i];
        let t = times[i];
        let cu = c_u[u] as f64;
        let ct = c_t[t] as f64;
        y_dm[i] = y[i] - y_u[u] / cu - y_t[t] / ct + y_grand;
        for j in 0..d {
            x_dm[i * d + j] = x[i * d + j] - x_u[u * d + j] / cu - x_t[t * d + j] / ct + x_grand[j];
        }
    }

    // OLS on double-demeaned data, then cluster SE by unit
    panel_fe_from_demeaned(&x_dm, &y_dm, n, d, units, n_units + n_times)
}

/// Internal: FE from pre-demeaned data with clustered SE.
fn panel_fe_from_demeaned(
    x_dm: &[f64], y_dm: &[f64], n: usize, d: usize,
    units: &[usize], absorbed_df: usize,
) -> FeResult {
    let n_units = *units.iter().max().unwrap_or(&0) + 1;

    let x_mat = Mat::from_vec(n, d, x_dm.to_vec());
    let beta = lstsq(&x_mat, y_dm);

    let y_dm_mean = y_dm.iter().sum::<f64>() / n as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let mut residuals = vec![0.0; n];
    for i in 0..n {
        let mut fitted = 0.0;
        for j in 0..d { fitted += beta[j] * x_dm[i * d + j]; }
        residuals[i] = y_dm[i] - fitted;
        ss_res += residuals[i] * residuals[i];
        ss_tot += (y_dm[i] - y_dm_mean).powi(2);
    }
    let r2_within = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

    // Clustered SE
    let mut xtx = vec![0.0; d * d];
    for i in 0..n {
        for j in 0..d {
            for k in 0..d {
                xtx[j * d + k] += x_dm[i * d + j] * x_dm[i * d + k];
            }
        }
    }
    let xtx_inv = inv(&Mat::from_vec(d, d, xtx))
        .unwrap_or_else(|| Mat::from_vec(d, d, vec![1.0; d * d]));

    let mut meat = vec![0.0; d * d];
    for u in 0..n_units {
        let mut xu_r = vec![0.0; d];
        for i in 0..n {
            if units[i] == u {
                for j in 0..d { xu_r[j] += x_dm[i * d + j] * residuals[i]; }
            }
        }
        for j in 0..d {
            for k in 0..d { meat[j * d + k] += xu_r[j] * xu_r[k]; }
        }
    }

    let nu = n_units as f64;
    let nt = n as f64;
    let correction = if n_units <= 1 {
        1.0
    } else {
        (nu / (nu - 1.0)) * ((nt - 1.0) / (nt - d as f64))
    };
    let meat_mat = Mat::from_vec(d, d, meat);
    let mut vcov = vec![0.0; d * d];
    for j in 0..d {
        for k in 0..d {
            for l in 0..d {
                let im = xtx_inv.data[j * d + l];
                for m in 0..d {
                    vcov[j * d + k] += im * meat_mat.data[l * d + m] * xtx_inv.data[m * d + k];
                }
            }
        }
    }

    let se_clustered: Vec<f64> = (0..d).map(|j| (vcov[j * d + j] * correction).abs().sqrt()).collect();
    let df = n.saturating_sub(absorbed_df).saturating_sub(d);

    FeResult { beta, se_clustered, r2_within, df }
}

// ═══════════════════════════════════════════════════════════════════════════
// Hausman Test (FE vs RE)
// ═══════════════════════════════════════════════════════════════════════════

/// Hausman specification test: H₀ = RE is consistent.
///
/// Compares FE and RE slope coefficients (excluding intercept).
pub fn hausman_test(fe: &FeResult, re_beta_slopes: &[f64], re_se_slopes: &[f64]) -> HausmanResult {
    let d = fe.beta.len();
    assert_eq!(re_beta_slopes.len(), d);
    assert_eq!(re_se_slopes.len(), d);

    // H = (β_FE - β_RE)' [V_FE - V_RE]⁻¹ (β_FE - β_RE)
    // Simplified: diagonal-only approximation
    let mut chi2 = 0.0;
    let mut any_negative = false;
    for j in 0..d {
        let diff = fe.beta[j] - re_beta_slopes[j];
        let v_diff = fe.se_clustered[j].powi(2) - re_se_slopes[j].powi(2);
        if v_diff > 0.0 {
            chi2 += diff * diff / v_diff;
        } else {
            any_negative = true;
        }
    }
    // Negative v_diff means RE variance exceeds FE — test assumptions violated
    let p_value = if any_negative {
        f64::NAN
    } else {
        crate::special_functions::chi2_sf(chi2, d as f64)
    };

    HausmanResult { chi2, df: d, p_value }
}

/// Full-matrix Hausman test using pseudoinverse (matches plm::phtest).
///
/// `v_fe`, `v_re`: d×d covariance matrices (row-major).
/// Uses Moore-Penrose pseudoinverse of (V_FE - V_RE), which handles
/// the common case where the difference matrix is not positive definite.
pub fn hausman_test_full(
    fe_beta: &[f64], re_beta: &[f64],
    v_fe: &[f64], v_re: &[f64],
    d: usize,
) -> HausmanResult {
    assert_eq!(fe_beta.len(), d);
    assert_eq!(re_beta.len(), d);
    assert_eq!(v_fe.len(), d * d);
    assert_eq!(v_re.len(), d * d);

    let diff: Vec<f64> = (0..d).map(|j| fe_beta[j] - re_beta[j]).collect();
    let v_diff: Vec<f64> = (0..d * d).map(|i| v_fe[i] - v_re[i]).collect();

    let v_diff_mat = Mat::from_vec(d, d, v_diff);
    let v_inv = pinv(&v_diff_mat);

    // χ² = diff' × V⁺ × diff
    let mut chi2 = 0.0;
    for j in 0..d {
        let mut v_inv_diff_j = 0.0;
        for k in 0..d { v_inv_diff_j += v_inv.data[j * d + k] * diff[k]; }
        chi2 += diff[j] * v_inv_diff_j;
    }
    chi2 = chi2.max(0.0); // numerical guard

    let p_value = crate::special_functions::chi2_sf(chi2, d as f64);
    HausmanResult { chi2, df: d, p_value }
}

// ═══════════════════════════════════════════════════════════════════════════
// Breusch-Pagan LM test for random effects
// ═══════════════════════════════════════════════════════════════════════════

/// Breusch-Pagan Lagrange Multiplier test for random effects.
///
/// H₀: σ²_α = 0 (no random effects, pooled OLS is adequate).
/// LM = [nT / (2(T-1))] × (A/B - 1)²  ~  χ²(1)
///
/// `residuals`: OLS residuals from pooled regression (length n).
/// `units`: unit index for each observation.
/// `n_times`: number of time periods (balanced panel).
///
/// Returns the LM statistic. Compare to χ²(1) critical value.
pub fn breusch_pagan_re(residuals: &[f64], units: &[usize]) -> f64 {
    let n_obs = residuals.len();
    let n_units = *units.iter().max().unwrap_or(&0) + 1;

    // A = Σ_i (Σ_t ê_it)²
    let mut unit_sums = vec![0.0; n_units];
    for i in 0..n_obs {
        unit_sums[units[i]] += residuals[i];
    }
    let a: f64 = unit_sums.iter().map(|s| s * s).sum();

    // B = Σ_i Σ_t ê²_it
    let b: f64 = residuals.iter().map(|r| r * r).sum();

    if b.abs() < 1e-15 { return 0.0; }

    let nt = n_obs as f64;
    let t = nt / n_units as f64; // average T
    let lm = (nt / (2.0 * (t - 1.0))) * (a / b - 1.0).powi(2);
    lm.max(0.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// RE quasi-demeaning factor θ
// ═══════════════════════════════════════════════════════════════════════════

/// Quasi-demeaning factor θ for random effects GLS.
///
/// θ = 0: pooled OLS. θ = 1: fixed effects.
/// RE = OLS on (y_{it} - θ·ȳ_i, x_{it} - θ·x̄_i).
pub fn re_theta(sigma2_eps: f64, sigma2_alpha: f64, t: f64) -> f64 {
    if sigma2_alpha <= 0.0 { return 0.0; }
    1.0 - (sigma2_eps / (t * sigma2_alpha + sigma2_eps)).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Two-Stage Least Squares (2SLS / IV)
// ═══════════════════════════════════════════════════════════════════════════

/// 2SLS instrumental variables estimator.
///
/// `x_endog`: endogenous regressors, n×d_endog row-major.
/// `z`: instruments, n×d_z row-major (must have d_z >= d_endog).
/// `y`: response (length n).
///
/// Stage 1: X̂ = Z(Z'Z)⁻¹Z'X (project X onto Z).
/// Stage 2: β = (X̂'X)⁻¹ X̂'y.
pub fn two_sls(
    x_endog: &[f64], z: &[f64], y: &[f64],
    n: usize, d_endog: usize, d_z: usize,
) -> TwoSlsResult {
    assert_eq!(x_endog.len(), n * d_endog);
    assert_eq!(z.len(), n * d_z);
    assert_eq!(y.len(), n);
    assert!(d_z >= d_endog, "need at least as many instruments as endogenous regressors");

    // Stage 1: project X onto column space of Z
    let z_mat = Mat::from_vec(n, d_z, z.to_vec());
    let mut x_hat = vec![0.0; n * d_endog];

    // For each endogenous variable, regress on Z
    let mut first_stage_f = 0.0;
    for col in 0..d_endog {
        let x_col: Vec<f64> = (0..n).map(|i| x_endog[i * d_endog + col]).collect();
        let gamma = lstsq(&z_mat, &x_col);

        // Fitted values
        let x_mean = x_col.iter().sum::<f64>() / n as f64;
        let mut ss_reg = 0.0;
        let mut ss_res = 0.0;
        for i in 0..n {
            let mut fitted = 0.0;
            for j in 0..d_z { fitted += gamma[j] * z[i * d_z + j]; }
            x_hat[i * d_endog + col] = fitted;
            ss_reg += (fitted - x_mean).powi(2);
            ss_res += (x_col[i] - fitted).powi(2);
        }
        let df_reg = d_z as f64;
        let df_res = (n as f64 - d_z as f64).max(1.0);
        let f = (ss_reg / df_reg) / (ss_res / df_res);
        first_stage_f += f; // average across endogenous vars
    }
    first_stage_f /= d_endog as f64;

    // Stage 2: β = (X̂'X)⁻¹ X̂'y
    // Use (X̂'X̂)⁻¹ X̂'y for simplicity (equivalent under correct specification)
    let xhat_mat = Mat::from_vec(n, d_endog, x_hat);
    let beta = lstsq(&xhat_mat, y);

    TwoSlsResult { beta, first_stage_f }
}

// ═══════════════════════════════════════════════════════════════════════════
// Difference-in-Differences
// ═══════════════════════════════════════════════════════════════════════════

/// Basic 2×2 difference-in-differences.
///
/// `y`: outcomes for all observations.
/// `treated`: true if unit is in treatment group.
/// `post`: true if observation is in post-treatment period.
pub fn did(y: &[f64], treated: &[bool], post: &[bool]) -> DidResult {
    let n = y.len();
    assert_eq!(treated.len(), n);
    assert_eq!(post.len(), n);

    let mut sums = [0.0_f64; 4]; // [ctrl_pre, ctrl_post, treat_pre, treat_post]
    let mut counts = [0usize; 4];

    for i in 0..n {
        let idx = (treated[i] as usize) * 2 + (post[i] as usize);
        sums[idx] += y[i];
        counts[idx] += 1;
    }

    // All four cells must be populated for valid DiD
    if counts.iter().any(|&c| c == 0) {
        return DidResult { att: f64::NAN, se: f64::NAN, t_stat: f64::NAN };
    }
    let means: Vec<f64> = (0..4).map(|i| sums[i] / counts[i] as f64).collect();

    // ATT = (Ȳ_treat_post - Ȳ_treat_pre) - (Ȳ_ctrl_post - Ȳ_ctrl_pre)
    let att = (means[3] - means[2]) - (means[1] - means[0]);

    // SE via pooled variance of the four cell means
    let mut ss = [0.0_f64; 4];
    for i in 0..n {
        let idx = (treated[i] as usize) * 2 + (post[i] as usize);
        ss[idx] += (y[i] - means[idx]).powi(2);
    }

    let var_att: f64 = (0..4).map(|i| {
        if counts[i] > 1 {
            ss[i] / ((counts[i] - 1) as f64 * counts[i] as f64)
        } else {
            0.0
        }
    }).sum();

    let se = var_att.sqrt();
    let t_stat = if se > 0.0 { att / se } else { 0.0 };

    DidResult { att, se, t_stat }
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

    // ── Fixed Effects ──────────────────────────────────────────────────

    #[test]
    fn fe_recovers_slope() {
        // y_it = 3*x_it + α_i + ε, different α per unit
        let n_units = 5;
        let t = 20;
        let n = n_units * t;
        let d = 1;
        let alpha = [0.0, 5.0, -3.0, 7.0, -1.0];

        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut units = Vec::with_capacity(n);
        let mut rng = 42u64;

        for u in 0..n_units {
            for ti in 0..t {
                let xi = ti as f64 / t as f64;
                x.push(xi);
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.3;
                y.push(3.0 * xi + alpha[u] + noise);
                units.push(u);
            }
        }

        let res = panel_fe(&x, &y, n, d, &units);
        close(res.beta[0], 3.0, 0.3, "FE slope");
        assert!(res.r2_within > 0.9, "R²_within={} should be high", res.r2_within);
        assert!(res.se_clustered[0] > 0.0, "SE should be positive");
    }

    #[test]
    fn fe_absorbs_unit_effects() {
        // Two units with vastly different intercepts, same slope
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut units = Vec::new();

        for u in 0..2 {
            let offset = if u == 0 { 0.0 } else { 1000.0 };
            for i in 0..50 {
                let xi = i as f64;
                x.push(xi);
                y.push(2.0 * xi + offset);
                units.push(u);
            }
        }

        let res = panel_fe(&x, &y, 100, 1, &units);
        close(res.beta[0], 2.0, 0.01, "FE slope with large offset");
    }

    // ── First Difference ───────────────────────────────────────────────

    #[test]
    fn fd_recovers_slope() {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut units = Vec::new();

        for u in 0..3 {
            let offset = u as f64 * 10.0;
            for t in 0..20 {
                x.push(t as f64);
                y.push(4.0 * t as f64 + offset);
                units.push(u);
            }
        }

        let res = panel_fd(&x, &y, 60, 1, &units);
        close(res.beta[0], 4.0, 0.01, "FD slope");
    }

    // ── Two-Way FE ─────────────────────────────────────────────────────

    #[test]
    fn twfe_recovers_slope() {
        // y = 2*x + α_i + λ_t, x has idiosyncratic (within-unit-time) variation
        let n_u = 4;
        let n_t = 15;
        let n = n_u * n_t;
        let alpha = [0.0, 3.0, -2.0, 5.0];
        let lambda: Vec<f64> = (0..n_t).map(|t| t as f64 * 0.5).collect();

        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut units = Vec::new();
        let mut times = Vec::new();
        let mut rng = 99u64;

        for u in 0..n_u {
            for t in 0..n_t {
                // x has unit component + time component + idiosyncratic component
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idio = (rng as f64 / u64::MAX as f64 - 0.5) * 4.0;
                let xi = u as f64 + t as f64 * 0.1 + idio;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.2;
                x.push(xi);
                y.push(2.0 * xi + alpha[u] + lambda[t] + noise);
                units.push(u);
                times.push(t);
            }
        }

        let res = panel_twfe(&x, &y, n, 1, &units, &times);
        close(res.beta[0], 2.0, 0.5, "TWFE slope");
    }

    // ── Hausman Test ───────────────────────────────────────────────────

    #[test]
    fn hausman_fe_re_agree() {
        // When RE is valid, Hausman should NOT reject (high p-value)
        let fe = FeResult {
            beta: vec![2.0],
            se_clustered: vec![0.5],
            r2_within: 0.9,
            df: 90,
        };
        let re_beta = vec![2.1];
        let re_se = vec![0.3]; // RE should have smaller SE (more efficient)

        let res = hausman_test(&fe, &re_beta, &re_se);
        assert!(res.p_value > 0.05, "p={} should not reject H₀ when FE≈RE", res.p_value);
    }

    #[test]
    fn hausman_fe_re_disagree() {
        // When RE is biased, large difference → reject
        let fe = FeResult {
            beta: vec![2.0],
            se_clustered: vec![0.2],
            r2_within: 0.9,
            df: 90,
        };
        let re_beta = vec![5.0]; // very different
        let re_se = vec![0.1];

        let res = hausman_test(&fe, &re_beta, &re_se);
        assert!(res.p_value < 0.05, "p={} should reject H₀ when FE≠RE", res.p_value);
    }

    // ── 2SLS ───────────────────────────────────────────────────────────

    #[test]
    fn two_sls_strong_instrument() {
        // y = 3*x + ε, but x is endogenous
        // z is a valid instrument: correlated with x, not with ε
        let n = 200;
        let mut z = vec![0.0; n];
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut rng = 77u64;

        for i in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let zi = (rng as f64 / u64::MAX as f64 - 0.5) * 10.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let ei = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let vi = (rng as f64 / u64::MAX as f64 - 0.5) * 1.0;

            z[i] = zi;
            x[i] = 2.0 * zi + vi; // x correlated with z
            y[i] = 3.0 * x[i] + ei; // true effect is 3.0
        }

        let res = two_sls(&x, &z, &y, n, 1, 1);
        close(res.beta[0], 3.0, 0.5, "2SLS coefficient");
        assert!(res.first_stage_f > 10.0, "First-stage F={} should be >10", res.first_stage_f);
    }

    // ── DiD ────────────────────────────────────────────────────────────

    #[test]
    fn did_detects_treatment_effect() {
        // Treatment effect of 5.0
        let mut y = Vec::new();
        let mut treated = Vec::new();
        let mut post = Vec::new();

        // Control pre
        for _ in 0..20 { y.push(10.0); treated.push(false); post.push(false); }
        // Control post
        for _ in 0..20 { y.push(12.0); treated.push(false); post.push(true); }
        // Treated pre
        for _ in 0..20 { y.push(10.0); treated.push(true); post.push(false); }
        // Treated post (12 + 5 = 17)
        for _ in 0..20 { y.push(17.0); treated.push(true); post.push(true); }

        let res = did(&y, &treated, &post);
        close(res.att, 5.0, 0.01, "DiD ATT");
    }

    #[test]
    fn did_no_effect() {
        let mut y = Vec::new();
        let mut treated = Vec::new();
        let mut post = Vec::new();

        for _ in 0..20 { y.push(10.0); treated.push(false); post.push(false); }
        for _ in 0..20 { y.push(12.0); treated.push(false); post.push(true); }
        for _ in 0..20 { y.push(10.0); treated.push(true); post.push(false); }
        for _ in 0..20 { y.push(12.0); treated.push(true); post.push(true); }

        let res = did(&y, &treated, &post);
        close(res.att, 0.0, 0.01, "DiD no effect");
    }

    // ── RE wrapper ─────────────────────────────────────────────────────

    #[test]
    fn re_delegates_to_lme() {
        let n = 60;
        let d = 1;
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut groups = Vec::new();

        for g in 0..3 {
            for i in 0..20 {
                let xi = i as f64 / 20.0;
                x.push(xi);
                y.push(1.0 + 2.0 * xi + g as f64 * 3.0);
                groups.push(g);
            }
        }

        let res = panel_re(&x, &y, n, d, &groups, 100, 1e-8);
        assert!((res.beta[1] - 2.0).abs() < 0.5, "RE slope={}", res.beta[1]);
        assert!(res.icc > 0.3, "ICC={} should reflect group structure", res.icc);
    }

    // ── Full-matrix Hausman ────────────────────────────────────────────

    #[test]
    fn hausman_full_matrix_agrees() {
        // 2D case: full-matrix Hausman with known covariance
        let fe_beta = vec![2.0, 1.0];
        let re_beta = vec![2.1, 1.05];
        // V_FE > V_RE (FE less efficient under H₀)
        let v_fe = vec![0.25, 0.01, 0.01, 0.16]; // diag(0.5², 0.4²)
        let v_re = vec![0.09, 0.005, 0.005, 0.04]; // diag(0.3², 0.2²)

        let res = hausman_test_full(&fe_beta, &re_beta, &v_fe, &v_re, 2);
        assert!(res.p_value > 0.05, "p={} — small diff should not reject", res.p_value);
    }

    #[test]
    fn hausman_full_matrix_rejects() {
        let fe_beta = vec![2.0, 1.0];
        let re_beta = vec![5.0, 3.0]; // very different
        let v_fe = vec![0.04, 0.0, 0.0, 0.04];
        let v_re = vec![0.01, 0.0, 0.0, 0.01];

        let res = hausman_test_full(&fe_beta, &re_beta, &v_fe, &v_re, 2);
        assert!(res.p_value < 0.01, "p={} — large diff should reject", res.p_value);
    }

    // ── Breusch-Pagan LM ──────────────────────────────────────────────

    #[test]
    fn bp_detects_random_effects() {
        // Units with very different means → pooled OLS residuals
        // will have high within-unit correlation → large LM
        let n_u = 5;
        let t = 20;
        let n = n_u * t;
        let group_means = [0.0, 10.0, -5.0, 8.0, -3.0];
        let mut residuals = Vec::with_capacity(n);
        let mut units = Vec::with_capacity(n);
        let mut rng = 55u64;

        // Simulate pooled OLS residuals (which would contain the group effect)
        for u in 0..n_u {
            for _ in 0..t {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.5;
                residuals.push(group_means[u] + noise);
                units.push(u);
            }
        }

        let lm = breusch_pagan_re(&residuals, &units);
        // With strong group effects, LM should be large
        assert!(lm > 3.84, "LM={} should exceed χ²(1) critical value 3.84", lm);
    }

    #[test]
    fn bp_no_random_effects() {
        // No group structure in residuals → small LM
        let n = 100;
        let mut residuals = Vec::with_capacity(n);
        let mut units = Vec::with_capacity(n);
        let mut rng = 42u64;

        for i in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 2.0;
            residuals.push(noise);
            units.push(i % 5);
        }

        let lm = breusch_pagan_re(&residuals, &units);
        assert!(lm < 10.0, "LM={} should be moderate with no group effects", lm);
    }

    // ── RE theta ───────────────────────────────────────────────────────

    #[test]
    fn theta_boundary_cases() {
        // No random effects → θ = 0 (pooled OLS)
        close(re_theta(1.0, 0.0, 10.0), 0.0, 1e-10, "θ with σ²_α=0");
        // σ²_α >> σ²_ε → θ → 1 (approaches FE)
        let theta = re_theta(0.01, 100.0, 10.0);
        assert!(theta > 0.99, "θ={} should approach 1 when σ²_α dominates", theta);
        // Known: σ²_ε=1, σ²_α=1, T=10 → θ = 1 - √(1/11) ≈ 0.699
        close(re_theta(1.0, 1.0, 10.0), 1.0 - (1.0_f64 / 11.0).sqrt(), 1e-10, "θ exact");
    }
}
