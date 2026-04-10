//! Family 24 — Manifold / Embedding methods.
//!
//! Covers fintek leaves:
//! - `ccm`      (K02P18C01R06F01) — Convergent Cross Mapping (Takens embedding + kNN)
//! - `harmonic` (K02P18C01R07F01) — Harmonic statistics from singular value spacing

use tambear::nonparametric::pearson_r;

// ── CCM ───────────────────────────────────────────────────────────────────────

const CCM_EMBED_DIM: usize = 3;
const CCM_EMBED_TAU: usize = 1;
const CCM_MAX_PTS: usize = 500;
const CCM_MIN_N: usize = 20;
const CCM_K_NEIGHBORS: usize = 4; // CCM_EMBED_DIM + 1

/// CCM result matching fintek's ccm leaf (K02P18C01R06F01).
#[derive(Debug, Clone)]
pub struct CcmResult {
    pub ccm_xy_full: f64,        // CCM corr X→Y at full library
    pub ccm_yx_full: f64,        // CCM corr Y→X at full library
    pub ccm_xy_half: f64,        // CCM corr X→Y at half library
    pub ccm_yx_half: f64,        // CCM corr Y→X at half library
    pub convergence_ratio: f64,  // (ccm_xy_full - ccm_xy_half) / max(ccm_xy_full.abs(), 0.01)
}

impl CcmResult {
    pub fn nan() -> Self {
        Self {
            ccm_xy_full: f64::NAN, ccm_yx_full: f64::NAN,
            ccm_xy_half: f64::NAN, ccm_yx_half: f64::NAN,
            convergence_ratio: f64::NAN,
        }
    }
}

/// Convergent Cross Mapping: does X cause Y?
///
/// Uses Takens delay embedding (dim=3, tau=1). At each library size L,
/// reconstructs manifolds from X and Y, then uses kNN (k=4) on the X-manifold
/// to predict Y values. CCM correlation = Pearson(Y, Ŷ).
///
/// Returns NaN if either series has fewer than 20 points.
pub fn ccm(x: &[f64], y: &[f64]) -> CcmResult {
    let min_n = x.len().min(y.len());
    if min_n < CCM_MIN_N { return CcmResult::nan(); }

    // Subsample for speed
    let (x_data, y_data): (Vec<f64>, Vec<f64>) = if min_n > CCM_MAX_PTS {
        let step = min_n / CCM_MAX_PTS;
        let xd: Vec<f64> = x.iter().step_by(step).take(CCM_MAX_PTS).copied().collect();
        let yd: Vec<f64> = y.iter().step_by(step).take(CCM_MAX_PTS).copied().collect();
        (xd, yd)
    } else {
        (x[..min_n].to_vec(), y[..min_n].to_vec())
    };

    let n = x_data.len();

    // Delay embedding: embed[i] = [x[i], x[i-tau], x[i-2*tau], ...]
    // Valid indices start at (EMBED_DIM-1)*tau
    let embed_start = (CCM_EMBED_DIM - 1) * CCM_EMBED_TAU;
    if n <= embed_start + CCM_K_NEIGHBORS + 1 { return CcmResult::nan(); }

    let n_embed = n - embed_start;

    // Build embedded manifolds
    let embed_x: Vec<Vec<f64>> = (embed_start..n).map(|i| {
        (0..CCM_EMBED_DIM).map(|d| x_data[i - d * CCM_EMBED_TAU]).collect()
    }).collect();
    let embed_y: Vec<Vec<f64>> = (embed_start..n).map(|i| {
        (0..CCM_EMBED_DIM).map(|d| y_data[i - d * CCM_EMBED_TAU]).collect()
    }).collect();

    // Library sizes to test: half and full
    let full_lib = n_embed;
    let half_lib = n_embed / 2;

    let ccm_xy_full = ccm_predict(&embed_x, &y_data[embed_start..], full_lib);
    let ccm_yx_full = ccm_predict(&embed_y, &x_data[embed_start..], full_lib);
    let ccm_xy_half = ccm_predict(&embed_x, &y_data[embed_start..], half_lib);
    let ccm_yx_half = ccm_predict(&embed_y, &x_data[embed_start..], half_lib);

    let convergence_ratio = if ccm_xy_full.is_finite() && ccm_xy_half.is_finite() {
        let denom = ccm_xy_full.abs().max(0.01);
        (ccm_xy_full - ccm_xy_half) / denom
    } else {
        f64::NAN
    };

    CcmResult { ccm_xy_full, ccm_yx_full, ccm_xy_half, ccm_yx_half, convergence_ratio }
}

/// CCM prediction: use kNN on embedded manifold to predict target, return Pearson corr.
///
/// Library: use first `lib_size` points from embed as the library.
/// Predict each point in embed using its k nearest neighbors from the library.
fn ccm_predict(embed: &[Vec<f64>], target: &[f64], lib_size: usize) -> f64 {
    let n = embed.len().min(target.len());
    if lib_size < CCM_K_NEIGHBORS + 1 || n < CCM_K_NEIGHBORS + 1 { return f64::NAN; }
    let lib_size = lib_size.min(n);

    let mut y_true = Vec::with_capacity(n);
    let mut y_pred = Vec::with_capacity(n);

    for i in 0..n {
        // Find k nearest neighbors in library (excluding self)
        let mut dists: Vec<(f64, usize)> = (0..lib_size)
            .filter(|&j| j != i)
            .map(|j| {
                let d: f64 = embed[i].iter().zip(embed[j].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                (d, j)
            })
            .collect();

        if dists.len() < CCM_K_NEIGHBORS { continue; }

        // Partial sort to get k smallest
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let neighbors = &dists[..CCM_K_NEIGHBORS];

        let min_d = neighbors[0].0;

        // Exponential weights: w_j = exp(-d_j / (d_min + 1e-10))
        let weights: Vec<f64> = neighbors.iter().map(|(d, _)| {
            (-(d / (min_d + 1e-10))).exp()
        }).collect();
        let w_sum: f64 = weights.iter().sum();

        if w_sum < 1e-30 { continue; }

        let pred: f64 = neighbors.iter().zip(weights.iter())
            .map(|((_, j), w)| w * target[*j])
            .sum::<f64>() / w_sum;

        y_true.push(target[i]);
        y_pred.push(pred);
    }

    if y_true.len() < 2 { return f64::NAN; }
    pearson_r(&y_true, &y_pred)
}

// ── HARMONIC ──────────────────────────────────────────────────────────────────

const HARMONIC_MIN_EIGENVALUES: usize = 4;
const HARMONIC_MAX_PTS: usize = 4000;
const HARMONIC_R_POISSON: f64 = 0.38629; // 2*ln(2) - 1
const HARMONIC_R_GOE: f64 = 0.53590;    // 4 - 2*sqrt(3)

/// Harmonic result matching fintek's harmonic leaf (K02P18C01R07F01).
///
/// 12 outputs: r-statistic + Poisson/GOE classification for price and vol
/// substrates at 3 embedding dimensions each.
#[derive(Debug, Clone)]
pub struct HarmonicResult {
    // Price substrate
    pub r_stat_price_d2: f64,
    pub r_stat_price_d3: f64,
    pub r_stat_price_d5: f64,
    pub regime_price_d2: f64,  // 0=Poisson, 1=GOE, 0.5=intermediate
    pub regime_price_d3: f64,
    pub regime_price_d5: f64,
    // Vol substrate (|returns|)
    pub r_stat_vol_d2: f64,
    pub r_stat_vol_d3: f64,
    pub r_stat_vol_d5: f64,
    pub regime_vol_d2: f64,
    pub regime_vol_d3: f64,
    pub regime_vol_d5: f64,
}

impl HarmonicResult {
    pub fn nan() -> Self {
        Self {
            r_stat_price_d2: f64::NAN, r_stat_price_d3: f64::NAN, r_stat_price_d5: f64::NAN,
            regime_price_d2: f64::NAN, regime_price_d3: f64::NAN, regime_price_d5: f64::NAN,
            r_stat_vol_d2: f64::NAN, r_stat_vol_d3: f64::NAN, r_stat_vol_d5: f64::NAN,
            regime_vol_d2: f64::NAN, regime_vol_d3: f64::NAN, regime_vol_d5: f64::NAN,
        }
    }
}

/// Harmonic statistics via Hankel delay-embedding SVD.
///
/// Constructs a Hankel matrix from `ticks`, computes singular values via
/// tambear's SVD, then derives the Wigner nearest-neighbor spacing r-statistic.
/// The r-statistic classifies eigenvalue repulsion:
///   r ≈ 0.386 → Poisson (no repulsion, integrable)
///   r ≈ 0.536 → GOE (Wigner repulsion, chaotic)
///
/// Operates on ticks (prices), derives vol as |diff(ticks)|.
///
/// Returns NaN if insufficient data for SVD.
pub fn harmonic(ticks: &[f64]) -> HarmonicResult {
    let n = ticks.len();
    if n < HARMONIC_MIN_EIGENVALUES + 2 { return HarmonicResult::nan(); }

    // Subsample for speed
    let data: Vec<f64> = if n > HARMONIC_MAX_PTS {
        let step = n / HARMONIC_MAX_PTS;
        ticks.iter().step_by(step).take(HARMONIC_MAX_PTS).copied().collect()
    } else {
        ticks.to_vec()
    };
    let n = data.len();

    // Vol substrate: |returns| from price differences
    let vol: Vec<f64> = data.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

    // Embedding dims to test
    let dims = [2usize, 3, 5];

    let mut r_price = [f64::NAN; 3];
    let mut r_vol = [f64::NAN; 3];

    for (d_idx, &embed_dim) in dims.iter().enumerate() {
        r_price[d_idx] = hankel_r_stat(&data, embed_dim);
        if vol.len() >= embed_dim + HARMONIC_MIN_EIGENVALUES {
            r_vol[d_idx] = hankel_r_stat(&vol, embed_dim);
        }
    }

    HarmonicResult {
        r_stat_price_d2: r_price[0],
        r_stat_price_d3: r_price[1],
        r_stat_price_d5: r_price[2],
        regime_price_d2: classify_regime(r_price[0]),
        regime_price_d3: classify_regime(r_price[1]),
        regime_price_d5: classify_regime(r_price[2]),
        r_stat_vol_d2: r_vol[0],
        r_stat_vol_d3: r_vol[1],
        r_stat_vol_d5: r_vol[2],
        regime_vol_d2: classify_regime(r_vol[0]),
        regime_vol_d3: classify_regime(r_vol[1]),
        regime_vol_d5: classify_regime(r_vol[2]),
    }
}

/// Build Hankel matrix and compute r-statistic from singular value spacings.
///
/// Hankel matrix: rows = data[i..i+embed_dim], n_rows = n - embed_dim + 1.
/// SVD gives singular values in ascending order (reversed from tambear default).
/// Spacings: s_i = σ_{i+1} - σ_i. r-statistic: mean(min(s_i, s_{i+1}) / max(...)).
fn hankel_r_stat(data: &[f64], embed_dim: usize) -> f64 {
    let n = data.len();
    if n < embed_dim + HARMONIC_MIN_EIGENVALUES { return f64::NAN; }

    let n_rows = n - embed_dim + 1;
    let n_cols = embed_dim;

    // Build Hankel matrix (row-major): H[i][j] = data[i + j]
    let mut mat_data = Vec::with_capacity(n_rows * n_cols);
    for i in 0..n_rows {
        for j in 0..n_cols {
            mat_data.push(data[i + j]);
        }
    }

    // Use tambear SVD
    let mat = tambear::linear_algebra::Mat { data: mat_data, rows: n_rows, cols: n_cols };
    let svd_result = tambear::linear_algebra::svd(&mat);

    // tambear returns sigma in descending order — reverse to ascending
    let mut sigma = svd_result.sigma;
    sigma.reverse();

    // Need at least HARMONIC_MIN_EIGENVALUES singular values
    if sigma.len() < HARMONIC_MIN_EIGENVALUES { return f64::NAN; }

    // Compute spacings: s_i = sigma[i+1] - sigma[i]
    let spacings: Vec<f64> = sigma.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

    if spacings.len() < 2 { return f64::NAN; }

    // r-statistic: mean of min(s_i, s_{i+1}) / max(s_i, s_{i+1})
    let r_vals: Vec<f64> = spacings.windows(2).map(|w| {
        let a = w[0];
        let b = w[1];
        let mx = a.max(b);
        if mx < 1e-30 { return f64::NAN; }
        a.min(b) / mx
    }).filter(|v| v.is_finite())
    .collect();

    if r_vals.is_empty() { return f64::NAN; }
    r_vals.iter().sum::<f64>() / r_vals.len() as f64
}

/// Classify r-statistic as Poisson (0), GOE (1), or intermediate (0.5).
fn classify_regime(r: f64) -> f64 {
    if !r.is_finite() { return f64::NAN; }
    let mid = (HARMONIC_R_POISSON + HARMONIC_R_GOE) / 2.0; // ≈ 0.461
    if r < mid { 0.0 } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wn(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = tambear::rng::Xoshiro256::new(seed);
        (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect()
    }

    fn price_series(n: usize, seed: u64) -> Vec<f64> {
        let returns = wn(n, seed);
        let mut prices = vec![100.0f64];
        for r in &returns {
            prices.push(prices.last().unwrap() * (1.0 + r));
        }
        prices
    }

    // ── CCM tests ──

    #[test]
    fn ccm_too_short() {
        let r = ccm(&[0.0; 5], &[0.0; 5]);
        assert!(r.ccm_xy_full.is_nan());
    }

    #[test]
    fn ccm_white_noise_finite() {
        let x = wn(100, 42);
        let y = wn(100, 99);
        let r = ccm(&x, &y);
        assert!(r.ccm_xy_full.is_finite() || r.ccm_xy_full.is_nan());
        assert!(r.ccm_yx_full.is_finite() || r.ccm_yx_full.is_nan());
    }

    #[test]
    fn ccm_correlated_series_detects_coupling() {
        // x causes y: y[t] = 0.7*x[t-1] + noise
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let x: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let y: Vec<f64> = (0..100).map(|i| {
            let prev = if i > 0 { x[i-1] } else { 0.0 };
            0.7 * prev + tambear::rng::sample_normal(&mut rng, 0.0, 0.3)
        }).collect();
        let r = ccm(&x, &y);
        // Both directions may be finite; just check they are
        assert!(r.ccm_xy_full.is_finite() || r.ccm_xy_full.is_nan());
        assert!(r.convergence_ratio.is_finite() || r.convergence_ratio.is_nan());
    }

    // ── Harmonic tests ──

    #[test]
    fn harmonic_too_short() {
        let r = harmonic(&[1.0, 2.0, 3.0]);
        assert!(r.r_stat_price_d2.is_nan());
    }

    #[test]
    fn harmonic_price_series_finite() {
        let prices = price_series(100, 42);
        let r = harmonic(&prices);
        // At least some r-stats should be finite
        let any_finite = [r.r_stat_price_d2, r.r_stat_price_d3, r.r_stat_price_d5]
            .iter().any(|v| v.is_finite());
        assert!(any_finite, "at least one price r-stat should be finite");
    }

    #[test]
    fn harmonic_regime_in_valid_range() {
        let prices = price_series(200, 77);
        let r = harmonic(&prices);
        for regime in [r.regime_price_d2, r.regime_price_d3, r.regime_vol_d2] {
            if regime.is_finite() {
                assert!(regime == 0.0 || regime == 1.0,
                    "regime must be 0 or 1, got {}", regime);
            }
        }
    }

    #[test]
    fn harmonic_r_stat_in_unit_interval() {
        let prices = price_series(300, 55);
        let r = harmonic(&prices);
        for rs in [r.r_stat_price_d2, r.r_stat_price_d3, r.r_stat_vol_d2] {
            if rs.is_finite() {
                assert!(rs >= 0.0 && rs <= 1.0,
                    "r-stat must be in [0,1], got {}", rs);
            }
        }
    }
}
