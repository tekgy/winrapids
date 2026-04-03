//! Linear regression via GPU tiled dot products.
//!
//! Normal equations: beta = (X'X)^{-1} X'y
//!
//! GPU work (TiledEngine DotProduct):
//!   X'X  — O(n * d^2) multiply-adds, tiled on GPU
//!   X'y  — O(n * d) multiply-adds, tiled on GPU
//!
//! CPU work (Cholesky solve):
//!   (X'X) beta = X'y — O(d^3), negligible for d < 1000
//!
//! "Tam doesn't fit. Tam accumulates the sufficient statistics."

use std::sync::Arc;
use winrapids_tiled::{TiledEngine, DotProductOp};

use super::cholesky;
use crate::intermediates::{DataId, IntermediateTag, SufficientStatistics, TamSession};

/// A fitted linear regression model.
pub struct LinearModel {
    /// Regression coefficients (one per feature).
    pub coefficients: Vec<f64>,
    /// Intercept term.
    pub intercept: f64,
    /// Coefficient of determination (1.0 = perfect fit, 0.0 = no better than mean).
    pub r_squared: f64,
    /// Root mean squared error.
    pub rmse: f64,
    /// Number of training samples.
    pub n_samples: usize,
    /// Number of features (excluding intercept).
    pub n_features: usize,
}

impl LinearModel {
    /// Predict y values for new X data.
    /// x is row-major: n_new x n_features.
    pub fn predict(&self, x: &[f64], n: usize) -> Vec<f64> {
        let d = self.n_features;
        assert_eq!(x.len(), n * d, "x must be {} x {}", n, d);
        let mut y = vec![self.intercept; n];
        for i in 0..n {
            for j in 0..d {
                y[i] += self.coefficients[j] * x[i * d + j];
            }
        }
        y
    }
}

/// Fit a linear regression model using GPU-accelerated normal equations.
///
/// - `x`: row-major f64 data, n rows x d columns
/// - `y`: target values, length n
/// - `n`: number of samples
/// - `d`: number of features
///
/// Returns a fitted LinearModel with coefficients, intercept, R^2, and RMSE.
pub fn fit(x: &[f64], y: &[f64], n: usize, d: usize) -> Result<LinearModel, Box<dyn std::error::Error>> {
    assert_eq!(x.len(), n * d, "x must be n*d = {}", n * d);
    assert_eq!(y.len(), n, "y must have n={} elements", n);
    assert!(n > d + 1, "need more samples than features (n={} > d+1={})", n, d + 1);

    let p = d + 1; // augmented dimension (features + intercept)

    eprintln!("[train.linear] {} samples, {} features", n, d);

    // -----------------------------------------------------------------------
    // Step 1: Augment X with intercept column → X_aug (n x p, row-major)
    //         Then transpose → X_aug_T (p x n, row-major)
    // -----------------------------------------------------------------------
    let t0 = std::time::Instant::now();

    let mut x_aug_t = vec![0.0f64; p * n]; // p rows x n cols (transposed)
    for i in 0..n {
        for j in 0..d {
            // X_aug[i, j] = x[i * d + j]
            // X_aug_T[j, i] = x[i * d + j]
            x_aug_t[j * n + i] = x[i * d + j];
        }
        // Intercept column: X_aug[i, d] = 1.0
        // X_aug_T[d, i] = 1.0
        x_aug_t[d * n + i] = 1.0;
    }

    // Also need X_aug in original layout for X'y... actually we need B = X_aug (n x p)
    let mut x_aug = vec![0.0f64; n * p];
    for i in 0..n {
        for j in 0..d {
            x_aug[i * p + j] = x[i * d + j];
        }
        x_aug[i * p + d] = 1.0;
    }

    let prep_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[train.linear] data prep: {:.1}ms", prep_ms);

    // -----------------------------------------------------------------------
    // Step 2: GPU tiled dot products
    //   X'X = X_aug_T (p x n) * X_aug (n x p) → p x p
    //   X'y = X_aug_T (p x n) * y (n x 1) → p x 1
    // -----------------------------------------------------------------------
    let tiled = TiledEngine::new(tam_gpu::detect());

    let t0 = std::time::Instant::now();
    let xtx = tiled.run(&DotProductOp, &x_aug_t, &x_aug, p, p, n)?;
    let xtx_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[train.linear] X'X ({0}x{0}): {1:.1}ms (TiledEngine DotProduct)", p, xtx_ms);

    let t0 = std::time::Instant::now();
    let xty = tiled.run(&DotProductOp, &x_aug_t, y, p, 1, n)?;
    let xty_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[train.linear] X'y ({0}x1): {1:.1}ms (TiledEngine DotProduct)", p, xty_ms);

    // -----------------------------------------------------------------------
    // Step 3: Cholesky solve (CPU, tiny d x d)
    // -----------------------------------------------------------------------
    let t0 = std::time::Instant::now();
    let beta = cholesky::solve(&xtx, &xty, p)
        .ok_or("Cholesky failed: X'X is not positive definite (collinear features?)")?;
    let solve_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[train.linear] Cholesky solve: {:.3}ms", solve_ms);

    // -----------------------------------------------------------------------
    // Step 4: Compute R^2 and RMSE
    // -----------------------------------------------------------------------
    let y_mean = y.iter().sum::<f64>() / n as f64;

    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..n {
        let mut y_hat = beta[d]; // intercept
        for j in 0..d {
            y_hat += beta[j] * x[i * d + j];
        }
        let residual = y[i] - y_hat;
        ss_res += residual * residual;
        let deviation = y[i] - y_mean;
        ss_tot += deviation * deviation;
    }

    let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 1.0 };
    let rmse = (ss_res / n as f64).sqrt();

    eprintln!("[train.linear] R^2 = {:.6}, RMSE = {:.6}", r_squared, rmse);
    eprintln!("[train.linear] coefficients: {:?}", &beta[..d]);
    eprintln!("[train.linear] intercept: {:.6}", beta[d]);

    Ok(LinearModel {
        coefficients: beta[..d].to_vec(),
        intercept: beta[d],
        r_squared,
        rmse,
        n_samples: n,
        n_features: d,
    })
}

// ---------------------------------------------------------------------------
// Per-column sufficient statistics
// ---------------------------------------------------------------------------

/// Synthetic grouping ID for per-column statistics of an n×d matrix.
///
/// Two matrices with the same (n, d) get the same grouping_id — that's correct
/// because the grouping pattern (column index) is identical. The data_id
/// differentiates the actual content.
fn column_grouping_id(n: usize, d: usize) -> DataId {
    let mut buf = [0u8; 24];
    buf[0..3].copy_from_slice(b"col");
    // skip bytes 3..8 (padding)
    buf[8..16].copy_from_slice(&(n as u64).to_le_bytes());
    buf[16..24].copy_from_slice(&(d as u64).to_le_bytes());
    DataId::from_bytes(&buf)
}

/// Compute per-column sufficient statistics for a row-major n×d feature matrix.
///
/// This is O(n*d) on CPU — fast for typical feature counts (d < 1000).
/// Computes raw (sum, sum_sqs, count) and lets `from_vecs` convert to
/// Welford's `m2` representation for numerically stable variance.
pub fn column_stats(x: &[f64], n: usize, d: usize) -> SufficientStatistics {
    assert_eq!(x.len(), n * d);
    let mut sums = vec![0.0f64; d];
    let mut sum_sqs = vec![0.0f64; d];
    for i in 0..n {
        for j in 0..d {
            let v = x[i * d + j];
            sums[j] += v;
            sum_sqs[j] += v * v;
        }
    }
    let counts = vec![n as f64; d];
    SufficientStatistics::from_vecs(d, sums, sum_sqs, counts)
}

// ---------------------------------------------------------------------------
// Session-aware fit with z-score normalization
// ---------------------------------------------------------------------------

/// Fit linear regression with automatic z-score normalization via session.
///
/// # What this does differently from [`fit`]
///
/// 1. Checks the session for pre-computed per-column `SufficientStatistics`
/// 2. If missing, computes them and registers for downstream consumers
/// 3. Z-scores features: `x_norm[i,j] = (x[i,j] - mean[j]) / std[j]`
/// 4. Fits the model in normalized space (better numerical conditioning)
/// 5. Transforms coefficients back to original space
///
/// The returned `LinearModel` has coefficients in the **original** feature space —
/// `predict()` works on un-normalized data, identical to [`fit`].
///
/// # Why z-score?
///
/// Features with different scales (e.g., price in dollars vs count in thousands)
/// produce an ill-conditioned X'X matrix. Z-scoring makes the condition number
/// close to 1, which improves Cholesky stability.
///
/// # Sharing pattern
///
/// If a scatter step already computed per-group statistics on the same data
/// (e.g., feature engineering), those statistics may already be in the session.
/// This function reuses them instead of recomputing.
pub fn fit_session(
    session: &mut TamSession,
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
) -> Result<LinearModel, Box<dyn std::error::Error>> {
    assert_eq!(x.len(), n * d);
    assert_eq!(y.len(), n);
    assert!(n > d + 1);

    let data_id = DataId::from_f64(x);
    let grouping_id = column_grouping_id(n, d);
    let tag = IntermediateTag::SufficientStatistics { data_id, grouping_id };

    // Check session for cached column statistics
    let stats: Arc<SufficientStatistics> = if let Some(cached) = session.get(&tag) {
        eprintln!("[train.linear] session HIT: reusing per-column statistics");
        cached
    } else {
        eprintln!("[train.linear] session MISS: computing per-column statistics");
        let s = Arc::new(column_stats(x, n, d));
        session.register(tag, Arc::clone(&s));
        s
    };

    // Compute per-column mean and std from sufficient statistics
    let means: Vec<f64> = (0..d).map(|j| stats.mean(j)).collect();
    let stds: Vec<f64> = (0..d).map(|j| {
        let s = stats.std(j);
        if s < 1e-15 { 1.0 } else { s } // constant columns: don't divide by zero
    }).collect();

    eprintln!("[train.linear] z-scoring {} features (session-aware)", d);

    // Z-score the features
    let mut x_norm = vec![0.0f64; n * d];
    for i in 0..n {
        for j in 0..d {
            x_norm[i * d + j] = (x[i * d + j] - means[j]) / stds[j];
        }
    }

    // Fit in normalized space
    let model_norm = fit(&x_norm, y, n, d)?;

    // Transform coefficients back to original space:
    //   y = Σ (β_norm_j / σ_j) * x_j + (b_norm - Σ β_norm_j * μ_j / σ_j)
    let mut coefficients = vec![0.0f64; d];
    let mut intercept = model_norm.intercept;
    for j in 0..d {
        coefficients[j] = model_norm.coefficients[j] / stds[j];
        intercept -= model_norm.coefficients[j] * means[j] / stds[j];
    }

    Ok(LinearModel {
        coefficients,
        intercept,
        r_squared: model_norm.r_squared,
        rmse: model_norm.rmse,
        n_samples: n,
        n_features: d,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_linear() {
        // y = 3*x1 + 2*x2 + 1 (exact, no noise)
        let n = 100;
        let d = 2;
        let mut x = vec![0.0f64; n * d];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let x1 = i as f64 / 10.0;
            let x2 = (i as f64).sin();
            x[i * d] = x1;
            x[i * d + 1] = x2;
            y[i] = 3.0 * x1 + 2.0 * x2 + 1.0;
        }

        let model = fit(&x, &y, n, d).unwrap();
        assert!((model.coefficients[0] - 3.0).abs() < 1e-6, "coeff[0]={}", model.coefficients[0]);
        assert!((model.coefficients[1] - 2.0).abs() < 1e-6, "coeff[1]={}", model.coefficients[1]);
        assert!((model.intercept - 1.0).abs() < 1e-6, "intercept={}", model.intercept);
        assert!(model.r_squared > 0.9999, "R^2={}", model.r_squared);
    }

    #[test]
    #[allow(deprecated)]
    fn column_stats_correct() {
        // 3 samples, 2 features
        let x = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
        ];
        let stats = column_stats(&x, 3, 2);
        assert_eq!(stats.n_groups, 2);
        assert!((stats.mean(0) - 2.0).abs() < 1e-10);
        assert!((stats.mean(1) - 20.0).abs() < 1e-10);
        // variance: E[X^2] - E[X]^2
        // col 0: (1+4+9)/3 - 4 = 14/3 - 4 = 2/3
        assert!((stats.variance(0) - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn fit_session_matches_fit() {
        // fit_session should produce identical coefficients to fit (in original space)
        // Features are intentionally different scales but NOT collinear
        let n = 200;
        let d = 3;
        let mut x = vec![0.0f64; n * d];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let x1 = (i as f64) / 20.0;         // range 0..10
            let x2 = ((i as f64) * 0.1).sin();   // range -1..1
            let x3 = ((i as f64) * 0.07).cos() * 500.0; // range -500..500, not collinear with x1
            x[i * d] = x1;
            x[i * d + 1] = x2;
            x[i * d + 2] = x3;
            y[i] = 2.5 * x1 - 1.3 * x2 + 0.01 * x3 + 4.2;
        }

        let mut session = TamSession::new();
        let model_s = fit_session(&mut session, &x, &y, n, d).unwrap();
        let model_r = fit(&x, &y, n, d).unwrap();

        // Both should recover the true coefficients
        for j in 0..d {
            let diff = (model_s.coefficients[j] - model_r.coefficients[j]).abs();
            assert!(diff < 1e-4, "coeff[{}] diverged: session={:.6} raw={:.6} diff={:.2e}",
                j, model_s.coefficients[j], model_r.coefficients[j], diff);
        }
        assert!((model_s.intercept - model_r.intercept).abs() < 1e-4,
            "intercept: session={:.6} raw={:.6}", model_s.intercept, model_r.intercept);
        assert!((model_s.r_squared - model_r.r_squared).abs() < 1e-6);
    }

    #[test]
    fn fit_session_caches_statistics() {
        let n = 100;
        let d = 2;
        let mut x = vec![0.0f64; n * d];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            x[i * d] = i as f64;
            x[i * d + 1] = (i as f64).sin();
            y[i] = 3.0 * x[i * d] + 2.0 * x[i * d + 1] + 1.0;
        }

        let mut session = TamSession::new();
        assert_eq!(session.len(), 0);

        // First call: computes and registers stats
        let _m1 = fit_session(&mut session, &x, &y, n, d).unwrap();
        assert_eq!(session.len(), 1, "statistics registered");

        // Second call: hits cache (no new registration)
        let _m2 = fit_session(&mut session, &x, &y, n, d).unwrap();
        assert_eq!(session.len(), 1, "no new registration — cache hit");
    }

    #[test]
    fn fit_session_predict_on_raw_data() {
        // Verify that predict() works on un-normalized data after fit_session
        let n = 100;
        let d = 2;
        let mut x = vec![0.0f64; n * d];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let x1 = i as f64 / 10.0;
            let x2 = (i as f64).sin();
            x[i * d] = x1;
            x[i * d + 1] = x2;
            y[i] = 3.0 * x1 + 2.0 * x2 + 1.0;
        }

        let mut session = TamSession::new();
        let model = fit_session(&mut session, &x, &y, n, d).unwrap();

        // predict on the ORIGINAL (un-normalized) data should match y
        let preds = model.predict(&x, n);
        for i in 0..n {
            assert!((preds[i] - y[i]).abs() < 1e-4,
                "predict[{}]: got {:.6}, expected {:.6}", i, preds[i], y[i]);
        }
    }
}
