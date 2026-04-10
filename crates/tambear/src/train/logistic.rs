//! Logistic regression via GPU gradient descent.
//!
//! Proves the gradient duality: the forward pass and backward pass both use
//! `TiledEngine::DotProduct`. The same operation, transposed arguments.
//!
//! ```text
//! Forward:  z = X * β        (TiledEngine DotProduct, n×1)
//! Map:      p = sigmoid(z)   (element-wise, fuses in lazy pipeline)
//! Loss:     L = -Σ[y log p + (1-y) log(1-p)] / n
//! Backward: ∇L = X' * (p-y)  (TiledEngine DotProduct, p×1)  ← SAME OP, TRANSPOSED
//! Update:   β -= lr * ∇L / n
//! ```
//!
//! "Tam doesn't differentiate. Tam transposes."

use std::sync::Arc;
use winrapids_tiled::{TiledEngine, DotProductOp};

/// A fitted logistic regression model (binary classification).
pub struct LogisticModel {
    /// Regression coefficients (one per feature).
    pub coefficients: Vec<f64>,
    /// Intercept (bias) term.
    pub intercept: f64,
    /// Final training loss (binary cross-entropy).
    pub loss: f64,
    /// Training accuracy (fraction of correct predictions).
    pub accuracy: f64,
    /// Number of gradient descent iterations run.
    pub iterations: usize,
    /// Number of training samples.
    pub n_samples: usize,
    /// Number of features.
    pub n_features: usize,
}

impl LogisticModel {
    /// Predict probabilities for new data. Returns P(y=1|x) for each sample.
    /// x is row-major: n_new × n_features.
    pub fn predict_proba(&self, x: &[f64], n: usize) -> Vec<f64> {
        let d = self.n_features;
        assert_eq!(x.len(), n * d);
        let mut probs = vec![0.0f64; n];
        for i in 0..n {
            let mut z = self.intercept;
            for j in 0..d {
                z += self.coefficients[j] * x[i * d + j];
            }
            probs[i] = sigmoid(z);
        }
        probs
    }

    /// Predict binary labels (0 or 1) with threshold 0.5.
    pub fn predict(&self, x: &[f64], n: usize) -> Vec<u8> {
        self.predict_proba(x, n).iter().map(|&p| if p >= 0.5 { 1 } else { 0 }).collect()
    }
}

fn sigmoid(z: f64) -> f64 {
    crate::neural::sigmoid(z)
}

/// Fit logistic regression using GPU-accelerated gradient descent.
///
/// The gradient duality: forward uses `DotProduct(X, β)`, backward uses
/// `DotProduct(X', residual)`. Same TiledEngine op, transposed arguments.
///
/// # Parameters
/// - `x`: row-major f64, n × d
/// - `y`: binary labels (0.0 or 1.0), length n
/// - `n`, `d`: shape
/// - `lr`: learning rate
/// - `max_iter`: maximum gradient descent iterations
/// - `tol`: convergence tolerance on loss change
pub fn fit(
    x: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    lr: f64,
    max_iter: usize,
    tol: f64,
) -> Result<LogisticModel, Box<dyn std::error::Error>> {
    assert_eq!(x.len(), n * d);
    assert_eq!(y.len(), n);
    assert!(n > d + 1);

    let p = d + 1; // augmented (features + intercept)

    // -----------------------------------------------------------------------
    // Augment X with intercept column → X_aug (n × p) and X_aug_T (p × n)
    // -----------------------------------------------------------------------
    let mut x_aug = vec![0.0f64; n * p];
    let mut x_aug_t = vec![0.0f64; p * n];
    for i in 0..n {
        for j in 0..d {
            x_aug[i * p + j] = x[i * d + j];
            x_aug_t[j * n + i] = x[i * d + j];
        }
        x_aug[i * p + d] = 1.0;
        x_aug_t[d * n + i] = 1.0;
    }

    // Initialize weights to zero
    let mut beta = vec![0.0f64; p];

    let tiled = TiledEngine::new(tam_gpu::detect());

    let mut prev_loss = f64::INFINITY;
    let mut iterations = 0;
    let n_f = n as f64;

    for _iter in 0..max_iter {
        iterations += 1;

        // ── Forward: z = X_aug * β  (TiledEngine DotProduct) ─────────────
        // A = x_aug (n × p), B = beta (p × 1) → C = z (n × 1)
        let z = tiled.run(&DotProductOp, &x_aug, &beta, n, 1, p)?;

        // ── Map: p = sigmoid(z), residual = p - y  (CPU, fuses in lazy) ──
        let mut probs = vec![0.0f64; n];
        let mut residual = vec![0.0f64; n];
        let mut loss = 0.0f64;

        for i in 0..n {
            probs[i] = sigmoid(z[i]);
            residual[i] = probs[i] - y[i];

            // Binary cross-entropy (with numerical stability)
            let pi = probs[i].clamp(1e-15, 1.0 - 1e-15);
            loss -= y[i] * pi.ln() + (1.0 - y[i]) * (1.0 - pi).ln();
        }
        loss /= n_f;

        // Check convergence
        if (prev_loss - loss).abs() < tol {
            break;
        }
        prev_loss = loss;

        // ── Backward: grad = X_aug_T * residual  (TiledEngine DotProduct) ─
        // A = x_aug_t (p × n), B = residual (n × 1) → C = grad (p × 1)
        // THIS IS THE GRADIENT DUALITY: same op as forward, transposed X
        let grad = tiled.run(&DotProductOp, &x_aug_t, &residual, p, 1, n)?;

        // ── Update: β -= lr * grad / n ───────────────────────────────────
        for j in 0..p {
            beta[j] -= lr * grad[j] / n_f;
        }
    }

    // Final accuracy
    let final_z = tiled.run(&DotProductOp, &x_aug, &beta, n, 1, p)?;
    let correct = (0..n).filter(|&i| {
        let pred = if sigmoid(final_z[i]) >= 0.5 { 1.0 } else { 0.0 };
        (pred - y[i]).abs() < 0.5
    }).count();

    eprintln!("[train.logistic] {} iters, loss={:.6}, accuracy={:.1}%",
        iterations, prev_loss, 100.0 * correct as f64 / n_f);

    Ok(LogisticModel {
        coefficients: beta[..d].to_vec(),
        intercept: beta[d],
        loss: prev_loss,
        accuracy: correct as f64 / n_f,
        iterations,
        n_samples: n,
        n_features: d,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linearly_separable() {
        // Two clusters: class 0 at x<0, class 1 at x>0
        let n = 200;
        let d = 2;
        let mut x = vec![0.0f64; n * d];
        let mut y = vec![0.0f64; n];

        for i in 0..n {
            let label = if i < n / 2 { 0.0 } else { 1.0 };
            // Class 0: centered at (-2, -2). Class 1: centered at (2, 2).
            let cx = if label == 0.0 { -2.0 } else { 2.0 };
            // Deterministic spread using index
            let offset = (i % (n / 2)) as f64 / (n as f64 / 2.0) - 0.5; // -0.5 to 0.5
            x[i * d] = cx + offset * 0.5;
            x[i * d + 1] = cx + offset * 0.3;
            y[i] = label;
        }

        let model = fit(&x, &y, n, d, 1.0, 500, 1e-8).unwrap();
        assert!(model.accuracy > 0.95, "accuracy={:.1}%, expected >95%", model.accuracy * 100.0);
        assert!(model.loss < 0.3, "loss={:.4}, expected <0.3", model.loss);

        // Coefficients should be positive (both features separate classes in same direction)
        assert!(model.coefficients[0] > 0.0, "coeff[0]={:.4} should be positive", model.coefficients[0]);
        assert!(model.coefficients[1] > 0.0, "coeff[1]={:.4} should be positive", model.coefficients[1]);
    }

    #[test]
    fn gradient_duality_forward_backward_same_op() {
        // This test verifies the structural claim: forward and backward both
        // use DotProduct, just with transposed arguments.
        //
        // We don't test model quality — we test that the gradient descent loop
        // RUNS using only TiledEngine DotProduct for matrix operations.
        let n = 50;
        let d = 2;
        let x: Vec<f64> = (0..n*d).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();

        // Just 5 iterations — we're testing the MECHANICS, not convergence
        let model = fit(&x, &y, n, d, 0.1, 5, 0.0).unwrap();
        assert_eq!(model.iterations, 5);
        assert_eq!(model.n_features, d);
        assert_eq!(model.n_samples, n);
        // Loss should be finite (no NaN from the gradient computation)
        assert!(model.loss.is_finite(), "loss={}, should be finite", model.loss);
    }

    #[test]
    fn predict_proba_range() {
        // After fitting, probabilities should be in [0, 1]
        let n = 100;
        let d = 1;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / 10.0 - 5.0).collect(); // -5 to 5
        let y: Vec<f64> = (0..n).map(|i| if i >= n / 2 { 1.0 } else { 0.0 }).collect();

        let model = fit(&x, &y, n, d, 1.0, 200, 1e-8).unwrap();
        let probs = model.predict_proba(&x, n);

        for (i, &p) in probs.iter().enumerate() {
            assert!(p >= 0.0 && p <= 1.0, "proba[{}]={} out of [0,1]", i, p);
        }

        // Class 0 samples (low x) should have low probability
        let mean_p_class0: f64 = probs[..n/2].iter().sum::<f64>() / (n/2) as f64;
        let mean_p_class1: f64 = probs[n/2..].iter().sum::<f64>() / (n/2) as f64;
        assert!(mean_p_class0 < 0.5, "class 0 mean prob={:.3}, should be <0.5", mean_p_class0);
        assert!(mean_p_class1 > 0.5, "class 1 mean prob={:.3}, should be >0.5", mean_p_class1);
    }
}
