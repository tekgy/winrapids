//! Naive Bayes classifiers — Gaussian, Multinomial, Bernoulli.
//!
//! All three share the same structure: estimate P(class) and P(features|class)
//! from training data, then classify via argmax P(class) * P(features|class).
//!
//! ## Architecture
//!
//! Training = accumulate per-class sufficient statistics (Kingdom A).
//! Prediction = gather class probabilities, argmax (Kingdom A).

use std::collections::HashMap;

/// Trained Gaussian Naive Bayes model.
#[derive(Debug, Clone)]
pub struct GaussianNB {
    /// Number of classes.
    pub n_classes: usize,
    /// Class labels (sorted).
    pub classes: Vec<i32>,
    /// Prior probabilities per class.
    pub class_prior: Vec<f64>,
    /// Per-class means: class_means[c][j] = mean of feature j in class c.
    pub class_means: Vec<Vec<f64>>,
    /// Per-class variances: class_vars[c][j] = variance of feature j in class c.
    pub class_vars: Vec<Vec<f64>>,
    /// Number of features.
    pub n_features: usize,
    /// Smoothing parameter added to variances (default 1e-9).
    pub var_smoothing: f64,
}

/// Fit a Gaussian Naive Bayes classifier.
///
/// `x`: n × d feature matrix (row-major).
/// `y`: class labels (length n, integer-valued).
/// `var_smoothing`: variance floor to prevent division by zero (default 1e-9).
pub fn gaussian_nb_fit(x: &[f64], y: &[i32], n: usize, d: usize, var_smoothing: Option<f64>) -> GaussianNB {
    assert_eq!(x.len(), n * d);
    assert_eq!(y.len(), n);
    let var_smoothing = var_smoothing.unwrap_or(1e-9);

    // Discover classes
    let mut class_counts: HashMap<i32, usize> = HashMap::new();
    for &label in y { *class_counts.entry(label).or_insert(0) += 1; }
    let mut classes: Vec<i32> = class_counts.keys().copied().collect();
    classes.sort();
    let n_classes = classes.len();
    let class_idx: HashMap<i32, usize> = classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    // Accumulate per-class sums and sum-of-squares
    let mut sums = vec![vec![0.0f64; d]; n_classes];
    let mut sq_sums = vec![vec![0.0f64; d]; n_classes];
    let mut counts = vec![0usize; n_classes];

    for i in 0..n {
        let ci = class_idx[&y[i]];
        counts[ci] += 1;
        for j in 0..d {
            let v = x[i * d + j];
            sums[ci][j] += v;
            sq_sums[ci][j] += v * v;
        }
    }

    // Compute means and variances
    let class_prior: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
    let class_means: Vec<Vec<f64>> = (0..n_classes).map(|ci| {
        let nc = counts[ci] as f64;
        (0..d).map(|j| if nc > 0.0 { sums[ci][j] / nc } else { 0.0 }).collect()
    }).collect();
    let class_vars: Vec<Vec<f64>> = (0..n_classes).map(|ci| {
        let nc = counts[ci] as f64;
        (0..d).map(|j| {
            if nc > 1.0 {
                (sq_sums[ci][j] / nc - class_means[ci][j].powi(2)).max(0.0) + var_smoothing
            } else {
                var_smoothing
            }
        }).collect()
    }).collect();

    GaussianNB { n_classes, classes, class_prior, class_means, class_vars, n_features: d, var_smoothing }
}

/// Predict class labels for new data.
pub fn gaussian_nb_predict(model: &GaussianNB, x: &[f64], n: usize) -> Vec<i32> {
    let d = model.n_features;
    assert_eq!(x.len(), n * d);

    (0..n).map(|i| {
        let mut best_class = model.classes[0];
        let mut best_log_prob = f64::NEG_INFINITY;

        for ci in 0..model.n_classes {
            let mut log_prob = model.class_prior[ci].max(1e-300).ln();
            for j in 0..d {
                let v = x[i * d + j];
                let mu = model.class_means[ci][j];
                let var = model.class_vars[ci][j];
                // Log of Gaussian PDF: -0.5 * (ln(2πσ²) + (x-μ)²/σ²)
                log_prob += -0.5 * (std::f64::consts::TAU * var).ln()
                    - 0.5 * (v - mu).powi(2) / var;
            }
            if log_prob > best_log_prob {
                best_log_prob = log_prob;
                best_class = model.classes[ci];
            }
        }
        best_class
    }).collect()
}

/// Predict class probabilities (posterior) for new data.
/// Returns n × n_classes matrix (row-major).
pub fn gaussian_nb_predict_proba(model: &GaussianNB, x: &[f64], n: usize) -> Vec<f64> {
    let d = model.n_features;
    let k = model.n_classes;
    assert_eq!(x.len(), n * d);

    let mut proba = vec![0.0f64; n * k];
    for i in 0..n {
        let mut log_probs = vec![0.0f64; k];
        for ci in 0..k {
            let mut lp = model.class_prior[ci].max(1e-300).ln();
            for j in 0..d {
                let v = x[i * d + j];
                let mu = model.class_means[ci][j];
                let var = model.class_vars[ci][j];
                lp += -0.5 * (std::f64::consts::TAU * var).ln()
                    - 0.5 * (v - mu).powi(2) / var;
            }
            log_probs[ci] = lp;
        }
        // Log-sum-exp for numerical stability
        let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
        let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_lp).exp()).sum();
        let log_sum = max_lp + sum_exp.ln();
        for ci in 0..k {
            proba[i * k + ci] = (log_probs[ci] - log_sum).exp();
        }
    }
    proba
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_nb_separable_data() {
        // Two well-separated 2D Gaussian clusters
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = crate::rng::Xoshiro256::new(42);
        // Class 0: centered at (0, 0)
        for _ in 0..50 {
            x.push(crate::rng::sample_normal(&mut rng, 0.0, 1.0));
            x.push(crate::rng::sample_normal(&mut rng, 0.0, 1.0));
            y.push(0);
        }
        // Class 1: centered at (5, 5)
        for _ in 0..50 {
            x.push(crate::rng::sample_normal(&mut rng, 5.0, 1.0));
            x.push(crate::rng::sample_normal(&mut rng, 5.0, 1.0));
            y.push(1);
        }

        let model = gaussian_nb_fit(&x, &y, 100, 2, None);
        assert_eq!(model.n_classes, 2);
        assert_eq!(model.classes, vec![0, 1]);

        // Predict on training data — should get most right
        let preds = gaussian_nb_predict(&model, &x, 100);
        let accuracy: f64 = preds.iter().zip(&y).filter(|(p, a)| p == a).count() as f64 / 100.0;
        assert!(accuracy > 0.95, "accuracy={} should be > 0.95 for separable data", accuracy);
    }

    #[test]
    fn gaussian_nb_predict_proba_sums_to_one() {
        let x = vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0];
        let y = vec![0, 0, 1, 1];
        let model = gaussian_nb_fit(&x, &y, 4, 2, None);

        let test_x = vec![0.5, 0.5, 5.5, 5.5];
        let proba = gaussian_nb_predict_proba(&model, &test_x, 2);
        for i in 0..2 {
            let row_sum: f64 = (0..2).map(|c| proba[i * 2 + c]).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "row {} sum={}", i, row_sum);
        }
    }

    #[test]
    fn gaussian_nb_three_classes() {
        let x = vec![
            0.0, 1.0, 0.0,  0.1, 0.9, 0.1,
            1.0, 0.0, 0.0,  0.9, 0.1, 0.1,
            0.0, 0.0, 1.0,  0.1, 0.1, 0.9,
        ];
        let y = vec![0, 0, 1, 1, 2, 2];
        let model = gaussian_nb_fit(&x, &y, 6, 3, None);
        assert_eq!(model.n_classes, 3);

        let preds = gaussian_nb_predict(&model, &x, 6);
        // Should classify training data correctly
        for i in 0..6 {
            assert_eq!(preds[i], y[i], "misclassified training point {}", i);
        }
    }
}
