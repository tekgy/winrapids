//! Weighted mean: Σ wᵢxᵢ / Σ wᵢ.
//!
//! Each observation has an importance weight. Generalizes arithmetic
//! mean (all weights equal). Used for: survey data, importance sampling,
//! exponential smoothing coefficients, portfolio returns.
//!
//! # Kingdom
//! A — accumulate(All, w*x, Add) and accumulate(All, w, Add), then divide.

/// Weighted mean. Returns NaN if weights sum to zero or inputs are empty.
pub fn mean_weighted(data: &[f64], weights: &[f64]) -> f64 {
    if data.is_empty() || data.len() != weights.len() { return f64::NAN; }
    if crate::nan_guard::has_nan(data) || crate::nan_guard::has_nan(weights) { return f64::NAN; }

    let w_sum: f64 = weights.iter().sum();
    if w_sum.abs() < 1e-300 { return f64::NAN; }

    let wx_sum: f64 = data.iter().zip(weights.iter()).map(|(&x, &w)| w * x).sum();
    wx_sum / w_sum
}

#[cfg(test)]
mod tests;
