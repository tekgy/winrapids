//! Quadratic mean (Root Mean Square / RMS): sqrt(Σxᵢ² / n).
//!
//! Measures the magnitude of a set of values, regardless of sign.
//! Used for: AC voltage/current, signal power, RMSE (when applied to errors),
//! standard deviation (RMS of deviations from mean).
//!
//! # Relationship to power_mean
//! This is power_mean with p = 2.
//!
//! # Kingdom
//! A — accumulate(All, x², Add) then gather(scalar, sqrt(/n)).

/// Quadratic mean (RMS). Returns NaN for empty input.
#[inline]
pub fn mean_quadratic(data: &[f64]) -> f64 {
    if data.is_empty() { return f64::NAN; }
    if crate::nan_guard::has_nan(data) { return f64::NAN; }
    let n = data.len() as f64;
    let sum_sq: f64 = data.iter().map(|&v| v * v).sum();
    (sum_sq / n).sqrt()
}

#[cfg(test)]
mod tests;
