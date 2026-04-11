//! Harmonic mean: n / Σ(1/xᵢ).
//!
//! The mean for rates and ratios. If you drive 60 mph one way and
//! 40 mph back, the average speed is harmonic_mean(60, 40) = 48 mph,
//! not arithmetic_mean = 50 mph.
//!
//! Used for: averaging rates, P/E ratios, F-score (harmonic mean of
//! precision and recall), parallel resistances.
//!
//! Requires all values > 0 (division by zero).
//!
//! # Relationship to power_mean
//! This is power_mean with p = -1.
//!
//! # Kingdom
//! A — accumulate(All, 1/x, Add) then gather(scalar, n/).

/// Harmonic mean. Returns NaN if any value ≤ 0 or input is empty.
#[inline]
pub fn mean_harmonic(data: &[f64]) -> f64 {
    if data.is_empty() { return f64::NAN; }
    if data.iter().any(|&v| v <= 0.0 || v.is_nan()) { return f64::NAN; }
    let n = data.len() as f64;
    let reciprocal_sum: f64 = data.iter().map(|&v| 1.0 / v).sum();
    n / reciprocal_sum
}

#[cfg(test)]
mod tests;
