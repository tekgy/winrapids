//! Geometric mean: exp(Σ ln(xᵢ) / n).
//!
//! The mean of multiplicative processes. Used for growth rates,
//! ratios, log-normal data, financial returns.
//!
//! Requires all values > 0 (ln is undefined for ≤ 0).
//!
//! # Relationship to power_mean
//! This is power_mean with p → 0 (the limit).
//! The implementation uses the log-exp form directly — no pow() needed.
//!
//! # Kingdom
//! A — accumulate(All, ln(x), Add) then gather(scalar, exp(/n)).

/// Geometric mean. Returns NaN if any value ≤ 0 or input is empty.
#[inline]
pub fn mean_geometric(data: &[f64]) -> f64 {
    if data.is_empty() { return f64::NAN; }
    if data.iter().any(|&v| v <= 0.0 || v.is_nan()) { return f64::NAN; }
    let n = data.len() as f64;
    let log_sum: f64 = data.iter().map(|&v| v.ln()).sum();
    (log_sum / n).exp()
}

#[cfg(test)]
mod tests;
