//! Trimmed mean: discard α fraction from each tail, average the rest.
//!
//! Robust to outliers. With α=0 it's the arithmetic mean.
//! With α=0.5 it's the median. Tunable tradeoff between
//! efficiency (arithmetic) and robustness (median).
//!
//! # Parameters
//! - `alpha`: fraction to trim from each tail (0..0.5)
//!
//! # Kingdom
//! A — sort, then accumulate(All, x, Add) over the middle portion.

/// Trimmed mean. Trims `alpha` fraction from each end.
///
/// - alpha = 0.0 → arithmetic mean
/// - alpha = 0.25 → interquartile mean
/// - alpha → 0.5 → approaches median
pub fn mean_trimmed(data: &[f64], alpha: f64) -> f64 {
    if data.is_empty() { return f64::NAN; }
    if alpha < 0.0 || alpha >= 0.5 || alpha.is_nan() { return f64::NAN; }
    if crate::nan_guard::has_nan(data) { return f64::NAN; }

    let sorted = crate::nan_guard::sorted_total(data);
    let n = sorted.len();
    let trim = (n as f64 * alpha).floor() as usize;

    if 2 * trim >= n { return f64::NAN; }

    let middle = &sorted[trim..n - trim];
    middle.iter().sum::<f64>() / middle.len() as f64
}

#[cfg(test)]
mod tests;
