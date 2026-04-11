//! Winsorized mean: clamp extremes to boundary values, then average.
//!
//! Unlike trimmed mean (which discards), winsorized mean REPLACES
//! extreme values with the nearest non-extreme value. Preserves n.
//!
//! # Parameters
//! - `alpha`: fraction to winsorize from each tail (0..0.5)
//!
//! # Kingdom
//! A — sort, clamp, then accumulate(All, x, Add) / n.

/// Winsorized mean. Clamps `alpha` fraction from each end to boundary values.
pub fn mean_winsorized(data: &[f64], alpha: f64) -> f64 {
    if data.is_empty() { return f64::NAN; }
    if alpha < 0.0 || alpha >= 0.5 || alpha.is_nan() { return f64::NAN; }
    if crate::nan_guard::has_nan(data) { return f64::NAN; }

    let mut sorted = crate::nan_guard::sorted_total(data);
    let n = sorted.len();
    let k = (n as f64 * alpha).floor() as usize;

    if k > 0 {
        let low = sorted[k];
        let high = sorted[n - 1 - k];
        for i in 0..k { sorted[i] = low; }
        for i in (n - k)..n { sorted[i] = high; }
    }

    sorted.iter().sum::<f64>() / n as f64
}

#[cfg(test)]
mod tests;
