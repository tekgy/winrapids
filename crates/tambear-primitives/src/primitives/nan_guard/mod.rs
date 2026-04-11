//! NaN-propagating comparisons and guards.
//!
//! Rust's `f64::min`/`f64::max` follow IEEE 754-2008 minNum/maxNum semantics:
//! they return the non-NaN operand when one is NaN. This silently swallows
//! invalid data. These primitives propagate NaN instead.

/// NaN-propagating minimum. Returns NaN if either operand is NaN.
#[inline]
pub fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) }
}

/// NaN-propagating maximum. Returns NaN if either operand is NaN.
#[inline]
pub fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
}

/// Returns true if any element is NaN.
#[inline]
pub fn has_nan(data: &[f64]) -> bool {
    data.iter().any(|v| v.is_nan())
}

/// Returns true if any element is non-finite (NaN or Inf).
#[inline]
pub fn has_non_finite(data: &[f64]) -> bool {
    data.iter().any(|v| !v.is_finite())
}

/// Filter to finite values only, preserving order.
#[inline]
pub fn finite_only(data: &[f64]) -> Vec<f64> {
    data.iter().copied().filter(|v| v.is_finite()).collect()
}

/// Sort using total_cmp (NaN-safe, deterministic ordering).
/// NaN sorts after +Inf.
#[inline]
pub fn sorted_total(data: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

/// Sort, excluding NaN. Returns sorted finite+infinite values.
#[inline]
pub fn sorted_finite(data: &[f64]) -> Vec<f64> {
    let mut clean: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    clean.sort_by(|a, b| a.total_cmp(b));
    clean
}

#[cfg(test)]
mod tests;
