//! # NaN Guard — Tambear's fifth invariant
//!
//! "Tam doesn't check for NaN. Tam knows the data is clean."
//!
//! NaN is an input boundary concern. Once data enters tambear, it is guaranteed
//! NaN-free. Every function can trust every other function. No per-function NaN
//! checks. No defensive `is_nan()` calls in inner loops. No silent garbage.
//!
//! ## How it works
//!
//! Data enters tambear through exactly three doors:
//! 1. `Frame::add_column()` — GPU/CPU buffer ingestion
//! 2. `TamPipeline::from_slice()` — direct f64 slice
//! 3. Standalone functions (`kaplan_meier`, `mean`, etc.) — raw slices
//!
//! Each door calls `NanGuard::check()` or `NanGuard::clean()`. After the door,
//! NaN does not exist.
//!
//! ## Policy
//!
//! - `Reject` — return error if any NaN found (strict, no data loss)
//! - `Omit` — remove NaN entries, report count (default for most pipelines)
//! - `Replace(f64)` — replace NaN with a value (e.g., 0.0, mean, sentinel)

/// What to do when NaN is found at the boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NanPolicy {
    /// Error if any NaN detected. Zero tolerance. No surprises.
    Reject,
    /// Silently omit NaN entries. Return cleaned data + count of removed.
    Omit,
    /// Replace NaN with the given value.
    Replace(f64),
}

impl Default for NanPolicy {
    fn default() -> Self {
        NanPolicy::Omit
    }
}

/// Result of a NaN guard check.
#[derive(Debug, Clone)]
pub struct NanReport {
    /// Number of NaN values found.
    pub nan_count: usize,
    /// Total values inspected.
    pub total: usize,
    /// Indices of NaN values (if policy is Omit or Replace).
    pub nan_indices: Vec<usize>,
}

impl NanReport {
    /// True if no NaN was found.
    pub fn is_clean(&self) -> bool {
        self.nan_count == 0
    }

    /// Fraction of data that was NaN.
    pub fn nan_fraction(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.nan_count as f64 / self.total as f64 }
    }
}

/// Check a slice for NaN and return a report.
pub fn check(data: &[f64]) -> NanReport {
    let mut nan_indices = Vec::new();
    for (i, &v) in data.iter().enumerate() {
        if v.is_nan() {
            nan_indices.push(i);
        }
    }
    NanReport {
        nan_count: nan_indices.len(),
        total: data.len(),
        nan_indices,
    }
}

/// Apply NaN policy to a slice. Returns cleaned data + report.
///
/// - `Reject`: returns Err if any NaN found.
/// - `Omit`: returns Ok with NaN entries removed.
/// - `Replace(v)`: returns Ok with NaN replaced by v.
pub fn clean(data: &[f64], policy: NanPolicy) -> Result<(Vec<f64>, NanReport), NanReport> {
    let report = check(data);
    if report.is_clean() {
        return Ok((data.to_vec(), report));
    }

    match policy {
        NanPolicy::Reject => Err(report),
        NanPolicy::Omit => {
            let cleaned: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
            Ok((cleaned, report))
        }
        NanPolicy::Replace(fill) => {
            let cleaned: Vec<f64> = data.iter().map(|&v| if v.is_nan() { fill } else { v }).collect();
            Ok((cleaned, report))
        }
    }
}

/// Clean parallel arrays (e.g., times + events in survival analysis).
/// Omits rows where ANY array has NaN. All arrays must have the same length.
pub fn clean_parallel(arrays: &[&[f64]], policy: NanPolicy) -> Result<(Vec<Vec<f64>>, NanReport), NanReport> {
    if arrays.is_empty() {
        return Ok((Vec::new(), NanReport { nan_count: 0, total: 0, nan_indices: Vec::new() }));
    }
    let n = arrays[0].len();
    for a in arrays {
        assert_eq!(a.len(), n, "clean_parallel: all arrays must have same length");
    }

    // Find rows where ANY value is NaN
    let mut nan_rows = Vec::new();
    for i in 0..n {
        if arrays.iter().any(|a| a[i].is_nan()) {
            nan_rows.push(i);
        }
    }

    let report = NanReport {
        nan_count: nan_rows.len(),
        total: n,
        nan_indices: nan_rows.clone(),
    };

    if report.is_clean() {
        let cleaned: Vec<Vec<f64>> = arrays.iter().map(|a| a.to_vec()).collect();
        return Ok((cleaned, report));
    }

    match policy {
        NanPolicy::Reject => Err(report),
        NanPolicy::Omit => {
            let nan_set: std::collections::HashSet<usize> = nan_rows.into_iter().collect();
            let cleaned: Vec<Vec<f64>> = arrays.iter().map(|a| {
                a.iter().enumerate()
                    .filter(|(i, _)| !nan_set.contains(i))
                    .map(|(_, &v)| v)
                    .collect()
            }).collect();
            Ok((cleaned, report))
        }
        NanPolicy::Replace(fill) => {
            let nan_set: std::collections::HashSet<usize> = nan_rows.into_iter().collect();
            let cleaned: Vec<Vec<f64>> = arrays.iter().map(|a| {
                a.iter().enumerate()
                    .map(|(i, &v)| if nan_set.contains(&i) { fill } else { v })
                    .collect()
            }).collect();
            Ok((cleaned, report))
        }
    }
}

/// Quick check: does this slice contain any NaN? No allocation.
#[inline]
pub fn has_nan(data: &[f64]) -> bool {
    data.iter().any(|v| v.is_nan())
}

/// Sort a slice using total_cmp, automatically excluding NaN.
/// Returns sorted finite values only. This is the canonical "tambear sort"
/// that replaces every `sort_by(partial_cmp().unwrap())` pattern.
pub fn sorted_finite(data: &[f64]) -> Vec<f64> {
    let mut clean: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    clean.sort_by(|a, b| a.total_cmp(b));
    clean
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_clean_data() {
        let data = vec![1.0, 2.0, 3.0];
        let report = check(&data);
        assert!(report.is_clean());
        assert_eq!(report.nan_count, 0);
    }

    #[test]
    fn check_nan_data() {
        let data = vec![1.0, f64::NAN, 3.0, f64::NAN];
        let report = check(&data);
        assert!(!report.is_clean());
        assert_eq!(report.nan_count, 2);
        assert_eq!(report.nan_indices, vec![1, 3]);
    }

    #[test]
    fn clean_reject() {
        let data = vec![1.0, f64::NAN, 3.0];
        let result = clean(&data, NanPolicy::Reject);
        assert!(result.is_err());
    }

    #[test]
    fn clean_omit() {
        let data = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
        let (cleaned, report) = clean(&data, NanPolicy::Omit).unwrap();
        assert_eq!(cleaned, vec![1.0, 3.0, 5.0]);
        assert_eq!(report.nan_count, 2);
    }

    #[test]
    fn clean_replace() {
        let data = vec![1.0, f64::NAN, 3.0];
        let (cleaned, report) = clean(&data, NanPolicy::Replace(0.0)).unwrap();
        assert_eq!(cleaned, vec![1.0, 0.0, 3.0]);
        assert_eq!(report.nan_count, 1);
    }

    #[test]
    fn clean_parallel_omit() {
        let times = vec![1.0, f64::NAN, 3.0, 4.0];
        let values = vec![10.0, 20.0, f64::NAN, 40.0];
        let (cleaned, report) = clean_parallel(&[&times, &values], NanPolicy::Omit).unwrap();
        // Rows 1 and 2 have NaN in at least one array
        assert_eq!(cleaned[0], vec![1.0, 4.0]);
        assert_eq!(cleaned[1], vec![10.0, 40.0]);
        assert_eq!(report.nan_count, 2);
    }

    #[test]
    fn has_nan_true() {
        assert!(has_nan(&[1.0, f64::NAN]));
    }

    #[test]
    fn has_nan_false() {
        assert!(!has_nan(&[1.0, 2.0, f64::INFINITY]));
    }

    #[test]
    fn sorted_finite_strips_nan() {
        let data = vec![3.0, f64::NAN, 1.0, f64::NAN, 2.0];
        assert_eq!(sorted_finite(&data), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn empty_data() {
        let report = check(&[]);
        assert!(report.is_clean());
        assert_eq!(report.nan_count, 0);
    }

    #[test]
    fn all_nan() {
        let data = vec![f64::NAN, f64::NAN, f64::NAN];
        let (cleaned, report) = clean(&data, NanPolicy::Omit).unwrap();
        assert!(cleaned.is_empty());
        assert_eq!(report.nan_count, 3);
    }
}
