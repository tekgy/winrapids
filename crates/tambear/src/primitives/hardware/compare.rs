//! Floating-point comparison primitives.
//!
//! IEEE 754 comparisons return ordered results for non-NaN inputs and
//! **unordered** results when either input is NaN. All comparisons involving
//! NaN (except `!=`) return `false`.
//!
//! The functions here return `bool`. For predicates that drive control flow
//! in a recipe, prefer these over `a < b` inline so the comparison is
//! explicit in the recipe's computation graph.
//!
//! # NaN semantics
//!
//! | Expression       | IEEE 754 result | Our result |
//! |------------------|-----------------|------------|
//! | `fcmp_eq(NaN, x)` | false           | false      |
//! | `fcmp_lt(NaN, x)` | false           | false      |
//! | `fcmp_le(NaN, x)` | false           | false      |
//! | `fcmp_gt(NaN, x)` | false           | false      |
//! | `fcmp_ge(NaN, x)` | false           | false      |
//!
//! In particular, `fcmp_eq(NaN, NaN) == false`. Two NaNs are never equal.

/// Equality comparison: `a == b`, false if either is NaN.
#[inline(always)]
pub fn fcmp_eq(a: f64, b: f64) -> bool {
    a == b
}

/// Strict less-than: `a < b`, false if either is NaN.
#[inline(always)]
pub fn fcmp_lt(a: f64, b: f64) -> bool {
    a < b
}

/// Less-than or equal: `a <= b`, false if either is NaN.
#[inline(always)]
pub fn fcmp_le(a: f64, b: f64) -> bool {
    a <= b
}

/// Strict greater-than: `a > b`, false if either is NaN.
#[inline(always)]
pub fn fcmp_gt(a: f64, b: f64) -> bool {
    a > b
}

/// Greater-than or equal: `a >= b`, false if either is NaN.
#[inline(always)]
pub fn fcmp_ge(a: f64, b: f64) -> bool {
    a >= b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_comparisons() {
        assert!(fcmp_eq(1.0, 1.0));
        assert!(!fcmp_eq(1.0, 2.0));
        assert!(fcmp_lt(1.0, 2.0));
        assert!(!fcmp_lt(2.0, 1.0));
        assert!(fcmp_le(1.0, 1.0));
        assert!(fcmp_le(1.0, 2.0));
        assert!(fcmp_gt(2.0, 1.0));
        assert!(fcmp_ge(2.0, 2.0));
    }

    #[test]
    fn nan_comparisons_all_false() {
        let n = f64::NAN;
        assert!(!fcmp_eq(n, 1.0));
        assert!(!fcmp_eq(1.0, n));
        assert!(!fcmp_eq(n, n));

        assert!(!fcmp_lt(n, 1.0));
        assert!(!fcmp_lt(1.0, n));
        assert!(!fcmp_lt(n, n));

        assert!(!fcmp_le(n, 1.0));
        assert!(!fcmp_le(1.0, n));

        assert!(!fcmp_gt(n, 1.0));
        assert!(!fcmp_gt(1.0, n));

        assert!(!fcmp_ge(n, 1.0));
        assert!(!fcmp_ge(1.0, n));
    }

    #[test]
    fn signed_zero_equality() {
        // IEEE 754: +0.0 == -0.0 is true
        assert!(fcmp_eq(0.0, -0.0));
        assert!(fcmp_le(0.0, -0.0));
        assert!(fcmp_ge(0.0, -0.0));
        assert!(!fcmp_lt(0.0, -0.0));
        assert!(!fcmp_gt(0.0, -0.0));
    }

    #[test]
    fn infinity_comparisons() {
        assert!(fcmp_lt(1.0, f64::INFINITY));
        assert!(fcmp_gt(1.0, f64::NEG_INFINITY));
        assert!(fcmp_lt(f64::NEG_INFINITY, f64::INFINITY));
        assert!(fcmp_eq(f64::INFINITY, f64::INFINITY));
    }
}
