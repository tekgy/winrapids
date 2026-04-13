//! IEEE 754 binary arithmetic primitives.
//!
//! Every function here compiles to a single hardware instruction on every
//! target tambear supports. The bodies use Rust's native operators; LLVM
//! emits the appropriate machine code.
//!
//! These are terminal operations — their single rounding is the last rounding
//! they perform. Recipes that need higher precision than f64 round-per-op
//! must compose these via the compensated arithmetic foundations
//! (`primitives::compensated`) or the `DoubleDouble` type
//! (`primitives::double_double`).

/// IEEE 754 binary64 addition with correct rounding (round-to-nearest ties-to-even).
///
/// # Properties
/// - Commutative: `fadd(a, b) == fadd(b, a)`.
/// - NOT associative in general: `fadd(fadd(a, b), c) != fadd(a, fadd(b, c))` when rounding differs.
/// - NaN-propagating: if either input is NaN, result is NaN.
/// - `fadd(x, 0.0) == x` except `fadd(-0.0, 0.0) == 0.0` (sign rule).
///
/// # Consumers
/// Every summation, every linear combination, every polynomial evaluation via
/// Horner's method. The hot path of essentially every recipe.
#[inline(always)]
pub fn fadd(a: f64, b: f64) -> f64 {
    a + b
}

/// IEEE 754 binary64 subtraction with correct rounding.
///
/// # Properties
/// - Anti-commutative: `fsub(a, b) == -fsub(b, a)`.
/// - NaN-propagating.
/// - **Cancellation-prone**: `fsub(a, b)` for `a ≈ b` loses precision dramatically.
///   Recipes that subtract nearly-equal values should consider a compensated
///   alternative (`two_diff`) to preserve the rounding error for downstream use.
///
/// # Consumers
/// Centered moments (x - mean), residuals (y - y_hat), difference formulas,
/// iterative refinement.
#[inline(always)]
pub fn fsub(a: f64, b: f64) -> f64 {
    a - b
}

/// IEEE 754 binary64 multiplication with correct rounding.
///
/// # Properties
/// - Commutative.
/// - NOT associative in general (rounding).
/// - NaN-propagating.
/// - `fmul(0.0, inf) == NaN`, `fmul(-0.0, x) == -0.0 * x` with sign rules.
///
/// # Consumers
/// Inner products, covariances, polynomial terms, scaling operations.
#[inline(always)]
pub fn fmul(a: f64, b: f64) -> f64 {
    a * b
}

/// IEEE 754 binary64 division with correct rounding.
///
/// # Properties
/// - NOT commutative.
/// - NaN-propagating.
/// - `fdiv(x, 0.0)` returns ±infinity (sign determined by x), `fdiv(0.0, 0.0) == NaN`.
///
/// # Consumers
/// Normalization (x / sum), ratios (count / total), reciprocals,
/// Newton-step updates.
///
/// # Note
/// Division is typically ~5-20x slower than multiplication on modern hardware.
/// Hot loops that compute many ratios with the same divisor should compute
/// `1.0 / divisor` once and multiply, rather than dividing repeatedly.
#[inline(always)]
pub fn fdiv(a: f64, b: f64) -> f64 {
    a / b
}

/// IEEE 754 binary64 square root with correct rounding.
///
/// # Properties
/// - Monotonic: `a < b` implies `fsqrt(a) <= fsqrt(b)` (for non-negative inputs).
/// - NaN-propagating.
/// - `fsqrt(x)` for `x < 0.0` returns NaN. `fsqrt(-0.0) == -0.0`, `fsqrt(+0.0) == +0.0`.
/// - `fsqrt(inf) == inf`.
///
/// # Consumers
/// Standard deviation, Euclidean norm, normalization, geometric mean
/// (via `exp(0.5 * log(x²))` or direct), Pythagorean-style distances.
#[inline(always)]
pub fn fsqrt(a: f64) -> f64 {
    a.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fadd_basic() {
        assert_eq!(fadd(1.0, 2.0), 3.0);
        assert_eq!(fadd(-1.5, 1.5), 0.0);
    }

    #[test]
    fn fadd_identity() {
        assert_eq!(fadd(42.0, 0.0), 42.0);
        assert_eq!(fadd(0.0, 42.0), 42.0);
    }

    #[test]
    fn fadd_nan_propagates() {
        assert!(fadd(f64::NAN, 1.0).is_nan());
        assert!(fadd(1.0, f64::NAN).is_nan());
        assert!(fadd(f64::NAN, f64::NAN).is_nan());
    }

    #[test]
    fn fadd_infinities() {
        assert_eq!(fadd(f64::INFINITY, 1.0), f64::INFINITY);
        assert_eq!(fadd(f64::NEG_INFINITY, 1.0), f64::NEG_INFINITY);
        // +inf + -inf = NaN (IEEE 754)
        assert!(fadd(f64::INFINITY, f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn fsub_basic() {
        assert_eq!(fsub(3.0, 2.0), 1.0);
        assert_eq!(fsub(1.0, 1.0), 0.0);
    }

    #[test]
    fn fsub_cancellation() {
        // Demonstrate catastrophic cancellation — this is the motivation for
        // compensated arithmetic alternatives.
        let a = 1.0 + 1e-15;
        let b = 1.0;
        let result = fsub(a, b);
        // We get ~1e-15 but the last bits are garbage.
        assert!((result - 1e-15).abs() < 1e-14);
    }

    #[test]
    fn fsub_nan_propagates() {
        assert!(fsub(f64::NAN, 1.0).is_nan());
        assert!(fsub(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn fmul_basic() {
        assert_eq!(fmul(3.0, 4.0), 12.0);
        assert_eq!(fmul(-2.0, 3.0), -6.0);
    }

    #[test]
    fn fmul_zero() {
        assert_eq!(fmul(42.0, 0.0), 0.0);
        assert_eq!(fmul(0.0, 42.0), 0.0);
    }

    #[test]
    fn fmul_zero_times_infinity_is_nan() {
        assert!(fmul(0.0, f64::INFINITY).is_nan());
        assert!(fmul(f64::INFINITY, 0.0).is_nan());
    }

    #[test]
    fn fmul_nan_propagates() {
        assert!(fmul(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn fdiv_basic() {
        assert_eq!(fdiv(10.0, 2.0), 5.0);
        assert_eq!(fdiv(1.0, 4.0), 0.25);
    }

    #[test]
    fn fdiv_by_zero() {
        assert_eq!(fdiv(1.0, 0.0), f64::INFINITY);
        assert_eq!(fdiv(-1.0, 0.0), f64::NEG_INFINITY);
        assert!(fdiv(0.0, 0.0).is_nan());
    }

    #[test]
    fn fdiv_nan_propagates() {
        assert!(fdiv(f64::NAN, 1.0).is_nan());
        assert!(fdiv(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn fsqrt_basic() {
        assert_eq!(fsqrt(4.0), 2.0);
        assert_eq!(fsqrt(0.0), 0.0);
        assert_eq!(fsqrt(1.0), 1.0);
    }

    #[test]
    fn fsqrt_negative_is_nan() {
        assert!(fsqrt(-1.0).is_nan());
        assert!(fsqrt(-0.5).is_nan());
    }

    #[test]
    fn fsqrt_neg_zero() {
        // IEEE 754: sqrt(-0.0) == -0.0
        let result = fsqrt(-0.0);
        assert_eq!(result, 0.0);
        assert!(result.is_sign_negative());
    }

    #[test]
    fn fsqrt_infinity() {
        assert_eq!(fsqrt(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn fsqrt_nan_propagates() {
        assert!(fsqrt(f64::NAN).is_nan());
    }
}
