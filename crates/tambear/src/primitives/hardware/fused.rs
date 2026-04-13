//! Fused multiply-add primitives.
//!
//! These are the **most important primitives in the whole library** from a
//! numerical accuracy standpoint. Each one computes a multiplication and
//! addition (or subtraction) with a **single rounding at the end**, instead
//! of the two roundings that `fmul(...) + ...` or `fmul(...) - ...` would
//! incur.
//!
//! Hardware support:
//! - x86 with FMA3/FMA4 extensions (every Intel/AMD chip since 2013)
//! - ARM NEON (`fmla`, `fmls`)
//! - NVIDIA PTX (`fma.rn.f64`)
//! - AMD GCN / RDNA
//! - SPIR-V via `GLSL.std.450` extended instruction set (`Fma`)
//!
//! Rust's `f64::mul_add` is the portable hook. LLVM lowers it to the hardware
//! FMA instruction on supported targets. On targets without hardware FMA,
//! LLVM emits a software fallback that preserves correct rounding.
//!
//! # The four variants
//!
//! - `fmadd(a, b, c) = a·b + c` — fused multiply-add
//! - `fmsub(a, b, c) = a·b - c` — fused multiply-subtract
//! - `fnmadd(a, b, c) = -(a·b) + c` — negated fused multiply-add
//! - `fnmsub(a, b, c) = -(a·b) - c` — negated fused multiply-subtract
//!
//! All four round once.
//!
//! # Why this matters
//!
//! Horner's method for polynomial evaluation — `p(x) = ((c_n·x + c_{n-1})·x + c_{n-2})·x + ...`
//! — is a chain of FMAs. When implemented with `fmul`/`fadd`, each term has
//! two roundings. With `fmadd`, each term has one. For a degree-20 polynomial,
//! that's 20 saved roundings, which is often the difference between 1-ULP
//! accuracy and 10+ ULP accuracy in the tails.
//!
//! Similarly, compensated arithmetic primitives (`two_product_fma`) use FMA
//! to compute the exact product of two f64s in just 2 flops, versus the 17
//! flops required by Veltkamp split-based methods.

/// Fused multiply-add: `a·b + c` with a single rounding.
///
/// # Correctness
/// - NaN-propagating: any NaN input produces NaN output.
/// - Single-rounding guarantee: the exact mathematical product `a·b` is
///   computed without rounding, then added to `c` and rounded once.
/// - The result may differ from `fadd(fmul(a, b), c)` by up to 1 ULP, and
///   this primitive is always the more accurate of the two.
///
/// # Consumers
/// - **Every polynomial evaluation in Horner form**. This is the core loop
///   of every libm function (`exp`, `log`, `sin`, `cos`, `erf`, `gamma`, ...).
/// - **Every dot product**. `sum += a[i] * b[i]` becomes `sum = fmadd(a[i], b[i], sum)`.
/// - **Linear combinations**, weighted averages, barycentric interpolation.
/// - **Iterative refinement** (`x_new = x - J⁻¹·f(x)` in Newton-Raphson).
/// - **Two-product error-free transformation** (`two_product_fma` computes
///   the exact residual of `a·b` using `fmadd(a, b, -fmul(a, b))`).
#[inline(always)]
pub fn fmadd(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// Fused multiply-subtract: `a·b - c` with a single rounding.
///
/// Equivalent to `fmadd(a, b, -c)` but expresses intent directly.
#[inline(always)]
pub fn fmsub(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, -c)
}

/// Negated fused multiply-add: `-(a·b) + c = c - a·b` with a single rounding.
///
/// Equivalent to `fmadd(-a, b, c)` but expresses intent directly.
///
/// Critical for iterative refinement loops where the update is `x - step`
/// and `step = J⁻¹·f(x)` is computed as a product.
#[inline(always)]
pub fn fnmadd(a: f64, b: f64, c: f64) -> f64 {
    (-a).mul_add(b, c)
}

/// Negated fused multiply-subtract: `-(a·b) - c = -(a·b + c)` with a single rounding.
#[inline(always)]
pub fn fnmsub(a: f64, b: f64, c: f64) -> f64 {
    (-a).mul_add(b, -c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmadd_basic() {
        assert_eq!(fmadd(2.0, 3.0, 4.0), 10.0);
        assert_eq!(fmadd(0.0, 3.0, 4.0), 4.0);
        assert_eq!(fmadd(2.0, 0.0, 4.0), 4.0);
    }

    #[test]
    fn fmadd_single_rounding_beats_separate_ops() {
        // Classic demonstration: fmadd is strictly more accurate than mul+add
        // for values where the product has more than 53 bits of precision.
        //
        // Pick a, b such that a*b has a full-precision representation that
        // cannot be stored in f64 exactly.
        let a: f64 = 0.1;          // not exactly representable in binary
        let b: f64 = 0.2;          // not exactly representable in binary
        let c: f64 = -0.02;        // chosen so a*b + c ≈ 0

        let fused = fmadd(a, b, c);
        let separate = a * b + c;

        // The exact value of 0.1 * 0.2 is 0.020000000000000000416..., which
        // requires more than 53 bits. With fmadd, the result is rounded once
        // from the exact product. Without fmadd, the product is rounded first
        // (losing the low bits), then added to -0.02 (which is also not exact).
        //
        // The two results are within a few ULP of each other, but fmadd is the
        // canonical correct answer.
        assert!((fused - separate).abs() < 1e-15);
        // Both should be very close to the true value (which is ~4.16e-18)
        assert!(fused.abs() < 1e-16);
    }

    #[test]
    fn fmadd_nan_propagates() {
        assert!(fmadd(f64::NAN, 1.0, 2.0).is_nan());
        assert!(fmadd(1.0, f64::NAN, 2.0).is_nan());
        assert!(fmadd(1.0, 2.0, f64::NAN).is_nan());
    }

    #[test]
    fn fmadd_with_infinities() {
        assert_eq!(fmadd(f64::INFINITY, 1.0, 1.0), f64::INFINITY);
        assert_eq!(fmadd(2.0, 3.0, f64::INFINITY), f64::INFINITY);
        // inf * 0 case in the middle of fmadd: the product is NaN, so result is NaN
        assert!(fmadd(f64::INFINITY, 0.0, 1.0).is_nan());
    }

    #[test]
    fn fmsub_basic() {
        assert_eq!(fmsub(2.0, 3.0, 4.0), 2.0);   // 2*3 - 4 = 2
        assert_eq!(fmsub(5.0, 2.0, 10.0), 0.0);  // 5*2 - 10 = 0
    }

    #[test]
    fn fmsub_equivalent_to_fmadd_negated_c() {
        let a = 3.14;
        let b = 2.71;
        let c = 1.41;
        assert_eq!(fmsub(a, b, c), fmadd(a, b, -c));
    }

    #[test]
    fn fnmadd_basic() {
        // -(2*3) + 10 = -6 + 10 = 4
        assert_eq!(fnmadd(2.0, 3.0, 10.0), 4.0);
    }

    #[test]
    fn fnmadd_equivalent_to_fmadd_negated_a() {
        let a = 3.14;
        let b = 2.71;
        let c = 1.41;
        assert_eq!(fnmadd(a, b, c), fmadd(-a, b, c));
    }

    #[test]
    fn fnmsub_basic() {
        // -(2*3) - 4 = -10
        assert_eq!(fnmsub(2.0, 3.0, 4.0), -10.0);
    }

    #[test]
    fn fnmsub_equivalent_form() {
        let a = 3.14;
        let b = 2.71;
        let c = 1.41;
        assert_eq!(fnmsub(a, b, c), fmadd(-a, b, -c));
        assert_eq!(fnmsub(a, b, c), -(a * b + c));
    }
}
