//! Power-of-two scaling primitives: `ldexp` and `frexp`.
//!
//! These manipulate the exponent field of an f64 directly, without performing
//! general multiplication or division. They are the building blocks for
//! range-reduction / reconstruction steps in libm recipes like `exp`, where
//! `exp(x) = 2^k * exp(r)` requires multiplying by an integer power of two
//! after reducing `x` into a small range.
//!
//! # Why they are primitives
//!
//! `ldexp(x, k)` is equivalent to `x * 2.0_f64.powi(k)` mathematically, but
//! the primitive form:
//! - Is exact when the result is representable (no rounding error from the
//!   multiply path).
//! - Handles subnormals and overflow uniformly.
//! - Compiles to a single integer add on the exponent bits for normal inputs.
//!
//! `frexp(x)` is the inverse: split `x` into a normalized mantissa in
//! `[0.5, 1.0)` and an integer exponent such that `x == mantissa * 2^exp`.
//! Used in log recipes for initial range reduction.

/// Scale `x` by `2^exp`. Equivalent to `x * 2^exp` but exact for normal results.
///
/// # Special cases
/// - `ldexp(0.0, _) == 0.0` (preserving sign of zero)
/// - `ldexp(NaN, _) == NaN`
/// - `ldexp(±inf, _) == ±inf`
/// - Overflow → ±inf; underflow → subnormal or ±0
#[inline(always)]
pub fn ldexp(x: f64, exp: i32) -> f64 {
    // Rust's std uses a multiply-by-power-of-two path that handles all edge
    // cases correctly. For normal inputs in range, this compiles to a single
    // exponent adjustment on most targets.
    //
    // We clamp `exp` into a safe range and split large shifts into two halves
    // to avoid overflow in the intermediate power computation. This mirrors
    // the libm `ldexp` reference implementation.
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let mut e = exp;
    let mut result = x;
    // Clamp huge exponents by repeated scaling. 2^1023 is the largest exact
    // power of two representable as a normal f64.
    while e > 1023 {
        result *= f64::from_bits(0x7FE_u64 << 52); // 2^1023
        e -= 1023;
    }
    while e < -1022 {
        // Use 2^-1022 (smallest normal) to avoid subnormal intermediate loss.
        result *= f64::from_bits(0x001_u64 << 52); // 2^-1022
        e += 1022;
    }
    // e is now in [-1022, 1023]; one final scale is safe.
    let scale = f64::from_bits(((e + 1023) as u64) << 52);
    result * scale
}

/// Split `x` into a normalized mantissa `m ∈ [0.5, 1.0)` and an integer
/// exponent `e` such that `x == m * 2^e`.
///
/// # Special cases
/// - `frexp(0.0) == (0.0, 0)` (preserving sign)
/// - `frexp(-0.0) == (-0.0, 0)`
/// - `frexp(NaN) == (NaN, 0)` (exponent unspecified, we return 0)
/// - `frexp(±inf) == (±inf, 0)`
/// - Subnormals are normalized first: the returned `m` is always in `[0.5, 1.0)`.
#[inline(always)]
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let mut exp = ((bits >> 52) & 0x7FF) as i32;
    if exp == 0 {
        // Subnormal: normalize by multiplying by 2^54, then adjust.
        let normalized = x * f64::from_bits(0x435_u64 << 52); // 2^54
        let nb = normalized.to_bits();
        exp = ((nb >> 52) & 0x7FF) as i32 - 1023 - 54 + 1;
        // Construct mantissa with biased exponent 1022 (so value in [0.5, 1)).
        let m_bits = (nb & 0x800F_FFFF_FFFF_FFFF) | (1022_u64 << 52);
        (f64::from_bits(m_bits), exp)
    } else {
        let biased_exp = exp - 1022;
        let m_bits = (bits & 0x800F_FFFF_FFFF_FFFF) | (1022_u64 << 52);
        (f64::from_bits(m_bits), biased_exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldexp_basic() {
        assert_eq!(ldexp(1.0, 0), 1.0);
        assert_eq!(ldexp(1.0, 1), 2.0);
        assert_eq!(ldexp(1.0, 10), 1024.0);
        assert_eq!(ldexp(1.5, 2), 6.0);
        assert_eq!(ldexp(1.0, -1), 0.5);
        assert_eq!(ldexp(1.0, -10), 1.0 / 1024.0);
    }

    #[test]
    fn ldexp_preserves_sign_of_zero() {
        let pos = ldexp(0.0, 5);
        assert_eq!(pos, 0.0);
        assert!(pos.is_sign_positive());

        let neg = ldexp(-0.0, 5);
        assert_eq!(neg, 0.0);
        assert!(neg.is_sign_negative());
    }

    #[test]
    fn ldexp_nan_and_inf() {
        assert!(ldexp(f64::NAN, 3).is_nan());
        assert_eq!(ldexp(f64::INFINITY, 3), f64::INFINITY);
        assert_eq!(ldexp(f64::NEG_INFINITY, 3), f64::NEG_INFINITY);
    }

    #[test]
    fn ldexp_negative_values() {
        assert_eq!(ldexp(-1.0, 3), -8.0);
        assert_eq!(ldexp(-1.5, 2), -6.0);
    }

    #[test]
    fn ldexp_large_positive_exp_overflows_to_inf() {
        assert_eq!(ldexp(1.0, 2000), f64::INFINITY);
    }

    #[test]
    fn ldexp_large_negative_exp_underflows_to_zero() {
        assert_eq!(ldexp(1.0, -2000), 0.0);
    }

    #[test]
    fn ldexp_roundtrip_with_frexp() {
        let values = [1.0, 3.14, -2.71, 1e100, 1e-100, 1024.0, 0.125];
        for &x in &values {
            let (m, e) = frexp(x);
            let reconstructed = ldexp(m, e);
            assert_eq!(reconstructed, x, "frexp/ldexp roundtrip failed for {x:e}");
        }
    }

    // ── frexp tests ─────────────────────────────────────────────────────────

    #[test]
    fn frexp_basic() {
        let (m, e) = frexp(1.0);
        assert_eq!(m, 0.5);
        assert_eq!(e, 1);

        let (m, e) = frexp(2.0);
        assert_eq!(m, 0.5);
        assert_eq!(e, 2);

        let (m, e) = frexp(3.0);
        assert_eq!(m, 0.75);
        assert_eq!(e, 2);

        let (m, e) = frexp(0.5);
        assert_eq!(m, 0.5);
        assert_eq!(e, 0);
    }

    #[test]
    fn frexp_mantissa_always_in_range() {
        let values = [1.0, 3.14, -2.71, 1e100, 1e-100, 1024.0, 0.125, -7.5];
        for &x in &values {
            let (m, _) = frexp(x);
            let abs_m = m.abs();
            assert!(
                (0.5..1.0).contains(&abs_m),
                "frexp({x:e}) mantissa {m} not in [0.5, 1.0)"
            );
        }
    }

    #[test]
    fn frexp_zero() {
        let (m, e) = frexp(0.0);
        assert_eq!(m, 0.0);
        assert_eq!(e, 0);
        assert!(m.is_sign_positive());

        let (m, e) = frexp(-0.0);
        assert_eq!(m, 0.0);
        assert_eq!(e, 0);
        assert!(m.is_sign_negative());
    }

    #[test]
    fn frexp_nan_and_inf() {
        let (m, _) = frexp(f64::NAN);
        assert!(m.is_nan());

        let (m, _) = frexp(f64::INFINITY);
        assert_eq!(m, f64::INFINITY);

        let (m, _) = frexp(f64::NEG_INFINITY);
        assert_eq!(m, f64::NEG_INFINITY);
    }

    #[test]
    fn frexp_negative_values() {
        let (m, e) = frexp(-1.0);
        assert_eq!(m, -0.5);
        assert_eq!(e, 1);

        let (m, e) = frexp(-6.0);
        assert_eq!(m, -0.75);
        assert_eq!(e, 3);
    }

    #[test]
    fn frexp_subnormal() {
        // Smallest positive subnormal: 2^-1074
        let x = f64::from_bits(1);
        let (m, e) = frexp(x);
        let abs_m = m.abs();
        assert!((0.5..1.0).contains(&abs_m));
        // Reconstruct should give us back x.
        assert_eq!(ldexp(m, e), x);
    }

    #[test]
    fn frexp_powers_of_two() {
        for k in -10..=10_i32 {
            let x = 2.0_f64.powi(k);
            let (m, e) = frexp(x);
            assert_eq!(m, 0.5, "frexp(2^{k}) mantissa should be 0.5");
            assert_eq!(e, k + 1, "frexp(2^{k}) exponent should be {}", k + 1);
        }
    }
}
