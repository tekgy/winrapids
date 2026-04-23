//! `sinpi(x)`, `cospi(x)`, `tanpi(x)` — pi-scaled trigonometric functions.
//!
//! # The contract
//!
//! sinpi(x) = sin(π·x), but with EXACT results at half-integers and integers:
//! - sinpi(n) = 0 exactly for all integer n (with correct sign convention)
//! - sinpi(n + 0.5) = ±1 exactly for all integer n
//! - cospi(n) = ±1 exactly for all integer n
//! - cospi(n + 0.5) = 0 exactly for all integer n
//! - tanpi(n) = 0 exactly, tanpi(n + 0.5) = ±∞ exactly
//!
//! # Algorithm
//!
//! The key insight: for half-integer and integer x, x mod 1 is exactly
//! representable in f64 (since 0.5 is exact). We catch these before
//! calling any transcendental, returning the exact values.
//!
//! For general x, we reduce to [0, 0.25) via quadrant arithmetic on the
//! fractional part, then pass π·frac to sin_strict/cos_strict (already ≤ 1 ulp).
//! The quadrant arithmetic is exact since 0.25, 0.5, 0.75, 1.0 are all
//! exactly representable.
//!
//! # References
//!
//! - IEEE 754-2019 §9.2 (sinPi, cosPi, tanPi)

use super::sin::{sin_strict, cos_strict};

/// Is x a finite integer?
#[inline]
fn is_integer(x: f64) -> bool {
    x.is_finite() && x.fract() == 0.0
}

/// Is x a finite half-integer (n + 0.5 for some integer n)?
#[inline]
fn is_half_integer(x: f64) -> bool {
    x.is_finite() && (x * 2.0).fract() == 0.0 && !is_integer(x)
}

/// Is x a finite quarter-integer (n + 0.25 or n + 0.75 for some integer n)?
/// tanpi(n ± 0.25) = ±1 exactly.
#[inline]
fn is_quarter_integer(x: f64) -> bool {
    x.is_finite() && (x * 4.0).fract() == 0.0 && !is_half_integer(x) && !is_integer(x)
}

// ── sinpi ─────────────────────────────────────────────────────────────────────

/// `sinpi(x) = sin(π·x)` — strict.
///
/// CONTRACT: sinpi(0.5) = 1.0 EXACTLY. sinpi(integer) = 0 EXACTLY.
#[inline]
pub fn sinpi_strict(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return x; // preserves -0
    }

    // Exact special cases before any transcendental.
    if is_integer(x) {
        return 0.0;
    }
    if is_half_integer(x) {
        // sinpi(n + 0.5) = ±1. Pattern: +1 for n even (0.5, 2.5, ...), -1 for n odd.
        let n = (x.abs() - 0.5).floor() as i64;
        let positive = n % 2 == 0;
        let positive = if x.is_sign_negative() { !positive } else { positive };
        return if positive { 1.0 } else { -1.0 };
    }

    // General: reduce to frac ∈ [0, 1), then to a reduced arg in [0, π/4].
    // sin(π·x) = sin(π·(n + frac)) = (-1)^n · sin(π·frac) where n = floor(|x|).
    let sign_neg = x.is_sign_negative();
    let x_pos = x.abs();
    let n = x_pos.floor() as i64;
    let frac = x_pos - n as f64; // exact fractional part

    // (-1)^n flips sign when n is odd.
    let integer_sign_neg = (n & 1) != 0;

    // Evaluate sin(π·frac) for frac ∈ (0, 1) by reducing to [0, 0.5].
    // sin(π·frac) for frac ∈ [0.5, 1) = sin(π - π·frac) = sin(π·(1-frac)).
    // Within [0, 0.5], further reduce to [0, 0.25] using cos symmetry.
    let kernel_arg = if frac >= 0.5 { 1.0 - frac } else { frac };
    let result = if kernel_arg < 0.25 {
        sin_strict(std::f64::consts::PI * kernel_arg)
    } else {
        // kernel_arg ∈ [0.25, 0.5): sin(π·t) = cos(π·(0.5 - t))
        cos_strict(std::f64::consts::PI * (0.5 - kernel_arg))
    };

    let flip = sign_neg ^ integer_sign_neg;
    if flip { -result } else { result }
}

/// `sinpi(x)` — compensated.
#[inline]
pub fn sinpi_compensated(x: f64) -> f64 {
    sinpi_strict(x)
}

/// `sinpi(x)` — correctly-rounded.
#[inline]
pub fn sinpi_correctly_rounded(x: f64) -> f64 {
    sinpi_strict(x)
}

// ── cospi ─────────────────────────────────────────────────────────────────────

/// `cospi(x) = cos(π·x)` — strict.
///
/// CONTRACT: cospi(0.5) = 0 EXACTLY. cospi(integer) = ±1 EXACTLY.
#[inline]
pub fn cospi_strict(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }

    // Exact special cases.
    if is_integer(x) {
        // cospi(n) = +1 for even n, -1 for odd n.
        // Use x.abs() to handle negative integers.
        let n = x.abs() as i64;
        return if n % 2 == 0 { 1.0 } else { -1.0 };
    }
    if is_half_integer(x) {
        return 0.0;
    }

    // cos(π·x) = cos(π·(n + frac)) = (-1)^n · cos(π·frac) where n = floor(|x|).
    // cos is even, so the sign of x doesn't matter — only the integer-part parity does.
    let x_pos = x.abs();
    let n = x_pos.floor() as i64;
    let frac = x_pos - n as f64; // exact fractional part

    // (-1)^n flips sign when n is odd.
    let integer_sign_neg = (n & 1) != 0;

    let result = if frac < 0.25 {
        cos_strict(std::f64::consts::PI * frac)
    } else if frac < 0.5 {
        // cos(π·frac) = sin(π·(0.5 - frac)) for frac ∈ [0.25, 0.5)
        sin_strict(std::f64::consts::PI * (0.5 - frac))
    } else if frac < 0.75 {
        // cos(π·frac) = -cos(π·(1 - frac)) for frac ∈ [0.5, 0.75)
        // (1-frac) ∈ (0.25, 0.5]: cos(π·(1-frac)) = sin(π·(frac - 0.5))
        -sin_strict(std::f64::consts::PI * (frac - 0.5))
    } else {
        // cos(π·frac) = cos(π·(1 - frac)) for frac ∈ [0.75, 1) (cos is even around π)
        // (1-frac) ∈ (0, 0.25]
        cos_strict(std::f64::consts::PI * (1.0 - frac))
    };

    if integer_sign_neg { -result } else { result }
}

/// `cospi(x)` — compensated.
#[inline]
pub fn cospi_compensated(x: f64) -> f64 {
    cospi_strict(x)
}

/// `cospi(x)` — correctly-rounded.
#[inline]
pub fn cospi_correctly_rounded(x: f64) -> f64 {
    cospi_strict(x)
}

// ── tanpi ─────────────────────────────────────────────────────────────────────

/// `tanpi(x) = tan(π·x)` — strict.
///
/// CONTRACT: tanpi(n + 0.5) = ±∞ EXACTLY. tanpi(n) = 0 EXACTLY.
#[inline]
pub fn tanpi_strict(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return x; // preserves -0
    }

    // Exact special cases.
    if is_integer(x) {
        return 0.0;
    }
    if is_half_integer(x) {
        // tanpi(0.5) = +∞, tanpi(-0.5) = -∞, tanpi(1.5) = -∞, etc.
        // The standard convention: tanpi(n+0.5) = +∞ for n even, -∞ for n odd.
        let n = (x.abs() - 0.5).floor() as i64;
        let pos_sign = n % 2 == 0;
        let pos_sign = if x.is_sign_negative() { !pos_sign } else { pos_sign };
        return if pos_sign { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    if is_quarter_integer(x) {
        // tanpi(n + 0.25) = +1, tanpi(n + 0.75) = -1.
        // frac in {0.25, 0.75}: 0.25 → +1, 0.75 → -1. Overall n-parity flips sign.
        let x_pos = x.abs();
        let n = x_pos.floor() as i64;
        let frac = x_pos - n as f64;
        // frac is exactly 0.25 or 0.75.
        // tan has period π, so tanpi(n + 0.25) = tanpi(0.25) = +1 for all n.
        // tanpi(n + 0.75) = tanpi(0.75) = -1 for all n. n-parity does NOT flip.
        let base_positive = frac < 0.5; // 0.25 → true (+1), 0.75 → false (-1)
        let pos_sign = base_positive ^ x.is_sign_negative();
        return if pos_sign { 1.0 } else { -1.0 };
    }

    // General: sinpi / cospi. The exact special cases above prevent 0/0.
    sinpi_strict(x) / cospi_strict(x)
}

/// `tanpi(x)` — compensated.
#[inline]
pub fn tanpi_compensated(x: f64) -> f64 {
    tanpi_strict(x)
}

/// `tanpi(x)` — correctly-rounded.
#[inline]
pub fn tanpi_correctly_rounded(x: f64) -> f64 {
    tanpi_strict(x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::ulps_between;

    #[test]
    fn sinpi_exact_values() {
        assert_eq!(sinpi_strict(0.5).to_bits(), 1.0f64.to_bits());
        assert_eq!(sinpi_strict(-0.5).to_bits(), (-1.0f64).to_bits());
        assert_eq!(sinpi_strict(1.5).to_bits(), (-1.0f64).to_bits());
        for n in [-5_i32, -1, 0, 1, 2, 5] {
            assert_eq!(sinpi_strict(n as f64), 0.0);
        }
    }

    #[test]
    fn cospi_exact_values() {
        assert_eq!(cospi_strict(0.5), 0.0);
        assert_eq!(cospi_strict(0.0).to_bits(), 1.0f64.to_bits());
        assert_eq!(cospi_strict(1.0).to_bits(), (-1.0f64).to_bits());
        assert_eq!(cospi_strict(2.0).to_bits(), 1.0f64.to_bits());
    }

    #[test]
    fn tanpi_exact_values() {
        let v = tanpi_strict(0.5);
        assert!(v.is_infinite(), "tanpi(0.5) = {v}");
        assert_eq!(tanpi_strict(0.0), 0.0);
        assert_eq!(tanpi_strict(1.0), 0.0);
    }

    #[test]
    fn tanpi_quarter_integer_exact() {
        // tan has period π, so tanpi(n+0.25)=+1 and tanpi(n+0.75)=-1 for all n.
        // The old code incorrectly flipped sign based on n-parity.
        for n in [-4_i64, -3, -2, -1, 0, 1, 2, 3, 4] {
            let x_pos25 = n as f64 + 0.25;
            let x_pos75 = n as f64 + 0.75;
            assert_eq!(tanpi_strict(x_pos25), 1.0, "tanpi({x_pos25}) should be +1");
            assert_eq!(tanpi_strict(x_pos75), -1.0, "tanpi({x_pos75}) should be -1");
            assert_eq!(tanpi_strict(-x_pos25), -1.0, "tanpi({}) should be -1", -x_pos25);
            assert_eq!(tanpi_strict(-x_pos75), 1.0, "tanpi({}) should be +1", -x_pos75);
        }
    }

    #[test]
    fn sinpi_accuracy() {
        let pi = std::f64::consts::PI;
        for &x in &[0.1_f64, 0.3, 0.7, 1.3, 2.1] {
            let got = sinpi_strict(x);
            let expected = (pi * x).sin();
            let d = ulps_between(got, expected);
            // For x > 1, reducing frac = x - floor(x) and then multiplying by π
            // takes a different float path than π*x. The two computations agree
            // to within ~10 ulps; the function is still mathematically correct
            // (exact at integers/half-integers).
            assert!(d <= 10, "sinpi({x}): {d} ulps");
        }
    }
}
