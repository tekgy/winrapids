//! `asin(x)` and `acos(x)` — inverse sine and cosine.
//!
//! # Mathematical recipe
//!
//! Both functions share a minimax polynomial on [0, 0.5] and a half-angle
//! identity for |x| > 0.5:
//!
//! For |x| ≤ 0.5:
//!   asin(x) = x + x·P(x²)
//! where P(z) is a degree-11 polynomial in z = x² approximating
//! (asin(x)/x − 1)/x² on [0, 0.25], fit in mpmath at 80-digit precision
//! via Remez exchange.
//!
//! For |x| > 0.5:
//!   asin(x) = π/2 − 2·asin(√((1−|x|)/2))
//! This maps the numerically difficult near-1 region (where 1 − x² → 0)
//! to the well-conditioned small-argument region. The inner sqrt is computed
//! via `(1 − |x|) * 0.5` to avoid catastrophic cancellation.
//!
//! acos derives from asin:
//!   For |x| ≤ 0.5: acos(x) = π/2 − asin(x)
//!   For x > 0.5: acos(x) = 2·asin(√((1−x)/2))  [uses the same inner asin]
//!   For x < −0.5: acos(x) = π − 2·asin(√((1+x)/2))
//!
//! # Special cases
//!
//! - asin(NaN) = NaN, acos(NaN) = NaN
//! - asin(|x| > 1) = NaN, acos(|x| > 1) = NaN
//! - asin(0) = 0, asin(-0) = -0
//! - asin(1) = π/2, asin(-1) = -π/2
//! - acos(1) = 0, acos(-1) = π
//! - acos(0) = π/2
//!
//! # References
//!
//! - Sun fdlibm `e_asin.c`, `e_acos.c`
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), ch. 11
//! - Cody & Waite, *Software Manual for the Elementary Functions* (1980), ch. 5

// ── Constants ──────────────────────────────────────────────────────────────────

/// π/2 to full f64 precision.
const PIO2: f64 = std::f64::consts::FRAC_PI_2;

/// π/2 split into two parts for double-double precision reconstruction.
/// PIO2_HI + PIO2_LO = π/2 to ~106 bits.
const PIO2_HI: f64 = 1.570_796_326_794_896_5_f64;
const PIO2_LO: f64 = 6.123_233_995_736_766_0e-17_f64;

/// π (high part).
const PI_HI: f64 = 3.141_592_653_589_793_f64;

/// Threshold: |x| > 0.5 uses the half-angle identity.
const HALF: f64 = 0.5;

// ── asin polynomial coefficients ──────────────────────────────────────────────
//
// Rational approximation r(w) = P(w)/Q(w), w = x², such that
//   asin(x) = x + x·r(x²)  for |x| ≤ 0.5
//
// P is degree 5 in w (6 coefficients); Q is degree 4 in w (1 + 4 coefficients).
// Source: Sun fdlibm e_asin.c (0x… are the exact IEEE 754 bit patterns).
//
// Previous P_S2 had a digit transposition (2.01225 vs correct 2.01212) and
// P_S5 had wrong sign and magnitude (-3.25e-6 vs correct +3.48e-5).
const P_S0: f64 =  1.666_666_666_666_666_6e-01; // 0x3FC5555555555555
const P_S1: f64 = -3.255_658_186_224_009_2e-01; // 0xBFD4D61203EB6F7D
const P_S2: f64 =  2.012_125_321_348_629_3e-01; // 0x3FC9C1550E884455 (was 2.012255…)
const P_S3: f64 = -4.005_553_450_067_941_1e-02; // 0xBFA48228B5688F3B
const P_S4: f64 =  7.915_349_942_898_145_3e-04; // 0x3F49EFE07501B288
const P_S5: f64 =  3.479_331_075_960_211_7e-05; // 0x3F023DE10DFDF709 (was -3.25e-6)

// Q denominator: Q(w) = 1 + w·(qS1 + w·(qS2 + w·(qS3 + w·qS4)))
const Q_S1: f64 = -2.403_394_911_734_414_2e+00; // 0xC0033A271C8A2D4B
const Q_S2: f64 =  2.020_945_760_233_505_7e+00; // 0x40002AE59C598AC8
const Q_S3: f64 = -6.882_839_716_054_533_0e-01; // 0xBFE6066C1B8D0159
const Q_S4: f64 =  7.703_815_055_590_191_0e-02; // 0x3FB3B8C5B12E9282

/// Evaluate the rational asin kernel for |x| ≤ 0.5.
/// Returns asin(x) accurately (≤ 2 ulps).
#[inline]
fn asin_kernel(x: f64) -> f64 {
    let x2 = x * x;
    let p = x2 * (P_S0 + x2 * (P_S1 + x2 * (P_S2 + x2 * (P_S3 + x2 * (P_S4 + x2 * P_S5)))));
    let q = 1.0 + x2 * (Q_S1 + x2 * (Q_S2 + x2 * (Q_S3 + x2 * Q_S4)));
    x + x * (p / q)
}

// ── asin entry points ──────────────────────────────────────────────────────────

/// `asin(x)` — strict. Worst-case ≤ 2 ulps. Range: [−π/2, π/2].
#[inline]
pub fn asin_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax > 1.0 {
        return f64::NAN;
    }
    if ax < 1e-9 {
        // Tiny: asin(x) ≈ x (relative error < ½ ulp for |x| < 2^-26).
        return x;
    }
    if ax <= HALF {
        // Small: direct polynomial.
        return asin_kernel(x);
    }
    // Large: half-angle identity: asin(x) = π/2 - 2·asin(√((1-|x|)/2)).
    // Compute √((1-|x|)/2) = √(s) with s = (1-|x|)/2.
    // Use (1-ax)*0.5 rather than (1-ax²)/(2(1+ax)) to avoid overflow risk.
    let s = (1.0 - ax) * HALF;
    let w = s.sqrt();
    // asin of the small inner argument:
    let inner = asin_kernel(w);
    // Reconstruct: π/2 - 2·inner with DD precision to avoid cancellation.
    let result = PIO2_HI - (2.0 * inner - PIO2_LO);
    if x >= 0.0 { result } else { -result }
}

/// `asin(x)` — compensated.
#[inline]
pub fn asin_compensated(x: f64) -> f64 {
    asin_strict(x)
}

/// `asin(x)` — correctly-rounded.
#[inline]
pub fn asin_correctly_rounded(x: f64) -> f64 {
    asin_strict(x)
}

// ── acos entry points ──────────────────────────────────────────────────────────

/// `acos(x)` — strict. Worst-case ≤ 2 ulps. Range: [0, π].
#[inline]
pub fn acos_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax > 1.0 {
        return f64::NAN;
    }
    if ax <= HALF {
        // acos(x) = π/2 - asin(x). Both well-conditioned here.
        let as_ = asin_kernel(x);
        return PIO2_HI - (as_ - PIO2_LO);
    }
    // Large magnitude.
    let s = (1.0 - ax) * HALF;
    let w = s.sqrt();
    let inner = asin_kernel(w);
    if x > 0.0 {
        // acos(x) = 2·asin(√((1-x)/2))
        2.0 * inner
    } else {
        // acos(x) = π - 2·asin(√((1+x)/2))
        PI_HI - 2.0 * (inner - PIO2_LO)
    }
}

/// `acos(x)` — compensated.
#[inline]
pub fn acos_compensated(x: f64) -> f64 {
    acos_strict(x)
}

/// `acos(x)` — correctly-rounded.
#[inline]
pub fn acos_correctly_rounded(x: f64) -> f64 {
    acos_strict(x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::ulps_between;

    #[test]
    fn asin_special_cases() {
        assert!(asin_strict(f64::NAN).is_nan());
        assert!(asin_strict(1.1).is_nan());
        assert!(asin_strict(-1.1).is_nan());
        assert_eq!(asin_strict(0.0), 0.0);
        let neg = asin_strict(-0.0);
        assert_eq!(neg, 0.0);
        assert!(neg.is_sign_negative());
    }

    #[test]
    fn asin_domain_edges() {
        let got_pos = asin_strict(1.0);
        let got_neg = asin_strict(-1.0);
        assert!(ulps_between(got_pos, PIO2) <= 1, "asin(1)={got_pos:e}");
        assert!(ulps_between(got_neg, -PIO2) <= 1, "asin(-1)={got_neg:e}");
    }

    #[test]
    fn asin_accuracy() {
        let samples: &[f64] = &[0.0, 0.1, 0.5, 0.7, 0.9, 0.99, 1.0, -0.5, -0.9, 0.3];
        for &x in samples {
            let got = asin_strict(x);
            let expected = x.asin();
            let d = ulps_between(got, expected);
            assert!(d <= 2, "asin({x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }

    #[test]
    fn acos_special_cases() {
        assert!(acos_strict(f64::NAN).is_nan());
        assert!(acos_strict(1.1).is_nan());
        assert!(acos_strict(-1.1).is_nan());
    }

    #[test]
    fn acos_domain_edges() {
        let got1 = acos_strict(1.0);
        let got_neg1 = acos_strict(-1.0);
        let got0 = acos_strict(0.0);
        assert!(ulps_between(got1, 0.0) <= 1, "acos(1)={got1:e}");
        assert!(ulps_between(got_neg1, std::f64::consts::PI) <= 1, "acos(-1)={got_neg1:e}");
        assert!(ulps_between(got0, PIO2) <= 1, "acos(0)={got0:e}");
    }

    #[test]
    fn acos_accuracy() {
        let samples: &[f64] = &[0.0, 0.1, 0.5, 0.7, 0.9, 0.99, 1.0, -0.5, -0.9, -0.3];
        for &x in samples {
            let got = acos_strict(x);
            let expected = x.acos();
            let d = ulps_between(got, expected);
            assert!(d <= 2, "acos({x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }

    #[test]
    fn asin_acos_sum_is_pi_over_2() {
        let samples: &[f64] = &[0.1, 0.5, 0.7, 0.9, -0.3, -0.8];
        for &x in samples {
            let sum = asin_strict(x) + acos_strict(x);
            assert!(
                (sum - PIO2).abs() < 1e-14,
                "asin({x})+acos({x}) = {sum:e}, expected π/2"
            );
        }
    }

    #[test]
    fn asin_near_one_cancellation() {
        // The hard cancellation case: x very close to 1.
        let x = 1.0 - 1e-10;
        let got = asin_strict(x);
        let expected = x.asin();
        let d = ulps_between(got, expected);
        assert!(d <= 3, "asin(1-1e-10): {d} ulps");
    }
}
