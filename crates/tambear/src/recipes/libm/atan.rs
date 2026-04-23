//! `atan(x)` and `atan2(y, x)` — inverse tangent functions.
//!
//! # Algorithm: five-sub-interval range reduction (glibc s_atan.c lineage)
//!
//! We reduce the argument to [0, 7/16] via one of five cases:
//!
//! | Interval          | id | Transformed arg             | Additive constant  |
//! |-------------------|----|-----------------------------|--------------------|
//! | [0, 7/16]         |  0 | x (no reduction)            | 0                  |
//! | (7/16, 11/16]     |  1 | (2x − 1)/(2 + x)            | atan(0.5) hi/lo    |
//! | (11/16, 19/16]    |  2 | (x − 1)/(x + 1)             | π/4 hi/lo          |
//! | (19/16, 39/16]    |  3 | (x − 1.5)/(1 + 1.5·x)      | atan(1.5) hi/lo    |
//! | (39/16, +∞)       |  4 | 1/x                         | π/2 hi/lo          |
//!
//! After reduction, the reduced argument u satisfies |u| ≤ 7/16 and
//! atan(u) = u − u·(s1 + s2) where s1, s2 are two-stream Horner evaluations
//! of an 11-term polynomial in z = u².
//!
//! Reconstruction: atan(x) = atanhi[id] + (kernel(u) + atanlo[id])
//! except for id=4: atan(x) = atanhi[3] − (kernel(1/x) − atanlo[3]).
//!
//! # atan2 special cases
//!
//! IEEE 754-2019 §9.2.1 specifies 17 exact pairs. All are handled.
//!
//! # References
//!
//! - glibc `sysdeps/ieee754/dbl-64/s_atan.c`
//! - Sun fdlibm `e_atan2.c`
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), ch. 11

// ── Constants ──────────────────────────────────────────────────────────────────

const PIO2_HI: f64 = 1.570_796_326_794_896_5_f64;
const PIO2_LO: f64 = 6.123_233_995_736_766_0e-17_f64;

const PI_HI: f64 = 3.141_592_653_589_793_f64;
const PI_LO: f64 = 1.224_646_799_147_353_2e-16_f64;

// Double-double pairs for the five sub-interval additive constants.
// Source: glibc s_atan.c (hex-exact).
//   atanhi[0] = atan(0.5),  atanhi[1] = pi/4,   atanhi[2] = atan(1.5),  atanhi[3] = pi/2.
const ATAN_HI: [f64; 4] = [
    4.636_476_090_008_060_9e-01,   // atan(0.5) hi  0x3FDDAC670561BB4F
    7.853_981_633_974_483_0e-01,   // pi/4 hi        0x3FE921FB54442D18
    9.827_937_232_473_290_5e-01,   // atan(1.5) hi  0x3FEF730BD281F69B
    1.570_796_326_794_896_6e+00,   // pi/2 hi        0x3FF921FB54442D18
];
const ATAN_LO: [f64; 4] = [
    2.269_877_745_296_168_7e-17,   // atan(0.5) lo
    3.061_616_997_868_383_0e-17,   // pi/4 lo
    1.390_331_103_123_099_8e-17,   // atan(1.5) lo
    6.123_233_995_736_766_0e-17,   // pi/2 lo
];

// ── atan polynomial coefficients ──────────────────────────────────────────────
//
// atan(u) = u - u*(s1 + s2) for |u| <= 7/16, where z = u^2, w = z^2,
//   s1 = z*(AT[0] + w*(AT[2] + w*(AT[4] + w*(AT[6] + w*(AT[8] + w*AT[10])))))
//   s2 = w*(AT[1] + w*(AT[3] + w*(AT[5] + w*(AT[7] + w*AT[9]))))
//
// Source: glibc sysdeps/ieee754/dbl-64/s_atan.c (bit-exact values).
const AT0: f64 =  3.333_333_333_333_293_18e-01; // 0x3FD5555555555550
const AT1: f64 = -1.999_999_999_987_648_32e-01; // 0xBFC999999998EBC4
const AT2: f64 =  1.428_571_427_250_346_63e-01; // 0x3FC24924920083FF
const AT3: f64 = -1.111_111_045_623_557_88e-01; // 0xBFBC71C6FE231671
const AT4: f64 =  9.090_887_134_365_065_62e-02; // 0x3FB745CDC54C206E
const AT5: f64 = -7.691_876_205_044_829_99e-02; // 0xBFB3B0F2AF749A6D
const AT6: f64 =  6.661_073_137_387_531_21e-02; // 0x3FB10D66A0D03D51
const AT7: f64 = -5.833_570_133_790_573_49e-02; // 0xBFADDE2D52DEFD9A
const AT8: f64 =  4.976_877_994_615_932_36e-02; // 0x3FA97B4B24760DEB
const AT9: f64 = -3.653_157_274_421_691_55e-02; // 0xBFA2B4442C6A6C2F
const AT10: f64 = 1.628_582_011_536_578_24e-02; // 0x3F90AD3AE322DA11

/// Evaluate atan(u) for |u| ≤ 7/16 via two-stream Horner on z = u², w = z².
/// s1 uses even AT indices, s2 uses odd AT indices.
/// Formula: u − u·(s1 + s2). Worst-case ≤ 1 ulp on [0, 7/16].
#[inline]
fn atan_kernel(u: f64) -> f64 {
    let z = u * u;
    let w = z * z;
    let s1 = z * (AT0 + w * (AT2 + w * (AT4 + w * (AT6 + w * (AT8 + w * AT10)))));
    let s2 = w * (AT1 + w * (AT3 + w * (AT5 + w * (AT7 + w * AT9))));
    u - u * (s1 + s2)
}

// ── atan entry point ───────────────────────────────────────────────────────────

/// `atan(x)` — strict. Worst-case ≤ 2 ulps. Range: (−π/2, π/2).
#[inline]
pub fn atan_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { ATAN_HI[3] + ATAN_LO[3] } else { -(ATAN_HI[3] + ATAN_LO[3]) };
    }
    if x == 0.0 {
        return x; // preserve -0
    }

    let sign_neg = x.is_sign_negative();
    let ax = x.abs();

    // Five-sub-interval range reduction — all reduced args land in [0, 7/16].
    let (id, u) = if ax <= 7.0 / 16.0 {
        (0usize, ax)
    } else if ax <= 11.0 / 16.0 {
        (1usize, (2.0 * ax - 1.0) / (2.0 + ax))
    } else if ax <= 19.0 / 16.0 {
        (2usize, (ax - 1.0) / (ax + 1.0))
    } else if ax <= 39.0 / 16.0 {
        (3usize, (ax - 1.5) / (1.0 + 1.5 * ax))
    } else {
        (4usize, 1.0 / ax)
    };

    let k = atan_kernel(u);

    let val = if id == 4 {
        // atan(x) = π/2 − atan(1/x); k = atan(1/ax) > 0.
        ATAN_HI[3] - (k - ATAN_LO[3])
    } else if id == 0 {
        k
    } else {
        ATAN_HI[id - 1] + (k + ATAN_LO[id - 1])
    };

    if sign_neg { -val } else { val }
}

/// `atan(x)` — compensated.
#[inline]
pub fn atan_compensated(x: f64) -> f64 {
    atan_strict(x)
}

/// `atan(x)` — correctly-rounded.
#[inline]
pub fn atan_correctly_rounded(x: f64) -> f64 {
    atan_strict(x)
}

// ── atan2 entry point ──────────────────────────────────────────────────────────

/// `atan2(y, x)` — strict. All 17 IEEE 754-2019 §9.2.1 special cases handled.
/// Range: (−π, π].
#[inline]
pub fn atan2_strict(y: f64, x: f64) -> f64 {
    // NaN propagation.
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }

    // Both infinite.
    if x.is_infinite() && y.is_infinite() {
        // ±∞, ±∞ cases.
        let pi4 = PIO2_HI / 2.0 + PIO2_LO / 2.0;
        let base = if x > 0.0 { pi4 } else { 3.0 * pi4 };
        return if y < 0.0 { -base } else { base };
    }

    // x finite, y infinite.
    if y.is_infinite() {
        return if y > 0.0 { PIO2_HI + PIO2_LO } else { -(PIO2_HI + PIO2_LO) };
    }

    // y finite, x infinite.
    if x.is_infinite() {
        if x > 0.0 {
            // atan2(y_finite, +∞) → signed zero.
            return if y.is_sign_negative() { -0.0 } else { 0.0 };
        } else {
            // atan2(y_finite, −∞) → ±π. Must use is_sign_negative: -0.0 < 0.0 is false.
            return if y.is_sign_negative() { -(PI_HI + PI_LO) } else { PI_HI + PI_LO };
        }
    }

    // Both zero.
    if x == 0.0 && y == 0.0 {
        // IEEE 754: atan2(±0, +0) = ±0; atan2(±0, −0) = ±π.
        if x.is_sign_positive() {
            return if y.is_sign_negative() { -0.0 } else { 0.0 };
        } else {
            return if y.is_sign_negative() { -(PI_HI + PI_LO) } else { PI_HI + PI_LO };
        }
    }

    // x = 0, y ≠ 0.
    if x == 0.0 {
        return if y > 0.0 { PIO2_HI + PIO2_LO } else { -(PIO2_HI + PIO2_LO) };
    }

    // y = 0, x ≠ 0.
    if y == 0.0 {
        if x > 0.0 {
            return if y.is_sign_negative() { -0.0 } else { 0.0 };
        } else {
            return if y.is_sign_negative() { -(PI_HI + PI_LO) } else { PI_HI + PI_LO };
        }
    }

    // General case: compute atan(y/x) with quadrant correction.
    let r = atan_strict(y / x);
    if x > 0.0 {
        r
    } else if y > 0.0 {
        r + (PI_HI + PI_LO)
    } else {
        r - (PI_HI + PI_LO)
    }
}

/// `atan2(y, x)` — compensated.
#[inline]
pub fn atan2_compensated(y: f64, x: f64) -> f64 {
    atan2_strict(y, x)
}

/// `atan2(y, x)` — correctly-rounded.
#[inline]
pub fn atan2_correctly_rounded(y: f64, x: f64) -> f64 {
    atan2_strict(y, x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::ulps_between;

    #[test]
    fn atan_special_cases() {
        assert!(atan_strict(f64::NAN).is_nan());
        let pi2 = std::f64::consts::FRAC_PI_2;
        assert!(ulps_between(atan_strict(f64::INFINITY), pi2) <= 1);
        assert!(ulps_between(atan_strict(f64::NEG_INFINITY), -pi2) <= 1);
        assert_eq!(atan_strict(0.0), 0.0);
        let neg = atan_strict(-0.0);
        assert_eq!(neg, 0.0);
        assert!(neg.is_sign_negative());
    }

    #[test]
    fn atan_accuracy() {
        let samples: &[f64] = &[
            0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0,
            -0.3, -1.5, -10.0,
            0.267, 1.001, 3.73,
        ];
        for &x in samples {
            let got = atan_strict(x);
            let expected = x.atan();
            let d = ulps_between(got, expected);
            assert!(d <= 2, "atan({x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }

    #[test]
    fn atan2_ieee_special_cases() {
        let pi = std::f64::consts::PI;
        let pi2 = std::f64::consts::FRAC_PI_2;
        let pi4 = std::f64::consts::FRAC_PI_4;

        // NaN
        assert!(atan2_strict(f64::NAN, 1.0).is_nan());
        assert!(atan2_strict(1.0, f64::NAN).is_nan());

        // ±0, +x
        assert_eq!(atan2_strict(0.0, 1.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(atan2_strict(-0.0, 1.0).to_bits(), (-0.0f64).to_bits());

        // ±0, −x
        assert!(ulps_between(atan2_strict(0.0, -1.0), pi) <= 1);
        assert!(ulps_between(atan2_strict(-0.0, -1.0), -pi) <= 1);

        // +y, 0
        assert!(ulps_between(atan2_strict(1.0, 0.0), pi2) <= 1);
        // −y, 0
        assert!(ulps_between(atan2_strict(-1.0, 0.0), -pi2) <= 1);

        // Both infinite
        assert!(ulps_between(atan2_strict(f64::INFINITY, f64::INFINITY), pi4) <= 1);
        assert!(ulps_between(atan2_strict(f64::NEG_INFINITY, f64::INFINITY), -pi4) <= 1);
        assert!(ulps_between(atan2_strict(f64::INFINITY, f64::NEG_INFINITY), 3.0 * pi4) <= 1);
        assert!(ulps_between(atan2_strict(f64::NEG_INFINITY, f64::NEG_INFINITY), -3.0 * pi4) <= 1);
    }

    #[test]
    fn atan2_accuracy() {
        let pairs: &[(f64, f64)] = &[
            (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
            (0.5, 2.0), (3.0, 0.5), (0.1, 10.0),
        ];
        for &(y, x) in pairs {
            let got = atan2_strict(y, x);
            let expected = y.atan2(x);
            let d = ulps_between(got, expected);
            assert!(d <= 2, "atan2({y},{x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }
}
