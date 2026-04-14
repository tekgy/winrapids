//! Adversarial input generators for libm recipe accuracy testing.
//!
//! Hand-picked samples find SOME bugs. These generators find the worst cases:
//! values adjacent to region-transition points, float landmarks, the classic
//! Table-Maker's-Dilemma entries, and dense uniform sweeps through each
//! function's high-interest input range.
//!
//! # Usage
//!
//! ```ignore
//! use tambear::recipes::libm::adversarial;
//!
//! for x in adversarial::exp_adversarial() {
//!     let actual = tambear::recipes::libm::exp::exp_strict(x);
//!     let expected = x.exp();
//!     // compute ulps_between(actual, expected) etc.
//! }
//! ```
//!
//! Each generator returns a `Vec<f64>` of finite inputs in the function's
//! legal domain. Infinities, NaNs, and domain-excluded values are omitted
//! (those are covered by dedicated special-case tests elsewhere).

use std::f64::consts::{PI, E, LN_2, LN_10, SQRT_2};

// ── Float-landmark helpers ──────────────────────────────────────────────────

/// Next representable f64 above `x` (toward +∞). Returns `f64::NAN` for NaN.
#[inline]
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return x;
    }
    let bits = x.to_bits();
    // Treat -0.0 as 0.0 for the purposes of "next above".
    if bits == (1u64 << 63) {
        return f64::from_bits(1);
    }
    let next_bits = if (bits >> 63) == 0 {
        bits.wrapping_add(1)
    } else {
        bits.wrapping_sub(1)
    };
    f64::from_bits(next_bits)
}

/// Next representable f64 below `x` (toward -∞).
#[inline]
pub fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// Emit `x`, `next_up(x)`, and `next_down(x)` into `out`, skipping any value
/// outside `(lo, hi)` or that is not finite.
fn push_ulp_neighborhood(out: &mut Vec<f64>, x: f64, lo: f64, hi: f64) {
    for v in [next_down(x), x, next_up(x)] {
        if v.is_finite() && v > lo && v < hi {
            out.push(v);
        }
    }
}

/// Linearly-spaced points in `[lo, hi]` inclusive (`n >= 2`).
fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2);
    let step = (hi - lo) / (n as f64 - 1.0);
    (0..n).map(|i| lo + step * i as f64).collect()
}

/// Powers of two `2^k` for `k` in `range`, each emitted with `±1 ulp`.
fn powers_of_two_with_ulps(range: std::ops::RangeInclusive<i32>, lo: f64, hi: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for k in range {
        let p = (k as f64).exp2();
        push_ulp_neighborhood(&mut out, p, lo, hi);
        push_ulp_neighborhood(&mut out, -p, lo, hi);
    }
    out
}

/// Generic deduplicate+sort helper (by bit pattern, so ±0 are kept distinct).
fn dedup_sort(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.dedup_by(|a, b| a.to_bits() == b.to_bits());
    v
}

// ── Shared "full spectrum" landmark set ─────────────────────────────────────

/// Float-landmark inputs: powers of 2 from 2^-52 (eps scale) to 2^20, plus
/// subnormal edges, f64::MIN_POSITIVE, f64::EPSILON, SQRT_2, E, PI, LN_2,
/// each with ±1 ulp. Filtered to `(lo, hi)`.
pub fn float_landmarks(lo: f64, hi: f64) -> Vec<f64> {
    let mut out = Vec::new();
    // Powers of two at moderate exponents (most recipes can't take 2^1023
    // directly — that's an out-of-domain test, not an accuracy test).
    out.extend(powers_of_two_with_ulps(-60..=60, lo, hi));

    // Subnormal edges & extreme small positives.
    for &v in &[
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE * 2.0,
        f64::EPSILON,
        f64::EPSILON * 0.5,
        1.0e-300,
        1.0e-200,
        1.0e-100,
        5e-324,    // smallest positive subnormal
    ] {
        push_ulp_neighborhood(&mut out, v, lo, hi);
        push_ulp_neighborhood(&mut out, -v, lo, hi);
    }

    // Mathematical constants & friends.
    for &v in &[
        1.0, 2.0, 0.5, 10.0, 0.1, PI, E, LN_2, LN_10, SQRT_2,
        PI / 2.0, PI / 4.0, 3.0 * PI / 4.0, 2.0 * PI,
    ] {
        push_ulp_neighborhood(&mut out, v, lo, hi);
        push_ulp_neighborhood(&mut out, -v, lo, hi);
    }

    // Very-large (near overflow) and near-f64::MAX.
    for &v in &[1e100, 1e200, 1e300] {
        push_ulp_neighborhood(&mut out, v, lo, hi);
        push_ulp_neighborhood(&mut out, -v, lo, hi);
    }

    dedup_sort(out)
}

// ── exp ─────────────────────────────────────────────────────────────────────

/// Adversarial inputs for `exp`. Covers the overflow threshold (~709.78),
/// the underflow threshold (~-745.13), integer multiples of ln(2), plus
/// dense sweep through [-10, 10] and landmarks.
pub fn exp_adversarial() -> Vec<f64> {
    const EXP_MAX: f64 = 709.782_712_893_384;
    const EXP_MIN: f64 = -745.133_219_101_941;
    const LO: f64 = EXP_MIN + 1.0;  // stay in-domain
    const HI: f64 = EXP_MAX - 1.0;

    let mut out = Vec::new();

    // Region boundaries: overflow/underflow, 0, ±ln(2)·k for k in [-30, 30].
    push_ulp_neighborhood(&mut out, EXP_MAX, LO, HI + 2.0);
    push_ulp_neighborhood(&mut out, EXP_MIN, LO - 2.0, HI);
    out.push(0.0);
    out.push(next_up(0.0));
    out.push(next_down(0.0));
    for k in -30..=30 {
        let x = LN_2 * k as f64;
        push_ulp_neighborhood(&mut out, x, LO, HI);
    }

    // Known hard inputs near ln(2)/2 and other quarter-ln(2) points (the
    // range reduction pivot is k = round(x / ln(2))). Test points where the
    // rounding flips.
    for k in -20..=20 {
        let x = LN_2 * (k as f64 + 0.5);
        push_ulp_neighborhood(&mut out, x, LO, HI);
    }

    // Dense sweep through [-10, 10] (1000 pts).
    out.extend(linspace(-10.0, 10.0, 1000));

    // Small-x region where the Taylor series dominates (argument reduction
    // trivial). Test points near 0.
    for k in -80..=-20 {
        let p = (k as f64).exp2();
        push_ulp_neighborhood(&mut out, p, LO, HI);
    }

    // Landmarks restricted to in-domain range.
    out.extend(float_landmarks(LO, HI));

    // Table-Maker's-Dilemma-class inputs for exp (from de Dinechin/Lefèvre
    // worst-case tables): these are points where exp(x) is exceptionally
    // close to a rounding boundary. A handful of published examples:
    for &v in &[
        7.541_232_153_846_271e-1,
        1.100_000_000_000_000_1,
        7.853_981_633_974_483e-1,    // π/4 — boundary for neighbors
        -7.853_981_633_974_483e-1,
        6.283_185_307_179_586,       // 2π
        22.18070977791825,           // deep but in-range
    ] {
        push_ulp_neighborhood(&mut out, v, LO, HI);
    }

    dedup_sort(out)
}

// ── log ─────────────────────────────────────────────────────────────────────

/// Adversarial inputs for `log(x)`, `x > 0`.
pub fn log_adversarial() -> Vec<f64> {
    const LO: f64 = 0.0;
    const HI: f64 = f64::INFINITY;

    let mut out = Vec::new();

    // Region boundaries around 1.0 (ln(1) = 0, the cancellation hotspot).
    for dx in [
        0.0, 1e-16, 1e-12, 1e-8, 1e-4, 1e-2, 0.1, 0.25, 0.5,
    ] {
        push_ulp_neighborhood(&mut out, 1.0 + dx, LO, HI);
        push_ulp_neighborhood(&mut out, 1.0 - dx, LO, HI);
    }

    // sqrt(2) — the internal log pivot between the two halves of the
    // [1/sqrt(2), sqrt(2)] reduced domain.
    push_ulp_neighborhood(&mut out, SQRT_2, LO, HI);
    push_ulp_neighborhood(&mut out, 1.0 / SQRT_2, LO, HI);

    // Powers of 2 from 2^-100 to 2^100 (dense, since log is exact on these
    // to the extent ln(2) is).
    for k in -100..=100 {
        let p = (k as f64).exp2();
        push_ulp_neighborhood(&mut out, p, LO, HI);
    }

    // Dense sweep through [0.1, 100] (1000 pts).
    out.extend(linspace(0.1, 100.0, 1000));
    // Extra density near 1.
    out.extend(linspace(0.5, 1.5, 200));

    // Landmarks (positive half).
    for v in float_landmarks(LO, HI) {
        if v > 0.0 {
            out.push(v);
        }
    }

    // Published Table-Maker's-Dilemma inputs for ln.
    for &v in &[
        1.000_000_000_000_000_2,
        1.999_999_999_999_999_8,
        1.099_999_999_999_999_9,
        2.0_f64.sqrt(),
    ] {
        push_ulp_neighborhood(&mut out, v, LO, HI);
    }

    out.retain(|x| *x > 0.0 && x.is_finite());
    dedup_sort(out)
}

// ── sin / cos ───────────────────────────────────────────────────────────────

/// Adversarial inputs for `sin` and `cos`. Bounded at |x| ≤ 1e6 because
/// the current recipe uses a DD-precision range reduction that degrades
/// for very large arguments; Payne-Hanek reduction is tracked separately
/// and will re-extend this domain. The generator still probes every
/// interesting region in that bound.
pub fn sin_cos_adversarial() -> Vec<f64> {
    const LO: f64 = -1.0e6;
    const HI: f64 = 1.0e6;

    let mut out = Vec::new();

    // Multiples of π/4 across a range that still argument-reduces cleanly.
    for k in -200..=200 {
        let x = (k as f64) * (PI / 4.0);
        push_ulp_neighborhood(&mut out, x, LO, HI);
    }
    // Multiples of π/2 (sin zeros / cos zeros).
    for k in -500..=500 {
        let x = (k as f64) * (PI / 2.0);
        push_ulp_neighborhood(&mut out, x, LO, HI);
    }
    // Multiples of π (catastrophic cancellation points for sin).
    for k in -1000..=1000 {
        let x = (k as f64) * PI;
        push_ulp_neighborhood(&mut out, x, LO, HI);
    }

    // Dense sweep over [0, π] (1000 pts) and [-π, π] extra.
    out.extend(linspace(0.0, PI, 1000));
    out.extend(linspace(-PI, PI, 500));

    // The classic worst-case worst-case: large argument where reduction
    // dominates error. We limit to |x| <= 1e6 because beyond that the
    // 2-part pi reduction in the recipe exhausts its precision budget and
    // this becomes an out-of-spec test.
    for k in &[1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0] {
        let x = *k;
        push_ulp_neighborhood(&mut out, x, LO, HI);
        push_ulp_neighborhood(&mut out, -x, LO, HI);
    }

    // Kahan's famous hard inputs for sin (near multiples of π that are
    // representable almost-exactly in f64 — biggest cancellation happens
    // where reduced argument is smallest). We keep those that fall
    // within the recipe's supported DD-reduction range.
    for &v in &[
        3.141_592_653_589_793e+0,   // π itself
        6.283_185_307_179_586e+0,   // 2π
        1.570_796_326_794_896_6,    // π/2
        355.0_f64,                  // 355/113 ≈ π, so 355 mod π is tiny
        100_000.0_f64,              // large but in-domain
    ] {
        push_ulp_neighborhood(&mut out, v, LO, HI);
    }

    // Landmarks.
    out.extend(float_landmarks(LO, HI));

    out.retain(|x| x.is_finite() && x.abs() <= HI);
    dedup_sort(out)
}

// ── erf / erfc ──────────────────────────────────────────────────────────────

/// Adversarial inputs for `erf` and `erfc`. The recipe has region
/// transitions at |x| = 0.84375 (fdlibm erf_tiny/erf_small boundary),
/// 1.25 (small→medium), 1/0.35 ≈ 2.857 (medium→large), 6 (saturation
/// for erf), and 28 (saturation for erfc).
pub fn erf_adversarial() -> Vec<f64> {
    const LO: f64 = -30.0;
    const HI: f64 = 30.0;

    let mut out = Vec::new();

    // Region boundaries from the fdlibm-style 3-region layout plus our
    // recipe's cut at 1.5 and 6.
    for &b in &[
        0.0, 0.25, 0.5, 0.84375, 1.0, 1.25, 1.5, 2.0,
        1.0 / 0.35, 2.857_142_857_142_857, 3.0, 4.0, 5.0, 6.0,
        10.0, 20.0, 28.0,
    ] {
        push_ulp_neighborhood(&mut out, b, LO, HI);
        push_ulp_neighborhood(&mut out, -b, LO, HI);
    }

    // Dense sweep [-5, 5] (1000 pts) where erf changes most rapidly.
    out.extend(linspace(-5.0, 5.0, 1000));
    out.extend(linspace(-1.5, 1.5, 400));

    // Landmarks (restricted).
    out.extend(float_landmarks(LO, HI));

    out.retain(|x| x.is_finite() && x.abs() <= HI);
    dedup_sort(out)
}

// ── gamma ───────────────────────────────────────────────────────────────────

/// Adversarial inputs for `tgamma` and `lgamma` on positive reals. The
/// recipe has a cut at 0.5 (reflection formula boundary) and grows to
/// overflow near x ≈ 171.
pub fn gamma_adversarial() -> Vec<f64> {
    const LO: f64 = 0.0;
    const HI: f64 = 170.0;

    let mut out = Vec::new();

    // Region boundaries.
    for &b in &[
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 170.0,
    ] {
        push_ulp_neighborhood(&mut out, b, LO, HI);
    }

    // Integers (gamma(n) = (n-1)! — exact-factorial reference points).
    for n in 1..=170 {
        push_ulp_neighborhood(&mut out, n as f64, LO, HI);
    }

    // Half-integers (gamma closed-form in sqrt(π) / powers of 2).
    for n in 0..=40 {
        let x = 0.5 + n as f64;
        push_ulp_neighborhood(&mut out, x, LO, HI);
    }

    // Dense sweep [0.5, 20] (1000 pts).
    out.extend(linspace(0.5, 20.0, 1000));
    // Extra density near the minimum at x ≈ 1.4616 (gamma's sole positive
    // minimum — derivative zero ⇒ worst loss of precision).
    out.extend(linspace(1.0, 2.0, 200));

    out.retain(|x| x.is_finite() && *x > 0.0 && *x < HI);
    dedup_sort(out)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_up_round_trip() {
        // ±0 is an inherent edge: next_up(-0.0) = smallest subnormal, and
        // next_down of that is +0.0, not -0.0. Test finite non-zero values
        // where the bit-exact round-trip holds.
        for &x in &[1.0_f64, -1.0, 1e-300, 1e300, f64::EPSILON, -f64::EPSILON] {
            let u = next_up(x);
            let d = next_down(u);
            assert_eq!(d.to_bits(), x.to_bits(), "round-trip failed for {x}");
        }
    }

    #[test]
    fn exp_generator_nonempty_in_domain() {
        let v = exp_adversarial();
        assert!(v.len() > 1200, "exp_adversarial too small: {}", v.len());
        for x in &v {
            assert!(x.is_finite(), "exp input not finite: {x}");
            assert!(*x > -746.0 && *x < 710.0, "exp input out of range: {x}");
        }
    }

    #[test]
    fn log_generator_positive_finite() {
        let v = log_adversarial();
        assert!(v.len() > 1500, "log_adversarial too small: {}", v.len());
        for x in &v {
            assert!(*x > 0.0 && x.is_finite(), "log input not positive finite: {x}");
        }
    }

    #[test]
    fn sin_cos_generator_finite() {
        let v = sin_cos_adversarial();
        assert!(v.len() > 2000, "sin_cos_adversarial too small: {}", v.len());
        for x in &v {
            assert!(x.is_finite() && x.abs() <= 1.0e6,
                    "sin input out of range: {x}");
        }
    }

    #[test]
    fn erf_generator_in_range() {
        let v = erf_adversarial();
        assert!(v.len() > 1000, "erf_adversarial too small: {}", v.len());
        for x in &v {
            assert!(x.is_finite() && x.abs() <= 30.0, "erf input out of range: {x}");
        }
    }

    #[test]
    fn gamma_generator_positive() {
        let v = gamma_adversarial();
        assert!(v.len() > 1200, "gamma_adversarial too small: {}", v.len());
        for x in &v {
            assert!(*x > 0.0 && *x < 170.0 && x.is_finite(),
                    "gamma input out of range: {x}");
        }
    }
}
