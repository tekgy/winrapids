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

// ── tan / cot / sec / csc ───────────────────────────────────────────────────

/// Adversarial inputs for `tan`, `cot`, `sec`, `csc`.
///
/// The primary challenge is the poles:
/// - `tan` and `sec` have poles at x = π/2 + kπ.
/// - `cot` and `csc` have poles at x = kπ.
///
/// Near-pole inputs are the critical accuracy region: a 1-ulp error in the
/// argument translates to a potentially enormous error in the output if the
/// denominator (cos or sin) is near zero. We include both near-pole inputs
/// AND the pole approach sequence to verify the recipe doesn't silently
/// return plausible-looking wrong values.
///
/// Inputs that land exactly on a pole representable in f64 are excluded
/// (those are covered by special-case tests in trig_adversarial.rs).
pub fn tan_adversarial() -> Vec<f64> {
    const LO: f64 = -1.0e6;
    const HI: f64 = 1.0e6;

    let mut out = Vec::new();

    // ── Near-tan/sec poles at π/2 + kπ ──────────────────────────────────
    // The pole itself is not representable exactly in f64, so the ±1/±2/±3
    // ulp neighborhood of float(π/2 + kπ) exercises near-pole amplification.
    let pio2 = PI / 2.0;
    for k in -200..=200 {
        let pole = pio2 + (k as f64) * PI;
        // Emit the float(pole) itself and its ±1/2/3 ulp neighbors.
        // These are the "should be very large" inputs, not "should be ≈ 1" inputs.
        for bits_off in [-3i64, -2, -1, 0, 1, 2, 3] {
            let v = f64::from_bits(pole.to_bits().wrapping_add_signed(bits_off));
            if v.is_finite() && v.abs() <= HI {
                out.push(v);
            }
        }
    }

    // ── Near-cot/csc poles at kπ ─────────────────────────────────────────
    // Exclude 0 (special-cased separately); include ±kπ neighbors.
    for k in -200..=200i32 {
        if k == 0 {
            continue;
        }
        let pole = (k as f64) * PI;
        for bits_off in [-3i64, -2, -1, 0, 1, 2, 3] {
            let v = f64::from_bits(pole.to_bits().wrapping_add_signed(bits_off));
            if v.is_finite() && v.abs() <= HI {
                out.push(v);
            }
        }
    }

    // ── Multiples of π/4 (tan = ±1) ─────────────────────────────────────
    for k in -400_i32..=400 {
        // Skip k ≡ 1 mod 4 and k ≡ 3 mod 4 — those are the poles.
        if (k.rem_euclid(4) == 1) || (k.rem_euclid(4) == 3) {
            continue;
        }
        let x = (k as f64) * (PI / 4.0);
        push_ulp_neighborhood(&mut out, x, LO, HI);
    }

    // ── Dense interior sweeps ────────────────────────────────────────────
    // Avoid the pole neighborhoods by sweeping (kπ + ε, (k+1)π - ε).
    for k in -5..=5 {
        let lo = (k as f64) * PI + 0.1;
        let hi = lo + PI - 0.2;
        out.extend(linspace(lo.max(LO), hi.min(HI), 200));
    }

    // ── Large arguments (Payne-Hanek regime) ────────────────────────────
    // Same as sin_cos_adversarial but for tan.
    for &mag in &[1.0e4, 1.0e5, 1.0e6] {
        push_ulp_neighborhood(&mut out, mag, LO, HI);
        push_ulp_neighborhood(&mut out, -mag, LO, HI);
    }

    // ── Landmarks ────────────────────────────────────────────────────────
    // Filter: keep only values where |cos(x)| > 2^-10 and |sin(x)| > 2^-10
    // (so neither tan nor cot has an amplified denominator in the landmark set).
    let landmarks = float_landmarks(LO, HI);
    for x in landmarks {
        if x.cos().abs() > 1.0 / 1024.0 && x.sin().abs() > 1.0 / 1024.0 {
            out.push(x);
        }
    }

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

// ── asin / acos ─────────────────────────────────────────────────────────────

/// Adversarial inputs for `asin(x)` and `acos(x)`.
///
/// The critical precision challenge is the **cancellation zone near ±1**:
/// - Naive implementations use `sqrt(1 - x²)`, which loses all bits when x ≈ 1
///   because `1 - x²` underflows to zero before `x²` loses its low bits.
/// - Correct implementations use the half-angle reduction:
///   `asin(x) = π/2 − 2·asin(√((1−|x|)/2))`.
///   This maps the hard region near 1 to the well-conditioned region near 0.
///
/// This generator probes every region boundary for both approaches, plus the
/// zero crossings and saturation points at ±1.
pub fn asin_adversarial() -> Vec<f64> {
    const LO: f64 = -1.0;
    const HI: f64 = 1.0;

    let mut out = Vec::new();

    // ── Domain edges (the correctness floor) ────────────────────────────
    // asin(±1) = ±π/2 exactly to 1 ULP.
    push_ulp_neighborhood(&mut out, 1.0, -1.1, 1.1);
    push_ulp_neighborhood(&mut out, -1.0, -1.1, 1.1);

    // ── The half-angle transition boundary at |x| = 0.5 ─────────────────
    // Implementations using the half-angle identity for |x| > 0.5 switch
    // algorithm at this boundary. The 3-ULP neighborhood probes the switch.
    push_ulp_neighborhood(&mut out, 0.5, LO, HI);
    push_ulp_neighborhood(&mut out, -0.5, LO, HI);

    // ── Cancellation zone: x approaching ±1 from below ──────────────────
    // These are the inputs where naive implementations produce catastrophically
    // wrong answers. A 1-ULP error in the output here is a silent failure.
    for dx_exp in [1, 2, 4, 6, 8, 10, 12, 14] {
        let dx = (10.0_f64).powi(-dx_exp);
        push_ulp_neighborhood(&mut out, 1.0 - dx, LO, HI);
        push_ulp_neighborhood(&mut out, -(1.0 - dx), LO, HI);
    }
    // Also: 1-ε where ε is machine epsilon and small multiples.
    for k in [1, 2, 4, 8, 16, 32, 64] {
        let dx = (k as f64) * f64::EPSILON;
        push_ulp_neighborhood(&mut out, 1.0 - dx, LO, HI);
        push_ulp_neighborhood(&mut out, -(1.0 - dx), LO, HI);
    }

    // ── Standard angle checkpoints ───────────────────────────────────────
    // asin(0) = 0, asin(√2/2) = π/4, asin(√3/2) = π/3, etc.
    use std::f64::consts::FRAC_1_SQRT_2;
    let sqrt3_over_2 = 3.0_f64.sqrt() / 2.0;
    for &v in &[
        0.0_f64,
        FRAC_1_SQRT_2,  // asin = π/4
        sqrt3_over_2,   // asin = π/3
        0.25, 0.75,     // common checkpoints
    ] {
        push_ulp_neighborhood(&mut out, v, LO, HI);
        push_ulp_neighborhood(&mut out, -v, LO, HI);
    }

    // ── Dense interior sweep: where asin changes most rapidly ─────────────
    // The derivative 1/√(1-x²) grows fastest near ±1. Dense sweep near 0
    // (polynomial regime) and around 0.5 (algorithm switch).
    out.extend(linspace(-1.0 + 1e-10, 1.0 - 1e-10, 1000));
    out.extend(linspace(-0.6, 0.6, 300));

    // ── Subnormal-scale inputs (tiny x: asin(x) ≈ x) ─────────────────────
    for &v in &[f64::MIN_POSITIVE, f64::EPSILON, 1e-100, 1e-15, 1e-8, 1e-4] {
        push_ulp_neighborhood(&mut out, v, LO, HI);
        push_ulp_neighborhood(&mut out, -v, LO, HI);
    }

    out.retain(|x| x.is_finite() && x.abs() <= 1.0);
    dedup_sort(out)
}

// ── atan / atan2 ─────────────────────────────────────────────────────────────

/// Adversarial inputs for `atan(x)` and `atan2(y, x)`.
///
/// The five-subinterval reduction (glibc lineage) has boundaries at
/// 7/16, 11/16, 19/16, 39/16. Each boundary is a potential precision trap:
/// an implementation that uses slightly different constants on either side
/// of a boundary will show a discontinuity at the boundary itself.
///
/// The poles of the related functions (atan2 with x=0, cot at x=0) are
/// not in atan's domain but atan's range approaches ±π/2 asymptotically —
/// the large-|x| behavior is critical.
pub fn atan_adversarial() -> Vec<f64> {
    const LO: f64 = f64::NEG_INFINITY;
    const HI: f64 = f64::INFINITY;

    let mut out = Vec::new();

    // ── Five-subinterval boundaries ───────────────────────────────────────
    // Each boundary is where the algorithm switches sub-polynomial.
    // The ±3 ULP neighborhood exercises the discontinuity threat.
    for &b in &[7.0_f64 / 16.0, 11.0 / 16.0, 19.0 / 16.0, 39.0 / 16.0] {
        for bits_off in [-3i64, -2, -1, 0, 1, 2, 3] {
            let v = f64::from_bits(b.to_bits().wrapping_add_signed(bits_off));
            if v.is_finite() {
                out.push(v);
                out.push(-v);
            }
        }
    }

    // ── Standard checkpoints ─────────────────────────────────────────────
    // atan(0) = 0, atan(1) = π/4, atan(∞) = π/2, etc.
    for &v in &[
        0.0_f64, 0.5, 1.0, 1.5,
        std::f64::consts::FRAC_1_SQRT_2,  // atan(1/√2) ≈ 0.6155
        3.0_f64.sqrt(),                    // atan(√3) = π/3
        1.0 / 3.0_f64.sqrt(),             // atan(1/√3) = π/6
    ] {
        push_ulp_neighborhood(&mut out, v, -1e20, 1e20);
        push_ulp_neighborhood(&mut out, -v, -1e20, 1e20);
    }

    // ── Large-|x| approach to ±π/2 ───────────────────────────────────────
    // atan(x) = π/2 - 1/x + O(1/x³) for large x.
    // The accuracy question: does the reconstruction of π/2 - atan(1/x)
    // preserve the high bits of π/2?
    for &mag in &[10.0_f64, 100.0, 1e4, 1e8, 1e15, 1e100, 1e300] {
        push_ulp_neighborhood(&mut out, mag, -f64::INFINITY, f64::INFINITY);
        push_ulp_neighborhood(&mut out, -mag, -f64::INFINITY, f64::INFINITY);
    }

    // ── Dense sweep through the principal range ───────────────────────────
    out.extend(linspace(-4.0, 4.0, 1000));
    out.extend(linspace(-1.5, 1.5, 400));  // extra density around 0

    // ── Small |x|: atan(x) ≈ x (Horner accuracy) ──────────────────────────
    for &v in &[f64::MIN_POSITIVE, f64::EPSILON, 1e-100, 1e-15, 1e-8, 1e-4, 0.001] {
        push_ulp_neighborhood(&mut out, v, -1e30, 1e30);
        push_ulp_neighborhood(&mut out, -v, -1e30, 1e30);
    }

    // ── Float landmarks in the atan-relevant range ────────────────────────
    out.extend(float_landmarks(-1e6, 1e6));

    out.retain(|x| x.is_finite());
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
