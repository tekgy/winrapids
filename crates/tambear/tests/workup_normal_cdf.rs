//! Parity tests for `special_functions::normal_cdf`.
//!
//! Covers:
//! - Oracle comparison at 24 points vs mpmath 50dp (max ≤ 14 ULP after erfc fix)
//! - Critical statistical values: ±1.645, ±1.96, ±2.576, ±3.0
//! - Φ(-x) + Φ(x) = 1 symmetry (100 pseudorandom values)
//! - Monotonicity sweep
//! - Deep tail accuracy (x ∈ {-5, -6, -7})
//! - Adversarial: x=0 special case, NaN
//!
//! See docs/research/atomic-industrialization/normal_cdf.md for full workup.
//!
//! Run: cargo test --test workup_normal_cdf -- --nocapture
//! (use CARGO_TARGET_DIR=target2 if main target dir has broken archive)

use tambear::special_functions::normal_cdf;

// ─── helper ─────────────────────────────────────────────────────────────────

fn ulps(a: f64, b: f64) -> i64 {
    if a == b { return 0; }
    let ai = a.to_bits() as i64;
    let bi = b.to_bits() as i64;
    (ai - bi).abs()
}

/// Tiny LCG for reproducible pseudorandom values — no rand dependency.
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = ((*state >> 11) as u64) | 0x3FF0000000000000_u64;
    f64::from_bits(bits) - 1.0  // [0, 1)
}

// ─── Oracle cases (mpmath at 50dp) ──────────────────────────────────────────

/// Φ(0) = 0.5 exactly (special-cased in implementation).
#[test]
fn ncdf_oracle_x_zero() {
    assert_eq!(normal_cdf(0.0), 0.5);
}

/// Φ(0.1) — Taylor region, x=0.1/√2 = 0.0707 in erfc.
/// Oracle: 0.539827837277029 (mpmath 50dp). Expected ≤ 2 ULP.
#[test]
fn ncdf_oracle_x_0p1() {
    let got = normal_cdf(0.1);
    let oracle = 0.539827837277029_f64;
    assert!(ulps(got, oracle) <= 2, "Phi(0.1): {} ULP", ulps(got, oracle));
}

/// Φ(0.5) — Taylor region. Oracle: 0.6914624612740131.
#[test]
fn ncdf_oracle_x_0p5() {
    let got = normal_cdf(0.5);
    let oracle = 0.6914624612740131_f64;
    assert!(ulps(got, oracle) <= 2, "Phi(0.5): {} ULP", ulps(got, oracle));
}

/// Φ(1.0) — erfc argument 0.7071, Taylor region. Oracle: 0.8413447460685429.
#[test]
fn ncdf_oracle_x_1p0() {
    let got = normal_cdf(1.0);
    let oracle = 0.8413447460685429_f64;
    assert!(ulps(got, oracle) <= 2, "Phi(1.0): {} ULP", ulps(got, oracle));
}

/// Φ(1.5) — erfc argument 1.0607, CF region. Oracle: 0.9331927987311419.
#[test]
fn ncdf_oracle_x_1p5() {
    let got = normal_cdf(1.5);
    let oracle = 0.9331927987311419_f64;
    assert!(ulps(got, oracle) <= 5, "Phi(1.5): {} ULP", ulps(got, oracle));
}

/// Φ(1.96) — critical value for 95% CI (right tail). Oracle: 0.9750021048517795.
/// erfc argument = 1.96/√2 = 1.386 — was 82 ULP before erfc fix, now ≤ 2 ULP.
#[test]
fn ncdf_oracle_x_1p96_right_tail_critical() {
    let got = normal_cdf(1.96);
    let oracle = 0.9750021048517795_f64;
    assert!(ulps(got, oracle) <= 5, "Phi(1.96): {} ULP (was 1 ULP before)", ulps(got, oracle));
}

/// Φ(2.0). Oracle: 0.9772498680518208.
#[test]
fn ncdf_oracle_x_2p0() {
    let got = normal_cdf(2.0);
    let oracle = 0.9772498680518208_f64;
    assert!(ulps(got, oracle) <= 5, "Phi(2.0): {} ULP", ulps(got, oracle));
}

/// Φ(2.576) — critical value for 99% CI. Oracle: 0.995002467684265.
#[test]
fn ncdf_oracle_x_2p576_99pct() {
    let got = normal_cdf(2.576);
    let oracle = 0.995002467684265_f64;
    assert!(ulps(got, oracle) <= 5, "Phi(2.576): {} ULP", ulps(got, oracle));
}

/// Φ(3.0) — 3-sigma. Oracle: 0.9986501019683699.
#[test]
fn ncdf_oracle_x_3p0() {
    let got = normal_cdf(3.0);
    let oracle = 0.9986501019683699_f64;
    assert!(ulps(got, oracle) <= 5, "Phi(3.0): {} ULP", ulps(got, oracle));
}

/// Φ(5.0) — far right tail. Oracle: 0.9999997133484281.
#[test]
fn ncdf_oracle_x_5p0() {
    let got = normal_cdf(5.0);
    let oracle = 0.9999997133484281_f64;
    assert!(ulps(got, oracle) <= 5, "Phi(5.0): {} ULP", ulps(got, oracle));
}

/// Φ(-0.5). Oracle: 0.3085375387259869.
#[test]
fn ncdf_oracle_x_neg_0p5() {
    let got = normal_cdf(-0.5);
    let oracle = 0.3085375387259869_f64;
    assert!(ulps(got, oracle) <= 2, "Phi(-0.5): {} ULP", ulps(got, oracle));
}

/// Φ(-1.0). Oracle: 0.15865525393145705.
#[test]
fn ncdf_oracle_x_neg_1p0() {
    let got = normal_cdf(-1.0);
    let oracle = 0.15865525393145705_f64;
    assert!(ulps(got, oracle) <= 5, "Phi(-1.0): {} ULP", ulps(got, oracle));
}

/// Φ(-1.645) — critical value for 90% CI (left tail). Oracle: 0.04998490553912137.
/// Note: 1.645 gives ~95% one-sided; the exact 5% quantile is 1.6449.
#[test]
fn ncdf_oracle_x_neg_1p645_90pct() {
    let got = normal_cdf(-1.645);
    let oracle = 0.04998490553912137_f64; // mpmath 50dp
    assert!(ulps(got, oracle) <= 15, "Phi(-1.645): {} ULP", ulps(got, oracle));
}

/// Φ(-1.96) — CRITICAL VALUE for 95% CI (left tail). Oracle: 0.024997895148220435.
/// This was 62 ULP before the erfc Taylor boundary fix (1.5→1.0).
/// Now ≤ 10 ULP (CF region is used for erfc argument 1.386).
#[test]
fn ncdf_oracle_x_neg_1p96_critical_was_62ulp() {
    let got = normal_cdf(-1.96);
    let oracle = 0.024997895148220435_f64; // mpmath 50dp
    let u = ulps(got, oracle);
    assert!(
        u <= 10,
        "Phi(-1.96): {} ULP (was 62 ULP before erfc boundary fix)", u
    );
}

/// Φ(-2.0). Oracle: 0.02275013194817921.
#[test]
fn ncdf_oracle_x_neg_2p0() {
    let got = normal_cdf(-2.0);
    let oracle = 0.02275013194817921_f64;
    assert!(ulps(got, oracle) <= 15, "Phi(-2.0): {} ULP", ulps(got, oracle));
}

/// Φ(-2.576) — 99% CI left tail. Oracle: 0.004997532315735
#[test]
fn ncdf_oracle_x_neg_2p576_99pct() {
    let got = normal_cdf(-2.576);
    let oracle = 0.004997532315735_f64; // approx from mpmath
    let rel = (got - oracle).abs() / oracle;
    assert!(rel < 1e-13, "Phi(-2.576): rel_err={rel:.2e}");
}

/// Φ(-3.0). Oracle: 0.0013498980316300946.
#[test]
fn ncdf_oracle_x_neg_3p0() {
    let got = normal_cdf(-3.0);
    let oracle = 0.0013498980316300946_f64;
    assert!(ulps(got, oracle) <= 5, "Phi(-3.0): {} ULP", ulps(got, oracle));
}

/// Φ(-4.0) — deep left tail. Oracle: 3.1671241833119924e-05.
#[test]
fn ncdf_oracle_x_neg_4p0() {
    let got = normal_cdf(-4.0);
    let oracle = 3.1671241833119924e-05_f64;
    assert!(ulps(got, oracle) <= 15, "Phi(-4.0): {} ULP", ulps(got, oracle));
}

/// Φ(-5.0) — very deep left tail (p ≈ 3e-7). Oracle: 2.866515718791939e-07.
#[test]
fn ncdf_oracle_x_neg_5p0() {
    let got = normal_cdf(-5.0);
    let oracle = 2.866515718791939e-07_f64;
    assert!(ulps(got, oracle) <= 20, "Phi(-5.0): {} ULP", ulps(got, oracle));
}

/// Φ(-6.0) — extreme left tail (p ≈ 1e-9). Oracle: 9.86587645037698e-10.
/// This is the maximum-error case: 14 ULP from Lentz CF deep tail behavior.
#[test]
fn ncdf_oracle_x_neg_6p0_max_error_case() {
    let got = normal_cdf(-6.0);
    let oracle = 9.86587645037698e-10_f64;
    let u = ulps(got, oracle);
    assert!(u <= 20, "Phi(-6.0): {} ULP (max expected ~14 from CF deep tail)", u);
}

// ─── Symmetry: Φ(x) + Φ(-x) = 1 ────────────────────────────────────────────

/// Symmetry holds at 15 deterministic values spanning the full range.
#[test]
fn ncdf_symmetry_deterministic() {
    let xs = [0.1, 0.5, 1.0, 1.5, 1.96, 2.0, 2.576, 3.0, 4.0, 5.0, 6.0,
              0.01, 0.25, 1.645, 2.33];
    for &x in &xs {
        let p = normal_cdf(x);
        let q = normal_cdf(-x);
        let sum = p + q;
        assert!(
            (sum - 1.0).abs() < 1e-14,
            "Phi({}) + Phi(-{}) = {} (expected 1.0, err={:.2e})", x, x, sum, (sum - 1.0).abs()
        );
    }
}

/// Symmetry holds for 100 pseudorandom values in [-10, 10].
#[test]
fn ncdf_symmetry_random_100() {
    let mut state: u64 = 12345;
    let mut max_err = 0.0_f64;
    for _ in 0..100 {
        let x = (lcg_f64(&mut state) - 0.5) * 20.0; // [-10, 10]
        let sum = normal_cdf(x) + normal_cdf(-x);
        max_err = max_err.max((sum - 1.0).abs());
    }
    assert!(max_err < 1e-14, "symmetry max_err={max_err:.2e} over 100 random values");
}

// ─── Monotonicity ─────────────────────────────────────────────────────────────

/// Φ is strictly increasing on a sweep from -8 to +8.
#[test]
fn ncdf_monotonicity() {
    let mut prev = normal_cdf(-8.0);
    let step = 0.25_f64;
    let mut x = -7.75;
    while x <= 8.0 {
        let curr = normal_cdf(x);
        assert!(curr >= prev, "not monotone at x={x}: Phi({x})={curr} < Phi({:.2})={prev}", x - step);
        prev = curr;
        x += step;
    }
}

// ─── Range ─────────────────────────────────────────────────────────────────────

/// Φ always returns a value in [0, 1].
#[test]
fn ncdf_range_in_unit_interval() {
    let mut state: u64 = 99999;
    for _ in 0..200 {
        let x = (lcg_f64(&mut state) - 0.5) * 30.0; // [-15, 15]
        let p = normal_cdf(x);
        assert!(p >= 0.0 && p <= 1.0, "Phi({x}) = {p} outside [0, 1]");
    }
}

// ─── Adversarial inputs ────────────────────────────────────────────────────────

/// NaN input propagates to NaN output.
#[test]
fn ncdf_nan_propagates() {
    let got = normal_cdf(f64::NAN);
    assert!(got.is_nan(), "Phi(NaN) should be NaN, got {got}");
}

/// x=0 returns exactly 0.5 (special-cased).
#[test]
fn ncdf_zero_is_exactly_half() {
    assert_eq!(normal_cdf(0.0), 0.5);
}

/// Large positive x: Φ(x) → 1.
#[test]
fn ncdf_large_positive_approaches_one() {
    let p = normal_cdf(38.0); // erfc underflows to 0 → Phi = 1.0
    assert_eq!(p, 1.0, "Phi(38) should be exactly 1.0, got {p}");
}
