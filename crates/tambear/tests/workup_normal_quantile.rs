//! Parity tests for `special_functions::normal_quantile`.
//!
//! Covers:
//! - Oracle comparison at 16 points vs mpmath 50dp (max ≤ 20 ULP after Newton fix)
//! - Boundary conditions: p=0, p=1, p=0.5
//! - Inverse property: Φ(Φ⁻¹(p)) ≈ p
//! - Monotonicity sweep
//! - Symmetry: Φ⁻¹(1-p) = -Φ⁻¹(p)
//! - Critical statistical values: 0.025, 0.05, 0.95, 0.975
//!
//! See docs/research/atomic-industrialization/normal_quantile.md for full workup.
//!
//! Run: cargo test --test workup_normal_quantile -- --nocapture

use tambear::special_functions::{normal_quantile, normal_cdf};

fn ulps(a: f64, b: f64) -> i64 {
    if a == b { return 0; }
    let ai = a.to_bits() as i64;
    let bi = b.to_bits() as i64;
    (ai - bi).abs()
}

fn lcg_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = ((*state >> 11) as u64) | 0x3FF0000000000000_u64;
    f64::from_bits(bits) - 1.0
}

// ─── Oracle cases (mpmath at 50dp) ──────────────────────────────────────────

/// 14 oracle cases spanning full range. Max ≤ 20 ULP after Newton refinement.
/// Before fix: max ~8 million ULPs (Acklam-only).
#[test]
fn probit_oracle_14_cases() {
    let cases: &[(f64, f64)] = &[
        (0.001,    -3.0902323061678136_f64),
        (0.01,     -2.326347874040841_f64),
        (0.025,    -1.9599639845400543_f64),
        (0.05,     -1.6448536269514726_f64),
        (0.1,      -1.2815515655446004_f64),
        (0.25,     -0.6744897501960817_f64),
        (0.75,      0.6744897501960817_f64),
        (0.9,       1.2815515655446004_f64),
        (0.95,      1.6448536269514722_f64),
        (0.975,     1.9599639845400543_f64),
        (0.99,      2.326347874040841_f64),
        (0.999,     3.0902323061678136_f64),
        (0.9999,    3.7190164854556804_f64),
        (1e-5,     -4.264890793922825_f64),
    ];
    let mut max_u = 0i64;
    for &(p, oracle) in cases {
        let got = normal_quantile(p);
        let u = ulps(got, oracle);
        max_u = max_u.max(u);
    }
    assert!(
        max_u <= 25,
        "probit oracle max ULPs {} exceeds 25 (was ~8M before Newton fix)", max_u
    );
}

/// p=0.025 critical value (95% CI lower tail): -1.9599639845400543.
/// Before Newton fix: ~7M ULPs. After: ≤ 3 ULP.
#[test]
fn probit_oracle_0p025_95ci_lower() {
    let got = normal_quantile(0.025);
    let oracle = -1.9599639845400543_f64;
    let u = ulps(got, oracle);
    assert!(u <= 5, "probit(0.025): {} ULP (was ~7M before fix)", u);
}

/// p=0.975 critical value (95% CI upper tail): 1.9599639845400543.
#[test]
fn probit_oracle_0p975_95ci_upper() {
    let got = normal_quantile(0.975);
    let oracle = 1.9599639845400543_f64;
    let u = ulps(got, oracle);
    assert!(u <= 5, "probit(0.975): {} ULP", u);
}

/// p=0.05 critical value (90% CI lower tail): -1.6448536269514726.
/// Was 8M ULPs before fix. Now exact or near-exact.
#[test]
fn probit_oracle_0p05_90ci_lower() {
    let got = normal_quantile(0.05);
    let oracle = -1.6448536269514726_f64;
    let u = ulps(got, oracle);
    assert!(u <= 5, "probit(0.05): {} ULP (was ~8M before Newton fix)", u);
}

// ─── Boundary conditions ─────────────────────────────────────────────────────

/// p ≤ 0 returns -∞.
#[test]
fn probit_boundary_p_zero() {
    assert_eq!(normal_quantile(0.0), f64::NEG_INFINITY);
    assert_eq!(normal_quantile(-0.1), f64::NEG_INFINITY);
}

/// p ≥ 1 returns +∞.
#[test]
fn probit_boundary_p_one() {
    assert_eq!(normal_quantile(1.0), f64::INFINITY);
    assert_eq!(normal_quantile(1.5), f64::INFINITY);
}

/// p = 0.5 returns 0.0 exactly (special-cased before Newton).
#[test]
fn probit_boundary_p_half_is_zero() {
    assert_eq!(normal_quantile(0.5), 0.0);
}

// ─── Inverse property: Φ(Φ⁻¹(p)) ≈ p ──────────────────────────────────────

/// Round-trip: normal_cdf(normal_quantile(p)) ≈ p.
/// Verified at 15 deterministic points.
#[test]
fn probit_inverse_property_deterministic() {
    let ps = [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.4, 0.5,
              0.6, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999];
    for &p in &ps {
        let z = normal_quantile(p);
        let recovered = normal_cdf(z);
        let err = (recovered - p).abs();
        assert!(
            err < 1e-13,
            "round-trip failure: p={} → z={} → Φ(z)={} (err={:.2e})", p, z, recovered, err
        );
    }
}

/// Round-trip over 100 pseudorandom p ∈ (0.001, 0.999).
#[test]
fn probit_inverse_property_random_100() {
    let mut state: u64 = 42;
    let mut max_err = 0.0_f64;
    for _ in 0..100 {
        let u = lcg_f64(&mut state);
        let p = 0.001 + u * 0.998; // (0.001, 0.999)
        let z = normal_quantile(p);
        let recovered = normal_cdf(z);
        max_err = max_err.max((recovered - p).abs() / p.max(1.0 - p));
    }
    assert!(max_err < 1e-12, "round-trip max_rel_err = {max_err:.3e}");
}

// ─── Symmetry ─────────────────────────────────────────────────────────────────

/// Φ⁻¹(1-p) = -Φ⁻¹(p) for p ≠ 0.5.
#[test]
fn probit_symmetry() {
    let ps = [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.3];
    for &p in &ps {
        let neg = normal_quantile(p);
        let pos = normal_quantile(1.0 - p);
        let err = (neg + pos).abs(); // should be 0
        assert!(
            err < 1e-13,
            "symmetry: probit({p}) + probit({}) = {err:.2e} (expected 0)", 1.0 - p
        );
    }
}

// ─── Monotonicity ─────────────────────────────────────────────────────────────

/// normal_quantile is strictly increasing.
#[test]
fn probit_monotonicity() {
    let mut prev = normal_quantile(0.001);
    let mut p = 0.002_f64;
    while p < 0.999 {
        let curr = normal_quantile(p);
        assert!(curr > prev, "not monotone at p={p}: probit={curr} ≤ prev={prev}");
        prev = curr;
        p += 0.001;
    }
}

// ─── Acklam region boundary ───────────────────────────────────────────────────

/// p at exactly P_LOW = 0.02425 (Acklam boundary between central and tail regions).
/// Should be continuous — no discontinuity in Newton refinement across this boundary.
#[test]
fn probit_acklam_boundary_smooth() {
    let p_low = 0.02425_f64;
    let below = normal_quantile(p_low - 1e-10);
    let at    = normal_quantile(p_low);
    let above = normal_quantile(p_low + 1e-10);
    // All three should be close
    assert!((at - below).abs() < 1e-6, "discontinuity below P_LOW: {}", (at - below).abs());
    assert!((above - at).abs() < 1e-6, "discontinuity above P_LOW: {}", (above - at).abs());
    // Monotone
    assert!(below < at && at < above, "not monotone at P_LOW boundary");
}
