//! Workup: special-function pole behavior — digamma, gamma, log_gamma, trigamma.
//!
//! ## Why poles matter
//!
//! Special functions at their poles are the adversarial boundary conditions for
//! any downstream consumer that iterates toward a pole. Gamma distribution MLE
//! Newton-Raphson can step into digamma(0) if the shape parameter reaches zero.
//! Bayesian conjugate priors with Dirichlet(α) need digamma(αₖ) for each
//! component; if any αₖ → 0, the pole behavior determines whether the optimizer
//! handles it gracefully or silently computes NaN.
//!
//! ## Correct behavior policy
//!
//! The policy for our special functions at poles:
//!
//! - `digamma(0)` → `f64::NEG_INFINITY` (ψ(x) → -∞ as x → 0⁺; residue = -1)
//! - `digamma(-n)` for n = 1, 2, 3, ... → `f64::NEG_INFINITY` (same residue structure)
//! - `digamma(x)` where x is a non-integer negative → finite (reflection formula)
//! - `log_gamma(x)` at non-positive integers → `f64::INFINITY` (|Γ| → ∞, log|Γ| → ∞)
//! - `log_gamma(x)` at negative non-integers → finite (reflection formula)
//! - `gamma(x)` at non-positive integers → `f64::INFINITY` (pole in magnitude)
//! - `gamma(x)` at negative non-integers → finite with correct sign
//! - `trigamma(x)` for x ≤ 0 → `f64::NAN` (current convention for invalid domain)
//!
//! ## What NOT to return at poles
//!
//! - `NaN` for digamma at poles is wrong — callers cannot distinguish "domain error"
//!   from "pole." `NEG_INFINITY` is the mathematically correct limit and lets callers
//!   use `is_infinite()` to detect and handle the pole.
//! - `INFINITY` for digamma at poles would also be wrong — the limit is -∞, not +∞.
//!
//! ## Consumer-derived critical inputs
//!
//! Downstream consumers that are most sensitive to pole behavior:
//! - Gamma MLE Newton step: `α_new = α - (log(ᾱ) - log(x̄) - ψ(α)) / ψ₁(α)`
//!   If α → 0, ψ(α) → -∞ and ψ₁(α) → +∞. Step size → 0, optimizer stalls.
//! - Dirichlet MLE: same structure, k components can each hit α_k → 0.
//! - incomplete_beta / regularized_gamma: never call log_gamma at poles directly
//!   because their input constraints (a, b > 0) exclude the poles.

use tambear::special_functions::{digamma, trigamma, gamma, log_gamma};

// ─── digamma poles ────────────────────────────────────────────────────────────

/// digamma(0) = -∞, not NaN.
/// ψ(x) has a simple pole at x=0 with residue -1. The limit from either side is -∞.
#[test]
fn digamma_pole_at_zero_is_neg_infinity() {
    let v = digamma(0.0);
    assert!(
        v == f64::NEG_INFINITY,
        "digamma(0.0) should be NEG_INFINITY, got {:?}",
        v
    );
}

/// digamma(0) from positive side: small positive x gives very negative value.
/// Tests that approach from above is consistent with the pole.
#[test]
fn digamma_approaches_neg_infinity_from_above() {
    // ψ(x) ~ -1/x - γ + ... for small x > 0
    // At x=1e-10, ψ(x) ≈ -1/1e-10 = -1e10
    let v = digamma(1e-10);
    assert!(
        v < -1e9,
        "digamma(1e-10) should be very negative (~-1e10), got {:.3e}",
        v
    );
}

/// digamma(-1) = -∞ (pole).
#[test]
fn digamma_pole_at_minus_one_is_neg_infinity() {
    let v = digamma(-1.0);
    assert!(
        v == f64::NEG_INFINITY,
        "digamma(-1.0) should be NEG_INFINITY, got {:?}",
        v
    );
}

/// digamma(-2) = -∞ (pole).
#[test]
fn digamma_pole_at_minus_two_is_neg_infinity() {
    let v = digamma(-2.0);
    assert!(
        v == f64::NEG_INFINITY,
        "digamma(-2.0) should be NEG_INFINITY, got {:?}",
        v
    );
}

/// digamma(-3), digamma(-10): all non-positive integer poles return NEG_INFINITY.
#[test]
fn digamma_poles_at_negative_integers() {
    for n in [3i32, 4, 5, 10, 50] {
        let x = -(n as f64);
        let v = digamma(x);
        assert!(
            v == f64::NEG_INFINITY,
            "digamma({:.0}) should be NEG_INFINITY at pole, got {:?}",
            x,
            v
        );
    }
}

/// digamma at non-integer negative values: finite (reflection formula applies).
/// ψ(-0.5) = ψ(1.5) - π·cot(-π/2) = ψ(1.5) - 0 = ψ(1.5)
/// ψ(1.5) = ψ(0.5) + 1/0.5 = (-γ - 2·ln2) + 2 ≈ -0.5772 - 1.3863 + 2 = 0.0365
/// More precisely: ψ(-0.5) = ψ(1.5) + π·cot(-π·(-0.5)) = ψ(1.5)
/// Wait: reflection is ψ(x) = ψ(1-x) - π·cot(πx)
/// At x=-0.5: ψ(-0.5) = ψ(1.5) - π·cot(-π/2) = ψ(1.5) - 0 = ψ(1.5)
/// ψ(1.5) = digamma(1.5). Exact value: -γ + 2(1 - ln 2) = 0.03648997...
/// scipy.special.digamma(-0.5) = 0.03648997161...
#[test]
fn digamma_negative_half_integer_is_finite() {
    let v = digamma(-0.5);
    assert!(
        v.is_finite(),
        "digamma(-0.5) should be finite (non-integer negative), got {:?}",
        v
    );
    // ψ(-0.5) = ψ(1.5) ≈ 0.03649
    let expected = 0.03648997161;
    assert!(
        (v - expected).abs() < 1e-6,
        "digamma(-0.5) = {:.8}, expected ≈ {:.8}",
        v,
        expected
    );
}

/// digamma(-1.5): non-integer negative, finite.
/// ψ(-1.5) = ψ(2.5) - π·cot(-1.5π)
/// ψ(2.5) = ψ(1.5) + 1/1.5 = 0.03649 + 0.66667 = 0.70316
/// cot(-1.5π) = cot(-270°) = 0
/// ψ(-1.5) = 0.70316 - 0 = 0.70316...
/// scipy.special.digamma(-1.5) ≈ 0.70315664...
#[test]
fn digamma_minus_three_half_is_finite() {
    let v = digamma(-1.5);
    assert!(
        v.is_finite(),
        "digamma(-1.5) should be finite (non-integer negative), got {:?}",
        v
    );
    let expected = 0.70315664;
    assert!(
        (v - expected).abs() < 1e-5,
        "digamma(-1.5) = {:.8}, expected ≈ {:.8}",
        v,
        expected
    );
}

/// digamma(NaN) → NaN (not an error, just NaN propagation).
#[test]
fn digamma_nan_propagates() {
    assert!(digamma(f64::NAN).is_nan(), "digamma(NaN) should be NaN");
}

/// digamma at +∞: ψ(x) ~ ln(x) → +∞.
#[test]
fn digamma_pos_infinity_is_pos_infinity() {
    let v = digamma(f64::INFINITY);
    assert!(
        v == f64::INFINITY,
        "digamma(+∞) should be +∞, got {:?}",
        v
    );
}

// ─── gamma poles ─────────────────────────────────────────────────────────────

/// gamma(-1) = +∞ (pole in magnitude).
/// Γ(x) = exp(log_gamma(x)). log_gamma(-1) = INFINITY → gamma(-1) = INFINITY.
#[test]
fn gamma_pole_at_minus_one_is_infinity() {
    let v = gamma(-1.0);
    assert!(
        v == f64::INFINITY,
        "gamma(-1.0) should be +INFINITY at pole, got {:?}",
        v
    );
}

/// gamma at non-positive integers: all poles return INFINITY.
#[test]
fn gamma_poles_at_non_positive_integers() {
    for n in [0i32, -1, -2, -3, -10] {
        let x = n as f64;
        let v = gamma(x);
        assert!(
            v == f64::INFINITY,
            "gamma({:.0}) should be INFINITY at pole, got {:?}",
            x,
            v
        );
    }
}

/// gamma at negative non-integers: finite, but sign is LOST.
///
/// True value: Γ(-0.5) = -2√π ≈ -3.5449...
/// Our implementation: gamma(x) = exp(log_gamma(x)) = exp(ln|Γ(x)|) = |Γ(x)|.
/// log_gamma computes ln|Γ(x)| via the reflection formula and returns the
/// unsigned logarithm. exp() of that is always positive, so the sign is lost.
///
/// This is a KNOWN LIMITATION, documented here. For applications that need the
/// signed Γ(x) at negative non-integers, use the reflection formula directly:
///   sgn(Γ(x)) = sgn(sin(πx)) · (-1)^floor(|x|)  (for -1 < x < 0: negative)
///
/// The test asserts the ACTUAL behavior (unsigned magnitude), not the ideal behavior.
/// This serves as a regression test — if the implementation changes to return the
/// correct signed value, this test should be updated to match.
#[test]
fn gamma_negative_half_magnitude_no_sign() {
    let v = gamma(-0.5);
    // Returns |Γ(-0.5)| = 2√π (magnitude, sign lost)
    let expected_magnitude = 2.0 * std::f64::consts::PI.sqrt();
    assert!(
        v.is_finite(),
        "gamma(-0.5) should be finite, got {:?}",
        v
    );
    assert!(
        (v - expected_magnitude).abs() < 1e-10,
        "gamma(-0.5) = {:.10}, expected magnitude {:.10} (NOTE: true value is negative, sign lost)",
        v,
        expected_magnitude
    );
    // Document the limitation explicitly — true value is negative
    // Γ(-0.5) = -2√π but we return +2√π
    assert!(v > 0.0, "gamma(-0.5) returns positive (sign lost): actual value is negative");
}

/// gamma(-1.5) = 4√π/3 ≈ 2.3632718...
/// Γ(-1.5) = Γ(-0.5) / (-1.5) = (-2√π) / (-1.5) = 4√π/3
#[test]
fn gamma_minus_three_halves_oracle() {
    let v = gamma(-1.5);
    let expected = 4.0 * std::f64::consts::PI.sqrt() / 3.0;
    assert!(
        (v - expected).abs() < 1e-10,
        "gamma(-1.5) = {:.10}, expected {:.10}",
        v,
        expected
    );
}

/// gamma(NaN) → NaN.
#[test]
fn gamma_nan_propagates() {
    assert!(gamma(f64::NAN).is_nan(), "gamma(NaN) should be NaN");
}

// ─── log_gamma poles ──────────────────────────────────────────────────────────

/// log_gamma at non-positive integers: INFINITY (|Γ| → ∞ → ln|Γ| → ∞).
#[test]
fn log_gamma_poles_at_non_positive_integers() {
    for n in [0i32, -1, -2, -3, -10] {
        let x = n as f64;
        let v = log_gamma(x);
        assert!(
            v == f64::INFINITY,
            "log_gamma({:.0}) should be INFINITY at pole, got {:?}",
            x,
            v
        );
    }
}

/// log_gamma at negative non-integers: finite.
/// log_gamma(-0.5) = ln|Γ(-0.5)| = ln(2√π) ≈ ln(3.5449) ≈ 1.2655...
/// More precisely: ln(2) + 0.5·ln(π) ≈ 0.6931 + 0.5724 = 1.2655
#[test]
fn log_gamma_negative_half_is_finite() {
    let v = log_gamma(-0.5);
    let expected = 2.0_f64.ln() + 0.5 * std::f64::consts::PI.ln();
    assert!(
        v.is_finite(),
        "log_gamma(-0.5) should be finite, got {:?}",
        v
    );
    assert!(
        (v - expected).abs() < 1e-10,
        "log_gamma(-0.5) = {:.10}, expected {:.10}",
        v,
        expected
    );
}

/// log_gamma(-1.5): ln|Γ(-1.5)| = ln(4√π/3) ≈ ln(2.3633) ≈ 0.8600...
/// scipy.special.gammaln(-1.5) ≈ 0.8600471857...
#[test]
fn log_gamma_minus_three_halves_oracle() {
    let v = log_gamma(-1.5);
    let expected = (4.0 * std::f64::consts::PI.sqrt() / 3.0).ln();
    assert!(
        (v - expected).abs() < 1e-10,
        "log_gamma(-1.5) = {:.10}, expected {:.10}",
        v,
        expected
    );
}

// ─── trigamma behavior near poles ────────────────────────────────────────────

/// trigamma at integer poles (x = 0, -1, -2, ...): returns +∞.
/// ψ₁(x) has second-order poles at non-positive integers with residue +1/x².
/// The limit from either side is +∞ (not -∞ — second-order poles are unsigned in the limit).
#[test]
fn trigamma_integer_poles_are_pos_infinity() {
    for &x in &[0.0_f64, -1.0, -2.0, -3.0, -10.0] {
        let v = trigamma(x);
        assert!(
            v == f64::INFINITY,
            "trigamma({}) should be +INFINITY at second-order pole, got {:?}",
            x,
            v
        );
    }
}

/// trigamma at negative non-integers: finite (reflection formula applies).
/// ψ₁(x) + ψ₁(1-x) = π²/sin²(πx). At x=-0.5: sin(-π/2)=-1, sin²=1.
/// ψ₁(-0.5) = π²/1 - ψ₁(1.5) = π² - (π²/6 - 1 + 1/(0.5²)) = π² - π²/6 + 1 - 4
/// More directly: ψ₁(-0.5) = π²/sin²(-π/2) - ψ₁(1.5) = π² - ψ₁(1.5)
/// ψ₁(1.5) = ψ₁(0.5) - 1/0.25 = π²/2 - 4 ≈ 4.9348 - 4 = 0.9348
/// ψ₁(-0.5) = π² - 0.9348 ≈ 9.8696 - 0.9348 = 8.9348
/// scipy.special.polygamma(1, -0.5) ≈ 8.934802...
#[test]
fn trigamma_negative_half_integer_is_finite() {
    let v = trigamma(-0.5);
    assert!(
        v.is_finite(),
        "trigamma(-0.5) should be finite (non-integer negative), got {:?}",
        v
    );
    // ψ₁(-0.5) ≈ 8.9348
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    let trigamma_1p5 = pi2 / 6.0 - 1.0 + 4.0; // ψ₁(1.5) = ψ₁(0.5) - 1/0.5² = π²/2 - 4... recompute
    // Use known value: scipy gives 8.934802...
    let expected = 8.934802;
    assert!(
        (v - expected).abs() < 1e-4,
        "trigamma(-0.5) = {:.6}, expected ≈ {:.6}",
        v,
        expected
    );
    let _ = trigamma_1p5; // suppress unused warning
}

/// trigamma approaches +∞ as x → 0⁺.
/// ψ₁(x) ~ 1/x² for small x > 0.
#[test]
fn trigamma_approaches_infinity_from_above() {
    let v = trigamma(1e-6);
    // ψ₁(1e-6) ≈ 1/(1e-6)² = 1e12
    assert!(
        v > 1e11,
        "trigamma(1e-6) should be very large (~1e12), got {:.3e}",
        v
    );
}

// ─── Consistency: poles and recurrence relations ──────────────────────────────

/// The recurrence ψ(x+1) = ψ(x) + 1/x connects finite values to pole behavior.
/// At x → 0⁺: ψ(1) = ψ(0⁺) + 1/0⁺ — pole makes the recurrence work.
/// Test: digamma(ε) + 1/ε → digamma(1+ε) as ε → 0.
#[test]
fn digamma_recurrence_near_pole_consistent() {
    let eps = 1e-8_f64;
    // ψ(1 + ε) ≈ -γ (ψ(1) = -γ = -0.5772...)
    let via_recurrence = digamma(eps) + 1.0 / eps;
    let direct = digamma(1.0 + eps);
    assert!(
        (via_recurrence - direct).abs() < 1e-4,
        "recurrence near pole: digamma({eps})+1/{eps} = {via_recurrence:.6}, \
         digamma(1+{eps}) = {direct:.6}"
    );
}

/// log_gamma reflection: for x ∈ (-1, 0) \ {0}, ln|Γ(x)| = ln(π) - ln|sin(πx)| - ln|Γ(1-x)|.
/// Verify at x = -0.5: ln|Γ(-0.5)| should equal ln(2√π) via both direct and reflection.
#[test]
fn log_gamma_reflection_formula_minus_half() {
    let x = -0.5_f64;
    let direct = log_gamma(x);
    // Reflection: ln|Γ(-0.5)| = ln(π) - ln|sin(-π/2)| - ln|Γ(1.5)|
    // |sin(-π/2)| = 1, ln(1) = 0
    // Γ(1.5) = 0.5 · Γ(0.5) = √π/2, ln(Γ(1.5)) = 0.5·ln(π) - ln(2)
    let lg_1p5 = log_gamma(1.5); // positive argument, well-conditioned
    let via_reflection = std::f64::consts::PI.ln()
        - (std::f64::consts::PI * x).sin().abs().ln()
        - lg_1p5;
    assert!(
        (direct - via_reflection).abs() < 1e-10,
        "log_gamma(-0.5) direct={:.10} vs reflection={:.10}",
        direct,
        via_reflection
    );
}
