//! Special functions — numerical infrastructure for hypothesis testing.
//!
//! ## What's here
//!
//! - `erf`, `erfc` — Abramowitz & Stegun rational approximation (max error < 1.5×10⁻⁷)
//! - `log_gamma` — Lanczos approximation (g=7, 9-term)
//! - `digamma`, `trigamma` — asymptotic expansion + recurrence (error < 1e-12)
//! - `regularized_incomplete_beta` — Lentz continued fraction + series
//! - `regularized_incomplete_gamma` — series (lower) + continued fraction (upper)
//! - `normal_cdf`, `normal_sf` — standard normal CDF/survival via erfc
//! - `t_cdf` — Student-t via incomplete beta
//! - `f_cdf` — F-distribution via incomplete beta
//! - `chi2_cdf` — chi-square via incomplete gamma
//!
//! ## Why from scratch
//!
//! Every p-value in the library flows through these. No vendor dependency.
//! Same bit-exact result on CUDA, Vulkan, Metal, CPU. These are the
//! irreducible numerical atoms; everything else composes from them.
//!
//! ## Accuracy
//!
//! All functions are validated against known values. Relative error < 1e-10
//! for gamma/beta functions, < 1.5e-7 for erf (sufficient for p-values).

// ═══════════════════════════════════════════════════════════════════════════
// Error function
// ═══════════════════════════════════════════════════════════════════════════

/// Error function: erf(x) = (2/√π) ∫₀ˣ e⁻ᵗ² dt.
///
/// Abramowitz & Stegun approximation 7.1.26, max |ε| < 1.5 × 10⁻⁷.
pub fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Complementary error function: erfc(x) = 1 - erf(x).
///
/// Uses direct formula for positive x to avoid catastrophic cancellation.
pub fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let poly = t * (0.254829592
            + t * (-0.284496736
            + t * (1.421413741
            + t * (-1.453152027
            + t * 1.061405429))));
        poly * (-x * x).exp()
    } else {
        2.0 - erfc(-x)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Log-gamma (Lanczos)
// ═══════════════════════════════════════════════════════════════════════════

/// Natural log of the gamma function: ln Γ(x).
///
/// Lanczos approximation with g=7, 9-term coefficients.
/// Valid for x > 0. Returns f64::INFINITY for x ≤ 0.
pub fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 { return f64::INFINITY; }

    // Coefficients from Paul Godfrey's compilation (g=7, n=9)
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1-x) = π / sin(πx)
        let lsin = (std::f64::consts::PI * x).sin().abs().ln();
        std::f64::consts::PI.ln() - lsin - log_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut ag = C[0];
        for i in 1..9 {
            ag += C[i] / (x + i as f64);
        }
        let t = x + 7.5; // x + g + 0.5
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + ag.ln()
    }
}

/// Gamma function: Γ(x) = exp(ln Γ(x)).
pub fn gamma(x: f64) -> f64 {
    log_gamma(x).exp()
}

/// Log of the beta function: ln B(a,b) = ln Γ(a) + ln Γ(b) - ln Γ(a+b).
pub fn log_beta(a: f64, b: f64) -> f64 {
    log_gamma(a) + log_gamma(b) - log_gamma(a + b)
}

// ═══════════════════════════════════════════════════════════════════════════
// Digamma and trigamma functions
// ═══════════════════════════════════════════════════════════════════════════

/// Digamma function: ψ(x) = d/dx ln Γ(x) = Γ'(x) / Γ(x).
///
/// Uses asymptotic expansion for large x, recurrence ψ(x+1) = ψ(x) + 1/x
/// to shift small x into the asymptotic region. Relative error < 1e-12
/// for x > 0.
///
/// Required for Gamma distribution MLE (the optimality condition is
/// ψ(α) = ln(x̄) - mean(ln x), which is Kingdom C τ=0: unique solution,
/// iterative via Newton's method with trigamma as the Hessian).
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() || x == 0.0 {
        return f64::NAN;
    }

    // ψ(x) has poles at non-positive integers.
    // Widen check: tan(nπ) ≈ ε_mach in f64, causing 10^16 blowup in reflection formula.
    if x < 0.0 {
        let rounded = x.round();
        if rounded <= 0.0 && (x - rounded).abs() < 1e-12 {
            return f64::NAN;
        }
    }

    // Reflection formula for x < 0: ψ(1-x) - ψ(x) = π·cot(πx)
    if x < 0.0 {
        return digamma(1.0 - x) - std::f64::consts::PI / (std::f64::consts::PI * x).tan();
    }

    // Shift x into asymptotic region (x ≥ 8) using recurrence ψ(x+1) = ψ(x) + 1/x
    let mut result = 0.0;
    let mut x = x;
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion: ψ(x) ~ ln(x) - 1/(2x) - Σ B_{2k}/(2k·x^{2k})
    // Bernoulli numbers B_2=1/6, B_4=-1/30, B_6=1/42, B_8=-1/30, B_10=5/66
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    result += x.ln() - 0.5 * inv_x
        - inv_x2 * (1.0 / 12.0
        - inv_x2 * (1.0 / 120.0
        - inv_x2 * (1.0 / 252.0
        - inv_x2 * (1.0 / 240.0
        - inv_x2 * (5.0 / 660.0)))));
    result
}

/// Trigamma function: ψ₁(x) = d²/dx² ln Γ(x) = dψ/dx.
///
/// Uses asymptotic expansion for large x, recurrence ψ₁(x+1) = ψ₁(x) - 1/x².
/// Required as the Hessian in Newton iterations for Gamma MLE.
pub fn trigamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }

    // Shift into asymptotic region using recurrence ψ₁(x+1) = ψ₁(x) - 1/x²
    let mut result = 0.0;
    let mut x = x;
    while x < 8.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }

    // Asymptotic expansion: ψ₁(x) ~ 1/x + 1/(2x²) + Σ B_{2k}/x^{2k+1}
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    result += inv_x + 0.5 * inv_x2
        + inv_x2 * inv_x * (1.0 / 6.0
        - inv_x2 * (1.0 / 30.0
        - inv_x2 * (1.0 / 42.0
        - inv_x2 * (1.0 / 30.0
        - inv_x2 * (5.0 / 66.0)))));
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Regularized incomplete beta function  I_x(a,b)
// ═══════════════════════════════════════════════════════════════════════════

/// Regularized incomplete beta function: I_x(a, b) = B(x; a,b) / B(a,b).
///
/// Uses the continued fraction representation (Lentz's algorithm) when
/// x > (a+1)/(a+b+2), otherwise uses the series expansion.
/// Satisfies I_x(a,b) + I_{1-x}(b,a) = 1.
///
/// Returns values in [0, 1]. Required for Student-t and F CDFs.
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    if a <= 0.0 || b <= 0.0 { return f64::NAN; }

    // Use symmetry to ensure convergence: I_x(a,b) = 1 - I_{1-x}(b,a)
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    // Front factor: x^a (1-x)^b / (a * B(a,b))
    let lbeta = log_beta(a, b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp() / a;

    // Lentz's continued fraction for I_x(a,b)
    // Uses the standard CF representation with coefficients d_m, e_m
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny { d = tiny; }
    d = 1.0 / d;
    let mut result = d;

    for m in 1..=max_iter {
        let m_f64 = m as f64;

        // Even step: d_{2m} = m(b-m)x / ((a+2m-1)(a+2m))
        let num = m_f64 * (b - m_f64) * x / ((a + 2.0 * m_f64 - 1.0) * (a + 2.0 * m_f64));
        d = 1.0 + num * d;
        if d.abs() < tiny { d = tiny; }
        c = 1.0 + num / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        result *= d * c;

        // Odd step: d_{2m+1} = -(a+m)(a+b+m)x / ((a+2m)(a+2m+1))
        let num = -(a + m_f64) * (a + b + m_f64) * x / ((a + 2.0 * m_f64) * (a + 2.0 * m_f64 + 1.0));
        d = 1.0 + num * d;
        if d.abs() < tiny { d = tiny; }
        c = 1.0 + num / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        let delta = d * c;
        result *= delta;

        if (delta - 1.0).abs() < eps { break; }
    }

    front * result
}

// ═══════════════════════════════════════════════════════════════════════════
// Regularized incomplete gamma functions  P(a,x) and Q(a,x)
// ═══════════════════════════════════════════════════════════════════════════

/// Lower regularized incomplete gamma: P(a, x) = γ(a,x) / Γ(a).
///
/// Uses series expansion for x < a+1, continued fraction otherwise.
pub fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 { return f64::NAN; }
    if x == 0.0 { return 0.0; }

    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// Upper regularized incomplete gamma: Q(a, x) = 1 - P(a, x) = Γ(a,x) / Γ(a).
pub fn regularized_gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 { return f64::NAN; }
    if x == 0.0 { return 1.0; }

    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_cf(a, x)
    }
}

/// Series expansion for P(a,x): γ(a,x)/Γ(a) = e⁻ˣ x^a Σ x^n / Γ(a+n+1)
fn gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;

    let mut term = 1.0 / a;
    let mut sum = term;
    for n in 1..=max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * eps { break; }
    }

    let log_prefix = a * x.ln() - x - log_gamma(a);
    sum * log_prefix.exp()
}

/// Continued fraction for Q(a,x) using Lentz's algorithm.
fn gamma_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny { d = tiny; }
        c = b + an / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps { break; }
    }

    let log_prefix = a * x.ln() - x - log_gamma(a);
    log_prefix.exp() * h
}

// ═══════════════════════════════════════════════════════════════════════════
// Distribution CDFs
// ═══════════════════════════════════════════════════════════════════════════

/// Standard normal CDF: Φ(x) = P(Z ≤ x) where Z ~ N(0,1).
///
/// Φ(x) = ½ erfc(-x/√2).
/// Special-cases x=0 → 0.5 exactly (the A&S polynomial has a 5e-10 bias at t=1).
pub fn normal_cdf(x: f64) -> f64 {
    if x == 0.0 { return 0.5; }
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Standard normal survival function: 1 - Φ(x).
///
/// Avoids catastrophic cancellation for large x.
/// Special-cases x=0 → 0.5 exactly.
pub fn normal_sf(x: f64) -> f64 {
    if x == 0.0 { return 0.5; }
    0.5 * erfc(x / std::f64::consts::SQRT_2)
}

/// Student-t CDF: P(T ≤ t) where T ~ t(ν).
///
/// Uses the incomplete beta function:
/// - P(T ≤ t) = 1 - ½ I_x(ν/2, 1/2) where x = ν/(ν + t²), for t ≥ 0
/// - Symmetry for t < 0
pub fn t_cdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 { return f64::NAN; }
    let x = df / (df + t * t);
    let ib = regularized_incomplete_beta(x, df / 2.0, 0.5);
    if t >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

/// F-distribution CDF: P(X ≤ x) where X ~ F(d₁, d₂).
///
/// Uses the incomplete beta function:
/// P(X ≤ x) = I_{d₁x/(d₁x+d₂)}(d₁/2, d₂/2)
pub fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if d1 <= 0.0 || d2 <= 0.0 { return f64::NAN; }
    let z = d1 * x / (d1 * x + d2);
    regularized_incomplete_beta(z, d1 / 2.0, d2 / 2.0)
}

/// Chi-square CDF: P(X ≤ x) where X ~ χ²(k).
///
/// Chi-square is Gamma(k/2, 2), so P(X ≤ x) = P(k/2, x/2).
pub fn chi2_cdf(x: f64, k: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if k <= 0.0 { return f64::NAN; }
    regularized_gamma_p(k / 2.0, x / 2.0)
}

/// Chi-square survival function: P(X > x) = 1 - CDF.
pub fn chi2_sf(x: f64, k: f64) -> f64 {
    if x <= 0.0 { return 1.0; }
    if k <= 0.0 { return f64::NAN; }
    regularized_gamma_q(k / 2.0, x / 2.0)
}

/// Standard normal two-tailed p-value: P(|Z| ≥ |z|) = 2 × SF(|z|).
pub fn normal_two_tail_p(z: f64) -> f64 {
    2.0 * normal_sf(z.abs())
}

/// Student-t two-tailed p-value: P(|T| ≥ |t|) where T ~ t(ν).
pub fn t_two_tail_p(t: f64, df: f64) -> f64 {
    2.0 * (1.0 - t_cdf(t.abs(), df))
}

/// F-distribution right-tail p-value: P(X ≥ x) where X ~ F(d₁, d₂).
pub fn f_right_tail_p(x: f64, d1: f64, d2: f64) -> f64 {
    1.0 - f_cdf(x, d1, d2)
}

/// Chi-square right-tail p-value: P(X ≥ x) where X ~ χ²(k).
pub fn chi2_right_tail_p(x: f64, k: f64) -> f64 {
    chi2_sf(x, k)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;
    const FINE_TOL: f64 = 1e-10;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() { return true; }
        (a - b).abs() < tol
    }

    // ── erf / erfc ───────────────────────────────────────────────────────

    #[test]
    fn erf_known_values() {
        assert!(approx(erf(0.0), 0.0, TOL));
        assert!(approx(erf(1.0), 0.8427007929, TOL));
        assert!(approx(erf(2.0), 0.9953222650, TOL));
        assert!(approx(erf(-1.0), -0.8427007929, TOL));
        // erf(∞) → 1
        assert!(approx(erf(6.0), 1.0, TOL));
    }

    #[test]
    fn erfc_known_values() {
        assert!(approx(erfc(0.0), 1.0, TOL));
        assert!(approx(erfc(1.0), 0.1572992070, TOL));
        assert!(approx(erfc(-1.0), 1.8427007929, TOL));
        // erfc(large) → 0
        assert!(erfc(6.0) < 1e-10);
    }

    // ── log_gamma ────────────────────────────────────────────────────────

    #[test]
    fn log_gamma_known_values() {
        // Γ(1) = 1 → ln(1) = 0
        assert!(approx(log_gamma(1.0), 0.0, FINE_TOL));
        // Γ(2) = 1 → ln(1) = 0
        assert!(approx(log_gamma(2.0), 0.0, FINE_TOL));
        // Γ(5) = 24 → ln(24) ≈ 3.178
        assert!(approx(log_gamma(5.0), 24.0_f64.ln(), FINE_TOL));
        // Γ(0.5) = √π → ln(√π) ≈ 0.5724
        assert!(approx(log_gamma(0.5), 0.5 * std::f64::consts::PI.ln(), 1e-8));
        // Γ(10) = 362880
        assert!(approx(log_gamma(10.0), 362880.0_f64.ln(), FINE_TOL));
    }

    // ── incomplete beta ──────────────────────────────────────────────────

    #[test]
    fn incomplete_beta_boundaries() {
        assert!(approx(regularized_incomplete_beta(0.0, 2.0, 3.0), 0.0, FINE_TOL));
        assert!(approx(regularized_incomplete_beta(1.0, 2.0, 3.0), 1.0, FINE_TOL));
    }

    #[test]
    fn incomplete_beta_known_values() {
        // I_0.5(1,1) = 0.5 (uniform)
        assert!(approx(regularized_incomplete_beta(0.5, 1.0, 1.0), 0.5, FINE_TOL));
        // I_0.5(2,2) = 0.5 (symmetric beta)
        assert!(approx(regularized_incomplete_beta(0.5, 2.0, 2.0), 0.5, FINE_TOL));
        // I_0.5(1,2) = 0.75
        assert!(approx(regularized_incomplete_beta(0.5, 1.0, 2.0), 0.75, 1e-8));
        // I_0.3(2,5) ≈ 0.57983 (scipy.special.betainc(2,5,0.3))
        assert!(approx(regularized_incomplete_beta(0.3, 2.0, 5.0), 0.57983, 1e-3));
    }

    #[test]
    fn incomplete_beta_symmetry() {
        // I_x(a,b) + I_{1-x}(b,a) = 1
        let x = 0.3;
        let a = 2.5;
        let b = 3.7;
        let sum = regularized_incomplete_beta(x, a, b) + regularized_incomplete_beta(1.0 - x, b, a);
        assert!(approx(sum, 1.0, 1e-10));
    }

    // ── incomplete gamma ─────────────────────────────────────────────────

    #[test]
    fn incomplete_gamma_boundaries() {
        assert!(approx(regularized_gamma_p(2.0, 0.0), 0.0, FINE_TOL));
        // P(a, ∞) → 1
        assert!(approx(regularized_gamma_p(2.0, 100.0), 1.0, FINE_TOL));
    }

    #[test]
    fn incomplete_gamma_known_values() {
        // P(1, x) = 1 - e^(-x) (exponential CDF)
        let x = 2.0_f64;
        let expected = 1.0 - (-x).exp();
        assert!(approx(regularized_gamma_p(1.0, x), expected, 1e-10));

        // P(1, 1) = 1 - 1/e ≈ 0.6321
        assert!(approx(regularized_gamma_p(1.0, 1.0), 0.6321205588, 1e-8));
    }

    #[test]
    fn gamma_pq_complement() {
        // P(a,x) + Q(a,x) = 1
        let a = 3.5;
        let x = 2.7;
        let sum = regularized_gamma_p(a, x) + regularized_gamma_q(a, x);
        assert!(approx(sum, 1.0, 1e-10));
    }

    // ── normal CDF ───────────────────────────────────────────────────────

    #[test]
    fn normal_cdf_known_values() {
        assert!(approx(normal_cdf(0.0), 0.5, TOL));
        assert!(approx(normal_cdf(1.96), 0.975002, TOL));
        assert!(approx(normal_cdf(-1.96), 0.024998, TOL));
        // 3-sigma
        assert!(approx(normal_cdf(3.0), 0.998650, TOL));
    }

    #[test]
    fn normal_cdf_symmetry() {
        // Φ(x) + Φ(-x) = 1
        let x = 1.5;
        assert!(approx(normal_cdf(x) + normal_cdf(-x), 1.0, TOL));
    }

    // ── Student-t CDF ────────────────────────────────────────────────────

    #[test]
    fn t_cdf_symmetry() {
        let df = 10.0;
        // P(T ≤ 0) = 0.5
        assert!(approx(t_cdf(0.0, df), 0.5, TOL));
        // Symmetry: P(T ≤ t) + P(T ≤ -t) = 1
        assert!(approx(t_cdf(2.0, df) + t_cdf(-2.0, df), 1.0, TOL));
    }

    #[test]
    fn t_cdf_df1_is_cauchy() {
        // t(1) = Cauchy: P(T ≤ 1) = 0.75
        assert!(approx(t_cdf(1.0, 1.0), 0.75, 1e-4));
    }

    #[test]
    fn t_cdf_large_df_approaches_normal() {
        // t(∞) → N(0,1): at t=1.96, P should be ≈ 0.975
        let p = t_cdf(1.96, 1000.0);
        assert!(approx(p, 0.975, 1e-3));
    }

    // ── F CDF ────────────────────────────────────────────────────────────

    #[test]
    fn f_cdf_zero() {
        assert!(approx(f_cdf(0.0, 5.0, 10.0), 0.0, FINE_TOL));
    }

    #[test]
    fn f_cdf_known_value() {
        // F(5,10): P(X ≤ 3.33) ≈ 0.95 (from F-table)
        let p = f_cdf(3.33, 5.0, 10.0);
        assert!((p - 0.95).abs() < 0.02);
    }

    // ── Chi-square CDF ──────────────────────────────────────────────────

    #[test]
    fn chi2_cdf_known_values() {
        // χ²(1): P(X ≤ 3.841) ≈ 0.95
        let p = chi2_cdf(3.841, 1.0);
        assert!(approx(p, 0.95, 1e-3));

        // χ²(2) = exponential with rate 0.5: P(X ≤ x) = 1 - e^(-x/2)
        let x = 4.0_f64;
        let expected = 1.0 - (-x / 2.0).exp();
        assert!(approx(chi2_cdf(x, 2.0), expected, 1e-6));
    }

    #[test]
    fn chi2_sf_complement() {
        let x = 5.0;
        let k = 3.0;
        let sum = chi2_cdf(x, k) + chi2_sf(x, k);
        assert!(approx(sum, 1.0, 1e-10));
    }

    // ── p-value helpers ──────────────────────────────────────────────────

    #[test]
    fn two_tail_p_values() {
        // z=1.96 → p ≈ 0.05
        let p = normal_two_tail_p(1.96);
        assert!(approx(p, 0.05, 1e-3));

        // t=2.228 with df=10 → p ≈ 0.05
        let p = t_two_tail_p(2.228, 10.0);
        assert!(approx(p, 0.05, 1e-2));
    }

    // ── digamma / trigamma ──────────────────────────────────────────────

    #[test]
    fn digamma_known_values() {
        // ψ(1) = -γ (Euler-Mascheroni constant)
        let euler_mascheroni = 0.5772156649015329;
        assert!(approx(digamma(1.0), -euler_mascheroni, FINE_TOL));

        // ψ(1/2) = -γ - 2·ln(2)
        let expected = -euler_mascheroni - 2.0 * 2.0_f64.ln();
        assert!(approx(digamma(0.5), expected, FINE_TOL));

        // ψ(2) = 1 - γ
        assert!(approx(digamma(2.0), 1.0 - euler_mascheroni, FINE_TOL));

        // ψ(3) = 1 + 1/2 - γ = 3/2 - γ
        assert!(approx(digamma(3.0), 1.5 - euler_mascheroni, FINE_TOL));

        // ψ(10) = H_9 - γ where H_9 = 1 + 1/2 + ... + 1/9
        let h9: f64 = (1..=9).map(|k| 1.0 / k as f64).sum();
        assert!(approx(digamma(10.0), h9 - euler_mascheroni, FINE_TOL));
    }

    #[test]
    fn digamma_recurrence() {
        // ψ(x+1) = ψ(x) + 1/x for several x values
        for &x in &[0.5, 1.0, 2.5, 5.0, 10.0, 100.0] {
            let lhs = digamma(x + 1.0);
            let rhs = digamma(x) + 1.0 / x;
            assert!(approx(lhs, rhs, FINE_TOL),
                "recurrence failed at x={}: {} vs {}", x, lhs, rhs);
        }
    }

    #[test]
    fn trigamma_known_values() {
        // ψ₁(1) = π²/6
        assert!(approx(trigamma(1.0), std::f64::consts::PI.powi(2) / 6.0, FINE_TOL));

        // ψ₁(2) = π²/6 - 1
        assert!(approx(trigamma(2.0), std::f64::consts::PI.powi(2) / 6.0 - 1.0, FINE_TOL));

        // ψ₁(1/2) = π²/2
        assert!(approx(trigamma(0.5), std::f64::consts::PI.powi(2) / 2.0, FINE_TOL));
    }

    #[test]
    fn trigamma_recurrence() {
        // ψ₁(x+1) = ψ₁(x) - 1/x² for several x values
        for &x in &[0.5, 1.0, 2.5, 5.0, 10.0, 100.0] {
            let lhs = trigamma(x + 1.0);
            let rhs = trigamma(x) - 1.0 / (x * x);
            assert!(approx(lhs, rhs, FINE_TOL),
                "recurrence failed at x={}: {} vs {}", x, lhs, rhs);
        }
    }

    #[test]
    fn digamma_trigamma_derivative_consistency() {
        // Numerical derivative of digamma should approximate trigamma
        let h = 1e-6;
        for &x in &[1.0, 2.0, 5.0, 10.0] {
            let numerical = (digamma(x + h) - digamma(x - h)) / (2.0 * h);
            let analytical = trigamma(x);
            assert!((numerical - analytical).abs() < 1e-5,
                "derivative mismatch at x={}: numerical={}, analytical={}", x, numerical, analytical);
        }
    }
}
