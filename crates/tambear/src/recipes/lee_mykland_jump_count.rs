//! Lee–Mykland jump count — count of returns that exceed a bipower-derived threshold.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over Kulisch-
//! backed bipower accumulation + threshold + masked count. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! The Lee–Mykland (2008) jump test uses the **bipower variation** as a
//! jump-robust local volatility estimator:
//!
//! ```text
//! BV         =  (π/2) · (1 / (n − 1)) · Σ |r[i]| · |r[i−1]|
//! σ̂_local   =  sqrt(BV)
//! threshold  =  C · σ̂_local
//! n_jumps    =  count(|r[i]| > threshold)
//! ```
//!
//! Bipower variation is consistent for integrated volatility under
//! continuous price processes but is robust to jumps because each term
//! involves *two adjacent* return magnitudes — a single jump
//! contributes to only one product, not the sum-of-squares.
//!
//! The two-pass structure: pass 1 computes BV (and therefore the
//! threshold); pass 2 counts returns above threshold. This is required
//! because the threshold depends on data-determined BV.
//!
//! # Reference
//!
//! Lee, S. S., & Mykland, P. A. (2008). Jumps in financial markets: A
//! new nonparametric test and jump dynamics. *Review of Financial
//! Studies* 21(6): 2535–2563.
//!
//! # Composition
//!
//! - **Pass 1 — bipower accumulation** — Kulisch-exact sum over
//!   `|r[i]| · |r[i−1]|` for i=1..n. Lowers to
//!   `accumulate(rets, Grouping::All, Op::Add, expr=|r|·|gather(r,-1)|)`
//!   once the lag-gather pattern is wired into the atom layer; today
//!   computed via a local Kulisch accumulator.
//! - **Threshold derivation** — pure scalar: `C · sqrt((π/2) · BV / (n-1))`.
//! - **Pass 2 — masked count** — count returns whose absolute value
//!   exceeds the threshold. Lowers to
//!   `accumulate(rets, Grouping::All, Op::Add, expr=mask(|r|>thr))`.
//!
//! # NaN/Inf policy
//!
//! Non-finite returns are skipped at the bipower step (Kulisch's
//! `is_finite` gate via `math::sum`-style accumulation). Skipped
//! returns also do not count toward n_jumps.
//!
//! # Default parameters
//!
//! - `c` — significance threshold multiplier. Lee–Mykland use ~4.0
//!   (corresponds to ~99.99% confidence in standard normal). SIP can
//!   override per call site.

/// Lee–Mykland jump count over a return series with a configurable
/// significance multiplier `c`.
///
/// Returns 0 if `rets.len() < 2` (no bipower term computable).
///
/// # Panics
///
/// Panics if `c <= 0.0` or `c` is non-finite.
pub fn lee_mykland_jump_count(rets: &[f64], c: f64) -> u32 {
    assert!(
        c.is_finite() && c > 0.0,
        "lee_mykland_jump_count: c must be finite and > 0, got {c}"
    );

    if rets.len() < 2 {
        return 0;
    }

    // Pass 1 — bipower accumulation over adjacent |r[i]| · |r[i-1]| terms.
    // Skip any pair where either return is non-finite.
    use crate::primitives::specialist::kulisch_accumulator::KulischAccumulator;
    let mut bv_acc = KulischAccumulator::new();
    let mut n_terms: usize = 0;
    for i in 1..rets.len() {
        let (a, b) = (rets[i - 1], rets[i]);
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        bv_acc.add_f64(a.abs() * b.abs());
        n_terms += 1;
    }

    if n_terms < 1 {
        return 0;
    }

    // Bipower-variation estimate of local variance, then √ for σ̂_local.
    let bv = std::f64::consts::FRAC_PI_2 * bv_acc.to_f64() / n_terms as f64;
    if !bv.is_finite() || bv < 0.0 {
        return 0;
    }
    let sigma_local = bv.sqrt();
    let threshold = c * sigma_local;

    // Pass 2 — masked count.
    let mut n_jumps: u32 = 0;
    for &r in rets {
        if r.is_finite() && r.abs() > threshold {
            n_jumps += 1;
        }
    }
    n_jumps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_single_no_jumps() {
        assert_eq!(lee_mykland_jump_count(&[], 4.0), 0);
        assert_eq!(lee_mykland_jump_count(&[0.01], 4.0), 0);
    }

    #[test]
    fn flat_returns_no_jumps() {
        // All zero returns → BV = 0 → threshold = 0. Any non-zero
        // would jump, but all are zero.
        let rets = vec![0.0; 100];
        assert_eq!(lee_mykland_jump_count(&rets, 4.0), 0);
    }

    #[test]
    fn small_uniform_returns_no_jumps() {
        // All returns the same magnitude → bipower estimate is the
        // same magnitude, threshold = c · |r|. With c=4, no return
        // exceeds 4·|r|.
        let rets = vec![0.001; 100];
        assert_eq!(lee_mykland_jump_count(&rets, 4.0), 0);
    }

    #[test]
    fn obvious_jump_detected() {
        // 99 small returns + 1 huge one. Bipower of small returns
        // sets a tight threshold; the huge one is well above.
        let mut rets = vec![0.001; 99];
        rets.push(1.0); // 1000x larger than baseline
        let n = lee_mykland_jump_count(&rets, 4.0);
        assert!(n >= 1, "expected at least 1 jump, got {n}");
    }

    #[test]
    fn tighter_c_finds_more_jumps() {
        // For a series with mild outliers, a smaller c flags more.
        let mut rets = vec![0.001; 90];
        for _ in 0..10 {
            rets.push(0.01); // 10x baseline — borderline for some c
        }
        let strict = lee_mykland_jump_count(&rets, 4.0);
        let loose = lee_mykland_jump_count(&rets, 1.5);
        assert!(loose >= strict, "looser c should find ≥ jumps: {loose} vs {strict}");
    }

    #[test]
    fn skips_non_finite_returns_in_bipower() {
        // Insert NaN/Inf among normal returns; result should match
        // computing on the finite-only subset.
        let rets_with_nan = vec![0.001, 0.002, f64::NAN, 0.001, f64::INFINITY, 0.5];
        let rets_clean = vec![0.001, 0.002, 0.001, 0.5];
        let n_with = lee_mykland_jump_count(&rets_with_nan, 4.0);
        let n_clean = lee_mykland_jump_count(&rets_clean, 4.0);
        // Both should return the same answer (the 0.5 outlier is the only
        // candidate for a jump, and the threshold computation skips
        // non-finite terms in both cases).
        assert_eq!(n_with, n_clean);
    }

    #[test]
    #[should_panic(expected = "c must be finite")]
    fn panics_on_zero_c() {
        let _ = lee_mykland_jump_count(&[0.01, 0.02], 0.0);
    }

    #[test]
    #[should_panic(expected = "c must be finite")]
    fn panics_on_nan_c() {
        let _ = lee_mykland_jump_count(&[0.01, 0.02], f64::NAN);
    }
}
