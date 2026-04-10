//! Adversarial Wave 23 — NaN-eating conditional-skip in Rényi entropy/divergence
//!
//! The observer found `renyi_entropy`'s alpha=∞ branch was fixed (nan_max applied).
//! This wave documents three REMAINING bugs in the alpha=0 and alpha=∞ branches
//! that were missed by the systemic fold-grep pass — they use conditional-skip
//! (`filter(|p| p > 0.0)`) rather than fold, so the grep didn't find them.
//!
//! Four confirmed bugs:
//!
//! 1. `renyi_entropy` alpha=0 (information_theory.rs:102):
//!    `probs.iter().filter(|&&p| p > 0.0).count()` — NaN probability silently
//!    excluded from support count. Returns ln(n_finite_nonzero) instead of NaN.
//!    Mathematical truth: if any probability is undefined (NaN), the entropy
//!    is undefined → result must be NaN.
//!
//! 2. `renyi_divergence` alpha=0 (information_theory.rs:1052):
//!    `filter(|(&pi, _)| pi > 0.0)` — NaN pi excluded from overlap computation.
//!    Returns finite -ln(overlap_of_valid_entries) instead of NaN.
//!
//! 3. `renyi_divergence` alpha=∞, first bug (information_theory.rs:1060):
//!    `filter(|(&pi, _)| pi > 0.0)` — NaN pi excluded from max-ratio computation.
//!    NaN pair silently not considered as a potential maximum.
//!
//! 4. `renyi_divergence` alpha=∞, second bug (information_theory.rs:1062):
//!    `fold(0.0f64, f64::max)` — wrong identity (should be NEG_INFINITY for max)
//!    AND NaN-eating (`f64::max(NaN, x) = x`). Compound failure.
//!
//! Root cause: all four bugs share the same underlying decision — `p > 0.0`
//! with NaN p evaluates to false, silently treating NaN as "zero probability"
//! instead of "undefined probability." This is the conditional-skip class of
//! NaN-eating identified in wave 16 (`hurst_rs` std > 0.0 pattern).
//!
//! The alpha=∞ fold bug (bug 4) is additionally a wrong-identity bug:
//! `fold(0.0, f64::max)` masks negative ratios (if all ratios < 0, returns 0).
//! But log probabilities are ≤ 0 and ratios pi/qi can be < 1 for diffuse
//! distributions. However in D_∞ context, pi/qi > 0 by definition (filtering
//! to pi > 0), so 0.0 as identity is wrong (should be NEG_INFINITY) but the
//! filter ensures all ratios are positive — making the wrong identity harmless
//! for non-NaN inputs. The NaN-eating is the live bug.
//!
//! All tests assert mathematical truths. Failures are bugs.

use tambear::{renyi_entropy, renyi_divergence};

// ═══════════════════════════════════════════════════════════════════════════
// Correctness baselines — verify clean-data behavior before testing NaN
// ═══════════════════════════════════════════════════════════════════════════

/// H_0([0.3, 0.5, 0.2]) = ln(3) (all three have positive support).
#[test]
fn renyi_entropy_alpha0_correct_support_count() {
    let result = renyi_entropy(&[0.3, 0.5, 0.2], 0.0);
    let expected = 3.0_f64.ln();
    assert!((result - expected).abs() < 1e-12,
        "H_0 with 3-element support should be ln(3) = {}, got {}", expected, result);
}

/// H_0([0.5, 0.0, 0.5]) = ln(2) (zero-probability element excluded from support).
/// Note: excluding p=0.0 is CORRECT (it's not in the support). This is different
/// from excluding NaN — p=0.0 means "impossible", NaN means "undefined."
#[test]
fn renyi_entropy_alpha0_zero_probability_correctly_excluded() {
    let result = renyi_entropy(&[0.5, 0.0, 0.5], 0.0);
    let expected = 2.0_f64.ln();
    assert!((result - expected).abs() < 1e-12,
        "H_0 with zero-prob element should give ln(2) (zero excluded from support), got {}", result);
}

/// H_∞([0.3, 0.5, 0.2]) = -ln(0.5) (min-entropy = -log of largest probability).
#[test]
fn renyi_entropy_alpha_inf_correct_min_entropy() {
    let result = renyi_entropy(&[0.3, 0.5, 0.2], f64::INFINITY);
    let expected = -(0.5_f64.ln());
    assert!((result - expected).abs() < 1e-12,
        "H_∞ min-entropy should be -ln(0.5) = {}, got {}", expected, result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Bug 1: renyi_entropy alpha=0 silently excludes NaN probabilities
// ═══════════════════════════════════════════════════════════════════════════

/// NaN probability in a 3-element distribution: entropy undefined.
///
/// `filter(|&&p| p > 0.0)` with NaN p: NaN > 0.0 = false → NaN silently excluded.
/// Support counted as 2 (entries 0.3 and 0.2 only).
/// Returns ln(2) ≈ 0.693 instead of NaN.
///
/// Mathematical truth: if any probability is NaN (undefined), the entire
/// probability distribution is undefined, so H_0 is undefined → NaN.
#[test]
fn renyi_entropy_alpha0_nan_probability_must_propagate() {
    let result = renyi_entropy(&[0.3, f64::NAN, 0.2], 0.0);
    assert!(result.is_nan(),
        "BUG: renyi_entropy with NaN probability and alpha=0 should return NaN, got {} — \
         `filter(|&&p| p > 0.0)` at information_theory.rs:102 treats NaN as p=0 \
         (NaN > 0.0 = false), counting only 2 non-NaN entries and returning ln(2). \
         NaN probability means undefined distribution → entropy is undefined.",
        result);
}

/// NaN as the only element: support = 0. Returns ln(0) = -∞.
/// But the correct answer is NaN (undefined), not -∞.
#[test]
fn renyi_entropy_alpha0_single_nan_probability_must_return_nan() {
    let result = renyi_entropy(&[f64::NAN], 0.0);
    assert!(result.is_nan(),
        "BUG: renyi_entropy([NaN], alpha=0) should return NaN (undefined distribution), \
         got {} — NaN filtered to empty support → ln(0) = -∞, not NaN",
        result);
}

/// NaN at the first position (makes NaN-eating easy to test directionally).
#[test]
fn renyi_entropy_alpha0_nan_first_must_propagate() {
    let result = renyi_entropy(&[f64::NAN, 0.5, 0.5], 0.0);
    assert!(result.is_nan(),
        "BUG: renyi_entropy with NaN at index 0 and alpha=0 should return NaN, got {}",
        result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Bug 2: renyi_divergence alpha=0 silently excludes NaN p values
// ═══════════════════════════════════════════════════════════════════════════

/// D_0(p || q) = -ln(Σ qᵢ · 1[pᵢ > 0]) — overlap of q-mass on p-support.
/// Correctness baseline: D_0([0.5, 0.5] || [0.4, 0.6]) = -ln(0.4 + 0.6) = 0.0.
#[test]
fn renyi_divergence_alpha0_correct_full_overlap() {
    let result = renyi_divergence(&[0.5, 0.5], &[0.4, 0.6], 0.0);
    // Both p_i > 0, overlap = q_0 + q_1 = 1.0, D_0 = -ln(1.0) = 0.0
    assert!(result.abs() < 1e-12,
        "D_0 with full q-support on p-support should be 0, got {}", result);
}

/// NaN in p: the support of p is undefined → overlap is undefined → D_0 = NaN.
///
/// ACTUAL (BUG): `filter(|(&pi, _)| pi > 0.0)` at information_theory.rs:1052
/// excludes NaN pi (NaN > 0.0 = false). Overlap computed from remaining
/// valid entries. Returns finite value instead of NaN.
#[test]
fn renyi_divergence_alpha0_nan_p_must_propagate() {
    let p = vec![0.5, f64::NAN, 0.5];
    let q = vec![0.3, 0.4, 0.3];
    let result = renyi_divergence(&p, &q, 0.0);
    assert!(result.is_nan(),
        "BUG: renyi_divergence with NaN in p and alpha=0 should return NaN, got {} — \
         `filter(|(&pi, _)| pi > 0.0)` treats NaN as p=0 (excluded from support). \
         NaN probability means undefined distribution → D_0 is undefined.",
        result);
}

/// NaN in q: q-mass at the NaN position is undefined → overlap undefined → D_0 = NaN.
#[test]
fn renyi_divergence_alpha0_nan_q_must_propagate() {
    let p = vec![0.5, 0.0, 0.5];
    let q = vec![0.3, f64::NAN, 0.3];
    let result = renyi_divergence(&p, &q, 0.0);
    assert!(result.is_nan(),
        "BUG: renyi_divergence with NaN in q and alpha=0 should return NaN, got {}",
        result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Bug 3 + 4: renyi_divergence alpha=∞ — compound NaN failure
// ═══════════════════════════════════════════════════════════════════════════

/// D_∞(p || q) = ln(max pᵢ/qᵢ) for pᵢ > 0.
/// Correctness baseline.
#[test]
fn renyi_divergence_alpha_inf_correct_max_ratio() {
    // p = [0.6, 0.4], q = [0.3, 0.7]
    // Ratios for pi > 0: 0.6/0.3 = 2.0, 0.4/0.7 ≈ 0.571
    // max ratio = 2.0 → D_∞ = ln(2.0) ≈ 0.693
    let result = renyi_divergence(&[0.6, 0.4], &[0.3, 0.7], f64::INFINITY);
    let expected = 2.0_f64.ln();
    assert!((result - expected).abs() < 1e-12,
        "D_∞ should be ln(2) = {}, got {}", expected, result);
}

/// NaN in p at alpha=∞: two bugs fire simultaneously.
///
/// Bug 3: `filter(|(&pi, _)| pi > 0.0)` excludes NaN pi.
///   NaN pair silently not considered as potential maximum.
///
/// Bug 4: `fold(0.0f64, f64::max)` — NaN-eating fold with wrong identity.
///   Even if NaN reached the fold, f64::max would eat it.
///
/// Result: finite max-ratio computed from non-NaN entries.
///
/// Mathematical truth: NaN in p → distribution undefined → D_∞ = NaN.
#[test]
fn renyi_divergence_alpha_inf_nan_p_must_propagate() {
    let p = vec![0.6, f64::NAN, 0.4];
    let q = vec![0.3, 0.4, 0.3];
    let result = renyi_divergence(&p, &q, f64::INFINITY);
    assert!(result.is_nan(),
        "BUG: renyi_divergence with NaN in p and alpha=∞ should return NaN, got {} — \
         Two bugs fire: (1) `filter(|(&pi, _)| pi > 0.0)` excludes NaN pi \
         (NaN > 0.0 = false), silently treating it as p=0; \
         (2) `fold(0.0f64, f64::max)` is a NaN-eating fold even if NaN somehow \
         reached it. NaN probability means undefined distribution → D_∞ undefined.",
        result);
}

/// NaN in q at alpha=∞: q_i appears in ratio p_i/q_i.
/// If q_i is NaN, the ratio for that pair is NaN.
/// But the filter `pi > 0.0` still passes (pi is valid), so NaN enters the fold.
/// `fold(0.0, f64::max)`: f64::max(NaN, x) = x → NaN eaten by fold.
/// Returns finite max-ratio computed without the NaN pair.
///
/// Mathematical truth: NaN in q → ratio undefined → D_∞ undefined → NaN.
#[test]
fn renyi_divergence_alpha_inf_nan_q_must_propagate() {
    let p = vec![0.6, 0.4];
    let q = vec![f64::NAN, 0.7];
    let result = renyi_divergence(&p, &q, f64::INFINITY);
    assert!(result.is_nan(),
        "BUG: renyi_divergence with NaN in q and alpha=∞ should return NaN, got {} — \
         The ratio pi/qi = 0.6/NaN = NaN enters the fold as pi > 0.0 passes, \
         but `fold(0.0f64, f64::max)` eats NaN: f64::max(0.0, NaN) = 0.0. \
         The NaN ratio is silently discarded, returning finite max from valid pairs.",
        result);
}

/// Both p and q NaN: complete failure.
#[test]
fn renyi_divergence_alpha_inf_both_nan_must_propagate() {
    let p = vec![f64::NAN, 0.5];
    let q = vec![f64::NAN, 0.5];
    let result = renyi_divergence(&p, &q, f64::INFINITY);
    assert!(result.is_nan(),
        "BUG: renyi_divergence with NaN in both p and q at alpha=∞ should return NaN, got {}",
        result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Shannon entropy — verify NaN propagation is already correct
// ═══════════════════════════════════════════════════════════════════════════

/// shannon_entropy with NaN probability should return NaN.
/// (This verifies the alpha→1 limit path works correctly.)
use tambear::shannon_entropy;

#[test]
fn shannon_entropy_nan_probability_returns_nan() {
    let result = shannon_entropy(&[0.3, f64::NAN, 0.2]);
    // shannon_entropy uses: if p > 0.0 { h -= p * p.ln(); }
    // NaN > 0.0 = false → NaN silently skipped
    // This may ALSO be a bug — document the current behavior.
    // If it returns NaN: the shannon_entropy path propagates correctly.
    // If it returns finite: same bug class as renyi_entropy alpha=0.
    if result.is_nan() {
        // Correct: NaN propagated
    } else {
        // Document: same conditional-skip bug in shannon_entropy
        assert!(result.is_nan(),
            "BUG: shannon_entropy with NaN probability returns {} instead of NaN — \
             `if p > 0.0` conditional skips NaN, treating it as p=0 (excluded from sum). \
             Consistent with renyi_entropy alpha=0 bug class.",
            result);
    }
}
