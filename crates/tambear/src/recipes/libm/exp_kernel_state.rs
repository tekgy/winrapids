//! `ExpKernelState` — the shared intermediate for the exp/log family.
//!
//! # Why this exists
//!
//! Per `docs/architecture/tambear-libm-factoring.md`, every member of
//! the exp/log family (`exp`, `log`, `exp2`, `log2`, `exp10`, `log10`,
//! `sinh`, `cosh`, `tanh`, `pow`) decomposes into:
//!
//! 1. A **range reduction** `x = k·ln(2) + r` with `|r| ≤ ln(2)/2`.
//! 2. A **precision-safe core** evaluation `expm1(r)`.
//! 3. A **per-recipe inverse transform** that reconstructs the named
//!    function from `(k, r, expm1_r)`.
//!
//! Steps 1 and 2 are *shared*: identical for every consumer that wants
//! exp-family at the same `x`. Step 3 is per-recipe. MSVC libm fails to
//! factor steps 1 and 2 out, so each function independently accumulates
//! Tang's k-multiplier rounding error (~280 ULP at large positive x).
//! Tambear's promise: factor 1 and 2 explicitly, register the result as
//! a shareable intermediate, and let every consumer pull from cache.
//!
//! # Design — concrete f64 first, generic later
//!
//! Aristotle's deconstruction (campsites/sweep-35/aristotle/) surfaced
//! 25 truths and 10 recommendations for the kernel-state pattern. The
//! full vision: a `KernelState<P: Precision>` trait + a
//! `ComplementaryArgumentTransform<F, G>` trait, generic over precision
//! contexts (P0F64, P1Extended, P2BigFloat) and complementary-argument
//! shapes (input-translation, input-scaling, output-translation). Plus
//! a `BidirectionalExpKernelState` for sinh/cosh/tanh's
//! correctness-invariant pairing, a `PowKernelState` for pow's
//! cross-family composition, and a door tag for per-vendor JIT outputs
//! (DEC-019).
//!
//! Sweep 35 ships the **concrete f64-only first instance** of that
//! pattern. The cache-key fields (`x_bits`, `precision_tag`, `door_tag`,
//! `branch_policy_tag`) are structurally present from day one — per
//! aristotle's T23 ("adding tags later means bumping IR_VERSION; adding
//! them now is free"). For Sweep 35 they're all fixed:
//!
//! - `precision_tag = 0` (P0F64 — f64 working precision)
//! - `door_tag = 0` (CPU — the only door wired in Sweep 35)
//! - `branch_policy_tag = 0` (RealAxis — the only policy on the real
//!   line; complex_log lands non-zero values in Phase D)
//!
//! The struct itself uses double-double `r` (preserving the ~106-bit
//! Cody-Waite reduction precision) and f64 `expm1_r` (the precision-safe
//! polynomial evaluation already shipped in `expm1.rs`). At higher
//! precision tiers, the struct gains BigFloat fields; the cache-key
//! `precision_tag` discriminates which struct shape applies.
//!
//! # Sharing contract (per Tambear Contract item 3)
//!
//! Two consumers can share a cached `ExpKernelState` iff:
//! 1. `x_bits` match (same input).
//! 2. `precision_tag` matches (same working precision).
//! 3. `door_tag` matches (same execution surface — bit-exactness is
//!    door-dependent under DEC-019 per-door JIT).
//! 4. `branch_policy_tag` matches (same branch-cut discipline).
//!
//! These four fields together form the content-addressed cache key per
//! `holonomic-architecture.md`. Same fields → same key → cache hit.
//! Any mismatch → fresh computation.
//!
//! # F13.C antibodies surfaced by aristotle
//!
//! - **Input canonicality** (A1, T19): for `f64`, `x_bits` is well-defined.
//!   For future BigFloat tiers, the cache-key construction must canonicalize
//!   before hashing — and that's the precondition that needs an antibody.
//!   In the f64 tier shipped here, `x_bits = x.to_bits()` is unambiguous;
//!   the antibody is structurally trivial.
//! - **NaN/non-canonical zero**: `to_bits()` distinguishes `+0.0` and
//!   `-0.0` (different bit patterns; signed-zero matters). NaN bit
//!   patterns are non-unique in IEEE 754; per Kahan, we canonicalize NaN
//!   to a single representative before hashing.
//! - **Result range** (T16): the constructor establishes `expm1_r > -1`
//!   (algebraically guaranteed for `|r| ≤ ln(2)/2`, since
//!   `expm1(-ln(2)/2) ≈ -0.293 > -1`). This invariant is documented;
//!   future hardening would lift it into a `NormalizedExpm1` newtype.
//!
//! # What's NOT shipped here (deliberately, anti-YAGNI)
//!
//! - Generic `KernelState<P>` trait — there's only one P; no trait needed yet.
//! - `BidirectionalExpKernelState` — sinh/cosh/tanh aren't wired in Phase B;
//!   their Phase C recipes will introduce it as a *second instance* of the
//!   pattern, validating the shape.
//! - `PowKernelState` — pow is composed in Phase C; promoting to its own
//!   kernel state if/when the composed precision proves insufficient.
//! - PrecisionContext type ladder — Sweep 36+ work.
//!
//! # References
//!
//! - `R:\winrapids\docs\architecture\tambear-libm-factoring.md` § "Where this lives in the holonomic taxonomy"
//! - `R:\winrapids\docs\architecture\holonomic-architecture.md` § "Cache-discipline placement"
//! - `R:\winrapids\campsites\sweep-35\aristotle\exp-kernel-state-deconstruction.md` (Phases 1-5)
//! - `R:\winrapids\campsites\sweep-35\aristotle\exp-kernel-state-deconstruction-phase6-8.md` (Phases 6-8)

use std::sync::Arc;

use crate::intermediates::{IntermediateTag, TamSession};
use crate::primitives::constants::LOG2_E_F64;
use crate::primitives::hardware::ffloor;

// Reuse the same Cody-Waite ln(2) split as `exp.rs` and `expm1.rs`.
// `LN_2_CW_HI` has 19 trailing zero mantissa bits ⇒ `k · LN_2_CW_HI`
// is exact for any |k| ≤ 1024 (covers the full finite exp range).
const LN_2_CW_HI: f64 = 6.931_471_803_691_238_2e-1_f64;
const LN_2_CW_LO: f64 = 1.908_214_929_270_587_7e-10_f64;

/// Precision-tag value reserved for the f64 working-precision tier
/// (P0F64). Sweep 35 only ships this tier; higher values are reserved
/// for future BigFloat-backed tiers.
pub const PRECISION_TAG_P0F64: u8 = 0;

/// Door-tag value reserved for the CPU execution surface. Sweep 35
/// only ships CPU; higher values are reserved for per-vendor JIT
/// outputs per DEC-019.
pub const DOOR_TAG_CPU: u8 = 0;

/// Branch-policy tag reserved for `RealAxis` (the trivial policy on
/// real-valued inputs). DEC-032's complex `BranchPolicy` enum will
/// claim non-zero values in Phase D's `complex_log` recipe.
pub const BRANCH_POLICY_TAG_REAL_AXIS: u8 = 0;

/// Shared kernel state for the exp/log family at f64 working precision.
///
/// Contains the result of the Cody-Waite reduction
/// `x = k · ln(2) + r_hi + r_lo` plus the precision-safe core
/// `expm1(r_hi + r_lo)`.
///
/// # Invariants
///
/// - `|r_hi| ≤ ln(2)/2 ≈ 0.347`.
/// - `r_lo` is the residual of the Cody-Waite subtraction; `|r_lo| ≪ |r_hi|`
///   when `r_hi ≠ 0`.
/// - `expm1_r > -1` (algebraically — `expm1(-ln(2)/2) ≈ -0.293`).
///
/// # Recipe interaction
///
/// Each recipe wrapper pulls the four fields and applies its inverse
/// transform:
/// - `exp(x) = (1 + expm1_r) · 2^k` → bit-shift via `ldexp(1 + expm1_r, k)`
/// - `expm1(x) = 2^k · (1 + expm1_r) − 1` → reconstruct per regime (see `expm1.rs`)
/// - `sinh(x) = (exp(x) − exp(−x)) / 2` → pull `ExpKernelState(x)` and `ExpKernelState(−x)`
/// - etc.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpKernelState {
    /// Integer quotient `k = round(x / ln(2))`.
    pub k: i32,
    /// High word of the reduced argument: `r_hi = x − k · LN_2_CW_HI` (exact).
    pub r_hi: f64,
    /// Low word of the reduced argument: `r_lo = −k · LN_2_CW_LO` folded
    /// in. Together `r_hi + r_lo ≈ x − k · ln(2)` to ~106 bits.
    pub r_lo: f64,
    /// Precision-safe core: `expm1(r_hi + r_lo)` via the fdlibm
    /// rational form (see `expm1.rs::expm1_small_strict`).
    pub expm1_r: f64,
}

impl ExpKernelState {
    /// Compute the kernel state for input `x` at f64 working precision.
    ///
    /// Does NOT register in any session. Use `compute_or_get` to share
    /// via TamSession.
    ///
    /// # Precondition
    ///
    /// `x` must be a *finite* f64 in the range where the reduction is
    /// well-defined (`|x| < ~709` for safe exp output, but the kernel
    /// state itself is defined for any finite x). Callers handle special
    /// cases (NaN, ±∞, overflow, underflow) before constructing the state.
    ///
    /// # Why the state is finite-only
    ///
    /// Constructing a kernel state for `x = +∞` would compute
    /// `k = round(∞ · log2(e)) = ∞` (saturating to `i32::MAX`), then
    /// `r_hi = ∞ − ∞ = NaN`. The cache would then poison every
    /// downstream consumer with NaN. Callers must filter specials first.
    pub fn compute(x: f64) -> Self {
        debug_assert!(
            x.is_finite(),
            "ExpKernelState::compute: x must be finite (got {x}); caller handles specials"
        );
        let k_f = ffloor(x * LOG2_E_F64 + 0.5);
        let k = k_f as i32;
        // Cody-Waite reduction: r = (x - k·LN_2_CW_HI) - k·LN_2_CW_LO.
        // The first subtraction is exact for |k| < 2^21.
        let r_hi = x - k_f * LN_2_CW_HI;
        // Track the low part as a separate field rather than collapsing
        // — preserves Cody-Waite's ~106-bit precision for downstream
        // consumers that want it (e.g., sinh's small-x polynomial).
        let r_lo = -(k_f * LN_2_CW_LO);
        let r = r_hi + r_lo;
        let expm1_r = super::expm1::expm1_small_strict_public(r);
        Self { k, r_hi, r_lo, expm1_r }
    }

    /// The collapsed reduced argument `r = r_hi + r_lo`. Use this when
    /// the consumer doesn't need the ~106-bit precision and just wants
    /// the f64 reduced argument.
    #[inline]
    pub fn r(&self) -> f64 {
        self.r_hi + self.r_lo
    }

    /// Build the cache key for this kernel state at the f64/CPU/RealAxis tier.
    ///
    /// Sweep 35 ships only one (precision_tag, door_tag, branch_policy_tag)
    /// triple; future sweeps will multiplex other values through these
    /// same bytes. The fields are part of the cache key from day one to
    /// avoid an IR_VERSION bump when other doors/precisions/policies land.
    #[inline]
    pub fn cache_key_for(x: f64) -> IntermediateTag {
        // Canonicalize NaN to a single bit pattern. f64 has many NaN bit
        // patterns; using `to_bits()` directly on a NaN would create
        // distinct cache keys for the same logical value. Defense in
        // depth — callers should already filter NaN, but if any slip
        // through, they all hash to the same key.
        let x_bits = if x.is_nan() {
            f64::NAN.to_bits()
        } else {
            x.to_bits()
        };
        IntermediateTag::ExpKernelState {
            x_bits,
            precision_tag: PRECISION_TAG_P0F64,
            door_tag: DOOR_TAG_CPU,
            branch_policy_tag: BRANCH_POLICY_TAG_REAL_AXIS,
        }
    }

    /// Compute the kernel state, going through TamSession's
    /// content-addressed cache.
    ///
    /// First call for a given `x` computes and registers. Subsequent
    /// calls (same `x`, same precision/door/policy tags) return the
    /// cached `Arc<ExpKernelState>` without recomputing.
    ///
    /// # Precondition
    ///
    /// Same as `compute`: `x` must be finite. Special-case filtering
    /// belongs in the caller (the recipe wrapper), not in the state.
    pub fn compute_or_get(session: &mut TamSession, x: f64) -> Arc<ExpKernelState> {
        let tag = Self::cache_key_for(x);
        if let Some(cached) = session.get::<ExpKernelState>(&tag) {
            return cached;
        }
        let state = Arc::new(Self::compute(x));
        // first-producer-wins per TamSession contract; if another thread
        // beat us we silently lose ours and re-fetch the winner.
        let registered = session.register(tag.clone(), state.clone());
        if registered {
            state
        } else {
            session
                .get::<ExpKernelState>(&tag)
                .expect("kernel state must be present after concurrent register")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_distinguishes_signed_zero() {
        let key_pos = ExpKernelState::cache_key_for(0.0);
        let key_neg = ExpKernelState::cache_key_for(-0.0);
        assert_ne!(
            key_pos, key_neg,
            "+0.0 and -0.0 must have distinct cache keys (signed zero matters)"
        );
    }

    #[test]
    fn cache_key_canonicalizes_nan() {
        let nan_a = f64::NAN;
        // Construct a different NaN bit pattern.
        let nan_b = f64::from_bits(f64::NAN.to_bits() ^ 0x1);
        assert!(nan_b.is_nan(), "test setup: nan_b must be NaN");
        let key_a = ExpKernelState::cache_key_for(nan_a);
        let key_b = ExpKernelState::cache_key_for(nan_b);
        assert_eq!(
            key_a, key_b,
            "different NaN bit patterns must canonicalize to the same cache key"
        );
    }

    #[test]
    fn cache_key_distinguishes_normal_values() {
        let k1 = ExpKernelState::cache_key_for(1.0);
        let k2 = ExpKernelState::cache_key_for(2.0);
        let k3 = ExpKernelState::cache_key_for(1.0 + f64::EPSILON);
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
        assert_ne!(k2, k3);
    }

    #[test]
    fn cache_key_carries_precision_door_policy_tags() {
        // The cache key MUST contain the precision, door, and branch
        // policy bytes. This test is a guard against future refactors
        // that might strip them: removing any tag would cause two
        // structurally distinct states to collide.
        match ExpKernelState::cache_key_for(1.5) {
            IntermediateTag::ExpKernelState {
                x_bits,
                precision_tag,
                door_tag,
                branch_policy_tag,
            } => {
                assert_eq!(x_bits, 1.5_f64.to_bits());
                assert_eq!(precision_tag, PRECISION_TAG_P0F64);
                assert_eq!(door_tag, DOOR_TAG_CPU);
                assert_eq!(branch_policy_tag, BRANCH_POLICY_TAG_REAL_AXIS);
            }
            other => panic!("cache_key_for must return ExpKernelState variant, got {other:?}"),
        }
    }

    #[test]
    fn compute_reduces_to_correct_quotient() {
        // For x = ln(2), k should round to 1 and r should be ≈ 0.
        let s = ExpKernelState::compute(std::f64::consts::LN_2);
        assert_eq!(s.k, 1, "k for x=ln(2) should be 1");
        // r = ln(2) - ln(2) ≈ 0; allow numerical noise.
        let r = s.r();
        assert!(r.abs() < 1e-15, "r for x=ln(2) should be near 0, got {r}");
        // expm1(r) ≈ 0.
        assert!(s.expm1_r.abs() < 1e-14, "expm1_r should be near 0, got {}", s.expm1_r);
    }

    #[test]
    fn compute_zero_gives_zero_state() {
        let s = ExpKernelState::compute(0.0);
        assert_eq!(s.k, 0);
        assert_eq!(s.r_hi, 0.0);
        assert_eq!(s.r_lo, 0.0);
        assert_eq!(s.expm1_r, 0.0);
    }

    #[test]
    fn compute_one_gives_finite_state() {
        // x = 1: k = round(1/ln(2)) = round(1.443) = 1.
        // r = 1 - 1·ln(2) ≈ 0.307.
        // expm1(r) ≈ 0.359.
        let s = ExpKernelState::compute(1.0);
        assert_eq!(s.k, 1);
        let r = s.r();
        assert!((r - 0.306_852_819).abs() < 1e-8, "r for x=1 should be ≈ 0.307, got {r}");
        assert!((s.expm1_r - 0.359_140_914).abs() < 1e-8, "expm1_r for x=1 ≈ 0.359, got {}", s.expm1_r);
    }

    #[test]
    fn compute_or_get_caches_on_second_call() {
        let mut session = TamSession::new();
        let s1 = ExpKernelState::compute_or_get(&mut session, 2.5);
        // Cache contains exactly one entry now.
        assert_eq!(session.len(), 1);
        let s2 = ExpKernelState::compute_or_get(&mut session, 2.5);
        // Still one entry; same Arc.
        assert_eq!(session.len(), 1);
        assert!(Arc::ptr_eq(&s1, &s2), "second call must return the cached Arc");
    }

    #[test]
    fn compute_or_get_distinguishes_inputs() {
        let mut session = TamSession::new();
        let s1 = ExpKernelState::compute_or_get(&mut session, 1.0);
        let s2 = ExpKernelState::compute_or_get(&mut session, 2.0);
        assert_eq!(session.len(), 2);
        assert!(!Arc::ptr_eq(&s1, &s2));
        assert_ne!(s1.k, s2.k);
    }

    #[test]
    fn compute_or_get_handles_signed_zero_separately() {
        // +0 and -0 produce structurally the same kernel state (k=0, r=0,
        // expm1_r=0), but the cache should still distinguish them because
        // downstream consumers might preserve sign-of-zero differently.
        let mut session = TamSession::new();
        let _ = ExpKernelState::compute_or_get(&mut session, 0.0);
        let _ = ExpKernelState::compute_or_get(&mut session, -0.0);
        assert_eq!(session.len(), 2, "+0 and -0 should be cached separately");
    }

    #[test]
    fn r_recovers_full_precision() {
        // For large k, r_hi and r_lo together preserve more bits than
        // r_hi alone. This test sanity-checks that r() returns their
        // sum (which is what consumers see).
        let s = ExpKernelState::compute(100.0);
        let r_combined = s.r();
        let r_hi_only = s.r_hi;
        // r_lo is non-zero for non-zero k:
        assert_ne!(s.r_lo, 0.0, "r_lo should be non-zero for large x where k > 0");
        // r() returns the sum:
        assert_eq!(r_combined, s.r_hi + s.r_lo);
        assert_ne!(r_combined, r_hi_only, "r() should be different from r_hi alone");
    }

    #[test]
    fn cross_precision_gauntlet_placeholder() {
        // Aristotle's T25: kernel state deserves its own cross-precision
        // direct gauntlet. Sweep 35 only ships P0F64, so the "cross-precision"
        // is trivially same-precision. This test asserts the gauntlet shape
        // exists; when P1Extended / P2BigFloat tiers land, replace the body
        // with an actual cross-tier comparison.
        let x_samples: &[f64] = &[0.1, 1.0, 10.0, -1.0, 100.0];
        for &x in x_samples {
            let s_a = ExpKernelState::compute(x);
            let s_b = ExpKernelState::compute(x);
            // Same precision, same input ⇒ bit-identical state.
            assert_eq!(s_a, s_b, "kernel state at x={x} must be deterministic");
        }
    }
}
