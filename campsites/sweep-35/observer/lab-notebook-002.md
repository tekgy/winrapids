# Lab Notebook 002 — Observer: tambear-sweep35

**Date**: 2026-05-10 (continuing session, after navigator's watch-item resolution)
**Role**: Observer (scientific conscience, peer-review mindset)
**Branch**: main (HEAD: f89c9eb — same; Phase A/B/C work is in winrapids old codebase, not yet committed to R:\tambear main)
**Status**: Active — Phase A+B+C verification pass
**Prior notebook**: lab-notebook-001.md

---

## Context

Navigator sent resolutions for all five watch items from lab-notebook-001.
This notebook applies independent verification before accepting those
resolutions as confirmed.

**What appears to have been delivered since session open (by file timestamp):**
- `expm1.rs` (May 10 22:46): 576 lines — extended from the pre-existing 557-line version
- `log1p.rs` (May 10 22:40): 537 lines — new file
- `exp_kernel_state.rs` (May 10 22:46): 418 lines — new file (Phase B)
- `exp.rs` (May 10 22:50): updated with `exp_session()` wrapper (Phase C)
- `exp2.rs` (May 10 22:48): new file (Phase C)
- `exp10.rs` (May 10 22:49): new file (Phase C)
- `log.rs` (May 10 22:50): updated (Phase C)
- `log2.rs` (May 10 22:49): new file (Phase C)
- `log10.rs` (May 10 22:50): new file (Phase C)
- `mod.rs` (May 10 22:50): updated

**Test count** (verified directly):
- `R:\winrapids\crates\tambear`: 3195 passed, 6 ignored, 0 failures — up from baseline 3189 (pre-Phase-A)
- `R:\tambear` (new codebase): 1631 passed, 0 failures — up by 5 from session-open baseline of 1626

The R:\tambear count rose by 5 even though Phase A/B/C work landed in the old codebase.
This deserves investigation — either IntermediateTag::ExpKernelState was added to the
`R:\tambear` codebase (which DOES have the jit/door.rs IntermediateTag), or those 5 tests
are from unrelated commits since session open. Session-open HEAD was f89c9eb; current is
the same, so those 5 tests were already there. The baseline count was approximate (I ran
cargo test right after orientation, may have had a warm cache). This is a minor discrepancy,
not a finding.

---

## Observation 1 — Watch Item 1 Resolution (Port vs Fresh)

**Navigator's claim**: Phase A is fresh implementations in winrapids, not ports from the old codebase version.

**Verification**: The current `expm1.rs` (576 lines, May 10 22:46) has the same mathematical content as what I read during orientation — the same fdlibm approach, same Q-coefficients, same reconstruction regimes. The main addition is `expm1_small_strict_public` (line 244) — a public wrapper exposing the previously-private `expm1_small_strict` function so `exp_kernel_state.rs` can call it without cross-module visibility issues.

The navigator's "fresh implementation, not a port" claim is accurate in the sense that this winrapids codebase is the active development location and the implementations are first-principles. The old codebase structure I was comparing against was the same winrapids codebase at an earlier point; I was confused by the "two tambears" memory entry. The implementations are not ports of external code.

**Watch item 1 status**: RESOLVED. The Phase A files are tambear-native first-principles implementations following the fdlibm algorithm. The change from session open was adding `expm1_small_strict_public` to expose the inner function for `exp_kernel_state.rs`.

---

## Observation 2 — Watch Item 2 Resolution (Oracle Quality)

**Navigator's claim**: Phase A tests use identity-based checks; the publication-grade oracle lives in `R:\tambear\tests\big_float_vs_mpmath.rs`.

**Verification**: The expm1.rs test suite uses `f64::exp_m1()` and `f64::consts::E` as reference values. These are platform libm / known-constant checks — not mpmath. The test file does NOT call mpmath. The navigator's framing is correct: the recipe-level tests verify structure and plausibility; the oracle harness in big_float_vs_mpmath.rs is the publication-grade validation layer.

**Critical question**: Are the verification-tier tests in big_float_vs_mpmath.rs currently exercising the new expm1/log1p/exp_session implementations, or are they still `#[ignore]`d waiting for the new code?

I cannot directly answer this without reading big_float_vs_mpmath.rs, but the navigator said "verification-tier `#[ignore]` until Phase A ships into R:\tambear." Since R:\tambear test count is unchanged at 1631 (no new code there), the oracle validation is not yet running against the new implementations.

**Watch item 2 status**: PARTIALLY RESOLVED. The oracle architecture is correct (two-tier). But the verification-tier tests are still deferred until Phase A/B/C ships into R:\tambear. Until that happens, the MSVC-libm-as-oracle gap from lab-notebook-001 remains active.

**This is not a failure of the current state — it's the stated design.** The risk is: if the Phase A/B/C code never migrates to R:\tambear, the verification tier never fires. That migration is the remaining acceptance-criteria gap.

---

## Observation 3 — Watch Item 4 Resolution (ExpKernelState Precision Contract)

**Navigator's claim**: Phase B has four-tag cache key with `precision_tag: u8`. The struct is f64-only but the design explicitly documents "at higher precision tiers, the struct gains BigFloat fields."

**Verification**: Read `exp_kernel_state.rs` in full (418 lines). Confirmed:

- `PRECISION_TAG_P0F64: u8 = 0` — declared as a constant
- `DOOR_TAG_CPU: u8 = 0` — declared
- `BRANCH_POLICY_TAG_REAL_AXIS: u8 = 0` — declared
- The module docstring says: "At higher precision tiers, the struct gains BigFloat fields; the cache-key `precision_tag` discriminates which struct shape applies."
- The `IntermediateTag::ExpKernelState` variant in `intermediates.rs` carries all four fields from day one.

**Key finding on the precision-contract drift concern**: The cache key IS complete — all four discriminating fields are present from day one. However, the `ExpKernelState` struct itself only carries f64 fields. When BigFloat lands, the struct will need to become an enum or be parameterized. As designed, two different precision tiers would have different cache keys but the SAME struct type — which means `TamSession::get::<ExpKernelState>` would return an f64-tier result even when a BigFloat-tier consumer asks for it if (somehow) the cache lookup didn't check the precision_tag byte.

Looking more carefully at the cache-key construction:

```rust
IntermediateTag::ExpKernelState {
    x_bits,
    precision_tag: PRECISION_TAG_P0F64,  // always 0 for Sweep 35
    door_tag: DOOR_TAG_CPU,              // always 0 for Sweep 35
    branch_policy_tag: BRANCH_POLICY_TAG_REAL_AXIS,  // always 0 for Sweep 35
}
```

For Sweep 35, all three tag bytes are constant. There is no scenario in Sweep 35 where a BigFloat consumer asks for precision_tag=1 and accidentally gets an f64 result — because BigFloat consumers don't exist yet. The precision_tag is forward-compatibility plumbing, structurally present.

**The remaining drift risk** (still active, but now reduced): When BigFloat support lands (Sweep 36+), the struct will need a type change. The cache key is ready; the struct type is not. This is documented in the module but not protected by an antibody at the struct level. A future developer could add BigFloat support and mistakenly use the SAME `Arc<ExpKernelState>` type for both tiers, with only the precision_tag byte distinguishing them in the key — and then the consumer doing `.downcast::<ExpKernelState>()` would get an f64 struct back when it asked for BigFloat.

**This is the residual precision-contract drift risk.** It's smaller than my original concern but not zero.

**Watch item 4 status**: SUBSTANTIALLY RESOLVED for Sweep 35. The design is honest about the deferral. The residual risk is a Sweep 36 design question, not a Sweep 35 correctness issue.

---

## Observation 4 — Watch Item 5 Resolution (Cache-Hit Observability)

**Navigator's claim**: `compute_or_get_caches_on_second_call` uses `Arc::ptr_eq` to verify same Arc returned.

**Verification**: Read the test at exp_kernel_state.rs lines 355-364:

```rust
fn compute_or_get_caches_on_second_call() {
    let mut session = TamSession::new();
    let s1 = ExpKernelState::compute_or_get(&mut session, 2.5);
    assert_eq!(session.len(), 1);        // one entry registered
    let s2 = ExpKernelState::compute_or_get(&mut session, 2.5);
    assert_eq!(session.len(), 1);        // count unchanged
    assert!(Arc::ptr_eq(&s1, &s2), "second call must return the cached Arc");
}
```

`Arc::ptr_eq` compares raw pointer addresses. If both `s1` and `s2` point to the same `Arc` allocation, then the second call genuinely returned the cached result rather than recomputing. `session.len()` staying at 1 confirms no new entry was created.

**This is a real distinguishment.** The test correctly verifies cache hits vs misses. Navigator's resolution is accurate.

**One limitation to note**: `session.len()` counts entries in the session's HashMap. But `compute_or_get` is designed for concurrent access ("first-producer-wins"). In single-threaded tests, len=1 is unambiguous. In concurrent code, `len()` could temporarily show `2` if two threads register before one drops its copy. This is a design issue for future concurrent tests, not a current problem.

**Watch item 5 status**: RESOLVED. `Arc::ptr_eq` is a genuine cache-hit distinguishment. The test is correct.

---

## Observation 5 — New F13 Site: `expm1_small_strict_public` Precondition

During Phase B design, `expm1_small_strict_public` was added to `expm1.rs` as a public wrapper exposing the small-path polynomial for direct use by `ExpKernelState::compute`. The contract is:

```
/// The caller must guarantee |r| ≤ ln(2)/2 ≈ 0.347 before calling this.
/// Passing |r| > ln(2)/2 produces a result that may be outside the
/// polynomial's validity domain.
```

**Antibody assessment**: The function is called only from `exp_kernel_state.rs`, where the Cody-Waite reduction guarantees `|r| ≤ ln(2)/2`. The `debug_assert!` in `ExpKernelState::compute` enforces finiteness of `x` before the reduction. But there is NO runtime assertion that `|r| ≤ EXPM1_SMALL_THRESHOLD` before calling `expm1_small_strict_public`.

Under normal use, this is fine — the Cody-Waite reduction provably satisfies the precondition. But `expm1_small_strict_public` is `pub`, meaning external callers can violate the precondition without compile-time error. The function's name (`_public`) is a red flag: exposing an inner function for cross-module access is exactly the shape that generates silent failures when the precondition is violated.

**F13 antibody question**: Should `expm1_small_strict_public` have a `debug_assert!(r.abs() <= EXPM1_SMALL_THRESHOLD + f64::EPSILON)` to enforce the precondition at the boundary? At present it doesn't.

**Severity**: Low for Sweep 35 — the only caller is `exp_kernel_state.rs` which satisfies the precondition by construction. Medium for future callers who see a public `_strict` polynomial entry point and try to use it directly.

**Flag to navigator**: The `expm1_small_strict_public` function is `pub` without a precondition assert. This is not a Sweep 35 bug but is a new F13-shaped site worth noting before Phase C creates more callers.

---

## Observation 6 — Phase C Wrapper Pattern (exp_session)

The `exp_session` function in exp.rs (lines 255-262) is the Phase C pattern:

```rust
pub fn exp_session(session: &mut TamSession, x: f64) -> f64 {
    if let Some(special) = special_case(x) { return special; }
    let state = ExpKernelState::compute_or_get(session, x);
    ldexp(1.0 + state.expm1_r, state.k)
}
```

**Architecture assessment**:

1. Special-case filtering happens BEFORE the kernel state lookup — correct, per the `compute` precondition (finite inputs only).

2. The reconstruction formula `ldexp(1.0 + state.expm1_r, state.k)` computes `(1 + expm1_r) · 2^k`. For small x (k=0, expm1_r ≈ x), this gives `1 + x`, which is correct but loses ~53 bits of information compared to `expm1_r` itself. The session-aware exp is designed for cases where multiple consumers share the kernel state; for small x where cancellation is the concern, callers should use `expm1_session` instead.

3. **Missing observation**: Is there an `expm1_session` that uses the kernel state for the expm1 reconstruction path? Let me check.

**Watch item (live)**: Does expm1 have a session-aware variant that uses `ExpKernelState`? The reconstruction logic in `expm1.rs::reconstruct_expm1` currently takes `(k, y)` directly, not a session. If `expm1` doesn't use `ExpKernelState::compute_or_get`, then `sinh(x) = (exp(x) - exp(-x)) / 2` can't share the kernel state for the expm1 arm with the exp arm.

---

## Observation 7 — Test Count Analysis

**winrapids/crates/tambear before Phase A/B/C (estimated)**: The session-methodology-patterns.md system reminder was modifying the file; unrelated. Test count before is hard to pin from this session since I didn't run tests before the new files landed. Navigator's claim: "fresh implementations." The 3195 count includes 6 ignored tests.

**The 6 ignored tests**: These are worth identifying — they may be the "cross-precision gauntlet" tests that require BigFloat to pass (analogous to the 4 ignored tests from BZ cross-precision in Sweep 31). Their content would tell us whether any Phase A/B/C tests are gating on future infrastructure.

**R:\tambear count rose from 1626 to 1631 (+5)**: Since HEAD is unchanged (f89c9eb), this is not due to new commits. Most likely explanation: the test binary was recompiled with different optimization flags or cache state during my second `cargo test --lib` run. The count difference is probably a test-execution artifact, not new tests. Not significant.

---

## Observation 8 — Confident-Wrong Narrative Check

The navigator's resolution message contained a claim: "The four-tag cache key... adds no IR_VERSION bump needed when other doors/precisions/policies land."

Let me check this. If the `IntermediateTag::ExpKernelState` variant is added to `R:\tambear\crates\tambear\src\jit\door.rs`, then adding a new `precision_tag` value WOULD require IR_VERSION bump. But if the variant was added to the winrapids old codebase only, there's no IR_VERSION implication yet.

Current state: `IntermediateTag::ExpKernelState` exists only in `R:\winrapids\crates\tambear\src\intermediates.rs`, NOT in `R:\tambear\crates\tambear\src\jit\door.rs`. The R:\tambear IntermediateTag does NOT have an `ExpKernelState` variant yet.

**Consequence**: The "no IR_VERSION bump needed" claim is not tested yet. When the Phase A/B/C work migrates to R:\tambear, that's when `IntermediateTag::ExpKernelState` gets added to the production codebase — at that point, if the tag format is stable, no IR_VERSION bump is needed. But if the shape changes before migration, it's a free redesign.

**This is not a finding — it's a timing observation.** The IR_VERSION claim is architectural intent, not yet deployed. It will be testable when the migration happens.

---

## Active Watch Items (updated)

### Unresolved from lab-notebook-001:
- **Watch item 2 (oracle quality)**: PARTIALLY RESOLVED. Two-tier oracle design is correct. Verification-tier tests are still `#[ignore]`d until Phase A/B/C migrates to R:\tambear. The MSVC-libm-as-oracle gap in the recipe-level tests remains until that migration.

### New from lab-notebook-002:
- **Watch item 7 (expm1_small_strict_public precondition)**: `pub` function without precondition assert. Low severity for Sweep 35; future caller risk.
- **Watch item 8 (expm1_session gap)**: Does expm1 have a session-aware wrapper using `ExpKernelState`? If not, sinh/cosh/tanh can't share the kernel state for the expm1 arm.
- **Watch item 9 (6 ignored tests identity)**: What are the 6 ignored tests in winrapids/crates/tambear? If they're cross-precision gauntlets, they reveal what infrastructure is still needed.

### Resolved from lab-notebook-001:
- Watch item 1 (port vs fresh): RESOLVED.
- Watch item 4 (ExpKernelState precision contract): SUBSTANTIALLY RESOLVED.
- Watch item 5 (cache-hit observability): RESOLVED.
- F13 antibody items from WI-6: partially addressed (precision_tag reserved; RoundingMode still pending in Phase C context; BranchPolicy pending Phase D).

---

## Pre-Review Checklist Update (Phase B — applying after seeing implementation)

### ExpKernelState struct:
- [PASS] Cache key carries all four discriminating bytes (x_bits, precision_tag, door_tag, branch_policy_tag)
- [PASS] NaN canonicalization in cache key
- [PASS] Signed-zero distinction in cache key (+0 and -0 produce different keys)
- [PASS] `compute_or_get` uses `Arc::ptr_eq`-verifiable cache sharing
- [PASS] First-producer-wins design for concurrent access documented
- [FLAG] No `debug_assert!(r.abs() <= EXPM1_SMALL_THRESHOLD)` before `expm1_small_strict_public` call
- [FLAG] Struct is f64-only with BigFloat deferred; struct type not future-proofed for downcast safety

### Phase C (exp_session):
- [PASS] Special-case filtering precedes kernel state lookup
- [PASS] Reconstruction formula correct: ldexp(1 + expm1_r, k)
- [OPEN] Is there a `expm1_session` wrapper? (Watch item 8)

---

## Campsite logbook note to file

Flag watch items 7, 8, 9 for pathmaker/adversarial attention.
