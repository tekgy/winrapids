# Lab Notebook 001 — Observer: tambear-sweep35

**Date**: 2026-05-10
**Role**: Observer (scientific conscience, peer-review mindset)
**Branch**: main (HEAD: f89c9eb)
**Status**: Active — initial orientation pass
**Session**: tambear-sweep35

---

## Context & Orientation

Continuation of tambear-sweep31-finish. Prior team shipped Sweeps 31-34,
DEC-032 (BranchPolicy), DEC-033 (TamSession dedupe), holonomic architecture
ratification, and the libm-factoring design synthesis. This team picks up
the implementation.

**Baseline test count (verified at session open)**:
- `cargo test --lib`: 1626 passed, 0 warnings (5 suites)
- HEAD: f89c9eb "oracle/tan: re-derive six follow-ups from substrate"

---

## Observation 1 — Substrate Inventory: Two Tambear Codebases

The team-briefing refers to `R:\tambear\` as the active codebase. But
`R:\winrapids\crates\tambear\` is a distinct older codebase. Memory entry
`project_two_tambears.md` says: "TWO tambear codebases: old winrapids/crates/tambear
(has recipes/libm/) vs new R:\tambear (locked vocab, no libm yet); trig hasn't
been ported."

**What the old codebase contains (verified by `ls`):**
- `R:\winrapids\crates\tambear\src\recipes\libm\exp.rs` — 479 lines
- `R:\winrapids\crates\tambear\src\recipes\libm\expm1.rs` — 557 lines
- `R:\winrapids\crates\tambear\src\recipes\libm\hyperbolic.rs` — exists
- `R:\winrapids\crates\tambear\src\recipes\libm\inv_hyperbolic.rs` — exists
- `R:\winrapids\crates\tambear\src\recipes\libm\mod.rs` — exists

**What `R:\tambear\` (the active codebase) contains:**
- No `expm1.rs`, no `exp.rs`, no `hyperbolic.rs`, no `libm/` directory

**What this means for Sweep 35:**

The pathmaker's campsite note says they found "hyperbolic.rs has temp expm1
(naive exp(x)-1 with 1e-9 Taylor cutoff)." This is NOT what I found.
`expm1.rs` in the old codebase is a full, production-quality implementation:
- fdlibm Q-coefficients (bit-identical claim)
- Three entry points: strict/compensated/correctly_rounded
- Reconstruction regime handling (k=0, k<56, k≥56, k<-56)
- Special cases: NaN/Inf/−Inf/±0/overflow/underflow
- 20+ tests including precision contract tests and identity verification

**The pathmaker may have been looking at `hyperbolic.rs` (which probably has
a naive `expm1` stub used for intermediate computation) rather than the
dedicated `expm1.rs`.** This is worth clarifying — if the full `expm1.rs`
exists in the old codebase, the porting question is whether to use that
as the template vs rewrite from scratch for the new tambear. Either way,
the design is done; the port is the work.

**Watch item #1**: Determine whether pathmaker's Phase A work is a fresh
implementation or a port/adaptation from the old codebase. The answer
changes the review criteria: a fresh implementation needs its polynomial
coefficients independently verified; a port needs its equivalence to the
original verified.

---

## Observation 2 — expm1.rs Quality Assessment (Old Codebase)

Read `R:\winrapids\crates\tambear\src\recipes\libm\expm1.rs` in full.

**Design quality**: High. The implementation follows fdlibm's approach
(s_expm1.c), which is the canonical reference for precision-safe expm1.

**Coefficient source claim**: "Q1..Q5 are bit-identical to fdlibm's
(hex-verified)" and references Sun fdlibm `s_expm1.c`. This is a verifiable
claim. Math-researcher should confirm.

**ULP target claims** (from the module's docstring table):
- `expm1_strict`: ≤ 2 ULP
- `expm1_compensated`: ≤ 1 ULP
- `expm1_correctly_rounded`: ≤ 1 ULP

**What the test suite actually tests against:**
- `f64::exp_m1()` — Rust's standard library expm1, which calls the platform
  libm (MSVC on Windows). This is NOT mpmath. This is a potential issue.
  If platform libm has errors (Sweep 34 showed MSVC exp has ≤1 ULP most
  places but some outliers), using `f64::exp_m1()` as reference means the
  test "passes" even if both the implementation AND the reference are wrong
  in the same direction.

**Watch item #2**: The existing `expm1.rs` tests use `f64::exp_m1()` as
oracle. For Sweep 35, the acceptance criterion requires mpmath validation
(Tambear Contract item 10). The old test suite is a *plausibility* check,
not a publication-grade check.

**Hypothesis for the strict-path budget (4 ULP stated in test body)**:
The module docstring claims ≤ 2 ULP for strict, but the test calls
`check_strategy(expm1_strict, "expm1_strict", 4)` with max_ulps=4. This
is a discrepancy: the docstring claims the tighter bound; the test only
asserts the looser one. Either the implementation is actually ≤ 4 ULP
(not ≤ 2) or the test is conservative. This needs resolution before
claiming the strict budget.

**Watch item #3**: ULP budget discrepancy in expm1.rs — docstring says
≤ 2 ULP strict, test body asserts ≤ 4 ULP. Whichever is true, the
mpmath validation will reveal it.

---

## Observation 3 — Phase A Precision Contract Analysis

The sweep-35-briefing specifies:
> "Cross-precision proptest gauntlet (per Phase C pattern from BZ unstub)
> for each: compute at p_high, round to p_low, verify ≤1 ULP cross-precision drift."

The briefing also specifies expm1's precision contract at different
PrecisionContext tiers (open question 6 in libm-factoring.md):
> "At P0F64, the kernel state's r needs ~53 bits; at P2BigFloat{1024},
> r needs different decomposition."

**Critical precision-contract question I'm watching:**

The old codebase's `expm1.rs` operates at f64 precision only. It has three
*quality* tiers (strict/compensated/correctly_rounded) but they all return
`f64`. There is no BigFloat path.

When pathmaker implements Phase B (`ExpKernelState`), the structure needs:
```rust
pub struct ExpKernelState {
    pub k: i32,
    pub r: f64,        // This is f64. At P2BigFloat{1024}, this won't do.
    pub expm1_r: f64,  // Same issue.
}
```

**The precision-contract drift risk (my primary watch item from the briefing)**:

If `ExpKernelState` is designed for P0F64 and the BigFloat precision path
is deferred, the design is incomplete. The holonomic-architecture.md says
the state is content-addressed by `(x_bits, precision_context)` — which
implies the precision context IS part of the cache key. If the struct
only holds f64 fields, there's no place to carry the BigFloat precision.

Two possible designs:
1. `ExpKernelState<P>` — generic over PrecisionContext, carries either
   f64 or BigFloat fields depending on P
2. `ExpKernelStateF64` and `ExpKernelStateBF{p}` as separate types

Either design is fine. What's NOT fine: a single f64-only struct with a
comment "BigFloat to be added later." That's tech debt in a system that
prohibits tech debt.

**Watch item #4**: When Phase B lands, does `ExpKernelState` have a
precision-context-parameterized design, or is it f64-only with a deferred
BigFloat path?

---

## Observation 4 — TamSession Architecture: What Exists vs What's Needed

I searched for TamSession, IntermediateTag, BranchPolicy in the active
`R:\tambear\` codebase.

**What exists:**
- `IntermediateTag` struct in `crates/tambear/src/jit/door.rs` — a
  content-addressed cache key for shareable intermediates
- `get_intermediate` / `store_intermediate` methods on the `ErasedDoorBackend`
  trait in `jit/door.rs`
- `holonomic_invariance.rs` test (recently committed at `1a69327`) —
  proptest for cache-key as monoid homomorphism

**What doesn't exist (yet):**
- `ExpKernelState` struct
- Registration of `expm1_r` as a shareable intermediate via `IntermediateTag`

**What the acceptance criteria says:**
> "ExpKernelState sharing via TamSession verified — re-running an op with
> same (x, p) hits cache"

For this to be testable, there must be:
1. A way to construct an `IntermediateTag` for `(x, precision_context)`
2. A test that calls the exp function twice with the same x and verifies
   that the second call returns a cache hit (not a fresh computation)

The current `IntermediateTag` implementation stores bytes (the content-
addressed hash). The test would need to observe a "hit" vs "miss" — which
requires either a counter on the tag or a mock backend that records calls.

**Watch item #5**: The "cache actually firing" criterion in the acceptance
criteria requires an observable distinction between cache hit and miss. How
will Phase B implement this observability? If there's no counter or mock
backend, the test can only verify that two calls return the same value —
which passes even if both are recomputed (no sharing).

---

## Observation 5 — F13-Shaped Antibody Audit: New Sites in Phase A–D

The briefing says to watch for "new F13-shaped antibodies surfacing."

F13.C is: signature-level non-defaulted parameters are the strongest
antibody. DEC-032 applied this to BranchPolicy (complex transcendental
branch cut conventions must be non-defaulted at every signature).

**New potential F13 sites in Sweep 35:**

1. **PrecisionContext in ExpKernelState construction**: If `ExpKernelState::new(x: f64)`
   doesn't require a PrecisionContext parameter, it silently constructs
   the state at f64 precision even when the caller intends BigFloat precision.
   The antibody: PrecisionContext non-defaulted at `ExpKernelState::new`.

2. **RoundingMode in expm1_r construction**: The `expm1_r` value inside
   the kernel state is computed with some rounding mode. If the rounding
   mode is implicit (defaulting to RNE), a caller who needs RTZ results
   will silently get RNE intermediates passed to them. The antibody:
   RoundingMode non-defaulted in the ExpKernelState computation path.

3. **BranchPolicy in complex_log (Phase D)**: Already ratified in DEC-032.
   Watch that the implementation actually enforces non-defaulted at every
   signature, not just at the recipe top level.

These are hypotheses. If Phase A/B/C land without them, flag to navigator.

---

## Watch Items (active at session open)

1. Is Phase A a fresh implementation or a port from the old codebase's
   `expm1.rs`? The old file is substantive; the port question matters for
   review criteria.

2. The old `expm1.rs` tests use `f64::exp_m1()` (platform libm) not mpmath
   as oracle. Sweep 35's acceptance criteria require mpmath. The old test
   suite is plausibility, not publication-grade verification.

3. ULP budget discrepancy in old `expm1.rs`: docstring claims ≤ 2 ULP
   strict; test body asserts ≤ 4 ULP. Needs resolution.

4. `ExpKernelState` precision-contract drift: will Phase B carry BigFloat-
   compatible fields, or defer BigFloat support?

5. "Cache actually firing" requires observable distinguishment between
   cache hit and miss. How will this be tested?

6. Three new potential F13 sites: PrecisionContext in kernel state
   construction; RoundingMode in expm1_r computation; BranchPolicy
   enforcement in complex_log (Phase D) at every signature level.

---

## Pre-Review Checklist (Phase A — written before seeing implementation)

Hypothesis before observation. Apply to expm1 and log1p when they land.

### For expm1:
- [ ] Polynomial coefficients verified against fdlibm's s_expm1.c (if porting)
      OR against Remez computation output (if fresh) — bit-exact hex constants
- [ ] Tiny path: |x| < 2^-54 returns x directly (no polynomial overhead)
- [ ] Small path: |x| < ln(2)/2, no reduction, polynomial at x
- [ ] Reconstruction regime: k=0, 0<|k|<56, |k|≥56 — each handled distinctly
- [ ] sign-of-zero: expm1(-0) = -0 (Kahan 1987 contract)
- [ ] Special cases: NaN propagates, +∞→+∞, -∞→-1, overflow→+∞, underflow→-1
- [ ] Tests use mpmath as oracle (not platform libm)
- [ ] Tests include adversarial regime boundaries (|k|=55/56/57, tiny/small transition)
- [ ] `cargo test --lib` green, 0 warnings

### For log1p:
- [ ] Domain check: log1p(-1) = -∞, log1p(x < -1) = NaN
- [ ] Small path: |x| < 2^-54 returns x directly
- [ ] Main path: standard log reduction + log1p polynomial
- [ ] sign-of-zero: log1p(-0) = -0
- [ ] Special cases: NaN, ±∞, x = -1 → -∞
- [ ] Tests use mpmath as oracle
- [ ] Tests include x near -1, x near 0, x >> 1

### Cross-cutting (Phase A):
- [ ] PrecisionContext parameter present at recipe level (not defaulted silently)
- [ ] No dependency on platform libm in implementation (only in tests, as comparison)
- [ ] Both recipes registered as TamSession-shareable intermediates (or design
      decision documented for why they're not)

---

## Campsite logbook entry

Marked campsite sweep-35/20260510223356-observer as active.
