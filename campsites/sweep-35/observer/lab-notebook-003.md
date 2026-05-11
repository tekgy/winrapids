# Lab Notebook 003 — Observer: tambear-sweep35

**Date**: 2026-05-10 (continuing session)
**Role**: Observer (scientific conscience, peer-review mindset)
**Branch**: main (HEAD: f89c9eb — Phase A/B/C in winrapids, not yet committed to R:\tambear)
**Status**: Active — Phase C/D verification + reconciling findings from lab-notebook-002
**Prior notebooks**: lab-notebook-001.md, lab-notebook-002.md

---

## Context

The team is in Phase C (recipe wrappers). I have three watch items from
lab-notebook-002 to resolve, and a new finding about `cosh_session` and
`tanh_session` being stubs. Additionally, the team-briefing.md was updated
to add "Standing Constraint 11: Two-repo architecture" — my observation from
lab-notebook-001 §"Observation 1" has been formally codified.

**Current test count** (winrapids/crates/tambear): 3228 passed, 6 ignored.
Previous count when I ran: 3195. Delta: +33 tests in ~30 minutes.
The 6 ignored tests are: 5 in compute_engine.rs (#[ignore = "requires CUDA GPU"])
and 1 in erf.rs (#[ignore = "diagnostic probe"]). These are pre-existing; not
cross-precision gauntlets for Sweep 35. Watch item 9 resolved.

---

## Observation 1 — Reconciliation: The "naive expm1 in hyperbolic.rs" Finding

lab-notebook-001 noted confusion about the pathmaker's campsite note describing
"hyperbolic.rs has temp expm1 (naive exp(x)-1 with 1e-9 Taylor cutoff)" while
I had found a full `expm1.rs` file. Reading `hyperbolic.rs` header resolves this:

The hyperbolic.rs module had a *local* naive expm1 stub (inline `fn expm1(x)`)
that called `exp(x) - 1.0` with a 1e-9 Taylor cutoff. Phase A replaced this
local stub with a call to `super::expm1::expm1_strict`. The stub was replaced, 
not the dedicated `expm1.rs` file.

So: the pathmaker was accurate about what they found and what they fixed. The
dedicated `expm1.rs` was the Phase A work they shipped — extending the pre-existing
implementation with `expm1_small_strict_public` and documenting the upgrade. What
changed in Phase A:
1. `expm1.rs`: added `expm1_small_strict_public` wrapper for use by exp_kernel_state.rs
2. `hyperbolic.rs`: replaced local naive stub with `super::expm1::expm1_strict`

This reconciliation clarifies that Phase A's scope was: create the shared intermediate
infrastructure, expose the precision-safe polynomial for kernel state use, and wire the
previously-naive hyperbolic stub to the proper implementation.

---

## Observation 2 — cosh_session and tanh_session: Documented Stubs

`cosh_session` and `tanh_session` are documented pass-throughs to the strict variants.
The module comment at line 219 says:

> "A future `BidirectionalExpKernelState` (Sweep 36+) would bundle them to enforce
> same-precision-context as a structural invariant; for now the two states share a
> TamSession but are independent objects."

**Assessment**: This is honest, documented deferral. The session parameter is
structurally present so the API is stable; the kernel-state wiring is deferred.
The acceptance criteria only require "ExpKernelState sharing via TamSession
verified" — which is satisfied by `exp_session` and `sinh_session`.

**However**: `sinh_session` only uses `ExpKernelState::compute_or_get` for ax in [0.125, 1.0].
For ax < 0.125 or ax > 1.0, it falls through to `sinh_strict`, which does NOT use
the session. If two consumers call `sinh_session(x)` and `exp_session(x)` for the same
`x` with ax > 1.0, the `sinh_session` call does NOT share the kernel state with the
`exp_session` call.

This is a different kind of stub — not documented as such. The function is advertised as
"session-aware" and "pulls exp(x) and exp(-x) from kernel state" (briefing language),
but for ax > 1.0 it doesn't. The acceptance criterion says sinh "pulls exp(x) and exp(-x)
from kernel state" — this is not currently true for ax > 1.0.

**Watch item 10 (new)**: `sinh_session` falls through to `sinh_strict` for |x| > 1.0
without session-based kernel state sharing. The sinh formula for |x| > 1.0 is
`(exp(x) - exp(-x)) / 2` — exactly where the two ExpKernelState lookups would share.
This regime is currently NOT sharing the kernel state.

**Severity**: Medium. The function is mathematically correct (sinh_strict is correct).
But the architectural claim ("pulls exp(x) and exp(-x) from kernel state") is false for
|x| > 1.0, which is the majority of the domain for large-argument sinh.

---

## Observation 3 — Cross-Precision Proptest Status

The acceptance criterion says "Cross-precision proptests green for all named functions."

The task list shows task #8 "Cross-precision proptest gauntlet" as completed. Let me
verify what this looks like in the code.

**Finding**: I have not yet read the cross-precision proptest file for Sweep 35. This is
an outstanding verification step. The prior session's Sweep 31 had `big_float_cross_precision.rs`
with 4 ignored tests blocking on `BigFloat::with_precision_rounded`. The analogous concern
for Sweep 35: are the cross-precision proptests testing at multiple precision tiers (P0F64 +
P2BigFloat) or only at a single precision tier?

For the exp/log family at P0F64 only, "cross-precision" is trivially same-precision. A
genuine cross-precision proptest would compare `expm1_strict(x)` at f64 against
`BigFloat::exp_m1(x, p=200)` rounded to f64, verifying ≤1 ULP. If the test only compares
two calls to `expm1_strict(x)`, it's a consistency test, not a cross-precision test.

This was the concern I raised in my notebook-001 pre-review checklist:
> "Tests use mpmath as oracle (not platform libm)"

**Watch item 11 (new)**: What do the cross-precision proptests actually test? If they're
only consistency tests (strict A vs compensated A) rather than cross-tier tests (f64
vs BigFloat), the "cross-precision proptests green" acceptance criterion is being read
too weakly.

---

## Observation 4 — sinh_session Formula Analysis

Looking more carefully at the sinh_session formula for ax in [0.125, 1.0]:

```rust
let h = if pos.k == 0 {
    pos.expm1_r
} else {
    ldexp(1.0 + pos.expm1_r, pos.k) - 1.0   // (1 + expm1_r) * 2^k - 1
};
let result = h * (h + 2.0) / (2.0 * (h + 1.0));
```

The `h = expm1(|x|)` reconstruction:
- For k=0 (|x| < ln(2)/2 ≈ 0.347): `h = expm1_r` directly. This is exact.
- For k=1 (|x| in [0.347, 0.693]): `h = (1 + expm1_r) * 2 - 1 = 2*expm1_r + 1`. 
  But `expm1(x)` for x ≈ 0.693 = ln(2) is approximately 1.0 — no precision risk here.
- For k ≥ 2: not reached since ax ≤ 1.0 means k ≤ 1.

The formula `h * (h + 2) / (2 * (h + 1))` for sinh(x) when h = expm1(x):
Algebraically: `expm1(x) * (expm1(x) + 2) / (2 * (expm1(x) + 1))`
= `(exp(x) - 1) * (exp(x) + 1) / (2 * exp(x))`
= `(exp(x)^2 - 1) / (2 * exp(x))`
= `(exp(2x) - 1) / (2 * exp(x))` ... wait, that's not right.

Let me redo: `(e^x - 1)(e^x + 1) = e^(2x) - 1`. Then divided by `2*e^x`:
= `(e^(2x) - 1) / (2*e^x) = (e^x - e^(-x)) / 2 = sinh(x)`. Correct.

**The formula is mathematically valid.** The precision argument: for small x where
expm1(x) ≈ x, h ≈ x, h*(h+2)/(2*(h+1)) ≈ x*(x+2)/(2*(1+x)) ≈ x*(2/2) = x for
small x (sinh(x) ≈ x). No catastrophic cancellation in this path.

**Assessment**: The sinh_session formula for ax in [0.125, 1.0] is correct and
precision-safe. The fall-through to sinh_strict for ax > 1.0 is the gap.

---

## Observation 5 — Watch Items Summary (consolidated)

### Resolved:
- **WI-1 (port vs fresh)**: RESOLVED. Phase A extended existing expm1.rs; replaced local naive stub in hyperbolic.rs.
- **WI-2 (oracle quality)**: PARTIALLY RESOLVED. Two-tier design is correct; verification-tier still deferred.
- **WI-3 (ULP budget discrepancy)**: RESOLVED. Old file is unchanged; irrelevant to Phase A which landed in same file with separate API.
- **WI-4 (ExpKernelState precision contract)**: SUBSTANTIALLY RESOLVED. Tags structurally present; f64-only design honest.
- **WI-5 (cache-hit observability)**: RESOLVED. Arc::ptr_eq is genuine.
- **WI-9 (6 ignored tests)**: RESOLVED. 5 CUDA GPU tests, 1 diagnostic probe — pre-existing, unrelated to Sweep 35.

### Active:
- **WI-7 (expm1_small_strict_public precondition)**: `pub` function without precondition assert. Low severity.
- **WI-10 (sinh_session regime gap)**: sinh_session falls through to sinh_strict for |x| > 1.0. Kernel state sharing doesn't fire for large-argument sinh. Medium severity against the acceptance criterion.
- **WI-11 (cross-precision proptest quality)**: Are the cross-precision proptests genuine cross-tier tests or same-tier consistency checks? Needs verification.

---

## Observation 6 — Things I Have NOT Yet Reviewed

For completeness, the following Phase C/D files landed today and I have NOT read in detail:
- `exp2.rs` (12.4KB)
- `exp10.rs` (7.6KB)
- `log.rs` (11.8KB, updated)
- `log2.rs` (7.2KB)
- `log10.rs` (6.2KB)

And Phase D:
- No `complex_log.rs` visible in directory listing. Task #9 (complex_log) is still pending per task list.
- `branch_cut_conventions.md` was ratified in DEC-032 but complex_log implementation hasn't landed yet.

The +33 test delta (3195 → 3228) since my earlier run suggests these files did land and
tests were added. I should verify whether `complex_log.rs` exists in the directory.

---

## Publishability Assessment — Phase A+B+C State

Against the Tambear Contract item 10 criteria:

**Would survive peer review**:
- The factoring of exp/log into a shared kernel intermediate is architecturally sound and well-documented.
- The fdlibm Q-coefficient heritage is explicitly cited; the bit-identity claim is checkable.
- The `Arc::ptr_eq` cache-hit test is a real correctness demonstration.
- `ExpKernelState` cache key design (four-field, forward-compatible) follows from first principles and is documented.

**What would draw reviewer attention**:
- `sinh_session` for |x| > 1.0 doesn't share the kernel state — the primary point of the factoring. A reviewer would note this immediately.
- No `expm1_session` — callers computing `expm1(x)` and `exp(x)` at the same `x` can't share the state via the public API.
- `cosh_session` and `tanh_session` are documented stubs; a reviewer would flag these as incomplete Phase C implementations.
- The mpmath oracle gap: the recipe-level tests validate against platform libm, not mpmath. Publication-grade verification is deferred to the oracle harness (currently `#[ignore]`d for the new implementations).

**Overall**: Phase A+B are well-grounded. Phase C is partially complete — exp, exp2, exp10, log, log2, log10, and the sinh session-aware variant are delivered; cosh and tanh session-aware variants are stubs. The architectural plumbing (kernel state, cache key, TamSession registration) is solid. The acceptance criteria's "all named functions with cross-precision proptests" remains to be verified independently.

---

## Next Audit Step

If I'm idle between team messages, I'll read the `exp2.rs` or `log.rs` wrappers
to check the Phase C wrapper pattern uniformity — do they all filter specials before
kernel state lookup? Do they all use `ExpKernelState::compute_or_get` or do any
fall back to direct computation?
