# Aristotle's 5+1 Liftability Attack Analysis

**Date**: 2026-04-22  
**Adversarial**: attacking `default_strategy()` decision tree  
**Assignment source**: aristotle's addendum-liftability-default.md

---

## The Question

Find `(JitOp, Grouping, Shape)` tuples where `default_strategy()` returns
`Lifted` but lifting produces WRONG results or is NOT YET EXECUTABLE. Five
candidates + one bonus from aristotle. Results below.

---

## Candidate 1 — ArgMax + Prefix → Lifted

**Decision tree path**: `lift_safe_with_index = true` → `Lifted`.

**Algebraic correctness**: CORRECT. The `(value, index)` state carries the
tiebreak information explicitly. `combine` is associative (stated in
`canonical_structure`, proven by test `argmax_lowest_index_tiebreak`). A
parallel tree reduces `(v_i, i)` pairs — at any tree node, `combine` picks
the higher value, breaking ties by lowest index. The result is independent
of reduction tree shape because the index IS the order information. No
semantic difference between sequential and parallel.

**Runtime executability**: BROKEN TODAY. `ArgMax::lift()` is a stub that
panics:

```rust
fn lift(&self, _x: f64) -> Self::State {
    crate::stub!(name: "argmax_lift_needs_index", domain: "ops",
                 blocks: ["sweep-3-yawni-primitives"], ...);
}
```

The oracle says `Lifted` but the lift function is unexecutable until Sweep 3
provides an index-aware dispatcher. Any caller that trusts `dispatch_strategy
() == Lifted` to mean "safe to call `lift()` now" will compile clean and
panic at runtime.

**Classification**: REAL GAP. Silent failure vector — no type error, no
compile error. Panics at runtime only.

**Severity**: High. Affects all ArgMax/ArgMin scans.

**Test contract written**: `GAP-LIFT-STUB-1` in `sweep_8_adversarial.rs`
(marked `#[ignore]`, will activate when Sweep 3 lands).

**Fix**: Either (a) `default_strategy()` returns `Sequential` for ArgMax
until Sweep 3 lands, or (b) the `ExecutionStrategy::Lifted` variant carries
a `requires_indexed_lift: bool` flag. Option (a) is conservative and safe.
Option (b) propagates the invariant to callers.

---

## Candidate 2 — Welford + Prefix → Lifted (bit-exact claim)

**Decision tree path**: `is_commutative() = true` → `Lifted`.

**Algebraic correctness**: CORRECT. Chan-Welford parallel merge is
associative AND commutative (proven by tests, declared in
`canonical_structure`). Sequential and tree reductions compute the same
mathematical result.

**Bit-exact claim**: OVERCLAIMS. `jit_op.rs` line 127 states "The backend
ensures bit-exact agreement on adversarial inputs." Chan-Welford is
algebraically associative but NOT bit-exact associative in floating point.
For pathological inputs (e.g., `[1.0, 1e15, 1.0]` — catastrophic
cancellation regime), sequential and tree reductions produce different bit
patterns:

- Sequential: delta terms accumulate in different order
- Tree: `combine(combine(lift(1.0), lift(1e15)), lift(1.0))` vs
  `combine(lift(1.0), combine(lift(1e15), lift(1.0)))`

The intermediate `delta * delta * n_a * n_b / n_total` terms differ due to
floating-point non-associativity of multiplication and addition.

**Classification**: REAL GAP. The claim at line 127 is wrong for float-
state Ops. It is only correct for integer-state Ops (e.g., `Count`).
Wrong claims in docs create false confidence and wrong user expectations about
cross-backend reproducibility (CPU vs GPU may differ; this is acceptable only
if documented).

**Severity**: Medium. Wrong documentation produces wrong user contracts.
Whether this causes actual silent wrong answers depends on whether consumers
rely on bit-exactness (e.g., BLAKE3 cache keying on output values, which
would hash different bit patterns from different backends).

**Test contract written**: `GAP-BIT-EXACT-1` in `sweep_8_adversarial.rs`
(marked `#[ignore]`, will activate when Welford JIT lands; test itself
demonstrates the bit-pattern difference).

**Fix**: Narrow line 127 to: "For Ops with integer-only state (Count, Add-
over-integers), bit-exact is guaranteed. For float-state Ops (Welford,
LogSumExp, AffineCompose), the backend guarantees mathematical equivalence,
not bit-identical results." Same narrowing applies to LogSumExp.

---

## Candidate 3 — LogSumExp + Windowed → Lifted

**Decision tree path**: `is_commutative() = true` → `Lifted`.

**Algebraic correctness**: CORRECT. LSE merge is commutative and
associative. Sequential and tree reductions produce the same mathematical
value. Traced through `[1.0, 100.0, 1.0, 1.0]`: both sequential and lifted
tree give `100 + ln(1.0 + 3*exp(-99)) ≈ 100.0`.

**Bit-exact claim**: Same issue as Welford — float-state, not bit-exact
across tree shapes. The `exp()` and `ln()` operations involved in the merge
accumulate rounding differently depending on which pairs are merged first.

**Classification**: SURVIVES algebraically. Same bit-exact overclaim as
Candidate 2. No separate test needed — covered by GAP-BIT-EXACT-1's
narrowing of line 127.

---

## Candidate 4 — DotProduct (tiled, centered) + Tiled → Lifted

**Decision tree path**: `is_commutative() = true` (declared in
`canonical_structure`) → `Lifted`.

**Algebraic correctness**: CORRECT for the algebra. State = `(sum_xy,
sum_x, sum_y, n)`. Merge of two tiles: sum fields add, n adds. Commutative
and associative. Tile order is genuinely independent for the centering
formula `cov = sum_xy/n - (sum_x/n)(sum_y/n)` applied at extract time.

**Runtime executability**: `DotProduct::lift()` is STUBBED, same structural
issue as ArgMax. The `lift()` stub panics:

```rust
fn lift(&self, _x: f64) -> Self::State {
    crate::stub!(name: "dot_product_lift_needs_pair", ...);
}
```

DotProduct lift requires a `(x_i, y_i)` pair, not a scalar. Same gap as
ArgMax — oracle says `Lifted` but lift is unexecutable.

**Classification**: Same as ArgMax. REAL GAP for runtime executability, not
for algebra. Already noted in the code; no separate test contract written
here (pattern is the same as GAP-LIFT-STUB-1 — the fix is the same: either
don't return `Lifted` until Sweep 3 lands pair-aware lift, or carry a
`requires_pair_lift` flag).

---

## Candidate 5 — AffineCompose + Segmented{bounds} → Lifted

**Decision tree path**: `is_commutative() = false`. `grouping.preserves_order
() = true` for `Segmented` → `Lifted` (non-commutative, order-preserving
path).

**Algebraic correctness**: CORRECT, conditionally. Parallel prefix within
each segment is exactly AffineCompose+Prefix applied per segment
independently. Segments are isolated — no state crosses a segment boundary.
Within each segment, the parallel prefix tree is left-to-right (order-
preserving), so AffineCompose's non-commutativity is respected.

**Condition that must hold**: the Segmented grouping must guarantee that (a)
segments are truly isolated (no combine across boundaries) and (b) the
parallel prefix within a segment preserves left-to-right order. This is an
implementation invariant in the codegen, not an algebra question. If the
codegen violates (a) or (b), results would be wrong without any algebra alarm.

**Classification**: SURVIVES at the algebra level. The potential failure is
a codegen invariant — the segment-boundary reset and intra-segment ordering
must be correct in the JIT backend. This is a Sweep 8E/8F concern, not a
`default_strategy()` oracle concern.

No test contract needed here beyond the existing `affine_compose_is_not_
commutative` and `affine_compose_sequential_vs_tree_must_match` tests.

---

## Bonus — ForceLifted on AlgebraBlocks: panic vs warn-and-fall-back

**Current behavior**: `lifted_strategy_or_panic()` panics when
`default_strategy()` would return `Sequential{AlgebraBlocks}`.

**Question from aristotle**: too restrictive for future conjugation patterns?

**Answer**: NO, panic is correct. If a future sweep adds a conjugation
pattern for an Op currently in AlgebraBlocks, that sweep changes
`default_strategy()` to return `LiftedConjugated` for that case — at which
point `lifted_strategy_or_panic` succeeds naturally. The panic protects
against users assuming liftability where none exists today. The design is
self-correcting: adding a conjugation removes the panic path for that Op.

Warn-and-fall-back would be worse: it silently uses the slower sequential
path when the user explicitly requested lifted. The user would observe
wrong performance (not a wrong answer, but a violated expectation) with no
error. Panic is the correct design — it surfaces the gap immediately.

**Classification**: SURVIVES as designed. Panic is the right sentinel.

---

## Summary Table

| Candidate | Oracle Decision | Algebra | Runtime | Gap? | Severity |
|---|---|---|---|---|---|
| ArgMax + Prefix | Lifted | Correct | STUB panics | GAP-LIFT-STUB-1 | High |
| Welford + Prefix | Lifted | Correct | OK (algebraic) | GAP-BIT-EXACT-1 (doc) | Medium |
| LogSumExp + Windowed | Lifted | Correct | OK (algebraic) | Same as above | Medium |
| DotProduct + Tiled | Lifted | Correct | STUB panics | Same pattern as LIFT-STUB-1 | High |
| AffineCompose + Segmented | Lifted | Correct | OK (codegen invariant) | — | Low |
| ForceLifted panic | — | — | Correct design | — | — |

**Two real gaps found**:
1. GAP-LIFT-STUB-1 (ArgMax, DotProduct): oracle says Lifted, lift is
   unexecutable today. Any caller trusting the oracle will get a runtime
   panic with no prior warning.
2. GAP-BIT-EXACT-1 (all float-state Ops): line 127 in jit_op.rs overclaims
   bit-exact; correct claim is mathematical equivalence only.

---

## Convergence Observation

Both gaps trace to the same structural pattern: **a promise made at the type
level that the implementation cannot yet honor**.

- GAP-LIFT-STUB-1: `ExecutionStrategy::Lifted` implies "lift() is callable"
  — it does not. The strategy type is not a readiness certificate.
- GAP-BIT-EXACT-1: doc comment implies "bit-exact cross-reduction-tree" —
  floating-point arithmetic doesn't provide this for non-trivial state types.

The fix for both is the same shape: **narrow the claim to what is actually
guaranteed**. For LIFT-STUB-1, either narrow the strategy return or add a
readiness predicate. For BIT-EXACT-1, narrow the doc comment to
integer-state Ops only.

**Test contracts written**: `sweep_8_adversarial.rs` now contains
`GAP-LIFT-STUB-1` and `GAP-BIT-EXACT-1` as `#[ignore]` tests. Both compile
clean and fail when run with `--include-ignored`. Both will be activated by
pathmaker when the corresponding fixes land.
