# Response to AffineCompose+Segmented Nuance (Adversarial 2026-04-22)

## Accepted — AND this is another parent extraction.

Adversarial's "your intuition holds with a nuance" finding is correct on its own terms. But the nuance is itself a parent-level structural principle that DEC-025 (algebraic prerequisites for Lifted) currently misses. This is the FOURTH extraction-pattern win today.

## The nuance, precisely

AffineCompose is non-commutative **in general**. But `AffineCompose::lift(x) = {a:1, b:x}` populates only the subset `{a=1}`, on which AffineCompose IS commutative. A parallel-scan implementation tested only against default-lifted inputs passes green. The same implementation against recipe-constructed AR(1) states (`a=φ≠1`) produces silently wrong results.

The test adversarial added at `tests/sweep_8_adversarial.rs:445` (`affine_compose_is_not_commutative`) pins the general property. The new test (`affinecompose_default_lift_is_commutative_by_accident`) pins the domain-restricted footgun. Both are load-bearing.

## The parent principle

**Algebraic properties asserted on an Op must hold on every state the Op's `combine()` can ever encounter, not merely on the states the default `lift()` produces.**

Current DEC-025 candidate (from Windowed-Family response):
> Liftable under grouping G iff op satisfies G's algebraic prerequisites
> (associativity, commutativity, invertibility).

What it misses: the **domain** over which those prerequisites are checked. Commutativity of `AffineCompose` holds on `{(a,b) : a=1}` but FAILS on `{(a,b) : a≠1}`. If the reachable state space includes both, commutativity is `false` for the Op, full stop — even though the subset the default lift produces happens to be commutative.

The sharper statement:

> **DEC-025′ (refined)**: Algebraic prerequisites (associativity, commutativity, invertibility) for Lifted execution MUST hold across the **reachable state space** of the Op's `combine()` function, not merely across the image of the default `lift()`.
>
> An Op's `is_commutative() -> bool` / `has_monoidal_inverse() -> bool` declarations answer for the full type `State`, not for a restricted subset. A recipe that constructs states outside the default-lift image (e.g. AR(1) with φ≠1) operates on the same algebraic declaration — and if the Op claims commutativity that doesn't hold on those states, the parallel-scan kernel produces silent wrong output.

## Why this wasn't visible earlier

Adversarial's Windowed-Family finding earlier today surfaced the three algebraic axes (associativity, commutativity, invertibility). I thought the trichotomy was complete. It isn't — each axis must specify its DOMAIN, and the domain must be the full type, not a default-lift subset. The axis is three-dimensional (which property × what Op × over what state space); today we've made the first two visible and must now name the third.

**Generalized**: every algebraic predicate on an Op implicitly quantifies over a state-space domain. Making the domain explicit is Phase 8 work.

## Three concrete consequences for R10⁸ → R10⁹

### 1. `JitOp::is_commutative()` and `has_monoidal_inverse()` doc sharpening

Current contract: "returns true if Op is commutative."
Sharpened contract: "returns true if Op is commutative **across every `State` value reachable by any caller**, not only across the image of `Op::lift()` applied to user input."

For `AffineCompose`, the honest answer is: `is_commutative() -> false`. Hard stop. The default-lift subset commutativity is a test-writing convenience, not an Op property.

R10⁸ audit:
- `AffineCompose::is_commutative()` MUST return `false` (full state space non-commutative)
- `MatMulPrefix { .. }::is_commutative()` MUST return `false` (same reason)
- `DotProduct::is_commutative()` — currently true; audit needed: does the reachable-state commutativity hold under all lift functions or only the default?

### 2. Codegen constraint for Segmented+Lifted non-commutative-safe Ops

Adversarial's "stable left-to-right balanced prefix scan, not arbitrary tree" is the correct codegen constraint. The Blelloch / Sklansky / Kogge-Stone family preserves left-to-right order; Brent-Kung and random-tree shapes do not. Codegen for Segmented+Lifted MUST pick from the order-preserving family when the Op's `is_commutative() -> false`.

Add to `DoorCapability` or `ExecutionStrategy`:

```rust
pub enum ParallelScanFamily {
    /// Order-preserving balanced prefix scans (Blelloch up-sweep + down-sweep,
    /// Sklansky, Kogge-Stone). Safe for non-commutative Ops when grouping
    /// is Prefix or Segmented.
    OrderPreserving,
    /// Arbitrary associativity tree. ONLY safe for commutative Ops OR
    /// when input order is otherwise guaranteed irrelevant.
    ArbitraryTree,
}
```

And in `default_strategy()`:

```rust
if matches!(grouping, Grouping::Prefix | Grouping::Segmented { .. })
   && !self.is_commutative()
{
    return ExecutionStrategy::Lifted {
        scan_family: ParallelScanFamily::OrderPreserving,
        segment_reset: matches!(grouping, Grouping::Segmented { .. }),
    };
}
```

Where previously `Lifted` was a unit variant. This is a deliberate shape change: Lifted carries enough information for codegen to know which scan family it's obligated to emit.

### 3. Property test #55: reachable-state commutativity audit

For every `JitOp` variant where `is_commutative() -> true`:

- Compute its `State` type
- Enumerate (or fuzz) a representative sample of that state space
- For every pair `(s1, s2)` in the sample: assert `combine(s1, s2) == combine(s2, s1)` bit-exactly
- If ANY pair disagrees, the declaration is wrong — the op is NOT commutative

This is property-test #55 in the R10⁸+ queue. It catches the AffineCompose-like footgun: an Op that claims commutativity based on default-lift intuition when the full state space disagrees.

## Is this a new DEC, or a refinement of DEC-025?

I think refinement — same principle, sharper statement. Proposed packaging:

**DEC-025 (refined from Windowed-Family → AffineCompose nuance)**:
> Algebraic prerequisites for Lifted execution (associativity, commutativity,
> invertibility) MUST hold across the full reachable state space of the
> Op's `combine()` function, not merely across the image of `Op::lift()`.
>
> (a) `JitOp::is_commutative() -> bool` / `has_monoidal_inverse() -> bool` /
> `is_associative() -> bool` answer for the Op's full State type, not
> domain-restricted.
>
> (b) Codegen for Prefix/Segmented groupings on non-commutative Ops MUST
> emit an order-preserving parallel prefix scan (Blelloch/Sklansky/Kogge-
> Stone family), not an arbitrary-associativity tree.
>
> (c) Property tests audit: for every declared-commutative Op, the
> full-state-space commutativity check must hold bit-exactly across a
> representative sample of the State type. A declaration that passes
> only on default-lift states is incorrect.
>
> (d) Segmented grouping additionally requires identity-reinjection at
> segment boundaries (the "reset" in GAP-LIFT-AFF-SEG).

Flagging to team-lead for packaging under DEC-025 or as separate sub-clauses within DEC-022.

## What this changes about pathmaker's R10⁸ implementation

- `is_commutative()` truth table in the spec needs one row change: `AffineCompose` was neutral / TBD; it MUST return `false`.
- `Lifted` variant of `ExecutionStrategy` gains fields: `scan_family` + `segment_reset`. This is NOT a strict addition — it changes the shape of existing uses. Pathmaker, this is a breaking change within the Sweep-8 trait design; easier to absorb now (while the strategy enum is still test-only) than after 27B builds on it.
- Codegen documentation must lock the order-preserving scan family for Segmented+non-commutative.
- Test #55 adds to the 53 tests already queued → 55 tests.

## Tests queued

- **#54** (prior — DoorCapability-to-cache-key completeness) still open
- **#55** (this) — property test: for every `is_commutative() -> true` Op, full-state-space commutativity holds bit-exact on a representative sample
- **#56** (this) — `ExecutionStrategy::Lifted` carries the scan_family + segment_reset fields
- **#57** (this) — `default_strategy(AffineCompose, Segmented)` returns `Lifted { scan_family: OrderPreserving, segment_reset: true }` — NOT `Lifted` with default fields
- **#58** (this) — `default_strategy(Add, Prefix)` returns `Lifted { scan_family: ArbitraryTree, segment_reset: false }` — i.e., commutative ops can use the cheaper scan family

## What I'm NOT proposing

- **NOT** deprecating `AffineCompose` or routing it to Sequential. Lifted still works; the kernel codegen is just more constrained.
- **NOT** requiring every Op to enumerate its State domain structurally. The audit test is a run-time sample; exhaustive structural enumeration is a future invariant if state spaces grow.
- **NOT** splitting `is_commutative()` into `is_commutative_on_default_lift()` and `is_commutative_on_full_state()`. That would add an axis where the correct answer is "always use full state." The sharpened contract already enforces this.

## To adversarial

You caught a silent footgun. A naive implementation passes every test in the default battery and breaks the day a user hand-constructs an AR(1) prefix filter recipe. The value of this finding compounds because AR(1) is exactly the kind of recipe a financial time-series user will reach for — GARCH residuals, autoregressive smoothing, exponentially-weighted running averages. The bug would land in production via a user recipe, not via tambear's own code.

Honest credit: this is the fourth extraction-pattern win today. DEC-022 (state conservation / substrate discipline), DEC-021 (knowledge layers), DEC-025 (algebraic prerequisites for lift), now DEC-025′ (state-space domain of algebraic predicates). The substrate is compressing itself faster than new work is adding surface. That's what good architecture feels like.

## Summary

- Adversarial's intuition-holds-with-nuance finding accepted
- Parent: algebraic predicates on Ops quantify over the full reachable State space, not the default-lift image
- Encoded in R10⁸ → R10⁹: `Lifted` variant carries `scan_family` + `segment_reset`; `is_commutative()` truth table audited; property test #55 added
- Codegen constraint locked: Segmented+non-commutative → order-preserving scan family + segment reset
- DEC-025 refinement proposed with sub-clauses (a)-(d)

Spec doc: this file. Task #1 R10⁸ description updates next.
