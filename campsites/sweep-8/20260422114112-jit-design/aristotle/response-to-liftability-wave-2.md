# Response — Adversarial 5+1 Liftability Wave

**Sweep 8 / Task 8A (still reopened)** · Author: aristotle · Date: 2026-04-22

Adversarial ran the five candidate attacks + one bonus I'd assigned after
the liftability-default addendum. Result: **two real gaps, three
survivors, one confirmed design.** Both gaps accept. Both fit the same
structural pattern adversarial identified in their wave-8 Sweep 27
writeup: **a promise made at the type or doc level that the
implementation cannot yet honor.**

Full adversarial writeup at `adversarial/20260422-aristotle-5plus1-
liftability-attacks.md`.

---

## GAP-LIFT-STUB-1 — Oracle says Lifted; lift() panics

**Verdict: ACCEPT. Real gap. High severity.**

Adversarial is right. `default_strategy()` returns `Lifted` for ArgMax
+ Prefix and for DotProduct + Tiled. Both Ops' `lift()` methods are
stubbed — they panic at runtime with `crate::stub!` calls. The oracle
says the user can lift; the substrate disagrees; the failure mode is
a runtime panic with no type-level warning.

This is the same structural class as the Sweep 27 scan_nonfinite bug:
**documentation says one thing, type system doesn't enforce it.** The
`ExecutionStrategy::Lifted` variant implies "you may call lift() now,"
and the Op's `lift()` method accepts the call — then panics.

### Adversarial's two options, neither of which I accept

Adversarial proposed:

- **(a)** `default_strategy()` returns `Sequential` for these Ops until
  Sweep 3 lands.
- **(b)** `ExecutionStrategy::Lifted` carries a `requires_indexed_lift`
  / `requires_pair_lift` flag.

I reject both. Reasoning:

- **(a) conflates two orthogonal concerns.** `Sequential` per DEC-021
  means "algebra doesn't admit lifting, OR user override, OR Fock
  boundary." It does NOT mean "algebra admits it but the dispatcher
  stub isn't ready." Returning `Sequential` for ArgMax would lie about
  the algebraic property (ArgMax + Prefix IS liftable; the issue is
  implementation readiness). When Sweep 3 lands, we'd have to
  retroactively change `default_strategy()`'s output for existing
  ArgMax recipes — their cache keys would shift from
  `Sequential{AlgebraBlocks}` to `Lifted`, invalidating warm caches
  for no algebraic reason.

- **(b) muddies the strategy enum with readiness state.** A `Lifted
  { requires_indexed_lift }` variant means every caller has to check
  the flag before they call lift(). That's discipline at the call
  site. Same failure mode we're trying to prevent — the documentation
  says "check the flag" and the type system doesn't enforce it.

### The Aristotelian move

Add a **readiness predicate on JitOp itself**, separate from the
algebra declaration, and add a new SequentialReason variant naming
the readiness gap explicitly:

```rust
impl JitOp {
    /// Whether this Op's lift() is currently executable. Returns
    /// false for Ops whose lift is stubbed pending future sweep work
    /// (currently: ArgMax, ArgMin, DotProduct, Distance — pending
    /// Sweep 3's YAWNI primitive dispatcher).
    ///
    /// When this returns false, `default_strategy()` short-circuits
    /// to `Sequential{NotYetImplemented{unblocker}}` regardless of
    /// algebra. When the unblocker sweep lands and the stub is
    /// removed, this returns true and `default_strategy()` naturally
    /// returns Lifted per the algebra.
    pub fn is_lift_ready(&self) -> Option<&'static str> {
        match self {
            JitOp::ArgMax | JitOp::ArgMin
                => Some("sweep-3-yawni-primitives"),
            JitOp::DotProduct | JitOp::Distance
                => Some("sweep-3-yawni-primitives"),
            JitOp::MatMulPrefix { .. }
                => Some("sweep-3-yawni-primitives"),
            _ => None,  // ready
        }
    }
}

pub enum SequentialReason {
    UserOverride,
    AlgebraBlocks,
    FockBoundary,
    /// New: algebra admits lifting, but the dispatcher isn't ready
    /// yet for this Op. Temporary — auto-cleared when the unblocker
    /// sweep lands and the relevant lift() stub is replaced with
    /// real codegen.
    NotYetImplemented { unblocker: &'static str },
}

// in default_strategy:
pub fn default_strategy(&self, shape: &Shape) -> ExecutionStrategy {
    if let Some(unblocker) = self.is_lift_ready() {
        return ExecutionStrategy::Sequential {
            reason: SequentialReason::NotYetImplemented {
                unblocker,
            },
        };
    }
    if !self.is_associative() {
        return ExecutionStrategy::Sequential {
            reason: SequentialReason::AlgebraBlocks,
        };
    }
    if self.is_commutative() || shape.grouping.preserves_order() {
        return ExecutionStrategy::Lifted;
    }
    if let Some(perm_kind) = conjugation_perm_for(self, shape) {
        return ExecutionStrategy::LiftedConjugated { perm_kind };
    }
    ExecutionStrategy::Sequential {
        reason: SequentialReason::AlgebraBlocks,
    }
}
```

**Why this is better than (a) or (b):**

- Keeps the three DEC-021-aligned sequential reasons (UserOverride,
  AlgebraBlocks, FockBoundary) algebraically meaningful. Adds a
  fourth for implementation-state, which is a different axis.
- Self-correcting: when Sweep 3 replaces the `stub!` with real lift
  code, `is_lift_ready()` for those variants returns None, and
  `default_strategy()` naturally returns Lifted. No retroactive
  changes to `default_strategy`'s algorithm.
- The `unblocker: &'static str` in `NotYetImplemented` names the
  specific sweep that closes the gap — user / debug / writeup can
  read "sequential because sweep-3-yawni-primitives pending" with
  no ambiguity.
- Cache-key discipline: `Sequential{NotYetImplemented{...}}` is a
  distinct cache-key variant (per the `tag()` serialization already
  in strategy.rs). When Sweep 3 lands, the user's pipeline's
  `default_strategy()` output for ArgMax recipes changes from
  `Sequential{NotYetImplemented}` to `Lifted`, which IS a different
  cache key — correct behavior (new lift kernel needs compile).
- State conservation per DEC-020: the "why sequential" reason is
  structurally exposed; downstream IDE / debug / writeup can all
  see "because Sweep 3 hasn't shipped" rather than "because
  algebra blocks" (which would be a lie).
- Phase-8 forced rejection: what if we just panicked more loudly
  in `lift()`? Same failure class — fails late, not early. The
  oracle's job is to surface readiness before the user tries to
  use the strategy.

This pattern also helps with the SumStrategy::Exact / Kulisch-
pending story from GAP-LIFT-1 (the "bit-exact Add" deconstruction
earlier today): when Sweep 3 delivers Kulisch, the same
`NotYetImplemented { unblocker: "sweep-3-yawni-primitives" }` variant
can close that gap symmetrically.

---

## GAP-BIT-EXACT-1 — `jit_op.rs` line 127 overclaims bit-exactness

**Verdict: ACCEPT. Doc gap plus a structural addition.**

Adversarial is right that `jit_op.rs` line 127 ("The backend ensures
bit-exact agreement on adversarial inputs") is wrong for float-state
Ops. Chan-Welford merge is algebraically associative but not
floating-point associative; sequential vs tree reduction produces
different bit patterns on pathological inputs.

### But the doc-narrowing isn't enough

Adversarial's proposed narrowing — "For integer-state Ops bit-exact
is guaranteed; for float-state Ops, mathematical equivalence only" —
is doc discipline. It's the right narrowing in content but wrong
locus per the same argument I made on the Sweep 27 DataProfile:
**type-system enforcement beats documentation discipline.**

Good news: the structural hook already exists.
`CompiledArtifact::determinism: DeterminismClass` with variants
`Deterministic`, `OrderDependent`, `NonDeterministic`,
`SeededDeterministic` (the latter added per wave-2 A2 accepted).
Today's `Deterministic` variant is ambiguous — it could mean
"bit-exact across any valid reduction tree" OR "mathematically
equivalent across any valid reduction tree." Those are different
guarantees.

### The Aristotelian move — split `Deterministic` into two variants

```rust
pub enum DeterminismClass {
    /// Bit-exact across ALL valid reduction strategies (tree,
    /// sequential, prefix-scan, tiled). Output bytes identical
    /// regardless of dispatch shape. Achievable only for integer-
    /// state Ops (Count, Add-over-integers) or float-state Ops
    /// explicitly using a bit-exact primitive (Kulisch-backed
    /// SumStrategy::Exact).
    BitExact,
    /// Mathematically equivalent across all valid reduction
    /// strategies (same algebraic result). Output BITS may differ
    /// between sequential and tree reductions for float-state Ops
    /// due to IEEE 754 non-associativity. This is the default for
    /// float-state Ops without Kulisch-backed summation.
    ///
    /// Cross-backend comparison SHOULD NOT expect byte equality
    /// under this class; SHOULD expect mathematical equivalence
    /// within floating-point precision.
    MathematicallyEquivalent,
    /// Same multiset of inputs gives the same output bit pattern,
    /// but order-of-arrival within a group affects output
    /// (ArgMax with lowest-index tiebreak). Within a fixed
    /// input order, bit-exact.
    OrderDependent,
    /// Output may differ across runs for the same input (GPU warp
    /// scheduling, CUDA atomics without deterministic mode).
    NonDeterministic,
    /// Identical output given the same RNG seed; different seeds
    /// produce different outputs. Seed is baked at compile time
    /// via `using(seed=N)`.
    SeededDeterministic { seed_hash: [u8; 32] },
}
```

### Op-level classification (for codegen to read)

```rust
impl JitOp {
    /// The weakest determinism guarantee this Op's canonical
    /// SumStrategy::Exact path provides. Codegen may upgrade (e.g.
    /// Welford with explicit Kulisch m_2 accumulation → BitExact)
    /// but may not downgrade.
    pub fn determinism_floor(&self, sum_strategy: SumStrategy)
        -> DeterminismClass
    {
        use SumStrategy::*;
        use DeterminismClass::*;
        match (self, sum_strategy) {
            // Lowest-index tiebreak is order-dependent regardless
            // of sum strategy.
            (JitOp::ArgMax, _) | (JitOp::ArgMin, _)
                => OrderDependent,
            // Float-state Ops with non-Exact sum strategy can only
            // promise math equivalence.
            (JitOp::Welford | JitOp::LogSumExp
             | JitOp::AffineCompose | JitOp::DotProduct
             | JitOp::Distance, Fast | Nondet)
                => match sum_strategy {
                    Nondet => NonDeterministic,
                    _ => MathematicallyEquivalent,
                },
            // Float-state Ops with Exact sum strategy are
            // bit-exact IF the Exact path uses Kulisch or
            // compensated summation (Sweep 3+).
            (JitOp::Welford | JitOp::LogSumExp
             | JitOp::AffineCompose | JitOp::DotProduct
             | JitOp::Distance, Exact) => BitExact,
            // Add itself: bit-exact with Exact, math-equivalent
            // otherwise.
            (JitOp::Add, Exact) => BitExact,
            (JitOp::Add, Fast) => MathematicallyEquivalent,
            (JitOp::Add, Nondet) => NonDeterministic,
            // Max/Min/lattice ops: bit-exact always (idempotent +
            // no float arithmetic).
            (JitOp::Max | JitOp::Min, _) => BitExact,
            // Scan semirings: depends on the semiring.
            (JitOp::Scan(SemiringKind::Boolean), _) => BitExact,
            (JitOp::Scan(_), Exact) => BitExact,
            (JitOp::Scan(_), Fast) => MathematicallyEquivalent,
            (JitOp::Scan(_), Nondet) => NonDeterministic,
            // MatMulPrefix: same float-state story.
            (JitOp::MatMulPrefix { .. }, Exact) => BitExact,
            (JitOp::MatMulPrefix { .. }, Fast) => MathematicallyEquivalent,
            (JitOp::MatMulPrefix { .. }, Nondet) => NonDeterministic,
        }
    }
}
```

### Doc update for `jit_op.rs` line 127

Change:
> "The backend ensures bit-exact agreement on adversarial inputs."

To:
> "The backend's determinism guarantee is declared per-(Op,
> SumStrategy) via `determinism_floor()` and carried on
> `CompiledArtifact::determinism`. For integer-state Ops and
> for float-state Ops with `SumStrategy::Exact`, the guarantee is
> `DeterminismClass::BitExact` (byte-identical across all valid
> reduction trees). For float-state Ops with `SumStrategy::Fast`,
> the guarantee is `MathematicallyEquivalent` only (algebraically
> identical; bit patterns may differ due to IEEE 754 non-
> associativity). Callers that require byte-equality MUST check the
> declared class on the artifact, not assume from the Op name."

### Why this is the right fix

- Downstream code that cares about byte-equality (cross-backend
  parity tests, cache-key-over-outputs, output hashing) can match
  on `DeterminismClass::BitExact` and know the guarantee.
- Downstream code that doesn't care (UI display, most statistical
  workflows) sees `MathematicallyEquivalent` and proceeds.
- The user's `using(sum_strategy="fast")` opt-in from GAP-LIFT-1
  naturally downgrades the class from BitExact to
  MathematicallyEquivalent — consistent per-axis.
- State conservation: the guarantee is structurally visible, not
  buried in docs.
- No more "the backend ensures X" claims that the backend can't
  actually ensure.

---

## Convergence observation — the pattern adversarial named

Both gaps are **the same structural class** adversarial identified in
their Sweep 27 writeup:

> *A promise made at the type or doc level that the implementation
> cannot yet honor.*

- **GAP-LIFT-STUB-1**: `ExecutionStrategy::Lifted` says "lift is
  callable" → implementation stub panics.
- **GAP-BIT-EXACT-1**: doc says "bit-exact" → floating-point arithmetic
  doesn't deliver it.
- **GAP-27-1 (scan_nonfinite)**: `has_known_non_finite: false` says
  "no NaN present" → sample scan can't verify the claim.

In all three, the fix shape is the same:

1. **Narrow the claim** to what's actually guaranteed.
2. **Carry the narrower claim in the TYPE SYSTEM**, not just the
   docs. Make violations code bugs, not discipline lapses.
3. **Tag the claim with the reason the guarantee is what it is** —
   so downstream code can reason about it without having to reverse-
   engineer from the Op / value / implementation.

This is now a first-class pattern I want named in the architecture
docs. Team-lead's call whether it lands as a note on LIVE_COMPILER
(section on "claims and their enforcement"), as an addition to DEC-
020 (state conservation extended to claim-quality), or as its own
ADR. The observation carries enough weight that I'll raise it
separately.

---

## Sweep 8 delta additions to R10″

This response adds to the R10′ → R10″ trait delta list (from wave-2):

**New (from this wave):**

- **`JitOp::is_lift_ready(&self) -> Option<&'static str>`** method
  returning the unblocker-sweep name when the Op's lift is stubbed
- **`SequentialReason::NotYetImplemented { unblocker: &'static str }`**
  new variant
- **`DeterminismClass::BitExact`** renamed from `Deterministic`;
  semantics tightened to "byte-identical across all valid reductions"
- **`DeterminismClass::MathematicallyEquivalent`** new variant for
  float-state Ops with non-Exact sum strategy
- **`JitOp::determinism_floor(&self, sum_strategy: SumStrategy) ->
  DeterminismClass`** method that codegen reads
- **`jit_op.rs` line 127 doc correction** replacing the bit-exact
  overclaim with the per-(Op, SumStrategy) formulation

**Cache-key impact:** the renamed `BitExact` variant changes the
`tag()` string from "det" → "bit_exact" (or preserves the old tag
and adds a new one for MathematicallyEquivalent). **Recommend
preserving "det" for BitExact** and adding "math_eq" for the new
variant; minimizes cache invalidation scope. Still bumps IR_VERSION
(I'd say 3 → 4; or 2 → 4 if not yet bumped).

**Cache-key discipline check:** pathmaker's `tag()` serialization
must include every new variant uniquely. The `strategy_tags_unique`
test in strategy.rs catches this at CI time.

---

## New tests for the convergence-check

Adding to the existing 12 tests queued from wave-2:

13. `argmax_default_strategy_is_sequential_not_yet_implemented`
14. `dotproduct_default_strategy_is_sequential_not_yet_implemented`
15. `matmul_prefix_default_strategy_is_sequential_not_yet_implemented`
16. `is_lift_ready_returns_none_for_ready_ops` (Add/Max/Min/
    Welford/LSE/AffineCompose/Scan)
17. `not_yet_implemented_tag_distinct_in_cache_key`
18. `determinism_floor_bit_exact_for_integer_ops` (Max/Min/
    BooleanSemiring)
19. `determinism_floor_math_equivalent_for_float_fast` (Welford/
    LSE/AffineCompose/Add with Fast)
20. `determinism_floor_bit_exact_for_float_exact_path` (same Ops
    with Exact — stubbed until Kulisch lands; `#[ignore]` until
    then)
21. `determinism_floor_order_dependent_for_argmax_argmin`
22. `determinism_floor_nondet_for_nondet_sum_strategy`

Total test addition: 22 new tests targeting wave-2 + wave-3 gaps.

---

## Asks

**For pathmaker (when you implement R10″):**
- Accept `is_lift_ready()` + `NotYetImplemented` variant +
  `determinism_floor()` + `BitExact`/`MathematicallyEquivalent`
  split? All additive; no redesign of landed code.
- IR_VERSION bump 2 → 4 captures wave-2 + wave-3 in one cache
  invalidation (cheaper than two bumps).

**For adversarial:**
- New standing attack: can you construct a (JitOp, SumStrategy,
  Shape) tuple where my `determinism_floor` classification lies?
  E.g., Welford + Fast that I classify `MathematicallyEquivalent`
  but that actually produces results differing by more than a few
  ULPs under specific reduction trees (algebraically non-equivalent,
  not just bit-non-identical)?
- The convergence observation ("narrow the claim + lift it into
  the type system") is now a pattern. Can you construct a fourth
  gap in the substrate that fits it — somewhere the docs make a
  claim the types don't enforce?

**For team-lead:**
- The "claims and their enforcement" pattern adversarial named
  and this deconstruction confirmed — does it warrant its own ADR?
  DEC-020 covers state conservation generally; this pattern is
  state-conservation applied specifically to claim-quality (every
  claim has a confidence level; the confidence level must be
  type-enforced, not doc-discipline). Candidate DEC-022. Your
  call.

Standing by for pathmaker's R10″ implementation; both gaps
folded into the convergence-check criteria.
