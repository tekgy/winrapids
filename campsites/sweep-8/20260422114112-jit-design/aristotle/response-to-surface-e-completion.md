# Response — Surface E (default_strategy() Oracle Correctness)

**Sweep 8 / Task 8A (still reopened)** · Author: aristotle · Date: 2026-04-22

Adversarial completed Surface E with three more bug contracts:
GAP-LIFT-LSE-WIN, GAP-LIFT-DOT-STUB, GAP-LIFT-AFF-SEG. All three accept,
all three classify cleanly, and together they reveal a structural gap
in `default_strategy()` itself: the oracle checks Op algebra ∧ Grouping
order-preservation but **doesn't check (Op × Grouping) codegen-readiness
or Grouping-specific semantic constraints**.

---

## Per-finding classification

### GAP-LIFT-DOT-STUB — already covered by R10⁶'s `is_lift_ready()`

**Verdict: ACCEPT, fix already in queue (R10⁶ wave-3).**

Identical structural pattern to GAP-LIFT-STUB-1 (ArgMax). Wave-3
landed `JitOp::is_lift_ready() -> Option<&'static str>` returning
`Some("sweep-3-yawni-primitives")` for ArgMax/ArgMin/DotProduct/
Distance/MatMulPrefix. `default_strategy()` short-circuits to
`Sequential{NotYetImplemented{unblocker}}` for those Ops.

Per the R10⁶ design (in task #1 description), DotProduct is
explicitly enumerated in the is_lift_ready set. When pathmaker
implements R10⁶, GAP-LIFT-DOT-STUB activates correctly:
`default_strategy(DotProduct, _)` returns
`Sequential{NotYetImplemented{unblocker: "sweep-3-yawni-primitives"}}`.

Adversarial's test will start passing as soon as R10⁶ lands. No
additional design change.

### GAP-LIFT-LSE-WIN — Windowed grouping needs special codegen

**Verdict: ACCEPT — REVEALS a structural gap in `default_strategy()`.**

This is the interesting one. LSE is associative AND commutative
(both true), Windowed `preserves_order()` returns true (also true
per its semantic), so per the current decision tree:

```
if !is_associative()                                   -> Sequential{AlgebraBlocks}
if is_commutative() OR grouping.preserves_order()      -> Lifted   ← LSE+Windowed hits here
if conjugation_perm_for(...).is_some()                 -> LiftedConjugated
else                                                   -> Sequential{AlgebraBlocks}
```

But "naive parallel reduction" treating Windowed as flat is wrong:
it produces global LSE across the whole input, not per-window. The
correct lift for Windowed reductions requires **prefix-subtraction**
(compute prefix sums; window[i] = prefix[i+w] − prefix[i]). For
LSE specifically, prefix-subtraction needs the Op to be
**invertible** under combine — `inverse(combine(a, b), b) == a`.
LSE has an inverse (`logsumexp(a, b) ⊟ b = log(exp(combined) − exp(b))`)
but it requires careful numerical handling.

**For Sweep 8, this is too sophisticated.** The clean fix is:
`default_strategy()` returns `Sequential{NotYetImplemented{
unblocker: "sweep-9-windowed-prefix-subtraction"}}` for any
(Op, Windowed) where Op doesn't implement an inverse-aware
windowed-lift codegen.

### GAP-LIFT-AFF-SEG — Segmented grouping needs reset-aware codegen

**Verdict: ACCEPT — same shape as GAP-LIFT-LSE-WIN.**

AffineCompose is associative (true), Segmented `preserves_order()`
returns true (also true — order preserved within each segment).
`default_strategy()` returns `Lifted`. But naive parallel prefix
ignores segment boundaries; output `[2.0, 5.0, 9.0]` instead of
correct `[2.0, 5.0, 4.0]`. **9.0 looks plausible**, hence silent
wrong.

Correct lift for Segmented prefix-scan exists (segmented prefix
scan is a well-known parallel primitive, Hillis-Steele or Brent-
Kung with segment-mask propagation). But the codegen is
specialized; it's not the same kernel as flat Prefix.

Same fix shape: `default_strategy()` returns
`Sequential{NotYetImplemented{unblocker: "sweep-9-segmented-prefix-scan"}}`
for any (Op, Segmented) where Op doesn't have a segment-aware
windowed-lift codegen.

---

## The structural observation — what `default_strategy()` is missing

GAP-LIFT-LSE-WIN and GAP-LIFT-AFF-SEG both fall into the pattern:

> The Op's algebra admits lifting in the abstract (associative,
> potentially commutative). The Grouping's order-preservation
> property is true. But the SPECIFIC (Op × Grouping) codegen
> requires a non-trivial kernel that doesn't exist yet (windowed
> prefix-subtraction; segmented prefix scan). A naive flat-prefix
> codegen produces wrong results.

My current `default_strategy()` checks two axes:
1. Op algebra (associative + commutative or not)
2. Grouping order-preservation

What's MISSING: a third check — **does an (Op, Grouping)-specific
codegen exist?** This is conceptually like `is_lift_ready()` but
parameterized on the Grouping too, not just the Op.

### The fix — extend `is_lift_ready()` to take Grouping

```rust
impl JitOp {
    /// Whether this Op's lift is currently executable AGAINST THE
    /// SPECIFIC GROUPING. Returns the unblocker-sweep name when
    /// the (Op, Grouping) codegen is pending.
    ///
    /// Examples:
    /// - (ArgMax, _) → Some("sweep-3-yawni-primitives") (lift stub)
    /// - (LSE, Windowed) → Some("sweep-9-windowed-prefix-subtraction")
    ///                     (LSE is associative but Windowed lift requires
    ///                     prefix-subtraction codegen that doesn't exist)
    /// - (AffineCompose, Segmented) → Some("sweep-9-segmented-prefix-scan")
    /// - (Add, All) → None (ready: simple tree reduction)
    /// - (Add, Prefix) → None (ready: parallel prefix scan)
    pub fn is_lift_ready_for(&self, grouping: &Grouping)
        -> Option<&'static str>
    {
        // Op-level readiness check first (covers DotProduct + others)
        if let Some(u) = self.op_lift_stub_unblocker() {
            return Some(u);
        }

        // (Op, Grouping)-pair check
        match (self, grouping) {
            // Windowed reductions need prefix-subtraction codegen.
            // Only Add (with Kahan-compensated subtraction) trivially
            // works; everything else needs Sweep 9.
            (JitOp::LogSumExp, Grouping::Windowed { .. })
            | (JitOp::Welford, Grouping::Windowed { .. })
            | (JitOp::DotProduct, Grouping::Windowed { .. })
            | (JitOp::Distance, Grouping::Windowed { .. })
                => Some("sweep-9-windowed-prefix-subtraction"),

            // Segmented scans need reset-aware prefix scan codegen.
            (JitOp::AffineCompose, Grouping::Segmented { .. })
            | (JitOp::MatMulPrefix { .. }, Grouping::Segmented { .. })
                => Some("sweep-9-segmented-prefix-scan"),

            // Note: Add and lattice ops (Max/Min) work over Windowed
            // and Segmented with simple codegen extensions; those land
            // in Sweep 8C/8D as part of the basic codegen surface.
            // ArgMax + Segmented also reasonable but ArgMax already
            // blocked at op-level by lift stub.

            _ => None,
        }
    }

    fn op_lift_stub_unblocker(&self) -> Option<&'static str> {
        match self {
            JitOp::ArgMax | JitOp::ArgMin
            | JitOp::DotProduct | JitOp::Distance
            | JitOp::MatMulPrefix { .. }
                => Some("sweep-3-yawni-primitives"),
            _ => None,
        }
    }
}

// Updated default_strategy:
pub fn default_strategy(&self, shape: &Shape) -> ExecutionStrategy {
    // Check (Op, Grouping)-pair readiness before algebra.
    if let Some(unblocker) = self.is_lift_ready_for(&shape.grouping) {
        return ExecutionStrategy::Sequential {
            reason: SequentialReason::NotYetImplemented { unblocker },
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

**Key change:** `is_lift_ready_for(grouping)` (replacing the
wave-3 `is_lift_ready()`) takes the Grouping as a parameter and
checks (Op, Grouping)-pair readiness, not just Op-level.

This is **structurally cleaner** than wave-3's Op-level check
because:
- It expresses readiness as a property of the (Op, Grouping)
  combination — which is what codegen actually compiles
- It captures both (a) Op-level lift stubs (DotProduct/ArgMax/etc.)
  and (b) Grouping-specific codegen gaps (LSE+Windowed,
  AffineCompose+Segmented) in one method
- New (Op, Grouping) pairs joining the substrate add one match arm
  with their unblocker; nothing else changes
- Self-correcting: when Sweep 9 lands windowed-prefix-subtraction,
  remove the LSE/Welford/etc. + Windowed entries from the match;
  default_strategy naturally returns Lifted again

### Why wave-3's `is_lift_ready()` (Op-only) was insufficient

I named the wave-3 method `is_lift_ready()` thinking "the Op is
ready or not." But Surface E proves readiness is a function of
(Op, Grouping), not just Op alone:

- Add + Prefix: ready
- Add + All: ready  
- LSE + Prefix: ready (parallel prefix scan, well-known)
- LSE + All: ready (tree reduction over LSE merge)
- LSE + Windowed: NOT ready (needs prefix-subtraction)

Same Op (LSE), different grouping → different readiness. The
wave-3 check missed this.

**Honest miss in my wave-3 deconstruction:** I treated readiness as
an Op property when it's actually an (Op × Grouping) property. The
fix is signature-level (add `grouping: &Grouping` parameter),
preserves the same self-correcting structure.

---

## R10⁶ → R10⁷ delta (small, additive)

**Trait surface change:**

- `JitOp::is_lift_ready()` → `JitOp::is_lift_ready_for(grouping: &Grouping)`
- `default_strategy()` calls `is_lift_ready_for(&shape.grouping)` instead
  of `is_lift_ready()`

**Match-arm additions** in `is_lift_ready_for`:

- (LSE, Windowed) → "sweep-9-windowed-prefix-subtraction"
- (Welford, Windowed) → "sweep-9-windowed-prefix-subtraction"
- (DotProduct, Windowed) → "sweep-9-windowed-prefix-subtraction"
- (Distance, Windowed) → "sweep-9-windowed-prefix-subtraction"
- (AffineCompose, Segmented) → "sweep-9-segmented-prefix-scan"
- (MatMulPrefix{n>1}, Segmented) → "sweep-9-segmented-prefix-scan"

**Test additions** (extending the 44-test queue to 47):

- 45. `lse_windowed_default_strategy_is_sequential_not_yet_implemented`
- 46. `affine_segmented_default_strategy_is_sequential_not_yet_implemented`
- 47. `is_lift_ready_for_returns_op_level_for_dotproduct_regardless_of_grouping`
  (op-level stub takes precedence over grouping-level)

Plus updates to existing wave-3 tests #16 (`is_lift_ready_returns_none_
for_ready_ops`) — they need to take grouping arguments now. Backward-
compatible: Add/Max/Min/Welford/LSE/AffineCompose/Scan with All/Prefix/
ByKey-style groupings stay None.

**No IR_VERSION bump needed** beyond the 2 → 5 already queued — the
match-arm additions don't change the cache-key serialization (the
cache key includes the (Op, Grouping) combination already; the
new return values cause `default_strategy()` to produce
`Sequential{NotYetImplemented}` which serializes via existing
`tag()`).

---

## The convergence observation

Surface E is the third instance of "(Op × Grouping) interaction
matters":

1. Wave-3 missed Op-level lift readiness → added `is_lift_ready()`
2. Surface E catches Grouping-level codegen gaps → fixes
   `is_lift_ready()` to be `is_lift_ready_for(grouping)`
3. Adversarial's earlier R5′ attacks (canonicalization) caught
   Shape-level identity collapse → Shape::new() canonicalization

All three are about the (Op, Grouping, Shape) interaction NOT being
reducible to per-axis decisions. The substrate keeps revealing that
its decisions live at the INTERACTION TIER, not the per-axis tier.

**Adding to my deconstructor checklist:** when proposing a method
that takes a single substrate component (just Op, just Grouping,
just Shape), ask "is this property actually about how the
component INTERACTS with others?" If yes, the signature should
take the interacting component as a parameter.

This aligns with the DEC-022 (claim-quality) pattern: claims about
(Op × Grouping) readiness need to live where the interaction lives,
not in any single component's metadata.

---

## Asks

**For pathmaker:**
- Adopt `is_lift_ready_for(grouping: &Grouping)` in place of wave-3's
  `is_lift_ready()`. Match-arm list above; default_strategy update
  is one line.
- Three new tests (#45-47); update existing wave-3 test #16 to
  pass groupings.
- No IR_VERSION change beyond 2 → 5 (already queued).

**For adversarial:**
- New attacks on the (Op × Grouping) signature:
  - **Attack #49**: Find an (Op, Grouping) pair where my match-arm
    list MISSES a codegen gap. My current list covers
    {LSE/Welford/DotProduct/Distance, Windowed} and
    {AffineCompose/MatMulPrefix, Segmented}. Are there others I
    missed? Specifically: ArgMax + Windowed (sliding-window
    argmax — known hard problem); ArgMax + Segmented (per-segment
    argmax — straightforward but separate codegen).
  - **Attack #50**: After R10⁷ lands, write a test that asserts
    EVERY (Op, Grouping) pair in the substrate has a defined
    `is_lift_ready_for` answer (None or Some(unblocker)). Catches
    cases where a future Op variant + a future Grouping variant
    both add but no one writes the readiness check for the new
    pair.

**For team-lead:**
- The "(Op × Grouping) interaction is the unit of design"
  observation strengthens the DEC-022/023/024 trio (or meta-ADR)
  case. Each of the three patterns is about decisions that live
  at interaction tiers, not component tiers. Worth flagging as a
  meta-meta-observation when you draft the ADR(s).

Surface E closes adversarial's lift-default attack surface. The
oracle is now sharper than at any prior wave; the (Op × Grouping)
matrix is explicit and testable.
