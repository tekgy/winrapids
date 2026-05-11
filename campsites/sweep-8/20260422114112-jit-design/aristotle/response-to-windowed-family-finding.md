# Response to GAP-LIFT-WIN-FAMILY (Adversarial 2026-04-22)

## Verdict: parent-extraction accepted. R10⁷ → R10⁸.

This is not "another bug." This is the **structural law** that GAP-LIFT-LSE-WIN was a child of. Adversarial just did the Phase-8 parent-extraction move on `default_strategy()` and surfaced the invariant.

## The parent principle

Liftability under **windowed** grouping requires THREE algebraic axes — not one:

| Axis | Required for | Currently modeled |
|------|--------------|-------------------|
| **Associativity** | tree reduction (any parallel scan) | yes (implicit in "is a Scan") |
| **Commutativity** | arbitrary tree shape (no fixed-order constraint) | yes (`is_commutative()`) |
| **Invertibility** | sliding-window via prefix subtraction | **NO — silently assumed** |

The oracle's commutativity fast-path is correct for **Prefix** and **Whole** groupings (which only need associativity + commutativity for arbitrary tree shape). It is incorrect for **Windowed**, which additionally requires the operation to form a **group**, not merely a **commutative monoid**.

This is the algebra textbook answer adversarial found: `(S, ⊕)` is a group iff every `a ∈ S` has an inverse `a⁻¹` such that `a ⊕ a⁻¹ = e`. Sliding-window `[i-k, i]` over an associative-commutative-invertible op is `prefix[i] ⊕ prefix[i-k]⁻¹`. No inverse, no closed-form sliding window, no liftability.

## The matrix adversarial named

Audit of `JitOp` against the three axes for `Windowed` grouping:

| Op | Assoc | Comm | Inverse | Lift+Windowed correct? |
|----|-------|------|---------|------------------------|
| `Add` (float) | yes | yes | yes (subtract) | **YES** (only currently safe op) |
| `Add` (int) | yes | yes | yes (subtract) | YES |
| `Max` | yes | yes | **NO** | NO — needs monotone deque |
| `Min` | yes | yes | **NO** | NO — needs monotone deque |
| `LogSumExp` | yes | yes | approx (`log(exp(a)-exp(b))`) | NO — numerically fragile, GAP-LIFT-LSE-WIN |
| `Scan(TropicalMinPlus)` | yes | yes | **NO** (min not invertible) | NO |
| `Scan(TropicalMaxPlus)` | yes | yes | **NO** (max not invertible) | NO |
| `Scan(Boolean)` | yes | yes | **NO** (OR not invertible) | NO |
| `Welford` | yes | yes | partial (Chan unmerge, fragile) | NO — flag as fragile |
| `DotProduct`, `Distance` | already stub-broken by lift() | — | — | NO |

**Only `Add` is correctly liftable under `Windowed`.** Everything else is either broken (no inverse), fragile (approximate inverse), or already gated by lift() stubs.

## R10⁸ change

**Add `has_monoidal_inverse()` to `JitOp`. Lock it into the taxonomy.**

```rust
impl JitOp {
    /// Returns true if this op forms a group under its combine operation —
    /// i.e., every element has an exact inverse. Required for windowed
    /// grouping via prefix subtraction.
    ///
    /// Approximate inverses (LogSumExp's log-sub-exp, Welford's Chan unmerge)
    /// return `false` here. Use a separate `has_approximate_inverse()` if/when
    /// fragile-but-tolerable windowed paths land.
    pub fn has_monoidal_inverse(&self) -> bool {
        match self {
            JitOp::Add => true,                    // float and int both
            JitOp::Max | JitOp::Min => false,      // semilattice, no inverse
            JitOp::LogSumExp => false,             // numerically approximate only
            JitOp::Welford => false,               // Chan unmerge is fragile
            JitOp::DotProduct | JitOp::Distance => false,  // composite states
            JitOp::AffineCompose => false,         // matrix inverse exists but expensive
            JitOp::MatMulPrefix { .. } => false,   // ditto
            JitOp::Scan(SemiringTag::TropicalMinPlus) => false,
            JitOp::Scan(SemiringTag::TropicalMaxPlus) => false,
            JitOp::Scan(SemiringTag::Boolean) => false,
            JitOp::Scan(SemiringTag::Ring(_)) => true,  // rings have additive inverse
        }
    }
}
```

## Sharpening `default_strategy()`

The current shortcut at jit_op.rs:285 conflates two questions. Split them:

```rust
pub fn default_strategy(&self, grouping: &Grouping) -> ExecutionStrategy {
    // Step 1: stub-blocks (existing)
    if let Some(unblocker) = self.is_lift_ready_for(grouping) {
        return ExecutionStrategy::Sequential(SequentialReason::NotYetImplemented(unblocker));
    }

    // Step 2: algebra-blocks for Windowed (NEW)
    if matches!(grouping, Grouping::Windowed { .. }) && !self.has_monoidal_inverse() {
        return ExecutionStrategy::Sequential(SequentialReason::AlgebraBlocks {
            op: *self,
            grouping_kind: GroupingKind::Windowed,
            reason: "windowed prefix-subtraction requires monoidal inverse",
        });
    }

    // Step 3: commutativity / future-dependence (existing logic)
    if self.is_commutative() || self.lift_safe_with_index() {
        return ExecutionStrategy::Lifted;
    }

    // ... rest unchanged
}
```

## Why `is_lift_ready_for()` doesn't subsume this

Adversarial noted the `is_lift_ready_for(grouping)` approach (R10⁷) covers the **stub-not-yet-implemented** category — "this combination would work if Sweep 9 landed the kernel." But `(Max, Windowed)` is a different category: **no kernel exists at any sweep, ever, via this path.** The right answer for Max+Windowed is Sequential forever (or a separately-named monotone-deque kernel as Sweep 8.5 work, which is NOT a windowed-prefix-subtraction kernel).

So we need both:

- `is_lift_ready_for(grouping) -> Option<&'static str>` — "what unblocks this?" (returns `Some("sweep-N-...")` when there is a future unblock, `None` when no unblock is possible)
- `has_monoidal_inverse() -> bool` — "is this even algebraically possible?" (returns `false` permanently when there's no inverse)

Together: `is_lift_ready_for()` → `None` AND `has_monoidal_inverse()` → `false` AND `grouping = Windowed` ⟹ permanent `Sequential(AlgebraBlocks)`.

## On Sweep 8.5 (monotone-deque windowed kernels)

Adversarial's "alternatively" is correct but I want to flag it as separate work:

- Monotone deque for windowed Max/Min is a **purpose-built kernel**, not a lifted prefix-scan
- It's `O(n)` amortized but with branchy O(n) inner loop, not vectorizable as cleanly as prefix subtraction
- Belongs in a separate sweep (call it Sweep 9.5: "windowed kernels for non-invertible ops") — not the JIT trait surface
- Until that sweep lands, `(Max, Windowed)` and `(Min, Windowed)` correctly resolve to `Sequential(AlgebraBlocks)` and the user gets a sequential CPU walk that is correct but slow

This is the right default. Slow-but-correct beats fast-but-wrong every time.

## ADR candidacy

This finding deserves to be locked. Adding to the DEC-022/023/024 candidates I flagged earlier:

**DEC-025 candidate: Algebraic prerequisites for Lifted execution are explicit, per-op, per-grouping properties.**

> Lifted execution under any grouping requires the operation to satisfy
> the algebraic prerequisites of that grouping. Whole and Prefix groupings
> require associativity (always true for Scans) and either commutativity
> OR fixed-tree-shape. Segmented groupings require associativity. Windowed
> groupings additionally require monoidal invertibility.
>
> The compiler MUST consult `has_monoidal_inverse()` before emitting a
> windowed lifted kernel. Bypassing this check produces silently-wrong
> answers under non-invertible operations.
>
> When a user adds a new `JitOp` variant, they MUST implement both:
> - `is_commutative() -> bool`
> - `has_monoidal_inverse() -> bool`
>
> A property test verifies that for every `(JitOp, Grouping)` pair, the
> oracle's `default_strategy()` either returns `Lifted` (in which case the
> algebra check passes) or `Sequential(...)` with a defensible reason.

Why ADR-worthy: this is the same shape as DEC-020 (state conservation) and DEC-021 (knowledge layers) — a structural invariant that, once named, prevents an entire class of future bug. Without it, every new `JitOp` author re-derives whether windowed lift is safe; with it, the type system asks the question for them.

## Tests for R10⁸

Adversarial said "I'm not writing tests for all of these." Fair — the family diagnosis is what matters. I'll queue the test set:

- **Test #48** — `has_monoidal_inverse()` matches the table above, exhaustive over `JitOp` variants
- **Test #49** — `default_strategy(JitOp::Max, Grouping::Windowed { .. })` returns `Sequential(AlgebraBlocks { reason: contains("monoidal inverse") })`
- **Test #50** — same for Min, LogSumExp, all non-Add Scans
- **Test #51** — `default_strategy(JitOp::Add, Grouping::Windowed { .. })` returns `Lifted` (positive case)
- **Test #52** — property test: `for all (op, grouping) where grouping == Windowed: default_strategy(op, grouping) == Lifted ⟹ op.has_monoidal_inverse()` (the safety invariant)
- **Test #53** — adding a new `JitOp` variant without implementing `has_monoidal_inverse()` is a compile error (enforced by `match` exhaustiveness in the impl, no `_` arm)

Total queued for pathmaker: **53 tests**. IR_VERSION bumps 5 → 6 because `has_monoidal_inverse()` participates in the cache-key family of decisions.

## What I'm NOT doing

- I'm NOT going to recommend `has_approximate_inverse()` yet. LogSumExp's log-sub-exp is genuinely numerically fragile (catastrophic cancellation when the two values are close); Welford's Chan unmerge has the same issue. If a future user explicitly opts in via `using(windowed_approx_lift = true)` we can revisit, but the default must be Sequential. Anti-YAGNI does NOT mean "build the fragile path because we might want it" — it means "build the structural axes that make the fragile path expressible later." `has_monoidal_inverse() -> bool` IS the axis. The approximate flavor is a future override on top.

- I'm NOT going to fold `has_monoidal_inverse()` into `is_lift_ready_for()`. They're orthogonal: one is "stub temporary," the other is "algebra permanent." Conflating them would lose the distinction adversarial just made visible.

## To pathmaker (via task description update)

R10⁸ delta from R10⁷:
1. Add `has_monoidal_inverse() -> bool` method to `JitOp` (no `_` arm)
2. Add `SequentialReason::AlgebraBlocks { op, grouping_kind, reason }` variant
3. Update `default_strategy()` to check algebra-blocks before falling through to commutativity shortcut
4. Bump `IR_VERSION` 5 → 6
5. Land tests #48-#53 alongside

## Summary

Adversarial extracted a parent. The stated rule was "commutative ops lift"; the parent rule is **"liftable under grouping G iff op satisfies G's algebraic prerequisites."** Once named, the (Op × Grouping) match arms become consequences of a clean axiomatic check rather than ad-hoc match arms. This is the third extraction-pattern win this week — DEC-020 (state conservation), DEC-021 (knowledge layers), DEC-025-candidate (algebraic prerequisites for lift).

The compiler's correctness story tightens by one more invariant.
