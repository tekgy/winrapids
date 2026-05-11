# Response to ArgMax OrderDependent Overcautious (Adversarial 2026-04-22)

## Accepted. FIFTH parent extraction today.

Adversarial is correct. ArgMax's combine with lowest-index tiebreak is commutative AND associative on the full State space. `DeterminismClass::OrderDependent` is the wrong classification. The correct class is `Deterministic` (R10⁸ spec) or `BitExact` (R10⁹ renaming).

## Phase-8 verification

`ArgMax.combine((v_a, i), (v_b, j))`:
- `v_a > v_b`: return `(v_a, i)` — commutative (swap a,b returns the same max)
- `v_b > v_a`: return `(v_b, j)`
- equal: return `(v, min(i,j))` — `min` is commutative + associative over `usize`

The tiebreak operation `min(i,j)` doesn't care what `i` and `j` *represent* — whether they're original input positions, recipe-computed indices, or anything else in `usize`. Commutativity is robust across the full `State = (f64, usize)` space. This is NOT the AffineCompose-nuance trap (where commutativity held on a default-lift subset but failed on a wider domain). ArgMax's commutativity holds on the full domain.

**Full state domain check (per DEC-025′ sub-clause a)**:
- `{(v, i) : v ∈ f64, i ∈ usize}` — full reachable State
- `combine` on any `(s1, s2)` pair: commutes, associates, bit-exact
- No privileged subset; no hidden footgun

Confirmed: `is_commutative() = true` for ArgMax is the honest answer.

## What this means for R10⁹

**Three deltas**:

### 1. `JitOp::ArgMax.is_commutative() -> true`

Currently wrong at `jit_op.rs:127`-ish or wherever `is_commutative()` matches ArgMax. Should flip to `true`.

### 2. `DeterminismClass::OrderDependent` variant is suspect

If ArgMax was the original motivating case for `OrderDependent` — and I can't currently remember another Op that needs it — then `OrderDependent` may be a zombie variant with no real user. Audit across `JitOp`:

- `ArgMax`: move to `Deterministic` (R10⁹ R8 variant if BitExact renaming happened: `BitExact`)
- `ArgMin`: same as ArgMax by symmetry — move to `Deterministic` / `BitExact`
- `AffineCompose`: non-commutative, but the correct codegen constraint is `Lifted { scan_family: OrderPreserving, segment_reset: true }` per DEC-025′ sub-clause (b). The DeterminismClass itself is `BitExact` (exact across runs with the order-preserving tree); it's the SCAN FAMILY that's constrained, not the determinism class.
- `MatMulPrefix`: same as AffineCompose.

**If no Op requires `OrderDependent` after audit**: the variant is dead code. R10⁹ should either remove it OR document explicitly which Op(s) legitimately need it (and why the `is_commutative()` + `scan_family` machinery doesn't cover their case).

### 3. Parent extraction: `DeterminismClass` and `is_commutative()` answer DIFFERENT questions

This is the structural finding underneath adversarial's correctness probe:

- `DeterminismClass` answers: **"What's the determinism guarantee of the output across runs?"** (BitExact / MathematicallyEquivalent / SeededDeterministic / SizeDeterministic / NonDeterministic / ... per R10⁸)
- `is_commutative()` answers: **"Can codegen use an arbitrary-tree parallel scan?"** (true = yes; false = must use OrderPreserving family per DEC-025′ sub-clause b)

The *third* question — **"Does the final result depend on input element ordering?"** — is a property of the **combine + lift pair**, not the determinism class. For ArgMax, the answer is yes (the result IS the minimum index where the max occurs), BUT this input-order-sensitivity has NO bearing on whether codegen can use an arbitrary-tree scan. The scan produces the correct (value, min-index) pair regardless of tree shape; what the user sees is the same answer across any correct implementation.

Adversarial's verdict is precise: `OrderDependent` conflated (a) "combine requires fixed tree shape" with (b) "result depends on input-element-ordering." These are independent. Classification based on (b) is a category mistake when the codegen-relevant question is (a).

## R10⁹ → R10¹⁰ proposed

This is a small shape change but a real one:

**Add to `JitOp`**:
```rust
/// Is the RESULT dependent on input element ordering?
///
/// This answers "does reordering the input elements change the OUTPUT?" —
/// a SEMANTIC property distinct from whether the combine operation admits
/// an arbitrary-shape tree.
///
/// - `Add`, `Max`, `LogSumExp`: false (same output regardless of input order)
/// - `ArgMax`, `ArgMin`: **true** (result IS the first-matching index, which
///   changes if the input is reordered)
/// - `AffineCompose`, `MatMulPrefix`: true (sequential composition — input
///   order IS the semantic)
///
/// This method is DISTINCT from `is_commutative()`:
/// - `is_commutative()` asks about the `combine` operation's algebra
/// - `result_depends_on_input_order()` asks about the `accumulate` function's
///   semantic behavior
///
/// Ops where `is_commutative() == true` and
/// `result_depends_on_input_order() == true` are correct and common (ArgMax).
/// Ops where `is_commutative() == false` imply
/// `result_depends_on_input_order() == true` (if combine non-commutes on
/// the full State, the composition order of lifted elements matters).
pub fn result_depends_on_input_order(&self) -> bool {
    match self {
        JitOp::Add | JitOp::Max | JitOp::Min | JitOp::LogSumExp | JitOp::Welford
        | JitOp::DotProduct | JitOp::Distance
        | JitOp::Scan(SemiringTag::Boolean)
        | JitOp::Scan(SemiringTag::TropicalMinPlus)
        | JitOp::Scan(SemiringTag::TropicalMaxPlus)
        | JitOp::Scan(SemiringTag::Ring(_))
            => false,
        JitOp::ArgMax | JitOp::ArgMin
        | JitOp::AffineCompose | JitOp::MatMulPrefix { .. }
            => true,
    }
}
```

This doesn't change codegen directly. It documents the semantic answer to "does input order matter to the user's final result?" — which is sometimes asked by recipe composition logic and currently would have to infer it. Making it explicit closes the ambiguity that produced the original `OrderDependent` mis-classification.

**Is this a useful addition vs. overdesign?** Honest read: it's borderline. If no consumer currently needs this distinction, it's speculative. If future consumer logic needs "does the first/last element matter?" (e.g., for stable sort guarantees, for streaming semantics, for determinism-class composition in recipe fusion), then having it explicit from the start is right. I'm going to recommend adding it because the `OrderDependent` mis-classification was itself evidence that someone *was* trying to answer this question, and the conflation is what produced the bug. But flagging uncertainty honestly.

**Alternative**: don't add `result_depends_on_input_order()`. Just fix ArgMax's DeterminismClass, remove the dead variant if dead, and let future consumer needs drive the question. This is the anti-YAGNI path, and it's defensible.

**Recommendation**: flip ArgMax's `is_commutative()` AND its `DeterminismClass` now. Audit `OrderDependent`'s other users. Defer `result_depends_on_input_order()` until a consumer surfaces. This is the smaller, more honest change.

## Tests queued

- **#59** (this) — `JitOp::ArgMax.is_commutative() == true`
- **#60** (this) — `JitOp::ArgMax.determinism_floor(...) ∈ { BitExact, Deterministic }` (never `OrderDependent`)
- **#61** (this) — `JitOp::ArgMin` matches ArgMax (symmetry)
- **#62** (if proceeding with the parent extraction) — audit test: for every `JitOp` variant, `DeterminismClass` is not `OrderDependent` OR the Op has a documented reason why `is_commutative() + scan_family = OrderPreserving` doesn't cover its case.

## DEC-025′ implication

This finding reinforces DEC-025′ sub-clause (b): codegen for Prefix/Segmented + non-commutative MUST use OrderPreserving scan family. The consequence now visible: **DeterminismClass should NOT be used as the proxy for "non-commutative combine."** `is_commutative()` IS the proxy. They answer different questions. Adversarial's ArgMax probe made this visible.

I'm going to flag this to team-lead as a clarification on DEC-025′: the codegen scan-family decision is driven by `is_commutative()`, NOT by `DeterminismClass`. Two different sub-clauses, two different invariants. Conflating them produces the wrong classification for ArgMax and mis-constrains codegen.

## Honest credit

This is the fifth parent-extraction win today. Pattern from the garden entry holds: same shape as the other four — *a claim was correct on a subset of what it described and wrong on the full domain.* Here the subset was "ArgMax's result depends on input order" and the full domain was "the `DeterminismClass` classification of ArgMax" — which is a different question entirely. Naming the difference resolves the bug.

The fifth extraction continued to feel normal. Infrastructure compounds.

## Summary

- ArgMax `is_commutative() = true` (was false)
- ArgMax `DeterminismClass = BitExact` / `Deterministic` (was `OrderDependent`)
- ArgMin same
- Audit `OrderDependent` variant's remaining users; may be dead code
- `DeterminismClass` and `is_commutative()` answer different questions;
  codegen scan-family decision uses `is_commutative()`, NOT DeterminismClass
- R10⁹ → R10¹⁰; tests #59-#62 queued

Spec: this file. Task #1 R10¹⁰ description update next.
