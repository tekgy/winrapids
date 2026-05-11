# Phase 1-8 — GAP-LIFT-1: Bit-Exact `Op::Add` Under Lift

**Sweep 8 / GAP-LIFT-1** · Author: aristotle · Date: 2026-04-22

**Trigger:** adversarial wave-2 found that the current `Op::Add` (plain
f64 `+`) loses associativity under lift. Adversarial framed:

> (A) Accept divergence. ~1 ULP. Strategy is in cache key so consumers
> see consistent-per-strategy results. What every other ML framework
> does.
>
> (B) Commit to bit-exact. Requires Kulisch (Sweep 3 primitive). Same
> output for both strategies. Correct for finance.

**My read:** the framing hides the fact that **the substrate already
chose**, and what looks like a binary decision is actually a third
option (C) the substrate implies. Phase 1-8 to make this rigorous.

---

## Phase 0 — Substrate audit (what's already decided)

Before assumptions, what does tambear's locked vocabulary say?

`docs/vocabulary.md` lines 202-204:

> **`Op::Add`** — additive monoid `(ℝ, +)`. **Default strategy:
> Kulisch-backed exact accumulation. Cross-platform bit-exact
> deterministic.**

`docs/vocabulary.md` lines 220-223:

> **Determinism contract:** every Op is cross-platform bit-exact
> deterministic by default. Same input → same bits regardless of thread
> count, execution order, backend, or CPU architecture. **Non-
> determinism is opt-in only via `using(sum_strategy: "nondet")`.**

`docs/vocabulary.md` line 297:

> **The whole chain is bit-exact deterministic.** Same pipeline + same
> input data + any valid combination of available surfaces → identical
> output bits.

The substrate has ALREADY committed to bit-exact as the default. The
"choice" adversarial is presenting is choosing whether to honor the
substrate's commitment. **Option (A) "accept divergence" violates the
locked vocabulary's Determinism Contract.** It is not a peer of (B).

Today's `Op::Add` lowering uses plain f64 `+` (per `accumulate.rs:268`)
and is therefore **already wrong** against the substrate — it just
happens to work bit-exactly for sequential because both lifted and
sequential use the same operator and adversarial happens not to have
exercised the divergence yet under sequential reordering.

This reframes the question entirely.

---

## Phase 1 — Assumption Autopsy

What was implicitly assumed by adversarial's framing?

- **A1 — The Determinism Contract is negotiable.** REJECTED. Per
  vocabulary.md, it's locked. Saying "we accept divergence" requires
  amending the locked vocabulary.
- **A2 — Bit-exact applies only to a single (Op, Shape, Strategy)
  combination.** REJECTED. The Determinism Contract is "same input
  → same bits regardless of … execution order." That's a statement
  about the OP, not about the strategy. Strategy participates in the
  cache key for OTHER reasons (different kernels for different
  parallelism patterns) but it doesn't suspend the contract.
- **A3 — Other ML frameworks accept divergence, so we can.** REJECTED.
  The whole tambear thesis is "kill scipy/R/MATLAB" by being the
  reference implementation. "Other frameworks do X" is the wrong
  argument; we're the answer to why other frameworks aren't enough.
  See CLAUDE.md DEC-001 + DEC-014.
- **A4 — Divergence is small (~1 ULP).** TRUE for clean inputs;
  CATASTROPHIC for adversarial inputs. The `[1e15, 1.0, -1e15]` test
  drops the `1.0` entirely — that's a 100% relative error, not 1 ULP.
  Heavy-tailed financial data (returns spanning 10 orders of
  magnitude) routinely produces these conditions.
- **A5 — Divergence is acceptable if documented.** REJECTED for any
  Op whose vocabulary entry says bit-exact. The user reading the docs
  expects bit-exact; the implementation must deliver it OR amend the
  docs OR put the divergent path behind opt-in.
- **A6 — The choice is between divergent-fast and bit-exact-slow.**
  PARTIALLY TRUE but misframed. The choice is between:
  - The default behavior promised by vocabulary (bit-exact)
  - The opt-in escape hatch (non-deterministic / fast / weakened
    promise)
  These COEXIST. Both must be in the substrate.
- **A7 — Sweep 8 must ship bit-exact NOW.** PARTIALLY TRUE. Sweep 8
  must NOT ship divergent-by-default. It must either (a) ship Kulisch
  now, or (b) document the contract violation and route to bit-exact
  in Sweep 3+ when Kulisch lands.
- **A8 — Kulisch is "expensive."** TRUE on naive impl, but
  `accumulate.rs:202-203` already ASSUMES Kulisch is the default
  lowering ("Kulisch-backed exact accumulation"). The vocabulary
  pre-commits to it being the implementation; the cost question is
  about *when* it lands, not whether.
- **A9 — Cache-key separation lets the two strategies coexist
  cleanly.** PARTIALLY TRUE. Yes, different strategies hit different
  cache keys, so consumers using `Lifted` always get the same result.
  But this misses the point: under the Determinism Contract, BOTH
  strategies should produce the same result for a given input. The
  cache-key separation is about kernel binaries, not about output
  divergence.
- **A10 — The "ignored" test
  `add_lifted_equals_sequential_on_adversarial_input_kulisch` is
  blocked on a design decision.** REJECTED. It's blocked on Sweep 3
  delivering the Kulisch primitive. The design decision was made by
  the locked vocabulary; the test is named "kulisch" precisely
  because that's what unblocks it.
- **A11 — Pipeline-level `using(strategy=)` and
  `using(sum_strategy=)` are the same axis.** REJECTED. They're
  orthogonal. Strategy = lifted vs sequential vs conjugated (which
  KERNEL to use). Sum_strategy = exact vs fast vs nondet (what
  ARITHMETIC the kernel performs internally). Sum strategy enters the
  cache key as `params:` per the existing fingerprint discipline.
- **A12 — Bit-exactness across (Op, Strategy) combinations forces
  Kulisch even when Kulisch isn't appropriate.** REJECTED. Kulisch is
  the right primitive for ADDITIVE reductions. Other Ops (Max, Min,
  ArgMax, AffineCompose) have different bit-exactness mechanisms
  that are already structurally bit-exact regardless of strategy
  (idempotent, lattice, fixed associativity tree). The contract
  applies to all Ops; the *implementation* differs per Op. Add needs
  Kulisch; Max doesn't need anything special.

---

## Phase 2 — Irreducible Truths

- **T1 — The Determinism Contract is locked vocabulary.** Bit-exact
  cross-(thread, strategy, backend, architecture) by default. Opt-in
  to non-determinism via `using(sum_strategy: "nondet")`.
- **T2 — Strategy is a parallelism axis (which kernel binary).
  Sum-strategy is an arithmetic-precision axis (what arithmetic that
  kernel performs).** Two orthogonal axes, both in the cache key, but
  they decide different things.
- **T3 — Op::Add's default lowering MUST be bit-exact across
  strategies.** Whatever primitive achieves that — Kulisch, pairwise-
  with-fixup, compensated summation, fixed-tree reduction with
  Neumaier compensation — is implementation detail. The contract
  says: same input, same bits, regardless of strategy.
- **T4 — Today's `Op::Add` impl (plain f64 `+`) violates T3 the
  moment any non-trivial parallelism kicks in.** It's tech debt
  against the substrate's locked contract. The current code happens
  to be bit-exact for SEQUENTIAL because plain `+` ordered left-to-
  right is deterministic; the bug is silent until `Lifted` codegen
  ships in 8C.
- **T5 — There is no bit-exactness obligation if the user opts in
  to non-determinism.** `using(sum_strategy="nondet")` allows the
  fastest path the backend can offer; output may diverge by ~ULP.
  This is the explicit escape hatch the vocabulary already provides.
- **T6 — Kulisch is the right primitive for additive bit-exact
  reduction.** ~4350-bit exact integer accumulator that absorbs
  every f64 partial sum without rounding. Per
  `crates/tambear/src/primitives/specialist/` (vocabulary line 256).
  Sweep 3 lands the primitive; Sweep 8C lowers Add through it.
- **T7 — Sweep 3 DOES NOT ship Kulisch yet.** Per accumulate.rs:262:
  > "For Sweep 0, State = f64. The Kulisch-backed deterministic
  > variant (StateBitExact wrapping a Kulisch accumulator) lands in
  > Sweep 3 when the primitive is ported."
  We are post-Sweep-0, but the comment indicates the Kulisch
  variant isn't yet in. Sweep 3 status check needed.
- **T8 — The cache key already distinguishes
  `using(sum_strategy=...)` choices via `params: &[u8]`.** Per the
  cache-bake principle audit and the existing fingerprint
  discipline. Two consumers with `sum_strategy="exact"` vs
  `sum_strategy="nondet"` get different cache entries.
- **T9 — The substrate's commitment is the SCIENTIFIC CLAIM that
  makes tambear different.** "Kill scipy/R/MATLAB" rests on
  reference-implementation correctness. Accepting divergence
  collapses the claim. CLAUDE.md DEC-014 (RunningMedianObservations
  warmup = NaN, not 0.0, for both backends) sets exactly this
  precedent — reject the popular wrong answer in favor of the
  correct contract.
- **T10 — There is a THIRD option (C) that the substrate implies:**
  - Default: bit-exact via Kulisch (or any sum-strategy that meets
    the contract)
  - Opt-in non-determinism: `using(sum_strategy="nondet")` enables
    plain f64 `+` with documented ULP-level divergence between
    strategies
  - Opt-in fast: `using(sum_strategy="fast")` enables compiler-
    fastmath flags (FMA contraction, FTZ, etc.) — distinct from
    nondet because it's still deterministic per-strategy, just
    weaker
  Three sum-strategy modes, all in the cache key as params, each
  distinguishable.

---

## Phase 3 — Reconstruction (the three options as concrete code shapes)

### Option (A) as adversarial framed it

Plain f64 `+` for Add; both strategies use it; both diverge ~1 ULP
between themselves on adversarial inputs; cache key separation hides
the divergence per-consumer-per-strategy.

**Substrate violation:** vocabulary.md says Op::Add is bit-exact by
default. (A) violates the contract.

### Option (B) as adversarial framed it

Kulisch for Add; both strategies use Kulisch internally; bit-exact
output across strategies. Need Sweep 3 to ship Kulisch.

**Substrate alignment:** matches vocabulary.md. But (B) ignores that
some users genuinely want the fast path (e.g. ML training where
ULP-level divergence is in the noise floor).

### Option (C) — the substrate's actual implication

```rust
pub trait OpKind {
    type State: Clone + Send + Sync + 'static;
    fn identity(&self) -> Self::State;
    fn lift(&self, x: f64) -> Self::State;
    fn combine(&self, a: Self::State, b: Self::State) -> Self::State;
    fn extract(&self, state: Self::State) -> f64;
    fn canonical_structure(&self) -> Option<Structure> { None }
    /// What sum-strategy this Op implements internally.
    /// Default: Exact (bit-exact across all strategies).
    fn sum_strategy(&self) -> SumStrategy { SumStrategy::Exact }
}

pub enum SumStrategy {
    /// Bit-exact across (thread, strategy, backend, arch). Kulisch
    /// for Add; idempotent for Max/Min; etc.
    Exact,
    /// Per-strategy deterministic but may differ between strategies
    /// by ~ULP. Plain f64 ops; compiler may use FMA / FTZ.
    Fast,
    /// Order-of-arrival nondeterministic. Backends may parallelize
    /// freely; thread-scheduling affects result. ML training default
    /// when bit-exact reproducibility is not required.
    Nondet,
}
```

`Add` ships TWO concrete impls:

- `Add` (default) — bit-exact, Kulisch-backed
- `AddFast` (opt-in via `using(sum_strategy="fast")`) — plain f64
- `AddNondet` (opt-in via `using(sum_strategy="nondet")`) — plain
  f64, parallel-reorder-free

OR (cleaner): single `Add` struct that takes a `sum_strategy: SumStrategy`
field at construction; the strategy enters the param hash; codegen
specializes per strategy.

```rust
pub struct Add {
    pub sum_strategy: SumStrategy,  // default Exact
}

impl Default for Add {
    fn default() -> Self { Self { sum_strategy: SumStrategy::Exact } }
}
```

When recipes write `accumulate(grouping, expr, Add::default(), data,
validity)`, they get bit-exact Add. When the user wires
`using(sum_strategy="fast")` through, the dispatcher constructs
`Add { sum_strategy: SumStrategy::Fast }` instead. The cache key sees
both in `params:` (because the dispatcher serializes the param bag
per the cache-bake discipline).

**This is the substrate's design.** Sum-strategy is part of the
"params" channel that the cache-bake principle reserves for "values
that change the instruction stream." Different sum-strategies bake
into different kernels.

---

## Phase 4 — Map: rejected assumption → replacing truth

| Assumption | Replacing truth |
|---|---|
| A1 contract negotiable | T1 contract is locked vocabulary |
| A2 bit-exact only per-strategy | T2 strategy ≠ sum-strategy axis |
| A3 other frameworks justify | T9 we're the answer to other frameworks |
| A4 divergence small | A4 false on adversarial input (100% rel error) |
| A5 documenting divergence is sufficient | T1 contract is the doc |
| A6 binary choice | T10 three sum-strategy modes |
| A7 must ship now | T7 ship contract now; Kulisch impl when 8C+3 lands |
| A8 Kulisch expensive | T6 Kulisch is the spec'd default |
| A9 cache separation hides divergence | T1+T2 still violates contract |
| A10 design decision pending | A10 false — vocabulary already chose |
| A11 strategy = sum-strategy | T2 orthogonal axes |
| A12 forces all Ops to use Kulisch | T3 contract is per-Op; impl differs |

---

## Phase 5 — The Aristotelian Move

**MOVE: ship (C) — the three-mode sum-strategy axis already implied
by the substrate.** Sweep 8 implementation phases:

1. **Today (no code change required for Sweep 8 spec):** The locked
   trait spec (per `trait-spec-locked.md`) is unchanged. The
   sum-strategy axis already fits the substrate as a `params:`
   contribution. No 8A re-open.

2. **Sweep 8C (when cranelift codegen for `Op::Add` lands):**
   - The `Add` impl reads its `sum_strategy` field.
   - For `SumStrategy::Exact` — codegen emits Kulisch-accumulator
     calls. **If Sweep 3 hasn't shipped Kulisch, Add's exact path
     stubs to NotYetImplemented; recipes calling Add with default
     sum_strategy panic with a clear message naming Sweep 3 as the
     blocker.**
   - For `SumStrategy::Fast` — codegen emits plain f64 `+` (current
     test behavior, but only when explicitly requested).
   - For `SumStrategy::Nondet` — codegen emits parallel-reorder-
     free f64 `+`; cache key marks NonDeterministic.

3. **Sweep 3 follow-up (Kulisch primitive):** lands the
   `kulisch_accumulator` primitive in
   `primitives/specialist/kulisch.rs`. Sweep 8C's Add Exact path
   calls into it. The `add_lifted_equals_sequential_on_adversarial
   _input_kulisch` test ungates and passes.

4. **Determinism class wiring:**
   - `Add` with `SumStrategy::Exact` → `DeterminismClass::Deterministic`
   - `Add` with `SumStrategy::Fast` → `DeterminismClass::Deterministic`
     (per-strategy bit-exact; cross-strategy may differ; the user
     opted in)
   - `Add` with `SumStrategy::Nondet` →
     `DeterminismClass::NonDeterministic`

**For the immediate question (ungate the
`add_lifted_equals_sequential_on_adversarial_input_kulisch` test):**

That test is **correctly ignored**, and it stays ignored until Sweep 3
lands Kulisch + Sweep 8C lowers Add through it. The decision the
team must make is NOT "(A) or (B)" — it's "what's the order in which
Sweep 3 (Kulisch) and Sweep 8C (Add codegen) land?"

---

## Phase 6/7 — Recursive challenge

- **Q-rec-1.** Does this re-open Sweep 8A? **No.** The trait surface
  is correct. `Op` carries enough state (via `params:`, via the
  `sum_strategy` field on Add) to express this. The cache key
  already serializes `params:`. No trait change.

- **Q-rec-2.** Does this require a new `SumStrategy` enum? **Yes,
  but as a Tier-2 artifact next to other Op parameters.** Lives in
  `crates/tambear/src/accumulate.rs` alongside `Op` definitions.
  Not in the trait surface (`door.rs` / `strategy.rs`).

- **Q-rec-3.** Does this require updating the `ExecutionStrategy`
  enum? **No.** ExecutionStrategy and SumStrategy are orthogonal:
  - ExecutionStrategy = which kernel topology (Lifted /
    LiftedConjugated / Sequential)
  - SumStrategy = what arithmetic the kernel performs (Exact /
    Fast / Nondet)

  Cartesian product = 9 cells; each gets a different cache key; each
  is a coherent kernel. The cache-bake principle handles this
  cleanly because both axes participate in the cache key (one via
  the strategy field, one via params).

- **Q-rec-4.** What about the Welford catastrophic-cancellation case
  from adversarial's lift-default counterexamples? Same shape:
  `WelfordVariance` should expose `sum_strategy` too, with `Exact`
  using a compensated-Welford variant. Per the same substrate. Open
  for adversarial to construct the equivalent adversarial input
  test and stub it as `welford_lifted_equals_sequential_on_
  adversarial_input_compensated`, ignored until the compensated
  primitive lands.

- **Q-rec-5.** What about Op::ArgMax tie-break under tree reduction
  (also from the addendum counterexamples)? Different category.
  ArgMax already declares `OrderDependent` determinism via
  `canonical_structure()` — `not commutative under lowest-index
  tiebreak`. The contract for ArgMax is "same multiset → same
  bits but order-of-arrival within ties matters." The tree reduction
  must preserve **leaf ordering** to honor this; that's a codegen
  discipline (use a deterministic balanced tree with stable
  associativity, NOT warp-shuffle reduction that scrambles pairs).
  This is a Sweep 8C codegen contract, not a sum-strategy axis.

- **Q-rec-6.** Are we forcing every recipe consumer to know about
  Kulisch? **No.** The default is `SumStrategy::Exact` and recipes
  pass `Add::default()` without thinking about it. The choice is
  invisible to the consumer until they need to opt out. Per state-
  conservation, the choice surfaces as a `using()` annotation when
  Adaptive resolves or when TAM picks a strategy in Sweep 23.

- **Q-rec-7.** What about the half-step where Sweep 8C lands BEFORE
  Sweep 3? My answer: codegen for `Add { SumStrategy::Exact }` panics
  with a clear message: *"Op::Add::Exact requires the Kulisch
  primitive (Sweep 3). Either land Sweep 3 first, or override at
  call site with `Add { sum_strategy: SumStrategy::Fast }` to use
  the per-strategy-deterministic path with documented cross-strategy
  ULP divergence."* The user keeps agency — they can choose to
  proceed with Fast under conscious tradeoff, or wait for Sweep 3.

---

## Phase 8 — Forced Rejection

- **What if there were NO sum-strategy axis at all — just one
  arithmetic mode per Op?** Then every consumer pays Kulisch's cost
  even when Fast would suffice. ML training, where ULP divergence is
  noise, would be unnecessarily slow. **Sum-strategy axis is
  necessary** for the substrate to serve both finance and ML.

- **What if the sum-strategy lived OUTSIDE the cache key?** Then
  switching between Exact and Fast would silently use stale kernels.
  Catastrophic correctness violation. **Sum-strategy MUST be in the
  cache key (via params).** Confirmed.

- **What if Add's default were `Fast` instead of `Exact`?** Then the
  Determinism Contract is violated by default; users have to opt
  IN to correctness. That's the wrong direction — defaults must
  match documented behavior. **Default = Exact.** Confirmed.

- **What if other ML frameworks' acceptance of divergence MEANS
  it's the right choice for tambear?** This is the core argument of
  Option (A). I addressed in T9: tambear's positioning IS being
  the framework that doesn't accept what others do. The DEC-014
  RunningMedianObservations precedent (NaN warmup, not 0.0) makes
  this explicit. We choose correctness over conformance.

- **What if the user's pipeline can't tolerate the Sweep-3 wait?**
  They opt in to `using(sum_strategy="fast")` per recipe call site.
  No global "weaken everything" knob; per-call agency.

- **What if Kulisch turns out to be too slow in practice?** Then
  there's a SECOND exact path — pairwise summation with Kahan
  compensation in a fixed tree, which is deterministic and exact
  for moderate magnitudes. Substrate already names `kahan_sum`,
  `pairwise_sum`, `neumaier_sum`, `dot_2`, `priest_sum` (vocabulary
  line 251-252). The implementation of `SumStrategy::Exact` can
  evolve while the contract stays put.

- **What MUST ALSO EXIST that we haven't named?** Ghost-hunting.
  Candidate: a **WITNESS** for the bit-exactness claim. When Add's
  Exact path runs, can the user verify that the result matches what
  Kulisch would produce? Yes, via the proof-engine: register a
  Tier B (computational) proof that "for the input fed, lifted ==
  sequential bit-exact." This already exists in tambear's proof
  framework (Sweep 1+). The witness is the integration test
  pattern. **No new mechanism needed.**

- **What if some door (NPU?) physically cannot run Kulisch?** Then
  that door's `supports(Op::Add { Exact }, ..., ...) == false`. The
  dispatcher feature-detects via the existing `DoorCodegen::supports`
  surface. Per state-conservation, the user sees a `using()`
  annotation: *"door X doesn't support Add::Exact; degrading to
  Fast — accept or pick a different door"*. Per-door capability
  participates correctly.

---

## Verdict

**Reject the framing of (A) vs (B). Adopt (C) — the three-mode
sum-strategy axis the substrate already implies.**

Specifically:
1. Locked vocabulary already mandates bit-exact default for Op::Add.
2. The opt-in escape via `using(sum_strategy="nondet")` is already
   spec'd.
3. Today's plain f64 `+` impl is tech debt against the contract.
4. The fix is `Op::Add { sum_strategy: SumStrategy }` with default
   `Exact`, the param entering the cache key.
5. Sweep 8A is NOT re-opened. The trait surface is correct as
   landed. The work belongs in Sweep 3 (Kulisch primitive) + Sweep
   8C (Add codegen specialization per sum_strategy).
6. The `add_lifted_equals_sequential_on_adversarial_input_kulisch`
   test stays ignored until Sweep 3 + 8C land. Its name correctly
   identifies the unblocker.

**Asks of the team:**

- **Adversarial:** is there a (Op, Sum-Strategy, Shape) combination
  where this three-mode design fails to capture an honest user
  intent? My claim: Exact / Fast / Nondet exhausts the meaningful
  choices. Counterexample?

- **Pathmaker:** Sweep 3 (Kulisch primitive) status check — is it
  scheduled? Should it land before Sweep 8C? Or do we ship 8C with
  the panic-trampoline-until-Sweep-3 stub for `Add::Exact`?

- **Team-lead/Tekgy:** ratify (C) as the answer to GAP-LIFT-1.
  This is a substrate-implied rather than a fresh design choice;
  the deconstruction's job was to surface it as such.

- **Consequence for the wave-2 lift counterexamples:** Welford,
  LSE, DotProduct similarly need their per-Op exact path for the
  contract to hold under lift. Each is a separate codegen ticket
  in 8C/8D. They are NOT separate design decisions — same
  substrate, same answer.

---

## Documentation deltas

If (C) is accepted, two small doc updates land:

1. `docs/vocabulary.md` line 220-223 (Determinism contract) gets
   expanded to name the three sum-strategy modes explicitly.
2. `crates/tambear/src/accumulate.rs` Op::Add doc-comment names the
   `sum_strategy` field and links to the determinism contract.

Both are doc-only; no API surface change.

---

## Note on framings

This deconstruction confirms my new heuristic from the negative-
formulation garden entry: **when a question is presented as a
binary, look for the substrate's implied third option.** The
substrate had already chosen; the framing presented the choice as
fresh because the substrate's commitment hadn't been audited. The
deconstruction's value here is recovering the substrate's own answer
and saving the team from relitigating a settled question.

The pattern: when `feels like a design decision`, check first
whether the docs/vocabulary already chose. If yes, the deconstruction
is "remind everyone of the choice + design the implementation that
honors it." If no, then run the full Phase 1-8 to make the choice
fresh.
