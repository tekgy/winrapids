# Design Anchor — What Sweep 8 Must Surface for Pipeline-Lift

**Sweep 8 / Task 8A** · Author: aristotle · Date: 2026-04-22

**Purpose:** Pipeline-level lift analysis (loop fusion + CSE at the
recipe-composition graph) is a future sweep (post-9). It will consume
Sweep 8's substrate but does not require Sweep 8 to expose anything
new — *as long as Sweep 8 doesn't accidentally close off the options
the pipeline-lift pass will need*.

This file captures the **substrate requirements** so the convergence-
check on pathmaker's implementation pass can verify them. None of these
add to Sweep 8's scope; they are *invariants* the existing trait spec
should satisfy by construction.

---

## The pipeline-lift pass — what it consumes from Sweep 8

Per team-lead's sketch, the pass:
- Lives in `crates/tambear/src/pipeline_lift/` (or `tambear-tam`)
- Runs at `tbs_compile(pipeline)` — early, cached
- Inputs: recipe composition graph + assumption-tag bag + proof-engine
  algebraic properties
- Outputs: annotated graph where each atom call has
  `ExecutionStrategy + Option<FusedWith(other_atom_call_id)>`
- Has its own cache, keyed by recipe-IR hash + assumption-tag bag
  (NOT data shape — schedule is a function of structure)

What this needs from Sweep 8's substrate:

---

## Requirement S1 — JitOp variants must be cheaply comparable for fusion

**The pipeline-lift pass needs:** a fast equality/hashing operation
on `JitOp` so it can identify pairs of atom calls that share the same
Op (both are `JitOp::Add`, both are `JitOp::Welford`, etc.).

**Sweep 8 spec:** `JitOp` is `#[derive(Debug, Clone, PartialEq)]` with
a stable `tag(): &'static str`. ✓ satisfied.

**Convergence-check question:** does pathmaker's implementation
preserve `PartialEq` and the stable `tag()`? If JitOp ever grows a
non-comparable inner field (e.g., a closure or a non-Eq type),
pipeline-lift can't hash it.

---

## Requirement S2 — Algebraic structure must be queryable per-Op

**The pipeline-lift pass needs:** for each (Op_i, Op_j) pair,
"do these compose into a single fused pass?" The decision rests on
shared monoid structure, identity, distributivity (for fused
multiplication+addition into a single SIMD lane), and commutativity
preservation.

**Sweep 8 spec:** `OpKind::canonical_structure() -> Option<Structure>`
already returns the proof-engine `Structure { laws: Vec<StructuralFact>,
... }`. JitOp variants share this via their corresponding OpKind impl
(see `accumulate.rs`). ✓ satisfied — proof engine is the source of
truth.

**Convergence-check question:** does the `JitOp` enum expose a way to
get back to the `canonical_structure()` of its corresponding OpKind?
Currently `JitOp::is_associative()` and `JitOp::is_commutative()`
exist; adding `JitOp::canonical_structure() -> Option<Structure>`
would let pipeline-lift read the full law list without round-tripping
through the OpKind trait. **This is a free addition; flag for
pathmaker.**

---

## Requirement S3 — Shape compatibility must be CSE-friendly

**The pipeline-lift pass needs:** to identify two atom calls whose
output shapes are compatible-for-sharing — even if the specific
recipe authors didn't tag them as a TamSession intermediate. The
analysis runs over the *whole pipeline graph* and may discover
sharing the recipe authors didn't.

**Sweep 8 spec:** `Shape::is_share_compatible_with(other: &Shape) ->
bool` already exists. Will be extended in R5′ to handle multi-dim
axes + symbolic_groups + alignment. ✓ satisfied (after R5′).

**Convergence-check question:** is `is_share_compatible_with`
**reflexive**, **anti-symmetric on stricter-producer→looser-consumer**,
and **transitive**? Pipeline-lift may need to compose compatibility
checks across chains of intermediates. Verify this in tests.

---

## Requirement S4 — ExecutionStrategy must be an open extension point

**The pipeline-lift pass needs:** to introduce a **fourth strategy
variant** — `Fused { with: AtomCallId }` — that says "this atom call
is fused into another atom call's pass." This is INVISIBLE to per-Op
`default_strategy(shape)` because it requires whole-pipeline analysis.

**Sweep 8 spec:** `ExecutionStrategy` is currently a closed enum with
three variants (Lifted, LiftedConjugated, Sequential). To add `Fused`
later, the enum must remain `#[non_exhaustive]` OR the cache key
contribution must be variant-tag-based so adding a variant doesn't
collide with existing entries.

**Convergence-check requirement:** mark `ExecutionStrategy` as
`#[non_exhaustive]` in the enum declaration. The cache key
serialization should hash the variant *tag string* not the variant
*ordinal* (so adding `Fused` later doesn't shift other variants'
cache keys and silently re-key all kernels).

**Flag for pathmaker:** add `#[non_exhaustive]` and use tag-based
serialization. Costs nothing now; protects pipeline-lift later.

---

## Requirement S5 — Atom calls need stable identity

**The pipeline-lift pass needs:** to refer to specific atom calls in
the pipeline graph by stable IDs. `Fused { with: AtomCallId }`
requires `AtomCallId` to be a thing.

**Sweep 8 spec:** No `AtomCallId` today. The `accumulate()` atom
function takes parameters and returns a result; it doesn't know
"which atom call" it is in the pipeline.

**Convergence-check question:** does Sweep 8 need to surface this?
**My answer: NO.** AtomCallId belongs to the pipeline-lift pass,
which constructs the pipeline graph by tracing recipe composition.
Sweep 8 just needs to ensure that the trait surface doesn't FORBID
the pipeline-lift pass from assigning IDs externally. It doesn't —
`accumulate()` is a free function, not a method on a stateful object.
The pass labels calls during graph construction.

**No Sweep 8 action.**

---

## Requirement S6 — Determinism class must compose correctly across fused atoms

**The pipeline-lift pass needs:** when fusing atom call A
(`DeterminismClass::Deterministic`) with atom call B
(`DeterminismClass::OrderDependent`), the fused kernel's determinism
is the *weakest* of the two — `OrderDependent`. This composition rule
must be visible.

**Sweep 8 spec:** `DeterminismClass` is on `CompiledArtifact`. Three
variants: `Deterministic`, `OrderDependent`, `NonDeterministic`. A
total order exists (Det < OrderDep < NonDet) and the "weakest"
operation is `max` over this order.

**Convergence-check requirement:** add a method
`DeterminismClass::weakest_of(self, other: Self) -> Self` (or
`Ord` impl such that fusion takes `max(a, b)`). Free addition; takes
five lines. Pipeline-lift will read it.

**Flag for pathmaker:** add the helper method with documentation
referencing this design anchor.

---

## Requirement S7 — Validity must compose correctly across fused atoms

**The pipeline-lift pass needs:** when fusing atom call A
(`Validity::Propagate`) with atom call B (`Validity::Ignore`), what's
the fused kernel's validity policy? Two atoms with conflicting
validity policies CANNOT fuse — they need different lift behavior on
NaN/Inf inputs.

**Sweep 8 spec:** `Validity` is in `CacheKey` already. Conflict means
two atoms compile to different kernels.

**Convergence-check requirement:** add a method
`Validity::can_fuse_with(self, other: Self) -> bool` returning true
iff `self == other`. (Conservative: identical policies fuse;
different policies don't.) Free addition.

**Flag for pathmaker:** add the helper. Pipeline-lift will use it as
a precondition for fusing two atom calls.

---

## Requirement S8 — Sharing tag canonicalization

**The pipeline-lift pass needs:** to compare assumption-tag bags
across atom calls quickly. R5′'s `tags: Vec<AssumptionTag>` is
already canonically sorted, so equality is a pointer-deep compare.

**Sweep 8 spec (R5′):** Tags are sorted; AssumptionTag has a stable
sort key. ✓ satisfied.

**Convergence-check requirement:** verify the sort is stable across
runs (already tested in `tags_sort_canonically`).

---

## Requirement S9 — IntermediateTag includes enough metadata for fusion

**The pipeline-lift pass needs:** when discovering CSE opportunities
the recipe author didn't tag, the pass MAY assign new
`IntermediateTag`s based on (Op, Shape, params). The TamSession
machinery should accept these synthetic tags the same way it accepts
recipe-authored ones.

**Sweep 8 spec:** `IntermediateTag::build(input_id, computation,
assumptions)` is already content-addressed BLAKE3. Synthetic tags
constructed by the pipeline-lift pass with `computation =
"synthetic_cse"` plus a stable assumption fingerprint will not
collide with recipe-authored tags. ✓ satisfied by construction.

**No Sweep 8 action.**

---

## Summary — what to flag for pathmaker

Five tiny additions, each five lines or less, free under YAWNI, that
unblock pipeline-lift later:

1. **`#[non_exhaustive]` on `ExecutionStrategy`** + tag-based cache
   key serialization (S4)
2. **`JitOp::canonical_structure() -> Option<Structure>`** delegating
   to its OpKind counterpart (S2 — already implicit, just expose)
3. **`DeterminismClass::weakest_of(self, other) -> Self`** OR an
   `Ord` impl (S6)
4. **`Validity::can_fuse_with(self, other) -> bool`** (S7)
5. **Doc comment on `is_share_compatible_with` asserting transitivity**
   + a property test (S3)

Plus one *non-action* — confirm we don't need `AtomCallId` at the
trait layer (S5; the pipeline-lift pass owns that).

These are not blockers for Sweep 8 closeout; they are **substrate
hygiene** that prevents pipeline-lift from needing to backfill them.
They cost minutes; they save a sweep.

---

## What this anchor does NOT commit to

- Pipeline-lift will live somewhere; the exact module path is post-9
  scope.
- The fusion algorithm itself (greedy / DP / SAT) is post-9 scope.
- The "fused with" cache-key contribution is post-9 scope.
- Self-reference detection (true non-liftability) is post-9 scope.
  Sweep 8's `SequentialReason::FockBoundary` reserves the slot;
  populating it requires the pipeline-lift pass.

This anchor only locks in what Sweep 8 must NOT close off.

---

## Convergence-check tie-in

When pathmaker's implementation lands, my re-pass will check:

- [ ] `ExecutionStrategy` carries `#[non_exhaustive]`
- [ ] Cache key serialization uses variant tags, not ordinals
- [ ] `JitOp::canonical_structure()` exists and delegates to OpKind
- [ ] `DeterminismClass::weakest_of` exists with property tests
- [ ] `Validity::can_fuse_with` exists with property tests
- [ ] `is_share_compatible_with` reflexive + transitive (property test)
- [ ] `IntermediateTag::build` doc-comments mention synthetic-CSE use case

If any of those are missing, the convergence check fires and 8A
re-opens for a 5-line patch.
