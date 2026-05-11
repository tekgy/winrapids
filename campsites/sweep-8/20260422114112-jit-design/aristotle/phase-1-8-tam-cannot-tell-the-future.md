# Phase 1-8 — "TAM Can't Tell the Future" Reframe

**Sweep 8 / Task 8A (ongoing)** · Author: aristotle · Date: 2026-04-22

**Trigger:** team-lead relaying Tekgy:

> "Self-reference" is the wrong word for the lift/sequential boundary.
> Cleaner: **TAM can't tell the future.** TAM knows EVERYTHING about
> the pipeline-plus-data-plus-hardware triple before any compute runs,
> except the actual numerical values that will emerge.

This is a conceptual reframe that retrospectively sharpens every
boundary decision I've been making. Running Phase 1-8 on the reframe
itself because the implications ripple further than a rename.

---

## Phase 1 — Assumption Autopsy

What was I (and the framing of `SequentialReason::FockBoundary`)
implicitly assuming?

- **A1 — "self-reference" is the right conceptual category for
  genuinely sequential computation.** REJECTED. Self-reference is a
  FORM; future-dependence is the underlying PROPERTY. A function
  that references itself may be liftable if the recursion is affine
  or a fixed-point with known count; a non-self-referential loop
  may be genuinely sequential if its iteration count depends on a
  future value.
- **A2 — The Fock boundary separates "parallel" from "sequential" by
  mathematical property of the operator.** PARTLY WRONG. The Fock
  boundary is really about **what the compiler can KNOW at compile
  time vs what requires runtime observation.** The operator's
  algebra determines liftability when the control flow is
  determinable; the data-dependence of control flow determines
  whether the compiler can even *evaluate* the operator's liftability
  claim.
- **A3 — Sequential is a KIND of computation.** REJECTED. Sequential
  is a **last-resort codegen** for when TAM can't predict the
  iteration count or termination condition. It's a capability gap on
  TAM's side, not a property of the math.
- **A4 — The proof engine tells TAM "this is sequential."** REJECTED.
  The proof engine tells TAM *algebraic facts* (associativity,
  commutativity, monoid, etc.). The sequential determination is a
  downstream judgment that reads: "given these facts + the control
  flow graph + the data profile, can I predict the iteration bounds?"
- **A5 — The boundary is binary (parallel vs sequential).** REJECTED.
  The boundary has a *structure*: liftable-with-fixed-iteration,
  liftable-with-data-driven-bound-but-known-termination,
  liftable-via-speculative-execution-with-rollback, liftable-via-
  probabilistic-termination, sequential-because-future-dependent.
  The reframe lets me see the gradient.
- **A6 — TAM's knowledge at compile time is static (recipe + Op
  algebra).** REJECTED. Per the extension: TAM profiles the USER'S
  DATA at pipeline-attach time. Data profile is part of TAM's
  compile-time state. The `has_known_non_finite` flag on Shape
  isn't just a codegen hint; it's an EXPRESSION of compile-time data
  knowledge.
- **A7 — `Validity::Error` dispatch is what catches future-dependent
  NaN propagation.** REJECTED. Data-quality analysis catches it
  BEFORE dispatch — "your column has 127 NaN; step 4 will produce
  NaN output; here are your options." The compile-time data profile
  upgrades from "unknown" to "known NaN present," and the compiler
  generates a kernel specialized for that.
- **A8 — TAM operates in three phases (parse, compile, dispatch).**
  REFINED. Per Tekgy's reframe: TAM has THREE KNOWLEDGE LAYERS,
  which is different from phases:
  - Eternal truths (proof engine; pre-pipeline)
  - Composition truths (pipeline structure + data profile;
    pre-compute)
  - Numerical truths (only after dispatch)
  Compile time sits at the boundary of layers 2 and 3. Dispatch
  accesses layer 3.
- **A9 — Data is an INPUT to dispatch, not a compile-time artifact.**
  REJECTED. Data is an input to BOTH compile and dispatch. At
  compile time, TAM profiles the data (NaN count, dtype, scale,
  sparsity, distribution shape). At dispatch, the numerical values
  flow through the kernel. The profile is compile-time knowledge;
  the values are runtime.
- **A10 — The liftability decision runs in isolation per atom.**
  REJECTED. The decision uses Op algebra (layer 1) AND pipeline
  structure (layer 2) AND data profile (layer 2 extended). All three
  feed the judgment.
- **A11 — Fock boundary is a theoretical concept only.** REJECTED.
  It has a CONCRETE operational test: "Does the control flow graph
  of this atom/pipeline segment have a control-dependent edge on a
  value that isn't computed yet?" If yes, sequential. If no,
  liftable.
- **A12 — My SequentialReason enum is complete.** REJECTED. Needs
  refinement. `FockBoundary` is the RIGHT slot but wrong name.
  `FutureDependent { description }` is better. Plus potential
  expansions: `FutureDependentStopCondition`, `FutureDependent
  IterationCount`, `FutureDependentBranch` — three sub-kinds of
  future dependence the pipeline compiler can distinguish. Defer
  the sub-kinds to the pipeline compiler sweep.

---

## Phase 2 — Irreducible Truths

- **T1 — TAM knows THREE things at compile time:**
  1. Algebraic properties of Ops (proof engine; eternal).
  2. Pipeline structure: recipes, their arrangement, data
     bindings, hardware assignment.
  3. Data profile: dtype inference, NaN/Inf count, scale, sparsity,
     distributional shape, cardinality (profile-summary, not
     values).
- **T2 — TAM DOES NOT KNOW numerical values at compile time.** Only
  after dispatch. This is the ONLY thing it doesn't know.
- **T3 — The only genuine forcer of sequential execution is:
  control flow whose branching/termination depends on a value TAM
  cannot yet predict.** Future-dependent control flow.
- **T4 — Data profile is COMPILE-TIME state, stored in Shape.** Fields
  like `has_known_non_finite`, per-axis length distribution,
  dtype-inference confidence, sparsity ratio, assumption-tag bag —
  all populated by the data-quality analyzer BEFORE compile.
- **T5 — Data-quality warnings are USING-ANNOTATIONS.** Per the
  using-annotation principle (from previous deconstruction):
  "column X has 127 NaN; step 4 under Propagate will produce NaN;
  alternatives: Ignore (drops N to 9873), Error (aborts), filter
  step" — this is a `using()` annotation form, author-visible,
  override-eligible.
- **T6 — Auto-inserted preprocess steps are PIPELINE MUTATIONS.**
  When TAM decides "your column Y is strings; I need to cast to
  i64," it INSERTS a preprocess step into the pipeline. This is a
  pipeline mutation; it must surface to the user (per state
  conservation). User sees the new step appear in their pipeline
  with rationale.
- **T7 — The proof engine is NOT the decider of sequential; it's an
  INPUT to the decider.** The decider reads (algebraic facts + CFG
  + data profile) and emits strategy + sequential-reason. Keeping
  the proof engine pure (facts only, not decisions) lets it feed
  many downstream deciders (scheduling, CSE, autodiff, retargeting,
  ...).
- **T8 — Sequential codegen is a FALLBACK, structurally.** The
  liftability decision tree's LAST branch. Not a peer. Three reasons
  it can fire:
  1. `UserOverride` — user chose sequential via `using()`.
  2. `AlgebraBlocks` — Op's algebra doesn't support lifting (no
     associativity, or non-commutative without fitting conjugation).
  3. `FutureDependent` — control flow awaits numerical values.
  (Previously `FockBoundary`; rename per Tekgy's reframe.)
- **T9 — Data-profile-driven specialization is LAYER 2 KNOWLEDGE,
  LAYER 1 CODEGEN.** TAM knows at compile time "no NaN in this
  column" (layer 2); the codegen elides the validity branch in the
  kernel (layer 1 effect). This is the profile → Shape.assumption_tags
  → codegen specialization pipeline already baked into Sweep 8's
  substrate.
- **T10 — Semantic-rewriting share-source decisions (future sweep)
  ARE compile-time decisions.** They don't require future values;
  they require comparing recipe call signatures + Shape + `using()`.
  Layer 2 only.
- **T11 — Newton-style iteration with convergence check IS
  fundamentally sequential for TAM.** The stop condition
  `||x - x'|| < ε` reads the current iteration's output. TAM can't
  predict the output. No amount of cleverness dissolves this. The
  primitive property is "the stop condition reads the output."
  This is the ONE thing sequential codegen genuinely captures.
- **T12 — Speculative lifting IS possible in principle.** Run K
  iterations speculatively in parallel, check convergence after;
  if converged at iteration 3, discard the rest. This is an
  optimization the pipeline compiler can make in a *future*
  sweep. For now, sequential is the fallback. Worth logging as a
  post-sweep optimization.

---

## Phase 3 — Reconstruction

What the reframe implies about the code shape, concretely.

### R1 — Minimal: rename `FockBoundary` → `FutureDependent`

```rust
pub enum SequentialReason {
    UserOverride,
    AlgebraBlocks,
    FutureDependent {
        /// Human-readable description of the future-dependence.
        description: String,
    },
}
```

Simplest change. Captures the reframe at the enum level. The doc
on `FutureDependent` says: "control flow depends on a value not yet
computed — iteration count, stop condition, or branch direction.
Tam cannot predict this from layers 1 or 2 of its knowledge."

### R2 — R1 + explicit `FutureDependence` sub-taxonomy

```rust
pub enum SequentialReason {
    UserOverride,
    AlgebraBlocks,
    FutureDependent {
        kind: FutureDependenceKind,
        description: String,
    },
}

pub enum FutureDependenceKind {
    /// Stop condition reads output of prior iteration
    /// (Newton residual, EM log-likelihood, MCMC convergence).
    StopCondition,
    /// Iteration count is computed from runtime value
    /// (adaptive mesh, branch-and-bound, reject-sampling).
    IterationCount,
    /// Branch direction is a runtime value
    /// (if X[k] > 0 then A else B, where X[k] depends on X[k-1]).
    BranchSelection,
}
```

More precise but adds categorization the pipeline compiler (future
sweep) does. For Sweep 8, R1 is enough. **R2 logged for the
pipeline-compiler sweep.**

### R3 — R1 + three-layer knowledge doc anchor

In addition to the rename, document the three-layer knowledge
architecture as a first-class concept in `docs/architecture.md` or
a new `docs/tam-knowledge-layers.md`. Pathmaker doesn't need to
code anything; the doc captures the framing so future contributors
share the vocabulary.

Contents:
- Layer 1 — eternal truths (proof engine)
- Layer 2 — composition truths (pipeline structure + data profile)
- Layer 3 — numerical truths (dispatch only)
- Lift/sequential boundary = layer 2 can predict control flow or not
- Data profile examples (NaN count, dtype, scale, distribution)
- Using-annotation surface for data-quality warnings

### R4 — R3 + reorient proof engine as fact-source-not-decider

```rust
// In proof/mod.rs — ensure the proof engine exposes FACTS, not
// DECISIONS. Downstream clients (schedulers, lift-deciders, CSE
// passes, autodiff, retargeters) consume the facts.
impl Structure {
    pub fn facts(&self) -> &[StructuralFact] { &self.laws }
    // ... no `fn is_sequential()`, no `fn recommended_strategy()`;
    // those belong elsewhere
}
```

This is probably already the case — the proof engine is already
fact-oriented. The R4 move is just documentation confirming it.

**Winner: R1 + R3 + R4.** Minimum code (rename), maximum clarity
(architecture doc anchoring three layers + proof-engine-is-facts).

---

## Phase 4 — Assumption → Truth Map

| Assumption | Replacing truth |
|---|---|
| A1 self-reference is the frame | T3 future-dependent control flow |
| A2 Fock is an operator property | T3 Fock is a "can TAM predict CF" property |
| A3 sequential is a kind | T8 sequential is a fallback codegen |
| A4 proof engine decides | T7 proof engine is a fact source; decision is downstream |
| A5 boundary binary | T8 boundary structured into three reasons (plus sub-kinds) |
| A6 TAM static at compile | T1+T4 TAM's layer-2 knowledge includes data profile |
| A7 Validity::Error catches it | T5 data-quality analyzer catches it BEFORE dispatch |
| A8 three phases | T1+T2 three KNOWLEDGE LAYERS (different concept) |
| A9 data = runtime only | T1+T4 profile is compile-time; values are runtime |
| A10 per-atom liftability | T10+T11 uses algebra + CFG + data profile |
| A11 Fock abstract | T11 operational: "does CF read yet-uncomputed value?" |
| A12 SequentialReason complete | T8 rename FockBoundary → FutureDependent |

---

## Phase 5 — The Move

1. **Rename `SequentialReason::FockBoundary` →
   `SequentialReason::FutureDependent { description: String }`.**
   (Or `Cow<'static, str>` if cheap-string-pool matters.)
   Doc-comment says: "Sequential because control flow (stop
   condition, iteration count, or branch direction) depends on a
   yet-uncomputed numerical value. TAM cannot predict this from
   layers 1 or 2 of its knowledge (algebra + pipeline/data profile);
   only layer 3 (runtime values) reveals it."

2. **Add `docs/tam-knowledge-layers.md`** documenting the three-
   layer knowledge architecture. Captures: layer 1 (proof engine,
   eternal), layer 2 (composition truths + data profile), layer 3
   (numerical values, runtime-only). Diagrams showing that Sweep 8
   lives at the layer-2/3 boundary; Sweep 23 (pipeline compiler)
   lives wholly at layer 2; proof engine is layer 1.

3. **Data profile enters Shape as layer-2 facts.** Already partly
   there via `has_known_non_finite: bool` and `AssumptionTag::*`.
   Plus additions once the data-quality analyzer ships (Sweep 27):
   - `DtypeConfidence` — "known f64" vs "inferred from strings"
   - `ScaleRange` — `(min, max)` from profiling
   - `SparsityRatio` — `f64` in `[0, 1]`
   - `CardinalityHint` — for categorical axis
   - `DistributionShape` — Gaussian / heavy-tailed / bimodal / flat
   These don't need to land in Sweep 8; flag as future Shape
   extensions for Sweep 27. Shape already handles `AssumptionTag::
   Custom(String)` so Sweep 27 can populate via tags while the
   typed fields get designed.

4. **Data-quality warnings surface as `using()` annotations.**
   Per the state-conservation principle: the warning isn't a
   side-channel log; it's an annotation on the pipeline at the
   affected step. Sweep 27 designs the annotation syntax; Sweep 8
   just needs to ensure Shape's assumption_tag bag is open enough
   to hold profile facts.

5. **Proof engine stays fact-oriented.** Document explicitly in
   `proof/mod.rs` that it outputs StructuralFacts, never strategies
   or schedules. Schedulers and lift-deciders consume the facts.

---

## Phase 6/7 — Recursive challenge

- **Q-rec-1.** Does `FutureDependent { description: String }` need
  more structure? E.g., a machine-readable kind so the pipeline
  compiler can reason about it? My answer: Sweep 8 doesn't need it;
  the pipeline-compiler sweep can upgrade to `FutureDependenceKind`
  enum per R2 when it lands. Keep the slot simple for now.

- **Q-rec-2.** What happens if TAM's data profile is WRONG? E.g.,
  TAM thinks no NaN based on the profile, codegen elides the
  validity branch, actual dispatch hits a NaN that slipped past
  the profiler. My answer: this is an INVARIANT VIOLATION, not a
  runtime gracefully-degrading case. The profile's job is to be
  accurate; if it fails, the kernel produces wrong output. The
  profiler must scan the full data (not sample) to claim
  `has_known_non_finite = false`. Or: the flag stays conservative
  (default `unknown`, `known: false` only after exhaustive scan).
  **Default is conservative ("unknown") and the profiler upgrades
  to definite claims only after exhaustive verification.**

- **Q-rec-3.** Can TAM profile data that's streaming / too-large-
  to-fit / privacy-restricted? Streaming: profile-as-you-go,
  refine profile incrementally; each refinement may trigger
  recompile. Too-large-to-fit: sample + stats-only inference with
  wide bounds. Privacy: user provides profile metadata; TAM
  treats as authoritative. All three are Sweep 27 concerns, not
  Sweep 8.

- **Q-rec-4.** Does "data profile" include BUFFER METADATA
  (alignment, stride) or only VALUE-STATISTICS? Both. Alignment is
  profile knowledge; it feeds codegen's choice of aligned-load
  instruction. Value statistics feed assumption tags and method
  picks. Unified under "layer 2 data profile."

- **Q-rec-5.** Does the reframe affect the design anchor for
  pipeline-lift? YES — the pipeline-lift pass consumes layer 2
  knowledge to decide fusions. "Self-reference detection" in the
  original anchor was my earlier naming; **rename in the anchor
  file to "future-dependent control flow detection."** Single-
  line edit.

- **Q-rec-6.** What's the compile-time shape of
  `FutureDependent { description }`? Strings in enums are awkward.
  Maybe `FutureDependent(FutureDependenceMarker)` where marker is
  a lightweight enum with a `Details(String)` variant for
  diagnostic text. Or just leave as `String` for Sweep 8 — a few
  bytes per atom call that hits this case is fine; the case is
  rare. Prefer simplicity; use `String`.

---

## Phase 8 — Forced Rejection

- **What if TAM KNEW the future?** Then every pipeline would be
  liftable (just run all iterations speculatively with foreknowledge
  of the stop condition). The fact that TAM CAN'T is what makes
  sequential a real last-resort. The reframe's negative framing
  ("TAM can't") is structurally load-bearing.

- **What if the three layers COLLAPSED to two?** E.g., if layer 2
  didn't exist (no data profile, no compile-time pipeline analysis).
  Then every data-profile-driven optimization would have to happen
  at runtime, every liftability check would be per-dispatch,
  performance collapses. Layer 2 is where the tambear ambition
  lives — "the compiler knows everything it possibly can before
  dispatch."

- **What if layer 3 never happened?** Fully statically-analyzable
  pipeline (all fixed-count iterations, no data-dependent control
  flow). Then everything lifts. This is the ideal; most real
  pipelines approach it except for specific layers
  (MCMC, Newton, EM, etc.) where layer 3 is forced.

- **What MUST ALSO EXIST that we haven't named?** Ghost-hunting.
  Candidate: **the layer-2 CACHE.** Just as there's a kernel cache
  (layer-1 materialization) and a pipeline-lift schedule cache
  (post-9), there must be a **data-profile cache.** TAM shouldn't
  re-profile the same data every session; the profile fingerprint
  (BLAKE3 of statistical summary, not of raw data) can persist
  alongside the pipeline. Same document-of-truth principle.
  **Three caches at three layers.**

  Layer 1 cache: eternal (proof engine has its own test-validated
  cache of which (Op, grouping) combinations produce which
  Structure; already partly landed in Sweep 1).
  Layer 2 cache: pipeline structure + data profile fingerprint.
  Layer 3 cache: kernel materialization (current Sweep 8 cache).

  Each cache serves one knowledge tier. Invalidation semantics:
  layer-1 cache invalidates never (eternal); layer-2 cache
  invalidates on pipeline edit or data update; layer-3 cache
  invalidates on layer-2 change.

  **Flag for pathmaker / Sweep 27:** the data-profile cache is not
  Sweep 8 scope but the existence of the concept should be
  acknowledged in the architecture doc. Sweep 27 will design it.

- **What if profile-vs-reality divergence were LET BE?** TAM profile
  says no-NaN; actual data has NaN; codegen elides the branch; hot
  path produces wrong answer silently. CATASTROPHIC. The profile
  discipline must be: "default = 'unknown/conservative'; definite
  claims require exhaustive evidence OR user-authored assertion."
  User can assert `using(assume_no_nan = true)` — that's their
  affirmation; if wrong, it's their error. TAM's own profile
  claims must be exhaustively verified. The discipline is: **profile
  says YES only after evidence; user can say YES by assertion.**

- **Does the reframe change what "proof" means?** Hmm. The proof
  engine verifies algebraic claims (Tier A deductive, Tier B
  exhaustive-computational, Tier C structural-statistical). The
  reframe doesn't change that. But: the data profile's "known no
  NaN" claim is *also* a kind of proof (exhaustive-computational —
  scanned every element). Same rigor, different layer. **The proof
  engine could naturally be extended to certify profile claims as
  Tier B proofs of layer-2 data facts.** Elegant unification;
  flag for post-Sweep-27 exploration.

---

## Spec delta — what changes in Sweep 8

1. Rename `SequentialReason::FockBoundary` →
   `SequentialReason::FutureDependent { description: String }`.
   Update doc comment.
2. Update `trait-spec-locked.md` to reference the renamed variant
   and cite `docs/tam-knowledge-layers.md` as the conceptual source.
3. Add `docs/tam-knowledge-layers.md` with the three-layer
   architecture. ~200 lines; a few diagrams.
4. Update `design-anchor-pipeline-lift-substrate-requirements.md` —
   replace "self-reference detection" with "future-dependent CF
   detection" in one line.

No code-structure changes. No new types beyond the rename.

Sweep 27 (data-quality) and Sweep 23 (pipeline compiler) scoped as
follow-on; concepts are in the architecture doc as forward
references.

---

## Cache-bake audit on the reframe

Does renaming `FockBoundary` → `FutureDependent` change any cache
keys? YES — if `ExecutionStrategy` serializes by variant tag (per
the earlier flagged substrate-hygiene item), then the tag string
changes from "fock_boundary" to "future_dependent." Every cached
kernel that had `Sequential { reason: FockBoundary }` in its key
becomes orphaned.

**Mitigation:** DEC-no-backward-compat — the kernel cache
versioning includes a Sweep number. Renaming this variant bumps the
cache-schema version; all cached kernels from pre-rename get
discarded. Acceptable.

**Alternative:** preserve the tag string "fock_boundary" for
backward compat even after the rename. REJECTED per DEC-no-backward-
compat; the cache can recompute. Rename cleanly.

---

## Action items for pathmaker

1. Rename `SequentialReason::FockBoundary` → `FutureDependent`
   with a `description: String` field (or equivalent).
2. Update ExecutionStrategy's `Display` to emit
   "sequential[reason: future_dependent, description: ...]"
   instead of the old Fock form.
3. Round-trip test: parse of `FutureDependent { description: "..." }`
   round-trips.
4. Create `docs/tam-knowledge-layers.md` — I'll draft this;
   pathmaker reviews.

## Action items for me (aristotle)

1. Draft `docs/tam-knowledge-layers.md` (~200 lines). Architecture
   doc; canonical reference for the three-layer knowledge concept.
2. Update design-anchor-pipeline-lift-substrate-requirements.md —
   one-line substitution of terminology.
3. Update principle-decisions-as-using-annotations.md to note that
   data-quality warnings ARE using-annotations (T5 above).
4. Garden: "the liberating negative formulation" — future-me will
   want the note about how "TAM can't tell the future" is a
   STRONGER design principle than "TAM is omniscient except for X,
   Y, Z" because the negative constraint focuses the design on
   what's genuinely impossible vs what's hard.

---

## Adversarial re-direction

Earlier attacks for adversarial to update:

- **Old wording:** "find an adversarial (Op, Shape, Grouping) pair
  where DeterminismClass cannot be reduced to {Deterministic,
  OrderDependent, NonDet}."
  **New phrasing:** now with three-layer knowledge framing: are
  there scenarios where the DeterminismClass is itself FUTURE-
  DEPENDENT (i.e., the determinism of the kernel depends on
  values TAM hasn't seen yet)? Example candidate: Probabilistic
  grouping with learned RNG seed from a prior dispatch — the
  determinism claim would need layer 3 to verify. Real?

- **Old wording:** "Validity::Error race under async dispatch."
  **New phrasing:** with data-quality analyzer, does Validity::Error
  ever FIRE in dispatch when the compile-time profile said
  `has_known_non_finite: false`? Only if the profile was wrong
  (violates Q-rec-2 invariant). Attack: construct a case where
  profile claim + actual data diverge and see what breaks.

Standing attacks preserved; reframing sharpens them.
