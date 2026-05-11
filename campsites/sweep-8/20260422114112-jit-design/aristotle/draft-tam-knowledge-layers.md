# DRAFT — `docs/tam-knowledge-layers.md`

**Status:** Draft for review. Will land in `R:\tambear\docs\` once
pathmaker + team-lead sign off. Authored by aristotle, 2026-04-22,
extracted from the "TAM can't tell the future" reframe.

---

# TAM Knowledge Layers

> *TAM knows everything before any compute runs, except the future.*

This document names the canonical architecture of TAM's knowledge.
It is referenced by:
- `docs/architecture.md`
- `crates/tambear/src/jit/strategy.rs` (sequential-reason rationale)
- `crates/tambear/src/jit/shape.rs` (Shape's role at layer 2)
- The future Sweep 23 (pipeline compiler) and Sweep 27 (data quality
  analyzer) READMEs

Read this before designing any tambear feature that crosses
compile/dispatch boundaries. Feature designs that violate the
three-layer separation produce architectural drift.

---

## The three layers

```
┌────────────────────────────────────────────────────────────────┐
│  LAYER 1 — ETERNAL TRUTHS                                      │
│  Pre-pipeline. Pre-everything. The mathematical facts about    │
│  Ops, Groupings, atoms — algebraic structure, monoid laws,    │
│  liftability eligibility, kingdom classification.              │
│  Source: proof engine (`crates/tambear/src/proof/`).           │
│  Cache: long-lived; invalidates only when proof engine itself  │
│  is updated (rare).                                            │
├────────────────────────────────────────────────────────────────┤
│  LAYER 2 — COMPOSITION TRUTHS                                  │
│  Pre-dispatch. Everything TAM can know from the                │
│  pipeline-plus-data-plus-hardware triple before any actual     │
│  numerical compute runs.                                       │
│   - Pipeline structure: which recipes, in what order,          │
│     which data bindings, which hardware doors                  │
│   - Data profile: dtype inference, NaN/Inf count, scale,       │
│     sparsity, distribution shape, cardinality                  │
│   - Hardware capability: ISA version, available SIMD width,    │
│     available memory, denormal mode                            │
│   - Op-composition facts: "step 7 produces f64", "step 35      │
│     needs an f64 input compatible with step 7's output",       │
│     "shared intermediates between steps A and B match under    │
│     assumption tag X"                                          │
│  Source: pipeline compiler (Sweep 23+) + data quality          │
│  analyzer (Sweep 27+) + door capability queries (Sweep 8).     │
│  Cache: per-pipeline + per-data-profile + per-hardware.        │
├────────────────────────────────────────────────────────────────┤
│  LAYER 3 — NUMERICAL TRUTHS                                    │
│  Only after dispatch. The actual values that emerge.           │
│  TAM cannot predict these from layers 1 or 2.                  │
│  Source: dispatch results.                                     │
│  Not cacheable as values (the result IS the value); cacheable  │
│  only as feedback to refine layer-2 profiles.                  │
└────────────────────────────────────────────────────────────────┘
```

## What this categorization GAINS us

1. **Crisp lift/sequential boundary.** Sequential codegen is needed
   only when control flow (branching, iteration count, stop
   condition) requires a value that lives in layer 3.

   Liftable: layer-1 algebra + layer-2 pipeline structure together
   determine the entire CFG.

   Sequential: at least one CFG decision reads a value not yet
   computed.

2. **Honest separation of compile-time from runtime.** Compile-time
   = layers 1 + 2. Runtime = layer 3. Anything that wants to "be
   compile-time" must derive purely from layers 1 + 2.

3. **Per-layer caching strategy.** Each layer has its own cache:
   layer 1 (proof engine; long-lived); layer 2 (pipeline +
   profile; refresh on edit); layer 3 (kernel materialization;
   refresh on (layer-1, layer-2) change). Three caches, three
   invalidation policies.

4. **Per-layer authority.** Algebraic claims belong in the proof
   engine (layer 1). Pipeline structure + data profile claims
   belong in the pipeline compiler / data analyzer (layer 2).
   Numerical claims are runtime artifacts. Mismatch = architectural
   bug.

5. **Per-layer testing.** Layer 1 tests = proof engine + bit-exact
   oracle suites. Layer 2 tests = pipeline structure validators,
   data profile correctness, share-compatibility transitivity.
   Layer 3 tests = full integration / end-to-end.

## The negative formulation

> **TAM cannot tell the future.**

This is the only structural limitation. Every other restriction
TAM appears to have (algebraic non-liftability, kingdom-B
classification, schedule conflicts) reduces to layer 1 or layer 2
gaps that further analysis can in principle close. The future-
dependence boundary is genuinely irreducible — even speculative
execution, in the limit, hits a probability where the speculation
is wrong and we've spent more compute than sequential would have.

This negative formulation is stronger than any positive list of
"TAM knows X, Y, Z." Use it when explaining the architecture.

## Examples

### Liftable (layer 1 + 2 sufficient)

- **Sum reduction over an array.** Layer 1: Add is a commutative
  monoid (proof engine). Layer 2: array length is N (known). Total
  iteration count is N; CFG is straight-line scan. Lift via
  parallel reduction.

- **EWMA prefix scan.** Layer 1: AffineCompose is associative
  (proof engine). Layer 2: array length N, alpha known from
  using(). CFG is straight-line N-step scan. Lift via prefix scan.

- **Kalman filter forward pass.** Layer 1: state-update is affine
  (proof engine). Layer 2: number of observations N is known. CFG
  is N-step prefix scan. Lift via AffineCompose / MatMulPrefix.

- **PCA via top-k eigenvectors.** Layer 1: power iteration converges
  to dominant eigenvector (proof engine). Layer 2: k is known
  (using()), max iterations is configurable (using() default).
  CFG: outer loop is k iterations, inner loop is fixed-max-iter
  with deflation. Lift inner via the substrate primitives.

### Sequential (requires layer 3)

- **Newton root-finding with `||x - x'|| < ε` stop.** Layer 1:
  Newton iteration is well-defined per step. Layer 2: starting
  point known, ε known. **Layer 3: the stop condition reads the
  current iteration's output.** Number of iterations is unknown
  in advance. Sequential.

- **EM with log-likelihood convergence check.** Same shape:
  iteration count depends on the just-computed log-likelihood.
  Sequential.

- **MCMC with Gelman-Rubin diagnostic stopping.** Iteration count
  depends on the chain's variance. Sequential.

- **Adaptive ODE solver with error-controlled step size.** Each
  step's size depends on the current step's local truncation
  error. Sequential per step.

- **Branch-and-bound optimization.** Pruning decisions depend on
  the current best-bound, which depends on prior values.
  Sequential.

### Boundary cases

- **Fixed-N Newton iteration.** If we say "always do 100 Newton
  iterations regardless of convergence," it lifts (CFG is determined
  layer-2). The user gives up the early-termination benefit but
  gains parallelism. This is a `using(strategy=lifted_fixed_N)`
  override on a normally-sequential method.

- **Speculative execution of stop-condition iterations.** Run K
  iterations in parallel, check convergence after; if converged
  at iteration 3, discard 4-K. Combines layer-2 speculation with
  layer-3 truth. Power tool; not in Sweep 8 scope.

## The data profile is layer 2

A common confusion: data is runtime, so the data profile must be
runtime, right?

No. The data **values** are runtime. The data **profile** —
statistical summary, NaN count, dtype inference, distribution
shape — is computed when the data is bound to the pipeline
(before any user-recipe runs) and lives at layer 2.

The Shape struct's `assumption_tags`, `has_known_non_finite`, and
future profile-augmented fields are all layer-2 artifacts. Codegen
specializes on them.

## The using-annotation surface is the layer-2 visibility channel

Per the using-annotation principle (separate doc): every layer-2
decision TAM makes — chosen execution strategy, chosen Adaptive
boundary, chosen share-source, inserted preprocess step, chosen
`using()` method, data-quality warning — surfaces in the pipeline
source as a `using()` annotation with rationale.

This makes layer 2 fully transparent to the user. The user reads
their pipeline file and sees both their authoring and TAM's
contributions.

Layer 1 facts (algebraic) are NOT typically annotated — they're
considered eternal truths that the user trusts the proof engine on.
Layer 3 numerical results are output, not annotation.

## Caching at three layers

| Layer | Cache content | Invalidation trigger |
|---|---|---|
| 1 | StructuralFacts per Op + Structure validations | proof engine release / never |
| 2 | Pipeline + profile + door-capability fingerprint → annotated schedule | pipeline edit, data update, capability change |
| 3 | (no value cache; results are outputs) | n/a |

Plus the **kernel materialization cache** (Sweep 8G), which is a
materialization of "given a layer-2 schedule, here's the door-
specific binary." This is keyed on the layer-2 schedule's
fingerprint plus door identity.

## Implications for Sweep 8 and forward

- Sweep 8 substrate (DoorBackend, Shape, JitOp, ExecutionStrategy,
  CacheKey) lives at the **layer-2/3 boundary**. The layer-3 side
  is the kernel cache; the layer-2 side is everything that feeds
  into the cache key.

- Sweep 9+ math expansion: pure layer 1 work. The proof engine
  certifies new Op/Grouping combinations as algebraic facts;
  Sweep 8 substrate consumes them.

- Sweep 23 (pipeline compiler): pure layer 2 work. Reads pipeline
  source + data profile + hardware capability; emits annotated
  schedule. Calls into Sweep 8 substrate to materialize kernels.

- Sweep 27 (data quality): pure layer 2 work. Profiles user data
  on attach; populates Shape's assumption_tags + future-extended
  profile fields; surfaces warnings as using-annotations.

- Sweeps 24/25/26 (IDE/TBS, academic write-up, hardware
  visualization): all consume layer-2 state and render it. Not
  layer 1 or 3.

## Glossary

- **Layer 1 / eternal truths:** algebraic facts about Ops,
  Groupings, atoms; proof engine output. Not data-dependent.
- **Layer 2 / composition truths:** facts derivable from
  pipeline-plus-data-plus-hardware before any numerical compute
  runs.
- **Layer 3 / numerical truths:** the actual values that emerge
  from dispatch.
- **TAM knowledge:** the union of layers 1 and 2; everything TAM
  can use to make compile-time decisions.
- **Future-dependent control flow:** a CFG decision that requires
  a layer-3 value; the only genuine forcer of sequential codegen.
- **Data profile:** statistical summary of user data (dtype,
  NaN count, scale, sparsity, distribution); a layer-2 artifact
  populated at pipeline-attach time, NOT at dispatch.

## See also

- `docs/decisions.md` — DEC-019 (native-door JIT) for the
  layer-3 hardware boundary.
- `docs/vocabulary.md` — Tier 1-5 terminology; this knowledge-
  layer doc is orthogonal to the tier hierarchy.
- `crates/tambear/src/proof/mod.rs` — proof engine (layer 1
  source).
- `crates/tambear/src/jit/strategy.rs` — `SequentialReason::
  FutureDependent` references this doc for the canonical
  formulation.

## Authorship + revision

Authored by aristotle 2026-04-22 from the "TAM can't tell the
future" reframe (team-lead/Tekgy DM). Lands in tambear/docs/ once
pathmaker + team-lead sign off.

Revisions:
- 2026-04-22 — initial draft.
