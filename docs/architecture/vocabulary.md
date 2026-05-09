# Tambear Vocabulary — Canonical Reference

**Status:** Locked. This is the single source of truth for tambear terminology.

**Locked:** 2026-04-17

**Read this before any other architecture document.** Older docs across the
repo (CLAUDE.md notes from earlier dates, `tambear-truths.md`, project memory
files, campsite logs, expedition writeups) may use words like *primitive*,
*method*, *specialist*, *operation*, *layer*, *kingdom*, *atom* with meanings
that differ from the locked vocabulary below. Where any document conflicts
with this one, this document wins.

---

## The five tiers

Every named thing in tambear lives at exactly one of these five tiers.

```
┌────────────────────────────────────────────────────────────────────┐
│  Tier 5  —  PIPELINES                                              │
│  Top layer. A user's full project: hardware bindings, data         │
│  bindings, recipes used, parameter overrides, interaction surface  │
│  (TBS / GUI / formal notations / script / mix). Compiles down to   │
│  a `.tam` IR + per-pass per-door kernel binaries.                  │
├────────────────────────────────────────────────────────────────────┤
│  Tier 4  —  RECIPES                                                │
│  Every named composition. Flat. No nesting. One thing per file.    │
│  Examples: mean, variance, correlation, log, exp, sin, sqrt,       │
│  parkinson_volatility, kyle_lambda, vpin, max_drawdown, efa,       │
│  garch_fit, sip_signal_bundle. NO inline math beyond composition.  │
│  Default parameters are tambear-best-practice; users override via  │
│  using().                                                          │
├────────────────────────────────────────────────────────────────────┤
│  Tier 3  —  ATOMS                                                  │
│  Two operations.                                                   │
│  accumulate(grouping, expr, op)                                    │
│  gather(addressing, ...)                                           │
│  Everything compiles down to these.                                │
├────────────────────────────────────────────────────────────────────┤
│  Tier 2  —  OP and EXPR                                            │
│  IEEE-level dials that parameterize the atoms.                     │
│  Op:    Add, Max, Min, ArgMax, ArgMin, DotProduct, Distance, ...   │
│  Expr:  Value, ValueSq, Custom("v*w"), ...                         │
│  Live as enums inside crates/tambear/src/accumulate.rs.            │
│  No folders, no separate files — small fixed sets.                 │
├────────────────────────────────────────────────────────────────────┤
│  Tier 1  —  PRIMITIVES                                             │
│  Low-level implementation machinery. The non-decomposable IEEE-    │
│  level building blocks plus the compensated-arithmetic foundations │
│  the compiler uses to lower Op/Expr into actual instructions.      │
│  Categories:                                                       │
│   - hardware/    : fadd, fmul, fmadd, fsqrt, fmin, fmax, ...       │
│   - compensated/ : two_sum, two_product_fma, kahan_sum, ...        │
│   - double_double/ : DoubleDouble type and its operations          │
│   - specialist/  : Kulisch accumulator, sum_k, etc.                │
│   - oracle/      : test infrastructure                             │
│   - constants/   : PI, E, LN_2 in multiple precisions              │
│  Lives at crates/tambear/src/primitives/.                          │
│  Usually not user-facing — recipes call these via Op/Expr/atom.    │
└────────────────────────────────────────────────────────────────────┘
```

**The compiler lowers everything to atoms.** A pipeline → its recipes →
the atoms those recipes call → the Op/Expr those atoms take → the
primitives that implement Op/Expr on the chosen backend. Anything that
escapes this chain is a vocabulary violation.

---

## Tier-by-tier definitions

### Tier 5 — Pipelines

A **pipeline** is the user's project. It captures:

- Which recipes are used
- The order they are called in (and any data dependencies between steps)
- Hardware constraints (allowed surfaces, throttles, usage limits)
- Data bindings (which input data flows into which recipe; column mappings)
- Parameter overrides (`using(...)` calls per recipe)
- The interaction surface that authored it (TBS syntax, GUI clicks, formal
  notations, script, or a mix)

Pipelines compile via the tambear compiler to two artifacts:

1. A **`.tam` IR** — orchestration plan that TAM (the runtime distributed
   compute scheduler) executes.
2. **Per-pass per-door kernel binaries** — actual compiled compute kernels,
   one per "pass" (maximal accumulate+gather fusion of pipeline steps),
   rendered for every door (CUDA, Vulkan, DX12, Metal, AMD, Intel, CPU).

A pipeline does NOT have inline arithmetic. It composes recipes.

### Tier 4 — Recipes

A **recipe** is any named mathematical operation. Every recipe is a
composition. Recipes never inline arithmetic that bypasses the atom layer
— if a recipe needs `x + y`, that's an `Op::Add` through `accumulate`, not
a Rust `+` operator.

Recipe rules:

- **Flat.** One file per recipe. No nesting by domain. Tags as metadata
  for multi-family membership (a recipe can belong to "statistics" AND
  "microstructure" AND "estimator" simultaneously).
- **Pure composition.** Body contains only calls to atoms, primitives
  (via Op/Expr), or other recipes. No inline math.
- **Default parameters are tambear-best-practice.** What a senior expert
  would pick if not told otherwise. Documented.
- **Every parameter overridable via `using()`.** Users opt out of the
  default at the call site without modifying the recipe.
- **Has an oracle test.** Compares against a high-precision reference
  (mpmath, Kulisch, closed-form). Adversarial inputs included.
- **Honestly declares its Kingdom (A/B/C/D)** so TAM knows how to
  schedule it. Most recipes are A. Some are C(A). Genuine B/D is rare
  and flagged.

Examples:

- Statistical: `mean`, `variance`, `std_dev`, `correlation`, `dot`,
  `covariance`, `kendall_tau`, `pearson_r`, `spearman_r`, `quantile`
- Transcendentals: `exp`, `log`, `sin`, `cos`, `tan`, `asin`, `atan2`,
  `erf`, `erfc`, `gamma`, `log_gamma`
- Microstructure: `parkinson_volatility`, `roll_spread`, `kyle_lambda`,
  `vpin`, `amihud_illiquidity`, `lee_mykland_jump_count`,
  `hawkes_intensity`
- Risk: `max_drawdown`, `cvar`, `hill_estimator`, `realized_vol_subsampled`
- Time series: `garch_filter`, `arma_filter`, `dfa`, `adf_test`,
  `jarque_bera`, `cusum`, `bocpd`
- Multivariate: `pca`, `factor_analysis`, `efa`, `lda`, `cca`
- Sketches: `ddsketch` (locked default — relative-value error,
  bit-exact distributed merge, no internal sorts, permutation-
  invariant — required for SIP's Merkle-anchored chain),
  `kll_sketch` (rank-error mergeable; `using(sketch: "kll")`),
  `tdigest` (best empirical tail accuracy; `using(sketch: "tdigest")`),
  `gk_sketch` (rank-error intrinsically mergeable; `using(sketch: "gk")`)
- Pipelines-as-recipes: `efa`, `garch_fit`, `sip_signal_bundle`,
  `two_group_comparison`, `regression_diagnostics` — these are also
  recipes; "pipeline" in this list means "multi-step composition," not
  "Tier 5 pipeline-the-user-project."

**No "method" tier.** Older docs distinguish "primitive" (small named math)
from "method" (orchestration). Under the locked vocabulary there is no
such distinction. Both are recipes. A recipe is one file, possibly large,
possibly small.

**No "specialist" tier.** Older docs use "specialist library" for what we
now call recipes. Same concept, different word. The folder
`crates/tambear/src/primitives/specialist/` is unrelated — it holds
implementation primitives (Kulisch, sum_k) that recipes consume.

### Tier 3 — Atoms

There are exactly two atoms:

```rust
accumulate<G: Grouping, E: Expr, Op: Combiner>(
    grouping: G,
    expr: E,
    op: Op,
    data: &[f64],
) -> AccResult

gather<A: Addressing>(
    addressing: A,
    source: &[f64],
) -> Vec<f64>
```

`accumulate` is the universal scatter / reduce / scan / prefix-scan operation.
`gather` is the universal indexed read operation.

**Scatter is a specialization of `accumulate`**, not a separate atom.
A by-key scatter is `accumulate(data, Grouping::ByKey{keys}, Op::Add)`.
Older docs sometimes list "scatter" as a third atom — incorrect under the
locked vocabulary.

The atoms take three families of parameters:

- **Grouping** — how data partitions:
  `All`, `ByKey(keys)`, `Prefix`, `Windowed(w)`, `Segmented(bounds)`,
  `Strided(s)`, `Tiled(m,n)`, `Circular(period)`, `Graph(adjacency)`
- **Op** — how to combine within a group (see Tier 2)
- **Expr** — what to compute per element before combining (see Tier 2)

For `gather`:

- **Addressing** — how to read:
  `ByIndex`, `Shuffle(perm)`, `KnnNeighbors(k)`, `GridLookup(coords)`,
  `Offset(delta)`, ...

The atoms are the universal substrate. Every Kingdom A computation —
which after compile-time Fock-raising includes most of mathematics — runs
through these two operations.

### Tier 2 — Op and Expr

**Op** is the combine operation a `Grouping` uses. The set is small and
fixed. Lives as a Rust enum inside `crates/tambear/src/accumulate.rs`.
Variants:

- `Add` — additive monoid `(ℝ, +)`. Default strategy: Kulisch-backed exact
  accumulation. Cross-platform bit-exact deterministic.
- `Max`, `Min` — IEEE 754-2019 fmax / fmin. Idempotent; deterministic by
  construction.
- `ArgMax`, `ArgMin` — value+index pair-reduction with lowest-index
  tiebreak.
- `DotProduct` — Kulisch over `two_product_fma`. For tiled grouping.
- `Distance` — Kulisch over centered squared differences. For tiled
  grouping.
- *(future: `LogSumExp`, `TropicalMinPlus`, `TropicalMaxPlus` as needed)*

**Expr** is the per-element function the atom applies before combining.
Also a small fixed enum in `accumulate.rs`. Variants like `Value`,
`ValueSq`, `WeightedByRef`, `Custom(formula)`.

**Determinism contract:** every Op is cross-platform bit-exact deterministic
by default. Same input → same bits regardless of thread count, execution
order, backend, or CPU architecture. Non-determinism is opt-in only via
`using(sum_strategy: "nondet")`. See `2026-04-11-op-default-deterministic-plan.md`
for the full Op contract.

**NaN/Inf policy:** two independent knobs, both default to `"propagate"`
(IEEE 754-aligned). Consumer opt-in to skip via
`using(nan_policy: "skip", inf_policy: "skip")`. SIP uses skip; general
tambear math uses propagate. See the determinism plan for the full table.

### Tier 1 — Primitives

A **primitive** is low-level implementation machinery. Either a single
IEEE 754 hardware instruction wrapper, or an error-free transformation /
compensated-reduction whose internal sequence of hardware ops is treated
as a unit by the compensated-arithmetic literature.

**Primitives are usually not user-facing.** The compiler uses them to
lower Op and Expr into concrete instruction sequences per backend.
Recipes call atoms with Op/Expr parameters; the compiler translates that
to primitive calls during lowering.

Categories (each lives flat within its subfolder under
`crates/tambear/src/primitives/`):

- **hardware/** — IEEE 754 hardware ops:
  `fadd, fsub, fmul, fdiv, fsqrt, fmadd, fmsub, fnmadd, fnmsub,
   fabs, fneg, fcopysign, fmin, fmax, fcmp_*, is_nan, is_inf,
   is_finite, signbit, frint, fround_ties_even, ffloor, fceil, ftrunc`
- **compensated/** — error-free transformations and core compensated ops:
  `two_sum, fast_two_sum, two_product_fma, two_diff, two_square,
   kahan_sum, neumaier_sum, pairwise_sum, dot_2, compensated_horner,
   fma_residual`
- **double_double/** — `DoubleDouble { hi, lo }` type and its ~8 operations
  (add, sub, mul, div, sqrt, conversions, mixed ops with f64).
- **specialist/** — built on demand: `kulisch_accumulator` (~4350-bit
  exact integer accumulator with `merge` for parallel reduction),
  `sum_k(data, k)`, `dot_k(x, y, k)`, `priest_sum`, etc.
- **oracle/** — shared test infrastructure (algorithm-properties catalog,
  reference-implementation harnesses).
- **constants/** — mathematical constants in multiple precisions (`PI`,
  `E`, `LN_2`, `SQRT_2` as f64 and DoubleDouble pairs).

The stopping rule for decomposition: **a recipe stops decomposing when
it bottoms out at a primitive.** Primitives have no further decomposition
visible above the hardware layer.

---

## How the layers compose at runtime

When a user authors a Tier 5 pipeline and hits RUN, the compiler walks
the composition tree:

1. **Pipeline** is parsed into the `.tam` IR — a structured description
   of orchestration: which recipe runs on which surface, data flow,
   barriers, throttles, what to do with results.

2. **Each recipe** in the pipeline is decomposed into its atom calls
   plus any other recipes it calls. The decomposition continues
   recursively until everything is atom-level.

3. **Each atom call** carries its Op/Expr/Grouping parameters. The
   compiler emits a kernel that fuses every atom call sharing a common
   grouping into a single pass.

4. **Each kernel** is rendered per-door — same arithmetic spec, different
   compiled binary per backend (CUDA PTX, Vulkan SPIR-V, DX12 DXIL,
   Metal `metallib`, native CPU). Inside each kernel, Op and Expr lower
   to primitive calls (Kulisch for Add, IEEE fmax for Max, etc.).

5. **TAM** (the runtime) reads the `.tam` IR and dispatches kernels
   through doors to ALUs. Per-surface partial results merge bit-identically
   via Kulisch merge (the partial-merge operation is associative by
   construction). Final results stream to the GUI / disk / next pass.

The whole chain is bit-exact deterministic. Same pipeline + same input
data + any valid combination of available surfaces → identical output
bits. This is what makes the `.tam` IR's cryptographic fingerprint
useful: it commits to the exact bit pattern that any valid execution
will produce.

---

## Vocabulary mapping — old terms → locked terms

When reading older docs (CLAUDE.md historical sections, `tambear-truths.md`,
project memory files, campsite logs from 2026-04-10 through 2026-04-16),
translate their terms to the locked vocabulary as follows:

| Older term | What was usually meant | Locked tier |
|---|---|---|
| "primitive" (in user-facing math sense) | a named math operation: mean, correlation, kendall_tau | **Recipe** (Tier 4) |
| "primitive" (in implementation sense) | hardware op or compensated arithmetic helper | **Primitive** (Tier 1) |
| "method" | an orchestrating composition of "primitives" | **Recipe** (Tier 4) |
| "specialist" | a named composition like a recipe | **Recipe** (Tier 4) |
| "Layer 0 — math primitives" (CLAUDE.md "Layers Above the Math") | the user-facing math tier | **Recipe** (Tier 4) |
| "Layer 1 — diagnostics" (auto-method-selection) | a higher-level recipe that picks among recipes | **Recipe** (Tier 4) — with a layer tag |
| "Layer 2 — using() override transparency" | a property of recipe behavior, not a tier | not a vocabulary tier; it's a runtime property |
| "Layer 3 — expert pipelines" | named multi-step recipes | **Recipe** (Tier 4) — `efa`, `garch_fit`, etc. |
| "Layer 4 — discover/superposition" | a property of recipe execution mode | not a tier; runtime mode |
| "atom" | accumulate, gather (correct usage) | **Atom** (Tier 3) |
| "operation" / "two operations" | accumulate, gather | **Atom** (Tier 3) |
| "scatter" (as a third atom) | a kind of accumulate | **Atom** (Tier 3) — `accumulate(..., Grouping::ByKey, ...)` |
| "kingdom" (A/B/C/D) | parallelizability classification | unchanged — Kingdom is a property of a recipe, orthogonal to the tier system |
| "fock boundary" | the limit between Kingdom A (parallelizable) and Kingdom B (genuinely sequential) | unchanged |
| "menu" / "menu choice" | parameter values for accumulate (grouping/expr/op/addressing) | the parameters at Tier 2 + Grouping (which is at Tier 3 as part of the atom signature) |

When an old document says "primitive" the locked translation depends on
context:

- "the `inversion_count` primitive" → that's a **recipe**
- "the `fmadd` primitive" → that's a **primitive** (matches locked tier)
- "the primitives layer" → in old docs this often meant the user-facing
  math tier (now **recipes**); in newer docs it means the implementation
  tier (correct, **primitives**). Read carefully and pick the right
  mapping.

---

## What stays the same

These concepts are NOT vocabulary changes; they keep their meaning across
all tambear documentation:

- **Kingdom A / B / C / D** — recipe classification by parallelizability
- **Fock boundary** — limit of parallelizability; raised by the compiler
  via the classification-bijection / three-criteria test
- **TAM** — the distributed compute scheduler / runtime that reads `.tam`
  IR and dispatches across surfaces. Mascot: cute bear above the ice.
- **TamSession** — the intermediate-sharing system that lets recipes
  share computed values (covariance matrix, distance matrix, FFT, etc.)
  with downstream recipes
- **`using()`** — the override mechanism for parameters at any depth in
  a composition
- **`.discover()`** — the superposition/multi-method execution mode
- **`.tam` IR** — the compiler's intermediate representation
- **MKTC / MKTF** — the on-disk file formats (separate from compute
  vocabulary)
- **Three Standing Constraints** (judgment-free, co-native, expansionist)
  — design values, not tiers

---

## Where vocabulary is canonically referenced

When in doubt, in this order:

1. **This document** — `R:\winrapids\docs\architecture\vocabulary.md` (you
   are here). Final authority on terminology.
2. **`atoms-primitives-recipes.md`** — same folder. Canonical reference
   for how the Tier 1-4 decomposition works in practice with code
   examples and the lowering strategies.
3. **`adding-a-recipe.md`** — same folder. Procedural playbook for adding
   new recipes; uses the locked vocabulary throughout.
4. **`R:\winrapids\CLAUDE.md`** — project-wide tambear contract and
   irrevocable principles. Uses the locked vocabulary throughout.

If any document conflicts with this one, this one wins.

---

## Principles that shape the vocabulary

Why the tiers are arranged this way:

- **The compiler must be able to lower any recipe to atom calls.** If a
  recipe inlines arithmetic (a Rust `+` operator on f64 values, a
  `.iter().sum()` call), the compiler can't see it as an atom invocation
  and the lowering breaks. NO inline math in recipes — full stop.
- **The atom layer is the universal substrate.** Two operations is
  enough because everything Kingdom A reduces to scan/reduce + indexed
  read. The compiler raises apparent Kingdom B to A wherever possible.
- **Op and Expr are small fixed sets.** They live as enums, not folders,
  because the cost of adding a new variant (forking dispatch logic) is
  high and the variants represent fundamental operations. New behaviors
  go through `using()`, not through new Op variants.
- **Primitives are flat within categories** because they're terminal —
  no further decomposition. Categorization (hardware / compensated /
  double_double / specialist) reflects implementation kind, not
  abstraction level.
- **Recipes are flat across all of math** because the value of a recipe
  is its name and signature, not its filesystem location. Multi-family
  membership via tags (a recipe can be in "statistics" AND
  "microstructure" AND "moments_consumer" simultaneously) is impossible
  with hierarchical folders. Flat with tags is the only structure that
  preserves multi-composability.
- **Pipelines compose recipes.** The user's project is a pipeline. The
  IDE renders pipelines as step-card sequences. RUN compiles a pipeline
  to a `.tam` plan plus per-pass kernel binaries.

---

## Contact / change process

The vocabulary is locked. Changes require:

1. Explicit user agreement in a session
2. Update to this document FIRST
3. Update to `atoms-primitives-recipes.md` to match
4. Annotation banner added to all docs that now drift
5. Memory entry added to `~/.claude/projects/R--winrapids/memory/` recording
   the vocabulary change with date and rationale

If a session encounters apparent conflicts, the resolution is: this
document wins, period. Do not invent reconciliations; ask the user.
