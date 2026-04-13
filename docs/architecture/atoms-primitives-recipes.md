# Atoms, Primitives, and Recipes

**Status**: Load-bearing architecture. Every recipe written from this point forward follows this structure.

**Relationship to CLAUDE.md**: This document extends the Tambear Contract and "Methods Are Compositions" principle with a concrete three-layer decomposition. CLAUDE.md establishes *why* methods decompose into primitives. This document establishes *what* the decomposition looks like in the filesystem, in the type system, and in the compilation pipeline.

---

## The three layers

Every piece of mathematics in tambear lives at exactly one of three layers:

```
┌────────────────────────────────────────────────────────────┐
│  Layer 3  —  RECIPES                                       │
│  Named compositions. Everything with a literature name.    │
│  pearson_r, exp, sigmoid, brusselator_rhs, kendall_tau     │
│  All recipes decompose to primitives via the two atoms.    │
├────────────────────────────────────────────────────────────┤
│  Layer 2  —  PRIMITIVES                                    │
│  Terminal operations. Hardware ops + compensated           │
│  arithmetic foundations. ~35 total.                        │
│  fmul, fmadd, fsqrt, two_sum, two_product_fma, ...         │
├────────────────────────────────────────────────────────────┤
│  Layer 1  —  ATOMS                                         │
│  The two orchestration operations.                         │
│  accumulate(grouping, expr, op)                            │
│  gather(addressing, ...)                                   │
└────────────────────────────────────────────────────────────┘
```

**Atoms** are the irreducible orchestration primitives from the Tambear Contract. There are exactly two, parameterized by slots (grouping, expr, op, addressing).

**Primitives** are terminal operations — single hardware instructions or small sequences of hardware instructions that form the compensated arithmetic vocabulary. They have no internal decomposition visible to higher layers.

**Recipes** are named compositions of primitives and other recipes. They have a name, a formula, and a tree-structured decomposition that bottoms out at the primitive layer.

**The compositional rule**: a recipe at any layer can call atoms, primitives, and lower-level recipes. A recipe cannot inline arithmetic that isn't expressed through atoms and primitives. If a recipe contains `x * y + z` as raw Rust, that `*` and `+` are implicit calls to `fmul` and `fadd` at the primitive layer. We make those calls explicit so the recipe tree is complete.

---

## Layer 1 — Atoms

See CLAUDE.md "Methods Are Compositions" and "TBS is the universal surface" for the full treatment. The atoms are:

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

Slot values (specific groupings, exprs, ops, addressings) are recipes, not atoms. The atoms are abstract shapes; slot values fill them with concrete mathematics.

Example slot values:
- **Groupings**: `All`, `ByKey(fn)`, `Prefix`, `Windowed(w)`, `Segmented(bounds)`, `Strided(s)`, `Tiled(m,n)`, `Circular(period)`, `Graph(adjacency)`
- **Ops**: `Add`, `Max`, `Min`, `Mul`, `Concat`, `AffineCompose`, `WelfordMerge`, `LogSumExp`, `TropicalMinPlus`, ...
- **Exprs**: `Identity`, `Square`, `Log`, `polynomial_ode_rhs`, `kernel_eval`, `sigmoid`, ...
- **Addressings**: `ByIndex`, `Shuffle(perm)`, `KnnNeighbors(k)`, `GridLookup(coords)`, ...

Each slot value is itself a recipe — a named composition whose role is "I plug into the grouping/expr/op/addressing slot of an atom."

---

## Layer 2 — Primitives

Primitives are terminal recipes: they have no decomposition visible above the hardware layer. The primitive list is small, finite, and nearly complete as of this document.

### Category A — IEEE 754 hardware operations (~20)

These are the operations every target CPU, GPU, and SPIR-V implementation supports as single instructions. They are our portability floor.

```
Arithmetic:       fadd, fsub, fmul, fdiv, fsqrt
Fused:            fmadd, fmsub, fnmadd, fnmsub
Unary:            fabs, fneg, fcopysign
Min/Max:          fmin, fmax        // IEEE 754-2019 minNum/maxNum semantics
                                    // NaN-PROPAGATING, unlike Rust's f64::min/max
Comparison:       fcmp_eq, fcmp_lt, fcmp_le, fcmp_gt, fcmp_ge
Classification:   is_nan, is_inf, is_finite, signbit
Rounding:         frint, fround_ties_even, ffloor, fceil, ftrunc
```

**Critical correctness note**: Rust's `f64::min(NaN, x) = x` and `f64::max(NaN, x) = x` are NOT IEEE 754-2019 compliant. 11 bugs in the 2026-04-10 session came from this. Our `fmin`/`fmax` primitives MUST propagate NaN. Recipes MUST call `fmin`/`fmax` from the primitive layer, NEVER `f64::min`/`f64::max` directly. This eliminates the entire bug class structurally.

### Category B — Compensated arithmetic foundations (~13)

These primitives implement the error-free transformations and core compensated operations that recipes can use to achieve higher effective precision than straight f64. They are terminal at the semantic level: a recipe using `two_sum` does not "see" the underlying `fadd` sequence.

**Tier 1 — Essentials (6 primitives):**
```
two_sum(a, b) -> (f64, f64)              // Knuth, 6 flops, exact a + b
fast_two_sum(a, b) -> (f64, f64)         // Dekker, 3 flops, requires |a| >= |b|
two_product_fma(a, b) -> (f64, f64)      // 2 flops with FMA, exact a * b
two_diff(a, b) -> (f64, f64)             // two_sum(a, -b)
two_square(a) -> (f64, f64)              // two_product_fma(a, a)
kahan_sum(data) -> f64                   // Compensated vector sum
```

**Tier 2 — libm foundations (5 primitives):**
```
neumaier_sum(data) -> f64                // Improved Kahan for mixed magnitudes
pairwise_sum(data) -> f64                // Tree reduction, O(log n) error growth
dot_2(x, y) -> f64                       // Rump-Ogita-Oishi compensated dot
compensated_horner(coefs, x) -> f64      // Compensated polynomial evaluation
fma_residual(a, b, c) -> (f64, f64)      // Exact residual of fmadd
```

**Tier 3 — Double-double as first-class type (~8 primitives):**
```
DoubleDouble { hi: f64, lo: f64 }
dd_from_f64, dd_to_f64, dd_to_f64_rounded
dd_neg, dd_add, dd_sub, dd_mul, dd_div, dd_sqrt, dd_recip
dd_add_f64, dd_mul_f64                   // Mixed operations
```

With these, any recipe can maintain 106-bit precision throughout and round once at the end. This is the implementation mechanism for the `correctly_rounded` lowering strategy.

**Tier 4 — Specialist (build on demand):**
```
sum_k(data, k)                           // Rump's k-fold compensation, parameterized
dot_k(x, y, k)                           // Same for dot products
priest_sum                                // Doubly compensated (adversarial inputs)
kulisch_accumulator                       // ~4000-bit integer accumulator, EXACT dot product
three_sum, four_sum                       // Higher-order EFTs
veltkamp_split                            // For non-FMA targets
```

`kulisch_accumulator` deserves early attention despite being Tier 4 — it's the perfect oracle for the replay harness (Peak 4), giving bit-exact results for any sum or dot product at ~10-50x f64 speed without needing bignum arithmetic.

### The stopping rule for decomposition

A recipe stops decomposing when it bottoms out at a primitive. The rule for what counts as a primitive:

**A primitive is a terminal operation — either a single IEEE 754 hardware instruction, or an error-free transformation / compensated reduction whose internal sequence of hardware ops is treated as a unit by the compensated-arithmetic literature.**

This rule is objective. The rule we rejected — "stop when nobody wants to factor further" — was subjective and led to arbitrary stopping depths. With the hardware-and-compensated-foundations rule, every recipe has a well-defined decomposition tree.

---

## Layer 3 — Recipes

Everything that isn't an atom or a primitive is a recipe. Recipes are the vocabulary most users reach for: `pearson_r`, `exp`, `sigmoid`, `kendall_tau`, `garch_filter`, `two_group_comparison`, `brusselator_simulate`.

### Recipe rules

A recipe is valid if and only if:

1. **It is a thin orchestration layer**, typically 20-50 lines of composition plus a formula. Recipes that exceed this size probably contain embedded primitives that should be extracted.
2. **Its body contains only calls to**: atoms (`accumulate`, `gather`), primitives (from the `primitives/` tree), other recipes (at any lower layer), and control flow (if/match/loop for orchestration). No inline arithmetic beyond primitive calls.
3. **It has an explicit lowering tag** (see below) declaring its precision needs.
4. **It is findable under multiple family tags**, not locked into a single hierarchical category.

### Recipe shape vs content

Many recipes decompose into a **generic shape** plus a **specific content** (parameters, coefficients, configuration). When this decomposition is available, prefer writing one shape-recipe with multiple content-recipes instead of N near-identical recipes.

**Example: ODE right-hand sides.** Brusselator, Lorenz, Lotka-Volterra, Van der Pol, Rössler, Duffing, and FitzHugh-Nagumo are all polynomial in their state variables. They differ only in which monomial terms have nonzero coefficients. The correct decomposition is:

- `polynomial_ode_rhs(state, &PolynomialOde)` — the generic shape (expr). One recipe, used by all polynomial ODEs.
- `brusselator_system(a, b) -> PolynomialOde` — the specific content (config data). One recipe per named system.
- `brusselator_simulate(a, b, x0, y0, t_end, n_steps)` — the composition recipe that calls `rk4_system` with `polynomial_ode_rhs` and the Brusselator config.

This is the same principle as `power_mean(x, p)` unifying arithmetic/geometric/harmonic/quadratic means through a single parameterized recipe — avoid enumerating near-duplicates when a parameterized family exists.

### Recipe example: sigmoid

```rust
// recipes/math/sigmoid.rs
use crate::primitives::{fneg, fadd, fdiv, one_f64};
use crate::recipes::libm::exp;

#[precision(strict)]
pub fn sigmoid(x: f64) -> f64 {
    let neg_x  = fneg(x);
    let exp_nx = exp(neg_x);
    let denom  = fadd(one_f64(), exp_nx);
    fdiv(one_f64(), denom)
}
```

Every operation is an explicit call to a primitive or a lower-layer recipe. The decomposition tree is:

```
sigmoid
├── fneg                    (primitive, hardware)
├── exp                     (recipe, libm)
│   ├── range_reduction     (recipe)
│   │   ├── fmul            (primitive)
│   │   └── frint           (primitive)
│   └── compensated_horner  (primitive)
│       └── [internally: fmadd chain]
├── fadd                    (primitive, hardware)
└── fdiv                    (primitive, hardware)
```

The sigmoid recipe is 4 lines of orchestration. It contains no inline arithmetic — every operation is a primitive call.

---

## Lowering strategies

A recipe is a **tree of computation**, not a specific sequence of hardware operations. The tree is the same regardless of how it gets realized. Different realizations trade accuracy for performance.

Three lowering strategies are available for every recipe:

### `strict` — default

Each primitive call maps directly to its hardware op. Each operation rounds independently. Fastest. Default for most recipes.

```rust
fn sigmoid_strict(x: f64) -> f64 {
    fdiv(1.0, fadd(1.0, exp_strict(fneg(x))))
}
```

### `compensated` — accuracy-critical

Primitives are lifted to double-double arithmetic via the Tier 3 `DoubleDouble` type. Intermediate values carry `(hi, lo)` pairs. Each operation uses the dd_* primitives. Final result rounds back to f64 once. ~3-4x slower.

```rust
fn sigmoid_compensated(x: f64) -> f64 {
    let neg_x = DoubleDouble::from(-x);
    let exp_nx = exp_compensated(neg_x);           // returns DoubleDouble
    let denom  = dd_add(DoubleDouble::ONE, exp_nx);
    dd_to_f64_rounded(dd_div(DoubleDouble::ONE, denom))
}
```

### `correctly_rounded` — oracle-grade

For libm functions and critical values. Uses compensated arithmetic throughout, plus a final rounding adjustment that guarantees ≤ 1 ULP from the mathematically exact answer. ~4-8x slower than strict.

For recipes where compensated is insufficient (e.g., special function poles, iterative refinement), this strategy can also dispatch to an mpfr-backed implementation that computes the result at ~100-digit precision and rounds once at the end. Used for oracle generation.

### The lowering is a compiler pass, not rewritten code

The recipe source code is written once. The lowering strategy is selected per call. The compiler walks the recipe tree and emits a different sequence of primitive calls for each strategy. This requires a proper IR (Peak 1 `.tam` IR territory) — until that lands, we implement the strategies by hand for critical recipes and rely on the strict default for the rest.

---

## Precision tag semantics

Every recipe is annotated with a default lowering strategy. The tag sets the default; user overrides take priority.

```rust
#[precision(strict)]              // Default. Fast. No compensation.
#[precision(compensated)]         // Default to compensated lowering.
#[precision(correctly_rounded)]   // ≤ 1 ULP guarantee. Libm functions, critical values.
```

### Selection priority

When a recipe is called, the lowering strategy is determined by this chain:

1. **Explicit `.using(precision=X)` on the call site** (highest priority)
2. **Persistent precision from the enclosing pipeline** (via `.using()` on a Level 2/3 recipe)
3. **The recipe's own `#[precision(...)]` tag**
4. **The session default** (lowest priority, typically `strict`)

Examples:

```
# Primitive-level override, consumed
pearson_r(col_x=0, col_y=1).with(precision="correctly_rounded")
# → just this call uses correctly_rounded, despite pearson_r's default tag

# Pipeline-level override, persistent
two_group_comparison(col=0).using(precision="compensated")
# → every primitive and recipe inside the pipeline uses compensated lowering
# → inner overrides can still win via .with(), but the default flows down

# Session default
session.set_default_precision("strict")
# → every call without override uses strict
```

This priority chain matches the using() three-scope design documented in CLAUDE.md: consumed at Level 0-1, persistent at Level 2-3, session as the final fallback.

### Development workflow

When writing a recipe, the tag is **discovered during testing**, not guessed in advance:

1. Write the recipe with `#[precision(strict)]` (the default).
2. Run against the oracle (mpmath or kulisch_accumulator reference) in the Replay harness.
3. If strict is within tolerance (typically ≤ 5 ULP at worst case), keep the tag. Done.
4. If strict shows regime-dependent errors (like the erfc boundary bugs), switch to `#[precision(compensated)]` and retest.
5. If the recipe is in a layer where correctness to 1 ULP is required (libm, critical values, oracle-dependent consumers), use `#[precision(correctly_rounded)]` from the start and validate against arbitrary precision.

The test harness automatically runs all tagged strategies and compares against the oracle, so the tag decision is evidence-based.

---

## Filesystem organization

```
crates/tambear/src/
  atoms/                          ← Layer 1
    accumulate.rs                 ← The accumulate atom
    gather.rs                     ← The gather atom
    mod.rs

  primitives/                     ← Layer 2
    hardware/                     ← IEEE 754 terminal operations
      fadd.rs, fsub.rs, fmul.rs, fdiv.rs, fsqrt.rs
      fmadd.rs, fmsub.rs, fnmadd.rs, fnmsub.rs
      fabs.rs, fneg.rs, fcopysign.rs
      fmin.rs, fmax.rs            ← IEEE 754-2019 propagating semantics
      fcmp.rs
      classify.rs                 ← is_nan, is_inf, is_finite, signbit
      rounding.rs                 ← frint, ffloor, fceil, ftrunc
      mod.rs
    compensated/                  ← Error-free transformations
      two_sum.rs
      fast_two_sum.rs
      two_product_fma.rs
      two_square.rs
      two_diff.rs
      kahan_sum.rs
      neumaier_sum.rs
      pairwise_sum.rs
      dot_2.rs
      compensated_horner.rs
      fma_residual.rs
      mod.rs
    double_double/                ← DoubleDouble as first-class type
      type.rs                     ← struct DoubleDouble { hi, lo }
      add.rs
      mul.rs
      div.rs
      sqrt.rs
      conversions.rs
      mod.rs
    specialist/                   ← Tier 4, on-demand
      sum_k.rs
      dot_k.rs
      priest_sum.rs
      kulisch_accumulator.rs
      mod.rs
    mod.rs

  recipes/                        ← Layer 3
    libm/                         ← Transcendentals (Peak 2)
      exp.rs                      ← [precision(correctly_rounded)]
      log.rs
      sin.rs, cos.rs, tan.rs
      atan2.rs
      erf.rs, erfc.rs
      gamma.rs, log_gamma.rs
      ...
    math/                         ← Compositional primitives above libm
      sigmoid.rs, softmax.rs
      kernel_eval.rs              ← Gaussian, Epanechnikov, tricube kernels
      monomial_eval.rs
      polynomial_eval.rs          ← Used by polynomial_ode_rhs and everyone else
      polynomial_ode_rhs.rs       ← Generic polynomial ODE right-hand side
      ...
    stats/                        ← Statistical operations
      moments.rs
      pearson_r.rs                ← [precision(compensated)]
      spearman.rs
      kendall_tau.rs
      ...
    dynamical_systems/            ← Named ODE systems as recipes
      brusselator.rs              ← PolynomialOde config + _simulate
      lorenz.rs
      lotka_volterra.rs
      ...
    ...

  lib.rs                          ← Flat re-exports for TBS surface
```

### Slot values live flat with tags

Inside the recipe tree, things like `polynomial_ode_rhs`, `kernel_eval`, `sigmoid` are recipes that happen to serve as slot values (exprs) for accumulate/gather. They are NOT in a separate `slots/` directory. Recipes are flat with **tags as metadata** — either as comments parsed by a lint tool, attribute macros, or a manifest file:

```rust
// recipes/math/kernel_eval.rs
#[tags(expr, kernel, kde, gp, svm, nonparametric, interpolation)]
#[precision(strict)]
pub fn kernel_eval(kernel: KernelType, u: f64) -> f64 { ... }
```

A recipe can belong to arbitrary many families. Nesting in folders would force a primary category and lose the multi-family membership. Flat with tags preserves it.

---

## Worked examples

### Example 1: `exp` as a correctly-rounded libm recipe

```rust
// recipes/libm/exp.rs
use crate::primitives::hardware::{fmul, fmadd, frint};
use crate::primitives::compensated::compensated_horner;

// Cody-Waite range reduction constants (high + low parts for accuracy)
const LN2_HI: f64 = 0.6931471805598903;
const LN2_LO: f64 = 5.497923018708371e-14;

// Minimax polynomial coefficients for exp(r) on r ∈ [-ln(2)/2, ln(2)/2]
// Generated by sollya or Remez algorithm, then rounded to f64
const EXP_POLY: &[f64] = &[
    1.0,
    1.0,
    0.5,
    0.16666666666666666,
    0.041666666666666664,
    0.008333333333333333,
    0.001388888888888889,
    // ... more coefficients for 1-ULP accuracy
];

#[precision(correctly_rounded)]
#[tags(expr, libm, transcendental, special_function)]
pub fn exp(x: f64) -> f64 {
    // Step 1: range reduction x = k·ln(2) + r, where |r| <= ln(2)/2
    let k = frint(fmul(x, LOG2_E));
    let r = fmadd(-k, LN2_LO, fmadd(-k, LN2_HI, x));  // x - k·ln(2) with compensation

    // Step 2: polynomial evaluation of exp(r) using compensated Horner
    let exp_r = compensated_horner(EXP_POLY, r);

    // Step 3: reconstruction exp(x) = 2^k · exp(r) via bit manipulation
    //         (ldexp is a primitive we haven't listed yet — probably should be)
    ldexp(exp_r, k as i32)
}
```

Every arithmetic operation is an explicit primitive call. The recipe is ~15 lines of orchestration plus two constant tables. It calls `fmul`, `fmadd`, `frint` (hardware primitives), `compensated_horner` (compensated primitive), and `ldexp` (a primitive we need to add — it decomposes `2^k · x` into a bit-level integer operation).

### Example 2: `brusselator_simulate` as a composed recipe

```rust
// recipes/dynamical_systems/brusselator.rs
use crate::recipes::math::polynomial_ode_rhs;
use crate::recipes::math::{PolynomialOde, PolyTerm};
use crate::recipes::integration::rk4_system;

#[tags(ode_system, polynomial, dynamical_system, bifurcation, hopf)]
pub fn brusselator_system(a: f64, b: f64) -> PolynomialOde {
    PolynomialOde {
        dim: 2,
        terms: vec![
            PolyTerm { coef: a,          exponents: vec![0, 0], target: 0 },
            PolyTerm { coef: -(b + 1.0), exponents: vec![1, 0], target: 0 },
            PolyTerm { coef: 1.0,        exponents: vec![2, 1], target: 0 },
            PolyTerm { coef: b,          exponents: vec![1, 0], target: 1 },
            PolyTerm { coef: -1.0,       exponents: vec![2, 1], target: 1 },
        ],
    }
}

#[precision(strict)]
#[tags(simulation, trajectory, ode, brusselator)]
pub fn brusselator_simulate(
    a: f64, b: f64, x0: f64, y0: f64, t_end: f64, n_steps: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let system = brusselator_system(a, b);
    rk4_system(
        |_t, state| polynomial_ode_rhs(state, &system),
        &[x0, y0],
        0.0, t_end, n_steps,
    )
}
```

Brusselator-specific arithmetic is ZERO. The entire formula lives in the `PolynomialOde` config data structure. The same `polynomial_ode_rhs` recipe is shared by every polynomial ODE in tambear. Adding a new polynomial ODE (Lorenz, Rössler, Duffing) is just a new config constructor plus a `_simulate` wrapper.

### Example 3: `pearson_r` as a compensated statistical recipe

```rust
// recipes/stats/pearson_r.rs
use crate::atoms::accumulate;
use crate::recipes::stats::moments::{WelfordTwoVar, welford_two_var};
use crate::primitives::hardware::{fsub, fmul, fdiv, fsqrt};

#[precision(compensated)]
#[tags(correlation, linear, moments_consumer, descriptive)]
pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    // Single-pass Welford accumulation of co-moments
    let stats = welford_two_var(x, y);

    // Pearson formula: r = cov(x,y) / (sd(x) · sd(y))
    let denom = fmul(fsqrt(stats.var_x), fsqrt(stats.var_y));
    fdiv(stats.cov_xy, denom)
}
```

The recipe is 4 lines plus imports. The heavy lifting lives in `welford_two_var` (itself a recipe that wraps `accumulate` with the `WelfordMerge` op and the two-variable Welford state as the expr). The final formula uses explicit primitive calls for the arithmetic. The `#[precision(compensated)]` tag flags this recipe as benefiting from compensated lowering — appropriate because the centered-moments computation `(x_i - mean_x)(y_i - mean_y)` is cancellation-prone for near-constant data.

---

## Relationship to existing architecture

This document is an extension, not a replacement, of the existing Tambear Contract. Specific relationships:

- **Principle 1 (Custom implemented, our way)**: primitives are the lowest level where we hand-write from first principles. Hardware ops call Rust stdlib which calls LLVM intrinsics which emit machine instructions. We do not wrap vendor libraries above the hardware primitive layer.
- **Principle 2 (Accumulate + gather decomposition)**: unchanged. The two atoms are the top of the tree. Recipes orchestrate accumulate/gather calls.
- **Principle 3 (Shareable intermediates via TamSession)**: unchanged. Intermediates register at the recipe level. Primitives are too low-level to share via TamSession; they are called fresh each time and rely on compiler inlining for efficiency.
- **Principle 4 (Every parameter tunable)**: extended. In addition to algorithm parameters, every recipe now has a tunable precision strategy via the `using()` bag.
- **Principle 5 (Every measure in every family)**: unchanged. The flat catalog is now more precisely "atoms × primitives × recipes with tags."
- **Principle 10 (Publication-grade rigor)**: extended. Every recipe's oracle comparison runs against all three lowering strategies. A recipe passes publication-grade rigor when its correctly_rounded lowering matches mpmath/kulisch to 1 ULP across the full domain.

The **Layers Above the Math** section of CLAUDE.md (L0 primitives → L1 diagnostics → L2 override → L3 pipelines → L4 discovery) is orthogonal to this document's three-layer decomposition. The CLAUDE.md layers describe the *vertical stack of abstractions users see*. This document's layers (atoms, primitives, recipes) describe the *horizontal decomposition of any single primitive*. A Level 0 primitive in the user-facing sense is a recipe in this document's sense; its horizontal decomposition bottoms out at the primitives layer.

---

## Open questions

Marked for future discussion, not blocking any current work.

1. **Compensated libm dispatch**. When `#[precision(correctly_rounded)]` is requested for a libm function, we need a correctly-rounded implementation. Two options: (a) write the compensated version by hand alongside the strict version; (b) compile the same source with a different lowering pass. Option (b) requires the `.tam` IR (Peak 1). Until IR is ready, (a) is the fallback.

2. **Precision strategy for recursive recipes**. If `pearson_r` uses `welford_two_var` internally, and the user requests `pearson_r.using(precision="correctly_rounded")`, does the recursive call also use correctly_rounded? (Probably yes — precision should propagate through composition.)

3. **Kulisch as both primitive and oracle**. We want `kulisch_accumulator` as a primitive for use inside recipes AND as an independent oracle for the test harness. These are the same implementation but different call paths. How to organize?

4. **Constants with multiple precisions**. Mathematical constants like `PI`, `E`, `LN_2`, `SQRT_2` need `f64` values for the strict layer and `DoubleDouble` pairs for the compensated layer. Do we store both in one place, or have a `constants/` subtree in primitives?

5. **Backend specialization beyond lowering strategy**. The strict lowering on a GPU target may want to use the native `__expf` (approximate, fast) instead of our polynomial `exp`. Is this a fourth lowering strategy (`fast_approximate`) or a backend-specific substitution?

---

## Migration path

This architecture describes the *target state*. Current tambear has ~95 source files with atoms and recipes mixed, hardware primitives implicit in `x + y` expressions, no precision tags, and inconsistent hierarchical organization.

Migration is not urgent — the existing library works. It is a ~200-violation audit that should be executed when we next need to modify a group of recipes. The migration steps for any given file:

1. Identify the recipe's primitive calls. Replace inline `x + y` arithmetic with explicit `fadd(x, y)` calls.
2. Replace `f64::min`/`f64::max` with our `fmin`/`fmax` primitives. (This fixes the NaN-eating bug class.)
3. Add the `#[precision(...)]` tag. Default to `strict` unless the recipe is known accuracy-critical.
4. Add the `#[tags(...)]` metadata for multi-family membership.
5. Move the file to its correct layer in the filesystem: `atoms/`, `primitives/hardware/`, `primitives/compensated/`, `primitives/double_double/`, or `recipes/<family>/`.
6. Run the existing tests. They should still pass — this is a refactor, not a functional change.

A systematic Sonnet-agent migration wave can do this across the full codebase once we've hand-piloted it on 2-3 files to surface the corner cases.

---

## Build order

The smallest viable slice that proves the architecture works end-to-end:

1. **Write this document** (done).
2. **Create `primitives/hardware/`** with the ~20 IEEE 754 terminals. Each is 1-5 lines of Rust wrapping `f64::mul_add` and friends. The `fmin`/`fmax` files implement IEEE 754-2019 propagating semantics explicitly (not `f64::min`/`max`).
3. **Create `primitives/compensated/`** with the Tier 1 essentials: `two_sum`, `fast_two_sum`, `two_product_fma`, `two_square`, `two_diff`, `kahan_sum`. Reference implementations are standard from the literature (Knuth TAoCP, Higham "Accuracy and Stability").
4. **Write `kahan_sum` tests** against a 50-digit-precision oracle to validate the compensated semantics.
5. **Pick one libm function — `exp`** — and implement it as a `#[precision(correctly_rounded)]` recipe using the primitive layer. Test against mpmath reference across the full domain.
6. **Pick one existing tambear recipe — `pearson_r`** — and migrate it to explicit primitive calls with the `#[precision(compensated)]` tag. Verify the lowering gives the expected accuracy improvement on adversarial inputs (large-magnitude near-constant series).

Steps 2-4 are Sonnet-agent work once step 1 is approved. Step 5 is the Peak 2 kickoff. Step 6 is the pilot migration for the Sonnet-agent wave that follows.
