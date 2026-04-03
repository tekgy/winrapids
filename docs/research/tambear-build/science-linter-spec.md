# .tbs Science Linter — Warning Specification

**Author**: Naturalist
**Date**: 2026-04-02
**Status**: Spec for task #21. Feeds pathmaker implementation.

---

## Overview

The science linter runs at parse time (static) and at execution time (dynamic). It examines the .tbs chain, annotates each step with its (T, K, O) coordinates, detects cross-kingdom dependency patterns, and emits warnings organized into five boundary types.

The linter does NOT prevent execution. It emits warnings with severity levels:
- **WARN**: likely user error or suboptimal choice
- **INFO**: correct but surprising behavior, worth documenting
- **HINT**: optimization opportunity (fuseable steps, redundant computation)

---

## Part 1: Kingdom Annotations

Every .tbs function gets a static (T, K, O) annotation. T = transform applied to data before the operation. K = kingdom (A, B, C). O = the oracle/combine operator.

### Kingdom A (commutative, single-pass, fuseable)

| Function | T | K | O | Notes |
|----------|---|---|---|-------|
| `normalize()` | Identity | A | Affine | Per-column mean/std, then affine map |
| `mean()`, `variance()`, `std()` | Identity | A | Welford | Single-pass moments |
| `skewness()`, `kurtosis()` | Identity | A | Welford(4) | Degree-4 moments |
| `correlation()` | Identity | A | DotProduct | Pairwise degree-2 |
| `covariance()` | Identity | A | DotProduct | Same as correlation without normalization |
| `train.linear()` | Identity | A | DotProduct | Gram matrix X'X + X'y, Cholesky solve |
| `distance(metric="euclidean")` | Identity | A | DotProduct | 3-field: sq_norm_x, sq_norm_y, dot_prod |
| `distance(metric="cosine")` | Identity | A | DotProduct | Same 3-field, different extraction |
| `distance(metric="poincare")` | Identity | A | DotProduct | Same 3-field, hyperbolic extraction |
| `ipw()` | Propensity | A | WeightedMean | Inverse probability weighting |
| `kaplan_meier()` | Identity | B | Multiply | Prefix product (but log-space → prefix sum) |
| `entropy()` | Identity | A | NegPLogP | Shannon: accumulate(-p log p) |
| `mutual_information()` | Identity | A | NegPLogP | Joint + marginal entropies |
| `bayesian.linear()` | Identity | A | DotProduct | Conjugate: Λ_n = Λ_0 + X'X |
| `cronbach_alpha()` | Identity | A | Welford | Item variances + total variance |
| `moran_i()` | Spatial | A | WeightedSum | Weighted spatial autocorrelation |
| `pca(n_components)` | Identity | A+C | DotProduct | A: covariance matrix, C: eigendecomposition |
| `realized_variance()` | DiffSquare | A | Add | Sum of squared returns |
| `dft(k)` | Identity | A | DotProduct | Each coefficient is a dot product |

### Kingdom B (sequential, prefix scan)

| Function | T | K | O | Notes |
|----------|---|---|---|-------|
| `ewma(alpha)` | Identity | B | Affine | α·x + (1-α)·state |
| `holt(alpha, beta)` | Identity | B | Affine(2D) | 2D affine state |
| `prefix_sum()` | Identity | B | Add | Running cumulative sum |
| `sort()` | Identity | B | Compare | Comparison-based ordering |
| `diff()` | Identity | B | Subtract | First differences (adjacent pairs) |
| `lag(k)` | Identity | B | Shift | k-step delay |
| `fft()` | Identity | B | Butterfly | O(N log N) via log N sequential levels |
| `ar(p)` | Identity | B | Affine(pD) | AR coefficient application |
| `garch.filter(omega, alpha, beta)` | Identity | B | Affine | σ² prefix scan (given params) |
| `bipower_variation()` | Identity | B | ProductPair | |r_t|·|r_{t-1}| adjacent pairs |

### Kingdom C (iterative, pipeline breaks)

| Function | T | K | O | Notes |
|----------|---|---|---|-------|
| `train.logistic()` | Identity | C(A) | IRLS | Inner: weighted Gram (A). Outer: iterate |
| `kmeans(k)` | Identity | C(A) | Assign+Centroid | Inner: distance (A) + assign (A). Outer: iterate |
| `dbscan(eps, min_pts)` | Identity | C(A) | NeighborExpand | Inner: range query (A). Outer: iterate clusters |
| `gmm(k)` | Identity | C(A) | EM | E-step (A) + M-step (A), iterate |
| `train.neural(...)` | Identity | C(B+A) | Backprop | Inner: forward (B) + gradient accumulate (A). Outer: iterate |
| `garch.fit()` | Identity | C(B) | MLE | Inner: filter (B). Outer: parameter search (C) |
| `cox_ph()` | Identity | C(B+A) | NewtonRaphson | Inner: risk set scan (B) + gradient (A). Outer: iterate |
| `mcmc(...)` | Identity | C(B) | MetropolisHastings | Sequential sampling to convergence |
| `variational(...)` | Identity | C(A) | ELBO | Iterate ELBO optimization |
| `huber_mean(k)` | Identity | C(A) | IRLS | Weighted mean where weights depend on estimate |
| `newton(f, df)` | Identity | C | Iterate | Generic root finding |
| `paf(n_factors)` | Identity | C(A) | EigenIterate | Iterative communality estimation |
| `lme(formula)` | Identity | C(A) | EM | Henderson equations iterated |

---

## Part 2: Boundary Warnings

### Type 1: Denominator Boundary

**Trigger**: operation requires a ratio where the denominator can be zero or near-zero.

| Warning ID | Function | Condition | Message | Severity |
|-----------|----------|-----------|---------|----------|
| D001 | `variance()` | n < 2 | "Variance undefined for n < 2 (division by n-1 = 0)" | WARN |
| D002 | `skewness()` | n < 3 | "Skewness undefined for n < 3" | WARN |
| D003 | `kurtosis()` | n < 4 | "Kurtosis undefined for n < 4" | WARN |
| D004 | `correlation()` | std(x) ≈ 0 or std(y) ≈ 0 | "Correlation undefined for constant series" | WARN |
| D005 | `train.linear()` | X'X singular | "Design matrix is rank-deficient; OLS undefined" | WARN |
| D006 | `clustered_se()` | n_clusters < 2 | "Clustered SE requires ≥ 2 clusters (correction = Inf)" | WARN |
| D007 | `r_hat()` | within-chain var ≈ 0 | "R-hat undefined for constant chains; returning 1.0" | INFO |
| D008 | `moran_i()` | spatial variance ≈ 0 | "Moran's I undefined for spatially constant field" | WARN |
| D009 | `cv()` | mean ≈ 0 | "Coefficient of variation undefined when mean ≈ 0" | WARN |

**Static check (parse time)**: D001-D003 can fire if the chain includes `variance()` on data known to be small (e.g., after a `filter()` that may reduce n).

**Dynamic check (runtime)**: All others require actual data values.

### Type 2: Convergence Boundary

**Trigger**: iterative algorithm reaches max_iter or produces out-of-range values.

| Warning ID | Function | Condition | Message | Severity |
|-----------|----------|-----------|---------|----------|
| C001 | Any Kingdom C | iterations = max_iter | "Reached max_iter without convergence (final delta = {d})" | WARN |
| C002 | `garch.fit()` | α + β ≥ 1 | "GARCH parameters at or beyond stationarity boundary (α+β = {v})" | WARN |
| C003 | `garch.fit()` | α + β = 1 exactly | "IGARCH: integrated GARCH, variance non-stationary" | INFO |
| C004 | `paf()` | any communality > 1.0 | "Heywood case: communality {v} > 1.0 — rank-deficient input?" | WARN |
| C005 | `train.logistic()` | separation detected | "Complete or quasi-complete separation: MLE does not exist" | WARN |
| C006 | `lme()` | σ²_u ≈ 0 at boundary | "Between-group variance near zero; EM convergence slow" | INFO |
| C007 | `gmm()` | component weight → 0 | "GMM component {k} has negligible weight; consider k-1 components" | WARN |
| C008 | `kmeans()` | empty cluster | "k-means: cluster {k} became empty during iteration" | WARN |
| C009 | `newton()` | derivative ≈ 0 | "Newton step: derivative near zero at x={x}, may diverge" | WARN |
| C010 | `mcmc()` | acceptance rate < 0.1 or > 0.5 | "MH acceptance rate {r}: tune proposal_sd (optimal ≈ 0.234)" | HINT |

**Split into 2a/2b** (from adversarial's taxonomy):
- **2a (slow convergence)**: C001, C003, C006, C010. Iteration IS converging, just slowly. Action: increase max_iter, switch method, or accept.
- **2b (divergence)**: C002 (α+β>1), C004 (Heywood), C005 (separation). Iteration is DIVERGING or undefined. Action: structural fix required.

### Type 3: Cancellation Boundary

**Trigger**: computation involves subtraction of nearly-equal quantities or mixed-sign accumulation.

| Warning ID | Function | Condition | Message | Severity |
|-----------|----------|-----------|---------|----------|
| N001 | `variance()` (naive) | — | "Use Welford or centered variance instead of Σx²-(Σx)²/n" | WARN |
| N002 | `omega()` | mixed-sign loadings | "McDonald's ω with bipolar loadings ({pos}+, {neg}-): reverse-score first" | WARN |
| N003 | `skewness()` | discrete data, few values | "Skewness on discrete data with < 10 unique values: interpret cautiously" | INFO |
| N004 | `correlation()` | near-constant + noise | "Near-constant series (CV < 0.01): correlation dominated by rounding" | WARN |
| N005 | `log_sum_exp()` | raw (no max subtraction) | "Use log-sum-exp with max subtraction for numerical stability" | WARN |
| N006 | `det()` | large matrix | "Determinant of large matrix: use log-determinant (sum of log diag of Cholesky)" | HINT |
| N007 | `difference()` on similar values | |a-b|/max(|a|,|b|) < ε | "Catastrophic cancellation: a-b loses {d} digits of precision" | WARN |

**Static check**: N001 can fire if the chain uses a naive variance formula. N005 can fire at parse time if `exp()` is called without prior max subtraction. N002 fires when `omega()` is called after `paf()` on data where signs can be checked.

### Type 4: Equipartition Boundary

**Trigger**: result is correct but reflects maximum entropy / minimum information, which may surprise.

| Warning ID | Function | Condition | Message | Severity |
|-----------|----------|-----------|---------|----------|
| E001 | `lme()` | singleton groups + ICC ≈ 0.5 | "ICC = 0.5 with singleton groups: EM splits variance equally (max entropy)" | INFO |
| E002 | `kriging()` | prediction far from all obs | "Kriging prediction {d}× variogram range from nearest obs: degrades to GLS mean" | INFO |
| E003 | `kmeans()` | k = n | "k = n: every point is its own cluster (trivially optimal)" | INFO |
| E004 | `pca()` | eigenvalues all ≈ equal | "All eigenvalues similar: no dominant principal component (isotropic data)" | INFO |
| E005 | `entropy()` | H ≈ log(n) | "Entropy near maximum: distribution is approximately uniform" | INFO |
| E006 | `bayesian.linear()` | posterior ≈ prior | "Posterior barely updated from prior: data has low information content" | INFO |
| E007 | `r_hat()` | all chains identical | "R-hat = 1.0 but all chains at same point: may indicate non-mixing" | INFO |

**These are all INFO severity** — the computation is correct, the result just reflects an edge of the parameter space.

### Type 5: Fock Boundary (Kingdom Crossing)

**Trigger**: the .tbs chain structure implies a kingdom crossing that prevents fusion or indicates a structural mismatch.

| Warning ID | Pattern | Message | Severity |
|-----------|---------|---------|----------|
| F001 | A → A → A (all fuseable) | "Chain is fully fuseable: {n} operations can merge into 1 kernel" | HINT |
| F002 | A → C → A | "Kingdom C step '{name}' breaks the pipeline: A steps before and after cannot fuse across it" | INFO |
| F003 | C wrapping A | "Iterative step '{name}' re-scans data each iteration. The inner A step is: {inner}. Consider caching the sufficient statistic." | HINT |
| F004 | A used as C | "'{name}' is Kingdom A (single-pass) but called inside an explicit loop. Remove the loop — one pass suffices." | WARN |
| F005 | C used where A suffices | "'{name}' uses iterative estimation, but the input is in the exponential family. Consider the closed-form conjugate solution." | HINT |
| F006 | B after sort-dependent A | "'{name}' requires sorted data but follows a commutative step. The sort is implicit — consider making it explicit for clarity." | INFO |
| F007 | Multiple C steps chained | "Chain contains {n} iterative steps in sequence. Each re-scans data. Consider restructuring to share the data pass." | HINT |
| F008 | A step with self-referential kernel | "'{name}' looks like a single-pass operation but its weight function depends on the output (self-referential). It requires iteration (Kingdom C)." | WARN |

**Static checks (parse time)**:
- F001: scan the chain for consecutive A-annotated steps → emit fusion hint
- F002: detect A-C-A patterns → emit pipeline break info
- F004: detect A operations inside explicit `iterate()` or `repeat()` wrappers
- F005: detect `mcmc()` or `variational()` applied to exponential-family models where `bayesian.linear()` would work
- F007: count consecutive C steps
- F008: detect known self-referential patterns (huber_mean, bisquare_mean — weight depends on output)

**The key implementation rule**: The linter walks the chain, looks up each step's K annotation, and builds a "kingdom sequence" like [A, A, C, A, A, B, A]. From this sequence:
1. Consecutive A runs are fusion candidates (F001)
2. C steps are pipeline breaks (F002)
3. B steps are scan boundaries (milder break — prefix scan is still parallel)
4. The longest fuseable A-run determines the maximum kernel fusion width

---

## Part 3: Cross-Kingdom Dependency Patterns

Beyond the per-step warnings, the linter detects PATTERNS across steps:

### Pattern 1: Redundant Data Scan

```
mean().variance().skewness().kurtosis()
```

Four separate data scans, but all are Kingdom A with Welford(4) oracle. One pass suffices. Linter emits:

> HINT: `mean()`, `variance()`, `skewness()`, `kurtosis()` share oracle Welford(4). Fuse to single `moments(4)` call.

**Detection**: group consecutive A steps by oracle type. If multiple steps share an oracle (or one oracle subsumes another), suggest fusion.

### Pattern 2: Iterated Single-Pass

```
for i in 1..100 { mean(data) }
```

Kingdom A inside an explicit loop. The mean doesn't change across iterations. Linter emits:

> WARN F004: `mean()` is Kingdom A — result is identical across iterations. Hoist outside the loop.

**Detection**: identify A-annotated steps inside loop constructs. If the step's input doesn't depend on the loop variable, it's hoistable.

### Pattern 3: Missed Conjugacy

```
mcmc(target=normal_posterior, ...)
```

MCMC (Kingdom C) applied to a normal posterior, which has a closed-form conjugate update (Kingdom A). Linter emits:

> HINT F005: Normal posterior has conjugate update. Use `bayesian.linear()` instead of `mcmc()` for exact results in one pass.

**Detection**: check the `target` argument's distributional family against a table of conjugate pairs.

### Pattern 4: Naive Variance in Chain

```
normalize().train.linear()
```

If `normalize()` computes variance using the naive formula (Σx² - (Σx)²/n), the linter warns:

> WARN N001: `normalize()` uses variance internally. Ensure Welford or centered formula for numerical stability.

**Detection**: trace through the call graph to identify internal uses of variance-like computations.

### Pattern 5: Self-Referential Detection

```
weighted_mean(data, weights=f(current_estimate))
```

A weighted mean where the weights depend on the current estimate is Kingdom C, not A. Linter emits:

> WARN F008: `weighted_mean()` with weights depending on the output is self-referential. This requires iteration (Kingdom C), not a single pass.

**Detection**: check whether any argument to a Kingdom A function references the function's own output or a variable updated in the same scope.

---

## Part 4: Implementation Notes

### Data Structures

```rust
/// Kingdom annotation for a .tbs step.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Kingdom {
    A,          // Commutative, single-pass, fuseable
    B,          // Sequential, prefix scan
    C,          // Iterative
    CA,         // C wrapping A (most common: IRLS, EM, k-means)
    CB,         // C wrapping B (GARCH MLE, neural training)
    CBA,        // C wrapping B+A (Cox PH, full neural training)
}

/// (T, K, O) annotation for a .tbs function.
struct TbsAnnotation {
    transform: &'static str,   // "Identity", "Log", "DiffSquare", etc.
    kingdom: Kingdom,
    oracle: &'static str,      // "Welford", "DotProduct", "Affine", etc.
    fuseable_with: &'static [&'static str],  // Oracle compatibility for fusion
}

/// Linter warning.
struct LintWarning {
    id: &'static str,          // "D001", "C004", "F001", etc.
    severity: Severity,        // Warn, Info, Hint
    span: (usize, usize),     // Character span in .tbs source
    message: String,
}
```

### Linter Passes

1. **Parse-time pass**: Walk the AST, look up each step's annotation, build the kingdom sequence. Emit F001 (fusion hints), F002 (pipeline breaks), F004 (A-in-loop), F005 (missed conjugacy), F007 (C-chain length). Emit N001 (naive variance) if detectable from the call graph.

2. **Pre-execution pass**: After argument resolution but before execution. Emit D001-D009 where n is known. Emit N002 (bipolar ω) if loadings are available from a prior step. Emit E002 (kriging range) if prediction points are known.

3. **Post-execution pass**: After each step completes. Emit C001-C010 based on actual convergence behavior. Emit E001, E003-E007 based on actual output values.

### Fusion Optimizer (feeds task #24)

The kingdom sequence from pass 1 directly informs the JIT compiler:

```
Chain: normalize().mean().variance().train.linear()
Kingdom: [A, A, A, A]
Fusion: all four merge into one kernel (emit scatter_multi_phi with 4 phi expressions)

Chain: normalize().ewma(0.5).train.logistic()
Kingdom: [A, B, C(A)]
Fusion groups: [normalize] | [ewma] | [train.logistic inner A per iteration]
Pipeline: map kernel → prefix scan kernel → iterate(scatter kernel)

Chain: mean().kmeans(3).normalize()
Kingdom: [A, C(A), A]
Fusion groups: [mean] | [kmeans iterations] | [normalize]
Note: first A and last A cannot fuse across the C step (F002)
```

The fusion optimizer groups consecutive same-kingdom steps, identifies shared oracles within groups, and emits the minimum number of kernel launches.

---

## Part 5: Oracle Compatibility for Fusion

Two Kingdom A steps fuse if their phi expressions can be evaluated in the same data pass. This requires:

1. **Same input traversal**: both iterate over the same data in the same order (always true for Kingdom A — order doesn't matter)
2. **Independent accumulators**: each step writes to its own output buffer (always true for scatter_multi_phi)
3. **No data dependency**: step 2 doesn't read step 1's output (true for A→A, false for A→B where B reads A's result)

**Oracle subsumption**: Welford(4) subsumes Welford(2) subsumes Welford(1). If one step needs mean (Welford(1)) and another needs kurtosis (Welford(4)), emit one Welford(4) kernel and extract both.

**Oracle families** (fuseable within family):

| Family | Members | Shared accumulator |
|--------|---------|-------------------|
| Welford | mean, variance, std, skewness, kurtosis, CV | {n, M1, M2, M3, M4} |
| DotProduct | correlation, covariance, train.linear, distance, PCA-cov | {Σx, Σy, Σxy, Σx², Σy²} per pair |
| MinMax | min, max, range, argmin, argmax | {min, max, argmin, argmax} |
| Count | count, count_nonzero, count_nan | {n, n_nonzero, n_nan} |
| Entropy | entropy, mutual_info, kl_divergence | {Σp·log(p)} |

Steps from different oracle families can STILL fuse — they just need separate accumulators in the same kernel. The `scatter_multi_phi` primitive already supports up to 5 phi expressions in one pass.
