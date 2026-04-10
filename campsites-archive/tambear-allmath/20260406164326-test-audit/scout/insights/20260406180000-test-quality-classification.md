# Test Quality Classification: tambear Test Suite
Created: 2026-04-06T18:00:00-05:00  
By: scout-2

---

## The Landscape

tambear has ~2861 total tests across integration test files and in-module tests.

### Integration Test Files

| File | Count | Quality | Notes |
|------|-------|---------|-------|
| `gold_standard_parity.rs` | ~489 | **GOLD** | scipy/numpy/sklearn/faer oracle comparisons |
| `adversarial_boundary.rs` | ~40 | ADVERSARIAL | Finds real bugs — TDA rips_h0 single-point bug confirmed |
| `adversarial_boundary2-10.rs` | ~500 | **MISLEADING** | 54 documented bugs, all tests PASS (see below) |
| `adversarial_disputed.rs` | ~20 | STALE | Documents old-code bugs; current code has fixes |
| `adversarial_tbs.rs` | ~30 | GOOD | TBS scripting language tests |
| `scale_ladder*.rs` | ~200 | PERFORMANCE | Not correctness — run with `--ignored --nocapture` |
| `svd_adversarial.rs` | ~30 | GOLD | Overlap with gold_standard_parity.rs SVD section |

---

## The Silent Bug Problem

**adversarial_boundary2 through adversarial_boundary10** contain a systemic pattern:

```rust
let result = some_function(nan_input);
match result {
    Ok(val) => {
        eprintln!("CONFIRMED BUG: function returns {} for degenerate case", val);
        // assertion that always passes, e.g.:
        assert!(val.is_finite() || !val.is_finite()); // always true!
    },
    Err(_) => {
        eprintln!("CONFIRMED BUG: function panics on degenerate case");
        // or no assertion at all
    }
}
```

`cargo test` is green. The bugs are silently logged to stderr. Nobody sees them unless they run `cargo test -- --nocapture`.

### Count by file
- boundary2: 5 eprintln patterns
- boundary5: 12
- boundary6: 10
- boundary7: 26
- boundary8: 29
- boundary9: 20
- boundary10: 37
- **Total: ~139 lines flagging bugs (54 unique bug descriptions)**

---

## Confirmed Bug Catalog (by severity)

### CRITICAL: Infinite Loops (hang, not crash)
These will hang a production system forever:

| Bug | Function | Trigger |
|-----|----------|---------|
| kaplan_meier infinite loop on NaN | `kaplan_meier` | NaN in times array |
| log_rank_test infinite loop on NaN | `log_rank_test` | NaN in times array |
| sample_geometric(p=0) infinite loop | `sample_geometric` | p=0 (never succeeds) |
| max_flow infinite loop when source==sink | `max_flow` | source==sink |

Note: the survival infinite loops are **post-fix regressions** — the panic was fixed by switching to `total_cmp`, but the NaN detection (`position(|&i| times[i].is_nan()).unwrap_or(n)`) may leave the while loop running past valid data.

### HIGH: Panics/Crashes on Valid Mathematical Inputs
These panic instead of returning an error:

- `ANOVA` on empty groups
- `correlation_matrix` on constant data
- `DID` with no post-treatment observations
- `GP regression` with `noise_var=0` (singular kernel)
- `Hotelling T²` with `n=1`
- `MCMC` when `burnin > n_samples`
- `MCMC` when `log_target` always returns `-Inf`
- `MCMC` with `proposal_sd=0`
- `mcd_2d` on collinear data
- `conv1d` with `stride=0` (division by zero)
- `Bayesian regression` on underdetermined system
- `nn_distances` on single point
- `propensity_scores` on perfect separation
- `knn_from_distance` with `k=0`

### MEDIUM: Silent NaN/Inf (wrong answer, no error)
These silently return garbage:

- `batch_norm` with eps=0 on constant data → NaN/Inf
- `BCE loss` with predicted exactly 0 or 1 → NaN
- `chi2 goodness of fit` with zero expected count → NaN
- `Clark-Evans R` with area=0 → NaN (div by zero)
- `cosine_similarity_loss` for zero vectors → NaN
- `cox_ph` with perfect separation → NaN (exp overflow)
- `global_avg_pool2d` with 0 spatial dimensions → NaN
- `GP` with `length_scale=0` → NaN/Inf
- `GP` with `noise=0` fails to interpolate at training points
- `KDE` with `bandwidth=0` → NaN/Inf
- `KNN` selects NaN-distance neighbor over finite neighbor
- `medcouple` for 2 data points → NaN
- `Moran's I` for constant values → NaN (0/0)
- `Ripley's K` with area=0 → non-finite
- `RoPE` with `base=0` → NaN/Inf
- `sample_exponential(lambda=0)` → returns value (should guard)
- `sample_gamma(alpha=0)` → NaN
- `silverman_bandwidth` for constant data → 0 or NaN (then KDE blows up)
- `temperature_scale` with T=0 → Inf

### LOW: Wrong Mathematical Answer
These return finite values that are mathematically incorrect:

- **Dijkstra with negative weights**: returns wrong shortest path (dist wrong instead of error)
- **erf(0) ≠ 0**: returns non-zero for zero input
- **Lagrange with duplicate x**: division by zero in basis polynomials → wrong result
- **Rényi entropy at α=1 ≠ Shannon entropy**: should converge to Shannon as α→1
- **R-hat for identical chains**: artifact → not ~1.0 as required
- **R-hat returns NaN** for single chain (between-chain variance undefined)
- **Richardson extrapolation with ratio=1**: ratio^p - 1 = 0 → division by zero
- **Tsallis entropy at q=1 ≠ Shannon**: should converge
- **breusch_pagan_re** with t=1: division by t-1=0
- **mat_mul**: no dimension mismatch check (silently wrong or random behavior)
- **Aitken delta² for constant sequence**: 0/0 → NaN

---

## The Stale Disputed Tests

`adversarial_disputed.rs` documents bugs in an **older version** of the code that have since been fixed:

1. **t-SNE "Gauss-Seidel gradient"**: Current `dim_reduction.rs` line 293 has explicit Jacobi comment + grad_buf pattern. Fixed.
2. **t-SNE "missing early exaggeration"**: Current `dim_reduction.rs` line 296: `let exag = if iter < 250 { 4.0 } else { 1.0 };`. Fixed.
3. **kaplan_meier panic on NaN**: Fixed by `total_cmp`. But a new infinite-loop regression was introduced.

The disputed tests still PASS because they only assert non-panic/non-degenerate output, not the specific algorithmic issue. These tests are now **misleading** — they claim bugs that no longer exist. Recommendation: replace with tests that positively assert the corrected behavior.

---

## Gold Standard Parity: Current Coverage

**gold_standard_parity.rs has 489 tests** — much larger than initially reported. Coverage:

| Family | Module | Status |
|--------|--------|--------|
| f06 Descriptive stats | `descriptive.rs` | COVERED: mean, variance, std, skewness, kurtosis, median, quartiles, gini, geometric/harmonic mean, MAD, trimmed mean |
| f07 Hypothesis tests | `hypothesis.rs` | COVERED: one-sample t, two-sample t, Welch, paired t, ANOVA, chi², proportions z, effect sizes, corrections |
| f08 Nonparametric | `nonparametric.rs` | COVERED: Spearman, Kendall, Mann-Whitney, Wilcoxon, KS, sign test |
| f23 Neural | `neural.rs` | COVERED: activations, losses, attention, batch norm, layer norm, conv1d, pooling |
| f25 Info theory | `information_theory.rs` | COVERED: Shannon, Rényi, Tsallis, KL, JS, cross-entropy |
| f28 Manifold | `multivariate.rs` | COVERED: Poincaré, spherical geodesic, mixture |
| f29 Graph | `graph.rs` | COVERED: Dijkstra, Bellman-Ford, Floyd-Warshall, MST, PageRank, connected components |
| f35 KNN | `knn.rs` | COVERED |
| f05 Optimization | `optimization.rs` | COVERED: Nelder-Mead, L-BFGS-B, golden section, box-constrained |
| SVD | `linear_algebra.rs` | COVERED vs faer oracle: ill-conditioned, rank-deficient, rectangular |

**Genuine coverage gaps in gold_standard_parity.rs:**
- Survival: KM, Cox PH, log-rank (4 tests in-module only)
- ARMA/GARCH: not covered
- KDE: not covered
- Physics: not covered (42 in-module tests, none external oracle)
- Bayesian MCMC: limited

---

## Recommendation for adversarial team

Priority order for test rewrites:

1. **Fix the 4 infinite-loop bugs** — they hang, they're not documented as OK. These need code fixes, not test fixes.
2. **Stale disputed tests** — rewrite to assert correct current behavior positively.
3. **Convert "CONFIRMED BUG" tests** — for bugs not yet fixed, convert from silent-pass to `#[should_panic]` or proper error return assertions. This makes test failures actionable.
4. **For fixed bugs** — add regression tests that assert the fix, not the old behavior.

The systemic problem: the adversarial test authors documented bugs using `eprintln!` because they couldn't modify production code. That's valid as a process, but the result is a green test suite hiding serious defects. The path forward is for pathmaker to fix the production bugs and for adversarial to convert the eprintln documentation into proper assertions.
