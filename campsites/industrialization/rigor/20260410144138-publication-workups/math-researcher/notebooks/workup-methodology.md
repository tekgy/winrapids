# Publication-Grade Workup Methodology

## What a Workup IS

A workup is the proof that a primitive is correct. It's structured as if preparing
for a Nature paper — not metaphorically, literally. The workup for `kendall_tau`
(see `tests/workup_kendall_tau.rs` and `tests/workup_pearson_r.rs`) sets the
pattern. This document codifies that pattern for all future primitives.

## Workup Structure (7 Sections)

### Section 1: Mathematical Definition

State the exact formula being implemented. Use the canonical reference paper.
Document every symbol, every subscript, every edge case definition.

Example (Kendall tau-b):
```
τ_b = (n_c - n_d) / √((n₀ - n₁)(n₀ - n₂))
where:
  n₀ = n(n-1)/2 (total pairs)
  n_c = concordant pairs
  n_d = discordant pairs
  n₁ = Σ t_i(t_i-1)/2 (ties in x)
  n₂ = Σ u_j(u_j-1)/2 (ties in y)
```

### Section 2: Assumptions

Explicit list of every mathematical assumption. Each assumption maps to either:
- A runtime check (detect violation, return NaN or error)
- A documented precondition (caller's responsibility)
- An ignored assumption (explain why)

| Assumption | Treatment | Justification |
|---|---|---|
| Inputs are same length | Runtime assert | Fundamental requirement |
| n ≥ 2 | Runtime → NaN | No pairs to count |
| No NaN in data | Caller filters | Performance: no per-element check |
| Not all values identical | Detected → NaN | Denominator is zero |

### Section 3: Parameters

Every tunable parameter documented:

| Parameter | Type | Range | Default | Meaning | When to tune |
|---|---|---|---|---|---|
| `x` | `&[f64]` | any finite | required | First variable | — |
| `y` | `&[f64]` | any finite | required | Second variable | — |

### Section 4: Edge Cases

Every degenerate input and what should happen:

| Input | Expected Output | Reason |
|---|---|---|
| Empty arrays | NaN | No data |
| n = 1 | NaN | No pairs |
| All x identical | NaN | All pairs are x-ties |
| All y identical | NaN | All pairs are y-ties |
| Perfect monotone | 1.0 | All concordant |
| Perfect anti-monotone | -1.0 | All discordant |
| ±Inf values | Correct ranking | Inf < Inf is false, Inf > x is true |
| Very large n (10⁶) | Correct | Must use O(n log n) algorithm |

### Section 5: Oracle Parity

Every implementation is verified against at least TWO independent oracles:

1. **mpmath (arbitrary precision)** — the ground truth
   - Compute at 50+ decimal digits
   - This is the reference value

2. **scipy / statsmodels / R / MATLAB** — the peer implementations
   - Same inputs, same parameters
   - Document version (e.g., scipy 1.17.1)
   - If they DISAGREE with mpmath: we found their bug → file upstream

3. **SymPy / closed-form** — for cases where analytical solutions exist
   - Monotone data → τ = 1, Pearson = 1, etc.

#### Parity Table Format

| Case | Input | mpmath (50 dp) | scipy | R | Our value | Match? |
|---|---|---|---|---|---|---|
| Monotone | [1,2,3,4,5] | 1.0 exactly | 1.0 | 1.0 | 1.0 | YES |
| Light ties | [...] | 0.948683... | 0.948683... | 0.948683... | 0.948683... | YES |
| Random n=20 | seed=42 | -0.042105... | -0.042105... | -0.042105... | -0.042105... | YES |

If a peer implementation disagrees, the workup documents:
- What they return
- What mpmath returns
- Analysis of why they're wrong
- Link to upstream bug report

### Section 6: Invariant Tests

Mathematical properties that must hold for ALL inputs:

| Invariant | Formula | Test Method |
|---|---|---|
| Symmetry | τ(x,y) = τ(y,x) | Random inputs, verify |
| Shift invariance | τ(x+c, y) = τ(x,y) | Shift by 100, compare |
| Scale invariance | τ(cx, y) = τ(x,y) for c>0 | Scale by 42, compare |
| Sign flip | τ(-x, y) = -τ(x,y) | Negate x, compare |
| Range | τ ∈ [-1, 1] | Random inputs, bound check |
| NaN propagation | NaN in → NaN out | Insert NaN, verify |

### Section 7: Adversarial Cases

Inputs designed to break naive implementations:

| Case | Why It's Hard | Expected Behavior |
|---|---|---|
| All ties | Denominator = 0 | NaN, not division by zero |
| Alternating ±ε | Near-zero correlation | Stable, not oscillating |
| n = 10⁶ | Must be O(n log n) | Complete in <1s |
| Extreme values (10³⁰⁸) | Overflow risk | Correct via careful arithmetic |
| Adversarial ordering | Worst case for merge-sort | Still O(n log n) |
| Nearly-sorted | Best case detection | Should not be slower |

---

## Benchmark Infrastructure

### Scale Ladder

Every primitive is benchmarked at exponentially increasing scales:

```
n ∈ [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
```

For each n, record:
- Wall time (median of 5 runs)
- Memory peak
- Throughput (elements/second)

Plot log-log to verify complexity class:
- O(n) → slope ≈ 1
- O(n log n) → slope ≈ 1.0-1.3
- O(n²) → slope ≈ 2

### Peer Benchmark

Same inputs, same parameters, compare:
- scipy (via Python subprocess or pre-computed)
- R (via pre-computed)
- Our implementation

Document the speedup factor at each scale.

### Numerical Precision Benchmark

At each scale, compare to mpmath at 50-digit precision.
Record max |error| and max relative error.
Document the precision floor (where floating-point noise dominates).

---

## Workup File Organization

```
docs/research/workups/<module>/<primitive>.md    ← the prose workup
tests/workup_<primitive>.rs                      ← the test suite

workup_<primitive>.rs contains:
  - Oracle parity tests (hardcoded expected values from mpmath)
  - Edge case tests
  - Invariant tests (random inputs, verify properties)
  - Adversarial tests
  - Scale ladder benchmarks (behind #[ignore] for CI speed)
```

---

## Primitives That Need Workups (Priority)

### Have Workups:
- kendall_tau ✓
- pearson_r ✓
- inversion_count ✓
- incomplete_beta ✓

### Need Workups — Tier 1 (core primitives used by many methods):
1. **spearman** — rank + Pearson composition, verify decomposition
2. **sample_entropy** — no competing implementations agree on edge cases
3. **dfa** — DFA alpha values vary across implementations
4. **svd** — foundational, must be bit-perfect vs LAPACK
5. **cholesky** — foundational, verify positive definiteness detection
6. **shannon_entropy** — deceptively simple, log(0) handling critical
7. **kl_divergence** — infinity handling, asymmetry
8. **mutual_information** — bias correction, small-sample behavior
9. **garch11_fit** — optimizer convergence, stationarity constraints
10. **adf_test** — critical values, lag selection, regression variant

### Need Workups — Tier 2 (complex implementations):
11. permutation_entropy
12. hurst_rs
13. mfdfa
14. ccm (convergent cross mapping)
15. pelt (changepoint)
16. bocpd (Bayesian changepoint)
17. stl_decompose
18. logistic_regression
19. hierarchical_clustering
20. shapiro_wilk

---

## The Workup Checklist (Filter Test for Rigor)

Before any primitive is called "done":

- [ ] Mathematical definition written with canonical reference
- [ ] All assumptions listed (runtime-checked, documented, or justified)
- [ ] All parameters documented (name, type, range, default, meaning, when to tune)
- [ ] All edge cases listed with expected behavior
- [ ] Oracle values computed via mpmath at 50+ digits
- [ ] Parity verified against scipy AND R (or equivalent)
- [ ] Discrepancies analyzed (our bug or theirs?)
- [ ] Invariant tests written (symmetry, shift/scale invariance, bounds)
- [ ] Adversarial tests written (ties, extreme values, degenerate inputs)
- [ ] Scale ladder run at 7 scales
- [ ] Complexity class verified from log-log slope
- [ ] Peer benchmark vs scipy/R recorded
- [ ] Numerical precision floor documented
