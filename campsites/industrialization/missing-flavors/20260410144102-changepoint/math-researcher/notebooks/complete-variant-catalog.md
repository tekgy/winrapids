# Changepoint Detection — Complete Variant Catalog

## What Exists

### In tambear::time_series
- `cusum_mean(data)` — CUSUM for mean shifts
- `cusum_binary_segmentation(data, min_seg, penalty)` — recursive CUSUM
- `pelt(data, min_seg, penalty)` — PELT (Pruned Exact Linear Time), Killick 2012
- `bocpd(data, max_run, hazard, threshold)` — Bayesian Online CPD, Adams & MacKay 2007
- `zivot_andrews_test(data)` — structural break in unit root context

### In tambear-fintek::family21_changepoint
- `pelt(returns)` → PeltResult (bridges to tambear::time_series::pelt)
- `bocpd(returns)` → BocpdResult (bridges to tambear::time_series::bocpd)

---

## What's MISSING — Complete Catalog

### A. Offline (Retrospective) Methods

1. **Binary segmentation** (BinSeg) — Scott & Knott 1974
   - Already have `cusum_binary_segmentation` but needs parameterization:
   - Missing: cost function choice (L2, L1, Poisson, negative binomial, MBIC)
   - Missing: stopping criterion variants (BIC, mBIC, SIC)
   - Parameters: `data`, `cost_fn`, `penalty`, `min_seg`, `max_changepoints`

2. **Wild Binary Segmentation** (WBS) — Fryzlewicz 2014
   - Random intervals instead of always halving
   - More robust to changepoints near boundaries
   - Parameters: `data`, `n_intervals`, `penalty`, `min_seg`, `seed`
   - Primitives: random_interval_sample → CUSUM on each → threshold

3. **Wild Binary Segmentation 2** (WBS2) — Fryzlewicz 2020
   - Deterministic intervals, solution path approach
   - Parameters: similar to WBS but no seed

4. **Narrowest Over Threshold** (NOT) — Baranowski et al. 2019
   - Finds shortest interval exceeding detection threshold
   - Better power for detecting short-lived changes
   - Parameters: `data`, `penalty`, `min_seg`

5. **Segment Neighborhood** — Auger & Lawrence 1989
   - Exact DP for known number of segments
   - O(Qn²) where Q = number of segments
   - Parameters: `data`, `n_segments`, `cost_fn`

6. **Optimal Partitioning** (OP) — Jackson et al. 2005
   - Exact DP with penalty, PELT is the pruned version
   - Already have PELT; this is the non-pruned exact version
   - Only needed for: verification, or when PELT pruning assumptions fail

7. **CROPS** (Changepoints for Range of Penalties) — Haynes et al. 2017
   - Runs PELT over a range of penalties, returns the segmentation path
   - Parameters: `data`, `penalty_range: (f64, f64)`, `cost_fn`
   - Output: `Vec<(f64, Vec<usize>)>` — penalty → changepoints

8. **E-Divisive** — Matteson & James 2014
   - Nonparametric, based on energy statistics (Euclidean distance)
   - Works for multivariate data
   - Parameters: `data`, `alpha` (moment index, default 1), `min_seg`, `perm_tests`
   - Primitive: energy_statistic between two samples

9. **E-CP3O** — James & Matteson 2015
   - Online version of E-Divisive
   - Parameters: similar + `window_size`

10. **Kernel change point** (KCP) — Arlot et al. 2019
    - Kernel-based cost function: C(segment) = Σᵢ Σⱼ k(xᵢ,xⱼ) / |segment|
    - Parameters: `data`, `kernel`, `penalty`, `min_seg`
    - Advantage: detects changes in distribution, not just mean

### B. Online Methods

1. **CUSUM variants**:
   - **Page's CUSUM** (1954) — the original, already implemented
   - **Geometric moving average CUSUM** — exponentially weighted
   - **Multichart CUSUM** — monitors mean, variance, and shape simultaneously
   - **Multivariate CUSUM** — for vector-valued data
   - Parameters: `data`, `target_arl0` (average run length under H0)

2. **EWMA control chart** — Exponentially Weighted Moving Average
   - λ-weighted average, detect when Z_t exceeds control limits
   - Parameters: `data`, `lambda`, `L` (control limit factor)
   - Primitives: exponential moving average → threshold test

3. **Shiryaev-Roberts** — Bayesian sequential detector
   - R_t = (1 + R_{t-1}) × (p₁(x_t) / p₀(x_t))
   - Parameters: `data`, `threshold`, `pre_change_dist`, `post_change_dist`

4. **Generalized Likelihood Ratio** (GLR)
   - max_{τ,θ} log(L(data|change at τ, parameter θ))
   - Parameters: `data`, `window`, `model` (Gaussian, Poisson, etc.)
   - Most powerful for parametric alternatives

5. **Score-CUSUM** (SCUSUM) — Mei 2010
   - Based on score function instead of likelihood ratio
   - Computationally cheaper than GLR

### C. Cost Functions (needed by PELT/BinSeg/OP)

1. **L2 cost** (mean change) — already used by PELT
   - C(y_{s:t}) = Σ(yᵢ - ȳ)² = Σyᵢ² - (Σyᵢ)²/n
   - Sufficient stats: cumsum, cumsum_of_squares

2. **L1 cost** (median change) — robust to outliers
   - C(y_{s:t}) = Σ |yᵢ - median(y_{s:t})|

3. **Poisson cost** — for count data
   - C(y_{s:t}) = -Σ yᵢ log(ȳ) + nȳ

4. **Gaussian likelihood** (mean+variance change)
   - C(y_{s:t}) = n log(σ̂²) where σ̂² = Σ(yᵢ-ȳ)²/n
   - Detects both mean and variance changes

5. **Negative binomial cost** — overdispersed counts

6. **MBIC cost** — Modified BIC (Zhang & Siegmund 2007)
   - Incorporates segment length into penalty

7. **AR cost** — detects changes in autoregressive parameters
   - C = -2 log L(AR fit on segment)

8. **Rank-based cost** — nonparametric
   - C based on rank statistics (Lung-Yut-Fong et al. 2015)

### D. Multivariate Changepoint

1. **Multivariate PELT** — with multivariate cost functions
   - Parameters: `data: &Mat`, `penalty`, `min_seg`
   - Cost: multivariate Gaussian log-likelihood

2. **Inspect** — Wang & Samworth 2018
   - Projects multivariate data onto informative directions
   - Parameters: `data: &Mat`, `penalty`
   - Handles sparse changes (only some dimensions change)

3. **Double CUSUM** — Cho 2016
   - Aggregates CUSUM across dimensions
   - Parameters: `data: &Mat`, `threshold`

### E. Structural Break Tests

1. **Chow test** — F-test for parameter change at known point
   - Parameters: `x`, `y`, `break_point`
   - Primitives: two OLS regressions → F-statistic

2. **Quandt-Andrews** / **Andrews' sup-Wald** — unknown break point
   - Test all possible break points, take supremum of Wald/LR/LM stats
   - Parameters: `x`, `y`, `trimming` (fraction to exclude from endpoints)
   - Already have Zivot-Andrews; this is the regression coefficient version

3. **Bai-Perron** — multiple structural breaks in regression
   - DP for optimal partition into k segments
   - Parameters: `x`, `y`, `max_breaks`, `min_seg`, `penalty`

4. **KPSS structural break** — break in KPSS-type test
   - Already have KPSS; this extends to break detection

### F. Penalty Selection

1. **BIC / SIC** — log(n) × (number of parameters changed)
2. **mBIC** — modified BIC (accounts for segment lengths)
3. **AIC** — 2 × (number of parameters)
4. **Hannan-Quinn** — 2 log log(n) × params
5. **Cross-validation** — held-out prediction error across folds
6. **Elbow / knee** — CROPS path analysis

---

## Decomposition into Primitives

```
cumulative_sum(data) ──────┬── cusum_statistic
cumulative_sum_sq(data) ───┤
                           ├── pelt_segment_cost (L2)
                           ├── binary_segmentation
                           ├── optimal_partitioning
                           └── crops

ols_fit(segment) ──────────┬── chow_test
                           ├── bai_perron
                           └── ar_cost

energy_statistic(x, y) ────── e_divisive

kernel_matrix(data) ───────── kcp

rank(data) ────────────────── rank_based_cost

exponential_ma(data) ──────── ewma_chart
```

## Sharing Map

| Intermediate | Consumers |
|---|---|
| Cumulative sums (Σ, Σ²) | PELT, BinSeg, OP, CUSUM, CROPS — ALL L2-based methods |
| Segment means/variances | PELT, BinSeg, OP, BOCPD, Gaussian cost |
| Distance matrix | E-Divisive, KCP |
| AR residuals | AR cost, structural break tests |
| OLS coefficients | Chow, Bai-Perron, Zivot-Andrews |

## Priority

**Tier 1** — Missing from current set that are widely used:
1. `wbs` (Wild Binary Segmentation) — most recommended offline method after PELT
2. `e_divisive` — nonparametric, multivariate capable
3. `crops` — penalty selection for PELT
4. `chow_test` — fundamental structural break test
5. Gaussian likelihood cost (mean+variance) for PELT — currently only L2

**Tier 2**:
6. `ewma_chart` — online monitoring
7. `bai_perron` — multiple breaks in regression
8. `kcp` — kernel-based (flexible)
9. Multivariate PELT
10. `wbs2` + `not`
