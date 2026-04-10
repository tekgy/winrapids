# Bootstrap — Complete Variant Catalog

## What Exists (tambear::nonparametric)
- `bootstrap_percentile(data, stat_fn, n_resamples, alpha)` — basic percentile CI

---

## What's MISSING — Complete Catalog

### A. Confidence Interval Methods

1. **Basic bootstrap** (reverse percentile)
   - CI = (2θ̂ - q_{1-α/2}*, 2θ̂ - q_{α/2}*)
   - More conservative than percentile
   - Parameters: `data`, `stat_fn`, `n_resamples`, `alpha`

2. **Bias-corrected (BC) bootstrap** — Efron 1981
   - Adjusts for median bias: z₀ = Φ⁻¹(proportion of θ* < θ̂)
   - CI endpoints at Φ(2z₀ ± z_{α/2})
   - Parameters: same as percentile
   - Primitives: normal_quantile, normal_cdf

3. **BCa (Bias-corrected and accelerated)** — Efron 1987
   - Adds acceleration constant a from jackknife
   - α₁ = Φ(z₀ + (z₀ + z_α)/(1 - a(z₀ + z_α)))
   - Parameters: same + internal jackknife for acceleration
   - Gold standard for nonparametric CIs
   - Primitives: jackknife values → acceleration constant

4. **Bootstrap-t (studentized)** — Efron & Tibshirani 1993
   - Pivotal quantity t* = (θ* - θ̂) / se*
   - CI = (θ̂ - t*_{1-α/2} × se, θ̂ - t*_{α/2} × se)
   - Parameters: `data`, `stat_fn`, `se_fn`, `n_resamples`, `alpha`
   - Needs: standard error estimate for each bootstrap sample
   - Most accurate order (second-order correct)

5. **Double bootstrap** — Hall 1986
   - Calibrates coverage probability using nested bootstrap
   - Extremely compute-intensive: n_outer × n_inner resamples
   - Parameters: `data`, `stat_fn`, `n_outer`, `n_inner`, `alpha`

6. **ABC bootstrap** (Approximate Bootstrap Confidence) — DiCiccio & Efron 1992
   - Analytical approximation to BCa without resampling
   - Uses: influence function, skewness, acceleration
   - Parameters: `data`, `stat_fn`, `alpha`

### B. Resampling Schemes

7. **Parametric bootstrap**
   - Fit distribution → simulate from fitted distribution
   - Parameters: `data`, `distribution`, `stat_fn`, `n_resamples`
   - Use case: when model is known but inference is complex

8. **Residual bootstrap** — for regression
   - Fit model → resample residuals → reconstruct responses
   - Parameters: `x`, `y`, `model_fn`, `stat_fn`, `n_resamples`
   - Assumption: homoscedastic errors

9. **Wild bootstrap** — Wu 1986, Liu 1988
   - For heteroscedastic regression
   - r_t* = r_t × v_t where v_t from auxiliary distribution
   - Rademacher: v_t = ±1 with equal probability
   - Mammen: v_t = -(√5-1)/2 or (√5+1)/2
   - Parameters: `residuals`, `x`, `wild_dist` ("rademacher" | "mammen")

10. **Block bootstrap** — Kunsch 1989, Liu & Singh 1992
    - For time series: resample blocks of consecutive observations
    - Moving block bootstrap (MBB): overlapping blocks of fixed length
    - Parameters: `data`, `block_size`, `stat_fn`, `n_resamples`

11. **Circular block bootstrap** — Politis & Romano 1992
    - Wraps data circularly to avoid end effects
    - Parameters: same as block bootstrap

12. **Stationary bootstrap** — Politis & Romano 1994
    - Random block lengths from geometric distribution
    - Expected block length = 1/p
    - Parameters: `data`, `p` (expected_block_length = 1/p), `stat_fn`, `n_resamples`
    - Produces stationary resampled series

13. **Subsampling** — Politis, Romano & Wolf 1999
    - Use all contiguous sub-sequences of length b
    - More general than bootstrap: works under minimal assumptions
    - Parameters: `data`, `subsample_size`, `stat_fn`

14. **m-out-of-n bootstrap** — Bickel et al. 1997
    - Resample m < n observations
    - Fixes bootstrap failures when √n consistency fails
    - Parameters: `data`, `m`, `stat_fn`, `n_resamples`

15. **Weighted bootstrap** — Newton & Raftery 1994
    - Assign random Dirichlet weights instead of resampling
    - Bayesian interpretation: Dirichlet process posterior
    - Parameters: `data`, `stat_fn`, `n_resamples`

16. **Balanced bootstrap** — Davison et al. 1986
    - Each observation appears exactly n_resamples times across all resamples
    - Reduces variance of bootstrap estimates
    - Parameters: `data`, `stat_fn`, `n_resamples`

### C. Hypothesis Testing

17. **Permutation test** (already have `permutation_test_mean_diff`)
    - Extend to: any test statistic, paired/unpaired, multi-group
    - Parameters: `groups`, `stat_fn`, `n_perms`, `alternative`

18. **Bootstrap hypothesis test**
    - Center bootstrap distribution under H₀
    - p-value = proportion of |θ*| ≥ |θ̂| under centered distribution
    - Parameters: `data`, `stat_fn`, `null_value`, `n_resamples`

19. **Bootstrap p-value calibration** — Hall 1986
    - Prepivoting: p* = G(F̂⁻¹(α)) where G from double bootstrap
    - Corrects bootstrap p-values

### D. Special Applications

20. **Bootstrap for correlation** — Fisher z-transform + bootstrap
    - Parameters: `x`, `y`, `n_resamples`, `alpha`, `method` ("percentile"|"bca")

21. **Bootstrap prediction intervals** — for forecasts
    - Parameters: `forecast_fn`, `residuals`, `horizon`, `n_resamples`, `alpha`

22. **Jackknife** (leave-one-out) — Quenouille 1949, Tukey 1958
    - θ_jack = n×θ̂ - (n-1)/n × Σ θ̂_{-i}
    - se_jack = √((n-1)/n × Σ(θ̂_{-i} - θ̄)²)
    - Parameters: `data`, `stat_fn`
    - Primitives: n evaluations of stat_fn on n-1 samples
    - Already used internally in BCa; should be standalone

23. **Jackknife-after-bootstrap** — Efron 1992
    - Assess bootstrap stability
    - Parameters: `data`, `stat_fn`, `n_resamples`

24. **Delete-d jackknife** — Shao & Wu 1989
    - Leave-d-out generalization
    - Parameters: `data`, `stat_fn`, `d`

---

## Decomposition into Primitives

```
resample_with_replacement(data, seed) ──── ALL non-block bootstraps

resample_blocks(data, block_size, seed) ─── block bootstrap (MBB, CBB)

resample_stationary(data, p, seed) ──────── stationary bootstrap

sort(bootstrap_stats) ──────────────────┬── percentile CI
                                        ├── basic CI
                                        └── bootstrap-t CI

jackknife_values(data, stat_fn) ────────┬── bca acceleration constant
                                        ├── jackknife bias/variance
                                        ├── jackknife-after-bootstrap
                                        └── delete-d jackknife

normal_cdf / normal_quantile ───────────── bc/bca adjustment

stat_fn(resampled_data) ────────────────── ALL variants (the user-supplied statistic)
```

## Sharing Map

| Intermediate | Consumers |
|---|---|
| Bootstrap distribution (sorted θ* values) | percentile, basic, BC, BCa CIs |
| Jackknife values (θ̂_{-i}) | BCa acceleration, jackknife SE, bias |
| Bias estimate z₀ | BC, BCa |
| Standard error estimates se* | Bootstrap-t |

## Priority

**Tier 1** — Widely needed, moderate effort:
1. `bootstrap_bca` — gold standard, most recommended
2. `bootstrap_basic` — simple complement to percentile
3. `jackknife(data, stat_fn)` — standalone, used by BCa and diagnostics
4. `block_bootstrap` — essential for time series
5. `stationary_bootstrap` — best time series bootstrap

**Tier 2**:
6. `bootstrap_studentized` — most accurate, but needs SE function
7. `wild_bootstrap` — heteroscedastic regression
8. `residual_bootstrap` — regression
9. `permutation_test` (generalized) — extend existing
10. `bootstrap_hypothesis_test` — centered H₀ test

**Tier 3**:
11-24: parametric, double, balanced, ABC, subsampling, etc.
