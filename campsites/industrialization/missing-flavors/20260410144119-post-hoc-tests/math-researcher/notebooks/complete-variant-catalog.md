# Post-Hoc Tests — Complete Variant Catalog

## What Exists (tambear::hypothesis + tambear::nonparametric)

### Multiple Comparison Corrections
- `bonferroni(p_values)` — p_adj = min(p × k, 1)
- `holm(p_values)` — Holm step-down
- `benjamini_hochberg(p_values)` — FDR control

### Pairwise Tests
- `tukey_hsd(data, groups, alpha)` — Tukey's HSD (requires balanced + equal var)
- `dunn_test(data, group_sizes)` — Dunn's test for Kruskal-Wallis follow-up

---

## What's MISSING — Complete Catalog

### A. Parametric Post-Hoc Tests (ANOVA follow-up)

1. **Scheffé's method** — Scheffé 1953
   - Most conservative; controls all possible contrasts (not just pairwise)
   - F_crit = (k-1) × F_{k-1, N-k, α}
   - Parameters: `groups: &[&[f64]]`, `alpha`
   - Use when: testing arbitrary linear combinations of means

2. **Tukey-Kramer** — extension of Tukey HSD for unbalanced designs
   - q_crit × √(0.5 × (1/nᵢ + 1/nⱼ)) × MSE
   - Parameters: `groups`, `alpha`
   - Already partially in Tukey HSD but needs unbalanced handling

3. **Dunnett's test** — Dunnett 1955
   - Each treatment vs a single control group
   - Parameters: `control: &[f64]`, `treatments: &[&[f64]]`, `alpha`, `alternative`
   - Fewer comparisons → more power than Tukey when only vs control matters
   - Uses multivariate t-distribution

4. **Games-Howell** — Games & Howell 1976
   - Like Tukey but for unequal variances (and unequal n)
   - Uses Welch-Satterthwaite df for each pair
   - Parameters: `groups`, `alpha`
   - Use when: Levene's test rejects homogeneity of variance

5. **Newman-Keuls** (Student-Newman-Keuls, SNK) — 1939/1952
   - Step-down procedure: test largest range first
   - Uses different critical values for different step sizes
   - Parameters: `groups`, `alpha`
   - Less conservative than Tukey; controls per-comparison rather than familywise

6. **Duncan's multiple range test** — Duncan 1955
   - Like SNK but uses protection levels that increase with step size
   - Parameters: `groups`, `alpha`
   - Even less conservative than SNK
   - Warning: does NOT control familywise error rate

7. **Fisher's LSD** (Least Significant Difference) — Fisher 1935
   - Protected: only test if ANOVA is significant
   - LSD = t_{α/2, N-k} × √(MSE × (1/nᵢ + 1/nⱼ))
   - Parameters: `groups`, `alpha`
   - Most powerful but weakest error control

8. **Tamhane's T2** — Tamhane 1977
   - Conservative test for unequal variances
   - Based on Welch t-tests with Bonferroni correction
   - Parameters: `groups`, `alpha`

9. **Hochberg's GT2** — Hochberg 1974
   - Uses studentized maximum modulus distribution
   - Parameters: `groups`, `alpha`

10. **Gabriel's test** — Gabriel 1978
    - Based on studentized maximum modulus
    - For unequal n; intermediate conservatism
    - Parameters: `groups`, `alpha`

11. **Waller-Duncan** — Waller & Duncan 1969
    - Uses Bayesian criterion (k-ratio)
    - Parameters: `groups`, `k_ratio` (typically 100)

### B. Nonparametric Post-Hoc Tests

12. **Steel-Dwass-Critchlow-Fligner** — pairwise Wilcoxon rank-sum comparisons
    - Nonparametric analog of Tukey
    - Parameters: `groups`, `alpha`
    - Uses studentized range distribution on rank sums

13. **Conover-Iman** — Conover 1999
    - Follow-up to Kruskal-Wallis using ranks
    - t-tests on ranks with pooled variance
    - Parameters: `data`, `group_sizes`, `alpha`

14. **Nemenyi test** — Nemenyi 1963
    - Pairwise comparisons for Friedman test (repeated measures)
    - Parameters: `ranks`, `n_treatments`, `alpha`
    - Uses: q_crit × √(k(k+1)/(6n))

15. **Dwass-Steel** — Mann-Whitney for all pairs
    - Parameters: `groups`, `alpha`

16. **Steel's test** — Steel 1959
    - Nonparametric Dunnett: each treatment vs control
    - Parameters: `control`, `treatments`, `alpha`

### C. Multiple Comparison p-value Adjustments

17. **Hochberg step-up** — Hochberg 1988
    - Less conservative than Holm; requires independence
    - p_adj(j) = max_{k≥j} min(p(k) × (m-k+1), 1)
    - Parameters: `p_values`

18. **Hommel** — Hommel 1988
    - Tighter than Hochberg/Holm; iterative
    - Parameters: `p_values`

19. **Šidák correction** — Šidák 1967
    - p_adj = 1 - (1-p)^k
    - Less conservative than Bonferroni (assumes independence)
    - Parameters: `p_values`

20. **Šidák step-down** — iterative Šidák
    - Parameters: `p_values`

21. **Benjamini-Yekutieli** — BY 2001
    - FDR control under arbitrary dependence
    - Parameters: `p_values`
    - More conservative than BH but valid for dependent tests

22. **Storey's q-value** — Storey 2003
    - Estimates proportion of true nulls π₀
    - More powerful than BH when many tests are truly alternative
    - Parameters: `p_values`, `lambda` (tuning)

23. **Westfall-Young max-T** — Westfall & Young 1993
    - Permutation-based: preserves correlation structure
    - Parameters: `data`, `groups`, `n_perms`
    - Best for: small number of tests with known correlation

### D. Contrast Tests

24. **Custom linear contrasts** — Σ cᵢ μᵢ = 0
    - F = (Σ cᵢ ȳᵢ)² / (MSE × Σ cᵢ²/nᵢ)
    - Parameters: `groups`, `contrast_weights`
    - Orthogonal contrasts: independent tests

25. **Polynomial contrasts** — linear, quadratic, cubic trends
    - Parameters: `groups` (ordered levels)
    - Coefficients from orthogonal polynomial tables

26. **Helmert contrasts** — each level vs mean of subsequent levels
    - Parameters: `groups`

27. **Simple contrasts** — each level vs reference level
    - Like Dunnett but as contrast framework

---

## Decomposition into Primitives

```
anova_mse(groups) ──────────┬── tukey_hsd
                            ├── scheffe
                            ├── fisher_lsd
                            ├── newman_keuls
                            ├── duncan
                            ├── dunnett
                            └── contrast tests

welch_satterthwaite_df ─────┬── games_howell
                            ├── tamhane_t2
                            └── tukey_kramer

studentized_range_cdf ──────┬── tukey_hsd / tukey_kramer
                            ├── newman_keuls
                            └── steel_dwass

rank(data) ─────────────────┬── dunn_test
                            ├── conover_iman
                            ├── nemenyi
                            └── steel_dwass

sort(p_values) ─────────────┬── holm (step-down)
                            ├── hochberg (step-up)
                            ├── hommel
                            ├── sidak_step
                            └── benjamini_yekutieli
```

## Priority

**Tier 1** — Most commonly requested:
1. `games_howell(groups, alpha)` — unequal variance pairwise (very commonly needed)
2. `dunnett(control, treatments, alpha)` — treatment vs control
3. `scheffe(groups, alpha)` — arbitrary contrasts
4. `sidak(p_values)` — better than Bonferroni
5. `hochberg_stepup(p_values)` — better than Holm

**Tier 2**:
6. `fisher_lsd(groups, alpha)` — least conservative
7. `nemenyi(ranks, n_treatments, alpha)` — Friedman follow-up
8. `conover_iman(data, groups, alpha)` — Kruskal-Wallis follow-up
9. `steel_dwass(groups, alpha)` — nonparametric Tukey
10. `benjamini_yekutieli(p_values)` — FDR under dependence
11. `custom_contrast(groups, weights)` — linear contrasts

**Tier 3**:
12-27: Newman-Keuls, Duncan, Tamhane, Gabriel, polynomial contrasts, etc.
