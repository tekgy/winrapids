<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

# Mean Variants — Complete Variant Catalog

## What Exists (tambear::descriptive)

- `moments_ungrouped(values)` — arithmetic mean + variance + skewness + kurtosis
- `geometric_mean(values)` — exp(mean(log(x)))
- `harmonic_mean(values)` — n / Σ(1/xᵢ)
- `trimmed_mean(sorted, fraction)` — remove fraction from each tail
- `winsorized_mean(sorted, fraction)` — clamp extremes

## What Exists (tambear::numerical)
- `weighted_mean(x, w)` — Σ wᵢxᵢ / Σ wᵢ

---

## What's MISSING — Complete Catalog

### A. Generalized Means (Power Mean Family)

All members of M_p(x) = (Σ xᵢ^p / n)^{1/p}:

1. **Power mean (generalized mean)** — M_p(x)
   - Parameters: `values`, `p: f64`
   - Special cases: p=-1 → harmonic, p=0 → geometric (limit), p=1 → arithmetic, p=2 → RMS
   - Already have p=-1,0,1; need general p and p=2

2. **Root mean square** (RMS / quadratic mean) — M_2(x) = √(Σxᵢ²/n)
   - Parameters: `values`
   - Use case: signal processing, voltage, error measures

3. **Cubic mean** — M_3(x) = (Σxᵢ³/n)^{1/3}
   - Parameters: `values`

4. **Contraharmonic mean** — C(x) = Σxᵢ² / Σxᵢ
   - Parameters: `values`
   - Always ≥ arithmetic mean (biased toward larger values)

5. **Lehmer mean** — L_p(x) = Σxᵢ^p / Σxᵢ^{p-1}
   - Parameters: `values`, `p: f64`
   - p=0 → harmonic, p=1 → arithmetic, p=2 → contraharmonic
   - Different family from power means but related

### B. Robust Location Estimators

6. **Huber M-estimator** — iteratively reweighted least squares
   - Parameters: `values`, `k` (threshold, default 1.345 for 95% efficiency)
   - ψ(u) = u if |u|≤k, k×sign(u) otherwise
   - Breakdown point: adjustable via k

7. **Tukey biweight (bisquare) M-estimator**
   - Parameters: `values`, `c` (default 4.685 for 95% efficiency)
   - ψ(u) = u(1-(u/c)²)² if |u|≤c, 0 otherwise
   - Zero influence for extreme outliers (hard rejection)

8. **Hodges-Lehmann estimator** — median of pairwise averages
   - HL = median{(xᵢ + xⱼ)/2 : i ≤ j}
   - Parameters: `values`
   - O(n² log n) — compute all pairwise averages, take median
   - More efficient than median, more robust than mean

9. **Gastworth mean** — G = 0.3 × Q₁ + 0.4 × median + 0.3 × Q₃
   - Parameters: `sorted`
   - Simple, easy to compute

10. **Interquartile mean** (IQM) — mean of values between Q₁ and Q₃
    - Parameters: `sorted`
    - 25% trimmed mean (already have trimmed_mean with fraction=0.25)
    - Should be a named alias

11. **Midhinge** — (Q₁ + Q₃) / 2
    - Parameters: `sorted`
    - Simple robust central tendency

12. **Trimean** — (Q₁ + 2×median + Q₃) / 4
    - Parameters: `sorted`
    - Tukey's trimean — weights median more heavily

13. **Truncated mean** — same as trimmed_mean but different naming convention
    - Alias for existing `trimmed_mean`

### C. Weighted/Exponential Means

14. **Exponential moving average** — EMA_t = α×x_t + (1-α)×EMA_{t-1}
    - Parameters: `values`, `alpha` (or `span`)
    - Already have in time_series as `simple_exponential_smoothing`
    - Should be a first-class mean primitive

15. **Weighted geometric mean** — exp(Σ wᵢ log xᵢ / Σ wᵢ)
    - Parameters: `values`, `weights`

16. **Weighted harmonic mean** — Σwᵢ / Σ(wᵢ/xᵢ)
    - Parameters: `values`, `weights`
    - Use case: averaging rates, F-measure

17. **Kernel-weighted mean** — Σ K(xᵢ-c, h) × xᵢ / Σ K(xᵢ-c, h)
    - Parameters: `values`, `center`, `bandwidth`, `kernel`
    - Local averaging with kernel weights

### D. Specialized Means

18. **Fréchet mean** — minimizes Σ d(x, μ)² for metric space distance d
    - Parameters: `points`, `distance_fn`, `tol`, `max_iter`
    - Iterative: Weiszfeld's algorithm for L₁, gradient descent for general
    - Use case: means on manifolds, non-Euclidean spaces

19. **Karcher mean** — Fréchet mean on Riemannian manifold
    - Specific to: positive definite matrices (geometric mean of matrices)
    - Parameters: `matrices: &[Mat]`
    - Use case: diffusion tensor imaging, covariance averaging

20. **Circular mean** — mean direction on a circle
    - θ̄ = atan2(Σ sin θᵢ, Σ cos θᵢ)
    - Parameters: `angles` (in radians)
    - Use case: directional statistics, phase averaging

21. **Log mean** — (a-b) / (ln a - ln b)
    - Parameters: `a`, `b` (two values, not a set)
    - Satisfies: min(a,b) ≤ log_mean ≤ (a+b)/2
    - Use case: heat transfer, thermodynamics

22. **Stolarsky mean** — S_p(a,b) = ((a^p - b^p)/(p(a-b)))^{1/(p-1)}
    - Parameters: `a`, `b`, `p`
    - Unifying family: p=1 → log mean, p=2 → arithmetic, p→0 → geometric

23. **Heronian mean** — H(a,b) = (a + √(ab) + b) / 3
    - Parameters: `a`, `b`

24. **Identric mean** — I(a,b) = (1/e)(b^b/a^a)^{1/(b-a)}
    - Parameters: `a`, `b`

### E. Running/Online Means

25. **Welford's online mean/variance** — numerically stable
    - Already used internally in MomentStats; should be exposed
    - Parameters: streaming updates

26. **Exponentially weighted mean and variance** (EWMV)
    - Parameters: `values`, `alpha`
    - For variance: V_t = (1-α)(V_{t-1} + α(x_t - μ_{t-1})²)

---

## Relationships and Inequalities

```
harmonic_mean ≤ geometric_mean ≤ arithmetic_mean ≤ rms ≤ contraharmonic
     (p=-1)          (p=0)            (p=1)        (p=2)     (Lehmer p=2)

For p < q: power_mean(p) ≤ power_mean(q)  [strict if values not all equal]

trimmed_mean(0.0) = arithmetic_mean
trimmed_mean(0.5) = median
winsorized_mean(0.0) = arithmetic_mean
```

## Decomposition into Primitives

```
sort(values) ──────────┬── trimmed_mean (index bounds)
                       ├── winsorized_mean (clamp at quantiles)
                       ├── interquartile_mean
                       ├── gastworth
                       ├── trimean
                       ├── midhinge
                       ├── hodges_lehmann
                       └── median

sum(values) ───────────┬── arithmetic_mean
sum(1/values) ─────────── harmonic_mean
sum(log(values)) ──────── geometric_mean
sum(values^p) ─────────── power_mean(p)
sum(values^p)/sum(values^{p-1}) ── lehmer_mean(p)

weighted_sum(values, w) ── weighted_mean (all weighted variants)

iterative_reweight ────┬── huber_m_estimator
                       └── tukey_biweight

atan2(sum_sin, sum_cos) ── circular_mean
```

## Priority

**Tier 1** — Should exist now:
1. `power_mean(values, p)` — unifies 5 existing variants into one parameterized family
2. `rms(values)` — extremely common (p=2 case)
3. `trimean(sorted)` — trivial from existing quartiles
4. `midhinge(sorted)` — trivial from existing quartiles
5. `hodges_lehmann(values)` — gold-standard robust estimator
6. `huber_m_estimator(values, k)` — standard robust location

**Tier 2**:
7. `lehmer_mean(values, p)` — second parameterized family
8. `circular_mean(angles)` — needed for any angular/phase data
9. `contraharmonic_mean(values)` — Lehmer p=2
10. `tukey_biweight(values, c)` — common robust alternative to Huber
11. `weighted_geometric_mean` / `weighted_harmonic_mean`

**Tier 3**:
12-24: Fréchet, Karcher, Stolarsky, Heronian, identric, etc.


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

