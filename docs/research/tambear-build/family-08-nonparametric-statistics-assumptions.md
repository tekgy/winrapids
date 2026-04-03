# Family 08: Non-Parametric Statistics — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (Commutative) — rank-based tests are sort → accumulate(rank_expr, Sum)

---

## Core Insight: The Ranking Primitive

Nearly every method in this family starts with **ranking**. This is the shared foundational primitive:

1. Radix sort → permutation π
2. Detect ties (parallel adjacent comparison)
3. Compute average ranks via segmented scan
4. Pre-compute tie correction sums: Σtⱼ, Σtⱼ², Σtⱼ³, Σtⱼ(tⱼ-1), Σtⱼ(tⱼ-1)(tⱼ-2), Σtⱼ(tⱼ-1)(2tⱼ+5)

**Rank once → feed 6+ tests.** Same sharing pattern as sort-once for descriptive stats.

---

## 1. Mann-Whitney U (Wilcoxon Rank-Sum)

### Formula
Pool n₁+n₂=N observations, rank 1..N. R₁ = rank sum of group 1.
```
U₁ = n₁n₂ + n₁(n₁+1)/2 - R₁
```

### Null Distribution
- Exact (n₁+n₂ ≤ 20): DP table, O(n₁·n₂·N)
- Normal approximation: z = (U₁ - n₁n₂/2) / √(Var) where
```
Var = (n₁n₂/12)(N+1 - Σ(tⱼ³-tⱼ)/(N(N-1)))
```

### Key: k=2 Kruskal-Wallis gives H = z² (same test, squared)

### Edge Cases
- All values identical → U = n₁n₂/2, Var = 0, return p = 1.0
- n₁=0 or n₂=0 → NaN

---

## 2. Wilcoxon Signed-Rank (Paired)

### Formula
dᵢ = xᵢ - yᵢ. Remove zeros (nᵣ = count of nonzero d). Rank |dᵢ| with ties.
```
W⁺ = Σ Rᵢ for dᵢ > 0
```

### Null Distribution
- Exact (nᵣ ≤ 25): enumerate 2^nᵣ sign assignments, DP table
- Normal: E[W⁺] = nᵣ(nᵣ+1)/4, Var = nᵣ(nᵣ+1)(2nᵣ+1)/24 - Σ(tⱼ³-tⱼ)/48

### Edge Cases
- All dᵢ = 0 → nᵣ = 0 → undefined → p = 1.0
- nᵣ = 1 → p = 1.0 (two-sided)

---

## 3. Kruskal-Wallis H

### Formula
k groups, sizes n₁..nₖ, total N. Rⱼ = rank sum of group j.
```
H = (12/(N(N+1))) · Σ(Rⱼ²/nⱼ) - 3(N+1)
H_corrected = H / (1 - Σ(tⱼ³-tⱼ)/(N³-N))
```

### Distribution: χ²(k-1) asymptotically

### Post-hoc: Dunn's test with Bonferroni/Holm correction
```
z_ij = (R̄ᵢ - R̄ⱼ) / √((N(N+1)/12)(1/nᵢ + 1/nⱼ)/C)
```

---

## 4. Friedman Test

### Formula
n subjects, k treatments. Rank within each block.
```
χ²_F = (12/(nk(k+1))) · Σ Rⱼ² - 3n(k+1)
```

F-form (better): F_F = (n-1)χ²_F / (n(k-1) - χ²_F), distributed F(k-1, (n-1)(k-1))

### Post-hoc: Nemenyi test using Studentized Range

---

## 5. Kolmogorov-Smirnov

### Formula (one-sample)
```
D⁺ = max_i(i/n - F₀(x_(i)))
D⁻ = max_i(F₀(x_(i)) - (i-1)/n)
D = max(D⁺, D⁻)
```

### CRITICAL: Lilliefors correction needed when parameters are estimated from data
Standard KS critical values are WRONG for composite hypotheses (e.g., "is this normal with unknown μ,σ?"). Must use Lilliefors tables or Monte Carlo.

### p-value: Kolmogorov distribution K(t) = 1 - 2·Σ(-1)^(k+1)·e^(-2k²t²)

### Two-sample: D_mn = max|F_m(x) - G_n(x)|

---

## 6. Anderson-Darling

### Formula
Sort data, compute zᵢ = F₀(x_(i)):
```
A² = -n - (1/n)·Σ(2i-1)[ln(zᵢ) + ln(1-z_(n+1-i))]
```

### CRITICAL: p-value formulas are DISTRIBUTION-SPECIFIC
- For normality: A*² = A²·(1 + 0.75/n + 2.25/n²), then piecewise polynomial
- For exponentiality: different modification factor
- For uniformity: different again

This is NOT like KS where one distribution covers all cases.

### Edge Cases
- zᵢ = 0 or zᵢ = 1 → ln(0) = -∞ → A² = ∞. Clamp zᵢ to [ε, 1-ε] where ε ≈ 10⁻¹⁰.

---

## 7. Shapiro-Wilk

### Formula
```
W = (Σ aᵢ·x_(i))² / Σ(xᵢ-x̄)²
```

where aᵢ coefficients come from expected normal order statistics.

### CRITICAL: aᵢ depend only on n, not on data
Pre-compute and cache for all n up to 5000. Use Royston (1995) polynomial approximations for n ≥ 12.

### p-value: Royston transform
```
z = (ln(1-W) - μ(n)) / σ(n)
```
where μ and σ are polynomials in ln(n). Then p = Φ(z).

### Limitation: n ≤ 5000. For larger n, use Anderson-Darling.

---

## 8. Bootstrap Methods

### 8a. Percentile
Sort B bootstrap estimates, take α/2 and (1-α/2) quantiles.

### 8b. BCa (bias-corrected and accelerated)
```
ẑ₀ = Φ⁻¹(#{θ̂*ᵇ < θ̂} / B)
â = Σ(θ̂_(·) - θ̂_(-i))³ / (6·[Σ(θ̂_(·) - θ̂_(-i))²]^(3/2))
```
Then adjusted percentiles α₁, α₂ using ẑ₀ and â.

### GPU: MOST GPU-FRIENDLY METHOD IN THE FAMILY
- B independent bootstrap samples = embarrassingly parallel
- Each thread block handles one bootstrap sample
- For simple statistics (mean, variance): avoid materializing resampled data

### Choice of B: ≥1000 for CI, ≥2000 for BCa, ≥999 for hypothesis testing

---

## 9. Permutation Tests

### Formula
```
p = (#{|T_π| ≥ |T_obs|} + 1) / (B + 1)
```
The +1/+1 (Phipson-Smyth 2010) ensures valid p-value.

### GPU: Embarrassingly parallel across permutations
- Exact (n ≤ 20): enumerate all C(N,m) combinations
- Approximate (B ~ 10000): random permutations via Fisher-Yates

---

## 10. Spearman Rank Correlation

### Formula (with ties)
```
rₛ = Pearson(R(x), R(y))
```
where R denotes ranks with average tie handling.

**WITHOUT ties only**: rₛ = 1 - 6·Σdᵢ²/(n(n²-1))

### TRAP: The shortcut formula is WRONG with ties. Must use Pearson-on-ranks.

### Significance: t = rₛ·√((n-2)/(1-rₛ²)), distributed t(n-2)

---

## 11. Kendall's Tau

### Three variants
- τ-a: (C-D)/C(n,2) — ignores ties
- **τ-b**: (C-D)/√((C(n,2)-Tₓ)(C(n,2)-Tᵧ)) — corrects for ties ← USE THIS
- τ-c: for ordinal contingency tables

### Computation
- O(n²) brute force: GPU-friendly for n ≤ 50K
- O(n log n) merge sort: better for large n but harder to parallelize

### Tie-corrected variance for significance testing:
```
Var(τ_b) = (v₀ - vₜ - vᵤ)/18 + v₁/(2n(n-1)) + v₂/(9n(n-1)(n-2))
```
where v₀, vₜ, vᵤ, v₁, v₂ all depend on tie structures.

---

## 12. Kernel Density Estimation

### Formula
```
f̂(y) = (1/nh)·Σ K((y-xᵢ)/h)
```

### Bandwidth selection
| Method | Formula | Quality |
|--------|---------|---------|
| Silverman | 0.9·min(σ̂, IQR/1.34)·n^(-1/5) | Good default |
| Scott | 1.06·σ̂·n^(-1/5) | Assumes normality |
| LSCV | Minimize integrated squared error | Best for non-normal |
| Sheather-Jones | Plug-in method | Generally best |

### GPU strategies
- Direct summation: O(nm), good for n ≤ 10⁶
- FFT binning: O(n + M·log(M)), excellent for large n
- LSCV bandwidth selection: O(n²) pairwise sums = tiled accumulate

### Epanechnikov is theoretically optimal but Gaussian is universally used (efficiency difference < 5%, smoother, simpler)

---

## Sharing Surface

### From the ranking primitive:
```
rank(data) → Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman, Spearman, Kendall
```

### From sorted data:
```
sort(data) → KS, Anderson-Darling, Shapiro-Wilk
```

### Independent:
```
original data → Bootstrap, Permutation, KDE
```

### The TamSession pattern:
```toml
[sharing]
provides_to_session = ["RankedData(data_id)", "TieStructure(data_id)"]
consumes_from_session = ["RankedData(data_id)", "TieStructure(data_id)"]
```

---

## Implementation Priority

**Phase 1** — Rank-based tests (share ranking primitive):
1. Ranking primitive (sort + tie handling + average ranks)
2. Mann-Whitney U
3. Wilcoxon signed-rank
4. Kruskal-Wallis H + Dunn's post-hoc
5. Spearman rank correlation

**Phase 2** — Normality tests:
6. Kolmogorov-Smirnov (one-sample, two-sample)
7. Anderson-Darling (normality, with Lilliefors)
8. Shapiro-Wilk (with Royston coefficients)

**Phase 3** — Resampling + rank association:
9. Bootstrap (percentile, BCa)
10. Permutation tests
11. Kendall's tau-b
12. Friedman test

**Phase 4** — Density estimation:
13. KDE (Gaussian kernel, Silverman bandwidth)
14. KDE (all kernels, CV bandwidth selection)
