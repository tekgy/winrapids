# Distributional Distance Genealogy

## The Two Great Families

All distributional distances in tambear belong to one of two families:

### Family 1: f-Divergences (Csiszár 1963, Ali & Silvey 1966)

D_f(P||Q) = Σ q(x) f(p(x)/q(x))

where f is convex with f(1) = 0.

**Properties** (all f-divergences share these):
- Non-negative (by Jensen's inequality)
- Zero iff P = Q (if f is strictly convex at 1)
- Data processing inequality: D_f(T(P)||T(Q)) ≤ D_f(P||Q)
- Invariant under sufficient statistics

**Members in tambear**:

| Measure | Generator f(t) | Symmetric? | Metric? | In tambear? |
|---|---|---|---|---|
| KL divergence | t log t | NO | NO | ✓ `kl_divergence` |
| Reverse KL | -log t | NO | NO | via `f_divergence` |
| Chi-squared | (t-1)² | NO | NO | ✓ `chi_squared_divergence` |
| Neyman chi-squared | (1-t)²/t | NO | NO | via `f_divergence` |
| Squared Hellinger | (√t - 1)² | YES | YES (as √H²) | ✓ `hellinger_distance` |
| Total variation | |t-1|/2 | YES | YES | ✓ `total_variation_distance` |
| Jensen-Shannon | (t log t - (t+1)log((t+1)/2))/2 | YES | YES (as √JS) | ✓ `js_divergence` |
| Alpha-divergence | (t^α - αt + α - 1)/(α(α-1)) | NO (except α=½) | NO | via `f_divergence` |
| Rényi divergence | not strictly f-div but related | NO | NO | ✓ `renyi_divergence` |

**Hierarchy of bounds**:
```
TV ≤ √(KL/2)                    [Pinsker's inequality]
TV ≤ √(2·JS)                    [tighter Pinsker for JS]
TV ≤ H ≤ √(2·TV)               [Hellinger-TV]
KL ≤ χ²                         [quadratic dominates log]
H² ≤ KL ≤ √(2·χ²)             [Hellinger-KL-chi chain]
JS ≤ KL                         [JS is always tighter]
D_α(P||Q) is non-decreasing in α [Rényi monotonicity]
```

### Family 2: Integral Probability Metrics (IPMs) (Müller 1997)

γ_F(P,Q) = sup_{f ∈ F} |E_P[f(X)] - E_Q[f(X)]|

where F is a function class.

**Properties** (all IPMs share these):
- Non-negative
- Symmetric
- Satisfy triangle inequality → true metrics
- Zero iff P = Q (when F is rich enough)

**Members in tambear**:

| Measure | Function class F | In tambear? |
|---|---|---|
| Total variation | {f : ||f||_∞ ≤ 1} | ✓ (also an f-div!) |
| Wasserstein-1 | {f : ||f||_Lip ≤ 1} (1-Lipschitz) | ✓ `wasserstein_1d` |
| Wasserstein-p | (inf_γ ∫ d^p dγ)^{1/p} | ✗ (taxonomy mentions sinkhorn) |
| MMD | {f : ||f||_H ≤ 1} (unit ball in RKHS) | ✓ `mmd_rbf` |
| Kolmogorov-Smirnov | {1_{(-∞,t]} : t ∈ R} | ✓ (in nonparametric) |
| Energy distance | {f : ||f||_∞ ≤ 1, concave} | ✓ `energy_distance` |

**Hierarchy**:
```
TV ≥ W_1 / diam(X)              [TV dominates normalized W1]
KS ≤ TV                          [KS tests fewer functions]
W_1 ≤ W_p × diam(X)^{1-1/p}    [Wasserstein ordering]
MMD ≤ TV (for characteristic k)  [MMD is tighter]
```

### The Bridge: Total Variation

Total variation is BOTH an f-divergence AND an integral probability metric:
- f-divergence with f(t) = |t-1|/2
- IPM with F = {f : ||f||_∞ ≤ 1}

This makes TV the unique bridge between the two families.

---

## Relationship Diagram

```
                        f-DIVERGENCES
                            |
                   ┌────────┼────────┐
                   |        |        |
                KL div   χ² div   Rényi div
                 |  \      |
                 |   JS    |
                 |   /     |
            Hellinger      |
                 \         |
                  \        |
          TOTAL VARIATION ←┘    ← BRIDGE
                  /
                 /
        ┌───────┴───────────┐
        |                   |
    Wasserstein          KS test
        |                   
    Energy dist          
        |
      MMD (RKHS)
        |
    INTEGRAL PROBABILITY METRICS
```

## Computational Complexity

| Measure | Discrete (k bins) | Samples (n,m) | Notes |
|---|---|---|---|
| KL, JS, Hellinger, TV, χ² | O(k) | O(n+m) after histogramming | Need same discretization |
| Bhattacharyya | O(k) | O(n+m) after histogramming | Needs sqrt per bin |
| Rényi divergence | O(k) | O(n+m) after histogramming | Needs pow per bin |
| KS test | - | O(n log n + m log m) | Sort-based |
| Wasserstein-1 (1D) | O(k) | O(n log n) | Sort-based |
| Wasserstein-p (general) | O(n³) exact, O(n²/ε²) Sinkhorn | - | Sinkhorn for large n |
| MMD | - | O(n² + m²) | Kernel matrix |
| Energy distance | - | O(n² + m²) | Pairwise distances |
| f-divergence (general) | O(k) | O(n+m) after histogramming | User-supplied f |

## What's Still Missing

### From f-divergence family:
1. **Alpha-divergence** — parameterized by α, gives different members
2. **Rényi divergence order selection** — which α to use?
3. **Symmetric KL** — (KL(P||Q) + KL(Q||P))/2 (not a named function)

### From IPM family:
4. **Wasserstein-p (p > 1)** — Sinkhorn algorithm (Kingdom C)
5. **Sinkhorn divergence** — entropy-regularized OT (debiased)
6. **Cramér distance** — ∫(F_P - F_Q)² dx (L2 analog of Wasserstein-1's L1)

### From neither family:
7. **Stein discrepancy** — based on Stein's identity, needs score function
8. **Fisher-Rao distance** — geodesic on statistical manifold
9. **Bregman divergence** — D_φ(p,q) = φ(p) - φ(q) - ⟨∇φ(q), p-q⟩

### Composition primitives:
10. **Symmetrize any divergence** — S(P,Q) = (D(P||Q) + D(Q||P))/2
11. **Metrize any divergence** — take sqrt if D satisfies D(P,Q)² ≤ D(P,Q)
12. **Mixture of distances** — weighted combination for ensemble comparison

## Sharing Opportunities

All f-divergences share the same histogram → probabilities step.
All sample-based IPMs share pairwise distance computation.
All kernel-based methods share the kernel matrix.

The `f_divergence(p, q, f)` primitive is the universal gateway:
any named f-divergence is just `f_divergence(p, q, |t| specific_f(t))`.
The named variants (kl, hellinger, etc.) exist for:
- Documentation (what the user asked for)
- Edge case handling (specific to each divergence)
- Performance (avoid closure overhead for hot paths)
