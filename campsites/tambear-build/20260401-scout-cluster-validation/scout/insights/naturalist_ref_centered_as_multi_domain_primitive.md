# Naturalist: RefCenteredStats Is a Multi-Domain Primitive

The `RefCenteredStatsEngine` scaffold in `stats.rs` was designed for numerically stable
variance: `Σ(x - ref_group)²`. But the "centered accumulation" pattern — scatter a
function of `(v - r)` grouped by some key — appears in at least four different domains
with completely different semantics.

---

## The Pattern

`scatter(ByKey{groups, K}, phi(v, r), Add)` where r = per-group reference value.

The `phi` function changes per domain. The grouping changes per domain.
The accumulate infrastructure is identical.

---

## Domain 1: Statistics — Variance (stats.rs)

```
phi(v, r) = (v - r)²    r = group mean estimate
result:     Σ(x - mean)² per group → divide by n → variance
```

The reference r is a pre-estimated group mean (first scatter: sum/count → mean estimate,
second scatter: sum((x - mean)²)). Two-pass algorithm avoids Welford division overhead.

---

## Domain 2: ML Clustering — HDBSCAN Stability

```
phi(v, r) = v - r       r = λ_birth (density at cluster birth)
result:     Σ(λ_p - λ_birth) per cluster → cluster stability score
```

The reference r is the density threshold at which this cluster was born.
Per-cluster reference, same scattered-sum structure.

From the original HDBSCAN paper (Campello 2013):
```
S(C_i) = Σ_{p ∈ C_i} (λ_p - λ_birth_i)
```

This is literally `scatter_phi("v - r", cluster_labels)` where:
- `v` = `λ_p` = density at which point p falls out of cluster i
- `r` = `λ_birth_i` = density at which cluster i was born
- `keys` = cluster membership

`PHI_CENTERED_SUM = "v - r"` in scatter_jit.rs is exactly this.

---

## Domain 3: Finance — Centralized Return Moments

```
phi(v, r) = (v - r)^k  r = group return mean (e.g. sector, day, market)
result:     Σ(r_i - mean)^k per group → moments of demeaned returns
```

Financial risk analysis often wants "excess returns above a reference" — same pattern.
The reference r is benchmark return (sector mean, market mean, risk-free rate).
Per-group reference, same scattered-sum structure.

---

## Domain 4: Linear Algebra — Frobenius Norm Decomposition

```
phi(v, r) = (v - r)²   r = current centroid[cluster]
result:     SS_W = Σ Σ (x_ij - centroid_k)² = within-cluster sum of squares
```

Davies-Bouldin's `s_k` (intra-cluster dispersion) and Calinski-Harabasz's `SS_W` are both
instances of this. The centroids are per-group references, the phi is squared deviation.
Same `RefCenteredStatsEngine` call.

---

## What This Means for Implementation

`RefCenteredStatsEngine` should be parameterized over `phi`:

```rust
pub enum CenteredPhi {
    Linear,          // phi = "v - r"       → HDBSCAN stability
    Quadratic,       // phi = "(v - r) * (v - r)"  → variance, SS_W
    Cubic,           // phi = "(v - r)^3 / s^3"    → centered skewness
    Quartic,         // phi = "(v - r)^4 / s^4"    → centered kurtosis
    Custom(&'static str),  // arbitrary JIT expression
}
```

The JIT kernel changes, but the scatter infrastructure is identical. The `ScatterJit`
already generates these kernels via `PHI_CENTERED_SUM`, `PHI_CENTERED_SUM_SQ`.
`RefCenteredStatsEngine` just needs to route to the right `phi_expr`.

---

## The Shape Metrics Connection

For centered skewness/kurtosis:
```
m3 = Σ(x - mean)³ / n   (3rd central moment, population)
m4 = Σ(x - mean)⁴ / n   (4th central moment)
```

These can be computed with `PHI_CENTERED_SUM` to the 3rd and 4th power.
The `scatter_multi_phi` path in ScatterJit supports multiple phi expressions per pass:

```
scatter_multi_phi([x, x², x³, x⁴, 1], [Value, ValueSq, Custom("v^3"), Custom("v^4"), One])
```

One GPU pass → {Σx, Σx², Σx³, Σx⁴, n} → all shape statistics.
This is the direct path for Family 06 (Descriptive Statistics).

---

## The Unification

All "centered accumulations" are instances of the same primitive:

```
accumulate(data, ByKey{groups, K}, Expr::Custom(phi_of_v_minus_r), Op::Add, ref_values)
```

The `ref_values` parameter (one per group) is what distinguishes this from uncentered
scatter. The `RefCenteredStatsEngine` scaffold in `stats.rs` correctly identifies this
as a distinct engine — it's not just "variance" but "any centered scatter."

This is worth implementing generically rather than specialized for each domain.
The general engine handles: variance, HDBSCAN stability, financial excess moments,
SS_W for validation metrics, all in one parameterized implementation.

---

## Why This Matters for Build Order

Implement `RefCenteredStatsEngine` (stats.rs) generically with `CenteredPhi` parameter:

1. **Immediately enables**: variance, std, SS_W for Calinski-Harabasz
2. **Also enables**: HDBSCAN cluster stability (no additional code needed)
3. **Also enables**: skewness/kurtosis as centered 3rd/4th moments
4. **Also enables**: financial excess-return moments

One engine, four domains, one implementation. The pattern wasn't designed to span these
domains — it does anyway because the underlying math is the same operation.
