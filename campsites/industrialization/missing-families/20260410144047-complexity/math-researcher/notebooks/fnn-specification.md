# False Nearest Neighbors — Mathematical Specification

## Reference

Kennel, Brown & Abarbanel (1992), "Determining embedding dimension for phase-space reconstruction using a geometrical construction and the false nearest neighbors method", Physical Review A, 45(6), 3403-3411.

## The Problem FNN Solves

When reconstructing phase space via time-delay embedding (Takens' theorem), you need to choose an embedding dimension d. Too small: the attractor self-intersects (false neighbors appear due to projection). Too large: noise amplification and computational waste.

FNN answers: "What fraction of nearest neighbors in dimension d are 'false' — i.e., they are only neighbors because the embedding is too low-dimensional?"

## Mathematical Definition

Given time series {x₁, ..., xₙ} and delay τ:

1. Form delay vectors in dimension d:
   **y**ᵢ^(d) = (xᵢ, xᵢ₊τ, xᵢ₊₂τ, ..., xᵢ₊(d-1)τ)

2. For each vector **y**ᵢ^(d), find its nearest neighbor **y**ⱼ^(d) (j ≠ i):
   R_d(i) = ||**y**ᵢ^(d) - **y**ⱼ^(d)||₂

3. Extend to dimension d+1 and compute the new distance:
   R_{d+1}(i) = ||**y**ᵢ^(d+1) - **y**ⱼ^(d+1)||₂

4. A neighbor is "false" if the distance ratio exceeds a threshold:
   FNN criterion: √(R_{d+1}²(i) - R_d²(i)) / R_d(i) > R_tol

   Equivalently: R_{d+1}² / R_d² > 1 + R_tol²

   In practice, R_tol = 10-15 is standard. Kennel et al. use R_tol = 15.

5. FNN fraction at dimension d:
   FNN(d) = (number of false neighbors) / (number of valid neighbors tested)

6. Optimal dimension: smallest d where FNN(d) < threshold (typically 0.01 or 0.10).

## Current Implementation

The current implementation in `family15_manifold_topology.rs:445` (`fnn_frac`):
- Uses Euclidean distance (L2): CORRECT per Kennel et al.
- R_tol = 15.0: CORRECT (matches Kennel et al.)
- Caps at n_cap = 200 points: APPROXIMATION (for speed in fintek per-bin context)
- Checks extra dimension: correctly computes R_{d+1}² = R_d² + (xᵢ₊dτ - xⱼ₊dτ)²
- Ratio test: new_dist2 / best_dist2 > rtol² = 225: CORRECT (Kennel's criterion)

## What Should Change When Promoted to Primitive

### Parameters
| Parameter | Type | Range | Default | Meaning |
|---|---|---|---|---|
| `data` | `&[f64]` | any finite | required | Time series |
| `d` | `usize` | ≥ 1 | required | Embedding dimension to test |
| `tau` | `usize` | ≥ 1 | 1 | Delay |
| `r_tol` | `f64` | > 0 | 15.0 | Distance ratio threshold |
| `max_points` | `Option<usize>` | > 0 | None (use all) | Cap for speed |

### Return Value
`f64` — fraction of false nearest neighbors, ∈ [0, 1].

### Assumptions
1. Data is finite (no NaN/Inf)
2. tau × d < data length
3. Data is NOT constant (constant data → all distances zero)
4. Stationarity (at least approximately — otherwise the embedding doesn't represent a single attractor)

### Failure Modes
| Input | Behavior | Reason |
|---|---|---|
| All constant | 1.0 or NaN | All distances zero |
| Pure noise | ~0.0 for any d | No false neighbors in noise (no structure to unfold) |
| Very short series | Unreliable | Insufficient neighbors |
| Very large tau | Gaps exceed correlation | Embedding loses temporal coherence |

### Second Criterion (Kennel's Criterion 2)

Kennel et al. also propose a second criterion based on the ratio R_{d+1}/R_A where R_A is a measure of attractor size (e.g., standard deviation of the data). This catches cases where the first criterion misses (e.g., when R_d is accidentally small).

Criterion 2: R_{d+1}(i) / R_A > A_tol (typically A_tol = 2)

The current implementation does NOT use criterion 2. For a complete primitive, it should be an optional parameter.

### Computational Complexity
- Current: O(n_cap² × d) — brute force NN search
- With kd-tree: O(n × d × log n) — but kd-tree doesn't help much for d > 5
- For production: approximate NN (random projection trees, locality-sensitive hashing)

### Sharing with TamSession

FNN shares with:
- `delay_embed(data, d, tau)` — the embedding matrix (Phylum 10)
- `pairwise_dists(embedded)` — if computing full distance matrix (Phylum 7)
- `knn_search(embedded, k=1)` — if using kNN infrastructure

The FNN result itself is NOT an intermediate for other methods — it's a diagnostic. But the embedding and distances it uses ARE shared with SampEn, correlation_dimension, Lyapunov, etc.

## The Full FNN Workflow (for Layer 1 auto-detection)

```
1. Estimate optimal tau via AMI first minimum (already in family15)
2. For d = 1, 2, 3, ..., max_d:
   a. Compute FNN(d, tau)
   b. If FNN(d) < threshold (0.01): set d_opt = d, break
3. Return (d_opt, tau_opt) as recommended embedding parameters
```

This workflow should be a Layer 1 method (`auto_embedding(data)`) that composes:
- `ami_at_lag` (already exists as private in family15)
- `false_nearest_neighbors` (to be promoted)
- `delay_embed` (already public in time_series)

## Connection to Other Complexity Measures

Every embedding-based method should, in principle, validate its embedding parameters:
- `sample_entropy(data, m, r)` — m IS the embedding dimension
- `correlation_dimension(data, m, tau)` — m and tau are embedding params
- `largest_lyapunov(data, m, tau, dt)` — same
- `rqa(data, m, tau, epsilon, lmin)` — same

Layer 1 could automatically determine (m, tau) via FNN + AMI before running any of these.
