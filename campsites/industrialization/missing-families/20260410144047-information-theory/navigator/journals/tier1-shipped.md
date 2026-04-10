# Information Theory — Tier 1 Primitives Shipped

## Status: DONE (Tier 1)

12 new primitives added to `crates/tambear/src/information_theory.rs` and exported in `lib.rs`.
74 information theory tests green.

## What shipped

### f-divergence family
- `hellinger_distance_sq`, `hellinger_distance` — √(1 - BC(p,q)); range [0,1]
- `total_variation_distance` — ½ Σ|p-q|; range [0,1]
- `chi_squared_divergence` — Σ (p-q)²/q
- `renyi_divergence(p, q, alpha)` — special cases α=0 (support ratio), α=1 (KL), α=∞ (max ratio)
- `bhattacharyya_coefficient`, `bhattacharyya_distance` — -ln(BC); range [0,∞)
- `f_divergence(p, q, f)` — general framework via closure; subsumes all the above

### Joint entropy and PMI
- `joint_entropy` — H(X,Y) from joint probability matrix
- `pointwise_mutual_information(positive: bool)` — PMI or PPMI

### Sample-based divergences
- `wasserstein_1d` — O(n log n) via sorted CDF merge; handles unequal sizes
- `mmd_rbf` — Maximum Mean Discrepancy with Gaussian RBF kernel; median bandwidth heuristic
- `energy_distance` — V-statistic form; clamped to 0 for finite-sample bias

## Tier 2 remaining (not yet implemented)
- KSG mutual information estimator (k-NN based, for continuous data)
- NSB entropy estimator (Bayesian, for sparse discrete data)
- Chao-Shen entropy (small-sample corrected)
- Conditional MI: I(X;Y|Z)
- Directed information: Massey's I(X→Y)
- Blahut-Arimoto: channel capacity via alternating optimization

## Commit
`0e9d42b`
