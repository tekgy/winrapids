# Distributional Distance Suite Taxonomy

This suite provides primitives for measuring the difference between two probability distributions.

## Primitives Catalog

| Primitive | Kingdom | Pattern | Parameters | Notes |
| :--- | :---: | :--- | :--- | :--- |
| `kl_divergence` | A | accumulate + gather | `epsilon` (f64) | Kullback-Leibler divergence. |
| `js_divergence` | A | accumulate + gather | `epsilon` (f64) | Jensen-Shannon divergence. |
| `hellinger_distance` | A | accumulate + gather | N/A | Hellinger distance. |
| `total_variation_distance` | A | accumulate + gather | N/A | Total Variation Distance. |
| `kolmogorov_smirnov` | A | sort $\rightarrow$ gather | N/A | KS distance ($\sup \|F_P - F_Q\|$). |
| `wasserstein_1` | A | sort $\rightarrow$ accumulate | N/A | Earth Mover's Distance (1-Wasserstein). |
| `wasserstein_p` | C | Iterative (Sinkhorn) | `p` (f64), `lambda` (f64) | General $L_p$ Wasserstein distance. |
| `maximum_mean_discrepancy` | A | accumulate + gather | `sigma` (f64) | MMD using kernel embeddings. |

## Global Parameters

- `epsilon`: Smoothing constant used to avoid $\log(0)$ or division by zero in divergences.- `sigma`: Kernel bandwidth for MMD.
- `lambda`: Entropy regularization coefficient for Sinkhorn iteration.
- `p`: The order of the Wasserstein distance.

## Kingdom Justifications

- **Kingdom A (Tensors/Parallel):** Most of these are point-wise operations or reductions that can be decomposed into `accumulate` (summing/aggregating) and `gather` (addressing data) patterns.
- **Kingdom C (Iterative Fixed-Point):** `wasserstein_p` (via Sinkhorn) requires iterative updates to a transport plan until convergence, fitting the Iterative Fixed-Point pattern.
