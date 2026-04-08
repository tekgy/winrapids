# DBSCAN via Tambear Primitives: The Accumulate Unification Made Concrete

## What was built

`tambear::ClusteringEngine` — density-based clustering (DBSCAN) composed from:
1. `TiledEngine::run(&DistanceOp, data, data_T, n, n, d)` → n×n pairwise distance matrix (GPU)
2. CPU density estimation: `density[i] = |{j : D[i,j] ≤ epsilon}|` (includes self)
3. CPU core identification: `is_core[i] = density[i] ≥ min_samples`
4. CPU union-find connected components over core-core edges
5. CPU border assignment: non-core within epsilon of a core → that core's cluster

## The design choice: GPU distance + CPU graph operations

The team lead proposed two approaches:
- **Scatter-iterate**: each core scatters label to neighbors via atomicMin, iterate until convergence
- **GPU distance + CPU union-find** (chosen)

The GPU distance matrix approach wins because:
- **O(n²) is the bottleneck anyway** — both approaches are O(n²). The GPU reduces the constant.
- **Union-find is O(n · α(n))** — near-linear on the already-materialized adjacency. Faster than scatter iterations which need O(diameter) passes.
- **No new GPU primitives needed** — atomicMin on u32 would need a new kernel. Union-find is 5 lines of Rust.
- **Correct in one pass** — union-find gives exact connected components without convergence detection.

The scatter-iterate approach would be better for VERY large n where the full n×n matrix doesn't fit in GPU memory. That's a future `approximate_discover_clusters` that uses ANN + sparse adjacency. The current implementation is exact and simple.

## The transpose insight

`TiledEngine::run(op, A, B, m, n, k)` computes `C[i,j] = op(A[i,:], B[:,j])` — A is m×k, B is k×n.

For self-distance: A = data (n×d), B must be data^T (d×n):
- A[i, k] = data[i*d + k]  — point i, dimension k
- B[k, j] = data_T[k*n + j] = data[j*d + k]  — dimension k of point j

If you pass data as both A and B, the kernel interprets B as d×n when data is stored n×d — wrong access pattern. The fix is one transpose: `data_T[k*n+j] = data[j*d+k]`.

This generalizes: any self-similarity computation via TiledEngine needs B = A^T.

## Distance as a parameter

`discover_clusters(data, n, d, epsilon_threshold, min_samples, distance_op: &dyn TiledOp)`

- `&DistanceOp` → squared L2 distance. Pass `epsilon_radius²` as threshold.
- `&CovarianceOp` → correlation-based distance. Pass appropriate threshold.
- Custom `TiledOp` → any distance metric implementable as `C[i,j] = accumulate(A[i,:], B[:,j])`

The `dbscan(data, n, d, epsilon_radius, min_samples)` convenience method handles the L2² squaring automatically.

## The DBSCAN convention: self-counting

DBSCAN counts the point itself in the neighborhood: `density[i] = |{j : D[i,j] ≤ epsilon}|` where `D[i,i] = 0 ≤ epsilon` always.

So for min_samples=2: a pair of nearby points makes both points cores (each has 2 neighbors including self). This is the standard DBSCAN definition (Wikipedia, sklearn).

The initial implementation used `i != j` exclusion, giving density 1 short — corrected to include self.

## What this enables next

The immediate composition: **KMeans via discover_clusters** — use the distance matrix already computed for clustering to also initialize centroids (density peaks are natural centroid seeds). The `kmeans.rs` module computes distances from scratch; wiring it to the `ClusteringEngine` distance matrix avoids recomputation.

The theoretical next step from the team lead's architecture: **MinLabelOp scan**. The current union-find is CPU O(n²). A GPU parallel scan with MinLabelOp would make label propagation O(n log n) on GPU — useful when n is large enough that O(n²) scan iterations dominate. But for n ≤ 5000, union-find is faster.

The approximate clustering path: when n > ~5000, the full distance matrix is too large. The path is:
1. `approximate_neighbors(data, k)` → sparse adjacency via HNSW/IVF (future)
2. Apply union-find on sparse adjacency (same code path, just fewer edges)
