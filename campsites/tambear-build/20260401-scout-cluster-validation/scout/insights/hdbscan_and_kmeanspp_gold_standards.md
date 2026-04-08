# HDBSCAN and KMeans++ — Gold Standard Scout Report

---

## HDBSCAN

**Gold standard**: Python `hdbscan` library (McInnes et al.) and `sklearn.cluster.HDBSCAN` (v1.3+)
**R**: `dbscan::hdbscan()` from mhahsler/dbscan
**Paper**: Campello, Moulavi, Sander. "Density-Based Clustering Based on Hierarchical
Density Estimates." PAKDD 2013.

### The core insight

HDBSCAN = DBSCAN run at ALL epsilon values simultaneously, then pick the best cut.
It's a hierarchy of DBSCAN solutions, with a principled extraction that selects which
"slices" of the hierarchy are stable enough to be real clusters.

DBSCAN asks: "at this epsilon, are there clusters?" HDBSCAN answers: "find the epsilon
ranges where each cluster is most stable."

### Algorithm Steps (exact)

**Step 1: Core distances**
```
core_dist_m(x) = distance to the m-th nearest neighbor of x
```
Parameter `m` = `min_cluster_size` (or separate `min_samples`).
→ Requires KNN extraction from the distance matrix.
→ From the n×n distance matrix already in TamSession: sort each row, take the m-th value.
→ This is `reduce(kth_order_stat)` over each row of the distance matrix.

**Step 2: Mutual Reachability Distance**
```
mrd(a, b) = max(core_dist(a), core_dist(b), dist(a, b))
```
→ Augments the distance matrix: `mrd[i,j] = max(core_dist[i], core_dist[j], dist[i,j])`
→ Effect: distances in low-density regions are inflated to reduce false connectivity.
→ Pure elementwise operation on the existing n×n distance matrix.
→ This is a `gather(core_dists) + elementwise_max(dist_matrix)` — no new GPU kernel.

**Step 3: Minimum Spanning Tree**
```
Build MST on the complete graph with edge weights = mrd(a, b)
```
→ Standard Prim's or Kruskal's algorithm on the n×n MRD matrix.
→ Prim's: O(n²) for dense graph — this is OPTIMAL for the dense case (n ≤ 5000).
→ CPU implementation is straightforward since the MRD matrix is already materialized.

**Step 4: Cluster Hierarchy (single-linkage on MRD)**
```
Sort MST edges by weight (ascending).
Process edges one by one: merging components they connect.
Record: at which lambda = 1/mrd weight did these two components merge?
```
→ This IS single-linkage hierarchical clustering on the MRD distances.
→ CPU: O(n²) sort + O(n·α(n)) union-find — same primitives as DBSCAN.

**Step 5: Condense the tree**
```
For each split in the hierarchy:
  If both resulting components have ≥ min_cluster_size points → real split → keep both
  If one component has < min_cluster_size points → it becomes noise at this lambda
```
→ Transforms the full hierarchy into a "condensed tree" with only meaningful splits.

**Step 6: Extract stable clusters (excess of mass)**
```
For each node in the condensed tree:
  stability(node) = Σ_{p in cluster} (lambda_p - lambda_birth)
  where lambda_birth = density at which this cluster was born
        lambda_p     = density at which point p falls out of cluster (or min of children)
```
→ Bottom-up: if children's total stability > parent's stability → keep children.
→ Otherwise → merge children back into parent (parent is more stable).
→ Final selection: a set of non-overlapping nodes with maximum total stability.

### Tambear decomposition of HDBSCAN

```
INPUTS: dist: Arc<DistanceMatrix> from TamSession (SHARED with DBSCAN!)
        min_cluster_size: usize
        min_samples: usize (defaults to min_cluster_size)

Step 1: Core distances
  → For each row i of dist.data: find m-th smallest value
  → reduce(ArgKth) over each row — CPU O(n²) given materialized matrix
  → Result: core_dists: Vec<f64> of length n

Step 2: MRD matrix
  → mrd[i,j] = max(core_dists[i], core_dists[j], dist.data[i*n + j])
  → Pure elementwise CPU: O(n²) — or GPU kernel for large n
  → Result: mrd_matrix: Vec<f64> of length n²

Step 3: Prim's MST on MRD
  → CPU O(n²): for dense graph this is optimal
  → Result: mst_edges: Vec<(usize, usize, f64)> of length n-1, sorted by weight

Step 4+5: Hierarchy extraction + tree condensation
  → Process MST edges in weight order, union-find on components
  → Track births/deaths in condensed tree
  → CPU: O(n log n)

Step 6: Stability-based cluster extraction
  → Bottom-up traversal of condensed tree
  → CPU: O(n)

Step 7: Label assignment
  → Core cluster members → cluster label
  → Non-core points within epsilon of a cluster → cluster label (or noise)
```

**The sharing insight**: Steps 1-3 use the SAME `Arc<DistanceMatrix>` that DBSCAN cached.
For users running both DBSCAN and HDBSCAN on the same data (e.g., comparing results):
O(n²d) GPU work runs ONCE. Both algorithms share that materialized matrix.

HDBSCAN's incremental work over DBSCAN:
1. Core distance extraction (row-wise argkth): O(n²) CPU
2. MRD computation (elementwise max): O(n²) CPU
3. MST construction: O(n²) CPU
4. Hierarchy + condensation + stability: O(n log n) CPU

All CPU steps given the shared distance matrix. This is efficient.

### Key parameters and their effect

```
min_cluster_size: minimum points to form a cluster (vs noise)
  → Controls condensation step — larger = fewer, larger clusters
  → This is HDBSCAN's version of DBSCAN's min_samples

min_samples: controls core distances (= m in core_dist_m)
  → Larger min_samples = more conservative, noisier data needed for cores
  → Default = min_cluster_size
  → When set separately: allows "soft" vs "hard" core identification

cluster_selection_method: 'eom' (excess of mass) or 'leaf'
  → 'eom': maximize stability sum — finds arbitrary-scale clusters
  → 'leaf': all leaves of condensed tree — finds most granular clusters
```

### Differences from DBSCAN

| Property | DBSCAN | HDBSCAN |
|----------|--------|---------|
| Parameters | epsilon + min_samples | min_cluster_size (+ optional min_samples) |
| Variable density | Struggles | Native support |
| Noise handling | Points not in core neighborhood | Points with low persistence |
| Reproducibility | Deterministic | Deterministic |
| Complexity | O(n²d) GPU + O(n²) CPU | Same O(n²d) GPU + O(n²) CPU |
| Cluster hierarchy | Flat solution | Full hierarchy available |

### Edge cases and numerical issues

- **Duplicate points**: `dist[i,j] = 0` for duplicates → `mrd[i,j] = max(core_dist(i), core_dist(j))`.
  Handle: duplicates become core if `min_samples ≥ 1` (they are in each other's neighborhood).
- **Single-point clusters**: impossible with `min_cluster_size ≥ 2`
- **All points identical**: all core distances = 0, MRD matrix = 0, all one cluster
- **n < 2·min_cluster_size**: likely all noise

---

## KMeans++

**Gold standard**: `sklearn.cluster.KMeans(init='k-means++')`
**Paper**: Arthur & Vassilvitskii, "k-means++: the advantages of careful seeding." SODA 2007.

### The Problem with Random Init

Standard KMeans (Lloyd) uses random initialization — pick k random points as initial
centroids. This leads to:
- Convergence to local optima (often bad)
- Non-deterministic results
- Slow convergence (many iterations)

KMeans++ provides probabilistic seeding with an O(log k) approximation guarantee to
the optimal K-means cost.

### The Algorithm

```
1. Choose c₁ uniformly at random from data points

2. For i = 2..k:
   a. Compute D²(x) = min distance² from x to nearest already-chosen centroid
      D²(x) = min_{j<i} dist(x, c_j)²
   b. Choose cᵢ with probability proportional to D²(x):
      P(choose x) = D²(x) / Σ_x D²(x)

3. Proceed with standard Lloyd's iterations
```

The D² weighting ensures that new centroids are far from existing ones, in a
density-aware way. Points far from any centroid AND in high-density regions are
favored over isolated outliers.

### sklearn's exact implementation

```python
# init_centroids from _kmeans_plusplus:
centers[0] = X[random_state.choice(n_samples)]

for c in range(1, n_clusters):
    # Distance from each point to nearest existing center
    closest_dist_sq = euclidean_distances(X, centers[:c], Y_norm_squared=...).min(axis=1)
    # D² probability distribution
    probs = closest_dist_sq / closest_dist_sq.sum()
    # Sample n_local_trials candidates
    candidates = random_state.choice(n_samples, size=n_local_trials, p=probs)
    # Pick the candidate that minimizes total inertia
    best_candidate = argmin(sum(min(D²(x), dist(x, candidate)) for x in X))
    centers[c] = X[best_candidate]
```

sklearn uses `n_local_trials = 2 + floor(log(k))` candidates per step (greedy improvement).

### Tambear decomposition

```
Step 1: Choose c₁ randomly
  → simple scalar random index

Step 2: For i = 2..k:
  a. Compute D²(x) for all x:
     → tiled_accumulate(data, centroids_so_far, distance_op) → n × i matrix
     → reduce(argmin_rowwise) → closest centroid per point
     → gather(distances, argmin_indices) → D²(x) for each x
  b. Normalize to probability: D² / sum(D²)
  c. Weighted random sample: sample from D² distribution
  d. (Optional) n_local_trials greedy: test best of L candidates

Alternative for b-c: weighted reservoir sampling on GPU
```

**GPU efficiency of KMeans++**:
- k initialization steps, each requiring distance computation to i < k existing centroids
- Total work: O(knd) distance computation (vs O(Tnd) for Lloyd's T iterations)
- For typical k=5..50 and T=100..300 iterations: initialization is < 1% of total cost
- BUT: KMeans++ reduces T significantly — fewer iterations to convergence

### Tambear-native improvement: density peak seeding

Alternative to KMeans++ that uses DBSCAN's existing density information:

```
If DBSCAN was run with the same data:
  → core points with highest density (largest neighborhood count) are natural centroid seeds
  → Pick k core points by density: avoid duplicates by requiring they're > epsilon apart
  → This is a heuristic but often better than KMeans++ for well-separated clusters
```

This is a tambear-native seeding strategy that emerges from the sharing infrastructure —
"if DBSCAN already found cluster centers, use them." Not in sklearn. Novel.

### The convergence criterion choice

Current kmeans.rs uses **label stability**: stop when no point changes cluster.
This is correct and avoids f32 atomicAdd ordering noise.

sklearn uses **centroid shift**: stop when `||centroids_new - centroids_old||² < tol`.
This requires centroid comparison, which is sensitive to f32 accumulation order.

Gold standard behavior comparison:
```python
# sklearn default: tol=1e-4 relative, max_iter=300
# sklearn label-stable: n_init=10 runs, pick best inertia
```

Label stability (current tambear) is more robust for f32 than centroid-shift tolerance.
Multiple restarts (`n_init > 1`) are NOT currently implemented — relevant for production.

### Multi-restart pattern

sklearn uses `n_init=10` KMeans runs, picks the one with lowest inertia.
For tambear: run k independent KMeans instances concurrently (different CUDA streams),
compare final SS_W, keep best. This is embarrassingly parallel.
Inertia = SS_W = `Σ_i D²(x_i, assigned_centroid)` — same as Calinski-Harabasz SS_W.

---

## Build priority note

For HDBSCAN: **no new primitives needed** given the DBSCAN distance matrix is already
materialized. The incremental implementation cost is:
1. Core distance extraction (row-wise k-th order stat): ~30 lines CPU
2. MRD elementwise max: ~10 lines
3. Prim's MST: ~50 lines CPU
4. Hierarchy + condensation: ~80 lines CPU
5. Stability extraction: ~60 lines CPU
→ ~230 lines of CPU code on top of the existing GPU distance matrix.

For KMeans++: modify `kmeans.rs::KMeansEngine::fit()` to accept `init='random'|'kmeans++'`.
The seeding loop is ~40 lines additional. Core of it is distance computation, which
reuses the existing ASSIGN_KERNEL (distances from points to centroids).
