# F29 Sharing Surface: Graph Algorithms as Affine Iteration + Laplacian

Created: 2026-04-01T06:30:36-05:00
By: navigator

Prerequisites: F01 complete (distance/adjacency), F22 complete (EigenDecomp), F17 complete (Affine scan).

---

## Core Insight: Three Graph Algorithm Patterns

All graph algorithms in F29 map to one of three patterns:

| Pattern | Algorithms | Primitive |
|---------|-----------|----------|
| **Power iteration** (Kingdom B) | PageRank, HITS, label propagation | Affine scan |
| **Spectral** (Kingdom A) | Spectral clustering, graph embedding | EigenDecomp of Laplacian |
| **BFS/Dijkstra** (sequential) | shortest paths, BFS, Bellman-Ford | F28 Dijkstra |

---

## Pattern 1: Power Iteration = Affine Scan (Kingdom B)

### PageRank

```
r_new = d · A · r + (1-d)/N · 1
```

Where A = column-normalized adjacency (A_ij = 1/out_degree(j) if edge j→i).
d = damping factor (0.85 typically).

This is **exactly the Affine scan**: `r_t = A · r_{t-1} + b`
where A = d · normalized_adjacency, b = (1-d)/N · 1 (constant).

The matrix-vector product `A · r` = sparse matrix-vector multiply (SpMV).
SpMV is the fundamental graph operation. For dense adjacency: F22's DotProductOp. For sparse: new primitive.

**Phase 1**: dense adjacency (N < 10K). Matrix-vector multiply = one row of GramMatrix computation.
**Phase 2**: sparse adjacency (N > 10K). Needs sparse CSR format + SpMV kernel.

### Label Propagation

```
Y_new = D^{-1} A Y + (1-α) Y_0
```

Same Affine scan structure. Y = label matrix (N × K classes).
D = degree matrix. D^{-1}A = row-normalized adjacency.

This is the same iteration as PageRank, but with multi-column Y.

**Convergence**: geometric series. For `||A|| < 1` (which normalized adjacency guarantees),
converges in O(log(1/ε)) iterations.

---

## Pattern 2: Spectral Algorithms = Laplacian + EigenDecomp

### Graph Laplacian

```
L = D - A    (unnormalized)
L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}    (symmetric normalized)
L_rw = D^{-1} L = I - D^{-1} A    (random walk normalized)
```

Where D = degree matrix (diagonal), A = adjacency matrix.

**Computing L from adjacency**: O(N²) element-wise — trivial after A is built.

### Spectral Clustering

```
1. Build adjacency A from RBF kernel of DistancePairs: A[i,j] = exp(-||x_i-x_j||²/σ²)
2. Compute L_sym = D^{-1/2}(D-A)D^{-1/2}
3. EigenDecomp(L_sym) → k smallest eigenvectors
4. K-means on eigenvector rows → cluster labels
```

**Tambear decomposition**:
1. RBF kernel: `exp(-γ · D²[i,j])` — free from DistancePairs (element-wise)
2. Degree matrix: `D[i,i] = Σ_j A[i,j]` — scatter_phi("sum", ByRow) = F06 grouped sum
3. D^{-1/2}: element-wise inverse square root
4. L_sym = I - D^{-1/2} A D^{-1/2}: matrix multiply, F22's DotProductOp
5. EigenDecomp(L_sym): F22 reuse
6. K-means on eigenvectors: F20 K-means reuse

**Total new code**: ~50 lines (adjacency construction, degree matrix, normalize). Everything else is F01+F22+F20.

This confirms the naturalist's prediction: **spectral clustering = same EigenDecomp pipeline as PCA**, applied to the graph Laplacian instead of covariance.

### Graph Embedding (Node2Vec, DeepWalk)

Random walk on graph → skip-gram word2vec → embedding.
Word2Vec = log-bilinear model = dot products of embeddings.
Training = F05 gradient descent on the skip-gram loss.
Negative sampling = DistancePairs-based.

**Phase 3**: requires random walk infrastructure. Interesting fintek application (market microstructure graphs).

---

## Pattern 3: Shortest Paths (Dijkstra Reuse)

F28 already designed Dijkstra for Isomap. F29 reuses it:
- Single-source shortest path: Dijkstra from one node
- All-pairs: N × Dijkstra (parallelizable across source nodes)
- Betweenness centrality: uses all-pairs shortest paths + counting

**F29 doesn't implement Dijkstra — it imports from F28.**

### Minimum Spanning Tree (MST)

MST from distance matrix: Prim's algorithm or Kruskal's on sorted edges.
Kruskal's = Union-Find on sorted edges — same union-find from F27 (TDA)!

**MST = F27 union-find on sorted DistancePairs.** F27 computes MST as a side effect of
H₀ persistent homology (birth times = MST edge weights). Explicitly expose it.

---

## New Infrastructure F29 Needs

**Sparse matrix representation**: CSR (Compressed Sparse Row) for large graphs.
- SpMV (sparse matrix-vector multiply): needed for PageRank, label propagation on large graphs
- SpGEMM (sparse matrix multiply): for some graph algorithms
- For N < 10K: dense adjacency (F22's DotProductOp sufficient)

This is the "F29 is genuinely new" capability — sparse operations that don't exist yet.
**Phase 1** avoids sparse entirely (N < 10K, use dense adjacency).
**Phase 2** adds sparse CSR + SpMV.

---

## MSR Types F29 Produces

```rust
pub struct GraphResult {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub directed: bool,

    pub pagerank: Option<Vec<f64>>,         // shape (N,)
    pub clustering_coeff: Option<Vec<f64>>, // local clustering per node
    pub betweenness: Option<Vec<f64>>,      // node betweenness centrality
    pub degrees: Vec<usize>,                // out-degree per node

    pub components: Option<Vec<usize>>,     // component ID per node (from F27 union-find)
    pub mst_edges: Option<Vec<(usize, usize, f64)>>,  // MST edge list

    pub spectral_embedding: Option<Arc<Vec<f64>>>,    // from F22 EigenDecomp of Laplacian
    pub community_labels: Option<Vec<usize>>,          // from spectral clustering
}
```

---

## Build Order

**Phase 1 (dense adjacency, N < 10K)**:
1. Adjacency from DistancePairs: threshold or RBF kernel (~10 lines)
2. Degree matrix + normalization (~10 lines)
3. PageRank: Affine iteration (reuse F17 AffineState structure) (~30 lines)
4. Graph Laplacian: D - A with normalization (~15 lines)
5. Spectral clustering: EigenDecomp(L) + K-means (F22 + F20) (~20 lines wiring)
6. MST: expose F27's union-find result as MST edges (~10 lines)
7. Tests: `networkx` Python for PageRank, `scipy.sparse.csgraph` for Laplacian

**Phase 2 (sparse, N > 10K)**:
1. CSR sparse matrix format
2. SpMV kernel (new GPU primitive — this is operator #9 candidate... but actually not a new operator,
   it's TiledAdd with sparse indexing. Investigate: is sparse indexing a new atom or an existing atom with different gather pattern?)

---

## Structural Rhyme

**PageRank : EWM :: graph adjacency : time series**
- EWM (F17): `r_t = α · r_{t-1} + (1-α) · x_t` — Affine scan on time axis
- PageRank: `r_new = d · A · r + (1-d)/N · 1` — Affine scan on graph nodes
- Both: damped random walk that converges to a stationary distribution
- Time series decay = graph diffusion, different geometry, same algebra

**Spectral clustering : K-means :: soft partitioning : hard partitioning**
- K-means: assign points to nearest centroid (ArgMin on Euclidean distances)
- Spectral clustering: project to eigenvector space, then K-means
- The eigenvector projection relaxes the combinatorial clustering problem to a continuous one
- Same K-means final step — they share F20's K-means infrastructure

---

## The Lab Notebook Claim

> Graph algorithms (F29) use three patterns: power iteration (PageRank = Affine scan, Kingdom B), spectral analysis (spectral clustering = EigenDecomp of Laplacian, reusing F22), and shortest paths (Dijkstra, reusing F28). MST = F27 union-find side effect. F29 Phase 1 adds ~80 lines of wiring for dense graphs (N < 10K). The genuine new capability is sparse matrix-vector multiply for large graphs — needed in Phase 2 and worth investigating whether it fits within the 8-operator model or constitutes a new primitive.
