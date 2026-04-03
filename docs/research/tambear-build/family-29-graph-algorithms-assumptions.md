# Family 29: Graph Algorithms — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: Mixed — A (PageRank/centrality = matrix-vector accumulate), B (BFS/SSSP = frontier scan), C (community detection = iterative)

---

## Core Insight: Graphs are Sparse Matrices

Every graph algorithm operates on the adjacency matrix A or Laplacian L = D - A. The GPU primitive is sparse matrix-vector multiply (SpMV). PageRank, eigenvector centrality, spectral embedding — all are iterative SpMV.

---

## 1. Graph Representation

### Adjacency Matrix A
- A_{ij} = w_{ij} if edge (i,j) exists, 0 otherwise
- Undirected: A symmetric. Directed: A may be asymmetric.

### Compressed Sparse Row (CSR)
- `row_ptr[i]`: start of row i's neighbors in `col_idx`
- `col_idx`: column indices of nonzero entries
- `values`: edge weights

### GPU: CSR is standard for SpMV. Use cuSPARSE-style kernels or custom.

### Laplacian
```
L = D - A    where D = diag(Σ_j A_{ij})    (degree matrix)
```
Normalized: L_norm = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}

---

## 2. PageRank

### Formula (power iteration)
```
r_{t+1} = α · A_norm' · r_t + (1-α) · (1/n) · 1
```
where A_norm = column-normalized adjacency (transition matrix), α = 0.85 (damping factor).

### Convergence
Rate: O(α^t). For α = 0.85: ~50 iterations to convergence. Each iteration = one SpMV.

### Implementation: Iterative SpMV. Initialize r = 1/n. Converge when ‖r_{t+1} - r_t‖₁ < ε.

### Edge Cases
- Dangling nodes (no outgoing edges): distribute rank uniformly. Modify: r_t = α·A'r_t + (α·d'r_t + (1-α))/n where d = indicator of dangling nodes.
- Disconnected graph: PageRank is well-defined (teleportation connects everything)

### Kingdom: C (iterative power method). Each iteration: A (SpMV = parallel accumulate over edges).

---

## 3. HITS (Hyperlink-Induced Topic Search)

### Authority and Hub Scores
```
a_{t+1} = A' · h_t     (authority = incoming hub scores)
h_{t+1} = A · a_t      (hub = outgoing authority scores)
```
Normalize after each step. Converges to dominant eigenvector of A'A (authority) and AA' (hub).

### Implementation: Two SpMVs per iteration. Same as power iteration for singular vectors.

---

## 4. Shortest Paths

### 4a. Dijkstra (Single-Source, Non-Negative Weights)
```
For source s:
  dist[s] = 0, dist[v] = ∞ for v ≠ s
  While unvisited nodes remain:
    u = node with minimum dist (priority queue)
    For each neighbor v of u:
      if dist[u] + w(u,v) < dist[v]:
        dist[v] = dist[u] + w(u,v)
```
Complexity: O((V + E) log V) with binary heap.

### 4b. Bellman-Ford (Single-Source, Handles Negative Weights)
```
For i = 1 to V-1:
  For each edge (u, v, w):
    if dist[u] + w < dist[v]:
      dist[v] = dist[u] + w
```
Detect negative cycles: run one more iteration. If any update → negative cycle exists.

### GPU: Bellman-Ford is more parallelizable than Dijkstra (all edges processed per iteration).

### 4c. Floyd-Warshall (All-Pairs)
```
For k = 1 to V:
  For i = 1 to V:
    For j = 1 to V:
      dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
```
O(V³). GPU: parallelize over (i,j) pairs for each k.

### 4d. Johnson's Algorithm (All-Pairs, Sparse)
1. Add vertex s with zero-weight edges to all others
2. Bellman-Ford from s → potential function h(v)
3. Reweight: w'(u,v) = w(u,v) + h(u) - h(v) ≥ 0
4. Run Dijkstra from each vertex on reweighted graph
O(VE + V² log V). Better than Floyd-Warshall for sparse graphs.

### Kingdom: Dijkstra = C (sequential frontier). Bellman-Ford = B (iterative relaxation scan). Floyd-Warshall = C (k-dependent).

---

## 5. Connected Components

### BFS-based
```
While unmarked nodes exist:
  Pick unmarked node u, BFS from u → mark all reachable as same component
```

### Union-Find (Disjoint Set)
```
For each edge (u, v):
  Union(Find(u), Find(v))
```
With path compression + union by rank: O(E · α(V)) ≈ O(E).

### GPU: Label propagation — each vertex takes minimum label of its neighbors. Iterate until convergence. O(diameter) iterations, each = SpMV-like operation.

---

## 6. Community Detection

### 6a. Louvain Algorithm
```
Phase 1 (Modularity optimization):
  For each node i:
    Move i to community of neighbor j that maximizes ΔQ
    Repeat until no improvement
Phase 2 (Aggregation):
  Contract communities to single nodes, aggregate edges
  Repeat from Phase 1 on coarsened graph
```

### Modularity
```
Q = (1/2m) Σ_{ij} [A_{ij} - k_i·k_j/(2m)] · δ(c_i, c_j)
```
where m = total edges, k_i = degree, c_i = community of node i.

### ΔQ for moving node i to community C
```
ΔQ = [Σ_in + 2·k_{i,in}]/(2m) - [(Σ_tot + k_i)/(2m)]²
   - [Σ_in/(2m) - (Σ_tot/(2m))² - (k_i/(2m))²]
```
where Σ_in = internal edge weight of C, Σ_tot = total degree of C, k_{i,in} = edges from i to C.

### 6b. Leiden Algorithm (Traag et al. 2019)
Improvement over Louvain: guarantees connected communities. Adds refinement phase between Louvain phases.

### 6c. Label Propagation
Each node takes the most frequent label among neighbors. Fast (O(E) per iteration) but non-deterministic.

### Kingdom: C (iterative for Louvain/Leiden). Label propagation = B (iterative scan).

---

## 7. Centrality Measures

### 7a. Betweenness Centrality
```
C_B(v) = Σ_{s≠v≠t} σ_{st}(v) / σ_{st}
```
where σ_{st} = number of shortest paths from s to t, σ_{st}(v) = those passing through v.

### Brandes' Algorithm: O(VE) — BFS from each source, accumulate dependencies.
GPU: Parallel across sources.

### 7b. Closeness Centrality
```
C_C(v) = (n-1) / Σ_{u≠v} d(v, u)
```
Requires all-pairs shortest paths. For large graphs: sample approximation.

### 7c. Eigenvector Centrality
```
x = (1/λ₁) · A · x    (dominant eigenvector of A)
```
Power iteration. Same as PageRank without damping.

### 7d. Katz Centrality
```
x = (I - αA)⁻¹ · 1 = Σ_{k=0}^{∞} α^k · A^k · 1
```
α < 1/λ₁ for convergence. Accounts for paths of all lengths (discounted).

### Kingdom: All centrality = C (iterative SpMV or BFS from all sources).

---

## 8. Graph Laplacian & Spectral Methods

### Eigenvalues of L
- λ₁ = 0 always (eigenvector = 1)
- Number of zero eigenvalues = number of connected components
- λ₂ = algebraic connectivity (Fiedler value). λ₂ > 0 iff graph is connected.
- Fiedler vector (eigenvector for λ₂): used for spectral bisection

### Spectral Clustering (same as F20)
1. Compute normalized Laplacian L_norm = I - D^{-1/2}AD^{-1/2}
2. Find k smallest eigenvectors
3. Stack as columns of U (n × k)
4. Normalize rows of U
5. K-means on rows of U

### Spectral Embedding
Eigenvectors of L embed graph vertices in R^k preserving proximity structure. Same as Laplacian eigenmap (F22 dimensionality reduction).

---

## 9. Random Walks on Graphs

### Transition Matrix
```
P = D⁻¹A    (row-normalized adjacency)
```
P_{ij} = probability of walking from i to j.

### Hitting Time
```
h(i, j) = E[min{t : X_t = j | X_0 = i}]
```
Satisfies: h(i,j) = 1 + Σ_{k≠j} P_{ik}·h(k,j) — linear system.

### Commute Distance
```
κ(i,j) = h(i,j) + h(j,i) = 2m · (e_i - e_j)' L⁺ (e_i - e_j)
```
where L⁺ = pseudoinverse of Laplacian.

### Node2Vec (Random Walk Embedding)
Biased random walks (parameters p, q control BFS vs DFS tendency) → sequence of nodes → Word2Vec skip-gram → node embeddings.

---

## 10. Graph Kernels

### Weisfeiler-Lehman (WL) Kernel
1. Initialize: node labels = degree
2. For h iterations: new label = hash(old label, sorted neighbor labels)
3. Kernel = count matching labels between two graphs

### Random Walk Kernel
K(G₁, G₂) = Σ_k λ^k · Σ_{paths_of_length_k} (product of node/edge labels match)

### Shortest Path Kernel
K(G₁, G₂) = Σ_{(u,v)∈G₁} Σ_{(u',v')∈G₂} k_v(u,u')·k_v(v,v')·k_e(d(u,v), d(u',v'))

---

## Sharing Surface

### Reuse from Other Families
- **F01 (Distance)**: Graph from distance matrix (ε-neighborhood, k-NN graph)
- **F02 (Linear Algebra)**: SpMV, sparse Cholesky, eigendecomposition
- **F20 (Clustering)**: Spectral clustering = graph Laplacian + K-means
- **F22 (Dimensionality Reduction)**: Laplacian eigenmap = spectral embedding
- **F22 (SVD)**: Eigendecomposition of Laplacian/adjacency
- **F32 (Numerical)**: Linear system solvers for hitting times

### Consumers of F29
- **F20 (Clustering)**: Spectral clustering, community detection
- **F22 (Dimensionality Reduction)**: Graph-based embeddings (t-SNE neighborhood, UMAP graph)
- **F27 (TDA)**: Simplicial complex from graph
- **F28 (Manifold)**: k-NN graph for manifold learning

### Structural Rhymes
- **PageRank = stationary distribution of random walk**: same as Markov chain (F17)
- **Graph Laplacian eigenvalues = frequencies on the graph**: same as FFT frequencies (F03) on regular lattice
- **Modularity = deviation from random graph null model**: same as χ² (observed - expected, F07)
- **BFS = level-set expansion**: same as distance transform in image processing
- **Spectral bisection = sign of Fiedler vector**: same as thresholding first PCA component (F22)

---

## Implementation Priority

**Phase 1** — Core graph primitives (~200 lines):
1. CSR representation + SpMV kernel
2. PageRank (power iteration)
3. BFS + connected components
4. Dijkstra + Bellman-Ford (SSSP)
5. Graph Laplacian computation

**Phase 2** — Centrality + community (~200 lines):
6. Betweenness centrality (Brandes)
7. Closeness, eigenvector, Katz centrality
8. Louvain community detection
9. Leiden algorithm
10. Label propagation

**Phase 3** — Spectral + paths (~150 lines):
11. Spectral embedding (Laplacian eigenvectors)
12. HITS (authority/hub scores)
13. Floyd-Warshall (all-pairs shortest paths)
14. Random walks + hitting times

**Phase 4** — Advanced (~100 lines):
15. Graph kernels (WL, random walk, shortest path)
16. Node2Vec embeddings
17. Minimum spanning tree (Kruskal/Prim)
18. Maximum flow (Ford-Fulkerson / push-relabel)

---

## Composability Contract

```toml
[family_29]
name = "Graph Algorithms"
kingdom = "A (SpMV accumulate) + B (BFS frontier) + C (community iterative)"

[family_29.shared_primitives]
csr_spmv = "Compressed Sparse Row + SpMV kernel"
bfs = "Breadth-first search (frontier expansion)"
page_rank = "Power iteration on transition matrix"
laplacian = "L = D - A (and normalized variant)"
sssp = "Single-source shortest path (Dijkstra/Bellman-Ford)"

[family_29.reuses]
f01_distance = "Distance matrix → graph construction"
f02_linalg = "SpMV, sparse eigendecomposition"
f22_svd = "Laplacian eigenvectors for spectral methods"
f32_numerical = "Linear solvers for hitting times"

[family_29.provides]
pagerank = "Node importance scores"
centrality = "Betweenness, closeness, eigenvector, Katz"
communities = "Louvain/Leiden community labels"
shortest_paths = "SSSP and APSP distance matrices"
spectral_embedding = "Graph Laplacian eigenvectors"

[family_29.consumers]
f20_clustering = "Spectral clustering, graph-based clustering"
f22_reduction = "Graph-based dimensionality reduction"
f27_tda = "Simplicial complex from graph"
f28_manifold = "k-NN graph for manifold learning"

[family_29.session_intermediates]
csr_graph = "CSRGraph(data_id) — adjacency in CSR format"
pagerank_scores = "PageRank(graph_id) — node scores"
community_labels = "Communities(graph_id) — node → community"
shortest_paths = "SSSP(graph_id, source) — distances from source"
laplacian_eigvecs = "LaplacianEig(graph_id, k) — bottom-k eigenvectors"
```
