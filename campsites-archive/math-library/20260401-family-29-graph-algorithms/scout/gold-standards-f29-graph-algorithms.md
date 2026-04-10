# F29 Graph Algorithms — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 29 (Graph Algorithms).
Documents: which library implements which algorithm, exact API signatures, formulas,
validation targets, and tambear decomposition by kingdom.

F29 is cross-kingdom with a strong CPU bias. The graph algorithms that ARE GPU-native
(power iteration, matrix-vector multiply) reuse the accumulate + gather primitives
already established. The rest are sequential graph traversals that belong on CPU.

**Connection map**:
- F01 (DistancePairs) → graph construction (build adjacency from distance matrix)
- F20 (Clustering) → community detection (Louvain/Leiden = graph clustering)
- F22 (DimReduction) → spectral graph embedding (eigendecomposition of Laplacian)
- F29 graph Laplacian → spectral clustering (F20 reuses the same eigenmap path)

---

## Graph Representations: What Every Library Expects

Before the algorithm APIs, the input format question must be settled.

### scipy.sparse formats (canonical for large graphs)

```python
import scipy.sparse as sp

# CSR (Compressed Sparse Row) — fastest for row slicing, matrix-vector multiply
W = sp.csr_matrix((data, (row, col)), shape=(n, n))

# COO (coordinate format) — easiest to build incrementally
W = sp.coo_matrix((data, (row, col)), shape=(n, n))
W_csr = W.tocsr()   # convert for computation

# CSC (Compressed Sparse Column) — fastest for column slicing
W = sp.csc_matrix(...)

# From dense numpy array:
W = sp.csr_matrix(dense_W)

# From NetworkX graph:
W = nx.to_scipy_sparse_array(G, weight='weight', format='csr')
```

### NetworkX formats

```python
import networkx as nx

# Undirected weighted graph:
G = nx.Graph()
G.add_edge(0, 1, weight=2.5)
G.add_edges_from([(1, 2, {'weight': 1.0}), (0, 2, {'weight': 3.1})])

# From adjacency matrix:
G = nx.from_numpy_array(W_dense)           # dense
G = nx.from_scipy_sparse_array(W_sparse)   # sparse

# Directed graph:
G = nx.DiGraph()
```

### igraph formats (much faster than NetworkX for large graphs)

```python
import igraph as ig

# From edge list:
g = ig.Graph(n=100, edges=[(0,1),(1,2),(2,3)], directed=False)
g.es['weight'] = [2.5, 1.0, 3.1]   # edge weights

# From adjacency matrix:
g = ig.Graph.Weighted_Adjacency(W_dense.tolist(), mode='undirected')

# From NetworkX:
g = ig.Graph.from_networkx(G)
```

**Key trap: NetworkX is pure Python.** For graphs with n > ~10,000 nodes or e > ~100,000 edges,
NetworkX is prohibitively slow. Use igraph (C-backed) or cuGraph (GPU) for performance.

---

## 1. Graph Laplacian and Spectral Graph Theory

### 1.1 Definitions

Given an undirected weighted graph with adjacency matrix W and degree matrix D:

```
D[i,i]  = Σⱼ W[i,j]          (row sum of adjacency)

Unnormalized Laplacian:
  L = D - W

Symmetric normalized Laplacian:
  L_sym = D^{-1/2} · L · D^{-1/2}
        = I - D^{-1/2} · W · D^{-1/2}
  Eigenvalues ∈ [0, 2]; L_sym has same eigenvector structure as L

Random walk Laplacian:
  L_rw = D^{-1} · L
       = I - D^{-1} · W
  Eigenvalues ∈ [0, 2]; L_rw not symmetric but has same eigenvalues as L_sym
```

**Properties that must hold (validation targets)**:
- All eigenvalues of L are ≥ 0 (positive semi-definite)
- Smallest eigenvalue = 0 (corresponding eigenvector = constant vector [1,1,...,1]/√n)
- Multiplicity of eigenvalue 0 = number of connected components
- Second smallest eigenvalue = **Fiedler value** (algebraic connectivity λ₂)
- λ₂ > 0 ↔ graph is connected

### 1.2 scipy.sparse.csgraph.laplacian

```python
from scipy.sparse.csgraph import laplacian

# Unnormalized Laplacian:
L = laplacian(W)           # returns dense array if W is dense, sparse if sparse
L_sparse = laplacian(sp.csr_matrix(W))

# Symmetric normalized Laplacian:
L_sym = laplacian(W, normed=True)   # returns D^{-1/2} L D^{-1/2}

# With return_diag (returns degree vector as well):
L, d = laplacian(W, return_diag=True)  # d = diagonal of D (degree vector)
```

**Trap: normed=True** computes L_sym, NOT L_rw. For L_rw you must compute manually:
```python
D_inv = sp.diags(1.0 / np.asarray(W.sum(axis=1)).ravel())
L_rw = sp.eye(n) - D_inv @ W
```

### 1.3 NetworkX Laplacian

```python
import networkx as nx

L_dense = nx.laplacian_matrix(G).toarray()           # unnormalized, returns sparse → convert
L_sym_dense = nx.normalized_laplacian_matrix(G).toarray()  # symmetric normalized
```

**Note**: `nx.laplacian_matrix` returns a scipy sparse matrix, not numpy array.
Must call `.toarray()` or `.todense()` to get dense.

### 1.4 Fiedler Value (Algebraic Connectivity)

```python
from scipy.sparse.linalg import eigsh

# Compute 2 smallest eigenvalues of Laplacian:
eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')  # SM = Smallest Magnitude
# eigenvalues[0] ≈ 0  (numerical, should be < 1e-10)
# eigenvalues[1] = Fiedler value

fiedler_value = eigenvalues[1]
fiedler_vector = eigenvectors[:, 1]
```

```python
# NetworkX (uses scipy internally):
fiedler_value = nx.algebraic_connectivity(G)
fiedler_vector = nx.fiedler_vector(G)   # eigenvector corresponding to λ₂
```

**Trap: `eigsh` with which='SM'**. For sparse matrices, 'SM' (smallest magnitude)
is numerically unstable for near-singular matrices. Use `sigma=0.0` shift-invert:
```python
eigenvalues, eigenvectors = eigsh(L, k=2, sigma=0.0, which='LM')
# Much more stable for Laplacians (which have a zero eigenvalue)
```

### 1.5 Validation Targets for Laplacian

```python
import numpy as np
from scipy.sparse.csgraph import laplacian

# Simple path graph: 0-1-2-3 (4 nodes, weights=1)
W_path = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
], dtype=float)

L = laplacian(W_path)
# Expected L:
# [[ 1, -1,  0,  0],
#  [-1,  2, -1,  0],
#  [ 0, -1,  2, -1],
#  [ 0,  0, -1,  1]]

eigenvalues = np.linalg.eigvalsh(L)
# Expected (ascending): [0, 0.586, 2.0, 3.414]
# = [0, 2-√2, 2, 2+√2]
# Fiedler value = 2 - √2 ≈ 0.5858

# Complete graph K4 (all weights=1, diagonal=0):
W_K4 = np.ones((4, 4)) - np.eye(4)
L_K4 = laplacian(W_K4)
eigenvalues_K4 = np.linalg.eigvalsh(L_K4)
# Expected: [0, 4, 4, 4]
# (n=4 nodes, all connected: λ₂ = λ₃ = λ₄ = n = 4)

# Disconnected graph (two components):
W_disc = np.block([
    [np.ones((2,2)) - np.eye(2), np.zeros((2,2))],
    [np.zeros((2,2)), np.ones((2,2)) - np.eye(2)]
])
L_disc = laplacian(W_disc)
eigenvalues_disc = np.linalg.eigvalsh(L_disc)
# Expected: [0, 0, 1, 1]  (two zero eigenvalues = two components)
```

### 1.6 R: igraph Laplacian

```r
library(igraph)

# Create graph:
g <- graph_from_adjacency_matrix(W_path, mode="undirected", weighted=TRUE)

# Laplacian matrix:
L <- laplacian_matrix(g, normalized=FALSE)  # unnormalized
L_sym <- laplacian_matrix(g, normalized=TRUE)  # symmetric normalized

# Algebraic connectivity:
ac <- algebraic_connectivity(g)   # Fiedler value
fv <- fiedler_vector(g)           # Fiedler vector
```

---

## 2. Shortest Paths

### 2.1 Algorithm Selection Guide

| Algorithm | Graph type | Complexity | When to use |
|-----------|-----------|-----------|------------|
| Dijkstra | Positive weights | O((V+E) log V) | Default for positive weights |
| Bellman-Ford | Negative weights OK | O(V·E) | Contains negative edges |
| Floyd-Warshall | Any weights (no neg cycles) | O(V³) | All-pairs, small graphs |
| Johnson's | Sparse, negative weights | O(V² log V + VE) | Sparse all-pairs |
| BFS | Unweighted | O(V+E) | Unweighted graphs |

### 2.2 scipy.sparse.csgraph

```python
from scipy.sparse.csgraph import (
    dijkstra,
    bellman_ford,
    johnson,
    shortest_path,     # wrapper that auto-selects algorithm
    floyd_warshall,
    breadth_first_search,
    depth_first_search,
)

# W: sparse or dense matrix; W[i,j] = edge weight (0 = no edge)
# For unweighted: use W[i,j] = 1

# All-pairs Dijkstra (positive weights only):
D = dijkstra(W)
# D[i,j] = shortest path length from i to j
# D[i,j] = inf if no path exists

# Single-source Dijkstra (from node 0):
D_row = dijkstra(W, indices=0)                   # 1D array, length n
D_rows = dijkstra(W, indices=[0, 5, 10])         # 3 rows of D

# With predecessor matrix (for reconstructing paths):
D, pred = dijkstra(W, return_predecessors=True)
# pred[i, j] = k means: on path i→j, the node before j is k
# pred[i, j] = -9999 if no path

# Reconstruct path from node 0 to node 5:
def get_path(pred, i, j):
    path = [j]
    while path[-1] != i:
        path.append(pred[i, path[-1]])
    return path[::-1]

path_0_to_5 = get_path(pred, 0, 5)

# Directed vs undirected:
D_directed = dijkstra(W, directed=True)    # default; W[i,j] ≠ W[j,i] possible
D_undir = dijkstra(W, directed=False)      # treat as undirected (uses min(W[i,j], W[j,i]))
```

```python
# Bellman-Ford (handles negative weights, detects negative cycles):
D, pred = bellman_ford(W, return_predecessors=True)
# Raises NegativeCycleError if a negative cycle is reachable from source

# All-pairs Floyd-Warshall:
D = floyd_warshall(W)
D, pred = floyd_warshall(W, return_predecessors=True)

# Auto-select (recommended for production):
D = shortest_path(W)           # auto-selects based on graph properties
D = shortest_path(W, method='D')  # force Dijkstra
D = shortest_path(W, method='BF') # force Bellman-Ford
D = shortest_path(W, method='FW') # force Floyd-Warshall
D = shortest_path(W, method='J')  # force Johnson
```

**Critical trap: weight convention in scipy.csgraph**.
`W[i,j]` = the cost of traversing edge i→j. This is the DISTANCE, not the STRENGTH.
For graphs built from similarities (high value = close), you must invert:
```python
# From affinity/similarity matrix to distance matrix for shortest path:
W_dist = 1.0 / W_similarity  # or -log(W_similarity), depending on context
np.fill_diagonal(W_dist, 0)   # self-distance = 0
```

**Critical trap: 0 in scipy.csgraph means no edge**, not zero-weight edge.
For a graph where edge weight = 0 is valid, use a special encoding or add an epsilon.

### 2.3 Validation Targets for Dijkstra

```python
import numpy as np
from scipy.sparse.csgraph import dijkstra

# Triangle with weights: 0-1 (weight 1), 1-2 (weight 2), 0-2 (weight 4)
W_triangle = np.array([
    [0, 1, 4],
    [1, 0, 2],
    [4, 2, 0],
], dtype=float)

D = dijkstra(W_triangle, directed=False)
# Expected D:
# [[0, 1, 3],    (0→2 via 1: 1+2=3, not direct 4)
#  [1, 0, 2],
#  [3, 2, 0]]

# Disconnected graph:
W_disc = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0],  # node 2 isolated
], dtype=float)
D_disc = dijkstra(W_disc, directed=False)
# Expected: D_disc[0,2] = inf, D_disc[1,2] = inf
```

### 2.4 NetworkX Shortest Paths

```python
import networkx as nx

G = nx.from_numpy_array(W_triangle)

# Single-source:
lengths = nx.single_source_dijkstra_path_length(G, source=0, weight='weight')
paths = nx.single_source_dijkstra_path(G, source=0, weight='weight')

# All-pairs:
all_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
# all_lengths[i][j] = shortest path from i to j

# Bellman-Ford:
pred, dist = nx.bellman_ford_predecessor_and_distance(G, source=0, weight='weight')

# Floyd-Warshall:
D_nx = nx.floyd_warshall_numpy(G, weight='weight')  # returns numpy array
```

### 2.5 R: igraph Shortest Paths

```r
library(igraph)

g <- graph_from_adjacency_matrix(W_triangle, mode="undirected", weighted=TRUE)

# All-pairs shortest paths:
D_r <- distances(g, algorithm="dijkstra")  # or "bellman-ford", "johnson", "unweighted"

# Single-source:
D_from_0 <- distances(g, v=1, algorithm="dijkstra")  # R is 1-indexed

# With actual paths:
paths <- shortest_paths(g, from=1, to=3, weights=E(g)$weight)
paths$vpath[[1]]  # node sequence
```

---

## 3. Minimum Spanning Tree

### 3.1 Definitions

MST: subset of edges that connects all n nodes with total minimum weight and no cycles.
For n nodes: MST has exactly n-1 edges.

**Algorithms**:
- **Kruskal**: sort all edges by weight, greedily add non-cycle edges. O(E log E).
- **Prim**: grow MST from one node, always add cheapest edge to frontier. O(E log V).

Kruskal is better for sparse graphs (E << V²). Prim is better for dense graphs.

### 3.2 scipy.sparse.csgraph.minimum_spanning_tree

```python
from scipy.sparse.csgraph import minimum_spanning_tree

# Returns sparse matrix with MST edges only (weight = original edge weight):
T = minimum_spanning_tree(W_sparse)  # W must be scipy sparse
# T is an upper-triangular sparse matrix (CSR format)
# T[i,j] = weight of MST edge if it exists, 0 otherwise

# Get MST edges as list:
T_coo = T.tocoo()
mst_edges = list(zip(T_coo.row, T_coo.col, T_coo.data))
# [(i1, j1, w1), (i2, j2, w2), ...]

# Total MST weight:
mst_total_weight = T.sum()

# Note: result is upper-triangular — edge (i,j) stored once with i < j
```

**Trap**: Input must be scipy sparse, not dense numpy array. Convert first:
```python
W_sparse = sp.csr_matrix(W_dense)
T = minimum_spanning_tree(W_sparse)
```

**Trap: scipy MST always assumes directed=False** (undirected). The upper-triangular
output represents undirected edges. There is no directed MST in scipy.csgraph.

### 3.3 Validation Targets for MST

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree

# 4-node graph with known MST:
W = np.array([
    [0, 1, 4, 0],
    [1, 0, 2, 5],
    [4, 2, 0, 3],
    [0, 5, 3, 0],
], dtype=float)

T = minimum_spanning_tree(sp.csr_matrix(W))
# MST edges: (0,1,1), (1,2,2), (2,3,3) — total weight = 6
# NOT (0,2,4) — edge (1,2,2) + (0,1,1) = path 0→1→2 is cheaper

T_array = T.toarray()
mst_weight = T.sum()
# Expected: mst_weight = 6.0

# Verify: T has exactly n-1=3 non-zero entries:
assert T.nnz == 3
```

### 3.4 NetworkX MST

```python
import networkx as nx

G = nx.from_numpy_array(W, create_using=nx.Graph())

# Kruskal's algorithm (default):
T_graph = nx.minimum_spanning_tree(G, algorithm='kruskal', weight='weight')
# T_graph is a Graph with only MST edges

# Prim's algorithm:
T_graph_prim = nx.minimum_spanning_tree(G, algorithm='prim', weight='weight')

# Get edge list:
mst_edges = list(T_graph.edges(data='weight'))
# [(i1, j1, w1), ...]

# Total weight:
total = T_graph.size(weight='weight')
```

### 3.5 R: igraph MST

```r
library(igraph)

g <- graph_from_adjacency_matrix(W, mode="undirected", weighted=TRUE)

# MST (default algorithm = prim for connected, kruskal for disconnected):
mst_g <- mst(g, weights=E(g)$weight, algorithm="kruskal")

# Edge list:
as_edgelist(mst_g)

# Total weight:
sum(E(mst_g)$weight)
```

### 3.6 Connection to HDBSCAN (F20)

HDBSCAN internally builds an MST of the **mutual reachability distance** graph:

```
mrd(xᵢ, xⱼ) = max(core_dist_k(xᵢ), core_dist_k(xⱼ), d(xᵢ, xⱼ))
```

where `core_dist_k(x)` = distance from x to its k-th nearest neighbor.

HDBSCAN's cluster hierarchy = single-linkage clustering on the MST of mrd.
When tambear builds HDBSCAN (F20), the MST primitive from F29 is reused directly.

```python
# From hdbscan package (inspect internals):
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(X)
clusterer.minimum_spanning_tree_.plot()   # visual inspection of MST
mst_data = clusterer.minimum_spanning_tree_.to_pandas()  # edge weights
```

---

## 4. Community Detection (Graph Clustering)

### 4.1 What Community Detection Computes

Given a graph G, partition nodes into communities (groups) that maximize intra-community
edges and minimize inter-community edges. No need to specify K in advance.

**Modularity Q** is the canonical objective:

```
Q = (1/2m) · Σᵢⱼ [Aᵢⱼ - kᵢkⱼ/(2m)] · δ(cᵢ, cⱼ)
```

where:
- `m` = total edge weight (Σᵢⱼ Aᵢⱼ / 2)
- `kᵢ` = degree of node i (row sum of A)
- `cᵢ` = community assignment of node i
- `δ(cᵢ, cⱼ)` = 1 if same community, 0 otherwise

Q ∈ (-1, 1). Q > 0.3 indicates meaningful community structure.
Maximizing Q is NP-hard. Louvain/Leiden are greedy approximations.

### 4.2 Louvain Algorithm

```python
# python-louvain (community package):
import community as community_louvain  # pip install python-louvain

partition = community_louvain.best_partition(G, weight='weight')
# Returns dict: {node_id: community_id}

modularity = community_louvain.modularity(partition, G, weight='weight')

# Convert to array:
labels = np.array([partition[i] for i in range(G.number_of_nodes())])

# Control resolution (higher = more, smaller communities):
partition_fine = community_louvain.best_partition(G, weight='weight', resolution=1.5)
partition_coarse = community_louvain.best_partition(G, weight='weight', resolution=0.5)
```

```python
# igraph Louvain (faster, C-backed):
import igraph as ig

g = ig.Graph.from_networkx(G)
community_result = g.community_louvain(weights='weight')

# Returns VertexClustering object:
labels_ig = community_result.membership       # list of community IDs
modularity_ig = community_result.modularity   # Q value
n_communities = len(community_result)         # number of communities
```

```r
library(igraph)

g <- graph_from_adjacency_matrix(W, mode="undirected", weighted=TRUE)
comm <- cluster_louvain(g, weights=E(g)$weight)

membership(comm)      # node-to-community mapping
modularity(comm)      # Q value
length(comm)          # number of communities
```

**Critical trap: Louvain is non-deterministic.** Results vary by random seed.
For reproducible research: fix seed and run multiple times, select best Q.
```python
# Multiple runs, select best modularity:
best_Q = -1
best_partition = None
for seed in range(20):
    partition = community_louvain.best_partition(G, weight='weight', random_state=seed)
    Q = community_louvain.modularity(partition, G)
    if Q > best_Q:
        best_Q = Q
        best_partition = partition
```

**Trap: resolution parameter.** Default resolution=1.0 matches the standard modularity
definition. Changing resolution changes the effective scale of detected communities.
Document which resolution was used in any validation target.

### 4.3 Leiden Algorithm (Improved Louvain)

Leiden guarantees that detected communities are internally connected — Louvain can
produce disconnected communities. Leiden is strictly superior to Louvain.

```python
import leidenalg  # pip install leidenalg
import igraph as ig

g = ig.Graph.from_networkx(G)

# Default (ModularityVertexPartition = maximize Q):
partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
labels_leiden = partition.membership
modularity_leiden = partition.quality()

# With resolution parameter:
partition_fine = leidenalg.find_partition(
    g,
    leidenalg.CPMVertexPartition,  # Constant Potts Model — resolution-dependent
    resolution_parameter=0.05
)

# Surprise partition (no resolution parameter, data-driven):
partition_surprise = leidenalg.find_partition(g, leidenalg.SurpriseVertexPartition)

# Reproducibility:
partition_rep = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=42)
```

### 4.4 Girvan-Newman (Edge Betweenness)

Detects communities by progressively removing edges with highest betweenness.

```python
import networkx as nx
from networkx.algorithms.community import girvan_newman

# Generator yielding communities at each edge removal step:
community_generator = girvan_newman(G)

# First split (2 communities):
top_level = next(community_generator)
communities_2 = sorted(map(sorted, top_level))

# Second split (3 communities):
second_level = next(community_generator)
communities_3 = sorted(map(sorted, second_level))

# Find optimal K by modularity:
best_Q = -1
best_partition = None
for communities in nx.algorithms.community.girvan_newman(G):
    partition = {n: i for i, c in enumerate(communities) for n in c}
    Q = nx.algorithms.community.modularity(G, communities)
    if Q > best_Q:
        best_Q = Q
        best_partition = communities
    if len(communities) > 10:  # stop early
        break
```

**Trap: Girvan-Newman is O(E · V)** per edge removal = extremely slow for large graphs.
Only use for small graphs (n < ~500) or for educational/oracle purposes.

### 4.5 Spectral Community Detection

Uses eigendecomposition of the Laplacian to embed nodes, then clusters in embedding.
This is the **tambear-native** path: reuses F22 EigenDecomposition.

```python
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def spectral_community_detection(W, n_communities, normalized=True):
    L = laplacian(W, normed=normalized)

    # First k eigenvectors (smallest eigenvalues):
    eigenvalues, eigenvectors = eigsh(L, k=n_communities, sigma=0.0, which='LM')

    # Normalize rows (for normalized Laplacian):
    if normalized:
        norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
        eigenvectors = eigenvectors / np.maximum(norms, 1e-10)

    # KMeans in spectral embedding:
    km = KMeans(n_clusters=n_communities, n_init=10, random_state=42)
    labels = km.fit_predict(eigenvectors)
    return labels
```

```python
# sklearn wrapper:
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(
    n_clusters=3,
    affinity='precomputed',   # pass W directly as affinity matrix
    assign_labels='kmeans',
    n_init=10,
    random_state=42,
)
labels_spectral = sc.fit_predict(W_affinity)
```

**Connection to F22**: The only new ingredient in spectral community detection vs.
plain spectral clustering (F20) is the construction of the Laplacian from the graph
rather than from a distance/kernel matrix. The EigenDecomposition step is identical.

### 4.6 Validation Targets for Community Detection

```python
import numpy as np
import networkx as nx
import community as community_louvain

# Two cliques connected by a single bridge:
G_test = nx.Graph()
# Clique 1: nodes 0-4
for i in range(5):
    for j in range(i+1, 5):
        G_test.add_edge(i, j, weight=1.0)
# Clique 2: nodes 5-9
for i in range(5, 10):
    for j in range(i+1, 10):
        G_test.add_edge(i, j, weight=1.0)
# Bridge:
G_test.add_edge(4, 5, weight=0.1)

partition = community_louvain.best_partition(G_test, random_state=42)
Q = community_louvain.modularity(partition, G_test)

# Expected: 2 communities ({0,1,2,3,4} and {5,6,7,8,9})
# Expected Q ≈ 0.48 (two dense cliques with weak bridge)
communities_found = set(partition.values())
assert len(communities_found) == 2

# Verify clique membership:
comm_of_0 = partition[0]
for node in range(1, 5):
    assert partition[node] == comm_of_0  # same community as node 0
```

---

## 5. PageRank and Centrality Measures

### 5.1 PageRank

PageRank computes the stationary distribution of a random walk with teleportation.
This is the canonical Kingdom C (iterative) algorithm.

**Formula**:
```
PR(v) = (1-d)/n + d · Σ_{u→v} PR(u) / out_degree(u)
```
where d = damping factor (typically 0.85), n = number of nodes.

Matrix form (power iteration):
```
r_{t+1} = d · A_rw · r_t + (1-d)/n · 1
```
where `A_rw = A_row_normalized` (each row sums to 1 = row-stochastic matrix).

Convergence: typically 50-100 iterations to tolerance 1e-6.

```python
import networkx as nx

G = nx.DiGraph()  # PageRank is for directed graphs
G.add_weighted_edges_from([(0,1,1), (0,2,1), (1,2,1), (2,0,1)])

# Default: damping=0.85, max_iter=100, tol=1e-6:
pr = nx.pagerank(G, alpha=0.85)               # dict: {node: pagerank}
pr_array = np.array([pr[i] for i in range(n)])

# Personalized PageRank (non-uniform teleportation):
personalization = {0: 1.0, 1: 0.0, 2: 0.0}  # teleport only to node 0
pr_personal = nx.pagerank(G, alpha=0.85, personalization=personalization)

# Weighted PageRank:
pr_weighted = nx.pagerank(G, alpha=0.85, weight='weight')
```

```python
# igraph (faster):
import igraph as ig
g_directed = ig.Graph(n=3, edges=[(0,1),(0,2),(1,2),(2,0)], directed=True)
pr_ig = g_directed.pagerank(damping=0.85)
# Returns list of pagerank values [pr_0, pr_1, pr_2]
```

```r
library(igraph)
g_r <- graph_from_adjacency_matrix(A_directed, mode="directed", weighted=TRUE)
pr_r <- page_rank(g_r, damping=0.85)$vector
```

### 5.2 Validation Targets for PageRank

```python
import networkx as nx
import numpy as np

# 3-cycle: 0→1→2→0
G_cycle = nx.DiGraph()
G_cycle.add_edges_from([(0,1),(1,2),(2,0)])

pr = nx.pagerank(G_cycle, alpha=0.85)
pr_array = np.array([pr[i] for i in range(3)])

# Expected: uniform [1/3, 1/3, 1/3] by symmetry
assert np.allclose(pr_array, [1/3, 1/3, 1/3], atol=1e-6)

# Hub-and-spoke: 0→1, 0→2, 0→3, 0→4 (node 0 points to all others)
# Nodes 1-4 have no outgoing edges (dangling nodes)
G_hub = nx.DiGraph()
G_hub.add_edges_from([(0,1),(0,2),(0,3),(0,4)])

pr_hub = nx.pagerank(G_hub, alpha=0.85)
# Node 0 is source (high out-degree but no in-edges except dangling mass redistribution)
# Nodes 1-4 receive all pagerank from node 0
# Expected: PR(1) ≈ PR(2) ≈ PR(3) ≈ PR(4) > PR(0)
pr_arr = np.array([pr_hub[i] for i in range(5)])
assert pr_arr[1] > pr_arr[0]  # spokes outrank hub
```

### 5.3 Other Centrality Measures

```python
import networkx as nx

G_undir = nx.karate_club_graph()  # classic test graph, 34 nodes

# Degree centrality (fraction of nodes connected to):
dc = nx.degree_centrality(G_undir)           # {node: 0..1}

# Betweenness centrality (fraction of shortest paths through node):
bc = nx.betweenness_centrality(G_undir, weight='weight')  # expensive: O(VE)
bc_edge = nx.edge_betweenness_centrality(G_undir, weight='weight')

# Closeness centrality (inverse mean shortest path length):
cc = nx.closeness_centrality(G_undir)

# Eigenvector centrality (PageRank without teleportation):
ec = nx.eigenvector_centrality(G_undir, max_iter=1000, weight='weight')
ec_np = nx.eigenvector_centrality_numpy(G_undir, weight='weight')  # exact via eigsh

# Katz centrality (attenuated paths):
kc = nx.katz_centrality(G_undir, alpha=0.1, beta=1.0, weight='weight')

# HITS (Hubs and Authorities):
hubs, authorities = nx.hits(G_undir)
```

```r
library(igraph)

g <- make_graph("Zachary")  # karate club

betweenness(g)              # betweenness centrality
closeness(g)                # closeness centrality
eigen_centrality(g)$vector  # eigenvector centrality
page_rank(g, damping=0.85)$vector  # PageRank
```

**Tambear decomposition of centrality measures**:

| Measure | Kingdom | Primitive |
|---------|---------|----------|
| Degree centrality | A | accumulate(ByKey(node), Count/Sum, Add) |
| PageRank | C | repeated sparse matrix-vector multiply (power iteration) |
| Eigenvector centrality | C | power method on adjacency matrix |
| Betweenness | CPU sequential | BFS/Dijkstra per source; not GPU-native |
| Closeness | CPU sequential | all-pairs shortest paths first |

---

## 6. Random Walks and Graph Embeddings

### 6.1 Node2Vec

Node2Vec generates node embeddings via biased random walks + Word2Vec on walk sequences.

```python
from node2vec import Node2Vec  # pip install node2vec

# Setup:
node2vec = Node2Vec(
    G,                    # NetworkX graph
    dimensions=64,        # embedding dimension
    walk_length=30,       # length of each random walk
    num_walks=200,        # number of walks per node
    p=1.0,                # return parameter (controls BFS tendency)
    q=1.0,                # in-out parameter (controls DFS tendency)
    workers=4,
)

# Train Word2Vec on walks:
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get embedding for node 0:
emb_0 = model.wv[str(0)]   # shape: (64,)

# All embeddings as matrix:
embeddings = np.array([model.wv[str(n)] for n in G.nodes()])  # (n, 64)
```

**Parameters**:
- `p` (return): p < 1 = BFS-like (explore nearby); p > 1 = DFS-like (explore far)
- `q` (in-out): q < 1 = DFS-like (explore outward); q > 1 = BFS-like (stay local)
- p=1, q=1 = unbiased random walk = DeepWalk

### 6.2 DeepWalk

DeepWalk = Node2Vec with p=1, q=1 (unbiased). The implementation is the same.

```python
# DeepWalk = Node2Vec with p=1, q=1:
deepwalk = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, p=1, q=1)
model_dw = deepwalk.fit(window=10, min_count=1)
```

### 6.3 Connection to Manifold Learning (F28)

Random walk embeddings have deep connections to manifold learning:

- **Diffusion Maps**: the diffusion distance between nodes is approximated by the
  eigendecomposition of the row-stochastic transition matrix P = D^{-1}·W.
  This is the same as the random walk Laplacian L_rw = I - P.

```python
# Diffusion maps via random walk Laplacian:
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

# Row-stochastic transition matrix:
degrees = np.asarray(W.sum(axis=1)).ravel()
D_inv = sp.diags(1.0 / degrees)
P = D_inv @ W  # transition matrix

# Eigendecomposition:
eigenvalues, eigenvectors = eigsh(P, k=k+1, which='LM')  # largest eigenvalues
# eigenvalues[0] = 1 (trivial), eigenvalues[1:] = diffusion spectrum

# Diffusion embedding (scale by eigenvalues for diffusion distance):
t = 1  # diffusion time
embedding = eigenvectors[:, 1:] * (eigenvalues[1:] ** t)
```

**Key observation**: Diffusion maps = spectral embedding of P = same EigenDecomposition
primitive as F22, just applied to P instead of the covariance matrix. F29 diffusion maps
are FREE once F22 EigenDecomposition exists.

### 6.4 Spectral Embedding

```python
from sklearn.manifold import SpectralEmbedding

# Precomputed affinity matrix:
se = SpectralEmbedding(
    n_components=2,
    affinity='precomputed',
    eigen_solver='arpack',
    random_state=42,
)
embedding = se.fit_transform(W_affinity)  # (n, 2) node coordinates

# From data (builds affinity via kNN or RBF internally):
se_data = SpectralEmbedding(n_components=2, n_neighbors=10, affinity='nearest_neighbors')
embedding_data = se_data.fit_transform(X)
```

---

## 7. Tambear Decomposition by Kingdom

### Kingdom A: Commutative Scatter (GPU-native)

| Operation | Primitive | Notes |
|-----------|----------|-------|
| Degree matrix D | accumulate(ByKey(node_id), Sum(W[i,j]), Add) | Row sums of adjacency |
| Adjacency from distances | gather(threshold) + accumulate | Binary threshold then scatter |
| Graph construction from F01 | gather(DistancePairs) → threshold | Zero-cost on cached matrix |
| Edge weight histogram | accumulate(ByKey(weight_bin), Count, Add) | Standard histogram |
| Degree distribution | accumulate(ByKey(degree), Count, Add) | After degree computation |

**Laplacian construction**: Given adjacency W (sparse):
```
Step 1: D[i] = accumulate(ByKey(i), W[i,j], Add)   # degree per node
Step 2: L = scatter(-W) + scatter_diag(D)            # L = D - W
```

This is a ByKey scatter — Kingdom A. The Laplacian itself is a one-pass scatter.

### Kingdom C: Iterative (GPU sparse matrix-vector multiply)

| Operation | Pattern | Notes |
|-----------|---------|-------|
| PageRank | power iteration on D^{-1}W | Each iteration = sparse MVM |
| Eigenvector centrality | power method on A | Each iteration = sparse MVM |
| Diffusion maps | eigendecomposition of P | Multiple sparse MVMs (ARPACK) |
| Laplacian eigendecomposition | eigsh(L) | Multiple sparse MVMs (ARPACK) |

Sparse MVM = `accumulate(Tiled(n, n_sparse), Multiply(v), Add)` on sparse A.
In tambear: this is a ByKey scatter of weighted values, NOT a dense tiled accumulate.

**Sparse MVM decomposition**:
```
(A · v)[i] = Σⱼ A[i,j] · v[j]
           = accumulate(ByKey(row_index), A[i,j] * v[j], Add)
           = sum of weighted values grouped by destination row
```

This is exactly the `accumulate(ByKey, Multiply+Add)` pattern. Kingdom A primitive,
run iteratively. The iterative loop is the Kingdom C wrapper.

### CPU-Appropriate (not GPU-native)

| Algorithm | Why CPU | Notes |
|-----------|---------|-------|
| BFS / DFS | Sequential frontier expansion | Cannot vectorize node-by-node traversal |
| Dijkstra | Sequential priority queue updates | Frontier is inherently sequential |
| Bellman-Ford | Sequential edge relaxation | Each pass depends on previous |
| Floyd-Warshall | Inherent n×n×n sequential dependency | GPU version exists but complex |
| Louvain/Leiden | Node moves one at a time | Sequential local optimization |
| Girvan-Newman | Sequential edge removal + BFS | Too slow for large n anyway |
| Kruskal MST | Sort + sequential union-find | Union-find is not GPU-native |

**Design rule for F29 tambear**: GPU handles Laplacian construction + power iteration.
CPU handles all path-based algorithms. The boundary is sharp and structural.

### Sharing from DistancePairs (F01)

When graph is built from data (not from an explicit edge list):

```
DistancePairs (F01 cache)
  │
  ├── threshold(eps) → binary adjacency W ∈ {0,1}   [DBSCAN radius graph]
  ├── kNN(k) → k-nearest-neighbor graph              [spectral clustering input]
  ├── exp(-γ·D) → RBF affinity W                    [kernel graph]
  └── top_k_per_row → sparse approximation           [large-scale graphs]
```

All of these graph constructions are zero-cost once DistancePairs is computed.
They are gather + threshold operations on the cached matrix.

---

## 8. Performance Reality Check

### NetworkX vs igraph vs cuGraph

| Task | NetworkX | igraph | cuGraph |
|------|---------|--------|---------|
| Build graph (10k nodes, 100k edges) | ~200ms | ~5ms | ~1ms |
| Dijkstra all-pairs (1k nodes) | ~500ms | ~20ms | ~2ms |
| Louvain (100k nodes, 1M edges) | timeout | ~2s | ~0.2s |
| PageRank (1M nodes) | timeout | ~10s | ~0.5s |

For tambear target scale (financial data, N up to ~100k nodes per day), igraph is
sufficient. cuGraph (RAPIDS) is available if needed.

```python
# cuGraph (GPU, requires RAPIDS):
import cugraph
import cudf

# Build graph from edge list:
gdf = cudf.DataFrame({'src': src_nodes, 'dst': dst_nodes, 'weights': weights})
G_cu = cugraph.Graph()
G_cu.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='weights')

# PageRank:
pr_result = cugraph.pagerank(G_cu, alpha=0.85)

# Louvain:
parts, modularity = cugraph.louvain(G_cu)

# Shortest path:
distances = cugraph.sssp(G_cu, source=0)  # single-source shortest path
```

---

## 9. Key Traps Summary

1. **NetworkX is pure Python — profoundly slow.** Never use for n > 10k. igraph is
   10-100x faster. cuGraph is 100-1000x faster. Document which backend was used.

2. **Louvain is non-deterministic.** Results vary by run and random seed. Always
   set `random_state` and run multiple times for research validation.

3. **Modularity maximization is NP-hard.** Louvain is a greedy approximation that
   can get stuck in local optima. Different runs may give different Q values.

4. **Laplacian eigenvalues: smallest = 0 (trivial).** The Fiedler value is the
   SECOND smallest eigenvalue. `eigsh(L, k=2)` — not k=1.

5. **scipy.csgraph: W[i,j]=0 means no edge.** Zero-weight edges cannot be
   represented. Use a small epsilon or a different encoding.

6. **Weight convention inversion.** In affinity matrices: high value = strong
   connection. In distance matrices: high value = weak connection. Shortest-path
   algorithms want distances. RBF kernel graph needs W=exp(-γ·D) for affinity,
   then W_dist=1/W for Dijkstra. Know which convention your data is in.

7. **scipy.csgraph MST takes sparse input only.** Dense array input may silently
   treat zeros as valid zero-weight edges. Always convert to scipy.sparse first.

8. **`eigsh(L, which='SM')` is numerically unstable for Laplacians.** Use
   `sigma=0.0, which='LM'` (shift-invert mode) for reliable Fiedler computation.

9. **Leiden strictly dominates Louvain.** If you're implementing community detection,
   Leiden is the right choice. Louvain is only needed for historical comparison.

10. **Graph embedding (Node2Vec) is inherently stochastic** — random walks are random.
    Embeddings vary between runs unless random seed is fixed throughout the pipeline.

---

## 10. Validation Pipeline for F29

```python
"""
Gold standard validation oracle for F29 Graph Algorithms.
Run this once, capture outputs, use as regression targets.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian, dijkstra, minimum_spanning_tree
import networkx as nx

np.random.seed(42)

# Test graph: Zachary's karate club (standard benchmark)
G_karate = nx.karate_club_graph()
n = G_karate.number_of_nodes()
W_karate = nx.to_numpy_array(G_karate)
W_sparse = sp.csr_matrix(W_karate)

# --- Laplacian validation ---
L = laplacian(W_sparse)
L_sym = laplacian(W_sparse, normed=True)

evals = np.linalg.eigvalsh(L.toarray())
print("Karate Laplacian eigenvalues (first 5):", np.round(evals[:5], 4))
# Expected: [0, 0.4685, 1.0, 1.0, 1.1294] (verify against networkx)
print("Fiedler value:", round(evals[1], 4))
# Expected: ~0.4685 (karate club is well-connected)

L_nx = nx.laplacian_matrix(G_karate).toarray().astype(float)
assert np.allclose(L.toarray(), L_nx, atol=1e-10), "Laplacian mismatch scipy vs networkx"

# --- Shortest path validation ---
D_paths = dijkstra(W_sparse, directed=False)
avg_path_length = D_paths[D_paths < np.inf].mean()
diameter = D_paths[D_paths < np.inf].max()
print("Average shortest path:", round(avg_path_length, 4))  # Expected: ~2.408
print("Graph diameter:", int(diameter))                       # Expected: 5
print("% reachable pairs:", round(100*(D_paths < np.inf).mean(), 1))  # Expected: 100.0

nx_avg = nx.average_shortest_path_length(G_karate)
assert abs(avg_path_length - nx_avg) < 1e-4, f"Dijkstra mean mismatch: {avg_path_length} vs {nx_avg}"

# --- MST validation ---
T = minimum_spanning_tree(W_sparse)
mst_weight = T.sum()
n_mst_edges = T.nnz
print("MST weight:", round(mst_weight, 4))  # capture oracle value
print("MST edges:", n_mst_edges)            # Expected: n-1 = 33

assert n_mst_edges == n - 1, f"MST should have n-1={n-1} edges, got {n_mst_edges}"

T_nx = nx.minimum_spanning_tree(G_karate, weight='weight')
assert abs(mst_weight - T_nx.size(weight='weight')) < 1e-6, "MST weight mismatch"

# --- PageRank validation ---
pr = nx.pagerank(G_karate, alpha=0.85)
pr_arr = np.array([pr[i] for i in range(n)])
print("PageRank sum:", round(pr_arr.sum(), 6))   # Expected: 1.0
print("Max PageRank node:", pr_arr.argmax())      # Expected: node 0 (highest degree)
print("Max PageRank value:", round(pr_arr.max(), 4))  # capture oracle value

assert abs(pr_arr.sum() - 1.0) < 1e-6, "PageRank must sum to 1"

# --- Community detection validation ---
import community as community_louvain
partition = community_louvain.best_partition(G_karate, random_state=42)
Q = community_louvain.modularity(partition, G_karate)
n_communities = len(set(partition.values()))
print("Louvain communities:", n_communities)  # Expected: 4 (karate known to have 4)
print("Louvain modularity:", round(Q, 4))     # Expected: ~0.37-0.39

# Verify against known karate club split (Zachary's assignment):
true_communities = nx.get_node_attributes(G_karate, 'club')
# 'Mr. Hi' vs 'Officer' — the known 2-way split
# Louvain may give 4 communities, all consistent with the 2-way split
```

---

## Gold Standard Library Versions (pin for reproducibility)

```
networkx==3.4.*
scipy==1.14.*
igraph==0.11.*
python-louvain==0.16
leidenalg==0.10.*
node2vec==0.4.*
cuGraph (RAPIDS 25.02 for CUDA 12+)
```

R packages:
```
igraph >= 2.0.0
```

All validation outputs should be regenerated and captured when library versions change.
