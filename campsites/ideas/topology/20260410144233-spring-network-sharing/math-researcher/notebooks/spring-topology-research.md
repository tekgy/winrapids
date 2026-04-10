# Spring Network Sharing — Mathematical Research Notes

## The Idea

TamSession's sharing graph can be modeled as a spring network / elastic graph.
Each intermediate is a node. Methods that share an intermediate are connected
by springs. The equilibrium configuration reveals the natural grouping of
computation — what should be computed together, what should be cached, what
should be recomputed.

## Mathematical Framework

### 1. Graph Laplacian

Given sharing graph G = (V, E) where:
- V = intermediates (MomentStats, CovarianceMatrix, SVD, ...)
- E = sharing edges (weighted by number of consumers)

The graph Laplacian L = D - A encodes the spring network:
- D = degree matrix (diagonal, d_ii = sum of edge weights from node i)
- A = adjacency matrix (a_ij = sharing weight between i and j)

### 2. Spectral clustering of the sharing graph

The Fiedler vector (second eigenvector of L) partitions the graph into
two communities. Recursive bipartition gives hierarchical clustering.

For TamSession: this reveals which intermediates form natural "phyla" —
groups that should be computed in the same pass.

### 3. Force-directed layout

Spring embedding: place nodes in 2D/3D, springs pull connected nodes together,
repulsion keeps them separated. The equilibrium reveals spatial structure.

Fruchterman-Reingold algorithm:
- Attractive force: f_a(d) = d² / k
- Repulsive force: f_r(d) = -k² / d
- Iterate until equilibrium

### 4. What this tells us about scheduling

**Strongly connected cliques** in the sharing graph = computation phases.
All intermediates in a clique should be computed in one pass.

**Bridges** between cliques = the minimum data that flows between phases.

**Articulation points** = intermediates whose removal disconnects the graph.
These are the critical cache entries — evicting them forces recomputation
of entire subgraphs.

## Connection to the Phyla Map

The 11 phyla I documented in the TamSession sharing graph map to this topology:

```
MomentStats ──── CovarianceMatrix ──── Eigendecomposition
                        │                      │
                        └── SVD ───────────────┘
                              │
Rank ──────── SortedData     │
                              │
DistanceMatrix ──── KernelMatrix
       │
DelayEmbedding
       │
ACF ──── FFT
```

The graph has two clear communities:
1. Linear algebra cluster: Moments → Covariance → Eigen/SVD
2. Time series cluster: ACF → FFT → DelayEmbedding → Distance

These should be the primary scheduling phases.

## Primitives needed

1. `graph_laplacian(adjacency)` — L = D - A
2. `fiedler_vector(laplacian)` — second eigenvector (spectral bisection)
3. `fruchterman_reingold(adjacency, n_iter)` — force-directed layout
4. `modularity(adjacency, partition)` — Newman modularity Q
5. `betweenness_centrality(adjacency)` — bridge detection
6. `articulation_points(adjacency)` — critical nodes

Most of these are graph.rs territory. The graph Laplacian is a matrix operation
(linear algebra primitive), and the Fiedler vector is an eigenvalue problem.
