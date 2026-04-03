# Family 27: Topological Data Analysis — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (H₀ = union-find on sorted distances) + custom (H₁ = boundary matrix column reduction)

---

## Core Insight: Two Very Different Computations

- **H₀ (connected components / persistent homology degree 0)**: Union-find on sorted distance pairs. Same as single-linkage clustering. ~100 lines. Uses existing DistancePairs.
- **H₁+ (loops, voids / persistent homology degree ≥ 1)**: Boundary matrix reduction. NO analog in existing tambear primitives. New algorithm needed.

---

## 1. Simplicial Complexes

### Vietoris-Rips Complex
For point cloud X and radius ε:
```
VR(X, ε) = {σ ⊂ X : d(x_i, x_j) ≤ ε for all x_i, x_j ∈ σ}
```
A simplex σ is included iff ALL pairwise distances are ≤ ε.

### Čech Complex (more precise but more expensive)
```
Č(X, ε) = {σ ⊂ X : ∩_{x∈σ} B(x, ε) ≠ ∅}
```
Requires checking if balls have common intersection. Computationally harder.

### Alpha Complex
Subcomplex of Delaunay triangulation. Same homotopy type as Čech but much smaller. Requires Delaunay triangulation (exact in low dimensions).

### For TDA: Rips complex is standard (only needs pairwise distances = F01).

---

## 2. Persistent Homology

### Key Idea
Instead of fixing ε, track topology as ε grows from 0 to ∞:
- At each ε, simplices appear (edges when d(x_i,x_j) ≤ ε)
- H₀: connected components merge (birth/death of components)
- H₁: loops form and fill in (birth/death of holes)
- H₂: voids form and fill in

### Filtration
Sequence of nested complexes: K_0 ⊂ K_1 ⊂ ... ⊂ K_m.
For Rips: the filtration value of edge (i,j) is d(x_i,x_j). The filtration value of a k-simplex = max edge weight.

### Persistence Diagram
Each topological feature is a point (birth, death) in the diagram.
- Long bars (|death - birth| large): robust features, real topology
- Short bars: noise
- Points near diagonal = noise

### Bottleneck Distance between diagrams
```
d_B(D₁, D₂) = inf_γ sup_p ‖p - γ(p)‖_∞
```
where γ ranges over bijections between diagrams (including matching to diagonal).

### Wasserstein Distance
```
W_p(D₁, D₂) = [inf_γ Σ ‖p - γ(p)‖_∞^p]^{1/p}
```

---

## 3. H₀: Persistent Homology Degree 0

### Algorithm: Modified Union-Find
```
1. Sort all edges by distance (filtration value)
2. Initialize: each point is its own component
3. For each edge (i,j) in sorted order:
   a. If Find(i) ≠ Find(j):
      - Record death of younger component (born later) at distance d(i,j)
      - Union(i, j)
```

### This IS single-linkage clustering dendrogram

### Implementation: Sort (F01 distance pairs) → union-find scan. ~100 lines.

### Kingdom: B (sequential scan of sorted edges with union-find state)

---

## 4. H₁+: Persistent Homology via Boundary Matrix Reduction

### Boundary Matrix
For a simplicial complex with simplices σ₁, ..., σ_m ordered by filtration value:
```
∂[σ_i, σ_j] = {  1  if σ_j is a face of σ_i (with sign for orientation)
               {  0  otherwise
```

For Rips: boundary of edge (i,j) = j - i. Boundary of triangle (i,j,k) = (j,k) - (i,k) + (i,j).

### Column Reduction (Standard Algorithm)
```
For j = 1 to m:
  While there exists j' < j with low(j') = low(j):
    ∂[·,j] = ∂[·,j] + ∂[·,j']    (over Z₂: XOR)
```
where low(j) = row index of lowest nonzero entry in column j.

After reduction:
- Columns with low(j) defined form persistence pairs: (low(j), j) = (birth, death)
- Zero columns: essential features (born, never die)

### Over Z₂ (mod 2 — standard for computational TDA)
All arithmetic is XOR. No sign tracking needed. Boundary of triangle: ∂(i,j,k) = (j,k) ⊕ (i,k) ⊕ (i,j).

### Complexity
- Standard algorithm: O(m³) worst case, but often much better in practice
- Twist optimization: skip already-paired columns
- **Chunk reduction** (GPU-friendly): process independent chunks in parallel

### CRITICAL: This is the only algorithm in the entire 35-family taxonomy that has no tambear primitive analog. It's a matrix reduction over Z₂ — neither a scan, nor an accumulate, nor an eigendecomposition.

### Optimized Algorithms
- **Persistent cohomology** (de Silva et al. 2011): Compute persistent cohomology instead — dual problem, often faster because the coboundary matrix is smaller.
- **Discrete Morse theory**: Reduce complex before computing homology. Can reduce by 90%+.
- **Clear/compress optimization**: Clear and compress columns during reduction.

### Kingdom: Custom — closest to C (iterative reduction) but the specific column reduction pattern is unique.

---

## 5. Persistence Landscapes

### Definition
For persistence diagram D with points (b_i, d_i):
```
λ_k(t) = k-th largest value of min(t - b_i, d_i - t)⁺
```
where x⁺ = max(x, 0).

### Properties
- λ_k is a piecewise linear function
- Lives in a Banach space → can compute means, do statistical tests
- More amenable to machine learning than raw diagrams

### Persistence Images (Adams et al. 2017)
Discretize persistence diagram on a grid, smooth with Gaussian:
```
ρ(x, y) = Σ_i w(b_i, d_i) · φ(x-b_i, y-d_i)
```
Flatten to vector → can use as feature for ML.

---

## 6. Mapper Algorithm

### Algorithm (Singh, Mémoli, Carlsson 2007)
```
1. Choose filter function f: X → R (e.g., density, eccentricity, first PCA component)
2. Cover range of f with overlapping intervals
3. For each interval: cluster the preimage using any clustering algorithm
4. Build graph: nodes = clusters, edges = shared points between clusters
```

### This IS a composition:
- f = F22 dimensionality reduction (PCA, eccentricity from F01)
- Clustering = F20 (DBSCAN or agglomerative per interval)
- Graph construction = F29

### No new math. Mapper = F22 + F20 wiring.

### Parameters
- Filter function: domain-specific. PCA₁ (variation), eccentricity (outlier detection), density (KDE from F08)
- Cover: number of intervals (resolution) + overlap percentage
- Clustering: algorithm + parameters per interval

---

## 7. Topological Features for Machine Learning

### Persistence Statistics
From diagram D = {(b_i, d_i)}:
- Max persistence: max(d_i - b_i)
- Mean persistence: mean(d_i - b_i)
- Persistence entropy: -Σ p_i log p_i where p_i = (d_i-b_i)/Σ(d_j-b_j)
- Betti numbers β_k at various scales

### Topological Signatures for Financial Time Series
- Sliding window embedding → point cloud → persistence → features
- H₁ persistence detects periodicities in time series (Perea & Harer 2015)
- Market crash detection via sudden topological changes

---

## Edge Cases

### Degenerate Distances
- All points coincident: one component immediately, no interesting topology
- Distance matrix has ties: need consistent tie-breaking for filtration ordering

### Computational Limits
- H₀: scales to millions of points (union-find is nearly O(n))
- H₁: Rips complex has O(n^3) triangles, O(n^4) tetrahedra → practical limit ~5000-10000 points
- **Mitigation**: Use sparse Rips (subsample + Čech approximation), alpha complex, or witness complex

### Numerical Issues
- Persistence ≈ 0 (diagonal points): threshold as noise. Stability theorem guarantees small diagram changes for small data perturbations.

---

## Sharing Surface

### Reuse from Other Families
- **F01 (Distance)**: ALL of TDA starts with pairwise distances
- **F20 (Clustering)**: H₀ = single-linkage dendrogram. Mapper uses clustering.
- **F22 (Dimensionality Reduction)**: Mapper filter functions (PCA, etc.)
- **F29 (Graph Algorithms)**: Mapper graph construction, union-find

### Consumers of F27
- **Fintek**: Topological market regime detection
- **F26 (Complexity)**: Topological complexity measures
- **F22 (Dimensionality Reduction)**: Topological features as embedding coordinates

### Structural Rhymes
- **H₀ = single-linkage clustering (F20)**: same union-find, different extraction
- **Mapper = cover + cluster + connect**: same as nerve theorem in algebraic topology
- **Persistence diagram = birth/death of features across scales**: same as wavelet multiresolution (F03)
- **Betti numbers = rank of homology groups**: discrete analog of Euler characteristic

---

## Implementation Priority

**Phase 1** — H₀ (~100 lines, as noted in task):
1. Sorted edge list from F01 distance pairs
2. Union-find with persistence tracking
3. Persistence diagram (H₀ birth/death pairs)
4. Persistence barcode visualization data

**Phase 2** — H₁ boundary matrix (~150 lines, as noted in task):
5. Rips complex construction (up to dimension 2)
6. Boundary matrix (sparse, over Z₂)
7. Column reduction (standard algorithm + twist optimization)
8. H₁ persistence pairs
9. Bottleneck + Wasserstein distance between diagrams

**Phase 3** — Mapper + features (~100 lines):
10. Mapper algorithm (F22 filter + F20 cluster + graph wiring)
11. Persistence landscapes
12. Persistence images
13. Persistence statistics (entropy, max, mean)

**Phase 4** — Optimizations (~100 lines):
14. Chunk parallelism for boundary reduction
15. Sparse Rips complex (for large point clouds)
16. Alpha complex (2D and 3D only)
17. Discrete Morse reduction (preprocessing)

---

## Composability Contract

```toml
[family_27]
name = "Topological Data Analysis"
kingdom = "A/B (H₀ = union-find on sorted distances) + custom (H₁ = boundary reduction)"

[family_27.shared_primitives]
h0_persistence = "Union-find on sorted distance pairs"
boundary_reduction = "Column reduction over Z₂ (NEW primitive)"
mapper = "F22 filter + F20 clustering + graph construction"
persistence_diagram = "Set of (birth, death) pairs per homology degree"

[family_27.reuses]
f01_distance = "Pairwise distance matrix for Rips complex"
f20_clustering = "H₀ = single-linkage. Mapper clustering."
f22_reduction = "Mapper filter functions (PCA, eccentricity)"
f29_graph = "Mapper graph construction, union-find"

[family_27.provides]
persistence_diagrams = "H₀, H₁, H₂ persistence"
mapper_graph = "Topological summary of data shape"
persistence_features = "Landscapes, images, statistics for ML"
diagram_distances = "Bottleneck, Wasserstein between diagrams"

[family_27.consumers]
fintek = "Market regime detection via topology"
f26_complexity = "Topological complexity measures"

[family_27.session_intermediates]
rips_complex = "RipsComplex(data_id, max_dim) — simplicial complex"
persistence = "PersistenceDiagram(data_id, degree) — birth/death pairs"
mapper = "MapperGraph(data_id, filter, cover) — graph + metadata"
```
