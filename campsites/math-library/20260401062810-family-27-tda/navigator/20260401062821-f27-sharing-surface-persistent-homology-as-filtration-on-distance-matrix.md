# F27 Sharing Surface: TDA as Filtration on Distance Matrix

Created: 2026-04-01T06:28:21-05:00
By: navigator

Prerequisites: F01 complete (DistancePairs).

---

## Core Insight: TDA = Sorted Distance Matrix + Graph Algorithms

Topological Data Analysis (TDA) builds a sequence of simplicial complexes from the data.
The fundamental operation: threshold the distance matrix at increasing values ε and track
the topology of the resulting graph.

**Vietoris-Rips filtration**:
```
VR(ε) = {σ : max_{u,v ∈ σ} d(u,v) ≤ ε}
```
At threshold ε: add an edge (i,j) if d(i,j) ≤ ε. Track when connected components merge and cycles appear.

**This IS the DistancePairs matrix, sorted by distance value.** The filtration is just
iterating through sorted distances — scanning the SortedPermutation of DistancePairs.

The key new operations:
1. **Union-Find** (for H₀ = connected components = Betti₀) — O(N α(N))
2. **Cycle detection** (for H₁ = loops = Betti₁) — boundary matrix reduction
3. **Persistence pairing** — track birth/death thresholds

None of these require new accumulate primitives. Union-Find is a CPU data structure.
Cycle detection via boundary matrix reduction is O(N³) in the worst case — for small N.

---

## Persistent Homology

**What it computes**: for each "topological feature" (connected component, loop, void),
its BIRTH (ε when it appears) and DEATH (ε when it merges into something older).

```
Persistence diagram = {(birth_k, death_k) : k = feature index}
Persistence barcode = same, drawn as intervals
Persistence landscape = functional summary of the diagram
```

**Betti numbers**: β₀ = connected components, β₁ = loops, β₂ = voids (3D).

### H₀ (Betti₀ — connected components)

At each threshold ε, as we add edges from the sorted distance matrix:
- When edge (i,j) is added: if i and j are in different components → MERGE (death of one)
- The last surviving component persists forever

This is **incremental Union-Find**: process edges in sorted order, union-find with timestamps.
The persistence of component k = `(birth_k, death_k)` = distance thresholds.

**Tambear decomposition**:
1. Sort all pairwise distances: `SortedPermutation(DistancePairs)` — F01 + sort
2. For each edge (i,j) in sorted order: `union_find.union(i, j)` — O(N α(N)) CPU
3. Record `(birth, death)` pairs when components merge

O(N² log N) for sorting + O(N² α(N)) for union-find. GPU-amenable for the sort; union-find is inherently sequential but fast.

### H₁ (Betti₁ — loops)

When adding edge (i,j) closes a loop (both i and j in same component): a 1-cycle is born.
Death = when a higher-dimensional simplex fills the loop (triangle {i,j,k}).

For Vietoris-Rips: 2-simplex {i,j,k} exists when all three pairwise distances ≤ ε.
Processing requires tracking triangles — O(N³) simplices in the worst case.

**Phase 1**: H₀ only (connected components). H₁+ is Phase 2.
**Justification**: H₀ is sufficient for most clustering applications and is O(N²) total.

### Persistent Diagram Distances

Comparing two datasets' topologies via:
- Bottleneck distance: `dB(D1, D2) = inf_matching max_k ||(birth_k, death_k) - (birth'_k, death'_k)||∞`
- Wasserstein-p distance: similar, with p-norm

Computing optimal matching = assignment problem (Hungarian algorithm). O(N³).
For N < 1000 features: feasible on CPU.

---

## Mapper Algorithm (TDA visualization)

Mapper creates a low-dimensional graph from a filter function f: X → ℝ:
1. Divide the range of f into overlapping intervals
2. For each interval, cluster the preimage f^{-1}(interval) via DBSCAN
3. Connect nodes when clusters share data points between adjacent intervals

**Tambear decomposition**:
- f(x) typically = first PCA component = F22 projection
- Interval clustering = DBSCAN on subset = F20 clustering
- Node connectivity = set intersection = CPU-side

Mapper = F22 (PCA) + F20 (DBSCAN) + CPU graph construction. No new primitives.

---

## MSR Types F27 Produces

```rust
pub struct PersistenceDiagram {
    pub dimension: u8,         // 0 for H₀, 1 for H₁, etc.
    pub features: Vec<(f64, f64)>,  // (birth_ε, death_ε) pairs; death=inf for essential features
    pub n_essential: usize,    // features that survive to death=∞ (Betti number)
}

pub struct BarcodeStats {
    pub betti: Vec<usize>,     // Betti numbers [β₀, β₁, β₂, ...]
    pub total_persistence: Vec<f64>,  // Σ (death - birth) per dimension
    pub entropy: Vec<f64>,            // persistence entropy per dimension
    pub max_persistence: Vec<f64>,    // max single feature lifespan
}

pub struct MapperGraph {
    pub nodes: Vec<usize>,         // cluster IDs
    pub node_size: Vec<usize>,     // number of points per node
    pub edges: Vec<(usize, usize)>, // (node_i, node_j) adjacency
    pub filter_values: Vec<f64>,   // mean filter value per node
}
```

---

## Build Order

**Phase 1 (H₀ persistent homology)**:
1. `SortedPermutation` of DistancePairs (O(N² log N)) — sort infrastructure from F08
2. Incremental union-find with timestamps (~50 lines CPU)
3. `PersistenceDiagram` for H₀ (~30 lines)
4. `BarcodeStats` from diagram (~20 lines)
5. Tests: `gudhi` Python library (H₀ persistence), `ripser` (H₀)

**Phase 2 (H₁, boundary matrix reduction)**:
1. Triangle enumeration (O(N³)) — only feasible for N < 5000
2. Boundary matrix reduction algorithm (~150 lines)
3. PersistenceDiagram for H₁

**Phase 3 (Mapper)**:
1. Filter function = F22 PCA projection (~5 lines wiring)
2. Interval clustering = F20 DBSCAN call per interval (~20 lines wiring)
3. Node connectivity from point overlap (~30 lines)

**Gold standards**:
- Python `gudhi` library: `gudhi.RipsComplex().persistence()`
- Python `ripser` library: `ripser.ripser(data)['dgms']`
- R `TDA` package: `ripsDiag()`

---

## The Lab Notebook Claim

> Persistent homology H₀ (connected components) is incremental union-find on sorted pairwise distances. The filtration IS the SortedPermutation of DistancePairs — scan through sorted edges, track component merges. F27 adds union-find (~50 lines CPU) and a persistence data structure (~50 lines) on top of F01's distance infrastructure. H₁ (loops) requires boundary matrix reduction (~150 lines) — the first algorithm in the library with no tambear primitive analog, just a specialized linear algebra over GF(2). H₀ alone covers most clustering/connectivity applications.
