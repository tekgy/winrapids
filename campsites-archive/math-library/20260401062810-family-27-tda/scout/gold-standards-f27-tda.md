# F27 Topological Data Analysis (TDA) — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 27 (Topological Data Analysis).
F27 is the first family in this library where the core algorithm (boundary matrix reduction)
has no tambear primitive analog — it is inherently sequential. The scout documents exactly
what each library computes, how to invoke it reproducibly, known output formats, and where
the tambear boundary lies: GPU for filtration construction, CPU for matrix reduction, GPU
again for persistence statistics.

Prerequisites: F01 complete (DistancePairs). The filtration IS the sorted DistancePairs matrix.

---

## Library Landscape

| Library | Language | What it computes | Notes |
|---------|----------|-----------------|-------|
| `ripser` | Python | Persistent homology (Vietoris-Rips), H0 + H1 + H2+ | Fastest Python TDA; C++ backend |
| `gudhi` | Python | Rips, Alpha, Cech complexes; persistence; landscapes | Most complete TDA library |
| `persim` | Python | Persistence diagram distances (Wasserstein, bottleneck) | Companion to ripser |
| `TDA` | R | `ripsDiag()`, `alphaComplexDiag()`, `kernelTDA()` | Wraps Ripser + GUDHI |
| `TDAstats` | R | Thin wrapper on `ripser` (via Rcpp) | Simpler API than TDA |
| `giotto-tda` | Python | High-level pipeline API around GUDHI | Good for sklearn integration |
| `scikit-tda` | Python | Meta-package; installs ripser + persim + kmapper | Convenience |
| `kmapper` | Python | Mapper algorithm | The canonical Python Mapper implementation |

**Primary oracle**: `ripser` (Python) for persistent homology — fastest, most tested.
**Secondary oracle**: `gudhi.RipsComplex` for cross-validation (different C++ code path).
**R oracle**: `TDA::ripsDiag()` for independent validation.
**Mapper oracle**: `kmapper` Python.
**Persistence statistics**: `persim` (Wasserstein/bottleneck).

---

## Part 1: Persistent Homology

### What It Computes

Given N points in ℝ^d:
1. Build Vietoris-Rips complex: at threshold ε, add simplex σ if max_{u,v ∈ σ} d(u,v) ≤ ε
2. Track topology as ε increases: connected components merge (H₀), loops appear/die (H₁),
   voids form/collapse (H₂), etc.
3. Output: persistence diagram = list of (birth_ε, death_ε) pairs per homology dimension

**Persistence barcode**: same data, drawn as horizontal intervals [birth, death].
**Essential features**: features with death = ∞ (survive forever = true topological features).
**Betti numbers**: β_k = count of essential H_k features at the end of filtration.

### Vietoris-Rips vs Alpha Complex

**Vietoris-Rips (ripser default)**:
- Conservative: simplex σ enters when ALL pairwise distances ≤ ε
- Works in any metric space — only needs pairwise distances
- Simplex count can be O(2^N) in worst case → max_dim matters

**Alpha complex (gudhi.AlphaComplex)**:
- Uses Delaunay triangulation to restrict simplices
- Much sparser than Rips — O(N) simplices in low dimensions
- Tighter filtration: only Delaunay simplices can appear
- Requires point coordinates, not just distances
- Generally preferred for low-dimensional point clouds (d ≤ 4)

**When to use which**:
- Raw points in ℝ^d, d ≤ 4: Alpha is faster and gives same H₀/H₁ results
- High-dimensional data (d >> 4): Rips from distance matrix
- Already have distance matrix (DistancePairs): Rips only, Alpha needs coordinates

---

### ripser (Python)

```python
import numpy as np
from ripser import ripser
from persim import plot_diagrams

# Basic usage — point cloud
X = np.random.randn(100, 2)  # 100 points in 2D
result = ripser(X)
# result['dgms'] = list of arrays, one per homology dimension
# result['dgms'][0] = H0 diagram: array of shape (n_components, 2)
# result['dgms'][1] = H1 diagram: array of shape (n_cycles, 2)
# Each row = (birth, death); death=inf for essential features

dgms = result['dgms']
h0 = dgms[0]  # H0: connected components
h1 = dgms[1]  # H1: loops (1-cycles)

print(f"H0 features: {len(h0)}")
print(f"H1 features: {len(h1)}")
print(f"Essential H0 (Betti0): {np.sum(np.isinf(h0[:, 1]))}")
print(f"Essential H1 (Betti1): {np.sum(np.isinf(h1[:, 1]))}")

# Optional parameters:
result = ripser(
    X,
    maxdim=2,          # compute H0, H1, H2 (default=1, H0+H1 only)
    thresh=2.0,        # stop filtration at distance threshold (default=inf)
    coeff=2,           # coefficient field (default=2; must be prime)
    distance_matrix=False,  # if True, X is treated as distance matrix
    metric='euclidean', # passed to sklearn.metrics.pairwise_distances if distance_matrix=False
    n_threads=-1,      # number of threads (-1 = all available)
)

# From a precomputed distance matrix (tambear path: pass DistancePairs):
D = np.zeros((100, 100))
# ... fill D with pairwise distances ...
result = ripser(D, distance_matrix=True)
```

**TRAP: max_dim defaults to 1 (H0 + H1 only).**
If you need H₂ (voids), explicitly set `maxdim=2`. Higher dims grow combinatorially.

**Output format for dgms[k]**:
- Array of shape `(n_features, 2)` where column 0 = birth, column 1 = death
- Infinite deaths are represented as `np.inf` (not -1 or any sentinel)
- The last H0 feature always has death = inf (the single final connected component)
- H0 features: sorted by birth ascending (all births = 0.0 for Rips with point cloud data)

```python
# Extract finite features only (ignore essential features):
def finite_features(dgm):
    return dgm[np.isfinite(dgm[:, 1])]

# Persistence (lifetime) of each feature:
def persistence(dgm):
    fin = finite_features(dgm)
    return fin[:, 1] - fin[:, 0]  # death - birth

# Betti number (count of essential features):
def betti(dgm):
    return np.sum(np.isinf(dgm[:, 1]))
```

**Validation target** (deterministic):
```python
import numpy as np
from ripser import ripser

# Circle in 2D: should have 1 H0 feature (1 component) and 1 H1 feature (1 loop)
np.random.seed(42)
theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
X_circle = np.column_stack([np.cos(theta), np.sin(theta)])
X_circle += np.random.randn(*X_circle.shape) * 0.05  # small noise

result = ripser(X_circle)
dgms = result['dgms']

# Expected:
# H0: all births=0, one death=inf, others finite
# H1: one prominent (birth, death) pair near (0.9, 2.1) range;
#     one essential feature (death=inf) indicating the loop

print("Betti0:", np.sum(np.isinf(dgms[0][:, 1])))   # expect: 1
print("Betti1:", np.sum(np.isinf(dgms[1][:, 1])))   # expect: 1

# Prominent H1 feature (finite: loop born, then filled):
finite_h1 = dgms[1][np.isfinite(dgms[1][:, 1])]
if len(finite_h1) > 0:
    max_persist_h1 = finite_h1[np.argmax(finite_h1[:, 1] - finite_h1[:, 0])]
    print(f"Most prominent finite H1: birth={max_persist_h1[0]:.4f}, death={max_persist_h1[1]:.4f}")
```

---

### gudhi (Python)

```python
import gudhi
import numpy as np

# RipsComplex — from point cloud
X = np.random.randn(100, 2)

rips = gudhi.RipsComplex(points=X, max_edge_length=2.0)
simplex_tree = rips.create_simplex_tree(max_dimension=2)
simplex_tree.compute_persistence()

# Get persistence pairs:
persistence = simplex_tree.persistence()
# persistence = list of (dimension, (birth, death)) tuples
# e.g., [(0, (0.0, 0.5)), (1, (0.3, 1.1)), (0, (0.0, inf)), ...]

# Extract per-dimension:
h0_gudhi = simplex_tree.persistence_intervals_in_dimension(0)  # array (n, 2)
h1_gudhi = simplex_tree.persistence_intervals_in_dimension(1)  # array (n, 2)

# From distance matrix:
rips_dm = gudhi.RipsComplex(distance_matrix=D, max_edge_length=2.0)
# NOTE: gudhi uses 'max_edge_length' not 'thresh' — same concept

# AlphaComplex (tighter, faster for low-d point clouds):
alpha = gudhi.AlphaComplex(points=X)
simplex_tree_alpha = alpha.create_simplex_tree()
simplex_tree_alpha.compute_persistence()
h0_alpha = simplex_tree_alpha.persistence_intervals_in_dimension(0)
h1_alpha = simplex_tree_alpha.persistence_intervals_in_dimension(1)

# Persistence image (functional summary):
pi = gudhi.representations.PersistenceImage(bandwidth=0.5, weight=lambda x: x[1], resolution=[20, 20])
h1_array = simplex_tree.persistence_intervals_in_dimension(1)
pi.fit([h1_array])
img = pi.transform([h1_array])  # shape (1, 400) for 20x20 image
```

**RipsComplex vs AlphaComplex output**: should give identical persistence diagrams for H₀ and H₁
on low-dimensional point clouds. Numerical differences of < 1e-10 expected from different
floating-point paths. Use for cross-validation.

**TRAP: gudhi `max_edge_length` is a HARD cutoff.** Features whose death > max_edge_length
will appear as essential (death = inf) even if they would die at a finite value. This is
the same as ripser's `thresh` parameter. For TDA in tambear: set high or np.inf to see
full diagram; apply threshold as a post-processing filter on the diagram.

**gudhi persistence output format**: list of `(dim, (birth, death))` tuples.
Infinite death is represented as `float('inf')`.
`persistence_intervals_in_dimension(k)` returns numpy array of shape `(n, 2)`.

```python
# Cross-validate ripser vs gudhi on same data:
import numpy as np
from ripser import ripser
import gudhi

np.random.seed(0)
X = np.random.randn(50, 3)

# ripser result:
r = ripser(X, maxdim=1)
h0_ripser = r['dgms'][0]
h1_ripser = r['dgms'][1]

# gudhi result:
rips = gudhi.RipsComplex(points=X, max_edge_length=1e10)
st = rips.create_simplex_tree(max_dimension=2)
st.compute_persistence()
h0_gudhi = st.persistence_intervals_in_dimension(0)
h1_gudhi = st.persistence_intervals_in_dimension(1)

# Sort both for comparison:
h0_r_sorted = np.sort(h0_ripser[np.isfinite(h0_ripser[:, 1]), 1])
h0_g_sorted = np.sort(h0_gudhi[np.isfinite(h0_gudhi[:, 1]), 1])
print(np.allclose(h0_r_sorted, h0_g_sorted, atol=1e-8))  # expect True
```

---

### R: TDA package

```r
library(TDA)

set.seed(42)
X <- matrix(rnorm(200), ncol = 2)

# Vietoris-Rips persistence diagram:
rips_diag <- ripsDiag(X, maxdimension = 1, maxscale = 5)
# Returns list with $diagram: matrix of (Dimension, Birth, Death)
# Infinite deaths stored as Inf

diagram <- rips_diag$diagram
h0 <- diagram[diagram[, "dimension"] == 0, ]  # H0 features
h1 <- diagram[diagram[, "dimension"] == 1, ]  # H1 features

print(paste("Betti0:", sum(is.infinite(h0[, "Death"]))))
print(paste("Betti1:", sum(is.infinite(h1[, "Death"]))))

# Alpha complex:
alpha_diag <- alphaComplexDiag(X, maxdimension = 1)
diagram_alpha <- alpha_diag$diagram

# From distance matrix:
D <- as.matrix(dist(X))
rips_diag_dm <- ripsDiag(D, maxdimension = 1, maxscale = 5, dist = "arbitrary")

# Bottleneck distance between diagrams:
db <- bottleneck(rips_diag$diagram, alpha_diag$diagram, dimension = 1)
print(paste("Bottleneck H1 (Rips vs Alpha):", db))  # expect near 0
```

---

### TDAstats (R)

```r
library(TDAstats)

X <- matrix(rnorm(200), ncol = 2)

# Simpler API wrapping ripser:
ph_result <- calculate_homology(X, dim = 1)
# Returns data frame with columns: dimension, birth, death

plot_barcode(ph_result)   # visualize barcode
plot_persist(ph_result)   # persistence diagram plot

# Access H1:
h1_df <- ph_result[ph_result$dimension == 1, ]
```

---

## Part 2: Persistence Statistics

### Bottleneck Distance

Measures topological similarity between two datasets.

```
dB(D1, D2) = min over perfect matchings η of max_k || p_k - q_{η(k)} ||_∞
```

Points can be matched to the diagonal (projection to birth=death line) — this handles
different numbers of features between D1 and D2.

**TRAP: Wasserstein distance between persistence diagrams is NOT the Wasserstein distance
between probability distributions.** `persim.wasserstein` computes diagram Wasserstein
(optimal matching of persistence pairs), not the 1D or 2D distribution Wasserstein.
These are completely different quantities.

```python
import numpy as np
from ripser import ripser
import persim

# Two point clouds:
np.random.seed(0)
X1 = np.random.randn(50, 2)
X2 = np.random.randn(50, 2) + 1.0  # shifted

dgms1 = ripser(X1)['dgms']
dgms2 = ripser(X2)['dgms']

# Bottleneck distance for H1:
d_bottleneck = persim.bottleneck(dgms1[1], dgms2[1])
print(f"Bottleneck H1: {d_bottleneck:.6f}")

# Wasserstein distance for H1 (diagram Wasserstein, not distribution Wasserstein!):
d_wasserstein = persim.wasserstein(dgms1[1], dgms2[1])
print(f"Wasserstein H1: {d_wasserstein:.6f}")

# Sliced Wasserstein (faster approximation):
sw = persim.sliced_wasserstein(dgms1[1], dgms2[1], M=50)  # M = number of slices
print(f"Sliced Wasserstein H1: {sw:.6f}")

# Persistence image distance (compare persistence images):
from persim import PersistenceImager
pimgr = PersistenceImager(pixel_size=0.2)
pimgr.fit(dgms1[1], dgms2[1])
img1 = pimgr.transform(dgms1[1])
img2 = pimgr.transform(dgms2[1])
pi_dist = np.linalg.norm(img1 - img2)
print(f"Persistence image L2 distance: {pi_dist:.6f}")
```

### Persistence Entropy

Shannon entropy of the probability distribution over feature lifetimes:

```
l_k = death_k - birth_k  (lifetime of feature k)
L = Σ_k l_k              (total persistence)
p_k = l_k / L            (normalized lifetime)
H = -Σ_k p_k * log(p_k) (persistence entropy)
```

```python
import numpy as np

def persistence_entropy(dgm, normalize=False):
    """
    Shannon entropy of persistence diagram.
    Uses finite features only (ignores essential features with death=inf).
    """
    fin = dgm[np.isfinite(dgm[:, 1])]
    if len(fin) == 0:
        return 0.0
    lifetimes = fin[:, 1] - fin[:, 0]
    lifetimes = lifetimes[lifetimes > 0]  # guard against zero-persistence features
    if len(lifetimes) == 0:
        return 0.0
    L = lifetimes.sum()
    p = lifetimes / L
    H = -np.sum(p * np.log(p))
    if normalize:
        H = H / np.log(len(lifetimes))  # divide by log(n) for H ∈ [0,1]
    return H

# gudhi has a built-in:
import gudhi.representations
pe = gudhi.representations.Entropy(mode='scalar')
# fit on list of diagrams, transform returns entropy per diagram
pe.fit([dgm_h1])
entropy_value = pe.transform([dgm_h1])
```

**Validation target**:
```python
import numpy as np
from ripser import ripser

np.random.seed(7)
# Uniform noise: many small features, low entropy
X_noise = np.random.rand(100, 2)
dgms_noise = ripser(X_noise)['dgms']

# Circle: one dominant feature, high entropy
theta = np.linspace(0, 2 * np.pi, 80)
X_circle = np.column_stack([np.cos(theta), np.sin(theta)])
X_circle += np.random.randn(*X_circle.shape) * 0.02
dgms_circle = ripser(X_circle)['dgms']

ent_noise = persistence_entropy(dgms_noise[1])
ent_circle = persistence_entropy(dgms_circle[1])
print(f"Entropy H1 (noise): {ent_noise:.4f}")
print(f"Entropy H1 (circle): {ent_circle:.4f}")
# Expect: circle has higher entropy (one dominant feature)
# OR circle has LOWER entropy (concentrated in one feature → low spread)
# Correct expectation: circle has LOWER entropy (one feature dominates → concentrated distribution)
```

**TRAP on persistence entropy direction**: A single dominant feature has all probability
mass on one lifetime → `p = [1.0]` → `H = 0`. Uniform many-feature diagram has
high entropy. So: structured (circle) → low entropy. Noisy → high entropy. This is
the OPPOSITE of intuition from signal processing where "structure = low entropy in spectrum."

### Betti Numbers

```python
import numpy as np
from ripser import ripser

def betti_numbers(X, max_dim=1, thresh=np.inf):
    """
    Compute Betti numbers β₀, β₁, ..., β_{max_dim} for point cloud X.
    These are the counts of essential (death=inf) features per dimension.
    """
    result = ripser(X, maxdim=max_dim, thresh=thresh)
    betti = []
    for k in range(max_dim + 1):
        dgm = result['dgms'][k]
        betti.append(int(np.sum(np.isinf(dgm[:, 1]))))
    return betti

# Validation:
np.random.seed(0)
# Torus (product of two circles) — β₀=1, β₁=2, β₂=1
# Full torus needs 3D; approximate with two-circle product:
theta = np.linspace(0, 2 * np.pi, 50)
X_figure8 = np.column_stack([
    np.concatenate([np.cos(theta), 2 + np.cos(theta)]),
    np.concatenate([np.sin(theta), np.sin(theta)])
])
X_figure8 += np.random.randn(*X_figure8.shape) * 0.03

b = betti_numbers(X_figure8, max_dim=1)
print(f"Figure-8 Betti numbers: β₀={b[0]}, β₁={b[1]}")
# Expect: β₀=1 (one component), β₁=2 (two loops)
```

### Total Persistence and Persistence Moments

```python
import numpy as np

def persistence_stats(dgm):
    """
    Compute standard scalar statistics from a persistence diagram.
    """
    fin = dgm[np.isfinite(dgm[:, 1])]
    if len(fin) == 0:
        return {
            'n_features': 0, 'total_persistence': 0.0,
            'mean_persistence': 0.0, 'max_persistence': 0.0,
            'entropy': 0.0
        }
    lifetimes = fin[:, 1] - fin[:, 0]
    L = lifetimes.sum()
    p = lifetimes / L if L > 0 else np.ones(len(lifetimes)) / len(lifetimes)
    H = -np.sum(p * np.log(np.maximum(p, 1e-15)))
    return {
        'n_features': len(fin),
        'total_persistence': float(L),
        'mean_persistence': float(np.mean(lifetimes)),
        'max_persistence': float(np.max(lifetimes)),
        'entropy': float(H),
        'std_persistence': float(np.std(lifetimes)),
    }
```

---

## Part 3: Mapper Algorithm

### What It Computes

Mapper produces a graph (nerve complex) summarizing the topological structure of data.

Algorithm:
1. Choose a filter function `f: X → ℝ` (e.g., first PCA component, eccentricity, density)
2. Cover range(f) with overlapping intervals `[a_k, b_k]`
3. For each interval: take preimage `f^{-1}([a_k, b_k])` (subset of points)
4. Cluster each preimage (typically DBSCAN or single-linkage)
5. Create node per cluster; add edge between nodes that share data points in adjacent intervals
6. Result: graph where nodes are local clusters, edges are overlaps

```python
import numpy as np
import kmapper as km
from sklearn import cluster, preprocessing

# Basic Mapper pipeline:
X = np.random.randn(200, 5)  # 200 points in 5D

mapper = km.KeplerMapper(verbose=0)

# Step 1: Projection (filter function) — lens = 1D projection
# Options: PCA projection, UMAP projection, eccentricity, density, custom
lens = mapper.fit_transform(X, projection='l2norm')  # L2 norm as filter
# Or: lens = mapper.fit_transform(X, projection=sklearn.decomposition.PCA(n_components=1))

# Step 2: Build Mapper graph
clusterer = cluster.DBSCAN(eps=0.5, min_samples=3)
graph = mapper.map(
    lens,
    X,
    clusterer=clusterer,
    cover=km.Cover(n_cubes=10, perc_overlap=0.5)
    # n_cubes: number of intervals in cover
    # perc_overlap: fraction of overlap between adjacent intervals (0.5 = 50%)
)

# graph is a dict with:
# graph['nodes'] = {'node_id': [point_indices], ...}
# graph['links'] = {'node_id': ['neighbor_node_id', ...], ...}
# graph['meta_nodes'] = {node_id: {cluster metadata}, ...}

n_nodes = len(graph['nodes'])
n_edges = sum(len(v) for v in graph['links'].values()) // 2
print(f"Mapper graph: {n_nodes} nodes, {n_edges} edges")

# Visualization (HTML):
html = mapper.visualize(graph, title="Mapper")
# mapper.visualize returns HTML string; write to file or display in browser

# Alternative lens: use PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
lens_pca = pca.fit_transform(X)
graph_pca = mapper.map(lens_pca, X, clusterer=clusterer,
                       cover=km.Cover(n_cubes=10, perc_overlap=0.5))
```

**kmapper Cover parameters**:
- `n_cubes`: number of intervals (more intervals = finer resolution, more nodes)
- `perc_overlap`: how much adjacent intervals overlap (0 = no overlap, 1 = full overlap)
  Higher overlap creates more connections. Typical: 0.3–0.5.

**TRAP: kmapper's `fit_transform` (lens computation) vs sklearn's `fit_transform`**.
`mapper.fit_transform(X)` projects X to a 1D lens using the specified projection.
It is NOT the same as calling a sklearn estimator directly. The return value is a 2D array
of shape `(n, 1)` not `(n,)` — pass the right shape to `mapper.map()`.

```python
# Correct:
lens = mapper.fit_transform(X, projection='l2norm')  # shape (200, 1)
graph = mapper.map(lens, X, ...)

# If using custom lens:
custom_lens = my_function(X).reshape(-1, 1)  # must be (n, 1)
graph = mapper.map(custom_lens, X, ...)
```

---

## Part 4: Euler Characteristic Curve

The Euler characteristic χ(ε) = β₀(ε) - β₁(ε) + β₂(ε) - ... as a function of ε.

```python
import gudhi
import numpy as np

X = np.random.randn(100, 2)

rips = gudhi.RipsComplex(points=X, max_edge_length=5.0)
st = rips.create_simplex_tree(max_dimension=2)
st.compute_persistence()

# Euler characteristic curve:
ec_curve = st.euler_characteristic_curve()
# Returns list of (filtration_value, euler_characteristic) pairs
# NOTE: gudhi 3.x API; may vary slightly

# Manual computation from Betti numbers:
# χ(ε) = Σ_k (-1)^k β_k(ε)
# At each filtration step, track running Betti numbers

# Alternative: from simplex tree directly (before persistence):
def euler_at_threshold(simplex_tree, eps):
    n_simplices = [0, 0, 0]  # count 0-, 1-, 2-simplices at threshold eps
    for simplex, filt_val in simplex_tree.get_filtration():
        if filt_val <= eps:
            dim = len(simplex) - 1
            if dim < 3:
                n_simplices[dim] += 1
    return n_simplices[0] - n_simplices[1] + n_simplices[2]
```

**Note on gudhi version**: `euler_characteristic_curve()` is available in gudhi >= 3.5.
For earlier versions, compute manually from persistence pairs.

---

## Tambear Decomposition

### The Algorithm Boundary

TDA has a sharp divide between GPU-amenable and CPU-necessary computation:

```
GPU side:
  1. Pairwise distances (F01 DistancePairs) — TiledEngine
  2. Sort pairwise distances (filtration order) — radix sort on GPU
  3. After persistence diagram is computed: statistics (F07/F25 reuse)

CPU side (inherently sequential):
  4. Incremental union-find for H₀ — O(N α(N))
  5. Boundary matrix reduction for H₁+ — O(N³) column reduction over GF(2)
  6. Optimal matching for diagram distances — O(N³) Hungarian
```

The Ripser algorithm is the state-of-the-art for step 5. It exploits specific structural
properties of the Vietoris-Rips boundary matrix (apparent pairs, clearing lemma) to
make column reduction sub-cubic in practice. It cannot be straightforwardly parallelized —
the clearing lemma depends on sequential column processing order.

### H₀: Union-Find on Sorted Edges

```
Tambear path for H₀:
1. D = DistancePairs(X)                    ← F01, TiledEngine GPU pass
2. edges = SortedPermutation(D, upper_triangle_only)  ← radix sort on GPU
3. union_find = UnionFind(N)               ← CPU data structure
4. for (i, j, d_ij) in edges:
     if union_find.find(i) != union_find.find(j):
         birth = d_ij
         prev_comp = union_find.find(j)
         death = d_ij
         // Record: merge component born at 0 (or prev birth), dies at death
         union_find.union(i, j)
5. Last surviving component: death = ∞
```

The sort is the expensive step: O(N² log N) on N² edge weights.
Union-find: O(N² α(N)) ≈ O(N²) — linear in number of edges.

GPU handles step 1 and 2. CPU handles steps 3-5 (~50 lines).

### H₁: Boundary Matrix Reduction

```
1. Enumerate 1-simplices (edges): at each threshold, edges are sorted distances
2. Enumerate 2-simplices (triangles): {i,j,k} appears at max(d_ij, d_ik, d_jk)
   → triangle count: O(N³) in worst case
3. Build boundary matrix ∂₁ (N_edges × N_vertices) and ∂₂ (N_triangles × N_edges)
   over GF(2) — boundary of each simplex = alternating sum of faces mod 2
4. Column-reduce ∂₂ using Gaussian elimination over GF(2) (pivot algorithm)
5. Persistence pairs: pivot columns → (birth, death) of H₁ features

Clearing lemma: if column k has lowest pivot row r, mark row r as "cleared"
— column r need not be reduced (saves ~half the work).
Apparent pairs: edge (birth) killed by the triangle containing it as max face (instant death).
```

This is what Ripser implements. For tambear: **delegate to ripser/gudhi via FFI or subprocess**.
Do not reimplement the boundary matrix reduction. The value is in the pipeline, not the algorithm.

### Persistence Statistics: Back to GPU

Once persistence diagram is in memory (N_features × 2 float array):

```
lifetimes = death - birth          ← element-wise subtract
L = sum(lifetimes)                 ← accumulate(All, Identity, Add)
p = lifetimes / L                  ← element-wise divide
H = -sum(p * log(p))               ← accumulate(All, -p*log(p), Add)  ← F25 Shannon entropy!
max_life = max(lifetimes)          ← accumulate(All, Identity, Max)
```

**The persistence statistics ARE F07/F25 statistics applied to the lifetime vector.**
After ripser produces the diagram, the tambear pipeline handles everything downstream.

### Diagram Distance: CPU Hungarian

Wasserstein and bottleneck between diagrams require optimal matching:

```
M[i,j] = || (b_i, d_i) - (b'_j, d'_j) ||_p   ← cost matrix (N_feat × N_feat)
Optimal matching: Hungarian algorithm O(N³) or Auction O(N² log N)
Wasserstein-p = (Σ M[i,η(i)]^p)^{1/p}         ← matched pair costs
Bottleneck = max_k M[k, η(k)]                   ← max matched pair
```

For N_features < 500: CPU Hungarian is fine (< 1ms).
For N_features > 1000: consider sliced Wasserstein approximation (O(N log N) per slice).

---

## Infrastructure Requirements

| Gap | Needed by | Severity |
|-----|-----------|---------|
| Radix sort on GPU (for edge filtration order) | H₀, H₁ | Medium — can use CPU std::sort for N < 10k |
| Union-Find CPU data structure | H₀ | Low — ~30 lines |
| ripser FFI or subprocess bridge | H₁+ | High — don't reimplement |
| Hungarian algorithm CPU | Diagram distances | Low — ~100 lines or use scipy.optimize.linear_sum_assignment |
| Persistence entropy: F25 reuse | Persistence stats | Free — already built |
| BarcodeStats type | All TDA outputs | Low — ~50 lines struct |

---

## Complete Validation Suite

```python
"""
F27 Gold Standard Validation Suite
Run this to generate reference values for tambear F27 implementation.
"""

import numpy as np
from ripser import ripser
import persim

np.random.seed(42)

# ─── Dataset 1: Circle (1 loop expected) ───────────────────────────────
theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
X_circle = np.column_stack([np.cos(theta), np.sin(theta)])
X_circle += np.random.randn(*X_circle.shape) * 0.05

r_circle = ripser(X_circle)
dgms_circle = r_circle['dgms']

print("=== Circle ===")
print(f"  β₀ = {np.sum(np.isinf(dgms_circle[0][:, 1]))}")  # expect 1
print(f"  β₁ = {np.sum(np.isinf(dgms_circle[1][:, 1]))}")  # expect 1

h1_fin = dgms_circle[1][np.isfinite(dgms_circle[1][:, 1])]
if len(h1_fin) > 0:
    best_h1 = h1_fin[np.argmax(h1_fin[:, 1] - h1_fin[:, 0])]
    print(f"  Best finite H1: birth={best_h1[0]:.4f}, death={best_h1[1]:.4f}")

# ─── Dataset 2: Two clusters (2 components in H₀) ─────────────────────
X_two = np.vstack([
    np.random.randn(50, 2) * 0.2 + [0, 0],
    np.random.randn(50, 2) * 0.2 + [5, 0],
])
r_two = ripser(X_two)
dgms_two = r_two['dgms']

print("\n=== Two Clusters ===")
print(f"  β₀ = {np.sum(np.isinf(dgms_two[0][:, 1]))}")  # expect 1 (final merge)
# H₀ has N-1 finite features plus 1 essential; one merge around dist≈4
h0_deaths = dgms_two[0][np.isfinite(dgms_two[0][:, 1]), 1]
h0_deaths_sorted = np.sort(h0_deaths)
print(f"  Largest H₀ death (cluster merge): {h0_deaths_sorted[-1]:.4f}")  # expect ≈4.0–5.0
print(f"  Total H₀ features: {len(dgms_two[0])}")  # expect 100 (one per point, all born at 0)

# ─── Dataset 3: Two circles (2 loops expected) ────────────────────────
theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
X_two_circles = np.vstack([
    np.column_stack([np.cos(theta), np.sin(theta)]),
    np.column_stack([np.cos(theta) + 3.0, np.sin(theta)]),
])
X_two_circles += np.random.randn(*X_two_circles.shape) * 0.03

r_tc = ripser(X_two_circles, maxdim=1)
dgms_tc = r_tc['dgms']

print("\n=== Two Circles ===")
print(f"  β₀ = {np.sum(np.isinf(dgms_tc[0][:, 1]))}")  # expect 1
print(f"  β₁ = {np.sum(np.isinf(dgms_tc[1][:, 1]))}")  # expect 2 (two loops)

# ─── Bottleneck Distance: same vs different ───────────────────────────
np.random.seed(1)
X_noise1 = np.random.rand(50, 2)
X_noise2 = np.random.rand(50, 2)
dgms_n1 = ripser(X_noise1)['dgms']
dgms_n2 = ripser(X_noise2)['dgms']

d_bn_same = persim.bottleneck(dgms_n1[1], dgms_n1[1])
d_bn_diff = persim.bottleneck(dgms_n1[1], dgms_n2[1])
print("\n=== Bottleneck Distance ===")
print(f"  dB(D, D) = {d_bn_same:.6f}")      # expect 0.0
print(f"  dB(D1, D2) = {d_bn_diff:.6f}")    # expect > 0

# ─── Persistence Entropy ──────────────────────────────────────────────
def persistence_entropy(dgm):
    fin = dgm[np.isfinite(dgm[:, 1])]
    if len(fin) == 0:
        return 0.0
    lifetimes = fin[:, 1] - fin[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    p = lifetimes / lifetimes.sum()
    return float(-np.sum(p * np.log(p)))

print("\n=== Persistence Entropy ===")
print(f"  H1 entropy (circle): {persistence_entropy(dgms_circle[1]):.6f}")
print(f"  H1 entropy (noise):  {persistence_entropy(dgms_n1[1]):.6f}")
# Expect: circle lower (few features), noise higher (many uniform features)

# ─── From distance matrix (tambear integration path) ─────────────────
from scipy.spatial.distance import squareform, pdist
D = squareform(pdist(X_circle))  # full symmetric distance matrix

r_dm = ripser(D, distance_matrix=True)
dgms_dm = r_dm['dgms']
print("\n=== Point cloud vs distance matrix (should match) ===")
h0_mismatch = not np.allclose(
    np.sort(dgms_circle[0][:, 1]),
    np.sort(dgms_dm[0][:, 1])
)
print(f"  H₀ diagrams match: {not h0_mismatch}")  # expect True
```

---

## Key Traps Summary

| Trap | Consequence | Mitigation |
|------|------------|-----------|
| `ripser` default `maxdim=1` | H₂+ silently not computed | Always specify `maxdim` explicitly |
| Infinite deaths as `np.inf` | Betti number count breaks with `dgm[:, 1] > threshold` comparisons | Always check `np.isinf(dgm[:, 1])` for essential features |
| `persim.wasserstein` is diagram Wasserstein | Confused with distribution Wasserstein (`scipy.stats.wasserstein_distance`) | These are different quantities; never mix |
| `gudhi.RipsComplex` `max_edge_length` truncates diagram | Features with large death appear essential | Set `max_edge_length=np.inf` for full diagram; filter post-hoc |
| Rips simplex count is O(2^N) | `maxdim=3+` explodes on N>200 | For H₂+: use Alpha complex or limit thresh |
| H₀ births all 0 for Rips on point clouds | Looks like all features born simultaneously | This is correct — all points appear at ε=0 in Rips |
| Mapper `perc_overlap=0` disconnects graph | All nodes isolated even if nearby | Use 0.3–0.5; never 0 |
| kmapper lens shape | `mapper.map()` expects `(n, 1)` not `(n,)` | Always `.reshape(-1, 1)` for custom lenses |
| Persistence entropy direction | Circle (structured) has LOW entropy, not high | Concentrated lifetime distribution → low entropy |
| Alpha vs Rips diagrams differ slightly | Not a bug — different filtrations, same H₀/H₁ for small noise | For validation: use same library on same input |

---

## Library Versions (for reproducibility)

```
ripser >= 0.6.4      (Python)
gudhi >= 3.5.0       (Python; euler_characteristic_curve requires 3.5+)
persim >= 0.3.1      (Python)
kmapper >= 2.0.1     (Python)
TDA >= 1.9           (R)
TDAstats >= 0.4.1    (R)
```

All Python libraries are pip-installable. `gudhi` has system dependencies (CGAL)
that can complicate installation on Windows — use conda or the pre-built wheel from
conda-forge: `conda install -c conda-forge gudhi`.

---

## MSR Types F27 Produces

```rust
pub struct PersistenceDiagram {
    pub dimension: u8,              // 0 for H₀, 1 for H₁, etc.
    pub pairs: Vec<(f32, f32)>,     // (birth_ε, death_ε); death=f32::INFINITY for essential
    pub n_essential: usize,         // count of essential features = Betti number
}

pub struct BarcodeStats {
    pub betti: Vec<usize>,          // [β₀, β₁, β₂, ...] per dimension
    pub total_persistence: Vec<f64>, // Σ (death - birth) per dimension (finite only)
    pub entropy: Vec<f64>,           // persistence entropy per dimension
    pub max_persistence: Vec<f64>,   // max single feature lifespan per dimension
    pub mean_persistence: Vec<f64>,  // mean lifespan per dimension
}

pub struct MapperGraph {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub node_sizes: Vec<usize>,         // number of points per node
    pub edges: Vec<(usize, usize)>,     // (node_i, node_j) adjacency
    pub filter_mean: Vec<f64>,          // mean filter value per node
}
```

BarcodeStats computation after ripser → tambear: all operations are F25 (information theory)
and F07 (statistics) applied to the lifetime vector. No new primitives needed.
