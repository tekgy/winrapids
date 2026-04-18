<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

# Clustering — Complete Variant Catalog

## What Exists (tambear::clustering + related)

### Partitional
- K-means via `clustering_from_distance` with kmeans-style iteration
- Union-find infrastructure (`uf_new`, `uf_find`, `uf_union`)

### Hierarchical
- `hierarchical_clustering(data, linkage, n_clusters)` — single/complete/average/ward/centroid/median

### Validation
- `calinski_harabasz_score` — variance ratio
- `davies_bouldin_score` — cluster similarity
- `silhouette_score` — separation vs cohesion
- `cluster_validation` — combined report
- `hopkins_statistic` — clustering tendency
- `gap_statistic` — compare to null reference
- `bic_score` / `aic_score` — information criterion

### In tambear::mixture
- GMM (Gaussian Mixture Model) — EM algorithm

### In tambear::kmeans
- K-means implementation

### In tambear::spectral_clustering
- Spectral clustering

---

## What's MISSING — Complete Catalog

### A. Density-Based Methods

1. **DBSCAN** — Ester et al. 1996
   - Core point if ≥ minPts neighbors within ε
   - Parameters: `data`, `eps`, `min_pts`
   - O(n log n) with spatial index, O(n²) naive
   - Already partially exists? Need to verify in clustering.rs
   - Primitives: distance_matrix → neighborhood_count → union_find

2. **HDBSCAN** — Campello et al. 2013
   - Hierarchical DBSCAN: varies ε over all scales
   - Mutual reachability distance → MST → condensed tree → extract
   - Parameters: `data`, `min_cluster_size`, `min_samples`
   - Output: labels + cluster persistence
   - Primitives: k-NN distances → mutual reachability → MST → condensed tree

3. **OPTICS** — Ankerst et al. 1999
   - Ordering by reachability distance
   - Produces reachability plot → extract clusters at any ε
   - Parameters: `data`, `min_pts`, `max_eps`
   - Primitives: k-NN queries, priority queue

4. **Mean shift** — Comaniciu & Meer 2002
   - Each point moves to kernel-weighted centroid until convergence
   - Parameters: `data`, `bandwidth`, `max_iter`, `tol`
   - Primitives: KDE gradient → iterative ascent
   - No need to specify k; discovers number of clusters

### B. Partitional Variants

5. **K-medoids (PAM)** — Kaufman & Rousseeuw 1987
   - Uses actual data points as centers (medoids), more robust than k-means
   - Parameters: `data`, `k`, `max_iter`
   - Objective: minimize Σ d(xᵢ, medoid_j)
   - O(n²k) per iteration

6. **K-medoids (CLARA)** — Kaufman & Rousseeuw 1990
   - Sampling-based approximation to PAM for large datasets
   - Parameters: `data`, `k`, `n_samples`, `sample_size`

7. **K-medoids (CLARANS)** — Ng & Han 2002
   - Randomized local search improvement
   - Parameters: `data`, `k`, `num_local`, `max_neighbor`

8. **Mini-batch K-means** — Sculley 2010
   - Updates centers using random mini-batches
   - Parameters: `data`, `k`, `batch_size`, `max_iter`
   - O(batch × k × d) per step — scalable

9. **K-means++** — Arthur & Vassilvitskii 2007
   - Better initialization: probabilistic seeding proportional to D²
   - Parameters: `data`, `k`, `seed`
   - O(nk) initialization, produces O(log k)-competitive solution
   - Should be default initialization for all k-means variants

10. **Bisecting K-means** — Steinbach et al. 2000
    - Top-down: recursively split worst cluster
    - Parameters: `data`, `k`, `max_iter`
    - Produces hierarchical partition

11. **X-means** — Pelleg & Moore 2000
    - Automatically determines k via BIC splitting
    - Parameters: `data`, `k_min`, `k_max`

12. **Fuzzy C-means** — Bezdek 1981
    - Soft clustering: each point has membership degree in each cluster
    - u_ij = 1 / Σ_k (d_ij/d_ik)^{2/(m-1)}
    - Parameters: `data`, `k`, `m` (fuzziness, default 2), `max_iter`, `tol`
    - Output: membership matrix u (n × k) + centers

### C. Model-Based

13. **Bayesian GMM** (DPGMM / VB-GMM) — Blei & Jordan 2006
    - Variational Bayes: automatic component selection
    - Parameters: `data`, `max_components`, `max_iter`, `tol`
    - No need to specify k; discovers it

14. **Student-t mixture model** — Peel & McLachlan 2000
    - Robust to outliers (heavier tails than Gaussian)
    - Parameters: `data`, `k`, `max_iter`, `tol`
    - EM with additional df parameter per component

### D. Subspace / Spectral

15. **Spectral clustering** — already exists
    - Missing variants: different similarity graphs (ε-neighborhood, k-NN, fully connected)
    - Missing: different Laplacians (unnormalized, Shi-Malik, Ng-Jordan-Weiss)
    - Parameters: `data`, `k`, `graph_type`, `laplacian_type`

16. **Subspace clustering** — Parsons et al. 2004
    - Clusters in different subspaces of the feature space
    - CLIQUE, SUBCLU, PROCLUS variants
    - Parameters: `data`, `k`, `n_subspace_dims`

17. **Self-Organizing Map** (SOM) — Kohonen 1982
    - Topology-preserving mapping to 2D grid
    - Parameters: `data`, `grid_x`, `grid_y`, `n_iter`, `learning_rate`

### E. Graph-Based

18. **Louvain** — Blondel et al. 2008
    - Community detection in graphs via modularity optimization
    - Parameters: `adjacency`, `resolution`
    - O(n log n) — very scalable

19. **Leiden** — Traag et al. 2019
    - Improved Louvain: guarantees well-connected communities
    - Parameters: `adjacency`, `resolution`

20. **Affinity Propagation** — Frey & Dueck 2007
    - Message-passing between data points
    - No need to specify k
    - Parameters: `similarity_matrix`, `damping`, `max_iter`, `preference`

21. **Label Propagation** — Raghavan et al. 2007
    - Each node adopts majority label of neighbors
    - Parameters: `adjacency`, `max_iter`
    - Very fast, non-deterministic

### F. Clustering for Specific Data Types

22. **Time series clustering** — using DTW or shape-based distance
    - K-means with DTW barycentric averaging
    - Parameters: `series`, `k`, `distance` ("dtw"|"euclidean"|"sbd")

23. **Consensus clustering** — Monti et al. 2003
    - Run clustering multiple times → consensus matrix → final clustering
    - Parameters: `data`, `k`, `n_runs`, `proportion`

### G. Missing Validation Metrics

24. **Dunn Index** — ratio of min inter-cluster to max intra-cluster distance
    - Parameters: `data`, `labels`

25. **Connectivity** — proportion of k-NN in same cluster
    - Parameters: `data`, `labels`, `k`

26. **Jaccard index** for cluster stability
    - Bootstrap → re-cluster → Jaccard between original and bootstrap clusters

27. **V-measure** — Rosenberg & Hirschberg 2007
    - Harmonic mean of homogeneity and completeness
    - Parameters: `labels_true`, `labels_pred`

28. **Fowlkes-Mallows Index** — geometric mean of precision and recall
    - Parameters: `labels_true`, `labels_pred`

---

## Decomposition into Primitives

```
distance_matrix(data) ──┬── k_medoids (PAM)
                        ├── dbscan
                        ├── optics
                        ├── hierarchical (all linkages)
                        ├── affinity_propagation
                        └── silhouette_score

k_nn(data, k) ─────────┬── hdbscan (mutual reachability)
                        ├── spectral (k-NN graph)
                        ├── mean_shift (neighbors)
                        └── connectivity (validation)

covariance_matrix ──────┬── gmm (E-step)
                        ├── bayesian_gmm
                        └── mahalanobis

eigendecomposition ─────── spectral clustering (graph Laplacian)

union_find ─────────────┬── dbscan
                        ├── hdbscan
                        └── mst-based clustering

mst(data) ──────────────┬── hdbscan
                        └── single-linkage hierarchical

kmeans_plus_plus_init ──── all k-means variants
```

## Priority

**Tier 1** — Most commonly needed:
1. `dbscan(data, eps, min_pts)` — fundamental density-based
2. `k_medoids(data, k)` — robust partitional
3. `kmeans_plus_plus_init` — should be default for all k-means
4. `fuzzy_c_means(data, k, m)` — soft clustering
5. `hdbscan(data, min_cluster_size)` — parameter-free density-based

**Tier 2**:
6. `mean_shift(data, bandwidth)` — discovers k
7. `mini_batch_kmeans(data, k, batch_size)` — scalable
8. `affinity_propagation` — no k needed
9. `louvain` / `leiden` — graph community detection
10. `bayesian_gmm` — automatic component selection

**Tier 3**:
11-28: OPTICS, CLARA/CLARANS, SOM, consensus, subspace, etc.


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

