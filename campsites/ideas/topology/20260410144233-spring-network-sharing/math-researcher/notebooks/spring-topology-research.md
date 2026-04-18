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

