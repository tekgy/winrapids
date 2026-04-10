# Tropical Semiring: Graph Atoms Already There, Kingdom B Is Small

*Scout, 2026-04-10 (from naturalist's message during hold)*

## Bellman-Ford is already tropical in graph.rs

`graph.rs:248` — `pub fn bellman_ford(g, source)` — iterative relaxation:
`dist[v] = min(dist[v], dist[u] + weight)` over all edges.

This IS:
```
accumulate(edges, Graph, dist[u] + weight, Op::TropicalMinPlus)
```

Tropical SpMV. Graph × TropicalMinPlus. Already in the codebase, unnamed as a
tropical operation.

Similarly:
- `graph.rs:212` — `dijkstra` — greedy tropical (priority-queue-accelerated version)
- `graph.rs:282` — `floyd_warshall` — tropical closure (iterated tropical matrix multiply)

All three shortest-path algorithms are tropical Graph operations. The framework already
classifies them correctly. The label was missing, not the structure.

---

## The eigenbasis table gains a semiring column

From the naturalist's one-algorithm-five-transforms framework:

| Atom | Standard semiring | Tropical semiring |
|------|-------------------|-------------------|
| All | Sum (tree reduction) | Min (parallel min) |
| Prefix | Blelloch scan | Tropical prefix scan (shortest paths) |
| Graph | SpMV | Tropical SpMV (Bellman-Ford, Floyd-Warshall) |

Same grouping atoms. Same eigenbasis transforms. Different algebraic ground.
The semiring is a parameter, not a structure. The implementation cost does NOT
double — the grouping kernel is the same, only the combine function changes:

```
accumulate(data, Prefix, expr, Op::Add)          // standard
accumulate(data, Prefix, expr, Op::TropicalMinPlus)  // tropical, same kernel
```

---

## The compression: Kingdom B is small

What appeared to be Kingdom A/B with a fuzzy boundary is actually:

**Kingdom A (data-determined, standard semiring):**
GARCH, EMA, EWMA, AR(p), Kalman, HMM forward, Kaplan-Meier, LogSumExp

**Kingdom A (data-determined, tropical semiring):**
Viterbi decoding, Bellman-Ford, Dijkstra, Floyd-Warshall, DTW,
PELT underlying DP recurrence (O(n²) work, O(log n) depth on GPU)

**Genuine Kingdom B (self-referential maps only):**
ARMA MA terms CSS implementation (but Kalman formulation = Kingdom A math),
BOCPD (transition matrix grows with t),
EGARCH (z_t = r_t/σ_t couples map to state),
TAR (branch selection is state-dependent),
MCMC (proposal and acceptance are state-dependent),
PELT pruning optimization (uses f[t] to adapt candidate set — adds Kingdom B for O(n) CPU)

**PELT GPU/CPU trade-off** (r-gap-scan): the pruning that gives PELT O(n) complexity
is a Kingdom B element added to a Kingdom A computation. Different hardware wants
different regimes:
- GPU: accept O(n²) work, get O(log n) depth — use tropical Kingdom A prefix scan
- CPU: accept sequential, get O(n) work — use PELT pruning (Kingdom B structure)

Same algorithm, different execution regime for different targets. TAM can choose
based on hardware: tropical scan on GPU, PELT pruning on CPU.

**The Fock boundary compresses:**
- Old statement: "sequential vs parallel"
- New statement: "self-referential vs data-determined" (plus: finitely-representable semigroup)

Self-referential = genuinely small. Data-determined + finitely-representable semigroup =
most of statistics and dynamic programming, parallelizable once the correct semiring
is identified. The "finitely-representable" condition catches floor-affine, logistic map,
and scaled sine — data-determined maps that still fail to be Kingdom A.

---

## Annotation needed in graph.rs

The three tropical operations in graph.rs should be annotated:

```rust
// Kingdom A: tropical SpMV = accumulate(edges, Graph, dist[u]+weight, TropicalMinPlus)
pub fn bellman_ford(g: &Graph, source: usize) -> Vec<f64> { ... }

// Kingdom A: greedy tropical (priority-queue accelerated Bellman-Ford)
pub fn dijkstra(g: &Graph, source: usize) -> Vec<f64> { ... }

// Kingdom A: tropical closure = iterated tropical matrix multiply
pub fn floyd_warshall(g: &Graph) -> Vec<Vec<f64>> { ... }
```

These annotations connect the existing implementations to the tropical framework
and make their Kingdom A status visible without needing to analyze the algorithm.
