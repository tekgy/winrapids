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

