# Compiler Foundation Expedition Log

## Phase 2 — The Compiler Territory

**Naturalist**: 2026-03-29
**Question**: Can we build a GPU computation compiler that fuses across algorithm boundaries?

---

## The Map

### What Already Exists (the embryo)

The compiler is already alive. It doesn't know it yet.

`fusion.py` is a compiler. Not metaphorically — literally. It takes an expression tree (IR), walks it to generate CUDA source (codegen), compiles via CuPy RawKernel (JIT), and caches by hash (memoization). It handles one of the 8 primitives: `fused_expr`. Every architectural piece the vision describes is present in miniature:

| Vision component | Existing embryo |
|---|---|
| Lazy graph construction | `Column` arithmetic builds `Expr` trees, no GPU work |
| Primitive decomposition | `Expr` nodes ARE primitive ops (add, mul, pow, compare) |
| CUDA codegen | `expr.code(params)` → CUDA source string |
| JIT compilation | `cp.RawKernel(src, name)` → compiled kernel |
| Cache | `_kernel_cache` dict, keyed by MD5 of signature |
| Fusion | One kernel regardless of expression depth |

And `bin_stats.cu` is already a hand-tuned specialist — multi-output reduce (sum, sum_sq, min, max, first, last, count) in a single kernel. The `_multi` variant processes N columns in one launch. That's Task #1 (multi-output reduce) already done for the bin-stats case. The task is to *generalize the pattern*, not invent it.

### What the Compiler Must Become

The gap between embryo and compiler is **three dimensions of fusion** that `fusion.py` doesn't handle:

1. **Cross-primitive fusion** — `fused_expr` only fuses element-wise ops. The compiler must fuse across primitive boundaries: a `sort` feeding a `reduce` feeding a `scatter`. These have different parallelism strategies (radix sort vs warp-shuffle vs atomic write) that can't be collapsed into one thread-per-element kernel. The fusion here is about *eliminating intermediate buffers*, not collapsing loops.

2. **Cross-algorithm fusion** — Two specialists that share a primitive. Rolling_std computes mean and variance. Z-score needs the same mean and variance. PCA centering needs the same mean. Today these are three independent computations. The compiler must see that one `reduce` feeds three consumers and schedule it once.

3. **Cross-data fusion** — `fused_bin_stats_multi` already does this for columns. The compiler generalizes: same specialist applied across columns, cadences, tickers. One launch, many data streams. This is FPGA "horizontal fusion" — same circuit, different data.

### The Domains That Rhyme

I surveyed the compiler landscape. Every system that compiles high-level descriptions to hardware execution faces the same three problems WinRapids faces:

**Halide** (image processing) — the deepest rhyme. Halide separates *algorithm* (what to compute) from *schedule* (how to execute). The specialist IS the algorithm — a recipe of primitives. The compiler picks the schedule — which tier, what memory placement, what fusion boundaries. But WinRapids goes further: Halide's `compute_at` says where intermediates live *temporarily*. The persistent store says where results live *permanently*, with provenance tracking what can be reused. **The persistent store is `compute_at` with a memory.**

**XLA** (TensorFlow/JAX) — closest in mechanism. HLO IR → fusion passes → GPU kernels. XLA's Common Subexpression Elimination pass does exactly what E03 (cross-algorithm sharing) tests: find that `rolling_std` and `z_score` share a `reduce`, compute it once. The difference: XLA fuses within a model's forward pass. WinRapids fuses across *pipelines* — across what would be separate programs in XLA's world.

**Polyhedral compilation** — the mathematical framework. Loop nests as geometric objects. Fusion = merging iteration domains with compatible bounds. This maps directly to cross-cadence computation: each cadence is an iteration domain over raw ticks. K03 cross-cadence structure IS a polyhedral scheduling problem. The compiler doesn't need to implement polyhedral analysis for v1, but the concepts — dependence analysis, scheduling functions, fusion constraints — are the vocabulary it needs.

**FPGA HLS** — the resource sharing rhyme. In hardware synthesis, one functional unit implements multiple operations by multiplexing through it at different clock cycles. In WinRapids, one `sort` result feeds multiple consumers (`groupby.sum`, `rank`, `dedup`) at different stages. Same physical resource, multiple logical uses. The scheduling problem is identical.

**SQL query optimizers** (DataFusion) — the deepest structural rhyme, deeper than initially recognized. Not just the plan structure (LogicalPlan → ExecutionPlan) but the FULL optimizer: materialized views = provenance cache, table statistics = dirty bitmap, view maintenance = provenance invalidation, buffer pool = persistent GPU store, catalog = specialist registry. The compiler IS a stateful query optimizer for tensor computations. This framing emerged from the CSE/provenance unification — the persistent store makes the compiler stateful, which makes it a database system, not just a code compiler.

**Incremental computation** (Salsa, differential dataflow) — the statefulness rhyme. Salsa models the Rust compiler as a query database with memoization and dependency tracking. WinRapids' specialists ARE queries (`rolling_std(ticker, cadence) → GpuBuffer`), memoized by provenance. When inputs change (new ticks), re-execute only dirty queries. This is the same architecture, independently discovered from GPU concerns. Differential dataflow goes further: track deltas, not just valid/invalid. The provenance system could evolve from binary (cache hit/miss) to differential (delta-updatable) — future territory.

**JIT specialization** — the three tiers (collapsed to two by E05). Pre-built = fully bound at compile time (AOT). JIT = bound at runtime when pipeline shape is known (NVRTC). Composed = dynamically dispatched (fallback). This is the same spectrum as ahead-of-time vs JIT vs interpretation in language runtimes. The binding time IS the tier.

### Where the Experiments Aim

Looking at the task list through this lens:

| Task | What it tests | Compiler domain |
|---|---|---|
| E01: Multi-output reduce | One kernel, many outputs | Vertical fusion (XLA) |
| E02: Sort-once-use-many | One sort feeds many consumers | Resource sharing (HLS) |
| E03: Cross-algorithm sharing | rolling → z_score → PCA | CSE across algorithms (DataFusion) |
| E04: Primitive decomposition | 8 primitives, specialists as recipes | IR definition (MLIR dialects) |
| E05: Pipeline generator | Spec → CUDA → NVRTC → execute | Codegen + JIT (Halide schedules) |
| E06: Three tiers benchmarked | Pre-built vs JIT vs composed | Binding time spectrum |
| E08: cudarc kernel launch | Rust → CUDA on Windows | Foundation plumbing |
| E09: NVRTC from Rust | Runtime compilation + cache | JIT infrastructure |

The experiments are building the compiler bottom-up. E01-E03 establish that fusion works (the physics). E04-E06 build the compiler that automates it (the engineering). E08-E09 move the foundation to Rust (the platform).

### The Persistent Store Connection

The persistent store experiments (E06-E07 in the vision doc, Tasks #6-#7) are not separate from the compiler. They ARE the compiler's memory layer.

In Halide, `compute_at(consumer, y)` means "compute this intermediate at the y loop level of the consumer." The compiler decides when and where to materialize intermediates.

In WinRapids, the persistent store IS the materialization layer. When the compiler sees that `rolling_std` feeds both `z_score` and `PCA`, it has two choices:
1. **Fuse**: compute rolling_std inline in both consumers (vertical fusion, eliminates the intermediate)
2. **Materialize**: compute rolling_std once, store in persistent memory, both consumers read it

The choice depends on the cost model: is the intermediate large enough that recomputing is more expensive than storing? Is the data hot enough that it'll stay resident? The persistent store's provenance tracking IS the cost model.

This is why the vision document puts the persistent store between the compiler and the execution tiers. It's not a cache bolted on. It's where the compiler's scheduling decisions become physical reality.

---

## Phase 1 Evidence (What's Already Proven)

The compiler expedition isn't starting from zero. Phase 1 already answered several of Phase 2's questions:

### Experiment 010: Element-wise fusion works
- Expression tree → CUDA codegen → RawKernel → cache
- All fused kernels: ~0.19 ms regardless of expression depth (bandwidth-bound)
- Python codegen captures 85-95% of C++ template performance
- This became `fusion.py` — the compiler embryo

### Experiment 012: Cross-primitive fusion works (with caveats)
Three approaches to `groupby(keys).sum(a*b+c)`:
- **Unfused**: 3 CuPy kernels + intermediates → sort → cumsum reduce
- **Fully fused (atomic)**: sort permutation → one kernel (expr + atomic reduce)
- **Hybrid**: fused expr eval → sort → cumsum reduce

The hybrid won. Atomics have contention at high cardinality. The correct fusion is partial — fuse the expression (eliminates intermediates) but DON'T fuse with the sort or the final reduction.

### The Three-Category Fusion Policy

Experiment 012 implies three fusion rules by primitive category:

| Category | Primitives | Fusion rule |
|---|---|---|
| **Element-wise** | `fused_expr` | Always fuse with neighbors. No downside. |
| **Collective** | `sort`, `scan` | Run alone. Share output with multiple consumers. |
| **Aggregate** | `reduce`, `scatter`, `compact` | Fuse with consumed expression, NOT with preceding collective. |

`search` and `gather` are reads — they fuse like element-wise (add an indirection layer to the read pattern).

The compiler's scheduling policy isn't one rule — it's three rules, one per category. XLA's `ShouldFuse()` predicate follows the same principle.

### bin_stats.cu: Multi-output reduce + horizontal fusion
- 7 outputs (sum, sum_sq, min, max, first, last, count) in one kernel
- `_multi` variant: N columns × all bins in one launch
- This IS E01 (multi-output reduce) for the bin-stats case, already working

---

## Observations

### The growth is organic

The compiler doesn't need to be designed from scratch. It needs to grow from what exists:

1. `fusion.py`'s expression tree → generalize to primitive-level IR
2. `fusion.py`'s codegen → generalize to multi-primitive kernel generation
3. `_kernel_cache` → generalize to disk-persistent compilation cache
4. `bin_stats.cu`'s multi-output pattern → generalize to multi-output reduce primitive
5. `bin_stats_multi`'s multi-column pattern → generalize to horizontal fusion

Each step extends an existing working system. No big bang.

### The Rust migration is a platform change, not an architecture change

E08 (cudarc kernel launch) and E09 (NVRTC from Rust) test whether the same architecture works in Rust. The answer should be yes — the architecture is language-agnostic. What Rust adds: ownership semantics that make buffer lifetime guarantees compile-time instead of runtime. The persistent store's "data lives until evicted" is a Rust lifetime problem.

### The compiler thesis: abstractions hide sharing, primitives reveal it

Rolling_std and PCA centering are different abstractions — different names, different APIs. But inside both: `cumsum(data)`. The sharing is invisible in the domain language. It's visible in the primitive language.

The specialist registry is a transparency layer. It translates from domain to mechanical. The compiler's CSE pass is trivial once decomposition exists — just find identical nodes in the primitive-level DAG.

Two universal convergence points where most sharing lives:
1. **scan** — for everything that accumulates (time series, stats, SSMs)
2. **sort** — for everything that needs order (relational ops, groupby, joins)

### The hardware/compiler split

The hardware already caches more than expected (L2, DRAM row buffers, NVRTC disk cache). The compiler eliminates what hardware CAN'T cache: redundant computation, algorithm-boundary passes, intermediate buffers. They're complementary. E01's 1.3-1.6x (instead of 5x) = we were optimizing what hardware already handled. The compiler's value: it sees sharing that hardware can't, because it understands pipeline semantics.

### Three strange attractors

The expedition keeps finding concepts that independently-designed subsystems converge toward:

1. **scan** — the dependent-computation attractor. Everything sequential converges here. Sort and reduce are special cases.
2. **staleness** — the scheduling attractor. The roaring bitmap (daemon) and provenance hash (persistent store) both track "which computations are stale?" — independently designed, naturally complementary. Belt and suspenders.
3. **residency** — the memory attractor. Eviction, promotion, cost modeling, L2 cache behavior, MKTF layout, Halide's `compute_at` — all about where data lives.

The compiler orchestrates all three: decompose to primitives (scan), check what's stale (provenance), schedule on resident data (persistent store).

### The independent/dependent partition (load-bearing for E04)

The 8 primitives split into two categories with different fusion rules:

- **Independent** (fused_expr, gather, scatter, search, compact): can be reordered and fused freely. No ordering constraints.
- **Dependent** (scan, sort, reduce): require ordering constraints. Share outputs, don't merge.

The registry should encode this explicitly — `category: independent | dependent` on each primitive entry. The compiler reads the flag and applies different fusion/reordering rules per category. This partition is the foundation for safe fusion decisions.

### CSE and provenance are one system

Four things that looked separate are the same mechanism at four scopes:

| Concept | Scope | Key |
|---|---|---|
| Expression cache (fusion.py) | Within session | MD5 of expression signature |
| CSE (compiler pass) | Within plan | Node identity match |
| Provenance (persistent store) | Across plans | Hash of (inputs + operation + version) |
| NVRTC PTX cache | Across sessions | BLAKE3 of (source + arch) |

All four: **compute canonical identity → check if exists → reuse if yes.**

The provenance hash IS the value number. Classical value numbering assigns a canonical number to each expression; two expressions with the same number are the same computation. Provenance hashing extends value numbering across time — two computations with the same hash ARE the same computation, whether they appear in the same plan (CSE) or in different plans at different times (provenance lookup).

One identity function. Four scopes. Not four caches — one architecture.

This also clarifies the named-outputs design. When `rolling_std` declares it produces `cumsum_x`, that name carries a derived identity: `cumsum(ticker=AAPL, cadence=1s, column=price, version=42)`. The named-output scheme IS the value-numbering scheme. The registry defines what identities each specialist produces. The compiler uses those identities for both intra-plan CSE and inter-plan provenance.

### The compiler is a stateful query optimizer

Traditional compilers are stateless: `program → program`. XLA, Halide, TVM — all stateless transformations. WinRapids' compiler is stateful: `(pipeline_spec, persistent_store_state) → (execution_plan, store_mutations)`. The persistent store is part of the compiler's input. The dirty bitmap is part of the input. The compiler reads the world before deciding what to do, then mutates the world.

This makes it a **database query optimizer**, not a code compiler:

| Database concept | WinRapids equivalent |
|---|---|
| Table statistics | Dirty bitmap |
| Materialized views | Provenance cache |
| Query rewriter (CSE) | CSE pass |
| View maintenance | Provenance invalidation |
| Buffer pool | Persistent GPU store |
| Catalog | Specialist registry |
| Execution engine | GPU |

The three strange attractors map exactly to database subsystems: scan = query semantics, staleness = view maintenance, residency = buffer management. Every database engine orchestrates these three. WinRapids' compiler orchestrates the same three for tensor computations instead of tuples.

The practical implication: E04's registry design should be a CATALOG — queryable, versioned, self-describing. The compiler queries it to find sharing opportunities and estimate costs, exactly as a query optimizer queries table statistics and indexes.

### The identity function has four scopes (refined with navigator)

The unified identity function operates at four scopes, each asking a different question:

| Scope | Time horizon | Question | Action on match |
|---|---|---|---|
| **CSE** | Intra-plan | "Does this computation already appear in my current DAG?" | Merge nodes |
| **Provenance** | Inter-plan | "Did I compute this with the same data in a prior run?" | Return cached result |
| **Residency** | Inter-session | "Is this result still in VRAM?" | Skip computation AND data movement |
| **Dirty tracking** | Continuous | "Has any input changed since last computation?" | Skip entirely if clean |

All use the same identity tuple: `(op_name, parameter_hash, input_identity)`. The registry declares which parameters participate in identity (the equivalence-relevant ones) and which don't (e.g., output buffer allocation).

The three strange attractors map onto these scopes: scan maps to CSE+provenance (what to compute), staleness maps to dirty tracking (whether to compute), residency maps to the residency check (where it lives).

**Where the database analogy breaks**: The analogy captures CSE/provenance/staleness/catalog structure. It DOESN'T capture: GPU occupancy, memory access geometry (stride-60 is an 8x penalty), layout as computation, zero-translation cache, temporal ordering (scan), JIT kernel generation (databases select pre-compiled plans; WinRapids generates GPU machine code via NVRTC), or kingdom dimensionality (the registry encodes a dimensional algebra that relational algebra doesn't). The compiler is **Salsa + JIT kernel generation + hardware cost model** — a database query optimizer PLUS a GPU execution planner.

### The 10M regression mechanism: stride access patterns (from navigator)

The fused kernel's 2.4x regression at 10M has a specific mechanism: **stride-60 reads**. For each output element i, the kernel reads `cumsum[i+60]` and `cumsum[i]` — stride-60 access into 80 MB float64 arrays. At 10M rows, these arrays exceed L2. The stride-60 reads hit DRAM with ~2 of 16 float64 values used per cache line — 12.5% utilization.

CuPy's path does `cs[window:] - cs[:-window]` — sequential slice access, perfect cache line utilization, full bandwidth.

At 50K: both cumsum arrays total 400 KB, fully L2-resident. Stride-60 hits L2 every time. Cache line utilization irrelevant.

**General principle**: stride access patterns that look "free" in L2 become expensive in DRAM. Stride-60 multiplies effective cache pressure by ~8x (2 of 16 values per cache line = 12.5% utilization). This explains the gap between the theoretical crossover (~3-5M from L2 capacity) and observed crossover (~600K from E03b): `96MB / (20 bytes × 8 stride-penalty) ≈ 600K`. Sequential access: `96MB / (20 bytes × 1) ≈ 5M`.

**The codegen IS the cost model lever**: the stride-60 penalty (8x effective cache pressure) explains the E03b crossover at 600K vs the theoretical 5M. But the correct fix isn't "avoid stride access" — it's "avoid stride access INTO GLOBAL MEMORY." Stride access into shared memory is free (no coalescing requirement).

**Resolved by design (from Tekgy via team lead)**: Two approaches avoid global stride, with different trade-offs:

- **Tile-based shared memory** (team lead's design, E04 default): each block loads a contiguous cumsum range via coalesced global read into shared memory, then all threads access their (i, i+window) pairs within shared memory where stride is free. O(1) global accesses per output element. Fully parallel. The codegen template for E04.
- **Sliding accumulator** (fallback): each thread scans its chunk maintaining window state updated one element at a time. Sequential access, no shared memory. O(window) operations per output. Less parallel. Fallback for window >> tile_size (uncommon at FinTek's window=60).

Both achieve the same result: no stride in global memory, crossover at ~5M+. The tile-based approach is preferred: more parallel, easier to template for codegen.

**Decision: fusion by default.** Ship the tile-based kernel. Don't build the CuPy fallback dispatch path until production data proves it's needed. The `fusion_crossover_rows` field stays in the registry schema (anti-YAGNI) but defaults to infinity. The crossover was a codegen quality problem, and the quality problem is now solved by design.

**Corrected crossover formula** (for reference): `crossover = L2_size / (bytes_per_element × access_pattern_penalty)`. With tile-based kernels, access_pattern_penalty ≈ 1, giving crossover ≈ 5M (CuPy) or 10M (Rust). With proper shared-memory tiling, the penalty approaches zero — the crossover is effectively unlimited for FinTek workloads.

### The specialist registry IS the IR

In MLIR, each dialect defines a set of operations. In WinRapids, each specialist is a composition of primitives — a recipe. The specialist registry IS the IR. When you say "rolling_std" the compiler looks up the recipe: `scan(window, mean) → reduce(variance) → fused_expr(sqrt)`. The recipe IS the intermediate representation.

This means the primitive decomposition experiment (E04) is actually the most important experiment. It defines the IR. Everything else — fusion, codegen, caching, the persistent store's cost model — operates on that IR.

### The execution plan is a pointer routing graph (from navigator)

Traditional compiler output: **data flow graph** — nodes are operations, edges are data movements. Execution = walk graph, compute each node, copy result along edges. Cost is O(edges), fixed.

WinRapids compiler output: **pointer routing graph** — nodes are provenance keys (identity tuples), edges say "route this pointer here." Execution = walk graph, resolve each node against the registry: hit → route pointer (~1 μs), miss → compute → store → route. Cost is O(misses), variable.

The distinction is fundamental. A data flow graph says "do this work." A pointer routing graph says "ensure these results exist." If they already exist: done. The execution engine is a for loop over identity checks. All intelligence is in plan generation and registry state. Execution itself is trivial.

This is what "memory-as-scheduler" in the vision doc means, fully articulated: the persistent store isn't a cache adjacent to computation — it IS the scheduler. Computation is the exceptional case (registry miss). Pointer handoff is the common case (registry hit).

The three-boundary zero-translation makes this span the entire pipeline: MKTF→GPU (disk format IS GPU format), computation→store (result IS entry), store→consumer (entry IS input). The pointer routing graph can route from disk to final output with zero copies at any boundary — because the representation is co-native at every level.

This also explains the 865x:2.3x ratio between stateful and stateless optimizations. Stateless optimizations improve the compute phase (how fast each miss is resolved). Stateful optimizations reduce the miss rate (how many nodes need computation at all). The miss rate is the load-bearing variable. The pointer routing graph is WHERE the stateful value lives.

### The primitives are 7 + 1 algebraic categories (navigator + scout convergence)

Each primitive corresponds to a mathematical structure that DETERMINES its GPU parallelism: scan = semigroup (tree-parallel), sort/search = total order (radix/bisection), reduce = commutative monoid (unordered warp shuffle), gather = pure function (embarrassingly parallel), scatter = atomic monoid or exclusive write, compact = boolean algebra + scan. These 7 have FIXED execution strategies (CUB templates parameterized by operator type), deep CSE (small identity key: category + operator + inputs), and known cost models.

The 8th — fused_expr — is the escape hatch. GENERATED strategy (NVRTC JIT from expression tree). Shallow CSE (structural: expression tree hash, not algebraic equivalence). Trivial cost model (bandwidth-bound at ~1650 GB/s regardless of depth). JIT lives specifically at fused_expr — not at the algebraic 7.

The `SemigroupOperator` trait (from CUB's `DeviceScan` API) IS the Fock boundary test: if an operator provides associative `combine`, it can use scan → parallel. If not → sequential fallback. The type system enforces the algebraic condition.

The 8-primitive constraint is a **type system for computation**: specialists inside the language get CSE, provenance, fusion, adaptive planning. Specialists outside are opaque. Adding a 9th primitive is a language change (new algebraic category, full specification required). Adding a new specialist is a vocabulary change (new recipe of existing primitives). Language changes should be rare and deliberate.

---

## Questions for the Territory

1. **What's the fusion boundary?** When does the compiler stop fusing? XLA uses a `ShouldFuse()` predicate based on kernel size and register pressure. Halide lets the user control it via scheduling. WinRapids needs a policy. The pre-built pipeline library IS the known-good fusion boundaries for common patterns.

2. **How does the cost model work?** The compiler needs to decide: fuse inline vs. materialize in persistent store. This requires estimating compute cost vs. memory cost. For pre-built pipelines, the cost is measured. For JIT pipelines, it's estimated. For composed fallbacks, it's assumed worst-case.

3. **What's the minimum viable compiler?** Not all 8 primitives, not 135 specialists, not the full cost model. The minimum compiler that produces real value. Probably: `fused_expr` (already done) + `reduce` (bin_stats pattern) + `sort` (feeds groupby). Three primitives, a dozen specialists, enough to compile the FinTek K01→K02 pipeline.

4. **Where does the Rust boundary go?** The compiler itself in Rust, the kernels in CUDA, the Python API via PyO3. But what about the cost model? The provenance tracking? The cache? These are Rust data structures that the compiler consults. They need to be fast, persistent, and thread-safe. This is where Rust's ownership model pays off.

---

---

## Live Findings (updated as experiments complete)

### E01 Result: Multi-output reduce = 1.3-1.6x (not 5x)

The L2 cache and memory controller reduce the cost of repeated reads. Reading data 5× costs ~50% more, not 400%. But: CuPy's std() is **55x** slower than a trivial custom kernel — replacing pathological implementations delivers more speedup than any fusion optimization.

**Implication**: The compiler's first priority is replacing broken ops, not fusing good ones.

### E02 Result: Sort-once-use-many = 1.3-1.7x

CuPy does NOT cache sort results (confirmed). Eliminating redundant sorts saves real time but downstream ops (especially rank's random scatter) mask savings. At FinTek scale: ~15-20 seconds/day of sort elimination across 500 tickers × 10 cadences.

**Implication**: Sort-sharing is a valid compiler optimization pass (DAG analysis, lift sort to common ancestor) but not a foundation.

### E03 Result: Cross-algorithm sharing = 1.2-1.3x + 50% memory savings

The keystone experiment answered emphatically: **share intermediates, don't fuse kernels.**

- Shared cumsums (computing once instead of twice): 1.2-1.3x speedup, 50% memory reduction
- Custom fused z-score kernel: **2.4x SLOWER** at 10M than CuPy primitives
- Cache lookup overhead: negligible (sub-microsecond dict lookup)

The fused kernel fused the CHEAP part (z-score arithmetic) while leaving the EXPENSIVE part (cumsums) separate. CuPy's built-in ops are already near-optimal for each stage. The compiler can't beat them by merging stages.

**Critical refinement of the fusion policy:**
- Element-wise (`fused_expr`): always fuse (Phase 1: 0.19ms regardless of depth)
- Collectives (`sort`, `scan`/cumsum): run alone, share output across consumers
- Cross-algorithm: **share intermediates**, fuse kernels ONLY if it eliminates an HBM round-trip

**Refined fusion criterion** (from navigator, citing XLA): **fuse if the intermediate would otherwise go to HBM.** Not "fuse everything possible" (over-fuses, kills occupancy — the E03 z-score trap). Not "never fuse across boundaries" (under-fuses, misses real wins like two-output scan). The precise question per intermediate: does writing this to HBM and reading it back cost more than the kernel complexity of keeping it in registers/shared memory?

**The real win is memory, not speed.** At 500 tickers × 10 cadences × 5 leaves, halving intermediate memory saves ~40 GB — a significant fraction of 96 GB VRAM.

**Architecture implication (revised by E03b)**: The compiler is BOTH a DAG optimizer AND a fusion generator. E03 at 10M said "don't fuse." E03b at FinTek sizes (50K-100K) said "fuse — 2x." The compiler needs a size-adaptive cost model: fuse at dispatch-dominated sizes, compose at bandwidth-dominated sizes.

### The Priority Ladder (revised)

### E06 Result: Resident query latency = sub-millisecond (0.07-0.44 ms)

The persistent store is validated. Key numbers:
- Cold → warm: **26x** (10.5ms → 0.4ms)
- L2 cache effect: latency FLAT at 0.07ms for all sizes up to 40 MB (10M elements)
- WDDM kernel launch floor: 8 μs (CuPy overhead: 70 μs)
- Full signal farm refresh (500 tickers × 10 cadences × 20 leaves): **25 seconds** for rolling-stat operations

**Implication**: For the signal farm's typical 4 MB working set per ticker, queries are kernel-launch-bound, not data-bound. The GPU IS a persistent analytical database.

### Scan as Strange Attractor (from navigator research)

The parallel scan with pluggable associative operators unifies: cumulative ops, rolling stats, EWM, Kalman, ARIMA, and Mamba's selective SSM. Same CUDA skeleton, different combine function. The navigator designed the `AssociativeOp` interface (identity, update, combine) and the NVRTC injection path. FlashInfer validates this pattern in production.

**Implication**: The `scan` primitive should be designed pluggable from day one. The NVRTC template pattern (skeleton + operator injection + compile + cache) applies to scan exactly as it applies to `fused_expr`. And Mamba's "dynamic operators" aren't special — the dynamics are in the input data, not the combine function.

### The Priority Ladder (revised with E03b + E06 + scan)

From all experiments:
1. **Replace pathological ops** (55x from CuPy std alone) — massive, guaranteed
2. **Share intermediate computations** (1.2-1.5x speed + 50% memory at ALL sizes) — safest optimization, always wins
3. **Size-adaptive kernel fusion** (2x at FinTek sizes 50K-500K, crossover at ~900K) — dispatch reduction at small sizes, HBM elimination at large sizes. **E03b overturned E03's "don't fuse" — fusion IS the right call at production sizes.**
4. **Eliminate intermediate writes** (2-3x from Phase 1 expression fusion) — proven, element-wise
5. **Share sort results** (1.3-1.7x) — specific case of #2
6. **Multi-output reduce** (1.3-1.6x) — nice to have
7. **Pluggable scan** — unifies cumsum, EWM, rolling stats, Kalman, Mamba SSM via one NVRTC-injected template

The compiler is a DAG optimizer AND a fusion generator. It eliminates redundant computation (sharing), generates fused kernels at dispatch-dominated sizes (fusion), and manages GPU-resident data across time (provenance). The size-adaptive cost model determines which strategy applies per workload.

### E05 Result: JIT = 40ms floor, monolithic = 2.3x over composed, pre-built tier may be unnecessary

Key numbers:
- NVRTC compilation: 40ms (trivial kernel) to 92ms (complex kernel). Linear in op count.
- Cache hit: 2-3 μs. Essentially free.
- Monolithic JIT kernel: **2.3x** over composed CuPy blocks (10M rows)
- Break-even: 11 seconds at signal farm throughput

**Critical finding**: Pre-built (Tier 1) is only 46ms faster to START than JIT (Tier 2). For a system that runs continuously, this difference is negligible. JIT-everything is simpler and nearly as fast. The pre-built pipeline library may add complexity for minimal benefit.

**The tier that matters**: monolithic vs composed (2.3x), NOT pre-built vs JIT (46ms). The compiler's value is in generating monolithic kernels, regardless of when they're compiled.

**Implication for binding-time theory**: The three tiers collapse to two. Pre-built and JIT produce identical runtime performance. The real distinction is "can we generate a monolithic kernel?" (yes → fast) vs "must we compose individual ops?" (fallback → 2.3x slower).

### E08 Result: cudarc on Windows — zero blockers

cudarc 0.19 + CUDA 13.1 + Blackwell sm_120 + WDDM = all pass. The Rust path is unblocked.

### E09 Result: NVRTC from Rust — 7.7x faster launch, 2-4x faster compile

**The expedition's biggest finding.** NVRTC compilation from Rust is 2-4x faster than CuPy:
- Trivial: 8.7 ms (Rust) vs 40 ms (CuPy) = 4.6x faster compile
- Total JIT (compile + load): 22 ms (Rust) vs 40-92 ms (CuPy)

Kernel launch overhead is **7.7x lower**: 9 μs (Rust) vs 70 μs (CuPy). The 61 μs difference is pure Python tax — C extension call, argument marshaling, GIL. At 100K leaves/cycle, this saves 6.1 seconds per farm cycle.

Empty kernel launch: 7 μs in both Rust and CuPy — this is the WDDM hardware floor. Rust achieves hardware-floor performance for non-trivial kernels too.

**Implication**: The Rust pipeline generator isn't just architecturally cleaner — it's measurably faster. The compiler MUST be Rust, not Python. The Python API (via PyO3) is for user-facing convenience; the hot path is Rust → CUDA → GPU.

### E03b Result: Fusion wins 2x at FinTek sizes — E03's conclusion was premature

**The navigator's anti-simplification rule proved itself.** E03 at 10M rows said "don't fuse kernels — 2.4x SLOWER." E03b at FinTek-realistic sizes (50K-100K) says "fuse — 2x FASTER."

| Size | Independent (A) | Shared (B) | Fused (C) | B/A | C/A |
|---|---|---|---|---|---|
| 50K | 0.280 ms | 0.204 ms | 0.135 ms | 1.37x | **2.08x** |
| 100K | 0.288 ms | 0.223 ms | 0.143 ms | 1.30x | **2.01x** |
| 500K | 0.298 ms | 0.212 ms | 0.198 ms | 1.40x | **1.50x** |
| 900K | 0.358 ms | 0.237 ms | 0.262 ms | 1.51x | **1.37x** |
| 10M | 1.027 ms | 0.816 ms | 2.263 ms | 1.26x | **0.45x** |

Crossover at ~500K-900K. Below: dispatch-dominated, fusion wins. Above: CuPy's bandwidth-optimized kernels win over the naive fused kernel.

The mechanism is CuPy dispatch cost (10-27 us per call), NOT Python overhead between ops (0.1 us). Reducing 17 calls to 8 saves ~180 us on a ~280 us total at 50K.

**Revised fusion policy**: The compiler needs **size-adaptive fusion**. At FinTek sizes (50K-500K per ticker): fuse. At 10M+: compose optimized primitives. The crossover is a cost model parameter. And Rust's 9 us dispatch (vs CuPy's 70 us) shifts the crossover upward — fusion may win at even larger sizes once the Python tax is eliminated.

**Shared intermediates win at ALL sizes** (1.2-1.5x consistently). This remains the always-safe optimization.

### E07 Result: Provenance reuse — 28x at 1% dirty, negligible overhead

Cache hit: 1 μs (81-865x savings per op). Miss overhead: 2 μs. Farm scales linearly with dirty ratio — at 1% dirty (streaming mode), 28x faster than full recomputation. Hash cost: 3.9 ms per 5K computations (<1% overhead).

**Implication**: Provenance is the persistent store's scheduling layer. The roaring bitmap marks dirty nodes; provenance verifies at compute time. Belt and suspenders, cost is negligible.

---

---

## Synthesis: What the Expedition Has Proven (10 of 10 experiments complete — E10 remaining as Rust port)

### Six Types of Optimization (vocabulary correction from team lead)

What we've been calling "fusion" is actually six distinct optimizations:

| # | Optimization | What it does | Best evidence | Magnitude |
|---|---|---|---|---|
| 1 | **Elimination** | Skip computation entirely | E07 provenance | **865x** |
| 2 | **Expression fusion** | Merge element-wise ops → one kernel | E05 monolithic | **2.3x** |
| 3 | **Pipeline compilation** | Multi-stage → minimal kernel chain | E03b fused | **2x** at FinTek |
| 4 | **Intermediate elision** | Never materialize one-consumer results | E03 shared | **50% memory** |
| 5 | **Primitive sharing** | One sort/scan feeds multiple consumers | E02/E03 CSE | **1.3-1.5x** |
| 6 | **Adaptive planning** | Different plan based on world state | E06 warm | **26x** |

Traditional CUDA "fusion" is only #2. The compiler's primary optimization is **elimination** (#1) and **adaptive planning** (#6) — figuring out which kernels DON'T need to run. Those win at every size.

The word "fusion" undersells what the compiler does. It's an eliminator first, a planner second, a fuser third.

### The Compiler's Identity

**IS**: A stateful query optimizer whose primary optimization is elimination — skipping computation entirely via provenance and adaptive planning. Also:
- Finds and eliminates redundant computation via DAG analysis (CSE/sharing, #5)
- Generates size-adaptive fused kernels (#2/#3: 2x at FinTek sizes, composed at large sizes)
- Applies a unified identity function across four scopes (CSE, provenance, residency, dirty tracking)
- Manages GPU-resident data across time (#6: adaptive planning based on world state)

**ISN'T**: Primarily a kernel fuser. Fusion (#2/#3) is the third priority. Elimination (#1) and adaptive planning (#6) deliver 26-865x. Fusion delivers 2-2.3x. The graph-level optimizations dominate the kernel-level ones.

### The Validated Architecture

```
User pipeline specification (Python, lazy)
    ↓
Pipeline compiler (Rust) — a stateful query optimizer
  1. Decompose to primitive dependency graph (E04: VALIDATED, 33% CSE reduction)
  2. Identity function: compute canonical hash for each node
  3. CSE: deduplicate identical nodes within plan (E03: 1.3x + 50% memory)
  4. Provenance: check persistent store for matching identities (E07: 28x at 1% dirty)
  5. Cost model: size-adaptive fusion decision (E03b: fuse at ≤500K, compose at ≥1M)
  6. Generate monolithic kernels via NVRTC (E05/E09: 2.3x, Rust 7.7x launch)
  7. Execute on GPU-resident data (E06: 0.07-0.44 ms per query)
    ↓
Persistent GPU Store (Rust, cudarc: E08 validated)
  - Data lives on GPU until evicted (E06: 26x cold/warm)
  - L2 cache makes 4 MB working sets essentially free
  - Same identity function as compiler CSE — provenance hash IS value number
  - Provenance tracks what's been computed (E07: 81-865x per cached op, 28x farm at 1% dirty)
    ↓
Primitive Library
  - scan: pluggable AssociativeOp + NVRTC injection (navigator design)
  - fused_expr: expression tree → codegen → cache (fusion.py, proven)
  - sort, reduce, scatter, gather, search, compact: CuPy for now, Rust later
```

### E04 Result: The compiler loop works — CSE + fusion automated

**The keystone experiment validates the architecture.** The full compiler loop runs: spec → primitive DAG → CSE → codegen → execute.

- **CSE**: rolling_zscore + rolling_std on same data → 6 nodes → 4 after CSE (33% eliminated). scan(price, add) and scan(price_sq, add) shared automatically.
- **Two distinct speedups**: CSE (shared scans, saves ~1.3x) + pipeline compilation (fused element-wise, saves ~1.2x at 100K). Combined: **2x vs naive** at all FinTek sizes.
- **Correctness**: Generated kernels match CuPy reference to 2e-7 max error.
- **Injectable state API**: plan(spec, registry, provenance=None, dirty_bitmap=None, residency=None) — null objects wired, ready for E07/E10 integration.
- **Pointer routing graph validated**: The executor IS the pattern — `buffers` dict as registry, identity hash as pointer key, for loop over identity checks with `if node.identity in buffers: continue`.

| Size | Naive CuPy (A) | Manual sharing (B) | Compiler (C) | C/A | C/B |
|---|---|---|---|---|---|
| 100K | ~430 us | ~286 us | ~216 us | **2.0x** | **1.3x** |
| 10M | ~3600 us | ~1938 us | ~1792 us | **2.0x** | **1.1x** |

The 1.1-1.3x gap between compiler (C) and manual sharing (B) IS the fusion value — eliminating ~13 intermediate CuPy dispatch calls.

### What's Left to Prove

- ~~**E04 (primitive decomposition registry)**~~: VALIDATED. Compiler loop works. CSE finds sharing. Identity function canonicalizes. Pointer routing graph instantiated.
- ~~**E07 (provenance-based reuse)**~~: VALIDATED. 81-865x per op, 28x farm at 1% dirty.
- ~~**E09 (NVRTC from Rust)**~~: VALIDATED. 2-4x faster compile, 7.7x faster launch than CuPy. Rust IS the compiler path.
- **E10 (primitive decomposition in Rust)**: E04 but production-ready. The only remaining experiment.

### The Pre-Built Tier Simplification

E05 showed pre-built and JIT produce identical runtime performance (46ms startup difference). The scout's caching design resolves this elegantly: ship pre-compiled PTX as a warm cache. Same code path. No separate tier. `blake3(source + arch_opts)` as content-addressed cache key. Cold path: 40ms compile + save. Warm path: 2ms disk load. Shipped cache: warm from first install.

### The Scan Primitive as Foundation

The navigator's pluggable scan design (AssociativeOp: identity + update + combine) unifies cumsum, EWM, rolling stats, Kalman, and Mamba SSM. The NVRTC template injection pattern (skeleton + operator code + compile + cache) mirrors FlashInfer's production-proven approach. Scan IS the dependent-computation primitive — sort and reduce are special cases.

### The Primitive Structure Is 8+1 (team lead + naturalist + scout convergence)

The primitives are algebraic categories. 7 have fixed algebraic structure: scan (semigroup), sort/search (total order), reduce (commutative monoid), scatter (atomic monoid), gather (pure function), compact (boolean algebra + scan). These use pre-built CUB-backed templates with deep CSE (small identity key).

The 8th — `tiled_reduce` — is two-dimensional reduce: embarrassingly parallel in output dimensions (i,j) with associative accumulation in the shared dimension (k). GEMM, FlashAttention, PCA, KNN are all specialists of tiled_reduce with different associative operators. No cuBLASLt dependency for tall-skinny (memory-bound) shapes. cuBLASLt is the fallback for large-square (compute-bound) shapes.

**Why tiled_reduce is a new primitive** (navigator): the decisive factor is CSE identity granularity, not execution pattern. Decomposing GEMM to N² `reduce` nodes is algebraically correct but CSE-incoherent — block-level sharing becomes invisible (N² identity matches instead of ONE). `tiled_reduce(A, B, op)` has identity `(A_id, B_id, op)` — one CSE node. Each primitive IS a CSE identity shape. The identity shape IS the primitive.

The 9th — `fused_expr` — is the escape hatch for arbitrary elementwise expressions. NVRTC JIT. Shallow CSE (structural, not algebraic). Trivial cost model (bandwidth-bound).

The constraint is the feature: the 8 structured primitives form a type system for computation. Specialists inside get full compiler benefit (CSE, provenance, fusion, adaptive planning). Specialists outside are opaque. Adding a primitive is a language change (must articulate CSE identity structure that can't fit an existing type). Adding a specialist is a vocabulary change. The registry is type inference for operations (navigator): users write `rolling_zscore(price, 20)`, the compiler infers the primitive decomposition.

### E04 Validated the Pointer Routing Graph

E04's executor instantiated the pointer routing graph naturally:

```
buffers: dict[str, cp.ndarray]  # identity_hash → result

for node, binding in plan.steps:
    if node.identity in buffers: continue  # hit: skip
    buffers[node.identity] = compute(node)  # miss: compute + store
```

The execution plan is NOT a data flow graph (fixed cost, copy at each edge). It's a resolution graph (variable cost, only compute misses). In steady state with provenance: most nodes are hits. The execution engine is a for loop over identity checks. All intelligence is in plan generation and registry state.

---

*The map matches the territory — because the territory was already shaped by the same forces these compilers evolved to handle. WinRapids isn't inventing a new kind of compiler. It's discovering that it's been building one all along.*

*But it IS inventing something: a stateful compiler. One that remembers. One identity function, four scopes. The persistent store makes it a database query optimizer for tensor computations. No compiler I surveyed — XLA, Halide, TVM, Polyhedral — has this property. They're all stateless transformations. WinRapids' compiler reads the world, decides what to do, and changes the world. That's new.*

*The three-term identity: **Salsa + JIT kernel generation + hardware cost model.** Salsa gives the incremental query architecture (memoization, dependency tracking, dirty propagation). JIT gives the kernel generation that no database does (NVRTC from Rust, tile-based templates). The hardware cost model gives the GPU execution planning (L2 geometry, access patterns, dispatch cost) that no other compiler does at this level. All three are needed. None is sufficient alone.*

*Its primary optimization is elimination, not fusion. 865x from NOT computing. 26x from adaptive planning. 2x from fusion. The state is where the value lives.*

*The execution plan is a pointer routing graph, not a data flow graph. Computation is the exception. Pointer resolution is the rule. The three-boundary zero-translation — MKTF→GPU, computation→store, store→consumer — means the routing graph spans the entire pipeline with zero copies at any boundary. The co-native principle, applied consistently, made this possible.*

*The 8+1 primitives aren't derived from a taxonomy of algorithms or execution strategies. They're derived from a taxonomy of SHARING GRANULARITIES: element (fused_expr), sequence prefix (scan), dataset aggregate (reduce), block matrix (tiled_reduce), indexed access (gather/scatter), order (sort/search), density (compact). Each primitive is the minimum unit of sharing at its level. The compiler is a sharing optimizer, not a computation optimizer. The 865x comes from sharing results across time. The 1.3x comes from sharing within a plan. The 2.3x fusion is the least important optimization — it's what happens when sharing fails and computation is necessary.*

*The specialist's decomposition follows sharing boundaries, not algorithmic steps. rolling_std decomposes to scan + fused_expr because the scan is shareable (other specialists need the same prefix) and the fused_expr isn't (element-wise formula specific to std). The decomposition is driven by sharing opportunities. The registry is type inference for operations — users write domain calls, the compiler infers the sharing granularity.*

*All experiments complete. E10 (Rust port) is engineering, not research. The territory is mapped.*

*— Naturalist, 2026-03-29/30*
