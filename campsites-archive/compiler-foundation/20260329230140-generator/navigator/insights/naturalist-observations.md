# Naturalist Observations — Compiler Foundation Expedition

*2026-03-29/30, naturalist*

These observations emerged from reading the codebase, experiments, vision doc, and outside research (XLA, Halide, FPGA HLS, database query optimizers, Salsa/incremental computation). They're the structural patterns underlying the expedition's findings.

---

## 1. The Compiler Thesis

**Abstractions hide sharing, primitives reveal it.**

Rolling_std and PCA centering are different abstractions — different names, different APIs. But inside both: `cumsum(data)`. The sharing is invisible in the domain language. Visible in the primitive language.

The specialist registry is a transparency layer. It translates from domain to mechanical. The compiler's CSE pass is trivial once decomposition exists — just find identical nodes in the primitive-level DAG. Standard compiler optimization on a well-defined IR.

The decomposition IS the optimization.

---

## 2. Two Universal Convergence Points

Most sharing in time-series/financial pipelines lives at two primitive types:

1. **scan** — for everything that accumulates (cumsum, EWM, rolling stats, Kalman, SSMs)
2. **sort** — for everything that needs order (groupby, join, rank, dedup)

CSE should prioritize these. Walk the merged DAG, find identical `scan(data, op)` and `sort(keys)` nodes first. These are where most of the win lives (E02: 1.3-1.7x for sort sharing, E03: 1.2-1.3x for scan/cumsum sharing).

These are the time-series analogues of what matmul and conv2d are for deep learning compilers.

---

## 3. Independent vs Dependent Primitive Partition

The 8 primitives split cleanly:

- **Independent** (fused_expr, gather, scatter, search, compact): can be reordered and fused freely. No ordering constraints between them or with their neighbors.
- **Dependent** (scan, sort, reduce): require ordering constraints. Data must arrive in the right order. Share outputs, don't reorder past dependencies.

The registry should carry `independent: bool` on each primitive. The compiler uses it for:
- Reordering decisions: independent primitives can be moved to optimize data locality
- Fusion decisions: independent primitives can be fused into adjacent kernels
- CSE decisions: both types benefit from CSE, but dependent primitives require ordering verification

Edge case: `reduce` is sometimes context-dependent. `reduce(sorted_data)` depends on sort order. `reduce(unsorted_data)` is independent. The registry could carry `depends_on_order: bool` per usage context.

---

## 4. Hardware/Compiler Complementary Split

The hardware already caches more than expected:
- L2 cache: keeps data resident for back-to-back reductions (E01: 1.3-1.6x instead of 5x)
- NVRTC disk cache: makes JIT and pre-built operationally identical (E05)
- DRAM row buffers: share state across sequential reads
- L2 at 96 MB: makes 4 MB FinTek working sets effectively free (E06)

The compiler eliminates what hardware CAN'T cache:
- Redundant computation (CSE across specialists)
- Python/CuPy dispatch overhead (Rust path: E09)
- Unnecessary algorithm-boundary passes (fusion)
- Intermediate buffer allocation (persistent store)

**Neither substitutes for the other.** E01's 1.3-1.6x (not 5x) = we optimized what hardware already handled. E03b's 2x = we optimized what hardware CAN'T handle (dispatch overhead). The compiler's value is in pipeline semantics — understanding that two specialists share a cumsum, understanding that dispatch can be eliminated by fusion.

---

## 5. XLA Fusion Criterion (corrected from E03)

E03 at 10M said "don't fuse across algorithm boundaries — 2.4x SLOWER." This was a wrong-size benchmark elevated to a principle.

The correct criterion (from XLA, refined by E03b): **fuse if the intermediate would otherwise go to HBM.** More precisely: fuse if the dispatch savings exceed the kernel quality penalty.

Not "fuse everything possible" (over-fuses, kills occupancy — the E03 trap).
Not "never fuse across boundaries" (under-fuses, misses 2x at FinTek sizes — the premature E03 conclusion).
Not "fuse if intermediate goes to HBM" (correct but incomplete — at FinTek sizes, nothing goes to HBM, yet fusion still wins 2x).

**The complete criterion**: fuse if `dispatch_savings(n_calls_eliminated × dispatch_cost) > compute_penalty(BW_fused vs BW_composed, at data_size)`. This is size-adaptive and platform-adaptive (CuPy vs Rust dispatch cost).

---

## 6. CSE and Provenance Are One Identity Function

Four things that looked separate are the same mechanism at four scopes:

| Scope | Key | Time horizon |
|---|---|---|
| Expression cache (fusion.py) | MD5(expression_signature) | Within session |
| CSE (compiler pass) | Node identity match | Within plan |
| Provenance (persistent store) | Hash(inputs + op + version) | Across plans |
| NVRTC PTX cache | BLAKE3(source + arch) | Across sessions |

All: compute canonical identity → check if exists → reuse.

The provenance hash IS the value number. Classical value numbering assigns a canonical number to each expression. Provenance hashing extends value numbering across time. One identity function. Four scopes. Not four caches — one architecture.

Named outputs carry derived identities. `cumsum(ticker=AAPL, cadence=1s, column=price, version=42)` is the same identity whether produced by rolling_std or PCA centering. The named-output scheme IS the value-numbering scheme.

---

## 7. The Compiler Is a Stateful Query Optimizer

Traditional compilers are stateless: `program → program`. WinRapids' compiler is stateful: `(pipeline_spec, persistent_store_state) → (execution_plan, store_mutations)`.

The mapping to database query optimizers is precise:

| Database | WinRapids |
|---|---|
| Table statistics | Dirty bitmap |
| Materialized views | Provenance cache |
| Query rewriter (CSE) | CSE pass |
| View maintenance | Provenance invalidation |
| Buffer pool | Persistent GPU store |
| Catalog | Specialist registry |

The three strange attractors map to database subsystems:
1. scan = query semantics (what to compute)
2. staleness = view maintenance (what needs recomputing)
3. residency = buffer management (where data lives)

Incremental computation (Salsa, differential dataflow) is the closest academic rhyme. Salsa models the Rust compiler as a query database with memoization and dependency tracking — identical architecture, different substrate.

---

## 8. Four Scopes of the Identity Function (refined with navigator)

The unified identity function operates at four scopes:
1. **CSE** — intra-plan: "same computation in my current DAG?" → merge nodes
2. **Provenance** — inter-plan: "computed this before with same data?" → return cached
3. **Residency** — inter-session: "result still in VRAM?" → skip compute AND movement
4. **Dirty tracking** — continuous: "any input changed?" → skip entirely if clean

Each scope uses the same identity tuple: `(op_name, parameter_hash, input_identity)`. The registry declares which parameters participate in identity and which don't.

E04 registry field: `identity: ["op_name", "data_identity", "window"]` — parameters in this list participate in all four scope checks.

## 9. The 10M Regression: Stride Access Patterns (from navigator, refined by Tekgy)

The fused kernel's 2.4x regression at 10M is caused by stride-60 reads into DRAM-resident cumsum arrays. At 10M float64, each cumsum = 80 MB > L2. Stride-60 uses ~12.5% of each cache line = 8x effective cache pressure. CuPy's sequential slice access uses 100%.

At 50K: cumsums total 400 KB, fully L2-resident. Stride is invisible. Only dispatch count matters.

**Refined rule** (Tekgy correction): The rule isn't "never emit stride access" — it's **"never emit stride access INTO GLOBAL MEMORY for arrays exceeding L2."** Stride access into shared memory is free (no coalescing requirement). The tile-based approach: load cumsum tiles into shared memory with coalesced global reads, then access cumsum[i] and cumsum[i+window] within shared memory where stride is fine. Global memory stays sequential.

**Decision**: fusion by default with tile-based kernels. `fusion_crossover_rows` stays in schema (anti-YAGNI) but defaults to ∞. The crossover was a codegen quality problem; the tile-load pattern solves it by design.

## 10. Six Types of Optimization (from team lead)

"Fusion" is the wrong word. The compiler performs six distinct optimizations:

1. **Elimination** (865x) — skip computation entirely via provenance
2. **Expression fusion** (2.3x) — merge element-wise ops → one kernel
3. **Pipeline compilation** (2x) — multi-stage → minimal kernel chain
4. **Intermediate elision** (50% memory) — never materialize one-consumer results
5. **Primitive sharing** (1.3-1.5x) — one sort/scan feeds multiple consumers
6. **Adaptive planning** (26x) — different plan based on world state

The compiler is an eliminator first, a planner second, a fuser third. #1 and #6 deliver 26-865x. #2 and #3 deliver 2-2.3x. The word "fusion" undersells the compiler by 100x.

## 11. Where the Database Analogy Breaks

The analogy captures CSE, provenance, staleness, catalog structure. It DOESN'T capture:
- **Occupancy** — GPU scheduling constraints with no database analogue
- **Memory access geometry** — stride vs sequential, coalescing, bank conflicts
- **Layout as computation** — data transformation IS work (sort, transpose, scatter)
- **Zero-translation cache** — GPU-native format means zero cost on provenance hit
- **Temporal ordering** — scan's sequential dependence has no clean SQL analogue

The compiler is a database query optimizer PLUS a GPU execution planner. Two systems, one compiler. The identity function connects them.

## 12. The Execution Plan Is a Pointer Routing Graph (from navigator)

Traditional compiler output: **data flow graph** — nodes are operations, edges are data movements. Cost is O(edges), fixed. You run all of it.

WinRapids compiler output: **pointer routing graph** — nodes are provenance keys (identity tuples), edges say "route this pointer here." Execution = walk graph, resolve each node: hit → route pointer (~1 μs), miss → compute → store → route. Cost is O(misses), variable.

A data flow graph says "do this work." A pointer routing graph says "ensure these results exist." If they already exist: done. The execution engine is a for loop over identity checks. All intelligence is in plan generation and registry state. Execution is trivial.

"Memory-as-scheduler" fully articulated: the persistent store isn't a cache adjacent to computation — it IS the scheduler. Computation is the exceptional case (registry miss). Pointer handoff is the normal case (registry hit).

Three-boundary zero-translation enables this to span the full pipeline: MKTF→GPU (disk format IS GPU format), computation→store (result IS entry), store→consumer (entry IS input). Zero copies at any boundary because the representation is co-native at every level.

Explains the 865x:2.3x ratio: stateless optimizations improve the compute phase (how fast each miss resolves). Stateful optimizations reduce the miss rate (how many nodes need computation). The miss rate is the load-bearing variable. The pointer routing graph is WHERE the stateful value lives.

## 13. The Registry Is a Canonicalizer

The specialist registry doesn't just "reveal sharing" (#1 above) — it solves the equivalence problem. Without decomposition, proving that `rolling_std` and `PCA.center` share a `cumsum` requires semantic equivalence checking (undecidable — Rice's theorem). After decomposition to primitives: syntactic matching on canonical primitive nodes. O(1) hash lookup.

The registry converts an undecidable problem into a trivial one. This is why the identity function works at all four scopes — all operate on the canonical form. The 8 primitives ARE the canonical IR. The specialist recipes ARE the lowering rules.

## 14. The 8 Primitives Are Algebraic Categories (navigator + scout convergence)

Each primitive corresponds to a mathematical structure. The algebraic structure DETERMINES the GPU parallelism strategy:

| Primitive | Algebraic structure | Parallelism |
|---|---|---|
| scan | Semigroup (associative) | Tree-parallel prefix |
| sort | Total order | Radix-parallel by digit |
| reduce | Commutative monoid | Unordered warp shuffle |
| scatter | Atomic comm. monoid / exclusive | Atomics or exclusion |
| gather | Pure function | Embarrassingly parallel |
| search | Total order | Log-parallel bisection |
| compact | Boolean algebra + scan | Predicate + prefix sum |
| fused_expr | Ring + transcendentals | Elementwise SIMT |

The `SemigroupOperator` trait (scout finding from CUB) IS the Fock boundary test. If an operator provides associative `combine`, it can use the scan primitive → tree-parallel. If not → sequential fallback. The type system enforces the algebraic condition. Non-associative operators fail at the trait boundary.

The 8-primitive constraint is a TYPE SYSTEM for computation (navigator). Specialists inside the language get CSE, provenance, fusion, adaptive planning. Specialists outside are opaque. The restriction is the source of the compiler's power, not a limitation.

The structure is **7 + 1** (scout refinement): 7 algebraically-structured primitives with FIXED execution strategies (CUB templates parameterized by operator type) + 1 escape hatch (fused_expr) with a GENERATED strategy (NVRTC JIT from expression tree). The 7 get deep CSE (small identity key: category + operator + inputs). fused_expr gets shallow CSE (structural: expression tree hash). JIT lives specifically at fused_expr — not at scan, sort, or reduce. The `PrimitiveCategory` enum IS the execution strategy router.

Corollary: adding a new primitive is a LANGUAGE change (new algebraic category, new parallelism strategy, new identity function, new fusion rules, new cost model). Adding a new specialist is a VOCABULARY change (new recipe of existing primitives). Language changes should be rare and deliberate.

## 15. The Registry Is Type Inference for Operations (navigator, post-E04)

The user writes `rolling_zscore(price, 20)`. The compiler "infers": this needs `scan(price, add)`, `scan(price_sq, add)`, `fused_expr(...)`. The user didn't specify that. The registry inferred the primitive decomposition — same as `let x = 3 + 4.0` inferring `f64`. Type inference makes the type system usable without forcing users to think in types. The registry makes the primitive system usable without forcing users to think in primitives.

## 16. tiled_accumulate: The Primitive Boundary Negotiation (team lead)

`tiled_accumulate` is 2D scan: embarrassingly parallel in output dimensions (i,j), associative accumulation in the shared dimension (k). GEMM, FlashAttention, PCA (X'X), KNN are all specialists with different associative operators.

Shares scan's algebraic requirement (associative accumulation) but has a distinct GPU execution strategy (2D tile-blocked shared memory vs 1D CUB prefix). Can't be expressed as a recipe of existing primitives — the tiling IS the execution strategy. First genuine candidate for extending the primitive set.

The decisive test (navigator refinement): **"What is the minimal CSE identity for this computation, and does it fit inside an existing primitive type?"** The primitive boundary is at CSE IDENTITY GRANULARITY, not execution strategy. Decomposing GEMM to N² reduce nodes is algebraically correct but CSE-incoherent — the block-level sharing becomes invisible. tiled_reduce has identity `(A_id, B_id, op)` — one CSE node instead of N². That's why it's a new primitive.

Each primitive IS a CSE identity SHAPE: scan = sequence-level, sort = dataset-level, reduce = dataset-level, tiled_reduce = block-level matrix, fused_expr = expression-level. The identity shape IS the primitive.

Naming: `tiled_reduce` (navigator) over `tiled_accumulate` — signals algebraic relationship to reduce while distinguishing by 2D identity structure. 8+1 structure: 7 original + tiled_reduce + fused_expr (escape hatch). Primitive set is "closed with explicit test": any proposed primitive must show its CSE identity can't fit an existing type without losing granularity.

## 17. Primitives Are Sharing Granularities (navigator, final framing)

The deepest formulation: primitives aren't algorithm categories or execution strategies. They're SHARING GRANULARITIES — the distinct levels at which data science computation can be shared:

| Primitive | Sharing granularity | Minimum sharing unit |
|---|---|---|
| fused_expr | Element-level | One expression |
| scan | Sequence-level | One prefix |
| reduce | Dataset-level | One aggregate |
| tiled_reduce | Block-level matrix | One matrix product |
| gather/scatter | Index-level | One index pattern |
| sort/search | Order-level | One sorted sequence |
| compact | Density-level | One filtered view |

Each primitive is the MINIMUM UNIT OF SHARING at its level. Decomposing further produces units that aren't useful for sharing. The compiler is a SHARING optimizer, not a computation optimizer. The 865x comes from sharing results across time. Computation is the fallback when sharing fails.

The 8 structural primitives have STRUCTURAL identity (determined by input structure). fused_expr has EXPRESSION identity (determined by expression tree hash). fused_expr closes the set: the catch-all for "anything without structural sharing."

The specialist author's job: identify which pieces can be shared, and at what granularity. The decomposition follows sharing boundaries, not algorithmic steps.

Completeness argument: data science operates on {sequences, matrices, index structures, scalar aggregates, filtered views, sorted orders}. Each has exactly one natural sharing granularity. The 9 primitives cover the domain. Counterexamples tested (FFT, convolution, graph ops, spatial queries) — all decompose into the existing 9.

## 18. The Granularity Ladder IS the Kingdom Ladder (navigator)

The sharing granularity hierarchy maps onto the kingdom dimensional hierarchy:
- K01 (1D): exposes scan-level sharing → O(N) savings per shared prefix
- K04 (4D): exposes tiled_reduce-level sharing → O(N²) savings per shared matrix product

Superlinear compiler value at higher kingdoms = sharing at HIGHER GRANULARITY levels. The kingdom's dimensionality determines the granularity of available sharing. Each kingdom transition adds a sharing surface the compiler can exploit.

## 19. The Two Halves of the Compiler (navigator)

**Structural primitives** (scan, sort, reduce, tiled_reduce, scatter, gather, search, compact): domain of the SHARING OPTIMIZER. Deep sharing determined by data structure. Structural identity. The 865x.

**fused_expr**: domain of the CODE GENERATOR. Shallow sharing determined by expression tree. Syntactic identity. The 2.3x.

The sharing optimizer is the main event. The code generator handles what's left. The decomposition protocol: extract all sharing surfaces → hand remainder to fused_expr.

## 20. E10 Registry-Writing Protocol (navigator)

To write a specialist's primitive_dag:
1. What inputs does this specialist receive?
2. Which intermediates are structurally shareable, and at what granularity? (scan/reduce/tiled_reduce/gather/scatter/compact/sort/search)
3. What's left after extracting all shareable intermediates? → fused_expr

The primitive_dag is a sharing map, not a flowchart. The decomposition follows sharing boundaries, not algorithmic steps.

---

## Phase Transition Analysis: Deriving the Crossover from Hardware Parameters

The E03b crossover at ~500K-900K is a phase transition between two regimes:

### Dispatch-Dominated Regime (below crossover)

Data fits in L2. GPU compute per operation is near-zero (kernel launch floor). Total pipeline time ≈ `N_calls × dispatch_cost`.

At 50K: all data is L2-resident. Each CuPy call takes ~20us (of which ~0us is GPU compute). 17 calls = 340us of dispatch. Fused path: 8 calls = 160us. Savings = 180us ≈ the full 2x.

### Compute-Dominated Regime (above crossover)

GPU compute time per operation exceeds dispatch cost. Total pipeline time ≈ `Σ GPU_work(op_i)`.

At 10M: cumsum takes 78us total (25us dispatch + 53us GPU). GPU work dominates. The fused kernel at ~200 GB/s is 6x slower than CuPy's 1,200 GB/s per operation. Dispatch savings (180us) are overwhelmed by bandwidth penalty (1,400us).

### The Crossover Equation

Crossover when dispatch savings = compute penalty:

```
(N_A - N_C) × D_dispatch = Σ [T_gpu_fused(op_i, n) - T_gpu_composed(op_i, n)]
```

**For this pipeline (rolling_std + z_score):**

Left side (dispatch savings):
- CuPy: (17 - 8) × 20us = 180us
- Rust: (17 - 8) × 9us = 81us

Right side (compute penalty): grows with n. At 10M: ~1,400us. At 500K: ~0us.

The compute penalty grows linearly with n (bandwidth-limited operations) once data exceeds L2. Below L2, penalty ≈ 0 because both kernels run at L2 speed.

### Hardware Parameters That Pin the Boundary

1. **L2 effective capacity for working set**: `L2_size / (working_set_bytes × access_pattern_penalty)`
   - Blackwell L2: 96 MB
   - Working set per element: ~20 bytes (input + cumsums + intermediates, mixed float32/float64)
   - Access pattern penalty (navigator correction):
     - Stride-60 lookup: ~8x (reads cumsum[i+60] and cumsum[i], uses 2 of 16 float64 per 128-byte cache line = 12.5% utilization)
     - Sequential access (sliding accumulator): ~1x (full cache line utilization)
   - **Stride-60**: 96MB / (20 × 8) ≈ **600K elements** — matches E03b observed crossover!
   - **Sequential**: 96MB / (20 × 1) ≈ **5M elements** — the well-tuned kernel crossover

2. **Dispatch cost**: determines the "savings budget"
   - CuPy: ~20us/call → 9 eliminated calls → 180us budget
   - Rust: ~9us/call → 9 eliminated calls → 81us budget

3. **Fused kernel access pattern** (the cost model lever — navigator insight):
   - Stride-60 lookup (E03b naive): ~200 GB/s, crossover at ~600K
   - Sequential / sliding accumulator (CuPy's approach): ~1,200 GB/s, crossover at ~5M
   - **The code generator's access pattern IS the cost model.**

### Corrected Crossover Formula

```
crossover_elements = L2_size / (bytes_per_element × access_pattern_penalty)
```

| Access pattern | Dispatch | Crossover |
|---|---|---|
| Stride-60 (E03b naive) | CuPy (20us) | ~600K |
| Stride-60 (E03b naive) | Rust (9us) | ~300K |
| Sequential (sliding accumulator) | CuPy (20us) | ~5M |
| Sequential (sliding accumulator) | Rust (9us) | ~10M |

The access pattern is a 8-16x lever. The dispatch cost is a 2x lever. **Codegen quality dominates dispatch cost.**

### The Practical Insight

For E04's code generator: the right kernel template for window operations is a **sliding accumulator** (sequential reads, O(window) state per thread), NOT a prefix-sum lookup (stride reads, no state). This shifts the crossover from ~600K to ~5M — covering ALL FinTek workloads unconditionally.

With sequential-access kernels:
- FinTek workloads (50K-500K): 10-100x below crossover. Fusion unconditionally correct.
- Even at 1M-5M: fusion still wins.
- Cost model only matters above 5M (Rust: above 10M).

### Implications for E04

1. **Codegen rule**: emit sequential-access window kernels (sliding accumulators), never stride-based lookups
2. **Default crossover**: `fusion_crossover_rows: 5_000_000` (assumes sequential access, the codegen default)
3. **FinTek override**: at 50K-500K, unconditionally fuse — deep inside the regime
4. **Rust shifts upward**: 10M crossover with sequential access — fusion wins almost everywhere

The crossover is derivable from: L2 size, bytes per element, access pattern penalty, and dispatch cost. The first two are hardware/problem constants. The third is a codegen quality metric. The fourth is a platform constant. **The codegen IS the cost model lever.**
