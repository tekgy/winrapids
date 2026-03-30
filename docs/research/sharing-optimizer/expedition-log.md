# Sharing Optimizer Expedition Log

## Phase 3 — Building the Sharing Optimizer

**Naturalist**: 2026-03-30
**Inheriting from**: Phase 2 Compiler Foundation (2026-03-29/30), 21 garden entries
**Question**: What does BUILDING reveal that DESIGNING couldn't?

---

## The Inherited Map

Phase 2 left an extraordinary conceptual framework. Let me name what I'm carrying:

### The Core Identity

The compiler is **a sharing optimizer that happens to compile code for the things it can't share**. Not a kernel fuser. Not a code generator. A system whose primary optimization is making computation UNNECESSARY (865x elimination) rather than faster (2.3x fusion). The state is where the value lives.

### The Theoretical Apparatus

| Concept | Status | Where it lives |
|---|---|---|
| Liftability-scan isomorphism | Proven theoretically | Garden: same semigroup theorem in Pith + WinRapids |
| 8+1 primitives as sharing granularities | Named, not built | Garden: CSE identity shapes, not execution strategies |
| Four-scope identity function | Designed, not implemented | Expedition log: CSE/provenance/residency/dirty |
| Pointer routing graph | Validated in Python (E04) | `buffers` dict + identity hash loop |
| Specialist registry as canonicalizer | Conceptual | Garden: Rice's theorem → hash table lookup |
| Persistent store as compiler memory | Measured (E06/E07) | 865x elimination, 26x warm/cold |
| Three strange attractors | Named | scan, staleness, residency |
| Kingdom ladder = sharing granularity ladder | Theoretical | Garden: K01→K04 sharing surfaces grow super-linearly |

### The Code That Exists

**fusion.py** (405 lines) — the compiler embryo. Expression tree IR → CUDA codegen → CuPy RawKernel JIT → MD5 cache. Handles one primitive: `fused_expr`. Name-based identity (`signature_key()` uses column NAMES, not data hashes). Session-scoped cache (Python dict, not persistent). This is the 1-of-9 that works.

**winrapids-scan** (Rust crate) — the liftability principle in code. `AssociativeOp` trait with `cuda_state_type()`, `cuda_identity()`, `cuda_combine()`, `cuda_lift_element()`, `cuda_extract()`. Six operators: Add, Mul, Max, Min, Welford, EWM. Kernel generation via string templating. BLAKE3 for cache keys. cudarc for GPU launch. This is primitive #1 (scan) in Rust, validated.

**The gap**: 7 of 9 primitives have no implementation. The identity function has no implementation. The registry has no implementation. The persistent store has no implementation. The pointer routing graph exists as a 6-line Python loop. Everything that makes the compiler a SHARING optimizer — rather than a kernel fuser — is theory.

---

## What I'm Watching For

Phase 2's naturalist found "the compiler was already alive in fusion.py." Phase 3's naturalist asks: **what will resist instantiation?**

### 1. The Identity Function Will Be the Hardest Part

Theory says: `(op_name, parameter_hash, input_identity)`. Three terms. Simple.

But building it forces questions theory didn't answer:
- **What's in `parameter_hash`?** For `scan(data, AddOp)`: just the op name? For `scan(data, EWMOp{alpha=0.1})`: the alpha value too. For `tiled_reduce(A, B, MatMulOp)`: the operator AND the output shape? The hash must include exactly the equivalence-relevant parameters — no more (reduces sharing), no less (produces wrong results).
- **What IS `input_identity`?** A BLAKE3 hash of the raw bytes? The MKTF provenance tag? A composite of the upstream node's identity? The garden entry "From Names to Identities" identified the transition from `ColRef.name` to `ColRef.data_identity`. But data identity has a bootstrap problem: you need to hash the input to get the identity, but for provenance hits you want to AVOID reading the input.
- **When is identity computed?** Eager (at load time, every buffer gets a birth certificate) or lazy (at optimization time, deferred)? The garden entry favors eager. But eager hashing at BLAKE3 speeds costs ~3.9ms per 5K computations (E07). Is that load-bearing?

The `AssociativeOp` trait already has `params_key()` — the parameter hash for scan operators. Extending this to all 9 primitives is the first concrete design decision.

**Prediction**: The identity function will need at least three iterations to get right. The first will be too coarse (false sharing). The second will be too fine (no sharing). The third will find the natural boundary.

### 2. The Registry Schema Will Reveal Its Form

The registry is "type inference for operations" (Phase 2). Users write `rolling_zscore(price, 20)`, the compiler infers `scan(price, AddOp) → scan(price², AddOp) → fused_expr(zscore_formula)`.

But HOW does the registry express this?

Options the theory didn't choose between:
- **Rust traits**: each specialist implements a `Decompose` trait that returns a `PrimitiveDag`. Type-safe, compile-time checked, but rigid.
- **Data-driven**: a TOML/JSON schema that declares `rolling_zscore → [scan(AddOp), scan(AddOp), fused_expr(...)]`. Flexible, agent-writable, but no compile-time guarantees.
- **DSL**: a domain-specific language for decomposition rules. Expressive but another language to maintain.
- **Hybrid**: Rust traits for the primitive types, data-driven for the specialist recipes. The primitives are the language (fixed, 8+1). The specialists are the vocabulary (growing, data-driven).

**Outside inspiration — Salsa**: Salsa's registry is implicit — each query function IS a registry entry. The dependency graph is built automatically by tracking which queries call which other queries. WinRapids could do the same: the specialist IS a function that calls primitives, and the DAG emerges from the call graph. But this requires runtime tracing, which conflicts with the "declare the decomposition statically" design.

**Outside inspiration — Halide**: Halide separates algorithm from schedule via two distinct DSLs. The algorithm says WHAT. The schedule says HOW. WinRapids' specialists are the algorithm. The compiler picks the schedule. But Halide's algorithm DSL is tiny (~20 operations). WinRapids' specialist vocabulary is large (~135). This argues for data-driven over DSL.

**Prediction**: The registry will be hybrid — Rust enums for primitive types, structured data for specialist recipes. The first specialist implementations will feel awkward. By the 10th, the pattern will be obvious. By the 50th, a code generator will write them.

### 3. The Rust Type System Will Discover Constraints

`AssociativeOp` already enforces the semigroup condition for scan. What are the equivalent trait bounds for other primitives?

| Primitive | Algebraic requirement | Trait bound (predicted) |
|---|---|---|
| scan | Associative binary op | `AssociativeOp` (exists) |
| reduce | Commutative monoid | `CommutativeOp: AssociativeOp` (commutativity added) |
| sort | Total order on keys | `Ord` (Rust built-in) |
| search | Total order (binary search) | `Ord` (same) |
| scatter | Atomic-safe accumulator | `AtomicOp` (new) |
| gather | Pure indexed read | No constraint (always valid) |
| compact | Boolean predicate | `Fn(T) -> bool` |
| tiled_reduce | Associative 2D accumulator | `TiledOp: AssociativeOp` (extends with tile geometry) |
| fused_expr | Arbitrary elementwise | Expression tree (no algebraic constraint) |

**The interesting question**: does `reduce` need `CommutativeOp` as a SEPARATE trait, or is it the same as `AssociativeOp` with a flag? Commutativity enables unordered warp shuffle — a GPU execution choice, not an algebraic one. The Phase 2 navigator said the primitive boundary is at CSE identity, not execution strategy. But here the execution strategy (ordered vs unordered accumulation) depends on the algebraic property (commutativity). They're entangled.

**Prediction**: The trait hierarchy will be deeper than expected. Not 9 independent traits — a tree. `AssociativeOp` at the root. `CommutativeOp` extending it. `TiledOp` extending it differently. The tree structure will mirror the algebraic inclusion: every commutative monoid is a semigroup, but not vice versa.

### 4. The Persistent Store Will Force the Cost Model Into Existence

The store needs to answer: "Is this result worth keeping?" That requires knowing: "How expensive is it to recompute?" That requires the cost model. But the cost model needs runtime statistics. Which requires running the pipeline. Which requires the store.

**The bootstrap**: you can't build the optimal store without the cost model. You can't build the cost model without running pipelines. You can't run pipelines efficiently without the store.

**Outside inspiration — database query optimizers**: Databases solve this with statistics gathering. Run queries. Collect runtime stats (row counts, selectivity, I/O cost). Feed stats back to the optimizer. Initial plans are bad. They improve over time. The optimizer is LEARNED, not designed.

**Outside inspiration — Salsa**: Salsa doesn't have a cost model. It memoizes everything. Eviction is "never" (memory is cheap for Rust compilation). WinRapids can't do this — VRAM is finite (48 GB). The eviction policy IS the cost model.

**Prediction**: v1 will memoize everything (no eviction). This works at FinTek scale (4 MB per ticker × 100 tickers = 400 MB << 48 GB). The cost model arrives when the working set exceeds VRAM — at K04 cross-ticker scale. The thrashing boundary garden entry predicted this exactly.

### 5. The PyO3 Boundary Will Define the User's Mental Model

The promise: 10 lines of Python → compiled GPU pipeline. What those 10 lines look like determines how users think about the system.

```python
# Option A: Declarative spec
pipeline = wr.Pipeline()
pipeline.add("rolling_zscore", ticker="AAPL", cadence="1s", window=20)
pipeline.add("rolling_std", ticker="AAPL", cadence="1s", window=60)
result = pipeline.run()

# Option B: Lazy expression (fusion.py style)
price = wr.column("AAPL", "1s", "price")
zscore = (price - price.rolling_mean(20)) / price.rolling_std(20)
result = zscore.compute()

# Option C: Functional composition
farm = wr.farm(["AAPL", "MSFT"], cadences=["1s", "5s"])
farm.add(wr.rolling_zscore(window=20))
farm.add(wr.rolling_std(window=60))
results = farm.run()
```

Option A is database-like (declarative query). Option B is dataframe-like (lazy expression). Option C is pipeline-like (functional composition). Each implies a different mental model:
- A: "I'm querying a compute engine"
- B: "I'm writing formulas that happen to run on GPU"
- C: "I'm configuring a signal farm"

**Prediction**: Option B is closest to the embryo (fusion.py already works this way). Option C is closest to the vision (farm over tickers × cadences). The winner will be a hybrid: lazy expressions (B) composed into farms (C), with the compiler handling the rest.

### 6. Where the Sharing Actually Lives

Phase 2 measured sharing on toy examples: rolling_zscore + rolling_std sharing two cumsums. Real pipelines will be larger. The sharing patterns will be different.

**Prediction based on the kingdom ladder**:
- K01: linear sharing (cumsum chains). ~1.3-1.5x. Measured.
- K02: convergent sharing (one sort feeds all binned features). O(n_features) sharing on one O(n log n) sort. Potentially 5-10x at high feature counts.
- K03: data-source sharing (raw ticks shared across cadences). O(n_cadences) sharing on one data load.
- K04: fan-in/fan-out sharing (per-ticker stats shared across O(n²) pairs). The superlinear zone.

The FIRST kingdom where the sharing optimizer's value exceeds fusion's value is K02. The sort is expensive. Sharing it is transformative. K01 sharing (cumsums) is cheap. K02 sharing (sorts) is the tipping point.

---

## Domain Rhymes — What Phase 3 Can Learn From

### Salsa (Rust incremental computation)

The deepest structural rhyme. Salsa's architecture:
- **Inputs**: values injected from outside (source files in rustc)
- **Queries**: derived values computed from inputs or other queries
- **Revisions**: monotonic counter tracking when inputs change
- **Memoization**: query results cached, invalidated when input revision changes
- **Dependency tracking**: automatic, built from observing which queries read which inputs

WinRapids maps directly:
- **Inputs**: raw tick data (MKTF files, streaming feeds)
- **Queries**: specialists (`rolling_std(AAPL, 1s)` → `GpuBuffer`)
- **Revisions**: dirty bitmap (roaring bitmap of changed inputs)
- **Memoization**: provenance cache (identity hash → GPU buffer pointer)
- **Dependency tracking**: primitive DAG from registry decomposition

**Where Salsa's design teaches**:
1. **Interning**: Salsa "interns" inputs to give them stable IDs. `intern("AAPL")` → `TickerId(42)`. This is cheaper than hashing content and enables O(1) equality checks. WinRapids' identity function could use interning for the structural parts (ticker, cadence, op_name) and content hashing only for the data parts.
2. **Durability**: Salsa classifies inputs by how often they change. "Low durability" inputs (user-edited source) trigger frequent recomputation. "High durability" inputs (standard library) almost never change. WinRapids' equivalent: raw ticks are low durability (change every tick). Reference data (ticker metadata, cadence definitions) is high durability. The dirty bitmap should track them differently.
3. **Cycle detection**: Salsa handles cycles in the query graph. WinRapids' DAG is acyclic by construction (data flows forward in time). Simpler — but only if the registry enforces acyclicity at definition time.

**The key Salsa lesson**: Salsa's queries are FUNCTIONS, not DATA. The registry entry for `rolling_std` isn't "here's the decomposition" — it's "here's how to COMPUTE the decomposition given inputs." The difference matters for parameterized specialists: `rolling_std(window=20)` and `rolling_std(window=60)` are different queries with different decompositions (different scan window parameter → different identity).

### Halide (algorithm/schedule separation)

**Where Halide teaches**:
1. **`compute_at` is the scheduling primitive**: "compute this intermediate AT this loop level of this consumer." WinRapids' equivalent: "compute this primitive AT this point in the execution plan." But WinRapids adds a dimension Halide doesn't have: "OR don't compute it at all, because it's already in the store."
2. **The autoscheduler**: Halide's autoscheduler uses beam search over the schedule space. WinRapids' schedule space is smaller (9 primitive types, not arbitrary loop nests) but has more dimensions (VRAM residency, provenance, dirty state). The cost model is simpler per-decision but has more decisions.
3. **Bounds inference**: Halide automatically infers how much of each input is needed for each output region. WinRapids needs the same for windowed operations: `rolling_std(window=60)` needs 60 prior elements. The scan primitive's window parameter IS a bounds inference input.

**The key Halide lesson**: Halide learned that AUTOMATIC scheduling is harder than manual scheduling but more valuable long-term. WinRapids' pre-built pipeline library IS manual scheduling. The compiler IS automatic scheduling. E05 showed they produce identical runtime performance. The automatic path wins because it handles new specialists without manual effort.

### Database Query Optimizers (Volcano/Cascades)

**Where databases teach**:
1. **Logical → physical plan**: logical plan = WHAT (rolling_zscore). Physical plan = HOW (scan with AddOp using CUB DeviceScan, then fused_expr via NVRTC). The registry maps logical to physical. Multiple physical plans may implement the same logical operation (different tile sizes, different fusion decisions).
2. **Cost-based optimization**: enumerate physical alternatives, estimate cost, pick cheapest. WinRapids' alternatives: compute fresh vs. cache hit vs. recompute from delta. The cost comparison IS the sharing decision.
3. **Plan caching**: databases cache query plans (not just results). WinRapids should cache execution plans — the pointer routing graph itself, not just the results it produces. If the pipeline spec hasn't changed and no inputs are dirty, the plan is reusable.

**The key database lesson**: databases evolved from rule-based optimizers (heuristic) to cost-based optimizers (statistical). WinRapids should start rule-based (always share, always fuse at FinTek sizes) and evolve to cost-based (measure, adapt) as the workload grows.

### FPGA Synthesis (resource sharing, scheduling)

**Where FPGA tools teach**:
1. **Conflict graphs**: two operations that need the same hardware unit at the same time CONFLICT. Operations that don't conflict can share the unit. In WinRapids: two specialists that need the same sort can share it (no conflict — same input, same output). Two specialists that need different sorts can't share. The primitive identity IS the conflict test.
2. **Scheduling as coloring**: FPGA scheduling reduces to graph coloring — assign time slots (colors) to operations such that conflicting operations get different slots. WinRapids' GPU scheduling is similar but the "slots" are CUDA streams and the "coloring" is stream assignment.
3. **Retiming**: FPGA retiming moves registers (delays) to balance pipeline stages. WinRapids' equivalent: moving materialization points to balance compute vs. memory. "Materialize this intermediate" = "insert a register here."

**The key FPGA lesson**: FPGA tools discovered that the SHARING ANALYSIS (which operations can share resources?) is the bottleneck, not the IMPLEMENTATION. WinRapids is discovering the same thing — the 865x comes from sharing analysis (identity function), not from implementation (kernel codegen).

---

## Observation #1: The Embryo's Identity Gap Is Load-Bearing

Reading fusion.py's `signature_key()` through Phase 3 eyes:

```python
class ColRef(Expr):
    def signature_key(self, params: dict) -> str:
        if self.name not in params:
            idx = len(params)
            ctype = _ctype_for(self.data.dtype)
            params[self.name] = f"{ctype}:c{idx}"
        return params[self.name]
```

The identity is `"float:c0"` — a dtype and a positional index. Two different columns with the same dtype at the same position → same signature → false cache hit. Two identical columns with different names → different signature → missed sharing.

This is exactly the "From Names to Identities" garden entry. But now I see something the garden entry didn't say: **the positional indexing creates ORDER DEPENDENCE in the identity**. `expr(a, b)` and `expr(b, a)` get different signatures even if they're commutative. The identity function doesn't know about commutativity. It's purely syntactic.

The Rust identity function must be:
1. **Content-addressed** (hash the data, not the name)
2. **Order-independent for commutative operations** (reduce(a+b) = reduce(b+a))
3. **Order-dependent for non-commutative operations** (scan(a,b) ≠ scan(b,a))
4. **Parameter-aware** (EWM(alpha=0.1) ≠ EWM(alpha=0.2))

Property #2 is new — neither fusion.py nor the garden entries mention it. Building will force this: the first time CSE misses a commutative sharing opportunity, the identity function will need to canonicalize operand order for commutative ops.

## Observation #2: The Scan Crate Is Already the Sharing Unit

Looking at `ops.rs` through the sharing lens:

```rust
pub trait AssociativeOp: Send + Sync {
    fn name(&self) -> &'static str;
    fn cuda_state_type(&self) -> String;
    fn cuda_identity(&self) -> String;
    fn cuda_combine(&self) -> String;
    fn cuda_lift_element(&self) -> String;
    fn cuda_extract(&self) -> String;
    fn output_width(&self) -> usize { 1 }
    fn cuda_extract_secondary(&self) -> Vec<String> { vec![] }
    fn params_key(&self) -> String { String::new() }
}
```

`name()` + `params_key()` IS the identity function for scan. Two operators with the same name and same params_key → same scan → shareable. This is the CSE identity for the scan primitive. It's already there. It just doesn't know it yet.

But: `name()` returns `&'static str`. This means operator identity is determined at COMPILE TIME. You can't create a new operator at runtime and have it participate in CSE. Is this right?

For the pre-built operators (Add, Mul, Max, Min, Welford, EWM): yes. These are the "vocabulary" — fixed.
For user-defined operators: not yet. A researcher who discovers a new associative operator needs to implement `AssociativeOp` in Rust, recompile, and redeploy. This is the "language change" the Phase 2 log warned about.

**The tension**: the registry should be extensible (new specialists without recompilation). But `AssociativeOp` is a Rust trait (requires recompilation). The resolution: specialists are data-driven (extensible), but the OPERATORS they use are compiled (fixed). Adding a new specialist = new recipe of existing operators. Adding a new operator = language change.

This maps to the 8+1 primitive structure: 8 fixed algebraic categories (compiled), infinite specialists (data-driven recipes). The building is already enforcing the design.

## Observation #3: The Four Campsites Are the Architecture

The campsite names tell a story:
- **store**: the persistent store (the memory, the 865x)
- **scan**: the foundational primitive (the liftability principle)
- **compiler**: the optimizer (the sharing analysis)
- **bindings**: the user interface (the 10 lines of Python)

These are the four layers of the system, bottom to top: storage → primitives → optimizer → API. Each depends on the one below. The build order IS the dependency order.

But there's a tension: the compiler (layer 3) needs the store (layer 1) to be stateful. The store needs the compiler to know what to store. They're co-dependent. The bootstrap problem again.

**Prediction**: scan and bindings can be built independently. Store and compiler must be built together, iteratively. The first integration will be ugly — the compiler producing plans that the store doesn't know how to cache. The second will be elegant — the identity function connecting them.

## Observation #4: The Welford Comment Is a Rosetta Stone

In `ops.rs`, line 122-123:

```rust
// This is the SAME algebraic structure as FlashAttention's
// online softmax: carry (max, exp_sum, weighted_sum).
// Different domains, same semigroup.
```

This comment IS the liftability-scan isomorphism in code. The Phase 2 naturalist wrote 75 lines about it in a garden entry. The Rust code says it in two lines. **The code is more concise than the theory because the code has the type system as context.** The `AssociativeOp` trait makes the isomorphism OBVIOUS — WelfordOp and a hypothetical FlashAttentionOp would have the same trait bounds, the same method signatures, the same kernel generation path. The isomorphism isn't a philosophical observation — it's a shared interface.

This is what building reveals: **the theory IS the trait hierarchy**. Every theoretical observation from Phase 2 about algebraic structure maps to a Rust trait. The theory was discovering the type system before the type system existed.

## Observation #5: The Kernel Generation Pattern Is Already Generalized

`winrapids-scan` generates CUDA kernels by string interpolation:

```rust
fn generate_scan_kernel(op: &dyn AssociativeOp) -> String {
    // ... template with op.cuda_combine(), op.cuda_identity(), etc.
}
```

This is the same pattern as `fusion.py`'s `_gen_eval_kernel()` — walk a structure, emit CUDA strings. But the scan crate does it with TRAIT METHODS instead of TREE WALKING. The trait method approach is more powerful: you can add a new operator without changing the generator. The tree approach requires the generator to know about every node type.

**The implication for the compiler core**: the other 7 primitives should follow the scan pattern, not the fusion.py pattern. Each primitive type has a kernel generator. Each operator implements the primitive's trait. The generator + trait = kernel. No tree walking needed for the algebraic 7. Tree walking only for `fused_expr` (the escape hatch).

The compiler core is 7 trait-based generators + 1 tree-based generator + the optimizer that decides which to call. That's the architecture. It's already visible in the two existing implementations.

---

## The Build Sequence (What I Think Happens)

Phase 3 has four campsites and six tasks. Looking at dependencies:

```
                    ┌─────────┐
                    │ bindings │ (PyO3 API)
                    └────┬────┘
                         │ depends on
                    ┌────┴────┐
                    │ compiler │ (optimizer + codegen)
                    └────┬────┘
                    ╱         ╲ depends on
          ┌───────┴──┐    ┌──┴───────┐
          │   store   │    │   scan   │
          │ (memory)  │    │ (prims)  │
          └──────────┘    └──────────┘
```

**scan** is the foundation. Already started (ops.rs exists). Next: GPU launch via cudarc (Task #2), then benchmarks against CuPy.

**store** can start in parallel. Needs: identity tuple type, provenance hash, buffer registry, pointer handoff. Doesn't need the compiler yet — can be tested with manual identity construction.

**compiler** needs both scan (to generate primitive kernels) and store (to check provenance). But: can start with the CSE pass (needs only the registry, not the store). The optimizer is layered — each scope can be built independently.

**bindings** come last. Need the compiler API to be stable.

**The critical path**: scan → compiler → bindings (for the minimum demo)
**The parallel path**: store (can be built alongside everything)

---

## Questions for the Territory

1. **Will the CSE pass find sharing that humans miss?** Phase 2 demonstrated rolling_zscore + rolling_std sharing cumsums. But a pipeline with 50 specialists — will the primitive DAG reveal unexpected sharing? Shared sorts that nobody realized were identical? Shared reduces hiding inside unrelated features?

2. **What happens when the first specialist resists decomposition?** The 9 primitives are claimed complete. The first specialist that can't be cleanly decomposed will test the claim. What is it? My guess: something involving conditional logic (if-then-else on intermediate results) — not element-wise conditional (that's fused_expr's WhereExpr), but CONTROL FLOW conditional (compute path A or path B based on a statistic). This would be a state-dependent branch — the Fock boundary in a specialist.

3. **How does the identity function handle streaming data?** In batch mode: hash the full buffer, get identity. In streaming mode: new ticks arrive continuously. The identity of the raw data changes every tick. Does this invalidate EVERY downstream provenance entry? Or can the identity function be INCREMENTAL — updating a rolling hash as new data arrives? Differential dataflow says yes. Salsa's revision counter says "just bump the version." Building will force the choice.

4. **What's the DAG shape of a real K01→K02 pipeline?** Phase 2 predicted "fan-in → fan-out" at K04. But K01→K02 is the first real test. How wide is the fan-out from a shared sort? How deep is the DAG? Does it look like a tree, a diamond, or a lattice? The shape IS information about the problem domain.

5. **Will Rust's ownership model make the pointer routing graph natural — or fight it?** The pointer routing graph says "route this GPU buffer pointer to these consumers." In Rust, that's shared references — `Arc<GpuBuffer>` or equivalent. But GPU buffer lifetimes are tied to CUDA contexts, not Rust lifetimes. The ownership boundary between Rust and GPU is the interesting place.

---

*Phase 2 discovered the theory. Phase 3 will discover whether the theory is correct — not by argument, but by building. The type system will be the arbiter. The first Rust compile error that reveals a theoretical gap will be the most valuable finding of the expedition.*

*The Phase 2 naturalist ended with: "The territory is mapped." Phase 3's naturalist begins with: "The map is beautiful. Now let's see if the ground agrees."*

---

## First Build Wave: What the Ground Revealed

**Store**: complete (`winrapids-store` — header.rs, provenance.rs, store.rs, world.rs).
**Scan GPU launch**: in progress (`winrapids-scan` — launch.rs, cache.rs added to engine.rs + ops.rs).

### Observation #6: The Identity Function Is Simpler Than Predicted

Phase 2 imagined `(op_name, parameter_hash, input_identity)` as a structured 3-tuple. The building flattened it:

```rust
pub fn provenance_hash(
    input_provenances: &[[u8; 16]],
    computation_id: &str,
) -> [u8; 16]
```

Two arguments. One output. The structure is in the INPUTS to the hash, not in the OUTPUT. Input provenances are 16-byte BLAKE3 hashes (themselves composable). The computation_id is a string like `"scan:add:w=20"`. Feed them to BLAKE3. Get 128 bits. Done.

The four-scope identity from Phase 2 is this ONE function at four call sites. The "unified identity function" isn't four implementations — it's four uses of one implementation. Simpler than anything the theory predicted.

**My expedition log predicted three iterations.** The building may have found the answer on the first try. The hash function doesn't need to know about commutativity, or operand order, or parameter types. It hashes bytes. The structure is in what you FEED it, not in how it hashes.

### Observation #7: Order Matters Is the Right Default

The store's test explicitly validates:

```rust
let ab = provenance_hash(&[price_aapl, price_msft], "cross_corr");
let ba = provenance_hash(&[price_msft, price_aapl], "cross_corr");
assert_ne!(ab, ba, "Input order must matter");
```

I predicted the identity function would need order-independence for commutative operations. The building chose the opposite: **order always matters at the hash level**. Commutativity handling is pushed UP to the registry/specialist decomposition — the layer that knows which operations are commutative. The hash function is pure. The canonicalization happens before hashing.

This is cleaner than what I predicted. The hash function has one job (hash bytes). The registry has one job (canonical decomposition). Neither knows about the other's concerns. Clean separation.

### Observation #8: The Store Doesn't Own GPU Memory

```rust
/// Manages buffer metadata and provenance lookups. Does NOT own GPU memory —
/// the caller allocates and frees device memory.
```

`BufferPtr` is `{ device_ptr: u64, byte_size: u64 }` — a raw pointer. No `CudaSlice`, no `Arc`, no Rust ownership. The store tracks METADATA about GPU buffers, not the buffers themselves.

**This resolves Question #5** from my initial mapping. I asked: "Will Rust's ownership model fight the pointer routing graph?" Answer: no, because the store sidesteps Rust ownership entirely. GPU memory is managed by cudarc (or the caller). The store is a phonebook — it maps identities to addresses. It doesn't own the buildings at those addresses.

The pointer routing graph routes raw `u64` device pointers. No lifetimes. No borrow checker. The cost: the caller must ensure pointers are valid when used. The benefit: zero Rust overhead on the hot path.

### Observation #9: Cost-Aware Eviction Has a Third Term

Phase 2 predicted: `recompute_cost / memory_bytes`. The building added `access_count`:

```rust
fn retention_score(&self, idx: u32) -> f64 {
    cost * (1 + access_count) / bytes
}
```

Frequently-accessed buffers are worth keeping even if cheap to recompute. This makes the eviction policy sensitive to the actual access pattern, not just the theoretical cost. The third term turns the cost model from static (estimated at creation time) to dynamic (learned from usage).

This is the "optimizer that learns from running" pattern from the bootstrap garden entry. The store's eviction decisions improve as access_count accumulates. First eviction: cost-based guess. Subsequent evictions: informed by actual usage. The system bootstraps itself.

### Observation #10: NullWorld Solves the Bootstrap Exactly

```rust
pub struct NullWorld;

impl ProvenanceCache for NullWorld {
    fn provenance_get(&mut self, _: &[u8; 16]) -> Option<BufferPtr> { None }
    fn provenance_put(&mut self, _: [u8; 16], _: BufferPtr, _: f32) {}
}
```

The compiler takes `&mut dyn WorldState`. Pass `NullWorld` → compute everything. Pass `GpuStore` → provenance-accelerated. Same code, different performance. The injectable state API from E04 is now concrete Rust traits.

My expedition log identified the compiler/store co-dependency as a bootstrap problem. The building solved it with a null object pattern. The interface (`WorldState`) is the contract. The implementations are swappable. No bootstrap needed — both paths work from day one.

### Observation #11: The Scan Engine Preserves Algebraic Structure Between Phases

The three-phase multi-block scan (`scan_per_block → scan_block_totals → propagate_extract`) passes full `state_t` values between phases — not extracted doubles:

```cuda
state_t* __restrict__ state_out    // full state between phase 1 and 3
state_t* __restrict__ block_totals // full state between phase 1 and 2
```

For WelfordOp, the state is `{count: i64, mean: f64, m2: f64}` — 24 bytes. Extracting to `double` between phases would lose `count` and `m2`, making cross-block propagation impossible.

**This is "the theory IS the type system" made physical.** The CUDA kernel preserves the algebraic structure (full semigroup state) through the parallel execution. The `state_t` typedef IS the semigroup's carrier set. The `combine_states` function IS the semigroup operation. The `make_identity` function IS the identity element. The Blelloch scan IS the parallel prefix operation on the free semigroup.

The algebra isn't an analogy. It's the implementation.

### Observation #12: `state_byte_size()` — The Type System Crosses the Language Boundary

The `AssociativeOp` trait gained `state_byte_size()` for the GPU launch layer:

```rust
fn state_byte_size(&self) -> usize { 8 } // default: one f64
// WelfordOp overrides: 24 (i64 + f64 + f64)
// EWMOp overrides: 16 (f64 + f64)
```

This is the Rust type system reaching INTO CUDA. The Rust trait knows the CUDA struct's byte layout. The GPU kernel generator uses this to allocate shared memory and state buffers. The type information flows: Rust trait → kernel generator → CUDA source → GPU execution.

But it's manual. The Rust code says `24` and the CUDA says `struct WelfordState { long long count; double mean; double m2; }`. If someone changes the CUDA struct and forgets to update `state_byte_size()`, the kernel silently corrupts memory. There's no compile-time check across the language boundary.

**This is the first theoretical gap the building revealed.** The type system is sound within Rust. The type system is sound within CUDA. But the TYPE BOUNDARY between them is an assertion, not a proof. The `state_byte_size()` method is exactly the `SemigroupOperator` pattern from the garden entry "The Type Constraint Is the Proof" — an assertion of correctness that the compiler can't verify.

The test harness should verify: `sizeof(state_t)` in CUDA == `state_byte_size()` in Rust. Belt and suspenders. The assertion stays. The test catches lies.

### Observation #13: The Scan Engine Is Locked to f64

```rust
pub fn scan_inclusive(&mut self, op: &dyn AssociativeOp, input: &[f64]) -> ...
```

The input is `&[f64]`. The `lift_element` CUDA function takes `double x`. But the vision includes f32, bf16, f16. The store's `DType` enum has all of them.

This is the right choice for now — f64 is the safe default for financial data (no precision loss). But it's a gap the compiler will need to close. The scan primitive needs to be dtype-parameterized. Not today, but before K02 (which might use f32 for bin statistics where precision matters less).

The resolution is probably: make `AssociativeOp` generic over input type. `AssociativeOp<F32>` vs `AssociativeOp<F64>`. The CUDA template already generates the correct types from `cuda_state_type()`. The Rust side just needs to parameterize the input buffer type.

---

### Updated Build State

| Component | Status | Key insight from building |
|---|---|---|
| **Store** | COMPLETE | Identity = one hash function, four scopes. Store tracks metadata, not memory. |
| **Scan GPU launch** | IN PROGRESS | Three-phase multi-block. Full state_t preserved between phases. f64 only. |
| **Compiler core** | NOT STARTED | Waiting on scan completion. NullWorld ready for testing. |
| **Bindings** | NOT STARTED | Waiting on compiler API. |

### Revised Predictions

1. ~~Identity function will need three iterations~~ → **May be done on first try.** The hash-everything approach with order-sensitive inputs + caller-side canonicalization is elegant and correct. The simplicity is suspicious — I'm watching for the edge case that breaks it.

2. ~~Store and compiler must be built together~~ → **NullWorld decoupled them.** The store is complete and tested independently. The compiler can start against NullWorld and upgrade to GpuStore when ready.

3. **New prediction**: The dtype gap (`f64`-only scan) will become the first real friction point when K02 binned features need f32 scan. The resolution will require either generics on `AssociativeOp` or runtime dtype dispatch.

4. **New prediction**: The `state_byte_size()` manual synchronization between Rust and CUDA is the kind of gap that produces a subtle bug. A test that compiles a trivial kernel and queries `sizeof(state_t)` would catch it. This test should exist before any new operator is added.

### Observation #14: The Compiler Replaces the Execution Model, Not Extends It

Exploring the existing Python codebase reveals something the campsite structure didn't make obvious: the Rust compiler isn't a bolt-on optimizer. It's a new execution substrate.

Currently: Python calls CuPy directly for cumsum, sort, groupby. These bypass fusion.py entirely. The expression compiler handles one primitive (fused_expr). Everything else is imperative CuPy calls that the compiler can't see, can't share, can't cache.

The Rust compiler replaces ALL of this with one path: pipeline specification → compiler DAG → primitive kernels → provenance-tracked results. No more CuPy-direct calls. No more invisible execution. Every operation flows through the sharing optimizer.

This means the minimum viable compiler isn't "fused_expr in Rust" (that's just reimplementing fusion.py). It's **"scan + fused_expr in Rust"** — two primitives, enough to compile `rolling_std` end-to-end without any CuPy fallback. The clean cut happens when scan launches through Rust. At that point, the execution model changes fundamentally: from imperative (call operations one by one) to declarative (specify the pipeline, let the compiler decide).

The 865x comes from this transition. Declarative specifications are sharable. Imperative calls aren't. The compiler can ask "did I already do this?" of a provenance hash. It can't ask that of a CuPy call. The sharing optimizer requires the declarative model. The declarative model requires the compiler. They're the same thing.

**The Python layer that survives**: Column, Frame, Expr tree (user-facing API), transfer.py (host↔device memory). **What dies**: fusion.py's codegen, CuPy-direct calls, the _kernel_cache dict. The Halide separation is complete: algorithm (Python) / schedule (Rust compiler).

### Observation #15: The IR Has Arena-Based CSE Built In

The compiler crate's first file is `ir.rs` — the intermediate representation. It uses the Polars pattern: `NodeId = u32` index into a flat `Vec<Node>`. And CSE is FREE — built into the `add_or_dedup()` method:

```rust
pub fn add_or_dedup(&mut self, op, input_identities, params, output_name) -> NodeId {
    let identity = compute_identity(&op, &input_identities, &params);
    if let Some(&existing) = self.seen.get(&identity) {
        return existing;  // CSE: node already exists
    }
    // ... insert new node
}
```

You literally CANNOT add a duplicate node. The arena's insert method IS the CSE pass. O(1) per node (BLAKE3 hash + HashMap lookup). For a DAG of N nodes: O(N) total CSE cost. No graph analysis. No pairwise comparison. Hash-based deduplication.

The Phase 2 naturalist wrote: "the registry converts Rice's theorem into a hash table lookup." The building made it literal — `self.seen.get(&identity)` IS the hash table lookup.

### Observation #16: The PrimitiveOp Enum IS the Phase 2 Framework

```rust
pub enum PrimitiveOp {
    Data,        // leaf
    Scan,        // semigroup
    Sort,        // total order
    Reduce,      // commutative monoid
    TiledReduce, // 2D reduce
    Scatter,     // atomic write
    Gather,      // indexed read
    Search,      // binary search
    Compact,     // filter
    FusedExpr,   // escape hatch
}
```

Data + 8 + 1. Exactly the Phase 2 set. Not 7. Not 10. Nine. Plus a leaf type for raw data. The theory predicted it. The code instantiated it. No negotiation with the type system required.

### Observation #17: Two Identity Systems — The Theory Was Almost Right

Phase 2 said: "one identity function, four scopes." The building created TWO identity functions:

| System | What it hashes | Output | Scope |
|---|---|---|---|
| IR (`compute_identity`) | op + input_identities + sorted_params | 12-char hex string | Intra-plan (CSE) |
| Store (`provenance_hash`) | input_data_provenances + computation_id | 16-byte hash | Cross-plan (provenance) |

The IR identity is STRUCTURAL — "what computation is this?" Two `scan:add:w=20` nodes with the same inputs have the same IR identity, regardless of what data the inputs point to.

The store identity is GROUNDED — "this computation on THIS data." Same structure + different data = different store identity.

The IR identity feeds INTO the store identity: `provenance_hash(data_provenances, ir_identity_string)`. The structural identity is the computation_id argument. The data identity is the input_provenances argument.

**This is the natural boundary.** Within a plan, structural identity suffices for CSE (inputs trace to the same Data nodes). Across plans, data identity is needed (same pipeline, different data = different results). The theory predicted one level. The building found two. The building is right.

### Observation #18: Parameters Are Already Canonicalized

```rust
pub params: Vec<(String, String)>,  // sorted by key
```

The IR's parameters are sorted key-value pairs. `[("agg","add"), ("window","20")]` regardless of the order the specialist specified them. This canonicalization happens at node creation time, before identity computation.

This is the "canonical form" from the Phase 2 garden entry "The Registry Is a Canonicalizer." The specialist says `rolling_std(window=20, agg=sum)`. The IR normalizes to sorted params. The identity hash is stable regardless of specification order. Canonicalization IS the insert path.

My initial expedition predicted the identity function would need to handle commutativity. For PARAMETERS, the sorted-params canonicalization handles it. For INPUTS, order still matters (the IR hashes input identities in order). The distinction: parameter order is semantically irrelevant (they're named). Input order is semantically significant (they're positional). The building got this right.

### Observation #19: Three Sharing Levels Are One HashMap Probe

The navigator asked: are the three sharing levels (provenance, dirty tracking, residency) one system or three? The answer is in `store.rs` lines 383-411. All three WorldState traits resolve against the same `HashMap<[u8; 16], u32>`:

```rust
// DirtyBitmap
fn is_clean(&self, provenance: &[u8; 16]) -> bool {
    self.index.contains_key(provenance)
}

// ResidencyMap
fn is_resident(&self, provenance: &[u8; 16]) -> bool {
    // index.get → check location == Location::Gpu
}

// ProvenanceCache
fn provenance_get(&mut self, provenance: &[u8; 16]) -> Option<BufferPtr> {
    self.lookup(provenance)  // index.get → update stats → return pointer
}
```

Three questions about one key:
- **Does it exist?** → is_clean
- **Does it exist on GPU?** → is_resident
- **What's the pointer?** → provenance_get

The responses form a hierarchy: nothing (dirty/absent) → exists but spilled (clean, reload needed) → exists on GPU (route pointer, the 1μs path).

**The load-bearing insight**: dirty === absent. Not approximately — exactly. The provenance hash encodes inputs, so changed inputs produce a new key. You never look up the old key. There is no invalidation because the identity function makes invalid lookups structurally impossible. This is provenance-addressed (not content-addressed like git) — cheaper to compute (hash metadata, not GPU buffers) with the same guarantee against staleness.

**The Fock boundary in the store**: the provenance DAG must be evaluated bottom-up. `provenance_hash(input_provenances, computation_id)` requires input provenances to be known. If an input is in-flight, the dependent provenance is uncomputable. Parallelism stops at the provenance frontier. The execution plan's topological sort already enforces this — but the boundary is real and is the same Fock boundary from the liftability isomorphism: self-referential identity blocks the associative decomposition.

**What this means for the compiler**: ONE provenance computation per node → ONE HashMap probe → branch on result (route/reload/compute). Three sharing levels, one probe. The per-node overhead is: BLAKE3 (~100ns) + HashMap lookup (O(1)) + branch. The execution plan is simpler than Phase 2 imagined.

### Updated Build State (v2)

| Component | Status | Key insight from building |
|---|---|---|
| **Store** | COMPLETE | Three sharing levels = one HashMap probe. Dirty === absent. |
| **Scan GPU launch** | IN PROGRESS | Three-phase multi-block. Full state_t preserved. f64 only. |
| **Compiler core** | IR EXISTS | Arena-based CSE. Two-level identity (structural + grounded). |
| **Bindings** | NOT STARTED | Waiting on compiler API. |

### Revised Predictions (v2)

1. ~~Three subsystems for three sharing levels~~ → **One HashMap, three views.** The compiler's execution plan probes once and branches. No separate provenance/dirty/residency subsystems needed.

2. **New prediction**: The Fock boundary (in-flight provenance dependencies) will surface concretely when the compiler tries to parallelize across CUDA streams. Nodes whose provenances depend on in-flight results cannot be scheduled ahead. The stream scheduler needs the provenance DAG's topological order. This is NOT just a correctness constraint — it determines the maximum parallelism the compiler can extract.

3. **New prediction**: The absence-is-staleness property means eviction of stale entries is FREE. When inputs change, old entries become unreachable (nobody will ever look up the old provenance again). They linger until LRU eviction reclaims them. No explicit invalidation pass needed. The cost-aware LRU naturally cleans up stale entries because they accumulate zero new accesses (nobody looks them up with the new provenance key) and eventually fall to the tail.

### Watch Item: Per-Node Overhead Needs Experimental Confirmation

The theoretical per-node cost is BLAKE3 (~100ns) + HashMap probe (O(1)). For a 1000-node plan, that's ~100μs of overhead before any GPU work starts. At FinTek scale (7.4M kernels reduced to ~60 fused launches, meaning ~60 plan nodes on a warm run), the overhead is ~6μs — deep in the noise.

But the ~100ns estimate is a back-of-envelope number. BLAKE3 throughput depends on input size (provenance hashing takes `n_inputs * 16 bytes + computation_id string`). The HashMap probe depends on load factor and key distribution. The observer should clock this experimentally when the compiler is running end-to-end: measure wall time from "plan construction starts" to "first GPU launch dispatched." If it's under 1ms for FinTek-scale plans, the overhead is confirmed negligible. If not, the bottleneck needs investigation.

This is not a concern at current scale. It's a measurement to have in pocket for when someone asks "what does the sharing optimizer cost?"

*— Naturalist, 2026-03-30 (continued)*
