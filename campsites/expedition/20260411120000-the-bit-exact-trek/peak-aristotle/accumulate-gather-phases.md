# Target: Why accumulate + gather?

**Deconstructor:** Aristotle
**Date opened:** 2026-04-11
**Status:** Phases 1–8 drafted. Second pass found stable.

Invariant under deconstruction: **I7 — Every primitive decomposes into accumulate + gather.**

Connected load-bearing claims:
- CLAUDE.md: "ALL primitives = accumulate(grouping, expr, op) + gather(addressing). 8 grouping patterns, 6 addressing patterns."
- The bit-exact trek's architectural claim rests on "same math, every ALU." If accumulate+gather is not actually a universal decomposition, every backend carries math the decomposition doesn't capture, and the cross-backend claim is weaker than advertised.
- The specialist registry, the JIT primitives, the persistent store, the gradient duality — every piece of the project's structural story references accumulate+gather as "the skeleton."

---

## Phase 1 — Assumption Autopsy

The phrase "accumulate + gather" has seven assumptions stacked inside it. I have to name them before I can test them.

1. **That mathematics decomposes.** Not all of it does. Some math is inherently non-compositional — fixed-point iteration without guaranteed convergence, undecidable predicates, oracle-parameterized algorithms. Assuming it all fits *a* decomposition is already a commitment.

2. **That there is a unique "natural" decomposition.** Even if all computable math decomposes, it decomposes many ways. Operator composition, category-theoretic arrows, lambda calculus, dataflow graphs, attribute grammars, term rewriting, SKI combinators — each claims to be universal. Choosing "accumulate + gather" is a choice among alternatives we haven't named.

3. **That the decomposition should be two-term.** Why two operations? Why not one (e.g., `fold` in the Bird-Meertens formalism) or three (map + fold + scan as in data-parallel algebra) or nine (the specialist primitives listed in project memory)? The cardinality "2" is doing work.

4. **That "accumulate" and "gather" are the right two.** Bird calls them `foldr/map`. APL calls them `reduce/index`. Futhark calls them `reduce/scatter`. MapReduce calls them `reduce/map`. Halide calls them `pure/reduction`. These are overlapping-but-distinct pairings. We picked "accumulate + gather." The label shapes the design space.

5. **That the decomposition is first-class — i.e., the *skeleton* of computation, not just one representation.** The tambear contract treats this decomposition as the filter test: if a primitive doesn't fit, it's declared to live in Kingdom B/C/D and TAM schedules it. That is to say, the *exceptions* are already enumerated. But enumerated exceptions to a "universal" decomposition are a tell that the decomposition isn't actually universal.

6. **That "every primitive" is a well-formed set.** What counts as a primitive? Is `rfft_real_only_length_power_of_two` a primitive or a specialization? Is `matmul` a primitive or a composition? The I7 claim applies only to whatever we choose to count, which means it's partly self-fulfilling.

7. **That decomposition *should* be the architectural axis in the first place.** Maybe the right axis is *data layout* (where does the tensor live?), or *cost topology* (which operations are O(n) vs O(n log n)?), or *stability class* (which kernels preserve condition number under composition?), or *information flow* (who produces what for whom?). Any of these could have been I7 instead.

Embedded sub-assumptions I also want to flag:

- **8 grouping patterns, 6 addressing patterns** — the catalog sizes are declarative. They're held as "the structure we found," not as "the structure we proved." Why 8 and 6? What would 9 and 7 look like? What happens at the boundaries we haven't tested?
- **"Kingdoms = tensor rank / cross-axis"** — couples the decomposition claim to a tensor-shaped data model. Non-tensor math (graphs, simplicial complexes, ragged structures) gets second-class treatment.
- **"Accumulate+gather runs on every backend"** — this is the portability claim, and it's load-bearing for the trek. If accumulate+gather compiles to fundamentally different primitives on GPU vs CPU vs NPU, then "runs everywhere" is a euphemism for "we've written backends that happen to agree."

---

## Phase 2 — Irreducible Truths

Stripping everything away. What is *undeniable*, not just useful?

1. **A finite computation produces a finite result from finite inputs via a finite number of primitive steps.** (This is just "computability." Without it, we have nothing.)

2. **A primitive step is an operation that, given its inputs, is defined in one atomic act by its ALU.** Add, multiply, load, store, compare. These are not decomposable *at our level of abstraction*; below our level they are microcode, but at our level they are opaque atoms.

3. **Parallel hardware executes multiple primitive steps simultaneously only when they are data-independent.** Any computation plan that wants to use parallel hardware must expose its data dependencies.

4. **Reproducibility requires a total order of operations.** If two runs execute different orders of the same operations, and the operations are not perfectly associative (fp isn't), the runs may differ in the last bit. "Same result" = "same ordering or an ordering proven to commute for this operator."

5. **Bit-exactness across hardware requires that the total order be the same on every hardware.** Not similar, not equivalent modulo associativity — *the same*. Because associativity is false for fp64.

6. **A compiler is a function from source representation to target representation.** It can only produce what its IR can name. If the IR can't name "this specific reduction order," the compiler can't enforce it.

7. **For a given kernel, there exist many valid orderings that differ only in bits.** Determinism picks one. Which one is picked is a choice, not a discovery.

8. **Any representation of computation must somehow encode: (a) what operations to perform, (b) on what data, (c) in what order, (d) into what destination.** These four are necessary. They are not necessarily *four separate things* — one representation can collapse (a) and (c), or (b) and (d). But all four *roles* must be filled.

9. **A decomposition into N primitives is useful iff:**
   - (a) Every intended computation maps to a composition of the N primitives, AND
   - (b) Each of the N primitives can be independently lowered to every target ALU with equivalent semantics, AND
   - (c) Composition preserves equivalence — if `A ≡ A'` and `B ≡ B'`, then `A ∘ B ≡ A' ∘ B'`.

   (c) is the compositional property. Without it, "decomposes" is meaningless.

Those are the irreducible truths I can defend. Everything else is engineering.

---

## Phase 3 — Reconstruction from Zero

Given only Phase 2's truths, what are the plausible skeletons for a cross-ALU numerical library? Ten reconstructions, ordered from simple and elegant to structurally ambitious.

### 1. **One primitive: `eval(expr_tree)`**
A single operation: given a DAG of operations with data references, evaluate it. No distinction between accumulate and gather; they're both node types in the DAG. The skeleton is the DAG itself. This is how most expression compilers work (Halide roughly, TVM roughly, XLA roughly).

- **Pro:** Maximum flexibility. No operations are second-class.
- **Con:** No structural guarantee that a DAG is parallelizable. The compiler has to re-discover accumulate+gather patterns from the graph shape to generate efficient code. What tambear's contract frames as "the decomposition" becomes an *analysis pass* over the DAG.

### 2. **Pure fold: the Bird-Meertens F-algebra**
One primitive: `foldr(op, seed, structure)`. Every computation is a fold over some algebraic structure. Accumulate is fold; gather is fold-over-indices; matmul is fold-over-dot-products. The catamorphism framework says this is universal for inductively defined data types.

- **Pro:** Deepest known mathematical justification. Category theory backs it.
- **Con:** Folds over irregular structures (graphs, sparse tensors, ragged arrays) require defining the initial algebra first. In practice, "every computation is a fold" becomes "after you define the right recursion scheme, every computation is a fold" — and the recursion schemes are the hard part. Also: the F-algebra doesn't tell you a *parallel execution order*. You still have to pick one.

### 3. **Three primitives: map, reduce, scan**
Blelloch's data-parallel canon. `map` is elementwise, `reduce` is associative fold, `scan` is prefix-fold. Every data-parallel algorithm decomposes into these three with some glue.

- **Pro:** Historically robust. Efficient parallel implementations well-understood.
- **Con:** Doesn't name `gather` explicitly — gather is a special case of map-with-index, but making it first-class simplifies a huge class of operations. Also missing: the *grouping* axis that tambear's accumulate has built in.

### 4. **Two primitives, but different: `map + reduce_by_key`**
MapReduce's view of the world. `reduce_by_key` is explicitly grouping-aware; `map` handles per-element transformation and gather via keys. This is close to accumulate+gather but named differently.

- **Pro:** Battle-tested at scale. The grouping is first-class.
- **Con:** Doesn't capture scan-like operations cleanly. Prefix operations become convoluted map-reduce chains. The "key" abstraction hides physical locality questions.

### 5. **The decomposition is accumulate + gather (current choice)**
Accumulate is "grouped reduce with op," gather is "address-pattern load." Every primitive is `accumulate(grouping, expr, op) + gather(addressing)`. Kingdoms declare dependency structure for non-liftable cases.

- **Pro:** Names grouping as primary, which matches statistical workloads. Names gather as primary, which matches permutation/sorting/join-like workloads. Two operations covers "what becomes what" and "what-goes-where" — the two eternal questions of data movement.
- **Con:** The kingdoms escape hatch admits that the decomposition isn't literally universal. Also, the precise semantics of "grouping" has to be extended repeatedly (8 patterns and growing) to cover real operations.

### 6. **Two primitives + explicit order: `scatter_add + ordered_fold`**
`scatter_add` is the inverse of gather — it *writes* to addresses instead of reading from them. `ordered_fold` is reduce with an explicit order hint. Together they force every computation to name both its destination pattern and its combining order.

- **Pro:** Determinism becomes structural: there is no "implicit reduction," every fold carries its ordering. The bit-exact trek's I5 becomes automatic.
- **Con:** The atomics question becomes the whole design. `scatter_add` with overlapping addresses requires either locks, a reordering, or a topological sort. On GPUs, you're back to the same problem accumulate has with atomicAdd.

### 7. **Decomposition by *time*: `pure_step + memoize`**
Every computation is a sequence of pure functions that memoize their intermediate results. The skeleton is time, not shape. The session cache (TamSession) already hints at this.

- **Pro:** Content-addressing of intermediates is natural. Sharing is automatic. Cross-primitive fusion is just "evaluate this chain lazily."
- **Con:** No structural help for *parallelism*. A pure-step decomposition doesn't tell the compiler how to split work across SMs. You need a second decomposition — spatial — to do that.

### 8. **Decomposition by *dependency width*: kingdoms as primary**
What the project calls Kingdoms (A: liftable/parallel-scan, B: sequential-recurrence, C: iterative-fixed-point, D: ...) becomes the *primary* axis. Operations are classified by their dependency topology; accumulate+gather is only the Kingdom-A special case.

- **Pro:** Every kind of math gets a first-class home. No "escape hatch" — everything is a kingdom member.
- **Con:** Forces the compiler to handle N dispatch cases, where N is the number of kingdoms. Loses the "one decomposition to rule them all" elegance. Also — the kingdoms are empirical, not formally classified. We don't have a proof that they're disjoint or exhaustive.

### 9. **The decomposition is the *algebraic structure* of the operation, not its dataflow shape**
Every primitive is characterized by its algebraic structure — semigroup, monoid, semiring, abelian group, ring, field — and its associated identity, inverse, distributive law. Composition is then "composition of algebraic structures," which has known rules. Dataflow shape (accumulate vs gather) is a *compilation target*, not a source representation.

- **Pro:** Captures what's actually load-bearing about most operations: their algebraic law. Sharing and fusion are provable from the algebra. The Kingdom classification becomes "which semigroup class does this operation belong to?"
- **Con:** Radically different IR. Harder to write. Requires the user (or the library author) to state the algebra, not just the dataflow. Historically, languages that require this have been niche.

### 10. **No compile-time decomposition at all: the IR is a *ledger of causal events*, and the compiler discovers structure at JIT time**
Represent every computation as a linear causal trace — an event stream where each event names its inputs and output — and let the compiler cluster, reorder, and schedule as it sees fit, subject only to causal dependencies. No static "shape classification." The notion of "primitive" disappears; the unit of compilation is the causal cluster that the JIT discovered.

- **Pro:** Maximum freedom. No assumption baked in that the library author couldn't have avoided.
- **Con:** Requires a heavy JIT. Cross-backend bit-exactness is now "across every machine, every JIT converges to the same causal cluster" — which is a much stronger claim, and probably false unless you pin every heuristic. You've moved the invariant battle from the source to the JIT.

---

## Phase 4 — Assumption vs Truth Map

Where does conventional thinking about "decomposition = accumulate + gather" lead astray when compared against Phase 2?

| Assumption | Matching truth | Where they collide |
|---|---|---|
| "Accumulate + gather is the natural decomposition" | T9: a decomposition is useful iff all intended computation maps to it + each primitive lowers equivalently + composition preserves equivalence | Accumulate+gather *mostly* satisfies this, but not for Kingdom B/C/D — which are declared as "honestly outside the decomposition." That declaration is evidence the decomposition is *useful* but not *irreducible*. |
| "It's two primitives" | T8: four roles (what/where/order/dest) must be filled | Two primitives can fill four roles only by overloading. Accumulate fills what+order+dest, gather fills where. The overloading is real: accumulate's `op` parameter is doing the work of "order" indirectly (by being associative). When op is not associative, the overloading breaks — and we invoke kingdoms. |
| "It's universal" | T9(a): every intended computation maps | The existence of Kingdoms B/C/D demonstrates that the "every" is aspirational, not proven. |
| "Grouping patterns are 8" | T2/T8: primitives are atomic at the ALU level | The 8 patterns are higher than ALU atomicity. They're a design-level abstraction. That's fine — but "there are exactly 8" is an empirical finding of what we've catalogued, not a mathematical derivation from first principles. |
| "Runs on every ALU" | T2, T5: ALU primitives are opaque atoms, bit-exactness requires same total order | Accumulate+gather does NOT specify total order — that's left to the backend. So "runs on every ALU" is true, but "runs with the same bits on every ALU" is a SEPARATE claim that I5 and I7 together try to enforce. I7 alone is not enough. This is where the trek's Peak 6 (determinism) comes in — it patches the gap I7 leaves. |

**The key insight from this map:** I7 is *one half* of what the trek actually needs. I7 gives us compositionality; I5 gives us determinism. The trek's architectural claim ("same bits, any ALU") requires both, but the accumulate+gather framing makes it easy to conflate them. A clean restatement:

> **I7′:** Every primitive decomposes into a *dataflow pattern* (which piece of data combines with which) and a *total order* (in what sequence the combinations happen). The dataflow pattern may be any of the kingdom classes; the total order must be explicitly represented.

Under I7′, accumulate+gather is the most common dataflow pattern (Kingdom A), and the total order is a separate first-class concept that every kernel must declare. This makes the determinism requirement structural rather than operational.

---

## Phase 5 — The Aristotelian Move

The highest-leverage action that emerges from first-principles thinking on this target:

**Extract "total order" as a first-class concept in the .tam IR, separate from the dataflow pattern.**

Concretely:

- Every accumulate in .tam IR must name its `order_strategy` — one of `{sequential_left, tree_fixed_fanout_N, segmented_fixed, pairwise_kahan, ...}`. Defaults are fine, but it must be *named*, not *inherited from the backend*.
- The IR verifier rejects any kernel whose order_strategy is not declared or is `backend_default`.
- The CPU interpreter executes the declared order strategy literally — no reordering. That becomes the ground truth.
- Each backend must implement every declared order strategy or refuse the kernel.
- The existing `reduce_block_add.f64` op from Peak 1's scope becomes `reduce_block_add_tree_power_of_two.f64` (explicit) or similar.

**Why this is high leverage:**

1. It makes I5 (no non-deterministic reductions) *derivable* from I7′, not an independent invariant to enforce.
2. It closes a loophole Peak 6 is spending a whole peak trying to close: "how do we guarantee the same reduction tree on every backend?" Answer: declare it in the IR.
3. It prevents a class of future bugs where a new backend (NPU, custom silicon) silently picks a different reduction tree for "optimization reasons."
4. It gives the Test Oracle a cleaner invariant to check: "same order_strategy → same bits, across backends." That's a structural check, not an empirical one.
5. It prevents the IR Architect from later discovering they need to retrofit total-order metadata into existing ops — doing it now is free; doing it in Phase 2 is a format bump.

**Why this is the *first-principles* move, not just a good idea:**

Because Phase 2 said total order is one of four necessary roles that any computation representation must fill, and accumulate+gather as written fills it *implicitly* (by choosing op to be associative). Implicit filling is a contract between the IR and the backend that isn't written down — exactly the kind of invisible assumption that gets violated when a new backend joins. Making it explicit is making a hidden truth into a stated truth.

**Recursion check — does this Aristotelian Move itself contain hidden assumptions?**

Yes. It assumes:
- That "total order" is itself a well-defined concept across all computation. For parallel scan (Kingdom A), yes. For iterative fixed-point (Kingdom C), "total order" has to include the convergence criterion and the iteration bound. For stochastic methods (MCMC, VI), "total order" has to include the seed and the RNG algorithm.
- That "order strategy" can be enumerated in advance. It probably can't, forever. New orderings (e.g., compensated summation variants, pairwise blocked, binary-tree-with-remainder) will keep being invented.

So the Aristotelian Move has its own recursion: it applies cleanly to Kingdom A today, but leaves a *tail* for the non-Kingdom-A cases. That tail is where Phase 6 of this deconstruction should go.

---

## Phase 6 — Recursion: challenge the Phase-5 Move itself

The Phase-5 Move is:
> Extract "total order" as a first-class concept in the .tam IR, separate from the dataflow pattern. Every accumulate declares an `order_strategy`; the verifier rejects `backend_default`; the CPU interpreter executes the declared order literally.

Adding this Move to the assumption list and running the Aristotle loop again.

### Phase 6.1 — Assumption autopsy on the Move

New assumptions, inside the Move:

**M1.** That "total order" is a well-defined unit of information — a thing the IR can *name*.

**M2.** That total order is *separable* from dataflow. That the two axes are orthogonal.

**M3.** That an enum (or named set) is the right data structure for carrying order information. Not a formula, not a proof term, not a reference to a first-class order-generating function.

**M4.** That `order_strategy` has bounded expressibility — that we can list the orderings that matter and reject the ones we haven't listed.

**M5.** That the CPU interpreter can execute *any* declared order literally. Some orderings (e.g. "the unique order that minimizes numerical error for THIS input") are defined by optimization over the input, not by the IR source alone; the interpreter would need to solve an optimization problem to execute them.

**M6.** That each backend can implement each declared order. A backend's hardware may fundamentally not support an ordering — e.g., a wavefront-based SIMT GPU can't naturally execute `sequential_left` on a vector of a million elements without serializing onto a single thread, which is a performance cliff, which is a pressure to relax the declared order.

**M7.** That declaring order in the IR doesn't over-constrain the compiler. If order_strategy is required, the compiler can't reorder for performance — but not reordering was the goal. However, if the compiler *cannot* do ANY reordering, it cannot fuse kernels (fusion is a reordering). The Move might break fusion.

**M8.** That order is *per-op*, not per-kernel or per-program. An entire kernel might have a single order strategy (left-to-right in source order) and individual ops inherit it. Or each op has its own. Or ops have defaults and the kernel overrides. Design space uncollapsed.

**M9.** That backends refusing an order_strategy is an acceptable failure mode. A new NPU that doesn't support `pairwise_kahan` would refuse kernels that declare it — even if there's a functionally equivalent ordering the NPU DOES support. The refusal is pedantically correct and operationally annoying.

**M10.** That the Test Oracle's "structural check" is actually a check of bit-exact reproducibility. It's checking: "both backends claim to have executed order_strategy X; do they agree?" But if the order_strategy is ambiguous in a corner (e.g. `tree_fixed_fanout_N` doesn't specify what happens when N doesn't divide the input size), the structural check passes while the actual bits differ. The Move shifts the ambiguity from the backend to the order_strategy spec — ambiguity isn't eliminated, it's relocated.

### Phase 6.2 — Irreducible truths visible at this level

1. **Order is a mathematical object, not a name.** A declared order is either a total order on a finite sequence (easy to represent: a permutation) or a procedure that generates a total order from some input description (hard: we need a procedure-naming mechanism).

2. **Procedures generating total orders form a space that cannot be finitely enumerated.** There are infinitely many ways to order `[a₀, ..., a_{n-1}]`. Most "practical" orderings are parameterized by a small strategy identifier (e.g., tree with fanout 2) plus implicit input dependencies (n, block size, grid size). But new strategies keep being invented.

3. **An enum is a closed representation; a named function in a registry is an open representation.** The Move's original framing said "enum." A registry (a set of named order-generating functions, each with a documented implementation) is strictly more expressive.

4. **Fusion is reordering.** If two accumulates are fused into a single pass, the fused version must produce bit-exactly the same answer as the two-pass version. This is possible iff the fused order IS the two-pass order — which is a constraint on what the fuser is allowed to do.

5. **Fusion is still desirable under the Move.** Most fusions don't actually reorder accumulates; they reorder *gathers* (loads) and interleave accumulate-steps. The fusion preserves the accumulate's declared order while eliminating redundant data movement. So the Move doesn't break fusion; it constrains it. That's different.

6. **A backend's refusal to support a declared order is a signal, not a failure.** If an NPU can't do `pairwise_kahan`, the right response is (a) document that the NPU can't, (b) let the caller choose: run on CPU/GPU that can, or declare a weaker order_strategy the NPU CAN handle. This is I6 territory ("no silent fallback for missing features") — the refusal is correct behavior.

7. **The Test Oracle's structural check becomes genuinely structural iff each named order_strategy has a formal specification that's bit-exact deterministic.** Under-specified strategies (like `tree_fixed_fanout_N` without a rule for non-dividing N) are bugs in the strategy spec, not bugs in the structural check.

### Phase 6.3 — Reconstructions of the Move itself

Given 6.2, how do we restate the Phase-5 Move to survive its own assumptions?

**Move v1 (original).** Enum `order_strategy` as a field on each accumulate op. Simple, limited.

**Move v2.** Open registry of named `OrderStrategy` functions, each with a formal spec and a bit-exact reference implementation. The IR carries a reference (name or content-hash) into the registry. Adding a new strategy adds a registry entry; old kernels keep working. This matches T3 in 6.2.

**Move v3.** Per-kernel default + per-op override. Every `.tam` kernel declares a default order_strategy at its entry; individual accumulates can override. The verifier checks that every op's effective order is a known registry entry. Matches M8 with a sensible hierarchy.

**Move v4.** Registry + formal specs + per-kernel default + compatibility metadata for fusion. When two accumulates fuse, the fuser consults the compatibility metadata on the two strategies: can strategy_A and strategy_B be interleaved without changing either one's bit output? If yes, fuse. If no, don't fuse, or emit two kernels. Matches T4 and T5.

**Move v5 (the one I'll recommend).** All of v4, plus:
- Every backend publishes a **capability matrix**: which registry entries it implements, bit-exactly, on which hardware. The matrix is part of the backend's contract. A kernel declaring an order the backend can't bit-exactly provide is rejected at compile time (I6 compliant), never silently relaxed.
- The CPU interpreter implements EVERY registry entry (it's the reference — it has to), which gives the Test Oracle an unambiguous ground truth for cross-backend diffs.
- New registry entries ship with: formal spec, reference implementation (in `.tam`, executable by the CPU interpreter), bit-exact test vectors, and compatibility notes for the fuser.

### Phase 6.4 — The refined Aristotelian Move

> **I7′ (refined):** Every primitive in .tam IR decomposes into a dataflow pattern (kingdom-classified) and a total order (referenced by name or content-hash into an open registry of `OrderStrategy` entries). Every OrderStrategy entry ships with: formal spec, reference implementation executable by the CPU interpreter, bit-exact test vectors, and fusion-compatibility metadata. Every backend publishes a capability matrix listing which OrderStrategy entries it implements bit-exactly; a kernel whose declared strategy is outside a backend's matrix is rejected at compile time, never silently relaxed.

This is the version to route to IR Architect. v1 (enum) is wrong in exactly the way Phase 6.1 suspected: it's closed, it fights fusion, and it shifts ambiguity instead of eliminating it. v5 preserves the spirit (order is first-class) while fixing the five assumptions the Move originally hid.

---

## Phase 7 — Stability check

Run the loop one more time, adding v5 to the assumption list.

**New assumptions to test:**
- Open registry vs closed enum — does "open" introduce its own problems?
- Formal specs — who writes them, who reviews them, what counts as "formal"?
- Capability matrices — who maintains them, what happens when hardware changes?
- Fusion compatibility metadata — is this another source of truth that can drift?

**Autopsy findings:**

- **Open registries have a coordination cost.** Two teams adding entries with slightly different meanings for "tree_reduce" would produce incompatible kernels. Solution: the registry is a single source of truth, and new entries require a spec review. This is one more thing to maintain, but it's concrete and one-time per strategy.
- **"Formal spec" is a word that can mean anything from "a paragraph in markdown" to "a Coq proof."** The pragmatic minimum is: a `.tam`-executable reference implementation + enough prose to let a human reviewer say "yes this matches the reference." Elevating to full formal verification is future work.
- **Capability matrices drift when hardware drivers update.** A Vulkan driver update could change the precision of a vector op and break a bit-exact claim without anyone noticing. Solution: the Test Oracle's cross-backend diff harness catches this automatically — when a declared matrix entry stops being bit-exact, the nightly run fails. Good.
- **Fusion compatibility metadata is self-testable.** The fuser generates both unfused and fused versions for a test kernel; the CPU interpreter runs both; if they agree, the metadata is correct for that example. Property-based testing of the fuser against the interpreter is a standard technique.

**Stability verdict:** Phase 6 collapsed v1 into v5. Phase 7 finds that v5's assumptions are answerable with conventional engineering — each has a concrete mitigation, and no new truths emerge. The deconstruction is **stable** at v5.

The remaining fragility is in human coordination (registry management, spec review), not in the architecture. That's acceptable and expected — no architectural move eliminates coordination cost, only shifts where it lives.

---

## Phase 8 — Forced Rejection

Forcibly reject the entire deconstruction. What if decomposition is not just non-irreducible, but *wrong as a framing*?

### The void: no decomposition at all

Suppose tambear has no "primitives," no "accumulate + gather," no kingdoms. Every kernel is an opaque function from (input buffer, parameters) to (output buffer). The contract is: the function is mathematically specified, and every backend that provides the function guarantees bit-exactness with the reference implementation. No decomposition. No compositional property. No shared intermediates. No `using()` plumbing.

What does the project look like?

- **Tambear becomes a certified kernel library** — like LAPACK, FFTW, or GSL, but with formal specs and bit-exact cross-hardware guarantees. Each function is individually verified against its spec. Composition of functions is a separate concern: the user composes functions, the library guarantees each function but not the composition.
- **TamSession content-addressing still works** — outputs are bitwise-hashable, so intermediate results can be cached across runs. But sharing across *different methods* would require an external registry of "these two methods use the same intermediate," which is manual.
- **The Tambear Contract's "every parameter tunable" clause becomes a nightmare** — each function exposes its own parameter space, and users have to learn each function's conventions. There's no unified `using()` because there's no composition to flow it through.
- **The specialist registry (135 specialists from 9 primitives) evaporates** — there are no primitives, so there's nothing to combine. Each specialist would be an independent kernel.
- **The gradient duality story dies** — "forward/backward is the same DotProduct transposed" relies on the underlying primitive being compositional. An opaque kernel can be manually differentiated but not automatically duality-exploited.
- **Fusion across methods becomes impossible** — you can't fuse two opaque kernels because you don't know their internal structure. Every kernel is a pass over memory, and you pay N passes for N methods instead of the 4 passes the current Phase-1 recipes achieve.
- **Cross-backend bit-exactness IS still achievable** — each backend implements each kernel to match a reference spec. The trek's central claim survives.
- **The product is MUCH slower on compound workloads** — because fusion is disabled, but benchmark-competitive on individual kernels.

### What forced rejection reveals

The project's *speed* story depends on decomposition + fusion. The project's *correctness* story does NOT — you can get bit-exact cross-hardware without any decomposition at all. If correctness is the #1 priority, decomposition is overkill. If speed is #1, decomposition is essential.

Tambear claims both. Decomposition is load-bearing for the *speed* axis, not the correctness axis. That's an important clarification: when the Tambear Contract says "accumulate + gather is the filter test," it's implicitly saying "we are optimizing for speed-via-fusion," which is a choice, not a necessity.

This doesn't invalidate the Move (v5). It *clarifies* what the Move is for. The refined I7′ gives us:
- Decomposition → fusion → speed.
- Declared total order → bit-exact → correctness.

Two axes, two benefits. The Move ties them together cleanly. Forced rejection confirms the Move is correctly targeted: it's the right structural upgrade for the existing (speed + correctness) architecture. If we rejected decomposition entirely, we'd give up the speed axis and keep only correctness. That's a different product, viable but not what's being built.

### The unseen first principle surfaced by forced rejection

**Decomposition is a speed story, not a correctness story.** The trek conflates them. I7's language ("every primitive decomposes") sounds like it's about correctness, but its actual load-bearing role is to enable fusion, which is a performance technique. Bit-exactness-cross-hardware is the correctness story, and it's cleanly separable from decomposition.

**Implication for the trek:** Peak 1's IR design doesn't need to be tied to "accumulate + gather" for the trek's *meta-goal* (bit-exact cross-hardware). The meta-goal is achievable with any IR that has enough precision to pin the op sequence. The reason Peak 1 uses accumulate + gather is performance — so the library can fuse Phase-1 recipes into 4 passes instead of N passes. That's a real reason, but it's not the reason the *trek* exists. Naming this cleanly would help the team separate trek-goals from library-goals.

---

## Status as of 2026-04-12

- Phases 1–8 drafted.
- **Final Move (v5):** Open registry of `OrderStrategy` entries, each with formal spec + reference implementation + bit-exact test vectors + fusion-compatibility metadata; per-kernel default with per-op override; backends publish capability matrices; CPU interpreter implements every registry entry.
- **Unseen first principle (from Phase 8):** Decomposition is a speed story; bit-exact cross-hardware is a correctness story. Tambear conflates them. The trek's meta-goal lives on the correctness axis; decomposition is there for performance.
- Second pass (Phase 7) found stability. No further recursion warranted.
- Ready to communicate the refinement to navigator.

---

## Addendum 2026-04-12 — Phase 2 gap: exception/special-value semantics is a fifth role

**Credit:** Adversarial Mathematician, via the Peak 4 baseline sweep (commit 35982d5). Found empirically through tbs::eval bugs (P17: `Sign(NaN) = 0`, P18: `Min(NaN, 5.0) = 5.0`, P20: `Eq` uses hardcoded `1e-15` epsilon).

My Phase 2 Truth 8 stated:
> Any representation of computation must somehow encode: (a) what operations to perform, (b) on what data, (c) in what order, (d) into what destination.

That enumeration is incomplete. There is a fifth role:

> **(e) exception/special-value semantics** — what does each op do when its input is NaN, ±Inf, ±0, or subnormal? How do these propagate through compositions? When does a semantic exception (divide-by-zero, overflow, invalid op) raise vs silently produce a special value?

IEEE-754 specifies defaults for most of these, but "the IR inherits IEEE-754 defaults" is exactly the kind of implicit contract I criticized accumulate+gather for leaving in the `op` parameter. If the IR doesn't **name** its special-value behavior, each backend may implement it slightly differently, and the bug shows up as "NaN propagation is argument-order-dependent in fmin/fmax" — the exact bug class adversarial found in tbs::eval.

This gap affects the I7′ v5 Move in one concrete way: the `OrderStrategy` registry entries should declare their special-value behavior alongside their order. An `OrderStrategy` that sums a vector containing NaN must either propagate NaN (IEEE-754 default), skip NaN (a la R's `na.rm = TRUE`), or error. These are three different strategies and they must be named registry entries, not implicit choices.

This pushes the registry entry schema from:

```
OrderStrategy {
  name, formal_spec, reference_impl, test_vectors, fusion_compat
}
```

to:

```
OrderStrategy {
  name, formal_spec, reference_impl, test_vectors, fusion_compat,
  special_value_behavior: {
    nan_policy:   Propagate | Skip | Error,
    inf_policy:   Propagate | Saturate | Error,
    signed_zero:  Preserve  | Collapse,
    subnormal:    Preserve  | FlushToZero   // backend-limited via capability matrix
  }
}
```

Each field is itself a choice. Each choice is documented. The verifier checks that the declared `special_value_behavior` matches the CPU interpreter's behavior on a NaN-containing reference input. Backends declare which combinations they support via the capability matrix.

**This is a genuine refinement to v5.** Adopting it doesn't invalidate anything in v5 — it adds detail the v5 schema didn't specify. I'm calling the refined version v5.1, not v6, because the skeleton is unchanged.

**Relationship to the meta-goal deconstruction (notebook 013):** In notebook 013's Phase 8 I proposed the three-precondition compositional claim: IR-precision, faithful-lowering, IEEE-754-compliance-for-the-ops-used. The fifth role (special-value semantics) is absorbed into precondition 3 — IEEE-754 specifies NaN/Inf/subnormal behavior, so "compliance for the ops used" includes "compliance with the special-value rules those ops invoke." ESC-001's Vulkan subnormal issue and adversarial's tbs::eval NaN bugs are the same precondition-3 failure class, wearing different clothes.

**What I should do next time:** Phase 2 enumeration of "truths" should be checked against a concrete example — e.g., "does the current IR handle `sum([1.0, NaN, 2.0])` correctly on every backend?" That test case, if asked at Phase 2 time, would have surfaced the missing role. Running concrete examples through the enumeration is a good discipline I didn't apply. Next deconstruction, I'll apply it.

---

## Phase 6 Deep Dive — Composition under Fusion

**Prompted by:** Navigator routing response, 2026-04-12. Navigator confirmed Phase 6 is worth deepening specifically on the composition-under-fusion question: "whose order wins when two ops fuse?" Navigator flagged this as "where I7 meets the fuse_passes machinery and the implicit assumption becomes most dangerous."

Navigator also sharpened a framing point worth recording: **the load-bearing insight of I7′ is naming the double duty of I7 (compositional property) and I5 (determinism contract) that the current `ReduceBlockAdd` is implicitly doing.** Whether that naming becomes a field on the op or a prose contract in the spec is a secondary engineering question; the primary win is that pathmaker makes the choice consciously rather than by default. This is navigator's peer-coordinator framing and it's right — my v5 full-engineering refinement is elaboration on top of the primary win, not the primary win itself. The primary win is the naming.

With that scoped, here's the deeper dive on composition-under-fusion.

### The scenario

Two accumulators live in the same kernel:
```
loop_grid_stride %i in [0, %n) {
  %v     = load.f64 %data, %i
  %acc1' = reduce_block_add.f64 %acc1, %v   ; declared SequentialLeft
  %acc2' = reduce_block_add.f64 %acc2, f(%v) ; declared TreeFixedFanout(4)
}
```

Pathmaker's fuser wants to compile this to one pass. Aristotle's I7′ says both accumulators have a declared order. What does the fuser do when the orders disagree?

### Five options

**Option A — Pick one, force the other to conform.**
The fuser picks SequentialLeft (say) and compiles `%acc2` as if it were SequentialLeft. The declared `TreeFixedFanout(4)` is silently ignored.

*Verdict:* **Wrong.** This is exactly the class of silent numerical behavior change I7′ was designed to prevent. SequentialLeft for `%acc2` may produce different bits than TreeFixedFanout(4) — and if the user declared TreeFixedFanout(4) for a stability reason, the fuser has silently broken their contract. Reject.

**Option B — Refuse to fuse.**
The fuser detects the disagreement and compiles to two passes. `%acc1` gets its own loop; `%acc2` gets its own loop. Speed regression.

*Verdict:* **Sound but conservative.** For Phase 1, this is a valid safe default. The recipes that currently compile to 4 passes might compile to more passes under strict I7′ — a real cost, but paid in performance, not in correctness.

**Option C — Carry both orderings within a single loop.**
Within the fused loop, `%acc1` ingests values in left-to-right order (matching SequentialLeft), and `%acc2` buffers values into blocks of 4 and reduces them in a separate internal tree (matching TreeFixedFanout(4)).

*Verdict:* **Works for specific combinations; doesn't work generally.** The test is: can the fused loop schedule produce bit-exactly the same bits for each accumulator as its declared order would in isolation?

For (SequentialLeft, TreeFixedFanout(k)) specifically, YES — the loop orders data left-to-right, `%acc1` consumes each element as it arrives, `%acc2` buffers k elements into a block, reduces them in the specified tree, and folds block sums according to the strategy. The two accumulators don't conflict because they consume the loop's data stream at different granularities.

For (TreeFixedFanout(4), TreeFixedFanout(8)) — both want out-of-order block grouping, but with different block sizes. In a single-pass loop, you can't run two different block schedules simultaneously without duplicating the data traversal. This combination forces Option B.

For (SequentialLeft, SequentialLeft) — trivially compatible.

For (SequentialLeft, PairwiseKahan) — PairwiseKahan is compensated summation, which consumes left-to-right and carries a running correction. Compatible with SequentialLeft at the loop level.

**Option D — Compatibility predicate on OrderStrategy.**
Each `OrderStrategy` registry entry declares a method `is_fusable_with(other: &OrderStrategy) -> bool`. The fuser calls this method; if true, fuses; if false, falls back to Option B. The predicate's semantics: "fusing these two accumulators in a single loop produces bit-exactly the same bits for each as running them in isolation."

*Verdict:* **This is the honest engineering shape.** It lets (SequentialLeft, TreeFixedFanout(k)) fuse when k is compatible with a left-to-right ingestion, lets (SequentialLeft, PairwiseKahan) fuse, refuses (TreeFixedFanout(4), TreeFixedFanout(8)). The predicate is specific to each pair of strategies and requires the registry to declare it.

Concrete default: `is_fusable_with(self, other) = (self == other)`. Pairs override for known compatibilities. Unknown pairs default to false (safe).

**Option E — Kernel-wide single order.**
Every `.tam` kernel declares ONE order strategy at its entry; all accumulators within the kernel inherit it. Fusion is trivial: everything uses the same order. Cross-kernel fusion is disallowed (or requires kernel merging with a compatible order).

*Verdict:* **Simplest, but loses expressivity.** A variance kernel that wants `PairwiseKahan` for Σx² (for numerical stability) and `SequentialLeft` for a count accumulator (trivially exact) has to declare one for both, which forces an over-conservative choice. Phase 1 could tolerate this; Phase 2+ will want something more.

### The recommendation for Phase 1

**Ship Option B as the default behavior, with Option D as the extension path.**

Concretely:
- The fuser, when it encounters two accumulators with declared orders, calls `is_fusable_with`.
- For Phase 1, the only registered strategies are `SequentialLeft` and `TreeFixedFanout(k)` (per navigator's minimum viable framing). Their compatibility relations are:
  - `SequentialLeft ↔ SequentialLeft` → fusable
  - `TreeFixedFanout(k) ↔ TreeFixedFanout(k)` → fusable (same k)
  - `SequentialLeft ↔ TreeFixedFanout(k)` → **not fusable in Phase 1** (we need to prove bit-exactness first; defer the proof to campsite 1.15 per navigator)
  - `TreeFixedFanout(k1) ↔ TreeFixedFanout(k2)` where k1 ≠ k2 → not fusable
- If not fusable, the fuser falls back to two passes. This is a performance regression *only* for kernels that have multiple accumulators with different declared orders, which Phase 1 recipes probably do not (all Phase 1 accumulators default to one order).
- Campsite 1.15 (the fusion compatibility work) is where Option D grows — it's the right place to prove `SequentialLeft ↔ TreeFixedFanout(k)` compatibility under specific conditions and register the broader predicate.

**This is a minimum that's safe and extensible.** It says "no" more often than Option D's full version, but every "yes" is provably correct. Expanding the "yes" set is future work with concrete proof obligations.

### Why this matters for Peak 6

Peak 6's determinism work currently plans to retrofit deterministic reductions on top of whatever Peak 3 emits. Under I7′ with Option B-or-D as the fusion rule, Peak 6's scope shrinks to:

- Implement each registered `OrderStrategy` bit-exactly on each backend.
- Publish the capability matrix (which backends support which strategies).
- The structural guarantee "same declared order → same bits cross-backend" follows from each backend's correctness for its implementations.

Peak 6 doesn't need to *design* determinism — it just implements what I7′ declared. The design moves from Peak 6 into Peak 1's campsite 1.15 (fusion compatibility) and into the registry itself.

**This is the load-bearing claim of the entire I7/I5 deconstruction.** Navigator named it correctly in their routing response: I7's implicit associativity assumption is doing double duty (compositional + deterministic). Under I7′ with the compatibility predicate, the two duties are separated cleanly — compositional property lives in the registry entry's semantics, deterministic property lives in the backend's capability matrix, and the compatibility predicate governs when both hold simultaneously under fusion.

### What this doesn't solve (honest scope)

- **Kingdom B/C/D operations still need their own story.** Sequential recurrence (Kingdom B), iterative fixed-point (Kingdom C), and stochastic methods (Kingdom D) aren't covered by `OrderStrategy` as formulated. They'll need parallel concepts: `RecurrenceStrategy`, `FixedPointStrategy`, `SamplingStrategy`. Each with its own compatibility predicate under fusion. Deferred to Phase 2+, per navigator's scoping.
- **Cross-kernel fusion** (the speed story where two kernels call the same primitive and share the work) is out of scope for Phase 1. When it lands, the compatibility predicate extends to cross-kernel queries.
- **The compatibility predicate is not transitive.** `A ↔ B` and `B ↔ C` do NOT imply `A ↔ C`. Each pair is independently declared. This is a maintenance cost but prevents the predicate from accumulating false positives through transitivity.
- **Proving `SequentialLeft ↔ TreeFixedFanout(k)` compatibility** is a non-trivial theorem obligation (when does the fused loop produce bit-exactly the same results as both strategies in isolation?). Campsite 1.15 is the right place for that proof. For Phase 1, we declare the pair non-fusable and pay the perf cost.

### The refined Move — I7′ v5.2

> **I7′ v5.2 (composition-under-fusion):** Every `OrderStrategy` registry entry declares a `is_fusable_with(other: &OrderStrategy) -> bool` predicate. The fuser calls this predicate when deciding whether two declared accumulators can run in the same pass. Default predicate: `self == other`. Broader compatibilities are declared per-pair with a written proof obligation. For Phase 1: SequentialLeft and TreeFixedFanout(k) are only self-fusable; mixed pairs fall back to multi-pass. Campsite 1.15 owns the proof-of-compatibility work for broader predicates.

This is additive to v5.1. Same skeleton, one more field per registry entry, one more fuser method.

### Relationship to navigator's MVP framing

Navigator's minimum viable framing is `order: OrderStrategy` on `ReduceBlockAdd` with `SequentialLeft` and `TreeFixedFanout(u32)` as Phase 1 strategies. My v5.2 adds: the compatibility predicate lives in the strategy definitions, and the fuser consults it. Navigator's MVP is compatible with this — the fuser calls the predicate at fusion time, and for Phase 1 with only self-fusability, the predicate is trivial (`self == other`). The MVP doesn't need to implement the broader predicate machinery now; it just needs to leave the door open.

The door-leaving-open is the critical part. If pathmaker implements `ReduceBlockAdd` with `order: OrderStrategy` as a bare enum field (no associated predicate), the fuser will hit the question eventually and have no answer. If pathmaker implements it with a predicate hook (even if the only initial predicate is `self == other`), the expansion path is clean.

**Concrete ask for pathmaker** (routed through navigator): when implementing the `OrderStrategy` type, make `is_fusable_with` a method (even if trivially defined). That preserves the extension path without adding Phase 1 work.

---

## Status as of 2026-04-12 (second update)

- Phases 1–8 drafted + v5.1 addendum on special-value semantics + Phase 6 deep dive on composition-under-fusion.
- **Final Move (v5.2):** v5.1 + `is_fusable_with` predicate on every `OrderStrategy` registry entry. Default `self == other`. Broader pairs declared with proof obligations. Phase 1 ships self-fusability only.
- **Navigator's MVP framing accepted:** The load-bearing insight is naming the I7/I5 double duty explicitly. The engineering shape (enum vs registry) is pathmaker's call. v5.2 is fully compatible with a simple enum field IF the type has a `is_fusable_with` method from the start.
- **Concrete ask to pathmaker via navigator:** make `is_fusable_with` a method on `OrderStrategy` from day one, even if trivially defined. Preserves the extension path.
- Peak 6 scope implication: determinism becomes a capability-matrix question for each backend, not a design question. This is correct under v5.2.
- Campsite 1.15 ownership: the fusion compatibility proof obligations live there. When the recipe library wants (SequentialLeft, TreeFixedFanout(k)) to fuse, campsite 1.15 is where the proof happens.
