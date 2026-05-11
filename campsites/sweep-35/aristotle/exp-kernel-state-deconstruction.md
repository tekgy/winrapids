# ExpKernelState + complementary-argument-transform — Phase 1-8 deconstruction

**Author**: aristotle, tambear-sweep35
**Date**: 2026-05-10
**Lane**: Sweep 35 main, task #12 (created from task #5)
**Pair**: pathmaker (before they lock Phase B design)
**Status**: Phase 1-5 first pass; Phase 6-8 in subsequent revisions

**Substrate consulted**:
- `R:\winrapids\docs\architecture\tambear-libm-factoring.md` (the synthesis doc)
- `R:\winrapids\docs\architecture\holonomic-architecture.md` (the cache-discipline test)
- `R:\winrapids\docs\architecture\internal-tameness-contracts.md` (the audit pattern)
- `R:\winrapids\docs\architecture\branch-cut-conventions.md` (DEC-032 BranchPolicy)
- `R:\tambear\oracle\tan\followups-rederived-2026-05-09.md`
- `R:\winrapids\crates\tambear\src\recipes\libm\sin.rs` (OLD codebase reference — see "Substrate finding 0" below)

---

## Substrate finding 0 (precondition to deconstruction)

**TrigKernelState is not shipped in `R:\tambear\`.** It is referenced in `tambear-libm-factoring.md` as "already implemented (TrigKernelState in past-Claude's expedition)". On disk:

- `R:\tambear\crates\tambear\src\recipes\` — no trig recipes; no `TrigKernelState` symbol; no `payne_hanek`.
- `R:\winrapids\crates\tambear\src\recipes\libm\sin.rs` — IS implemented as a function `reduce_trig(x) -> (i32, f64, f64)` returning a tuple, with Cody-Waite + Payne-Hanek strategies. **There is no struct `TrigKernelState`. There is no TamSession registration. There is no content-addressed cache key.**

The TrigKernelState named in the libm-factoring doc is conceptual — it abstracts the `(q, r_hi, r_lo)` tuple as if it were a struct registered as a shareable intermediate, but the actual code shape is a function returning a tuple and getting recomputed per call.

**Implication for the deconstruction**: when the libm-factoring doc says "ExpKernelState is the analog of TrigKernelState", the analogy load-bears two different things:
- (a) The *factoring shape* — a reduced argument + a precision-safe core + an inverse transform. True of trig (in code).
- (b) The *shareable-intermediate-via-TamSession shape* — a struct with a content-addressed cache key. **NOT true of trig in code.** Trig recomputes per call.

The proposed ExpKernelState would be the *first actual instance* of "kernel state as TamSession-registered shareable intermediate," not the second. The first cell of a new pattern, not the parallel growth of an existing one. **This is load-bearing for the deconstruction** because every "we already proved this works for trig" claim in the briefing-substrate is actually "we drew it on paper for trig, but the code path is per-call recomputation."

---

## Phase 1 — Assumption autopsy

Inherited assumptions embedded in the ExpKernelState proposal:

### A1. The (k, r, expm1_r) triple is the right state shape

> `pub struct ExpKernelState { k: i32, r: f64, expm1_r: f64 }`

**Inherited from**: symmetry with TrigKernelState's `(q, r_hi, r_lo, s, c)`. Symmetry alone is not evidence.

### A2. expm1 is the precision-safe core

**Inherited from**: past-Claude's April 13 *complementary-argument* essay. The essay is correct that subtracting 1 from a near-1 quantity is catastrophic; expm1 = exp(x) - 1 computed *as if* the subtraction were never structural is the standard libm trick.

**The hidden assumption**: expm1 is the *unique* precision-safe core. There are alternatives — e.g., `2^r - 1` (a "base-2 expm1"), `exp(r) - exp(0)` viewed as a divided difference, or a Padé/continued-fraction representation that bypasses cancellation differently. The proposal assumes one shape.

### A3. Reduction is "k * ln(2)" — Tang style

> `r: f64, // reduced argument: x - k * ln(2)`

**Inherited from**: Tang's 1989 paper (which MSVC degrades by independently doing the k-multiplier reduction per function). The assumption is that the same reduction works at every precision tier.

**The hidden assumption**: ln(2) is a single constant. At `P0F64` it's `0x3FE62E42FEFA39EF` plus low part `0x3C7ABC9E3B39803F`. At `P2BigFloat{1024}` it's a 1024-bit value. A reduction parameterized on the constant's precision is structurally different from a reduction parameterized on the constant's *value at a specific tier*.

### A4. Cache key = `(x_bits, precision_context_tag_bytes)` is content-addressable

**Inherited from**: holonomic-architecture.md's rule for recipe-tier intermediates.

**The hidden assumption**: `x_bits` is well-defined. For `f64`, yes. For BigFloat, "the bits of x" is a serialization choice — and serialization choices depend on canonicalization (which is itself the source of multiple BZ bugs per `internal-tameness-contracts.md`). The cache key composes a kernel-state-precondition (the input must be canonical) with the kernel-state-content. If the precondition is non-fingerprintable, the holonomic claim fails silently.

### A5. The k step is "exact" via bit-shift

> `exp(x) = (1 + expm1_r) << k` ... "Bit-shift is exact"

**Inherited from**: floating-point folklore. **True for `f64` IEEE 754 normal range.** Not true at subnormal boundary, not true for `k > 1023`, not true if `1 + expm1_r` is itself denormalized, not true under directed-rounding modes if the "exact" bit-shift produces a value that the surrounding code path then re-rounds.

This is **F13.C-shaped**: an "exactness" precondition that holds for the input subspace the algorithm was designed for, with no antibody at the boundary. The `k` value comes from `round(x / ln(2))`; for `x` near `f64::MAX`, `k ≈ 1024`. Multiplying by `2^1024` overflows `f64` even if the result `(1 + expm1_r) << k` is *mathematically* representable as a different exponent assignment — because the IEEE bit-shift operation doesn't fuse.

### A6. The same struct serves every consumer in the family

> `exp(x) = ..., sinh(x) = (exp(x) - exp(-x))/2, hypot(a,b) = ...`

**Inherited from**: the framing "factor out the shared step." But sinh(x) needs *both* expm1(x) and expm1(-x), or equivalently exp(x) and exp(-x). With a single ExpKernelState(x), sinh has to *also* compute ExpKernelState(-x), or the struct has to carry the negated companion.

The proposal as written carries one direction of the reduction. The consumer-side composition then needs:
- `sinh, cosh, tanh`: companion ExpKernelState(-x)
- `pow`: ExpKernelState(y · log(x)), nested
- `hypot`: not obviously derivable from ExpKernelState at all (see Phase 6 of libm-factoring's open questions)

### A7. The complementary-argument transform is a single meta-primitive

> Past-Claude's April 13 essay names log1p, expm1, sinpi, hypot, cosm1 as instances of one pattern: `complementary_arg_transform(x; F, G) = inverse(stable_at_transform_to_distance_from(x, F, G))`.

**Inherited from**: the recognition that they share a *precision-preservation property* under cancellation.

**The hidden assumption**: sharing a property means sharing structure. The property is "the fixed point F has a precision-preserving form near it." The structure is "transform-distance, stable-eval, inverse-transform." But:

- For `log1p` and `expm1`, the fixed point is **on the input side** (input is "1+ε" or "0+ε", the transform extracts ε).
- For `sinpi` and `tanpi`, the fixed point is **on the input side as a scaling** (input is "π · ε", transform extracts ε from a different unit).
- For `cosm1` and `sinm1`, the fixed point is **on the output side** (subtract 1 from the result; the cancellation lives at the result, not the input).
- For `hypot`, there is no obvious fixed point — there's a *scale issue* (`a² + b²` overflows even when `√(a²+b²)` doesn't), addressed by `max(|a|,|b|) · √(1 + (min/max)²)`. That's not a complementary-argument transform; it's a different precision technique (homogeneous-scaling).
- For `gamma` (Lanczos), there's a similar "no fixed point" problem; Lanczos is a *different functional form*.

So "complementary-argument transform" may be **at least three distinct shapes** wearing one name:
1. Input-side fixed point with translation transform (log1p, expm1)
2. Input-side fixed point with scaling transform (sinpi, tanpi)
3. Output-side fixed point with translation transform (cosm1, sinm1)

And hypot/gamma/Lanczos are not instances of any of the three — they're *different precision concerns* (homogeneous scaling, asymptotic-series-around-pole) that got bundled together because they share the *outcome* (precision preserved) rather than the *mechanism*.

### A8. ExpKernelState is content-addressed; placement is provenance-addressed

**Inherited from**: holonomic-architecture.md's tier separation.

**The hidden assumption**: the kernel-state computation IS path-independent. The kernel state's value is "expm1(reduced_arg) under reduction policy R." If reduction policy R is itself a parameter (e.g., Cody-Waite vs Payne-Hanek vs higher-order), then the state's value depends on policy, and the cache key must include the policy. This is fine in principle (add the tag byte). But **at extreme x, the policy choice itself can change the meaning of the state** — Cody-Waite produces garbage for `x > 2^20 · ln(2)`; Payne-Hanek-analog isn't defined for exp/log (no "2/π table" analog at large exp arguments). The cache key would tag the policy, but the policy is not a free parameter — it's a consequence of input magnitude.

If the policy is determined by input magnitude, the cache key tagging is redundant with `x_bits` and the content-addressing works. If the policy is *also* selectable for cross-validation purposes (run both and compare), the cache key needs tagging. **Decision pending; surface to pathmaker.**

### A9. The recipe wrapper handles the "rest" cheaply

> `exp(x) = (1 + expm1_r) << k`

**Inherited from**: the factoring frame's promise that the per-recipe cost is small.

**The hidden assumption**: the per-recipe cost is *uniform* across recipes. For `exp` and `expm1`, it's a single operation. For `sinh`, it's a subtract + divide. For `tanh`, it's two subtracts + a divide + a divide. For `pow`, it's a multiplication + a log + an exp — the "wrapper" is half the computation. The factoring frame is *most* powerful when wrappers are O(1) per consumer; for `pow` the wrapper is O(kernel-state).

### A10. The expm1 polynomial is the *minimax* polynomial

**Inherited from**: past-Claude's `~/.claude/garden/2026-04-13-the-trig-bundle.md` says "minimax polynomials" without commitment to *which* minimax. The sin.rs file at `R:\winrapids\crates\tambear\src\recipes\libm\sin.rs` uses Remez exchange in mpmath at 80-digit precision — a specific minimax-discovery algorithm. The exp/log family hasn't been similarly committed.

**The hidden assumption**: "minimax" is uniquely defined. It is not — minimax-error in absolute terms is different from minimax-error in relative terms, and Chebyshev approximation is yet a third thing. For exp/log near 0, *relative* error is the natural metric (because the function value itself spans many orders); minimax-absolute would over-tolerate at large input and under-tolerate at small.

### A11. The kernel state is computed *once*

**Inherited from**: the TamSession sharing model.

**The hidden assumption**: every consumer is willing to *wait* for the kernel state. If `exp(x)` triggers computation of the state, and `sinh(x)` is the second consumer, sinh blocks on exp's computation. In a pipeline with parallel-leaf evaluation, this serializes leaves that would otherwise be independent. The IR-layer's placement decision needs to handle this — but **the recipe-tier cache discipline doesn't model time**. The "compute once" promise is *correctness*, not *latency*; latency is the IR's problem.

This is a latent assumption that the IR layer can route around. Recording for the IR-tier deconstruction (out of scope for Phase B).

### A12. ULP budget is preserved through the wrapper

**Inherited from**: the implicit expectation that "ULP of result = ULP of kernel state + ULP of wrapper."

**The hidden assumption**: ULP composition is additive. **It is not in general.** For `exp(x) = (1 + expm1_r) · 2^k`, the relative error in `expm1_r` becomes a relative error in `1 + expm1_r` (which is *additive* in absolute terms but *changes regime* near k=0 where 1+expm1_r ≈ 1). For `tanh = sinh/cosh = (exp - 1/exp)/(exp + 1/exp)`, the division near tanh ≈ 0 amplifies relative error; near tanh ≈ ±1 it cancels. **ULP-budget bookkeeping is per-recipe, not inherited from the kernel state.**

This is the trig-side question (1) from `tan/followups-rederived-2026-05-09.md` reappearing for exp/log: "what's the right ULP-budget shape per regime?" — and the answer is **not** "inherit from kernel state."

### A13. The "every parameter tunable" contract applies to ExpKernelState's parameters

The struct has no declared parameters. The reduction strategy, the polynomial choice, the precision contract — these are all *implicit*. Per Tambear Contract item 4, every knob a domain expert would touch must be a parameter with a documented default.

**Hidden parameters that need surfacing**:
- `reduction_strategy`: Tang vs Cody-Waite-analog vs alternate
- `polynomial_form`: minimax-rel vs minimax-abs vs Chebyshev vs Padé
- `polynomial_degree`: e.g., expm1 polynomial of degree 5, 6, 7, ...
- `precision_contract`: P0F64 vs P1DD vs P2BigFloat (auto-routed via PrecisionContext, but the *choice point* itself is a parameter)
- `branch_policy_for_real_axis_edges`: even on the real-valued exp/log, there are corners (e.g., log(0) = -inf vs log(-0) = -inf+iπ if extended to complex; this is *not* a complex-only question)
- `ln2_constant_precision`: how many bits of ln(2) does the reduction use?

The current struct hides all six. The Tambear-Contract-graded version of the struct would carry them as `using()` parameters with defaults documented.

### A14. The proposal generalizes to other transcendental families

The libm-factoring doc lists `gamma` and `Lanczos` as "different shapes" — open question 3. The implicit assumption in the Phase B → Phase D progression is that ExpKernelState's success is *evidence* that gamma will fit the same pattern. **It is not.** Gamma's precision problems are around poles (reflection identity), around 1 and 2 (where Γ = 1), and at integer arguments (combinatorial blowup). None of these is structurally a "shared reduced argument with stable core." Treating ExpKernelState as the template for gamma is the YAGNI mirror-image error — over-generalizing from one instance.

### A15. "kernel state" is the right level of abstraction

The kernel state is a *struct*. An alternative is a *protocol* (a trait): every member of the family declares "what reduction I need + what core I pull from + what inverse I apply." The IR/TAM layer then *constructs* a state object per pipeline context. The struct-first design assumes the state shape is uniform across the family.

The trait-first design would absorb the cases where the state shape differs (sinh needing both directions, pow needing nested state, hypot needing scaled state). The struct-first design hits these as "special cases" with ad-hoc fields.

---

## Phase 2 — Irreducible truths

Strip every assumption. What remains as undeniably true about the exp/log family precision problem?

### T1. exp(x) has dynamic range that exceeds f64 representable range

For x > ~709, exp(x) overflows f64. For x < ~-745, exp(x) underflows. **The range-reduction problem is real and structurally necessary** — not because of algorithmic preference, but because the output type is finite.

### T2. log(x) has unbounded gradient at x = 0 and at x = 1

`d/dx log(x) = 1/x`. At x = 1, the derivative is finite but `log(1 + ε)` has cancellation against the unit value (this is the "near-1" precision regime). At x → 0+, log(x) → -∞ and any rounding of x compounds catastrophically into the result. **Two distinct precision regimes**, structurally non-collapsible.

### T3. exp and log are inverses on (0, ∞) ↔ ℝ

`log(exp(x)) = x` and `exp(log(x)) = x` to within the precision of the implementations. **This is the symmetry that makes "factoring" possible** — they share an inverse-function relationship, so the same reduction discipline works on both.

### T4. Computing the inverse via "compose with the other" is precision-lossy

`pow(x, y) = exp(y · log(x))` loses precision in the multiplication step even if both `log` and `exp` are correctly rounded. For `pow(1+ε, n)` with small `ε` and integer `n`, the answer is `1 + nε + O(ε²)`; computed via the composed form, the relative error compounds. **This is structurally true; no implementation discipline eliminates it.**

### T5. The reduction `x = k · ln(2) + r` with `|r| < ln(2)/2` is correct iff k is the unique integer and r is computed exactly

This is a *mathematical identity*, not an algorithm. The algorithm's job is to find k (the rounding step) and r (the subtraction step). Both are subject to rounding. **The "exact reduction" is not literally exact in f64 — it's exact only when the chosen representation can hold `x mod ln(2)` exactly.**

### T6. The polynomial approximation of expm1 on a bounded interval has a minimum achievable error

The Weierstrass approximation theorem guarantees polynomial approximation exists; the Remez algorithm finds the minimax polynomial of fixed degree. **There is a function-dependent floor on polynomial precision per degree.** For expm1 on `[-ln(2)/2, ln(2)/2]`, degree-7 minimax-rel polynomial achieves ~10^-16 relative error (below 1 ULP in f64). Higher precision tiers need higher-degree polynomials.

### T7. expm1(r) and log1p(r) for small r are independent of any reduction strategy

These are the *cores*. At small `r`, no reduction is needed — the input is already in the precision-safe regime. Therefore: **the precision-safe core is logically prior to the reduction.** Reduction exists to bring large `x` into the regime where the core works; it doesn't define the core.

### T8. Sharing intermediate results is correct iff the sharer and the sharee agree on the meaning of the intermediate

Per Tambear Contract item 3 (with compatibility enforcement): two consumers can share a cached value iff the metadata fingerprints match. **This is the only correctness invariant on the cache.** Same shape + same dtype is not enough; same *meaning* is required.

### T9. The pipeline-level placement of the kernel-state computation is path-dependent

Per holonomic-architecture.md: the IR-tier discipline is provenance-addressed. **Whether the state computes inline, hoisted, fused, or recomputed depends on the global pipeline.** This is not subject to revision by the kernel-state struct design.

### T10. F13.C-shaped antibodies require enforcement at every signature

Per `internal-tameness-contracts.md` and F13.C ratification: an antibody graduates from local pattern to structural invariant only when required at every call site. **Any precondition on ExpKernelState's input must be enforced wherever ExpKernelState is constructed.** Single construction site (a constructor) makes this single-sited (F13.A-shaped). Multiple construction sites make it multi-sited (F13.C-shaped).

### T11. The recipe tier is content-addressed iff the parameter bag is fingerprintable

Per holonomic-architecture.md + the path-independence test + the cache-key serialization. **If a parameter slot resists clean tagging, it's a signal of non-holonomic structure** — that parameter belongs at the IR tier, not the recipe tier.

### T12. The Tambear Contract's 10-point filter test applies to ExpKernelState

It is a primitive being shipped. It must pass the filter. **No exceptions for "internal kernel state".**

### T13. Honesty about what's shipped vs designed is load-bearing

Substrate-over-memory (per global CLAUDE.md): the libm-factoring doc's claim that "TrigKernelState already shipped" is a context-state error, not a substrate fact. **The deconstruction work proceeds from substrate, not from briefing-text.**

---

## Phase 3 — Reconstruction from zero (10 approaches, simple → impossible-seeming)

Rebuilding the exp/log precision problem from T1-T13 alone, with no prior approach as scaffold.

### R1 (simplest) — Per-function implementation, no factoring

Each named function (`exp`, `log`, `sinh`, ...) has its own implementation file. Each does its own reduction, its own polynomial, its own output composition. No shared state. Per-function oracle validation per the Tambear Contract.

**Tradeoff**: maximally simple per-recipe; maximally redundant in aggregate (15+ functions repeat the same reduction). MSVC's pattern; produces the Tang-degradation we're trying to fix.

**Why I'd choose this**: never. T1 + T8 + the Sweep 34 finding (MSVC's 280 ULP) make this strictly inferior.

### R2 — Function-tuple intermediates, no cache

Each function still has its own entry point. The reduction is a shared function `reduce_to_core(x) -> (k, r)` and the core is a shared function `expm1_core(r) -> f64`. No struct, no cache; just function-level sharing.

This is what `R:\winrapids\crates\tambear\src\recipes\libm\sin.rs` actually does for trig (function `reduce_trig(x)` returning a tuple).

**Tradeoff**: the function-call boundary is the unit of sharing. Two consumers of `expm1_core(r)` recompute it — the compiler may CSE if the call is inlined, but cross-pipeline reuse is not possible.

**Why I'd choose this**: low-cost first step. **The trig family in winrapids/crates is at this level.** The Tambear Contract item 3 (TamSession sharing) is unmet here, but item 2 (accumulate+gather decomposition) doesn't require caching.

### R3 — Struct + TamSession registration, content-addressed

The libm-factoring proposal. `ExpKernelState { k, r, expm1_r }` registered under content-addressed key `(x_bits, precision_context)`. Every consumer pulls via `TamSession::get_or_compute`.

**Tradeoff**: solves cross-pipeline reuse and makes the sharing explicit. Cache-key fingerprint correctness is now *load-bearing* — bugs in serialization become silent correctness bugs (per A4).

**Why I'd choose this**: the proposed approach. Want to pressure-test it further before locking.

### R4 — Trait-based kernel-state protocol, struct-per-family

Define a trait:
```rust
trait KernelState {
    type Input;
    type Output;
    fn reduce(input: Self::Input, ctx: &PrecisionContext) -> Self;
    fn fingerprint(&self) -> CacheKey;
    fn core_value(&self) -> Self::Output;
}
```

Implement `ExpKernelState`, `LogKernelState`, `BidirectionalExpKernelState` (for sinh/cosh/tanh), `HypotKernelState`, ... as concrete types satisfying the trait. The shared discipline is the *protocol*, not the struct shape.

**Tradeoff**: handles A6 (sinh/cosh need both directions; hypot doesn't fit the simple struct) by letting different family members declare different state shapes. Cost: one more level of abstraction. Benefit: structurally honest about the diversity within the "complementary-argument" family.

**Why I'd choose this**: when the per-family-member state-shape diversity becomes load-bearing. **Phase 5 candidate.**

### R5 — Stateless protocol, computation in the recipe

Each family member declares "I am a *(reduction, core, inverse)* triple." The recipe machinery constructs the implicit pipeline `reduce → core → inverse` at call time. No persistent state object; the cache works at the level of the *intermediate values* `(k, r, expm1_r)` whether or not they're bundled in a struct.

**Tradeoff**: eliminates the struct as a first-class artifact. Pure-functional view. Loses the ability to share state *across* family members within one pipeline pass unless the IR layer does CSE on intermediate values (which is the IR's job anyway).

**Why I'd choose this**: if the kernel-state struct is purely an organizational device, not a runtime artifact. Connects to T9 (placement is IR's job) — maybe the recipe tier shouldn't even *try* to manage sharing; let the IR do it.

### R6 — Reduction-as-monoid, core-as-element

The reduction `x = k · ln(2) + r` defines a homomorphism into `(ℤ, +) × ([-ln(2)/2, ln(2)/2], +)`. The "kernel state" is then an element of that target group, and `exp` is a group homomorphism back: `exp(k · ln(2) + r) = 2^k · exp(r) = 2^k · (1 + expm1_r)`.

The factoring is then *categorical*: every exp/log family member is a natural transformation from the input to the output, factored through this monoid. The "kernel state" is the image of x under the reduction-monoid map.

**Tradeoff**: mathematically clean; computationally indistinguishable from R3 at the f64 level. The categorical view buys precision *only if* it surfaces invariants that f64-level coding doesn't. **Possibly: the reduction monoid's homomorphism property is the formal version of "k step is exact" (A5) — and the failure modes at the boundary of f64 are exactly where the homomorphism breaks (k outside ℤ-image, r outside the interval). Naming this gives a tighter precondition.**

**Why I'd choose this**: when math-researcher wants a publication-grade framing. Otherwise, equivalent to R5 in code.

### R7 — Holonomic-by-construction: kernel state IS the cache key's input bag

Build the struct so that the *struct value* is the cache key. The struct is `ExpKernelState { tag_bytes: [u8; N] }`; deserialization gives `(k, r, expm1_r)`. Construction *enforces* the path-independence — there is no path-dependent way to reach a state value, because the state IS the serialized form.

**Tradeoff**: serialization-first design. Eliminates A4's risk by making the serialization the *only* representation. Cost: every internal use must deserialize. Benefit: structural enforcement of the holonomic property.

**Why I'd choose this**: when the cache-key correctness becomes load-bearing for downstream operations (e.g., if the kernel state is used as a sub-component of a larger cache key — pow's kernel state composing log + exp states would need this).

### R8 — Multi-precision-aware kernel state with explicit precision-contract carrier

Drop A3's hidden assumption that `r: f64`. The struct becomes:
```rust
pub struct ExpKernelState<P: Precision> {
    k: i64,                          // expanded from i32 — large k at low-precision input
    r: P::Representation,            // f64 at P0; (f64, f64) DD at P1; BigFloat at P2
    expm1_r: P::Representation,
    reduction_strategy: ReductionTag, // antibody for A8
    ln2_precision_bits: u32,          // antibody for A3
}
```

Every parameter explicit. Every precondition tagged. The struct *is* the parameter bag; the cache key is BLAKE3 of the bag.

**Tradeoff**: large struct. More fields = more cache-key fingerprint bytes = more cache-key bugs possible. But each field is a *named precondition* that can be checked at construction.

**Why I'd choose this**: the Tambear-Contract-graded version of R3. **Phase 5 candidate.**

### R9 — Kernel state as an IR-level cache entry, not a recipe-level struct

Per holonomic-architecture.md: the IR layer is provenance-addressed. **What if the kernel state lives at the IR tier, not the recipe tier?** The recipe tier declares "I need expm1 of (reduced) x"; the IR builds the cache, places the computation, handles sharing. The recipe doesn't know about ExpKernelState; it only knows about expm1, log1p, and the inverse transforms.

The recipe tier stays content-addressed (each recipe's cache key is its own parameter bag). The kernel-state caching is *invisible* to the recipe — it's a feature of the IR's optimization pass.

**Tradeoff**: cleaner tier separation (per the holonomic doc's letter). Costs: the IR layer must be smart enough to recognize cross-recipe sharing opportunities; the kernel-state struct becomes an IR artifact, not a public API.

**Why I'd choose this**: **this is the deepest reconstruction.** It says the libm-factoring doc's framing has the wrong tier. The shared intermediate isn't a recipe-tier object; it's an IR-tier optimization. The recipe tier just declares "I need expm1(r)"; the IR tier sees that 6 recipes in the pipeline all need expm1(r_i) for related r_i and fuses the computation. **This is what the cliffhanger sentence at `important-conversation.md` line 1062 is asking for** — primitives that share an accumulator but differ in gather, with the IR deciding placement.

**Why this is "borderline insane"**: it requires the IR layer to do cross-recipe optimization that doesn't exist yet (per `holonomic-architecture.md` § "What the IR layer has (not 'needs')" — the machinery exists in principle but the formalization is partial). It defers Sweep 35's deliverable from "kernel state struct + recipe wrappers" to "kernel-state-aware IR pass + recipe declarations." That's a different sweep — but it may be the *right* sweep.

### R10 (most impossible-seeming) — No factoring; per-input symbolic representation

Don't reduce at all. Don't approximate at all. Build `exp(x)` as a *symbolic expression* that evaluates lazily at the precision the consumer demands. For `pow(x, y) = exp(y · log(x))`, the symbolic representation captures the chain; the leaf precision is determined by the consumer's ULP budget.

This is **interval-arithmetic-on-steroids** combined with **lazy evaluation**. The kernel state isn't a struct; it's a node in a computation graph that materializes when forced.

**Tradeoff**: monstrous in implementation cost. But it's the *only* approach that decouples per-precision implementation effort from precision-tier count. Add a new precision tier (P3, P4, P5) and the symbolic representation handles it for free.

**Why this is "borderline insane"**: this is **NOT YAGNI**. The project's roadmap has `P2BigFloat{1024}` and points toward arbitrary precision. The symbolic representation is the structurally-guaranteed need at the limit. **The naturalist's "every function we ship at publication grade is a contribution to the factoring" generalizes to "every function we ship at publication grade is a contribution to the symbolic graph."** At infinite precision, there is no factoring (no approximation polynomial) — there is only the symbolic identity.

**Preliminary first principle this would need that we don't yet have**: a **precision-bounded lazy evaluation contract**. A node in the symbolic graph commits to *deliver* its value at requested precision P with worst-case effort E(P). The IR layer schedules evaluation to amortize effort across consumers.

**Find it**: the preliminary principle is **precision-on-demand-with-cost-amortization**. It's the missing piece that connects:
- The complementary-argument-transform (precision-preservation at fixed points)
- The kernel-state factoring (sharing reduction work)
- The symbolic graph (deferring evaluation to consumption time)
- The IR-tier provenance-addressed cache (lineage-aware reuse)

All four are *moves on the same axis*: how to deliver requested precision at minimum effort, given that the consumer's precision request is part of the cache key. This is the **"precision is a runtime parameter, not a compile-time choice"** principle that future-tambear needs.

The reconstruction at R10 says: the right Sweep 35 work, if we believed in the limit, is to **build the precision-on-demand contract** (even at toy scale) and let ExpKernelState be the first instance. ExpKernelState then isn't a struct — it's an *implementation* of the contract for the exp/log family, parameterized by precision request.

---

## Phase 4 — Assumption-vs-truth map

| Original assumption | What replaced it (after Phase 2-3) |
|---|---|
| A1: (k, r, expm1_r) is the right state shape | T1+T6+R8: state shape is precision-parameterized; the f64-tuple is one cell of a family |
| A2: expm1 is the unique precision-safe core | T6+T7: expm1 is *one* core; alternative bases (2^r-1), alternative forms (Padé), and alternative reductions are all valid cores per the same factoring frame |
| A3: Reduction = "x = k · ln(2) + r" with ln(2) as a single constant | T5+R8: reduction is parameterized on ln(2)-precision per tier; the "constant" is a *function of precision context* |
| A4: Cache key = `(x_bits, precision_context)` is well-defined | T11+R7: cache key is the *serialized parameter bag*; correctness depends on canonical serialization, which is a precondition with antibody (F13.C-shaped) |
| A5: "Bit-shift by k is exact" | T1+T10: exact within IEEE-normal range; needs an antibody at the boundary (k near ±1023 for f64) — the precondition is multi-sited (every consumer of `(1 + expm1_r) << k`) |
| A6: Same struct serves every consumer | R4+R5: family-member state shapes differ; protocol-based factoring handles the divergence; struct-first design hits "special cases" |
| A7: Complementary-argument-transform is one meta-primitive | T7+R6: it's at least 3 distinct shapes (input-translation, input-scaling, output-translation), plus hypot/gamma as different precision concerns; "one meta-primitive" is the *intended* unification, but the unification has to be done deliberately |
| A8: ExpKernelState is content-addressed | T11+R9: at the *recipe* tier yes; the kernel state may actually belong at the *IR* tier (provenance-addressed by sharing topology) |
| A9: Recipe wrappers are uniformly cheap | T4: pow is half the computation; sinh requires bidirectional state; the "wrapper" cost varies — this is data for design, not a flaw in factoring |
| A10: Polynomial is "minimax" | T6+R8: minimax is a *family* parameterized by error metric (abs/rel/Chebyshev); the choice is a `using()` parameter |
| A11: Kernel state computes once | T9: correctness "once", latency "depends on placement"; placement is IR's problem |
| A12: ULP budget is preserved through wrapper | T4+T8: per-recipe ULP bookkeeping; not inherited from kernel state |
| A13: Every parameter tunable contract met | T12+R8: 6+ hidden parameters need surfacing; struct-as-proposed fails the Filter Test |
| A14: ExpKernelState's success implies gamma fits the same pattern | T7: gamma is not an instance of the complementary-argument transform; Phase B → Phase D progression must not over-generalize |
| A15: "Kernel state" struct is the right abstraction | R4+R5+R9: trait-based, stateless, or IR-tier alternatives are live |

The map's pattern: **most assumptions survive in a sharpened form, not by replacement.** The original ExpKernelState design is *almost* right; the sharpening lives in (i) precision-parameterization, (ii) F13.C antibodies for boundary preconditions, (iii) honest accounting for family-member divergence, (iv) recognizing the IR-tier placement question, (v) the precision-on-demand contract as the limit-direction.

---

## Phase 5 — The Aristotelian move

Six candidate moves emerged across R1-R10. Ranking by leverage:

### Move-1: Implement at R3 (the proposal as written) with R8's parameter surfacing and F13.C antibodies on the boundary preconditions

**Specifics**:
- ExpKernelState carries the 6 hidden parameters (A13) as named fields with documented defaults.
- The struct's construction enforces three preconditions, each with a signature-level antibody:
  - **P1**: `x` is canonical (BigFloat canonical-form or finite-f64); construction returns `Result<Self, ExpKernelError>`
  - **P2**: The precision-context for `ln(2)` matches `r`'s representation type; ill-typed construction is a compile-time error (Rust type system carries it)
  - **P3**: `k` is within the representable range of the output precision; saturation at the boundary (per F13.C tameness pattern)
- Cache key is BLAKE3 of `(IR_VERSION, "ExpKernelState", x_bits, precision_context_tag, reduction_tag, polynomial_tag, ln2_precision_bits, branch_policy)`. The branch_policy slot is `BranchPolicy::Real(...)` at this layer; the slot exists to be DEC-032 compliant from day one.

**Trade-offs**:
- This is the proposal **made antibody-aware**. The original libm-factoring proposal doesn't name P1/P2/P3; without them, the proposal admits exactly the silent-failure modes F13.C is meant to prevent.
- The cache key is wider than the original (more tag bytes). That's fine — the cache key is bytes; more bytes is more discrimination, not a cost.

**Why this is the move**:
1. It lands in Sweep 35's scope (a real Phase B).
2. It compiles to a Phase C that's mechanical (wrappers compose the state).
3. The F13.C antibodies are *required* by the holonomic + tameness architecture — without them, Phase B ships a non-holonomic kernel.
4. It defers R9 (IR-tier placement) without committing to it; the recipe-tier struct can be lifted to IR-tier intermediate in a future sweep without breaking consumers.

### Move-2: Don't ship a struct for sinh/cosh/tanh — ship `BidirectionalExpKernelState`

The fact that sinh/cosh/tanh need both `ExpKernelState(x)` and `ExpKernelState(-x)` is **not** a wrinkle. It's a structural fact about the hyperbolic family. The proposal as written hides it (each recipe constructs two states). A `BidirectionalExpKernelState` carries `(expm1_pos, expm1_neg)` *together*, with a single cache key over `x` (signed) and both directions populated by one reduction.

**Why this is part of the move**: it surfaces A6's hidden divergence as a *named state-variant*. The recipe tier has two kernel-state types (`ExpKernelState`, `BidirectionalExpKernelState`); the trait `KernelState` (per R4) unifies them. Future state-variants (e.g., `ScaledHypotKernelState` for hypot, `LanczosKernelState` for gamma) extend the trait without forcing the original into special-case fields.

This is **R3 + R4 — struct registered through trait**.

### Move-3: Treat the complementary-argument-transform as a *protocol*, not a struct

Per A7's deconstruction: complementary-arg-transform names a *family* of three shapes, not one shape. Make it a trait:

```rust
trait ComplementaryArgumentTransform {
    type FixedPoint;
    type Input;
    type Output;
    fn transform_to_distance(input: Self::Input, fp: Self::FixedPoint) -> /* distance-type */;
    fn stable_evaluation(distance: /* distance-type */) -> /* stable-output */;
    fn inverse_transform(stable: /* stable-output */, fp: Self::FixedPoint) -> Self::Output;
}
```

Each `-m1`-suffixed function declares its instance. The shared "meta-primitive" is the trait. Implementations are per-function.

This is **R4 applied at the meta-level** (the meta-primitive is a trait, not a single function).

### The actionable Aristotelian move (combining 1+2+3)

**Recommend to pathmaker before Phase B locks**:

1. **Define a `KernelState` trait** with `Input`, `Output`, `reduce`, `fingerprint`, `core_value` (per R4).
2. **Implement `ExpKernelState`** as one instance (per R3 + R8 — full parameter surfacing). It satisfies the trait. Cache key is the parameter bag.
3. **Implement `BidirectionalExpKernelState`** as a second instance (for sinh/cosh/tanh). It carries both directions. Same trait.
4. **Define a `ComplementaryArgumentTransform` trait** capturing the meta-primitive shape (per R4 at the meta-level). Mark `expm1`, `log1p`, `cosm1`, `sinm1` as instances. Mark `hypot` as **explicitly not** an instance (it's a homogeneous-scaling transform). Mark `gamma` as **explicitly not** an instance (it's an asymptotic-series-around-pole transform).
5. **Apply F13.C antibodies** at every kernel-state construction site:
   - P1 (canonical input): structural via `Result<Self, ExpKernelError>` return
   - P2 (precision-context type match): structural via generic parameter `P: Precision`
   - P3 (output-range saturation): structural via saturating arithmetic on k (per `internal-tameness-contracts.md`)
6. **Carry `BranchPolicy` slot in the cache key from day one** (DEC-032 compliance ahead of Phase D). For the real-valued exp/log family, `BranchPolicy::Real`. For Phase D's `complex_log`, `BranchPolicy::{Principal, AntiPrincipal, NumericallyStable, Discovery}`.
7. **Defer R9 (IR-tier kernel state) as a future-sweep candidate.** Note it in the design doc. The recipe-tier struct doesn't preclude the future IR-tier lift.

**The contrarian aspect** (what conventional analysis would never surface):
- The "kernel state struct" is **not the answer** — it's *one instance* of a family of answers. The trait-based design is the move that handles per-family-member divergence honestly.
- The complementary-argument-transform is **not one meta-primitive** — it's *a family of three shapes plus exceptions*. Naming the exceptions explicitly is more honest than fitting hypot and gamma into a meta-primitive they don't belong to.
- The cache key's discriminator bytes are **the antibodies**. Tag every precondition; fingerprint the bag; the cache key's correctness IS the holonomic invariant per `holonomic-architecture.md` § "Cache keys as operationalization."

---

## Forward path

**For pathmaker** (before locking Phase B design): incorporate Moves 1+2+3, OR push back on which parts of the move you disagree with. The deconstruction's value is in surfacing the tradeoffs; the choice is yours.

**For Phase 6-8** (next aristotle revision): challenge round on Moves 1-3 (does the trait-based approach hide a hidden assumption? does the trait API hold up under sinh/cosh/tanh implementation?). Recursive challenge until stability. Forced rejection (what if the trait is wrong? what if ExpKernelState is wrong? what does the void shape look like?).

**For the tameness-audit lane**: P1/P2/P3 above are the antibodies that need to be enforced in Phase B's implementation code. The audit-pass methodology from `internal-tameness-contracts.md` should be run *on the ExpKernelState construction site* before the code ships, not after adversarial generators fire. This is the F13.C graduation condition applied forward to new code.

**For navigator**: this deconstruction is a finding suitable for routing if pathmaker is about to lock Phase B without these considerations. Otherwise, it's substrate for pathmaker's design choices.

---

## Open questions surfaced (for future deconstruction or math-researcher)

1. **Is the trait-based KernelState design path-independent?** The trait API is content-addressable iff every implementation's fingerprint depends only on its parameter bag. Verify the trait can't admit an implementation that violates holonomicity.

2. **Does R10 (precision-on-demand symbolic graph) deserve a deconstruction of its own?** It pulls a thread that goes far beyond Sweep 35.

3. **The complementary-argument-transform trait — three instances or three traits?** If the three shapes (input-translation, input-scaling, output-translation) need different APIs, they should be separate traits. If they share enough surface, one trait with three instances. **Math-researcher's call.**

4. **Cross-family sharing**: if `BidirectionalExpKernelState` is registered, can `cosh` (which only needs the *sum* `exp(x) + exp(-x)`) hit the cache from a sinh-driven pipeline that registered the bidirectional state? Cache-key compatibility check (per Tambear Contract item 3) — the bidirectional state contains a superset of what cosh needs, so YES, **but the compatibility-tag must encode the relation**. This is the *substrate* of the IR-tier sharing-opportunity question.

5. **At P2BigFloat{1024}, what does ExpKernelState look like?** A 1024-bit `r`, a 1024-bit `expm1_r`, a much larger `k` (possibly i64). The reduction algorithm at this precision is *not* Tang's k·ln(2) — it's a high-precision arithmetic operation on BigFloat. The struct's generic parameter `P: Precision` is doing real work here, not just tagging.

6. **Does the kernel-state trait fit non-libm primitives?** sketch_gk, sketch_tdigest, COPA, partition_select — all have intermediate state that consumers share. If the trait is well-designed, *they should fit too*. If not, the trait is libm-specific (and that's fine — naming it as such avoids over-generalization).

---

*The deconstruction surfaces what the structure already implies. ExpKernelState — proposed cleanly as a struct — admits silent failure modes that the holonomic + tameness architecture flags as structural omissions. The fix is the architecture's own discipline applied forward, not new principles.*
