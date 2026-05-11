# ExpKernelState deconstruction — Phases 6-8

**Companion to**: `exp-kernel-state-deconstruction.md` (Phases 1-5)
**Date**: 2026-05-10
**Author**: aristotle, tambear-sweep35

---

## Phase 6 — Challenge round (assumptions added in Phases 1-5 audited as Phase-1 inputs)

The Aristotelian move from Phase 5 introduced 10 new assumptions of its own. Listing them as Phase-1 inputs and re-running:

### B1. The `KernelState` trait is the right level of abstraction

**Inherited from**: R4. Hidden assumption: that the trait can be defined at all in a way that's both rigorous (so all consumers can be implemented through it) and minimal (so the trait surface is not itself a leaky abstraction).

**Challenge**: `sinh = (exp(x) - exp(-x))/2` requires a value of type `Output`, not a `KernelState`. The trait's `core_value() -> Output` is fine for `exp` (one output). For `sinh` consuming `BidirectionalExpKernelState`, the consumer needs `(pos.core_value, neg.core_value)` and then composes. The trait doesn't help — sinh still does the arithmetic at the recipe layer.

So **the trait's value is at the *cache* layer, not the *math* layer.** Recipes still write their own math; the trait is the discipline that says "if your math needs `expm1(r)`, here's how the framework gets it to you." Naming this honestly: the trait is **a caching protocol, not a mathematical protocol**.

**Truth surfaced**: T14. The trait abstracts caching, not mathematics. Recipes own their math; the trait owns the sharing.

### B2. Sinh/cosh/tanh need `BidirectionalExpKernelState`

**Hidden assumption**: that bundling positive and negative directions is the right factoring. Alternative: register `ExpKernelState(x)` and `ExpKernelState(-x)` independently; let the cache-key system handle the deduplication.

**Challenge**: registering two independent states means two cache lookups per sinh call. The bidirectional struct is *one* lookup. **The cache-key system handles correctness either way; the bidirectional struct is a latency optimization.**

But: the bidirectional struct also encodes the *invariant* that `expm1(-x)` is computed at the same precision context as `expm1(x)`. With independent registration, a precision-context drift could give you `expm1(x) @ P0F64` paired with `expm1(-x) @ P2BigFloat{1024}` — *probably* a no-op by the cache-key matching, but the invariant isn't *structural*.

**Truth surfaced**: T15. The bidirectional struct is a correctness invariant carrier, not just a latency optimization. Worth keeping at the trait-instance level, not as a "for performance" wrinkle.

### B3. F13.C antibodies are needed at *every* construction site

**Hidden assumption**: that the construction site is the right boundary. Alternative: the antibody could be at *use* site (every `expm1_r` consumption verifies the precondition).

**Challenge**: per F13's open question #2 ("when does the antibody belong at construction time vs use time?") — construction-time is correct when the precondition is on the *input*; use-time when the precondition is on the *output's downstream use*. Here: the precondition is on the input (canonical, in-range), so construction-time is correct.

But the *post-construction state* is itself a precondition for `(1 + expm1_r) << k`: it requires `expm1_r > -1` (otherwise `1 + expm1_r ≤ 0`, log of which is undefined). Is *that* a construction-time invariant of ExpKernelState, or a use-time precondition on `exp`?

**Truth surfaced**: T16. The kernel-state struct's *value* has invariants beyond its constructor's input. The constructor must establish `expm1_r > -1` AND that establishment must be *visible* in the type. A `NormalizedExpm1` newtype wrapper carries the invariant in the type system. This is the F13.B (single-sited type-level enforcement) analog applied within the kernel state.

### B4. The Tambear-Contract Filter Test applies to a struct

**Hidden assumption**: filter-tested primitives are functions, not data. The contract's items (custom-implemented, accumulate+gather, every-parameter-tunable, etc.) read as function-level claims.

**Challenge**: ExpKernelState is data. The filter test items map onto a struct only by extension:
- Item 1 (custom-implemented): the struct's *constructor* and *consumers* are custom-implemented.
- Item 2 (accumulate+gather decomposition): the constructor's reduction step is `accumulate(All, eval_reduction, op=Identity)`; the polynomial evaluation is `accumulate(All, eval_polynomial, op=Identity)`. The struct itself is data; its *creation* is two atom operations.
- Item 3 (TamSession shareable): the struct *is* registered via TamSession; this is the struct's purpose.
- Item 4 (every-parameter-tunable): R8's expanded struct surface satisfies this.
- Item 5 (every measure in every family): N/A for a state struct; applies to the recipes that consume it.

**Truth surfaced**: T17. The Filter Test maps cleanly onto kernel-state structs IF the struct is viewed as the *data form of a computation*. The struct + its consumers are filter-tested as a unit, not the struct alone.

### B5. Defer R9 (IR-tier kernel state) is the right call

**Hidden assumption**: that Sweep 35 should land a recipe-tier solution. Alternative: do the IR-tier work and skip the recipe-tier struct.

**Challenge**: IR-tier sharing requires the IR layer to be able to *recognize* the cross-recipe sharing opportunity. Per `holonomic-architecture.md` § "What the IR layer has": "Pipeline fingerprints in the IR-level cache key" and "Cooperative dispatch declarations" — these are listed as elaborations, not as shipped machinery. Sweep 35 can't depend on them.

But: the recipe-tier struct, if it's the cache primitive, **is itself the cooperative-dispatch declaration**. The IR layer, when it lands, will recognize "two recipes both want `ExpKernelState(x, ctx)`" by seeing the cache-key hits. The recipe-tier struct *is* the foundation for the IR-tier optimization.

So R9 is not "after R3"; **R9 is enabled by R3**. The recipe-tier struct is the substrate the IR-tier sharing-recognition rests on.

**Truth surfaced**: T18. The "defer IR-tier" framing is wrong. Building R3 (with R4+R8 sharpening) IS building the foundation R9 will use. There is no "choose one"; R3 lands first, R9 lands later on top of it.

### B6. The trait-based design is precision-parameterized "for free"

**Hidden assumption**: a trait with `Input` and `Output` types handles precision automatically.

**Challenge**: Rust's type system makes it look automatic, but the **fingerprint serialization** must include the precision-context tag. `ExpKernelState<P0F64>` and `ExpKernelState<P2BigFloat<1024>>` have different cache keys *only if* the type tag is serialized into the cache key. Generic-parameter erasure in the cache-key path is a real bug risk.

**Truth surfaced**: T19. Generic parameters carry precision; cache-key serialization must include the type-name tag. This is the F13.C-shaped antibody for generic-type-tagging. Without it, two different-precision states could collide. **Almost certainly needs a `const TYPE_TAG: [u8; N]` per impl.**

### B7. `pow(x, y) = exp(y · log(x))` composes via nested kernel states

**Hidden assumption**: that composition through nesting is correctness-preserving. A `pow` recipe constructs `LogKernelState(x)`, extracts `log_x`, multiplies by `y`, constructs `ExpKernelState(y · log_x)`, extracts `exp_result`.

**Challenge**: the multiplication `y · log_x` happens at the recipe layer (T4). Its precision is bounded by the *input precision context* — but `log_x` is the *output* of one kernel state and the *input* to another. If the precision contracts don't align, the composition introduces error that's not visible in either kernel state's cache key.

**Truth surfaced**: T20. Cross-kernel-state composition needs a *composition cache key* — the cache key for `pow(x, y)` is not derivable from the cache keys of `LogKernelState(x)` and `ExpKernelState(y · log_x)` alone, because the intermediate `y · log_x` is not cached. **Pow needs its own kernel state**: `PowKernelState { log_x_state, y_times_log_x, exp_state }`. Otherwise the composition is correctness-fragile.

This was libm-factoring open question 1; the deconstruction sharpens it: **yes, pow deserves its own kernel state**, structurally, not as optimization. The composed-form alternative is correctness-fragile.

### B8. The "complementary-argument-transform" name is honest

**Hidden assumption**: that "complementary-argument" is the right name for the meta-pattern.

**Challenge**: the *complementary argument* in `log1p(x)` is `1+x` (the function's natural argument), not `x` (the user's input). For `expm1(x)`, the complementary argument is `x` and the complementary *output* is `result + 1`. For `sinpi(x)`, the complementary argument is `π·x`. For `hypot(a,b)`, there is no complementary argument — there's a *complementary scale*.

The name "complementary-argument" applies to two of the three shapes from Phase 4 (input-translation, input-scaling). The third shape (output-translation) is **complementary-output**. The fourth shape (homogeneous-scaling for hypot) is **complementary-scale**.

**Truth surfaced**: T21. The meta-primitive's *name* is doing more work than its shape supports. Either:
- Tighten the name to "complementary-argument-transform" and exclude output-side variants (cosm1, sinm1 — would need their own meta-name).
- Generalize the name to "complementary-form transform" and admit it covers a family with sub-variants (argument, output, scale).

This is past-Claude's framing being challenged. Past-Claude was right that there's a unifying pattern; the unification has more internal structure than one name conveys.

### B9. The recipe-tier vs IR-tier separation is clean for kernel states

**Hidden assumption**: that kernel states fit cleanly at the recipe tier per holonomic-architecture.md.

**Challenge**: if a kernel state's *cache key* depends on which recipes will consume it (e.g., a "minimal" state for just `exp` is smaller than a "full" state for `sinh/cosh/tanh`), then the cache key has pipeline context. That's IR-tier.

But: making one fixed struct for the family, *including* the bidirectional fields even when only `exp` is the consumer, eliminates the pipeline-context dependency. The cost is wasted computation when only `exp` is needed. The benefit is clean tier separation.

**Truth surfaced**: T22. Picking the right struct shape is a tradeoff between "recipe-tier clean + occasional waste" and "IR-tier elegance + harder cache discipline." For Sweep 35: prefer recipe-tier clean. The waste is bounded (one extra `expm1(-x)` computation); the cache discipline is much simpler. **Don't over-engineer the kernel state to be context-aware.**

### B10. The deconstruction has been deep enough

**Hidden assumption**: that Phases 1-6 have found everything load-bearing.

**Challenge**: have I missed a structural issue? Let me audit my Phase-1 list against the substrate one more time...

- A1-A15 covered the proposal's surface.
- T1-T13 (Phase 2) covered the irreducible truths.
- T14-T22 (Phase 6 round 1) covered the move's own assumptions.
- R1-R10 spanned simple-to-impossible reconstructions.

**What I haven't covered**:
- **Concurrency**: if two threads concurrently `get_or_compute` the same ExpKernelState, the cache must serialize correctly. This is a TamSession property, not a kernel-state property. Out of scope.
- **Determinism across hardware**: per DEC-019 (no middleware, per-door JIT), the kernel-state computation on CPU vs GPU vs NPU must produce the same f64 bits. This is the **bit-exactness requirement**. ExpKernelState's value depends on the FMA availability, the rounding mode, the parallel reduction order. If the kernel state is computed differently on different doors, the cache-key match returns wrong values when crossing doors. **This is load-bearing for Sweep 35.**

**Truth surfaced (Round 2)**: T23. The kernel state's value must be **door-independent or door-tagged in the cache key**. If door-independent, the implementation must produce identical bits across CPU/GPU/NPU (challenging for fused-MAC vs separate-mul-add). If door-tagged, the cache key includes the door; cross-door cache hits never happen. **Door-tagging is the safe default; door-independence is the optimization.**

This was *not* in the briefing-substrate. It surfaces from T9 (placement is IR's job) intersected with DEC-019 (per-door JIT with vendor IR). The kernel state's cache key needs a door byte. Phase B's design should include it.

---

## Phase 7 — Recursive challenge until stability

Round 3 audit: assumptions added in Round 2.

### Round 3.1: Is T23 (door-tagging) load-bearing for Sweep 35?

T23 says cache keys must tag the door. **But**: Sweep 35 is implementing CPU-side recipes first (per pathmaker's Phase A→D progression). The door is implicitly CPU. Does T23 matter *now*?

**Answer**: yes, because the cache key shipped in Phase B is the cache key forever (modulo IR_VERSION bumps). Adding a door tag later means bumping IR_VERSION and invalidating all caches. Adding it now is free. **F13.C-shaped: structural fix at the cache-key boundary, not retrofit.**

### Round 3.2: T14 says trait abstracts caching, not mathematics. Does the trait need a *mathematics* counterpart?

If the trait abstracts caching, there should be a parallel discipline for *mathematics* — what's the contract a recipe satisfies vis-à-vis its kernel state?

**Answer**: the parallel discipline is the **complementary-argument-transform trait** (Move 3). It IS the mathematical contract: declare your fixed point, your transform-to-distance, your stable evaluation, your inverse transform. The two traits *together* are the factoring discipline:
- `KernelState` (caching): how the framework gets data to you
- `ComplementaryArgumentTransform` (mathematics): what your math has to be shaped like

A recipe implements both traits if it's in the factored family. The cache-tier (KernelState) and math-tier (CompArgTransform) are independent surfaces.

**Truth surfaced**: T24. The two traits compose. Implementing both is the discipline for the factored libm; implementing one (e.g., a non-factored primitive that just caches an intermediate) is the discipline for other shareable computations.

### Round 3.3: Have I considered the cross-precision proptest gauntlet?

Per Phase C of BZ unstub (and Sweep 35 acceptance criteria): cross-precision proptest gauntlet. Compute at p_high, round to p_low, verify ≤1 ULP cross-precision drift.

How does this interact with ExpKernelState?

**Answer**: the gauntlet tests the *recipe wrapper*, not the kernel state directly. The kernel state at p_high produces `expm1_r_high`; at p_low, `expm1_r_low`. These should agree at the p_low precision. **The gauntlet verifies T22 (the bit-exactness of the kernel state across precision contexts) and T23 (across doors, if multi-door)** — but indirectly, via the recipe's output.

The kernel state itself can have a *direct* gauntlet: construct at p_high and at p_low; round the p_high state to p_low; bit-compare. If they don't match, the kernel state has precision drift independent of any recipe.

**Truth surfaced**: T25. The kernel state deserves its own cross-precision direct gauntlet, not just an indirect one through the recipes. **Add to adversarial's Phase B test plan.**

### Stability check

Round 4 audit: assumptions added in Round 3.
- T23 (door-tagging): structural; same argument as T22.
- T24 (two traits compose): structural; emerges from the cache-vs-math separation.
- T25 (kernel state direct gauntlet): a testing discipline addition; doesn't add new assumptions.

No new structural assumptions added in Round 3. **The deconstruction reaches stability after Round 3.**

---

## Phase 8 — Forced rejection

What if EACH of the truths is forcibly rejected? What does the void look like?

### Reject T1 (exp's dynamic range exceeds f64)

Imagine f64 had infinite range. There'd be no need for range reduction. The kernel state would be `ExpKernelState { x: f64, expm1_x: f64 }` — no `k`. Every recipe would call `expm1_x` directly. The factoring frame would still apply (sharing the polynomial evaluation), but the *structural pressure* for the reduction would vanish.

**Void**: T1 IS the structural pressure for k. Without T1, there's no k. With T1, k is forced. **k exists because the output type is finite.**

### Reject T2 (log's unbounded gradient at 0 and 1)

Imagine log had bounded gradient everywhere. No precision-regime split. log would be a polynomial-around-1 with no special boundary at 0. The complementary-argument transform wouldn't be needed.

**Void**: T2 IS the existence of the complementary-argument transform. Without T2, log1p collapses into log; expm1 collapses into exp. **The complementary forms exist because their non-complementary parents have singularities.**

### Reject T3 (exp/log are inverses)

If exp and log weren't inverses, they'd be two separate problems. The factoring frame would still apply per-family, but cross-family composition (`pow = exp ∘ log`) wouldn't exist. The PowKernelState (T20) would be moot.

**Void**: T3 IS the structural reason pow is a composition. Without T3, pow is its own primitive. **Pow's compositional nature requires the inverse relationship.**

### Reject T9 (placement is IR's job)

If the recipe tier had to do its own placement, the cache discipline would have to be path-dependent (which recipes are in scope, which can share). The recipe tier wouldn't be content-addressed — it'd be provenance-addressed.

**Void**: T9 IS what makes the recipe tier holonomic. Without T9, the holonomic-architecture's tier separation collapses. **The IR layer is the *reason* the recipe tier can be path-independent.**

### Reject T10 (F13.C requires signature-level enforcement)

If antibodies could be at the public API only, the BZ bugs from 2026-05-08/09 wouldn't have been fixable at scale. Each fix would have to be repeated at every internal call site as a local check; tech debt would compound.

**Void**: T10 IS why "no tech debt ever" is achievable. Without T10, the project's irrevocable principle becomes aspirational rather than enforced.

### Reject T13 (substrate-over-memory honesty)

If briefing-text were trustworthy as substrate, the "TrigKernelState already shipped" claim would be load-bearing fact. Phase B would be built on a foundation that doesn't exist.

**Void**: T13 IS what catches the foundation error. Without T13, Sweep 35 ships on sand.

### What MUST also exist (implied by the truths)

The truths imply a missing piece I haven't named:

**The precision-on-demand contract** (R10's preliminary first principle). T1+T6+T8 together say: *the framework knows what precision is needed and can synthesize implementations at that precision*. The struct-based approach is one *implementation* of this contract for the f64 + DD + BigFloat tiers. **The contract itself is the first principle.**

The kernel-state struct is a *value type* in a category whose *morphisms* are precision-context arrows. Each precision context P is an object; `ExpKernelState<P>` is a functor's value at P; the cache keys are the structure-preserving maps.

**Find it**: the precision-on-demand contract is the missing principle. The kernel-state design is fine for Sweep 35, **but the contract needs a name and a doc**. Otherwise, the next family (gamma) will hit the same "is this struct shape right?" question without the contract to answer it.

**Recommend**: surface to navigator + math-researcher as a *future work* item, not a Sweep 35 blocker. The contract documents the *interface* that future kernel-state designs satisfy. ExpKernelState is the first instance; future instances (LogKernelState, BidirectionalExpKernelState, PowKernelState, eventually GammaKernelState) satisfy the same contract.

### What if order had to differ?

The libm-factoring doc's Phase A → B → C → D order is: expm1 + log1p (cores) → ExpKernelState (struct + cache) → wrappers → complex_log. What if the order had to be reversed?

**D → C → B → A** (most-complex first):
- Implement `complex_log` first. BranchPolicy machinery is exercised first.
- Implement wrappers using the BranchPolicy-aware kernel state from the start.
- Implement ExpKernelState as needed by wrappers, with BranchPolicy slot.
- Implement expm1 + log1p as needed by the kernel state.

**Void from reversed order**: BranchPolicy is exercised on the *first* recipe, not the *last*. Every kernel state ships with BranchPolicy from day one — no "add BranchPolicy slot in Phase D" backfill. **This is structurally cleaner.** Suggests Phase B should *start* with BranchPolicy slot included (per Move 1, item 6 of Phase 5).

### Hint: what does the void of "no factoring" tell us?

If ExpKernelState didn't exist, every recipe would do its own reduction. The structural void:
- 15+ duplicate reduction implementations
- Tang-degradation (per MSVC's pathology)
- No path to sharing across recipes
- The "every primitive every time" Tambear Contract item 5 becomes unmanageable (the per-recipe duplication multiplies; "every measure in every family" times "every implementation isolated" = quadratic effort)

**The void's shape**: factoring is not optimization; it's *what makes Tambear Contract item 5 implementable*. Without factoring, the contract's scope (all of mathematics) exceeds any implementation budget. **The factoring is the engineering enabler of the scope.**

This argument applies *generally*: any sub-family of primitives shares enough structure that factoring is non-optional. The kernel-state pattern is the *first instance* of a discipline that the Tambear Contract demands across all families. Sweep 35 isn't just shipping a libm refactor — **it's establishing the factoring discipline that every other family will rely on**.

---

## Stability summary

The deconstruction reaches stability at Round 3 of Phase 7. New truths since Phase 5:

- T14: Trait abstracts caching, not mathematics.
- T15: Bidirectional struct is a correctness-invariant carrier, not a latency optimization.
- T16: Normalized-Expm1 newtype carries the `expm1_r > -1` invariant in the type.
- T17: Filter Test maps onto structs viewed as "data form of a computation."
- T18: R3 enables R9; not "choose one."
- T19: Generic-parameter cache-key tagging is F13.C-shaped.
- T20: Pow deserves its own kernel state (PowKernelState).
- T21: "Complementary-argument-transform" name covers three shapes; the unification has more internal structure than one name conveys.
- T22: Don't over-engineer kernel state to be IR-context-aware.
- T23: Door-tagging in cache key from day one.
- T24: KernelState + ComplementaryArgumentTransform are two composable traits.
- T25: Kernel state direct cross-precision gauntlet.

Phase 8 forced-rejection surfaces:
- **The missing principle**: precision-on-demand contract (R10's first principle, deferred but named).
- **Order suggestion**: BranchPolicy from Phase B day one (not Phase D backfill).
- **The factoring's role**: not optimization; structural enabler of Tambear Contract item 5's scope.

---

## Final recommendation (synthesizing all phases)

**For pathmaker before locking Phase B**:

1. Define `KernelState` trait (caching protocol).
2. Define `ComplementaryArgumentTransform` trait (mathematics protocol) with three sub-shapes.
3. Implement `ExpKernelState<P: Precision>` as first KernelState instance.
4. Plan `BidirectionalExpKernelState<P>` as second instance for sinh/cosh/tanh.
5. Plan `PowKernelState<P>` as third instance, *not* a composition.
6. F13.C antibodies on input canonicality, type matching, output range saturation.
7. Generic-parameter tagging in cache keys.
8. Door byte in cache key from day one.
9. BranchPolicy slot in cache key from day one (DEC-032 forward-compatibility).
10. Cross-precision direct gauntlet for the kernel state itself.

**For navigator** (routing to team-lead): finding 0 (TrigKernelState not shipped in R:\tambear\) is the most time-sensitive. The recipe-tier design recommendations are the second wave. The "precision-on-demand contract" as a future-work principle is the third.

**For adversarial** (parallel coordination): F13.C antibodies P1/P2/P3 (Phase 5 Move 1) are the audit targets for the kernel state. The tameness audit's audit-pass methodology applies *to the kernel state construction code* before it ships.

**For math-researcher**: ComplementaryArgumentTransform trait's three sub-shapes (input-translation, input-scaling, output-translation) are a literature-grade structural question. Is there a 4-or-more-shapes generalization? Lanczos and homogeneous-scaling (hypot) are explicitly outside; what are they instead?

**For naturalist** (idle invitation): "Past-naturalist's day-two open question was group-theoretic instantiation of the complementary-argument-transform." The deconstruction's T21 says the unification has more internal structure than one name conveys. **The group-theoretic instantiation might NOT be one group; it might be a *coproduct* of three groups** (additive translation, multiplicative scaling, output-translation). Possibly a category-theoretic structure (a functor with three slices). If you pull on this thread, please do.

---

*Stability achieved. The recommendation is the architecture's own discipline applied forward, sharpened by Phase 8's forced rejection. ExpKernelState as a struct is fine; ExpKernelState as the only instance of a generalizable pattern is the wrong framing.*
