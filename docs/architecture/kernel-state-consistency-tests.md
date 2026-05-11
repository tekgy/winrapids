# Kernel-state-consistency-tests — the composition-time antibody class

**Status**: New architecture doc, math-researcher, 2026-05-10 (Sweep 35).

**Audience**: pathmaker (and any future role) who is about to ship a recipe that consumes a tambear kernel state (`TrigKernelState`, `ExpKernelState`, `LogKernelState`, future `UnitVectorState`, ...). Read this before locking the recipe contract.

**Companions**:
- `internal-tameness-contracts.md` — F13 antibody pattern; **signature-time** discipline.
- `tambear-libm-factoring.md` — where the kernel states live; design substrate.
- `holonomic-architecture.md` — recipe-tier content-addressing vs IR-tier provenance-addressing.
- `confident-wrong-narratives.md` — apparatus-first investigation; relevant when a consistency-test fails surprisingly.
- `~/.claude/garden/2026-04-11-fma-consistency-not-accuracy.md` — past-naturalist's canonical framing of *consistency dominates accuracy* in tambear's value function. This doc is the same principle applied at the kernel-state-vs-standalone-implementation seam.
- `~/.claude/garden/2026-04-10-nan-eating-is-a-sharing-bug.md` — past-naturalist's "bug count per operation type = number of independent implementations = number of missing sharing edges." Kernel-state-consistency-tests are the structural enforcement of that sharing.

---

## 1. The failure class

**Setup**. A *kernel state* in tambear is a shared intermediate value cached via TamSession at the recipe tier. When multiple recipes consume the same kernel state, they all derive their output from the same internal representation — which is exactly the libm-factoring frame's point.

**Concrete instance** (Sweep 35). `ExpKernelState` carries `(k, r_repr, expm1_r_repr)`. The recipe wrapper for `exp(x)` reconstructs the output as `(1 + expm1_r) << k`. But `exp(x)` *also* has a standalone implementation in `crates/tambear/src/recipes/libm/exp.rs` from prior work (Sweep 34 era), which computes its own range reduction and its own polynomial evaluation.

After Phase B of Sweep 35 lands, there are *two* implementations of `exp(x)`:

```
Implementation A (kernel-state-backed):
  let state = ExpKernelState::compute(x, precision_context);
  let result = (1.0 + state.expm1_r) << state.k;

Implementation B (standalone, pre-existing):
  exp_correctly_rounded(x)  // own k, own r, own polynomial
```

The kernel-state design *assumes* A and B produce bit-identical output for every (x, precision_context). The TamSession's content-addressed cache key for `ExpKernelState(x, p)` is meaningful only if every consumer of the cached state produces the same final output as if they had computed standalone.

**The failure mode**: A and B diverge silently. The cache is correct (same input → same state), but the recipe-layer reconstruction in A uses different polynomial coefficients, a different reduction algorithm, or a different reconstruction formula than B's. Outputs differ by 1+ ulps. No signature-level antibody fires because both A and B have the right *signature*. The divergence appears only when a caller alternates between them — or, worse, when a pipeline calls both expecting them identical and then breaks downstream.

**Why this is structurally distinct from F13.C**:

| Property | F13.C (signature antibody) | Kernel-state-consistency-tests |
|----------|---------------------------|-------------------------------|
| When the bug fires | At construction time (call-site) | At composition time (cross-implementation seam) |
| What's wrong | Wrong parameter, missing parameter, default | Two implementations of same operation diverge |
| What the antibody catches | Mis-routing | Silent semantic drift |
| Mechanism | Non-defaulted parameter at signature | Generated proptest comparing A vs B |
| Where the contract lives | API surface | Inside the kernel-state architecture |
| Without the antibody | Caller forgets a parameter | Caller can't tell A and B disagree |

**This is the same shape past-naturalist named for FMA contraction**. The Qt Quick rendering bug (`2026-04-11-fma-consistency-not-accuracy.md`) was: the FMA-contracted version of `1.0 - i * (1.0/i)` is *strictly more accurate* than the non-FMA version (it's closer to the true mathematical zero), but the program depended on the inaccuracy of non-FMA. The accuracy gain broke a program that depended on the prior inaccuracy. Same shape here: implementation A (kernel-state-backed) and implementation B (standalone) can both be "correct" individually — same paper formula, both within 1 ulp of true math — and yet they can differ from each other by 1+ ulps because they made different micro-choices (polynomial degree, coefficient set, reduction split). **The accuracy story per-implementation is separate from the consistency story across-implementations.** Tambear's value function ranks consistency-across-implementations above per-implementation accuracy.

**This is also the same shape past-naturalist named for NaN-eating** (`2026-04-10-nan-eating-is-a-sharing-bug.md`): 11 places independently implementing NaN-aware-max-reduce with subtly different (broken) NaN handling. The bug count IS the number of independent implementations. The fix is *not* patching each implementation — it's *extracting one primitive and wiring all consumers to share it*. Kernel-state-consistency-tests enforce that wiring at the cross-implementation seam.

---

## 2. The antibody

**Shape**. For every recipe that consumes a kernel state (call it `recipe_R`, returning `R(x, ctx)`), if there exists a standalone implementation `recipe_R_standalone(x, ctx)` that does NOT consume the kernel state, then there MUST exist a proptest:

```rust
#[proptest]
fn kernel_state_consistency_R(x in arb_input(), ctx in arb_precision_context()) {
    let kernel_backed = recipe_R(x, ctx);
    let standalone = recipe_R_standalone(x, ctx);
    assert_eq!(
        kernel_backed.to_bits(),
        standalone.to_bits(),
        "kernel-state-backed R({x}, {ctx:?}) = {kernel_backed:?} ≠ standalone = {standalone:?}"
    );
}
```

**The assertion is bit-equality, not ≤1-ulp**. The kernel-state design's whole point is that A and B compute the same thing via different bookkeeping; if they differ even by 1 ulp, the kernel state is not actually shareable across both. The cache-key correctness depends on bit-equality.

**When the test fails**, the resolution is *never* "loosen the assertion to ≤1 ulp." The resolution is one of:
1. **Refit one of them** to match the other's micro-choices (polynomial coefficient set, reduction split, reconstruction formula). The kernel-state version typically wins because it's the *shared* implementation; standalone becomes a thin wrapper that calls into the kernel-state path.
2. **Document the divergence as intentional** if the standalone path is preserved for a specific reason (e.g., a low-precision fast path that consciously trades 1 ulp for speed). In this case, the standalone path stops claiming to implement the same operation; it gets a distinct name (`exp_fast` or `exp_p53_fast`), and only the renamed version exists outside the kernel-state path.
3. **Delete the standalone**. Often the right answer once the kernel-state version is correct.

**Anti-pattern: tolerance-as-bandaid**. Loosening to ≤1 ulp is the same shape as "I'll add a `NaN ? NaN : f64::max` check at each call site" instead of extracting one NaN-aware-max-reduce primitive. The tolerance hides the drift; the drift comes back later as either a downstream bug (some consumer notices) or a precision-tier-promotion failure (at BigFloat p=200, the 1 ulp at f64 becomes 50+ ulps at the higher precision because both A and B independently re-derive their coefficient sets and drift further).

**Per recipe wrapper, the test set**:

```rust
// Tier 1: bit-equality vs standalone (the main antibody)
#[proptest] fn R_kernel_vs_standalone_bit_equal(x, ctx) { ... }

// Tier 2: identity round-trips that span the kernel-state seam
#[proptest] fn R_round_trip_via_kernel(x, ctx) { ... }
//   e.g., log(exp(x)) ≈ x to ≤1 ulp; if log and exp both go through
//   ExpKernelState/LogKernelState, the round-trip is bit-exact;
//   if they diverge, the round-trip drifts.

// Tier 3: cross-recipe consistency (when multiple recipes share state)
#[proptest] fn sinh_cosh_pythagorean_via_kernel(x, ctx) { ... }
//   sinh^2(x) + cosh^2(x) = ... (identity that holds bit-exact when
//   both share ExpKernelState; drifts when each computes independently)
```

**The generated proptest is itself a kernel-state-consumer in the apparatus-first sense** (`confident-wrong-narratives.md`): the test apparatus *consumes* the same kernel state as the production code, so if the test passes, the apparatus and production are using the same intermediate; if the test fails, either the kernel state is wrong, the recipe is wrong, or — most informatively — the standalone implementation is at a different fidelity tier than the kernel-state path expects.

---

## 3. When to apply

**Apply unconditionally**:

- Every time a new recipe consumes a kernel state AND a standalone implementation of the same operation already exists in the codebase.
- Every time a new kernel state is introduced AND any existing recipe is migrated to consume it (the migration must include the consistency test; otherwise the migration is incomplete).
- Every time a precision-tier promotion is added to a kernel state (p=53 → p=200 → p=1024): the consistency test must run at every supported tier, because the standalone path may behave differently at higher precisions than the kernel-state path.

**Apply conditionally**:

- When a kernel state is shared across multiple recipes (e.g., `ExpKernelState` shared by `exp`, `expm1`, `sinh`, `cosh`, `tanh`): pairwise consistency tests between the recipes. `sinh(x)·cosh(x) = 0.5·sinh(2x)` is a kernel-state-consistency identity if all three derive from `ExpKernelState`; if they diverge, the identity drifts.
- When pow or other Kingdom-III composite-kernel-state recipes are added: the composition's output must equal the composition of the standalone components. `pow(x, y) == exp(y · log(x))` is bit-exact under correct kernel-state design; if the standalone `exp` and `log` diverge from the kernel-state versions, the equality breaks.

**Apply with care**:

- When the standalone implementation is *known* to be at a different precision tier than the kernel-state version (e.g., a deliberately fast f32 fallback), the consistency test must scope to the matching tier. Don't compare kernel-state-at-p=200 against a f32-standalone — that's a category error.
- When the standalone implementation is being *deleted* in the same change that introduces the kernel-state version: skip the consistency test, but document the deletion in the commit and the recipe doc so future-Claude knows the standalone path *existed* and was deliberately removed.

**Skip when**:

- No standalone implementation exists (the recipe is born inside the kernel-state architecture). No A/B to compare.
- The standalone implementation is in a different crate or library entirely (e.g., glibc's `exp`); use the oracle harness (`R:\tambear\oracle\exp\`) for *external* cross-validation, not this antibody. This antibody is for *intra-tambear* cross-implementation drift, not platform-vs-tambear drift.

---

## 4. Why F13 alone doesn't cover this

F13 antibodies (signature-level, non-defaulted parameters) catch *one* failure class: a caller invokes a recipe without providing a required precondition parameter. The antibody is structural — the compiler refuses to compile a call site that omits the parameter.

F13 does NOT catch divergence *inside* the implementation between two implementations of the same operation. Consider:

```rust
// Both have correct F13.C signatures:
fn exp(x: f64, ctx: PrecisionContext) -> Result<f64, _> { ... }    // standalone
fn exp_via_kernel_state(x: f64, ctx: PrecisionContext) -> Result<f64, _> { ... }  // kernel-state-backed
```

F13 catches: someone calls `exp(x)` without `ctx`. Won't compile. F13 protects.

F13 does NOT catch: the two implementations return different values for the same `(x, ctx)`. They both honor the signature. Both type-check. Both produce a `Result<f64, _>`. The divergence is *semantic*, not *structural*.

**Kernel-state-consistency-tests catch the semantic divergence**. The proptest doesn't care about the signature — both implementations have the same signature; that's the *premise* of the test. It cares about the output bits. When the bits differ, the test fires; the divergence is forced into the light at composition time, not silently propagated.

**The two antibody classes are complementary**, not redundant:

- F13 protects the *contract* at the API surface.
- Kernel-state-consistency-tests protect the *implementation invariants* at the cross-implementation seam.

A recipe shipped with F13 but without consistency-tests is *signature-correct* and *implementation-fragile*. A recipe shipped with consistency-tests but without F13 is *implementation-consistent* but vulnerable to mis-routing. Production-grade tambear recipes need both.

---

## 5. The Sweep 35 instance — worked example

**Setup**: `ExpKernelState` lands in Phase B (`#5` in task list, completed). Its `expm1_r` field is bit-exact per Tang/fdlibm. The recipe `exp_via_kernel_state(x) = (1.0 + state.expm1_r) << state.k`.

**The standalone `exp.rs`** (pre-existing at `R:\winrapids\crates\tambear\src\recipes\libm\exp.rs`) ships with **raw Taylor coefficients** (`1/k!` for k = 0..13 strict; 0..16 correctly_rounded). The standalone uses Horner evaluation of those coefficients on `r` after Cody-Waite reduction.

**The divergence**: the Tang/fdlibm Remez-fit P1..P5 rational reconstruction inside `expm1_r` (and hence inside `exp_via_kernel_state`) is **NOT** the same polynomial as the raw Taylor that `exp.rs` ships. They produce different f64 outputs on the same `(x, ctx)` for many inputs. Difference can be 1-2 ulps on the polynomial-eval range endpoints (|r| ≈ 0.347).

**The catch path**:

1. Without the consistency test: the divergence is silent. Callers who use `exp(x)` directly get the standalone's answer; callers who pull `exp` through `ExpKernelState` get the kernel-state answer. Same source code, two answers. *Exactly the failure class past-naturalist named for FMA contraction* — both correct individually, inconsistent across the seam.

2. With the consistency test (this doc's antibody): the proptest fires on the first non-bit-equal `(x, ctx)`. Pathmaker is forced to choose:
   - Option (1): Refit `exp.rs` to the fdlibm P1..P5 rational reconstruction (matching the kernel state's choice). My coefficient-verification doc § Part 3 already specified this; the antibody enforces it at test time.
   - Option (2): Document the divergence as intentional (e.g., `exp_legacy_taylor` as a named path). Not the right call here, but the option exists.
   - Option (3): Delete `exp.rs`'s standalone and route everything through the kernel state. Clean; future-Claude has one implementation to reason about.

**The recommendation from coefficient-verification doc**: Option (1). Refit `exp.rs` to the kernel-state-compatible coefficients. The consistency test fires until refit lands; once refit lands, test stays green and the divergence cannot regress without a future PR breaking the test.

**Why the consistency test is the *structural* solution, not a one-time fix**: without the test, a future PR could "innocently" change one implementation's polynomial degree, coefficient set, or reduction strategy, re-introducing the divergence. The test makes divergence *impossible without an explicit decision* — the divergence either fails the test (caller intervenes) or it's documented as intentional and the test is amended (caller acknowledges).

This is past-naturalist's "bug count per operation type = number of independent implementations = number of missing sharing edges" applied at the antibody level. The test enforces *one implementation per (operation, kernel-state-architecture)*; future PRs that try to slip in a second implementation get refused.

---

## 6. Test-design patterns for kernel-state-consistency

### Pattern A — bit-equality on every input

The basic form. Generate `(x, ctx)` proptest inputs covering the canonical domain; assert bit-equal output.

```rust
#[proptest(cases = 10000)]
fn exp_kernel_vs_standalone_bit_equal(
    x in arb_f64_in_range(-700.0, 700.0),
    ctx in arb_precision_context_p53(),
) {
    prop_assert_eq!(
        exp_via_kernel_state(x, ctx).unwrap().to_bits(),
        exp_standalone(x, ctx).unwrap().to_bits(),
        "x = {x}, ctx = {ctx:?}"
    );
}
```

### Pattern B — identity round-trip across the seam

For pairs of recipes that round-trip mathematically (`log(exp(x)) = x`):

```rust
#[proptest(cases = 10000)]
fn log_exp_round_trip_via_kernel(
    x in arb_f64_in_range(-700.0, 700.0),
    ctx in arb_precision_context_p53(),
) {
    let exp_x = exp_via_kernel_state(x, ctx).unwrap();
    let recovered = log_via_kernel_state(exp_x, ctx).unwrap();
    // ≤1 ulp round-trip; if both go through the same kernel-state architecture,
    // bit-equal; if they diverge anywhere in the seam, drifts.
    prop_assert!(
        ulps_between(x, recovered) <= 1,
        "round trip drift: x = {x}, recovered = {recovered}"
    );
}
```

### Pattern C — Pythagorean / multi-recipe identity

When multiple recipes share a kernel state, identities that hold mathematically should hold bit-exact (or ≤1 ulp) at the f64 output:

```rust
#[proptest(cases = 10000)]
fn cosh_squared_minus_sinh_squared_via_kernel(
    x in arb_f64_in_range(-700.0, 700.0),
    ctx in arb_precision_context_p53(),
) {
    let s = sinh_via_kernel_state(x, ctx).unwrap();
    let c = cosh_via_kernel_state(x, ctx).unwrap();
    let identity = c * c - s * s;
    // cosh²-sinh² = 1 mathematically; ≤2 ulps allowance for the
    // multiplication at f64. If sinh and cosh share ExpKernelState,
    // this passes uniformly; if they diverge, the identity holds at
    // most where they happen to drift consistently.
    prop_assert!(
        ulps_between(identity, 1.0) <= 2,
        "cosh²−sinh² ≠ 1 at x={x}: identity = {identity}"
    );
}
```

### Pattern D — cross-precision-tier consistency

When the kernel state supports multiple precision tiers, round-trip across tiers should not drift:

```rust
#[proptest(cases = 1000)]
fn exp_p200_to_p53_bit_equal_to_p53_direct(
    x in arb_f64_in_range(-700.0, 700.0),
) {
    let high_then_round = {
        let res_p200 = exp_via_kernel_state(x, PrecisionContext::P2BigFloat { p: 200 }).unwrap();
        round_p200_to_p53(res_p200)
    };
    let direct_p53 = exp_via_kernel_state(x, PrecisionContext::P0F64).unwrap();
    prop_assert_eq!(
        high_then_round.to_bits(),
        direct_p53.to_bits(),
        "x = {x}: p200 → p53 round = {high_then_round}, p53 direct = {direct_p53}"
    );
}
```

This is the strongest form — it's the BZ-multi-limb pattern from Sweep 31's Phase C, applied to kernel-state consistency.

---

## 7. Failure-mode catalog

When a consistency test fires, the diagnostic is one of:

1. **Polynomial coefficient mismatch** (Sweep 35's exp.rs case). One side uses Taylor; the other uses Remez-fit. Fix: pick one, refit the other, document the choice in coefficient-verification doc.

2. **Reduction strategy mismatch**. One side uses Cody-Waite 2-step; the other uses Cody-Waite 3-step or DD reduction. Fix: align the reduction.

3. **Reconstruction formula mismatch**. One side computes `(1.0 + expm1_r) << k`; the other computes `(1.0 + expm1_r) * 2.0_f64.powi(k)`. The shift form is exact; the multiplication form rounds. Fix: align to the exact form.

4. **Special-case handling mismatch**. One side returns `+0.0` for `exp(-Inf)`; the other returns `0.0` (sign-unspecified). Fix: align IEEE 754 special-case semantics across both.

5. **Precision-context interpretation mismatch**. The kernel-state side interprets `PrecisionContext::P0F64` as "round all intermediates at p=53"; the standalone interprets it as "p=53 working precision but DD intermediates allowed." Fix: pin the precision contract; document at the kernel-state field level.

6. **Cache invalidation bug**. The TamSession cache returns a stale `ExpKernelState` from a prior `(x, ctx)` even though `ctx` has changed. Fix: include precision-context-tag-bytes in the cache key (per coefficient-verification doc § Part 4).

7. **Different rounding modes**. One side defaults to round-to-nearest-even; the other uses round-toward-zero. Fix: pin the rounding mode; surface in the precision context if it's a runtime knob.

Each of these is a *real* failure mode I've observed or read about in the floating-point literature. The consistency test surfaces them at test time rather than at production time.

---

## 8. The naming — why "consistency-tests" not "equivalence-tests"

The choice of word matters. Past-naturalist's April 11 essay establishes: **tambear's value function ranks consistency above accuracy.** Naming the antibody class "consistency-tests" carries that framing forward — it's not "test that A and B are equivalent in some abstract mathematical sense"; it's "enforce that A and B agree bit-for-bit within tambear's consistency contract."

"Equivalence" implies a math-truth relation independent of implementation; "consistency" implies an implementation-truth relation that's the contract. The latter is what tambear's architecture promises consumers.

The naming also distinguishes from:
- **Conformance tests**: tambear vs glibc/mpmath/external reference. Different antibody class (oracle-harness pattern).
- **Property tests**: mathematical identities (Pythagorean, monotonicity). Overlapping but broader — property tests can run on a single implementation; consistency tests require A AND B.
- **Regression tests**: pin a specific past behavior. Time-direction asymmetric; consistency tests are time-direction symmetric (both implementations are present-tense).

---

## 9. Cross-references and forward implications

**For Sweep 35**:
- Pathmaker's Phase B (`#5`, completed): ExpKernelState struct shape locks in. The consistency-test discipline applies as soon as Phase C wrappers ship.
- Pathmaker's Phase C (`#6, #7`, in_progress + pending): each `exp`, `log`, `exp2`, `log2`, `exp10`, `log10`, `sinh`, `cosh`, `tanh`, `hypot`, `pow` wrapper that has a pre-existing standalone implementation in `R:\winrapids\crates\tambear\src\recipes\libm\` is a consistency-test target. The proptest harness from cross-precision proptest gauntlet (`#8`, completed) extends naturally to this pattern.
- Phase D (`#9`, pending — complex_log): once first complex-transcendental ships, its standalone counterpart (if any) needs the consistency test. The branch-cut sign-of-zero tests (`#10`, completed) are an *adversarial* test class; this antibody is a *symmetric* test class. Both are needed.

**For Sweep 36+**:
- Every new kernel state (e.g., `UnitVectorState` for hypot/atan2/cabs family, asymptotic-series state for gamma family) triggers the same antibody review.
- Migration path: when an existing recipe is moved INTO a kernel-state architecture, the *first* PR is the migration + the consistency test; the test must pass before merge.

**For the broader antibody catalog**:
- F13 (signature-level): caller mis-routing.
- Kernel-state-consistency-tests (composition-level): cross-implementation drift inside the architecture.
- Internal-tameness contracts (per `internal-tameness-contracts.md`): wider arithmetic-site audit pattern.
- Oracle harnesses (per `R:\tambear\oracle\`): tambear-vs-external-reference conformance.

These four classes are complementary. A production-grade recipe ships with antibodies from all four classes; their absence at any class is a structural debt that surfaces eventually as a real bug.

---

## 10. Postscript — what this doc is and isn't

**Is**: a named antibody class with the test-design pattern made structural; cross-referenced to the past-naturalist consistency-dominates-accuracy framing; applied with worked example (Sweep 35 exp.rs).

**Isn't**: a test framework (the proptest harness already exists in tambear). Isn't a complete typing of every consistency-test instance. Isn't a guarantee — like every antibody, it catches only what it's designed to catch.

**The discipline this doc itself models**: name the failure class first; map the antibody mechanism; cross-reference past-substrate where the same shape is already named; show the worked example; specify where to apply and where to skip. Same discipline as `internal-tameness-contracts.md` and `branch-cut-conventions.md`. The architecture catalog grows by *naming* — past-scout's April 10 essay says it: *naming makes a concept checkable.* This doc names "kernel-state-consistency-tests" so that future-pathmaker can ask "have I added the consistency test?" — a question that *requires the name* to be answerable.

— math-researcher, 2026-05-10
