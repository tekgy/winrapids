---
campsite: tambear-formalize/survey/20260508123003-math-researcher
role: math-researcher
date: 2026-05-08
subject: assumption document — trig_reduce TamSession sharing contract
status: draft for pathmaker review
audience: pathmaker (formalization), aristotle (definitional review)
---

# Assumption Document: `trig_reduce` Sharing Contract

> **Purpose.** The trig recipes in `R:\winrapids\crates\tambear\src\recipes\libm\` declare a TamSession sharing contract via `[sharing] reads = ["trig_reduce"]; writes = ["trig_reduce"]`. When formalizing onto the locked-vocabulary substrate at `R:\tambear\`, we need a precise definition of what is shared, when sharing is sound, and what the cache-key fingerprint must contain. This document gives that definition.

> **Substrate at the destination.** Two distinct cache types exist at `R:\tambear\`:
> - `tambear-tam::ComputedTag` (`R:\tambear\crates\tambear-tam\src\tag.rs`) — value-sharing across recipes. BLAKE3 of `(input_id, computation_name, assumptions)`. Used by TamSession.
> - `tambear::jit::door::IntermediateTag` (`R:\tambear\crates\tambear\src\jit\door.rs`) — JIT compiled-kernel cache. BLAKE3 of the producer's full assumption fingerprint (op + shape + strategy + door + params). Used by per-door kernel cache.
>
> **`trig_reduce` is a `ComputedTag`-class shared value, not a kernel cache key.** The reduced (q, r_hi, r_lo) tuple is a value to share between sin/cos/tan/sincos/atan2pi/etc. on the same input column, not a compiled kernel.

---

## 1. What is shared

A range-reduction result for a column of f64 inputs:

```
ReducedTrig {
    q:    Vec<i32>,   // quadrant index per element ∈ {0, 1, 2, 3}
    r_hi: Vec<f64>,   // reduced argument, high part, ∈ [-π/4, π/4]
    r_lo: Vec<f64>,   // reduced argument, low part, |r_lo| ≤ ½·ulp(r_hi)
}
```

with the invariant `x = q·(π/2) + r_hi + r_lo` (mod 2π), where `r_hi + r_lo` is computed with sufficient precision that the kernel polynomial sees the right argument modulo 2^-100 or so. The exact precision target depends on the `range_reduction` strategy:

- **Cody-Waite three-part π/2** (PIO2_1..3 + tails PIO2_1T..3T): ~151 bits of precision in `r_hi + r_lo`. Valid for `|x| < 2^20·π/2`.
- **Payne-Hanek 1200-bit 2/π table**: ~1200-bit precision in the product, ~106-bit precision in the f64 + f64 representation of `r_hi + r_lo`. Valid for all finite f64.

## 2. Who reads, who writes

**Writers** (per the spec.tomls):
- `sin` writes `trig_reduce` after its first reduction.
- `cos` writes `trig_reduce` after its first reduction.
- `sincos` writes `trig_reduce` (and `sincos_kernel`, `sincos_result`).
- `tan` writes `trig_reduce` (and reads `sincos_kernel` if present).

**Readers**:
- Any subsequent `sin`, `cos`, `tan`, `cot`, `sec`, `csc` on the same input column.
- `sincos` reads to skip its own reduction if a prior call already ran.
- Pi-scaled variants (`sinpi`, `cospi`, `tanpi`) **do not** read `trig_reduce`. They reduce on the fractional part of `π·x`, not on `x`. Different reduction. Different cache key.
- `atan2pi` does not read `trig_reduce` either — it reduces atan results, not the input directly.

## 3. Sharing-correctness invariant

The TamSession sharing contract says: **two recipes share an intermediate iff the cached intermediate was computed under assumptions that match the consumer's requirements.** Mismatched assumptions → different tags → no incorrect sharing.

For `trig_reduce`, the assumptions that gate sharing are:

| Assumption | Reason |
|---|---|
| `range_reduction` strategy (`auto` / `cody_waite` / `payne_hanek`) | Cody-Waite is **wrong** for `|x| > 2^20·π/2` — silent error growth. A consumer that sees `cody_waite` in the cache must trust the producer used it within its valid domain. |
| `angle_unit` (`radians` / `degrees` / `gradians` / `turns` / `pi_scaled`) | Reduction happens AFTER the angle-unit pre-multiply to radians. A `degrees` reduction is identical to a `radians` reduction of the multiplied input — but the cache must NOT serve a `radians`-input cache to a `degrees`-input consumer. |
| Input column identity (input_id) | Different columns get different reductions even if the data happens to overlap. |
| Element count / shape | A scalar reduction of `x[0]` does not satisfy a vector-reduction request for `x[..]`. |
| Subnormal-input policy | If subnormals flush to zero (some hardware flag), reduction differs at the subnormal boundary. |

**Assumptions that do NOT gate sharing**:

- `precision` (`strict` / `compensated` / `correctly_rounded`). Range reduction is a property of the input mapping, not the polynomial-kernel precision. The reduced argument is the same; only the kernel evaluation differs. So `sin_strict` and `cos_compensated` SHOULD share the same `trig_reduce`. This is load-bearing for performance — without it, the three-strategy-API forces 3× reductions per column.
- Quadrant fixup table. Every recipe applies its own; the reduction output is shared.

## 4. Canonical cache key

Per `R:\tambear\crates\tambear-tam\src\tag.rs`:

```rust
ComputedTag::build(input_id, computation_name, assumptions)
```

For `trig_reduce`:

```rust
let computation_name = "trig_reduce";

// Assumptions: 1 byte for range_reduction strategy + 1 byte for angle_unit
// + 1 byte for subnormal policy. Length-prefixed by ComputedTag::build.
let assumptions = {
    let mut buf = [0u8; 3];
    buf[0] = match range_reduction_strategy {
        RangeReduction::Auto       => 0,
        RangeReduction::CodyWaite  => 1,
        RangeReduction::PayneHanek => 2,
    };
    buf[1] = match angle_unit {
        AngleUnit::Radians   => 0,
        AngleUnit::Degrees   => 1,
        AngleUnit::Gradians  => 2,
        AngleUnit::Turns     => 3,
        AngleUnit::PiScaled  => 4,
    };
    buf[2] = if flush_subnormals_to_zero { 1 } else { 0 };
    buf
};

let tag = ComputedTag::build(input_column_id, computation_name, &assumptions);
```

**Important normalization**: `range_reduction = "auto"` should be **resolved to either CodyWaite or PayneHanek before tag construction**, based on the input range scan. Otherwise two callers — one with `auto` (resolved to PH because some `x_i > 2^20·π/2`), one explicitly with `payne_hanek` — would get different tags despite computing the same thing. The tag must reflect what was actually run, not what was requested.

## 5. The asymmetry: precision is a kernel property, not a reduction property

This is the clarifying observation that drives the architecture: **range reduction is a precision-independent operation**. The reduced (q, r_hi, r_lo) is the same regardless of which polynomial kernel will consume it. Three strategies (`strict` / `compensated` / `correctly_rounded`) exist for the kernel evaluation step, NOT the reduction step.

Implication: the spec.toml's `precision` parameter SHOULD NOT enter the `trig_reduce` cache key. Doing so would force redundant reductions when a pipeline computes both `sin_strict` (for a fast path) and `sin_correctly_rounded` (for an oracle column) on the same input.

This is a divergence from the simple "fingerprint everything" rule. The right rule is finer: **fingerprint the assumptions that affect THIS computation's output**, not all parameters in scope. For `trig_reduce`, those are: range_reduction strategy (after `auto` resolution), angle_unit, and subnormal-flush policy. Not precision.

## 6. Boundary cases and their handling

### 6.1 Special-case inputs

`x ∈ {±0, ±∞, NaN}` are handled by `special_case_trig` BEFORE reduction. They never enter the cache. The cache stores reduction results only for finite, non-special inputs.

When a column has mixed finite and special inputs, the cache stores the full-length output with sentinel values at special positions. Consumers re-check the special cases on read. This is cheaper than partitioning the column.

### 6.2 Cody-Waite domain violation

If a user forces `range_reduction = "cody_waite"` and the input column contains `|x| > 2^20·π/2`, the result is silently wrong (Cody-Waite's PIO2_3 extension fails beyond its domain). Two options at formalization:

- **(a) Strict**: detect the violation at recipe entry, return an error. Deterministic; surfaces user error.
- **(b) Lax**: warn and proceed, document the silent-error region in the spec. Lets researchers experiment.

**Recommendation**: option (a). The Tambear Contract §10 (publication-grade rigor) requires that violated assumptions produce errors, not garbage. The user can override with `using(range_reduction="payne_hanek")` if they need the full domain.

### 6.3 Payne-Hanek table determinism

The 1200-bit 2/π table is a constant. Different builds, different platforms, must use the same table. Per CLAUDE.md irrevocable principle "no vendor lock-in," this means tambear ships its own table at build time, not via a dependency. Verify against fdlibm `e_rem_pio2.c`'s `__ieee754_rem_pio2` table data.

### 6.4 The (q, r_hi, r_lo) tuple format

q is `i32` per element. r_hi and r_lo are `f64`. Cache layout: three parallel vectors, not a struct-of-three. This matches the SoA convention used by sin.rs internally and is more cache-friendly for downstream kernels that read `r_hi` densely.

## 7. Sharing benefits — quantification

Per `MATH_ROADMAP.md` §"Cross-family patterns", a typical fintek bin runs ~15 methods that share the same intermediate. For `trig_reduce`:

- A pipeline computing `{sin, cos, tan, atan2pi}` on the same column gets 4 reductions. With `trig_reduce` sharing: 1 reduction + 4 kernel evaluations.
- Reduction cost in Cody-Waite ≈ 12 fmadds + 1 frint ≈ 13 flops/element.
- Reduction cost in Payne-Hanek ≈ 80-150 flops/element + table reads.
- Kernel cost ≈ 6-12 flops/element.

So sharing `trig_reduce` saves 75% of the reduction cost on a 4-recipe pipeline, which is the dominant cost when `|x| > 2^20·π/2` (Payne-Hanek dominates).

## 8. The two-cache topology — `sincos_kernel`

`sincos.spec.toml` writes a SECOND intermediate: `sincos_kernel` = `(sin_k, cos_k)` — the polynomial-kernel results BEFORE quadrant fixup.

```rust
SincosKernel {
    sin_k: Vec<f64>,  // r_hi + r_hi³·P(z), folded with r_lo
    cos_k: Vec<f64>,  // 1 - z/2 + z²·Q(z), folded with r_lo
}
```

This lets `tan` form `sin_k / cos_k` (with quadrant correction) without re-running the polynomial kernels. Adds another assumption to the cache key:

| Assumption (in addition to `trig_reduce` set) | Reason |
|---|---|
| `precision` (`strict` / `compensated` / `correctly_rounded`) | The kernel polynomial IS precision-dependent. A `compensated`-kernel result is NOT a valid input to a `strict`-kernel consumer. |

So `sincos_kernel` has a strictly larger assumption set than `trig_reduce`. They are separate ComputedTags.

## 9. Open questions for aristotle / pathmaker

1. **Auto-resolution timing**: `range_reduction="auto"` resolves based on input range scan. When does the scan happen — before tag construction (so the tag reflects the actual strategy)? Or at kernel dispatch (so the same tag may serve different actual strategies)? **My recommendation: before tag construction**, per §4. But this requires a column-min/max scan in the recipe entry.

2. **Per-element strategy split**: a column with mixed magnitudes — some `|x_i| < 2^20·π/2`, some not — could in principle use Cody-Waite for the small entries and Payne-Hanek for the large. Worth the complexity? Probably not at first pull; uniform-strategy-per-column is simpler. Document as a future variant.

3. **Cache eviction**: `trig_reduce` is per-column. If a pipeline runs over many columns, the cache fills with one tuple per column. Eviction policy? LRU? Per-pipeline scope? This is a TamSession-level decision, not a recipe-level one. Defer to TamSession spec.

4. **Cross-recipe versioning**: if `cos.rs` is updated to use a tighter polynomial kernel, but a cached `sincos_kernel` from the old version is still in the cache, we get a wrong answer. The `assumption fingerprint` must include a recipe-source version hash, not just parameter values. This is a Sweep 8/12B-level concern; the current `IntermediateTag` cache key includes "params" but not source-hash. **Surface to pathmaker.**

5. **Subnormal flush flag**: I included `flush_subnormals_to_zero` in the assumption bytes, but tambear may have a global determinism contract (per Sweep 0 / DEC-007 era) that subnormals are always handled deterministically without an FTZ flag. If so, drop that byte. **Verify with pathmaker.**

## 10. Filter Test recap

For the `trig_reduce` sharing pattern under the locked vocabulary:

- ✅ Custom-implemented (we author the reduction; we author the cache; we author the table data)
- ✅ Atom decomposition: reduction is a Kingdom A `accumulate(All, Expr::TrigReduce, Op::Identity)` per-element scan
- ✅ Shareable intermediates with assumption fingerprint per the §3 invariant
- ✅ Every parameter tunable (range_reduction is exposed)
- ✅ Every variant (Cody-Waite + Payne-Hanek) implemented; auto-dispatch based on input
- ✅ Optimized for 2026 hardware (per-element parallelism; table reads coalesce well on GPU)
- ✅ No vendor lock-in (table is ours)
- ✅ No OS lock-in (pure arithmetic + table reads)
- ✅ Lifting to TAM: per-column cache; TAM scheduler manages residency
- ✅ Publication-grade rigor: every assumption documented above

## 11. Provenance

- Authored 2026-05-08 by math-researcher in team `tambear-formalize`, during the first-leg survey of `tambear-trig`.
- Sources: `R:\winrapids\crates\tambear\src\recipes\libm\sin.rs`, `sincos.spec.toml`, `sin.spec.toml`, `R:\tambear\crates\tambear-tam\src\tag.rs`, `R:\tambear\crates\tambear\src\jit\door.rs`.
- Cross-references: CLAUDE.md §"Tambear Contract" point 3; `vocabulary.md` §"What stays the same" — `TamSession`; `MATH_ROADMAP.md` §"Cross-family patterns".
- This is a draft. Open questions in §9 require pathmaker / aristotle review before we lock the contract for libm-trig formalization.
