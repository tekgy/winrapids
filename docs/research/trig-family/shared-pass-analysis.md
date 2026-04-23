# The Shared-Pass Question — Is Trig a MomentStats?

*TRIG-4 analysis. Naturalist, 2026-04-13.*

*A structural reading of the shared-pass question. Navigator's framing:*
> *"sincos is the trig equivalent of MomentStats. If that's true, the entire trig family can be expressed as (1) reduce_trig(x) → TrigIntermediate; (2) Each function is a pure kernel applied to TrigIntermediate."*

*This document examines whether the analogy holds, where it breaks, and what the break tells us.*

---

## 1. What MomentStats is

In `intermediates.rs:405–415`, `IntermediateTag::MomentStats` is defined as:

> **Moment statistics for an ungrouped numeric array: {count, sum, min, max, m2, m3, m4}.**
>
> This is the **minimum sufficient representation (MSR)** for all moment-based statistics. A single scatter pass produces these 7 accumulators; every descriptive stat, hypothesis test, and normalization step derives from them in O(1) arithmetic — zero re-scanning.
>
> Consumed by: hypothesis tests (one/two-sample t, ANOVA), normalization, z-scoring, Pearson correlation, regression preprocessing.

Three structural properties make MomentStats what it is:

1. **Minimum sufficient**: the 7-tuple carries enough information to reconstruct every moment-based statistic in O(1). You cannot shrink it without losing a consumer; you cannot grow it without carrying dead weight.
2. **Scatter-computed**: one O(n) pass over the column produces it. The pass is content-independent — the same loop builds it regardless of which downstream consumer will read it.
3. **Fan-out by composition**: ~15 downstream stats read the *same* intermediate and each does a ~3-line O(1) derivation. The intermediate amortizes the expensive step across all consumers.

**The shared-pass win is quantitatively: one O(n) scan instead of 15.**

---

## 2. What trig_reduce is

From the spec.toml files pathmaker has shipped today (sin, cos, tan, cot, sec, csc, sincos, sincospi, sinpi, cospi, tanpi, versin, haversin), the declared sharing contract is a 2-tag pattern:

- **`trig_reduce`** — the range-reduction intermediate `(q, r_hi, r_lo)`: quadrant mod 4, high and low residual parts in `[-π/4, π/4]`.
- **`sincos_kernel`** — the kernel output `(sin_k, cos_k)` before quadrant fixup: the raw polynomial result on the reduced residual.

Let me hold these up against MomentStats' three properties.

### 2.1 Minimum sufficient? — **Yes, at each layer.**

`trig_reduce` has three f64 slots: `q` (the coset index mod 4), `r_hi` (double-word high part), `r_lo` (double-word low part). That's the minimum information needed to reconstruct *any* trig function's output. You cannot shrink it:
- Drop `q` → lose the sign/swap table, everything becomes quadrant-0 valued.
- Drop `r_lo` → lose ~24 bits of precision; sin/cos degrades to ~1200 ulps for large arguments (as erf/erfc did pre-tightening).
- Drop `r_hi` → nothing left.

You cannot grow it either — `r²` is derivable from `r_hi` in O(1) and adding it would be dead weight *for sin alone*, but would pay off for *the kernel tier* (see below).

`sincos_kernel` has two slots: `sin_k`, `cos_k`. Minimum for tan/cot/sec/csc and the fused outputs.

### 2.2 Scatter-computed? — **Yes, with one wrinkle.**

The reduction pass is content-independent: Payne-Hanek doesn't care whether you'll subsequently compute sin, cos, tan, or haversin. The loop is the same loop. Register the result once; all consumers read.

The wrinkle: the reduction's *cost curve* is piecewise. Cody-Waite costs ~6 FMAs for `|x| < 2^20·π/2`; Payne-Hanek costs ~50 FMAs + a 1200-bit table multiply for large arguments. MomentStats has a single cost curve (always O(n)); trig_reduce has two.

This matters for the sharing calculus: if all your x values are tiny, the reduction is so cheap that computing it twice might be cheaper than the HashMap lookup overhead. If some x values are large, the reduction dominates and sharing is always a win. **The shared-pass is conditionally worth it in a way that MomentStats' shared-pass is not.**

### 2.3 Fan-out by composition? — **Yes, but the fan-out has two tiers.**

Fan-out looks like this (from the spec.toml files):

| Intermediate | Readers | Pure kernel? | Notes |
|---|---|---|---|
| `trig_reduce` | sin, cos, tan, cot, sec, csc, sincos, versin, haversin, sinpi, cospi, tanpi | yes | The 1-tuple tier. Every forward trig reads this. |
| `sincos_kernel` | tan, cot, sec, csc, sincos, sincospi | yes | The 2-tuple tier. Only consumed after an intermediate kernel eval. |

13 downstream consumers for `trig_reduce`, 6 for `sincos_kernel`. The fan-out ratio (consumers / producer cost) is even higher than MomentStats' on real workloads — because *every* forward trig function hits `trig_reduce`, not just "many."

---

## 3. Where the analogy holds and where it breaks

### Holds

- **MSR principle**: both are minimum sufficient representations. Both shrink-to-ruin under reduction and carry no dead weight.
- **Scatter-pass structure**: one O(n) pass builds the intermediate; O(1) derivations per consumer.
- **Fan-out amortization**: the expensive step runs once and pays off across many consumers.
- **Composability**: higher-level functions (tan, sec, haversin) compose from lower-level intermediates with no re-scanning.

### Breaks

Four structural differences. Each one is a clue about trig's shape.

**3.1 Trig has TWO tiers; MomentStats has one.**

MomentStats is flat: one scan, 15 O(1) readers. Trig has `trig_reduce` (reduction tier) *and* `sincos_kernel` (evaluation tier). The kernel tier only exists because the reduction tier exists — it's a derivative shared intermediate, produced by the first consumer that needs it and reused by all subsequent ones.

This two-tier structure is what Aristotle's first-principles T5 + T6 predict: reduction is a *geometric* operation (canonical), while the kernel is an *algorithmic* one (has arithmetic choices). They separate naturally. The spec.toml files already reflect this in the `writes = ["trig_reduce", "sincos_kernel"]` pattern.

*Implication*: when designing new shared intermediates for other families, ask whether there's a two-tier structure. Hyperbolics almost certainly have one (`exp_bundle(x) = (exp(x), exp(-x))` is the reduction tier; `(sinh_k, cosh_k)` is the evaluation tier). That's the rhyme math-researcher should check for in TRIG-15.

**3.2 Trig's intermediate carries a DISCRETE coordinate (quadrant); MomentStats is fully continuous.**

`trig_reduce` is `(q, r_hi, r_lo)` — one integer + two floats. The integer is load-bearing: it selects among 4 sign/swap patterns at the output stage.

This is not a weird trig thing — it's a *group-theoretic* thing. S¹ (the circle) has a Z/4 subgroup of quadrant symmetries, and the reduction projects onto the quotient S¹/(Z/4) = [−π/4, π/4]. The integer `q` is the coset representative. MomentStats has no such group structure because ℝ has no non-trivial finite subgroup of interest.

*Implication*: any shared intermediate over a group-valued function will carry a coset representative alongside the continuous residual. Inverse hyperbolic trig (acoth, asech) and complex-argument trig will have the same shape.

**3.3 Trig's intermediate has PRECISION tiering; MomentStats has accuracy but no tiering.**

At `strict` precision, `trig_reduce` is `(i32, f64, f64_with_lower_bits_maybe_junk)`. At `compensated`, both residual parts carry meaningful bits — the low part is the algebraic remainder of the reduction. At `correctly_rounded`, a third residual part `r_ll` is introduced for a 159-bit composite representation.

The **same intermediate** has three different precision *contents*. A `trig_reduce` cached at `strict` precision is NOT compatible with a `correctly_rounded` consumer — the low bits are undefined.

This is exactly the **compatibility tag** problem from the Tambear Contract §3 ("Sharing is conditional, not automatic"). The shared intermediate must carry a precision fingerprint, and consumers must check `is_compatible(tag, my_requirements)` before pulling.

*Implication*: the IntermediateTag for trig_reduce needs a `precision_tier` field. Proposed shape:
```rust
TrigReduce {
    data_id: DataId,
    angle_unit: AngleUnit,   // radians / pi_scaled / degrees / turns
    precision: PrecisionTier, // strict / compensated / correctly_rounded
}
```

The `angle_unit` field is required too: `sinpi(x)` reduces `x mod 2` via `frint`, while `sin(x)` reduces `x mod π/2` via Payne-Hanek. The two reductions produce *different* (q, r_hi, r_lo) tuples for the same x value. They are NOT interchangeable cache entries.

MomentStats has one strategy, one content. Trig has 3 × 5 = 15 content variants for the "same" x.

**3.4 Trig's intermediate is USEFUL AT BATCH; MomentStats is useful at bin.**

MomentStats was designed for the financial binning pattern: one bin, many downstream stats read the same reduction. The bin defines the sharing scope.

Trig is different. `trig_reduce` for `x = 1.5708` is useful *forever* — for every subsequent call of any trig function at exactly that x, across every bin, across every session. The sharing scope is `content_hash(x)`, not `bin_id`.

This suggests trig intermediates want a different cache strategy: content-addressed, longer-lived, potentially cross-session. The `data_id` field in the proposed tag is a `DataId` (content hash) precisely to support this.

*Implication*: trig_reduce is a *cross-bin* sharing candidate in a way MomentStats isn't. On a workload where the same x values recur (e.g., a fixed lookup table, a precomputed angle array, physical constants), the cache hit rate could be very high.

---

## 4. The answer to TRIG-4, in one sentence

> **Yes, sincos is the MomentStats of trig — at two tiers, with precision + angle-unit compatibility fingerprints, and with a sharing scope wider than MomentStats' bin-scope because trig is content-addressable.**

The analogy holds in the load-bearing ways (MSR, scatter-compute, fan-out). Where it breaks, the breaks are **structurally informative**:

- The two-tier structure (`trig_reduce` → `sincos_kernel`) is a signature that will repeat for hyperbolics and anywhere else there's a reduction-then-kernel pattern.
- The discrete quadrant coordinate is a group-theory tell; it will appear in any S¹-valued or Z_n-symmetric function.
- The precision + unit fingerprint is the Tambear Contract §3 compatibility requirement made concrete. Every shared intermediate for any libm recipe will need something like it.
- The content-addressable cache scope is a genuinely new sharing pattern that MomentStats doesn't use.

This is why the navigator's framing was so productive: asking "is trig a MomentStats?" isn't a yes/no question; it's a *structural alignment* question. The answer is yes-with-four-generalizations, and those four generalizations are findings in their own right.

---

## 5. Consequences for pathmaker

### 5.1 The IntermediateTag for trig

Recommend adding to `intermediates.rs`:

```rust
TrigReduce {
    data_id: DataId,
    angle_unit: AngleUnit,
    precision: PrecisionTier,
},
SincosKernel {
    data_id: DataId,
    angle_unit: AngleUnit,
    precision: PrecisionTier,
},
```

Tier 1 caches the reduction; Tier 2 caches the kernel. Both keyed by content hash + the two content-varying parameters.

### 5.2 The kernel tier only materializes when there are ≥2 consumers of the same (angle, precision) pair on the same bin

If only `sin(x)` is called, the `sincos_kernel` tier is wasteful — you computed `cos_k` and nothing reads it. The session should materialize the kernel tier lazily, only when a second consumer arrives. This is an implementation detail but worth flagging so the session doesn't eagerly cache half-used kernels.

### 5.3 The extraction path

The shared-pass analysis recommends moving:
- `reduce_trig` from `sin.rs` → `primitives/compensated/rem_pio2.rs` (Aristotle's P2/P3)
- `eval_sincos` from `sin.rs` → `recipes/libm/sincos_kernel.rs`
- `special_case_trig` from `sin.rs` → `recipes/libm/trig_special_cases.rs`

After extraction, `sin.rs` becomes ~100 lines of thin wrapper. `cos.rs`, `tan.rs`, etc. become similar. The mass moves to the shared module — which is the periodic-table structure expressing itself in the directory layout.

---

## 6. What this means for other families (preview)

The same analysis applied to the other families will probably land here:

- **Hyperbolics**: one-tier `exp_pair(x) = (exp(x), exp(-x))`; kernel derivations for sinh/cosh/tanh are trivial on top. Precision tier matters; angle_unit doesn't (hyperbolic functions take real inputs).
- **Inverse trig**: `atan2(y, x)` is the primitive, but it does NOT share a reduction with sin/cos — different coordinate, different reduction. May have its own two-tier shared pass (`atan_reduce` → `atan_kernel`), to be determined by math-researcher.
- **Inverse hyperbolics**: pure `log` compositions; no shared trig intermediate, but share the log reduction (`log(x+1) = log1p(x)` for near-unity arguments).
- **Pi-scaled**: shares `trig_reduce` *with a different unit fingerprint*; the reduction is `x mod 2` (exact) not `x mod π/2` (Payne-Hanek). Same intermediate type, different compatibility tag.

The compatibility-tag approach makes all of this work without proliferation: one `TrigReduce` tag type, many compatibility variants, one cache lookup per call.

---

## Final note

This analysis was triggered by navigator's direct framing ("is sincos the MomentStats of trig?"). pathmaker's sincos.spec.toml calls it "the MomentStats of trigonometry" in its very first comment line — the team has already adopted the framing. My contribution here was to *stress-test* the analogy and catalog where it breaks, because the breaks are more informative than the agreement.

The breaks tell us that sharing contracts in tambear need:
1. Compatibility fingerprints (precision, unit, strategy)
2. Two-tier (or n-tier) layered sharing for reduction-then-kernel patterns
3. Content-addressable scope for mathematically pure functions
4. Group-theoretic coordinates (coset representatives) for functions over symmetric domains

These aren't trig-specific findings. They're sharing-contract design guidance for every libm function, every correlation metric, every distance, every transform.

— the naturalist
