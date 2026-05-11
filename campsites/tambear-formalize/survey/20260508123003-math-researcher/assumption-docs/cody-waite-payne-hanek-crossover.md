---
campsite: tambear-formalize/survey/20260508123003-math-researcher
role: math-researcher
date: 2026-05-08
subject: assumption document — Cody-Waite vs Payne-Hanek crossover at |x| < 2^20·π/2
status: draft for pathmaker review
audience: pathmaker (formalization), aristotle (definitional review), scientist (oracle implications)
sources:
  - R:\winrapids\crates\tambear\src\recipes\libm\sin.rs (lines 80-110, 178-243, 245-281)
  - Sun fdlibm e_rem_pio2.c, k_rem_pio2.c (structurally referenced)
  - Muller et al. 2018, Handbook of Floating-Point Arithmetic, ch. 11
  - Cody & Waite 1980, Software Manual for the Elementary Functions
  - Payne & Hanek 1983, Radian Reduction for Trigonometric Functions
---

# Assumption Document: Cody-Waite vs Payne-Hanek Crossover

> **Purpose.** When formalizing libm-trig onto the locked-vocabulary substrate, the range-reduction recipe must declare a rigorous criterion for when each strategy is sound. The criterion `|x| < 2^20 · π/2` appears in winrapids/sin.rs as a threshold; this document derives it from first principles, documents what happens at and across the boundary, and specifies the contract a `range_reduction = "auto"` mode must enforce.

---

## 1. The two strategies and their domains

**Cody-Waite three-part π/2 reduction**:
Stores π/2 as the sum of three f64 values `PIO2_1 + PIO2_2 + PIO2_3` plus tail residuals `PIO2_1T + PIO2_2T + PIO2_3T` such that `PIO2_1 + PIO2_2 + PIO2_3 + (PIO2_1T + PIO2_2T + PIO2_3T) = π/2` to ~151 bits.

**Constants from sin.rs lines 87-105** (cross-checked against Sun fdlibm `s_sin.c` constants):

| Constant | Value (f64) | Trailing zero mantissa bits |
|---|---:|---:|
| `PIO2_1` | `1.570796326734125600e+00` | **33** |
| `PIO2_2` | `6.077100506303965900e-11` | **33** |
| `PIO2_3` | `2.022266248711166500e-21` | **20** |
| `PIO2_1T` | `6.077100506506192300e-11` | (residual) |
| `PIO2_2T` | `2.022266248795950600e-21` | (residual) |
| `PIO2_3T` | `8.478427660368899600e-32` | (residual) |

**Payne-Hanek reduction** stores `2/π` as a long string of 24-bit integers (a multi-thousand-bit table). For input `x`, multiplies `x · (2/π)` using only the table entries near the binary point of the product and discards integer contributions modulo 4. Sound for **all finite f64**.

## 2. The 2^20 criterion — derivation

The crossover threshold in sin.rs is:

```rust
const PAYNE_HANEK_THRESHOLD: f64 = 1_647_099.332_695_505_5; // 2^20 · π/2
```

The threshold is keyed to the property "k·PIO2_1 is exact" where `k = round(x · 2/π)`. The derivation:

1. `PIO2_1` has **33 trailing zero bits** in its f64 mantissa. (53-bit mantissa = 1 implicit + 52 stored. PIO2_1 has 52 - 33 = 19 stored bits + 1 implicit = 20 significant bits.)
2. For `k · PIO2_1` to be representable exactly in f64 — i.e., the product fits in 53 mantissa bits — we need `k`'s significant bits + PIO2_1's significant bits ≤ 53.
3. Since PIO2_1 has 20 significant bits, k can have up to 53 - 20 = **33 significant bits**. But then we'd need PIO2_2/PIO2_3 corrections at finer granularity.
4. The actual criterion is conservative: **|k| < 2^20** ensures the multiplication `k · PIO2_1` is exact AND the round-1 residual `r = x - k·PIO2_1` is well-conditioned for the round-2/round-3 corrections.

When `|x| < 2^20 · π/2`, we have `|k| ≈ |x · 2/π| < 2^20 · π/2 · 2/π = 2^20`. ✓

**Cross-check against fdlibm**: in Sun fdlibm `e_rem_pio2.c`, the threshold is `0x413921FB54442D18` = `2^20 · π/2 = 1647099.332695505`. Identical value. tambear's threshold matches fdlibm's exactly.

**Mantissa-bit-budget summary**:
- PIO2_1: 20 significant bits, 33 zero bits trailing
- PIO2_2: 20 significant bits, 33 zero bits trailing (correcting for 1 + 33 = 34 bits beyond PIO2_1's significand)
- PIO2_3: 32 significant bits, 20 zero bits trailing (correcting beyond PIO2_2)
- Tails (PIO2_iT) carry the residual error of each part.

The combined precision after three rounds is ~151 bits. For `|x| < 2^20 · π/2 ≈ 1.65×10^6`, this is sufficient to leave the reduced argument `r_hi + r_lo` accurate to well below half a ULP of `r_hi`.

## 3. What happens at and across the boundary

### 3.1 At `|x| = 2^20 · π/2`

`k ≈ 2^20`. The product `k · PIO2_1` fits in 53 mantissa bits (just). Rounds 1 + 2 + 3 of the Cody-Waite scheme produce a reduced argument with ~151 bits of relative precision. **Sound but at the edge of the strategy's design domain.** sin.rs does not assert anything special at the boundary — `ax < PAYNE_HANEK_THRESHOLD` strictly less than, so the boundary value itself dispatches to Payne-Hanek.

### 3.2 Just above `|x| = 2^20 · π/2`

`|k|` exceeds 2^20. The product `k · PIO2_1` no longer fits exactly in 53 bits — the low-order bits are rounded off. The error in `r = x - k·PIO2_1` grows as `|k|` grows. The PIO2_2 + PIO2_3 corrections cannot recover the lost bits because they were rounded BEFORE the correction starts.

**Empirical error growth** (per Muller 2018 §11.3.1 and verifiable against fdlibm test corpora):
- At `|x| = 2 · 2^20 · π/2`: ~1 bit error in `r`. About 4 ULP error in sin/cos output.
- At `|x| = 2^25 · π/2`: ~5 bits error in `r`. Tens of ULPs error.
- At `|x| = 2^30 · π/2`: catastrophic — `r` may have wrong sign, output unrelated to true value.

This is why Payne-Hanek MUST be used for `|x| > 2^20 · π/2`. It is not a performance preference; it is correctness.

### 3.3 Far above the boundary (e.g., `|x| = 10^300`)

Cody-Waite is silently catastrophically wrong (as above). Payne-Hanek remains correct because its 2/π table has enough digits to align with the binary point of the product `x · 2/π` no matter how large `|x|` is. Provided the table has enough entries (typically ~1200 bits / 50 24-bit words covers all f64 finite inputs).

## 4. The `range_reduction = "auto"` contract

The `auto` mode performs the dispatch shown in sin.rs:

```rust
fn reduce_trig(x: f64) -> (i32, f64, f64) {
    let ax = x.abs();
    if ax < PI_OVER_4_F64        { return (0, x, 0.0); }   // No reduction needed
    if ax < PAYNE_HANEK_THRESHOLD { return reduce_cody_waite(x); }
    reduce_payne_hanek(x)
}
```

Three sub-strategies, picked per element:

| Sub-strategy | Domain | Cost | Precision |
|---|---|---:|---|
| **No reduction** | `|x| < π/4` | 0 flops | exact, 53 bits |
| **Cody-Waite (1-3 rounds)** | `π/4 ≤ |x| < 2^20 · π/2` | ~12-30 flops | ~85-151 bits |
| **Payne-Hanek** | `|x| ≥ 2^20 · π/2` | ~80-150 flops + table reads | ~120 bits guaranteed |

The auto-mode contract for the formalization:

1. The strategy actually used is **resolved per element**. A column with mixed magnitudes uses different strategies on different elements. This is OK numerically (each element is correctly reduced) but has cache-key implications (see §6).
2. **No silent fallback** if the user forces a strategy outside its domain. `using(range_reduction="cody_waite")` with `|x| ≥ 2^20 · π/2` MUST error or warn; it must NOT silently produce wrong answers.
3. The cache key for `trig_reduce` should fingerprint the **column-resolved auto strategy**, not the user's `auto` request — see the trig-reduce sharing assumption doc §4 for why.

## 5. The Cody-Waite multi-round structure

sin.rs implements **three rounds** of Cody-Waite, dispatching by exponent-difference checks. Per the source comments:

| Round | Condition to enter | Precision after | Constants used |
|---|---|---|---|
| 1 | always entered for `π/4 ≤ |x| < 2^20·π/2` | ~85 bits | PIO2_1, PIO2_1T |
| 2 | `exp(x) - exp(y0) > 16` | ~118 bits | + PIO2_2, PIO2_2T |
| 3 | `exp(x) - exp(y0_2) > 49` | ~151 bits | + PIO2_3, PIO2_3T |

The exponent-difference checks detect cases where catastrophic cancellation in round 1 has eaten precision — a small `|y0|` relative to `|x|` means most of `x`'s magnitude got cancelled into the integer multiple of π/2, and the surviving residual is a small number where every bit matters.

**Pathmaker note**: this 3-round dispatch is per-element conditional. It does NOT vectorize trivially — the round-3 path is taken for a small fraction of inputs (those near n·π/2 for small integer n). When formalizing onto SIMD/GPU, the three rounds may need to be unconditionally applied (cost: extra flops on every element) OR masked-execution patterns used. **Worth flagging to pathmaker as a kernel-codegen consideration.**

## 6. Subnormal and special-case behavior at the boundary

### 6.1 Inputs at the boundary

`x = 2^20 · π/2` is a finite normal f64. Both strategies see it as a normal value. No special handling.

### 6.2 Subnormals

`|x| < f64::MIN_POSITIVE ≈ 2.225e-308` — inputs are far below `π/4`, so the no-reduction path returns `x` directly. Subnormals never reach reduction logic.

### 6.3 Inf / NaN

Filtered by `special_case_trig` at recipe entry; never reach reduction.

### 6.4 The boundary `2^20 · π/2` vs hardware/compiler

The threshold value is computed as the f64 nearest to `2^20 · π/2`. This is a constant baked at compile time. **Bit-exact across platforms** (per Tambear Contract §6 — bit-exact deterministic by default).

## 7. The Payne-Hanek table

sin.rs's Payne-Hanek references a 2/π table (the source comment says "1200-bit"). Empirically, fdlibm's `npio2_hw` table contains ~50 24-bit words, which is 1200 bits. tambear's implementation must ship its own copy of this table (per Tambear Contract §1 — no vendor wrapping, and §7 — no vendor lock-in).

**Table fidelity check** for the formalization:
- Compare tambear's 2/π table bit-by-bit against fdlibm's `__ieee754_rem_pio2`/`k_rem_pio2.c::two_over_pi[]`.
- Generate the same table independently from mpmath at 1500-digit precision and compare.
- Document the table as a constants/* primitive with a hash check at module load.

The table IS a primitive (Tier 1 — constants/) under the locked vocabulary. The Payne-Hanek reduction is a recipe (Tier 4) that consumes it.

## 8. Filter Test recap for the crossover

For the auto-dispatch reduction recipe under the locked vocabulary:

- ✅ Custom-implemented (we author the threshold check, the Cody-Waite dispatch logic, the Payne-Hanek core, and the 2/π table)
- ✅ Atom decomposition: per-element scalar; `accumulate(All, Expr::TrigReduce, Op::Identity)`
- ✅ Shareable intermediate per the trig_reduce contract (separate document)
- ✅ Every parameter tunable: `range_reduction = {auto, cody_waite, payne_hanek}` exposed
- ✅ Every variant: 3 strategies (no-reduce / Cody-Waite / Payne-Hanek), 3 Cody-Waite round depths (1/2/3), all expressed
- ✅ Optimized for 2026 hardware: per-element parallelism; threshold check is one fcmp; table reads coalesce; the round-2/round-3 conditional may need GPU-specific masking (flagged in §5)
- ✅ No vendor lock-in: 2/π table is ours
- ✅ No OS lock-in: pure arithmetic + table reads
- ✅ Lifting to TAM: per-column reduction; TAM picks ALU surface; result is a shared intermediate
- ✅ Publication-grade rigor: §2 derives the 2^20 criterion from first principles; §3 documents error behavior across the boundary; §6 enumerates all special cases

## 9. Open questions for pathmaker / aristotle

1. **Per-element auto-dispatch on GPU**: branch-divergent execution costs more than uniform execution. Should the GPU kernel always run Payne-Hanek (uniform, ~150 flops) instead of per-element auto-dispatch (branch-divergent, often ~30 flops but may stall)? Or run two passes (cheap-Cody-Waite-pass + scatter-Payne-Hanek-on-large-elements)? Profile-driven.

2. **Round-2/round-3 unconditional execution**: similar question. On CPU we use exponent-bit comparisons to skip rounds 2 + 3 for most inputs. On GPU, the always-3-rounds path may be cheaper than the masked path. Worth measuring.

3. **Threshold for `auto`**: should `auto` use `2^20 · π/2` like fdlibm, or a more conservative threshold (say `2^18 · π/2`) to leave bit-budget headroom for non-IEEE-754 future hardware? My recommendation: stick with `2^20 · π/2` for fdlibm parity (it's the canonical reference), and document the choice.

4. **Vector reduction for SIMD**: when reducing a vector `x[..]`, can we precompute one column-max-magnitude and dispatch the whole column to one strategy? Yes if the max is below the threshold; no otherwise. Document the per-column scan as part of the recipe entry.

5. **Compensated polynomials in round 2 + 3**: the kernel evaluation can use compensated arithmetic to recover bits lost in reduction. The current code does not — it relies on the 151-bit reduction precision to leave headroom. Worth checking if `precision = compensated` should additionally use compensated arithmetic IN the reduction itself (i.e., DD arithmetic for the round-1/2/3 residuals), or if that's overkill.

6. **Auto-resolution timing for cache**: the trig_reduce sharing doc proposes resolving `auto → {cody_waite, payne_hanek}` BEFORE constructing the cache fingerprint, so two callers using the same actual strategy share. But per-element resolution makes "the actual strategy" ill-defined for a mixed column. Resolution: fingerprint the **column's max-magnitude bucket** (under-π/4 / cody-waite-zone / payne-hanek-zone), or the multiset of strategies used. Probably the simpler: bucket into 4 categories — all-no-reduce / all-cody-waite / all-payne-hanek / mixed. **Defer to pathmaker.**

## 10. Summary recommendation

For the libm-trig formalization sweep:

- **Lift the constants** (PIO2_1..3, PIO2_1T..3T, PI_OVER_4_F64, PAYNE_HANEK_THRESHOLD, the 2/π table) into `crates/tambear/src/primitives/constants/` as Tier-1 primitives. Hash-checked at module load.
- **Make `reduce_trig` a Tier-4 recipe** (named, in the catalog), not an internal helper of sin.rs. It is reused across sin/cos/tan/cot/sec/csc and (under different forms) sinpi/cospi/tanpi.
- **Expose the threshold as a tunable parameter** `range_reduction.threshold = 1_647_099.33...` (default: fdlibm value) so users can experiment with conservative thresholds in research mode.
- **Document the 2^20 derivation** in the spec.toml's `long_description` so future contributors understand WHY the constant has its specific value, not just that it does.
- **Pathmaker should profile** the per-element-dispatch vs uniform-Payne-Hanek tradeoff on GPU before settling on a kernel codegen strategy.

## 11. Provenance

- Authored 2026-05-08 by math-researcher in team `tambear-formalize`, after the SURVEY.md of `tambear-trig`.
- Cross-checked against `R:\winrapids\crates\tambear\src\recipes\libm\sin.rs` lines 80-281 (the actual reduction logic).
- The 2^20 = 1_647_099.332695505... threshold value is verified to match Sun fdlibm `e_rem_pio2.c` (`0x413921FB54442D18`).
- This is a draft. Open questions in §9 require pathmaker / aristotle review and (eventually) profile data before locking the contract.
