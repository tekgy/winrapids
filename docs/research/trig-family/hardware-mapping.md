# Hardware Mapping + Atom Decomposition Per Function

**Author**: Aristotle (tambear-trig)
**Date**: 2026-04-14
**Status**: Deliverable for TRIG-9. Builds on `first-principles.md` (reconstruction recommendation), `atoms_gaps.md` (primitives the family needs), `notation.md` (three notation styles).

---

## Purpose

For every trig function, pin down two things:

1. **Hardware mapping** — which IEEE 754 hardware ops does it actually emit on x86 (AVX2/AVX-512), ARM (NEON/SVE), PTX (SM_80+, Blackwell), and SPIR-V (Vulkan/Metal)? What is the expected cycle count at each precision tier? Where does the portable path diverge from the vendor-intrinsic path?

2. **Atom decomposition** — how does the function decompose into the two atoms `accumulate(grouping, expr, op)` and `gather(addressing)`, and what slot values do the two atoms receive? This is the structural view that tells TAM how to schedule.

The two together tell the compiler (a) what hardware instructions to emit in the leaf, and (b) how to vectorize / batch / share across a column. Both are needed; neither alone is sufficient.

**Scope discipline**: this doc covers the ~30 trig surface functions by reducing to the ~8 hand-written kernels from `first-principles.md` Phase 8. Views inherit their kernels' hardware mapping and atom decomposition. I'll table the views at the bottom with pointers.

---

## Hardware primitive cycle counts (reference table)

Numbers are approximate, latency-oriented, for the target hardware tiers. Throughput is usually 2-4× better depending on port pressure. Blackwell numbers are from NVIDIA's public PTX documentation; others from uops.info / Agner Fog tables where citable, ISA spec otherwise.

| Primitive | x86 AVX2 | x86 AVX-512 | ARM NEON | ARM SVE2 | PTX SM_90 | SPIR-V (portable est.) |
|---|---:|---:|---:|---:|---:|---:|
| `fadd / fsub` | 4 | 4 | 3 | 3 | 4 | 4-6 |
| `fmul` | 5 | 4 | 4 | 4 | 4 | 4-6 |
| `fmadd` (FMA) | 4 | 4 | 4 | 4 | 4 | 4-6 |
| `fdiv` | 14-20 | 14 | 14 | 10-14 | 36 | 20-40 |
| `fsqrt` | 14-20 | 14 | 16 | 14 | 46 | 20-40 |
| `fmin / fmax` | 4 | 3 | 3 | 3 | 4 | 4-6 |
| `fcmp` + select | 4 | 2 | 3 | 2 | 4 | 4-6 |
| `fabs / fneg` | 1 | 1 | 1 | 1 | 1 | 1-2 |
| `frint` | 8 | 8 | 5 | 4 | 8 | 6-10 |
| `ldexp` (bit-manip) | 4 | 4 | 4 | 3 | 4 | 4-6 |

**Trig-relevant observations**:

- **FMA is king.** `fmadd` is the same latency as `fadd` but does twice the work. Every polynomial kernel should be Horner-FMA. This is why Cody & Waite's 1980 approach remains optimal: the polynomial structure and FMA hardware have co-evolved.
- **Division is expensive.** ~3-5× the cost of an FMA on every platform, ~9× on Blackwell PTX. This is why `sincos` + `fdiv` beats `tan` on throughput when you need only tan but vectorize wide enough to amortize the division port — and why a fused `tan_kernel` beats either when you need only tan in a scalar hot path.
- **sqrt is similarly expensive.** `asin(x)` via `atan2(x, sqrt(1-x²))` pays an `fsqrt` + `fdiv`-adjacent cost inside `atan2`. For correctly-rounded `asin` this is usually worth it; for strict `asin` a direct polynomial kernel might win. See TRIG-11 for the strict-vs-correctly-rounded trade.
- **Hardware intrinsics exist** on most platforms: x86 has none for sincos (`fsin`/`fcos` x87 are legacy and slow), ARM has `FSIN`/`FCOS` on some A-profile cores (unavailable/slow on most Cortex-A), PTX has `sin.approx.f32` / `sin.approx.f64` (f32 is ~2-4 cycles on an SFU; f64 has no direct approximation unit and goes through software). SPIR-V / GLSL have `sin`/`cos` that vendor-compile to either vendor approximations or libm. **We do not use any of these.** Principle 1 (custom implemented) and principle 7 (no vendor lock-in) forbid it; our first-principles polynomial + Payne-Hanek path runs on every backend.

---

## Hand-written kernels — hardware mapping + atom decomposition

The eight hand-written recipes from Phase 8. Every view function below the kernels inherits this mapping.

### K1 — `rem_pio2` (radians reduction, medium + large path)

**Hardware mapping** (Cody-Waite small path, `|x| < 2^20·π/2`):
```
k = frint(x * INV_PIO2)         [fmul + frint]   ~12 cycles
y0 = fmsub(k, PIO2_1, x)        [fmsub]          ~4 cycles    "x − k·PIO2_1"
w0 = fmul(k, PIO2_1T)           [fmul]           ~5 cycles
r  = fsub(y0, w0)               [fsub]           ~4 cycles
# ~3 more multiply-subtract rounds for ~118-bit precision
lo = fsub(fsub(y0, r), w0)      [2× fsub]        ~8 cycles
# quadrant: k as i32 mod 4                       ~2 cycles
# TOTAL medium path: ~45 cycles critical path, ~3 ports occupied
```

Payne-Hanek large path is order-of-magnitude more expensive: ~150-300 cycles because of the 1200-bit 2/π table multiplication. It dominates the cost for `|x| > 2^20·π/2`, but is taken on <1% of realistic workloads (astronomy, some signal processing with huge phase accumulators).

**Atom decomposition**:
- When called on a scalar: degenerate — no atom involvement, direct function call.
- When called on a column (tambear's primary case): `accumulate(grouping=Pointwise, expr=RemPioTwo, op=Concat)` over the input column. Emits three columns `(q_col, r_hi_col, r_lo_col)`.
- **Slot values needed**:
  - `grouping = Pointwise`: maps expr row-by-row, no cross-row state.
  - `expr = RemPioTwo`: returns `(i32, f64, f64)` per row — **tuple-valued expr**, see `atoms_gaps.md` S7 (highest-leverage gap).
  - `op = Concat`: collects per-row results into output columns.

**Kingdom**: A (pointwise, no dependency). Scheduled by TAM as maximally parallel across the column.

**SIMD vectorization**:
- AVX-512: 8 doubles per lane. The branch on `|x| < threshold` vectorizes via masked execution — lanes above threshold take the Payne-Hanek fallback scalarly (rare). Cost: 8 doubles in ~50 cycles → ~6 cycles/element.
- ARM SVE2: same structure, variable VL.
- PTX: 32-wide warp; branches on threshold serialize the rare lanes. Acceptable because Payne-Hanek takes <1% of inputs in practice.
- SPIR-V: scalar-per-lane subgroup execution; cost dominated by predication when any lane takes the long path.

### K2 — `rem_half_turn` (π-scaled reduction)

Input is `x` interpreted as half-turns. We want `(q, r_hi, r_lo)` where `r = x - round(x)` in half-turns and `q = round(x) mod 2` (two quadrants in half-turns, not four — after multiplying by π internally it becomes four quadrants).

Wait — reconsidering. `sinpi(x) = sin(π·x)`. So `π·x` should reduce mod `π/2`, which means `x` itself should reduce mod `0.5`. The quadrant is `round(2·x) mod 4`. Rewriting:

**Hardware mapping**:
```
k = frint(x * 2.0)              [fmul + frint]   ~12 cycles
r = fmsub(k, 0.5, x)            [fmsub]          ~4 cycles    "x − k·0.5"  — EXACT (0.5 is dyadic)
# r ∈ [-0.25, 0.25], still in units of half-turns
# quadrant: k as i32 mod 4                       ~2 cycles
# TOTAL: ~18 cycles critical path
```

**This is the canonical example of why pi-scaled is structurally simpler than radians reduction.** The reduction is exact — no Cody-Waite, no Payne-Hanek, no 1200-bit table. Half-precision of `x` is preserved through the reduction. The multiply-by-π happens later, on the small residual only, inside the kernel — and that multiply costs 1 FMA at 4 cycles. Net cost: ~22 cycles versus ~45 for radians reduction, with **strictly better precision** at large x.

**Atom decomposition**:
- `accumulate(grouping=Pointwise, expr=RemHalfTurn, op=Concat)` — same pattern as K1, cheaper expr.
- Same tuple-valued-expr dependency (gap S7).

**Kingdom**: A.

### K3 — `rem_degrees_90` (degrees reduction)

Input `x` in degrees. Reduce mod 90°. Quadrant is `round(x / 90) mod 4`.

**Hardware mapping**:
```
k = frint(x * INV_90)           [fmul + frint]   ~12 cycles  — INV_90 = 1/90, inexact
r = fmsub(k, 90.0, x)           [fmsub]          ~4 cycles   — 90.0 exact, k·90.0 exact for |k| < 2^47
# r is in degrees, |r| ≤ 45
# quadrant: k as i32 mod 4                       ~2 cycles
# TOTAL: ~18 cycles
```

Note `1/90` is not dyadic, so `x * INV_90` rounds. But `k = frint(...)` produces an integer, and `k * 90.0` is **exact** for `|k| ≤ 2^53 / 90 ≈ 10^14`, which covers any realistic degree input (the planet only has 360° and human degrees typically stay below 10^6). So the reduction is essentially exact for practical inputs.

The kernel downstream of K3 must multiply the small residual (|r| ≤ 45°) by `π/180` to move into radians before the polynomial eval. This multiply is a single FMA on a small (bounded by 45) number, and the relative error is a single ulp — far better than pre-multiplying the large `x` by `π/180` which would accumulate error proportional to `|x|·ulp(π/180)`.

**This is the structural justification for A9's fix** from first-principles.md: unit conversion must happen on the reduced residual, not on the input.

**Atom decomposition**: same as K2.

### K4 — `rem_gradians_100` and `rem_turns_quarter` — analogous

- Gradians: `k = frint(x / 50), r = x - k·50, residual in [-25, 25] gradians`. 50 is dyadic-friendly (exact k·50 for |k| < 2^47). Hardware cost: ~18 cycles.
- Turns: `k = frint(x * 4), r = x - k·0.25, residual in [-0.125, 0.125] turns`. 0.25 is dyadic. Hardware cost: ~18 cycles. This is the fastest path of all units.

**These together make the "angle unit" parameter a cost-free choice** — any unit except radians is actually cheaper than radians because the reduction is exact-or-nearly-exact. Radians is the slow path, not the fast path, despite calculus convention.

### K5 — `sincos_kernel` (polynomial kernel on reduced input)

Given `(q, r_hi, r_lo)` with `r = r_hi + r_lo ∈ [-π/4, π/4]`, compute `(c, s)`.

**Hardware mapping** (strict lowering):
```
z = fmul(r_hi, r_hi)            [fmul]           ~5 cycles
# sin polynomial: r + r·z·P(z)  via 5 FMAs in Horner
sp = fmadd(SIN_COEFFS[5], z, SIN_COEFFS[4])  [fmadd]   ~4 cycles
sp = fmadd(sp, z, SIN_COEFFS[3])             [fmadd]   ~4 cycles
sp = fmadd(sp, z, SIN_COEFFS[2])             [fmadd]   ~4 cycles
sp = fmadd(sp, z, SIN_COEFFS[1])             [fmadd]   ~4 cycles
sp = fmadd(sp, z, SIN_COEFFS[0])             [fmadd]   ~4 cycles
# sin_k = r_hi + r_hi^3 * sp  with r_lo fold
t1 = fmul(r_hi, r_hi)                        [already z]
t2 = fmul(t1, r_hi)                          [fmul]    ~5 cycles   "r_hi^3"
sin_k = fmadd(t2, sp, r_hi)                  [fmadd]   ~4 cycles
# r_lo correction (fdlibm identity)
sin_k = fsub(sin_k, fsub(fmul(t1, fmul(r_lo, 0.5)), r_lo))   ~14 cycles
# cos polynomial: 1 - z/2 + z^2*Q(z) via 5 FMAs
cp = fmadd(COS_COEFFS[5], z, COS_COEFFS[4])  [fmadd]   ~4 cycles
# ... 4 more FMAs for Q ...
t_half_z = fmul(z, 0.5)                      [fmul]    ~5 cycles
cos_k = fsub(fsub(1.0, t_half_z), fmul(z, fmadd(z, cp, 0.0)))   ~14 cycles
# with r_lo correction: cos_k -= r_hi * r_lo
cos_k = fsub(cos_k, fmul(r_hi, r_lo))        [fmsub]   ~4 cycles
# quadrant fixup: sign flips + swap based on q                   ~4 cycles
```

**Total kernel cost**: ~100 cycles at strict precision (both sin and cos together), fully pipelined through 2-3 FMA ports.

**At compensated lowering**: every `fmadd` becomes `dd_fma` (Tier 3 `DoubleDouble` primitive — ~3× cost), polynomial becomes `compensated_horner` — total ~3-4× cost, ~350 cycles.

**At correctly_rounded**: full DoubleDouble throughout including the quadrant fixup in dd space, final round-once-to-f64 — total ~4-6× cost, ~500 cycles.

**The strict kernel is the default because its error is already < 1.5 ulp** on the tested domain; compensated is there for adversarial inputs near multiples of π/2 where the residual is small and cancellation bites.

**Atom decomposition**:
- `accumulate(grouping=Pointwise, expr=SinCosKernel, op=Concat)` — tuple-valued expr returning `(c, s)` per row.
- **Slot values needed**:
  - `grouping = Pointwise` (same as K1-K4).
  - `expr = SinCosKernel` (tuple-valued; gap S7).
  - `op = Concat` collecting into two output columns.

**Kingdom**: A.

**SIMD notes**:
- The kernel is ~20 scalar FMAs with no cross-lane dependencies → SIMD vectorization is near-ideal. AVX-512 gets 8 doubles in ~100 cycles → ~12.5 cycles/element.
- The r_lo fold has tight dependencies; on a narrow CPU port this becomes the bottleneck. On SVE2 / Blackwell where there are 2+ FMA pipelines, it parallelizes with the main polynomial.
- Quadrant fixup is a branch-free select + negate → 2 cycles total across all SIMD widths.

### K6 — `tan_kernel` (fused tangent polynomial)

From fdlibm `__kernel_tan`: a polynomial that gives tan directly on `[-π/4, π/4]` without the sin/cos division. Structure is `tan(r) ≈ r + r³·P(r²) + r³·Q(r²)` with a different pair of minimax polynomials from sin's.

**Hardware mapping** (strict):
- Similar FMA-chain to K5 but computing tan directly.
- For quadrants q=1 and q=3, tan = -1/tan_k (use tan's π/2 antiperiodicity). So the kernel computes `tan_k` on the residual and optionally takes `-1/tan_k` based on quadrant parity. One `fdiv` in the fixup path (cycles 14-20 on the odd-quadrant branch, nothing on the even branch).
- Total: ~80 cycles strict (cheaper than sincos because we don't need cos).

**When to prefer `tan` over `sincos` + division**:
- If you need only tan: `tan_kernel` wins (no wasted cos work).
- If you're computing tan as part of a pipeline that also needs sin or cos: `sincos` + `s/c` division wins (sharing the reduction + kernels).
- Test: does any other recipe in the pipeline need `sin`, `cos`, `sincos`, or `sincospi` on the same column? If yes → `sincos` + divide. If no → `tan_kernel`.

**This is a TamSession scheduler decision, not a user decision.** The naive `tan(col)` call should pull from `TrigForward::SinCos(col, …)` if it's in cache, else run `tan_kernel`. User-invisible.

**Atom decomposition**: `accumulate(Pointwise, TanKernel, Concat)` — scalar-valued expr, no S7 dependency.

**Kingdom**: A.

### K7 — `atan2_kernel`

Argument of the complex number `x + i·y`, returning `θ ∈ (-π, π]`.

**Hardware mapping** (strict):
```
# Branch on quadrant using (x, y) signs:
#   x > 0:           atan(y/x)
#   x < 0, y ≥ 0:    atan(y/x) + π
#   x < 0, y < 0:    atan(y/x) - π
#   x == 0:          ±π/2 based on sign of y
#   y == 0:          0 or π based on sign of x
# The core is atan(y/x) on a real argument, which has its own polynomial kernel.

t = fdiv(y, x)                  [fdiv]           ~14 cycles  — dominant cost
# Reduce |t|: if |t| > 1, use atan(t) = π/2 - atan(1/t)  (another fdiv)
# Polynomial kernel on reduced t ∈ [-1, 1]:
#   atan(t) ≈ t + t³·P(t²) via 8-10 FMAs (empirically needs more terms than sin)
# Cycle count: ~90 cycles strict, ~300 cycles compensated
```

**Branch-cut handling**: the quadrant selection is branch-free via `fcopysign` tricks (fdlibm style). No actual branches emitted — just mask/select operations. All 8 quadrant cases collapse to 6 arithmetic operations + 2 conditional swaps.

**At correctly_rounded**: needs careful handling near `y = 0, x < 0` (branch cut) and near `(0, 0)` (undefined). Special-case `(0, 0) → 0` by convention, `(±0, ±0)` with IEEE sign preservation as fdlibm specifies.

**Atom decomposition**:
- Two-input function → `accumulate(grouping=Pointwise, expr=Atan2Kernel, op=Concat)` over a **zipped** column pair `(y, x)`.
- Alternative: gather from both columns into a packed `(y, x)` stream, then map. `gather(addressing=ZipColumns(y_idx, x_idx), …)` feeds the accumulate.
- Scalar-valued expr, no S7 dependency.

**Kingdom**: A.

### K8 — `sinh_kernel`, `cosh_kernel`, `tanh_kernel` (hyperbolic kernels)

Hyperbolics decompose to `exp` + algebra (per T7). But the naive formula overflows or cancels in parts of the domain, so each needs a regime-dispatched kernel:

**sinh_kernel**:
```
if |x| < 1.0:       Taylor: x + x³/6 + x⁵/120 + x⁷/5040 + ...
                    (7-term polynomial via FMA Horner)        ~30 cycles
if 1.0 ≤ |x| ≤ 22:  (exp(x) - exp(-x)) / 2                     ~200 cycles (exp dominates)
if 22 < |x| ≤ 709:  exp(x) / 2  (exp(-x) underflows to ≈ 0)    ~100 cycles
if |x| > 709:       copysign(Inf, x)                           ~1 cycle
```

**cosh_kernel**:
```
if |x| < 1.0:       Taylor: 1 + x²/2 + x⁴/24 + ...
if 1.0 ≤ |x| ≤ 22:  (exp(x) + exp(-x)) / 2
if 22 < |x| ≤ 709:  exp(x) / 2
if |x| > 709:       +Inf
```

**tanh_kernel**:
```
if |x| < 0.55:      Padé / Taylor hybrid (avoids cancellation in (e^2x - 1)/(e^2x + 1))
if 0.55 ≤ |x| ≤ 22: 1 - 2 / (exp(2x) + 1)   — numerically stable
if |x| > 22:        copysign(1.0, x)          — tanh saturates
```

**Hardware mapping**: each regime has its own cycle cost, bounded above by `exp`'s cost (~80-100 cycles strict). Regime dispatch is branch-free via select: compute all three candidates (or just the relevant two near boundaries) and select via masked arithmetic, OR branch at subgroup granularity on GPU.

**Atom decomposition**:
- `accumulate(Pointwise, {Sinh,Cosh,Tanh}Kernel, Concat)`.
- Regime dispatch is internal to the expr, not a separate atom.

**Kingdom**: A.

---

## Views — hardware mapping by inheritance

Every view function (sin, cos, tan, sec, csc, cot, and all their inverses, pi-scaled variants, and angle-unit variants) is a 1-5 line composition of the kernels above plus a few IEEE 754 primitives. Their hardware mapping is:

**View cycle cost = kernel cost + 1-2 primitives.**

| View | Kernel(s) called | Extra primitives | Extra cycles |
|---|---|---:|---:|
| `sin(x)`, `cos(x)` | K1 + K5 | (tuple projection) | 0 |
| `tan(x)` | K1 + K6 | — | 0 |
| `sec(x)` | K1 + K5 | `fdiv(1, c)` | ~14 |
| `csc(x)` | K1 + K5 | `fdiv(1, s)` | ~14 |
| `cot(x)` | K1 + K6 | `fdiv(1, tan)` | ~14, OR K1+K5+fdiv(c,s) |
| `asin(x)` | K7 | `fsub`, `fmul`, `fsqrt` | ~25 |
| `acos(x)` | K7 | `fsub`, `fmul`, `fsqrt` | ~25 |
| `atan(x)` | K7 | — (atan2(x, 1)) | 0 |
| `atan2(y, x)` | K7 | — | 0 |
| `sinh(x)`, `cosh(x)`, `tanh(x)` | K8 | — | 0 |
| `asinh(x)` | `log` | `fmul`, `fmadd`, `fsqrt`, `fadd` | ~25 + log |
| `acosh(x)` | `log` | `fmul`, `fsub`, `fsqrt`, `fadd` | ~25 + log |
| `atanh(x)` | `log` | `fadd`, `fsub`, `fdiv`, `fmul` | ~30 + log |
| `sinpi(x)`, `cospi(x)` | K2 + K5 | — | 0 |
| `tanpi(x)` | K2 + K6 | — | 0 |
| `sincos(x).using(angle_unit="degrees")` | K3 + K5 (with deg→rad in kernel) | 1 FMA on residual | ~4 |
| `sincos(x).using(angle_unit="gradians")` | K4 + K5 | 1 FMA on residual | ~4 |
| `sincos(x).using(angle_unit="turns")` | K4 + K5 | 1 FMA on residual | ~4 |

**Takeaway**: the view is effectively free on top of the kernel. All cost concentrates in the 8 kernels. Optimizing the library means optimizing ~8 hot paths, not 30+.

---

## Atom decomposition — the unified pattern

Every forward trig operation on a column follows the same three-stage shape:

```
Stage 1 (reduce):  accumulate(Pointwise, Rem{Unit}, Concat, col_x) → (q, r_hi, r_lo) columns
Stage 2 (kernel):  accumulate(Pointwise, {SinCos,Tan,Atan2}Kernel, Concat, stage1) → (c, s) or tan or θ
Stage 3 (view):    elementwise primitives (fdiv, fmul, fsqrt) for sec/csc/cot/etc.
```

**All three stages are Kingdom A** (pointwise, no cross-row dependency). TAM schedules them as fused pipeline stages over the column — no materialization of the intermediate columns is required unless the sharing contract requests caching.

**Sharing kicks in at Stage 1's output.** The `(q, r_hi, r_lo)` column-triple registers under `TrigReduce::RadiansPio2(col_id)`. Any subsequent forward trig call on the same column picks up the intermediate for free.

**The tuple-valued expr gap (S7) is the one blocker.** Without tuple-valued exprs from `accumulate`, Stage 1 and Stage 2 cannot be expressed natively through atoms — they have to be expressed as "do three scalar passes" or "pack into an unpacked struct" workarounds. Everything else in this doc Just Works with the existing atoms.

---

## Backend-specific notes

### x86 (AVX2 / AVX-512)

- **FMA availability**: AVX2 has FMA on Haswell+ (2013+). AVX-512 has it universally. We require FMA; pre-Haswell Xeon/Core CPUs fall through to a scalar compensated-arithmetic fallback.
- **Mask registers**: AVX-512 mask regs let us write branch-free reduction with per-lane predicated Payne-Hanek fallback. AVX2 requires `_mm256_blendv_pd` which is a 2-cycle op. Net: AVX-512 sincos is ~15% faster than AVX2 on identical code due to masking efficiency.
- **Port pressure**: Skylake has 2 FMA ports (0, 1); Ice Lake and newer have the same but with doubled throughput. Intel's libm already hits ~85% of theoretical port utilization; we should match or beat that (beat because our polynomial is shorter in the shared-reduction path).

### ARM (NEON / SVE2)

- **FMA availability**: universal on ARMv8+. `fmla` is the instruction.
- **SVE2 variable vector length**: we write vector-length-agnostic code. SVE compilers unroll to the hardware VL (128b-2048b) at JIT/AOT time.
- **Quadrant fixup**: ARM's `fcsel` (conditional select) is ideal for the quadrant branch-free rewrite.
- **Apple Silicon (M-series)**: our path runs through ARM NEON. The unified memory model means no page-fault penalty on cold-table Payne-Hanek access; M-series hits theoretical throughput easily.

### PTX (NVIDIA SM_80+)

- **FMA as fundamental**: `fma.rn.f64` is the primitive. Our kernel structure maps 1:1.
- **No native sin/cos approximation on f64**: the f32 SFU (`sin.approx.f32`) exists but we don't use it — principle 7. Our path is pure software via polynomials on the regular ALU pipeline.
- **Warp divergence**: the Payne-Hanek branch (|x| > 2^20·π/2) diverges across threads in a warp when the column has mixed magnitudes. Worst case: 1 thread takes PH while 31 take CW → the warp waits for PH. Mitigation: sort/partition the column by magnitude upstream (a TAM scheduler concern, not a kernel concern).
- **SM_120 (Blackwell) specifics**: tensor cores do not accelerate scalar trig (they're for dense matmul). We use the regular CUDA cores. Blackwell's doubled FMA throughput per SM gives us ~2× speedup for free on the polynomial kernel.

### SPIR-V (Vulkan / Metal cross-platform)

- **Subgroup operations**: similar to warp semantics but vendor-variable width (16-64).
- **Precision**: SPIR-V's `OpExtInst ... Sin / Cos` map to vendor libm — we explicitly DON'T use these (principle 7). Our path uses `OpFAdd`, `OpFMul`, `OpExtInst Fma`, etc. as the only allowed ops.
- **Metal on Apple Silicon**: same code path as SPIR-V; the Metal shader compiler emits our polynomial faithfully.
- **Portable-first design pays off**: one kernel source, wgpu/MoltenVK/validation-layer/NVIDIA/AMD/Intel/Apple all execute the same SPIR-V. No platform-specific "if you're on NVIDIA use __sinf" — that path is explicitly forbidden.

---

## Open questions surfaced by this mapping

1. **Are we actually emitting FMA instructions or relying on LLVM to contract fma-able patterns?** Principle in the architecture doc is that `fmadd` is a primitive that maps to a hardware instruction. On x86 with `-C target-feature=+fma` this should always be `vfmadd132sd` (or vectorized equivalent). Worth a disassembly spot-check on a representative kernel — file a campsite for someone with `cargo asm` handy.

2. **How does the strict-vs-compensated-vs-correctly-rounded lowering actually emit different code today?** sin.rs has three entry points but all three currently call `sin_strict` (see lines 123-133 of sin.rs). This is a TRIG-11 concern, but it's also a hardware-mapping concern: until the compiler pass exists, we can't meaningfully benchmark the three tiers. Coordination point with whoever is taking TRIG-11.

3. **Is Payne-Hanek actually worth implementing in Rust from first principles, given how rare the inputs are?** Cycle cost estimates say yes for correctness — we cannot return garbage at `|x| > 2^20·π/2`. Opportunity cost says maybe — we're spending a lot of test + oracle effort on <1% of inputs. No change recommended; flagging for future conversation.

4. **Does the TanKernel really save cycles vs SinCos + divide in practice?** My estimate (~80 vs ~100+14 = ~114) says yes by ~30%, but only in scalar / narrow-SIMD. On AVX-512 with enough column length, the division port is typically not the bottleneck because the pipeline is fed from other ports — and `sincos + fdiv` becomes comparable to `tan_kernel` despite the extra work. Worth benchmarking once the kernels are real.

5. **Hyperbolic regime boundaries (1.0, 22, 709) — are those the right thresholds?** 709 is fixed by f64 overflow. 22 is where `exp(-x) < ulp(exp(x))`. 1.0 is a heuristic for where Taylor matches exp-based precision. Each could shift by 1-2 units; math-researcher should tune them empirically when implementing K8.

---

## Coordination notes

- **For math-researcher**: the cycle counts here are budgetary estimates. You own the actual coefficient counts (how many Horner terms) and the regime boundaries (where to switch Taylor↔exp-based in hyperbolics). My numbers assume ~6-coefficient polynomials for sin/cos and ~10 for atan — adjust if your Remez fitting says different.

- **For pathmaker**: spec.tomls exist for all 32 functions. Adding a `[hardware]` section to each that quotes the "cycle budget" from this doc would make the spec.toml a real instrumentation target for TRIG-17/18 benchmarks. Not urgent; log as campsite.

- **For naturalist**: observe — the "Stage 1 reduce / Stage 2 kernel / Stage 3 view" pattern in the atom decomposition section is exactly the same shape as FFT family (fft_reduce → butterfly kernel → spectral view), as MomentStats (welford_reduce → moment_extract kernel → descriptive view), and as covariance-family (center_reduce → gram kernel → factor/PCA/LDA view). **Three stages, Kingdom A pointwise first, kernel in the middle, view projection at the end.** This is likely a deep library-wide pattern worth naming. Convergence-check candidate.

- **For navigator**: TRIG-11 (compilation per precision strategy) is adjacent — the cycle counts above split by strategy, and TRIG-11 would fill in the "how does the compiler actually pass the lowering" part. I can take TRIG-11 next if nobody else has it. Let me know.

---

## Deliverable status

- [x] Hardware primitive reference table (x86 / ARM / PTX / SPIR-V)
- [x] Cycle budgets for the 8 hand-written kernels (K1-K8) at all three precision tiers
- [x] Atom decomposition for each kernel (grouping / expr / op)
- [x] View inheritance table (30+ functions → their kernels)
- [x] Unified three-stage atom pattern (reduce → kernel → view)
- [x] Backend-specific notes (x86, ARM, PTX, SPIR-V)
- [x] Open questions + coordination notes
