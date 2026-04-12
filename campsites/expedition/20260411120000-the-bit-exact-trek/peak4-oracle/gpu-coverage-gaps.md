# GPU Test Coverage Gap Analysis

*Produced by adversarial mathematician, 2026-04-11.*
*Feeds campsite 4.6 (porting GPU tests to new harness) and 4.7 (hard-cases suite).*

## The fundamental gap

Every test in `tests/gpu_end_to_end.rs` uses this assertion pattern:

```
assert!((gpu_v - cpu_v).abs() < 1e-N)
```

That checks **CPU-GPU self-consistency**, not **correctness**. If both backends compute
the same wrong answer (e.g., one-pass variance on large-mean data), both tests pass and
no alarm fires. The tests provide zero protection against algorithmic bugs that affect
all backends uniformly.

The oracle for every test must be an independent closed-form or mpmath reference value,
not the CPU result.

---

## Per-test analysis

### T1 — `gpu_sum_matches_cpu`

**Recipe**: `sum()` — one slot: `Σ x`, Op::Add  
**Input**: `[1..=10_000]`, all positive integers  
**N**: 10,000

**Checks**: CPU result == 50,005,000 (closed-form ✓), GPU within 1e-6 of CPU  

**Missing inputs**:

| Gap | Why it matters |
|-----|----------------|
| NaN in input | `atomicAdd` with NaN → undefined behavior. CPU sum returns NaN; GPU atomicAdd with NaN has undefined PTX semantics — may silently drop or propagate depending on driver. |
| Empty buffer (N=0) | Grid of 0 elements: grid = max(0/256, 1) = 1. One block launches, `gi` starts at 0, loop `i < 0` never fires. `acc_0` stays 0.0. `atomicAdd` writes 0.0. Should be 0.0 — but kernel launches 256 threads for zero work. Not tested. |
| Single element | N=1: should return that element. Grid=1, one thread active. Works — but untested. |
| Large-N overflow | N such that the exact sum exceeds f64::MAX. Not hit at N=10,000 but worth one test near the overflow boundary. |
| Negative values | [-N..0]: sum = -N*(N+1)/2. Untested sign handling. |

---

### T2 — `gpu_mean_matches_cpu`

**Recipe**: `mean_arithmetic()` — two slots: `Σ x`, `count`, gather `sum/count`  
**Input**: `[1..=1,000]`  
**N**: 1,000

**Checks**: CPU within 1e-12 of 500.5 (closed-form ✓), GPU within 1e-9 of CPU

**Missing inputs**:

| Gap | Why it matters |
|-----|----------------|
| NaN in input | mean of `[1.0, NaN, 3.0]` — CPU should return NaN (sum is NaN); GPU atomicAdd with NaN has platform-dependent behavior. |
| Single element | mean of `[42.0]` = 42.0. Gather divides by count=1 — works but untested. |
| Large-mean data | Mean of `[1e14, 1e14+1, 1e14+2, ...]` — not a catastrophic cancellation problem for mean itself, but worth verifying GPU agrees to full precision. |

---

### T3 — `gpu_variance_matches_cpu`

**Recipe**: `variance()` — three slots: `Σ x`, `Σ x²`, `count`, one-pass formula in gather  
**Input**: `(0..10_000).map(|i| (i as f64).sin() * 10.0 + 5.0)` — sinusoidal, mean ≈ 5.0  
**N**: 10,000

**Checks**: GPU relative error < 1e-10 vs CPU

**Critical gaps**:

| Gap | Why it matters |
|-----|----------------|
| **Large-mean data** | Input mean is ~5.0 — this is the SAFE case. The one-pass formula only fails when mean >> spread. A test with mean=1e9 and spread=1e-3 would FAIL ON BOTH CPU AND GPU identically (negative variance), and this test would never catch it because it only checks CPU-GPU agreement. |
| NaN input | variance of `[1.0, NaN, 2.0]` — should return NaN. CPU returns NaN (sum_sq accumulates NaN). GPU: atomicAdd(NaN) is UB in CUDA — may silently produce garbage. |
| Empty input | variance of `[]` — should be NaN (N=0, division by zero in gather). GPU: kernel dispatches 0 elements, outputs 0.0/0.0 in gather? |
| Single element | variance of `[x]` — sample variance requires n≥2. Gather computes `(0 - x²/1)/0` — division by zero. Should be NaN or inf. |
| Subnormal inputs | variance of subnormals — the squared values are deeply subnormal, may flush to zero with FTZ. |

---

### T4 — `gpu_rms_matches_cpu`

**Recipe**: `mean_quadratic()` — `Σ x²` + `count`, gather `sqrt(ss/count)`  
**Input**: `[1..=2048]`  
**N**: 2,048

**Checks**: GPU relative error < 1e-12 vs CPU

**Missing inputs**:

| Gap | Why it matters |
|-----|----------------|
| NaN propagation | `sqrt(NaN)` — both CUDA's `sqrt` and Rust's `f64::sqrt` produce NaN. But the path to NaN via accumulation differs: CPU sum_sq accumulates NaN, GPU atomicAdd(NaN) is UB. |
| Subnormal inputs | `mean_quadratic` of `[MIN_POSITIVE/2; N]` — squares are deeply subnormal. FTZ on GPU would flush to 0.0 and return 0.0 instead of a non-zero subnormal. |
| Negative inputs | RMS of negatives should work identically (squares are always positive). Untested. |

---

### T5 — `gpu_sum_of_squares_matches_cpu`

**Recipe**: `sum_of_squares()` — `Σ x²`  
**Input**: `(0..5_000).map(|i| (i as f64) / 100.0)` — range [0.0, 49.99]  
**N**: 5,000

**Checks**: GPU relative error < 1e-11 vs CPU

**Missing inputs**:

| Gap | Why it matters |
|-----|----------------|
| Near-f64::MAX inputs | `(f64::MAX / N as f64)^2 = f64::MAX^2/N^2` — overflows. The overflow behavior (inf vs NaN) must be consistent between CPU and GPU. |
| NaN inputs | Single NaN should propagate to sum_sq. |
| Subnormal inputs | `x²` of a subnormal is deeply subnormal — FTZ check. |

---

### T6 — `gpu_l1_norm_matches_cpu`

**Recipe**: `l1_norm()` — `Σ |x|`  
**Input**: `(0..4096).map(|i| if i%2==0 { i as f64 } else { -(i as f64) })` — alternating signs  
**N**: 4,096

**Checks**: GPU relative error < 1e-12 vs CPU

**Missing inputs**:

| Gap | Why it matters |
|-----|----------------|
| NaN input | `|NaN| = NaN` in IEEE 754. `fabs(NaN)` in CUDA should return NaN. CPU `Abs(NaN)` — Rust's `f64::abs()` returns NaN. Behavior should be identical, but untested. |
| Inf input | `|+inf| = +inf`. L1 norm of a buffer with one inf should be inf. |
| Subnormal inputs | `fabs(subnormal) = subnormal` in IEEE 754 with FTZ off. GPU with FTZ may return 0.0. |

---

### T7 — `gpu_pearson_matches_cpu_two_input`

**Recipe**: `pearson_r()` — 5 slots: `Σ x`, `Σ y`, `Σ x²`, `Σ y²`, `Σ xy`  
**Input**: Perfect linear `y = 2.5x + 7`, N=1,024  
**N**: 1,024

**Checks**: CPU r ≈ 1.0 within 1e-10, GPU r ≈ 1.0 within 1e-10, GPU-CPU within 1e-12

**Critical gaps**:

| Gap | Why it matters |
|-----|----------------|
| **Large-mean data** | Same catastrophic cancellation as variance — Σx² - (Σx)²/n loses precision. A test with x near 1e9 (price series) would return |r| > 1 or garbage. Both CPU and GPU would agree on the garbage. |
| Near-zero correlation | `r ≈ 0.0` input — the formula produces 0/something, which is fine. But near-zero correlation is computed as the ratio of two near-zero quantities, each computed by catastrophic cancellation. |
| NaN in one column | `x = [1, NaN, 3]`, `y = [4, 5, 6]` — all 5 accumulated slots are affected. |
| Identical x column | `x = [c, c, c, ...]` — variance_x = 0, pearson denominator = 0, result = NaN or inf. |

---

### T8 — `gpu_dispatches_custom_expression`

**Recipe**: custom `Σ |ln(x)|`  
**Input**: `[1..=1000]` (all positive integers)  
**N**: 1,000

**Checks**: GPU relative error < 1e-11 vs CPU

**Critical gaps**:

| Gap | Why it matters |
|-----|----------------|
| `ln(0)` or `ln(negative)` | `log(0)` in CUDA returns `-inf`; `log(-1)` returns NaN. CPU `Expr::Ln` calls Rust's `f64::ln`. Are they identical? CUDA's `log` is `__nv_log` (inlined), which may differ from glibc by 1-2 ULPs. **This is the pitfall documented in navigator/state.md: the gpu_end_to_end.rs comment says ~2.8e-15 relative error from `__nv_log` diverging from `f64::ln`.** |
| Subnormal inputs | `ln(MIN_POSITIVE/2)` — both CUDA and Rust will return a very negative number, but precision may diverge. |
| `ln(inf)` | Should return `+inf`. |
| NaN input | `ln(NaN)` should return NaN. CUDA's `log(NaN)` is NaN; Rust's `f64::ln(NaN)` is NaN. Identical, but untested. |

**This test is the most dangerous.** It uses `log()` from CUDA's implicit libm (`__nv_log`),
which is the I1 violation we're replacing. The 1e-11 tolerance hides the 2.8e-15 divergence
between `__nv_log` and glibc's `log` — but only at the current test scale (N=1000, inputs 1..1000).
At scale or with different inputs, the divergence grows.

---

### T9 — `gpu_device_name_printed`

**Recipe**: None — only checks that `CudaBackend::new()` succeeds and returns a non-empty name.  

**Coverage**: Zero mathematical coverage. Infrastructure smoke test only.

---

## Structural gaps across all 9 tests

### 1. No oracle comparison

Every test compares GPU to CPU. None compares either to an analytical closed form (where
available) or mpmath reference. The CPU result IS the oracle — which means bugs in the
CPU path that affect all backends uniformly are invisible.

**Tests with computable closed forms** (current tests don't use them for GPU validation):

| Test | Closed form |
|------|-------------|
| sum [1..N] | N*(N+1)/2 |
| mean [1..N] | (N+1)/2 |
| mean [sin*10+5] | Σsin(0..N-1)/N * 10 + 5 ≈ 5.0 (sin average ≈ 0 for many periods) |
| pearson (y=ax+b) | r = 1.0 exactly |
| sum_of_squares [i/100] | Σ(i/100)² = (1/10000) * N*(N-1)*(2N-1)/6 |

### 2. No NaN inputs tested anywhere

Not one of the 9 tests passes a NaN to a GPU kernel. The GPU's NaN handling is entirely
untested. This matters because:

- CUDA's `atomicAdd(ptr, NaN)` has implementation-defined behavior
- PTX's fp add with NaN follows IEEE 754 on Blackwell — but only if `.contract` is disabled
- The GPU Min/Max codegen uses `fmin`/`fmax`, which propagate NaN in CUDA (unlike the CPU
  `Expr::Min`/`Expr::Max` which silently drop NaN in the current buggy tbs::eval — but which
  now behave correctly in `execute_pass_cpu` after the NaN-sticky fix)

**Asymmetry risk**: after the accumulate-layer NaN fix, CPU now propagates NaN. GPU
uses `atomicAdd` which has undefined behavior with NaN. CPU and GPU may now DISAGREE
on NaN inputs even before we switch to PTX.

### 3. No empty buffer dispatch

No test dispatches a GPU kernel with N=0. The current kernel launch uses `max(grid, 1)`,
which means at least 1 block launches even for N=0. The grid-stride loop body (`i < 0`)
never executes, accumulators stay 0.0, and `atomicAdd` writes 0.0. This is probably
correct for `sum` but wrong for `count` (should be 0) and problematic for `variance`
(gather computes 0.0/0.0 = NaN — is that tested?).

### 4. No subnormal inputs

The GPU may have FTZ (Flush-To-Zero) enabled depending on the PTX compilation flags.
With NVRTC, the `-ftz` flag defaults to `--ftz=false` for `--gpu-architecture sm_120`
— but this should be explicitly tested, not assumed.

### 5. The `Eq` epsilon inconsistency

`expr_to_cuda` at line 99 emits:
```c
(fabs((A) - (B)) < 1e-15 ? 1.0 : 0.0)
```
This matches the `tbs::eval` bug (absolute 1e-15 epsilon). Both CPU and GPU use the
same wrong epsilon. No current recipe uses `Expr::Eq`, so no test exposes this. When
a recipe does use `Expr::Eq`, the GPU and CPU will agree — both will be wrong in the
same way. The fix must land in both `tbs::eval` and `expr_to_cuda` simultaneously.

### 6. `Sign(NaN)` inconsistency between CPU and GPU

GPU codegen (`cuda.rs` line 63):
```c
((v)>0.0 ? 1.0 : ((v)<0.0 ? -1.0 : 0.0))
```
CPU (`tbs::eval`): currently broken — `Sign(NaN)` returns 0.0 (the else branch).

After the CPU fix (`if v.is_nan() { return v; }`), the CPU will return NaN for Sign(NaN).
The GPU will still return 0.0 (the ternary's final else branch fires because NaN comparisons
are false). **After the CPU tbs::eval fix lands, CPU and GPU will disagree on Sign(NaN).**
This must be fixed in `expr_to_cuda` at the same time as `tbs::eval`.

GPU fix for Sign:
```c
(isnan(v) ? (double)NAN : ((v)>0.0 ? 1.0 : ((v)<0.0 ? -1.0 : 0.0)))
```

---

## Recommended additions for campsite 4.6 (GPU test porting)

When porting these tests to the new harness, add the following cases. Each one must be
tested against a closed-form or mpmath oracle, NOT just against the CPU result:

1. **NaN-in-input for every recipe** — confirm GPU matches CPU (both should return NaN
   after the accumulate-layer fix). Explicit oracle: NaN.
2. **Empty input dispatch** — N=0. Oracle: sum=0, count=0, mean=NaN, variance=NaN.
3. **Large-mean variance** — mean=1e9, spread=1e-3. Oracle: ~8.34e-8. (Currently fails
   on CPU; should fail on GPU too. Pin red until two-pass fix.)
4. **Single element** — variance=[x] should be NaN. Oracle: NaN.
5. **Subnormal sum** — confirm GPU doesn't FTZ. Oracle: N * (MIN_POSITIVE/2).
6. **`Σ |ln(x)|` with domain edges** — `ln(0)` → -inf, `ln(negative)` → NaN. The current
   test uses `[1..=1000]` which avoids all interesting behavior.
7. **Sign(NaN) consistency** — confirm CPU and GPU agree after both fixes land.
