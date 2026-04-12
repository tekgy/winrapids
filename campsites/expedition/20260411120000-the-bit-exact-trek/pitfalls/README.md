# Pitfall Journal

*Running log of near-misses, wrong turns, and things that will tempt you off the path.*

Started by scout (2026-04-11). Add entries as you find them.

---

## From the trek-plan (Part VII) — the 15 named pitfalls

### P01: Forgetting `.contract false` / `.rn` in PTX

**What**: PTX defaults to FMA contraction when `mul.f64` result is used immediately by `add.f64`.
A kernel that looks like `a*b + c` silently becomes a single FMA instruction.
FMA gives a different answer than `mul` + `add` separately (one rounding step vs two).

**Why it's dangerous**: The output looks correct for most inputs. It diverges from the CPU
interpreter only in the last 1-2 bits, which tests with `< 1e-10` relative error will miss.
But Peak 6's `to_bits() ==` assertion will catch it — by which time you've already built
Peak 3-5 on a broken foundation.

**Defense**: Every fp op in the PTX emitter includes `.rn`. The rule: if there is no `.rn`,
the emitter should not compile. Make it impossible to emit an op without a rounding mode.

---

### P02: Wrong rounding mode — bare `add.f64` vs `add.rn.f64`

**What**: Bare `add.f64` is not fully specified. `add.rn.f64` is round-to-nearest-even, which
matches IEEE 754 and matches the CPU interpreter.

**Defense**: Same as P01 — every emitted fp op carries `.rn`.

---

### P03: Parameter passing garbage

**What**: Kernel parameters in PTX go through `.param` space and must be loaded explicitly
with `ld.param.u64`. Getting the address computation wrong means the kernel silently reads
garbage. The kernel may launch successfully, produce zeros, and you'll think it works.

**How to detect**: The hello-world test (42.0) explicitly tests parameter passing.
After that: the `identity_copy` kernel (3.8) — `out[i] = in[i]` — catches address bugs.

**Failure mode**: `out[i] = 0.0` silently. The kernel launches, the dispatch returns OK,
the readback gives zeros. Not an error.

---

### P04: Block-zero thread only for reduction write-back

**What**: In `reduce_block_add`, only thread 0 of each block writes the partial sum.
If the predicate is wrong, multiple threads write → wrong answer. If the predicate is
inverted → zero writes → the output stays at 0.0.

**Defense**: Test with a single-block dispatch first. Then multi-block. The single-block
test makes the predicate easier to reason about.

---

### P05: `cuModuleLoadDataEx` error messages are terrible

**What**: PTX syntax errors report as line numbers in the PTX text. The error message is a
cryptic driver code + a line number. A missing semicolon two lines before the actual problem
produces a "register not declared" error pointing at a completely unrelated line.

**Defense**: During development only — run `ptxas --verify` on the emitted PTX text before
loading. This gives better errors. Never in production — `ptxas` is a toolkit binary, NOT
a runtime dep (I2 violation if called from code).

---

### P06: NVRTC as "just for this one op" fallback

**What**: When an op is hard to translate, the temptation is "just call NVRTC for this
kernel, it's just one." I2 is absolute. There is no "just one."

**Defense**: If an op is too hard to emit ourselves, the right answer is to DEFER the op
(don't translate kernels that use it yet), not to call NVRTC.

---

### P07: Interpreter calling `f64::exp`

**What**: When wiring libm into the interpreter, the path of least resistance is
`Expr::Exp(a) => eval(a,...).exp()` — i.e., Rust's `f64::exp` which calls glibc.
This violates I1 and I8 and makes the CPU interpreter useless as an oracle (it's just
reading glibc back to you).

**Defense**: The interpreter MUST call the tam_exp function from tambear-libm's .tam IR,
recursively. If tambear-libm isn't ready yet, `tam_exp` panics with "not yet in libm."
That panic is the correct behavior — it means Peak 2 has a blocker.

---

### P08: One-pass variance formula catastrophic cancellation

**Verified**: `data = [1e9, 1e9+1, ..., 1e9+9]` → one-pass formula returns 0.0 (exactly).
True variance = 9.1667. Relative error = 100%.

`data = [1e9 + x/1e6 for x in range(100)]` → one-pass formula returns 1.65e+02.
True variance = 8.4e-10. Relative error = 2e+11.

**Root cause**: Both `Σx²` and `(Σx)²/n` round to the same f64 value when the mean is
large relative to the spread. The difference is lost entirely.

**Current state**: The existing recipe's tests use small integers (1..5), which pass.
The GPU tests use `sin(i)*10+5` which avoids this. Production data will trigger this.

**Resolution**: Welford's online algorithm accumulates `(n, mean, M2)` with:
```
  delta  = x - mean
  mean  += delta / n
  M2    += delta * (x - mean)
  var    = M2 / (n-1)
```
This is numerically stable because it centers on a running mean, never computes x².
Welford is order-DEPENDENT but numerically stable. RFA is order-independent.
For variance: use Welford. For sum: use RFA.

**See also**: `pitfalls/variance-one-pass.md`

---

### P09: Partial-sum buffer off-by-one (Peak 6)

**What**: The partial-sum buffer for block reductions must have exactly `n_blocks` slots.
If `n_blocks` is calculated differently between the launch and the buffer allocation,
threads write past the end or into wrong slots. Silent data corruption.

**Defense**: `launch_config(n) -> (grid, block)` must be a single function. Both the
kernel dispatch and the buffer allocation must call the same function with the same n.
Test: assert `grid[0] == partial_buffer.len()`.

---

### P10: Shared memory tree reduce missing `__syncthreads()`

**What**: At each step of the butterfly tree reduce in shared memory, every thread must
wait for all threads to complete the previous step. Missing even one `bar.sync` causes
race conditions that produce wrong results on some runs but not others.

**Defense**: Every level of the tree reduce has `bar.sync 0` before and after. Test with
N much larger than the warp size (so multiple warps actually diverge). Run 100 times,
check for variation.

---

### P11: SPIR-V `NoContraction` decoration on result id

**What**: In SPIR-V, `NoContraction` decorates the *result id* of the instruction, not
the opcode. `OpDecorate %result_id NoContraction`. If you forget the decoration on any
fp op that could be contracted, the Vulkan driver is permitted to fuse it. Different
driver, different fusion = different bits.

**The tricky part**: Every `OpFAdd` and `OpFMul` result id needs its own `OpDecorate`.
A loop body that executes N add+mul ops needs N * 2 `OpDecorate` instructions. This is
verbose but non-negotiable for I3.

**Defense**: The SPIR-V emitter emits `OpDecorate %id NoContraction` immediately after
every `OpFAdd` and `OpFMul` instruction. Make it impossible in the emitter to add a
fp op without the decoration.

---

### P12: SPIR-V `std140` vs `std430` buffer layout

**What**: Declaring an SSBO with `std140` layout puts 16-byte alignment on vec3/vec4 and
adds implicit padding. For a `float64` array, this means elements are 16 bytes apart
instead of 8. The shader reads every other element.

**Detection**: Sum of `[1.0, 2.0, 3.0]` returns `1.0 + 3.0 = 4.0` instead of `6.0`.

**Defense**: Always declare `layout(std430)` or the SPIR-V equivalent:
`OpDecorate %_arr_f64 ArrayStride 8`. Test: identity copy kernel.

---

### P13: Vulkan descriptor set complexity vs CUDA "here are pointers"

**What**: CUDA passes buffer pointers as kernel arguments directly. Vulkan requires:
buffer → VkBuffer → VkDescriptorSet → VkPipelineLayout → VkPipeline → VkCommandBuffer.
Every layer has alignment, type, and binding index requirements. Getting any one wrong
produces a device lost or a validation error.

**Defense**: Use Vulkan validation layers during development. `VK_LAYER_KHRONOS_validation`
will catch descriptor set mismatches, wrong buffer usage flags, missing synchronization.
Never debug without it.

---

### P14: Polynomial coefficient precision in libm

**What**: If polynomial coefficients for `exp` or `log` are stored as anything less than
exact fp64, you lose bits in the evaluation. A coefficient that "should be" `0.166666666666666657`
stored as `0.1666666667` has ~9 bits of error and will give you a ULP bound of ~64 instead of 1.

**Defense**: Every polynomial coefficient is a named constant stored as an exact fp64 hex
literal (e.g., `0x1.5555555555555p-3`). Never derive coefficients at runtime. Never store
them as decimal with fewer than 17 significant digits.

---

### P15: Subnormal handling in exp reassembly

**What**: After polynomial evaluation, `exp(x) = 2^n * poly(r)`. The `ldexp`-equivalent
(bit-shift the exponent field) must handle the case where `n` is very negative and the
result is in the subnormal range. A naive bit-shift produces 0.0 or infinity.

**Defense**: The subnormal case: `n < -1022` means the result is subnormal. The reassembly
must use two multiplications: `poly(r) * 2^(n/2) * 2^(n/2)` to avoid intermediate underflow.
Test: `exp(-745)` must be a valid subnormal fp64 (near `5e-324`), not 0.0.

---

## New pitfalls discovered during the trek

### P16: Variance needs BOTH Welford AND RFA

**Discovered by**: Scout, 2026-04-11.

**What**: The trek-plan treats variance as a simple recipe test case. But variance has TWO
distinct numerical problems:
1. **Catastrophic cancellation** (P08 above) — one-pass formula fails when mean is large.
   Fixed by: Welford's algorithm.
2. **Cross-hardware non-determinism** — even Welford's result depends on summation order
   if the mean-update step accumulates differently across thread blocks.
   Fixed by: RFA (Reproducible Floating-point Accumulation).

These are different algorithms solving different problems. RFA alone (the Peak 6 plan)
does NOT fix catastrophic cancellation. Welford alone does NOT give cross-hardware
bit-exact results.

**Action needed**: The current variance recipe (in `recipes/mod.rs`) uses the one-pass
formula and will produce garbage for any real financial data (prices around 100-500 with
daily changes of 0.01-1%). The recipe must be replaced with either:
(a) Welford for numerically stable variance (still non-deterministic across runs), or
(b) RFA over centered data (numerically stable AND reproducible, but requires a two-pass or
    running-mean pre-pass).

The fix should happen at the recipe level, not just the reduction level. Filed for the
IR Architect and Math Researcher.

---

### P17: Vulkan fp64 subnormal behavior is undefined on this device

**Discovered by**: Scout, 2026-04-11.

**What**: `vulkaninfo` shows `shaderDenormPreserveFloat64 = false` and
`shaderDenormFlushToZeroFloat64 = false`. This means the GPU makes NO guarantee about
subnormal fp64 behavior in shaders. It may preserve them or flush to zero — behavior
is implementation-defined.

**Impact**: The Peak 7 bit-exact test (`cpu.to_bits() == cuda.to_bits() == vulkan.to_bits()`)
will fail for subnormal inputs if the Vulkan backend flushes denorms and the CPU interpreter
preserves them.

**Action needed**: The cross-backend diff harness (Peak 4) needs a `ToleranceSpec::except_subnormals()`
variant that permits disagreement for inputs that would produce subnormal results. This is
NOT a relaxation of the bit-exact claim — the claim is limited to normal fp64 inputs.
Document explicitly in the architectural claim statement.

---

### P18: `reduce_rfa.f64` is I7-compliant — but requires vector-valued accumulators

**Discovered by**: Scout, 2026-04-11.

**What**: The RFA algorithm accumulates each element into one of K bins. The accumulator
is a K-vector, not a scalar. The current `AccumulateSlot` structure has one output per slot.
RFA needs K outputs from one logical "slot."

**Impact on `.tam` IR design**: The IR Architect needs to decide how to represent
vector-valued accumulators. Options:
(a) K separate scalar slots (one per bin) — fits current structure, noisy API
(b) A single `reduce_rfa.f64` op with K as a parameter — cleaner, but new op type
(c) A new `Grouping::ExponentBin(K)` variant — most natural in the accumulate framework

Option (c) is the right structural answer: it's a new grouping type, and it naturally
expresses "group elements by their exponent-aligned bin." The `fuse_passes` machinery
already handles different groupings as separate passes.

This should be discussed with the IR Architect before 1.2 (Rust AST types), as the
`Grouping` enum needs this variant.

---

## Adversarial baseline bugs (P19–P21)

*Found by the adversarial team during the baseline sweep, 2026-04-11. P19 and P20 are FIXED (commit ad84a51). P21 is the open variance issue routed to campsite 1.4.*

### P19: NaN silently ignored by min/max (FIXED in ad84a51)

`min_all([3.0, NaN, 1.0])` returned 1.0 instead of NaN. Cause: IEEE 754 defines `NaN < x` as false, so NaN never writes to the accumulator. Fix: `if val.is_nan() || val < accs[j] { accs[j] = val; }` — makes NaN sticky.

**Affects**: `min_all`, `max_all`, `range_all`, `midrange`, `linf_norm`. Full details: `pitfalls/nan-silent-ignored-by-min-max.md`.

---

### P20: Identity value leaks on empty or all-NaN input (FIXED in ad84a51)

`min_all([])` returned `+Inf` (the Min identity). `max_all([])` returned `-Inf`. `linf_norm([NaN, NaN])` returned `-Inf` (same root cause as P19). The identity value escapes because the accumulation loop never runs. Fix: post-loop check: if no valid element was accumulated, set Min/Max result to NaN.

**Correct contract** (confirmed in fix): propagate-NaN by default. `using(na_rm: true)` for skip-NaN semantics, explicitly acknowledged by the caller.

Full details: `pitfalls/identity-value-leaked-on-empty-input.md`.

---

### P21: Variance catastrophic cancellation — adversarial confirmation (OPEN, blocks campsite 1.4)

The adversarial team confirmed P08/P16 with financial-scale data: `data = [1e9 + k*1e-6 for k in 0..1000]` — true variance ~8.34e-8, formula returned **-4592.1** (negative, 55 billion times wrong). A second test (`1e8+1` / `1e8-1` alternating, 10000 samples) returned **exactly 0.0**.

The negative result occurs because fp rounding makes `Σx²` slightly smaller than `(Σx)²/n` in certain accumulation sequences — the formula produces a negative "variance" before even being divided by `n-1`. This is an especially bad failure because the result is plausible-looking to a downstream consumer who doesn't inspect the sign.

**Fix**: Two-pass variance in campsite 1.4. First pass: `mean = sum / count`. Second pass: `Σ(x - mean)²` with mean as a loop-invariant constant. Both passes are expressible in the Phase 1 .tam IR. The pathmaker's CPU interpreter already confirmed Welford is expressible; navigator confirmed the two-pass approach as the architectural fix.

Two tests `variance_catastrophic_cancellation_exposed` and `variance_welford_vs_onepass_stress` remain pinned-red in `adversarial_baseline.rs`. They are the acceptance criteria for campsite 1.4.

Full details: `pitfalls/variance-catastrophic-cancellation.md`. Also see `pitfalls/variance-one-pass.md` (scout's earlier analysis) and `pitfalls/recipe-stability-map.md` (full stability classification of all 26 recipes).

---

### P22: Cross-backend agreement is not correctness (STRUCTURAL — ongoing)

**Discovered by**: Adversarial team, 2026-04-11. Fixed in `tbs/mod.rs` (sign, min, max NaN bugs). Structural concern remains.

**What**: The 9 GPU end-to-end tests in `tests/gpu_end_to_end.rs` compare CPU results to NVRTC-compiled GPU results. If the formula is wrong — as with variance on financial data — both backends compute the same wrong answer and the test passes. Agreement is tested; correctness is not.

**Why this is dangerous**: A recipe that catastrophically cancels and returns a negative variance passes all cross-backend consistency tests. The test suite signals "everything is fine" while the answer is 55 billion times wrong. This false confidence is more insidious than a test failure.

**The two distinct properties** (must never be conflated):
- **Consistency** (`cpu.to_bits() == gpu.to_bits()`): tests that backends agree. This is what I3/I5/I7 enforce. Cross-backend diff harness (Peak 4) tests this.
- **Correctness** (`|result - true_value| ≤ tolerance`): tests that the answer is mathematically right. Only mpmath oracle comparison (I9) tests this.

**Defense**: Every recipe needs BOTH a cross-backend consistency test AND an oracle comparison test using known ground truth. Create test data with analytically known answers (e.g., `[1, 2, 3, 4, 5]` whose variance is `2.5` exactly), assert the answer matches, then separately assert CPU == GPU.

**See also**: `pitfalls/gpu-tests-missing-adversarial-coverage.md`
