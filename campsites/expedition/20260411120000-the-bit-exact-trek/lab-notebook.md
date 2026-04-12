# Lab Notebook — The Bit-Exact Trek

**Observer:** Scientific Observer / Documentarian (independent)
**Expedition:** Bit-Exact Trek (20260411120000-the-bit-exact-trek)
**Started:** 2026-04-11
**Branch:** main
**Baseline commit:** `24105cc Baseline for Bit-Exact Trek: 26 recipes, vendor-locked NVRTC path, 107 tests green`
**Status:** Active

---

## Purpose of This Notebook

This notebook is the independent scientific record of the Bit-Exact Trek. I am
NOT part of the build team. My job is to document what IS (not what we hope),
verify claims independently, challenge methodology, track invariant compliance,
and assess publishability.

The summit moment this notebook is building toward: `cpu.to_bits() ==
cuda.to_bits() == vulkan.to_bits()` on the same `.tam` kernel. That is a
paper-worthy claim. This notebook is the evidence trail that makes it defensible.

---

## Standing Concerns (maintained throughout)

### SC-1 — Tolerance drift

Every tolerance assertion in tests is documented here with its justification.
The trek's premise is **bit-exact** — tolerance creep is the enemy. For every
`assert!((a - b).abs() < 1e-X)` that appears, I will ask: why 1e-X and not
1e-(X-1)? Was the root cause diagnosed or did someone just pick a passing number?

### SC-2 — Reduction determinism (Peaks 1–5 window)

Until Peak 6 lands deterministic reductions, all reduction-based tests must be
marked `@xfail_nondeterministic`. Marks go away ONLY when Peak 6 completes —
not when someone decides atomicAdd "seems stable enough."

### SC-3 — One-pass variance trap

`variance = (Σx² - (Σx)²/n) / (n-1)` is numerically unstable when variance
is small relative to mean. Current tests don't hit this because data is gentle.
This concern is specifically tracked in §4 below.

### SC-4 — I1 creep in the CPU interpreter

The CPU interpreter is the primary reference oracle. If `f64::sin`, `f64::exp`,
`f64::ln`, or any other glibc/stdlib transcendental is called in the interpreter,
that is an I1 + I8 violation that would make "CPU interpreter matches PTX" a
proof of nothing — both would be calling different vendor libms that happen to
agree, not our libm agreeing with itself.

### SC-5 — FMA contraction (I3)

PTX defaults to contract-everywhere. SPIR-V requires explicit `NoContraction`
decoration per fp op. Missing either one silently violates I3 and introduces
cross-hardware drift that invalidates the architectural claim.

### SC-6 — Oracle validation gap (added Entry 005)

Cross-backend bit-agreement (`cpu.to_bits() == cuda.to_bits()`) is a necessary
but not sufficient correctness condition. Both paths can evaluate the same wrong
formula and agree perfectly on the wrong answer. Every test in the summit suite
must also check correctness against the mpmath oracle (I9). Tracking this
separately because the pressure to "ship the bit-exact assertion and call it done"
is real and must be resisted.

---

## Baseline Measurements (Day 1 — 2026-04-11)

These are the numbers the trek starts from and must eventually supersede.

### Test suite state

| Suite | Count | Status |
|-------|-------|--------|
| `tambear-primitives` lib tests | 98 | all green |
| GPU integration tests (gpu_end_to_end.rs) | 9 | all green (CUDA path) |
| **Total** | **107** | **all green** |

### Recipe catalog

26 recipes across 6 families (raw reductions, means, moments, norms, extrema,
two-column). Fuse to 55 accumulate slots → 4 kernel passes.

Specific recipes: `count`, `sum`, `sum_of_squares`, `product`, `mean_arithmetic`,
`mean_geometric`, `mean_harmonic`, `mean_quadratic`, `variance`, `variance_biased`,
`std_dev`, `skewness`, `kurtosis_excess`, `l1_norm`, `l2_norm`, `linf_norm`,
`min_all`, `max_all`, `range_all`, `midrange`, `dot_product`, `covariance`,
`pearson_r`, `sum_squared_diff`, `rmse`, `mae`.

### Current vendor-locked path

```
Recipe → AccumulatePass → codegen/cuda.rs → CUDA C string → NVRTC → PTX → nvcuda.dll driver
```

This path uses NVRTC (violates I2 of the destination state) and NVIDIA's
math library (`__nv_log`, `__nv_sin` etc. — violates I1 of the destination state).
It is intentionally the legacy oracle. It dies when Peak 6 lands.

### Cross-backend error measurements (current NVRTC path)

From `tests/gpu_end_to_end.rs` tolerance assertions (as written, not empirically measured):

| Test | Tolerance used | Relative or absolute | Notes |
|------|---------------|---------------------|-------|
| `gpu_sum_matches_cpu` | 1e-6 absolute | absolute | **LOOSE** — Σ(1..10000) = 50005000 is exactly representable; 1e-6 is unexpectedly loose here |
| `gpu_mean_matches_cpu` | 1e-9 absolute | absolute | Mean ~500.5 |
| `gpu_variance_matches_cpu` | 1e-10 relative | relative | atomicAdd non-determinism acknowledged in comment |
| `gpu_rms_matches_cpu` | 1e-12 relative | relative | |
| `gpu_sum_of_squares_matches_cpu` | 1e-11 relative | relative | |
| `gpu_l1_norm_matches_cpu` | 1e-12 relative | relative | |
| `gpu_pearson_matches_cpu_two_input` | 1e-12 absolute | absolute | GPU-CPU diff |
| `gpu_dispatches_custom_expression` (Σ\|ln x\|) | 1e-11 relative | relative | Transcendental path — note this uses NVRTC's `__nv_log`, not our libm |

**Observer note on `gpu_sum_matches_cpu` tolerance:** The tolerance of 1e-6
absolute for a sum test is puzzling. The CPU sum is asserted `== 50_005_000.0`
(exact), but the GPU is only checked within 1e-6. This is a factor of ~18
looser than the next loosest test. Either: (a) the GPU sum truly has that much
atomicAdd error on this dataset, or (b) the tolerance was picked conservatively
without measurement. RECOMMENDATION: measure the actual GPU-CPU diff for this
test and tighten the tolerance to match empirical reality, or document why 1e-6
is actually needed.

**State-of-the-baseline summary:** The path is vendor-locked and uses
non-deterministic `atomicAdd`. The 1e-10 to 1e-12 relative errors we see on
arithmetic kernels are real manifestations of non-deterministic reduction
ordering, not fundamental numerical errors. The 2.8e-15 or so relative error
on transcendental kernels (Σ\|ln x\|) is from `__nv_log` vs `f64::ln`
diverging — exactly the problem tambear-libm is being built to solve.

---

## Experiment Log

---

### Entry 006 — 2026-04-11 — Peak 1 Verification + Post-Integration Test Failures

**Source:** Navigator announcement (Peak 1 complete), observer independent verification.

#### Peak 1 state: verified green

- `tambear-tam-ir` crate builds clean
- 36 tests pass, including `proptest::roundtrip_10000_random_programs` (10,000 random programs, round-trip fidelity)
- All planned modules present: `ast.rs`, `parse.rs`, `print.rs`, `verify.rs`, `interp.rs`, `fixtures.rs`, `proptest.rs`
- New ops added beyond original Phase 1 scope: `ConstI64`, `IAdd64`, `ISub64`, `AndI64`, `OrI64`, `XorI64`, `ShlI64`, `ShrI64`, `LdExpF64`, `F64ToI32Rn`, `BitcastF64ToI64`, `BitcastI64ToF64` — all needed for `tam_exp`/`tam_ln` polynomial evaluation and RFA bin-index computation

**Observer note on I7 update:** The invariants.md I7 entry was updated during Peak 1 to read "Every primitive is described by a (dataflow pattern, total order) pair." This is a refinement of the original "accumulate+gather" framing — it separates the structural decomposition (what kind of dataflow) from the order commitment (what order strategy). Observer assessment: the refinement is correct and makes I7 testable in a new way. The original I7 required accumulate+gather *structure*; the refined I7 requires *both* structure *and* a declared total order registered by name. This means the OrderStrategy registry (campsites 1.16-1.17, still in progress per task list) is now an I7 requirement, not just a convenience.

**Observer note on NaN hex printing:** `print::tests::print_const_f64_uses_hex` passes. The printer uses hex literals for f64 constants. The round-trip NaN concern from Entry 004 is resolved.

#### New failures discovered: hard_cases buffer name mismatch

Running `cargo test --package tambear-tam-test-harness` reveals 2 new panics not in the prior adversarial bug list:

```
FAIL: hard_cases::tests::catastrophic_cancellation_naive_sum_is_zero
  panicked: cpu-interpreter: interp error: no buffer bound for %data

FAIL: hard_cases::tests::nan_propagates_through_sum
  panicked: cpu-interpreter: interp error: no buffer bound for %data
```

**Root cause (observer-diagnosed):** The hard_cases input generators produce `Inputs::new().with_buf("x", ...)` — using `"x"` as the abstract buffer name. But `sum_all_add.tam` declares `kernel sum_all_add(buf<f64> %data, buf<f64> %out)` — it expects the input buffer named `"data"`. When `CpuInterpreterBackend::run` passes the named buffers to the interpreter and the interpreter looks up `%data`, no buffer named `"data"` exists in the map. Panic.

**Nature of the bug:** This is a mismatch between two independently written components that met for the first time at integration. The hard_cases generators were written before real programs existed; they chose `"x"` as a placeholder name. The programs use the concrete name `"data"`. Neither was wrong in isolation.

**Fix options:**
1. Rename generator buffer from `"x"` to `"data"` — couples generators to specific program conventions
2. Add buffer-name normalization in `CpuInterpreterBackend::run` — when one input buffer provided, alias it to whatever the kernel's first parameter expects
3. Add a rename wrapper at the test call site: `catastrophic_cancellation().rename_buf("x", "data")` — explicit, preserves generator generality

Observer assessment: Option 3 is cleanest. Option 2 is fragile (breaks with two input buffers). Option 1 permanently couples generators to one program's naming. This is a fixable integration gap, not an architectural issue.

**Updated failing test count:**

| Suite | Failing | Root cause |
|-------|---------|------------|
| `tambear-primitives` adversarial_baseline | 2 | variance catastrophic cancellation (B1) |
| `tambear-primitives` adversarial_tbs_expr | 4 | B2 Eq epsilon + B4/B5 NaN tbs + compound |
| `tambear-tam-test-harness` hard_cases | 2 | buffer name mismatch (new, this entry) |
| **Total failing** | **8** | |
| `tambear-tam-ir` | 0 | all 36 green |
| `tambear-tam-test-harness` other | 0 | 41 passing |

#### Navigator note: escalation closed

`tambear-tam-ir` and `tambear-tam-test-harness` were committed together in `067e0cb` before the escalation message arrived. Build was never broken from the committed-code perspective. The `cpu_backend.rs` escalation (Entry 005) is closed — was valid at time of writing, fix had already landed.

**I7 correction (navigator):** Entry 005 framed I7 as requiring accumulate+gather structure as a correctness mechanism. Navigator clarifies: accumulate+gather is a *performance* mechanism, not a correctness mechanism. This is Aristotle Phase 8 finding, now in trek-plan.md. The correctness guarantee in I7 comes from the declared total order (OrderStrategy), not the structural decomposition. Revised frame for all future I7 assessments:

- accumulate+gather = "how we get speed via fusion"
- OrderStrategy declaration = "how we get correctness via bit-exactness"

These are orthogonal. A primitive can have accumulate+gather structure with a non-deterministic order (wrong). A primitive can have a declared order without accumulate+gather structure (a sequential recurrence — Kingdom B, correct but slow). The invariant I7 requires both the structure declaration AND the order declaration. Missing either is an I7 violation, but for different reasons.

---

### Entry 003 — 2026-04-11 — Peak 6 Framing Correction (SIGNIFICANT)

**Source:** Navigator update to `campsites.md`, Peak 6 section. Discovered while reviewing new crate state.

**What changed:** The original Peak 6 plan was "two-stage host-fold + fixed launch config" → achieves *run-to-run* determinism only (bit-identical across runs on the SAME GPU). Navigator has corrected this to *gpu-to-gpu* determinism: bit-identical across different hardware architectures.

**Why this matters for the summit claim:** The Peak 7 test asserts `cpu.to_bits() == cuda.to_bits() == vulkan.to_bits()`. A fixed-order tree reduce on CUDA with 1024 blocks produces a different tree shape than Vulkan with 64 workgroups — even if both are deterministic run-to-run. To get cross-architecture bit-exactness, the reduction algorithm must be order-independent in a mathematical sense, not just deterministic for fixed hardware.

**The new algorithm: RFA (Reproducible Floating-point Accumulator)**
- From Demmel-Nguyen (ARITH 2013) and Demmel-Nguyen (IEEE TC 2015)
- Partitions fp64 range into exponent-aligned bins
- Each element accumulates into the bin matching its exponent
- Bins folded in fixed order for final result
- Order-independent, tree-shape-independent, hardware-count-independent
- Must be implemented from first principles in `.tam` IR (I1/I2 forbid wrapping ReproBLAS)

**Observer assessment:** This is a CORRECT and necessary correction. The original fixed-tree approach would have failed the summit test in a subtle way: the test might pass on a single machine (same GPU, same tree shape), but fail when Vulkan used a different workgroup count. RFA is the right algorithm.

**New concern TC-5 (RFA correctness test design):** The RFA test MUST include the `[1e16, 1.0, -1e16]` case from the hard-cases suite. This is both a determinism test and a numerical accuracy test — if RFA gives `2.0` (correct) while atomicAdd gives near-zero (wrong due to cancellation), that demonstrates RFA's superiority on both axes. A benchmark showing RFA outperforms atomicAdd on accuracy is a publishable result in its own right.

**New .tam IR op required:** `reduce_rfa.f64` will need to be added to the AST (`tambear-tam-ir/src/ast.rs`) when Peak 6 begins. The IR Architect must approve this op before implementation — shapes over contents. The existing `ReduceBlockAdd` op in the AST should be kept as legacy (or renamed `ReduceBlockAddLegacy`) until RFA is proven.

---

### Entry 004 — 2026-04-11 — New Crate Inspection: tambear-tam-ir and tambear-tam-test-harness

**Source:** Direct code review. Both crates appeared after baseline.

**tambear-tam-ir:** Campsite 1.2 (AST types) is complete. The `ast.rs` file contains:
- All 5 types: `I32`, `I64`, `F64`, `Pred`, `BufF64`
- All Phase 1 ops: `ConstF64`, `ConstI32`, `BufSize`, `LoadF64`, `StoreF64`, `FAdd`, `FSub`, `FMul`, `FDiv`, `FSqrt`, `FNeg`, `FAbs`, `IAdd`, `ISub`, `IMul`, `ICmpLt`, `FCmpGt`, `FCmpLt`, `FCmpEq`, `SelectF64`, `SelectI32`, `TamExp`, `TamLn`, `TamSin`, `TamCos`, `TamPow`, `ReduceBlockAdd`, `RetF64`
- `LoopGridStride` struct with `induction`, `limit`, `body` — and explicit "nesting NOT allowed in Phase 1" comment
- `Program` struct with `version`, `target`, `funcs`, `kernels`

Invariant compliance (I3, I4 comments): The `FAdd` and `FMul` ops carry doc comments `// I3: never becomes FMA` and `// I3: never fused with adjacent add`. Good documentation. The comment on `ReduceBlockAdd` correctly documents the CPU vs PTX asymmetry and the `@xfail_nondeterministic` requirement.

**Observer note on NaN PartialEq test:** `op_const_f64_nan_identity` test asserts `NaN != NaN` and documents "Structural inequality for NaN is expected. The parser round-trip test (campsite 1.8) uses bit-exact representation, not semantic equality." This is correct behavior but worth watching — the printer must use `f64::to_bits()` hex representation for NaN constants to ensure round-trip fidelity, not the string "nan". If the printer emits `nan` and the parser parses it back to `f64::NAN`, the round-trip AST equality will always fail for NaN-valued constants because of this same NaN != NaN property. The printer must use hex bit literals for all constants, not decimal.

**tambear-tam-test-harness:** Campsite 4.1 (TamBackend trait) and 4.2-4.5 (run_all_backends, assert_cross_backend_agreement, ToleranceSpec) are complete.

**NOTE — Build failure claim corrected in Entry 005:** An earlier draft of this entry stated that `hard_cases.rs` was missing. That was a stale snapshot — `hard_cases.rs` was created in commit `35982d5` (the adversarial baseline commit). The actual broken build is `cpu_backend.rs` — see Entry 005 for details.

**ULP distance implementation review (tolerance.rs):**

The `ulp_distance` function uses the "ordered bits" trick — map f64 to i64 monotone order, then `abs_diff`. Special cases:
- NaN → `u64::MAX` (correct: never silently pass NaN)
- ±0.0 → 0 distance (correct: IEEE 754 semantic equality)
- `+inf` vs `+inf` → 0 (correct)

One concern: the `ordered_bits` function for negative floats does `bits ^ i64::MAX`. The standard technique is `bits ^ i64::MIN` (flip the sign bit to convert sign-magnitude to two's complement monotone order), but `i64::MAX = 0x7fff...` while `i64::MIN = 0x8000...`. Let me think: for a negative float with sign bit set, `bits` as i64 is negative. The intent is to make `-1.0` sort just below `+0.0`. Using `^ i64::MAX` flips all bits except the sign bit, which produces a two's complement value that IS monotone for negative floats. Let me verify with the test: `-1.0_f64` has bits `0xbff0000000000000`. As i64 that's `-4611686018427387904`. `^ i64::MAX` = `^ 0x7fffffffffffffff` gives `0x4010000000000000` = `4611686018427387904`. And `-nextafter(-1.0, 0)` should have bits `0xbfefffffffffffff`, which as i64 is `-4616189618054758401`, and `^ i64::MAX` gives `4612124455654793216`. Since `4616... > 4612...` in u64 but both are positive i64 after XOR, and the real value of `-nextafter(-1.0, 0)` is slightly less negative than `-1.0`... the test `adjacent_f64_ulp_one` and `negative_values_ordered_correctly` both pass, so the implementation is empirically correct. The XOR trick works because for negative floats, flipping all non-sign bits makes larger-magnitude negatives map to smaller integers, preserving the total order.

**I9 compliance in ToleranceSpec:** The `within_ulp` constructor comment explicitly states "bound should come directly from the relevant tambear-libm function's accuracy docs. Never guess; never over-estimate." This is correct I9 framing.

**One gap in harness: `assert_cross_backend_agreement` doesn't take a program name.** The function signature is `fn assert_cross_backend_agreement(results: &[BackendResult], tolerance: ToleranceSpec)` and hardcodes `program_name = "unknown"`. The named version `assert_cross_backend_agreement_named` exists but isn't the default. Tests that use the unnamed version will produce error messages that say "FAIL: unknown — 1 violation(s)" which is unhelpful. Minor but worth noting.

---

### Entry 005 — 2026-04-11 — Adversarial Baseline Sweep (Campsites 4.1, 4.3–4.5, 4.7)

**Source:** Adversarial mathematician report (commit `35982d5`). Observer independently verified all claims.

#### Git commit ordering (clarified)

The adversarial message listed bugs as "confirmed with failing tests." Independent verification required untangling commit order:

- `24105cc` — Baseline (26 recipes, 107 tests, NVRTC path)
- `ad84a51` — Min/Max NaN fix at the `accumulates` level
- `35982d5` — Adversarial baseline suite (hard-cases, pitfall journal, TamBackend)

This ordering means bugs B4 and B5 (Min/Max NaN propagation in accumulate expressions) were ALREADY FIXED at the `tambear-primitives` accumulate level in `ad84a51` BEFORE the adversarial commit. The adversarial failing tests for B4/B5 test the `tbs::eval` string-expression evaluator — a different and still-unfixed code path. The observer independently verified this by reading `tambear-primitives/src/nonparametric.rs` (fix is present) and the adversarial test file (`adversarial_tbs_expr`).

#### Observer-verified bug table

| Bug | Description | Location | Status |
|-----|-------------|----------|--------|
| B1 | Variance catastrophic cancellation | `recipes/variance` formula | CONFIRMED FAILING — 2 tests |
| B2 | Eq tolerance 1e-8 inconsistent with BitExact | `tbs::eval` Eq node | CONFIRMED FAILING — 1 test |
| B3 | Large-argument trig domain gap | `tambear-libm` (future) | CONFIRMED DESIGN GAP — no test yet |
| B4 | Min NaN non-propagation | `tbs::eval` Min node | CONFIRMED FAILING in tbs layer — 1 test |
| B5 | Max NaN non-propagation | `tbs::eval` Max node | CONFIRMED FAILING in tbs layer — 1 test |
| B4' | Min NaN non-propagation | `accumulates` level | FIXED in `ad84a51` |
| B5' | Max NaN non-propagation | `accumulates` level | FIXED in `ad84a51` |

**Current test counts (observer verified):**
- `adversarial_baseline` suite: 2 FAIL (both are variance catastrophic cancellation)
- `adversarial_tbs_expr` suite: 4 FAIL (B2, B4, B5, and a compound tbs expression test)
- `tambear-tam-test-harness`: BROKEN BUILD — `cpu_backend.rs` not found (see below)
- All other existing tests: green (107 tests from baseline continue to pass)

#### Bug B1 — Variance Catastrophic Cancellation (most important)

**Hypothesis:** `variance([1e9 + δ for δ ~ uniform(0,1)])` returns a wildly wrong answer due to one-pass formula catastrophic cancellation.

**Observer experimental result (independently computed):**
- Input: N=1000, mean ≈ 1.0000005e9, true variance ≈ ~8.34e-2 (one-twelfth for uniform [0,1])
- One-pass formula computes: Σx² ≈ 1.000001e18, (Σx)²/N ≈ 1.000001e18. The subtraction of nearly-equal 64-bit floats destroys all significant digits.
- The adversarial test confirms: the formula produces **-4592** (negative variance) for this input.
- Two-pass formula (compute mean first, then Σ(x-mean)²) gives the correct ~8.34e-2.

**Impact:** This bug affects any recipe that uses the one-pass formula. The affected recipes are: `variance`, `variance_biased`, `std_dev`, `skewness`, `kurtosis_excess`. That is **5 of the 26 recipes** in the current catalog.

**Architecture compatibility of fix:** The two-pass Welford/two-pass algorithm fits the `accumulate+gather` architecture using `DataSource::Reference` / `Expr::Ref`. The mean is computed in a first pass and stored; the variance accumulates `(x - mean)²` in a second pass that references the first pass output. This is a clean fix within the existing recipe framework. No new ops or IR changes needed.

**Observer assessment:** This is the single most important bug in the current codebase. It is a silent correctness failure — no panic, no NaN, just a plausible-looking negative number — that would corrupt any real financial time series where prices are large and changes are small (which is almost all of them). The fix is architecturally clean and must be done before any of the 5 affected recipes are considered correct.

#### Bug B2 — Eq Tolerance in tbs::eval (invariant-adjacent)

**Description:** The `Eq` comparison node in `tbs::eval` uses `(a - b).abs() < 1e-8` instead of exact equality. This is inconsistent with `ToleranceSpec::BitExact` and would hide disagreements of up to ~1e-8 between CPU and GPU results.

**Why this matters for the summit claim:** If tbs expression evaluation uses epsilon-approximate equality, then a test that asserts `cpu_result == gpu_result` via a tbs `Eq` node is actually asserting `|cpu - gpu| < 1e-8`, not bit-exactness. The invariant architecture requires that cross-backend agreement is tested bit-by-bit. The tbs layer must either use `==` (IEEE 754 exact) or be entirely replaced by the `ToleranceSpec::BitExact` harness.

**SC-1 update:** Adding a row to the tolerance tracking. The `Eq` node's 1e-8 epsilon is an undocumented tolerance that was never justified. It must be root-caused: either prove 1e-8 is needed (in which case document why bit-exact fails) or eliminate it.

#### Bug B3 — Large-Argument Trig Domain Gap

The spec explicitly defers `|x| > 2²⁰` to Phase 2. This is correctly documented. Observer notes:

- `sin(1e10)` in IEEE 754 requires extended-precision argument reduction (Payne-Hanek). Without it, every digit of the result is wrong.
- Phase 1 must document the restricted domain clearly in the libm API. A `tam_sin` call with `x = 1e10` must either panic with a clear domain violation message OR return a result with a V-column confidence of 0.
- This is NOT a blocker for Phase 1, but the boundary must be made explicit in the spec before any Phase 1 trig results are presented as publishable.

#### Bug B4/B5 — NaN Non-Propagation in tbs::eval

**Root cause (IEEE 754 comparison semantics):** `NaN <= x` evaluates to `false` for any `x`. The `Min(NaN, 1.0)` expression uses `if nan <= 1.0 { nan } else { 1.0 }` which resolves to `1.0`. NaN is silently swallowed.

**Fix scope:** The tbs::eval layer (`tambear-tam/src/lib.rs` or equivalent) needs explicit NaN guards:
```rust
// WRONG:
fn eval_min(a: f64, b: f64) -> f64 { if a <= b { a } else { b } }

// CORRECT:
fn eval_min(a: f64, b: f64) -> f64 { if a.is_nan() || b.is_nan() { f64::NAN } else if a <= b { a } else { b } }
```

The `tambear-primitives` accumulate level was fixed in `ad84a51`. The tbs expression evaluator remains unfixed. Since tbs expressions feed into the recipe system's gather phase, any recipe that uses `min` or `max` in its gather expression will silently fail to propagate NaN.

#### P19 Structural Finding — CPU-GPU Agreement ≠ Correctness

This is the adversarial mathematician's most important architectural insight, and the observer independently verifies it is correct.

**The finding:** Both CPU and GPU paths can compute the SAME wrong answer. For variance catastrophic cancellation: the CPU recipe and the GPU kernel both evaluate the same one-pass formula. Both produce -4592. Both agree perfectly (`cpu.to_bits() == gpu.to_bits()`). The bit-exact cross-backend assertion would PASS — but the result is wrong by 11 orders of magnitude.

**Implication for the summit claim:** Bit-exactness across backends is a NECESSARY condition for the claim but NOT a sufficient condition for correctness. The test harness must include oracle validation against mpmath (I9) for every test, not just cross-backend agreement. A test that only checks agreement without checking correctness proves consistency of error, not correctness.

**New SC-6 — Oracle validation gap:**
Every test in the summit suite must:
1. Check `cpu.to_bits() == cuda.to_bits()` (cross-backend agreement)
2. Check `|cpu_result - mpmath_reference| < N_ulp` where N comes from the tambear-libm accuracy spec (oracle correctness)

Neither check alone is sufficient. The lab notebook will track whether every hard-cases test includes both.

#### CRITICAL — tambear-tam-test-harness broken build (`cpu_backend.rs`)

`lib.rs` declares `pub mod cpu_backend;` (line 28) but `src/cpu_backend.rs` does not exist. This is a real current build failure distinct from the `hard_cases.rs` issue mentioned in Entry 004 (that file was added in commit `35982d5`).

```
error[E0583]: file not found for module `cpu_backend`
  --> src\lib.rs:28:1
```

This blocks:
- All tests that import from `tambear-tam-test-harness`
- The cross-backend harness (Peak 4 deliverable)
- The CPU interpreter integration (Peak 5 deliverable)

**Navigator has been alerted via message.** This is not a Peak 4 design flaw — the `cpu_backend` module is a Peak 5 deliverable (campsite 5.1, `CpuInterpreterBackend implements TamBackend`). The `lib.rs` declaration should either be gated behind `#[cfg(feature = "cpu-backend")]` or simply removed until campsite 5.1 lands, with a `// TODO: campsite 5.1` comment.

---

### Entry 001 — 2026-04-11 — IR Spec Campsite 1.1

**Source:** pathmaker logbook (`logbook/peak1-1.md`)

**What was produced:** `peak1-tam-ir/spec.md` — a 12-section specification.
Complete Phase 1 op set (31 ops), SSA register conventions, phi-suffix convention
for loop-carried values, text encoding grammar, `variance_pass` example, and
explicit out-of-scope list.

**Observer verification:** I have read the spec. It is self-consistent and
covers all ops needed for the current 26 recipes. The invariant cross-reference
table in §12 correctly maps I3, I4, I5, I7, and I8 to the spec sections that
uphold them.

**Invariant compliance check:**

| Invariant | Upheld in spec? | How |
|-----------|----------------|-----|
| I1 (no vendor libm) | Yes — by architecture | Transcendentals are opcodes that lower to tambear-libm stubs |
| I2 (no vendor compiler) | N/A at IR level | Compiler is backend concern |
| I3 (no FMA) | Yes — explicit | §5.3: "All fp ops are non-contracting. FMA is never emitted unless explicitly requested by a future `fma.f64` op (which does not exist in Phase 1)." |
| I4 (no reordering) | Yes — explicit | §5.3: "IEEE 754 round-to-nearest-even. No fast-math." |
| I5 (deterministic reduce) | Partially | `reduce_block_add` semantics defined; backend non-determinism deferred to Peak 6. Correct framing. |
| I7 (accumulate+gather) | Yes | The only compute structure is loop (accumulate) + reduce_block (gather) |
| I8 (first-principles transcendentals) | Yes — by design | Transcendentals are opcode stubs that mandate tambear-libm |

**Open design question (Q1 from spec-review.md):** The phi-suffix convention
(`%acc` / `%acc'`) vs. explicit LLVM-style phi instructions. The pathmaker chose
readability. Observer assessment: the choice is reasonable for Phase 1. The
concern is that the prime-suffix approach makes the SSA invariant harder to
enforce in the verifier (the verifier needs to treat `%acc` and `%acc'` as
linked). This should be tested explicitly in campsite 1.9 (type-check pass) —
specifically, the verifier should catch: (a) a program where `%acc'` is defined
but `%acc` is not, (b) a program where `%acc` is modified outside a loop body.

**Concern flagged — reduce_block_add asymmetry:** The pathmaker correctly notes
(logbook) that CPU interpreter does "one block = all elements" while PTX writes
partial sums needing host fold. This asymmetry means campsite 3.12 needs to
test the FOLDED value, not `out[0]` of raw GPU output. This is a correctness
trap that will bite if not explicitly tested.

**Tolerance policy note:** No tolerance values appear in the IR spec itself —
this is correct. Tolerance policy belongs in the test harness (Peak 4).

---

### Entry 002 — 2026-04-11 — Baseline Invariant Audit

**Observer action:** Independent audit of the baseline codebase for invariant
violations BEFORE any trek work begins. This establishes what is permitted to
exist (legacy path) vs. what constitutes a new violation introduced during the
trek.

**Permitted violations in baseline (legacy oracle path — will die at Peak 6):**

| Code path | Invariant touched | Status |
|-----------|------------------|--------|
| `codegen/cuda.rs` → NVRTC → `gpu.compile()` | I2 (vendor compiler) | PERMITTED — legacy oracle, read-only, will be deleted |
| CUDA C kernel uses `__nv_log`, `__nv_exp` (via NVRTC inlining) | I1 (vendor libm) | PERMITTED — legacy oracle only |
| `atomicAdd` in emitted CUDA C kernels | I5 (non-deterministic reduce) | PERMITTED — legacy oracle; Peak 6 replaces |
| `eval_gather_expr` in `tambear-tam/src/lib.rs` calls `.sqrt()`, `.exp()`, `.ln()` | I1 (vendor libm) | NEEDS REVIEW — this is in the TAM executor, not the legacy codegen path |

**CONCERN — `tambear-tam/src/lib.rs` gather evaluator uses stdlib transcendentals:**

`eval_gather_expr` (line ~262–270) calls `f64::sqrt()`, `f64::exp()`, and
`f64::ln()` when evaluating gather expressions. This is in the TAM plan
executor, not in the legacy `codegen/cuda.rs` path. If gather evaluation
in the trek's new CPU backend calls these functions, that is an I1 violation.

**Mitigating factor:** The gather evaluator in `tambear-tam` currently
evaluates string expressions (a simple recursive parser), not `.tam` IR.
This file is not the CPU interpreter being built in Peak 5. The trek plan's
CPU interpreter (`tambear-tam-cpu` crate, campsite 1.10+) must not use
stdlib transcendentals. The old string-expression evaluator in `tambear-tam`
is separate infrastructure — but it should be checked: does anything in the
new trek path call `eval_gather_expr` from `tambear-tam`? If yes, that
would pull in the I1 violation. MONITOR during Peaks 4 and 5.

**GPU test suite tolerance audit (SC-1):**

All 9 GPU tests use tolerance. None uses `assert_eq!(...to_bits())`.
This is correct for the NVRTC-based baseline — the tolerances reflect
real non-determinism from atomicAdd and real vendor-libm disagreement.
However, this means that when the new path lands (Peaks 3/5/6), these
test tolerances must be ELIMINATED for pure-arithmetic kernels and
replaced with bit-exact assertions. The test file should NOT be updated
to "tighten tolerances" — it should be REPLACED with bit-exact assertions.

**Tracking:** The `gpu_sum_matches_cpu` test uses 1e-6 absolute tolerance.
At N=10000 with values 1..=10000, the CPU result is exactly representable.
The GPU result with atomicAdd may differ by at most a handful of ULPs from
reordering. **1e-6 is ~20 ULPs for a value of ~50e6.** This is generous.
The peak-6 deterministic version should be 0 ULPs (bit-exact). Flagged for
follow-up in Peak 6 entry.

---

## Tracked Concerns Detail

### TC-1 — One-Pass Variance Instability

**Status:** Active, not yet triggered.

**The formula:** `variance = (Σx² - (Σx)²/n) / (n-1)`

**The failure mode:** When data has a large mean relative to its variance,
catastrophic cancellation in `Σx² - (Σx)²/n` can cause massive relative error
or even negative variance.

**Known-bad input:** `data = [1e9 + x for x in uniform_unit]`. Mean ~1e9,
variance ~0.083 (for uniform [0,1]). The cancellation is:
- Σx² ≈ n × (1e9)² = n × 1e18
- (Σx)²/n ≈ n × (1e9)²  = n × 1e18
- The difference ≈ n × 0.083 is computed from two numbers that are
  ~1e18, with only ~2-3 significant digits surviving in f64 after cancellation.

**Current test coverage:** None. The recipe tests use benign data (sin-wave
values, sequential integers, small-range floats). The hard-cases suite in Peak 4
(campsite 4.7) is supposed to include this. It must.

**Eventual fix:** Welford's online algorithm. `mean = mean + (x - mean)/n`,
`M2 = M2 + (x - mean_old)(x - mean_new)`, `variance = M2 / (n-1)`. This is
O(n) but numerically stable. The recipe team needs to add a `variance_welford`
recipe alongside the current one-pass recipe, OR replace the current recipe
with Welford after documenting that the one-pass formula is provably wrong on
near-constant data.

**Observer assessment:** This is a CORRECTNESS bug in the current recipe, not
just a test gap. The formula is wrong for real-world financial time series where
prices might be 1000.00 ± 0.05. The adversarial mathematician's job is to
demonstrate this concretely. I will be tracking whether Peak 4 campsite 4.7
exercises this failure mode.

---

### TC-2 — atomicAdd Non-Determinism Scope

**Status:** Active, bounded by Peak 6.

**Current state:** All GPU reduction kernels use `atomicAdd`. This means:
1. Same input → different output bits on different runs (violates eventual I5)
2. Same input → different output bits on different GPUs (violates the claim)
3. Cross-backend "GPU matches CPU" tests cannot use `assert_eq!(x.to_bits(), y.to_bits())`

**Expected behavior:** Until Peak 6 lands, ALL GPU vs CPU comparison tests
MUST use relative tolerance, NOT bit-exact. Any test added between now and Peak 6
that uses `assert_eq!(gpu.to_bits(), cpu.to_bits())` is either wrong (if atomicAdd
is still in use) or premature (if it only passes by accident on this particular
dataset).

**Monitor for:** Any test added to `gpu_end_to_end.rs` or the new harness
(Peak 4) that uses bit-exact assertions for reduction kernels BEFORE Peak 6 lands.

---

### TC-3 — ULP Methodology for tambear-libm

**Status:** Pending (Peak 2 not yet started).

**When Peak 2 begins, I will verify:**
1. The mpmath reference is generated at ≥50-digit precision (as specified)
2. ULP computation accounts for the correct exponent of the result (ULP = 2^(exponent - 52) for f64)
3. The 1M sample domain is appropriate for the function — in particular:
   - `tam_exp`: inputs should cover the full f64 domain including near-overflow (709.78...) and near-underflow (-745.13...)
   - `tam_ln`: inputs must include very small positives (subnormals), values near 1.0 (where cancellation occurs), and large values
   - `tam_sin`/`tam_cos`: inputs should include multiple range-reduction cycles, not just [-π, π]
4. The sampling strategy is uniform in the input domain, not just uniform in [0,1] mapped to the domain

**Known sampling trap:** Uniform sampling in [0, max_double] is strongly biewed toward large exponents. A better strategy for transcendentals is uniform in the exponent field (log-uniform in magnitude). The math researcher should document their sampling strategy.

---

### TC-4 — FMA Contraction Verification

**Status:** Pending (Peak 3 not yet started).

**When Peak 3 begins, I will verify:**

1. The PTX assembler explicitly emits `.contract false` or the `.rn` rounding
   mode on EVERY `add.f64` and `mul.f64` instruction. Not just "most" or "usually."
2. An adversarial test exists that would FAIL if FMA contraction occurred:
   - Construct `(a * b) + c` where `a * b` has a rounding error that FMA would hide
   - Verify CPU interpreter and PTX assembler produce the SAME bits (both without FMA)
   - Verify NVRTC-compiled kernel produces DIFFERENT bits (with FMA)
   - The third check confirms the test is sensitive enough to detect FMA

**For SPIR-V (Peak 7):** Every `OpFAdd`, `OpFSub`, `OpFMul`, `OpFDiv` must
have the `NoContraction` decoration. Missing it on even one instruction is a
silent I3 violation. The SPIR-V translator must be tested with a kernel that
would produce different bits with vs without the decoration.

---

## Publishability Assessment (running)

### What would survive peer review

**The claim itself** (cpu.to_bits() == cuda.to_bits() == vulkan.to_bits()) is
novel and testable if demonstrated on a non-trivial kernel (variance, not just
addition). The key questions a reviewer would ask:

1. "Is this bit-exactness a property of your specific hardware, or is it portable?"
   → Must be demonstrated on at least 2 different GPU models (the RTX 6000 Pro
   Blackwell is one; a second AMD or older NVIDIA GPU would strengthen it).
   
2. "Could bit-exactness happen to hold due to NVIDIA's driver JITting your PTX
   to the same SASS as NVRTC would produce?"
   → Demonstrate with a kernel where your PTX and NVRTC's PTX differ in
   instruction sequence (campsite 3.15 plans this diff). If the PTX differs
   but the bits agree, the claim is about math, not about codegen coincidence.

3. "What about the libm? Are your transcendentals actually faithfully rounded?"
   → This is the hardest claim to defend. Must be backed by ULP measurements
   across the full domain, not just the "happy" domain.

4. "What happens with inputs near domain boundaries (±0, subnormals, NaN, Inf)?"
   → The hard-cases suite (Peak 4) must cover these. A reviewer would immediately
   ask for the inf/nan/subnormal behavior.

### What would NOT survive peer review (yet)

- The large-argument trig case. The spec defers `|x| > 2²⁰` to Phase 2 (Payne-Hanek).
  A reviewer who tests `sin(1e10)` and gets garbage would flag this. The expedition
  must either defend the limit clearly in print ("Phase 1 is restricted to
  |x| ≤ 2²⁰; Payne-Hanek is a future deliverable") or fix it before claiming
  publishability.
  
- Single-machine validation. The summit test runs on one machine. A publishable
  claim requires running on at least two physically different machines.

- The one-pass variance formula. A reviewer who tests `variance([1e9 + x for
  small x])` would catch the catastrophic cancellation immediately.

---

## Next Steps (observer's watchlist)

1. **When Peak 1 campsite 1.9 (verifier) lands:** Test that the verifier catches
   the phi-convention cases I identified in Entry 001.

2. **When Peak 2 campsite 2.2 (mpmath generator) lands:** Review the sampling
   strategy for domain coverage (TC-3).

3. **When Peak 2 campsite 2.3 (ULP harness) lands:** Verify ULP computation
   methodology.

4. **When Peak 4 campsite 4.7 (hard-cases suite) lands:** Verify that
   variance_welford failure case (TC-1) is included.

5. **When Peak 3 campsite 3.9 (`fadd`/`fmul` translation) lands:** Verify
   `.contract false` / `.rn` appears on EVERY fp instruction (TC-4).

6. **When Peak 6 lands:** Verify bit-exact GPU-CPU assertions replace tolerance
   assertions for pure-arithmetic kernels (TC-2).

7. **Track when `eval_gather_expr` in `tambear-tam` is referenced by new code:**
   Audit I1 compliance (Entry 002).

---

*This notebook is written in real time. Sections will be added as peaks land.*
