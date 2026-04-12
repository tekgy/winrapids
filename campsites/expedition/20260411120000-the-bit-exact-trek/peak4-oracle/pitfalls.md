# Pitfall Journal — Test Oracle / Adversarial Mathematician

*Started: 2026-04-11. Maintained by the Test Oracle role.*

Every known trap, quicksand patch, and near-miss on the Bit-Exact Trek, with
evidence and fix direction. Entries from Part VII of the trek-plan are the
seed. New discoveries are appended as the trek progresses.

---

## Seed Pitfalls — from trek-plan Part VII

These 15 pitfalls are the map the Navigator drew before the expedition began.
Each one is documented here as it was *verified* through testing, with test
evidence where available.

---

### P01 — NVRTC for "just this one case"

**The trap**: "Let me just use NVRTC for this one case — the op is hard to emit in PTX."

**Why it kills the claim**: Every op that flows through NVRTC is outside our control.
NVIDIA's optimizer may apply FMA contractions, reorder fp operations, or emit different
code on different toolchain versions. The "one case" exception becomes the rule.

**The rule**: Every op flows through the tambear PTX assembler. If an op is too hard to
emit, defer it — write a `todo!()` in the assembler and mark it as blocked on peak N.
Do not use NVRTC as a staging ground.

**Status**: No violations detected yet. PTX assembler (Peak 3) not yet started.
The guardrail is that `compile_ptx_with_opts` (NVRTC) is explicitly forbidden by I2.
Any call to it is a build-time or review-time flag.

---

### P02 — Calling `f64::sin()` in the interpreter

**The trap**: "The interpreter just needs to pass the test — I'll call `f64::sin()` and
replace it with tambear-libm later."

**Why it kills the claim**: The interpreter is the CPU reference that ALL backends diff
against. If the CPU reference uses glibc's sin and the GPU backend uses tambear-libm's
sin, they will disagree in the last 1–4 ULPs. "Replace it later" never happens cleanly;
the mismatched oracle poisons every downstream comparison.

**The rule**: The interpreter calls tambear-libm's transcendentals exclusively.
The only fp operations allowed in the interpreter's eval loop are: `fadd`, `fsub`,
`fmul`, `fdiv`, `fsqrt`. Everything else is a libm call.

**Status**: Not verified yet (interpreter not built). The `first-week-directives.md`
explicitly forbids calling `f64::exp`, `f64::sin`, `f64::ln` in the interpreter.
When the interpreter lands, audit every `f64::` call — any transcendental is a violation.

---

### P03 — Raising tolerance to match NVRTC error

**The trap**: "The NVRTC version has 1 ULP of error, let's raise the test tolerance
to 1 ULP so our new path can match it."

**Why it kills the claim**: If the tolerance was raised to match a bug in the old
(NVRTC) path, it is not a tolerance — it is a buried correctness failure. Every
tolerance widening must be root-caused against mpmath (I9), not against another
implementation.

**The rule**: Raise tolerance only when the libm function's documented ULP bound
requires it. Source: the relevant tambear-libm function's accuracy docs. Never
raise tolerance to match another implementation.

**Status**: Two NVRTC-vs-new-path comparison tests will be written when Peak 3
(PTX assembler) lands. The `ToleranceSpec` in `tambear-tam-test-harness` enforces
this: `within_ulp(bound)` requires a `bound` argument from the caller, so the
caller must consciously choose a number rather than "whatever the other impl gives."

---

### P04 — FMA: "it's more accurate"

**The trap**: "FMA computes (a*b)+c without rounding the product — it gives a more
accurate result, so let's enable it."

**Why it kills the claim**: Different backends will FMA in different places. PTX's
`.fma.rn.f64` is one operation; the CPU interpreter's `a * b + c` is two. They
produce different bits. The "more accurate" FMA answer on GPU differs from the
two-operation CPU answer by exactly one rounding. Cross-backend diff fails, and
the failure is untraceable to any bug.

**The rule (I3/I4)**: Either ALL backends emit FMA (as an explicit `tam.fma` op),
or NO backend uses FMA. Phase 1 uses no FMA. If FMA is needed later, it becomes a
new explicit IR op — never implicit contraction.

**Status**: `.contract` is explicitly documented as "must be disabled" in the PTX
assembler design. In PTX, every fp op must include `.rn` rounding mode explicitly.
The default contract mode in PTX is ON — an omission is a violation.

---

### P05 — Porting musl's sin instead of deriving from first principles

**The trap**: "musl's sin implementation is well-tested and fast. Let me just port
the coefficients from musl."

**Why it kills the claim (I8)**: Porting musl's coefficients means we inherit their
accuracy choices without understanding them. Worse, it means we inherit their bugs
without knowing which ones they are. tambear-libm's value is *proven* accuracy from
first principles — Cody-Waite reduction, Remez-optimal coefficients, oracle-verified
ULP bounds. Porting someone else's implementation is a copy of someone else's tradeoffs.

**The rule**: The math-researcher role reads the *papers*, not the *source*. The
coefficients are derived independently, then compared to reference implementations
to confirm we found bugs in them — not the other way around.

**Status**: Not yet verified (libm not built). The `first-week-directives.md`
explicitly says "Do NOT look at glibc/musl/sun-libm source code during design."

---

### P06 — Varying block count with N for performance

**The trap**: "For N=100, launching 1 block is wasteful. Let me vary the block count
as a function of N."

**Why it kills the claim**: If block count varies, the reduction tree topology
changes. Atomics give different answers depending on the order threads win the race,
and that order depends on the number of blocks. Different N → different reduction
order → different bits. The determinism guarantee (I5, addressed in Peak 6) depends
on a fixed launch configuration.

**The rule**: Deterministic launch config for any kernel where the result depends
on reduction order. Vary the block count only for embarrassingly parallel kernels
where each element's result is independent.

**Status**: Current NVRTC path is not deterministic. This is documented and blocked
by I5 (`#[ignore = "xfail_nondeterministic"]` in the harness). Peak 6 fixes this.

---

### P07 — "atomicAdd is good enough for now"

**The trap**: "The tests pass, atomicAdd introduces at most 1 ULP of error, and we
can fix it later when we care about determinism."

**Why it kills the claim (I5)**: Non-deterministic reductions mean two identical runs
on the same hardware give different bits. Cross-backend diff can still pass (both
non-deterministic backends agree on *average* but not on every run), giving false
confidence. More importantly: if we ship non-deterministic reductions and then try
to make them deterministic later, we break all existing cross-backend diffs.

**The rule**: Every reduction test that uses atomicAdd is marked
`#[ignore = "xfail_nondeterministic — remove after Peak 6"]`. The harness enforces
this: the Test Oracle (not the backend implementer) removes these annotations after
Peak 6 lands.

**Status**: Four `#[ignore]` tests exist in `tambear-tam-test-harness/src/hard_cases.rs`
with correct annotations. They are not removed by anyone other than the Test Oracle.

---

### P08 — Adding IR ops to "make my life easier"

**The trap**: "I need a conditional-move in my libm implementation. Let me just add
`tam.select(cond, a, b)` to the IR — it'll be useful for other things too."

**Why it kills the claim**: Every op added to the IR must be implemented in EVERY
backend: PTX assembler, SPIR-V translator, CPU interpreter. Adding ops casually
creates N-backend implementation debt. The IR Architect has veto specifically to
prevent IR scope creep.

**The rule**: New IR ops require escalation to Navigator. The proposal must name the
specific backends where implementation is confirmed and the libm functions that
require it. Proposals go in `peak1-tam-ir/future-ops.md` first.

**Status**: No violations detected yet. IR is still being designed.

---

### P09 — Using rspirv / cranelift as semantic authorities

**The trap**: "rspirv can generate valid SPIR-V and cranelift can optimize the code.
Let me let them control the output and just wrap the input."

**Why it kills the claim**: Using rspirv or cranelift as semantic authorities means
they control rounding modes, instruction selection, and optimization. Our invariants
require explicit rounding mode on every op, no FMA contraction, no implicit reordering.
A compiler that "optimizes" our code will violate these.

**The rule**: `rspirv` is a *typewriter* — we tell it exactly which bytes to write
and it writes them. Same with cranelift. They translate our explicit IR into machine
code, not the other way around. If a library wants to apply optimizations, that
library is a semantic authority, not a typewriter, and must be wrapped or rejected.

**Status**: Not yet verified (SPIR-V/cranelift not started). The principle is clear.

---

### P10 — "The test passes on my machine"

**The trap**: "The sum_all kernel works correctly on my RTX 6000 Pro. Shipping it."

**Why it kills the claim**: The architectural claim requires all backends to agree.
One backend passing its own self-test is a proof of concept, not a proof of
architecture. The RTX 6000 Pro may give the correct answer for reasons that don't
generalize to other hardware (e.g., the GPU happens to execute atomics in a consistent
order on Blackwell, but not on Turing).

**The rule**: All backends pass, or nothing passes. The cross-backend diff
(`assert_cross_backend_agreement`) is the only valid acceptance criterion.

**Status**: The harness infrastructure (`tambear-tam-test-harness`) enforces this.
Single-backend tests are smoke tests only; the final gate is cross-backend agreement.

---

### P11 — Single-precision anywhere

**The trap**: "For this intermediate accumulator, f32 is fine — we're going to sum
into f64 at the end anyway."

**Why it kills the claim**: Any f32 intermediate introduces up to 16M ULPs of error
compared to a fully-f64 path. The cross-backend diff will flag it, but only if the
other backend is f64. If both backends use f32 for the same intermediate, both are
wrong and the diff passes.

**The rule**: Every kernel declares fp64 explicitly. Every accumulator is f64.
The only f32 in the system is in the final output conversion, where it is explicit
and documented.

**Status**: Current recipes all use f64. No f32 accumulators. The PTX assembler
must enforce `.f64` on every op.

---

### P12 — One-pass variance instability

**The trap**: The formula `(Σx² - (Σx)²/n) / (n-1)` is fast, easy to express as
a single-pass accumulate, and "usually works."

**Why it kills the claim**: When mean >> std (every financial price series), the
formula subtracts two nearly-equal large numbers. The result has fewer significant
bits than the true variance — or goes negative (physically impossible).

**CONFIRMED BUG with failing tests:**

- Test `variance_catastrophic_cancellation_exposed` — data: 1000 values near 1e9
  with spread 1e-3. True variance ≈ 8.34e-8. Computed: **-4592** (negative!)
- Test `variance_welford_vs_onepass_stress` — 10000 values alternating 1e8+1 and
  1e8-1. True variance = 1.0. Computed: **0.0** (total precision destruction)

**Fix direction**: Two-pass variance. Pass 1 computes mean exactly. Pass 2 computes
Σ(x-μ)² with the mean as a reference. Both passes expressible in accumulate+gather.
Welford's sequential algorithm is NOT needed — the two-pass approach is both
parallelizable and stable.

**Affected recipes**: `variance`, `variance_biased`, `std_dev`, `covariance`, `pearson_r`.

See `pitfalls/variance-catastrophic-cancellation.md` for full root-cause analysis.

---

### P13 — Assuming `rint` rounds to nearest-even

**The trap**: "I need an integer conversion in the libm range-reduction step.
`rint(x)` will work."

**Why it kills the claim**: `rint` uses the current rounding mode. The default
IEEE rounding mode is round-to-nearest-even, but nothing in our kernel establishes
this. On some hardware configurations (particularly when coming from an FP-intensive
kernel), the rounding mode may be left in a non-default state.

**The rule**: Be explicit about rounding mode on every conversion and every fp op.
In PTX: every fp instruction must carry `.rn` (round to nearest-even). In the CPU
interpreter: use explicit round-to-nearest-even operations, not `rint`.

**Status**: Not yet verified. PTX assembler must enforce `.rn` on every fp op.
The libm implementer must not use `rint` in algorithm design without explicitly
confirming which rounding mode is in effect.

---

### P14 — Shared memory bank conflicts

**The trap**: "My reduction kernel uses shared memory and some threads access the
same bank — I'll fix the layout now to avoid bank conflicts."

**Why it kills the claim**: Bank conflicts are a performance issue, not a correctness
issue. Fixing them in Phase 1 means optimizing before correctness is established,
which distracts from the invariant work and may introduce subtle bugs (padding changes
the memory layout, which changes which thread writes which element).

**The rule**: Don't optimize for shared memory bank conflicts in Phase 1. Get
correctness first. The performance work is Phase 3 or later. A comment noting
the known bank conflict is acceptable and encouraged.

**Status**: Not yet relevant (GPU kernels being replaced, not optimized).

---

### P15 — Forgetting the `.param` → register load at kernel entry in PTX

**The trap**: "The PTX kernel compiled and the output was all zeros. Must be a bug
in the math."

**Why it kills the claim (silent zeros failure mode)**: PTX kernel parameters arrive
in `.param` space. They must be explicitly loaded into registers with `ld.param.u64`
(or the appropriate type) before any computation. If the load is missing, the
register holds its initialization value (zero for most types). The kernel "runs"
correctly from the PTX assembler's perspective, produces all zeros, and every test
that expects zero-valued outputs passes. Only tests with non-zero expected values
catch it — and those tests may look like a logic bug, not a missing load.

**The rule**: Every PTX kernel must begin with explicit `.param` loads. The PTX
assembler (Peak 3) must emit these as part of the kernel preamble, before any
user-defined ops. Verify by checking the hello-world kernel output (should be 42.0,
not 0.0).

**Status**: Not yet verified (PTX assembler not built). This is the primary test
for campsite 3.5 (dispatch hello-world, confirm `out[0] == 42.0`).

---

## New Pitfalls — discovered during expedition

These were not in the trek-plan's initial list. They were found during adversarial
testing in the baseline session (2026-04-11).

---

### P16 — NaN silently ignored by Min/Max in tbs::eval

**The trap**: `Min(NaN, 5.0)` returns `5.0` instead of `NaN`.

**Root cause**: IEEE 754 NaN comparison semantics. The expression `va <= vb` is
`false` when either operand is NaN, so the `else { vb }` branch fires, silently
dropping the NaN and returning the non-NaN operand.

**Why this is dangerous**: A NaN in input data should propagate through every
downstream result. If Min/Max silently drop NaN, a single corrupt data point
produces plausible-looking output (the second operand) rather than a visible error.
This is the worst class of bug: no crash, wrong answer.

**CONFIRMED FAILING TESTS (now fixed):**
- `tbs_min_nan_x_is_nan` — `Min(NaN, 5.0)` returned `5.0`, now returns `NaN`
- `tbs_max_nan_x_is_nan` — `Max(NaN, 5.0)` returned `5.0`, now returns `NaN`

**Fix applied**: NaN guard before comparisons — both sides checked:
```rust
Expr::Min(a, b) => {
    let va = eval(a, ...);
    let vb = eval(b, ...);
    if va.is_nan() || vb.is_nan() { f64::NAN } else if va <= vb { va } else { vb }
}
```

**Status**: FIXED. Tests `tbs_min_nan_x_is_nan`, `tbs_min_x_nan_is_nan`, `tbs_max_nan_x_is_nan`,
`tbs_max_x_nan_is_nan` all pass.

**Location**: `crates/tambear-primitives/src/tbs/mod.rs` lines 247–256.

See `pitfalls/nan-silent-ignored-by-min-max.md` for full analysis.

---

### P17 — Sign(NaN) silently returns 0 instead of NaN

**The trap**: `Sign(NaN)` in `tbs::eval` returns `0.0` rather than propagating NaN.

**Root cause**: The sign implementation tests `v > 0.0` and `v < 0.0`, both of which
are `false` for NaN, so the `else { 0.0 }` branch fires.

**Why this is dangerous**: `Sign` is used to implement direction-dependent logic.
If NaN silently becomes "zero" (the neutral sign), downstream multiplications by the
sign value produce `0 * something = 0.0` — a plausible number that hides the
data corruption.

**CONFIRMED FAILING TEST (now fixed):**
- `sign_of_nan_is_zero_not_nan` — `Sign(NaN)` returned `0.0`, now propagates NaN

**Fix applied**: `if v.is_nan() { v } else if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }`

**Status**: FIXED. Oracle verified: tests in `adversarial_tbs_expr.rs` confirm NaN propagation.

**Location**: `crates/tambear-primitives/src/tbs/mod.rs` line 226.

See `pitfalls/tbs-eval-nan-and-epsilon-bugs.md` for full analysis.

---

### P18 — Eq uses hardcoded 1e-15 epsilon, hiding 1-ULP differences

**The trap**: `Eq(1.0, nextafter(1.0))` returns `1.0` (equal) because the epsilon
`1e-15` is larger than 1 ULP near 1.0.

**Root cause**: The `Eq` expression in `tbs::eval` uses an absolute epsilon of `1e-15`
rather than bit-exact comparison. 1 ULP near 1.0 is approximately `2.2e-16` — smaller
than the epsilon, so values that differ by 1 ULP compare as equal.

**Why this is dangerous**: `Eq` is used in conditional logic (recipes can branch on
equality). If `Eq` silently collapses 1-ULP differences into equality, then
cross-backend comparison using `Eq` will pass even when backends disagree by 1 ULP.
The tolerance policy is buried inside the expression evaluator, invisible to the caller.

**CONFIRMED FAILING TEST:**
- `eq_with_values_differing_at_last_bit_is_one` — two values 1 ULP apart compare as equal

**Fix (Oracle verdict)**: Use `va == vb` (IEEE 754 equality), NOT `a.to_bits() == b.to_bits()`.
The `to_bits()` approach treats `0.0` and `-0.0` as unequal (different bit patterns), which
is mathematically wrong — they represent the same real number. IEEE 754 equality (`==`)
gives the correct semantics: `0.0 == -0.0` is `true`, `NaN == anything` is `false`.
NaN must be guarded separately: `if va.is_nan() || vb.is_nan() { NaN } else if va == vb { 1.0 } else { 0.0 }`.
If "close enough" equality is needed, it should be a separate `ApproxEq(a, b, tol)` expression.

**Status**: FIXED. oracle verification tests added to `tbs::tests`:
- `eq_exact_for_nearby_floats` — confirms 1-ULP values compare as unequal
- `eq_nan_propagates` — confirms NaN propagates through Eq
- `gt_nan_propagates`, `lt_nan_propagates` — same fix applied to Gt/Lt comparisons

**Also fixed**: `Gt` and `Lt` had the same NaN non-propagation bug (not in adversarial's
original P18 but same root cause). All three comparison operators now propagate NaN.

**Location**: `crates/tambear-primitives/src/tbs/mod.rs` lines 274–288.

See `pitfalls/tbs-eval-nan-and-epsilon-bugs.md` for full analysis.

---

### P19 — GPU oracle gap: CPU-GPU agreement ≠ correctness

**The trap**: "The 9 GPU end-to-end tests pass (CPU matches GPU). The code is correct."

**Why this is dangerous**: CPU-GPU agreement only proves that two implementations agree
*with each other*. It does not prove either is correct. If the NVRTC-compiled GPU kernel
and the CPU recipe both compute the one-pass variance formula, they will agree — and both
will be wrong on catastrophic cancellation inputs.

**Evidence**: The adversarial suite (`adversarial_baseline.rs`) passes the same data
through the CPU recipe path and finds 2 failures. The GPU tests would likely show
CPU-GPU agreement on the same data — two wrong answers agreeing.

**The fix**: Every backend's test suite must include comparison against mpmath (I9) —
not just cross-backend agreement. The Test Oracle's role is to provide mpmath references
for every recipe. No test that relies solely on self-consistency passes muster.

See `pitfalls/gpu-tests-missing-adversarial-coverage.md` for full analysis.

---

### P20 — NaN argument-order dependence in comparison-based accumulators

**The trap**: `Min(x, NaN)` propagates NaN (correct behavior due to IEEE `>` returning
false when NaN is second arg). `Min(NaN, x)` drops NaN (bug — `<=` returns false when
NaN is first arg). Behavior depends on which argument is NaN.

**Root cause**: The accumulate pass in `execute_pass_cpu` had this asymmetry before it
was fixed. The `tbs::eval` Min/Max still has it (see P16).

**Why this is dangerous**: If NaN propagation depends on argument order, then whether
data corruption is detected depends on when in the input stream the corrupt value appears.
An early NaN (in the first position) may be silently dropped. A late NaN (in the second
position) propagates. This makes bugs intermittently detectable — the worst kind.

**Current status**: `accumulates::execute_pass_cpu` has been FIXED (NaN-sticky logic
plus identity-guard for empty/all-NaN data). `tbs::eval` Min/Max is NOT yet fixed.
Tests P16 document the remaining failure.

See `pitfalls/nan-silent-ignored-by-min-max.md` for full analysis.

---

---

### P21 — Cross-backend agreement is a consistency check, not a correctness check

**The trap**: "CPU and GPU agree. The implementation is correct."

**The architectural principle**: Two backends agreeing means their implementations are
consistent with each other. It does NOT mean they are consistent with the mathematics.
Two wrong answers agreeing is worse than one failing test — it creates false confidence
that propagates through every downstream use.

**The distinction:**
- **WithinBackends agreement**: `cpu.result == gpu.result`. Checks consistency. Catches
  bugs that affect one backend but not another (wrong CUDA codegen, wrong PTX instruction,
  etc.). Does NOT catch bugs that both backends share (wrong formula, shared algorithmic
  mistake, same incorrect intermediate).
- **WithinOracle agreement**: `backend.result ≈ mpmath.result`. Checks correctness.
  Catches bugs in the formula itself, regardless of how many backends implement it the
  same way.

**Both checks are required, and they test orthogonal properties.** A test suite that only
checks cross-backend agreement will pass for every shared algorithmic bug.

**Current gap**: The 9 GPU end-to-end tests (`gpu_end_to_end.rs`) only check CPU-GPU
agreement. None compare against an mpmath oracle. The catastrophic cancellation inputs
would likely show CPU-GPU agreement (both give 0.0) while the correct answer is 1.0.

**Fix direction**: Every test that checks a numerical result must either:
1. Compare against a hardcoded mpmath reference (preferred for deterministic functions), or
2. Be marked with a comment explaining why cross-backend agreement is sufficient (acceptable
   only for tests that verify invariant-enforcement behavior, not numerical correctness).

**See also**: P19 (concrete diagnosis of the GPU test suite gap), `pitfalls/ulp-bounds-compose-additively.md` (naturalist's two-axis framing: WithinBackends vs WithinOracle).

**Status**: Documented. The fix requires adding mpmath references to the GPU test suite
— a separate campsite (probably 4.6 augmentation or a new 4.8).

---

### P22 — fcmp+select NaN drop: the PTX assembler trap for NaN-propagating min/max

**The trap**: Implementing `min(a, b)` in .tam IR as:
```
%cmp = fcmp_lt.f64 %a, %b        ; false if either is NaN (correct IEEE behavior)
%r   = select.f64 %cmp, %a, %b  ; if false → picks %b (NaN in %a silently dropped)
```

When `%a` is NaN: `fcmp_lt.f64(NaN, b)` → false (IEEE 754 §5.11: all ordered comparisons
with NaN return false). `select.f64(false, NaN, b)` → picks `%b`. The NaN disappears.

**What this is and is not:**
- `select.f64` is a control mux, not an arithmetic op. IEEE 754 is silent on select
  semantics — select is not an IEEE operation. `select` swallowing the unselected branch's
  NaN is NOT an I11 violation in `select` itself. I11 applies to arithmetic ops and
  comparison-based accumulators, not to mux primitives.
- The `fcmp` returning false for NaN IS correct IEEE 754 §5.11 behavior.
- The **I11 obligation falls on the caller**: if a program needs NaN-propagating min/max,
  it must NOT use bare `fcmp + select`. IEEE `fcmp` intentionally loses the NaN signal in
  service of `fmin/fmax` semantics — which are valid when you want to ignore missing data.

**Two valid semantics — the caller must choose explicitly:**
1. **IEEE fmin semantics** (`minNum`): NaN in one argument → return the other (non-NaN).
   Used by C `fmin()`, CUDA `fmin()`, PTX `min.f64`. Designed for ignoring missing data.
   Implementation: `fcmp_lt + select`. Correct for this semantic.
2. **Propagating min semantics**: NaN in either argument → NaN out.
   Used by our `tbs::eval` after the P16 fix. Required by I11 when tambear's contracts
   guarantee NaN propagation.
   Implementation: `isnan` guard before the comparison:
   ```
   %isnan_a = fcmp_unord.f64 %a, %a   ; true iff %a is NaN
   %isnan_b = fcmp_unord.f64 %b, %b   ; true iff %b is NaN
   %either_nan = or.i1 %isnan_a, %isnan_b
   %cmp = fcmp_lt.f64 %a, %b
   %min_ab = select.f64 %cmp, %a, %b
   %r = select.f64 %either_nan, NaN_CONST, %min_ab
   ```

**The PTX assembler trap (Peak 3 specific):**
When the Peak 3 scout writes the PTX lowering for min/max operations in tambear kernels,
the choice of which semantic is needed must be explicit. PTX `min.f64` implements IEEE
fmin semantics (NaN-dropping). If the kernel requires NaN-propagating min, the PTX
lowering must emit the guard sequence, not a bare `min.f64`.

This is the failure mode to watch for: the PTX assembler correctly lowers `min.f64` to
PTX `min.f64`, which is semantically right for fmin but wrong for propagating min. The
bug will not be caught by a ULP comparison — it will be caught only by a test that injects
NaN into a min operation and checks that NaN comes out. The hard_cases NaN propagation
generator exists for this reason.

**Current state:**
- `tbs::eval` Min/Max: FIXED to propagating semantics (P16). Correct.
- `.tam` IR: no canonical `fmin.f64` or `pmin.f64` op exists yet. When added, the IR spec
  must document which semantic each carries. The IR Architect (pathmaker) owns this naming.
- Peak 3 PTX lowering: not yet written. The trap fires here. See §5.5 in the IR spec for
  the note on `select` and NaN semantics (commit d36f64c covers this).

**The test that catches it:**
```rust
// In the PTX test suite when Peak 3 lands:
let inputs = Inputs::new().with_buf("a", vec![f64::NAN]).with_buf("b", vec![1.0]);
let result = backend.run(propagating_min_kernel, inputs);
assert!(result[0].is_nan(), "propagating min must return NaN when a is NaN");

let inputs2 = Inputs::new().with_buf("a", vec![1.0]).with_buf("b", vec![f64::NAN]);
let result2 = backend.run(propagating_min_kernel, inputs2);
assert!(result2[0].is_nan(), "propagating min must return NaN when b is NaN");
```

The second test (`b` is NaN) is the one bare `fcmp + select` passes incorrectly — it
returns 1.0 instead of NaN.

**Status**: Documented as a PTX-assembler trap. No escalation required (navigator ruling
2026-04-12: this is a caller obligation, not an I11 violation in select). When Peak 3
begins, the scout must not use bare PTX `min.f64` for kernels that require NaN propagation.

---

## Status summary (as of 2026-04-12)

| Pitfall | Status | Tests |
|---------|--------|-------|
| P01 NVRTC exception | Awaiting Peak 3 | None yet |
| P02 f64::sin in interpreter | Awaiting Peak 5 | None yet |
| P03 Tolerance to match NVRTC | Awaiting Peak 3 | harness enforces bound-explicit API |
| P04 FMA contraction | Awaiting Peak 3/7 | I3 invariant |
| P05 Porting musl/glibc | Awaiting Peak 2 | None yet |
| P06 Varying block count | Awaiting Peak 6 | xfail_nondeterministic marks |
| P07 atomicAdd | Awaiting Peak 6 | xfail_nondeterministic marks |
| P08 IR scope creep | Active (Peak 1) | None yet |
| P09 rspirv/cranelift authority | Awaiting Peak 7 | None yet |
| P10 Single-backend test | Infrastructure | harness enforces cross-backend |
| P11 f32 anywhere | Active | No f32 in current recipes |
| P12 One-pass variance | **CONFIRMED BUG** | 2 FAILING tests (red until pathmaker 1.4) |
| P13 rint rounding mode | Awaiting Peak 2/3 | None yet |
| P14 Bank conflicts | Not yet relevant | — |
| P15 .param load missing | Awaiting Peak 3 | campsite 3.5 checks this |
| P16 NaN in Min/Max tbs::eval | ~~CONFIRMED BUG~~ **FIXED** | tests pass (scout 2026-04-11) |
| P17 Sign(NaN) returns 0 | ~~CONFIRMED BUG~~ **FIXED** | tests pass (scout 2026-04-11) |
| P18 Eq epsilon hides ULP | ~~CONFIRMED BUG~~ **FIXED** | tests pass (scout+scientist 2026-04-11) |
| P19 GPU oracle gap | Documented | adversarial_baseline; GPU tests still lack mpmath |
| P20 NaN arg-order dependence | Partial fix | accumulate FIXED; tbs FIXED (P16 fix) |
| P21 Agreement ≠ correctness | Documented | requires mpmath references in GPU tests |
| P22 fcmp+select NaN drop | Documented — PTX trap | Not an I11 violation in select; caller obligation. Guard required in Peak 3 PTX lowering for propagating-min kernels. |
