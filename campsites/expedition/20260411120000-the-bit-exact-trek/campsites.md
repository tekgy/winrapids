# The Campsite List

~80 campsites across 7 peaks. Each is "one person, focused work session, ends with a test."

Format: `<peak>.<n>` — <title>. <acceptance criterion>.

---

## Peak 1 — `.tam` IR (15 campsites)

**Owner: IR Architect (+ CPU Backend for 1.10+).**

- **1.1** Write a 2-page spec: types, ops, entry format, text encoding. No code. Review by whole team. Accept when ≥5 team members can read it and agree they understand the same thing.
- **1.2** Rust types for the AST (`enum Op`, `struct Program`, etc.). Accept when `cargo check` passes and the types compile without any functions yet.
- **1.3** Hand-write the `sum_all_add` program in the text encoding. Accept when a human reader who wasn't on the team can read it and explain what it does.
- **1.4** Hand-write `variance_pass` in the text encoding. Accept when it matches the Recipe structure from `tambear_primitives::recipes::variance`.
- **1.5** Hand-write `pearson_r_pass` (two inputs). Accept when it compiles in your head.
- **1.6** Build the text printer: AST → string. Accept when printing the hand-written variance AST gives back the exact same text.
- **1.7** Build the text parser: string → AST. Accept when round-tripping variance preserves the AST (equality).
- **1.8** Round-trip property test: generate random (well-typed) programs, print, parse, assert equal. 10,000 iterations. Accept zero divergences.
- **1.9** Type-check / verify pass. Every register is defined exactly once; every operand's type matches the op's signature; every buffer index is `i32`. Accept when it catches 5 hand-crafted broken programs.
- **1.10** Start the CPU interpreter (becomes Peak 5 in earnest). Minimal: `const`, `fadd`, `fmul`, `load`, `store`, `loop_grid_stride` (serial execution). Accept when the `sum_all_add` program runs on `[1..=10]` and returns `55.0`.
- **1.11** Extend interpreter with `fsub`, `fdiv`, `fsqrt`, `fneg`, `fabs`, integer ops, predicates, select.
- **1.12** Run `variance_pass` through the interpreter, assert the answer matches `tambear_primitives::recipes::variance` via the current CPU executor. Within exact equality, because both are pure Rust fp64.
- **1.13** Define the libm ops as stubs in the AST — `tam_exp`, `tam_ln`, etc. — that the interpreter currently dispatches by panicking ("not yet in libm"). This creates the contract that libm must fulfill.
- **1.14** `.tam` files on disk: load a `.tam` file, execute, write result. Accept when there's a `cargo run -- path/to/program.tam [inputs...]` that works.
- **1.15** Document the IR in a long-form reference: every op, its signature, its semantics, its test. This becomes the binding contract for every backend.

---

## Peak 2 — tambear-libm Phase 1 (20 campsites)

**Owner: Libm Implementer + Math Researcher.**

- **2.1** Pick the accuracy target. Recommendation: `max_ulp_error ≤ 1.0` for all Phase 1 functions, measured over 1M random samples in the function's domain. Document.
- **2.2** Build the mpmath reference generator. Python script that takes a function name and a domain, generates N samples, computes fp64 input + 50-digit reference, writes as a `.csv` or `.bin` that Rust can read. Accept when `gen_reference --function sin --n 1000000` produces a usable file.
- **2.3** Build the ULP-measurement harness in Rust: given (our function, reference file), compute `max_ulp`, `mean_ulp`, `stddev_ulp`. Used by every function's test.
- **2.4** `tam_sqrt`: single op, dispatches to `fsqrt.f64`. IEEE correctly rounded by spec. Accept when ULP test shows `max_ulp = 0`.
- **2.5** `tam_exp`: write the algorithm design doc first. Range reduction strategy, polynomial coefficients source, reassembly. Review by IR Architect + Math Researcher before any code.
- **2.6** `tam_exp` implementation in `.tam` IR (text file). Hand-written, uses only const/fadd/fmul/fsub/fdiv.
- **2.7** `tam_exp` passes through CPU interpreter. Measure ULP vs mpmath. If > 1 ULP, diagnose — don't paper over.
- **2.8** `tam_exp` subnormal test: `exp(-700)` must not flush to 0.
- **2.9** `tam_exp` overflow test: `exp(1000)` must produce `inf`, not garbage.
- **2.10** `tam_ln`: design doc. Range extraction, polynomial on [1, 2).
- **2.11** `tam_ln` implementation, test against mpmath. Accept ≤ 1 ULP.
- **2.12** `tam_ln` special values: `ln(0) = -inf`, `ln(-1) = nan`, `ln(1) = 0` exactly.
- **2.13** `tam_sin` and `tam_cos`: design doc. Cody-Waite range reduction with dual-precision π/2 constants. Polynomial on [-π/4, π/4]. Quadrant handling.
- **2.14** `tam_sin` implementation, test.
- **2.15** `tam_cos` implementation, test.
- **2.16** `tam_sin`/`tam_cos` large-argument test: `sin(1e10)` is currently unreliable everywhere; decide — do we require correct answer or do we document "|x| > 2²⁰ is undefined behavior"? Recommendation: document the limit, defer Payne-Hanek.
- **2.17** `tam_pow`: implementation as `exp(b * log(a))` + special cases. Extensive special-value test.
- **2.18** `tam_tanh`, `tam_sinh`, `tam_cosh`: from exp. Watch overflow.
- **2.19** `tam_atan`: polynomial. Then `tam_asin`, `tam_acos` via identities.
- **2.20** `tam_atan2`: full quadrant handling. This function has more special cases than anything else.

Each campsite's acceptance test: `max_ulp ≤ 1.0` on 1M samples across the declared domain. If it fails, diagnose.

---

## Peak 3 — tam→PTX assembler + raw driver load (15 campsites)

**Owner: PTX Assembler.** Blocks on Peak 1 (IR) and Peak 5.1–5.4 (CPU interpreter as reference).

- **3.1** Read the PTX ISA reference, chapters we need. Produce a 1-page "PTX subset we'll emit" document. Accept when the team can review it.
- **3.2** `PtxBuilder` Rust type — structured emission API. `builder.directive(".version 8.5")`, `.reg(".f64", "%fd0", 16)`, `.instr("add.rn.f64", ["%fd0", "%fd1", "%fd2"])`. Accept when you can emit a syntactically valid minimal kernel by calling methods.
- **3.3** Hand-write the hello-world PTX (writes 42.0 to `out[0]`). NOT through the translator — by hand, as a string literal. Verify the text is valid PTX.
- **3.4** Add `CudaBackend::compile_ptx_raw(ptx: &str, entry: &str)` — uses `cudarc::driver::CudaContext::load_module` on the raw bytes. **Must NOT call `compile_ptx_with_opts` or any NVRTC function.** Accept when hello-world loads.
- **3.5** Dispatch hello-world. Assert `out[0] == 42.0`. **First proof we have a vendor-compiler-free path.**
- **3.6** Translator: emit the kernel signature from a `.tam` kernel decl (entry name, buffer params, register file). Accept by dumping to a string and eyeballing.
- **3.7** Translator: `const.f64` → `mov.f64` with explicit bit literal (`0d<16hex>` form, using `f64::to_bits`). Test: constants round-trip through an empty kernel.
- **3.8** Translator: `load.f64` / `store.f64` including address calculation with base pointer + index * 8. Test: identity kernel copies `in[i]` to `out[i]`.
- **3.9** Translator: `fadd`, `fmul`, `fsub`, `fdiv`, `fneg`, `fabs`, `fsqrt` — each with `.rn` and explicit non-contracted form. Test: kernel computes `(a + b) * c` on a single-thread dispatch, verify.
- **3.10** Translator: `loop_grid_stride`. This is the cliff. Deliverables: computing `gi`, `stride`, loop header label, increment, comparison, branch, phi merging at block boundaries. Accept when a loop that computes `Σ_{i=0..N} x[i]` gives the right answer.
- **3.11** Translator: `reduce_block_add.f64` using `atomicAdd` (deterministic form deferred to Peak 6). Accept when the sum kernel matches CPU.
- **3.12** Translate `variance_pass` from Peak 1. Dispatch. Compare to CPU interpreter. **Must match bit-exact for this pure-arithmetic kernel.**
- **3.13** Translate all Peak-1 recipe kernels. Cross-check each against the CPU interpreter. Any divergence is a translator bug, diagnose.
- **3.14** Translator: inline `tam_exp`, `tam_ln`, etc., using the libm `.tam` source. Accept when `Σ |ln x|` kernel matches the CPU interpreter.
- **3.15** Diff our PTX against NVRTC's PTX for variance. Document the differences in a `PTX_DIFFS.md`. They will be substantial. The point is we understand what each difference does.

---

## Peak 4 — Replay harness (8 campsites)

**Owner: Test Oracle.** Runs in parallel with every other peak.

- **4.1** `trait TamBackend` with `name`, `run(&self, program: &TamProgram, inputs: &Inputs) -> Outputs`. Three impls planned: CpuInterpreter, CudaPtxRaw, CudaNvrtcLegacy.
- **4.2** `run_all_backends(&program, &inputs) -> Vec<(Backend, Outputs)>` — runs every available backend, collects results.
- **4.3** `assert_cross_backend_agreement(results, tolerance: ToleranceSpec)` — diff-checks with the given tolerance policy.
- **4.4** `ToleranceSpec::bit_exact()` for pure-arithmetic programs.
- **4.5** `ToleranceSpec::within_ulp(bound)` for transcendental programs.
- **4.6** Port every test from today's `gpu_end_to_end.rs` to use the harness.
- **4.7** Build the "hard cases" suite: 12–15 adversarial inputs per recipe (catastrophic cancellation, subnormal, nan/inf, huge N, empty, single-element, etc.).
- **4.8** Continuous run: the full harness + hard-case suite runs on every PR. Accept when running this locally takes < 2 minutes.

---

## Peak 5 — tam→native CPU formalization (6 campsites)

**Owner: CPU Backend Implementer (often same as IR Architect early).**

- **5.1** `CpuInterpreterBackend` — the formal backend wrapper around the interpreter from Peak 1. Implements `TamBackend`.
- **5.2** Wire tambear-libm: when the interpreter sees `tam_exp`, it looks up and recursively executes the `tam_exp` function defined in the loaded libm `.tam` file. **No calls to `f64::exp`, `f64::sin`, etc.** Accept when a test confirms tambear-libm is in the call chain.
- **5.3** Cross-check: interpreter+libm on CPU matches mpmath on 1M points for each transcendental. Same ULP bound as the PTX backend.
- **5.4** Record CPU interpreter as the canonical reference for cross-backend diffing.
- **5.5** (Later) cranelift JIT backend. Replaces the match-on-opcode with cranelift IR lowering. Must produce bit-identical results to the interpreter for every test.
- **5.6** Benchmark interpreter vs JIT vs PTX at N=1e6, N=1e8. Record crossover points. This is the document that tells future users when to prefer which backend.

---

## Peak 6 — Deterministic reductions (10 campsites)

**Owner: PTX Assembler (+ SPIR-V Assembler for parallel work on Vulkan).** After Peaks 1, 3.

> **FRAMING CORRECTION (2026-04-11, navigator):** The original sketch here (two-stage host-fold + fixed launch config) achieves `run_to_run` determinism — bit-identical across runs on the same GPU. But the Peak 7 summit test requires `cpu.to_bits() == cuda.to_bits() == vulkan.to_bits()`, which is `gpu_to_gpu` class determinism: bit-identical *across architectures* regardless of tree shape or workgroup count.
>
> The right algorithm is the **Reproducible Floating-point Accumulator (RFA)** from Demmel-Ahrens-Nguyen. It partitions the fp64 range into exponent-aligned bins, accumulates each element into the bin whose exponent range contains it, and folds the bins in fixed order. The result is order-independent, tree-shape-independent, and hardware-count-independent. This is NOT wrapping ReproBLAS (that would violate I1/I2) — it's implementing the algorithm from the paper in `.tam` IR from first principles.
>
> **Implementation note:** The bin-accumulate step is `accumulate(exponent_bin_grouping, identity, add)` + `gather(fixed_bin_order, add)` — I7-compliant, Kingdom A commutative monoid with vector state (rhymes with Welford).
>
> **Papers to read before 6.1** (mandatory, in order):
> 1. Demmel & Nguyen, "Fast Reproducible Floating-Point Summation," ARITH 2013
> 2. Demmel & Nguyen, "Parallel Reproducible Summation," IEEE TC 2015
> 3. Ahrens, Demmel, Nguyen, ReproBLAS tech report EECS-2016-121

- **6.1** Decision doc: explicitly state the target tier (`gpu_to_gpu`), argue why RFA is the right algorithm over fixed-tree, pick the bin count and width. Accept when the rationale is written and reviewer can confirm the tier claim.
- **6.2** Launch config policy: given N, `block` and `grid` are deterministic. Accept when there's a `fn launch_config(n: usize) -> (grid, block)` with tests. (Still needed — RFA is order-independent but we still need consistent dispatch for other ops.)
- **6.3** `reduce_rfa.f64` in `.tam` IR — semantic: "accumulate element into exponent-aligned bin vector; fold bins in fixed order for final result." Replaces `reduce_block_partial`.
- **6.4** CPU interpreter: implement RFA accumulate + fold sequentially. Accept when it matches mpmath for catastrophic-cancellation inputs (the `[1e16, 1.0, -1e16]` test from the hard-cases suite).
- **6.5** PTX translator: RFA bin-accumulate in shared memory, warp-level reduce per bin, block-zero thread writes bin vector, host folds bin vectors in fixed order.
- **6.6** Determinism test: run a kernel 100 times on the same input, assert bit-identical output every time.
- **6.7** Cross-architecture correctness: bit-exact match CPU interpreter == CUDA for variance, sum, pearson_r. This was the original 6.7; now it has teeth because the algorithm is order-independent.
- **6.8** Update the entire test suite from Peak 4 to use RFA reduce; the `@xfail_nondeterministic` marks go away.
- **6.9** Update the invariant document: I5 is now provably enforced. Also note: Peak 7's summit test now has a credible algorithm backing it.
- **6.10** Benchmark the perf cost of RFA vs atomicAdd vs fixed-tree at various N. Document. (Bin-count is fixed; cost scales with bins * threads, not N. Crossover point is the interesting result.)

---

## Peak 7 — Vulkan / SPIR-V second door (12 campsites)

**Owner: SPIR-V Assembler.** After Peaks 1, 3, 6.

- **7.1** `vulkaninfo` on this machine. Identify fp64 support per device. Accept when documented.
- **7.2** `ash` hello-world: instance → device → queue → name print. No compute yet.
- **7.3** Hand-author a SPIR-V module (bytes) that writes 42.0 to `out[0]`. Use `rspirv` builder for structured construction.
- **7.4** Create Vulkan compute pipeline from the module. Allocate a buffer. Dispatch. Read back. Assert 42.0. **First Vulkan dispatch working.**
- **7.5** `VulkanBackend` in `tam-gpu` implementing the `TamGpu` trait. Parallel to `CudaBackend`.
- **7.6** tam→SPIR-V translator: `const`, `fadd` (with `NoContraction`), `fmul`, `load` (via storage buffer), `store`. Accept when identity kernel copies `in[i]` to `out[i]`.
- **7.7** tam→SPIR-V: `loop_grid_stride` via `OpLoopMerge` + phi nodes + conditional branches. Structured control flow.
- **7.8** tam→SPIR-V: shared memory (`Workgroup` storage class) + `OpControlBarrier` + block tree reduce.
- **7.9** tam→SPIR-V: `fsqrt` via `GLSL.std.450 Sqrt`. Transcendentals via inlined tambear-libm.
- **7.10** Translate all Peak-1 recipes. Dispatch on Vulkan. Compare to CPU interpreter.
- **7.11** **Summit.** Three-backend diff test: CPU interpreter, CUDA raw, Vulkan raw — same `.tam` program, same inputs, assert `bit-exact` on variance, sum, l1_norm, mean, rms, pearson_r; assert `within-ulp-bound` on Σ|ln x|. Accept when it passes.
- **7.12** Document the cross-hardware agreement in a results file. This is the paper-worthy artifact: "tambear runs recipe X on NVIDIA Blackwell via PTX and on Intel/AMD/NVIDIA via Vulkan SPIR-V, with identical bits." First empirical proof of the architectural claim.

---

## Dependency graph

```
Peak 1 ─────────┬─→ Peak 5 (interpreter formalization)
                ├─→ Peak 3 (PTX) ────────┐
                ├─→ Peak 7 (SPIR-V) ─────┤
                └─→ Peak 2 (libm, blocks only on 1.13) 
                                        ├─→ Peak 6 (determinism)
Peak 4 ─── runs parallel to everything ─┘
```

**What can start day one in parallel:**
- Peak 1: IR Architect on 1.1 (spec draft)
- Peak 2: Libm Implementer on 2.2, 2.3 (mpmath reference + ULP harness — no IR dependency yet)
- Peak 3: PTX Assembler on 3.1, 3.2 (read spec, build printer skeleton — no IR dependency yet for the printer; translator comes later)
- Peak 4: Test Oracle on 4.1 (trait definition — only depends on knowing what a backend is conceptually)
- Adversary: hit current `tambear_primitives::recipes` with pathological inputs and start the pitfall journal

Five agents, five independent threads, day one.
