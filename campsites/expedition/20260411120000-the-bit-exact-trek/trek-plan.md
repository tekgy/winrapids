# The Trek: From Vendor-C-String Codegen to Bit-Exact Cross-Hardware Tambear

*The authoritative plan. Every campsite, every role, every pitfall, every rule.*

A full map. Seven peaks, lots of small campsites in between, the quicksands marked, the weather noted.

---

## Part I — What we're actually building, and why every step is necessary

The goal isn't "run math on GPUs." The goal is a single architectural claim, on which the whole of tambear rests:

> **One compiled `.tam` kernel, the same source of math, the same numerical answers, running on any ALU through only its driver, with no vendor compiler and no vendor math library anywhere in the path.**

Every step below exists because if you remove it, the claim collapses somewhere.

- Without **.tam IR**, every backend sees a different input and the "same source of math" part is a fiction. The IR is the invariant the claim centers on — it's what "same" *means*.
- Without **tambear-libm**, `tam.log(x)` on CUDA is `__nv_log` and on CPU is glibc's `log` and on Metal is Apple's, and they disagree in the last 2–4 ULPs. That disagreement is the entire problem. We have to supply log ourselves.
- Without a **real tam→PTX assembler** (not our current CUDA-C-via-NVRTC shortcut), we're using NVIDIA's compiler to translate NVIDIA's C dialect to PTX. The lock-in is total. We have to emit PTX ourselves, from our IR, and hand it to the driver raw.
- Without **replay tests**, we can't verify that any of this actually works — the tests we wrote today are the oracle for the *wrong* path and have to become the oracle for the *right* path.
- Without **tam→native CPU**, there's no portable reference to diff against. CPU is the slowest and the most trustworthy — it's the truth we compare GPU to.
- Without **deterministic reductions**, bit-exactness is impossible even if everything else is perfect, because `atomicAdd` means two identical runs disagree in the last bit and there's nothing we can do about it at compile time.
- Without **at least a second vendor door** (Vulkan/Metal), the cross-hardware claim is untested. One backend running its own math is a proof of concept; two backends agreeing bit-for-bit on non-trivial math is a proof of architecture.

Each step makes the next one meaningful. Skip any of them and the architectural claim is untested at that layer.

---

## Part II — The invariants (never negotiable)

See `invariants.md` for the full table and escalation protocol. Quick list:

- **I1** — No vendor math library in any path
- **I2** — No vendor source compiler in any path
- **I3** — No FMA contraction unless explicit
- **I4** — No implicit reordering of fp operations
- **I5** — No non-deterministic reductions
- **I6** — No silent fallback for missing hardware features
- **I7** — Every primitive is described by a (dataflow pattern, total order) pair
- **I8** — First-principles only for transcendentals
- **I9** — mpmath (or equivalent arbitrary-precision) is the oracle
- **I10** — Cross-backend diff is continuous, not a final audit

---

## Part II.5 — The Guarantee Spectrum: why bit-exact, for whom, and at what cost

*Added 2026-04-12 after Aristotle deconstructions of I7 and the bit-exact meta-goal. Revised same day to incorporate Phase 8 Forced Rejection findings. This section reframes the trek's architectural claim from "we picked bit-exact because it's correct" to "we picked bit-exact because a specific downstream architecture depends on it, and here's that architecture." The reframing survives scope-collapse pressure at hard cases: when someone argues "bounded-ULP is fine here, let's relax," the framing makes visible exactly what guarantee they'd be giving up.*

### The claim is compositional, not unilateral

The trek's central guarantee is not "tambear produces identical bits everywhere." It is:

> **Given .tam source, a backend that faithfully lowers .tam to its target ISA, and hardware that implements IEEE-754 for the ops in use, the output is bit-exact across all such (backend, hardware) pairs.**

Three preconditions, three sources of responsibility:

1. **IR-precision** — the `.tam` source is precise enough that there is exactly one meaning for each written kernel, with no freedom for a backend to reinterpret. *Tambear owns this*, via the IR design (declared total order via OrderStrategy, explicit rounding modes, bit-exact constants, no ambiguous ops).

2. **Faithful lowering** — each backend (CPU interpreter, tam→PTX, tam→SPIR-V, and future doors) translates `.tam` IR to its target instruction stream without reinterpretation, without silent reordering, without fallback substitutions. *Tambear owns this too*, via our own assemblers and our own tambear-libm.

3. **IEEE-754 compliance for the ops in use** — the hardware ALU implements IEEE-754 semantics for every op the kernel uses. *This is a hardware prerequisite*, documented per device, not something we control. When a device lacks a required feature (e.g., `shaderDenormPreserveFloat64` absent on the Blackwell's Vulkan stack), the kernel is either rejected at compile time or the hardware prerequisite is declared and the kernel's domain is narrowed to the ops the hardware does comply for.

**This matters because it makes the claim honest.** If we stated bit-exactness unconditionally, ESC-001 (Vulkan subnormal behavior) would read as a claim failure. Under the compositional framing, ESC-001 is a precondition-3 narrowing: the hardware doesn't comply with IEEE-754 for subnormal fp64 ops unless you enable a driver feature; we declare that feature as a prerequisite and document the narrower domain. The claim still holds for every (backend, hardware) pair where all three preconditions hold. Future hardware surprises will land in the same slot — precondition 3 is the "hardware reality" box, and ESC-001 is the template for how to handle surprises without relaxing the core claim.

**Every invariant in I1–I10 protects one of these three preconditions.** The Guarantee Ledger (at `guarantee-ledger.md`) maps each invariant to the specific precondition it protects and the exact cost of relaxing it. When someone proposes relaxing an invariant, the ledger is the mandatory checklist — they must cite which precondition they're violating and what user-facing property is being traded away. The ledger becomes institutional memory that outlasts individual reviewer judgment.

### Two axes, not one

The first and most important clarification: **cross-hardware numerical libraries have two separable design axes, and the trek targets one of them.**

| Axis | What it's about | What tambear does on it |
|---|---|---|
| **Speed** — does this workload run in 1 pass or N passes over memory? Does it fuse? Does it parallelize? | Fusion, scheduling, parallelism, grouping patterns | **Decomposition into accumulate + gather** (what I7 used to say). This is a performance mechanism. The Kingdom A recipes fuse into minimum passes; Kingdom B/C/D decompose differently. |
| **Correctness** — given the same source, does this kernel produce the *same bits* on different hardware? | Total order, rounding mode, NaN semantics, reduction shape | **Declared total order + explicit rounding + no implicit reordering** (I3, I4, I5, and the refined I7). This is the bit-exact guarantee. |

**These two axes are orthogonal and should never be conflated.** The original I7 framing ("every primitive decomposes into accumulate + gather") sounded like a correctness principle but was actually a speed principle. Decomposition enables fusion, which is a speed win. Bit-exactness across hardware comes from the *total order* axis, not the decomposition axis — a library without accumulate+gather could still be bit-exact across hardware by pinning every op's order. It would just be slow.

The trek's stated meta-goal ("one compiled `.tam` kernel, the same source of math, the same numerical answers, running on any ALU") lives on the **correctness** axis. Decomposition is load-bearing for tambear the product (it's why fusion works), but it's not load-bearing for the trek's claim. The trek proves correctness; decomposition is how the library stays fast while being correct.

### The spectrum of cross-hardware guarantees

A cross-hardware numerical library can target any of these points:

| Tier | What it guarantees | What the backend can do | What the user can do |
|---|---|---|---|
| **T0 — no cross-hardware claim** | each backend is its own numerical world | anything | replicate per-backend; nothing cross-machine |
| **T1 — loose bounded-ULP (≤ 10 ULPs)** | two backends agree within 10 ULPs on any input | use its own libm, its own reductions, its own FMA usage | use results for "close enough" decisions only |
| **T2 — tight bounded-ULP (≤ 1 ULP)** | two backends agree within 1 ULP on any input | some latitude on FMA and reduction shape | use results for most decisions, but cannot rely on equality tests |
| **T3 — bit-exact for pure arithmetic, bounded-ULP for transcendentals** | `a + b` is bit-identical cross-backend; `sin(x)` is within the libm's ULP bound | freedom in transcendental implementation, rigor in arithmetic | equality-test arithmetic results; tolerate libm-bounded variation |
| **T4 — fully bit-exact** | every declared op produces identical 64-bit patterns cross-backend | zero latitude — every op sequence is pinned | equality test any output; hash outputs; content-address them; cache across machines |

**Phase 1 of the trek targets T4 for Kingdom A accumulate+gather operations with declared OrderStrategy.** Phases 2+ may extend to Kingdom B/C/D, stochastic methods, and iterative fixed-point — those may need to live at T3 or T2 initially. The weaker tiers are **declared future work**, not alternatives-considered-and-rejected. A user who needs T1 today runs tambear with a "loose mode" that doesn't exist yet; when we build it, every T4 kernel also satisfies T2/T1/T0 automatically.

### Which users need which tier?

The spectrum matters because different users need different guarantees:

- **Research prototyping** (someone trying a method for the first time): T0 or T1 is fine. They're looking for the shape of the answer, not reproducibility.
- **Standard scientific computing** (climate models, simulations, stats): T2 is usually sufficient. Paper-worthy results don't depend on the last 3 ULPs.
- **ML training and inference**: T1–T2 for training (gradient noise dominates anyway); T2–T3 for inference (reproducibility for debugging, not for the math).
- **Financial settlement** (risk calculations, position marking, regulatory reporting): T4 required. Two runs on two machines must produce the same number to the last bit, or the settlement fails audit.
- **Physics simulation with checkpointing** (restart a long-running simulation on a different machine): T4 required for restart-from-checkpoint to land in the same state.
- **Audit-required regulated compute** (pharmaceutical trials, clinical decision support, certified control systems): T4 required plus formal verification of the tambear-libm implementations themselves.
- **Content-addressed persistent store** (TamSession in tambear itself): **T4 required**. This is the load-bearing case inside tambear's own architecture. TamSession caches intermediate results by content hash so that "have I computed this already?" is an O(1) hash lookup. If the same source produces different bits on different machines, the cache never hits across machines, and the sharing contract collapses. The entire TamSession sharing story — 15 methods sharing a distance matrix, 6 sharing an FFT — depends on bit-exact cross-hardware outputs.

**TamSession content-addressing is the tambear-specific case that bounded-ULP cannot satisfy.** Even T3 (bit-exact arithmetic, bounded-ULP transcendentals) breaks TamSession, because a kernel using `tam_log` inside a gather would produce different hashes on different backends under T3. The only tier that makes TamSession's cross-machine cache work is T4, everywhere, including transcendentals. This is why tambear-libm exists: our own transcendentals, bit-identical across every backend, so content-addressing works for every recipe including the ones that use `log`, `exp`, `sin`, `pow`.

### The cost of T4 — and why we pay it

T4 is the most expensive point on the spectrum. It requires:
- Our own transcendental library (Peak 2) — we cannot use vendor libms
- Our own PTX assembler and SPIR-V assembler (Peaks 3 and 7) — we cannot use vendor source compilers
- Deterministic reductions (Peak 6) via RFA — we cannot use atomicAdd for user-visible results
- Declared total order on every accumulate (the refined I7) — we cannot let the backend pick
- Cross-backend diffing as continuous integration (Peak 4) — we cannot defer verification to the end

The work is expensive. The payoff is the TamSession sharing contract, cross-machine cache hits, auditable reproducibility, and equality-testable outputs. These are specific downstream benefits, not vague "better software" claims.

### What this framing protects against

When Peak 3 or Peak 6 hits a hard case and someone argues "bounded-ULP would be sufficient here — let's relax," the framing makes the cost of relaxation visible. Relaxing bit-exact anywhere breaks TamSession's content-addressing for the kernels that touch the relaxed region. The question stops being "is close enough close enough?" and becomes "are we willing to give up cross-machine cache hits for these kernels?" That question has a concrete answer per kernel, and it keeps the team honest when the work gets hard.

### Deferred scope, explicitly

The following are explicitly **future work**, not in scope for the trek:
- Bounded-ULP modes (T1, T2, T3) as opt-in faster paths for users who don't need TamSession sharing
- Bit-exactness for Kingdom B/C/D (iterative fixed-point, stochastic methods) — some of these may never be achievable, and that's acceptable
- Formal verification of tambear-libm implementations (Coq, Lean, or similar) — the current trek targets mpmath-oracle ULP verification, not full formal proof
- Subnormal bit-exactness on Vulkan backends lacking `shaderDenormPreserveFloat64` (see ESC-001 for the decided scope carve-out)
- **The post-IEEE-754 future** (its own subsection below — too significant to leave as a one-liner)

Each of these is a real user need that a future phase of tambear may target. None of them invalidate the T4 Phase 1 work — a T4 kernel automatically satisfies any weaker tier a user might select.

### The post-IEEE-754 future: software fp64 as a path off IEEE-754-dependent hardware

*Named but deferred per team-lead decision 2026-04-12 — the logical extension of the "use/force our version" principle after ESC-002 / vendor-bugs.md VB-004. Not Phase 1 scope. Worth naming now so Phase 1 decisions don't close the door.*

Phase 1's compositional claim rests on three preconditions: IR-precision (P1, we own), faithful-lowering (P2, we own), and IEEE-754-compliance-for-the-ops-used (P3, hardware owns and we gate via capability matrix). The weak point is P3 — it's the only precondition we don't control, and it's the reason ESC-001 (Vulkan subnormal) and VB-001–004 exist. Every entry in `vendor-bugs.md` is, at root, "a vendor's P3 implementation doesn't match what we need, so we work around it."

The logical endgame is to remove P3 from the list of preconditions entirely by **emitting software fp64 implementations of every arithmetic op, composed from the target hardware's native integer primitives (or posit primitives, or whatever the target provides)**. Under this model:

- `.tam` IR says `fadd.rn.f64 %dst, %a, %b`
- Backend lowers to a 200-line integer-arithmetic sequence that implements IEEE-754 fp64 addition bit-for-bit: sign/exponent/mantissa extraction, alignment shifting, guard bits, round-to-nearest-even, overflow/underflow handling, denormal paths, NaN/Inf propagation
- Target hardware only needs integer ops + memory; no fp64 adder required
- The compositional claim becomes: given `.tam` source, a backend that faithfully lowers `.tam` to its target's *integer* ISA, and hardware that implements *integer* arithmetic correctly, the output is bit-exact across all such (backend, hardware) pairs
- P3 narrows from "IEEE-754 for fp64 ops" to "integer arithmetic on ≥32-bit words" — a vastly weaker requirement that essentially every compute substrate satisfies

**What this unlocks:**
- **Posit-based hardware** (Unum III, John Gustafson's posits)
- **Approximate / stochastic compute** (probabilistic results, analog fp)
- **Integer-only microcontrollers** (Cortex-M0-class embedded, IoT)
- **Custom silicon without fp64** (NPUs, domain-specific accelerators, future ASICs)
- **Quantum compute substrates** (hybrid classical-quantum workloads). ⚠ **This horizon is closer than it looks.** Microsoft announced the Majorana 1 topological qubit processor in February 2025 ([azure.microsoft.com/.../majorana-1](https://azure.microsoft.com/en-us/blog/quantum/2025/02/19/microsoft-unveils-majorana-1-the-worlds-first-quantum-processor-powered-by-topological-qubits/)). Topological qubits have dramatically longer coherence times than conventional qubits, which is the engineering barrier that's kept useful quantum compute "10–20 years away" for the last decade. If topological qubit architectures scale as hoped, hybrid classical-quantum workloads become a real 2–5 year horizon. Hybrid workloads need classical arithmetic on whatever the quantum accelerator's classical host processor is — which may not be IEEE-754 fp64, depending on the architecture. A tambear that runs on the classical half of a Majorana-class system would need software fp64 if the host doesn't provide hardware fp64 natively. **This is currently the most likely driver pulling the software-fp64 path from Phase 3 into Phase 2.**
- **GPUs that silently flush subnormals, mis-handle NaN, or contract FMA** — which is most current Vulkan hardware per ESC-001, ESC-002, VB-001, VB-004 — because we no longer depend on those ops being IEEE-compliant

**What this costs:**
- **Performance.** Hardware fp64 adders take ~1ns; software fp64 add takes ~100–1000ns (20–50 integer ops including conditional branches for denormal handling). Arithmetic kernels slow down by 100–1000×.
- **Implementation effort.** ~3–5k lines of careful integer code total across `fadd`, `fsub`, `fmul`, `fdiv`, `fsqrt`, plus test vectors for every IEEE-754 corner case. 1–2 person-weeks for a first cut, longer for full edge-case coverage.
- **Verification burden.** Every software-fp64 op must match hardware fp64 on every input — which means testing against hardware as the oracle, which is circular unless we test against mpmath (which is still the ground truth).

**The intermediate stop — Phase 1.5 or early Phase 2:**

Software fp64 implementations can exist as a **cross-check oracle alongside the hardware fp64 production path**, not as a replacement. The CPU interpreter gains a second mode: "software fp64" mode that routes every arithmetic op through the software implementation instead of `std::f64`. The two modes run on the same test inputs, and we assert bit equality. This gives us:

1. **Independent verification that hardware fp64 matches spec** — if the software and hardware paths disagree, we've found either a hardware bug (like the Intel FDIV bug) or a bug in our software implementation. Both are valuable findings.
2. **A fifth oracle in the cross-backend diff harness** (CPU-hardware, CPU-software, CUDA PTX, Vulkan SPIR-V, mpmath). Five sources of truth strictly stronger than four.
3. **Proof of understanding** — the team has to really know IEEE-754 to write a bit-correct software adder. That knowledge is durable and transfers to every future numerical work.
4. **A staged path to Phase 2/3** — the software implementations start as an oracle and graduate to a production backend once hardware fp64 dependence becomes a limiting factor.

This Phase 1.5 scope is optional and additive. It doesn't change the current trek. It doesn't block any peak. It's a direction to walk *after* Peak 7 summits and the Phase 1 architectural claim is proven.

**Phase 2 or Phase 3 — when software fp64 becomes the production path:**

The triggering conditions for a full software-fp64 production path are:
- A user emerges whose target hardware doesn't implement fp64 at all, or implements it non-IEEE
- A vendor gap in `vendor-bugs.md` becomes too expensive to work around with composed primitives (none so far, but hypothetically)
- A regulatory or audit requirement demands "zero trust on vendor ALU behavior" — every bit provably derived from code we wrote
- A research direction demands running on a novel substrate where fp64 doesn't exist
- **A hybrid classical-quantum workload** emerges on a Majorana-class topological quantum processor whose classical host doesn't provide native hardware fp64

**Of these, the quantum path is currently the nearest-term**: Microsoft's Majorana 1 (Feb 2025) is the first topological qubit processor demonstrated, and if the architecture scales, hybrid workloads could be a 2–5 year horizon. Phase 1 proceeds with hardware fp64 and composed-primitive workarounds for Bucket B ops. Phase 1.5 adds the software-fp64 oracle if and when the team has bandwidth. Phase 2/3 promotes software fp64 to production if and when the triggering conditions appear — and the quantum trigger might come first.

**Why we're naming this now instead of discovering it later:**

The "use/force our version" principle from team-lead's ESC-002 ruling has a logical endpoint: *never trust a vendor op for anything*. Phase 1 stops short of this endpoint because the elementary IEEE-754 ops (Bucket A) are spec-pinned and using them IS using our version. But that stopping point is a choice, not an irreducible limit. If future Phase-1-unsatisfied users need a more aggressive "our version" guarantee — one that doesn't depend on the vendor implementing IEEE-754 correctly — the software-fp64 path gets them there. **Naming it now keeps the door open and prevents Phase 1 decisions from accidentally closing it** (e.g., by hard-coding hardware fp64 assumptions into the IR semantics or the backend trait).

**Concretely, Phase 1 should avoid:**
- IR ops whose semantics only make sense on IEEE-754 hardware (e.g., assuming a specific NaN encoding in the bit layout — we use IEEE-754's canonical NaN but don't exploit it)
- Backend traits that assume the target has fp64 at all (the trait should accept a "fp64 availability" capability bit)
- Tests that depend on hardware-specific fp64 behavior beyond what IEEE-754 specifies

None of these are current violations. Phase 1's IR is clean on all three. This subsection is preventive documentation, not a corrective action.

**Status:** deferred, named, documented. Revisit after Peak 7 summits.

---

## Part III — Principles that bind the journey (softer but important)

- **Incrementality.** Every campsite is small enough to finish in one focused work session, test in isolation, and commit. If a campsite feels bigger than "one person, one day, one test file," split it.
- **Incremental oracle.** Every new op, every new function, every new backend ends with a test that compares the result to a previously-trusted reference. We never grow the trusted set without evidence.
- **Refuse convenience.** When the team wants to cut a corner ("just inline it", "just use the existing function", "just trust the library"), the convenience is almost always the direction of the pitfalls we're avoiding. The team should assume that the easy path is the wrong path and have to prove otherwise.
- **Text-first, binary-later.** Every IR we design has a human-readable text encoding *first*. Binary comes later, optionally. Text round-trips are trivially debuggable; binary formats hide bugs.
- **Shapes over contents.** The IR architect decides what *kinds* of things exist before the contents of any particular kind. The libm person writes functions, but only in terms of the ops the IR architect has declared. Don't let the libm person demand new ops to "make their life easier" — that inverts the authority.
- **Document the weather, not the work.** Campsites record what *almost* went wrong, what the pitfalls were, what tempted the implementer off-path. The diffs themselves are self-documenting; the *near misses* are what help the next person.
- **Adversarial pressure from day one.** The adversarial mathematician role isn't a final check — it's a continuous pressure. Every recipe and every libm function gets hit with pathological inputs *during* implementation, not *after*.

---

## Part IV — The Seven Peaks (deep)

### Peak 1 — `.tam` IR: the shape authority

**What it is.** A small, typed, explicit instruction set that expresses "compute this expression per element, reduce into these slots, write here." Big enough to hold today's recipes. Small enough to fit in a page.

**Not.** Not LLVM. Not MLIR. Not a compete-with-SPIR-V. Not a tensor DSL. We are explicitly avoiding the temptation to over-engineer. If it's not needed for a recipe we actually have, it does not exist.

**Scope for Phase 1 (accumulates on ℝⁿ → ℝ):**

```
Types:
  i32, i64, f64
  buf<f64>   ; a flat fp64 buffer with a length
  (no vectors, no tensors, no matrices, no shape system)

Values:
  %r0, %r1, ...  virtual registers, SSA form
  (no aliasing, no mutable slots other than explicit buffer stores)

Ops (the complete initial set):
  ; constants
  const.f64  %dst = <lit>
  const.i32  %dst = <lit>

  ; buffer
  bufsize    %dst:i32 = %buf
  load.f64   %dst = %buf, %idx:i32
  store.f64  %buf, %idx:i32, %val

  ; arithmetic (non-contracting)
  fadd.f64   %dst = %a, %b
  fsub.f64   %dst = %a, %b
  fmul.f64   %dst = %a, %b
  fdiv.f64   %dst = %a, %b
  fsqrt.f64  %dst = %a
  fneg.f64   %dst = %a
  fabs.f64   %dst = %a

  ; integer
  iadd.i32   %dst = %a, %b
  isub.i32   %dst = %a, %b
  imul.i32   %dst = %a, %b
  icmp_lt.i32 %dst:pred = %a, %b
  (no i32 div, no modulo — not needed in Phase 1)

  ; comparisons (predicate-typed)
  fcmp_gt.f64 %dst:pred = %a, %b
  fcmp_lt.f64 %dst:pred = %a, %b
  fcmp_eq.f64 %dst:pred = %a, %b

  ; control
  select.f64 %dst = %pred, %t, %f   ; branch-free
  loop_grid_stride { ... }          ; structured block, see below

  ; transcendental stubs — always lower to calls to tambear-libm
  tam_exp.f64 %dst = %a
  tam_ln.f64  %dst = %a
  tam_sin.f64 %dst = %a
  tam_cos.f64 %dst = %a
  tam_pow.f64 %dst = %a, %b
  (more as needed; each one MUST have a libm implementation in same IR)

  ; reduction
  reduce_block_add.f64  %dst_buf, %slot_idx:i32, %val     ; writes per-block partial
  (no reduce_global yet; CPU does the final fold)

Entry points:
  kernel <name>(buf<f64> %x, buf<f64> %y, buf<f64> %out) { body }
  func <name>(f64 %a) -> f64 { body }       ; libm functions live here
```

The key design choices:

- **SSA, not stack.** Registers are single-assignment. Easier to verify, easier to lower.
- **Structured loop, not goto.** `loop_grid_stride { ... }` is an atom. Each backend knows how to lower "iterate from `gi` to `n` with stride `stride`." No raw branches in user-written .tam.
- **Predicates are a type.** Comparisons produce `pred`, not `f64`. Select is the only branch. No if/else in the IR body.
- **Transcendentals are ops, not library calls.** `tam_exp` is an opcode, not a function call with ABI baggage. Each backend inlines tambear-libm's implementation of that opcode. This means the libm lives in the *same* IR.
- **Reductions are explicit.** We have one reduction op, `reduce_block_add.f64`, with a known semantic (per-block partials into an output buffer slot, CPU finalizes). Later we add `reduce_global_add`, `reduce_block_max`, etc. when we need them.

**Text encoding — example, the variance kernel hand-written:**

```
.tam 0.1
.target cross

kernel variance_pass(buf<f64> %data, buf<f64> %out) {
entry:
  %n     = bufsize %data
  %acc0  = const.f64 0.0
  %acc1  = const.f64 0.0
  %acc2  = const.f64 0.0
  loop_grid_stride %i in [0, %n) {
    %v     = load.f64 %data, %i
    %v2    = fmul.f64 %v, %v
    %one   = const.f64 1.0
    %acc0' = fadd.f64 %acc0, %v
    %acc1' = fadd.f64 %acc1, %v2
    %acc2' = fadd.f64 %acc2, %one
  }
  %s0 = const.i32 0
  %s1 = const.i32 1
  %s2 = const.i32 2
  reduce_block_add.f64 %out, %s0, %acc0'
  reduce_block_add.f64 %out, %s1, %acc1'
  reduce_block_add.f64 %out, %s2, %acc2'
}
```

(Loop-carried values — `%acc0 → %acc0'` — need real phi semantics. The loop block takes phi-nodes at entry. The detailed syntax is the IR architect's call.)

**Why this is a peak and not just a file.** Because every op added here binds every backend, every libm function, every future vendor door, and every test oracle. The wrong set of ops early turns into retrofits forever.

---

### Peak 2 — `tambear-libm` Phase 1: our own transcendentals

**What it is.** A library of `.tam` functions that implement `exp`, `log`, `sin`, `cos`, `sqrt`, `pow`, `tanh`, `sinh`, `cosh`, `asin`, `acos`, `atan`, `atan2`. Each is a sequence of `fadd` / `fmul` / `fdiv` / `fsqrt` plus constants. That's all. No calls to anything external.

**Accuracy target (the most important decision).** The team must pick ONE of:

- **Correctly rounded (0.5 ULP):** the answer matches the exact mathematical answer rounded to the nearest fp64 value. This is what CRlibm / SUN aim for. VERY expensive, especially for trig.
- **Faithfully rounded (1 ULP):** within 1 ULP, no guarantee it's *the* correct rounding. This is what most libms actually hit on average.
- **Bounded ULP (≤ N ULPs for documented N):** our stated guarantee. 3–4 ULPs is enough for almost every statistical use.

Recommendation: **faithfully rounded (≤ 1 ULP)** as the default, with a per-function measured max ULP recorded in the docs. Correctly rounded is a follow-on project for specific functions if users need it.

**Reference.** mpmath at ≥ 50-digit precision (we've already used it elsewhere in winrapids). The test matrix is: generate 1M random fp64 inputs in each function's domain, compute reference to 50 digits, compute ours, measure max and mean ULP error, assert max ≤ target.

**Structure of a libm function:**

```
.tam 0.1
func tam_exp(f64 %x) -> f64 {
  ; 1. Argument reduction:  x = n * ln2 + r  with |r| ≤ ln2/2
  ; 2. Compute exp(r) via polynomial
  ; 3. Reassemble:  exp(x) = 2^n * exp(r)
  ; ... (real PTX-level fadd/fmul soup)
}
```

The algorithmic choices per function:

- **sqrt:** direct hardware op via `fsqrt.f64`. IEEE-specified to be correctly rounded. Done.
- **exp:** Cody-Waite split `ln(2)` into high and low parts (each fp64) to preserve precision during reduction. Core polynomial of degree ~10 (Remez minimax) on the reduced interval. Reassembly via `ldexp` (= bit manipulation of the exponent field).
- **log:** extract exponent to get `m * 2^e`, compute `log(m)` on `[1, 2)` via polynomial, combine with `e * ln(2)`.
- **sin/cos:** Cody-Waite reduction of `x` to `[-π/4, π/4]` by subtracting multiples of `π/2`, tracking the quadrant in an integer. Small-range polynomial (degree ~8 for sin, ~7 for cos). Big-argument cases (|x| > 2²⁰) need Payne-Hanek reduction — **defer** to Phase 2.
- **pow:** `pow(a, b) = exp(b * log(a))` with careful handling of `a ≤ 0`, integer `b`, `0^0`, `inf`, nan, etc. Special cases are most of the code.
- **tanh/sinh/cosh:** compositions of exp. Watch for overflow.
- **asin/acos/atan:** polynomial on reduced range. `atan` is the base; `asin`/`acos` compose via identities and sqrt.
- **atan2:** atan plus quadrant handling.

**The pitfalls here are brutal:**

- Catastrophic cancellation in argument reduction: `x - n*π/2` loses precision when `n*π/2` ≈ `x`. Cody-Waite's dual-precision constant trick exists exactly to avoid this. The implementer must understand it.
- Polynomial coefficient precision: naive Horner with single-precision coefficients silently loses 2–3 bits. Coefficients must be stored as fp64 literals with bit-exact values.
- Subnormal handling: `exp(-700)` is in the subnormal range. The reassembly step must not flush to zero.
- Special values: `log(0) = -inf`, `log(-1) = nan`, `exp(inf) = inf`, `pow(0, 0) = 1` (by convention), `atan2(0, 0) = 0` (by convention). Every single one of these must be tested.
- The polynomial evaluation order: Horner's scheme has a specific sequence of `fmul`+`fadd` that must not be reassociated. Every backend must produce the same instruction sequence.

Recommendation: the libm implementer starts with **sqrt** (one line), then **exp** (hardest of the "fundamental" set — teaches range reduction, polynomial, reassembly in one function), then everything else falls out.

---

### Peak 3 — tam→PTX assembler: our own door, raw

**What it is.** A translator from `.tam` IR to valid PTX text, plus a `cudarc` integration that loads PTX bytes via `cuModuleLoadDataEx` directly (no NVRTC, no CUDA C).

**PTX in one paragraph.** PTX is NVIDIA's virtual ISA — text-based, register-based, versioned. A PTX file looks like:

```ptx
.version 8.5
.target sm_120
.address_size 64

.visible .entry variance_pass(
  .param .u64 data,
  .param .u64 out,
  .param .u32 n
) {
  .reg .b64 %rd<4>;
  .reg .b32 %r<4>;
  .reg .f64 %fd<8>;
  .reg .pred %p<2>;
  ld.param.u64 %rd1, [data];
  ld.param.u64 %rd2, [out];
  ld.param.u32 %r1, [n];
  cvta.to.global.u64 %rd1, %rd1;
  cvta.to.global.u64 %rd2, %rd2;
  ...
}
```

Our assembler produces this text from .tam IR. The driver (`nvcuda.dll`) JITs it to the actual SASS (the real machine code) at load time.

**Campsite sequence (this peak is the biggest cliff on the trek):**

1. **Read and index the PTX ISA spec.** NVIDIA's PDF, chapters on `.version`, `.target`, data types, memory spaces, the instruction set. Take notes specifically on:
   - Required directives (`.version`, `.target`, `.address_size`)
   - Parameter passing convention for kernel entry
   - Address space conversions (`cvta.to.global`)
   - `.contract` suffix and default behavior (**critical for I3**)
   - FP64 atomic support (`atom.global.add.f64` requires sm_60+)
   - Shared memory declaration syntax (`.shared .align 8 .b8 smem[N]`)
   - Branch and predication syntax

2. **Build a PTX *printer* (not a parser).** Structured Rust types → valid PTX text. Start with the simplest possible program: empty kernel that does nothing.

3. **Raw module loading in tam-gpu.** Add `CudaBackend::compile_ptx_raw(ptx: &str, entry: &str) -> Kernel` that uses `cudarc::driver::CudaContext::load_module(ptx_bytes)` — which, crucially, accepts PTX text and calls the driver's load path, NOT NVRTC. This is the change that removes the vendor compiler.

4. **Hello-world PTX kernel.** Write by hand the minimum kernel that writes `42.0` to `out[0]`. Load it through the new raw path. Dispatch. Read back. Assert `out[0] == 42.0`. This is the first proof that our path works end-to-end without NVRTC.

5. **Extend the hand-written kernel:** `out[0] = in[0] + in[1]`. Tests `ld.global.f64`, `st.global.f64`, `add.f64 .contract false`, parameter passing.

6. **Now, the translator.** Implement one .tam op at a time:
   - 6a. `const.f64` → `mov.f64 %fdN, 0d40091eb851eb851f;` (with exact bit-encoding of the constant — this is where `f64::to_bits` + formatting goes in)
   - 6b. `load.f64` → `ld.global.f64 %fdN, [%rdM];` (with address computation)
   - 6c. `store.f64` → symmetrical
   - 6d. `fadd.f64` → `add.f64 %dst, %a, %b; // .contract false is the default when we specify it`. **Must explicitly emit `.rn` rounding mode.**
   - 6e. `fmul.f64` → `mul.f64 .rn %dst, %a, %b;`
   - 6f. `fsub`, `fdiv`, `fsqrt`, `fneg`, `fabs` — one each
   - 6g. `loop_grid_stride` → this is the hard one. Requires:
     - Computing `gi = blockIdx.x * blockDim.x + threadIdx.x` (`mov.u32 %tid, %ntid.x;` etc.)
     - Computing `stride = gridDim.x * blockDim.x`
     - A loop header label, increment by stride, comparison against n, conditional branch
     - Phi nodes for loop-carried accumulators — the trickiest single thing in the translator
   - 6h. `reduce_block_add.f64` — per-block shared memory tree reduction, final block-zero thread writes to `out[slot]`. (For Phase 1 we can cheat and atomicAdd here; Peak 6 makes it deterministic.)

7. **Register allocation.** PTX has unlimited virtual registers (`%fd0`, `%fd1`, ..., `%fd<N>`), so naive allocation is "give every SSA value a new register." That works and is correct; we don't need liveness analysis for correctness, only for register count minimization. Start naive.

8. **First real program:** translate the hand-written variance kernel from Peak 1 through the assembler. Load via raw path. Dispatch. Verify against CPU interpreter from Peak 5.

9. **Translate every Peak-1 recipe** — sum, mean, sum_sq, l1_norm, rms, pearson_r, the custom Σ|ln x|.

10. **Integrate tambear-libm.** When .tam IR has `tam_ln.f64`, the translator inlines the PTX for `tam_ln` — i.e., it function-calls into our own `.func` emitted alongside the kernel, or it inlines the body at the call site (simpler for Phase 1, worse for code size). Recommendation: inline. Functions are a later cleanup.

11. **Cross-check against NVRTC.** For every kernel, compile the *same* math two ways: (a) through our tam→PTX path, (b) through NVRTC using the old CUDA-C codegen we built today. Run both. Compare. Agreement within the libm ULP bound is the target. **Disagreement is a bug in one of the two; it must be diagnosed, not relaxed.**

12. **Diff the PTX itself.** Run `diff` on NVRTC's PTX output vs ours for `variance_pass`. Expect it to be quite different (NVRTC optimizes aggressively) — the point is that we *own* the sequence. Document the differences.

**Pitfalls on this peak:**

- **Forgetting `.contract false`.** PTX defaults to contract-everywhere, which emits FMA whenever it can. This silently violates I3 and introduces cross-hardware drift. Every `add.f64` after a `mul.f64` that uses the result must be separately emitted, or we must use `.contract off` on the `add`. Easy to miss.
- **Wrong rounding mode.** `add.f64` is ambiguous; `add.rn.f64` is round-to-nearest-even. Every fp op needs the explicit `.rn`.
- **Parameter passing.** Kernel parameters in PTX go through `.param` space, loaded explicitly. Getting the addresses right is fiddly and the failure mode is "kernel launches, reads garbage, dispatches silently, returns zeros."
- **Block-zero thread handling for reductions.** Only one thread per block should write the partial. Miss the predicate and you double-write → wrong answer.
- **`cuModuleLoadDataEx` error messages are terrible.** A PTX syntax error gives you a line number and a cryptic message. The implementer must build a local PTX linter (even `ptxas --verify` from the CUDA toolkit, used only during dev, not as a runtime dep) to sanity-check emitted code.
- **The urge to use NVRTC as a "just for this one op" fallback.** No. Every op must flow through our assembler. If an op is too hard to emit ourselves, that's a reason to defer the op, not to fall back.

---

### Peak 4 — Replay: the continuous oracle

**What it is.** A test harness that runs any .tam program through multiple backends and asserts the results are bit-exact (for pure arithmetic) or within the libm ULP bound (for transcendentals). Every test in `tests/gpu_end_to_end.rs` ports here.

**Structure:**

```rust
fn oracle_diff(program: &TamProgram, inputs: &Inputs) {
    let a = run_backend(Backend::CpuInterpreter, program, inputs);
    let b = run_backend(Backend::CudaPtxRaw,     program, inputs);
    let c = run_backend(Backend::CudaNvrtcRef,   program, inputs); // legacy path, kept for diff
    // Later: Vulkan, Metal, DX12

    for (name, values) in [("cpu", a), ("cuda_raw", b), ("cuda_ref", c)] {
        assert_bit_exact_or_within_ulp_bound(&a, &values, name);
    }
}
```

**Tolerance policy (the hardest part):**

- **Pure arithmetic kernels** (no transcendentals): `==` bit-exact. No tolerance. If two backends disagree, one of them is lowering fadd wrong.
- **Transcendental kernels**: per-function ULP bound, from tambear-libm's published accuracy. So if `tam_exp` is documented at ≤ 1 ULP, the end-to-end test tolerates up to `exp_error * n_calls` ULPs — very conservatively.
- **Reduction kernels with non-deterministic reduce** (Peak 1–3): legacy, will be fixed in Peak 6. Mark as `@xfail_nondeterministic` until Peak 6 lands.

**The "hard cases" suite** — adversarial inputs:

- `data = [1e16, 1.0, -1e16]` — catastrophic cancellation in `sum`
- `data = [subnormal, subnormal, ...]` — near-underflow
- `data = [nan, 1.0, 2.0]` — nan propagation
- `data = [inf, -inf, 1.0]` — inf arithmetic
- `data = [1.0; 1_000_000_000]` — large-N scaling
- `data = [empty]` — empty input edge case
- `data = [one_element]` — degenerate variance (div by zero)
- `variance([x+1e6 for x in uniform])` — the classic one-pass-variance failure mode

This suite becomes the ongoing regression test. New backends must pass it.

---

### Peak 5 — tam→native CPU: the oracle, realized

**What it is.** A CPU implementation of the .tam IR. Two layers:

- **5a: Interpreter** — walks the .tam AST, executes each op as a Rust function call. Slow, trivially correct, pure `std::f64` operations. The reference oracle.
- **5b: JIT codegen (later)** — translates .tam IR to machine code via cranelift (or another Rust-native JIT). Fast, but only after the interpreter is battle-tested.

Peak 1 already includes the interpreter as a deliverable (1.10). This peak is the formalization — making the interpreter a first-class backend with its own test suite and its own libm handling.

**Critical decision: how CPU runs tambear-libm.**

Option A: The CPU interpreter executes tambear-libm's .tam IR, op by op, the same way it executes a user kernel. Slow but pure: `tam_sin` on CPU = interpret the .tam polynomial → pure fadd/fmul sequence → same result as PTX-emitted version.

Option B: Short-circuit — `tam_sin` on CPU calls `f64::sin`. Fast but violates I1 / I8 / I9 — we'd be comparing against glibc's sin, which is the enemy.

**Take Option A.** Pay the speed cost. This is why the interpreter matters more than the JIT.

**Campsites:**

- 5.1. Define `trait TamBackend` with: `run(&self, program: &TamProgram, inputs: Inputs) -> Outputs`.
- 5.2. Implement `CpuInterpreterBackend` — huge match on opcodes, one arm per op, uses `std::f64::*` only for `+ - * / sqrt`, nothing else.
- 5.3. Port today's tests to run through the backend.
- 5.4. Wire tambear-libm's .tam functions into the interpreter. When a kernel calls `tam_exp`, the interpreter recursively runs the `tam_exp` .tam function. No shortcuts.
- 5.5. Compare interpreter-with-libm output against mpmath reference.
- 5.6. (Later, after Peak 6) JIT via cranelift. Each .tam op → one or two cranelift IR ops. cranelift then emits x86-64/aarch64. Still goes through our libm.

**Pitfalls:**

- **Interpreter calls `f64::sin` "just to get unblocked."** No. The whole point is that the interpreter is the reference, and the reference is *our math*.
- **JIT introduces new behavior.** If the JIT and the interpreter disagree, the interpreter is correct. Rule: any JIT divergence is a JIT bug.
- **Cranelift does implicit FMA?** Check. If yes, explicit suppression.

---

### Peak 6 — Deterministic reductions

**What it is.** Replace `atomicAdd`-based reductions with a fixed-order tree reduce that produces bit-identical output on every run, every hardware.

**Strategy: two-stage reduce.**

1. **In-kernel stage:** each block computes a local reduction over its assigned grid-stride slice via shared memory tree reduce. Block 0 thread 0 writes the block's partial sum to `partials[block_idx]`.
2. **Host stage:** the host reads `partials[0..n_blocks]` and folds them in a fixed order (e.g., left-to-right), producing the final answer.

Why two-stage and not single-kernel:
- The host fold is trivially deterministic.
- It's simple.
- Phase-1 `n_blocks` is small (tens to hundreds), so the host fold is free.
- We can revisit once we need to fuse the fold into the kernel for speed.

**Fixed launch config rule.** For a given `N`, the `(grid, block)` pair must be a deterministic function of `N`. Document: e.g., `block = 256`, `grid = min(ceil(N / block), 1024)`, with a grid-stride loop picking up the rest. Any variation produces a different reduction tree and therefore different bits.

**New .tam IR ops:**

- `reduce_block_partial.f64 %partials_buf, %block_slot_offset, %val`
- (optional later) `reduce_global_add.f64 %out_buf, %slot, %val` that hides the two-stage internally.

**Test:**

```rust
#[test]
fn reduction_is_deterministic_across_runs() {
    let data = large_hard_dataset();
    let first = run_reduction(&data);
    for _ in 0..100 {
        let next = run_reduction(&data);
        assert_eq!(first.to_bits(), next.to_bits(),
            "reduction is non-deterministic");
    }
}
```

**Then the acceptance test from Peak 4:**

```rust
#[test]
fn cpu_gpu_bit_exact_pure_arithmetic() {
    let cpu = run_cpu_interp(&variance_program, &data);
    let gpu = run_cuda_raw   (&variance_program, &data);
    assert_eq!(cpu.to_bits(), gpu.to_bits()); // no tolerance
}
```

**Pitfalls:**

- **Partial-sum buffer size must match `n_blocks`.** Off-by-one here is silent.
- **Block-reduce shared memory allocation.** Declaring `.shared .align 8 .b8 smem[block_size*8]` and indexing correctly with threadIdx.
- **`__syncthreads()` at every tree step.** Missing one → race condition → flaky bits.
- **Empty blocks.** If grid-stride doesn't touch a block, its partial must be the identity (0.0 for Add). Initialization matters.
- **Odd block sizes.** Tree reduce assumes power-of-two block. Either enforce it or handle the remainder carefully.

---

### Peak 7 — Second vendor door: Vulkan via SPIR-V (Windows, this machine)

**What it is.** Repeat Peaks 3 for a different hardware. The architectural claim is tested by producing bit-identical pure-arithmetic outputs from CPU, CUDA, and Vulkan on the same `.tam` program.

**Why Vulkan second and not Metal:** Metal requires macOS; this machine is Windows. Vulkan has drivers on Windows for NVIDIA, AMD, and Intel — so it also exercises "same SPIR-V, different GPU vendor" in the future. It's the right next step for *this* machine.

**SPIR-V in one paragraph.** Binary module format. Sequence of 32-bit words. Header + bound + instruction stream. Each instruction is `(word_count, opcode, operand, operand, ...)`. Types, constants, functions, control flow. More formal than PTX, less forgiving to generate by hand, but there are good Rust crates (`rspirv`) that give us structured construction.

**Vulkan compute in one paragraph.** Create instance → enumerate devices → create logical device + compute queue → create shader module from SPIR-V bytes → create pipeline → allocate buffers → descriptor sets → command buffer → dispatch → wait → readback. Way more ceremony than CUDA. `ash` (Rust vulkan bindings) or `vulkano` (higher-level) are the options. Recommendation: `ash` — thinner, maps directly to the spec, no translation layer to disagree with.

**Campsites:**

- 7.1. Verify Vulkan drivers work on this machine. `vulkaninfo` or equivalent. Confirm fp64 support in shaders (RTX 6000 Pro Blackwell has it, AMD's consumer parts often don't — this is the kind of feature-gap that I6 says we don't paper over).
- 7.2. `ash` hello-world: instance → device → queue. Pull up device name.
- 7.3. Hello-world SPIR-V module — writes 42.0 to `out[0]`. Hand-authored bytes, or via `rspirv` builder, whichever is less painful. Compile Rust → load module → create pipeline → dispatch → readback → assert 42.0.
- 7.4. `VulkanBackend` in `tam-gpu`. Parallel structure to `CudaBackend`: alloc, copy, compile, dispatch, sync. Note: `compile` here takes SPIR-V *bytes*, not text.
- 7.5. tam→SPIR-V translator. One op at a time:
  - `const.f64` → `OpConstant f64 <bits>`
  - `fadd.f64` → `OpFAdd` with `NoContraction` decoration (**I3**)
  - `fmul.f64` → `OpFMul` with `NoContraction`
  - `load.f64` → `OpLoad` through storage buffer
  - `store.f64` → `OpStore`
  - `loop_grid_stride` → `OpLoopMerge` + phi + conditional branch (structured control flow)
  - `fsqrt.f64` → `OpExtInst std_450 Sqrt`
  - `reduce_block_*` → shared memory (`Workgroup` storage class) + `OpControlBarrier` + tree reduce
  - tambear-libm ops → inlined SPIR-V sequences, same approach as PTX
- 7.6. Translate each Peak-1 recipe, dispatch on Vulkan, compare to CPU interpreter.
- 7.7. **The moment.** On this machine, run `variance_pass` through all three backends:
  - CPU interpreter (pure Rust, tambear-libm interpreted)
  - CUDA raw (our PTX, our libm, deterministic reduce)
  - Vulkan (our SPIR-V, our libm, deterministic reduce)

  Assert `cpu.to_bits() == cuda.to_bits() == vulkan.to_bits()`.

  If that passes, the architectural claim is demonstrated. That's the summit.

- 7.8. (Metal later, macOS only — same shape, different door, no new ground.)
- 7.9. (DX12 later — HLSL via DXIL, same shape again.)

**Pitfalls:**

- **fp64 feature gating.** Vulkan's `shaderFloat64` is optional. Must be queried and enabled explicitly. If missing, I6 says: error, don't silently downgrade.
- **Storage buffer layout.** `std430` is the right layout for fp64 arrays. `std140` adds 16-byte alignment per field which will scramble our stride. Get this right or data reads are garbage.
- **`NoContraction` decoration.** SPIR-V has explicit "don't contract this" via a decoration on each fp op. Forget it, and drivers may contract, and we're back to cross-hardware drift.
- **Descriptor set complexity.** CUDA is "here are pointers," Vulkan is "here are buffers, bound to bindings, in sets, in layouts." Ten-minute task vs two-day task. Budget accordingly.
- **`ash` ergonomics.** Raw vulkan is verbose. The Vulkan implementer will want `vulkano` or similar. Fine for the backend boilerplate; but SPIR-V emission must stay in our hands.
- **Subgroup / workgroup size tuning differences per vendor.** Start with a fixed safe config (workgroup = 64 or 128). Don't let per-vendor tuning in at this stage — that breaks determinism.

---

## Part V — The team

### Roles

| Role | Mission | Key skills | What they own |
|---|---|---|---|
| **Navigator** (me / orchestrator) | Keep the team on architecture, refuse invariant violations, broker decisions between roles | Tambear philosophy, accumulate+gather, I1–I10, pitfall recognition | The map, the invariants, escalation calls |
| **IR Architect** | Define and maintain `.tam` — the shape authority | Compiler IR design, SSA, small languages, text/binary formats | `.tam` spec, parser, printer, verifier, versioning policy |
| **Libm Implementer** | Build tambear-libm from first principles | Numerical analysis, IEEE 754, range reduction, polynomial approximation, ULP bounds | Every transcendental function, its tests, its accuracy docs |
| **PTX Assembler** | Lower `.tam` to PTX and load it via the driver | PTX ISA, cudarc driver API, register allocation, low-level GPU | tam→PTX, raw module loading, determinism in PTX kernels |
| **SPIR-V Assembler** | Lower `.tam` to SPIR-V and dispatch via Vulkan | SPIR-V binary format, Vulkan compute, `ash`, storage buffers | tam→SPIR-V, Vulkan backend, cross-hardware validation |
| **CPU Backend Implementer** | Interpreter first, JIT later; the reference oracle | Rust, AST walkers, (later) cranelift | CPU interpreter, eventually cranelift JIT |
| **Test Oracle** | Builds and runs the cross-backend diff harness; maintains mpmath references | Numerical testing, property-based testing, mpmath/SymPy | Oracle harness, ULP measurement, regression matrix, determinism tests |
| **Adversarial Mathematician** | Find numerical weak points in recipes, libm, and codegen | Error analysis, catastrophic cancellation, subnormals, pathological inputs | The "hard cases" suite, root-cause analysis of drift |
| **Math Researcher** | Research correct algorithms (which Remez polynomial? which range-reduction scheme?) | Literature, libm history, scientific computing | Algorithm selection docs per function |
| **Scientist / Logbook Keeper** | Document every campsite's *near-misses*, not just outcomes | Technical writing, lab notebook discipline | Campsite logs, pitfall journal, lessons-learned |

The team doesn't need 10 people. Roles can double up:

- IR Architect + CPU Backend: natural pairing — the architect needs an interpreter to validate the IR, so they often are the same person early.
- PTX Assembler + SPIR-V Assembler: can be the same person sequentially (not simultaneously — each is a deep dive).
- Libm Implementer + Math Researcher: same person, because the research informs the implementation intimately.
- Test Oracle + Adversarial Mathematician: same person, or tightly paired.

Minimum viable team: **IR Architect, Libm, Assembler, Test Oracle, Adversary**. Five.

### How the roles interact

```
              IR Architect
             /     |      \
            /      |       \
          Libm   Assemblers  CPU Backend
            \      |       /
             \     |      /
              Test Oracle
                   |
              Adversary
                   |
              (finds pitfalls → everyone)
```

**Decisions flow down from IR Architect.** New op? Architect approves. Type system question? Architect. Text encoding? Architect. The architect's authority is narrow — *only* the shape — but within that scope it's final.

**Test Oracle is the referee.** Disputes between backends (CPU says X, GPU says Y) go to the oracle, who builds a test case that pins the truth. Whichever backend diverges from the oracle is the bug.

**Adversary is the loyal opposition.** Their job is to make everyone else's life harder in a way that improves the product. Every week they pick one thing ("let me try subnormal inputs to your exp", "let me test variance on data where mean is 1e10 and std is 1e-5") and break it.

**Navigator escalates only on invariant violations or scope creep.** Day-to-day decisions stay within roles.

---

## Part VI — The campsite list

See `campsites.md` for the full numbered list.

---

## Part VII — Pitfalls and quicksand, marked on the map

See `pitfalls.md` for the full list with rationale. Quick reference of top pitfalls:

1. **"Let me just use NVRTC for this one case."** No. Every op flows through our assembler. If too hard, defer the op.
2. **"Let me just call `f64::sin()` in the interpreter."** No. The interpreter calls tambear-libm, all the way down.
3. **"The NVRTC version has 1 ULP of error, let's raise the tolerance."** No. Root-cause before relaxing.
4. **"FMA is fine, it's more accurate."** Different bits. Either ALL backends FMA (new explicit op) or NONE.
5. **"Let me just port musl's sin."** No. I8. First principles.
6. **"The block count should vary with N for performance."** No. Deterministic launch config.
7. **"atomicAdd is good enough for now."** Only until Peak 6. Mark @xfail_nondeterministic.
8. **"The IR needs more ops to make my life easier."** IR Architect has veto.
9. **"Let me just use rspirv / cranelift."** As typewriters, yes. As semantic authorities, no.
10. **"The test passes on my machine."** All backends pass, or nothing passes.
11. **Defaulting to single-precision anywhere.** Every kernel declares fp64 explicitly.
12. **One-pass variance instability.** Known pitfall, document + plan Welford alternative.
13. **Assuming `rint` rounds to nearest-even.** Be explicit everywhere.
14. **Shared memory bank conflicts.** Not a correctness issue — defer to Phase 3.
15. **Forgetting the `.param` → register load at kernel entry in PTX.** Silent zeros failure mode.

---

## Part VIII — Weather and pace

- **The first week is Peaks 1 and 5.1–5.4 together.** .tam IR, interpreter, sum and variance working end-to-end through pure Rust. No GPU. No libm. Just "can we express our recipes in our own IR and run them on our own interpreter." This is the warm-up camp.
- **The second push is libm.** `sqrt → exp → log → sin/cos → pow`. Each one on the mpmath-reference accuracy bar. This is slow, patient, numerical work. It's the part of the trek that can't be rushed.
- **The third push is the PTX assembler.** The single biggest cliff. Allow it to take longer than expected. Cost of rushing here is subtle bugs that show up only later.
- **Peak 6 (determinism) is a small detour, not a peak, but it unlocks the summit.**
- **Peak 7 is the summit push.** Short and intense. Most of the work is Vulkan API boilerplate, not `.tam` work; the translator is mostly the same shape as the PTX translator.

**Pace rule:** at most one peak is "active" at a time. Lower peaks are frozen (no new features, only bug fixes) once the team moves on. This keeps the cross-backend diff meaningful — we're not chasing a moving target.

**Escalation rule:** any campsite that hits an invariant wall (I1–I10) halts, and the team surfaces it to Navigator. Navigator adjudicates within a day; if they can't, we bring in fresh eyes. Never paper over an invariant violation by quiet mutual agreement.

**Logbook rule:** every campsite ends with a short entry: what was done, what almost went wrong, what tempted me off-path. The entries are the most valuable document we produce, because they're the map for the next traveler.

---

## Part IX — What's different about this trek

Most compiler projects are vertical — pick a target, go deep, stabilize, move to the next. This trek is *horizontal* — the primary deliverable is **cross-backend equivalence**, not any individual backend's capability. That inverts the usual rhythm:

- You don't "finish" PTX and then "start" SPIR-V. You land the sum kernel on PTX, verify it matches CPU, then land the sum kernel on SPIR-V and verify it matches both. The integration is continuous, not final.
- You don't "optimize" anything until all backends agree at the correctness level.
- You don't trust any single backend — two-backend quorum is the minimum for believing a result is right.

It also has a property most compiler trips don't: **the hardest part is the math, not the codegen.** tambear-libm is *the* challenge. If you get libm right, the backends are just transcription. If you get libm wrong, no amount of backend cleverness recovers.

---

## Part X — Starting state

We have:
- 107 tests green on the easy path
- Working `codegen/cuda.rs` (vendor-locked, to be replaced)
- `tam-gpu` with a live `CudaBackend` on this machine's Blackwell
- Recipes that compile fusion down to 4 passes
- CLAUDE.md invariants that describe the destination

We need:
- `.tam` IR definition + parser + printer + interpreter (Peak 1)
- The team, spinning up now.

See `navigator/state.md` for a complete snapshot.
