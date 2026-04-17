# TAM as Distributed Compute Scheduler

**Status:** architecture capture, pre-code.
**Supersedes:** the "CPU first / GPU later" sequencing in `2026-04-11-op-default-deterministic-plan.md`.
**Driver:** redirect from the user on 2026-04-17: tambear is not being built as a CPU library that later adds GPU backends. It is being built as a distributed compute product from the start, where "use every available surface" is the default and single-surface pinning is the opt-out.

---

## TAM in one paragraph

TAM is the orchestrator that sits above every piece of hardware in reach. It takes a compiled pipeline and dispatches pieces of it to every available surface — full CPU, every GPU, remote GPUs over NVIDIA Sync, remote-mounted compute, future accelerators — then collects bit-identical partial results from each surface and merges them via Kulisch. The default behavior is "use every surface"; pinning to specific resources is the user's opt-out via the IDE. TAM itself does not know arithmetic. It knows available hardware, throttles, doorways, orchestration plans, and how to assemble bit-identical partials from heterogeneous sources.

## TAM as a mascot

TAM is a cute bear sitting on the ice above a pond. Under the ice, fish swim. Each fish can only see its own current. The bear sees the whole pond. Most algorithms written in literature look sequential from the fish's perspective because each fish can only see one current at a time. The bear sees that the currents are actually flowing in parallel — they just needed an observer above the ice to recognize the structure.

This mascot image carries the full technical story:

- **Kingdom A**: no ice. Just water. The fish swim in parallel; TAM doesn't need to raise anything.
- **Kingdom C**: TAM stands on ice, feeding work to the fish one iteration at a time. The ice is the outer loop; the water is the inner A.
- **Kingdom B**: the ice is above TAM. Genuine self-referential wall. The *compiler* tries to dissolve it into A via the classification-bijection / three-criteria test; if it succeeds, the ice melts. If it doesn't, the algorithm stays single-surface-sequential.
- **Kingdom D**: the ice is above TAM and the water is murky with stochastic eddies. No fixed intermediates possible. Not parallelizable at all.

**The raising happens at compile time, not at runtime.** TAM doesn't pause mid-swim to figure out if a current is actually parallel. The compiler does the dissolution pass. TAM reads IR that has already been raised.

## The Fock-raising theorem, operational

From `campsites/theory/20260410-fock-boundary-theorem/` and `campsites/theory/20260410164337-classification-bijection/`:

> A recurrence `s_{t+1} = M(???) · s_t` is Kingdom A iff its update map is a **monoid homomorphism** from data sequences (with concatenation) into a fixed-dimension endomorphism monoid on the state space. Equivalently: `M` depends on input data only, not on the current state value, and composing any k steps stays in the same fixed-dimension representation.

**The three-criteria test** (applied by the compiler, not by TAM):

1. **Affine augmentation** — can the recurrence be written as `s_{t+1} = A · s_t + B_t` where `A` is a constant matrix and `B_t` depends only on input data?
2. **Associative composition** — if not directly affine, does there exist an associative semigroup `(S, ·)` and a map `φ: D → S` such that `F(x, d) = φ(d)(x)`?
3. **Fixed-dimension representation** — composing k such maps stays in the same dimension. No representation growth.

Pass all three → Kingdom A. The compiler emits scan/scatter IR.

Worked dissolutions already shipped in tambear:

- **EMA** (`signal_processing.rs`): `s_t = α·x_t + (1-α)·s_{t-1}` → affine `(a=1-α, b=α·x_t)` → prefix scan.
- **GARCH(1,1) filter** (`volatility.rs`): `σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}` → companion matrix → prefix scan. Was labeled B for years before the dissolution.
- **ARMA MA terms** — were labeled B for the CSS implementation; Kalman-form dissolution reveals Kingdom A math with a constant `M` and data-determined `b_t`.
- **BOCPD** — `p_t = D(x_t) · H · p_{t-1} + reset_weight · e_0` where `D(x_t)` is data-determined diagonal and `H` is constant hazard matrix. Linear prefix scan on vector state.
- **Collatz verification** (`collatz_structural.rs`): per-step map on k-bit suffixes is affine; compose 2^k precomputed transforms; ~6 compositions for a 71-bit number instead of ~400 serial steps.

## The compile → artifacts → runtime pipeline

A project in the IDE is a pipeline: step 1, step 2, …, step N, each composed from atoms + primitives + recipes + slots via TBS syntax / formal notations / GUI / script. When the user hits RUN, compile produces **three kinds of artifact**:

### 1. The `.tam` orchestration plan

A text/structured description TAM reads at runtime. Answers:
- Which kernel binary runs on which surface
- What data feeds into that kernel (and where to fetch it from)
- What throttles and usage limits apply
- Where the kernel's output buffer lives
- What barrier / sync point follows
- What to do with collected partials (merge via Kulisch, write to disk, stream to GUI, etc.)
- What kernel runs next
- How to report progress

This is TAM's script. TAM reads it and acts.

### 2. Compiled kernel binaries, per pass, per door

A **pass** is a maximal fusion of pipeline steps that share one accumulate+gather sweep. The compiler performs this fusion aggressively:
- Adjacent steps that share a scatter structure fuse into the same pass.
- **Non-adjacent steps** that share a scatter structure *also* fuse. If step 2, step 5, and step 257 can all run in one sweep, they compile into one kernel.
- Steps that genuinely depend on a prior pass's output land in pass 2.
- Further dependencies → pass 3, and so on.

Each pass gets rendered as a kernel per **door**:
- `kernel1-cuda.ptx` — NVIDIA direct
- `kernel1-vulkan.spv` — Vulkan compute shader
- `kernel1-dx12.dxil` — DirectX 12 compute shader
- `kernel1-metal.metallib` — Apple Metal
- `kernel1-cpu-native.bin` — native Rust via rayon
- `kernel1-amd.bin` — AMD direct (ROCm)
- `kernel1-intel.bin` — Intel direct

Same math in every binary. Different source rendering per door. Vendor provides the doorway; we own every ALU instruction past the door.

**wgpu is not in the door list.** We are replacing wgpu by talking directly to each vendor's API. wgpu may ship as an optional convenience door late in the product, but not as the mechanism that carries the Kulisch arithmetic.

### 3. Project state persistence (TBS + IR for reopen)

When the user saves (as opposed to runs), the IDE persists project state as TBS and IR artifacts so the project reopens in the GUI with the same composition, the same parameters, the same hardware bindings, the same data bindings. Compile is triggered by RUN; save is cheaper than compile.

## Pass fusion rules

The compiler's fusion pass looks for steps that share:
- The same **grouping** (by-key on bucket_id, prefix scan, reduce to all, tiled block)
- The same **input batch** they read from (same source columns, same masking)
- Compatible **phi expressions** (multiple phi functions can ride the same scatter — tambear already supports 1–5 phi expressions in one kernel, see `scatter_multi_phi`)

When those line up, the compiler emits one kernel for the whole set. Non-adjacent pipeline steps can fuse if they read compatible inputs — the SIP Launch 1 / Launch 2 design is a worked example: 11 C* accumulations over the same tick stream fuse into one kernel; 23 E* prefix scans fuse into another. SIP's whole hour fits in 2 passes.

## The arithmetic chain across surfaces

Every surface runs the **same Kulisch math** on its shard of the data. Kulisch is 34 × i128 integer words with a radix at bit 2100 — enough headroom for any sum of finite f64 values, and **exactly associative under merge** because `KulischAccumulator::merge` is word-wise signed add with carry propagation (integer arithmetic, no rounding).

Cross-surface determinism chain:

1. Surface A computes partial sum over shard A → Kulisch register A
2. Surface B computes partial sum over shard B → Kulisch register B
3. Surface ... → Kulisch register ...
4. TAM collects all registers from all surfaces
5. TAM merges them via Kulisch merge (associative) → one register
6. Final `to_f64()` is correctly rounded to nearest-even

**This is bit-identical to running the entire computation on one surface.** Not approximately equal, not close to — literally the same bits. Any surface (CPU, CUDA, Vulkan, DX12, Metal, remote-sync GPU, remote-mounted NVMe compute) that implements Kulisch correctly produces partial registers that merge into the same total.

The `.tam` IR's cryptographic fingerprint is the commitment: "running this IR on any valid surface combination produces this exact bit pattern." That's what lets SIP's Merkle chain work whether SIP runs on one laptop or a 47-GPU cluster. The output is the same bits either way.

## Kingdom assumptions TAM relies on

| Kingdom | TAM's role | Realization |
|---|---|---|
| **A** | First-class. Dispatch across every surface. Merge via Kulisch. | The base product. Every surface runs the same kernel on its shard; partials merge bit-identically. |
| **C** | Run the outer loop; dispatch each iteration's A-inner across every surface. | Newton-fit, L-BFGS, EM, EKF, GARCH fitting. TAM coordinates iteration count; each iteration's inner A uses every surface. |
| **B** | Run on one surface, sequentially. Still Kulisch-deterministic. | Compiler flags genuine B with a warning. User can override if they're sure (some Kingdom B algorithms are still useful — just not parallelizable). |
| **D** | Reject at compile time, or run strictly sequential with explicit "no determinism guarantees across resampling" marker. | Particle filters, MCMC. These exist but are not what tambear optimizes for. |

Most algorithms the compiler encounters are dissolvable-B → A. The compiler's job is the dissolution pass. TAM trusts the classification and dispatches accordingly.

## The recipe / IDE / runtime split

- **Recipe** is **pre-compile, pre-IR**. It lives in the IDE. A recipe is a named composition of atoms + primitives + other recipes + slot parameters, like a named macro the user can pick from the GUI.
- **Pipeline** is a user-composed sequence of recipe calls, script fragments, formal notation blocks, GUI-clicked steps — any mix. Parameters are tuned interactively.
- **Compile** turns a pipeline into `.tam` + per-pass per-door kernel binaries + TBS/IR persistence for project reopen.
- **TAM runtime** reads the `.tam`, dispatches kernels through doors to ALUs, barriers, merges, writes outputs, emits intelligent reports, streams to the GUI which polls live.

None of this is arithmetic. The arithmetic is entirely in the kernels, which are opaque payloads TAM ships through doorways.

## The two shipping shapes

### Full tambear product

- IDE (visual composition, script editing, formal notation blocks)
- TBS syntax and the spec compiler
- Formal notation layer
- Compile toolchain (IR generation, pass fusion, per-door kernel rendering)
- TAM runtime (hardware discovery, orchestration, multi-surface dispatch, merge)
- Full recipe catalog
- Full primitive library (Kulisch, two-sum, two-product-fma, sum_k, etc.)
- Atoms: accumulate, gather

This is the authoring / research / production tool. Users compose pipelines interactively, hit RUN, and the product scales from one CPU to a distributed cluster.

### tambear-sip (standalone binary)

SIP ships without TBS/IDE/TAM runtime. One compiled pipeline, baked in, runs anywhere. The compiled kernels target whatever doors the SIP binary's deployment machine has, and the kernels run exactly the math SIP's header fields need, in the byte-exact layout SIP's NYXL/NYX format requires.

If SIP later wants to scale across distributed hardware, there are two options:
- Build SIP-specific orchestration (minimal TAM subset) for SIP's exact pipeline
- Switch SIP's authoring to use the by-then-complete tambear product

Both paths are OK. SIP's initial ship is standalone-binary, no TAM dependency.

## What this means for the 2026-04-11 plan

### Redlines

- **"CPU first / GPU later"** is wrong. Every math, every surface, parallel from first principles. Single-threaded CPU is a reference oracle for bit-identity verification, not a product surface.
- **"Step 5: parallel CPU tree-merge via rayon"** becomes part of step 3 (parallel from start) or step 3 becomes the reference oracle only.
- **Step 8 (GPU deterministic scatter)** stops being deferred; it's a first-class product surface alongside every other door.
- **wgpu** moves to an optional/maybe-later door. Direct vendor APIs (CUDA, Vulkan, DX12, Metal, AMD, Intel) are first-class.

### New sequencing (sketch, to be refined)

1. **Kulisch merge** ✅ done (step 1)
2. **Op determinism contract docstrings** ✅ done (step 2)
3. **CPU reference oracle via Kulisch** ✅ done (step 3 — but as oracle, not product)
4. **Parallel CPU Kulisch via rayon tree-merge** — first real surface
5. **Cross-vendor GPU Kulisch**:
   - Design: Kulisch compute shader with workgroup-local registers, cross-workgroup merge, exact `to_f64()` in shader or on host after readback
   - Render per door: CUDA kernel, Vulkan SPIR-V, DX12 DXIL, Metal shader, AMD/Intel direct
6. **TAM runtime skeleton** — hardware discovery, `.tam` plan parser, multi-surface dispatch, result collection, Kulisch merge across surfaces
7. **Spec compiler** — classification-bijection test, pass fusion, per-door kernel emission
8. **IDE + TBS integration** — GUI, project save/reopen, RUN trigger
9. **Library-wide determinism migration** — audit every `+=` and `.sum()` in recipes, replace with Kulisch helpers, add determinism tests per recipe
10. **Cross-surface bit-exact harness** — extend `determinism_contract.rs` to cycle through surfaces

### What stays valid from the 4/11 plan

- Kulisch as the default strategy for `Op::Add` ✅
- `using()` as the opt-out mechanism for strategy selection ✅
- `Op` enum does not grow with new variants ✅
- NaN/Inf policy with two independent knobs (both default propagate, skip is SIP's opt-in) ✅
- All-invalid emit asymmetry (Add → 0.0 preserves flatline, Max/Min → NaN) ✅
- The 6 resolved decisions from 4/16 + 4/17 refinements ✅

### What is genuinely new work

- TAM runtime (does not exist yet)
- Per-door kernel rendering (exists for CUDA; others need to be built)
- Pass fusion logic in the spec compiler (partial — EMA compiles to prefix scan; fusion needs more work)
- Multi-surface dispatch and merge
- IDE (does not exist yet)

This is a multi-month arc, not a single-session sprint. The architecture doc's job is to keep every session pointed at the same long-arc shape so individual implementation decisions don't drift.

## Pointers

- Fock Boundary Theorem: `campsites/theory/20260410-fock-boundary-theorem/navigator/20260410-theorem-state.md`
- Classification-bijection theory: `campsites/theory/20260410164337-classification-bijection/scout/insights/`
- Kingdom taxonomy (canonical): `campsites/theory/20260410164337-classification-bijection/scout/insights/kingdom-taxonomy-final.md`
- EMA-is-Kingdom-A spec: `campsites/industrialization/architecture/20260410164422-ema-is-kingdom-a/scout/insights/ema-kingdom-a-spec.md`
- Collatz structural verifier (worked Fock-raising example): `crates/tambear/src/collatz_structural.rs`, `collatz_parallel.rs`
- Op determinism plan (companion): `campsites/industrialization/architecture/2026-04-11-op-default-deterministic-plan.md`
- SIP column-graph (fusion example): `R:/ternyx-sip/docs/column-graph.md`
- Atoms/primitives/recipes reference: `docs/architecture/atoms-primitives-recipes.md`
