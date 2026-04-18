<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

# Current State — the baseline the team starts from

*Snapshot taken 2026-04-11 at the start of the Bit-Exact Trek. This is the code-and-tests reality the team inherits.*

## What's green

- **`crates/tambear-primitives`** — 98 lib tests + 9 GPU integration tests = **107 tests passing**.
- **26 recipes** in the flat catalog across 6 families (raw reductions, means, moments, norms, extrema, two-column). Full list in `crates/tambear-primitives/src/recipes/mod.rs`.
- **Fusion works.** The 26 recipes compile into 55 accumulate slots, which fuse to **4 kernel passes** (47 of 55 slots are in ONE `Primary/All/Add` loop).
- **`crates/tam-gpu`** — live `CudaBackend` running on this machine's RTX 6000 Pro Blackwell. Uses `cudarc`, dynamic-loads `nvcuda.dll`, targets `sm_120`. **Currently uses NVRTC to compile CUDA C source — this is the path we're replacing.**
- **`tests/gpu_end_to_end.rs`** in `tambear-primitives` — 9 tests that take recipes, fuse, codegen to CUDA C, compile via NVRTC, dispatch, and compare to CPU. Results match to ~1e-15 relative error (close to machine epsilon).

## What's in `codegen/cuda.rs` (the path being replaced)

`Expr → CUDA C string` + `AccumulatePass → kernel source` + 9 unit tests. **This code is the vendor-locked path.** It does not violate any principle *given what it is*, but it is NOT the path we're going to. It stays as a reference oracle during the transition — we'll diff our proper tam→PTX output against this NVRTC-reference output until we trust the real path.

**Team members should treat this code as read-only legacy.** Do not add features to it. Do not remove it yet. It dies with honor once Peak 6 lands.

## What's not committed

As of the start of this expedition, the working tree has uncommitted changes:
- `codegen/cuda.rs` creation
- `recipes/mod.rs` expansion (5 → 26 recipes)
- `tests/gpu_end_to_end.rs` creation
- Several files in `crates/tambear/src/` with unrelated adversarial fixes from prior sessions

**The team should commit these before starting Peak 1 work.** A clean baseline commit labeled something like `"Baseline for Bit-Exact Trek: 26 recipes, vendor-locked NVRTC path, 107 tests green"` is the right starting point. Ask Navigator if unsure.

## Where new code goes

Phase 1 introduces several new crates. Recommended layout:

```
crates/
├── tambear-primitives/         (existing — recipes, Expr, fused passes)
├── tam-gpu/                    (existing — hardware abstraction, CudaBackend)
├── tambear-tam-ir/             (NEW — .tam IR AST, parser, printer, verifier)
├── tambear-tam-cpu/            (NEW — CPU interpreter, eventually JIT)
├── tambear-tam-ptx/            (NEW — tam→PTX translator + raw driver loader)
├── tambear-tam-spirv/          (NEW — tam→SPIR-V translator, Peak 7)
├── tambear-libm/               (NEW — .tam IR source for transcendentals)
├── tambear-tam-test-harness/   (NEW — TamBackend trait, cross-backend diff, ULP harness)
```

Each new crate should have **zero deps except other tambear crates + `std`**. Exceptions:
- `tambear-tam-ptx` depends on `cudarc` (runtime driver only, not NVRTC)
- `tambear-tam-spirv` depends on `ash` and `rspirv` (both transparent typewriters)
- `tambear-libm` tests depend on `mpmath` (Python, via subprocess for reference generation)
- `tambear-tam-cpu` eventually depends on `cranelift` (JIT only, later phase)

**If a crate reaches for a new dep, that's an escalation to Navigator.** We don't silently grow the dep tree.

## Environment notes

- **OS**: Windows 11 Pro for Workstations. Some team members may not be on Windows — we test on Linux/Mac too. Portability is a feature.
- **GPU**: NVIDIA RTX 6000 Pro Blackwell (compute capability 12.0, sm_120). CUDA driver installed, no toolkit required (`cudarc` dynamic-loads `nvcuda.dll`).
- **Rust**: stable, edition 2021.
- **Python**: use `uv venv` + `uv pip` for mpmath reference generation. Never raw `python` / `pip`.
- **Shell**: Bash on Windows (Git Bash or similar). Use forward slashes and Unix-style paths in commands where possible.

## Known pitfalls already documented

See `../pitfalls/` (to be populated during the trek). As of baseline:
- The one-pass variance formula `(Σx² - (Σx)²/n) / (n-1)` is numerically unstable when the data has small variance relative to a large mean. Current tests don't exercise this. The adversary should hit it early.
- `atomicAdd` on fp64 is non-deterministic in reduction order. Current tests pass because the reductions are small, but this breaks determinism once we chase bit-exactness. Peak 6 addresses it.
- NVRTC inlines `__nv_sin`, `__nv_log`, `__nv_exp`, which are NOT the same implementations as glibc's. Current `Σ|ln x|` test shows ~2.8e-15 relative error, mostly from `__nv_log` diverging from `f64::ln`. This is exactly why tambear-libm has to exist.

## Navigator commitments

- I (Navigator) will be available to arbitrate escalations, review the IR spec before code starts, and pair with IR Architect on early design.
- I will review every peak's completion before the next peak starts. Frozen-peak rule is enforced by me.
- I will not micromanage campsites. Pick them up, work them, log them, commit them. I'm here for the moments that need me.


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

