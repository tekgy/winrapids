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
