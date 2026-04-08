# Accumulate Core — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01 (Phase 2)
**Files**: `src/accumulate.rs` (1266 lines), `src/compute_engine.rs`, `src/reduce_op.rs`

---

## A1 [HIGH]: CPU/CUDA behavioral divergence for all-NaN groups

**Location**: compute_engine.rs:270-276

**Bug**: When all values in a group are NaN:
- **CPU path**: NaN > sentinel is false, output stays at `f64::NEG_INFINITY` (max) or `f64::INFINITY` (min)
- **CUDA path**: `fmax(NaN, NaN)` returns NaN per IEEE 754

The contract for "what does max of all-NaN return?" is undefined. Cross-backend tests will fail.

**Test vector**:
```rust
accumulate(
    &[f64::NAN, f64::NAN],
    Grouping::ByKey { keys: &[0, 0], n_groups: 1 },
    Expr::Value,
    Op::Max
)
// CPU: Scalar(-inf)
// CUDA: Scalar(NaN)
```

**Fix**: Document the contract. If NaN→sentinel, add NaN check in CUDA kernel. If NaN→NaN, add NaN check in CPU path.

---

## A2 [HIGH]: Negative key values — out-of-bounds / wild GPU write

**Location**: compute_engine.rs:208, 229, 255, 271

**Bug**: `keys[i]` is `i32`. Negative values: `keys[i] as usize` wraps to huge positive → OOB access on CPU. On CUDA, `int g = keys[gid]` as negative → wild memory write (buffer overflow).

**Test vector**:
```rust
accumulate(
    &[1.0],
    Grouping::ByKey { keys: &[-1], n_groups: 1 },
    Expr::Value,
    Op::Add
)
// CPU: panics (index out of bounds)
// CUDA: undefined behavior (GPU buffer overflow)
```

**Fix**: Bounds check at entry: `assert!(keys.iter().all(|&k| k >= 0 && (k as usize) < n_groups))`. CUDA: `if (g < 0 || g >= n_groups) return;`.

---

## A3 [MEDIUM]: Three independent CUDA contexts

**Location**: accumulate.rs:185-191

`AccumulateEngine::new()` calls `tam_gpu::detect()` three times — three CUDA contexts, ~300MB VRAM overhead, no buffer sharing between sub-engines.

**Fix**: Share one `Arc<dyn TamGpu>`.

---

## A4 [MEDIUM]: Softmax — single NaN contaminates all outputs

**Location**: accumulate.rs:423-440

A NaN in input produces `exp(NaN - max_val) = NaN`, which contaminates `sum_exp` via additive scatter. All outputs become `NaN / NaN = NaN`.

**Test**: `softmax(&[1.0, f64::NAN, 2.0])` → `[NaN, NaN, NaN]`.

**Fix**: Either filter NaN (output 0.0 at NaN positions) or document NaN-free input requirement.

---

## A5 [MEDIUM]: n as i32 overflow for >2^31 elements

**Location**: reduce_op.rs:77

`n as i32` wraps for n > 2,147,483,647. CUDA kernel sees negative `n`, most threads skip, results are wrong.

**Fix**: Use `i64` or bounds check.

---

## A6 [MEDIUM]: n as u32 overflow in grid dimensions

**Location**: compute_engine.rs:329, 379, 424, 465, 496, 529

`n as u32` wraps for n > 4,294,967,295. Grid too small → partial computation → silently wrong results.

**Fix**: Use `u64` arithmetic.

---

## A7 [LOW]: Empty input sentinel values undocumented

Max of empty → `-∞`, ArgMin of empty → `(∞, usize::MAX)`. Defensible but implicit.

## A8 [LOW]: Kernel cache grows unboundedly

Cache key includes `n` (array size). Every new size compiles a new kernel (~40ms). No eviction.

---

## What's NOT broken (positive findings)

- **No naive variance** — accumulate provides only sum/sum_sq/count primitives; formula is caller's responsibility
- **No partial_cmp().unwrap()** — zero NaN panic instances
- **No hardcoded thresholds** — only test code has constants
- **Softmax is numerically stable** — log-sum-exp trick correctly applied
- **Expression dispatch is complete** — all (Grouping, Op) combinations either dispatch or return explicit errors
- **Float associativity acknowledged** — atomic adds on GPU are inherently non-deterministic; this is architecture, not a bug
