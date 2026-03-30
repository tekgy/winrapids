# sizeof Validation: Closing the Rust/CUDA Type Boundary

*Naturalist journal — 2026-03-30*

---

## The vulnerability

The scan engine allocates shared memory based on `op.state_byte_size()` — a number returned by Rust code, calculated by hand:

```rust
// WelfordOp
fn state_byte_size(&self) -> usize { 24 } // i64(8) + f64(8) + f64(8)

// EWMOp
fn state_byte_size(&self) -> usize { 24 } // f64(8) + f64(8) + i64(8)
```

Meanwhile, CUDA has its own `sizeof(state_t)` for the same struct, determined by the CUDA compiler's layout rules (alignment, padding between fields, tail padding).

If these disagree, `shared_bytes = BLOCK_SIZE * state_byte_size` is wrong. The Blelloch scan reads and writes at `shared[tid]` — indexing by `state_t` stride. Wrong stride = overlapping reads = silent memory corruption. Not a crash. Not a wrong answer you can detect. Corrupted intermediate state that produces plausible-looking but incorrect results.

This is the most dangerous class of bug: it can't be caught by looking at outputs (the scan still produces numbers), only by comparing against a known-good reference (which is what the correctness tests do — but they might pass anyway if the corruption happens to cancel out for small inputs).

## The fix — already implemented

Two changes, already in the codebase:

### 1. engine.rs: `query_sizeof` kernel

Appended to the multiblock scan CUDA source, after `propagate_extract`:

```cuda
// sizeof validation: returns sizeof(state_t) so Rust can verify
// that its state_byte_size() matches the actual CUDA struct layout.
extern "C" __global__ void query_sizeof(int* __restrict__ result) {
    result[0] = (int)sizeof(state_t);
}
```

This kernel lives in the SAME compilation unit as the scan kernels. It sees the SAME `state_t` typedef. It queries the SAME struct the scan will use. If there's a layout difference between what NVRTC compiles and what Rust expects, this catches it.

### 2. launch.rs: validation in `ensure_module()`

After loading the three scan functions and BEFORE inserting the module into the cache:

```rust
// sizeof validation: query CUDA's sizeof(state_t) and compare
// against Rust's state_byte_size(). One-time check per operator.
let f_sizeof = module.load_function("query_sizeof")?;
let mut sizeof_buf: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
unsafe {
    self.stream.launch_builder(&f_sizeof)
        .arg(&mut sizeof_buf)
        .launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        })?;
}
self.stream.synchronize()?;
let sizeof_result = self.stream.clone_dtoh(&sizeof_buf)?;
let cuda_sizeof = sizeof_result[0] as usize;
let rust_sizeof = op.state_byte_size();
assert_eq!(
    cuda_sizeof, rust_sizeof,
    "sizeof mismatch for operator '{}': CUDA sizeof(state_t) = {} bytes, \
     Rust state_byte_size() = {} bytes. This would cause silent memory \
     corruption in shared memory scans.",
    op.name(), cuda_sizeof, rust_sizeof
);
```

**Execution cost**: one kernel launch with 1 thread, one i32 device alloc, one synchronize, one device-to-host copy. Total: ~10-50μs. Runs once per operator per engine lifetime (gated by `self.modules.contains_key()`). Negligible.

**Failure mode**: `assert_eq!` panics with a clear message naming the operator, both sizes, and the consequence. The panic happens BEFORE any data touches the scan kernel. No computation runs with a corrupted layout.

## Analysis of current operators

### Scalar ops (AddOp, MulOp, MaxOp, MinOp)

```cuda
typedef double state_t;   // sizeof = 8
```
Rust: `state_byte_size() = 8` (default)

**Risk: none.** `double` is 8 bytes everywhere. No struct, no padding, no alignment issue.

### WelfordOp

```cuda
struct WelfordState { long long count; double mean; double m2; };
typedef WelfordState state_t;   // sizeof = ?
```

Layout analysis:
- `long long count`: 8 bytes at offset 0
- `double mean`: 8 bytes at offset 8 (naturally aligned, no padding needed)
- `double m2`: 8 bytes at offset 16
- Total: 24 bytes, no tail padding (already 8-byte aligned)

Rust: `state_byte_size() = 24`

**Risk: none for current layout.** All fields are 8-byte types. No padding is needed between any adjacent pair. The struct is naturally 8-byte aligned and 24 bytes total.

**Where this WOULD break**: if someone reordered to `{ double mean; long long count; double m2; }` — same fields, same sizeof (24, still no padding needed since all are 8-byte). But if a future operator used mixed sizes like `{ int flags; double value; }` — the CUDA compiler inserts 4 bytes of padding after `flags` for alignment, making sizeof = 16. A hand-calculated `state_byte_size() = 12` would be wrong.

### EWMOp

```cuda
struct EWMState { double weight; double value; long long count; };
typedef EWMState state_t;   // sizeof = ?
```

Layout analysis:
- `double weight`: 8 bytes at offset 0
- `double value`: 8 bytes at offset 8
- `long long count`: 8 bytes at offset 16
- Total: 24 bytes, no tail padding

Rust: `state_byte_size() = 24`

**Risk: none for current layout.** Same reasoning as WelfordOp.

## When this validation saves us

The validation becomes load-bearing when:

1. **KalmanOp ships** — `struct KalmanState { double a; double b; double c; double eta; double j; }` — 5 doubles, 40 bytes. Should be safe (all same type), but the validation confirms it automatically.

2. **Mixed-type structs appear** — e.g., a quantized operator with `{ float value; int count; short flags; }`. The CUDA compiler will pad `flags` to align the struct to 4 bytes (or 8 with certain alignment pragmas). Hand-calculating this is error-prone. The validation catches it.

3. **Platform differences** — NVRTC on different architectures (sm_120 Blackwell vs sm_89 Ada) might lay out structs differently in theory (in practice, CUDA ABI is stable across architectures, but the validation costs nothing).

4. **Refactoring** — someone reorders struct fields for readability, breaking the assumed layout. The validation catches it on first use, not after weeks of corrupted results.

## What the validation does NOT catch

- **Semantic mismatches**: if `cuda_combine_body()` reads `a.count` but the struct has `count` at a different semantic position than Rust expects. The sizeof matches but the field interpretation is wrong.
- **Endianness**: CUDA and Rust on the same machine share endianness, so this isn't an issue in practice.
- **Dynamic state**: if the state size depended on runtime values (it doesn't — all current operators have fixed state).

The sizeof check is necessary but not sufficient. The correctness tests (cumsum vs NumPy, Welford vs NumPy mean/var) are the semantic validation. sizeof catches the structural class of bugs; correctness tests catch the algebraic class.

## Summary

| Operator | CUDA sizeof (expected) | Rust state_byte_size() | Risk | Padding? |
|---|---|---|---|---|
| AddOp | 8 | 8 | None | No struct |
| MulOp | 8 | 8 | None | No struct |
| MaxOp | 8 | 8 | None | No struct |
| MinOp | 8 | 8 | None | No struct |
| WelfordOp | 24 | 24 | None | All 8-byte fields |
| EWMOp | 24 | 24 | None | All 8-byte fields |

All current operators are safe. The validation exists for future operators — particularly KalmanOp and any mixed-type state structs that arrive as the operator vocabulary grows.

---

*The type boundary between Rust and CUDA is an assertion, not a proof. The sizeof kernel makes it a runtime proof — checked once, at module load, before any data is at risk. The cost is 10μs per operator lifetime. The alternative is silent memory corruption. This is the cheapest insurance in the system.*
