# `.tam` IR — Op Reference (Phase 1)

**Campsite 1.15 — binding contract for every backend.**

Every op listed here is specified by:
- **Syntax** — the canonical text representation as emitted by `print.rs`
- **Signature** — types of inputs and outputs
- **Semantics** — what the op computes; invariants that every backend must preserve
- **Test** — which test(s) exercise this op

Backends: CPU interpreter (`interp.rs`), PTX assembler (Peak 3), SPIR-V assembler (Peak 7).

---

## Table of Contents

1. Constants
2. Buffer ops
3. Floating-point arithmetic
4. Integer arithmetic
5. Floating-point comparisons
6. Integer comparisons
7. Select
8. Transcendental stubs (libm contract)
9. Reduction
10. Return
11. Structured control flow
12. Function and kernel declarations
13. Program header

---

## 1. Constants

### `const.f64`

```
%dst = const.f64 <hex-or-decimal>
```

**Signature:** `() → f64`

**Semantics:** Define register `%dst` with the literal f64 value. The printer always emits the bit-exact hex form (`0x<16 lowercase hex digits>`), guaranteeing round-trip equality for all bit patterns including -0.0, NaN, infinities, and subnormals. The parser also accepts decimal form and the keywords `inf`, `-inf`, `nan`.

**Invariants:**
- Every f64 bit pattern is representable and round-trips through print→parse.
- The value is a compile-time constant; it does not depend on runtime state.

**Tests:** `print_const_f64_uses_hex`, `print_f64_hex_zero`, `print_f64_hex_neg_zero`, `print_f64_hex_inf`, `print_f64_hex_neg_inf`, `parse_const_f64_hex`, `parse_const_f64_decimal`, `op_const_f64_nan_identity`, `op_const_f64_partialeq`

---

### `const.i32`

```
%dst = const.i32 <decimal-i32>
```

**Signature:** `() → i32`

**Semantics:** Define register `%dst` with the signed 32-bit integer literal. Printed as a signed decimal integer (e.g. `0`, `-1`, `42`).

**Tests:** all `reduce_block_add` tests (slot index is always a `const.i32`)

---

## 2. Buffer ops

### `bufsize`

```
%dst = bufsize %buf
```

**Signature:** `(buf<f64>) → i32`

**Semantics:** Returns the number of elements in `%buf` as a signed 32-bit integer. The maximum representable size is 2^31 − 1 = 2,147,483,647. Programs that require buffers larger than 2B elements must use a future `bufsize.i64` op (not in Phase 1).

**Invariants:**
- The return type is `i32`, not `i64`. This is an intentional Phase 1 constraint.
- Every kernel begins with `%n = bufsize %data0` by convention (not required by the IR).

**Tests:** All interpreter tests that run a kernel exercise this op.

---

### `load.f64`

```
%dst = load.f64 %buf, %idx
```

**Signature:** `(buf<f64>, i32) → f64`

**Semantics:** Load the f64 at position `%idx` in `%buf`. The index is zero-based. Out-of-bounds access is undefined behavior (the interpreter panics; GPU backends may produce an arbitrary value or fault).

**Tests:** All loop-based interpreter tests exercise this op.

---

### `store.f64`

```
store.f64 %buf, %idx, %val
```

**Signature:** `(buf<f64>, i32, f64) → ()`

**Semantics:** Write `%val` to position `%idx` in `%buf`. No return value. Out-of-bounds access is undefined behavior.

**Tests:** Not directly exercised in Phase 1 reference programs (reduction uses `reduce_block_add` instead of explicit stores), but the op is in the AST and the interpreter handles it.

---

## 3. Floating-point arithmetic

All f64 arithmetic ops observe IEEE 754-2008 round-to-nearest-even semantics. Invariants I3 and I4 apply:

- **I3 (no FMA):** There is no `fma` op. Backends must not silently fuse `fadd(fmul(a,b),c)` into a hardware FMA. If a backend generates code that might contract, it must emit the IEEE-compliant non-contracting form.
- **I4 (no reordering):** Ops execute in program order. No backend may reorder floating-point operations.

### `fadd.f64`

```
%dst = fadd.f64 %a, %b
```

**Signature:** `(f64, f64) → f64`

**Semantics:** `%dst = %a + %b` (IEEE 754 addition, round-to-nearest-even)

**Tests:** `sum_all_add_1_to_10`, `roundtrip_10000_random_programs`

---

### `fsub.f64`

```
%dst = fsub.f64 %a, %b
```

**Signature:** `(f64, f64) → f64`

**Semantics:** `%dst = %a - %b` (IEEE 754 subtraction)

**Tests:** `roundtrip_10000_random_programs`

---

### `fmul.f64`

```
%dst = fmul.f64 %a, %b
```

**Signature:** `(f64, f64) → f64`

**Semantics:** `%dst = %a * %b` (IEEE 754 multiplication, no FMA — I3)

**Tests:** `variance_pass_matches_expected`, `roundtrip_10000_random_programs`

---

### `fdiv.f64`

```
%dst = fdiv.f64 %a, %b
```

**Signature:** `(f64, f64) → f64`

**Semantics:** `%dst = %a / %b` (IEEE 754 division). Division by zero follows IEEE 754: `x/0 = ±inf`, `0/0 = NaN`.

**Tests:** Not directly used in Phase 1 reference programs; covered in proptest round-trips (generator does not emit fdiv, but the AST and interpreter support it).

---

### `fsqrt.f64`

```
%dst = fsqrt.f64 %a
```

**Signature:** `(f64) → f64`

**Semantics:** `%dst = √%a` (IEEE 754 square root, correctly rounded). `fsqrt(-x)` for x > 0 produces NaN. `fsqrt(-0.0) = -0.0`.

**Tests:** `fsqrt_correct` (fsqrt(4.0) = 2.0, fsqrt(2.0) ≈ 1.4142135623730951)

---

### `fneg.f64`

```
%dst = fneg.f64 %a
```

**Signature:** `(f64) → f64`

**Semantics:** `%dst = -%a` (IEEE 754 negation; flips the sign bit only, no rounding).

**Tests:** `roundtrip_10000_random_programs`

---

### `fabs.f64`

```
%dst = fabs.f64 %a
```

**Signature:** `(f64) → f64`

**Semantics:** `%dst = |%a|` (IEEE 754 absolute value; clears the sign bit). `fabs(NaN)` preserves the NaN payload and returns a positive NaN.

**Tests:** AST and interpreter support; not in Phase 1 reference programs.

---

## 4. Integer arithmetic

All integer ops use signed 32-bit (i32) arithmetic with two's-complement wrap semantics.

### `iadd.i32`

```
%dst = iadd.i32 %a, %b
```

**Signature:** `(i32, i32) → i32`

**Semantics:** `%dst = %a + %b` (wrapping addition)

---

### `isub.i32`

```
%dst = isub.i32 %a, %b
```

**Signature:** `(i32, i32) → i32`

**Semantics:** `%dst = %a - %b` (wrapping subtraction)

---

### `imul.i32`

```
%dst = imul.i32 %a, %b
```

**Signature:** `(i32, i32) → i32`

**Semantics:** `%dst = %a * %b` (wrapping multiplication)

---

## 5. Floating-point comparisons

All comparison ops produce a `pred` (boolean predicate) register.

### `fcmp_gt.f64`

```
%dst = fcmp_gt.f64 %a, %b
```

**Signature:** `(f64, f64) → pred`

**Semantics:** `%dst = (%a > %b)`. NaN comparisons follow IEEE 754 (NaN > x is false for all x).

---

### `fcmp_lt.f64`

```
%dst = fcmp_lt.f64 %a, %b
```

**Signature:** `(f64, f64) → pred`

**Semantics:** `%dst = (%a < %b)`. NaN comparisons: NaN < x is false.

---

### `fcmp_eq.f64`

```
%dst = fcmp_eq.f64 %a, %b
```

**Signature:** `(f64, f64) → pred`

**Semantics:** `%dst = (%a == %b)`. NaN == anything is false (including NaN == NaN).

---

## 6. Integer comparisons

### `icmp_lt`

```
%dst = icmp_lt %a, %b
```

**Signature:** `(i32, i32) → pred`

**Semantics:** `%dst = (%a < %b)` (signed 32-bit comparison)

---

## 7. Select

### `select.f64`

```
%dst = select.f64 %pred, %on_true, %on_false
```

**Signature:** `(pred, f64, f64) → f64`

**Semantics:** `%dst = %pred ? %on_true : %on_false`. Both branches are evaluated before selection (no short-circuit). This is a ternary select, not a branch.

**Tests:** `select_f64_branch_free`

---

### `select.i32`

```
%dst = select.i32 %pred, %on_true, %on_false
```

**Signature:** `(pred, i32, i32) → i32`

**Semantics:** `%dst = %pred ? %on_true : %on_false`. Both branches evaluated. No short-circuit.

---

## 8. Transcendental stubs (libm contract)

These ops are defined in Phase 1 as structural placeholders. The CPU interpreter panics when they are reached. Campsite 5.2 wires in `tambear-libm` implementations.

Every backend must implement these ops identically to the `tambear-libm` Phase 1 algorithms. No backend may substitute `glibc exp`, `__nv_exp`, `std::f64::exp`, or any other vendor implementation. Invariants I1 and I8 apply.

### `tam_exp.f64`

```
%dst = tam_exp.f64 %a
```

**Signature:** `(f64) → f64`

**Semantics:** `%dst = eˡᵃ` — base-e exponential. Defined by `tambear-libm::exp_f64`, which is first-principles, mpmath-oracled, phase-1 algorithm.

**Contract for libm:** Must match mpmath `exp(a)` to within ≤1 ULP for all finite inputs. Must return `+inf` for overflow, `0.0` for underflow, `1.0` for input `0.0`, handle NaN transparently (NaN → NaN).

---

### `tam_ln.f64`

```
%dst = tam_ln.f64 %a
```

**Signature:** `(f64) → f64`

**Semantics:** `%dst = ln(%a)` — natural logarithm. `ln(0) = -inf`, `ln(x<0) = NaN`, `ln(NaN) = NaN`.

---

### `tam_sin.f64`

```
%dst = tam_sin.f64 %a
```

**Signature:** `(f64) → f64`

**Semantics:** `%dst = sin(%a)` in radians. Full-period argument reduction required for large inputs.

---

### `tam_cos.f64`

```
%dst = tam_cos.f64 %a
```

**Signature:** `(f64) → f64`

**Semantics:** `%dst = cos(%a)` in radians. Same argument reduction requirement as `tam_sin`.

---

### `tam_pow.f64`

```
%dst = tam_pow.f64 %a, %b
```

**Signature:** `(f64, f64) → f64`

**Semantics:** `%dst = %aˡᵇ`. Edge cases follow IEEE 754: `pow(1, NaN) = 1`, `pow(NaN, 0) = 1`, `pow(x, ±0) = 1` for all x, `pow(0, y<0) = +inf`, etc.

---

## 9. Reduction

### `reduce_block_add.f64`

```
reduce_block_add.f64 %out_buf, %slot_idx, %val
```

**Signature:** `(buf<f64>, i32, f64) → ()`

**Semantics (CPU):** `out_buf[slot_idx] = val`. On CPU there is exactly one "block" containing all elements, so the partial reduction is the total reduction. No host-side fold needed.

**Semantics (GPU):** Perform a shared-memory parallel tree reduction within the thread block. Block 0 writes its partial sum to `out_buf[slot_idx]`. All other blocks write to `out_buf[slot_idx + n_blocks * slot_count]` or equivalent layout (backend-specific). The host folds all partials after kernel completion.

**Invariants (I5, I6):**
- **I5 (no non-deterministic reductions):** GPU backends must use a fixed-order tree reduction, not `atomicAdd`. The partial sum written by each block is deterministic regardless of scheduling.
- **I6 (no silent fallback):** A backend must not fall back from tree reduction to atomicAdd for performance reasons without explicitly declaring the loss of determinism.

**Tests:** All multi-accumulator tests: `variance_pass_matches_expected`, `sum_all_add_1_to_10`

---

## 10. Return

### `ret.f64`

```
ret.f64 %val
```

**Signature:** `(f64) → ()`

**Semantics:** Return `%val` from a function body. Must appear exactly once, as the last op in every `func` body. Illegal inside a `kernel` body (verifier enforces this).

**Tests:** `verify_catches_ret_in_kernel`, `verify_catches_missing_ret_in_func`

---

## 11. Structured control flow

### `loop_grid_stride`

```
loop_grid_stride %i in [0, %n) {
  <ops>
}
```

**Signature:** `(i32 limit) → ()`

**Semantics (CPU):** Serial loop. `%i` takes values `0, 1, 2, ..., %n - 1`. The induction variable `%i` is read-only inside the loop body.

**Semantics (GPU):** Grid-stride loop. Each thread starts at `blockIdx.x * blockDim.x + threadIdx.x` and steps by `gridDim.x * blockDim.x`. Threads with starting index `≥ %n` execute zero iterations.

**Loop-carried values (phi convention):** A register `%acc` used inside the loop that needs to carry its updated value back to the next iteration uses the prime-suffix convention:
- `%acc` — the value entering the loop iteration (defined outside the loop)
- `%acc'` — the updated value exiting the loop iteration (defined inside the loop body)

After each loop iteration, `%acc'` becomes `%acc` for the next iteration. After the loop exits, `%acc'` holds the final accumulated value.

This is equivalent to LLVM phi nodes but without explicit phi instructions. The convention is: if a register is defined both outside the loop (as `%acc`) and inside the loop (as `%acc'`), it is a loop-carried value.

**Phase 1 constraint:** At most one `loop_grid_stride` per kernel (verifier enforces this). Multiple loops are a Phase 2 feature.

**Tests:** `sum_all_add_1_to_10`, `variance_pass_matches_expected`, `roundtrip_10000_random_programs`

---

## 12. Function and kernel declarations

### `func`

```
func <name>(<params>) -> f64 {
entry:
  <ops>
}
```

Where `<params>` is a comma-separated list of `f64 %reg` pairs.

A `func` contains a flat sequence of ops followed by exactly one `ret.f64`. No loops. No buffers. Used for libm implementations and helper functions.

### `kernel`

```
kernel <name>(<params>) {
entry:
  <ops including at most one loop>
}
```

Where `<params>` is a comma-separated list of `<ty> %reg` pairs. Types: `buf<f64>`, `i32`, `i64`, `f64`, `pred`.

A `kernel` may contain one `loop_grid_stride` block. No `ret.f64` allowed. Kernels accumulate into output buffers via `reduce_block_add` or `store.f64`.

---

## 13. Program header

Every `.tam` file begins with two header lines:

```
.tam <major>.<minor>
.target <platform>
```

**`.tam` version:** Phase 1 is `.tam 0.1`. The version identifies the IR revision, not the compiler version.

**`.target` platforms:** `cross` (CPU-and-GPU agnostic), or backend-specific strings. Phase 1 uses `cross` for all reference programs.

**Ordering:** Functions appear before kernels. Within each group, order is preserved by the printer and required by the parser.

---

## Appendix: Type system summary

| Type | Register suffix convention | Description |
|------|---------------------------|-------------|
| `f64` | `%v`, `%acc`, `%r` | IEEE 754 double-precision float |
| `i32` | `%n`, `%i`, `%s0` | Signed 32-bit integer |
| `i64` | (reserved Phase 2) | Signed 64-bit integer |
| `pred` | `%p`, `%cond` | Boolean predicate (comparison result) |
| `buf<f64>` | `%data`, `%out` | Reference to an f64 buffer; not a value |

`buf<f64>` registers are not scalar values — they are names for external buffers passed as kernel parameters. They cannot appear as operands to arithmetic ops; only `bufsize`, `load.f64`, `store.f64`, and `reduce_block_add.f64` operate on them.

---

## Appendix: Invariants quick reference

| ID | Name | Enforced by |
|----|------|-------------|
| I1 | No vendor math | AST has no direct calls; libm stubs panic |
| I2 | No vendor compiler | Not in this crate (TAM responsibility) |
| I3 | No FMA contraction | No `fma` op; backends must not fuse |
| I4 | No fp reordering | Interpreter executes in program order |
| I5 | No non-deterministic reductions | GPU must use tree reduce, not atomicAdd |
| I6 | No silent fallback | Transcendentals panic until libm wired |
| I7 | Accumulate + gather | Every kernel: loop (acc) + reduce (gather) |
| I8 | First-principles transcendentals | Stubs declared; libm fills them at 5.2 |
| I9 | mpmath oracle | Tests must match mpmath at 50+ digits |
| I10 | Cross-backend diff | Replay harness (Peak 4) enforces this |
