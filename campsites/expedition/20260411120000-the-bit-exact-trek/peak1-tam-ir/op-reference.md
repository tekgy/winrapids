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
4. Integer arithmetic (i32)
5. Integer arithmetic (i64) — added for RFA and bitcasting
6. Float ↔ integer conversion — added for tam_exp/tam_ln and RFA
7. Floating-point comparisons
8. Integer comparisons
9. Select
10. Transcendental stubs (libm contract)
11. Reduction
12. Return
13. Structured control flow
14. Function and kernel declarations
15. Program header

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

## 4. Integer arithmetic (i32)

All i32 ops use signed 32-bit arithmetic with two's-complement wrap semantics.

### `const.i32`

```
%dst = const.i32 <decimal>
```

**Signature:** `() → i32`

**Semantics:** Signed 32-bit integer literal.

---

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

## 5. Integer arithmetic (i64)

Added to support Peak 6 RFA (reproducible floating-point accumulation) and exponent manipulation in `tam_ln`.

All i64 ops use signed 64-bit two's-complement arithmetic.

### `const.i64`

```
%dst = const.i64 <decimal>
```

**Signature:** `() → i64`

**Semantics:** Signed 64-bit integer literal.

---

### `iadd.i64`

```
%dst = iadd.i64 %a, %b
```

**Signature:** `(i64, i64) → i64`

**Semantics:** `%dst = %a + %b` (wrapping)

---

### `isub.i64`

```
%dst = isub.i64 %a, %b
```

**Signature:** `(i64, i64) → i64`

**Semantics:** `%dst = %a - %b` (wrapping)

---

### `and.i64`

```
%dst = and.i64 %a, %b
```

**Signature:** `(i64, i64) → i64`

**Semantics:** `%dst = %a & %b` (bitwise AND)

---

### `or.i64`

```
%dst = or.i64 %a, %b
```

**Signature:** `(i64, i64) → i64`

**Semantics:** `%dst = %a | %b` (bitwise OR). Requested by math-researcher for RFA exponent extraction.

---

### `xor.i64`

```
%dst = xor.i64 %a, %b
```

**Signature:** `(i64, i64) → i64`

**Semantics:** `%dst = %a ^ %b` (bitwise XOR)

---

### `shl.i64`

```
%dst = shl.i64 %a, %shift
```

**Signature:** `(i64, i32) → i64`

**Semantics:** `%dst = %a << %shift` (logical shift left, wrapping for shift ≥ 64)

---

### `shr.i64`

```
%dst = shr.i64 %a, %shift
```

**Signature:** `(i64, i32) → i64`

**Semantics:** `%dst = %a >> %shift` (arithmetic shift right, sign-extending)

---

## 6. Float ↔ integer conversion

Added to support `tam_exp` (range reduction) and `tam_ln` (exponent extraction), and Peak 6 RFA bin-index computation.

### `ldexp.f64`

```
%dst = ldexp.f64 %mantissa, %exp
```

**Signature:** `(f64, i32) → f64`

**Semantics:** `%dst = %mantissa * 2^%exp`. Correct IEEE 754 semantics including subnormal handling, overflow to ±infinity, underflow to ±0.0. Equivalent to C `ldexp`. Required by `tam_exp` for range reconstruction after polynomial evaluation.

**Invariants:** Does NOT call any vendor `ldexp`. CPU interpreter uses bit-manipulation on the exponent field. PTX backend emits `ex2.approx.f64` + scaling or equivalent correct form. SPIR-V backend uses `OpLdexp`.

**Tests:** Not yet tested directly; exercised via `tam_exp` when libm lands.

---

### `f64_to_i32_rn`

```
%dst = f64_to_i32_rn %a
```

**Signature:** `(f64) → i32`

**Semantics:** Convert f64 to i32 using round-to-nearest-even (banker's rounding). Saturates at `INT32_MAX` / `INT32_MIN` for out-of-range values. NaN → 0. Required by `tam_exp` to compute `n = round(x / ln2)` for argument reduction.

**Tests:** Not yet tested directly.

---

### `bitcast.f64.i64`

```
%dst = bitcast.f64.i64 %a
```

**Signature:** `(f64) → i64`

**Semantics:** Reinterpret the bit pattern of `%a` (IEEE 754 double) as a signed 64-bit integer. No value conversion. The sign bit of the resulting i64 is the sign bit of the f64. Used for exponent extraction in `tam_ln` and RFA bin-index computation.

**CPU:** `f64::to_bits() as i64`. **PTX:** `mov.b64`. **SPIR-V:** `OpBitcast`.

**Tests:** Not yet tested directly.

---

### `bitcast.i64.f64`

```
%dst = bitcast.i64.f64 %a
```

**Signature:** `(i64) → f64`

**Semantics:** Reinterpret the bit pattern of `%a` (i64) as an IEEE 754 double. Inverse of `bitcast.f64.i64`. Used to reconstruct f64 with a manipulated exponent field.

**CPU:** `f64::from_bits(v as u64)`. **PTX:** `mov.b64`. **SPIR-V:** `OpBitcast`.

**Tests:** Not yet tested directly. `bitcast.f64.i64 ∘ bitcast.i64.f64 = identity` is the round-trip invariant.

---

## 7. Floating-point comparisons

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

**Semantics:** `%dst = (%a == %b)` using IEEE-754 equality, not bitwise equality.

Key cases:
- `+0.0 == -0.0` → `true` (IEEE-754 §5.10: the two zeros are equal)
- `NaN == NaN` → `false` (IEEE-754 §5.11: any comparison with NaN is false)
- `NaN == x` → `false` for any `x`, including `NaN`

**Backend lowering:** PTX `setp.eq.f64`, SPIR-V `OpFOrdEqual`. Both implement IEEE-754 equality. Do **not** use `OpIEqual` (bitwise) — it would distinguish `+0` from `-0`.

**I11 note:** `fcmp_eq` always returns `false` for NaN inputs — it does not propagate NaN as a value. If you need to detect NaN, use explicit `is_nan` guards (see §5.5).

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

## §5.5 NaN propagation — invariant I11

**Added campsite I11 (2026-04-12), promoted from adversarial bug P17/P18/P20 +
independent convergence by naturalist + scout.**

### The invariant

> NaN propagates through every op on every backend.

Any op that takes a NaN input must produce NaN output. This applies to arithmetic,
comparisons, selects, min, max, clamp, and every other op in the IR.

### Why this is an invariant, not a choice

PTX `min.f64`, SPIR-V `OpFMin`, and CPU `f64::min` all handle NaN differently:

- PTX `min.f64 %r, %a, %b`: if either input is NaN, the result is the other input
  (NaN is treated as missing data, not a propagating signal).
- SPIR-V `OpFMin` (Vulkan): the behavior is undefined when either input is NaN.
- CPU `f64::min`: result is the other input (same as PTX).

Without explicit guards, `min(NaN, x) = x` on PTX and CPU, but may be
implementation-defined on SPIR-V. Two backends silently disagree. This is a
cross-backend bit-exactness failure that pointwise tests miss unless they inject NaN.

### The select trap (naturalist's finding)

The most common form of the bug is the select pattern used to implement min/max:

```
%pred = fcmp_gt.f64 %a, %b    ; NaN > x = false (IEEE 754)
%r    = select.f64  %pred, %a, %b
```

If `%a` is NaN: `fcmp_gt(NaN, x) = false` → `select(false, NaN, x) = x`.
The NaN is silently swallowed.

The fix in the CPU interpreter: `SelectF64` checks BOTH value operands for NaN
before evaluating the predicate. If either is NaN, the result is NaN.

### Per-op NaN behavior (Phase 1 complete op list)

| Op | NaN behavior |
|---|---|
| `const.f64 nan` | Produces NaN (the canonical quiet NaN bit pattern) |
| `fadd.f64 NaN, x` | NaN |
| `fsub.f64 NaN, x` | NaN |
| `fmul.f64 NaN, x` | NaN |
| `fdiv.f64 NaN, x` | NaN |
| `fneg.f64 NaN` | NaN (sign bit flips, payload preserved) |
| `fabs.f64 NaN` | NaN (sign bit cleared, payload preserved) |
| `fsqrt.f64 NaN` | NaN |
| `fcmp_gt.f64 NaN, x` | `false` (IEEE 754: all comparisons with NaN are false) |
| `fcmp_lt.f64 NaN, x` | `false` |
| `fcmp_eq.f64 NaN, x` | `false` (including NaN == NaN) |
| `select.f64 _, NaN, _` | NaN (I11 guard: if either value is NaN, result is NaN) |
| `select.f64 _, _, NaN` | NaN (I11 guard) |
| `select.f64 p, x, y` (x,y finite) | `x` if p else `y` (no NaN) |
| `reduce_block_add.f64` | NaN (fold propagates NaN naturally via addition) |
| `load.f64` | Passes through whatever bits are in the buffer |
| `store.f64` | Stores whatever bits (including NaN) |
| `bitcast.f64.i64` | Returns the bit pattern of the NaN (not NaN per se — an i64) |
| `bitcast.i64.f64` | Returns whatever f64 the bits represent (may be NaN) |
| `f64_to_i32_rn NaN` | 0 (saturation: NaN maps to 0, documented in §6) |
| `ldexp.f64 NaN, n` | NaN |
| `tam_exp NaN` | NaN (libm contract) |
| `tam_ln NaN` | NaN |
| `tam_sin NaN` | NaN |
| `tam_cos NaN` | NaN |
| `tam_pow NaN, x` | NaN |
| `tam_pow x, NaN` | NaN |

### Backend emit targets for NaN-safe min/max (Peak 3 / Peak 7)

When Peak 3 (PTX) and Peak 7 (SPIR-V) implement patterns that produce min/max
semantics, they must use NaN-propagating variants:

**PTX:**
- `min.NaN.f64 %r, %a, %b` — NaN-propagating min (PTX ISA §9.7.3, requires sm_80+)
- `max.NaN.f64 %r, %a, %b` — NaN-propagating max
- For older targets: `testp.notanumber.f64 %pred, %a; @%pred mov.f64 %r, %a` guard.

**SPIR-V / Vulkan:**
- `OpIsNan` + `OpSelect` guard: explicitly check both inputs before OpFMin/OpFMax.
- Or: use SPIR-V extension `SPV_INTEL_float_controls2` if available.

The CPU interpreter (Peak 5) implements the guard directly in `SelectF64`.

### Test coverage

Three tests added in campsite I11:
1. `i11_select_f64_nan_on_true_propagates` — select(false, NaN, 1.0) → NaN
2. `i11_select_f64_nan_on_false_propagates` — select(true, 1.0, NaN) → NaN
3. `i11_select_f64_comparison_with_nan_input_propagates` — abs(NaN) → NaN (the naturalist's bug)

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
| I11 | NaN propagates everywhere | SelectF64 guards in interp.rs; PTX min.NaN.f64 |
