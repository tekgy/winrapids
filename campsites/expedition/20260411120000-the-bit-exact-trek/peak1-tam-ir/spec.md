# `.tam` IR Specification — Phase 1

**Version:** 0.1  
**Status:** Draft — awaiting review (see `spec-review.md`)  
**Author:** pathmaker  
**Date:** 2026-04-11

---

## 1. What `.tam` is

`.tam` is a typed, register-based intermediate representation for mathematical
kernels that compute over flat fp64 buffers. It has one purpose: express the
same math once, in one format, and let every backend (CPU interpreter, PTX
assembler, SPIR-V assembler) lower it to their own instruction stream.

It is NOT a general-purpose language. It is NOT LLVM IR. It is NOT MLIR. It
does not have structs, enums, closures, heap allocation, or strings. If it is
not needed for a recipe we actually have, it does not exist.

**Scope for Phase 1:** accumulations on ℝⁿ → ℝᵏ. One input pattern:
iterate over a flat buffer, accumulate into slots, write partials to an output
buffer. Higher-rank tensors, matrix math, and shaped arrays are out of scope.

---

## 2. Module structure

Every `.tam` file is a **module**. A module contains zero or more function
definitions and zero or more kernel definitions. Functions are used by libm;
kernels are the compute entry points.

```
.tam 0.1
.target cross

<function-def>*
<kernel-def>*
```

`.target cross` means "this module is hardware-agnostic." Backend-specific
modules may declare `.target ptx` or `.target spirv` in future phases, but
Phase 1 only emits `.target cross`.

---

## 3. Type system

Phase 1 has exactly five types. No vectors, no tuples, no pointers, no shapes.

| Type      | Width    | Meaning                                         |
|-----------|----------|-------------------------------------------------|
| `i32`     | 32 bits  | Signed integer. Used for indices and counts.    |
| `i64`     | 64 bits  | Signed integer. For large-N indices.            |
| `f64`     | 64 bits  | IEEE 754 binary64. The math type.               |
| `pred`    | 1 bit    | Boolean predicate. Result of comparisons only.  |
| `buf<f64>`| —        | A flat, length-bearing buffer of f64 values.    |

`buf<f64>` is a parameter type only — it cannot appear as a register type. You
cannot store a `buf<f64>` in a register; you can only load from one or store
into one, and only by index.

`pred` is a result type only — it cannot be an input to an op that expects
`f64` or `i32`. The only consumers of `pred` are `select`.

---

## 4. Register file (SSA)

Registers are written with a `%` prefix: `%r0`, `%acc`, `%v`, `%n`. Names
are alphanumeric plus `_`. Register names can have prime suffixes for
loop-carried phi values: `%acc'` means "the updated value of `%acc` that exits
the loop body."

**SSA invariant:** every register is defined exactly once in the static text.
The only exception is phi nodes in loops, handled by the prime-suffix convention
(see §7).

**Type annotation:** register types are always inferred from the defining op.
You never write `f64 %r` — the type is determined by the op that creates `%r`.

---

## 5. Ops — the complete Phase 1 set

### 5.1 Constants

```
const.f64  %dst = <f64-literal>
const.i32  %dst = <i32-literal>
```

Literal syntax: standard Rust-style floating-point (`1.0`, `-0.5`, `3.14e7`,
`inf`, `-inf`, `nan`) or hex bit-literal (`0x3ff0000000000000` for 1.0).
Integer literals are decimal only.

### 5.2 Buffer ops

```
bufsize    %dst:i32 = %buf          ; length of buf (as i32)
load.f64   %dst     = %buf, %idx   ; %idx must be i32
store.f64  %buf, %idx, %val        ; %idx:i32, %val:f64; no %dst
```

`bufsize` yields the number of elements. Index arithmetic is the caller's
responsibility. Out-of-bounds access is undefined behavior — the verifier does
not check it.

### 5.3 Floating-point arithmetic

All fp ops are non-contracting (I3). FMA is never emitted unless explicitly
requested by a future `fma.f64` op (which does not exist in Phase 1). Every
backend must honor this — PTX emits `.contract false` on every fp instruction.

```
fadd.f64  %dst = %a, %b
fsub.f64  %dst = %a, %b
fmul.f64  %dst = %a, %b
fdiv.f64  %dst = %a, %b
fsqrt.f64 %dst = %a
fneg.f64  %dst = %a
fabs.f64  %dst = %a
```

All operations follow IEEE 754 round-to-nearest-even (ties-to-even). No
flush-to-zero, no denormal suppression, no fast-math. These are not
preferences — violating them violates invariants I3 and I4.

### 5.4 Integer arithmetic

```
iadd.i32  %dst = %a, %b
isub.i32  %dst = %a, %b
imul.i32  %dst = %a, %b
icmp_lt   %dst:pred = %a:i32, %b:i32
```

No `idiv` or `imod` in Phase 1. Not needed for any current recipe. If a future
recipe requires them, add them then.

### 5.4a i64 integer arithmetic

Needed for bitcast-based exponent manipulation in libm and RFA. No i64 division
in Phase 1.

```
const.i64   %dst:i64 = <decimal>
iadd.i64    %dst:i64 = %a:i64, %b:i64
isub.i64    %dst:i64 = %a:i64, %b:i64
and.i64     %dst:i64 = %a:i64, %b:i64
or.i64      %dst:i64 = %a:i64, %b:i64
xor.i64     %dst:i64 = %a:i64, %b:i64
shl.i64     %dst:i64 = %a:i64, %shift:i32   ; logical shift left
shr.i64     %dst:i64 = %a:i64, %shift:i32   ; arithmetic shift right (sign-preserving)
```

### 5.4b Float ↔ integer conversion and reinterpretation

These are pure bit-level operations. I1 and I8 do not apply — they are not
transcendentals. Every backend lowers them to a register alias or a single
instruction (PTX `mov.b64`, SPIR-V `OpBitcast`, CPU `f64::to_bits` /
`f64::from_bits`).

```
bitcast.f64.i64  %dst:i64 = %a:f64    ; reinterpret f64 bits as i64 (no conversion)
bitcast.i64.f64  %dst:f64 = %a:i64    ; reinterpret i64 bits as f64 (no conversion)
f64_to_i32_rn    %dst:i32 = %a:f64    ; f64→i32 round-to-nearest-even, saturate at INT32 bounds
ldexp.f64        %dst:f64 = %m:f64, %n:i32   ; %m * 2^%n, correct IEEE boundary handling
```

**`f64_to_i32_rn` semantics:** Round-to-nearest-even (banker's rounding). Matches
PTX `cvt.rni.s32.f64`. Saturates at INT32_MIN / INT32_MAX for out-of-range inputs.
For NaN input, result is 0.

**`ldexp.f64` semantics:** `%dst = %m * 2^%n`. Handles subnormals, overflow to
infinity, and underflow to zero per IEEE 754. Equivalent to C's `ldexp(m, n)`.
On PTX: implemented via exponent-field integer add in the bit representation —
NOT via any vendor libm call. The CPU interpreter uses bit manipulation; the PTX
backend does likewise.

**Design note:** `ldexp.f64` could alternatively be composed from `bitcast.f64.i64`,
integer add on the exponent field, and `bitcast.i64.f64`. That composition is
valid and more explicit, but `ldexp.f64` is retained as a first-class op because
it has non-trivial boundary behavior (subnormal inputs, exponent underflow/overflow)
that is easier to specify and test at one named location than distributed across
a bitcast + integer sequence.

### 5.5 Floating-point comparisons

Comparisons produce `pred`-typed registers.

```
fcmp_gt.f64  %dst:pred = %a, %b
fcmp_lt.f64  %dst:pred = %a, %b
fcmp_eq.f64  %dst:pred = %a, %b
```

NaN behavior follows IEEE 754: all comparisons with NaN produce `false`
(except `fcmp_eq` with two NaN operands, which also produces `false`).

**NaN propagation through select:** Because comparisons produce `pred = false`
for any NaN input, a `select.f64` downstream of a NaN-producing comparison will
always select the `on_false` branch — it does NOT propagate NaN. This is
argument-order-dependent: `fcmp_gt.f64 NaN, 5.0` and `fcmp_gt.f64 5.0, NaN`
both produce `false`, but reversing the `select` arguments changes which branch
is taken. Programs that need NaN propagation through conditionals must check for
NaN explicitly with a dedicated `isnan` test before the comparison. There is no
`isnan` op in Phase 1; if needed, use `fcmp_eq.f64 %x, %x` (produces `false`
exactly when `%x` is NaN) then `select`.

This is a correctness-affecting semantics choice, not just documentation. Every
backend must implement this behavior identically.

### 5.6 Select (branch-free conditional)

```
select.f64  %dst = %pred, %t, %f
select.i32  %dst = %pred, %t, %f
```

`%t` and `%f` must match the result type. This is the only conditional in Phase
1. There is no if/else, no goto, no conditional branch in user-written `.tam`.
Backends lower `select` to their hardware's conditional move or predicated
instruction.

### 5.7 Transcendental stubs

These are opcodes — not function calls. Each backend inlines the tambear-libm
implementation of the function at the call site. In the CPU interpreter before
libm is wired in, these panic with "not yet in libm."

```
tam_exp.f64  %dst = %a
tam_ln.f64   %dst = %a
tam_sin.f64  %dst = %a
tam_cos.f64  %dst = %a
tam_pow.f64  %dst = %a, %b
```

More transcendentals are added here as libm gains implementations. The IR
architect controls when a new opcode is added; libm does not demand opcodes.

### 5.8 Reduction op

```
reduce_block_add.f64  %out_buf, %slot_idx:i32, %val:f64
```

Semantic: "the accumulated value of `%val` over this block's slice is written
into `%out_buf[%slot_idx]`." In the CPU interpreter this is a direct store
(one "block" contains all elements). In the PTX backend this becomes a shared
memory tree reduction followed by a block-zero thread write. The final host-side
fold over all block partials happens outside the kernel.

This is the only reduction op in Phase 1. `reduce_block_max`,
`reduce_block_mul`, etc., exist as future ops; they do not exist now.

---

## 6. Kernel definitions

```
kernel <name>( <param-list> ) {
entry:
  <op>*
  <loop-or-reduce>*
}
```

Parameters are declared as `buf<f64> %name` for buffers and `i32 %name` for
scalars. The parameter list is comma-separated. Multiple buffers are allowed.

Example signature:
```
kernel variance_pass(buf<f64> %data, buf<f64> %out) {
```

The entry block label `entry:` is required. It is the only label a kernel may
have outside of loop bodies.

---

## 7. `loop_grid_stride` — structured iteration

```
loop_grid_stride %i in [0, %n) {
  ; body
  ; phi updates use prime-suffix convention
}
```

This is a structured block, not a goto. Semantics:

- `%i` is the induction variable, type `i32`. It is defined by the loop.
- `[0, %n)` means indices 0 through `%n - 1` inclusive.
- The loop body sees `%i` as the current element index.
- **Grid-stride:** on GPU backends, `%i` starts at `blockIdx.x * blockDim.x + threadIdx.x` and increments by `gridDim.x * blockDim.x`. Multiple threads each process a non-overlapping stride. On CPU the loop is serial starting at 0.

**Phi nodes (loop-carried values):** values initialized before the loop and
updated inside it use the prime-suffix convention.

```
%acc  = const.f64 0.0          ; initial value — "phi input"
loop_grid_stride %i in [0, %n) {
  %v    = load.f64 %data, %i
  %acc' = fadd.f64 %acc, %v   ; updated value — "phi output"
}
; after the loop, %acc' holds the final accumulated value
```

Each loop-carried register pair (`%x`, `%x'`) is a phi node: `%x` is the
value entering the current iteration; `%x'` is the value exiting it and
entering the next. The verifier checks that every prime-suffixed register is
used exactly once after the loop as the final result.

Only registers with matching base names participate in the phi. Registers
without a prime suffix that are used inside the loop but defined outside it are
read-only loop invariants.

**Nesting:** `loop_grid_stride` blocks may not be nested in Phase 1. One
level of structured iteration is sufficient for all current recipes.

---

## 8. Function definitions (for libm)

```
func <name>( <f64-param-list> ) -> f64 {
entry:
  <op>*
  ret.f64 %result
}
```

Functions may only appear in libm modules. They may not contain
`loop_grid_stride` or `reduce_block_add`. They may call other functions via
transcendental opcodes (so `tam_exp` can call `tam_ln` through the stub
mechanism, but only if `tam_ln` is defined in the same or a linked module).

`ret.f64 %result` exits the function and returns `%result`. Every function
must have exactly one `ret` in Phase 1 (no early returns, no multiple exits).

---

## 9. Text encoding — formal grammar (EBNF sketch)

```ebnf
module      = header func-def* kernel-def* ;
header      = ".tam" version-lit newline ".target" target-lit newline ;
version-lit = "0.1" ;
target-lit  = "cross" ;

kernel-def  = "kernel" ident "(" param-list ")" "{" "entry:" body "}" ;
func-def    = "func" ident "(" f64-param-list ")" "->" "f64" "{" "entry:" func-body "}" ;

param-list  = param ("," param)* ;
param       = buf-type reg | scalar-type reg ;
buf-type    = "buf<f64>" ;
scalar-type = "i32" | "i64" | "f64" ;

f64-param-list = "f64" reg ("," "f64" reg)* ;

body        = stmt* loop-stmt? stmt* ;
func-body   = stmt* "ret.f64" reg ;

stmt        = reg "=" op newline | store-op newline | reduce-op newline ;

reg         = "%" ident prime? ;
prime       = "'" ;
ident       = [a-zA-Z_][a-zA-Z0-9_]* ;

op          = const-op | buf-op | arith-op | int-op | cmp-op | sel-op | trans-op ;
```

Whitespace (spaces, tabs) separates tokens. Lines are terminated by newline.
Comments begin with `;` and run to end of line.

---

## 10. Complete example — `variance_pass`

```
.tam 0.1
.target cross

; Three-slot accumulation: sum(x), sum(x^2), count
; The host folds block partials and computes variance from the three sums.
kernel variance_pass(buf<f64> %data, buf<f64> %out) {
entry:
  %n    = bufsize %data
  %acc0 = const.f64 0.0
  %acc1 = const.f64 0.0
  %acc2 = const.f64 0.0
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

---

## 11. What does NOT exist in Phase 1

To keep scope honest, the following are explicitly out of scope. If someone is
tempted to add them, they must write a case in `peak1-tam-ir/future-ops.md`
and wait for a separate campsite:

- Vector/SIMD types
- Tensor types or shaped arrays
- `fma.f64` (FMA is always forbidden unless explicitly needed and invariant-checked)
- Integer division or modulo
- Multiple return values from functions
- Early returns (`ret` must be the last op in a function)
- Goto or unconditional branch
- Nested loops
- Conditionally-terminated loops (while loops)
- Global mutable state
- Calling kernels from functions or functions from kernels (except transcendental stubs)
- Binary encoding (text only in Phase 1)
- Integer division or remainder (use the `and.i64` + power-of-two trick for `mod` where needed)
- `f32` type (fp64 only in Phase 1)

**Note (campsite 1.14 amendment):** `f64_to_i32_rn`, `bitcast.f64.i64`, `bitcast.i64.f64`, and `ldexp.f64` are *in scope* as of campsite 1.14. The original "Type casts between f64 and i32 are out of scope" line has been removed — it predates the libm ops decision. §5.4a and §5.4b define the full set.

---

## 12. Invariants the spec upholds

| Invariant | How this spec upholds it |
|-----------|--------------------------|
| I3 (no FMA) | §5.3 explicitly forbids contraction; no `fma.f64` op exists |
| I4 (no reordering) | §5.3 mandates IEEE 754 RNE; no fast-math flags exist at the IR level |
| I5 (deterministic reduce) | `reduce_block_add` has a fixed two-stage semantic; non-determinism is in the backend, addressed by Peak 6 |
| I7 (accumulate+gather) | The only compute structure is loop (accumulate) + reduce_block (gather); no other patterns exist |
| I8 (first-principles transcendentals) | Transcendentals are opcode stubs; the IR mandates they lower to tambear-libm |
