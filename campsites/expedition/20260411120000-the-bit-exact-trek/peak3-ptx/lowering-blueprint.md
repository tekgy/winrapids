# PTX Lowering Blueprint

*Campsite 3.2 pre-work — compiled by naturalist, 2026-04-11.*
*Read ptx-subset.md first — this document assumes that vocabulary.*

This document answers: "given a .tam IR node, what PTX do I emit?"
It is a per-op translation table plus a complete worked example
(variance_pass.tam → PTX). The pathmaker can use it as a
code-generation spec for the assembler's emit phase.

---

## Naming convention

Every SSA register in .tam maps to a unique PTX register. The
naming scheme the assembler should use:

| .tam entity | PTX register name | PTX type |
|---|---|---|
| `%foo` (f64 value) | `%fd_foo` | `.f64` |
| `%foo'` (phi output, f64) | `%fd_foo_phi` | `.f64` |
| `%i` (loop index, i32) | `%r_i` | `.b32` |
| `%n` (bufsize result) | `%r_n` | `.b32` |
| `%s0` (const.i32 = 0) | `%r_s0` | `.b32` |
| `%ptr_data` (buf param) | `%rd_data` | `.b64` |
| `%ptr_out` (buf param) | `%rd_out` | `.b64` |

Primed names (`%acc'`) and unprimed (`%acc`) are distinct registers
— that is the point of the prime. The assembler allocates separate
PTX registers for each.

---

## Op-by-op lowering table

### Kernel parameters

Each `buf<f64>` parameter in the .tam kernel signature becomes a
`.param .u64` in PTX, loaded and `cvta`-converted at kernel entry.

For a kernel `foo(buf<f64> %data, buf<f64> %out)`:

```ptx
.visible .entry foo(
    .param .u64 param_data,
    .param .u64 param_out
)
{
    .reg .b64 %rd_data, %rd_out;
    // ... other register declarations ...

    ld.param.u64  %rd_data, [param_data];
    ld.param.u64  %rd_out,  [param_out];
    cvta.to.global.u64 %rd_data, %rd_data;
    cvta.to.global.u64 %rd_out,  %rd_out;
```

For N buffer parameters: N consecutive `.param .u64 paramN` entries,
N loads, N cvta conversions.

---

### `bufsize %buf`

```ptx
// .tam:  %n = bufsize %data
// PTX: length is passed as an additional u32 parameter
// (the .tam IR does not pass length explicitly — the assembler
//  adds a length parameter per buffer when emitting the PTX kernel)
//
// At kernel entry (appended parameter, emitted after all buf params):
//    .param .u32 param_n_data,   // length of %data
// Load:
    ld.param.u32 %r_n, [param_n_data];
```

**Design note**: .tam's `bufsize` op references a named buffer. The
PTX assembler must append one `u32` length parameter per `buf<f64>`
parameter in the kernel signature, in the same order as the buffer
parameters. The caller (cudarc launch site) must pass lengths
alongside buffer device pointers.

---

### `const.f64 <value>`

```ptx
// .tam:  %acc = const.f64 0.0
    mov.f64 %fd_acc, 0d0000000000000000;
```

Use `format!("0d{:016X}", value.to_bits())` in Rust to produce the
bit-exact hex literal. Never use decimal literals for f64 constants.

---

### `const.i32 <value>`

```ptx
// .tam:  %s0 = const.i32 0
    mov.b32 %r_s0, 0;
```

---

### `loop_grid_stride %i in [0, %n)` — loop prologue

Before the loop, compute thread index and stride, then initialize
all phi registers (unprimed) to their pre-loop values:

```ptx
    // Thread index and stride
    .reg .b32 %r_gi, %r_stride, %r_tid_x, %r_ntid_x, %r_ctaid_x, %r_nctaid_x;
    mov.u32 %r_tid_x,    %tid.x;
    mov.u32 %r_ntid_x,   %ntid.x;
    mov.u32 %r_ctaid_x,  %ctaid.x;
    mov.u32 %r_nctaid_x, %nctaid.x;
    mul.lo.u32 %r_gi,     %r_ctaid_x, %r_ntid_x;
    add.u32    %r_gi,     %r_gi, %r_tid_x;
    mul.lo.u32 %r_stride, %r_nctaid_x, %r_ntid_x;

    // Initialize loop index to global thread index
    mov.u32 %r_i, %r_gi;

    // Initialize phi registers (pre-loop values)
    // (one mov per phi pair declared in the loop)
    mov.f64 %fd_acc, 0d0000000000000000;   // %acc = const.f64 0.0

LOOP_HEADER_foo:
    .reg .pred %p_cond;
    setp.lt.u32 %p_cond, %r_i, %r_n;
    @!%p_cond bra LOOP_EXIT_foo;
```

**Label uniqueness**: if the kernel has multiple loops, each loop
gets a unique label suffix (use the loop index register name or a
counter).

---

### `loop_grid_stride` body — phi update pattern

Inside the loop, the phi output register (`%acc'`) is a fresh PTX
register. At loop end, copy the primed register back to the unprimed
register (implementing the phi semantics):

```ptx
    // --- loop body for: %acc' = fadd.f64 %acc, %v ---
    add.rn.f64 %fd_acc_phi, %fd_acc, %fd_v;

    // --- end of loop body ---
    // phi update: acc ← acc'
    mov.f64 %fd_acc, %fd_acc_phi;

    // Loop increment
    add.u32 %r_i, %r_i, %r_stride;
    bra LOOP_HEADER_foo;

LOOP_EXIT_foo:
    // After loop exit, %fd_acc holds the final accumulated value.
    // Use %fd_acc (not %fd_acc_phi) for any post-loop ops.
    // The convention: the post-loop value of a phi is the unprimed register.
```

**Why copy-back instead of reuse**: in .tam semantics, `%acc` is the
loop-entry value and `%acc'` is the loop-exit value. Using both as
source in the same iteration is valid. In PTX, the copy-back at the
end of each iteration implements this: `%fd_acc` holds the entry
value during iteration, `%fd_acc_phi` holds the exit value, and the
copy runs after all uses of `%fd_acc` in that iteration.

---

### `load.f64 %buf, %idx`

```ptx
// .tam:  %v = load.f64 %data, %i
    .reg .b64 %rd_load_addr;
    mul.wide.u32  %rd_load_addr, %r_i, 8;
    add.u64       %rd_load_addr, %rd_data, %rd_load_addr;
    ld.global.f64 %fd_v, [%rd_load_addr];
```

Address register can be reused if the assembler tracks liveness;
or give each load a unique scratch register (simpler, driver
optimizes).

---

### `fadd.f64`, `fsub.f64`, `fmul.f64`, `fdiv.f64`

```ptx
// .tam:  %r = fadd.f64 %a, %b
    add.rn.f64 %fd_r, %fd_a, %fd_b;

// .tam:  %r = fsub.f64 %a, %b
    sub.rn.f64 %fd_r, %fd_a, %fd_b;

// .tam:  %r = fmul.f64 %a, %b
    mul.rn.f64 %fd_r, %fd_a, %fd_b;

// .tam:  %r = fdiv.f64 %a, %b
    div.rn.f64 %fd_r, %fd_a, %fd_b;
```

**Mandatory `.rn` on every fp op. No exceptions.** (I3)

---

### `fsqrt.f64`, `fneg.f64`, `fabs.f64`

```ptx
// .tam:  %r = fsqrt.f64 %a
    sqrt.rn.f64 %fd_r, %fd_a;

// .tam:  %r = fneg.f64 %a
    neg.f64 %fd_r, %fd_a;

// .tam:  %r = fabs.f64 %a
    abs.f64 %fd_r, %fd_a;
```

---

### `fcmp_gt.f64`, `fcmp_lt.f64`, `fcmp_eq.f64`

```ptx
// .tam:  %p = fcmp_gt.f64 %a, %b
    .reg .pred %p_p;
    setp.gt.f64 %p_p, %fd_a, %fd_b;

// .tam:  %p = fcmp_lt.f64 %a, %b
    setp.lt.f64 %p_p, %fd_a, %fd_b;

// .tam:  %p = fcmp_eq.f64 %a, %b
    setp.eq.f64 %p_p, %fd_a, %fd_b;
```

**NaN behavior**: IEEE 754 comparisons return false for NaN. See
pitfall P20 and the pending I11 question. Downstream `select.f64`
with a NaN-produced predicate will take the false branch — document
this in §5.5 of the spec.

---

### `select.f64 %cond, %a, %b`

```ptx
// .tam:  %r = select.f64 %cond, %a, %b
//        (result = %a if %cond is true, %b otherwise)
    @%p_cond  mov.f64 %fd_r, %fd_a;
    @!%p_cond mov.f64 %fd_r, %fd_b;
```

---

### `reduce_block_add.f64 %out, %slot_idx, %val`

Phase 1 (atomicAdd, I5 violation flagged):

```ptx
// .tam:  reduce_block_add.f64 %out, %s0, %acc'
//        where %s0 was: const.i32 0
//
// Compute the address of out[0]:
    .reg .b64 %rd_slot_addr;
    mul.wide.u32  %rd_slot_addr, %r_s0, 8;   // offset = slot * 8
    add.u64       %rd_slot_addr, %rd_out, %rd_slot_addr;
// I5: atomicAdd is non-deterministic. Temporary. Peak 6 replaces this.
    atom.global.add.f64 [%rd_slot_addr], %fd_acc_phi;
```

The slot index is always a `const.i32` value — the assembler should
inline the constant offset directly when possible:

```ptx
// Optimized: slot 0 → offset 0, slot 1 → offset 8, slot 2 → offset 16
    atom.global.add.f64 [%rd_out],      %fd_acc0_phi;  // slot 0
    atom.global.add.f64 [%rd_out+8],    %fd_acc1_phi;  // slot 1
    atom.global.add.f64 [%rd_out+16],   %fd_acc2_phi;  // slot 2
```

The `[%rd_out+8]` PTX immediate-offset form is valid and preferred
over a separate address computation register.

---

## Worked example: variance_pass.tam → PTX

Source (from peak1-tam-ir/programs/variance_pass.tam):

```
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

Target PTX:

```ptx
.version 8.5
.target sm_120
.address_size 64

.visible .entry variance_pass(
    .param .u64 param_data,
    .param .u64 param_out,
    .param .u32 param_n_data    // bufsize for %data — added by assembler
)
{
    // Register declarations
    .reg .b64  %rd_data, %rd_out, %rd_load_addr;
    .reg .b32  %r_n, %r_i, %r_stride;
    .reg .b32  %r_tid_x, %r_ntid_x, %r_ctaid_x, %r_nctaid_x;
    .reg .pred %p_cond;
    .reg .f64  %fd_acc0, %fd_acc1, %fd_acc2;
    .reg .f64  %fd_acc0_phi, %fd_acc1_phi, %fd_acc2_phi;
    .reg .f64  %fd_v, %fd_v2, %fd_one;

    // Load and address-convert buffer parameters
    ld.param.u64       %rd_data, [param_data];
    ld.param.u64       %rd_out,  [param_out];
    cvta.to.global.u64 %rd_data, %rd_data;
    cvta.to.global.u64 %rd_out,  %rd_out;

    // bufsize %data → %r_n
    ld.param.u32 %r_n, [param_n_data];

    // const.f64 0.0 × 3 — phi pre-loop initialization
    mov.f64 %fd_acc0, 0d0000000000000000;
    mov.f64 %fd_acc1, 0d0000000000000000;
    mov.f64 %fd_acc2, 0d0000000000000000;

    // Grid-stride loop setup
    mov.u32 %r_tid_x,    %tid.x;
    mov.u32 %r_ntid_x,   %ntid.x;
    mov.u32 %r_ctaid_x,  %ctaid.x;
    mov.u32 %r_nctaid_x, %nctaid.x;
    mul.lo.u32 %r_i,      %r_ctaid_x, %r_ntid_x;
    add.u32    %r_i,      %r_i, %r_tid_x;
    mul.lo.u32 %r_stride, %r_nctaid_x, %r_ntid_x;

LOOP_HEADER:
    setp.lt.u32 %p_cond, %r_i, %r_n;
    @!%p_cond bra LOOP_EXIT;

    // load.f64 %data, %i  →  %v
    mul.wide.u32  %rd_load_addr, %r_i, 8;
    add.u64       %rd_load_addr, %rd_data, %rd_load_addr;
    ld.global.f64 %fd_v, [%rd_load_addr];

    // fmul.f64 %v, %v  →  %v2
    mul.rn.f64 %fd_v2, %fd_v, %fd_v;

    // const.f64 1.0  →  %one
    mov.f64 %fd_one, 0d3FF0000000000000;

    // fadd.f64 %acc0, %v  →  %acc0'
    add.rn.f64 %fd_acc0_phi, %fd_acc0, %fd_v;

    // fadd.f64 %acc1, %v2  →  %acc1'
    add.rn.f64 %fd_acc1_phi, %fd_acc1, %fd_v2;

    // fadd.f64 %acc2, %one  →  %acc2'
    add.rn.f64 %fd_acc2_phi, %fd_acc2, %fd_one;

    // Phi copy-back: acc ← acc' (implements loop-carried update)
    mov.f64 %fd_acc0, %fd_acc0_phi;
    mov.f64 %fd_acc1, %fd_acc1_phi;
    mov.f64 %fd_acc2, %fd_acc2_phi;

    // Loop increment
    add.u32 %r_i, %r_i, %r_stride;
    bra     LOOP_HEADER;

LOOP_EXIT:
    // reduce_block_add.f64 %out, 0, %acc0'
    // reduce_block_add.f64 %out, 1, %acc1'
    // reduce_block_add.f64 %out, 2, %acc2'
    // I5: atomicAdd is non-deterministic. Temporary. Peak 6 replaces this.
    atom.global.add.f64 [%rd_out],    %fd_acc0_phi;
    atom.global.add.f64 [%rd_out+8],  %fd_acc1_phi;
    atom.global.add.f64 [%rd_out+16], %fd_acc2_phi;

    ret;
}
```

---

## Key assembler decisions encoded above

1. **One PTX length parameter per buffer parameter**, appended after
   the buffer pointers in declaration order. The launch site must pass
   `(data_ptr, out_ptr, data_len)`.

2. **Phi registers: unprimed = entry value, primed = exit value.**
   Both are live simultaneously within the loop body. The copy-back
   at loop end is the implementation of phi semantics.

3. **`const.f64` inside loops is hoisted to pre-loop by default.**
   The `%one = const.f64 1.0` inside the loop body is re-initialized
   each iteration in the naive lowering above (correct but wasteful).
   The assembler should move all `const.f64` / `const.i32` that don't
   depend on loop-carried values to the pre-loop region.

4. **`reduce_block_add` slot-constant immediate offset — optimization pass, not mandatory.**
   When the slot index is a `const.i32` (it always is in Phase 1),
   the offset `slot * 8` can be computed at assemble time and emitted
   as a PTX immediate: `[%rd_out+8]` rather than a separate address
   register. This saves two PTX instructions per reduce op.
   **Implementation order:** get the naive lowering correct first
   (runtime multiply → add → atom with a separate `%rd_slot_addr`
   register). Verify it against the CPU interpreter output. Then add
   the immediate-offset optimization as a second assembler pass.
   The naive lowering is the testable reference; the optimization must
   not change output.

5. **All fp ops carry `.rn`.** Every `add`, `sub`, `mul`, `div`,
   `sqrt` in the emitted PTX must have the `.rn` modifier. The
   assembler should enforce this as a post-pass lint: if any emitted
   line contains a bare fp op without `.rn`, that is a code-gen bug.

---

## What this document does NOT cover

- **`select.f64` / `select.i32`**: see op table above; no worked
  example yet. Trivial extension of the predicate pattern.
- **`fcmp_*` and `icmp_lt`**: see op table; exercised only by
  programs that use conditional expressions (none in Phase 1 programs
  yet).
- **`tam_exp`, `tam_ln`, etc.**: transcendental stubs. These lower to
  a series of arithmetic ops defined by the libm implementation in
  Peak 2. Not addressed here; Peak 3 cannot fully lower these until
  Peak 2 is complete.
- **Phase 6 `reduce_block_add` replacement**: shared-memory tree
  reduce with `bar.sync`. The ptx-subset.md §Shared memory section
  covers the pattern; the worked example here uses Phase 1 atomicAdd
  deliberately, with the I5 flag.
- **Multi-kernel programs**: Phase 1 has only single-kernel programs.
  The assembler's program-level dispatch (which kernel to lower,
  how to export module symbols) is not addressed here.
- **`min.f64` / `max.f64` and I11 NaN semantics**: Phase 1 has no
  min/max ops in the IR op set, so this isn't a current concern. But
  when they are added: the correct PTX emit depends on the target ISA version.
  See `vendor-bugs.md` VB-004 for the full policy. Summary:
  - **PTX ISA 7.5+ (SM 80 Ampere and later, including RTX 6000 Pro Blackwell SM 120)**:
    emit `min.NaN.f64` / `max.NaN.f64`. The `.NaN` modifier forces NaN
    propagation — if either operand is NaN, the result is NaN. This is the
    correct I11-compliant path. Never emit bare `min.f64` / `max.f64` — these
    follow C99 `fmin` semantics (return the non-NaN operand), violating I11.
  - **Pre-ISA 7.5 targets (Turing/Volta, SM 70/75)**: compose from
    `setp.unordered.f64` (isnan check) + `setp.lt.f64` + `selp.f64`, same
    logical structure as the SPIR-V six-instruction composition in VB-001.
    See VB-004 for the full six-instruction PTX sequence.
  The PTX path is simpler than the SPIR-V path because PTX has a native
  NaN-propagating variant (`.NaN` modifier). SPIR-V has none; it must always
  compose (VB-001, ESC-002 Option 3).
