# PTX Subset We Emit

*Campsite 3.1 research — compiled by scout, 2026-04-11.*
*Reference: NVIDIA PTX ISA 8.5 (for sm_120, Blackwell architecture)*

This document defines the exact PTX subset our assembler emits. Nothing more.
If an op is not listed here, we do not emit it in Phase 1.

---

## Required file header

Every PTX file we emit begins with exactly:

```ptx
.version 8.5
.target sm_120
.address_size 64
```

**Why 8.5**: PTX 8.5 is the version that introduces sm_100a/sm_120 (Blackwell) support.
Using an older version on sm_120 may cause the driver to reject the module.

**Why `address_size 64`**: We're on a 64-bit platform; all device pointers are u64.
Omitting this causes pointer truncation bugs that are silent and catastrophic.

---

## Type system (what we use)

| PTX type | Rust equivalent | Usage |
|---|---|---|
| `.f64` | `f64` | All floating-point values |
| `.u64` | pointer (`*const f64`) | Buffer device pointers |
| `.u32` / `.b32` | `u32` | Loop counters, block/thread IDs |
| `.pred` | `bool` | Predicate registers for branches |

We do NOT use: `.f32`, `.s32`, `.s64`, `.b64`, or any vector types.

---

## Register file

All registers are virtual in PTX (unlimited). We declare them at the function head:

```ptx
.reg .f64  %fd<N>;    // N = number of f64 virtual registers needed
.reg .b64  %rd<M>;    // M = number of pointer registers
.reg .b32  %r<K>;     // K = number of 32-bit registers (loop counters, IDs)
.reg .pred %p<J>;     // J = number of predicate registers
```

The `<N>` in `%fd<N>` declares N registers named `%fd0` through `%fd{N-1}`.
Naive register allocation: every SSA value gets a fresh register. No reuse.
This is correct and generates valid PTX; the driver's JIT will optimize.

---

## Kernel entry point

```ptx
.visible .entry kernel_name(
    .param .u64 param0,    // pointer to input buffer (data_x)
    .param .u64 param1,    // pointer to output buffer  
    .param .u32 param2     // n_elements (if not baked in)
)
{
    // register declarations here
    
    // load parameters
    ld.param.u64 %rd0, [param0];
    ld.param.u64 %rd1, [param1];
    
    // convert generic to global address space
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;
    
    // ... body ...
    
    ret;
}
```

**Critical: `cvta.to.global.u64`**. Parameters arrive in the generic address space.
Before any `ld.global` or `st.global`, you MUST convert with `cvta.to.global`.
Without this, the load silently reads from address 0 (or undefined behavior).
This is Pitfall P03.

---

## Thread ID computation (for grid-stride loop)

```ptx
// Get global thread index and stride
.reg .b32 %tid, %ntid, %ctaid, %nctaid, %gi, %stride;

mov.u32 %tid,    %tid.x;
mov.u32 %ntid,   %ntid.x;
mov.u32 %ctaid,  %ctaid.x;
mov.u32 %nctaid, %nctaid.x;

// gi = blockIdx.x * blockDim.x + threadIdx.x
mul.lo.u32 %gi, %ctaid, %ntid;
add.u32    %gi, %gi, %tid;

// stride = gridDim.x * blockDim.x
mul.lo.u32 %stride, %nctaid, %ntid;
```

---

## Floating-point instructions (the complete subset)

**I3 rule: every fp instruction MUST carry `.rn` (round-to-nearest-even).**
**I3 rule: NEVER emit bare `add.f64` — always `add.rn.f64`.**

When `.rn` is specified explicitly, the driver's JIT will NOT contract the instruction
into an FMA even if the sequence matches `a * b + c`. This is the only way to satisfy I3
without PTX-level `.contract` pragmas.

```ptx
// Arithmetic (all MUST use .rn)
add.rn.f64   %dst, %a, %b;
sub.rn.f64   %dst, %a, %b;
mul.rn.f64   %dst, %a, %b;
div.rn.f64   %dst, %a, %b;    // correct division (not approximate)
sqrt.rn.f64  %dst, %a;        // IEEE-correctly-rounded sqrt

// Negation and absolute value (no rounding mode needed)
neg.f64      %dst, %a;
abs.f64      %dst, %a;
```

**DO NOT use:**
- `fma.rn.f64 %dst, %a, %b, %c` — this is FMA, which is contraction (I3 violation)
- `add.f64` (bare, no rounding mode) — ambiguous, may be contracted
- `div.approx.f64` — approximate division, wrong answers
- `rcp.approx.f64` — approximate reciprocal, I1 violation (vendor math)
- `sqrt.approx.f64` — approximate sqrt, I1 violation

---

## Load and store

```ptx
// Load f64 from global memory: dst = data[index]
// Address = base_ptr + index * 8
mul.wide.u32  %rd_offset, %r_index, 8;    // offset in bytes (8 = sizeof(f64))
add.u64       %rd_addr, %rd_base, %rd_offset;
ld.global.f64 %fd_dst, [%rd_addr];

// Store f64 to global memory: data[index] = src
mul.wide.u32  %rd_offset, %r_index, 8;
add.u64       %rd_addr, %rd_base, %rd_offset;
st.global.f64 [%rd_addr], %fd_src;
```

`mul.wide.u32` sign-extends the 32-bit index to 64 bits during the multiply.
Without this, the index computation wraps for arrays > 2^32 bytes (512M elements).

---

## Constants

f64 constants in PTX must use the bit-exact hex literal form, not decimal:

```ptx
// WRONG (loses precision):
mov.f64 %fd0, 1.5;

// CORRECT:
mov.f64 %fd0, 0d3FF8000000000000;   // 1.5 exactly
```

The `0d` prefix followed by 16 hex digits is the bit-exact PTX literal for f64.
In Rust: `format!("0d{:016X}", value.to_bits())`.

For the ln(2) dual-precision trick in libm:
```ptx
mov.f64 %ln2_hi, 0d3FE62E42FEFA3800;   // ln(2) high part
mov.f64 %ln2_lo, 0d3D2EF357AF1D4400;   // ln(2) low part (Cody-Waite split)
```

---

## Grid-stride loop structure

```ptx
// Pseudocode → PTX for grid-stride loop
// for (int i = gi; i < n; i += stride) { body }

    // Initialize loop variable
    mov.u32 %r_i, %gi;           // i = gi

LOOP_HEADER:
    // Check condition: i < n
    setp.lt.u32 %p_cond, %r_i, %r_n;    // p_cond = (i < n)
    @!%p_cond bra LOOP_EXIT;             // if !cond, jump to exit

    // Loop body here: load data[i], compute, update accumulators
    // ...
    
    // Increment: i += stride
    add.u32 %r_i, %r_i, %stride;
    bra     LOOP_HEADER;

LOOP_EXIT:
    // Continue with reduction
```

**Loop-carried accumulators** are pre-initialized scalar registers before the loop
and updated inside. They are NOT phi nodes in PTX — PTX uses SSA-like virtual registers
but the loop update is a mutation of the named register:

```ptx
// Initialize accumulators before loop
mov.f64 %acc0, 0d0000000000000000;   // 0.0
mov.f64 %acc1, 0d0000000000000000;

LOOP_HEADER:
    // ...
    // Update: acc0 += v
    add.rn.f64 %acc0, %acc0, %fd_v;   // acc0 is both source and destination (valid in PTX)
    add.rn.f64 %acc1, %acc1, %fd_vsq;
    // ...
```

In PTX, a register can appear as both source and destination. This is how loop-carried
accumulators work — no phi nodes needed.

---

## Block reduction (Phase 1: atomicAdd, Phase 6: RFA)

**Phase 1 (temporary, I5 violation, replaced in Peak 6):**

```ptx
// atomicAdd for block-level reduction
// atom.global.add.f64 requires sm_60+; Blackwell = sm_120, fine.
atom.global.add.f64 [%rd_out_slot_addr], %acc0;
```

**Why atomicAdd is used first:**
- It's 3 lines of PTX and the simplest correctness path
- It violates I5 (non-deterministic across runs) — explicitly flagged
- Peak 6 replaces this with shared memory tree reduce → fixed-order host fold → RFA

**Phase 1 flag in code:**
```ptx
// I5: atomicAdd is non-deterministic. Temporary. Peak 6 replaces this.
atom.global.add.f64 [%rd_out_addr], %acc0;
```

---

## Shared memory (needed for Peak 6 tree reduce)

```ptx
// Declare block-local shared memory (256 threads * 8 bytes = 2048 bytes)
.shared .align 8 .b8 smem[2048];

// Load shared address for this thread
mov.u32       %r_tid, %tid.x;
mul.wide.u32  %rd_smem_off, %r_tid, 8;
mov.u64       %rd_smem_base, smem;
add.u64       %rd_smem_addr, %rd_smem_base, %rd_smem_off;

// Store accumulator to shared memory
st.shared.f64 [%rd_smem_addr], %acc0;

// Barrier: all threads must write before any reads
bar.sync 0;

// Tree reduce: stride = 128, 64, 32, 16, 8, 4, 2, 1
// (for 256-thread block, this is 8 iterations)
```

**`bar.sync 0`** is the PTX barrier instruction. All threads in the block must reach
it before any thread proceeds. MUST appear before and after every tree-reduce step.
Missing one = race condition = wrong answer. See Pitfall P10.

---

## Predicated execution

PTX has explicit predicate registers for branch-free code:

```ptx
// Compute a comparison
setp.gt.f64  %p0, %fd_a, %fd_b;     // p0 = (a > b)

// Predicated instructions (execute only if predicate is true/false)
@%p0  mov.f64 %fd_result, %fd_a;    // if p0: result = a
@!%p0 mov.f64 %fd_result, %fd_b;   // if !p0: result = b
```

This is the branch-free select pattern (maps to `Expr::If`).

---

## What we do NOT emit (explicit exclusions)

| PTX feature | Reason excluded |
|---|---|
| `fma.rn.f64` | I3: FMA contraction |
| `rcp.f64` | I1: use `div.rn.f64 1.0, x` instead |
| `sqrt.approx.f64` | I1: approximate = vendor math choice |
| `sin.approx.f64`, `cos.approx.f64` | I1: vendor math |
| `lg2.approx.f64`, `ex2.approx.f64` | I1: vendor math |
| `atom.global.add.f64` | I5 (after Peak 6) |
| `.pragma "nounroll"` | compiler directive, not needed |
| Texture/surface instructions | not used |
| Tensor core instructions | not used |
| FP16/BF16 instructions | not used (fp64 only) |
| `call` instruction | libm is inlined, not called |

---

## FMA contraction — the full story

PTX has a `.contract` pragma that affects whether `mul` + `add` sequences can be
automatically contracted into `fma`. The default behavior:

```ptx
// Without explicit .rn, this sequence MAY be contracted:
mul.f64  %a, %b, %c;
add.f64  %d, %a, %e;    // driver may fuse into fma.f64 %d, %b, %c, %e

// With .rn, contraction is suppressed:
mul.rn.f64  %a, %b, %c;
add.rn.f64  %d, %a, %e;    // NEVER contracted, guaranteed
```

The PTX ISA spec (section 10.7.1) states: "When a rounding modifier is specified,
the instruction is NOT contracted." This is the rule we rely on. The rule holds for
all PTX versions >= 1.0.

**The `.contract false` pragma** is an alternative but requires specifying it per-file
or per-function, and its interaction with optimization passes is less clear. The per-op
`.rn` approach is more local and verifiable. We use `.rn`.
