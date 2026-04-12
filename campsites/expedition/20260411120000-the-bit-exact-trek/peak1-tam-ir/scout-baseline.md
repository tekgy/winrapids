# Scout Report: Peak 1 Baseline

*Scout: claude-sonnet-4-6 | Date: 2026-04-11*

Pre-loaded terrain for the IR Architect (pathmaker). Everything below is from source,
not inference.

---

## 1. The codebase the IR Architect inherits

### 1.1 What already exists (`crates/tambear-primitives/src/`)

**`tbs/mod.rs`** — The `Expr` AST. 42 variants, all implemented:

| Category | Variants |
|---|---|
| Leaves | `Val`, `Val2`, `Col(usize)`, `Ref`, `Lit(f64)`, `Var(String)` |
| Unary | `Neg`, `Abs`, `Recip`, `Sq`, `Sqrt`, `Ln`, `Exp`, `Floor`, `Ceil`, `Sign`, `IsFinite` |
| Trig | `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Tanh` |
| Round | `Round`, `Trunc` |
| Binary | `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Min`, `Max`, `Atan2`, `Mod` |
| Ternary | `Clamp`, `If` |
| Comparison | `Gt`, `Lt`, `Eq` |

The `eval()` function interprets `Expr` in pure Rust. This is NOT the IR — this is
the current CPU evaluator. The IR Architect's job is to define a `.tam` AST that
maps these `Expr` variants to opcodes.

Key insight: `Expr::Sq` is sugar for `x * x` (NOT an opcode) — the Trek plan
uses `fmul.f64` for both. `Expr::Recip` is sugar for `1.0 / x` using `fdiv.f64`.
`Expr::Abs` → the `fabs.f64` opcode.

**`accumulates/mod.rs`** — The fuse/execute machinery:

- `Grouping`: `All`, `ByKey`, `Prefix`, `Segmented`, `Windowed`, `Tiled`, `Graph`
- `Op`: `Add`, `Max`, `Min`, `Mul`, `And`, `Or`
- `fuse_passes()` groups slots by `(DataSource, Grouping, Op)` — identical key = same kernel
- `execute_pass_cpu()` is ONE loop with ALL slot expressions evaluated per element

**`recipes/mod.rs`** — 26 recipes, 6 families:

| Family | Count | Ops used |
|---|---|---|
| Raw reductions | 4 | Add, Mul |
| Means | 4 | Add only |
| Moments | 5 | Add only |
| Norms | 3 | Add, Max |
| Extrema | 4 | Min, Max |
| Two-column | 6 | Add only |

Fusion result: all 26 recipes → **4 kernel passes** (Add/All, Max/All, Min/All, Mul/All).
The primary Add pass holds 47 of 55 accumulate slots. This means the IR Architect
needs to handle the one-pass reduction pattern before anything else — that's the
critical path.

**`codegen/cuda.rs`** — the legacy path:

- `expr_to_cuda()`: `Expr` → CUDA C string
- `pass_to_cuda_kernel()`: `AccumulatePass` → complete CUDA C source
- Only supports `Grouping::All` + `Op::Add` (the dominant path)
- Uses `atomicAdd` for reduction (I5 violation — non-deterministic, flagged in state.md)
- Uses `log()`, `exp()`, `sin()` etc. — calls NVRTC's math library (I1 violation)
- This code is read-only legacy; the new path replaces it

---

## 2. The mapping from `Expr` to `.tam` ops

Every `Expr` variant has a direct translation:

```
Expr::Val       → load.f64 %dst = %x_buf, %i
Expr::Val2      → load.f64 %dst = %y_buf, %i
Expr::Lit(c)    → const.f64 %dst = c
Expr::Var(name) → [gather-time only; forbidden in accumulate body]
Expr::Neg(a)    → fneg.f64 %dst = %a
Expr::Abs(a)    → fabs.f64 %dst = %a
Expr::Recip(a)  → fdiv.f64 %dst = const.f64(1.0), %a
Expr::Sq(a)     → fmul.f64 %dst = %a, %a  [computes %a once, uses twice — phi needed]
Expr::Sqrt(a)   → fsqrt.f64 %dst = %a
Expr::Ln(a)     → tam_ln.f64 %dst = %a     [transcendental stub]
Expr::Exp(a)    → tam_exp.f64 %dst = %a
Expr::Sin(a)    → tam_sin.f64 %dst = %a
Expr::Cos(a)    → tam_cos.f64 %dst = %a
Expr::Tan(a)    → tam_sin / tam_cos + fdiv
Expr::Add(a,b)  → fadd.f64 %dst = %a, %b
Expr::Sub(a,b)  → fsub.f64 %dst = %a, %b
Expr::Mul(a,b)  → fmul.f64 %dst = %a, %b
Expr::Div(a,b)  → fdiv.f64 %dst = %a, %b
Expr::Gt(a,b)   → fcmp_gt.f64 %p = %a, %b; then select.f64 %dst = %p, 1.0, 0.0
Expr::If(c,t,e) → [compile c to pred] then select.f64 %dst = %pred, %t, %e
```

**Note on `Expr::Sq`:** the current CPU eval does `let v = eval(a, ...); v * v`.
In the IR, the inner expression gets a register, then `fmul.f64 %sq = %v, %v`.
No special "square" opcode needed — SSA handles it naturally.

---

## 3. The fused-pass structure the IR must express

Every fused pass for `Grouping::All, Op::Add` has this shape:

```
1. N threads, each responsible for a grid-stride slice
2. Each thread: for i in [gi, n, stride):
       load x; [optionally load y]
       compute expr_0(x, y) → acc_0 += result
       compute expr_1(x, y) → acc_1 += result
       ...
       compute expr_k(x, y) → acc_k += result
3. Thread writes partials to output buffer
```

This maps to `loop_grid_stride` with loop-carried accumulators (the phi nodes).
The variance recipe (3 slots: sum, sum_sq, count) has 3 accumulators, all
in ONE pass. The IR needs phi syntax for loop-carried state.

**The critical question for the IR Architect:** how do phi nodes work in the text
format? The plan suggests `%acc0' = fadd.f64 %acc0, %v` where the prime notation
indicates the updated value. Pitfall: this looks like SSA but isn't — `%acc0` and
`%acc0'` both refer to the loop-carried accumulator. The implementer needs to decide
whether the text format uses Prolog-style `%acc0'` or explicit `phi` blocks or
labeled "accumulator slots" at the loop header.

Recommendation: the simplest unambiguous syntax is labeled slots at the loop header:
```
loop_grid_stride %i in [0, %n) accumulators(%acc0: f64 = 0.0, %acc1: f64 = 0.0) {
    ...
    %acc0 = fadd.f64 %acc0, %v     ; reuse of name = "update this slot"
}
```
Inside the loop, reassignment of a declared accumulator = update the slot,
not a new SSA value. This is a pragmatic compromise with pure SSA that keeps the
text format readable without full basic block phi nodes.

---

## 4. What "variance looks like when fused"

The pathmaker will be asked this. Here it is, complete:

```
// From recipes/mod.rs: variance() recipe
// slots: [sum_slot(), sum_sq_slot(), count_slot()]
// All share: DataSource::Primary, Grouping::All, Op::Add
// → fuse_passes produces ONE AccumulatePass with slots:
//   (Expr::Val,        "sum")
//   (Expr::Sq(Val),    "sum_sq")
//   (Expr::Lit(1.0),   "count")

// Hand-written .tam equivalent:
.tam 0.1
.target cross

kernel variance_pass(buf<f64> %data, buf<f64> %out) {
entry:
    %n    = bufsize %data
    loop_grid_stride %i in [0, %n) accumulators(%s: f64 = 0.0, %ss: f64 = 0.0, %c: f64 = 0.0) {
        %v    = load.f64 %data, %i
        %v2   = fmul.f64 %v, %v      // x²
        %one  = const.f64 1.0
        %s    = fadd.f64 %s, %v      // Σx
        %ss   = fadd.f64 %ss, %v2    // Σx²
        %c    = fadd.f64 %c, %one    // count
    }
    // reduce partials into output buffer
    %sl0  = const.i32 0
    %sl1  = const.i32 1
    %sl2  = const.i32 2
    reduce_block_add.f64 %out, %sl0, %s
    reduce_block_add.f64 %out, %sl1, %ss
    reduce_block_add.f64 %out, %sl2, %c
}
```

Then the gather (CPU-side) computes: `(out[1] - out[0]*out[0]/out[2]) / (out[2] - 1.0)`.

---

## 5. What the gather stage looks like in the current code

The `Gather` struct (`gathers/mod.rs`) holds an `Expr` over `Var(name)` references
to the accumulated outputs. The `eval()` call resolves these from a `HashMap<String, f64>`.

In the new IR, gather expressions don't go into `.tam` IR — they stay as `Expr` evaluated
on the CPU after the GPU writes partials back. The `.tam` IR is purely the *accumulate* pass.
Gather is always CPU; only the hot loop is GPU/IR. This is the correct separation.

---

## 6. The TBS → .tam IR compiler path

The existing `codegen/cuda.rs::expr_to_cuda()` is a template for the new
`tambear-tam-ir::lower_expr()`. Same recursive tree walk, same structure,
different output:
- old: emits CUDA C string
- new: emits a sequence of `.tam` IR opcodes (Rust enum nodes)

The pathmaker should model the new module on the exact structure of `expr_to_cuda()`
but target an IR AST instead of a string.

---

## 7. Gaps the IR Architect must fill

Things NOT in the current code that the .tam IR needs:

1. **`bufsize` op** — getting the length of a buffer. Currently baked into CUDA via `#define TAM_N`.
   In .tam IR this is an explicit op: `%n = bufsize %data`.

2. **`reduce_block_add.f64`** — the per-block partial write. Currently `atomicAdd` in CUDA C.
   The IR needs this as an explicit op with its own semantic.

3. **Loop phi nodes** — the accumulator update syntax. Currently just Rust mutation in the CPU executor.

4. **Type system** — the current `Expr` is untyped (everything is f64 at runtime). The IR
   needs to track `f64`, `i32`, and `pred` types per register.

5. **`const.i32` and `iadd.i32`** — for loop counter and buffer indexing. Currently implicit.
   These need explicit opcodes in the IR.

---

## 8. Known pitfall: the `Expr::Sq` redundant computation

In `codegen/cuda.rs`, `Expr::Sq(a)` generates `(({0})*({0}))` where `{0}` is the
result of `rec(a)`. If `a` itself has side effects (function calls), this evaluates
`a` **twice**. Current CUDA C is fine because C compilers CSE aggressively. But in
the .tam IR, the IR must explicitly assign `%v = [compute a]` and then `%sq = fmul.f64 %v, %v`,
never inline the sub-expression twice. This is a correctness-and-performance issue in
the lowering pass.

---

## 9. What the variance CPU test currently asserts

From `recipes/mod.rs` test:
```rust
let v = run_recipe(&variance(), &[1.0, 2.0, 3.0, 4.0, 5.0], &[]);
assert!((v - 2.5).abs() < 1e-14);
```

The GPU test (`gpu_end_to_end.rs`) runs with `data = [sin(i)*10+5, i in 0..10000]`
and checks `rel < 1e-10` due to non-deterministic `atomicAdd`. The new deterministic
path (Peak 6) will tighten this to `to_bits() ==`.

---

## 10. File layout the IR Architect will create

Per `navigator/state.md`:
- `crates/tambear-tam-ir/src/` — IR AST types (`TamOp`, `TamFunc`, `TamKernel`, `TamProgram`)
- `crates/tambear-tam-ir/src/parser.rs` — text format → AST
- `crates/tambear-tam-ir/src/printer.rs` — AST → text format (round-trip oracle)
- `crates/tambear-tam-ir/src/verifier.rs` — type/SSA checks
- `crates/tambear-tam-ir/src/lower.rs` — `Expr + AccumulatePass → TamProgram` (the key entry point)

The `lower.rs` module is what Peak 1 is really about. Parser/printer are secondary;
they're needed for the text format but the compiler-in, compiler-out pipeline is
`lower.rs`.
