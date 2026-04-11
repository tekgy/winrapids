# Compiler Vision

## The Goal

Entire pipeline → one compiled plan → minimum passes → any ALU.

The user builds a 47-step pipeline in the IDE. TAM sees ALL steps before executing ANY. It collects every accumulate slot across the entire pipeline, groups by `(source, grouping, op)`, and emits the minimum number of passes. Each pass is one kernel. A slot from step 1 and a slot from step 47 that share `(source, grouping, op)` execute in the SAME kernel — even though step 47's output isn't consumed until later. The result sits in memory until needed.

Not per-step execution. Not per-recipe compilation. GLOBAL pipeline fusion.

## The Compilation Flow

```
Pipeline (all recipes across all stages)
    ↓
TAM collects ALL AccumulateSlots
    ↓
Groups by (DataSource, Grouping, Op)
    ↓
Each group → one kernel (one pass through that data)
    ↓
Gathers execute over accumulated results (O(1) per gather, not O(n))
    ↓
Vendor door: kernel → hardware (CUDA driver, Vulkan, Metal, CPU)
```

## The Key Insight

The number of data passes = the number of unique `(DataSource, Grouping, Op)` triples across the ENTIRE pipeline. Not the number of steps. Not the number of recipes. Not the number of statistics.

A 47-step pipeline computing 200 statistics might have only 3 unique triples:
- (Raw, All, Add) — one pass: sum, count, sum_sq, sum_cubed, sum_4th, log_sum, reciprocal_sum, sum_xy, count_positive, count_finite, etc. ALL in one loop.
- (Raw, All, Max) — one pass: max value, max absolute value, etc.
- (Raw, All, Min) — one pass: min value.

Three passes. 200 outputs. One pipeline.

## Precomputation

If step 47 needs `Accumulate(Raw, All, Add)` with `Expr::val()`, and step 1 already declared the same slot, TAM doesn't compute it again. It was already computed in the global pass. Step 47 just reads the precomputed result.

The same applies to gathers. If multiple steps gather `sum / count` from the same accumulated values, TAM can batch those gathers. The gather is O(1) over scalars — almost free.

## What We Have Now

- `fuse_passes()` — groups AccumulateSlots by `(source, grouping, op)`. Already global — takes ALL slots regardless of recipe.
- `execute_pass_cpu()` — reference implementation. One loop per fused group, all Expr outputs computed per element.
- `DataSource` — tracks what data each slot operates over (Primary, Column, Prior).
- Recipes declared as data — AccumulateSlots + Gather Exprs.

## What We Need

- `.tam` IR format — our compiled representation. Not CUDA, not WGSL, not SPIR-V. Ours.
- TAM compiler — reads the global fused plan, emits `.tam` IR per kernel.
- Vendor doors — translate `.tam` IR to hardware-specific dispatch:
  - CUDA: `.tam` → PTX (via CUDA driver API, no nvcc, no CUDA strings)
  - Vulkan: `.tam` → SPIR-V (via vulkan compute)
  - Metal: `.tam` → Metal IR
  - CPU: `.tam` → native machine code (or interpreted)

## The CPU Executor's Role

`execute_pass_cpu()` is the oracle. Every compiled kernel must produce the same output as the CPU executor on the same input. The CPU path is:
1. How we test recipes during development
2. The fallback when no GPU is available
3. The gold standard for correctness

## When to Build This

After the atoms are complete and we have enough recipes to prove the pattern. The compiler transforms a solved problem (recipes) into an optimization problem (kernel emission). The atoms come first.
