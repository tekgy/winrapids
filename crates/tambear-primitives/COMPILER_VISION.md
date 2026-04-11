# Compiler Vision

## The Goal

One recipe → one compiled kernel → any ALU.

The user writes `mean().variance().pearson_r()`. TAM compiles it to ONE `.tam` artifact that runs on whatever hardware is available. Not "fused passes." Not "one loop per grouping." ONE kernel, the whole pipeline.

## The Path

```
Recipe (Expr + Grouping + Op + Gather)
    ↓
TAM compiler: recipe → .tam (single compiled kernel)
    ↓
Vendor door: .tam → hardware execution (CUDA driver, Vulkan, Metal, CPU)
```

## What We Have Now

- `execute_pass_cpu()` — the REFERENCE IMPLEMENTATION. Interprets recipes on CPU. For testing and correctness validation. NOT the production path.
- `ComputeEngine` + `ScatterJit` in the old crate — already JIT-compiles multiple phi expressions into ONE GPU kernel. This is the right idea, wrong vendor coupling (emits CUDA strings).

## What We Need

- `.tam` IR format — our compiled representation. Not CUDA, not WGSL, not SPIR-V. Ours.
- TAM compiler — reads recipes, emits `.tam` IR.
- Vendor doors — translate `.tam` IR to hardware-specific dispatch:
  - CUDA: `.tam` → PTX (via CUDA driver API, no nvcc)
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
