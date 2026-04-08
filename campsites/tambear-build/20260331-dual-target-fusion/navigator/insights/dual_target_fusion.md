# Dual-Target Scatter Fusion: Gap #4

## What was built

`ScatterJit::scatter_phi_dual_target` — computes phi once, scatters to two independent groupings:

```c
extern "C" __global__ void scatter_phi_dual_target(
    const int* keys0, const int* keys1,
    const double* values,
    double* out0, double* out1, int n
) {
    double phi_val = ({phi});              // computed ONCE
    atomicAdd(&out0[keys0[gid]], phi_val); // scattered to grouping 0
    atomicAdd(&out1[keys1[gid]], phi_val); // scattered to grouping 1
}
```

Cache key: `"dual:{phi_expr}"`. Two independent group arrays, two independent output arrays — all in one kernel pass.

`AccumulateEngine::accumulate_dual(values, grouping0, grouping1, expr)` is the API surface. Helper `grouping_to_keys(grouping, n)` converts Grouping enum → (Vec<i32>, n_groups) for either Grouping::All (all zeros, 1 group) or Grouping::ByKey/Masked.

## Why this matters

The compiler fusion rule: `scatter_multi_phi` fuses multiple expressions per grouping. `scatter_phi_dual_target` fuses multiple groupings per expression. Together they span the compiler's scatter fusion space:

- `scatter_multi_phi([expr0, expr1, expr2], keys)` → one pass for sum/sumsq/count per group
- `scatter_phi_dual_target(expr, keys0, keys1)` → one pass for the same expression across two different segmentations

The archetype: `phi = v` with grouping0 = full scan (All), grouping1 = ByKey. One pass returns (total_sum, per_group_sums). Previously: two separate kernel launches.

## What this enables

Multi-level aggregation in a single pass. For example:
- Total revenue AND per-region revenue simultaneously
- Global mean AND per-segment mean (for within-segment z-scoring)
- Grand sum AND grouped sum (for computing group fractions without two passes)

This is the foundation for the "normalize within group" pattern: gather totals and subtotals in one kernel, then divide via map_phi.

## Generalization path

The current implementation is dual (two targets). The natural generalization is N-target:
```
scatter_phi_multi_target(phi_expr, keys: &[&[i32]], n_groups: &[usize], values)
```
This requires JIT-generating a kernel with N output arrays and N `atomicAdd` calls. The NVRTC code gen handles this naturally. The cache key becomes `"multi_target:N:phi_expr"`.

Not built yet — but the dual case proves the pattern.
