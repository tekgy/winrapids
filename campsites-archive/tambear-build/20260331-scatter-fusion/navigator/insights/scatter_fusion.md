# Scatter Fusion: The Compiler's First Working Primitive

## What was built

`ScatterJit::scatter_multi_phi` — N scatter_phi calls fused into one kernel pass.

```rust
let results = jit.scatter_multi_phi(
    &[PHI_SUM, PHI_SUM_SQ, PHI_COUNT],
    keys, values, None, n_groups,
)?;
// results[0] = sums, results[1] = sum_sqs, results[2] = counts
// One memory pass. Three atomicAdds per element. Identical to scatter_stats.
```

## Why this is the compiler's fusion operation

The decomposable accumulation campsite identified the fusion rule:

> If (key, ⊕) are the same, multiple (lift_1, lift_2, ..., lift_k) fuse into one
> lift: (lift_1, ..., lift_k) with pointwise ⊕.

`scatter_multi_phi` IS this fusion rule, executable:
- Same key: ✓ (same keys array)
- Same monoid (ℝ, +): ✓ (always atomicAdd)
- Different lifts: ✓ (phi_exprs[i] is lift_i)

The generated CUDA kernel is the fused (lift_1, ..., lift_k) with pointwise +:

```cuda
if (gid < n) {
    int g = keys[gid]; double v = values[gid]; double r = refs[g];
    atomicAdd(&out0[g], (v));       // lift_0 = PHI_SUM
    atomicAdd(&out1[g], (v * v));   // lift_1 = PHI_SUM_SQ
    atomicAdd(&out2[g], (1.0));     // lift_2 = PHI_COUNT
}
```

`scatter_stats` is NOT a special case — `scatter_stats` IS `scatter_multi_phi([v, v*v, 1.0])`.
The generalization has zero cost: the CUDA generation is a loop.

## The impl decision: variable-arg dispatch via slice pattern match

The hard part: CUDA kernels have fixed parameter counts, but the compiler generates
kernels for any N. The cudarc `launch_builder` API is a typed builder that can't
accept a runtime-length arg list.

Solution: generate the kernel with exactly N named parameters (`out0..outN-1`),
and match on the output slice length to pick the right builder chain:

```rust
match outputs.as_mut_slice() {
    [o0] => stream.launch_builder(f).arg(keys)...arg(o0).arg(&n),
    [o0, o1] => stream.launch_builder(f).arg(keys)...arg(o0).arg(o1).arg(&n),
    // etc. up to 8
}
```

This is verbose but:
1. Correct — each arm has the right static type for cudarc
2. Inlineable — the match is O(1) branch on a known N
3. Compile-time checked — if cudarc's arg types ever change, the Rust compiler catches it

The limit of 8 is practical, not fundamental. Pipeline statistics rarely exceed 5-6
aggregates per groupby key. If N>8 is ever needed, the pattern extends trivially.

## The cache key design

Single-phi cache: `phi_expr` (the expression string itself)
Multi-phi cache:  `"multi:{n}:{phi0}|{phi1}|..."` (prefixed, can't collide)

One HashMap holds both. The prefix ensures no collision even if a phi expression
contains a `|` character (unlikely, illegal CUDA, but safe by construction).

## What this enables

The compiler can now express any of the following as a single kernel pass:

| Pattern | phi_exprs |
|---|---|
| GroupBy statistics | `[v, v*v, 1.0]` → mean, variance, count |
| Weighted statistics | `[v*w, w, 1.0]` → weighted mean, total weight, count |
| Centered moments | `[v-r, (v-r)*(v-r)]` (two-pass: first pass gives r=means) |
| Log-likelihood components | `[log(v), v, 1.0]` → log-sum, sum, count |
| Custom 5-stat pack | any 5 expressions → 5 outputs, 1 pass |

Each of these would have required 2-5 kernel launches before; now it's 1.
The memory bandwidth reduction = (N-1)/N. For N=3 (the common case): 67% reduction.

## The relationship to scatter_stats

`HashScatterEngine::scatter_stats` computes (sum, sum_sq, count) with a hardcoded
3-atomicAdd kernel. It's correct and fast.

`scatter_multi_phi([v, v*v, 1.0])` computes the same thing via JIT generation.

The JIT version:
- Is ~40ms slower on FIRST call (NVRTC compilation)
- Is identical on subsequent calls (cached PTX)
- Is MORE GENERAL: arbitrary phi expressions, not just (sum, sum_sq, count)
- Is the foundation the compiler needs to generate novel fusions at query time

`scatter_stats` can now be understood as: `scatter_multi_phi` specialized to the
standard triple, with the JIT compilation cost pre-paid.

## The hierarchy of fusion

```
Level 0: N separate scatter_phi calls       → N kernel launches, N memory passes
Level 1: scatter_multi_phi([phi_0, ..., phi_{N-1}])  → 1 kernel launch, 1 memory pass
Level 2: (future) scatter_multi_phi + shared-mem privatization → 1 launch, reduced atomicAdd contention
```

Level 1 is now implemented. Level 2 (smem privatized multi-phi) would combine
the smem approach from scatter_stats_smem with the JIT phi generation.
That's the next natural build if contention becomes the bottleneck.

## What the test suite validates

1. `multi_phi_matches_three_separate_calls`: fused == 3 × separate, bitwise
2. `multi_phi_is_scatter_stats_generalized`: concrete (sum, sum_sq, count) values
3. `multi_phi_cached`: second call reuses compiled kernel (cache.len == 1 multi entry)
4. `multi_phi_two_outputs`: N=2 works correctly (exercises the 2-arm match)
