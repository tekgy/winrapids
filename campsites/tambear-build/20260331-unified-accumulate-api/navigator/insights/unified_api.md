# Unified Accumulate API: The Four-Menu Interface

## What was built

`tambear::accumulate` — the single dispatch surface for all grouping patterns.

```rust
engine.accumulate(values, grouping, expr, op) → AccResult
engine.accumulate_multi(values, grouping, &[phi1, phi2, ...]) → Vec<Vec<f64>>
```

The 4 parameters ARE the 4 menus from the unification doc:
- `values` — the data (after any gather/addressing)
- `Grouping` — WHERE to write / how to partition (dispatch axis)
- `Expr` — WHAT to compute per element (the lift function)
- `Op` — HOW to combine (the monoid)

## The dispatch table (current state)

| Grouping | Op | Routes to | Status |
|----------|-----|-----------|--------|
| `All` | `Add` | `scatter_phi(keys=0, n=1)` | ✓ |
| `All` | `ArgMin` | `ReduceOp::argmin` | ✓ |
| `All` | `ArgMax` | `ReduceOp::argmax` | ✓ |
| `ByKey` | `Add` | `ScatterJit::scatter_phi` | ✓ |
| `Masked` | `Add` | `ScatterJit::scatter_phi_masked` | ✓ |
| `ByKey` (multi) | `Add` | `ScatterJit::scatter_multi_phi` | ✓ (accumulate_multi) |
| `Masked` (multi) | `Add` | `ScatterJit::scatter_multi_phi_masked` | ✓ (accumulate_multi) |
| `Prefix` | any | winrapids-scan | todo — named |
| `Segmented` | any | winrapids-scan | todo — named |
| `Tiled` | any | winrapids-tiled TiledEngine | todo — named |
| `Windowed` | any | prefix subtraction trick | todo — named |

`todo — named` means: the dispatch arm exists with a clear error message pointing to the right primitive. Wiring it is a one-liner when the backends are connected.

## The design insight: `&mut self` is fine

The underlying primitives (ScatterJit, ReduceOp) take `&mut self` because they own CUDA state (compiled kernels, streams). The `AccumulateEngine` takes `&mut self` too. This is correct — accumulation IS stateful when it caches compiled kernels.

If shared/concurrent access is needed later: wrap in `Mutex<AccumulateEngine>`.

## The naming: four menus, not nine primitives

Old framing: "9 primitives" (reduce, scatter, scan, tiled, filter, gather, ...)

New framing: ONE primitive (`accumulate`) with 4 parameters. The 9 primitives are the SAME primitive with different `Grouping` values.

This matters because:
1. **User API**: `accumulate(data, ByKey, v*v, Add)` reads as "sum of squared values per group"
2. **Compiler**: detects when two accumulates share `(data, expr)` → fuses into one kernel pass
3. **Specialist registry**: each specialist IS a `(Grouping, Expr, Op)` triple
4. **Future extension**: adding `Grouping::Bidirectional` or `Grouping::Sparse` slots into the existing dispatch table

## What accumulate_multi IS

`accumulate_multi(values, grouping, &[phi1, phi2, phi3])` is the explicit compiler fusion primitive. Three accumulates with identical `(data, grouping)` but different `Expr` → one kernel.

This is decomposable accumulation's lift fusion:
```
lift(x) = (phi1(x), phi2(x), phi3(x))  // 3-dimensional lift
⊕ = (+, +, +)                           // pointwise addition
```

The `scatter_multi_phi` kernel generates this in one CUDA kernel with three atomicAdds.

The cost of not using `accumulate_multi`: 3× memory passes. With N=3, that's 67% wasted bandwidth.

## The natural next step: `Grouping::Tiled` wiring

```rust
Grouping::Tiled { m, n, k } => {
    let tiled = TiledEngine::new(gpu.clone());
    let c = tiled.run(op_for_tiled, values, b_values, m, n, k)?;
    Ok(AccResult::Matrix(c, m, n))
}
```

One problem: `Grouping::Tiled` needs TWO input matrices (A and B), not one `values`. The current `accumulate(values, ...)` signature assumes a single input. The tiled case is `accumulate(A, B, grouping, expr, op)`.

Options:
1. Add `accumulate_tiled(a, b, m, n, k, op)` as a separate method (simple)
2. Change `values` to an enum `Input::Single(...)` / `Input::Pair(...)` (unified but complex)
3. Keep `TiledEngine` as the direct user API (it's clean enough already)

The campsite answer: option 3 for now. `TiledEngine::run` IS the unified interface for tiled ops. The `accumulate` API doesn't need to absorb it today.
