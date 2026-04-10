# Gradient as Primitive: The Scatter/Gather Duality

## The Structural Claim

Every backward pass through a tambear primitive IS another tambear primitive.

This is not a design choice — it's a theorem. Gradient propagation is a specific composition of the same operations that comprise forward computation, just run in a different direction.

## The Scatter/Gather Adjoint Pair

**Forward — scatter**: `output[g] += phi(values[i])` for all i where `keys[i] == g`

**Backward — gather + map**:
```
grad_values[i] = d_phi(values[i]) * grad_output[keys[i]]
```

Which decomposes as:
1. Gather: `upstream_i = grad_output[keys[i]]` — indexed read into the upstream gradient
2. Map: `grad_values[i] = d_phi_expr(v_i) * upstream_i` — element-wise multiply

Both are tambear primitives. The gather is GatherOp. The multiply is `map_phi2("a * b", upstream_gathered, d_phi_values)`.

**Forward — gather**: `output[i] = values[keys[i]]`

**Backward — scatter**:
```
grad_values[keys[i]] += grad_output[i]
```

This IS scatter_phi("v", ...) — scatter sum of upstream gradients. Already implemented.

**Scatter and gather are each other's adjoints.**

## The Full Backward Pass Mapping

| Forward primitive | Backward primitive |
|-------------------|--------------------|
| `scatter_phi(phi, keys, values)` | `gather(grad_output, keys)` → `map_phi2(d_phi, upstream, values)` |
| `gather(values, keys)` | `scatter_phi("v", keys, grad_output)` |
| `map_phi(phi, values)` | `map_phi2(d_phi, grad_output, values)` |
| `map_phi2(phi, a, b)` | `map_phi2(d_phi_da, grad_out, b)` + `map_phi2(d_phi_db, grad_out, a)` |
| `TiledEngine::run(DotProduct, X, W)` | `TiledEngine::run(DotProduct, X.T, grad_out)` for dW; `TiledEngine::run(DotProduct, grad_out, W.T)` for dX |
| `accumulate(All, "v*v", Add)` — MSE | `map_phi2("2.0 * a * b / {n}", grad_scalar, residuals)` |

## The New Primitive That Unlocks This

`map_phi2(phi_expr, a, b)` — two-input element-wise map. Built today.

Variables `a` and `b` in phi_expr, both f64 arrays. Examples:
- `"a * b"` — gradient chain rule (upstream × local derivative)
- `"a - lr * b"` — SGD parameter update (baked-in lr)
- `"b > 0.0 ? a : 0.0"` — ReLU backward (a=upstream, b=pre-activation)
- `"a / (b + 1e-8)"` — LayerNorm backward (safe divide)

## The Training Loop as Primitive Composition

Linear regression training iteration, zero new primitives:

```
// Forward
z = TiledEngine::run(DotProduct, X, W, n, k, d)   // n×k output
residuals = map_phi2("a - b", z, y)                // z - y, element-wise

// Loss (optional, for monitoring)
loss = accumulate(residuals, All, "v*v", Add) / n

// Backward through MSE loss
dL_dz = map_phi2("2.0 * a / {n}", residuals, residuals)  // 2*(z-y)/n, bake n

// Backward through dot product
dW = TiledEngine::run(DotProduct, X_T, dL_dz, d, k, n)  // d×k gradient

// Parameter update (SGD)
W_new = map_phi2("a - {lr} * b", W, dW)  // bake lr
```

All of this is TODAY possible with existing primitives. Five primitive calls for one training iteration.

## Why This Matters

Traditional ML frameworks (PyTorch, JAX) separate the "forward computation" engine from the "gradient computation" engine. Autodiff is a separate system that traces operations and builds a backward graph.

The tambear insight: **for the primitive set we've built, no separate system is needed**. The gradient of each primitive is expressible within the same primitive set. Differentiation is not a feature to be added — it's already present in the structure.

This doesn't mean tambear will replace full autodiff frameworks. Complex architectures with dynamic control flow need full AD. But for the class of operations tambear targets — scatter-based aggregation, tiled matrix operations, element-wise maps — the backward pass is just the forward pass run backwards, and the primitives are already there.

## The Next Piece

The missing link: `scatter_phi_backward` — a helper that takes `(phi_expr, d_phi_expr, keys, grad_output, values)` and returns `grad_values`. This wraps the gather + map_phi2 pattern into a single call with a clear gradient semantics.

Not built yet. The insight is here. The pieces (GatherOp + map_phi2) are implemented. The wrapper just makes the gradient semantics explicit.

## The KMeans Convergence Insight (Adjacent)

Also discovered today: f32 atomicAdd over large arrays has non-deterministic rounding depending on GPU thread scheduling. Distance-based convergence for KMeans (rel_change < tol) is not reproducible across runs with different GPU warmup states.

Fix: label stability — check if any label changed between iterations. Labels are discrete (u32), immune to floating-point noise. If `curr_labels == prev_labels`, the centroids are guaranteed fixed points regardless of f32 arithmetic order. This is the correct convergence criterion for KMeans.
