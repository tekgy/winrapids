# Tiled Accumulate GPU Launch: Wiring the Last Mile

## The gap

`winrapids-tiled` has:
- `TiledOp` trait: 5 operators (GEMM, covariance, L2 distance, outer product, FlashAttention)
- `generate_tiled_kernel(op)` → CUDA source string
- `TiledCache`: in-memory cache of generated sources

It does NOT have: anything that actually runs on a GPU.

The kernel exists. The cable isn't plugged in.

## Why this is the highest-unlock gap

Plugging in the cable enables ALL of:
- **GEMM** (DotProductOp): matrix multiply, neural network layers
- **Covariance** (CovarianceOp): PCA, factor models, fused centering
- **L2 Distance** (DistanceOp): KNN, clustering, similarity search
- **FlashAttention** (SoftmaxWeightedOp): attention scores, query-key scaling

Every one of these is CURRENTLY IMPLEMENTED at the algorithm level. They need one new file.

## The implementation

`TiledEngine` in `src/dispatch.rs`:
- Takes `Arc<dyn TamGpu>` — uses `tam-gpu` as the backend
- `run(op, A, B, M, N, K)` → `Vec<f64>`: end-to-end GPU execution
- Kernel cache: `Mutex<HashMap<String, Arc<Kernel>>>` — Arc so we can clone out and release the lock before dispatch

The kernel parameter change: `int M, int N, int K` → `const int* dims`:
- Lets us pass [M, N, K] as a single 3-element i32 buffer
- Reduces buffer count to 4 (A, B, C, dims) — well within the 8-buffer limit
- Cache key already incorporates the full CUDA source, so no collision risk

Grid/block:
```
grid  = (ceil(N/16), ceil(M/16), 1)
block = (TILE_N=16, TILE_M=16, 1)  → 256 threads per block
```
Each thread block computes one 16×16 output tile of C.

## The accumulate unification, made executable

The decomposable accumulation theorem says:
```
C(S) = extract(⊕ᵢ lift(sᵢ))
```

For tiled ops, the grouping is 2D: output element (i,j) accumulates over the K dimension.

```
accumulate(A, grouping=(row i, col j), expr=lift(a,b), op=⊕)
```

Where:
- `DotProductOp`: lift=(a*b), ⊕=+, extract=identity
- `DistanceOp`:   lift=(a-b)², ⊕=+, extract=sqrt (or keep squared for KNN)
- `CovarianceOp`: lift=(a-μₐ)(b-μᵦ), ⊕=+, extract=÷(n-1)
- `SoftmaxWeightedOp`: lift=(score,value), ⊕=online_softmax_merge, extract=weighted_sum/exp_sum

All four are `accumulate` with a 2D grouping and different (lift, ⊕, extract). The tiled engine executes ALL of them with the same dispatch path — only the operator changes.

## Why TamGpu, not cudarc directly

`winrapids-tiled` is the first NEW crate that uses `tam-gpu` as intended: as the abstraction layer. It doesn't know or care whether the backend is CUDA or CPU. The `TiledEngine` works on both:
- On CUDA: NVRTC compiles the kernel, cudarc dispatches it
- On CPU: the `CpuBackend` would need a `tiled_accumulate` entry — OR we add a CPU fallback in `dispatch.rs`

The CPU fallback matters for:
- Development without GPU
- Testing correctness against a reference impl
- Environments where CUDA isn't available

The CPU reference implementation is a triple-nested loop — trivial to add.

## What becomes possible next

Once the GPU launch is wired:
1. **Distance → clustering**: `DistanceOp` computes the full distance matrix. K-means = scatter (centroid update) composed with tiled (distance to centroids). This is the `kmeans` module in tambear.
2. **Softmax → attention**: `SoftmaxWeightedOp` IS FlashAttention one-pass. Full attention with one kernel.
3. **Covariance → PCA**: eigendecomposition of the covariance matrix via power iteration (scan primitive), seeded by the covariance matrix (tiled primitive).

The tiled GPU launch is the keystone: everything above it is scaffolded and waiting.
