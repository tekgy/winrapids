//! GPU dispatch for tiled accumulation operators.
//!
//! [`TiledEngine`] wraps any [`TamGpu`] backend and provides end-to-end
//! execution of tiled operators: generate source (CUDA or WGSL) → compile →
//! dispatch → download result.
//!
//! ## Kernel parameter convention
//!
//! The `tiled_accumulate` kernel takes 4 buffers:
//! - `A`: M×K input matrix, row-major, f64
//! - `B`: K×N input matrix, row-major, f64
//! - `C`: M×N output matrix, row-major, f64 (zero-initialised by alloc)
//! - `dims`: [M, N, K] as i32
//!
//! Grid/block: `(ceil(N/TILE_N), ceil(M/TILE_M), 1)` × `(TILE_N, TILE_M, 1)`.
//! Each thread block computes one 16×16 output tile.
//!
//! ## Kernel cache
//!
//! Compiled kernels are cached in a `Mutex<HashMap<key, Arc<Kernel>>>`.
//! The Arc lets us clone the kernel handle out of the map before releasing
//! the lock — so dispatch doesn't hold the mutex during execution.

use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use tam_gpu::{TamGpu, TamResult, TamGpuError, ShaderLang, upload, download};

use crate::ops::TiledOp;
use crate::cache::cache_key;
use crate::engine::{generate_tiled_kernel, generate_tiled_kernel_wgsl};

const TILE_M: u32 = 16;
const TILE_N: u32 = 16;

/// GPU execution engine for tiled accumulation operators.
///
/// Compiles, caches, and dispatches `tiled_accumulate` kernels for any [`TiledOp`].
/// Compatible with any [`TamGpu`] backend: CUDA (NVRTC + cudarc) or CPU fallback.
///
/// # Example
/// ```no_run
/// use std::sync::Arc;
/// use tam_gpu::detect;
/// use winrapids_tiled::{TiledEngine, DotProductOp};
///
/// let gpu = Arc::from(detect());
/// let engine = TiledEngine::new(gpu);
///
/// // A (2×3) × B (3×2) → C (2×2)
/// let a = vec![1.0f64, 2.0, 3.0,  4.0, 5.0, 6.0];
/// let b = vec![7.0f64, 8.0,  9.0, 10.0,  11.0, 12.0];
/// let c = engine.run(&DotProductOp, &a, &b, 2, 2, 3).unwrap();
/// // c[0] = 1*7 + 2*9 + 3*11 = 58,  c[1] = 1*8 + 2*10 + 3*12 = 64
/// // c[2] = 4*7 + 5*9 + 6*11 = 139, c[3] = 4*8 + 5*10 + 6*12 = 154
/// ```
pub struct TiledEngine {
    gpu: Arc<dyn TamGpu>,
    cache: Mutex<HashMap<String, Arc<tam_gpu::Kernel>>>,
}

impl TiledEngine {
    /// Create a new engine backed by `gpu`.
    pub fn new(gpu: Arc<dyn TamGpu>) -> Self {
        Self { gpu, cache: Mutex::new(HashMap::new()) }
    }

    /// Get a compiled kernel for `op`, compiling if not already cached.
    ///
    /// Returns `Arc<Kernel>` so the caller holds a reference independent of
    /// the cache lock — the mutex is not held during dispatch.
    fn get_or_compile(&self, op: &dyn TiledOp) -> TamResult<Arc<tam_gpu::Kernel>> {
        let key = cache_key(op);

        // Fast path: cache hit (lock briefly, clone Arc, release).
        {
            let cache = self.cache.lock().unwrap();
            if let Some(k) = cache.get(&key) {
                return Ok(Arc::clone(k));
            }
        }

        // Slow path: compile (~40 ms on first CUDA call; near-instant for WGSL/CPU).
        let source = match self.gpu.shader_lang() {
            ShaderLang::Cuda => generate_tiled_kernel(op),
            _                => generate_tiled_kernel_wgsl(op),
        };
        let kernel = self.gpu.compile(&source, "tiled_accumulate")
            .map_err(|e| TamGpuError::Compile(format!("{op}: {e}", op = op.name())))?;
        let arc = Arc::new(kernel);

        // Insert under the key (another thread may have beaten us; keep first winner).
        let mut cache = self.cache.lock().unwrap();
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry))
    }

    /// Execute `op` on GPU: A (M×K) × B (K×N) → C (M×N).
    ///
    /// `a` and `b` are row-major f64 slices. Returns C row-major.
    ///
    /// On CUDA: first call compiles via NVRTC (~40 ms); subsequent calls hit the cache.
    /// On CPU: the CPU backend executes a reference implementation directly.
    pub fn run(
        &self,
        op: &dyn TiledOp,
        a: &[f64],
        b: &[f64],
        m: usize,
        n: usize,
        k: usize,
    ) -> TamResult<Vec<f64>> {
        assert_eq!(a.len(), m * k, "A must be m×k ({m}×{k} = {}), got {}", m * k, a.len());
        assert_eq!(b.len(), k * n, "B must be k×n ({k}×{n} = {}), got {}", k * n, b.len());

        let kernel = self.get_or_compile(op)?;

        // CUDA kernels operate in f64; WGSL kernels operate in f32.
        // Upload/download must match the element type the kernel expects.
        let grid_x = (n as u32 + TILE_N - 1) / TILE_N;
        let grid_y = (m as u32 + TILE_M - 1) / TILE_M;
        let dim_buf = upload::<i32>(&*self.gpu, &[m as i32, n as i32, k as i32])?;

        if matches!(self.gpu.shader_lang(), ShaderLang::Cuda | ShaderLang::Cpu) {
            let a_buf = upload::<f64>(&*self.gpu, a)?;
            let b_buf = upload::<f64>(&*self.gpu, b)?;
            let c_buf = self.gpu.alloc(m * n * std::mem::size_of::<f64>())?;
            self.gpu.dispatch(&*kernel, [grid_x, grid_y, 1], [TILE_N, TILE_M, 1],
                &[&a_buf, &b_buf, &c_buf, &dim_buf], 0)?;
            self.gpu.sync()?;
            download::<f64>(&*self.gpu, &c_buf, m * n)
        } else {
            // f32 path (WGSL backends: Vulkan, Metal, DX12)
            let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
            let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
            let a_buf = upload::<f32>(&*self.gpu, &a_f32)?;
            let b_buf = upload::<f32>(&*self.gpu, &b_f32)?;
            let c_buf = self.gpu.alloc(m * n * std::mem::size_of::<f32>())?;
            self.gpu.dispatch(&*kernel, [grid_x, grid_y, 1], [TILE_N, TILE_M, 1],
                &[&a_buf, &b_buf, &c_buf, &dim_buf], 0)?;
            self.gpu.sync()?;
            let c_f32: Vec<f32> = download::<f32>(&*self.gpu, &c_buf, m * n)?;
            Ok(c_f32.iter().map(|&x| x as f64).collect())
        }
    }

    /// The shader language of the underlying GPU backend.
    pub fn shader_lang(&self) -> ShaderLang {
        self.gpu.shader_lang()
    }

    /// Execute a pre-generated multi-output tiled kernel (e.g. `ManifoldMixtureOp`).
    ///
    /// `source` is a complete CUDA kernel source (used for CUDA and CPU backends).
    /// `key` is a unique cache key (e.g. BLAKE3 hash of source).
    /// `n_outputs` is the number of output values written per (i,j) position.
    ///
    /// Returns `n_outputs` distance matrices, each of shape M×N (row-major).
    /// Layout: the kernel writes `C[(i*N + j) * n_outputs + k]` for output k.
    ///
    /// # Errors
    /// Returns `Err` on WGSL backends (composite struct accumulators not yet supported
    /// in WGSL). Callers should fall back to N separate `run()` calls for WGSL.
    pub fn run_raw_mixture(
        &self,
        source: &str,
        key: &str,
        n_outputs: usize,
        a: &[f64],
        b: &[f64],
        m: usize,
        n: usize,
        k: usize,
    ) -> TamResult<Vec<Vec<f64>>> {
        assert!(n_outputs > 0, "run_raw_mixture: n_outputs must be > 0");

        // Compile / cache
        let kernel = {
            let cache = self.cache.lock().unwrap();
            if let Some(kn) = cache.get(key) {
                Arc::clone(kn)
            } else {
                drop(cache);
                let compiled = self.gpu.compile(source, "tiled_accumulate")
                    .map_err(|e| TamGpuError::Compile(format!("mixture kernel: {e}")))?;
                let arc = Arc::new(compiled);
                let mut cache = self.cache.lock().unwrap();
                let entry = cache.entry(key.to_string()).or_insert_with(|| Arc::clone(&arc));
                Arc::clone(entry)
            }
        };

        let grid_x = (n as u32 + TILE_N - 1) / TILE_N;
        let grid_y = (m as u32 + TILE_M - 1) / TILE_M;
        let dim_buf = upload::<i32>(&*self.gpu, &[m as i32, n as i32, k as i32])?;

        // Mixture kernels always use f64 (CUDA + CPU paths only)
        let a_buf = upload::<f64>(&*self.gpu, a)?;
        let b_buf = upload::<f64>(&*self.gpu, b)?;
        let c_buf = self.gpu.alloc(m * n * n_outputs * std::mem::size_of::<f64>())?;

        self.gpu.dispatch(&*kernel, [grid_x, grid_y, 1], [TILE_N, TILE_M, 1],
            &[&a_buf, &b_buf, &c_buf, &dim_buf], 0)?;
        self.gpu.sync()?;

        let c_flat = download::<f64>(&*self.gpu, &c_buf, m * n * n_outputs)?;

        // Deinterleave: c_flat[(i*n + j)*n_outputs + mk] → result[mk][i*n + j]
        let mut result = vec![vec![0.0f64; m * n]; n_outputs];
        for ij in 0..(m * n) {
            for mk in 0..n_outputs {
                result[mk][ij] = c_flat[ij * n_outputs + mk];
            }
        }
        Ok(result)
    }

    /// Cache size: number of compiled kernels held.
    pub fn cache_len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

// ---------------------------------------------------------------------------
// Tests — TiledEngine + CpuBackend (no GPU required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::*;
    use tam_gpu::CpuBackend;

    fn cpu_engine() -> TiledEngine {
        TiledEngine::new(Arc::new(CpuBackend::new()))
    }

    #[test]
    fn dot_product_2x3_times_3x2() {
        let engine = cpu_engine();
        let a = vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0,  9.0, 10.0,  11.0, 12.0];
        let c = engine.run(&DotProductOp, &a, &b, 2, 2, 3).unwrap();
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn dot_product_identity_matrix() {
        let engine = cpu_engine();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let i = vec![1.0, 0.0, 0.0, 1.0];
        let c = engine.run(&DotProductOp, &a, &i, 2, 2, 2).unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn dot_product_vector_times_matrix() {
        // Row vector (1×3) × (3×2) → (1×2)
        let engine = cpu_engine();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0,  6.0, 7.0,  8.0, 9.0];
        let c = engine.run(&DotProductOp, &a, &b, 1, 2, 3).unwrap();
        // 1*4+2*6+3*8 = 40, 1*5+2*7+3*9 = 46
        assert_eq!(c, vec![40.0, 46.0]);
    }

    #[test]
    fn l2_distance_self() {
        let engine = cpu_engine();
        // 3 points in 2D: [1,0], [0,1], [1,1]
        // A(3×2), B = A^T (2×3)
        let a = vec![1.0, 0.0,  0.0, 1.0,  1.0, 1.0];
        let b = vec![1.0, 0.0, 1.0,  0.0, 1.0, 1.0]; // transposed: 2×3
        let c = engine.run(&DistanceOp, &a, &b, 3, 3, 2).unwrap();
        // d(p0,p0)=0, d(p0,p1)=2, d(p0,p2)=1
        // d(p1,p0)=2, d(p1,p1)=0, d(p1,p2)=1
        // d(p2,p0)=1, d(p2,p1)=1, d(p2,p2)=0
        assert_eq!(c, vec![0.0, 2.0, 1.0,  2.0, 0.0, 1.0,  1.0, 1.0, 0.0]);
    }

    #[test]
    fn dot_product_scalar() {
        let engine = cpu_engine();
        let c = engine.run(&DotProductOp, &[5.0], &[7.0], 1, 1, 1).unwrap();
        assert_eq!(c, vec![35.0]);
    }

    #[test]
    fn kernel_cache_works() {
        let engine = cpu_engine();
        assert_eq!(engine.cache_len(), 0);
        engine.run(&DotProductOp, &[1.0], &[1.0], 1, 1, 1).unwrap();
        assert_eq!(engine.cache_len(), 1);
        // Second call hits cache — still 1 entry
        engine.run(&DotProductOp, &[2.0], &[3.0], 1, 1, 1).unwrap();
        assert_eq!(engine.cache_len(), 1);
        // Different op → new cache entry
        engine.run(&DistanceOp, &[1.0], &[1.0], 1, 1, 1).unwrap();
        assert_eq!(engine.cache_len(), 2);
    }

    #[test]
    fn softmax_weighted_uniform() {
        // softmax([0,0,0]) = [1/3,1/3,1/3]; B col = [3,6,9] → mean = 6
        let engine = cpu_engine();
        let a = vec![0.0, 0.0, 0.0]; // 1×3 scores
        let b = vec![3.0, 6.0, 9.0]; // 3×1 values
        let c = engine.run(&SoftmaxWeightedOp, &a, &b, 1, 1, 3).unwrap();
        assert!((c[0] - 6.0).abs() < 1e-10, "got {}", c[0]);
    }

    #[test]
    fn softmax_weighted_large_scores_stable() {
        // Scores [1000, 1001]: softmax = [sigmoid(-1), sigmoid(1)]
        // B = [0, 1] → output = sigmoid(1) ≈ 0.7311
        let engine = cpu_engine();
        let c = engine.run(&SoftmaxWeightedOp, &[1000.0, 1001.0], &[0.0, 1.0], 1, 1, 2).unwrap();
        let expected = 1.0 / (1.0 + (-1.0_f64).exp());
        assert!((c[0] - expected).abs() < 1e-10, "got {} expected {}", c[0], expected);
    }
}
