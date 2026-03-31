//! Parallel reduction primitives: argmin, argmax.
//!
//! Two-phase design:
//! 1. **GPU phase**: each block reduces its 256 elements to one (value, index) pair,
//!    writing n_blocks results to device memory.
//! 2. **Host phase**: reduce the n_blocks pairs to the global result.
//!
//! Phase 2 is O(n_blocks) = O(n/256) — for 2M rows: 8K pairs, microseconds on CPU.
//! The GPU phase is O(n/n_blocks) = O(256) depth per block, fully parallel.
//!
//! **NaN handling**: NaN rows are excluded. Argmin maps NaN → +∞ (neutral);
//! argmax maps NaN → −∞ (neutral). If all values are NaN, returns the sentinel
//! `(f64::INFINITY, usize::MAX)` for argmin or `(f64::NEG_INFINITY, usize::MAX)` for argmax.
//!
//! **Tie-breaking**: equal values → smaller index wins (deterministic).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const REDUCE_BLOCK_SIZE: u32 = 256;
/// Shared memory per block: REDUCE_BLOCK_SIZE doubles (svals) + REDUCE_BLOCK_SIZE ints (sidxs).
const REDUCE_SMEM_BYTES: u32 = REDUCE_BLOCK_SIZE * (8 + 4);

/// GPU argmin / argmax operator.
///
/// Kernels are compiled lazily on first use and cached for reuse.
pub struct ReduceOp {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    argmin_fn: Option<CudaFunction>,
    argmax_fn: Option<CudaFunction>,
}

impl ReduceOp {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::on_device(0)
    }

    pub fn on_device(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream, argmin_fn: None, argmax_fn: None })
    }

    /// Argmin over a host slice. Returns `(min_value, index_of_min)`.
    /// NaN rows are excluded. Empty input → `(f64::INFINITY, usize::MAX)`.
    pub fn argmin(&mut self, values: &[f64]) -> Result<(f64, usize), Box<dyn std::error::Error>> {
        if values.is_empty() {
            return Ok((f64::INFINITY, usize::MAX));
        }
        let values_dev = self.stream.clone_htod(values)?;
        self.argmin_gpu(&values_dev, values.len())
    }

    /// Argmax over a host slice. Returns `(max_value, index_of_max)`.
    /// NaN rows are excluded. Empty input → `(f64::NEG_INFINITY, usize::MAX)`.
    pub fn argmax(&mut self, values: &[f64]) -> Result<(f64, usize), Box<dyn std::error::Error>> {
        if values.is_empty() {
            return Ok((f64::NEG_INFINITY, usize::MAX));
        }
        let values_dev = self.stream.clone_htod(values)?;
        self.argmax_gpu(&values_dev, values.len())
    }

    /// Argmin on GPU-resident data. Synchronizes before returning.
    pub fn argmin_gpu(
        &mut self,
        values: &CudaSlice<f64>,
        n: usize,
    ) -> Result<(f64, usize), Box<dyn std::error::Error>> {
        self.ensure_kernels()?;
        let n_blocks = ((n as u32) + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
        let mut block_vals: CudaSlice<f64> = self.stream.alloc_zeros(n_blocks as usize)?;
        let mut block_idxs: CudaSlice<i32> = self.stream.alloc_zeros(n_blocks as usize)?;
        let n_i32 = n as i32;
        let f = self.argmin_fn.as_ref().unwrap();
        let cfg = reduce_launch_cfg(n);
        unsafe {
            self.stream.launch_builder(f)
                .arg(values).arg(&mut block_vals).arg(&mut block_idxs).arg(&n_i32)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;
        let bvals = self.stream.clone_dtoh(&block_vals)?;
        let bidxs = self.stream.clone_dtoh(&block_idxs)?;
        Ok(final_argmin(&bvals, &bidxs))
    }

    /// Argmax on GPU-resident data. Synchronizes before returning.
    pub fn argmax_gpu(
        &mut self,
        values: &CudaSlice<f64>,
        n: usize,
    ) -> Result<(f64, usize), Box<dyn std::error::Error>> {
        self.ensure_kernels()?;
        let n_blocks = ((n as u32) + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
        let mut block_vals: CudaSlice<f64> = self.stream.alloc_zeros(n_blocks as usize)?;
        let mut block_idxs: CudaSlice<i32> = self.stream.alloc_zeros(n_blocks as usize)?;
        let n_i32 = n as i32;
        let f = self.argmax_fn.as_ref().unwrap();
        let cfg = reduce_launch_cfg(n);
        unsafe {
            self.stream.launch_builder(f)
                .arg(values).arg(&mut block_vals).arg(&mut block_idxs).arg(&n_i32)
                .launch(cfg)?;
        }
        self.stream.synchronize()?;
        let bvals = self.stream.clone_dtoh(&block_vals)?;
        let bidxs = self.stream.clone_dtoh(&block_idxs)?;
        Ok(final_argmax(&bvals, &bidxs))
    }

    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }
    pub fn ctx(&self) -> &Arc<CudaContext> { &self.ctx }

    fn ensure_kernels(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.argmin_fn.is_none() {
            let opts = CompileOptions { arch: Some("sm_120"), ..Default::default() };
            let ptx = compile_ptx_with_opts(REDUCE_OP_SOURCE, opts)?;
            let module = self.ctx.load_module(ptx)?;
            self.argmin_fn = Some(module.load_function("argmin_f64")?);
            self.argmax_fn = Some(module.load_function("argmax_f64")?);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Host-side final reduce over block results
// ---------------------------------------------------------------------------

fn final_argmin(bvals: &[f64], bidxs: &[i32]) -> (f64, usize) {
    let (best_val, best_idx) = bvals.iter().zip(bidxs.iter())
        .fold((f64::INFINITY, i32::MAX), |(bv, bi), (&v, &i)| {
            if v < bv || (v == bv && i < bi) { (v, i) } else { (bv, bi) }
        });
    if best_idx == i32::MAX { (f64::INFINITY, usize::MAX) } else { (best_val, best_idx as usize) }
}

fn final_argmax(bvals: &[f64], bidxs: &[i32]) -> (f64, usize) {
    let (best_val, best_idx) = bvals.iter().zip(bidxs.iter())
        .fold((f64::NEG_INFINITY, i32::MAX), |(bv, bi), (&v, &i)| {
            if v > bv || (v == bv && i < bi) { (v, i) } else { (bv, bi) }
        });
    if best_idx == i32::MAX { (f64::NEG_INFINITY, usize::MAX) } else { (best_val, best_idx as usize) }
}

// ---------------------------------------------------------------------------
// CUDA source — two kernels in one compilation unit
// ---------------------------------------------------------------------------

const REDUCE_OP_SOURCE: &str = r#"
// Two-phase argmin/argmax reduction.
//
// Each block reduces blockDim.x elements to one (value, index) pair.
// NaN inputs are replaced by the neutral element (no NaN contamination in result).
// Tie-breaking: equal values → smaller index wins.
//
// Shared memory layout:
//   [0 .. blockDim.x*8)        : svals (double[blockDim.x])
//   [blockDim.x*8 .. *12)      : sidxs (int[blockDim.x])
//
// block_vals[blockIdx.x] and block_idxs[blockIdx.x] receive the block's local result.
// Host reduces the n_blocks block results to the global (value, index) pair.

extern "C" __global__ void argmin_f64(
    const double* __restrict__ values,
    double* __restrict__ block_vals,
    int* __restrict__ block_idxs,
    int n
) {
    extern __shared__ double svals[];
    int* sidxs = (int*)(svals + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    double pos_inf = __longlong_as_double(0x7FF0000000000000ULL);  // +inf

    if (gid < n) {
        double v = values[gid];
        if (v != v) { v = pos_inf; }  // NaN → +inf (excluded from argmin)
        svals[tid] = v;
        sidxs[tid] = (v == pos_inf) ? 2147483647 : gid;
    } else {
        svals[tid] = pos_inf;
        sidxs[tid] = 2147483647;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double va = svals[tid], vb = svals[tid + s];
            int   ia = sidxs[tid], ib = sidxs[tid + s];
            if (vb < va || (vb == va && ib < ia)) {
                svals[tid] = vb;
                sidxs[tid] = ib;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = svals[0];
        block_idxs[blockIdx.x] = sidxs[0];
    }
}

extern "C" __global__ void argmax_f64(
    const double* __restrict__ values,
    double* __restrict__ block_vals,
    int* __restrict__ block_idxs,
    int n
) {
    extern __shared__ double svals[];
    int* sidxs = (int*)(svals + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    double neg_inf = __longlong_as_double(0xFFF0000000000000ULL);  // -inf

    if (gid < n) {
        double v = values[gid];
        if (v != v) { v = neg_inf; }  // NaN → -inf (excluded from argmax)
        svals[tid] = v;
        sidxs[tid] = (v == neg_inf) ? 2147483647 : gid;
    } else {
        svals[tid] = neg_inf;
        sidxs[tid] = 2147483647;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            double va = svals[tid], vb = svals[tid + s];
            int   ia = sidxs[tid], ib = sidxs[tid + s];
            if (vb > va || (vb == va && ib < ia)) {
                svals[tid] = vb;
                sidxs[tid] = ib;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = svals[0];
        block_idxs[blockIdx.x] = sidxs[0];
    }
}
"#;

fn reduce_launch_cfg(n: usize) -> LaunchConfig {
    let n_blocks = ((n as u32) + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (REDUCE_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: REDUCE_SMEM_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmin_basic() {
        let mut op = ReduceOp::new().unwrap();
        let values = vec![5.0, 3.0, 8.0, 1.0, 6.0];
        let (val, idx) = op.argmin(&values).unwrap();
        assert_eq!(val, 1.0);
        assert_eq!(idx, 3);
    }

    #[test]
    fn argmax_basic() {
        let mut op = ReduceOp::new().unwrap();
        let values = vec![5.0, 3.0, 8.0, 1.0, 6.0];
        let (val, idx) = op.argmax(&values).unwrap();
        assert_eq!(val, 8.0);
        assert_eq!(idx, 2);
    }

    #[test]
    fn argmin_nan_excluded() {
        let mut op = ReduceOp::new().unwrap();
        let values = vec![f64::NAN, 3.0, f64::NAN, 7.0];
        let (val, idx) = op.argmin(&values).unwrap();
        assert_eq!(val, 3.0);
        assert_eq!(idx, 1);
    }

    #[test]
    fn argmax_nan_excluded() {
        let mut op = ReduceOp::new().unwrap();
        let values = vec![f64::NAN, 3.0, f64::NAN, 7.0];
        let (val, idx) = op.argmax(&values).unwrap();
        assert_eq!(val, 7.0);
        assert_eq!(idx, 3);
    }

    #[test]
    fn argmin_all_nan_returns_sentinel() {
        let mut op = ReduceOp::new().unwrap();
        let values = vec![f64::NAN, f64::NAN];
        let (val, idx) = op.argmin(&values).unwrap();
        assert!(val.is_infinite() && val > 0.0, "sentinel val should be +inf");
        assert_eq!(idx, usize::MAX);
    }

    #[test]
    fn argmin_empty_returns_sentinel() {
        let mut op = ReduceOp::new().unwrap();
        let (val, idx) = op.argmin(&[]).unwrap();
        assert!(val.is_infinite() && val > 0.0);
        assert_eq!(idx, usize::MAX);
    }

    #[test]
    fn argmin_tie_breaks_by_smaller_index() {
        let mut op = ReduceOp::new().unwrap();
        // min value 1.0 appears at indices 1 and 4 — smaller index wins
        let values = vec![5.0, 1.0, 8.0, 3.0, 1.0];
        let (val, idx) = op.argmin(&values).unwrap();
        assert_eq!(val, 1.0);
        assert_eq!(idx, 1, "index 1 < index 4 — tie-break to 1");
    }

    #[test]
    fn argmax_tie_breaks_by_smaller_index() {
        let mut op = ReduceOp::new().unwrap();
        let values = vec![8.0, 3.0, 8.0, 1.0];
        let (val, idx) = op.argmax(&values).unwrap();
        assert_eq!(val, 8.0);
        assert_eq!(idx, 0, "index 0 < index 2 — tie-break to 0");
    }

    #[test]
    fn argmin_multi_block() {
        // 512 elements = 2 blocks of 256. Min at position 300 (second block).
        let mut op = ReduceOp::new().unwrap();
        let mut values: Vec<f64> = (0..512).map(|i| i as f64).collect();
        values[300] = -999.0;
        let (val, idx) = op.argmin(&values).unwrap();
        assert_eq!(val, -999.0);
        assert_eq!(idx, 300);
    }

    #[test]
    fn argmax_multi_block() {
        let mut op = ReduceOp::new().unwrap();
        let mut values: Vec<f64> = (0..512).map(|i| i as f64).collect();
        values[42] = 9999.0;
        let (val, idx) = op.argmax(&values).unwrap();
        assert_eq!(val, 9999.0);
        assert_eq!(idx, 42);
    }

    #[test]
    fn argmin_single_element() {
        let mut op = ReduceOp::new().unwrap();
        let (val, idx) = op.argmin(&[42.0]).unwrap();
        assert_eq!(val, 42.0);
        assert_eq!(idx, 0);
    }
}
