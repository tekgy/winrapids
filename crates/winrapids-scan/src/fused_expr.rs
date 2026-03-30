//! Fused expression kernels for rolling statistics.
//!
//! Computes element-wise rolling statistics from prefix scan outputs.
//! These are the "fused_expr" primitives in the compiler's IR — downstream
//! consumers of scan(data, AddOp) that apply window-bounded formulas.
//!
//! Architecture: each formula is a single-pass element-wise kernel.
//! Every thread handles one output element independently (given the prefix
//! sums already computed by the scan engine). No shared memory needed.
//!
//! Supported formulas (matching the E04 specialist registry):
//!   - "rolling_mean"   : mean over window from prefix sum
//!   - "rolling_std"    : std dev from prefix sums of x and x²
//!   - "rolling_zscore" : z-score = (x - mean) / std
//!
//! Input ordering (ordered raw device pointers per formula):
//!   rolling_mean   : [cs]          — prefix sum of x
//!   rolling_std    : [cs, cs2]     — prefix sums of x and x²
//!   rolling_zscore : [x, cs, cs2]  — raw data + both prefix sums

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};

use crate::cache::KernelCache;

/// Thread block size for fused expression kernels (element-wise, no shared memory).
const BLOCK_SIZE: usize = 256;

// --- CUDA kernel sources ---

fn rolling_mean_source() -> &'static str {
    r#"
extern "C" __global__ void rolling_mean_kernel(
    const double* __restrict__ cs,
    double* __restrict__ out,
    int n,
    int window
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    double prev = (i >= window) ? cs[i - window] : 0.0;
    double sum  = cs[i] - prev;
    int count   = (i + 1 < window) ? i + 1 : window;
    out[i] = sum / (double)count;
}
"#
}

fn rolling_std_source() -> &'static str {
    r#"
extern "C" __global__ void rolling_std_kernel(
    const double* __restrict__ cs,
    const double* __restrict__ cs2,
    double* __restrict__ out,
    int n,
    int window
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    double prev  = (i >= window) ? cs[i  - window] : 0.0;
    double prev2 = (i >= window) ? cs2[i - window] : 0.0;
    double sum   = cs[i]  - prev;
    double sum2  = cs2[i] - prev2;
    int count    = (i + 1 < window) ? i + 1 : window;
    double n_w   = (double)count;
    double mean  = sum / n_w;
    double var   = sum2 / n_w - mean * mean;
    out[i] = sqrt(fmax(0.0, var));
}
"#
}

fn rolling_zscore_source() -> &'static str {
    r#"
extern "C" __global__ void rolling_zscore_kernel(
    const double* __restrict__ x,
    const double* __restrict__ cs,
    const double* __restrict__ cs2,
    double* __restrict__ out,
    int n,
    int window
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    double prev  = (i >= window) ? cs[i  - window] : 0.0;
    double prev2 = (i >= window) ? cs2[i - window] : 0.0;
    double sum   = cs[i]  - prev;
    double sum2  = cs2[i] - prev2;
    int count    = (i + 1 < window) ? i + 1 : window;
    double n_w   = (double)count;
    double mean  = sum / n_w;
    double var   = sum2 / n_w - mean * mean;
    double std   = sqrt(fmax(0.0, var));
    out[i] = (std > 1e-12) ? (x[i] - mean) / std : 0.0;
}
"#
}

/// Output of a fused expression kernel. Owns the GPU output buffer.
/// Kept alive by `CudaKernelDispatcher::fused_outputs` until dispatch ends.
pub struct FusedExprOutput {
    data: CudaSlice<f64>,
    len: usize,
    stream: Arc<CudaStream>,
}

impl FusedExprOutput {
    /// Raw device pointer for the output buffer.
    pub fn device_ptr(&self) -> u64 {
        let (ptr, _guard) = self.data.device_ptr(&self.stream);
        ptr
    }

    /// Number of f64 elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Byte size of output buffer.
    pub fn byte_size(&self) -> u64 {
        (self.len * 8) as u64
    }

    /// Copy output to host for validation.
    pub fn to_host(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        Ok(self.stream.clone_dtoh(&self.data)?)
    }
}

/// GPU engine for fused expression kernels.
///
/// Shares CudaContext with ScanEngine via `with_context()` so scan output
/// device pointers are directly readable without cross-context copies.
pub struct FusedExprEngine {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    cache: KernelCache,
    /// Compiled functions by formula name.
    funcs: HashMap<String, CudaFunction>,
}

impl FusedExprEngine {
    /// Create on GPU 0 with its own context.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::on_device(0)
    }

    /// Create on a specific GPU device.
    pub fn on_device(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream, cache: KernelCache::new(), funcs: HashMap::new() })
    }

    /// Create sharing an existing CUDA context + stream.
    ///
    /// Use this when FusedExprEngine will read scan outputs: scan outputs are
    /// allocated in `ctx` and are directly accessible only in the same context.
    pub fn with_context(ctx: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
        Self { ctx, stream, cache: KernelCache::new(), funcs: HashMap::new() }
    }

    /// Dispatch a rolling formula by name.
    ///
    /// Input pointer ordering per formula:
    ///   "rolling_mean"   → inputs = [cs]         — prefix sum of x
    ///   "rolling_std"    → inputs = [cs, cs2]    — prefix sums of x and x²
    ///   "rolling_zscore" → inputs = [x, cs, cs2] — raw data + both prefix sums
    ///
    /// # Safety
    /// All pointers must be valid GPU allocations on the same context, containing
    /// at least `n` contiguous f64 elements.
    pub unsafe fn dispatch(
        &mut self,
        formula: &str,
        window: usize,
        inputs: &[u64],
        n: usize,
    ) -> Result<FusedExprOutput, Box<dyn std::error::Error>> {
        match formula {
            "rolling_mean" => {
                self.rolling_mean(inputs[0], n, window)
            }
            "rolling_std" => {
                self.rolling_std(inputs[0], inputs[1], n, window)
            }
            "rolling_zscore" => {
                self.rolling_zscore(inputs[0], inputs[1], inputs[2], n, window)
            }
            other => Err(format!("Unknown fused_expr formula: '{}'", other).into()),
        }
    }

    // rolling_mean[i] = (cs[i] - cs[i-w]) / count(i, w)
    unsafe fn rolling_mean(
        &mut self,
        cs_ptr: u64,
        n: usize,
        window: usize,
    ) -> Result<FusedExprOutput, Box<dyn std::error::Error>> {
        self.ensure_func("rolling_mean", rolling_mean_source())?;

        let cs = ManuallyDrop::new(self.stream.upgrade_device_ptr::<f64>(cs_ptr, n));
        let mut out: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let func = self.funcs.get("rolling_mean").unwrap();
        self.stream.launch_builder(func)
            .arg(&*cs)
            .arg(&mut out)
            .arg(&(n as i32))
            .arg(&(window as i32))
            .launch(LaunchConfig {
                grid_dim: (n_blocks as u32, 1, 1),
                block_dim: (BLOCK_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
        self.stream.synchronize()?;
        Ok(FusedExprOutput { data: out, len: n, stream: self.stream.clone() })
    }

    // rolling_std[i] = sqrt(sum_sq/n_w - (sum/n_w)^2)
    unsafe fn rolling_std(
        &mut self,
        cs_ptr: u64,
        cs2_ptr: u64,
        n: usize,
        window: usize,
    ) -> Result<FusedExprOutput, Box<dyn std::error::Error>> {
        self.ensure_func("rolling_std", rolling_std_source())?;

        let cs  = ManuallyDrop::new(self.stream.upgrade_device_ptr::<f64>(cs_ptr, n));
        let cs2 = ManuallyDrop::new(self.stream.upgrade_device_ptr::<f64>(cs2_ptr, n));
        let mut out: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let func = self.funcs.get("rolling_std").unwrap();
        self.stream.launch_builder(func)
            .arg(&*cs)
            .arg(&*cs2)
            .arg(&mut out)
            .arg(&(n as i32))
            .arg(&(window as i32))
            .launch(LaunchConfig {
                grid_dim: (n_blocks as u32, 1, 1),
                block_dim: (BLOCK_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
        self.stream.synchronize()?;
        Ok(FusedExprOutput { data: out, len: n, stream: self.stream.clone() })
    }

    // rolling_zscore[i] = (x[i] - mean[i]) / std[i]
    unsafe fn rolling_zscore(
        &mut self,
        x_ptr: u64,
        cs_ptr: u64,
        cs2_ptr: u64,
        n: usize,
        window: usize,
    ) -> Result<FusedExprOutput, Box<dyn std::error::Error>> {
        self.ensure_func("rolling_zscore", rolling_zscore_source())?;

        let x   = ManuallyDrop::new(self.stream.upgrade_device_ptr::<f64>(x_ptr, n));
        let cs  = ManuallyDrop::new(self.stream.upgrade_device_ptr::<f64>(cs_ptr, n));
        let cs2 = ManuallyDrop::new(self.stream.upgrade_device_ptr::<f64>(cs2_ptr, n));
        let mut out: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        let func = self.funcs.get("rolling_zscore").unwrap();
        self.stream.launch_builder(func)
            .arg(&*x)
            .arg(&*cs)
            .arg(&*cs2)
            .arg(&mut out)
            .arg(&(n as i32))
            .arg(&(window as i32))
            .launch(LaunchConfig {
                grid_dim: (n_blocks as u32, 1, 1),
                block_dim: (BLOCK_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            })?;
        self.stream.synchronize()?;
        Ok(FusedExprOutput { data: out, len: n, stream: self.stream.clone() })
    }

    /// Compile and cache a fused expression kernel by name (idempotent).
    fn ensure_func(
        &mut self,
        name: &str,
        source: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.funcs.contains_key(name) {
            return Ok(());
        }
        // Content-based key: hash name + source so different formulas never
        // collide at the 16-char disk path truncation boundary.
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fused_expr");
        hasher.update(name.as_bytes());
        hasher.update(source.as_bytes());
        let cache_key = hasher.finalize().to_hex().to_string();

        let ptx = self.cache.compile_source(source, &cache_key)?;
        let module = self.ctx.load_module(ptx)?;
        let func = module.load_function(&format!("{}_kernel", name))?;
        self.funcs.insert(name.to_string(), func);
        Ok(())
    }
}
