//! Hash scatter groupby engine.
//!
//! The sort-free groupby: one GPU pass, O(n), no sort, naturally parallel.
//!
//! ## The kernel
//!
//! ```text
//! for each element i (parallel, one thread per element):
//!     g = keys[i]                       // group id from GroupIndex
//!     atomicAdd(&group_sums[g],   values[i])
//!     atomicAdd(&group_sum_sqs[g], values[i] * values[i])
//!     atomicAdd(&group_counts[g], 1)
//! ```
//!
//! Multi-aggregation (sum + sum_sq + count) in one pass costs LESS than
//! argsort alone. Three aggregations for one memory read.
//!
//! ## The 17x
//!
//! Measured on 1M rows, 4,600 groups, NVIDIA Blackwell:
//! - Sort-based groupby:  1.04 ms (argsort = 0.49ms = 47% of total)
//! - Hash scatter sum:    0.06 ms
//! - Speedup:             17x

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// CUDA source — two kernels in one module
// ---------------------------------------------------------------------------

const SCATTER_CUDA_SOURCE: &str = r#"
// tambear scatter kernels — sort-free groupby on GPU.
// Each thread reads one element, atomically scatters to group accumulator.
// One pass. O(n). No sort. Naturally parallel.

// scatter_sum: just sum per group. Fastest possible groupby.
extern "C" __global__ void scatter_sum(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    double* __restrict__ sums,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        atomicAdd(&sums[keys[gid]], values[gid]);
    }
}

// scatter_stats: sum + sum_sq + count in ONE pass (naive — 3 global atomicAdds per element).
extern "C" __global__ void scatter_stats(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    double* __restrict__ sums,
    double* __restrict__ sum_sqs,
    double* __restrict__ counts,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int g = keys[gid];
        double v = values[gid];
        atomicAdd(&sums[g], v);
        atomicAdd(&sum_sqs[g], v * v);
        atomicAdd(&counts[g], 1.0);
    }
}

// scatter_stats_smem: shared-memory privatized scatter.
// Each block maintains local accumulators in shared memory, merges to global at the end.
// Reduces global atomicAdd contention from O(n/n_groups) to O(n_blocks/n_groups).
// Shared memory atomicAdd is ~10x faster than global. Big win for moderate n_groups.
// Shared memory cost: 3 * n_groups * 8 bytes. Fits default 48KB for n_groups <= 2048.
extern "C" __global__ void scatter_stats_smem(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    double* __restrict__ sums,
    double* __restrict__ sum_sqs,
    double* __restrict__ counts,
    int n,
    int n_groups
) {
    extern __shared__ double smem[];
    double* local_sums    = smem;
    double* local_sum_sqs = smem + n_groups;
    double* local_counts  = smem + 2 * n_groups;

    // Zero shared memory
    for (int i = threadIdx.x; i < n_groups; i += blockDim.x) {
        local_sums[i]    = 0.0;
        local_sum_sqs[i] = 0.0;
        local_counts[i]  = 0.0;
    }
    __syncthreads();

    // Local scatter to shared memory (fast atomics)
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int g = keys[gid];
        double v = values[gid];
        atomicAdd(&local_sums[g], v);
        atomicAdd(&local_sum_sqs[g], v * v);
        atomicAdd(&local_counts[g], 1.0);
    }
    __syncthreads();

    // Merge to global (one atomicAdd per active group per block)
    for (int i = threadIdx.x; i < n_groups; i += blockDim.x) {
        if (local_counts[i] > 0.0) {
            atomicAdd(&sums[i],    local_sums[i]);
            atomicAdd(&sum_sqs[i], local_sum_sqs[i]);
            atomicAdd(&counts[i],  local_counts[i]);
        }
    }
}

// scatter_stats_warp: warp-aggregated atomics via __match_any_sync.
// Threads with the same key within a warp reduce locally via full-warp shuffle,
// then one elected leader does a single atomicAdd per group per warp.
// Best when n_groups << warp_size (high intra-warp collision rate).
extern "C" __global__ void scatter_stats_warp(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    double* __restrict__ sums,
    double* __restrict__ sum_sqs,
    double* __restrict__ counts,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    int g = keys[gid];
    double v = values[gid];
    double v2 = v * v;
    int lane = threadIdx.x & 31;

    // Find all threads in this warp with the same key
    unsigned int peers = __match_any_sync(0xFFFFFFFF, g);
    int peer_count = __popc(peers);

    // Reduce v and v2 within peer group using full-warp broadcast
    double sum_v = 0.0, sum_v2 = 0.0;
    for (int src = 0; src < 32; src++) {
        double sv  = __shfl_sync(0xFFFFFFFF, v, src);
        double sv2 = __shfl_sync(0xFFFFFFFF, v2, src);
        if (peers & (1u << src)) {
            sum_v  += sv;
            sum_v2 += sv2;
        }
    }

    // Elect leader: lowest lane in peer group
    int leader = __ffs(peers) - 1;
    if (lane == leader) {
        atomicAdd(&sums[g],    sum_v);
        atomicAdd(&sum_sqs[g], sum_v2);
        atomicAdd(&counts[g],  (double)peer_count);
    }
}
"#;

// ---------------------------------------------------------------------------
// GroupByResult — per-group statistics from a single scatter pass
// ---------------------------------------------------------------------------

/// Result of a hash scatter groupby: sum + sum_sq + count per group.
///
/// From these three scalars, derive mean, variance, and std in O(n_groups).
/// The scatter was O(n). The derivation is negligible.
pub struct GroupByResult {
    pub n_groups: usize,
    pub sums: Vec<f64>,
    pub sum_sqs: Vec<f64>,
    /// Stored as f64 for GPU atomicAdd compatibility. Exact for counts < 2^53.
    pub counts: Vec<f64>,
}

impl GroupByResult {
    /// Per-group means. NaN for empty groups.
    pub fn means(&self) -> Vec<f64> {
        self.sums.iter().zip(&self.counts)
            .map(|(&s, &c)| if c > 0.0 { s / c } else { f64::NAN })
            .collect()
    }

    /// Per-group variances (Bessel-corrected, sample variance).
    /// Uses naive formula: var = (sum_sq - sum²/n) / (n-1).
    /// For numerically stable version, see RefCenteredStatsEngine.
    pub fn variances(&self) -> Vec<f64> {
        self.sums.iter()
            .zip(&self.sum_sqs)
            .zip(&self.counts)
            .map(|((&s, &sq), &c)| {
                if c > 1.0 {
                    (sq - s * s / c) / (c - 1.0)
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Per-group standard deviations.
    pub fn stds(&self) -> Vec<f64> {
        self.variances().into_iter().map(|v| v.sqrt()).collect()
    }
}

// ---------------------------------------------------------------------------
// HashScatterEngine — compiled CUDA kernels for sort-free groupby
// ---------------------------------------------------------------------------

/// The hash scatter groupby engine.
///
/// Compiles scatter CUDA kernels at creation time via NVRTC.
/// All operations are O(n) single-pass with atomic scatter.
pub struct HashScatterEngine {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    f_scatter_sum: CudaFunction,
    f_scatter_stats: CudaFunction,
    f_scatter_stats_smem: CudaFunction,
    f_scatter_stats_warp: CudaFunction,
}

impl HashScatterEngine {
    /// Create a new engine on GPU 0.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::on_device(0)
    }

    /// Create a new engine on a specific GPU.
    pub fn on_device(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();

        let opts = CompileOptions {
            arch: Some("sm_120"),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(SCATTER_CUDA_SOURCE, opts)?;
        let module = ctx.load_module(ptx)?;

        Ok(Self {
            f_scatter_sum: module.load_function("scatter_sum")?,
            f_scatter_stats: module.load_function("scatter_stats")?,
            f_scatter_stats_smem: module.load_function("scatter_stats_smem")?,
            f_scatter_stats_warp: module.load_function("scatter_stats_warp")?,
            ctx,
            stream,
        })
    }

    /// Access the CUDA stream (for synchronization, dtoh copies).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Access the CUDA context (for sharing with other engines).
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    // -----------------------------------------------------------------------
    // Host API — copy in, scatter, copy out
    // -----------------------------------------------------------------------

    /// Sum values per group. Keys must be in [0, n_groups).
    ///
    /// Single atomicAdd per element. Fastest possible groupby aggregation.
    pub fn scatter_sum(
        &self,
        keys: &[i32],
        values: &[f64],
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = keys.len();
        assert_eq!(n, values.len(), "keys and values must have same length");
        if n == 0 {
            return Ok(vec![0.0; n_groups]);
        }

        let keys_dev = self.stream.clone_htod(keys)?;
        let values_dev = self.stream.clone_htod(values)?;
        let mut sums_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;

        self.launch_scatter_sum(&keys_dev, &values_dev, &mut sums_dev, n)?;
        self.stream.synchronize()?;

        Ok(self.stream.clone_dtoh(&sums_dev)?)
    }

    /// Count rows per group. Returns `Vec<f64>` of length `n_groups`.
    ///
    /// Equivalent to `scatter_sum` with an all-ones value array: one atomicAdd per element.
    /// Faster than `groupby` when you only need counts (no sum, no mean, no variance).
    pub fn value_counts(
        &self,
        keys: &[i32],
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = keys.len();
        if n == 0 {
            return Ok(vec![0.0; n_groups]);
        }
        let ones = vec![1.0f64; n];
        self.scatter_sum(keys, &ones, n_groups)
    }

    /// Multi-stat groupby: sum + sum_sq + count in ONE pass.
    ///
    /// Three atomicAdds per element. From these, derive mean, variance, std.
    /// Costs less than argsort alone — three aggregations for one memory read.
    pub fn groupby(
        &self,
        keys: &[i32],
        values: &[f64],
        n_groups: usize,
    ) -> Result<GroupByResult, Box<dyn std::error::Error>> {
        let n = keys.len();
        assert_eq!(n, values.len(), "keys and values must have same length");
        if n == 0 {
            return Ok(GroupByResult {
                n_groups,
                sums: vec![0.0; n_groups],
                sum_sqs: vec![0.0; n_groups],
                counts: vec![0.0; n_groups],
            });
        }

        let keys_dev = self.stream.clone_htod(keys)?;
        let values_dev = self.stream.clone_htod(values)?;
        let mut sums_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut sum_sqs_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut counts_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;

        self.launch_scatter_stats(
            &keys_dev, &values_dev,
            &mut sums_dev, &mut sum_sqs_dev, &mut counts_dev,
            n,
        )?;
        self.stream.synchronize()?;

        Ok(GroupByResult {
            n_groups,
            sums: self.stream.clone_dtoh(&sums_dev)?,
            sum_sqs: self.stream.clone_dtoh(&sum_sqs_dev)?,
            counts: self.stream.clone_dtoh(&counts_dev)?,
        })
    }

    // -----------------------------------------------------------------------
    // GPU-resident API — for benchmarking and pipeline integration
    // -----------------------------------------------------------------------

    /// Scatter sum with data already on GPU. Returns GPU-resident sums buffer.
    pub fn scatter_sum_gpu(
        &self,
        keys: &CudaSlice<i32>,
        values: &CudaSlice<f64>,
        n: usize,
        n_groups: usize,
    ) -> Result<CudaSlice<f64>, Box<dyn std::error::Error>> {
        let mut sums_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        self.launch_scatter_sum(keys, values, &mut sums_dev, n)?;
        Ok(sums_dev)
    }

    /// Scatter stats with data already on GPU. Returns GPU-resident buffers.
    pub fn groupby_gpu(
        &self,
        keys: &CudaSlice<i32>,
        values: &CudaSlice<f64>,
        n: usize,
        n_groups: usize,
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>, CudaSlice<f64>), Box<dyn std::error::Error>> {
        let mut sums_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut sum_sqs_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut counts_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;

        self.launch_scatter_stats(
            keys, values,
            &mut sums_dev, &mut sum_sqs_dev, &mut counts_dev,
            n,
        )?;

        Ok((sums_dev, sum_sqs_dev, counts_dev))
    }

    // -----------------------------------------------------------------------
    // Optimized groupby variants
    // -----------------------------------------------------------------------

    /// Shared-memory privatized groupby. Each block reduces locally in shared
    /// memory, then merges to global. Massive contention reduction when
    /// n_groups fits in shared memory (<= ~2048 groups at default 48KB).
    pub fn groupby_smem(
        &self,
        keys: &[i32],
        values: &[f64],
        n_groups: usize,
    ) -> Result<GroupByResult, Box<dyn std::error::Error>> {
        let n = keys.len();
        assert_eq!(n, values.len(), "keys and values must have same length");
        if n == 0 {
            return Ok(GroupByResult {
                n_groups,
                sums: vec![0.0; n_groups],
                sum_sqs: vec![0.0; n_groups],
                counts: vec![0.0; n_groups],
            });
        }

        let keys_dev = self.stream.clone_htod(keys)?;
        let values_dev = self.stream.clone_htod(values)?;
        let mut sums_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut sum_sqs_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut counts_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;

        self.launch_scatter_stats_smem(
            &keys_dev, &values_dev,
            &mut sums_dev, &mut sum_sqs_dev, &mut counts_dev,
            n, n_groups,
        )?;
        self.stream.synchronize()?;

        Ok(GroupByResult {
            n_groups,
            sums: self.stream.clone_dtoh(&sums_dev)?,
            sum_sqs: self.stream.clone_dtoh(&sum_sqs_dev)?,
            counts: self.stream.clone_dtoh(&counts_dev)?,
        })
    }

    /// Warp-aggregated groupby. Threads with the same key in a warp reduce
    /// locally via shuffle, then one leader does the atomicAdd. Best when
    /// n_groups is small (high intra-warp collision rate).
    pub fn groupby_warp(
        &self,
        keys: &[i32],
        values: &[f64],
        n_groups: usize,
    ) -> Result<GroupByResult, Box<dyn std::error::Error>> {
        let n = keys.len();
        assert_eq!(n, values.len(), "keys and values must have same length");
        if n == 0 {
            return Ok(GroupByResult {
                n_groups,
                sums: vec![0.0; n_groups],
                sum_sqs: vec![0.0; n_groups],
                counts: vec![0.0; n_groups],
            });
        }

        let keys_dev = self.stream.clone_htod(keys)?;
        let values_dev = self.stream.clone_htod(values)?;
        let mut sums_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut sum_sqs_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut counts_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;

        self.launch_scatter_stats_warp(
            &keys_dev, &values_dev,
            &mut sums_dev, &mut sum_sqs_dev, &mut counts_dev,
            n,
        )?;
        self.stream.synchronize()?;

        Ok(GroupByResult {
            n_groups,
            sums: self.stream.clone_dtoh(&sums_dev)?,
            sum_sqs: self.stream.clone_dtoh(&sum_sqs_dev)?,
            counts: self.stream.clone_dtoh(&counts_dev)?,
        })
    }

    /// GPU-resident smem groupby.
    pub fn groupby_smem_gpu(
        &self,
        keys: &CudaSlice<i32>,
        values: &CudaSlice<f64>,
        n: usize,
        n_groups: usize,
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>, CudaSlice<f64>), Box<dyn std::error::Error>> {
        let mut sums_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut sum_sqs_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut counts_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;

        self.launch_scatter_stats_smem(
            keys, values,
            &mut sums_dev, &mut sum_sqs_dev, &mut counts_dev,
            n, n_groups,
        )?;

        Ok((sums_dev, sum_sqs_dev, counts_dev))
    }

    /// GPU-resident warp-aggregated groupby.
    pub fn groupby_warp_gpu(
        &self,
        keys: &CudaSlice<i32>,
        values: &CudaSlice<f64>,
        n: usize,
        n_groups: usize,
    ) -> Result<(CudaSlice<f64>, CudaSlice<f64>, CudaSlice<f64>), Box<dyn std::error::Error>> {
        let mut sums_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut sum_sqs_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut counts_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;

        self.launch_scatter_stats_warp(
            keys, values,
            &mut sums_dev, &mut sum_sqs_dev, &mut counts_dev,
            n,
        )?;

        Ok((sums_dev, sum_sqs_dev, counts_dev))
    }

    // -----------------------------------------------------------------------
    // Kernel launchers
    // -----------------------------------------------------------------------

    fn launch_scatter_sum(
        &self,
        keys: &CudaSlice<i32>,
        values: &CudaSlice<f64>,
        sums: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = launch_cfg(n);
        unsafe {
            self.stream.launch_builder(&self.f_scatter_sum)
                .arg(keys)
                .arg(values)
                .arg(sums)
                .arg(&(n as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    fn launch_scatter_stats(
        &self,
        keys: &CudaSlice<i32>,
        values: &CudaSlice<f64>,
        sums: &mut CudaSlice<f64>,
        sum_sqs: &mut CudaSlice<f64>,
        counts: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = launch_cfg(n);
        unsafe {
            self.stream.launch_builder(&self.f_scatter_stats)
                .arg(keys)
                .arg(values)
                .arg(sums)
                .arg(sum_sqs)
                .arg(counts)
                .arg(&(n as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    fn launch_scatter_stats_smem(
        &self,
        keys: &CudaSlice<i32>,
        values: &CudaSlice<f64>,
        sums: &mut CudaSlice<f64>,
        sum_sqs: &mut CudaSlice<f64>,
        counts: &mut CudaSlice<f64>,
        n: usize,
        n_groups: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let shared_bytes = 3 * n_groups as u32 * 8; // 3 accumulators × n_groups × f64
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        unsafe {
            self.stream.launch_builder(&self.f_scatter_stats_smem)
                .arg(keys)
                .arg(values)
                .arg(sums)
                .arg(sum_sqs)
                .arg(counts)
                .arg(&(n as i32))
                .arg(&(n_groups as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    fn launch_scatter_stats_warp(
        &self,
        keys: &CudaSlice<i32>,
        values: &CudaSlice<f64>,
        sums: &mut CudaSlice<f64>,
        sum_sqs: &mut CudaSlice<f64>,
        counts: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = launch_cfg(n);
        unsafe {
            self.stream.launch_builder(&self.f_scatter_stats_warp)
                .arg(keys)
                .arg(values)
                .arg(sums)
                .arg(sum_sqs)
                .arg(counts)
                .arg(&(n as i32))
                .launch(cfg)?;
        }
        Ok(())
    }
}

fn launch_cfg(n: usize) -> LaunchConfig {
    let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}
