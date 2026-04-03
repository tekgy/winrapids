//! GatherOp — GPU memory permutation kernels for gather-scan-scatter.
//!
//! Bridges the GroupIndex inverted index to the scan engine.
//! Two operations:
//!
//! **Gather**: `gathered[i] = values[rows_by_group[i]]`
//!   Reorders values from original row order into group-sorted order.
//!   After gather, all rows of group g are contiguous at positions
//!   `group_offsets[g]..group_offsets[g+1]`.
//!
//! **ScatterBack**: `output[rows_by_group[i]] = gathered[i]`
//!   Inverse permutation: restores from group-sorted order to original row order.
//!   Used after the per-group scan to write results back to original positions.
//!
//! These two operations bookend the per-group scan:
//!
//! ```text
//! gather(values, rows_by_group) → gathered      // O(n), one pass
//! scan(gathered, by group_offsets) → scanned     // O(n) total, O(n/k) latency
//! scatter_back(scanned, rows_by_group) → output  // O(n), one pass
//! ```
//!
//! Both kernels are fixed (no JIT parameterization). They compile once per
//! GatherOp instance and are reused for all subsequent calls.
//! Works on any TamGpu backend (CUDA, CPU, future Vulkan/Metal).

use std::sync::Arc;

use tam_gpu::{Buffer, Kernel, ShaderLang, TamGpu};

const BLOCK_SIZE: u32 = 256;

/// GPU gather and scatter-back operator.
///
/// Compiles the two fixed kernels once at construction. Subsequent calls
/// reuse the compiled kernels — no re-compilation overhead.
/// Works on any TamGpu backend.
pub struct GatherOp {
    gpu: Arc<dyn TamGpu>,
    gather_kernel: Option<Kernel>,
    scatter_back_kernel: Option<Kernel>,
}

impl GatherOp {
    /// Create a GatherOp using the auto-detected backend.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::with_backend(tam_gpu::detect()))
    }

    /// Create a GatherOp with a specific backend.
    pub fn with_backend(gpu: Arc<dyn TamGpu>) -> Self {
        Self { gpu, gather_kernel: None, scatter_back_kernel: None }
    }

    /// Gather: `gathered[i] = values[rows_by_group[i]]` for all i.
    ///
    /// Reorders `values` using the permutation in `rows_by_group`.
    /// `rows_by_group` has length `n_out` — the number of output elements.
    /// `values` must have at least `max(rows_by_group) + 1` elements.
    ///
    /// For standard group-reorder use: `values.len() == rows_by_group.len()`.
    /// For lag-gather use: `values` has one extra "default" element at position
    /// `n_ticks`; `rows_by_group` has length `n_ticks` with sentinel index
    /// `n_ticks` pointing at the default. See [`lag_indices`].
    pub fn gather(
        &mut self,
        values: &[f64],
        rows_by_group: &[u32],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n_out = rows_by_group.len();
        self.ensure_kernels()?;

        let values_buf = tam_gpu::upload(&*self.gpu, values)?;
        let rows_buf = tam_gpu::upload(&*self.gpu, rows_by_group)?;
        let gathered_buf = self.gpu.alloc(n_out * 8)?; // f64

        self.dispatch_gather(&values_buf, &rows_buf, &gathered_buf, n_out)?;
        self.gpu.sync()?;

        let gathered: Vec<f64> = tam_gpu::download(&*self.gpu, &gathered_buf, n_out)?;
        Ok(gathered)
    }

    /// Gather on device-resident buffers. Returns device-resident output.
    ///
    /// `n_out`: number of output elements (= rows_by_group length).
    /// `values` must have at least `max(rows_by_group) + 1` elements.
    pub fn gather_buf(
        &mut self,
        values: &Buffer,
        rows_by_group: &Buffer,
        n_out: usize,
    ) -> Result<Buffer, Box<dyn std::error::Error>> {
        self.ensure_kernels()?;
        let gathered = self.gpu.alloc(n_out * 8)?;
        self.dispatch_gather(values, rows_by_group, &gathered, n_out)?;
        Ok(gathered)
    }

    /// ScatterBack: `output[rows_by_group[i]] = gathered[i]` for all i.
    ///
    /// Inverse of `gather`: restores from group-sorted order to original row order.
    /// `gathered` and `rows_by_group` must have the same length n.
    pub fn scatter_back(
        &mut self,
        gathered: &[f64],
        rows_by_group: &[u32],
        n_rows: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = gathered.len();
        assert_eq!(n, rows_by_group.len(), "gathered and rows_by_group must have same length");
        assert_eq!(n, n_rows, "gathered length must equal n_rows");
        self.ensure_kernels()?;

        let gathered_buf = tam_gpu::upload(&*self.gpu, gathered)?;
        let rows_buf = tam_gpu::upload(&*self.gpu, rows_by_group)?;
        let output_buf = self.gpu.alloc(n * 8)?;

        self.dispatch_scatter_back(&gathered_buf, &rows_buf, &output_buf, n)?;
        self.gpu.sync()?;

        let output: Vec<f64> = tam_gpu::download(&*self.gpu, &output_buf, n)?;
        Ok(output)
    }

    /// ScatterBack on device-resident buffers. Returns device-resident output.
    pub fn scatter_back_buf(
        &mut self,
        gathered: &Buffer,
        rows_by_group: &Buffer,
        n: usize,
    ) -> Result<Buffer, Box<dyn std::error::Error>> {
        self.ensure_kernels()?;
        let output = self.gpu.alloc(n * 8)?;
        self.dispatch_scatter_back(gathered, rows_by_group, &output, n)?;
        Ok(output)
    }

    fn dispatch_gather(
        &self,
        values: &Buffer,
        rows_by_group: &Buffer,
        gathered: &Buffer,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let kernel = self.gather_kernel.as_ref().unwrap();
        match self.gpu.shader_lang() {
            ShaderLang::Cuda => {
                let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let n_buf = tam_gpu::upload(&*self.gpu, &[n as i32])?;
                self.gpu.dispatch(
                    kernel,
                    [n_blocks, 1, 1],
                    [BLOCK_SIZE, 1, 1],
                    &[values, rows_by_group, gathered, &n_buf],
                    0,
                )?;
            }
            _ => {
                // CpuBackend gather_f64: [values, rows_by_group, output]
                self.gpu.dispatch(
                    kernel,
                    [1, 1, 1],
                    [1, 1, 1],
                    &[values, rows_by_group, gathered],
                    0,
                )?;
            }
        }
        Ok(())
    }

    fn dispatch_scatter_back(
        &self,
        gathered: &Buffer,
        rows_by_group: &Buffer,
        output: &Buffer,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let kernel = self.scatter_back_kernel.as_ref().unwrap();
        match self.gpu.shader_lang() {
            ShaderLang::Cuda => {
                let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
                let n_buf = tam_gpu::upload(&*self.gpu, &[n as i32])?;
                self.gpu.dispatch(
                    kernel,
                    [n_blocks, 1, 1],
                    [BLOCK_SIZE, 1, 1],
                    &[gathered, rows_by_group, output, &n_buf],
                    0,
                )?;
            }
            _ => {
                // CpuBackend scatter_back_f64: [gathered, rows_by_group, output]
                self.gpu.dispatch(
                    kernel,
                    [1, 1, 1],
                    [1, 1, 1],
                    &[gathered, rows_by_group, output],
                    0,
                )?;
            }
        }
        Ok(())
    }

    fn ensure_kernels(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.gather_kernel.is_none() {
            let source = match self.gpu.shader_lang() {
                ShaderLang::Cuda => GATHER_SCATTER_BACK_SOURCE_CUDA,
                _ => "",
            };
            self.gather_kernel = Some(self.gpu.compile(source, "gather_f64")?);
            self.scatter_back_kernel = Some(self.gpu.compile(source, "scatter_back_f64")?);
        }
        Ok(())
    }
}

/// CUDA source for gather and scatter-back kernels.
///
/// Both kernels are plain memory permutations: O(n) work, fully coalesced
/// when rows_by_group contains a random-but-uniform permutation (typical for
/// tickers interleaved in time order). GPU bandwidth-bound: ~16μs for 2M rows.
///
/// n is passed as a single-element i32 buffer (TamGpu convention).
const GATHER_SCATTER_BACK_SOURCE_CUDA: &str = r#"
// gather_f64: gathered[i] = values[rows_by_group[i]]
// Reorders values from row order into group-sorted order.
// rows_by_group: inverted index from GroupIndex. Length = n.
extern "C" __global__ void gather_f64(
    const double* __restrict__ values,
    const unsigned int* __restrict__ rows_by_group,
    double* __restrict__ gathered,
    const int* __restrict__ n_ptr
) {
    int n = *n_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        gathered[i] = values[rows_by_group[i]];
    }
}

// scatter_back_f64: output[rows_by_group[i]] = gathered[i]
// Inverse permutation: restores from group-sorted order to original row order.
// Use after per-group scan to write results to original positions.
extern "C" __global__ void scatter_back_f64(
    const double* __restrict__ gathered,
    const unsigned int* __restrict__ rows_by_group,
    double* __restrict__ output,
    const int* __restrict__ n_ptr
) {
    int n = *n_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[rows_by_group[i]] = gathered[i];
    }
}
"#;

/// Generate permutation indices for a lag-k gather within bins.
///
/// For tick i at position ≥ k within its bin, `lag_indices[i] = i - k`.
/// For tick i within the first k positions of its bin (no predecessor), the
/// index points to position `n_ticks` — the caller must append a default
/// value at that position in the values array (typically 0.0).
///
/// **Constraint:** ticks must be in original temporal order within each bin.
/// The output is undefined if the tick stream has been reordered (e.g.,
/// group-sorted via GatherOp + GroupIndex) before calling this function.
///
/// # Example
/// ```
/// use tambear::gather_op::lag_indices;
/// // Two bins: bin 0 = ticks 0..3, bin 1 = ticks 3..5
/// let bin_starts = vec![0usize, 3];
/// let n = 5;
/// let idx = lag_indices(&bin_starts, n, 1);
/// // tick 0 (bin 0 start): sentinel (n=5)
/// // tick 1: 0, tick 2: 1 (within bin 0)
/// // tick 3 (bin 1 start): sentinel
/// // tick 4: 3 (within bin 1)
/// assert_eq!(idx, vec![5, 0, 1, 5, 3]);
/// ```
pub fn lag_indices(bin_starts: &[usize], n_ticks: usize, lag: usize) -> Vec<u32> {
    let sentinel = n_ticks as u32; // caller appends default at this position
    let mut indices = vec![sentinel; n_ticks];
    for b in 0..bin_starts.len() {
        let start = bin_starts[b];
        let end = if b + 1 < bin_starts.len() { bin_starts[b + 1] } else { n_ticks };
        for i in (start + lag)..end {
            indices[i] = (i - lag) as u32;
        }
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: gather then scatter_back should give original values.
    #[test]
    fn gather_scatter_back_roundtrip() {
        let mut op = GatherOp::new().unwrap();
        // 6 values interleaved: 3 groups (A=0, B=1, C=2)
        // rows_by_group from a GroupIndex with keys [A,B,B,A,C,A]:
        //   group A: rows 0,3,5  group B: rows 1,2  group C: row 4
        //   rows_by_group = [0,3,5, 1,2, 4]
        let values: Vec<f64>      = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let rows_by_group: Vec<u32> = vec![0, 3, 5, 1, 2, 4];

        // Gather: gathered = [values[0], values[3], values[5], values[1], values[2], values[4]]
        //                   = [10, 40, 60, 20, 30, 50]
        let gathered = op.gather(&values, &rows_by_group).unwrap();
        assert_eq!(gathered, vec![10.0, 40.0, 60.0, 20.0, 30.0, 50.0]);

        // ScatterBack: output[rows_by_group[i]] = gathered[i]
        // output[0]=10, output[3]=40, output[5]=60, output[1]=20, output[2]=30, output[4]=50
        let restored = op.scatter_back(&gathered, &rows_by_group, 6).unwrap();
        assert_eq!(restored, values, "scatter_back should restore original order");
    }

    /// Gather preserves within-group order (temporal order of rows).
    #[test]
    fn gather_preserves_temporal_order() {
        let mut op = GatherOp::new().unwrap();
        // 4 rows: keys = [0, 1, 0, 1] (two tickers, interleaved)
        // rows_by_group = [0, 2, 1, 3] (group 0 rows first, then group 1 rows)
        let values: Vec<f64>      = vec![1.0, 10.0, 2.0, 20.0];
        let rows_by_group: Vec<u32> = vec![0, 2, 1, 3];

        let gathered = op.gather(&values, &rows_by_group).unwrap();
        // group 0: rows 0,2 → values 1.0, 2.0 (temporal order preserved)
        // group 1: rows 1,3 → values 10.0, 20.0 (temporal order preserved)
        assert_eq!(gathered, vec![1.0, 2.0, 10.0, 20.0]);
    }

    /// lag_indices: sentinel positions at bin boundaries, predecessors elsewhere.
    #[test]
    fn lag_indices_two_bins() {
        // bin 0: ticks 0..3, bin 1: ticks 3..5
        let idx = lag_indices(&[0, 3], 5, 1);
        // tick 0: bin start → sentinel (5)
        // tick 1: predecessor 0
        // tick 2: predecessor 1
        // tick 3: bin start → sentinel (5)
        // tick 4: predecessor 3
        assert_eq!(idx, vec![5, 0, 1, 5, 3]);
    }

    /// lag_indices: lag-2 skips the first two ticks of each bin.
    #[test]
    fn lag_indices_lag2() {
        // single bin: ticks 0..5
        let idx = lag_indices(&[0], 5, 2);
        // ticks 0,1: no lag-2 predecessor → sentinel (5)
        // tick 2: predecessor 0
        // tick 3: predecessor 1
        // tick 4: predecessor 2
        assert_eq!(idx, vec![5, 5, 0, 1, 2]);
    }

    /// Bipower variation via lag-gather + pointwise multiply + sum.
    ///
    /// Bipower = Σ |r_i| · |r_{i-1}|. Normally computed with a carry register.
    /// Here: expressed as lag-1 GatherOp + pointwise multiply + scalar sum.
    /// Proves the carry-augmented → K01 lag-gather + K02 Add decomposition.
    #[test]
    fn bipower_via_lag_gather() {
        let mut op = GatherOp::new().unwrap();

        // 5 ticks in one bin: abs_r values (already absolute)
        let abs_r = vec![0.0_f64, 0.1, 0.05, 0.2, 0.15];
        let n = abs_r.len();

        // Expected bipower: sum of abs_r[i] * abs_r[i-1] for i >= 1
        // (first tick contributes 0 because no predecessor within the bin)
        let expected: f64 = abs_r[1] * abs_r[0]   // 0.1 * 0.0 = 0.0
                          + abs_r[2] * abs_r[1]   // 0.05 * 0.1 = 0.005
                          + abs_r[3] * abs_r[2]   // 0.2 * 0.05 = 0.01
                          + abs_r[4] * abs_r[3];  // 0.15 * 0.2 = 0.03

        // Step 1: generate lag-1 indices for one bin (bin_start = 0)
        let idx = lag_indices(&[0], n, 1);
        // idx = [5, 0, 1, 2, 3]  (tick 0 → sentinel 5, rest → predecessor)

        // Step 2: append default value 0.0 at position n (the sentinel index)
        let mut abs_r_with_default = abs_r.clone();
        abs_r_with_default.push(0.0);

        // Step 3: lag-gather (GatherOp permutation)
        let lagged = op.gather(&abs_r_with_default, &idx).unwrap();
        // lagged = [0.0, abs_r[0], abs_r[1], abs_r[2], abs_r[3]]
        //        = [0.0, 0.0, 0.1, 0.05, 0.2]

        // Step 4: pointwise multiply and sum (K02 accumulate)
        let bipower: f64 = abs_r.iter().zip(lagged.iter()).map(|(a, b)| a * b).sum();

        assert!((bipower - expected).abs() < 1e-12,
            "bipower via lag-gather = {bipower}, expected {expected}");
    }

    /// Single-element permutation: trivial identity.
    #[test]
    fn gather_single_element() {
        let mut op = GatherOp::new().unwrap();
        let gathered = op.gather(&[42.0], &[0u32]).unwrap();
        assert_eq!(gathered, vec![42.0]);
        let restored = op.scatter_back(&gathered, &[0u32], 1).unwrap();
        assert_eq!(restored, vec![42.0]);
    }
}
