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

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const BLOCK_SIZE: u32 = 256;

/// GPU gather and scatter-back operator.
///
/// Compiles the two fixed kernels once at construction. Subsequent calls
/// reuse the compiled PTX — no re-compilation overhead.
pub struct GatherOp {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    gather_fn: Option<CudaFunction>,
    scatter_back_fn: Option<CudaFunction>,
}

impl GatherOp {
    /// Create a GatherOp on GPU 0. Kernels are compiled lazily on first use.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::on_device(0)
    }

    /// Create a GatherOp on a specific GPU.
    pub fn on_device(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream, gather_fn: None, scatter_back_fn: None })
    }

    /// Gather: `gathered[i] = values[rows_by_group[i]]` for all i.
    ///
    /// Reorders `values` from original row order into group-sorted order.
    /// `rows_by_group`: from `GroupIndex::rows_by_group` (the inverted index).
    /// `values` and `rows_by_group` must have the same length n.
    pub fn gather(
        &mut self,
        values: &[f64],
        rows_by_group: &[u32],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = values.len();
        assert_eq!(n, rows_by_group.len(), "values and rows_by_group must have same length");

        let values_dev = self.stream.clone_htod(values)?;
        let rows_dev = self.stream.clone_htod(rows_by_group)?;
        let gathered = self.gather_gpu(&values_dev, &rows_dev, n)?;
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&gathered)?)
    }

    /// Gather on GPU-resident data. Returns GPU-resident output.
    pub fn gather_gpu(
        &mut self,
        values: &CudaSlice<f64>,
        rows_by_group: &CudaSlice<u32>,
        n: usize,
    ) -> Result<CudaSlice<f64>, Box<dyn std::error::Error>> {
        self.ensure_kernels()?;
        let f = self.gather_fn.as_ref().unwrap();
        let mut gathered: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let cfg = launch_cfg(n);
        let n_i32 = n as i32;
        unsafe {
            self.stream.launch_builder(f)
                .arg(values)
                .arg(rows_by_group)
                .arg(&mut gathered)
                .arg(&n_i32)
                .launch(cfg)?;
        }
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

        let gathered_dev = self.stream.clone_htod(gathered)?;
        let rows_dev = self.stream.clone_htod(rows_by_group)?;
        let output = self.scatter_back_gpu(&gathered_dev, &rows_dev, n)?;
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&output)?)
    }

    /// ScatterBack on GPU-resident data. Returns GPU-resident output.
    pub fn scatter_back_gpu(
        &mut self,
        gathered: &CudaSlice<f64>,
        rows_by_group: &CudaSlice<u32>,
        n: usize,
    ) -> Result<CudaSlice<f64>, Box<dyn std::error::Error>> {
        self.ensure_kernels()?;
        let f = self.scatter_back_fn.as_ref().unwrap();
        let mut output: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let cfg = launch_cfg(n);
        let n_i32 = n as i32;
        unsafe {
            self.stream.launch_builder(f)
                .arg(gathered)
                .arg(rows_by_group)
                .arg(&mut output)
                .arg(&n_i32)
                .launch(cfg)?;
        }
        Ok(output)
    }

    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }
    pub fn ctx(&self) -> &Arc<CudaContext> { &self.ctx }

    /// Compile both kernels from the shared source in one pass, if not yet compiled.
    /// Subsequent calls are no-ops. Both kernels live in the same CUDA compilation unit.
    fn ensure_kernels(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.gather_fn.is_none() || self.scatter_back_fn.is_none() {
            let opts = CompileOptions { arch: Some("sm_120"), ..Default::default() };
            let ptx = compile_ptx_with_opts(GATHER_SCATTER_BACK_SOURCE, opts)?;
            let module = self.ctx.load_module(ptx)?;
            self.gather_fn = Some(module.load_function("gather_f64")?);
            self.scatter_back_fn = Some(module.load_function("scatter_back_f64")?);
        }
        Ok(())
    }
}

/// CUDA source for gather and scatter-back kernels.
///
/// Both kernels are plain memory permutations: O(n) work, fully coalesced
/// when rows_by_group contains a random-but-uniform permutation (typical for
/// tickers interleaved in time order). GPU bandwidth-bound: ~16μs for 2M rows.
const GATHER_SCATTER_BACK_SOURCE: &str = r#"
// gather_f64: gathered[i] = values[rows_by_group[i]]
// Reorders values from row order into group-sorted order.
// rows_by_group: inverted index from GroupIndex. Length = n.
extern "C" __global__ void gather_f64(
    const double* __restrict__ values,
    const unsigned int* __restrict__ rows_by_group,
    double* __restrict__ gathered,
    int n
) {
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
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[rows_by_group[i]] = gathered[i];
    }
}
"#;

fn launch_cfg(n: usize) -> LaunchConfig {
    let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
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
