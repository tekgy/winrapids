//! JIT-compiled additive scatter operations.
//!
//! The universal additive scatter: for each row i, compute φ(v_i) and atomicAdd
//! it to the accumulator for group keys[i].
//!
//! ```text
//! output[g] += φ(values[i])   for all i where keys[i] == g
//! ```
//!
//! `φ` is a CUDA expression in terms of:
//! - `v`: the current value (double)
//! - `r`: the ref value for this group (double; 0.0 if no refs provided)
//! - `g`: the group index (int)
//!
//! Common φ expressions:
//! - `"v"` → sum
//! - `"v * v"` → sum of squares
//! - `"v - r"` → centered sum (pass group means as refs)
//! - `"1.0"` → count
//! - `"v * v - r * r"` → centered sum of squares
//! - `"log(v)"` → log-sum (valid only for positive values)
//!
//! ## JIT caching
//!
//! Each unique φ expression is compiled to PTX once and cached. Subsequent calls
//! with the same expression use the cached kernel. Compilation cost: ~40ms first
//! call, ~35ns on cache hit. The scatter itself is always O(n).
//!
//! ## Composing multi-output statistics
//!
//! To compute multiple aggregates (e.g., sum + sum_sq + count), run three separate
//! scatter_phi calls. The memory cost is 3x vs. scatter_stats's 1x, but the
//! compositional flexibility is worth it for non-standard aggregates.
//!
//! For the standard (sum, sum_sq, count) triple, prefer HashScatterEngine::groupby(),
//! which fuses all three into one kernel pass (3 atomicAdds per element).
//! Use scatter_phi when you need non-standard φ functions.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const BLOCK_SIZE: u32 = 256;

/// JIT-compiled scatter kernel cache.
///
/// Compiles one CUDA kernel per unique φ expression, cached by expression string.
/// The CUDA context must outlive all scatter operations.
pub struct ScatterJit {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Cached compiled kernels indexed by φ expression.
    cache: HashMap<String, CudaFunction>,
}

impl ScatterJit {
    /// Create a new JIT scatter cache on GPU 0.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::on_device(0)
    }

    /// Create a new JIT scatter cache on a specific GPU.
    pub fn on_device(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream, cache: HashMap::new() })
    }

    /// Scatter `φ(values[i])` to `output[keys[i]]` for all i.
    ///
    /// `phi_expr`: CUDA expression in terms of `v` (value), `r` (ref), `g` (group index).
    /// `refs`: optional per-group reference values. If None, `r = 0.0` in the expression.
    /// `keys`, `values`, `refs`: host slices, transferred to GPU internally.
    /// `n_groups`: accumulator array size. Keys must be in `[0, n_groups)`.
    pub fn scatter_phi(
        &mut self,
        phi_expr: &str,
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = keys.len();
        assert_eq!(n, values.len(), "keys and values must have same length");
        if let Some(r) = refs {
            assert_eq!(r.len(), n_groups, "refs must have length n_groups");
        }

        let keys_dev = self.stream.clone_htod(keys)?;
        let values_dev = self.stream.clone_htod(values)?;
        let refs_dev: CudaSlice<f64> = match refs {
            Some(r) => self.stream.clone_htod(r)?,
            None => self.stream.alloc_zeros(n_groups)?,
        };
        let mut output_dev: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;

        self.launch_phi(phi_expr, &keys_dev, &values_dev, &refs_dev, &mut output_dev, n)?;
        self.stream.synchronize()?;

        Ok(self.stream.clone_dtoh(&output_dev)?)
    }

    /// Scatter with data already on GPU. Returns GPU-resident output.
    pub fn scatter_phi_gpu(
        &mut self,
        phi_expr: &str,
        keys: &CudaSlice<i32>,
        values: &CudaSlice<f64>,
        refs: &CudaSlice<f64>,
        n: usize,
        n_groups: usize,
    ) -> Result<CudaSlice<f64>, Box<dyn std::error::Error>> {
        let mut output: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        self.launch_phi(phi_expr, keys, values, refs, &mut output, n)?;
        Ok(output)
    }

    fn launch_phi(
        &mut self,
        phi_expr: &str,
        keys: &CudaSlice<i32>,
        values: &CudaSlice<f64>,
        refs: &CudaSlice<f64>,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Phase 1: compile if not cached (mutates self.cache).
        if !self.cache.contains_key(phi_expr) {
            let src = build_scatter_phi_source(phi_expr);
            let opts = CompileOptions {
                arch: Some("sm_120"),
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(&src, opts)?;
            let module = self.ctx.load_module(ptx)?;
            let f = module.load_function("scatter_phi")?;
            self.cache.insert(phi_expr.to_string(), f);
        }

        // Phase 2: launch. Split borrow: f borrows self.cache, self.stream is separate.
        let f = self.cache.get(phi_expr).unwrap();
        let cfg = launch_cfg(n);
        unsafe {
            self.stream.launch_builder(f)
                .arg(keys)
                .arg(values)
                .arg(refs)
                .arg(output)
                .arg(&(n as i32))
                .launch(cfg)?;
        }
        Ok(())
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Scatter N φ functions over the same data in a single kernel pass.
    ///
    /// Equivalent to N separate `scatter_phi` calls but fused: one memory pass,
    /// N atomicAdds per element. This is the compiler's scatter fusion operation:
    /// adjacent scatter_phi calls with the same (keys, monoid) are fused into
    /// one kernel with a multi-dimensional lift.
    ///
    /// `phi_exprs`: 1–8 CUDA expressions. Same variables as `scatter_phi`.
    /// Returns a `Vec` of N output vectors, one per φ expression.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tambear::{ScatterJit, PHI_SUM, PHI_SUM_SQ, PHI_COUNT};
    /// let mut jit = ScatterJit::new().unwrap();
    /// let keys   = vec![0i32, 0, 1, 1];
    /// let values = vec![3.0,  4.0, 1.0, 2.0];
    /// // One kernel pass computes sum, sum_sq, and count simultaneously.
    /// let results = jit.scatter_multi_phi(
    ///     &[PHI_SUM, PHI_SUM_SQ, PHI_COUNT],
    ///     &keys, &values, None, 2,
    /// ).unwrap();
    /// // results[0] = sums, results[1] = sum_sqs, results[2] = counts
    /// ```
    pub fn scatter_multi_phi(
        &mut self,
        phi_exprs: &[&str],
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        n_groups: usize,
    ) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let n_phi = phi_exprs.len();
        assert!(n_phi >= 1 && n_phi <= 8, "scatter_multi_phi: 1–8 phi expressions supported, got {}", n_phi);
        let n = keys.len();
        assert_eq!(n, values.len(), "keys and values must have same length");
        if let Some(r) = refs {
            assert_eq!(r.len(), n_groups, "refs must have length n_groups");
        }

        let keys_dev = self.stream.clone_htod(keys)?;
        let values_dev = self.stream.clone_htod(values)?;
        let refs_dev: CudaSlice<f64> = match refs {
            Some(r) => self.stream.clone_htod(r)?,
            None => self.stream.alloc_zeros(n_groups)?,
        };
        let mut output_devs: Vec<CudaSlice<f64>> = (0..n_phi)
            .map(|_| self.stream.alloc_zeros::<f64>(n_groups))
            .collect::<Result<Vec<_>, _>>()?;

        self.launch_multi_phi(phi_exprs, &keys_dev, &values_dev, &refs_dev, &mut output_devs, n)?;
        self.stream.synchronize()?;

        output_devs.iter()
            .map(|dev| self.stream.clone_dtoh(dev).map_err(Into::into))
            .collect()
    }

    fn launch_multi_phi(
        &mut self,
        phi_exprs: &[&str],
        keys: &CudaSlice<i32>,
        values: &CudaSlice<f64>,
        refs: &CudaSlice<f64>,
        outputs: &mut Vec<CudaSlice<f64>>,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Cache key distinguishes multi-phi from single-phi: "multi:<n>:<phi0>|<phi1>|..."
        // Single-phi keys never start with "multi:" so no collision.
        let cache_key = format!("multi:{}:{}", phi_exprs.len(), phi_exprs.join("|"));

        if !self.cache.contains_key(&cache_key) {
            let src = build_scatter_multi_phi_source(phi_exprs);
            let opts = CompileOptions {
                arch: Some("sm_120"),
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(&src, opts)?;
            let module = self.ctx.load_module(ptx)?;
            let f = module.load_function("scatter_multi_phi")?;
            self.cache.insert(cache_key.clone(), f);
        }

        let f = self.cache.get(&cache_key).unwrap();
        let cfg = launch_cfg(n);
        let n_i32 = n as i32;

        // Variable-arg launch: match on the output slice length.
        // Each arm pushes (keys, values, refs, out0..outK-1, n) in kernel-parameter order.
        // The CUDA kernel signature is generated to match exactly.
        unsafe {
            match outputs.as_mut_slice() {
                [o0] => self.stream.launch_builder(f)
                    .arg(keys).arg(values).arg(refs)
                    .arg(o0)
                    .arg(&n_i32).launch(cfg)?,
                [o0, o1] => self.stream.launch_builder(f)
                    .arg(keys).arg(values).arg(refs)
                    .arg(o0).arg(o1)
                    .arg(&n_i32).launch(cfg)?,
                [o0, o1, o2] => self.stream.launch_builder(f)
                    .arg(keys).arg(values).arg(refs)
                    .arg(o0).arg(o1).arg(o2)
                    .arg(&n_i32).launch(cfg)?,
                [o0, o1, o2, o3] => self.stream.launch_builder(f)
                    .arg(keys).arg(values).arg(refs)
                    .arg(o0).arg(o1).arg(o2).arg(o3)
                    .arg(&n_i32).launch(cfg)?,
                [o0, o1, o2, o3, o4] => self.stream.launch_builder(f)
                    .arg(keys).arg(values).arg(refs)
                    .arg(o0).arg(o1).arg(o2).arg(o3).arg(o4)
                    .arg(&n_i32).launch(cfg)?,
                [o0, o1, o2, o3, o4, o5] => self.stream.launch_builder(f)
                    .arg(keys).arg(values).arg(refs)
                    .arg(o0).arg(o1).arg(o2).arg(o3).arg(o4).arg(o5)
                    .arg(&n_i32).launch(cfg)?,
                [o0, o1, o2, o3, o4, o5, o6] => self.stream.launch_builder(f)
                    .arg(keys).arg(values).arg(refs)
                    .arg(o0).arg(o1).arg(o2).arg(o3).arg(o4).arg(o5).arg(o6)
                    .arg(&n_i32).launch(cfg)?,
                [o0, o1, o2, o3, o4, o5, o6, o7] => self.stream.launch_builder(f)
                    .arg(keys).arg(values).arg(refs)
                    .arg(o0).arg(o1).arg(o2).arg(o3).arg(o4).arg(o5).arg(o6).arg(o7)
                    .arg(&n_i32).launch(cfg)?,
                _ => return Err("scatter_multi_phi: max 8 phi expressions".into()),
            }
        }
        Ok(())
    }
}

/// Build CUDA source for a scatter kernel with the given φ expression.
///
/// The φ expression is substituted verbatim into the kernel body.
/// Variables available to φ:
/// - `v`: the current value (double)
/// - `r`: the ref value for this group (double; 0.0 if refs are all-zero)
/// - `g`: the group index (int)
fn build_scatter_phi_source(phi_expr: &str) -> String {
    format!(r#"
// JIT scatter kernel: output[g] += phi(v) for elements in group g.
// phi = {phi}
extern "C" __global__ void scatter_phi(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    const double* __restrict__ refs,    // per-group ref values (0.0 if unused)
    double* __restrict__ output,
    int n
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {{
        int g = keys[gid];
        double v = values[gid];
        double r = refs[g];
        double phi = ({phi});
        atomicAdd(&output[g], phi);
    }}
}}
"#, phi = phi_expr)
}

/// Build CUDA source for a fused multi-scatter kernel with N φ expressions.
///
/// Generates a kernel with N named output arrays (out0..outN-1). One atomicAdd
/// per (element, phi) pair, all in a single thread. This is the minimum-cost
/// fusion: one memory pass over keys+values, N atomicAdds per element.
///
/// The Rust launcher in `launch_multi_phi` uses slice-pattern matching to push
/// exactly the right number of output pointers as kernel parameters.
fn build_scatter_multi_phi_source(phi_exprs: &[&str]) -> String {
    let n = phi_exprs.len();

    // "    double* __restrict__ out0,\n    double* __restrict__ out1,\n..."
    let out_params: String = (0..n)
        .map(|i| format!("    double* __restrict__ out{},\n", i))
        .collect();

    // "        atomicAdd(&out0[g], (v));\n        atomicAdd(&out1[g], (v * v));\n..."
    let atomic_adds: String = phi_exprs.iter().enumerate()
        .map(|(i, phi)| format!("        atomicAdd(&out{}[g], ({}));\n", i, phi))
        .collect();

    format!(r#"
// JIT fused scatter kernel: {n} outputs, one memory pass.
// phi expressions: [{phis}]
extern "C" __global__ void scatter_multi_phi(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    const double* __restrict__ refs,
{out_params}    int n
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {{
        int g = keys[gid];
        double v = values[gid];
        double r = refs[g];
{atomic_adds}    }}
}}
"#,
        n = n,
        phis = phi_exprs.join(", "),
        out_params = out_params,
        atomic_adds = atomic_adds,
    )
}

fn launch_cfg(n: usize) -> LaunchConfig {
    let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// Well-known phi expressions as named constants
// ---------------------------------------------------------------------------

/// φ = v: scatter sum. Equivalent to HashScatterEngine::scatter_sum.
pub const PHI_SUM: &str = "v";

/// φ = v*v: scatter sum of squares.
pub const PHI_SUM_SQ: &str = "v * v";

/// φ = v - r: scatter centered sum (r = group mean from previous pass).
pub const PHI_CENTERED_SUM: &str = "v - r";

/// φ = (v-r)*(v-r): scatter centered sum of squares.
pub const PHI_CENTERED_SUM_SQ: &str = "(v - r) * (v - r)";

/// φ = 1.0: scatter count.
pub const PHI_COUNT: &str = "1.0";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_sum_matches_scatter_sum() {
        let mut jit = ScatterJit::new().unwrap();
        let keys = vec![0i32, 0, 1, 1, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = jit.scatter_phi(PHI_SUM, &keys, &values, None, 3).unwrap();
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 7.0).abs() < 1e-10);
        assert!((result[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn phi_sum_sq() {
        let mut jit = ScatterJit::new().unwrap();
        let keys = vec![0i32, 0];
        let values = vec![2.0, 3.0];
        let result = jit.scatter_phi(PHI_SUM_SQ, &keys, &values, None, 1).unwrap();
        // 2^2 + 3^2 = 4 + 9 = 13
        assert!((result[0] - 13.0).abs() < 1e-10);
    }

    #[test]
    fn phi_centered_sum() {
        let mut jit = ScatterJit::new().unwrap();
        let keys = vec![0i32, 0, 1];
        let values = vec![1.0, 3.0, 10.0];
        let refs = vec![2.0, 10.0];  // group means
        let result = jit.scatter_phi(PHI_CENTERED_SUM, &keys, &values, Some(&refs), 2).unwrap();
        // group 0: (1-2) + (3-2) = -1 + 1 = 0
        // group 1: (10-10) = 0
        assert!(result[0].abs() < 1e-10);
        assert!(result[1].abs() < 1e-10);
    }

    #[test]
    fn phi_compiled_once_cached() {
        let mut jit = ScatterJit::new().unwrap();
        let keys = vec![0i32];
        let values = vec![1.0f64];
        // First call: compiles
        jit.scatter_phi(PHI_SUM, &keys, &values, None, 1).unwrap();
        // Second call: cache hit
        jit.scatter_phi(PHI_SUM, &keys, &values, None, 1).unwrap();
        assert_eq!(jit.cache.len(), 1, "should have exactly one cached kernel");
    }

    // --- scatter_multi_phi tests ---

    /// Fused (sum, sum_sq, count) in one kernel pass must equal three separate calls.
    #[test]
    fn multi_phi_matches_three_separate_calls() {
        let mut jit = ScatterJit::new().unwrap();
        let keys   = vec![0i32, 0, 1, 1, 2];
        let values = vec![1.0,  2.0, 3.0, 4.0, 5.0];

        let fused = jit.scatter_multi_phi(
            &[PHI_SUM, PHI_SUM_SQ, PHI_COUNT],
            &keys, &values, None, 3,
        ).unwrap();

        let sums   = jit.scatter_phi(PHI_SUM,    &keys, &values, None, 3).unwrap();
        let sq     = jit.scatter_phi(PHI_SUM_SQ, &keys, &values, None, 3).unwrap();
        let counts = jit.scatter_phi(PHI_COUNT,  &keys, &values, None, 3).unwrap();

        for g in 0..3 {
            assert!((fused[0][g] - sums[g]).abs()   < 1e-10, "sum   mismatch group {}", g);
            assert!((fused[1][g] - sq[g]).abs()     < 1e-10, "sq    mismatch group {}", g);
            assert!((fused[2][g] - counts[g]).abs() < 1e-10, "count mismatch group {}", g);
        }
    }

    /// scatter_multi_phi([sum, sum_sq, count]) is semantically identical to scatter_stats.
    #[test]
    fn multi_phi_is_scatter_stats_generalized() {
        let mut jit = ScatterJit::new().unwrap();
        let keys   = vec![0i32, 0, 1, 1, 2];
        let values = vec![1.0,  2.0, 3.0, 4.0, 5.0];

        let r = jit.scatter_multi_phi(
            &[PHI_SUM, PHI_SUM_SQ, PHI_COUNT],
            &keys, &values, None, 3,
        ).unwrap();

        // group 0: [1,2]  → sum=3, sum_sq=1+4=5, count=2
        assert!((r[0][0] - 3.0).abs() < 1e-10);
        assert!((r[1][0] - 5.0).abs() < 1e-10);
        assert!((r[2][0] - 2.0).abs() < 1e-10);
        // group 1: [3,4]  → sum=7, sum_sq=9+16=25, count=2
        assert!((r[0][1] - 7.0).abs()  < 1e-10);
        assert!((r[1][1] - 25.0).abs() < 1e-10);
        assert!((r[2][1] - 2.0).abs()  < 1e-10);
        // group 2: [5]    → sum=5, sum_sq=25, count=1
        assert!((r[0][2] - 5.0).abs()  < 1e-10);
        assert!((r[1][2] - 25.0).abs() < 1e-10);
        assert!((r[2][2] - 1.0).abs()  < 1e-10);
    }

    /// Multi-phi kernel is compiled once and reused on subsequent calls.
    #[test]
    fn multi_phi_cached() {
        let mut jit = ScatterJit::new().unwrap();
        let keys   = vec![0i32, 1];
        let values = vec![1.0, 2.0];
        let phis = [PHI_SUM, PHI_COUNT];

        jit.scatter_multi_phi(&phis, &keys, &values, None, 2).unwrap();
        jit.scatter_multi_phi(&phis, &keys, &values, None, 2).unwrap();

        let multi_count = jit.cache.keys().filter(|k| k.starts_with("multi:")).count();
        assert_eq!(multi_count, 1, "should cache exactly one multi-phi kernel");
    }

    /// Two-phi fused kernel works correctly.
    #[test]
    fn multi_phi_two_outputs() {
        let mut jit = ScatterJit::new().unwrap();
        let keys   = vec![0i32, 0];
        let values = vec![3.0, 4.0];

        let r = jit.scatter_multi_phi(&[PHI_SUM, PHI_SUM_SQ], &keys, &values, None, 1).unwrap();
        // sum = 7, sum_sq = 9 + 16 = 25
        assert!((r[0][0] - 7.0).abs()  < 1e-10);
        assert!((r[1][0] - 25.0).abs() < 1e-10);
    }
}
