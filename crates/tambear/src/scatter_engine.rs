//! Backend-agnostic scatter/map engine — the vendor door.
//!
//! `ScatterEngine` replaces `ScatterJit`'s direct cudarc usage with the
//! [`TamGpu`] abstraction. Same phi expressions, same kernel fusion, but
//! dispatched through the uniform GPU backend — works on CUDA, CPU, and
//! (future) Vulkan/Metal/DX12.
//!
//! ## Architecture
//!
//! ```text
//! ScatterEngine
//!   ├── gpu: Arc<dyn TamGpu>     ← backend-agnostic
//!   ├── cache: HashMap<key, Kernel>  ← compiled kernel cache
//!   │
//!   ├── scatter_phi()         ← single phi, one pass
//!   ├── scatter_multi_phi()   ← N fused phi, one pass
//!   ├── map_phi()             ← element-wise f(v)
//!   └── map_phi2()            ← element-wise f(a, b)
//! ```
//!
//! ## Migration from ScatterJit
//!
//! | ScatterJit (CUDA-only)         | ScatterEngine (any backend)       |
//! |-------------------------------|----------------------------------|
//! | `ScatterJit::new()`           | `ScatterEngine::new(detect())`   |
//! | `jit.scatter_phi(phi, ...)`   | `engine.scatter_phi(phi, ...)`   |
//! | `jit.scatter_multi_phi(..)`   | `engine.scatter_multi_phi(..)`   |
//! | `jit.map_phi(phi, values)`    | `engine.map_phi(phi, values)`    |
//! | `jit.map_phi2(phi, a, b)`     | `engine.map_phi2(phi, a, b)`     |
//!
//! ScatterJit remains for CUDA-specific features (warp scatter, masked scatter,
//! dual-target scatter) until those are lifted into ScatterEngine.

use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use tam_gpu::{TamGpu, TamResult, TamGpuError, ShaderLang, Kernel, upload, download};

const BLOCK_SIZE: u32 = 256;

// ═══════════════════════════════════════════════════════════════════════════
// ScatterEngine
// ═══════════════════════════════════════════════════════════════════════════

/// Backend-agnostic scatter/map dispatch engine.
///
/// Generates, compiles, caches, and dispatches scatter and map kernels for
/// any [`TamGpu`] backend. The vendor door for scatter operations.
///
/// # Example
/// ```no_run
/// use std::sync::Arc;
/// use tam_gpu::detect;
/// use tambear::scatter_engine::ScatterEngine;
///
/// let engine = ScatterEngine::new(detect());
///
/// // Sum by group — same API as ScatterJit but runs on any backend
/// let keys = vec![0i32, 0, 1, 1, 2];
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let sums = engine.scatter_phi("v", &keys, &values, None, 3).unwrap();
/// assert_eq!(sums, vec![3.0, 7.0, 5.0]);
/// ```
pub struct ScatterEngine {
    gpu: Arc<dyn TamGpu>,
    cache: Mutex<HashMap<String, Arc<Kernel>>>,
}

impl ScatterEngine {
    /// Create a new engine backed by `gpu`.
    pub fn new(gpu: Arc<dyn TamGpu>) -> Self {
        Self { gpu, cache: Mutex::new(HashMap::new()) }
    }

    /// The shader language of the underlying GPU backend.
    pub fn shader_lang(&self) -> ShaderLang {
        self.gpu.shader_lang()
    }

    /// Number of compiled kernels in the cache.
    pub fn cache_len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    // ─────────────────────────────────────────────────────────────────────
    // scatter_phi: output[keys[i]] += phi(values[i], refs[keys[i]], keys[i])
    // ─────────────────────────────────────────────────────────────────────

    /// Scatter `φ(v, r, g)` into group accumulators.
    ///
    /// For each element `i`: `output[keys[i]] += φ(values[i], refs[keys[i]], keys[i])`.
    ///
    /// `phi_expr`: expression in terms of `v` (value), `r` (group ref), `g` (group index).
    /// `refs`: per-group reference values. Pass `None` for `r = 0.0`.
    /// `n_groups`: number of output accumulators. Keys must be in `[0, n_groups)`.
    pub fn scatter_phi(
        &self,
        phi_expr: &str,
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        n_groups: usize,
    ) -> TamResult<Vec<f64>> {
        let n = keys.len();
        assert_eq!(n, values.len(), "keys and values must have same length");
        if let Some(r) = refs {
            assert_eq!(r.len(), n_groups, "refs must have length n_groups");
        }

        let kernel = self.get_or_compile_scatter(phi_expr)?;
        let zeros = vec![0.0f64; n_groups];
        let refs_data = refs.unwrap_or(&zeros);

        let k_buf = upload::<i32>(&*self.gpu, keys)?;
        let v_buf = upload::<f64>(&*self.gpu, values)?;
        let r_buf = upload::<f64>(&*self.gpu, refs_data)?;
        let o_buf = self.gpu.alloc(n_groups * 8)?; // zero-init
        let p_buf = upload::<i32>(&*self.gpu, &[n as i32])?;

        let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        self.gpu.dispatch(&*kernel, [n_blocks, 1, 1], [BLOCK_SIZE, 1, 1],
            &[&k_buf, &v_buf, &r_buf, &o_buf, &p_buf], 0)?;
        self.gpu.sync()?;
        download::<f64>(&*self.gpu, &o_buf, n_groups)
    }

    fn get_or_compile_scatter(&self, phi_expr: &str) -> TamResult<Arc<Kernel>> {
        let key = format!("scatter:{}", phi_expr);
        {
            let cache = self.cache.lock().unwrap();
            if let Some(k) = cache.get(&key) {
                return Ok(Arc::clone(k));
            }
        }

        let (source, entry) = match self.gpu.shader_lang() {
            ShaderLang::Cuda => (gen_scatter_phi_cuda(phi_expr), "scatter_phi"),
            ShaderLang::Cpu  => (gen_scatter_phi_cpu(phi_expr), "scatter_phi"),
            _ => return Err(TamGpuError::Compile(
                "scatter_phi: WGSL backend not yet supported (atomic f64 requires CAS loop)".into()
            )),
        };
        let kernel = self.gpu.compile(&source, entry)?;
        let arc = Arc::new(kernel);
        let mut cache = self.cache.lock().unwrap();
        let entry_ref = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry_ref))
    }

    // ─────────────────────────────────────────────────────────────────────
    // scatter_multi_phi: N fused phi expressions, one pass
    // ─────────────────────────────────────────────────────────────────────

    /// Scatter N `φ` expressions over the same data in a single pass.
    ///
    /// Returns a `Vec` of N output vectors, one per phi expression.
    /// Maximum 8 phi expressions.
    pub fn scatter_multi_phi(
        &self,
        phi_exprs: &[&str],
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        n_groups: usize,
    ) -> TamResult<Vec<Vec<f64>>> {
        let n_phi = phi_exprs.len();
        assert!(n_phi >= 1 && n_phi <= 8, "1–8 phi expressions, got {}", n_phi);
        let n = keys.len();
        assert_eq!(n, values.len(), "keys and values must have same length");
        if let Some(r) = refs {
            assert_eq!(r.len(), n_groups, "refs must have length n_groups");
        }

        let kernel = self.get_or_compile_multi_scatter(phi_exprs)?;
        let zeros = vec![0.0f64; n_groups];
        let refs_data = refs.unwrap_or(&zeros);

        let k_buf = upload::<i32>(&*self.gpu, keys)?;
        let v_buf = upload::<f64>(&*self.gpu, values)?;
        let r_buf = upload::<f64>(&*self.gpu, refs_data)?;

        // Allocate N output buffers
        let out_bufs: Vec<_> = (0..n_phi)
            .map(|_| self.gpu.alloc(n_groups * 8))
            .collect::<TamResult<Vec<_>>>()?;

        let p_buf = upload::<i32>(&*self.gpu, &[n as i32])?;

        // Build buffer slice: [keys, values, refs, out0, ..., outK-1, params]
        let mut all_bufs: Vec<&tam_gpu::Buffer> = vec![&k_buf, &v_buf, &r_buf];
        for ob in &out_bufs {
            all_bufs.push(ob);
        }
        all_bufs.push(&p_buf);

        let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        self.gpu.dispatch(&*kernel, [n_blocks, 1, 1], [BLOCK_SIZE, 1, 1],
            &all_bufs, 0)?;
        self.gpu.sync()?;

        out_bufs.iter()
            .map(|ob| download::<f64>(&*self.gpu, ob, n_groups))
            .collect()
    }

    fn get_or_compile_multi_scatter(&self, phi_exprs: &[&str]) -> TamResult<Arc<Kernel>> {
        let key = format!("multi_scatter:{}:{}", phi_exprs.len(), phi_exprs.join("|"));
        {
            let cache = self.cache.lock().unwrap();
            if let Some(k) = cache.get(&key) {
                return Ok(Arc::clone(k));
            }
        }

        let (source, entry) = match self.gpu.shader_lang() {
            ShaderLang::Cuda => (gen_scatter_multi_phi_cuda(phi_exprs), "scatter_multi_phi"),
            ShaderLang::Cpu  => (gen_scatter_multi_phi_cpu(phi_exprs), "scatter_multi_phi"),
            _ => return Err(TamGpuError::Compile(
                "scatter_multi_phi: WGSL backend not yet supported".into()
            )),
        };
        let kernel = self.gpu.compile(&source, entry)?;
        let arc = Arc::new(kernel);
        let mut cache = self.cache.lock().unwrap();
        let entry_ref = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry_ref))
    }

    // ─────────────────────────────────────────────────────────────────────
    // map_phi: output[i] = phi(values[i])
    // ─────────────────────────────────────────────────────────────────────

    /// Element-wise map: `output[i] = φ(values[i])`.
    ///
    /// No grouping, no atomics. Each element is independent.
    /// Variable `v` is available in the phi expression.
    pub fn map_phi(
        &self,
        phi_expr: &str,
        values: &[f64],
    ) -> TamResult<Vec<f64>> {
        let n = values.len();
        if n == 0 { return Ok(vec![]); }

        let kernel = self.get_or_compile_map(phi_expr)?;
        let v_buf = upload::<f64>(&*self.gpu, values)?;
        let o_buf = self.gpu.alloc(n * 8)?;
        let p_buf = upload::<i32>(&*self.gpu, &[n as i32])?;

        let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        self.gpu.dispatch(&*kernel, [n_blocks, 1, 1], [BLOCK_SIZE, 1, 1],
            &[&v_buf, &o_buf, &p_buf], 0)?;
        self.gpu.sync()?;
        download::<f64>(&*self.gpu, &o_buf, n)
    }

    fn get_or_compile_map(&self, phi_expr: &str) -> TamResult<Arc<Kernel>> {
        let key = format!("map:{}", phi_expr);
        {
            let cache = self.cache.lock().unwrap();
            if let Some(k) = cache.get(&key) {
                return Ok(Arc::clone(k));
            }
        }

        let (source, entry) = match self.gpu.shader_lang() {
            ShaderLang::Cuda => (gen_map_phi_cuda(phi_expr), "map_phi_kernel"),
            ShaderLang::Cpu  => (gen_map_phi_cpu(phi_expr), "map_phi"),
            _ => return Err(TamGpuError::Compile("map_phi: WGSL not yet supported".into())),
        };
        let kernel = self.gpu.compile(&source, entry)?;
        let arc = Arc::new(kernel);
        let mut cache = self.cache.lock().unwrap();
        let entry_ref = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry_ref))
    }

    // ─────────────────────────────────────────────────────────────────────
    // map_phi2: output[i] = phi(a[i], b[i])
    // ─────────────────────────────────────────────────────────────────────

    /// Two-input element-wise map: `output[i] = φ(a[i], b[i])`.
    ///
    /// Variables `a` and `b` available in the phi expression.
    /// This is the gradient communication primitive.
    pub fn map_phi2(
        &self,
        phi_expr: &str,
        a: &[f64],
        b: &[f64],
    ) -> TamResult<Vec<f64>> {
        let n = a.len();
        assert_eq!(b.len(), n, "a and b must have same length");
        if n == 0 { return Ok(vec![]); }

        let kernel = self.get_or_compile_map2(phi_expr)?;
        let a_buf = upload::<f64>(&*self.gpu, a)?;
        let b_buf = upload::<f64>(&*self.gpu, b)?;
        let o_buf = self.gpu.alloc(n * 8)?;
        let p_buf = upload::<i32>(&*self.gpu, &[n as i32])?;

        let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        self.gpu.dispatch(&*kernel, [n_blocks, 1, 1], [BLOCK_SIZE, 1, 1],
            &[&a_buf, &b_buf, &o_buf, &p_buf], 0)?;
        self.gpu.sync()?;
        download::<f64>(&*self.gpu, &o_buf, n)
    }

    fn get_or_compile_map2(&self, phi_expr: &str) -> TamResult<Arc<Kernel>> {
        let key = format!("map2:{}", phi_expr);
        {
            let cache = self.cache.lock().unwrap();
            if let Some(k) = cache.get(&key) {
                return Ok(Arc::clone(k));
            }
        }

        let (source, entry) = match self.gpu.shader_lang() {
            ShaderLang::Cuda => (gen_map_phi2_cuda(phi_expr), "map_phi2_kernel"),
            ShaderLang::Cpu  => (gen_map_phi2_cpu(phi_expr), "map_phi2"),
            _ => return Err(TamGpuError::Compile("map_phi2: WGSL not yet supported".into())),
        };
        let kernel = self.gpu.compile(&source, entry)?;
        let arc = Arc::new(kernel);
        let mut cache = self.cache.lock().unwrap();
        let entry_ref = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry_ref))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CUDA source generators
// ═══════════════════════════════════════════════════════════════════════════
// These produce identical kernels to scatter_jit.rs but with buffer-based
// parameter passing (const int* params instead of int n) for TamGpu compat.

fn gen_scatter_phi_cuda(phi_expr: &str) -> String {
    format!(r#"
// Scatter phi kernel: {phi}
extern "C" __global__ void scatter_phi(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    const double* __restrict__ refs,
    double* __restrict__ output,
    const int* __restrict__ params
) {{
    int n = params[0];
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

fn gen_scatter_multi_phi_cuda(phi_exprs: &[&str]) -> String {
    let n = phi_exprs.len();
    let out_params: String = (0..n)
        .map(|i| format!("    double* __restrict__ out{}", i))
        .collect::<Vec<_>>()
        .join(",\n");
    let atomic_adds: String = phi_exprs.iter().enumerate()
        .map(|(i, phi)| format!("        atomicAdd(&out{}[g], ({}));", i, phi))
        .collect::<Vec<_>>()
        .join("\n");

    format!(r#"
// Scatter multi-phi kernel: {phis}
extern "C" __global__ void scatter_multi_phi(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    const double* __restrict__ refs,
{out_params},
    const int* __restrict__ params
) {{
    int n = params[0];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {{
        int g = keys[gid];
        double v = values[gid];
        double r = refs[g];
{atomic_adds}
    }}
}}
"#,
        phis = phi_exprs.join("|"),
        out_params = out_params,
        atomic_adds = atomic_adds,
    )
}

fn gen_map_phi_cuda(phi_expr: &str) -> String {
    format!(r#"
// Map phi kernel: {phi}
extern "C" __global__ void map_phi_kernel(
    const double* __restrict__ values,
    double* __restrict__ output,
    const int* __restrict__ params
) {{
    int n = params[0];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {{
        double v = values[gid];
        output[gid] = ({phi});
    }}
}}
"#, phi = phi_expr)
}

fn gen_map_phi2_cuda(phi_expr: &str) -> String {
    format!(r#"
// Map phi2 kernel: {phi}
extern "C" __global__ void map_phi2_kernel(
    const double* __restrict__ vals_a,
    const double* __restrict__ vals_b,
    double* __restrict__ output,
    const int* __restrict__ params
) {{
    int n = params[0];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {{
        double a = vals_a[gid];
        double b = vals_b[gid];
        output[gid] = ({phi});
    }}
}}
"#, phi = phi_expr)
}

// ═══════════════════════════════════════════════════════════════════════════
// CPU source generators (comment-based — CpuBackend parses these)
// ═══════════════════════════════════════════════════════════════════════════

fn gen_scatter_phi_cpu(phi_expr: &str) -> String {
    format!("// Scatter phi kernel: {}", phi_expr)
}

fn gen_scatter_multi_phi_cpu(phi_exprs: &[&str]) -> String {
    format!("// Scatter multi-phi kernel: {}", phi_exprs.join("|"))
}

fn gen_map_phi_cpu(phi_expr: &str) -> String {
    format!("// Map phi kernel: {}", phi_expr)
}

fn gen_map_phi2_cpu(phi_expr: &str) -> String {
    format!("// Map phi2 kernel: {}", phi_expr)
}

// ═══════════════════════════════════════════════════════════════════════════
// Re-export well-known phi constants (same values as scatter_jit)
// ═══════════════════════════════════════════════════════════════════════════

/// φ = v: scatter sum.
pub const PHI_SUM: &str = "v";

/// φ = v*v: scatter sum of squares.
pub const PHI_SUM_SQ: &str = "v * v";

/// φ = v - r: scatter centered sum (r = group mean from previous pass).
pub const PHI_CENTERED_SUM: &str = "v - r";

/// φ = (v-r)*(v-r): scatter centered sum of squares.
pub const PHI_CENTERED_SUM_SQ: &str = "(v - r) * (v - r)";

/// φ = 1.0: scatter count.
pub const PHI_COUNT: &str = "1.0";

// ═══════════════════════════════════════════════════════════════════════════
// Tests — all on CpuBackend (no GPU required)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use tam_gpu::CpuBackend;

    fn cpu_engine() -> ScatterEngine {
        ScatterEngine::new(Arc::new(CpuBackend::new()))
    }

    // ── scatter_phi ──────────────────────────────────────────────────────

    #[test]
    fn scatter_phi_sum() {
        let e = cpu_engine();
        let keys = vec![0i32, 0, 1, 1, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = e.scatter_phi(PHI_SUM, &keys, &values, None, 3).unwrap();
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 7.0).abs() < 1e-10);
        assert!((result[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn scatter_phi_sum_sq() {
        let e = cpu_engine();
        let keys = vec![0i32, 0];
        let values = vec![2.0, 3.0];
        let result = e.scatter_phi(PHI_SUM_SQ, &keys, &values, None, 1).unwrap();
        assert!((result[0] - 13.0).abs() < 1e-10); // 4 + 9
    }

    #[test]
    fn scatter_phi_centered_sum() {
        let e = cpu_engine();
        let keys = vec![0i32, 0, 1];
        let values = vec![1.0, 3.0, 10.0];
        let refs = vec![2.0, 10.0];
        let result = e.scatter_phi(PHI_CENTERED_SUM, &keys, &values, Some(&refs), 2).unwrap();
        // group 0: (1-2) + (3-2) = 0
        // group 1: (10-10) = 0
        assert!(result[0].abs() < 1e-10);
        assert!(result[1].abs() < 1e-10);
    }

    #[test]
    fn scatter_phi_count() {
        let e = cpu_engine();
        let keys = vec![0i32, 0, 1, 1, 1];
        let values = vec![99.0; 5];
        let result = e.scatter_phi(PHI_COUNT, &keys, &values, None, 2).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn scatter_phi_centered_sum_sq() {
        let e = cpu_engine();
        let keys = vec![0i32, 0];
        let values = vec![1.0, 5.0];
        let refs = vec![3.0]; // mean
        let result = e.scatter_phi(PHI_CENTERED_SUM_SQ, &keys, &values, Some(&refs), 1).unwrap();
        // (1-3)² + (5-3)² = 4 + 4 = 8
        assert!((result[0] - 8.0).abs() < 1e-10);
    }

    // ── scatter_multi_phi ────────────────────────────────────────────────

    #[test]
    fn multi_phi_sum_sq_count() {
        let e = cpu_engine();
        let keys = vec![0i32, 0, 1, 1, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let r = e.scatter_multi_phi(
            &[PHI_SUM, PHI_SUM_SQ, PHI_COUNT],
            &keys, &values, None, 3,
        ).unwrap();

        // group 0: sum=3, sum_sq=5, count=2
        assert!((r[0][0] - 3.0).abs() < 1e-10);
        assert!((r[1][0] - 5.0).abs() < 1e-10);
        assert!((r[2][0] - 2.0).abs() < 1e-10);
        // group 1: sum=7, sum_sq=25, count=2
        assert!((r[0][1] - 7.0).abs() < 1e-10);
        assert!((r[1][1] - 25.0).abs() < 1e-10);
        assert!((r[2][1] - 2.0).abs() < 1e-10);
        // group 2: sum=5, sum_sq=25, count=1
        assert!((r[0][2] - 5.0).abs() < 1e-10);
        assert!((r[1][2] - 25.0).abs() < 1e-10);
        assert!((r[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn multi_phi_two_outputs() {
        let e = cpu_engine();
        let keys = vec![0i32, 0];
        let values = vec![3.0, 4.0];

        let r = e.scatter_multi_phi(&[PHI_SUM, PHI_SUM_SQ], &keys, &values, None, 1).unwrap();
        assert!((r[0][0] - 7.0).abs() < 1e-10);
        assert!((r[1][0] - 25.0).abs() < 1e-10);
    }

    // ── map_phi ──────────────────────────────────────────────────────────

    #[test]
    fn map_phi_identity() {
        let e = cpu_engine();
        let result = e.map_phi("v", &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn map_phi_square() {
        let e = cpu_engine();
        let result = e.map_phi("v * v", &[2.0, 3.0, 4.0]).unwrap();
        assert_eq!(result, vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn map_phi_exp() {
        let e = cpu_engine();
        let result = e.map_phi("exp(v)", &[0.0, 1.0]).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn map_phi_empty() {
        let e = cpu_engine();
        let result = e.map_phi("v", &[]).unwrap();
        assert!(result.is_empty());
    }

    // ── map_phi2 ─────────────────────────────────────────────────────────

    #[test]
    fn map_phi2_multiply() {
        let e = cpu_engine();
        let a = vec![2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0];
        let result = e.map_phi2("a * b", &a, &b).unwrap();
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 18.0).abs() < 1e-10);
        assert!((result[2] - 28.0).abs() < 1e-10);
    }

    #[test]
    fn map_phi2_relu_backward() {
        let e = cpu_engine();
        let upstream = vec![1.0, 2.0, 3.0, 4.0];
        let activations = vec![-0.5, 0.3, -1.0, 2.0];
        let grad = e.map_phi2("b > 0.0 ? a : 0.0", &upstream, &activations).unwrap();
        assert_eq!(grad[0], 0.0);
        assert!((grad[1] - 2.0).abs() < 1e-10);
        assert_eq!(grad[2], 0.0);
        assert!((grad[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn map_phi2_sgd_update() {
        let e = cpu_engine();
        let lr = 0.1_f64;
        let params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.5, 1.0, 1.5];
        let phi = format!("a - {lr:.17} * b");
        let updated = e.map_phi2(&phi, &params, &grads).unwrap();
        assert!((updated[0] - 0.95).abs() < 1e-10);
        assert!((updated[1] - 1.90).abs() < 1e-10);
        assert!((updated[2] - 2.85).abs() < 1e-10);
    }

    #[test]
    fn map_phi2_empty() {
        let e = cpu_engine();
        let result = e.map_phi2("a + b", &[], &[]).unwrap();
        assert!(result.is_empty());
    }

    // ── cache ────────────────────────────────────────────────────────────

    #[test]
    fn kernel_cache_works() {
        let e = cpu_engine();
        assert_eq!(e.cache_len(), 0);
        e.scatter_phi(PHI_SUM, &[0i32], &[1.0], None, 1).unwrap();
        assert_eq!(e.cache_len(), 1);
        // Same phi → cache hit
        e.scatter_phi(PHI_SUM, &[0i32], &[2.0], None, 1).unwrap();
        assert_eq!(e.cache_len(), 1);
        // Different phi → new entry
        e.scatter_phi(PHI_SUM_SQ, &[0i32], &[1.0], None, 1).unwrap();
        assert_eq!(e.cache_len(), 2);
    }

    #[test]
    fn multi_phi_cached_separately() {
        let e = cpu_engine();
        e.scatter_phi(PHI_SUM, &[0i32], &[1.0], None, 1).unwrap();
        e.scatter_multi_phi(&[PHI_SUM, PHI_COUNT], &[0i32], &[1.0], None, 1).unwrap();
        assert_eq!(e.cache_len(), 2); // scatter:v and multi_scatter:2:v|1.0
    }
}
