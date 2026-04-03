//! Backend-agnostic compute engine.
//!
//! Wraps [`TamGpu`] and provides scatter/map operations that work on
//! any backend: CUDA, Vulkan, Metal, or CPU fallback.
//!
//! The GPU path uses [`codegen`](crate::codegen) to generate kernel source,
//! [`TamGpu::compile`] to JIT-compile, and [`TamGpu::dispatch`] to launch.
//!
//! The CPU path evaluates phi expressions directly in Rust — no shader
//! compilation, works everywhere.
//!
//! ```no_run
//! use tambear::compute_engine::ComputeEngine;
//! use tam_gpu::detect;
//!
//! let mut ce = ComputeEngine::new(detect());
//! let keys = vec![0i32, 0, 1, 1, 2];
//! let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let sums = ce.scatter_phi("v", &keys, &vals, None, 3).unwrap();
//! // [3.0, 7.0, 5.0]
//! ```

use std::collections::HashMap;

use std::sync::Arc;
use tam_gpu::{Backend, Buffer, Kernel, TamGpu};

use crate::codegen::CodegenTarget;

const BLOCK_SIZE: u32 = 256;

/// Backend-agnostic compute engine.
///
/// Owns a [`TamGpu`] backend and a kernel cache. All scatter/map operations
/// dispatch through the backend — CUDA, Vulkan, or CPU.
pub struct ComputeEngine {
    gpu: Arc<dyn TamGpu>,
    cache: HashMap<String, Kernel>,
}

impl ComputeEngine {
    pub fn new(gpu: Arc<dyn TamGpu>) -> Self {
        ComputeEngine { gpu, cache: HashMap::new() }
    }

    pub fn backend(&self) -> Backend {
        self.gpu.backend()
    }

    /// Returns the codegen target for the current backend, or None for CPU.
    #[allow(dead_code)]
    fn codegen_target(&self) -> Option<CodegenTarget> {
        match self.gpu.backend() {
            Backend::Cuda => Some(CodegenTarget::CudaC),
            Backend::Vulkan | Backend::Metal | Backend::Dx12 => Some(CodegenTarget::Wgsl),
            Backend::Cpu => None,
        }
    }

    // =======================================================================
    // Scatter: output[key[i]] += phi(v, r, g)
    // =======================================================================

    /// Scatter phi(v, r, g) into group accumulators.
    ///
    /// For each element i: `output[keys[i]] += phi(values[i], refs[keys[i]], keys[i])`.
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

        match self.gpu.backend() {
            Backend::Cpu => self.scatter_phi_cpu(phi_expr, keys, values, refs, n_groups),
            Backend::Cuda => self.scatter_phi_cuda(phi_expr, keys, values, refs, n_groups),
            other => Err(format!("scatter_phi not yet implemented for {:?}", other).into()),
        }
    }

    /// Fused multi-scatter: N phi expressions, one memory pass.
    pub fn scatter_multi_phi(
        &mut self,
        phi_exprs: &[&str],
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        n_groups: usize,
    ) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let n_phi = phi_exprs.len();
        assert!(n_phi >= 1 && n_phi <= 5, "1-5 phi expressions supported, got {}", n_phi);
        let n = keys.len();
        assert_eq!(n, values.len());

        match self.gpu.backend() {
            Backend::Cpu => self.scatter_multi_phi_cpu(phi_exprs, keys, values, refs, n_groups),
            Backend::Cuda => self.scatter_multi_phi_cuda(phi_exprs, keys, values, refs, n_groups),
            other => Err(format!("scatter_multi_phi not yet implemented for {:?}", other).into()),
        }
    }

    // =======================================================================
    // Scatter Masked: output[key[i]] += phi(v, r, g) where mask bit is set
    // =======================================================================

    /// Masked scatter: only accumulate rows where `mask` bit is 1.
    ///
    /// `mask` is a packed u64 bitmask: bit `i` of `mask[i/64]` controls row i.
    pub fn scatter_phi_masked(
        &mut self,
        phi_expr: &str,
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        mask: &[u64],
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = keys.len();
        assert_eq!(n, values.len());
        assert!(mask.len() >= (n + 63) / 64, "mask too short");

        match self.gpu.backend() {
            Backend::Cpu => self.scatter_phi_masked_cpu(phi_expr, keys, values, refs, mask, n_groups),
            Backend::Cuda => self.scatter_phi_masked_cuda(phi_expr, keys, values, refs, mask, n_groups),
            other => Err(format!("scatter_phi_masked not yet implemented for {:?}", other).into()),
        }
    }

    // =======================================================================
    // Extremum scatter: output[g] = max/min(output[g], v) for elements in group g
    // =======================================================================

    /// Scatter per-group extremum (max or min) using a CAS-loop atomic.
    ///
    /// `is_max = true` → max; `is_max = false` → min.
    /// Output is initialized to -∞ (max) or +∞ (min) before scattering.
    /// `keys[i]` must be in `[0, n_groups)`.
    pub fn scatter_extremum(
        &mut self,
        is_max: bool,
        keys: &[i32],
        values: &[f64],
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        assert_eq!(keys.len(), values.len(), "keys and values must have same length");
        match self.gpu.backend() {
            Backend::Cpu  => self.scatter_extremum_cpu(is_max, keys, values, n_groups),
            Backend::Cuda => self.scatter_extremum_cuda(is_max, keys, values, n_groups),
            other => Err(format!("scatter_extremum not yet implemented for {:?}", other).into()),
        }
    }

    // =======================================================================
    // Map: output[i] = phi(values[i])
    // =======================================================================

    /// Element-wise unary map: `output[i] = phi(values[i])`.
    pub fn map_phi(
        &mut self,
        phi_expr: &str,
        values: &[f64],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        match self.gpu.backend() {
            Backend::Cpu => self.map_phi_cpu(phi_expr, values),
            Backend::Cuda => self.map_phi_cuda(phi_expr, values),
            other => Err(format!("map_phi not yet implemented for {:?}", other).into()),
        }
    }

    /// Element-wise binary map: `output[i] = phi(a[i], b[i])`.
    pub fn map_phi2(
        &mut self,
        phi_expr: &str,
        a: &[f64],
        b: &[f64],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        assert_eq!(a.len(), b.len(), "a and b must have same length");
        match self.gpu.backend() {
            Backend::Cpu => self.map_phi2_cpu(phi_expr, a, b),
            Backend::Cuda => self.map_phi2_cuda(phi_expr, a, b),
            other => Err(format!("map_phi2 not yet implemented for {:?}", other).into()),
        }
    }

    // =======================================================================
    // CPU path — direct Rust evaluation
    // =======================================================================

    fn scatter_phi_cpu(
        &self,
        phi_expr: &str,
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let eval = phi_scatter_eval(phi_expr)?;
        let zero_refs = vec![0.0f64; n_groups];
        let refs = refs.unwrap_or(&zero_refs);
        let mut output = vec![0.0f64; n_groups];
        for i in 0..keys.len() {
            let g = keys[i] as usize;
            output[g] += eval(values[i], refs[g]);
        }
        Ok(output)
    }

    fn scatter_multi_phi_cpu(
        &self,
        phi_exprs: &[&str],
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        n_groups: usize,
    ) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let evals: Vec<_> = phi_exprs.iter()
            .map(|p| phi_scatter_eval(p))
            .collect::<Result<_, _>>()?;
        let zero_refs = vec![0.0f64; n_groups];
        let refs = refs.unwrap_or(&zero_refs);
        let mut outputs: Vec<Vec<f64>> = (0..evals.len()).map(|_| vec![0.0f64; n_groups]).collect();
        for i in 0..keys.len() {
            let g = keys[i] as usize;
            let v = values[i];
            let r = refs[g];
            for (j, eval) in evals.iter().enumerate() {
                outputs[j][g] += eval(v, r);
            }
        }
        Ok(outputs)
    }

    fn scatter_phi_masked_cpu(
        &self,
        phi_expr: &str,
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        mask: &[u64],
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let eval = phi_scatter_eval(phi_expr)?;
        let zero_refs = vec![0.0f64; n_groups];
        let refs = refs.unwrap_or(&zero_refs);
        let mut output = vec![0.0f64; n_groups];
        for i in 0..keys.len() {
            let word = mask[i / 64];
            if (word >> (i % 64)) & 1 == 0 { continue; }
            let g = keys[i] as usize;
            output[g] += eval(values[i], refs[g]);
        }
        Ok(output)
    }

    fn scatter_extremum_cpu(
        &self,
        is_max: bool,
        keys: &[i32],
        values: &[f64],
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let init = if is_max { f64::NEG_INFINITY } else { f64::INFINITY };
        let mut output = vec![init; n_groups];
        for i in 0..keys.len() {
            let g = keys[i] as usize;
            if is_max {
                if values[i] > output[g] { output[g] = values[i]; }
            } else {
                if values[i] < output[g] { output[g] = values[i]; }
            }
        }
        Ok(output)
    }

    fn map_phi_cpu(
        &self,
        phi_expr: &str,
        values: &[f64],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let eval = phi_map_eval(phi_expr)?;
        Ok(values.iter().map(|&v| eval(v)).collect())
    }

    fn map_phi2_cpu(
        &self,
        phi_expr: &str,
        a: &[f64],
        b: &[f64],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let eval = phi_map2_eval(phi_expr)?;
        Ok(a.iter().zip(b.iter()).map(|(&a, &b)| eval(a, b)).collect())
    }

    // =======================================================================
    // CUDA path — codegen → compile → dispatch
    // =======================================================================

    fn scatter_phi_cuda(
        &mut self,
        phi_expr: &str,
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = keys.len();
        let cache_key = format!("scatter:{}:{}", phi_expr, n);

        if !self.cache.contains_key(&cache_key) {
            let source = emit_scatter_cuda_tamgpu(phi_expr, n);
            let kernel = self.gpu.compile(&source, "scatter_phi")?;
            self.cache.insert(cache_key.clone(), kernel);
        }

        let zero_refs = vec![0.0f64; n_groups];
        let refs_data = refs.unwrap_or(&zero_refs);

        let b_keys = tam_gpu::upload(&*self.gpu, keys)?;
        let b_vals = tam_gpu::upload(&*self.gpu, values)?;
        let b_refs = tam_gpu::upload(&*self.gpu, refs_data)?;
        let b_out = self.gpu.alloc(n_groups * 8)?;

        let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let kernel = self.cache.get(&cache_key).unwrap();
        self.gpu.dispatch(
            kernel,
            [n_blocks, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &[&b_keys, &b_vals, &b_refs, &b_out],
            0,
        )?;
        self.gpu.sync()?;

        Ok(tam_gpu::download(&*self.gpu, &b_out, n_groups)?)
    }

    fn scatter_multi_phi_cuda(
        &mut self,
        phi_exprs: &[&str],
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        n_groups: usize,
    ) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let n = keys.len();
        let n_phi = phi_exprs.len();
        let cache_key = format!("scatter_multi:{}:{}", phi_exprs.join("|"), n);

        if !self.cache.contains_key(&cache_key) {
            let source = emit_scatter_multi_cuda_tamgpu(phi_exprs, n);
            let kernel = self.gpu.compile(&source, "scatter_multi_phi")?;
            self.cache.insert(cache_key.clone(), kernel);
        }

        let zero_refs = vec![0.0f64; n_groups];
        let refs_data = refs.unwrap_or(&zero_refs);

        let b_keys = tam_gpu::upload(&*self.gpu, keys)?;
        let b_vals = tam_gpu::upload(&*self.gpu, values)?;
        let b_refs = tam_gpu::upload(&*self.gpu, refs_data)?;

        let mut out_bufs: Vec<Buffer> = Vec::new();
        for _ in 0..n_phi {
            out_bufs.push(self.gpu.alloc(n_groups * 8)?);
        }

        // Build buffer slice: keys, values, refs, out0..outN-1
        let mut all_bufs: Vec<&Buffer> = vec![&b_keys, &b_vals, &b_refs];
        for ob in &out_bufs {
            all_bufs.push(ob);
        }

        let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let kernel = self.cache.get(&cache_key).unwrap();
        self.gpu.dispatch(
            kernel,
            [n_blocks, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &all_bufs,
            0,
        )?;
        self.gpu.sync()?;

        let mut results = Vec::new();
        for ob in &out_bufs {
            results.push(tam_gpu::download(&*self.gpu, ob, n_groups)?);
        }
        Ok(results)
    }

    fn scatter_phi_masked_cuda(
        &mut self,
        phi_expr: &str,
        keys: &[i32],
        values: &[f64],
        refs: Option<&[f64]>,
        mask: &[u64],
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = keys.len();
        let cache_key = format!("scatter_masked:{}:{}", phi_expr, n);

        if !self.cache.contains_key(&cache_key) {
            let source = emit_scatter_masked_cuda_tamgpu(phi_expr, n);
            let kernel = self.gpu.compile(&source, "scatter_phi_masked")?;
            self.cache.insert(cache_key.clone(), kernel);
        }

        let zero_refs = vec![0.0f64; n_groups];
        let refs_data = refs.unwrap_or(&zero_refs);

        let b_keys = tam_gpu::upload(&*self.gpu, keys)?;
        let b_vals = tam_gpu::upload(&*self.gpu, values)?;
        let b_refs = tam_gpu::upload(&*self.gpu, refs_data)?;
        let b_mask = tam_gpu::upload(&*self.gpu, mask)?;
        let b_out = self.gpu.alloc(n_groups * 8)?;

        let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let kernel = self.cache.get(&cache_key).unwrap();
        self.gpu.dispatch(
            kernel,
            [n_blocks, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &[&b_keys, &b_vals, &b_refs, &b_mask, &b_out],
            0,
        )?;
        self.gpu.sync()?;

        Ok(tam_gpu::download(&*self.gpu, &b_out, n_groups)?)
    }

    fn scatter_extremum_cuda(
        &mut self,
        is_max: bool,
        keys: &[i32],
        values: &[f64],
        n_groups: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = keys.len();
        let op_name = if is_max { "max" } else { "min" };
        let cache_key = format!("scatter_extremum:{}:{}", op_name, n);

        if !self.cache.contains_key(&cache_key) {
            let source = emit_scatter_extremum_cuda_tamgpu(is_max, n);
            let kernel = self.gpu.compile(&source, "scatter_extremum")?;
            self.cache.insert(cache_key.clone(), kernel);
        }

        let b_keys = tam_gpu::upload(&*self.gpu, keys)?;
        let b_vals = tam_gpu::upload(&*self.gpu, values)?;
        // Output must start at -inf (max) or +inf (min) — not zeros.
        let init: Vec<f64> = if is_max {
            vec![f64::NEG_INFINITY; n_groups]
        } else {
            vec![f64::INFINITY; n_groups]
        };
        let b_out = tam_gpu::upload(&*self.gpu, &init)?;

        let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let kernel = self.cache.get(&cache_key).unwrap();
        self.gpu.dispatch(
            kernel,
            [n_blocks, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &[&b_keys, &b_vals, &b_out],
            0,
        )?;
        self.gpu.sync()?;

        Ok(tam_gpu::download(&*self.gpu, &b_out, n_groups)?)
    }

    fn map_phi_cuda(
        &mut self,
        phi_expr: &str,
        values: &[f64],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = values.len();
        let cache_key = format!("map:{}:{}", phi_expr, n);

        if !self.cache.contains_key(&cache_key) {
            let source = emit_map_cuda_tamgpu(phi_expr, n);
            let kernel = self.gpu.compile(&source, "map_phi_kernel")?;
            self.cache.insert(cache_key.clone(), kernel);
        }

        let b_vals = tam_gpu::upload(&*self.gpu, values)?;
        let b_out = self.gpu.alloc(n * 8)?;

        let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let kernel = self.cache.get(&cache_key).unwrap();
        self.gpu.dispatch(
            kernel,
            [n_blocks, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &[&b_vals, &b_out],
            0,
        )?;
        self.gpu.sync()?;

        Ok(tam_gpu::download(&*self.gpu, &b_out, n)?)
    }

    fn map_phi2_cuda(
        &mut self,
        phi_expr: &str,
        a: &[f64],
        b: &[f64],
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let n = a.len();
        let cache_key = format!("map2:{}:{}", phi_expr, n);

        if !self.cache.contains_key(&cache_key) {
            let source = emit_map2_cuda_tamgpu(phi_expr, n);
            let kernel = self.gpu.compile(&source, "map_phi2_kernel")?;
            self.cache.insert(cache_key.clone(), kernel);
        }

        let b_a = tam_gpu::upload(&*self.gpu, a)?;
        let b_b = tam_gpu::upload(&*self.gpu, b)?;
        let b_out = self.gpu.alloc(n * 8)?;

        let n_blocks = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let kernel = self.cache.get(&cache_key).unwrap();
        self.gpu.dispatch(
            kernel,
            [n_blocks, 1, 1],
            [BLOCK_SIZE, 1, 1],
            &[&b_a, &b_b, &b_out],
            0,
        )?;
        self.gpu.sync()?;

        Ok(tam_gpu::download(&*self.gpu, &b_out, n)?)
    }
}

// ===========================================================================
// CPU phi evaluators
// ===========================================================================

/// Parse a scatter phi expression into a Rust closure.
/// Variables: v (value), r (ref).
fn phi_scatter_eval(phi_expr: &str) -> Result<Box<dyn Fn(f64, f64) -> f64>, Box<dyn std::error::Error>> {
    let f: Box<dyn Fn(f64, f64) -> f64> = match phi_expr {
        "v" => Box::new(|v, _r| v),
        "v * v" => Box::new(|v, _r| v * v),
        "1.0" => Box::new(|_v, _r| 1.0),
        "(v - r)" | "v - r" => Box::new(|v, r| v - r),
        "(v - r) * (v - r)" => Box::new(|v, r| (v - r) * (v - r)),
        "fabs(v - r)" | "abs(v - r)" => Box::new(|v, r| (v - r).abs()),
        "(v - r) * (v - r) * (v - r)" => Box::new(|v, r| { let d = v - r; d * d * d }),
        "(v - r) * (v - r) * (v - r) * (v - r)" => Box::new(|v, r| { let d = v - r; let d2 = d * d; d2 * d2 }),
        "v * r" => Box::new(|v, r| v * r),
        "log(v)" => Box::new(|v, _r| v.ln()),
        "1.0 / v" => Box::new(|v, _r| 1.0 / v),
        other => return Err(format!("CPU: unsupported scatter phi '{}'", other).into()),
    };
    Ok(f)
}

/// Parse a unary map phi expression into a Rust closure. Variable: v.
fn phi_map_eval(phi_expr: &str) -> Result<Box<dyn Fn(f64) -> f64>, Box<dyn std::error::Error>> {
    let f: Box<dyn Fn(f64) -> f64> = match phi_expr {
        "v" => Box::new(|v| v),
        "v * v" => Box::new(|v| v * v),
        "v * v + 1.0" => Box::new(|v| v * v + 1.0),
        "log(v)" => Box::new(|v| v.ln()),
        "exp(v)" => Box::new(|v| v.exp()),
        "sqrt(v)" => Box::new(|v| v.sqrt()),
        "abs(v)" | "fabs(v)" => Box::new(|v| v.abs()),
        other => return Err(format!("CPU: unsupported map phi '{}'", other).into()),
    };
    Ok(f)
}

/// Parse a binary map phi expression into a Rust closure. Variables: a, b.
fn phi_map2_eval(phi_expr: &str) -> Result<Box<dyn Fn(f64, f64) -> f64>, Box<dyn std::error::Error>> {
    let f: Box<dyn Fn(f64, f64) -> f64> = match phi_expr {
        "a + b" => Box::new(|a, b| a + b),
        "a * b" => Box::new(|a, b| a * b),
        "a - b" => Box::new(|a, b| a - b),
        "a * b + 1.0" => Box::new(|a, b| a * b + 1.0),
        other => return Err(format!("CPU: unsupported map2 phi '{}'", other).into()),
    };
    Ok(f)
}

// ===========================================================================
// TamGpu-compatible CUDA kernel sources (n baked as #define)
// ===========================================================================

/// Extremum scatter kernel — CAS-loop f64 atomic max or min.
///
/// Buffers: [keys:i32, values:f64, output:f64] = 3 buffers (no refs).
/// Output must be pre-initialized to -∞ (max) or +∞ (min) before launch.
///
/// The CAS loop: read current bits, compute fmax/fmin with the new value,
/// CAS the bits atomically. If the comparison wouldn't change the value,
/// break early — no unnecessary CAS.
fn emit_scatter_extremum_cuda_tamgpu(is_max: bool, n: usize) -> String {
    let cmp_fn = if is_max { "fmax" } else { "fmin" };
    let op_comment = if is_max { "max" } else { "min" };
    format!(
        r#"
#define PARAM_N {n}
// JIT scatter extremum ({op}): output[g] = {cmp}(output[g], v). n baked for TamGpu dispatch.
// Output must be initialised to -{init_sign}infinity before launch.
// CAS-loop: no native f64 atomicMax/Min in CUDA — use atomicCAS on bit patterns.
extern "C" __global__ void scatter_extremum(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    double* __restrict__ output
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < PARAM_N) {{
        int g = keys[gid];
        double val = values[gid];
        unsigned long long* addr = (unsigned long long*)(&output[g]);
        unsigned long long old_bits = *addr, assumed, new_bits;
        do {{
            assumed = old_bits;
            double new_val = {cmp}(__longlong_as_double(assumed), val);
            new_bits = __double_as_longlong(new_val);
            if (new_bits == assumed) break;
            old_bits = atomicCAS(addr, assumed, new_bits);
        }} while (assumed != old_bits);
    }}
}}
"#,
        n = n,
        op = op_comment,
        cmp = cmp_fn,
        init_sign = if is_max { "" } else { "+" },
    )
}

/// Scatter kernel with n baked in — no scalar args, TamGpu-compatible.
/// Buffers: [keys:i32, values:f64, refs:f64, output:f64] = 4 buffers.
fn emit_scatter_cuda_tamgpu(phi_expr: &str, n: usize) -> String {
    format!(
        r#"
#define PARAM_N {n}
// JIT scatter: output[g] += phi(v). n baked for TamGpu dispatch.
// phi = {phi}
extern "C" __global__ void scatter_phi(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    const double* __restrict__ refs,
    double* __restrict__ output
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < PARAM_N) {{
        int g = keys[gid];
        double v = values[gid];
        double r = refs[g];
        double phi = ({phi});
        atomicAdd(&output[g], phi);
    }}
}}
"#,
        n = n,
        phi = phi_expr,
    )
}

/// Multi-scatter with n baked in.
/// Buffers: [keys:i32, values:f64, refs:f64, out0:f64, ..., outN-1:f64]
fn emit_scatter_multi_cuda_tamgpu(phi_exprs: &[&str], n: usize) -> String {
    let n_phi = phi_exprs.len();
    let out_params: String = (0..n_phi)
        .map(|i| format!("    double* __restrict__ out{}", i))
        .collect::<Vec<_>>()
        .join(",\n");

    let atomic_adds: String = phi_exprs
        .iter()
        .enumerate()
        .map(|(i, phi)| format!("        atomicAdd(&out{}[g], ({}));", i, phi))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"
#define PARAM_N {n}
// JIT fused scatter: {n_phi} outputs, n baked for TamGpu dispatch.
extern "C" __global__ void scatter_multi_phi(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    const double* __restrict__ refs,
{out_params}
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < PARAM_N) {{
        int g = keys[gid];
        double v = values[gid];
        double r = refs[g];
{atomic_adds}
    }}
}}
"#,
        n = n,
        n_phi = n_phi,
        out_params = out_params,
        atomic_adds = atomic_adds,
    )
}

/// Map kernel with n baked in. Buffers: [values:f64, output:f64] = 2 buffers.
fn emit_map_cuda_tamgpu(phi_expr: &str, n: usize) -> String {
    format!(
        r#"
#define PARAM_N {n}
// JIT map: output[i] = phi(v[i]). n baked for TamGpu dispatch.
// phi = {phi}
extern "C" __global__ void map_phi_kernel(
    const double* __restrict__ values,
    double* __restrict__ output
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < PARAM_N) {{
        double v = values[gid];
        output[gid] = ({phi});
    }}
}}
"#,
        n = n,
        phi = phi_expr,
    )
}

/// Map2 kernel with n baked in. Buffers: [vals_a:f64, vals_b:f64, output:f64] = 3 buffers.
fn emit_map2_cuda_tamgpu(phi_expr: &str, n: usize) -> String {
    format!(
        r#"
#define PARAM_N {n}
// JIT map2: output[i] = phi(a[i], b[i]). n baked for TamGpu dispatch.
// phi = {phi}
extern "C" __global__ void map_phi2_kernel(
    const double* __restrict__ vals_a,
    const double* __restrict__ vals_b,
    double* __restrict__ output
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < PARAM_N) {{
        double a = vals_a[gid];
        double b = vals_b[gid];
        output[gid] = ({phi});
    }}
}}
"#,
        n = n,
        phi = phi_expr,
    )
}

/// Masked scatter kernel with n baked in.
/// Buffers: [keys:i32, values:f64, refs:f64, mask:u64, output:f64] = 5 buffers.
fn emit_scatter_masked_cuda_tamgpu(phi_expr: &str, n: usize) -> String {
    format!(
        r#"
#define PARAM_N {n}
// JIT masked scatter: output[g] += phi(v) for rows where mask bit is set.
// phi = {phi}
extern "C" __global__ void scatter_phi_masked(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    const double* __restrict__ refs,
    const unsigned long long* __restrict__ mask,
    double* __restrict__ output
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < PARAM_N) {{
        unsigned long long word = mask[gid >> 6];
        if ((word >> (gid & 63)) & 1ULL) {{
            int g = keys[gid];
            double v = values[gid];
            double r = refs[g];
            atomicAdd(&output[g], ({phi}));
        }}
    }}
}}
"#,
        n = n,
        phi = phi_expr,
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tam_gpu::CpuBackend;

    fn cpu_engine() -> ComputeEngine {
        ComputeEngine::new(std::sync::Arc::new(CpuBackend::new()))
    }

    // -----------------------------------------------------------------------
    // Scatter (CPU)
    // -----------------------------------------------------------------------

    #[test]
    fn scatter_sum_cpu() {
        let mut ce = cpu_engine();
        let keys = vec![0i32, 0, 1, 1, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = ce.scatter_phi("v", &keys, &vals, None, 3).unwrap();
        assert_eq!(out, vec![3.0, 7.0, 5.0]);
    }

    #[test]
    fn scatter_sum_sq_cpu() {
        let mut ce = cpu_engine();
        let keys = vec![0i32, 0, 1];
        let vals = vec![3.0, 4.0, 5.0];
        let out = ce.scatter_phi("v * v", &keys, &vals, None, 2).unwrap();
        assert_eq!(out, vec![25.0, 25.0]); // 9+16, 25
    }

    #[test]
    fn scatter_count_cpu() {
        let mut ce = cpu_engine();
        let keys = vec![0i32, 0, 1, 2, 2, 2];
        let vals = vec![0.0; 6];
        let out = ce.scatter_phi("1.0", &keys, &vals, None, 3).unwrap();
        assert_eq!(out, vec![2.0, 1.0, 3.0]);
    }

    #[test]
    fn scatter_centered_cpu() {
        let mut ce = cpu_engine();
        let keys = vec![0i32, 0, 1, 1];
        let vals = vec![10.0, 20.0, 30.0, 40.0];
        let refs = vec![15.0, 35.0]; // group means
        let out = ce.scatter_phi("(v - r) * (v - r)", &keys, &vals, Some(&refs), 2).unwrap();
        // group 0: (10-15)^2 + (20-15)^2 = 25+25 = 50
        // group 1: (30-35)^2 + (40-35)^2 = 25+25 = 50
        assert_eq!(out, vec![50.0, 50.0]);
    }

    // -----------------------------------------------------------------------
    // Scatter Multi (CPU)
    // -----------------------------------------------------------------------

    #[test]
    fn scatter_multi_sum_and_count_cpu() {
        let mut ce = cpu_engine();
        let keys = vec![0i32, 0, 1, 1, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let results = ce.scatter_multi_phi(&["v", "1.0"], &keys, &vals, None, 3).unwrap();
        assert_eq!(results[0], vec![3.0, 7.0, 5.0]); // sums
        assert_eq!(results[1], vec![2.0, 2.0, 1.0]); // counts
    }

    #[test]
    fn scatter_multi_three_phis_cpu() {
        let mut ce = cpu_engine();
        let keys = vec![0i32, 0];
        let vals = vec![3.0, 4.0];
        let results = ce.scatter_multi_phi(&["v", "v * v", "1.0"], &keys, &vals, None, 1).unwrap();
        assert_eq!(results[0], vec![7.0]);  // 3+4
        assert_eq!(results[1], vec![25.0]); // 9+16
        assert_eq!(results[2], vec![2.0]);  // count
    }

    // -----------------------------------------------------------------------
    // Map (CPU)
    // -----------------------------------------------------------------------

    #[test]
    fn map_identity_cpu() {
        let mut ce = cpu_engine();
        let vals = vec![1.0, 2.0, 3.0];
        let out = ce.map_phi("v", &vals).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn map_square_cpu() {
        let mut ce = cpu_engine();
        let vals = vec![2.0, 3.0, 4.0];
        let out = ce.map_phi("v * v", &vals).unwrap();
        assert_eq!(out, vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn map_log_cpu() {
        let mut ce = cpu_engine();
        let vals = vec![1.0, std::f64::consts::E, std::f64::consts::E * std::f64::consts::E];
        let out = ce.map_phi("log(v)", &vals).unwrap();
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
        assert!((out[2] - 2.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Map2 (CPU)
    // -----------------------------------------------------------------------

    #[test]
    fn map2_add_cpu() {
        let mut ce = cpu_engine();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let out = ce.map_phi2("a + b", &a, &b).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn map2_mul_cpu() {
        let mut ce = cpu_engine();
        let a = vec![2.0, 3.0];
        let b = vec![5.0, 7.0];
        let out = ce.map_phi2("a * b", &a, &b).unwrap();
        assert_eq!(out, vec![10.0, 21.0]);
    }

    // -----------------------------------------------------------------------
    // CUDA kernel source generation
    // -----------------------------------------------------------------------

    #[test]
    fn cuda_tamgpu_scatter_bakes_n() {
        let src = emit_scatter_cuda_tamgpu("v * v", 1000);
        assert!(src.contains("#define PARAM_N 1000"));
        assert!(src.contains("if (gid < PARAM_N)"));
        assert!(src.contains("double phi = (v * v)"));
        assert!(src.contains("atomicAdd(&output[g], phi)"));
        // Should NOT have `int n` as a parameter
        assert!(!src.contains("int n\n") && !src.contains("int n)"));
    }

    #[test]
    fn cuda_tamgpu_scatter_multi_bakes_n() {
        let src = emit_scatter_multi_cuda_tamgpu(&["v", "v * v", "1.0"], 500);
        assert!(src.contains("#define PARAM_N 500"));
        assert!(src.contains("out0"));
        assert!(src.contains("out1"));
        assert!(src.contains("out2"));
        assert!(src.contains("atomicAdd(&out0[g], (v))"));
        assert!(src.contains("atomicAdd(&out2[g], (1.0))"));
    }

    #[test]
    fn cuda_tamgpu_map_bakes_n() {
        let src = emit_map_cuda_tamgpu("v * v", 2048);
        assert!(src.contains("#define PARAM_N 2048"));
        assert!(src.contains("output[gid] = (v * v)"));
    }

    #[test]
    fn cuda_tamgpu_map2_bakes_n() {
        let src = emit_map2_cuda_tamgpu("a * b", 4096);
        assert!(src.contains("#define PARAM_N 4096"));
        assert!(src.contains("output[gid] = (a * b)"));
    }

    // -----------------------------------------------------------------------
    // Scatter Masked (CPU)
    // -----------------------------------------------------------------------

    #[test]
    fn scatter_masked_sum_cpu() {
        let mut ce = cpu_engine();
        let keys = vec![0i32, 0, 1, 1, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // mask: bits 0,2,4 set (rows 0, 2, 4 pass)
        let mask = vec![0b10101u64]; // bits 0,2,4
        let out = ce.scatter_phi_masked("v", &keys, &vals, None, &mask, 3).unwrap();
        // row 0 (key=0, v=1.0) passes, row 2 (key=1, v=3.0) passes, row 4 (key=2, v=5.0) passes
        assert_eq!(out, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn scatter_masked_all_pass_cpu() {
        let mut ce = cpu_engine();
        let keys = vec![0i32, 0, 1];
        let vals = vec![1.0, 2.0, 3.0];
        let mask = vec![0b111u64]; // all 3 bits set
        let out = ce.scatter_phi_masked("v", &keys, &vals, None, &mask, 2).unwrap();
        assert_eq!(out, vec![3.0, 3.0]); // same as unmasked scatter
    }

    #[test]
    fn scatter_masked_none_pass_cpu() {
        let mut ce = cpu_engine();
        let keys = vec![0i32, 0, 1];
        let vals = vec![1.0, 2.0, 3.0];
        let mask = vec![0u64]; // no bits set
        let out = ce.scatter_phi_masked("v", &keys, &vals, None, &mask, 2).unwrap();
        assert_eq!(out, vec![0.0, 0.0]); // nothing passes
    }

    // -----------------------------------------------------------------------
    // CUDA kernel source: masked scatter
    // -----------------------------------------------------------------------

    #[test]
    fn cuda_tamgpu_scatter_masked_bakes_n() {
        let src = emit_scatter_masked_cuda_tamgpu("v * v", 1000);
        assert!(src.contains("#define PARAM_N 1000"));
        assert!(src.contains("unsigned long long* __restrict__ mask"));
        assert!(src.contains("mask[gid >> 6]"));
        assert!(src.contains("atomicAdd(&output[g], (v * v))"));
    }

    // -----------------------------------------------------------------------
    // Unsupported phi on CPU
    // -----------------------------------------------------------------------

    #[test]
    fn unsupported_scatter_phi_errors() {
        let mut ce = cpu_engine();
        let keys = vec![0i32];
        let vals = vec![1.0];
        let result = ce.scatter_phi("sin(v) * cos(r)", &keys, &vals, None, 1);
        assert!(result.is_err());
    }

    #[test]
    fn unsupported_map_phi_errors() {
        let mut ce = cpu_engine();
        let result = ce.map_phi("tanh(v) + 42", &[1.0]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // CUDA path (requires GPU — run with `cargo test -- --ignored`)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn scatter_sum_cuda() {
        let gpu = tam_gpu::detect();
        if gpu.backend() != Backend::Cuda { return; }
        let mut ce = ComputeEngine::new(gpu);
        let keys = vec![0i32, 0, 1, 1, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = ce.scatter_phi("v", &keys, &vals, None, 3).unwrap();
        assert_eq!(out, vec![3.0, 7.0, 5.0]);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn scatter_multi_cuda() {
        let gpu = tam_gpu::detect();
        if gpu.backend() != Backend::Cuda { return; }
        let mut ce = ComputeEngine::new(gpu);
        let keys = vec![0i32, 0, 1, 1, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let results = ce.scatter_multi_phi(&["v", "v * v", "1.0"], &keys, &vals, None, 3).unwrap();
        assert_eq!(results[0], vec![3.0, 7.0, 5.0]);
        assert!((results[1][0] - 5.0).abs() < 1e-10); // 1+4
        assert_eq!(results[2], vec![2.0, 2.0, 1.0]);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn map_square_cuda() {
        let gpu = tam_gpu::detect();
        if gpu.backend() != Backend::Cuda { return; }
        let mut ce = ComputeEngine::new(gpu);
        let vals = vec![2.0, 3.0, 4.0];
        let out = ce.map_phi("v * v", &vals).unwrap();
        assert_eq!(out, vec![4.0, 9.0, 16.0]);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn map2_add_cuda() {
        let gpu = tam_gpu::detect();
        if gpu.backend() != Backend::Cuda { return; }
        let mut ce = ComputeEngine::new(gpu);
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let out = ce.map_phi2("a + b", &a, &b).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn scatter_masked_cuda() {
        let gpu = tam_gpu::detect();
        if gpu.backend() != Backend::Cuda { return; }
        let mut ce = ComputeEngine::new(gpu);
        let keys = vec![0i32, 0, 1, 1, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // mask: bits 0,2,4 set (rows 0, 2, 4 pass)
        let mask = vec![0b10101u64];
        let out = ce.scatter_phi_masked("v", &keys, &vals, None, &mask, 3).unwrap();
        assert_eq!(out, vec![1.0, 3.0, 5.0]);
    }
}
