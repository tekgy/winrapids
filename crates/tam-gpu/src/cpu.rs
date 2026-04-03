//! CPU fallback backend — runs on any hardware via native Rust.
//!
//! Kernels are native Rust implementations selected by entry name.
//! The `compile` step validates the entry name and returns a handle.
//! The `dispatch` step executes the operation sequentially (no SIMD/Rayon yet).
//!
//! ## Supported entry names
//!
//! | Entry              | Buffers (in order)                          | n = bufs[0] element count |
//! |--------------------|---------------------------------------------|--------------------------|
//! | `scatter_sum`      | keys:i32, values:f64, output:f64            | keys.len()               |
//! | `scatter_count`    | keys:i32, output:f64                        | keys.len()               |
//! | `gather_f64`       | values:f64, rows_by_group:u32, output:f64   | rbg.len()                |
//! | `scatter_back_f64` | gathered:f64, rows_by_group:u32, output:f64 | gathered.len()           |
//! | `argmin_f64`       | values:f64, out_val:f64(1), out_idx:i32(1)  | values.len()             |
//! | `argmax_f64`       | values:f64, out_val:f64(1), out_idx:i32(1)  | values.len()             |
//! | `noop`             | (any)                                       | —                        |
//! | `tiled_accumulate` | A:f64(M×K), B:f64(K×N), C:f64(M×N), dims:i32(3) | M*K = A.len()       |
//!
//! The `tiled_accumulate` entry supports DotProduct, L2Distance, OuterProduct,
//! Covariance, SoftmaxWeighted, ManifoldDistance, and ManifoldMixture operations.
//! The specific op is determined from the first-line comment of the kernel source
//! passed to [`compile`]:
//! ```text
//! // Tiled accumulation kernel for operator: dot_product
//! ```

use std::sync::{Arc, Mutex, MutexGuard};

use bytemuck::{cast_slice, cast_slice_mut};

use crate::{Backend, Buffer, Kernel, ShaderLang, TamGpu, TamGpuError, TamResult};

// ---------------------------------------------------------------------------
// Internal buffer type
// ---------------------------------------------------------------------------

/// CPU buffer — heap memory behind an Arc<Mutex> for interior mutability.
///
/// The Mutex is needed for `dispatch(&self, ...)`: the trait takes shared refs
/// to all buffers (including outputs), so interior mutability is required to
/// write results back.
#[derive(Clone)]
pub(crate) struct CpuBuffer {
    pub data: Arc<Mutex<Vec<u8>>>,
}

impl CpuBuffer {
    fn new(bytes: usize) -> Self {
        CpuBuffer { data: Arc::new(Mutex::new(vec![0u8; bytes])) }
    }
}

// ---------------------------------------------------------------------------
// Internal kernel type
// ---------------------------------------------------------------------------

pub(crate) struct CpuKernel {
    pub entry: String,
    /// For `tiled_accumulate`: the operator name parsed from the kernel source
    /// comment (e.g. `"dot_product"`, `"l2_distance"`).
    pub op_name: Option<String>,
    /// For `tiled_accumulate`: the params key parsed from the second line
    /// (`// params: euclidean`, `// params: poincare(c=-1.0000)`, etc.).
    pub params_key: String,
}

// ---------------------------------------------------------------------------
// CpuBackend
// ---------------------------------------------------------------------------

/// CPU fallback backend.
///
/// No driver, no SDK, no toolkit. Works everywhere.
/// Operations run sequentially; GPU backends will be orders of magnitude faster.
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self { CpuBackend }
}

impl Default for CpuBackend {
    fn default() -> Self { CpuBackend }
}

impl TamGpu for CpuBackend {
    fn name(&self) -> String {
        "CPU (native Rust)".to_string()
    }

    fn backend(&self) -> Backend { Backend::Cpu }
    fn shader_lang(&self) -> ShaderLang { ShaderLang::Cpu }

    /// Look up `entry` in the CPU kernel registry.
    ///
    /// For most entries `source` is ignored. For `"tiled_accumulate"`, the
    /// first line of `source` is parsed to extract the operator name (e.g.
    /// `"dot_product"`, `"l2_distance"`).
    fn compile(&self, source: &str, entry: &str) -> TamResult<Kernel> {
        let op_name = match entry {
            "scatter_sum"      |
            "scatter_count"    |
            "gather_f64"       |
            "scatter_back_f64" |
            "argmin_f64"       |
            "argmax_f64"       |
            "noop"             => None,

            "tiled_accumulate" => {
                // Parse op name from the first-line comment emitted by
                // generate_tiled_kernel / generate_tiled_kernel_wgsl:
                //   "// Tiled accumulation kernel for operator: dot_product"
                //   "// WGSL tiled accumulation kernel for operator: l2_distance"
                // Parse params_key from the second-line comment:
                //   "// params: poincare(c=-1.0000)"
                let mut lines = source.lines();
                let first = lines.next().unwrap_or("");
                let op_name_parsed = first.rsplit("operator: ").next()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "dot_product".to_string());
                let second = lines.next().unwrap_or("");
                let params_key_parsed = second.strip_prefix("// params: ")
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                return Ok(Kernel {
                    inner: Box::new(CpuKernel {
                        entry: entry.to_string(),
                        op_name: Some(op_name_parsed),
                        params_key: params_key_parsed,
                    }),
                    entry: entry.to_string(),
                });
            }

            other => return Err(TamGpuError::EntryNotFound(format!(
                "CpuBackend has no implementation for '{}'. \
                 Supported: scatter_sum, scatter_count, gather_f64, \
                 scatter_back_f64, argmin_f64, argmax_f64, noop, \
                 tiled_accumulate", other
            ))),
        };
        Ok(Kernel {
            inner: Box::new(CpuKernel { entry: entry.to_string(), op_name, params_key: String::new() }),
            entry: entry.to_string(),
        })
    }

    fn alloc(&self, bytes: usize) -> TamResult<Buffer> {
        Ok(Buffer { inner: Box::new(CpuBuffer::new(bytes)), size: bytes })
    }

    fn free(&self, buf: Buffer) -> TamResult<()> {
        drop(buf);
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: &Buffer) -> TamResult<()> {
        let inner = cpu_buf(dst)?;
        let mut data = inner.data.lock().unwrap();
        if src.len() > data.len() {
            return Err(TamGpuError::Transfer(format!(
                "copy_h2d: src {} bytes > dst {} bytes", src.len(), data.len()
            )));
        }
        data[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, src: &Buffer, dst: &mut [u8]) -> TamResult<()> {
        let inner = cpu_buf(src)?;
        let data = inner.data.lock().unwrap();
        if dst.len() > data.len() {
            return Err(TamGpuError::Transfer(format!(
                "copy_d2h: dst {} bytes > src {} bytes", dst.len(), data.len()
            )));
        }
        dst.copy_from_slice(&data[..dst.len()]);
        Ok(())
    }

    fn dispatch(
        &self,
        kernel: &Kernel,
        _grid: [u32; 3],
        _block: [u32; 3],
        bufs: &[&Buffer],
        _shared_mem: u32,
    ) -> TamResult<()> {
        let k = kernel.inner.downcast_ref::<CpuKernel>()
            .ok_or_else(|| TamGpuError::Dispatch("not a CPU kernel".into()))?;

        let buf = |i: usize| -> TamResult<Arc<Mutex<Vec<u8>>>> {
            if i >= bufs.len() {
                return Err(TamGpuError::InvalidArgument(format!("buffer index {i} out of range ({} bufs)", bufs.len())));
            }
            Ok(cpu_buf(bufs[i])?.data.clone())
        };

        match k.entry.as_str() {
            // ----------------------------------------------------------------
            "noop" => {}

            // ----------------------------------------------------------------
            // scatter_sum: output[keys[i]] += values[i]
            // bufs: [keys:i32, values:f64, output:f64]
            "scatter_sum" => {
                let keys_arc = buf(0)?;
                let vals_arc = buf(1)?;
                let out_arc  = buf(2)?;
                // Read inputs first, release locks, then write output.
                let keys: Vec<i32> = read_as::<i32>(&keys_arc.lock().unwrap()).to_vec();
                let vals: Vec<f64> = read_as::<f64>(&vals_arc.lock().unwrap()).to_vec();
                let n = keys.len();
                let mut out_g = out_arc.lock().unwrap();
                let out: &mut [f64] = write_as::<f64>(&mut out_g);
                for i in 0..n {
                    out[keys[i] as usize] += vals[i];
                }
            }

            // ----------------------------------------------------------------
            // scatter_count: output[keys[i]] += 1.0
            // bufs: [keys:i32, output:f64]
            "scatter_count" => {
                let keys_arc = buf(0)?;
                let out_arc  = buf(1)?;
                let keys: Vec<i32> = read_as::<i32>(&keys_arc.lock().unwrap()).to_vec();
                let n = keys.len();
                let mut out_g = out_arc.lock().unwrap();
                let out: &mut [f64] = write_as::<f64>(&mut out_g);
                for i in 0..n {
                    out[keys[i] as usize] += 1.0;
                }
            }

            // ----------------------------------------------------------------
            // gather_f64: output[i] = values[rows_by_group[i]]
            // bufs: [values:f64, rows_by_group:u32, output:f64]
            "gather_f64" => {
                let vals_arc = buf(0)?;
                let rbg_arc  = buf(1)?;
                let out_arc  = buf(2)?;
                let vals: Vec<f64> = read_as::<f64>(&vals_arc.lock().unwrap()).to_vec();
                let rbg:  Vec<u32> = read_as::<u32>(&rbg_arc.lock().unwrap()).to_vec();
                let n = rbg.len();
                let mut out_g = out_arc.lock().unwrap();
                let out: &mut [f64] = write_as::<f64>(&mut out_g);
                for i in 0..n {
                    out[i] = vals[rbg[i] as usize];
                }
            }

            // ----------------------------------------------------------------
            // scatter_back_f64: output[rows_by_group[i]] = gathered[i]
            // bufs: [gathered:f64, rows_by_group:u32, output:f64]
            "scatter_back_f64" => {
                let gath_arc = buf(0)?;
                let rbg_arc  = buf(1)?;
                let out_arc  = buf(2)?;
                let gath: Vec<f64> = read_as::<f64>(&gath_arc.lock().unwrap()).to_vec();
                let rbg:  Vec<u32> = read_as::<u32>(&rbg_arc.lock().unwrap()).to_vec();
                let n = gath.len();
                let mut out_g = out_arc.lock().unwrap();
                let out: &mut [f64] = write_as::<f64>(&mut out_g);
                for i in 0..n {
                    out[rbg[i] as usize] = gath[i];
                }
            }

            // ----------------------------------------------------------------
            // argmin_f64: find (min_value, index_of_min). NaN rows excluded.
            // bufs: [values:f64, out_val:f64(≥1 elem), out_idx:i32(≥1 elem)]
            "argmin_f64" => {
                let (best_val, best_idx) = argmin_cpu(&buf(0)?.lock().unwrap());
                write_scalar_f64(&buf(1)?, best_val)?;
                write_scalar_i32(&buf(2)?, best_idx as i32)?;
            }

            // ----------------------------------------------------------------
            // argmax_f64: find (max_value, index_of_max). NaN rows excluded.
            // bufs: [values:f64, out_val:f64(≥1 elem), out_idx:i32(≥1 elem)]
            "argmax_f64" => {
                let (best_val, best_idx) = argmax_cpu(&buf(0)?.lock().unwrap());
                write_scalar_f64(&buf(1)?, best_val)?;
                write_scalar_i32(&buf(2)?, best_idx as i32)?;
            }

            // ----------------------------------------------------------------
            // tiled_accumulate: C[i,j] = op(A[i,:], B[:,j]) over K
            // bufs: [A:f64(M×K), B:f64(K×N), C:f64(M×N), dims:i32(3)]
            "tiled_accumulate" => {
                let a_arc   = buf(0)?;
                let b_arc   = buf(1)?;
                let c_arc   = buf(2)?;
                let dim_arc = buf(3)?;

                let dims: Vec<i32> = read_as::<i32>(&dim_arc.lock().unwrap()).to_vec();
                if dims.len() < 3 {
                    return Err(TamGpuError::InvalidArgument(
                        format!("tiled_accumulate: dims buffer has {} elements, need 3", dims.len())
                    ));
                }
                let m = dims[0] as usize;
                let n = dims[1] as usize;
                let kd = dims[2] as usize;

                let a: Vec<f64> = read_as::<f64>(&a_arc.lock().unwrap()).to_vec();
                let b: Vec<f64> = read_as::<f64>(&b_arc.lock().unwrap()).to_vec();

                let op = k.op_name.as_deref().unwrap_or("dot_product");

                let mut c_g = c_arc.lock().unwrap();
                let c: &mut [f64] = write_as::<f64>(&mut c_g);

                match op {
                    // GEMM: C[i,j] = sum_k(A[i,k] * B[k,j])
                    "dot_product" | "outer_product" => {
                        for i in 0..m {
                            for j in 0..n {
                                let mut acc = 0.0_f64;
                                for ki in 0..kd {
                                    acc += a[i * kd + ki] * b[ki * n + j];
                                }
                                c[i * n + j] = acc;
                            }
                        }
                    }

                    // L2 distance: C[i,j] = sum_k((A[i,k] - B[k,j])^2)
                    "l2_distance" => {
                        for i in 0..m {
                            for j in 0..n {
                                let mut acc = 0.0_f64;
                                for ki in 0..kd {
                                    let diff = a[i * kd + ki] - b[ki * n + j];
                                    acc += diff * diff;
                                }
                                c[i * n + j] = acc;
                            }
                        }
                    }

                    // Covariance: C[i,j] = sum_k(A[i,k] * B[k,j]) / (kd - 1)
                    // Note: centering (pre-transform) must be applied to A, B
                    // before calling TiledEngine. This is the raw accumulation.
                    "covariance" => {
                        let denom = if kd > 1 { (kd - 1) as f64 } else { 1.0 };
                        for i in 0..m {
                            for j in 0..n {
                                let mut acc = 0.0_f64;
                                for ki in 0..kd {
                                    acc += a[i * kd + ki] * b[ki * n + j];
                                }
                                c[i * n + j] = acc / denom;
                            }
                        }
                    }

                    // SoftmaxWeightedOp: online softmax — FlashAttention pattern
                    // C[i,j] = sum_k softmax(A[i,:])[k] * B[k,j]
                    // One-pass numerically stable: carry (max, exp_sum, weighted_sum)
                    "softmax_weighted" => {
                        for i in 0..m {
                            for j in 0..n {
                                let mut max_val = f64::NEG_INFINITY;
                                let mut exp_sum = 0.0_f64;
                                let mut weighted_sum = 0.0_f64;
                                for ki in 0..kd {
                                    let score = a[i * kd + ki];
                                    if score > max_val {
                                        let scale = (max_val - score).exp();
                                        exp_sum *= scale;
                                        weighted_sum *= scale;
                                        max_val = score;
                                    }
                                    let w = (score - max_val).exp();
                                    exp_sum += w;
                                    weighted_sum += w * b[ki * n + j];
                                }
                                c[i * n + j] = if exp_sum > 0.0 { weighted_sum / exp_sum } else { 0.0 };
                            }
                        }
                    }

                    // ManifoldDistanceOp: geometry-parameterized distance
                    // params_key distinguishes euclidean / sphere / poincare
                    "manifold_distance" => {
                        let params = k.params_key.as_str();
                        if let Some(kappa) = poincare_kappa_from_params(params) {
                            // Poincaré ball: 3-field per-dimension accumulation
                            // d(x,y) = (2/√κ) * arccosh(1 + 2κ*||x-y||²/((1-κ||x||²)(1-κ||y||²)))
                            for i in 0..m {
                                for j in 0..n {
                                    let mut sq_dist   = 0.0_f64;
                                    let mut sq_norm_x = 0.0_f64;
                                    let mut sq_norm_y = 0.0_f64;
                                    for ki in 0..kd {
                                        let ax = a[i * kd + ki];
                                        let by = b[ki * n + j];
                                        let diff = ax - by;
                                        sq_dist   += diff * diff;
                                        sq_norm_x += ax * ax;
                                        sq_norm_y += by * by;
                                    }
                                    let denom = ((1.0 - kappa * sq_norm_x) * (1.0 - kappa * sq_norm_y)).max(1e-15);
                                    let arg   = (1.0 + 2.0 * kappa * sq_dist / denom).max(1.0);
                                    c[i * n + j] = (2.0 / kappa.sqrt()) * arg.acosh();
                                }
                            }
                        } else if params.starts_with("sphere_geodesic") {
                            // SphericalGeodesic: arccos(dot / sqrt(sq_norm_x · sq_norm_y))
                            // Must check sphere_geodesic before sphere (prefix match).
                            for i in 0..m {
                                for j in 0..n {
                                    let mut sq_norm_x = 0.0_f64;
                                    let mut sq_norm_y = 0.0_f64;
                                    let mut dot_prod  = 0.0_f64;
                                    for ki in 0..kd {
                                        let ax = a[i * kd + ki];
                                        let by = b[ki * n + j];
                                        sq_norm_x += ax * ax;
                                        sq_norm_y += by * by;
                                        dot_prod  += ax * by;
                                    }
                                    let denom = (sq_norm_x * sq_norm_y).max(1e-60).sqrt();
                                    let cos_theta = (dot_prod / denom).clamp(-1.0, 1.0);
                                    c[i * n + j] = cos_theta.acos();
                                }
                            }
                        } else if params.starts_with("sphere") {
                            // Sphere: dot product → cosine distance = 1 - dot(a, b)
                            // Input vectors must be unit-normalized by the caller.
                            for i in 0..m {
                                for j in 0..n {
                                    let mut acc = 0.0_f64;
                                    for ki in 0..kd {
                                        acc += a[i * kd + ki] * b[ki * n + j];
                                    }
                                    c[i * n + j] = 1.0 - acc;
                                }
                            }
                        } else {
                            // Euclidean L2Sq (default, includes params = "euclidean")
                            for i in 0..m {
                                for j in 0..n {
                                    let mut acc = 0.0_f64;
                                    for ki in 0..kd {
                                        let diff = a[i * kd + ki] - b[ki * n + j];
                                        acc += diff * diff;
                                    }
                                    c[i * n + j] = acc;
                                }
                            }
                        }
                    }

                    // ManifoldMixtureOp: composite 3-field kernel for all manifolds at once.
                    // C has shape M×N×n_manifolds; params_key is pipe-separated manifold names.
                    "manifold_mixture" => {
                        let specs = parse_manifold_mixture_specs(&k.params_key);
                        let nm = specs.len();
                        if nm == 0 {
                            return Err(TamGpuError::InvalidArgument("manifold_mixture: empty params_key".into()));
                        }
                        for i in 0..m {
                            for j in 0..n {
                                let mut sq_norm_x = 0.0_f64;
                                let mut sq_norm_y = 0.0_f64;
                                let mut dot_prod  = 0.0_f64;
                                for ki in 0..kd {
                                    let ax = a[i * kd + ki];
                                    let by = b[ki * n + j];
                                    sq_norm_x += ax * ax;
                                    sq_norm_y += by * by;
                                    dot_prod  += ax * by;
                                }
                                let sq_dist = sq_norm_x + sq_norm_y - 2.0 * dot_prod;
                                let base = (i * n + j) * nm;
                                for (mk, spec) in specs.iter().enumerate() {
                                    c[base + mk] = match spec {
                                        CpuMixtureManifold::Euclidean =>
                                            sq_dist,
                                        CpuMixtureManifold::Poincare { kappa } => {
                                            let denom = ((1.0 - kappa * sq_norm_x) * (1.0 - kappa * sq_norm_y)).max(1e-15);
                                            let arg = (1.0 + 2.0 * kappa * sq_dist / denom).max(1.0);
                                            (2.0 / kappa.sqrt()) * arg.acosh()
                                        }
                                        CpuMixtureManifold::Sphere =>
                                            1.0 - dot_prod,
                                        CpuMixtureManifold::SphereGeodesic => {
                                            let denom = (sq_norm_x * sq_norm_y).max(1e-60).sqrt();
                                            (dot_prod / denom).clamp(-1.0, 1.0).acos()
                                        }
                                    };
                                }
                            }
                        }
                    }

                    other => return Err(TamGpuError::EntryNotFound(
                        format!("CpuBackend tiled_accumulate: unknown op '{}'. \
                                 Supported: dot_product, outer_product, l2_distance, \
                                 covariance, softmax_weighted, manifold_distance, \
                                 manifold_mixture", other)
                    )),
                }
            }

            other => return Err(TamGpuError::EntryNotFound(other.to_string())),
        }

        Ok(())
    }

    fn sync(&self) -> TamResult<()> {
        // CPU is synchronous — nothing to wait for.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Mixture manifold specification (CPU-local, no dependency on tambear types).
enum CpuMixtureManifold {
    Euclidean,
    Poincare { kappa: f64 },
    Sphere,
    SphereGeodesic,
}

/// Parse mixture params_key (e.g. `"euclidean|poincare(c=-1.0000)|sphere_geodesic(r=1.0000)"`)
/// into a list of `CpuMixtureManifold` variants.
fn parse_manifold_mixture_specs(params_key: &str) -> Vec<CpuMixtureManifold> {
    params_key.split('|').filter_map(|s| {
        let s = s.trim();
        if s.is_empty() {
            None
        } else if s.starts_with("poincare") {
            let kappa = poincare_kappa_from_params(s).unwrap_or(1.0);
            Some(CpuMixtureManifold::Poincare { kappa })
        } else if s.starts_with("sphere_geodesic") {
            // Must check sphere_geodesic before sphere (prefix match)
            Some(CpuMixtureManifold::SphereGeodesic)
        } else if s.starts_with("sphere") {
            Some(CpuMixtureManifold::Sphere)
        } else {
            Some(CpuMixtureManifold::Euclidean)
        }
    }).collect()
}

/// Parse the Poincaré curvature magnitude κ = |c| from a params_key string.
///
/// Expects format `"poincare(c=-1.0000)"` (emitted by `Manifold::name()`).
/// Returns `Some(kappa)` where kappa > 0 if parsing succeeds, `None` otherwise.
fn poincare_kappa_from_params(params: &str) -> Option<f64> {
    let s = params.strip_prefix("poincare(c=")?;
    let s = s.strip_suffix(')')?;
    s.parse::<f64>().ok().map(f64::abs)
}

fn cpu_buf(buf: &Buffer) -> TamResult<&CpuBuffer> {
    buf.inner.downcast_ref::<CpuBuffer>()
        .ok_or_else(|| TamGpuError::Dispatch("expected a CPU buffer".into()))
}

fn read_as<'a, T: bytemuck::Pod>(guard: &'a MutexGuard<'_, Vec<u8>>) -> &'a [T] {
    cast_slice(guard.as_slice())
}

fn write_as<'a, T: bytemuck::Pod>(guard: &'a mut MutexGuard<'_, Vec<u8>>) -> &'a mut [T] {
    cast_slice_mut(guard.as_mut_slice())
}

fn argmin_cpu(guard: &MutexGuard<Vec<u8>>) -> (f64, usize) {
    let vals: &[f64] = cast_slice(guard.as_slice());
    vals.iter().enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .fold((f64::INFINITY, usize::MAX), |(bv, bi), (i, &v)| {
            if v < bv || (v == bv && i < bi) { (v, i) } else { (bv, bi) }
        })
}

fn argmax_cpu(guard: &MutexGuard<Vec<u8>>) -> (f64, usize) {
    let vals: &[f64] = cast_slice(guard.as_slice());
    vals.iter().enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .fold((f64::NEG_INFINITY, usize::MAX), |(bv, bi), (i, &v)| {
            if v > bv || (v == bv && i < bi) { (v, i) } else { (bv, bi) }
        })
}

fn write_scalar_f64(arc: &Arc<Mutex<Vec<u8>>>, val: f64) -> TamResult<()> {
    let mut g = arc.lock().unwrap();
    let out: &mut [f64] = cast_slice_mut(g.as_mut_slice());
    if out.is_empty() {
        return Err(TamGpuError::InvalidArgument("output buffer too small for f64 scalar".into()));
    }
    out[0] = val;
    Ok(())
}

fn write_scalar_i32(arc: &Arc<Mutex<Vec<u8>>>, val: i32) -> TamResult<()> {
    let mut g = arc.lock().unwrap();
    let out: &mut [i32] = cast_slice_mut(g.as_mut_slice());
    if out.is_empty() {
        return Err(TamGpuError::InvalidArgument("output buffer too small for i32 scalar".into()));
    }
    out[0] = val;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{upload, download, TamGpu};

    fn gpu() -> CpuBackend { CpuBackend::new() }

    #[test]
    fn cpu_name_and_backend() {
        let g = gpu();
        assert_eq!(g.backend(), Backend::Cpu);
        assert!(g.name().contains("CPU"));
    }

    #[test]
    fn cpu_alloc_copy_roundtrip() {
        let g = gpu();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let buf = upload(&g, &data).unwrap();
        assert_eq!(buf.size, 24);  // 3 * 8 bytes
        let out: Vec<f64> = download(&g, &buf, 3).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn cpu_scatter_sum() {
        let g = gpu();
        let keys: Vec<i32> = vec![0, 1, 0, 2, 1];
        let vals: Vec<f64> = vec![1.0, 10.0, 2.0, 100.0, 20.0];
        let k = g.compile("", "scatter_sum").unwrap();
        let b_keys  = upload(&g, &keys).unwrap();
        let b_vals  = upload(&g, &vals).unwrap();
        let b_out   = g.alloc(3 * 8).unwrap();  // 3 groups × f64
        g.dispatch(&k, [1,1,1], [1,1,1], &[&b_keys, &b_vals, &b_out], 0).unwrap();
        let out: Vec<f64> = download(&g, &b_out, 3).unwrap();
        // group 0: 1+2=3, group 1: 10+20=30, group 2: 100
        assert_eq!(out, vec![3.0, 30.0, 100.0]);
    }

    #[test]
    fn cpu_scatter_count() {
        let g = gpu();
        let keys: Vec<i32> = vec![0, 0, 1, 2, 2, 2];
        let k = g.compile("", "scatter_count").unwrap();
        let b_keys = upload(&g, &keys).unwrap();
        let b_out  = g.alloc(3 * 8).unwrap();
        g.dispatch(&k, [1,1,1], [1,1,1], &[&b_keys, &b_out], 0).unwrap();
        let out: Vec<f64> = download(&g, &b_out, 3).unwrap();
        assert_eq!(out, vec![2.0, 1.0, 3.0]);
    }

    #[test]
    fn cpu_gather_and_scatter_back() {
        let g = gpu();
        // 6 values: groups [A=0, B=1, B=1, A=0, C=2, A=0]
        // rows_by_group = [0,3,5, 1,2, 4]
        let values: Vec<f64>  = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let rbg: Vec<u32>     = vec![0, 3, 5, 1, 2, 4];

        let kg = g.compile("", "gather_f64").unwrap();
        let ks = g.compile("", "scatter_back_f64").unwrap();

        let b_vals  = upload(&g, &values).unwrap();
        let b_rbg   = upload(&g, &rbg).unwrap();
        let b_gath  = g.alloc(6 * 8).unwrap();
        g.dispatch(&kg, [1,1,1], [1,1,1], &[&b_vals, &b_rbg, &b_gath], 0).unwrap();
        let gathered: Vec<f64> = download(&g, &b_gath, 6).unwrap();
        // gathered = [values[0], values[3], values[5], values[1], values[2], values[4]]
        assert_eq!(gathered, vec![10.0, 40.0, 60.0, 20.0, 30.0, 50.0]);

        let b_out = g.alloc(6 * 8).unwrap();
        g.dispatch(&ks, [1,1,1], [1,1,1], &[&b_gath, &b_rbg, &b_out], 0).unwrap();
        let restored: Vec<f64> = download(&g, &b_out, 6).unwrap();
        assert_eq!(restored, values, "scatter_back should restore original order");
    }

    #[test]
    fn cpu_argmin() {
        let g = gpu();
        let vals: Vec<f64> = vec![5.0, 3.0, 8.0, 1.0, 6.0];
        let k = g.compile("", "argmin_f64").unwrap();
        let b_vals    = upload(&g, &vals).unwrap();
        let b_out_val = g.alloc(8).unwrap();
        let b_out_idx = g.alloc(4).unwrap();
        g.dispatch(&k, [1,1,1], [1,1,1], &[&b_vals, &b_out_val, &b_out_idx], 0).unwrap();
        let out_val: Vec<f64> = download(&g, &b_out_val, 1).unwrap();
        let out_idx: Vec<i32> = download(&g, &b_out_idx, 1).unwrap();
        assert_eq!(out_val[0], 1.0);
        assert_eq!(out_idx[0], 3);
    }

    #[test]
    fn cpu_argmax() {
        let g = gpu();
        let vals: Vec<f64> = vec![5.0, 3.0, 8.0, 1.0, 6.0];
        let k = g.compile("", "argmax_f64").unwrap();
        let b_vals    = upload(&g, &vals).unwrap();
        let b_out_val = g.alloc(8).unwrap();
        let b_out_idx = g.alloc(4).unwrap();
        g.dispatch(&k, [1,1,1], [1,1,1], &[&b_vals, &b_out_val, &b_out_idx], 0).unwrap();
        let out_val: Vec<f64> = download(&g, &b_out_val, 1).unwrap();
        let out_idx: Vec<i32> = download(&g, &b_out_idx, 1).unwrap();
        assert_eq!(out_val[0], 8.0);
        assert_eq!(out_idx[0], 2);
    }

    #[test]
    fn cpu_argmin_nan_excluded() {
        let g = gpu();
        let vals: Vec<f64> = vec![f64::NAN, 3.0, f64::NAN, 7.0];
        let k = g.compile("", "argmin_f64").unwrap();
        let b_vals    = upload(&g, &vals).unwrap();
        let b_out_val = g.alloc(8).unwrap();
        let b_out_idx = g.alloc(4).unwrap();
        g.dispatch(&k, [1,1,1], [1,1,1], &[&b_vals, &b_out_val, &b_out_idx], 0).unwrap();
        let out_val: Vec<f64> = download(&g, &b_out_val, 1).unwrap();
        let out_idx: Vec<i32> = download(&g, &b_out_idx, 1).unwrap();
        assert_eq!(out_val[0], 3.0);
        assert_eq!(out_idx[0], 1);
    }

    #[test]
    fn cpu_compile_unknown_entry_errors() {
        let g = gpu();
        let err = g.compile("", "matrix_multiply").unwrap_err();
        assert!(matches!(err, TamGpuError::EntryNotFound(_)));
    }

    #[test]
    fn cpu_noop() {
        let g = gpu();
        let k = g.compile("", "noop").unwrap();
        g.dispatch(&k, [1,1,1], [1,1,1], &[], 0).unwrap();
    }

    #[test]
    fn cpu_sync_is_noop() {
        let g = gpu();
        g.sync().unwrap();
    }

    // ---------------------------------------------------------------
    // tiled_accumulate tests
    // ---------------------------------------------------------------

    /// Helper: dispatch tiled_accumulate with given op_name comment.
    fn tiled_run(op_name: &str, a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
        let g = gpu();
        let source = format!("// Tiled accumulation kernel for operator: {op_name}");
        let kernel = g.compile(&source, "tiled_accumulate").unwrap();

        let a_buf   = upload(&g, a).unwrap();
        let b_buf   = upload(&g, b).unwrap();
        let c_buf   = g.alloc(m * n * 8).unwrap();
        let dim_buf = upload(&g, &[m as i32, n as i32, k as i32]).unwrap();

        g.dispatch(&kernel, [1,1,1], [1,1,1], &[&a_buf, &b_buf, &c_buf, &dim_buf], 0).unwrap();
        download(&g, &c_buf, m * n).unwrap()
    }

    #[test]
    fn cpu_tiled_dot_product_2x3_times_3x2() {
        // A(2×3) = [[1,2,3],[4,5,6]]
        // B(3×2) = [[7,8],[9,10],[11,12]]
        // C = A*B = [[58,64],[139,154]]
        let a = vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0,  9.0, 10.0,  11.0, 12.0];
        let c = tiled_run("dot_product", &a, &b, 2, 2, 3);
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn cpu_tiled_dot_product_1x1() {
        // Scalar: [3] * [4] = [12]
        let c = tiled_run("dot_product", &[3.0], &[4.0], 1, 1, 1);
        assert_eq!(c, vec![12.0]);
    }

    #[test]
    fn cpu_tiled_dot_product_identity() {
        // A(2×2) * I(2×2) = A
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let i = vec![1.0, 0.0, 0.0, 1.0];
        let c = tiled_run("dot_product", &a, &i, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cpu_tiled_l2_distance() {
        // A(2×2) = [[1,0],[0,1]], B(2×2) = [[1,0],[0,1]] (B = A^T for self-distance)
        // dist[0,0] = (1-1)^2 + (0-0)^2 = 0
        // dist[0,1] = (1-0)^2 + (0-1)^2 = 2
        // dist[1,0] = (0-1)^2 + (1-0)^2 = 2
        // dist[1,1] = (0-0)^2 + (1-1)^2 = 0
        let a = vec![1.0, 0.0,  0.0, 1.0];
        let b = vec![1.0, 0.0,  0.0, 1.0]; // K×N = 2×2 (same as A here)
        let c = tiled_run("l2_distance", &a, &b, 2, 2, 2);
        assert_eq!(c, vec![0.0, 2.0, 2.0, 0.0]);
    }

    #[test]
    fn cpu_tiled_covariance() {
        // Pre-centered A(2×3) = [[-1,0,1],[1,0,-1]], B(3×2) = A^T
        // cov = A*A^T / (3-1) = [[2, -2],[-2, 2]] / 2 = [[1,-1],[-1,1]]
        let a = vec![-1.0, 0.0, 1.0,  1.0, 0.0, -1.0];
        let b = vec![-1.0, 1.0,  0.0, 0.0,  1.0, -1.0]; // K×N = 3×2
        let c = tiled_run("covariance", &a, &b, 2, 2, 3);
        assert_eq!(c, vec![1.0, -1.0, -1.0, 1.0]);
    }

    #[test]
    fn cpu_tiled_outer_product() {
        // Same as dot_product mathematically
        let a = vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0,  9.0, 10.0,  11.0, 12.0];
        let c = tiled_run("outer_product", &a, &b, 2, 2, 3);
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn cpu_tiled_compile_wgsl_source() {
        // WGSL source has different comment prefix — should still parse op name
        let g = gpu();
        let source = "// WGSL tiled accumulation kernel for operator: l2_distance\n@compute...";
        let kernel = g.compile(source, "tiled_accumulate").unwrap();
        assert_eq!(kernel.entry, "tiled_accumulate");
    }

    #[test]
    fn cpu_tiled_unknown_op_errors() {
        let g = gpu();
        let source = "// Tiled accumulation kernel for operator: unknown_op";
        let kernel = g.compile(source, "tiled_accumulate").unwrap();
        let a_buf = upload(&g, &[1.0_f64]).unwrap();
        let b_buf = upload(&g, &[1.0_f64]).unwrap();
        let c_buf = g.alloc(8).unwrap();
        let d_buf = upload(&g, &[1_i32, 1, 1]).unwrap();
        let err = g.dispatch(&kernel, [1,1,1], [1,1,1], &[&a_buf, &b_buf, &c_buf, &d_buf], 0);
        assert!(err.is_err());
    }

    #[test]
    fn cpu_tiled_softmax_weighted_uniform_scores() {
        // A has uniform scores → softmax is uniform → output = mean of B column
        // A: 1×3, all zeros → softmax([0,0,0]) = [1/3, 1/3, 1/3]
        // B: 3×1, values [3, 6, 9] → weighted sum = (3+6+9)/3 = 6
        let c = tiled_run("softmax_weighted", &[0.0, 0.0, 0.0], &[3.0, 6.0, 9.0], 1, 1, 3);
        assert!((c[0] - 6.0).abs() < 1e-10, "expected 6.0, got {}", c[0]);
    }

    #[test]
    fn cpu_tiled_softmax_weighted_dominant_score() {
        // One score much larger → softmax concentrates on that row's B value
        // A: 1×3, scores [0, 100, 0] → softmax ≈ [0, 1, 0]
        // B: 3×1, values [10, 20, 30] → output ≈ 20
        let c = tiled_run("softmax_weighted", &[0.0, 100.0, 0.0], &[10.0, 20.0, 30.0], 1, 1, 3);
        assert!((c[0] - 20.0).abs() < 1e-6, "expected ~20.0, got {}", c[0]);
    }

    #[test]
    fn cpu_tiled_softmax_weighted_numerical_stability() {
        // Large scores that would overflow naive exp() — online softmax handles this
        // A: 1×2, scores [1000, 1001] → softmax = [e^(-1)/(1+e^(-1)), 1/(1+e^(-1))]
        // B: 2×1, values [0, 1] → output = softmax[1] ≈ 1/(1+e^(-1)) ≈ 0.7311
        let c = tiled_run("softmax_weighted", &[1000.0, 1001.0], &[0.0, 1.0], 1, 1, 2);
        let expected = 1.0 / (1.0 + (-1.0_f64).exp()); // sigmoid(1)
        assert!((c[0] - expected).abs() < 1e-10,
            "expected {expected}, got {} (numerical stability)", c[0]);
    }

    #[test]
    fn cpu_tiled_softmax_weighted_2x2() {
        // A: 2×2 (scores), B: 2×2 (values)
        // Row 0 of A: [0, 0] → uniform softmax [0.5, 0.5]
        // Row 1 of A: [0, 100] → concentrated softmax [~0, ~1]
        // B = [[1, 2], [3, 4]]
        // C[0,0] = 0.5*1 + 0.5*3 = 2.0
        // C[0,1] = 0.5*2 + 0.5*4 = 3.0
        // C[1,0] ≈ 0*1 + 1*3 = 3.0
        // C[1,1] ≈ 0*2 + 1*4 = 4.0
        let a = [0.0, 0.0, 0.0, 100.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let c = tiled_run("softmax_weighted", &a, &b, 2, 2, 2);
        assert!((c[0] - 2.0).abs() < 1e-10, "C[0,0]={}", c[0]);
        assert!((c[1] - 3.0).abs() < 1e-10, "C[0,1]={}", c[1]);
        assert!((c[2] - 3.0).abs() < 1e-6, "C[1,0]={}", c[2]);
        assert!((c[3] - 4.0).abs() < 1e-6, "C[1,1]={}", c[3]);
    }
}
