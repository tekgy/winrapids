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
    /// `source` is ignored — no shader compilation on CPU.
    fn compile(&self, _source: &str, entry: &str) -> TamResult<Kernel> {
        match entry {
            "scatter_sum"      |
            "scatter_count"    |
            "gather_f64"       |
            "scatter_back_f64" |
            "argmin_f64"       |
            "argmax_f64"       |
            "noop"             => {}
            other => return Err(TamGpuError::EntryNotFound(format!(
                "CpuBackend has no implementation for '{}'. \
                 Supported: scatter_sum, scatter_count, gather_f64, \
                 scatter_back_f64, argmin_f64, argmax_f64, noop", other
            ))),
        }
        Ok(Kernel {
            inner: Box::new(CpuKernel { entry: entry.to_string() }),
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
}
