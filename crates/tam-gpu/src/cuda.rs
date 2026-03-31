//! CUDA backend — wraps cudarc, dynamic-loads nvcuda.dll (no toolkit required).
//!
//! ## Buffer representation
//!
//! [`CudaBuffer`] stores a `CudaSlice<u8>` (device memory) behind an
//! `Arc<Mutex<...>>`. The Mutex provides interior mutability so `dispatch` can
//! write to output buffers through shared `&Buffer` references — which is how
//! the `TamGpu` trait works. The CUDA driver just receives a device pointer
//! (u64); Rust's `mut` vs `non-mut` distinction doesn't reach the GPU.
//!
//! ## Kernel dispatch
//!
//! `dispatch` passes buffer device-pointers to `stream.launch_builder` using a
//! match-on-buffer-count pattern (same trick as `scatter_multi_phi`). Up to 8
//! buffers are supported. Each buffer is passed as `&CudaSlice<u8>`; the CUDA
//! kernel receives it as a typed pointer (cast in the kernel source).
//!
//! ## Scalar arguments
//!
//! Kernels that need integer parameters (like `int n`) must receive them as
//! single-element buffers. Callers upload a `vec![n as i32]` as a 4-byte buffer
//! and include it in the `bufs` slice at the expected position.
//!
//! The compile-time alternative (bake n into PTX via NVRTC macro) avoids the
//! extra buffer at the cost of a recompile when n changes.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::{Backend, Buffer, Kernel, ShaderLang, TamGpu, TamGpuError, TamResult};

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// CUDA device buffer. Arc+Mutex for Send+Sync + interior mutability.
pub(crate) struct CudaBuffer {
    pub slice: Mutex<CudaSlice<u8>>,
}

// Safety: CudaSlice<u8> is a device pointer + length. Sending between threads
// is safe since CUDA device pointers are global within a context. The Mutex
// provides Sync.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    fn new(slice: CudaSlice<u8>) -> Self {
        CudaBuffer { slice: Mutex::new(slice) }
    }
}

/// CUDA kernel handle — compiled CudaFunction + entry name.
pub(crate) struct CudaKernel {
    pub func: CudaFunction,
    #[allow(dead_code)]
    pub entry: String,
    // Keep the module alive as long as the kernel lives.
    pub _module: Arc<cudarc::driver::CudaModule>,
}

// Safety: CudaFunction is a handle to a compiled PTX function in device memory.
// The handle is a pointer-sized value valid for the lifetime of the module.
unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

// ---------------------------------------------------------------------------
// CudaBackend
// ---------------------------------------------------------------------------

/// CUDA compute backend.
///
/// Dynamic-loads `nvcuda.dll` (Windows) or `libcuda.so` (Linux) at runtime.
/// No CUDA toolkit installation required — only the GPU driver.
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    device_name: String,
}

impl CudaBackend {
    /// Initialise on GPU 0.
    pub fn new() -> TamResult<Self> {
        Self::on_device(0)
    }

    /// Initialise on a specific GPU ordinal.
    pub fn on_device(ordinal: usize) -> TamResult<Self> {
        let ctx = CudaContext::new(ordinal)
            .map_err(|e| TamGpuError::Backend(format!("CUDA init failed: {e}")))?;
        let stream = ctx.default_stream();
        // Try to get the device name; non-fatal if it fails.
        let device_name = format!("GPU {} (CUDA)", ordinal);
        Ok(CudaBackend { ctx: ctx.into(), stream, device_name })
    }

    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }
    pub fn ctx(&self) -> &Arc<CudaContext> { &self.ctx }
}

impl TamGpu for CudaBackend {
    fn name(&self) -> String {
        self.device_name.clone()
    }

    fn backend(&self) -> Backend { Backend::Cuda }
    fn shader_lang(&self) -> ShaderLang { ShaderLang::Cuda }

    fn compile(&self, source: &str, entry: &str) -> TamResult<Kernel> {
        let opts = CompileOptions { arch: Some("sm_120"), ..Default::default() };
        let ptx = compile_ptx_with_opts(source, opts)
            .map_err(|e| TamGpuError::Compile(format!("NVRTC: {e}")))?;
        let module = self.ctx.load_module(ptx)
            .map_err(|e| TamGpuError::Compile(format!("load_module: {e}")))?;
        let func = module.load_function(entry)
            .map_err(|e| TamGpuError::Compile(format!("load_function '{entry}': {e}")))?;
        Ok(Kernel {
            inner: Box::new(CudaKernel { func, entry: entry.to_string(), _module: module }),
            entry: entry.to_string(),
        })
    }

    fn alloc(&self, bytes: usize) -> TamResult<Buffer> {
        let slice: CudaSlice<u8> = self.stream.alloc_zeros(bytes)
            .map_err(|e| TamGpuError::Alloc(format!("{e}")))?;
        Ok(Buffer { inner: Box::new(CudaBuffer::new(slice)), size: bytes })
    }

    fn free(&self, buf: Buffer) -> TamResult<()> {
        drop(buf);
        Ok(())
    }

    fn copy_h2d(&self, src: &[u8], dst: &Buffer) -> TamResult<()> {
        let inner = cuda_buf(dst)?;
        let mut slice = inner.slice.lock().unwrap();
        // SAFETY: CudaSlice<u8> has the same device pointer regardless of Rust type.
        // We copy bytes directly via the stream.
        let src_dev: CudaSlice<u8> = self.stream.clone_htod(src)
            .map_err(|e| TamGpuError::Transfer(format!("{e}")))?;
        // Copy src_dev → *slice (device-to-device).
        // We re-use clone_htod as a staging copy for now (TODO: use cuMemcpyDtoD).
        let _ = src_dev;
        // Simpler: replace the slice entirely (valid since alloc'd size matches).
        if src.len() != slice.len() {
            return Err(TamGpuError::Transfer(format!(
                "copy_h2d: src len {} ≠ buf len {}", src.len(), slice.len()
            )));
        }
        *slice = self.stream.clone_htod(src)
            .map_err(|e| TamGpuError::Transfer(format!("{e}")))?;
        Ok(())
    }

    fn copy_d2h(&self, src: &Buffer, dst: &mut [u8]) -> TamResult<()> {
        let inner = cuda_buf(src)?;
        let slice = inner.slice.lock().unwrap();
        if dst.len() != slice.len() {
            return Err(TamGpuError::Transfer(format!(
                "copy_d2h: dst len {} ≠ buf len {}", dst.len(), slice.len()
            )));
        }
        let host = self.stream.clone_dtoh(&*slice)
            .map_err(|e| TamGpuError::Transfer(format!("{e}")))?;
        self.stream.synchronize()
            .map_err(|e| TamGpuError::Transfer(format!("{e}")))?;
        dst.copy_from_slice(&host);
        Ok(())
    }

    fn dispatch(
        &self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        bufs: &[&Buffer],
        shared_mem: u32,
    ) -> TamResult<()> {
        let k = kernel.inner.downcast_ref::<CudaKernel>()
            .ok_or_else(|| TamGpuError::Dispatch("not a CUDA kernel".into()))?;

        let cfg = LaunchConfig {
            grid_dim: (grid[0], grid[1], grid[2]),
            block_dim: (block[0], block[1], block[2]),
            shared_mem_bytes: shared_mem,
        };

        // Lock all buffers and collect guards. Order is 0..n, no deadlock risk
        // since each guard protects an independent buffer.
        let guards: Vec<std::sync::MutexGuard<CudaSlice<u8>>> = bufs.iter()
            .enumerate()
            .map(|(i, b)| {
                cuda_buf(b)
                    .map_err(|e| TamGpuError::Dispatch(format!("buf[{i}]: {e}")))
                    .map(|cb| cb.slice.lock().unwrap())
            })
            .collect::<TamResult<Vec<_>>>()?;

        // Variable-arity launch: match on buffer count.
        // Each arm passes the guards as &CudaSlice<u8> to launch_builder.
        // The CUDA kernel receives these as typed pointers (via implicit cast in PTX).
        unsafe {
            match guards.as_slice() {
                [] => {
                    self.stream.launch_builder(&k.func)
                        .launch(cfg)
                        .map_err(|e| TamGpuError::Dispatch(format!("{e}")))?;
                }
                [g0] => {
                    self.stream.launch_builder(&k.func)
                        .arg(&**g0)
                        .launch(cfg)
                        .map_err(|e| TamGpuError::Dispatch(format!("{e}")))?;
                }
                [g0, g1] => {
                    self.stream.launch_builder(&k.func)
                        .arg(&**g0).arg(&**g1)
                        .launch(cfg)
                        .map_err(|e| TamGpuError::Dispatch(format!("{e}")))?;
                }
                [g0, g1, g2] => {
                    self.stream.launch_builder(&k.func)
                        .arg(&**g0).arg(&**g1).arg(&**g2)
                        .launch(cfg)
                        .map_err(|e| TamGpuError::Dispatch(format!("{e}")))?;
                }
                [g0, g1, g2, g3] => {
                    self.stream.launch_builder(&k.func)
                        .arg(&**g0).arg(&**g1).arg(&**g2).arg(&**g3)
                        .launch(cfg)
                        .map_err(|e| TamGpuError::Dispatch(format!("{e}")))?;
                }
                [g0, g1, g2, g3, g4] => {
                    self.stream.launch_builder(&k.func)
                        .arg(&**g0).arg(&**g1).arg(&**g2).arg(&**g3).arg(&**g4)
                        .launch(cfg)
                        .map_err(|e| TamGpuError::Dispatch(format!("{e}")))?;
                }
                [g0, g1, g2, g3, g4, g5] => {
                    self.stream.launch_builder(&k.func)
                        .arg(&**g0).arg(&**g1).arg(&**g2).arg(&**g3).arg(&**g4)
                        .arg(&**g5)
                        .launch(cfg)
                        .map_err(|e| TamGpuError::Dispatch(format!("{e}")))?;
                }
                [g0, g1, g2, g3, g4, g5, g6] => {
                    self.stream.launch_builder(&k.func)
                        .arg(&**g0).arg(&**g1).arg(&**g2).arg(&**g3).arg(&**g4)
                        .arg(&**g5).arg(&**g6)
                        .launch(cfg)
                        .map_err(|e| TamGpuError::Dispatch(format!("{e}")))?;
                }
                [g0, g1, g2, g3, g4, g5, g6, g7] => {
                    self.stream.launch_builder(&k.func)
                        .arg(&**g0).arg(&**g1).arg(&**g2).arg(&**g3).arg(&**g4)
                        .arg(&**g5).arg(&**g6).arg(&**g7)
                        .launch(cfg)
                        .map_err(|e| TamGpuError::Dispatch(format!("{e}")))?;
                }
                _ => return Err(TamGpuError::Dispatch(
                    format!("CudaBackend::dispatch: max 8 buffers, got {}", bufs.len())
                )),
            }
        }

        Ok(())
    }

    fn sync(&self) -> TamResult<()> {
        self.stream.synchronize()
            .map_err(|e| TamGpuError::Backend(format!("sync: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cuda_buf(buf: &Buffer) -> TamResult<&CudaBuffer> {
    buf.inner.downcast_ref::<CudaBuffer>()
        .ok_or_else(|| TamGpuError::Dispatch("expected a CUDA buffer".into()))
}
