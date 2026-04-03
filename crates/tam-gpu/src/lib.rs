//! # Tambear GPU Runtime
//!
//! Unified compute backend: same algorithm, any hardware.
//!
//! ```text
//! "Tam doesn't need CUDA. Tam doesn't need Vulkan.
//!  Tam doesn't need Metal. Tam needs cores. Any cores."
//! ```
//!
//! ## Architecture
//!
//! The [`TamGpu`] trait defines a minimal compute API:
//! allocate, transfer, compile, dispatch, sync. Backend implementations
//! translate this contract to the underlying driver.
//!
//! | Backend       | Driver         | Shader lang | Crate/feature |
//! |---------------|----------------|-------------|---------------|
//! | [`CudaBackend`] | nvcuda.dll    | CUDA C      | `cuda` (default) |
//! | [`CpuBackend`]  | —             | native Rust | always available |
//! | VulkanBackend | vulkan-1.dll    | SPIR-V      | `vulkan` (planned) |
//! | MetalBackend  | native macOS   | MSL         | `metal` (planned)  |
//!
//! ## Auto-detection
//!
//! ```rust,no_run
//! use tam_gpu::detect;
//! let gpu = detect();
//! println!("{}", gpu.name());  // "NVIDIA RTX PRO 6000 (CUDA)" or "CPU (native Rust)"
//! ```
//!
//! ## Buffer convention
//!
//! CPU kernels infer `n` (element count) from `bufs[0].size / elem_size`.
//! CUDA kernels embed `n` in the compiled PTX via NVRTC macros or pass it as a
//! 4-byte scalar buffer. Either convention works; callers must match the kernel
//! source they compile.

mod error;
pub mod cpu;
pub mod phi_eval;
#[cfg(feature = "cuda")]
pub mod cuda;

pub use error::TamGpuError;
pub use cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

pub type TamResult<T> = std::result::Result<T, TamGpuError>;

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffer({}B)", self.size)
    }
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kernel(\"{}\")", self.entry)
    }
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Which compute driver the backend uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend { Cuda, Vulkan, Metal, Dx12, Cpu }

/// Which shader language this backend compiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderLang {
    /// CUDA C (NVRTC → PTX).
    Cuda,
    /// WGSL source (wgpu → SPIR-V/MSL/DX12 at runtime).
    Wgsl,
    /// SPIR-V binary (raw Vulkan compute pipeline).
    SpirV,
    /// Metal Shading Language (macOS/iOS).
    Msl,
    /// HLSL (DirectX 12 compute).
    Hlsl,
    /// Native Rust — no shader compilation.
    Cpu,
}

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

/// Opaque GPU-resident (or host-pinned) buffer.
///
/// Created by [`TamGpu::alloc`], freed by [`TamGpu::free`] or by dropping.
/// The internal representation is backend-specific and not accessible to callers.
pub struct Buffer {
    pub(crate) inner: Box<dyn std::any::Any + Send + Sync>,
    /// Size in bytes.
    pub size: usize,
}

impl Buffer {
    /// Construct a buffer wrapping a backend-specific inner value.
    ///
    /// Backend implementors outside this crate use this to create `Buffer` handles.
    pub fn new(inner: Box<dyn std::any::Any + Send + Sync>, size: usize) -> Self {
        Self { inner, size }
    }

    /// Downcast the opaque inner value to a concrete backend type.
    pub fn downcast_inner<T: 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }
}

/// Compiled GPU kernel handle.
///
/// Created by [`TamGpu::compile`]. Reuse across multiple dispatches — each
/// compile call incurs JIT cost (NVRTC ~40 ms); subsequent dispatches are fast.
// No #[derive(Debug)]: inner is Box<dyn Any>, which isn't Debug.
// Implement manually to show at least the entry name.
pub struct Kernel {
    pub(crate) inner: Box<dyn std::any::Any + Send + Sync>,
    /// Entry point name (e.g. `"scatter_phi"`, `"argmin_f64"`).
    pub entry: String,
}

impl Kernel {
    /// Construct a kernel wrapping a backend-specific inner value.
    pub fn new(inner: Box<dyn std::any::Any + Send + Sync>, entry: impl Into<String>) -> Self {
        Self { inner, entry: entry.into() }
    }

    /// Downcast the opaque inner value to a concrete backend type.
    pub fn downcast_inner<T: 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }
}

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

/// Unified GPU compute backend.
///
/// Callers allocate buffers, compile kernels, copy data, dispatch kernels,
/// and sync — without knowing which driver is underneath.
pub trait TamGpu: Send + Sync {
    /// Human-readable device + backend label.
    ///
    /// Example: `"NVIDIA RTX PRO 6000 (CUDA)"`, `"Apple M4 Pro (Metal)"`,
    /// or `"CPU (native Rust)"`.
    fn name(&self) -> String;

    fn backend(&self) -> Backend;
    fn shader_lang(&self) -> ShaderLang;

    /// Compile `source` with entry point `entry`.
    ///
    /// - **CUDA**: `source` is CUDA C, compiled via NVRTC to PTX.
    /// - **Vulkan**: `source` is WGSL or SPIR-V bytes.
    /// - **CPU**: `source` is ignored; `entry` is looked up in the builtin registry.
    ///
    /// Returns a [`Kernel`] handle that may be reused indefinitely.
    fn compile(&self, source: &str, entry: &str) -> TamResult<Kernel>;

    /// Allocate `bytes` of zero-initialised backend memory.
    fn alloc(&self, bytes: usize) -> TamResult<Buffer>;

    /// Free a buffer immediately. Equivalent to `drop(buf)`.
    fn free(&self, buf: Buffer) -> TamResult<()>;

    /// Copy host slice → device buffer. `src.len()` must be ≤ `dst.size`.
    fn copy_h2d(&self, src: &[u8], dst: &Buffer) -> TamResult<()>;

    /// Copy device buffer → host slice. `dst.len()` must be ≤ `src.size`.
    fn copy_d2h(&self, src: &Buffer, dst: &mut [u8]) -> TamResult<()>;

    /// Launch `kernel` over a `grid × block` thread space.
    ///
    /// `bufs`: device buffers in the order the kernel expects them.
    /// `shared_mem`: per-block dynamic shared memory in bytes.
    ///
    /// The total thread count is `grid[0] * block[0]` (y/z dims reserved for
    /// future 2-D and 3-D kernels).
    fn dispatch(
        &self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        bufs: &[&Buffer],
        shared_mem: u32,
    ) -> TamResult<()>;

    /// Block until all in-flight operations complete.
    ///
    /// For [`CpuBackend`] this is a no-op (CPU is always synchronous).
    fn sync(&self) -> TamResult<()>;
}

// ---------------------------------------------------------------------------
// Auto-detection
// ---------------------------------------------------------------------------

/// Auto-detect and return the best available backend.
///
/// Detection order: **CUDA** (if feature `cuda` compiled in and driver present)
/// → **CPU fallback** (always available).
///
/// Vulkan and Metal will be inserted ahead of CPU when implemented.
///
/// The result is cached in a process-wide [`OnceLock`] — CUDA initialisation
/// only happens once, regardless of how many concurrent callers race here.
pub fn detect() -> std::sync::Arc<dyn TamGpu> {
    use std::sync::{Arc, OnceLock};
    static GLOBAL_GPU: OnceLock<Arc<dyn TamGpu>> = OnceLock::new();
    GLOBAL_GPU.get_or_init(|| {
        #[cfg(feature = "cuda")]
        if let Ok(b) = CudaBackend::new() {
            return Arc::new(b);
        }
        Arc::new(CpuBackend::new())
    }).clone()
}

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

/// Upload a typed slice to a freshly-allocated buffer.
///
/// Equivalent to `alloc(n * size_of::<T>())` + `copy_h2d(cast_slice(data), &buf)`.
pub fn upload<T: bytemuck::Pod>(gpu: &dyn TamGpu, data: &[T]) -> TamResult<Buffer> {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let buf = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, &buf)?;
    Ok(buf)
}

/// Download a buffer into a typed Vec.
///
/// `n_elems` is the number of `T` elements to read.
pub fn download<T: bytemuck::Pod>(gpu: &dyn TamGpu, buf: &Buffer, n_elems: usize) -> TamResult<Vec<T>> {
    let mut bytes = vec![0u8; n_elems * std::mem::size_of::<T>()];
    gpu.copy_d2h(buf, &mut bytes)?;
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}
