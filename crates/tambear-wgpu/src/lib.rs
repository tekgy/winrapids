//! wgpu compute backend for TamGpu.
//!
//! `WgpuBackend` wraps `wgpu::Device` + `wgpu::Queue` and implements the
//! full [`TamGpu`] trait. WGSL shader source is compiled to the backend's
//! native ISA by wgpu/naga at runtime.
//!
//! Platform support (wgpu auto-selects):
//! - Windows: Vulkan (NVIDIA/AMD/Intel) → DX12 → DX11
//! - macOS/iOS: Metal
//! - Linux: Vulkan
//!
//! ## Precision note
//!
//! WGSL does not support `f64`. All tiled operations run in `f32` (7 decimal
//! digits vs. 15 for CUDA `f64`). Sufficient for ML inference and most signal
//! processing. For financial covariance with tight tolerances, use the CUDA backend.

use std::sync::Arc;

use tam_gpu::{Backend, Buffer, Kernel, ShaderLang, TamGpu, TamGpuError, TamResult};

// ---------------------------------------------------------------------------
// Internal buffer type
// ---------------------------------------------------------------------------

struct WgpuBuffer {
    buf: Arc<wgpu::Buffer>,
}

// wgpu types are thread-safe (wgpu is designed for multithreaded use)
unsafe impl Send for WgpuBuffer {}
unsafe impl Sync for WgpuBuffer {}

// ---------------------------------------------------------------------------
// Internal kernel type
// ---------------------------------------------------------------------------

struct WgpuKernel {
    pipeline:         wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

unsafe impl Send for WgpuKernel {}
unsafe impl Sync for WgpuKernel {}

// ---------------------------------------------------------------------------
// WgpuBackend
// ---------------------------------------------------------------------------

/// wgpu compute backend — WGSL → SPIR-V/MSL/DX12 at runtime.
///
/// Automatically selects the highest-performance adapter on the system.
/// Use [`WgpuBackend::new`] to initialize; returns `None` if no GPU is found.
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue:  Arc<wgpu::Queue>,
    name:   String,
}

impl WgpuBackend {
    /// Initialize wgpu, select the best adapter, and create a device.
    ///
    /// Returns `None` if no GPU adapter is available (very rare on modern hardware).
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            },
        ))?;

        let info = adapter.get_info();
        let name = format!("{} (wgpu/{:?})", info.name, info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("tambear-wgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_compute_workgroup_size_x: 1024,
                    max_compute_workgroup_size_y: 1024,
                    max_compute_invocations_per_workgroup: 1024,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )).ok()?;

        Some(Self {
            device: Arc::new(device),
            queue:  Arc::new(queue),
            name,
        })
    }
}

impl TamGpu for WgpuBackend {
    fn name(&self) -> String { self.name.clone() }

    fn backend(&self) -> Backend { Backend::Vulkan }

    /// Returns `ShaderLang::Wgsl` — TiledEngine will call `generate_tiled_kernel_wgsl`.
    fn shader_lang(&self) -> ShaderLang { ShaderLang::Wgsl }

    /// Compile WGSL `source` into a `ComputePipeline` for `entry`.
    ///
    /// wgpu/naga translates WGSL → SPIR-V/MSL/DX12 HLSL internally.
    /// First call is ~1 ms (vs. CUDA NVRTC ~40 ms).
    fn compile(&self, source: &str, entry: &str) -> TamResult<Kernel> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some(entry),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some(entry),
            layout:              None,    // auto-layout from WGSL reflection
            module:              &shader,
            entry_point:         Some(entry),
            compilation_options: Default::default(),
            cache:               None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Ok(Kernel::new(
            Box::new(WgpuKernel { pipeline, bind_group_layout }),
            entry,
        ))
    }

    /// Allocate `bytes` of zero-initialised GPU storage (STORAGE | COPY_SRC | COPY_DST).
    fn alloc(&self, bytes: usize) -> TamResult<Buffer> {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              None,
            size:               bytes as u64,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::COPY_SRC
                              | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Buffer::new(Box::new(WgpuBuffer { buf: Arc::new(buf) }), bytes))
    }

    fn free(&self, buf: Buffer) -> TamResult<()> {
        drop(buf);
        Ok(())
    }

    /// Host → device: writes `src` bytes into the storage buffer via `queue.write_buffer`.
    fn copy_h2d(&self, src: &[u8], dst: &Buffer) -> TamResult<()> {
        let inner = wgpu_buf(dst)?;
        self.queue.write_buffer(&inner.buf, 0, src);
        Ok(())
    }

    /// Device → host: copies storage buffer into `dst` via a staging buffer.
    fn copy_d2h(&self, src: &Buffer, dst: &mut [u8]) -> TamResult<()> {
        let inner = wgpu_buf(src)?;

        // Create a MAP_READ staging buffer.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("tambear-staging"),
            size:               dst.len() as u64,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Submit: copy storage → staging.
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&inner.buf, 0, &staging, 0, dst.len() as u64);
        self.queue.submit(Some(enc.finish()));

        // Map staging and read back.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| TamGpuError::Transfer("map_async channel closed".into()))?
            .map_err(|e| TamGpuError::Transfer(format!("map_async failed: {:?}", e)))?;

        let mapped = slice.get_mapped_range();
        dst.copy_from_slice(&mapped[..dst.len()]);
        drop(mapped);
        staging.unmap();
        Ok(())
    }

    /// Dispatch `kernel` with the given workgroup `grid`.
    ///
    /// `block` is ignored — the workgroup size is declared in WGSL source.
    /// `shared_mem` is ignored — workgroup memory is declared in WGSL source.
    fn dispatch(
        &self,
        kernel: &Kernel,
        grid:       [u32; 3],
        _block:     [u32; 3],
        bufs:       &[&Buffer],
        _shared_mem: u32,
    ) -> TamResult<()> {
        let k = kernel.downcast_inner::<WgpuKernel>()
            .ok_or_else(|| TamGpuError::Dispatch("not a wgpu kernel".into()))?;

        // Collect inner wgpu buffers before building bind group entries.
        let wbufs: Vec<&WgpuBuffer> = bufs.iter()
            .map(|b| wgpu_buf(b))
            .collect::<TamResult<Vec<_>>>()?;

        let entries: Vec<wgpu::BindGroupEntry<'_>> = wbufs.iter().enumerate()
            .map(|(i, wb)| wgpu::BindGroupEntry {
                binding:  i as u32,
                resource: wb.buf.as_entire_binding(),
            })
            .collect();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &k.bind_group_layout,
            entries: &entries,
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
        }
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }

    /// Block until all submitted GPU work completes.
    fn sync(&self) -> TamResult<()> {
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private helper
// ---------------------------------------------------------------------------

fn wgpu_buf(buf: &Buffer) -> TamResult<&WgpuBuffer> {
    buf.downcast_inner::<WgpuBuffer>()
        .ok_or_else(|| TamGpuError::Dispatch("expected a wgpu buffer".into()))
}

// ---------------------------------------------------------------------------
// Auto-detection
// ---------------------------------------------------------------------------

/// Detect and return the best available wgpu backend, or fall back to CPU.
///
/// Inserts `WgpuBackend` ahead of the CPU fallback in the detection chain.
/// Call this instead of `tam_gpu::detect()` when you want Vulkan/Metal/DX12
/// support.
pub fn detect_wgpu() -> Box<dyn TamGpu> {
    if let Some(b) = WgpuBackend::new() {
        return Box::new(b);
    }
    tam_gpu::detect()
}
