//! GPU launch layer for parallel scan.
//!
//! Wires the kernel generation + caching to actual GPU execution via cudarc.
//! Three-phase multi-block scan:
//!   1. scan_per_block — per-block inclusive scan
//!   2. scan_block_totals — single-block scan of block totals
//!   3. propagate_extract — propagate prefixes + extract outputs
//!
//! Supports up to BLOCK_SIZE × 1024 elements (1M with BLOCK_SIZE=1024).

use std::mem::ManuallyDrop;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};

use crate::cache::{cache_key, KernelCache};
use crate::engine::generate_multiblock_scan;
use crate::ops::AssociativeOp;

/// Block size for scan kernels. Power of 2.
const BLOCK_SIZE: usize = 1024;

/// Maximum number of blocks (limited by single-block scan_block_totals).
const MAX_BLOCKS: usize = 1024;

/// Result of a scan operation.
pub struct ScanResult {
    /// Primary output (e.g., cumulative sum, running mean).
    pub primary: Vec<f64>,
    /// Secondary outputs (e.g., running variance for WelfordOp).
    /// Empty for single-output operators.
    pub secondary: Vec<Vec<f64>>,
}

/// Result of a device-to-device scan. Owns the output GPU buffers.
///
/// The buffers remain allocated on GPU until this struct is dropped.
/// Use `primary_device_ptr()` to get the raw pointer for downstream kernels.
pub struct ScanDeviceOutput {
    primary: CudaSlice<f64>,
    primary_len: usize,
    secondary: Option<CudaSlice<f64>>,
    stream: Arc<CudaStream>,
}

impl ScanDeviceOutput {
    /// Raw device pointer for the primary output buffer.
    pub fn primary_device_ptr(&self) -> u64 {
        let (ptr, _guard) = self.primary.device_ptr(&self.stream);
        ptr
    }

    /// Number of f64 elements in primary output.
    pub fn primary_len(&self) -> usize {
        self.primary_len
    }

    /// Byte size of the primary output.
    pub fn primary_byte_size(&self) -> u64 {
        (self.primary_len * 8) as u64
    }

    /// Raw device pointer for secondary output (if any).
    pub fn secondary_device_ptr(&self) -> Option<u64> {
        self.secondary.as_ref().map(|s| {
            let (ptr, _guard) = s.device_ptr(&self.stream);
            ptr
        })
    }

    /// Copy primary output back to host. Useful for validation.
    pub fn primary_to_host(&self) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        Ok(self.stream.clone_dtoh(&self.primary)?)
    }
}

/// Compiled scan module: three functions for one operator.
struct ScanModule {
    scan_per_block: CudaFunction,
    scan_block_totals: CudaFunction,
    propagate_extract: CudaFunction,
}

/// GPU scan engine. Wraps CudaContext + KernelCache.
pub struct ScanEngine {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    cache: KernelCache,
    /// Cached compiled modules (op name + params → functions).
    modules: std::collections::HashMap<String, ScanModule>,
}

impl ScanEngine {
    /// Create a new scan engine on GPU 0.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::on_device(0)
    }

    /// Create a scan engine on a specific GPU.
    pub fn on_device(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let cache = KernelCache::new();
        Ok(Self {
            ctx,
            stream,
            cache,
            modules: std::collections::HashMap::new(),
        })
    }

    /// Run an inclusive parallel scan with the given associative operator.
    ///
    /// Returns the primary output (and secondary outputs for multi-output ops).
    pub fn scan_inclusive(
        &mut self,
        op: &dyn AssociativeOp,
        input: &[f64],
    ) -> Result<ScanResult, Box<dyn std::error::Error>> {
        let n = input.len();
        if n == 0 {
            return Ok(ScanResult { primary: vec![], secondary: vec![] });
        }

        let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        assert!(
            n_blocks <= MAX_BLOCKS,
            "Input too large: {} elements requires {} blocks (max {})",
            n, n_blocks, MAX_BLOCKS
        );

        let state_bytes = op.state_byte_size();
        let shared_bytes = (BLOCK_SIZE * state_bytes) as u32;

        // Ensure module is compiled and loaded
        let module_key = self.ensure_module(op)?;

        // Allocate device buffers
        let input_dev: CudaSlice<f64> = self.stream.clone_htod(input)?;
        // State buffer: n × state_byte_size bytes
        let mut state_dev: CudaSlice<u8> = self.stream.alloc_zeros(n * state_bytes)?;
        // Block totals: n_blocks × state_byte_size bytes
        let mut totals_dev: CudaSlice<u8> = self.stream.alloc_zeros(n_blocks * state_bytes)?;
        // Output buffers
        let mut out0_dev: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let out1_n = if op.output_width() > 1 { n } else { 1 };
        let mut out1_dev: CudaSlice<f64> = self.stream.alloc_zeros(out1_n)?;

        // Phase 1: scan_per_block
        {
            let module = self.modules.get(&module_key).unwrap();
            let cfg = LaunchConfig {
                grid_dim: (n_blocks as u32, 1, 1),
                block_dim: (BLOCK_SIZE as u32, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            unsafe {
                self.stream.launch_builder(&module.scan_per_block)
                    .arg(&input_dev)
                    .arg(&mut state_dev)
                    .arg(&mut totals_dev)
                    .arg(&(n as i32))
                    .launch(cfg)?;
            }
        }

        // Phase 2: scan_block_totals (only needed for multi-block)
        if n_blocks > 1 {
            let module = self.modules.get(&module_key).unwrap();
            let totals_block_size = n_blocks.next_power_of_two();
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (totals_block_size as u32, 1, 1),
                shared_mem_bytes: (totals_block_size * state_bytes) as u32,
            };
            unsafe {
                self.stream.launch_builder(&module.scan_block_totals)
                    .arg(&mut totals_dev)
                    .arg(&(n_blocks as i32))
                    .launch(cfg)?;
            }
        }

        // Phase 3: propagate_extract
        {
            let module = self.modules.get(&module_key).unwrap();
            let cfg = LaunchConfig {
                grid_dim: (n_blocks as u32, 1, 1),
                block_dim: (BLOCK_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream.launch_builder(&module.propagate_extract)
                    .arg(&mut out0_dev)
                    .arg(&mut out1_dev)
                    .arg(&state_dev)
                    .arg(&totals_dev)
                    .arg(&(n as i32))
                    .launch(cfg)?;
            }
        }

        // Synchronize and copy results back
        self.stream.synchronize()?;
        let primary = self.stream.clone_dtoh(&out0_dev)?;
        let secondary = if op.output_width() > 1 {
            vec![self.stream.clone_dtoh(&out1_dev)?]
        } else {
            vec![]
        };

        Ok(ScanResult { primary, secondary })
    }

    /// Run scan on data already resident on GPU. Zero-copy device-to-device.
    ///
    /// Returns owned output buffers — caller keeps them alive as long as the
    /// pointers are in use. No host round-trip.
    ///
    /// # Safety
    /// `input_ptr` must be a valid device pointer to at least `n` contiguous
    /// f64 elements on the same GPU context. Caller retains ownership of the
    /// input buffer.
    pub unsafe fn scan_device_ptr(
        &mut self,
        op: &dyn AssociativeOp,
        input_ptr: u64,
        n: usize,
    ) -> Result<ScanDeviceOutput, Box<dyn std::error::Error>> {
        if n == 0 {
            let primary = self.stream.alloc_zeros::<f64>(1)?;
            return Ok(ScanDeviceOutput {
                primary, primary_len: 0, secondary: None,
                stream: self.stream.clone(),
            });
        }

        let n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        assert!(
            n_blocks <= MAX_BLOCKS,
            "Input too large: {} elements requires {} blocks (max {})",
            n, n_blocks, MAX_BLOCKS
        );

        let state_bytes = op.state_byte_size();
        let shared_bytes = (BLOCK_SIZE * state_bytes) as u32;

        let module_key = self.ensure_module(op)?;

        // Wrap input device pointer without taking ownership.
        // ManuallyDrop prevents CudaSlice from freeing the caller's buffer.
        let input_dev = ManuallyDrop::new(
            self.stream.upgrade_device_ptr::<f64>(input_ptr, n)
        );

        // Allocate output-side device buffers (we own these)
        let mut state_dev: CudaSlice<u8> = self.stream.alloc_zeros(n * state_bytes)?;
        let mut totals_dev: CudaSlice<u8> = self.stream.alloc_zeros(n_blocks * state_bytes)?;
        let mut out0_dev: CudaSlice<f64> = self.stream.alloc_zeros(n)?;
        let out1_n = if op.output_width() > 1 { n } else { 1 };
        let mut out1_dev: CudaSlice<f64> = self.stream.alloc_zeros(out1_n)?;

        // Phase 1: scan_per_block
        {
            let module = self.modules.get(&module_key).unwrap();
            let cfg = LaunchConfig {
                grid_dim: (n_blocks as u32, 1, 1),
                block_dim: (BLOCK_SIZE as u32, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            self.stream.launch_builder(&module.scan_per_block)
                .arg(&*input_dev)
                .arg(&mut state_dev)
                .arg(&mut totals_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        // Phase 2: scan_block_totals (multi-block only)
        if n_blocks > 1 {
            let module = self.modules.get(&module_key).unwrap();
            let totals_block_size = n_blocks.next_power_of_two();
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (totals_block_size as u32, 1, 1),
                shared_mem_bytes: (totals_block_size * state_bytes) as u32,
            };
            self.stream.launch_builder(&module.scan_block_totals)
                .arg(&mut totals_dev)
                .arg(&(n_blocks as i32))
                .launch(cfg)?;
        }

        // Phase 3: propagate_extract
        {
            let module = self.modules.get(&module_key).unwrap();
            let cfg = LaunchConfig {
                grid_dim: (n_blocks as u32, 1, 1),
                block_dim: (BLOCK_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            self.stream.launch_builder(&module.propagate_extract)
                .arg(&mut out0_dev)
                .arg(&mut out1_dev)
                .arg(&state_dev)
                .arg(&totals_dev)
                .arg(&(n as i32))
                .launch(cfg)?;
        }

        self.stream.synchronize()?;

        let secondary = if op.output_width() > 1 {
            Some(out1_dev)
        } else {
            None
        };

        Ok(ScanDeviceOutput {
            primary: out0_dev,
            primary_len: n,
            secondary,
            stream: self.stream.clone(),
        })
    }

    /// Access the stream (for dtoh copies on ScanDeviceOutput).
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Access the CUDA context (for sharing with FusedExprEngine).
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Ensure the operator's PTX module is compiled and loaded.
    ///
    /// On first load, validates that the CUDA sizeof(state_t) matches the
    /// operator's state_byte_size(). Mismatch would cause silent memory
    /// corruption in shared memory scans — better to panic here.
    fn ensure_module(
        &mut self,
        op: &dyn AssociativeOp,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let key = format!("scan_mb_{}{}", op.name(), op.params_key());

        if !self.modules.contains_key(&key) {
            let source = generate_multiblock_scan(op);
            let ck = cache_key(op, &source);
            let ptx = self.cache.compile_source(&source, &ck)?;

            let module = self.ctx.load_module(ptx)?;
            let f1 = module.load_function("scan_per_block")?;
            let f2 = module.load_function("scan_block_totals")?;
            let f3 = module.load_function("propagate_extract")?;

            // sizeof validation: query CUDA's sizeof(state_t) and compare
            // against Rust's state_byte_size(). One-time check per operator.
            let f_sizeof = module.load_function("query_sizeof")?;
            let mut sizeof_buf: CudaSlice<i32> = self.stream.alloc_zeros(1)?;
            unsafe {
                self.stream.launch_builder(&f_sizeof)
                    .arg(&mut sizeof_buf)
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
            self.stream.synchronize()?;
            let sizeof_result = self.stream.clone_dtoh(&sizeof_buf)?;
            let cuda_sizeof = sizeof_result[0] as usize;
            let rust_sizeof = op.state_byte_size();
            assert_eq!(
                cuda_sizeof, rust_sizeof,
                "sizeof mismatch for operator '{}': CUDA sizeof(state_t) = {} bytes, \
                 Rust state_byte_size() = {} bytes. This would cause silent memory \
                 corruption in shared memory scans.",
                op.name(), cuda_sizeof, rust_sizeof
            );

            self.modules.insert(key.clone(), ScanModule {
                scan_per_block: f1,
                scan_block_totals: f2,
                propagate_extract: f3,
            });
        }

        Ok(key)
    }
}
