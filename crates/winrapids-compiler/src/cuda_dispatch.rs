//! GPU kernel dispatch: routes PrimitiveOp → real CUDA kernels.
//!
//! This is the last wire: the compiler plans, execute.rs walks the plan,
//! and CudaKernelDispatcher fires actual GPU kernels for each MISS step.
//!
//! Dispatches:
//!   - Scan     → ScanEngine (multi-block Blelloch scan)
//!   - FusedExpr → FusedExprEngine (rolling_mean / rolling_std / rolling_zscore)
//!
//! Both engines share one CudaContext so scan output pointers are directly
//! readable by fused_expr kernels without cross-context copies.

use winrapids_scan::{ScanEngine, ScanDeviceOutput, AddOp, WelfordOp, EWMOp, KalmanAffineOp};
use winrapids_scan::fused_expr::{FusedExprEngine, FusedExprOutput};
use winrapids_scan::ops::AssociativeOp;
use winrapids_store::header::BufferPtr;

use crate::ir::PrimitiveOp;
use crate::execute::KernelDispatcher;

/// Real GPU kernel dispatcher. Routes primitive ops to CUDA kernel launches.
///
/// ScanEngine and FusedExprEngine share one CudaContext — scan outputs are
/// allocated there and accessible directly by fused expression kernels.
pub struct CudaKernelDispatcher {
    scan_engine: ScanEngine,
    fused_engine: FusedExprEngine,
    /// Scan output buffers kept alive until dispatcher drops.
    scan_outputs: Vec<ScanDeviceOutput>,
    /// Fused expression output buffers kept alive until dispatcher drops.
    fused_outputs: Vec<FusedExprOutput>,
}

impl CudaKernelDispatcher {
    /// Create a dispatcher on GPU 0.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let scan_engine = ScanEngine::new()?;
        let ctx = scan_engine.ctx().clone();
        let stream = scan_engine.stream().clone();
        let fused_engine = FusedExprEngine::with_context(ctx, stream);
        Ok(Self {
            scan_engine,
            fused_engine,
            scan_outputs: Vec::new(),
            fused_outputs: Vec::new(),
        })
    }

    /// Create a dispatcher on a specific GPU.
    pub fn on_device(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let scan_engine = ScanEngine::on_device(ordinal)?;
        let ctx = scan_engine.ctx().clone();
        let stream = scan_engine.stream().clone();
        let fused_engine = FusedExprEngine::with_context(ctx, stream);
        Ok(Self {
            scan_engine,
            fused_engine,
            scan_outputs: Vec::new(),
            fused_outputs: Vec::new(),
        })
    }

    /// Access the scan engine (for validation, dtoh copies, etc.).
    pub fn engine(&self) -> &ScanEngine {
        &self.scan_engine
    }

    /// Copy a device buffer back to host. Useful for numerical validation.
    ///
    /// # Safety
    /// `ptr.device_ptr` must be a valid allocation of at least `ptr.byte_size` bytes
    /// on the same CUDA context as this dispatcher.
    pub unsafe fn copy_to_host(&self, ptr: winrapids_store::header::BufferPtr)
        -> Result<Vec<f64>, Box<dyn std::error::Error>>
    {
        let n = (ptr.byte_size / 8) as usize;
        let stream = self.scan_engine.stream();
        let slice = std::mem::ManuallyDrop::new(
            stream.upgrade_device_ptr::<f64>(ptr.device_ptr, n)
        );
        Ok(stream.clone_dtoh(&*slice)?)
    }
}

/// Parse scan aggregation type from step params.
fn parse_scan_op(params: &[(String, String)]) -> Result<Box<dyn AssociativeOp>, Box<dyn std::error::Error>> {
    let agg = params.iter()
        .find(|(k, _)| k == "agg")
        .map(|(_, v)| v.as_str())
        .ok_or("Scan step missing 'agg' parameter")?;

    match agg {
        "add" => Ok(Box::new(AddOp)),
        "welford" => Ok(Box::new(WelfordOp)),
        "ewm" => {
            let alpha: f64 = params.iter()
                .find(|(k, _)| k == "alpha")
                .map(|(_, v)| v.parse::<f64>())
                .ok_or("EWM scan missing 'alpha' parameter")??;
            Ok(Box::new(EWMOp { alpha }))
        }
        "kalman_affine" => {
            let get = |key: &str| -> Result<f64, Box<dyn std::error::Error>> {
                params.iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v.parse::<f64>())
                    .ok_or_else(|| -> Box<dyn std::error::Error> { format!("kalman_affine missing '{}' param", key).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })
            };
            Ok(Box::new(KalmanAffineOp::new(get("F")?, get("H")?, get("Q")?, get("R")?)))
        }
        other => Err(format!("Unknown scan aggregation: {}", other).into()),
    }
}

/// Extract a named param value from the params list.
fn get_param<'a>(params: &'a [(String, String)], key: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    params.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .ok_or_else(|| format!("Missing param '{}' in fused_expr step", key).into())
}

impl KernelDispatcher for CudaKernelDispatcher {
    fn dispatch(
        &mut self,
        op: &PrimitiveOp,
        params: &[(String, String)],
        input_ptrs: &[BufferPtr],
    ) -> Result<BufferPtr, Box<dyn std::error::Error>> {
        match op {
            PrimitiveOp::Scan => {
                let scan_op = parse_scan_op(params)?;
                let input = &input_ptrs[0];
                let n = (input.byte_size / 8) as usize; // f64 elements

                let output = unsafe {
                    self.scan_engine.scan_device_ptr(
                        scan_op.as_ref(),
                        input.device_ptr,
                        n,
                    )?
                };

                let ptr = BufferPtr {
                    device_ptr: output.primary_device_ptr(),
                    byte_size: output.primary_byte_size(),
                };

                // Keep the output alive — downstream nodes read this pointer.
                self.scan_outputs.push(output);
                Ok(ptr)
            }

            PrimitiveOp::FusedExpr => {
                let formula = get_param(params, "formula")?;
                let window: usize = get_param(params, "window")?.parse()?;

                // n is derived from the cumsum input (always input[1] for all
                // E04 formulas: [data, cs] or [data, cs, cs2]).
                let n = (input_ptrs[1].byte_size / 8) as usize;

                // Extract the formula-specific input pointers in the order
                // that FusedExprEngine::dispatch expects.
                let fused_inputs: Vec<u64> = match formula {
                    "rolling_mean" => {
                        // inputs: [data(0), cs(1)] — only cs needed
                        vec![input_ptrs[1].device_ptr]
                    }
                    "rolling_std" => {
                        // inputs: [data(0), cs(1), cs2(2)] — cs and cs2 needed
                        vec![input_ptrs[1].device_ptr, input_ptrs[2].device_ptr]
                    }
                    "rolling_zscore" => {
                        // inputs: [data(0), cs(1), cs2(2)] — all needed
                        vec![
                            input_ptrs[0].device_ptr,
                            input_ptrs[1].device_ptr,
                            input_ptrs[2].device_ptr,
                        ]
                    }
                    other => return Err(format!("Unknown fused_expr formula: '{}'", other).into()),
                };

                let output = unsafe {
                    self.fused_engine.dispatch(formula, window, &fused_inputs, n)?
                };

                let ptr = BufferPtr {
                    device_ptr: output.device_ptr(),
                    byte_size: output.byte_size(),
                };

                // Keep output alive — the store holds this device pointer.
                self.fused_outputs.push(output);
                Ok(ptr)
            }

            other => Err(format!("primitive {:?} not yet dispatched — engine not built", other).into()),
        }
    }
}
