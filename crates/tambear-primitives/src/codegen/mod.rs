//! Codegen: lower tambear passes to vendor-specific source.
//!
//! Each submodule is one vendor door. The output of every door is a
//! compilable string plus enough metadata (entry name, buffer arity)
//! for the runtime to dispatch.

pub mod cuda;

pub use cuda::{expr_to_cuda, pass_to_cuda_kernel, CudaKernelSource, CodegenError};
