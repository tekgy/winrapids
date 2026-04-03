//! # winrapids-tiled
//!
//! Tiled accumulation primitive — 2D blocked reduction with pluggable operators.
//!
//! The 2D analog of scan's AssociativeOp pattern. Covers:
//! - GEMM (DotProductOp)
//! - PCA covariance (CovarianceOp — fused centering)
//! - KNN distance (DistanceOp)
//! - FlashAttention (SoftmaxWeightedOp — online softmax)
//!
//! CSE identity: (A_id, B_id, op_name, params). Block-level sharing —
//! one node per matrix pair, not N² per element.
//!
//! The fusion advantage: pre-transforms (centering, normalization) are
//! applied during tile load, not as separate passes. This reads data
//! once where cuBLASLt would materialize intermediates.

pub mod ops;
pub mod engine;
pub mod cache;
pub mod dispatch;

pub use ops::{TiledOp, DotProductOp, OuterProductOp, CovarianceOp, DistanceOp, SoftmaxWeightedOp};
pub use engine::{generate_tiled_kernel, generate_tiled_kernel_wgsl};
pub use cache::{TiledCache, cache_key};
pub use dispatch::TiledEngine;
