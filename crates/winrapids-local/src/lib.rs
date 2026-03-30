//! # winrapids-local
//!
//! Local context primitive — multi-offset gather + fused feature computation.
//!
//! The most-used primitive for time series: every lag feature, every return,
//! every peak detection, every local trend — all from one kernel, one read.
//!
//! Replaces O(n) separate kernels for shift/diff/rolling/peak with a single
//! O(n) fused kernel that reads data once and writes all features.
//!
//! This is fixed-offset local attention: where self-attention is O(n²) with
//! learned offsets, local_context is O(n) with fixed structural offsets.
//! For structured data (time series, sequences), the offsets are known and
//! fixed — no need to learn them.

pub mod ops;
pub mod engine;
pub mod cache;

pub use ops::{LocalFeature, LocalContextSpec};
pub use engine::generate_local_context_kernel;
pub use cache::{LocalContextCache, cache_key};
