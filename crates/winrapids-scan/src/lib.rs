//! # winrapids-scan
//!
//! Pluggable parallel scan with associative operators on GPU.
//!
//! The liftability principle in code: any sequential algorithm whose update
//! step composes associatively can be parallelized from O(n) to O(log n).
//!
//! The trait bound IS the scannability test. If your operator implements
//! [`AssociativeOp`], it's parallelizable. If it doesn't, you've hit the
//! Fock boundary.

pub mod ops;
pub mod engine;
pub mod cache;

pub use ops::{AssociativeOp, AddOp, MulOp, MaxOp, MinOp, WelfordOp, EWMOp};
pub use engine::generate_scan_kernel;
pub use cache::KernelCache;
