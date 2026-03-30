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
//!
//! # GPU Launch
//!
//! ```no_run
//! use winrapids_scan::{ScanEngine, AddOp};
//!
//! let mut engine = ScanEngine::new().unwrap();
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let result = engine.scan_inclusive(&AddOp, &data).unwrap();
//! assert_eq!(result.primary, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
//! ```

pub mod ops;
pub mod engine;
pub mod cache;
pub mod launch;
pub mod fused_expr;

pub use ops::{AssociativeOp, AddOp, MulOp, MaxOp, MinOp, WelfordOp, KalmanOp, EWMOp, KalmanAffineOp};
pub use engine::{generate_scan_kernel, generate_multiblock_scan};
pub use cache::KernelCache;
pub use launch::{ScanEngine, ScanResult, ScanDeviceOutput};
pub use fused_expr::{FusedExprEngine, FusedExprOutput};

// Re-export cudarc traits needed by downstream crates for device pointer access
pub use cudarc::driver::DevicePtr;
