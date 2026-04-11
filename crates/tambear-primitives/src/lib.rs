//! # tambear-primitives
//!
//! Flat catalog of mathematical primitives. One folder per primitive.
//! The filesystem IS the inventory.
//!
//! ## Architecture
//!
//! ```text
//! src/primitives/
//! ├── log_sum_exp/
//! │   ├── mod.rs           — the implementation
//! │   ├── params.toml      — machine-readable contract (IDE, TBS, sharing)
//! │   ├── primitive.tbs    — TBS script (the language, not just notation)
//! │   ├── workup.md        — Principle 10 publication-grade workup
//! │   ├── oracle.py        — mpmath/sympy 50-digit reference
//! │   ├── benchmark.rs     — scale sweep + competitor comparison
//! │   ├── README.md        — tutorial: what, when, how, composes with
//! │   └── tests/
//! │       ├── basic.rs     — golden path
//! │       ├── adversarial.rs — NaN, Inf, empty, degenerate
//! │       ├── oracle.rs    — bit-perfect against oracle.py
//! │       └── parity.rs    — vs scipy/R/MATLAB
//! └── pearson_r/
//!     └── ... (same structure)
//! ```
//!
//! ## Principles
//!
//! - **One primitive = one folder.** If it exists, the folder exists.
//! - **Flat, not nested.** Families are tags in params.toml, not directories.
//!   `log_sum_exp` belongs to [information_theory, numerical, probabilistic].
//! - **Self-describing.** params.toml has everything the IDE needs:
//!   parameters, types, defaults, sharing contract, using() keys, kingdom.
//! - **Self-proving.** workup.md + oracle.py + tests/ prove correctness.
//!   Missing files = unproven primitive.
//! - **node===node.** Every primitive has identical folder structure
//!   regardless of complexity. `log_sum_exp` looks like `eigendecomposition`.

pub mod primitives;
pub mod catalog;

// Flat re-export: `use tambear_primitives::log_sum_exp`
pub use primitives::*;
