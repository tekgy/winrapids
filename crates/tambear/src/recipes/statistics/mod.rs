//! Statistics recipes — correlation, factor analysis, descriptive stats.
//!
//! This subdirectory currently holds `.spec.toml` schema files for
//! statistics recipes whose `.rs` implementations live elsewhere
//! (`src/factor_analysis.rs`, `src/descriptive.rs`, etc.). As recipes
//! migrate to the new `recipes/<family>/` layout, their implementations
//! will move here to live next to their specs.
//!
//! The spec.toml files here are the single source of truth for
//! parameter defaults, output shapes, and IDE rendering metadata —
//! consumed by both the Rust schema layer and the tambear-ide.
