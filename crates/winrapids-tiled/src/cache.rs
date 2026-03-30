//! Kernel cache for tiled accumulation.
//!
//! BLAKE3 hash of (operator name + params + CUDA source) → cache key.
//! Source generation is cached in memory. NVRTC compilation is deferred
//! to when cudarc is added (same pattern as winrapids-scan/cache.rs).

use std::collections::HashMap;

use crate::ops::TiledOp;
use crate::engine::generate_tiled_kernel;

pub struct TiledCache {
    /// Cache: key → generated CUDA source.
    sources: HashMap<String, String>,
}

impl TiledCache {
    pub fn new() -> Self {
        Self { sources: HashMap::new() }
    }

    /// Get or generate CUDA source for an operator. Cached by identity key.
    pub fn get_or_generate(&mut self, op: &dyn TiledOp) -> &str {
        let key = cache_key(op);
        self.sources.entry(key).or_insert_with(|| generate_tiled_kernel(op))
    }

    /// Generate source without caching (for inspection/testing).
    pub fn generate_source(op: &dyn TiledOp) -> String {
        generate_tiled_kernel(op)
    }
}

/// Compute cache key for a tiled operator.
pub fn cache_key(op: &dyn TiledOp) -> String {
    let source = generate_tiled_kernel(op);
    let mut hasher = blake3::Hasher::new();
    hasher.update(op.name().as_bytes());
    hasher.update(op.params_key().as_bytes());
    hasher.update(source.as_bytes());
    hasher.finalize().to_hex().to_string()
}
