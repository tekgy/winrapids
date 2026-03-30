//! Kernel cache for local context.
//!
//! BLAKE3 hash of (identity_key + CUDA source) → cache key.
//! Source generation is cached in memory. NVRTC compilation is deferred
//! to when cudarc is added.

use std::collections::HashMap;

use crate::ops::LocalContextSpec;
use crate::engine::generate_local_context_kernel;

pub struct LocalContextCache {
    /// Cache: key → generated CUDA source.
    sources: HashMap<String, String>,
}

impl LocalContextCache {
    pub fn new() -> Self {
        Self { sources: HashMap::new() }
    }

    /// Get or generate CUDA source for a spec. Cached by identity key.
    pub fn get_or_generate(&mut self, spec: &LocalContextSpec) -> &str {
        let key = cache_key(spec);
        self.sources.entry(key).or_insert_with(|| generate_local_context_kernel(spec))
    }
}

/// Compute cache key for a local context spec.
pub fn cache_key(spec: &LocalContextSpec) -> String {
    let source = generate_local_context_kernel(spec);
    let mut hasher = blake3::Hasher::new();
    hasher.update(spec.identity_key().as_bytes());
    hasher.update(source.as_bytes());
    hasher.finalize().to_hex().to_string()
}
