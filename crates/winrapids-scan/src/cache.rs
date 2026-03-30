//! Kernel compilation cache.
//!
//! BLAKE3 hash of (operator name + params + CUDA source) -> cached PTX.
//! In-memory HashMap for hot path. Disk cache for cross-process persistence.

use crate::ops::AssociativeOp;
use crate::engine::generate_scan_kernel;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use std::collections::HashMap;
use std::path::PathBuf;

pub struct KernelCache {
    memory: HashMap<String, Ptx>,
    cache_dir: PathBuf,
}

impl KernelCache {
    pub fn new() -> Self {
        let cache_dir = cache_dir_path();
        std::fs::create_dir_all(&cache_dir).ok();
        Self {
            memory: HashMap::new(),
            cache_dir,
        }
    }

    /// Get compiled PTX for an operator, compiling via NVRTC if needed.
    pub fn get_or_compile(
        &mut self,
        op: &dyn AssociativeOp,
    ) -> Result<Ptx, Box<dyn std::error::Error>> {
        let source = generate_scan_kernel(op);
        let key = cache_key(op, &source);

        // Check in-memory cache
        if let Some(ptx) = self.memory.get(&key) {
            return Ok(ptx.clone());
        }

        // Check disk cache
        let disk_path = self.cache_dir.join(format!("{}.ptx", &key[..16]));
        if disk_path.exists() {
            let ptx_bytes = std::fs::read_to_string(&disk_path)?;
            let ptx = Ptx::from_src(ptx_bytes);
            self.memory.insert(key, ptx.clone());
            return Ok(ptx);
        }

        // Compile via NVRTC
        let opts = CompileOptions {
            arch: Some("sm_120"), // Blackwell
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(source, opts)?;

        // Save to disk cache
        std::fs::write(&disk_path, ptx.to_src()).ok();

        // Save to memory cache
        self.memory.insert(key, ptx.clone());
        Ok(ptx)
    }

    /// Generate CUDA source without compiling (for inspection/testing).
    pub fn generate_source(&self, op: &dyn AssociativeOp) -> String {
        generate_scan_kernel(op)
    }
}

fn cache_key(op: &dyn AssociativeOp, source: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(op.name().as_bytes());
    hasher.update(op.params_key().as_bytes());
    hasher.update(source.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn cache_dir_path() -> PathBuf {
    let base = std::env::var("LOCALAPPDATA")
        .unwrap_or_else(|_| ".cache".into());
    PathBuf::from(base).join("winrapids").join("kernel_cache")
}
