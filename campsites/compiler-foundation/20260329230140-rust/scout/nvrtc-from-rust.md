# NVRTC from Rust: API, Caching, and E15 Design

*Scout: 2026-03-29*

---

## cudarc NVRTC Module

The `cudarc::nvrtc::safe` module provides:

```rust
use cudarc::nvrtc::safe::{compile_ptx, compile_ptx_with_opts, CompileOptions, Ptx};

// Simple compile:
let ptx: Ptx = compile_ptx(cuda_source_str)?;

// With architecture targeting:
let ptx: Ptx = compile_ptx_with_opts(cuda_source_str, CompileOptions {
    options: vec![
        "--gpu-architecture=compute_100".into(),  // Blackwell virtual arch
        "--std=c++17".into(),
    ],
    ..Default::default()
})?;
```

**NVRTC is included in cudarc's default features** — no extra Cargo feature flag required.

---

## Ptx Struct API

The `Ptx` struct provides:

```rust
// From runtime compilation:
let ptx = compile_ptx(src)?;               // → Ptx

// From pre-compiled PTX string (load from cache):
let ptx = Ptx::from_src(ptx_string);       // from String / &str

// From pre-compiled PTX file on disk:
let ptx = Ptx::from_file(path);            // from file path

// From CUBIN binary:
let ptx = Ptx::from_binary(bytes);         // from binary CUBIN

// Serialize PTX to string (for caching):
let ptx_string: &str = ptx.to_src();       // panics if from binary
let ptx_bytes: Option<&[u8]> = ptx.as_bytes(); // if programmatically compiled
```

`Ptx` implements `Clone`, `Debug`, and `From<String>`.

---

## The Caching Pattern for E15

Kernel source → hash → cache hit/miss → load from disk or compile → load into device:

```rust
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

fn get_or_compile_ptx(
    source: &str,
    arch_opts: &[&str],
    cache_dir: &Path,
) -> Result<Ptx, Box<dyn std::error::Error>> {
    // Stable key: hash of (source + options)
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    arch_opts.hash(&mut hasher);
    let key = format!("{:016x}.ptx", hasher.finish());

    let cache_path = cache_dir.join(&key);

    if cache_path.exists() {
        // Cache hit: load PTX from disk
        let ptx_str = std::fs::read_to_string(&cache_path)?;
        Ok(Ptx::from_src(ptx_str))
    } else {
        // Cache miss: compile, save, return
        let opts = CompileOptions {
            options: arch_opts.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(source, opts)?;
        // Save PTX string to disk
        std::fs::create_dir_all(cache_dir)?;
        std::fs::write(&cache_path, ptx.to_src())?;
        Ok(ptx)
    }
}
```

**Cache location**: `~/.winrapids/kernel_cache/<hash>.ptx`

**Hash considerations**: `DefaultHasher` is NOT stable across Rust versions. For a persistent cache, use BLAKE3 or SHA-256 as the cache key. `DefaultHasher` changes between Rust releases, meaning cache invalidation on toolchain upgrade. Use `blake3::hash(source.as_bytes())` instead.

---

## First Compile Time

NVRTC first compile: ~40ms for a typical reduce kernel. This is a one-time cost:
- Load `winrapids` Python module → check cache → cache hit → ~1ms to load PTX from disk → ~1ms to load module into CUDA driver
- Total "warm" path: ~2ms
- Total "cold" path (first ever): ~42ms

For WinRapids, the pre-built pipeline library should ship with pre-compiled PTX for all ~100 shapes. Cold compilation only happens for user-defined JIT pipelines.

---

## Architecture Targeting for E15

For the pipeline generator, compile with both virtual and native arch:

```rust
let opts = CompileOptions {
    options: vec![
        "--gpu-architecture=compute_100".into(),    // virtual: PTX for forward compat
        "--generate-code=arch=compute_100,code=sm_100".into(), // native Blackwell cubin
    ],
    ..Default::default()
};
```

Or detect at runtime:
```rust
let device = CudaDevice::new(0)?;
let props = device.device_properties()?;
let arch_flag = format!("--gpu-architecture=sm_{}{}",
    props.major, props.minor);
```

For Blackwell sm_100: `--gpu-architecture=sm_100` → optimal Blackwell SASS.
For forward compatibility: include PTX via `compute_100`.

---

## E15 Experiment Design

1. **Generate** a simple reduction kernel source string (sum of float32 array)
2. **Compile** via NVRTC, time it: first compile ~40ms
3. **Save** PTX to cache dir
4. **Load from cache** on second call, time it: ~1-2ms
5. **Execute** compiled kernel, verify correctness
6. **Benchmark** throughput: how many unique kernel shapes can WinRapids compile per second?

Expected findings:
- First compile: 40-50ms per kernel shape
- Cached load: <2ms
- For a pipeline with 60 unique shapes: cold start = 2.4s, subsequent runs = 0.12s

The 40ms first-compile cost is real but bounded. The cache makes it a one-time cost per shape per machine.

---

## Key Insight: Content-Addressed Kernel Cache

The kernel cache should use a CONTENT hash of the source (not a filename or sequential ID), so:
- Same kernel source compiled on different machines → same cache key → can be shared
- Ship pre-compiled PTX in the package for the ~100 pre-built shapes
- Users' JIT kernels are cached locally at `~/.winrapids/kernel_cache/`
- Cache is machine-local (PTX is architecture-specific; can't share between sm_89 and sm_100)

The architecture should be baked into the cache key: `blake3(source + arch_opts)`.
