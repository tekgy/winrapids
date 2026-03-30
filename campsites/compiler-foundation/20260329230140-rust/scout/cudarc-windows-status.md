# cudarc on Windows: Comprehensive Status

*Scout: 2026-03-29 — based on primary source research*

---

## Current Version and Windows Build Target

- **Version**: 0.19.4 (current as of March 2026)
- **Windows MSVC build**: first-class — docs.rs builds for `x86_64-pc-windows-msvc` explicitly
- **CUDA version support**: 11.4 through 13.2 (feature flag `cuda-13020` for 13.2)
- **Blackwell (sm_100)**: supported via CUDA 13.x

---

## Default Feature Set

cudarc's defaults in Cargo.toml include:
```
"std", "cublas", "cublaslt", "curand", "driver", "runtime", "nvrtc", "fallback-dynamic-loading"
```

**NVRTC is included by default** — no extra feature flag needed. The `fallback-dynamic-loading` mode means CUDA .dlls are loaded at runtime, not linked at build time. This is the correct mode for pip-installable packages: no CUDA headers needed on the build machine.

---

## Library Support Matrix

| Library | dynamic-load | dynamic-link | static-link |
|---|---|---|---|
| cuBLAS | ✅ | ✅ | ✅ |
| cuBLASLt | ✅ | ✅ | ✅ |
| cuFFT | ✅ | ❌ | ❌ |
| cuRAND | ✅ | ✅ | ❌ |
| NVRTC | ✅ | — | — |
| cuDNN | ✅ | — | — |

**cuFFT limitation**: only dynamic-load works. This means `cufft64_11.dll` must be on the system at runtime. For the FFT primitive in `winrapids-primitives`, use dynamic-load path. This is fine for any CUDA-installed system (cuFFT .dll ships with CUDA toolkit) but must be documented.

---

## Windows Runtime Requirements

For NVRTC specifically, must have CUDA bin directory on `%PATH%`:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin
```

A pip-installable package can set this in its `__init__.py` before importing the Rust extension:
```python
import os
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
if cuda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = cuda_bin + ";" + os.environ.get("PATH", "")
```

Also: `cl.exe` (MSVC compiler) must be on PATH for NVRTC to compile CUDA C source. This is available in any Visual Studio installation.

---

## Known Windows-Specific Issues

No documented Windows-specific bugs were found in the cudarc repository. The platform is actively supported. However:

1. **WDDM kernel launch overhead**: ~7 μs per launch vs ~2-3 μs on Linux/TCC. Not a cudarc issue — inherent to WDDM.
2. **TDR (Timeout Detection and Recovery)**: kernels running >2 seconds are killed by Windows OS. Not a cudarc issue, but affects any long-running kernel design.
3. **cuFFT linking**: documented limitation above.

---

## NVRTC API Shape

```rust
use cudarc::nvrtc::safe::{compile_ptx, compile_ptx_with_opts, Ptx};

// Compile at runtime:
let ptx: Ptx = compile_ptx("extern \"C\" __global__ void my_kernel(...) { ... }")?;

// With options (e.g., target architecture):
let opts = CompileOptions {
    options: vec!["--gpu-architecture=compute_100".to_string()],
    ..Default::default()
};
let ptx: Ptx = compile_ptx_with_opts(src, opts)?;

// Load into device:
let module = ctx.load_module(ptx)?;
let func = module.load_function("my_kernel")?;
```

---

## Blackwell Targeting

For Blackwell (RTX PRO 6000):
```
--gpu-architecture=compute_100     // virtual arch: forward-compatible PTX
--gpu-architecture=sm_100          // native cubin: max Blackwell performance
--gpu-architecture=sm_100a         // full Blackwell features (tcgen05, not forward-compat)
```

Ship dual: `sm_100` (default, portable) + `sm_100a` (optional, max Blackwell). Select at runtime based on `cudaGetDeviceProperties`.

---

## Assessment for E12

**E12 (cudarc kernel launch on Windows) is unblocked.** No known issues. Estimated effort: ~50-100 lines of Rust. The experiment is essentially confirming what the docs promise, plus measuring WDDM kernel launch latency baseline.

Recommended E12 kernel: vector addition (trivial correctness verification) + timing loop to measure per-launch overhead under WDDM.
