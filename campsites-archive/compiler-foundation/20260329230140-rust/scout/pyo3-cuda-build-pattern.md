# PyO3 + CUDA Build Pattern on Windows

*Scout: 2026-03-29*

---

## The Stack

```
Python (user API)
  ↕  pyo3 + maturin
Rust (winrapids-py crate)
  ↕  cudarc
CUDA (kernels via NVRTC or pre-compiled PTX)
```

No prior art found for this exact stack on Windows. E13 is genuinely novel ground.

---

## Maturin: The Right Build Tool

Maturin (`pip install maturin`) handles the full build pipeline for PyO3 crates on Windows:
- Detects `pyo3` dependency in Cargo.toml automatically
- Handles `.dll` → `.pyd` renaming (required for Python to import on Windows)
- Produces a `.whl` wheel file with `maturin build --release`
- `maturin develop` for development installs

**Cargo.toml for winrapids-py**:
```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module", "abi3-py39"] }
cudarc = { version = "0.19", features = ["cuda-13020", "nvrtc"] }
```

The `abi3-py39` feature makes the wheel compatible with Python 3.9+, avoiding separate wheels per Python version.

---

## Windows Build Configuration (build.rs)

For the NVRTC path (JIT compilation), no special build.rs is needed — cudarc's dynamic-loading handles CUDA .dlls at runtime.

For pre-compiled CUDA C kernels (E12 variant using `.cu` files):
```rust
// build.rs
fn main() {
    // Compile the .cu file to PTX using nvcc
    let cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0";

    // Option 1: use the `cc` crate with CUDA support
    cc::Build::new()
        .cuda(true)
        .flag("--gpu-architecture=compute_100")
        .file("src/kernels/reduce.cu")
        .compile("reduce_kernels");

    // Tell linker where to find CUDA runtime
    println!("cargo:rustc-link-search={}/lib/x64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
}
```

**However**: for E12 and E13, using NVRTC (compile at runtime, not build time) avoids the build.rs CUDA compilation entirely. The `.cu` source is a Rust string literal; cudarc compiles it at runtime. This is cleaner for the initial experiments.

---

## E13 Prototype Structure

Minimal Python → Rust → CUDA → Rust → Python roundtrip:

```rust
// src/lib.rs
use pyo3::prelude::*;
use cudarc::driver::*;
use cudarc::nvrtc::safe::compile_ptx;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void double_vec(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.0f;
}
"#;

#[pyfunction]
fn double_on_gpu(py: Python, data: Vec<f32>) -> PyResult<Vec<f32>> {
    let n = data.len();
    let ctx = CudaContext::new(0).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let stream = ctx.new_stream().unwrap();

    let mut gpu_data = ctx.htod_copy(data).unwrap();

    let ptx = compile_ptx(KERNEL_SRC).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let func = module.load_function("double_vec").unwrap();

    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe { stream.launch_kernel(&func, cfg, (&mut gpu_data, n as i32)).unwrap(); }
    stream.synchronize().unwrap();

    let result = ctx.dtoh_sync_copy(&gpu_data).unwrap();
    Ok(result)
}

#[pymodule]
fn winrapids(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double_on_gpu, m)?)?;
    Ok(())
}
```

**Python test**:
```python
import winrapids
result = winrapids.double_on_gpu([1.0, 2.0, 3.0, 4.0])
assert result == [2.0, 4.0, 6.0, 8.0]
```

---

## Key Build Steps for E13

```bash
# 1. Install maturin
pip install maturin

# 2. Build and install in development mode
maturin develop --release

# 3. Or build a wheel
maturin build --release
pip install target/wheels/winrapids-*.whl

# Windows: ensure CUDA PATH is set before import
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;%PATH%
```

---

## Known Gotchas for Windows

1. **MSVC toolchain required**: cudarc + NVRTC on Windows requires MSVC `cl.exe` for CUDA source compilation. Visual Studio Build Tools 2022 (free) provides this.

2. **PATH at runtime**: NVRTC needs CUDA bin on PATH at the time the Python process loads the extension. Set it in Python's `__init__.py` before the import.

3. **No existing cudarc+PyO3+Windows examples found**: This is genuinely unexplored territory. E13 is the experiment that will establish this works and document any surprises.

4. **CudaContext is not Send**: cudarc's `CudaContext` and `CudaStream` require careful threading with PyO3. Use `pyo3::Python::allow_threads` for long-running GPU operations to release the GIL, but ensure CUDA objects stay on the thread that created them. May need `Arc<Mutex<CudaContext>>` pattern.

---

## Overhead to Measure in E13

- `htod_copy` latency for small arrays (10K, 100K, 1M elements)
- `dtoh_sync_copy` latency
- `compile_ptx` first-time cost (~40ms; amortized)
- `load_module` cost (~1ms)
- `launch_kernel` + `synchronize` overhead
- Total roundtrip: Python → Rust → CUDA → Rust → Python for 1M floats

Expected baseline: ~5-10ms for 1M floats (dominated by PCIe transfer, ~3ms each way at ~8 GB/s).
