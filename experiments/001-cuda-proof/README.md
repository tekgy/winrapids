# Experiment 001: CUDA Proof of Life

## Hypothesis

CUDA 13.1 can compile and execute kernels natively on Windows 11 targeting the RTX PRO 6000 Blackwell GPU with full compute capability support.

## Method

1. Compile `.cu` file with `nvcc -arch=sm_120` using MSVC 19.44 as host compiler
2. Query device properties via `cudaGetDeviceProperties` and `cudaDeviceGetAttribute`
3. Run vector add at three scales (1K, 1M, 64M elements) with correctness verification
4. Run parallel reduction at two scales (1M, 64M elements)
5. Test managed memory allocation, prefetch, and kernel execution

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
- Compute capability: 12.0 (sm_120)
- CUDA Toolkit: 13.1 (V13.1.115)
- CUDA Runtime/Driver: 13.0
- Host compiler: MSVC 19.44.35224 (VS 2022 Community)
- OS: Windows 11 Pro for Workstations
- Driver model: WDDM (not TCC)

## Results

### Device Properties

| Property | Value |
|----------|-------|
| SMs | 188 |
| Max threads/block | 1024 |
| Warp size | 32 |
| Global memory | 95.6 GB |
| Memory bus | 512 bits |
| L2 cache | 128 MB |
| Shared mem/block | 48 KB |
| Memory clock | 14001 MHz |
| GPU clock | 2280 MHz |
| Unified addressing | Yes |
| Managed memory | Yes |
| Compute preemption | Yes |

### Performance

| Test | Kernel Time | Effective BW | Result |
|------|------------|-------------|--------|
| Vector add 1K | 0.009 ms | 1.29 GB/s | PASS |
| Vector add 1M | 0.011 ms | 1187 GB/s | PASS |
| Vector add 64M | 0.480 ms | 1677 GB/s | PASS |
| Reduction 1M | 0.012 ms | 347 GB/s | PASS |
| Reduction 64M | 2.401 ms | 112 GB/s | PASS |

### Memory Transfer (PCIe)

| Direction | 64M elements | Bandwidth |
|-----------|-------------|-----------|
| Host → Device | 34.5 ms | 15.5 GB/s |
| Device → Host | 39.8 ms | 6.75 GB/s |

### Managed Memory (WDDM)

| Operation | Time | Notes |
|-----------|------|-------|
| cudaMallocManaged (4MB) | 66.6 ms | Very slow — WDDM overhead |
| cudaMemPrefetchAsync | N/A | NOT SUPPORTED on WDDM |
| Kernel on managed mem | 2.576 ms | Works but ~100x slower than explicit |

## Conclusions

1. **CUDA 13.1 works on Windows.** Full compilation and execution pipeline confirmed.

2. **API changes in CUDA 13:** `cudaDeviceProp.memoryClockRate` and `.clockRate` removed — must use `cudaDeviceGetAttribute` instead. `cudaMemPrefetchAsync` signature changed to use `cudaMemLocation` struct instead of raw device int.

3. **Blackwell sm_120 is fully supported.** 188 SMs, 128 MB L2 cache, compute 12.0.

4. **WDDM imposes real constraints:**
   - Managed memory prefetch is NOT supported
   - Managed allocation is very slow (~66ms for 4MB)
   - Must use explicit cudaMalloc/cudaMemcpy for performance

5. **PCIe bandwidth is asymmetric:** H2D ~15.5 GB/s, D2H ~6.75 GB/s. This matters for data pipeline design — prefer to keep data on GPU.

6. **GPU compute bandwidth is excellent:** 1677 GB/s effective on vector add approaches theoretical peak for this hardware.

7. **Key implication for WinRapids:** On Windows/WDDM, explicit memory management is not optional — it's required for performance. This is a fundamental difference from Linux/TCC where managed memory with prefetch can be competitive.

## Files

- `cuda_proof.cu` — Main experiment source
- `build.bat` — Build script (sets up MSVC paths, compiles with nvcc)
