# DirectStorage Readiness: What to Test with R595
Created: 2026-03-27
By: navigator (written during driver update pause)

## UPDATE: MCDM Is Not Accessible On This Machine (2026-03-27)

Scout research closed this. Key corrections:

1. **MCDM requires headless GPU** — our RTX PRO 6000 Workstation Edition drives monitors.
   Display-attached → stays WDDM regardless of R595. MCDM is Server Edition default.
   To get MCDM: add a cheap secondary GPU (GT 710) to take display, free RTX PRO to go headless.

2. **MCDM wouldn't help anyway** — NVMe→GPU P2P DMA is Linux-only (GPUDirect Storage,
   kernel 6.2+, CUDA 12.8+). Windows has no equivalent in ANY driver mode.
   DirectStorage 1.4 requires system RAM staging, is DirectX 12 only (no CUDA).
   The NVMe→RAM→GPU path is mandatory on Windows.

3. **The actual ceiling**: pinned async reads already at 8.5 GB/s. Concurrent 4-worker reads
   saturate NVMe → ~1.5ms per ticker effective read time. That's the Windows ceiling.

4. **R595 is still worth installing** — LMEM reduction, CUDA 13.1 memory pool improvements.
   Just no DirectStorage gains.

The Tests 1-4 below are partly moot, but kept for reference.

---

## Original Analysis (written before scout research — assumptions corrected above)

The driver update from R581 → R595 is not routine maintenance. R595 is the threshold
for MCDM (Microsoft Compute Driver Model) mode on Windows — which is what unlocks
DirectStorage path for NVMe → GPU VRAM without CPU involvement.

~~Before this update: DirectStorage was hypothetically available but practically unreachable
on this hardware. After: the hardware path exists. First time we can actually measure it.~~
**CORRECTED: See above. MCDM inaccessible on display-attached workstation card.**

## Current Baseline (to beat)

From expedition benchmarks:
- MKTF float32 disk→GPU: **5.9ms** per ticker (598K rows, 15.5MB)
- Bottleneck breakdown: ~4.8ms disk read + ~1.1ms H2D transfer
- Theoretical NVMe bandwidth: 7 GB/s
- Actual measured: ~2.5 GB/s (35% utilization — Windows buffered I/O overhead)
- Concurrent 4-worker estimate: ~1.5ms per ticker (saturates ~10 GB/s effective)

The H2D step (1.1ms, 27 GB/s PCIe) is NOT the bottleneck. Disk read IS.

DirectStorage collapses both steps: NVMe → GPU VRAM directly, bypassing CPU/system RAM.
The theoretical ceiling: 7 GB/s NVMe → 15.5MB = **2.2ms**. But that assumes:
- Zero CPU overhead on the transfer path
- No format conversion mid-flight (raw bytes must match GPU memory layout)
- Sufficient queue depth (DirectStorage is async + batched)

MKTF v3 is ALREADY designed for this — 4096-byte column alignment, no compression,
no encoding that requires CPU decode. The file layout is the GPU memory layout.

## What to Test (Priority Order)

### Test 1: MCDM mode confirmation
Before writing any code — verify R595 actually activated MCDM.

```python
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_model.current", "--format=csv,noheader"],
    capture_output=True, text=True
)
print(result.stdout)  # should show "WDDM" or "MCDM"
```

If still WDDM: DirectStorage won't work, but concurrent async I/O path still valid.
If MCDM: proceed to Test 2.

### Test 2: DirectStorage availability probe
```python
import ctypes
try:
    ds = ctypes.windll.LoadLibrary("dstorage.dll")
    print("DirectStorage DLL: available")
except OSError:
    print("DirectStorage DLL: not found — install DirectStorage runtime")
```

DirectStorage runtime is separate from the driver. May need:
`winget install Microsoft.DirectStorage` or SDK from Windows Dev Center.

### Test 3: Pinned memory async pipeline (buildable now, R595 or not)
This is the pragmatic path — doesn't require DirectStorage API, uses
cudaMemcpyAsync + pinned host memory + overlapped reads:

```python
import concurrent.futures
import numpy as np
import cupy as cp

def async_load_ticker(path, stream):
    """Load one ticker: disk read → pinned host → async H2D."""
    data = read_mktf_data(path)  # disk read (CPU)
    pinned = cp.cuda.alloc_pinned_memory(data.nbytes)
    # ... copy to pinned, launch async H2D on stream
    return gpu_arr

# Pipeline: while GPU processes ticker N, disk reads ticker N+1
```

Expected gain: 5.9ms → ~2-3ms per ticker via overlap.
This is testable TODAY regardless of driver.

### Test 4: DirectStorage proper (if Tests 1+2 pass)
Full DirectStorage path requires COM initialization + IDStorageFactory.
Prototype structure:
1. `DStorageGetFactory()` → IDStorageFactory
2. `CreateQueue()` with `DSTORAGE_QUEUE_DESC` (GPU destination)
3. `OpenFile()` → IDStorageFile
4. `EnqueueRequest()` with column offset + size → direct to CUDA pointer
5. `Submit()` + wait on fence

Key: each MKTF column becomes one DirectStorage request. The 4096-byte alignment
exists for exactly this — O_DIRECT / DirectStorage needs sector-aligned buffers.

## Design Implications for MKTF

MKTF v3 is already DirectStorage-compatible. Nothing to change. Specifics:
- 4096-byte column alignment: matches NVMe sector AND DirectStorage buffer requirements
- No compression: DirectStorage can do GPU-side decompression (GDeflate), but
  float32 market data doesn't compress anyway (tested: LZ4 ratio 1.000x)
- Absolute file offsets in column directory: DirectStorage requests are
  (file, offset, size, gpu_ptr) — exactly what the column directory provides
- 15.5MB per ticker: small enough to pipeline many concurrent requests

## The Real Question

The concurrent async path (Test 3) is probably within reach of DirectStorage speeds
for our file sizes. At 15.5MB per file, the overhead of the DirectStorage COM API
might cost more than it saves vs. a clean async Python I/O pipeline.

DirectStorage shines for larger transfers (game assets, textures, bulk data).
For market ticks, the win may come from **batch reads** — queuing all 4604 tickers
into a single DirectStorage submission and letting the hardware sort out the schedule.

The bench_concurrent.py script (just committed) tests the Python async path.
We need a comparable DirectStorage bench to know which actually wins.

## Open Question for After Driver Update

Does MCDM mode change the behavior of `cudaMemcpyAsync` for pinned memory?
Specifically: does R595 in MCDM enable CUDA to initiate DMA without CPU involvement
even without the DirectStorage API?

The CUDA docs suggest that pinned (page-locked) memory + async copy already bypasses
CPU for the transfer itself, regardless of driver model. MCDM may affect the NVMe →
host portion more than the host → GPU portion.

Worth testing: bench pinned async H2D transfer speed before and after the driver update
on a fixed dataset. If it changes, MCDM is doing something. If it doesn't, the gain
will only come from the full DirectStorage pipeline.
