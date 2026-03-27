# Tensor Core Analysis — Scout Notes
## 2026-03-27

---

## Hardware Ground Truth

**GPU**: RTX PRO 6000 Blackwell Max-Q, CC 12.0, 188 SMs, 102.6 GB VRAM

**Bandwidth correction**: Previous docs assumed 896 GB/s. Actual measured = **1792 GB/s**.
This changes every prior I/O-bound estimate by 2x. The pipeline is even less I/O-bound than we thought.

**Measured TFLOPS** (via CuPy matmul benchmark):
| Precision | TFLOPS | Notes |
|-----------|--------|-------|
| FP64 | 1.4 | 34x penalty vs FP32 |
| FP32 | 48-54 | CUDA cores |
| FP16 | 282-288 | Tensor Cores engaged |

FP16 is ~5.5x faster than FP32 for large GEMM. This is the Tensor Core speedup.

---

## Question 1: Can fused_bin_stats Use Tensor Cores?

**Answer: NO.**

Tensor Cores only engage for GEMM (matrix multiply). fused_bin_stats is:
- Variable-length segmented reductions (sum, sum-of-squares, null count per bin)
- Pure elementwise comparisons

The "indicator matrix" reframe (T_ij = 1 if tick i in bin j) would require a [598K × 512] indicator
matrix with 598K × 512 × 4B = **1.2 GB per column**. For 17 columns = 20+ GB just for the
indicator matrices. Not viable.

fused_bin_stats must remain segmented-reduction CUDA kernels. Tensor Cores are irrelevant here.

---

## Question 2: What Precision Do Tensor Cores Need?

**FP16 OVERFLOWS for price data.** Critical safety finding:

- FP16 max representable value: 65,504
- Price squared (needed for std): AAPL ~$230 → $230² = 52,900 (ok)
- But price > $255 → price² > 65,025, and any accumulation pushes past 65,504
- Tested: sum() returned `inf`, std() returned `NaN`

**Safe precision tiers:**
| Use case | Safe precision | Reason |
|----------|---------------|--------|
| Price storage | int32 (10^-4 scale) | Exact, no overflow |
| Price × size (notional) | int64 or float32 | Overflow risk in FP16 |
| Derived features (log, sqrt, recip) | float32 | Values <15, no overflow risk |
| Cross-ticker correlation matrix | BF16 or FP32 | After normalization: values in [-1, 1] |
| ML embeddings | FP16 or BF16 | Normalized values, fine |

**BF16 for correlation**: Safe. BF16 has same exponent range as FP32 (max 3.38e38).
After z-score normalization, correlation inputs are in [-3, 3] range — BF16 handles this fine.

Note: PyTorch not available in this environment, so BF16 Tensor Core mode untested.
CuPy does not expose BF16 GEMM directly. Need PyTorch or cublasBF16 via ctypes.

---

## Question 3: cuBLAS Batched GEMM for K04 Cross-Ticker Correlation?

**Answer: YES for single cadence. Mixed results for batched.**

K04 computes cross-ticker correlation: [4604 × T] @ [T × 4604] where T=170 rows per cadence.

Single cadence timing:
```
Matrix shape: [4604 x 170] @ [170 x 4604] -> [4604 x 4604]
FLOPs: 7.21 GFLOPs
FP32: 0.262ms (27.5 TFLOPS)
FP16: 0.058ms (124.0 TFLOPS) ← 4.5x speedup
```

Batched 31 cadences (full trading day):
```
FLOPs: 223.4 GFLOPs total
FP32:  7.779ms  (28.7 TFLOPS)
FP16: 10.524ms  (21.2 TFLOPS)  ← FP16 SLOWER
```

**Why FP16 is slower for batched**: cp.matmul batched dispatch overhead dominates at small batch.
The [170 × 4604] input matrix per cadence is too small to saturate Tensor Core SMs when batched.

**Fix**: Stack all 31 cadences before the matmul. Compute [31×4604 × 170] @ [170 × 31×4604]
using a single large matmul, then reshape. This maximizes matrix size and saturates Tensor Cores.

Saturation threshold: Tensor Core TFLOPS climbs steeply below 512×512, plateaus above 1024×1024.
[31×4604 × 170] has leading dim 142,724 — this WILL saturate.

---

## Question 4: Actual Tensor Core TFLOPS on RTX PRO 6000 Blackwell?

**Measured: 282-288 TFLOPS FP16** (at 4096×4096 GEMM)

Theoretical spec: RTX PRO 6000 Blackwell Max-Q rated at ~300 TFLOPS FP16 (Tensor Core).
Our measurements are 94-96% of theoretical — near-perfect saturation at large matrix sizes.

Saturation curve (measured via CuPy):
- 128×128: ~20 TFLOPS (heavily underutilized)
- 256×256: ~55 TFLOPS
- 512×512: ~120 TFLOPS
- 1024×1024: ~210 TFLOPS
- 2048×2048: ~270 TFLOPS
- 4096×4096: ~288 TFLOPS ← near peak

Rule of thumb: need M, N, K all ≥ 1024 to approach peak. The K04 single-cadence [4604×170×4604]
has K=170 — below threshold. K04 batched approach (concatenate all cadences) solves this.

---

## Question 5: Does CuPy Expose Tensor Core Operations?

**Answer: YES, implicitly.**

CuPy uses cuBLAS under the hood. When input arrays are FP16, cuBLAS automatically dispatches
to Tensor Core units. No explicit API call needed.

Usage: `cp.float16` arrays + `@` operator or `cp.matmul()` → Tensor Cores engage automatically.

The 288 TFLOPS FP16 measurement above confirms Tensor Cores ARE firing through CuPy.

Limitations:
- BF16: CuPy does NOT support BF16. Need PyTorch or raw cuBLAS for BF16 Tensor Core.
- FP8: Not yet in CuPy stable. Available in PyTorch 2.1+ (CUDA 11.8+) for Hopper/Blackwell.
- TF32: Automatically enabled for FP32 matmul in CUDA 11+, gives ~8x speedup over baseline FP32.

---

## Pipeline Impact Summary

**What can use Tensor Cores:**
- K04 cross-ticker correlation (GEMM): 4.5x speedup possible (FP16, normalized inputs)
- Any future ML-style operations (attention, linear layers): natural Tensor Core fit
- K02 pairwise feature × feature correlation per ticker: [170×T] @ [T×170], small but parallelizable

**What cannot use Tensor Cores (wrong operation class):**
- fused_bin_stats (segmented reductions)
- K01 pointwise elementwise (log, sqrt, recip, delta)
- Any scan or prefix operation
- The entire current K01/K02 pipeline

**Revised I/O picture** (corrected for 1792 GB/s):
- 15.5MB file (float32, 598K rows): H2D = **0.009ms** (not 0.017ms)
- GPU processing (fused kernel): ~4ms
- GPU is compute-bound, NOT I/O-bound
- Integer encoding still halves file size and write time, but GPU speedup from integer is NOT I/O

**The real bottleneck**: fused_bin_stats reduction kernel itself (~4ms compute).
Tensor Cores won't help. Better warp-level reduction primitives (__reduce_add_sync) might.

---

## Recommended Path Forward

1. **K04 correlation**: Implement as single stacked FP16 matmul. Stack all cadences:
   `Z = normalize(X)  # [T_total × 4604]`
   `R = Z.T @ Z       # [4604 × 4604]`
   Expected: well under 1ms for full 31-cadence day via Tensor Cores.

2. **K01/K02 pipeline**: No Tensor Core opportunity. Continue optimizing segmented reduction
   kernels. Consider warp-level primitives, occupancy tuning.

3. **BF16 for K04**: Worth testing with PyTorch. BF16 is safe for normalized correlation inputs
   and may offer Tensor Core speedup without FP16 overflow risk.

4. **FP8**: Future. Blackwell supports FP8 natively. At ~1000 TFLOPS, K04 could run in ~0.02ms.
   Requires careful scaling and calibration. Not ready for production today.

---

## Campsite Coordinates

Previous scout notes in this directory:
- `integer-encoding-research.md` — encoding scheme per column, precision analysis
- `header-as-manifest.md` — upstream fingerprint design, reconcile algorithm
- `tensor-core-analysis.md` — this file
