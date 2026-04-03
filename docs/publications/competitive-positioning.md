# Competitive Positioning — Verified Claims

## DO NOT CLAIM (RAPIDS has these)
- GPU DBSCAN (cuML)
- GPU descriptive statistics (cuDF)
- GPU KMeans (cuML)
- GPU PCA (cuML)
- GPU linear regression (cuML)

## CLAIM WITH EVIDENCE (genuine differentiators)
1. FFT-based KDE at scale — cuML KDE is O(n²) pairwise, broken at 200K (PR #7833)
2. Windows native GPU — RAPIDS is Linux-only, hard wall
3. Cross-algorithm sharing via TamSession — no RAPIDS analog
4. Any-GPU (Vulkan/Metal/DX12) — RAPIDS is CUDA-only
5. Accumulate decomposition framework — no RAPIDS analog
6. 1B+ scale (IF MEASURED) — cuDF benchmarks stop at 5M

## CLAIM CAREFULLY
- "Faster than RAPIDS" — must benchmark head-to-head on same hardware
- "1B elements" — only after actual measured benchmark
- "Replaces cuBLAS/cuFFT" — true architecturally, need performance parity proof

## THE REAL DIFFERENTIATORS (no competitor has ANY of these)
1. The accumulate decomposition (Papers 1, 5)
2. The MSR principle (Paper 2)
3. Cross-algorithm sharing (Paper 3)
4. Gradient duality without autodiff (Paper 4)
5. 33 structural rhymes (Paper 5)
6. Numerical stability by construction (Paper 6)
7. Superposition architecture (Paper 7)
8. Multi-backend from .tbs (Paper 8)

Source: scout competitive research 2026-04-01, cuML docs, RAPIDS GitHub issues
