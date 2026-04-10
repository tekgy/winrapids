# E05 three-tier revision

Created: 2026-03-29T23:29:21-05:00
By: navigator

---

## Finding

E05 measured: JIT first compile = 40-92ms. Cache hit = 0.002ms. Disk caching (CUPY_CACHE_DIR) persists across process restarts. A JIT kernel that has been compiled once is identical to a pre-built kernel in steady-state.

Pre-built tier overhead: 100 kernels × 40ms = 4 seconds total startup. Same as "compile everything at first boot."

## What This Changes

The vision.md three-tier architecture is:
```
Pre-built: ~100 hand-tuned pipeline shapes (AOT)
JIT: generator produces optimal CUDA, NVRTC compiles (~40ms)
Fallback: individual specialists composed at runtime
```

The revised architecture based on E05:
```
Tier 1 — JIT with disk cache: compile on first use, cache forever
           First run: 40ms per kernel. All subsequent runs: 0.002ms.
           100 kernels → 4 seconds total first-boot warmup, then identical to pre-built.
Tier 2 — Composed specialists: fallback for novel patterns, 2.3x slower
           Use as prototype/debugging tier only.
```

Pre-built as a concept survives — it means "JIT on first use, cached." It does NOT mean "distribute compiled binaries." The disk cache IS the pre-built library.

## What This Does NOT Change

The Rust-side story may differ. E09 (NVRTC from Rust via cudarc) will measure whether Rust NVRTC has the same 40ms floor or something different. If cudarc's NVRTC is faster (no Python overhead), pre-compilation might recover value. Don't revise vision.md until E09 is measured.

Hand-tuning (occupancy optimization, specific algorithm selection for data shapes) is still valuable — that's separate from whether the compilation happens at install time vs first use.

## Action

Update vision.md's three-tier description after E09 confirms or contradicts this finding.

