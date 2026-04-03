# Paper 8: Five GPU Primitives for All of Mathematics

## Target
HPC/Systems: SC (Supercomputing), PPoPP, or ICS.

## Core Claim
Five GPU kernel types (scatter_multi_phi, DotProductOp, DistanceOp, AffineOp, FFT) implement 35 algorithm families (500+ algorithms). Nine families need ZERO GPU code. ~28,000 lines of Rust < one RAPIDS subpackage.

## Outline
1. The proliferation problem: cuBLAS + cuFFT + cuML + cuDNN + cuSOLVER = thousands of hand-tuned kernels
2. The five primitives: what each does, how each maps to accumulate grouping patterns
3. The nine zero-GPU families: pure CPU extraction from accumulated state
4. Multi-backend: same algorithm on CUDA, Vulkan, Metal, CPU — codegen branches on shader_lang()
5. The compilation budget: NVRTC dominates compute at data-science scales (5820x session speedup)
6. The column-partition mega-kernel: 11 accumulators × 31 cadences in 1% of Blackwell registers
7. Cross-platform verification: f64 on CUDA/CPU, f32 on WGSL, quantified divergence
8. Benchmarks: tambear vs RAPIDS vs scipy vs sklearn (fair comparison methodology)

## Evidence
- Navigator: multi-backend architecture, WgpuBackend, codegen module
- Pathmaker: all implementations, ComputeEngine, TiledEngine
- Observer: cross-platform parity tests
- Adversarial: f32/f64 precision analysis, compilation budget measurements
- Naturalist: 8-operator model, zero-GPU family census
