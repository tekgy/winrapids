# DGX Spark — Specs Research
## Scout Notes — 2026-03-27

Official spec sheet confirmed by Tekgy. Research agent filled in the gaps specs don't cover.

---

## Official Specs (Founders Edition, 4TB — Tekgy's unit)

```
Architecture:     NVIDIA Grace Blackwell
GPU:              Blackwell (GB10 SoC, SM12.1)
CPU:              20-core Arm (10x Cortex-X925 + 10x Cortex-A725)
CUDA Cores:       6,144 (48 SMs × 128 cores/SM — confirmed via deviceQuery)
Tensor Cores:     5th Generation
Tensor Perf:      Up to 1 PFLOP FP4 (= ~427 TFLOPS dense FP4, 2× sparsity to reach 1 PFLOP)
System Memory:    128 GB LPDDR5x, coherent unified CPU+GPU
Memory Interface: 256-bit, 273 GB/s bandwidth
Storage:          4 TB NVMe M.2 w/ self-encryption (PCIe Gen5)
USB:              4x USB Type-C (USB 3.2 Gen 2x2, 20 Gbps — NOT USB4/Thunderbolt)
Ethernet:         1x RJ-45, 10 GbE
NIC:              ConnectX-7, 200 Gbps (2x QSFP112 ports)
Wi-Fi:            Wi-Fi 7
Bluetooth:        BT 5.4
Display:          1x HDMI 2.1a
Power Supply:     240W
GB10 TDP:         140W
Dimensions:       150mm × 150mm × 50.5mm
Weight:           1.2 kg
OS:               NVIDIA DGX OS (Ubuntu 24.04 LTS base, kernel 6.17)
```

**"Foundation Edition" naming**: The official product is "Founders Edition." The name on Tekgy's
unit may be a regional OEM variant name or pre-release naming convention. Core specs appear
identical to the Founders Edition.

---

## Real-World Compute Performance

Official spec sheet omits actual TFLOPS by precision. Research fills this in:

| Precision | TFLOPS | Notes |
|-----------|--------|-------|
| FP4 (dense) | ~427 | 2× sparsity doubles to "1 PFLOP" in marketing |
| FP8 | ~213 | Dense, FP32 accumulator |
| FP16 | ~213 | Dense, Tensor Cores (5th gen) |
| BF16 | ~213 | Same Tensor Core path |
| FP32 | ~53 | CUDA cores, no Tensor Core acceleration |
| FP64 | ~0.8 | 1:64 ratio — consumer part, effectively absent |

**Comparison to our RTX PRO 6000 Blackwell workstation**:
- FP16/BF16 GEMM: Spark 213 TFLOPS vs RTX PRO 6000 288 TFLOPS (measured)
- FP32: Spark 53 TFLOPS vs RTX PRO 6000 54 TFLOPS (roughly equal)
- VRAM/unified: Spark 128 GB vs RTX PRO 6000 102.6 GB

**The Spark's GPU compute is NOT faster than our workstation GPU for GEMM.**
Its advantage is the 128 GB unified memory — not raw throughput.

### GB10 vs datacenter Blackwell — critical distinction
GB10 is SM12.1 (consumer Blackwell, same family as RTX 5090 at SM12.0).
Datacenter Blackwell (B100/B200) is SM10.0 — a fundamentally different architecture.

What GB10 LACKS vs datacenter Blackwell:
- No TMEM (Tensor Memory)
- No `tcgen05` instruction set
- No multi-SM cooperative MMA
- Only 128 KB shared memory/SM (datacenter has 228 KB)
- FlashAttention 4, FlashMLA: NOT compatible (need SM10x backends)
- FP8 block-scaled ops: NOT available

GB10 is roughly a "very large RTX 5070 Ti with 128 GB of shared memory."
The unified memory is the design win, not the GPU silicon itself.

---

## Network Interfaces — The Critical Section for WinRapids

### 10 GbE RJ-45 (the practical path)
One standard copper Ethernet port. Any Windows PC with a 10GbE NIC connects directly.
- Hardware needed: Intel X550-T1 or Realtek 10GbE NIC, ~$50–100, standard Cat6A cable
- Sustained transfer: **~700–900 MB/s** in practice
- Transfer time for 71 GB (4,604 tickers × 15.5 MB): **~80–100 seconds**

### ConnectX-7 @ 200 Gbps (2× QSFP112 ports)
Each physical QSFP port presents as 2 logical interfaces — ConnectX-7 attaches via 2× PCIe Gen5 x4.
Theoretical: ~200 Gbps. Measured with RoCE, both ports: **185–198 Gbps (~23 GB/s)**.
- Hardware needed on Windows: ConnectX-7 NIC + QSFP112 DAC cable
- ConnectX-7 cost: **~$1,500 market** (no cheap path)
- QSFP breakout cables: explicitly not validated by NVIDIA, flagged as potential issue
- Transfer time for 71 GB at 23 GB/s: **~3 seconds**

### Wi-Fi 7
No extra hardware on Windows (if on Wi-Fi 7 router). Theoretical ceiling 5.8 Gbps.
Realistic sustained: **1–2 Gbps**, ~35–70 seconds for 71 GB.
Higher latency, but surprisingly viable for bulk transfer if Wi-Fi 7 router is present.

### USB-C ports — NOT a network path
USB 3.2 Gen 2x2 (20 Gbps), NOT USB4. No Thunderbolt, no PCIe tunneling.
Attached NVMe appears as /dev/sda (USB mass storage), ~1.8 GB/s.
Can be used as "sneakernet" — copy to external NVMe, physically transfer, plug in.

---

## Software Stack

- **OS**: DGX OS 7.4.0 (Ubuntu 24.04 LTS), kernel 6.17
- **GPU driver**: 580.126.09, **open kernel module REQUIRED**
- **CUDA**: 13.0.2 (SM12x requires CUDA 13+)
- **Architecture**: aarch64 (ARM64)

### The CUDA 13 + aarch64 gap — real friction

The GB10 requires CUDA 13 (CUDA 12 doesn't support SM12x). Most pip-installable Python
ML packages ship CUDA 12 x86_64 wheels. The combo of cu130 + aarch64 is rare.

Package status:
| Package | Status |
|---------|--------|
| PyTorch | 2.9.0+ cu130+aarch64 wheels available. SM12x compatibility warning safe to ignore. |
| **CuPy** | **No confirmed cu130+aarch64 wheels on PyPI. Must build from source.** |
| vLLM | Nightly cu130 wheels at wheels.vllm.ai — works but fragile |
| flash-attn | No prebuilt wheel; compile from source, no SM12x optimization |
| TensorRT-LLM | DGX-specific build, works well |
| Triton | Works with TRITON_PTXAS_PATH set |
| Transformers | Works with CUDA 13 PyTorch |

**For WinRapids specifically**: CuPy (our primary GPU toolkit) has no ready-to-install
package. Building from source is required. This is a real setup cost for any workflow
that uses CuPy on the Spark.

---

## Evaluation for WinRapids Use Case

**The question**: Is DGX Spark a good batch compute node for model training and K04 cross-ticker
correlation, with data living on the workstation's NVMe?

### Network bottleneck analysis

Full universe dataset: 4,604 tickers × ~15.5 MB = **~71 GB**

| Connection | Transfer rate | Time for 71 GB | Cost |
|------------|--------------|----------------|------|
| 10 GbE | 700–900 MB/s | **80–100 seconds** | ~$75 NIC |
| Wi-Fi 7 | 1–2 GB/s | 35–70 seconds | ~$0 (if router exists) |
| ConnectX-7 | ~23 GB/s | **~3 seconds** | ~$1,500 NIC |

For daily workflows transferring the full universe, 80–100 seconds over 10 GbE is
probably acceptable if the Spark then spends minutes computing. Less acceptable for
frequent small jobs.

**Key insight**: if the 4 TB local NVMe stores a working copy of the dataset, transfer
only happens once per update cycle (not per job). This changes the calculus entirely.

### Where Spark wins vs. local RTX PRO 6000

1. **Memory capacity**: 128 GB unified vs 102.6 GB VRAM. For K04 with all tickers:
   - Full universe × full feature set may approach or exceed 102.6 GB
   - Spark can hold everything in unified memory with no paging
   - RTX PRO 6000 is close but may need to process in chunks for very large matrices

2. **Model training**: Large models that don't fit in 102.6 GB VRAM fit in 128 GB unified.
   No VRAM spill to system RAM (the killer for training).

3. **Dedicated compute node**: Offloads batch work from the workstation, keeping it
   available for interactive work.

### Where RTX PRO 6000 wins vs. Spark for our workload

1. **Raw GEMM throughput**: 288 TFLOPS FP16 vs 213 TFLOPS — workstation is faster for K04
2. **GPU bandwidth**: 1792 GB/s vs 273 GB/s — workstation is 6.5× faster for memory-bound ops
3. **CuPy availability**: Works out of box vs. build-from-source on Spark
4. **No data transfer needed**: Data lives locally on workstation NVMe
5. **FP64**: 1.4 TFLOPS vs ~0.8 TFLOPS (minor but workstation wins)

### Summary verdict

The Spark is a **memory-capacity node**, not a raw-throughput node.

It makes most sense as:
- Batch inference / model eval server (large models that need 128 GB)
- Jobs that run once and leave data on the Spark's local NVMe (not daily ETL)
- Future multi-node scale-out (if ConnectX-7 investment is made)

It does NOT replace the RTX PRO 6000 for day-to-day K01/K02/K04 pipeline work.
Bandwidth mismatch (273 GB/s vs 1792 GB/s) alone makes it slower for reduction-heavy kernels.

The most valuable use case pairing:
- **Workstation RTX PRO 6000**: K01/K02 (binning, pointwise), daily fresh data from local NVMe
- **DGX Spark**: Large model training, K04 correlation over full universe when data is pre-staged,
  inference serving for signal consumption

---

## Open Questions

1. What is the actual NVMe sequential read/write on the 4 TB PCIe Gen5 drive?
   (PCIe Gen5 implies ~12–14 GB/s; not yet benchmarked in public reviews)
2. Exact CuPy build status for cu130+aarch64 — is there a nightly wheel or must we maintain
   a build?
3. For K04 stacked-matrix approach: does the GB10 Tensor Core path perform differently
   than our Blackwell workstation for the [142724×4604] matmul?
4. Can the Spark's 4 TB NVMe act as a secondary MKTF store, with data pushed from workstation
   via 10 GbE on a daily basis?
