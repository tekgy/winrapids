# MKTF Format Research: Cross-Domain Expedition Log

**Naturalist: Cross-domain format survey**
**Date: 2026-03-27**

---

## Mission

How do other domains store their "hot data" — data that must move from storage to GPU (or FPGA, or specialized hardware) as fast as possible? What patterns translate to GPU-native market data?

Surveyed: **genomics** (VCF/BCF/SeqArray), **game engines** (Nanite, virtual textures, megatexture), **HFT firmware** (ITCH protocol, FPGA parsers), **satellite imagery** (Cloud Optimized GeoTIFF, Zarr, TileDB), **radio astronomy** (FITS, filterbank), **neuroimaging** (NIfTI/CIFTI). Plus: Windows DirectStorage/GDeflate, Lance format.

---

## Domain Surveys

### 1. Genomics: VCF/BCF/SeqArray

**The problem**: Whole-genome variant data for thousands of samples. Billions of genotype calls. Row-wise VCF format is fundamentally unsuitable for columnar access.

**BCF** (Binary VCF): BGZF-compressed binary encoding of VCF. Row-oriented — efficient for "give me all data for one variant" but terrible for "give me one field across all variants." Same problem as Parquet row groups for our use case.

**SeqArray** (the breakthrough): Columnar decomposition of VCF into a hierarchical GDS container.
- **Results**: 5.4x smaller than VCF (2.6 GB vs 14 GB for 1000 Genomes Phase 3)
- **Speed**: 2-3x faster than BCF via htslib C library for genotype reads; 16x faster than vcftools for allele frequency
- **Key insight**: Information density of 14.1 genotypes/bit with LZMA compression
- **Architecture**: Each field stored as a separate array. Hierarchical container with array-level metadata.

**VCF-Zarr** (emerging): Converting VCF to Zarr chunked arrays for GPU-accelerated processing. The genomics community is discovering what satellite/climate people already knew: chunked N-dimensional arrays are the natural shape for columnar scientific data.

**What to steal for MKTF**:
- Columnar decomposition is not optional — it's the single biggest performance lever
- Per-column compression choices (genotypes compress differently than quality scores, just as timestamps compress differently than prices)
- The SeqArray pattern of "hierarchical container with typed arrays" is exactly a column directory

### 2. Game Engines: Nanite + Virtual Textures

**The problem**: Stream billions of triangles / thousands of gigapixels of texture from NVMe to GPU, on demand, at 60+ FPS.

**Nanite (Unreal Engine 5)**:
- Meshes decomposed into ~128-triangle **clusters**
- Clusters organized into hierarchical LOD tree (always GPU-resident: just bounds + parent/child links)
- Clusters grouped into **streaming pages** (the I/O unit)
- Pages stored **compressed on disk**, decompressed on GPU
- Vertex attributes: **quantized and bit-packed** in on-disk format, transcoded to GPU-native format during stream-in
- **Visibility buffer**: Per-pixel cluster IDs. Only load what's visible. The indirection table IS the format.

**Virtual Textures / Megatexture (id Software)**:
- Single massive texture tiled into 128x128 pixel tiles
- **Indirection table**: Maps virtual texture coordinates to physical tile locations
- Hardware-accelerated via Partially Resident Textures / Tiled Resources (DX11.2+)
- Only tiles visible to the camera are resident in GPU memory
- The tile IS the unit of I/O, caching, and eviction

**What to steal for MKTF**:
- **Page-based I/O**: The unit of storage should match the unit of transfer. Don't read a 50MB file to get one column.
- **Quantization is compression**: Nanite bit-packs vertex positions to the minimum precision needed. We can do the same — float32 for derived features that don't need float64 precision.
- **Hierarchical directory always resident**: Small metadata (cluster hierarchy / column directory) stays in memory; bulk data streams on demand.
- **Two representations**: On-disk (compressed/packed) and in-memory (GPU-native). The transcode step should be trivially fast or GPU-accelerated.
- At our current scale (~46MB per ticker-day), the whole file IS one page. But for multi-ticker or multi-day files, the page concept becomes important.

### 3. HFT Firmware: ITCH Protocol + FPGA Parsers

**The problem**: Parse market data feed at line rate (10+ Gbit/s) with sub-microsecond latency. Every nanosecond of parsing overhead = money lost.

**ITCH protocol design**:
- **Fixed-length messages**: Each message type has a known, fixed byte length
- **Fixed-width fields**: Every field at a known offset within its message type
- **First byte = message type**: Single byte dispatch, zero ambiguity
- **Big-endian binary integers**: No text parsing, no delimiters, no escaping
- **Implied precision for prices**: Integer field + precision metadata → no floating point on the wire
- **No framing beyond message boundaries**: Messages are self-contained units

**FPGA parsers**:
- Message description → **hardware decoder generated automatically** (compiler from spec to Verilog)
- Sub-25ns parsing latency for NASDAQ ITCH
- Key insight: **the format IS the parser**. When fields are at fixed offsets, "parsing" is just pointer arithmetic. The hardware reads bytes at known positions — there is no parse step.

**What to steal for MKTF**:
- **The format IS the in-memory representation**. Zero deserialization. mmap the file → cast to struct → done.
- **Fixed offsets are everything**. The column directory tells you byte offset + dtype for each column. Reading column N means: seek to offset, read N_rows * sizeof(dtype) bytes. That's it.
- **First bytes = dispatch**: Magic + version in first 8 bytes. A reader knows immediately what it's dealing with.
- **Implied precision**: Store prices as integer ticks with a scale factor in the header, rather than float64. Smaller, no floating-point ambiguity, GPU integer ops.
- The **compiler-from-spec-to-parser** idea: MKTF's column directory IS a machine-readable spec. A GPU kernel can be generated from it.

### 4. Satellite Imagery: Cloud Optimized GeoTIFF + Zarr + TileDB

**The problem**: Petabytes of Earth observation imagery. Must support range requests (HTTP or filesystem seek) to read tiles without downloading whole files. Must feed into GPU-accelerated ML pipelines.

**Cloud Optimized GeoTIFF (COG)**:
- Standard GeoTIFF but reorganized: **all metadata first, tiled data after**
- Supports pyramidal overviews (multi-resolution)
- Key insight: **metadata placement determines random-access performance**. COG puts the IFD (Image File Directory) at the start, not the end. One small read tells you where every tile lives.

**Zarr v3**:
- N-dimensional arrays chunked into regular pieces
- Each chunk independently readable and compressible
- **Sharding codec (ZEP-2)**: Groups multiple chunks into one storage object ("shard") with an **offset index at the end of the shard**
- Codec pipeline: array → bytes → compressed bytes (pluggable at each stage)
- The chunk IS the unit of parallelism. Each chunk → one GPU thread block.

**TileDB**:
- Columnar format: each attribute stored in a separate file
- Dense arrays: space tile = data tile (no sparse overhead)
- Variable-length attributes: separate offset file (sorted → highly compressible)
- Performance: Random subarray reads scale independently of total array size (80ms for 1K×1K regardless of whether total array is 6GB or 600GB)

**What to steal for MKTF**:
- **Metadata first** (COG pattern): Column directory at byte 0. One read to know the whole layout.
- **Zarr sharding**: For multi-symbol files, each symbol's data = one "shard" with its own offset index. Enables reading AAPL without touching MSFT's bytes.
- **TileDB's attribute separation**: Separate storage per column maximizes column-selective I/O. But for our case, a single file with an offset directory achieves the same thing with less filesystem overhead.
- **The chunk = the transfer unit**: When reading a column, the transfer to GPU should be one contiguous DMA operation per column.

### 5. Radio Astronomy: FITS + Filterbank

**The problem**: Terabytes of spectral/temporal data from radio telescopes. Must process in real-time or near-real-time, increasingly on GPUs.

**FITS format**:
- **2880-byte block boundary**: Every header and data unit is an exact multiple of 2880 bytes. Chosen for compatibility with 1979-era tape drives. Now a historical artifact, but the PRINCIPLE is sound: alignment to hardware I/O boundaries.
- **Header = ASCII keyword/value pairs**: Human-readable metadata, machine-parseable. 80-character fixed-length "cards."
- **Binary Table Extensions**: Tabular data in binary representation. Columns can be N-dimensional arrays (fixed or variable length).
- **HDU structure**: Header Data Unit = header + data. Multiple HDUs per file.
- **Endianness**: Big-endian (network byte order). Historical choice, costs nothing on modern hardware that can byte-swap in the load path.

**Filterbank format**:
- Regularly sampled data with **implicit time axis** — header specifies t_start, t_step, n_samples. No explicit timestamp column needed.
- The entire time axis is derived from three numbers. This is the ultimate delta encoding.
- Spectral channels stored as contiguous arrays — natural for GPU FFT pipelines.

**What to steal for MKTF**:
- **Implicit time axis**: For regularly-sampled data (e.g., 1-second bins), store t_start + t_step + n_rows instead of a full timestamp column. Saves 8 bytes/row. For irregular tick data, use delta encoding from a base timestamp.
- **Alignment to I/O boundaries**: FITS uses 2880 bytes (tape era). We should use 64 bytes (GPU cache line) or 4096 bytes (NVMe sector). Column data starting at 64-byte-aligned offsets enables optimal DMA.
- **The HDU pattern**: Header + Data as a unit that can be independently read. Each column in MKTF is essentially a mini-HDU: its directory entry (header) + its data bytes.
- **Fixed-width ASCII headers** were brilliant for their era. Our equivalent: a fixed-size binary header parseable in a single struct read.

### 6. Neuroimaging: NIfTI / CIFTI

**The problem**: 3D/4D brain imaging data (spatial + temporal). Must feed into GPU-accelerated statistical analysis and machine learning.

**NIfTI-1**:
- **352-byte fixed header** — C struct, one read, one cast, done
- Data immediately follows header (or in separate .img file)
- Header contains: dimensions, data type, voxel size, orientation, scaling
- **Scaling slope/intercept in header**: Raw integer voxels → real values via `value = slope * raw + intercept`. Data stays in compact integer format; interpretation is in metadata.
- File = header + contiguous 3D/4D array. **No internal structure beyond the header**.

**NIfTI-2**:
- 540-byte header (larger fields for bigger datasets)
- Same principle: fixed header → raw data

**CIFTI**:
- Built on NIfTI-2 but adds XML-based metadata for mapping between brain structures
- Single rectangular data matrix (usually 2D)
- Each dimension described by one of five "mapping types"
- Handles heterogeneous structures (cortical surfaces + subcortical volumes) in one file

**What to steal for MKTF**:
- **The 352-byte fixed header is the gold standard for simplicity**. One read, one cast to a C struct, all metadata available. No JSON parsing, no variable-length anything in the header.
- **Scaling slope/intercept**: Store raw data in compact integer types, with per-column scaling factors in the directory. E.g., prices as int32 with `price = raw * scale + offset`. Halves the column size vs float64.
- **File = header + raw data, nothing else**. The simplest format that could possibly work. Our MKTF v1 should be this simple.
- **CIFTI's mapping types**: The column directory entries in MKTF are analogous — each column has a "type" that tells you how to interpret the raw bytes.

---

## Synthesized Patterns: What Translates to GPU-Native Market Data

### Pattern 1: "The Anatomy" — Fixed Header + Directory + Raw Data

Every high-performance format follows this structure:
```
[Fixed Header] → [Directory/Index] → [Contiguous Data]
```

| Domain | Header | Directory | Data |
|--------|--------|-----------|------|
| NIfTI | 352-byte C struct | (implicit — single array) | Contiguous 3D/4D array |
| FITS | 2880-byte ASCII cards | Keywords specify dimensions | Contiguous block |
| ITCH | 1-byte message type | (fixed offsets per type) | Fixed-width fields |
| COG | IFD at file start | Tile offsets in IFD | Tiled raster |
| Lance | Footer manifest | Fragment + column offsets | Column chunks |

**MKTF v1**: Fixed 512-byte header (magic, version, n_rows, n_cols, flags) → column directory entries (name, dtype, offset, nbytes, compression, scale/intercept) → raw column data at 64-byte-aligned offsets.

### Pattern 2: "Zero Deserialization" — The Format IS the Memory Layout

This is the ITCH principle: if the on-disk bytes ARE the in-memory representation, "reading" is just mapping.

| Domain | How they achieve it |
|--------|-------------------|
| ITCH | Fixed-width fields at fixed offsets. Pointer arithmetic = parsing. |
| NIfTI | mmap file, cast header pointer, data pointer = header + 352 |
| Raw binary (ours) | Column at offset, dtype known → `np.frombuffer` or `cudaMemcpy` |
| Nanite | GPU-native quantized format (in-memory representation optimized for GPU) |

**MKTF v1**: Column data stored in native little-endian format matching numpy/CUDA dtype. Reading column X:
1. Read 512-byte header (or mmap entire file)
2. Look up column X in directory → offset, nbytes, dtype
3. `cudaMemcpy(d_ptr, file_ptr + offset, nbytes)` — or GPUDirect: DMA directly from NVMe

**Zero parsing. Zero conversion. Zero allocation beyond the destination buffer.**

### Pattern 3: "64-Byte Alignment" — DMA Lane Optimization

The magic number is **64 bytes** — GPU cache line width on NVIDIA hardware. Secondary alignment: 4096 bytes (NVMe sector / OS page size).

| Domain | Alignment |
|--------|-----------|
| Arrow IPC | 64-byte alignment for SIMD |
| GDeflate | 64 KiB tiles for GPU decompression parallelism |
| FITS | 2880-byte blocks (historical) |
| GPUDirect Storage | Byte-aligned but optimal at page boundaries |

**MKTF v1**: Every column's data starts at a 64-byte-aligned file offset. Padding bytes between columns are wasted but trivial (at most 63 bytes per column gap × 10 columns = 630 bytes overhead on a 46MB file = 0.001%).

### Pattern 4: "Implied Axes" — Don't Store What You Can Derive

Radio astronomy's filterbank format stores `(t_start, t_step, n_samples)` instead of a timestamp array. NIfTI stores voxel dimensions + affine transform instead of explicit coordinates.

**MKTF v1**: For regularly-sampled data:
- Timestamps: `t_start + t_step * index` (3 header fields replace 8 bytes × N rows)
- For irregular ticks: `t_base + delta_array` where deltas are int32 (saves 4 bytes/row vs int64)

Savings on 598K ticks: 598,057 × 4 bytes = **2.3 MB saved** on timestamps alone.

### Pattern 5: "Scaling Slope/Intercept" — Compact Storage, Rich Interpretation

NIfTI's `scl_slope` and `scl_inter` fields let you store int16 voxels but interpret them as float64 values. The data stays small; the metadata carries the precision.

**MKTF v1**: Per-column `scale` and `offset` fields in the directory:
- Prices: `int32 × 1e-6 + 0` → 4 bytes/row instead of 8 bytes/row for float64
- Sizes: `uint32` (trade sizes don't need float64)
- Derived features (ln_price, sqrt_price): `float32` is plenty → 4 bytes/row

Savings: Converting 7 float64 columns to float32 or scaled int32 saves ~598K × 7 × 4 = **16.7 MB** on a 46MB file = **36% reduction**.

### Pattern 6: "Bitmask Packing" — Boolean Fields as Bit Vectors

BCF packs genotype data as bit vectors. Our existing design already packs 26 boolean conditions into uint32.

**MKTF v1**: Condition flags column: `uint32` bitmask instead of 26 × `uint8`.
- Saves 26 - 4 = 22 bytes per tick × 598K = **13.2 MB**
- GPU bitwise operations (`&`, `|`, `popcount`) are single-cycle

### Pattern 7: "Sharding" — Multiple Logical Units in One Physical File

Zarr v3 sharding, Nanite streaming pages, Lance fragments: group related data into one file to reduce filesystem overhead.

**MKTF v1**: One file per ticker-day (current approach) is fine at 4,604 tickers.
**MKTF v2** (future): Shard file containing all tickers for one day, with a top-level directory mapping ticker → offset within the file. Reduces 4,604 files to 1 file per day. But: benchmark first — Windows NTFS handles many files differently than ext4.

### Pattern 8: "GPU Decompression Pipeline" — DirectStorage + GDeflate

The game engine world has solved NVMe → GPU with hardware-accelerated decompression. Windows DirectStorage + GDeflate on our RTX 6000 Pro Blackwell.

**MKTF v1**: No compression initially — raw binary is already 4x faster than Parquet. Compression adds latency that may exceed savings from smaller I/O.
**MKTF v2** (if needed): GDeflate-compressed columns in 64 KiB tiles, loaded via DirectStorage API with GPU decompression. This is the Nanite path — compressed on disk, native in VRAM. But only worth it if I/O bound (unlikely on NVMe for 46MB files).

---

## The MKTF Design Principles (Derived from Cross-Domain Survey)

1. **The file IS the memory layout.** Zero deserialization. mmap or DMA directly.
2. **Fixed header, first read tells all.** Column directory in first 512-4096 bytes.
3. **64-byte alignment for every column.** GPU DMA optimal.
4. **Columnar storage is non-negotiable.** SeqArray proved 16x. Every domain confirms.
5. **Don't store what you can derive.** Implicit timestamps, scaling factors, bitmasks.
6. **Match the hardware's natural unit.** 64 bytes (cache line), 4096 bytes (NVMe sector), 64 KiB (GDeflate tile).
7. **The simplest format that could possibly work.** NIfTI's 352-byte header serves petabytes of neuroscience. Complexity is a bug.
8. **Compression is a v2 feature.** Raw binary at 9ms read is already fast. GDeflate path exists when we need it.
9. **The column directory is the API.** It's machine-readable, GPU-readable, human-inspectable.
10. **Build for the RTX 6000 Pro Blackwell, not for the general case.** Every format above was shaped by its target hardware. Ours should be too.

---

## Recommended MKTF v1 Layout

```
Bytes 0-3:     Magic "MKTF" (4 bytes)
Bytes 4-5:     Version uint16 (2 bytes)
Bytes 6-7:     Flags uint16 (2 bytes) — bit 0: has_implicit_timestamps
Bytes 8-15:    n_rows uint64 (8 bytes)
Bytes 16-17:   n_cols uint16 (2 bytes)
Bytes 18-25:   t_start int64 (8 bytes) — nanosecond epoch
Bytes 26-33:   t_step int64 (8 bytes) — 0 if irregular
Bytes 34-63:   Reserved/padding to 64 bytes

Bytes 64-...:  Column Directory (64 bytes per entry, n_cols entries)
  Per entry:
    Bytes 0-31:   Column name (32 bytes, null-padded ASCII)
    Bytes 32-32:  dtype_code uint8
    Bytes 33-33:  compression_code uint8 (0=none, 1=gdeflate, 2=lz4)
    Bytes 34-41:  offset uint64 — byte offset from file start
    Bytes 42-49:  nbytes uint64
    Bytes 50-57:  scale float64 (1.0 if raw)
    Bytes 58-63:  Reserved (6 bytes)

Bytes ceil64(64 + 64*n_cols) ... EOF:  Column data
  Each column starts at a 64-byte-aligned offset.
  Data is in native little-endian format matching dtype_code.
  No padding between rows within a column.
  Padding (zeros) between columns to maintain alignment.
```

Total header overhead for 10 columns: 64 + 64×10 = **704 bytes** (0.0015% of a 46MB file).

The entire directory fits in 11 cache lines. Readable in one NVMe sector read.

---

## Sources

### Genomics
- [SeqArray — columnar WGS variant format](https://academic.oup.com/bioinformatics/article/33/15/2251/3072873)
- [VCF-Zarr — Zarr-based VCF at biobank scale](https://www.biorxiv.org/content/10.1101/2024.06.11.598241v3)
- [VCF and BCF overview](https://evomics.org/vcf-and-bcf/)

### Game Engines
- [Nanite Virtualized Geometry (Epic)](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-virtualized-geometry-in-unreal-engine)
- [Nanite technical breakdown](https://medium.com/@GroundZer0/nanite-epics-practical-implementation-of-virtualized-geometry-e6a9281e7f52)
- [Nanite streaming and memory budgets](https://medium.com/@GroundZer0/nanite-streaming-and-memory-budgets-managing-geometry-at-scale-4c54bfa5d5b1)
- [Nanite in WebGPU (open source)](https://github.com/Scthe/nanite-webgpu)
- [Virtual texturing overview](https://playerunknownproductions.net/news/virtual-texturing)
- [Megatexture history](https://pixelbear.dev/blog/Mega-Textures/)

### HFT / FPGA
- [NASDAQ TotalView-ITCH 5.0 spec (PDF)](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf)
- [Sub-25ns FPGA ITCH parser](https://github.com/mbattyani/sub-25-ns-nasdaq-itch-fpga-parser)
- [ITCH protocol overview (Databento)](https://databento.com/microstructure/itch)
- [FPGA in HFT (Velvetech)](https://velvetech.com/blog/fpga-in-high-frequency-trading/)

### Satellite Imagery
- [Cloud Optimized GeoTIFF spec](https://cogeo.org/)
- [Zarr v3 core specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
- [Zarr sharding codec (ZEP-2)](https://zarr.dev/zeps/accepted/ZEP0002.html)
- [Cloud-native geospatial formats explained](https://forrest.nyc/cloud-native-geospatial-formats-geoparquet-zarr-cog-and-pmtiles-explained/)
- [TileDB deep dive](https://www.tiledb.com/blog/a-deep-dive-into-the-tiledb-data-format-storage-engine)

### Radio Astronomy
- [FITS primer (NASA)](https://fits.gsfc.nasa.gov/fits_primer.html)
- [FITS standard v4.0 (PDF)](https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf)
- [FITS overview (HEASARC)](https://heasarc.gsfc.nasa.gov/docs/heasarc/fits_overview.html)

### Neuroimaging
- [NIfTI official site](https://nifti.nimh.nih.gov/)
- [NIfTI file format explained](https://brainder.org/2012/09/23/the-nifti-file-format/)
- [NiBabel NIfTI documentation](https://nipy.org/nibabel/nifti_images.html)
- [CIFTI/BALSA file types](https://balsa.wustl.edu/about/fileTypes)

### GPU / DirectStorage
- [GPUDirect Storage design guide (NVIDIA)](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html)
- [DirectStorage API (Microsoft)](https://github.com/microsoft/DirectStorage)
- [GDeflate for DirectStorage (NVIDIA blog)](https://developer.nvidia.com/blog/accelerating-load-times-for-directx-games-and-apps-with-gdeflate-for-directstorage/)
- [GDeflate in nvCOMP](https://docs.nvidia.com/cuda/nvcomp/gdeflate.html)

### Other Formats
- [Lance columnar format](https://lancedb.com/docs/overview/lance/)
- [Lance GitHub](https://github.com/lance-format/lance)

---

# Part 2: Integer Encoding & The FP64 Catastrophe

**Naturalist: Precision/throughput analysis**
**Date: 2026-03-27 (afternoon session)**

---

## The Discovery: FP64 is 64x Slower on Our Hardware

The RTX 6000 Pro Blackwell (GB202) has this throughput hierarchy:

| Type | Peak Throughput | Relative to FP32 |
|------|----------------|-------------------|
| **FP64** | **~1.95 TFLOPS** | **1/64x** |
| FP32 | ~125 TFLOPS | 1x (baseline) |
| INT32 | ~125 TFLOPS | 1x (shares unified pipeline) |
| FP16/BF16 | ~250 TFLOPS (Tensor Cores) | 2x |
| INT8 | ~500+ TFLOPS (Tensor Cores) | 4-16x |

The GB202 has only **2 FP64 cores per SM** (384 total) vs **128 FP32/INT32 cores per SM** (24,064 total). FP64 on consumer/prosumer GPUs exists to ensure correctness, not performance. The 384 FP64 cores are there so double-precision code *runs*, not so it runs *fast*.

**Our fused pointwise kernel uses `double` for everything.** Every `log()`, `sqrt()`, `sin()`, `cos()` call in `fused_pointwise()` is running at 1/64th of the GPU's capability.

### Transcendentals Are Even Worse Than 64x

FP32 transcendental functions (`logf`, `sqrtf`, `sinf`, `cosf`) execute on the **Special Function Unit (SFU)** — dedicated hardware that produces 16 results per clock per SM. FP64 transcendentals must be **emulated** using sequences of FP64 multiply-add instructions on the 2 FP64 ALUs per SM.

| Operation | FP32 (SFU) | FP64 (emulated) | Speedup |
|-----------|-----------|-----------------|---------|
| logf/log | 16/clock/SM | ~1-2/clock/SM | **8-16x** |
| sqrtf/sqrt | 16/clock/SM | ~1-2/clock/SM | **8-16x** |
| sinf/sin | 16/clock/SM | multi-instruction | **8-16x** |
| cosf/cos | 16/clock/SM | multi-instruction | **8-16x** |

The fused kernel computes: `log()` × 3, `sqrt()` × 1, `sin()` × 1, `cos()` × 1. That's 6 transcendental calls per thread, all at 1/8th to 1/16th throughput. Switching to float32 makes transcendentals 8-16x faster via SFU, AND all arithmetic 64x faster via the FP32 pipeline.

**Combined effect**: A kernel that's 50% transcendentals + 50% arithmetic could see roughly **30-40x speedup** from FP64 → FP32, even before considering memory bandwidth savings.

### Blackwell's Unified INT32/FP32 Pipeline

Critical architectural nuance: **Blackwell unified the INT32 and FP32 datapaths into a single execution cluster.** Only one type — FP32 or INT32 — can decode and execute per cycle. This means:

- INT32 does NOT give a throughput advantage over FP32
- Mixed INT32/FP32 instruction chains incur a 4-6x latency penalty (pipeline switching)
- Previous architectures (Turing, Ampere, Hopper) had separate INT32 and FP32 pipes that could dual-issue

**Implication for MKTF**: Storing data as int32 and computing in float32 is fine — the `int2float` conversion is a single instruction in the FP32 pipeline. But we should NOT design kernels that interleave int32 arithmetic and float32 arithmetic. Convert early, compute in float32, convert back late.

---

## Precision Analysis: What Does Market Data Actually Need?

### Price Data

| Representation | Bytes | Max Value | Decimal Precision | Exact? |
|----------------|-------|-----------|-------------------|--------|
| float64 | 8 | ~1.8×10^308 | ~15.9 digits | No (binary fractions) |
| float32 | 4 | ~3.4×10^38 | ~7.2 digits | No (binary fractions) |
| int32 × 10^-4 | 4 | $214,748.3647 | 4 decimal places | **Yes** |
| uint32 × 10^-4 | 4 | $429,496.7295 | 4 decimal places | **Yes** |
| int32 × 10^-2 | 4 | $21,474,836.47 | 2 decimal places | **Yes** |
| int64 × 10^-8 | 8 | $92,233,720,368.54 | 8 decimal places | **Yes** |
| int64 × 10^-4 | 8 | $922,337,203,685,477.58 | 4 decimal places | **Yes** |

**Key facts:**
- US equities have minimum tick size of $0.01 (penny) or $0.0001 (sub-penny for some venues)
- FINRA standardizes to 6 decimal places for price reporting
- ITCH protocol uses integer prices with implied 4-decimal precision
- Crypto can have 8+ decimal places
- BRK.A at ~$700,000 exceeds uint32 × 10^-4 (max $429K) but fits int32 × 10^-2

**Recommendation**: `int32 × 10^-4` (4 implied decimals) for 99.97% of securities. For BRK.A and crypto edge cases, flag in header to use `int64 × 10^-8`.

float64 is the WORST representation: 8 bytes, inexact binary fractions (`0.1` can't be represented exactly), AND 64x slower compute. Integer fixed-point is smaller, exact, and the same speed (after cast to float32 for computation).

### Trade Size

| Representation | Bytes | Max Value |
|----------------|-------|-----------|
| float64 | 8 | Overkill |
| uint32 | 4 | 4,294,967,295 shares |
| uint16 | 2 | 65,535 shares |

Trade sizes are whole numbers. Storing as float64 is absurd — uint32 handles every realistic trade size. Most trades are < 65,535 shares → uint16 would work for ~99.9% of ticks.

**Recommendation**: `uint32` (4 bytes). Lossless, covers all cases, half the current size.

### Timestamps

| Representation | Bytes | Range / Precision |
|----------------|-------|-------------------|
| int64 nanoseconds | 8 | Full epoch, nanosecond precision |
| int32 ms from day start | 4 | 24.8 days at millisecond precision |
| Implicit (header only) | 0 | Regular intervals only |

For irregular tick data with nanosecond precision, int64 is unavoidable. But consider:
- If timestamps are regular (e.g., 1-second bins): `t_start + t_step × i` — **zero bytes per row**
- If microsecond precision is enough: int32 ms from t_base covers ~24 days
- Delta encoding: first tick as int64, subsequent as int32 microsecond deltas (max gap ~35 minutes between ticks)

**Recommendation**: int64 nanoseconds for K01 tick data (can't safely compress). Flag for K02 binned data to use implicit timestamps (zero-cost).

### Condition Flags (26 booleans)

| Representation | Bytes per tick | Total for 598K ticks |
|----------------|---------------|---------------------|
| 26 × uint8 | 26 | 15.5 MB |
| uint32 bitmask | 4 | 2.4 MB |
| **Savings** | **22** | **13.1 MB** |

GPU bitwise ops (`&`, `|`, `>>`, `popcount`) are single-cycle. Extracting bit N from a uint32 costs one AND + one shift.

### Derived Columns: The Nuclear Option

This is the biggest insight: **don't store derived columns AT ALL**.

The fused pointwise kernel computes 13+ outputs (notional, ln_price, ln_size, ln_notional, sqrt_price, recip_price, elapsed, sin_time, cos_time, round_lot, odd_lot, sub_penny, round_price) from 3 inputs (price, size, timestamp) in **0.1ms on GPU**.

Reading ONE float64 column from NVMe: 598K × 8 bytes = 4.8 MB at ~5 GB/s = **~0.96ms**.

| Approach | Time | Storage |
|----------|------|---------|
| Store 13 derived columns, read from disk | 13 × 0.96ms = **12.5ms** | 62 MB |
| Store 0 derived columns, recompute on GPU | **0.1ms** | 0 MB |
| **Savings** | **125x faster** | **62 MB** |

GPU recomputation is 125x faster than reading stored results. And this is with FP64 compute — with FP32, recomputation would be ~0.003ms.

**The file should store ONLY irreducible source data**: price, size, timestamp, exchange, condition_bitmask. Everything else is derived faster than it can be read.

---

## The Optimal MKTF Data Budget

For 598,057 ticks:

| Column | Type | Bytes/tick | Total | Notes |
|--------|------|-----------|-------|-------|
| price | int32 (×10^-4) | 4 | 2.39 MB | Exact to 4 decimals |
| size | uint32 | 4 | 2.39 MB | Covers all trade sizes |
| timestamp | int64 (ns) | 8 | 4.78 MB | Nanosecond precision |
| conditions | uint32 (bitmask) | 4 | 2.39 MB | 26 flags + 6 spare |
| exchange | uint8 | 1 | 0.60 MB | ~20 exchanges |
| **Total** | | **21** | **12.55 MB** | |

vs. Current (10 columns, mostly float64): **47.8 MB**

**3.8x smaller. Zero information loss. Zero compression.**

With 64-byte alignment padding: ~12.56 MB. Header + directory overhead: ~384 bytes.

### Universe-Scale Impact

| Metric | Current (47.8 MB/ticker) | Proposed (12.6 MB/ticker) |
|--------|--------------------------|--------------------------|
| Per ticker-day | 47.8 MB | 12.6 MB |
| Daily (4,604 tickers) | 220 GB | 58 GB |
| Annual (250 days) | 55 TB | 14.5 TB |
| 5-year backtest | 275 TB | 72.5 TB |
| Read time per ticker (NVMe ~5 GB/s) | ~9.6ms | ~2.5ms |
| H2D per ticker (PCIe 5 ~64 GB/s) | ~0.75ms | ~0.20ms |
| Full universe read | ~44s | ~11.5s |

**Savings: 200+ TB over a 5-year backtest. 32+ seconds per universe pass.**

---

## Encoding Strategies: From Conservative to Radical

### Strategy A: "Kill FP64" (Conservative, Immediate Win)

Change: Store everything as float32 instead of float64. Compute in float32.

- File size: 47.8 MB → ~28 MB (42% reduction)
- GPU compute: 64x faster (FP32 vs FP64 pipeline)
- Precision: ~7 decimal digits (sufficient for all signal detection)
- Effort: Change `double` to `float` in CUDA kernels, `float64` to `float32` in numpy
- Risk: Near-zero. float32 is standard for ML/signal processing.

### Strategy B: "Integer Source, Float Compute" (Recommended)

Change: Store source data as integers (int32 prices, uint32 sizes, uint32 bitmasks). Don't store derived columns. GPU kernel casts int→float32 on load, computes in float32.

- File size: 47.8 MB → ~12.6 MB (74% reduction)
- GPU compute: 64x faster + 125x less I/O for derived data
- Precision: Exact storage, ~7 digit compute precision
- Effort: New file format + modified CUDA kernels
- Risk: Low. This is exactly how HFT systems work (ITCH stores int, software casts).

### Strategy C: "Quantized Columns" (Aggressive)

Change: Per-column quantization. Prices as uint16 with per-column min/max. Sizes as uint16. Direction as 2-bit code.

- File size: ~12.6 MB → ~7 MB (another 44% reduction)
- Precision: Lossy — 65,536 levels per column. For a stock moving $4 in a day, that's $0.00006 resolution.
- Use case: ML features where 16-bit precision is adequate
- Risk: Medium. Must validate that quantization noise doesn't corrupt signals.

### Strategy D: "The Posit Dream" (Experimental/Future)

Custom number format: Posit<16,1> or Posit<20,2>. Better precision distribution than IEEE floats for values near 1.0. A 16-bit posit can replace a 32-bit float for many applications.

- Not natively supported on NVIDIA GPUs (would need software emulation or FPGA)
- Theoretically optimal: more precision-per-bit than any IEEE format
- Research interest only for now

### Strategy E: "bfloat16 Derived Features" (Tensor Core Path)

For K02 binned data where we compute 170 characterizations per bin: store as bfloat16, compute aggregations via Tensor Cores at 250+ TFLOPS.

- 8 exponent bits (same range as float32), 7 mantissa bits (~2.3 decimal digits)
- Sufficient for: mean, std, min, max of binned financial data
- Uses Tensor Cores for matmul-style aggregations
- CUDA has full `__nv_bfloat16` support with intrinsics

---

## The Integer-Only Pipeline (Strategy B, Detailed)

```
┌─────────────────────────────────────────────┐
│ ON DISK (MKTF v1)                           │
│                                             │
│ price:      int32  (×10^-4, 4 bytes/tick)   │
│ size:       uint32 (whole shares, 4 bytes)  │
│ timestamp:  int64  (nanoseconds, 8 bytes)   │
│ conditions: uint32 (bitmask, 4 bytes)       │
│ exchange:   uint8  (enum, 1 byte)           │
│                                             │
│ Total: 21 bytes/tick × 598K = 12.6 MB       │
└────────────────┬────────────────────────────┘
                 │ NVMe read: ~2.5ms
                 │ (or DirectStorage: NVMe → GPU VRAM directly)
                 ▼
┌─────────────────────────────────────────────┐
│ GPU REGISTERS (fused pointwise kernel)      │
│                                             │
│ // Load + cast (1 instruction each)         │
│ float p = (float)price_int[i] * 1e-4f;     │
│ float s = (float)size_uint[i];              │
│ long long t = timestamp[i];                 │
│ uint32_t flags = conditions[i];             │
│                                             │
│ // Compute in float32 (125 TFLOPS)          │
│ float n = p * s;                            │
│ float lnp = logf(fmaxf(p, 1e-38f)); // SFU │
│ float sqp = sqrtf(fmaxf(p, 0.0f));  // SFU │
│ float rcp = 1.0f / p;               // SFU │
│ float snT = sinf(2.0f * PI * tf);   // SFU │
│ float csT = cosf(2.0f * PI * tf);   // SFU │
│                                             │
│ // Bitmask extraction (1 cycle each)        │
│ uint8_t round_lot = (flags >> 0) & 1;       │
│ uint8_t odd_lot   = (flags >> 1) & 1;       │
│ uint8_t is_trf    = (flags >> 2) & 1;       │
│                                             │
│ // ALL 20+ outputs computed in ~0.003ms     │
└────────────────┬────────────────────────────┘
                 │ Results stay on GPU
                 ▼
┌─────────────────────────────────────────────┐
│ GPU: Sequential + Windowed ops              │
│ (diff, lag, cumsum, rolling mean/std)       │
│ All in float32, all on GPU                  │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ GPU: Fused bin statistics (K02)             │
│ float32 → sum/mean/std/min/max/first/last   │
│ Per bin, per cadence, per column            │
└────────────────┬────────────────────────────┘
                 │ D2H or direct write to NVMe
                 ▼
┌─────────────────────────────────────────────┐
│ WRITE BACK (MKTF v1)                       │
│ K02 results as float32 or int32             │
│ With bfloat16 option for bulk features      │
└─────────────────────────────────────────────┘
```

---

## Cross-Domain Validation: Who Else Does This?

| Domain | Store As | Compute As | Why |
|--------|----------|-----------|-----|
| **HFT (ITCH)** | Integer (implied decimal) | Integer or cast to float | Exact prices, hardware-parseable |
| **Quantized ML** | INT8/INT16 | INT8 via Tensor Cores | 4-16x throughput vs FP32 |
| **Game engines** | Quantized/bit-packed vertices | Float32 after GPU transcode | Smaller storage, GPU does the cast |
| **Radio astronomy** | INT8 samples | Float32 for FFT/correlation | Sensor data IS integer; float is for math |
| **Genomics (SeqArray)** | Bit-packed genotypes (0.07 bytes/genotype) | Integer or float on access | 5.4x smaller than text |
| **NIfTI** | INT16 voxels + slope/intercept | Float32/64 after scaling | Storage-compact, compute-rich |
| **Our proposed MKTF** | INT32 prices, UINT32 sizes, UINT32 bitmask | Float32 via SFU | Exact, 4x smaller, 64x faster compute |

Every domain that optimizes for throughput stores integers and computes in float. We are not inventing a pattern — we are discovering one that already exists in every domain we surveyed.

---

## Risks and Mitigations

### Risk 1: Float32 precision loss in accumulation
**Scenario**: Summing 598K float32 prices for mean calculation. Worst case: $230 × 598K ≈ 1.4×10^8. Float32 has ~7 digits. The sum has 9 digits → 2 digits of accumulated error.
**Mitigation**: Kahan compensated summation (adds one extra float32 per accumulator, removes drift). Or accumulate in float64 for reductions only (tiny fraction of compute).

### Risk 2: BRK.A-class prices exceed int32 × 10^-4
**Scenario**: $700,000 × 10^4 = 7×10^9 > int32 max (2.1×10^9).
**Mitigation**: Per-file precision flag in MKTF header. 99.97% of securities use int32 × 10^-4. BRK.A uses int64 × 10^-8 (same kernel, template parameter).

### Risk 3: Mixed INT32/FP32 pipeline switching penalty on Blackwell
**Scenario**: Kernel interleaves integer loads and float32 math → 4-6x latency blowup.
**Mitigation**: Convert ALL integers to float32 in the first few instructions. Do all math in float32. Convert back at the end. Minimize pipeline switches.

### Risk 4: Sub-penny pricing needs more than 4 decimal places
**Scenario**: Some venues report prices with 6+ decimal places (midpoint prices, crypto).
**Mitigation**: Scale factor in column directory. Default 10^-4, configurable to 10^-6 or 10^-8. int32 × 10^-6 has max $2,147.48 — too small for most stocks. Use int64 × 10^-8 when > 4 decimals needed.

---

## Sources (Part 2)

- [RTX PRO 6000 Blackwell Architecture Whitepaper (NVIDIA)](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf)
- [RTX Blackwell GPU Architecture v1.1](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- [Dissecting Blackwell with Microbenchmarks](https://arxiv.org/html/2507.10789v2)
- [Unified INT32/FP32 Execution Unit in Blackwell](https://www.emergentmind.com/topics/unified-int32-fp32-execution-unit)
- [SFU Performance discussion (NVIDIA Forums)](https://forums.developer.nvidia.com/t/sfu-performance-in-a100/197699)
- [CUDA Math Intrinsics: Single Precision](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html)
- [CUDA Type Casting Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html)
- [ITCH Protocol Overview (OnixS)](https://www.onixs.biz/itch-protocol.html)
- [HFT Fixed-Point Numeric Preprocessing](https://www.linkedin.com/pulse/out-of-band-fpga-numeric-preprocessing-ull-hft-khaled-a-b-aly-phd)
- [Integer Quantization for Deep Learning Inference (NVIDIA)](https://arxiv.org/pdf/2004.09602)
- [Implementing High-Precision Decimal Arithmetic with CUDA int128](https://developer.nvidia.com/blog/implementing-high-precision-decimal-arithmetic-with-cuda-int128/)
- [Posits: Beating Floating Point at its Own Game](https://www.cl.cam.ac.uk/research/srg/han/hprls/orangepath/kiwic-demos/kiwi-posit-unum-and-custom-arithmetics/posits-gustafson-137-897-1-PB.pdf)
- [bfloat16 Precision Intrinsics (CUDA)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__BFLOAT16.html)
- [GTC'25: CUDA Techniques to Maximize Compute Throughput](https://shreyansh26.github.io/post/2025-04-04_gtc25-maximize-compute-instruction-throughput/)

---

# Part 3: Windows-Native NVMe → GPU I/O Strategy

**Naturalist: I/O path research**
**Date: 2026-03-27 (evening session)**

---

## The Problem: GPUDirect Storage is Linux-Only

NVIDIA's GPUDirect Storage (cuFile) enables DMA directly from NVMe to GPU VRAM, bypassing the CPU entirely. KvikIO (from RAPIDS) provides Python bindings. **Neither works on Windows.**

Microsoft's DirectStorage is the Windows equivalent, but it's a DirectX 12 API designed for game asset loading. Using it with CUDA requires D3D12-CUDA interop (`cudaGraphicsD3D12RegisterResource`), which is complex and fragile.

## What Actually Works on Windows: Pipelined Pinned-Memory I/O

The llama.cpp team prototyped this exact pipeline for loading large model tensors on Windows ([PR #7796](https://github.com/ggml-org/llama.cpp/pull/7796)). Their results:

| Approach | Throughput | Notes |
|----------|-----------|-------|
| mmap + H2D | 3-4 GB/s | Sequential, CPU bounce |
| Unbuffered I/O to pinned mem, pipelined H2D | **8.5 GB/s** | 4 × 1MB pinned buffers, overlapped |
| DirectStorage + CUDA interop | Theoretical max | Complex D3D12 interop, not benchmarked |

The pipelined approach is simple and fast:

```
Buffer A: [read from NVMe]  →  [H2D copy to GPU]
Buffer B:                      [read from NVMe]  →  [H2D copy to GPU]
Buffer C:                                           [read from NVMe]  → ...
```

While buffer A is being H2D-copied, buffer B is being filled from NVMe. The NVMe read and PCIe H2D transfer run in parallel on different hardware (NVMe controller vs PCIe DMA engine).

### Implementation Recipe (Windows-Native)

```python
# 1. Open file with NO_BUFFERING + OVERLAPPED
handle = CreateFileW(path,
    GENERIC_READ,
    FILE_SHARE_READ,
    None,
    OPEN_EXISTING,
    FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
    None)

# 2. Allocate N pinned host buffers
pinned_bufs = [cuda.mem_alloc_host(CHUNK_SIZE) for _ in range(N_BUFFERS)]

# 3. For each column in MKTF file:
#    - Seek to column offset (from directory)
#    - ReadFile into pinned_buf[i % N]  (overlapped, non-blocking)
#    - When complete: cudaMemcpyAsync(d_ptr, pinned_buf, nbytes, H2D, stream)
```

### Performance Estimate for MKTF

With 12.6 MB files (Strategy B integer encoding) at 8.5 GB/s pipelined:

| Metric | Value |
|--------|-------|
| NVMe read (12.6 MB) | ~1.5ms |
| H2D transfer (12.6 MB, PCIe 5 x16) | ~0.2ms (overlapped with read) |
| GPU recompute derived columns (FP32) | ~0.003ms |
| **Total per ticker** | **~1.5ms** |
| **Full universe (4,604 tickers)** | **~6.9 seconds** |

vs. Current pipeline (~44 seconds for full universe read): **6.4x speedup**.

### Why Not DirectStorage?

For our specific case:
1. Our files are 12.6 MB — not large enough to benefit from GPU decompression (GDeflate overhead > savings)
2. The pipelined pinned-memory approach already achieves ~85% of NVMe line rate
3. DirectStorage requires DirectX 12 interop, adding significant complexity
4. The pinned-memory approach works with pure CUDA — no DirectX dependency

DirectStorage becomes interesting at scale: reading 58 GB (full daily universe) where GPU decompression could free up CPU. But with Strategy B's 12.6 MB files (no compression needed), the simple path wins.

### Windows I/O Ring API

Windows 11 added I/O rings (similar to Linux `io_uring`) via `BuildIoRingReadFile()`. Benchmarks show only ~2-3% improvement over I/O completion ports. Not worth the complexity for our use case.

The bottleneck is NVMe sequential read bandwidth, not kernel call overhead. With 12.6 MB files, we make one `ReadFile` call per column (5 columns × ~2.5 MB avg). The overlapped I/O path handles this efficiently.

### Future: DirectStorage for K02 Bulk Data

K02 produces 170 characterizations × 31 cadences × 4,604 tickers per day. This is much larger than K01, and here DirectStorage + GDeflate might pay off:
- K02 data is highly compressible (many zeros, repeated patterns)
- GDeflate compresses on CPU, decompresses on GPU
- DirectStorage reads compressed data from NVMe and decompresses on GPU in one pipeline
- Potentially 2-3x throughput improvement via compression ratio

This is a v2 optimization — the pipelined pinned-memory approach is the correct v1.

---

## Sources (Part 3)

- [llama.cpp DirectStorage CUDA interop PR #7796](https://github.com/ggml-org/llama.cpp/pull/7796)
- [GPUDirect Storage Overview (NVIDIA)](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
- [GPUDirect Storage Design Guide](https://docs.nvidia.com/gpudirect-storage/design-guide/index.html)
- [KvikIO — GPU-accelerated I/O (RAPIDS)](https://github.com/rapidsai/kvikio)
- [PyTorch torch.cuda.gds](https://docs.pytorch.org/tutorials/unstable/gpu_direct_storage.html)
- [Windows I/O Ring API](https://learn.microsoft.com/en-us/windows/win32/api/ioringapi/)
- [I/O Rings — When One I/O Operation is Not Enough](https://windows-internals.com/i-o-rings-when-one-i-o-operation-is-not-enough/)
- [DirectStorage GitHub (Microsoft)](https://github.com/microsoft/DirectStorage)
- [ssd-gpu-dma: Userspace NVMe drivers with CUDA](https://github.com/enfiskutensykkel/ssd-gpu-dma)
