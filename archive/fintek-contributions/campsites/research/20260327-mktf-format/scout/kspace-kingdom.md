# K-Space Kingdom — Scout Design Notes
## 2026-03-27

---

## The Core Insight

The MKTF columnar structure is already coordinate-system agnostic.

In time-domain K01 KO00: rows = ticks, columns = features (price, size, log_price…)
In frequency-domain K02 KO01: rows = frequency bins, columns = spectral features (power, phase, coherence…)
In wavelet K02 KO04: rows = scale×shift pairs, columns = wavelet coefficients
In sufficient-statistics K02 KO05: rows = statistics (mean, var, skew…), columns = cross-ticker projections

The column directory already has `n_elements` per column. The data layout is already columnar.
**The format already works for k-space. What's missing is a domain descriptor in Block 0.**

---

## What Needs to Change in the Header

Current Block 0 reserved space after upstream fingerprint table: `[344..4096)` = 3,752 bytes free.

**Proposed domain descriptor block at offset 344 (64 bytes):**

```
[344:348]  domain_type       uint32   0=time, 1=DFT, 2=DCT, 3=wavelet, 4=sufficient_stats, 5=custom
[348:352]  domain_version    uint32   versioning for transform parameter format
[352:360]  transform_flags   uint64   bitmask: bit0=complex_valued, bit1=normalized, bit2=centered, bit3=log_scale
[360:408]  transform_params  bytes48  domain-specific parameters (see below)
```

Transform params by domain type:
```
DFT (domain_type=1):
  [0:4]   n_fft         uint32   FFT window size
  [4:8]   hop_length    uint32   hop in samples
  [8:12]  window_type   uint32   0=rect, 1=hann, 2=hamming, 3=blackman
  [12:16] freq_min_mhz  float32  min frequency (mHz = milli-Hz for market timescales)
  [16:20] freq_max_mhz  float32  max frequency
  [20:24] freq_bins     uint32   number of frequency bins
  [24:48] reserved

Wavelet (domain_type=3):
  [0:4]   wavelet_type  uint32   0=haar, 1=daubechies4, 2=sym8, 3=coif5
  [4:8]   levels        uint32   number of decomposition levels
  [8:12]  scale_min     float32  finest scale (= 1/Nyquist)
  [12:16] scale_max     float32  coarsest scale
  [16:48] reserved

Sufficient stats (domain_type=4):
  [0:4]   stat_type     uint32   0=moments, 1=cumulants, 2=quantiles, 3=spectral_moments,
                                 4=PHASE (cos/sin pair per target period — see K02 KO05 PHASE)
  [4:8]   order_max     uint32   maximum order (e.g., 4 = up to kurtosis); unused for stat_type=4
  [8:48]  reserved
```

**For time-domain files (domain_type=0): all transform fields are zero.**
This is backwards compatible — v3 files with reserved bytes = 0 are implicitly time-domain.

---

## The Cadence-Nyquist Connection

The cadence IS the frequency grid in k-space. This connects Thread 1 and Thread 2 directly.

- 1ms cadence → Nyquist = 500 Hz (can see sub-millisecond periodicity)
- 100ms cadence → Nyquist = 5 Hz (can see ~200ms cycles)
- 1s cadence → Nyquist = 0.5 Hz (can see 2s periodicity, straddles HFT round-second)
- 5min cadence → Nyquist = 1.67 mHz (can see intraday rhythms down to 10-minute cycles)

**The MI cadence optimization (naturalist's Thread 1) tells us which cadences are informationally
rich. These are exactly the cadences that produce K02 KO01 files with maximum spectral discriminating
power for gaming detection.**

The measurement function and detection function want their cadences for the SAME reason in k-space:
high MI = high spectral information content = can distinguish "gaming" spectral signature from noise.

The cadence grid and the frequency resolution grid are the SAME optimization.

A cadence that's maximally informative per MI analysis will produce K02 KO01 data where the
gaming signature (periodicity just below round boundaries) has maximum SNR.

---

## Complex Dtypes — Column Directory Addition

Current `dtype_code` values (from mktf_v3.py column directory):
```
float32=1, float64=2, int32=3, int64=4, uint32=5, uint16=6, int16=7, uint8=8
```

K-space data is often complex-valued. Need to add:
```
complex64=9   (2× float32, real + imag interleaved)
complex128=10 (2× float64)
```

Or alternatively: store real and imaginary parts as separate columns (complex64 → two float32 columns
with names like "power_re" and "power_im"). This is simpler and avoids dtype changes.

**Recommendation**: store real/imag as separate float32 columns. The domain descriptor's
`transform_flags` bit0 (complex_valued=1) signals that columns come in (re, im) pairs.
Pathmaker can validate this convention without a dtype change.

---

## Upstream Fingerprint Chain for K-Space

The upstream fingerprint design already handles k-space staleness with no changes needed.

A K02 KO01 file (DFT of K01):
```
upstream_fingerprints[0]:
  upstream_leaf_id  = "K01P01"      # leaf_id of the source time-domain file
  upstream_write_ts = <K01's write_timestamp_ns at time K02 KO01 was computed>
  upstream_data_hash = xxHash64(K01 data bytes)  # optional hash for content verification
```

The daemon's staleness check: "if K01 has been updated since K02 KO01 was computed → K02 KO01 is stale."
Same algorithm. Same header-only read. Zero new machinery.

Multi-upstream k-space is also handled. A cross-ticker coherence file (K04 KO01) might have:
```
upstream_fingerprints[0]: K01P01 (ticker A)
upstream_fingerprints[1]: K01P01 (ticker B)  ← different ticker, same leaf type
```

Up to 4 upstreams = 4 tickers can be named in Block 0. For the K04 full cross-ticker
correlation matrix (all 4,604 tickers), the upstream table would overflow 4 entries.

**Suggested extension for K04-class files**: add a `upstream_batch_hash` field in the
domain descriptor block (bytes [400:408]) = xxHash64 of all upstream write timestamps
concatenated. For K04, instead of naming 4,604 individual upstreams, store one hash of the
entire upstream state. The staleness check becomes: "recompute all upstream timestamps, hash
them, compare with stored hash." Still a single Block 0 comparison.

---

## KO Code Registry

The `ko_type` field in the LAYOUT header mirrors the KO code from the leaf filename.
If filename and header disagree, the header is authoritative (belt and suspenders).

| KO Code | Name | domain_type | Description |
|---------|------|-------------|-------------|
| KO00 | Columnar | 0 (time) | Default — raw bins, pointwise features (K01, K02, K03) |
| KO01 | FFT Cartesian | 1 (DFT) | DFT output in Cartesian (re, im) columns |
| KO02 | FFT Radial | 1 (DFT) | DFT output as (amplitude, phase) columns |
| KO03 | FFT Spiral | 1 (DFT) | Chirp-Z or log-spaced frequency grid |
| KO04 | Wavelet | 3 (wavelet) | Haar, Daubechies, sym8, coif5, etc. |
| KO05 | Sufficient stats | 4 (sufficient_stats) | Composable — moments, cumulants, PHASE, MI score |
| KO06 | Correlation matrix | 0 (time) | Cross-ticker Pearson/Spearman (K04) |
| KO07 | Eigenvectors/PCA | 4 (sufficient_stats) | Principal components of feature matrix |
| KO08 | Compressed sensing | 5 (custom) | Sparse measurement matrix |
| KO09 | Grid/tensor | 0 (time) | Cross-ticker × time tensor (K03-class) |
| KO10 | Sparse | 5 (custom) | Nonzero entries only, COO or CSR layout |

---

## Kingdom Taxonomy Draft

| Kingdom | Domain | KO code | Leaf ID prefix | Typical n_rows | Columns |
|---------|--------|---------|----------------|----------------|---------|
| Time Kingdom | Time-domain binned features | KO00 | K01, K02, K03 | ticks (100K-1M) | price, size, log_price, rolling_mean… |
| Frequency Kingdom | DFT / DCT | KO01, KO02 | K02 KO01, K02 KO02 | freq_bins (512-4096) | power_re, power_im, phase, coherence |
| Scale Kingdom | Wavelet | KO04 | K02 KO04 | scale×shift pairs | wavelet_coeff, detail, approx |
| Statistics Kingdom | Sufficient stats | KO05 | K02 KO05 | stat_order (4-20) | moments, cumulants, quantiles, phase |
| Correlation Kingdom | Cross-ticker | KO06 | K04 | ticker_pairs (4604²) | pearson_r, spearman_r, cov |

Each Kingdom is a valid, self-consistent representation of the same underlying market.
Each uses the SAME format, SAME daemon, SAME reconcile algorithm.
The only difference is the domain descriptor in Block 0.

---

## What This Changes in the Implementation Delta

Adding k-space support to MKTF v3:

1. Add `DOMAIN_DESCRIPTOR_OFFSET = 344`, `DOMAIN_DESCRIPTOR_SIZE = 64`
2. Add `domain_type` and `transform_params` to `MKTFHeader.__slots__`
3. Pack/unpack domain descriptor in `write_mktf()` and `read_header()`
4. Add `complex64` convention (real + imag as paired columns, flagged in transform_flags)
5. **Set `alignment=64` for KO05 files** (Block 0 offset [10:12]). KO00 keeps `alignment=4096`.
6. **Fix reader for small files**: single `f.read()` of entire file into buffer, then `np.frombuffer`
   with offsets. Current `read_columns()` opens the file TWICE and does 25 individual seeks —
   4,876us/file bulk vs 105us warm. Rust reader must be designed single-read from the start.

For time-domain files: domain_type=0, all zeros = backwards compatible with v3.

**KO00 vs KO05 alignment**:
- `KO00`: `alignment=4096` — GPU DMA page-aligned, large columns, CUDA memcpy target
- `KO05`: `alignment=64` — GPU cache-line aligned, tiny stat columns, CPU consumption

25 stat columns × 64 bytes = 1,600 bytes overhead. Same at 4096 = 102,400 bytes overhead.
File size: ~110 KB → ~3-4 KB. Same format, same reader, same writer. Parameter only.

**KO05 reads — bulk performance (Exp 20+21)**:
- Production reader bulk: 4,657us/file (double-open + 25 seeks — 0.06% NVMe utilization)
- **Fixed reader bulk: 103us/file** — cold ≈ warm, 45x speedup, ~15 line change (task #48)
- K04 screening with fix: **0.47s/cadence** ≈ embedded progressive (0.4s) — architecture neutral
- Rust reader design: single `f.read()` into buffer → `np.frombuffer` with offsets, mandatory

The daemon, reconcile, and BitmapStateDB code are unchanged. They read `is_complete` and
`upstream_write_ts` from Block 0. Domain type is irrelevant to operational decisions.

Only computational kernels (K02 KO01 FFT kernel, K02 KO04 wavelet kernel) need to know about domain.
Analytical tools that want to understand the domain read the descriptor. The daemon never does.

---

## Open Questions for Naturalist

1. Should K-space files carry the Nyquist frequency explicitly in the domain descriptor,
   or is it derivable from the source cadence via upstream_leaf_id lookup?
   (Explicit is safer — doesn't require reading upstream header to understand this file.)

2. For the gaming detection use case: is the interesting spectral feature the DFT of the
   raw tick series, or the DFT of the inter-arrival times (gaps between ticks)?
   These have very different Nyquist properties. The gap-series DFT would live in
   a different K02 KO01 leaf than the price-series DFT.

3. For the MI cadence optimization: can the MI score itself be stored as a K02 KO05 leaf
   (sufficient statistics leaf with stat_type=MI_SCORE)? This would make the cadence
   selection reproducible and auditable via the standard MKTF pipeline.
   **ANSWERED: YES.** K02 KO05 MI_SCORE leaf design specified in cadence-optimization-framework.md.
   MI experiment now unblocked (fused kernel wired in, commit ba66b70).

---

## K02 KO05 PHASE Leaf — Production (V columns carry confidence)

**Historical note (2026-03-28)**: The three-clock finding that motivated this leaf did not
survive validation — bootstrap CI per ticker = ±170°, split-day phase drift = 172°, the
AAPL-NVDA 0.9° coincidence was a cancellation artifact. The FINDING was retracted; the
LEAF is production. See market-agency-layers.md for full retraction details.

The farm computes PHASE for every ticker, every day, regardless of stability. V columns
carry the signal about the signal — consumers decide what to trust.

**Schema (production):**

```
leaf_id:    "K02P##C##.TI00TO00.KI00KO05"   (stat_type=4 PHASE in Block 0 transform_params)
domain:     sufficient_stats (domain_type=4)
stat_type:  4 (PHASE)
upstream:   K01 (for this ticker+day)

Rows: one row per candidate period

DO01: phase_cos        float32   cos(phase) at target period
DO02: phase_sin        float32   sin(phase) at target period
DO03: phase_excess     float32   spectral excess (detrended) — the robust feature

V01:  within_day_r     float32   phase stability: 0=junk, 1=locked (mean resultant R across segments)
V02:  bootstrap_ci_deg float32   95% CI width in degrees (500 block-bootstrap replicates)
V03:  n_segments_used  uint32    how many intraday segments contributed to phase estimate
```

V01 = 0.15 means "unstable today." V01 = 0.85 means "phase is locked." DO03 (phase_excess)
is always meaningful — the amplitude fingerprint (NVDA 28.5x, CHWY 1.8x) is robust regardless
of V01. DO01/DO02 (phase angles) are meaningful when V01 is high.

**Open investigation**: Execution regime (1-30s) phase stability not yet tested. May show
higher V01 than 5min. Shorter intraday windows conditioned on high-amplitude segments also
untested (within-segment R = 0.399 at 5min is weak but nonzero).
