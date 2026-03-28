# K-Space Kingdom — Scout Design Notes
## 2026-03-27

---

## The Core Insight

The MKTF columnar structure is already coordinate-system agnostic.

In time-domain K01: rows = ticks, columns = features (price, size, log_price…)
In frequency-domain K-F01: rows = frequency bins, columns = spectral features (power, phase, coherence…)
In wavelet K-W01: rows = scale×shift pairs, columns = wavelet coefficients
In sufficient-statistics K-SS01: rows = statistics (mean, var, skew…), columns = cross-ticker projections

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
  [0:4]   stat_type     uint32   0=moments, 1=cumulants, 2=quantiles, 3=spectral_moments
  [4:8]   order_max     uint32   maximum order (e.g., 4 = up to kurtosis)
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
rich. These are exactly the cadences that produce K-F01 files with maximum spectral discriminating
power for gaming detection.**

The measurement function and detection function want their cadences for the SAME reason in k-space:
high MI = high spectral information content = can distinguish "gaming" spectral signature from noise.

The cadence grid and the frequency resolution grid are the SAME optimization.

A cadence that's maximally informative per MI analysis will produce K-F01 data where the
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

A K-F01 file (DFT of K01):
```
upstream_fingerprints[0]:
  upstream_leaf_id  = "K01P01"      # leaf_id of the source time-domain file
  upstream_write_ts = <K01's write_timestamp_ns at time K-F01 was computed>
  upstream_data_hash = xxHash64(K01 data bytes)  # optional hash for content verification
```

The daemon's staleness check: "if K01 has been updated since K-F01 was computed → K-F01 is stale."
Same algorithm. Same header-only read. Zero new machinery.

Multi-upstream k-space is also handled. A cross-ticker coherence K-COH01 file might have:
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

## Kingdom Taxonomy Draft

| Kingdom | Domain | Leaf ID prefix | Typical n_rows | Columns |
|---------|--------|----------------|----------------|---------|
| Time Kingdom | Time-domain binned features | K01, K02, K03 | ticks (100K-1M) | price, size, log_price, rolling_mean… |
| Frequency Kingdom | DFT / DCT | K-F01, K-F02 | freq_bins (512-4096) | power_re, power_im, phase, coherence |
| Scale Kingdom | Wavelet | K-W01 | scale×shift pairs | wavelet_coeff, detail, approx |
| Statistics Kingdom | Sufficient stats | K-SS01 | stat_order (4-20) | moments, cumulants, quantiles |
| Correlation Kingdom | Cross-ticker | K04 | ticker_pairs (4604²) | pearson_r, spearman_r, cov |

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

For time-domain files: domain_type=0, all zeros = backwards compatible with v3.

The daemon, reconcile, and BitmapStateDB code are unchanged. They read `is_complete` and
`upstream_write_ts` from Block 0. Domain type is irrelevant to operational decisions.

Only computational kernels (K-F01 FFT kernel, K-W01 wavelet kernel) need to know about domain.
Analytical tools that want to understand the domain read the descriptor. The daemon never does.

---

## Open Questions for Naturalist

1. Should K-space files carry the Nyquist frequency explicitly in the domain descriptor,
   or is it derivable from the source cadence via upstream_leaf_id lookup?
   (Explicit is safer — doesn't require reading upstream header to understand this file.)

2. For the gaming detection use case: is the interesting spectral feature the DFT of the
   raw tick series, or the DFT of the inter-arrival times (gaps between ticks)?
   These have very different Nyquist properties. The gap-series DFT would live in
   a different K-F leaf than the price-series DFT.

3. For the MI cadence optimization: can the MI score itself be stored as a K-SS01 leaf
   (sufficient statistics leaf with stat_type=MI_SCORE)? This would make the cadence
   selection reproducible and auditable via the standard MKTF pipeline.
   **ANSWERED: YES.** K-SS01(MI_SCORE) leaf design specified in cadence-optimization-framework.md.
   MI experiment now unblocked (fused kernel wired in, commit ba66b70).

---

## K-SS01(PHASE) Leaf — Confirmed Data Product

Phase analysis of 5min inter-arrival time periodicity (10 tickers, circular k-means) confirmed
that phase is a genuine per-ticker feature. Three sector clocks exist at 5min:
- Mega-cap tech (AAPL/NVDA/TSLA): center -18°, internal R=0.968
- Consumer/value (KO/BRK.B/JNJ/CHWY): center +72°, internal R=0.883
- Phase-shifted tech (MSFT/AMD/META): center -157°, internal R=0.843

**Schema: K-SS01(PHASE)**

```
leaf_id:    "K-SS01-PHASE"
domain:     sufficient_stats (domain_type=4)
upstream:   K01 (for this ticker+day)

Columns (n_candidates rows — one row per candidate period):
  candidate_period_ms   float32
  phase_cos             float32   cos(phase at this period)
  phase_sin             float32   sin(phase at this period)
  amplitude             float32   spectral excess at this period (detrended)
  rayleigh_r            float32   local R (from multi-day averaging, if available)

Block 0 transform_params (stat_type=PHASE):
  period_range_min_ms, period_range_max_ms
  n_candidates
  detrend_method        uint8   0=none, 1=mean_iat_subtraction
```

**Why (cos, sin) and not raw angle:**
- Circular quantities don't have a natural linear distance — (cos, sin) makes Euclidean
  K04 correlation well-defined and continuous across the ±180° wrap point
- Standard Pearson on (cos, sin) approximates circular correlation (rho_c) well for
  moderate phase differences — no custom distance function needed in K04
- The three sector clocks at 120° spacing fall out automatically from Euclidean K04:
  cluster centroids in (cos, sin) space are [(-0.31, -0.95), (-0.53, +0.85), (-0.97, -0.25)]

**Upstream**: K01 → stale if K01 updates (daily recompute).
**Relationship to K-SS01(MI_SCORE)**: separate leaves, different stat_type values.
K-SS01(PHASE) answers "when does this ticker's algorithm pulse?"
K-SS01(MI_SCORE) answers "how much information does each cadence contain?"
Both live in domain_type=4 (sufficient_stats), different stat_type codes.
