# Family 03: Signal Processing — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: REVIEWED
**Code**: `crates/tambear/src/signal_processing.rs`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| FFT (Cooley-Tukey radix-2) | signal_processing.rs:103-137 | OK |
| IFFT | signal_processing.rs:154-162 | OK |
| rfft/irfft | signal_processing.rs:168-187 | OK |
| 2D FFT | signal_processing.rs:192+ | OK |
| periodogram | signal_processing.rs:299-313 | OK |
| Welch's method | signal_processing.rs:320+ | OK |
| Convolution (FFT-based) | | OK |
| Window functions | | OK |
| Butterworth (biquad) | signal_processing.rs:597-612 | OK |
| Butterworth cascade | signal_processing.rs:659+ | OK |
| Hilbert transform | signal_processing.rs:764+ | OK |
| DCT-II/III | | OK |
| Median filter | signal_processing.rs:997-1016 | **HIGH** (NaN panic) |
| Savitzky-Golay | signal_processing.rs:740+ | OK (delegates to polyfit) |

---

## Finding F03-1: Median Filter NaN Panic (HIGH)

**Bug**: `median_filter` at line 1007 uses `a.partial_cmp(b).unwrap()` which panics when any value in the window is NaN.

**Impact**: Thread panic. Financial time series with missing data encoded as NaN will crash.

**Fix**: `a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)` or pre-filter NaN.

Same pattern as F26-1 (complexity.rs) and pervasive across the codebase.

---

## Finding F03-2: FFT Twiddle Factor Accumulation (LOW)

**Note**: The FFT (line 118-125) multiplies `w = c_mul(w, wn)` iteratively within each stage. For large N (2^20+), the accumulated rounding error in the twiddle factor grows as O(half * eps). For half = 2^19, error ≈ 5e-11.

For financial signal processing (typical N < 2^16), this gives error < 3e-12. Acceptable.

For high-precision scientific work, recompute `w = c_exp_i(angle * j)` per iteration. But the perf cost (trig per butterfly) makes this undesirable.

---

## Positive Findings

**Butterworth biquad design is correct.** Bilinear transform with proper pre-warping. Cascade structure avoids high-order polynomial instability. DC gain = 1.0 verified.

**Hilbert transform is correct.** Standard frequency-domain approach: zero negative frequencies, double positive. Analytic signal properly formed.

**Periodogram scaling is correct.** `|FFT(x)|²/(fs*N)` with 2x factor for non-DC/Nyquist bins.

**Welch's method is correct.** Hann window, proper overlap-add, window normalization `win_norm = Σw²`.

---

## Test Vectors

### TV-F03-MED-01: Median filter NaN (BUG CHECK)
```
data = [1.0, 2.0, NaN, 4.0, 5.0], window=3
Expected: graceful handling
Currently: PANIC
```

### TV-F03-FFT-01: Parseval's theorem
```
data = [random; 1024]
sum(|x|²) should equal sum(|FFT(x)|²) / N within 1e-10
```

### TV-F03-FFT-02: Known DFT
```
data = [1.0, 0.0, 0.0, 0.0]
Expected FFT: [1, 1, 1, 1] (all real, magnitude 1)
```

### TV-F03-INV-01: FFT roundtrip
```
ifft(fft(data)) ≈ data within 1e-12
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F03-1: Median NaN panic | **HIGH** | Thread panic | unwrap_or(Equal) |
| F03-2: Twiddle accumulation | **LOW** | ~1e-11 error for N=2^20 | Recompute per butterfly |
