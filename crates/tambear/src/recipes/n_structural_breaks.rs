//! Structural-break count via CUSUM of squared returns.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! Kulisch-backed squared-return mean + a prefix-sum scan of centred
//! squared returns + a pointwise threshold + a run-length reduction.
//! See `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! A structural break in a return series is a change in the variance
//! regime. The Inclán–Tiao (1994) "IT" CUSUM-of-squares detector tracks
//! the running sum of squared returns minus its expected proportion and
//! flags the peak-amplitude point as a candidate break when the
//! amplitude exceeds a critical value:
//!
//! ```text
//! c[k]  =  Σ_{t=1..k} r[t]²
//! D[k]  =  c[k] / c[n]  −  k / n
//! ```
//!
//! `D[k]` is a standardized CUSUM: it starts and ends at 0 and excursions
//! are proportional to √n times the Kolmogorov–Smirnov statistic under
//! homoscedasticity. Under the null of constant variance, `max_k |D[k]|`
//! has a known asymptotic distribution with 5% critical value `1.358`
//! (Inclán & Tiao 1994 Table 1).
//!
//! This recipe counts structural breaks by **iteratively splitting**:
//! find the peak-`|D|` point, record a break if `max|D| > τ`, then
//! recurse on the two sub-series. Breaks that are too close together
//! (smaller than `min_spacing`) are merged.
//!
//! ```text
//! count_breaks(r[0..n], τ, min_spacing):
//!     if n < 2 · min_spacing:           return 0
//!     k*, max_d  =  argmax |D[k]|
//!     if max_d ≤ τ:                     return 0
//!     return 1 + count_breaks(r[0..k*], τ, min_spacing)
//!              + count_breaks(r[k*..n], τ, min_spacing)
//! ```
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` the per-
//! hour `n_structural_breaks` header field (u16) is the count of
//! variance-regime changes within the hour. With 100 ms buckets that
//! is up to 36000 return observations per hour — but in practice
//! structural breaks are rare events (`n < 10` per hour even in
//! stressed markets), so u16 is far more than sufficient.
//!
//! # Reference
//!
//! Inclán, C., & Tiao, G. C. (1994). Use of cumulative sums of squares
//! for retrospective detection of changes of variance. *Journal of the
//! American Statistical Association* 89(427): 913–923.
//!
//! # Composition
//!
//! - **Kulisch-backed sum of squares** — `c[n] = Σ r[t]²`. Lowers to
//!   `accumulate(r, All, Op::Add, expr=Custom("v·v"))`.
//! - **Prefix-sum of squares** — `c[k]` for all `k`. Lowers to
//!   `accumulate(r, Prefix, Op::Add, expr=Custom("v·v"))`.
//! - **Pointwise standardization** — `D[k] = c[k]/c[n] − k/n`.
//! - **Argmax-abs reduction** — `accumulate(|D|, All, Op::MaxIdx)`.
//! - **Recursive split** — a control-flow loop, not an atom.
//!
//! # NaN/Inf policy
//!
//! Follows the standard accumulate-layer contract: `!is_finite(x) →
//! skip` per element. The recipe pre-filters the input to its finite
//! subset before running the CUSUM. This prevents the
//! prefix-flatline-vs-index-growth mismatch that would otherwise cause
//! a run of non-finite values to spuriously inflate `|D[k]|` into a
//! false break.
//!
//! - Length `< 2·min_spacing` (after filtering) → returns 0 (cannot
//!   resolve a break).
//! - Non-finite returns → skipped in both the squared-return sum and
//!   the index proportion (the CUSUM is computed on the compacted
//!   finite-only sequence).
//! - Constant-variance input (`Σr² = 0`) → returns 0 (no amplitude,
//!   no break).
//!
//! # Default parameters
//!
//! - `critical_value` — 1.358. Inclán–Tiao 5% critical value for the
//!   standardized CUSUM-of-squares statistic. Lower it (1.224, 1%) for
//!   stricter break detection or raise it (1.036, 10%) for looser.
//! - `min_spacing` — 5. Breaks within 5 observations of each other are
//!   merged into one, preventing double-counting of a single variance
//!   change.
//!
//! Both are exposed on `n_structural_breaks_with_params`;
//! `n_structural_breaks` uses the SIP defaults.

/// Result of structural-break detection: count plus the full list of
/// break locations and their CUSUM magnitudes, sorted descending by
/// magnitude.
#[derive(Debug, Clone)]
pub struct NStructuralBreaksResult {
    /// Total number of structural breaks detected.
    pub count: u32,
    /// Number of finite observations the detector ran on.
    pub n_observations: u64,
    /// Break locations in the *filtered* (finite-only) sequence,
    /// sorted descending by magnitude. Mapping back to the original
    /// input index is the caller's responsibility (since we skipped
    /// non-finite values).
    pub locations: Vec<u32>,
    /// CUSUM magnitudes corresponding to `locations` (same order, same
    /// length). Larger magnitudes are stronger evidence of a break.
    pub magnitudes: Vec<f64>,
}

impl NStructuralBreaksResult {
    /// Empty result — no breaks detected, no finite data.
    #[inline]
    pub fn empty(n_observations: u64) -> Self {
        Self {
            count: 0,
            n_observations,
            locations: Vec::new(),
            magnitudes: Vec::new(),
        }
    }
}

/// SIP-default structural-break detector via CUSUM of squared returns.
///
/// Returns the full `NStructuralBreaksResult` with all detected break
/// locations and magnitudes sorted descending. The caller can truncate
/// to a fixed top-N (e.g., SIP takes top 8 for the hourly header).
///
/// Uses `critical_value = 1.358` (Inclán–Tiao 5%) and `min_spacing = 5`.
pub fn n_structural_breaks(returns: &[f64]) -> NStructuralBreaksResult {
    n_structural_breaks_with_params(returns, 1.358, 5)
}

/// Structural-break detector with configurable critical value and
/// minimum spacing between breaks.
///
/// `critical_value` controls strictness: smaller values flag more
/// breaks. Standard Inclán–Tiao values: 1.358 (5%), 1.224 (10%), 1.628
/// (1%). `min_spacing` merges breaks that are closer than this many
/// observations.
///
/// # Panics
///
/// Panics if `critical_value <= 0.0` or non-finite, or if
/// `min_spacing == 0`.
pub fn n_structural_breaks_with_params(
    returns: &[f64],
    critical_value: f64,
    min_spacing: usize,
) -> NStructuralBreaksResult {
    assert!(
        critical_value.is_finite() && critical_value > 0.0,
        "n_structural_breaks: critical_value must be finite and > 0, got {critical_value}"
    );
    assert!(
        min_spacing > 0,
        "n_structural_breaks: min_spacing must be > 0"
    );

    // Pre-filter to finite values. Running the CUSUM on a series with
    // flatlined (skipped) prefix positions but still-growing k/n ratios
    // would produce spurious breaks at the edges of NaN runs — so we
    // compact to finite-only first per the accumulate contract.
    let filtered: Vec<f64> = returns.iter().copied().filter(|v| v.is_finite()).collect();
    let n = filtered.len();
    if n < 2 * min_spacing {
        return NStructuralBreaksResult::empty(n as u64);
    }
    let returns: &[f64] = &filtered;

    // Iterative split via stack of (start, end) ranges. Collect every
    // detected break with its location and amplitude.
    let mut breaks: Vec<(u32, f64)> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(0, n)];

    while let Some((start, end)) = stack.pop() {
        let len = end - start;
        if len < 2 * min_spacing {
            continue;
        }
        let sub = &returns[start..end];
        let (k_peak, max_d) = peak_cusum_abs(sub);
        if max_d <= critical_value {
            continue;
        }
        // Enforce spacing from the segment edges.
        if k_peak < min_spacing || len - k_peak < min_spacing {
            continue;
        }
        let global_index = start + k_peak;
        breaks.push((global_index as u32, max_d));
        stack.push((start, global_index));
        stack.push((global_index, end));
    }

    // Sort descending by magnitude; stable so duplicate magnitudes
    // keep split-order.
    breaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let count = breaks.len() as u32;
    let (locations, magnitudes): (Vec<u32>, Vec<f64>) = breaks.into_iter().unzip();

    NStructuralBreaksResult {
        count,
        n_observations: n as u64,
        locations,
        magnitudes,
    }
}

/// Peak of `|D[k]| · √(n/2)` over interior `k`, where `D[k]` is the
/// standardized CUSUM of squared returns. Returns `(k_peak, max_d)`.
/// `max_d` is already scaled to compare directly against Inclán–Tiao
/// critical values.
fn peak_cusum_abs(returns: &[f64]) -> (usize, f64) {
    use crate::primitives::specialist::kulisch_accumulator::KulischAccumulator;

    let n = returns.len();
    if n < 2 {
        return (0, 0.0);
    }

    // Total sum of squares (Kulisch-exact).
    let mut total = KulischAccumulator::new();
    for &r in returns {
        if r.is_finite() {
            total.add_f64(r * r);
        }
    }
    let c_n = total.to_f64();
    if !(c_n > 0.0) {
        return (0, 0.0);
    }

    // Scan c[k] and maximize |c[k]/c[n] − k/n|, scaled by √(n/2) to
    // produce the Inclán–Tiao-normalized statistic directly.
    let mut running = KulischAccumulator::new();
    let mut k_peak: usize = 0;
    let mut max_d: f64 = 0.0;
    let nf = n as f64;
    let scale = (nf / 2.0).sqrt();
    for (k, &r) in returns.iter().enumerate() {
        if r.is_finite() {
            running.add_f64(r * r);
        }
        let kf = (k + 1) as f64;
        let d = (running.to_f64() / c_n - kf / nf).abs() * scale;
        if d > max_d {
            max_d = d;
            k_peak = k + 1;
        }
    }
    (k_peak, max_d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_short_zero_breaks() {
        let r = n_structural_breaks(&[]);
        assert_eq!(r.count, 0);
        assert!(r.locations.is_empty());
        assert!(r.magnitudes.is_empty());

        let r = n_structural_breaks(&[0.01, -0.01, 0.02]);
        assert_eq!(r.count, 0);
    }

    #[test]
    fn constant_variance_zero_breaks() {
        let rets: Vec<f64> = (0..500)
            .map(|i| if i % 2 == 0 { 0.001 } else { -0.001 })
            .collect();
        let r = n_structural_breaks(&rets);
        assert_eq!(r.count, 0);
        assert_eq!(r.n_observations, 500);
    }

    #[test]
    fn regime_shift_detected_with_location() {
        // Shift at index 200 of a 400-point series.
        let mut rets = Vec::with_capacity(400);
        for i in 0..200 {
            rets.push(if i % 2 == 0 { 0.0001 } else { -0.0001 });
        }
        for i in 0..200 {
            rets.push(if i % 2 == 0 { 0.05 } else { -0.05 });
        }
        let r = n_structural_breaks(&rets);
        assert!(r.count >= 1, "expected ≥ 1 break, got {}", r.count);
        assert_eq!(r.locations.len(), r.count as usize);
        assert_eq!(r.magnitudes.len(), r.count as usize);
        // The strongest break should be near index 200.
        let strongest = r.locations[0] as i64;
        assert!(
            (strongest - 200).abs() < 30,
            "strongest break at {strongest}, expected ~200"
        );
    }

    #[test]
    fn magnitudes_sorted_descending() {
        // Triple regime with shifts near indices 200 and 400.
        let mut rets = Vec::with_capacity(600);
        for _ in 0..200 {
            rets.push(0.0001);
        }
        for _ in 0..200 {
            rets.push(0.05);
        }
        for _ in 0..200 {
            rets.push(0.0001);
        }
        for i in (0..rets.len()).step_by(2) {
            rets[i] = -rets[i];
        }
        let r = n_structural_breaks(&rets);
        assert!(r.count >= 2);
        // Magnitudes must be non-increasing.
        for w in r.magnitudes.windows(2) {
            assert!(w[0] >= w[1], "not sorted: {} then {}", w[0], w[1]);
        }
        // All magnitudes should exceed the 5% critical value.
        for &m in &r.magnitudes {
            assert!(m > 1.358);
        }
    }

    #[test]
    fn tighter_critical_value_finds_fewer_or_equal() {
        let mut rets = Vec::with_capacity(400);
        for i in 0..200 {
            rets.push(if i % 2 == 0 { 0.001 } else { -0.001 });
        }
        for i in 0..200 {
            rets.push(if i % 2 == 0 { 0.003 } else { -0.003 });
        }
        let strict = n_structural_breaks_with_params(&rets, 1.628, 5); // 1%
        let loose = n_structural_breaks_with_params(&rets, 1.036, 5); // 10%
        assert!(
            loose.count >= strict.count,
            "loose {} < strict {}",
            loose.count,
            strict.count
        );
    }

    #[test]
    fn nan_run_does_not_inject_false_break() {
        let mut rets = vec![0.001; 400];
        for i in (0..rets.len()).step_by(2) {
            rets[i] = -0.001;
        }
        for i in 100..150 {
            rets[i] = f64::NAN;
        }
        let r = n_structural_breaks(&rets);
        assert_eq!(r.count, 0, "NaN run injected a false break");
        // n_observations should reflect the finite subset.
        assert_eq!(r.n_observations, 350);
    }

    #[test]
    fn scattered_non_finites_do_not_change_count() {
        let base_finite: Vec<f64> = (0..400)
            .map(|i| if i % 2 == 0 { 0.001 } else { -0.001 })
            .collect();
        let mut with_holes = base_finite.clone();
        for &k in &[17_usize, 42, 99, 200, 333] {
            with_holes[k] = f64::NAN;
        }
        assert_eq!(n_structural_breaks(&base_finite).count, 0);
        assert_eq!(n_structural_breaks(&with_holes).count, 0);
    }

    #[test]
    fn min_spacing_merges_close_candidates() {
        // Same double-shift as above, but with min_spacing forced wider
        // than the spacing between shift points should not explode into
        // more breaks than the segment can support.
        let mut rets = Vec::with_capacity(400);
        for i in 0..200 {
            rets.push(if i % 2 == 0 { 0.0001 } else { -0.0001 });
        }
        for i in 0..200 {
            rets.push(if i % 2 == 0 { 0.05 } else { -0.05 });
        }
        let n_tight = n_structural_breaks_with_params(&rets, 1.358, 2);
        let n_wide = n_structural_breaks_with_params(&rets, 1.358, 50);
        // Tighter spacing allows at least as many breaks as wider spacing.
        assert!(
            n_tight.count >= n_wide.count,
            "tight {} < wide {}",
            n_tight.count,
            n_wide.count
        );
    }

    #[test]
    #[should_panic(expected = "critical_value")]
    fn panics_on_bad_critical_value() {
        let _ = n_structural_breaks_with_params(&[0.01, 0.02, 0.03, 0.04, 0.05], 0.0, 2);
    }

    #[test]
    #[should_panic(expected = "min_spacing")]
    fn panics_on_zero_min_spacing() {
        let _ = n_structural_breaks_with_params(&[0.01, 0.02, 0.03, 0.04, 0.05], 1.358, 0);
    }
}
