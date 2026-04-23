//! Amihud (2002) illiquidity ratio — absolute return per unit of dollar
//! volume.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over
//! Kulisch-backed sums plus a scalar ratio. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! Amihud's illiquidity ratio measures the absolute return per dollar
//! of trading activity — a price-impact proxy that is large when small
//! volumes move the price a lot (illiquid) and small when large volumes
//! are absorbed with little price movement (liquid).
//!
//! Two common formulations:
//!
//! **Per-tick average.** For each trade `i` with return `r[i]` and
//! positive dollar volume `v[i]`:
//!
//! ```text
//! amihud  =  (1 / n) · Σ_{i : v[i] > 0}  |r[i]| / v[i]
//! ```
//!
//! **Window ratio (the SIP form).** Sum of absolute returns divided by
//! the sum of dollar volumes across a bucket / window:
//!
//! ```text
//! amihud  =  Σ |r[i]|  /  Σ v[i]
//!         =  sum_abs_ret  /  sum_pv
//! ```
//!
//! The two forms coincide exactly when every trade has the same dollar
//! volume; in general they emphasize different aspects of the same
//! price-impact idea. The window form is the one SIP uses because its
//! two inputs are already accumulated in the bucket header.
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` the per-
//! bucket `amihud_illiquidity` field is the window ratio
//! `sum_abs_ret / sum_pv`. Both numerator and denominator are existing
//! `TradePayload` columns, so this recipe contributes only the scalar
//! ratio — no new accumulation is needed in the per-bucket pass.
//!
//! # Reference
//!
//! Amihud, Y. (2002). Illiquidity and stock returns: cross-section and
//! time-series effects. *Journal of Financial Markets* 5(1): 31–56.
//!
//! # Composition
//!
//! - **Two Kulisch-backed sums** — `Σ|r|` and `Σv`. Lower to
//!   `accumulate(abs(log_ret), ByKey{bucket}, Op::Add, expr=Abs)` and
//!   `accumulate(usd_ref_value, ByKey{bucket}, Op::Add, expr=Value)`.
//!   For SIP both are already in the bucket header; the recipe reads
//!   them rather than re-accumulating.
//! - **Scalar ratio** — `Σ|r| / Σv` with an empty-bucket guard.
//!
//! # NaN/Inf policy
//!
//! - `sum_abs_ret < 0` or non-finite → returns NaN (data error
//!   upstream).
//! - `sum_pv <= 0` or non-finite → returns NaN (no dollar volume to
//!   normalize against).
//! - Empty input → returns NaN.
//!
//! The per-tick form skips trades with non-positive or non-finite
//! volume so a single zero-volume crossed trade does not poison the
//! window.
//!
//! # Default parameters
//!
//! None. Amihud's estimator has no tunable parameters at this level;
//! upstream scaling choices (e.g., multiplying by 10⁶ to express in
//! basis-points-per-million-dollars) are policy decisions handled by
//! the caller.

/// Amihud illiquidity from pre-computed window sums (the SIP form).
///
/// `sum_abs_ret = Σ|r[i]|` and `sum_pv = Σ v[i]`. Returns NaN if either
/// input is non-finite, `sum_abs_ret < 0`, or `sum_pv <= 0`.
#[inline]
pub fn amihud_illiquidity_from_sums(sum_abs_ret: f64, sum_pv: f64) -> f64 {
    if !sum_abs_ret.is_finite() || !sum_pv.is_finite() || sum_abs_ret < 0.0 || sum_pv <= 0.0 {
        return f64::NAN;
    }
    sum_abs_ret / sum_pv
}

/// Amihud illiquidity as the per-tick average `(1/n) · Σ |r[i]|/v[i]`.
///
/// Returns NaN if no trade has positive-finite volume (the average is
/// undefined). Trades with non-positive, zero, or non-finite volume, or
/// non-finite return, are skipped in both the sum and the count.
///
/// # Panics
///
/// Panics if `returns` and `volumes` have mismatched lengths.
pub fn amihud_illiquidity_per_tick(returns: &[f64], volumes: &[f64]) -> f64 {
    assert_eq!(
        returns.len(),
        volumes.len(),
        "amihud_illiquidity_per_tick: returns and volumes must match length"
    );
    use crate::primitives::specialist::kulisch_accumulator::KulischAccumulator;

    let mut sum = KulischAccumulator::new();
    let mut n: u64 = 0;
    for (r, v) in returns.iter().zip(volumes.iter()) {
        if !r.is_finite() || !v.is_finite() || *v <= 0.0 {
            continue;
        }
        sum.add_f64(r.abs() / v);
        n += 1;
    }

    if n == 0 {
        return f64::NAN;
    }
    sum.to_f64() / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_sums_basic_ratio() {
        // sum_abs_ret = 0.05, sum_pv = 1000 → 0.05 / 1000 = 5e-5
        let a = amihud_illiquidity_from_sums(0.05, 1000.0);
        assert!((a - 5e-5).abs() < 1e-12);
    }

    #[test]
    fn from_sums_nan_on_invalid() {
        assert!(amihud_illiquidity_from_sums(f64::NAN, 100.0).is_nan());
        assert!(amihud_illiquidity_from_sums(0.01, f64::INFINITY).is_nan());
        assert!(amihud_illiquidity_from_sums(-0.01, 100.0).is_nan());
        assert!(amihud_illiquidity_from_sums(0.01, 0.0).is_nan());
        assert!(amihud_illiquidity_from_sums(0.01, -100.0).is_nan());
    }

    #[test]
    fn per_tick_average_closed_form() {
        // Three trades with uniform volume 100: |0.01|/100 = 1e-4,
        // |-0.02|/100 = 2e-4, |0.03|/100 = 3e-4. Average = 2e-4.
        let rets = vec![0.01, -0.02, 0.03];
        let vols = vec![100.0, 100.0, 100.0];
        let a = amihud_illiquidity_per_tick(&rets, &vols);
        assert!((a - 2e-4).abs() < 1e-12);
    }

    #[test]
    fn per_tick_skips_zero_volume_trades() {
        // One legitimate trade, one zero-volume trade. Average reflects
        // only the legit one.
        let rets = vec![0.01, 0.5];
        let vols = vec![100.0, 0.0];
        let a = amihud_illiquidity_per_tick(&rets, &vols);
        assert!((a - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn per_tick_skips_non_finite() {
        let rets = vec![0.01, f64::NAN, 0.02];
        let vols = vec![100.0, 100.0, 200.0];
        let a = amihud_illiquidity_per_tick(&rets, &vols);
        // |0.01|/100 = 1e-4, |0.02|/200 = 1e-4. Avg = 1e-4.
        assert!((a - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn per_tick_nan_when_no_eligible_trades() {
        let rets = vec![0.01, 0.02];
        let vols = vec![0.0, -10.0];
        assert!(amihud_illiquidity_per_tick(&rets, &vols).is_nan());
        assert!(amihud_illiquidity_per_tick(&[], &[]).is_nan());
    }

    #[test]
    fn uniform_volume_two_forms_agree() {
        // When all volumes equal, the per-tick average should match
        // sum_abs_ret / (n · v_const) = sum_abs_ret / sum_pv.
        let rets = vec![0.01, -0.02, 0.03, 0.04, -0.01];
        let vols = vec![250.0; 5];
        let sum_abs: f64 = rets.iter().map(|r: &f64| r.abs()).sum();
        let sum_pv: f64 = vols.iter().sum();

        let window = amihud_illiquidity_from_sums(sum_abs, sum_pv);
        let per_tick = amihud_illiquidity_per_tick(&rets, &vols);
        assert!((window - per_tick).abs() < 1e-12, "{window} vs {per_tick}");
    }

    #[test]
    fn illiquidity_monotonic_in_abs_returns() {
        // Same volume total; larger absolute returns → larger illiquidity.
        let a_small = amihud_illiquidity_from_sums(0.01, 1000.0);
        let a_big = amihud_illiquidity_from_sums(0.10, 1000.0);
        assert!(a_big > a_small);
    }

    #[test]
    fn illiquidity_inverse_in_volume() {
        // Same absolute returns; larger volume → smaller illiquidity.
        let a_low_v = amihud_illiquidity_from_sums(0.05, 1_000.0);
        let a_high_v = amihud_illiquidity_from_sums(0.05, 10_000.0);
        assert!(a_low_v > a_high_v);
    }

    #[test]
    fn non_negative_output() {
        // Amihud is always ≥ 0 by construction.
        let cases = [
            (0.01_f64, 100.0_f64),
            (0.0, 100.0),
            (1.0, 1.0),
            (1e-10, 1e10),
        ];
        for (s, v) in cases {
            let a = amihud_illiquidity_from_sums(s, v);
            assert!(a >= 0.0, "negative amihud {a} for ({s}, {v})");
        }
    }

    #[test]
    #[should_panic(expected = "match length")]
    fn per_tick_panics_on_mismatched_lengths() {
        let _ = amihud_illiquidity_per_tick(&[0.01, 0.02], &[100.0]);
    }
}
