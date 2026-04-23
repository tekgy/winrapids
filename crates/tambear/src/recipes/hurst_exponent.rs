//! Hurst exponent via rescaled-range (R/S) analysis.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! block-partition gather + per-block mean/range/std accumulates + a
//! log-log OLS regression. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! The Hurst exponent `H` measures long-range dependence in a time
//! series. Rescaled-range analysis (Mandelbrot & Wallis 1969) partitions
//! the series into non-overlapping blocks of size `w`, computes the
//! ratio of the range of cumulative mean-deviations to the block
//! standard deviation for each block, averages across blocks at that
//! `w`, and regresses `log E[R/S]` against `log w`. The slope is `H`:
//!
//! ```text
//! For each block b of size w:
//!   μ_b         =  (1/w) · Σ x[b,t]
//!   y[b,t]      =  Σ_{s≤t} (x[b,s] − μ_b)
//!   R_b         =  max(y[b,t]) − min(y[b,t])
//!   S_b         =  sample_std(x[b,·])        // uses (w-1) denominator
//!   RS_b        =  R_b / S_b                 // undefined if S_b = 0
//!
//! E[R/S](w)     =  mean over blocks of RS_b
//! log E[R/S]    =  H · log(w) + c            // log-log regression
//! ```
//!
//! Interpretation:
//! - `H ≈ 0.5` — random walk (no memory).
//! - `H > 0.5` — persistent (trending); large moves tend to follow
//!   large moves.
//! - `H < 0.5` — anti-persistent (mean-reverting).
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` the per-
//! hour `hurst_exponent` header field is computed on log returns. Window
//! sizes are geometric (×1.5 per step, starting at 10) up to `n/2` —
//! this matches the implementation below and gives 5–7 points for a
//! typical hour of SIP return data.
//!
//! # Reference
//!
//! Hurst, H. E. (1951). Long-term storage capacity of reservoirs.
//! *Transactions of the American Society of Civil Engineers* 116:
//! 770–799.
//!
//! Mandelbrot, B. B., & Wallis, J. R. (1969). Computer experiments with
//! fractional Gaussian noises. *Water Resources Research* 5(1):
//! 228–241.
//!
//! # Composition
//!
//! - **Block partition** — `gather(data, Addressing::BlockRange{w, b})`
//!   once the block-gather addressing lands; today implemented as
//!   slice indexing.
//! - **Per-block mean** — `accumulate(block, All, Op::Add) / w`.
//! - **Cumulative-deviation scan** — `accumulate(block, Prefix,
//!   Op::Add, expr=(v − μ))`.
//! - **Range reduction** — `accumulate(cum_dev, All, Op::Max) −
//!   accumulate(cum_dev, All, Op::Min)`.
//! - **Sample std** — `sqrt(accumulate(block, All, Op::Add,
//!   expr=Custom("(v−μ)²")) / (w − 1))`.
//! - **Log-log OLS** — `linear_algebra::ols_slope` over
//!   `(log(w), log(E[R/S]))` pairs.
//!
//! # NaN/Inf policy
//!
//! Follows the standard accumulate-layer contract: `!is_finite(x) →
//! skip` per element (here at the block level — a block containing
//! any non-finite value contributes no R/S point at its window size
//! rather than poisoning the entire estimate).
//!
//! - Input length `< 20` → returns NaN (too short for meaningful
//!   block-range statistics).
//! - A block with any non-finite element → that block is skipped at
//!   its window size. Other blocks at the same `w` still contribute.
//! - All blocks at some `w` skipped or zero-variance → that `w`
//!   contributes no point to the regression; if fewer than 2 (w, R/S)
//!   pairs survive, the result is NaN.
//!
//! # Default parameters
//!
//! - `min_block` — 10. Smallest block size for R/S analysis. Smaller
//!   blocks have noisy R/S estimates.
//! - `growth` — 1.5. Multiplicative block-size step (geometric
//!   spacing). Controls the density of (w, R/S) pairs on the log axis.
//!
//! Both are exposed as parameters on `hurst_exponent_with_params`;
//! `hurst_exponent` uses the SIP defaults.

/// SIP-default Hurst exponent via rescaled-range analysis.
///
/// Uses `min_block = 10` and geometric block-size growth of 1.5.
pub fn hurst_exponent(data: &[f64]) -> f64 {
    hurst_exponent_with_params(data, 10, 1.5)
}

/// Hurst exponent with configurable block-size schedule.
///
/// `min_block` is the smallest window considered. `growth > 1.0` is
/// the multiplicative step between consecutive block sizes.
///
/// # Panics
///
/// Panics if `min_block < 2` or `growth <= 1.0`.
pub fn hurst_exponent_with_params(data: &[f64], min_block: usize, growth: f64) -> f64 {
    assert!(
        min_block >= 2,
        "hurst_exponent: min_block must be >= 2, got {min_block}"
    );
    assert!(
        growth > 1.0 && growth.is_finite(),
        "hurst_exponent: growth must be > 1.0 and finite, got {growth}"
    );

    let n = data.len();
    if n < 20 {
        return f64::NAN;
    }

    let max_block = n / 2;
    let mut log_ws: Vec<f64> = Vec::new();
    let mut log_rs: Vec<f64> = Vec::new();

    let mut w = min_block;
    while w <= max_block {
        let n_blocks = n / w;
        if n_blocks < 1 {
            break;
        }

        // Per-block R/S → average across blocks at this w.
        let mut rs_sum: f64 = 0.0;
        let mut rs_count: usize = 0;
        for b in 0..n_blocks {
            let start = b * w;
            let block = &data[start..start + w];
            // Per-block skip: any non-finite element inside this block
            // means the cumulative-deviation sum is undefined, so the
            // block contributes nothing at this window size. Other
            // blocks at the same w still participate.
            if block.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let mu = crate::math::mean(block);
            if !mu.is_finite() {
                continue;
            }
            let mut cum: f64 = 0.0;
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            let mut ss: f64 = 0.0;
            for &v in block {
                let d = v - mu;
                cum += d;
                if cum < lo {
                    lo = cum;
                }
                if cum > hi {
                    hi = cum;
                }
                ss += d * d;
            }
            let range = hi - lo;
            // Sample std with (w − 1) denominator.
            let std_dev = (ss / (w - 1) as f64).sqrt();
            if std_dev > 0.0 && range.is_finite() {
                rs_sum += range / std_dev;
                rs_count += 1;
            }
        }

        if rs_count > 0 {
            let rs_avg = rs_sum / rs_count as f64;
            if rs_avg > 0.0 && rs_avg.is_finite() {
                log_ws.push((w as f64).ln());
                log_rs.push(rs_avg.ln());
            }
        }

        // Geometric progression of block sizes.
        let next_w = ((w as f64) * growth).ceil() as usize;
        w = if next_w > w { next_w } else { w + 1 };
    }

    if log_ws.len() < 2 {
        return f64::NAN;
    }

    crate::linear_algebra::ols_slope(&log_ws, &log_rs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_series_nan() {
        let short: Vec<f64> = (0..19).map(|i| i as f64).collect();
        assert!(hurst_exponent(&short).is_nan());
    }

    #[test]
    fn isolated_non_finite_does_not_poison_estimate() {
        // Per the contract: a single NaN should be SKIPPED at the block
        // level, not poison the whole series. A 500-point linear ramp
        // with one NaN should still produce a finite Hurst estimate
        // (roughly equal to the clean-ramp value).
        let clean: Vec<f64> = (0..500).map(|i| i as f64).collect();
        let mut with_nan = clean.clone();
        with_nan[42] = f64::NAN;
        let h_clean = hurst_exponent(&clean);
        let h_nan = hurst_exponent(&with_nan);
        assert!(h_clean.is_finite(), "clean H is NaN");
        assert!(h_nan.is_finite(), "NaN-contaminated H should still be finite");
        // The two estimates will differ slightly because one block at
        // each window size is dropped, but should be in the same
        // ballpark.
        assert!(
            (h_clean - h_nan).abs() < 0.3,
            "estimates too different: clean={h_clean} nan={h_nan}"
        );
    }

    #[test]
    fn all_non_finite_is_nan() {
        // If every element is non-finite, no blocks survive → NaN.
        let x = vec![f64::NAN; 500];
        assert!(hurst_exponent(&x).is_nan());
    }

    #[test]
    fn constant_series_nan() {
        // All equal → zero std in every block → no regression points.
        let x = vec![7.0; 200];
        assert!(hurst_exponent(&x).is_nan());
    }

    #[test]
    fn iid_noise_near_half() {
        // IID zero-mean noise (the INCREMENTS of a random walk) → no
        // memory → H ≈ 0.5 under R/S. Note: applying R/S to a random-
        // walk TRAJECTORY gives H → 1 by construction (pure trend),
        // which is not what this test checks.
        let n = 2000;
        let mut seed: u64 = 0xdeadbeefcafebabe;
        let mut x = Vec::with_capacity(n);
        for _ in 0..n {
            // xorshift64* for a deterministic pseudo-random sequence.
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64; // in [0, 1)
            x.push(u - 0.5);
        }
        let h = hurst_exponent(&x);
        // With 2000 IID samples we expect H within ~±0.15 of 0.5.
        assert!(h.is_finite(), "H is NaN");
        assert!((h - 0.5).abs() < 0.25, "iid noise H = {h}");
    }

    #[test]
    fn random_walk_trajectory_above_half() {
        // The cumulative-sum of IID noise is a random walk TRAJECTORY.
        // R/S on the trajectory is dominated by the √n range growth
        // of Brownian motion, producing H ≈ 1 rather than 0.5 —
        // verify we see a value well above 0.5.
        let n = 2000;
        let mut seed: u64 = 0x0123456789abcdef;
        let mut x = Vec::with_capacity(n);
        let mut cum = 0.0_f64;
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            cum += u - 0.5;
            x.push(cum);
        }
        let h = hurst_exponent(&x);
        assert!(h.is_finite(), "H is NaN");
        assert!(h > 0.7, "random walk trajectory H = {h} should be well above 0.5");
    }

    #[test]
    fn trending_series_above_half() {
        // Strong linear trend dominates the rescaled range → H should
        // land well above 0.5 (persistent).
        let n = 500;
        let x: Vec<f64> = (0..n).map(|i| i as f64 + (i as f64 * 0.01).sin()).collect();
        let h = hurst_exponent(&x);
        assert!(h.is_finite(), "H is NaN");
        assert!(h > 0.5, "trending H = {h} should be > 0.5");
    }

    #[test]
    fn mean_reverting_below_half() {
        // Strong alternation → anti-persistent → H < 0.5.
        let n = 500;
        let x: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let h = hurst_exponent(&x);
        assert!(h.is_finite(), "H is NaN");
        assert!(h < 0.5, "alternating H = {h} should be < 0.5");
    }

    #[test]
    fn output_in_reasonable_range() {
        // On any reasonable input the Hurst estimate should land in
        // [-1, 2] (R/S can occasionally produce slightly-out-of-[0,1]
        // finite-sample estimates, especially with few points).
        let xs: Vec<Vec<f64>> = vec![
            (0..200).map(|i| i as f64 * 0.01).collect(),
            (0..200).map(|i| (i as f64).sin()).collect(),
        ];
        for x in xs {
            let h = hurst_exponent(&x);
            if h.is_finite() {
                assert!(
                    (-1.0..=2.0).contains(&h),
                    "H out of reasonable range: {h}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "min_block")]
    fn panics_on_tiny_min_block() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let _ = hurst_exponent_with_params(&x, 1, 1.5);
    }

    #[test]
    #[should_panic(expected = "growth")]
    fn panics_on_bad_growth() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let _ = hurst_exponent_with_params(&x, 10, 1.0);
    }
}
