//! Family 19 — Shape signature analysis.
//!
//! Covers fintek leaf: `shape` (K02P08C01R01).
//!
//! 21 scalar features from returns describing bin shape:
//!   DO01-DO06: monotonicity group
//!   DO07-DO12: extrema group
//!   DO13-DO18: gradient stats
//!   DO19-DO21: curvature group

const MIN_N: usize = 10;

/// Shape feature result matching fintek's `shape.rs` (K02P08C01R01).
///
/// All 21 scalar features derived from log-return series.
#[derive(Debug, Clone)]
pub struct ShapeResult {
    // Monotonicity
    pub monotonicity_score: f64,
    pub fraction_up: f64,
    pub longest_run: f64,
    pub n_segments: f64,
    pub mean_segment_length: f64,
    pub monotonicity_index: f64,
    // Extrema
    pub n_extrema: f64,
    pub n_maxima: f64,
    pub n_minima: f64,
    pub extrema_rate: f64,
    pub mean_peak_to_trough: f64,
    pub mean_trough_to_peak: f64,
    // Gradient stats
    pub gradient_mean: f64,
    pub gradient_std: f64,
    pub gradient_skewness: f64,
    pub gradient_kurtosis: f64,
    pub gradient_max: f64,
    pub gradient_min: f64,
    // Curvature
    pub mean_laplacian: f64,
    pub laplacian_std: f64,
    pub n_inflection_points: f64,
}

impl ShapeResult {
    pub fn nan() -> Self {
        Self {
            monotonicity_score: f64::NAN,
            fraction_up: f64::NAN,
            longest_run: f64::NAN,
            n_segments: f64::NAN,
            mean_segment_length: f64::NAN,
            monotonicity_index: f64::NAN,
            n_extrema: f64::NAN,
            n_maxima: f64::NAN,
            n_minima: f64::NAN,
            extrema_rate: f64::NAN,
            mean_peak_to_trough: f64::NAN,
            mean_trough_to_peak: f64::NAN,
            gradient_mean: f64::NAN,
            gradient_std: f64::NAN,
            gradient_skewness: f64::NAN,
            gradient_kurtosis: f64::NAN,
            gradient_max: f64::NAN,
            gradient_min: f64::NAN,
            mean_laplacian: f64::NAN,
            laplacian_std: f64::NAN,
            n_inflection_points: f64::NAN,
        }
    }
}

/// Compute 21 shape features from a return series.
///
/// `returns`: pre-computed log returns (prices already differenced).
/// Returns `ShapeResult::nan()` if `returns.len() < 10`.
pub fn shape(returns: &[f64]) -> ShapeResult {
    let n = returns.len();
    if n < MIN_N { return ShapeResult::nan(); }

    // === Monotonicity (DO01-DO06) ===
    let mut n_pos = 0u32;
    let mut n_neg = 0u32;
    let mut longest_run = 1u32;
    let mut current_run = 1u32;
    let mut n_segments = 1u32;
    let mut cumsum_sign: i64 = 0;

    for j in 0..n {
        let sign: i32 = if returns[j] > 0.0 { 1 } else if returns[j] < 0.0 { -1 } else { 0 };
        if sign > 0 { n_pos += 1; }
        if sign < 0 { n_neg += 1; }
        cumsum_sign += sign as i64;

        if j > 0 {
            let prev_sign: i32 = if returns[j-1] > 0.0 { 1 } else if returns[j-1] < 0.0 { -1 } else { 0 };
            if sign == prev_sign && sign != 0 {
                current_run += 1;
            } else {
                longest_run = longest_run.max(current_run);
                current_run = 1;
                if sign != prev_sign { n_segments += 1; }
            }
        }
    }
    longest_run = longest_run.max(current_run);

    let monotonicity_score = (n_pos as f64 - n_neg as f64) / n as f64;
    let fraction_up = n_pos as f64 / n as f64;
    let mean_segment_length = n as f64 / n_segments as f64;
    let monotonicity_index = cumsum_sign.unsigned_abs() as f64 / n as f64;

    // === Extrema (DO07-DO12) ===
    let mut n_maxima = 0u32;
    let mut n_minima = 0u32;
    let mut peak_to_trough_sum = 0.0f64;
    let mut trough_to_peak_sum = 0.0f64;
    let mut n_p2t = 0u32;
    let mut n_t2p = 0u32;
    let mut last_extremum_val = returns[0];
    let mut last_was_peak = false;
    let mut have_extremum = false;

    for j in 1..n {
        let is_max = returns[j-1] > 0.0 && returns[j] < 0.0;
        let is_min = returns[j-1] < 0.0 && returns[j] > 0.0;

        if is_max {
            n_maxima += 1;
            if have_extremum && !last_was_peak {
                trough_to_peak_sum += (returns[j] - last_extremum_val).abs();
                n_t2p += 1;
            }
            last_extremum_val = returns[j];
            last_was_peak = true;
            have_extremum = true;
        } else if is_min {
            n_minima += 1;
            if have_extremum && last_was_peak {
                peak_to_trough_sum += (returns[j] - last_extremum_val).abs();
                n_p2t += 1;
            }
            last_extremum_val = returns[j];
            last_was_peak = false;
            have_extremum = true;
        }
    }

    let n_extrema = n_maxima + n_minima;
    let extrema_rate = n_extrema as f64 / n as f64;
    let mean_peak_to_trough = if n_p2t > 0 { peak_to_trough_sum / n_p2t as f64 } else { 0.0 };
    let mean_trough_to_peak = if n_t2p > 0 { trough_to_peak_sum / n_t2p as f64 } else { 0.0 };

    // === Gradient stats (DO13-DO18) ===
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut gmax = f64::NEG_INFINITY;
    let mut gmin = f64::INFINITY;
    for &r in returns {
        sum += r; sum2 += r * r;
        if r > gmax { gmax = r; }
        if r < gmin { gmin = r; }
    }
    let mean = sum / n as f64;
    let var = (sum2 / n as f64 - mean * mean).max(0.0);
    let std = var.sqrt();

    let mut m3 = 0.0f64;
    let mut m4 = 0.0f64;
    for &r in returns {
        let d = r - mean;
        let d2 = d * d;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m3 /= n as f64;
    m4 /= n as f64;
    let skew = if std > 1e-30 { m3 / (std * std * std) } else { 0.0 };
    let kurt = if std > 1e-30 { m4 / (var * var) - 3.0 } else { 0.0 };

    // === Curvature / second differences (DO19-DO21) ===
    let (mean_laplacian, laplacian_std, n_inflection_points) = if n >= 2 {
        let mut lap_sum = 0.0f64;
        let mut lap_sum2 = 0.0f64;
        let mut n_inflection = 0u32;
        let n_sd = n - 1;

        for j in 0..n_sd {
            let sd = returns[j+1] - returns[j];
            lap_sum += sd;
            lap_sum2 += sd * sd;

            if j > 0 {
                let prev_sd = returns[j] - returns[j-1];
                if (sd > 0.0 && prev_sd < 0.0) || (sd < 0.0 && prev_sd > 0.0) {
                    n_inflection += 1;
                }
            }
        }
        let lap_mean = lap_sum / n_sd as f64;
        let lap_var = (lap_sum2 / n_sd as f64 - lap_mean * lap_mean).max(0.0);
        (lap_mean, lap_var.sqrt(), n_inflection as f64)
    } else {
        (f64::NAN, f64::NAN, 0.0)
    };

    ShapeResult {
        monotonicity_score,
        fraction_up,
        longest_run: longest_run as f64,
        n_segments: n_segments as f64,
        mean_segment_length,
        monotonicity_index,
        n_extrema: n_extrema as f64,
        n_maxima: n_maxima as f64,
        n_minima: n_minima as f64,
        extrema_rate,
        mean_peak_to_trough,
        mean_trough_to_peak,
        gradient_mean: mean,
        gradient_std: std,
        gradient_skewness: skew,
        gradient_kurtosis: kurt,
        gradient_max: gmax,
        gradient_min: gmin,
        mean_laplacian,
        laplacian_std,
        n_inflection_points,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_returns(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = tambear::rng::Xoshiro256::new(seed);
        (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect()
    }

    #[test]
    fn shape_too_short() {
        let r = shape(&[1.0, 2.0, 3.0]);
        assert!(r.monotonicity_score.is_nan());
    }

    #[test]
    fn shape_white_noise_finite() {
        let returns = make_returns(100, 42);
        let r = shape(&returns);
        assert!(r.monotonicity_score.is_finite());
        assert!(r.fraction_up >= 0.0 && r.fraction_up <= 1.0);
        assert!(r.longest_run >= 1.0);
        assert!(r.n_segments >= 1.0);
        assert!(r.extrema_rate >= 0.0);
        assert!(r.gradient_std >= 0.0);
        assert!(r.laplacian_std >= 0.0);
        assert!(r.n_inflection_points >= 0.0);
    }

    #[test]
    fn shape_monotone_up() {
        // Strictly increasing returns → high fraction_up, few extrema
        let returns: Vec<f64> = (0..50).map(|i| (i as f64 + 1.0) * 0.001).collect();
        let r = shape(&returns);
        assert!(r.fraction_up > 0.9, "fraction_up should be > 0.9 for monotone up, got {}", r.fraction_up);
        assert!(r.monotonicity_score > 0.9, "monotonicity_score should be > 0.9, got {}", r.monotonicity_score);
    }

    #[test]
    fn shape_alternating() {
        // Alternating +/- → many extrema, near-zero monotonicity
        let returns: Vec<f64> = (0..50).map(|i| if i % 2 == 0 { 0.01 } else { -0.01 }).collect();
        let r = shape(&returns);
        assert!(r.extrema_rate > 0.5, "alternating series should have high extrema_rate, got {}", r.extrema_rate);
        assert!(r.monotonicity_score.abs() < 0.2, "alternating should have near-zero monotonicity, got {}", r.monotonicity_score);
    }

    #[test]
    fn shape_outputs_21_fields() {
        let returns = make_returns(50, 7);
        let r = shape(&returns);
        // Spot-check all 21 are finite (for white noise with n=50)
        assert!(r.monotonicity_score.is_finite());
        assert!(r.fraction_up.is_finite());
        assert!(r.longest_run.is_finite());
        assert!(r.n_segments.is_finite());
        assert!(r.mean_segment_length.is_finite());
        assert!(r.monotonicity_index.is_finite());
        assert!(r.n_extrema.is_finite());
        assert!(r.n_maxima.is_finite());
        assert!(r.n_minima.is_finite());
        assert!(r.extrema_rate.is_finite());
        assert!(r.gradient_mean.is_finite());
        assert!(r.gradient_std.is_finite());
        assert!(r.gradient_skewness.is_finite());
        assert!(r.gradient_kurtosis.is_finite());
        assert!(r.gradient_max.is_finite());
        assert!(r.gradient_min.is_finite());
        assert!(r.mean_laplacian.is_finite());
        assert!(r.laplacian_std.is_finite());
        assert!(r.n_inflection_points.is_finite());
    }
}
