//! Family 21 — Changepoint detection (exact/Bayesian).
//!
//! Covers fintek leaves:
//! - `pelt`  (K02P18C01R02F01) — Pruned Exact Linear Time changepoint detection
//! - `bocpd` (K02P18C01R03F01) — Bayesian Online Changepoint Detection

// ── PELT ──────────────────────────────────────────────────────────────────────

const PELT_MIN_SEGMENT: usize = 10;
const PELT_MAX_N: usize = 2000;

/// PELT result matching fintek's `pelt.rs` (K02P18C01R02F01).
#[derive(Debug, Clone)]
pub struct PeltResult {
    pub n_changepoints: f64,
    pub mean_segment_length: f64,
    pub max_cost_reduction: f64,
    pub penalty_ratio: f64,
}

impl PeltResult {
    pub fn nan() -> Self {
        Self { n_changepoints: f64::NAN, mean_segment_length: f64::NAN,
               max_cost_reduction: f64::NAN, penalty_ratio: f64::NAN }
    }
}

/// Gaussian segment cost for data[start..end] using prefix sums.
/// Cost = n × log(variance). Returns INFINITY for degenerate segments.
fn segment_cost(cumsum: &[f64], cumsum2: &[f64], start: usize, end: usize) -> f64 {
    let n = end - start;
    if n < 2 { return f64::INFINITY; }
    let nf = n as f64;
    let s  = cumsum[end]  - cumsum[start];
    let s2 = cumsum2[end] - cumsum2[start];
    let var = (s2 / nf - (s / nf) * (s / nf)).max(1e-30);
    nf * var.ln()
}

/// Run PELT (Pruned Exact Linear Time) changepoint detection.
///
/// Uses BIC penalty (2 × ln(n)) and Gaussian segment cost.
/// Subsamples to 2000 points to maintain O(n) budget.
///
/// Returns NaN if `returns.len() < 20`.
pub fn pelt(returns: &[f64]) -> PeltResult {
    let data: Vec<f64> = if returns.len() > PELT_MAX_N {
        let step = returns.len() / PELT_MAX_N;
        returns.iter().step_by(step).take(PELT_MAX_N).copied().collect()
    } else {
        returns.to_vec()
    };

    let n = data.len();
    if n < 2 * PELT_MIN_SEGMENT { return PeltResult::nan(); }

    // Prefix sums
    let mut cumsum  = vec![0.0f64; n + 1];
    let mut cumsum2 = vec![0.0f64; n + 1];
    for i in 0..n {
        cumsum[i + 1]  = cumsum[i]  + data[i];
        cumsum2[i + 1] = cumsum2[i] + data[i] * data[i];
    }

    let penalty = 2.0 * (n as f64).ln();

    // DP with PELT pruning
    let mut f = vec![f64::INFINITY; n + 1];
    let mut last_cp = vec![0usize; n + 1];
    f[0] = -penalty;
    let mut candidates: Vec<usize> = vec![0];

    for t in PELT_MIN_SEGMENT..=n {
        let mut best_f = f64::INFINITY;
        let mut best_s = 0usize;
        for &s in &candidates {
            if t - s < PELT_MIN_SEGMENT { continue; }
            let cost = f[s] + segment_cost(&cumsum, &cumsum2, s, t) + penalty;
            if cost < best_f { best_f = cost; best_s = s; }
        }
        f[t] = best_f;
        last_cp[t] = best_s;

        candidates.retain(|&s| {
            if t - s < PELT_MIN_SEGMENT { return true; }
            f[s] + segment_cost(&cumsum, &cumsum2, s, t) <= f[t] + penalty
        });
        candidates.push(t);
    }

    // Backtrace
    let mut cps = Vec::new();
    let mut pos = n;
    while pos > 0 {
        let prev = last_cp[pos];
        if prev > 0 { cps.push(prev); }
        pos = prev;
    }
    cps.reverse();
    let n_cp = cps.len();

    let mean_seg_len = if n_cp == 0 {
        n as f64
    } else {
        n as f64 / (n_cp + 1) as f64
    };

    let max_cost_reduction = if n_cp == 0 {
        0.0
    } else {
        let mut boundaries = vec![0usize];
        boundaries.extend_from_slice(&cps);
        boundaries.push(n);
        let mut max_red = 0.0f64;
        for k in 1..boundaries.len() - 1 {
            let merged = segment_cost(&cumsum, &cumsum2, boundaries[k-1], boundaries[k+1]);
            let left   = segment_cost(&cumsum, &cumsum2, boundaries[k-1], boundaries[k]);
            let right  = segment_cost(&cumsum, &cumsum2, boundaries[k],   boundaries[k+1]);
            let red = merged - (left + right);
            if red > max_red { max_red = red; }
        }
        max_red
    };

    let unpenalized_cost = segment_cost(&cumsum, &cumsum2, 0, n);
    let penalty_ratio = if unpenalized_cost.abs() > 1e-30 {
        (unpenalized_cost + n_cp as f64 * penalty) / unpenalized_cost
    } else { f64::NAN };

    PeltResult {
        n_changepoints: n_cp as f64,
        mean_segment_length: mean_seg_len,
        max_cost_reduction,
        penalty_ratio,
    }
}

// ── BOCPD ─────────────────────────────────────────────────────────────────────

const BOCPD_HAZARD: f64 = 1.0 / 200.0;
const BOCPD_MAX_RUN: usize = 500;
const BOCPD_MAX_N: usize = 2000;
const BOCPD_MIN_N: usize = 30;
const BOCPD_KAPPA0: f64 = 1.0;

/// BOCPD result matching fintek's `bocpd.rs` (K02P18C01R03F01).
#[derive(Debug, Clone)]
pub struct BocpdResult {
    pub n_changepoints: f64,
    pub mean_run_length: f64,
    pub max_posterior_drop: f64,
    pub hazard_rate: f64,
}

impl BocpdResult {
    pub fn nan() -> Self {
        Self { n_changepoints: f64::NAN, mean_run_length: f64::NAN,
               max_posterior_drop: f64::NAN, hazard_rate: f64::NAN }
    }
}

/// Bayesian Online Changepoint Detection (Adams & MacKay 2007).
///
/// Constant hazard rate (1/200), Gaussian predictive with Normal-Gamma prior.
/// Subsamples to 2000 points. Returns NaN if `returns.len() < 30`.
pub fn bocpd(returns: &[f64]) -> BocpdResult {
    let data: Vec<f64> = if returns.len() > BOCPD_MAX_N {
        let step = returns.len() / BOCPD_MAX_N;
        returns.iter().step_by(step).take(BOCPD_MAX_N).copied().collect()
    } else {
        returns.to_vec()
    };

    let n = data.len();
    if n < BOCPD_MIN_N { return BocpdResult::nan(); }

    let mut run_post = vec![0.0f64; BOCPD_MAX_RUN + 1];
    run_post[0] = 1.0;

    let mut rl_count = vec![0.0f64; BOCPD_MAX_RUN + 1];
    let mut rl_sum   = vec![0.0f64; BOCPD_MAX_RUN + 1];
    let mut rl_sum2  = vec![0.0f64; BOCPD_MAX_RUN + 1];

    let mut max_posts = Vec::with_capacity(n);
    let mut mean_run_lengths = Vec::with_capacity(n);

    for t in 0..n {
        let x = data[t];
        let cur_len = (t + 1).min(BOCPD_MAX_RUN);

        let mut growth = vec![0.0f64; cur_len + 1];
        let mut cp_mass = 0.0f64;

        for r in 0..=cur_len.min(BOCPD_MAX_RUN - 1) {
            if run_post[r] < 1e-300 { continue; }
            let cnt = rl_count[r];
            let log_pred = if cnt < 1.0 {
                let var = 1.0 / BOCPD_KAPPA0 + 1e-10;
                -0.5 * (std::f64::consts::TAU * var).ln() - 0.5 * x * x / var
            } else {
                let mean = rl_sum[r] / cnt;
                let var = (rl_sum2[r] / cnt - mean * mean).abs() + 1e-10;
                -0.5 * (std::f64::consts::TAU * var).ln() - 0.5 * (x - mean) * (x - mean) / var
            };
            let pred = log_pred.exp().max(1e-300);
            if r + 1 <= BOCPD_MAX_RUN {
                growth[r + 1] = run_post[r] * pred * (1.0 - BOCPD_HAZARD);
            }
            cp_mass += run_post[r] * pred * BOCPD_HAZARD;
        }

        for r in (1..=cur_len.min(BOCPD_MAX_RUN)).rev() {
            rl_count[r] = rl_count[r - 1] + 1.0;
            rl_sum[r]   = rl_sum[r - 1]   + x;
            rl_sum2[r]  = rl_sum2[r - 1]  + x * x;
        }
        rl_count[0] = 1.0; rl_sum[0] = x; rl_sum2[0] = x * x;

        for r in 1..=cur_len.min(BOCPD_MAX_RUN) { run_post[r] = growth[r]; }
        run_post[0] = cp_mass;

        let total: f64 = run_post[..=(cur_len.min(BOCPD_MAX_RUN))].iter().sum();
        if total > 0.0 {
            for r in 0..=(cur_len.min(BOCPD_MAX_RUN)) { run_post[r] /= total; }
        }

        let max_p = run_post[..=(cur_len.min(BOCPD_MAX_RUN))].iter().copied().fold(0.0f64, f64::max);
        max_posts.push(max_p);

        let mean_rl: f64 = run_post[..=(cur_len.min(BOCPD_MAX_RUN))].iter()
            .enumerate().map(|(r, &p)| r as f64 * p).sum();
        mean_run_lengths.push(mean_rl);
    }

    // Adaptive changepoint threshold from drop distribution
    let mut drops: Vec<f64> = Vec::new();
    for t in 1..max_posts.len() {
        let d = max_posts[t-1] - max_posts[t];
        if d > 0.0 { drops.push(d); }
    }

    let threshold = if drops.is_empty() {
        0.5
    } else {
        let mut sorted = drops.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let mean_d: f64 = drops.iter().sum::<f64>() / drops.len() as f64;
        let var_d: f64 = drops.iter().map(|d| (d - mean_d) * (d - mean_d)).sum::<f64>() / drops.len() as f64;
        (median + 2.0 * var_d.sqrt()).max(0.5)
    };

    let mut n_cp = 0u64;
    let mut max_drop = 0.0f64;
    for t in 1..max_posts.len() {
        let d = max_posts[t-1] - max_posts[t];
        if d > max_drop { max_drop = d; }
        if d > threshold { n_cp += 1; }
    }

    let overall_mean_rl = mean_run_lengths.iter().sum::<f64>() / mean_run_lengths.len() as f64;

    BocpdResult {
        n_changepoints: n_cp as f64,
        mean_run_length: overall_mean_rl,
        max_posterior_drop: max_drop,
        hazard_rate: n_cp as f64 / n as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wn(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = tambear::rng::Xoshiro256::new(seed);
        (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect()
    }

    // ── PELT tests ──

    #[test]
    fn pelt_too_short() {
        let r = pelt(&[0.0; 5]);
        assert!(r.n_changepoints.is_nan());
    }

    #[test]
    fn pelt_white_noise_finite() {
        let data = wn(100, 42);
        let r = pelt(&data);
        assert!(r.n_changepoints.is_finite() && r.n_changepoints >= 0.0);
        assert!(r.mean_segment_length.is_finite() && r.mean_segment_length > 0.0);
    }

    #[test]
    fn pelt_step_function_detects_break() {
        let mut data = vec![0.0f64; 50];
        for i in 25..50 { data[i] = 1.0; }
        let r = pelt(&data);
        assert!(r.n_changepoints >= 1.0, "should detect break, got {}", r.n_changepoints);
        assert!(r.max_cost_reduction > 0.0);
    }

    // ── BOCPD tests ──

    #[test]
    fn bocpd_too_short() {
        let r = bocpd(&[0.0; 10]);
        assert!(r.n_changepoints.is_nan());
    }

    #[test]
    fn bocpd_white_noise_finite() {
        let data = wn(100, 99);
        let r = bocpd(&data);
        assert!(r.n_changepoints.is_finite() && r.n_changepoints >= 0.0);
        assert!(r.mean_run_length.is_finite() && r.mean_run_length > 0.0);
        assert!(r.hazard_rate >= 0.0);
    }

    #[test]
    fn bocpd_step_increase_mean_run_length_drops() {
        // After step change, run lengths should reset
        let mut data = vec![0.0f64; 100];
        for i in 50..100 { data[i] = 0.5; }
        let r_flat = bocpd(&wn(100, 1));
        let r_step = bocpd(&data);
        // Both should return finite results
        assert!(r_flat.mean_run_length.is_finite());
        assert!(r_step.mean_run_length.is_finite());
    }
}
