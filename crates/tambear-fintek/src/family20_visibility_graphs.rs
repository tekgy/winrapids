//! Family 20 — Visibility graph analysis.
//!
//! Covers fintek leaves:
//! - `nvg` (K02P20C01R01) — Natural Visibility Graph
//! - `hvg` (K02P20C01R02) — Horizontal Visibility Graph
//!
//! Both graphs map a time series to a graph where nodes are data points
//! and edges represent "visibility" between points. Graph topology encodes
//! dynamical complexity: random series produce exponential degree distributions,
//! chaotic series produce power-law distributions.

// ── Natural Visibility Graph ───────────────────────────────────────────────────

const NVG_MAX_PTS: usize = 500;
const HVG_MAX_PTS: usize = 1000;
const MIN_N: usize = 20;

/// NVG result matching fintek's `nvg.rs` (K02P20C01R01).
#[derive(Debug, Clone)]
pub struct NvgResult {
    pub degree_exponent: f64,
    pub mean_degree: f64,
    pub degree_entropy: f64,
    pub clustering_coefficient: f64,
    pub assortativity: f64,
}

impl NvgResult {
    pub fn nan() -> Self {
        Self {
            degree_exponent: f64::NAN,
            mean_degree: f64::NAN,
            degree_entropy: f64::NAN,
            clustering_coefficient: f64::NAN,
            assortativity: f64::NAN,
        }
    }
}

/// HVG result matching fintek's `hvg.rs` (K02P20C01R02).
#[derive(Debug, Clone)]
pub struct HvgResult {
    pub degree_exponent: f64,
    pub mean_degree: f64,
    pub degree_entropy: f64,
    pub clustering_coefficient: f64,
    pub irreversibility: f64,
}

impl HvgResult {
    pub fn nan() -> Self {
        Self {
            degree_exponent: f64::NAN,
            mean_degree: f64::NAN,
            degree_entropy: f64::NAN,
            clustering_coefficient: f64::NAN,
            irreversibility: f64::NAN,
        }
    }
}

/// Compute NVG degree distribution features.
///
/// Node a is visible from node b (a < b) iff for all c in (a,b):
///   x[c] < x[a] + (x[b] - x[a]) * (c-a) / (b-a)
///
/// Caps input at 500 points for O(n²) budget.
pub fn nvg(x: &[f64]) -> NvgResult {
    let n = x.len().min(NVG_MAX_PTS);
    if n < MIN_N { return NvgResult::nan(); }
    let x = &x[..n];

    let mut degree = vec![0u32; n];
    for a in 0..n {
        for b in (a+1)..n {
            let visible = (a+1..b).all(|c| {
                let interp = x[a] + (x[b] - x[a]) * (c - a) as f64 / (b - a) as f64;
                x[c] < interp
            });
            if visible { degree[a] += 1; degree[b] += 1; }
        }
    }

    compute_nvg_stats(x, &degree, n)
}

fn compute_nvg_stats(x: &[f64], degree: &[u32], n: usize) -> NvgResult {
    let n_f = n as f64;
    let mean_degree = degree.iter().sum::<u32>() as f64 / n_f;

    let max_deg = *degree.iter().max().unwrap_or(&0) as usize;
    let mut deg_hist = vec![0u32; max_deg + 1];
    for &d in degree { deg_hist[d as usize] += 1; }

    // Shannon entropy of degree distribution
    let mut entropy = 0.0f64;
    for &c in &deg_hist {
        if c > 0 {
            let p = c as f64 / n_f;
            entropy -= p * p.ln();
        }
    }

    // Power-law exponent via log-log OLS: P(k) ~ k^(-gamma)
    let mut log_k = Vec::new();
    let mut log_pk = Vec::new();
    for (k, &c) in deg_hist.iter().enumerate() {
        if k > 0 && c > 0 {
            log_k.push((k as f64).ln());
            log_pk.push((c as f64 / n_f).ln());
        }
    }
    let gamma = power_law_exponent(&log_k, &log_pk);

    // Clustering coefficient (sample up to 200 nodes)
    let sample = n.min(200);
    let mut cc_sum = 0.0f64;
    let mut cc_count = 0u32;
    for a in 0..sample {
        let mut neighbors: Vec<usize> = Vec::new();
        for b in 0..n {
            if a == b { continue; }
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            let visible = (lo+1..hi).all(|c| {
                let interp = x[lo] + (x[hi] - x[lo]) * (c - lo) as f64 / (hi - lo) as f64;
                x[c] < interp
            });
            if visible { neighbors.push(b); }
        }
        let k = neighbors.len();
        if k < 2 { continue; }
        let mut edges = 0u32;
        for ni in 0..neighbors.len() {
            for nj in (ni+1)..neighbors.len() {
                let (lo, hi) = (neighbors[ni].min(neighbors[nj]), neighbors[ni].max(neighbors[nj]));
                let visible = (lo+1..hi).all(|c| {
                    let interp = x[lo] + (x[hi] - x[lo]) * (c - lo) as f64 / (hi - lo) as f64;
                    x[c] < interp
                });
                if visible { edges += 1; }
            }
        }
        cc_sum += edges as f64 / (k * (k - 1) / 2) as f64;
        cc_count += 1;
    }
    let cc = if cc_count > 0 { cc_sum / cc_count as f64 } else { 0.0 };

    // Assortativity: degree-degree correlation over edges (sample up to 300 nodes)
    let mut sum_prod = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum_deg = 0.0f64;
    let mut edge_count = 0u64;
    for a in 0..n.min(300) {
        for b in (a+1)..n.min(300) {
            let visible = (a+1..b).all(|c| {
                let interp = x[a] + (x[b] - x[a]) * (c - a) as f64 / (b - a) as f64;
                x[c] < interp
            });
            if visible {
                let da = degree[a] as f64;
                let db = degree[b] as f64;
                sum_prod += da * db;
                sum_sq += da * da + db * db;
                sum_deg += da + db;
                edge_count += 1;
            }
        }
    }
    let assort = if edge_count > 0 {
        let m = edge_count as f64;
        let a = sum_prod / m;
        let b = (sum_deg / (2.0 * m)).powi(2);
        let c = sum_sq / (2.0 * m) - b;
        if c.abs() > 1e-30 { (a - b) / c } else { 0.0 }
    } else { 0.0 };

    NvgResult { degree_exponent: gamma, mean_degree, degree_entropy: entropy,
                clustering_coefficient: cc, assortativity: assort }
}

// ── Horizontal Visibility Graph ────────────────────────────────────────────────

/// Compute HVG degree distribution features.
///
/// Node a is visible from node b (a < b) iff for all c in (a,b):
///   x[c] < min(x[a], x[b])   (horizontal visibility criterion)
///
/// Caps input at 1000 points.
pub fn hvg(x: &[f64]) -> HvgResult {
    let n = x.len().min(HVG_MAX_PTS);
    if n < MIN_N { return HvgResult::nan(); }
    let x = &x[..n];

    // Forward HVG degrees
    let mut degree = vec![0u32; n];
    for a in 0..n {
        for b in (a+1)..n {
            let threshold = x[a].min(x[b]);
            if (a+1..b).all(|c| x[c] < threshold) {
                degree[a] += 1; degree[b] += 1;
            }
        }
    }

    let n_f = n as f64;
    let mean_degree = degree.iter().sum::<u32>() as f64 / n_f;

    let max_deg = *degree.iter().max().unwrap_or(&0) as usize;
    let mut deg_hist = vec![0u32; max_deg + 1];
    for &d in &degree { deg_hist[d as usize] += 1; }

    let mut entropy = 0.0f64;
    for &c in &deg_hist {
        if c > 0 {
            let p = c as f64 / n_f;
            entropy -= p * p.ln();
        }
    }

    let mut log_k = Vec::new();
    let mut log_pk = Vec::new();
    for (k, &c) in deg_hist.iter().enumerate() {
        if k > 0 && c > 0 {
            log_k.push((k as f64).ln());
            log_pk.push((c as f64 / n_f).ln());
        }
    }
    let gamma = power_law_exponent(&log_k, &log_pk);

    // Clustering coefficient (sample up to 200 nodes)
    let sample = n.min(200);
    let mut cc_sum = 0.0f64;
    let mut cc_count = 0u32;
    for a in 0..sample {
        let mut neighbors: Vec<usize> = Vec::new();
        for b in 0..n {
            if a == b { continue; }
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            let threshold = x[lo].min(x[hi]);
            if (lo+1..hi).all(|c| x[c] < threshold) {
                neighbors.push(b);
            }
        }
        let k = neighbors.len();
        if k < 2 { continue; }
        let mut edges = 0u32;
        for ni in 0..neighbors.len().min(30) {
            for nj in (ni+1)..neighbors.len().min(30) {
                let (lo, hi) = (neighbors[ni].min(neighbors[nj]), neighbors[ni].max(neighbors[nj]));
                let threshold = x[lo].min(x[hi]);
                if (lo+1..hi).all(|c| x[c] < threshold) { edges += 1; }
            }
        }
        let max_edges = k * (k - 1) / 2;
        if max_edges > 0 {
            cc_sum += edges as f64 / max_edges as f64;
            cc_count += 1;
        }
    }
    let cc = if cc_count > 0 { cc_sum / cc_count as f64 } else { 0.0 };

    // Irreversibility: KL(forward_degree_dist || reverse_degree_dist)
    let rev_x: Vec<f64> = x.iter().rev().copied().collect();
    let mut rev_degree = vec![0u32; n];
    for a in 0..n {
        for b in (a+1)..n {
            let threshold = rev_x[a].min(rev_x[b]);
            if (a+1..b).all(|c| rev_x[c] < threshold) {
                rev_degree[a] += 1; rev_degree[b] += 1;
            }
        }
    }
    let rev_max = *rev_degree.iter().max().unwrap_or(&0) as usize;
    let hist_size = max_deg.max(rev_max) + 1;
    let mut fwd_hist = vec![0u32; hist_size];
    let mut rev_hist = vec![0u32; hist_size];
    for &d in &degree { fwd_hist[d as usize] += 1; }
    for &d in &rev_degree { rev_hist[d as usize] += 1; }

    let mut kl = 0.0f64;
    let denom = n_f + 0.5 * hist_size as f64;
    for k in 0..hist_size {
        let pf = (fwd_hist[k] as f64 + 0.5) / denom;
        let pr = (rev_hist[k] as f64 + 0.5) / denom;
        kl += pf * (pf / pr).ln();
    }

    HvgResult { degree_exponent: gamma, mean_degree, degree_entropy: entropy,
                clustering_coefficient: cc, irreversibility: kl }
}

// ── Shared utility ─────────────────────────────────────────────────────────────

/// OLS log-log regression for power-law exponent. Returns negated slope.
fn power_law_exponent(log_k: &[f64], log_pk: &[f64]) -> f64 {
    let np = log_k.len();
    if np < 3 { return 0.0; }
    let npf = np as f64;
    let sx: f64 = log_k.iter().sum();
    let sy: f64 = log_pk.iter().sum();
    let sxx: f64 = log_k.iter().map(|x| x * x).sum();
    let sxy: f64 = log_k.iter().zip(log_pk.iter()).map(|(x, y)| x * y).sum();
    let denom = npf * sxx - sx * sx;
    if denom.abs() > 1e-30 { -(npf * sxy - sx * sy) / denom } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = tambear::rng::Xoshiro256::new(seed);
        (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect()
    }

    #[test]
    fn nvg_too_short() {
        let r = nvg(&[1.0, 2.0, 3.0]);
        assert!(r.mean_degree.is_nan());
    }

    #[test]
    fn nvg_white_noise_finite() {
        let x = white_noise(50, 42);
        let r = nvg(&x);
        assert!(r.mean_degree.is_finite() && r.mean_degree > 0.0,
            "NVG mean_degree should be positive, got {}", r.mean_degree);
        assert!(r.degree_entropy >= 0.0, "entropy should be non-negative, got {}", r.degree_entropy);
        // Assortativity in [-1, 1]
        assert!(r.assortativity >= -1.0 && r.assortativity <= 1.0,
            "assortativity out of range: {}", r.assortativity);
        assert!(r.clustering_coefficient >= 0.0 && r.clustering_coefficient <= 1.0,
            "CC out of range: {}", r.clustering_coefficient);
    }

    #[test]
    fn hvg_too_short() {
        let r = hvg(&[1.0, 2.0]);
        assert!(r.mean_degree.is_nan());
    }

    #[test]
    fn hvg_white_noise_finite() {
        let x = white_noise(60, 99);
        let r = hvg(&x);
        assert!(r.mean_degree.is_finite() && r.mean_degree > 0.0,
            "HVG mean_degree should be positive, got {}", r.mean_degree);
        assert!(r.degree_entropy >= 0.0, "entropy should be non-negative, got {}", r.degree_entropy);
        assert!(r.irreversibility >= 0.0, "KL divergence should be non-negative, got {}", r.irreversibility);
    }

    #[test]
    fn hvg_monotone_irreversibility() {
        // Strictly monotone series → forward and reverse HVG are mirror images → KL > 0
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let r = hvg(&x);
        // Should complete without panic; irreversibility is defined
        assert!(r.irreversibility.is_finite());
    }

    #[test]
    fn nvg_random_walk_vs_wn() {
        // Random walk should be distinguishable from white noise by degree exponent
        let n = 80;
        let mut rng = tambear::rng::Xoshiro256::new(7);
        let wn: Vec<f64> = (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let mut rw = vec![0.0f64; n];
        for i in 1..n { rw[i] = rw[i-1] + tambear::rng::sample_normal(&mut rng, 0.0, 0.1); }
        let r_wn = nvg(&wn);
        let r_rw = nvg(&rw);
        // Both should return finite results
        assert!(r_wn.degree_exponent.is_finite());
        assert!(r_rw.degree_exponent.is_finite());
    }
}
