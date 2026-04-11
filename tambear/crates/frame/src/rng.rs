//! # Family 04 — Random Number Generation
//!
//! cuRAND replacement. From first principles.
//!
//! ## What lives here
//!
//! **Generators**: LCG, xoshiro256**, SplitMix64
//! **Uniform**: u64, f64 in [0,1), range
//! **Distributions**: Normal (Box-Muller), Exponential, Gamma, Beta, Poisson,
//!   Binomial, Chi-square, Student-t, F, Cauchy, Lognormal, Bernoulli
//! **Sampling**: shuffle, sample without replacement, weighted sampling
//! **Sequences**: Sobol quasi-random (low discrepancy)
//!
//! ## Architecture
//!
//! All generators are deterministic given a seed — critical for reproducibility.
//! The `TamRng` trait provides a uniform interface. xoshiro256** is the default
//! for quality; LCG for speed when quality isn't critical.
//!
//! ## MSR insight
//!
//! A PRNG state is the MSR of its entire future output sequence.
//! 256 bits of xoshiro state generates 2^256 unique values.
//! The seed IS the sufficient statistic.

use std::f64::consts::PI;

/// Trait for random number generators.
pub trait TamRng {
    /// Generate next u64.
    fn next_u64(&mut self) -> u64;

    /// Generate f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate u64 in [0, n).
    fn next_range(&mut self, n: u64) -> u64 {
        if n == 0 { return 0; }
        // Rejection sampling to avoid modulo bias
        let limit = u64::MAX - (u64::MAX % n);
        loop {
            let x = self.next_u64();
            if x < limit {
                return x % n;
            }
        }
    }

    /// Generate f64 in [lo, hi).
    fn next_f64_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

// ─── SplitMix64 ─────────────────────────────────────────────────────

/// SplitMix64 — fast, simple, 64-bit state.
///
/// Primarily used for seeding other generators. Good statistical quality
/// for a single-state generator.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
}

impl TamRng for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

// ─── xoshiro256** ───────────────────────────────────────────────────

/// xoshiro256** — high quality, 256-bit state, period 2^256 - 1.
///
/// The default PRNG. Passes all BigCrush tests. Fast on modern hardware.
#[derive(Debug, Clone)]
pub struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Create from a 64-bit seed (uses SplitMix64 to expand).
    pub fn new(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        Xoshiro256 {
            s: [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()],
        }
    }

    /// Jump 2^128 steps (for parallel streams).
    pub fn jump(&mut self) {
        const JUMP: [u64; 4] = [
            0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
            0xa9582618e03fc9aa, 0x39abdc4529b1661c,
        ];
        let mut s = [0u64; 4];
        for &j in &JUMP {
            for b in 0..64 {
                if j & (1u64 << b) != 0 {
                    for i in 0..4 { s[i] ^= self.s[i]; }
                }
                self.next_u64();
            }
        }
        self.s = s;
    }
}

impl TamRng for Xoshiro256 {
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
}

// ─── LCG ────────────────────────────────────────────────────────────

/// Linear Congruential Generator — simple, fast, low quality.
///
/// Suitable for non-critical applications (shuffling, test data).
/// Uses the Knuth parameters.
#[derive(Debug, Clone)]
pub struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    pub fn new(seed: u64) -> Self {
        Lcg64 { state: seed }
    }
}

impl TamRng for Lcg64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
}

// ─── Distributions ──────────────────────────────────────────────────

/// Standard normal (Box-Muller transform).
///
/// Generates two independent N(0,1) values. Returns one, caches other.
pub fn normal_pair(rng: &mut dyn TamRng) -> (f64, f64) {
    loop {
        let u1 = rng.next_f64();
        let u2 = rng.next_f64();
        if u1 > 1e-300 {
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            return (r * theta.cos(), r * theta.sin());
        }
    }
}

/// Sample from Normal(mu, sigma²).
pub fn sample_normal(rng: &mut dyn TamRng, mu: f64, sigma: f64) -> f64 {
    let (z, _) = normal_pair(rng);
    mu + sigma * z
}

/// Sample from Exponential(lambda). Mean = 1/lambda.
pub fn sample_exponential(rng: &mut dyn TamRng, lambda: f64) -> f64 {
    if lambda < 0.0 { return f64::NAN; }
    if lambda == 0.0 { return f64::INFINITY; }
    loop {
        let u = rng.next_f64();
        if u > 1e-300 {
            return -u.ln() / lambda;
        }
    }
}

/// Sample from Gamma(alpha, beta) using Marsaglia & Tsang's method.
///
/// Shape alpha > 0, rate beta > 0. Mean = alpha/beta.
pub fn sample_gamma(rng: &mut dyn TamRng, alpha: f64, beta: f64) -> f64 {
    if alpha < 1.0 {
        // Gamma(α) = Gamma(α+1) * U^(1/α) for α < 1
        let g = sample_gamma(rng, alpha + 1.0, 1.0);
        return g * rng.next_f64().powf(1.0 / alpha) / beta;
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (x, _) = normal_pair(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 { continue; }
        let v3 = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v3 / beta;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3 / beta;
        }
    }
}

/// Sample from Beta(alpha, beta).
pub fn sample_beta(rng: &mut dyn TamRng, alpha: f64, beta: f64) -> f64 {
    let x = sample_gamma(rng, alpha, 1.0);
    let y = sample_gamma(rng, beta, 1.0);
    if x + y > 0.0 { x / (x + y) } else { 0.5 }
}

/// Sample from Chi-squared(k).
pub fn sample_chi2(rng: &mut dyn TamRng, k: f64) -> f64 {
    sample_gamma(rng, k / 2.0, 0.5)
}

/// Sample from Student-t(nu).
pub fn sample_t(rng: &mut dyn TamRng, nu: f64) -> f64 {
    let (z, _) = normal_pair(rng);
    let chi2 = sample_chi2(rng, nu);
    z / (chi2 / nu).sqrt()
}

/// Sample from F(d1, d2).
pub fn sample_f(rng: &mut dyn TamRng, d1: f64, d2: f64) -> f64 {
    let x1 = sample_chi2(rng, d1) / d1;
    let x2 = sample_chi2(rng, d2) / d2;
    if x2 > 0.0 { x1 / x2 } else { f64::INFINITY }
}

/// Sample from Cauchy(x0, gamma).
pub fn sample_cauchy(rng: &mut dyn TamRng, x0: f64, gamma: f64) -> f64 {
    let u = rng.next_f64();
    x0 + gamma * (PI * (u - 0.5)).tan()
}

/// Sample from Lognormal(mu, sigma).
pub fn sample_lognormal(rng: &mut dyn TamRng, mu: f64, sigma: f64) -> f64 {
    sample_normal(rng, mu, sigma).exp()
}

/// Sample from Bernoulli(p).
pub fn sample_bernoulli(rng: &mut dyn TamRng, p: f64) -> bool {
    rng.next_f64() < p
}

/// Sample from Poisson(lambda) via Knuth's algorithm.
pub fn sample_poisson(rng: &mut dyn TamRng, lambda: f64) -> u64 {
    if lambda <= 0.0 { return 0; }
    if lambda < 30.0 {
        // Direct method
        let l = (-lambda).exp();
        let mut k = 0u64;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.next_f64();
            if p < l { return k - 1; }
        }
    } else {
        // Normal approximation for large lambda
        let z = sample_normal(rng, lambda, lambda.sqrt());
        z.round().max(0.0) as u64
    }
}

/// Sample from Binomial(n, p).
pub fn sample_binomial(rng: &mut dyn TamRng, n: u64, p: f64) -> u64 {
    if n < 30 {
        // Direct method
        (0..n).filter(|_| sample_bernoulli(rng, p)).count() as u64
    } else {
        // Normal approximation
        let mu = n as f64 * p;
        let sigma = (n as f64 * p * (1.0 - p)).sqrt();
        sample_normal(rng, mu, sigma).round().max(0.0).min(n as f64) as u64
    }
}

/// Sample from Geometric(p). Returns number of failures before first success.
pub fn sample_geometric(rng: &mut dyn TamRng, p: f64) -> u64 {
    if p >= 1.0 { return 0; }
    if p <= 0.0 { return u64::MAX; }
    let u = rng.next_f64();
    if u > 1e-300 {
        (u.ln() / (1.0 - p).ln()).floor() as u64
    } else {
        0
    }
}

// ─── Sampling algorithms ────────────────────────────────────────────

/// Fisher-Yates shuffle (in-place).
pub fn shuffle<T>(rng: &mut dyn TamRng, data: &mut [T]) {
    let n = data.len();
    for i in (1..n).rev() {
        let j = rng.next_range((i + 1) as u64) as usize;
        data.swap(i, j);
    }
}

/// Sample k elements without replacement (Floyd's algorithm).
///
/// Returns indices into the original array.
pub fn sample_without_replacement(rng: &mut dyn TamRng, n: usize, k: usize) -> Vec<usize> {
    let k = k.min(n);
    if k == n {
        return (0..n).collect();
    }
    // Floyd's algorithm O(k)
    let mut selected = Vec::with_capacity(k);
    let mut set = std::collections::HashSet::with_capacity(k);
    for j in (n - k)..n {
        let t = rng.next_range((j + 1) as u64) as usize;
        if set.contains(&t) {
            selected.push(j);
            set.insert(j);
        } else {
            selected.push(t);
            set.insert(t);
        }
    }
    selected
}

/// Weighted sampling with replacement (alias method setup not needed for small n).
///
/// Returns `k` indices sampled according to weights.
pub fn sample_weighted(rng: &mut dyn TamRng, weights: &[f64], k: usize) -> Vec<usize> {
    let n = weights.len();
    if n == 0 || k == 0 { return vec![]; }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 { return vec![0; k]; }

    // Build CDF
    let mut cdf = Vec::with_capacity(n);
    let mut cum = 0.0;
    for &w in weights {
        cum += w / total;
        cdf.push(cum);
    }

    (0..k).map(|_| {
        let u = rng.next_f64();
        match cdf.binary_search_by(|c| c.total_cmp(&u)) {
            Ok(i) => i,
            Err(i) => i.min(n - 1),
        }
    }).collect()
}

// ─── Fill arrays ────────────────────────────────────────────────────

/// Fill a slice with uniform f64 in [0, 1).
pub fn fill_uniform(rng: &mut dyn TamRng, out: &mut [f64]) {
    for v in out.iter_mut() { *v = rng.next_f64(); }
}

/// Fill a slice with normal N(mu, sigma²).
pub fn fill_normal(rng: &mut dyn TamRng, out: &mut [f64], mu: f64, sigma: f64) {
    let mut i = 0;
    while i + 1 < out.len() {
        let (z1, z2) = normal_pair(rng);
        out[i] = mu + sigma * z1;
        out[i + 1] = mu + sigma * z2;
        i += 2;
    }
    if i < out.len() {
        out[i] = sample_normal(rng, mu, sigma);
    }
}

/// Generate a vector of n standard normal values.
pub fn randn(rng: &mut dyn TamRng, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    fill_normal(rng, &mut out, 0.0, 1.0);
    out
}

/// Generate a vector of n uniform [0,1) values.
pub fn randu(rng: &mut dyn TamRng, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    fill_uniform(rng, &mut out);
    out
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Generators ──

    #[test]
    fn splitmix_deterministic() {
        let mut rng1 = SplitMix64::new(42);
        let mut rng2 = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn xoshiro_deterministic() {
        let mut rng1 = Xoshiro256::new(42);
        let mut rng2 = Xoshiro256::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn xoshiro_different_seeds() {
        let mut rng1 = Xoshiro256::new(1);
        let mut rng2 = Xoshiro256::new(2);
        let seq1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let seq2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();
        assert_ne!(seq1, seq2);
    }

    #[test]
    fn uniform_range() {
        let mut rng = Xoshiro256::new(42);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "out of range: {}", v);
        }
    }

    // ── Distributions ──

    #[test]
    fn normal_mean_variance() {
        let mut rng = Xoshiro256::new(42);
        let n = 10000;
        let samples: Vec<f64> = (0..n).map(|_| sample_normal(&mut rng, 5.0, 2.0)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!((mean - 5.0).abs() < 0.1, "mean = {}", mean);
        assert!((var - 4.0).abs() < 0.5, "var = {}", var);
    }

    #[test]
    fn exponential_mean() {
        let mut rng = Xoshiro256::new(42);
        let n = 10000;
        let lambda = 2.0;
        let samples: Vec<f64> = (0..n).map(|_| sample_exponential(&mut rng, lambda)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.05, "exp mean = {}", mean);
        assert!(samples.iter().all(|&x| x >= 0.0), "negative exponential");
    }

    #[test]
    fn gamma_mean() {
        let mut rng = Xoshiro256::new(42);
        let n = 10000;
        let alpha = 3.0;
        let beta = 2.0;
        let samples: Vec<f64> = (0..n).map(|_| sample_gamma(&mut rng, alpha, beta)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let expected = alpha / beta;
        assert!((mean - expected).abs() < 0.1, "gamma mean = {} vs {}", mean, expected);
    }

    #[test]
    fn beta_in_unit_interval() {
        let mut rng = Xoshiro256::new(42);
        for _ in 0..1000 {
            let x = sample_beta(&mut rng, 2.0, 5.0);
            assert!(x >= 0.0 && x <= 1.0, "beta out of range: {}", x);
        }
    }

    #[test]
    fn beta_mean() {
        let mut rng = Xoshiro256::new(42);
        let n = 10000;
        let a = 2.0;
        let b = 5.0;
        let samples: Vec<f64> = (0..n).map(|_| sample_beta(&mut rng, a, b)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let expected = a / (a + b);
        assert!((mean - expected).abs() < 0.05, "beta mean = {} vs {}", mean, expected);
    }

    #[test]
    fn poisson_mean() {
        let mut rng = Xoshiro256::new(42);
        let n = 10000;
        let lambda = 5.0;
        let samples: Vec<u64> = (0..n).map(|_| sample_poisson(&mut rng, lambda)).collect();
        let mean = samples.iter().sum::<u64>() as f64 / n as f64;
        assert!((mean - lambda).abs() < 0.2, "poisson mean = {} vs {}", mean, lambda);
    }

    #[test]
    fn binomial_mean() {
        let mut rng = Xoshiro256::new(42);
        let n_trials = 20u64;
        let p = 0.3;
        let n = 5000;
        let samples: Vec<u64> = (0..n).map(|_| sample_binomial(&mut rng, n_trials, p)).collect();
        let mean = samples.iter().sum::<u64>() as f64 / n as f64;
        let expected = n_trials as f64 * p;
        assert!((mean - expected).abs() < 0.5, "binomial mean = {} vs {}", mean, expected);
    }

    #[test]
    fn cauchy_median() {
        let mut rng = Xoshiro256::new(42);
        let n = 10001;
        let mut samples: Vec<f64> = (0..n).map(|_| sample_cauchy(&mut rng, 3.0, 1.0)).collect();
        samples.sort_by(|a, b| a.total_cmp(b));
        let median = samples[n / 2];
        assert!((median - 3.0).abs() < 0.2, "cauchy median = {}", median);
    }

    #[test]
    fn lognormal_positive() {
        let mut rng = Xoshiro256::new(42);
        for _ in 0..1000 {
            let x = sample_lognormal(&mut rng, 0.0, 1.0);
            assert!(x > 0.0, "lognormal negative: {}", x);
        }
    }

    // ── Sampling ──

    #[test]
    fn shuffle_preserves_elements() {
        let mut rng = Xoshiro256::new(42);
        let mut data: Vec<i32> = (0..10).collect();
        let orig: Vec<i32> = data.clone();
        shuffle(&mut rng, &mut data);
        data.sort();
        assert_eq!(data, orig);
    }

    #[test]
    fn sample_without_replacement_unique() {
        let mut rng = Xoshiro256::new(42);
        let indices = sample_without_replacement(&mut rng, 100, 10);
        assert_eq!(indices.len(), 10);
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 10, "duplicates in sample");
        assert!(sorted.iter().all(|&i| i < 100));
    }

    #[test]
    fn weighted_sampling_biased() {
        let mut rng = Xoshiro256::new(42);
        let weights = vec![0.0, 0.0, 1.0]; // always pick index 2
        let samples = sample_weighted(&mut rng, &weights, 100);
        assert!(samples.iter().all(|&i| i == 2));
    }

    // ── Fill ──

    #[test]
    fn randn_correct_count() {
        let mut rng = Xoshiro256::new(42);
        let v = randn(&mut rng, 100);
        assert_eq!(v.len(), 100);
    }

    #[test]
    fn fill_normal_statistics() {
        let mut rng = Xoshiro256::new(42);
        let mut data = vec![0.0; 10000];
        fill_normal(&mut rng, &mut data, 0.0, 1.0);
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        assert!((mean).abs() < 0.05, "mean = {}", mean);
        assert!((var - 1.0).abs() < 0.1, "var = {}", var);
    }

    // ── Edge cases ──

    #[test]
    fn next_range_small() {
        let mut rng = Xoshiro256::new(42);
        for _ in 0..100 {
            let v = rng.next_range(3);
            assert!(v < 3);
        }
    }

    #[test]
    fn geometric_mean() {
        let mut rng = Xoshiro256::new(42);
        let p = 0.5;
        let n = 10000;
        let samples: Vec<u64> = (0..n).map(|_| sample_geometric(&mut rng, p)).collect();
        let mean = samples.iter().sum::<u64>() as f64 / n as f64;
        let expected = (1.0 - p) / p; // 1.0
        assert!((mean - expected).abs() < 0.2, "geometric mean = {} vs {}", mean, expected);
    }
}
