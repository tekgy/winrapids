//! # Multi-Adic Sweep Infrastructure
//!
//! Compute in multiple p-adic representations simultaneously.
//! For any integer n, view it through ℤ₂, ℤ₃, ℤ₅, ℤ₇, ... lenses
//! and track how these views co-evolve under discrete dynamical maps.
//!
//! ## Architecture
//!
//! - **p-adic valuation**: v_p(n) = largest power of p dividing n
//! - **p-adic distance**: d_p(a,b) = p^{-v_p(a-b)}
//! - **p-adic expansion**: digits in base p (low-order first)
//! - **Multi-adic profile**: {v₂, v₃, v₅, v₇, ...} simultaneously
//! - **Trajectory analysis**: track multi-adic profiles along T^k(n)
//! - **Synergy detection**: does combined profile explain more than any single?
//!
//! ## Connection to equipartition
//!
//! The primes {2, 3, 5, 7, ...} ARE the natural scales for equipartition
//! fold detection. The multi-adic profile of a trajectory measures how
//! the system visits different "energy levels" in each prime's geometry.

// ═══════════════════════════════════════════════════════════════════════════
// p-adic valuation
// ═══════════════════════════════════════════════════════════════════════════

/// p-adic valuation: v_p(n) = largest k such that p^k divides n.
///
/// v_p(0) = ∞ by convention; we return u32::MAX.
/// Requires p ≥ 2.
pub fn valuation(n: u64, p: u64) -> u32 {
    debug_assert!(p >= 2);
    if n == 0 {
        return u32::MAX;
    }
    let mut n = n;
    let mut v = 0u32;
    while n % p == 0 {
        n /= p;
        v += 1;
    }
    v
}

/// p-adic valuation for signed integers.
/// v_p(n) = v_p(|n|), v_p(0) = ∞ (returns u32::MAX).
pub fn valuation_signed(n: i64, p: u64) -> u32 {
    valuation(n.unsigned_abs(), p)
}

/// p-adic valuation for the DIFFERENCE a - b.
/// v_p(a - b) without computing a - b (avoids overflow for large values).
pub fn valuation_diff(a: u64, b: u64, p: u64) -> u32 {
    if a == b {
        return u32::MAX;
    }
    let diff = if a > b { a - b } else { b - a };
    valuation(diff, p)
}

// ═══════════════════════════════════════════════════════════════════════════
// p-adic distance
// ═══════════════════════════════════════════════════════════════════════════

/// p-adic distance: d_p(a, b) = p^{-v_p(a-b)}.
///
/// d_p(a, a) = 0.
/// Satisfies the ultrametric inequality: d(a,c) ≤ max(d(a,b), d(b,c)).
pub fn p_adic_distance(a: u64, b: u64, p: u64) -> f64 {
    let v = valuation_diff(a, b, p);
    if v == u32::MAX {
        0.0
    } else {
        (p as f64).powi(-(v as i32))
    }
}

/// p-adic norm: |n|_p = p^{-v_p(n)}.
///
/// |0|_p = 0.
pub fn p_adic_norm(n: u64, p: u64) -> f64 {
    let v = valuation(n, p);
    if v == u32::MAX {
        0.0
    } else {
        (p as f64).powi(-(v as i32))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// p-adic expansion
// ═══════════════════════════════════════════════════════════════════════════

/// p-adic expansion of n: digits in base p, low-order first.
///
/// Returns the unique sequence (d₀, d₁, d₂, ...) with 0 ≤ dᵢ < p
/// such that n = Σ dᵢ · pⁱ.
///
/// For n = 0, returns an empty vector.
/// `max_digits` limits the output length (for truncation).
pub fn p_adic_digits(n: u64, p: u64, max_digits: usize) -> Vec<u64> {
    debug_assert!(p >= 2);
    if n == 0 {
        return Vec::new();
    }
    let mut digits = Vec::with_capacity(max_digits.min(64));
    let mut n = n;
    let mut count = 0;
    while n > 0 && count < max_digits {
        digits.push(n % p);
        n /= p;
        count += 1;
    }
    digits
}

/// Reconstruct n from its p-adic digits (low-order first).
pub fn from_p_adic_digits(digits: &[u64], p: u64) -> u64 {
    let mut n = 0u64;
    let mut power = 1u64;
    for &d in digits {
        n = n.wrapping_add(d.wrapping_mul(power));
        power = power.wrapping_mul(p);
    }
    n
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-adic profile
// ═══════════════════════════════════════════════════════════════════════════

/// The default primes for multi-adic analysis.
pub const DEFAULT_PRIMES: &[u64] = &[2, 3, 5, 7];

/// Multi-adic profile of a single integer.
///
/// Simultaneously computes the p-adic valuation for each prime in the set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiAdicProfile {
    /// The primes used.
    pub primes: Vec<u64>,
    /// v_p(n) for each prime, in the same order.
    pub valuations: Vec<u32>,
}

impl MultiAdicProfile {
    /// Compute the multi-adic profile of n for the given primes.
    pub fn of(n: u64, primes: &[u64]) -> Self {
        MultiAdicProfile {
            primes: primes.to_vec(),
            valuations: primes.iter().map(|&p| valuation(n, p)).collect(),
        }
    }

    /// Compute the profile using the default primes {2, 3, 5, 7}.
    pub fn default_of(n: u64) -> Self {
        Self::of(n, DEFAULT_PRIMES)
    }

    /// Get v_p(n) for a specific prime. Returns None if p is not in the profile.
    pub fn v(&self, p: u64) -> Option<u32> {
        self.primes.iter().position(|&q| q == p).map(|i| self.valuations[i])
    }

    /// Total valuation weight: Σ v_p(n). Measures how "smooth" n is.
    pub fn total_weight(&self) -> u64 {
        self.valuations.iter().map(|&v| {
            if v == u32::MAX { 0 } else { v as u64 }
        }).sum()
    }

    /// Is n a unit in ALL p-adic rings? (i.e., v_p(n) = 0 for all p)
    pub fn is_coprime_to_all(&self) -> bool {
        self.valuations.iter().all(|&v| v == 0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trajectory analysis
// ═══════════════════════════════════════════════════════════════════════════

/// A trajectory step: the value and its multi-adic profile.
#[derive(Debug, Clone)]
pub struct TrajectoryStep {
    pub value: u64,
    pub profile: MultiAdicProfile,
}

/// A complete trajectory with multi-adic profiles at each step.
#[derive(Debug, Clone)]
pub struct MultiAdicTrajectory {
    /// The starting value.
    pub seed: u64,
    /// The primes used for profiling.
    pub primes: Vec<u64>,
    /// Each step: (value, profile).
    pub steps: Vec<TrajectoryStep>,
}

/// Standard test maps for trajectory analysis.
#[derive(Debug, Clone, Copy)]
pub enum DynamicalMap {
    /// Collatz: n/2 if even, 3n+1 if odd.
    Collatz,
    /// Generalized: n/d if d|n, mn+1 otherwise. (Collatz = m=3, d=2)
    Generalized { m: u64, d: u64 },
}

impl DynamicalMap {
    /// Apply the map once.
    /// Apply the map once. Returns None on overflow.
    pub fn apply(&self, n: u64) -> u64 {
        match self {
            DynamicalMap::Collatz => {
                if n % 2 == 0 { n / 2 } else { 3u64.saturating_mul(n).saturating_add(1) }
            }
            DynamicalMap::Generalized { m, d } => {
                if n % d == 0 { n / d } else { m.saturating_mul(n).saturating_add(1) }
            }
        }
    }
}

/// Compute a multi-adic trajectory of a dynamical map.
///
/// Runs the map for `max_steps` or until the trajectory enters a cycle
/// (revisits a value), whichever comes first.
pub fn multi_adic_trajectory(
    seed: u64,
    map: DynamicalMap,
    primes: &[u64],
    max_steps: usize,
) -> MultiAdicTrajectory {
    let mut steps = Vec::with_capacity(max_steps);
    let mut seen = std::collections::HashSet::new();
    let mut n = seed;

    for _ in 0..max_steps {
        if !seen.insert(n) {
            break; // cycle detected
        }
        steps.push(TrajectoryStep {
            value: n,
            profile: MultiAdicProfile::of(n, primes),
        });
        if n <= 1 {
            break; // fixed point
        }
        let next = map.apply(n);
        if next == u64::MAX {
            break; // overflow
        }
        n = next;
    }

    MultiAdicTrajectory {
        seed,
        primes: primes.to_vec(),
        steps,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trajectory statistics
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics of a multi-adic trajectory for a single prime.
#[derive(Debug, Clone)]
pub struct PrimeTrajectoryStats {
    pub prime: u64,
    /// Mean valuation along the trajectory.
    pub mean_valuation: f64,
    /// Variance of valuation along the trajectory.
    pub var_valuation: f64,
    /// Maximum valuation seen.
    pub max_valuation: u32,
    /// Fraction of steps where v_p(n) > 0 (n divisible by p).
    pub divisibility_rate: f64,
}

/// Joint statistics across multiple primes.
#[derive(Debug, Clone)]
pub struct TrajectoryStats {
    /// Per-prime statistics.
    pub per_prime: Vec<PrimeTrajectoryStats>,
    /// Correlation matrix of valuations across primes.
    /// Entry (i, j) = Pearson correlation of v_{p_i} and v_{p_j} along trajectory.
    pub correlation_matrix: Vec<Vec<f64>>,
    /// Synergy score: how much variance is explained by the joint profile
    /// beyond what individual profiles explain.
    /// synergy = 1 - var(residual_joint) / var(residual_best_single)
    /// Positive synergy means the primes interact.
    pub synergy: f64,
    /// Trajectory length.
    pub length: usize,
}

/// Compute trajectory statistics from a multi-adic trajectory.
pub fn trajectory_stats(traj: &MultiAdicTrajectory) -> TrajectoryStats {
    let n_primes = traj.primes.len();
    let n_steps = traj.steps.len();

    if n_steps < 2 {
        return TrajectoryStats {
            per_prime: traj.primes.iter().map(|&p| PrimeTrajectoryStats {
                prime: p,
                mean_valuation: 0.0,
                var_valuation: 0.0,
                max_valuation: 0,
                divisibility_rate: 0.0,
            }).collect(),
            correlation_matrix: vec![vec![0.0; n_primes]; n_primes],
            synergy: 0.0,
            length: n_steps,
        };
    }

    // Extract valuation time series for each prime
    let val_series: Vec<Vec<f64>> = (0..n_primes).map(|i| {
        traj.steps.iter().map(|s| {
            let v = s.profile.valuations[i];
            if v == u32::MAX { 0.0 } else { v as f64 }
        }).collect()
    }).collect();

    // Per-prime stats
    let per_prime: Vec<PrimeTrajectoryStats> = (0..n_primes).map(|i| {
        let vals = &val_series[i];
        let n = vals.len() as f64;
        let m = crate::descriptive::moments_ungrouped(vals);
        let mean = m.mean();
        let var = m.variance(0);
        let max_v = traj.steps.iter().map(|s| {
            let v = s.profile.valuations[i];
            if v == u32::MAX { 0 } else { v }
        }).max().unwrap_or(0);
        let div_rate = vals.iter().filter(|&&v| v > 0.0).count() as f64 / n;

        PrimeTrajectoryStats {
            prime: traj.primes[i],
            mean_valuation: mean,
            var_valuation: var,
            max_valuation: max_v,
            divisibility_rate: div_rate,
        }
    }).collect();

    // Correlation matrix
    let mut corr = vec![vec![0.0; n_primes]; n_primes];
    for i in 0..n_primes {
        for j in 0..n_primes {
            corr[i][j] = pearson(&val_series[i], &val_series[j]);
        }
    }

    // Synergy detection: compare joint prediction to best single prediction
    // Using log(value) as the target variable — how well do valuations predict it?
    let log_vals: Vec<f64> = traj.steps.iter().map(|s| (s.value as f64).ln()).collect();
    let synergy = compute_synergy(&val_series, &log_vals);

    TrajectoryStats {
        per_prime,
        correlation_matrix: corr,
        synergy,
        length: n_steps,
    }
}

/// Pearson correlation coefficient.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx < 1e-15 || vy < 1e-15 {
        0.0
    } else {
        cov / (vx * vy).sqrt()
    }
}

/// Synergy: how much additional variance is explained by using ALL primes
/// vs the best single prime.
///
/// Uses R² (coefficient of determination) as the metric.
/// synergy = R²_joint - max(R²_single)
fn compute_synergy(val_series: &[Vec<f64>], target: &[f64]) -> f64 {
    let n = target.len() as f64;
    let mean_target = target.iter().sum::<f64>() / n;
    let ss_total: f64 = target.iter().map(|&y| (y - mean_target) * (y - mean_target)).sum();

    if ss_total < 1e-15 {
        return 0.0;
    }

    // R² for each single prime predictor (simple linear regression)
    let mut best_single_r2 = f64::NEG_INFINITY;
    for series in val_series.iter() {
        let r2 = simple_r2(series, target, ss_total);
        if r2 > best_single_r2 {
            best_single_r2 = r2;
        }
    }

    // R² for joint predictor (multiple linear regression via normal equations)
    let joint_r2 = multiple_r2(val_series, target, ss_total);

    // Synergy = improvement from using all primes together
    joint_r2 - best_single_r2.max(0.0)
}

/// R² from simple linear regression of x → target.
fn simple_r2(x: &[f64], target: &[f64], ss_total: f64) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = target.iter().sum::<f64>() / n;

    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        sxy += dx * (target[i] - my);
        sxx += dx * dx;
    }

    if sxx < 1e-15 {
        return 0.0;
    }

    let beta = sxy / sxx;
    let alpha = my - beta * mx;

    let ss_res: f64 = (0..x.len())
        .map(|i| {
            let pred = alpha + beta * x[i];
            (target[i] - pred) * (target[i] - pred)
        })
        .sum();

    1.0 - ss_res / ss_total
}

/// R² from multiple linear regression (all predictors → target).
/// Uses normal equations: β = (X'X)^{-1} X'y.
fn multiple_r2(predictors: &[Vec<f64>], target: &[f64], ss_total: f64) -> f64 {
    let n = target.len();
    let p = predictors.len();

    if p == 0 || n <= p + 1 {
        return 0.0;
    }

    // Build X (with intercept column) and y
    // X is n × (p+1), column 0 = 1 (intercept)
    let cols = p + 1;

    // X'X: (p+1) × (p+1)
    let mut xtx = vec![0.0; cols * cols];
    let mut xty = vec![0.0; cols];

    for i in 0..n {
        // row i of X: [1, pred[0][i], pred[1][i], ...]
        let mut xi = vec![1.0];
        for k in 0..p {
            xi.push(predictors[k][i]);
        }

        for a in 0..cols {
            for b in 0..cols {
                xtx[a * cols + b] += xi[a] * xi[b];
            }
            xty[a] += xi[a] * target[i];
        }
    }

    // Solve X'X β = X'y via Cholesky-like direct solve for small systems
    // For p ≤ ~10 primes, this is trivially small
    let beta = match solve_small_system(&xtx, &xty, cols) {
        Some(b) => b,
        None => return 0.0,
    };

    // Compute residuals
    let ss_res: f64 = (0..n)
        .map(|i| {
            let mut pred = beta[0]; // intercept
            for k in 0..p {
                pred += beta[k + 1] * predictors[k][i];
            }
            (target[i] - pred) * (target[i] - pred)
        })
        .sum();

    1.0 - ss_res / ss_total
}

/// Solve Ax = b for small dense systems via Gaussian elimination with pivoting.
fn solve_small_system(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return None; // singular
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[col * (n + 1) + j];
                aug[col * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / aug[col * (n + 1) + col];
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }

    Some(x)
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch multi-adic profiles (GPU-ready)
// ═══════════════════════════════════════════════════════════════════════════

/// Batch-compute multi-adic profiles for a range of integers.
///
/// Returns a flat array: profiles[i * n_primes + j] = v_{primes[j]}(start + i).
/// This layout is GPU-friendly (coalesced access per prime).
pub fn batch_profiles(start: u64, count: usize, primes: &[u64]) -> Vec<u32> {
    let n_primes = primes.len();
    let mut result = vec![0u32; count * n_primes];

    for i in 0..count {
        let n = start + i as u64;
        for (j, &p) in primes.iter().enumerate() {
            result[i * n_primes + j] = valuation(n, p);
        }
    }

    result
}

/// Multi-adic distance matrix for a set of integers.
///
/// For each prime p, computes the p-adic distance between all pairs.
/// Returns one flat triangular matrix per prime.
pub fn multi_adic_distance_matrices(values: &[u64], primes: &[u64]) -> Vec<Vec<f64>> {
    let n = values.len();
    let num_pairs = n * (n - 1) / 2;

    primes.iter().map(|&p| {
        let mut dists = Vec::with_capacity(num_pairs);
        for i in 0..n {
            for j in (i + 1)..n {
                dists.push(p_adic_distance(values[i], values[j], p));
            }
        }
        dists
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Valuation ──────────────────────────────────────────────────────

    #[test]
    fn test_valuation_basic() {
        assert_eq!(valuation(12, 2), 2);  // 12 = 2² × 3
        assert_eq!(valuation(12, 3), 1);
        assert_eq!(valuation(12, 5), 0);
        assert_eq!(valuation(8, 2), 3);   // 8 = 2³
        assert_eq!(valuation(1, 2), 0);
        assert_eq!(valuation(0, 2), u32::MAX);
    }

    #[test]
    fn test_valuation_primes() {
        // v_p(p) = 1 for all primes
        for &p in &[2, 3, 5, 7, 11, 13] {
            assert_eq!(valuation(p, p), 1);
        }
    }

    #[test]
    fn test_valuation_powers() {
        // v_p(p^k) = k
        assert_eq!(valuation(32, 2), 5);    // 2^5
        assert_eq!(valuation(243, 3), 5);   // 3^5
        assert_eq!(valuation(3125, 5), 5);  // 5^5
    }

    #[test]
    fn test_valuation_diff() {
        // v_2(10 - 6) = v_2(4) = 2
        assert_eq!(valuation_diff(10, 6, 2), 2);
        // v_3(10 - 1) = v_3(9) = 2
        assert_eq!(valuation_diff(10, 1, 3), 2);
        // symmetric
        assert_eq!(valuation_diff(6, 10, 2), 2);
        // same value
        assert_eq!(valuation_diff(5, 5, 2), u32::MAX);
    }

    // ─── p-adic distance ────────────────────────────────────────────────

    #[test]
    fn test_p_adic_distance() {
        // d_2(0, 8) = 2^{-3} = 0.125
        assert!((p_adic_distance(0, 8, 2) - 0.125).abs() < 1e-15);
        // d_2(0, 4) = 2^{-2} = 0.25
        assert!((p_adic_distance(0, 4, 2) - 0.25).abs() < 1e-15);
        // d_2(a, a) = 0
        assert_eq!(p_adic_distance(7, 7, 2), 0.0);
    }

    #[test]
    fn test_ultrametric_inequality() {
        // d(a,c) ≤ max(d(a,b), d(b,c)) — the ultrametric property
        let triples = [(1, 3, 5), (2, 6, 10), (7, 15, 31)];
        for &p in &[2u64, 3, 5] {
            for &(a, b, c) in &triples {
                let dac = p_adic_distance(a, c, p);
                let dab = p_adic_distance(a, b, p);
                let dbc = p_adic_distance(b, c, p);
                assert!(
                    dac <= dab.max(dbc) + 1e-15,
                    "Ultrametric failed for p={}: d({},{})={} > max(d({},{})={}, d({},{})={})",
                    p, a, c, dac, a, b, dab, b, c, dbc
                );
            }
        }
    }

    // ─── p-adic expansion ───────────────────────────────────────────────

    #[test]
    fn test_p_adic_digits_binary() {
        // 13 = 1101 in binary → digits [1, 0, 1, 1] (low-order first)
        assert_eq!(p_adic_digits(13, 2, 10), vec![1, 0, 1, 1]);
    }

    #[test]
    fn test_p_adic_digits_ternary() {
        // 23 in base 3: 23 = 2·9 + 1·3 + 2 → digits [2, 1, 2]
        assert_eq!(p_adic_digits(23, 3, 10), vec![2, 1, 2]);
    }

    #[test]
    fn test_roundtrip_digits() {
        for &p in &[2, 3, 5, 7, 10] {
            for n in [0, 1, 42, 100, 12345, 999999] {
                let digits = p_adic_digits(n, p, 64);
                let reconstructed = from_p_adic_digits(&digits, p);
                assert_eq!(reconstructed, n, "Roundtrip failed for n={}, p={}", n, p);
            }
        }
    }

    // ─── Multi-adic profile ─────────────────────────────────────────────

    #[test]
    fn test_multi_adic_profile() {
        // 60 = 2² × 3 × 5
        let prof = MultiAdicProfile::default_of(60);
        assert_eq!(prof.v(2), Some(2));
        assert_eq!(prof.v(3), Some(1));
        assert_eq!(prof.v(5), Some(1));
        assert_eq!(prof.v(7), Some(0));
    }

    #[test]
    fn test_profile_coprime() {
        // 11 is coprime to {2, 3, 5, 7}
        let prof = MultiAdicProfile::default_of(11);
        assert!(prof.is_coprime_to_all());
        assert_eq!(prof.total_weight(), 0);
    }

    #[test]
    fn test_profile_smooth_number() {
        // 2^3 × 3^2 × 5 × 7 = 8 × 9 × 5 × 7 = 2520
        let prof = MultiAdicProfile::default_of(2520);
        assert_eq!(prof.v(2), Some(3));
        assert_eq!(prof.v(3), Some(2));
        assert_eq!(prof.v(5), Some(1));
        assert_eq!(prof.v(7), Some(1));
        assert_eq!(prof.total_weight(), 7);
    }

    // ─── Trajectory ─────────────────────────────────────────────────────

    #[test]
    fn test_collatz_trajectory() {
        let traj = multi_adic_trajectory(27, DynamicalMap::Collatz, DEFAULT_PRIMES, 200);
        // 27 takes 111 steps to reach 1 in Collatz
        assert!(traj.steps.len() > 50); // should be long
        assert_eq!(traj.steps[0].value, 27);
        // Last value should be 1 or 2 (cycle)
        let last = traj.steps.last().unwrap().value;
        assert!(last <= 4, "Collatz should reach small values, got {}", last);
    }

    #[test]
    fn test_collatz_profile_step() {
        // Collatz(4) = 2, then 1
        let traj = multi_adic_trajectory(4, DynamicalMap::Collatz, DEFAULT_PRIMES, 10);
        // 4 = 2², v₂(4) = 2
        assert_eq!(traj.steps[0].profile.v(2), Some(2));
        // 4 → 2, v₂(2) = 1
        assert_eq!(traj.steps[1].value, 2);
        assert_eq!(traj.steps[1].profile.v(2), Some(1));
    }

    #[test]
    fn test_generalized_5n_plus_1() {
        let map = DynamicalMap::Generalized { m: 5, d: 2 };
        let traj = multi_adic_trajectory(7, map, DEFAULT_PRIMES, 100);
        assert_eq!(traj.steps[0].value, 7);
        // 7 is odd → 5*7+1 = 36
        assert_eq!(traj.steps[1].value, 36);
    }

    // ─── Trajectory stats ───────────────────────────────────────────────

    #[test]
    fn test_trajectory_stats_collatz() {
        let traj = multi_adic_trajectory(27, DynamicalMap::Collatz, DEFAULT_PRIMES, 200);
        let stats = trajectory_stats(&traj);

        // Should have stats for 4 primes
        assert_eq!(stats.per_prime.len(), 4);

        // In Collatz, v₂ should have positive mean (division by 2 is frequent)
        let v2_stats = &stats.per_prime[0];
        assert_eq!(v2_stats.prime, 2);
        assert!(v2_stats.mean_valuation > 0.0);
        assert!(v2_stats.divisibility_rate > 0.0);

        // Correlation matrix should be 4×4
        assert_eq!(stats.correlation_matrix.len(), 4);
        assert_eq!(stats.correlation_matrix[0].len(), 4);

        // Diagonal should be 1.0 (or 0.0 if variance is 0)
        for i in 0..4 {
            let diag = stats.correlation_matrix[i][i];
            assert!(diag >= -1e-10, "Diagonal correlation should be ≥ 0, got {}", diag);
        }
    }

    #[test]
    fn test_synergy_collatz() {
        let traj = multi_adic_trajectory(27, DynamicalMap::Collatz, DEFAULT_PRIMES, 200);
        let stats = trajectory_stats(&traj);
        // Synergy is a real number (could be positive or negative)
        assert!(stats.synergy.is_finite());
    }

    #[test]
    fn test_compare_maps() {
        // Compare Collatz (3n+1) vs generalized (5n+1)
        let primes = &[2, 3, 5, 7];
        let seed = 27;

        let traj_3 = multi_adic_trajectory(seed, DynamicalMap::Collatz, primes, 200);
        let traj_5 = multi_adic_trajectory(
            seed,
            DynamicalMap::Generalized { m: 5, d: 2 },
            primes,
            200,
        );

        let stats_3 = trajectory_stats(&traj_3);
        let stats_5 = trajectory_stats(&traj_5);

        // Both should produce valid stats
        assert!(stats_3.length > 0);
        assert!(stats_5.length > 0);

        // The 5n+1 map should interact differently with v₅
        // (just check it's computable, not a specific value)
        assert!(stats_5.per_prime.iter().any(|s| s.prime == 5));
    }

    // ─── Batch operations ───────────────────────────────────────────────

    #[test]
    fn test_batch_profiles() {
        let primes = &[2u64, 3, 5];
        let result = batch_profiles(10, 5, primes);
        // 5 values × 3 primes = 15 entries
        assert_eq!(result.len(), 15);

        // Check individual values
        // n=10=2×5: v₂=1, v₃=0, v₅=1
        assert_eq!(result[0], 1); // v₂(10)
        assert_eq!(result[1], 0); // v₃(10)
        assert_eq!(result[2], 1); // v₅(10)

        // n=12=2²×3: v₂=2, v₃=1, v₅=0
        assert_eq!(result[6], 2); // v₂(12)
        assert_eq!(result[7], 1); // v₃(12)
        assert_eq!(result[8], 0); // v₅(12)
    }

    #[test]
    fn test_distance_matrices() {
        let values = [6, 12, 18]; // 2×3, 2²×3, 2×3²
        let primes = &[2u64, 3];
        let matrices = multi_adic_distance_matrices(&values, primes);

        // 2 primes → 2 matrices, each with C(3,2)=3 entries
        assert_eq!(matrices.len(), 2);
        assert_eq!(matrices[0].len(), 3);

        // d_2(6, 12) = 2^{-v₂(6)} = 2^{-1} = 0.5 (since 12-6=6=2×3, v₂(6)=1)
        assert!((matrices[0][0] - 0.5).abs() < 1e-15);
    }

    // ─── p-adic norm ────────────────────────────────────────────────────

    #[test]
    fn test_p_adic_norm() {
        assert!((p_adic_norm(8, 2) - 0.125).abs() < 1e-15); // |8|₂ = 2^{-3}
        assert!((p_adic_norm(9, 3) - 1.0 / 9.0).abs() < 1e-15); // |9|₃ = 3^{-2}
        assert!((p_adic_norm(7, 2) - 1.0).abs() < 1e-15); // |7|₂ = 2^0 = 1
        assert_eq!(p_adic_norm(0, 2), 0.0);
    }
}
