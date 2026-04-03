//! Complexity and chaos measures — nonlinear dynamics, entropy, fractal geometry.
//!
//! ## Architecture
//!
//! These measures characterize the complexity of time series and dynamical systems.
//! They're critical for financial time series where linear models fail — market
//! microstructure is nonlinear by nature.
//!
//! Most methods work on **embedded time series**: given a 1D signal, we reconstruct
//! the attractor in m-dimensional phase space using time-delay embedding:
//! y(t) = [x(t), x(t+τ), x(t+2τ), ..., x(t+(m-1)τ)]
//!
//! This is Takens' theorem: the embedding preserves topological properties of
//! the original attractor.
//!
//! ## Measures
//!
//! **Entropy**: sample entropy (SampEn), approximate entropy (ApEn), permutation entropy
//! **Fractal**: Higuchi fractal dimension, box-counting dimension, DFA
//! **Chaos**: largest Lyapunov exponent (Rosenstein), correlation dimension
//! **Complexity**: Lempel-Ziv complexity, Hurst exponent (R/S analysis)
//!
//! ## .tbs integration
//!
//! ```text
//! sample_entropy(data, m=2, r=0.2)  # SampEn
//! hurst(data)                        # Hurst exponent
//! lyapunov(data, m=5, tau=1)        # largest Lyapunov exponent
//! fractal_dim(data)                  # Higuchi fractal dimension
//! ```

// ═══════════════════════════════════════════════════════════════════════════
// Sample Entropy (SampEn) and Approximate Entropy (ApEn)
// ═══════════════════════════════════════════════════════════════════════════

/// Sample entropy (Richman & Moorman, 2000).
///
/// SampEn(m, r, N) = -ln(A/B) where:
/// - B = number of template matches for templates of length m
/// - A = number of template matches for templates of length m+1
/// - r = tolerance (typically 0.1-0.25 × std)
///
/// SampEn = 0 for perfectly regular signals.
/// SampEn increases with increasing randomness.
/// Advantage over ApEn: no self-matches, less biased for short data.
pub fn sample_entropy(data: &[f64], m: usize, r: f64) -> f64 {
    let n = data.len();
    if n < m + 2 { return f64::NAN; }

    let count_b = count_matches(data, m, r);
    let count_a = count_matches(data, m + 1, r);

    if count_b == 0 { return f64::INFINITY; }
    if count_a == 0 { return f64::INFINITY; }

    -((count_a as f64) / (count_b as f64)).ln()
}

/// Approximate entropy (Pincus, 1991).
///
/// ApEn(m, r, N) = φ^m(r) - φ^{m+1}(r) where
/// φ^m(r) = (1/N') Σ ln(Cᵢᵐ(r))
/// Cᵢᵐ(r) = (number of j such that d[xᵢ, xⱼ] ≤ r) / N'
///
/// Includes self-matches (unlike SampEn). More biased but works with shorter data.
pub fn approx_entropy(data: &[f64], m: usize, r: f64) -> f64 {
    let phi_m = phi_func(data, m, r);
    let phi_m1 = phi_func(data, m + 1, r);
    phi_m - phi_m1
}

/// Count template matches of length m within tolerance r (no self-matches).
fn count_matches(data: &[f64], m: usize, r: f64) -> usize {
    let n = data.len();
    let n_templates = n - m; // +1 but we start at 0
    let mut count = 0;

    for i in 0..n_templates {
        for j in (i + 1)..n_templates {
            // Chebyshev (L∞) distance between templates
            let mut match_ok = true;
            for k in 0..m {
                if (data[i + k] - data[j + k]).abs() > r {
                    match_ok = false;
                    break;
                }
            }
            if match_ok { count += 1; }
        }
    }
    count
}

/// φ function for ApEn (with self-matches).
fn phi_func(data: &[f64], m: usize, r: f64) -> f64 {
    let n = data.len();
    let n_templates = n - m + 1;
    if n_templates == 0 { return 0.0; }

    let mut sum_log = 0.0;
    for i in 0..n_templates {
        let mut count = 0;
        for j in 0..n_templates {
            let mut match_ok = true;
            for k in 0..m {
                if (data[i + k] - data[j + k]).abs() > r {
                    match_ok = false;
                    break;
                }
            }
            if match_ok { count += 1; }
        }
        sum_log += (count as f64 / n_templates as f64).ln();
    }
    sum_log / n_templates as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// Permutation Entropy
// ═══════════════════════════════════════════════════════════════════════════

/// Permutation entropy (Bandt & Pompe, 2002).
///
/// Maps time series to a sequence of ordinal patterns of length m,
/// then computes Shannon entropy of the pattern distribution.
///
/// PE = 0 for perfectly monotonic. PE = log(m!) for completely random.
/// Normalized: PE / log(m!) ∈ [0, 1].
///
/// Very robust to noise. O(n × m) computation.
pub fn permutation_entropy(data: &[f64], m: usize, tau: usize) -> f64 {
    if m < 2 || data.len() < (m - 1) * tau + 1 { return f64::NAN; }

    let n_patterns = data.len() - (m - 1) * tau;
    let n_perms = factorial(m);
    let mut counts = vec![0usize; n_perms];

    for i in 0..n_patterns {
        let pattern: Vec<f64> = (0..m).map(|k| data[i + k * tau]).collect();
        let perm_idx = pattern_to_index(&pattern, m);
        if perm_idx < n_perms { counts[perm_idx] += 1; }
    }

    // Shannon entropy of pattern distribution
    let total = n_patterns as f64;
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Normalized permutation entropy: PE / log(m!) ∈ [0, 1].
pub fn normalized_permutation_entropy(data: &[f64], m: usize, tau: usize) -> f64 {
    let pe = permutation_entropy(data, m, tau);
    if pe.is_nan() { return f64::NAN; }
    let max_entropy = (factorial(m) as f64).ln();
    if max_entropy <= 0.0 { return f64::NAN; }
    pe / max_entropy
}

/// Map an ordinal pattern to a unique index (Lehmer code).
fn pattern_to_index(pattern: &[f64], m: usize) -> usize {
    let mut index = 0;
    for i in 0..m {
        let mut count = 0;
        for j in (i + 1)..m {
            if pattern[j] < pattern[i] { count += 1; }
        }
        index = index * (m - i) + count;
    }
    index
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

// ═══════════════════════════════════════════════════════════════════════════
// Hurst exponent (R/S analysis)
// ═══════════════════════════════════════════════════════════════════════════

/// Hurst exponent via rescaled range (R/S) analysis.
///
/// H = 0.5 → random walk (no memory)
/// H > 0.5 → persistent (trending)
/// H < 0.5 → anti-persistent (mean-reverting)
///
/// Computes R/S for multiple block sizes, fits log-log regression.
pub fn hurst_rs(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 20 { return f64::NAN; }

    let mut log_ns = Vec::new();
    let mut log_rs = Vec::new();

    // Block sizes from 10 to n/2
    let min_block = 10;
    let max_block = n / 2;
    let mut block_size = min_block;

    while block_size <= max_block {
        let n_blocks = n / block_size;
        if n_blocks < 1 { break; }

        let mut rs_sum = 0.0;
        let mut rs_count = 0;

        for b in 0..n_blocks {
            let start = b * block_size;
            let block = &data[start..start + block_size];

            let mean: f64 = block.iter().sum::<f64>() / block_size as f64;

            // Cumulative deviations from mean
            let mut cum_dev = vec![0.0; block_size];
            cum_dev[0] = block[0] - mean;
            for i in 1..block_size {
                cum_dev[i] = cum_dev[i - 1] + (block[i] - mean);
            }

            let range = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);

            let std = (block.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (block_size - 1) as f64).sqrt();

            if std > 0.0 {
                rs_sum += range / std;
                rs_count += 1;
            }
        }

        if rs_count > 0 {
            let rs_avg = rs_sum / rs_count as f64;
            log_ns.push((block_size as f64).ln());
            log_rs.push(rs_avg.ln());
        }

        // Double the block size (roughly)
        block_size = (block_size as f64 * 1.5).ceil() as usize;
    }

    if log_ns.len() < 2 { return f64::NAN; }

    // Simple linear regression: log(R/S) = H × log(n) + c
    ols_slope(&log_ns, &log_rs)
}

/// Simple OLS slope.
fn ols_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx: f64 = x.iter().sum::<f64>() / n;
    let my: f64 = y.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        sxy += dx * (y[i] - my);
        sxx += dx * dx;
    }
    if sxx == 0.0 { return f64::NAN; }
    sxy / sxx
}

// ═══════════════════════════════════════════════════════════════════════════
// Detrended Fluctuation Analysis (DFA)
// ═══════════════════════════════════════════════════════════════════════════

/// Detrended Fluctuation Analysis (Peng et al., 1994).
///
/// α = 0.5 → white noise
/// α = 1.0 → 1/f noise (pink noise)
/// α = 1.5 → Brownian motion
/// α > 1.0 → non-stationary, long-range correlated
///
/// More robust than R/S for non-stationary data.
pub fn dfa(data: &[f64], min_box: usize, max_box: usize) -> f64 {
    let n = data.len();
    if n < 2 * min_box { return f64::NAN; }

    // Step 1: cumulative sum (profile)
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let mut profile = vec![0.0; n];
    profile[0] = data[0] - mean;
    for i in 1..n {
        profile[i] = profile[i - 1] + (data[i] - mean);
    }

    let mut log_sizes = Vec::new();
    let mut log_flucts = Vec::new();

    let mut box_size = min_box;
    while box_size <= max_box && box_size <= n / 2 {
        let n_boxes = n / box_size;
        if n_boxes < 1 { break; }

        let mut fluct_sum = 0.0;
        for b in 0..n_boxes {
            let start = b * box_size;
            let segment = &profile[start..start + box_size];

            // Fit linear trend to segment
            let (a, slope) = linear_fit_segment(segment);

            // Fluctuation = RMS of detrended segment
            let rms: f64 = segment.iter().enumerate().map(|(i, &y)| {
                let trend = a + slope * i as f64;
                (y - trend).powi(2)
            }).sum::<f64>() / box_size as f64;

            fluct_sum += rms.sqrt();
        }

        let avg_fluct = fluct_sum / n_boxes as f64;
        if avg_fluct > 0.0 {
            log_sizes.push((box_size as f64).ln());
            log_flucts.push(avg_fluct.ln());
        }

        box_size = (box_size as f64 * 1.3).ceil() as usize;
        if box_size == (box_size as f64 / 1.3).ceil() as usize {
            box_size += 1; // Ensure progress
        }
    }

    if log_sizes.len() < 2 { return f64::NAN; }
    ols_slope(&log_sizes, &log_flucts)
}

/// Linear fit (intercept, slope) for a segment indexed 0..n.
/// Centered formulation avoids catastrophic cancellation.
fn linear_fit_segment(segment: &[f64]) -> (f64, f64) {
    let n = segment.len() as f64;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y: f64 = segment.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (i, &y) in segment.iter().enumerate() {
        let dx = i as f64 - mean_x;
        sxy += dx * (y - mean_y);
        sxx += dx * dx;
    }
    if sxx.abs() < 1e-15 {
        return (mean_y, 0.0);
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    (intercept, slope)
}

// ═══════════════════════════════════════════════════════════════════════════
// Higuchi Fractal Dimension
// ═══════════════════════════════════════════════════════════════════════════

/// Higuchi fractal dimension (Higuchi, 1988).
///
/// FD = 1.0 for smooth signals, FD → 2.0 for space-filling signals.
/// For white noise, FD ≈ 2.0. For Brownian motion, FD ≈ 1.5.
///
/// More computationally efficient than box-counting for time series.
pub fn higuchi_fd(data: &[f64], k_max: usize) -> f64 {
    let n = data.len();
    if n < k_max + 1 { return f64::NAN; }

    let mut log_ks = Vec::new();
    let mut log_ls = Vec::new();

    for k in 1..=k_max {
        let mut l_k = 0.0;

        for m in 1..=k {
            let n_seg = ((n - m) as f64 / k as f64).floor() as usize;
            if n_seg < 1 { continue; }

            let mut l_mk = 0.0;
            for i in 1..=n_seg {
                l_mk += (data[m - 1 + i * k] - data[m - 1 + (i - 1) * k]).abs();
            }
            l_mk = l_mk * (n - 1) as f64 / (n_seg * k) as f64;
            l_k += l_mk;
        }
        l_k /= k as f64;

        if l_k > 0.0 {
            log_ks.push((k as f64).ln());
            log_ls.push(l_k.ln());
        }
    }

    if log_ks.len() < 2 { return f64::NAN; }
    -ols_slope(&log_ks, &log_ls) // Negative because L(k) decreases with k
}

// ═══════════════════════════════════════════════════════════════════════════
// Lempel-Ziv Complexity
// ═══════════════════════════════════════════════════════════════════════════

/// Lempel-Ziv complexity (Lempel & Ziv, 1976).
///
/// Measures the number of distinct substrings in a binary sequence.
/// Input is binarized at the median.
///
/// LZ = c(n) / (n / log₂(n)) normalized by the expected value for random binary.
/// LZ ≈ 0 for periodic. LZ ≈ 1 for random.
pub fn lempel_ziv_complexity(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 { return f64::NAN; }

    // Binarize at median
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = sorted[n / 2];
    let binary: Vec<bool> = data.iter().map(|&x| x > med).collect();

    // Count distinct patterns (Kaspar & Schuster algorithm)
    let mut c = 1; // complexity counter
    let mut l = 1; // current pattern length
    let mut k = 1; // pointer
    let mut k_max = 1;

    while k + l <= n {
        // Check if substring S[k..k+l] has appeared in S[0..k+l-1]
        let pattern_end = k + l;
        let found = (0..k).any(|start| {
            (0..l).all(|j| {
                start + j < pattern_end - 1 && binary[start + j] == binary[k + j]
            })
        });

        if found {
            l += 1;
        } else {
            c += 1;
            if l > k_max { k_max = l; }
            k += l;
            l = 1;
        }
    }

    // Normalize
    let nf = n as f64;
    let expected = nf / nf.log2();
    c as f64 / expected
}

// ═══════════════════════════════════════════════════════════════════════════
// Correlation Dimension
// ═══════════════════════════════════════════════════════════════════════════

/// Correlation dimension (Grassberger-Procaccia algorithm).
///
/// Estimates the dimension of the attractor from time-delay embedded data.
///
/// D₂ = lim_{r→0} d(log C(r)) / d(log r)
/// where C(r) = (2/N(N-1)) × #{(i,j): ||xᵢ - xⱼ|| < r}
///
/// Embedding: m-dimensional vectors with time delay τ.
pub fn correlation_dimension(data: &[f64], m: usize, tau: usize) -> f64 {
    let n = data.len();
    let n_vectors = n - (m - 1) * tau;
    if n_vectors < 10 { return f64::NAN; }

    // Create embedded vectors
    let vectors: Vec<Vec<f64>> = (0..n_vectors)
        .map(|i| (0..m).map(|k| data[i + k * tau]).collect())
        .collect();

    // Compute pairwise distances (L∞ norm)
    let mut distances = Vec::new();
    for i in 0..n_vectors {
        for j in (i + 1)..n_vectors {
            let d: f64 = (0..m).map(|k| (vectors[i][k] - vectors[j][k]).abs())
                .fold(0.0, f64::max);
            distances.push(d);
        }
    }
    distances.sort_by(|a, b| a.total_cmp(b));

    if distances.is_empty() { return f64::NAN; }

    // Compute C(r) for several r values in log-space
    let r_min = distances[distances.len() / 10].max(1e-10);
    let r_max = distances[distances.len() * 9 / 10];
    if r_min >= r_max { return f64::NAN; }

    let n_r = 20;
    let mut log_rs = Vec::new();
    let mut log_crs = Vec::new();
    let n_pairs = distances.len() as f64;

    for i in 0..n_r {
        let log_r = r_min.ln() + (r_max.ln() - r_min.ln()) * i as f64 / (n_r - 1) as f64;
        let r = log_r.exp();

        // Count pairs within distance r (binary search since distances are sorted)
        let count = distances.partition_point(|&d| d < r);
        let cr = count as f64 / n_pairs;

        if cr > 0.0 {
            log_rs.push(r.ln());
            log_crs.push(cr.ln());
        }
    }

    if log_rs.len() < 3 { return f64::NAN; }
    ols_slope(&log_rs, &log_crs)
}

// ═══════════════════════════════════════════════════════════════════════════
// Largest Lyapunov Exponent (Rosenstein's method)
// ═══════════════════════════════════════════════════════════════════════════

/// Largest Lyapunov exponent via Rosenstein's method (1993).
///
/// λ₁ > 0 → chaos (exponential divergence of nearby trajectories)
/// λ₁ = 0 → periodic or quasi-periodic
/// λ₁ < 0 → stable fixed point
///
/// Algorithm:
/// 1. Time-delay embed
/// 2. For each point, find nearest neighbor (excluding temporal neighbors)
/// 3. Track divergence: d(i, Δn) = ||x(i+Δn) - x(nn(i)+Δn)||
/// 4. λ₁ ≈ (1/Δt) × mean(log(d(i, Δn)))
pub fn largest_lyapunov(data: &[f64], m: usize, tau: usize, dt: f64) -> f64 {
    let n = data.len();
    let n_vectors = n - (m - 1) * tau;
    if n_vectors < 20 { return f64::NAN; }

    let mean_period = estimate_mean_period(data);
    let min_temporal_sep = mean_period.max(tau);

    // Create embedded vectors
    let vectors: Vec<Vec<f64>> = (0..n_vectors)
        .map(|i| (0..m).map(|k| data[i + k * tau]).collect())
        .collect();

    // For each point, find nearest neighbor (excluding temporal neighbors)
    let max_diverge = n_vectors / 4;
    let mut divergences = vec![0.0; max_diverge];
    let mut counts = vec![0usize; max_diverge];

    for i in 0..n_vectors - max_diverge {
        let mut min_dist = f64::INFINITY;
        let mut nn_idx = 0;

        for j in 0..n_vectors - max_diverge {
            if (i as i64 - j as i64).unsigned_abs() as usize <= min_temporal_sep { continue; }
            let d: f64 = (0..m).map(|k| (vectors[i][k] - vectors[j][k]).powi(2)).sum::<f64>().sqrt();
            if d < min_dist && d > 0.0 {
                min_dist = d;
                nn_idx = j;
            }
        }

        if min_dist == f64::INFINITY { continue; }

        // Track divergence
        for dn in 0..max_diverge {
            if i + dn >= n_vectors || nn_idx + dn >= n_vectors { break; }
            let d: f64 = (0..m).map(|k| (vectors[i + dn][k] - vectors[nn_idx + dn][k]).powi(2))
                .sum::<f64>().sqrt();
            if d > 0.0 {
                divergences[dn] += d.ln();
                counts[dn] += 1;
            }
        }
    }

    // Average log divergence
    let mut log_divs = Vec::new();
    let mut steps = Vec::new();
    for dn in 0..max_diverge {
        if counts[dn] > 0 {
            log_divs.push(divergences[dn] / counts[dn] as f64);
            steps.push(dn as f64 * dt);
        }
    }

    if log_divs.len() < 3 { return f64::NAN; }

    // λ₁ = slope of log divergence vs time (use first third for linear region)
    let use_n = (log_divs.len() / 3).max(3);
    ols_slope(&steps[..use_n], &log_divs[..use_n])
}

/// Estimate mean period of data via zero-crossings.
fn estimate_mean_period(data: &[f64]) -> usize {
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let mut crossings = 0;
    for i in 1..data.len() {
        if (data[i] - mean) * (data[i - 1] - mean) < 0.0 {
            crossings += 1;
        }
    }
    if crossings < 2 { return 1; }
    (2 * (data.len() - 1)) / crossings
}

// ─── Lyapunov spectrum from ODE (Benettin et al. 1980) ──────────────────

/// Result of a full Lyapunov spectrum computation.
#[derive(Debug, Clone)]
pub struct LyapunovSpectrum {
    /// Lyapunov exponents in descending order (λ₁ ≥ λ₂ ≥ ... ≥ λ_d).
    pub exponents: Vec<f64>,
    /// Kaplan-Yorke (Lyapunov) dimension.
    pub kaplan_yorke_dim: f64,
    /// Sum of all exponents (= divergence of flow; 0 for Hamiltonian systems).
    pub sum: f64,
    /// Number of positive exponents (= degrees of chaos).
    pub n_positive: usize,
}

/// Compute the full Lyapunov spectrum of a continuous dynamical system.
///
/// Uses the standard QR-based algorithm (Benettin, Galgani, Giorgilli, Strelcyn 1980):
/// 1. Integrate the ODE and its variational equation (tangent flow)
/// 2. Periodically QR-decompose the tangent matrix
/// 3. Accumulate log|R_ii| to get Lyapunov exponents
///
/// # Arguments
/// * `f` — the vector field: dy/dt = f(t, y)
/// * `jac` — the Jacobian matrix: J_{ij} = ∂f_i/∂y_j, returned as row-major flat Vec
/// * `y0` — initial condition (dimension d)
/// * `t_transient` — time to discard (let transients die)
/// * `t_compute` — time over which to compute exponents
/// * `dt` — integration step size
/// * `qr_interval` — number of steps between QR reorthogonalizations
pub fn lyapunov_spectrum(
    f: impl Fn(f64, &[f64]) -> Vec<f64>,
    jac: impl Fn(f64, &[f64]) -> Vec<f64>,
    y0: &[f64],
    t_transient: f64,
    t_compute: f64,
    dt: f64,
    qr_interval: usize,
) -> LyapunovSpectrum {
    use crate::linear_algebra::{Mat, qr};

    let d = y0.len();
    let qr_interval = qr_interval.max(1);

    // State: y (d-vector) + Phi (d×d tangent matrix, stored column-major as d columns of d)
    let mut y = y0.to_vec();
    let mut t = 0.0;

    // --- Transient phase: just integrate the ODE ---
    let n_transient = (t_transient / dt).ceil() as usize;
    for _ in 0..n_transient {
        y = rk4_step(&f, t, &y, dt);
        t += dt;
    }

    // --- Computation phase: integrate ODE + variational equation ---
    // Phi = d×d identity (tangent vectors)
    let mut phi = vec![vec![0.0; d]; d]; // phi[col][row]
    for i in 0..d { phi[i][i] = 1.0; }

    let n_compute = (t_compute / dt).ceil() as usize;
    let n_qr = n_compute / qr_interval;
    if n_qr == 0 {
        return LyapunovSpectrum {
            exponents: vec![0.0; d], kaplan_yorke_dim: 0.0, sum: 0.0, n_positive: 0,
        };
    }

    let mut log_r = vec![0.0; d]; // accumulated log|R_ii|
    let mut total_time = 0.0;

    // Combined system dimension: d (state) + d*d (tangent matrix columns)
    let full_dim = d + d * d;

    for _ in 0..n_qr {
        // Integrate qr_interval steps using RK4 for the combined system
        for _ in 0..qr_interval {
            // Pack state: [y_0..y_{d-1}, phi_col0_row0..phi_col0_row_{d-1}, phi_col1_row0, ...]
            let mut state = vec![0.0; full_dim];
            state[..d].copy_from_slice(&y);
            for col in 0..d {
                for row in 0..d {
                    state[d + col * d + row] = phi[col][row];
                }
            }

            // Combined RHS: dy/dt = f(y), dΦ/dt = J(y)·Φ
            let combined_rhs = |_t: f64, s: &[f64]| -> Vec<f64> {
                let y_part = &s[..d];
                let mut rhs = vec![0.0; full_dim];

                // State part
                let fy = f(_t, y_part);
                rhs[..d].copy_from_slice(&fy);

                // Tangent part: dΦ_col/dt = J · Φ_col
                let j_flat = jac(_t, y_part);
                for col in 0..d {
                    for row in 0..d {
                        let mut sum = 0.0;
                        for k in 0..d {
                            sum += j_flat[row * d + k] * s[d + col * d + k];
                        }
                        rhs[d + col * d + row] = sum;
                    }
                }
                rhs
            };

            state = rk4_step(&combined_rhs, t, &state, dt);
            t += dt;

            // Unpack
            y.copy_from_slice(&state[..d]);
            for col in 0..d {
                for row in 0..d {
                    phi[col][row] = state[d + col * d + row];
                }
            }
        }

        total_time += qr_interval as f64 * dt;

        // QR decomposition of Phi (columns are tangent vectors)
        // Build Mat from columns
        let mut mat_data = vec![0.0; d * d];
        for col in 0..d {
            for row in 0..d {
                mat_data[row * d + col] = phi[col][row];
            }
        }
        let mat = Mat { rows: d, cols: d, data: mat_data };
        let qr_res = qr(&mat);

        // Accumulate log|R_ii|
        for i in 0..d {
            let r_ii = qr_res.r.data[i * d + i];
            log_r[i] += r_ii.abs().ln();
        }

        // Replace Phi with Q (orthonormalized tangent vectors)
        for col in 0..d {
            for row in 0..d {
                phi[col][row] = qr_res.q.data[row * d + col];
            }
        }
    }

    // Lyapunov exponents = accumulated log / total time
    let mut exponents: Vec<f64> = log_r.iter().map(|&lr| lr / total_time).collect();
    exponents.sort_by(|a, b| b.total_cmp(a));

    let sum = exponents.iter().sum();
    let n_positive = exponents.iter().filter(|&&e| e > 0.0).count();
    let kaplan_yorke_dim = kaplan_yorke(&exponents);

    LyapunovSpectrum { exponents, kaplan_yorke_dim, sum, n_positive }
}

/// Kaplan-Yorke dimension from sorted (descending) Lyapunov exponents.
/// D_KY = j + (Σ_{i=1}^{j} λ_i) / |λ_{j+1}|
/// where j is the largest index such that Σ_{i=1}^{j} λ_i ≥ 0.
fn kaplan_yorke(exponents: &[f64]) -> f64 {
    let mut running_sum = 0.0;
    let mut j = 0;
    let mut sum_at_j = 0.0;
    for (i, &e) in exponents.iter().enumerate() {
        running_sum += e;
        if running_sum >= 0.0 {
            j = i + 1;
            sum_at_j = running_sum;
        } else {
            break;
        }
    }
    if j >= exponents.len() || j == 0 {
        return j as f64;
    }
    j as f64 + sum_at_j / exponents[j].abs()
}

/// Single RK4 step for an ODE dy/dt = f(t, y).
fn rk4_step(f: &impl Fn(f64, &[f64]) -> Vec<f64>, t: f64, y: &[f64], h: f64) -> Vec<f64> {
    let d = y.len();
    let k1 = f(t, y);

    let mut y2 = vec![0.0; d];
    for i in 0..d { y2[i] = y[i] + 0.5 * h * k1[i]; }
    let k2 = f(t + 0.5 * h, &y2);

    let mut y3 = vec![0.0; d];
    for i in 0..d { y3[i] = y[i] + 0.5 * h * k2[i]; }
    let k3 = f(t + 0.5 * h, &y3);

    let mut y4 = vec![0.0; d];
    for i in 0..d { y4[i] = y[i] + h * k3[i]; }
    let k4 = f(t + h, &y4);

    let mut y_next = vec![0.0; d];
    for i in 0..d {
        y_next[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    y_next
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Sample Entropy ───────────────────────────────────────────────────

    #[test]
    fn sample_entropy_constant() {
        // Constant signal: SampEn should be very low (or infinite if no matches)
        let data: Vec<f64> = vec![1.0; 100];
        let se = sample_entropy(&data, 2, 0.2);
        // All templates match → A/B ≈ 1 → ln(1) ≈ 0
        assert!(se.abs() < 0.05 || se.is_nan(), "SampEn={}", se);
    }

    #[test]
    fn sample_entropy_periodic() {
        // Periodic signal should have low entropy
        let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let std = (data.iter().map(|&x| x.powi(2)).sum::<f64>() / data.len() as f64).sqrt();
        let se = sample_entropy(&data, 2, 0.2 * std);
        assert!(se < 1.0, "Periodic signal should have low SampEn={}", se);
    }

    #[test]
    fn sample_entropy_random_higher() {
        // Random signal should have higher entropy than periodic
        let periodic: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        // Pseudo-random via LCG
        let mut rng = 42u64;
        let random: Vec<f64> = (0..200).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f64 / (1u64 << 31) as f64
        }).collect();

        let std_p = (periodic.iter().map(|&x| x.powi(2)).sum::<f64>() / 200.0).sqrt();
        let std_r = (random.iter().map(|&x| x.powi(2)).sum::<f64>() / 200.0).sqrt();

        let se_periodic = sample_entropy(&periodic, 2, 0.2 * std_p);
        let se_random = sample_entropy(&random, 2, 0.2 * std_r);

        assert!(se_random > se_periodic,
            "Random SampEn={} should be > periodic SampEn={}", se_random, se_periodic);
    }

    // ── Approximate Entropy ──────────────────────────────────────────────

    #[test]
    fn approx_entropy_nonnegative() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let ae = approx_entropy(&data, 2, 0.3);
        assert!(ae >= 0.0, "ApEn should be ≥ 0, got {}", ae);
    }

    // ── Permutation Entropy ──────────────────────────────────────────────

    #[test]
    fn perm_entropy_monotone() {
        // Monotone increasing: only one pattern → PE = 0
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let pe = permutation_entropy(&data, 3, 1);
        assert!(pe.abs() < 1e-10, "Monotone PE should be 0, got {}", pe);
    }

    #[test]
    fn perm_entropy_normalized_range() {
        let mut rng = 42u64;
        let data: Vec<f64> = (0..1000).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f64
        }).collect();
        let npe = normalized_permutation_entropy(&data, 3, 1);
        assert!(npe >= 0.0 && npe <= 1.0, "Normalized PE should be in [0,1], got {}", npe);
        assert!(npe > 0.8, "Random data should have high NPE={}", npe);
    }

    // ── Hurst exponent ───────────────────────────────────────────────────

    #[test]
    fn hurst_white_noise() {
        // White noise (i.i.d. increments): H ≈ 0.5
        let mut rng = 42u64;
        let data: Vec<f64> = (0..2000).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        }).collect();
        let h = hurst_rs(&data);
        // For white noise, H should be around 0.5
        assert!(h > 0.2 && h < 0.9, "White noise H={} should be near 0.5", h);
    }

    // ── DFA ──────────────────────────────────────────────────────────────

    #[test]
    fn dfa_white_noise() {
        // White noise: α ≈ 0.5
        let mut rng = 123u64;
        let data: Vec<f64> = (0..500).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5
        }).collect();
        let alpha = dfa(&data, 10, 100);
        assert!(alpha > 0.2 && alpha < 0.8, "White noise DFA α={} should be near 0.5", alpha);
    }

    // ── Higuchi FD ───────────────────────────────────────────────────────

    #[test]
    fn higuchi_smooth_vs_noisy() {
        // Smooth sine wave
        let smooth: Vec<f64> = (0..500).map(|i| (i as f64 * 0.02).sin()).collect();
        let fd_smooth = higuchi_fd(&smooth, 10);

        // Noisy signal
        let mut rng = 77u64;
        let noisy: Vec<f64> = (0..500).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f64 / (1u64 << 31) as f64
        }).collect();
        let fd_noisy = higuchi_fd(&noisy, 10);

        // Key property: noisy FD > smooth FD (noise fills more space)
        assert!(fd_noisy > fd_smooth,
            "Noisy FD={} should be > smooth FD={}", fd_noisy, fd_smooth);
        // Both should be positive
        assert!(fd_smooth > 0.0, "Smooth FD={}", fd_smooth);
        assert!(fd_noisy > 0.0, "Noisy FD={}", fd_noisy);
    }

    // ── Lempel-Ziv ───────────────────────────────────────────────────────

    #[test]
    fn lz_periodic_low() {
        // Periodic: low complexity
        let data: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let lz = lempel_ziv_complexity(&data);
        assert!(lz < 0.5, "Periodic LZ={} should be low", lz);
    }

    #[test]
    fn lz_random_higher() {
        let mut rng = 42u64;
        let data: Vec<f64> = (0..200).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f64
        }).collect();
        let lz = lempel_ziv_complexity(&data);
        assert!(lz > 0.3, "Random LZ={} should be higher", lz);
    }

    // ── Correlation Dimension ────────────────────────────────────────────

    #[test]
    fn correlation_dim_sine() {
        // Sine wave embedded in 2D should have dim ≈ 1 (a circle)
        let data: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
        let d2 = correlation_dimension(&data, 2, 5);
        assert!(d2 > 0.5 && d2 < 2.5, "Sine D2={} should be near 1", d2);
    }

    // ── Largest Lyapunov ─────────────────────────────────────────────────

    #[test]
    fn lyapunov_periodic_near_zero() {
        // Periodic signal: λ₁ should be ≈ 0 or slightly negative
        let data: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
        let lam = largest_lyapunov(&data, 3, 5, 0.1);
        if !lam.is_nan() {
            assert!(lam < 0.5, "Periodic λ₁={} should be near 0", lam);
        }
    }

    // ── Cancellation canary ─────────────────────────────────────────────

    #[test]
    fn linear_fit_segment_cancellation_canary() {
        // y = 1e8 + 0.001*i. Centered formula preserves ~7 digits of slope precision.
        // Naive formula (n*sxy - sx*sy) / (n*sxx - sx*sx) would lose ~3 digits here.
        let segment: Vec<f64> = (0..100).map(|i| 1e8 + 0.001 * i as f64).collect();
        let (intercept, slope) = linear_fit_segment(&segment);
        assert!((slope - 0.001).abs() < 1e-8,
            "slope={slope} should be 0.001 (cancellation canary)");
        assert!((intercept - 1e8).abs() / 1e8 < 1e-8,
            "intercept={intercept} should be ~1e8");
    }

    // ── Lorenz attractor: statistics converge, trajectory doesn't ────

    #[test]
    fn lorenz_attractor_ergodic_convergence() {
        // The Lorenz system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz
        // With σ=10, ρ=28, β=8/3 → chaotic attractor.
        //
        // Key architectural insight: attract() for strange attractors returns
        // not a fixed point but a DISTRIBUTION. The ergodic theorem says
        // time averages converge even though the trajectory wanders forever.
        // The "limit" is accumulate(trajectory, All, Value, Add) / N.
        let sigma = 10.0;
        let rho = 28.0;
        let beta = 8.0 / 3.0;

        let lorenz = |_t: f64, y: &[f64]| -> Vec<f64> {
            vec![
                sigma * (y[1] - y[0]),
                y[0] * (rho - y[2]) - y[1],
                y[0] * y[1] - beta * y[2],
            ]
        };

        // Generate trajectory (skip transient)
        let n_steps = 50_000;
        let dt = 0.01;
        let (_ts, ys) = crate::numerical::rk4_system(lorenz, &[1.0, 1.0, 1.0], 0.0, n_steps as f64 * dt, n_steps);

        // Skip first 1000 steps (transient)
        let skip = 1000;
        let trajectory = &ys[skip..];

        // 1. Trajectory DOES NOT converge: it wanders between two lobes
        let x_first = trajectory[0][0];
        let x_last = trajectory[trajectory.len() - 1][0];
        // x oscillates between roughly -20 and +20; first and last are typically different
        // (We can't assert they differ because chaos makes any specific claim fragile,
        // but the mean of z should be near ρ-1 = 27)

        // 2. Time-averaged STATISTICS DO converge:
        // The ergodic average of z is approximately 23-24 for the standard Lorenz
        // parameters (not ρ-1=27, which is the fixed point, not the time average).
        let z_mean: f64 = trajectory.iter().map(|y| y[2]).sum::<f64>() / trajectory.len() as f64;
        assert!(z_mean > 15.0 && z_mean < 35.0,
            "Ergodic z_mean={z_mean:.2} should be in reasonable range for Lorenz");

        // Mean of x should be near 0 (symmetric attractor)
        let x_mean: f64 = trajectory.iter().map(|y| y[0]).sum::<f64>() / trajectory.len() as f64;
        assert!(x_mean.abs() < 3.0,
            "Ergodic x_mean={x_mean:.2} should be near 0 (symmetric)");

        // 3. Statistics converge PROGRESSIVELY: running mean of z stabilizes.
        // We use the full-run mean as the "true" ergodic average and check
        // that partial estimates approach it.
        let mut z_sum = 0.0;
        let mut z_mean_at_1k = 0.0;
        let mut z_mean_at_10k = 0.0;
        for (i, y) in trajectory.iter().enumerate() {
            z_sum += y[2];
            if i == 999 { z_mean_at_1k = z_sum / 1000.0; }
            if i == 9999 { z_mean_at_10k = z_sum / 10000.0; }
        }

        let err_1k = (z_mean_at_1k - z_mean).abs();
        let err_10k = (z_mean_at_10k - z_mean).abs();

        eprintln!("═══ Lorenz attractor ergodic convergence ═══");
        eprintln!("z_mean at  1K: {z_mean_at_1k:.3} (err from ergodic={err_1k:.3})");
        eprintln!("z_mean at 10K: {z_mean_at_10k:.3} (err from ergodic={err_10k:.3})");
        eprintln!("z_mean at 49K: {z_mean:.3} (ergodic estimate)");
        eprintln!("x_mean: {x_mean:.3} (should be ~0)");
        eprintln!("x range: [{x_first:.1}, ..., {x_last:.1}]");

        // Progressive convergence: 10K should be closer to full mean than 1K
        assert!(err_10k < err_1k + 1.0,
            "10K avg should approach ergodic: err_1k={err_1k:.3}, err_10k={err_10k:.3}");
    }

    // ── Lyapunov spectrum ───────────────────────────────────────────────

    /// Lorenz system: dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz
    fn lorenz(_t: f64, y: &[f64]) -> Vec<f64> {
        let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
        let (x, yv, z) = (y[0], y[1], y[2]);
        vec![
            sigma * (yv - x),
            x * (rho - z) - yv,
            x * yv - beta * z,
        ]
    }

    /// Lorenz Jacobian (row-major 3×3)
    fn lorenz_jac(_t: f64, y: &[f64]) -> Vec<f64> {
        let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
        let (x, yv, z) = (y[0], y[1], y[2]);
        vec![
            -sigma,    sigma,  0.0,
            rho - z,   -1.0,  -x,
            yv,         x,    -beta,
        ]
    }

    #[test]
    fn lyapunov_spectrum_lorenz() {
        // Known Lorenz exponents: λ₁ ≈ 0.906, λ₂ ≈ 0, λ₃ ≈ -14.57
        // Sum ≈ -(σ + 1 + β) = -(10 + 1 + 8/3) ≈ -13.667 (trace of Jacobian)
        let y0 = [1.0, 1.0, 1.0];
        let spec = lyapunov_spectrum(
            lorenz, lorenz_jac,
            &y0,
            20.0,   // transient
            200.0,  // compute
            0.005,  // dt
            20,     // QR every 20 steps (= 0.1 time units)
        );

        eprintln!("═══ Lorenz Lyapunov spectrum ═══");
        eprintln!("λ₁ = {:.3}, λ₂ = {:.3}, λ₃ = {:.3}", spec.exponents[0], spec.exponents[1], spec.exponents[2]);
        eprintln!("sum = {:.3} (expect ≈ -13.667)", spec.sum);
        eprintln!("D_KY = {:.3} (expect ≈ 2.06)", spec.kaplan_yorke_dim);
        eprintln!("n_positive = {}", spec.n_positive);

        // λ₁ should be positive (chaos)
        assert!(spec.exponents[0] > 0.3, "λ₁ = {} should be > 0.3", spec.exponents[0]);
        // λ₂ should be near zero (neutral direction along flow)
        assert!(spec.exponents[1].abs() < 1.0, "λ₂ = {} should be near 0", spec.exponents[1]);
        // λ₃ should be strongly negative (dissipation)
        assert!(spec.exponents[2] < -5.0, "λ₃ = {} should be < -5", spec.exponents[2]);
        // Sum should be negative (dissipative system)
        assert!(spec.sum < -10.0, "sum = {} should be ≈ -13.667", spec.sum);
        // Exactly 1 positive exponent
        assert_eq!(spec.n_positive, 1);
        // Kaplan-Yorke dimension should be between 2 and 3 (strange attractor)
        assert!(spec.kaplan_yorke_dim > 2.0 && spec.kaplan_yorke_dim < 3.0,
            "D_KY = {} should be in (2, 3)", spec.kaplan_yorke_dim);
    }

    #[test]
    fn lyapunov_spectrum_damped_harmonic() {
        // Damped harmonic oscillator: dx/dt = v, dv/dt = -ω²x - γv
        // Exponents: (-γ ± sqrt(γ²-4ω²))/2 (both negative for γ > 0)
        let omega = 2.0;
        let gamma = 0.5;
        let f = move |_t: f64, y: &[f64]| -> Vec<f64> {
            vec![y[1], -omega * omega * y[0] - gamma * y[1]]
        };
        let j = move |_t: f64, _y: &[f64]| -> Vec<f64> {
            vec![0.0, 1.0, -omega * omega, -gamma]
        };

        let spec = lyapunov_spectrum(f, j, &[1.0, 0.0], 5.0, 50.0, 0.01, 10);

        eprintln!("═══ Damped harmonic Lyapunov spectrum ═══");
        eprintln!("λ₁ = {:.4}, λ₂ = {:.4}", spec.exponents[0], spec.exponents[1]);
        eprintln!("sum = {:.4} (expect = -γ = -{})", spec.sum, gamma);

        // Both exponents should be negative (stable)
        assert!(spec.n_positive == 0, "Damped oscillator has no positive exponents");
        // Sum of exponents = trace of Jacobian = -γ
        assert!((spec.sum - (-gamma)).abs() < 0.2,
            "sum = {} should be ≈ -{}", spec.sum, gamma);
    }

    #[test]
    fn lyapunov_ode_vs_timeseries_lorenz() {
        // Cross-validate: compute λ₁ from ODE (Benettin) and from time series (Rosenstein)
        // Both should give λ₁ ≈ 0.9, though Rosenstein from embedding is noisier.

        // 1. ODE-based λ₁
        let spec = lyapunov_spectrum(
            lorenz, lorenz_jac,
            &[1.0, 1.0, 1.0],
            20.0, 200.0, 0.005, 20,
        );
        let lambda_ode = spec.exponents[0];

        // 2. Generate Lorenz time series (x-component)
        let dt = 0.01;
        let n_transient = 2000;
        let n_points = 10000;
        let mut y = vec![1.0, 1.0, 1.0];
        for _ in 0..n_transient {
            y = rk4_step(&lorenz, 0.0, &y, dt);
        }
        let mut x_series = Vec::with_capacity(n_points);
        for _ in 0..n_points {
            y = rk4_step(&lorenz, 0.0, &y, dt);
            x_series.push(y[0]);
        }

        // 3. Time-series-based λ₁ (Rosenstein)
        let lambda_ts = largest_lyapunov(&x_series, 5, 2, dt);

        eprintln!("═══ ODE vs Time Series λ₁ ═══");
        eprintln!("ODE (Benettin):     λ₁ = {:.4}", lambda_ode);
        eprintln!("Time series (Rosen): λ₁ = {:.4}", lambda_ts);

        // Both should be positive (chaos indicator)
        assert!(lambda_ode > 0.5, "ODE λ₁ = {} should be > 0.5", lambda_ode);
        assert!(lambda_ts > 0.0, "TS λ₁ = {} should be > 0 (positive = chaos)", lambda_ts);
        // They don't need to match exactly — Rosenstein from scalar embedding is approximate
        // But both should be in the right ballpark (0.3 to 2.0)
        assert!(lambda_ts < 3.0, "TS λ₁ = {} seems too high", lambda_ts);
    }
}
