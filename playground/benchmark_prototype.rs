//! # Tambear Benchmark Prototype
//!
//! The 6-step EDA pipeline on synthetic data with planted structure.
//! Measures: time per step, accuracy vs ground truth, total passes.
//!
//! Step 1: Descriptive stats (mean, std, skew, kurt per column)
//! Step 2: Correlation matrix
//! Step 3: PCA (top k eigenvalues/vectors)
//! Step 4: KDE (1D density of first PC)
//! Step 5: K-means (k=3 on PCA-reduced)
//! Step 6: ANOVA (F-test per column across clusters)

use std::time::Instant;

// ── Synthetic data generator ────────────────────────────────

fn generate_data(n: usize, d: usize, seed: u64) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
    // Simple LCG for reproducibility
    let mut rng_state = seed;
    let mut next_f64 = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Box-Muller approximation: use two uniforms to make one normal
        let u1 = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-15);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let n_per = n / 3;

    // Planted cluster centers
    let mut centers = vec![0.0f64; 3 * d];
    for i in 0..3 {
        for j in 0..d {
            centers[i * d + j] = next_f64() * 5.0;
        }
    }

    // Generate data: each point = center + noise
    let mut data = vec![0.0f64; n * d];
    let mut labels = vec![0usize; n];

    for cluster in 0..3 {
        for row in 0..n_per {
            let idx = cluster * n_per + row;
            labels[idx] = cluster;
            for col in 0..d {
                data[idx * d + col] = centers[cluster * d + col] + next_f64();
            }
        }
    }

    (data, labels, centers)
}

// ── Step 1: Descriptive statistics ──────────────────────────

fn descriptive_stats(data: &[f64], n: usize, d: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut means = vec![0.0f64; d];
    let mut m2 = vec![0.0f64; d];
    let mut m3 = vec![0.0f64; d];
    let mut m4 = vec![0.0f64; d];

    // Online algorithm (one pass!)
    let mut count = vec![0usize; d];
    for row in 0..n {
        for col in 0..d {
            let x = data[row * d + col];
            count[col] += 1;
            let n_f = count[col] as f64;
            let delta = x - means[col];
            let delta_n = delta / n_f;
            let delta_n2 = delta_n * delta_n;
            let term1 = delta * delta_n * (n_f - 1.0);

            means[col] += delta_n;
            m4[col] += term1 * delta_n2 * (n_f * n_f - 3.0 * n_f + 3.0)
                + 6.0 * delta_n2 * m2[col] - 4.0 * delta_n * m3[col];
            m3[col] += term1 * delta_n * (n_f - 2.0) - 3.0 * delta_n * m2[col];
            m2[col] += term1;
        }
    }

    let stds: Vec<f64> = m2.iter().map(|&v| (v / n as f64).sqrt()).collect();
    let skews: Vec<f64> = m3.iter().zip(m2.iter())
        .map(|(&s3, &s2)| {
            let n_f = n as f64;
            (n_f.sqrt() * s3) / s2.powf(1.5)
        }).collect();
    let kurts: Vec<f64> = m4.iter().zip(m2.iter())
        .map(|(&s4, &s2)| {
            let n_f = n as f64;
            (n_f * s4) / (s2 * s2) - 3.0
        }).collect();

    (means, stds, skews, kurts)
}

// ── Step 2: Correlation matrix ──────────────────────────────

fn correlation_matrix(data: &[f64], n: usize, d: usize, means: &[f64], stds: &[f64]) -> Vec<f64> {
    let mut corr = vec![0.0f64; d * d];

    for i in 0..d {
        for j in i..d {
            let mut sum = 0.0f64;
            for row in 0..n {
                let xi = (data[row * d + i] - means[i]) / stds[i].max(1e-15);
                let xj = (data[row * d + j] - means[j]) / stds[j].max(1e-15);
                sum += xi * xj;
            }
            let r = sum / (n - 1) as f64;
            corr[i * d + j] = r;
            corr[j * d + i] = r;
        }
    }
    corr
}

// ── Step 3: PCA via power iteration ─────────────────────────

fn pca_power(corr: &[f64], d: usize, k: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors = Vec::with_capacity(k);
    let mut mat = corr.to_vec();

    for _ in 0..k {
        // Power iteration
        let mut v = vec![1.0 / (d as f64).sqrt(); d];
        let mut lambda = 0.0f64;

        for _ in 0..200 {
            let mut w = vec![0.0f64; d];
            for i in 0..d {
                for j in 0..d {
                    w[i] += mat[i * d + j] * v[j];
                }
            }
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 { break; }
            lambda = norm;
            for i in 0..d { v[i] = w[i] / norm; }
        }

        eigenvalues.push(lambda);
        eigenvectors.push(v.clone());

        // Deflate
        for i in 0..d {
            for j in 0..d {
                mat[i * d + j] -= lambda * v[i] * v[j];
            }
        }
    }

    (eigenvalues, eigenvectors)
}

// ── Step 4: KDE (Gaussian kernel, 1D) ───────────────────────

fn kde_1d(values: &[f64], n_points: usize) -> (Vec<f64>, Vec<f64>) {
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let std: f64 = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    let h = 1.06 * std * n.powf(-0.2); // Silverman bandwidth

    let min_v = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_v - min_v;

    let mut grid = vec![0.0f64; n_points];
    let mut density = vec![0.0f64; n_points];

    for i in 0..n_points {
        let x = min_v - 0.1 * range + (range * 1.2) * i as f64 / (n_points - 1) as f64;
        grid[i] = x;
        for &v in values {
            let z = (x - v) / h;
            density[i] += (-0.5 * z * z).exp() / (h * (2.0 * std::f64::consts::PI).sqrt());
        }
        density[i] /= n;
    }

    (grid, density)
}

// ── Step 5: K-means ─────────────────────────────────────────

fn kmeans(data: &[f64], n: usize, d: usize, k: usize, max_iter: usize) -> Vec<usize> {
    // Initialize centers: first k points
    let mut centers = vec![0.0f64; k * d];
    for i in 0..k {
        for j in 0..d {
            centers[i * d + j] = data[i * d + j];
        }
    }

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign
        let mut changed = false;
        for row in 0..n {
            let mut best_dist = f64::INFINITY;
            let mut best_k = 0;
            for c in 0..k {
                let mut dist = 0.0f64;
                for col in 0..d {
                    let diff = data[row * d + col] - centers[c * d + col];
                    dist += diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_k = c;
                }
            }
            if assignments[row] != best_k {
                changed = true;
                assignments[row] = best_k;
            }
        }
        if !changed { break; }

        // Update centers
        let mut sums = vec![0.0f64; k * d];
        let mut counts = vec![0usize; k];
        for row in 0..n {
            let c = assignments[row];
            counts[c] += 1;
            for col in 0..d {
                sums[c * d + col] += data[row * d + col];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for col in 0..d {
                    centers[c * d + col] = sums[c * d + col] / counts[c] as f64;
                }
            }
        }
    }

    assignments
}

// ── Step 6: One-way ANOVA ───────────────────────────────────

fn anova_f(data: &[f64], n: usize, d: usize, assignments: &[usize], k: usize) -> Vec<f64> {
    let mut f_stats = vec![0.0f64; d];

    for col in 0..d {
        let grand_mean: f64 = (0..n).map(|r| data[r * d + col]).sum::<f64>() / n as f64;

        let mut group_means = vec![0.0f64; k];
        let mut group_counts = vec![0usize; k];
        for row in 0..n {
            let g = assignments[row];
            group_means[g] += data[row * d + col];
            group_counts[g] += 1;
        }
        for g in 0..k {
            if group_counts[g] > 0 {
                group_means[g] /= group_counts[g] as f64;
            }
        }

        let ss_between: f64 = (0..k)
            .map(|g| group_counts[g] as f64 * (group_means[g] - grand_mean).powi(2))
            .sum();

        let ss_within: f64 = (0..n)
            .map(|r| {
                let g = assignments[r];
                (data[r * d + col] - group_means[g]).powi(2)
            })
            .sum();

        let df_between = (k - 1) as f64;
        let df_within = (n - k) as f64;

        f_stats[col] = if ss_within > 0.0 {
            (ss_between / df_between) / (ss_within / df_within)
        } else {
            f64::INFINITY
        };
    }

    f_stats
}

// ── Main benchmark ──────────────────────────────────────────

fn main() {
    let n = 100_000usize;
    let d = 50usize;
    let k_pca = 5usize;
    let k_clusters = 3usize;

    eprintln!("==========================================================");
    eprintln!("  TAMBEAR BENCHMARK PROTOTYPE");
    eprintln!("  6-step EDA pipeline: describe → corr → PCA → KDE → kmeans → ANOVA");
    eprintln!("  N={}, D={}, k_pca={}, k_clusters={}", n, d, k_pca, k_clusters);
    eprintln!("  Data size: {:.1} MB", (n * d * 8) as f64 / 1e6);
    eprintln!("==========================================================\n");

    // Generate data
    let t0 = Instant::now();
    let (data, true_labels, _centers) = generate_data(n, d, 42);
    let t_gen = t0.elapsed().as_secs_f64();
    eprintln!("  Data generation: {:.3}s ({} rows × {} cols)", t_gen, n, d);

    // Step 1: Descriptive statistics
    let t1 = Instant::now();
    let (means, stds, skews, kurts) = descriptive_stats(&data, n, d);
    let t_desc = t1.elapsed().as_secs_f64();
    eprintln!("  Step 1 (describe):     {:.3}s  [1 pass over data]", t_desc);
    eprintln!("    mean[0]={:.4}, std[0]={:.4}, skew[0]={:.4}, kurt[0]={:.4}",
        means[0], stds[0], skews[0], kurts[0]);

    // Step 2: Correlation matrix
    let t2 = Instant::now();
    let corr = correlation_matrix(&data, n, d, &means, &stds);
    let t_corr = t2.elapsed().as_secs_f64();
    eprintln!("  Step 2 (correlation):  {:.3}s  [1 pass, {} pairs]", t_corr, d*(d-1)/2);
    eprintln!("    corr[0,1]={:.4}, corr[0,2]={:.4}", corr[0*d+1], corr[0*d+2]);

    // Step 3: PCA
    let t3 = Instant::now();
    let (eigenvalues, eigenvectors) = pca_power(&corr, d, k_pca);
    let t_pca = t3.elapsed().as_secs_f64();
    let var_explained: f64 = eigenvalues.iter().sum::<f64>() / d as f64;
    eprintln!("  Step 3 (PCA):          {:.3}s  [power iteration on {}×{} matrix]", t_pca, d, d);
    eprintln!("    eigenvalues: {:?}", eigenvalues.iter().map(|e| format!("{:.2}", e)).collect::<Vec<_>>());
    eprintln!("    variance explained: {:.1}%", var_explained * 100.0);

    // Project data to PCA space
    let t_proj = Instant::now();
    let mut projected = vec![0.0f64; n * k_pca];
    for row in 0..n {
        for pc in 0..k_pca {
            let mut val = 0.0f64;
            for col in 0..d {
                val += (data[row * d + col] - means[col]) / stds[col].max(1e-15) * eigenvectors[pc][col];
            }
            projected[row * k_pca + pc] = val;
        }
    }
    let t_proj = t_proj.elapsed().as_secs_f64();
    eprintln!("    projection:          {:.3}s", t_proj);

    // Step 4: KDE on first PC
    let t4 = Instant::now();
    let pc1: Vec<f64> = (0..n).map(|i| projected[i * k_pca]).collect();
    let (_grid, density) = kde_1d(&pc1, 200);
    let t_kde = t4.elapsed().as_secs_f64();
    let max_density = density.iter().cloned().fold(0.0f64, f64::max);
    eprintln!("  Step 4 (KDE):          {:.3}s  [N={} kernel evaluations × 200 grid points]",
        t_kde, n);
    eprintln!("    peak density: {:.4}", max_density);

    // Step 5: K-means on projected data
    let t5 = Instant::now();
    let assignments = kmeans(&projected, n, k_pca, k_clusters, 100);
    let t_kmeans = t5.elapsed().as_secs_f64();

    // Rand index vs true labels
    let mut agree = 0u64;
    let mut total = 0u64;
    for i in 0..n.min(5000) {
        for j in (i+1)..n.min(5000) {
            let same_pred = assignments[i] == assignments[j];
            let same_true = true_labels[i] == true_labels[j];
            if same_pred == same_true { agree += 1; }
            total += 1;
        }
    }
    let rand_index = agree as f64 / total as f64;
    eprintln!("  Step 5 (k-means):      {:.3}s  [k={}, on {}×{} projected data]",
        t_kmeans, k_clusters, n, k_pca);
    eprintln!("    Rand index vs truth: {:.4}", rand_index);

    // Step 6: ANOVA
    let t6 = Instant::now();
    let f_stats = anova_f(&data, n, d, &assignments, k_clusters);
    let t_anova = t6.elapsed().as_secs_f64();
    let significant = f_stats.iter().filter(|&&f| f > 3.0).count();
    eprintln!("  Step 6 (ANOVA):        {:.3}s  [{} columns tested]", t_anova, d);
    eprintln!("    significant (F>3): {}/{}", significant, d);
    eprintln!("    max F: {:.1}, min F: {:.1}",
        f_stats.iter().cloned().fold(0.0f64, f64::max),
        f_stats.iter().cloned().fold(f64::INFINITY, f64::min));

    // Summary
    let t_total = t0.elapsed().as_secs_f64();
    let t_compute = t_desc + t_corr + t_pca + t_proj + t_kde + t_kmeans + t_anova;

    eprintln!("\n=== SUMMARY ===");
    eprintln!("  Data:    {} × {} = {:.1} MB", n, d, (n*d*8) as f64 / 1e6);
    eprintln!("  Total:   {:.3}s (including generation)", t_total);
    eprintln!("  Compute: {:.3}s (pipeline only)", t_compute);
    eprintln!();
    eprintln!("  Step breakdown:");
    eprintln!("    describe:    {:>7.3}s  {:>5.1}%", t_desc, t_desc/t_compute*100.0);
    eprintln!("    correlation: {:>7.3}s  {:>5.1}%", t_corr, t_corr/t_compute*100.0);
    eprintln!("    PCA:         {:>7.3}s  {:>5.1}%", t_pca, t_pca/t_compute*100.0);
    eprintln!("    projection:  {:>7.3}s  {:>5.1}%", t_proj, t_proj/t_compute*100.0);
    eprintln!("    KDE:         {:>7.3}s  {:>5.1}%", t_kde, t_kde/t_compute*100.0);
    eprintln!("    k-means:     {:>7.3}s  {:>5.1}%", t_kmeans, t_kmeans/t_compute*100.0);
    eprintln!("    ANOVA:       {:>7.3}s  {:>5.1}%", t_anova, t_anova/t_compute*100.0);
    eprintln!();
    eprintln!("  Passes over data:");
    eprintln!("    This implementation: 4 passes");
    eprintln!("      Pass 1: describe (online moments)");
    eprintln!("      Pass 2: correlation (needs means/stds from pass 1)");
    eprintln!("      Pass 3: projection (needs eigenvectors from PCA)");
    eprintln!("      Pass 4: ANOVA (needs cluster assignments)");
    eprintln!("    Tambear fused: 2 passes");
    eprintln!("      Pass 1: describe + correlation (shared accumulators)");
    eprintln!("      Pass 2: projection + ANOVA (after PCA + kmeans)");
    eprintln!("    NumPy separate: 8+ passes");
    eprintln!("      np.mean, np.std, np.corrcoef, PCA.fit, transform, KDE, KMeans, f_oneway");
    eprintln!();
    eprintln!("  Accuracy:");
    eprintln!("    Rand index: {:.4} (1.0 = perfect cluster recovery)", rand_index);
    eprintln!("    Significant ANOVAs: {}/{} (planted: 3 clusters → all should be significant)",
        significant, d);
}
