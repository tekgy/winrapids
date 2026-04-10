//! Scale Ladder Benchmarks
//!
//! Push every algorithm to its breaking point. Document where each fails.
//! Run: cargo test --test scale_ladder --release -- --ignored --nocapture
//!
//! Scale ladder:
//!   n=10, 1K, 100K, 1M, 10M, 100M, 1B (descriptive stats)
//!   n=10, 50, 100, 200, 500, 1000 (linear algebra — O(n^3))
//!   n=100, 1K, 10K, 100K (graph algorithms)
//!   n=10, 50, 100, 200, 500, 1000, 2000 (matrix multiply)

use tambear::descriptive::*;
use tambear::linear_algebra::*;
use tambear::graph::{self, Graph};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════
// Deterministic data generators (no rand crate)
// ═══════════════════════════════════════════════════════════════════════

/// Deterministic f64 values using sin-based formula.
fn make_f64_vec(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64 * 1.23456789 + 0.1).sin()).collect()
}

/// Kahan-summed mean for reference accuracy.
fn kahan_mean(data: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &x in data {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum / data.len() as f64
}

/// Deterministic n x n matrix with interesting values.
fn make_mat(n: usize) -> Mat {
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let v = ((i * 73 + j * 37 + 11) as f64 * 0.0031415926).sin();
            data.push(v);
        }
    }
    Mat::from_vec(n, n, data)
}

/// Deterministic symmetric positive definite matrix: A^T * A + eps * I
fn make_spd_mat(n: usize) -> Mat {
    let a = make_mat(n);
    let at = a.t();
    let mut ata = mat_mul(&at, &a);
    // Add eps*I for positive definiteness
    for i in 0..n {
        let cur = ata.get(i, i);
        ata.set(i, i, cur + n as f64 * 0.1);
    }
    ata
}

/// Deterministic sparse graph: each node connects to ~5 neighbors.
fn make_sparse_graph(n: usize) -> Graph {
    let mut g = Graph::new(n);
    for i in 0..n {
        // Connect to ~5 deterministic neighbors
        for k in 1..=5 {
            let j = (i * 7 + k * 13 + 37) % n;
            if j != i {
                let w = ((i * 31 + j * 17) % 100) as f64 + 1.0;
                g.add_edge(i, j, w);
            }
        }
    }
    g
}

/// Deterministic sparse UNDIRECTED graph.
fn make_undirected_sparse_graph(n: usize) -> Graph {
    let mut g = Graph::new(n);
    for i in 0..n {
        for k in 1..=3 {
            let j = (i * 7 + k * 13 + 37) % n;
            if j > i {
                let w = ((i * 31 + j * 17) % 100) as f64 + 1.0;
                g.add_undirected(i, j, w);
            }
        }
    }
    // Ensure connected: chain
    for i in 0..n.saturating_sub(1) {
        g.add_undirected(i, i + 1, 1.0);
    }
    g
}

/// Format a number with comma separators.
fn fmt_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Format throughput as elements/sec with SI suffix.
fn fmt_throughput(elements: usize, secs: f64) -> String {
    if secs <= 0.0 {
        return "inf".to_string();
    }
    let t = elements as f64 / secs;
    if t >= 1e9 {
        format!("{:.2} G/s", t / 1e9)
    } else if t >= 1e6 {
        format!("{:.2} M/s", t / 1e6)
    } else if t >= 1e3 {
        format!("{:.2} K/s", t / 1e3)
    } else {
        format!("{:.2} /s", t)
    }
}

/// Format time nicely.
fn fmt_time(secs: f64) -> String {
    if secs < 0.001 {
        format!("{:.1} us", secs * 1e6)
    } else if secs < 1.0 {
        format!("{:.2} ms", secs * 1e3)
    } else if secs < 60.0 {
        format!("{:.3} s", secs)
    } else {
        format!("{:.1} min", secs / 60.0)
    }
}

/// Format GFLOPS.
fn fmt_gflops(flops: f64, secs: f64) -> String {
    if secs <= 0.0 {
        return "inf".to_string();
    }
    let gf = flops / secs / 1e9;
    format!("{:.3} GFLOPS", gf)
}

// ═══════════════════════════════════════════════════════════════════════
// 1. Descriptive Statistics — Scale Ladder
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_descriptive_stats_scale_ladder() {
    let scales: &[(usize, &str)] = &[
        (10, "10"),
        (1_000, "1K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
        (10_000_000, "10M"),
        (100_000_000, "100M"),
    ];

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("DESCRIPTIVE STATISTICS -- Scale Ladder");
    eprintln!("========================================================================");
    eprintln!("{:<14} {:>12} {:>16} {:>12} {:>14}",
             "n", "time", "throughput", "memory(MB)", "mean_error");
    eprintln!("------------------------------------------------------------------------");

    let mut prev_secs = 0.0_f64;

    for &(n, label) in scales {
        let mem_mb = (n * 8) as f64 / 1_000_000.0;

        // Skip if extrapolated time > 60s
        if prev_secs > 0.0 {
            let estimated = prev_secs * 10.0; // linear scaling
            if estimated > 60.0 {
                eprintln!("{:<14} {:>12} {:>16} {:>12.1} {:>14}",
                         label, "SKIPPED", format!("est ~{:.0}s", estimated), mem_mb, "--");
                continue;
            }
        }

        // Generate data
        let data = make_f64_vec(n);

        // Reference mean via Kahan summation
        let ref_mean = kahan_mean(&data);

        // Time moments_ungrouped
        let t0 = Instant::now();
        let stats = moments_ungrouped(&data);
        let elapsed = t0.elapsed().as_secs_f64();

        prev_secs = elapsed;

        // Compute relative error in mean
        let tambear_mean = stats.mean();
        let rel_err = if ref_mean.abs() > 1e-15 {
            ((tambear_mean - ref_mean) / ref_mean).abs()
        } else {
            (tambear_mean - ref_mean).abs()
        };

        eprintln!("{:<14} {:>12} {:>16} {:>12.1} {:>14.2e}",
                 label, fmt_time(elapsed), fmt_throughput(n, elapsed), mem_mb, rel_err);
        eprintln!("    count={}, mean={:.10}, std={:.10}, skew={:.6}, kurt={:.6}",
                 stats.n(), stats.mean(), stats.variance(0).sqrt(), stats.skewness(true), stats.kurtosis(true, true));
    }

    eprintln!("========================================================================");
    eprintln!();
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Descriptive Statistics — 1 Billion Elements
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_descriptive_stats_1b() {
    eprintln!();
    eprintln!("========================================================================");
    eprintln!("DESCRIPTIVE STATISTICS -- 1 Billion Elements (~8 GB)");
    eprintln!("========================================================================");

    let n: usize = 1_000_000_000;
    let mem_mb = (n * 8) as f64 / 1_000_000.0;
    eprintln!("Target: {} elements, {:.0} MB ({:.1} GB)", fmt_num(n), mem_mb, mem_mb / 1000.0);
    eprintln!();

    // Try to allocate
    let alloc_start = Instant::now();
    let data_result = std::panic::catch_unwind(|| {
        // Allocate and fill
        let mut data = Vec::<f64>::new();
        match data.try_reserve_exact(n) {
            Ok(()) => {},
            Err(e) => {
                eprintln!("ALLOCATION FAILED: {} (requested {:.1} GB)", e, mem_mb / 1000.0);
                return None;
            }
        }
        // Fill deterministically in chunks to show progress
        let chunk = 100_000_000;
        for start in (0..n).step_by(chunk) {
            let end = (start + chunk).min(n);
            for i in start..end {
                data.push((i as f64 * 1.23456789 + 0.1).sin());
            }
            eprintln!("  allocated {}/{} ({:.0}%)",
                     fmt_num(end), fmt_num(n), end as f64 / n as f64 * 100.0);
        }
        Some(data)
    });

    match data_result {
        Ok(Some(data)) => {
            let alloc_time = alloc_start.elapsed().as_secs_f64();
            eprintln!("Allocation + fill: {}", fmt_time(alloc_time));

            // Compute
            let t0 = Instant::now();
            let stats = moments_ungrouped(&data);
            let elapsed = t0.elapsed().as_secs_f64();

            eprintln!();
            eprintln!("moments_ungrouped on 1B elements:");
            eprintln!("  Time:       {}", fmt_time(elapsed));
            eprintln!("  Throughput: {}", fmt_throughput(n, elapsed));
            eprintln!("  Mean:       {:.15}", stats.mean());
            eprintln!("  Std:        {:.15}", stats.variance(0).sqrt());
            eprintln!("  Skew:       {:.10}", stats.skewness(true));
            eprintln!("  Kurt:       {:.10}", stats.kurtosis(true, true));
            eprintln!("  Min:        {:.15}", stats.min);
            eprintln!("  Max:        {:.15}", stats.max);

            // Reference mean
            let ref_mean = kahan_mean(&data);
            let rel_err = if ref_mean.abs() > 1e-15 {
                ((stats.mean() - ref_mean) / ref_mean).abs()
            } else {
                (stats.mean() - ref_mean).abs()
            };
            eprintln!("  Mean relative error vs Kahan: {:.2e}", rel_err);
        },
        Ok(None) => {
            eprintln!("Allocation reported failure (see above). Memory ceiling reached.");
        },
        Err(_) => {
            eprintln!("PANICKED during allocation. Memory ceiling reached at {:.1} GB.", mem_mb / 1000.0);
        }
    }

    eprintln!("========================================================================");
    eprintln!();
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Sort + Quantile — Scale Ladder
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_sort_quantile_scale_ladder() {
    let scales: &[(usize, &str)] = &[
        (10, "10"),
        (1_000, "1K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
        (10_000_000, "10M"),
        (100_000_000, "100M"),
    ];

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("SORT + QUANTILE -- Scale Ladder");
    eprintln!("========================================================================");
    eprintln!("{:<10} {:>12} {:>12} {:>12} {:>12} {:>10}",
             "n", "sort", "median", "quartiles", "iqr", "MB");
    eprintln!("------------------------------------------------------------------------");

    let mut prev_sort_secs = 0.0_f64;

    for &(n, label) in scales {
        let mem_mb = (n * 8) as f64 / 1_000_000.0;

        // Skip if sort would take > 60s (O(n log n) scaling)
        if prev_sort_secs > 0.0 && n > 1_000 {
            let scale_factor = (n as f64) / ((n / 10) as f64);
            let log_factor = (n as f64).ln() / ((n / 10) as f64).ln();
            let estimated = prev_sort_secs * scale_factor * log_factor / 10.0;
            if estimated > 60.0 {
                eprintln!("{:<10} {:>12} {:>12} {:>12} {:>12} {:>10.1}",
                         label, "SKIPPED", "--", "--", "--", mem_mb);
                eprintln!("    estimated sort time: ~{:.0}s", estimated);
                continue;
            }
        }

        let mut data = make_f64_vec(n);

        // Time sort
        let t0 = Instant::now();
        data.sort_by(|a, b| a.total_cmp(b));
        let sort_time = t0.elapsed().as_secs_f64();
        prev_sort_secs = sort_time;

        // Time median
        let t1 = Instant::now();
        let med = median(&data);
        let median_time = t1.elapsed().as_secs_f64();

        // Time quartiles
        let t2 = Instant::now();
        let (q1, q2, q3) = quartiles(&data);
        let quartiles_time = t2.elapsed().as_secs_f64();

        // Time IQR
        let t3 = Instant::now();
        let iqr_val = iqr(&data);
        let iqr_time = t3.elapsed().as_secs_f64();

        eprintln!("{:<10} {:>12} {:>12} {:>12} {:>12} {:>10.1}",
                 label, fmt_time(sort_time), fmt_time(median_time),
                 fmt_time(quartiles_time), fmt_time(iqr_time), mem_mb);
        eprintln!("    median={:.10}, Q1={:.10}, Q3={:.10}, IQR={:.10}",
                 med, q1, q3, iqr_val);

        // Also time quantile at various points
        let t4 = Instant::now();
        let p01 = quantile(&data, 0.01, QuantileMethod::Linear);
        let p99 = quantile(&data, 0.99, QuantileMethod::Linear);
        let quantile_time = t4.elapsed().as_secs_f64();
        eprintln!("    P01={:.10}, P99={:.10}, quantile_time={}", p01, p99, fmt_time(quantile_time));
        let _ = q2; // suppress warning
    }

    eprintln!("========================================================================");
    eprintln!();
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Linear Algebra — Scale Ladder (tambear vs faer)
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_linear_algebra_scale_ladder() {
    let scales: &[(usize, &str)] = &[
        (10, "10"),
        (50, "50"),
        (100, "100"),
        (200, "200"),
        (500, "500"),
        (1000, "1000"),
    ];

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("LINEAR ALGEBRA -- Scale Ladder (tambear vs faer)");
    eprintln!("========================================================================");

    // Track previous times for extrapolation (keyed by operation)
    let mut prev_svd = 0.0_f64;
    let mut prev_qr = 0.0_f64;
    let mut prev_lu = 0.0_f64;
    let mut prev_chol = 0.0_f64;
    let mut prev_eigen = 0.0_f64;

    for &(n, label) in scales {
        let mem_mb = (n * n * 8) as f64 / 1_000_000.0;
        let flops_mm = 2.0 * (n as f64).powi(3); // approximate for O(n^3) ops

        eprintln!();
        eprintln!("--- n={} (matrix {}x{}, {:.1} MB) ---", label, n, n, mem_mb);
        eprintln!("{:<12} {:>14} {:>14} {:>10} {:>14}",
                 "Operation", "tambear", "faer", "ratio", "GFLOPS(tam)");
        eprintln!("------------------------------------------------------------");

        let a = make_mat(n);
        let spd = make_spd_mat(n);

        // Build faer matrices
        let fa = faer::Mat::<f64>::from_fn(n, n, |i, j| a.get(i, j));
        let fa_spd = faer::Mat::<f64>::from_fn(n, n, |i, j| spd.get(i, j));

        // --- SVD ---
        let skip_svd = prev_svd > 0.0 && estimate_cubic(prev_svd, n) > 60.0;
        if skip_svd {
            eprintln!("{:<12} {:>14} {:>14} {:>10} {:>14}",
                     "SVD", "SKIPPED", "--", "--",
                     format!("est ~{:.0}s", estimate_cubic(prev_svd, n)));
        } else {
            let t0 = Instant::now();
            let _svd_res = svd(&a);
            let tam_time = t0.elapsed().as_secs_f64();
            prev_svd = tam_time;

            let t1 = Instant::now();
            let _faer_svd = fa.svd().expect("faer SVD failed");
            let faer_time = t1.elapsed().as_secs_f64();

            let ratio = if faer_time > 0.0 { tam_time / faer_time } else { f64::INFINITY };
            eprintln!("{:<12} {:>14} {:>14} {:>10.2}x {:>14}",
                     "SVD", fmt_time(tam_time), fmt_time(faer_time), ratio,
                     fmt_gflops(flops_mm, tam_time));
        }

        // --- QR ---
        let skip_qr = prev_qr > 0.0 && estimate_cubic(prev_qr, n) > 60.0;
        if skip_qr {
            eprintln!("{:<12} {:>14} {:>14} {:>10} {:>14}",
                     "QR", "SKIPPED", "--", "--",
                     format!("est ~{:.0}s", estimate_cubic(prev_qr, n)));
        } else {
            let t0 = Instant::now();
            let _qr_res = qr(&a);
            let tam_time = t0.elapsed().as_secs_f64();
            prev_qr = tam_time;

            let t1 = Instant::now();
            let _faer_qr = fa.qr();
            let faer_time = t1.elapsed().as_secs_f64();

            let ratio = if faer_time > 0.0 { tam_time / faer_time } else { f64::INFINITY };
            eprintln!("{:<12} {:>14} {:>14} {:>10.2}x {:>14}",
                     "QR", fmt_time(tam_time), fmt_time(faer_time), ratio,
                     fmt_gflops(flops_mm, tam_time));
        }

        // --- LU ---
        let skip_lu = prev_lu > 0.0 && estimate_cubic(prev_lu, n) > 60.0;
        if skip_lu {
            eprintln!("{:<12} {:>14} {:>14} {:>10} {:>14}",
                     "LU", "SKIPPED", "--", "--",
                     format!("est ~{:.0}s", estimate_cubic(prev_lu, n)));
        } else {
            let t0 = Instant::now();
            let _lu_res = lu(&a);
            let tam_time = t0.elapsed().as_secs_f64();
            prev_lu = tam_time;

            let t1 = Instant::now();
            let _faer_lu = fa.partial_piv_lu();
            let faer_time = t1.elapsed().as_secs_f64();

            let ratio = if faer_time > 0.0 { tam_time / faer_time } else { f64::INFINITY };
            eprintln!("{:<12} {:>14} {:>14} {:>10.2}x {:>14}",
                     "LU", fmt_time(tam_time), fmt_time(faer_time), ratio,
                     fmt_gflops(flops_mm, tam_time));
        }

        // --- Cholesky (SPD matrix) ---
        let skip_chol = prev_chol > 0.0 && estimate_cubic(prev_chol, n) > 60.0;
        if skip_chol {
            eprintln!("{:<12} {:>14} {:>14} {:>10} {:>14}",
                     "Cholesky", "SKIPPED", "--", "--",
                     format!("est ~{:.0}s", estimate_cubic(prev_chol, n)));
        } else {
            let t0 = Instant::now();
            let chol_res = cholesky(&spd);
            let tam_time = t0.elapsed().as_secs_f64();
            prev_chol = tam_time;

            let t1 = Instant::now();
            let _faer_chol = fa_spd.llt(faer::Side::Lower);
            let faer_time = t1.elapsed().as_secs_f64();

            let ratio = if faer_time > 0.0 { tam_time / faer_time } else { f64::INFINITY };
            let status = if chol_res.is_some() { "OK" } else { "FAIL" };
            eprintln!("{:<12} {:>14} {:>14} {:>10.2}x {:>14}",
                     format!("Cholesky({})", status), fmt_time(tam_time), fmt_time(faer_time), ratio,
                     fmt_gflops(flops_mm / 3.0, tam_time)); // Cholesky is ~n^3/3
        }

        // --- Symmetric Eigendecomposition ---
        let skip_eigen = prev_eigen > 0.0 && estimate_cubic(prev_eigen, n) > 60.0;
        if skip_eigen {
            eprintln!("{:<12} {:>14} {:>14} {:>10} {:>14}",
                     "SymEigen", "SKIPPED", "--", "--",
                     format!("est ~{:.0}s", estimate_cubic(prev_eigen, n)));
        } else {
            // sym_eigen needs symmetric matrix, use spd
            let t0 = Instant::now();
            let (_eigenvalues, _eigenvectors) = sym_eigen(&spd);
            let tam_time = t0.elapsed().as_secs_f64();
            prev_eigen = tam_time;

            // faer symmetric eigen
            let t1 = Instant::now();
            let _faer_eigen = fa_spd.self_adjoint_eigen(faer::Side::Lower);
            let faer_time = t1.elapsed().as_secs_f64();

            let ratio = if faer_time > 0.0 { tam_time / faer_time } else { f64::INFINITY };
            eprintln!("{:<12} {:>14} {:>14} {:>10.2}x {:>14}",
                     "SymEigen", fmt_time(tam_time), fmt_time(faer_time), ratio,
                     fmt_gflops(flops_mm, tam_time));
        }

        // --- Solve Ax=b ---
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7 + 0.3).cos()).collect();
        let t0 = Instant::now();
        let x = solve(&a, &b);
        let solve_time = t0.elapsed().as_secs_f64();
        let status = if x.is_some() { "OK" } else { "FAIL" };
        eprintln!("{:<12} {:>14} {:>14} {:>10} {:>14}",
                 format!("Solve({})", status), fmt_time(solve_time), "--", "--",
                 fmt_gflops(flops_mm, solve_time));
    }

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("NOTE: tambear is pure Rust from scratch. faer uses optimized BLAS-like");
    eprintln!("      kernels with vectorization. Ratio > 1 means tambear is slower.");
    eprintln!("========================================================================");
    eprintln!();
}

/// Estimate time for the next scale based on O(n^3) growth.
/// prev_time is for the previous n; current n is implied by context.
/// We assume the caller passes the new n and the ratio is (new_n / prev_n)^3.
/// Since scales double or more, we use n directly.
fn estimate_cubic(prev_time: f64, _n: usize) -> f64 {
    // Crude: each step roughly doubles, so 8x growth.
    // But our scale ladder is irregular. Just use 8x as heuristic.
    prev_time * 8.0
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Graph Algorithms — Scale Ladder
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_graph_scale_ladder() {
    let scales: &[(usize, &str)] = &[
        (100, "100"),
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
    ];

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("GRAPH ALGORITHMS -- Scale Ladder (sparse, ~5 edges/node)");
    eprintln!("========================================================================");
    eprintln!("{:<10} {:>12} {:>12} {:>12} {:>12} {:>12}",
             "nodes", "dijkstra", "components", "pagerank", "kruskal", "bellman_f");
    eprintln!("------------------------------------------------------------------------");

    let mut prev_dijkstra = 0.0_f64;
    let mut prev_components = 0.0_f64;
    let mut prev_pagerank = 0.0_f64;
    let mut prev_kruskal = 0.0_f64;
    let mut prev_bellman = 0.0_f64;

    for &(n, label) in scales {
        let directed = make_sparse_graph(n);
        let undirected = make_undirected_sparse_graph(n);
        let n_edges = directed.n_edges();

        eprint!("{:<10}", label);

        // --- Dijkstra (O((V+E) log V)) ---
        let skip_d = prev_dijkstra > 0.0 && prev_dijkstra * 12.0 > 60.0;
        if skip_d {
            eprint!(" {:>12}", "SKIP");
        } else {
            let t0 = Instant::now();
            let (_dist, _parent) = graph::dijkstra(&directed, 0);
            let elapsed = t0.elapsed().as_secs_f64();
            prev_dijkstra = elapsed;
            eprint!(" {:>12}", fmt_time(elapsed));
        }

        // --- Connected Components (O(V+E)) ---
        let skip_c = prev_components > 0.0 && prev_components * 12.0 > 60.0;
        if skip_c {
            eprint!(" {:>12}", "SKIP");
        } else {
            let t0 = Instant::now();
            let _comps = graph::connected_components(&directed);
            let elapsed = t0.elapsed().as_secs_f64();
            prev_components = elapsed;
            eprint!(" {:>12}", fmt_time(elapsed));
        }

        // --- PageRank (iterative, O(iter * E)) ---
        let skip_p = prev_pagerank > 0.0 && prev_pagerank * 12.0 > 60.0;
        if skip_p {
            eprint!(" {:>12}", "SKIP");
        } else {
            let t0 = Instant::now();
            let _ranks = graph::pagerank(&directed, 0.85, 100, 1e-6);
            let elapsed = t0.elapsed().as_secs_f64();
            prev_pagerank = elapsed;
            eprint!(" {:>12}", fmt_time(elapsed));
        }

        // --- Kruskal MST (O(E log E)) --- uses undirected graph
        let skip_k = prev_kruskal > 0.0 && prev_kruskal * 12.0 > 60.0;
        if skip_k {
            eprint!(" {:>12}", "SKIP");
        } else {
            let t0 = Instant::now();
            let _mst = graph::kruskal(&undirected);
            let elapsed = t0.elapsed().as_secs_f64();
            prev_kruskal = elapsed;
            eprint!(" {:>12}", fmt_time(elapsed));
        }

        // --- Bellman-Ford (O(V*E)) --- only run up to 10K
        if n <= 10_000 {
            let skip_b = prev_bellman > 0.0 && prev_bellman * 100.0 > 60.0; // O(V*E) grows fast
            if skip_b {
                eprint!(" {:>12}", "SKIP");
            } else {
                let t0 = Instant::now();
                let _res = graph::bellman_ford(&directed, 0);
                let elapsed = t0.elapsed().as_secs_f64();
                prev_bellman = elapsed;
                eprint!(" {:>12}", fmt_time(elapsed));
            }
        } else {
            eprint!(" {:>12}", "N/A(V*E)");
        }

        eprintln!();
        eprintln!("    edges={}", fmt_num(n_edges));
    }

    eprintln!("========================================================================");
    eprintln!();
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Matrix Multiply — Scale Ladder (tambear vs faer)
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_mat_mul_scale_ladder() {
    let scales: &[(usize, &str)] = &[
        (10, "10"),
        (50, "50"),
        (100, "100"),
        (200, "200"),
        (500, "500"),
        (1000, "1000"),
        (2000, "2000"),
    ];

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("MATRIX MULTIPLY -- Scale Ladder (tambear vs faer)");
    eprintln!("========================================================================");
    eprintln!("{:<8} {:>14} {:>14} {:>10} {:>14} {:>14} {:>10}",
             "n", "tambear", "faer", "ratio", "tam GFLOPS", "faer GFLOPS", "MB");
    eprintln!("------------------------------------------------------------------------");

    let mut prev_tam = 0.0_f64;

    for &(n, label) in scales {
        let flops = 2.0 * (n as f64).powi(3);
        let mem_mb = 3.0 * (n * n * 8) as f64 / 1_000_000.0; // A + B + C

        // Skip if estimated > 60s
        if prev_tam > 0.0 {
            // mat_mul is O(n^3); scale ratio depends on step
            let estimated = prev_tam * 8.0; // rough doubling estimate
            if estimated > 60.0 {
                eprintln!("{:<8} {:>14} {:>14} {:>10} {:>14} {:>14} {:>10.1}",
                         label, "SKIPPED", "--", "--", "--", "--", mem_mb);
                eprintln!("    estimated: ~{:.0}s", estimated);
                continue;
            }
        }

        let a = make_mat(n);
        let b = make_mat(n);

        // tambear mat_mul
        let t0 = Instant::now();
        let _c = mat_mul(&a, &b);
        let tam_time = t0.elapsed().as_secs_f64();
        prev_tam = tam_time;

        // faer mat_mul
        let fa = faer::Mat::<f64>::from_fn(n, n, |i, j| a.get(i, j));
        let fb = faer::Mat::<f64>::from_fn(n, n, |i, j| b.get(i, j));

        let t1 = Instant::now();
        let _fc = &fa * &fb;
        let faer_time = t1.elapsed().as_secs_f64();

        let ratio = if faer_time > 0.0 { tam_time / faer_time } else { f64::INFINITY };

        eprintln!("{:<8} {:>14} {:>14} {:>10.2}x {:>14} {:>14} {:>10.1}",
                 label,
                 fmt_time(tam_time),
                 fmt_time(faer_time),
                 ratio,
                 fmt_gflops(flops, tam_time),
                 fmt_gflops(flops, faer_time),
                 mem_mb);
    }

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("NOTE: faer uses AVX2/AVX-512 vectorized BLAS kernels.");
    eprintln!("      tambear is triple-loop from scratch. Expected ratio: 10-100x.");
    eprintln!("      This measures the gap that GPU offload needs to close.");
    eprintln!("========================================================================");
    eprintln!();
}

// ═══════════════════════════════════════════════════════════════════════
// 7. KNN from Distance Matrix — Scale Ladder
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_knn_scale_ladder() {
    use tambear::intermediates::{DistanceMatrix, Metric};

    let scales: &[(usize, &str)] = &[
        (100, "100"),
        (500, "500"),
        (1_000, "1K"),
        (2_000, "2K"),
        (5_000, "5K"),
    ];

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("KNN FROM DISTANCE MATRIX -- Scale Ladder");
    eprintln!("========================================================================");
    eprintln!("{:<8} {:>10} {:>14} {:>14} {:>14} {:>10}",
             "n", "k", "dist_build", "knn_time", "throughput", "MB");
    eprintln!("------------------------------------------------------------------------");

    let mut prev_knn = 0.0_f64;

    for &(n, label) in scales {
        let k = 10.min(n - 1);
        let mem_mb = (n * n * 8) as f64 / 1_000_000.0;

        // KNN is O(n^2 * k), distance matrix build is O(n^2)
        if prev_knn > 0.0 {
            let estimated = prev_knn * 4.0; // rough quadratic
            if estimated > 60.0 {
                eprintln!("{:<8} {:>10} {:>14} {:>14} {:>14} {:>10.1}",
                         label, k, "SKIPPED", "SKIPPED", "--", mem_mb);
                continue;
            }
        }

        // Build a deterministic distance matrix
        let t0 = Instant::now();
        let mut dist_data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    dist_data.push(0.0);
                } else {
                    let d = (i as f64 - j as f64).powi(2) * 0.001
                           + ((i * 7 + j * 13) as f64 * 0.0001).sin().abs();
                    dist_data.push(d);
                }
            }
        }
        let dist = DistanceMatrix::from_vec(Metric::L2Sq, n, dist_data);
        let build_time = t0.elapsed().as_secs_f64();

        // KNN
        let t1 = Instant::now();
        let result = tambear::knn::knn_from_distance(&dist, k);
        let knn_time = t1.elapsed().as_secs_f64();
        prev_knn = knn_time;

        let throughput = fmt_throughput(n, knn_time);

        eprintln!("{:<8} {:>10} {:>14} {:>14} {:>14} {:>10.1}",
                 label, k, fmt_time(build_time), fmt_time(knn_time), throughput, mem_mb);

        // Sanity check
        assert_eq!(result.n, n);
        assert_eq!(result.k, k);
    }

    eprintln!("========================================================================");
    eprintln!();
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Geometric Mean — Scale Ladder
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_geometric_mean_scale_ladder() {
    let scales: &[(usize, &str)] = &[
        (10, "10"),
        (1_000, "1K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
        (10_000_000, "10M"),
        (100_000_000, "100M"),
    ];

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("GEOMETRIC MEAN -- Scale Ladder");
    eprintln!("========================================================================");
    eprintln!("{:<14} {:>12} {:>16} {:>12}",
             "n", "time", "throughput", "memory(MB)");
    eprintln!("------------------------------------------------------------------------");

    for &(n, label) in scales {
        let mem_mb = (n * 8) as f64 / 1_000_000.0;

        // Geometric mean needs positive values
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 1.23456789 + 0.1).sin().abs() + 0.001).collect();

        let t0 = Instant::now();
        let gm = geometric_mean(&data);
        let elapsed = t0.elapsed().as_secs_f64();

        eprintln!("{:<14} {:>12} {:>16} {:>12.1}",
                 label, fmt_time(elapsed), fmt_throughput(n, elapsed), mem_mb);
        eprintln!("    geometric_mean = {:.15}", gm);
    }

    eprintln!("========================================================================");
    eprintln!();
}

// ═══════════════════════════════════════════════════════════════════════
// 9. Quantile Method Comparison
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_quantile_methods_comparison() {
    let n = 10_000_000;
    let mut data = make_f64_vec(n);
    data.sort_by(|a, b| a.total_cmp(b));

    let methods = [
        (QuantileMethod::InverseCdf, "InverseCdf"),
        (QuantileMethod::Linear4, "Linear4"),
        (QuantileMethod::Hazen, "Hazen"),
        (QuantileMethod::Weibull, "Weibull"),
        (QuantileMethod::Linear, "Linear"),
        (QuantileMethod::MedianUnbiased, "MedianUnbiased"),
        (QuantileMethod::NormalUnbiased, "NormalUnbiased"),
    ];

    let quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("QUANTILE METHOD COMPARISON -- n={}", fmt_num(n));
    eprintln!("========================================================================");
    eprintln!("{:<18} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
             "Method", "P01", "P10", "P25", "P50", "P75", "P90", "P99");
    eprintln!("------------------------------------------------------------------------");

    for (method, name) in &methods {
        eprint!("{:<18}", name);
        for &q in &quantiles {
            let v = quantile(&data, q, *method);
            eprint!(" {:>10.6}", v);
        }
        eprintln!();
    }

    // Time each method
    eprintln!();
    eprintln!("{:<18} {:>12}", "Method", "time(P50)");
    eprintln!("------------------------------");
    for (method, name) in &methods {
        let t0 = Instant::now();
        for _ in 0..1000 {
            let _ = quantile(&data, 0.5, *method);
        }
        let elapsed = t0.elapsed().as_secs_f64() / 1000.0;
        eprintln!("{:<18} {:>12}", name, fmt_time(elapsed));
    }

    eprintln!("========================================================================");
    eprintln!();
}

// ═══════════════════════════════════════════════════════════════════════
// 10. Summary — Run All and Produce Final Report
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_summary_report() {
    eprintln!();
    eprintln!("########################################################################");
    eprintln!("#                     SCALE LADDER SUMMARY REPORT                      #");
    eprintln!("########################################################################");
    eprintln!();
    eprintln!("Run each benchmark individually for detailed results:");
    eprintln!("  cargo test --test scale_ladder --release -- --ignored --nocapture bench_descriptive_stats_scale_ladder");
    eprintln!("  cargo test --test scale_ladder --release -- --ignored --nocapture bench_descriptive_stats_1b");
    eprintln!("  cargo test --test scale_ladder --release -- --ignored --nocapture bench_sort_quantile_scale_ladder");
    eprintln!("  cargo test --test scale_ladder --release -- --ignored --nocapture bench_linear_algebra_scale_ladder");
    eprintln!("  cargo test --test scale_ladder --release -- --ignored --nocapture bench_graph_scale_ladder");
    eprintln!("  cargo test --test scale_ladder --release -- --ignored --nocapture bench_mat_mul_scale_ladder");
    eprintln!("  cargo test --test scale_ladder --release -- --ignored --nocapture bench_knn_scale_ladder");
    eprintln!("  cargo test --test scale_ladder --release -- --ignored --nocapture bench_geometric_mean_scale_ladder");
    eprintln!("  cargo test --test scale_ladder --release -- --ignored --nocapture bench_quantile_methods_comparison");
    eprintln!();
    eprintln!("Or run ALL benchmarks:");
    eprintln!("  cargo test --test scale_ladder --release -- --ignored --nocapture");
    eprintln!();

    // Quick smoke test: smallest scale for each domain
    eprintln!("--- Quick smoke test (smallest scale per domain) ---");
    eprintln!();

    // Descriptive
    let data = make_f64_vec(1000);
    let t0 = Instant::now();
    let stats = moments_ungrouped(&data);
    let t_desc = t0.elapsed().as_secs_f64();
    eprintln!("Descriptive (n=1K):  {} (mean={:.6})", fmt_time(t_desc), stats.mean());

    // Sort + quantile
    let mut sorted = data.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let t0 = Instant::now();
    let med = median(&sorted);
    let t_med = t0.elapsed().as_secs_f64();
    eprintln!("Median (n=1K):       {} (median={:.6})", fmt_time(t_med), med);

    // Linear algebra
    let a = make_mat(50);
    let t0 = Instant::now();
    let _s = svd(&a);
    let t_svd = t0.elapsed().as_secs_f64();
    eprintln!("SVD (50x50):         {}", fmt_time(t_svd));

    let t0 = Instant::now();
    let _q = qr(&a);
    let t_qr = t0.elapsed().as_secs_f64();
    eprintln!("QR (50x50):          {}", fmt_time(t_qr));

    let t0 = Instant::now();
    let _l = lu(&a);
    let t_lu = t0.elapsed().as_secs_f64();
    eprintln!("LU (50x50):          {}", fmt_time(t_lu));

    // Mat mul
    let b = make_mat(50);
    let t0 = Instant::now();
    let _c = mat_mul(&a, &b);
    let t_mm = t0.elapsed().as_secs_f64();
    eprintln!("MatMul (50x50):      {}", fmt_time(t_mm));

    // Graph
    let g = make_sparse_graph(1000);
    let t0 = Instant::now();
    let _d = graph::dijkstra(&g, 0);
    let t_dijk = t0.elapsed().as_secs_f64();
    eprintln!("Dijkstra (1K nodes): {}", fmt_time(t_dijk));

    let t0 = Instant::now();
    let _pr = graph::pagerank(&g, 0.85, 100, 1e-6);
    let t_pr = t0.elapsed().as_secs_f64();
    eprintln!("PageRank (1K nodes): {}", fmt_time(t_pr));

    eprintln!();
    eprintln!("All smoke tests passed.");
    eprintln!("########################################################################");
    eprintln!();
}

// ═══════════════════════════════════════════════════════════════════════
// F20: DBSCAN at scale + session sharing (Paper 3 evidence)
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_dbscan_scale_ladder() {
    use tambear::clustering::ClusteringEngine;

    let scales: &[(usize, &str)] = &[
        (50, "50"),
        (100, "100"),
        (500, "500"),
        (1_000, "1K"),
        (2_000, "2K"),
        (5_000, "5K"),
    ];

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("DBSCAN -- Scale Ladder (eps=5.0, min_pts=3, 2D points)");
    eprintln!("========================================================================");
    eprintln!("{:<8} {:>14} {:>10} {:>10} {:>10}",
             "n", "time", "clusters", "noise", "MB");
    eprintln!("------------------------------------------------------------------------");

    let mut engine = ClusteringEngine::new().unwrap();
    let mut prev_time = 0.0f64;

    for &(n, label) in scales {
        let mem_mb = (n * n * 8) as f64 / 1_000_000.0;

        // Skip if previous time suggests this will be too slow
        if prev_time > 0.0 {
            let estimated = prev_time * 4.0; // quadratic
            if estimated > 120.0 {
                eprintln!("{:<8} {:>14} {:>10} {:>10} {:>10.1}",
                         label, "SKIPPED", "--", "--", mem_mb);
                continue;
            }
        }

        // Generate 2D clustered data: 5 Gaussian clusters
        let mut data = Vec::with_capacity(n * 2);
        let centers = [(10.0, 10.0), (30.0, 30.0), (50.0, 10.0), (10.0, 50.0), (50.0, 50.0)];
        for i in 0..n {
            let c = &centers[i % 5];
            let noise_x = ((i * 73 + 11) as f64 * 0.001).sin() * 3.0;
            let noise_y = ((i * 37 + 7) as f64 * 0.001).cos() * 3.0;
            data.push(c.0 + noise_x);
            data.push(c.1 + noise_y);
        }

        let t0 = Instant::now();
        let result = engine.dbscan(&data, n, 2, 5.0, 3);
        let time = t0.elapsed().as_secs_f64();
        prev_time = time;

        match result {
            Ok(r) => {
                let noise = r.labels.iter().filter(|&&l| l == -1).count();
                eprintln!("{:<8} {:>14} {:>10} {:>10} {:>10.1}",
                         label, fmt_time(time), r.n_clusters, noise, mem_mb);
            }
            Err(e) => {
                eprintln!("{:<8} {:>14} {:>10} {:>10} {:>10.1}",
                         label, fmt_time(time), "ERR", &format!("{}", e), mem_mb);
            }
        }
    }

    eprintln!("------------------------------------------------------------------------");
    eprintln!("O(n²d) distance matrix dominates. Ceiling: ~10K pts on 32GB (800MB dist).");
    eprintln!("sklearn: same O(n²) but Python overhead → 5-50x slower.");
    eprintln!("========================================================================");
}

// ═══════════════════════════════════════════════════════════════════════
// F20: DBSCAN → KNN Session Sharing Benchmark (Paper 3 headline)
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_session_sharing_dbscan_to_knn() {
    use tambear::clustering::ClusteringEngine;
    use tambear::knn::knn_session;
    use tambear::intermediates::TamSession;

    let scales: &[(usize, &str)] = &[
        (100, "100"),
        (500, "500"),
        (1_000, "1K"),
        (2_000, "2K"),
    ];

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("SESSION SHARING: DBSCAN → KNN (Paper 3 headline claim)");
    eprintln!("========================================================================");
    eprintln!("{:<8} {:>12} {:>12} {:>12} {:>8} {:>8}",
             "n", "DBSCAN(ms)", "KNN_new(ms)", "KNN_shared", "speedup", "reuse?");
    eprintln!("------------------------------------------------------------------------");

    let mut engine = ClusteringEngine::new().unwrap();

    for &(n, label) in scales {
        // Generate clustered 2D data
        let mut data = Vec::with_capacity(n * 2);
        let centers = [(10.0, 10.0), (50.0, 50.0), (30.0, 70.0)];
        for i in 0..n {
            let c = &centers[i % 3];
            let noise_x = ((i * 73 + 11) as f64 * 0.001).sin() * 3.0;
            let noise_y = ((i * 37 + 7) as f64 * 0.001).cos() * 3.0;
            data.push(c.0 + noise_x);
            data.push(c.1 + noise_y);
        }
        let d = 2;
        let k = 5.min(n - 1);

        // Path A: KNN standalone (computes distance matrix from scratch)
        let mut session_standalone = TamSession::new();
        let t0 = Instant::now();
        let _ = knn_session(&mut session_standalone, &data, n, d, k);
        let knn_new_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Path B: DBSCAN first, then KNN reuses the distance matrix
        let mut session_shared = TamSession::new();

        let t0 = Instant::now();
        let _ = engine.dbscan_session(&mut session_shared, &data, n, d, 5.0, 3);
        let dbscan_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let session_after_dbscan = session_shared.len();

        let t0 = Instant::now();
        let _ = knn_session(&mut session_shared, &data, n, d, k);
        let knn_shared_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let session_after_knn = session_shared.len();
        let reused = session_after_knn == session_after_dbscan;
        let speedup = if knn_shared_ms > 0.001 { knn_new_ms / knn_shared_ms } else { f64::NAN };

        eprintln!("{:<8} {:>12.1} {:>12.1} {:>12.1} {:>7.1}x {:>8}",
                 label, dbscan_ms, knn_new_ms, knn_shared_ms, speedup,
                 if reused { "YES" } else { "NO" });
    }

    eprintln!("------------------------------------------------------------------------");
    eprintln!("KNN_shared skips O(n²d) GPU distance recompute → only O(n²k) CPU sort.");
    eprintln!("This is the headline claim of Paper 3: zero-GPU-cost sharing.");
    eprintln!("========================================================================");
}

// ═══════════════════════════════════════════════════════════════════════
// F02: SVD accuracy vs condition number ladder
// ═══════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn bench_svd_accuracy_vs_condition() {
    let kappas: &[f64] = &[1.0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e15];

    eprintln!();
    eprintln!("========================================================================");
    eprintln!("SVD ACCURACY vs CONDITION NUMBER (5×5 diagonal, one-sided Jacobi)");
    eprintln!("========================================================================");
    eprintln!("{:>10} {:>14} {:>14} {:>14} {:>10}",
             "kappa", "max_sig_err", "recon_fro", "pinv_resid", "verdict");
    eprintln!("------------------------------------------------------------------------");

    for &kappa in kappas {
        let n = 5;
        let sigmas: Vec<f64> = (0..n).map(|i| {
            kappa.powf(-(i as f64) / (n as f64 - 1.0))
        }).collect();

        let mut data = vec![0.0; n * n];
        for i in 0..n { data[i * n + i] = sigmas[i]; }

        let a = Mat::from_vec(n, n, data);
        let r = svd(&a);

        // Singular value accuracy
        let max_sig_err: f64 = r.sigma.iter().zip(sigmas.iter())
            .map(|(got, exp)| {
                if exp.abs() > 1e-15 { (got - exp).abs() / exp.abs() } else { got.abs() }
            })
            .fold(0.0f64, f64::max);

        // Reconstruction: ||A - USVt||_F
        let mut recon_err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for l in 0..r.sigma.len() {
                    s += r.u.get(i, l) * r.sigma[l] * r.vt.get(l, j);
                }
                let expected = if i == j { sigmas[i] } else { 0.0 };
                recon_err += (s - expected).powi(2);
            }
        }
        recon_err = recon_err.sqrt();

        // Pseudoinverse residual: ||A A+ A - A||_F
        let a_pinv = pinv(&a, None);
        let mut pinv_resid = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut aaa = 0.0;
                for k in 0..n {
                    let mut aa_plus = 0.0;
                    for l in 0..n {
                        aa_plus += a.get(i, l) * a_pinv.get(l, k);
                    }
                    aaa += aa_plus * a.get(k, j);
                }
                pinv_resid += (aaa - a.get(i, j)).powi(2);
            }
        }
        pinv_resid = pinv_resid.sqrt();

        let verdict = if max_sig_err < 1e-14 { "EXACT" }
            else if max_sig_err < 1e-10 { "EXCELLENT" }
            else if max_sig_err < 1e-6 { "GOOD" }
            else if max_sig_err < 0.01 { "FAIR" }
            else { "DEGRADED" };

        eprintln!("{:>10.0e} {:>14.2e} {:>14.2e} {:>14.2e} {:>10}",
                 kappa, max_sig_err, recon_err, pinv_resid, verdict);
    }

    eprintln!("------------------------------------------------------------------------");
    eprintln!("One-sided Jacobi avoids κ² amplification of A^T A approach.");
    eprintln!("Expected: EXACT through κ=1e12, degrades gracefully at κ=1e15.");
    eprintln!("========================================================================");
}
