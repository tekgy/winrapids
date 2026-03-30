//! Observer benchmark: winrapids-store provenance lookup latency.
//!
//! Measures the critical path: lookup() on a populated store.
//! Phase 2 Python baseline: 1.0 us per hit.
//! Phase 3 target: < 0.5 us per hit.
//!
//! Standing methodology: 3 warmup, 20 timed runs, report p50/p99/mean.

use std::time::Instant;

use winrapids_store::*;

fn main() {
    println!("{}", "=".repeat(70));
    println!("Observer Benchmark: winrapids-store provenance lookup latency");
    println!("{}", "=".repeat(70));

    bench_provenance_hash();
    bench_lookup_single();
    bench_lookup_populated();
    bench_register();
    bench_farm_simulation();

    println!("\n{}", "=".repeat(70));
    println!("Benchmark complete.");
    println!("{}", "=".repeat(70));
}

/// Benchmark BLAKE3 provenance hash computation.
fn bench_provenance_hash() {
    println!("\n--- BLAKE3 Provenance Hash ---\n");

    let short_id = "price:AAPL:2026-03-30:1s";
    let long_id = "price:AAPL:2026-03-30:1s:rolling_mean:w=60:cadence=1s:kingdom=K02";

    // Short identity
    let times = bench_ns(|| {
        std::hint::black_box(data_provenance(std::hint::black_box(short_id)));
    }, 3, 100_000);
    println!("  data_provenance (short): p50={:.0} ns  p99={:.0} ns  mean={:.0} ns",
        times.p50, times.p99, times.mean);

    // Long identity
    let times = bench_ns(|| {
        std::hint::black_box(data_provenance(std::hint::black_box(long_id)));
    }, 3, 100_000);
    println!("  data_provenance (long):  p50={:.0} ns  p99={:.0} ns  mean={:.0} ns",
        times.p50, times.p99, times.mean);

    // Provenance hash with 1 input
    let input = data_provenance(short_id);
    let times = bench_ns(|| {
        std::hint::black_box(provenance_hash(
            std::hint::black_box(&[input]),
            std::hint::black_box("scan:add:w=20"),
        ));
    }, 3, 100_000);
    println!("  provenance_hash (1 in): p50={:.0} ns  p99={:.0} ns  mean={:.0} ns",
        times.p50, times.p99, times.mean);

    // Provenance hash with 3 inputs
    let inputs = [input, input, input];
    let times = bench_ns(|| {
        std::hint::black_box(provenance_hash(
            std::hint::black_box(&inputs),
            std::hint::black_box("fused_expr:zscore"),
        ));
    }, 3, 100_000);
    println!("  provenance_hash (3 in): p50={:.0} ns  p99={:.0} ns  mean={:.0} ns",
        times.p50, times.p99, times.mean);
}

/// Benchmark single lookup on a small store.
fn bench_lookup_single() {
    println!("\n--- Lookup: Single Entry Store ---\n");

    let mut store = GpuStore::new(100_000_000);
    let prov = data_provenance("price:AAPL:2026-03-30:1s");
    let ptr = BufferPtr { device_ptr: 0x7F0000001000, byte_size: 400_000 };
    let header = BufferHeader::new(prov, 100.0, DType::F32, 100_000);
    store.register(header, ptr);

    // Hit
    let times = bench_ns(|| {
        std::hint::black_box(store.lookup(std::hint::black_box(&prov)));
    }, 3, 100_000);
    println!("  lookup (hit, 1 entry):  p50={:.0} ns  p99={:.0} ns  mean={:.0} ns",
        times.p50, times.p99, times.mean);

    // Miss
    let miss_prov = data_provenance("price:MSFT:2026-03-30:1s");
    let times = bench_ns(|| {
        std::hint::black_box(store.lookup(std::hint::black_box(&miss_prov)));
    }, 3, 100_000);
    println!("  lookup (miss, 1 entry): p50={:.0} ns  p99={:.0} ns  mean={:.0} ns",
        times.p50, times.p99, times.mean);
}

/// Benchmark lookup on a realistic-sized store.
fn bench_lookup_populated() {
    println!("\n--- Lookup: Populated Store (farm-realistic) ---\n");

    // Populate with realistic farm sizes
    for &n_entries in &[100, 1_000, 5_000, 10_000, 50_000] {
        let mut store = GpuStore::new(n_entries as u64 * 400_000);

        // Register n_entries buffers
        let mut provenances = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            let prov = data_provenance(&format!("ticker{}:cadence{}:leaf{}",
                i / 50, (i / 10) % 5, i % 10));
            let ptr = BufferPtr { device_ptr: 0x7F0000001000 + (i as u64 * 0x1000), byte_size: 400_000 };
            let header = BufferHeader::new(prov, 100.0, DType::F32, 100_000);
            store.register(header, ptr);
            provenances.push(prov);
        }

        // Benchmark lookups hitting random entries
        let mut idx = 0usize;
        let times = bench_ns(|| {
            let prov = &provenances[idx % provenances.len()];
            std::hint::black_box(store.lookup(std::hint::black_box(prov)));
            idx += 1;
        }, 3, 100_000);

        println!("  lookup (hit, {} entries): p50={:.0} ns  p99={:.0} ns  mean={:.0} ns",
            n_entries, times.p50, times.p99, times.mean);
    }
}

/// Benchmark register (cache miss path).
fn bench_register() {
    println!("\n--- Register (Miss Path) ---\n");

    // Register to a store with space (no eviction)
    let mut store = GpuStore::new(1_000_000_000_000); // huge budget
    let mut i = 0u64;
    let times = bench_ns(|| {
        let prov = data_provenance(&format!("bench:register:{}", i));
        let ptr = BufferPtr { device_ptr: 0x1000 + i * 0x1000, byte_size: 400 };
        let header = BufferHeader::new(prov, 100.0, DType::F32, 100);
        std::hint::black_box(store.register(header, ptr));
        i += 1;
    }, 3, 100_000);
    println!("  register (no eviction): p50={:.0} ns  p99={:.0} ns  mean={:.0} ns",
        times.p50, times.p99, times.mean);
    println!("  (includes BLAKE3 hash + HashMap insert + LRU push + header creation)");
}

/// Simulate a farm cycle: 100 tickers x 5 cadences x 10 leaves.
fn bench_farm_simulation() {
    println!("\n--- Farm Simulation (100 tickers x 5 cadences x 10 leaves) ---\n");

    let n_tickers = 100;
    let n_cadences = 5;
    let n_leaves = 10;
    let total = n_tickers * n_cadences * n_leaves;

    // Populate store
    let mut store = GpuStore::new(total as u64 * 400_000);
    let mut all_provenances = Vec::with_capacity(total);

    for t in 0..n_tickers {
        let data_prov = data_provenance(&format!("ticker{}:v1", t));
        for c in 0..n_cadences {
            for l in 0..n_leaves {
                let comp_id = format!("leaf{}:cadence{}", l, c);
                let prov = provenance_hash(&[data_prov], &comp_id);
                let ptr = BufferPtr {
                    device_ptr: 0x7F0000001000 + (all_provenances.len() as u64 * 0x1000),
                    byte_size: 400_000,
                };
                let header = BufferHeader::new(prov, 100.0, DType::F32, 100_000);
                store.register(header, ptr);
                all_provenances.push((t, prov));
            }
        }
    }

    // Simulate farm cycles at different dirty ratios
    for &dirty_pct in &[100, 50, 20, 10, 5, 1] {
        let n_dirty = (n_tickers * dirty_pct / 100).max(1);

        // Pre-compute new provenances for dirty tickers
        let mut cycle_provenances = Vec::with_capacity(total);
        for t in 0..n_tickers {
            let version = if t < n_dirty { "v2" } else { "v1" };
            let data_prov = data_provenance(&format!("ticker{}:{}", t, version));
            for c in 0..n_cadences {
                for l in 0..n_leaves {
                    let comp_id = format!("leaf{}:cadence{}", l, c);
                    let prov = provenance_hash(&[data_prov], &comp_id);
                    cycle_provenances.push(prov);
                }
            }
        }

        // Benchmark the lookup-only portion of a farm cycle
        let start = Instant::now();
        let mut hits = 0u32;
        let mut misses = 0u32;
        for prov in &cycle_provenances {
            if store.lookup(prov).is_some() {
                hits += 1;
            } else {
                misses += 1;
            }
        }
        let elapsed_us = start.elapsed().as_nanos() as f64 / 1000.0;

        let hit_rate = hits as f64 / (hits + misses) as f64 * 100.0;
        let per_lookup_ns = start.elapsed().as_nanos() as f64 / total as f64;

        println!("  {:>3}% dirty: {:.1} us total, {:.0} ns/lookup, hits={}, misses={}, hit_rate={:.1}%",
            dirty_pct, elapsed_us, per_lookup_ns, hits, misses, hit_rate);
    }

    // Compare with Python baseline
    println!("\n  Python E07 baseline: 1000 ns/lookup, 9300 us at 1% dirty");
    println!("  Target: < 500 ns/lookup, < 5000 us at 1% dirty");
}

// ────────────────────────────────────────────────────────────
// Benchmark harness
// ────────────────────────────────────────────────────────────

struct BenchResult {
    p50: f64,
    p99: f64,
    mean: f64,
}

/// Run a function `iterations` times per run, for `warmup + 20` runs.
/// Returns per-iteration nanoseconds.
fn bench_ns<F: FnMut()>(mut f: F, warmup: usize, iterations: usize) -> BenchResult {
    // Warmup
    for _ in 0..warmup {
        for _ in 0..iterations {
            f();
        }
    }

    // Timed runs
    let n_runs = 20;
    let mut run_times = Vec::with_capacity(n_runs);

    for _ in 0..n_runs {
        let start = Instant::now();
        for _ in 0..iterations {
            f();
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        run_times.push(elapsed_ns / iterations as f64);
    }

    run_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    BenchResult {
        p50: run_times[run_times.len() / 2],
        p99: run_times[(run_times.len() as f64 * 0.99) as usize],
        mean: run_times.iter().sum::<f64>() / run_times.len() as f64,
    }
}
