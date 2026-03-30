//! Observer benchmark: store hit vs miss path timing.
//!
//! Simulates a FinTek-scale farm cycle:
//! - 100 tickers × 5 cadences × 10 leaves = 5,000 computations
//! - Each computation has a provenance hash
//! - First cycle: all misses (populate cache)
//! - Second cycle with X% dirty: compute dirty, lookup clean
//!
//! Measures:
//! 1. Miss path: register() cost (store metadata + eviction check)
//! 2. Hit path: lookup() cost (HashMap probe + LRU touch)
//! 3. Farm cycle time at various dirty ratios
//! 4. Savings ratio vs E07's 865x claim
//!
//! Note: This measures the STORE overhead, not the GPU compute.
//! The 865x from E07 compares (GPU compute + store miss) vs (store hit).
//! Here we measure just the store side: lookup cost.

use std::time::Instant;
use winrapids_store::header::{BufferHeader, BufferPtr, DType, Location};
use winrapids_store::provenance::{provenance_hash, data_provenance};
use winrapids_store::store::{GpuStore, StoreStats};

fn main() {
    println!("{}", "=".repeat(70));
    println!("Observer Benchmark: Store Hit vs Miss Path");
    println!("{}", "=".repeat(70));

    bench_lookup_cost();
    bench_register_cost();
    bench_farm_simulation();
    bench_savings_ratio();

    println!("\n{}", "=".repeat(70));
    println!("Benchmark complete.");
    println!("{}", "=".repeat(70));
}

fn make_provenance(ticker: usize, cadence: usize, leaf: usize) -> [u8; 16] {
    let ticker_prov = data_provenance(&format!("ticker_{}", ticker));
    let cadence_prov = data_provenance(&format!("cadence_{}", cadence));
    provenance_hash(
        &[ticker_prov, cadence_prov],
        &format!("leaf_{}", leaf),
    )
}

fn make_header(prov: [u8; 16], cost_us: f32) -> BufferHeader {
    BufferHeader {
        provenance: prov,
        cost_us,
        access_count: 0,
        location: Location::Gpu,
        dtype: DType::F64,
        ndim: 1,
        flags: 0,
        _align: [0; 4],
        len: 100_000,
        byte_size: 800_000,
        created_ns: 0,
        last_access_ns: 0,
    }
}

fn bench_lookup_cost() {
    println!("\n--- Lookup (Hit Path) Cost ---\n");
    println!("  Phase 2 Entry 003: 35 ns (provenance lookup)");
    println!("  E07 Python: ~1 us (dict lookup + MD5 compare)\n");

    let mut store = GpuStore::new(1 << 40); // huge capacity, no evictions

    // Populate with 5000 entries
    let n_entries = 5000;
    let mut provs = Vec::with_capacity(n_entries);
    for i in 0..n_entries {
        let prov = make_provenance(i / 50, (i / 10) % 5, i % 10);
        let header = make_header(prov, 100.0);
        let ptr = BufferPtr { device_ptr: 0x1000 + i as u64 * 0x1000, byte_size: 800_000 };
        store.register(header, ptr);
        provs.push(prov);
    }

    // Warmup
    for _ in 0..3 {
        for p in &provs {
            let _ = store.lookup(p);
        }
    }

    // Timed: lookup all entries
    let mut times = Vec::with_capacity(20);
    for _ in 0..20 {
        let t0 = Instant::now();
        for p in &provs {
            let r = store.lookup(p);
            assert!(r.is_some());
        }
        let elapsed_ns = t0.elapsed().as_nanos() as f64;
        times.push(elapsed_ns / n_entries as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[10];
    let p99 = times[19];
    let mean = times.iter().sum::<f64>() / times.len() as f64;

    println!("  Per-lookup (5000 entries): p50={:.1} ns  p99={:.1} ns  mean={:.1} ns", p50, p99, mean);
}

fn bench_register_cost() {
    println!("\n--- Register (Miss Path) Cost ---\n");

    // Measure register() without eviction
    let mut store = GpuStore::new(1 << 40);

    let mut times = Vec::with_capacity(20);
    for run in 0..20 {
        let mut store = GpuStore::new(1 << 40);
        let n = 1000;
        let t0 = Instant::now();
        for i in 0..n {
            let prov = make_provenance(run * 1000 + i, 0, 0);
            let header = make_header(prov, 100.0);
            let ptr = BufferPtr { device_ptr: 0x1000 + i as u64 * 0x1000, byte_size: 800_000 };
            store.register(header, ptr);
        }
        let elapsed_ns = t0.elapsed().as_nanos() as f64;
        times.push(elapsed_ns / n as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[10];

    println!("  Per-register (no eviction): p50={:.1} ns", p50);

    // Measure register() WITH eviction (store full)
    let capacity = 1000 * 800_000; // exactly 1000 entries worth
    let mut store = GpuStore::new(capacity);

    // Fill store
    for i in 0..1000 {
        let prov = make_provenance(i, 0, 0);
        let header = make_header(prov, 100.0);
        let ptr = BufferPtr { device_ptr: 0x1000 + i as u64 * 0x1000, byte_size: 800_000 };
        store.register(header, ptr);
    }

    // Now register new entries (will trigger eviction)
    let mut evict_times = Vec::with_capacity(20);
    for run in 0..20 {
        let prov = make_provenance(2000 + run, 0, 0);
        let header = make_header(prov, 100.0);
        let ptr = BufferPtr { device_ptr: 0xFFFF_0000 + run as u64 * 0x1000, byte_size: 800_000 };
        let t0 = Instant::now();
        let evicted = store.register(header, ptr);
        let elapsed_ns = t0.elapsed().as_nanos() as f64;
        evict_times.push(elapsed_ns);
        assert!(!evicted.is_empty(), "Should evict when full");
    }
    evict_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("  Per-register (with eviction): p50={:.1} ns", evict_times[10]);
}

fn bench_farm_simulation() {
    println!("\n--- Farm Simulation: 100 tickers × 5 cadences × 10 leaves ---\n");

    let n_tickers = 100;
    let n_cadences = 5;
    let n_leaves = 10;
    let total = n_tickers * n_cadences * n_leaves;

    // First cycle: populate (all misses)
    let mut store = GpuStore::new(1 << 40);
    let mut all_provs: Vec<[u8; 16]> = Vec::with_capacity(total);

    let t0 = Instant::now();
    for t in 0..n_tickers {
        for c in 0..n_cadences {
            for l in 0..n_leaves {
                let prov = make_provenance(t, c, l);
                let header = make_header(prov, 100.0);
                let ptr = BufferPtr { device_ptr: all_provs.len() as u64 * 0x1000, byte_size: 800_000 };
                store.register(header, ptr);
                all_provs.push(prov);
            }
        }
    }
    let populate_us = t0.elapsed().as_nanos() as f64 / 1000.0;

    println!("  Populate ({} entries): {:.1} us ({:.1} ns/entry)",
        total, populate_us, populate_us * 1000.0 / total as f64);

    // Second cycle: various dirty ratios
    for &dirty_pct in &[100, 50, 20, 10, 5, 1] {
        let n_dirty = n_tickers * dirty_pct / 100;

        let t0 = Instant::now();
        let mut hits = 0u64;
        let mut misses = 0u64;

        for t in 0..n_tickers {
            let is_dirty = t < n_dirty;
            for c in 0..n_cadences {
                for l in 0..n_leaves {
                    let prov = make_provenance(t, c, l);
                    if is_dirty {
                        // Miss: re-register with updated provenance
                        let new_prov = provenance_hash(&[prov], "updated_v2");
                        let header = make_header(new_prov, 100.0);
                        let ptr = BufferPtr { device_ptr: 0xDEAD_0000, byte_size: 800_000 };
                        store.register(header, ptr);
                        misses += 1;
                    } else {
                        // Hit: lookup cached result
                        let result = store.lookup(&prov);
                        assert!(result.is_some());
                        hits += 1;
                    }
                }
            }
        }
        let cycle_us = t0.elapsed().as_nanos() as f64 / 1000.0;

        println!("  {:>3}% dirty: cycle={:7.1} us  hits={:>5}  misses={:>5}  ({:.1} ns/op)",
            dirty_pct, cycle_us, hits, misses, cycle_us * 1000.0 / total as f64);
    }
}

fn bench_savings_ratio() {
    println!("\n--- Savings Ratio: Hit vs Compute ---\n");
    println!("  E07 claim: 865x savings for rolling_std (hit vs compute)");
    println!("  E07 compute time: ~0.9ms for rolling_std on 10M");
    println!("  E07 hit time:     ~0.001ms (1 us dict lookup in Python)\n");

    // Rust store lookup: ~35ns (Entry 003)
    // GPU compute for rolling_std at FinTek size: ~440us (E06 rolling mean on 10M)
    // Ratio: 440,000 ns / 35 ns = 12,571x

    // More conservative: scan at 100K (FinTek typical): ~195us (Entry 009)
    // Ratio: 195,000 ns / 35 ns = 5,571x

    // Most conservative: CuPy cumsum at 100K: ~36us (Entry 001)
    // Ratio: 36,000 ns / 35 ns = 1,029x

    let lookup_ns = 35.0; // Entry 003

    println!("  Rust lookup cost: {:.0} ns (Entry 003)", lookup_ns);
    println!();
    println!("  Savings ratios (compute_time / lookup_time):");
    println!("    CuPy cumsum 100K (36 us):     {:>7.0}x", 36_000.0 / lookup_ns);
    println!("    Rust scan 100K (195 us):       {:>7.0}x  (includes transfer)", 195_000.0 / lookup_ns);
    println!("    CuPy rolling mean 10M (440 us):{:>7.0}x", 440_000.0 / lookup_ns);
    println!("    CuPy rolling std 10M (900 us): {:>7.0}x  (E07 baseline)", 900_000.0 / lookup_ns);
    println!();
    println!("  E07 865x was Python dict + MD5 (~1 us) vs rolling_std (~0.9ms)");
    println!("  Rust store achieves {:>.0}x for the same operation", 900_000.0 / lookup_ns);
    println!("  because Rust lookup (35 ns) is ~29x faster than Python dict (1 us)");
}
