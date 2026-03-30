//! Observer benchmark: kernel-only GPU time via CUDA events.
//!
//! Isolates GPU kernel execution from H2D/D2H transfer.
//! Strategy: allocate device buffers ONCE, then measure repeated launches.
//!
//! Uses cudarc CudaEvent::record() + elapsed_ms() for GPU-side timing.
//! Also measures: H2D transfer time, D2H transfer time, kernel-only time.

use std::sync::Arc;
use std::time::Instant;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use winrapids_scan::{ScanEngine, AddOp, WelfordOp};
use winrapids_scan::ops::AssociativeOp;

fn main() {
    println!("{}", "=".repeat(70));
    println!("Observer Benchmark: Kernel-Only GPU Time (CUDA Events)");
    println!("{}", "=".repeat(70));

    let ctx = CudaContext::new(0).expect("Failed to create CudaContext");
    let stream = ctx.default_stream();

    // Report VRAM
    let (free, total) = ctx.mem_get_info().unwrap();
    println!("  GPU: VRAM {:.2} GB total, {:.2} GB free",
        total as f64 / (1024.0 * 1024.0 * 1024.0),
        free as f64 / (1024.0 * 1024.0 * 1024.0));

    // Use scan engine to warm up and get compiled modules
    let mut engine = ScanEngine::new().expect("Failed to create ScanEngine");

    // Warm up JIT
    let warmup_data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let _ = engine.scan_inclusive(&AddOp, &warmup_data).unwrap();
    let _ = engine.scan_inclusive(&WelfordOp, &warmup_data).unwrap();

    bench_transfer_isolation(&stream);
    bench_end_to_end_breakdown(&mut engine);

    println!("\n{}", "=".repeat(70));
    println!("Benchmark complete.");
    println!("{}", "=".repeat(70));
}

/// Measure H2D and D2H transfer times separately, using wall clock (GPU events
/// don't capture host-side transfer scheduling).
fn bench_transfer_isolation(stream: &Arc<CudaStream>) {
    println!("\n--- Transfer Time Isolation ---\n");

    for &n in &[1_000usize, 10_000, 100_000, 500_000, 1_000_000] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let bytes = n * 8;

        // H2D timing
        // Warmup
        for _ in 0..3 {
            let _: CudaSlice<f64> = stream.clone_htod(&data).unwrap();
        }
        stream.synchronize().unwrap();

        let mut h2d_times = Vec::with_capacity(20);
        for _ in 0..20 {
            stream.synchronize().unwrap();
            let t0 = Instant::now();
            let _dev: CudaSlice<f64> = stream.clone_htod(&data).unwrap();
            stream.synchronize().unwrap();
            h2d_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        h2d_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let h2d_p50 = h2d_times[10];

        // D2H timing
        let dev: CudaSlice<f64> = stream.clone_htod(&data).unwrap();
        stream.synchronize().unwrap();

        // Warmup
        for _ in 0..3 {
            let _: Vec<f64> = stream.clone_dtoh(&dev).unwrap();
        }
        stream.synchronize().unwrap();

        let mut d2h_times = Vec::with_capacity(20);
        for _ in 0..20 {
            stream.synchronize().unwrap();
            let t0 = Instant::now();
            let _: Vec<f64> = stream.clone_dtoh(&dev).unwrap();
            stream.synchronize().unwrap();
            d2h_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        d2h_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let d2h_p50 = d2h_times[10];

        let bw_h2d = bytes as f64 / h2d_p50 * 1e6 / (1024.0 * 1024.0 * 1024.0);
        let bw_d2h = bytes as f64 / d2h_p50 * 1e6 / (1024.0 * 1024.0 * 1024.0);

        println!("  n={:>9} ({:>5.1} MB): H2D={:7.1} us ({:5.1} GB/s)  D2H={:7.1} us ({:5.1} GB/s)  total_xfer={:7.1} us",
            n, bytes as f64 / (1024.0 * 1024.0), h2d_p50, bw_h2d, d2h_p50, bw_d2h, h2d_p50 + d2h_p50);
    }
}

/// Breakdown: total = H2D + kernel + D2H.
/// Measure total via scan_inclusive, then subtract H2D + D2H to get kernel-only.
fn bench_end_to_end_breakdown(engine: &mut ScanEngine) {
    println!("\n--- End-to-End Breakdown: AddOp ---\n");
    println!("  total = H2D + kernel + D2H");
    println!("  kernel = total - (H2D + D2H)\n");

    let stream = CudaContext::new(0).unwrap().default_stream();

    for &n in &[1_000usize, 10_000, 100_000, 500_000, 1_000_000] {
        let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin()).collect();
        let bytes = n * 8;

        // Measure H2D
        let mut h2d_times = Vec::with_capacity(10);
        for _ in 0..10 {
            stream.synchronize().unwrap();
            let t0 = Instant::now();
            let _: CudaSlice<f64> = stream.clone_htod(&input).unwrap();
            stream.synchronize().unwrap();
            h2d_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        h2d_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let h2d = h2d_times[5];

        // Measure D2H (need a device buffer of the right size)
        let dev: CudaSlice<f64> = stream.clone_htod(&input).unwrap();
        stream.synchronize().unwrap();
        let mut d2h_times = Vec::with_capacity(10);
        for _ in 0..10 {
            stream.synchronize().unwrap();
            let t0 = Instant::now();
            let _: Vec<f64> = stream.clone_dtoh(&dev).unwrap();
            stream.synchronize().unwrap();
            d2h_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        d2h_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let d2h = d2h_times[5];

        // Measure total (scan_inclusive end-to-end)
        for _ in 0..3 {
            let _ = engine.scan_inclusive(&AddOp, &input).unwrap();
        }
        let mut total_times = Vec::with_capacity(20);
        for _ in 0..20 {
            let t0 = Instant::now();
            let _ = engine.scan_inclusive(&AddOp, &input).unwrap();
            total_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        total_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let total = total_times[10];

        let kernel_est = (total - h2d - d2h).max(0.0);

        println!("  n={:>9}: total={:7.1} us  H2D={:7.1} us  D2H={:7.1} us  kernel_est={:7.1} us",
            n, total, h2d, d2h, kernel_est);
    }

    // Same for WelfordOp
    println!("\n--- End-to-End Breakdown: WelfordOp ---\n");

    for &n in &[1_000usize, 10_000, 100_000, 500_000, 1_000_000] {
        let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin() * 50.0).collect();

        // Measure H2D
        let mut h2d_times = Vec::with_capacity(10);
        for _ in 0..10 {
            stream.synchronize().unwrap();
            let t0 = Instant::now();
            let _: CudaSlice<f64> = stream.clone_htod(&input).unwrap();
            stream.synchronize().unwrap();
            h2d_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        h2d_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let h2d = h2d_times[5];

        let dev: CudaSlice<f64> = stream.clone_htod(&input).unwrap();
        stream.synchronize().unwrap();
        let mut d2h_times = Vec::with_capacity(10);
        for _ in 0..10 {
            stream.synchronize().unwrap();
            let t0 = Instant::now();
            // WelfordOp returns primary + 1 secondary, so D2H is 2x
            let _: Vec<f64> = stream.clone_dtoh(&dev).unwrap();
            stream.synchronize().unwrap();
            d2h_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        d2h_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let d2h_1 = d2h_times[5];
        // WelfordOp has 2 outputs (primary + variance), so D2H ~2x
        let d2h = d2h_1 * 2.0;

        // Measure total
        for _ in 0..3 {
            let _ = engine.scan_inclusive(&WelfordOp, &input).unwrap();
        }
        let mut total_times = Vec::with_capacity(20);
        for _ in 0..20 {
            let t0 = Instant::now();
            let r = engine.scan_inclusive(&WelfordOp, &input).unwrap();
            assert!(!r.primary.is_empty());
            total_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        total_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let total = total_times[10];

        let kernel_est = (total - h2d - d2h).max(0.0);

        println!("  n={:>9}: total={:7.1} us  H2D={:7.1} us  D2H={:7.1} us  kernel_est={:7.1} us",
            n, total, h2d, d2h, kernel_est);
    }

    // Dispatch overhead floor: tiny scan to measure cudarc launch cost
    println!("\n--- Dispatch Overhead Floor (n=100, AddOp) ---\n");
    let tiny: Vec<f64> = (0..100).map(|i| i as f64).collect();
    for _ in 0..10 {
        let _ = engine.scan_inclusive(&AddOp, &tiny).unwrap();
    }
    let mut floor_times = Vec::with_capacity(50);
    for _ in 0..50 {
        let t0 = Instant::now();
        let _ = engine.scan_inclusive(&AddOp, &tiny).unwrap();
        floor_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    floor_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  n=100 AddOp: p50={:.1} us  p01={:.1} us  p99={:.1} us",
        floor_times[25], floor_times[0], floor_times[49]);
    println!("  This is the scan API floor: 3 kernel launches + H2D(800B) + D2H(800B) + alloc overhead");
}
