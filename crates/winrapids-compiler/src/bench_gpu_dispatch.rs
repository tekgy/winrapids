//! Observer benchmark: plan-to-GPU-dispatch wall time.
//!
//! Now that CudaKernelDispatcher exists, we measure the REAL path:
//! 1. plan() compilation (CSE + topological sort + provenance)
//! 2. execute() with CudaKernelDispatcher (real GPU kernel launch)
//! 3. Provenance reuse on second run (GpuStore warm path)
//!
//! Key measurements:
//! A) Cold dispatch: plan() + execute() with all misses, real GPU scan
//! B) Warm dispatch: same plan, GpuStore has results, zero kernel launches
//! C) Per-step GPU dispatch overhead vs MockDispatcher
//!
//! Standing methodology: 3 warmup, 20 timed, p50/p99/mean.

use std::collections::HashMap;
use std::time::Instant;

use winrapids_compiler::plan::{PipelineSpec, SpecialistCall, plan};
use winrapids_compiler::execute::{execute, MockDispatcher};
use winrapids_compiler::cuda_dispatch::CudaKernelDispatcher;
use winrapids_compiler::registry::build_e04_registry;
use winrapids_scan::DevicePtr;
use winrapids_store::header::BufferPtr;
use winrapids_store::store::GpuStore;
use winrapids_store::world::NullWorld;
use winrapids_store::provenance::data_provenance;

fn main() {
    println!("{}", "=".repeat(70));
    println!("Observer Benchmark: Plan-to-GPU-Dispatch Wall Time");
    println!("{}", "=".repeat(70));

    bench_cold_gpu_dispatch();
    bench_warm_provenance_reuse();
    bench_gpu_vs_mock();
    bench_scan_only_dispatch();

    println!("\n{}", "=".repeat(70));
    println!("Benchmark complete.");
    println!("{}", "=".repeat(70));
}

/// Full plan() + execute() with CudaKernelDispatcher, all misses.
/// rolling_mean = scan(data, add) + fused_expr(sum/n).
/// Scan dispatches to GPU; FusedExpr stubs (expected error).
fn bench_cold_gpu_dispatch() {
    println!("\n--- Cold GPU Dispatch (NullWorld, All Misses) ---\n");

    let registry = build_e04_registry();
    let n = 100_000usize;

    // Create GPU-resident data
    let mut dispatcher = CudaKernelDispatcher::new()
        .expect("Failed to create CudaKernelDispatcher");
    let stream = dispatcher.engine().stream().clone();

    let host_data: Vec<f64> = (1..=n as u64).map(|x| x as f64).collect();
    let input_dev = stream.clone_htod(&host_data).unwrap();
    let input_ptr = { let (p, _g) = input_dev.device_ptr(&stream); p };
    let byte_size = (n * 8) as u64;

    // rolling_mean: scan(data, add) → will dispatch on GPU
    // rolling_mean also has fused_expr → will error
    // Use a custom 1-step pipeline that's ONLY a scan to get clean measurement
    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall {
                specialist: "rolling_mean".into(),
                data_var: "price".into(),
                window: 20,
            },
        ],
    };

    // Warmup: plan + execute with mock (to warm the registry code path)
    for _ in 0..3 {
        let exec_plan = plan(&spec, &registry, &mut NullWorld, None);
        let mut data_ptrs = HashMap::new();
        data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: input_ptr, byte_size });
        data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: input_ptr, byte_size });
        let mut mock = MockDispatcher::new();
        let _ = execute(&exec_plan, &mut NullWorld, &mut mock, &data_ptrs);
    }

    // Timed: plan + execute with real GPU dispatch
    // Execute will error at FusedExpr after dispatching Scan on GPU.
    // We measure to the error point — Scan dispatch IS the GPU work.
    let mut times = Vec::with_capacity(20);
    for _ in 0..20 {
        let t0 = Instant::now();
        let exec_plan = plan(&spec, &registry, &mut NullWorld, None);
        let mut data_ptrs = HashMap::new();
        data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: input_ptr, byte_size });
        data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: input_ptr, byte_size });
        let _ = execute(&exec_plan, &mut NullWorld, &mut dispatcher, &data_ptrs);
        let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
        times.push(elapsed_us);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = times[10];
    let p99 = times[19];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let p01 = times[0];

    println!("  Pipeline: rolling_mean(price, w=20) on {} elements", n);
    println!("  plan() + execute() with CudaKernelDispatcher:");
    println!("    p01={:.1} us  p50={:.1} us  p99={:.1} us  mean={:.1} us", p01, p50, p99, mean);
    println!("  (Scan dispatched on real GPU, FusedExpr stubs)");
}

/// Plan + execute with GpuStore containing prior results.
/// On warm run: all provenance hits → zero kernel launches.
fn bench_warm_provenance_reuse() {
    println!("\n--- Warm Provenance Reuse (GpuStore, All Hits) ---\n");

    let registry = build_e04_registry();
    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall {
                specialist: "rolling_zscore".into(),
                data_var: "price".into(),
                window: 20,
            },
            SpecialistCall {
                specialist: "rolling_std".into(),
                data_var: "price".into(),
                window: 20,
            },
        ],
    };

    let mut input_provs = HashMap::new();
    input_provs.insert("price".into(), data_provenance("price:AAPL:2026-03-30:1s"));

    let mut store = GpuStore::new(1_000_000_000);

    // Populate the store with a cold run
    let exec_plan = plan(&spec, &registry, &mut store, Some(&input_provs));
    let mut data_ptrs = HashMap::new();
    data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: 0x100, byte_size: 8000 });
    data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: 0x200, byte_size: 8000 });
    let mut mock = MockDispatcher::new();
    let (_, stats1) = execute(&exec_plan, &mut store, &mut mock, &data_ptrs).unwrap();
    println!("  Cold run: {} misses, {} hits", stats1.misses, stats1.hits);

    // Warmup: warm plan + execute
    for _ in 0..3 {
        let exec_plan = plan(&spec, &registry, &mut store, Some(&input_provs));
        let mut mock = MockDispatcher::new();
        let _ = execute(&exec_plan, &mut store, &mut mock, &data_ptrs);
    }

    // Timed: plan + execute on warm store (should be 100% hits, zero dispatch)
    let mut times = Vec::with_capacity(20);
    let mut steps_count = 0u64;
    for _ in 0..20 {
        let t0 = Instant::now();
        let exec_plan = plan(&spec, &registry, &mut store, Some(&input_provs));
        let mut mock = MockDispatcher::new();
        let (_, stats) = execute(&exec_plan, &mut store, &mut mock, &data_ptrs).unwrap();
        let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
        times.push(elapsed_us);
        steps_count = stats.total_steps;
        assert_eq!(stats.hits, stats.total_steps, "Warm run should be 100% hits");
        assert_eq!(mock.dispatch_log.len(), 0, "Zero dispatches on warm path");
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = times[10];
    let p99 = times[19];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let p01 = times[0];

    println!("  Pipeline: rolling_zscore + rolling_std (price, w=20)");
    println!("  Warm plan() + execute() (100% provenance hits):");
    println!("    p01={:.1} us  p50={:.1} us  p99={:.1} us  mean={:.1} us", p01, p50, p99, mean);
    println!("    {} steps, all hits, zero kernel dispatches", steps_count);
    println!("    Per-step overhead: {:.1} us", p50 / steps_count as f64);
}

/// Compare GPU dispatch vs Mock dispatch to isolate kernel launch cost.
fn bench_gpu_vs_mock() {
    println!("\n--- GPU vs Mock Dispatch Comparison ---\n");

    let registry = build_e04_registry();
    let n = 100_000usize;

    // Create GPU-resident data
    let mut gpu_disp = CudaKernelDispatcher::new().unwrap();
    let stream = gpu_disp.engine().stream().clone();
    let host_data: Vec<f64> = (1..=n as u64).map(|x| x as f64).collect();
    let input_dev = stream.clone_htod(&host_data).unwrap();
    let input_ptr = { let (p, _g) = input_dev.device_ptr(&stream); p };
    let byte_size = (n * 8) as u64;

    let spec = PipelineSpec {
        calls: vec![
            SpecialistCall {
                specialist: "rolling_mean".into(),
                data_var: "price".into(),
                window: 20,
            },
        ],
    };

    // Mock dispatch timing
    let mut mock_times = Vec::with_capacity(20);
    for _ in 0..3 {
        let exec_plan = plan(&spec, &registry, &mut NullWorld, None);
        let mut data_ptrs = HashMap::new();
        data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: 0x100, byte_size: 8000 });
        data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: 0x200, byte_size: 8000 });
        let mut mock = MockDispatcher::new();
        let _ = execute(&exec_plan, &mut NullWorld, &mut mock, &data_ptrs);
    }
    for _ in 0..20 {
        let exec_plan = plan(&spec, &registry, &mut NullWorld, None);
        let mut data_ptrs = HashMap::new();
        data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: 0x100, byte_size: 8000 });
        data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: 0x200, byte_size: 8000 });
        let mut mock = MockDispatcher::new();
        let t0 = Instant::now();
        let _ = execute(&exec_plan, &mut NullWorld, &mut mock, &data_ptrs);
        mock_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    mock_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // GPU dispatch timing (execute only, plan pre-computed)
    let exec_plan = plan(&spec, &registry, &mut NullWorld, None);
    let mut gpu_times = Vec::with_capacity(20);
    // Warmup GPU path
    for _ in 0..3 {
        let mut data_ptrs = HashMap::new();
        data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: input_ptr, byte_size });
        data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: input_ptr, byte_size });
        let _ = execute(&exec_plan, &mut NullWorld, &mut gpu_disp, &data_ptrs);
    }
    for _ in 0..20 {
        let mut data_ptrs = HashMap::new();
        data_ptrs.insert("data:price".into(), BufferPtr { device_ptr: input_ptr, byte_size });
        data_ptrs.insert("data_sq:price".into(), BufferPtr { device_ptr: input_ptr, byte_size });
        let t0 = Instant::now();
        let _ = execute(&exec_plan, &mut NullWorld, &mut gpu_disp, &data_ptrs);
        gpu_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mock_p50 = mock_times[10];
    let gpu_p50 = gpu_times[10];
    let overhead = gpu_p50 - mock_p50;

    println!("  Pipeline: rolling_mean on {} elements", n);
    println!("  Execute-only (plan pre-compiled):");
    println!("    Mock dispatch p50:  {:.1} us", mock_p50);
    println!("    GPU dispatch p50:   {:.1} us", gpu_p50);
    println!("    GPU kernel overhead: {:.1} us (gpu - mock)", overhead);
    println!("  (Scan only — FusedExpr stubs in both paths)");
}

/// Isolated scan dispatch: bypass the compiler entirely, measure
/// CudaKernelDispatcher::dispatch() for a single Scan step.
fn bench_scan_only_dispatch() {
    println!("\n--- Isolated Scan Dispatch (no compiler) ---\n");

    use winrapids_compiler::ir::PrimitiveOp;
    use winrapids_compiler::execute::KernelDispatcher;

    let mut dispatcher = CudaKernelDispatcher::new().unwrap();
    let stream = dispatcher.engine().stream().clone();

    for &n in &[1_000, 10_000, 100_000, 500_000] {
        let host_data: Vec<f64> = (1..=n as u64).map(|x| x as f64).collect();
        let input_dev = stream.clone_htod(&host_data).unwrap();
        let input_ptr = { let (p, _g) = input_dev.device_ptr(&stream); p };
        let byte_size = (n * 8) as u64;

        let params = vec![("agg".to_string(), "add".to_string())];
        let input_ptrs = vec![BufferPtr { device_ptr: input_ptr, byte_size }];

        // Warmup
        for _ in 0..3 {
            let _ = dispatcher.dispatch(&PrimitiveOp::Scan, &params, &input_ptrs);
        }

        // Timed
        let mut times = Vec::with_capacity(20);
        for _ in 0..20 {
            let t0 = Instant::now();
            let _ = dispatcher.dispatch(&PrimitiveOp::Scan, &params, &input_ptrs);
            times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = times[10];
        let p99 = times[19];
        let p01 = times[0];

        println!("  n={:>7}: dispatch p01={:7.1} us  p50={:7.1} us  p99={:7.1} us",
            n, p01, p50, p99);
    }

    println!("\n  This is scan_device_ptr: GPU→GPU, no H2D/D2H transfer tax.");
    println!("  Compare vs scan_inclusive (Entry 009): includes PCIe round-trip.");
}
