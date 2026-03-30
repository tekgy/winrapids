use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::time::Instant;

const TRIVIAL_KERNEL: &str = r#"
extern "C" __global__
void add_one(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = input[idx] + 1.0f;
}
"#;

const MEDIUM_KERNEL: &str = r#"
extern "C" __global__
void fused_expr(const float* a, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double va = (double)a[idx];
        double r = va * va + va * 2.0 - 1.0;
        r = r / (va + 1.0001);
        r = r + sqrt(fabs(va));
        double s = log(fabs(va) + 1.0) * exp(-va * va * 0.01);
        s = s + pow(fabs(va), 0.3333);
        r = r + s * 0.1;
        r = fmin(fmax(r, -1e6), 1e6);
        out[idx] = r;
    }
}
"#;

const COMPLEX_KERNEL: &str = r#"
extern "C" __global__
void rolling_zscore(const float* data, float* z_out,
                    const double* cumsum, const double* cumsum_sq,
                    int n, int window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int data_idx = idx + window - 1;
    if (data_idx < n) {
        double sum_w = cumsum[data_idx + 1] - cumsum[data_idx + 1 - window];
        double sq_w = cumsum_sq[data_idx + 1] - cumsum_sq[data_idx + 1 - window];
        double mean = sum_w / (double)window;
        double mean_sq = sq_w / (double)window;
        double var = mean_sq - mean * mean;
        if (var < 0.0) var = 0.0;
        double std_val = sqrt(var);
        double val = (double)data[data_idx];
        double z = (std_val > 1e-10) ? (val - mean) / std_val : 0.0;
        z_out[idx] = (float)z;
    }
}
"#;

fn time_compile(label: &str, src: &str) -> f64 {
    let start = Instant::now();
    match compile_ptx(src) {
        Ok(_) => {
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            println!("  {label}: {ms:.1} ms");
            ms
        }
        Err(e) => {
            println!("  {label}: FAILED - {e}");
            0.0
        }
    }
}

fn main() {
    println!("=== E09: NVRTC from Rust ===\n");

    // Warmup: first NVRTC call loads the DLL
    println!("--- Warmup (loads nvrtc DLL) ---");
    let start = Instant::now();
    let _ = compile_ptx(TRIVIAL_KERNEL);
    println!("  warmup: {:.1} ms\n", start.elapsed().as_secs_f64() * 1000.0);

    // Compilation benchmarks
    println!("--- NVRTC Compilation Time (5 runs each) ---\n");

    let kernels = [
        ("trivial (1 op)", TRIVIAL_KERNEL),
        ("medium (10 ops)", MEDIUM_KERNEL),
        ("complex (pipeline)", COMPLEX_KERNEL),
    ];

    for (label, src) in &kernels {
        let mut times = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let _ = compile_ptx(src);
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[2];
        let min = times[0];
        let max = times[4];
        println!("  {label}:");
        println!("    median={median:.1} ms, min={min:.1} ms, max={max:.1} ms");
    }

    // End-to-end: compile + load + launch + verify
    println!("\n--- End-to-End: Compile + Load + Launch + Verify ---\n");

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Compile
    let start = Instant::now();
    let ptx = compile_ptx(TRIVIAL_KERNEL).unwrap();
    let compile_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Load module
    let start = Instant::now();
    let module = ctx.load_module(ptx).unwrap();
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Get function
    let f = module.load_function("add_one").unwrap();

    println!("  NVRTC compile: {compile_ms:.1} ms");
    println!("  Module load:   {load_ms:.1} ms");
    println!("  Total JIT:     {:.1} ms", compile_ms + load_ms);

    // Prepare data
    let n: usize = 1_000_000;
    let host_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input = stream.clone_htod(&host_data).unwrap();
    let mut output = stream.alloc_zeros::<f32>(n).unwrap();

    let threads = 256u32;
    let blocks = ((n as u32) + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    // First launch
    let start = Instant::now();
    unsafe {
        stream.launch_builder(&f)
            .arg(&input)
            .arg(&mut output)
            .arg(&(n as i32))
            .launch(cfg)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let first_launch = start.elapsed().as_secs_f64() * 1000.0;

    // Verify correctness
    let result = stream.clone_dtoh(&output).unwrap();
    let correct = result.iter().enumerate().all(|(i, &v)| (v - (i as f32 + 1.0)).abs() < 1e-5);
    println!("\n  First launch:  {first_launch:.3} ms (1M elements)");
    println!("  Correctness:   {}", if correct { "PASS" } else { "FAIL" });

    // Cached launches (kernel already loaded, measure raw launch overhead)
    let mut launch_times = Vec::new();
    for _ in 0..100 {
        stream.synchronize().unwrap();
        let start = Instant::now();
        unsafe {
            stream.launch_builder(&f)
                .arg(&input)
                .arg(&mut output)
                .arg(&(n as i32))
                .launch(cfg)
                .unwrap();
        }
        stream.synchronize().unwrap();
        launch_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    launch_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let launch_p50 = launch_times[50];
    let launch_best = launch_times[0];
    let launch_p99 = launch_times[99];

    println!("\n  Cached launch latency (100 runs):");
    println!("    p50:  {launch_p50:.3} ms");
    println!("    best: {launch_best:.3} ms");
    println!("    p99:  {launch_p99:.3} ms");
    println!("\n  CuPy launch overhead (from E06): ~0.070 ms");
    println!("  Rust advantage: {:.1}x", 0.070 / launch_p50);

    // Empty kernel launch overhead
    let empty_src = r#"extern "C" __global__ void empty_kern() {}"#;
    let empty_ptx = compile_ptx(empty_src).unwrap();
    let empty_mod = ctx.load_module(empty_ptx).unwrap();
    let empty_fn = empty_mod.load_function("empty_kern").unwrap();

    let empty_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warmup
    for _ in 0..10 {
        unsafe {
            stream.launch_builder(&empty_fn)
                .launch(empty_cfg)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();

    let mut empty_times = Vec::new();
    for _ in 0..100 {
        stream.synchronize().unwrap();
        let start = Instant::now();
        unsafe {
            stream.launch_builder(&empty_fn)
                .launch(empty_cfg)
                .unwrap();
        }
        stream.synchronize().unwrap();
        empty_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    empty_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("\n  Empty kernel launch:");
    println!("    p50:  {:.3} ms", empty_times[50]);
    println!("    best: {:.3} ms", empty_times[0]);
    println!("    p99:  {:.3} ms", empty_times[99]);
    println!("    CuPy empty: ~0.008 ms (from E06)");

    println!("\n=== E09 COMPLETE ===");
}
