//! End-to-end test: recipe → fuse → codegen → NVRTC → GPU dispatch → verify.
//!
//! This is the "does the vendor door actually work on silicon?" test.
//! It takes a recipe, fuses it, emits CUDA C, compiles via tam-gpu's
//! CudaBackend (which dynamic-loads nvcuda.dll), runs the kernel on the
//! GPU, reads back results, and compares to the pure-Rust executor.
//!
//! If no CUDA-capable GPU is present, every test in this file no-ops
//! with a printed skip message — so the crate still builds and `cargo
//! test` still passes on CPU-only machines.

use tambear_primitives::accumulates::{fuse_passes, execute_pass_cpu, Grouping, Op, DataSource};
use tambear_primitives::codegen::pass_to_cuda_kernel;
use tambear_primitives::recipes::{
    variance, mean_arithmetic, sum, sum_of_squares, mean_quadratic, pearson_r, l1_norm,
};
use tambear_primitives::tbs::{Expr, eval};
use tambear_primitives::accumulates::AccumulateSlot;
use tambear_primitives::gathers::Gather;
use tambear_primitives::recipes::Recipe;

use std::collections::HashMap;

use tam_gpu::{TamGpu, CudaBackend, upload, download};

/// Try to get a working CUDA backend; if unavailable (no driver, no GPU,
/// container, etc.), print why and return `None` so the test becomes a
/// no-op instead of a failure.
fn try_cuda() -> Option<CudaBackend> {
    match CudaBackend::new() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping GPU test — CUDA backend unavailable: {e:?}");
            None
        }
    }
}

/// Dispatch a fused pass on GPU and return the per-slot output values.
///
/// Layout contract with `pass_to_cuda_kernel`:
///   - buffer 0: data_x (f64, n elements)
///   - buffer 1: data_y if `n_inputs == 2` (f64, n elements)
///   - last buffer: out_slots (f64, n_outputs elements, zero-initialized)
fn dispatch_pass(
    gpu: &dyn TamGpu,
    pass: &tambear_primitives::accumulates::AccumulatePass,
    x: &[f64],
    y: &[f64],
    reference: f64,
) -> Vec<f64> {
    let entry = format!("tam_{}_{:?}_{:?}", pass_name_hint(pass), pass.grouping, pass.op)
        .replace("::", "_")
        .to_lowercase();
    let entry = sanitize_ident(&entry);

    let src = pass_to_cuda_kernel(pass, &entry, x.len(), reference)
        .expect("pass must be lowerable");

    // Compile. NVRTC cost is ~40 ms the first time; cached thereafter.
    let kernel = gpu.compile(&src.source, &src.entry)
        .unwrap_or_else(|e| panic!("NVRTC compile failed:\n{e:?}\n\n--- source ---\n{}", src.source));

    // Upload input(s).
    let x_buf = upload::<f64>(gpu, x).expect("upload x");
    let y_buf_opt: Option<tam_gpu::Buffer> = if src.n_inputs == 2 {
        Some(upload::<f64>(gpu, y).expect("upload y"))
    } else {
        None
    };

    // Allocate zero-initialized output slots.
    let out_bytes = src.n_outputs * std::mem::size_of::<f64>();
    let out_buf = gpu.alloc(out_bytes).expect("alloc out");

    // Build buffer list. atomicAdd requires the output to start at 0.0 —
    // `alloc` gives us zeroed memory, so we're good without an explicit clear.
    let buffers: Vec<&tam_gpu::Buffer> = if let Some(ref yb) = y_buf_opt {
        vec![&x_buf, yb, &out_buf]
    } else {
        vec![&x_buf, &out_buf]
    };

    // Reasonable launch config: 256 threads/block, enough blocks to cover.
    let block = 256u32;
    let grid  = ((x.len() as u32 + block - 1) / block).max(1);

    gpu.dispatch(&kernel, [grid, 1, 1], [block, 1, 1], &buffers, 0)
        .expect("dispatch");
    gpu.sync().expect("sync");

    download::<f64>(gpu, &out_buf, src.n_outputs).expect("download")
}

fn pass_name_hint(pass: &tambear_primitives::accumulates::AccumulatePass) -> String {
    // Use the first slot's output name as an entry hint.
    pass.slots.first().map(|(_, n)| n.clone()).unwrap_or_else(|| "pass".into())
}

fn sanitize_ident(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(true) {
        out.insert(0, 'k');
    }
    out
}

/// Run the full recipe: dispatch each fused pass on GPU, evaluate gathers,
/// return the final result.
fn run_recipe_gpu(
    gpu: &dyn TamGpu,
    recipe: &Recipe,
    x: &[f64],
    y: &[f64],
    reference: f64,
) -> f64 {
    let passes = fuse_passes(&recipe.slots);
    let mut vars: HashMap<String, f64> = HashMap::new();
    for pass in &passes {
        let outs = dispatch_pass(gpu, pass, x, y, reference);
        for ((_, name), val) in pass.slots.iter().zip(outs.iter()) {
            vars.insert(name.clone(), *val);
        }
    }
    let mut final_val = 0.0;
    for g in &recipe.gathers {
        let v = eval(&g.expr, 0.0, 0.0, reference, &vars);
        vars.insert(g.output.clone(), v);
        if g.output == recipe.result { final_val = v; }
    }
    final_val
}

/// Same thing on CPU for comparison.
fn run_recipe_cpu(recipe: &Recipe, x: &[f64], y: &[f64], reference: f64) -> f64 {
    let passes = fuse_passes(&recipe.slots);
    let mut vars: HashMap<String, f64> = HashMap::new();
    for pass in &passes {
        let results = execute_pass_cpu(pass, x, reference, y);
        for (name, val) in results {
            vars.insert(name, val);
        }
    }
    let mut final_val = 0.0;
    for g in &recipe.gathers {
        let v = eval(&g.expr, 0.0, 0.0, reference, &vars);
        vars.insert(g.output.clone(), v);
        if g.output == recipe.result { final_val = v; }
    }
    final_val
}

// ═══════════════════════════════════════════════════════════════════
// The tests
// ═══════════════════════════════════════════════════════════════════

#[test]
fn gpu_sum_matches_cpu() {
    let gpu = match try_cuda() { Some(g) => g, None => return };
    eprintln!("device: {}", gpu.name());
    let data: Vec<f64> = (1..=10_000).map(|i| i as f64).collect();
    let cpu_v = run_recipe_cpu(&sum(), &data, &[], 0.0);
    let gpu_v = run_recipe_gpu(&gpu, &sum(), &data, &[], 0.0);
    // Expected: n*(n+1)/2 = 50005000.0 — exactly representable in f64.
    assert_eq!(cpu_v, 50_005_000.0);
    assert!((gpu_v - cpu_v).abs() < 1e-6,
        "GPU sum {gpu_v} differs from CPU {cpu_v}");
}

#[test]
fn gpu_mean_matches_cpu() {
    let gpu = match try_cuda() { Some(g) => g, None => return };
    let data: Vec<f64> = (1..=1_000).map(|i| i as f64).collect();
    let cpu_v = run_recipe_cpu(&mean_arithmetic(), &data, &[], 0.0);
    let gpu_v = run_recipe_gpu(&gpu, &mean_arithmetic(), &data, &[], 0.0);
    assert!((gpu_v - cpu_v).abs() < 1e-9,
        "mean: GPU {gpu_v} vs CPU {cpu_v}");
    // Also check against the closed form: (1+1000)/2 = 500.5
    assert!((cpu_v - 500.5).abs() < 1e-12);
}

#[test]
fn gpu_variance_matches_cpu() {
    let gpu = match try_cuda() { Some(g) => g, None => return };
    // Uses 3 fused slots: sum, sum_sq, count — all Add/All.
    let data: Vec<f64> = (0..10_000)
        .map(|i| (i as f64).sin() * 10.0 + 5.0)
        .collect();
    let cpu_v = run_recipe_cpu(&variance(), &data, &[], 0.0);
    let gpu_v = run_recipe_gpu(&gpu, &variance(), &data, &[], 0.0);
    let rel = (gpu_v - cpu_v).abs() / cpu_v.abs().max(1e-12);
    eprintln!("variance: cpu={cpu_v} gpu={gpu_v} rel_err={rel:e}");
    // GPU atomicAdd is non-deterministic but the magnitudes here allow
    // ~12 digits of agreement.
    assert!(rel < 1e-10, "variance relative error {rel} too large");
}

#[test]
fn gpu_rms_matches_cpu() {
    let gpu = match try_cuda() { Some(g) => g, None => return };
    let data: Vec<f64> = (1..=2048).map(|i| i as f64).collect();
    let cpu_v = run_recipe_cpu(&mean_quadratic(), &data, &[], 0.0);
    let gpu_v = run_recipe_gpu(&gpu, &mean_quadratic(), &data, &[], 0.0);
    let rel = (gpu_v - cpu_v).abs() / cpu_v.abs().max(1e-12);
    assert!(rel < 1e-12, "rms: cpu={cpu_v} gpu={gpu_v}");
}

#[test]
fn gpu_sum_of_squares_matches_cpu() {
    let gpu = match try_cuda() { Some(g) => g, None => return };
    let data: Vec<f64> = (0..5_000).map(|i| (i as f64) / 100.0).collect();
    let cpu_v = run_recipe_cpu(&sum_of_squares(), &data, &[], 0.0);
    let gpu_v = run_recipe_gpu(&gpu, &sum_of_squares(), &data, &[], 0.0);
    let rel = (gpu_v - cpu_v).abs() / cpu_v.abs().max(1e-12);
    assert!(rel < 1e-11, "ss: cpu={cpu_v} gpu={gpu_v}");
}

#[test]
fn gpu_l1_norm_matches_cpu() {
    let gpu = match try_cuda() { Some(g) => g, None => return };
    let data: Vec<f64> = (0..4096)
        .map(|i| if i % 2 == 0 { i as f64 } else { -(i as f64) })
        .collect();
    let cpu_v = run_recipe_cpu(&l1_norm(), &data, &[], 0.0);
    let gpu_v = run_recipe_gpu(&gpu, &l1_norm(), &data, &[], 0.0);
    let rel = (gpu_v - cpu_v).abs() / cpu_v.abs().max(1e-12);
    assert!(rel < 1e-12, "l1: cpu={cpu_v} gpu={gpu_v}");
}

#[test]
fn gpu_pearson_matches_cpu_two_input() {
    let gpu = match try_cuda() { Some(g) => g, None => return };
    // Perfect linear: y = 2.5*x + 7
    let n = 1024;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.5 * xi + 7.0).collect();
    let cpu_v = run_recipe_cpu(&pearson_r(), &x, &y, 0.0);
    let gpu_v = run_recipe_gpu(&gpu, &pearson_r(), &x, &y, 0.0);
    eprintln!("pearson: cpu={cpu_v} gpu={gpu_v}");
    assert!((cpu_v - 1.0).abs() < 1e-10);
    assert!((gpu_v - 1.0).abs() < 1e-10);
    assert!((gpu_v - cpu_v).abs() < 1e-12);
}

#[test]
fn gpu_dispatches_custom_expression() {
    let gpu = match try_cuda() { Some(g) => g, None => return };
    // A bespoke recipe: Σ |ln(x)|  → one slot, Add/All, Primary.
    let custom = Recipe {
        name: "sum_abs_ln".into(),
        slots: vec![AccumulateSlot {
            source: DataSource::Primary,
            expr: Expr::val().ln().abs(),
            grouping: Grouping::All,
            op: Op::Add,
            output: "s".into(),
        }],
        gathers: vec![Gather {
            expr: Expr::var("s"),
            output: "result".into(),
        }],
        result: "result".into(),
    };
    let data: Vec<f64> = (1..=1000).map(|i| i as f64).collect();
    let cpu_v = run_recipe_cpu(&custom, &data, &[], 0.0);
    let gpu_v = run_recipe_gpu(&gpu, &custom, &data, &[], 0.0);
    let rel = (gpu_v - cpu_v).abs() / cpu_v.abs().max(1e-12);
    eprintln!("Σ|ln x|: cpu={cpu_v} gpu={gpu_v} rel={rel:e}");
    assert!(rel < 1e-11);
}

#[test]
fn gpu_device_name_printed() {
    let gpu = match try_cuda() { Some(g) => g, None => return };
    // Just confirm detection works and name is non-empty.
    let name = gpu.name();
    assert!(!name.is_empty());
    eprintln!("GPU device: {name}");
}
