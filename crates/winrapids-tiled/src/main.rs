//! Tiled accumulation validation — kernel generation + GPU dispatch tests.

use winrapids_tiled::ops::*;
use winrapids_tiled::engine::generate_tiled_kernel;
use winrapids_tiled::cache::cache_key;
use winrapids_tiled::TiledEngine;

fn main() {
    println!("{}", "=".repeat(70));
    println!("winrapids-tiled validation");
    println!("{}", "=".repeat(70));

    test_dot_product();
    test_outer_product();
    test_covariance();
    test_distance();
    test_softmax_weighted();
    test_operator_properties();
    test_gpu_dispatch();

    println!("\n{}", "=".repeat(70));
    println!("ALL TILED TESTS PASSED");
    println!("{}", "=".repeat(70));
}

fn test_gpu_dispatch() {
    println!("\n--- Test 7: GPU dispatch (DotProduct 2×3 × 3×2) ---");

    let gpu = tam_gpu::detect();
    println!("  Backend: {}", gpu.name());

    let engine = TiledEngine::new(std::sync::Arc::clone(&gpu));

    // A (2×3) × B (3×2) → C (2×2)
    // A = [[1,2,3],[4,5,6]]   B = [[7,8],[9,10],[11,12]]
    // C[0,0] = 1*7+2*9+3*11=58   C[0,1] = 1*8+2*10+3*12=64
    // C[1,0] = 4*7+5*9+6*11=139  C[1,1] = 4*8+5*10+6*12=154
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];

    let c = engine.run(&DotProductOp, &a, &b, 2, 2, 3).unwrap();
    assert!((c[0] - 58.0).abs() < 1e-9, "C[0,0] expected 58, got {}", c[0]);
    assert!((c[1] - 64.0).abs() < 1e-9, "C[0,1] expected 64, got {}", c[1]);
    assert!((c[2] - 139.0).abs() < 1e-9, "C[1,0] expected 139, got {}", c[2]);
    assert!((c[3] - 154.0).abs() < 1e-9, "C[1,1] expected 154, got {}", c[3]);
    println!("  DotProduct 2×3 × 3×2: {:?}  PASS", c);

    // Cache: second call hits cache, same result
    let c2 = engine.run(&DotProductOp, &a, &b, 2, 2, 3).unwrap();
    assert_eq!(c, c2, "Cached kernel must produce identical result");
    assert_eq!(engine.cache_len(), 1, "Should have exactly 1 cached kernel");
    println!("  Cache hit: cache_len={}  PASS", engine.cache_len());

    // Distance: [[0,0]] vs [[3,4]] → L2² = 9+16 = 25
    let p = vec![0.0f64, 0.0];
    let q = vec![3.0f64, 4.0];
    let dist_sq = engine.run(&DistanceOp, &p, &q, 1, 1, 2).unwrap();
    assert!((dist_sq[0] - 25.0).abs() < 1e-9, "L2² expected 25, got {}", dist_sq[0]);
    println!("  DistanceOp 1×2 L2²=25: {:?}  PASS", dist_sq);
}

fn test_dot_product() {
    println!("\n--- Test 1: DotProduct kernel ---");
    let kernel = generate_tiled_kernel(&DotProductOp);

    assert!(kernel.contains("tiled_accumulate"), "Should contain kernel name");
    assert!(kernel.contains("acc += a_val * b_val"), "Should contain dot product accumulate");
    assert!(kernel.contains("typedef double acc_t"), "Should have scalar accumulator");
    assert!(kernel.contains("return x;"), "Identity pre-transform");
    assert!(kernel.contains("const int* __restrict__ dims"), "Should use dims buffer for M/N/K");
    assert!(kernel.contains("int M = dims[0]"), "Should extract M from dims");

    println!("  DotProductOp: {} bytes  PASS", kernel.len());
}

fn test_outer_product() {
    println!("\n--- Test 2: OuterProduct kernel ---");
    let kernel = generate_tiled_kernel(&OuterProductOp);

    assert!(kernel.contains("acc += a_val * b_val"), "Same accumulate as dot product");
    println!("  OuterProductOp: {} bytes  PASS", kernel.len());
}

fn test_covariance() {
    println!("\n--- Test 3: Covariance kernel (fused centering) ---");

    // Without centering
    let cov_raw = CovarianceOp {
        n_cols: 100,
        mean_a_expr: String::new(),
        mean_b_expr: String::new(),
    };
    let kernel_raw = generate_tiled_kernel(&cov_raw);
    assert!(kernel_raw.contains("return x;"), "No centering = identity transform");
    assert!(kernel_raw.contains("/ (double)(100 - 1)"), "Should normalize by n-1");
    println!("  CovarianceOp (raw): {} bytes  PASS", kernel_raw.len());

    // With centering — the fusion advantage
    let cov_centered = CovarianceOp {
        n_cols: 100,
        mean_a_expr: "row_mean_a".into(),
        mean_b_expr: "row_mean_b".into(),
    };
    let kernel_centered = generate_tiled_kernel(&cov_centered);
    assert!(kernel_centered.contains("(x - row_mean_a)"), "Should fuse A centering");
    assert!(kernel_centered.contains("(x - row_mean_b)"), "Should fuse B centering");
    println!("  CovarianceOp (centered): {} bytes  PASS", kernel_centered.len());

    // Auto-covariance (A = B, same means)
    let cov_auto = CovarianceOp {
        n_cols: 50,
        mean_a_expr: "mu".into(),
        mean_b_expr: String::new(),
    };
    let kernel_auto = generate_tiled_kernel(&cov_auto);
    assert!(kernel_auto.contains("(x - mu)"), "Should center both A and B with same mean");
    println!("  CovarianceOp (auto): {} bytes  PASS", kernel_auto.len());
}

fn test_distance() {
    println!("\n--- Test 4: L2 Distance kernel ---");
    let kernel = generate_tiled_kernel(&DistanceOp);

    assert!(kernel.contains("double diff = a_val - b_val"), "Should compute difference");
    assert!(kernel.contains("acc += diff * diff"), "Should accumulate squared diff");
    println!("  DistanceOp: {} bytes  PASS", kernel.len());
}

fn test_softmax_weighted() {
    println!("\n--- Test 5: SoftmaxWeighted kernel (FlashAttention pattern) ---");
    let kernel = generate_tiled_kernel(&SoftmaxWeightedOp);

    assert!(kernel.contains("SoftmaxAcc"), "Should have struct accumulator");
    assert!(kernel.contains("max_val"), "Should track running max");
    assert!(kernel.contains("exp_sum"), "Should track exp denominator");
    assert!(kernel.contains("weighted_sum"), "Should track weighted numerator");
    assert!(kernel.contains("exp(acc.max_val - score)"), "Should rescale on new max");
    println!("  SoftmaxWeightedOp: {} bytes  PASS", kernel.len());
}

fn test_operator_properties() {
    println!("\n--- Test 6: Operator properties ---");

    let ops: Vec<(&str, Box<dyn TiledOp>)> = vec![
        ("DotProduct", Box::new(DotProductOp)),
        ("OuterProduct", Box::new(OuterProductOp)),
        ("Covariance(100)", Box::new(CovarianceOp {
            n_cols: 100,
            mean_a_expr: "mu".into(),
            mean_b_expr: String::new(),
        })),
        ("L2Distance", Box::new(DistanceOp)),
        ("SoftmaxWeighted", Box::new(SoftmaxWeightedOp)),
    ];

    for (label, op) in &ops {
        let acc_kind = if op.cuda_acc_type().contains("struct") { "struct" } else { "scalar" };
        let has_pre = op.cuda_pre_transform_a() != "x" || op.cuda_pre_transform_b() != "x";
        println!("  {:20} acc={:6} bytes={:2}  pre_transform={}  params={}",
            label, acc_kind, op.acc_byte_size(), has_pre, op.params_key());
    }

    // Verify sizes
    assert_eq!(DotProductOp.acc_byte_size(), 8);
    assert_eq!(SoftmaxWeightedOp.acc_byte_size(), 24);
    println!("  Accumulator byte sizes  PASS");

    // Cache key uniqueness (BLAKE3-based)
    let key_dot = cache_key(&DotProductOp);
    let key_dist = cache_key(&DistanceOp);
    assert_ne!(key_dot, key_dist, "DotProduct and Distance must have different cache keys");
    println!("  DotProduct vs Distance cache keys differ  PASS");

    let cov1 = CovarianceOp { n_cols: 100, mean_a_expr: String::new(), mean_b_expr: String::new() };
    let cov2 = CovarianceOp { n_cols: 200, mean_a_expr: String::new(), mean_b_expr: String::new() };
    assert_ne!(cache_key(&cov1), cache_key(&cov2), "Different n_cols must have different cache keys");
    println!("  Covariance(100) vs Covariance(200) cache keys differ  PASS");
}
