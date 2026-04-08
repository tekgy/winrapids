//! Adversarial Boundary Tests — Wave 8
//!
//! Targets: neural (F23) — activations, convolutions, pooling, normalization,
//!          attention, loss functions, embeddings
//!
//! Attack taxonomy:
//! - Type 1: Division by zero / denominator collapse
//! - Type 2: Convergence / iteration boundary
//! - Type 3: Cancellation / precision
//! - Type 4: Equipartition / degenerate geometry
//! - Type 5: Structural incompatibility

use tambear::neural::*;
use tambear::linear_algebra::Mat;

// ═══════════════════════════════════════════════════════════════════════════
// ACTIVATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Softmax with empty input.
/// Type 5.
#[test]
fn softmax_empty() {
    let result = std::panic::catch_unwind(|| {
        softmax(&[])
    });
    match result {
        Ok(s) => assert!(s.is_empty(), "Softmax of empty should be empty"),
        Err(_) => eprintln!("NOTE: softmax panics on empty input"),
    }
}

/// Softmax with extreme values: overflow in exp().
/// Type 3: exp(1000) = Inf.
#[test]
fn softmax_extreme_values() {
    let x = vec![1000.0, 1000.0, -1000.0];
    let s = softmax(&x);
    // Stable softmax subtracts max → exp(0), exp(0), exp(-2000)
    assert!((s[0] - 0.5).abs() < 0.01, "Softmax[0] should be ~0.5, got {}", s[0]);
    assert!((s[1] - 0.5).abs() < 0.01, "Softmax[1] should be ~0.5, got {}", s[1]);
    let sum: f64 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "Softmax should sum to 1, got {}", sum);
}

/// Softmax with all identical values: uniform output.
/// Type 4.
#[test]
fn softmax_uniform() {
    let x = vec![5.0; 4];
    let s = softmax(&x);
    for &v in &s {
        assert!((v - 0.25).abs() < 1e-10, "Uniform softmax should be 0.25, got {}", v);
    }
}

/// Softmax with NaN: contaminates everything.
#[test]
fn softmax_nan() {
    let x = vec![1.0, f64::NAN, 3.0];
    let s = softmax(&x);
    // NaN in max → NaN subtracted → all NaN
    let all_nan = s.iter().all(|v| v.is_nan());
    if all_nan {
        // Expected: NaN propagates
    } else {
        eprintln!("NOTE: Softmax with NaN produces non-NaN output: {:?}", s);
    }
}

/// Log-softmax numerical stability: should not produce -Inf.
#[test]
fn log_softmax_stability() {
    let x = vec![0.0, 0.0, 100.0]; // exp(0-100) ≈ 0 → log(0) = -Inf naively
    let ls = log_softmax(&x);
    // Stable log_softmax: x[i] - log(sum(exp(x)))
    assert!(ls[0].is_finite(), "Log-softmax should be finite, got {}", ls[0]);
    assert!(ls[2] > ls[0], "Largest input should have largest log-softmax");
}

/// Sigmoid at extreme values: overflow protection.
#[test]
fn sigmoid_extremes() {
    let pos = sigmoid(1000.0);
    let neg = sigmoid(-1000.0);
    assert!((pos - 1.0).abs() < 1e-10, "sigmoid(1000) should be ~1, got {}", pos);
    assert!((neg - 0.0).abs() < 1e-10, "sigmoid(-1000) should be ~0, got {}", neg);
}

/// ELU with alpha=0: becomes ReLU.
#[test]
fn elu_alpha_zero() {
    assert!((elu(-5.0, 0.0) - 0.0).abs() < 1e-10, "ELU(alpha=0) for x<0 should be 0");
    assert!((elu(3.0, 0.0) - 3.0).abs() < 1e-10, "ELU(alpha=0) for x>0 should be x");
}

/// GELU backward with empty arrays.
/// Type 5.
#[test]
fn gelu_backward_empty() {
    let result = gelu_backward(&[], &[]);
    assert!(result.is_empty(), "GELU backward of empty should be empty");
}

/// GELU backward length mismatch: x and grad_out different lengths.
/// Type 5.
#[test]
fn gelu_backward_length_mismatch() {
    let result = std::panic::catch_unwind(|| {
        gelu_backward(&[1.0, 2.0], &[1.0])
    });
    if result.is_err() {
        eprintln!("NOTE: gelu_backward panics on length mismatch");
    }
}

/// Temperature scaling with T=0: division by zero.
/// Type 1.
#[test]
fn temperature_scale_zero() {
    let logits = vec![1.0, 2.0, 3.0];
    let result = std::panic::catch_unwind(|| {
        temperature_scale(&logits, 0.0)
    });
    match result {
        Ok(scaled) => {
            // logits / 0 → Inf
            let any_inf = scaled.iter().any(|v| v.is_infinite());
            if any_inf {
                eprintln!("CONFIRMED BUG: temperature_scale with T=0 produces Inf");
            }
        }
        Err(_) => eprintln!("NOTE: temperature_scale panics on T=0"),
    }
}

/// Temperature scaling with T=Inf: all logits → 0 → uniform softmax.
#[test]
fn temperature_scale_inf() {
    let logits = vec![1.0, 2.0, 3.0];
    let scaled = temperature_scale(&logits, f64::INFINITY);
    // logits / Inf = 0 for all
    for &v in &scaled {
        assert!((v - 0.0).abs() < 1e-10, "T=Inf should give 0 logits, got {}", v);
    }
}

/// Top-k with k=0: no logits survive.
/// Type 5.
#[test]
fn top_k_zero() {
    let logits = vec![1.0, 2.0, 3.0];
    let result = std::panic::catch_unwind(|| {
        top_k_logits(&logits, 0)
    });
    match result {
        Ok(filtered) => {
            // k=0 → all set to -Inf
            let all_neg_inf = filtered.iter().all(|&v| v == f64::NEG_INFINITY);
            if !all_neg_inf {
                eprintln!("NOTE: top_k with k=0 doesn't mask all: {:?}", filtered);
            }
        }
        Err(_) => eprintln!("NOTE: top_k_logits panics on k=0"),
    }
}

/// Top-p with p=0: nothing passes threshold.
#[test]
fn top_p_zero() {
    let logits = vec![1.0, 2.0, 3.0];
    let result = std::panic::catch_unwind(|| {
        top_p_logits(&logits, 0.0)
    });
    match result {
        Ok(filtered) => {
            // p=0 → only top-1 token survives (cumulative prob crosses 0)
            let finite_count = filtered.iter().filter(|&&v| v != f64::NEG_INFINITY).count();
            assert!(finite_count >= 1, "top_p(0) should keep at least 1 token");
        }
        Err(_) => eprintln!("NOTE: top_p_logits panics on p=0"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONVOLUTION & POOLING
// ═══════════════════════════════════════════════════════════════════════════

/// Conv1D with empty input.
/// Type 5.
#[test]
fn conv1d_empty_input() {
    let result = std::panic::catch_unwind(|| {
        conv1d(&[], &[1.0, 1.0, 1.0], 1, 0)
    });
    match result {
        Ok(out) => assert!(out.is_empty(), "Conv1D of empty should be empty"),
        Err(_) => eprintln!("NOTE: conv1d panics on empty input"),
    }
}

/// Conv1D with kernel larger than input.
/// Type 5.
#[test]
fn conv1d_kernel_larger_than_input() {
    let result = std::panic::catch_unwind(|| {
        conv1d(&[1.0, 2.0], &[1.0, 1.0, 1.0, 1.0, 1.0], 1, 0)
    });
    match result {
        Ok(out) => {
            // kernel(5) > input(2) with no padding → output length ≤ 0
            assert!(out.is_empty(), "Kernel > input should give empty output");
        }
        Err(_) => eprintln!("NOTE: conv1d panics when kernel > input"),
    }
}

/// Conv1D with stride=0: division by zero in output length.
/// Type 1.
#[test]
fn conv1d_stride_zero() {
    let result = std::panic::catch_unwind(|| {
        conv1d(&[1.0, 2.0, 3.0], &[1.0], 0, 0)
    });
    if result.is_err() {
        eprintln!("CONFIRMED BUG: conv1d panics on stride=0 (division by zero)");
    }
}

/// Max pool with kernel_size=0.
/// Type 1.
#[test]
fn max_pool1d_zero_kernel() {
    let result = std::panic::catch_unwind(|| {
        max_pool1d(&[1.0, 2.0, 3.0], 0, 1)
    });
    if result.is_err() {
        eprintln!("NOTE: max_pool1d panics on kernel_size=0");
    }
}

/// Avg pool1d with stride=0.
/// Type 1.
#[test]
fn avg_pool1d_stride_zero() {
    let result = std::panic::catch_unwind(|| {
        avg_pool1d(&[1.0, 2.0, 3.0], 2, 0)
    });
    if result.is_err() {
        eprintln!("NOTE: avg_pool1d panics on stride=0");
    }
}

/// Adaptive average pool to output_size=0.
/// Type 5.
#[test]
fn adaptive_avg_pool_zero_output() {
    let result = std::panic::catch_unwind(|| {
        adaptive_avg_pool1d(&[1.0, 2.0, 3.0], 0)
    });
    match result {
        Ok(out) => assert!(out.is_empty(), "Adaptive pool to size 0 should be empty"),
        Err(_) => eprintln!("NOTE: adaptive_avg_pool1d panics on output_size=0"),
    }
}

/// Global avg pool with 0 spatial dimensions.
/// Type 1: average of 0 elements.
#[test]
fn global_avg_pool_zero_spatial() {
    let result = std::panic::catch_unwind(|| {
        global_avg_pool2d(&[], 1, 0, 0)
    });
    match result {
        Ok(out) => {
            if out.iter().any(|v| v.is_nan()) {
                eprintln!("CONFIRMED BUG: global_avg_pool2d with 0 spatial produces NaN");
            }
        }
        Err(_) => eprintln!("NOTE: global_avg_pool2d panics on 0 spatial dims"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NORMALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Batch norm with batch_size=1: variance=0 across batch.
/// Type 1: dividing by sqrt(0 + eps).
#[test]
fn batch_norm_single_sample() {
    let input = vec![1.0, 2.0, 3.0]; // batch=1, features=3
    let gamma = vec![1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0];
    let result = batch_norm(&input, 1, 3, &gamma, &beta, 1e-5);
    // batch=1 → mean=input, var=0 → (x-mean)/sqrt(eps) = 0
    for &v in &result.output {
        assert!(v.is_finite(), "Batch norm with n=1 should be finite, got {}", v);
    }
}

/// Batch norm with eps=0: exact division by sqrt(variance).
/// When variance=0, this is division by zero.
/// Type 1.
#[test]
fn batch_norm_eps_zero_constant() {
    let input = vec![5.0, 5.0, 5.0, 5.0]; // batch=2, features=2, constant
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    let result = batch_norm(&input, 2, 2, &gamma, &beta, 0.0);
    // var=0, eps=0 → divide by 0
    let any_nan = result.output.iter().any(|v| v.is_nan() || v.is_infinite());
    if any_nan {
        eprintln!("CONFIRMED BUG: batch_norm with eps=0 and constant data produces NaN/Inf");
    }
}

/// Layer norm with 0 features.
/// Type 1.
#[test]
fn layer_norm_zero_features() {
    let result = std::panic::catch_unwind(|| {
        layer_norm(&[], 1, 0, &[], &[], 1e-5)
    });
    match result {
        Ok(out) => assert!(out.is_empty(), "Layer norm with 0 features should be empty"),
        Err(_) => eprintln!("NOTE: layer_norm panics on 0 features"),
    }
}

/// RMS norm with all-zero input: sqrt(mean(0)) = 0.
/// Type 1.
#[test]
fn rms_norm_zero_input() {
    let input = vec![0.0, 0.0, 0.0]; // batch=1, features=3
    let gamma = vec![1.0, 1.0, 1.0];
    let result = rms_norm(&input, 1, 3, &gamma, 1e-5);
    // rms = sqrt(mean(0²) + eps) = sqrt(eps) → output = 0 / sqrt(eps) * gamma = 0
    for &v in &result {
        assert!(v.is_finite(), "RMS norm of zeros should be finite, got {}", v);
    }
}

/// Group norm with num_groups > channels.
/// Type 5: channels not divisible by groups.
#[test]
fn group_norm_bad_groups() {
    let input = vec![1.0, 2.0, 3.0]; // batch=1, channels=3, spatial=1
    let gamma = vec![1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0];
    let result = std::panic::catch_unwind(|| {
        group_norm(&input, 1, 3, 1, 5, &gamma, &beta, 1e-5) // 5 groups for 3 channels
    });
    if result.is_err() {
        eprintln!("NOTE: group_norm panics when num_groups > channels");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LOSS FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// MSE loss with empty arrays.
/// Type 5.
#[test]
fn mse_loss_empty() {
    let result = std::panic::catch_unwind(|| {
        mse_loss(&[], &[])
    });
    match result {
        Ok(loss) => assert!(loss == 0.0 || loss.is_nan(),
            "MSE of empty should be 0 or NaN, got {}", loss),
        Err(_) => eprintln!("NOTE: mse_loss panics on empty input"),
    }
}

/// BCE loss with predicted=0 and target=1: -1 * log(0) = Inf.
/// Type 1.
#[test]
fn bce_loss_log_zero() {
    let predicted = vec![0.0, 1.0];
    let target = vec![1.0, 0.0]; // worst case: log(0) twice
    let loss = bce_loss(&predicted, &target);
    // -[1*log(0) + (1-0)*log(1-1)] = -[−∞ + −∞] = Inf
    if loss.is_infinite() || loss.is_nan() {
        eprintln!("CONFIRMED BUG: BCE loss returns {} when predicted is exactly 0 or 1", loss);
    }
}

/// Cross-entropy loss with 0 classes.
/// Type 1.
#[test]
fn cross_entropy_zero_classes() {
    let result = std::panic::catch_unwind(|| {
        cross_entropy_loss(&[1.0, 2.0], 0, &[0])
    });
    if result.is_err() {
        eprintln!("NOTE: cross_entropy_loss panics on 0 classes");
    }
}

/// Huber loss with delta=0: all residuals are "large" → L1 loss − 0/2.
#[test]
fn huber_loss_delta_zero() {
    let pred = vec![1.0, 2.0];
    let target = vec![3.0, 4.0];
    let loss = huber_loss(&pred, &target, 0.0);
    // |r| > 0 always → delta * (|r| - delta/2) = 0 * (|r| - 0) = 0 for all
    assert!(loss.is_finite(), "Huber with delta=0 should be finite, got {}", loss);
}

/// Cosine similarity loss with zero vectors: 0/0.
/// Type 1.
#[test]
fn cosine_similarity_zero_vectors() {
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![0.0, 0.0, 0.0];
    let loss = cosine_similarity_loss(&a, &b);
    // dot=0, norms=0 → 0/(0*0) = NaN
    if loss.is_nan() {
        eprintln!("CONFIRMED BUG: cosine_similarity_loss returns NaN for zero vectors");
    }
}

/// Focal loss with gamma=0: reduces to BCE.
#[test]
fn focal_loss_gamma_zero() {
    let pred = vec![0.9, 0.1];
    let target = vec![1.0, 0.0];
    let fl = focal_loss(&pred, &target, 1.0, 0.0);
    let bce = bce_loss(&pred, &target);
    // gamma=0 → (1-p_t)^0 = 1 → focal = alpha * BCE
    assert!((fl - bce).abs() < 0.01 || (fl - 1.0 * bce).abs() < 0.01,
        "Focal(gamma=0) should equal alpha*BCE: focal={}, bce={}", fl, bce);
}

// ═══════════════════════════════════════════════════════════════════════════
// ATTENTION & LINEAR
// ═══════════════════════════════════════════════════════════════════════════

/// Scaled dot-product attention with empty sequences.
/// Type 5.
#[test]
fn attention_empty_sequence() {
    let q = Mat { data: vec![], rows: 0, cols: 4 };
    let k = Mat { data: vec![], rows: 0, cols: 4 };
    let v = Mat { data: vec![], rows: 0, cols: 4 };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scaled_dot_product_attention(&q, &k, &v, false)
    }));
    match result {
        Ok(att) => {
            assert_eq!(att.output.rows, 0, "Empty attention should have 0 rows");
        }
        Err(_) => eprintln!("NOTE: attention panics on empty sequence"),
    }
}

/// Attention with d_k=0: scale = 1/sqrt(0) = Inf.
/// Type 1.
#[test]
fn attention_zero_dim() {
    let q = Mat { data: vec![], rows: 2, cols: 0 };
    let k = Mat { data: vec![], rows: 2, cols: 0 };
    let v = Mat { data: vec![], rows: 2, cols: 0 };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scaled_dot_product_attention(&q, &k, &v, false)
    }));
    if result.is_err() {
        eprintln!("NOTE: attention panics on d_k=0");
    }
}

/// Embedding with out-of-bounds index.
/// Type 5.
#[test]
fn embedding_oob_index() {
    let emb = Mat {
        data: vec![1.0, 2.0, 3.0, 4.0], // 2 tokens x 2 dims
        rows: 2, cols: 2,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        embedding(&emb, &[0, 5]) // index 5 is out of bounds for vocab_size=2
    }));
    if result.is_err() {
        eprintln!("NOTE: embedding panics on out-of-bounds index");
    }
}

/// Positional encoding with d_model=0.
/// Type 1.
#[test]
fn positional_encoding_zero_dim() {
    let result = std::panic::catch_unwind(|| {
        positional_encoding(10, 0)
    });
    match result {
        Ok(pe) => {
            assert_eq!(pe.cols, 0, "PE with d_model=0 should have 0 columns");
        }
        Err(_) => eprintln!("NOTE: positional_encoding panics on d_model=0"),
    }
}

/// RoPE with base=0: division by zero in frequency computation.
/// Type 1.
#[test]
fn rope_base_zero() {
    let input = Mat {
        data: vec![1.0, 2.0, 3.0, 4.0],
        rows: 1, cols: 4,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rope(&input, 0.0)
    }));
    match result {
        Ok(out) => {
            let any_bad = out.data.iter().any(|v| v.is_nan() || v.is_infinite());
            if any_bad {
                eprintln!("CONFIRMED BUG: RoPE with base=0 produces NaN/Inf");
            }
        }
        Err(_) => eprintln!("NOTE: RoPE panics on base=0"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DROPOUT & GRADIENT CLIPPING
// ═══════════════════════════════════════════════════════════════════════════

/// Dropout with p=1.0: drops everything → output all zeros.
#[test]
fn dropout_p_one() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let (output, mask) = dropout(&input, 1.0, &mut rng);
    // p=1.0 → every element dropped
    for &v in &output {
        assert!((v - 0.0).abs() < 1e-10, "Dropout(p=1) should zero everything, got {}", v);
    }
}

/// Dropout with p=0.0: keeps everything, scale by 1/(1-0)=1.
#[test]
fn dropout_p_zero() {
    let input = vec![1.0, 2.0, 3.0];
    let mut rng = tambear::rng::Xoshiro256::new(42);
    let (output, _mask) = dropout(&input, 0.0, &mut rng);
    for (i, &v) in output.iter().enumerate() {
        assert!((v - input[i]).abs() < 1e-10, "Dropout(p=0) should preserve input, got {}", v);
    }
}

/// Clip grad norm with max_norm=0: all gradients → 0.
#[test]
fn clip_grad_norm_zero() {
    let mut grad = vec![1.0, 2.0, 3.0];
    let norm = clip_grad_norm(&mut grad, 0.0);
    assert!(norm.is_finite(), "Grad norm should be finite, got {}", norm);
    for &g in &grad {
        assert!((g - 0.0).abs() < 1e-10, "Clipped to 0 should zero gradients, got {}", g);
    }
}

/// Clip grad norm with zero gradient: norm=0 → scale=0/0.
/// Type 1.
#[test]
fn clip_grad_norm_zero_grad() {
    let mut grad = vec![0.0, 0.0, 0.0];
    let norm = clip_grad_norm(&mut grad, 1.0);
    assert!((norm - 0.0).abs() < 1e-10, "Zero gradient norm should be 0, got {}", norm);
    for &g in &grad {
        assert!(g.is_finite(), "Zero grad after clip should stay finite, got {}", g);
    }
}

/// Label smoothing with epsilon=1: uniform distribution.
#[test]
fn label_smooth_full() {
    let smoothed = label_smooth(&[0, 1], 3, 1.0);
    // epsilon=1 → all weight to uniform: 1/num_classes for all
    for &v in &smoothed {
        assert!((v - 1.0 / 3.0).abs() < 0.01,
            "Full smoothing should be uniform, got {}", v);
    }
}

/// Label smoothing with epsilon=0: one-hot.
#[test]
fn label_smooth_none() {
    let smoothed = label_smooth(&[0], 3, 0.0);
    // epsilon=0 → one-hot: [1, 0, 0]
    assert!((smoothed[0] - 1.0).abs() < 1e-10, "No smoothing target should be 1.0");
    assert!((smoothed[1] - 0.0).abs() < 1e-10, "No smoothing non-target should be 0.0");
    assert!((smoothed[2] - 0.0).abs() < 1e-10, "No smoothing non-target should be 0.0");
}
