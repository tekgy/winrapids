//! Tiled accumulation operators.
//!
//! Each operator defines how a tile of A and a tile of B are combined
//! into an accumulator. The key insight: this is the SAME trait pattern
//! as scan's AssociativeOp, but for 2D blocked computation.
//!
//! The operator defines:
//! - An accumulator type (what accumulates per output tile)
//! - An identity element (zero accumulator)
//! - An accumulate function (how one A×B tile pair contributes)
//! - An extract function (accumulator → output value)
//! - Optional: a pre-transform on inputs (centering, normalization)
//!
//! The pre-transform is the fusion advantage over cuBLASLt:
//! we read data once and transform on the fly, instead of
//! materializing centered/normalized intermediates.
//!
//! CSE identity: (A_id, B_id, op_name, op_params) — block-level,
//! not N² per element.

/// The tiled accumulation trait. Implement this for your operator,
/// and the tiled engine generates a CUDA kernel that tiles the
/// computation automatically.
///
/// This is the 2D analog of scan's AssociativeOp.
pub trait TiledOp: Send + Sync {
    /// Name for kernel cache key differentiation.
    fn name(&self) -> &'static str;

    /// CUDA type declaration for the accumulator.
    /// e.g., "double" for DotProduct, "struct { double max; double sum; }" for SoftmaxWeighted.
    fn cuda_acc_type(&self) -> String;

    /// CUDA expression for the identity (zero) accumulator.
    fn cuda_identity(&self) -> String;

    /// CUDA function body that accumulates one element pair.
    ///
    /// Variables in scope:
    /// - `acc`: the current accumulator (mutable)
    /// - `a_val`: element from A tile (after pre-transform)
    /// - `b_val`: element from B tile (after pre-transform)
    ///
    /// Must modify `acc` in-place.
    fn cuda_accumulate_body(&self) -> String;

    /// CUDA expression that extracts the output from the final accumulator.
    /// Variable `acc` is in scope.
    fn cuda_extract(&self) -> String;

    /// Optional CUDA expression to pre-transform A elements before accumulation.
    /// Variable `x` is in scope (raw element value).
    /// Default: no transform (identity).
    fn cuda_pre_transform_a(&self) -> String {
        "x".into()
    }

    /// Optional CUDA expression to pre-transform B elements before accumulation.
    /// Variable `x` is in scope (raw element value).
    /// Default: no transform (identity).
    fn cuda_pre_transform_b(&self) -> String {
        "x".into()
    }

    /// Size of the accumulator type in bytes.
    fn acc_byte_size(&self) -> usize { 8 }

    /// Parameters that affect the kernel (for cache key uniqueness).
    fn params_key(&self) -> String { String::new() }

    /// Whether A should be transposed before tiling.
    fn transpose_a(&self) -> bool { false }

    /// Whether B should be transposed before tiling.
    fn transpose_b(&self) -> bool { false }

    // -----------------------------------------------------------------------
    // WGSL variants (default: adapt from CUDA methods)
    // -----------------------------------------------------------------------

    /// WGSL type for the accumulator. Defaults to `f32` (WGSL doesn't have f64).
    ///
    /// Override for struct accumulators (e.g. SoftmaxWeighted).
    fn wgsl_acc_type(&self) -> String {
        "f32".into()
    }

    /// WGSL expression for the identity (zero) accumulator.
    fn wgsl_identity(&self) -> String {
        self.cuda_identity()
    }

    /// WGSL function body that accumulates one element pair (f32).
    ///
    /// Default: translate the CUDA body by replacing `double ` with `let `.
    /// `a_val`, `b_val`, and `acc` are in scope (all `f32`).
    ///
    /// Override when the CUDA body uses C-specific constructs.
    fn wgsl_accumulate_body(&self) -> String {
        self.cuda_accumulate_body()
            .replace("double ", "var ")
    }

    /// WGSL expression that extracts the output from the accumulator.
    fn wgsl_extract(&self) -> String {
        self.cuda_extract()
    }
}

// ============================================================
// DotProductOp — standard GEMM: C[i,j] = sum(A[i,k] * B[k,j])
// ============================================================

pub struct DotProductOp;

impl TiledOp for DotProductOp {
    fn name(&self) -> &'static str { "dot_product" }
    fn cuda_acc_type(&self) -> String { "double".into() }
    fn cuda_identity(&self) -> String { "0.0".into() }
    fn cuda_accumulate_body(&self) -> String {
        "    acc += a_val * b_val;".into()
    }
    fn cuda_extract(&self) -> String { "acc".into() }
}

// ============================================================
// OuterProductOp — C[i,j] = A[i] * B[j] (rank-1 update)
// ============================================================

pub struct OuterProductOp;

impl TiledOp for OuterProductOp {
    fn name(&self) -> &'static str { "outer_product" }
    fn cuda_acc_type(&self) -> String { "double".into() }
    fn cuda_identity(&self) -> String { "0.0".into() }
    fn cuda_accumulate_body(&self) -> String {
        "    acc += a_val * b_val;".into()
    }
    fn cuda_extract(&self) -> String { "acc".into() }
}

// ============================================================
// CovarianceOp — C[i,j] = sum((A[i,k]-mean_i) * (B[j,k]-mean_j)) / (n-1)
//
// The pre-transform fuses centering into the accumulation:
// data is read ONCE, centered on the fly. cuBLASLt would need
// a separate centering pass that materializes N×D intermediate.
// ============================================================

pub struct CovarianceOp {
    /// Number of columns (for normalization in extract).
    pub n_cols: usize,
    /// Row means for A (precomputed via reduce primitive).
    /// Used in pre-transform. Empty = no centering (raw covariance).
    pub mean_a_expr: String,
    /// Row means for B. Empty = same as A (auto-covariance).
    pub mean_b_expr: String,
}

impl TiledOp for CovarianceOp {
    fn name(&self) -> &'static str { "covariance" }
    fn cuda_acc_type(&self) -> String { "double".into() }
    fn cuda_identity(&self) -> String { "0.0".into() }
    fn cuda_accumulate_body(&self) -> String {
        "    acc += a_val * b_val;".into()
    }
    fn cuda_extract(&self) -> String {
        format!("(acc / (double)({} - 1))", self.n_cols)
    }
    fn cuda_pre_transform_a(&self) -> String {
        if self.mean_a_expr.is_empty() {
            "x".into()
        } else {
            format!("(x - {})", self.mean_a_expr)
        }
    }
    fn cuda_pre_transform_b(&self) -> String {
        if self.mean_b_expr.is_empty() {
            if self.mean_a_expr.is_empty() {
                "x".into()
            } else {
                format!("(x - {})", self.mean_a_expr)
            }
        } else {
            format!("(x - {})", self.mean_b_expr)
        }
    }
    fn params_key(&self) -> String {
        format!("n_cols={}", self.n_cols)
    }
}

// ============================================================
// DistanceOp — C[i,j] = sum((A[i,k] - B[j,k])^2)
//
// KNN distance matrix. Pre-transform is identity — the
// squaring is in the accumulate step.
// ============================================================

pub struct DistanceOp;

impl TiledOp for DistanceOp {
    fn name(&self) -> &'static str { "l2_distance" }
    fn cuda_acc_type(&self) -> String { "double".into() }
    fn cuda_identity(&self) -> String { "0.0".into() }
    fn cuda_accumulate_body(&self) -> String {
        "    double diff = a_val - b_val;\n    acc += diff * diff;".into()
    }
    fn cuda_extract(&self) -> String { "acc".into() }
}

// ============================================================
// SoftmaxWeightedOp — FlashAttention pattern (one pass)
//
// Computes softmax(A) @ B: output[i,j] = sum_k softmax(A[i,k]) * B[k,j].
// This is ONE tiled pass of the two-pass FlashAttention pipeline.
// Full FlashAttention fuses Q·K^T + softmax + ·V in one pass,
// avoiding materializing the N×N score matrix. That requires a
// three-input tiled reduction (Q, K, V) — a future specialist
// that composes this op with DotProductOp.
//
// acc = { max_so_far, exp_sum, weighted_sum }
// This is the online softmax: carry (max, denominator, numerator).
// Same algebraic structure as Welford's online algorithm.
// ============================================================

pub struct SoftmaxWeightedOp;

impl TiledOp for SoftmaxWeightedOp {
    fn name(&self) -> &'static str { "softmax_weighted" }

    fn cuda_acc_type(&self) -> String {
        r#"struct SoftmaxAcc { double max_val; double exp_sum; double weighted_sum; }"#.into()
    }

    fn cuda_identity(&self) -> String {
        "{-1e308, 0.0, 0.0}".into()
    }

    fn cuda_accumulate_body(&self) -> String {
        r#"    double score = a_val;  // attention score
    if (score > acc.max_val) {
        double scale = exp(acc.max_val - score);
        acc.exp_sum *= scale;
        acc.weighted_sum *= scale;
        acc.max_val = score;
    }
    double w = exp(score - acc.max_val);
    acc.exp_sum += w;
    acc.weighted_sum += w * b_val;"#.into()
    }

    fn cuda_extract(&self) -> String {
        "(acc.exp_sum > 0.0 ? acc.weighted_sum / acc.exp_sum : 0.0)".into()
    }

    fn acc_byte_size(&self) -> usize { 24 }
}
