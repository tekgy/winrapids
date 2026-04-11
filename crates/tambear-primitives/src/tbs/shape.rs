//! Shape signatures: data dimensions flowing through the pipeline.
//!
//! Every Expr, Accumulate, and Gather has a shape signature:
//! what shape goes in, what shape comes out. TAM uses shapes to:
//! 1. Verify recipe chains are consistent (type checking)
//! 2. Determine which slots can fuse (same input shape)
//! 3. Allocate output buffers (known output size)
//! 4. Generate kernel grid dimensions
//!
//! Shapes are the "types" of the accumulate+gather system.

/// A data shape — dimensions of the input or output.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Shape {
    /// A single scalar value. ℝ.
    Scalar,
    /// A 1D vector of length n. ℝⁿ.
    Vector(Dim),
    /// A 2D matrix of shape m × n. ℝᵐˣⁿ.
    Matrix(Dim, Dim),
    /// A 3D tensor. ℝᵈ¹ˣᵈ²ˣᵈ³.
    Tensor3(Dim, Dim, Dim),
    /// General N-dimensional. ℝᵈ¹ˣ...ˣᵈᴺ.
    NdArray(Vec<Dim>),
}

/// A dimension — can be known at compile time or determined at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    /// Known at recipe definition time.
    Known(usize),
    /// Determined by the data at runtime. Named for tracking.
    Runtime(String),
    /// Same as input dimension (passthrough).
    Same,
}

impl Dim {
    pub fn n() -> Self { Dim::Runtime("n".into()) }
    pub fn m() -> Self { Dim::Runtime("m".into()) }
    pub fn k() -> Self { Dim::Runtime("k".into()) }
    pub fn known(val: usize) -> Self { Dim::Known(val) }
}

/// Shape signature for an operation: input shape → output shape.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeSig {
    pub input: Vec<Shape>,
    pub output: Shape,
}

impl ShapeSig {
    /// Scalar → Scalar (element-wise transform, or scalar gather)
    pub fn scalar_to_scalar() -> Self {
        Self { input: vec![Shape::Scalar], output: Shape::Scalar }
    }

    /// ℝⁿ → ℝ (All reduction)
    pub fn vector_to_scalar() -> Self {
        Self {
            input: vec![Shape::Vector(Dim::n())],
            output: Shape::Scalar,
        }
    }

    /// ℝⁿ → ℝⁿ (Prefix scan, element-wise)
    pub fn vector_to_vector() -> Self {
        Self {
            input: vec![Shape::Vector(Dim::n())],
            output: Shape::Vector(Dim::Same),
        }
    }

    /// ℝⁿ → ℝᵏ (ByKey reduction, k groups)
    pub fn vector_to_groups() -> Self {
        Self {
            input: vec![Shape::Vector(Dim::n())],
            output: Shape::Vector(Dim::k()),
        }
    }

    /// ℝᵐˣᵏ × ℝᵏˣⁿ → ℝᵐˣⁿ (Tiled matrix multiply)
    pub fn matmul() -> Self {
        Self {
            input: vec![
                Shape::Matrix(Dim::m(), Dim::k()),
                Shape::Matrix(Dim::k(), Dim::n()),
            ],
            output: Shape::Matrix(Dim::m(), Dim::n()),
        }
    }

    /// ℝⁿ × ℝⁿ → ℝ (two-column reduction: covariance, correlation)
    pub fn two_vectors_to_scalar() -> Self {
        Self {
            input: vec![Shape::Vector(Dim::n()), Shape::Vector(Dim::n())],
            output: Shape::Scalar,
        }
    }

    /// ℝⁿ → ℝⁿˣⁿ (pairwise distance matrix)
    pub fn vector_to_square_matrix() -> Self {
        Self {
            input: vec![Shape::Vector(Dim::n())],
            output: Shape::Matrix(Dim::n(), Dim::n()),
        }
    }
}

/// Shape signature for each Grouping + Op combination.
pub fn grouping_shape(grouping: &crate::accumulates::Grouping) -> ShapeSig {
    use crate::accumulates::Grouping;
    match grouping {
        Grouping::All       => ShapeSig::vector_to_scalar(),
        Grouping::ByKey     => ShapeSig::vector_to_groups(),
        Grouping::Prefix    => ShapeSig::vector_to_vector(),
        Grouping::Segmented => ShapeSig::vector_to_vector(),
        Grouping::Windowed  => ShapeSig::vector_to_vector(),
        Grouping::Tiled     => ShapeSig::matmul(),
        Grouping::Graph     => ShapeSig::vector_to_vector(),
    }
}

/// Shape of an Expr transform: element-wise, so scalar → scalar.
/// Binary transforms (MulPair etc.) take two scalars.
pub fn expr_shape(expr: &crate::tbs::Expr) -> ShapeSig {
    use crate::tbs::Expr;
    match expr {
        // Binary: two inputs
        Expr::Mul(_, _) | Expr::Add(_, _) | Expr::Sub(_, _) |
        Expr::Div(_, _) | Expr::Pow(_, _) | Expr::Min(_, _) |
        Expr::Max(_, _) | Expr::Mod(_, _) | Expr::Atan2(_, _) |
        Expr::Gt(_, _) | Expr::Lt(_, _) | Expr::Eq(_, _) => {
            ShapeSig {
                input: vec![Shape::Scalar, Shape::Scalar],
                output: Shape::Scalar,
            }
        }
        // Ternary
        Expr::Clamp(_, _, _) | Expr::If(_, _, _) => {
            ShapeSig {
                input: vec![Shape::Scalar, Shape::Scalar, Shape::Scalar],
                output: Shape::Scalar,
            }
        }
        // Everything else: unary scalar → scalar
        _ => ShapeSig::scalar_to_scalar(),
    }
}

/// Render a Shape as formal math notation.
pub fn shape_notation(shape: &Shape) -> String {
    match shape {
        Shape::Scalar => "ℝ".into(),
        Shape::Vector(d) => format!("ℝ{}", dim_super(d)),
        Shape::Matrix(m, n) => format!("ℝ{}×{}", dim_super(m), dim_super(n)),
        Shape::Tensor3(a, b, c) => format!("ℝ{}×{}×{}", dim_super(a), dim_super(b), dim_super(c)),
        Shape::NdArray(dims) => {
            let parts: Vec<String> = dims.iter().map(dim_super).collect();
            format!("ℝ{}", parts.join("×"))
        }
    }
}

fn dim_super(d: &Dim) -> String {
    match d {
        Dim::Known(v) => format!("{v}"),
        Dim::Runtime(name) => format!("{name}"),
        Dim::Same => "·".into(),
    }
}

/// Render a ShapeSig as a typed function signature.
/// e.g. "ℝⁿ → ℝ" or "ℝᵐˣᵏ × ℝᵏˣⁿ → ℝᵐˣⁿ"
pub fn sig_notation(sig: &ShapeSig) -> String {
    let inputs: Vec<String> = sig.input.iter().map(shape_notation).collect();
    let output = shape_notation(&sig.output);
    format!("{} → {}", inputs.join(" × "), output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_notation() {
        assert_eq!(shape_notation(&Shape::Scalar), "ℝ");
    }

    #[test]
    fn vector_notation() {
        assert_eq!(shape_notation(&Shape::Vector(Dim::n())), "ℝn");
    }

    #[test]
    fn matrix_notation() {
        assert_eq!(shape_notation(&Shape::Matrix(Dim::m(), Dim::n())), "ℝm×n");
    }

    #[test]
    fn reduction_sig() {
        let sig = ShapeSig::vector_to_scalar();
        assert_eq!(sig_notation(&sig), "ℝn → ℝ");
    }

    #[test]
    fn scan_sig() {
        let sig = ShapeSig::vector_to_vector();
        assert_eq!(sig_notation(&sig), "ℝn → ℝ·");
    }

    #[test]
    fn matmul_sig() {
        let sig = ShapeSig::matmul();
        assert_eq!(sig_notation(&sig), "ℝm×k × ℝk×n → ℝm×n");
    }

    #[test]
    fn two_column_sig() {
        let sig = ShapeSig::two_vectors_to_scalar();
        assert_eq!(sig_notation(&sig), "ℝn × ℝn → ℝ");
    }

    #[test]
    fn grouping_all_is_reduction() {
        use crate::accumulates::Grouping;
        let sig = grouping_shape(&Grouping::All);
        assert_eq!(sig.output, Shape::Scalar);
    }

    #[test]
    fn grouping_prefix_preserves_length() {
        use crate::accumulates::Grouping;
        let sig = grouping_shape(&Grouping::Prefix);
        assert_eq!(sig.output, Shape::Vector(Dim::Same));
    }

    #[test]
    fn grouping_tiled_is_matmul() {
        use crate::accumulates::Grouping;
        let sig = grouping_shape(&Grouping::Tiled);
        assert_eq!(sig.output, Shape::Matrix(Dim::m(), Dim::n()));
    }
}
