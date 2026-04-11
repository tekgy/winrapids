//! Recipe: the accumulate+gather decomposition of any primitive.
//!
//! A recipe is a sequence of steps that expresses a mathematical
//! computation as accumulate+gather operations. The recipe is
//! backend-agnostic — TAM decides which ALU runs it (CPU, GPU,
//! NPU, any vendor, any OS, any architecture).
//!
//! The recipe IS the primitive's contract with TAM. Two primitives
//! with identical accumulate steps share that step automatically.
//! The recipe IS the sharing specification.
//!
//! # Architecture
//!
//! ```text
//! tambear-primitives: declares recipes (this crate)
//!           ↓
//! tambear-tbs: compiles TBS scripts to recipes
//!           ↓
//! tambear-tam: reads recipes, fuses steps, dispatches to ALU
//!           ↓
//! tam-gpu / tambear-wgpu: vendor doors (CUDA, Vulkan, Metal, CPU)
//! ```

/// A single step in a computation recipe.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// Accumulate: scatter/reduce values into accumulators.
    ///
    /// This is the fundamental "write" operation. Every accumulate
    /// with the same (grouping, expr, op) on the same data is
    /// automatically shared — computed once, reused by all consumers.
    Accumulate {
        /// How to partition: All, ByKey, Prefix, Segmented, Windowed, Tiled.
        grouping: GroupingKind,
        /// What to compute per element before combining.
        expr: ExprKind,
        /// How to combine accumulators.
        op: OpKind,
        /// Name for the output (used by subsequent steps to reference this result).
        output: &'static str,
    },

    /// Gather: read from accumulated results using an addressing pattern.
    ///
    /// This is the fundamental "read" operation. Gathers can reference
    /// outputs from prior Accumulate steps by name.
    Gather {
        /// Expression over named outputs. e.g. "sum / count"
        expr: &'static str,
        /// Name for the output.
        output: &'static str,
    },

    /// Element-wise transform: apply a function to each element.
    /// Fuses into the expr of a subsequent Accumulate when possible.
    Transform {
        /// The transform: "ln", "exp", "sqrt", "abs", "reciprocal", etc.
        func: &'static str,
        /// Input (name of a prior step's output, or "input" for raw data).
        input: &'static str,
        /// Name for the output.
        output: &'static str,
    },
}

/// How to partition data for accumulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupingKind {
    /// N → 1: all elements into one accumulator.
    All,
    /// N → K: scatter by key column.
    ByKey,
    /// N → N: prefix (forward scan).
    Prefix,
    /// N → N: prefix with segment resets.
    Segmented,
    /// N → N: rolling window.
    Windowed,
    /// M×K × K×N → M×N: blocked matrix accumulation.
    Tiled,
    /// N → N: scatter by graph adjacency.
    Graph,
}

/// What to compute per element before combining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExprKind {
    /// Raw value: v
    Value,
    /// Squared: v * v
    ValueSq,
    /// Constant 1 (for counting)
    One,
    /// v * reference (for cross-products)
    CrossRef,
    /// ln(v)
    Ln,
    /// 1/v
    Reciprocal,
    /// v^p (needs parameter)
    Pow,
    /// |v - ref|  (for deviations)
    AbsDev,
    /// (v - ref)^2  (for squared deviations)
    SqDev,
}

/// How to combine accumulators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Real addition. The default.
    Add,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
    /// Argmin (value + index).
    ArgMin,
    /// Argmax (value + index).
    ArgMax,
    /// Matrix dot product.
    DotProduct,
    /// L2-squared distance.
    Distance,
    /// Scan over a named semiring.
    Semiring(&'static str),
}

/// A complete recipe: the decomposition of one primitive.
#[derive(Debug, Clone)]
pub struct Recipe {
    /// Name of the primitive this recipe computes.
    pub name: &'static str,
    /// The steps, in dependency order.
    pub steps: &'static [Step],
    /// Which step's output is the final result.
    pub result: &'static str,
}

// ═══════════════════════════════════════════════════════════════════
// Standard recipes for built-in primitives
// ═══════════════════════════════════════════════════════════════════

/// Recipe for mean_arithmetic: sum / count.
pub const MEAN_ARITHMETIC: Recipe = Recipe {
    name: "mean_arithmetic",
    steps: &[
        Step::Accumulate {
            grouping: GroupingKind::All,
            expr: ExprKind::Value,
            op: OpKind::Add,
            output: "sum",
        },
        Step::Accumulate {
            grouping: GroupingKind::All,
            expr: ExprKind::One,
            op: OpKind::Add,
            output: "count",
        },
        Step::Gather {
            expr: "sum / count",
            output: "mean",
        },
    ],
    result: "mean",
};

/// Recipe for mean_geometric: exp(sum(ln(x)) / count).
pub const MEAN_GEOMETRIC: Recipe = Recipe {
    name: "mean_geometric",
    steps: &[
        Step::Accumulate {
            grouping: GroupingKind::All,
            expr: ExprKind::Ln,
            op: OpKind::Add,
            output: "log_sum",
        },
        Step::Accumulate {
            grouping: GroupingKind::All,
            expr: ExprKind::One,
            op: OpKind::Add,
            output: "count",
        },
        Step::Gather {
            expr: "exp(log_sum / count)",
            output: "geometric_mean",
        },
    ],
    result: "geometric_mean",
};

/// Recipe for mean_harmonic: count / sum(1/x).
pub const MEAN_HARMONIC: Recipe = Recipe {
    name: "mean_harmonic",
    steps: &[
        Step::Accumulate {
            grouping: GroupingKind::All,
            expr: ExprKind::Reciprocal,
            op: OpKind::Add,
            output: "reciprocal_sum",
        },
        Step::Accumulate {
            grouping: GroupingKind::All,
            expr: ExprKind::One,
            op: OpKind::Add,
            output: "count",
        },
        Step::Gather {
            expr: "count / reciprocal_sum",
            output: "harmonic_mean",
        },
    ],
    result: "harmonic_mean",
};

/// Recipe for mean_quadratic (RMS): sqrt(sum(x²) / count).
pub const MEAN_QUADRATIC: Recipe = Recipe {
    name: "mean_quadratic",
    steps: &[
        Step::Accumulate {
            grouping: GroupingKind::All,
            expr: ExprKind::ValueSq,
            op: OpKind::Add,
            output: "sum_sq",
        },
        Step::Accumulate {
            grouping: GroupingKind::All,
            expr: ExprKind::One,
            op: OpKind::Add,
            output: "count",
        },
        Step::Gather {
            expr: "sqrt(sum_sq / count)",
            output: "rms",
        },
    ],
    result: "rms",
};

/// Recipe for variance: sum((x - mean)²) / (count - 1).
/// Shows sharing: reuses the "sum" and "count" from mean_arithmetic.
pub const VARIANCE: Recipe = Recipe {
    name: "variance",
    steps: &[
        Step::Accumulate {
            grouping: GroupingKind::All,
            expr: ExprKind::Value,
            op: OpKind::Add,
            output: "sum",
        },
        Step::Accumulate {
            grouping: GroupingKind::All,
            expr: ExprKind::ValueSq,
            op: OpKind::Add,
            output: "sum_sq",
        },
        Step::Accumulate {
            grouping: GroupingKind::All,
            expr: ExprKind::One,
            op: OpKind::Add,
            output: "count",
        },
        Step::Gather {
            expr: "(sum_sq - sum * sum / count) / (count - 1)",
            output: "variance",
        },
    ],
    result: "variance",
};

/// Recipe for cumulative sum: prefix scan over addition.
pub const CUMSUM: Recipe = Recipe {
    name: "cumsum",
    steps: &[
        Step::Accumulate {
            grouping: GroupingKind::Prefix,
            expr: ExprKind::Value,
            op: OpKind::Add,
            output: "cumsum",
        },
    ],
    result: "cumsum",
};

// ═══════════════════════════════════════════════════════════════════
// Sharing analysis
// ═══════════════════════════════════════════════════════════════════

impl Recipe {
    /// Extract all accumulate steps (the shareable intermediates).
    pub fn accumulate_steps(&self) -> Vec<&Step> {
        self.steps.iter().filter(|s| matches!(s, Step::Accumulate { .. })).collect()
    }

    /// Find shared accumulate steps between two recipes.
    /// Two accumulate steps are shared if they have the same (grouping, expr, op).
    pub fn shared_with(&self, other: &Recipe) -> usize {
        let mut count = 0;
        for a in self.accumulate_steps() {
            for b in other.accumulate_steps() {
                if let (
                    Step::Accumulate { grouping: g1, expr: e1, op: o1, .. },
                    Step::Accumulate { grouping: g2, expr: e2, op: o2, .. },
                ) = (a, b) {
                    if g1 == g2 && e1 == e2 && o1 == o2 {
                        count += 1;
                    }
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests;
