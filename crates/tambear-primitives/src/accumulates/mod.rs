//! Accumulates: how to combine transformed elements.
//!
//! An accumulate is a (Grouping, Op) pair. Grouping says WHERE results
//! go. Op says HOW they combine.
//!
//! Multiple transforms feeding the SAME (Grouping, Op) fuse into a
//! single pass — the transform becomes an inline expression in the
//! kernel, and each transform's output is a separate accumulator slot.
//!
//! On GPU: one kernel launch per unique (Grouping, Op), regardless of
//! how many transforms feed it.
//! On CPU: one loop per unique (Grouping, Op), with all transforms
//! computed per element inside the loop.

use crate::transforms::Transform;

/// WHERE results go — how to partition input elements into accumulators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Grouping {
    /// N → 1. All elements into one accumulator. Scalar result.
    All,
    /// N → K. Elements scatter by key into K accumulators.
    ByKey,
    /// N → N. Forward inclusive scan (prefix sum, cumsum, etc).
    Prefix,
    /// N → N. Scan with resets at segment boundaries.
    Segmented,
    /// N → N. Rolling window via prefix subtraction trick.
    Windowed,
    /// M×K × K×N → M×N. Blocked matrix accumulation.
    Tiled,
    /// N → N. Scatter by graph adjacency (GNN, Laplacian).
    Graph,
}

/// HOW to combine accumulators — the associative binary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// a + b. The default for sums, counts, moments, dot products.
    Add,
    /// max(a, b). For running max, argmax prefix, extrema.
    Max,
    /// min(a, b). For running min, argmin prefix, extrema.
    Min,
    /// a * b. For products, factorial-like accumulations.
    Mul,
    /// a && b (as f64: 1.0 * 1.0 = 1.0, else 0.0). Logical AND.
    And,
    /// a || b (as f64: max(a, b) where a,b ∈ {0,1}). Logical OR.
    Or,
}

/// One accumulation slot: a transform feeding a (Grouping, Op).
#[derive(Debug, Clone)]
pub struct AccumulateSlot {
    /// What to compute per element before combining.
    pub transform: Transform,
    /// Where results go.
    pub grouping: Grouping,
    /// How to combine.
    pub op: Op,
    /// Name for this slot's output (referenced by gathers).
    pub output: String,
}

/// An accumulate pass: all slots that share (Grouping, Op) and
/// execute in ONE loop/kernel.
#[derive(Debug, Clone)]
pub struct AccumulatePass {
    pub grouping: Grouping,
    pub op: Op,
    /// All transforms computed in this single pass.
    pub slots: Vec<(Transform, String)>,  // (transform, output_name)
}

/// Given a list of slots, fuse into minimal passes.
/// All slots with the same (Grouping, Op) merge into one pass.
pub fn fuse_passes(slots: &[AccumulateSlot]) -> Vec<AccumulatePass> {
    use std::collections::HashMap;

    let mut groups: HashMap<(Grouping, Op), Vec<(Transform, String)>> = HashMap::new();

    for slot in slots {
        groups.entry((slot.grouping, slot.op))
            .or_default()
            .push((slot.transform, slot.output.clone()));
    }

    groups.into_iter()
        .map(|((grouping, op), slots)| AccumulatePass { grouping, op, slots })
        .collect()
}

/// Execute a single fused pass on CPU: one loop, many outputs.
pub fn execute_pass_cpu(
    pass: &AccumulatePass,
    data: &[f64],
    reference: f64,       // for centered transforms
    second_col: &[f64],   // for binary transforms (can be empty)
) -> Vec<(String, f64)> {
    let has_second = !second_col.is_empty();

    // Initialize accumulators
    let mut accs: Vec<f64> = pass.slots.iter().map(|_| match pass.op {
        Op::Add => 0.0,
        Op::Max => f64::NEG_INFINITY,
        Op::Min => f64::INFINITY,
        Op::Mul => 1.0,
        Op::And => 1.0,
        Op::Or  => 0.0,
    }).collect();

    // Single pass through data
    for (i, &x) in data.iter().enumerate() {
        let second = if has_second && i < second_col.len() { second_col[i] } else { 0.0 };

        for (j, (transform, _)) in pass.slots.iter().enumerate() {
            let val = transform.apply(x, reference, second);
            match pass.op {
                Op::Add => accs[j] += val,
                Op::Max => if val > accs[j] { accs[j] = val; },
                Op::Min => if val < accs[j] { accs[j] = val; },
                Op::Mul => accs[j] *= val,
                Op::And => accs[j] = if accs[j] > 0.5 && val > 0.5 { 1.0 } else { 0.0 },
                Op::Or  => accs[j] = if accs[j] > 0.5 || val > 0.5 { 1.0 } else { 0.0 },
            }
        }
    }

    pass.slots.iter().zip(accs.iter())
        .map(|((_, name), &val)| (name.clone(), val))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuse_all_add_slots() {
        let slots = vec![
            AccumulateSlot { transform: Transform::Identity, grouping: Grouping::All, op: Op::Add, output: "sum".into() },
            AccumulateSlot { transform: Transform::Const(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
            AccumulateSlot { transform: Transform::Square, grouping: Grouping::All, op: Op::Add, output: "sum_sq".into() },
            AccumulateSlot { transform: Transform::Ln, grouping: Grouping::All, op: Op::Add, output: "log_sum".into() },
            AccumulateSlot { transform: Transform::Reciprocal, grouping: Grouping::All, op: Op::Add, output: "recip_sum".into() },
        ];
        let passes = fuse_passes(&slots);
        assert_eq!(passes.len(), 1, "all 5 slots should fuse into 1 pass");
        assert_eq!(passes[0].slots.len(), 5);
    }

    #[test]
    fn execute_fused_moments() {
        let slots = vec![
            AccumulateSlot { transform: Transform::Identity, grouping: Grouping::All, op: Op::Add, output: "sum".into() },
            AccumulateSlot { transform: Transform::Const(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
            AccumulateSlot { transform: Transform::Square, grouping: Grouping::All, op: Op::Add, output: "sum_sq".into() },
        ];
        let passes = fuse_passes(&slots);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let results = execute_pass_cpu(&passes[0], &data, 0.0, &[]);
        let sum = results.iter().find(|(n, _)| n == "sum").unwrap().1;
        let count = results.iter().find(|(n, _)| n == "count").unwrap().1;
        let sum_sq = results.iter().find(|(n, _)| n == "sum_sq").unwrap().1;

        assert_eq!(sum, 15.0);
        assert_eq!(count, 5.0);
        assert_eq!(sum_sq, 55.0); // 1+4+9+16+25

        // Mean from gathered results
        let mean = sum / count;
        assert_eq!(mean, 3.0);

        // Variance from gathered results
        let var = (sum_sq - sum * sum / count) / (count - 1.0);
        assert!((var - 2.5).abs() < 1e-14);
    }

    #[test]
    fn cross_product_fuses() {
        let slots = vec![
            AccumulateSlot { transform: Transform::Identity, grouping: Grouping::All, op: Op::Add, output: "sum_x".into() },
            AccumulateSlot { transform: Transform::Square, grouping: Grouping::All, op: Op::Add, output: "sum_sq_x".into() },
            AccumulateSlot { transform: Transform::MulPair, grouping: Grouping::All, op: Op::Add, output: "sum_xy".into() },
            AccumulateSlot { transform: Transform::Const(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
        ];
        let passes = fuse_passes(&slots);
        assert_eq!(passes.len(), 1, "cross-product fuses with sums");

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0]; // y = 2x, perfect correlation

        let results = execute_pass_cpu(&passes[0], &x, 0.0, &y);
        let sum_xy = results.iter().find(|(n, _)| n == "sum_xy").unwrap().1;
        assert_eq!(sum_xy, 1.0*2.0 + 2.0*4.0 + 3.0*6.0); // 2+8+18 = 28
    }

    #[test]
    fn different_ops_dont_fuse() {
        let slots = vec![
            AccumulateSlot { transform: Transform::Identity, grouping: Grouping::All, op: Op::Add, output: "sum".into() },
            AccumulateSlot { transform: Transform::Identity, grouping: Grouping::All, op: Op::Max, output: "max".into() },
            AccumulateSlot { transform: Transform::Identity, grouping: Grouping::All, op: Op::Min, output: "min".into() },
        ];
        let passes = fuse_passes(&slots);
        assert_eq!(passes.len(), 3, "Add, Max, Min are 3 separate passes");
    }
}
