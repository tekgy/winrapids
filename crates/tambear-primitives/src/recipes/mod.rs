//! Recipes: named compositions of transforms → accumulates → gathers.
//!
//! A recipe is a TEACHING NAME for a specific chain of atoms.
//! "Mean arithmetic" is not a primitive. It's a recipe:
//!   Identity → Accumulate(All, Add) → "sum"
//!   Const(1) → Accumulate(All, Add) → "count"    ← FUSES
//!   Gather(sum / count)
//!
//! The recipe is what the IDE shows. The atoms are what TAM compiles.
//! The recipe is how humans think. The atoms are how the machine thinks.

use crate::tbs::Expr;
use crate::accumulates::{AccumulateSlot, Grouping, Op};
use crate::gathers::{Gather, GatherComputation};

/// A complete recipe: accumulate slots + gather expressions.
#[derive(Debug, Clone)]
pub struct Recipe {
    /// Teaching name (what the user sees).
    pub name: String,
    /// The accumulate slots (transforms + grouping + op).
    pub slots: Vec<AccumulateSlot>,
    /// The gather expressions (how to combine accumulated results).
    pub gathers: Vec<Gather>,
    /// Which gather output is the final result.
    pub result: String,
}

/// Build the mean_arithmetic recipe.
pub fn mean_arithmetic() -> Recipe {
    Recipe {
        name: "mean_arithmetic".into(),
        slots: vec![
            AccumulateSlot { expr: Expr::val(), grouping: Grouping::All, op: Op::Add, output: "sum".into() },
            AccumulateSlot { expr: Expr::lit(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
        ],
        gathers: vec![
            Gather {
                inputs: vec!["sum".into(), "count".into()],
                computation: GatherComputation::Divide("sum".into(), "count".into()),
                output: "mean".into(),
            },
        ],
        result: "mean".into(),
    }
}

/// Build the variance recipe. Shares sum + count with mean.
pub fn variance() -> Recipe {
    Recipe {
        name: "variance".into(),
        slots: vec![
            AccumulateSlot { expr: Expr::val(), grouping: Grouping::All, op: Op::Add, output: "sum".into() },
            AccumulateSlot { expr: Expr::val().sq(), grouping: Grouping::All, op: Op::Add, output: "sum_sq".into() },
            AccumulateSlot { expr: Expr::lit(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
        ],
        gathers: vec![
            Gather {
                inputs: vec!["sum".into(), "sum_sq".into(), "count".into()],
                computation: GatherComputation::VarianceFormula {
                    sum: "sum".into(), sum_sq: "sum_sq".into(), count: "count".into(),
                },
                output: "variance".into(),
            },
        ],
        result: "variance".into(),
    }
}

/// Build the geometric mean recipe. Shares count with mean.
pub fn mean_geometric() -> Recipe {
    Recipe {
        name: "mean_geometric".into(),
        slots: vec![
            AccumulateSlot { expr: Expr::val().ln(), grouping: Grouping::All, op: Op::Add, output: "log_sum".into() },
            AccumulateSlot { expr: Expr::lit(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
        ],
        gathers: vec![
            Gather {
                inputs: vec!["log_sum".into(), "count".into()],
                computation: GatherComputation::Exp(
                    Box::new(GatherComputation::Divide("log_sum".into(), "count".into()))
                ),
                output: "geometric_mean".into(),
            },
        ],
        result: "geometric_mean".into(),
    }
}

/// Build the Pearson correlation recipe. Two-column.
pub fn pearson_r() -> Recipe {
    Recipe {
        name: "pearson_r".into(),
        slots: vec![
            AccumulateSlot { expr: Expr::val(), grouping: Grouping::All, op: Op::Add, output: "sum_x".into() },
            AccumulateSlot { expr: Expr::val().sq(), grouping: Grouping::All, op: Op::Add, output: "sum_sq_x".into() },
            AccumulateSlot { expr: Expr::val().mul(Expr::val2()), grouping: Grouping::All, op: Op::Add, output: "sum_xy".into() },
            AccumulateSlot { expr: Expr::lit(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
            // Note: sum_y and sum_sq_y come from running the same slots on column y
        ],
        gathers: vec![
            Gather {
                inputs: vec!["sum_x".into(), "sum_y".into(), "sum_sq_x".into(),
                             "sum_sq_y".into(), "sum_xy".into(), "count".into()],
                computation: GatherComputation::PearsonFormula {
                    sum_x: "sum_x".into(), sum_y: "sum_y".into(),
                    sum_sq_x: "sum_sq_x".into(), sum_sq_y: "sum_sq_y".into(),
                    sum_xy: "sum_xy".into(), count: "count".into(),
                },
                output: "r".into(),
            },
        ],
        result: "r".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulates::fuse_passes;

    #[test]
    fn mean_is_two_slots_one_pass() {
        let r = mean_arithmetic();
        assert_eq!(r.slots.len(), 2);
        let passes = fuse_passes(&r.slots);
        assert_eq!(passes.len(), 1, "both slots fuse into one pass");
    }

    #[test]
    fn variance_is_three_slots_one_pass() {
        let r = variance();
        assert_eq!(r.slots.len(), 3);
        let passes = fuse_passes(&r.slots);
        assert_eq!(passes.len(), 1);
    }

    #[test]
    fn mean_and_variance_share() {
        let m = mean_arithmetic();
        let v = variance();
        // Merge all slots and fuse
        let mut all_slots = m.slots.clone();
        all_slots.extend(v.slots.clone());
        let passes = fuse_passes(&all_slots);
        assert_eq!(passes.len(), 1, "all fuse into one pass");
        // But the pass has 3 unique transforms (identity, const(1), square)
        // not 5 (identity, const, identity, square, const)
        // because fuse_passes deduplicates by (grouping, op), and all are (All, Add)
        assert_eq!(passes[0].slots.len(), 5, "5 total slots (dedup happens in TAM, not here)");
    }

    #[test]
    fn pearson_has_cross_product() {
        let r = pearson_r();
        let has_mul = r.slots.iter().any(|s| matches!(&s.expr, Expr::Mul(_, _)));
        assert!(has_mul, "pearson needs MulPair for sum_xy");
    }

    #[test]
    fn all_moment_recipes_fuse() {
        let recipes = vec![
            mean_arithmetic(), mean_geometric(), variance(), pearson_r(),
        ];
        let all_slots: Vec<AccumulateSlot> = recipes.iter()
            .flat_map(|r| r.slots.iter().cloned())
            .collect();
        let passes = fuse_passes(&all_slots);
        assert_eq!(passes.len(), 1, "ALL moment recipes fuse into ONE pass");
    }
}
