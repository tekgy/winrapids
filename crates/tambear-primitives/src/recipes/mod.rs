//! Recipes: named compositions of TBS expressions → accumulates → gathers.
//!
//! A recipe is a TEACHING NAME for a specific chain of atoms.
//! The recipe is what the IDE shows. The atoms are what TAM compiles.

use crate::tbs::Expr;
use crate::accumulates::{AccumulateSlot, Grouping, Op};
use crate::gathers::Gather;

/// A complete recipe: accumulate slots + gather expressions.
#[derive(Debug, Clone)]
pub struct Recipe {
    /// Teaching name (what the user sees).
    pub name: String,
    /// The accumulate slots (TBS expr + grouping + op).
    pub slots: Vec<AccumulateSlot>,
    /// The gather expressions (TBS expr over accumulated outputs).
    pub gathers: Vec<Gather>,
    /// Which gather output is the final result.
    pub result: String,
}

// ═══════════════════════════════════════════════════════════════════
// Standard recipes
// ═══════════════════════════════════════════════════════════════════

pub fn mean_arithmetic() -> Recipe {
    Recipe {
        name: "mean_arithmetic".into(),
        slots: vec![
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::val(), grouping: Grouping::All, op: Op::Add, output: "sum".into() },
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::lit(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
        ],
        gathers: vec![
            Gather { expr: Expr::var("sum").div(Expr::var("count")), output: "mean".into() },
        ],
        result: "mean".into(),
    }
}

pub fn variance() -> Recipe {
    Recipe {
        name: "variance".into(),
        slots: vec![
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::val(), grouping: Grouping::All, op: Op::Add, output: "sum".into() },
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::val().sq(), grouping: Grouping::All, op: Op::Add, output: "sum_sq".into() },
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::lit(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
        ],
        gathers: vec![
            Gather {
                expr: Expr::var("sum_sq")
                    .sub(Expr::var("sum").mul(Expr::var("sum")).div(Expr::var("count")))
                    .div(Expr::var("count").sub(Expr::lit(1.0))),
                output: "variance".into(),
            },
        ],
        result: "variance".into(),
    }
}

pub fn mean_geometric() -> Recipe {
    Recipe {
        name: "mean_geometric".into(),
        slots: vec![
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::val().ln(), grouping: Grouping::All, op: Op::Add, output: "log_sum".into() },
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::lit(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
        ],
        gathers: vec![
            Gather { expr: Expr::var("log_sum").div(Expr::var("count")).exp(), output: "geometric_mean".into() },
        ],
        result: "geometric_mean".into(),
    }
}

pub fn mean_harmonic() -> Recipe {
    Recipe {
        name: "mean_harmonic".into(),
        slots: vec![
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::val().recip(), grouping: Grouping::All, op: Op::Add, output: "recip_sum".into() },
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::lit(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
        ],
        gathers: vec![
            Gather { expr: Expr::var("count").div(Expr::var("recip_sum")), output: "harmonic_mean".into() },
        ],
        result: "harmonic_mean".into(),
    }
}

pub fn pearson_r() -> Recipe {
    let n = || Expr::var("count");
    Recipe {
        name: "pearson_r".into(),
        slots: vec![
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::val(), grouping: Grouping::All, op: Op::Add, output: "sum_x".into() },
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::val().sq(), grouping: Grouping::All, op: Op::Add, output: "sum_sq_x".into() },
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::val().mul(Expr::val2()), grouping: Grouping::All, op: Op::Add, output: "sum_xy".into() },
            AccumulateSlot { source: crate::accumulates::DataSource::Primary, expr: Expr::lit(1.0), grouping: Grouping::All, op: Op::Add, output: "count".into() },
        ],
        gathers: vec![
            Gather {
                expr: {
                    let num = Expr::var("sum_xy")
                        .sub(Expr::var("sum_x").mul(Expr::var("sum_y")).div(n()));
                    let den_x = Expr::var("sum_sq_x")
                        .sub(Expr::var("sum_x").mul(Expr::var("sum_x")).div(n()));
                    let den_y = Expr::var("sum_sq_y")
                        .sub(Expr::var("sum_y").mul(Expr::var("sum_y")).div(n()));
                    num.div(den_x.mul(den_y).sqrt())
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
        assert_eq!(passes.len(), 1);
    }

    #[test]
    fn variance_is_three_slots_one_pass() {
        let r = variance();
        assert_eq!(r.slots.len(), 3);
        let passes = fuse_passes(&r.slots);
        assert_eq!(passes.len(), 1);
    }

    #[test]
    fn all_moment_recipes_fuse() {
        let recipes = vec![
            mean_arithmetic(), mean_geometric(), mean_harmonic(), variance(), pearson_r(),
        ];
        let all_slots: Vec<AccumulateSlot> = recipes.iter()
            .flat_map(|r| r.slots.iter().cloned())
            .collect();
        let passes = fuse_passes(&all_slots);
        assert_eq!(passes.len(), 1, "ALL recipes fuse into ONE pass");
    }

    #[test]
    fn pearson_has_cross_product() {
        let r = pearson_r();
        let has_mul = r.slots.iter().any(|s| matches!(&s.expr, Expr::Mul(_, _)));
        assert!(has_mul);
    }
}

#[cfg(test)]
#[path = "coverage_test.rs"]
mod coverage_test;
