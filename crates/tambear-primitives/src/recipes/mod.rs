//! Recipes: named compositions of TBS expressions → accumulates → gathers.
//!
//! A recipe is a TEACHING NAME for a specific chain of atoms.
//! The recipe is what the IDE shows. The atoms are what TAM compiles.
//!
//! Recipes are DATA, not code. Every recipe is:
//!   - a `Vec<AccumulateSlot>` (TBS expr + Grouping + Op per slot)
//!   - a `Vec<Gather>` (TBS expr over accumulated values)
//!   - a `result` name picking which gather output is the final answer
//!
//! All slots with the same (DataSource, Grouping, Op) across ALL recipes
//! fuse into ONE kernel. A pipeline computing 40 statistics might run in
//! 2 passes total — that's the payoff.

use crate::tbs::Expr;
use crate::accumulates::{AccumulateSlot, DataSource, Grouping, Op};
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
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn add_all(expr: Expr, output: &str) -> AccumulateSlot {
    AccumulateSlot {
        source: DataSource::Primary,
        expr,
        grouping: Grouping::All,
        op: Op::Add,
        output: output.into(),
    }
}

fn op_all(expr: Expr, op: Op, output: &str) -> AccumulateSlot {
    AccumulateSlot {
        source: DataSource::Primary,
        expr,
        grouping: Grouping::All,
        op,
        output: output.into(),
    }
}

fn count_slot() -> AccumulateSlot { add_all(Expr::lit(1.0), "count") }
fn sum_slot()   -> AccumulateSlot { add_all(Expr::val(), "sum") }
fn sum_sq_slot()-> AccumulateSlot { add_all(Expr::val().sq(), "sum_sq") }
fn sum_cb_slot()-> AccumulateSlot { add_all(Expr::val().sq().mul(Expr::val()), "sum_cb") }
fn sum_4_slot() -> AccumulateSlot { add_all(Expr::val().sq().sq(), "sum_4") }

// Two-column helpers (val2 = second input column)
fn sum_y_slot()    -> AccumulateSlot { add_all(Expr::val2(), "sum_y") }
fn sum_sq_y_slot() -> AccumulateSlot { add_all(Expr::val2().sq(), "sum_sq_y") }
fn sum_xy_slot()   -> AccumulateSlot { add_all(Expr::val().mul(Expr::val2()), "sum_xy") }

// ═══════════════════════════════════════════════════════════════════
// FAMILY: raw reductions
// ═══════════════════════════════════════════════════════════════════

/// N. Count of elements.
pub fn count() -> Recipe {
    Recipe {
        name: "count".into(),
        slots: vec![count_slot()],
        gathers: vec![Gather { expr: Expr::var("count"), output: "n".into() }],
        result: "n".into(),
    }
}

/// Σxᵢ. Total sum.
pub fn sum() -> Recipe {
    Recipe {
        name: "sum".into(),
        slots: vec![sum_slot()],
        gathers: vec![Gather { expr: Expr::var("sum"), output: "s".into() }],
        result: "s".into(),
    }
}

/// Σxᵢ². Sum of squares.
pub fn sum_of_squares() -> Recipe {
    Recipe {
        name: "sum_of_squares".into(),
        slots: vec![sum_sq_slot()],
        gathers: vec![Gather { expr: Expr::var("sum_sq"), output: "ss".into() }],
        result: "ss".into(),
    }
}

/// Πxᵢ. Product of all elements.
pub fn product() -> Recipe {
    Recipe {
        name: "product".into(),
        slots: vec![op_all(Expr::val(), Op::Mul, "prod")],
        gathers: vec![Gather { expr: Expr::var("prod"), output: "p".into() }],
        result: "p".into(),
    }
}

// ═══════════════════════════════════════════════════════════════════
// FAMILY: means
// ═══════════════════════════════════════════════════════════════════

/// (Σxᵢ) / n. Arithmetic mean.
pub fn mean_arithmetic() -> Recipe {
    Recipe {
        name: "mean_arithmetic".into(),
        slots: vec![sum_slot(), count_slot()],
        gathers: vec![Gather {
            expr: Expr::var("sum").div(Expr::var("count")),
            output: "mean".into(),
        }],
        result: "mean".into(),
    }
}

/// exp((Σ ln xᵢ) / n). Geometric mean.
pub fn mean_geometric() -> Recipe {
    Recipe {
        name: "mean_geometric".into(),
        slots: vec![
            add_all(Expr::val().ln(), "log_sum"),
            count_slot(),
        ],
        gathers: vec![Gather {
            expr: Expr::var("log_sum").div(Expr::var("count")).exp(),
            output: "geometric_mean".into(),
        }],
        result: "geometric_mean".into(),
    }
}

/// n / (Σ 1/xᵢ). Harmonic mean.
pub fn mean_harmonic() -> Recipe {
    Recipe {
        name: "mean_harmonic".into(),
        slots: vec![
            add_all(Expr::val().recip(), "recip_sum"),
            count_slot(),
        ],
        gathers: vec![Gather {
            expr: Expr::var("count").div(Expr::var("recip_sum")),
            output: "harmonic_mean".into(),
        }],
        result: "harmonic_mean".into(),
    }
}

/// √((Σxᵢ²) / n). Quadratic mean (RMS).
pub fn mean_quadratic() -> Recipe {
    Recipe {
        name: "mean_quadratic".into(),
        slots: vec![sum_sq_slot(), count_slot()],
        gathers: vec![Gather {
            expr: Expr::var("sum_sq").div(Expr::var("count")).sqrt(),
            output: "rms".into(),
        }],
        result: "rms".into(),
    }
}

// ═══════════════════════════════════════════════════════════════════
// FAMILY: moments
// ═══════════════════════════════════════════════════════════════════

/// (Σxᵢ² - (Σxᵢ)²/n) / (n-1). Unbiased sample variance.
pub fn variance() -> Recipe {
    Recipe {
        name: "variance".into(),
        slots: vec![sum_slot(), sum_sq_slot(), count_slot()],
        gathers: vec![Gather {
            expr: Expr::var("sum_sq")
                .sub(Expr::var("sum").mul(Expr::var("sum")).div(Expr::var("count")))
                .div(Expr::var("count").sub(Expr::lit(1.0))),
            output: "variance".into(),
        }],
        result: "variance".into(),
    }
}

/// (Σxᵢ² - (Σxᵢ)²/n) / n. Biased variance (MLE).
pub fn variance_biased() -> Recipe {
    Recipe {
        name: "variance_biased".into(),
        slots: vec![sum_slot(), sum_sq_slot(), count_slot()],
        gathers: vec![Gather {
            expr: Expr::var("sum_sq")
                .sub(Expr::var("sum").mul(Expr::var("sum")).div(Expr::var("count")))
                .div(Expr::var("count")),
            output: "variance_biased".into(),
        }],
        result: "variance_biased".into(),
    }
}

/// √variance. Sample standard deviation.
pub fn std_dev() -> Recipe {
    Recipe {
        name: "std_dev".into(),
        slots: vec![sum_slot(), sum_sq_slot(), count_slot()],
        gathers: vec![Gather {
            expr: Expr::var("sum_sq")
                .sub(Expr::var("sum").mul(Expr::var("sum")).div(Expr::var("count")))
                .div(Expr::var("count").sub(Expr::lit(1.0)))
                .sqrt(),
            output: "std".into(),
        }],
        result: "std".into(),
    }
}

/// Fisher skewness from raw moments.
/// g₁ = (m₃ - 3·m̄·m₂ + 2·m̄³) / σ³   where σ² = m₂ - m̄², m̄ = μ.
pub fn skewness() -> Recipe {
    let n   = || Expr::var("count");
    let m1  = || Expr::var("sum").div(n());              // mean
    let m2  = || Expr::var("sum_sq").div(n());           // raw 2nd
    let m3  = || Expr::var("sum_cb").div(n());           // raw 3rd
    // Central 2nd moment σ² = m2 - m1²
    let var_c = || m2().sub(m1().mul(m1()));
    // Central 3rd moment μ₃ = m3 - 3·m1·m2 + 2·m1³
    let cent3 = || m3()
        .sub(Expr::lit(3.0).mul(m1()).mul(m2()))
        .add(Expr::lit(2.0).mul(m1()).mul(m1()).mul(m1()));
    Recipe {
        name: "skewness".into(),
        slots: vec![sum_slot(), sum_sq_slot(), sum_cb_slot(), count_slot()],
        gathers: vec![Gather {
            expr: cent3().div(var_c().sqrt().mul(var_c())),
            output: "skew".into(),
        }],
        result: "skew".into(),
    }
}

/// Excess kurtosis from raw moments.
/// g₂ = μ₄ / σ⁴ - 3   where μ₄ = m4 - 4·m1·m3 + 6·m1²·m2 - 3·m1⁴.
pub fn kurtosis_excess() -> Recipe {
    let n   = || Expr::var("count");
    let m1  = || Expr::var("sum").div(n());
    let m2  = || Expr::var("sum_sq").div(n());
    let m3  = || Expr::var("sum_cb").div(n());
    let m4  = || Expr::var("sum_4").div(n());
    let var_c = || m2().sub(m1().mul(m1()));
    // μ₄ = m4 - 4 m1 m3 + 6 m1² m2 - 3 m1⁴
    let cent4 = || m4()
        .sub(Expr::lit(4.0).mul(m1()).mul(m3()))
        .add(Expr::lit(6.0).mul(m1()).mul(m1()).mul(m2()))
        .sub(Expr::lit(3.0).mul(m1()).mul(m1()).mul(m1()).mul(m1()));
    Recipe {
        name: "kurtosis_excess".into(),
        slots: vec![sum_slot(), sum_sq_slot(), sum_cb_slot(), sum_4_slot(), count_slot()],
        gathers: vec![Gather {
            expr: cent4().div(var_c().mul(var_c())).sub(Expr::lit(3.0)),
            output: "kurt".into(),
        }],
        result: "kurt".into(),
    }
}

// ═══════════════════════════════════════════════════════════════════
// FAMILY: norms
// ═══════════════════════════════════════════════════════════════════

/// Σ|xᵢ|. L1 norm.
pub fn l1_norm() -> Recipe {
    Recipe {
        name: "l1_norm".into(),
        slots: vec![add_all(Expr::val().abs(), "abs_sum")],
        gathers: vec![Gather { expr: Expr::var("abs_sum"), output: "l1".into() }],
        result: "l1".into(),
    }
}

/// √(Σxᵢ²). L2 norm.
pub fn l2_norm() -> Recipe {
    Recipe {
        name: "l2_norm".into(),
        slots: vec![sum_sq_slot()],
        gathers: vec![Gather { expr: Expr::var("sum_sq").sqrt(), output: "l2".into() }],
        result: "l2".into(),
    }
}

/// max|xᵢ|. L∞ norm.
pub fn linf_norm() -> Recipe {
    Recipe {
        name: "linf_norm".into(),
        slots: vec![op_all(Expr::val().abs(), Op::Max, "max_abs")],
        gathers: vec![Gather { expr: Expr::var("max_abs"), output: "linf".into() }],
        result: "linf".into(),
    }
}

// ═══════════════════════════════════════════════════════════════════
// FAMILY: extrema
// ═══════════════════════════════════════════════════════════════════

/// min xᵢ.
pub fn min_all() -> Recipe {
    Recipe {
        name: "min_all".into(),
        slots: vec![op_all(Expr::val(), Op::Min, "min")],
        gathers: vec![Gather { expr: Expr::var("min"), output: "min".into() }],
        result: "min".into(),
    }
}

/// max xᵢ.
pub fn max_all() -> Recipe {
    Recipe {
        name: "max_all".into(),
        slots: vec![op_all(Expr::val(), Op::Max, "max")],
        gathers: vec![Gather { expr: Expr::var("max"), output: "max".into() }],
        result: "max".into(),
    }
}

/// max - min. Range. (2 passes: Min/Max don't fuse.)
pub fn range_all() -> Recipe {
    Recipe {
        name: "range_all".into(),
        slots: vec![
            op_all(Expr::val(), Op::Min, "min"),
            op_all(Expr::val(), Op::Max, "max"),
        ],
        gathers: vec![Gather {
            expr: Expr::var("max").sub(Expr::var("min")),
            output: "range".into(),
        }],
        result: "range".into(),
    }
}

/// (min + max) / 2. Midrange.
pub fn midrange() -> Recipe {
    Recipe {
        name: "midrange".into(),
        slots: vec![
            op_all(Expr::val(), Op::Min, "min"),
            op_all(Expr::val(), Op::Max, "max"),
        ],
        gathers: vec![Gather {
            expr: Expr::var("min").add(Expr::var("max")).div(Expr::lit(2.0)),
            output: "midrange".into(),
        }],
        result: "midrange".into(),
    }
}

// ═══════════════════════════════════════════════════════════════════
// FAMILY: two-column statistics
// ═══════════════════════════════════════════════════════════════════

/// Σ xᵢ·yᵢ. Dot product / inner product.
pub fn dot_product() -> Recipe {
    Recipe {
        name: "dot_product".into(),
        slots: vec![sum_xy_slot()],
        gathers: vec![Gather { expr: Expr::var("sum_xy"), output: "dot".into() }],
        result: "dot".into(),
    }
}

/// Sample covariance: (Σxy - ΣxΣy/n) / (n-1).
pub fn covariance() -> Recipe {
    let n = || Expr::var("count");
    Recipe {
        name: "covariance".into(),
        slots: vec![sum_slot(), sum_y_slot(), sum_xy_slot(), count_slot()],
        gathers: vec![Gather {
            expr: Expr::var("sum_xy")
                .sub(Expr::var("sum").mul(Expr::var("sum_y")).div(n()))
                .div(n().sub(Expr::lit(1.0))),
            output: "cov".into(),
        }],
        result: "cov".into(),
    }
}

/// Pearson correlation coefficient r.
/// r = (Σxy - ΣxΣy/n) / √((Σx² - (Σx)²/n)(Σy² - (Σy)²/n))
pub fn pearson_r() -> Recipe {
    let n = || Expr::var("count");
    let num = Expr::var("sum_xy")
        .sub(Expr::var("sum").mul(Expr::var("sum_y")).div(n()));
    let den_x = Expr::var("sum_sq")
        .sub(Expr::var("sum").mul(Expr::var("sum")).div(n()));
    let den_y = Expr::var("sum_sq_y")
        .sub(Expr::var("sum_y").mul(Expr::var("sum_y")).div(n()));
    Recipe {
        name: "pearson_r".into(),
        slots: vec![
            sum_slot(),
            sum_sq_slot(),
            sum_y_slot(),
            sum_sq_y_slot(),
            sum_xy_slot(),
            count_slot(),
        ],
        gathers: vec![Gather {
            expr: num.div(den_x.mul(den_y).sqrt()),
            output: "r".into(),
        }],
        result: "r".into(),
    }
}

/// Sum of squared differences Σ(xᵢ-yᵢ)². Useful for SSE/RMSE.
pub fn sum_squared_diff() -> Recipe {
    // (x-y)² = x² - 2xy + y²  — but also just compose directly as expr.
    let diff_sq = Expr::val().sub(Expr::val2()).sq();
    Recipe {
        name: "sum_squared_diff".into(),
        slots: vec![add_all(diff_sq, "ssd")],
        gathers: vec![Gather { expr: Expr::var("ssd"), output: "ssd".into() }],
        result: "ssd".into(),
    }
}

/// √(Σ(xᵢ-yᵢ)² / n). Root mean square error.
pub fn rmse() -> Recipe {
    let diff_sq = Expr::val().sub(Expr::val2()).sq();
    Recipe {
        name: "rmse".into(),
        slots: vec![add_all(diff_sq, "ssd"), count_slot()],
        gathers: vec![Gather {
            expr: Expr::var("ssd").div(Expr::var("count")).sqrt(),
            output: "rmse".into(),
        }],
        result: "rmse".into(),
    }
}

/// Σ|xᵢ-yᵢ| / n. Mean absolute error.
pub fn mae() -> Recipe {
    let abs_diff = Expr::val().sub(Expr::val2()).abs();
    Recipe {
        name: "mae".into(),
        slots: vec![add_all(abs_diff, "sad"), count_slot()],
        gathers: vec![Gather {
            expr: Expr::var("sad").div(Expr::var("count")),
            output: "mae".into(),
        }],
        result: "mae".into(),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Catalog
// ═══════════════════════════════════════════════════════════════════

/// Every recipe in the library, in one list — the teaching table.
pub fn catalog() -> Vec<Recipe> {
    vec![
        count(), sum(), sum_of_squares(), product(),
        mean_arithmetic(), mean_geometric(), mean_harmonic(), mean_quadratic(),
        variance(), variance_biased(), std_dev(), skewness(), kurtosis_excess(),
        l1_norm(), l2_norm(), linf_norm(),
        min_all(), max_all(), range_all(), midrange(),
        dot_product(), covariance(), pearson_r(), sum_squared_diff(), rmse(), mae(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulates::{fuse_passes, execute_pass_cpu};
    use crate::tbs::eval;
    use std::collections::HashMap;

    /// Build var bindings from a fused-pass execution of a recipe's slots,
    /// then evaluate every gather, returning the requested result.
    fn run_recipe(recipe: &Recipe, x: &[f64], y: &[f64]) -> f64 {
        let passes = fuse_passes(&recipe.slots);
        let mut vars: HashMap<String, f64> = HashMap::new();
        for pass in &passes {
            let results = execute_pass_cpu(pass, x, 0.0, y);
            for (name, val) in results {
                vars.insert(name, val);
            }
        }
        let mut final_val = 0.0;
        for g in &recipe.gathers {
            let v = eval(&g.expr, 0.0, 0.0, 0.0, &vars);
            vars.insert(g.output.clone(), v);
            if g.output == recipe.result { final_val = v; }
        }
        final_val
    }

    #[test]
    fn count_is_n() {
        let v = run_recipe(&count(), &[1.0, 2.0, 3.0, 4.0], &[]);
        assert_eq!(v, 4.0);
    }

    #[test]
    fn sum_is_sum() {
        let v = run_recipe(&sum(), &[1.0, 2.0, 3.0, 4.0], &[]);
        assert_eq!(v, 10.0);
    }

    #[test]
    fn product_of_range() {
        let v = run_recipe(&product(), &[1.0, 2.0, 3.0, 4.0], &[]);
        assert_eq!(v, 24.0);
    }

    #[test]
    fn arithmetic_mean() {
        let v = run_recipe(&mean_arithmetic(), &[1.0, 2.0, 3.0, 4.0, 5.0], &[]);
        assert_eq!(v, 3.0);
    }

    #[test]
    fn geometric_mean_of_ones_is_one() {
        let v = run_recipe(&mean_geometric(), &[1.0, 1.0, 1.0], &[]);
        assert!((v - 1.0).abs() < 1e-14);
    }

    #[test]
    fn harmonic_mean_known() {
        // H({1,2,4}) = 3 / (1 + 0.5 + 0.25) = 3 / 1.75 = 12/7
        let v = run_recipe(&mean_harmonic(), &[1.0, 2.0, 4.0], &[]);
        assert!((v - 12.0/7.0).abs() < 1e-14);
    }

    #[test]
    fn rms_known() {
        // √((1+4+9)/3) = √(14/3)
        let v = run_recipe(&mean_quadratic(), &[1.0, 2.0, 3.0], &[]);
        assert!((v - (14.0_f64/3.0).sqrt()).abs() < 1e-14);
    }

    #[test]
    fn unbiased_variance() {
        let v = run_recipe(&variance(), &[1.0, 2.0, 3.0, 4.0, 5.0], &[]);
        assert!((v - 2.5).abs() < 1e-14);
    }

    #[test]
    fn biased_variance() {
        // pop variance of 1..5 = 2
        let v = run_recipe(&variance_biased(), &[1.0, 2.0, 3.0, 4.0, 5.0], &[]);
        assert!((v - 2.0).abs() < 1e-14);
    }

    #[test]
    fn std_dev_matches_sqrt_variance() {
        let s = run_recipe(&std_dev(), &[1.0, 2.0, 3.0, 4.0, 5.0], &[]);
        let v = run_recipe(&variance(), &[1.0, 2.0, 3.0, 4.0, 5.0], &[]);
        assert!((s - v.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn skewness_symmetric_is_zero() {
        // symmetric around mean → skew = 0
        let v = run_recipe(&skewness(), &[1.0, 2.0, 3.0, 4.0, 5.0], &[]);
        assert!(v.abs() < 1e-12);
    }

    #[test]
    fn kurtosis_uniform_is_negative() {
        // kurt of uniform discrete (1..5) < 0
        let v = run_recipe(&kurtosis_excess(), &[1.0, 2.0, 3.0, 4.0, 5.0], &[]);
        assert!(v < 0.0, "uniform data has negative excess kurtosis, got {}", v);
    }

    #[test]
    fn l1_norm_known() {
        let v = run_recipe(&l1_norm(), &[1.0, -2.0, 3.0, -4.0], &[]);
        assert_eq!(v, 10.0);
    }

    #[test]
    fn l2_norm_pythagorean() {
        let v = run_recipe(&l2_norm(), &[3.0, 4.0], &[]);
        assert!((v - 5.0).abs() < 1e-14);
    }

    #[test]
    fn linf_is_max_abs() {
        let v = run_recipe(&linf_norm(), &[1.0, -7.0, 3.0, -2.0], &[]);
        assert_eq!(v, 7.0);
    }

    #[test]
    fn min_max_range_midrange() {
        let x = [2.0, 8.0, 5.0, 1.0, 9.0];
        assert_eq!(run_recipe(&min_all(), &x, &[]), 1.0);
        assert_eq!(run_recipe(&max_all(), &x, &[]), 9.0);
        assert_eq!(run_recipe(&range_all(), &x, &[]), 8.0);
        assert_eq!(run_recipe(&midrange(), &x, &[]), 5.0);
    }

    #[test]
    fn dot_product_orthogonal() {
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        assert_eq!(run_recipe(&dot_product(), &x, &y), 0.0);
    }

    #[test]
    fn covariance_linear() {
        // y = 2x + 1, n=4 → cov should be positive
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [3.0, 5.0, 7.0, 9.0];
        let c = run_recipe(&covariance(), &x, &y);
        // cov = 2 * var(x) = 2 * 5/3 for unbiased
        assert!((c - 10.0/3.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_perfect_positive() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        let r = run_recipe(&pearson_r(), &x, &y);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [8.0, 6.0, 4.0, 2.0];
        let r = run_recipe(&pearson_r(), &x, &y);
        assert!((r + 1.0).abs() < 1e-12);
    }

    #[test]
    fn rmse_zero_when_equal() {
        let x = [1.0, 2.0, 3.0];
        let v = run_recipe(&rmse(), &x, &x);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn mae_known() {
        let x = [1.0, 2.0, 3.0];
        let y = [2.0, 4.0, 6.0];
        let v = run_recipe(&mae(), &x, &y);
        // |1-2| + |2-4| + |3-6| = 1+2+3 = 6; mae = 2
        assert!((v - 2.0).abs() < 1e-14);
    }

    // ═══════════════════════════════════════════════════════════════
    // FUSION TESTS — the whole point
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn moment_family_fuses_to_one_pass() {
        let recipes = vec![
            count(), sum(), sum_of_squares(),
            mean_arithmetic(), mean_quadratic(),
            variance(), variance_biased(), std_dev(),
            skewness(), kurtosis_excess(),
        ];
        let all_slots: Vec<AccumulateSlot> = recipes.iter()
            .flat_map(|r| r.slots.iter().cloned())
            .collect();
        let passes = fuse_passes(&all_slots);
        assert_eq!(passes.len(), 1,
            "10 moment recipes should fuse to 1 pass, got {}", passes.len());
    }

    #[test]
    fn full_catalog_minimum_passes() {
        // 26 recipes across 4 Ops over Primary/All:
        //   Add → all sums (count, sum, sum_sq, sum_cb, sum_4, log_sum,
        //                   recip_sum, abs_sum, sum_y, sum_sq_y, sum_xy, ssd, sad)
        //   Mul → product
        //   Min → min
        //   Max → max, max_abs (linf)
        // → 4 passes for the ENTIRE library.
        let all: Vec<_> = catalog().iter()
            .flat_map(|r| r.slots.iter().cloned()).collect();
        let passes = fuse_passes(&all);
        let total_slots: usize = catalog().iter().map(|r| r.slots.len()).sum();
        eprintln!("catalog: {} recipes, {} slots, {} passes",
                  catalog().len(), total_slots, passes.len());
        for p in &passes {
            eprintln!("  {:?}/{:?}/{:?}: {} slots",
                      p.source, p.grouping, p.op, p.slots.len());
        }
        assert_eq!(passes.len(), 4,
            "expected 4 passes for full catalog, got {}", passes.len());
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
