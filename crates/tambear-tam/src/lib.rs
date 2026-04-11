//! TAM: the accumulate+gather compiler.
//!
//! Takes a set of recipes (from tambear-primitives), finds shared
//! accumulate steps, deduplicates into a fused execution plan,
//! and executes on any available ALU.
//!
//! ```text
//! recipes → Plan → Execute → results
//! ```
//!
//! The Plan is the fused IR. Shared accumulate steps appear once.
//! Each recipe's gather expressions reference the shared outputs.

use std::collections::HashMap;
use tambear_primitives::recipe::*;

// ═══════════════════════════════════════════════════════════════════
// Plan: the fused execution IR
// ═══════════════════════════════════════════════════════════════════

/// A unique accumulate operation. Two recipes that both need
/// Accumulate(All, Value, Add) get ONE FusedAccumulate with
/// two output aliases.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AccumulateKey {
    pub grouping: GroupingKind,
    pub expr: ExprKind,
    pub op: OpKind,
}

/// A fused execution plan. Shared accumulates appear once.
#[derive(Debug)]
pub struct Plan {
    /// Unique accumulate steps (deduplicated across all recipes).
    pub accumulates: Vec<AccumulateKey>,
    /// For each recipe: which accumulate indices it needs + its gather expression.
    pub recipe_plans: Vec<RecipePlan>,
}

/// One recipe's view of the fused plan.
#[derive(Debug)]
pub struct RecipePlan {
    pub recipe_name: &'static str,
    /// Maps output name → index into Plan::accumulates.
    pub acc_bindings: HashMap<&'static str, usize>,
    /// Gather expressions that reference accumulated outputs.
    pub gathers: Vec<GatherStep>,
    /// Which gather output is the final result.
    pub result_name: &'static str,
}

#[derive(Debug)]
pub struct GatherStep {
    pub expr: &'static str,
    pub output: &'static str,
}

impl Plan {
    /// Compile a set of recipes into a fused plan.
    pub fn compile(recipes: &[&Recipe]) -> Self {
        let mut accumulates: Vec<AccumulateKey> = Vec::new();
        let mut acc_index: HashMap<AccumulateKey, usize> = HashMap::new();
        let mut recipe_plans = Vec::new();

        for recipe in recipes {
            let mut bindings: HashMap<&'static str, usize> = HashMap::new();
            let mut gathers = Vec::new();

            for step in recipe.steps.iter() {
                match step {
                    Step::Accumulate { grouping, expr, op, output } => {
                        let key = AccumulateKey {
                            grouping: *grouping,
                            expr: *expr,
                            op: *op,
                        };
                        let idx = if let Some(&existing) = acc_index.get(&key) {
                            existing // SHARED — reuse existing accumulate
                        } else {
                            let idx = accumulates.len();
                            acc_index.insert(key.clone(), idx);
                            accumulates.push(key);
                            idx
                        };
                        bindings.insert(output, idx);
                    }
                    Step::Gather { expr, output } => {
                        gathers.push(GatherStep { expr, output });
                    }
                    Step::Transform { .. } => {
                        // TODO: fuse transforms into accumulate expr
                    }
                }
            }

            recipe_plans.push(RecipePlan {
                recipe_name: recipe.name,
                acc_bindings: bindings,
                gathers,
                result_name: recipe.result,
            });
        }

        Plan { accumulates, recipe_plans }
    }

    /// How many accumulate passes does this plan need?
    /// This is the number of UNIQUE (grouping, expr, op) triples.
    pub fn n_accumulates(&self) -> usize {
        self.accumulates.len()
    }

    /// How many accumulates were saved by sharing?
    pub fn n_saved(&self, recipes: &[&Recipe]) -> usize {
        let total_without_sharing: usize = recipes.iter()
            .map(|r| r.accumulate_steps().len())
            .sum();
        total_without_sharing - self.n_accumulates()
    }
}

// ═══════════════════════════════════════════════════════════════════
// CPU Executor: run the plan on data
// ═══════════════════════════════════════════════════════════════════

/// Results of executing a plan.
#[derive(Debug)]
pub struct PlanResults {
    /// Accumulated values, indexed by Plan::accumulates position.
    pub acc_values: Vec<f64>,
    /// Final results per recipe, keyed by recipe name.
    pub results: HashMap<String, f64>,
}

/// Execute a plan on CPU. This is the reference implementation.
/// TAM's GPU backend would replace this with kernel dispatch.
pub fn execute_cpu(plan: &Plan, data: &[f64]) -> PlanResults {
    let n = data.len();

    // Phase 1: Execute all accumulates (ONE pass per unique accumulate)
    let mut acc_values: Vec<f64> = Vec::with_capacity(plan.accumulates.len());

    for acc in &plan.accumulates {
        let value = match (&acc.grouping, &acc.expr, &acc.op) {
            // All + Value + Add = sum
            (GroupingKind::All, ExprKind::Value, OpKind::Add) => {
                data.iter().sum::<f64>()
            }
            // All + One + Add = count
            (GroupingKind::All, ExprKind::One, OpKind::Add) => {
                n as f64
            }
            // All + ValueSq + Add = sum of squares
            (GroupingKind::All, ExprKind::ValueSq, OpKind::Add) => {
                data.iter().map(|&v| v * v).sum::<f64>()
            }
            // All + Ln + Add = sum of logs
            (GroupingKind::All, ExprKind::Ln, OpKind::Add) => {
                data.iter().map(|&v| v.ln()).sum::<f64>()
            }
            // All + Reciprocal + Add = sum of reciprocals
            (GroupingKind::All, ExprKind::Reciprocal, OpKind::Add) => {
                data.iter().map(|&v| 1.0 / v).sum::<f64>()
            }
            // All + Value + Max
            (GroupingKind::All, ExprKind::Value, OpKind::Max) => {
                data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            }
            // All + Value + Min
            (GroupingKind::All, ExprKind::Value, OpKind::Min) => {
                data.iter().cloned().fold(f64::INFINITY, f64::min)
            }
            _ => {
                f64::NAN // unimplemented combination
            }
        };
        acc_values.push(value);
    }

    // Phase 2: Execute gathers (evaluate expressions over accumulated values)
    let mut results: HashMap<String, f64> = HashMap::new();

    for rp in &plan.recipe_plans {
        // Build a variable map: output_name → accumulated value
        let mut vars: HashMap<&str, f64> = HashMap::new();
        for (&name, &idx) in &rp.acc_bindings {
            vars.insert(name, acc_values[idx]);
        }

        // Evaluate each gather expression
        for gather in &rp.gathers {
            let value = eval_gather_expr(gather.expr, &vars);
            vars.insert(gather.output, value);
        }

        // The result is the gather output matching result_name
        if let Some(&val) = vars.get(rp.result_name) {
            results.insert(rp.recipe_name.to_string(), val);
        }
    }

    PlanResults { acc_values, results }
}

/// Simple expression evaluator for gather expressions.
/// Supports: +, -, *, /, sqrt(), exp(), ln(), variable names.
fn eval_gather_expr(expr: &str, vars: &HashMap<&str, f64>) -> f64 {
    let expr = expr.trim();

    // Handle function wrappers: sqrt(...), exp(...), ln(...)
    if let Some(inner) = expr.strip_prefix("sqrt(").and_then(|s| s.strip_suffix(')')) {
        return eval_gather_expr(inner, vars).sqrt();
    }
    if let Some(inner) = expr.strip_prefix("exp(").and_then(|s| s.strip_suffix(')')) {
        return eval_gather_expr(inner, vars).exp();
    }
    if let Some(inner) = expr.strip_prefix("ln(").and_then(|s| s.strip_suffix(')')) {
        return eval_gather_expr(inner, vars).ln();
    }

    // Handle binary ops: find the LAST +/- at top level (lowest precedence)
    let mut paren_depth = 0i32;
    let bytes = expr.as_bytes();
    let mut last_add_sub = None;
    let mut last_mul_div = None;

    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => paren_depth += 1,
            b')' => paren_depth -= 1,
            b'+' | b'-' if paren_depth == 0 && i > 0 => last_add_sub = Some(i),
            b'*' | b'/' if paren_depth == 0 => last_mul_div = Some(i),
            _ => {}
        }
    }

    if let Some(pos) = last_add_sub {
        let left = eval_gather_expr(&expr[..pos], vars);
        let right = eval_gather_expr(&expr[pos+1..], vars);
        return match bytes[pos] {
            b'+' => left + right,
            b'-' => left - right,
            _ => unreachable!(),
        };
    }

    if let Some(pos) = last_mul_div {
        let left = eval_gather_expr(&expr[..pos], vars);
        let right = eval_gather_expr(&expr[pos+1..], vars);
        return match bytes[pos] {
            b'*' => left * right,
            b'/' => left / right,
            _ => unreachable!(),
        };
    }

    // Handle parenthesized expression
    if expr.starts_with('(') && expr.ends_with(')') {
        return eval_gather_expr(&expr[1..expr.len()-1], vars);
    }

    // Try as number
    if let Ok(val) = expr.parse::<f64>() {
        return val;
    }

    // Try as variable
    if let Some(&val) = vars.get(expr) {
        return val;
    }

    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;
    use tambear_primitives::recipe::*;

    #[test]
    fn mean_arithmetic_compiles_and_executes() {
        let plan = Plan::compile(&[&MEAN_ARITHMETIC]);
        assert_eq!(plan.n_accumulates(), 2); // sum + count
        let result = execute_cpu(&plan, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mean = result.results.get("mean_arithmetic").unwrap();
        assert!((mean - 3.0).abs() < 1e-14, "mean = {mean}");
    }

    #[test]
    fn variance_compiles_and_executes() {
        let plan = Plan::compile(&[&VARIANCE]);
        assert_eq!(plan.n_accumulates(), 3); // sum + sum_sq + count
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = execute_cpu(&plan, &data);
        let var = result.results.get("variance").unwrap();
        // Known variance of this dataset = 4.571...
        assert!((var - 4.571428571428571).abs() < 1e-10, "var = {var}");
    }

    #[test]
    fn mean_and_variance_fused_shares_two_steps() {
        let recipes: Vec<&Recipe> = vec![&MEAN_ARITHMETIC, &VARIANCE];
        let plan = Plan::compile(&recipes);

        // Without sharing: 2 + 3 = 5 accumulates
        // With sharing: sum + count shared → 3 unique
        assert_eq!(plan.n_accumulates(), 3,
            "mean(2) + variance(3) should fuse to 3 unique accumulates, not 5");
        assert_eq!(plan.n_saved(&recipes), 2,
            "should save 2 accumulates by sharing sum + count");

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = execute_cpu(&plan, &data);

        let mean = result.results.get("mean_arithmetic").unwrap();
        let var = result.results.get("variance").unwrap();
        assert!((mean - 3.0).abs() < 1e-14);
        assert!((var - 2.5).abs() < 1e-14);
    }

    #[test]
    fn four_recipes_fused() {
        let recipes: Vec<&Recipe> = vec![
            &MEAN_ARITHMETIC, &MEAN_GEOMETRIC, &MEAN_HARMONIC, &MEAN_QUADRATIC,
        ];
        let plan = Plan::compile(&recipes);

        // Unique accumulates:
        // sum (Value+Add), count (One+Add), log_sum (Ln+Add),
        // reciprocal_sum (Reciprocal+Add), sum_sq (ValueSq+Add)
        // count is shared across ALL four → 5 unique, not 8
        assert_eq!(plan.n_accumulates(), 5);
        assert_eq!(plan.n_saved(&recipes), 3, "count shared 3 times");

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = execute_cpu(&plan, &data);

        let arith = *result.results.get("mean_arithmetic").unwrap();
        let geo = *result.results.get("mean_geometric").unwrap();
        let harm = *result.results.get("mean_harmonic").unwrap();
        let quad = *result.results.get("mean_quadratic").unwrap();

        // Power mean inequality: harmonic ≤ geometric ≤ arithmetic ≤ quadratic
        assert!(harm <= geo + 1e-10, "harm {harm} > geo {geo}");
        assert!(geo <= arith + 1e-10, "geo {geo} > arith {arith}");
        assert!(arith <= quad + 1e-10, "arith {arith} > quad {quad}");
    }

    #[test]
    fn five_recipes_maximal_sharing() {
        let recipes: Vec<&Recipe> = vec![
            &MEAN_ARITHMETIC, &MEAN_GEOMETRIC, &MEAN_HARMONIC,
            &MEAN_QUADRATIC, &VARIANCE,
        ];
        let plan = Plan::compile(&recipes);

        // 5 unique accumulates: sum, count, log_sum, reciprocal_sum, sum_sq
        // Total without sharing: 2+2+2+2+3 = 11
        // Saved: 11 - 5 = 6
        assert_eq!(plan.n_accumulates(), 5);
        assert_eq!(plan.n_saved(&recipes), 6);
    }

    #[test]
    fn eval_simple_division() {
        let mut vars = HashMap::new();
        vars.insert("sum", 15.0);
        vars.insert("count", 5.0);
        let result = eval_gather_expr("sum / count", &vars);
        assert!((result - 3.0).abs() < 1e-14);
    }

    #[test]
    fn eval_sqrt() {
        let mut vars = HashMap::new();
        vars.insert("sum_sq", 50.0);
        vars.insert("count", 5.0);
        let result = eval_gather_expr("sqrt(sum_sq / count)", &vars);
        assert!((result - (10.0_f64).sqrt()).abs() < 1e-14);
    }

    #[test]
    fn eval_exp_of_division() {
        let mut vars = HashMap::new();
        vars.insert("log_sum", 0.0);
        vars.insert("count", 1.0);
        let result = eval_gather_expr("exp(log_sum / count)", &vars);
        assert!((result - 1.0).abs() < 1e-14);
    }
}
