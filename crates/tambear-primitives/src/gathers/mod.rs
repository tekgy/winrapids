//! Gathers: how to read and combine accumulated results.
//!
//! After the accumulate pass(es), the results are scalar values
//! (for Grouping::All), per-group arrays (for ByKey), or per-element
//! arrays (for Prefix/Scan). The gather phase combines these into
//! the final output.
//!
//! Gathers operate on accumulated SCALARS, not on the raw data.
//! They're O(1) for scalar results, O(k) for per-group, never O(n).
//! The cost is always in the accumulate phase.

use std::collections::HashMap;

/// A gather expression: compute a final result from accumulated values.
#[derive(Debug, Clone)]
pub struct Gather {
    /// Named accumulated values this gather reads.
    pub inputs: Vec<String>,
    /// The computation to perform.
    pub computation: GatherComputation,
    /// Name for the output.
    pub output: String,
}

/// What computation to perform over accumulated values.
#[derive(Debug, Clone)]
pub enum GatherComputation {
    /// Return one accumulated value directly.
    Passthrough(String),
    /// a / b
    Divide(String, String),
    /// sqrt(expr)
    Sqrt(Box<GatherComputation>),
    /// exp(expr)
    Exp(Box<GatherComputation>),
    /// a - b
    Subtract(String, String),
    /// a * b
    Multiply(String, String),
    /// (sum_sq - sum * sum / count) / (count - 1)
    /// Variance formula — common enough to be a named computation.
    VarianceFormula { sum: String, sum_sq: String, count: String },
    /// Pearson r formula
    PearsonFormula {
        sum_x: String, sum_y: String,
        sum_sq_x: String, sum_sq_y: String,
        sum_xy: String, count: String,
    },
    /// General expression string (for the evaluator).
    Expression(String),
}

/// Execute a gather computation given accumulated values.
pub fn execute_gather(
    gather: &Gather,
    acc_values: &HashMap<String, f64>,
) -> f64 {
    eval_computation(&gather.computation, acc_values)
}

fn eval_computation(comp: &GatherComputation, vars: &HashMap<String, f64>) -> f64 {
    match comp {
        GatherComputation::Passthrough(name) => {
            *vars.get(name.as_str()).unwrap_or(&f64::NAN)
        }
        GatherComputation::Divide(a, b) => {
            let va = vars.get(a.as_str()).unwrap_or(&f64::NAN);
            let vb = vars.get(b.as_str()).unwrap_or(&f64::NAN);
            va / vb
        }
        GatherComputation::Sqrt(inner) => {
            eval_computation(inner, vars).sqrt()
        }
        GatherComputation::Exp(inner) => {
            eval_computation(inner, vars).exp()
        }
        GatherComputation::Subtract(a, b) => {
            let va = vars.get(a.as_str()).unwrap_or(&f64::NAN);
            let vb = vars.get(b.as_str()).unwrap_or(&f64::NAN);
            va - vb
        }
        GatherComputation::Multiply(a, b) => {
            let va = vars.get(a.as_str()).unwrap_or(&f64::NAN);
            let vb = vars.get(b.as_str()).unwrap_or(&f64::NAN);
            va * vb
        }
        GatherComputation::VarianceFormula { sum, sum_sq, count } => {
            let s = vars.get(sum.as_str()).unwrap_or(&f64::NAN);
            let ss = vars.get(sum_sq.as_str()).unwrap_or(&f64::NAN);
            let n = vars.get(count.as_str()).unwrap_or(&f64::NAN);
            (ss - s * s / n) / (n - 1.0)
        }
        GatherComputation::PearsonFormula { sum_x, sum_y, sum_sq_x, sum_sq_y, sum_xy, count } => {
            let sx = vars.get(sum_x.as_str()).unwrap_or(&f64::NAN);
            let sy = vars.get(sum_y.as_str()).unwrap_or(&f64::NAN);
            let ssx = vars.get(sum_sq_x.as_str()).unwrap_or(&f64::NAN);
            let ssy = vars.get(sum_sq_y.as_str()).unwrap_or(&f64::NAN);
            let sxy = vars.get(sum_xy.as_str()).unwrap_or(&f64::NAN);
            let n = vars.get(count.as_str()).unwrap_or(&f64::NAN);
            let num = sxy - sx * sy / n;
            let den_x = ssx - sx * sx / n;
            let den_y = ssy - sy * sy / n;
            let den = (den_x * den_y).sqrt();
            if den.abs() < 1e-300 { f64::NAN } else { num / den }
        }
        GatherComputation::Expression(_expr) => {
            // TODO: general expression evaluator
            f64::NAN
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_gather() {
        let mut vars = HashMap::new();
        vars.insert("sum".to_string(), 15.0);
        vars.insert("count".to_string(), 5.0);
        let g = Gather {
            inputs: vec!["sum".into(), "count".into()],
            computation: GatherComputation::Divide("sum".into(), "count".into()),
            output: "mean".into(),
        };
        assert_eq!(execute_gather(&g, &vars), 3.0);
    }

    #[test]
    fn variance_gather() {
        let mut vars = HashMap::new();
        vars.insert("sum".to_string(), 15.0);
        vars.insert("sum_sq".to_string(), 55.0);
        vars.insert("count".to_string(), 5.0);
        let g = Gather {
            inputs: vec!["sum".into(), "sum_sq".into(), "count".into()],
            computation: GatherComputation::VarianceFormula {
                sum: "sum".into(), sum_sq: "sum_sq".into(), count: "count".into(),
            },
            output: "variance".into(),
        };
        assert!((execute_gather(&g, &vars) - 2.5).abs() < 1e-14);
    }

    #[test]
    fn geometric_mean_gather() {
        let mut vars = HashMap::new();
        vars.insert("log_sum".to_string(), 3.0_f64.ln() * 3.0); // ln(3)*3 = sum of ln(3,3,3)
        vars.insert("count".to_string(), 3.0);
        let g = Gather {
            inputs: vec!["log_sum".into(), "count".into()],
            computation: GatherComputation::Exp(
                Box::new(GatherComputation::Divide("log_sum".into(), "count".into()))
            ),
            output: "geometric_mean".into(),
        };
        assert!((execute_gather(&g, &vars) - 3.0).abs() < 1e-14);
    }

    #[test]
    fn pearson_perfect_correlation() {
        let mut vars = HashMap::new();
        // x = [1,2,3], y = [2,4,6] (y = 2x)
        vars.insert("sum_x".to_string(), 6.0);
        vars.insert("sum_y".to_string(), 12.0);
        vars.insert("sum_sq_x".to_string(), 14.0);
        vars.insert("sum_sq_y".to_string(), 56.0);
        vars.insert("sum_xy".to_string(), 28.0);
        vars.insert("count".to_string(), 3.0);
        let g = Gather {
            inputs: vec!["sum_x".into(), "sum_y".into(), "sum_sq_x".into(),
                         "sum_sq_y".into(), "sum_xy".into(), "count".into()],
            computation: GatherComputation::PearsonFormula {
                sum_x: "sum_x".into(), sum_y: "sum_y".into(),
                sum_sq_x: "sum_sq_x".into(), sum_sq_y: "sum_sq_y".into(),
                sum_xy: "sum_xy".into(), count: "count".into(),
            },
            output: "r".into(),
        };
        assert!((execute_gather(&g, &vars) - 1.0).abs() < 1e-14);
    }
}
