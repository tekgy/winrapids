//! Gathers: how to read and combine accumulated results.
//!
//! After the accumulate pass(es), the results are scalar values
//! (for Grouping::All), per-group arrays (for ByKey), or per-element
//! arrays (for Prefix/Scan). The gather phase combines these into
//! the final output using a TBS expression over named variables.
//!
//! Gathers operate on accumulated SCALARS, not on the raw data.
//! They're O(1) for scalar results, O(k) for per-group, never O(n).

use crate::tbs::Expr;
use std::collections::HashMap;

/// A gather: compute a final result from accumulated values using a TBS expression.
#[derive(Debug, Clone)]
pub struct Gather {
    /// The TBS expression to evaluate. References accumulated values via Var("name").
    pub expr: Expr,
    /// Name for the output.
    pub output: String,
}

/// Execute a gather expression given accumulated values as variable bindings.
pub fn execute_gather(gather: &Gather, vars: &HashMap<String, f64>) -> f64 {
    crate::tbs::eval(&gather.expr, 0.0, 0.0, 0.0, vars)
}

// ═══════════════════════════════════════════════════════════════════
// Common gather expressions (convenience constructors)
// ═══════════════════════════════════════════════════════════════════

/// sum / count
pub fn mean_gather() -> Gather {
    Gather {
        expr: Expr::var("sum").div(Expr::var("count")),
        output: "mean".into(),
    }
}

/// (sum_sq - sum² / count) / (count - 1)
pub fn variance_gather() -> Gather {
    Gather {
        expr: Expr::var("sum_sq")
            .sub(Expr::var("sum").mul(Expr::var("sum")).div(Expr::var("count")))
            .div(Expr::var("count").sub(Expr::lit(1.0))),
        output: "variance".into(),
    }
}

/// sqrt(variance)
pub fn std_gather() -> Gather {
    Gather {
        expr: Expr::var("sum_sq")
            .sub(Expr::var("sum").mul(Expr::var("sum")).div(Expr::var("count")))
            .div(Expr::var("count").sub(Expr::lit(1.0)))
            .sqrt(),
        output: "std".into(),
    }
}

/// exp(log_sum / count)
pub fn geometric_mean_gather() -> Gather {
    Gather {
        expr: Expr::var("log_sum").div(Expr::var("count")).exp(),
        output: "geometric_mean".into(),
    }
}

/// count / reciprocal_sum
pub fn harmonic_mean_gather() -> Gather {
    Gather {
        expr: Expr::var("count").div(Expr::var("recip_sum")),
        output: "harmonic_mean".into(),
    }
}

/// Pearson r: (sum_xy - sum_x*sum_y/n) / sqrt((ss_x - sx²/n)(ss_y - sy²/n))
pub fn pearson_r_gather() -> Gather {
    let n = Expr::var("count");
    let num = Expr::var("sum_xy")
        .sub(Expr::var("sum_x").mul(Expr::var("sum_y")).div(n.clone()));
    let den_x = Expr::var("sum_sq_x")
        .sub(Expr::var("sum_x").mul(Expr::var("sum_x")).div(n.clone()));
    let den_y = Expr::var("sum_sq_y")
        .sub(Expr::var("sum_y").mul(Expr::var("sum_y")).div(n));
    Gather {
        expr: num.div(den_x.mul(den_y).sqrt()),
        output: "r".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean() {
        let mut vars = HashMap::new();
        vars.insert("sum".to_string(), 15.0);
        vars.insert("count".to_string(), 5.0);
        assert_eq!(execute_gather(&mean_gather(), &vars), 3.0);
    }

    #[test]
    fn variance() {
        let mut vars = HashMap::new();
        vars.insert("sum".to_string(), 15.0);
        vars.insert("sum_sq".to_string(), 55.0);
        vars.insert("count".to_string(), 5.0);
        assert!((execute_gather(&variance_gather(), &vars) - 2.5).abs() < 1e-14);
    }

    #[test]
    fn geometric_mean() {
        let mut vars = HashMap::new();
        vars.insert("log_sum".to_string(), 3.0_f64.ln() * 3.0);
        vars.insert("count".to_string(), 3.0);
        assert!((execute_gather(&geometric_mean_gather(), &vars) - 3.0).abs() < 1e-14);
    }

    #[test]
    fn pearson_perfect() {
        let mut vars = HashMap::new();
        vars.insert("sum_x".to_string(), 6.0);
        vars.insert("sum_y".to_string(), 12.0);
        vars.insert("sum_sq_x".to_string(), 14.0);
        vars.insert("sum_sq_y".to_string(), 56.0);
        vars.insert("sum_xy".to_string(), 28.0);
        vars.insert("count".to_string(), 3.0);
        assert!((execute_gather(&pearson_r_gather(), &vars) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn custom_gather_expression() {
        // Any TBS expression works as a gather — no enum needed
        let g = Gather {
            expr: Expr::var("a").mul(Expr::var("b")).add(Expr::var("c")),
            output: "result".into(),
        };
        let mut vars = HashMap::new();
        vars.insert("a".to_string(), 3.0);
        vars.insert("b".to_string(), 4.0);
        vars.insert("c".to_string(), 5.0);
        assert_eq!(execute_gather(&g, &vars), 17.0); // 3*4 + 5
    }
}
