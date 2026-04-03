//! Cholesky solve for small dense systems — thin wrapper over F02 (linear_algebra).
//!
//! All logic lives in `linear_algebra::{cholesky, cholesky_solve}`.
//! This module keeps the flat-slice interface used by `train/linear.rs`.

use crate::linear_algebra::{Mat, cholesky as la_cholesky, cholesky_solve};

/// Solve A x = b where A is symmetric positive definite (flat row-major, d×d).
/// Returns None if Cholesky fails (A is not positive definite).
pub fn solve(a: &[f64], b: &[f64], d: usize) -> Option<Vec<f64>> {
    let mat = Mat::from_vec(d, d, a.to_vec());
    let l = la_cholesky(&mat)?;
    Some(cholesky_solve(&l, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_2x2() {
        // A = [[4, 2], [2, 3]], b = [8, 8] → x = [1, 2]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let b = vec![8.0, 8.0];
        let x = solve(&a, &b, 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10, "x[0]={}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-10, "x[1]={}", x[1]);
    }

    #[test]
    fn solve_3x3() {
        // X'X = [[3,6],[6,14]], X'y = [6, 14] → beta = [0, 1]
        let xtx = vec![3.0, 6.0, 6.0, 14.0];
        let xty = vec![6.0, 14.0];
        let beta = solve(&xtx, &xty, 2).unwrap();
        assert!((beta[0] - 0.0).abs() < 1e-10, "beta[0]={}", beta[0]);
        assert!((beta[1] - 1.0).abs() < 1e-10, "beta[1]={}", beta[1]);
    }

    #[test]
    fn not_positive_definite_returns_none() {
        // A = [[-1, 0], [0, 1]] — not SPD
        let a = vec![-1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!(solve(&a, &b, 2).is_none());
    }
}
