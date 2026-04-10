//! Temporary oracle check for pinv — run to capture actual tambear output
//! cargo test --test pinv_oracle_check -- --nocapture
//! Delete after workup complete.

use tambear::linear_algebra::{Mat, pinv};

fn mat2(data: &[f64]) -> Mat {
    Mat::from_vec(2, 2, data.to_vec())
}

fn max_err(a: &Mat, b: &Mat) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..a.rows {
        for j in 0..a.cols {
            m = m.max((a.get(i, j) - b.get(i, j)).abs());
        }
    }
    m
}

fn mat_mul(a: &Mat, b: &Mat) -> Mat {
    tambear::linear_algebra::mat_mul(a, b)
}

#[test]
fn pinv_oracle_check() {
    // Case 1: identity
    let i3 = Mat::eye(3);
    let pi1 = pinv(&i3, None);
    let e1 = max_err(&pi1, &i3);
    println!("Case 1 (I_3x3): max_err={:.2e}", e1);

    // Case 2: 2x2 full rank
    let a2 = mat2(&[1.0, 2.0, 3.0, 4.0]);
    let pi2 = pinv(&a2, None);
    // numpy oracle: [[-2, 1], [1.5, -0.5]]
    let oracle2 = mat2(&[-2.0, 1.0, 1.5, -0.5]);
    let e2 = max_err(&pi2, &oracle2);
    println!("Case 2 (2x2 full rank): max_err={:.2e}", e2);
    println!("  pinv row0: {:.6}, {:.6}", pi2.get(0,0), pi2.get(0,1));
    println!("  pinv row1: {:.6}, {:.6}", pi2.get(1,0), pi2.get(1,1));

    // Case 3: rank-deficient 2x2
    let a3 = mat2(&[1.0, 2.0, 2.0, 4.0]);
    let pi3 = pinv(&a3, None);
    // numpy: [[0.04, 0.08], [0.08, 0.16]]
    let oracle3 = mat2(&[0.04, 0.08, 0.08, 0.16]);
    let e3 = max_err(&pi3, &oracle3);
    println!("Case 3 (rank-deficient): max_err={:.2e}", e3);
    println!("  pinv row0: {:.6}, {:.6}", pi3.get(0,0), pi3.get(0,1));

    // Case 6: diagonal with sv=[1e6, 5e-11] — tests relative rcond
    let a6 = mat2(&[1e6, 0.0, 0.0, 5e-11]);
    let pi6 = pinv(&a6, None);
    println!("Case 6 (diag(1e6, 5e-11)): pinv(0,0)={:.3e}, pinv(1,1)={:.3e}",
             pi6.get(0,0), pi6.get(1,1));
    // Expected: (1e-6, 0.0) with relative rcond; (1e-6, 2e10) with absolute 1e-12

    // Case 7: zero matrix
    let a7 = Mat::zeros(3, 3);
    let pi7 = pinv(&a7, None);
    println!("Case 7 (zero matrix): max={:.2e}", {
        let mut m = 0.0_f64;
        for i in 0..3 { for j in 0..3 { m = m.max(pi7.get(i,j).abs()); } }
        m
    });

    // Moore-Penrose property check on case 2
    let a2pi2 = mat_mul(&a2, &mat_mul(&pi2, &a2));
    let mp_err2 = max_err(&a2pi2, &a2);
    println!("Case 2 Moore-Penrose A A+ A = A: max_err={:.2e}", mp_err2);

    // Moore-Penrose on case 3
    let a3pi3 = mat_mul(&a3, &mat_mul(&pi3, &a3));
    let mp_err3 = max_err(&a3pi3, &a3);
    println!("Case 3 Moore-Penrose A A+ A = A: max_err={:.2e}", mp_err3);
}
