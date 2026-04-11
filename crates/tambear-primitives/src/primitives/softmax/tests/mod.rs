use super::*;

#[test]
fn sums_to_one() {
    let x = vec![1.0, 2.0, 3.0];
    let s = softmax(&x);
    let total: f64 = s.iter().sum();
    assert!((total - 1.0).abs() < 1e-14);
}

#[test]
fn largest_input_gets_largest_probability() {
    let x = vec![0.1, 5.0, 0.3, 0.2];
    let s = softmax(&x);
    let argmax = s.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0;
    assert_eq!(argmax, 1);
}

#[test]
fn uniform_input_gives_uniform_output() {
    let x = vec![1.0, 1.0, 1.0, 1.0];
    let s = softmax(&x);
    for &p in &s {
        assert!((p - 0.25).abs() < 1e-14);
    }
}

#[test]
fn log_softmax_consistency() {
    let x = vec![1.0, 2.0, 3.0];
    let ls = log_softmax(&x);
    let s = softmax(&x);
    for (lsi, si) in ls.iter().zip(s.iter()) {
        assert!((lsi - si.ln()).abs() < 1e-14);
    }
}

#[test]
fn log_softmax_sums_via_lse() {
    // log_softmax values should lse to 0 (= log(1))
    let x = vec![1.0, 2.0, 3.0];
    let ls = log_softmax(&x);
    let total = crate::log_sum_exp::log_sum_exp(&ls);
    assert!(total.abs() < 1e-14, "lse of log_softmax should be 0, got {total}");
}

#[test]
fn empty() {
    assert!(softmax(&[]).is_empty());
    assert!(log_softmax(&[]).is_empty());
}

#[test]
fn nan_propagates() {
    let s = softmax(&[1.0, f64::NAN, 3.0]);
    assert!(s.iter().all(|v| v.is_nan()));
}
