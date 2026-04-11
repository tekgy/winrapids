use super::*;
use crate::semiring::*;

#[test]
fn cumsum_via_additive() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = prefix_scan_inclusive::<Additive>(&data);
    assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn cumsum_exclusive() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let result = prefix_scan_exclusive::<Additive>(&data);
    assert_eq!(result, vec![0.0, 1.0, 3.0, 6.0]);
}

#[test]
fn running_min_via_tropical() {
    let data = vec![5.0, 3.0, 7.0, 1.0, 4.0];
    let result = prefix_scan_inclusive::<TropicalMinPlus>(&data);
    assert_eq!(result, vec![5.0, 3.0, 3.0, 1.0, 1.0]);
}

#[test]
fn running_max_via_tropical() {
    let data = vec![1.0, 4.0, 2.0, 5.0, 3.0];
    let result = prefix_scan_inclusive::<TropicalMaxPlus>(&data);
    assert_eq!(result, vec![1.0, 4.0, 4.0, 5.0, 5.0]);
}

#[test]
fn log_domain_forward_via_lse() {
    // Three log-probabilities: log(0.2), log(0.3), log(0.5)
    let data = vec![0.2_f64.ln(), 0.3_f64.ln(), 0.5_f64.ln()];
    let result = prefix_scan_inclusive::<LogSumExp>(&data);
    // After all three: lse = log(0.2 + 0.3 + 0.5) = log(1.0) = 0.0
    assert!((result[2] - 0.0).abs() < 1e-12);
}

#[test]
fn boolean_reachability() {
    let data = vec![false, false, true, false, true];
    let result = prefix_scan_inclusive::<Boolean>(&data);
    assert_eq!(result, vec![false, false, true, true, true]);
}

#[test]
fn reduce_additive() {
    assert_eq!(reduce::<Additive>(&[1.0, 2.0, 3.0]), 6.0);
}

#[test]
fn reduce_empty() {
    assert_eq!(reduce::<Additive>(&[]), 0.0); // zero
    assert_eq!(reduce::<TropicalMinPlus>(&[]), f64::INFINITY); // tropical zero
}

#[test]
fn segmented_scan() {
    let data = vec![1.0, 2.0, 3.0, 10.0, 20.0];
    let starts = vec![false, false, false, true, false]; // new segment at index 3
    let result = prefix_scan_segmented::<Additive>(&data, &starts);
    assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0, 30.0]);
}

#[test]
fn segmented_each_element() {
    let data = vec![5.0, 3.0, 7.0];
    let starts = vec![true, true, true]; // every element is its own segment
    let result = prefix_scan_segmented::<Additive>(&data, &starts);
    assert_eq!(result, vec![5.0, 3.0, 7.0]); // no accumulation
}
