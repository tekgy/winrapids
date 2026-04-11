use super::*;

#[test]
fn constant_input() {
    let data = vec![5.0; 10];
    let ema = mean_exponential_moving(&data, 0.3);
    for &v in &ema {
        assert!((v - 5.0).abs() < 1e-14);
    }
}

#[test]
fn converges_to_new_level() {
    // Step from 0 to 1 — EMA should approach 1
    let mut data = vec![0.0; 10];
    data.extend(vec![1.0; 100]);
    let ema = mean_exponential_moving(&data, 0.1);
    assert!(ema.last().unwrap() > &0.99);
}

#[test]
fn higher_alpha_faster_response() {
    let data = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let slow = mean_exponential_moving(&data, 0.1);
    let fast = mean_exponential_moving(&data, 0.9);
    // After the step at index 3, fast should be closer to 1 than slow
    assert!(fast[4] > slow[4]);
}

#[test]
fn period_conversion() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let from_period = mean_exponential_moving_period(&data, 9); // alpha = 2/10 = 0.2
    let from_alpha = mean_exponential_moving(&data, 0.2);
    for (a, b) in from_period.iter().zip(from_alpha.iter()) {
        assert!((a - b).abs() < 1e-14);
    }
}

#[test]
fn empty() {
    assert!(mean_exponential_moving(&[], 0.5).is_empty());
}

#[test]
fn bad_alpha() {
    let ema = mean_exponential_moving(&[1.0, 2.0], 0.0);
    assert!(ema[0].is_nan());
}
