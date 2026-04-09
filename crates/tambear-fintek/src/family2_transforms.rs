//! Family 2 — Returns & pointwise transforms.
//!
//! All DIRECT mappings: pure pointwise operations on price/size arrays.
//! Embarrassingly parallel — ideal for scatter-based JIT compilation.
//!
//! Covers fintek leaves: `returns`, `log_transform`, `sqrt_transform`,
//! `reciprocal`, `notional`, `delta_value`, `delta_log`, `delta_percent`,
//! `delta_direction`, `elapsed`, `cyclical`.

/// Log returns: ln(p_t / p_{t-1}).
///
/// Returns a vector of length `prices.len() - 1`. The i-th entry is
/// ln(prices[i+1] / prices[i]). Returns NaN for non-positive prices.
///
/// Equivalent to fintek's `returns.rs` log-return computation.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 { return vec![]; }
    prices.windows(2).map(|w| {
        if w[0] > 0.0 && w[1] > 0.0 { (w[1] / w[0]).ln() } else { f64::NAN }
    }).collect()
}

/// Simple percent returns: (p_t - p_{t-1}) / p_{t-1}.
pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 { return vec![]; }
    prices.windows(2).map(|w| {
        if w[0] != 0.0 { (w[1] - w[0]) / w[0] } else { f64::NAN }
    }).collect()
}

/// Natural log transform, with NaN for non-positive inputs.
///
/// Equivalent to `log_transform.rs`.
pub fn log_transform(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| if v > 0.0 { v.ln() } else { f64::NAN }).collect()
}

/// Square-root transform.
///
/// Equivalent to `sqrt_transform.rs`. Returns NaN for negative inputs.
pub fn sqrt_transform(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| if v >= 0.0 { v.sqrt() } else { f64::NAN }).collect()
}

/// Reciprocal: 1/x. Returns NaN when |x| < 1e-300.
///
/// Equivalent to `reciprocal.rs`.
pub fn reciprocal(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| if v.abs() > 1e-300 { 1.0 / v } else { f64::NAN }).collect()
}

/// Notional: price × size (pointwise product).
///
/// Equivalent to `notional.rs`.
pub fn notional(price: &[f64], size: &[f64]) -> Vec<f64> {
    assert_eq!(price.len(), size.len(), "price and size must have same length");
    price.iter().zip(size.iter()).map(|(&p, &s)| p * s).collect()
}

/// Lag difference: x_t - x_{t-lag}.
///
/// Returns a vector of length `x.len()` with NaN for the first `lag` entries.
pub fn delta_value(x: &[f64], lag: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; x.len()];
    for i in lag..x.len() {
        out[i] = x[i] - x[i - lag];
    }
    out
}

/// Lag log difference: ln(x_t) - ln(x_{t-lag}).
pub fn delta_log(x: &[f64], lag: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; x.len()];
    for i in lag..x.len() {
        if x[i] > 0.0 && x[i - lag] > 0.0 {
            out[i] = x[i].ln() - x[i - lag].ln();
        }
    }
    out
}

/// Lag percent difference: (x_t - x_{t-lag}) / x_{t-lag}.
pub fn delta_percent(x: &[f64], lag: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; x.len()];
    for i in lag..x.len() {
        if x[i - lag] != 0.0 {
            out[i] = (x[i] - x[i - lag]) / x[i - lag];
        }
    }
    out
}

/// Sign of the lag difference: -1, 0, +1.
pub fn delta_direction(x: &[f64], lag: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; x.len()];
    for i in lag..x.len() {
        let d = x[i] - x[i - lag];
        out[i] = if d > 0.0 { 1.0 } else if d < 0.0 { -1.0 } else { 0.0 };
    }
    out
}

/// Elapsed minutes within the current day.
///
/// `timestamps_ns`: nanoseconds since epoch (or any nanosecond basis).
/// Returns minutes in [0, 1440). Matches fintek's `Elapsed` leaf
/// (`leaves/elapsed.rs`, K01P02C02R04) which emits F32 minutes.
pub fn elapsed_minutes_of_day(timestamps_ns: &[u64]) -> Vec<f32> {
    const DAY_NS: u64 = 86_400_000_000_000;
    const MINUTE_NS: f64 = 60.0 * 1e9;
    timestamps_ns
        .iter()
        .map(|&t| (((t % DAY_NS) as f64) / MINUTE_NS) as f32)
        .collect()
}

/// Deprecated alias kept for callers that haven't migrated. Returns seconds (f64).
/// Prefer [`elapsed_minutes_of_day`] which matches fintek's contract exactly.
#[deprecated(note = "use elapsed_minutes_of_day to match fintek's Elapsed leaf")]
pub fn elapsed_seconds_of_day(timestamps_ns: &[u64]) -> Vec<f64> {
    const DAY_NS: u64 = 86_400_000_000_000;
    timestamps_ns.iter().map(|&t| ((t % DAY_NS) as f64) / 1e9).collect()
}

/// Cyclical encoding: sin and cos of a time-of-day angle.
///
/// Returns (sin_values, cos_values) each of length `timestamps_ns.len()`.
/// The angle is 2π · (time_of_day / 86400s).
pub fn cyclical_time_of_day(timestamps_ns: &[u64]) -> (Vec<f64>, Vec<f64>) {
    const DAY_NS: u64 = 86_400_000_000_000;
    let two_pi = std::f64::consts::TAU;
    let mut sin_v = Vec::with_capacity(timestamps_ns.len());
    let mut cos_v = Vec::with_capacity(timestamps_ns.len());
    for &t in timestamps_ns {
        let frac = (t % DAY_NS) as f64 / DAY_NS as f64;
        let angle = two_pi * frac;
        sin_v.push(angle.sin());
        cos_v.push(angle.cos());
    }
    (sin_v, cos_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_returns_basic() {
        let prices = vec![100.0, 105.0, 110.25];
        let lr = log_returns(&prices);
        assert_eq!(lr.len(), 2);
        assert!((lr[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-12);
        assert!((lr[1] - (110.25_f64 / 105.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn log_returns_zero_price() {
        let prices = vec![100.0, 0.0, 100.0];
        let lr = log_returns(&prices);
        assert!(lr[0].is_nan());
        assert!(lr[1].is_nan());
    }

    #[test]
    fn log_returns_too_short() {
        assert!(log_returns(&[]).is_empty());
        assert!(log_returns(&[1.0]).is_empty());
    }

    #[test]
    fn simple_returns_basic() {
        let prices = vec![100.0, 110.0];
        let r = simple_returns(&prices);
        assert!((r[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn log_transform_positive() {
        let x = vec![1.0, std::f64::consts::E, 10.0];
        let out = log_transform(&x);
        assert!((out[0] - 0.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log_transform_negative_nan() {
        let x = vec![-1.0, 0.0, 5.0];
        let out = log_transform(&x);
        assert!(out[0].is_nan());
        assert!(out[1].is_nan()); // log(0) = -inf but we treat as NaN (non-positive guard)
        assert!((out[2] - 5.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn sqrt_transform_basic() {
        let x = vec![4.0, 9.0, -1.0];
        let out = sqrt_transform(&x);
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], 3.0);
        assert!(out[2].is_nan());
    }

    #[test]
    fn reciprocal_basic() {
        let x = vec![2.0, -4.0, 0.0];
        let out = reciprocal(&x);
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], -0.25);
        assert!(out[2].is_nan());
    }

    #[test]
    fn notional_basic() {
        let price = vec![100.0, 200.0];
        let size = vec![10.0, 5.0];
        let n = notional(&price, &size);
        assert_eq!(n, vec![1000.0, 1000.0]);
    }

    #[test]
    fn delta_value_lag1() {
        let x = vec![10.0, 12.0, 11.0, 15.0];
        let d = delta_value(&x, 1);
        assert!(d[0].is_nan());
        assert_eq!(d[1], 2.0);
        assert_eq!(d[2], -1.0);
        assert_eq!(d[3], 4.0);
    }

    #[test]
    fn delta_log_lag2() {
        let x = vec![1.0, 2.0, 4.0, 8.0];
        let d = delta_log(&x, 2);
        assert!(d[0].is_nan() && d[1].is_nan());
        // ln(4/1) = ln(4)
        assert!((d[2] - 4.0_f64.ln()).abs() < 1e-12);
        assert!((d[3] - 4.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn delta_direction_basic() {
        let x = vec![10.0, 11.0, 10.0, 10.0];
        let d = delta_direction(&x, 1);
        assert!(d[0].is_nan());
        assert_eq!(d[1], 1.0);  // up
        assert_eq!(d[2], -1.0); // down
        assert_eq!(d[3], 0.0);  // unchanged
    }

    #[test]
    fn elapsed_minutes_basic() {
        // Midnight = 0
        let t0: u64 = 1_234_567_890 * 1_000_000_000; // some epoch seconds
        let rounded_to_midnight = (t0 / (86_400 * 1_000_000_000)) * (86_400 * 1_000_000_000);
        let e = elapsed_minutes_of_day(&[rounded_to_midnight]);
        assert!(e[0] < 1e-3);
        // 12 hours later = 720 minutes
        let noon = rounded_to_midnight + 43_200 * 1_000_000_000;
        let e = elapsed_minutes_of_day(&[noon]);
        assert!((e[0] - 720.0).abs() < 1e-2,
            "noon should be 720 minutes since midnight, got {}", e[0]);
        // 23:59 = 1439 minutes
        let almost_midnight = rounded_to_midnight + (23 * 3600 + 59 * 60) as u64 * 1_000_000_000;
        let e = elapsed_minutes_of_day(&[almost_midnight]);
        assert!((e[0] - 1439.0).abs() < 1e-2);
    }

    #[test]
    fn cyclical_basic() {
        const DAY_NS: u64 = 86_400_000_000_000;
        // At quarter day: angle = π/2 → sin=1, cos=0
        let t = DAY_NS / 4;
        let (s, c) = cyclical_time_of_day(&[t]);
        assert!((s[0] - 1.0).abs() < 1e-10);
        assert!(c[0].abs() < 1e-10);
        // At half day: angle = π → sin=0, cos=-1
        let t = DAY_NS / 2;
        let (s, c) = cyclical_time_of_day(&[t]);
        assert!(s[0].abs() < 1e-10);
        assert!((c[0] + 1.0).abs() < 1e-10);
    }
}
