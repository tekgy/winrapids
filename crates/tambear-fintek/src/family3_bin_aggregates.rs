//! Family 3 — Bin aggregates: OHLCV, counts, validity, variability.
//!
//! Covers fintek leaves: `ohlcv`, `counts`, `validity`, `variability`.

/// OHLCV features for a single bin.
///
/// Fintek columns (K02P01C02):
/// - open, high, low, close, volume, vwap (volume-weighted average price)
#[derive(Debug, Clone)]
pub struct OhlcvResult {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub vwap: f64,
}

impl OhlcvResult {
    pub fn nan() -> Self {
        Self {
            open: f64::NAN, high: f64::NAN, low: f64::NAN, close: f64::NAN,
            volume: f64::NAN, vwap: f64::NAN,
        }
    }
}

/// Compute OHLCV for a single bin.
///
/// `prices`: tick-level prices within the bin.
/// `sizes`: tick-level trade sizes within the bin (must be same length).
pub fn ohlcv(prices: &[f64], sizes: &[f64]) -> OhlcvResult {
    assert_eq!(prices.len(), sizes.len(), "prices and sizes must have same length");
    let n = prices.len();
    if n == 0 { return OhlcvResult::nan(); }

    let open = prices[0];
    let close = prices[n - 1];
    let mut high = prices[0];
    let mut low = prices[0];
    let mut volume = 0.0_f64;
    let mut notional_sum = 0.0_f64;

    for i in 0..n {
        let p = prices[i];
        if p > high { high = p; }
        if p < low { low = p; }
        volume += sizes[i];
        notional_sum += p * sizes[i];
    }

    let vwap = if volume > 0.0 { notional_sum / volume } else { f64::NAN };

    OhlcvResult { open, high, low, close, volume, vwap }
}

/// Tick counts features for a single bin.
///
/// - `tick_count`: number of ticks in the bin
/// - `unique_prices`: distinct price count (via sorted dedup)
/// - `upticks`: count of ticks where price strictly increased
/// - `downticks`: count where price strictly decreased
/// - `unchanged`: count where price was unchanged
#[derive(Debug, Clone)]
pub struct CountsResult {
    pub tick_count: f64,
    pub unique_prices: f64,
    pub upticks: f64,
    pub downticks: f64,
    pub unchanged: f64,
}

impl CountsResult {
    pub fn nan() -> Self {
        Self {
            tick_count: f64::NAN, unique_prices: f64::NAN,
            upticks: f64::NAN, downticks: f64::NAN, unchanged: f64::NAN,
        }
    }
}

/// Compute tick-count features.
pub fn counts(prices: &[f64]) -> CountsResult {
    let n = prices.len();
    if n == 0 { return CountsResult::nan(); }

    // Unique prices
    let mut sorted = prices.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut unique = 1.0_f64;
    for i in 1..n {
        if sorted[i] != sorted[i - 1] {
            unique += 1.0;
        }
    }

    // Up/down/unchanged
    let mut up = 0.0_f64;
    let mut down = 0.0_f64;
    let mut flat = 0.0_f64;
    for i in 1..n {
        let d = prices[i] - prices[i - 1];
        if d > 0.0 { up += 1.0; }
        else if d < 0.0 { down += 1.0; }
        else { flat += 1.0; }
    }

    CountsResult {
        tick_count: n as f64,
        unique_prices: unique,
        upticks: up,
        downticks: down,
        unchanged: flat,
    }
}

/// Data validity metrics for a single bin.
///
/// - `n_nan`: count of NaN values in prices
/// - `n_neg`: count of non-positive prices
/// - `n_zero_size`: count of ticks with zero size
/// - `completeness`: 1 - (n_nan + n_neg) / n, in [0, 1]
#[derive(Debug, Clone)]
pub struct ValidityResult {
    pub n_nan: f64,
    pub n_neg: f64,
    pub n_zero_size: f64,
    pub completeness: f64,
}

impl ValidityResult {
    pub fn nan() -> Self {
        Self { n_nan: f64::NAN, n_neg: f64::NAN, n_zero_size: f64::NAN, completeness: f64::NAN }
    }
}

pub fn validity(prices: &[f64], sizes: &[f64]) -> ValidityResult {
    assert_eq!(prices.len(), sizes.len());
    let n = prices.len();
    if n == 0 { return ValidityResult::nan(); }

    let mut n_nan = 0.0_f64;
    let mut n_neg = 0.0_f64;
    let mut n_zero_size = 0.0_f64;
    for i in 0..n {
        if prices[i].is_nan() { n_nan += 1.0; }
        else if prices[i] <= 0.0 { n_neg += 1.0; }
        if sizes[i] <= 0.0 { n_zero_size += 1.0; }
    }
    let completeness = 1.0 - (n_nan + n_neg) / n as f64;

    ValidityResult { n_nan, n_neg, n_zero_size, completeness }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ohlcv_basic() {
        let prices = vec![100.0, 101.0, 99.0, 102.0, 100.5];
        let sizes  = vec![10.0,  20.0,  15.0,  5.0,  30.0];
        let r = ohlcv(&prices, &sizes);
        assert_eq!(r.open, 100.0);
        assert_eq!(r.close, 100.5);
        assert_eq!(r.high, 102.0);
        assert_eq!(r.low, 99.0);
        assert_eq!(r.volume, 80.0);
        // vwap = (100*10 + 101*20 + 99*15 + 102*5 + 100.5*30) / 80
        //     = (1000 + 2020 + 1485 + 510 + 3015) / 80 = 8030 / 80 = 100.375
        assert!((r.vwap - 100.375).abs() < 1e-10);
    }

    #[test]
    fn ohlcv_empty() {
        let r = ohlcv(&[], &[]);
        assert!(r.open.is_nan());
    }

    #[test]
    fn ohlcv_single() {
        let r = ohlcv(&[100.0], &[5.0]);
        assert_eq!(r.open, 100.0);
        assert_eq!(r.close, 100.0);
        assert_eq!(r.high, 100.0);
        assert_eq!(r.low, 100.0);
        assert_eq!(r.volume, 5.0);
        assert_eq!(r.vwap, 100.0);
    }

    #[test]
    fn counts_basic() {
        let prices = vec![100.0, 101.0, 101.0, 100.0, 102.0];
        let r = counts(&prices);
        assert_eq!(r.tick_count, 5.0);
        assert_eq!(r.unique_prices, 3.0); // {100, 101, 102}
        assert_eq!(r.upticks, 2.0);    // 100→101, 100→102
        assert_eq!(r.downticks, 1.0);  // 101→100
        assert_eq!(r.unchanged, 1.0);  // 101→101
    }

    #[test]
    fn validity_basic() {
        let prices = vec![100.0, f64::NAN, -1.0, 101.0, 0.0];
        let sizes  = vec![10.0,  5.0,       2.0,  0.0,   0.0];
        let r = validity(&prices, &sizes);
        assert_eq!(r.n_nan, 1.0);
        assert_eq!(r.n_neg, 2.0); // -1 and 0
        assert_eq!(r.n_zero_size, 2.0);
        assert!((r.completeness - 2.0/5.0).abs() < 1e-12);
    }

    #[test]
    fn validity_all_good() {
        let prices = vec![100.0, 101.0, 102.0];
        let sizes = vec![1.0, 2.0, 3.0];
        let r = validity(&prices, &sizes);
        assert_eq!(r.n_nan, 0.0);
        assert_eq!(r.n_neg, 0.0);
        assert_eq!(r.n_zero_size, 0.0);
        assert_eq!(r.completeness, 1.0);
    }
}
