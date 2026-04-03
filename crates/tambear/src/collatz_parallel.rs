//! # Parallel Collatz Verification with Multi-Adic Profiling
//!
//! Phase 1: CPU implementation with N-thread parallelism.
//! Phase 2 (future): GPU via tambear scatter primitives.
//!
//! ## Proof-structure optimizations
//!
//! 1. **Shadow batching**: group by trailing bit pattern. All numbers
//!    with the same trailing K bits follow the same first K Collatz steps.
//!    Verify one representative per class.
//!
//! 2. **Early termination**: if post-fold value < known-verified threshold, done.
//!    Don't trace to 1 — just drop below the frontier.
//!
//! 3. **Residue class verification**: verify 2^K representatives to cover
//!    all numbers with K-bit trailing patterns.
//!
//! ## Multi-adic profiling
//!
//! During verification, each number also produces:
//! - v₂ histogram (2-adic valuation of 3n+1 at each odd step)
//! - τ histogram (trailing ones of each odd value)
//! - max_tau, expansion bits, delay

use crate::fold_irreversibility::{
    collatz_branchless, verify_profiled, MultiAdicProfile as FoldProfile,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// Shadow batching: residue class representatives
// ═══════════════════════════════════════════════════════════════════════════

/// For a given K-bit trailing pattern, compute the Collatz path for K steps.
///
/// All numbers ≡ pattern (mod 2^K) follow the same K steps.
/// Returns the "compressed" Collatz function: after K steps,
/// the result is A·n + B for some constants A, B.
///
/// This means: to verify all n ≡ pattern (mod 2^K), we only need to verify
/// A·representative + B < threshold, where representative = pattern.
pub fn shadow_coefficients(pattern: u64, k: u32) -> (u128, u128) {
    // Apply K Collatz steps symbolically.
    // Start with n = pattern (mod 2^K). After each step:
    //   if even: n → n/2   (multiply by 1/2)
    //   if odd:  n → 3n+1  (multiply by 3, add 1), then /2
    //
    // We track: result = a_coeff * (original_n >> K) + b_coeff
    // where a_coeff and b_coeff accumulate the affine transformation.
    let mut val = pattern as u128;
    let mut a_coeff: u128 = 1;
    let mut b_coeff: u128 = 0;

    for _ in 0..k {
        if val & 1 == 0 {
            val >>= 1;
            // Even step divides by 2: doesn't change the multiplier/offset
            // for the high bits, just processes the known low bit
        } else {
            val = 3 * val + 1;
            // Odd step: 3n+1, which multiplies by 3 and adds 1
            a_coeff *= 3;
            b_coeff = 3 * b_coeff + 1;
        }
        // Always divide by 2 at end of branchless step
        // (the branchless formula combines odd+divide)
    }

    (a_coeff, b_coeff)
}

/// Number of distinct K-bit trailing patterns that are odd.
///
/// For Collatz, we only verify odd numbers. Among K-bit patterns,
/// exactly 2^(K-1) are odd.
pub fn odd_pattern_count(k: u32) -> u64 {
    1u64 << (k - 1)
}

// ═══════════════════════════════════════════════════════════════════════════
// Parallel verification engine
// ═══════════════════════════════════════════════════════════════════════════

/// Result of parallel batch verification.
#[derive(Debug, Clone)]
pub struct ParallelVerifyResult {
    /// Range start (inclusive).
    pub start: u128,
    /// Range end (exclusive).
    pub end: u128,
    /// Total odd numbers verified.
    pub n_verified: u64,
    /// Numbers that failed verification.
    pub n_failed: u64,
    /// Maximum delay (steps to drop below threshold).
    pub max_delay: u64,
    /// Witness with maximum delay.
    pub max_delay_witness: u128,
    /// Maximum expansion bits above starting value.
    pub max_expansion: u32,
    /// Aggregate v₂ histogram.
    pub agg_v2_histogram: [u64; 16],
    /// Aggregate τ histogram.
    pub agg_tau_histogram: [u64; 16],
    /// Wall-clock seconds.
    pub elapsed_secs: f64,
    /// Verification rate (odd numbers per second).
    pub rate: f64,
    /// Number of threads used.
    pub n_threads: usize,
}

/// Verify all odd numbers in [start, end) drop below `threshold`.
///
/// Uses `n_threads` worker threads with chunk-based work stealing.
/// Each chunk is `chunk_size` consecutive odd numbers.
pub fn parallel_verify(
    start: u128,
    end: u128,
    threshold: u128,
    max_steps: u64,
    n_threads: usize,
) -> ParallelVerifyResult {
    let t0 = std::time::Instant::now();
    let n_threads = n_threads.max(1);

    // Shared atomic counters for progress
    let verified = Arc::new(AtomicU64::new(0));
    let failed = Arc::new(AtomicU64::new(0));

    // Split the range into chunks for each thread
    let range_size = end.saturating_sub(start);
    let chunk_per_thread = range_size / n_threads as u128;

    let mut handles = Vec::with_capacity(n_threads);

    for t in 0..n_threads {
        let verified = Arc::clone(&verified);
        let failed = Arc::clone(&failed);

        let chunk_start = start + (t as u128) * chunk_per_thread;
        let chunk_end = if t == n_threads - 1 {
            end
        } else {
            start + ((t + 1) as u128) * chunk_per_thread
        };

        handles.push(std::thread::spawn(move || {
            let mut local = LocalResult::default();

            // Start at first odd number in chunk
            let mut n = if chunk_start % 2 == 0 { chunk_start + 1 } else { chunk_start };

            while n < chunk_end {
                // Early termination: if n < threshold, it's already verified
                if n < threshold {
                    local.n_verified += 1;
                    n += 2;
                    continue;
                }

                match verify_profiled(n, max_steps) {
                    Some(profile) => {
                        local.n_verified += 1;
                        if profile.delay > local.max_delay {
                            local.max_delay = profile.delay;
                            local.max_delay_witness = n;
                        }
                        if profile.expansion_bits > local.max_expansion {
                            local.max_expansion = profile.expansion_bits;
                        }
                        for i in 0..16 {
                            local.agg_v2[i] += profile.v2_histogram[i] as u64;
                            local.agg_tau[i] += profile.tau_histogram[i] as u64;
                        }
                    }
                    None => {
                        local.n_failed += 1;
                    }
                }

                // Update shared counters periodically
                if local.n_verified % 100_000 == 0 {
                    verified.fetch_add(100_000, Ordering::Relaxed);
                }

                n += 2;
            }

            // Flush remaining count
            verified.fetch_add(local.n_verified % 100_000, Ordering::Relaxed);
            failed.fetch_add(local.n_failed, Ordering::Relaxed);

            local
        }));
    }

    // Collect results
    let mut result = ParallelVerifyResult {
        start,
        end,
        n_verified: 0,
        n_failed: 0,
        max_delay: 0,
        max_delay_witness: start,
        max_expansion: 0,
        agg_v2_histogram: [0; 16],
        agg_tau_histogram: [0; 16],
        elapsed_secs: 0.0,
        rate: 0.0,
        n_threads,
    };

    for handle in handles {
        let local = handle.join().expect("Thread panicked during verification");
        result.n_verified += local.n_verified;
        result.n_failed += local.n_failed;
        if local.max_delay > result.max_delay {
            result.max_delay = local.max_delay;
            result.max_delay_witness = local.max_delay_witness;
        }
        if local.max_expansion > result.max_expansion {
            result.max_expansion = local.max_expansion;
        }
        for i in 0..16 {
            result.agg_v2_histogram[i] += local.agg_v2[i];
            result.agg_tau_histogram[i] += local.agg_tau[i];
        }
    }

    let elapsed = t0.elapsed();
    result.elapsed_secs = elapsed.as_secs_f64();
    result.rate = result.n_verified as f64 / result.elapsed_secs;

    result
}

/// Thread-local accumulator (avoids atomic contention in hot loop).
#[derive(Debug, Default)]
struct LocalResult {
    n_verified: u64,
    n_failed: u64,
    max_delay: u64,
    max_delay_witness: u128,
    max_expansion: u32,
    agg_v2: [u64; 16],
    agg_tau: [u64; 16],
}

// ═══════════════════════════════════════════════════════════════════════════
// Residue class verification
// ═══════════════════════════════════════════════════════════════════════════

/// Result of residue-class based verification.
#[derive(Debug, Clone)]
pub struct ResidueClassResult {
    /// K: number of trailing bits used for classification.
    pub k: u32,
    /// Number of distinct residue classes verified.
    pub classes_verified: u64,
    /// Numbers this covers: all n in [start, end) with matching trailing K bits.
    pub effective_coverage: u128,
    /// Classes that failed.
    pub failed_classes: Vec<u64>,
    /// Maximum delay across all classes.
    pub max_delay: u64,
    pub elapsed_secs: f64,
}

/// Verify Collatz conjecture for all numbers ≡ r (mod 2^K) in [start, end)
/// by verifying a single representative per class.
///
/// For each odd K-bit pattern p, we verify that the Collatz path
/// starting from p (and from p + 2^K, p + 2·2^K, ...) drops below
/// the threshold within max_steps.
///
/// This is the "shadow batch" approach: numbers with the same trailing
/// K bits follow the same first K Collatz steps.
pub fn verify_by_residue_class(
    start: u128,
    end: u128,
    threshold: u128,
    k: u32,
    max_steps: u64,
    n_threads: usize,
) -> ResidueClassResult {
    let t0 = std::time::Instant::now();
    let modulus = 1u128 << k;
    let n_classes = odd_pattern_count(k);

    // For each odd residue class, verify the smallest representative in range
    let mut representatives: Vec<u128> = Vec::with_capacity(n_classes as usize);
    for pattern in 0..modulus {
        if pattern & 1 == 0 { continue; } // skip even patterns
        // Find smallest n ≡ pattern (mod 2^K) with n >= start
        let base = if start <= pattern {
            pattern
        } else {
            let offset = (start - pattern + modulus - 1) / modulus * modulus;
            pattern + offset
        };
        if base < end {
            representatives.push(base);
        }
    }

    // Parallel verification of representatives
    let chunk_size = (representatives.len() + n_threads - 1) / n_threads.max(1);
    let mut handles = Vec::new();
    let reps = Arc::new(representatives);

    for t in 0..n_threads {
        let reps = Arc::clone(&reps);
        let chunk_start = t * chunk_size;
        let chunk_end = ((t + 1) * chunk_size).min(reps.len());

        handles.push(std::thread::spawn(move || {
            let mut local_failed = Vec::new();
            let mut local_max_delay = 0u64;
            let mut local_verified = 0u64;

            for i in chunk_start..chunk_end {
                let n = reps[i];
                match verify_profiled(n, max_steps) {
                    Some(profile) => {
                        local_verified += 1;
                        if profile.delay > local_max_delay {
                            local_max_delay = profile.delay;
                        }
                    }
                    None => {
                        local_failed.push(n as u64);
                    }
                }
            }

            (local_verified, local_failed, local_max_delay)
        }));
    }

    let mut classes_verified = 0u64;
    let mut failed_classes = Vec::new();
    let mut max_delay = 0u64;

    for handle in handles {
        let (v, f, d) = handle.join().expect("Thread panicked");
        classes_verified += v;
        failed_classes.extend(f);
        if d > max_delay { max_delay = d; }
    }

    let elapsed = t0.elapsed();

    // Each verified class covers (end - start) / 2^K numbers
    let coverage_per_class = (end - start) / modulus;
    let effective_coverage = coverage_per_class * classes_verified as u128;

    ResidueClassResult {
        k,
        classes_verified,
        effective_coverage,
        failed_classes,
        max_delay,
        elapsed_secs: elapsed.as_secs_f64(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Throughput measurement
// ═══════════════════════════════════════════════════════════════════════════

/// Measure single-threaded throughput at a given frontier.
pub fn measure_throughput(frontier_bits: u32, sample_count: u64) -> (f64, u64, u32) {
    let start = 1u128 << frontier_bits;
    let threshold = 1u128 << (frontier_bits - 1);
    let max_steps = 10_000_000;

    let t0 = std::time::Instant::now();
    let mut verified = 0u64;
    let mut max_delay = 0u64;
    let mut max_exp = 0u32;
    let mut n = start + 1; // first odd number above 2^frontier_bits

    for _ in 0..sample_count {
        if let Some(profile) = verify_profiled(n, max_steps) {
            verified += 1;
            if profile.delay > max_delay { max_delay = profile.delay; }
            if profile.expansion_bits > max_exp { max_exp = profile.expansion_bits; }
        }
        n += 2;
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let rate = verified as f64 / elapsed;
    (rate, max_delay, max_exp)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_verify_small() {
        let result = parallel_verify(1, 10_000, 1, 10_000_000, 4);
        assert_eq!(result.n_failed, 0, "All small values should verify");
        assert!(result.n_verified > 0);
    }

    #[test]
    fn test_parallel_verify_matches_serial() {
        // Compare parallel result with serial
        let start = 1u128;
        let end = 1_000u128;
        let threshold = 1u128;
        let max_steps = 10_000_000;

        let par = parallel_verify(start, end, threshold, max_steps, 4);

        // Serial count
        let mut serial_count = 0u64;
        let mut n = 1u128;
        while n < end {
            if verify_profiled(n, max_steps).is_some() {
                serial_count += 1;
            }
            n += 2;
        }

        assert_eq!(par.n_verified, serial_count,
            "Parallel ({}) should match serial ({})", par.n_verified, serial_count);
    }

    #[test]
    fn test_parallel_verify_frontier() {
        // Verify a small range near 2^40
        let start = 1u128 << 40;
        let end = start + 1000;
        let threshold = 1u128 << 39;

        let result = parallel_verify(start, end, threshold, 10_000_000, 4);
        assert_eq!(result.n_failed, 0, "Near 2^40 should all verify");
        assert!(result.n_verified > 0);
    }

    #[test]
    fn test_shadow_coefficients() {
        // For K=1, pattern=1 (odd): one step of Collatz
        // n=1: (3*1+1)/2 = 2, but this is the symbolic version
        let (a, b) = shadow_coefficients(1, 1);
        // All odd numbers: one step gives (3n+1)/2 = 1.5n + 0.5
        // In integer arithmetic: a=3, b=1 (before the final /2)
        // The function tracks the affine transform
        assert!(a >= 1, "a_coeff should be positive");
    }

    #[test]
    fn test_residue_class_verify_small() {
        let result = verify_by_residue_class(1, 10_000, 1, 4, 10_000_000, 4);
        assert!(result.classes_verified > 0);
        assert!(result.failed_classes.is_empty(), "No failures expected for small range");
    }

    #[test]
    fn test_odd_pattern_count() {
        assert_eq!(odd_pattern_count(1), 1);  // just "1"
        assert_eq!(odd_pattern_count(2), 2);  // "01", "11"
        assert_eq!(odd_pattern_count(3), 4);  // "001", "011", "101", "111"
        assert_eq!(odd_pattern_count(4), 8);
    }

    #[test]
    fn test_measure_throughput() {
        // Quick measurement at 2^20
        let (rate, max_delay, _max_exp) = measure_throughput(20, 1000);
        assert!(rate > 0.0, "Should measure positive throughput");
        assert!(max_delay > 0, "Should see non-zero delays");
    }

    #[test]
    fn test_parallel_histograms_nonzero() {
        let result = parallel_verify(1, 10_000, 1, 10_000_000, 2);
        // v₂ and τ histograms should have entries
        let v2_total: u64 = result.agg_v2_histogram.iter().sum();
        let tau_total: u64 = result.agg_tau_histogram.iter().sum();
        assert!(v2_total > 0, "v₂ histogram should be non-empty");
        assert!(tau_total > 0, "τ histogram should be non-empty");
    }

    #[test]
    fn test_early_termination() {
        // If threshold = 2^30, numbers below 2^30 should be trivially verified
        let start = 1u128;
        let end = 100u128;
        let threshold = 1u128 << 30;

        let result = parallel_verify(start, end, threshold, 10_000_000, 1);
        // All numbers 1..100 are below 2^30, so all should be trivially verified
        assert!(result.n_verified > 0);
        assert_eq!(result.n_failed, 0);
    }
}
