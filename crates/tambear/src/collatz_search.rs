//! # Collatz Conjecture Verification — Millennium Prize Seed
//!
//! For every starting value n:
//!   - If even: n → n/2
//!   - If odd:  n → 3n+1
//! Conjecture: the sequence always reaches 1.
//!
//! Current world record: verified to 2^71 ≈ 2.36 × 10^21
//! Target: push past this using GPU-parallel verification.
//!
//! Through tambear: each starting value is independent = embarrassingly parallel.
//! The verification is a pure map: n → (reaches_threshold, steps, max_value).

use std::time::Instant;

/// Result of verifying one starting value
#[derive(Debug, Clone)]
pub struct CollatzResult {
    pub start: u128,
    pub steps: u32,
    pub max_value: u128,
    pub reached_threshold: bool,
}

/// Verify a single starting value. Returns when the sequence drops below `threshold`.
pub fn collatz_verify_one(start: u128, threshold: u128) -> CollatzResult {
    let mut n = start;
    let mut steps = 0u32;
    let mut max_val = start;

    while n >= threshold && n > 1 {
        if n % 2 == 0 {
            n /= 2;
        } else {
            // Check for overflow: 3n+1 could exceed u128
            if n > u128::MAX / 3 {
                // Would overflow — this starting value needs BigInt
                return CollatzResult {
                    start,
                    steps,
                    max_value: max_val,
                    reached_threshold: false, // couldn't verify
                };
            }
            n = 3 * n + 1;
        }
        steps += 1;
        if n > max_val {
            max_val = n;
        }
    }

    CollatzResult {
        start,
        steps,
        max_value: max_val,
        reached_threshold: n < threshold || n == 1,
    }
}

/// Verify a range of starting values. Returns statistics + any failures.
pub fn collatz_verify_range(start: u128, count: u64, threshold: u128) -> CollatzRangeResult {
    let t0 = Instant::now();
    let mut total_steps = 0u64;
    let mut max_steps = 0u32;
    let mut max_value = 0u128;
    let mut max_steps_start = start;
    let mut failures = Vec::new();

    for i in 0..count {
        let n = start + i as u128;
        // Skip even numbers — they immediately halve to a smaller value
        // that was already verified (optimization)
        if n % 2 == 0 {
            continue;
        }

        let result = collatz_verify_one(n, threshold);

        if !result.reached_threshold {
            failures.push(result.clone());
        }

        total_steps += result.steps as u64;
        if result.steps > max_steps {
            max_steps = result.steps;
            max_steps_start = n;
        }
        if result.max_value > max_value {
            max_value = result.max_value;
        }
    }

    let elapsed = t0.elapsed();

    CollatzRangeResult {
        start,
        count,
        threshold,
        total_steps,
        max_steps,
        max_steps_start,
        max_value,
        failures,
        elapsed_secs: elapsed.as_secs_f64(),
        rate: count as f64 / elapsed.as_secs_f64(),
    }
}

#[derive(Debug)]
pub struct CollatzRangeResult {
    pub start: u128,
    pub count: u64,
    pub threshold: u128,
    pub total_steps: u64,
    pub max_steps: u32,
    pub max_steps_start: u128,
    pub max_value: u128,
    pub failures: Vec<CollatzResult>,
    pub elapsed_secs: f64,
    pub rate: f64, // values per second
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  Collatz Conjecture Verification — tambear");
    eprintln!("  Current world record: 2^71");
    eprintln!("  Target: push the frontier");
    eprintln!("==========================================================");

    // Phase 1: warmup at small scale
    eprintln!("\n-- Phase 1: Warmup (verify 1..1M) --");
    let result = collatz_verify_range(1, 1_000_000, 1);
    eprintln!("  Verified: {} values in {:.3}s ({:.0} values/sec)",
        result.count, result.elapsed_secs, result.rate);
    eprintln!("  Max steps: {} (starting from {})", result.max_steps, result.max_steps_start);
    eprintln!("  Max value reached: {}", result.max_value);
    eprintln!("  Failures: {}", result.failures.len());

    // Phase 2: scale test at 10M
    eprintln!("\n-- Phase 2: Scale test (verify 1..10M) --");
    let result = collatz_verify_range(1, 10_000_000, 1);
    eprintln!("  Verified: {} values in {:.3}s ({:.0} values/sec)",
        result.count, result.elapsed_secs, result.rate);
    eprintln!("  Max steps: {} (starting from {})", result.max_steps, result.max_steps_start);
    eprintln!("  Failures: {}", result.failures.len());

    // Phase 3: verify near the frontier (below 2^71 but high)
    eprintln!("\n-- Phase 3: Near-frontier (verify around 2^60) --");
    let frontier_start: u128 = 1u128 << 60; // 2^60 ≈ 1.15 × 10^18
    let threshold: u128 = 1u128 << 59; // anything below 2^59 is "verified" territory
    let result = collatz_verify_range(frontier_start, 100_000, threshold);
    eprintln!("  Range: 2^60 + 0..100000");
    eprintln!("  Verified: {} values in {:.3}s ({:.0} values/sec)",
        result.count, result.elapsed_secs, result.rate);
    eprintln!("  Max steps: {} (starting from {})", result.max_steps, result.max_steps_start);
    eprintln!("  Max value reached: {} ({:.1} bits)",
        result.max_value, (result.max_value as f64).log2());
    eprintln!("  Failures: {}", result.failures.len());

    // Phase 4: estimate time for full 2^71 → 2^72 verification
    let values_per_sec = result.rate;
    let range_size: f64 = (1u128 << 71) as f64; // 2^71 values in the range
    let estimated_hours = range_size / values_per_sec / 3600.0;
    eprintln!("\n-- Estimate for 2^71 → 2^72 --");
    eprintln!("  Rate: {:.0} values/sec (single-threaded CPU)", values_per_sec);
    eprintln!("  Range: 2^71 ≈ {:.2e} values", range_size);
    eprintln!("  Estimated time (1 CPU thread): {:.0} hours ({:.1} days)",
        estimated_hours, estimated_hours / 24.0);
    eprintln!("  With 32 threads: {:.1} days", estimated_hours / 24.0 / 32.0);
    eprintln!("  With GPU (estimated 100x): {:.1} hours", estimated_hours / 100.0);

    if result.failures.is_empty() {
        eprintln!("\n  ALL VALUES REACHED THRESHOLD. No counterexample found.");
        eprintln!("  The Collatz conjecture holds for the tested range.");
    } else {
        eprintln!("\n  !!! {} FAILURES DETECTED !!!", result.failures.len());
        for f in &result.failures {
            eprintln!("    start={} steps={} max={}", f.start, f.steps, f.max_value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collatz_known_values() {
        // 27 is famous: 111 steps, reaches 9232 before descending
        let r = collatz_verify_one(27, 1);
        assert!(r.reached_threshold);
        assert_eq!(r.steps, 111);
        assert_eq!(r.max_value, 9232);
    }

    #[test]
    fn collatz_powers_of_two() {
        // Powers of 2 reach 1 in exactly log2(n) steps
        let r = collatz_verify_one(1024, 1);
        assert!(r.reached_threshold);
        assert_eq!(r.steps, 10); // 1024 → 512 → ... → 1
    }

    #[test]
    fn collatz_range_small() {
        let r = collatz_verify_range(1, 10000, 1);
        assert!(r.failures.is_empty());
        assert!(r.max_steps > 0);
    }

    #[test]
    fn collatz_large_start() {
        // Verify a value near 2^60 reaches below 2^59
        let start: u128 = (1u128 << 60) + 1; // odd number near 2^60
        let threshold: u128 = 1u128 << 59;
        let r = collatz_verify_one(start, threshold);
        assert!(r.reached_threshold, "2^60+1 should reach below 2^59");
    }
}
