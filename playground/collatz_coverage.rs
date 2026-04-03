//! Monte Carlo estimation of backward Collatz tree coverage.
//!
//! Sample random numbers in [2^lo, 2^hi], run forward Collatz,
//! measure how quickly they drop below 2^lo (the "verified" threshold).
//! This estimates the backward tree coverage at each depth.

use std::time::Instant;

/// Simple LCG for deterministic random u128 in a range
fn lcg_u128(state: &mut u64) -> u128 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let hi = *state as u128;
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let lo = *state as u128;
    (hi << 64) | lo
}

fn random_in_range(state: &mut u64, lo: u128, hi: u128) -> u128 {
    let range = hi - lo;
    lo + (lcg_u128(state) % range)
}

/// Run Collatz forward from n, return the number of steps to drop below threshold.
/// Returns None if we exceed max_steps without dropping below.
fn steps_to_threshold(n: u128, threshold: u128, max_steps: u32) -> Option<u32> {
    let mut val = n;
    for step in 0..max_steps {
        if val < threshold {
            return Some(step);
        }
        if val == 1 {
            return Some(step);
        }
        if val % 2 == 0 {
            val /= 2;
        } else {
            if val > u128::MAX / 3 {
                return None; // overflow
            }
            val = 3 * val + 1;
        }
    }
    None // didn't converge
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  Collatz Backward Tree Coverage Estimation");
    eprintln!("  Monte Carlo: sample forward, measure backward density");
    eprintln!("==========================================================\n");

    let mut rng_state: u64 = 42;
    let n_samples = 100_000;
    let max_steps = 10_000;

    // Test coverage at different octaves above the "verified" threshold
    let verified_bits = 60u32; // assume [1, 2^60] is verified
    let threshold: u128 = 1u128 << verified_bits;

    eprintln!("Verified threshold: 2^{} = {:.2e}", verified_bits, threshold as f64);
    eprintln!("Samples per octave: {}\n", n_samples);

    eprintln!("{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "octave", "converged", "pct", "med_steps", "max_steps", "overflow", "time_ms");

    for target_bits in (verified_bits + 1)..=(verified_bits + 15) {
        let lo: u128 = 1u128 << (target_bits - 1);
        let hi: u128 = 1u128 << target_bits;

        let t0 = Instant::now();
        let mut converged = 0u32;
        let mut step_counts = Vec::new();
        let mut overflows = 0u32;

        for _ in 0..n_samples {
            let n = random_in_range(&mut rng_state, lo, hi);
            // Only test odd numbers (even numbers trivially halve)
            let n = if n % 2 == 0 { n + 1 } else { n };

            match steps_to_threshold(n, threshold, max_steps) {
                Some(steps) => {
                    converged += 1;
                    step_counts.push(steps);
                }
                None => {
                    overflows += 1;
                }
            }
        }

        let elapsed_ms = t0.elapsed().as_millis();
        let pct = 100.0 * converged as f64 / n_samples as f64;

        step_counts.sort();
        let median = if step_counts.is_empty() { 0 } else { step_counts[step_counts.len() / 2] };
        let max_s = step_counts.last().copied().unwrap_or(0);

        eprintln!("{:>6}b {:>10} {:>9.1}% {:>10} {:>10} {:>10} {:>9}",
            target_bits, converged, pct, median, max_s, overflows, elapsed_ms);
    }

    eprintln!("\n-- Interpretation --");
    eprintln!("  'converged' = dropped below 2^{} within {} steps", verified_bits, max_steps);
    eprintln!("  This estimates the backward tree coverage at each depth.");
    eprintln!("  If 99%+ converge: backward tree covers that octave.");
    eprintln!("  'overflow' = trajectory exceeded u128 before converging.");
    eprintln!("  Median steps = typical depth of backward tree needed.");
}
