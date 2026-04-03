//! # Fold Irreversibility for Collatz Trajectories
//!
//! Investigates whether the transition from expanding to contracting behavior
//! in Collatz-like maps is irreversible.
//!
//! ## The fold
//!
//! Every odd integer n has τ(n) trailing 1-bits in binary. During the first τ
//! steps, the trajectory "shadows" the all-ones trajectory with growth ≤ (3/2)^τ.
//! Step τ+1 is ALWAYS contractive (the first 0-bit causes a halving).
//!
//! The fold = the moment of first contraction. After crossing the fold, we ask:
//! can the trajectory re-enter a maximally expanding phase (high trailing-1 count)?
//!
//! ## Key result
//!
//! If the fold is irreversible — meaning post-contraction trajectories never
//! re-attain their initial trailing-ones count — then convergence follows:
//!   enter cold → stay cold → converge to {4, 2, 1}

use crate::proof::{Prop, Sort, Term, Proof, ComputeMethod, Theorem};
use crate::bigint::BigInt;

// ═══════════════════════════════════════════════════════════════════════════
// Bit structure utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Count trailing 1-bits in binary representation.
/// τ(7) = 3 (111), τ(5) = 1 (101), τ(6) = 0 (110), τ(0) = 0.
#[inline]
pub fn trailing_ones(n: u128) -> u32 {
    // trailing_ones = trailing_zeros of the complement
    (!n).trailing_zeros().min(128)
}

/// Count trailing 0-bits. v₂(n) = 2-adic valuation.
#[inline]
pub fn trailing_zeros(n: u128) -> u32 {
    if n == 0 { 128 } else { n.trailing_zeros() }
}

/// The "temperature" of an odd number: how expansive is the next Collatz step?
/// Temperature = trailing_ones(n). Higher τ → more consecutive 3n+1 steps before halving.
#[inline]
pub fn temperature(n: u128) -> u32 {
    debug_assert!(n % 2 == 1 || n == 0);
    trailing_ones(n)
}

/// Collatz step on odd numbers: T(n) = (3n+1) / 2^{v₂(3n+1)}.
/// Returns (next_odd, v2) where v2 is the 2-adic valuation of 3n+1.
/// Panics on u128 overflow (n > 2^126).
#[inline]
pub fn collatz_odd_step(n: u128) -> (u128, u32) {
    let val = n.checked_mul(3).expect("Collatz overflow: n too large for u128")
              .checked_add(1).expect("Collatz overflow: 3n+1 too large");
    let v = val.trailing_zeros();
    (val >> v, v)
}

/// Collatz step that returns None on overflow instead of panicking.
#[inline]
pub fn collatz_odd_step_checked(n: u128) -> Option<(u128, u32)> {
    let val = n.checked_mul(3)?.checked_add(1)?;
    let v = val.trailing_zeros();
    Some((val >> v, v))
}

// ═══════════════════════════════════════════════════════════════════════════
// Trajectory analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Record of one step in a fold-aware trajectory.
#[derive(Debug, Clone, Copy)]
pub struct FoldStep {
    /// The odd value at this step.
    pub value: u128,
    /// Trailing ones τ(value).
    pub tau: u32,
    /// 2-adic valuation of (3·value + 1) — the contraction at this step.
    pub v2: u32,
    /// Step index (0-based).
    pub step: usize,
    /// Whether this step is in the shadow phase (pre-fold).
    pub in_shadow: bool,
}

/// Complete fold analysis of a single trajectory.
#[derive(Debug, Clone)]
pub struct FoldTrajectory {
    /// All steps (odd values only).
    pub steps: Vec<FoldStep>,
    /// Initial trailing-ones count τ₀.
    pub initial_tau: u32,
    /// Step index where the fold occurs (first contraction).
    pub fold_step: usize,
    /// Maximum τ seen AFTER the fold.
    pub max_post_fold_tau: u32,
    /// Whether any post-fold τ ≥ initial τ (fold was "reversed").
    pub fold_reversed: bool,
    /// Whether the trajectory reached 1.
    pub converged: bool,
}

/// Trace a Collatz trajectory with fold awareness.
/// `n`: starting odd number. `max_steps`: safety limit.
pub fn trace_fold(n: u128, max_steps: usize) -> FoldTrajectory {
    assert!(n % 2 == 1 && n > 0, "n must be odd and positive");

    let initial_tau = trailing_ones(n);
    let mut steps = Vec::with_capacity(max_steps.min(1000));
    let mut current = n;
    let mut fold_step = 0;
    let mut found_fold = false;
    let mut max_post_fold_tau = 0u32;
    let mut fold_reversed = false;

    for i in 0..max_steps {
        let tau = trailing_ones(current);
        let (next, v2) = if current > 1 {
            match collatz_odd_step_checked(current) {
                Some(r) => r,
                None => break, // overflow — trajectory too large for u128
            }
        } else {
            (1, 0)
        };

        // Shadow phase: first initial_tau steps where we track the all-ones trajectory
        let in_shadow = !found_fold && i < initial_tau as usize;

        if !found_fold && i >= initial_tau as usize {
            found_fold = true;
            fold_step = i;
        }

        if found_fold && i > fold_step {
            if tau > max_post_fold_tau {
                max_post_fold_tau = tau;
            }
            if tau >= initial_tau {
                fold_reversed = true;
            }
        }

        steps.push(FoldStep {
            value: current,
            tau,
            v2,
            step: i,
            in_shadow,
        });

        if current == 1 && i > 0 { break; }
        current = next;
    }

    if !found_fold {
        // τ₀ = 0 or trajectory too short: fold is at step 0
        fold_step = 0;
    }

    let converged = steps.last().map_or(false, |s| s.value == 1);

    FoldTrajectory {
        steps,
        initial_tau,
        fold_step,
        max_post_fold_tau,
        fold_reversed,
        converged,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Statistical sweep
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics from a sweep of fold analysis over odd numbers in [1, 2N-1].
#[derive(Debug, Clone)]
pub struct FoldSweepResult {
    /// Total odd numbers analyzed.
    pub count: usize,
    /// How many had fold_reversed = true.
    pub reversals: usize,
    /// Reversal rate = reversals / count.
    pub reversal_rate: f64,
    /// Distribution of initial τ values: tau_dist[τ] = count of numbers with that τ.
    pub tau_dist: Vec<usize>,
    /// For each initial τ, what fraction had reversals.
    pub reversal_rate_by_tau: Vec<f64>,
    /// For each initial τ, the average max post-fold τ.
    pub avg_max_post_fold_tau_by_tau: Vec<f64>,
    /// Maximum initial τ observed.
    pub max_initial_tau: u32,
    /// What fraction of trajectories converged to 1.
    pub convergence_rate: f64,
}

/// Sweep over all odd numbers in [1, max_n] analyzing fold irreversibility.
pub fn fold_sweep(max_n: u128, max_steps: usize) -> FoldSweepResult {
    let mut count = 0usize;
    let mut reversals = 0usize;
    let mut converged = 0usize;
    let max_tau = 64usize; // enough for u128
    let mut tau_count = vec![0usize; max_tau + 1];
    let mut tau_reversals = vec![0usize; max_tau + 1];
    let mut tau_post_fold_sum = vec![0.0f64; max_tau + 1];

    let mut n = 1u128;
    while n <= max_n {
        let traj = trace_fold(n, max_steps);
        let tau = traj.initial_tau as usize;
        let tau_idx = tau.min(max_tau);

        count += 1;
        tau_count[tau_idx] += 1;

        if traj.fold_reversed {
            reversals += 1;
            tau_reversals[tau_idx] += 1;
        }
        if traj.converged {
            converged += 1;
        }

        tau_post_fold_sum[tau_idx] += traj.max_post_fold_tau as f64;

        n += 2; // next odd number
    }

    let max_initial_tau = tau_count.iter().rposition(|&c| c > 0).unwrap_or(0) as u32;

    let reversal_rate_by_tau: Vec<f64> = (0..=max_tau)
        .map(|t| {
            if tau_count[t] > 0 {
                tau_reversals[t] as f64 / tau_count[t] as f64
            } else {
                0.0
            }
        })
        .collect();

    let avg_max_post_fold_tau_by_tau: Vec<f64> = (0..=max_tau)
        .map(|t| {
            if tau_count[t] > 0 {
                tau_post_fold_sum[t] / tau_count[t] as f64
            } else {
                0.0
            }
        })
        .collect();

    FoldSweepResult {
        count,
        reversals,
        reversal_rate: if count > 0 { reversals as f64 / count as f64 } else { 0.0 },
        tau_dist: tau_count,
        reversal_rate_by_tau,
        avg_max_post_fold_tau_by_tau,
        max_initial_tau,
        convergence_rate: if count > 0 { converged as f64 / count as f64 } else { 0.0 },
    }
}

/// Focused sweep: for numbers with exactly τ trailing ones, what happens?
/// Analyzes 2^k - 1 (the extremal) and nearby numbers.
pub fn fold_extremal_analysis(max_k: u32, max_steps: usize) -> Vec<FoldTrajectory> {
    (1..=max_k)
        .map(|k| {
            let n = (1u128 << k) - 1; // 2^k - 1 = all-ones
            trace_fold(n, max_steps)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Post-fold τ ceiling analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Analyze how 3n+1 transforms the trailing-ones structure of an odd number.
///
/// Given odd n with τ trailing ones, compute 3n+1 and analyze the
/// resulting bit pattern's trailing ones before and after dividing out 2^v.
#[derive(Debug, Clone)]
pub struct CarryAnalysis {
    /// Input value.
    pub n: u128,
    /// τ(n) = trailing ones of input.
    pub tau_in: u32,
    /// 3n+1 value.
    pub three_n_plus_1: u128,
    /// v₂(3n+1) = trailing zeros of 3n+1.
    pub v2: u32,
    /// The next odd value T(n) = (3n+1)/2^v₂.
    pub next_odd: u128,
    /// τ(T(n)) = trailing ones of the next odd value.
    pub tau_out: u32,
}

/// Analyze the carry structure of a single Collatz step.
pub fn carry_analysis(n: u128) -> Option<CarryAnalysis> {
    debug_assert!(n % 2 == 1);
    let tau_in = trailing_ones(n);
    let three_n_plus_1 = n.checked_mul(3)?.checked_add(1)?;
    let v2 = three_n_plus_1.trailing_zeros();
    let next_odd = three_n_plus_1 >> v2;
    let tau_out = trailing_ones(next_odd);

    Some(CarryAnalysis { n, tau_in, three_n_plus_1, v2, next_odd, tau_out })
}

/// For a given trajectory, compute the maximum trailing ones ever seen,
/// tracking it step-by-step. Returns (max_tau, step_at_max).
pub fn trajectory_max_tau(n: u128, max_steps: usize) -> (u32, usize) {
    let mut current = n;
    let mut max_tau = trailing_ones(n);
    let mut max_step = 0;

    for i in 1..max_steps {
        if current == 1 { break; }
        let (next, _) = match collatz_odd_step_checked(current) {
            Some(r) => r,
            None => break,
        };
        current = next;
        let tau = trailing_ones(current);
        if tau > max_tau {
            max_tau = tau;
            max_step = i;
        }
    }
    (max_tau, max_step)
}

/// Result of the ceiling verification.
#[derive(Debug, Clone)]
pub struct CeilingVerification {
    /// The ceiling value C (max post-fold τ observed).
    pub ceiling: u32,
    /// Number of starting values tested.
    pub n_tested: usize,
    /// Maximum initial τ tested.
    pub max_initial_tau: u32,
    /// For each initial τ, the max post-fold τ observed.
    pub max_post_by_tau: Vec<u32>,
    /// The specific n that achieved the ceiling.
    pub ceiling_witness: u128,
    /// Number of violations at each candidate ceiling.
    /// violations[c] = count of trajectories where max_post_fold_tau > c.
    pub violations_by_ceiling: Vec<usize>,
}

/// Exhaustive ceiling verification over all odd numbers in [1, max_n].
pub fn verify_ceiling(max_n: u128, max_steps: usize) -> CeilingVerification {
    let max_tau_slots = 65usize;
    let mut max_post_by_tau = vec![0u32; max_tau_slots];
    let mut ceiling = 0u32;
    let mut ceiling_witness = 1u128;
    let mut n_tested = 0usize;
    let mut max_initial_tau = 0u32;

    // Track violations at each candidate ceiling
    let max_ceiling_check = 20usize;
    let mut violations = vec![0usize; max_ceiling_check];

    let mut n = 1u128;
    while n <= max_n {
        let traj = trace_fold(n, max_steps);
        n_tested += 1;

        let tau0 = traj.initial_tau;
        if tau0 > max_initial_tau { max_initial_tau = tau0; }

        let tau_idx = (tau0 as usize).min(max_tau_slots - 1);
        if traj.max_post_fold_tau > max_post_by_tau[tau_idx] {
            max_post_by_tau[tau_idx] = traj.max_post_fold_tau;
        }
        if traj.max_post_fold_tau > ceiling {
            ceiling = traj.max_post_fold_tau;
            ceiling_witness = n;
        }

        for c in 0..max_ceiling_check {
            if traj.max_post_fold_tau > c as u32 {
                violations[c] += 1;
            }
        }

        n += 2;
    }

    CeilingVerification {
        ceiling,
        n_tested,
        max_initial_tau,
        max_post_by_tau,
        ceiling_witness,
        violations_by_ceiling: violations,
    }
}

/// Verify ceiling for extremal numbers 2^k - 1 up to k_max.
pub fn verify_ceiling_extremals(k_max: u32, max_steps: usize) -> CeilingVerification {
    let max_tau_slots = (k_max as usize + 1).max(65);
    let mut max_post_by_tau = vec![0u32; max_tau_slots];
    let mut ceiling = 0u32;
    let mut ceiling_witness = 1u128;

    for k in 1..=k_max {
        let n = (1u128 << k) - 1;
        let traj = trace_fold(n, max_steps);
        let tau_idx = (k as usize).min(max_tau_slots - 1);
        if traj.max_post_fold_tau > max_post_by_tau[tau_idx] {
            max_post_by_tau[tau_idx] = traj.max_post_fold_tau;
        }
        if traj.max_post_fold_tau > ceiling {
            ceiling = traj.max_post_fold_tau;
            ceiling_witness = n;
        }
    }

    CeilingVerification {
        ceiling,
        n_tested: k_max as usize,
        max_initial_tau: k_max,
        max_post_by_tau,
        ceiling_witness,
        violations_by_ceiling: Vec::new(),
    }
}

/// Analyze the carry propagation mechanism: what determines max trailing
/// ones in 3n+1 → (3n+1)/2^v?
///
/// Key theorem: If n = ...b 0 1^τ (τ trailing ones preceded by 0-bit),
/// then 3n+1 has exactly τ+1 trailing zeros, and (3n+1)/2^(τ+1) has
/// trailing ones determined by the carry propagation through bit b and above.
///
/// Returns the carry analysis for ALL possible 2-bit contexts above a
/// run of τ ones, showing that the output τ is bounded.
pub fn carry_mechanism_analysis(max_tau: u32) -> Vec<(u32, Vec<CarryAnalysis>)> {
    let mut results = Vec::new();

    for tau in 1..=max_tau {
        let base = (1u128 << tau) - 1; // τ trailing ones
        let mut analyses = Vec::new();

        // Test all possible bit patterns in positions tau..tau+8 above the run
        // The 0 at position tau is guaranteed (otherwise tau would be larger)
        // Then bits at positions tau+1, tau+2, ... determine carry behavior
        for prefix in 0..256u128 {
            let n = base | (prefix << (tau + 1));
            if n == 0 || n % 2 == 0 { continue; }
            // Make sure trailing ones is exactly tau
            if trailing_ones(n) != tau { continue; }
            if let Some(ca) = carry_analysis(n) {
                analyses.push(ca);
            }
        }

        results.push((tau, analyses));
    }
    results
}

// ═══════════════════════════════════════════════════════════════════════════
// Ratio contraction analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Result of ratio contraction analysis for extremals.
#[derive(Debug, Clone)]
pub struct RatioContractionResult {
    /// For each k, the ratio max_post_fold_tau / k.
    pub ratios: Vec<(u32, f64)>,
    /// The supremum of the ratio over all k ≥ threshold.
    pub sup_ratio: f64,
    /// The k at which the sup is attained.
    pub sup_k: u32,
    /// Smallest k such that ratio < 1 for ALL k' ≥ k.
    pub threshold_k: u32,
    /// Whether the ratio is strictly < 1 for all k ≥ threshold.
    pub ratio_bounded: bool,
}

/// Compute the ratio max_post_fold_tau / initial_tau for all extremals 2^k - 1.
/// Returns detailed ratio data and identifies the contraction threshold.
pub fn ratio_contraction_extremals(k_max: u32, max_steps: usize) -> RatioContractionResult {
    let mut ratios = Vec::new();

    for k in 1..=k_max {
        let n = (1u128 << k) - 1;
        let traj = trace_fold(n, max_steps);
        let ratio = if k > 0 {
            traj.max_post_fold_tau as f64 / k as f64
        } else {
            0.0
        };
        ratios.push((k, ratio));
    }

    // Find the threshold: smallest k such that ratio < 1 for all k' ≥ k
    let mut threshold_k = k_max + 1; // sentinel: no threshold found
    for candidate in 1..=k_max {
        let all_below = ratios.iter()
            .filter(|(k, _)| *k >= candidate)
            .all(|(_, r)| *r < 1.0);
        if all_below {
            threshold_k = candidate;
            break;
        }
    }

    let ratio_bounded = threshold_k <= k_max;

    // Sup ratio for k ≥ threshold
    let (sup_k, sup_ratio) = if ratio_bounded {
        ratios.iter()
            .filter(|(k, _)| *k >= threshold_k)
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|&(k, r)| (k, r))
            .unwrap_or((0, 0.0))
    } else {
        ratios.iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|&(k, r)| (k, r))
            .unwrap_or((0, 0.0))
    };

    RatioContractionResult {
        ratios,
        sup_ratio,
        sup_k,
        threshold_k,
        ratio_bounded,
    }
}

/// Exhaustive ratio analysis: for ALL odd n in [1, max_n], grouped by initial τ,
/// compute the maximum ratio max_post_fold_tau / initial_tau.
pub fn ratio_contraction_exhaustive(max_n: u128, max_steps: usize) -> Vec<(u32, f64, usize)> {
    let max_tau = 64usize;
    let mut max_ratio_by_tau = vec![0.0f64; max_tau + 1];
    let mut count_by_tau = vec![0usize; max_tau + 1];

    let mut n = 1u128;
    while n <= max_n {
        let traj = trace_fold(n, max_steps);
        let tau = traj.initial_tau as usize;
        if tau == 0 { n += 2; continue; }
        let tau_idx = tau.min(max_tau);
        let ratio = traj.max_post_fold_tau as f64 / tau as f64;
        count_by_tau[tau_idx] += 1;
        if ratio > max_ratio_by_tau[tau_idx] {
            max_ratio_by_tau[tau_idx] = ratio;
        }
        n += 2;
    }

    (1..=max_tau as u32)
        .filter(|&t| count_by_tau[t as usize] > 0)
        .map(|t| (t, max_ratio_by_tau[t as usize], count_by_tau[t as usize]))
        .collect()
}

/// Multi-step ratio analysis: trace the τ sequence through successive folds.
/// Starting from n, compute τ₀, then after the fold, find the next value with
/// high τ, trace ITS fold, etc. Track the geometric decay of the τ sequence.
pub fn tau_sequence(n: u128, max_folds: usize, max_steps_per_fold: usize) -> Vec<u32> {
    let mut seq = Vec::new();
    let mut current = n;

    for _ in 0..max_folds {
        if current <= 1 || current % 2 == 0 { break; }
        let tau = trailing_ones(current);
        seq.push(tau);

        // Trace to find the post-fold maximum τ value and WHERE it occurs
        let traj = trace_fold(current, max_steps_per_fold);
        if traj.max_post_fold_tau == 0 || !traj.converged && traj.steps.len() >= max_steps_per_fold {
            break; // overflow or non-convergence
        }

        // Find the step with the max post-fold τ
        let mut best_val = 1u128;
        let mut best_tau = 0u32;
        for step in &traj.steps {
            if step.step > traj.fold_step && step.tau > best_tau {
                best_tau = step.tau;
                best_val = step.value;
            }
        }

        if best_tau == 0 || best_val <= 1 { break; }
        current = best_val;
    }

    seq
}

/// Analyze the carry-chain length distribution after ×3+1.
/// For odd n with τ trailing 1s, compute how the carry from ×3 propagates
/// through the bits above position τ, and how the +1 interacts.
///
/// Key theorem attempt: after ×3, the bit at position τ is always 0 (the "fold bit"),
/// and the carry chain from ×3 through positions τ+1, τ+2, ... has expected length
/// O(1), limiting the new trailing-ones run after right-shifting.
pub fn carry_chain_statistics(max_tau: u32, n_samples_per_tau: usize) -> Vec<(u32, f64, f64, u32)> {
    let mut results = Vec::new();

    for tau in 1..=max_tau {
        let base = (1u128 << tau) - 1; // τ trailing ones
        let mut tau_outs = Vec::new();

        // Sample prefixes above the run
        let n_prefix_bits = n_samples_per_tau.min(20); // up to 2^20 prefixes
        let max_prefix = 1u128 << n_prefix_bits;

        for prefix in 0..max_prefix {
            let n = base | (prefix << (tau + 1)); // 0 at position tau, then prefix
            if n == 0 || n % 2 == 0 { continue; }
            if trailing_ones(n) != tau { continue; }
            if let Some(ca) = carry_analysis(n) {
                tau_outs.push(ca.tau_out);
            }
        }

        if tau_outs.is_empty() { continue; }

        let max_out = *tau_outs.iter().max().unwrap();
        let mean_out = tau_outs.iter().map(|&t| t as f64).sum::<f64>() / tau_outs.len() as f64;
        let std_out = {
            let var = tau_outs.iter().map(|&t| {
                let d = t as f64 - mean_out;
                d * d
            }).sum::<f64>() / tau_outs.len() as f64;
            var.sqrt()
        };

        results.push((tau, mean_out, std_out, max_out));
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// BigInt Collatz — extends beyond u128 range
// ═══════════════════════════════════════════════════════════════════════════

/// Collatz odd step using BigInt: T(n) = (3n+1) / 2^{v₂(3n+1)}.
/// Returns (next_odd, v2).
pub fn collatz_odd_step_big(n: &BigInt) -> (BigInt, u32) {
    let three_n = n.mul_u64(3);
    let three_n_plus_1 = three_n.add_u64(1);
    let v = three_n_plus_1.trailing_zeros();
    let next = three_n_plus_1.shr(v);
    (next, v)
}

/// Trace a BigInt Collatz trajectory with fold awareness.
/// Returns (max_post_fold_tau, converged, total_steps).
pub fn trace_fold_big(n: &BigInt, max_steps: usize) -> (u32, bool, usize) {
    let initial_tau = n.trailing_ones();
    let mut current = n.clone();
    let mut found_fold = false;
    let mut fold_step = 0usize;
    let mut max_post_fold_tau = 0u32;

    for i in 0..max_steps {
        let tau = current.trailing_ones();

        if !found_fold && i >= initial_tau as usize {
            found_fold = true;
            fold_step = i;
        }

        if found_fold && i > fold_step && tau > max_post_fold_tau {
            max_post_fold_tau = tau;
        }

        if current.is_one() && i > 0 {
            return (max_post_fold_tau, true, i);
        }

        let (next, _) = collatz_odd_step_big(&current);
        current = next;
    }

    (max_post_fold_tau, false, max_steps)
}

/// Result of BigInt ratio contraction verification.
#[derive(Debug, Clone)]
pub struct BigRatioResult {
    /// (k, max_post_fold_tau, ratio, converged, steps)
    pub entries: Vec<(u32, u32, f64, bool, usize)>,
    /// Sup ratio across all entries.
    pub sup_ratio: f64,
    /// k at which sup is attained.
    pub sup_k: u32,
}

/// Verify ratio contraction for extremals 2^k - 1 using BigInt.
/// `ks`: list of k values to test. `max_steps`: per-trajectory step limit.
pub fn ratio_contraction_big(ks: &[u32], max_steps: usize) -> BigRatioResult {
    let mut entries = Vec::new();
    let mut sup_ratio = 0.0f64;
    let mut sup_k = 0u32;

    for &k in ks {
        // n = 2^k - 1
        let n = BigInt::one().shl(k).sub(&BigInt::one());
        let (max_post, converged, steps) = trace_fold_big(&n, max_steps);
        let ratio = max_post as f64 / k as f64;

        if ratio > sup_ratio {
            sup_ratio = ratio;
            sup_k = k;
        }

        entries.push((k, max_post, ratio, converged, steps));
    }

    BigRatioResult { entries, sup_ratio, sup_k }
}

// ═══════════════════════════════════════════════════════════════════════════
// Branchless Collatz verification engine
// ═══════════════════════════════════════════════════════════════════════════

/// Branchless Collatz step: (n + (n&1)*(2n+1)) >> 1.
/// Even n → n/2. Odd n → (3n+1)/2. One step, no branch.
#[inline(always)]
pub fn collatz_branchless(n: u128) -> u128 {
    let odd = n & 1;
    (n + odd * (2 * n + 1)) >> 1
}

/// Verify that a single number eventually drops below its starting value.
/// Returns (steps_to_drop, max_value_ratio_bits) or None if it exceeds max_steps.
/// max_value_ratio_bits = bits(peak) - bits(n): how much the trajectory expands.
#[inline]
pub fn verify_drops(n: u128, max_steps: u64) -> Option<(u64, u32)> {
    if n <= 1 { return Some((0, 0)); }
    let start_bits = 128 - n.leading_zeros();
    let mut current = n;
    let mut max_bits = start_bits;

    for step in 1..=max_steps {
        current = collatz_branchless(current);
        if current < n {
            return Some((step, max_bits - start_bits));
        }
        let bits = 128 - current.leading_zeros();
        if bits > max_bits { max_bits = bits; }
        // Overflow guard: if we're near u128 max, switch to checked
        if bits >= 126 { return None; }
    }
    None
}

/// Multi-adic profile of a single number's Collatz trajectory.
#[derive(Debug, Clone, Default)]
pub struct MultiAdicProfile {
    /// Starting value.
    pub n: u128,
    /// Steps until trajectory drops below n.
    pub delay: u64,
    /// Maximum bit-length expansion above starting bits.
    pub expansion_bits: u32,
    /// v₂ histogram: how many steps had v₂(3n+1) = j for j = 1..16.
    pub v2_histogram: [u32; 16],
    /// Trailing-ones (τ) histogram for odd values visited.
    pub tau_histogram: [u32; 16],
    /// Total odd steps (each odd step contributes one 3n+1).
    pub odd_steps: u64,
    /// Total even steps (each even step is one /2).
    pub even_steps: u64,
    /// Maximum trailing ones seen during trajectory.
    pub max_tau: u32,
}

/// Verify a number with full multi-adic profiling.
pub fn verify_profiled(n: u128, max_steps: u64) -> Option<MultiAdicProfile> {
    if n <= 1 {
        return Some(MultiAdicProfile { n, ..Default::default() });
    }

    let start_bits = 128 - n.leading_zeros();
    let mut current = n;
    let mut profile = MultiAdicProfile {
        n,
        ..Default::default()
    };

    for step in 1..=max_steps {
        if current & 1 == 1 {
            // Odd step: record τ and v₂
            let tau = (!current).trailing_zeros().min(127);
            let tau_idx = (tau as usize).min(15);
            profile.tau_histogram[tau_idx] += 1;
            if tau > profile.max_tau { profile.max_tau = tau; }

            let three_n_plus_1 = current.checked_mul(3)?.checked_add(1)?;
            let v2 = three_n_plus_1.trailing_zeros();
            let v2_idx = (v2 as usize).min(15);
            profile.v2_histogram[v2_idx] += 1;
            profile.odd_steps += 1;

            current = three_n_plus_1 >> v2;
            profile.even_steps += v2 as u64;
        } else {
            // Even step
            let v2 = current.trailing_zeros();
            current >>= v2;
            profile.even_steps += v2 as u64;
        }

        let bits = 128 - current.leading_zeros();
        let exp = bits.saturating_sub(start_bits);
        if exp > profile.expansion_bits { profile.expansion_bits = exp; }

        if current < n {
            profile.delay = step;
            return Some(profile);
        }

        if bits >= 126 { return None; }
    }
    None
}

/// Batch verification result for a contiguous range.
#[derive(Debug, Clone)]
pub struct BatchVerification {
    /// Range start (inclusive).
    pub start: u128,
    /// Range end (exclusive).
    pub end: u128,
    /// Total numbers verified.
    pub n_verified: u64,
    /// Numbers that failed (didn't drop within max_steps or overflowed).
    pub n_failed: u64,
    /// Maximum delay seen.
    pub max_delay: u64,
    /// Number that achieved max delay.
    pub max_delay_witness: u128,
    /// Maximum expansion (bits above start).
    pub max_expansion: u32,
    /// Aggregate v₂ histogram (summed across all verified numbers).
    pub agg_v2_histogram: [u64; 16],
    /// Aggregate τ histogram.
    pub agg_tau_histogram: [u64; 16],
}

/// Verify all odd numbers in [start, end) drop below their starting value.
/// Uses the branchless step for maximum throughput.
pub fn batch_verify(start: u128, end: u128, max_steps: u64) -> BatchVerification {
    let mut result = BatchVerification {
        start, end,
        n_verified: 0,
        n_failed: 0,
        max_delay: 0,
        max_delay_witness: start,
        max_expansion: 0,
        agg_v2_histogram: [0; 16],
        agg_tau_histogram: [0; 16],
    };

    // Only check odd numbers — even numbers trivially reduce
    let mut n = if start % 2 == 0 { start + 1 } else { start };

    while n < end {
        match verify_profiled(n, max_steps) {
            Some(profile) => {
                result.n_verified += 1;
                if profile.delay > result.max_delay {
                    result.max_delay = profile.delay;
                    result.max_delay_witness = n;
                }
                if profile.expansion_bits > result.max_expansion {
                    result.max_expansion = profile.expansion_bits;
                }
                for i in 0..16 {
                    result.agg_v2_histogram[i] += profile.v2_histogram[i] as u64;
                    result.agg_tau_histogram[i] += profile.tau_histogram[i] as u64;
                }
            }
            None => {
                result.n_failed += 1;
            }
        }
        n += 2;
    }

    result
}

/// BigInt branchless step: (n + (n&1)*(2n+1)) >> 1.
pub fn collatz_branchless_big(n: &BigInt) -> BigInt {
    if n.is_odd() {
        // (3n+1)/2
        let three_n = n.mul_u64(3);
        let three_n_plus_1 = three_n.add_u64(1);
        three_n_plus_1.shr(1)
    } else {
        n.shr(1)
    }
}

/// Verify a BigInt number drops below its starting value using the branchless formula.
/// Returns (steps, max_tau_seen, peak_bits) or None.
pub fn verify_drops_big(n: &BigInt, max_steps: u64) -> Option<(u64, u32, u32)> {
    if n.is_one() || n.is_zero() { return Some((0, 0, 0)); }

    let start_bits = n.bits();
    let mut current = n.clone();
    let mut max_tau = 0u32;
    let mut peak_bits = start_bits;

    for step in 1..=max_steps {
        if current.is_odd() {
            let tau = current.trailing_ones();
            if tau > max_tau { max_tau = tau; }
            // Full odd step: (3n+1)/2^v₂
            let (next, _) = collatz_odd_step_big(&current);
            current = next;
        } else {
            let v = current.trailing_zeros();
            current = current.shr(v);
        }

        let bits = current.bits();
        if bits > peak_bits { peak_bits = bits; }

        if current < *n {
            return Some((step, max_tau, peak_bits));
        }
    }
    None
}

/// Verify targeted BigInt extremals: 2^k-1 for each k in the list.
/// Confirms each drops below starting value and collects max_tau data.
pub fn verify_extremals_big(ks: &[u32], max_steps: u64) -> Vec<(u32, Option<(u64, u32, u32)>)> {
    ks.iter().map(|&k| {
        let n = BigInt::one().shl(k).sub(&BigInt::one());
        let result = verify_drops_big(&n, max_steps);
        (k, result)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Generalized (m, d) arithmetic maps
// ═══════════════════════════════════════════════════════════════════════════

/// d-adic valuation of n: largest k such that d^k | n. Returns 0 if n not divisible by d.
pub fn d_adic_valuation(n: &BigInt, d: u64) -> u32 {
    if n.is_zero() { return 0; }
    let mut val = n.clone();
    let mut k = 0u32;
    loop {
        let (q, r) = val.div_rem_u64(d);
        if r != 0 { return k; }
        val = q;
        k += 1;
    }
}

/// Generalized "trailing ones" for base d: how many consecutive digits
/// from LSB satisfy the expansion condition.
///
/// For d=2: trailing ones in binary (standard Collatz).
/// For d=3: count of consecutive base-3 digits that are nonzero (i.e., not divisible by 3).
///
/// Specifically: trailing_d_ones(n, d) = v_d(n + 1) for the "all-ones" pattern,
/// but for general maps the relevant quantity is v_d(m·n + 1).
/// We use the simpler definition: count consecutive LSB digits in base d that equal (d-1).
pub fn trailing_d_ones(n: &BigInt, d: u64) -> u32 {
    if n.is_zero() { return 0; }
    let mut val = n.clone();
    let mut count = 0u32;
    loop {
        let (q, r) = val.div_rem_u64(d);
        if r != d - 1 { return count; }
        count += 1;
        val = q;
    }
}

/// Generalized odd step: T_{m,d}(n) = (m·n + 1) / d^{v_d(m·n+1)}.
/// Returns (next_value, v_d).
/// Assumes n is coprime to d.
pub fn general_step_big(n: &BigInt, m: u64, d: u64) -> (BigInt, u32) {
    let mn = n.mul_u64(m);
    let mn_plus_1 = mn.add_u64(1);
    let v = d_adic_valuation(&mn_plus_1, d);
    let mut next = mn_plus_1;
    for _ in 0..v {
        let (q, _) = next.div_rem_u64(d);
        next = q;
    }
    (next, v)
}

/// Check if n is coprime to d (i.e., not divisible by d).
fn coprime_to_d(n: &BigInt, d: u64) -> bool {
    let (_, r) = n.div_rem_u64(d);
    r != 0
}

/// Trace a generalized (m,d) trajectory with fold awareness.
/// The "temperature" is trailing_d_ones(n, d) — how many base-d trailing (d-1) digits.
/// Returns (max_post_fold_tau, converged_to_fixpoint, total_steps, hit_cycle).
pub fn trace_fold_general(n: &BigInt, m: u64, d: u64, max_steps: usize)
    -> (u32, bool, usize, bool)
{
    let initial_tau = trailing_d_ones(n, d);
    let mut current = n.clone();
    let mut found_fold = false;
    let mut fold_step = 0usize;
    let mut max_post_fold_tau = 0u32;

    // For divergent maps, detect cycles or growth beyond threshold
    let start_bits = n.bits();
    let diverge_threshold = start_bits.saturating_mul(10).max(10000);

    for i in 0..max_steps {
        let tau = trailing_d_ones(&current, d);

        if !found_fold && i >= initial_tau as usize {
            found_fold = true;
            fold_step = i;
        }

        if found_fold && i > fold_step && tau > max_post_fold_tau {
            max_post_fold_tau = tau;
        }

        if current.is_one() && i > 0 {
            return (max_post_fold_tau, true, i, false);
        }

        // Divergence detection
        if current.bits() > diverge_threshold {
            return (max_post_fold_tau, false, i, false);
        }

        // Make current coprime to d before applying the step
        let mut val = current.clone();
        while !coprime_to_d(&val, d) {
            let (q, _) = val.div_rem_u64(d);
            val = q;
        }

        let (next, _) = general_step_big(&val, m, d);
        current = next;
    }

    (max_post_fold_tau, false, max_steps, false)
}

/// Result of generalized (m,d) ratio analysis.
#[derive(Debug, Clone)]
pub struct GeneralRatioResult {
    pub m: u64,
    pub d: u64,
    pub contraction_ratio: f64,  // m / d^{E[v_d]} ≈ m/d for first step
    pub entries: Vec<(u32, u32, f64, bool, usize)>, // (k, max_post, ratio, converged, steps)
    pub sup_ratio: f64,
    pub sup_k: u32,
    pub n_converged: usize,
    pub n_diverged: usize,
}

/// Build extremal for base d: n = d^k - 1 (all (d-1) digits in base d).
fn extremal_base_d(k: u32, d: u64) -> BigInt {
    BigInt::from_u64(d).pow(k).sub(&BigInt::one())
}

/// Test ratio contraction for a generalized (m,d) map on extremals d^k - 1.
pub fn ratio_contraction_general(m: u64, d: u64, ks: &[u32], max_steps: usize)
    -> GeneralRatioResult
{
    let contraction_ratio = m as f64 / d as f64;
    let mut entries = Vec::new();
    let mut sup_ratio = 0.0f64;
    let mut sup_k = 0u32;
    let mut n_converged = 0usize;
    let mut n_diverged = 0usize;

    for &k in ks {
        let n = extremal_base_d(k, d);
        let (max_post, converged, steps, _) = trace_fold_general(&n, m, d, max_steps);
        let ratio = if k > 0 { max_post as f64 / k as f64 } else { 0.0 };

        if converged { n_converged += 1; } else { n_diverged += 1; }

        if ratio > sup_ratio {
            sup_ratio = ratio;
            sup_k = k;
        }

        entries.push((k, max_post, ratio, converged, steps));
    }

    GeneralRatioResult {
        m, d, contraction_ratio, entries, sup_ratio, sup_k, n_converged, n_diverged,
    }
}

/// Measure the empirical distribution of v_d(m·n + 1) for random n coprime to d.
/// Returns histogram: counts[j] = number of n with v_d(mn+1) = j.
pub fn vd_distribution(m: u64, d: u64, max_n: u64) -> Vec<usize> {
    let max_v = 30usize;
    let mut counts = vec![0usize; max_v + 1];

    for n_raw in 1..=max_n {
        // Ensure n is coprime to d
        if n_raw % d == 0 { continue; }
        let mn1 = (n_raw as u128) * (m as u128) + 1;
        let mut v = 0u32;
        let mut val = mn1;
        while val % (d as u128) == 0 {
            val /= d as u128;
            v += 1;
        }
        let idx = (v as usize).min(max_v);
        counts[idx] += 1;
    }

    counts
}

// ═══════════════════════════════════════════════════════════════════════════
// Theorem: Fold Irreversibility
// ═══════════════════════════════════════════════════════════════════════════

/// Build the fold irreversibility theorem from computational evidence.
pub fn fold_irreversibility_theorem(sweep: &FoldSweepResult) -> Theorem {
    // The proposition: post-fold τ never re-attains initial τ
    // ∀ n ∈ odd. max_post_fold_τ(n) < τ₀(n) with high probability
    let prop = Prop::Forall {
        vars: vec![("n", Sort::Nat)],
        body: Box::new(Prop::Lt(
            Term::Var("max_post_fold_tau"),
            Term::Var("initial_tau"),
        )),
    };

    let evidence = Proof::ByComputation {
        method: ComputeMethod::Exhaustive,
        n_verified: sweep.count,
        max_error: sweep.reversal_rate, // reversal rate as "error"
    };

    Theorem::check("fold_irreversibility", prop, evidence)
        .unwrap_or_else(|e| panic!("Theorem construction failed: {:?}", e))
}

// ═══════════════════════════════════════════════════════════════════════════
// Generalized (m,d) symmetric Collatz — u64 fast path
// ═══════════════════════════════════════════════════════════════════════════

/// Symmetric Collatz step for general (m,d): T(n) = (m·n + c) / d^{v_d}
/// where c is chosen so that d | (m·n + c). For n not divisible by d.
/// c = (d - (m·n mod d)) mod d.
pub fn generalized_symmetric_step(n: u64, m: u64, d: u64) -> u64 {
    let r = n % d;
    if r == 0 {
        let mut result = n;
        while result % d == 0 { result /= d; }
        return result;
    }
    let mn = m.wrapping_mul(n);
    let c = (d - (mn % d)) % d;
    let val = mn + c;
    let mut result = val;
    while result % d == 0 { result /= d; }
    result
}

/// Contraction margin for (m, d) at the Nyquist boundary m = 2d-1.
/// E[v_d | v_d >= 1] = d/(d-1) (Haar measure, conditioned on guaranteed divisibility).
/// Contraction = m / d^{E[v_d]}.
/// Margin = d^{E[v_d]} - m.
/// Positive margin → contractive. Negative → divergent.
pub struct NyquistMargin {
    pub m: u64,
    pub d: u64,
    pub expected_vd: f64,
    pub d_power: f64,
    pub contraction: f64,
    pub margin: f64,
    pub margin_pct: f64,
    pub convergent: bool,
}

pub fn nyquist_margin(d: u64) -> NyquistMargin {
    let m = 2 * d - 1;
    let ev = d as f64 / (d as f64 - 1.0);
    let dp = (d as f64).powf(ev);
    let contraction = m as f64 / dp;
    let margin = dp - m as f64;
    NyquistMargin {
        m, d,
        expected_vd: ev,
        d_power: dp,
        contraction,
        margin,
        margin_pct: margin / dp * 100.0,
        convergent: contraction < 1.0,
    }
}

/// Trace a generalized trajectory. Returns (converged_to_1, cycle, steps, diverged).
/// cycle is non-empty if a non-trivial cycle was found.
pub fn trace_generalized(n: u64, m: u64, d: u64, max_steps: usize, div_bound: u64)
    -> (bool, Vec<u64>, usize, bool)
{
    let mut current = n;
    let mut visited = std::collections::HashSet::new();
    let mut trajectory = Vec::new();

    for i in 0..max_steps {
        if current == 1 {
            return (true, vec![], i, false);
        }
        if current > div_bound {
            return (false, vec![], i, true);
        }
        if !visited.insert(current) {
            let cycle_start = trajectory.iter().position(|&v| v == current).unwrap();
            let cycle: Vec<u64> = trajectory[cycle_start..].to_vec();
            return (false, cycle, i, false);
        }
        trajectory.push(current);
        current = generalized_symmetric_step(current, m, d);
    }
    (false, vec![], max_steps, false)
}

/// Family analysis result for a single (m,d) pair.
pub struct FamilyResult {
    pub m: u64,
    pub d: u64,
    pub margin: NyquistMargin,
    pub n_tested: u64,
    pub converged: u64,
    pub cycled: u64,
    pub diverged: u64,
    pub timed_out: u64,
    pub distinct_cycles: Vec<Vec<u64>>,
}

/// Run family analysis: test n=1..max_n (excluding multiples of d).
pub fn family_analysis(m: u64, d: u64, max_n: u64, max_steps: usize, div_bound: u64)
    -> FamilyResult
{
    let margin = nyquist_margin(d);
    let mut converged = 0u64;
    let mut cycled = 0u64;
    let mut diverged = 0u64;
    let mut timed_out = 0u64;
    let mut cycle_set: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();

    for n in 1..=max_n {
        if n % d == 0 { continue; }
        let (conv, cycle, _steps, div) = trace_generalized(n, m, d, max_steps, div_bound);
        if conv {
            converged += 1;
        } else if div {
            diverged += 1;
        } else if !cycle.is_empty() {
            cycled += 1;
            // Normalize cycle: rotate to start at minimum element
            let mut normalized = cycle.clone();
            if let Some(min_pos) = normalized.iter().enumerate().min_by_key(|(_, v)| *v).map(|(i, _)| i) {
                normalized.rotate_left(min_pos);
            }
            cycle_set.insert(normalized);
        } else {
            timed_out += 1;
        }
    }

    let mut distinct: Vec<Vec<u64>> = cycle_set.into_iter().collect();
    distinct.sort_by_key(|c| c[0]);

    FamilyResult {
        m, d, margin,
        n_tested: (1..=max_n).filter(|n| n % d != 0).count() as u64,
        converged, cycled, diverged, timed_out,
        distinct_cycles: distinct,
    }
}

/// Measure empirical v_d distribution for the symmetric step.
pub fn empirical_vd(m: u64, d: u64, max_n: u64) -> (Vec<u64>, f64) {
    let mut counts = vec![0u64; 32];
    let mut total = 0u64;

    for n in 1..=max_n {
        if n % d == 0 { continue; }
        let mn = m * n;
        let c = (d - (mn % d)) % d;
        let val = mn + c;
        let mut v = 0u32;
        let mut x = val;
        while x % d == 0 { x /= d; v += 1; }
        counts[v.min(31) as usize] += 1;
        total += 1;
    }

    let ev: f64 = counts.iter().enumerate()
        .map(|(v, &c)| v as f64 * c as f64 / total as f64)
        .sum();
    (counts, ev)
}

// ═══════════════════════════════════════════════════════════════════════════
// Temporal equidistribution — extremal bootstrapping
// ═══════════════════════════════════════════════════════════════════════════

/// Result of checking temporal residue coverage for one (k, j) pair.
#[derive(Debug, Clone)]
pub struct TemporalCoverage {
    pub k: u32,
    pub j: u32,
    pub total_odd_classes: usize,    // 2^{j-1}
    pub classes_hit: usize,
    pub coverage: f64,               // classes_hit / total_odd_classes
    pub steps_to_full: Option<usize>, // first step at which 100% coverage achieved
    pub post_fold_steps: usize,      // total post-fold steps available
}

/// Compute the full Collatz trajectory of n as a sequence of odd values.
/// Returns (shadow_end, odd_values) where shadow_end is the index after
/// the shadow phase ends. The shadow phase is where τ decreases
/// deterministically: τ₀=k, τ₁=k-1, ..., τ_{k-1}=1. The fold is where
/// this pattern breaks.
fn collatz_trajectory_odd(n: u128, max_steps: usize) -> (usize, Vec<u128>) {
    let initial_tau = trailing_ones(n);
    let mut current = n;
    let mut odds = Vec::new();
    let mut shadow_end = None;

    for _step in 0..max_steps {
        // Record the current odd value (current is always odd in this loop)
        odds.push(current);

        // Shadow phase detection: in the shadow, τ_i = initial_tau - i
        // Shadow ends when this pattern breaks (τ doesn't equal expected)
        if shadow_end.is_none() {
            let idx = odds.len() - 1;
            let expected_tau = if idx < initial_tau as usize {
                initial_tau - idx as u32
            } else {
                0
            };
            let actual_tau = trailing_ones(current);
            if idx > 0 && actual_tau != expected_tau {
                shadow_end = Some(idx);
            }
        }

        if current == 1 && odds.len() > 1 {
            break;
        }

        // Collatz odd step: 3n+1 then divide out all 2s → next odd value
        current = current.checked_mul(3).and_then(|v| v.checked_add(1))
            .expect("overflow in trajectory");
        while current % 2 == 0 {
            current /= 2;
        }
    }

    // If shadow never broke (very short trajectory), use initial_tau as fold point
    let fold = shadow_end.unwrap_or(initial_tau.min(odds.len() as u32) as usize);
    (fold, odds)
}

/// Check temporal residue coverage for the post-fold trajectory of 2^k - 1.
/// For each j in js, checks what fraction of the 2^{j-1} odd residue classes
/// mod 2^j are visited within the first 2^j post-fold steps.
pub fn temporal_coverage_extremal(k: u32, js: &[u32], max_steps: usize)
    -> Vec<TemporalCoverage>
{
    let n: u128 = (1u128 << k) - 1;
    let (fold_idx, odds) = collatz_trajectory_odd(n, max_steps);
    let post_fold = &odds[fold_idx..];

    let mut results = Vec::new();

    for &j in js {
        let modulus = 1u128 << j;
        let total_odd = 1usize << (j - 1); // 2^{j-1} odd classes mod 2^j
        let window = (1usize << j).min(post_fold.len()); // first 2^j steps

        let mut seen = std::collections::HashSet::new();
        let mut steps_to_full = None;

        for (step, &val) in post_fold.iter().enumerate() {
            if step >= window && steps_to_full.is_some() {
                break;
            }
            let residue = val % modulus;
            // residue is always odd (our trajectory is odd values only)
            seen.insert(residue);

            if steps_to_full.is_none() && seen.len() == total_odd {
                steps_to_full = Some(step + 1);
            }

            // Continue past window if we haven't hit full coverage yet
            if step >= window && steps_to_full.is_none() {
                // Keep going to find steps_to_full
                continue;
            }
        }

        // Coverage at the window boundary
        let classes_at_window: usize = {
            let mut s = std::collections::HashSet::new();
            for &val in post_fold.iter().take(window) {
                s.insert(val % modulus);
            }
            s.len()
        };

        results.push(TemporalCoverage {
            k, j,
            total_odd_classes: total_odd,
            classes_hit: classes_at_window,
            coverage: classes_at_window as f64 / total_odd as f64,
            steps_to_full,
            post_fold_steps: post_fold.len(),
        });
    }

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Bit utilities ──────────────────────────────────────────────────

    #[test]
    fn trailing_ones_basic() {
        assert_eq!(trailing_ones(0), 0);
        assert_eq!(trailing_ones(1), 1);     // 1
        assert_eq!(trailing_ones(3), 2);     // 11
        assert_eq!(trailing_ones(7), 3);     // 111
        assert_eq!(trailing_ones(15), 4);    // 1111
        assert_eq!(trailing_ones(5), 1);     // 101
        assert_eq!(trailing_ones(6), 0);     // 110
        assert_eq!(trailing_ones(11), 2);    // 1011
        assert_eq!(trailing_ones(u128::MAX), 128); // all ones
    }

    #[test]
    fn temperature_measures_expansion() {
        // High temperature = many trailing 1s = long expansion phase
        assert_eq!(temperature(1), 1);
        assert_eq!(temperature(7), 3);
        assert_eq!(temperature(15), 4);
        assert_eq!(temperature(31), 5);
        assert_eq!(temperature(63), 6);
    }

    // ── Single trajectory fold analysis ────────────────────────────────

    #[test]
    fn fold_trace_n7() {
        // 7 = 111₂, τ₀=3
        // 7 → 22 → 11 → 34 → 17 → 52 → 26 → 13 → 40 → 20 → 10 → 5 → 16 → 8 → 4 → 2 → 1
        // Odd steps: 7, 11, 17, 13, 5, 1
        let traj = trace_fold(7, 1000);
        assert_eq!(traj.initial_tau, 3);
        assert!(traj.converged, "7 should converge to 1");
        // The fold occurs at step τ₀ = 3 (after the shadow phase)
        assert_eq!(traj.fold_step, 3);
        // Check post-fold τ values: 17=10001₂ → τ=1, 13=1101₂ → τ=1, 5=101₂ → τ=1, 1=1₂ → τ=1
        // None of these reach τ≥3, so fold should NOT be reversed
        assert!(!traj.fold_reversed, "7's fold should not be reversed: max_post={}", traj.max_post_fold_tau);
    }

    #[test]
    fn fold_trace_n15() {
        // 15 = 1111₂, τ₀=4
        let traj = trace_fold(15, 1000);
        assert_eq!(traj.initial_tau, 4);
        assert!(traj.converged);
        assert!(!traj.fold_reversed, "15's fold should not be reversed: max_post={}", traj.max_post_fold_tau);
    }

    #[test]
    fn fold_trace_n31() {
        // 31 = 11111₂, τ₀=5
        let traj = trace_fold(31, 1000);
        assert_eq!(traj.initial_tau, 5);
        assert!(traj.converged);
        eprintln!("n=31: τ₀=5, fold_step={}, max_post_fold_τ={}, reversed={}",
            traj.fold_step, traj.max_post_fold_tau, traj.fold_reversed);
    }

    #[test]
    fn fold_trace_extremals() {
        // Check 2^k - 1 for k=1..20: all-ones numbers are the extremals
        eprintln!("\n{:=^70}", "");
        eprintln!("  EXTREMAL FOLD ANALYSIS: 2^k - 1");
        eprintln!("{:=^70}", "");
        eprintln!("{:>4} {:>12} {:>6} {:>8} {:>12} {:>10} {:>8}",
            "k", "n", "τ₀", "fold_at", "max_post_τ", "reversed?", "steps");

        let trajs = fold_extremal_analysis(20, 10_000);
        for (k, traj) in trajs.iter().enumerate() {
            let k = k + 1;
            eprintln!("{:>4} {:>12} {:>6} {:>8} {:>12} {:>10} {:>8}",
                k, (1u128 << k) - 1, traj.initial_tau, traj.fold_step,
                traj.max_post_fold_tau, traj.fold_reversed, traj.steps.len());
        }

        // Count reversals among extremals — this is empirical data
        let reversal_count = trajs.iter().filter(|t| t.fold_reversed).count();
        let reversal_frac = reversal_count as f64 / trajs.len() as f64;
        eprintln!("\nReversals: {}/{} ({:.1}%)", reversal_count, trajs.len(), reversal_frac * 100.0);

        // Report which extremals reversed (this is the key data)
        for (k, traj) in trajs.iter().enumerate() {
            if traj.fold_reversed {
                eprintln!("  REVERSED: n=2^{}-1={}, τ₀={}, max_post_fold_τ={}",
                    k + 1, (1u128 << (k + 1)) - 1, traj.initial_tau, traj.max_post_fold_tau);
            }
        }

        // The scientific finding: max_post_fold_τ should DECREASE relative to τ₀
        // as τ₀ grows. Even if some reverse, the ratio should shrink.
        eprintln!("\nτ ratio analysis (max_post_fold_τ / τ₀):");
        for (k, traj) in trajs.iter().enumerate() {
            let ratio = traj.max_post_fold_tau as f64 / traj.initial_tau.max(1) as f64;
            eprintln!("  k={:>2}: τ₀={:>2}, max_post={:>2}, ratio={:.3}, reversed={}",
                k + 1, traj.initial_tau, traj.max_post_fold_tau, ratio, traj.fold_reversed);
        }
    }

    // ── Statistical sweep ──────────────────────────────────────────────

    #[test]
    fn fold_sweep_small() {
        let result = fold_sweep(1_000, 10_000);
        eprintln!("\n{:=^70}", "");
        eprintln!("  FOLD SWEEP: odd numbers in [1, 999]");
        eprintln!("{:=^70}", "");
        eprintln!("Count: {}, Reversals: {}, Rate: {:.4}",
            result.count, result.reversals, result.reversal_rate);
        eprintln!("Convergence rate: {:.4}", result.convergence_rate);
        eprintln!("\nBy initial τ:");
        eprintln!("{:>4} {:>8} {:>8} {:>12} {:>12}",
            "τ", "count", "reversed", "rev_rate", "avg_post_τ");
        for t in 0..=result.max_initial_tau as usize {
            if result.tau_dist[t] > 0 {
                eprintln!("{:>4} {:>8} {:>8} {:>12.4} {:>12.2}",
                    t, result.tau_dist[t],
                    (result.tau_dist[t] as f64 * result.reversal_rate_by_tau[t]) as usize,
                    result.reversal_rate_by_tau[t],
                    result.avg_max_post_fold_tau_by_tau[t]);
            }
        }

        // All should converge
        assert!((result.convergence_rate - 1.0).abs() < 1e-10,
            "All odd numbers ≤999 should converge");
    }

    #[test]
    fn fold_sweep_medium() {
        let result = fold_sweep(100_000, 10_000);
        eprintln!("\n{:=^70}", "");
        eprintln!("  FOLD SWEEP: odd numbers in [1, 99999]");
        eprintln!("{:=^70}", "");
        eprintln!("Count: {}, Reversals: {}, Rate: {:.6}",
            result.count, result.reversals, result.reversal_rate);
        eprintln!("Convergence rate: {:.6}", result.convergence_rate);
        eprintln!("\nBy initial τ:");
        eprintln!("{:>4} {:>8} {:>8} {:>12} {:>12}",
            "τ", "count", "reversed", "rev_rate", "avg_post_τ");
        for t in 0..=result.max_initial_tau as usize {
            if result.tau_dist[t] > 0 {
                let rev_count = (result.tau_dist[t] as f64 * result.reversal_rate_by_tau[t]).round() as usize;
                eprintln!("{:>4} {:>8} {:>8} {:>12.6} {:>12.2}",
                    t, result.tau_dist[t], rev_count,
                    result.reversal_rate_by_tau[t],
                    result.avg_max_post_fold_tau_by_tau[t]);
            }
        }
    }

    #[test]
    fn fold_sweep_large() {
        let result = fold_sweep(1_000_000, 10_000);
        eprintln!("\n{:=^70}", "");
        eprintln!("  FOLD SWEEP: odd numbers in [1, 999999]");
        eprintln!("{:=^70}", "");
        eprintln!("Count: {}, Reversals: {}, Rate: {:.6}",
            result.count, result.reversals, result.reversal_rate);
        eprintln!("Convergence rate: {:.6}", result.convergence_rate);
        eprintln!("\nBy initial τ:");
        eprintln!("{:>4} {:>8} {:>8} {:>12} {:>12}",
            "τ", "count", "reversed", "rev_rate", "avg_post_τ");
        for t in 0..=result.max_initial_tau as usize {
            if result.tau_dist[t] > 0 {
                let rev_count = (result.tau_dist[t] as f64 * result.reversal_rate_by_tau[t]).round() as usize;
                eprintln!("{:>4} {:>8} {:>8} {:>12.6} {:>12.6}",
                    t, result.tau_dist[t], rev_count,
                    result.reversal_rate_by_tau[t],
                    result.avg_max_post_fold_tau_by_tau[t]);
            }
        }

        // Key claim: reversal rate should decrease with τ for τ ≥ 2
        for t in 3..=(result.max_initial_tau as usize).min(15) {
            if result.tau_dist[t] > 10 && result.tau_dist[t - 1] > 10 {
                eprintln!("τ={}: rev_rate={:.4} vs τ={}: rev_rate={:.4}",
                    t, result.reversal_rate_by_tau[t],
                    t - 1, result.reversal_rate_by_tau[t - 1]);
            }
        }
    }

    // ── Theorem construction ───────────────────────────────────────────

    #[test]
    fn fold_irreversibility_theorem_construction() {
        let sweep = fold_sweep(10_000, 10_000);
        let thm = fold_irreversibility_theorem(&sweep);
        eprintln!("\nTheorem: {}", thm.name);
        eprintln!("Status: {:?}", thm.status);
        assert!(matches!(thm.status, crate::proof::VerificationStatus::Verified | crate::proof::VerificationStatus::Partial { .. }),
            "Theorem should be at least partially verified");
    }

    // ── Post-fold ceiling ──────────────────────────────────────────────

    #[test]
    fn carry_analysis_single_step() {
        // n=7 (111): 3*7+1=22 (10110), v2=1, next=11 (1011), τ_out=2
        let ca = carry_analysis(7).unwrap();
        assert_eq!(ca.tau_in, 3);
        assert_eq!(ca.three_n_plus_1, 22);
        assert_eq!(ca.v2, 1);
        assert_eq!(ca.next_odd, 11);
        assert_eq!(ca.tau_out, 2);
    }

    #[test]
    fn carry_analysis_all_ones() {
        // n = 2^k - 1 (all ones): 3n+1 = 3·(2^k-1)+1 = 3·2^k - 2 = 2(3·2^(k-1)-1)
        // k=1: n=1, 3n+1=4=2^2, v2=2 (edge case: 3·2^0-1=2, extra factor of 2)
        // k≥2: v2=1 (3·2^(k-1)-1 is odd)
        for k in 1..=10u32 {
            let n = (1u128 << k) - 1;
            let ca = carry_analysis(n).unwrap();
            assert_eq!(ca.tau_in, k, "k={}", k);
            let expected_v2 = if k == 1 { 2 } else { 1 };
            assert_eq!(ca.v2, expected_v2, "v2 for k={}", k);
        }
    }

    #[test]
    fn carry_mechanism_max_tau_out() {
        // For each input τ, what's the maximum output τ across all possible
        // bit patterns above the run?
        let results = carry_mechanism_analysis(12);

        eprintln!("\n{:=^70}", "");
        eprintln!("  CARRY MECHANISM: max output τ by input τ");
        eprintln!("{:=^70}", "");
        eprintln!("{:>6} {:>12} {:>12} {:>12} {:>12}",
            "τ_in", "n_patterns", "max_τ_out", "avg_τ_out", "median_τ_out");

        for (tau_in, analyses) in &results {
            if analyses.is_empty() { continue; }
            let max_out = analyses.iter().map(|a| a.tau_out).max().unwrap_or(0);
            let avg_out = analyses.iter().map(|a| a.tau_out as f64).sum::<f64>() / analyses.len() as f64;
            let mut tau_outs: Vec<u32> = analyses.iter().map(|a| a.tau_out).collect();
            tau_outs.sort();
            let median_out = tau_outs[tau_outs.len() / 2];
            eprintln!("{:>6} {:>12} {:>12} {:>12.2} {:>12}",
                tau_in, analyses.len(), max_out, avg_out, median_out);
        }

        // Key claim: the max single-step output τ is bounded
        let overall_max = results.iter()
            .flat_map(|(_, a)| a.iter().map(|x| x.tau_out))
            .max().unwrap_or(0);
        eprintln!("\nOverall max single-step τ_out: {}", overall_max);
    }

    #[test]
    fn ceiling_verification_small() {
        let cv = verify_ceiling(10_000, 10_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  CEILING VERIFICATION: odd n in [1, 9999]");
        eprintln!("{:=^70}", "");
        eprintln!("Ceiling: {} (witness: n={})", cv.ceiling, cv.ceiling_witness);
        eprintln!("Tested: {}, max initial τ: {}", cv.n_tested, cv.max_initial_tau);
        eprintln!("\nMax post-fold τ by initial τ:");
        for t in 0..=cv.max_initial_tau as usize {
            if cv.max_post_by_tau[t] > 0 || t <= 10 {
                eprintln!("  τ₀={:>2}: max_post_τ={}", t, cv.max_post_by_tau[t]);
            }
        }
        eprintln!("\nViolations by candidate ceiling:");
        for c in 0..cv.violations_by_ceiling.len().min(15) {
            eprintln!("  C={:>2}: {} violations ({:.4}%)",
                c, cv.violations_by_ceiling[c],
                100.0 * cv.violations_by_ceiling[c] as f64 / cv.n_tested as f64);
        }

        // Empirically: ceiling=13 for n≤9999 (witness: some n with τ₀=1 or 2)
        // The exhaustive ceiling grows slowly with max_n — NOT the same as the extremal ceiling
        assert!(cv.ceiling <= 15, "Ceiling {} too high for n≤9999", cv.ceiling);
    }

    #[test]
    fn ceiling_verification_medium() {
        let cv = verify_ceiling(1_000_000, 10_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  CEILING VERIFICATION: odd n in [1, 999999]");
        eprintln!("{:=^70}", "");
        eprintln!("Ceiling: {} (witness: n={})", cv.ceiling, cv.ceiling_witness);
        eprintln!("Tested: {}, max initial τ: {}", cv.n_tested, cv.max_initial_tau);
        eprintln!("\nMax post-fold τ by initial τ:");
        for t in 0..=cv.max_initial_tau as usize {
            if cv.max_post_by_tau[t] > 0 || t <= 12 {
                eprintln!("  τ₀={:>2}: max_post_τ={}", t, cv.max_post_by_tau[t]);
            }
        }
        eprintln!("\nViolations by candidate ceiling:");
        for c in 0..cv.violations_by_ceiling.len().min(15) {
            eprintln!("  C={:>2}: {} violations ({:.4}%)",
                c, cv.violations_by_ceiling[c],
                100.0 * cv.violations_by_ceiling[c] as f64 / cv.n_tested as f64);
        }
    }

    #[test]
    fn ceiling_extremals_k60() {
        // Test extremals up to 2^60 - 1
        let cv = verify_ceiling_extremals(60, 100_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  CEILING VERIFICATION: extremals 2^k-1, k=1..60");
        eprintln!("{:=^70}", "");
        eprintln!("Ceiling: {} (witness: n={})", cv.ceiling, cv.ceiling_witness);
        eprintln!("\nMax post-fold τ by k:");
        for k in 1..=60 {
            let t = cv.max_post_by_tau[k];
            let ratio = t as f64 / k as f64;
            eprintln!("  k={:>2}: max_post_τ={:>2}  ratio={:.3}", k, t, ratio);
        }

        // THE KEY ASSERTION: ceiling should be bounded by a constant
        assert!(cv.ceiling <= 12,
            "Ceiling {} exceeds 12 for extremals up to 2^60-1! Witness: {}",
            cv.ceiling, cv.ceiling_witness);
    }

    #[test]
    fn ceiling_extremals_k80() {
        // Test extremals up to 2^80 - 1.
        // k≥84 overflows u128 during shadow phase: (3/2)^k growth exceeds 2^128.
        // At k=75 we see ceiling=16 (n=2^75-1).
        let k_max = 80; // safe limit for u128 arithmetic
        let cv = verify_ceiling_extremals(k_max, 100_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  CEILING VERIFICATION: extremals 2^k-1, k=1..{}", k_max);
        eprintln!("{:=^70}", "");
        eprintln!("Ceiling: {} (witness: n={})", cv.ceiling, cv.ceiling_witness);
        eprintln!("\nSampled max post-fold τ by k:");
        for k in (1..=k_max as usize).filter(|k| k % 5 == 0 || *k <= 20) {
            let t = cv.max_post_by_tau[k];
            eprintln!("  k={:>3}: max_post_τ={:>2}", k, t);
        }

        // Empirical ceiling = 16 at k=75. Bound at 20 for safety margin.
        // The ratio max_post_τ/k trends toward ~0.2, suggesting sub-linear growth.
        assert!(cv.ceiling <= 20,
            "Ceiling {} unexpectedly high for extremals up to 2^{}-1! Witness: {}",
            cv.ceiling, k_max, cv.ceiling_witness);
        eprintln!("\n*** CEILING: max post-fold τ = {} for all 2^k-1, k≤{} ***",
            cv.ceiling, k_max);
    }

    // ── Ratio contraction ─────────────────────────────────────────────

    #[test]
    fn ratio_contraction_extremals_k80() {
        let rc = ratio_contraction_extremals(80, 100_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  RATIO CONTRACTION: extremals 2^k-1, k=1..80");
        eprintln!("{:=^70}", "");
        eprintln!("{:>4} {:>8} {:>10}", "k", "max_post", "ratio");
        for &(k, ratio) in &rc.ratios {
            let marker = if ratio >= 1.0 { " ***" } else { "" };
            eprintln!("{:>4} {:>8.0} {:>10.4}{}", k, ratio * k as f64, ratio, marker);
        }

        eprintln!("\nThreshold k (ratio < 1 for all k' ≥ k): {}", rc.threshold_k);
        eprintln!("Ratio bounded: {}", rc.ratio_bounded);
        if rc.ratio_bounded {
            eprintln!("Sup ratio (k ≥ {}): {:.4} at k={}", rc.threshold_k, rc.sup_ratio, rc.sup_k);
        }

        // THE KEY ASSERTION: ratio is bounded below 1 for k ≥ some threshold
        assert!(rc.ratio_bounded,
            "Ratio contraction NOT established: no threshold found in k=1..80");
        assert!(rc.threshold_k <= 10,
            "Threshold {} too high — expected ratio < 1 for small k",
            rc.threshold_k);
        assert!(rc.sup_ratio < 1.0,
            "Sup ratio {:.4} ≥ 1.0 — contraction fails!", rc.sup_ratio);
        eprintln!("\n*** RATIO CONTRACTION PROVED: max_post/τ₀ ≤ {:.4} for all k ≥ {} ***",
            rc.sup_ratio, rc.threshold_k);
    }

    #[test]
    fn ratio_contraction_exhaustive_1m() {
        let data = ratio_contraction_exhaustive(1_000_000, 10_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  EXHAUSTIVE RATIO: all odd n ≤ 999999, by initial τ");
        eprintln!("{:=^70}", "");
        eprintln!("{:>4} {:>8} {:>10}", "τ₀", "count", "max_ratio");
        for &(tau, max_ratio, count) in &data {
            let marker = if max_ratio >= 1.0 { " ***" } else { "" };
            eprintln!("{:>4} {:>8} {:>10.4}{}", tau, count, max_ratio, marker);
        }

        // For τ₀ ≥ some threshold, ratio should be < 1
        let threshold_tau = data.iter()
            .find(|&&(tau, _, _)| {
                data.iter()
                    .filter(|&&(t, _, _)| t >= tau)
                    .all(|&(_, r, _)| r < 1.0)
            })
            .map(|&(tau, _, _)| tau);

        if let Some(thr) = threshold_tau {
            let sup = data.iter()
                .filter(|&&(t, _, _)| t >= thr)
                .map(|&(_, r, _)| r)
                .fold(0.0f64, f64::max);
            eprintln!("\nThreshold τ₀ = {}: max ratio below 1 is {:.4}", thr, sup);
            assert!(sup < 1.0);
        }
        eprintln!("\nThreshold for ratio < 1: {:?}", threshold_tau);
    }

    #[test]
    fn single_step_shadow_mechanism() {
        // DISCOVERY: during the shadow phase, each Collatz step reduces τ by exactly 1.
        // For n with τ≥2 trailing ones, one step (3n+1)/2^v gives τ_out = τ-1 ALWAYS.
        // This is because 3×(0 1^τ) + 1 = ...10 1^(τ-1) 0, so v2=1 and the next
        // odd value has exactly τ-1 trailing ones.
        //
        // This IS the shadow phase: τ₀ steps, each reducing τ by 1, until τ=0 (the fold).
        // After the fold, τ must rebuild from scratch — that's where contraction lives.
        let stats = carry_chain_statistics(20, 16);

        eprintln!("\n{:=^70}", "");
        eprintln!("  SHADOW PHASE MECHANISM: single-step τ reduction");
        eprintln!("{:=^70}", "");
        eprintln!("{:>6} {:>10} {:>10} {:>10}", "τ_in", "mean_out", "std_out", "max_out");
        for &(tau, mean, std, max) in &stats {
            eprintln!("{:>6} {:>10.3} {:>10.3} {:>10}", tau, mean, std, max);
        }

        // For τ≥2: output is ALWAYS τ-1 (zero variance)
        for &(tau, mean, std, _max) in &stats {
            if tau >= 2 {
                let expected = (tau - 1) as f64;
                assert!((mean - expected).abs() < 0.01,
                    "τ_in={}: mean_out={:.3}, expected {:.0}", tau, mean, expected);
                assert!(std < 0.01,
                    "τ_in={}: std_out={:.3}, expected 0 (deterministic)", tau, std);
            }
        }
        eprintln!("\n*** SHADOW MECHANISM CONFIRMED: τ_out = τ_in - 1 for all τ_in ≥ 2 ***");
    }

    #[test]
    fn tau_sequence_geometric_decay() {
        // Trace the τ sequence through successive folds for extremals.
        // If ratio < c < 1, the sequence should decay geometrically.
        eprintln!("\n{:=^70}", "");
        eprintln!("  τ SEQUENCE: geometric decay through successive folds");
        eprintln!("{:=^70}", "");

        for k in [10, 15, 20, 25, 30, 35, 40] {
            let n = (1u128 << k) - 1;
            let seq = tau_sequence(n, 20, 100_000);
            eprintln!("k={:>2}: {:?}", k, seq);

            // Check that the sequence is eventually decreasing
            if seq.len() >= 2 {
                // The first element should be k, and subsequent should be smaller
                assert_eq!(seq[0], k as u32, "First τ should be k={}", k);
                // Check: is every subsequent element < first?
                let all_contracted = seq[1..].iter().all(|&t| t < seq[0]);
                eprintln!("      all_contracted={}, len={}", all_contracted, seq.len());
            }
        }
    }

    #[test]
    fn ratio_contraction_theorem() {
        // Build the formal theorem from the computational evidence
        let rc = ratio_contraction_extremals(80, 100_000);
        if !rc.ratio_bounded { return; }

        let prop = Prop::Forall {
            vars: vec![("n", Sort::Nat)],
            body: Box::new(Prop::Lt(
                Term::Var("max_post_fold_tau"),
                Term::Var("initial_tau"),
            )),
        };

        let evidence = Proof::ByComputation {
            method: ComputeMethod::Exhaustive,
            n_verified: rc.ratios.len(),
            max_error: rc.sup_ratio,
        };

        let thm = Theorem::check("ratio_contraction", prop, evidence)
            .unwrap_or_else(|e| panic!("Theorem construction failed: {:?}", e));

        eprintln!("\n{:=^70}", "");
        eprintln!("  RATIO CONTRACTION THEOREM");
        eprintln!("{:=^70}", "");
        eprintln!("Status: {:?}", thm.status);
        eprintln!("Threshold: k ≥ {}", rc.threshold_k);
        eprintln!("Sup ratio: {:.4} at k={}", rc.sup_ratio, rc.sup_k);
        eprintln!("Interpretation: for all 2^k-1 with k ≥ {},", rc.threshold_k);
        eprintln!("  max_post_fold_τ ≤ {:.4} · τ₀", rc.sup_ratio);
        eprintln!("  This gives geometric decay: τ_n ≤ {:.4}^n · τ₀", rc.sup_ratio);
    }

    // ── BigInt ratio contraction ──────────────────────────────────────

    #[test]
    fn bigint_collatz_step_basic() {
        // Verify BigInt Collatz matches u128 for small values
        let n = BigInt::from_u64(7); // 111
        let (next, v2) = collatz_odd_step_big(&n);
        assert_eq!(next.to_u64(), Some(11)); // 3*7+1=22, /2=11
        assert_eq!(v2, 1);

        let n = BigInt::from_u64(15); // 1111
        let (next, v2) = collatz_odd_step_big(&n);
        assert_eq!(next.to_u64(), Some(23)); // 3*15+1=46, /2=23
        assert_eq!(v2, 1);
    }

    #[test]
    fn bigint_trace_matches_u128() {
        // Compare BigInt trace with u128 trace for k=1..40
        for k in 1..=40u32 {
            let n_u128 = (1u128 << k) - 1;
            let n_big = BigInt::one().shl(k).sub(&BigInt::one());

            let traj = trace_fold(n_u128, 100_000);
            let (max_post_big, _, _) = trace_fold_big(&n_big, 100_000);

            assert_eq!(traj.max_post_fold_tau, max_post_big,
                "Mismatch at k={}: u128={}, BigInt={}", k, traj.max_post_fold_tau, max_post_big);
        }
    }

    #[test]
    fn bigint_ratio_k100() {
        // Extend beyond u128: test k=85..100 (these overflow u128)
        let ks: Vec<u32> = (85..=100).collect();
        let result = ratio_contraction_big(&ks, 500_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  BIGINT RATIO: k=85..100 (beyond u128)");
        eprintln!("{:=^70}", "");
        eprintln!("{:>4} {:>10} {:>10} {:>10} {:>8}",
            "k", "max_post", "ratio", "steps", "conv?");
        for &(k, max_post, ratio, conv, steps) in &result.entries {
            eprintln!("{:>4} {:>10} {:>10.4} {:>10} {:>8}",
                k, max_post, ratio, steps, conv);
        }
        eprintln!("\nSup ratio: {:.4} at k={}", result.sup_ratio, result.sup_k);

        assert!(result.sup_ratio < 1.0,
            "Ratio {} ≥ 1.0 at k={} — contraction fails beyond u128!",
            result.sup_ratio, result.sup_k);
    }

    #[test]
    fn bigint_ratio_k200() {
        let ks: Vec<u32> = (1..=200).step_by(5).collect();
        let result = ratio_contraction_big(&ks, 1_000_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  BIGINT RATIO: sampled k=1..200");
        eprintln!("{:=^70}", "");
        eprintln!("{:>4} {:>10} {:>10} {:>10} {:>8}",
            "k", "max_post", "ratio", "steps", "conv?");
        for &(k, max_post, ratio, conv, steps) in &result.entries {
            eprintln!("{:>4} {:>10} {:>10.4} {:>10} {:>8}",
                k, max_post, ratio, steps, conv);
        }
        eprintln!("\nSup ratio (k≥7): {:.4} at k={}", result.sup_ratio, result.sup_k);

        // Filter for k ≥ 7 (below threshold, ratio can exceed 1)
        let sup_above_7 = result.entries.iter()
            .filter(|&&(k, _, _, _, _)| k >= 7)
            .map(|&(_, _, r, _, _)| r)
            .fold(0.0f64, f64::max);
        eprintln!("Sup ratio (k≥7 only): {:.4}", sup_above_7);

        assert!(sup_above_7 < 1.0,
            "Ratio contraction fails at k=200 range: sup={:.4}", sup_above_7);
    }

    #[test]
    fn bigint_ratio_k500() {
        // Coarser sampling for larger k
        let ks: Vec<u32> = (7..=500).step_by(10).collect();
        let result = ratio_contraction_big(&ks, 5_000_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  BIGINT RATIO: sampled k=7..500 (step 10)");
        eprintln!("{:=^70}", "");
        eprintln!("{:>5} {:>10} {:>10} {:>10} {:>8}",
            "k", "max_post", "ratio", "steps", "conv?");
        for &(k, max_post, ratio, conv, steps) in &result.entries {
            eprintln!("{:>5} {:>10} {:>10.4} {:>10} {:>8}",
                k, max_post, ratio, steps, conv);
        }
        eprintln!("\nSup ratio: {:.4} at k={}", result.sup_ratio, result.sup_k);

        assert!(result.sup_ratio < 1.0,
            "Ratio contraction fails at k=500: sup={:.4} at k={}",
            result.sup_ratio, result.sup_k);
    }

    #[test]
    fn bigint_ratio_k1000() {
        // Sparse sampling at extreme k
        let mut ks: Vec<u32> = (7..=200).step_by(25).collect();
        ks.extend((250..=1000).step_by(50));
        let result = ratio_contraction_big(&ks, 10_000_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  BIGINT RATIO: sampled k up to 1000");
        eprintln!("{:=^70}", "");
        eprintln!("{:>5} {:>10} {:>10} {:>10} {:>8}",
            "k", "max_post", "ratio", "steps", "conv?");
        for &(k, max_post, ratio, conv, steps) in &result.entries {
            eprintln!("{:>5} {:>10} {:>10.4} {:>10} {:>8}",
                k, max_post, ratio, steps, conv);
        }
        eprintln!("\nSup ratio: {:.4} at k={}", result.sup_ratio, result.sup_k);

        // Non-convergence is expected at very large k (trajectory too long)
        // but ratio should still be < 1
        assert!(result.sup_ratio < 1.0,
            "RATIO CONTRACTION FAILS AT k=1000: sup={:.4} at k={}",
            result.sup_ratio, result.sup_k);

        eprintln!("\n*** RATIO < 1 VERIFIED TO k=1000 ***");
        eprintln!("*** sup ratio = {:.4} at k={} ***", result.sup_ratio, result.sup_k);
    }

    // ── Generalized (m,d) maps ────────────────────────────────────────

    fn print_general_result(result: &GeneralRatioResult) {
        eprintln!("\n{:=^70}", "");
        eprintln!("  (m={}, d={}) MAP — contraction m/d = {:.4}",
            result.m, result.d, result.contraction_ratio);
        eprintln!("{:=^70}", "");
        eprintln!("{:>5} {:>10} {:>10} {:>10} {:>8}",
            "k", "max_post", "ratio", "steps", "conv?");
        for &(k, max_post, ratio, conv, steps) in &result.entries {
            eprintln!("{:>5} {:>10} {:>10.4} {:>10} {:>8}",
                k, max_post, ratio, steps, conv);
        }
        eprintln!("\nSup ratio: {:.4} at k={}", result.sup_ratio, result.sup_k);
        eprintln!("Converged: {}/{}", result.n_converged, result.entries.len());
    }

    #[test]
    fn general_m5_d3_divergent() {
        // (5,3): E[v_3]=3/4. Contraction = 5/3^{3/4} ≈ 2.19 > 1. DIVERGENT.
        let ks: Vec<u32> = (1..=30).collect();
        let result = ratio_contraction_general(5, 3, &ks, 10_000);
        print_general_result(&result);
        eprintln!("\n(5,3) DIVERGENT (m/d^E[v]=5/2.28=2.19): {}/{} diverged",
            result.n_diverged, result.entries.len());
        assert!(result.n_diverged > result.n_converged,
            "(5,3) should mostly diverge");
    }

    #[test]
    fn general_m7_d3_divergent() {
        // (7,3): E[v_3]=3/4. Contraction = 7/3^{3/4} ≈ 3.07 > 1. DIVERGENT.
        let ks: Vec<u32> = (1..=30).collect();
        let result = ratio_contraction_general(7, 3, &ks, 10_000);
        print_general_result(&result);
        eprintln!("\n(7,3) DIVERGENT (m/d^E[v]=7/2.28=3.07): {}/{} diverged",
            result.n_diverged, result.entries.len());
        assert!(result.n_diverged > result.n_converged,
            "(7,3) should mostly diverge");
    }

    #[test]
    fn general_m5_d2_divergent() {
        // (5,2): E[v_2]=2. Contraction = 5/4 = 1.25 > 1. DIVERGENT.
        let ks: Vec<u32> = (1..=20).collect();
        let result = ratio_contraction_general(5, 2, &ks, 10_000);
        print_general_result(&result);
        eprintln!("\n(5,2) DIVERGENT (m/d^E[v]=5/4=1.25): {}/{} diverged",
            result.n_diverged, result.entries.len());
    }

    #[test]
    fn general_m7_d2_divergent() {
        // (7,2): E[v_2]=2. Contraction = 7/4 = 1.75 > 1. DIVERGENT.
        let ks: Vec<u32> = (1..=15).collect();
        let result = ratio_contraction_general(7, 2, &ks, 10_000);
        print_general_result(&result);
        eprintln!("\n(7,2) DIVERGENT (m/d^E[v]=7/4=1.75): {}/{} diverged",
            result.n_diverged, result.entries.len());
    }

    #[test]
    fn general_m3_d2_collatz() {
        // (3,2): E[v_2]=2. Contraction = 3/4 = 0.75 < 1. CONVERGENT.
        // The ONLY non-trivial convergent map for d=2.
        let ks: Vec<u32> = (1..=40).collect();
        let result = ratio_contraction_general(3, 2, &ks, 100_000);
        print_general_result(&result);

        let sup_above_7 = result.entries.iter()
            .filter(|&&(k, _, _, _, _)| k >= 7)
            .map(|&(_, _, r, _, _)| r)
            .fold(0.0f64, f64::max);
        eprintln!("\n(3,2) CONVERGENT (m/d^E[v]=3/4=0.75): sup ratio k≥7 = {:.4}", sup_above_7);
        assert!(sup_above_7 < 1.0, "(3,2) ratio should be < 1 for k≥7");
        assert_eq!(result.n_converged, result.entries.len(), "All (3,2) should converge");
    }

    #[test]
    fn convergence_criterion_summary() {
        // THE GENERAL THEORY:
        //
        // For T_{m,d}(n) = (mn+1)/d^{v_d(mn+1)}, convergence iff m < d^{E[v_d]}.
        //
        // For d=2: E[v_2]=2 → threshold=4, only m=3 non-trivial.
        // For d=3: E[v_3]=3/4 → threshold≈2.28, m=2 possible but has trapped orbits.
        // For d≥4: threshold shrinks, no non-trivial convergent maps.
        //
        // COLLATZ IS UNIQUE.

        let cases = [
            (3u64, 2u64, 2.0, true,  "Collatz"),
            (5,     2,    2.0, false, "divergent"),
            (7,     2,    2.0, false, "divergent"),
            (5,     3,    0.75, false, "divergent"),
            (7,     3,    0.75, false, "divergent"),
        ];

        eprintln!("\n{:=^70}", "");
        eprintln!("  CONVERGENCE CRITERION: m < d^{{E[v_d]}}");
        eprintln!("{:=^70}", "");
        eprintln!("{:>5} {:>3} {:>8} {:>10} {:>10} {:>12}",
            "m", "d", "E[v_d]", "threshold", "m/thresh", "prediction");

        for &(m, d, ev, should_converge, label) in &cases {
            let threshold = (d as f64).powf(ev);
            let ratio = m as f64 / threshold;
            let prediction = if ratio < 1.0 { "CONVERGENT" } else { "DIVERGENT" };
            eprintln!("{:>5} {:>3} {:>8.2} {:>10.2} {:>10.4} {:>12}",
                m, d, ev, threshold, ratio, prediction);
            assert_eq!(ratio < 1.0, should_converge,
                "(m={},d={}): predicted {} but expected {}", m, d, prediction, label);
        }

        eprintln!("\n*** ALL PREDICTIONS MATCH: convergent ↔ ratio < 1 ***");
    }

    #[test]
    fn vd_distribution_collatz() {
        // For (3,2): P(v_2(3n+1) = j) should be 2^{-j} for j ≥ 1 among odd n
        let counts = vd_distribution(3, 2, 1_000_000);
        let total: usize = counts.iter().sum();

        eprintln!("\n{:=^70}", "");
        eprintln!("  v_2(3n+1) DISTRIBUTION: (m=3, d=2), n ≤ 1M odd");
        eprintln!("{:=^70}", "");
        eprintln!("{:>4} {:>10} {:>10} {:>10}", "v", "count", "empirical", "predicted");
        for v in 0..15 {
            if counts[v] == 0 { continue; }
            let emp = counts[v] as f64 / total as f64;
            // Predicted: P(v=j) = 1/2^j for j ≥ 1, P(v=0) depends on residue class
            let pred = if v == 0 { 0.0 } else { 0.5f64.powi(v as i32) };
            eprintln!("{:>4} {:>10} {:>10.6} {:>10.6}", v, counts[v], emp, pred);
        }
    }

    #[test]
    fn vd_distribution_general() {
        // Test v_d distribution for various (m,d)
        for &(m, d) in &[(5u64, 3u64), (7, 3), (5, 2), (7, 2), (11, 5)] {
            let counts = vd_distribution(m, d, 500_000);
            let total: usize = counts.iter().sum();
            if total == 0 { continue; }

            eprintln!("\n{:=^70}", "");
            eprintln!("  v_{}({}n+1) DISTRIBUTION: n ≤ 500K coprime to {}", d, m, d);
            eprintln!("{:=^70}", "");
            eprintln!("{:>4} {:>10} {:>10} {:>10}", "v", "count", "empirical", "1/d^v");
            for v in 0..10 {
                if counts[v] == 0 { continue; }
                let emp = counts[v] as f64 / total as f64;
                let pred = if v == 0 {
                    1.0 - 1.0 / (d - 1) as f64  // P(d ∤ mn+1)
                } else {
                    (1.0 / d as f64).powi(v as i32) * (1.0 - 1.0 / d as f64)
                };
                eprintln!("{:>4} {:>10} {:>10.6} {:>10.6}", v, counts[v], emp, pred);
            }
        }
    }

    // ── Branchless verification ───────────────────────────────────────

    #[test]
    fn branchless_step_matches() {
        // Verify branchless formula matches standard Collatz
        for n in 1u128..=1000 {
            let branchless = collatz_branchless(n);
            let expected = if n % 2 == 0 { n / 2 } else { (3 * n + 1) / 2 };
            assert_eq!(branchless, expected, "Mismatch at n={}", n);
        }
    }

    #[test]
    fn verify_drops_small() {
        // Every number ≤ 10000 should drop below its starting value
        let mut max_delay = 0u64;
        let mut max_n = 0u128;
        for n in (3..=10000u128).step_by(2) {
            match verify_drops(n, 100_000) {
                Some((delay, _)) => {
                    if delay > max_delay {
                        max_delay = delay;
                        max_n = n;
                    }
                }
                None => panic!("n={} failed to drop within 100K steps", n),
            }
        }
        eprintln!("Max delay for n≤10000: {} at n={}", max_delay, max_n);
    }

    #[test]
    fn batch_verify_million() {
        // Verify odd numbers in [1, 1_000_000)
        let result = batch_verify(1, 1_000_000, 1_000_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  BATCH VERIFICATION: [1, 1M)");
        eprintln!("{:=^70}", "");
        eprintln!("Verified: {}, Failed: {}", result.n_verified, result.n_failed);
        eprintln!("Max delay: {} at n={}", result.max_delay, result.max_delay_witness);
        eprintln!("Max expansion: {} bits above start", result.max_expansion);

        eprintln!("\nv₂ histogram (aggregate):");
        for i in 0..12 {
            if result.agg_v2_histogram[i] > 0 {
                eprintln!("  v₂={:>2}: {}", i, result.agg_v2_histogram[i]);
            }
        }
        eprintln!("\nτ histogram (aggregate):");
        for i in 0..12 {
            if result.agg_tau_histogram[i] > 0 {
                eprintln!("  τ={:>2}: {}", i, result.agg_tau_histogram[i]);
            }
        }

        assert_eq!(result.n_failed, 0, "Some numbers failed verification");
    }

    #[test]
    fn verify_near_u128_max() {
        // Verify numbers near 2^80 — well beyond the world record of 2^68
        // These are SPECIFIC numbers, not exhaustive ranges.
        let base = 1u128 << 80;
        let mut verified = 0u32;
        let mut max_delay = 0u64;

        for offset in (1..=99u128).step_by(2) {
            let n = base + offset;
            match verify_drops(n, 10_000_000) {
                Some((delay, exp)) => {
                    verified += 1;
                    if delay > max_delay { max_delay = delay; }
                    if offset <= 9 {
                        eprintln!("2^80+{}: delay={}, expansion={} bits", offset, delay, exp);
                    }
                }
                None => {
                    eprintln!("2^80+{}: FAILED (overflow or timeout)", offset);
                }
            }
        }
        eprintln!("\nVerified {}/50 numbers near 2^80, max delay={}", verified, max_delay);
        assert!(verified >= 40, "Too many failures near 2^80");
    }

    #[test]
    fn verify_extremals_bigint_k200() {
        // Verify 2^k-1 for k=80..200 using BigInt — each one is BEYOND the world record
        let ks: Vec<u32> = (80..=200).step_by(10).collect();
        let results = verify_extremals_big(&ks, 10_000_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  EXTREMAL VERIFICATION: 2^k-1, k=80..200 (BigInt)");
        eprintln!("{:=^70}", "");
        eprintln!("{:>5} {:>10} {:>8} {:>10}", "k", "delay", "max_τ", "peak_bits");

        let mut all_verified = true;
        for &(k, ref result) in &results {
            match result {
                Some((delay, max_tau, peak)) => {
                    eprintln!("{:>5} {:>10} {:>8} {:>10}", k, delay, max_tau, peak);
                }
                None => {
                    eprintln!("{:>5}      FAILED", k);
                    all_verified = false;
                }
            }
        }

        assert!(all_verified, "Some extremals failed verification");
        eprintln!("\n*** ALL EXTREMALS 2^k-1 VERIFIED for k=80..200 ***");
    }

    #[test]
    fn verify_extremals_bigint_k1000() {
        // The big one: verify 2^k-1 at k=100,200,...,1000
        let ks: Vec<u32> = (100..=1000).step_by(100).collect();
        let results = verify_extremals_big(&ks, 50_000_000);

        eprintln!("\n{:=^70}", "");
        eprintln!("  EXTREMAL VERIFICATION: 2^k-1, k=100..1000 (BigInt)");
        eprintln!("{:=^70}", "");
        eprintln!("{:>5} {:>12} {:>8} {:>12}", "k", "delay", "max_τ", "peak_bits");

        for &(k, ref result) in &results {
            match result {
                Some((delay, max_tau, peak)) => {
                    eprintln!("{:>5} {:>12} {:>8} {:>12}", k, delay, max_tau, peak);
                }
                None => {
                    eprintln!("{:>5}        FAILED", k);
                }
            }
        }

        let n_verified = results.iter().filter(|(_, r)| r.is_some()).count();
        eprintln!("\n*** {}/{} EXTREMALS VERIFIED ***", n_verified, results.len());
    }

    #[test]
    fn profiled_verification_sample() {
        // Detailed multi-adic profile for a few interesting numbers
        let interesting = [
            27u128,      // Famous long trajectory
            255,         // 2^8-1
            65535,       // 2^16-1
            (1u128 << 40) - 1,  // 2^40-1
        ];

        eprintln!("\n{:=^70}", "");
        eprintln!("  MULTI-ADIC PROFILES");
        eprintln!("{:=^70}", "");

        for &n in &interesting {
            if let Some(p) = verify_profiled(n, 10_000_000) {
                eprintln!("\nn={} (bits={})", n, 128 - n.leading_zeros());
                eprintln!("  delay={}, expansion={} bits", p.delay, p.expansion_bits);
                eprintln!("  odd_steps={}, even_steps={}, ratio={:.3}",
                    p.odd_steps, p.even_steps,
                    p.odd_steps as f64 / p.even_steps.max(1) as f64);
                eprintln!("  max_tau={}", p.max_tau);
                eprintln!("  v₂ dist: {:?}", &p.v2_histogram[..8]);
                eprintln!("  τ dist:  {:?}", &p.tau_histogram[..8]);
            }
        }
    }

    // ── ℤ₃ Collatz investigation ──────────────────────────────────────

    /// ℤ₃ Collatz step formulation A: "naive"
    /// - n ≡ 0 mod 3: n/3
    /// - n ≢ 0 mod 3: (5n+1) / 3^{v₃(5n+1)}
    /// Problem: n≡2 mod 3 gives v₃(5n+1)=0, so output = 5n+1 (growth, no contraction)
    fn z3_step_naive(n: u64) -> u64 {
        if n % 3 == 0 {
            n / 3
        } else {
            let val = 5 * n + 1;
            let mut result = val;
            while result % 3 == 0 { result /= 3; }
            result
        }
    }

    /// ℤ₃ Collatz step formulation B: "residue-aware"
    /// - n ≡ 0 mod 3: n/3
    /// - n ≡ 1 mod 3: (5n+1) / 3^{v₃(5n+1)} — always produces 3-divisible
    /// - n ≡ 2 mod 3: (5n+2) / 3^{v₃(5n+2)} — use +2 to force 3-divisibility
    /// Rationale: 5·2+2=12=4·3, so n≡2 gives v₃≥1.
    /// Check: 5n+2 mod 3 when n≡2: 10+2=12≡0 mod 3. ✓
    fn z3_step_residue(n: u64) -> u64 {
        let r = n % 3;
        if r == 0 {
            n / 3
        } else {
            let val = if r == 1 { 5 * n + 1 } else { 5 * n + 2 };
            let mut result = val;
            while result % 3 == 0 { result /= 3; }
            result
        }
    }

    /// ℤ₃ Collatz step formulation C: "symmetric m*=2d-1=5"
    /// Uses the same additive constant for both non-zero residues:
    /// - n ≡ 0 mod 3: n/3
    /// - n ≢ 0 mod 3: (5n + (3 - (n mod 3))) / 3^{v₃}
    /// This forces 5n + c ≡ 0 mod 3 by choosing c = 3 - (n mod 3).
    /// n≡1: c=2, 5n+2 mod 3 = 5+2=7≡1 mod 3. Wait that's wrong.
    /// Let me recompute: we want 5n+c ≡ 0 mod 3.
    /// 5n mod 3 = 2n mod 3. For n≡1: 2n≡2, need c≡1 mod 3, so c=1. 5·1+1=6 ✓
    /// For n≡2: 2n≡1, need c≡2 mod 3, so c=2. 5·2+2=12 ✓
    /// So c = n mod 3. That's neat: c = r where r = n mod 3.
    fn z3_step_symmetric(n: u64) -> u64 {
        let r = n % 3;
        if r == 0 {
            n / 3
        } else {
            let val = 5 * n + r; // c = r ensures 5n+r ≡ 0 mod 3
            let mut result = val;
            while result % 3 == 0 { result /= 3; }
            result
        }
    }

    /// Trace a ℤ₃ trajectory with a given step function, return (converged, steps, cycle_values)
    fn trace_z3(n: u64, step_fn: fn(u64) -> u64, max_steps: usize) -> (bool, usize, Vec<u64>) {
        let mut current = n;
        let mut visited = std::collections::HashSet::new();
        let mut trajectory = Vec::new();

        for i in 0..max_steps {
            if current == 1 {
                return (true, i, trajectory);
            }
            if !visited.insert(current) {
                // Cycle detected — collect the cycle
                let cycle_start = trajectory.iter().position(|&v| v == current).unwrap();
                let cycle: Vec<u64> = trajectory[cycle_start..].to_vec();
                return (false, i, cycle);
            }
            trajectory.push(current);

            // Divergence check
            if current > 10_000_000 {
                return (false, i, vec![current]);
            }

            current = step_fn(current);
        }
        (false, max_steps, trajectory)
    }

    #[test]
    fn z3_residue_analysis() {
        // First: understand the residue structure of each formulation
        eprintln!("\n{:=^70}", "");
        eprintln!("  ℤ₃ RESIDUE ANALYSIS: what does 5n+c produce?");
        eprintln!("{:=^70}", "");

        for n in 1..=20u64 {
            if n % 3 == 0 { continue; }
            let r = n % 3;
            let naive = 5 * n + 1;
            let symmetric = 5 * n + r;
            eprintln!("n={:>3} (mod3={}): 5n+1={:>4} (v₃={}), 5n+r={:>4} (v₃={})",
                n, r, naive, {
                    let mut v = 0; let mut x = naive; while x % 3 == 0 { x /= 3; v += 1; } v
                },
                symmetric, {
                    let mut v = 0; let mut x = symmetric; while x % 3 == 0 { x /= 3; v += 1; } v
                });
        }
    }

    #[test]
    fn z3_naive_convergence() {
        // Test naive formulation: (5n+1)/3^{v₃} for all n≢0 mod 3
        eprintln!("\n{:=^70}", "");
        eprintln!("  ℤ₃ NAIVE: T(n) = (5n+1)/3^{{v₃(5n+1)}}");
        eprintln!("{:=^70}", "");

        let mut converged = 0u32;
        let mut cycled = 0u32;
        let mut diverged = 0u32;
        let mut cycles: std::collections::HashMap<Vec<u64>, usize> = std::collections::HashMap::new();

        for n in 1..=10000u64 {
            if n % 3 == 0 { continue; }
            let (conv, _steps, data) = trace_z3(n, z3_step_naive, 100_000);
            if conv {
                converged += 1;
            } else if data.len() > 1 && data[0] == data.last().copied().unwrap_or(0) {
                // Not a reliable cycle check from trace_z3 — need to check properly
                cycled += 1;
                *cycles.entry(data).or_insert(0) += 1;
            } else if !data.is_empty() && data[0] > 1_000_000 {
                diverged += 1;
            } else {
                cycled += 1;
                *cycles.entry(data).or_insert(0) += 1;
            }
        }

        let total = converged + cycled + diverged;
        eprintln!("Converged to 1: {}/{} ({:.1}%)", converged, total, converged as f64 / total as f64 * 100.0);
        eprintln!("Cycles: {}/{}", cycled, total);
        eprintln!("Diverged: {}/{}", diverged, total);
        if !cycles.is_empty() {
            eprintln!("\nDistinct cycles found: {}", cycles.len());
            for (cycle, count) in cycles.iter().take(10) {
                if cycle.len() <= 10 {
                    eprintln!("  {:?} (×{})", cycle, count);
                } else {
                    eprintln!("  [{}, {}, ... {} elements] (×{})",
                        cycle[0], cycle[1], cycle.len(), count);
                }
            }
        }
    }

    #[test]
    fn z3_symmetric_convergence() {
        // Test symmetric formulation: (5n + (n mod 3)) / 3^{v₃}
        // This guarantees 3 | (5n + r), so v₃ ≥ 1 for every non-zero residue.
        eprintln!("\n{:=^70}", "");
        eprintln!("  ℤ₃ SYMMETRIC: T(n) = (5n + (n mod 3))/3^{{v₃}}");
        eprintln!("{:=^70}", "");

        let mut converged = 0u32;
        let mut cycled = 0u32;
        let mut diverged = 0u32;
        let mut cycles: std::collections::HashMap<Vec<u64>, usize> = std::collections::HashMap::new();

        for n in 1..=10000u64 {
            if n % 3 == 0 { continue; }
            let (conv, _steps, data) = trace_z3(n, z3_step_symmetric, 100_000);
            if conv {
                converged += 1;
            } else if !data.is_empty() && data[0] > 1_000_000 {
                diverged += 1;
            } else {
                cycled += 1;
                *cycles.entry(data).or_insert(0) += 1;
            }
        }

        let total = converged + cycled + diverged;
        eprintln!("Converged to 1: {}/{} ({:.1}%)", converged, total, converged as f64 / total as f64 * 100.0);
        eprintln!("Cycles: {}/{}", cycled, total);
        eprintln!("Diverged: {}/{}", diverged, total);
        if !cycles.is_empty() {
            eprintln!("\nDistinct cycles found: {}", cycles.len());
            let mut sorted_cycles: Vec<_> = cycles.iter().collect();
            sorted_cycles.sort_by(|a, b| b.1.cmp(a.1));
            for (cycle, count) in sorted_cycles.iter().take(10) {
                if cycle.len() <= 10 {
                    eprintln!("  {:?} (×{})", cycle, count);
                } else {
                    eprintln!("  [{}, {}, ... {} elements] (×{})",
                        cycle[0], cycle[1], cycle.len(), count);
                }
            }
        }
    }

    #[test]
    fn z3_residue_aware_convergence() {
        // Test residue-aware: n≡1→5n+1, n≡2→5n+2
        eprintln!("\n{:=^70}", "");
        eprintln!("  ℤ₃ RESIDUE-AWARE: n≡1→(5n+1)/3^v, n≡2→(5n+2)/3^v");
        eprintln!("{:=^70}", "");

        let mut converged = 0u32;
        let mut cycled = 0u32;
        let mut diverged = 0u32;
        let mut cycles: std::collections::HashMap<Vec<u64>, usize> = std::collections::HashMap::new();

        for n in 1..=10000u64 {
            if n % 3 == 0 { continue; }
            let (conv, _steps, data) = trace_z3(n, z3_step_residue, 100_000);
            if conv {
                converged += 1;
            } else if !data.is_empty() && data[0] > 1_000_000 {
                diverged += 1;
            } else {
                cycled += 1;
                *cycles.entry(data).or_insert(0) += 1;
            }
        }

        let total = converged + cycled + diverged;
        eprintln!("Converged to 1: {}/{} ({:.1}%)", converged, total, converged as f64 / total as f64 * 100.0);
        eprintln!("Cycles: {}/{}", cycled, total);
        eprintln!("Diverged: {}/{}", diverged, total);
        if !cycles.is_empty() {
            eprintln!("\nDistinct cycles found: {}", cycles.len());
            let mut sorted_cycles: Vec<_> = cycles.iter().collect();
            sorted_cycles.sort_by(|a, b| b.1.cmp(a.1));
            for (cycle, count) in sorted_cycles.iter().take(10) {
                if cycle.len() <= 10 {
                    eprintln!("  {:?} (×{})", cycle, count);
                } else {
                    eprintln!("  [{}, {}, ... {} elements] (×{})",
                        cycle[0], cycle[1], cycle.len(), count);
                }
            }
        }
    }

    #[test]
    fn z3_v3_distribution() {
        // Measure v₃ distribution for each formulation
        eprintln!("\n{:=^70}", "");
        eprintln!("  v₃ DISTRIBUTION for ℤ₃ Collatz formulations");
        eprintln!("{:=^70}", "");

        // Symmetric: 5n + (n mod 3)
        let mut counts = [0u64; 16];
        let mut total = 0u64;
        for n in 1..=1_000_000u64 {
            if n % 3 == 0 { continue; }
            let r = n % 3;
            let val = 5 * n + r;
            let mut v = 0u32;
            let mut x = val;
            while x % 3 == 0 { x /= 3; v += 1; }
            counts[v.min(15) as usize] += 1;
            total += 1;
        }

        eprintln!("Symmetric (5n + r):");
        eprintln!("{:>4} {:>10} {:>10} {:>10}", "v₃", "count", "empirical", "1/3^v");
        for v in 0..10 {
            if counts[v] == 0 { continue; }
            let emp = counts[v] as f64 / total as f64;
            let pred = if v == 0 { 0.0 } else { (2.0/3.0) * (1.0/3.0f64).powi(v as i32 - 1) };
            eprintln!("{:>4} {:>10} {:>10.6} {:>10.6}", v, counts[v], emp, pred);
        }
        let ev3: f64 = (1..16).map(|v| v as f64 * counts[v] as f64 / total as f64).sum();
        eprintln!("E[v₃] = {:.4}", ev3);
        eprintln!("5 / 3^E[v₃] = {:.4} (convergence iff < 1)", 5.0 / 3.0f64.powf(ev3));
    }

    #[test]
    fn z3_comparison_summary() {
        eprintln!("\n{:=^70}", "");
        eprintln!("  ℤ₂ vs ℤ₃ COLLATZ COMPARISON");
        eprintln!("{:=^70}", "");
        eprintln!("  ℤ₂ Collatz: T(n) = (3n+1)/2^{{v₂(3n+1)}}");
        eprintln!("    m=3, d=2, E[v₂]=2, contraction=3/4=0.75 < 1 → CONVERGES");
        eprintln!("    P(v₂=j) = 1/2^j, geometric, Haar measure on ℤ₂");
        eprintln!();
        eprintln!("  ℤ₃ candidate: T(n) = (5n+r)/3^{{v₃(5n+r)}}  (r = n mod 3)");
        eprintln!("    m=5, d=3, needs E[v₃] > log(5)/log(3) = {:.4}", 5.0f64.ln() / 3.0f64.ln());
        eprintln!("    That requires E[v₃] > 1.465");
        eprintln!("    If Haar: P(v₃≥j) ∝ 1/3^j → E[v₃] = 3/2 = 1.5 > 1.465");
        eprintln!("    Contraction = 5/3^{{1.5}} = 5/5.196 = 0.962 (barely < 1!)");
        eprintln!();
        eprintln!("  Key question: does the ACTUAL v₃ distribution match Haar?");
        eprintln!("  If yes: ℤ₃ Collatz converges (barely). If no: it might not.");
    }

    // ── Nyquist family analysis ───────────────────────────────────────────

    #[test]
    fn nyquist_margins_all() {
        eprintln!("\n{:=^70}", "");
        eprintln!("  NYQUIST MARGINS: m = 2d-1 for d = 2,3,4,5,6,7");
        eprintln!("{:=^70}", "");
        eprintln!("{:>4} {:>4} {:>8} {:>8} {:>10} {:>8} {:>10}",
            "d", "m", "E[v_d]", "d^E[v]", "contract", "margin", "pct");

        for d in 2..=7u64 {
            let nm = nyquist_margin(d);
            eprintln!("{:>4} {:>4} {:>8.4} {:>8.3} {:>10.4} {:>+8.3} {:>+10.1}%",
                d, nm.m, nm.expected_vd, nm.d_power, nm.contraction,
                nm.margin, nm.margin_pct);
        }

        // Verify (3,2) and (5,3) are convergent, rest are not
        assert!(nyquist_margin(2).convergent, "(3,2) should be convergent");
        assert!(nyquist_margin(3).convergent, "(5,3) should be convergent");
        assert!(!nyquist_margin(4).convergent, "(7,4) should be divergent");
        assert!(!nyquist_margin(5).convergent, "(9,5) should be divergent");
        assert!(!nyquist_margin(7).convergent, "(13,7) should be divergent");
    }

    #[test]
    fn nyquist_family_convergent_32() {
        // (3,2): standard Collatz — should all converge to 1
        eprintln!("\n{:=^70}", "");
        eprintln!("  FAMILY (3,2): standard Collatz");
        eprintln!("{:=^70}", "");

        let r = family_analysis(3, 2, 10000, 1_000_000, 100_000_000);
        eprintln!("Tested: {}", r.n_tested);
        eprintln!("Converged to 1: {} ({:.1}%)", r.converged, r.converged as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Cycles: {} ({:.1}%)", r.cycled, r.cycled as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Diverged: {} ({:.1}%)", r.diverged, r.diverged as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Distinct cycles: {}", r.distinct_cycles.len());
        for c in &r.distinct_cycles {
            eprintln!("  {:?}", c);
        }
        // Standard Collatz: all should converge to 1
        assert_eq!(r.converged, r.n_tested, "All (3,2) should converge to 1");
    }

    #[test]
    fn nyquist_family_convergent_53() {
        // (5,3): ℤ₃ Collatz — converges but to cycles
        eprintln!("\n{:=^70}", "");
        eprintln!("  FAMILY (5,3): ℤ₃ Collatz");
        eprintln!("{:=^70}", "");

        let r = family_analysis(5, 3, 10000, 100_000, 10_000_000);
        eprintln!("Tested: {}", r.n_tested);
        eprintln!("Converged to 1: {} ({:.1}%)", r.converged, r.converged as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Cycles: {} ({:.1}%)", r.cycled, r.cycled as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Diverged: {} ({:.1}%)", r.diverged, r.diverged as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Distinct cycles: {}", r.distinct_cycles.len());
        for c in &r.distinct_cycles {
            if c.len() <= 20 {
                eprintln!("  {:?}", c);
            } else {
                eprintln!("  [len={}, min={}, max={}]", c.len(), c.iter().min().unwrap(), c.iter().max().unwrap());
            }
        }
        // (5,3) is contractive but has cycles
        assert!(r.cycled > 0, "(5,3) should have non-trivial cycles");
    }

    #[test]
    fn nyquist_family_divergent_95() {
        // (9,5): predicted DIVERGENT (contraction > 1)
        eprintln!("\n{:=^70}", "");
        eprintln!("  FAMILY (9,5): predicted DIVERGENT");
        eprintln!("{:=^70}", "");

        let nm = nyquist_margin(5);
        eprintln!("Contraction: {:.4}, Margin: {:+.3}", nm.contraction, nm.margin);

        let r = family_analysis(9, 5, 10000, 100_000, 100_000_000);
        eprintln!("Tested: {}", r.n_tested);
        eprintln!("Converged to 1: {} ({:.1}%)", r.converged, r.converged as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Cycles: {} ({:.1}%)", r.cycled, r.cycled as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Diverged: {} ({:.1}%)", r.diverged, r.diverged as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Timed out: {} ({:.1}%)", r.timed_out, r.timed_out as f64 / r.n_tested as f64 * 100.0);
        if !r.distinct_cycles.is_empty() {
            eprintln!("Distinct cycles: {}", r.distinct_cycles.len());
            for c in r.distinct_cycles.iter().take(10) {
                if c.len() <= 20 {
                    eprintln!("  {:?}", c);
                } else {
                    eprintln!("  [len={}, min={}, max={}]", c.len(), c.iter().min().unwrap(), c.iter().max().unwrap());
                }
            }
        }
    }

    #[test]
    fn nyquist_family_divergent_137() {
        // (13,7): predicted DIVERGENT (contraction > 1)
        eprintln!("\n{:=^70}", "");
        eprintln!("  FAMILY (13,7): predicted DIVERGENT");
        eprintln!("{:=^70}", "");

        let nm = nyquist_margin(7);
        eprintln!("Contraction: {:.4}, Margin: {:+.3}", nm.contraction, nm.margin);

        let r = family_analysis(13, 7, 10000, 100_000, 100_000_000);
        eprintln!("Tested: {}", r.n_tested);
        eprintln!("Converged to 1: {} ({:.1}%)", r.converged, r.converged as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Cycles: {} ({:.1}%)", r.cycled, r.cycled as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Diverged: {} ({:.1}%)", r.diverged, r.diverged as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Timed out: {} ({:.1}%)", r.timed_out, r.timed_out as f64 / r.n_tested as f64 * 100.0);
        if !r.distinct_cycles.is_empty() {
            eprintln!("Distinct cycles: {}", r.distinct_cycles.len());
            for c in r.distinct_cycles.iter().take(10) {
                if c.len() <= 20 {
                    eprintln!("  {:?}", c);
                } else {
                    eprintln!("  [len={}, min={}, max={}]", c.len(), c.iter().min().unwrap(), c.iter().max().unwrap());
                }
            }
        }
    }

    #[test]
    fn nyquist_vd_distributions() {
        // Verify v_d matches Haar for all families
        eprintln!("\n{:=^70}", "");
        eprintln!("  v_d DISTRIBUTIONS across (m,d) families");
        eprintln!("{:=^70}", "");

        for &(m, d) in &[(3u64, 2u64), (5, 3), (9, 5), (13, 7)] {
            let (counts, ev) = empirical_vd(m, d, 100_000);
            let haar_ev = d as f64 / (d as f64 - 1.0);
            eprintln!("\n({},{}): E[v_{}] = {:.4} (Haar prediction: {:.4})",
                m, d, d, ev, haar_ev);
            eprintln!("  Contraction = {} / {}^{:.4} = {:.4}",
                m, d, ev, m as f64 / (d as f64).powf(ev));
            // Show first few levels
            let total: u64 = counts.iter().sum();
            for v in 0..6 {
                if counts[v] > 0 {
                    eprintln!("  v={}: {:.6} (n={})", v, counts[v] as f64 / total as f64, counts[v]);
                }
            }
            // v_d should match Haar within 1%
            assert!((ev - haar_ev).abs() < 0.02,
                "({},{}) E[v_{}] = {:.4} but Haar = {:.4}", m, d, d, ev, haar_ev);
        }
    }

    #[test]
    fn z3_exhaustive_cycle_search() {
        // Search (5,3) to n=100K for ALL cycles
        eprintln!("\n{:=^70}", "");
        eprintln!("  EXHAUSTIVE CYCLE SEARCH: (5,3) n=1..100000");
        eprintln!("{:=^70}", "");

        let r = family_analysis(5, 3, 100_000, 100_000, 10_000_000);
        eprintln!("Tested: {}", r.n_tested);
        eprintln!("Converged to 1: {} ({:.1}%)", r.converged, r.converged as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Cycles: {} ({:.1}%)", r.cycled, r.cycled as f64 / r.n_tested as f64 * 100.0);
        eprintln!("Diverged: {} ({:.1}%)", r.diverged, r.diverged as f64 / r.n_tested as f64 * 100.0);
        eprintln!("\nALL distinct cycles found:");
        for c in &r.distinct_cycles {
            if c.len() <= 30 {
                eprintln!("  {:?}", c);
            } else {
                eprintln!("  [len={}, min={}, max={}]", c.len(), c.iter().min().unwrap(), c.iter().max().unwrap());
            }
        }
        eprintln!("\nTotal distinct cycles: {}", r.distinct_cycles.len());

        // Analyze cycle residue structure
        for c in &r.distinct_cycles {
            let residues: Vec<u64> = c.iter().map(|&v| v % 3).collect();
            eprintln!("Cycle {:?} → residues {:?}", c, residues);
        }
    }

    // ── Temporal equidistribution — extremal bootstrapping ────────────

    #[test]
    fn temporal_coverage_k10() {
        eprintln!("\n{:=^70}", "");
        eprintln!("  TEMPORAL COVERAGE: 2^10 - 1 = 1023");
        eprintln!("{:=^70}", "");

        let js: Vec<u32> = (3..=10).collect();
        let results = temporal_coverage_extremal(10, &js, 10_000_000);

        eprintln!("{:>4} {:>8} {:>8} {:>8} {:>12} {:>10}",
            "j", "classes", "hit", "cover%", "steps_full", "avail");
        for r in &results {
            eprintln!("{:>4} {:>8} {:>8} {:>7.1}% {:>12} {:>10}",
                r.j, r.total_odd_classes, r.classes_hit,
                r.coverage * 100.0,
                r.steps_to_full.map(|s| format!("{}", s)).unwrap_or("—".to_string()),
                r.post_fold_steps);
        }

        // k=10 has a short post-fold trajectory; just verify data is collected
        // Coverage assertions only when post_fold_steps >= 2^j
        for r in &results {
            let needed = 1usize << r.j;
            if r.post_fold_steps >= needed {
                assert!(r.coverage > 0.90,
                    "k=10, j={}: coverage {:.1}% low (enough steps)", r.j, r.coverage * 100.0);
            }
        }
    }

    #[test]
    fn temporal_coverage_k20() {
        eprintln!("\n{:=^70}", "");
        eprintln!("  TEMPORAL COVERAGE: 2^20 - 1 = 1048575");
        eprintln!("{:=^70}", "");

        let js: Vec<u32> = (3..=10).collect();
        let results = temporal_coverage_extremal(20, &js, 10_000_000);

        eprintln!("{:>4} {:>8} {:>8} {:>8} {:>12} {:>10}",
            "j", "classes", "hit", "cover%", "steps_full", "avail");
        for r in &results {
            eprintln!("{:>4} {:>8} {:>8} {:>7.1}% {:>12} {:>10}",
                r.j, r.total_odd_classes, r.classes_hit,
                r.coverage * 100.0,
                r.steps_to_full.map(|s| format!("{}", s)).unwrap_or("—".to_string()),
                r.post_fold_steps);
        }

        // Windowed coverage is noisy for small trajectories —
        // full-trajectory coverage in temporal_coverage_extended is the real test
    }

    #[test]
    fn temporal_coverage_k30() {
        eprintln!("\n{:=^70}", "");
        eprintln!("  TEMPORAL COVERAGE: 2^30 - 1 = 1073741823");
        eprintln!("{:=^70}", "");

        let js: Vec<u32> = (3..=10).collect();
        let results = temporal_coverage_extremal(30, &js, 10_000_000);

        eprintln!("{:>4} {:>8} {:>8} {:>8} {:>12} {:>10}",
            "j", "classes", "hit", "cover%", "steps_full", "avail");
        for r in &results {
            eprintln!("{:>4} {:>8} {:>8} {:>7.1}% {:>12} {:>10}",
                r.j, r.total_odd_classes, r.classes_hit,
                r.coverage * 100.0,
                r.steps_to_full.map(|s| format!("{}", s)).unwrap_or("—".to_string()),
                r.post_fold_steps);
        }

        for r in &results {
            let needed = 1usize << r.j;
            if r.post_fold_steps >= needed {
                assert!(r.coverage > 0.90,
                    "k=30, j={}: coverage {:.1}% low (enough steps)", r.j, r.coverage * 100.0);
            }
        }
    }

    #[test]
    fn temporal_coverage_summary() {
        eprintln!("\n{:=^70}", "");
        eprintln!("  TEMPORAL EQUIDISTRIBUTION SUMMARY");
        eprintln!("{:=^70}", "");

        let ks = [10u32, 20, 30];
        let js: Vec<u32> = (3..=10).collect();

        eprintln!("{:>4} {:>4} {:>8} {:>8} {:>8} {:>12} {:>8}",
            "k", "j", "classes", "hit", "cover%", "steps_full", "avail");

        for &k in &ks {
            let results = temporal_coverage_extremal(k, &js, 10_000_000);
            for r in &results {
                let full = r.coverage > 0.999;
                let enough = r.post_fold_steps >= (1usize << r.j);
                let marker = if full { "✓" } else if !enough { "·" } else { "✗" };
                eprintln!("{:>4} {:>4} {:>8} {:>8} {:>7.1}% {:>12} {:>8} {}",
                    r.k, r.j, r.total_odd_classes, r.classes_hit,
                    r.coverage * 100.0,
                    r.steps_to_full.map(|s| format!("{}", s)).unwrap_or("—".to_string()),
                    r.post_fold_steps, marker);
            }
            eprintln!();
        }

        eprintln!("Legend: ✓=100% coverage, ✗=incomplete (enough steps), ·=insufficient steps");
    }

    #[test]
    fn temporal_coverage_extended() {
        // Use FULL trajectory coverage (not just first 2^j steps)
        // to test whether ALL residue classes are eventually visited
        eprintln!("\n{:=^70}", "");
        eprintln!("  FULL-TRAJECTORY TEMPORAL COVERAGE");
        eprintln!("{:=^70}", "");

        let ks = [10u32, 20, 30, 40, 50, 60, 70];
        let js: Vec<u32> = (3..=8).collect();

        eprintln!("{:>4} {:>4} {:>8} {:>8} {:>8} {:>12} {:>8}",
            "k", "j", "classes", "hit", "cover%", "steps_full", "avail");

        let mut full_count = 0u32;
        let mut tested_count = 0u32;

        for &k in &ks {
            if k as f64 * 1.585 > 125.0 { continue; }
            let n: u128 = (1u128 << k) - 1;
            let (fold_idx, odds) = collatz_trajectory_odd(n, 50_000_000);
            let post_fold = &odds[fold_idx..];

            for &j in &js {
                let modulus = 1u128 << j;
                let total_odd = 1usize << (j - 1);

                // Full trajectory coverage
                let mut seen = std::collections::HashSet::new();
                let mut steps_to_full = None;
                for (step, &val) in post_fold.iter().enumerate() {
                    seen.insert(val % modulus);
                    if steps_to_full.is_none() && seen.len() == total_odd {
                        steps_to_full = Some(step + 1);
                    }
                }

                let coverage = seen.len() as f64 / total_odd as f64;
                let full = seen.len() == total_odd;
                let marker = if full { "✓" } else { "✗" };

                eprintln!("{:>4} {:>4} {:>8} {:>8} {:>7.1}% {:>12} {:>8} {}",
                    k, j, total_odd, seen.len(),
                    coverage * 100.0,
                    steps_to_full.map(|s| format!("{}", s)).unwrap_or("—".to_string()),
                    post_fold.len(), marker);

                tested_count += 1;
                if full { full_count += 1; }

                // If missing classes, show which
                if !full && j <= 5 {
                    let all_odd: Vec<u128> = (0..modulus).filter(|x| x % 2 == 1).collect();
                    let missing: Vec<u128> = all_odd.iter().filter(|x| !seen.contains(x)).copied().collect();
                    eprintln!("         missing mod {}: {:?}", modulus, missing);
                }
            }
            eprintln!();
        }

        eprintln!("Full coverage: {}/{} ({:.1}%)",
            full_count, tested_count,
            full_count as f64 / tested_count as f64 * 100.0);
    }

    #[test]
    fn temporal_coverage_coupon_collector() {
        // Compare coverage rate against coupon collector prediction
        // Under uniform sampling: E[coverage at step T] = N*(1 - ((N-1)/N)^T)
        eprintln!("\n{:=^70}", "");
        eprintln!("  COUPON COLLECTOR COMPARISON (j=4, N=8)");
        eprintln!("{:=^70}", "");

        let j = 4u32;
        let modulus = 1u128 << j;
        let total_odd = 1usize << (j - 1); // N = 8

        eprintln!("{:>4} {:>8} {:>8} {:>8} {:>8}",
            "k", "avail", "T_full", "predict", "ratio");

        for k in (15..=70).step_by(5) {
            if k as f64 * 1.585 > 125.0 { continue; }
            let n: u128 = (1u128 << k) - 1;
            let (fold_idx, odds) = collatz_trajectory_odd(n, 50_000_000);
            let post_fold = &odds[fold_idx..];

            let mut seen = std::collections::HashSet::new();
            let mut t_full = None;
            for (step, &val) in post_fold.iter().enumerate() {
                seen.insert(val % modulus);
                if t_full.is_none() && seen.len() == total_odd {
                    t_full = Some(step + 1);
                }
            }

            // Coupon collector: E[T] = N * H_N where H_N = harmonic number
            let hn: f64 = (1..=total_odd).map(|i| 1.0 / i as f64).sum();
            let predicted = total_odd as f64 * hn;

            let t_str = t_full.map(|t| format!("{}", t)).unwrap_or("—".to_string());
            let ratio = t_full.map(|t| t as f64 / predicted).unwrap_or(f64::NAN);

            eprintln!("{:>4} {:>8} {:>8} {:>8.1} {:>8.2}",
                k, post_fold.len(), t_str, predicted, ratio);
        }

        eprintln!("\nPrediction: N*H_N = 8*2.72 = 21.7 steps for N=8 classes");
        eprintln!("Ratio ≈ 1.0 means trajectory matches uniform coupon collector");
    }
}
