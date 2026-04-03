//! # Collatz 2-adic Analysis
//!
//! The all-ones class (2^k - 1) is the integer closest to -1/3
//! in the 2-adic metric. The cascade decay rate measures drift
//! from this 2-adic singularity.
//!
//! Key experiment: compute v₂(3·(2^k-1)+1) — the 2-adic valuation
//! of the first Collatz step on the all-ones class. This determines
//! how many halvings occur, hence the contraction in the first step.
//!
//! If v₂ grows with k, the all-ones class eventually contracts fast.
//! If v₂ is bounded, it stays hard forever.

use std::time::Instant;

/// 2-adic valuation: largest power of 2 dividing n
fn v2(n: u128) -> u32 {
    if n == 0 { return 128; } // convention
    n.trailing_zeros()
}

/// For the all-ones class r = 2^k - 1:
/// First step: 3r + 1 = 3(2^k - 1) + 1 = 3·2^k - 2 = 2(3·2^{k-1} - 1)
/// So v₂(3r+1) = v₂(2(3·2^{k-1} - 1)) = 1 + v₂(3·2^{k-1} - 1)
///
/// For k ≥ 1: 3·2^{k-1} - 1 is always odd (since 3·2^{k-1} is even for k≥2,
/// so 3·2^{k-1} - 1 is odd). Wait: 3·2^0 - 1 = 2 (k=1), 3·2^1 - 1 = 5 (k=2),
/// 3·2^2 - 1 = 11 (k=3), ...
///
/// Actually let me just compute directly.

fn main() {
    eprintln!("==========================================================");
    eprintln!("  Collatz 2-adic Analysis");
    eprintln!("  All-ones class = closest to -1/3 in ℤ₂");
    eprintln!("==========================================================\n");

    // Part 1: v₂(3·(2^k - 1) + 1) for k = 1..64
    eprintln!("--- v₂(3r+1) for r = 2^k - 1 (all-ones class) ---");
    eprintln!("  {:>4}  {:>20}  {:>6}  {:>10}  {:>10}", "k", "r = 2^k-1", "v₂", "contraction", "net_factor");

    for k in 1..=64u32 {
        let r: u128 = (1u128 << k) - 1;
        let val = 3u128.checked_mul(r).and_then(|x| x.checked_add(1));

        if let Some(v) = val {
            let v2_val = v2(v);
            // After first Collatz step on odd r: result = (3r+1) / 2^v₂
            // Contraction of this step: 3 / 2^v₂ (approximately, for large r)
            let contraction = 3.0f64 / (1u128 << v2_val.min(63)) as f64;
            let net = contraction; // net multiplicative factor

            eprintln!("  {:>4}  {:>20}  {:>6}  {:>10.6}  {:>10.4}",
                k, r, v2_val, contraction, if net < 1.0 { "CONTRACT" } else { "EXPAND" });
        }
    }

    // Part 2: Full trajectory of the all-ones class mod 2^16
    // Track the 2-adic valuation at each step
    eprintln!("\n--- Full trajectory of r=65535 (2^16-1), first 50 steps ---");
    eprintln!("  {:>4}  {:>20}  {:>6}  {:>8}", "step", "n", "v₂(n)", "parity");

    let mut n: u128 = 65535;
    for step in 0..50 {
        let parity = if n % 2 == 1 { "odd" } else { "even" };
        if step < 20 || step % 5 == 0 {
            eprintln!("  {:>4}  {:>20}  {:>6}  {:>8}", step, n, v2(n), parity);
        }
        if n == 1 { break; }
        if n % 2 == 0 { n /= 2; } else { n = 3 * n + 1; }
    }

    // Part 3: 2-adic distance from -1/3 along the trajectory
    // In ℤ₂, -1/3 has 2-adic expansion ...111111
    // d₂(n, -1/3) = 2^{-v₂(3n+1)} for odd n (measures how many
    // low bits of n match the pattern of -1/3)
    //
    // Actually, d₂(n, -1/3) = |n - (-1/3)|₂ = |n + 1/3|₂ = |3n+1|₂ / 3
    // Since |3|₂ = 1 (3 is a 2-adic unit), d₂(n, -1/3) = |3n+1|₂ = 2^{-v₂(3n+1)}

    eprintln!("\n--- 2-adic distance from -1/3 along trajectory of 2^k-1 ---");
    eprintln!("  {:>4}  {:>4}  {:>12}  {:>12}", "k", "step", "v₂(3n+1)", "d₂(n,-1/3)");

    for &k in &[8u32, 12, 16, 20, 24, 28, 32] {
        if k > 40 { continue; }
        let r: u128 = (1u128 << k) - 1;
        let mut n = r;

        // Track v₂(3n+1) for first few odd values in trajectory
        let mut odd_count = 0;
        for step in 0..200 {
            if n % 2 == 1 {
                let v = v2(3 * n + 1);
                let d2 = (0.5f64).powi(v as i32);
                if odd_count < 5 {
                    eprintln!("  {:>4}  {:>4}  {:>12}  {:>12.2e}", k, step, v, d2);
                }
                odd_count += 1;
                n = 3 * n + 1;
            } else {
                n /= 2;
            }
            if n == 1 { break; }
        }
        eprintln!();
    }

    // Part 4: The pattern of v₂(3·2^k + 1) — without the -1
    // 3·2^k + 1: for k=0: 4 (v₂=2), k=1: 7 (v₂=0), k=2: 13 (v₂=0),
    // k=3: 25 (v₂=0), k=4: 49 (v₂=0)...
    // These are always odd for k ≥ 1 (3·2^k is even, +1 makes odd)
    // So v₂ = 0 for k ≥ 1. Not useful.
    //
    // But 3·(2^k - 1) + 1 = 3·2^k - 2 = 2(3·2^{k-1} - 1)
    // So v₂ = 1 + v₂(3·2^{k-1} - 1)
    // And 3·2^{k-1} - 1 for k ≥ 2:
    //   k=2: 3·2 - 1 = 5 (odd, v₂=0) → v₂(3r+1) = 1
    //   k=3: 3·4 - 1 = 11 (odd, v₂=0) → v₂ = 1
    //   k=4: 3·8 - 1 = 23 (odd, v₂=0) → v₂ = 1
    //
    // So for all k ≥ 2: v₂(3·(2^k-1)+1) = 1.
    // The first step on the all-ones class ALWAYS gives exactly one halving.
    // Contraction = 3/2 > 1. ALWAYS EXPANDING on first step.
    //
    // This is WHY the all-ones class is always hard!
    // After the first step: (3·(2^k-1)+1)/2 = (3·2^k - 2)/2 = 3·2^{k-1} - 1
    // This is 3·2^{k-1} - 1, which has NO trailing zeros (it's odd for k≥2).
    // The SECOND step: 3·(3·2^{k-1}-1)+1 = 9·2^{k-1} - 2 = 2(9·2^{k-2}-1)
    // So v₂ = 1 again! (for k ≥ 3, 9·2^{k-2}-1 is odd)
    //
    // Pattern: the all-ones class generates a sequence of v₂ = 1 halvings,
    // meaning contraction 3/2 at EVERY odd step. This is maximally expanding.

    eprintln!("--- Pattern: v₂(3r+1) for consecutive all-ones classes ---");
    eprintln!("  For r = 2^k - 1, v₂(3r+1) = 1 for all k ≥ 2.");
    eprintln!("  This means: first Collatz step on all-ones ALWAYS gives contraction 3/2.");
    eprintln!("  The all-ones class is maximally expanding because EVERY ODD STEP");
    eprintln!("  produces exactly ONE halving — the minimum possible.");
    eprintln!();

    // Part 5: What about the SECOND odd step?
    // After two Collatz steps on 2^k-1:
    // Step 1 (odd): (3·(2^k-1)+1)/2 = 3·2^{k-1} - 1 (odd for k≥2)
    // Step 2 (odd): (3·(3·2^{k-1}-1)+1)/2 = (9·2^{k-1}-2)/2 = 9·2^{k-2} - 1
    // For k ≥ 3: 9·2^{k-2} - 1 is odd (9·even - 1 = odd)
    // Step 3 (odd): (3·(9·2^{k-2}-1)+1)/2 = (27·2^{k-2}-2)/2 = 27·2^{k-3} - 1
    // For k ≥ 4: 27·2^{k-3} - 1 is odd
    //
    // Pattern: after j odd steps (no even steps in between!):
    // n_j = 3^j · 2^{k-j} - 1 (for j ≤ k)
    //
    // When does this stop being odd?
    // n_j is odd iff 3^j · 2^{k-j} is even iff k > j.
    // So for the first k steps, EVERY step is odd!
    //
    // But wait: 3^j · 2^{k-j} - 1: if k-j ≥ 1, then 3^j · 2^{k-j} is even,
    // so 3^j · 2^{k-j} - 1 is odd. And the Collatz step gives:
    // (3·n_j + 1) / 2 = (3·(3^j·2^{k-j}-1)+1)/2 = (3^{j+1}·2^{k-j}-2)/2
    //                  = 3^{j+1}·2^{k-j-1} - 1 = n_{j+1}
    //
    // So n_j = 3^j · 2^{k-j} - 1 for j = 0, 1, ..., k-1.
    // After k-1 odd steps: n_{k-1} = 3^{k-1} · 2 - 1 = 2·3^{k-1} - 1 (odd)
    // Step k: (3·(2·3^{k-1}-1)+1)/2 = (6·3^{k-1}-2)/2 = 3^k - 1
    // Now 3^k - 1 is EVEN (3^k is odd, -1 makes even).
    // v₂(3^k - 1) = ? This is where it gets interesting.

    eprintln!("--- The all-ones trajectory formula ---");
    eprintln!("  Starting from r = 2^k - 1:");
    eprintln!("  After j Collatz steps (all odd): n_j = 3^j · 2^(k-j) - 1");
    eprintln!("  After k steps: n_k = 3^k - 1 (EVEN!)");
    eprintln!("  Then v₂(3^k - 1) halvings occur.");
    eprintln!();
    eprintln!("  v₂(3^k - 1) for k = 1..50:");

    let mut vals: Vec<(u32, u32)> = Vec::new();
    for k in 1..=50u32 {
        // 3^k - 1
        // For large k, use u128
        if k <= 80 {
            let pow3: u128 = 3u128.pow(k);
            let val = pow3 - 1;
            let v = v2(val);
            vals.push((k, v));
            eprintln!("    k={:>3}: 3^k-1 = {:>30}, v₂ = {:>3}", k, val, v);
        }
    }

    // Analyze the pattern of v₂(3^k - 1)
    eprintln!("\n  Pattern analysis:");
    eprintln!("  v₂(3^k - 1) = v₂(k) + 1 for k even, = 1 for k odd");
    eprintln!("  (Because 3^k - 1 = (3-1)(3^{{k-1}} + ... + 1) = 2·S");
    eprintln!("   where S = 1 + 3 + 3² + ... + 3^{{k-1}}.)");
    eprintln!("  If k is odd: S has k terms (odd many), S is odd → v₂ = 1");
    eprintln!("  If k is even: S has k terms (even many), S is even → v₂ ≥ 2");

    // Part 6: Net contraction for all-ones class
    // After k odd steps (each ×3/2) and then v₂(3^k-1) halvings:
    // Total factor: 3^k / 2^k · 1/2^{v₂(3^k-1)}
    // = 3^k / 2^{k + v₂(3^k-1)}
    eprintln!("\n--- Net contraction after processing all k bits + v₂ halvings ---");
    for &(k, v) in &vals {
        if k <= 30 {
            let factor = (3.0f64).powi(k as i32) / (2.0f64).powi((k + v) as i32);
            let contracting = factor < 1.0;
            eprintln!("    k={:>3}: 3^k / 2^(k+v₂) = 3^{} / 2^{} = {:.4}  {}",
                k, k, k + v, factor,
                if contracting { "✓ CONTRACT" } else { "✗ EXPAND" });
        }
    }

    // Part 7: The key question — after the all-ones phase completes
    // (k odd steps + v₂ halvings), what's the bit density of the RESULT?
    // If the result has typical bit density (~0.5), the NEXT phase contracts.
    eprintln!("\n--- Bit density of result after all-ones phase ---");
    for k in [8u32, 12, 16, 20, 24] {
        let r: u128 = (1u128 << k) - 1;
        let mut n = r;
        let mut steps = 0;
        // Run k odd steps (we proved they're all odd)
        for _ in 0..k {
            n = (3 * n + 1) / 2;
            steps += 1;
        }
        // Now n = 3^k - 1, which is even. Halve until odd.
        while n % 2 == 0 && n > 0 {
            n /= 2;
            steps += 1;
        }

        let bits = 128 - n.leading_zeros();
        let ones = n.count_ones();
        let density = ones as f64 / bits as f64;
        eprintln!("    k={:>3}: after phase, n={}, bits={}, ones={}, density={:.3}",
            k, n, bits, ones, density);
    }
}
