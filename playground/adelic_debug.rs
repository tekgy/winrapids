//! Debug: what does the free energy landscape ACTUALLY look like
//! for various N? Don't force solutions — observe what's there.

fn free_energy(p: f64, s: f64) -> f64 {
    let val = 1.0 - p.powf(-s);
    if val <= 0.0 { return f64::INFINITY; }
    -val.ln()
}

fn main() {
    let primes: Vec<f64> = vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0];

    eprintln!("=== Free energy landscape: Σ F(pᵢ,s) vs target ===\n");

    for n in [2, 3, 4, 5, 7, 10] {
        let p = &primes[..n];
        let p1 = p[0];
        let pn = *p.last().unwrap();
        let target = (1.0 / n as f64) * (pn / p1).ln();

        eprintln!("N={}, primes={:?}, target={:.6}", n, p.iter().map(|x| *x as u64).collect::<Vec<_>>(), target);
        eprintln!("  {:>8} {:>14} {:>14} {:>10}", "s", "Σ F(pᵢ,s)", "target", "side");

        for &s in &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0] {
            let f_sum: f64 = p.iter().map(|&pi| free_energy(pi, s)).sum();
            let side = if !f_sum.is_finite() {
                "overflow"
            } else if (f_sum - target).abs() < 0.01 {
                "≈ MATCH"
            } else if f_sum > target {
                "above"
            } else {
                "below"
            };
            eprintln!("  {:>8.2} {:>14.6} {:>14.6} {:>10}", s, f_sum, target, side);
        }
        eprintln!();
    }

    // Now try the PAIRWISE sum target instead
    eprintln!("=== Alternative target: ½·Σ ln(pᵢ₊₁/pᵢ) (sum of pairwise) ===\n");
    for n in [2, 3, 4, 5, 7, 10] {
        let p = &primes[..n];
        let target_pairwise: f64 = (0..n-1).map(|i| 0.5 * (p[i+1] / p[i]).ln()).sum();
        let target_endpoint = (1.0 / n as f64) * (p.last().unwrap() / p[0]).ln();

        eprintln!("N={}: pairwise_target={:.6}, endpoint_target={:.6}, ratio={:.3}",
            n, target_pairwise, target_endpoint, target_pairwise / target_endpoint);
    }

    // Binary search that ACTUALLY works
    eprintln!("\n=== Robust solver: find s where Σ F = target ===\n");
    for n in [2, 3, 4, 5, 7, 10] {
        let p = &primes[..n];
        let target = (1.0 / n as f64) * (p.last().unwrap() / p[0]).ln();

        // Sweep s from 0.1 to 30 in fine steps, find where we cross target
        let mut found = None;
        let mut prev_f = f64::INFINITY;
        for i in 1..3000 {
            let s = i as f64 * 0.01;
            let f: f64 = p.iter().map(|&pi| free_energy(pi, s)).sum();
            if !f.is_finite() { prev_f = f; continue; }
            if prev_f > target && f <= target {
                // Crossed! Refine
                let mut lo = (i - 1) as f64 * 0.01;
                let mut hi = s;
                for _ in 0..100 {
                    let mid = (lo + hi) / 2.0;
                    let fm: f64 = p.iter().map(|&pi| free_energy(pi, mid)).sum();
                    if fm > target { lo = mid; } else { hi = mid; }
                }
                found = Some((lo + hi) / 2.0);
                break;
            }
            prev_f = f;
        }

        if let Some(s) = found {
            let f: f64 = p.iter().map(|&pi| free_energy(pi, s)).sum();
            let partition: Vec<f64> = p.iter().map(|&pi| {
                let e = free_energy(pi, s);
                e / f * 100.0
            }).collect();
            eprintln!("  N={:>2}: s*={:.6}, Σ F={:.6}, target={:.6}, p2={:.1}%",
                n, s, f, target, partition[0]);
        } else {
            eprintln!("  N={:>2}: NO SOLUTION FOUND (target={:.6})", n, target);
        }
    }
}
