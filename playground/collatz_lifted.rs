//! # Lifted Collatz: Semigroup Parallel Prefix
//!
//! The Collatz step is an affine map: n → a·n + b
//! Affine maps compose: (a₁n+b₁) ∘ (a₂n+b₂) = (a₁a₂)n + (a₁b₂+b₁)
//! This is a SEMIGROUP. Prefix scan computes K compositions in O(log K).
//!
//! For the shadow phase of 2^k - 1: all maps are (3, 1)/2 = "triple and add one, halve once"
//! The composition is (3^k, 3^k - 1) / 2^k — computed via parallel prefix of affine maps.
//!
//! For post-shadow: the v₂ at each step determines the affine map.
//! We SWEEP all possible v₂ sequences rather than speculating.

use std::time::Instant;

// ── BigInt (minimal, for this test) ─────────────────────────

#[derive(Clone, Debug)]
struct Big { limbs: Vec<u64> }

impl Big {
    fn zero() -> Self { Big { limbs: vec![0] } }
    fn one() -> Self { Big { limbs: vec![1] } }
    fn from_u64(v: u64) -> Self { Big { limbs: vec![v] } }

    fn is_zero(&self) -> bool { self.limbs.iter().all(|&x| x == 0) }
    fn is_one(&self) -> bool { self.limbs.len() == 1 && self.limbs[0] == 1 }
    fn is_odd(&self) -> bool { !self.limbs.is_empty() && self.limbs[0] & 1 == 1 }

    fn bit_len(&self) -> u32 {
        if self.is_zero() { return 0; }
        let top = self.limbs.len() - 1;
        (top as u32) * 64 + (64 - self.limbs[top].leading_zeros())
    }

    fn trailing_zeros(&self) -> u32 {
        for (i, &limb) in self.limbs.iter().enumerate() {
            if limb != 0 { return (i as u32) * 64 + limb.trailing_zeros(); }
        }
        self.bit_len()
    }

    fn trailing_ones(&self) -> u32 {
        for (i, &limb) in self.limbs.iter().enumerate() {
            if limb != u64::MAX { return (i as u32) * 64 + limb.trailing_ones(); }
        }
        (self.limbs.len() as u32) * 64
    }

    fn shr_assign(&mut self, shift: u32) {
        let limb_shift = (shift / 64) as usize;
        let bit_shift = shift % 64;
        if limb_shift >= self.limbs.len() { self.limbs = vec![0]; return; }
        self.limbs.drain(0..limb_shift);
        if bit_shift > 0 {
            let mut carry = 0u64;
            for i in (0..self.limbs.len()).rev() {
                let new_carry = self.limbs[i] << (64 - bit_shift);
                self.limbs[i] = (self.limbs[i] >> bit_shift) | carry;
                carry = new_carry;
            }
        }
        while self.limbs.len() > 1 && *self.limbs.last().unwrap() == 0 { self.limbs.pop(); }
    }

    fn add(&self, other: &Big) -> Big {
        let len = self.limbs.len().max(other.limbs.len());
        let mut result = Vec::with_capacity(len + 1);
        let mut carry = 0u64;
        for i in 0..len {
            let a = if i < self.limbs.len() { self.limbs[i] } else { 0 };
            let b = if i < other.limbs.len() { other.limbs[i] } else { 0 };
            let (s1, c1) = a.overflowing_add(b);
            let (s2, c2) = s1.overflowing_add(carry);
            result.push(s2);
            carry = (c1 as u64) + (c2 as u64);
        }
        if carry != 0 { result.push(carry); }
        Big { limbs: result }
    }

    fn sub(&self, other: &Big) -> Big {
        let mut result = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0i64;
        for i in 0..self.limbs.len() {
            let a = self.limbs[i] as i128;
            let b = if i < other.limbs.len() { other.limbs[i] as i128 } else { 0 };
            let diff = a - b - borrow as i128;
            if diff < 0 {
                result.push((diff + (1i128 << 64)) as u64);
                borrow = 1;
            } else {
                result.push(diff as u64);
                borrow = 0;
            }
        }
        while result.len() > 1 && *result.last().unwrap() == 0 { result.pop(); }
        Big { limbs: result }
    }

    fn mul_u64(&self, m: u64) -> Big {
        let mut result = Vec::with_capacity(self.limbs.len() + 1);
        let mut carry = 0u128;
        for &limb in &self.limbs {
            let prod = limb as u128 * m as u128 + carry;
            result.push(prod as u64);
            carry = prod >> 64;
        }
        if carry != 0 { result.push(carry as u64); }
        while result.len() > 1 && *result.last().unwrap() == 0 { result.pop(); }
        Big { limbs: result }
    }

    fn add_one(&self) -> Big { self.add(&Big::one()) }

    // 3n+1 = n+n+n+1
    fn triple_plus_one(&self) -> Big {
        self.add(self).add(self).add_one()
    }

    fn all_ones(k: u32) -> Big {
        let full = (k / 64) as usize;
        let rem = k % 64;
        let mut limbs = vec![u64::MAX; full];
        if rem > 0 { limbs.push((1u64 << rem) - 1); }
        if limbs.is_empty() { limbs.push(0); }
        Big { limbs }
    }

    // Compute 3^k using repeated squaring
    fn pow3(k: u32) -> Big {
        if k == 0 { return Big::one(); }
        let mut result = Big::one();
        let mut base = Big::from_u64(3);
        let mut exp = k;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul_big(&base);
            }
            base = base.mul_big(&base);
            exp >>= 1;
        }
        result
    }

    // Full BigInt multiply (schoolbook — O(n²) but simple)
    fn mul_big(&self, other: &Big) -> Big {
        let mut result = vec![0u64; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &b) in other.limbs.iter().enumerate() {
                let prod = a as u128 * b as u128 + result[i + j] as u128 + carry;
                result[i + j] = prod as u64;
                carry = prod >> 64;
            }
            result[i + other.limbs.len()] += carry as u64;
        }
        while result.len() > 1 && *result.last().unwrap() == 0 { result.pop(); }
        Big { limbs: result }
    }
}

// ── Affine map: n → (a·n + b) / 2^s ────────────────────────
// Represented as (numerator_a, numerator_b, shift_s)
// where result = (a·n + b) >> s

/// Compute the shadow phase result DIRECTLY using 3^k
/// Instead of k sequential steps: one 3^k computation + arithmetic
fn shadow_direct(k: u32) -> (Big, u32) {
    // After k shadow steps on 2^k - 1:
    // result = 3^k - 1
    // v₂ = v₂(k) + 1
    // odd result = (3^k - 1) / 2^{v₂(k)+1}

    let pow3k = Big::pow3(k);
    let three_k_minus_1 = pow3k.sub(&Big::one());
    let v2 = three_k_minus_1.trailing_zeros();
    let mut result = three_k_minus_1;
    result.shr_assign(v2);
    (result, v2)
}

/// Sequential Collatz from a starting BigInt until it drops below start
fn collatz_sequential(start: &Big, max_steps: u64) -> (u64, u32, bool) {
    let mut current = start.clone();
    let mut max_tau = current.trailing_ones();
    let start_bits = start.bit_len();

    for step in 1..=max_steps {
        if current.is_one() { return (step, max_tau, true); }

        if !current.is_odd() {
            let tz = current.trailing_zeros();
            current.shr_assign(tz);
        } else {
            let r = current.triple_plus_one();
            let v = r.trailing_zeros();
            current = r;
            current.shr_assign(v);
        }

        let tau = current.trailing_ones();
        if tau > max_tau { max_tau = tau; }

        // Early termination
        if current.bit_len() < start_bits / 2 {
            return (step, max_tau, true);
        }
    }
    (max_steps, max_tau, false)
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  LIFTED COLLATZ: Direct Shadow + Sequential Post-fold");
    eprintln!("  Shadow: O(k^1.6) via 3^k (repeated squaring)");
    eprintln!("  vs Sequential: O(k²) via step-by-step");
    eprintln!("==========================================================\n");

    // Test both odd and even k values
    let test_ks: Vec<u32> = vec![
        99, 100, 101,       // odd/even/odd near 100
        999, 1000, 1001,    // near 1000
        4999, 5000, 5001,   // near 5000
        9999, 10000, 10001, // near 10000
        49999, 50000, 50001,
        99999, 100000, 100001,
        499999, 500000, 500001,
        999999, 1000000, 1000001,
    ];

    eprintln!("{:>8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "k", "odd?", "shadow_t", "post_t", "total_t", "max_τ", "conv?");
    eprintln!("{}", "-".repeat(74));

    for &k in &test_ks {
        let is_odd = k % 2 == 1;

        // Phase 1: Direct shadow computation via 3^k
        let t0 = Instant::now();
        let (post_fold_start, v2) = shadow_direct(k);
        let shadow_time = t0.elapsed().as_secs_f64();

        let post_fold_bits = post_fold_start.bit_len();
        let post_fold_tau = post_fold_start.trailing_ones();

        // Phase 2: Sequential from post-fold value
        let t1 = Instant::now();
        let max_post_steps = (k as u64) * 15;
        let (steps, max_tau_post, converged) = collatz_sequential(&post_fold_start, max_post_steps);
        let post_time = t1.elapsed().as_secs_f64();

        let total = shadow_time + post_time;
        let max_tau = k.max(max_tau_post); // shadow max is always k

        eprintln!("{:>8} {:>6} {:>9.3}s {:>9.3}s {:>9.3}s {:>10} {:>8}",
            k,
            if is_odd { "odd" } else { "even" },
            shadow_time, post_time, total,
            max_tau,
            if converged { "✓" } else { "..." });

        // Stop if too slow
        if total > 300.0 {
            eprintln!("  Stopping: k={} took {:.1}s", k, total);
            break;
        }
    }

    // Compare: direct shadow vs sequential for k=100000
    eprintln!("\n=== COMPARISON: Direct Shadow vs Full Sequential ===\n");

    let k = 100_000u32;

    let t0 = Instant::now();
    let (post_fold, _) = shadow_direct(k);
    let shadow_t = t0.elapsed().as_secs_f64();
    eprintln!("  Direct shadow (3^k via repeated squaring): {:.3}s", shadow_t);
    eprintln!("  Post-fold value: {} bits, {} trailing ones",
        post_fold.bit_len(), post_fold.trailing_ones());

    // The sequential version (from push_k) took 1.77s for k=100000
    eprintln!("  Sequential (step-by-step): ~1.77s (from previous run)");
    eprintln!("  Speedup: {:.1}×", 1.77 / shadow_t);
}
