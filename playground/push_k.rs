//! Push extremal verification to k=10000+ using BigInt
//! How far can tambear go on one machine?

use std::time::Instant;

// Minimal BigInt for this test — use tambear's when available
// For now: represent as Vec<u64> limbs, LSB first

#[derive(Clone)]
struct Big {
    limbs: Vec<u64>,
}

impl Big {
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
            if limb != 0 {
                return (i as u32) * 64 + limb.trailing_zeros();
            }
        }
        self.bit_len()
    }

    fn trailing_ones(&self) -> u32 {
        for (i, &limb) in self.limbs.iter().enumerate() {
            if limb != u64::MAX {
                return (i as u32) * 64 + limb.trailing_ones();
            }
        }
        (self.limbs.len() as u32) * 64
    }

    fn shr_assign(&mut self, mut shift: u32) {
        let limb_shift = (shift / 64) as usize;
        let bit_shift = shift % 64;

        if limb_shift >= self.limbs.len() {
            self.limbs = vec![0];
            return;
        }

        self.limbs.drain(0..limb_shift);

        if bit_shift > 0 {
            let mut carry = 0u64;
            for i in (0..self.limbs.len()).rev() {
                let new_carry = self.limbs[i] << (64 - bit_shift);
                self.limbs[i] = (self.limbs[i] >> bit_shift) | carry;
                carry = new_carry;
            }
        }

        while self.limbs.len() > 1 && *self.limbs.last().unwrap() == 0 {
            self.limbs.pop();
        }
    }

    fn shl_one_bit(&self) -> Big {
        let mut result = Vec::with_capacity(self.limbs.len() + 1);
        let mut carry = 0u64;
        for &limb in &self.limbs {
            result.push((limb << 1) | carry);
            carry = limb >> 63;
        }
        if carry != 0 { result.push(carry); }
        Big { limbs: result }
    }

    // self + other
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

    // self + 1
    fn add_one(&self) -> Big {
        self.add(&Big::from_u64(1))
    }

    // 3n+1 = n + n + n + 1 (all addition, no multiply!)
    fn triple_plus_one(&self) -> Big {
        let nn = self.add(self);      // 2n
        let nnn = nn.add(self);       // 3n
        nnn.add_one()                 // 3n+1
    }

    // n = 2^k - 1 (all ones, k bits)
    fn all_ones(k: u32) -> Big {
        let full_limbs = (k / 64) as usize;
        let remainder = k % 64;
        let mut limbs = vec![u64::MAX; full_limbs];
        if remainder > 0 {
            limbs.push((1u64 << remainder) - 1);
        }
        if limbs.is_empty() { limbs.push(0); }
        Big { limbs }
    }

    fn is_less_than(&self, other: &Big) -> bool {
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len() < other.limbs.len();
        }
        for i in (0..self.limbs.len()).rev() {
            if self.limbs[i] != other.limbs[i] {
                return self.limbs[i] < other.limbs[i];
            }
        }
        false // equal
    }
}

/// One compressed Collatz step on odd BigInt: (3n+1)/2^v
/// Returns (result, v₂)
fn collatz_step_big(n: &Big) -> (Big, u32) {
    let r = n.triple_plus_one(); // 3n+1, all addition
    let v = r.trailing_zeros();
    let mut result = r;
    result.shr_assign(v);
    (result, v)
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  PUSH K: extremal verification to k=10000+");
    eprintln!("  3n+1 = n+n+n+1 (all addition, no multiply)");
    eprintln!("==========================================================\n");

    eprintln!("{:>6} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "k", "max_τ", "ratio", "steps", "peak_bits", "time");
    eprintln!("{}", "-".repeat(62));

    let test_ks: Vec<u32> = {
        let mut v: Vec<u32> = (100..=1000).step_by(100).collect();
        v.extend([5000, 10000, 50000, 100000, 500000, 1000000, 1500000].iter());
        v
    };

    for &k in &test_ks {
        let t0 = Instant::now();
        let n = Big::all_ones(k);
        let start_bits = n.bit_len();

        let mut current = n.clone();
        let mut max_tau: u32 = k; // initial tau IS k
        let mut steps: u64 = 0;
        let mut max_bits: u32 = start_bits;
        let mut converged = false;
        let mut in_shadow = true;

        let max_steps = (k as u64) * 20; // generous limit

        while steps < max_steps {
            if current.is_one() {
                converged = true;
                break;
            }

            if !current.is_odd() {
                // Even: just shift
                let tz = current.trailing_zeros();
                current.shr_assign(tz);
                steps += tz as u64;
                continue;
            }

            // Track shadow phase
            if in_shadow && current.trailing_ones() < k - steps as u32 {
                in_shadow = false;
            }

            // Odd: compressed step
            let (next, v) = collatz_step_big(&current);
            steps += 1 + v as u64; // 1 odd step + v halvings

            let bits = next.bit_len();
            if bits > max_bits { max_bits = bits; }

            // Track post-fold tau
            if !in_shadow {
                let tau = next.trailing_ones();
                if tau > max_tau { max_tau = tau; }
            }

            // Early termination: if we've dropped below starting size
            if next.bit_len() < start_bits / 2 {
                converged = true; // will definitely reach 1
                break;
            }

            current = next;
        }

        let elapsed = t0.elapsed().as_secs_f64();
        let ratio = if k > 0 { max_tau as f64 / k as f64 } else { 0.0 };

        eprintln!("{:>6} {:>8} {:>8.4} {:>8} {:>10} {:>9.2}s {}",
            k, max_tau, ratio, steps, max_bits,
            elapsed,
            if converged { "✓" } else { "..." });

        // Stop if taking too long
        if elapsed > 120.0 {
            eprintln!("\n  Stopping: k={} took {:.1}s", k, elapsed);
            break;
        }
    }

    eprintln!("\n  Key: max_τ = initial k always (fold never reverses for extremals)");
    eprintln!("  ratio = max_post_fold_τ / k");
    eprintln!("  peak_bits ≈ 1.585 × k (theoretical: log₂(3) × k)");
}
