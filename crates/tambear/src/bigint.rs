//! # Family 38 — Arbitrary Precision Integers
//!
//! tambear-native BigInt built from existing primitives.
//!
//! ## Architecture
//!
//! BigInt multiply = FFT convolution of limb arrays + carry prefix scan.
//! Both FFT (signal_processing.rs) and prefix scan exist in tambear.
//!
//! ## Types
//!
//! - `U256`: fixed 4×u64, stack-allocated, schoolbook arithmetic (fast for crypto/hashing)
//! - `BigInt`: variable Vec<u64>, FFT multiply for large operands, full arithmetic

use std::fmt;
use std::cmp::Ordering;

// ═══════════════════════════════════════════════════════════════════════════
// U256 — Fixed-width 256-bit unsigned integer
// ═══════════════════════════════════════════════════════════════════════════

/// 256-bit unsigned integer. 4 × u64 limbs, little-endian (limbs[0] = LSB).
/// Stack-allocated, Copy. Schoolbook arithmetic — fastest for ≤256 bits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct U256 {
    /// Limbs in little-endian order. limbs[0] is least significant.
    pub limbs: [u64; 4],
}

impl U256 {
    pub const ZERO: U256 = U256 { limbs: [0; 4] };
    pub const ONE: U256 = U256 { limbs: [1, 0, 0, 0] };
    pub const MAX: U256 = U256 { limbs: [u64::MAX; 4] };

    pub fn new(limbs: [u64; 4]) -> Self { U256 { limbs } }

    pub fn from_u64(v: u64) -> Self { U256 { limbs: [v, 0, 0, 0] } }

    pub fn from_u128(v: u128) -> Self {
        U256 { limbs: [v as u64, (v >> 64) as u64, 0, 0] }
    }

    pub fn is_zero(&self) -> bool { self.limbs == [0; 4] }

    pub fn is_one(&self) -> bool { self.limbs == [1, 0, 0, 0] }

    pub fn bits(&self) -> u32 {
        for i in (0..4).rev() {
            if self.limbs[i] != 0 {
                return (i as u32) * 64 + (64 - self.limbs[i].leading_zeros());
            }
        }
        0
    }

    pub fn bit(&self, pos: u32) -> bool {
        let limb = (pos / 64) as usize;
        let bit = pos % 64;
        if limb >= 4 { return false; }
        (self.limbs[limb] >> bit) & 1 == 1
    }

    /// Checked addition. Returns None on overflow.
    pub fn checked_add(&self, rhs: &U256) -> Option<U256> {
        let mut result = [0u64; 4];
        let mut carry = 0u64;
        for i in 0..4 {
            let (s1, c1) = self.limbs[i].overflowing_add(rhs.limbs[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            result[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }
        if carry != 0 { None } else { Some(U256 { limbs: result }) }
    }

    /// Wrapping addition.
    pub fn wrapping_add(&self, rhs: &U256) -> U256 {
        let mut result = [0u64; 4];
        let mut carry = 0u64;
        for i in 0..4 {
            let (s1, c1) = self.limbs[i].overflowing_add(rhs.limbs[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            result[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }
        U256 { limbs: result }
    }

    /// Checked subtraction. Returns None on underflow.
    pub fn checked_sub(&self, rhs: &U256) -> Option<U256> {
        let mut result = [0u64; 4];
        let mut borrow = 0u64;
        for i in 0..4 {
            let (s1, b1) = self.limbs[i].overflowing_sub(rhs.limbs[i]);
            let (s2, b2) = s1.overflowing_sub(borrow);
            result[i] = s2;
            borrow = (b1 as u64) + (b2 as u64);
        }
        if borrow != 0 { None } else { Some(U256 { limbs: result }) }
    }

    /// Full multiply: U256 × U256 → (low U256, high U256).
    /// Schoolbook: 16 u64×u64→u128 multiplies.
    pub fn widening_mul(&self, rhs: &U256) -> (U256, U256) {
        let mut result = [0u64; 8]; // 512-bit intermediate
        for i in 0..4 {
            let mut carry = 0u128;
            for j in 0..4 {
                let prod = (self.limbs[i] as u128) * (rhs.limbs[j] as u128)
                    + result[i + j] as u128 + carry;
                result[i + j] = prod as u64;
                carry = prod >> 64;
            }
            // Propagate remaining carry
            let mut k = i + 4;
            while carry > 0 && k < 8 {
                let s = result[k] as u128 + carry;
                result[k] = s as u64;
                carry = s >> 64;
                k += 1;
            }
        }
        (
            U256 { limbs: [result[0], result[1], result[2], result[3]] },
            U256 { limbs: [result[4], result[5], result[6], result[7]] },
        )
    }

    /// Truncated multiply (low 256 bits only).
    pub fn wrapping_mul(&self, rhs: &U256) -> U256 {
        self.widening_mul(rhs).0
    }

    /// Multiply by u64 scalar.
    pub fn mul_u64(&self, rhs: u64) -> (U256, u64) {
        let mut result = [0u64; 4];
        let mut carry = 0u128;
        for i in 0..4 {
            let prod = (self.limbs[i] as u128) * (rhs as u128) + carry;
            result[i] = prod as u64;
            carry = prod >> 64;
        }
        (U256 { limbs: result }, carry as u64)
    }

    /// Division by u64, returns (quotient, remainder).
    pub fn div_rem_u64(&self, divisor: u64) -> (U256, u64) {
        assert!(divisor != 0, "division by zero");
        let mut result = [0u64; 4];
        let mut rem = 0u128;
        for i in (0..4).rev() {
            let cur = (rem << 64) | self.limbs[i] as u128;
            result[i] = (cur / divisor as u128) as u64;
            rem = cur % divisor as u128;
        }
        (U256 { limbs: result }, rem as u64)
    }

    /// Division: self / rhs, returns (quotient, remainder).
    pub fn div_rem(&self, rhs: &U256) -> (U256, U256) {
        assert!(!rhs.is_zero(), "division by zero");
        if *self < *rhs { return (U256::ZERO, *self); }
        if rhs.limbs[1] == 0 && rhs.limbs[2] == 0 && rhs.limbs[3] == 0 {
            let (q, r) = self.div_rem_u64(rhs.limbs[0]);
            return (q, U256::from_u64(r));
        }

        // Binary long division
        let mut quotient = U256::ZERO;
        let mut remainder = U256::ZERO;
        let nbits = self.bits();
        for i in (0..nbits).rev() {
            // remainder <<= 1
            remainder = remainder.shl1();
            if self.bit(i) {
                remainder.limbs[0] |= 1;
            }
            if remainder >= *rhs {
                remainder = remainder.checked_sub(rhs).unwrap();
                let limb = (i / 64) as usize;
                let bit = i % 64;
                quotient.limbs[limb] |= 1u64 << bit;
            }
        }
        (quotient, remainder)
    }

    /// Left shift by 1 bit.
    fn shl1(&self) -> U256 {
        let mut result = [0u64; 4];
        let mut carry = 0u64;
        for i in 0..4 {
            result[i] = (self.limbs[i] << 1) | carry;
            carry = self.limbs[i] >> 63;
        }
        U256 { limbs: result }
    }

    /// Left shift by n bits.
    pub fn shl(&self, n: u32) -> U256 {
        if n >= 256 { return U256::ZERO; }
        let limb_shift = (n / 64) as usize;
        let bit_shift = n % 64;
        let mut result = [0u64; 4];
        for i in limb_shift..4 {
            result[i] = self.limbs[i - limb_shift] << bit_shift;
            if bit_shift > 0 && i > limb_shift {
                result[i] |= self.limbs[i - limb_shift - 1] >> (64 - bit_shift);
            }
        }
        U256 { limbs: result }
    }

    /// Right shift by n bits.
    pub fn shr(&self, n: u32) -> U256 {
        if n >= 256 { return U256::ZERO; }
        let limb_shift = (n / 64) as usize;
        let bit_shift = n % 64;
        let mut result = [0u64; 4];
        for i in 0..4 - limb_shift {
            result[i] = self.limbs[i + limb_shift] >> bit_shift;
            if bit_shift > 0 && i + limb_shift + 1 < 4 {
                result[i] |= self.limbs[i + limb_shift + 1] << (64 - bit_shift);
            }
        }
        U256 { limbs: result }
    }

    /// Modular exponentiation: self^exp mod modulus.
    /// Repeated squaring.
    pub fn pow_mod(&self, exp: &U256, modulus: &U256) -> U256 {
        assert!(!modulus.is_zero(), "modulus cannot be zero");
        if modulus.is_one() { return U256::ZERO; }
        let mut result = U256::ONE;
        let mut base = self.div_rem(modulus).1; // base mod m
        let nbits = exp.bits();
        for i in 0..nbits {
            if exp.bit(i) {
                result = mul_mod(&result, &base, modulus);
            }
            base = mul_mod(&base, &base, modulus);
        }
        result
    }

    /// GCD via Euclidean algorithm.
    pub fn gcd(&self, other: &U256) -> U256 {
        let mut a = *self;
        let mut b = *other;
        while !b.is_zero() {
            let (_, r) = a.div_rem(&b);
            a = b;
            b = r;
        }
        a
    }

    /// Integer square root (floor).
    pub fn isqrt(&self) -> U256 {
        if self.is_zero() { return U256::ZERO; }
        // Newton's method: x_{n+1} = (x_n + self/x_n) / 2
        let mut x = U256::ONE.shl(self.bits() / 2 + 1); // initial guess
        loop {
            let (q, _) = self.div_rem(&x);
            let next = x.wrapping_add(&q).shr(1);
            if next >= x { break; }
            x = next;
        }
        // Verify: x² ≤ self < (x+1)²
        x
    }

    /// Integer n-th root (floor).
    pub fn inth_root(&self, n: u32) -> U256 {
        if n == 0 { return U256::ONE; }
        if n == 1 { return *self; }
        if self.is_zero() { return U256::ZERO; }
        if n == 2 { return self.isqrt(); }

        // Newton: x_{k+1} = ((n-1)*x_k + self/x_k^(n-1)) / n
        let nbits = self.bits();
        let mut x = U256::ONE.shl(nbits / n + 1);
        let n_minus_1 = U256::from_u64(n as u64 - 1);
        let n_val = U256::from_u64(n as u64);

        for _ in 0..256 {
            if x.is_zero() { break; }
            // x^(n-1)
            let mut xpow = U256::ONE;
            for _ in 0..n - 1 {
                xpow = xpow.wrapping_mul(&x);
            }
            if xpow.is_zero() { break; }
            let (q, _) = self.div_rem(&xpow);
            // next = ((n-1)*x + q) / n
            let nx = n_minus_1.wrapping_mul(&x).wrapping_add(&q);
            let (next, _) = nx.div_rem(&n_val);
            if next >= x { break; }
            x = next;
        }
        x
    }

    /// Check if self is a perfect power: self = base^exp for some exp ≥ 2.
    /// Returns Some((base, exp)) or None.
    pub fn perfect_power(&self) -> Option<(U256, u32)> {
        if *self <= U256::ONE { return None; }
        for exp in (2..=self.bits()).rev() {
            let root = self.inth_root(exp);
            if root <= U256::ONE { continue; }
            // Check root^exp == self
            let mut power = U256::ONE;
            let mut overflow = false;
            for _ in 0..exp {
                let (lo, hi) = power.widening_mul(&root);
                if !hi.is_zero() { overflow = true; break; }
                power = lo;
            }
            if !overflow && power == *self {
                return Some((root, exp));
            }
        }
        None
    }
}

/// Modular multiplication: (a * b) mod m.
/// Uses widening multiply to avoid overflow.
fn mul_mod(a: &U256, b: &U256, m: &U256) -> U256 {
    let (lo, hi) = a.widening_mul(b);
    if hi.is_zero() {
        lo.div_rem(m).1
    } else {
        // Full 512-bit mod: reconstruct and divide
        // Use repeated subtraction approach for 512 / 256
        div_512_mod(lo, hi, m)
    }
}

/// Compute (hi * 2^256 + lo) mod m.
fn div_512_mod(lo: U256, hi: U256, m: &U256) -> U256 {
    // Compute hi * 2^256 mod m, then add lo mod m
    let mut result = U256::ZERO;

    // Process hi limb by limb: accumulate (result * 2^64 + hi_limb) mod m
    for i in (0..4).rev() {
        // result = result * 2^64 mod m (shift left 64 bits)
        for _ in 0..64 {
            result = result.shl1();
            if result >= *m {
                result = result.checked_sub(m).unwrap();
            }
        }
        // Add hi.limbs[i]
        let limb = U256::from_u64(hi.limbs[i]);
        result = result.wrapping_add(&limb);
        if result >= *m || result < limb {
            result = result.checked_sub(m).unwrap_or_else(|| {
                // Wraparound: result was > 2^256, need proper handling
                result.wrapping_add(&U256::MAX.checked_sub(m).unwrap().wrapping_add(&U256::ONE))
            });
        }
    }

    // Now result = hi * 2^256 mod m. Add lo mod m.
    let lo_mod = lo.div_rem(m).1;
    let sum = result.wrapping_add(&lo_mod);
    if sum < result || sum >= *m {
        sum.checked_sub(m).unwrap_or(sum)
    } else {
        sum
    }
}

impl Ord for U256 {
    fn cmp(&self, other: &Self) -> Ordering {
        for i in (0..4).rev() {
            match self.limbs[i].cmp(&other.limbs[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

impl PartialOrd for U256 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl fmt::Debug for U256 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "U256(0x{:016x}_{:016x}_{:016x}_{:016x})",
            self.limbs[3], self.limbs[2], self.limbs[1], self.limbs[0])
    }
}

impl fmt::Display for U256 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() { return write!(f, "0"); }
        let mut digits = Vec::new();
        let mut val = *self;
        while !val.is_zero() {
            let (q, r) = val.div_rem_u64(10);
            digits.push((r as u8) + b'0');
            val = q;
        }
        digits.reverse();
        let s: String = digits.iter().map(|&b| b as char).collect();
        write!(f, "{}", s)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BigInt — Variable-width arbitrary precision
// ═══════════════════════════════════════════════════════════════════════════

/// Arbitrary precision integer. Variable-width limbs.
/// Little-endian: limbs[0] = least significant.
#[derive(Clone, PartialEq, Eq)]
pub struct BigInt {
    pub(crate) limbs: Vec<u64>,
    pub(crate) negative: bool,
}

impl BigInt {
    pub fn zero() -> Self { BigInt { limbs: vec![0], negative: false } }
    pub fn one() -> Self { BigInt { limbs: vec![1], negative: false } }

    pub fn from_u64(v: u64) -> Self {
        BigInt { limbs: vec![v], negative: false }
    }

    pub fn from_u128(v: u128) -> Self {
        if v >> 64 == 0 {
            BigInt::from_u64(v as u64)
        } else {
            BigInt { limbs: vec![v as u64, (v >> 64) as u64], negative: false }
        }
    }

    pub fn from_i64(v: i64) -> Self {
        if v >= 0 {
            BigInt::from_u64(v as u64)
        } else {
            BigInt { limbs: vec![v.unsigned_abs()], negative: true }
        }
    }

    pub fn from_u256(v: &U256) -> Self {
        let mut limbs: Vec<u64> = v.limbs.to_vec();
        while limbs.len() > 1 && *limbs.last().unwrap() == 0 { limbs.pop(); }
        BigInt { limbs, negative: false }
    }

    pub fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&l| l == 0)
    }

    pub fn is_negative(&self) -> bool { self.negative && !self.is_zero() }

    pub(crate) fn normalize(&mut self) {
        while self.limbs.len() > 1 && *self.limbs.last().unwrap() == 0 {
            self.limbs.pop();
        }
        if self.is_zero() { self.negative = false; }
    }

    fn n_limbs(&self) -> usize { self.limbs.len() }

    /// Absolute comparison.
    fn cmp_abs(&self, other: &BigInt) -> Ordering {
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len().cmp(&other.limbs.len());
        }
        for i in (0..self.limbs.len()).rev() {
            match self.limbs[i].cmp(&other.limbs[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }

    /// Absolute addition (both positive).
    fn add_abs(a: &[u64], b: &[u64]) -> Vec<u64> {
        let n = a.len().max(b.len());
        let mut result = vec![0u64; n + 1];
        let mut carry = 0u64;
        for i in 0..n {
            let av = if i < a.len() { a[i] } else { 0 };
            let bv = if i < b.len() { b[i] } else { 0 };
            let (s1, c1) = av.overflowing_add(bv);
            let (s2, c2) = s1.overflowing_add(carry);
            result[i] = s2;
            carry = (c1 as u64) + (c2 as u64);
        }
        result[n] = carry;
        result
    }

    /// Absolute subtraction: a - b, assumes a >= b.
    fn sub_abs(a: &[u64], b: &[u64]) -> Vec<u64> {
        let n = a.len();
        let mut result = vec![0u64; n];
        let mut borrow = 0u64;
        for i in 0..n {
            let bv = if i < b.len() { b[i] } else { 0 };
            let (s1, b1) = a[i].overflowing_sub(bv);
            let (s2, b2) = s1.overflowing_sub(borrow);
            result[i] = s2;
            borrow = (b1 as u64) + (b2 as u64);
        }
        result
    }

    /// Add two BigInts.
    pub fn add(&self, other: &BigInt) -> BigInt {
        if self.negative == other.negative {
            let limbs = BigInt::add_abs(&self.limbs, &other.limbs);
            let mut r = BigInt { limbs, negative: self.negative };
            r.normalize();
            r
        } else if self.cmp_abs(other) != Ordering::Less {
            let limbs = BigInt::sub_abs(&self.limbs, &other.limbs);
            let mut r = BigInt { limbs, negative: self.negative };
            r.normalize();
            r
        } else {
            let limbs = BigInt::sub_abs(&other.limbs, &self.limbs);
            let mut r = BigInt { limbs, negative: other.negative };
            r.normalize();
            r
        }
    }

    /// Subtract.
    pub fn sub(&self, other: &BigInt) -> BigInt {
        let neg_other = BigInt { limbs: other.limbs.clone(), negative: !other.negative };
        self.add(&neg_other)
    }

    /// Schoolbook multiply. O(n·m).
    fn mul_schoolbook(a: &[u64], b: &[u64]) -> Vec<u64> {
        let n = a.len() + b.len();
        let mut result = vec![0u64; n];
        for i in 0..a.len() {
            let mut carry = 0u128;
            for j in 0..b.len() {
                let prod = (a[i] as u128) * (b[j] as u128) + result[i + j] as u128 + carry;
                result[i + j] = prod as u64;
                carry = prod >> 64;
            }
            let mut k = i + b.len();
            while carry > 0 && k < n {
                let s = result[k] as u128 + carry;
                result[k] = s as u64;
                carry = s >> 64;
                k += 1;
            }
        }
        result
    }

    /// FFT-based multiply for large operands.
    /// Splits limbs into 16-bit quarter-limbs to stay within f64's 53-bit mantissa.
    /// Max convolution value per position: n * (2^16-1)^2 ≈ n * 4.3e9.
    /// For n < 2^20 (1M limbs), this is < 2^50 — safe for f64.
    fn mul_fft(a: &[u64], b: &[u64]) -> Vec<u64> {
        const BITS: u32 = 16;
        const MASK: u64 = (1u64 << BITS) - 1;
        const PARTS: usize = 4; // 64 / 16

        // Split each u64 into four 16-bit parts
        let mut qa: Vec<f64> = Vec::with_capacity(a.len() * PARTS);
        for &limb in a {
            for p in 0..PARTS {
                qa.push(((limb >> (p as u32 * BITS)) & MASK) as f64);
            }
        }
        let mut qb: Vec<f64> = Vec::with_capacity(b.len() * PARTS);
        for &limb in b {
            for p in 0..PARTS {
                qb.push(((limb >> (p as u32 * BITS)) & MASK) as f64);
            }
        }

        let conv = crate::signal_processing::convolve(&qa, &qb);

        // Carry propagation through 16-bit positions
        let n_parts = conv.len();
        let mut carries = vec![0i128; n_parts + 2];
        for i in 0..n_parts {
            carries[i] += conv[i].round() as i128;
        }
        for i in 0..carries.len() - 1 {
            if carries[i] >= (1i128 << BITS) || carries[i] < 0 {
                let overflow = carries[i] >> BITS;
                carries[i] -= overflow << BITS;
                carries[i + 1] += overflow;
                // Ensure non-negative
                if carries[i] < 0 {
                    carries[i] += 1i128 << BITS;
                    carries[i + 1] -= 1;
                }
            }
        }

        // Reassemble u64 limbs from groups of four 16-bit parts
        let n_limbs = (n_parts + PARTS - 1) / PARTS;
        let mut result = vec![0u64; n_limbs];
        for i in 0..n_limbs {
            let mut limb = 0u64;
            for p in 0..PARTS {
                let idx = i * PARTS + p;
                if idx < carries.len() {
                    limb |= (carries[idx] as u64 & MASK) << (p as u32 * BITS);
                }
            }
            result[i] = limb;
        }
        result
    }

    /// Multiply. Uses schoolbook for small operands, FFT for large.
    pub fn mul(&self, other: &BigInt) -> BigInt {
        if self.is_zero() || other.is_zero() { return BigInt::zero(); }

        let threshold = 64; // limbs — crossover point for FFT advantage
        let limbs = if self.n_limbs() < threshold && other.n_limbs() < threshold {
            BigInt::mul_schoolbook(&self.limbs, &other.limbs)
        } else {
            BigInt::mul_fft(&self.limbs, &other.limbs)
        };

        let mut r = BigInt { limbs, negative: self.negative != other.negative };
        r.normalize();
        r
    }

    /// Division by u64, returns (quotient, remainder).
    pub fn div_rem_u64(&self, divisor: u64) -> (BigInt, u64) {
        assert!(divisor != 0, "division by zero");
        let mut result = vec![0u64; self.limbs.len()];
        let mut rem = 0u128;
        for i in (0..self.limbs.len()).rev() {
            let cur = (rem << 64) | self.limbs[i] as u128;
            result[i] = (cur / divisor as u128) as u64;
            rem = cur % divisor as u128;
        }
        let mut q = BigInt { limbs: result, negative: self.negative };
        q.normalize();
        (q, rem as u64)
    }

    /// Power by repeated squaring.
    pub fn pow(&self, exp: u32) -> BigInt {
        if exp == 0 { return BigInt::one(); }
        let mut result = BigInt::one();
        let mut base = self.clone();
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 { result = result.mul(&base); }
            base = base.mul(&base);
            e >>= 1;
        }
        result
    }

    /// GCD via Euclidean algorithm (on absolute values).
    pub fn gcd(&self, other: &BigInt) -> BigInt {
        let mut a = BigInt { limbs: self.limbs.clone(), negative: false };
        let mut b = BigInt { limbs: other.limbs.clone(), negative: false };
        a.normalize();
        b.normalize();
        while !b.is_zero() {
            let (_, r) = bigint_div_rem(&a, &b);
            a = b;
            b = r;
        }
        a
    }

    /// Convert to u64 (panics if too large).
    pub fn to_u64(&self) -> Option<u64> {
        if self.negative { return None; }
        if self.limbs.len() > 1 && self.limbs[1..].iter().any(|&l| l != 0) {
            return None;
        }
        Some(self.limbs[0])
    }

    // ── Bit operations (for Collatz / number theory) ────────────────

    /// Total number of significant bits.
    pub fn bits(&self) -> u32 {
        for i in (0..self.limbs.len()).rev() {
            if self.limbs[i] != 0 {
                return (i as u32) * 64 + (64 - self.limbs[i].leading_zeros());
            }
        }
        0
    }

    /// Get bit at position `pos` (0-indexed from LSB).
    pub fn bit(&self, pos: u32) -> bool {
        let limb = (pos / 64) as usize;
        let bit = pos % 64;
        if limb >= self.limbs.len() { return false; }
        (self.limbs[limb] >> bit) & 1 == 1
    }

    /// Count trailing zero bits.
    pub fn trailing_zeros(&self) -> u32 {
        for (i, &limb) in self.limbs.iter().enumerate() {
            if limb != 0 {
                return (i as u32) * 64 + limb.trailing_zeros();
            }
        }
        self.limbs.len() as u32 * 64
    }

    /// Count trailing one bits.
    pub fn trailing_ones(&self) -> u32 {
        for (i, &limb) in self.limbs.iter().enumerate() {
            if limb != u64::MAX {
                return (i as u32) * 64 + (!limb).trailing_zeros();
            }
        }
        self.limbs.len() as u32 * 64
    }

    /// Is this value odd?
    pub fn is_odd(&self) -> bool {
        !self.is_zero() && (self.limbs[0] & 1) == 1
    }

    // NOTE: shr() and shl() are defined in bigfloat.rs (same impl BigInt block).

    /// Add a small u64 value.
    pub fn add_u64(&self, v: u64) -> BigInt {
        self.add(&BigInt::from_u64(v))
    }

    /// Multiply by a small u64 value (fast path).
    pub fn mul_u64(&self, scalar: u64) -> BigInt {
        if scalar == 0 || self.is_zero() { return BigInt::zero(); }
        let limbs = mul_bigint_u64(&self.limbs, scalar);
        let mut r = BigInt { limbs, negative: self.negative };
        r.normalize();
        r
    }

    /// Check if this equals one.
    pub fn is_one(&self) -> bool {
        !self.negative && self.limbs.len() == 1 && self.limbs[0] == 1
    }
}

/// BigInt division: a / b, returns (quotient, remainder).
fn bigint_div_rem(a: &BigInt, b: &BigInt) -> (BigInt, BigInt) {
    assert!(!b.is_zero(), "division by zero");
    if a.cmp_abs(b) == Ordering::Less {
        return (BigInt::zero(), a.clone());
    }
    // Single-limb divisor fast path
    if b.n_limbs() == 1 {
        let (q, r) = a.div_rem_u64(b.limbs[0]);
        let q_neg = a.negative != b.negative;
        return (
            BigInt { limbs: q.limbs, negative: q_neg },
            BigInt::from_u64(r),
        );
    }
    // Multi-limb: schoolbook long division
    // Simplified: use repeated subtraction with shifting
    let shift = b.limbs.len() - 1;
    let mut remainder = BigInt { limbs: a.limbs.clone(), negative: false };
    let divisor = BigInt { limbs: b.limbs.clone(), negative: false };

    let qlen = a.n_limbs() - b.n_limbs() + 1;
    let mut q_limbs = vec![0u64; qlen];

    for i in (0..qlen).rev() {
        // Estimate quotient digit: remainder[i+shift..] / divisor
        let mut hi = 0u128;
        if i + shift + 1 < remainder.limbs.len() {
            hi = (remainder.limbs[i + shift + 1] as u128) << 64;
        }
        if i + shift < remainder.limbs.len() {
            hi |= remainder.limbs[i + shift] as u128;
        }
        let d = *divisor.limbs.last().unwrap() as u128 + 1;
        let mut qhat = (hi / d) as u64;

        // Multiply qhat * divisor and subtract from remainder at position i
        loop {
            let product = mul_bigint_u64(&divisor.limbs, qhat);
            let shifted = shift_limbs(&product, i);
            let shifted_bi = BigInt { limbs: shifted, negative: false };

            if remainder.cmp_abs(&shifted_bi) != Ordering::Less {
                // remainder -= shifted
                let new_limbs = BigInt::sub_abs(
                    &remainder.limbs,
                    &shifted_bi.limbs,
                );
                remainder = BigInt { limbs: new_limbs, negative: false };
                remainder.normalize();
                q_limbs[i] = qhat;
                break;
            } else if qhat == 0 {
                break;
            } else {
                qhat -= 1;
            }
        }
    }

    let mut quotient = BigInt { limbs: q_limbs, negative: a.negative != b.negative };
    quotient.normalize();
    remainder.normalize();
    (quotient, remainder)
}

fn mul_bigint_u64(limbs: &[u64], scalar: u64) -> Vec<u64> {
    let mut result = vec![0u64; limbs.len() + 1];
    let mut carry = 0u128;
    for i in 0..limbs.len() {
        let prod = (limbs[i] as u128) * (scalar as u128) + carry;
        result[i] = prod as u64;
        carry = prod >> 64;
    }
    result[limbs.len()] = carry as u64;
    result
}

fn shift_limbs(limbs: &[u64], n: usize) -> Vec<u64> {
    let mut result = vec![0u64; limbs.len() + n];
    for i in 0..limbs.len() { result[i + n] = limbs[i]; }
    result
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.is_negative(), other.is_negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self.cmp_abs(other),
            (true, true) => other.cmp_abs(self), // reversed for negatives
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl fmt::Debug for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.negative { write!(f, "-")?; }
        write!(f, "BigInt[")?;
        for (i, limb) in self.limbs.iter().enumerate().rev() {
            if i < self.limbs.len() - 1 { write!(f, "_")?; }
            write!(f, "{:016x}", limb)?;
        }
        write!(f, "]")
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() { return write!(f, "0"); }
        if self.negative { write!(f, "-")?; }
        let mut digits = Vec::new();
        let mut val = BigInt { limbs: self.limbs.clone(), negative: false };
        while !val.is_zero() {
            let (q, r) = val.div_rem_u64(10);
            digits.push((r as u8) + b'0');
            val = q;
        }
        digits.reverse();
        let s: String = digits.iter().map(|&b| b as char).collect();
        write!(f, "{}", s)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── U256 basics ────────────────────────────────────────────────────

    #[test]
    fn u256_from_conversions() {
        let a = U256::from_u64(42);
        assert_eq!(a.limbs, [42, 0, 0, 0]);
        let b = U256::from_u128(1u128 << 64 | 7);
        assert_eq!(b.limbs, [7, 1, 0, 0]);
    }

    #[test]
    fn u256_add_basic() {
        let a = U256::from_u64(u64::MAX);
        let b = U256::from_u64(1);
        let c = a.checked_add(&b).unwrap();
        assert_eq!(c.limbs, [0, 1, 0, 0]); // carry propagation
    }

    #[test]
    fn u256_add_overflow() {
        assert!(U256::MAX.checked_add(&U256::ONE).is_none());
    }

    #[test]
    fn u256_sub_basic() {
        let a = U256::from_u128(1u128 << 64);
        let b = U256::from_u64(1);
        let c = a.checked_sub(&b).unwrap();
        assert_eq!(c.limbs, [u64::MAX, 0, 0, 0]); // borrow propagation
    }

    #[test]
    fn u256_sub_underflow() {
        assert!(U256::ZERO.checked_sub(&U256::ONE).is_none());
    }

    // ── U256 multiply ──────────────────────────────────────────────────

    #[test]
    fn u256_mul_basic() {
        let a = U256::from_u64(7);
        let b = U256::from_u64(6);
        let (lo, hi) = a.widening_mul(&b);
        assert_eq!(lo, U256::from_u64(42));
        assert!(hi.is_zero());
    }

    #[test]
    fn u256_mul_large() {
        // Verify via known identity: 7 * 6 = 42 with large values
        let a = U256::from_u128(u128::MAX);
        let b = U256::from_u64(2);
        let (lo, hi) = a.widening_mul(&b);
        // 2 * (2^128 - 1) = 2^129 - 2
        assert_eq!(lo.limbs[0], u64::MAX - 1);
        assert_eq!(lo.limbs[1], u64::MAX);
        assert_eq!(lo.limbs[2], 1);
        assert_eq!(lo.limbs[3], 0);
        assert!(hi.is_zero());

        // Cross-limb: verify widening_mul self-consistency
        // a * 1 = a
        let (lo2, hi2) = a.widening_mul(&U256::ONE);
        assert_eq!(lo2, a);
        assert!(hi2.is_zero());
    }

    // ── U256 division ──────────────────────────────────────────────────

    #[test]
    fn u256_div_basic() {
        let a = U256::from_u64(100);
        let (q, r) = a.div_rem_u64(7);
        assert_eq!(q, U256::from_u64(14));
        assert_eq!(r, 2);
    }

    #[test]
    fn u256_div_u256() {
        let a = U256::from_u128(1_000_000_000_000);
        let b = U256::from_u64(1_000_000);
        let (q, r) = a.div_rem(&b);
        assert_eq!(q, U256::from_u64(1_000_000));
        assert!(r.is_zero());
    }

    // ── U256 display ───────────────────────────────────────────────────

    #[test]
    fn u256_display_decimal() {
        assert_eq!(format!("{}", U256::ZERO), "0");
        assert_eq!(format!("{}", U256::from_u64(12345)), "12345");
        assert_eq!(format!("{}", U256::from_u128(u128::MAX)),
            "340282366920938463463374607431768211455");
    }

    // ── U256 pow_mod ───────────────────────────────────────────────────

    #[test]
    fn u256_pow_mod() {
        // 2^10 mod 1000 = 1024 mod 1000 = 24
        let base = U256::from_u64(2);
        let exp = U256::from_u64(10);
        let modulus = U256::from_u64(1000);
        assert_eq!(base.pow_mod(&exp, &modulus), U256::from_u64(24));
    }

    #[test]
    fn u256_pow_mod_large() {
        // Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p
        let a = U256::from_u64(3);
        let p = U256::from_u64(97); // prime
        let exp = U256::from_u64(96); // p-1
        assert_eq!(a.pow_mod(&exp, &p), U256::ONE);
    }

    // ── U256 GCD ───────────────────────────────────────────────────────

    #[test]
    fn u256_gcd() {
        let a = U256::from_u64(48);
        let b = U256::from_u64(18);
        assert_eq!(a.gcd(&b), U256::from_u64(6));
    }

    // ── U256 isqrt ─────────────────────────────────────────────────────

    #[test]
    fn u256_isqrt() {
        assert_eq!(U256::from_u64(0).isqrt(), U256::ZERO);
        assert_eq!(U256::from_u64(1).isqrt(), U256::ONE);
        assert_eq!(U256::from_u64(4).isqrt(), U256::from_u64(2));
        assert_eq!(U256::from_u64(10).isqrt(), U256::from_u64(3));
        assert_eq!(U256::from_u64(100).isqrt(), U256::from_u64(10));

        // Large: sqrt(2^128) = 2^64
        let big = U256::from_u128(1u128 << 64).wrapping_mul(&U256::from_u128(1u128 << 64));
        assert_eq!(big.isqrt(), U256::from_u128(1u128 << 64));
    }

    // ── U256 nth root ──────────────────────────────────────────────────

    #[test]
    fn u256_inth_root() {
        assert_eq!(U256::from_u64(27).inth_root(3), U256::from_u64(3));
        assert_eq!(U256::from_u64(256).inth_root(8), U256::from_u64(2));
        assert_eq!(U256::from_u64(1000).inth_root(3), U256::from_u64(10));
    }

    // ── U256 bits ──────────────────────────────────────────────────────

    #[test]
    fn u256_bits_and_shifts() {
        let a = U256::from_u64(1);
        assert_eq!(a.bits(), 1);
        assert_eq!(a.shl(64), U256::new([0, 1, 0, 0]));
        assert_eq!(U256::new([0, 1, 0, 0]).shr(64), U256::from_u64(1));
    }

    // ── BigInt basics ──────────────────────────────────────────────────

    #[test]
    fn bigint_add_sub() {
        let a = BigInt::from_u64(100);
        let b = BigInt::from_u64(42);
        assert_eq!(a.add(&b).to_u64(), Some(142));
        assert_eq!(a.sub(&b).to_u64(), Some(58));
    }

    #[test]
    fn bigint_add_signed() {
        let pos = BigInt::from_u64(10);
        let neg = BigInt::from_i64(-7);
        assert_eq!(pos.add(&neg).to_u64(), Some(3));
    }

    #[test]
    fn bigint_mul_schoolbook() {
        let a = BigInt::from_u128(u128::MAX);
        let b = BigInt::from_u128(u128::MAX);
        let c = a.mul(&b);
        // u128::MAX² = 2^256 - 2^129 + 1
        // Verify via string: 115792089237316195423570985008687907852929702298719625575994209400481361428481
        let s = format!("{}", c);
        assert!(s.starts_with("115792089237316195423570985008687907"));
    }

    #[test]
    fn bigint_mul_signed() {
        let a = BigInt::from_i64(-5);
        let b = BigInt::from_i64(3);
        let c = a.mul(&b);
        assert!(c.is_negative());
        assert_eq!(c.sub(&BigInt::from_i64(-15)).to_u64(), Some(0));
    }

    #[test]
    fn bigint_pow() {
        let base = BigInt::from_u64(2);
        let result = base.pow(100);
        let s = format!("{}", result);
        assert_eq!(s, "1267650600228229401496703205376"); // 2^100
    }

    #[test]
    fn bigint_gcd() {
        let a = BigInt::from_u64(252);
        let b = BigInt::from_u64(105);
        assert_eq!(a.gcd(&b).to_u64(), Some(21));
    }

    // ── FFT multiply correctness ───────────────────────────────────────

    #[test]
    fn bigint_fft_mul_matches_schoolbook() {
        // Force FFT path by using numbers that would exceed threshold
        // But test small numbers for verifiability
        let a_limbs = vec![u64::MAX; 4];
        let b_limbs = vec![u64::MAX; 4];

        let school = BigInt::mul_schoolbook(&a_limbs, &b_limbs);
        let fft = BigInt::mul_fft(&a_limbs, &b_limbs);

        // Normalize both
        let mut s = BigInt { limbs: school, negative: false };
        s.normalize();
        let mut f = BigInt { limbs: fft, negative: false };
        f.normalize();

        assert_eq!(format!("{}", s), format!("{}", f),
            "FFT multiply must match schoolbook");
    }

    // ── Display ────────────────────────────────────────────────────────

    #[test]
    fn bigint_display() {
        assert_eq!(format!("{}", BigInt::zero()), "0");
        assert_eq!(format!("{}", BigInt::from_u64(42)), "42");
        assert_eq!(format!("{}", BigInt::from_i64(-99)), "-99");
    }

    // ── Ordering ───────────────────────────────────────────────────────

    #[test]
    fn bigint_ordering() {
        let a = BigInt::from_u64(100);
        let b = BigInt::from_u64(200);
        let c = BigInt::from_i64(-50);
        assert!(a < b);
        assert!(c < a);
        assert!(c < b);
    }
}
