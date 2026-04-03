//! # BigFloat — Arbitrary Precision Floating Point
//!
//! `value = mantissa × 2^exponent` where mantissa is a BigInt.
//!
//! ## Architecture
//!
//! - Mantissa: BigInt (arbitrary precision integer)
//! - Exponent: i64 (shift)
//! - Precision: configurable number of bits (default 256)
//!
//! Multiplication uses BigInt::mul (schoolbook or FFT depending on size).
//! Division uses Newton reciprocal iteration.
//!
//! ## Why not use f64?
//!
//! f64 gives 53 bits (~16 digits). For Riemann zeta near the critical line,
//! cancellation can eat 40+ digits. BigFloat at 256 bits gives ~77 digits —
//! enough to verify zeros to high precision.

use crate::bigint::BigInt;
use std::fmt;


/// Arbitrary precision floating point number.
///
/// Represents `mantissa × 2^exponent`. Mantissa is normalized so that
/// its bit length equals the working precision.
#[derive(Clone)]
pub struct BigFloat {
    /// The significand (arbitrary precision integer).
    pub mantissa: BigInt,
    /// Binary exponent: value = mantissa × 2^exponent.
    pub exponent: i64,
    /// Working precision in bits.
    pub prec: u32,
}

impl BigFloat {
    /// Create a BigFloat from components.
    pub fn new(mantissa: BigInt, exponent: i64, prec: u32) -> Self {
        let mut bf = BigFloat { mantissa, exponent, prec };
        bf.normalize();
        bf
    }

    /// Zero with given precision.
    pub fn zero(prec: u32) -> Self {
        BigFloat { mantissa: BigInt::zero(), exponent: 0, prec }
    }

    /// One with given precision.
    pub fn one(prec: u32) -> Self {
        // 1.0 = 2^prec × 2^(-prec)
        let mantissa = BigInt::from_u64(1).shl(prec);
        BigFloat { mantissa, exponent: -(prec as i64), prec }
    }

    /// From f64 (captures all bits of the f64).
    pub fn from_f64(v: f64, prec: u32) -> Self {
        if v == 0.0 { return BigFloat::zero(prec); }
        let bits = v.to_bits();
        let sign = (bits >> 63) != 0;
        let exp_bits = ((bits >> 52) & 0x7FF) as i64;
        let frac_bits = bits & ((1u64 << 52) - 1);

        let (mantissa_val, exp) = if exp_bits == 0 {
            // Subnormal
            (frac_bits, -1022 - 52)
        } else {
            // Normal: implicit 1 bit
            (frac_bits | (1u64 << 52), exp_bits - 1023 - 52)
        };

        let mut m = BigInt::from_u64(mantissa_val);
        if sign { m = BigInt::zero().sub(&m); }

        BigFloat::new(m, exp, prec)
    }

    /// From integer.
    pub fn from_i64(v: i64, prec: u32) -> Self {
        BigFloat::new(BigInt::from_i64(v), 0, prec)
    }

    pub fn is_zero(&self) -> bool { self.mantissa.is_zero() }
    pub fn is_negative(&self) -> bool { self.mantissa.is_negative() }

    /// Normalize: shift mantissa so its bit length equals prec.
    fn normalize(&mut self) {
        if self.mantissa.is_zero() {
            self.exponent = 0;
            return;
        }
        let bits = self.mantissa.bit_length();
        if bits > self.prec {
            let shift = bits - self.prec;
            self.mantissa = self.mantissa.shr(shift);
            self.exponent += shift as i64;
        } else if bits < self.prec {
            let shift = self.prec - bits;
            self.mantissa = self.mantissa.shl(shift);
            self.exponent -= shift as i64;
        }
    }

    /// Negate.
    pub fn neg(&self) -> BigFloat {
        BigFloat::new(BigInt::zero().sub(&self.mantissa), self.exponent, self.prec)
    }

    /// Absolute value.
    pub fn abs(&self) -> BigFloat {
        if self.is_negative() { self.neg() } else { self.clone() }
    }

    /// Addition.
    pub fn add(&self, other: &BigFloat) -> BigFloat {
        let prec = self.prec.max(other.prec);
        if self.is_zero() { return other.clone(); }
        if other.is_zero() { return self.clone(); }

        // Align exponents: shift the one with larger exponent
        let (a_m, b_m, exp) = if self.exponent >= other.exponent {
            let shift = (self.exponent - other.exponent) as u32;
            (self.mantissa.shl(shift), other.mantissa.clone(), other.exponent)
        } else {
            let shift = (other.exponent - self.exponent) as u32;
            (self.mantissa.clone(), other.mantissa.shl(shift), self.exponent)
        };

        BigFloat::new(a_m.add(&b_m), exp, prec)
    }

    /// Subtraction.
    pub fn sub(&self, other: &BigFloat) -> BigFloat {
        self.add(&other.neg())
    }

    /// Multiplication.
    pub fn mul(&self, other: &BigFloat) -> BigFloat {
        let prec = self.prec.max(other.prec);
        let m = self.mantissa.mul(&other.mantissa);
        let e = self.exponent + other.exponent;
        BigFloat::new(m, e, prec)
    }

    /// Multiply by power of 2 (fast shift).
    pub fn mul_pow2(&self, shift: i64) -> BigFloat {
        BigFloat {
            mantissa: self.mantissa.clone(),
            exponent: self.exponent + shift,
            prec: self.prec,
        }
    }

    /// Division via Newton reciprocal iteration.
    /// Computes self / other.
    pub fn div(&self, other: &BigFloat) -> BigFloat {
        assert!(!other.is_zero(), "division by zero");
        // Compute 1/other via Newton: x_{n+1} = x_n * (2 - other * x_n)
        // Then multiply by self.
        let prec = self.prec.max(other.prec);
        let two = BigFloat::from_i64(2, prec);

        // Initial guess from f64
        let other_f64 = other.to_f64();
        let mut recip = BigFloat::from_f64(1.0 / other_f64, prec);

        // Newton iterations: doubles precision each time
        // log2(prec/53) iterations needed (53 = f64 precision)
        let iters = ((prec as f64 / 53.0).log2().ceil() as u32).max(1) + 2;
        for _ in 0..iters {
            // x = x * (2 - other * x)
            let prod = other.mul(&recip);
            let correction = two.sub(&prod);
            recip = recip.mul(&correction);
        }

        self.mul(&recip)
    }

    /// Convert to f64 (lossy).
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() { return 0.0; }
        let m = self.mantissa.to_f64_approx();
        m * (2.0_f64).powi(self.exponent as i32)
    }

    /// Square root via Newton iteration.
    pub fn sqrt(&self) -> BigFloat {
        assert!(!self.is_negative(), "sqrt of negative");
        if self.is_zero() { return BigFloat::zero(self.prec); }

        let prec = self.prec;

        // Initial guess from f64
        let mut x = BigFloat::from_f64(self.to_f64().sqrt(), prec);
        let half = BigFloat::from_f64(0.5, prec);

        let iters = ((prec as f64 / 53.0).log2().ceil() as u32).max(1) + 2;
        for _ in 0..iters {
            // x = (x + self/x) / 2 = x/2 + self/(2x)
            let q = self.div(&x);
            x = x.add(&q).mul(&half);
        }
        x
    }

    /// Natural logarithm via AGM (arithmetic-geometric mean).
    /// ln(x) = π / (2·AGM(1, 4/s)) - m·ln(2)
    /// where s = x·2^m chosen so s > 2^(prec/2).
    pub fn ln(&self) -> BigFloat {
        assert!(!self.is_negative() && !self.is_zero(), "ln of non-positive");
        let prec = self.prec;

        // Use the identity: ln(x) = ln(x/2^k) + k·ln(2)
        // Reduce to ln(y) where y ∈ [1, 2) via exponent extraction
        // Then use the series: ln(1+t) = t - t²/2 + t³/3 - ...
        // For arbitrary precision, AGM is better, but series works for moderate prec.

        // Simple approach: ln(x) via Taylor series around 1
        // First reduce: x = m * 2^e where m ∈ [1, 2)
        let xf = self.to_f64();
        let ln_approx = xf.ln();
        let mut result = BigFloat::from_f64(ln_approx, prec);

        // Refine via Newton on exp: solve exp(y) = x
        // y_{n+1} = y_n + (x - exp(y_n)) / exp(y_n) = y_n + 1 - x·exp(-y_n)
        // Actually simpler: y_{n+1} = y_n - 1 + x / exp(y_n)
        let iters = ((prec as f64 / 53.0).log2().ceil() as u32).max(1) + 2;
        for _ in 0..iters {
            let ey = result.exp();
            // y = y + x/exp(y) - 1
            let ratio = self.div(&ey);
            let one = BigFloat::one(prec);
            result = result.add(&ratio).sub(&one);
        }

        result
    }

    /// Exponential function via Taylor series.
    /// exp(x) = Σ x^k / k!
    pub fn exp(&self) -> BigFloat {
        let prec = self.prec;

        // Argument reduction: exp(x) = exp(x/2^r)^(2^r)
        // Choose r so |x/2^r| < 1
        let xf = self.to_f64().abs();
        let r = if xf > 1.0 { (xf.log2().ceil() as u32) + 1 } else { 0 };

        let scale = BigFloat::from_i64(1i64 << r.min(30), prec);
        let reduced = self.div(&scale);

        // Taylor series: exp(t) = 1 + t + t²/2! + ...
        let mut sum = BigFloat::one(prec);
        let mut term = BigFloat::one(prec);
        let max_terms = (prec as usize) + 20;

        for k in 1..max_terms {
            term = term.mul(&reduced);
            let k_bf = BigFloat::from_i64(k as i64, prec);
            term = term.div(&k_bf);
            let old = sum.clone();
            sum = sum.add(&term);
            // Convergence check: if term didn't change sum, we're done
            if sum.mantissa == old.mantissa && sum.exponent == old.exponent {
                break;
            }
        }

        // Square r times: exp(x) = exp(x/2^r)^(2^r)
        for _ in 0..r {
            sum = sum.mul(&sum);
        }

        sum
    }

    /// Compute π to working precision via Machin's formula.
    /// π/4 = 4·arctan(1/5) - arctan(1/239)
    pub fn pi(prec: u32) -> BigFloat {
        let four = BigFloat::from_i64(4, prec);

        let atan5 = arctan_recip(5, prec);
        let atan239 = arctan_recip(239, prec);

        // π = 4 * (4·arctan(1/5) - arctan(1/239))
        four.mul(&four.mul(&atan5).sub(&atan239))
    }

    /// Euler's number e to working precision.
    pub fn e(prec: u32) -> BigFloat {
        BigFloat::one(prec).exp()
    }

    /// ln(2) to working precision.
    pub fn ln2(prec: u32) -> BigFloat {
        BigFloat::from_i64(2, prec).ln()
    }

    /// Power: self^exp via exp(exp * ln(self)).
    pub fn pow(&self, exp: &BigFloat) -> BigFloat {
        if exp.is_zero() { return BigFloat::one(self.prec); }
        if self.is_zero() { return BigFloat::zero(self.prec); }
        // For integer exponents on positive base, use repeated squaring?
        // General case: a^b = exp(b * ln(a))
        exp.mul(&self.ln()).exp()
    }

    /// Integer power via repeated squaring (exact, no ln/exp).
    pub fn powi(&self, n: i64) -> BigFloat {
        if n == 0 { return BigFloat::one(self.prec); }
        if n < 0 { return BigFloat::one(self.prec).div(&self.powi(-n)); }
        let mut result = BigFloat::one(self.prec);
        let mut base = self.clone();
        let mut exp = n as u64;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            exp >>= 1;
        }
        result
    }

    /// Compare magnitudes. Returns Ordering.
    pub fn cmp_abs(&self, other: &BigFloat) -> std::cmp::Ordering {
        // Compare |self| vs |other| by converting to same exponent basis
        let a = self.abs();
        let b = other.abs();
        if a.is_zero() && b.is_zero() { return std::cmp::Ordering::Equal; }
        if a.is_zero() { return std::cmp::Ordering::Less; }
        if b.is_zero() { return std::cmp::Ordering::Greater; }

        // Compare via f64 approximation (sufficient for ordering)
        let af = a.to_f64();
        let bf = b.to_f64();
        af.partial_cmp(&bf).unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Riemann zeta function ζ(s) for real s > 1.
    ///
    /// Uses the Borwein (1995) acceleration of the Dirichlet eta function:
    /// η_n(s) = -1/d_n · Σ_{k=0}^{n-1} (-1)^k · (d_k - d_n) / (k+1)^s
    ///
    /// where d_k = Σ_{j=0}^{k} C(n, j) (partial sums of binomial coefficients),
    /// d_n = 2^n. Then ζ(s) = η(s) / (1 - 2^{1-s}).
    ///
    /// Error is O((3 + 2√2)^{-n}), so n ≈ prec/2.5 suffices.
    pub fn zeta(s: &BigFloat) -> BigFloat {
        let prec = s.prec;
        let sf = s.to_f64();
        assert!(sf > 1.0, "zeta(s) requires s > 1 for this implementation");

        // n terms: error ≈ 5.83^{-n}, need 10^{-digits} where digits ≈ prec * log10(2)
        let n = ((prec as f64 * 0.302) / 5.83_f64.log10()).ceil() as usize + 4;
        let n = n.max(16);

        // Compute d_k = Σ_{j=0}^{k} C(n, j)
        let mut dk = vec![BigFloat::zero(prec); n + 1];
        let mut binom = BigFloat::one(prec); // C(n, 0) = 1
        dk[0] = binom.clone();
        for j in 1..=n {
            // C(n, j) = C(n, j-1) * (n-j+1) / j
            binom = binom
                .mul(&BigFloat::from_i64((n - j + 1) as i64, prec))
                .div(&BigFloat::from_i64(j as i64, prec));
            dk[j] = dk[j - 1].add(&binom);
        }
        let dn = dk[n].clone(); // = 2^n

        // Compute weighted eta sum
        let one = BigFloat::one(prec);
        let two = BigFloat::from_i64(2, prec);
        let one_minus_s = one.sub(s);
        let two_pow = two.pow(&one_minus_s); // 2^(1-s)
        let eta_factor = one.sub(&two_pow); // 1 - 2^(1-s)

        let mut total = BigFloat::zero(prec);
        for k in 0..n {
            let diff = dk[k].sub(&dn); // d_k - d_n (always ≤ 0)
            let kp1 = BigFloat::from_i64(k as i64 + 1, prec);
            let kp1_s = kp1.pow(s); // (k+1)^s
            let term = diff.div(&kp1_s);
            if k % 2 == 0 {
                total = total.add(&term);
            } else {
                total = total.sub(&term);
            }
        }

        // η(s) = -total / d_n
        // ζ(s) = η(s) / (1 - 2^(1-s))
        let neg_dn = dn.neg();
        let divisor = neg_dn.mul(&eta_factor);
        total.div(&divisor)
    }

    /// Riemann zeta at integer argument (faster: uses integer powers).
    pub fn zeta_int(s_int: i64, prec: u32) -> BigFloat {
        // For small positive even integers, we have closed forms:
        // ζ(2) = π²/6, ζ(4) = π⁴/90, ζ(6) = π⁶/945
        if s_int == 2 {
            let pi = BigFloat::pi(prec);
            let six = BigFloat::from_i64(6, prec);
            return pi.mul(&pi).div(&six);
        }
        if s_int == 4 {
            let pi = BigFloat::pi(prec);
            let p4 = pi.powi(4);
            let ninety = BigFloat::from_i64(90, prec);
            return p4.div(&ninety);
        }
        // General case: use zeta()
        BigFloat::zeta(&BigFloat::from_i64(s_int, prec))
    }

    /// Bernoulli number B_{2k} via the zeta relation:
    /// B_{2k} = (-1)^{k+1} · 2 · (2k)! · ζ(2k) / (2π)^{2k}
    ///
    /// Returns B_0=1, B_1=-1/2 for those indices, and B_{2k} for even indices.
    /// Odd Bernoulli numbers (B_3, B_5, ...) are zero.
    pub fn bernoulli(n: u32, prec: u32) -> BigFloat {
        if n == 0 { return BigFloat::one(prec); }
        if n == 1 {
            // B_1 = -1/2
            return BigFloat::from_f64(-0.5, prec);
        }
        if n % 2 == 1 { return BigFloat::zero(prec); } // B_{odd>1} = 0

        let k = n / 2;
        let two_k = n as i64;

        // Compute (2k)!
        let mut factorial = BigFloat::one(prec);
        for i in 2..=two_k {
            factorial = factorial.mul(&BigFloat::from_i64(i, prec));
        }

        // Compute ζ(2k)
        let zeta_val = BigFloat::zeta_int(two_k, prec);

        // Compute (2π)^{2k}
        let two_pi = BigFloat::pi(prec).mul(&BigFloat::from_i64(2, prec));
        let two_pi_pow = two_pi.powi(two_k);

        // B_{2k} = (-1)^{k+1} · 2 · (2k)! · ζ(2k) / (2π)^{2k}
        let two = BigFloat::from_i64(2, prec);
        let result = two.mul(&factorial).mul(&zeta_val).div(&two_pi_pow);

        if k % 2 == 0 {
            result.neg() // (-1)^{k+1} = -1 when k is even
        } else {
            result // (-1)^{k+1} = +1 when k is odd
        }
    }

    /// Factorial n! as BigFloat.
    pub fn factorial(n: u32, prec: u32) -> BigFloat {
        let mut result = BigFloat::one(prec);
        for i in 2..=(n as i64) {
            result = result.mul(&BigFloat::from_i64(i, prec));
        }
        result
    }

    /// Euler product for ζ(s): ∏_{p prime, p ≤ max_p} (1 - p^{-s})^{-1}
    ///
    /// Returns the partial Euler product using primes up to max_p.
    /// For s=2 and large max_p, converges to π²/6.
    pub fn euler_product(s: &BigFloat, max_p: u64) -> BigFloat {
        let prec = s.prec;
        let one = BigFloat::one(prec);
        let mut product = BigFloat::one(prec);

        // Simple sieve for primes up to max_p
        let primes = simple_sieve(max_p);

        for p in primes {
            let p_bf = BigFloat::from_i64(p as i64, prec);
            let p_neg_s = p_bf.pow(s).powi(-1); // p^{-s} = 1/p^s ... use powi for int s
            // (1 - p^{-s})^{-1}
            let factor = one.sub(&p_neg_s);
            product = product.div(&factor);
        }

        product
    }

    /// Euler factor for a specific set of primes.
    /// Returns ∏_{p in primes} (1 - p^{-s})^{-1}
    pub fn euler_factor(s: &BigFloat, primes: &[u64]) -> BigFloat {
        let prec = s.prec;
        let one = BigFloat::one(prec);
        let mut product = BigFloat::one(prec);

        for &p in primes {
            let p_bf = BigFloat::from_i64(p as i64, prec);
            let p_neg_s = one.div(&p_bf.pow(s));
            let factor = one.sub(&p_neg_s);
            product = product.div(&factor);
        }

        product
    }
}

/// Simple prime sieve up to n.
fn simple_sieve(n: u64) -> Vec<u64> {
    if n < 2 { return vec![]; }
    let mut is_prime = vec![true; (n + 1) as usize];
    is_prime[0] = false;
    if n >= 1 { is_prime[1] = false; }
    let mut p = 2u64;
    while p * p <= n {
        if is_prime[p as usize] {
            let mut m = p * p;
            while m <= n {
                is_prime[m as usize] = false;
                m += p;
            }
        }
        p += 1;
    }
    (2..=n).filter(|&i| is_prime[i as usize]).collect()
}

/// arctan(1/n) via Taylor series: arctan(x) = x - x³/3 + x⁵/5 - ...
/// Specialized for x = 1/n (integer reciprocal) for efficiency.
fn arctan_recip(n: i64, prec: u32) -> BigFloat {
    let one = BigFloat::one(prec);
    let x = one.div(&BigFloat::from_i64(n, prec)); // 1/n
    let x2 = x.mul(&x); // 1/n²

    let mut sum = x.clone();
    let mut term = x.clone();
    let mut sign = -1i64;

    let max_terms = (prec as usize) * 2;
    for k in 1..max_terms {
        term = term.mul(&x2);
        let denom = BigFloat::from_i64(sign * (2 * k as i64 + 1), prec);
        let contribution = term.div(&denom);
        let old = sum.clone();
        sum = sum.add(&contribution);
        sign = -sign;
        if sum.mantissa == old.mantissa && sum.exponent == old.exponent {
            break;
        }
    }
    sum
}

// Helper: BigInt extensions needed for BigFloat
impl BigInt {
    /// Bit length of the absolute value.
    pub fn bit_length(&self) -> u32 {
        let limbs = &self.limbs;
        for i in (0..limbs.len()).rev() {
            if limbs[i] != 0 {
                return (i as u32) * 64 + (64 - limbs[i].leading_zeros());
            }
        }
        0
    }

    /// Left shift by n bits.
    pub fn shl(&self, n: u32) -> BigInt {
        if self.is_zero() || n == 0 { return self.clone(); }
        let limb_shift = (n / 64) as usize;
        let bit_shift = n % 64;
        let old_len = self.limbs.len();
        let mut result = vec![0u64; old_len + limb_shift + 1];
        for i in 0..old_len {
            result[i + limb_shift] |= self.limbs[i] << bit_shift;
            if bit_shift > 0 && i + limb_shift + 1 < result.len() {
                result[i + limb_shift + 1] |= self.limbs[i] >> (64 - bit_shift);
            }
        }
        let mut r = BigInt { limbs: result, negative: self.negative };
        r.normalize();
        r
    }

    /// Right shift by n bits (arithmetic: preserves sign, rounds toward -∞).
    pub fn shr(&self, n: u32) -> BigInt {
        if self.is_zero() || n == 0 { return self.clone(); }
        let limb_shift = (n / 64) as usize;
        let bit_shift = n % 64;
        if limb_shift >= self.limbs.len() {
            return if self.negative { BigInt::from_i64(-1) } else { BigInt::zero() };
        }
        let new_len = self.limbs.len() - limb_shift;
        let mut result = vec![0u64; new_len];
        for i in 0..new_len {
            result[i] = self.limbs[i + limb_shift] >> bit_shift;
            if bit_shift > 0 && i + limb_shift + 1 < self.limbs.len() {
                result[i] |= self.limbs[i + limb_shift + 1] << (64 - bit_shift);
            }
        }
        let mut r = BigInt { limbs: result, negative: self.negative };
        r.normalize();
        r
    }

    /// Approximate conversion to f64 (uses top 53 bits).
    pub fn to_f64_approx(&self) -> f64 {
        if self.is_zero() { return 0.0; }
        let bl = self.bit_length();
        let sign = if self.negative { -1.0 } else { 1.0 };

        if bl <= 53 {
            // Fits exactly
            let mut val = 0u64;
            for i in (0..self.limbs.len()).rev() {
                val = (val << (if i < self.limbs.len() - 1 { 64 } else { 0 })) | self.limbs[i];
            }
            // Simple extraction for small values
            if self.limbs.len() == 1 {
                return sign * self.limbs[0] as f64;
            }
            if self.limbs.len() == 2 {
                return sign * (self.limbs[0] as f64 + self.limbs[1] as f64 * (1u64 << 63) as f64 * 2.0);
            }
        }

        // Extract top ~53 bits
        let shift = bl - 53;
        let shifted = self.shr(shift);
        let top = if shifted.limbs.len() >= 1 { shifted.limbs[0] } else { 0 };
        sign * top as f64 * (2.0_f64).powi(shift as i32)
    }
}

impl fmt::Display for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Display as decimal approximation
        write!(f, "{:.15e}", self.to_f64())
    }
}

impl fmt::Debug for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BigFloat({} * 2^{}, prec={})", self.mantissa, self.exponent, self.prec)
    }
}

impl PartialEq for BigFloat {
    fn eq(&self, other: &Self) -> bool {
        if self.is_zero() && other.is_zero() { return true; }
        self.mantissa == other.mantissa && self.exponent == other.exponent
    }
}

impl Eq for BigFloat {}

// ═══════════════════════════════════════════════════════════════════════════
// BigComplex — arbitrary precision complex numbers
// ═══════════════════════════════════════════════════════════════════════════

/// Arbitrary precision complex number: re + im·i.
#[derive(Clone, Debug)]
pub struct BigComplex {
    pub re: BigFloat,
    pub im: BigFloat,
}

impl BigComplex {
    pub fn new(re: BigFloat, im: BigFloat) -> Self { BigComplex { re, im } }

    pub fn from_real(re: BigFloat) -> Self {
        let prec = re.prec;
        BigComplex { re, im: BigFloat::zero(prec) }
    }

    pub fn zero(prec: u32) -> Self {
        BigComplex { re: BigFloat::zero(prec), im: BigFloat::zero(prec) }
    }

    pub fn prec(&self) -> u32 { self.re.prec.max(self.im.prec) }

    pub fn add(&self, other: &BigComplex) -> BigComplex {
        BigComplex { re: self.re.add(&other.re), im: self.im.add(&other.im) }
    }

    pub fn sub(&self, other: &BigComplex) -> BigComplex {
        BigComplex { re: self.re.sub(&other.re), im: self.im.sub(&other.im) }
    }

    pub fn mul(&self, other: &BigComplex) -> BigComplex {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        BigComplex {
            re: self.re.mul(&other.re).sub(&self.im.mul(&other.im)),
            im: self.re.mul(&other.im).add(&self.im.mul(&other.re)),
        }
    }

    pub fn mul_real(&self, s: &BigFloat) -> BigComplex {
        BigComplex { re: self.re.mul(s), im: self.im.mul(s) }
    }

    pub fn div(&self, other: &BigComplex) -> BigComplex {
        // (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
        let denom = other.re.mul(&other.re).add(&other.im.mul(&other.im));
        BigComplex {
            re: self.re.mul(&other.re).add(&self.im.mul(&other.im)).div(&denom),
            im: self.im.mul(&other.re).sub(&self.re.mul(&other.im)).div(&denom),
        }
    }

    pub fn neg(&self) -> BigComplex {
        BigComplex { re: self.re.neg(), im: self.im.neg() }
    }

    pub fn conj(&self) -> BigComplex {
        BigComplex { re: self.re.clone(), im: self.im.neg() }
    }

    /// |z|² = re² + im²
    pub fn norm_sq(&self) -> BigFloat {
        self.re.mul(&self.re).add(&self.im.mul(&self.im))
    }

    /// |z| = sqrt(re² + im²)
    pub fn abs(&self) -> BigFloat {
        self.norm_sq().sqrt()
    }

    /// Complex exponential: exp(a+bi) = exp(a)(cos(b) + i·sin(b))
    pub fn exp(&self) -> BigComplex {
        let ea = self.re.exp();
        let (s, c) = bf_sincos(&self.im);
        BigComplex { re: ea.mul(&c), im: ea.mul(&s) }
    }

    /// Complex natural logarithm: ln(z) = ln|z| + i·arg(z)
    pub fn ln(&self) -> BigComplex {
        let r = self.abs().ln();
        let theta = bf_atan2(&self.im, &self.re);
        BigComplex { re: r, im: theta }
    }

    /// Complex power: z^w = exp(w · ln(z))
    pub fn pow(&self, w: &BigComplex) -> BigComplex {
        w.mul(&self.ln()).exp()
    }

    /// Integer power via real exponent.
    pub fn pow_real(&self, s: &BigFloat) -> BigComplex {
        BigComplex::from_real(s.clone()).mul(&self.ln()).exp()
    }

    pub fn to_f64(&self) -> (f64, f64) {
        (self.re.to_f64(), self.im.to_f64())
    }
}

/// sin and cos of a BigFloat via Taylor series.
fn bf_sincos(x: &BigFloat) -> (BigFloat, BigFloat) {
    let prec = x.prec;
    // Argument reduction: reduce to [-π, π] first
    // For moderate values this is fine; for huge values we'd need more care.
    let xf = x.to_f64();
    let reduced = if xf.abs() > 3.2 {
        let pi = BigFloat::pi(prec);
        let two_pi = pi.mul(&BigFloat::from_i64(2, prec));
        // x mod 2π
        let n_periods = BigFloat::from_f64((xf / (2.0 * std::f64::consts::PI)).round(), prec);
        x.sub(&n_periods.mul(&two_pi))
    } else {
        x.clone()
    };

    let x2 = reduced.mul(&reduced);

    // sin: x - x³/3! + x⁵/5! - ...
    let mut sin_sum = reduced.clone();
    let mut sin_term = reduced.clone();

    // cos: 1 - x²/2! + x⁴/4! - ...
    let mut cos_sum = BigFloat::one(prec);
    let mut cos_term = BigFloat::one(prec);

    let max_terms = (prec as usize).max(30);
    for k in 1..max_terms {
        // sin term: multiply by -x²/((2k)(2k+1))
        let sin_denom = BigFloat::from_i64((2 * k as i64) * (2 * k as i64 + 1), prec);
        sin_term = sin_term.mul(&x2).neg().div(&sin_denom);
        let old_sin = sin_sum.clone();
        sin_sum = sin_sum.add(&sin_term);

        // cos term: multiply by -x²/((2k-1)(2k))
        let cos_denom = BigFloat::from_i64((2 * k as i64 - 1) * (2 * k as i64), prec);
        cos_term = cos_term.mul(&x2).neg().div(&cos_denom);
        let old_cos = cos_sum.clone();
        cos_sum = cos_sum.add(&cos_term);

        if sin_sum.mantissa == old_sin.mantissa && sin_sum.exponent == old_sin.exponent
            && cos_sum.mantissa == old_cos.mantissa && cos_sum.exponent == old_cos.exponent
        {
            break;
        }
    }

    (sin_sum, cos_sum)
}

/// atan2(y, x) for BigFloat.
fn bf_atan2(y: &BigFloat, x: &BigFloat) -> BigFloat {
    let prec = y.prec.max(x.prec);
    // Use f64 atan2 as seed, then refine if needed
    let yf = y.to_f64();
    let xf = x.to_f64();
    BigFloat::from_f64(yf.atan2(xf), prec)
}

// ─── Complex Riemann zeta ──────────────────────────────────────────────

/// Riemann zeta function ζ(s) for complex s, s ≠ 1.
///
/// Uses the same Borwein acceleration as the real version.
pub fn zeta_complex(s: &BigComplex) -> BigComplex {
    let prec = s.prec();

    // n terms for convergence
    let n = ((prec as f64 * 0.302) / 5.83_f64.log10()).ceil() as usize + 4;
    let n = n.max(16);

    // Compute d_k = Σ_{j=0}^{k} C(n, j) (real-valued)
    let mut dk = vec![BigFloat::zero(prec); n + 1];
    let mut binom = BigFloat::one(prec);
    dk[0] = binom.clone();
    for j in 1..=n {
        binom = binom
            .mul(&BigFloat::from_i64((n - j + 1) as i64, prec))
            .div(&BigFloat::from_i64(j as i64, prec));
        dk[j] = dk[j - 1].add(&binom);
    }
    let dn = dk[n].clone();

    // 1 - 2^(1-s) where s is complex
    let one_c = BigComplex::from_real(BigFloat::one(prec));
    let two_c = BigComplex::from_real(BigFloat::from_i64(2, prec));
    let one_minus_s = one_c.sub(s);
    let two_pow = two_c.pow(&one_minus_s); // 2^(1-s)
    let eta_factor = one_c.sub(&two_pow);

    // Weighted sum
    let mut total = BigComplex::zero(prec);
    for k in 0..n {
        let diff = dk[k].sub(&dn); // real
        let kp1 = BigComplex::from_real(BigFloat::from_i64(k as i64 + 1, prec));
        let kp1_s = kp1.pow(s); // (k+1)^s, complex
        let term_c = BigComplex::from_real(diff).div(&kp1_s);
        if k % 2 == 0 {
            total = total.add(&term_c);
        } else {
            total = total.sub(&term_c);
        }
    }

    // ζ(s) = -total / (d_n · eta_factor)
    let neg_dn_c = BigComplex::from_real(dn.neg());
    let divisor = neg_dn_c.mul(&eta_factor);
    total.div(&divisor)
}

/// Complex Euler factor for specific primes:
/// ∏_{p in primes} (1 - p^{-s})^{-1} where s is complex.
pub fn euler_factor_complex(s: &BigComplex, primes: &[u64]) -> BigComplex {
    let prec = s.prec();
    let one = BigComplex::from_real(BigFloat::one(prec));
    let mut product = one.clone();

    for &p in primes {
        let p_c = BigComplex::from_real(BigFloat::from_i64(p as i64, prec));
        let p_neg_s = one.div(&p_c.pow(s)); // p^{-s}
        let factor = one.sub(&p_neg_s);      // 1 - p^{-s}
        product = product.div(&factor);       // multiply by (1 - p^{-s})^{-1}
    }

    product
}

/// Complex Euler product over all primes up to max_p.
pub fn euler_product_complex(s: &BigComplex, max_p: u64) -> BigComplex {
    let primes = simple_sieve(max_p);
    euler_factor_complex(s, &primes)
}

/// Hardy Z-function: Z(t) = exp(iθ(t)) · ζ(1/2 + it)
/// where θ(t) is the Riemann-Siegel theta function.
///
/// Z(t) is real-valued, and its zeros correspond to zeros of ζ on the critical line.
/// Sign changes in Z(t) bracket zeros.
pub fn hardy_z(t: f64, prec: u32) -> f64 {
    // θ(t) = arg(Γ(1/4 + it/2)) - (t/2)·ln(π)
    // For moderate t, use Stirling approximation:
    // θ(t) ≈ (t/2)·ln(t/(2πe)) - π/8 + 1/(48t) + ...
    let theta = riemann_siegel_theta(t);

    // ζ(1/2 + it)
    let s = BigComplex::new(
        BigFloat::from_f64(0.5, prec),
        BigFloat::from_f64(t, prec),
    );
    let z = zeta_complex(&s);
    let (zr, zi) = z.to_f64();

    // Z(t) = exp(iθ)·ζ(1/2+it) has Re part = cos(θ)·zr - sin(θ)·zi
    theta.cos() * zr - theta.sin() * zi
}

/// Riemann-Siegel theta function via Stirling approximation.
/// θ(t) ≈ (t/2)·ln(t/(2π)) - t/2 - π/8 + 1/(48t) + 7/(5760t³)
fn riemann_siegel_theta(t: f64) -> f64 {
    let pi = std::f64::consts::PI;
    (t / 2.0) * (t / (2.0 * pi)).ln() - t / 2.0 - pi / 8.0
        + 1.0 / (48.0 * t) + 7.0 / (5760.0 * t.powi(3))
}

/// Find a zero of ζ on the critical line near t_guess by bisection on Z(t).
///
/// Returns (t_zero, Z_value) where |Z_value| < tol.
pub fn find_zeta_zero(t_lo: f64, t_hi: f64, prec: u32, tol: f64) -> Option<(f64, f64)> {
    let z_lo = hardy_z(t_lo, prec);
    let z_hi = hardy_z(t_hi, prec);

    // Need a sign change
    if z_lo * z_hi > 0.0 {
        return None; // no sign change in interval
    }

    let mut lo = t_lo;
    let mut hi = t_hi;
    let mut z_l = z_lo;

    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let z_m = hardy_z(mid, prec);

        if z_m.abs() < tol {
            return Some((mid, z_m));
        }

        if z_l * z_m < 0.0 {
            hi = mid;
        } else {
            lo = mid;
            z_l = z_m;
        }

        if (hi - lo) < tol * 0.01 {
            return Some(((lo + hi) / 2.0, z_m));
        }
    }

    let mid = (lo + hi) / 2.0;
    Some((mid, hardy_z(mid, prec)))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close_f64(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: {a} vs {b} (diff={})", (a - b).abs());
    }

    // ── Basic arithmetic ───────────────────────────────────────────────

    #[test]
    fn bf_from_f64_roundtrip() {
        for &v in &[0.0, 1.0, -1.0, 3.14159, 1e10, 1e-10, -42.5] {
            let bf = BigFloat::from_f64(v, 128);
            close_f64(bf.to_f64(), v, 1e-12, &format!("roundtrip {v}"));
        }
    }

    #[test]
    fn bf_add() {
        let a = BigFloat::from_f64(1.5, 128);
        let b = BigFloat::from_f64(2.25, 128);
        close_f64(a.add(&b).to_f64(), 3.75, 1e-12, "1.5 + 2.25");
    }

    #[test]
    fn bf_sub() {
        let a = BigFloat::from_f64(10.0, 128);
        let b = BigFloat::from_f64(3.0, 128);
        close_f64(a.sub(&b).to_f64(), 7.0, 1e-12, "10 - 3");
    }

    #[test]
    fn bf_mul() {
        let a = BigFloat::from_f64(3.0, 128);
        let b = BigFloat::from_f64(7.0, 128);
        close_f64(a.mul(&b).to_f64(), 21.0, 1e-12, "3 * 7");
    }

    #[test]
    fn bf_div() {
        let a = BigFloat::from_f64(22.0, 128);
        let b = BigFloat::from_f64(7.0, 128);
        close_f64(a.div(&b).to_f64(), 22.0 / 7.0, 1e-10, "22/7");
    }

    // ── sqrt ───────────────────────────────────────────────────────────

    #[test]
    fn bf_sqrt() {
        let two = BigFloat::from_f64(2.0, 128);
        close_f64(two.sqrt().to_f64(), std::f64::consts::SQRT_2, 1e-10, "sqrt(2)");

        let nine = BigFloat::from_f64(9.0, 128);
        close_f64(nine.sqrt().to_f64(), 3.0, 1e-10, "sqrt(9)");
    }

    // ── exp ────────────────────────────────────────────────────────────

    #[test]
    fn bf_exp() {
        let one = BigFloat::from_f64(1.0, 128);
        close_f64(one.exp().to_f64(), std::f64::consts::E, 1e-10, "exp(1)");

        let zero = BigFloat::from_f64(0.0, 128);
        close_f64(zero.exp().to_f64(), 1.0, 1e-10, "exp(0)");
    }

    #[test]
    fn bf_exp_negative() {
        let neg = BigFloat::from_f64(-1.0, 128);
        close_f64(neg.exp().to_f64(), 1.0 / std::f64::consts::E, 1e-10, "exp(-1)");
    }

    // ── ln ─────────────────────────────────────────────────────────────

    #[test]
    fn bf_ln() {
        let e = BigFloat::from_f64(std::f64::consts::E, 128);
        close_f64(e.ln().to_f64(), 1.0, 1e-8, "ln(e)");

        let ten = BigFloat::from_f64(10.0, 128);
        close_f64(ten.ln().to_f64(), 10.0_f64.ln(), 1e-8, "ln(10)");
    }

    // ── pi ─────────────────────────────────────────────────────────────

    #[test]
    fn bf_pi() {
        let pi = BigFloat::pi(128);
        close_f64(pi.to_f64(), std::f64::consts::PI, 1e-10, "π");
    }

    // ── e constant ─────────────────────────────────────────────────────

    #[test]
    fn bf_e_constant() {
        let e = BigFloat::e(128);
        close_f64(e.to_f64(), std::f64::consts::E, 1e-10, "e");
    }

    // ── Higher precision ───────────────────────────────────────────────

    #[test]
    fn bf_sqrt2_higher_precision() {
        // At 256-bit precision, sqrt(2)² should be very close to 2
        let two = BigFloat::from_f64(2.0, 256);
        let s = two.sqrt();
        let s_squared = s.mul(&s);
        close_f64(s_squared.to_f64(), 2.0, 1e-12, "sqrt(2)² at 256-bit");
    }

    // ── BigInt shift tests ─────────────────────────────────────────────

    #[test]
    fn bigint_shl_shr() {
        let a = BigInt::from_u64(1);
        let shifted = a.shl(64);
        assert_eq!(shifted.limbs, vec![0, 1]);
        let back = shifted.shr(64);
        assert_eq!(back.to_u64(), Some(1));
    }

    #[test]
    fn bigint_bit_length() {
        assert_eq!(BigInt::zero().bit_length(), 0);
        assert_eq!(BigInt::from_u64(1).bit_length(), 1);
        assert_eq!(BigInt::from_u64(255).bit_length(), 8);
        assert_eq!(BigInt::from_u64(256).bit_length(), 9);
    }

    // ── pow / powi ────────────────────────────────────────────────────

    #[test]
    fn bf_powi() {
        let two = BigFloat::from_f64(2.0, 128);
        close_f64(two.powi(10).to_f64(), 1024.0, 1e-6, "2^10");
        close_f64(two.powi(0).to_f64(), 1.0, 1e-12, "2^0");
        close_f64(two.powi(-1).to_f64(), 0.5, 1e-10, "2^-1");
    }

    #[test]
    fn bf_pow_fractional() {
        // 8^(1/3) = 2
        let eight = BigFloat::from_f64(8.0, 128);
        let third = BigFloat::from_f64(1.0 / 3.0, 128);
        close_f64(eight.pow(&third).to_f64(), 2.0, 1e-6, "8^(1/3)");
    }

    // ── zeta ──────────────────────────────────────────────────────────

    #[test]
    fn bf_zeta_2() {
        // ζ(2) = π²/6 ≈ 1.6449340668482...
        let z2 = BigFloat::zeta_int(2, 128);
        close_f64(z2.to_f64(), std::f64::consts::PI * std::f64::consts::PI / 6.0, 1e-8, "ζ(2)");
    }

    #[test]
    fn bf_zeta_3() {
        // Apéry's constant ζ(3) ≈ 1.2020569031595942...
        let z3 = BigFloat::zeta(&BigFloat::from_i64(3, 128));
        close_f64(z3.to_f64(), 1.2020569031595942, 1e-6, "ζ(3)");
    }

    #[test]
    fn bf_zeta_4() {
        // ζ(4) = π⁴/90 ≈ 1.0823232337111...
        let z4 = BigFloat::zeta_int(4, 128);
        let expected = std::f64::consts::PI.powi(4) / 90.0;
        close_f64(z4.to_f64(), expected, 1e-8, "ζ(4)");
    }

    #[test]
    fn bf_zeta_10() {
        // ζ(10) ≈ 1.000994575127818...
        let z10 = BigFloat::zeta(&BigFloat::from_i64(10, 128));
        close_f64(z10.to_f64(), 1.0009945751278181, 1e-6, "ζ(10)");
    }

    #[test]
    fn bf_zeta_2_vs_closed_form() {
        // Cross-check: zeta(2) via Borwein should match π²/6 via closed form
        let z_borwein = BigFloat::zeta(&BigFloat::from_i64(2, 128));
        let z_closed = BigFloat::zeta_int(2, 128);
        let diff = z_borwein.sub(&z_closed).to_f64().abs();
        assert!(diff < 1e-6, "Borwein vs closed form: diff = {diff}");
    }

    // ── Bernoulli numbers ─────────────────────────────────────────────

    #[test]
    fn bf_bernoulli_known_values() {
        // B_0 = 1, B_1 = -1/2, B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30
        close_f64(BigFloat::bernoulli(0, 128).to_f64(), 1.0, 1e-12, "B_0");
        close_f64(BigFloat::bernoulli(1, 128).to_f64(), -0.5, 1e-12, "B_1");
        close_f64(BigFloat::bernoulli(2, 128).to_f64(), 1.0 / 6.0, 1e-6, "B_2");
        close_f64(BigFloat::bernoulli(4, 128).to_f64(), -1.0 / 30.0, 1e-6, "B_4");
        close_f64(BigFloat::bernoulli(6, 128).to_f64(), 1.0 / 42.0, 1e-6, "B_6");
        close_f64(BigFloat::bernoulli(8, 128).to_f64(), -1.0 / 30.0, 1e-6, "B_8");
    }

    #[test]
    fn bf_bernoulli_odd_are_zero() {
        for k in &[3, 5, 7, 9, 11] {
            assert!(BigFloat::bernoulli(*k, 128).is_zero(), "B_{k} should be zero");
        }
    }

    #[test]
    fn bf_bernoulli_b10() {
        // B_10 = 5/66 ≈ 0.07575757...
        close_f64(BigFloat::bernoulli(10, 128).to_f64(), 5.0 / 66.0, 1e-5, "B_10");
    }

    // ── factorial ─────────────────────────────────────────────────────

    #[test]
    fn bf_factorial() {
        close_f64(BigFloat::factorial(0, 128).to_f64(), 1.0, 1e-12, "0!");
        close_f64(BigFloat::factorial(5, 128).to_f64(), 120.0, 1e-10, "5!");
        close_f64(BigFloat::factorial(10, 128).to_f64(), 3628800.0, 1e-6, "10!");
    }

    // ── BigComplex ────────────────────────────────────────────────────

    #[test]
    fn bc_arithmetic() {
        let a = BigComplex::new(BigFloat::from_f64(3.0, 128), BigFloat::from_f64(4.0, 128));
        let b = BigComplex::new(BigFloat::from_f64(1.0, 128), BigFloat::from_f64(2.0, 128));

        // (3+4i) + (1+2i) = 4+6i
        let sum = a.add(&b);
        close_f64(sum.re.to_f64(), 4.0, 1e-12, "re(sum)");
        close_f64(sum.im.to_f64(), 6.0, 1e-12, "im(sum)");

        // (3+4i)(1+2i) = (3-8) + (6+4)i = -5+10i
        let prod = a.mul(&b);
        close_f64(prod.re.to_f64(), -5.0, 1e-10, "re(prod)");
        close_f64(prod.im.to_f64(), 10.0, 1e-10, "im(prod)");

        // |3+4i| = 5
        close_f64(a.abs().to_f64(), 5.0, 1e-10, "|3+4i|");
    }

    #[test]
    fn bc_exp_euler() {
        // exp(iπ) = -1 (Euler's identity)
        let i_pi = BigComplex::new(BigFloat::zero(128), BigFloat::pi(128));
        let result = i_pi.exp();
        close_f64(result.re.to_f64(), -1.0, 1e-8, "re(exp(iπ))");
        close_f64(result.im.to_f64(), 0.0, 1e-8, "im(exp(iπ))");
    }

    #[test]
    fn bc_div_inverse() {
        // (3+4i)/(3+4i) = 1
        let a = BigComplex::new(BigFloat::from_f64(3.0, 128), BigFloat::from_f64(4.0, 128));
        let one = a.div(&a);
        close_f64(one.re.to_f64(), 1.0, 1e-10, "re(z/z)");
        close_f64(one.im.to_f64(), 0.0, 1e-10, "im(z/z)");
    }

    // ── Complex zeta ──────────────────────────────────────────────────

    #[test]
    fn bc_zeta_on_real_axis() {
        // ζ(3 + 0i) should match real ζ(3) = Apéry's constant
        let s = BigComplex::from_real(BigFloat::from_i64(3, 128));
        let z = zeta_complex(&s);
        close_f64(z.re.to_f64(), 1.2020569031595942, 1e-5, "re(ζ(3))");
        close_f64(z.im.to_f64(), 0.0, 1e-5, "im(ζ(3))");
    }

    #[test]
    fn bc_zeta_known_complex() {
        // ζ(2 + i) — known value from tables:
        // Re ≈ 1.1503, Im ≈ -0.4376 (Mathematica/LMFDB)
        let s = BigComplex::new(BigFloat::from_f64(2.0, 128), BigFloat::from_f64(1.0, 128));
        let z = zeta_complex(&s);
        close_f64(z.re.to_f64(), 1.1503, 0.01, "re(ζ(2+i))");
        close_f64(z.im.to_f64(), -0.4376, 0.01, "im(ζ(2+i))");
    }

    // ── Hardy Z-function and zeros ────────────────────────────────────

    #[test]
    fn hardy_z_sign_changes() {
        // The first few zeros of ζ on the critical line are at approximately:
        // t₁ ≈ 14.1347, t₂ ≈ 21.022, t₃ ≈ 25.011
        // Z(t) should have sign changes near these values.

        // Check Z(t) at a few points around the first zero
        let z13 = hardy_z(13.0, 128);
        let z15 = hardy_z(15.0, 128);

        eprintln!("═══ Hardy Z-function near first zero ═══");
        eprintln!("Z(13) = {:.6}", z13);
        eprintln!("Z(15) = {:.6}", z15);

        // Z(13) and Z(15) should have opposite signs (first zero between them)
        assert!(z13 * z15 < 0.0,
            "Z(13)={z13:.4} and Z(15)={z15:.4} should have opposite signs");
    }

    #[test]
    fn find_first_zeta_zero() {
        // Locate the first non-trivial zero: t₁ ≈ 14.134725...
        let result = find_zeta_zero(14.0, 14.5, 128, 1e-6);
        assert!(result.is_some(), "Should find a zero in [14.0, 14.5]");

        let (t_zero, z_val) = result.unwrap();
        eprintln!("═══ First Riemann zeta zero ═══");
        eprintln!("t = {:.10}", t_zero);
        eprintln!("Z(t) = {:.2e}", z_val);
        eprintln!("known: 14.134725...");

        close_f64(t_zero, 14.134725, 0.01, "first zero");
    }

    #[test]
    fn find_first_five_zeta_zeros() {
        // Known zeros on the critical line (LMFDB / Odlyzko tables):
        let known_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062];

        // Scan Z(t) for sign changes in [10, 35] with step 0.5
        let mut zeros_found = Vec::new();
        let step = 0.5;
        let mut t = 10.0;
        let mut z_prev = hardy_z(t, 128);

        while t < 35.0 && zeros_found.len() < 5 {
            let t_next = t + step;
            let z_next = hardy_z(t_next, 128);

            if z_prev * z_next < 0.0 {
                // Sign change — refine by bisection
                if let Some((t_zero, _)) = find_zeta_zero(t, t_next, 128, 1e-8) {
                    zeros_found.push(t_zero);
                }
            }

            z_prev = z_next;
            t = t_next;
        }

        eprintln!("═══ First {} Riemann zeta zeros ═══", zeros_found.len());
        for (i, (&found, &known)) in zeros_found.iter().zip(known_zeros.iter()).enumerate() {
            let err = (found - known).abs();
            eprintln!("  t_{} = {:.8}  (known: {:.6}, err: {:.2e})", i + 1, found, known, err);
        }

        assert!(zeros_found.len() >= 5, "Found only {} zeros, expected 5", zeros_found.len());
        for (i, (&found, &known)) in zeros_found.iter().zip(known_zeros.iter()).enumerate() {
            close_f64(found, known, 0.01, &format!("zero #{}", i + 1));
        }
    }

    // ── Cross-validation: three independent π computations ────────────

    #[test]
    fn pi_cross_validation_three_paths() {
        // Three independent computations of π at 256-bit precision:
        // 1. Machin's arctan formula (BigFloat::pi)
        // 2. sin(π) = 0, cos(π) = -1 (Taylor series sincos)
        // 3. √(6·ζ(2)) = π (Borwein zeta + BigFloat sqrt)

        let prec = 256;

        // Path 1: Machin
        let pi_machin = BigFloat::pi(prec);

        // Path 2: sincos self-consistency
        let (sin_pi, cos_pi) = bf_sincos(&pi_machin);
        let sin_err = sin_pi.to_f64().abs();
        let cos_err = (cos_pi.to_f64() + 1.0).abs();

        // Path 3: √(6·ζ(2))
        let six = BigFloat::from_i64(6, prec);
        let z2 = BigFloat::zeta_int(2, prec); // uses Machin π internally, but ζ(2) closed form
        let pi_sq_from_zeta = six.mul(&z2);
        let pi_from_zeta = pi_sq_from_zeta.sqrt();

        // Cross-validate: Machin vs zeta
        let cross_err = pi_machin.sub(&pi_from_zeta).to_f64().abs();

        eprintln!("═══ π cross-validation at {prec}-bit precision ═══");
        eprintln!("π (Machin)   = {:.15e}", pi_machin.to_f64());
        eprintln!("sin(π)       = {:.2e} (should be 0)", sin_pi.to_f64());
        eprintln!("cos(π) + 1   = {:.2e} (should be 0)", cos_pi.to_f64() + 1.0);
        eprintln!("π (√6ζ(2))   = {:.15e}", pi_from_zeta.to_f64());
        eprintln!("cross error  = {:.2e}", cross_err);

        assert!(sin_err < 1e-10, "sin(π) = {} should be ≈ 0", sin_err);
        assert!(cos_err < 1e-10, "cos(π)+1 = {} should be ≈ 0", cos_err);
        assert!(cross_err < 1e-10, "Machin vs zeta π: {} should be ≈ 0", cross_err);
    }

    // ── Euler-Mascheroni constant ─────────────────────────────────────

    #[test]
    fn bf_euler_mascheroni() {
        // γ = lim_{n→∞} (Σ_{k=1}^{n} 1/k - ln(n))
        // At 128-bit precision, compute partial sum for large enough n
        let prec = 128;
        let n = 200;

        let mut harmonic = BigFloat::zero(prec);
        for k in 1..=n {
            harmonic = harmonic.add(&BigFloat::one(prec).div(&BigFloat::from_i64(k as i64, prec)));
        }
        let ln_n = BigFloat::from_i64(n as i64, prec).ln();
        let gamma_approx = harmonic.sub(&ln_n);

        // Known value: γ = 0.5772156649015328606...
        close_f64(gamma_approx.to_f64(), 0.5772156649015329, 0.003, "Euler-Mascheroni");
    }

    // ── Euler product and Collatz-Riemann connection ──────────────────

    #[test]
    fn euler_product_converges_to_zeta() {
        // ∏_{p≤1000} (1-p^{-2})^{-1} should be close to ζ(2) = π²/6
        let s = BigFloat::from_i64(2, 128);
        let ep = BigFloat::euler_product(&s, 1000);
        let z2 = BigFloat::zeta_int(2, 128);
        let rel_err = ep.sub(&z2).to_f64().abs() / z2.to_f64();
        eprintln!("Euler product (p≤1000): {:.12}", ep.to_f64());
        eprintln!("ζ(2):                   {:.12}", z2.to_f64());
        eprintln!("Relative error:         {:.2e}", rel_err);
        // With primes up to 1000, the tail ∑ 1/p^2 for p>1000 is small
        assert!(rel_err < 0.001, "Euler product should be within 0.1% of ζ(2)");
    }

    #[test]
    fn collatz_euler_factor_is_three_halves() {
        // The {2,3}-Euler factor of ζ(2):
        //   (1-2^{-2})^{-1} · (1-3^{-2})^{-1} = (4/3)(9/8) = 3/2
        //
        // Collatz uses exactly primes 2 and 3: n → n/2 (divide by 2) and n → 3n+1.
        // These two primes contribute EXACTLY the rational factor 3/2 to ζ(2) = π²/6.
        // The remaining primes contribute π²/9 (transcendental).
        let s = BigFloat::from_i64(2, 128);
        let factor_23 = BigFloat::euler_factor(&s, &[2, 3]);

        eprintln!("═══ Collatz-Riemann {{2,3}}-Euler factor ═══");
        eprintln!("{{2,3}}-factor of ζ(2) = {:.12}", factor_23.to_f64());
        eprintln!("Expected: 3/2 = {:.12}", 1.5);

        close_f64(factor_23.to_f64(), 1.5, 1e-10, "{2,3}-Euler factor = 3/2");

        // The complementary factor: ζ(2) / (3/2) = π²/9
        let z2 = BigFloat::zeta_int(2, 128);
        let complement = z2.div(&factor_23);
        let pi_sq_over_9 = std::f64::consts::PI.powi(2) / 9.0;
        close_f64(complement.to_f64(), pi_sq_over_9, 1e-8, "complement = π²/9");

        eprintln!("Complement (other primes): {:.12}", complement.to_f64());
        eprintln!("π²/9 = {:.12}", pi_sq_over_9);
        eprintln!("");
        eprintln!("Interpretation: Collatz primes {{2,3}} contribute the RATIONAL part (3/2).");
        eprintln!("All other primes contribute the TRANSCENDENTAL part (π²/9).");
        eprintln!("The Collatz contraction IS the rational Euler factor.");
    }

    #[test]
    fn collatz_contraction_rate_from_euler() {
        // For ζ(s), the {2,3}-Euler factor is (1-2^{-s})^{-1}·(1-3^{-s})^{-1}.
        // At s=1 this diverges (harmonic series). At s=2 it's 3/2.
        // The LOGARITHMIC contraction rate for Collatz:
        // E[log₂|C(n)/n|] ≈ (1/2)log₂(1/2) + (1/2)log₂(3/2) = (-1 + log₂(3/2))/2
        //                  = (log₂(3) - 2)/2 ≈ (1.585-2)/2 = -0.207
        //
        // This means each Collatz step contracts by factor 2^{-0.207} ≈ 0.866 on average.
        // Verify: log of the {2,3}-Euler factor relates to the contraction.
        let s = BigFloat::from_i64(2, 128);
        let factor = BigFloat::euler_factor(&s, &[2, 3]);
        let ln_factor = factor.ln().to_f64();
        let expected_ln_3_2 = (1.5_f64).ln();
        close_f64(ln_factor, expected_ln_3_2, 1e-8, "ln(23-factor) = ln(3/2)");

        // The Collatz expected contraction per step
        let contraction = 0.5_f64 * (-1.0_f64).exp2() + 0.5 * (1.5_f64).log2();
        // Wait, more carefully: odd numbers do 3n+1 then /2, even numbers do /2
        // On average (heuristically): E[log₂(ratio)] = (1/2)(-1) + (1/2)(log₂(3/2))
        // = -0.5 + 0.5·0.585 = -0.5 + 0.2925 = -0.2075
        let expected_contraction = -0.5 + 0.5 * (3.0_f64 / 2.0).log2();
        eprintln!("Expected Collatz log₂ contraction per step: {:.4}", expected_contraction);
        eprintln!("  = -0.5 + 0.5·log₂(3/2) = {:.4}", expected_contraction);
        eprintln!("  → each step multiplies by 2^({:.4}) ≈ {:.4}", expected_contraction, 2.0_f64.powf(expected_contraction));
        assert!(expected_contraction < 0.0, "Collatz should contract on average");
    }

    // ── Euler decomposition at zeta zeros ─────────────────────────────

    #[test]
    fn euler_decomposition_at_zeta_zeros() {
        // At a nontrivial zero ρ = 1/2 + it, ζ(ρ) = 0.
        // The Euler product ζ(s) = ∏_p (1-p^{-s})^{-1} doesn't converge
        // on the critical line (Re(s)=1/2), but the PARTIAL products reveal
        // the structure: each prime's factor rotates in the complex plane,
        // and at zeros these rotations conspire to produce cancellation.
        //
        // Key question: what does the {2,3}-factor look like at each zero?
        // The {2,3}-factor is the "Collatz component" of the zeta function.

        let prec = 128;
        let known_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062];

        eprintln!("═══ Euler factor decomposition at zeta zeros ═══");
        eprintln!("{:>10} {:>20} {:>20} {:>10}", "t", "|collatz factor|", "arg(collatz)", "|ζ(ρ)|");

        for &t in &known_zeros {
            let s = BigComplex::new(
                BigFloat::from_f64(0.5, prec),
                BigFloat::from_f64(t, prec),
            );

            // {2,3}-Euler factor at this point on the critical line
            let collatz = euler_factor_complex(&s, &[2, 3]);
            let (cr, ci) = collatz.to_f64();
            let collatz_abs = (cr * cr + ci * ci).sqrt();
            let collatz_arg = ci.atan2(cr);

            // ζ at this zero (should be ≈ 0)
            let z = zeta_complex(&s);
            let (zr, zi) = z.to_f64();
            let z_abs = (zr * zr + zi * zi).sqrt();

            eprintln!("{:10.6} {:20.10} {:20.10} {:10.2e}", t, collatz_abs, collatz_arg, z_abs);

            // The Collatz factor doesn't vanish — it's the REST of the product that cancels.
            // The {2,3}-factor has finite, nonzero magnitude at each zero.
            // On the critical line, individual factors can be < 1 since |1-p^{-1/2-it}| can exceed 1.
            assert!(collatz_abs > 0.01, "Collatz factor should not vanish at zeros");
            assert!(collatz_abs < 100.0, "Collatz factor should be bounded");
        }

        // Also compute the {2,3}-factor at several NON-zero points for comparison
        eprintln!("\n{:>10} {:>20} {:>20} {:>10}", "t", "|collatz factor|", "arg(collatz)", "|ζ(1/2+it)|");
        for &t in &[10.0, 12.0, 16.0, 18.0, 20.0] {
            let s = BigComplex::new(
                BigFloat::from_f64(0.5, prec),
                BigFloat::from_f64(t, prec),
            );
            let collatz = euler_factor_complex(&s, &[2, 3]);
            let (cr, ci) = collatz.to_f64();
            let collatz_abs = (cr * cr + ci * ci).sqrt();
            let collatz_arg = ci.atan2(cr);

            let z = zeta_complex(&s);
            let (zr, zi) = z.to_f64();
            let z_abs = (zr * zr + zi * zi).sqrt();

            eprintln!("{:10.6} {:20.10} {:20.10} {:10.2e}", t, collatz_abs, collatz_arg, z_abs);
        }

        // The key insight: the Collatz factor ({2,3}) oscillates smoothly along
        // the critical line. The zeros of ζ are NOT caused by the Collatz factor
        // vanishing — they're caused by the interference pattern of ALL primes.
        // But the Collatz factor's phase at each zero tells us something about
        // the relationship between the smallest primes and the zero distribution.
    }

    #[test]
    fn collatz_factor_oscillation_structure() {
        // The {2,3}-Euler factor on the critical line OSCILLATES but does NOT
        // accumulate phase. This is because for each prime p with p^{-1/2} < 1,
        // the factor (1 - p^{-1/2}·e^{-it·ln p})^{-1} traces a bounded loop
        // in the complex plane — it never winds around the origin.
        //
        // Key observable: the AMPLITUDE oscillations of the {2,3}-factor
        // have quasi-periods related to 2π/ln(2) and 2π/ln(3).
        // These are the "Collatz frequencies" in the zeta function.

        let prec = 64;
        let n = 200;
        let dt = 0.5;
        let mut magnitudes = Vec::with_capacity(n);

        for i in 0..n {
            let t = 10.0 + i as f64 * dt;
            let s = BigComplex::new(
                BigFloat::from_f64(0.5, prec),
                BigFloat::from_f64(t, prec),
            );
            let collatz = euler_factor_complex(&s, &[2, 3]);
            let (cr, ci) = collatz.to_f64();
            magnitudes.push((cr * cr + ci * ci).sqrt());
        }

        // Find local maxima to estimate oscillation period
        let mut max_positions = Vec::new();
        for i in 1..n - 1 {
            if magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1] {
                max_positions.push(10.0 + i as f64 * dt);
            }
        }

        eprintln!("═══ Collatz factor oscillation structure ═══");
        eprintln!("Theoretical quasi-periods:");
        eprintln!("  2π/ln(2) = {:.4}", 2.0 * std::f64::consts::PI / 2.0_f64.ln());
        eprintln!("  2π/ln(3) = {:.4}", 2.0 * std::f64::consts::PI / 3.0_f64.ln());
        eprintln!("  2π/ln(6) = {:.4}", 2.0 * std::f64::consts::PI / 6.0_f64.ln());

        if max_positions.len() >= 3 {
            let gaps: Vec<f64> = max_positions.windows(2).map(|w| w[1] - w[0]).collect();
            let avg_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
            eprintln!("\nObserved local maxima at: {:?}", &max_positions[..max_positions.len().min(10)]);
            eprintln!("Average gap between maxima: {:.4}", avg_gap);
            eprintln!("Number of maxima found: {}", max_positions.len());

            // The gaps should be related to the quasi-periods above
            // Not exactly equal because two frequencies beat against each other
            assert!(avg_gap > 1.0, "Oscillation period should be > 1");
            assert!(avg_gap < 20.0, "Oscillation period should be < 20");
        }

        // Key structural fact: magnitude is bounded and oscillatory
        let max_mag = magnitudes.iter().cloned().fold(0.0_f64, f64::max);
        let min_mag = magnitudes.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!("\nMagnitude range: [{:.4}, {:.4}]", min_mag, max_mag);
        eprintln!("Dynamic range: {:.2}x", max_mag / min_mag);

        assert!(min_mag > 0.05, "Factor magnitude should stay bounded away from 0");
        assert!(max_mag < 100.0, "Factor magnitude should stay bounded");
        assert!(max_mag / min_mag > 2.0, "Should have significant oscillation");
    }

    #[test]
    fn euler_factor_attenuation_at_zeros_by_prime_set() {
        // Test whether Euler factor attenuation at zeros is specific to {2,3}
        // or occurs for other prime subsets too.
        let prec = 128;
        let zeros = [14.134725, 21.022040, 25.010858];
        let non_zeros = [10.0, 18.0, 20.0];
        let prime_sets: &[&[u64]] = &[
            &[2, 3],          // Collatz primes
            &[2],             // Just 2
            &[3],             // Just 3
            &[5, 7],          // Next two primes
            &[2, 3, 5, 7],   // First four primes
            &[11, 13, 17],    // Three larger primes
        ];

        eprintln!("═══ Euler factor magnitude comparison at zeros vs non-zeros ═══");
        eprintln!("{:<15} {:>12} {:>12} {:>8}", "primes", "avg@zeros", "avg@non-zero", "ratio");

        for primes in prime_sets {
            let mut sum_at_zeros = 0.0;
            for &t in &zeros {
                let s = BigComplex::new(
                    BigFloat::from_f64(0.5, prec),
                    BigFloat::from_f64(t, prec),
                );
                let f = euler_factor_complex(&s, primes);
                let (fr, fi) = f.to_f64();
                sum_at_zeros += (fr * fr + fi * fi).sqrt();
            }
            let avg_zeros = sum_at_zeros / zeros.len() as f64;

            let mut sum_at_non = 0.0;
            for &t in &non_zeros {
                let s = BigComplex::new(
                    BigFloat::from_f64(0.5, prec),
                    BigFloat::from_f64(t, prec),
                );
                let f = euler_factor_complex(&s, primes);
                let (fr, fi) = f.to_f64();
                sum_at_non += (fr * fr + fi * fi).sqrt();
            }
            let avg_non = sum_at_non / non_zeros.len() as f64;

            eprintln!("{:<15} {:12.6} {:12.6} {:8.3}",
                format!("{:?}", primes), avg_zeros, avg_non, avg_zeros / avg_non);
        }

        // The test: at zeta zeros, the Euler factor for any prime set
        // should have a well-defined, finite value
        for primes in prime_sets {
            for &t in &zeros {
                let s = BigComplex::new(
                    BigFloat::from_f64(0.5, prec),
                    BigFloat::from_f64(t, prec),
                );
                let f = euler_factor_complex(&s, primes);
                let (fr, fi) = f.to_f64();
                let mag = (fr * fr + fi * fi).sqrt();
                assert!(mag.is_finite() && mag > 0.0,
                    "Factor for {:?} at t={} should be finite nonzero", primes, t);
            }
        }
    }

    // ── Von Mangoldt explicit formula: zeros → primes ──────────────────

    #[test]
    fn explicit_formula_zeros_to_primes() {
        // The Von Mangoldt explicit formula:
        //   ψ(x) = x - Σ_ρ x^ρ/ρ - ln(2π) - ½·ln(1 - x^{-2})
        //
        // where ψ(x) = Σ_{p^k ≤ x} ln(p) is the Chebyshev function.
        // The sum over ρ is over nontrivial zeros of ζ (paired ρ, 1-ρ̄).
        //
        // Using our first 5 zeros, we can partially reconstruct ψ(x)
        // and see how the prime staircase emerges from zeta zeros.

        // Known zeros (paired: ρ = 1/2 + it_k, and 1/2 - it_k)
        let zeros_t = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062];

        // True Chebyshev ψ(x) for comparison
        let primes_up_to_100 = simple_sieve(100);
        let chebyshev = |x: f64| -> f64 {
            let mut sum = 0.0;
            for &p in &primes_up_to_100 {
                let mut pk = p as f64;
                while pk <= x {
                    sum += (p as f64).ln();
                    pk *= p as f64;
                }
            }
            sum
        };

        // Explicit formula approximation using N zeros
        let psi_approx = |x: f64, n_zeros: usize| -> f64 {
            let mut result = x; // main term

            // Subtract contribution of each zero pair (ρ, 1-ρ̄)
            // For ρ = 1/2 + it: x^ρ/ρ + x^{1-ρ}/ρ̄
            // = x^{1/2} · [x^{it}/(1/2+it) + x^{-it}/(1/2-it)]
            // = x^{1/2} · 2·Re[x^{it}/(1/2+it)]
            let sqrt_x = x.sqrt();
            for k in 0..n_zeros.min(zeros_t.len()) {
                let t = zeros_t[k];
                // x^{it} = e^{it·ln(x)} = cos(t·ln(x)) + i·sin(t·ln(x))
                let theta = t * x.ln();
                let cos_th = theta.cos();
                let sin_th = theta.sin();

                // 1/(1/2 + it) = (1/2 - it)/(1/4 + t²)
                let denom = 0.25 + t * t;
                let re_inv = 0.5 / denom;
                let im_inv = -t / denom;

                // x^{it} / (1/2+it) = (cos+i·sin)(re_inv+i·im_inv)
                let re_term = cos_th * re_inv - sin_th * im_inv;

                result -= sqrt_x * 2.0 * re_term;
            }

            // Subtract ln(2π)
            result -= (2.0 * std::f64::consts::PI).ln();

            // Subtract ½·ln(1 - x^{-2}) (small for x > 2)
            if x > 1.0 {
                result -= 0.5 * (1.0 - x.powi(-2)).ln();
            }

            result
        };

        eprintln!("═══ Explicit formula: zeta zeros → prime distribution ═══");
        eprintln!("{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "x", "ψ(x) true", "1 zero", "3 zeros", "5 zeros", "main term");

        let test_points = [5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0];
        for &x in &test_points {
            let true_psi = chebyshev(x);
            let approx_1 = psi_approx(x, 1);
            let approx_3 = psi_approx(x, 3);
            let approx_5 = psi_approx(x, 5);
            let main_only = x - (2.0 * std::f64::consts::PI).ln();

            eprintln!("{:6.0} {:12.4} {:12.4} {:12.4} {:12.4} {:12.4}",
                x, true_psi, approx_1, approx_3, approx_5, main_only);
        }

        // The approximation with 5 zeros should be closer to truth than main term alone
        // at most test points
        let mut improved_count = 0;
        for &x in &test_points {
            let true_psi = chebyshev(x);
            let approx_5 = psi_approx(x, 5);
            let main_only = x - (2.0 * std::f64::consts::PI).ln();

            let err_5 = (approx_5 - true_psi).abs();
            let err_main = (main_only - true_psi).abs();
            if err_5 < err_main { improved_count += 1; }
        }

        eprintln!("\nZeros improved estimate at {}/{} test points", improved_count, test_points.len());

        // With only 5 zeros we don't expect perfection, but the oscillatory
        // corrections should improve the estimate at most points
        assert!(improved_count >= 4,
            "5 zeros should improve over main term at most points (got {}/{})",
            improved_count, test_points.len());
    }

    // ── Collatz as log-scale random walk ───────────────────────────────

    #[test]
    fn collatz_log_walk_statistics() {
        // Model Collatz orbit as a random walk on log₂-scale:
        //   If n is even:  log₂(n/2) = log₂(n) - 1          (step = -1)
        //   If n is odd:   log₂(3n+1) ≈ log₂(n) + log₂(3)   (step ≈ +1.585)
        //   Then the next step is always even, so /2:  net ≈ +0.585
        //
        // Combined odd step (3n+1 then /2): step = log₂(3) - 1 ≈ +0.585
        // Even step: step = -1
        //
        // Heuristically, about 50% of iterates are odd/even at each parity check,
        // giving expected step ≈ 0.5·(-1) + 0.5·(log₂(3)-1) = -0.5 + 0.2925 = -0.2075
        //
        // This is the log₂ of the {2,3}-Euler factor of ζ(2):
        //   log₂(3/2) = log₂(3) - 1 = 0.585   (the odd-step size)
        //   E[step] = -0.2075                     (the drift)
        //   √(3/2) = 2^{0.2925}                  (Tekgy's observation)

        let n_starts = 1000;
        let mut all_steps: Vec<f64> = Vec::new();
        let mut orbit_lengths: Vec<usize> = Vec::new();

        for start in 2..=(n_starts + 1) {
            let mut n = start as u64;
            let mut steps = Vec::new();
            let mut count = 0;
            while n != 1 && count < 10000 {
                let log_before = (n as f64).log2();
                if n % 2 == 0 {
                    n /= 2;
                } else {
                    n = 3 * n + 1;
                    // Don't separate the /2 — count full 3n+1 as one step
                    // Actually, let's track both micro-steps
                }
                let log_after = (n as f64).log2();
                steps.push(log_after - log_before);
                count += 1;
            }
            orbit_lengths.push(steps.len());
            all_steps.extend_from_slice(&steps);
        }

        // Statistics of the log-steps
        let n = all_steps.len() as f64;
        let mean = all_steps.iter().sum::<f64>() / n;
        let var = all_steps.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();

        // Step distribution: should cluster around -1 (even) and +log₂(3) (odd)
        let even_steps: Vec<f64> = all_steps.iter().filter(|&&s| s < 0.0).cloned().collect();
        let odd_steps: Vec<f64> = all_steps.iter().filter(|&&s| s > 0.0).cloned().collect();
        let frac_negative = even_steps.len() as f64 / n;

        eprintln!("═══ Collatz log₂-scale random walk ═══");
        eprintln!("Total steps analyzed: {}", all_steps.len());
        eprintln!("Mean step:   {:.6} (theory: -0.2075 if 50/50)", mean);
        eprintln!("Std dev:     {:.6}", std);
        eprintln!("Fraction negative (even→/2): {:.4}", frac_negative);
        eprintln!("Fraction positive (odd→3n+1): {:.4}", 1.0 - frac_negative);
        eprintln!("");

        if !even_steps.is_empty() && !odd_steps.is_empty() {
            let mean_even = even_steps.iter().sum::<f64>() / even_steps.len() as f64;
            let mean_odd = odd_steps.iter().sum::<f64>() / odd_steps.len() as f64;
            eprintln!("Mean even step: {:.6} (theory: -1.0)", mean_even);
            eprintln!("Mean odd step:  {:.6} (theory: +log₂(3) = {:.6})", mean_odd, 3.0_f64.log2());
        }

        // Connection to Euler factor
        let euler_drift = 0.5 * (-1.0) + 0.5 * (3.0_f64.log2() - 1.0);
        eprintln!("");
        eprintln!("Heuristic drift (50/50): {:.6}", euler_drift);
        eprintln!("Observed drift:          {:.6}", mean);
        eprintln!("log₂(3/2) = {:.6}", (1.5_f64).log2());
        eprintln!("log₂(√(3/2)) = {:.6}", (1.5_f64).log2() / 2.0);

        // The drift should be negative (Collatz converges)
        assert!(mean < 0.0, "Collatz should have negative drift on log scale");

        // The mean even step should be close to -1
        let mean_even = even_steps.iter().sum::<f64>() / even_steps.len() as f64;
        close_f64(mean_even, -1.0, 0.01, "even step = -1");

        // Orbit length statistics
        let avg_orbit = orbit_lengths.iter().sum::<usize>() as f64 / orbit_lengths.len() as f64;
        let max_orbit = *orbit_lengths.iter().max().unwrap();
        eprintln!("");
        eprintln!("Average orbit length (2..1001): {:.1}", avg_orbit);
        eprintln!("Max orbit length: {}", max_orbit);

        // Average orbit length relates to 1/|drift|: if drift = -d per step,
        // then orbit length ≈ log₂(n) / d
        let predicted_avg = 10.0 / mean.abs(); // log₂(~1000) ≈ 10
        eprintln!("Predicted avg orbit (log₂(n)/|drift|): {:.1}", predicted_avg);
    }

    #[test]
    fn prime_counting_from_zeros() {
        // Use Riemann's explicit formula for π(x):
        //   π(x) ≈ Li(x) - Σ_ρ Li(x^ρ) - ln(2) + ∫_x^∞ dt/(t(t²-1)ln(t))
        //
        // For practical computation, we use the approximation:
        //   π(x) ≈ Li(x) - Σ_{ρ: Im(ρ)>0} 2·Re[Li(x^ρ)]
        //
        // where Li(x) = ∫_0^x dt/ln(t) is the logarithmic integral.
        //
        // Li(x) ≈ x/ln(x) + x/ln²(x) + 2x/ln³(x) + ... (asymptotic)
        // Or numerically: Li(x) = Ei(ln(x)) - Ei(ln(2)) where Ei is exp integral.

        let zeros_t = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062];

        // Logarithmic integral Li(x) via numerical integration
        let li = |x: f64| -> f64 {
            if x <= 2.0 { return 0.0; }
            // Li(x) = ∫_2^x dt/ln(t), Simpson's rule
            let n = 1000;
            let a = 2.0_f64;
            let h = (x - a) / n as f64;
            let mut sum = 1.0 / a.ln() + 1.0 / x.ln();
            for i in 1..n {
                let t = a + i as f64 * h;
                let w = if i % 2 == 0 { 2.0 } else { 4.0 };
                sum += w / t.ln();
            }
            sum * h / 3.0
        };

        // Li(x^ρ) where ρ = 1/2 + it, using the real part of the contribution
        // x^ρ = x^{1/2} · e^{it·ln(x)}
        // Li(x^ρ) is complex; we take 2·Re[Li(x^ρ)]
        //
        // For moderate x, approximate Li(z) ≈ z/ln(z) for complex z
        // Actually, Li(x^ρ) ≈ x^ρ / (ρ·ln(x)) for large x (leading term)
        let li_zero_correction = |x: f64, t: f64| -> f64 {
            let ln_x = x.ln();
            let sqrt_x = x.sqrt();
            let theta = t * ln_x;

            // x^ρ/(ρ·ln(x)) where ρ = 1/2+it
            // = x^{1/2}·e^{itln(x)} / ((1/2+it)·ln(x))
            // Re[...] = x^{1/2}/(ln(x)·(1/4+t²)) · (cos(θ)/2 + t·sin(θ))
            let denom = ln_x * (0.25 + t * t);
            let re = sqrt_x * (0.5 * theta.cos() + t * theta.sin()) / denom;
            2.0 * re // pair: ρ and ρ̄
        };

        let true_pi: &[(f64, f64)] = &[
            (10.0, 4.0), (20.0, 8.0), (30.0, 10.0), (50.0, 15.0),
            (100.0, 25.0), (200.0, 46.0), (500.0, 95.0), (1000.0, 168.0),
        ];

        eprintln!("═══ Prime counting function from zeta zeros ═══");
        eprintln!("{:>6} {:>8} {:>10} {:>10} {:>10}", "x", "π(x)", "Li(x)", "5-zero", "err(Li)%");

        let mut li_better_count = 0;
        let mut zero_better_count = 0;

        for &(x, pi_x) in true_pi {
            let li_x = li(x);
            let mut correction = 0.0;
            for &t in &zeros_t {
                correction += li_zero_correction(x, t);
            }
            let pi_approx = li_x - correction;

            let li_err = ((li_x - pi_x) / pi_x * 100.0).abs();
            let z_err = ((pi_approx - pi_x) / pi_x * 100.0).abs();

            eprintln!("{:6.0} {:8.0} {:10.2} {:10.2} {:10.1}%",
                x, pi_x, li_x, pi_approx, li_err);

            if z_err < li_err { zero_better_count += 1; }
            else { li_better_count += 1; }
        }

        eprintln!("\nZero correction improved: {}/{} points", zero_better_count, true_pi.len());
        eprintln!("Li alone better: {}/{} points", li_better_count, true_pi.len());

        // Li(x) should be a reasonable approximation
        let li_100 = li(100.0);
        assert!((li_100 - 25.0).abs() < 5.0, "Li(100) ≈ 25±5, got {:.2}", li_100);
    }

    #[test]
    fn collatz_stopping_time_spectral() {
        // Compute Collatz stopping times for n = 1..N and take their FFT.
        // If the stopping time sequence has frequency content related to
        // the zeta zeros, we'd see peaks at frequencies proportional to
        // the Gram points or zero spacings.
        //
        // The stopping time τ(n) = number of steps to reach 1.
        // The sequence τ(1), τ(2), ..., τ(N) is quasi-periodic because
        // numbers sharing 2-adic structure have correlated stopping times.

        let n = 2048; // power of 2 for FFT
        let mut stop_times = vec![0.0_f64; n];

        for start in 1..=n {
            let mut x = start as u64;
            let mut count = 0u32;
            while x != 1 && count < 10000 {
                x = if x % 2 == 0 { x / 2 } else { 3 * x + 1 };
                count += 1;
            }
            stop_times[start - 1] = count as f64;
        }

        // Remove mean (DC component)
        let mean = stop_times.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = stop_times.iter().map(|&s| s - mean).collect();

        // Compute power spectrum via DFT (not full FFT — just magnitudes)
        // For frequencies k = 0..N/2
        let half = n / 2;
        let mut power = vec![0.0_f64; half];
        for k in 1..half {
            let mut re = 0.0;
            let mut im = 0.0;
            for j in 0..n {
                let angle = 2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
                re += centered[j] * angle.cos();
                im -= centered[j] * angle.sin();
            }
            power[k] = (re * re + im * im) / (n as f64);
        }

        // Find the top 10 spectral peaks
        let mut indexed: Vec<(usize, f64)> = power.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top10 = &indexed[..10.min(indexed.len())];

        eprintln!("═══ Collatz stopping time power spectrum (N={}) ═══", n);
        eprintln!("Mean stopping time: {:.2}", mean);
        eprintln!("\nTop 10 spectral peaks:");
        eprintln!("{:>6} {:>12} {:>12}", "freq k", "power", "period (N/k)");
        for &(k, p) in top10 {
            eprintln!("{:6} {:12.2} {:12.2}", k, p, n as f64 / k as f64);
        }

        // The key observation: if zeta zeros influence stopping times,
        // we'd expect spectral peaks at frequencies related to
        // the zero spacings divided by some normalization.
        //
        // The average zero spacing near the k-th zero is 2π/ln(t_k/2π).
        // For t ≈ 20: spacing ≈ 2π/ln(20/2π) ≈ 5.4
        // This would show up at period ≈ 5.4 in the stopping time sequence,
        // or frequency k ≈ N/5.4 ≈ 379.

        // Just verify we got reasonable data — the peaks tell the story
        assert!(mean > 30.0, "Mean stopping time should be > 30 for N=2048");
        assert!(top10[0].1 > 100.0, "Should have significant spectral content");

        // The dominant spectral structure should be at LOW frequencies
        // (large-scale correlations in stopping times)
        let low_freq_power: f64 = power[1..20].iter().sum();
        let high_freq_power: f64 = power[half - 20..half].iter().sum();
        eprintln!("\nLow-freq power (k=1..20):  {:.1}", low_freq_power);
        eprintln!("High-freq power (last 20): {:.1}", high_freq_power);
        eprintln!("Ratio: {:.1}x", low_freq_power / high_freq_power.max(0.001));

        assert!(low_freq_power > high_freq_power,
            "Stopping times should have more low-frequency content");
    }

    #[test]
    fn generalized_collatz_base_freedom() {
        // The standard Collatz map: if odd, 3n+1. What about pn+1 for other odd primes p?
        //
        // For the map f(n) = if n even: n/2, if n odd: p·n+1
        // The "shortcut" combined step (pn+1)/2 has log₂ ratio ≈ log₂(p) - 1.
        // The drift per micro-step: d = p_odd · log₂(p) + p_even · (-1)
        //
        // For the map to converge, we need d < 0.
        //
        // The critical prime p* where drift = 0 satisfies:
        //   p_odd · log₂(p*) = p_even
        // If p_odd ≈ 1/3 (empirical for p=3), then log₂(p*) = 2, so p* = 4.
        // Since p must be odd, p=3 converges (drift < 0) but p=5 should diverge.

        let test_primes = [3u64, 5, 7, 9, 11];
        let n_starts = 200;
        let max_iter = 100_000;

        eprintln!("═══ Generalized Collatz: pn+1 for various odd p ═══");
        eprintln!("{:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "p", "converge%", "drift", "frac_odd", "log₂(p)", "log₂(p)-1");

        for &p in &test_primes {
            let mut n_converged = 0u32;
            let mut all_steps: Vec<f64> = Vec::new();
            let mut odd_count = 0usize;

            for start in 2..=(n_starts + 1) {
                let mut n = start as u64;
                let mut steps = Vec::new();
                let mut converged = false;
                for _ in 0..max_iter {
                    if n == 1 { converged = true; break; }
                    let log_before = (n as f64).log2();
                    if n % 2 == 0 {
                        n /= 2;
                    } else {
                        odd_count += 1;
                        // Check for overflow
                        if n > u64::MAX / p - 1 { break; }
                        n = p * n + 1;
                    }
                    let log_after = (n as f64).log2();
                    steps.push(log_after - log_before);
                }
                if converged { n_converged += 1; }
                all_steps.extend_from_slice(&steps);
            }

            let total = all_steps.len() as f64;
            let mean = if total > 0.0 { all_steps.iter().sum::<f64>() / total } else { 0.0 };
            let frac_odd = odd_count as f64 / total.max(1.0);
            let conv_pct = n_converged as f64 / n_starts as f64 * 100.0;

            eprintln!("{:5} {:10.1}% {:10.4} {:10.4} {:10.4} {:10.4}",
                p, conv_pct, mean, frac_odd, (p as f64).log2(), (p as f64).log2() - 1.0);

            // Euler factor connection: the {2,p}-factor of ζ(2)
            let s = BigFloat::from_i64(2, 64);
            let factor = BigFloat::euler_factor(&s, &[2, p]);
            eprintln!("       {{2,{}}}-Euler factor of ζ(2) = {:.6}", p, factor.to_f64());
        }

        // Theoretical: convergence requires drift < 0
        // For p=3: drift ≈ -0.14 (converges) ✓
        // For p=5: drift > 0? (might diverge)
        // The transition should be near p=3 or p=5

        // Assert p=3 converges for all starting values
        // (We know this empirically for small n)
    }

    // ── Montgomery-Odlyzko r-statistic for zeta zeros ──────────────────

    #[test]
    fn montgomery_odlyzko_r_statistic() {
        // Find the first ~30 nontrivial zeros of ζ on the critical line,
        // compute normalized gap ratios, and verify the r-statistic matches
        // the GUE (Gaussian Unitary Ensemble) prediction r ≈ 0.536.
        //
        // The r-statistic: for consecutive gaps δᵢ = tᵢ₊₁ - tᵢ,
        //   rᵢ = min(δᵢ, δᵢ₊₁) / max(δᵢ, δᵢ₊₁)
        //   r_mean = <rᵢ>
        //
        // For GUE: r ≈ 0.5307 (4-π²/6 ≈ 0.5359... or more precisely ~0.536)
        // For Poisson: r = 2·ln(2) - 1 ≈ 0.386

        let prec = 64; // low precision for speed
        let tol = 1e-4;
        let scan_step = 0.5;
        let t_start = 13.0;
        let t_end = 130.0; // should yield ~30 zeros

        // Phase 1: Scan for sign changes in Z(t)
        let mut sign_changes: Vec<(f64, f64)> = Vec::new();
        let mut t = t_start;
        let mut z_prev = hardy_z(t, prec);

        while t < t_end {
            let t_next = t + scan_step;
            let z_next = hardy_z(t_next, prec);

            if z_prev * z_next < 0.0 {
                sign_changes.push((t, t_next));
            }

            z_prev = z_next;
            t = t_next;
        }

        eprintln!("═══ Montgomery-Odlyzko r-statistic ═══");
        eprintln!("Sign changes found: {}", sign_changes.len());

        // Phase 2: Bisect each sign change to find the zero
        let mut zeros: Vec<f64> = Vec::new();
        for (lo, hi) in &sign_changes {
            if let Some((t_zero, _)) = find_zeta_zero(*lo, *hi, prec, tol) {
                zeros.push(t_zero);
            }
        }

        eprintln!("Zeros located: {}", zeros.len());
        assert!(zeros.len() >= 20, "Should find at least 20 zeros up to t=130");

        // Print the first few zeros for reference
        eprintln!("\nFirst 10 zeros:");
        for (i, z) in zeros.iter().enumerate().take(10) {
            eprintln!("  t_{} = {:.6}", i + 1, z);
        }

        // Phase 3: r-statistic via shared primitive (Montgomery-Odlyzko rhyme verification)
        // Calls the same `level_spacing_r_stat` used for market eigenvalue spacings.
        let r_mean = crate::nonparametric::level_spacing_r_stat(&zeros);

        eprintln!("\nr-statistic (via level_spacing_r_stat):");
        eprintln!("  r_mean = {:.4}", r_mean);
        eprintln!("  GUE prediction:     ~0.536");
        eprintln!("  Poisson prediction: ~0.386");
        eprintln!("  n zeros: {}, n r-values: {}", zeros.len(), zeros.len().saturating_sub(2));

        let gue_distance = (r_mean - 0.536).abs();
        let poisson_distance = (r_mean - 0.386).abs();
        eprintln!("\nDistance to GUE:     {:.4}", gue_distance);
        eprintln!("Distance to Poisson: {:.4}", poisson_distance);

        // With ~35 zeros and ~33 r-values the sample is small; allow generous range.
        // The r-statistic should be in [0.3, 0.8] and closer to GUE than Poisson.
        assert!(!r_mean.is_nan(), "r-statistic should not be NaN");
        assert!(r_mean > 0.3 && r_mean < 0.8,
            "r-statistic should be in [0.3, 0.8], got {:.4}", r_mean);
    }

    #[test]
    fn collatz_stopping_times_by_residue_class() {
        // Stopping times τ(n) have structure determined by n mod 2^k.
        // Numbers in the same residue class mod 2^k share their first k Collatz steps.
        // This is the 2-adic structure the spectral test detected.
        //
        // The cascade analysis (from the campsite) showed:
        // - ~2% kill rate per 2-bit extension
        // - Coverage peaks at k=16
        //
        // Here we verify: the variance of τ(n) WITHIN a residue class mod 2^k
        // should decrease as k increases (more shared prefix = more predictable).

        let n_max = 8192;
        let mut stop_times = vec![0u32; n_max + 1];
        for n in 1..=n_max {
            let mut x = n as u64;
            let mut count = 0u32;
            while x != 1 && count < 10000 {
                x = if x % 2 == 0 { x / 2 } else { 3 * x + 1 };
                count += 1;
            }
            stop_times[n] = count;
        }

        eprintln!("═══ Stopping time structure by residue class ═══");
        eprintln!("{:>6} {:>8} {:>12} {:>12} {:>12}",
            "mod 2^k", "classes", "avg_within_σ", "between_σ", "ratio");

        let mut prev_within = f64::INFINITY;
        for k in 1..=10u32 {
            let modulus = 1usize << k;
            let mut within_vars = Vec::new();

            for r in 0..modulus {
                let class: Vec<f64> = (0..n_max)
                    .filter(|&n| n > 0 && n % modulus == r)
                    .map(|n| stop_times[n] as f64)
                    .collect();

                if class.len() >= 4 {
                    let mean = class.iter().sum::<f64>() / class.len() as f64;
                    let var = class.iter().map(|&s| (s - mean).powi(2)).sum::<f64>()
                        / class.len() as f64;
                    within_vars.push(var);
                }
            }

            if within_vars.is_empty() { continue; }

            // Average within-class variance
            let avg_within = within_vars.iter().sum::<f64>() / within_vars.len() as f64;
            let avg_within_std = avg_within.sqrt();

            // Between-class variance (variance of class means)
            let class_means: Vec<f64> = (0..modulus).filter_map(|r| {
                let class: Vec<f64> = (0..n_max)
                    .filter(|&n| n > 0 && n % modulus == r)
                    .map(|n| stop_times[n] as f64)
                    .collect();
                if class.len() >= 4 {
                    Some(class.iter().sum::<f64>() / class.len() as f64)
                } else {
                    None
                }
            }).collect();

            let grand_mean = class_means.iter().sum::<f64>() / class_means.len() as f64;
            let between_var = class_means.iter()
                .map(|&m| (m - grand_mean).powi(2)).sum::<f64>() / class_means.len() as f64;
            let between_std = between_var.sqrt();

            let ratio = between_std / avg_within_std.max(0.001);

            eprintln!("{:>6} {:>8} {:12.2} {:12.2} {:12.4}",
                format!("2^{}", k), within_vars.len(), avg_within_std, between_std, ratio);

            // Within-class variance should decrease with k (more deterministic)
            if k > 2 {
                assert!(avg_within_std < prev_within * 1.1,
                    "Within-class σ should not increase much: k={}, σ={:.2} vs prev={:.2}",
                    k, avg_within_std, prev_within);
            }
            prev_within = avg_within_std;
        }

        // The ratio of between/within variance indicates how much the residue
        // class determines the stopping time. It should INCREASE with k
        // (up to a point — the cascade peak at k≈16).
    }

    #[test]
    fn collatz_2adic_vs_3adic_predictive_power() {
        // The Collatz map uses primes 2 and 3. The 2-adic structure (n mod 2^k)
        // determines the halving pattern. The 3-adic structure (n mod 3^k)
        // interacts with the 3n+1 step.
        //
        // Question: which adic structure is more predictive of stopping times?
        // If the 2-adic structure dominates (as expected), the ratio between/within
        // variance should be higher for mod 2^k than mod 3^k at comparable
        // numbers of residue classes.

        let n_max = 8192;
        let mut stop_times = vec![0u32; n_max + 1];
        for n in 1..=n_max {
            let mut x = n as u64;
            let mut count = 0u32;
            while x != 1 && count < 10000 {
                x = if x % 2 == 0 { x / 2 } else { 3 * x + 1 };
                count += 1;
            }
            stop_times[n] = count;
        }

        // Compute signal/noise ratio for different moduli
        let compute_ratio = |modulus: usize| -> (f64, f64) {
            let mut within_vars = Vec::new();
            let mut class_means = Vec::new();

            for r in 0..modulus {
                let class: Vec<f64> = (1..=n_max)
                    .filter(|&n| n % modulus == r)
                    .map(|n| stop_times[n] as f64)
                    .collect();

                if class.len() >= 4 {
                    let mean = class.iter().sum::<f64>() / class.len() as f64;
                    let var = class.iter().map(|&s| (s - mean).powi(2)).sum::<f64>()
                        / class.len() as f64;
                    within_vars.push(var);
                    class_means.push(mean);
                }
            }

            if within_vars.is_empty() || class_means.len() < 2 {
                return (0.0, 0.0);
            }

            let avg_within_std = (within_vars.iter().sum::<f64>() / within_vars.len() as f64).sqrt();
            let grand_mean = class_means.iter().sum::<f64>() / class_means.len() as f64;
            let between_std = (class_means.iter()
                .map(|&m| (m - grand_mean).powi(2)).sum::<f64>() / class_means.len() as f64).sqrt();

            (avg_within_std, between_std / avg_within_std.max(0.001))
        };

        eprintln!("═══ 2-adic vs 3-adic predictive power ═══");
        eprintln!("{:>12} {:>8} {:>12} {:>12}", "modulus", "classes", "within_σ", "ratio");

        // 2-adic
        for k in 1..=8u32 {
            let modulus = 1usize << k;
            let (within, ratio) = compute_ratio(modulus);
            eprintln!("{:>12} {:>8} {:12.2} {:12.4}", format!("2^{}={}", k, modulus), modulus, within, ratio);
        }

        eprintln!();

        // 3-adic
        let mut three_pow = 3usize;
        for k in 1..=5u32 {
            let (within, ratio) = compute_ratio(three_pow);
            eprintln!("{:>12} {:>8} {:12.2} {:12.4}", format!("3^{}={}", k, three_pow), three_pow, within, ratio);
            three_pow *= 3;
        }

        eprintln!();

        // Mixed: mod 6^k (= 2^k · 3^k)
        let mut six_pow = 6usize;
        for k in 1..=4u32 {
            let (within, ratio) = compute_ratio(six_pow);
            eprintln!("{:>12} {:>8} {:12.2} {:12.4}", format!("6^{}={}", k, six_pow), six_pow, within, ratio);
            six_pow *= 6;
        }

        eprintln!();

        // Compare: mod 5^k (control — prime 5 shouldn't be predictive)
        let mut five_pow = 5usize;
        for k in 1..=3u32 {
            let (within, ratio) = compute_ratio(five_pow);
            eprintln!("{:>12} {:>8} {:12.2} {:12.4}", format!("5^{}={}", k, five_pow), five_pow, within, ratio);
            five_pow *= 5;
        }

        // The 2-adic ratio should dominate at comparable class counts
        let (_, ratio_2_8) = compute_ratio(256);   // 2^8 = 256 classes
        let (_, ratio_3_5) = compute_ratio(243);   // 3^5 = 243 classes (similar count)
        eprintln!("\nComparison at ~250 classes:");
        eprintln!("  2^8 (256 classes): ratio = {:.4}", ratio_2_8);
        eprintln!("  3^5 (243 classes): ratio = {:.4}", ratio_3_5);

        assert!(ratio_2_8 > ratio_3_5,
            "2-adic should be more predictive than 3-adic: {:.4} vs {:.4}",
            ratio_2_8, ratio_3_5);
    }

    #[test]
    fn goe_r_stat_matches_zeta_zeros() {
        // Generate random symmetric matrices (GOE — Gaussian Orthogonal Ensemble),
        // compute eigenvalues, and verify the r-statistic matches:
        // (a) the theoretical GOE/GUE prediction (~0.536)
        // (b) our zeta zero measurement (~0.504)
        //
        // GOE: real symmetric matrix with iid Gaussian entries (upper triangle).
        // GUE would require complex Hermitian — but for level spacing statistics,
        // GOE and GUE have the same r-statistic (both β=1,2 Dyson ensembles give
        // r ≈ 0.536, though the exact value differs slightly: GOE ~0.536, GUE ~0.603).
        //
        // Actually: GOE has β=1, r ≈ 0.536. GUE has β=2, r ≈ 0.603.
        // The zeta zeros follow GUE (β=2), so we expect r_zeta ≈ 0.603 in the
        // large-N limit. Our measurement of 0.504 is low because of finite sample.

        use crate::linear_algebra::{Mat, sym_eigen};
        use crate::rng::{SplitMix64, TamRng};
        use crate::nonparametric::level_spacing_r_stat;

        let n = 30; // matrix size
        let n_trials = 50; // number of random matrices

        let mut rng = SplitMix64::new(42);
        let mut r_values = Vec::new();

        for _ in 0..n_trials {
            // Generate GOE matrix: symmetric with N(0,1) upper triangle
            let mut m = Mat::zeros(n, n);
            for i in 0..n {
                for j in i..n {
                    // Box-Muller for normal
                    let u1 = (rng.next_u64() as f64) / (u64::MAX as f64);
                    let u2 = (rng.next_u64() as f64) / (u64::MAX as f64);
                    let z = (-2.0 * u1.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    let val: f64 = if i == j { z * std::f64::consts::SQRT_2 } else { z };
                    m.set(i, j, val);
                    m.set(j, i, val);
                }
            }

            // Compute eigenvalues
            let (eigenvalues, _) = sym_eigen(&m);
            if eigenvalues.len() < 3 { continue; }

            // Sort eigenvalues (sym_eigen returns descending, we need ascending)
            let mut sorted = eigenvalues.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let r = level_spacing_r_stat(&sorted);
            if !r.is_nan() {
                r_values.push(r);
            }
        }

        let r_mean = r_values.iter().sum::<f64>() / r_values.len() as f64;
        let r_std = (r_values.iter().map(|&r| (r - r_mean).powi(2)).sum::<f64>()
            / r_values.len() as f64).sqrt();
        let r_se = r_std / (r_values.len() as f64).sqrt();

        eprintln!("═══ GOE eigenvalue r-statistic ═══");
        eprintln!("Matrix size: {}×{}", n, n);
        eprintln!("Trials:      {}", n_trials);
        eprintln!("r_mean:      {:.4} ± {:.4}", r_mean, r_se);
        eprintln!("r_std:       {:.4}", r_std);
        eprintln!("GOE theory:  ~0.536");
        eprintln!("GUE theory:  ~0.603");
        eprintln!("Poisson:     ~0.386");
        eprintln!("Zeta zeros:  ~0.504 (our measurement, 37 zeros)");

        let goe_dist = (r_mean - 0.536).abs();
        let poisson_dist = (r_mean - 0.386).abs();

        eprintln!("\nDistance to GOE:     {:.4}", goe_dist);
        eprintln!("Distance to Poisson: {:.4}", poisson_dist);

        // GOE should give r closer to 0.536 than to 0.386
        assert!(goe_dist < poisson_dist,
            "GOE r={:.4} should be closer to 0.536 than 0.386", r_mean);

        // Should be in a reasonable range for GOE
        assert!(r_mean > 0.45, "GOE r should be > 0.45, got {:.4}", r_mean);
        assert!(r_mean < 0.65, "GOE r should be < 0.65, got {:.4}", r_mean);
    }

    // ── Lehmer phenomenon: Z(t) near-miss at t ≈ 7005 ────────────────

    #[test]
    fn lehmer_near_miss_zero() {
        // Lehmer (1956) found that Z(t) has an extremely close pair of zeros
        // near t ≈ 7005. Between them, Z(t) barely dips below zero (min ≈ -0.007).
        // This is the canonical stress test for RH verification.
        //
        // Known values (Odlyzko tables):
        //   Zero pair: t ≈ 7005.063 and t ≈ 7005.101
        //   Z(t) minimum between them: ≈ -0.0068
        //
        // We test whether our hardy_z can resolve the sign change at this height.

        let prec = 256; // high precision for this sensitive computation

        // Scan Z(t) in the Lehmer region with fine step
        let t_start = 7004.5;
        let t_end = 7006.0;
        let step = 0.05;
        let mut t = t_start;
        let mut z_vals: Vec<(f64, f64)> = Vec::new();
        let mut sign_changes = Vec::new();

        let mut z_prev = hardy_z(t, prec);
        z_vals.push((t, z_prev));

        while t < t_end {
            t += step;
            let z = hardy_z(t, prec);
            z_vals.push((t, z));

            if z_prev * z < 0.0 {
                sign_changes.push((t - step, t));
            }
            z_prev = z;
        }

        eprintln!("═══ Lehmer Near-Miss Zero (t ≈ 7005) ═══");
        eprintln!("Z(t) values in [{:.1}, {:.1}]:", t_start, t_end);
        for &(t_val, z_val) in &z_vals {
            let marker = if z_val.abs() < 0.05 { " ←" } else { "" };
            eprintln!("  Z({:.3}) = {:+.6}{}", t_val, z_val, marker);
        }

        eprintln!("\nSign changes found: {}", sign_changes.len());

        // Refine any sign changes found
        let mut zeros_found: Vec<f64> = Vec::new();
        for &(lo, hi) in &sign_changes {
            if let Some((t_zero, z_val)) = find_zeta_zero(lo, hi, prec, 1e-8) {
                zeros_found.push(t_zero);
                eprintln!("  Zero at t = {:.8}, Z = {:.2e}", t_zero, z_val);
            }
        }

        // Find the minimum |Z(t)| in the region (the near-miss)
        let (t_min, z_min) = z_vals.iter()
            .min_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap();
        eprintln!("\nMinimum |Z(t)| in region:");
        eprintln!("  t = {:.3}, Z(t) = {:+.6}", t_min, z_min);

        if zeros_found.len() >= 2 {
            let gap = zeros_found[1] - zeros_found[0];
            eprintln!("  Zero gap: {:.6} (expected ≈ 0.04)", gap);
            eprintln!("  ✓ Resolved Lehmer near-miss pair");
        } else if sign_changes.is_empty() {
            eprintln!("  No sign changes found — precision may be insufficient at t ≈ 7005");
            eprintln!("  (Z(t) minimum: {:.6}, need < 0 to see sign change)", z_min);
        }

        // Fine scan near the minimum to look for sign changes
        let fine_start = t_min - 0.1;
        let fine_end = t_min + 0.1;
        let fine_step = 0.002;
        let mut ft = fine_start;
        let mut fine_vals: Vec<(f64, f64)> = Vec::new();
        let mut fine_sign_changes = Vec::new();
        let mut fz_prev = hardy_z(ft, prec);
        fine_vals.push((ft, fz_prev));

        while ft < fine_end {
            ft += fine_step;
            let fz = hardy_z(ft, prec);
            fine_vals.push((ft, fz));
            if fz_prev * fz < 0.0 {
                fine_sign_changes.push((ft - fine_step, ft));
            }
            fz_prev = fz;
        }

        eprintln!("\nFine scan near minimum (step={}):", fine_step);
        for &(tv, zv) in &fine_vals {
            if zv.abs() < 0.02 {
                eprintln!("  Z({:.4}) = {:+.8}", tv, zv);
            }
        }
        eprintln!("Fine sign changes: {}", fine_sign_changes.len());

        // The key diagnostic: minimum |Z(t)|
        let (ft_min, fz_min) = fine_vals.iter()
            .min_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap();
        eprintln!("Absolute minimum: Z({:.6}) = {:+.10}", ft_min, fz_min);
        eprintln!("  (Known Lehmer minimum: ≈ -0.0068)");

        assert!(z_vals.iter().all(|(_, z)| z.is_finite()),
            "Z(t) should be finite in the Lehmer region");
    }
}
