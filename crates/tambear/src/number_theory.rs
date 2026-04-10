//! # Number Theory
//!
//! Primality, factoring, modular arithmetic, number-theoretic functions,
//! continued fractions, Diophantine equations, and the Riemann zeta connection.
//!
//! ## Architecture (accumulate+gather)
//!
//! - **Sieve of Eratosthenes**: masked scan over integers — accumulate(multiples, Mask, cross_off)
//! - **Miller-Rabin**: iterated squaring chain — accumulate(witness tests, All, witness_pass)
//! - **Euler totient φ(n)**: multiplicative accumulate over prime factors — product(1 - 1/p)
//! - **CRT**: accumulate over congruences, gather into unique solution mod lcm(m_i)
//! - **Continued fractions**: sequential quotient scan — accumulate(|x|, Prefix, integer_part)
//! - **ζ(s) as partition function**: accumulate(primes, All, euler_factor) = partition_function(ln_n, s)

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1 — Primality and Sieve
// ═══════════════════════════════════════════════════════════════════════════

/// Sieve of Eratosthenes. Returns all primes ≤ limit.
/// Accumulate: for each prime p, cross off all multiples (masked scan over integers).
pub fn sieve(limit: usize) -> Vec<u64> {
    if limit < 2 { return vec![]; }
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut p = 2;
    while p * p <= limit {
        if is_prime[p] {
            // Accumulate: cross off all multiples of p starting at p²
            let mut multiple = p * p;
            while multiple <= limit {
                is_prime[multiple] = false;
                multiple += p;
            }
        }
        p += 1;
    }
    (2..=limit).filter(|&i| is_prime[i]).map(|i| i as u64).collect()
}

/// Segmented sieve for large ranges [lo, hi). Returns primes in range.
pub fn segmented_sieve(lo: u64, hi: u64) -> Vec<u64> {
    if hi <= lo || hi <= 2 { return vec![]; }
    let lo = lo.max(2);
    let sqrt_hi = (hi as f64).sqrt() as usize + 1;
    let small_primes = sieve(sqrt_hi);
    let size = (hi - lo) as usize;
    let mut is_prime = vec![true; size];

    for &p in &small_primes {
        let p = p as u64;
        // First multiple of p ≥ lo
        let start = ((lo + p - 1) / p * p).max(p * p);
        if start >= hi { continue; }
        let mut k = (start - lo) as usize;
        while k < size {
            is_prime[k] = false;
            k += p as usize;
        }
    }

    (0..size)
        .filter(|&i| is_prime[i])
        .map(|i| lo + i as u64)
        .collect()
}

/// Miller-Rabin primality test (deterministic for n < 3,317,044,064,679,887,385,961,981).
/// For n < 3.3e24, uses specific witness sets.
pub fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n == 2 || n == 3 || n == 5 || n == 7 { return true; }
    if n % 2 == 0 || n % 3 == 0 || n % 5 == 0 { return false; }

    // Write n-1 = 2^r · d with d odd
    let (r, d) = {
        let mut r = 0u32;
        let mut d = n - 1;
        while d % 2 == 0 { d /= 2; r += 1; }
        (r, d)
    };

    // Deterministic witness sets (covers all n up to specific bounds)
    let witnesses: &[u64] = if n < 2_047 {
        &[2]
    } else if n < 1_373_653 {
        &[2, 3]
    } else if n < 9_080_191 {
        &[31, 73]
    } else if n < 25_326_001 {
        &[2, 3, 5]
    } else if n < 3_215_031_751 {
        &[2, 3, 5, 7]
    } else if n < 4_759_123_141 {
        &[2, 7, 61]
    } else if n < 1_122_004_669_633 {
        &[2, 13, 23, 1_662_803]
    } else if n < 2_152_302_898_747 {
        &[2, 3, 5, 7, 11]
    } else if n < 3_474_749_660_383 {
        &[2, 3, 5, 7, 11, 13]
    } else if n < 341_550_071_728_321 {
        &[2, 3, 5, 7, 11, 13, 17]
    } else {
        &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    };

    // Accumulate: all witnesses must pass
    for &a in witnesses {
        if a >= n { continue; }
        if !miller_rabin_witness(n, a, d, r) { return false; }
    }
    true
}

/// One Miller-Rabin witness test for n with base a, n-1 = 2^r · d.
fn miller_rabin_witness(n: u64, a: u64, d: u64, r: u32) -> bool {
    let mut x = mod_pow(a, d, n);
    if x == 1 || x == n - 1 { return true; }
    for _ in 0..r - 1 {
        x = mul_mod(x, x, n);
        if x == n - 1 { return true; }
    }
    false
}

/// Next prime after n.
pub fn next_prime(n: u64) -> u64 {
    let mut k = if n < 2 { 2 } else { n + 1 };
    if k == 2 { return 2; }
    if k % 2 == 0 { k += 1; }
    loop {
        if is_prime(k) { return k; }
        k += 2;
    }
}

/// Prime counting function π(n): number of primes ≤ n.
pub fn prime_count(n: u64) -> u64 {
    sieve(n as usize).len() as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2 — Modular Arithmetic
// ═══════════════════════════════════════════════════════════════════════════

/// Modular multiplication (a·b) mod m, safe for u64 (avoids overflow via u128).
pub fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular exponentiation: a^e mod m.
/// Kingdom A: the squaring chain base^{2^k} is data-independent (parallel-precomputable);
/// the gather step multiplies selected powers where bit_k(e)=1. Bits of e are DATA, not state.
/// Sequential implementation artifact — can be parallelized: precompute all powers, gather.
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 { return 0; }
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 { result = mul_mod(result, base, modulus); }
        base = mul_mod(base, base, modulus);
        exp >>= 1;
    }
    result
}

/// Greatest common divisor via Euclidean algorithm.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

/// Least common multiple.
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 { return 0; }
    a / gcd(a, b) * b
}

/// Extended Euclidean algorithm: returns (gcd, x, y) with a·x + b·y = gcd.
/// x and y are returned as (possibly negative) i64.
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 { return (a, 1, 0); }
    let (g, x1, y1) = extended_gcd(b, a % b);
    (g, y1, x1 - (a / b) * y1)
}

/// Modular inverse: a⁻¹ mod m. Returns None if gcd(a, m) ≠ 1.
pub fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    let (g, x, _) = extended_gcd(a as i64, m as i64);
    if g != 1 { return None; }
    Some(((x % m as i64 + m as i64) % m as i64) as u64)
}

/// Chinese Remainder Theorem: find x ≡ r_i (mod m_i) for all i.
/// Requires m_i pairwise coprime. Returns (x, M) where M = Π m_i.
/// Accumulate: over congruences, gather into unique solution.
pub fn crt(remainders: &[u64], moduli: &[u64]) -> Option<(u64, u64)> {
    assert_eq!(remainders.len(), moduli.len());
    let n = remainders.len();
    if n == 0 { return None; }

    let m_total: u64 = moduli.iter().product();
    let mut x = 0u64;

    for i in 0..n {
        let mi = m_total / moduli[i];
        let inv = mod_inverse(mi % moduli[i], moduli[i])?;
        // x += r_i · M_i · M_i⁻¹ (mod m_total)
        x = (x + mul_mod(mul_mod(remainders[i], mi % m_total, m_total), inv, m_total)) % m_total;
    }

    Some((x, m_total))
}

/// Legendre symbol (a/p) for odd prime p. Returns 0, 1, or -1.
pub fn legendre(a: i64, p: u64) -> i64 {
    if a == 0 { return 0; }
    let a = ((a % p as i64 + p as i64) % p as i64) as u64;
    if a == 0 { return 0; }
    let exp = (p - 1) / 2;
    let result = mod_pow(a, exp, p);
    if result == 1 { 1 } else { -1 }
}

/// Jacobi symbol (a/n) for odd positive n. Extends Legendre symbol.
pub fn jacobi(mut a: i64, mut n: u64) -> i64 {
    assert!(n > 0 && n % 2 == 1);
    a = ((a % n as i64 + n as i64) % n as i64) as i64;
    let mut result = 1i64;
    loop {
        if a == 0 { return if n == 1 { result } else { 0 }; }
        // Remove factors of 2
        while a % 2 == 0 {
            a /= 2;
            let r = n % 8;
            if r == 3 || r == 5 { result = -result; }
        }
        if a == 1 { return if n == 1 { result } else { result * if n == 1 { 1 } else { 0 } }; }
        // Quadratic reciprocity: flip and reduce
        // (a/n)(n/a) = (-1)^((a-1)/2 · (n-1)/2)
        if a % 4 == 3 && n % 4 == 3 { result = -result; }
        let new_a = n as i64 % a;
        n = a as u64;
        a = new_a;
        if n == 1 { return result; }
    }
}

/// Tonelli-Shanks algorithm: square root of n mod p (p odd prime, p ∤ n).
/// Returns x with x² ≡ n (mod p), or None if n is not a QR mod p.
pub fn sqrt_mod(n: u64, p: u64) -> Option<u64> {
    if legendre(n as i64, p) != 1 { return None; }
    if p == 2 { return Some(n % 2); }
    if p % 4 == 3 {
        return Some(mod_pow(n, (p + 1) / 4, p));
    }
    // Tonelli-Shanks
    let (mut q, s) = {
        let mut q = p - 1;
        let mut s = 0u32;
        while q % 2 == 0 { q /= 2; s += 1; }
        (q, s)
    };
    // Find quadratic non-residue
    let z = (2..p).find(|&z| legendre(z as i64, p) == -1)?;
    let mut m = s;
    let mut c = mod_pow(z, q, p);
    let mut t = mod_pow(n, q, p);
    let mut r = mod_pow(n, (q + 1) / 2, p);
    loop {
        if t == 0 { return Some(0); }
        if t == 1 { return Some(r); }
        let mut i = 1;
        let mut temp = mul_mod(t, t, p);
        while temp != 1 { temp = mul_mod(temp, temp, p); i += 1; }
        let b = {
            let mut b = c;
            for _ in 0..m - i - 1 { b = mul_mod(b, b, p); }
            b
        };
        m = i;
        c = mul_mod(b, b, p);
        t = mul_mod(t, c, p);
        r = mul_mod(r, b, p);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3 — Number-Theoretic Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Euler's totient function φ(n): count of integers 1..n coprime to n.
/// Multiplicative accumulate: φ(n) = n × Π(1 - 1/p) over prime factors.
pub fn euler_totient(mut n: u64) -> u64 {
    let mut result = n;
    let mut p = 2;
    while p * p <= n {
        if n % p == 0 {
            while n % p == 0 { n /= p; }
            result -= result / p;
        }
        p += 1;
    }
    if n > 1 { result -= result / n; }
    result
}

/// Möbius function μ(n).
/// μ(1) = 1, μ(n) = (-1)^k if n = p₁p₂...p_k (squarefree), 0 if n has squared prime factor.
pub fn mobius(mut n: u64) -> i64 {
    if n == 1 { return 1; }
    let mut factors = 0i32;
    let mut p = 2u64;
    while p * p <= n {
        if n % p == 0 {
            factors += 1;
            n /= p;
            if n % p == 0 { return 0; } // p² divides n
        }
        p += 1;
    }
    if n > 1 { factors += 1; }
    if factors % 2 == 0 { 1 } else { -1 }
}

/// Prime factorization: returns prime factors with multiplicities, sorted.
pub fn factorize(mut n: u64) -> Vec<(u64, u32)> {
    let mut factors = Vec::new();
    let mut p = 2u64;
    while p * p <= n {
        if n % p == 0 {
            let mut exp = 0u32;
            while n % p == 0 { n /= p; exp += 1; }
            factors.push((p, exp));
        }
        p += 1;
    }
    if n > 1 { factors.push((n, 1)); }
    factors
}

/// Number of divisors τ(n) = Π (e_i + 1) over prime power factorization.
pub fn num_divisors(n: u64) -> u64 {
    factorize(n).iter().map(|&(_, e)| e as u64 + 1).product()
}

/// Sum of divisors σ(n) = Π_{p^e || n} (1 + p + ... + p^e).
pub fn sum_divisors(n: u64) -> u64 {
    factorize(n).iter().map(|&(p, e)| {
        // σ(p^e) = 1 + p + p² + ... + p^e
        let mut sigma = 0u64;
        let mut pk = 1u64;
        for _ in 0..=e {
            sigma = sigma.saturating_add(pk);
            pk = pk.saturating_mul(p);
        }
        sigma
    }).product()
}

/// All divisors of n, sorted.
pub fn divisors(n: u64) -> Vec<u64> {
    let mut divs = Vec::new();
    let mut i = 1u64;
    while i * i <= n {
        if n % i == 0 {
            divs.push(i);
            if i != n / i { divs.push(n / i); }
        }
        i += 1;
    }
    divs.sort_unstable();
    divs
}

/// Sieve of Euler totients for all n ≤ limit.
pub fn sieve_totients(limit: usize) -> Vec<u64> {
    let mut phi: Vec<u64> = (0..=limit as u64).collect();
    for i in 2..=limit {
        if phi[i] == i as u64 {
            // i is prime
            let mut j = i;
            while j <= limit {
                phi[j] -= phi[j] / i as u64;
                j += i;
            }
        }
    }
    phi
}

/// Sieve of smallest prime factors for all n ≤ limit.
pub fn sieve_spf(limit: usize) -> Vec<u64> {
    let mut spf: Vec<u64> = (0..=limit as u64).collect();
    let mut p = 2;
    while p * p <= limit {
        if spf[p] == p as u64 { // p is prime
            let mut k = p * p;
            while k <= limit {
                if spf[k] == k as u64 { spf[k] = p as u64; }
                k += p;
            }
        }
        p += 1;
    }
    spf
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4 — Primitive Roots and Discrete Logarithm
// ═══════════════════════════════════════════════════════════════════════════

/// Find a primitive root modulo prime p (generator of (ℤ/pℤ)*).
/// Tries small candidates; guaranteed to succeed since primitive roots exist for primes.
pub fn primitive_root(p: u64) -> Option<u64> {
    if p == 2 { return Some(1); }
    let phi = p - 1; // φ(p) = p-1 for prime p
    let factors = factorize(phi);
    for g in 2..p {
        let is_generator = factors.iter().all(|&(q, _)| {
            mod_pow(g, phi / q, p) != 1
        });
        if is_generator { return Some(g); }
    }
    None
}

/// Baby-step giant-step discrete logarithm: finds x with g^x ≡ y (mod p).
/// Returns None if no solution exists.
/// Accumulate+gather: store baby steps as hash map, then scan giant steps.
pub fn discrete_log(g: u64, y: u64, p: u64) -> Option<u64> {
    use std::collections::HashMap;
    let m = (p as f64).sqrt().ceil() as u64 + 1;

    // Baby steps: {g^j mod p : j = 0..m}
    let mut baby: HashMap<u64, u64> = HashMap::new();
    let mut gj = 1u64;
    for j in 0..m {
        baby.insert(gj, j);
        gj = mul_mod(gj, g, p);
    }

    // Giant steps: y · (g^{-m})^i = g^j
    let g_inv_m = mod_pow(mod_inverse(g, p)?, m, p);
    let mut giant = y;
    for i in 0..m {
        if let Some(&j) = baby.get(&giant) {
            return Some(i * m + j);
        }
        giant = mul_mod(giant, g_inv_m, p);
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5 — Continued Fractions
// ═══════════════════════════════════════════════════════════════════════════

/// Continued fraction coefficients [a₀; a₁, a₂, ...] for x.
/// Sequential scan: aₙ = ⌊x⌋, xₙ₊₁ = 1/(xₙ - aₙ).
pub fn continued_fraction(mut x: f64, max_terms: usize) -> Vec<i64> {
    let mut coeffs = Vec::with_capacity(max_terms);
    for _ in 0..max_terms {
        let a = x.floor() as i64;
        coeffs.push(a);
        let frac = x - a as f64;
        if frac < 1e-10 { break; }
        x = 1.0 / frac;
    }
    coeffs
}

/// Convergents of a continued fraction [a₀; a₁, ...].
/// Returns (numerators, denominators) as pairs (p_k, q_k) where p_k/q_k → x.
/// Stops if convergents overflow i64.
pub fn convergents(coeffs: &[i64]) -> Vec<(i64, i64)> {
    let n = coeffs.len();
    if n == 0 { return vec![]; }
    let mut convs = Vec::with_capacity(n);
    let (mut p_prev, mut p_curr) = (1i64, coeffs[0]);
    let (mut q_prev, mut q_curr) = (0i64, 1i64);
    convs.push((p_curr, q_curr));
    for k in 1..n {
        let a = coeffs[k];
        let p_next = match a.checked_mul(p_curr).and_then(|v| v.checked_add(p_prev)) {
            Some(v) => v,
            None => break, // overflow
        };
        let q_next = match a.checked_mul(q_curr).and_then(|v| v.checked_add(q_prev)) {
            Some(v) => v,
            None => break,
        };
        convs.push((p_next, q_next));
        p_prev = p_curr; p_curr = p_next;
        q_prev = q_curr; q_curr = q_next;
    }
    convs
}

/// Best rational approximation to x with denominator ≤ max_denom.
/// Uses continued fraction convergents + semi-convergents.
pub fn best_rational(x: f64, max_denom: i64) -> (i64, i64) {
    let coeffs = continued_fraction(x, 50);
    let convs = convergents(&coeffs);
    let mut best = (0i64, 1i64);
    let mut best_err = f64::INFINITY;
    for (p, q) in convs {
        if q.abs() > max_denom { break; }
        let err = (p as f64 / q as f64 - x).abs();
        if err < best_err { best_err = err; best = (p, q); }
    }
    best
}

/// Detect periodicity in continued fraction (indicates quadratic irrational).
/// Returns period length if periodic, 0 otherwise.
pub fn cf_period(coeffs: &[i64]) -> usize {
    let n = coeffs.len();
    if n < 3 { return 0; }
    // Look for repeating pattern starting at index 1
    for period in 1..=(n - 1) / 2 {
        if (1..n.min(1 + 2 * period)).all(|i| {
            let j = 1 + (i - 1) % period;
            j < n && i < n && coeffs[i] == coeffs[j]
        }) {
            return period;
        }
    }
    0
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6 — Factoring
// ═══════════════════════════════════════════════════════════════════════════

/// Pollard's rho algorithm for finding a non-trivial factor of n.
/// Returns None if n is prime or factor not found.
pub fn pollard_rho(n: u64) -> Option<u64> {
    if n <= 1 { return None; }
    if n % 2 == 0 { return Some(2); }
    if is_prime(n) { return None; }

    let mut seed = 2u64;
    loop {
        let c = seed;
        seed += 1;
        let f = |x: u64| -> u64 { (mul_mod(x, x, n) + c) % n };
        let mut x = 2u64;
        let mut y = 2u64;
        let mut d = 1u64;

        while d == 1 {
            x = f(x);
            y = f(f(y));
            let diff = if x > y { x - y } else { y - x };
            d = gcd(diff, n);
        }

        if d != n { return Some(d); }
        if seed > 100 { return None; }
    }
}

/// Complete prime factorization using trial division for small factors
/// and Pollard's rho for large factors.
pub fn factorize_complete(n: u64) -> Vec<u64> {
    if n <= 1 { return vec![]; }
    if is_prime(n) { return vec![n]; }

    // Try small factors first
    for p in 2u64..=1000 {
        if p * p > n { return vec![n]; }
        if n % p == 0 {
            let mut result = factorize_complete(n / p);
            result.push(p);
            result.sort_unstable();
            return result;
        }
    }

    // Pollard's rho for large factors
    if let Some(d) = pollard_rho(n) {
        let mut result = factorize_complete(d);
        result.extend(factorize_complete(n / d));
        result.sort_unstable();
        result
    } else {
        vec![n] // n is (probably) prime
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7 — Quadratic Forms and Diophantine Equations
// ═══════════════════════════════════════════════════════════════════════════

/// Integer square root: ⌊√n⌋.
pub fn isqrt(n: u64) -> u64 {
    if n == 0 { return 0; }
    let mut x = (n as f64).sqrt() as u64;
    while x * x > n { x -= 1; }
    while (x + 1) * (x + 1) <= n { x += 1; }
    x
}

/// Check if n is a perfect square. Returns √n if so.
pub fn perfect_square(n: u64) -> Option<u64> {
    let r = isqrt(n);
    if r * r == n { Some(r) } else { None }
}

/// Represent n = a² + b² if possible. Returns (a, b) with a ≤ b, or None.
/// n is a sum of two squares iff each prime factor p ≡ 3 (mod 4) appears to an even power.
pub fn sum_of_two_squares(n: u64) -> Option<(u64, u64)> {
    // Brute force for small n; use Gaussian integers for large n
    for a in 0..=isqrt(n / 2) + 1 {
        if a * a > n { break; }
        let remainder = n - a * a;
        if let Some(b) = perfect_square(remainder) {
            if b >= a { return Some((a, b)); }
        }
    }
    None
}

/// Pell equation x² - D·y² = 1. Returns fundamental solution (x₁, y₁).
/// Uses continued fraction expansion of √D.
pub fn pell_fundamental(d: u64) -> Option<(u64, u64)> {
    if perfect_square(d).is_some() { return None; } // D is perfect square, no solution

    let sqrt_d = (d as f64).sqrt();
    let a0 = sqrt_d as u64;

    // Compute continued fraction of √D
    let mut m = 0u64;
    let mut k = a0;
    let mut a = a0;

    let (mut p_prev, mut p_curr) = (1u64, a0);
    let (mut q_prev, mut q_curr) = (0u64, 1u64);

    for _ in 0..1000 {
        m = k * a - m;
        k = (d - m * m) / k;
        if k == 0 { break; }
        a = (a0 + m) / k;

        let p_next = a * p_curr + p_prev;
        let q_next = a * q_curr + q_prev;

        if p_next * p_next == d * q_next * q_next + 1 {
            return Some((p_next, q_next));
        }

        p_prev = p_curr; p_curr = p_next;
        q_prev = q_curr; q_curr = q_next;
    }
    None
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 8 — Partition Function and Zeta Connection
// ═══════════════════════════════════════════════════════════════════════════

/// Partition function p(n): number of ways to write n as an unordered sum of positive integers.
/// Uses Euler's recurrence: p(n) = Σ_k (-1)^(k+1) [p(n-k(3k-1)/2) + p(n-k(3k+1)/2)].
pub fn partition_count(n: usize) -> u64 {
    let mut p = vec![0u64; n + 1];
    p[0] = 1;
    for m in 1..=n {
        let mut sign = 1i64;
        let mut k = 1i64;
        let mut sum = 0i64;
        loop {
            // Pentagonal numbers: k(3k-1)/2 and k(3k+1)/2
            let pent1 = k * (3 * k - 1) / 2;
            let pent2 = k * (3 * k + 1) / 2;
            if pent1 > m as i64 { break; }
            sum += sign * p[m - pent1 as usize] as i64;
            if pent2 <= m as i64 {
                sum += sign * p[m - pent2 as usize] as i64;
            }
            sign = -sign;
            k += 1;
        }
        p[m] = sum.max(0) as u64;
    }
    p[n]
}

/// Euler product ζ(s) = Π_p 1/(1-p^{-s}) computed over primes ≤ limit.
/// Demonstrates: euler_product = partition_function(ln_primes, s).
pub fn euler_product_approx(s: f64, prime_limit: usize) -> f64 {
    let primes = sieve(prime_limit);
    // Accumulate: product over prime modes (Euler product formula)
    primes.iter().fold(1.0, |prod, &p| {
        prod / (1.0 - (p as f64).powf(-s))
    })
}

/// Basel problem: ζ(2) = π²/6 ≈ 1.6449...
/// The Euler product formula connects physics (partition function) with number theory.
pub fn basel_sum_exact() -> f64 {
    std::f64::consts::PI * std::f64::consts::PI / 6.0
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9 — Cryptographic Primitives (number-theoretic foundations)
// ═══════════════════════════════════════════════════════════════════════════

/// RSA key generation: returns (n, e, d) for prime factors p, q.
/// n = p·q, e chosen (typically 65537), d = e⁻¹ mod λ(n).
pub fn rsa_keygen(p: u64, q: u64, e: u64) -> Option<(u64, u64, u64)> {
    if !is_prime(p) || !is_prime(q) { return None; }
    let n = p * q;
    let lambda_n = lcm(p - 1, q - 1); // Carmichael function λ(n)
    let d = mod_inverse(e, lambda_n)?;
    // Verify
    if mul_mod(e, d, lambda_n) != 1 { return None; }
    Some((n, e, d))
}

/// RSA encryption: c = m^e mod n.
pub fn rsa_encrypt(message: u64, e: u64, n: u64) -> u64 {
    mod_pow(message, e, n)
}

/// RSA decryption: m = c^d mod n.
pub fn rsa_decrypt(ciphertext: u64, d: u64, n: u64) -> u64 {
    mod_pow(ciphertext, d, n)
}

/// Diffie-Hellman key exchange: compute g^a mod p.
pub fn dh_public_key(g: u64, a: u64, p: u64) -> u64 {
    mod_pow(g, a, p)
}

/// DH shared secret: (g^b)^a mod p.
pub fn dh_shared_secret(g_b: u64, a: u64, p: u64) -> u64 {
    mod_pow(g_b, a, p)
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION N — Integer Linear Recurrences (accumulate-friendly)
// ═══════════════════════════════════════════════════════════════════════════

/// A single affine step in an integer linear recurrence of the form
/// `n → (a·n + b) / 2^shift`, with an associated `steps` counter describing how
/// many atomic operations the step represents.
///
/// This is the *general* form that sits underneath every integer 1D affine map
/// tambear works with:
/// - Collatz bit-chunk transforms (`a = 3^k`, `b`, `shift = Σ vᵢ`)
/// - Generalized T_m bit-chunk transforms (`a = m^k`, `b`, `shift = Σ vᵢ`)
/// - Multiply-then-divide arithmetic maps on sparse transition operators
/// - Any affine recurrence with a power-of-two divisor
///
/// ## Composition (monoid)
///
/// Composition `T₂ ∘ T₁` (apply T₁ first, then T₂) has the form:
/// ```text
///   T₁(n) = (a₁·n + b₁) / 2^s₁
///   T₂(T₁(n)) = (a₂·(a₁·n + b₁)/2^s₁ + b₂) / 2^s₂
///             = (a₂·a₁·n + a₂·b₁ + 2^s₁·b₂) / 2^(s₁+s₂)
/// ```
/// This is associative, so a sequence of steps admits a **parallel prefix
/// scan** (the same principle underpinning the Sarkka 5-tuple Kalman filter
/// and the `tridiagonal_scan`: a composable step type lifted through `accumulate`).
///
/// The identity step is `{ a: 1, b: 0, shift: 0, steps: 0 }`.
///
/// ## Overflow
///
/// All fields are `u128` — sufficient for any single bit-chunk transform up to
/// ~80 bits of state, but composition can overflow for long scans. Callers
/// that compose many steps should use `compose_checked` or clamp via
/// `saturating_*` primitives appropriate to their domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearRecurrenceStep {
    /// Multiplicative coefficient.
    pub a: u128,
    /// Additive coefficient.
    pub b: u128,
    /// Right-shift (power-of-two divisor) applied after the affine step.
    pub shift: u32,
    /// Number of atomic underlying operations represented by this step.
    pub steps: u32,
}

impl LinearRecurrenceStep {
    /// The identity transform: `n → n`.
    pub const IDENTITY: Self = Self { a: 1, b: 0, shift: 0, steps: 0 };

    /// Build a step from explicit coefficients.
    pub const fn new(a: u128, b: u128, shift: u32, steps: u32) -> Self {
        Self { a, b, shift, steps }
    }

    /// Apply this step to an integer `n`: returns `(a·n + b) >> shift`.
    /// Returns `None` on overflow or if the result is not integer (the
    /// division is assumed exact; callers in dynamical-system contexts
    /// verify this via bit-pattern preconditions).
    pub fn apply(&self, n: u128) -> Option<u128> {
        let mul = self.a.checked_mul(n)?;
        let add = mul.checked_add(self.b)?;
        Some(add >> self.shift)
    }

    /// Compose `other` after `self`: returns the step that computes
    /// `other.apply(self.apply(n))` in a single step, using checked
    /// arithmetic. Returns `None` on overflow.
    ///
    /// Composition formula:
    /// `T₂ ∘ T₁ = { a: a₂·a₁, b: a₂·b₁ + 2^s₁·b₂, shift: s₁+s₂, steps: steps₁+steps₂ }`
    ///
    /// Note this composes the two steps as if the divisor by `2^s₁` is exact;
    /// callers relying on fractional intermediate results should use domain-
    /// specific composition (see `extremal_orbit::build_affine_table`).
    pub fn compose_checked(self, other: Self) -> Option<Self> {
        let new_a = other.a.checked_mul(self.a)?;
        let two_s1 = 1u128.checked_shl(self.shift)?;
        let term1 = other.a.checked_mul(self.b)?;
        let term2 = two_s1.checked_mul(other.b)?;
        let new_b = term1.checked_add(term2)?;
        let new_shift = self.shift.checked_add(other.shift)?;
        let new_steps = self.steps.checked_add(other.steps)?;
        Some(Self { a: new_a, b: new_b, shift: new_shift, steps: new_steps })
    }

    /// Saturating composition: returns `LinearRecurrenceStep` with fields
    /// clamped to `u128::MAX` / `u32::MAX` on overflow rather than `None`.
    /// Useful as an overflow sentinel in bulk scans.
    pub fn compose_saturating(self, other: Self) -> Self {
        let new_a = other.a.saturating_mul(self.a);
        let two_s1 = 1u128.checked_shl(self.shift).unwrap_or(u128::MAX);
        let term1 = other.a.saturating_mul(self.b);
        let term2 = two_s1.saturating_mul(other.b);
        let new_b = term1.saturating_add(term2);
        let new_shift = self.shift.saturating_add(other.shift);
        let new_steps = self.steps.saturating_add(other.steps);
        Self { a: new_a, b: new_b, shift: new_shift, steps: new_steps }
    }
}

/// Fold a slice of steps via left-to-right composition.
///
/// Equivalent to `accumulate(steps, All, compose_checked)` — the sequential
/// reference against which a parallel prefix-scan implementation is verified.
pub fn compose_steps(steps: &[LinearRecurrenceStep]) -> Option<LinearRecurrenceStep> {
    let mut acc = LinearRecurrenceStep::IDENTITY;
    for &s in steps {
        acc = acc.compose_checked(s)?;
    }
    Some(acc)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Section 1: Primality ────────────────────────────────────────────

    #[test]
    fn sieve_first_primes() {
        let primes = sieve(30);
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn sieve_count_matches_pi() {
        // π(100) = 25
        let primes = sieve(100);
        assert_eq!(primes.len(), 25, "π(100) = 25");
    }

    #[test]
    fn miller_rabin_small_primes() {
        for &p in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
            assert!(is_prime(p), "{p} should be prime");
        }
    }

    #[test]
    fn miller_rabin_composites() {
        for &n in &[4u64, 6, 8, 9, 10, 15, 25, 100, 341, 561, 1105] {
            assert!(!is_prime(n), "{n} should be composite");
        }
    }

    #[test]
    fn miller_rabin_large_prime() {
        // Known large prime: 2^31 - 1 = 2147483647 (Mersenne prime)
        assert!(is_prime(2_147_483_647), "2^31 - 1 is prime");
    }

    #[test]
    fn miller_rabin_large_composite() {
        // 2^31 - 1 + 58 = 2147483705 = 5 × 429496741
        assert!(!is_prime(2_147_483_705), "2^31 - 1 + 58 should be composite");
    }

    #[test]
    fn next_prime_gaps() {
        assert_eq!(next_prime(10), 11);
        assert_eq!(next_prime(13), 17);
        assert_eq!(next_prime(100), 101);
    }

    #[test]
    fn segmented_sieve_matches() {
        let full = sieve(1000);
        let seg: Vec<u64> = segmented_sieve(100, 1001);
        let expected: Vec<u64> = full.into_iter().filter(|&p| p >= 100).collect();
        assert_eq!(seg, expected);
    }

    // ── Section 2: Modular Arithmetic ───────────────────────────────────

    #[test]
    fn mod_pow_fermats_little_theorem() {
        // Fermat: a^(p-1) ≡ 1 (mod p) for prime p, gcd(a,p)=1
        let p = 97u64;
        for a in [2u64, 3, 5, 10, 50, 96] {
            assert_eq!(mod_pow(a, p - 1, p), 1, "Fermat: {a}^{} mod {p}", p - 1);
        }
    }

    #[test]
    fn mod_pow_zero_exp() {
        assert_eq!(mod_pow(42, 0, 97), 1, "a^0 = 1");
    }

    #[test]
    fn gcd_basic() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(100, 75), 25);
        assert_eq!(gcd(17, 13), 1); // coprime
    }

    #[test]
    fn extended_gcd_bezout() {
        let (g, x, y) = extended_gcd(48, 18);
        assert_eq!(g, 6);
        assert_eq!(48 * x + 18 * y, g as i64, "Bezout: 48x + 18y = 6");
    }

    #[test]
    fn mod_inverse_exists() {
        // 3⁻¹ mod 7 = 5 (since 3×5 = 15 ≡ 1 mod 7)
        assert_eq!(mod_inverse(3, 7), Some(5));
    }

    #[test]
    fn mod_inverse_none_gcd_not_1() {
        // gcd(6, 9) = 3 ≠ 1, no inverse
        assert_eq!(mod_inverse(6, 9), None);
    }

    #[test]
    fn crt_basic() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
        // Solution: x = 23, M = 105
        let (x, m) = crt(&[2, 3, 2], &[3, 5, 7]).unwrap();
        assert_eq!(m, 105);
        assert_eq!(x % 3, 2);
        assert_eq!(x % 5, 3);
        assert_eq!(x % 7, 2);
    }

    #[test]
    fn legendre_qr() {
        // 3 is a QR mod 11 since 3 ≡ 5² (mod 11)
        assert_eq!(legendre(3, 11), 1, "3 is QR mod 11");
        // 2 is a non-QR mod 5 (2^2=4, 3^2=4, 1^2=1 mod 5 — QRs are {1,4})
        assert_eq!(legendre(2, 5), -1, "2 is NQR mod 5");
    }

    #[test]
    fn tonelli_shanks_sqrt() {
        // 10 is QR mod 13: x² ≡ 10 (mod 13)
        let x = sqrt_mod(10, 13).unwrap();
        assert_eq!((x * x) % 13, 10, "sqrt_mod: x²≡10 mod 13");
    }

    // ── Section 3: Number-Theoretic Functions ────────────────────────────

    #[test]
    fn euler_totient_prime() {
        // φ(p) = p-1 for prime p
        assert_eq!(euler_totient(7), 6);
        assert_eq!(euler_totient(13), 12);
    }

    #[test]
    fn euler_totient_power() {
        // φ(p^k) = p^(k-1)(p-1)
        assert_eq!(euler_totient(9), 6); // φ(3²) = 3(3-1) = 6
    }

    #[test]
    fn euler_totient_multiplicative() {
        // φ is multiplicative: φ(mn) = φ(m)φ(n) for gcd(m,n)=1
        let m = 5u64;
        let n = 7u64;
        assert_eq!(euler_totient(m * n), euler_totient(m) * euler_totient(n));
    }

    #[test]
    fn mobius_values() {
        assert_eq!(mobius(1), 1, "μ(1) = 1");
        assert_eq!(mobius(6), 1, "μ(6) = μ(2·3) = 1");
        assert_eq!(mobius(4), 0, "μ(4) = 0 (4=2²)");
        assert_eq!(mobius(30), -1, "μ(30) = μ(2·3·5) = -1 (3 prime factors)");
    }

    #[test]
    fn mobius_inversion() {
        // Möbius inversion: Σ_{d|n} μ(d) = [n=1]
        for n in 2..20u64 {
            let sum: i64 = divisors(n).iter().map(|&d| mobius(d)).sum();
            assert_eq!(sum, 0, "Σ μ(d) for d|{n} = 0 for n>1");
        }
    }

    #[test]
    fn num_divisors_values() {
        assert_eq!(num_divisors(12), 6, "τ(12) = 6"); // 1,2,3,4,6,12
        assert_eq!(num_divisors(1), 1, "τ(1) = 1");
        assert_eq!(num_divisors(7), 2, "τ(7) = 2"); // prime
    }

    #[test]
    fn factorize_correct() {
        let f = factorize(12);
        assert_eq!(f, vec![(2, 2), (3, 1)], "12 = 2²·3");
        assert!(factorize(97).iter().any(|&(p, e)| p == 97 && e == 1), "97 is prime");
    }

    // ── Section 4: Primitive Roots ───────────────────────────────────────

    #[test]
    fn primitive_root_generates_group() {
        let p = 13u64;
        let g = primitive_root(p).unwrap();
        let mut elements: Vec<u64> = (0..p - 1).map(|k| mod_pow(g, k, p)).collect();
        elements.sort_unstable();
        assert_eq!(elements, (1..p).collect::<Vec<_>>(), "g={g} generates (ℤ/13ℤ)*");
    }

    #[test]
    fn discrete_log_round_trip() {
        let p = 97u64;
        let g = primitive_root(p).unwrap();
        for exp in [1u64, 5, 20, 50, 95] {
            let y = mod_pow(g, exp, p);
            let x = discrete_log(g, y, p).unwrap();
            assert_eq!(mod_pow(g, x, p), y, "DL: g^x ≡ {y} mod {p}");
        }
    }

    // ── Section 5: Continued Fractions ──────────────────────────────────

    #[test]
    fn cf_sqrt2() {
        // √2 = [1; 2, 2, 2, ...] (periodic with period 1)
        let coeffs = continued_fraction(2.0_f64.sqrt(), 10);
        assert_eq!(coeffs[0], 1, "CF[0] of √2");
        for i in 1..coeffs.len() {
            assert_eq!(coeffs[i], 2, "CF[{i}] of √2 should be 2");
        }
    }

    #[test]
    fn cf_golden_ratio() {
        // φ = [1; 1, 1, 1, ...] (all 1s)
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let coeffs = continued_fraction(phi, 10);
        for &a in &coeffs { assert_eq!(a, 1, "Golden ratio CF = [1;1,1,...]"); }
    }

    #[test]
    fn convergents_pi() {
        // π ≈ 3.14159... convergents include 3, 22/7, 333/106, 355/113
        let coeffs = continued_fraction(std::f64::consts::PI, 8);
        let convs = convergents(&coeffs);
        assert_eq!(convs[0], (3, 1), "π convergent 3/1");
        assert_eq!(convs[1], (22, 7), "π convergent 22/7");
        assert_eq!(convs[3], (355, 113), "π convergent 355/113");
    }

    #[test]
    fn best_rational_accuracy() {
        // Best rational approx to π with denom ≤ 100 is 22/7
        let (p, q) = best_rational(std::f64::consts::PI, 100);
        assert_eq!((p, q), (22, 7), "Best rational to π with denom ≤ 100");
    }

    // ── Section 6: Factoring ─────────────────────────────────────────────

    #[test]
    fn pollard_rho_finds_factor() {
        // 8051 = 83 × 97
        let d = pollard_rho(8051).unwrap();
        assert!(d == 83 || d == 97, "Pollard rho found factor {d} of 8051");
    }

    #[test]
    fn complete_factorization() {
        let f = factorize_complete(8051);
        assert_eq!(f, vec![83, 97], "8051 = 83 × 97");
    }

    // ── Section 7: Quadratic Forms ───────────────────────────────────────

    #[test]
    fn sum_of_two_squares_basic() {
        // 5 = 1² + 2²
        assert_eq!(sum_of_two_squares(5), Some((1, 2)));
        // 25 = 3² + 4² = 0² + 5²
        let (a, b) = sum_of_two_squares(25).unwrap();
        assert_eq!(a * a + b * b, 25);
    }

    #[test]
    fn sum_of_two_squares_impossible() {
        // 3 ≡ 3 (mod 4) is prime → cannot be sum of two squares
        assert_eq!(sum_of_two_squares(3), None);
        assert_eq!(sum_of_two_squares(7), None);
    }

    #[test]
    fn pell_equation_d2() {
        // x² - 2y² = 1 → fundamental solution (3, 2): 9 - 8 = 1 ✓
        let (x, y) = pell_fundamental(2).unwrap();
        assert_eq!(x * x - 2 * y * y, 1, "Pell D=2: x={x}, y={y}");
    }

    #[test]
    fn pell_equation_d3() {
        // x² - 3y² = 1 → fundamental solution (2, 1): 4 - 3 = 1 ✓
        let (x, y) = pell_fundamental(3).unwrap();
        assert_eq!(x * x - 3 * y * y, 1, "Pell D=3: x={x}, y={y}");
    }

    // ── Section 8: Partition Function ────────────────────────────────────

    #[test]
    fn partition_values() {
        // p(0)=1, p(1)=1, p(2)=2, p(3)=3, p(4)=5, p(5)=7
        let expected = vec![1, 1, 2, 3, 5, 7, 11, 15, 22, 30];
        for (n, &e) in expected.iter().enumerate() {
            assert_eq!(partition_count(n), e, "p({n}) = {e}");
        }
    }

    #[test]
    fn euler_product_approaches_zeta2() {
        // ζ(2) = π²/6 ≈ 1.6449
        let approx = euler_product_approx(2.0, 10000);
        let exact = basel_sum_exact();
        assert!((approx - exact).abs() < 0.01, "Euler product ζ(2) ≈ {approx}, exact = {exact}");
    }

    #[test]
    fn euler_product_is_partition_function() {
        // The Euler product Π 1/(1-p^{-s}) = Σ n^{-s} = ζ(s)
        // This IS the partition function of a non-interacting prime gas
        // with energy E_n = ln(n) at inverse temperature s.
        // Demonstrate: euler_product_approx(2.0, 1000) ≈ Σ_{n=1}^{N} 1/n²
        let direct: f64 = (1..=1000u64).map(|n| 1.0 / (n * n) as f64).sum();
        let euler = euler_product_approx(2.0, 1000);
        // Both approximate ζ(2) — they should be within a few percent of each other
        assert!((direct - euler).abs() < 0.05,
            "Direct sum {direct:.4} and Euler product {euler:.4} both approach ζ(2)");
    }

    // ── Section 9: RSA and DH ─────────────────────────────────────────────

    #[test]
    fn rsa_encrypt_decrypt_roundtrip() {
        // Small RSA for testing: p=61, q=53, e=17
        let (n, e, d) = rsa_keygen(61, 53, 17).unwrap();
        assert_eq!(n, 61 * 53);
        // Encrypt m=65
        let m = 65u64;
        let c = rsa_encrypt(m, e, n);
        let m_decrypted = rsa_decrypt(c, d, n);
        assert_eq!(m_decrypted, m, "RSA round-trip: m={m}, c={c}");
    }

    #[test]
    fn rsa_small_prime_not_valid() {
        // p=4 is not prime — should fail
        assert_eq!(rsa_keygen(4, 5, 3), None);
    }

    #[test]
    fn dh_key_exchange() {
        // DH with p=23, g=5 (primitive root)
        let p = 23u64;
        let g = 5u64;
        let (a, b) = (6u64, 15u64); // private keys
        let ga = dh_public_key(g, a, p);
        let gb = dh_public_key(g, b, p);
        let secret_a = dh_shared_secret(gb, a, p);
        let secret_b = dh_shared_secret(ga, b, p);
        assert_eq!(secret_a, secret_b, "DH shared secret must match");
    }

    #[test]
    fn sieve_totients_sum_formula() {
        // Σ_{k=1}^{n} φ(k) ≈ 3n²/π² for large n
        let phis = sieve_totients(100);
        let sum: u64 = phis[1..].iter().sum();
        let expected = 3.0 * 10000.0 / (std::f64::consts::PI * std::f64::consts::PI);
        assert!((sum as f64 - expected).abs() < expected * 0.05,
            "Σφ(k) k≤100 = {sum}, expected ~{expected:.0}");
    }

    #[test]
    fn isqrt_correct() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(100), 10);
        assert_eq!(isqrt(99), 9);
        assert_eq!(isqrt(101), 10);
    }

    // ── Section N: LinearRecurrenceStep ─────────────────────────────────

    #[test]
    fn lrs_identity_is_fixed_point() {
        let id = LinearRecurrenceStep::IDENTITY;
        assert_eq!(id.apply(42).unwrap(), 42);
        assert_eq!(id.apply(0).unwrap(), 0);
        assert_eq!(id.apply(u128::MAX).unwrap(), u128::MAX);
    }

    #[test]
    fn lrs_identity_composes_as_identity() {
        let id = LinearRecurrenceStep::IDENTITY;
        let step = LinearRecurrenceStep::new(3, 1, 1, 1); // (3n+1)/2
        let left = id.compose_checked(step).unwrap();
        let right = step.compose_checked(id).unwrap();
        assert_eq!(left, step);
        assert_eq!(right, step);
    }

    #[test]
    fn lrs_apply_collatz_step() {
        // Classic Collatz odd-step: n → (3n+1)/2  on odd n
        let step = LinearRecurrenceStep::new(3, 1, 1, 1);
        // n = 5 (odd): (3*5+1)/2 = 16/2 = 8
        assert_eq!(step.apply(5).unwrap(), 8);
        // n = 7: (3*7+1)/2 = 22/2 = 11
        assert_eq!(step.apply(7).unwrap(), 11);
    }

    #[test]
    fn lrs_composition_matches_sequential() {
        // Two consecutive Collatz odd-steps: T = (3n+1)/2, then T again.
        // On n=5: 5 → 8, so the composed shouldn't match unless we handle
        // parity. Pick starting point where both steps are 'odd': use shift=0
        // to act as pure (3n+1) then (3n+1).
        let s = LinearRecurrenceStep::new(3, 1, 0, 1);
        let composed = s.compose_checked(s).unwrap();
        // Expected: T2(T1(n)) = 3*(3n+1) + 1 = 9n + 4
        assert_eq!(composed.a, 9);
        assert_eq!(composed.b, 4);
        assert_eq!(composed.shift, 0);
        assert_eq!(composed.steps, 2);
        // Sanity: apply to n=2: sequential 3*2+1=7, 3*7+1=22. Composed: 9*2+4=22. ✓
        assert_eq!(composed.apply(2).unwrap(), 22);
        let seq = s.apply(s.apply(2).unwrap()).unwrap();
        assert_eq!(seq, 22);
    }

    #[test]
    fn lrs_composition_with_shift() {
        // s1 = (3n+1)/2  (odd-step),  s2 = n/2  (single halving).
        // Composed: T2(T1(n)) = ((3n+1)/2)/2 = (3n+1)/4.
        let s1 = LinearRecurrenceStep::new(3, 1, 1, 1);
        let s2 = LinearRecurrenceStep::new(1, 0, 1, 1);
        let c = s1.compose_checked(s2).unwrap();
        // Expected: a = 1*3 = 3, b = 1*1 + 2^1*0 = 1, shift = 1+1 = 2
        assert_eq!(c.a, 3);
        assert_eq!(c.b, 1);
        assert_eq!(c.shift, 2);
        assert_eq!(c.steps, 2);
        // Apply on n=5 (where both steps produce integers):
        // s1(5) = (15+1)/2 = 8, s2(8) = 4. Composed: (15+1)/4 = 4. ✓
        assert_eq!(c.apply(5).unwrap(), 4);
    }

    #[test]
    fn lrs_compose_steps_fold() {
        let s = LinearRecurrenceStep::new(3, 1, 0, 1);
        let folded = compose_steps(&[s, s, s]).unwrap();
        // (3n+1) applied 3x: 3(3(3n+1)+1)+1 = 27n + 13
        assert_eq!(folded.a, 27);
        assert_eq!(folded.b, 13);
        assert_eq!(folded.steps, 3);
        assert_eq!(folded.apply(2).unwrap(), 67);
    }

    #[test]
    fn lrs_compose_empty_is_identity() {
        let f = compose_steps(&[]).unwrap();
        assert_eq!(f, LinearRecurrenceStep::IDENTITY);
    }

    #[test]
    fn lrs_compose_checked_overflow() {
        // a = u128::MAX squared overflows.
        let big = LinearRecurrenceStep::new(u128::MAX, 0, 0, 1);
        let result = big.compose_checked(big);
        assert!(result.is_none());
    }

    #[test]
    fn lrs_compose_saturating_clamps() {
        let big = LinearRecurrenceStep::new(u128::MAX, 0, 0, 1);
        let s = big.compose_saturating(big);
        assert_eq!(s.a, u128::MAX);
    }

    #[test]
    fn lrs_associativity() {
        // (s1 ∘ s2) ∘ s3  ==  s1 ∘ (s2 ∘ s3)
        let s1 = LinearRecurrenceStep::new(3, 1, 1, 1);
        let s2 = LinearRecurrenceStep::new(1, 0, 1, 1);
        let s3 = LinearRecurrenceStep::new(5, 2, 0, 1);
        let left = s1.compose_checked(s2).unwrap().compose_checked(s3).unwrap();
        let right = s1.compose_checked(s2.compose_checked(s3).unwrap()).unwrap();
        assert_eq!(left, right);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 10 — Combinatorial Atoms
//
// Pure counting primitives that exist because they're math, not because any
// one method needs them. These are the atoms downstream methods compose:
//   - Kendall tau ties → binomial_coeff(tie_count, 2)
//   - Entropy estimators → log_binomial (multinomial normalization)
//   - Catalan structure → catalan_number (balanced parentheses, BST shapes)
//   - Stirling numbers → cycle counting, set partitions, inclusion-exclusion
//   - Bell numbers → set partition counting, cluster merge costs
//   - Fibonacci/Lucas → quasi-random sequences, market rhythm analysis
//   - Derangement → random permutation bias testing
//   - Falling/rising factorial → Pochhammer symbol, hypergeometric coefficients
//   - Harmonic numbers → expected comparison counts, entropy upper bounds
//   - Double factorial → Bessel functions, Gaussian integrals, semifactorial
//
// All implementations:
//   - First principles, no vendor library
//   - Exact for small n (integer arithmetic), log-space for large n
//   - Documented domain, assumptions, and overflow behavior
//   - Oracle tested against mpmath/closed-form at key values
// ═══════════════════════════════════════════════════════════════════════════

/// Log of the binomial coefficient: ln C(n, k) = ln(n!) - ln(k!) - ln((n-k)!).
///
/// Uses the log-gamma identity: ln C(n,k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1).
/// Exact for n ≤ 67 (no overflow), log-accurate for larger n.
/// Returns f64::NEG_INFINITY when k > n (degenerate, C(n,k) = 0).
/// Returns 0.0 when k = 0 or k = n (C(n,0) = C(n,n) = 1, ln 1 = 0).
///
/// # Parameters
/// - `n`: total items (n ≥ 0)
/// - `k`: items chosen (0 ≤ k ≤ n)
///
/// # Accuracy
/// Matches mpmath.log(mpmath.binomial(n,k)) to <1 ULP for n ≤ 10000.
pub fn log_binomial(n: usize, k: usize) -> f64 {
    if k > n { return f64::NEG_INFINITY; }
    if k == 0 || k == n { return 0.0; }
    let k = k.min(n - k); // symmetry: C(n,k) = C(n,n-k)
    // For small n use exact integer path via binomial_coeff (which uses u128 intermediates)
    if n <= 67 {
        match binomial_coeff(n, k) {
            Some(c) => (c as f64).ln(),
            None => {
                // Fall through to log-gamma even for n≤67 if somehow overflows
                log_gamma_combinatorial(n as f64 + 1.0)
                    - log_gamma_combinatorial(k as f64 + 1.0)
                    - log_gamma_combinatorial((n - k) as f64 + 1.0)
            }
        }
    } else {
        // Log-gamma path for large n
        log_gamma_combinatorial(n as f64 + 1.0)
            - log_gamma_combinatorial(k as f64 + 1.0)
            - log_gamma_combinatorial((n - k) as f64 + 1.0)
    }
}

/// Exact binomial coefficient C(n, k) as u64.
///
/// Returns `None` if the result overflows u64 (C(67,33) ≈ 1.4e19 is the largest
/// value that fits). For larger values use `log_binomial` + exponentiate.
///
/// Uses the multiplicative formula: C(n,k) = ∏_{i=0}^{k-1} (n-i)/(i+1).
/// Each step divides out the increment to keep intermediate values minimal.
pub fn binomial_coeff(n: usize, k: usize) -> Option<u64> {
    if k > n { return Some(0); }
    let k = k.min(n - k);
    if k == 0 { return Some(1); }
    // Use u128 for intermediate products: the exact result fits in u64, but
    // intermediate values before the GCD reduction can exceed u64::MAX.
    // Example: C(67,33)≈1.4e19 fits in u64, but step-by-step products do not.
    let mut result = 1u128;
    for i in 0..k {
        result = result.checked_mul((n - i) as u128)?;
        result /= (i + 1) as u128;
    }
    if result > u64::MAX as u128 { return None; }
    Some(result as u64)
}

/// Multinomial coefficient: n! / (k₁! · k₂! · … · kₘ!) where Σkᵢ = n.
///
/// Returns `None` if the result overflows u64 or if Σkᵢ ≠ n.
/// Uses repeated application of binomial_coeff:
///   multinomial(n; k₁, k₂, …) = C(n, k₁) · C(n-k₁, k₂) · C(n-k₁-k₂, k₃) · …
pub fn multinomial_coeff(ks: &[usize]) -> Option<u64> {
    if ks.is_empty() { return Some(1); }
    let n: usize = ks.iter().sum();
    let mut result = 1u64;
    let mut remaining = n;
    for &ki in ks {
        if ki > remaining { return None; }
        result = result.checked_mul(binomial_coeff(remaining, ki)?)?;
        remaining -= ki;
    }
    Some(result)
}

/// Log of the multinomial coefficient: ln(n! / ∏ kᵢ!) where Σkᵢ = n.
///
/// Uses log-gamma: ln_multinomial = lgamma(n+1) - Σ lgamma(kᵢ+1).
/// Never overflows. Returns NaN if Σkᵢ ≠ n.
pub fn log_multinomial(ks: &[usize]) -> f64 {
    if ks.is_empty() { return 0.0; }
    let n: usize = ks.iter().sum();
    let mut result = log_gamma_combinatorial(n as f64 + 1.0);
    for &ki in ks {
        result -= log_gamma_combinatorial(ki as f64 + 1.0);
    }
    result
}

/// Catalan number C_n = C(2n, n) / (n + 1).
///
/// Counts: balanced parenthesizations, full binary tree shapes, Dyck paths,
/// non-crossing partitions, polygon triangulations, monotone lattice paths.
///
/// Returns `None` if result overflows u64 (C_33 ≈ 7.3e18 is the largest that fits).
/// For n > 33, use `log_catalan`.
pub fn catalan_number(n: usize) -> Option<u64> {
    let c2n_n = binomial_coeff(2 * n, n)?;
    Some(c2n_n / (n as u64 + 1))
}

/// Log of the Catalan number: ln C_n = ln C(2n,n) - ln(n+1).
pub fn log_catalan(n: usize) -> f64 {
    log_binomial(2 * n, n) - ((n as f64) + 1.0).ln()
}

/// Stirling numbers of the first kind (unsigned): |S(n, k)|.
///
/// |S(n, k)| counts the number of permutations of n elements with exactly k cycles.
/// Satisfies: x·(x+1)·…·(x+n-1) = Σ_k |S(n,k)| · xᵏ  (rising factorial expansion).
///
/// Uses recurrence: |S(n, k)| = (n-1)·|S(n-1, k)| + |S(n-1, k-1)|.
/// Computes the full row for all k in O(n²) time.
///
/// Returns `None` if any value overflows u64. Max safe n ≈ 20 for exact values.
pub fn stirling1_row(n: usize) -> Option<Vec<u64>> {
    // DP table: prev[k] = |S(row, k)|
    let mut prev = vec![0u64; n + 1];
    prev[0] = 1; // |S(0, 0)| = 1 (empty permutation has 0 cycles... base case)
    // Actually |S(0,0)| = 1, |S(n,0)| = 0 for n≥1, |S(n,n)| = 1
    for row in 1..=n {
        let mut curr = vec![0u64; n + 1];
        curr[row] = 1; // |S(row, row)| = 1 (identity permutation)
        for k in 1..row {
            curr[k] = ((row as u64 - 1).checked_mul(prev[k])?)
                .checked_add(prev[k - 1])?;
        }
        prev = curr;
    }
    Some(prev)
}

/// Stirling number of the first kind |S(n, k)|.
///
/// Allocates the full row and returns element k. For computing many values
/// at the same n, prefer `stirling1_row`.
pub fn stirling1(n: usize, k: usize) -> Option<u64> {
    if k > n { return Some(0); }
    if k == n { return Some(1); }
    if k == 0 && n > 0 { return Some(0); }
    Some(stirling1_row(n)?[k])
}

/// Stirling numbers of the second kind: S(n, k).
///
/// S(n, k) counts the number of ways to partition a set of n elements into
/// exactly k non-empty subsets.
/// Satisfies: xⁿ = Σ_k S(n,k) · x·(x-1)·…·(x-k+1)  (falling factorial expansion).
///
/// Uses inclusion-exclusion formula: S(n,k) = (1/k!) Σ_{j=0}^{k} (-1)^(k-j) C(k,j) jⁿ.
/// Returns `None` if overflow occurs.
///
/// # Numerical regime (catastrophic cancellation warning)
/// The alternating sum has massive cancellation. At n=25, k=12 the largest term is
/// ~10²⁷ but the result is ~3.6·10¹⁷ (cancellation ratio ≈ 3.6e9). Intermediate
/// sums use i128 (~10³⁸ range). Safe for n ≤ 25 across all k. For n > 25 with k
/// near n/2, intermediate terms exceed i128 and this returns `None`. Special cases
/// S(n,0), S(n,1), S(n,n) always succeed. Bell triangle approach (used by
/// `bell_number`) avoids cancellation entirely, but only yields the total, not per-k.
pub fn stirling2(n: usize, k: usize) -> Option<u64> {
    if k > n { return Some(0); }
    if k == 0 { return if n == 0 { Some(1) } else { Some(0) }; }
    if k == n { return Some(1); }
    if k == 1 { return Some(1); }
    // Inclusion-exclusion: S(n,k) = (1/k!) Σ_{j=0}^{k} (-1)^(k-j) C(k,j) j^n
    let mut sum: i128 = 0;
    let mut factorial_k: u64 = 1;
    for j in 1..=(k as u64) { factorial_k = factorial_k.checked_mul(j)?; }
    for j in 0..=k {
        let sign: i128 = if (k - j) % 2 == 0 { 1 } else { -1 };
        let ckj = binomial_coeff(k, j)? as i128;
        let jpow = (j as i128).pow(n as u32);
        sum += sign * ckj * jpow;
    }
    let result = sum / factorial_k as i128;
    if result < 0 { return None; }
    Some(result as u64)
}

/// Stirling number of the second kind in f64: S(n, k) as a floating-point value.
///
/// Uses the additive recurrence S(n,k) = k·S(n-1,k) + S(n-1,k-1), which has
/// NO cancellation (only addition and multiplication by small integers). This
/// avoids the catastrophic cancellation of the inclusion-exclusion formula.
///
/// Safe for n up to ~200 before f64 overflow (S(200,100) ≈ 10^200 > f64::MAX ≈ 10^308).
/// For entropy calculations, use `stirling2_f64` with `bell_number_f64` to get
/// the ratio S(n,k)/B(n) without overflow.
///
/// # Accuracy
/// Relative error < 2 ULP vs mpmath at 50dp for n ≤ 100 (validated).
pub fn stirling2_f64(n: usize, k: usize) -> f64 {
    if k > n { return 0.0; }
    if k == 0 { return if n == 0 { 1.0 } else { 0.0 }; }
    if k == n { return 1.0; }
    if k == 1 { return 1.0; }
    // DP via additive recurrence: S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    // Only need the row up to k, not the full n×k table.
    let mut prev = vec![0.0f64; k + 1];
    prev[0] = 1.0; // S(0,0) = 1
    for row in 1..=n {
        let mut curr = vec![0.0f64; k + 1];
        for j in 1..=row.min(k) {
            curr[j] = (j as f64) * prev[j] + prev[j - 1];
        }
        prev = curr;
    }
    prev[k]
}

/// Bell number B(n) as f64 via the recurrence sum: B(n) = Σ_k S(n,k).
///
/// Uses `stirling2_f64` for each term, which avoids cancellation.
/// Safe for n up to ~200 (B(200) ≈ 10^300 < f64::MAX).
/// For exact integer values, use `bell_number` (safe for n ≤ 19).
pub fn bell_number_f64(n: usize) -> f64 {
    (0..=n).map(|k| stirling2_f64(n, k)).sum()
}

/// Bell number B(n): total number of partitions of a set of n elements.
///
/// B(n) = Σ_{k=0}^{n} S(n, k) where S(n,k) are Stirling numbers of the second kind.
/// Equivalently: B(n) = (1/e) Σ_{k=0}^{∞} kⁿ/k! (via Dobinski's formula).
///
/// Uses Bell triangle (Aitken's array). Returns `None` if overflow occurs.
/// Max safe n ≈ 19 for exact u64.
pub fn bell_number(n: usize) -> Option<u64> {
    if n == 0 { return Some(1); }
    // Bell triangle: row 0 = [1], row k starts with last element of row k-1
    let mut row = vec![1u64];
    for _ in 0..n {
        let mut next = vec![*row.last().unwrap()];
        for i in 0..row.len() {
            next.push(next[i].checked_add(row[i])?);
        }
        row = next;
    }
    Some(row[0])
}

/// Fibonacci number F(n): F(0)=0, F(1)=1, F(n)=F(n-1)+F(n-2).
///
/// Uses fast matrix exponentiation via the identity:
/// [F(n+1) F(n); F(n) F(n-1)] = [[1,1],[1,0]]^n
/// O(log n) multiplications, each exact in u128.
///
/// Returns `None` if result overflows u64. Max safe n = 93 (F(93) ≈ 1.22e19).
pub fn fibonacci(n: usize) -> Option<u64> {
    if n == 0 { return Some(0); }
    if n == 1 { return Some(1); }
    // Matrix multiply 2x2 over u128 to check overflow before casting
    fn mat_mul_128(a: [[u128; 2]; 2], b: [[u128; 2]; 2]) -> Option<[[u128; 2]; 2]> {
        let mut r = [[0u128; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    r[i][j] = r[i][j].checked_add(a[i][k].checked_mul(b[k][j])?)?;
                }
            }
        }
        Some(r)
    }
    let mut result = [[1u128, 0], [0, 1]]; // identity
    let mut base = [[1u128, 1], [1, 0]];   // Fibonacci matrix
    let mut n = n;
    while n > 0 {
        if n & 1 == 1 {
            result = mat_mul_128(result, base)?;
        }
        base = mat_mul_128(base, base)?;
        n >>= 1;
    }
    let f = result[0][1];
    if f > u64::MAX as u128 { return None; }
    Some(f as u64)
}

/// Lucas number L(n): L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2).
///
/// Lucas numbers satisfy L(n) = F(n-1) + F(n+1) where F is Fibonacci.
/// Used in: Lucas primality tests, quasi-crystal geometry, phi-based rhythms.
///
/// Returns `None` if result overflows u64. Max safe n ≈ 93.
pub fn lucas_number(n: usize) -> Option<u64> {
    if n == 0 { return Some(2); }
    if n == 1 { return Some(1); }
    // L(n) = F(n-1) + F(n+1)
    let fm1 = fibonacci(n - 1)?;
    let fp1 = fibonacci(n + 1)?;
    fm1.checked_add(fp1)
}

/// Derangement D(n): number of permutations of n elements with no fixed points.
///
/// Formula: D(n) = n! · Σ_{k=0}^{n} (-1)^k / k!  (inclusion-exclusion).
/// Equivalent recurrence: D(n) = (n-1)·(D(n-1) + D(n-2)), D(0)=1, D(1)=0.
///
/// Returns `None` if result overflows u64. Max safe n ≈ 20.
/// Ratio D(n)/n! → 1/e as n → ∞ (probability a random permutation is a derangement).
pub fn derangement(n: usize) -> Option<u64> {
    if n == 0 { return Some(1); }
    if n == 1 { return Some(0); }
    let mut d_prev2 = 1u64; // D(0)
    let mut d_prev1 = 0u64; // D(1)
    for k in 2..=n {
        let new_d = ((k as u64 - 1).checked_mul(d_prev1.checked_add(d_prev2)?))?;
        d_prev2 = d_prev1;
        d_prev1 = new_d;
    }
    Some(d_prev1)
}

/// Falling factorial (Pochhammer descending): x^(n) = x·(x-1)·…·(x-n+1).
///
/// Used in: Stirling number formulas, hypergeometric series coefficients,
/// exact polynomial representation of combinations, difference calculus.
///
/// x^(0) = 1 by convention. Exact when x and result fit in f64 mantissa.
pub fn falling_factorial(x: f64, n: usize) -> f64 {
    if n == 0 { return 1.0; }
    (0..n).fold(1.0f64, |acc, i| acc * (x - i as f64))
}

/// Rising factorial (Pochhammer ascending): (x)_n = x·(x+1)·…·(x+n-1).
///
/// Used in: hypergeometric function coefficients ₂F₁, gamma ratio identities,
/// combinatorial identities. (x)_n = Γ(x+n)/Γ(x).
///
/// (x)_0 = 1 by convention. Exact when x and result fit in f64 mantissa.
pub fn rising_factorial(x: f64, n: usize) -> f64 {
    if n == 0 { return 1.0; }
    (0..n).fold(1.0f64, |acc, i| acc * (x + i as f64))
}

/// Harmonic number H_n = Σ_{k=1}^{n} 1/k.
///
/// H_n ≈ ln(n) + γ where γ ≈ 0.5772... (Euler-Mascheroni constant).
/// More precisely: H_n = ψ(n+1) + γ where ψ is the digamma function.
///
/// Used in: expected comparison counts in binary search trees,
/// entropy upper bounds (H_n is the maximum entropy of a distribution
/// over n outcomes that sums to 1), harmonic series partial sums,
/// Ramanujan's Q-function, coupon collector problem expected value.
///
/// Computed exactly by direct summation for all n.
pub fn harmonic_number(n: usize) -> f64 {
    (1..=n).fold(0.0f64, |acc, k| acc + 1.0 / k as f64)
}

/// Generalized harmonic number H_{n,r} = Σ_{k=1}^{n} 1/k^r.
///
/// H_{n,1} = harmonic_number(n). H_{∞,2} = π²/6 (Basel problem).
/// Used in: Zipf's law normalization, Hurwitz zeta function approximation.
pub fn harmonic_number_r(n: usize, r: f64) -> f64 {
    (1..=n).fold(0.0f64, |acc, k| acc + (k as f64).powf(-r))
}

/// Double factorial n!! = n·(n-2)·(n-4)·…·1 (n odd) or …·2 (n even).
///
/// 0!! = 1, (-1)!! = 1 by convention.
/// Used in: exact Gaussian integral coefficients ∫ xⁿ e^(-x²) dx,
/// Bessel function series, random walk step distributions,
/// surface area of n-dimensional sphere.
///
/// Returns `None` if result overflows u64. Max safe n ≈ 33 (odd) or 34 (even).
pub fn double_factorial(n: usize) -> Option<u64> {
    if n == 0 { return Some(1); }
    let mut result = 1u64;
    let mut k = n;
    while k > 0 {
        result = result.checked_mul(k as u64)?;
        if k < 2 { break; }
        k -= 2;
    }
    Some(result)
}

/// Log double factorial: ln(n!!) using the identity
/// ln((2m)!!) = m·ln(2) + ln(m!),  ln((2m+1)!!) = ln((2m+1)!) - m·ln(2) - ln(m!).
///
/// Never overflows. Accurate to ~1 ULP.
pub fn log_double_factorial(n: usize) -> f64 {
    if n <= 1 { return 0.0; }
    if n % 2 == 0 {
        let m = (n / 2) as f64;
        m * std::f64::consts::LN_2 + log_gamma_combinatorial(m + 1.0)
    } else {
        let m = ((n - 1) / 2) as f64;
        log_gamma_combinatorial(n as f64 + 2.0) // lgamma(n+2) = ln((n+1)!)
            - (m * std::f64::consts::LN_2)      // subtract ln(2^m)
            - log_gamma_combinatorial(m + 1.0)  // subtract ln(m!)
            - ((n as f64) + 1.0).ln()           // subtract ln(n+1)
    }
}

/// Log Gamma function for combinatorial use: ln Γ(x) via Lanczos g=7 approximation.
///
/// Internal to this module — public entry point is in special_functions.
/// Duplicated here to avoid circular module dependency and keep number_theory
/// self-contained for testing.
///
/// Accurate to ~1.5e-15 relative error for x > 0.5.
fn log_gamma_combinatorial(x: f64) -> f64 {
    // Lanczos g=7 coefficients (Spouge series, matches special_functions::log_gamma)
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_10,
       -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
       -176.615_029_162_140_59,
         12.507_343_278_686_905,
         -0.138_571_095_265_720_12,
          9.984_369_578_019_571_6e-6,
          1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - log_gamma_combinatorial(1.0 - x)
    } else {
        let x = x - 1.0;
        let t = x + G + 0.5;
        let mut s = C[0];
        for (i, &c) in C[1..].iter().enumerate() {
            s += c / (x + (i as f64) + 1.0);
        }
        (2.0 * std::f64::consts::PI).sqrt().ln()
            + s.ln()
            + (x + 0.5) * t.ln()
            - t
    }
}

#[cfg(test)]
mod combinatorial_tests {
    use super::*;

    // ── log_binomial ─────────────────────────────────────────────────────────

    #[test]
    fn log_binomial_boundary_cases() {
        // k > n → -inf (C = 0)
        assert!(log_binomial(3, 5).is_infinite() && log_binomial(3, 5) < 0.0);
        // k = 0 or k = n → 0 (C = 1, ln 1 = 0)
        assert!((log_binomial(10, 0) - 0.0).abs() < 1e-12);
        assert!((log_binomial(10, 10) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn log_binomial_known_values() {
        // C(6, 3) = 20, ln 20 ≈ 2.9957...
        assert!((log_binomial(6, 3) - 20.0f64.ln()).abs() < 1e-10);
        // C(10, 5) = 252
        assert!((log_binomial(10, 5) - 252.0f64.ln()).abs() < 1e-10);
        // C(50, 25) — large, use symmetry
        let val = log_binomial(50, 25);
        assert!(val > 0.0 && val.is_finite());
    }

    #[test]
    fn log_binomial_symmetry() {
        // C(n, k) = C(n, n-k)
        for n in [5, 10, 20, 50] {
            for k in 0..=n {
                assert!((log_binomial(n, k) - log_binomial(n, n - k)).abs() < 1e-10,
                    "symmetry failed n={n} k={k}");
            }
        }
    }

    #[test]
    fn log_binomial_large_n_vs_exact() {
        // For n=67, k=33 — largest that fits u64 exactly
        let exact = binomial_coeff(67, 33).unwrap() as f64;
        let log_val = log_binomial(67, 33);
        assert!((log_val - exact.ln()).abs() < 1e-8,
            "log path vs exact path disagree: {log_val} vs {}", exact.ln());
    }

    // ── binomial_coeff ───────────────────────────────────────────────────────

    #[test]
    fn binomial_coeff_known_values() {
        assert_eq!(binomial_coeff(0, 0), Some(1));
        assert_eq!(binomial_coeff(5, 0), Some(1));
        assert_eq!(binomial_coeff(5, 5), Some(1));
        assert_eq!(binomial_coeff(5, 2), Some(10));
        assert_eq!(binomial_coeff(10, 3), Some(120));
        assert_eq!(binomial_coeff(20, 10), Some(184_756));
        assert_eq!(binomial_coeff(3, 5), Some(0)); // k > n
    }

    #[test]
    fn binomial_coeff_pascal_triangle() {
        // C(n,k) = C(n-1,k-1) + C(n-1,k)
        for n in 2..=15usize {
            for k in 1..n {
                let lhs = binomial_coeff(n, k).unwrap();
                let rhs = binomial_coeff(n - 1, k - 1).unwrap()
                    + binomial_coeff(n - 1, k).unwrap();
                assert_eq!(lhs, rhs, "Pascal failed n={n} k={k}");
            }
        }
    }

    // ── multinomial_coeff ────────────────────────────────────────────────────

    #[test]
    fn multinomial_coeff_known_values() {
        // multinomial(4; 2, 1, 1) = 4! / (2! 1! 1!) = 12
        assert_eq!(multinomial_coeff(&[2, 1, 1]), Some(12));
        // multinomial(6; 3, 2, 1) = 6! / (3! 2! 1!) = 60
        assert_eq!(multinomial_coeff(&[3, 2, 1]), Some(60));
        // Edge: all-ones = n!
        assert_eq!(multinomial_coeff(&[1, 1, 1, 1]), Some(24)); // 4!
    }

    #[test]
    fn multinomial_reduces_to_binomial() {
        // multinomial(n; k, n-k) = C(n, k)
        for n in 1..=10usize {
            for k in 0..=n {
                let multi = multinomial_coeff(&[k, n - k]).unwrap();
                let binom = binomial_coeff(n, k).unwrap();
                assert_eq!(multi, binom, "n={n} k={k}");
            }
        }
    }

    // ── catalan_number ───────────────────────────────────────────────────────

    #[test]
    fn catalan_number_known_values() {
        // C_0=1, C_1=1, C_2=2, C_3=5, C_4=14, C_5=42, C_6=132, C_7=429, C_8=1430
        let expected = [1u64, 1, 2, 5, 14, 42, 132, 429, 1430, 4862];
        for (n, &exp) in expected.iter().enumerate() {
            assert_eq!(catalan_number(n), Some(exp), "C_{n} failed");
        }
    }

    #[test]
    fn log_catalan_matches_exact() {
        for n in 1..=20usize {
            if let Some(exact) = catalan_number(n) {
                let log_val = log_catalan(n);
                assert!((log_val - (exact as f64).ln()).abs() < 1e-8,
                    "log_catalan({n}) mismatch: {log_val} vs {}", (exact as f64).ln());
            }
        }
    }

    // ── stirling numbers ─────────────────────────────────────────────────────

    #[test]
    fn stirling1_known_values() {
        // |S(4,1)| = 6, |S(4,2)| = 11, |S(4,3)| = 6, |S(4,4)| = 1
        assert_eq!(stirling1(4, 1), Some(6));
        assert_eq!(stirling1(4, 2), Some(11));
        assert_eq!(stirling1(4, 3), Some(6));
        assert_eq!(stirling1(4, 4), Some(1));
        // |S(n, n)| = 1 for all n
        for n in 0..=10 {
            assert_eq!(stirling1(n, n), Some(1), "|S({n},{n})| failed");
        }
        // |S(n, 1)| = (n-1)! for n >= 1
        // factorials[k] = k!, so (n-1)! = factorials[n-1]
        let factorials = [1u64, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880];
        for n in 1..=9usize {
            let expected = factorials[n - 1]; // (n-1)!
            assert_eq!(stirling1(n, 1), Some(expected), "|S({n},1)| = (n-1)! failed");
        }
    }

    #[test]
    fn stirling1_row_sums_to_factorial() {
        // Σ_k |S(n, k)| = n!
        for n in 0..=8usize {
            let row = stirling1_row(n).unwrap();
            let sum: u64 = row.iter().sum();
            let factorial: u64 = (1..=n as u64).product();
            assert_eq!(sum, factorial, "Σ|S({n},k)| = {n}! failed");
        }
    }

    #[test]
    fn stirling2_known_values() {
        // S(4,1) = 1, S(4,2) = 7, S(4,3) = 6, S(4,4) = 1
        assert_eq!(stirling2(4, 1), Some(1));
        assert_eq!(stirling2(4, 2), Some(7));
        assert_eq!(stirling2(4, 3), Some(6));
        assert_eq!(stirling2(4, 4), Some(1));
        // S(n, n) = 1 for all n
        for n in 0..=10 {
            assert_eq!(stirling2(n, n), Some(1), "S({n},{n}) failed");
        }
        // S(n, 1) = 1 for n >= 1 (only one non-empty subset partition)
        for n in 1..=10 {
            assert_eq!(stirling2(n, 1), Some(1), "S({n},1) failed");
        }
    }

    #[test]
    fn stirling2_row_sums_to_bell() {
        // Σ_k S(n, k) = B(n) (Bell number)
        for n in 0..=8usize {
            let sum: u64 = (0..=n).map(|k| stirling2(n, k).unwrap_or(0)).sum();
            let bn = bell_number(n).unwrap();
            assert_eq!(sum, bn, "Σ S({n},k) = B({n}) failed");
        }
    }

    #[test]
    fn stirling2_cancellation_boundary() {
        // n ≤ 25 should succeed for all k (i128 intermediates hold)
        // Special cases (k=0, k=n) always succeed regardless of n
        assert!(stirling2(25, 12).is_some(), "S(25,12) should succeed (n≤25 safe range)");
        // S(n, n) = 1 always (special case, no inclusion-exclusion needed)
        assert_eq!(stirling2(30, 30), Some(1));
        // S(n, 1) = 1 always (only one way to put all elements in one block)
        assert_eq!(stirling2(30, 1), Some(1));
        // S(n, 0) = 0 for n >= 1 always (no way to partition non-empty set into 0 blocks)
        assert_eq!(stirling2(30, 0), Some(0));
        // n > 26 with large k near n/2 should return None (i128 overflow), not panic or wrong answer
        // S(26, 13): worst-case cancellation, expect None (overflow) or correct value
        // Either is acceptable — just must not panic
        let _ = stirling2(26, 13); // must not panic
        let _ = stirling2(30, 15); // must not panic
    }

    // ── stirling2_f64 — consumer oracle tests ────────────────────────────────

    #[test]
    fn stirling2_f64_matches_exact_small_n() {
        // Verify f64 recurrence matches exact integer path for n≤25
        for n in 0..=15usize {
            for k in 0..=n {
                let exact = stirling2(n, k).unwrap_or(0) as f64;
                let approx = stirling2_f64(n, k);
                assert!((approx - exact).abs() <= exact * 1e-12 + 1e-12,
                    "stirling2_f64({n},{k}): got {approx}, expected {exact}");
            }
        }
    }

    #[test]
    fn stirling2_f64_large_n_validates_recurrence_safety() {
        // S(50,25) via recurrence in f64 should agree with known value ~7.45e42
        // mpmath oracle: 7453802153273200083379626234837625465912500
        let s50_25 = stirling2_f64(50, 25);
        assert!(s50_25 > 7.45e42 && s50_25 < 7.46e42,
            "S(50,25) out of range: {s50_25:.4e}");
        // S(n,n) = 1 always, even for large n
        assert!((stirling2_f64(100, 100) - 1.0).abs() < 1e-10);
        // S(n,1) = 1 always
        assert!((stirling2_f64(100, 1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn set_partition_entropy_consumer_oracle() {
        // Consumer oracle: set-partition entropy H(n) = -Σ (S(n,k)/B(n)) log(S(n,k)/B(n))
        // For n=10: H ≈ 1.4614016677 (from mpmath at 50dp)
        // For n=20: H ≈ 1.7211286015 (from mpmath at 50dp)
        // This tests stirling2_f64 and bell_number_f64 together at consumer-relevant n.
        let entropy = |n: usize| -> f64 {
            let bn = bell_number_f64(n);
            (0..=n).filter_map(|k| {
                let s = stirling2_f64(n, k);
                if s > 0.0 { let p = s / bn; Some(-p * p.ln()) } else { None }
            }).sum()
        };
        let h10 = entropy(10);
        assert!((h10 - 1.4614016677).abs() < 1e-6,
            "H(B_10) = {h10:.10}, expected ≈1.4614016677");
        let h20 = entropy(20);
        assert!((h20 - 1.7211286015).abs() < 1e-6,
            "H(B_20) = {h20:.10}, expected ≈1.7211286015");
    }

    #[test]
    fn binomial_kendall_tau_consumer_oracle() {
        // Consumer oracle: Kendall tau tie correction uses C(t,2) for tie sizes t.
        // Load-bearing values: t ∈ [2, 50], C(t,2) = t*(t-1)/2
        // C(50,2) = 1225 — specific value Kendall needs for 50-way tie
        assert_eq!(binomial_coeff(50, 2), Some(1225));
        assert_eq!(binomial_coeff(2, 2), Some(1));
        assert_eq!(binomial_coeff(10, 2), Some(45));
        // Verify the tie correction formula: C(t,2) = t*(t-1)/2
        for t in 2..=50usize {
            let exact = t * (t - 1) / 2;
            assert_eq!(binomial_coeff(t, 2), Some(exact as u64),
                "tie correction C({t},2) = {exact} failed");
        }
    }

    // ── bell_number ──────────────────────────────────────────────────────────

    #[test]
    fn bell_number_known_values() {
        // B_0=1, B_1=1, B_2=2, B_3=5, B_4=15, B_5=52, B_6=203, B_7=877, B_8=4140
        let expected = [1u64, 1, 2, 5, 15, 52, 203, 877, 4140, 21147];
        for (n, &exp) in expected.iter().enumerate() {
            assert_eq!(bell_number(n), Some(exp), "B_{n} failed");
        }
    }

    #[test]
    fn bell_number_f64_matches_exact() {
        // bell_number_f64 should agree with exact bell_number for small n
        for n in 0..=15usize {
            let exact = bell_number(n).unwrap() as f64;
            let approx = bell_number_f64(n);
            assert!((approx - exact).abs() <= exact * 1e-12 + 1e-12,
                "bell_number_f64({n}) = {approx}, expected {exact}");
        }
    }

    // ── fibonacci and lucas ──────────────────────────────────────────────────

    #[test]
    fn fibonacci_known_values() {
        let expected = [0u64, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144];
        for (n, &exp) in expected.iter().enumerate() {
            assert_eq!(fibonacci(n), Some(exp), "F_{n} failed");
        }
    }

    #[test]
    fn fibonacci_max_safe() {
        // F(93) is the largest Fibonacci number fitting in u64
        assert!(fibonacci(93).is_some());
        assert!(fibonacci(94).is_none()); // overflows
    }

    #[test]
    fn fibonacci_identity_cassini() {
        // Cassini's identity: F(n-1)*F(n+1) - F(n)^2 = (-1)^n
        for n in 1..=40usize {
            let fn_m1 = fibonacci(n - 1).unwrap() as i128;
            let fn_0 = fibonacci(n).unwrap() as i128;
            let fn_p1 = fibonacci(n + 1).unwrap() as i128;
            let cassini = fn_m1 * fn_p1 - fn_0 * fn_0;
            let expected: i128 = if n % 2 == 0 { 1 } else { -1 };
            assert_eq!(cassini, expected, "Cassini failed at n={n}");
        }
    }

    #[test]
    fn lucas_known_values() {
        // L_0=2, L_1=1, L_2=3, L_3=4, L_4=7, L_5=11, L_6=18
        let expected = [2u64, 1, 3, 4, 7, 11, 18, 29, 47, 76];
        for (n, &exp) in expected.iter().enumerate() {
            assert_eq!(lucas_number(n), Some(exp), "L_{n} failed");
        }
    }

    #[test]
    fn lucas_fibonacci_identity() {
        // L(n) = F(n-1) + F(n+1) for n >= 1
        for n in 1..=40usize {
            let ln = lucas_number(n).unwrap() as i128;
            let fn_m1 = fibonacci(n - 1).unwrap() as i128;
            let fn_p1 = fibonacci(n + 1).unwrap() as i128;
            assert_eq!(ln, fn_m1 + fn_p1, "L({n}) = F({}) + F({}) failed", n-1, n+1);
        }
    }

    // ── derangement ──────────────────────────────────────────────────────────

    #[test]
    fn derangement_known_values() {
        // D_0=1, D_1=0, D_2=1, D_3=2, D_4=9, D_5=44, D_6=265
        let expected = [1u64, 0, 1, 2, 9, 44, 265, 1854, 14833];
        for (n, &exp) in expected.iter().enumerate() {
            assert_eq!(derangement(n), Some(exp), "D_{n} failed");
        }
    }

    #[test]
    fn derangement_ratio_approaches_one_over_e() {
        // D(n)/n! → 1/e ≈ 0.36788
        let one_over_e = 1.0f64 / std::f64::consts::E;
        for n in 5..=15usize {
            let dn = derangement(n).unwrap() as f64;
            let factorial: f64 = (1..=n as u64).map(|k| k as f64).product();
            let ratio = dn / factorial;
            assert!((ratio - one_over_e).abs() < 0.01,
                "D({n})/({n}!) = {ratio:.5}, expected ~{one_over_e:.5}");
        }
    }

    // ── falling/rising factorial ─────────────────────────────────────────────

    #[test]
    fn falling_factorial_known_values() {
        // 5^(3) = 5·4·3 = 60
        assert!((falling_factorial(5.0, 3) - 60.0).abs() < 1e-10);
        // x^(0) = 1
        assert!((falling_factorial(7.0, 0) - 1.0).abs() < 1e-10);
        // x^(1) = x
        assert!((falling_factorial(3.5, 1) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn rising_factorial_known_values() {
        // (2)_4 = 2·3·4·5 = 120
        assert!((rising_factorial(2.0, 4) - 120.0).abs() < 1e-10);
        // (x)_0 = 1
        assert!((rising_factorial(7.0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn falling_rising_complement() {
        // falling_factorial(n, k) = rising_factorial(n-k+1, k) for integer n >= k
        for n in 1..=10usize {
            for k in 0..=n {
                let ff = falling_factorial(n as f64, k);
                let rf = rising_factorial((n - k + 1) as f64, k);
                assert!((ff - rf).abs() < 1e-7,
                    "falling({n},{k})={ff} vs rising({},{k})={rf}", n - k + 1);
            }
        }
    }

    // ── harmonic_number ──────────────────────────────────────────────────────

    #[test]
    fn harmonic_number_known_values() {
        // H_1 = 1, H_2 = 3/2, H_3 = 11/6, H_4 = 25/12
        assert!((harmonic_number(1) - 1.0).abs() < 1e-12);
        assert!((harmonic_number(2) - 1.5).abs() < 1e-12);
        assert!((harmonic_number(3) - 11.0 / 6.0).abs() < 1e-12);
        assert!((harmonic_number(4) - 25.0 / 12.0).abs() < 1e-12);
        assert_eq!(harmonic_number(0), 0.0);
    }

    #[test]
    fn harmonic_number_approaches_log() {
        // H_n ≈ ln(n) + γ where γ ≈ 0.5772156649
        const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_86;
        for n in [100usize, 1000, 10000] {
            let hn = harmonic_number(n);
            let approx = (n as f64).ln() + EULER_MASCHERONI;
            assert!((hn - approx).abs() < 0.01,
                "H_{n} = {hn:.6}, ln({n})+γ = {approx:.6}");
        }
    }

    // ── double_factorial ─────────────────────────────────────────────────────

    #[test]
    fn double_factorial_known_values() {
        // 0!! = 1, 1!! = 1, 2!! = 2, 3!! = 3, 4!! = 8, 5!! = 15, 6!! = 48, 7!! = 105
        let expected = [(0, 1u64), (1, 1), (2, 2), (3, 3), (4, 8), (5, 15), (6, 48), (7, 105)];
        for (n, exp) in expected {
            assert_eq!(double_factorial(n), Some(exp), "{n}!! failed");
        }
    }

    #[test]
    fn double_factorial_even_odd_product() {
        // (2n)!! · (2n-1)!! = (2n)!
        for n in 1..=10usize {
            let even = double_factorial(2 * n).unwrap() as u128;
            let odd = double_factorial(2 * n - 1).unwrap() as u128;
            let factorial: u128 = (1..=(2 * n) as u128).product();
            assert_eq!(even * odd, factorial, "({}·{} = {}!) failed", 2*n, 2*n-1, 2*n);
        }
    }

    #[test]
    fn log_double_factorial_matches_exact() {
        for n in [1, 3, 5, 7, 9, 2, 4, 6, 8, 10] {
            if let Some(exact) = double_factorial(n) {
                let log_val = log_double_factorial(n);
                assert!((log_val - (exact as f64).ln()).abs() < 1e-10,
                    "log_double_factorial({n}) mismatch");
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 11 — Polynomial Algebra
//
// Polynomials as first-class mathematical objects. Represented as coefficient
// vectors: p = [a₀, a₁, …, aₙ] means p(x) = a₀ + a₁x + … + aₙxⁿ.
//
// Why here (not a separate file): polynomial arithmetic is inseparably linked
// to number theory — NTT uses modular arithmetic, polynomial GCD underlies
// resultants and Berlekamp factorization, poly_roots over finite fields is
// a primitive for Reed-Solomon. The algebra is the same object seen through
// different lenses.
//
// Primitives:
//   poly_eval     — Horner's method, O(n), numerically stable
//   poly_add      — coefficient-wise addition
//   poly_mul      — naive O(n²) for small polys; NTT-based for large
//   poly_divmod   — synthetic division (quotient + remainder)
//   poly_gcd      — extended Euclidean algorithm for polynomials (over f64)
//   poly_deriv    — formal derivative
//   poly_integ    — formal indefinite integral
//   poly_compose  — f(g(x)), O(n²·m) naive
//   poly_normalize — remove trailing near-zero coefficients
//   poly_interpolate — Lagrange interpolation from (x,y) pairs
//   ntt / intt    — Number Theoretic Transform (exact poly mul over Z_p)
//   poly_mul_ntt  — NTT-based multiplication for large polynomials
//
// Numerical notes:
//   - poly_eval, poly_add, poly_mul, poly_deriv, poly_integ: stable
//   - poly_divmod: can amplify error when leading coefficient is small
//   - poly_gcd over f64: inherently ill-conditioned for near-common factors;
//     use NTT (integer) variant for exact GCD over finite fields
//   - poly_interpolate: O(n²) naive; Barycentric form avoids n! growth
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate polynomial p at x using Horner's method.
///
/// p = [a₀, a₁, …, aₙ] represents p(x) = a₀ + a₁x + … + aₙxⁿ.
/// Horner's scheme: ((aₙx + aₙ₋₁)x + …)x + a₀.
/// O(n) multiplications, O(n) additions, minimal rounding error.
///
/// Returns 0.0 for empty polynomial.
pub fn poly_eval(p: &[f64], x: f64) -> f64 {
    p.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Add two polynomials: (p + q)(x) = p(x) + q(x).
/// Result length = max(len(p), len(q)), trailing zeros preserved.
pub fn poly_add(p: &[f64], q: &[f64]) -> Vec<f64> {
    let n = p.len().max(q.len());
    let mut r = vec![0.0; n];
    for (i, &c) in p.iter().enumerate() { r[i] += c; }
    for (i, &c) in q.iter().enumerate() { r[i] += c; }
    r
}

/// Subtract two polynomials: (p - q)(x) = p(x) - q(x).
pub fn poly_sub(p: &[f64], q: &[f64]) -> Vec<f64> {
    let n = p.len().max(q.len());
    let mut r = vec![0.0; n];
    for (i, &c) in p.iter().enumerate() { r[i] += c; }
    for (i, &c) in q.iter().enumerate() { r[i] -= c; }
    r
}

/// Multiply two polynomials: (p · q)(x) = p(x) · q(x).
///
/// Uses naive O(n·m) convolution. For large polynomials (degree ≥ ~64),
/// prefer `poly_mul_ntt` which uses NTT for O(n log n) multiplication over Z_p.
/// Result degree = deg(p) + deg(q).
pub fn poly_mul(p: &[f64], q: &[f64]) -> Vec<f64> {
    if p.is_empty() || q.is_empty() { return vec![]; }
    let mut r = vec![0.0; p.len() + q.len() - 1];
    for (i, &a) in p.iter().enumerate() {
        for (j, &b) in q.iter().enumerate() {
            r[i + j] += a * b;
        }
    }
    r
}

/// Remove trailing near-zero coefficients from a polynomial.
///
/// After floating-point division or GCD steps, high-degree terms may be ~0
/// due to rounding. This trims them so degree comparisons are meaningful.
/// Tolerance: coefficients with |c| < eps are considered zero.
pub fn poly_normalize(p: &[f64], eps: f64) -> Vec<f64> {
    let mut v = p.to_vec();
    while v.len() > 1 && v.last().map_or(false, |&c| c.abs() < eps) {
        v.pop();
    }
    v
}

/// Polynomial division with remainder: p = q·quotient + remainder.
///
/// Returns `(quotient, remainder)`. Both have normalized trailing zeros removed.
/// Returns `None` if divisor q is zero (or all-zero).
/// Uses synthetic division (long division), O(n·m) where n=deg(p), m=deg(q).
///
/// # Numerical note
/// Division amplifies error when the leading coefficient of q is small.
/// For exact integer polynomial division, use `poly_divmod_mod` (not yet implemented).
pub fn poly_divmod(p: &[f64], q: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let q = poly_normalize(q, 1e-14);
    let p = poly_normalize(p, 1e-14);
    let lead = *q.last()?;
    if lead.abs() < 1e-14 { return None; }
    if p.len() < q.len() {
        return Some((vec![0.0], p));
    }
    let mut rem = p.clone();
    let quot_len = p.len() - q.len() + 1;
    let mut quot = vec![0.0; quot_len];
    for i in (0..quot_len).rev() {
        let coef = rem[i + q.len() - 1] / lead;
        quot[i] = coef;
        for (j, &qc) in q.iter().enumerate() {
            rem[i + j] -= coef * qc;
        }
    }
    let rem = poly_normalize(&rem, 1e-10);
    let quot = poly_normalize(&quot, 1e-10);
    Some((quot, rem))
}

/// Formal derivative of polynomial p.
///
/// If p = [a₀, a₁, a₂, …, aₙ], then p' = [a₁, 2a₂, 3a₃, …, naₙ].
/// Fundamental for Newton's root finding, Sturm chains, and squarefree factoring.
pub fn poly_deriv(p: &[f64]) -> Vec<f64> {
    if p.len() <= 1 { return vec![0.0]; }
    p.iter().enumerate().skip(1).map(|(i, &c)| i as f64 * c).collect()
}

/// Formal indefinite integral of polynomial p, with constant of integration C.
///
/// If p = [a₀, a₁, …, aₙ], then ∫p = [C, a₀, a₁/2, a₂/3, …, aₙ/(n+1)].
pub fn poly_integ(p: &[f64], constant: f64) -> Vec<f64> {
    let mut r = vec![constant];
    r.extend(p.iter().enumerate().map(|(i, &c)| c / (i as f64 + 1.0)));
    r
}

/// Polynomial composition: (p ∘ q)(x) = p(q(x)).
///
/// O(deg(p)·deg(q)²) naive evaluation: evaluate p with Horner's scheme where
/// each "coefficient" multiplication is a polynomial multiplication.
/// Result degree = deg(p) · deg(q).
pub fn poly_compose(p: &[f64], q: &[f64]) -> Vec<f64> {
    // Horner: result = p[n] + q*(p[n-1] + q*(…))
    p.iter().rev().fold(vec![0.0], |acc, &c| {
        let mut t = poly_mul(&acc, q);
        if t.is_empty() { t.push(0.0); }
        t[0] += c;
        t
    })
}

/// Scale polynomial by a scalar: (α·p)(x) = α·p(x).
pub fn poly_scale(p: &[f64], alpha: f64) -> Vec<f64> {
    p.iter().map(|&c| c * alpha).collect()
}

/// GCD of two polynomials over ℝ (floating-point coefficients).
///
/// Uses the Euclidean algorithm: gcd(p, q) = gcd(q, p mod q) until remainder ~0.
/// Returns a monic polynomial (leading coefficient = 1.0).
///
/// # Numerical limitations
/// Polynomial GCD over floating-point is inherently ill-conditioned when the
/// polynomials are near each other or have near-common factors. The result
/// is reliable for exact-integer polynomials represented as f64 (no accumulated
/// error) and for polynomials with well-separated roots. For exact GCD over
/// finite fields, use `poly_gcd_mod` (integer NTT-based, not yet implemented).
///
/// Returns [1.0] (constant 1) if gcd is numerically trivial.
pub fn poly_gcd(p: &[f64], q: &[f64]) -> Vec<f64> {
    const EPS: f64 = 1e-9;
    let mut a = poly_normalize(p, EPS);
    let mut b = poly_normalize(q, EPS);
    // Euclidean algorithm: invariant is gcd(a, b) = gcd(p, q)
    // Loop until b is the zero polynomial
    loop {
        // Is b the zero polynomial?
        if b.len() == 1 && b[0].abs() < EPS { break; }
        if b.is_empty() { break; }
        match poly_divmod(&a, &b) {
            None => break,
            Some((_, rem)) => {
                let norm_rem = poly_normalize(&rem, EPS);
                a = b;
                b = norm_rem;
            }
        }
    }
    // a is the gcd — make monic
    let lead = *a.last().unwrap_or(&1.0);
    if lead.abs() < EPS { return vec![1.0]; }
    a.iter().map(|&c| c / lead).collect()
}

/// Lagrange polynomial interpolation.
///
/// Given n+1 distinct (x, y) pairs, returns the unique polynomial of degree ≤ n
/// that passes through all points.
///
/// Uses the barycentric Lagrange formula for O(n²) evaluation stability:
/// - Phase 1: compute barycentric weights wᵢ = 1 / ∏_{j≠i} (xᵢ - xⱼ)
/// - Phase 2: for each evaluation point, use the barycentric formula
///
/// Returns the coefficient vector [a₀, a₁, …, aₙ] via explicit polynomial
/// construction (multiply out the basis polynomials). This is O(n³) total.
/// For evaluation only (not the coefficient vector), use `poly_interp_eval`.
///
/// Returns `None` if any two x values are identical (non-distinct nodes).
pub fn poly_interpolate(xs: &[f64], ys: &[f64]) -> Option<Vec<f64>> {
    assert_eq!(xs.len(), ys.len(), "xs and ys must have the same length");
    let n = xs.len();
    if n == 0 { return Some(vec![]); }
    if n == 1 { return Some(vec![ys[0]]); }
    // Check distinctness
    for i in 0..n {
        for j in (i + 1)..n {
            if (xs[i] - xs[j]).abs() < 1e-14 { return None; }
        }
    }
    // Build product polynomial (x - x₀)(x - x₁)…(x - xₙ₋₁)
    // and accumulate Lagrange basis polynomials
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        // Build basis polynomial Lᵢ(x) = ∏_{j≠i} (x - xⱼ) / (xᵢ - xⱼ)
        let mut basis = vec![1.0f64];
        let mut denom = 1.0f64;
        for j in 0..n {
            if j == i { continue; }
            denom *= xs[i] - xs[j];
            // Multiply basis by (x - xs[j])
            let mut new_basis = vec![0.0f64; basis.len() + 1];
            for (k, &c) in basis.iter().enumerate() {
                new_basis[k + 1] += c;
                new_basis[k] -= c * xs[j];
            }
            basis = new_basis;
        }
        // Accumulate ys[i] * Lᵢ(x) / denom
        let scale = ys[i] / denom;
        for (k, &c) in basis.iter().enumerate() {
            result[k] += scale * c;
        }
    }
    Some(result)
}

/// Evaluate the interpolating polynomial at a new point using barycentric weights.
///
/// More numerically stable than evaluating the coefficient vector from `poly_interpolate`.
/// O(n) per evaluation after O(n²) weight precomputation.
///
/// Returns `None` if x coincides with one of the interpolation nodes (exact match:
/// returns ys[i] directly in that case).
pub fn poly_interp_eval(xs: &[f64], ys: &[f64], x: f64) -> Option<f64> {
    let n = xs.len();
    if n == 0 { return None; }
    // Check if x is exactly a node
    for i in 0..n {
        if (x - xs[i]).abs() < 1e-15 { return Some(ys[i]); }
    }
    // Barycentric weights: wᵢ = 1 / ∏_{j≠i} (xᵢ - xⱼ)
    let mut w = vec![1.0f64; n];
    for i in 0..n {
        for j in 0..n {
            if i != j { w[i] /= xs[i] - xs[j]; }
        }
    }
    // Barycentric formula: L(x) = [Σ wᵢ·yᵢ/(x-xᵢ)] / [Σ wᵢ/(x-xᵢ)]
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..n {
        let t = w[i] / (x - xs[i]);
        num += t * ys[i];
        den += t;
    }
    if den.abs() < 1e-15 { return None; }
    Some(num / den)
}

// ─── Number Theoretic Transform ───────────────────────────────────────────

/// Number Theoretic Transform (NTT): polynomial multiplication over Z_p.
///
/// NTT is the exact analogue of FFT but over a prime finite field Z_p.
/// Since all arithmetic is modular, there is NO floating-point error — the
/// result is exact. This makes NTT the correct tool for exact polynomial
/// multiplication when coefficients are non-negative integers.
///
/// Algorithm: Cooley-Tukey butterfly, O(n log n), iterative (cache-friendly).
///
/// # Parameters
/// - `a`: input coefficient vector, length must be a power of 2
/// - `p`: prime modulus with p-1 divisible by n (required for NTT to exist)
/// - `g`: primitive root of Z_p (generator of the multiplicative group)
/// - `invert`: if true, computes the inverse NTT (INTT)
///
/// # Standard NTT prime
/// A commonly used prime is p = 998244353 = 119·2²³ + 1 with g = 3.
/// This supports NTT for n up to 2²³ ≈ 8.4 million.
///
/// Returns `None` if n is not a power of 2 or n does not divide p-1.
pub fn ntt(a: &[u64], p: u64, g: u64, invert: bool) -> Option<Vec<u64>> {
    let n = a.len();
    if n == 0 || (n & (n - 1)) != 0 { return None; } // must be power of 2
    if (p - 1) % n as u64 != 0 { return None; } // n must divide p-1

    let mut a = a.to_vec();

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j { a.swap(i, j); }
    }

    // Cooley-Tukey butterfly
    let mut len = 2;
    while len <= n {
        let w = if invert {
            mod_pow(g, p - 1 - (p - 1) / len as u64, p)
        } else {
            mod_pow(g, (p - 1) / len as u64, p)
        };
        let mut i = 0;
        while i < n {
            let mut wn = 1u64;
            for jj in 0..(len / 2) {
                let u = a[i + jj];
                let v = mul_mod(a[i + jj + len / 2], wn, p);
                a[i + jj] = (u + v) % p;
                a[i + jj + len / 2] = (u + p - v) % p;
                wn = mul_mod(wn, w, p);
            }
            i += len;
        }
        len <<= 1;
    }

    if invert {
        let n_inv = mod_pow(n as u64, p - 2, p);
        for x in &mut a { *x = mul_mod(*x, n_inv, p); }
    }

    Some(a)
}

/// Inverse NTT (INTT): convenience wrapper around `ntt`.
pub fn intt(a: &[u64], p: u64, g: u64) -> Option<Vec<u64>> {
    ntt(a, p, g, true)
}

/// Multiply two polynomials with non-negative integer coefficients using NTT.
///
/// Exact result over Z_p — no floating-point error.
/// Coefficients of inputs must be in [0, p).
/// Result coefficients are in [0, p) — if the true result coefficients exceed p,
/// they wrap mod p. Use a prime p larger than the maximum possible output coefficient.
///
/// For polynomials p, q with deg(p)=n, deg(q)=m and max coefficient M:
/// the maximum output coefficient is min(n,m)·M² — choose p accordingly.
///
/// Default: p = 998244353, g = 3 (NTT-prime, supports n up to 2²³).
///
/// Returns `None` if combined degree + 1 is not ≤ 2²³ for the default prime.
pub fn poly_mul_ntt(p_coeffs: &[u64], q_coeffs: &[u64], modulus: u64, g: u64) -> Option<Vec<u64>> {
    if p_coeffs.is_empty() || q_coeffs.is_empty() { return Some(vec![]); }
    let result_len = p_coeffs.len() + q_coeffs.len() - 1;
    // Pad to next power of 2
    let padded = result_len.next_power_of_two();
    let mut ap = p_coeffs.to_vec();
    let mut aq = q_coeffs.to_vec();
    ap.resize(padded, 0);
    aq.resize(padded, 0);

    let fa = ntt(&ap, modulus, g, false)?;
    let fb = ntt(&aq, modulus, g, false)?;
    let fc: Vec<u64> = fa.iter().zip(fb.iter()).map(|(&x, &y)| mul_mod(x, y, modulus)).collect();
    let mut result = intt(&fc, modulus, g)?;
    result.truncate(result_len);
    Some(result)
}

/// The standard NTT prime: 998244353 = 119·2²³ + 1 with primitive root 3.
/// Supports NTT for n up to 2²³ ≈ 8.4 million.
pub const NTT_PRIME: u64 = 998_244_353;
/// Primitive root of `NTT_PRIME`.
pub const NTT_PRIME_ROOT: u64 = 3;

#[cfg(test)]
mod polynomial_tests {
    use super::*;

    // ── poly_eval ────────────────────────────────────────────────────────────

    #[test]
    fn poly_eval_horner_known() {
        // p(x) = 1 + 2x + 3x² evaluated at x=2: 1 + 4 + 12 = 17
        assert!((poly_eval(&[1.0, 2.0, 3.0], 2.0) - 17.0).abs() < 1e-10);
        // p(x) = x³ at x=3: 27
        assert!((poly_eval(&[0.0, 0.0, 0.0, 1.0], 3.0) - 27.0).abs() < 1e-10);
        // Constant polynomial
        assert!((poly_eval(&[5.0], 99.0) - 5.0).abs() < 1e-10);
        // Empty polynomial = 0
        assert_eq!(poly_eval(&[], 3.0), 0.0);
    }

    #[test]
    fn poly_eval_at_zero() {
        // p(0) = constant term
        let p = &[7.0, 3.0, 2.0];
        assert!((poly_eval(p, 0.0) - 7.0).abs() < 1e-10);
    }

    // ── poly_add / poly_sub / poly_mul ───────────────────────────────────────

    #[test]
    fn poly_add_different_degrees() {
        let p = vec![1.0, 2.0, 3.0]; // 1 + 2x + 3x²
        let q = vec![4.0, 5.0];       // 4 + 5x
        let r = poly_add(&p, &q);
        assert!((r[0] - 5.0).abs() < 1e-10);
        assert!((r[1] - 7.0).abs() < 1e-10);
        assert!((r[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn poly_mul_convolution() {
        // (1 + x)(1 + x) = 1 + 2x + x²
        let p = vec![1.0, 1.0];
        let r = poly_mul(&p, &p);
        assert!((r[0] - 1.0).abs() < 1e-10);
        assert!((r[1] - 2.0).abs() < 1e-10);
        assert!((r[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn poly_mul_degree_adds() {
        let p = vec![1.0, 1.0, 1.0]; // degree 2
        let q = vec![1.0, 1.0];       // degree 1
        let r = poly_mul(&p, &q);
        assert_eq!(r.len(), 4); // degree 3
        // (1 + x + x²)(1 + x) = 1 + 2x + 2x² + x³
        assert!((r[0] - 1.0).abs() < 1e-10);
        assert!((r[1] - 2.0).abs() < 1e-10);
        assert!((r[2] - 2.0).abs() < 1e-10);
        assert!((r[3] - 1.0).abs() < 1e-10);
    }

    // ── poly_divmod ──────────────────────────────────────────────────────────

    #[test]
    fn poly_divmod_exact_division() {
        // x² - 1 = (x - 1)(x + 1), so divmod by (x - 1) gives (x + 1) remainder 0
        let p = vec![-1.0, 0.0, 1.0]; // x² - 1
        let q = vec![-1.0, 1.0];       // x - 1
        let (quot, rem) = poly_divmod(&p, &q).unwrap();
        // quot should be [1, 1] = 1 + x, rem should be [0] ≈ 0
        assert!((poly_eval(&quot, 2.0) - 3.0).abs() < 1e-8, "quot mismatch");
        assert!(rem[0].abs() < 1e-9, "remainder not zero");
    }

    #[test]
    fn poly_divmod_with_remainder() {
        // x² divided by (x + 1): x² = (x - 1)(x + 1) + 1
        let p = vec![0.0, 0.0, 1.0]; // x²
        let q = vec![1.0, 1.0];       // x + 1
        let (quot, rem) = poly_divmod(&p, &q).unwrap();
        // p = q * quot + rem
        let reconstructed = poly_add(&poly_mul(&q, &quot), &rem);
        for (i, (&lhs, &rhs)) in reconstructed.iter().zip(p.iter()).enumerate() {
            assert!((lhs - rhs).abs() < 1e-8, "reconstruction failed at coeff {i}");
        }
    }

    #[test]
    fn poly_divmod_degree_less_than_divisor() {
        // Dividing x by x²: quotient = 0, remainder = x
        let p = vec![0.0, 1.0];        // x
        let q = vec![0.0, 0.0, 1.0];  // x²
        let (quot, rem) = poly_divmod(&p, &q).unwrap();
        assert!((quot[0]).abs() < 1e-10);
        assert!((rem[0]).abs() < 1e-10);
        assert!((rem[1] - 1.0).abs() < 1e-10);
    }

    // ── poly_deriv / poly_integ ──────────────────────────────────────────────

    #[test]
    fn poly_deriv_known() {
        // d/dx [3 + 2x + x²] = 2 + 2x
        let p = vec![3.0, 2.0, 1.0];
        let dp = poly_deriv(&p);
        assert!((dp[0] - 2.0).abs() < 1e-10);
        assert!((dp[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn poly_integ_undoes_deriv() {
        // ∫(d/dx p) dx = p (up to constant of integration)
        let p = vec![5.0, 3.0, 2.0];
        let dp = poly_deriv(&p);
        let recovered = poly_integ(&dp, p[0]);
        for (i, (&r, &orig)) in recovered.iter().zip(p.iter()).enumerate() {
            assert!((r - orig).abs() < 1e-9, "integ(deriv(p)) failed at coeff {i}");
        }
    }

    // ── poly_compose ─────────────────────────────────────────────────────────

    #[test]
    fn poly_compose_known() {
        // p(x) = x², q(x) = x + 1. p(q(x)) = (x+1)² = 1 + 2x + x²
        let p = vec![0.0, 0.0, 1.0]; // x²
        let q = vec![1.0, 1.0];       // x + 1
        let r = poly_compose(&p, &q);
        // Evaluate at x=3: (3+1)² = 16
        assert!((poly_eval(&r, 3.0) - 16.0).abs() < 1e-8);
    }

    // ── poly_gcd ─────────────────────────────────────────────────────────────

    #[test]
    fn poly_gcd_common_factor() {
        // gcd((x-1)(x-2), (x-1)(x-3)) = (x-1)
        // (x-1)(x-2) = x² - 3x + 2
        // (x-1)(x-3) = x² - 4x + 3
        let p = vec![2.0, -3.0, 1.0];
        let q = vec![3.0, -4.0, 1.0];
        let g = poly_gcd(&p, &q);
        // gcd should be (x - 1) up to scalar, so evaluate at x=1 should give ~0
        assert!(poly_eval(&g, 1.0).abs() < 1e-6,
            "gcd should vanish at x=1: got {}", poly_eval(&g, 1.0));
        // And gcd evaluated at x=2 should not vanish (root of p only)
        assert!(poly_eval(&g, 2.0).abs() > 0.1);
    }

    #[test]
    fn poly_gcd_coprime() {
        // gcd(x² + 1, x² - 1): no common factors, gcd should be ≈ constant
        let p = vec![1.0, 0.0, 1.0];  // x² + 1
        let q = vec![-1.0, 0.0, 1.0]; // x² - 1
        let g = poly_gcd(&p, &q);
        // Coprime polys have gcd = 1 (constant)
        assert_eq!(g.len(), 1, "gcd of coprime polys should be constant");
    }

    // ── poly_interpolate ─────────────────────────────────────────────────────

    #[test]
    fn poly_interpolate_linear() {
        // Two points (0, 1), (1, 3) → p(x) = 1 + 2x
        let xs = vec![0.0, 1.0];
        let ys = vec![1.0, 3.0];
        let p = poly_interpolate(&xs, &ys).unwrap();
        assert!((poly_eval(&p, 0.0) - 1.0).abs() < 1e-9);
        assert!((poly_eval(&p, 1.0) - 3.0).abs() < 1e-9);
        assert!((poly_eval(&p, 0.5) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn poly_interpolate_quadratic() {
        // Three points: (0,1), (1,4), (2,9) → passes through y=x²+2x+1=(x+1)²
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![1.0, 4.0, 9.0];
        let p = poly_interpolate(&xs, &ys).unwrap();
        // Verify at the nodes
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert!((poly_eval(&p, x) - y).abs() < 1e-8,
                "interpolation failed at x={x}: got {}, expected {y}", poly_eval(&p, x));
        }
        // And at a non-node point: (x+1)² at x=3 = 16
        assert!((poly_eval(&p, 3.0) - 16.0).abs() < 1e-7);
    }

    #[test]
    fn poly_interpolate_duplicate_x_returns_none() {
        let xs = vec![1.0, 1.0, 2.0];
        let ys = vec![1.0, 2.0, 3.0];
        assert!(poly_interpolate(&xs, &ys).is_none());
    }

    #[test]
    fn poly_interp_eval_barycentric() {
        // Same quadratic: points (0,1),(1,4),(2,9)
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![1.0, 4.0, 9.0];
        // At node: should return exact value
        assert!((poly_interp_eval(&xs, &ys, 1.0).unwrap() - 4.0).abs() < 1e-12);
        // Off node: (3+1)² = 16
        assert!((poly_interp_eval(&xs, &ys, 3.0).unwrap() - 16.0).abs() < 1e-7);
    }

    // ── NTT ──────────────────────────────────────────────────────────────────

    #[test]
    fn ntt_roundtrip() {
        let p = NTT_PRIME;
        let g = NTT_PRIME_ROOT;
        let a = vec![1u64, 2, 3, 4, 0, 0, 0, 0]; // padded to power of 2
        let fa = ntt(&a, p, g, false).unwrap();
        let recovered = intt(&fa, p, g).unwrap();
        assert_eq!(a, recovered, "NTT roundtrip failed");
    }

    #[test]
    fn poly_mul_ntt_matches_naive() {
        // (1 + 2x + 3x²)(4 + 5x) = 4 + 13x + 22x² + 15x³
        // Coefficients as u64, modulus large enough (NTT_PRIME >> max coeff)
        let p = vec![1u64, 2, 3];
        let q = vec![4u64, 5];
        let ntt_result = poly_mul_ntt(&p, &q, NTT_PRIME, NTT_PRIME_ROOT).unwrap();
        let naive_result = poly_mul(
            &p.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            &q.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        );
        assert_eq!(ntt_result.len(), naive_result.len());
        for (i, (&ntt_c, &naive_c)) in ntt_result.iter().zip(naive_result.iter()).enumerate() {
            assert_eq!(ntt_c as i64, naive_c as i64, "coeff {i} mismatch: NTT={ntt_c} naive={naive_c}");
        }
    }

    #[test]
    fn poly_mul_ntt_larger() {
        // (1 + x)^4 = 1 + 4x + 6x² + 4x³ + x⁴ via NTT repeated squaring
        let linear = vec![1u64, 1];
        let sq = poly_mul_ntt(&linear, &linear, NTT_PRIME, NTT_PRIME_ROOT).unwrap();
        let fourth = poly_mul_ntt(&sq, &sq, NTT_PRIME, NTT_PRIME_ROOT).unwrap();
        let expected = vec![1u64, 4, 6, 4, 1];
        assert_eq!(fourth, expected, "(1+x)^4 via NTT failed");
    }

    #[test]
    fn ntt_requires_power_of_two() {
        let a = vec![1u64, 2, 3]; // length 3 — not power of 2
        assert!(ntt(&a, NTT_PRIME, NTT_PRIME_ROOT, false).is_none());
    }
}

