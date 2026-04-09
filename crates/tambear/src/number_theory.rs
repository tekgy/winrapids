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
/// Accumulate: sequential squaring chain (Kingdom B).
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

