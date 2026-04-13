//! Kulisch exact accumulator for sums of f64 values.
//!
//! The Kulisch accumulator is a fixed-point integer wide enough to represent
//! any sum of IEEE 754 double-precision values without loss of information.
//! It is named after Ulrich Kulisch, who proposed the "long accumulator"
//! as a standard hardware feature for exact floating-point summation in
//! *Computer Arithmetic in Theory and Practice* (1981).
//!
//! # Design
//!
//! We use 34 × 128-bit signed words in two's complement, little-endian
//! order. Total width is 4352 bits. The virtual radix point sits 2100 bits
//! above the least significant bit, which covers:
//!
//! - The smallest non-zero f64 magnitude: 2^-1074 (subnormal minimum)
//! - The largest non-zero f64 magnitude: 2^1024 (just above f64::MAX)
//! - Headroom of ~1000 bits above f64::MAX for summing up to 2^1000
//!   positive values of magnitude 2^1024 (more than enough)
//!
//! Each `add_f64(x)` call is exact: no rounding occurs. A mantissa with up
//! to 53 significant bits is shifted into the accumulator at the bit
//! position determined by its binary exponent and signed-added with full
//! carry propagation.
//!
//! # When to use
//!
//! - As the oracle in correctness tests for `kahan_sum`, `neumaier_sum`,
//!   `dot_2`, `compensated_horner`.
//! - When you need to prove a libm recipe is correctly rounded by
//!   comparing its output to the exact reference.
//! - Never in a hot loop — each `add_f64` costs approximately 10-20×
//!   a plain floating-point add.
//!
//! # Not supported (by design)
//!
//! - NaN / infinity inputs are silently ignored. The accumulator is for
//!   finite sums only; recipes must handle non-finite inputs upstream.
//! - Multiplication and division: use the `double_double` module or the
//!   `bigfloat` module for those.
//! - Exact product addition: can be built on top by decomposing
//!   `a * b` via `two_product_fma(a, b) = (hi, lo)` and adding both
//!   components, but this is not exposed as a dedicated method here to
//!   keep the core API small.

/// Number of i128 words in the accumulator.
const NUM_WORDS: usize = 34;

/// Width of a single word in bits.
const WORD_BITS: usize = 128;

/// Position of the radix point, measured in bits from the LSB.
///
/// A mantissa representing `2^(-1074)` (the smallest subnormal f64) lands
/// at bit position `-1074 + RADIX_BITS = 1026 ≥ 0`. A mantissa representing
/// `2^1023` (near f64::MAX) lands at bit position `1023 + RADIX_BITS = 3123`,
/// well below `NUM_WORDS * WORD_BITS - 1 = 4351`.
const RADIX_BITS: i32 = 2100;

/// Exact fixed-point accumulator for finite f64 sums.
///
/// Construct via `new()` or `default()`, add values with `add_f64`, read
/// back with `to_f64()`. The instance is not thread-safe and must be held
/// by exactly one writer at a time.
#[derive(Clone)]
pub struct KulischAccumulator {
    words: [i128; NUM_WORDS],
}

impl KulischAccumulator {
    /// Create a new accumulator set to exact zero.
    pub fn new() -> Self {
        Self {
            words: [0_i128; NUM_WORDS],
        }
    }

    /// Add an f64 value to the accumulator exactly.
    ///
    /// `±0.0`, NaN, and `±inf` are silently ignored. Finite subnormal and
    /// normal values are added with full precision.
    pub fn add_f64(&mut self, x: f64) {
        if x == 0.0 || !x.is_finite() {
            return;
        }

        let bits = x.to_bits();
        let sign_bit = (bits >> 63) & 1;
        let raw_exp = ((bits >> 52) & 0x7FF) as i32;
        let raw_mant = bits & 0x000F_FFFF_FFFF_FFFF;

        // Decompose into (mantissa, exponent) such that the mathematical
        // value is `mantissa * 2^exponent`.
        let (mantissa_u, exp) = if raw_exp == 0 {
            // Subnormal: no implicit leading 1, true exponent is -1074.
            (raw_mant, -1074_i32)
        } else {
            // Normal: implicit leading 1 at bit 52, unbiased exponent is
            // raw_exp - 1023, and the mantissa's true binary value is
            // shifted so its LSB represents 2^(exp - 52).
            (raw_mant | (1_u64 << 52), raw_exp - 1023 - 52)
        };

        // Apply the sign.
        let signed_mant: i128 = if sign_bit == 0 {
            mantissa_u as i128
        } else {
            -(mantissa_u as i128)
        };

        // Bit position of the mantissa's LSB in the accumulator.
        let bit_pos_signed = exp + RADIX_BITS;
        debug_assert!(
            bit_pos_signed >= 0 && (bit_pos_signed as usize) < NUM_WORDS * WORD_BITS,
            "Kulisch range exceeded: bit_pos = {bit_pos_signed}"
        );
        let bit_pos = bit_pos_signed as usize;
        let word_idx = bit_pos / WORD_BITS;
        let bit_in_word = (bit_pos % WORD_BITS) as u32;

        self.add_shifted_value(word_idx, bit_in_word, signed_mant);
    }

    /// Core: add `signed_mant << bit_in_word` at word position `word_idx`
    /// with full sign extension into higher words.
    ///
    /// Uses unsigned multi-word add with carry. For words beyond the first
    /// two "data" words, the value added is the sign fill (all-1s if
    /// `signed_mant` is negative, else 0), which correctly sign-extends the
    /// value across the rest of the accumulator.
    fn add_shifted_value(&mut self, word_idx: usize, bit_in_word: u32, signed_mant: i128) {
        let sign_fill: u128 = if signed_mant < 0 { u128::MAX } else { 0 };

        // Compute the two "data" words covering the shifted value. For
        // `bit_in_word == 0`, the value sits entirely in `data0` and the
        // sign extension starts at word 1 (= sign_fill). For other shifts,
        // the value spills into word 1, and sign extension starts at word 2.
        let (data0, data1): (u128, u128) = if bit_in_word == 0 {
            (signed_mant as u128, sign_fill)
        } else {
            let low = (signed_mant as u128).wrapping_shl(bit_in_word);
            // Arithmetic right shift preserves sign extension for the high word.
            let high = (signed_mant >> (WORD_BITS as u32 - bit_in_word)) as u128;
            (low, high)
        };

        let mut carry: u128 = 0;
        let mut i = 0_usize;
        loop {
            let idx = word_idx + i;
            if idx >= NUM_WORDS {
                break;
            }
            let val: u128 = match i {
                0 => data0,
                1 => data1,
                _ => sign_fill,
            };
            let a = self.words[idx] as u128;
            let (s1, c1) = a.overflowing_add(val);
            let (s2, c2) = s1.overflowing_add(carry);
            self.words[idx] = s2 as i128;
            carry = (c1 as u128) | (c2 as u128);

            // Early termination once we're past the two data words and in
            // a steady state:
            //   - positive signed_mant: sign_fill = 0. If carry = 0, nothing
            //     further changes.
            //   - negative signed_mant: sign_fill = u128::MAX. If carry = 1,
            //     each subsequent word sees (word + MAX + 1) mod 2^128 = word,
            //     which is a no-op, so we can stop.
            if i >= 2 {
                if sign_fill == 0 && carry == 0 {
                    break;
                }
                if sign_fill == u128::MAX && carry == 1 {
                    break;
                }
            }

            i += 1;
        }
    }

    /// Subtract an f64 value from the accumulator exactly.
    ///
    /// Equivalent to `add_f64(-x)`. Convenience for test writing.
    pub fn sub_f64(&mut self, x: f64) {
        self.add_f64(-x);
    }

    /// Add each element of `xs` to the accumulator.
    pub fn add_slice(&mut self, xs: &[f64]) {
        for &x in xs {
            self.add_f64(x);
        }
    }

    /// Return the current accumulator value as the closest f64, rounded
    /// to nearest even. For values in the f64 normal range this is the
    /// correctly-rounded result.
    pub fn to_f64(&self) -> f64 {
        // Determine overall sign from the most significant word.
        let is_negative = self.words[NUM_WORDS - 1] < 0;

        // Take absolute value into an unsigned magnitude representation.
        let mag = self.magnitude(is_negative);

        // Find the highest set bit.
        let msb_bit = match find_msb(&mag) {
            None => return if is_negative { -0.0 } else { 0.0 },
            Some(b) => b,
        };

        let exp_unbiased = (msb_bit as i32) - RADIX_BITS;

        // Overflow → ±inf.
        if exp_unbiased > 1023 {
            return if is_negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }

        let sign_bit = if is_negative { 1_u64 << 63 } else { 0 };

        // ── Subnormal path ──────────────────────────────────────────────────
        // For `exp_unbiased <= -1023`, the value is smaller than `2^-1022`
        // and must be represented as a subnormal. The subnormal mantissa
        // stores 52 bits starting at accumulator bit 1026 (= 2^-1074), with
        // the round bit at 1025 and sticky below.
        //
        // If `msb_bit < 1026`, the value is smaller than the smallest
        // representable subnormal and rounds to ±0.
        if exp_unbiased <= -1023 {
            // Smallest subnormal bit position.
            const SUB_LSB: usize = (RADIX_BITS - 1074) as usize; // = 1026

            if msb_bit < SUB_LSB {
                // Below minimum subnormal — check sticky for round-to-nearest.
                // If msb_bit == SUB_LSB - 1, value is exactly 2^-1075, which
                // rounds to 0 (ties-to-even, and 0 is even).
                // Otherwise rounds to 0 as well.
                return if is_negative { -0.0 } else { 0.0 };
            }

            let mantissa_sub = extract_bits(&mag, SUB_LSB, 52);
            let round = if SUB_LSB >= 1 {
                extract_bits(&mag, SUB_LSB - 1, 1) & 1
            } else {
                0
            };
            let sticky = if SUB_LSB >= 2 {
                !bits_below_are_zero(&mag, SUB_LSB - 1)
            } else {
                false
            };

            let mut final_mant = mantissa_sub;
            let round_up = round == 1 && (sticky || (final_mant & 1) == 1);
            if round_up {
                final_mant += 1;
            }

            // If rounding carried into the implicit-1 position, this subnormal
            // just became the smallest normal.
            if final_mant == (1_u64 << 52) {
                let biased_exp = 1_u64;
                let bits = sign_bit | (biased_exp << 52) | 0;
                return f64::from_bits(bits);
            }

            let bits = sign_bit | 0 /* biased_exp = 0 */ | final_mant;
            return f64::from_bits(bits);
        }

        // ── Normal path ─────────────────────────────────────────────────────
        // Extract 52 explicit mantissa bits, round bit, sticky.
        let mantissa_52 = extract_bits(&mag, msb_bit - 52, 52);
        let round = if msb_bit >= 53 {
            extract_bits(&mag, msb_bit - 53, 1) & 1
        } else {
            0
        };
        let sticky = if msb_bit >= 54 {
            !bits_below_are_zero(&mag, msb_bit - 53)
        } else {
            false
        };

        let round_up = round == 1 && (sticky || (mantissa_52 & 1) == 1);

        let mut final_mant = mantissa_52;
        let mut final_exp = exp_unbiased;
        if round_up {
            final_mant += 1;
            if final_mant == (1_u64 << 52) {
                final_mant = 0;
                final_exp += 1;
                if final_exp > 1023 {
                    return if is_negative {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    };
                }
            }
        }

        let biased_exp = (final_exp + 1023) as u64;
        let bits = sign_bit | (biased_exp << 52) | final_mant;
        f64::from_bits(bits)
    }

    fn magnitude(&self, is_negative: bool) -> [u128; NUM_WORDS] {
        let mut mag = [0_u128; NUM_WORDS];
        if is_negative {
            // Two's complement negation across NUM_WORDS.
            let mut carry = 1_u128;
            for i in 0..NUM_WORDS {
                let inv = !(self.words[i] as u128);
                let (sum, c) = inv.overflowing_add(carry);
                mag[i] = sum;
                carry = c as u128;
            }
        } else {
            for i in 0..NUM_WORDS {
                mag[i] = self.words[i] as u128;
            }
        }
        mag
    }
}

impl Default for KulischAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// Find the position (0-indexed from LSB) of the highest set bit in a
/// multi-word unsigned magnitude. Returns `None` if the magnitude is zero.
fn find_msb(mag: &[u128]) -> Option<usize> {
    for (i, &word) in mag.iter().enumerate().rev() {
        if word != 0 {
            let bits_in_word = WORD_BITS - (word.leading_zeros() as usize);
            return Some(i * WORD_BITS + bits_in_word - 1);
        }
    }
    None
}

/// Extract `count` bits starting at bit position `start` from the magnitude.
/// Returns the bits as a u64 (caller must ensure `count <= 64`).
fn extract_bits(mag: &[u128], start: usize, count: usize) -> u64 {
    debug_assert!(count <= 64);
    if count == 0 {
        return 0;
    }
    let end = start + count - 1;
    let start_word = start / WORD_BITS;
    let end_word = end / WORD_BITS;
    let start_bit_in_word = start % WORD_BITS;

    if start_word >= mag.len() {
        return 0;
    }

    if start_word == end_word || end_word >= mag.len() {
        let word = mag[start_word];
        let shifted = word >> start_bit_in_word;
        let mask = if count == 64 {
            u64::MAX as u128
        } else {
            (1_u128 << count) - 1
        };
        (shifted & mask) as u64
    } else {
        // Spans two words.
        let low = mag[start_word] >> start_bit_in_word;
        let high = mag[end_word] << (WORD_BITS - start_bit_in_word);
        let combined = low | high;
        let mask = if count == 64 {
            u64::MAX as u128
        } else {
            (1_u128 << count) - 1
        };
        (combined & mask) as u64
    }
}

/// Return true if every bit strictly below bit position `pos` in `mag` is zero.
fn bits_below_are_zero(mag: &[u128], pos: usize) -> bool {
    let word_idx = pos / WORD_BITS;
    let bit_in_word = pos % WORD_BITS;
    // Fully zero words below word_idx
    for i in 0..word_idx.min(mag.len()) {
        if mag[i] != 0 {
            return false;
        }
    }
    // Partial word
    if word_idx < mag.len() && bit_in_word > 0 {
        let mask = (1_u128 << bit_in_word) - 1;
        if (mag[word_idx] & mask) != 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accumulator_is_zero() {
        let acc = KulischAccumulator::new();
        assert_eq!(acc.to_f64(), 0.0);
    }

    #[test]
    fn single_value_roundtrip() {
        let values = [1.0, 3.14, -2.71, 1e100, 1e-100, 1024.0, 0.125, -7.5];
        for &x in &values {
            let mut acc = KulischAccumulator::new();
            acc.add_f64(x);
            assert_eq!(
                acc.to_f64(),
                x,
                "single-value roundtrip failed for {x:e}"
            );
        }
    }

    #[test]
    fn ignores_zero_nan_inf() {
        let mut acc = KulischAccumulator::new();
        acc.add_f64(0.0);
        acc.add_f64(-0.0);
        acc.add_f64(f64::NAN);
        acc.add_f64(f64::INFINITY);
        acc.add_f64(f64::NEG_INFINITY);
        assert_eq!(acc.to_f64(), 0.0);
    }

    #[test]
    fn integer_sum_is_exact() {
        let mut acc = KulischAccumulator::new();
        for i in 1..=100_i32 {
            acc.add_f64(i as f64);
        }
        assert_eq!(acc.to_f64(), 5050.0);
    }

    #[test]
    fn cancellation_recovery() {
        // The class test for exact summation: 1e17 + 1 + (-1e17) = 1.
        // Plain fp64 drops the 1.
        let mut acc = KulischAccumulator::new();
        acc.add_f64(1e17);
        acc.add_f64(1.0);
        acc.add_f64(-1e17);
        assert_eq!(acc.to_f64(), 1.0);
    }

    #[test]
    fn large_cancellation_chain() {
        // A harder version: many alternating large ± small values.
        let mut acc = KulischAccumulator::new();
        for _ in 0..1000 {
            acc.add_f64(1e100);
            acc.add_f64(1.0);
            acc.add_f64(-1e100);
        }
        assert_eq!(acc.to_f64(), 1000.0);
    }

    #[test]
    fn subnormal_accumulation() {
        let tiny = f64::from_bits(1); // smallest positive subnormal
        let mut acc = KulischAccumulator::new();
        for _ in 0..1000 {
            acc.add_f64(tiny);
        }
        // Exact answer: 1000 × 2^-1074.
        assert_eq!(acc.to_f64(), 1000.0 * tiny);
    }

    #[test]
    fn sign_handling() {
        let mut acc = KulischAccumulator::new();
        acc.add_f64(5.0);
        acc.add_f64(-3.0);
        assert_eq!(acc.to_f64(), 2.0);

        acc.add_f64(-10.0);
        assert_eq!(acc.to_f64(), -8.0);
    }

    #[test]
    fn zero_result_from_exact_cancellation() {
        let mut acc = KulischAccumulator::new();
        acc.add_f64(1e50);
        acc.add_f64(-1e50);
        assert_eq!(acc.to_f64(), 0.0);
    }

    #[test]
    fn sub_f64_matches_add_negation() {
        let mut a = KulischAccumulator::new();
        let mut b = KulischAccumulator::new();
        a.add_f64(5.0);
        a.sub_f64(2.0);
        b.add_f64(5.0);
        b.add_f64(-2.0);
        assert_eq!(a.to_f64(), b.to_f64());
        assert_eq!(a.to_f64(), 3.0);
    }

    #[test]
    fn add_slice_equivalent_to_loop() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut a = KulischAccumulator::new();
        let mut b = KulischAccumulator::new();
        a.add_slice(&xs);
        for &x in &xs {
            b.add_f64(x);
        }
        assert_eq!(a.to_f64(), b.to_f64());
        assert_eq!(a.to_f64(), 15.0);
    }

    #[test]
    fn beats_naive_sum_on_stress_input() {
        // The classic Kahan stress: one large value plus many small ones.
        let mut xs = vec![1.0];
        for _ in 0..10_000 {
            xs.push(1e-10);
        }
        let mut acc = KulischAccumulator::new();
        acc.add_slice(&xs);
        let expected = 1.0 + 10_000.0 * 1e-10;
        // Kulisch is exact; the only error is in rounding the final result
        // back to f64, which for this input is at most half an ulp.
        let err = (acc.to_f64() - expected).abs();
        assert!(err < 1e-15, "Kulisch sum err {err:e}");
    }

    #[test]
    fn cross_validate_with_kahan_on_wellbehaved() {
        use crate::primitives::compensated::sums::kahan_sum;
        let xs: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();
        let mut acc = KulischAccumulator::new();
        acc.add_slice(&xs);
        let kulisch = acc.to_f64();
        let kahan = kahan_sum(&xs);
        // For well-conditioned input, Kahan should agree with Kulisch
        // to within a few ulps.
        let err = (kulisch - kahan).abs();
        assert!(
            err < 1e-12,
            "Kahan {kahan} vs Kulisch {kulisch}, diff {err:e}"
        );
    }
}
