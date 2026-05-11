//! Diagnostic trace for the phase_c_sub_alignment_stress 2-limb gap.
//!
//! This is an investigation file, not a production test. Drop this in
//! `crates/tambear/tests/` to instrument the failing case.
//!
//! Aristotle's deconstruction needed actual bit-level intermediate
//! values to distinguish "structural rule's bound" from "guard-bit
//! error two steps before the final round." The latter requires
//! tracing what each intermediate state of path 1 vs path 2 looks
//! like.

use tambear::lattice::RoundingMode;
use tambear::primitives::big_float::BigFloat;

const RTE: RoundingMode = RoundingMode::RoundToNearestTiesEven;

#[test]
fn trace_sub_2limb_gap() {
    let p_low = 107u32;
    let p_high = 157u32;
    let n_high = ((p_high + 63) / 64) as usize;

    // Use the minimal failing input from the proptest:
    let a_mantissa_low = 1110511917290206315u64;
    let b_mantissa_low = 562949953421313u64;
    let delta = 3u32;

    let mut a_limbs = vec![0u64; n_high];
    a_limbs[0] = a_mantissa_low;
    let top_pos = (p_high - 1) % 64;
    a_limbs[n_high - 1] |= 1u64 << top_pos;

    let mut b_limbs = vec![0u64; n_high];
    b_limbs[0] = b_mantissa_low;
    b_limbs[n_high - 1] |= 1u64 << top_pos;

    let a_high = BigFloat::from_raw_limbs(false, 0i64, p_high, a_limbs);
    let b_high = BigFloat::from_raw_limbs(false, -(delta as i64), p_high, b_limbs);

    eprintln!("=== INPUTS ===");
    eprintln!("a_high: exp={}, p={}, limbs={:#x?}",
        a_high.exponent(), a_high.precision_bits(), a_high.limbs());
    eprintln!("b_high: exp={}, p={}, limbs={:#x?}",
        b_high.exponent(), b_high.precision_bits(), b_high.limbs());

    eprintln!("\n=== PATH 1: round operands → sub at p_low ===");
    let a_low = a_high.with_precision_rounded(p_low, RTE);
    let b_low = b_high.with_precision_rounded(p_low, RTE);
    eprintln!("a_low : exp={}, limbs={:#x?}", a_low.exponent(), a_low.limbs());
    eprintln!("b_low : exp={}, limbs={:#x?}", b_low.exponent(), b_low.limbs());

    let direct = a_low.sub(&b_low, RTE);
    eprintln!("direct: exp={}, limbs={:#x?}", direct.exponent(), direct.limbs());

    eprintln!("\n=== PATH 2: sub at p_high → round to p_low ===");
    let diff_high = a_high.sub(&b_high, RTE);
    eprintln!("diff_high: exp={}, p={}, limbs={:#x?}",
        diff_high.exponent(), diff_high.precision_bits(), diff_high.limbs());

    let via_high = diff_high.with_precision_rounded(p_low, RTE);
    eprintln!("via_high: exp={}, limbs={:#x?}", via_high.exponent(), via_high.limbs());

    eprintln!("\n=== GAP ===");
    eprintln!("direct.limbs[0]   = 0x{:016x} = {}", direct.limbs()[0], direct.limbs()[0]);
    eprintln!("via_high.limbs[0] = 0x{:016x} = {}", via_high.limbs()[0], via_high.limbs()[0]);
    eprintln!("delta             = {}", via_high.limbs()[0] as i64 - direct.limbs()[0] as i64);

    // Hypothesis 1: direct rounding of a_low and b_low
    // independently lost information needed for the sub.
    //
    // Hypothesis 2: diff_high at p_high preserves info that
    // round(a_high)-round(b_high) cannot.
    //
    // The key question: where in the 50 dropped bits does the
    // disagreement originate?

    // Let's manually check: what bits below p_low are present in
    // a_high and b_high?
    eprintln!("\n=== Inspecting a_high low bits below new LSB at p_low=107 ===");
    // p_low=107 means top bit at pos 42 of limbs[1] in p_low rep.
    // In the p_high=157 rep, the top is at pos 28 of limbs[2].
    // The "p_low LSB" is 107 bits below the top, i.e., position
    // (28 + 64*2) - 106 = 156 - 106 = 50 of the original mag.
    // So bits 0..49 are below the p_low LSB; they're the round/sticky region.
    eprintln!("a_high.limbs[0] (bits 0..63): 0x{:016x}", a_high.limbs()[0]);
    eprintln!("  bit 49 (round): {}", (a_high.limbs()[0] >> 49) & 1);
    eprintln!("  bits 0..48 (sticky region): 0x{:013x}",
        a_high.limbs()[0] & ((1u64 << 49) - 1));

    eprintln!("b_high.limbs[0] (bits 0..63): 0x{:016x}", b_high.limbs()[0]);
    eprintln!("  bit 49 (round): {}", (b_high.limbs()[0] >> 49) & 1);
    eprintln!("  bits 0..48 (sticky region): 0x{:013x}",
        b_high.limbs()[0] & ((1u64 << 49) - 1));

    // For path 1: does with_precision_rounded round a_high up or down?
    // If bit 49 of a_high is 0 → round down (truncate).
    // If bit 49 is 1: tie-break depends on bits below. If sticky != 0 → round up.
    //                 If sticky == 0 → round to even (depends on bit 50, the new LSB).
}
