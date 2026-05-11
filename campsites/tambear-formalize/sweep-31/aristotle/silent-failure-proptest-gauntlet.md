# Silent-Failure Proptest Gauntlet — Sweep 31

**Created:** 2026-05-08
**Author:** aristotle
**Task:** #29 (math-researcher routed); covers adversarial's Gap 2.
**Substrate:** F13 DRAFT-PRIVATE antibody pattern, per the cross-cutting finding #5 in `dec031-invariants-deconstruction.md` ("the non-monotone path antibody rhymes with F11 recognition/design discipline at a different layer; recognition without antibody is just a rule, recognition with antibody is enforceable substrate").

**Inputs:**
- DEC-031 §3.1, §3.2, §3.3, §3.4, §3.7, §6 enforcement (lines 3310-3478 of `R:\tambear\docs\decisions.md`)
- ATK-DEC031-3 (round-trip identity stub elevation)
- ATK-DEC031-4 (non-monotone path silent under-report)
- chains-E/F/G subnormal regime
- My DEC-031 invariants deconstruction (`dec031-invariants-deconstruction.md`)

---

## The seven surfaces

A silent failure is a class of error where the system produces a *plausible-looking output* that's wrong, with no panic, no error return, no log warning. The user reads a result they trust and proceeds with broken precision. Each surface below is a class of silent failure DEC-031 must reject by construction.

The gauntlet is the antibody suite. Each proptest is a *witness*: when the rule under test breaks, the proptest produces a regression instance (a specific input) that demonstrates the failure. The witness is what makes the antibody trustworthy — not "we tested some inputs and it seemed fine," but "we have a generator that catches violations and a regression instance pinned in source if it ever fires."

---

## Surface 1 — Non-monotone path silent under-report (~46 OOM canonical case)

**Failure mode:** path-builder accepts a non-monotone path; the destination-dominated rule applied to it under-reports error by 46 orders of magnitude. Canonical: `BigFloat(200) → f64 → BigFloat(200)`.

**The rule's scope precondition** (§3.2): destination-dominated budget applies ONLY to monotonically-coarsening paths. Non-monotone (coarsen-then-refine) violates the precondition. The antibody (§6 #12) requires path-builder to reject such paths at construction time.

**Proptest assertion structure:**

```rust
proptest! {
    #[test]
    fn non_monotone_paths_rejected_at_construction(
        steps in monotonic_or_not_strategy()
    ) {
        match PrecisionPath::try_from_steps(&steps) {
            Ok(path) => {
                // Path constructed → it MUST be monotone-coarsening.
                prop_assert!(path.is_monotonically_coarsening());
            }
            Err(PathError::NonMonotone) => {
                // Path rejected → it MUST contain a refining step after a coarsening step.
                prop_assert!(steps_contain_refine_after_coarsen(&steps));
            }
            Err(PathError::NotMonotonelyCoarsening) => {
                // Path rejected → monotonically refining, which destination-dominated doesn't bound.
                prop_assert!(steps_are_monotonically_refining(&steps));
            }
        }
    }
}
```

**Input distribution:**

- Strategy `monotonic_or_not_strategy()` generates `Vec<PrecisionLevel>` of length 2-6.
- 40% probability: monotonic-coarsening (sorted descending by precision_bits).
- 30% probability: monotonic-refining (sorted ascending).
- 30% probability: non-monotone (random with at least one refine-after-coarsen).
- Within each class, sample precision_bits from {53, 106, 107, 200, 500, 1024}.
- Edge cases included as fixed regression witnesses (see below).

**Regression witnesses (pinned, must always be in the proptest harness):**

```rust
const REGRESSION_NON_MONOTONE: &[&[PrecisionLevel]] = &[
    // The canonical ATK-DEC031-4 case:
    &[BigFloat(200), F64, BigFloat(200)],
    // Variant: coarsen all the way down, then refine partway up:
    &[BigFloat(500), F64, DD, BigFloat(107)],
    // Two-step non-monotone:
    &[BigFloat(200), F64, BigFloat(53)], // refine via f64 to BigFloat(53) — rounded into f64 first
];
```

**Why this is an antibody, not just a test:** without `Err(PathError::NonMonotone)` at construction, the user could write `BigFloat(200) → f64 → BigFloat(200)` thinking they got a no-op, but lose 147 bits of precision silently. The proptest witnesses every non-monotone path the user might construct; if path-builder ever loosens to allow non-monotone paths, the proptest fires with a specific input as the regression. The antibody is *enforced by tests, regression-witnessed at all times*.

**Cross-link:** §6 #12 enforcement states this directly. The proptest IS §6 #12 made executable.

---

## Surface 2 — Subnormal regime crossing (chains-E/F/G boundary)

**Failure mode:** an operation crosses the f64 subnormal boundary (|x| < 2^-1022) silently. The destination-dominated rule assumes relative-bounded ULP; in the subnormal regime, error is absolute-bounded at 2^-1074 instead of relative. A path that includes a subnormal value as input or intermediate underestimates its own error.

**Proptest assertion structure:**

```rust
proptest! {
    #[test]
    fn subnormal_regime_uses_absolute_bound(
        path in any_monotone_coarsening_path(),
        x_seed in any::<u64>()
    ) {
        let x = f64::from_bits(x_seed);
        prop_assume!(x.is_finite()); // Skip NaN/Inf for this surface.

        let actual_error = compute_actual_error(&path, x);
        let bound = ulp_budget_path(&path, x);

        // The rule's actual error MUST be bounded by the budget — INCLUDING the subnormal regime.
        prop_assert!(
            actual_error <= bound,
            "Bound violation: x={x:e}, actual={actual_error:e}, bound={bound:e}"
        );

        // Regime check: if x was subnormal at any point, the bound should reference absolute (not relative).
        if path_contains_subnormal_point(&path, x) {
            prop_assert!(bound.uses_absolute_bound(),
                "Path crosses subnormal regime but bound uses relative ULP for x={x:e}");
        }
    }
}
```

**Input distribution:**

- 25% probability: x sampled from `[-2^-1022, 2^-1022]` (subnormal range).
- 25% probability: x near tier boundaries: 2^-1022 ± k·2^-1074 for k ∈ {1, 2, ..., 100}.
- 25% probability: x in normal range with magnitude close to subnormal floor: |x| ∈ [2^-1022, 2^-1018].
- 25% probability: x in well-normal range.
- Path: sample from monotone-coarsening generator (Surface 1 covers non-monotone rejection).

**Regression witnesses:**

```rust
const REGRESSION_SUBNORMAL: &[(f64, &[PrecisionLevel])] = &[
    // The smallest positive subnormal at f64:
    (f64::from_bits(1), &[BigFloat(500), DD, F64]),
    // Just below the normal/subnormal boundary:
    (f64::MIN_POSITIVE * (1.0 - f64::EPSILON), &[BigFloat(200), F64]),
    // Tier-boundary near MIN_POSITIVE — exact boundary value:
    (f64::MIN_POSITIVE, &[BigFloat(200), F64]),
    // Subnormal that's a pure power-of-two:
    (f64::from_bits(1u64 << 51), &[BigFloat(106), F64]),
];
```

**Why this is an antibody:** without explicit subnormal-regime checks, an implementation that uses one global ULP-budget formula will silently miscalculate error in the subnormal regime. The proptest catches the off-by-2^52 (relative-vs-absolute) gap at construction time. Implementations that handle subnormal correctly pass; implementations that don't are flagged by the regression witnesses with specific inputs.

**Cross-link:** §3.2 chains-E/F/G scope precondition. The rule weakens to per-step subnormal-aware bounds; the proptest verifies the weakening is applied where required.

---

## Surface 3 — Storage-vs-operation cache-key confusion

**Failure mode:** cache returns a result computed at one (storage, operation-precision) coordinate when the consumer asks at a different coordinate. Per §3.3, storage and operation-precision are orthogonal axes; the cache key MUST distinguish them. A naive cache that keys only on operation precision returns wrong results when the consumer's storage differs.

**Proptest assertion structure:**

```rust
proptest! {
    #[test]
    fn cache_key_distinguishes_storage_and_operation(
        op_a in any_op_strategy(),
        storage_a in any_scalar_ty_strategy(),
        op_precision_a in any_precision_bits_strategy(),
        storage_b in any_scalar_ty_strategy(),
        op_precision_b in any_precision_bits_strategy(),
    ) {
        let key_a = build_cache_key(&op_a, &storage_a, op_precision_a);
        let key_b = build_cache_key(&op_a, &storage_b, op_precision_b);

        // If storage XOR op-precision differ, keys MUST differ.
        let storage_differs = storage_a != storage_b;
        let op_precision_differs = op_precision_a != op_precision_b;

        if storage_differs || op_precision_differs {
            prop_assert_ne!(
                key_a, key_b,
                "Cache key collision: storage_a={storage_a:?} op_prec_a={op_precision_a} vs storage_b={storage_b:?} op_prec_b={op_precision_b}"
            );
        } else {
            // All three coordinates match → same key.
            prop_assert_eq!(key_a, key_b);
        }
    }
}
```

**Input distribution:**

- `any_scalar_ty_strategy()`: f64, DD, BigFloat(p) for p ∈ {53, 106, 107, 200, 500}.
- `any_precision_bits_strategy()`: u32 in {53, 106, 107, 200, 500, 1024}.
- `any_op_strategy()`: a fixed Op enum variant for the surface (test runs per-Op-variant with same storage/precision pairing).
- 30% probability: storage = ScalarTy at native of op_precision (the "matched" case).
- 30% probability: storage > op_precision (storage carries more bits than op uses).
- 30% probability: storage < op_precision (op uses more bits than storage carries — should error, but cache-key distinction comes first).
- 10%: edge cases at tier boundaries.

**Regression witnesses:**

```rust
const REGRESSION_CACHE_KEY: &[(ScalarTy, u32, ScalarTy, u32)] = &[
    // §6 #11 canonical: BigFloat(200) storage + op(53) vs BigFloat(200) storage + op(200)
    (ScalarTy::BigFloat { precision_bits: 200 }, 53,
     ScalarTy::BigFloat { precision_bits: 200 }, 200),
    // Storage diff at same op-precision:
    (ScalarTy::F64, 53, ScalarTy::DD, 53),
    // Tier-boundary case:
    (ScalarTy::DD, 106, ScalarTy::BigFloat { precision_bits: 107 }, 106),
];
```

**Why this is an antibody:** without distinct keys for distinct (storage, op-precision) pairs, the cache silently returns a result computed under different precision assumptions. The user requested `using(precision=200)` over BigFloat(200) storage, gets back the result of `using(precision=53)` from a prior call. **This is the precision-lattice analog of F12's silent-claim-collapse.** The antibody is the cache key construction; the proptest verifies the key actually distinguishes.

**Cross-link:** §3.7 cache-key extension; §6 #11 storage-vs-operation cache-key separation; DEC-024 sub-clause C.

---

## Surface 4 — Tier-dispatch boundary discontinuity

**Failure mode:** A precision request just inside one tier dispatches differently than just outside, but the implementation handles the dispatch boundary inconsistently. Specifically: requested=53 → P0F64; requested=54 → P1DD (effective 106); requested=106 → P1DD; requested=107 → P2BigFloat(107). At each boundary, dispatch-level changes. An implementation that confuses requested vs dispatched precision silently produces wrong results.

**Proptest assertion structure:**

```rust
proptest! {
    #[test]
    fn tier_dispatch_boundary_is_consistent(
        requested_bits in 1u32..=2048,
        rounding in any_rounding_mode_strategy(),
    ) {
        let context = PrecisionContext { requested_precision_bits: requested_bits, rounding };

        let dispatch_level = context.dispatch_level();
        let dispatched_bits = context.dispatched_precision_bits();

        // Tier boundaries per §3.4 are bit-exact:
        match requested_bits {
            0..=53 => {
                prop_assert_eq!(dispatch_level, PrecisionLevel::P0F64);
                prop_assert_eq!(dispatched_bits, 53);
            }
            54..=106 => {
                prop_assert_eq!(dispatch_level, PrecisionLevel::P1DoubleDouble);
                prop_assert_eq!(dispatched_bits, 106);
            }
            107.. => {
                prop_assert_eq!(dispatch_level, PrecisionLevel::P2BigFloat { precision_bits: requested_bits });
                prop_assert_eq!(dispatched_bits, requested_bits);
            }
        }

        // Dispatched precision MUST be ≥ requested (per §3.4 "by NATIVE precision, not requested").
        prop_assert!(dispatched_bits >= requested_bits);
    }
}
```

**Input distribution:**

- `requested_bits`: full range 1-2048 with bias toward boundaries.
- 20% probability: `requested_bits ∈ {52, 53, 54, 105, 106, 107}` (boundary-adjacent).
- 20% probability: `requested_bits` uniform in [1, 53] (P0F64 region).
- 20% probability: `requested_bits` uniform in [54, 106] (P1DD region).
- 20% probability: `requested_bits` uniform in [107, 1024] (P2BigFloat normal-range).
- 20% probability: `requested_bits ≥ 1024` (saturation regime per §3.8).

**Regression witnesses:**

```rust
const REGRESSION_TIER_BOUNDARY: &[u32] = &[
    1, 53, 54, 106, 107, 200, 500, 1023, 1024, 1025, 2048
];
// All must produce the right (dispatch_level, dispatched_bits) per §3.4.
```

**Why this is an antibody:** without tier-boundary discontinuity tests, an off-by-one error in `dispatch_level()` (e.g., `match { 0..53 → P0; 53..106 → P1; ... }` instead of `0..=53` and `54..=106`) goes undetected. The proptest catches every boundary cell. The witness `requested_bits ∈ {53, 54, 106, 107}` pins the four-cell discontinuity test in source.

**Cross-link:** §3.4 tier-dispatch boundary discontinuity. The proptest is §3.4 made falsifiable.

---

## Surface 5 — Diamond commutativity violation footprint

**Failure mode:** the two paths f64→BigFloat(p) — direct and through-DD — disagree at the bit level for some f64 input + p≥53 + rounding mode. This is the §6 #3 invariant violation. Specific failure shapes from the Phase 1-8 deconstruction Phase 7: at p∈[53,106), the through-DD path may introduce 0.5-ULP rounding while the direct path is exact, IF the implementation doesn't special-case lo=0.

**Proptest assertion structure:**

```rust
proptest! {
    #[test]
    fn diamond_commutativity_holds(
        x_seed in any::<u64>(),
        precision_bits in 53u32..=2048,
        rounding in any_rounding_mode_strategy(),
    ) {
        let x = f64::from_bits(x_seed);

        let direct = BigFloat::from_f64_with_rounding(x, precision_bits, rounding);
        let through_dd = {
            let dd = DoubleDouble::from(x);
            BigFloat::from_dd_with_rounding(dd, precision_bits, rounding)
        };

        // Diamond commutativity: bit-exact equal at canonical form.
        prop_assert!(
            direct.canonical_eq(&through_dd),
            "Diamond commutativity violation: x_bits=0x{x_seed:016x} (={x:e}), p={precision_bits}, rounding={rounding:?}\n\
             direct        = {direct:?}\n\
             through_dd    = {through_dd:?}"
        );
    }
}
```

**Input distribution:**

- `x_seed`: full u64 range, with biases:
  - 10% probability: NaN bit patterns (exponent all-1, non-zero mantissa) — covers 2^53 distinct payloads, sample uniform over payloads + signaling-bit.
  - 10% probability: ±0, ±Inf.
  - 15% probability: subnormals (exponent zero, mantissa non-zero).
  - 15% probability: tier-boundary values (powers of 2; values near MIN_POSITIVE; values near MAX_FINITE).
  - 50% probability: random finite normals.
- `precision_bits`: full range 53-2048 with bias:
  - 25% probability: `precision_bits = 53` (the f64-equivalence point).
  - 25% probability: `precision_bits ∈ (53, 106)` (the lo=0-special-case region).
  - 25% probability: `precision_bits = 106` (the DD-equivalence point).
  - 25% probability: `precision_bits > 106` (above-DD region).
- `rounding`: all five RoundingMode variants, uniform.

**Regression witnesses:**

```rust
const REGRESSION_DIAMOND: &[(u64, u32, RoundingMode)] = &[
    // f64→BigFloat(53) at all rounding modes — bit-equal must hold:
    (1.0_f64.to_bits(), 53, RoundingMode::RoundToNearestTiesEven),
    (1.0_f64.to_bits(), 53, RoundingMode::RoundTowardZero),
    (1.0_f64.to_bits(), 53, RoundingMode::RoundTowardPositiveInfinity),
    (1.0_f64.to_bits(), 53, RoundingMode::RoundTowardNegativeInfinity),
    (1.0_f64.to_bits(), 53, RoundingMode::RoundToNearestTiesAwayFromZero),

    // The lo=0 boundary case at p=80 (Phase 7 finding):
    (3.14_f64.to_bits(), 80, RoundingMode::RoundToNearestTiesEven),
    (1e308_f64.to_bits(), 80, RoundingMode::RoundToNearestTiesEven),

    // NaN payload preservation diamond:
    (0x7FF8_0000_0000_0001, 100, RoundingMode::RoundToNearestTiesEven), // qNaN with payload 1
    (0x7FF0_0000_0000_0001, 100, RoundingMode::RoundToNearestTiesEven), // sNaN with payload 1
    (0xFFF0_0000_0000_0001, 100, RoundingMode::RoundToNearestTiesEven), // negative sNaN

    // Subnormal: smallest positive subnormal:
    (0x0000_0000_0000_0001, 200, RoundingMode::RoundToNearestTiesEven),

    // Tier-boundary near MIN_POSITIVE:
    (f64::MIN_POSITIVE.to_bits(), 200, RoundingMode::RoundToNearestTiesEven),
];
```

**Why this is an antibody:** Phase 7 of my deconstruction surfaced that at p∈(53,106), the through-DD path can introduce 0.5-ULP rounding if the implementation doesn't special-case lo=0. The proptest is the antibody: when implementation correctness fails (lo=0 not special-cased; NaN payload normalized; etc.), the proptest produces a specific x_bits + precision_bits witness. The witnesses pinned in source cover the load-bearing edge classes.

**The lo=0 sub-finding I surfaced becomes testable here.** If math-researcher's BigFloat impl forgets to special-case lo=0 in DD→BigFloat refinement, this proptest fires with an x ∈ (53, 106)-precision case. The proptest IS the operationalization of the Phase 7 refinement.

**Cross-link:** §6 #3 diamond commutativity invariant; my deconstruction Phase 7 lo=0 finding.

---

## Surface 6 — Round-trip identity edge cases

**Failure mode:** `BigFloat::from_f64(value, 53).to_f64() == value` fails for some f64 input. Specifically: subnormals lose representation, ±0 sign collapses, NaN payload canonicalizes (under strict mode), ±Inf direction flips, tier-boundary near MIN_POSITIVE flushes to zero.

**NaN-payload policy is parameterized — see math-researcher's DESIGN.md §5 Q3.** Two modes:

- **Strict mode (`NAN_PAYLOAD_PRESERVE = true`):** asserts full bit-equality of NaN inputs. Implementation must store all 52 mantissa bits of f64 NaN through BigFloat; `BigFloatKind::NaN` carries a `NaNPayload` variant. This is the publication-grade-rigor reading per my deconstruction Phase 8 Rejection 1.

- **Permissive mode (`NAN_PAYLOAD_PRESERVE = false`):** asserts sign-bit + NaN-ness preservation only; specific payload bits may differ. Implementation discards payload at `from_f64`, returns canonical NaN at `to_f64`. This is math-researcher's DESIGN.md §5 Q3 proposed v2 default.

**The team must pick one mode at ratification time.** The proptest harness compiles both; CI runs whichever mode the spec.toml-equivalent config (or `cfg!(feature = "nan_payload_preserve")` flag) selects.

**Proptest assertion structure:**

```rust
proptest! {
    #[test]
    fn round_trip_identity_at_tier_boundary(x_seed in any::<u64>()) {
        let x = f64::from_bits(x_seed);
        let big = BigFloat::from_f64(x, 53);
        let round_tripped = big.to_f64();

        if x.is_nan() {
            // NaN-class assertion depends on policy:
            #[cfg(feature = "nan_payload_preserve")]
            {
                // Strict: bit-equal including payload.
                prop_assert_eq!(
                    round_tripped.to_bits(), x.to_bits(),
                    "Strict NaN round-trip violated (payload not preserved):\n\
                     x.to_bits()             = 0x{x_bits:016x} ({x_class})\n\
                     round_tripped.to_bits() = 0x{rt_bits:016x}",
                    x_bits = x.to_bits(),
                    rt_bits = round_tripped.to_bits(),
                    x_class = classify_f64(x),
                );
            }
            #[cfg(not(feature = "nan_payload_preserve"))]
            {
                // Permissive: NaN-ness + sign preserved; payload may differ.
                prop_assert!(round_tripped.is_nan(),
                    "NaN round-trip lost NaN-ness: x={:#018x}, rt={:#018x}",
                    x.to_bits(), round_tripped.to_bits());
                prop_assert_eq!(
                    round_tripped.is_sign_negative(),
                    x.is_sign_negative(),
                    "NaN round-trip flipped sign: x={:#018x} sign={}, rt={:#018x} sign={}",
                    x.to_bits(), x.is_sign_negative(),
                    round_tripped.to_bits(), round_tripped.is_sign_negative()
                );
            }
        } else {
            // All non-NaN inputs: bit-equality regardless of policy.
            prop_assert_eq!(
                round_tripped.to_bits(), x.to_bits(),
                "Round-trip identity violated:\n\
                 x.to_bits()             = 0x{x_bits:016x} ({x_class})\n\
                 round_tripped.to_bits() = 0x{rt_bits:016x}",
                x_bits = x.to_bits(),
                rt_bits = round_tripped.to_bits(),
                x_class = classify_f64(x),
            );
        }
    }
}
```

**Note on policy choice:** the strict-vs-permissive split is itself an F12 design-claim under F11's recognition/design lens. Math-researcher's Q3 frames the choice as design (with rationale: "most libraries discard"). My deconstruction's Phase 8 Rejection 1 frames it as recognition (publication-grade rigor demands preservation). The team must pick. Whichever is chosen, the OTHER policy's regression witnesses become "MUST-FAIL-IF-WRONG-POLICY-ACTIVE" — the harness MUST emit a test-suite-mismatch warning if both policies' witnesses are pinned simultaneously without the corresponding feature flag.

**Input distribution:** EXHAUSTIVE coverage by class, not just sampled. The full f64 bit space is 2^64 — too many for total exhaustion, but the classes are finite and small for the boundary cases:

- **Specials (small finite set, exhaustive):**
  - `0x0000_0000_0000_0000` (+0)
  - `0x8000_0000_0000_0000` (-0)
  - `0x7FF0_0000_0000_0000` (+Inf)
  - `0xFFF0_0000_0000_0000` (-Inf)
  - All 2^53 NaN payloads sampled at 10K random bit-patterns + the signaling/quiet boundary + payload-zero (canonical NaN) + payload-all-1s.

- **Tier-boundary near MIN_POSITIVE (exhaustive):**
  - `f64::MIN_POSITIVE`, `MIN_POSITIVE - ulp`, `MIN_POSITIVE + ulp`, `MIN_POSITIVE / 2`, etc.

- **Subnormals (sampled):**
  - All 2^52 subnormal bit-patterns are too many; sample 50K uniformly + the smallest (0x1) and largest (boundary) explicitly.

- **Finite normals (sampled):**
  - 50K uniform bit-pattern samples in normal range.
  - Include known-tricky values: powers of 2, 1/3, π, e, 1e308, 1e-308.

**Regression witnesses (must always pin):**

```rust
const REGRESSION_ROUND_TRIP: &[u64] = &[
    // Specials:
    0x0000_0000_0000_0000, // +0
    0x8000_0000_0000_0000, // -0
    0x7FF0_0000_0000_0000, // +Inf
    0xFFF0_0000_0000_0000, // -Inf

    // NaN classes — payload preservation is the hard part:
    0x7FF8_0000_0000_0000, // canonical qNaN
    0x7FF0_0000_0000_0001, // sNaN with payload 1
    0x7FFF_FFFF_FFFF_FFFF, // qNaN with all-1 payload
    0xFFF0_0000_0000_0001, // negative sNaN
    0x7FF0_0000_DEAD_BEEF, // sNaN with diagnostic payload (real-world test)

    // Subnormals:
    0x0000_0000_0000_0001, // smallest positive subnormal
    0x000F_FFFF_FFFF_FFFF, // largest subnormal
    0x0008_0000_0000_0000, // mid-range subnormal

    // Tier boundary:
    f64::MIN_POSITIVE.to_bits(),                     // smallest positive normal
    (f64::MIN_POSITIVE - f64::EPSILON.to_bits()),    // adjacent, may underflow
    (-f64::MIN_POSITIVE).to_bits(),                  // negative boundary
];
```

**Why this is an antibody:** §6 #13 is the recognition that round-trip identity must hold; the proptest is the operationalization with specific witnesses. ATK-DEC031-3 elevated this from a stub because v1 had it broken. The proptest both catches future regressions AND surfaces the "this implementation doesn't preserve NaN payloads" failure mode by witness rather than by silent loss.

**Cross-link:** §6 #13 round-trip identity invariant; ATK-DEC031-3 attack history; my deconstruction Invariant 2 Phase 5 ("round-trip identity is the witness for BigFloat-as-f64-extension claim").

---

## Surface 7 — DD↔BigFloat boundary off-by-one

**Failure mode:** the precision threshold for accepting DD↔BigFloat refinement (§3.1) is wrong by one bit. p<53 should reject; p∈[53,106) should be RoundingEquivalent; p≥106 should be Strict. Off-by-one in the threshold check (e.g., `p > 53` instead of `p >= 53`) causes DD→BigFloat(53) to be misclassified as Strict when it's RoundingEquivalent OR rejected when it should accept.

**Proptest assertion structure:**

```rust
proptest! {
    #[test]
    fn dd_bigfloat_boundary_classification(
        precision_bits in 0u32..=2048,
    ) {
        // Determine the classification per §3.1:
        let expected_class = match precision_bits {
            0..52 => PairClass::Rejected,
            53..=105 => PairClass::RoundingEquivalent { ulps: 0.5, at: PrecisionLevel::P2BigFloat { precision_bits } },
            106.. => PairClass::Strict,
        };

        let actual_class = classify_dd_to_bigfloat_refine(precision_bits);
        prop_assert_eq!(actual_class, expected_class);
    }

    #[test]
    fn f64_bigfloat_boundary_classification(
        precision_bits in 0u32..=2048,
    ) {
        let expected_class = match precision_bits {
            0..52 => PairClass::Rejected,
            53.. => PairClass::Strict,
        };

        let actual_class = classify_f64_to_bigfloat_refine(precision_bits);
        prop_assert_eq!(actual_class, expected_class);
    }
}
```

**Input distribution:**

- `precision_bits` full range 0-2048, with bias:
  - 30% at `{52, 53, 54}` — the f64↔BigFloat boundary.
  - 30% at `{105, 106, 107}` — the DD↔BigFloat-Strict boundary.
  - 20% at exact powers of 2.
  - 20% uniform in remaining range.

**Regression witnesses (the off-by-one cluster):**

```rust
const REGRESSION_BOUNDARY_OFF_BY_ONE: &[u32] = &[
    // f64↔BigFloat boundary:
    52, // rejected
    53, // Strict (the smallest accepted)
    54, // Strict

    // DD↔BigFloat boundary:
    105, // RoundingEquivalent
    106, // Strict (the smallest Strict)
    107, // Strict

    // Edge cases at the boundary of the boundary:
    0,  // rejected
    1,  // rejected
    51, // rejected
    52, // rejected
];
```

**Why this is an antibody:** the §3.1 classification table is small (12 rows), but a per-row implementation can have an off-by-one in any of the boundary checks. The proptest catches every cell. **Without this surface coverage, an implementation that uses `>` instead of `>=` somewhere in the threshold ladder silently misclassifies edge cases.** The witness pin set covers all ten edge-cells of the two ladders.

**Cross-link:** §3.1 per-pair commutativity classes table.

---

## Cross-cutting structure

All seven surfaces share the same antibody-pattern shape (per F13 DRAFT-PRIVATE):

| Surface | Rule | Scope precondition | Antibody (proptest) |
|---|---|---|---|
| 1 | destination-dominated budget | monotonically-coarsening | path-builder rejects non-monotone |
| 2 | destination-dominated budget | above subnormal floor | regime-detection switches to absolute bound |
| 3 | cache-key composition | distinct (storage, op-prec) keys | cache-key construction includes both axes |
| 4 | tier-dispatch | requested→dispatched mapping | dispatch_level() match-arm exhaustive at boundaries |
| 5 | diamond commutativity | f64-source + p≥53 | both paths produce bit-equal canonical BigFloat |
| 6 | round-trip identity | tier=53 boundary | from_f64 ∘ to_f64 = id over all f64 bit patterns |
| 7 | per-pair commutativity classification | precision threshold | classification ladder uses `>=` and `<=` correctly at edges |

Every surface is the same thing at different layers: a rule with a scope precondition, antibodied by a test that catches scope-violation by witness. The proptest gauntlet IS the antibody suite. Together they make the precision lattice's rules trustworthy as substrate.

**This is F13 made operational at gauntlet scope.** F13's recognition: "rules without scope-enforcement antibodies silently fail in their out-of-scope domains." F13's design: each rule must ship with a proptest that catches out-of-scope application. F13's mechanical artifact: this gauntlet.

---

## Implementation notes for math-researcher

- **Generator quality matters more than count.** A generator that biases toward boundaries is worth 100x more than one that samples uniformly. Use proptest's `prop_oneof!` to route to per-class generators with explicit weights.
- **Regression witnesses MUST be pinned in source.** Even if the proptest runs 10K iterations and never replays the same x, the regression list is the canonical "these MUST always pass" set. If a future refactor breaks one, CI flags immediately.
- **Witness format must include classifying context.** Print the input class (e.g., "subnormal", "qNaN with payload 0x7FF...", "tier-boundary near MIN_POSITIVE") in the failure message. The class IS the witness's antibody-target identification.
- **Classify_f64 helper.** Worth writing once: `fn classify_f64(x: f64) -> &'static str` that returns the f64 class. Improves all proptest failure messages across surfaces 5 and 6.
- **Performance:** proptests with 10K iterations × 7 surfaces = 70K test evaluations. At ~0.1 ms per evaluation, gauntlet runs in ~7 seconds. Acceptable for CI; can be relaxed in development with `proptest-config { cases: 100 }`.

---

## Status

Gauntlet design complete; covers seven silent-failure surfaces with antibody-shape (rule + scope + proptest + regression witnesses).

Output at `R:\winrapids\campsites\tambear-formalize\sweep-31\aristotle\silent-failure-proptest-gauntlet.md`.

**F13 DRAFT-PRIVATE status:** if math-researcher and team-lead want F13 elevated to ratified status, this gauntlet is the operationalization (recognition + design + mechanical artifact = full F-series piece per math-researcher's meta-principle). The precision-lattice instance + the F12 schema-doc instance together make F13 a cross-domain pattern.

If F13 is kept DRAFT-PRIVATE indefinitely, the gauntlet still works as Sweep 31 substrate.

Routing to math-researcher with a brief; flagging F13-status question to navigator separately.
