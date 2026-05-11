---
date: 2026-04-23
from: scout
status: FIXED — recording pre-fix bugs and fdlibm lineage for audit trail
---

# asin Polynomial Coefficient Audit

Source file: `crates/tambear/src/recipes/libm/asin.rs`

## What the polynomial is

fdlibm `e_asin.c` uses a rational approximation on |x| ≤ 0.5:

```
asin(x) = x + x·P(x²)/Q(x²)
```

P is degree 5 in w = x² (6 coefficients P_S0..P_S5).
Q is degree 4 in w (4 coefficients Q_S1..Q_S4, with implicit leading 1).

This is ONE polynomial for the entire |x| ≤ 0.5 region — not two separate
polynomials for sub-regions. The session summary describing "P_S2/P_S4/P_S5
discrepancies" were in THIS single polynomial, not across multiple regions.

## Pre-fix bugs (now corrected in source)

**P_S2 — digit transposition**
- Wrong:   `2.012255…e-01` (digits 2/1 transposed in positions 4-5)
- Correct: `2.012125321348629e-01` (hex `0x3FC9C1550E884455`)
- fdlibm lineage: exact match to `e_asin.c` `pS2 = 2.01212532134862925665e-01`
- Effect: corrupts the x⁴ contribution to the polynomial. For |x| near 0.5,
  the error accumulates to multiple ULP beyond the ≤2 ULP contract.

**P_S5 — invented constant, wrong sign and magnitude**
- Wrong:   `-3.25e-06` (negative; bit pattern `0xBECB49E6…` — no fdlibm lineage)
- Correct: `+3.479331075960212e-05` (hex `0x3F023DE10DFDF709`)
- fdlibm lineage: `pS5 = 3.47933107596021167570e-05`
- Effect: sign flip at the x¹⁰ term causes visible accuracy loss in mid-range
  inputs. The wrong value has no traceable origin in fdlibm or any known
  asin polynomial catalog.

**P_S4 — decimal round-trip discrepancy (minor)**
- Value: `7.915349942898145e-04` (hex `0x3F49EFE07501B288`)
- fdlibm: `pS4 = 7.91534994289814532176e-04`
- At f64 precision (53 bits), these round to the same bit pattern.
  The discrepancy was in the decimal display, not the stored bit pattern.
  No bug — informational only.

## Current state

The fix is applied. Source comments at lines 64-71 record the pre-fix values:

```rust
// Previous P_S2 had a digit transposition (2.01225 vs correct 2.01212) and
// P_S5 had wrong sign and magnitude (-3.25e-6 vs correct +3.48e-5).
const P_S2: f64 =  2.012_125_321_348_629_3e-01; // 0x3FC9C1550E884455 (was 2.012255…)
const P_S5: f64 =  3.479_331_075_960_211_7e-05; // 0x3F023DE10DFDF709 (was -3.25e-6)
```

## fdlibm reference for future auditors

The canonical reference is Sun Microsystems fdlibm `e_asin.c`. The 6 P
coefficients and 4 Q coefficients can be verified by:

1. Fetching the file from the NetLib or FreeBSD source tree
2. Extracting the hex constants next to each `pS`/`qS` label
3. Checking `f64::from_bits(0x3FC9C1550E884455)` matches the decimal value

The polynomial structure is rational (P/Q), not pure polynomial — the Q
denominator improves accuracy in the mid-range where the pure P approximation
would need higher degree. Any future re-derivation from mpmath should target
the same rational structure, not replace it with a pure minimax polynomial
(which would need more terms for the same ULP budget).

## What was not a bug

P_S0, P_S1, P_S3 matched fdlibm exactly. Q_S1..Q_S4 matched fdlibm exactly.
The half-angle reduction (`ax > 0.5` path) was correct.
The special-case handling (NaN, |x|>1, ±0, ±1) was correct.
