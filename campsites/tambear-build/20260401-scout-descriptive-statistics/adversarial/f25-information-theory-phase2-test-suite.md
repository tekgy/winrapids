# F25 Information Theory — Adversarial Test Suite (Phase 2 Update)

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**File**: `src/information_theory.rs` (~32KB)

---

## Previously reported (confirmed still present):

### IT1 [HIGH]: i32 overflow in joint_histogram composite key
**Location**: information_theory.rs:319
`x * ny as i32 + y` overflows for `nx * ny > 2^31`. STILL PRESENT.

---

## New findings:

### IT2 [HIGH]: Negative labels cause panic in contingency_from_labels

**Location**: information_theory.rs:335-339

`labels_a = [0, -1]` → `(-1i32) as usize = 18446744073709551615` → index out of bounds PANIC.

**Test**: `contingency_from_labels(&[0, -1], &[0, 1])` → panics.

**Fix**: `assert!(labels_a.iter().all(|&x| x >= 0))`.

---

### IT3 [MEDIUM]: Renyi entropy H_0 of zero distribution returns -inf

**Location**: information_theory.rs:92-96

`probs = [0.0, 0.0]` with `alpha=0.0` → `(0.0).ln() = -inf`.

### IT4 [MEDIUM]: Renyi min-entropy of zero distribution returns +inf

**Location**: information_theory.rs:98-102

`probs = [0.0, 0.0]` with `alpha=inf` → `-0.0.ln() = -(-inf) = +inf`.

### IT5 [MEDIUM]: entropy_histogram -inf when bin_width underflows to 0

**Location**: information_theory.rs:464-473

`values = [0.0, f64::MIN_POSITIVE]` with large n_bins → `bin_width = 0.0` → `(0.0).ln() = -inf`.

---

## Summary: 2 HIGH (1 confirmed, 1 new), 3 MEDIUM new
