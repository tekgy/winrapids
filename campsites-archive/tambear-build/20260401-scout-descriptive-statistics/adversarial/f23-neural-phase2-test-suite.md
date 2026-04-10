# F23 Neural — Adversarial Test Suite (Phase 2 Deep Review)

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**File**: `src/neural.rs` (1901 lines)

---

## Previously reported:
- NN1 [MEDIUM]: max_pool1d/2d NaN→NEG_INFINITY — CONFIRMED STILL PRESENT (lines 452, 500)

## New findings:

### NN2 [MEDIUM]: batch_norm batch_size=0 → inf

**Location**: neural.rs:599

`1.0 / batch_size as f64` — if batch_size=0, produces inf. No guard. Mean computation then produces `0.0 * inf = NaN`, all outputs NaN.

### NN3 [MEDIUM]: layer_norm features=0 → inf

**Location**: neural.rs:644

`1.0 / features as f64` — same pattern. If features=0, all NaN.

### NN4 [MEDIUM]: Attention all-masked row → NaN propagation

**Location**: neural.rs:968

When causal mask sets all keys to −∞ (e.g., attending to nothing), softmax computes `NEG_INFINITY - NEG_INFINITY = NaN`. All attention weights become NaN. This NaN propagates through the value projection and the rest of the network.

**Test**: `scaled_dot_product_attention` with `q = [[1,0]], k = [[0,1]], v = [[1,1]], causal=true` where all keys are in the future → all masked → NaN output.

### NN5 [LOW]: softmax all-NEG_INFINITY produces uniform 1/n instead of NaN

**Location**: neural.rs:242-245

Actually, `max_val = NEG_INFINITY`, then `v - max_val = NEG_INFINITY - NEG_INFINITY = NaN`, `exp(NaN) = NaN`, `sum = NaN`, result = `NaN/NaN = NaN`. So this IS NaN propagation, not uniform. Consistent with NN4.

### Positive findings:
- ✅ Softmax uses log-sum-exp trick correctly (line 242)
- ✅ Log-softmax computes stably (line 252)
- ✅ BCE loss clamps predictions to [eps, 1-eps] — no log(0) (line 1068)
- ✅ Focal loss clamps pt to eps — no log(0) (line 1162)
- ✅ Dropout handles p=1.0 correctly (line 783)
- ✅ Softmax backward is correct (line 258-260)
- ✅ BatchNorm uses centered variance (line 614), not naive formula

---

## Summary: 0 HIGH, 3 MEDIUM (new) + 1 MEDIUM (confirmed), 1 LOW
