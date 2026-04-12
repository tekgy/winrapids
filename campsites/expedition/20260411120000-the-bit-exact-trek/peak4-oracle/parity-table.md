# Parity Table — Peak 4 Oracle

**Purpose:** Running record of what each reference implementation says for each recipe/function, and whether tambear agrees.

**Legend:**
- `?` — not yet measured
- `MATCH` — bit-exact agreement (or within documented ULP for transcendentals)
- `DIFF(n ULP)` — differs by n ULP — under investigation
- `BUG(impl)` — we found a bug in the reference; filed upstream

**The oracle:** mpmath at ≥50-digit precision is the ground truth (I9).
R and Python are peers, not authorities. When they disagree with each other
or with tambear, mpmath breaks the tie.

---

## Pure arithmetic (should be bit-exact across all backends)

| Recipe | tambear | R (`Rscript`) | Python (`numpy`) | mpmath (50-dig) | CPU↔GPU | Notes |
|--------|---------|---------------|------------------|-----------------|---------|-------|
| sum | ? | ? | ? | ? | ? | Pending Peak 5 |
| mean | ? | ? | ? | ? | ? | Pending Peak 5 |
| variance | **BUG(tambear)** | CORRECT | CORRECT | CORRECT | — | One-pass formula: 0.0 vs true 0.087 on data∈[1e9,1e9+1) — 100% error. Fix: two-pass .tam IR (pathmaker 1.4). See variance-numerical-analysis.md |
| sum_sq | ? | ? | ? | ? | ? | Pending Peak 5 |
| l1_norm | ? | ? | ? | ? | ? | Pending Peak 5 |
| rms | ? | ? | ? | ? | ? | Pending Peak 5 |
| pearson_r | ? | ? | ? | ? | ? | Pending Peak 5 |

## Transcendentals (ULP-bounded; mpmath is the oracle)

| Function | tambear max ULP | R max ULP | Python max ULP | mpmath ref | Notes |
|----------|-----------------|-----------|----------------|------------|-------|
| tam_sqrt | ? | ? | ? | — (IEEE exact) | Pending Peak 2 |
| tam_exp | ? | ? | ? | ? | Pending Peak 2 |
| tam_ln | ? | ? | ? | ? | Pending Peak 2 |
| tam_sin | ? | ? | ? | ? | Pending Peak 2 |
| tam_cos | ? | ? | ? | ? | Pending Peak 2 |
| tam_pow | ? | ? | ? | ? | Pending Peak 2 |
| tam_tanh | ? | ? | ? | ? | Pending Peak 2 |
| tam_atan | ? | ? | ? | ? | Pending Peak 2 |
| tam_atan2 | ? | ? | ? | ? | Pending Peak 2 |

## Special-value behavior

| Function | Input | Expected | tambear | Notes |
|----------|-------|----------|---------|-------|
| tam_ln | 0.0 | -inf | ? | Pending Peak 2 |
| tam_ln | -1.0 | NaN | ? | Pending Peak 2 |
| tam_ln | 1.0 | 0.0 (exact) | ? | Pending Peak 2 |
| tam_exp | 1000.0 | +inf | ? | Pending Peak 2 |
| tam_exp | -700.0 | subnormal | ? | must not flush to 0 |
| tam_pow | 0.0, 0.0 | 1.0 (convention) | ? | Pending Peak 2 |
| tam_atan2 | 0.0, 0.0 | 0.0 (convention) | ? | Pending Peak 2 |

---

## Update log

| Date | What changed |
|------|-------------|
| 2026-04-11 | Variance row updated: BUG(tambear) confirmed — one-pass formula gives 0.0 on near-1e9 data. See variance-numerical-analysis.md for full quantitative evidence and two-pass design. |
| 2026-04-11 | Table created; all entries pending (harness skeleton landed) |
