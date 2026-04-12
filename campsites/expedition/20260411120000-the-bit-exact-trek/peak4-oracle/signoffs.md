# Sign-Off Log — Peak 4 Oracle

**Protocol:** an entry here means the Test Oracle has personally verified:
1. Gold-standard comparison (tambear vs R vs Python vs mpmath)
2. Synthetic ground truth (known parameters recovered from known distribution)
3. Real data agreement (CPU interpreter vs GPU backend)
4. Cross-platform agreement (every available backend)
5. Hard-cases suite (all adversarial inputs pass)

No entry = not yet certified.

---

| Recipe / Function | Date | Evidence | Backend comparison | Notes |
|-------------------|------|----------|-------------------|-------|
| *(none yet)* | — | — | — | Harness skeleton landed 2026-04-11 |

---

## When to add an entry

An entry is added when ALL of the following are true:

- [ ] tambear result matches mpmath reference at ≤ documented ULP bound
- [ ] tambear result matches R result (or discrepancy is a filed R bug)
- [ ] tambear result matches Python/numpy/scipy result (or discrepancy filed)
- [ ] CPU interpreter agrees bit-exactly with GPU backend (for pure arithmetic)
- [ ] Every hard case from `hard_cases.rs` produces expected behavior (no panic, correct special values)
- [ ] Synthetic ground truth test: known-parameter synthetic data → algorithm → recovered parameters within expected tolerance

## Escalation trigger

If tambear's result disagrees with mpmath and tambear appears correct, the
Test Oracle files this as a potential mpmath bug (unlikely but possible) and
escalates to Navigator before signing off.  We do NOT silently accept a ULP
bound wider than documented.
