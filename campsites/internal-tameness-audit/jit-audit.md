# JIT surface — internal-tameness audit (Phase B, part 1)

Auditor: adversarial  
Date: 2026-05-10  
Source: `R:\tambear\crates\tambear\src\jit\`  
Files covered: `fingerprint.rs`, `dispatcher.rs`, `using_annotation.rs`, `jit_op.rs` (pub fn surface only), `strategy.rs`, `shape.rs`, `door.rs`, `element_id.rs`, `cpu_cranelift.rs`

The JIT tier does not do arithmetic on BigFloat values — it produces kernel IR. The tameness audit here shifts from arithmetic overflow sites to **cache-key tameness**: the implicit predicate that distinct inputs produce distinct cache keys, and that the same input always produces the same cache key.

---

## fingerprint.rs — cache-key tameness

### `FingerprintHasher`

The tameness predicate for cache keys:
- **Soundness**: any two `(JitOp, Shape, PrecisionContext, params, door)` tuples that differ must produce different `CacheKey` values.
- **Completeness**: the same tuple always produces the same key.

**Intermediates and findings:**

**Finding J-1 (LINT-3 analogue — special-value dispatch consistency):** `feed_f64_bits` hashes by bit pattern, so `+0.0 ≠ -0.0` and distinct NaN payloads hash distinctly. This is the correct semantics for cache keys (two kernels seeded with `+0.0` vs `-0.0` should produce the same output but different provenance). However, `feed_f64_bits` is used for `UsingAnnotation` values. If a user parameter is an f64 threshold (e.g., `using(threshold = 0.0)`), the sign of zero affects the cache key even though both `+0.0` and `-0.0` are functionally identical threshold values. This creates a silent cache miss: `using(threshold = -0.0)` misses the cache for `using(threshold = +0.0)`.

**Severity:** LOW — requires a user to explicitly pass `-0.0` as a parameter, which is unusual. But the failure is silent: the kernel is recompiled unnecessarily, not incorrectly. Not a correctness risk, a performance risk.

**Tag:** LINT-3-adjacent. No existing lint captures "f64 parameter equality should be value-based not bit-based." Candidate for a new lint if f64 `using()` parameters become common.

---

**Finding J-2 (LINT-3 — IR_VERSION change invalidates all caches but doesn't signal users):** The version history comment at line ~45 shows that v8→v9 was a fix for a hash collision where `KnownAbsent` and `Unknown` mapped to the same tag. The version bump is the recovery mechanism. But the tameness predicate violated *before* the bump is: "tag 0x18 used for two distinct semantic purposes simultaneously" — the `feed_non_finite_claim` and `feed_strategy` both used 0x18 under v8. The positional firewall prevented actual collisions in v8, but the invariant was broken.

**Observation (not a new finding):** The tag-collision pattern is self-documenting in the version history. The antibody (version bump on any structural change) exists. What's missing is a test that FAILS if any two `feed_*` methods share a tag — a test that would have caught the v8 collision before it shipped.

**Tag:** LINT-3. Gap: no `assert_all_tags_distinct` test.

---

**Finding J-3 (LINT-4 — f64 seed without finiteness check in fingerprint):** `feed_f64_bits` takes an f64 and hashes its bits unconditionally. No check for `is_finite()` or `is_nan()`. If a NaN f64 parameter reaches the cache key, two distinct NaN bit patterns produce two distinct cache keys. This is correct for content-addressed caching (NaN parameters should produce distinct kernels), but it means the same NaN value (same bit pattern) produces stable keys — no tameness issue. This lint doesn't apply here.

---

**`feed_grouping` — `ByKey` canonicalization**

The relabeling of group keys to their first-appearance index (lines ~258–270) is a well-implemented canonicalization. A potential tameness gap: if `keys` contains `i64::MIN` and an adjacent key's relabeling produces `usize as u64`, there's a risk that the `seen.iter().position()` scan over a very large key array runs in O(n^2). This is a performance tameness issue (pathological input → pathological runtime), not a correctness tameness issue.

**Finding J-4 (Performance tameness — LINT-NEW-2 candidate):** `ByKey` with n=10,000 unique keys triggers O(n^2) `position()` scan. A HashMap-based implementation would be O(n). Not a correctness issue; a silent performance cliff.

---

## dispatcher.rs — strategy tameness

**`dispatch_strategy`:** Pure routing logic. The only tameness question: does `ForceLifted` with an op that blocks lifting always panic explicitly? Yes — `lifted_strategy_or_panic` is named to enforce this. No silent degradation.

**Finding J-5 (INFO):** The panic on `ForceLifted` for a non-liftable op is the CORRECT behavior per the spec. The tameness predicate "no silent degradation" is structurally enforced by the naming convention. Good antibody.

---

## using_annotation.rs — provenance tameness

`Provenance` is not part of the cache key (per DEC-020 sub-clause). The tameness predicate here is: provenance metadata must not silently affect the computed result.

**Finding J-6 (INFO — no issue):** `Provenance` variants carry human-readable strings but no numeric parameters that enter arithmetic. The `display_as_using_annotation` method is presentation-only. No tameness sites.

---

## Summary: jit tameness sites

| ID | File | Lint | Severity | Status |
|---|---|---|---|---|
| J-1 | fingerprint.rs | LINT-3-adjacent | LOW | `-0.0` vs `+0.0` f64 param = unnecessary cache miss |
| J-2 | fingerprint.rs | LINT-3 | LOW | No `assert_all_tags_distinct` test; past collision history |
| J-3 | fingerprint.rs | LINT-4 (N/A) | INFO | NaN f64 params hash by bit pattern — correct behavior |
| **J-2** | fingerprint.rs | LINT-3 | **MEDIUM** | Tag-collision risk unguarded by failing test |
| J-4 | fingerprint.rs | Performance | LOW | `ByKey` large-n scan is O(n^2) |
| J-5 | dispatcher.rs | INFO | — | ForceLifted panic is correct antibody |
| J-6 | using_annotation.rs | INFO | — | Provenance strings don't enter arithmetic |

**Highest-value new test from this audit:** A test that iterates all `feed_*` methods in `FingerprintHasher` and asserts all tag bytes are distinct. Would have caught the v8 tag-0x18 collision.

