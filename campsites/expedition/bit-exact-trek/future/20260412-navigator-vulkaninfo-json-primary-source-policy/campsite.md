<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

# Navigator Campsite: vulkaninfo JSON as Primary Source Policy

**Created by:** navigator
**Session:** Bit-Exact Trek, 2026-04-12
**Type:** Protocol documentation — evidence artifact policy

---

## What this is

This session surfaced a subtle gap in the expedition's documentation practice: scout confirmed device property values by running `vulkaninfo --json` and reading the JSON output, then wrote those values into `capability-matrix-vulkan-row.md`. The JSON artifact itself was not committed.

That's an evidence-chain gap. The capability matrix says "shaderSignedZeroInfNanPreserveFloat64 = true" but the primary source (the JSON output) lives only in scout's session context, which is ephemeral.

This campsite documents why this matters and what the policy should be.

---

## The evidence chain as it stands

```
[physical device]
  → vulkaninfo --json
    → [JSON output in scout's session — EPHEMERAL]
      → [capability-matrix-vulkan-row.md entries — PERMANENT]
        → [guarantee-ledger.md I11 cross-reference — PERMANENT]
          → [P2 ruling in escalations.md — PERMANENT]
```

The chain has a broken link: the permanent documents cite values derived from an ephemeral artifact. If anyone ever asks "how do we know shaderSignedZeroInfNanPreserveFloat64 = true?", the answer is "scout confirmed it in a session that no longer exists."

For expedition documentation, this is acceptable (the device is physical and can be re-queried). For publication-grade rigor (tambear's standard), it's not.

---

## Why this matters more than it seems

**1. The device changes.** Drivers update. Vulkan layers change. The JSON value that scout confirmed in April 2026 may differ from the value a future developer gets when they re-run the trek on the same hardware with a newer driver. Without the committed JSON, there's no way to detect driver regressions or layer changes.

**2. The device is specific.** The RTX 6000 Pro Blackwell in this lab is one device. The capability matrix describes this device's properties. Future trek work on different devices (RTX 5000, A100, Radeon) will need their own rows, and those rows' primary sources should also be committed.

**3. Claims in the guarantee ledger point back to the device properties.** I11 says "NaN propagation guaranteed on Vulkan because `shaderSignedZeroInfNanPreserveFloat64 = true`." That claim's truth value is conditioned on the device property value. The device property value should have a committed primary source.

**4. Crossover protection.** If two scouts independently query the device and get slightly different JSON (due to different vulkaninfo versions, different `--json` flags, different API layer loading), the committed artifact is the tiebreaker.

---

## Proposed policy

### For every device property used in a guarantee-ledger claim or a capability-matrix entry:

1. **Commit the raw primary source.** For Vulkan: `vulkaninfo --json` output. For CUDA: `deviceQuery` or `nvcc --query` output. For CPU: `cpuid` or `lscpu` output. Store at `<expedition-root>/evidence/<device-id>-<property-set>-<date>.json` (or `.txt`).

2. **In the capability matrix row, cite the evidence file.** Not just the value — the file. Example: "shaderSignedZeroInfNanPreserveFloat64 = true [source: evidence/rtx6000pro-blackwell-vulkanprops-20260411.json, field: VkPhysicalDeviceVulkanMemoryModelFeatures]"

3. **Version the evidence.** Driver version and Vulkan layer version should be logged in the evidence file header or filename. This allows future developers to know which driver generation the guarantee applies to.

4. **Mark properties as "evidence on file" vs "claimed from session."** Properties without committed evidence get a `[SESSION-CLAIM]` tag in the capability matrix, indicating they need verification before being used in a guarantee.

### Practical first step for next trek session:

Run `vulkaninfo --json` on the RTX 6000 Pro Blackwell and commit the output to `campsites/expedition/20260411120000-the-bit-exact-trek/evidence/`. Then update the capability-matrix-vulkan-row.md entries to cite the evidence file.

This is ~15 minutes of work that closes the evidence chain for the entire Peak 7 foundation.

---

## The broader principle

The pattern here generalizes: **every piece of evidence that a permanent expedition document cites should itself be committed to the expedition**. This includes:

- Device property outputs (vulkaninfo, deviceQuery, cpuid)
- Benchmark results (timing data, throughput numbers)
- Reference implementations (the specific scipy/R/MATLAB version that was tested against)
- Numerical reference values (the mpmath output that oracle comparisons are based on)
- Compiler/driver version snapshots (so "this worked on CUDA 13.0" is reproducible)

The principle is the same as publication-grade reproducibility: the paper doesn't just cite a value — it cites the method used to obtain the value, and both are permanent.

---

## Note on the "falls between roles" observation

During this session, the vulkaninfo JSON felt like it "fell between" scout (who generated it) and navigator (who needed to know about it for the guarantee ledger). Scout was focused on the capability matrix; navigator was focused on the guarantee ledger. Neither role "owned" the evidence artifact.

That's the structural gap: evidence artifacts are a separate category that doesn't map cleanly to any single role. The solution is a shared evidence directory that any role can contribute to, with a policy (this campsite) that says who is responsible for committing what.

**Proposed responsibility:** Scout owns the device evidence artifacts for terrain they've surveyed. When scout confirms a device property and uses it in the capability matrix, scout commits the raw source. Navigator's review checklist includes "does every capability matrix entry have an evidence citation?" as a line item.

See also: `navigator-p2-audit-checklist` campsite, Q2 (subnormal behavior — requires device evidence).


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

