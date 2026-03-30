# Revert Surface Map: Branch-Free → Branching Scan

*Naturalist journal — 2026-03-30*

---

## Context

Entry 022 showed the branch-free scan is a net negative (+7-18μs from padding overhead). The revert is queued after SarkkaOp confirms the picture. This journal maps exactly what needs to change.

## Two files, three kernels

The branch-free changes span two files:

### engine.rs — kernel source generation

**scan_per_block** (Phase 1):
- Line 203: `shared[tid] = lift_element(input[gid]);` — assumes padded input
- Line 224: `state_out[gid] = shared[tid];` — writes beyond n without check
- Revert: `shared[tid] = (gid < n) ? lift_element(input[gid]) : make_identity();` and `if (gid < n) state_out[gid] = shared[tid];`
- Note: `n` is already passed as a kernel argument but never read in the branch-free version. Vestigial.

**scan_block_totals** (Phase 2): No change needed. Already uses `(tid < n_blocks) ? totals[tid] : make_identity()` — this was never made branch-free because it's a single-block kernel with variable block count.

**propagate_extract** (Phase 3):
- Lines 269-273: reads `state_in[gid]` and writes outputs without `gid < n` guard
- Revert: wrap the entire body in `if (gid < n) { ... }`

### launch.rs — padding allocation

**scan_device_ptr** (device path — THE revert target):
- Lines 264-270: allocates padded buffer (`alloc_zeros(padded_n)`) + `memcpy_dtod` to copy input. This is the 5-18μs overhead.
- Revert: use `input_dev` directly (via ManuallyDrop) without padding. The state_dev and totals_dev buffers still need padded_n allocation (Blelloch tree operates on full blocks internally).
- Output buffer (`out0_dev`): can stay at padded_n allocation since we truncate on readback anyway, or shrink to n.

**scan_inclusive** (host path — keep as-is):
- Line 141-142: `padded_input.resize(padded_n, 0.0)` — Vec resize is essentially free on the host.
- No revert needed. Host-side padding has no measurable cost.

## The asymmetry

The revert is asymmetric: device path reverts, host path stays. This makes sense:
- Host path: padding is a memset in host memory, microseconds before a PCI-E transfer that dwarfs it
- Device path: padding requires a NEW GPU allocation + device-to-device copy, which is the same order of magnitude as the scan kernel itself

## Internal buffers

The state_dev, totals_dev, and out0_dev allocations use padded_n even after the revert. This is correct — the Blelloch tree's up-sweep and down-sweep operate on full BLOCK_SIZE arrays. The last block always has BLOCK_SIZE threads even if only a few have real data. The identity elements in shared memory handle the unused positions.

The difference: currently the identity comes from `lift_element(0.0)` (padding). After revert, it comes from `make_identity()` (explicit). Both are correct for all current operators, but `make_identity()` is universally safe (it's the defined identity element), while `lift(0.0)` depends on the operator tolerating zero input.

## Total revert: ~15 lines changed

- engine.rs: ~8 lines (add 3 bounds checks, restore 2 ternaries)
- launch.rs scan_device_ptr: ~7 lines (remove padding allocation, use input directly)

Small, surgical. The revert is trivially reviewable.

---

*Documenting the revert surface before it's needed, so whoever does it has a clean map. The revert is mechanical — no design decisions, no risk of regressions. Just restore the three `gid < n` checks and remove the device-path padding allocation.*
