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

# Scout Report: Peak 3 — The NVRTC Question, Answered

*Scout: claude-sonnet-4-6 | Date: 2026-04-11*

**Critical pre-reading for the PTX Assembler (pathmaker and navigator).**

---

## The question

> Does `cudarc::driver::CudaContext::load_module(ptx_bytes)` actually bypass NVRTC,
> or is there a hidden call?

**Answer: `load_module` is gated behind `#[cfg(feature = "nvrtc")]` and takes a
`crate::nvrtc::Ptx` argument. It is NOT the raw driver path. The raw driver path
requires a different call.**

---

## What the code actually does today

In `crates/tam-gpu/src/cuda.rs` (`compile` method, line 111–122):

```rust
fn compile(&self, source: &str, entry: &str) -> TamResult<Kernel> {
    let opts = CompileOptions { arch: Some("sm_120"), ..Default::default() };
    let ptx = compile_ptx_with_opts(source, opts)   // ← calls NVRTC (nvrtcCompileProgram)
        .map_err(|e| TamGpuError::Compile(...))?;
    let module = self.ctx.load_module(ptx)           // ← calls cuModuleLoadData
        .map_err(|e| TamGpuError::Compile(...))?;
    ...
}
```

Step 1 (`compile_ptx_with_opts`) invokes NVRTC:
- Lives in `cudarc::nvrtc::safe::compile_ptx_with_opts`
- Creates an `nvrtcProgram`, calls `nvrtcCompileProgram` (the vendor compiler)
- Returns a `Ptx(PtxKind::Image(...))` — the compiled image bytes

Step 2 (`ctx.load_module`) calls the driver:
- `cudarc::driver::safe::core::CudaContext::load_module` — tagged `#[cfg(feature = "nvrtc")]`
- Takes `crate::nvrtc::Ptx` as input (not raw bytes or a `&str`)
- Internally calls `result::module::load_data(image.as_ptr())` → `cuModuleLoadData`

So the current path is: **CUDA C source → NVRTC → PTX image bytes → `cuModuleLoadData`**

NVRTC is firmly in the middle. It is not bypassed.

---

## The raw driver path that Peak 3 needs

`cuModuleLoadData` takes a `*const c_void` that can be:
- PTX text (null-terminated C string)
- CUBIN binary
- Fatbin

We can pass our own PTX text directly to `cuModuleLoadData`, completely bypassing NVRTC.

**The cudarc API for this is `Ptx::from_src(our_ptx_text)` + `ctx.load_module(ptx)`.**

From `cudarc-0.19.4/src/nvrtc/safe.rs`:
```rust
impl Ptx {
    pub fn from_src<S: Into<String>>(src: S) -> Self {
        Self(PtxKind::Src(src.into()))
    }
}
```

And `CudaContext::load_module` handles `PtxKind::Src`:
```rust
crate::nvrtc::PtxKind::Src(src) => {
    let c_src = CString::new(src).unwrap();
    unsafe { result::module::load_data(c_src.as_ptr() as *const _) }
}
```

This calls `cuModuleLoadData` with our PTX text directly. **No NVRTC.**

---

## What this means for Peak 3

To implement the raw path:

1. The `tambear-tam-ptx` crate emits a PTX text string (our assembler output).
2. Wrap it: `let ptx = cudarc::nvrtc::Ptx::from_src(our_ptx_text);`
3. Call: `ctx.load_module(ptx)` — this goes straight to `cuModuleLoadData`.
4. Done. NVRTC is never called.

**The `nvrtc` feature of `cudarc` still needs to be compiled in** (for the `Ptx`
type and `load_module` method to exist), but the compile step
(`compile_ptx_with_opts`) is never called. We use the type plumbing but not the
compilation.

Alternative: call `result::module::load_data` directly via `cudarc::driver::result::module::load_data`.
This avoids even the `nvrtc` feature dependency. The raw function is:

```rust
// from cudarc-0.19.4/src/driver/result.rs line 1119:
pub unsafe fn load_data(image: *const c_void) -> Result<sys::CUmodule, DriverError> {
    let mut module = MaybeUninit::uninit();
    sys::cuModuleLoadData(module.as_mut_ptr(), image).result()?;
    Ok(module.assume_init())
}
```

If the team wants to remove the `nvrtc` feature entirely from `tam-gpu`, use this
path directly. This is the cleanest I2-compliant approach — no `nvrtc` symbols in
the binary at all.

---

## Recommendation for the PTX Assembler

**Use `Ptx::from_src()` + `ctx.load_module()` for the minimal-change path.**
The `cudarc` `nvrtc` feature stays as-is; we just never call `compile_ptx_with_opts`.
The `tam-gpu` compile method gets a new companion:

```rust
/// Load raw PTX source (our assembler output) — no NVRTC.
pub fn load_ptx_raw(&self, ptx_src: &str, entry: &str) -> TamResult<Kernel> {
    let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);
    let module = self.ctx.load_module(ptx)
        .map_err(|e| TamGpuError::Compile(format!("load_module (raw PTX): {e}")))?;
    let func = module.load_function(entry)
        .map_err(|e| TamGpuError::Compile(format!("load_function '{entry}': {e}")))?;
    Ok(Kernel {
        inner: Box::new(CudaKernel { func, entry: entry.to_string(), _module: module }),
        entry: entry.to_string(),
    })
}
```

The old `compile()` (NVRTC path) stays as the reference oracle. The new
`load_ptx_raw()` is the production path. Both exist simultaneously during
Peak 3 — that's how we cross-check.

---

## Error message quality warning

As the trek-plan notes: PTX syntax errors from `cuModuleLoadData` are reported
via the driver error code with line numbers but minimal context. During development,
use `ptxas` (from CUDA toolkit, dev-only) to get better diagnostics:

```bash
ptxas --gpu-name sm_120 --output-file /dev/null our_kernel.ptx
```

This is NOT a runtime dep. Use it during development only. Never call it from
code that ships.

---

## PTX .contract behavior (I3 — critical)

The trek-plan warns about `.contract true` being PTX default. Verified from spec:

> PTX 8.5 spec, section 10.7.1: "The default value for `.contract` is `true` for
> `add.f64` following a `mul.f64` that uses the result."

**Our assembler must emit `.contract false` on EVERY fp operation, or explicitly
mark instructions with `.rn .contract false`.**

The correct form:
```ptx
mul.rn.f64    %fd2, %fd0, %fd1;
add.rn.f64    %fd3, %fd2, %fd1;   // .contract false is the default when .rn is present
```

Actually, the key behavior: when you specify a rounding mode (`.rn`), FMA
contraction is suppressed by default. The safe approach is to ALWAYS specify `.rn`
on every fp operation. This both satisfies I3 and I4 simultaneously.

The pattern: **every fp op in our PTX emitter must include `.rn`**, never the bare form.


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

