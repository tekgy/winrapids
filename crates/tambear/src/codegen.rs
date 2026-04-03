//! Multi-backend kernel codegen.
//!
//! Same phi expression → same semantics on any GPU backend.
//! Emits CUDA C (for NVRTC) or WGSL (for wgpu/naga → SPIR-V/MSL/HLSL).
//!
//! ```text
//! "Tam doesn't need CUDA. Tam needs cores."
//! ```
//!
//! ## Phi expression convention
//!
//! Phi expressions are arithmetic strings over named variables:
//! - Scatter: `v` (value), `r` (group ref), `g` (group index)
//! - Map: `v` (value)
//! - Map2: `a`, `b` (two inputs)
//!
//! Use float literals (`1.0`, not `1`). Math functions (`log`, `exp`, `sqrt`,
//! `abs`, `pow`) translate automatically between CUDA C and WGSL.

/// Target language for kernel codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodegenTarget {
    /// CUDA C — compiled via NVRTC to PTX.
    CudaC,
    /// WGSL — compiled via wgpu/naga to SPIR-V (Vulkan), MSL (Metal), or HLSL (DX12).
    Wgsl,
}

/// Generated kernel source with its entry point name.
#[derive(Debug, Clone)]
pub struct KernelSource {
    /// Complete kernel source in the target language.
    pub source: String,
    /// Entry point function name.
    pub entry: String,
    /// Target language that produced this source.
    pub target: CodegenTarget,
}

// ---------------------------------------------------------------------------
// Phi expression translation
// ---------------------------------------------------------------------------

/// Translate a phi expression from CUDA-flavored C to WGSL.
///
/// Simple arithmetic (`v * v`, `v - r`, `1.0`) is identical in both languages.
/// Known differences:
/// - `fabs(x)` → `abs(x)`
/// - `(double)` casts → removed (WGSL uses `f32`)
fn translate_phi_to_wgsl(phi_expr: &str) -> String {
    phi_expr
        .replace("fabs(", "abs(")
        .replace("(double)", "")
}

// ===========================================================================
// Scatter: output[key[i]] += phi(v, r, g)
// ===========================================================================

/// Emit a scatter kernel: `output[key[i]] += phi(v, r, g)` for each element.
///
/// Variables available to phi:
/// - `v`: current value (f64 in CUDA, f32 in WGSL)
/// - `r`: per-group ref value
/// - `g`: group index (int/u32)
pub fn emit_scatter(phi_expr: &str, target: CodegenTarget) -> KernelSource {
    match target {
        CodegenTarget::CudaC => emit_scatter_cuda(phi_expr),
        CodegenTarget::Wgsl => emit_scatter_wgsl(phi_expr),
    }
}

fn emit_scatter_cuda(phi_expr: &str) -> KernelSource {
    let source = format!(
        r#"
// JIT scatter kernel: output[g] += phi(v) for elements in group g.
// phi = {phi}
extern "C" __global__ void scatter_phi(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    const double* __restrict__ refs,
    double* __restrict__ output,
    int n
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {{
        int g = keys[gid];
        double v = values[gid];
        double r = refs[g];
        double phi = ({phi});
        atomicAdd(&output[g], phi);
    }}
}}
"#,
        phi = phi_expr
    );
    KernelSource { source, entry: "scatter_phi".into(), target: CodegenTarget::CudaC }
}

fn emit_scatter_wgsl(phi_expr: &str) -> KernelSource {
    let phi_wgsl = translate_phi_to_wgsl(phi_expr);
    let source = format!(
        r#"// JIT scatter kernel: output[g] += phi(v) for elements in group g.
// phi = {phi}
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> refs: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: vec2<u32>;

fn atomic_add_f32(idx: u32, val: f32) {{
    var old_bits = atomicLoad(&output[idx]);
    loop {{
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&output[idx], old_bits, new_bits);
        if result.exchanged {{ break; }}
        old_bits = result.old_value;
    }}
}}

@compute @workgroup_size(256)
fn scatter_phi(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.x) {{ return; }}
    let g = keys[idx];
    let v = values[idx];
    let r = refs[g];
    let phi: f32 = ({phi_wgsl});
    atomic_add_f32(g, phi);
}}
"#,
        phi = phi_expr,
        phi_wgsl = phi_wgsl,
    );
    KernelSource { source, entry: "scatter_phi".into(), target: CodegenTarget::Wgsl }
}

// ===========================================================================
// Scatter Multi: N fused outputs in one pass
// ===========================================================================

/// Emit a fused scatter kernel with N phi expressions, one memory pass.
///
/// Each phi produces its own output array. One atomicAdd per (element, phi).
pub fn emit_scatter_multi(phi_exprs: &[&str], target: CodegenTarget) -> KernelSource {
    match target {
        CodegenTarget::CudaC => emit_scatter_multi_cuda(phi_exprs),
        CodegenTarget::Wgsl => emit_scatter_multi_wgsl(phi_exprs),
    }
}

fn emit_scatter_multi_cuda(phi_exprs: &[&str]) -> KernelSource {
    let n = phi_exprs.len();

    let out_params: String = (0..n)
        .map(|i| format!("    double* __restrict__ out{},\n", i))
        .collect();

    let atomic_adds: String = phi_exprs
        .iter()
        .enumerate()
        .map(|(i, phi)| format!("        atomicAdd(&out{}[g], ({}));\n", i, phi))
        .collect();

    let source = format!(
        r#"
// JIT fused scatter kernel: {n} outputs, one memory pass.
// phi expressions: [{phis}]
extern "C" __global__ void scatter_multi_phi(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    const double* __restrict__ refs,
{out_params}    int n
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {{
        int g = keys[gid];
        double v = values[gid];
        double r = refs[g];
{atomic_adds}    }}
}}
"#,
        n = n,
        phis = phi_exprs.join(", "),
        out_params = out_params,
        atomic_adds = atomic_adds,
    );
    KernelSource { source, entry: "scatter_multi_phi".into(), target: CodegenTarget::CudaC }
}

fn emit_scatter_multi_wgsl(phi_exprs: &[&str]) -> KernelSource {
    let n = phi_exprs.len();

    // Bindings: 0=keys, 1=values, 2=refs, 3..3+n-1=out0..outN-1, 3+n=params
    let out_bindings: String = (0..n)
        .map(|i| {
            format!(
                "@group(0) @binding({}) var<storage, read_write> out{}: array<atomic<u32>>;\n",
                3 + i,
                i
            )
        })
        .collect();

    let params_binding = 3 + n;

    let atomic_adds: String = phi_exprs
        .iter()
        .enumerate()
        .map(|(i, phi)| {
            let phi_wgsl = translate_phi_to_wgsl(phi);
            format!("    atomic_add_f32_out{}(g, ({}));\n", i, phi_wgsl)
        })
        .collect();

    // Generate one atomic_add helper per output (each references a different binding).
    let atomic_helpers: String = (0..n)
        .map(|i| {
            format!(
                r#"fn atomic_add_f32_out{i}(idx: u32, val: f32) {{
    var old_bits = atomicLoad(&out{i}[idx]);
    loop {{
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&out{i}[idx], old_bits, new_bits);
        if result.exchanged {{ break; }}
        old_bits = result.old_value;
    }}
}}

"#,
                i = i
            )
        })
        .collect();

    let source = format!(
        r#"// JIT fused scatter kernel: {n} outputs, one memory pass.
// phi expressions: [{phis}]
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> refs: array<f32>;
{out_bindings}@group(0) @binding({params_binding}) var<uniform> params: vec2<u32>;

{atomic_helpers}@compute @workgroup_size(256)
fn scatter_multi_phi(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.x) {{ return; }}
    let g = keys[idx];
    let v = values[idx];
    let r = refs[g];
{atomic_adds}}}
"#,
        n = n,
        phis = phi_exprs.join(", "),
        out_bindings = out_bindings,
        params_binding = params_binding,
        atomic_helpers = atomic_helpers,
        atomic_adds = atomic_adds,
    );
    KernelSource { source, entry: "scatter_multi_phi".into(), target: CodegenTarget::Wgsl }
}

// ===========================================================================
// Scatter Masked: output[key[i]] += phi(v, r, g) where mask bit is set
// ===========================================================================

/// Emit a masked scatter kernel: scatter only rows where the bitmask bit is 1.
///
/// Mask format:
/// - CUDA: packed u64 (`unsigned long long`), bit `gid` of `mask[gid/64]`
/// - WGSL: packed u32, bit `idx` of `mask[idx/32]`
pub fn emit_scatter_masked(phi_expr: &str, target: CodegenTarget) -> KernelSource {
    match target {
        CodegenTarget::CudaC => emit_scatter_masked_cuda(phi_expr),
        CodegenTarget::Wgsl => emit_scatter_masked_wgsl(phi_expr),
    }
}

fn emit_scatter_masked_cuda(phi_expr: &str) -> KernelSource {
    let source = format!(
        r#"
// JIT masked scatter: output[g] += phi(v) for rows where mask bit is set.
// phi = {phi}
// mask: packed u64, bit gid of mask[gid>>6] = 1 means row gid passes.
extern "C" __global__ void scatter_phi_masked(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    const double* __restrict__ refs,
    const unsigned long long* __restrict__ mask,
    double* __restrict__ output,
    int n
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {{
        unsigned long long word = mask[gid >> 6];
        if ((word >> (gid & 63)) & 1ULL) {{
            int g = keys[gid];
            double v = values[gid];
            double r = refs[g];
            atomicAdd(&output[g], ({phi}));
        }}
    }}
}}
"#,
        phi = phi_expr
    );
    KernelSource { source, entry: "scatter_phi_masked".into(), target: CodegenTarget::CudaC }
}

fn emit_scatter_masked_wgsl(phi_expr: &str) -> KernelSource {
    let phi_wgsl = translate_phi_to_wgsl(phi_expr);
    let source = format!(
        r#"// JIT masked scatter: output[g] += phi(v) for rows where mask bit is set.
// phi = {phi}
// mask: packed u32, bit idx of mask[idx>>5] = 1 means row idx passes.
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> refs: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: vec2<u32>;

fn atomic_add_f32(idx: u32, val: f32) {{
    var old_bits = atomicLoad(&output[idx]);
    loop {{
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&output[idx], old_bits, new_bits);
        if result.exchanged {{ break; }}
        old_bits = result.old_value;
    }}
}}

@compute @workgroup_size(256)
fn scatter_phi_masked(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.x) {{ return; }}
    let word = mask[idx >> 5u];
    if ((word >> (idx & 31u)) & 1u) == 0u {{ return; }}
    let g = keys[idx];
    let v = values[idx];
    let r = refs[g];
    let phi: f32 = ({phi_wgsl});
    atomic_add_f32(g, phi);
}}
"#,
        phi = phi_expr,
        phi_wgsl = phi_wgsl,
    );
    KernelSource { source, entry: "scatter_phi_masked".into(), target: CodegenTarget::Wgsl }
}

// ===========================================================================
// Map: output[i] = phi(v[i])
// ===========================================================================

/// Emit a unary element-wise map kernel: `output[i] = phi(values[i])`.
///
/// Variables available to phi: `v` (the current value).
pub fn emit_map(phi_expr: &str, target: CodegenTarget) -> KernelSource {
    match target {
        CodegenTarget::CudaC => emit_map_cuda(phi_expr),
        CodegenTarget::Wgsl => emit_map_wgsl(phi_expr),
    }
}

fn emit_map_cuda(phi_expr: &str) -> KernelSource {
    let source = format!(
        r#"
// JIT map kernel: output[i] = phi(v[i]) for each element.
// phi = {phi}
extern "C" __global__ void map_phi_kernel(
    const double* __restrict__ values,
    double* __restrict__ output,
    int n
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {{
        double v = values[gid];
        output[gid] = ({phi});
    }}
}}
"#,
        phi = phi_expr
    );
    KernelSource { source, entry: "map_phi_kernel".into(), target: CodegenTarget::CudaC }
}

fn emit_map_wgsl(phi_expr: &str) -> KernelSource {
    let phi_wgsl = translate_phi_to_wgsl(phi_expr);
    let source = format!(
        r#"// JIT map kernel: output[i] = phi(v[i]) for each element.
// phi = {phi}
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

@compute @workgroup_size(256)
fn map_phi_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.x) {{ return; }}
    let v = values[idx];
    output[idx] = ({phi_wgsl});
}}
"#,
        phi = phi_expr,
        phi_wgsl = phi_wgsl,
    );
    KernelSource { source, entry: "map_phi_kernel".into(), target: CodegenTarget::Wgsl }
}

// ===========================================================================
// Map2: output[i] = phi(a[i], b[i])
// ===========================================================================

/// Emit a binary element-wise map kernel: `output[i] = phi(a[i], b[i])`.
///
/// Variables available to phi: `a`, `b` (the two input values).
pub fn emit_map2(phi_expr: &str, target: CodegenTarget) -> KernelSource {
    match target {
        CodegenTarget::CudaC => emit_map2_cuda(phi_expr),
        CodegenTarget::Wgsl => emit_map2_wgsl(phi_expr),
    }
}

fn emit_map2_cuda(phi_expr: &str) -> KernelSource {
    let source = format!(
        r#"
// JIT two-input map kernel: output[i] = phi(a[i], b[i]).
// phi = {phi}
extern "C" __global__ void map_phi2_kernel(
    const double* __restrict__ vals_a,
    const double* __restrict__ vals_b,
    double* __restrict__ output,
    int n
) {{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {{
        double a = vals_a[gid];
        double b = vals_b[gid];
        output[gid] = ({phi});
    }}
}}
"#,
        phi = phi_expr
    );
    KernelSource { source, entry: "map_phi2_kernel".into(), target: CodegenTarget::CudaC }
}

fn emit_map2_wgsl(phi_expr: &str) -> KernelSource {
    let phi_wgsl = translate_phi_to_wgsl(phi_expr);
    let source = format!(
        r#"// JIT two-input map kernel: output[i] = phi(a[i], b[i]).
// phi = {phi}
@group(0) @binding(0) var<storage, read> vals_a: array<f32>;
@group(0) @binding(1) var<storage, read> vals_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec2<u32>;

@compute @workgroup_size(256)
fn map_phi2_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.x) {{ return; }}
    let a = vals_a[idx];
    let b = vals_b[idx];
    output[idx] = ({phi_wgsl});
}}
"#,
        phi = phi_expr,
        phi_wgsl = phi_wgsl,
    );
    KernelSource { source, entry: "map_phi2_kernel".into(), target: CodegenTarget::Wgsl }
}

// ===========================================================================
// Reduce Sum: tree reduction in shared memory
// ===========================================================================

/// Emit a tree-reduction sum kernel.
///
/// Each workgroup reduces its elements in shared memory, then atomicAdds
/// the partial sum to a global partials array. A second pass (or host code)
/// reduces the partials.
pub fn emit_reduce_sum(target: CodegenTarget) -> KernelSource {
    match target {
        CodegenTarget::CudaC => emit_reduce_sum_cuda(),
        CodegenTarget::Wgsl => emit_reduce_sum_wgsl(),
    }
}

fn emit_reduce_sum_cuda() -> KernelSource {
    let source = r#"
// JIT reduce sum: tree reduction per block, atomicAdd partials.
extern "C" __global__ void reduce_sum(
    const double* __restrict__ input,
    double* __restrict__ partials,
    int n
) {
    extern __shared__ double shmem[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    shmem[lid] = (gid < n) ? input[gid] : 0.0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shmem[lid] += shmem[lid + stride];
        }
        __syncthreads();
    }

    if (lid == 0) {
        atomicAdd(&partials[blockIdx.x], shmem[0]);
    }
}
"#
    .to_string();
    KernelSource { source, entry: "reduce_sum".into(), target: CodegenTarget::CudaC }
}

fn emit_reduce_sum_wgsl() -> KernelSource {
    let source = r#"// JIT reduce sum: tree reduction per workgroup, atomic partial sums.
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

var<workgroup> shmem: array<f32, 256>;

fn atomic_add_f32(idx: u32, val: f32) {
    var old_bits = atomicLoad(&partials[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&partials[idx], old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}

@compute @workgroup_size(256)
fn reduce_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let idx = gid.x;
    if (idx < params.x) {
        shmem[lid.x] = input[idx];
    } else {
        shmem[lid.x] = 0.0;
    }
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if stride == 0u { break; }
        if lid.x < stride {
            shmem[lid.x] = shmem[lid.x] + shmem[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if lid.x == 0u {
        atomic_add_f32(wid.x, shmem[0]);
    }
}
"#
    .to_string();
    KernelSource { source, entry: "reduce_sum".into(), target: CodegenTarget::Wgsl }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Scatter
    // -----------------------------------------------------------------------

    #[test]
    fn scatter_cuda_contains_phi() {
        let ks = emit_scatter("v * v", CodegenTarget::CudaC);
        assert_eq!(ks.entry, "scatter_phi");
        assert!(ks.source.contains("extern \"C\" __global__"));
        assert!(ks.source.contains("double phi = (v * v)"));
        assert!(ks.source.contains("atomicAdd(&output[g], phi)"));
    }

    #[test]
    fn scatter_wgsl_contains_phi() {
        let ks = emit_scatter("v * v", CodegenTarget::Wgsl);
        assert_eq!(ks.entry, "scatter_phi");
        assert!(ks.source.contains("@compute @workgroup_size(256)"));
        assert!(ks.source.contains("let phi: f32 = (v * v)"));
        assert!(ks.source.contains("atomic_add_f32(g, phi)"));
        assert!(ks.source.contains("array<atomic<u32>>"));
    }

    #[test]
    fn scatter_both_targets_same_entry() {
        let cuda = emit_scatter("v", CodegenTarget::CudaC);
        let wgsl = emit_scatter("v", CodegenTarget::Wgsl);
        assert_eq!(cuda.entry, wgsl.entry);
    }

    #[test]
    fn scatter_phi_translation_fabs() {
        let ks = emit_scatter("fabs(v - r)", CodegenTarget::Wgsl);
        assert!(ks.source.contains("abs(v - r)"), "fabs should translate to abs");
        // The comment preserves the original phi; check that the executable code uses abs.
        assert!(ks.source.contains("let phi: f32 = (abs(v - r))"));
    }

    // -----------------------------------------------------------------------
    // Scatter Multi
    // -----------------------------------------------------------------------

    #[test]
    fn scatter_multi_cuda_3_outputs() {
        let ks = emit_scatter_multi(&["v", "v * v", "1.0"], CodegenTarget::CudaC);
        assert_eq!(ks.entry, "scatter_multi_phi");
        assert!(ks.source.contains("double* __restrict__ out0"));
        assert!(ks.source.contains("double* __restrict__ out1"));
        assert!(ks.source.contains("double* __restrict__ out2"));
        assert!(ks.source.contains("atomicAdd(&out0[g], (v))"));
        assert!(ks.source.contains("atomicAdd(&out1[g], (v * v))"));
        assert!(ks.source.contains("atomicAdd(&out2[g], (1.0))"));
    }

    #[test]
    fn scatter_multi_wgsl_3_outputs() {
        let ks = emit_scatter_multi(&["v", "v * v", "1.0"], CodegenTarget::Wgsl);
        assert_eq!(ks.entry, "scatter_multi_phi");
        // 3 output bindings at indices 3, 4, 5
        assert!(ks.source.contains("@binding(3)"));
        assert!(ks.source.contains("@binding(4)"));
        assert!(ks.source.contains("@binding(5)"));
        // params at binding 6
        assert!(ks.source.contains("@binding(6)"));
        // 3 atomic add helpers
        assert!(ks.source.contains("atomic_add_f32_out0"));
        assert!(ks.source.contains("atomic_add_f32_out1"));
        assert!(ks.source.contains("atomic_add_f32_out2"));
    }

    #[test]
    fn scatter_multi_wgsl_bindings_contiguous() {
        // Verify that N outputs produce bindings 3..3+N, params at 3+N
        let ks = emit_scatter_multi(&["v", "v * v"], CodegenTarget::Wgsl);
        assert!(ks.source.contains("@binding(3)")); // out0
        assert!(ks.source.contains("@binding(4)")); // out1
        assert!(ks.source.contains("@binding(5)")); // params
        assert!(!ks.source.contains("@binding(6)")); // nothing beyond
    }

    // -----------------------------------------------------------------------
    // Scatter Masked
    // -----------------------------------------------------------------------

    #[test]
    fn scatter_masked_cuda_structure() {
        let ks = emit_scatter_masked("v * v", CodegenTarget::CudaC);
        assert_eq!(ks.entry, "scatter_phi_masked");
        assert!(ks.source.contains("unsigned long long* __restrict__ mask"));
        assert!(ks.source.contains("mask[gid >> 6]"));
        assert!(ks.source.contains("(word >> (gid & 63)) & 1ULL"));
        assert!(ks.source.contains("atomicAdd(&output[g], (v * v))"));
    }

    #[test]
    fn scatter_masked_wgsl_structure() {
        let ks = emit_scatter_masked("v * v", CodegenTarget::Wgsl);
        assert_eq!(ks.entry, "scatter_phi_masked");
        assert!(ks.source.contains("mask: array<u32>"));
        assert!(ks.source.contains("mask[idx >> 5u]"));
        assert!(ks.source.contains("(word >> (idx & 31u)) & 1u"));
        assert!(ks.source.contains("@binding(3)")); // mask
        assert!(ks.source.contains("@binding(4)")); // output
        assert!(ks.source.contains("@binding(5)")); // params
    }

    #[test]
    fn scatter_masked_both_targets_same_entry() {
        let cuda = emit_scatter_masked("v", CodegenTarget::CudaC);
        let wgsl = emit_scatter_masked("v", CodegenTarget::Wgsl);
        assert_eq!(cuda.entry, wgsl.entry);
    }

    // -----------------------------------------------------------------------
    // Map
    // -----------------------------------------------------------------------

    #[test]
    fn map_cuda_basic() {
        let ks = emit_map("v * v + 1.0", CodegenTarget::CudaC);
        assert_eq!(ks.entry, "map_phi_kernel");
        assert!(ks.source.contains("output[gid] = (v * v + 1.0)"));
        assert!(!ks.source.contains("atomicAdd"), "map should not use atomics");
    }

    #[test]
    fn map_wgsl_basic() {
        let ks = emit_map("v * v + 1.0", CodegenTarget::Wgsl);
        assert_eq!(ks.entry, "map_phi_kernel");
        assert!(ks.source.contains("output[idx] = (v * v + 1.0)"));
        assert!(!ks.source.contains("atomic"), "map should not use atomics");
    }

    // -----------------------------------------------------------------------
    // Map2
    // -----------------------------------------------------------------------

    #[test]
    fn map2_cuda_basic() {
        let ks = emit_map2("a * b + 1.0", CodegenTarget::CudaC);
        assert_eq!(ks.entry, "map_phi2_kernel");
        assert!(ks.source.contains("output[gid] = (a * b + 1.0)"));
        assert!(ks.source.contains("double a = vals_a[gid]"));
        assert!(ks.source.contains("double b = vals_b[gid]"));
    }

    #[test]
    fn map2_wgsl_basic() {
        let ks = emit_map2("a * b + 1.0", CodegenTarget::Wgsl);
        assert_eq!(ks.entry, "map_phi2_kernel");
        assert!(ks.source.contains("output[idx] = (a * b + 1.0)"));
        assert!(ks.source.contains("let a = vals_a[idx]"));
        assert!(ks.source.contains("let b = vals_b[idx]"));
    }

    // -----------------------------------------------------------------------
    // Reduce Sum
    // -----------------------------------------------------------------------

    #[test]
    fn reduce_sum_cuda_structure() {
        let ks = emit_reduce_sum(CodegenTarget::CudaC);
        assert_eq!(ks.entry, "reduce_sum");
        assert!(ks.source.contains("extern __shared__ double shmem[]"));
        assert!(ks.source.contains("__syncthreads()"));
        assert!(ks.source.contains("atomicAdd(&partials[blockIdx.x], shmem[0])"));
    }

    #[test]
    fn reduce_sum_wgsl_structure() {
        let ks = emit_reduce_sum(CodegenTarget::Wgsl);
        assert_eq!(ks.entry, "reduce_sum");
        assert!(ks.source.contains("var<workgroup> shmem: array<f32, 256>"));
        assert!(ks.source.contains("workgroupBarrier()"));
        assert!(ks.source.contains("atomic_add_f32(wid.x, shmem[0])"));
    }

    // -----------------------------------------------------------------------
    // Cross-target invariants
    // -----------------------------------------------------------------------

    #[test]
    fn same_phi_both_targets_same_entry() {
        let exprs = &["v", "v * v", "(v - r) * (v - r)", "1.0"];
        for phi in exprs {
            let cuda = emit_scatter(phi, CodegenTarget::CudaC);
            let wgsl = emit_scatter(phi, CodegenTarget::Wgsl);
            assert_eq!(cuda.entry, wgsl.entry, "entry mismatch for phi={}", phi);
            assert_eq!(cuda.target, CodegenTarget::CudaC);
            assert_eq!(wgsl.target, CodegenTarget::Wgsl);
        }
    }

    #[test]
    fn cuda_output_matches_existing_scatter_jit() {
        // The CUDA output should match the pattern from scatter_jit::build_scatter_phi_source.
        // Key invariants: same param order, same variable names, same atomicAdd pattern.
        let ks = emit_scatter("v", CodegenTarget::CudaC);
        assert!(ks.source.contains("const int* __restrict__ keys"));
        assert!(ks.source.contains("const double* __restrict__ values"));
        assert!(ks.source.contains("const double* __restrict__ refs"));
        assert!(ks.source.contains("double* __restrict__ output"));
        assert!(ks.source.contains("int n"));
        assert!(ks.source.contains("int g = keys[gid]"));
        assert!(ks.source.contains("double v = values[gid]"));
        assert!(ks.source.contains("double r = refs[g]"));
    }

    #[test]
    fn wgsl_scatter_has_required_bindings() {
        let ks = emit_scatter("v", CodegenTarget::Wgsl);
        assert!(ks.source.contains("@group(0) @binding(0)"));
        assert!(ks.source.contains("@group(0) @binding(1)"));
        assert!(ks.source.contains("@group(0) @binding(2)"));
        assert!(ks.source.contains("@group(0) @binding(3)"));
        assert!(ks.source.contains("@group(0) @binding(4)"));
    }

    /// Verify all 5 well-known phi expressions generate valid kernels for both targets.
    #[test]
    fn all_standard_phis_codegen() {
        let phis = &["v", "v * v", "(v - r)", "(v - r) * (v - r)", "1.0"];
        for target in &[CodegenTarget::CudaC, CodegenTarget::Wgsl] {
            for phi in phis {
                let ks = emit_scatter(phi, *target);
                assert!(!ks.source.is_empty(), "empty source for {:?} phi={}", target, phi);
                assert!(!ks.entry.is_empty());
            }
            let ks = emit_scatter_multi(phis, *target);
            assert!(!ks.source.is_empty());
        }
    }
}
