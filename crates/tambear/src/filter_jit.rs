//! JIT-compiled predicate evaluation → packed u64 bitmask.
//!
//! Implements tambear's mask-not-filter invariant: instead of compacting
//! the data array (which destroys row indices and forces an O(n) copy),
//! filter sets bits in a packed u64 bitmask. Downstream operations are
//! mask-aware and skip rows with their bit unset.
//!
//! "Tam doesn't filter. Tam knows which rows matter."
//!
//! ## Predicate expressions
//!
//! Predicates are CUDA expressions in terms of `v` (the current value, double):
//! - `"v > 110.0"` → true if value > 110
//! - `"v >= 0.0 && v <= 1.0"` → true if value in [0,1]
//! - `"v != v"` → true if value is NaN (NaN != NaN)
//!
//! ## Bitmask layout
//!
//! Output is a Vec of u64 words. Bit i of word w corresponds to row (w*64 + i).
//! A set bit means the row passes the predicate.
//! Length: ceil(n_rows / 64) words. Trailing bits in the last word are 0.
//!
//! ## Kernel design
//!
//! Uses `__ballot_sync` with 64 threads/block (two warps). Each block produces
//! exactly one u64 mask word. No atomics needed: shared memory collects the
//! two 32-bit ballot results, then one thread writes the combined u64.
//!
//! ## JIT caching
//!
//! Each unique predicate expression is compiled once (~40ms, NVRTC).
//! Subsequent calls with the same predicate use the cached kernel (~35ns).

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

/// 64 threads per block = two warps = one u64 mask word per block.
const FILTER_BLOCK_SIZE: u32 = 64;

/// JIT-compiled predicate evaluator cache.
///
/// Compiles one CUDA kernel per unique predicate expression.
/// Kernels produce packed u64 bitmasks for use as Frame row masks.
pub struct FilterJit {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    cache: HashMap<String, CudaFunction>,
}

impl FilterJit {
    /// Create a new FilterJit on GPU 0.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::on_device(0)
    }

    /// Create a new FilterJit on a specific GPU.
    pub fn on_device(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream, cache: HashMap::new() })
    }

    /// Evaluate `predicate` for each element of `values`, return packed u64 bitmask.
    ///
    /// `predicate`: CUDA expression in terms of `v` (double).
    /// `values`: host slice of f64 values, transferred to GPU internally.
    /// Returns `ceil(n / 64)` u64 words. Bit i of word w = row (w*64+i) passes.
    pub fn filter_mask(
        &mut self,
        predicate: &str,
        values: &[f64],
    ) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
        let n = values.len();
        let n_words = (n + 63) / 64;

        let values_dev = self.stream.clone_htod(values)?;
        let mask_dev = self.eval_predicate_gpu(predicate, &values_dev, n, n_words)?;
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&mask_dev)?)
    }

    /// Evaluate predicate on GPU-resident data. Returns GPU-resident bitmask.
    ///
    /// For pipeline use: keeps results on GPU, avoids round-trip to host.
    pub fn eval_predicate_gpu(
        &mut self,
        predicate: &str,
        values: &CudaSlice<f64>,
        n: usize,
        n_words: usize,
    ) -> Result<CudaSlice<u64>, Box<dyn std::error::Error>> {
        // Compile if not cached.
        if !self.cache.contains_key(predicate) {
            let src = build_filter_source(predicate);
            let opts = CompileOptions {
                arch: Some("sm_120"),
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(&src, opts)?;
            let module = self.ctx.load_module(ptx)?;
            let f = module.load_function("eval_filter")?;
            self.cache.insert(predicate.to_string(), f);
        }

        let mut mask_dev: CudaSlice<u64> = self.stream.alloc_zeros(n_words)?;
        let f = self.cache.get(predicate).unwrap();
        let cfg = filter_launch_cfg(n);
        let n_i32 = n as i32;

        unsafe {
            self.stream.launch_builder(f)
                .arg(values)
                .arg(&mut mask_dev)
                .arg(&n_i32)
                .launch(cfg)?;
        }
        Ok(mask_dev)
    }

    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }
    pub fn ctx(&self) -> &Arc<CudaContext> { &self.ctx }
}

/// Count the number of set bits (rows passing the predicate) in a bitmask.
///
/// CPU-side popcount over the returned Vec<u64>. Used for validation and reporting.
pub fn mask_popcount(mask: &[u64], n_rows: usize) -> usize {
    let full_words = n_rows / 64;
    let remainder = n_rows % 64;

    let full_count: usize = mask[..full_words].iter().map(|w| w.count_ones() as usize).sum();

    if remainder == 0 {
        full_count
    } else {
        // Last word: only the first `remainder` bits are valid.
        let last = mask.get(full_words).copied().unwrap_or(0);
        let valid_mask = (1u64 << remainder) - 1;
        full_count + (last & valid_mask).count_ones() as usize
    }
}

// ---------------------------------------------------------------------------
// CPU-side mask algebra — boolean calculus on packed bitmasks
// ---------------------------------------------------------------------------

/// AND two bitmasks. Result bit i = a[i] & b[i]. O(n_words). Slices must be same length.
pub fn mask_and(a: &[u64], b: &[u64]) -> Vec<u64> {
    assert_eq!(a.len(), b.len(), "mask_and: length mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x & y).collect()
}

/// OR two bitmasks. Result bit i = a[i] | b[i]. O(n_words).
pub fn mask_or(a: &[u64], b: &[u64]) -> Vec<u64> {
    assert_eq!(a.len(), b.len(), "mask_or: length mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x | y).collect()
}

/// XOR two bitmasks. Result bit i = a[i] ^ b[i]. O(n_words).
pub fn mask_xor(a: &[u64], b: &[u64]) -> Vec<u64> {
    assert_eq!(a.len(), b.len(), "mask_xor: length mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
}

/// NOT a bitmask. Flips all bits, then zeros trailing bits in the last word.
///
/// `n_rows` is required: NOT would flip zero-padding bits (rows that don't exist)
/// into 1s, creating phantom active rows. The last word is masked to only the
/// `remainder = n_rows % 64` valid bits. If n_rows is a multiple of 64, no masking needed.
pub fn mask_not(mask: &[u64], n_rows: usize) -> Vec<u64> {
    let mut result: Vec<u64> = mask.iter().map(|w| !w).collect();
    let remainder = n_rows % 64;
    if remainder != 0 {
        let valid_mask = (1u64 << remainder) - 1;
        result[n_rows / 64] &= valid_mask;
    }
    result
}

/// Build CUDA source for a filter kernel with the given predicate expression.
///
/// Uses `__ballot_sync` with 64 threads/block (2 warps). Each block:
/// 1. Evaluates the predicate for its 64 rows.
/// 2. Each warp collects 32 bits via `__ballot_sync`.
/// 3. Warp leaders write their 32-bit ballot to shared memory.
/// 4. Thread 0 combines the two 32-bit halves into one u64 mask word.
///
/// Out-of-bounds threads (gid >= n): predicate forced to false, v = 0.0.
/// The last block's trailing bits are therefore 0. ✓
///
/// No atomics needed: each u64 word is written by exactly one thread (thread 0
/// of each block). Shared memory synchronization within the block suffices.
fn build_filter_source(predicate: &str) -> String {
    format!(r#"
// JIT filter kernel: predicate → packed u64 bitmask.
// predicate = {pred}
// 64 threads/block → one u64 mask word/block.
// Bit layout: bit (blockIdx.x * 64 + threadIdx.x) set iff predicate(values[gid]) is true.
extern "C" __global__ void eval_filter(
    const double* __restrict__ values,
    unsigned long long* __restrict__ mask,
    int n
) {{
    __shared__ unsigned int warp_ballots[2];

    int gid = blockIdx.x * 64 + threadIdx.x;
    int warp_id = threadIdx.x >> 5;   // 0 or 1

    // Load value; out-of-bounds → 0.0 (predicate will be forced false anyway).
    double v = (gid < n) ? values[gid] : 0.0;
    // Predicate: 1 if row is in-bounds AND predicate holds.
    int passes = (gid < n) && ({pred});

    // Warp-level ballot: collects 32 bits per warp in one instruction.
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, passes);

    // Each warp's lane-0 stores its 32-bit ballot result to shared memory.
    if ((threadIdx.x & 31) == 0) {{
        warp_ballots[warp_id] = ballot;
    }}
    __syncthreads();

    // Thread 0 packs the two 32-bit halves into one u64 and writes the mask word.
    // warp_ballots[0] = bits 0..31 (warp 0, threads  0-31, rows gid  0-31)
    // warp_ballots[1] = bits 32..63 (warp 1, threads 32-63, rows gid 32-63)
    if (threadIdx.x == 0) {{
        unsigned long long word =
            (unsigned long long)warp_ballots[0] |
            ((unsigned long long)warp_ballots[1] << 32);
        mask[blockIdx.x] = word;
    }}
}}
"#, pred = predicate)
}

fn filter_launch_cfg(n: usize) -> LaunchConfig {
    let n_blocks = ((n as u32) + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (FILTER_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 8,  // 2 × u32 for warp_ballots
    }
}

// ---------------------------------------------------------------------------
// Named predicate constants
// ---------------------------------------------------------------------------

/// Predicate: v > threshold. Substitute the actual threshold in the string.
/// Example: `"v > 110.0"`
pub const PRED_GT: &str = "v > ";  // append threshold

/// Predicate: v is not NaN (NaN != NaN).
pub const PRED_NOT_NAN: &str = "v == v";

/// Predicate: v is finite (not NaN, not inf).
pub const PRED_FINITE: &str = "v == v && v * 0.0 == 0.0";

/// Predicate: v is positive.
pub const PRED_POSITIVE: &str = "v > 0.0";

// ---------------------------------------------------------------------------
// GPU-resident mask algebra — MaskOp
// ---------------------------------------------------------------------------

const MASK_BLOCK_SIZE: u32 = 256;

/// GPU bitwise operations on packed u64 bitmasks.
///
/// For pipeline use: when masks are already GPU-resident (produced by FilterJit),
/// combining them via MaskOp avoids a GPU→CPU→GPU round-trip. All `_gpu` methods
/// operate on GPU-resident `CudaSlice<u64>` directly. Host-side wrappers handle
/// the transfer for callers with CPU data.
///
/// CPU-resident mask algebra is also available as standalone free functions:
/// `mask_and`, `mask_or`, `mask_not`, `mask_xor`.
pub struct MaskOp {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    and_fn: Option<CudaFunction>,
    or_fn: Option<CudaFunction>,
    not_fn: Option<CudaFunction>,
    xor_fn: Option<CudaFunction>,
    popcount_fn: Option<CudaFunction>,
}

impl MaskOp {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::on_device(0)
    }

    pub fn on_device(ordinal: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream, and_fn: None, or_fn: None, not_fn: None, xor_fn: None, popcount_fn: None })
    }

    /// AND two host masks on GPU, return host result.
    pub fn mask_and(&mut self, a: &[u64], b: &[u64]) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
        let n_words = a.len();
        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let result = self.mask_and_gpu(&a_dev, &b_dev, n_words)?;
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&result)?)
    }

    /// OR two host masks on GPU, return host result.
    pub fn mask_or(&mut self, a: &[u64], b: &[u64]) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
        let n_words = a.len();
        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let result = self.mask_or_gpu(&a_dev, &b_dev, n_words)?;
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&result)?)
    }

    /// XOR two host masks on GPU, return host result.
    pub fn mask_xor(&mut self, a: &[u64], b: &[u64]) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
        let n_words = a.len();
        let a_dev = self.stream.clone_htod(a)?;
        let b_dev = self.stream.clone_htod(b)?;
        let result = self.mask_xor_gpu(&a_dev, &b_dev, n_words)?;
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&result)?)
    }

    /// NOT a host mask on GPU, return host result.
    pub fn mask_not(&mut self, mask: &[u64], n_rows: usize) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
        let n_words = mask.len();
        let mask_dev = self.stream.clone_htod(mask)?;
        let result = self.mask_not_gpu(&mask_dev, n_words, n_rows)?;
        self.stream.synchronize()?;
        Ok(self.stream.clone_dtoh(&result)?)
    }

    /// AND two GPU-resident masks. Both must be `n_words` long.
    pub fn mask_and_gpu(
        &mut self,
        a: &CudaSlice<u64>,
        b: &CudaSlice<u64>,
        n_words: usize,
    ) -> Result<CudaSlice<u64>, Box<dyn std::error::Error>> {
        self.ensure_kernels()?;
        let f = self.and_fn.as_ref().unwrap();
        let mut output: CudaSlice<u64> = self.stream.alloc_zeros(n_words)?;
        let n_i32 = n_words as i32;
        unsafe {
            self.stream.launch_builder(f)
                .arg(a).arg(b).arg(&mut output).arg(&n_i32)
                .launch(mask_launch_cfg(n_words))?;
        }
        Ok(output)
    }

    /// OR two GPU-resident masks.
    pub fn mask_or_gpu(
        &mut self,
        a: &CudaSlice<u64>,
        b: &CudaSlice<u64>,
        n_words: usize,
    ) -> Result<CudaSlice<u64>, Box<dyn std::error::Error>> {
        self.ensure_kernels()?;
        let f = self.or_fn.as_ref().unwrap();
        let mut output: CudaSlice<u64> = self.stream.alloc_zeros(n_words)?;
        let n_i32 = n_words as i32;
        unsafe {
            self.stream.launch_builder(f)
                .arg(a).arg(b).arg(&mut output).arg(&n_i32)
                .launch(mask_launch_cfg(n_words))?;
        }
        Ok(output)
    }

    /// XOR two GPU-resident masks.
    pub fn mask_xor_gpu(
        &mut self,
        a: &CudaSlice<u64>,
        b: &CudaSlice<u64>,
        n_words: usize,
    ) -> Result<CudaSlice<u64>, Box<dyn std::error::Error>> {
        self.ensure_kernels()?;
        let f = self.xor_fn.as_ref().unwrap();
        let mut output: CudaSlice<u64> = self.stream.alloc_zeros(n_words)?;
        let n_i32 = n_words as i32;
        unsafe {
            self.stream.launch_builder(f)
                .arg(a).arg(b).arg(&mut output).arg(&n_i32)
                .launch(mask_launch_cfg(n_words))?;
        }
        Ok(output)
    }

    /// NOT a GPU-resident mask. Trailing bits of the last word are zeroed.
    /// `n_rows` required to correctly zero trailing bits.
    pub fn mask_not_gpu(
        &mut self,
        a: &CudaSlice<u64>,
        n_words: usize,
        n_rows: usize,
    ) -> Result<CudaSlice<u64>, Box<dyn std::error::Error>> {
        self.ensure_kernels()?;
        let f = self.not_fn.as_ref().unwrap();
        let mut output: CudaSlice<u64> = self.stream.alloc_zeros(n_words)?;
        let n_words_i32 = n_words as i32;
        let n_rows_i32 = n_rows as i32;
        unsafe {
            self.stream.launch_builder(f)
                .arg(a).arg(&mut output).arg(&n_words_i32).arg(&n_rows_i32)
                .launch(mask_launch_cfg(n_words))?;
        }
        Ok(output)
    }

    /// GPU popcount: count set bits over the valid rows. Trailing bits excluded.
    /// Synchronous: blocks until the count is ready.
    pub fn mask_popcount_gpu(
        &mut self,
        mask: &CudaSlice<u64>,
        n_words: usize,
        n_rows: usize,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        self.ensure_kernels()?;
        let f = self.popcount_fn.as_ref().unwrap();
        let mut count: CudaSlice<u64> = self.stream.alloc_zeros(1)?;
        let n_words_i32 = n_words as i32;
        let n_rows_i32 = n_rows as i32;
        unsafe {
            self.stream.launch_builder(f)
                .arg(mask).arg(&mut count).arg(&n_words_i32).arg(&n_rows_i32)
                .launch(mask_popcount_launch_cfg(n_words))?;
        }
        self.stream.synchronize()?;
        let result = self.stream.clone_dtoh(&count)?;
        Ok(result[0])
    }

    pub fn stream(&self) -> &Arc<CudaStream> { &self.stream }
    pub fn ctx(&self) -> &Arc<CudaContext> { &self.ctx }

    /// Compile all five kernels in one NVRTC pass, if not already compiled.
    fn ensure_kernels(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.and_fn.is_none() {
            let opts = CompileOptions { arch: Some("sm_120"), ..Default::default() };
            let ptx = compile_ptx_with_opts(MASK_OP_SOURCE, opts)?;
            let module = self.ctx.load_module(ptx)?;
            self.and_fn      = Some(module.load_function("mask_and_u64")?);
            self.or_fn       = Some(module.load_function("mask_or_u64")?);
            self.not_fn      = Some(module.load_function("mask_not_u64")?);
            self.xor_fn      = Some(module.load_function("mask_xor_u64")?);
            self.popcount_fn = Some(module.load_function("mask_popcount_u64")?);
        }
        Ok(())
    }
}

const MASK_OP_SOURCE: &str = r#"
// mask_and_u64: output[i] = a[i] & b[i]
extern "C" __global__ void mask_and_u64(
    const unsigned long long* __restrict__ a,
    const unsigned long long* __restrict__ b,
    unsigned long long* __restrict__ output,
    int n_words
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_words) output[i] = a[i] & b[i];
}

// mask_or_u64: output[i] = a[i] | b[i]
extern "C" __global__ void mask_or_u64(
    const unsigned long long* __restrict__ a,
    const unsigned long long* __restrict__ b,
    unsigned long long* __restrict__ output,
    int n_words
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_words) output[i] = a[i] | b[i];
}

// mask_xor_u64: output[i] = a[i] ^ b[i]
extern "C" __global__ void mask_xor_u64(
    const unsigned long long* __restrict__ a,
    const unsigned long long* __restrict__ b,
    unsigned long long* __restrict__ output,
    int n_words
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_words) output[i] = a[i] ^ b[i];
}

// mask_not_u64: output[i] = ~a[i], trailing bits of last word zeroed.
// n_rows: total row count used to compute the valid bit count in the last word.
extern "C" __global__ void mask_not_u64(
    const unsigned long long* __restrict__ a,
    unsigned long long* __restrict__ output,
    int n_words,
    int n_rows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_words) {
        unsigned long long v = ~a[i];
        int remainder = n_rows & 63;
        if (remainder != 0 && i == n_words - 1) {
            v &= (1ULL << remainder) - 1;
        }
        output[i] = v;
    }
}

// mask_popcount_u64: count set bits over the valid rows.
// Parallel reduction with __popcll and shared memory.
// Trailing bits of the last word are excluded from the count.
extern "C" __global__ void mask_popcount_u64(
    const unsigned long long* __restrict__ mask,
    unsigned long long* __restrict__ count,
    int n_words,
    int n_rows
) {
    extern __shared__ unsigned long long sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long w = 0ULL;
    if (i < n_words) {
        w = mask[i];
        int remainder = n_rows & 63;
        if (remainder != 0 && i == n_words - 1) {
            w &= (1ULL << remainder) - 1;
        }
    }
    sdata[tid] = (unsigned long long)__popcll(w);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(count, sdata[0]);
}
"#;

fn mask_launch_cfg(n_words: usize) -> LaunchConfig {
    let n_blocks = ((n_words as u32) + MASK_BLOCK_SIZE - 1) / MASK_BLOCK_SIZE;
    LaunchConfig { grid_dim: (n_blocks, 1, 1), block_dim: (MASK_BLOCK_SIZE, 1, 1), shared_mem_bytes: 0 }
}

fn mask_popcount_launch_cfg(n_words: usize) -> LaunchConfig {
    let n_blocks = ((n_words as u32) + MASK_BLOCK_SIZE - 1) / MASK_BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (MASK_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: MASK_BLOCK_SIZE * 8,  // one u64 per thread
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_gt_threshold() {
        let mut jit = FilterJit::new().unwrap();
        let values = vec![1.0, 5.0, 3.0, 7.0, 2.0, 8.0];
        let mask = jit.filter_mask("v > 4.0", &values).unwrap();

        // Row 0: 1.0 → fails. Row 1: 5.0 → passes. Row 2: 3.0 → fails.
        // Row 3: 7.0 → passes. Row 4: 2.0 → fails. Row 5: 8.0 → passes.
        // Expected bits: 0b101010 = 0x2A
        assert_eq!(mask[0] & 0x3F, 0b101010, "bits 0-5 should be 0b101010");
    }

    #[test]
    fn filter_all_pass() {
        let mut jit = FilterJit::new().unwrap();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mask = jit.filter_mask("v > 0.0", &values).unwrap();
        assert_eq!(mask[0] & 0xF, 0xF, "all 4 rows should pass");
    }

    #[test]
    fn filter_none_pass() {
        let mut jit = FilterJit::new().unwrap();
        let values = vec![1.0, 2.0, 3.0];
        let mask = jit.filter_mask("v > 100.0", &values).unwrap();
        assert_eq!(mask[0] & 0x7, 0, "no rows should pass");
    }

    #[test]
    fn filter_exactly_64_rows() {
        let mut jit = FilterJit::new().unwrap();
        // 64 values: even indices are 0.0, odd indices are 1.0
        let values: Vec<f64> = (0..64).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let mask = jit.filter_mask("v > 0.5", &values).unwrap();
        assert_eq!(mask.len(), 1, "64 rows → exactly 1 u64 word");
        // Odd rows pass: bits 1, 3, 5, ... 63 → alternating pattern
        // Expected: 0xAAAAAAAAAAAAAAAA
        assert_eq!(mask[0], 0xAAAAAAAAAAAAAAAAu64);
    }

    #[test]
    fn filter_65_rows_two_words() {
        let mut jit = FilterJit::new().unwrap();
        let values: Vec<f64> = vec![1.0; 65];
        let mask = jit.filter_mask("v > 0.5", &values).unwrap();
        assert_eq!(mask.len(), 2, "65 rows → 2 u64 words");
        assert_eq!(mask[0], u64::MAX, "word 0: all 64 rows pass");
        assert_eq!(mask[1] & 1, 1, "word 1: bit 0 (row 64) passes");
        assert_eq!(mask[1] >> 1, 0, "word 1: bits 1-63 are 0 (out-of-bounds)");
    }

    #[test]
    fn filter_mask_popcount() {
        // mask[0] = 0b0110: bits 1,2 set (rows 1, 2)
        // mask[1] = 0b1001: bits 0,3 set (rows 64, 67)
        let mask = vec![0b0110u64, 0b1001u64];

        // 128 rows: both words count → 4 active rows total
        assert_eq!(mask_popcount(&mask, 128), 4);
        // 66 rows: word 0 full (2 bits) + bits 0-1 of word 1 (bit 0 set = 1 bit) = 3
        assert_eq!(mask_popcount(&mask, 66), 3);
        // 6 rows: first 6 bits of word 0 only → bits 1,2 set = 2 rows
        assert_eq!(mask_popcount(&mask, 6), 2);
    }

    #[test]
    fn filter_cached() {
        let mut jit = FilterJit::new().unwrap();
        let values = vec![1.0f64, 2.0, 3.0];
        jit.filter_mask("v > 1.5", &values).unwrap();
        jit.filter_mask("v > 1.5", &values).unwrap();
        assert_eq!(jit.cache.len(), 1, "should have one cached kernel");
    }

    // -----------------------------------------------------------------------
    // CPU mask algebra
    // -----------------------------------------------------------------------

    #[test]
    fn cpu_mask_and() {
        // a: rows 0,2 set (0b0101), b: rows 1,2 set (0b0110) → AND: row 2 only (0b0100)
        assert_eq!(mask_and(&[0b0101u64], &[0b0110u64]), vec![0b0100u64]);
    }

    #[test]
    fn cpu_mask_or() {
        // OR: rows 0,1,2 (0b0111)
        assert_eq!(mask_or(&[0b0101u64], &[0b0110u64]), vec![0b0111u64]);
    }

    #[test]
    fn cpu_mask_xor() {
        // XOR: rows 0,1 (0b0011)
        assert_eq!(mask_xor(&[0b0101u64], &[0b0110u64]), vec![0b0011u64]);
    }

    #[test]
    fn cpu_mask_not_full_word() {
        // 64 rows, all set → NOT = all clear
        assert_eq!(mask_not(&[u64::MAX], 64), vec![0u64]);
    }

    #[test]
    fn cpu_mask_not_partial_last_word() {
        // 4 rows: bits 0-3 valid. mask = 0b0101 (rows 0,2 set)
        // NOT: 0b1010 (rows 1,3 set); bits 4-63 must remain 0.
        let result = mask_not(&[0b0101u64], 4);
        assert_eq!(result, vec![0b1010u64]);
        assert_eq!(result[0] >> 4, 0, "trailing bits must be zero");
    }

    #[test]
    fn cpu_mask_not_trailing_bits_zeroed() {
        // 4 rows, none set → NOT should yield bits 0-3 all set, bits 4-63 = 0.
        let result = mask_not(&[0u64], 4);
        assert_eq!(result, vec![0b1111u64]);
        assert_eq!(result[0] >> 4, 0);
    }

    // -----------------------------------------------------------------------
    // GPU mask algebra (MaskOp)
    // -----------------------------------------------------------------------

    #[test]
    fn gpu_mask_and() {
        let mut op = MaskOp::new().unwrap();
        let result = op.mask_and(&[0b0101u64], &[0b0110u64]).unwrap();
        assert_eq!(result, vec![0b0100u64]);
    }

    #[test]
    fn gpu_mask_or() {
        let mut op = MaskOp::new().unwrap();
        let result = op.mask_or(&[0b0101u64], &[0b0110u64]).unwrap();
        assert_eq!(result, vec![0b0111u64]);
    }

    #[test]
    fn gpu_mask_xor() {
        let mut op = MaskOp::new().unwrap();
        let result = op.mask_xor(&[0b0101u64], &[0b0110u64]).unwrap();
        assert_eq!(result, vec![0b0011u64]);
    }

    #[test]
    fn gpu_mask_not_partial_word() {
        let mut op = MaskOp::new().unwrap();
        let result = op.mask_not(&[0b0101u64], 4).unwrap();
        assert_eq!(result, vec![0b1010u64]);
        assert_eq!(result[0] >> 4, 0, "trailing bits must be zero");
    }

    #[test]
    fn gpu_mask_popcount_full_words() {
        let mut op = MaskOp::new().unwrap();
        // 128 rows: word 0 has bits 0,2 set (2 rows), word 1 has bits 0,3 set (2 rows) = 4
        let mask = vec![0b0101u64, 0b1001u64];
        let mask_dev = op.stream.clone_htod(&mask).unwrap();
        let count = op.mask_popcount_gpu(&mask_dev, 2, 128).unwrap();
        assert_eq!(count, 4);
    }

    #[test]
    fn gpu_mask_popcount_partial_last_word() {
        let mut op = MaskOp::new().unwrap();
        // 66 rows: word 0 = all 64 set, word 1 = bits 0,1 set (rows 64,65 valid)
        let mask = vec![u64::MAX, 0b11u64];
        let mask_dev = op.stream.clone_htod(&mask).unwrap();
        let count = op.mask_popcount_gpu(&mask_dev, 2, 66).unwrap();
        assert_eq!(count, 66);
    }

    #[test]
    fn gpu_mask_and_then_popcount() {
        // Compose: filter a AND b → count. 128 rows, 4 rows in intersection.
        let mut op = MaskOp::new().unwrap();
        let a = vec![0b0101u64, 0b0101u64];  // rows 0,2,64,66
        let b = vec![0b0110u64, 0b0110u64];  // rows 1,2,65,66
        let a_dev = op.stream.clone_htod(&a).unwrap();
        let b_dev = op.stream.clone_htod(&b).unwrap();
        let and_dev = op.mask_and_gpu(&a_dev, &b_dev, 2).unwrap();
        let count = op.mask_popcount_gpu(&and_dev, 2, 128).unwrap();
        assert_eq!(count, 2);  // rows 2 and 66
    }
}
