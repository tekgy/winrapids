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
}
