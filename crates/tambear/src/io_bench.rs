//! I/O Path Benchmarks: File-Mapped GPU vs cudaMalloc
//!
//! Path 1: Pinned file-mapped memory
//!   CreateFileMapping → MapViewOfFile → cudaHostRegister(DEVICEMAP) → cudaHostGetDevicePointer
//!   GPU reads the .tb file directly through PCIe BAR. No cudaMemcpy.
//!
//! Path 3: CUDA VMM (Virtual Memory Management)
//!   cuMemAddressReserve → cuMemCreate → cuMemMap → cuMemSetAccess
//!   Reserve GPU VA for whole file, allocate physical only for accessed columns.
//!
//! Baseline: cudaMalloc + clone_htod (standard path)
//!
//! Run with: cargo run --bin io-bench --release

use std::path::Path;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::driver::sys as cuda_sys;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// CUDA kernel: reduction sum (to measure kernel perf on different memory)
// ---------------------------------------------------------------------------

const REDUCE_CUDA: &str = r#"
// Simple block-level reduction. Enough to measure memory access perf.
extern "C" __global__ void reduce_sum(
    const double* __restrict__ data,
    double* __restrict__ out,
    int n
) {
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double val = 0.0;
    // Grid-stride loop
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        val += data[i];
    }
    sdata[tid] = val;
    __syncthreads();

    // Block reduce
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// scatter_stats: the real workload (same as tambear)
extern "C" __global__ void scatter_stats(
    const int* __restrict__ keys,
    const double* __restrict__ values,
    double* __restrict__ sums,
    double* __restrict__ sum_sqs,
    double* __restrict__ counts,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int g = keys[gid];
        double v = values[gid];
        atomicAdd(&sums[g], v);
        atomicAdd(&sum_sqs[g], v * v);
        atomicAdd(&counts[g], 1.0);
    }
}
"#;

struct BenchEngine {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    f_reduce: CudaFunction,
    f_scatter: CudaFunction,
}

impl BenchEngine {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let opts = CompileOptions {
            arch: Some("sm_120"),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(REDUCE_CUDA, opts)?;
        let module = ctx.load_module(ptx)?;
        Ok(Self {
            f_reduce: module.load_function("reduce_sum")?,
            f_scatter: module.load_function("scatter_stats")?,
            ctx,
            stream,
        })
    }

    fn reduce_sum_dev(&self, data_ptr: u64, n: usize) -> Result<f64, Box<dyn std::error::Error>> {
        let mut out: CudaSlice<f64> = self.stream.alloc_zeros(1)?;
        let n_blocks = ((n as u32) + 255) / 256;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks.min(1024), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.f_reduce)
                .arg(&data_ptr)
                .arg(&mut out)
                .arg(&(n as i32))
                .launch(cfg)?;
        }
        self.stream.synchronize()?;
        let result = self.stream.clone_dtoh(&out)?;
        Ok(result[0])
    }

    fn scatter_stats_dev(
        &self,
        keys_ptr: u64,
        vals_ptr: u64,
        n: usize,
        n_groups: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
        let mut sums: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut sq: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let mut counts: CudaSlice<f64> = self.stream.alloc_zeros(n_groups)?;
        let n_blocks = ((n as u32) + 255) / 256;
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.f_scatter)
                .arg(&keys_ptr)
                .arg(&vals_ptr)
                .arg(&mut sums)
                .arg(&mut sq)
                .arg(&mut counts)
                .arg(&(n as i32))
                .launch(cfg)?;
        }
        self.stream.synchronize()?;
        Ok((
            self.stream.clone_dtoh(&sums)?,
            self.stream.clone_dtoh(&sq)?,
            self.stream.clone_dtoh(&counts)?,
        ))
    }
}

// ---------------------------------------------------------------------------
// Windows file mapping
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod win {
    use std::ffi::c_void;

    // Windows constants
    pub const GENERIC_READ: u32 = 0x80000000;
    pub const GENERIC_WRITE: u32 = 0x40000000;
    pub const OPEN_EXISTING: u32 = 3;
    pub const FILE_FLAG_RANDOM_ACCESS: u32 = 0x10000000;
    pub const PAGE_READWRITE: u32 = 0x04;
    pub const FILE_MAP_ALL_ACCESS: u32 = 0xF001F;
    pub const INVALID_HANDLE_VALUE: isize = -1;

    extern "system" {
        pub fn CreateFileW(
            name: *const u16,
            access: u32,
            share: u32,
            security: *const c_void,
            disposition: u32,
            flags: u32,
            template: *const c_void,
        ) -> isize;
        pub fn CreateFileMappingW(
            file: isize,
            security: *const c_void,
            protect: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> isize;
        pub fn MapViewOfFile(
            mapping: isize,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        pub fn VirtualLock(addr: *const c_void, size: usize) -> i32;
        pub fn VirtualUnlock(addr: *const c_void, size: usize) -> i32;
        pub fn UnmapViewOfFile(addr: *const c_void) -> i32;
        pub fn CloseHandle(handle: isize) -> i32;
    }

    /// Memory-map a file for read/write. Returns (base_ptr, file_size, h_file, h_map).
    pub unsafe fn mmap_file(path: &std::path::Path) -> Result<(*mut c_void, usize, isize, isize), String> {
        let file_size = std::fs::metadata(path)
            .map_err(|e| format!("metadata: {}", e))?
            .len() as usize;

        // Encode path as wide string
        let wide: Vec<u16> = path.to_string_lossy()
            .encode_utf16()
            .chain(std::iter::once(0u16))
            .collect();

        let h_file = CreateFileW(
            wide.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            1, // FILE_SHARE_READ
            std::ptr::null(),
            OPEN_EXISTING,
            FILE_FLAG_RANDOM_ACCESS,
            std::ptr::null(),
        );
        if h_file == INVALID_HANDLE_VALUE {
            return Err("CreateFileW failed".into());
        }

        let h_map = CreateFileMappingW(h_file, std::ptr::null(), PAGE_READWRITE, 0, 0, std::ptr::null());
        if h_map == 0 {
            CloseHandle(h_file);
            return Err("CreateFileMappingW failed".into());
        }

        let base = MapViewOfFile(h_map, FILE_MAP_ALL_ACCESS, 0, 0, 0);
        if base.is_null() {
            CloseHandle(h_map);
            CloseHandle(h_file);
            return Err("MapViewOfFile failed".into());
        }

        // Pin pages
        VirtualLock(base, file_size);

        Ok((base, file_size, h_file, h_map))
    }

    pub unsafe fn munmap_file(base: *mut c_void, size: usize, h_file: isize, h_map: isize) {
        VirtualUnlock(base, size);
        UnmapViewOfFile(base);
        CloseHandle(h_map);
        CloseHandle(h_file);
    }
}

// ---------------------------------------------------------------------------
// Path 1: Pinned file-mapped memory
// ---------------------------------------------------------------------------

#[cfg(windows)]
fn bench_path1_pinned(
    engine: &BenchEngine,
    tb_path: &Path,
    prices: &[f64],
    keys: &[i32],
    n_groups: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::DevicePtr;

    println!("\n=== Path 1: Pinned File-Mapped Memory ===");
    println!("  CreateFileMapping -> MapViewOfFile -> cudaHostRegister -> cudaHostGetDevicePointer");

    let n = prices.len();
    let file_size;
    let base_ptr;
    let h_file;
    let h_map;

    // Memory-map the .tb file
    unsafe {
        let (b, sz, hf, hm) = win::mmap_file(tb_path).map_err(|e| format!("mmap: {}", e))?;
        base_ptr = b;
        file_size = sz;
        h_file = hf;
        h_map = hm;
    }
    println!("  File mapped: {} bytes at {:p}", file_size, base_ptr);

    // Read the .tb header to find data offsets
    let header = unsafe {
        let buf = std::slice::from_raw_parts(base_ptr as *const u8, 4096);
        &*(buf.as_ptr() as *const tambear::TbFileHeader)
    };
    assert!(header.is_valid(), "invalid .tb file");

    let price_col_idx = 1u32; // second column = price
    let key_col_idx = 0u32;   // first column = minute key
    let price_data_offset = header.tile_data_offset(price_col_idx, 0) as usize;
    let key_data_offset = header.tile_data_offset(key_col_idx, 0) as usize;

    // Register the mapped memory with CUDA
    let t0 = Instant::now();
    let register_result = unsafe {
        cuda_sys::cuMemHostRegister_v2(
            base_ptr,
            file_size,
            cuda_sys::CU_MEMHOSTREGISTER_DEVICEMAP,
        )
    };
    let register_us = t0.elapsed().as_secs_f64() * 1e6;

    if register_result != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cudaHostRegister FAILED: {:?}", register_result);
        println!("  (This can happen if file is too large for pinned memory quota)");
        unsafe { win::munmap_file(base_ptr, file_size, h_file, h_map); }
        return Ok(());
    }
    println!("  cudaHostRegister: {:.0}us", register_us);

    // Get device pointer
    let mut dev_ptr: u64 = 0;
    let t0 = Instant::now();
    let get_result = unsafe {
        cuda_sys::cuMemHostGetDevicePointer_v2(
            &mut dev_ptr,
            base_ptr,
            0,
        )
    };
    let get_us = t0.elapsed().as_secs_f64() * 1e6;

    if get_result != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cudaHostGetDevicePointer FAILED: {:?}", get_result);
        unsafe {
            cuda_sys::cuMemHostUnregister(base_ptr);
            win::munmap_file(base_ptr, file_size, h_file, h_map);
        }
        return Ok(());
    }
    println!("  cudaHostGetDevicePointer: {:.0}us (dev_ptr=0x{:x})", get_us, dev_ptr);

    // Compute device pointers to column data within the file
    let price_dev_ptr = dev_ptr + price_data_offset as u64;
    let key_dev_ptr = dev_ptr + key_data_offset as u64;

    // -----------------------------------------------------------------------
    // Benchmark 1: reduce_sum on file-mapped vs cudaMalloc'd
    // -----------------------------------------------------------------------
    println!("\n  --- reduce_sum ({} elements) ---", n);

    // File-mapped: warm up
    let sum_mapped = engine.reduce_sum_dev(price_dev_ptr, n)?;
    let sum_expected: f64 = prices.iter().sum();
    let err = (sum_mapped - sum_expected).abs() / sum_expected.abs();
    println!("  Correctness: mapped_sum={:.2}, expected={:.2}, rel_err={:.2e}", sum_mapped, sum_expected, err);

    // File-mapped: benchmark
    let n_iters = 50;
    engine.stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.reduce_sum_dev(price_dev_ptr, n)?;
    }
    engine.stream.synchronize()?;
    let mapped_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    // cudaMalloc baseline
    let prices_dev = engine.stream.clone_htod(prices)?;
    let (prices_dev_ptr, _guard) = prices_dev.device_ptr(&engine.stream);
    engine.stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.reduce_sum_dev(prices_dev_ptr, n)?;
    }
    engine.stream.synchronize()?;
    let malloc_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    println!("  cudaMalloc:   {:.0}us/op", malloc_us);
    println!("  File-mapped:  {:.0}us/op", mapped_us);
    println!("  Ratio:        {:.1}x", mapped_us / malloc_us);

    let data_bytes = n * 8;
    let bw_malloc = data_bytes as f64 / (malloc_us * 1e-6) / 1e9;
    let bw_mapped = data_bytes as f64 / (mapped_us * 1e-6) / 1e9;
    println!("  BW cudaMalloc: {:.1} GB/s", bw_malloc);
    println!("  BW file-mapped: {:.1} GB/s", bw_mapped);

    // -----------------------------------------------------------------------
    // Benchmark 2: scatter_stats (the real workload)
    // Keys must be i32 — use cudaMalloc'd keys, only values from file-mapped.
    // This isolates the values-column access pattern.
    // -----------------------------------------------------------------------
    println!("\n  --- scatter_stats ({} elements, {} groups) ---", n, n_groups);
    println!("  (keys via cudaMalloc, values via file-mapped — isolating value access)");

    let keys_dev = engine.stream.clone_htod(keys)?;
    let (keys_dev_ptr, _guard2) = keys_dev.device_ptr(&engine.stream);

    // File-mapped scatter (keys from cudaMalloc, values from mmap)
    let (sums_m, _, counts_m) = engine.scatter_stats_dev(keys_dev_ptr, price_dev_ptr, n, n_groups)?;
    let active_m = counts_m.iter().filter(|&&c| c > 0.0).count();
    println!("  File-mapped: {} active groups", active_m);

    // Benchmark file-mapped values
    engine.stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.scatter_stats_dev(keys_dev_ptr, price_dev_ptr, n, n_groups)?;
    }
    engine.stream.synchronize()?;
    let scatter_mapped_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    // cudaMalloc scatter baseline (both keys and values from cudaMalloc)
    engine.stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.scatter_stats_dev(keys_dev_ptr, prices_dev_ptr, n, n_groups)?;
    }
    engine.stream.synchronize()?;
    let scatter_malloc_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    println!("  cudaMalloc:   {:.0}us/op", scatter_malloc_us);
    println!("  File-mapped:  {:.0}us/op", scatter_mapped_us);
    println!("  Ratio:        {:.1}x", scatter_mapped_us / scatter_malloc_us);

    // -----------------------------------------------------------------------
    // First-access latency
    // -----------------------------------------------------------------------
    println!("\n  --- First-access latency ---");

    // Unregister and re-register to simulate cold start
    unsafe { cuda_sys::cuMemHostUnregister(base_ptr); }

    let t0 = Instant::now();
    unsafe {
        cuda_sys::cuMemHostRegister_v2(base_ptr, file_size, cuda_sys::CU_MEMHOSTREGISTER_DEVICEMAP);
        cuda_sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, base_ptr, 0);
    }
    let price_dev_ptr = dev_ptr + price_data_offset as u64;
    let _ = engine.reduce_sum_dev(price_dev_ptr, n)?;
    let first_access_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Register + first kernel: {:.2}ms", first_access_ms);

    // Cleanup
    unsafe {
        cuda_sys::cuMemHostUnregister(base_ptr);
        win::munmap_file(base_ptr, file_size, h_file, h_map);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Path 3: CUDA VMM (Virtual Memory Management)
// ---------------------------------------------------------------------------

fn bench_path3_vmm(
    engine: &BenchEngine,
    prices: &[f64],
    keys: &[i32],
    n_groups: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use cudarc::driver::DevicePtr;

    println!("\n=== Path 3: CUDA VMM (Virtual Memory Management) ===");
    println!("  cuMemAddressReserve -> cuMemCreate -> cuMemMap -> cuMemSetAccess");

    let n = prices.len();

    // Get allocation granularity
    let mut granularity: usize = 0;
    let mut prop: cuda_sys::CUmemAllocationProp_st = unsafe { std::mem::zeroed() };
    prop.type_ = cuda_sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = cuda_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.__bindgen_anon_1.id = 0; // GPU 0

    let res = unsafe {
        cuda_sys::cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            cuda_sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
    };
    if res != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cuMemGetAllocationGranularity FAILED: {:?}", res);
        return Ok(());
    }
    println!("  Allocation granularity: {} bytes ({:.0} KB)", granularity, granularity as f64 / 1024.0);

    // Round up sizes to granularity
    let price_bytes = n * 8;
    let key_bytes = n * 4;
    // We'll put keys (i32) then prices (f64) contiguously. But VMM needs to handle i32.
    // Simpler: convert keys to f64 for uniform 8-byte access.
    let keys_f64: Vec<f64> = keys.iter().map(|&k| k as f64).collect();

    let col_size = ((price_bytes + granularity - 1) / granularity) * granularity;
    let total_va = col_size * 2; // two columns

    println!("  VA reservation: {} bytes ({:.1} MB), per-column: {} bytes",
        total_va, total_va as f64 / 1e6, col_size);

    // Step 1: Reserve virtual address space
    let mut va_ptr: u64 = 0;
    let t0 = Instant::now();
    let res = unsafe {
        cuda_sys::cuMemAddressReserve(&mut va_ptr, total_va, granularity, 0, 0)
    };
    let reserve_us = t0.elapsed().as_secs_f64() * 1e6;
    if res != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cuMemAddressReserve FAILED: {:?}", res);
        return Ok(());
    }
    println!("  cuMemAddressReserve: {:.0}us (VA=0x{:x})", reserve_us, va_ptr);

    // Step 2: Create physical allocations (one per column)
    let mut handle_keys: u64 = 0;
    let mut handle_prices: u64 = 0;

    let t0 = Instant::now();
    let res = unsafe { cuda_sys::cuMemCreate(&mut handle_keys, col_size, &prop, 0) };
    if res != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cuMemCreate (keys) FAILED: {:?}", res);
        unsafe { cuda_sys::cuMemAddressFree(va_ptr, total_va); }
        return Ok(());
    }
    let res = unsafe { cuda_sys::cuMemCreate(&mut handle_prices, col_size, &prop, 0) };
    if res != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cuMemCreate (prices) FAILED: {:?}", res);
        unsafe {
            cuda_sys::cuMemRelease(handle_keys);
            cuda_sys::cuMemAddressFree(va_ptr, total_va);
        }
        return Ok(());
    }
    let create_us = t0.elapsed().as_secs_f64() * 1e6;
    println!("  cuMemCreate (2 columns): {:.0}us", create_us);

    // Step 3: Map physical to VA
    let t0 = Instant::now();
    let res = unsafe { cuda_sys::cuMemMap(va_ptr, col_size, 0, handle_keys, 0) };
    if res != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cuMemMap (keys) FAILED: {:?}", res);
        unsafe {
            cuda_sys::cuMemRelease(handle_keys);
            cuda_sys::cuMemRelease(handle_prices);
            cuda_sys::cuMemAddressFree(va_ptr, total_va);
        }
        return Ok(());
    }
    let res = unsafe { cuda_sys::cuMemMap(va_ptr + col_size as u64, col_size, 0, handle_prices, 0) };
    if res != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cuMemMap (prices) FAILED: {:?}", res);
        unsafe {
            cuda_sys::cuMemUnmap(va_ptr, col_size);
            cuda_sys::cuMemRelease(handle_keys);
            cuda_sys::cuMemRelease(handle_prices);
            cuda_sys::cuMemAddressFree(va_ptr, total_va);
        }
        return Ok(());
    }
    let map_us = t0.elapsed().as_secs_f64() * 1e6;
    println!("  cuMemMap (2 columns): {:.0}us", map_us);

    // Step 4: Set access permissions
    let access_desc = cuda_sys::CUmemAccessDesc_st {
        location: cuda_sys::CUmemLocation_st {
            type_: cuda_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            __bindgen_anon_1: cuda_sys::CUmemLocation_st__bindgen_ty_1 { id: 0 },
        },
        flags: cuda_sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };

    let t0 = Instant::now();
    let res = unsafe { cuda_sys::cuMemSetAccess(va_ptr, total_va, &access_desc, 1) };
    if res != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cuMemSetAccess FAILED: {:?}", res);
        // Cleanup
        unsafe {
            cuda_sys::cuMemUnmap(va_ptr, col_size);
            cuda_sys::cuMemUnmap(va_ptr + col_size as u64, col_size);
            cuda_sys::cuMemRelease(handle_keys);
            cuda_sys::cuMemRelease(handle_prices);
            cuda_sys::cuMemAddressFree(va_ptr, total_va);
        }
        return Ok(());
    }
    let access_us = t0.elapsed().as_secs_f64() * 1e6;
    println!("  cuMemSetAccess: {:.0}us", access_us);

    let key_dev_ptr = va_ptr;
    let price_dev_ptr = va_ptr + col_size as u64;

    // Copy data to the VMM-mapped memory
    let t0 = Instant::now();
    let res = unsafe {
        cuda_sys::cuMemcpyHtoD_v2(key_dev_ptr, keys_f64.as_ptr() as *const _, key_bytes)
    };
    if res != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cuMemcpyHtoD (keys) FAILED: {:?}", res);
    }
    // Keys are stored as f64 in VMM, but scatter_stats expects i32.
    // Copy actual i32 keys:
    let res = unsafe {
        cuda_sys::cuMemcpyHtoD_v2(key_dev_ptr, keys.as_ptr() as *const _, key_bytes)
    };
    if res != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cuMemcpyHtoD (i32 keys) FAILED: {:?}", res);
    }
    let res = unsafe {
        cuda_sys::cuMemcpyHtoD_v2(price_dev_ptr, prices.as_ptr() as *const _, price_bytes)
    };
    if res != cuda_sys::CUresult::CUDA_SUCCESS {
        println!("  cuMemcpyHtoD (prices) FAILED: {:?}", res);
    }
    engine.stream.synchronize()?;
    let copy_us = t0.elapsed().as_secs_f64() * 1e6;
    println!("  H2D copy: {:.0}us", copy_us);

    // -----------------------------------------------------------------------
    // Benchmark: reduce_sum on VMM memory vs cudaMalloc
    // -----------------------------------------------------------------------
    println!("\n  --- reduce_sum ({} elements) ---", n);

    // VMM
    let sum_vmm = engine.reduce_sum_dev(price_dev_ptr, n)?;
    let sum_expected: f64 = prices.iter().sum();
    let err = (sum_vmm - sum_expected).abs() / sum_expected.abs();
    println!("  Correctness: vmm_sum={:.2}, expected={:.2}, rel_err={:.2e}", sum_vmm, sum_expected, err);

    let n_iters = 50;
    engine.stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.reduce_sum_dev(price_dev_ptr, n)?;
    }
    engine.stream.synchronize()?;
    let vmm_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    // cudaMalloc baseline
    let prices_dev = engine.stream.clone_htod(prices)?;
    let (prices_dev_ptr_malloc, _guard) = prices_dev.device_ptr(&engine.stream);
    engine.stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.reduce_sum_dev(prices_dev_ptr_malloc, n)?;
    }
    engine.stream.synchronize()?;
    let malloc_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    println!("  cudaMalloc: {:.0}us/op", malloc_us);
    println!("  CUDA VMM:   {:.0}us/op", vmm_us);
    println!("  Ratio:      {:.2}x", vmm_us / malloc_us);

    // -----------------------------------------------------------------------
    // scatter_stats on VMM
    // -----------------------------------------------------------------------
    println!("\n  --- scatter_stats ({} elements, {} groups) ---", n, n_groups);

    let (sums_v, _, counts_v) = engine.scatter_stats_dev(key_dev_ptr, price_dev_ptr, n, n_groups)?;
    let active_v = counts_v.iter().filter(|&&c| c > 0.0).count();
    println!("  VMM: {} active groups", active_v);

    engine.stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.scatter_stats_dev(key_dev_ptr, price_dev_ptr, n, n_groups)?;
    }
    engine.stream.synchronize()?;
    let vmm_scatter_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    let keys_dev = engine.stream.clone_htod(keys)?;
    let (keys_dev_ptr_malloc, _guard2) = keys_dev.device_ptr(&engine.stream);
    engine.stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.scatter_stats_dev(keys_dev_ptr_malloc, prices_dev_ptr_malloc, n, n_groups)?;
    }
    engine.stream.synchronize()?;
    let malloc_scatter_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    println!("  cudaMalloc: {:.0}us/op", malloc_scatter_us);
    println!("  CUDA VMM:   {:.0}us/op", vmm_scatter_us);
    println!("  Ratio:      {:.2}x", vmm_scatter_us / malloc_scatter_us);

    // Total setup time
    let setup_total = reserve_us + create_us + map_us + access_us;
    println!("\n  --- Setup cost ---");
    println!("  Total VMM setup: {:.0}us", setup_total);
    println!("  (Reserve {:.0} + Create {:.0} + Map {:.0} + Access {:.0})",
        reserve_us, create_us, map_us, access_us);

    // Cleanup
    unsafe {
        cuda_sys::cuMemUnmap(va_ptr, col_size);
        cuda_sys::cuMemUnmap(va_ptr + col_size as u64, col_size);
        cuda_sys::cuMemRelease(handle_keys);
        cuda_sys::cuMemRelease(handle_prices);
        cuda_sys::cuMemAddressFree(va_ptr, total_va);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn load_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    let n = bytes.len() / 4;
    let mut out = vec![0f32; n];
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len()); }
    out
}

fn load_i64(path: &Path) -> Vec<i64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    let n = bytes.len() / 8;
    let mut out = vec![0i64; n];
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len()); }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== I/O Path Benchmark: File-Mapped GPU ===\n");

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let tb_path = data_dir.join("aapl_2025-09-02.tb");

    // Ensure .tb file exists (write it if not)
    if !tb_path.exists() {
        println!("Writing .tb file first...");
        let prices_f32 = load_f32(&data_dir.join("aapl_prices_f32.bin"));
        let timestamps = load_i64(&data_dir.join("aapl_timestamps_i64.bin"));
        let min_ts = timestamps.iter().cloned().min().unwrap_or(0);
        let ns_per_minute: i64 = 60_000_000_000;
        let minute_bins: Vec<i32> = timestamps.iter()
            .map(|&ts| ((ts - min_ts) / ns_per_minute) as i32)
            .collect();
        let prices: Vec<f64> = prices_f32.iter().map(|&p| p as f64).collect();
        tambear::write_tb(
            &tb_path,
            &[
                tambear::TbColumnWrite::key_column("minute", &minute_bins),
                tambear::TbColumnWrite::f64_column("price", prices),
            ],
            65_536,
        )?;
    }

    // Load data for baseline comparison
    let prices_f32 = load_f32(&data_dir.join("aapl_prices_f32.bin"));
    let timestamps = load_i64(&data_dir.join("aapl_timestamps_i64.bin"));
    let min_ts = timestamps.iter().cloned().min().unwrap_or(0);
    let ns_per_minute: i64 = 60_000_000_000;
    let keys: Vec<i32> = timestamps.iter()
        .map(|&ts| ((ts - min_ts) / ns_per_minute) as i32)
        .collect();
    let prices: Vec<f64> = prices_f32.iter().map(|&p| p as f64).collect();
    let n_groups = (*keys.iter().max().unwrap_or(&0) + 1) as usize;
    let n = prices.len();

    println!("{} elements, {} groups, {:.1} MB data",
        n, n_groups, (n * 8) as f64 / 1e6);

    let engine = BenchEngine::new()?;
    println!("NVRTC compiled. Starting benchmarks...");

    // Path 1: Pinned file-mapped
    #[cfg(windows)]
    bench_path1_pinned(&engine, &tb_path, &prices, &keys, n_groups)?;

    // Path 3: CUDA VMM
    bench_path3_vmm(&engine, &prices, &keys, n_groups)?;

    println!("\n=== Summary ===");
    println!("The question: does file-mapped memory perform THE SAME as cudaMalloc'd?");
    println!("If yes: we never 'load' data. We open workspaces.");
    Ok(())
}
