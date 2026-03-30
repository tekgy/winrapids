use cudarc::driver::{CudaContext, CudaSlice};

fn main() {
    println!("=== cudarc Windows + CUDA 13.1 Test ===\n");

    // 1. Initialize CUDA driver and get device 0
    let ctx = match CudaContext::new(0) {
        Ok(c) => {
            println!("[OK] CudaContext::new(0) succeeded");
            c
        }
        Err(e) => {
            eprintln!("[FAIL] CudaContext::new(0) failed: {e:?}");
            std::process::exit(1);
        }
    };

    // 2. Query device name
    match ctx.name() {
        Ok(name) => println!("[OK] Device name: {name}"),
        Err(e) => eprintln!("[WARN] Could not query device name: {e:?}"),
    }

    // 3. Query compute capability
    match ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) {
        Ok(major) => {
            match ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) {
                Ok(minor) => println!("[OK] Compute capability: {major}.{minor}"),
                Err(e) => eprintln!("[WARN] Could not query compute minor: {e:?}"),
            }
        }
        Err(e) => eprintln!("[WARN] Could not query compute major: {e:?}"),
    }

    // 4. Get default stream
    let stream = ctx.default_stream();

    // 5. Allocate a small buffer on the GPU (256 floats = 1 KB)
    let host_data: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let gpu_buf: CudaSlice<f32> = match stream.clone_htod(&host_data) {
        Ok(buf) => {
            println!("[OK] Allocated and copied 256 f32s (1 KB) to GPU");
            buf
        }
        Err(e) => {
            eprintln!("[FAIL] clone_htod failed: {e:?}");
            std::process::exit(1);
        }
    };

    // 6. Copy back and verify
    match stream.clone_dtoh(&gpu_buf) {
        Ok(result) => {
            if result == host_data {
                println!("[OK] Device-to-host copy matches — round-trip verified");
            } else {
                eprintln!("[FAIL] Data mismatch after round-trip!");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("[FAIL] clone_dtoh failed: {e:?}");
            std::process::exit(1);
        }
    }

    // 7. Report memory info
    match cudarc::driver::result::mem_get_info() {
        Ok((free, total)) => {
            println!(
                "[OK] GPU memory: {:.1} GB free / {:.1} GB total",
                free as f64 / 1e9,
                total as f64 / 1e9
            );
        }
        Err(e) => eprintln!("[WARN] Could not query memory info: {e:?}"),
    }

    println!("\n=== ALL TESTS PASSED ===");
}
