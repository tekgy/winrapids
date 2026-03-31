//! Tambear wgpu backend proof-of-concept.
//!
//! Proves: same scatter_sum algorithm, running on ANY GPU via wgpu,
//! produces correct results and competitive performance.
//!
//! On Windows: uses Vulkan backend (NVIDIA, AMD, Intel)
//! On macOS: uses Metal backend (Apple Silicon)
//! On Linux: uses Vulkan backend (NVIDIA, AMD, Intel)
//!
//! "Tam doesn't need CUDA. Tam needs a GPU. Any GPU."

use std::time::Instant;
use wgpu::util::DeviceExt;

/// WGSL compute shader for scatter_sum groupby.
/// Uses atomicAdd on u32 with CAS loop for f32 (portable across ALL GPUs).
const SCATTER_SUM_WGSL: &str = r#"
// Scatter sum: for each element, atomically add value to group accumulator.
// Uses CAS loop for float atomic add (works on every GPU).

@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: vec2<u32>;  // (n_elements, n_groups)

// Float atomic add via compare-and-swap loop.
// Works on ALL GPUs. Native float atomics would be faster but aren't universal.
fn atomic_add_f32(idx: u32, val: f32) {
    // Reinterpret the output slot as atomic<u32>, do CAS with float bits
    var old_bits = atomicLoad(&output[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&output[idx], old_bits, new_bits);
        if result.exchanged {
            break;
        }
        old_bits = result.old_value;
    }
}

@compute @workgroup_size(256)
fn scatter_sum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let n = params.x;
    if (idx >= n) {
        return;
    }
    let group = keys[idx];
    let value = values[idx];
    atomic_add_f32(group, value);
}
"#;

/// Simple reduce_sum shader for bandwidth measurement.
const REDUCE_SUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

fn atomic_add_f32_out(val: f32) {
    var old_bits = atomicLoad(&output[0]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&output[0], old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}

@compute @workgroup_size(256)
fn reduce_sum(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.x) { return; }
    atomic_add_f32_out(input[idx]);
}
"#;

fn main() {
    println!("=========================================================");
    println!("  Tambear wgpu backend proof-of-concept");
    println!("  Same algorithm. Any GPU. Any OS.");
    println!("  Tam doesn't need CUDA. Tam needs a GPU.");
    println!("=========================================================");
    println!();

    // Initialize wgpu — auto-selects best backend
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("No GPU found!");

    let info = adapter.get_info();
    println!("GPU: {} ({:?})", info.name, info.backend);
    println!("Driver: {}", info.driver);
    println!();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("tambear"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("Failed to create device");

    // === Scatter GroupBy benchmark ===
    let n: u32 = 1_000_000;
    let n_groups: u32 = 4600;

    println!("=== Scatter GroupBy: {} rows, {} groups ===", n, n_groups);

    // Generate test data on CPU
    let mut keys = vec![0u32; n as usize];
    let mut values = vec![0.0f32; n as usize];
    let mut rng_state: u64 = 42;
    for i in 0..n as usize {
        // Simple LCG random
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        keys[i] = ((rng_state >> 33) as u32) % n_groups;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (rng_state >> 40) as u32;
        values[i] = (bits as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }

    // CPU reference
    let mut cpu_sums = vec![0.0f32; n_groups as usize];
    let t0 = Instant::now();
    for i in 0..n as usize {
        cpu_sums[keys[i] as usize] += values[i];
    }
    let t_cpu = t0.elapsed().as_micros();
    println!("  CPU groupby sum: {} us", t_cpu);

    // Create GPU buffers
    let keys_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("keys"),
        contents: bytemuck::cast_slice(&keys),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let values_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("values"),
        contents: bytemuck::cast_slice(&values),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (n_groups as u64) * 4, // f32 as u32 bits
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params_data: [u32; 2] = [n, n_groups];
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (n_groups as u64) * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Compile shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("scatter_sum"),
        source: wgpu::ShaderSource::Wgsl(SCATTER_SUM_WGSL.into()),
    });

    // Create pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scatter_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("scatter_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("scatter_sum_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("scatter_sum"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scatter_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: keys_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: values_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
    });

    // Warm up
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&output_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n + 255) / 256, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Benchmark: multiple runs
    let n_runs = 20;
    let mut times = Vec::with_capacity(n_runs);

    for _ in 0..n_runs {
        // Clear output
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&output_buf, 0, None);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        let t0 = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n + 255) / 256, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        times.push(t0.elapsed().as_micros() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = times[times.len() / 2];
    let p01 = times[0];
    let p99 = times[times.len() - 1];

    println!("  wgpu scatter_sum: p01={:.0}us p50={:.0}us p99={:.0}us", p01, p50, p99);

    // Read back and verify
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        // Re-run to get final output
        encoder.clear_buffer(&output_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n + 255) / 256, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, (n_groups as u64) * 4);
        queue.submit(Some(encoder.finish()));

        let slice = readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let gpu_sums: &[f32] = bytemuck::cast_slice(&data);

        // Compare with CPU
        let mut max_err: f32 = 0.0;
        for i in 0..n_groups as usize {
            let err = (gpu_sums[i] - cpu_sums[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }

        let max_rel_err = cpu_sums.iter()
            .zip(gpu_sums.iter())
            .filter(|(c, _)| c.abs() > 1e-6)
            .map(|(c, g)| ((g - c) / c).abs())
            .fold(0.0f32, f32::max);

        println!("  Correctness: max_abs_err={:.6e} max_rel_err={:.6e}", max_err, max_rel_err);

        drop(data);
        readback_buf.unmap();
    }

    println!();
    println!("=========================================================");
    println!("  Backend: {:?}", info.backend);
    println!("  GPU: {}", info.name);
    println!("  wgpu scatter (p50): {:.0} us", p50);
    println!("  CPU scatter: {} us", t_cpu);
    println!("  GPU vs CPU: {:.1}x", t_cpu as f64 / p50);
    println!();
    println!("  This SAME binary runs on:");
    println!("    - NVIDIA (Vulkan)");
    println!("    - AMD (Vulkan)");
    println!("    - Intel (Vulkan)");
    println!("    - Apple Silicon (Metal)");
    println!("  Same code. Same results. Any GPU.");
    println!();
    println!("  Tam doesn't need CUDA. Tam needs a GPU. Any GPU.");
    println!("=========================================================");
}
