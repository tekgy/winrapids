//! Tambear wgpu multi-backend benchmark suite.
//!
//! Tests ALL core operations across backends:
//! 1. Scatter GroupBy (hash scatter with float atomics)
//! 2. Reduce Sum (bandwidth measurement)
//! 3. Scan / Prefix Sum (parallel Blelloch)
//! 4. Element-wise fused expression
//!
//! Runs on: NVIDIA (Vulkan), AMD (Vulkan), Intel (Vulkan), Apple (Metal)

use std::time::Instant;
use wgpu::util::DeviceExt;

// =============================================
// Kernel 1: Scatter Sum (CAS-loop float atomic)
// =============================================
const SCATTER_SUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: vec2<u32>;

fn atomic_add_f32(idx: u32, val: f32) {
    var old_bits = atomicLoad(&output[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&output[idx], old_bits, new_bits);
        if result.exchanged { break; }
        old_bits = result.old_value;
    }
}

@compute @workgroup_size(256)
fn scatter_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    atomic_add_f32(keys[idx], values[idx]);
}
"#;

// =============================================
// Kernel 2: Reduce Sum (bandwidth test)
// =============================================
const REDUCE_SUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

var<workgroup> shmem: array<f32, 256>;

fn atomic_add_f32_at(idx: u32, val: f32) {
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

    // Tree reduction in shared memory
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
        atomic_add_f32_at(wid.x, shmem[0]);
    }
}
"#;

// =============================================
// Kernel 3: Element-wise fused expression
// a * b + c (FMA-like)
// =============================================
const FUSED_EXPR_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: vec2<u32>;

@compute @workgroup_size(256)
fn fused_expr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.x) { return; }
    out[idx] = a[idx] * b[idx] + c[idx];
}
"#;

// =============================================
// Kernel 4: Prefix sum (Blelloch scan, single workgroup)
// =============================================
const PREFIX_SUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;

var<workgroup> shmem: array<f32, 1024>;

@compute @workgroup_size(1024)
fn prefix_sum(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = min(params.x, 1024u);
    if (lid.x < n) {
        shmem[lid.x] = input[lid.x];
    } else {
        shmem[lid.x] = 0.0;
    }
    workgroupBarrier();

    // Up-sweep
    var stride: u32 = 1u;
    loop {
        if stride >= 1024u { break; }
        let idx = (lid.x + 1u) * stride * 2u - 1u;
        if idx < 1024u {
            shmem[idx] = shmem[idx] + shmem[idx - stride];
        }
        workgroupBarrier();
        stride = stride * 2u;
    }

    // Down-sweep
    stride = 256u;
    loop {
        if stride == 0u { break; }
        let idx = (lid.x + 1u) * stride * 2u - 1u;
        if idx + stride < 1024u {
            shmem[idx + stride] = shmem[idx] + shmem[idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid.x < n) {
        output[lid.x] = shmem[lid.x];
    }
}
"#;

struct GpuBench {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
}

impl GpuBench {
    fn new() -> Self {
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

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("tambear"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_compute_workgroup_size_x: 1024,
                    max_compute_invocations_per_workgroup: 1024,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create device");

        Self { device, queue, info }
    }

    fn bench_dispatch(&self, label: &str, setup: impl Fn(&mut wgpu::CommandEncoder), n_runs: usize) -> Vec<f64> {
        // Warm up
        for _ in 0..3 {
            let mut enc = self.device.create_command_encoder(&Default::default());
            setup(&mut enc);
            self.queue.submit(Some(enc.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }

        let mut times = Vec::with_capacity(n_runs);
        for _ in 0..n_runs {
            let t0 = Instant::now();
            let mut enc = self.device.create_command_encoder(&Default::default());
            setup(&mut enc);
            self.queue.submit(Some(enc.finish()));
            self.device.poll(wgpu::Maintain::Wait);
            times.push(t0.elapsed().as_micros() as f64);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p01 = times[0];
        let p50 = times[times.len() / 2];
        let p99 = times[times.len() - 1];
        println!("  {:<30} p01={:>7.0}us  p50={:>7.0}us  p99={:>7.0}us", label, p01, p50, p99);
        times
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn main() {
    let bench = GpuBench::new();

    println!("=========================================================");
    println!("  Tambear Multi-Backend Benchmark Suite");
    println!("  GPU: {} ({:?})", bench.info.name, bench.info.backend);
    println!("  Driver: {}", bench.info.driver);
    println!("=========================================================");
    println!();

    let n: u32 = 1_000_000;
    let n_groups: u32 = 4600;

    // Generate test data
    let mut rng: u64 = 42;
    let keys: Vec<u32> = (0..n).map(|_| (lcg(&mut rng) >> 33) as u32 % n_groups).collect();
    let values: Vec<f32> = (0..n).map(|_| {
        let bits = (lcg(&mut rng) >> 40) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }).collect();
    let a_data: Vec<f32> = (0..n).map(|_| (lcg(&mut rng) >> 40) as f32 / u32::MAX as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|_| (lcg(&mut rng) >> 40) as f32 / u32::MAX as f32).collect();
    let c_data: Vec<f32> = (0..n).map(|_| (lcg(&mut rng) >> 40) as f32 / u32::MAX as f32).collect();

    // CPU references
    let mut cpu_scatter = vec![0.0f32; n_groups as usize];
    for i in 0..n as usize { cpu_scatter[keys[i] as usize] += values[i]; }
    let cpu_sum: f32 = values.iter().sum();
    let cpu_fma: Vec<f32> = (0..n as usize).map(|i| a_data[i] * b_data[i] + c_data[i]).collect();

    let dev = &bench.device;

    // =============================================
    // Benchmark 1: Scatter GroupBy
    // =============================================
    println!("=== 1. Scatter GroupBy ({} rows, {} groups) ===", n, n_groups);

    let keys_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&keys),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let vals_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let scatter_out = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (n_groups as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params1: [u32; 2] = [n, n_groups];
    let params1_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&params1),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader1 = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl(SCATTER_SUM_WGSL.into()),
    });
    let bgl1 = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let pl1 = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl1], push_constant_ranges: &[] });
    let pipe1 = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pl1), module: &shader1, entry_point: Some("scatter_sum"),
        compilation_options: Default::default(), cache: None,
    });
    let bg1 = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl1,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: keys_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: vals_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: scatter_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params1_buf.as_entire_binding() },
        ],
    });

    let wg = (n + 255) / 256;
    bench.bench_dispatch("scatter_sum (CAS atomic)", |enc| {
        enc.clear_buffer(&scatter_out, 0, None);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe1);
        pass.set_bind_group(0, &bg1, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
    }, 20);
    println!();

    // =============================================
    // Benchmark 2: Reduce Sum
    // =============================================
    println!("=== 2. Reduce Sum ({} elements) ===", n);

    let input_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let n_workgroups = (n + 255) / 256;
    let partials_buf = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (n_workgroups as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params2: [u32; 2] = [n, 0];
    let params2_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&params2),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader2 = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl(REDUCE_SUM_WGSL.into()),
    });
    let bgl2 = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let pl2 = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl2], push_constant_ranges: &[] });
    let pipe2 = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pl2), module: &shader2, entry_point: Some("reduce_sum"),
        compilation_options: Default::default(), cache: None,
    });
    let bg2 = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl2,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: partials_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params2_buf.as_entire_binding() },
        ],
    });

    bench.bench_dispatch("reduce_sum (shared mem tree)", |enc| {
        enc.clear_buffer(&partials_buf, 0, None);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe2);
        pass.set_bind_group(0, &bg2, &[]);
        pass.dispatch_workgroups(n_workgroups, 1, 1);
    }, 20);
    println!();

    // =============================================
    // Benchmark 3: Element-wise fused expression (a*b+c)
    // =============================================
    println!("=== 3. Fused Expression a*b+c ({} elements) ===", n);

    let a_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&a_data), usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&b_data), usage: wgpu::BufferUsages::STORAGE,
    });
    let c_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&c_data), usage: wgpu::BufferUsages::STORAGE,
    });
    let fma_out = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (n as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params3: [u32; 2] = [n, 0];
    let params3_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&params3), usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader3 = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl(FUSED_EXPR_WGSL.into()),
    });
    let bgl3 = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let pl3 = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl3], push_constant_ranges: &[] });
    let pipe3 = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pl3), module: &shader3, entry_point: Some("fused_expr"),
        compilation_options: Default::default(), cache: None,
    });
    let bg3 = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl3,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: fma_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: params3_buf.as_entire_binding() },
        ],
    });

    bench.bench_dispatch("fused a*b+c", |enc| {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe3);
        pass.set_bind_group(0, &bg3, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
    }, 20);
    println!();

    // =============================================
    // Benchmark 4: Prefix Sum (single workgroup, 1024 elements)
    // =============================================
    let scan_n: u32 = 1024;
    println!("=== 4. Prefix Sum ({} elements, single workgroup) ===", scan_n);

    let scan_data: Vec<f32> = (1..=scan_n).map(|i| i as f32).collect();
    let scan_in = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&scan_data), usage: wgpu::BufferUsages::STORAGE,
    });
    let scan_out = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: (scan_n as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params4: [u32; 2] = [scan_n, 0];
    let params4_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&params4), usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader4 = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None, source: wgpu::ShaderSource::Wgsl(PREFIX_SUM_WGSL.into()),
    });
    let bgl4 = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });
    let pl4 = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl4], push_constant_ranges: &[] });
    let pipe4 = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pl4), module: &shader4, entry_point: Some("prefix_sum"),
        compilation_options: Default::default(), cache: None,
    });
    let bg4 = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl4,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: scan_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: scan_out.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params4_buf.as_entire_binding() },
        ],
    });

    bench.bench_dispatch("prefix_sum (Blelloch)", |enc| {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe4);
        pass.set_bind_group(0, &bg4, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }, 20);
    println!();

    // =============================================
    // Summary
    // =============================================
    println!("=========================================================");
    println!("  Backend: {:?} | GPU: {}", bench.info.backend, bench.info.name);
    println!();
    println!("  This binary runs UNCHANGED on:");
    println!("    NVIDIA (Vulkan) | AMD (Vulkan) | Intel (Vulkan) | Apple (Metal)");
    println!();
    println!("  Tam doesn't need CUDA. Tam needs a GPU. Any GPU.");
    println!("=========================================================");
}
