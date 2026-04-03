//! # GPU KMeans via Tambear Primitives
//!
//! KMeans clustering decomposed into 3 tambear primitives:
//! 1. `tiled_accumulate` — pairwise distance computation (each point vs each centroid)
//! 2. `reduce(argmin)` — find nearest centroid per point
//! 3. `scatter(mean)` — update centroids from assigned points
//!
//! No cuML. No scikit-learn. No external ML library.
//! Pure tambear primitives. Any GPU. 10 lines from Python.
//!
//! ```python
//! import tambear as tb
//! data = tb.read("customers.tb").select("feature1", "feature2", "feature3")
//! labels, centroids = tb.kmeans(data, k=5)
//! ```

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;
use std::time::Instant;

/// KMeans result: cluster assignments + final centroids
pub struct KMeansResult {
    pub labels: Vec<u32>,        // cluster assignment per point
    pub centroids: Vec<f32>,     // k × d flattened centroid matrix
    pub k: usize,
    pub d: usize,
    pub iterations: usize,
    pub converged: bool,
}

/// GPU KMeans engine
pub struct KMeansEngine {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

// CUDA kernel: compute distances + assign clusters in ONE fused kernel
// This IS tiled_accumulate + reduce(argmin) fused together
const ASSIGN_KERNEL: &str = r#"
extern "C" __global__ void assign_clusters(
    const float* __restrict__ data,       // n × d, row-major
    const float* __restrict__ centroids,  // k × d, row-major
    unsigned int* __restrict__ labels,    // n assignments
    float* __restrict__ distances,        // n min-distances (for convergence check)
    int n, int k, int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float min_dist = 1e30f;
    unsigned int min_label = 0;

    // For each centroid, compute squared L2 distance
    for (int c = 0; c < k; c++) {
        float dist = 0.0f;
        for (int j = 0; j < d; j++) {
            float diff = data[idx * d + j] - centroids[c * d + j];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            min_label = (unsigned int)c;
        }
    }

    labels[idx] = min_label;
    distances[idx] = min_dist;
}
"#;

// CUDA kernel: update centroids via scatter-add + count
// This IS scatter(sum) + scatter(count) → divide = scatter(mean)
const UPDATE_KERNEL: &str = r#"
extern "C" __global__ void update_centroids(
    const float* __restrict__ data,       // n × d
    const unsigned int* __restrict__ labels, // n assignments
    float* __restrict__ new_centroids,    // k × d (accumulator, zeroed before call)
    unsigned int* __restrict__ counts,    // k counts (zeroed before call)
    int n, int k, int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int label = labels[idx];

    // Atomic add each dimension to the centroid accumulator
    for (int j = 0; j < d; j++) {
        atomicAdd(&new_centroids[label * d + j], data[idx * d + j]);
    }
    atomicAdd(&counts[label], 1u);
}

extern "C" __global__ void normalize_centroids(
    float* __restrict__ centroids,        // k × d
    const unsigned int* __restrict__ counts, // k
    int k, int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;

    unsigned int count = counts[idx];
    if (count > 0) {
        for (int j = 0; j < d; j++) {
            centroids[idx * d + j] /= (float)count;
        }
    }
}
"#;

impl KMeansEngine {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream })
    }

    /// Run KMeans clustering on GPU.
    /// data: n × d matrix (row-major, f32)
    /// k: number of clusters
    /// max_iter: maximum iterations (hard cap)
    ///
    /// Convergence criterion: label stability — stops when no point changes cluster.
    /// This is correct and deterministic regardless of f32 atomicAdd ordering noise.
    pub fn fit(
        &self,
        data: &[f32],
        n: usize,
        d: usize,
        k: usize,
        max_iter: usize,
    ) -> Result<KMeansResult, Box<dyn std::error::Error>> {
        assert_eq!(data.len(), n * d, "data length must be n * d");
        assert!(k <= n, "k must be <= n");

        let t_start = Instant::now();

        // Compile kernels
        let assign_opts = CompileOptions { arch: Some("sm_120"), ..Default::default() };
        let assign_ptx = compile_ptx_with_opts(ASSIGN_KERNEL, assign_opts)?;
        let assign_module = self.ctx.load_module(assign_ptx)?;
        let assign_fn = assign_module.load_function("assign_clusters")?;

        let update_ptx = compile_ptx_with_opts(UPDATE_KERNEL, CompileOptions { arch: Some("sm_120"), ..Default::default() })?;
        let update_module = self.ctx.load_module(update_ptx)?;
        let update_fn = update_module.load_function("update_centroids")?;
        let normalize_fn = update_module.load_function("normalize_centroids")?;

        let t_compile = t_start.elapsed().as_millis();

        // Upload data to GPU
        let d_data = self.stream.clone_htod(data)?;

        // Initialize centroids: evenly spaced across dataset (deterministic, avoids cold-start collapse)
        let step = n / k;
        let init_centroids: Vec<f32> = (0..k)
            .flat_map(|c| data[c * step * d..(c * step * d + d)].iter().copied())
            .collect();
        let mut d_centroids = self.stream.clone_htod(&init_centroids)?;

        // Allocate working buffers
        let mut d_labels: CudaSlice<u32> = self.stream.alloc_zeros(n)?;
        let mut d_distances: CudaSlice<f32> = self.stream.alloc_zeros(n)?;
        let mut d_new_centroids: CudaSlice<f32> = self.stream.alloc_zeros(k * d)?;
        let mut d_counts: CudaSlice<u32> = self.stream.alloc_zeros(k)?;

        let threads = 256u32;
        let blocks_n = ((n as u32) + threads - 1) / threads;
        let blocks_k = ((k as u32) + threads - 1) / threads;

        let n_i = n as i32;
        let k_i = k as i32;
        let d_i = d as i32;

        let mut iterations = 0;
        let mut converged = false;
        let mut prev_labels: Vec<u32> = vec![u32::MAX; n]; // sentinel: no previous labels

        let t_iter_start = Instant::now();

        for _iter in 0..max_iter {
            // Step 1: Assign clusters (fused distance + argmin)
            unsafe {
                self.stream.launch_builder(&assign_fn)
                    .arg(&d_data)
                    .arg(&d_centroids)
                    .arg(&mut d_labels)
                    .arg(&mut d_distances)
                    .arg(&n_i)
                    .arg(&k_i)
                    .arg(&d_i)
                    .launch(LaunchConfig {
                        grid_dim: (blocks_n, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }

            // Check convergence: label stability (discrete — immune to f32 atomicAdd noise)
            // Label stability is the correct stopping criterion: if no point changes cluster,
            // further iterations are guaranteed to produce identical centroids.
            let curr_labels = self.stream.clone_dtoh(&d_labels)?;
            iterations += 1;
            if curr_labels == prev_labels {
                converged = true;
                break;
            }
            prev_labels = curr_labels;

            // Step 2: Update centroids (scatter mean)
            // Zero accumulators
            self.stream.memset_zeros(&mut d_new_centroids)?;
            self.stream.memset_zeros(&mut d_counts)?;

            unsafe {
                self.stream.launch_builder(&update_fn)
                    .arg(&d_data)
                    .arg(&d_labels)
                    .arg(&mut d_new_centroids)
                    .arg(&mut d_counts)
                    .arg(&n_i)
                    .arg(&k_i)
                    .arg(&d_i)
                    .launch(LaunchConfig {
                        grid_dim: (blocks_n, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }

            // Normalize: divide by counts
            unsafe {
                self.stream.launch_builder(&normalize_fn)
                    .arg(&mut d_new_centroids)
                    .arg(&d_counts)
                    .arg(&k_i)
                    .arg(&d_i)
                    .launch(LaunchConfig {
                        grid_dim: (blocks_k, 1, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }

            // Swap centroids
            std::mem::swap(&mut d_centroids, &mut d_new_centroids);
        }

        let t_total = t_iter_start.elapsed().as_micros();

        // Read back results
        let labels_host = self.stream.clone_dtoh(&d_labels)?;
        let centroids_host = self.stream.clone_dtoh(&d_centroids)?;

        eprintln!(
            "KMeans: n={} d={} k={} iters={} converged={} compile={}ms compute={}us ({:.0}us/iter)",
            n, d, k, iterations, converged, t_compile, t_total, t_total as f64 / iterations as f64
        );

        Ok(KMeansResult {
            labels: labels_host,
            centroids: centroids_host,
            k,
            d,
            iterations,
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let engine = KMeansEngine::new().expect("No GPU");

        // 3 obvious clusters in 2D
        let mut data = Vec::new();
        let mut rng: u64 = 42;
        let centers = [(0.0f32, 0.0), (10.0, 10.0), (-10.0, 5.0)];

        for &(cx, cy) in &centers {
            for _ in 0..1000 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let x = cx + ((rng >> 40) as f32 / 1e12 - 0.5) * 2.0;
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let y = cy + ((rng >> 40) as f32 / 1e12 - 0.5) * 2.0;
                data.push(x);
                data.push(y);
            }
        }

        // Label-stability convergence: discrete, deterministic regardless of GPU thread scheduling.
        let result = engine.fit(&data, 3000, 2, 3, 200).unwrap();

        assert!(result.converged, "Should converge within 200 iterations (got {})", result.iterations);
        assert!(result.iterations < 100, "Should converge quickly (got {})", result.iterations);
        assert_eq!(result.labels.len(), 3000);
        assert_eq!(result.centroids.len(), 6); // 3 × 2

        // Check that we found 3 distinct clusters
        let mut cluster_counts = [0u32; 3];
        for &l in &result.labels {
            cluster_counts[l as usize] += 1;
        }
        // Each cluster should have ~1000 points
        for count in &cluster_counts {
            assert!(*count > 500 && *count < 1500, "Cluster balance: {:?}", cluster_counts);
        }

        println!("KMeans test PASSED: {:?} points per cluster, {} iterations",
                 cluster_counts, result.iterations);
    }
}
