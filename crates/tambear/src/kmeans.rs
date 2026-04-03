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

use std::sync::Arc;
use std::time::Instant;

use tam_gpu::{Kernel, ShaderLang, TamGpu};

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
    gpu: Arc<dyn TamGpu>,
}

impl KMeansEngine {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self { gpu: tam_gpu::detect() })
    }

    pub fn with_backend(gpu: Arc<dyn TamGpu>) -> Self {
        Self { gpu }
    }

    /// Run KMeans clustering.
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

        match self.gpu.shader_lang() {
            ShaderLang::Cuda => self.fit_cuda(data, n, d, k, max_iter),
            _ => self.fit_cpu(data, n, d, k, max_iter),
        }
    }

    fn fit_cuda(
        &self,
        data: &[f32],
        n: usize,
        d: usize,
        k: usize,
        max_iter: usize,
    ) -> Result<KMeansResult, Box<dyn std::error::Error>> {
        let t_start = Instant::now();

        // Compile kernels with n,k,d baked in (constant for this fit() call)
        let assign_source = emit_assign_kernel(n, k, d);
        let update_source = emit_update_kernel(n, k, d);

        let assign_kernel = self.gpu.compile(&assign_source, "assign_clusters")?;
        let update_kernel = self.gpu.compile(&update_source, "update_centroids")?;
        let normalize_kernel = self.gpu.compile(&update_source, "normalize_centroids")?;

        let t_compile = t_start.elapsed().as_millis();

        // Upload data to GPU
        let d_data = tam_gpu::upload(&*self.gpu, data)?;

        // Initialize centroids: evenly spaced across dataset (deterministic)
        let step = n / k;
        let init_centroids: Vec<f32> = (0..k)
            .flat_map(|c| data[c * step * d..(c * step * d + d)].iter().copied())
            .collect();
        let mut d_centroids = tam_gpu::upload(&*self.gpu, &init_centroids)?;

        // Allocate working buffers
        let mut d_labels = self.gpu.alloc(n * 4)?;       // u32
        let mut d_distances = self.gpu.alloc(n * 4)?;     // f32
        let mut d_new_centroids = self.gpu.alloc(k * d * 4)?; // f32
        let mut d_counts = self.gpu.alloc(k * 4)?;        // u32

        let threads = 256u32;
        let blocks_n = ((n as u32) + threads - 1) / threads;
        let blocks_k = ((k as u32) + threads - 1) / threads;

        let mut iterations = 0;
        let mut converged = false;
        let mut prev_labels: Vec<u32> = vec![u32::MAX; n];

        let t_iter_start = Instant::now();

        for _iter in 0..max_iter {
            // Step 1: Assign clusters (fused distance + argmin)
            self.gpu.dispatch(
                &assign_kernel,
                [blocks_n, 1, 1],
                [threads, 1, 1],
                &[&d_data, &d_centroids, &d_labels, &d_distances],
                0,
            )?;

            // Check convergence: label stability
            self.gpu.sync()?;
            let curr_labels: Vec<u32> = tam_gpu::download(&*self.gpu, &d_labels, n)?;
            iterations += 1;
            if curr_labels == prev_labels {
                converged = true;
                break;
            }
            prev_labels = curr_labels;

            // Step 2: Reset accumulators (free + alloc = zero-initialized)
            self.gpu.free(d_new_centroids)?;
            self.gpu.free(d_counts)?;
            d_new_centroids = self.gpu.alloc(k * d * 4)?;
            d_counts = self.gpu.alloc(k * 4)?;

            // Step 3: Update centroids (scatter mean)
            self.gpu.dispatch(
                &update_kernel,
                [blocks_n, 1, 1],
                [threads, 1, 1],
                &[&d_data, &d_labels, &d_new_centroids, &d_counts],
                0,
            )?;

            // Normalize: divide by counts
            self.gpu.dispatch(
                &normalize_kernel,
                [blocks_k, 1, 1],
                [threads, 1, 1],
                &[&d_new_centroids, &d_counts],
                0,
            )?;

            // Swap centroids
            std::mem::swap(&mut d_centroids, &mut d_new_centroids);
        }

        let t_total = t_iter_start.elapsed().as_micros();

        // Read back results
        self.gpu.sync()?;
        let labels_host: Vec<u32> = tam_gpu::download(&*self.gpu, &d_labels, n)?;
        let centroids_host: Vec<f32> = tam_gpu::download(&*self.gpu, &d_centroids, k * d)?;

        eprintln!(
            "KMeans: n={} d={} k={} iters={} converged={} compile={}ms compute={}us ({:.0}us/iter)",
            n, d, k, iterations, converged, t_compile, t_total, t_total as f64 / iterations as f64
        );

        Ok(KMeansResult { labels: labels_host, centroids: centroids_host, k, d, iterations, converged })
    }

    fn fit_cpu(
        &self,
        data: &[f32],
        n: usize,
        d: usize,
        k: usize,
        max_iter: usize,
    ) -> Result<KMeansResult, Box<dyn std::error::Error>> {
        // Initialize centroids: evenly spaced
        let step = n / k;
        let mut centroids: Vec<f32> = (0..k)
            .flat_map(|c| data[c * step * d..(c * step * d + d)].iter().copied())
            .collect();

        let mut labels = vec![0u32; n];
        let mut iterations = 0;
        let mut converged = false;

        for _iter in 0..max_iter {
            let prev_labels = labels.clone();

            // Assign: each point to nearest centroid
            for i in 0..n {
                let mut min_dist = f32::MAX;
                let mut min_label = 0u32;
                for c in 0..k {
                    let mut dist = 0.0f32;
                    for j in 0..d {
                        let diff = data[i * d + j] - centroids[c * d + j];
                        dist += diff * diff;
                    }
                    if dist < min_dist {
                        min_dist = dist;
                        min_label = c as u32;
                    }
                }
                labels[i] = min_label;
            }

            iterations += 1;
            if labels == prev_labels {
                converged = true;
                break;
            }

            // Update centroids
            let mut new_centroids = vec![0.0f32; k * d];
            let mut counts = vec![0u32; k];
            for i in 0..n {
                let c = labels[i] as usize;
                counts[c] += 1;
                for j in 0..d {
                    new_centroids[c * d + j] += data[i * d + j];
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..d {
                        new_centroids[c * d + j] /= counts[c] as f32;
                    }
                }
            }
            centroids = new_centroids;
        }

        Ok(KMeansResult { labels, centroids, k, d, iterations, converged })
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel source generators (n, k, d baked in at compile time)
// ---------------------------------------------------------------------------

fn emit_assign_kernel(n: usize, k: usize, d: usize) -> String {
    format!(r#"
#define N {}
#define K {}
#define D {}

extern "C" __global__ void assign_clusters(
    const float* __restrict__ data,
    const float* __restrict__ centroids,
    unsigned int* __restrict__ labels,
    float* __restrict__ distances
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float min_dist = 1e30f;
    unsigned int min_label = 0;

    for (int c = 0; c < K; c++) {{
        float dist = 0.0f;
        for (int j = 0; j < D; j++) {{
            float diff = data[idx * D + j] - centroids[c * D + j];
            dist += diff * diff;
        }}
        if (dist < min_dist) {{
            min_dist = dist;
            min_label = (unsigned int)c;
        }}
    }}

    labels[idx] = min_label;
    distances[idx] = min_dist;
}}
"#, n, k, d)
}

fn emit_update_kernel(n: usize, k: usize, d: usize) -> String {
    format!(r#"
#define N {}
#define K {}
#define D {}

extern "C" __global__ void update_centroids(
    const float* __restrict__ data,
    const unsigned int* __restrict__ labels,
    float* __restrict__ new_centroids,
    unsigned int* __restrict__ counts
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    unsigned int label = labels[idx];
    for (int j = 0; j < D; j++) {{
        atomicAdd(&new_centroids[label * D + j], data[idx * D + j]);
    }}
    atomicAdd(&counts[label], 1u);
}}

extern "C" __global__ void normalize_centroids(
    float* __restrict__ centroids,
    const unsigned int* __restrict__ counts
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    unsigned int count = counts[idx];
    if (count > 0) {{
        for (int j = 0; j < D; j++) {{
            centroids[idx * D + j] /= (float)count;
        }}
    }}
}}
"#, n, k, d)
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
