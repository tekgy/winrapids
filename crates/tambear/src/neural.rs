//! # Family 23 — Neural Network Ops
//!
//! cuDNN replacement. From first principles.
//!
//! ## What lives here
//!
//! **Activations**: ReLU, LeakyReLU, ELU, SELU, GELU, Swish/SiLU, Mish,
//!   Sigmoid, Tanh, Softmax, LogSoftmax, Softplus, Softsign, HardSwish, HardSigmoid
//! **Convolution**: 1D/2D forward (im2col + GEMM), transposed conv2d
//! **Pooling**: max, average, global average (1D/2D), adaptive
//! **Normalization**: BatchNorm, LayerNorm, GroupNorm, RMSNorm, InstanceNorm
//! **Dropout**: element-wise with deterministic mask from TamRng
//! **Attention**: scaled dot-product, multi-head, causal masking
//! **Linear**: dense layer (mat_mul + bias), bilinear
//! **Embedding**: lookup table, positional encoding (sinusoidal)
//! **Loss**: MSE, CrossEntropy, BCE, Huber, CTC-inspired edit distance
//!
//! ## Architecture
//!
//! Every op has a forward function returning activations. Backward passes
//! return gradients as plain Vec<f64>. No autograd graph — the caller
//! (training infrastructure, F24) composes backward passes explicitly.
//!
//! Conv2D uses im2col to reshape input patches into a matrix, then
//! matrix multiply with the kernel matrix. This is how cuDNN works
//! internally — the insight is that convolution IS matrix multiplication
//! after the right reshape.
//!
//! ## MSR insight
//!
//! A neural network layer's MSR is its weight tensor + the activation
//! function choice. The forward pass is a deterministic function of
//! (weights, input). The backward pass is a deterministic function of
//! (weights, input, grad_output). No hidden state beyond the weights.
//!
//! Attention's MSR is even simpler: Q, K, V projections. The attention
//! matrix is an intermediate — the sufficient statistic is just the
//! three projection matrices.

use std::f64::consts::PI;
use crate::linear_algebra::{Mat, mat_mul};
use crate::rng::TamRng;

// ═══════════════════════════════════════════════════════════════════════
// ACTIVATION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════

/// ReLU: max(0, x)
#[inline]
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// ReLU applied element-wise to a slice.
pub fn relu_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| relu(v)).collect()
}

/// ReLU backward: 1 if x > 0, else 0.
pub fn relu_backward(x: &[f64], grad_out: &[f64]) -> Vec<f64> {
    x.iter().zip(grad_out).map(|(&xi, &g)| if xi > 0.0 { g } else { 0.0 }).collect()
}

/// Leaky ReLU: x if x > 0, else alpha * x.
#[inline]
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * x }
}

pub fn leaky_relu_vec(x: &[f64], alpha: f64) -> Vec<f64> {
    x.iter().map(|&v| leaky_relu(v, alpha)).collect()
}

pub fn leaky_relu_backward(x: &[f64], grad_out: &[f64], alpha: f64) -> Vec<f64> {
    x.iter().zip(grad_out).map(|(&xi, &g)| if xi > 0.0 { g } else { alpha * g }).collect()
}

/// ELU: x if x > 0, else alpha * (exp(x) - 1).
#[inline]
pub fn elu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

pub fn elu_vec(x: &[f64], alpha: f64) -> Vec<f64> {
    x.iter().map(|&v| elu(v, alpha)).collect()
}

pub fn elu_backward(x: &[f64], grad_out: &[f64], alpha: f64) -> Vec<f64> {
    x.iter().zip(grad_out).map(|(&xi, &g)| {
        if xi > 0.0 { g } else { g * alpha * xi.exp() }
    }).collect()
}

/// SELU: lambda * (x if x > 0, else alpha * (exp(x) - 1)).
/// Uses Klambauer et al. 2017 constants for self-normalizing property.
pub const SELU_ALPHA: f64 = 1.6732632423543772;
pub const SELU_LAMBDA: f64 = 1.0507009873554805;

#[inline]
pub fn selu(x: f64) -> f64 {
    SELU_LAMBDA * if x > 0.0 { x } else { SELU_ALPHA * (x.exp() - 1.0) }
}

pub fn selu_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| selu(v)).collect()
}

/// GELU: x * Φ(x) where Φ is the standard normal CDF.
/// Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[inline]
pub fn gelu(x: f64) -> f64 {
    let c = (2.0 / PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

pub fn gelu_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| gelu(v)).collect()
}

pub fn gelu_backward(x: &[f64], grad_out: &[f64]) -> Vec<f64> {
    let c = (2.0 / PI).sqrt();
    x.iter().zip(grad_out).map(|(&xi, &g)| {
        let inner = c * (xi + 0.044715 * xi * xi * xi);
        let tanh_val = inner.tanh();
        let sech2 = 1.0 - tanh_val * tanh_val;
        let d_inner = c * (1.0 + 3.0 * 0.044715 * xi * xi);
        g * (0.5 * (1.0 + tanh_val) + 0.5 * xi * sech2 * d_inner)
    }).collect()
}

/// Swish (SiLU): x * sigmoid(x)
#[inline]
pub fn swish(x: f64) -> f64 {
    x * sigmoid(x)
}

pub fn swish_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| swish(v)).collect()
}

pub fn swish_backward(x: &[f64], grad_out: &[f64]) -> Vec<f64> {
    x.iter().zip(grad_out).map(|(&xi, &g)| {
        let s = sigmoid(xi);
        g * (s + xi * s * (1.0 - s))
    }).collect()
}

/// Mish: x * tanh(softplus(x))
#[inline]
pub fn mish(x: f64) -> f64 {
    x * softplus(x).tanh()
}

pub fn mish_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| mish(v)).collect()
}

/// Sigmoid: 1 / (1 + exp(-x)), numerically stable.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

pub fn sigmoid_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| sigmoid(v)).collect()
}

pub fn sigmoid_backward(x: &[f64], grad_out: &[f64]) -> Vec<f64> {
    x.iter().zip(grad_out).map(|(&xi, &g)| {
        let s = sigmoid(xi);
        g * s * (1.0 - s)
    }).collect()
}

/// Tanh applied element-wise.
pub fn tanh_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| v.tanh()).collect()
}

pub fn tanh_backward(x: &[f64], grad_out: &[f64]) -> Vec<f64> {
    x.iter().zip(grad_out).map(|(&xi, &g)| {
        let t = xi.tanh();
        g * (1.0 - t * t)
    }).collect()
}

/// Softplus: ln(1 + exp(x)), numerically stable.
#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x  // exp(x) dominates, ln(1 + exp(x)) ≈ x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

pub fn softplus_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| softplus(v)).collect()
}

/// Softsign: x / (1 + |x|)
#[inline]
pub fn softsign(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

pub fn softsign_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| softsign(v)).collect()
}

/// Hard sigmoid: clamp((x + 3) / 6, 0, 1)
#[inline]
pub fn hard_sigmoid(x: f64) -> f64 {
    ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

pub fn hard_sigmoid_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| hard_sigmoid(v)).collect()
}

/// Hard swish: x * hard_sigmoid(x)
#[inline]
pub fn hard_swish(x: f64) -> f64 {
    x * hard_sigmoid(x)
}

pub fn hard_swish_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| hard_swish(v)).collect()
}

/// Softmax over a slice. Numerically stable (subtract max).
/// NaN propagates: if any input is NaN the output contains NaN.
pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() { return vec![]; }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
    let exps: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-softmax: log(softmax(x)), numerically stable via log-sum-exp.
/// NaN propagates: if any input is NaN the output contains NaN.
pub fn log_softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() { return vec![]; }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
    let lse = max_val + x.iter().map(|&v| (v - max_val).exp()).sum::<f64>().ln();
    x.iter().map(|&v| v - lse).collect()
}

/// Softmax backward: given softmax output s and grad_output g,
/// returns grad_input = s * (g - sum(s * g)).
pub fn softmax_backward(softmax_out: &[f64], grad_out: &[f64]) -> Vec<f64> {
    let dot: f64 = softmax_out.iter().zip(grad_out).map(|(&s, &g)| s * g).sum();
    softmax_out.iter().zip(grad_out).map(|(&s, &g)| s * (g - dot)).collect()
}

// ═══════════════════════════════════════════════════════════════════════
// CONVOLUTION (im2col approach)
// ═══════════════════════════════════════════════════════════════════════

/// 1D convolution. Input shape: [length], kernel shape: [kernel_size].
/// Returns output of length (input_len - kernel_size) / stride + 1.
pub fn conv1d(input: &[f64], kernel: &[f64], stride: usize, padding: usize) -> Vec<f64> {
    let n = input.len();
    let k = kernel.len();
    let stride = stride.max(1);

    // Pad input
    let padded_len = n + 2 * padding;
    let mut padded = vec![0.0; padded_len];
    for i in 0..n {
        padded[i + padding] = input[i];
    }

    let out_len = (padded_len - k) / stride + 1;
    let mut output = vec![0.0; out_len];
    for i in 0..out_len {
        let start = i * stride;
        let mut sum = 0.0;
        for j in 0..k {
            sum += padded[start + j] * kernel[j];
        }
        output[i] = sum;
    }
    output
}

/// Multi-channel 1D convolution.
/// input: [in_channels][length], kernels: [out_channels][in_channels][kernel_size], bias: [out_channels]
pub fn conv1d_multi(
    input: &[Vec<f64>],
    kernels: &[Vec<Vec<f64>>],
    bias: Option<&[f64]>,
    stride: usize,
    padding: usize,
) -> Vec<Vec<f64>> {
    let out_channels = kernels.len();
    let mut output = Vec::with_capacity(out_channels);
    for oc in 0..out_channels {
        let mut channel_out: Option<Vec<f64>> = None;
        for (ic, inp_ch) in input.iter().enumerate() {
            let conv = conv1d(inp_ch, &kernels[oc][ic], stride, padding);
            match channel_out {
                None => channel_out = Some(conv),
                Some(ref mut out) => {
                    for (o, c) in out.iter_mut().zip(conv.iter()) {
                        *o += c;
                    }
                }
            }
        }
        let mut out = channel_out.unwrap_or_default();
        if let Some(b) = bias {
            for o in out.iter_mut() {
                *o += b[oc];
            }
        }
        output.push(out);
    }
    output
}

/// 2D convolution via im2col + GEMM.
/// input: [in_channels][height][width] flattened as [in_channels * h * w]
/// kernel: [out_channels][in_channels][kh][kw] flattened as [out_channels * in_channels * kh * kw]
/// Returns [out_channels * oh * ow].
pub fn conv2d(
    input: &[f64],
    in_channels: usize,
    h: usize,
    w: usize,
    kernel: &[f64],
    out_channels: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> Vec<f64> {
    let stride = stride.max(1);
    let ph = h + 2 * padding;
    let pw = w + 2 * padding;
    let oh = (ph - kh) / stride + 1;
    let ow = (pw - kw) / stride + 1;

    // Build padded input
    let mut padded = vec![0.0; in_channels * ph * pw];
    for c in 0..in_channels {
        for r in 0..h {
            for col in 0..w {
                padded[c * ph * pw + (r + padding) * pw + (col + padding)] =
                    input[c * h * w + r * w + col];
            }
        }
    }

    // im2col: [in_channels * kh * kw, oh * ow]
    let col_rows = in_channels * kh * kw;
    let col_cols = oh * ow;
    let mut col_matrix = vec![0.0; col_rows * col_cols];

    for c in 0..in_channels {
        for kr in 0..kh {
            for kc in 0..kw {
                let row_idx = c * kh * kw + kr * kw + kc;
                for i in 0..oh {
                    for j in 0..ow {
                        let r = i * stride + kr;
                        let col_pos = j + i * ow;
                        col_matrix[row_idx * col_cols + col_pos] =
                            padded[c * ph * pw + r * pw + (j * stride + kc)];
                    }
                }
            }
        }
    }

    // Kernel matrix: [out_channels, in_channels * kh * kw]
    // kernel is already in this layout
    let kernel_mat = Mat::from_vec(out_channels, col_rows, kernel.to_vec());
    let col_mat = Mat::from_vec(col_rows, col_cols, col_matrix);
    let result = mat_mul(&kernel_mat, &col_mat);

    result.data
}

/// Transposed 2D convolution (deconvolution / upsampling).
/// Inserts (stride-1) zeros between input elements, then convolves.
pub fn conv2d_transpose(
    input: &[f64],
    in_channels: usize,
    h: usize,
    w: usize,
    kernel: &[f64],
    out_channels: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) -> Vec<f64> {
    let stride = stride.max(1);
    // Upsample: insert stride-1 zeros between elements
    let up_h = (h - 1) * stride + 1;
    let up_w = (w - 1) * stride + 1;
    let mut upsampled = vec![0.0; in_channels * up_h * up_w];
    for c in 0..in_channels {
        for r in 0..h {
            for col in 0..w {
                upsampled[c * up_h * up_w + r * stride * up_w + col * stride] =
                    input[c * h * w + r * w + col];
            }
        }
    }

    // Flip kernel: for each (oc, ic) pair, rotate 180 degrees
    // Original: [out_ch][in_ch][kh][kw] — transposed conv uses [in_ch][out_ch][kh_flip][kw_flip]
    let mut flipped = vec![0.0; out_channels * in_channels * kh * kw];
    for oc in 0..out_channels {
        for ic in 0..in_channels {
            for kr in 0..kh {
                for kc in 0..kw {
                    flipped[ic * out_channels * kh * kw + oc * kh * kw + (kh - 1 - kr) * kw + (kw - 1 - kc)] =
                        kernel[oc * in_channels * kh * kw + ic * kh * kw + kr * kw + kc];
                }
            }
        }
    }

    // Full convolution with padding = kh - 1 - padding
    let full_pad = kh - 1 - padding.min(kh - 1);
    conv2d(&upsampled, in_channels, up_h, up_w, &flipped, out_channels, kh, kw, 1, full_pad)
}

// ═══════════════════════════════════════════════════════════════════════
// POOLING
// ═══════════════════════════════════════════════════════════════════════

/// 1D max pooling. Returns (pooled_values, argmax_indices).
pub fn max_pool1d(input: &[f64], kernel_size: usize, stride: usize) -> (Vec<f64>, Vec<usize>) {
    let n = input.len();
    let stride = stride.max(1);
    let out_len = (n - kernel_size) / stride + 1;
    let mut output = Vec::with_capacity(out_len);
    let mut indices = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let start = i * stride;
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx = start;
        for j in 0..kernel_size {
            if input[start + j] > max_val {
                max_val = input[start + j];
                max_idx = start + j;
            }
        }
        output.push(max_val);
        indices.push(max_idx);
    }
    (output, indices)
}

/// 1D average pooling.
pub fn avg_pool1d(input: &[f64], kernel_size: usize, stride: usize) -> Vec<f64> {
    let n = input.len();
    let stride = stride.max(1);
    let out_len = (n - kernel_size) / stride + 1;
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let start = i * stride;
        let sum: f64 = input[start..start + kernel_size].iter().sum();
        output.push(sum / kernel_size as f64);
    }
    output
}

/// 2D max pooling. Input: [channels * h * w]. Returns (pooled, argmax).
pub fn max_pool2d(
    input: &[f64],
    channels: usize,
    h: usize,
    w: usize,
    pool_h: usize,
    pool_w: usize,
    stride: usize,
) -> (Vec<f64>, Vec<usize>) {
    let stride = stride.max(1);
    let oh = (h - pool_h) / stride + 1;
    let ow = (w - pool_w) / stride + 1;
    let mut output = Vec::with_capacity(channels * oh * ow);
    let mut indices = Vec::with_capacity(channels * oh * ow);

    for c in 0..channels {
        let base = c * h * w;
        for i in 0..oh {
            for j in 0..ow {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_idx = 0;
                for pr in 0..pool_h {
                    for pc in 0..pool_w {
                        let r = i * stride + pr;
                        let col = j * stride + pc;
                        let idx = base + r * w + col;
                        if input[idx] > max_val {
                            max_val = input[idx];
                            max_idx = idx;
                        }
                    }
                }
                output.push(max_val);
                indices.push(max_idx);
            }
        }
    }
    (output, indices)
}

/// 2D average pooling.
pub fn avg_pool2d(
    input: &[f64],
    channels: usize,
    h: usize,
    w: usize,
    pool_h: usize,
    pool_w: usize,
    stride: usize,
) -> Vec<f64> {
    let stride = stride.max(1);
    let oh = (h - pool_h) / stride + 1;
    let ow = (w - pool_w) / stride + 1;
    let area = (pool_h * pool_w) as f64;
    let mut output = Vec::with_capacity(channels * oh * ow);

    for c in 0..channels {
        let base = c * h * w;
        for i in 0..oh {
            for j in 0..ow {
                let mut sum = 0.0;
                for pr in 0..pool_h {
                    for pc in 0..pool_w {
                        sum += input[base + (i * stride + pr) * w + (j * stride + pc)];
                    }
                }
                output.push(sum / area);
            }
        }
    }
    output
}

/// Global average pooling over spatial dims. Input: [channels * h * w] → [channels].
pub fn global_avg_pool2d(input: &[f64], channels: usize, h: usize, w: usize) -> Vec<f64> {
    let spatial = h * w;
    if spatial == 0 { return vec![0.0; channels]; }
    let inv = 1.0 / spatial as f64;
    (0..channels).map(|c| {
        let base = c * spatial;
        input[base..base + spatial].iter().sum::<f64>() * inv
    }).collect()
}

/// Adaptive average pooling (1D): resizes to target output size.
pub fn adaptive_avg_pool1d(input: &[f64], output_size: usize) -> Vec<f64> {
    let n = input.len();
    (0..output_size).map(|i| {
        let start = (i * n) / output_size;
        let end = ((i + 1) * n) / output_size;
        let sum: f64 = input[start..end].iter().sum();
        sum / (end - start) as f64
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════
// NORMALIZATION
// ═══════════════════════════════════════════════════════════════════════

/// Batch normalization result.
#[derive(Debug, Clone)]
pub struct BatchNormResult {
    pub output: Vec<f64>,
    pub mean: Vec<f64>,
    pub var: Vec<f64>,
}

/// Batch normalization. Input: [batch * features], gamma/beta: [features].
/// Normalizes across batch dimension for each feature.
pub fn batch_norm(
    input: &[f64],
    batch_size: usize,
    features: usize,
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
) -> BatchNormResult {
    let mut mean = vec![0.0; features];
    let mut var = vec![0.0; features];
    let inv_n = 1.0 / batch_size as f64;

    // Compute mean per feature
    for b in 0..batch_size {
        for f in 0..features {
            mean[f] += input[b * features + f];
        }
    }
    for f in 0..features {
        mean[f] *= inv_n;
    }

    // Compute variance per feature
    for b in 0..batch_size {
        for f in 0..features {
            let d = input[b * features + f] - mean[f];
            var[f] += d * d;
        }
    }
    for f in 0..features {
        var[f] *= inv_n;
    }

    // Normalize + scale + shift
    let eps = eps.max(f64::MIN_POSITIVE);
    let mut output = vec![0.0; batch_size * features];
    for b in 0..batch_size {
        for f in 0..features {
            let x_hat = (input[b * features + f] - mean[f]) / (var[f] + eps).sqrt();
            output[b * features + f] = gamma[f] * x_hat + beta[f];
        }
    }

    BatchNormResult { output, mean, var }
}

/// Layer normalization. Normalizes across features for each sample.
/// Input: [batch * features], gamma/beta: [features].
pub fn layer_norm(
    input: &[f64],
    batch_size: usize,
    features: usize,
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
) -> Vec<f64> {
    let inv_d = 1.0 / features as f64;
    let mut output = vec![0.0; batch_size * features];

    for b in 0..batch_size {
        let base = b * features;
        let mean: f64 = input[base..base + features].iter().sum::<f64>() * inv_d;
        let var: f64 = input[base..base + features].iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>() * inv_d;
        let inv_std = 1.0 / (var + eps).sqrt();
        for f in 0..features {
            output[base + f] = gamma[f] * (input[base + f] - mean) * inv_std + beta[f];
        }
    }
    output
}

/// RMS normalization (used in LLaMA, etc). No centering — just scale by RMS.
/// Input: [batch * features], gamma: [features].
pub fn rms_norm(
    input: &[f64],
    batch_size: usize,
    features: usize,
    gamma: &[f64],
    eps: f64,
) -> Vec<f64> {
    let inv_d = 1.0 / features as f64;
    let mut output = vec![0.0; batch_size * features];

    for b in 0..batch_size {
        let base = b * features;
        let rms: f64 = (input[base..base + features].iter()
            .map(|&x| x * x)
            .sum::<f64>() * inv_d + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for f in 0..features {
            output[base + f] = gamma[f] * input[base + f] * inv_rms;
        }
    }
    output
}

/// Group normalization. Divides features into groups, normalizes within each group.
/// Input: [batch * channels * spatial], gamma/beta: [channels].
pub fn group_norm(
    input: &[f64],
    batch_size: usize,
    channels: usize,
    spatial: usize,
    num_groups: usize,
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
) -> Vec<f64> {
    assert_eq!(channels % num_groups, 0);
    let ch_per_group = channels / num_groups;
    let group_size = ch_per_group * spatial;
    let inv_gs = 1.0 / group_size as f64;
    let sample_size = channels * spatial;
    let mut output = vec![0.0; input.len()];

    for b in 0..batch_size {
        let sample_base = b * sample_size;
        for g in 0..num_groups {
            let ch_start = g * ch_per_group;
            // Compute mean and var over group
            let mut mean = 0.0;
            for c in ch_start..ch_start + ch_per_group {
                for s in 0..spatial {
                    mean += input[sample_base + c * spatial + s];
                }
            }
            mean *= inv_gs;

            let mut var = 0.0;
            for c in ch_start..ch_start + ch_per_group {
                for s in 0..spatial {
                    let d = input[sample_base + c * spatial + s] - mean;
                    var += d * d;
                }
            }
            var *= inv_gs;

            let inv_std = 1.0 / (var + eps).sqrt();
            for c in ch_start..ch_start + ch_per_group {
                for s in 0..spatial {
                    let idx = sample_base + c * spatial + s;
                    output[idx] = gamma[c] * (input[idx] - mean) * inv_std + beta[c];
                }
            }
        }
    }
    output
}

/// Instance normalization: group_norm with num_groups = channels.
pub fn instance_norm(
    input: &[f64],
    batch_size: usize,
    channels: usize,
    spatial: usize,
    gamma: &[f64],
    beta: &[f64],
    eps: f64,
) -> Vec<f64> {
    group_norm(input, batch_size, channels, spatial, channels, gamma, beta, eps)
}

// ═══════════════════════════════════════════════════════════════════════
// DROPOUT
// ═══════════════════════════════════════════════════════════════════════

/// Dropout with deterministic mask from TamRng.
/// Returns (output, mask). Mask is 1.0 where kept, 0.0 where dropped.
/// Scales kept values by 1/(1-p) to maintain expected value (inverted dropout).
pub fn dropout(input: &[f64], p: f64, rng: &mut dyn TamRng) -> (Vec<f64>, Vec<f64>) {
    if p <= 0.0 {
        return (input.to_vec(), vec![1.0; input.len()]);
    }
    if p >= 1.0 {
        return (vec![0.0; input.len()], vec![0.0; input.len()]);
    }
    let scale = 1.0 / (1.0 - p);
    let mut output = Vec::with_capacity(input.len());
    let mut mask = Vec::with_capacity(input.len());
    for &x in input {
        if rng.next_f64() >= p {
            mask.push(1.0);
            output.push(x * scale);
        } else {
            mask.push(0.0);
            output.push(0.0);
        }
    }
    (output, mask)
}

/// Apply a pre-computed dropout mask (for backward pass consistency).
pub fn dropout_backward(grad_out: &[f64], mask: &[f64], p: f64) -> Vec<f64> {
    let scale = if p < 1.0 { 1.0 / (1.0 - p) } else { 0.0 };
    grad_out.iter().zip(mask).map(|(&g, &m)| g * m * scale).collect()
}

// ═══════════════════════════════════════════════════════════════════════
// LINEAR / DENSE LAYER
// ═══════════════════════════════════════════════════════════════════════

/// Dense linear layer: output = input @ weight^T + bias.
/// input: [batch, in_features], weight: [out_features, in_features], bias: [out_features]
pub fn linear(input: &Mat, weight: &Mat, bias: Option<&[f64]>) -> Mat {
    let out = mat_mul(input, &weight.t());
    match bias {
        None => out,
        Some(b) => {
            let mut result = out;
            for row in 0..result.rows {
                for col in 0..result.cols {
                    result.data[row * result.cols + col] += b[col];
                }
            }
            result
        }
    }
}

/// Linear backward: given grad_output [batch, out_features],
/// returns (grad_input, grad_weight, grad_bias).
pub fn linear_backward(
    input: &Mat,
    weight: &Mat,
    grad_output: &Mat,
) -> (Mat, Mat, Vec<f64>) {
    // grad_input = grad_output @ weight
    let grad_input = mat_mul(grad_output, weight);
    // grad_weight = grad_output^T @ input
    let grad_weight = mat_mul(&grad_output.t(), input);
    // grad_bias = sum(grad_output, dim=0)
    let mut grad_bias = vec![0.0; weight.rows];
    for row in 0..grad_output.rows {
        for col in 0..grad_output.cols {
            grad_bias[col] += grad_output.data[row * grad_output.cols + col];
        }
    }
    (grad_input, grad_weight, grad_bias)
}

/// Bilinear: output[i] = x1^T @ W[i] @ x2 + bias[i]
/// x1: [batch, in1], x2: [batch, in2], weight: [out, in1, in2], bias: [out]
pub fn bilinear(
    x1: &Mat,
    x2: &Mat,
    weight: &[f64],
    out_features: usize,
    bias: Option<&[f64]>,
) -> Mat {
    let batch = x1.rows;
    let in1 = x1.cols;
    let in2 = x2.cols;
    let mut output = Mat::zeros(batch, out_features);

    for b in 0..batch {
        for o in 0..out_features {
            let w_base = o * in1 * in2;
            let mut val = 0.0;
            for i in 0..in1 {
                for j in 0..in2 {
                    val += x1.data[b * in1 + i] * weight[w_base + i * in2 + j] * x2.data[b * in2 + j];
                }
            }
            if let Some(bi) = bias {
                val += bi[o];
            }
            output.data[b * out_features + o] = val;
        }
    }
    output
}

// ═══════════════════════════════════════════════════════════════════════
// EMBEDDING & POSITIONAL ENCODING
// ═══════════════════════════════════════════════════════════════════════

/// Embedding lookup table. embedding_matrix: [vocab_size, embed_dim].
/// indices: token IDs. Returns [len(indices), embed_dim].
pub fn embedding(embedding_matrix: &Mat, indices: &[usize]) -> Mat {
    let dim = embedding_matrix.cols;
    let mut output = Mat::zeros(indices.len(), dim);
    for (i, &idx) in indices.iter().enumerate() {
        let src_base = idx * dim;
        let dst_base = i * dim;
        output.data[dst_base..dst_base + dim]
            .copy_from_slice(&embedding_matrix.data[src_base..src_base + dim]);
    }
    output
}

/// Sinusoidal positional encoding (Vaswani et al. 2017).
/// Returns [max_len, d_model] matrix.
pub fn positional_encoding(max_len: usize, d_model: usize) -> Mat {
    let mut pe = Mat::zeros(max_len, d_model);
    for pos in 0..max_len {
        for i in 0..d_model / 2 {
            let angle = pos as f64 / (10000.0_f64).powf(2.0 * i as f64 / d_model as f64);
            pe.data[pos * d_model + 2 * i] = angle.sin();
            pe.data[pos * d_model + 2 * i + 1] = angle.cos();
        }
        // Handle odd d_model
        if d_model % 2 == 1 {
            let i = d_model / 2;
            let angle = pos as f64 / (10000.0_f64).powf(2.0 * i as f64 / d_model as f64);
            pe.data[pos * d_model + 2 * i] = angle.sin();
        }
    }
    pe
}

/// Rotary positional embedding (RoPE). Applies rotation to pairs of dimensions.
/// input: [seq_len, d_model], applies rotation in-place-style, returns new matrix.
pub fn rope(input: &Mat, base: f64) -> Mat {
    let seq_len = input.rows;
    let d = input.cols;
    let base = base.max(1.0); // base must be positive; default 10000
    let mut output = Mat::zeros(seq_len, d);

    for pos in 0..seq_len {
        for i in 0..d / 2 {
            let theta = pos as f64 / base.powf(2.0 * i as f64 / d as f64);
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let x0 = input.data[pos * d + 2 * i];
            let x1 = input.data[pos * d + 2 * i + 1];
            output.data[pos * d + 2 * i] = x0 * cos_t - x1 * sin_t;
            output.data[pos * d + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
        }
        if d % 2 == 1 {
            output.data[pos * d + d - 1] = input.data[pos * d + d - 1];
        }
    }
    output
}

// ═══════════════════════════════════════════════════════════════════════
// ATTENTION
// ═══════════════════════════════════════════════════════════════════════

/// Attention result.
#[derive(Debug, Clone)]
pub struct AttentionResult {
    pub output: Mat,
    pub weights: Mat,
}

/// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V.
/// Q: [seq_q, d_k], K: [seq_k, d_k], V: [seq_k, d_v].
/// Optional causal mask zeros out future positions.
pub fn scaled_dot_product_attention(
    q: &Mat,
    k: &Mat,
    v: &Mat,
    causal: bool,
) -> AttentionResult {
    let d_k = q.cols as f64;
    let scale = 1.0 / d_k.sqrt();

    // scores = Q @ K^T * scale
    let kt = k.t();
    let mut scores = mat_mul(q, &kt);
    for val in scores.data.iter_mut() {
        *val *= scale;
    }

    // Causal mask: set future positions to -inf
    if causal {
        for i in 0..scores.rows {
            for j in (i + 1)..scores.cols {
                scores.data[i * scores.cols + j] = f64::NEG_INFINITY;
            }
        }
    }

    // Row-wise softmax
    let mut weights = Mat::zeros(scores.rows, scores.cols);
    for i in 0..scores.rows {
        let row_start = i * scores.cols;
        let row = &scores.data[row_start..row_start + scores.cols];
        let sm = softmax(row);
        weights.data[row_start..row_start + scores.cols].copy_from_slice(&sm);
    }

    // output = weights @ V
    let output = mat_mul(&weights, v);

    AttentionResult { output, weights }
}

/// Multi-head attention. Splits Q, K, V into heads, applies attention, concatenates.
/// q/k/v: [seq, d_model]. w_q/w_k/w_v: [d_model, d_model]. w_o: [d_model, d_model].
pub fn multi_head_attention(
    q: &Mat,
    k: &Mat,
    v: &Mat,
    w_q: &Mat,
    w_k: &Mat,
    w_v: &Mat,
    w_o: &Mat,
    num_heads: usize,
    causal: bool,
) -> AttentionResult {
    let d_model = q.cols;
    let d_head = d_model / num_heads;
    let seq_q = q.rows;
    let seq_k = k.rows;

    // Project
    let q_proj = mat_mul(q, &w_q.t());
    let k_proj = mat_mul(k, &w_k.t());
    let v_proj = mat_mul(v, &w_v.t());

    // Allocate output
    let mut concat = Mat::zeros(seq_q, d_model);
    let mut all_weights = Mat::zeros(seq_q, seq_k); // average weights across heads

    for h in 0..num_heads {
        // Extract head h: columns [h*d_head .. (h+1)*d_head]
        let mut q_h = Mat::zeros(seq_q, d_head);
        let mut k_h = Mat::zeros(seq_k, d_head);
        let mut v_h = Mat::zeros(seq_k, d_head);

        for i in 0..seq_q {
            for j in 0..d_head {
                q_h.data[i * d_head + j] = q_proj.data[i * d_model + h * d_head + j];
            }
        }
        for i in 0..seq_k {
            for j in 0..d_head {
                k_h.data[i * d_head + j] = k_proj.data[i * d_model + h * d_head + j];
                v_h.data[i * d_head + j] = v_proj.data[i * d_model + h * d_head + j];
            }
        }

        let attn = scaled_dot_product_attention(&q_h, &k_h, &v_h, causal);

        // Place head output into concat
        for i in 0..seq_q {
            for j in 0..d_head {
                concat.data[i * d_model + h * d_head + j] = attn.output.data[i * d_head + j];
            }
        }

        // Accumulate weights (for diagnostics)
        let inv_h = 1.0 / num_heads as f64;
        for i in 0..seq_q {
            for j in 0..seq_k {
                all_weights.data[i * seq_k + j] += attn.weights.data[i * seq_k + j] * inv_h;
            }
        }
    }

    // Output projection
    let output = mat_mul(&concat, &w_o.t());

    AttentionResult { output, weights: all_weights }
}

// ═══════════════════════════════════════════════════════════════════════
// LOSS FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════

/// Mean squared error loss.
pub fn mse_loss(predicted: &[f64], target: &[f64]) -> f64 {
    let n = predicted.len() as f64;
    predicted.iter().zip(target).map(|(&p, &t)| (p - t) * (p - t)).sum::<f64>() / n
}

/// MSE gradient: 2(predicted - target) / n
pub fn mse_loss_backward(predicted: &[f64], target: &[f64]) -> Vec<f64> {
    let inv_n = 2.0 / predicted.len() as f64;
    predicted.iter().zip(target).map(|(&p, &t)| (p - t) * inv_n).collect()
}

/// Binary cross-entropy loss. predicted = sigmoid probabilities in (0, 1).
pub fn bce_loss(predicted: &[f64], target: &[f64]) -> f64 {
    let n = predicted.len() as f64;
    let eps = 1e-12;
    -predicted.iter().zip(target).map(|(&p, &t)| {
        let p = p.clamp(eps, 1.0 - eps);
        t * p.ln() + (1.0 - t) * (1.0 - p).ln()
    }).sum::<f64>() / n
}

/// BCE gradient.
pub fn bce_loss_backward(predicted: &[f64], target: &[f64]) -> Vec<f64> {
    let inv_n = 1.0 / predicted.len() as f64;
    let eps = 1e-12;
    predicted.iter().zip(target).map(|(&p, &t)| {
        let p = p.clamp(eps, 1.0 - eps);
        (-t / p + (1.0 - t) / (1.0 - p)) * inv_n
    }).collect()
}

/// Cross-entropy loss from logits. target: class indices (as f64).
/// Applies log-softmax internally for numerical stability.
pub fn cross_entropy_loss(logits: &[f64], num_classes: usize, targets: &[usize]) -> f64 {
    let batch_size = targets.len();
    let mut total = 0.0;
    for b in 0..batch_size {
        let row = &logits[b * num_classes..(b + 1) * num_classes];
        let lsm = log_softmax(row);
        total -= lsm[targets[b]];
    }
    total / batch_size as f64
}

/// Cross-entropy gradient w.r.t. logits.
pub fn cross_entropy_loss_backward(logits: &[f64], num_classes: usize, targets: &[usize]) -> Vec<f64> {
    let batch_size = targets.len();
    let inv_n = 1.0 / batch_size as f64;
    let mut grad = vec![0.0; logits.len()];
    for b in 0..batch_size {
        let row = &logits[b * num_classes..(b + 1) * num_classes];
        let sm = softmax(row);
        for c in 0..num_classes {
            let target_val = if c == targets[b] { 1.0 } else { 0.0 };
            grad[b * num_classes + c] = (sm[c] - target_val) * inv_n;
        }
    }
    grad
}

/// Huber loss (smooth L1). Delta controls the transition point.
pub fn huber_loss(predicted: &[f64], target: &[f64], delta: f64) -> f64 {
    let n = predicted.len() as f64;
    predicted.iter().zip(target).map(|(&p, &t)| {
        let d = (p - t).abs();
        if d <= delta {
            0.5 * d * d
        } else {
            delta * (d - 0.5 * delta)
        }
    }).sum::<f64>() / n
}

/// Huber loss gradient.
pub fn huber_loss_backward(predicted: &[f64], target: &[f64], delta: f64) -> Vec<f64> {
    let inv_n = 1.0 / predicted.len() as f64;
    predicted.iter().zip(target).map(|(&p, &t)| {
        let d = p - t;
        if d.abs() <= delta {
            d * inv_n
        } else {
            delta * d.signum() * inv_n
        }
    }).collect()
}

/// Cosine similarity loss: 1 - cos_sim(a, b).
pub fn cosine_similarity_loss(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let denom = na * nb;
    if denom < 1e-12 { return 1.0; }
    1.0 - dot / denom
}

/// Hinge loss: max(0, 1 - y * pred). y ∈ {-1, +1}.
pub fn hinge_loss(predicted: &[f64], target: &[f64]) -> f64 {
    let n = predicted.len() as f64;
    predicted.iter().zip(target).map(|(&p, &t)| (1.0 - t * p).max(0.0)).sum::<f64>() / n
}

/// Focal loss: -alpha * (1-p)^gamma * log(p) for true class.
/// Useful for class imbalance. predicted: softmax probabilities.
pub fn focal_loss(predicted: &[f64], target: &[f64], alpha: f64, gamma: f64) -> f64 {
    let n = predicted.len() as f64;
    let eps = 1e-12;
    predicted.iter().zip(target).map(|(&p, &t)| {
        let p = p.clamp(eps, 1.0 - eps);
        let pt = if t > 0.5 { p } else { 1.0 - p };
        -alpha * (1.0 - pt).powf(gamma) * pt.ln()
    }).sum::<f64>() / n
}

// ═══════════════════════════════════════════════════════════════════════
// UTILITY OPS
// ═══════════════════════════════════════════════════════════════════════

/// Flatten: reshapes [batch, c, h, w, ...] to [batch, c*h*w*...].
/// This is a conceptual reshape — just returns the data unchanged since
/// our storage is already flat. Returns (data_ref, new_shape).
pub fn flatten_shape(batch: usize, dims: &[usize]) -> (usize, usize) {
    let flat: usize = dims.iter().product();
    (batch, flat)
}

/// Residual connection: output = x + f(x).
/// Takes the input and the layer output, returns their sum.
pub fn residual_add(x: &[f64], fx: &[f64]) -> Vec<f64> {
    x.iter().zip(fx).map(|(&a, &b)| a + b).collect()
}

/// Gradient clipping by norm. If ||grad|| > max_norm, scale down.
pub fn clip_grad_norm(grad: &mut [f64], max_norm: f64) -> f64 {
    let norm: f64 = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
    norm
}

/// Gradient clipping by value. Clamps each element.
pub fn clip_grad_value(grad: &mut [f64], clip_value: f64) {
    for g in grad.iter_mut() {
        *g = g.clamp(-clip_value, clip_value);
    }
}

/// Label smoothing: converts hard target to (1-ε)*one_hot + ε/num_classes.
pub fn label_smooth(targets: &[usize], num_classes: usize, epsilon: f64) -> Vec<f64> {
    let smooth = epsilon / num_classes as f64;
    let confident = 1.0 - epsilon + smooth;
    let mut output = vec![smooth; targets.len() * num_classes];
    for (i, &t) in targets.iter().enumerate() {
        output[i * num_classes + t] = confident;
    }
    output
}

/// Temperature scaling for logits.
pub fn temperature_scale(logits: &[f64], temperature: f64) -> Vec<f64> {
    let inv_t = 1.0 / temperature.max(f64::MIN_POSITIVE);
    logits.iter().map(|&x| x * inv_t).collect()
}

/// Top-k sampling: zeros out all but top k logits, returns modified logits.
pub fn top_k_logits(logits: &[f64], k: usize) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    let threshold = if k < indexed.len() { indexed[k].1 } else { f64::NEG_INFINITY };
    logits.iter().map(|&x| if x >= threshold { x } else { f64::NEG_INFINITY }).collect()
}

/// Top-p (nucleus) sampling: zeros out logits outside the top-p probability mass.
pub fn top_p_logits(logits: &[f64], p: f64) -> Vec<f64> {
    let probs = softmax(logits);
    let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut cumsum = 0.0;
    let mut cutoff = indexed.len();
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff = i + 1;
            break;
        }
    }

    let threshold = if cutoff < indexed.len() { indexed[cutoff].1 } else { 0.0 };
    logits.iter().zip(probs.iter()).map(|(&l, &pr)| {
        if pr >= threshold { l } else { f64::NEG_INFINITY }
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xoshiro256;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── Activation tests ────────────────────────────────────────────

    #[test]
    fn test_relu() {
        assert_eq!(relu(3.0), 3.0);
        assert_eq!(relu(-2.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
        let v = relu_vec(&[-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(v, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_backward() {
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        let g = vec![1.0, 1.0, 1.0, 1.0];
        let grad = relu_backward(&x, &g);
        assert_eq!(grad, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_leaky_relu() {
        assert_eq!(leaky_relu(1.0, 0.01), 1.0);
        assert!(approx_eq(leaky_relu(-1.0, 0.01), -0.01, 1e-10));
    }

    #[test]
    fn test_elu() {
        assert_eq!(elu(1.0, 1.0), 1.0);
        assert!(approx_eq(elu(-1.0, 1.0), -1.0 + 1.0_f64.exp().recip(), 1e-6));
    }

    #[test]
    fn test_selu_constants() {
        // SELU should preserve mean 0, var 1 for standard normal input
        assert!(SELU_ALPHA > 1.6);
        assert!(SELU_LAMBDA > 1.05);
        assert!(approx_eq(selu(0.0), 0.0, 1e-10));
        assert!(selu(1.0) > 1.0); // lambda > 1
    }

    #[test]
    fn test_gelu() {
        // GELU(0) = 0
        assert!(approx_eq(gelu(0.0), 0.0, 1e-10));
        // GELU is asymptotically identity for large positive
        assert!(approx_eq(gelu(5.0), 5.0, 0.01));
        // GELU is asymptotically 0 for large negative
        assert!(approx_eq(gelu(-5.0), 0.0, 0.01));
    }

    #[test]
    fn test_gelu_backward() {
        // Numerical gradient check
        let x = vec![0.0, 1.0, -1.0];
        let g = vec![1.0, 1.0, 1.0];
        let grad = gelu_backward(&x, &g);
        let eps = 1e-5;
        for i in 0..x.len() {
            let numerical = (gelu(x[i] + eps) - gelu(x[i] - eps)) / (2.0 * eps);
            assert!(approx_eq(grad[i], numerical, 1e-4), "GELU grad mismatch at x={}: got {}, expected {}", x[i], grad[i], numerical);
        }
    }

    #[test]
    fn test_swish() {
        assert!(approx_eq(swish(0.0), 0.0, 1e-10));
        // swish(x) → x for large x
        assert!(approx_eq(swish(10.0), 10.0, 0.01));
    }

    #[test]
    fn test_mish() {
        assert!(approx_eq(mish(0.0), 0.0, 1e-10));
        // Mish is smooth and non-monotonic near 0
        assert!(mish(-0.5) < 0.0);
    }

    #[test]
    fn test_sigmoid() {
        assert!(approx_eq(sigmoid(0.0), 0.5, 1e-10));
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
        // Numerically stable for extreme values
        assert!(!sigmoid(1000.0).is_nan());
        assert!(!sigmoid(-1000.0).is_nan());
    }

    #[test]
    fn test_sigmoid_backward() {
        let x = vec![0.0];
        let g = vec![1.0];
        let grad = sigmoid_backward(&x, &g);
        assert!(approx_eq(grad[0], 0.25, 1e-10)); // sigmoid'(0) = 0.25
    }

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let sm = softmax(&x);
        // Sum to 1
        assert!(approx_eq(sm.iter().sum::<f64>(), 1.0, 1e-10));
        // Monotonic
        assert!(sm[0] < sm[1] && sm[1] < sm[2]);
        // Known values
        let e1 = 1.0_f64.exp();
        let e2 = 2.0_f64.exp();
        let e3 = 3.0_f64.exp();
        let total = e1 + e2 + e3;
        assert!(approx_eq(sm[2], e3 / total, 1e-10));
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values shouldn't overflow
        let x = vec![1000.0, 1001.0, 1002.0];
        let sm = softmax(&x);
        assert!(approx_eq(sm.iter().sum::<f64>(), 1.0, 1e-10));
        assert!(!sm.iter().any(|&v| v.is_nan()));
    }

    #[test]
    fn test_softmax_nan_propagates() {
        // NaN input must propagate through; softmax must not silently swallow it.
        let x = vec![1.0, f64::NAN, 2.0];
        let sm = softmax(&x);
        assert!(sm.iter().any(|v| v.is_nan()),
            "softmax should propagate NaN input, got {:?}", sm);
    }

    #[test]
    fn test_log_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let lsm = log_softmax(&x);
        let sm = softmax(&x);
        for i in 0..3 {
            assert!(approx_eq(lsm[i], sm[i].ln(), 1e-10));
        }
    }

    #[test]
    fn test_log_softmax_nan_propagates() {
        let x = vec![1.0, f64::NAN, 2.0];
        let lsm = log_softmax(&x);
        assert!(lsm.iter().any(|v| v.is_nan()),
            "log_softmax should propagate NaN input, got {:?}", lsm);
    }

    #[test]
    fn test_softmax_backward() {
        let x = vec![1.0, 2.0, 3.0];
        let sm = softmax(&x);
        let g = vec![1.0, 0.0, 0.0];
        let grad = softmax_backward(&sm, &g);
        // Sum of softmax gradient should be 0 (output sums to constant 1)
        assert!(approx_eq(grad.iter().sum::<f64>(), 0.0, 1e-10));
    }

    #[test]
    fn test_softplus() {
        assert!(approx_eq(softplus(0.0), 2.0_f64.ln(), 1e-10));
        // For large x, softplus(x) ≈ x
        assert!(approx_eq(softplus(50.0), 50.0, 1e-10));
        // For very negative x, softplus(x) ≈ 0
        assert!(approx_eq(softplus(-50.0), 0.0, 1e-10));
    }

    #[test]
    fn test_hard_swish() {
        assert!(approx_eq(hard_swish(0.0), 0.0, 1e-10));
        // For x >= 3, hard_swish(x) = x
        assert!(approx_eq(hard_swish(5.0), 5.0, 1e-10));
        // For x <= -3, hard_swish(x) = 0
        assert!(approx_eq(hard_swish(-5.0), 0.0, 1e-10));
    }

    // ── Convolution tests ───────────────────────────────────────────

    #[test]
    fn test_conv1d_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 0.0, -1.0];
        let out = conv1d(&input, &kernel, 1, 0);
        // [1*1+2*0+3*(-1), 2*1+3*0+4*(-1), 3*1+4*0+5*(-1)]
        assert_eq!(out, vec![-2.0, -2.0, -2.0]);
    }

    #[test]
    fn test_conv1d_with_padding() {
        let input = vec![1.0, 2.0, 3.0];
        let kernel = vec![1.0, 1.0, 1.0];
        let out = conv1d(&input, &kernel, 1, 1);
        // padded: [0, 1, 2, 3, 0]
        // [0+1+2, 1+2+3, 2+3+0]
        assert_eq!(out, vec![3.0, 6.0, 5.0]);
    }

    #[test]
    fn test_conv1d_stride() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 1.0];
        let out = conv1d(&input, &kernel, 2, 0);
        // stride 2: [1+2, 3+4]
        assert_eq!(out, vec![3.0, 7.0]);
    }

    #[test]
    fn test_conv2d_identity() {
        // 1x1 identity kernel on 1-channel 3x3 input
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let kernel = vec![1.0]; // 1x1x1x1
        let out = conv2d(&input, 1, 3, 3, &kernel, 1, 1, 1, 1, 0);
        assert_eq!(out, input);
    }

    #[test]
    fn test_conv2d_edge_detect() {
        // Laplacian kernel on 3x3 input
        let input = vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let kernel = vec![
            0.0, -1.0, 0.0,
            -1.0, 4.0, -1.0,
            0.0, -1.0, 0.0,
        ];
        let out = conv2d(&input, 1, 3, 3, &kernel, 1, 3, 3, 1, 0);
        // Single output: 0*0+(-1)*0+0*0+(-1)*0+4*1+(-1)*0+0*0+(-1)*0+0*0 = 4
        assert_eq!(out.len(), 1);
        assert!(approx_eq(out[0], 4.0, 1e-10));
    }

    #[test]
    fn test_conv2d_multi_channel() {
        // 2 input channels, 1 output channel, 1x1 kernel
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0: 2x2
            5.0, 6.0, 7.0, 8.0, // channel 1: 2x2
        ];
        let kernel = vec![1.0, 2.0]; // [1, 2, 1, 1] = oc=1, ic=2, kh=1, kw=1
        let out = conv2d(&input, 2, 2, 2, &kernel, 1, 1, 1, 1, 0);
        // out[i] = 1*ch0[i] + 2*ch1[i]
        assert_eq!(out, vec![11.0, 14.0, 17.0, 20.0]);
    }

    // ── Pooling tests ───────────────────────────────────────────────

    #[test]
    fn test_max_pool1d() {
        let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let (vals, idxs) = max_pool1d(&input, 2, 1);
        assert_eq!(vals, vec![3.0, 3.0, 5.0, 5.0]);
        assert_eq!(idxs, vec![1, 1, 3, 3]);
    }

    #[test]
    fn test_avg_pool1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = avg_pool1d(&input, 2, 2);
        assert_eq!(out, vec![1.5, 3.5]);
    }

    #[test]
    fn test_max_pool2d() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let (vals, _) = max_pool2d(&input, 1, 4, 4, 2, 2, 2);
        assert_eq!(vals, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_global_avg_pool2d() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0: 2x2
            5.0, 6.0, 7.0, 8.0, // channel 1: 2x2
        ];
        let out = global_avg_pool2d(&input, 2, 2, 2);
        assert!(approx_eq(out[0], 2.5, 1e-10));
        assert!(approx_eq(out[1], 6.5, 1e-10));
    }

    #[test]
    fn test_adaptive_avg_pool1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = adaptive_avg_pool1d(&input, 3);
        // Bins: [0..2), [2..4), [4..6) → means of (1,2), (3,4), (5,6)
        assert!(approx_eq(out[0], 1.5, 1e-10));
        assert!(approx_eq(out[1], 3.5, 1e-10));
        assert!(approx_eq(out[2], 5.5, 1e-10));
    }

    // ── Normalization tests ─────────────────────────────────────────

    #[test]
    fn test_batch_norm() {
        // 2 samples, 2 features
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let result = batch_norm(&input, 2, 2, &gamma, &beta, 1e-5);
        // mean = [2, 3], var = [1, 1]
        // (1-2)/1 = -1, (2-3)/1 = -1, (3-2)/1 = 1, (4-3)/1 = 1
        assert!(approx_eq(result.output[0], -1.0, 0.01));
        assert!(approx_eq(result.output[1], -1.0, 0.01));
        assert!(approx_eq(result.output[2], 1.0, 0.01));
        assert!(approx_eq(result.output[3], 1.0, 0.01));
    }

    #[test]
    fn test_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0];
        let beta = vec![0.0, 0.0];
        let out = layer_norm(&input, 2, 2, &gamma, &beta, 1e-5);
        // Each row normalized: (1-1.5)/0.5=-1, (2-1.5)/0.5=1, (3-3.5)/0.5=-1, (4-3.5)/0.5=1
        assert!(approx_eq(out[0], -1.0, 0.01));
        assert!(approx_eq(out[1], 1.0, 0.01));
        assert!(approx_eq(out[2], -1.0, 0.01));
        assert!(approx_eq(out[3], 1.0, 0.01));
    }

    #[test]
    fn test_rms_norm() {
        let input = vec![3.0, 4.0]; // RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        let gamma = vec![1.0, 1.0];
        let out = rms_norm(&input, 1, 2, &gamma, 1e-5);
        let rms = (12.5_f64).sqrt();
        assert!(approx_eq(out[0], 3.0 / rms, 1e-6));
        assert!(approx_eq(out[1], 4.0 / rms, 1e-6));
    }

    #[test]
    fn test_group_norm() {
        // 1 sample, 4 channels, 1 spatial, 2 groups
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let out = group_norm(&input, 1, 4, 1, 2, &gamma, &beta, 1e-5);
        // Group 0: channels [0,1] = [1,2], mean=1.5, var=0.25
        // Group 1: channels [2,3] = [3,4], mean=3.5, var=0.25
        assert!(approx_eq(out[0], -1.0, 0.01));
        assert!(approx_eq(out[1], 1.0, 0.01));
        assert!(approx_eq(out[2], -1.0, 0.01));
        assert!(approx_eq(out[3], 1.0, 0.01));
    }

    // ── Dropout test ────────────────────────────────────────────────

    #[test]
    fn test_dropout() {
        let mut rng = Xoshiro256::new(42);
        let input = vec![1.0; 1000];
        let (out, mask) = dropout(&input, 0.5, &mut rng);

        // Roughly half should be dropped
        let kept: usize = mask.iter().filter(|&&m| m > 0.0).count();
        assert!(kept > 300 && kept < 700, "Expected ~500 kept, got {}", kept);

        // Kept values should be scaled by 2 (1/(1-0.5))
        for (i, &m) in mask.iter().enumerate() {
            if m > 0.0 {
                assert!(approx_eq(out[i], 2.0, 1e-10));
            } else {
                assert!(approx_eq(out[i], 0.0, 1e-10));
            }
        }

        // Expected value should be preserved
        let mean_out = out.iter().sum::<f64>() / out.len() as f64;
        assert!(approx_eq(mean_out, 1.0, 0.1));
    }

    // ── Linear layer test ───────────────────────────────────────────

    #[test]
    fn test_linear_forward() {
        let input = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let weight = Mat::from_vec(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let bias = vec![0.1, 0.2];
        let out = linear(&input, &weight, Some(&bias));
        // Row 0: [1*1+2*0+3*0+0.1, 1*0+2*1+3*0+0.2] = [1.1, 2.2]
        // Row 1: [4*1+5*0+6*0+0.1, 4*0+5*1+6*0+0.2] = [4.1, 5.2]
        assert!(approx_eq(out.data[0], 1.1, 1e-10));
        assert!(approx_eq(out.data[1], 2.2, 1e-10));
        assert!(approx_eq(out.data[2], 4.1, 1e-10));
        assert!(approx_eq(out.data[3], 5.2, 1e-10));
    }

    #[test]
    fn test_linear_backward() {
        let input = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let weight = Mat::from_vec(2, 3, vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0]);
        let grad_out = Mat::from_vec(1, 2, vec![1.0, 1.0]);
        let (gi, _gw, gb) = linear_backward(&input, &weight, &grad_out);
        // grad_input = [1,1] @ [[1,0,-1],[0,1,0]] = [1, 1, -1]
        assert!(approx_eq(gi.data[0], 1.0, 1e-10));
        assert!(approx_eq(gi.data[1], 1.0, 1e-10));
        assert!(approx_eq(gi.data[2], -1.0, 1e-10));
        // grad_bias = [1, 1]
        assert_eq!(gb, vec![1.0, 1.0]);
    }

    // ── Embedding tests ─────────────────────────────────────────────

    #[test]
    fn test_embedding_lookup() {
        let embed = Mat::from_vec(4, 3, vec![
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6,
            0.7, 0.8, 0.9,
            1.0, 1.1, 1.2,
        ]);
        let out = embedding(&embed, &[2, 0, 3]);
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 3);
        assert!(approx_eq(out.data[0], 0.7, 1e-10)); // token 2
        assert!(approx_eq(out.data[3], 0.1, 1e-10)); // token 0
        assert!(approx_eq(out.data[6], 1.0, 1e-10)); // token 3
    }

    #[test]
    fn test_positional_encoding() {
        let pe = positional_encoding(10, 8);
        assert_eq!(pe.rows, 10);
        assert_eq!(pe.cols, 8);
        // Position 0 should be [sin(0), cos(0), ...] = [0, 1, 0, 1, ...]
        assert!(approx_eq(pe.data[0], 0.0, 1e-10)); // sin(0)
        assert!(approx_eq(pe.data[1], 1.0, 1e-10)); // cos(0)
    }

    #[test]
    fn test_rope() {
        let input = Mat::from_vec(2, 4, vec![
            1.0, 0.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
        ]);
        let out = rope(&input, 10000.0);
        // Position 0: theta=0 for all dims, so cos(0)=1, sin(0)=0, output = input
        assert!(approx_eq(out.data[0], 1.0, 1e-10));
        assert!(approx_eq(out.data[1], 0.0, 1e-10));
        // Position 1: some rotation applied
        assert!(out.data[4] != 1.0); // rotated
    }

    // ── Attention tests ─────────────────────────────────────────────

    #[test]
    fn test_scaled_dot_product_attention() {
        let q = Mat::from_vec(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let k = Mat::from_vec(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let v = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let result = scaled_dot_product_attention(&q, &k, &v, false);
        // Q[0] attends more to K[0], Q[1] attends more to K[1]
        // Both attention weights should sum to 1
        assert!(approx_eq(result.weights.data[0] + result.weights.data[1], 1.0, 1e-10));
        assert!(result.weights.data[0] > result.weights.data[1]); // Q[0] aligns with K[0]
    }

    #[test]
    fn test_causal_attention() {
        let q = Mat::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let k = Mat::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let v = Mat::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);

        let result = scaled_dot_product_attention(&q, &k, &v, true);
        // Position 0 can only attend to position 0
        assert!(approx_eq(result.weights.data[0], 1.0, 1e-10)); // w[0][0] = 1
        assert!(approx_eq(result.weights.data[1], 0.0, 1e-10)); // w[0][1] = 0
        assert!(approx_eq(result.weights.data[2], 0.0, 1e-10)); // w[0][2] = 0
    }

    #[test]
    fn test_multi_head_attention() {
        let d_model = 4;
        let num_heads = 2;

        let q = Mat::from_vec(2, d_model, vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        let k = q.clone();
        let v = q.clone();

        // Identity projections
        let w_q = Mat::eye(d_model);
        let w_k = Mat::eye(d_model);
        let w_v = Mat::eye(d_model);
        let w_o = Mat::eye(d_model);

        let result = multi_head_attention(&q, &k, &v, &w_q, &w_k, &w_v, &w_o, num_heads, false);
        assert_eq!(result.output.rows, 2);
        assert_eq!(result.output.cols, d_model);
    }

    // ── Loss function tests ─────────────────────────────────────────

    #[test]
    fn test_mse_loss() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(mse_loss(&pred, &target), 0.0, 1e-10));

        let pred2 = vec![1.0, 2.0, 3.0];
        let target2 = vec![2.0, 3.0, 4.0];
        assert!(approx_eq(mse_loss(&pred2, &target2), 1.0, 1e-10));
    }

    #[test]
    fn test_bce_loss() {
        // Perfect prediction
        let pred = vec![0.99, 0.01];
        let target = vec![1.0, 0.0];
        let loss = bce_loss(&pred, &target);
        assert!(loss < 0.05);

        // Bad prediction
        let pred2 = vec![0.01, 0.99];
        let loss2 = bce_loss(&pred2, &target);
        assert!(loss2 > 2.0);
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Perfect logit
        let logits = vec![10.0, -10.0, -10.0, -10.0, 10.0, -10.0];
        let targets = vec![0usize, 1];
        let loss = cross_entropy_loss(&logits, 3, &targets);
        assert!(loss < 0.001);
    }

    #[test]
    fn test_cross_entropy_backward() {
        let logits = vec![2.0, 1.0, 0.0];
        let targets = vec![0usize];
        let grad = cross_entropy_loss_backward(&logits, 3, &targets);
        // grad = softmax - one_hot
        let sm = softmax(&logits);
        assert!(approx_eq(grad[0], sm[0] - 1.0, 1e-10));
        assert!(approx_eq(grad[1], sm[1], 1e-10));
        assert!(approx_eq(grad[2], sm[2], 1e-10));
    }

    #[test]
    fn test_huber_loss() {
        let pred = vec![0.0];
        let target = vec![0.5];
        // |d| = 0.5 <= 1.0, so loss = 0.5 * 0.25 = 0.125
        assert!(approx_eq(huber_loss(&pred, &target, 1.0), 0.125, 1e-10));

        // |d| = 2.0 > 1.0, so loss = 1.0 * (2.0 - 0.5) = 1.5
        let pred2 = vec![0.0];
        let target2 = vec![2.0];
        assert!(approx_eq(huber_loss(&pred2, &target2, 1.0), 1.5, 1e-10));
    }

    #[test]
    fn test_focal_loss() {
        // High confidence correct prediction → very low loss
        let pred = vec![0.99];
        let target = vec![1.0];
        let loss = focal_loss(&pred, &target, 1.0, 2.0);
        assert!(loss < 0.0001);
    }

    #[test]
    fn test_hinge_loss() {
        // Correct with margin
        let pred = vec![2.0];
        let target = vec![1.0];
        assert!(approx_eq(hinge_loss(&pred, &target), 0.0, 1e-10));

        // Wrong
        let pred2 = vec![-0.5];
        assert!(approx_eq(hinge_loss(&pred2, &target), 1.5, 1e-10));
    }

    // ── Utility tests ───────────────────────────────────────────────

    #[test]
    fn test_clip_grad_norm() {
        let mut grad = vec![3.0, 4.0]; // norm = 5
        let norm = clip_grad_norm(&mut grad, 2.5);
        assert!(approx_eq(norm, 5.0, 1e-10));
        let new_norm: f64 = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
        assert!(approx_eq(new_norm, 2.5, 1e-10));
    }

    #[test]
    fn test_label_smoothing() {
        let targets = vec![0usize, 2];
        let smooth = label_smooth(&targets, 3, 0.1);
        // [0.9333, 0.0333, 0.0333, 0.0333, 0.0333, 0.9333]
        assert!(approx_eq(smooth[0], 0.9 + 0.1 / 3.0, 1e-10));
        assert!(approx_eq(smooth[1], 0.1 / 3.0, 1e-10));
    }

    #[test]
    fn test_top_k_logits() {
        let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let out = top_k_logits(&logits, 2);
        // Top 2: indices 1 (5.0) and 4 (4.0)
        assert!(out[1] == 5.0);
        assert!(out[4] == 4.0);
        assert!(out[0] == f64::NEG_INFINITY);
    }

    #[test]
    fn test_top_p_logits() {
        let logits = vec![10.0, 1.0, 0.0, -10.0];
        let out = top_p_logits(&logits, 0.9);
        // Most mass is on index 0 (exp(10) dominates)
        assert!(out[0] == 10.0); // kept
    }

    #[test]
    fn test_residual_add() {
        let x = vec![1.0, 2.0, 3.0];
        let fx = vec![0.1, 0.2, 0.3];
        let out = residual_add(&x, &fx);
        assert_eq!(out, vec![1.1, 2.2, 3.3]);
    }

    #[test]
    fn test_temperature_scaling() {
        let logits = vec![2.0, 4.0];
        let scaled = temperature_scale(&logits, 2.0);
        assert_eq!(scaled, vec![1.0, 2.0]);
    }

    #[test]
    fn test_conv2d_transpose_upsamples() {
        // 1-channel 2x2 input, 1x1x1x1 kernel, stride 2 → should produce ~3x3
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![1.0];
        let out = conv2d_transpose(&input, 1, 2, 2, &kernel, 1, 1, 1, 2, 0);
        // Upsampled to 3x3 with zeros, then conv with identity → sparse output
        assert_eq!(out.len(), 9);
        assert!(approx_eq(out[0], 1.0, 1e-10));
        assert!(approx_eq(out[2], 2.0, 1e-10));
    }

    #[test]
    fn test_cosine_similarity_loss() {
        // Same direction
        let a = vec![1.0, 0.0];
        let b = vec![2.0, 0.0];
        assert!(approx_eq(cosine_similarity_loss(&a, &b), 0.0, 1e-10));

        // Orthogonal
        let c = vec![0.0, 1.0];
        assert!(approx_eq(cosine_similarity_loss(&a, &c), 1.0, 1e-10));
    }
}
