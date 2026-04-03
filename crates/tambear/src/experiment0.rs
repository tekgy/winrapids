//! # Experiment 0: Can Tambear Learn Language?
//!
//! Character-level neural net with context window, trained via gradient duality.
//! Two linear layers + ReLU. Input = concatenated one-hots of last `ctx` characters.
//!
//! ```text
//! Input:  concat(one_hot(byte[-ctx]), ..., one_hot(byte[-1]))  → 256*ctx dims
//! Hidden: W1 (256*ctx × H) + ReLU
//! Output: W2 (H × 256) + softmax → next character probabilities
//! ```
//!
//! "Tam doesn't train forward then backward. Tam accumulates."

use std::sync::Arc;
use std::time::Instant;
use winrapids_tiled::{TiledEngine, DotProductOp};

pub struct CharModel {
    w1: Vec<f64>,      // input_dim × hidden
    w2: Vec<f64>,      // hidden × 256
    hidden: usize,
    vocab: usize,      // 256
    ctx: usize,        // context window size
    input_dim: usize,  // vocab * ctx
}

impl CharModel {
    pub fn new(hidden: usize, ctx: usize) -> Self {
        let vocab = 256;
        let input_dim = vocab * ctx;
        let mut w1 = vec![0.0f64; input_dim * hidden];
        let mut w2 = vec![0.0f64; hidden * vocab];

        // Xavier initialization with deterministic LCG, symmetric [-scale, +scale]
        let scale1 = (2.0 / (input_dim + hidden) as f64).sqrt();
        let scale2 = (2.0 / (hidden + vocab) as f64).sqrt();
        let mut rng: u64 = 42;
        for w in w1.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng >> 33) as f64 / (1u64 << 30) as f64 - 1.0;
            *w = u * scale1;
        }
        for w in w2.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng >> 33) as f64 / (1u64 << 30) as f64 - 1.0;
            *w = u * scale2;
        }

        Self { w1, w2, hidden, vocab, ctx, input_dim }
    }

    /// Encode a context window into concatenated one-hot vector.
    fn encode_context(&self, text: &[u8], pos: usize) -> Vec<f64> {
        let mut x = vec![0.0f64; self.input_dim];
        for c in 0..self.ctx {
            if pos >= self.ctx - c {
                let byte = text[pos - (self.ctx - c)] as usize;
                x[c * self.vocab + byte] = 1.0;
            }
            // else: zero-padded for positions before start of text
        }
        x
    }

    pub fn train(
        &mut self,
        text: &[u8],
        epochs: usize,
        lr: f64,
        batch_size: usize,
    ) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
        let tiled = TiledEngine::new(tam_gpu::detect());
        self.train_with(&tiled, text, epochs, lr, batch_size)
    }

    /// Train using a specific TiledEngine (any backend: CUDA, CPU, Vulkan).
    pub fn train_with(
        &mut self,
        tiled: &TiledEngine,
        text: &[u8],
        epochs: usize,
        lr: f64,
        batch_size: usize,
    ) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
        let v = self.vocab;
        let h = self.hidden;
        let d = self.input_dim;

        let n_pairs = text.len() - 1;
        let params = d * h + h * v;
        eprintln!("[experiment0] {} bytes, ctx={}, input_dim={}, hidden={}, params={}",
            text.len(), self.ctx, d, h, params);

        let mut losses = Vec::new();
        let t_start = Instant::now();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch_start in (0..n_pairs).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_pairs);
                let bs = batch_end - batch_start;
                if bs < 2 { continue; }

                // Build context-window input matrix X (bs × input_dim)
                let mut x = vec![0.0f64; bs * d];
                let mut targets = vec![0usize; bs];
                for i in 0..bs {
                    let pos = batch_start + i + 1; // predict text[pos] from context ending at text[pos-1]
                    let ctx_vec = self.encode_context(text, pos);
                    x[i * d..(i + 1) * d].copy_from_slice(&ctx_vec);
                    targets[i] = text[pos] as usize;
                }

                // Forward: h_raw = X @ W1 (bs × H)
                let h_raw = tiled.run(&DotProductOp, &x, &self.w1, bs, h, d)?;

                // ReLU + mask
                let mut h_act = vec![0.0f64; bs * h];
                let mut relu_mask = vec![0.0f64; bs * h];
                for i in 0..bs * h {
                    if h_raw[i] > 0.0 {
                        h_act[i] = h_raw[i];
                        relu_mask[i] = 1.0;
                    }
                }

                // Forward: logits = h_act @ W2 (bs × 256)
                let logits = tiled.run(&DotProductOp, &h_act, &self.w2, bs, v, h)?;

                // Softmax + cross-entropy
                let mut probs = vec![0.0f64; bs * v];
                let mut batch_loss = 0.0;
                for i in 0..bs {
                    let row = &logits[i * v..(i + 1) * v];
                    let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum_exp = 0.0f64;
                    for j in 0..v {
                        probs[i * v + j] = (row[j] - max_val).exp();
                        sum_exp += probs[i * v + j];
                    }
                    for j in 0..v {
                        probs[i * v + j] /= sum_exp;
                    }
                    let p = probs[i * v + targets[i]].max(1e-15);
                    batch_loss -= p.ln();
                }
                batch_loss /= bs as f64;
                epoch_loss += batch_loss;
                n_batches += 1;

                // Backward: δ_out = probs - one_hot(target)
                let mut d_out = probs.clone();
                for i in 0..bs {
                    d_out[i * v + targets[i]] -= 1.0;
                    for j in 0..v {
                        d_out[i * v + j] /= bs as f64;
                    }
                }

                // ∇W2 = h_act' @ δ_out
                let mut h_act_t = vec![0.0f64; h * bs];
                for i in 0..bs {
                    for j in 0..h {
                        h_act_t[j * bs + i] = h_act[i * h + j];
                    }
                }
                let grad_w2 = tiled.run(&DotProductOp, &h_act_t, &d_out, h, v, bs)?;

                // δ_hidden = δ_out @ W2' ⊙ relu_mask
                let mut w2_t = vec![0.0f64; v * h];
                for i in 0..h {
                    for j in 0..v {
                        w2_t[j * h + i] = self.w2[i * v + j];
                    }
                }
                let d_hid_raw = tiled.run(&DotProductOp, &d_out, &w2_t, bs, h, v)?;
                let mut d_hid = vec![0.0f64; bs * h];
                for i in 0..bs * h {
                    d_hid[i] = d_hid_raw[i] * relu_mask[i];
                }

                // ∇W1 = X' @ δ_hidden
                let mut x_t = vec![0.0f64; d * bs];
                for i in 0..bs {
                    for j in 0..d {
                        x_t[j * bs + i] = x[i * d + j];
                    }
                }
                let grad_w1 = tiled.run(&DotProductOp, &x_t, &d_hid, d, h, bs)?;

                // Update weights
                for i in 0..self.w1.len() {
                    self.w1[i] -= lr * grad_w1[i];
                }
                for i in 0..self.w2.len() {
                    self.w2[i] -= lr * grad_w2[i];
                }
            }

            let avg_loss = if n_batches > 0 { epoch_loss / n_batches as f64 } else { f64::NAN };
            losses.push(avg_loss);

            if epoch % 20 == 0 || epoch == epochs - 1 {
                let elapsed = t_start.elapsed().as_secs_f64();
                eprintln!("[experiment0] epoch {}/{}: loss={:.4} ({:.1}s)",
                    epoch + 1, epochs, avg_loss, elapsed);
            }
        }

        let final_loss = *losses.last().unwrap_or(&f64::NAN);
        Ok((final_loss, losses))
    }

    /// Predict next character probabilities from a context window.
    pub fn predict_probs_ctx(&self, context: &[u8]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let tiled = TiledEngine::new(tam_gpu::detect());
        self.predict_probs_ctx_with(&tiled, context)
    }

    /// Predict using a specific TiledEngine (any backend).
    pub fn predict_probs_ctx_with(&self, tiled: &TiledEngine, context: &[u8]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let v = self.vocab;
        let h = self.hidden;
        let d = self.input_dim;

        // Build context input
        let mut x = vec![0.0f64; d];
        for c in 0..self.ctx {
            if c < context.len() {
                let byte = context[context.len() - 1 - c] as usize;
                x[(self.ctx - 1 - c) * v + byte] = 1.0;
            }
        }

        let h_raw = tiled.run(&DotProductOp, &x, &self.w1, 1, h, d)?;
        let h_act: Vec<f64> = h_raw.iter().map(|&x| x.max(0.0)).collect();
        let logits = tiled.run(&DotProductOp, &h_act, &self.w2, 1, v, h)?;

        let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = logits.iter().map(|&l| (l - max_val).exp()).collect();
        let sum: f64 = probs.iter().sum();
        for p in probs.iter_mut() { *p /= sum; }

        Ok(probs)
    }

    /// Generate text autoregressively from a seed.
    pub fn generate(&self, seed: &[u8], length: usize, greedy: bool) -> Result<String, Box<dyn std::error::Error>> {
        let mut result: Vec<u8> = seed.to_vec();
        let mut rng: u64 = 1337;

        for _ in 0..length {
            let ctx_start = if result.len() > self.ctx { result.len() - self.ctx } else { 0 };
            let context = &result[ctx_start..];
            let probs = self.predict_probs_ctx(context)?;

            let next = if greedy {
                probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u8
            } else {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (rng >> 33) as f64 / (1u64 << 31) as f64;
                let mut cumulative = 0.0;
                let mut chosen = 0u8;
                for (i, &p) in probs.iter().enumerate() {
                    cumulative += p;
                    if cumulative > u {
                        chosen = i as u8;
                        break;
                    }
                }
                chosen
            };

            result.push(next);
        }

        Ok(String::from_utf8_lossy(&result).to_string())
    }

    /// Parallel generation: render all positions simultaneously, converge by confidence.
    /// This is the Pith-style renderer for text.
    pub fn render_parallel(
        &self,
        seed: &[u8],
        length: usize,
        max_iterations: usize,
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let total_len = seed.len() + length;
        let mut canvas: Vec<u8> = vec![b' '; total_len]; // initialize with spaces
        let mut confidence: Vec<f64> = vec![0.0; total_len];

        // Plant the seed — these positions are fixed
        for (i, &b) in seed.iter().enumerate() {
            canvas[i] = b;
            confidence[i] = 1.0; // seed positions are 100% confident
        }

        let mut snapshots: Vec<Vec<u8>> = Vec::new();
        snapshots.push(canvas.clone());

        for _iter in 0..max_iterations {
            let mut changed = false;

            // Update all non-seed positions simultaneously
            // (we iterate but conceptually this is one parallel step)
            let prev_canvas = canvas.clone();
            for pos in seed.len()..total_len {
                let ctx_start = if pos > self.ctx { pos - self.ctx } else { 0 };
                let context = &prev_canvas[ctx_start..pos]; // use PREVIOUS state, not current

                let probs = self.predict_probs_ctx(context)?;

                // Argmax + confidence
                let (best_idx, &best_prob) = probs.iter().enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap();

                if best_idx as u8 != canvas[pos] || (best_prob - confidence[pos]).abs() > 0.01 {
                    changed = true;
                }

                canvas[pos] = best_idx as u8;
                confidence[pos] = best_prob;
            }

            snapshots.push(canvas.clone());

            if !changed { break; }
        }

        Ok(snapshots)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  EXPERIMENT 0: Can Tambear Learn Language?");
    eprintln!("  Context window + ReLU net via pure TiledEngine DotProduct");
    eprintln!("  No PyTorch. No autodiff. Just accumulate.");
    eprintln!("═══════════════════════════════════════════════════════════");

    let text = b"the cat sat on the mat. the dog sat on the log. \
the cat chased the dog around the garden. the dog chased the cat across the yard. \
the bird sang in the tree. the fish swam in the pond. \
the cat sat on the mat and watched the bird. the dog sat on the log and watched the fish. \
the cat chased the dog around the garden again. the dog chased the cat across the yard again. \
the bird sang in the tree all day. the fish swam in the pond all day. \
the cat and the dog are friends. the bird and the fish are not friends. \
the cat sat on the mat. the dog sat on the log. the bird sang in the tree. the fish swam in the pond. \
the cat chased the dog. the dog chased the cat. the bird sang. the fish swam. \
the cat sat on the mat and the dog sat on the log and the bird sang in the tree and the fish swam in the pond. \
the cat is on the mat. the dog is on the log. the bird is in the tree. the fish is in the pond. \
the cat chased the dog around the garden. the dog chased the cat across the yard. \
the bird sang in the tree. the fish swam in the pond. the cat sat on the mat. the dog sat on the log. ";

    let ctx = 8;       // see last 8 characters
    let hidden = 128;
    let epochs = 200;
    let lr = 0.01;     // smaller lr for larger model
    let batch_size = 64;

    let mut model = CharModel::new(hidden, ctx);
    let params = model.input_dim * hidden + hidden * 256;
    eprintln!("\nModel: {} params, ctx={}, hidden={}, {} bytes of text",
        params, ctx, hidden, text.len());
    eprintln!("Training: {} epochs, lr={}, batch_size={}\n", epochs, lr, batch_size);

    // ── Train ────────────────────────────────────────────────────────
    let (final_loss, losses) = model.train(text, epochs, lr, batch_size)?;

    // ── Generate (autoregressive) ────────────────────────────────────
    eprintln!("\n── Autoregressive generation (greedy) ──");
    let generated = model.generate(b"the cat", 60, true)?;
    eprintln!("{}", generated);

    eprintln!("\n── Autoregressive generation (sampled) ──");
    let generated = model.generate(b"the dog", 60, false)?;
    eprintln!("{}", generated);

    eprintln!("\n── Autoregressive generation (sampled) ──");
    let generated = model.generate(b"the ", 60, false)?;
    eprintln!("{}", generated);

    // ── Parallel render (Pith-style) ─────────────────────────────────
    eprintln!("\n── Parallel render (Pith-style crystallization) ──");
    let snapshots = model.render_parallel(b"the cat ", 40, 10)?;
    for (i, snap) in snapshots.iter().enumerate() {
        let s = String::from_utf8_lossy(snap);
        eprintln!("  iter {}: {}", i, s);
    }

    // ── Summary ──────────────────────────────────────────────────────
    let initial = losses.first().unwrap_or(&0.0);
    let final_l = losses.last().unwrap_or(&0.0);
    let reduction = if *initial > 0.0 { (1.0 - final_l / initial) * 100.0 } else { 0.0 };
    let random_baseline = (256.0f64).ln();
    eprintln!("\n── Summary ──");
    eprintln!("Loss: {:.4} → {:.4} ({:.1}% reduction)", initial, final_l, reduction);
    eprintln!("Random baseline: {:.4}", random_baseline);
    eprintln!("Bits per character: {:.2} (random: {:.2})", final_l / 2.0_f64.ln(), random_baseline / 2.0_f64.ln());

    if *final_l < random_baseline * 0.3 {
        eprintln!("\n✓ MODEL LEARNED WORDS. Loss is below 30% of random baseline.");
    } else if *final_l < random_baseline * 0.5 {
        eprintln!("\n✓ MODEL LEARNED PATTERNS. Loss is below 50% of random baseline.");
    } else {
        eprintln!("\n✗ Loss still high. Need more data or tuning.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_creates_and_predicts() {
        let model = CharModel::new(32, 4);
        let probs = model.predict_probs_ctx(b"test").unwrap();
        assert_eq!(probs.len(), 256);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probs don't sum to 1: {}", sum);
    }

    #[test]
    fn training_reduces_loss() {
        let text = b"abababababababababababababababababababababababababababababababab";
        let mut model = CharModel::new(16, 4);
        let (final_loss, losses) = model.train(text, 50, 0.1, 16).unwrap();
        let initial_loss = losses[0];
        assert!(final_loss < initial_loss, "loss should decrease: {} → {}", initial_loss, final_loss);
    }

    #[test]
    fn parallel_render_converges() {
        let text = b"ababababababababababababababababababababababababababababababababababababababababab";
        let mut model = CharModel::new(16, 4);
        model.train(text, 50, 0.1, 16).unwrap();
        let snapshots = model.render_parallel(b"ab", 10, 5).unwrap();
        // Should have at least 2 snapshots (initial + at least 1 iteration)
        assert!(snapshots.len() >= 2, "should iterate at least once");
        // Final snapshot should be different from initial (all spaces)
        assert_ne!(snapshots.last().unwrap(), &snapshots[0], "should change from initial");
    }

    /// Gradient duality proof on pure CPU — no GPU required.
    ///
    /// This validates notebook 005's claim: a 2-layer neural net trains using
    /// only DotProduct accumulate calls. The same code that runs on CUDA runs
    /// on CpuBackend through TiledEngine's TamGpu abstraction.
    ///
    /// 5 DotProduct calls per forward+backward (2L+1 for L=2):
    ///   Forward:  X@W1, h_act@W2
    ///   Backward: h_act'@δ_out, δ_out@W2', X'@δ_hidden
    #[test]
    fn gradient_duality_on_cpu_backend() {
        let cpu = Arc::new(tam_gpu::CpuBackend::new());
        let tiled = TiledEngine::new(cpu);

        let text = b"abababababababababababababababababababababababababababababababab";
        let mut model = CharModel::new(16, 4);
        let (final_loss, losses) = model.train_with(&tiled, text, 50, 0.1, 16).unwrap();
        let initial_loss = losses[0];
        assert!(
            final_loss < initial_loss,
            "loss should decrease on CPU: {initial_loss} → {final_loss}"
        );

        // Verify prediction works on CPU too
        let probs = model.predict_probs_ctx_with(&tiled, b"abab").unwrap();
        assert_eq!(probs.len(), 256);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probs sum to {sum}, not 1.0");
    }
}
