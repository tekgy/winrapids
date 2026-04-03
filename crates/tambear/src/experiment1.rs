//! # Experiment 1: Does Superposition Beat Single?
//!
//! Same parameter budget, two variants:
//! A) Single-view: one path with ctx=8, all params in one network
//! B) Superposition: two paths (ctx=8 long-range + ctx=2 short-range),
//!    learned combination weight α, half params per path
//!
//! Hypothesis: B beats A because two complementary views capture more
//! structure than one view with the same total parameters.
//!
//! From Pith's convergence guarantee: O(exp(-λ·order)). Two views = order 2.
//! Should converge faster and to a lower loss.

use std::sync::Arc;
use std::time::Instant;
use winrapids_tiled::{TiledEngine, DotProductOp};

// ─── Single-view model (baseline) ───────────────────────────────────────

struct SingleViewModel {
    w1: Vec<f64>,  // input_dim × hidden
    w2: Vec<f64>,  // hidden × 256
    hidden: usize,
    vocab: usize,
    ctx: usize,
    input_dim: usize,
}

// ─── Superposition model (two views, learned combination) ───────────────

struct SuperpositionModel {
    // Path A: sequential context (last 8 chars, contiguous)
    w1a: Vec<f64>,  // input_dim_a × hidden_a
    w2a: Vec<f64>,  // hidden_a × 256

    // Path B: skip-gram context (chars at -1, -2, -4, -8 — different temporal scales)
    w1b: Vec<f64>,  // input_dim_b × hidden_b
    w2b: Vec<f64>,  // hidden_b × 256

    // Combination: learned α per output class
    alpha: Vec<f64>,  // 256 weights (one per output byte)

    hidden_a: usize,
    hidden_b: usize,
    vocab: usize,
    ctx_a: usize,      // sequential window size
    skip_offsets: Vec<usize>,  // skip-gram offsets for path B
    input_dim_a: usize,
    input_dim_b: usize,
}

fn init_weights(size: usize, fan_in: usize, fan_out: usize, seed: &mut u64) -> Vec<f64> {
    let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let mut w = vec![0.0f64; size];
    for v in w.iter_mut() {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (*seed >> 33) as f64 / (1u64 << 30) as f64 - 1.0;
        *v = u * scale;
    }
    w
}

fn encode_context(text: &[u8], pos: usize, ctx: usize, vocab: usize) -> Vec<f64> {
    let input_dim = vocab * ctx;
    let mut x = vec![0.0f64; input_dim];
    for c in 0..ctx {
        if pos >= ctx - c {
            let byte = text[pos - (ctx - c)] as usize;
            x[c * vocab + byte] = 1.0;
        }
    }
    x
}

fn softmax_cross_entropy(logits: &[f64], targets: &[usize], vocab: usize, bs: usize) -> (Vec<f64>, f64) {
    let mut probs = vec![0.0f64; bs * vocab];
    let mut loss = 0.0;
    for i in 0..bs {
        let row = &logits[i * vocab..(i + 1) * vocab];
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum_exp = 0.0f64;
        for j in 0..vocab {
            probs[i * vocab + j] = (row[j] - max_val).exp();
            sum_exp += probs[i * vocab + j];
        }
        for j in 0..vocab {
            probs[i * vocab + j] /= sum_exp;
        }
        let p = probs[i * vocab + targets[i]].max(1e-15);
        loss -= p.ln();
    }
    loss /= bs as f64;
    (probs, loss)
}

fn transpose(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = m[i * cols + j];
        }
    }
    t
}

impl SingleViewModel {
    fn new(hidden: usize, ctx: usize) -> Self {
        let vocab = 256;
        let input_dim = vocab * ctx;
        let mut seed = 42u64;
        let w1 = init_weights(input_dim * hidden, input_dim, hidden, &mut seed);
        let w2 = init_weights(hidden * vocab, hidden, vocab, &mut seed);
        Self { w1, w2, hidden, vocab, ctx, input_dim }
    }

    fn param_count(&self) -> usize {
        self.input_dim * self.hidden + self.hidden * self.vocab
    }

    fn train(&mut self, tiled: &TiledEngine, text: &[u8], epochs: usize, lr: f64, batch_size: usize)
        -> Result<Vec<f64>, Box<dyn std::error::Error>>
    {
        let v = self.vocab;
        let h = self.hidden;
        let d = self.input_dim;
        let n_pairs = text.len() - 1;
        let mut losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch_start in (0..n_pairs).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_pairs);
                let bs = batch_end - batch_start;
                if bs < 2 { continue; }

                let mut x = vec![0.0f64; bs * d];
                let mut targets = vec![0usize; bs];
                for i in 0..bs {
                    let pos = batch_start + i + 1;
                    let ctx_vec = encode_context(text, pos, self.ctx, v);
                    x[i * d..(i + 1) * d].copy_from_slice(&ctx_vec);
                    targets[i] = text[pos] as usize;
                }

                // Forward
                let h_raw = tiled.run(&DotProductOp, &x, &self.w1, bs, h, d)?;
                let mut h_act = vec![0.0f64; bs * h];
                let mut relu_mask = vec![0.0f64; bs * h];
                for i in 0..bs * h {
                    if h_raw[i] > 0.0 { h_act[i] = h_raw[i]; relu_mask[i] = 1.0; }
                }
                let logits = tiled.run(&DotProductOp, &h_act, &self.w2, bs, v, h)?;

                let (probs, batch_loss) = softmax_cross_entropy(&logits, &targets, v, bs);
                epoch_loss += batch_loss;
                n_batches += 1;

                // Backward
                let mut d_out = probs;
                for i in 0..bs {
                    d_out[i * v + targets[i]] -= 1.0;
                    for j in 0..v { d_out[i * v + j] /= bs as f64; }
                }

                let h_act_t = transpose(&h_act, bs, h);
                let grad_w2 = tiled.run(&DotProductOp, &h_act_t, &d_out, h, v, bs)?;

                let w2_t = transpose(&self.w2, h, v);
                let d_hid_raw = tiled.run(&DotProductOp, &d_out, &w2_t, bs, h, v)?;
                let d_hid: Vec<f64> = d_hid_raw.iter().zip(&relu_mask).map(|(d, m)| d * m).collect();

                let x_t = transpose(&x, bs, d);
                let grad_w1 = tiled.run(&DotProductOp, &x_t, &d_hid, d, h, bs)?;

                for i in 0..self.w1.len() { self.w1[i] -= lr * grad_w1[i]; }
                for i in 0..self.w2.len() { self.w2[i] -= lr * grad_w2[i]; }
            }

            let avg = if n_batches > 0 { epoch_loss / n_batches as f64 } else { f64::NAN };
            losses.push(avg);
            if epoch % 50 == 0 || epoch == epochs - 1 {
                eprintln!("  [single] epoch {}/{}: loss={:.4}", epoch + 1, epochs, avg);
            }
        }
        Ok(losses)
    }
}

impl SuperpositionModel {
    fn new(hidden_a: usize, hidden_b: usize, ctx_a: usize, skip_offsets: Vec<usize>) -> Self {
        let vocab = 256;
        let input_dim_a = vocab * ctx_a;
        let input_dim_b = vocab * skip_offsets.len();
        let mut seed = 42u64;
        let w1a = init_weights(input_dim_a * hidden_a, input_dim_a, hidden_a, &mut seed);
        let w2a = init_weights(hidden_a * vocab, hidden_a, vocab, &mut seed);
        let w1b = init_weights(input_dim_b * hidden_b, input_dim_b, hidden_b, &mut seed);
        let w2b = init_weights(hidden_b * vocab, hidden_b, vocab, &mut seed);
        let alpha = vec![0.0f64; vocab]; // start at 0 = sigmoid(0) = 0.5 = equal
        Self { w1a, w2a, w1b, w2b, alpha, hidden_a, hidden_b, vocab, ctx_a, skip_offsets, input_dim_a, input_dim_b }
    }

    fn param_count(&self) -> usize {
        self.input_dim_a * self.hidden_a + self.hidden_a * self.vocab
        + self.input_dim_b * self.hidden_b + self.hidden_b * self.vocab
        + self.vocab // alpha
    }

    fn train(&mut self, tiled: &TiledEngine, text: &[u8], epochs: usize, lr: f64, batch_size: usize)
        -> Result<Vec<f64>, Box<dyn std::error::Error>>
    {
        let v = self.vocab;
        let ha = self.hidden_a;
        let hb = self.hidden_b;
        let da = self.input_dim_a;
        let db = self.input_dim_b;
        let n_pairs = text.len() - 1;
        let mut losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            for batch_start in (0..n_pairs).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_pairs);
                let bs = batch_end - batch_start;
                if bs < 2 { continue; }

                // Build inputs for both paths
                let mut xa = vec![0.0f64; bs * da];
                let mut xb = vec![0.0f64; bs * db];
                let mut targets = vec![0usize; bs];
                for i in 0..bs {
                    let pos = batch_start + i + 1;
                    // Path A: sequential context (last ctx_a chars)
                    let ctx_a_vec = encode_context(text, pos, self.ctx_a, v);
                    xa[i * da..(i + 1) * da].copy_from_slice(&ctx_a_vec);
                    // Path B: skip-gram context (chars at specific offsets)
                    for (slot, &offset) in self.skip_offsets.iter().enumerate() {
                        if pos > offset {
                            let byte = text[pos - 1 - offset] as usize;
                            xb[i * db + slot * v + byte] = 1.0;
                        }
                    }
                    targets[i] = text[pos] as usize;
                }

                // ── Path A forward (long-range) ──────────────────────────
                let ha_raw = tiled.run(&DotProductOp, &xa, &self.w1a, bs, ha, da)?;
                let mut ha_act = vec![0.0f64; bs * ha];
                let mut relu_mask_a = vec![0.0f64; bs * ha];
                for i in 0..bs * ha {
                    if ha_raw[i] > 0.0 { ha_act[i] = ha_raw[i]; relu_mask_a[i] = 1.0; }
                }
                let logits_a = tiled.run(&DotProductOp, &ha_act, &self.w2a, bs, v, ha)?;

                // ── Path B forward (short-range) ─────────────────────────
                let hb_raw = tiled.run(&DotProductOp, &xb, &self.w1b, bs, hb, db)?;
                let mut hb_act = vec![0.0f64; bs * hb];
                let mut relu_mask_b = vec![0.0f64; bs * hb];
                for i in 0..bs * hb {
                    if hb_raw[i] > 0.0 { hb_act[i] = hb_raw[i]; relu_mask_b[i] = 1.0; }
                }
                let logits_b = tiled.run(&DotProductOp, &hb_act, &self.w2b, bs, v, hb)?;

                // ── Combine with sigmoid(alpha) ──────────────────────────
                let mut combined_logits = vec![0.0f64; bs * v];
                let sigmoid_alpha: Vec<f64> = self.alpha.iter().map(|&a| 1.0 / (1.0 + (-a).exp())).collect();
                for i in 0..bs {
                    for j in 0..v {
                        let sa = sigmoid_alpha[j];
                        combined_logits[i * v + j] = sa * logits_a[i * v + j] + (1.0 - sa) * logits_b[i * v + j];
                    }
                }

                let (probs, batch_loss) = softmax_cross_entropy(&combined_logits, &targets, v, bs);
                epoch_loss += batch_loss;
                n_batches += 1;

                // ── Backward through combination ─────────────────────────
                let mut d_combined = probs;
                for i in 0..bs {
                    d_combined[i * v + targets[i]] -= 1.0;
                    for j in 0..v { d_combined[i * v + j] /= bs as f64; }
                }

                // Gradient for alpha
                let mut grad_alpha = vec![0.0f64; v];
                for i in 0..bs {
                    for j in 0..v {
                        let sa = sigmoid_alpha[j];
                        let d_alpha = d_combined[i * v + j] * (logits_a[i * v + j] - logits_b[i * v + j]);
                        grad_alpha[j] += d_alpha * sa * (1.0 - sa); // sigmoid derivative
                    }
                }
                for j in 0..v { grad_alpha[j] /= bs as f64; }

                // Split gradient to paths
                let mut d_out_a = vec![0.0f64; bs * v];
                let mut d_out_b = vec![0.0f64; bs * v];
                for i in 0..bs {
                    for j in 0..v {
                        d_out_a[i * v + j] = d_combined[i * v + j] * sigmoid_alpha[j];
                        d_out_b[i * v + j] = d_combined[i * v + j] * (1.0 - sigmoid_alpha[j]);
                    }
                }

                // ── Path A backward ──────────────────────────────────────
                let ha_act_t = transpose(&ha_act, bs, ha);
                let grad_w2a = tiled.run(&DotProductOp, &ha_act_t, &d_out_a, ha, v, bs)?;
                let w2a_t = transpose(&self.w2a, ha, v);
                let d_hid_a_raw = tiled.run(&DotProductOp, &d_out_a, &w2a_t, bs, ha, v)?;
                let d_hid_a: Vec<f64> = d_hid_a_raw.iter().zip(&relu_mask_a).map(|(d, m)| d * m).collect();
                let xa_t = transpose(&xa, bs, da);
                let grad_w1a = tiled.run(&DotProductOp, &xa_t, &d_hid_a, da, ha, bs)?;

                // ── Path B backward ──────────────────────────────────────
                let hb_act_t = transpose(&hb_act, bs, hb);
                let grad_w2b = tiled.run(&DotProductOp, &hb_act_t, &d_out_b, hb, v, bs)?;
                let w2b_t = transpose(&self.w2b, hb, v);
                let d_hid_b_raw = tiled.run(&DotProductOp, &d_out_b, &w2b_t, bs, hb, v)?;
                let d_hid_b: Vec<f64> = d_hid_b_raw.iter().zip(&relu_mask_b).map(|(d, m)| d * m).collect();
                let xb_t = transpose(&xb, bs, db);
                let grad_w1b = tiled.run(&DotProductOp, &xb_t, &d_hid_b, db, hb, bs)?;

                // ── Update all weights ───────────────────────────────────
                for i in 0..self.w1a.len() { self.w1a[i] -= lr * grad_w1a[i]; }
                for i in 0..self.w2a.len() { self.w2a[i] -= lr * grad_w2a[i]; }
                for i in 0..self.w1b.len() { self.w1b[i] -= lr * grad_w1b[i]; }
                for i in 0..self.w2b.len() { self.w2b[i] -= lr * grad_w2b[i]; }
                for i in 0..v { self.alpha[i] -= lr * grad_alpha[i]; }
            }

            let avg = if n_batches > 0 { epoch_loss / n_batches as f64 } else { f64::NAN };
            losses.push(avg);
            if epoch % 50 == 0 || epoch == epochs - 1 {
                eprintln!("  [super]  epoch {}/{}: loss={:.4}", epoch + 1, epochs, avg);
            }
        }
        Ok(losses)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  EXPERIMENT 1: Does Superposition Beat Single?");
    eprintln!("  Same parameter budget. One view vs two views.");
    eprintln!("  Hypothesis: two complementary views > one view.");
    eprintln!("═══════════════════════════════════════════════════════════\n");

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

    let tiled = TiledEngine::new(tam_gpu::detect());

    let epochs = 200;
    let lr = 0.01;
    let batch_size = 64;

    // ── Model A: Single view, ctx=8, hidden=128 ─────────────────────
    // Param budget: 256*8*128 + 128*256 = 262144 + 32768 = 294912
    let mut single = SingleViewModel::new(128, 8);
    eprintln!("Model A (single view): {} params, ctx=8, hidden=128", single.param_count());

    // ── Model B: Superposition with ORTHOGONAL views ──────────────
    // Path A: sequential (ctx=8) — captures spelling, bigrams, local syntax
    // Path B: skip-gram (offsets 0,3,7,15) — captures word-level patterns, repetition, rhythm
    //   offset 0 = char at -1 (immediate predecessor, like path A)
    //   offset 3 = char at -4 (within same word typically)
    //   offset 7 = char at -8 (previous word boundary)
    //   offset 15 = char at -16 (two words back — longer range than path A!)
    // These are ORTHOGONAL: sequential sees contiguous local structure,
    // skip-gram sees multi-scale temporal structure. Different projections.
    let skip_offsets = vec![0, 3, 7, 15];
    let mut superpos = SuperpositionModel::new(80, 48, 8, skip_offsets);
    eprintln!("Model B (superposition): {} params", superpos.param_count());
    eprintln!("  Path A (sequential ctx=8): {} params", superpos.input_dim_a * superpos.hidden_a + superpos.hidden_a * 256);
    eprintln!("  Path B (skip-gram [-1,-4,-8,-16]): {} params", superpos.input_dim_b * superpos.hidden_b + superpos.hidden_b * 256);
    eprintln!("  Combination (alpha): {} params", 256);

    eprintln!("\nTraining both for {} epochs, lr={}, batch={}\n", epochs, lr, batch_size);

    // ── Train single ─────────────────────────────────────────────────
    eprintln!("── Training Model A (single view) ──");
    let losses_single = single.train(&tiled, text, epochs, lr, batch_size)?;

    // ── Train superposition ──────────────────────────────────────────
    eprintln!("\n── Training Model B (superposition) ──");
    let losses_super = superpos.train(&tiled, text, epochs, lr, batch_size)?;

    // ── Compare ──────────────────────────────────────────────────────
    let final_single = losses_single.last().unwrap_or(&f64::NAN);
    let final_super = losses_super.last().unwrap_or(&f64::NAN);

    eprintln!("\n══════════════════════════════════════════════════════");
    eprintln!("  RESULTS");
    eprintln!("══════════════════════════════════════════════════════");
    eprintln!("  Single view:   final loss = {:.4} ({} params)", final_single, single.param_count());
    eprintln!("  Superposition: final loss = {:.4} ({} params)", final_super, superpos.param_count());

    if final_super < final_single {
        let improvement = (1.0 - final_super / final_single) * 100.0;
        eprintln!("\n  ✓ SUPERPOSITION WINS by {:.1}% with FEWER parameters!", improvement);
        eprintln!("  Two complementary views beat one view. The hypothesis holds.");
    } else {
        let deficit = (final_super / final_single - 1.0) * 100.0;
        eprintln!("\n  ✗ Single view wins by {:.1}%.", deficit);
        eprintln!("  Possible reasons: too few epochs, wrong lr, views not complementary enough.");
    }

    // ── Report learned combination weights ───────────────────────────
    let sigmoid_alpha: Vec<f64> = superpos.alpha.iter().map(|&a| 1.0 / (1.0 + (-a).exp())).collect();
    let mean_alpha: f64 = sigmoid_alpha.iter().sum::<f64>() / 256.0;
    eprintln!("\n  Learned combination: mean α={:.3} (0.5=equal, >0.5=favors sequential, <0.5=favors skip-gram)",
        mean_alpha);

    // Show α for key characters
    let chars_to_show = [b' ', b't', b'h', b'e', b'.', b'a', b'c', b'd', b'o', b'n'];
    eprintln!("  Per-character α (sequential vs skip-gram):");
    for &ch in &chars_to_show {
        let a = sigmoid_alpha[ch as usize];
        let label = if a > 0.6 { "sequential dominates" }
                    else if a < 0.4 { "skip-gram dominates" }
                    else { "balanced" };
        eprintln!("    '{}': α={:.3} ({})", ch as char, a, label);
    }

    // ── Convergence comparison ───────────────────────────────────────
    eprintln!("\n  Convergence comparison (loss at epochs 50, 100, 150, 200):");
    for &ep in &[49, 99, 149, 199] {
        if ep < losses_single.len() && ep < losses_super.len() {
            eprintln!("    epoch {:>3}: single={:.4}  super={:.4}  Δ={:.4}",
                ep + 1,
                losses_single[ep],
                losses_super[ep],
                losses_single[ep] - losses_super[ep]);
        }
    }

    Ok(())
}
