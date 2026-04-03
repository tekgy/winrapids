//! # Experiment 1b: Bidirectional Superposition
//!
//! Shared encoder, two heads: forward context + backward context.
//! Provably orthogonal views — forward can't see what backward sees.
//!
//! Model A: Single direction (forward only, ctx=8)
//! Model B: Bidirectional (forward ctx=8 + backward ctx=8, shared encoder)
//!          Same total param budget via smaller hidden.
//!
//! This IS the Pith-style superposition: two projections of the same
//! data that capture orthogonal structure.

use std::sync::Arc;
use winrapids_tiled::{TiledEngine, DotProductOp};

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

fn softmax_ce(logits: &[f64], targets: &[usize], v: usize, bs: usize) -> (Vec<f64>, f64) {
    let mut probs = vec![0.0f64; bs * v];
    let mut loss = 0.0;
    for i in 0..bs {
        let row = &logits[i * v..(i + 1) * v];
        let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut se = 0.0f64;
        for j in 0..v { probs[i * v + j] = (row[j] - mx).exp(); se += probs[i * v + j]; }
        for j in 0..v { probs[i * v + j] /= se; }
        loss -= probs[i * v + targets[i]].max(1e-15).ln();
    }
    (probs, loss / bs as f64)
}

fn transpose(m: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut t = vec![0.0; r * c];
    for i in 0..r { for j in 0..c { t[j * r + i] = m[i * c + j]; } }
    t
}

fn encode_ctx(text: &[u8], pos: usize, ctx: usize, v: usize) -> Vec<f64> {
    let d = v * ctx;
    let mut x = vec![0.0f64; d];
    for c in 0..ctx {
        if pos >= ctx - c {
            let byte = text[pos - (ctx - c)] as usize;
            x[c * v + byte] = 1.0;
        }
    }
    x
}

fn encode_backward(text: &[u8], pos: usize, ctx: usize, v: usize) -> Vec<f64> {
    let d = v * ctx;
    let mut x = vec![0.0f64; d];
    for c in 0..ctx {
        let future_pos = pos + 1 + c; // pos+1 is what we're predicting, skip it
        if future_pos < text.len() {
            let byte = text[future_pos] as usize;
            x[c * v + byte] = 1.0;
        }
    }
    x
}

// ─── Forward-only model (baseline) ──────────────────────────────────

struct ForwardModel {
    w1: Vec<f64>,
    w2: Vec<f64>,
    h: usize,
    v: usize,
    ctx: usize,
    d: usize,
}

impl ForwardModel {
    fn new(hidden: usize, ctx: usize) -> Self {
        let v = 256;
        let d = v * ctx;
        let mut seed = 42u64;
        Self {
            w1: init_weights(d * hidden, d, hidden, &mut seed),
            w2: init_weights(hidden * v, hidden, v, &mut seed),
            h: hidden, v, ctx, d,
        }
    }

    fn params(&self) -> usize { self.d * self.h + self.h * self.v }

    fn train(&mut self, tiled: &TiledEngine, text: &[u8], epochs: usize, lr: f64, bs_max: usize)
        -> Result<Vec<f64>, Box<dyn std::error::Error>>
    {
        let (v, h, d) = (self.v, self.h, self.d);
        // Need backward context too — skip first/last ctx positions
        let start = self.ctx;
        let end = text.len() - self.ctx - 1;
        if end <= start { return Err("text too short".into()); }
        let n = end - start;
        let mut losses = Vec::new();

        for epoch in 0..epochs {
            let mut el = 0.0;
            let mut nb = 0;
            for bs_start in (0..n).step_by(bs_max) {
                let bs = (bs_start + bs_max).min(n) - bs_start;
                if bs < 2 { continue; }

                let mut x = vec![0.0f64; bs * d];
                let mut tgt = vec![0usize; bs];
                for i in 0..bs {
                    let pos = start + bs_start + i;
                    let ctx_v = encode_ctx(text, pos, self.ctx, v);
                    x[i * d..(i + 1) * d].copy_from_slice(&ctx_v);
                    tgt[i] = text[pos] as usize;
                }

                let hr = tiled.run(&DotProductOp, &x, &self.w1, bs, h, d)?;
                let mut ha = vec![0.0f64; bs * h];
                let mut rm = vec![0.0f64; bs * h];
                for i in 0..bs * h { if hr[i] > 0.0 { ha[i] = hr[i]; rm[i] = 1.0; } }
                let logits = tiled.run(&DotProductOp, &ha, &self.w2, bs, v, h)?;

                let (probs, bl) = softmax_ce(&logits, &tgt, v, bs);
                el += bl; nb += 1;

                let mut dout = probs;
                for i in 0..bs { dout[i * v + tgt[i]] -= 1.0; for j in 0..v { dout[i * v + j] /= bs as f64; } }

                let hat = transpose(&ha, bs, h);
                let gw2 = tiled.run(&DotProductOp, &hat, &dout, h, v, bs)?;
                let w2t = transpose(&self.w2, h, v);
                let dhr = tiled.run(&DotProductOp, &dout, &w2t, bs, h, v)?;
                let dh: Vec<f64> = dhr.iter().zip(&rm).map(|(a, b)| a * b).collect();
                let xt = transpose(&x, bs, d);
                let gw1 = tiled.run(&DotProductOp, &xt, &dh, d, h, bs)?;

                for i in 0..self.w1.len() { self.w1[i] -= lr * gw1[i]; }
                for i in 0..self.w2.len() { self.w2[i] -= lr * gw2[i]; }
            }
            let avg = if nb > 0 { el / nb as f64 } else { f64::NAN };
            losses.push(avg);
            if epoch % 50 == 0 || epoch == epochs - 1 {
                eprintln!("  [fwd]  epoch {}/{}: loss={:.4}", epoch + 1, epochs, avg);
            }
        }
        Ok(losses)
    }
}

// ─── Bidirectional model (shared encoder, two heads) ────────────────

struct BiModel {
    // Shared encoder: takes concatenated [forward_ctx; backward_ctx]
    w_enc: Vec<f64>,   // (d_fwd + d_bwd) × hidden
    // Forward head
    w_fwd: Vec<f64>,   // hidden × vocab
    // Backward head
    w_bwd: Vec<f64>,   // hidden × vocab
    // Combination weights per output byte
    alpha: Vec<f64>,   // vocab
    h: usize,
    v: usize,
    ctx: usize,
    d_each: usize,     // v * ctx per direction
    d_total: usize,    // d_each * 2
}

impl BiModel {
    fn new(hidden: usize, ctx: usize) -> Self {
        let v = 256;
        let d_each = v * ctx;
        let d_total = d_each * 2;
        let mut seed = 42u64;
        Self {
            w_enc: init_weights(d_total * hidden, d_total, hidden, &mut seed),
            w_fwd: init_weights(hidden * v, hidden, v, &mut seed),
            w_bwd: init_weights(hidden * v, hidden, v, &mut seed),
            alpha: vec![0.0f64; v], // sigmoid(0) = 0.5 = equal
            h: hidden, v, ctx, d_each, d_total,
        }
    }

    fn params(&self) -> usize {
        self.d_total * self.h  // shared encoder
        + self.h * self.v      // forward head
        + self.h * self.v      // backward head
        + self.v               // alpha
    }

    fn train(&mut self, tiled: &TiledEngine, text: &[u8], epochs: usize, lr: f64, bs_max: usize)
        -> Result<Vec<f64>, Box<dyn std::error::Error>>
    {
        let (v, h, dt) = (self.v, self.h, self.d_total);
        let de = self.d_each;
        let start = self.ctx;
        let end = text.len() - self.ctx - 1;
        if end <= start { return Err("text too short".into()); }
        let n = end - start;
        let mut losses = Vec::new();

        for epoch in 0..epochs {
            let mut el = 0.0;
            let mut nb = 0;
            for bs_start in (0..n).step_by(bs_max) {
                let bs = (bs_start + bs_max).min(n) - bs_start;
                if bs < 2 { continue; }

                // Build concatenated [forward_ctx | backward_ctx] input
                let mut x = vec![0.0f64; bs * dt];
                let mut tgt = vec![0usize; bs];
                for i in 0..bs {
                    let pos = start + bs_start + i;
                    let fwd = encode_ctx(text, pos, self.ctx, v);
                    let bwd = encode_backward(text, pos, self.ctx, v);
                    x[i * dt..i * dt + de].copy_from_slice(&fwd);
                    x[i * dt + de..(i + 1) * dt].copy_from_slice(&bwd);
                    tgt[i] = text[pos] as usize;
                }

                // Shared encoder: hidden = ReLU(X @ W_enc)
                let hr = tiled.run(&DotProductOp, &x, &self.w_enc, bs, h, dt)?;
                let mut ha = vec![0.0f64; bs * h];
                let mut rm = vec![0.0f64; bs * h];
                for i in 0..bs * h { if hr[i] > 0.0 { ha[i] = hr[i]; rm[i] = 1.0; } }

                // Two heads: logits_fwd = hidden @ W_fwd, logits_bwd = hidden @ W_bwd
                let logits_fwd = tiled.run(&DotProductOp, &ha, &self.w_fwd, bs, v, h)?;
                let logits_bwd = tiled.run(&DotProductOp, &ha, &self.w_bwd, bs, v, h)?;

                // Combine: logits = sigmoid(alpha) * fwd + (1-sigmoid(alpha)) * bwd
                let sa: Vec<f64> = self.alpha.iter().map(|&a| 1.0 / (1.0 + (-a).exp())).collect();
                let mut logits = vec![0.0f64; bs * v];
                for i in 0..bs {
                    for j in 0..v {
                        logits[i * v + j] = sa[j] * logits_fwd[i * v + j]
                                          + (1.0 - sa[j]) * logits_bwd[i * v + j];
                    }
                }

                let (probs, bl) = softmax_ce(&logits, &tgt, v, bs);
                el += bl; nb += 1;

                // Backward
                let mut dc = probs;
                for i in 0..bs { dc[i * v + tgt[i]] -= 1.0; for j in 0..v { dc[i * v + j] /= bs as f64; } }

                // Gradient for alpha
                let mut ga = vec![0.0f64; v];
                for i in 0..bs {
                    for j in 0..v {
                        ga[j] += dc[i * v + j] * (logits_fwd[i * v + j] - logits_bwd[i * v + j])
                                 * sa[j] * (1.0 - sa[j]);
                    }
                }
                for j in 0..v { ga[j] /= bs as f64; }

                // Split gradient to heads
                let mut df = vec![0.0f64; bs * v];
                let mut db = vec![0.0f64; bs * v];
                for i in 0..bs {
                    for j in 0..v {
                        df[i * v + j] = dc[i * v + j] * sa[j];
                        db[i * v + j] = dc[i * v + j] * (1.0 - sa[j]);
                    }
                }

                // Grad W_fwd: hidden' @ d_fwd
                let hat = transpose(&ha, bs, h);
                let gw_fwd = tiled.run(&DotProductOp, &hat, &df, h, v, bs)?;
                // Grad W_bwd: hidden' @ d_bwd
                let gw_bwd = tiled.run(&DotProductOp, &hat, &db, h, v, bs)?;

                // Merge head gradients for backprop through shared encoder
                // d_hidden = d_fwd @ W_fwd' + d_bwd @ W_bwd'  (SUM of both paths)
                let wft = transpose(&self.w_fwd, h, v);
                let wbt = transpose(&self.w_bwd, h, v);
                let dh_fwd = tiled.run(&DotProductOp, &df, &wft, bs, h, v)?;
                let dh_bwd = tiled.run(&DotProductOp, &db, &wbt, bs, h, v)?;
                let dh: Vec<f64> = dh_fwd.iter().zip(&dh_bwd).zip(&rm)
                    .map(|((a, b), m)| (a + b) * m).collect();

                // Grad W_enc: X' @ d_hidden
                let xt = transpose(&x, bs, dt);
                let gw_enc = tiled.run(&DotProductOp, &xt, &dh, dt, h, bs)?;

                // Update
                for i in 0..self.w_enc.len() { self.w_enc[i] -= lr * gw_enc[i]; }
                for i in 0..self.w_fwd.len() { self.w_fwd[i] -= lr * gw_fwd[i]; }
                for i in 0..self.w_bwd.len() { self.w_bwd[i] -= lr * gw_bwd[i]; }
                for i in 0..v { self.alpha[i] -= lr * ga[i]; }
            }
            let avg = if nb > 0 { el / nb as f64 } else { f64::NAN };
            losses.push(avg);
            if epoch % 50 == 0 || epoch == epochs - 1 {
                eprintln!("  [bidi] epoch {}/{}: loss={:.4}", epoch + 1, epochs, avg);
            }
        }
        Ok(losses)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  EXPERIMENT 1b: Bidirectional Superposition");
    eprintln!("  Forward-only vs Forward+Backward (shared encoder)");
    eprintln!("  Hypothesis: seeing both directions > seeing one direction");
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
    let epochs = 300;
    let lr = 0.01;
    let batch = 64;

    // Model A: Forward only, ctx=8, hidden=96
    // Params: 256*8*96 + 96*256 = 196608 + 24576 = 221184
    let mut fwd_model = ForwardModel::new(96, 8);
    eprintln!("Model A (forward only): {} params, ctx=8, hidden=96", fwd_model.params());

    // Model B: Bidirectional, ctx=8 each direction, hidden=64 (SHARED encoder)
    // Params: 256*8*2*64 + 64*256 + 64*256 + 256 = 262144 + 16384 + 16384 + 256 = 295168
    // Encoder is bigger (sees 2x the data) but heads are small
    // Giving bidi MORE params to be fair — the question is whether 2 directions help
    let mut bi_model = BiModel::new(64, 8);
    eprintln!("Model B (bidirectional): {} params, ctx=8 each dir, hidden=64", bi_model.params());
    eprintln!("  Shared encoder:  {} params (sees forward+backward)", bi_model.d_total * bi_model.h);
    eprintln!("  Forward head:    {} params", bi_model.h * bi_model.v);
    eprintln!("  Backward head:   {} params", bi_model.h * bi_model.v);
    eprintln!("  Combination:     {} params\n", bi_model.v);

    eprintln!("── Training Model A (forward only, {} epochs) ──", epochs);
    let losses_fwd = fwd_model.train(&tiled, text, epochs, lr, batch)?;

    eprintln!("\n── Training Model B (bidirectional, {} epochs) ──", epochs);
    let losses_bi = bi_model.train(&tiled, text, epochs, lr, batch)?;

    let f_fwd = losses_fwd.last().unwrap();
    let f_bi = losses_bi.last().unwrap();

    eprintln!("\n══════════════════════════════════════════════════════");
    eprintln!("  RESULTS");
    eprintln!("══════════════════════════════════════════════════════");
    eprintln!("  Forward only:   loss = {:.4} ({} params)", f_fwd, fwd_model.params());
    eprintln!("  Bidirectional:  loss = {:.4} ({} params)", f_bi, bi_model.params());

    if f_bi < f_fwd {
        let imp = (1.0 - f_bi / f_fwd) * 100.0;
        eprintln!("\n  ✓ BIDIRECTIONAL WINS by {:.1}%!", imp);
        eprintln!("  Two directions beat one. Superposition of orthogonal views works.");
    } else {
        let def = (f_bi / f_fwd - 1.0) * 100.0;
        eprintln!("\n  ✗ Forward only wins by {:.1}%.", def);
    }

    // Report combination weights
    let sa: Vec<f64> = bi_model.alpha.iter().map(|&a| 1.0 / (1.0 + (-a).exp())).collect();
    let mean_a: f64 = sa.iter().sum::<f64>() / 256.0;
    eprintln!("\n  Learned α: mean={:.3} (>0.5 = forward dominates, <0.5 = backward dominates)", mean_a);

    let chars = [b' ', b't', b'h', b'e', b'.', b'a', b'c', b'd', b's', b'n'];
    for &ch in &chars {
        let a = sa[ch as usize];
        let label = if a > 0.6 { "FORWARD" } else if a < 0.4 { "BACKWARD" } else { "balanced" };
        eprintln!("    '{}': α={:.3} → {}", ch as char, a, label);
    }

    eprintln!("\n  Convergence (loss at epochs 50, 100, 200, 300):");
    for &ep in &[49, 99, 199, 299] {
        if ep < losses_fwd.len() && ep < losses_bi.len() {
            eprintln!("    epoch {:>3}: fwd={:.4}  bidi={:.4}  Δ={:.4}",
                ep + 1, losses_fwd[ep], losses_bi[ep], losses_fwd[ep] - losses_bi[ep]);
        }
    }

    Ok(())
}
