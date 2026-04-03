//! # Experiment 2: Multi-Resolution Dimensional Ladder
//!
//! One encoder, seven resolution rungs via prefix slicing.
//! Each rung sees the same hidden state at different resolutions.
//! FMM-style combination learns which resolution each character needs.
//!
//! Like fintek cadences: 1s, 5s, 30s, 1min... each captures different structure.
//! Like wavelets: coarse + detail + finer detail.
//! The combination weights ARE the multi-resolution analysis of the data.

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

// ─── Single-rung baseline ───────────────────────────────────────────

struct SingleModel {
    w1: Vec<f64>,
    w2: Vec<f64>,
    h: usize, v: usize, ctx: usize, d: usize,
}

impl SingleModel {
    fn new(h: usize, ctx: usize) -> Self {
        let v = 256; let d = v * ctx;
        let mut seed = 42u64;
        Self {
            w1: init_weights(d * h, d, h, &mut seed),
            w2: init_weights(h * v, h, v, &mut seed),
            h, v, ctx, d,
        }
    }
    fn params(&self) -> usize { self.d * self.h + self.h * self.v }

    fn train(&mut self, tiled: &TiledEngine, text: &[u8], epochs: usize, lr: f64, bs_max: usize)
        -> Result<Vec<f64>, Box<dyn std::error::Error>>
    {
        let (v, h, d) = (self.v, self.h, self.d);
        let n = text.len() - 1;
        let mut losses = Vec::new();
        for epoch in 0..epochs {
            let mut el = 0.0; let mut nb = 0;
            for bs0 in (0..n).step_by(bs_max) {
                let bs = (bs0 + bs_max).min(n) - bs0;
                if bs < 2 { continue; }
                let mut x = vec![0.0f64; bs * d];
                let mut tgt = vec![0usize; bs];
                for i in 0..bs {
                    let pos = bs0 + i + 1;
                    let cv = encode_ctx(text, pos, self.ctx, v);
                    x[i * d..(i + 1) * d].copy_from_slice(&cv);
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
                eprintln!("  [single] epoch {}/{}: loss={:.4}", epoch + 1, epochs, avg);
            }
        }
        Ok(losses)
    }
}

// ─── Multi-resolution ladder model ──────────────────────────────────

struct LadderModel {
    w_enc: Vec<f64>,        // d × h_full (one shared encoder)
    heads: Vec<Vec<f64>>,   // one W_head per rung: h_rung × vocab
    alphas: Vec<Vec<f64>>,  // per-rung per-output-byte combination weight
    rungs: Vec<usize>,      // hidden dims for each rung (prefix sizes)
    h_full: usize,          // largest hidden dim
    v: usize,
    ctx: usize,
    d: usize,
    n_rungs: usize,
}

impl LadderModel {
    fn new(h_full: usize, ctx: usize) -> Self {
        let v = 256;
        let d = v * ctx;
        let mut seed = 42u64;

        // Rungs: h_full, h_full/2, h_full/4, ..., down to 16 minimum
        let mut rungs = Vec::new();
        let mut h = h_full;
        while h >= 16 {
            rungs.push(h);
            h /= 2;
        }
        let n_rungs = rungs.len();

        let w_enc = init_weights(d * h_full, d, h_full, &mut seed);

        // One head per rung, each head reads first h_rung dims of hidden
        let mut heads = Vec::new();
        for &rung_h in &rungs {
            heads.push(init_weights(rung_h * v, rung_h, v, &mut seed));
        }

        // Combination weights: per-rung per-output (initialize to equal = 0)
        let alphas = vec![vec![0.0f64; v]; n_rungs];

        Self { w_enc, heads, alphas, rungs, h_full, v, ctx, d, n_rungs }
    }

    fn params(&self) -> usize {
        let enc = self.d * self.h_full;
        let heads: usize = self.rungs.iter().zip(&self.heads).map(|(h, w)| h * self.v).sum();
        let alphas = self.n_rungs * self.v;
        enc + heads + alphas
    }

    fn train(&mut self, tiled: &TiledEngine, text: &[u8], epochs: usize, lr: f64, bs_max: usize)
        -> Result<Vec<f64>, Box<dyn std::error::Error>>
    {
        let (v, d, hf) = (self.v, self.d, self.h_full);
        let n = text.len() - 1;
        let nr = self.n_rungs;
        let mut losses = Vec::new();

        for epoch in 0..epochs {
            let mut el = 0.0; let mut nb = 0;
            for bs0 in (0..n).step_by(bs_max) {
                let bs = (bs0 + bs_max).min(n) - bs0;
                if bs < 2 { continue; }

                let mut x = vec![0.0f64; bs * d];
                let mut tgt = vec![0usize; bs];
                for i in 0..bs {
                    let pos = bs0 + i + 1;
                    let cv = encode_ctx(text, pos, self.ctx, v);
                    x[i * d..(i + 1) * d].copy_from_slice(&cv);
                    tgt[i] = text[pos] as usize;
                }

                // ── Shared encoder: full hidden = ReLU(X @ W_enc) ────────
                let hr = tiled.run(&DotProductOp, &x, &self.w_enc, bs, hf, d)?;
                let mut ha_full = vec![0.0f64; bs * hf];
                let mut rm_full = vec![0.0f64; bs * hf];
                for i in 0..bs * hf {
                    if hr[i] > 0.0 { ha_full[i] = hr[i]; rm_full[i] = 1.0; }
                }

                // ── Each rung: slice hidden prefix → head → logits ───────
                let mut all_logits: Vec<Vec<f64>> = Vec::new();
                for r in 0..nr {
                    let rh = self.rungs[r];
                    // Slice first rh dims of each sample's hidden
                    let mut ha_slice = vec![0.0f64; bs * rh];
                    for i in 0..bs {
                        ha_slice[i * rh..(i + 1) * rh]
                            .copy_from_slice(&ha_full[i * hf..i * hf + rh]);
                    }
                    let logits_r = tiled.run(&DotProductOp, &ha_slice, &self.heads[r], bs, v, rh)?;
                    all_logits.push(logits_r);
                }

                // ── Softmax combination weights per rung ─────────────────
                let mut rung_weights = vec![vec![0.0f64; v]; nr]; // softmax(alphas) per byte
                for j in 0..v {
                    let mut max_a = f64::NEG_INFINITY;
                    for r in 0..nr { max_a = max_a.max(self.alphas[r][j]); }
                    let mut sum_exp = 0.0f64;
                    for r in 0..nr {
                        rung_weights[r][j] = (self.alphas[r][j] - max_a).exp();
                        sum_exp += rung_weights[r][j];
                    }
                    for r in 0..nr { rung_weights[r][j] /= sum_exp; }
                }

                // ── Combined logits ──────────────────────────────────────
                let mut logits = vec![0.0f64; bs * v];
                for i in 0..bs {
                    for j in 0..v {
                        for r in 0..nr {
                            logits[i * v + j] += rung_weights[r][j] * all_logits[r][i * v + j];
                        }
                    }
                }

                let (probs, bl) = softmax_ce(&logits, &tgt, v, bs);
                el += bl; nb += 1;

                // ── Backward ─────────────────────────────────────────────
                let mut dc = probs;
                for i in 0..bs { dc[i * v + tgt[i]] -= 1.0; for j in 0..v { dc[i * v + j] /= bs as f64; } }

                // Gradient for alphas (softmax combination)
                let mut grad_alphas = vec![vec![0.0f64; v]; nr];
                for j in 0..v {
                    // d_loss/d_alpha_r = sum_i dc[i,j] * (logits_r[i,j] - combined_logits[i,j]) * w_r
                    // This is the softmax-of-experts gradient
                    for r in 0..nr {
                        let wr = rung_weights[r][j];
                        for i in 0..bs {
                            let diff = all_logits[r][i * v + j] - logits[i * v + j];
                            grad_alphas[r][j] += dc[i * v + j] * diff * wr;
                        }
                        grad_alphas[r][j] /= bs as f64;
                    }
                }

                // Split gradient to each rung's head
                let mut enc_grad_total = vec![0.0f64; bs * hf]; // accumulate encoder grad from all rungs

                for r in 0..nr {
                    let rh = self.rungs[r];

                    // d_rung = dc * rung_weight (scale gradient by this rung's importance)
                    let mut d_rung = vec![0.0f64; bs * v];
                    for i in 0..bs {
                        for j in 0..v {
                            d_rung[i * v + j] = dc[i * v + j] * rung_weights[r][j];
                        }
                    }

                    // Grad head: ha_slice' @ d_rung
                    let mut ha_slice = vec![0.0f64; bs * rh];
                    for i in 0..bs {
                        ha_slice[i * rh..(i + 1) * rh]
                            .copy_from_slice(&ha_full[i * hf..i * hf + rh]);
                    }
                    let hst = transpose(&ha_slice, bs, rh);
                    let gh = tiled.run(&DotProductOp, &hst, &d_rung, rh, v, bs)?;

                    // Backprop through head to hidden slice
                    let ht = transpose(&self.heads[r], rh, v);
                    let dhs = tiled.run(&DotProductOp, &d_rung, &ht, bs, rh, v)?;

                    // Accumulate into encoder gradient (only first rh dims of each sample)
                    for i in 0..bs {
                        for j in 0..rh {
                            enc_grad_total[i * hf + j] += dhs[i * rh + j] * rm_full[i * hf + j];
                        }
                    }

                    // Update head weights
                    for i in 0..self.heads[r].len() { self.heads[r][i] -= lr * gh[i]; }
                }

                // Grad encoder: X' @ enc_grad_total
                let xt = transpose(&x, bs, d);
                let genc = tiled.run(&DotProductOp, &xt, &enc_grad_total, d, hf, bs)?;

                // Update encoder + alphas
                for i in 0..self.w_enc.len() { self.w_enc[i] -= lr * genc[i]; }
                for r in 0..nr {
                    for j in 0..v { self.alphas[r][j] -= lr * grad_alphas[r][j]; }
                }
            }

            let avg = if nb > 0 { el / nb as f64 } else { f64::NAN };
            losses.push(avg);
            if epoch % 50 == 0 || epoch == epochs - 1 {
                eprintln!("  [ladder] epoch {}/{}: loss={:.4}", epoch + 1, epochs, avg);
            }
        }
        Ok(losses)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  EXPERIMENT 2: Multi-Resolution Dimensional Ladder");
    eprintln!("  One encoder, 7 resolution rungs via prefix slicing");
    eprintln!("  Hypothesis: multi-resolution > single resolution");
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

    // ── Model A: Single rung, hidden=128 ────────────────────────────
    let mut single = SingleModel::new(128, 8);
    eprintln!("Model A (single h=128): {} params", single.params());

    // ── Model B: Ladder, h_full=128, rungs=[128,64,32,16] ──────────
    let mut ladder = LadderModel::new(128, 8);
    eprintln!("Model B (ladder h=128→16): {} params, {} rungs {:?}",
        ladder.params(), ladder.n_rungs, ladder.rungs);
    for (r, &rh) in ladder.rungs.iter().enumerate() {
        eprintln!("  Rung {}: h={}, head={} params", r, rh, rh * 256);
    }

    eprintln!("\nTraining both for {} epochs\n", epochs);

    eprintln!("── Training Model A (single resolution) ──");
    let losses_s = single.train(&tiled, text, epochs, lr, batch)?;

    eprintln!("\n── Training Model B (multi-resolution ladder) ──");
    let losses_l = ladder.train(&tiled, text, epochs, lr, batch)?;

    let fs = losses_s.last().unwrap();
    let fl = losses_l.last().unwrap();

    eprintln!("\n══════════════════════════════════════════════════════");
    eprintln!("  RESULTS");
    eprintln!("══════════════════════════════════════════════════════");
    eprintln!("  Single resolution: loss = {:.4} ({} params)", fs, single.params());
    eprintln!("  Ladder resolution: loss = {:.4} ({} params)", fl, ladder.params());

    if fl < fs {
        let imp = (1.0 - fl / fs) * 100.0;
        eprintln!("\n  ✓ LADDER WINS by {:.1}%!", imp);
    } else {
        let def = (fl / fs - 1.0) * 100.0;
        eprintln!("\n  ✗ Single wins by {:.1}%.", def);
    }

    // ── Resolution fingerprint ───────────────────────────────────────
    eprintln!("\n  Resolution fingerprint (which rung dominates per character):");
    let chars = [b' ', b't', b'h', b'e', b'.', b'a', b'c', b'd', b's', b'o'];
    for &ch in &chars {
        let j = ch as usize;
        let mut max_a = f64::NEG_INFINITY;
        for r in 0..ladder.n_rungs { max_a = max_a.max(ladder.alphas[r][j]); }
        let mut sum = 0.0f64;
        let mut weights = Vec::new();
        for r in 0..ladder.n_rungs {
            let w = (ladder.alphas[r][j] - max_a).exp();
            weights.push(w);
            sum += w;
        }
        for w in weights.iter_mut() { *w /= sum; }
        let dominant = weights.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        let rung_h = ladder.rungs[dominant.0];
        let weight_strs: Vec<String> = weights.iter().zip(&ladder.rungs)
            .map(|(w, h)| format!("h{}:{:.0}%", h, w * 100.0))
            .collect();
        eprintln!("    '{}': {} → dominant=h{} ({:.0}%)",
            ch as char, weight_strs.join(" "), rung_h, dominant.1 * 100.0);
    }

    // ── Convergence comparison ───────────────────────────────────────
    eprintln!("\n  Convergence (loss at epochs 50, 100, 200, 300):");
    for &ep in &[49, 99, 199, 299] {
        if ep < losses_s.len() && ep < losses_l.len() {
            eprintln!("    epoch {:>3}: single={:.4}  ladder={:.4}  Δ={:.4}",
                ep + 1, losses_s[ep], losses_l[ep], losses_s[ep] - losses_l[ep]);
        }
    }

    Ok(())
}
