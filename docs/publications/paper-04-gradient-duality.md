# Paper 4: Backpropagation as Transposed Tiled Accumulation

## Target
ML theory: JMLR, NeurIPS, or ICML.

## Core Claim
Forward and backward passes in neural networks are the SAME tiled accumulate operation with transposed arguments. The chain rule is closed under accumulate. No autodiff framework needed.

## Outline
1. The standard model: forward → tape → backward → optimizer = 3 separate passes, autograd required
2. The duality: forward = accumulate(Tiled{X,W}, DotProduct), backward = accumulate(Tiled{X',δ}, DotProduct). Same op, transposed.
3. Activations as element-wise masks that fuse into the next accumulate (not separate operations)
4. Proof: 2-layer neural net trains via 5 DotProduct calls (2L+1 for L layers). Verified on CPU and GPU.
5. Newton's method: same + one extra DotProduct for Hessian (not "double-transposed" — fresh DotProduct on weighted data)
6. Extension: BatchNorm needs Grouping::All (reduce across batch), LayerNorm needs Grouping::Segmented — both in the accumulate taxonomy
7. Implications: tb.train() needs no PyTorch, no TensorFlow, no JAX. Just accumulate.

## Evidence
- Pathmaker: experiment0.rs (working neural net via pure DotProduct), train/logistic.rs
- Observer: gradient_duality_on_cpu_backend test (structure not substrate)
- Math researcher: F07 verification (gradient formulas correct)
- Lab notebook 005 (pathmaker): formal theorem statement + corollary
