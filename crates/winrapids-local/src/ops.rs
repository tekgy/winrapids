//! Local context features — fused multi-offset gather + feature computation.
//!
//! For each position i, gather values at fixed offsets and compute features
//! from the gathered neighborhood. Everything in one kernel, one read.
//!
//! This replaces:
//! - shift(n) for each lag → one gather
//! - diff(n) for each lag → one gather + one subtract
//! - rolling(w).mean/std → one gather + fused reduction
//! - peak detection → one gather + fused comparison
//!
//! The feature functions are composable: pick any subset of features,
//! the kernel generates code for exactly those, no wasted computation.

/// A feature computed from the local context (gathered neighborhood).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LocalFeature {
    /// Raw gathered value at offset k: data[i + offset_k]
    RawValue { offset_idx: usize },

    /// Delta: data[i] - data[i + offset]
    Delta { offset_idx: usize },

    /// Log ratio: log(data[i] / data[i + offset])
    LogRatio { offset_idx: usize },

    /// Direction: sign(data[i] - data[i + offset]) → {-1, 0, +1}
    Direction { offset_idx: usize },

    /// Local mean over all gathered offsets
    LocalMean,

    /// Local std over all gathered offsets
    LocalStd,

    /// Slope: linear regression slope over ordered offsets
    Slope,

    /// Peak detection: 1.0 if data[i] > all gathered neighbors, -1.0 if < all, else 0.0
    PeakDetect,
}

impl LocalFeature {
    /// CUDA expression for this feature.
    ///
    /// Variables in scope:
    /// - `center`: data[i] (the center value)
    /// - `vals[k]`: gathered value at offset k (k = 0..n_offsets-1)
    /// - `n_offsets`: number of offsets (compile-time constant)
    /// - `offsets[k]`: the offset values (compile-time constants)
    pub fn cuda_expr(&self, _n_offsets: usize, _offsets: &[i32]) -> String {
        match self {
            LocalFeature::RawValue { offset_idx } => {
                format!("vals[{}]", offset_idx)
            }
            LocalFeature::Delta { offset_idx } => {
                format!("(center - vals[{}])", offset_idx)
            }
            LocalFeature::LogRatio { offset_idx } => {
                format!("(vals[{k}] > 0.0 ? log(center / vals[{k}]) : 0.0)", k = offset_idx)
            }
            LocalFeature::Direction { offset_idx } => {
                format!("(center > vals[{k}] ? 1.0 : (center < vals[{k}] ? -1.0 : 0.0))", k = offset_idx)
            }
            LocalFeature::LocalMean => {
                format!("local_mean")
            }
            LocalFeature::LocalStd => {
                format!("local_std")
            }
            LocalFeature::Slope => {
                // Linear regression slope: sum((x_k - x_mean) * (y_k - y_mean)) / sum((x_k - x_mean)^2)
                // where x_k = offsets[k], y_k = vals[k]
                format!("slope")
            }
            LocalFeature::PeakDetect => {
                format!("peak")
            }
        }
    }

    /// Name for this feature (used in output column naming).
    pub fn name(&self, offsets: &[i32]) -> String {
        match self {
            LocalFeature::RawValue { offset_idx } => format!("val_at_{}", offsets[*offset_idx]),
            LocalFeature::Delta { offset_idx } => format!("delta_{}", offsets[*offset_idx]),
            LocalFeature::LogRatio { offset_idx } => format!("logratio_{}", offsets[*offset_idx]),
            LocalFeature::Direction { offset_idx } => format!("dir_{}", offsets[*offset_idx]),
            LocalFeature::LocalMean => "local_mean".into(),
            LocalFeature::LocalStd => "local_std".into(),
            LocalFeature::Slope => "slope".into(),
            LocalFeature::PeakDetect => "peak".into(),
        }
    }
}

/// A local context specification: offsets + which features to compute.
#[derive(Clone, Debug)]
pub struct LocalContextSpec {
    /// Fixed offsets to gather. e.g. [-10, -5, -3, -1, 0, 1, 3, 5, 10]
    pub offsets: Vec<i32>,
    /// Features to compute from the gathered neighborhood.
    pub features: Vec<LocalFeature>,
}

impl LocalContextSpec {
    /// Number of output columns.
    pub fn output_width(&self) -> usize {
        self.features.len()
    }

    /// Whether any feature needs local_mean (so we compute it once).
    pub fn needs_local_mean(&self) -> bool {
        self.features.iter().any(|f| matches!(f,
            LocalFeature::LocalMean | LocalFeature::LocalStd | LocalFeature::Slope))
    }

    /// Whether any feature needs local_std.
    pub fn needs_local_std(&self) -> bool {
        self.features.iter().any(|f| matches!(f, LocalFeature::LocalStd))
    }

    /// Whether any feature needs slope.
    pub fn needs_slope(&self) -> bool {
        self.features.iter().any(|f| matches!(f, LocalFeature::Slope))
    }

    /// Whether peak detection is needed.
    pub fn needs_peak(&self) -> bool {
        self.features.iter().any(|f| matches!(f, LocalFeature::PeakDetect))
    }

    /// CSE identity key: offsets + feature set determine the kernel.
    pub fn identity_key(&self) -> String {
        let off_str: Vec<String> = self.offsets.iter().map(|o| o.to_string()).collect();
        let feat_str: Vec<String> = self.features.iter().map(|f| format!("{:?}", f)).collect();
        format!("local_context:off=[{}]:feat=[{}]", off_str.join(","), feat_str.join(","))
    }
}
