use std::fmt;

/// All errors produced by the tam-gpu runtime.
#[derive(Debug)]
pub enum TamGpuError {
    /// Backend initialisation or driver error.
    Backend(String),
    /// Shader / NVRTC compilation failure.
    Compile(String),
    /// Device memory allocation failure.
    Alloc(String),
    /// Host ↔ device transfer error.
    Transfer(String),
    /// Kernel dispatch error (wrong buffer count, launch failure).
    Dispatch(String),
    /// The requested entry point has no CPU implementation.
    EntryNotFound(String),
    /// Bad argument (buffer too small, index out of range, etc.).
    InvalidArgument(String),
}

impl fmt::Display for TamGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(s)         => write!(f, "backend: {s}"),
            Self::Compile(s)         => write!(f, "compile: {s}"),
            Self::Alloc(s)           => write!(f, "alloc: {s}"),
            Self::Transfer(s)        => write!(f, "transfer: {s}"),
            Self::Dispatch(s)        => write!(f, "dispatch: {s}"),
            Self::EntryNotFound(s)   => write!(f, "entry not found: {s}"),
            Self::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
        }
    }
}

impl std::error::Error for TamGpuError {}

impl From<Box<dyn std::error::Error>> for TamGpuError {
    fn from(e: Box<dyn std::error::Error>) -> Self {
        Self::Backend(e.to_string())
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for TamGpuError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self::Backend(e.to_string())
    }
}
