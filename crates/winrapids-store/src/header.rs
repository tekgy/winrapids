//! 64-byte self-describing buffer header.
//!
//! Every buffer in the persistent store carries this header.
//! The header IS the buffer's identity — reading it tells you
//! what the data is, where it lives, how expensive it was to
//! compute, and how often it's been used.
//!
//! Cache-line aligned (64 bytes). No padding surprises.

/// Storage tier for a buffer.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    /// GPU VRAM (hot path — zero-copy access).
    Gpu = 0,
    /// Pinned host memory (fast re-promote to GPU).
    Pinned = 1,
    /// Pageable host memory (slower re-promote).
    Cpu = 2,
    /// On disk (cold storage, full reload required).
    Disk = 3,
}

/// Element data type.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32 = 0,
    F64 = 1,
    I32 = 2,
    I64 = 3,
    U32 = 4,
    U64 = 5,
    BF16 = 6,
    F16 = 7,
}

impl DType {
    /// Size of one element in bytes.
    pub fn byte_size(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::BF16 | DType::F16 => 2,
        }
    }
}

/// 64-byte buffer header. Cache-line aligned.
///
/// Layout (repr(C)):
///   [0..16)   provenance    [u8; 16]   BLAKE3 truncated to 128 bits
///   [16..20)  cost_us       f32        microseconds to recompute
///   [20..24)  access_count  u32        times accessed via lookup
///   [24]      location      u8         storage tier
///   [25]      dtype         u8         element type
///   [26]      ndim          u8         tensor rank
///   [27]      flags         u8         reserved
///   [28..32)  _align        [u8; 4]    alignment padding for u64 fields
///   [32..40)  len           u64        element count
///   [40..48)  byte_size     u64        data bytes (excluding header)
///   [48..56)  created_ns    u64        creation timestamp
///   [56..64)  last_access   u64        last access timestamp
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BufferHeader {
    /// BLAKE3 provenance hash truncated to 128 bits.
    /// Identity = hash(input_provenances + computation_identity).
    /// Same inputs + same computation = same hash = reuse.
    pub provenance: [u8; 16],

    /// Estimated cost to recompute this buffer, in microseconds.
    /// Drives cost-aware eviction: cheap buffers evicted first.
    pub cost_us: f32,

    /// Number of times this buffer has been accessed via lookup().
    pub access_count: u32,

    /// Current storage tier.
    pub location: Location,

    /// Element data type.
    pub dtype: DType,

    /// Number of dimensions (1=vector, 2=matrix).
    pub ndim: u8,

    /// Reserved flags.
    pub flags: u8,

    /// Explicit alignment padding for the u64 fields that follow.
    pub _align: [u8; 4],

    /// Number of elements.
    pub len: u64,

    /// Total data bytes (excluding this header).
    pub byte_size: u64,

    /// Nanoseconds since epoch when this buffer was created.
    pub created_ns: u64,

    /// Nanoseconds since epoch when this buffer was last accessed.
    pub last_access_ns: u64,
}

impl BufferHeader {
    /// Create a new header for a freshly computed buffer.
    pub fn new(
        provenance: [u8; 16],
        cost_us: f32,
        dtype: DType,
        len: u64,
    ) -> Self {
        let now = now_ns();
        Self {
            provenance,
            cost_us,
            access_count: 0,
            location: Location::Gpu,
            dtype,
            ndim: 1,
            flags: 0,
            _align: [0; 4],
            len,
            byte_size: len * dtype.byte_size() as u64,
            created_ns: now,
            last_access_ns: now,
        }
    }
}

/// A raw device pointer + size. The zero-translation cache entry.
///
/// This IS the consumer's input. No copy. Just hand off the pointer.
/// result === cache entry === consumer input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferPtr {
    /// Raw CUDA device pointer (CUdeviceptr).
    pub device_ptr: u64,
    /// Total bytes of data at this pointer.
    pub byte_size: u64,
}

/// An entry that was evicted from the store.
/// The caller is responsible for freeing or spilling the GPU memory.
#[derive(Clone, Copy, Debug)]
pub struct EvictedEntry {
    pub provenance: [u8; 16],
    pub ptr: BufferPtr,
    pub cost_us: f32,
    pub location: Location,
}

/// Current time in nanoseconds since epoch.
pub fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
