//! MKTF v4 format — constants, header layout, pack/unpack.
//!
//! Block 0 (4096 bytes): Fixed header. One NVMe sector read.
//! Block 1+: Byte range directory (128 bytes per entry).
//! Optional: JSON metadata block.
//! Data: Column arrays, each alignment-byte aligned.
//! Trailing: 2 status bytes at EOF (is_complete, is_dirty).
//!
//! All fields fixed-offset. Little-endian. memcpy + cast.

use sha2::{Sha256, Digest};

// ── Magic & Version ──────────────────────────────────────────────

pub const MAGIC: &[u8; 4] = b"MKTF";
pub const FORMAT_VERSION: u16 = 4;
pub const ALIGNMENT: u16 = 4096;
pub const BLOCK_SIZE: usize = 4096;

// ── Flag Bits (uint32 at offset 6) ──────────────────────────────

pub const FLAG_HAS_METADATA: u32   = 0x0001;
pub const FLAG_HAS_QUALITY: u32    = 0x0002;
pub const FLAG_HAS_STATISTICS: u32 = 0x0004;
pub const FLAG_HAS_SPATIAL: u32    = 0x0008;
pub const FLAG_HAS_ASSET: u32      = 0x0010;
pub const FLAG_HAS_TEMPORAL: u32   = 0x0020;
pub const FLAG_HAS_UPSTREAM: u32   = 0x0040;
pub const FLAG_IS_VALIDATED: u32   = 0x0080;
pub const FLAG_IS_COMPRESSED: u32  = 0x0100;
pub const FLAG_GPU_RESIDENT: u32   = 0x0200;

// ── Dtype Codes ──────────────────────────────────────────────────

pub const DTYPE_FLOAT32: u8 = 0;
pub const DTYPE_FLOAT64: u8 = 1;
pub const DTYPE_INT32: u8   = 2;
pub const DTYPE_INT64: u8   = 3;
pub const DTYPE_UINT8: u8   = 4;
pub const DTYPE_INT8: u8    = 5;
pub const DTYPE_UINT32: u8  = 6;
pub const DTYPE_UINT16: u8  = 7;
pub const DTYPE_UINT64: u8  = 8;
pub const DTYPE_INT16: u8   = 9;
pub const DTYPE_BOOL: u8    = 10;

/// Byte size of each dtype code.
pub fn dtype_size(code: u8) -> usize {
    match code {
        DTYPE_FLOAT32 => 4,
        DTYPE_FLOAT64 => 8,
        DTYPE_INT32   => 4,
        DTYPE_INT64   => 8,
        DTYPE_UINT8   => 1,
        DTYPE_INT8    => 1,
        DTYPE_UINT32  => 4,
        DTYPE_UINT16  => 2,
        DTYPE_UINT64  => 8,
        DTYPE_INT16   => 2,
        DTYPE_BOOL    => 1,
        _ => 1,
    }
}

// ── Compression Algorithm ────────────────────────────────────────

pub const COMPRESS_NONE: u8       = 0;
pub const COMPRESS_LZ4: u8        = 1;
pub const COMPRESS_ZSTD: u8       = 2;
pub const COMPRESS_LZ4_HC: u8     = 3;
pub const COMPRESS_SNAPPY: u8     = 4;
pub const COMPRESS_BROTLI: u8     = 5;
pub const COMPRESS_CUSTOM_01: u8  = 64;

// ── Pre-Filter ───────────────────────────────────────────────────

pub const FILTER_NONE: u8          = 0;
pub const FILTER_SHUFFLE: u8       = 1;
pub const FILTER_DELTA: u8         = 2;
pub const FILTER_DELTA_SHUFFLE: u8 = 3;
pub const FILTER_XOR: u8           = 4;
pub const FILTER_XOR_SHUFFLE: u8   = 5;
pub const FILTER_BITSHUFFLE: u8    = 6;

// ── Block 0 Section Offsets ──────────────────────────────────────

pub const OFF_FORMAT: usize     = 0;       // [0:16)
pub const OFF_IDENTITY: usize   = 16;      // [16:128)
pub const OFF_TREE: usize       = 128;     // [128:176)
pub const OFF_DIMENSIONS: usize = 176;     // [176:240)
pub const OFF_TEMPORAL: usize   = 240;     // [240:288)
pub const OFF_QUALITY: usize    = 288;     // [288:368)
pub const OFF_PROVENANCE: usize = 368;     // [368:448)
pub const OFF_LAYOUT: usize     = 448;     // [448:512)
pub const OFF_ASSET: usize      = 512;     // [512:576)
pub const OFF_UPSTREAM: usize   = 576;     // [576:1600)
pub const OFF_STATISTICS: usize = 1600;    // [1600:1728)
pub const OFF_SPATIAL: usize    = 1728;    // [1728:1856)
pub const OFF_STATUS: usize     = 4094;    // [4094:4096)

pub const MAX_UPSTREAM: usize = 16;
pub const UPSTREAM_ENTRY_SIZE: usize = 64;
pub const ENTRY_SIZE: usize = 128;

/// Offset of data_checksum within Block 0.
pub const DATA_CHECKSUM_OFFSET: usize = 332;

// ── Dataclasses ──────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct UpstreamFingerprint {
    pub leaf_id: String,
    pub write_ts_ns: i64,
    pub data_hash: u64,
    pub ti: u8,
    pub to: u8,
}

#[derive(Clone, Debug)]
pub struct ByteEntry {
    // Identity + layout [0:72)
    pub name: String,
    pub dtype_code: u8,
    pub n_elements: u64,
    pub data_offset: u64,
    pub data_nbytes: u64,
    pub null_count: u64,
    // Statistics [72:112)
    pub min_value: f64,
    pub max_value: f64,
    pub mean_value: f64,
    pub scale_factor: f64,
    pub sentinel_value: f64,
    // Compression descriptor [112:128)
    pub compression_algo: u8,
    pub pre_filter: u8,
    pub typesize: u8,
    pub filter_param: u8,
    pub compressed_size: u64,
}

impl Default for ByteEntry {
    fn default() -> Self {
        Self {
            name: String::new(),
            dtype_code: DTYPE_FLOAT32,
            n_elements: 0,
            data_offset: 0,
            data_nbytes: 0,
            null_count: 0,
            min_value: f64::NAN,
            max_value: f64::NAN,
            mean_value: f64::NAN,
            scale_factor: 1.0,
            sentinel_value: f64::NAN,
            compression_algo: COMPRESS_NONE,
            pre_filter: FILTER_NONE,
            typesize: 4,
            filter_param: 0,
            compressed_size: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MktfHeader {
    // Format
    pub format_version: u16,
    pub flags: u32,
    pub alignment: u16,
    pub header_blocks: u16,

    // Identity
    pub leaf_id: String,
    pub ticker: String,
    pub day: String,
    pub ti: u8,
    pub to: u8,
    pub leaf_version: String,
    pub schema_fingerprint: [u8; 16],
    pub leaf_id_hash: u32,

    // Tree
    pub kingdom: u8,
    pub phylum: u8,
    pub class: u8,
    pub rank: u8,
    pub family: u8,
    pub genus: u8,
    pub species: u8,
    pub depth: u8,
    pub computation_shape: u8,
    pub compute_target: u8,
    pub precision: u8,
    pub leaf_type: u8,
    pub n_upstream: u32,
    pub ki: u8,
    pub ko: u8,
    pub domain: u8,

    // Dimensions
    pub n_rows: u64,
    pub n_cols: u16,
    pub n_bins: u32,
    pub n_cadences: u32,
    pub n_tickers: u32,
    pub bytes_data: u64,
    pub bytes_file: u64,

    // Temporal
    pub ts_first_ns: i64,
    pub ts_last_ns: i64,
    pub ts_range_ns: i64,
    pub market_open_ns: i64,
    pub market_close_ns: i64,

    // Quality
    pub total_nulls: u64,
    pub total_nans: u64,
    pub total_infs: u64,
    pub null_ppm: u32,
    pub effective_rank: f64,
    pub compression_ratio: f64,
    pub data_checksum: u64,

    // Provenance
    pub write_timestamp_ns: i64,
    pub write_duration_ms: u32,
    pub compute_duration_ms: u32,
    pub rewrite_count: u32,
    pub engine_version: u32,
    pub source_data_hash: u64,
    pub compute_host: String,
    pub cuda_version: u32,
    pub driver_version: u32,

    // Layout
    pub dir_offset: u64,
    pub dir_entries: u64,
    pub meta_offset: u64,
    pub meta_size: u64,
    pub data_start: u64,

    // Asset
    pub exchange_code: u16,
    pub asset_class: u8,
    pub universe_tier: u8,
    pub tick_count: u32,
    pub market_cap_tier: u32,
    pub avg_spread_e8: i64,

    // Upstream
    pub upstream: Vec<UpstreamFingerprint>,

    // Statistics
    pub global_mean: f64,
    pub global_std: f64,
    pub global_min: f64,
    pub global_max: f64,
    pub global_median: f64,
    pub global_skew: f64,
    pub global_kurtosis: f64,
    pub global_entropy: f64,

    // Spatial
    pub spatial_dims: u32,
    pub atlas_version: u32,
    pub coordinates: [f64; 8],

    // Status
    pub is_complete: bool,
    pub is_dirty: bool,

    // Byte range directory (not in Block 0)
    pub columns: Vec<ByteEntry>,
}

impl Default for MktfHeader {
    fn default() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            flags: 0,
            alignment: ALIGNMENT,
            header_blocks: 0,
            leaf_id: String::new(),
            ticker: String::new(),
            day: String::new(),
            ti: 0,
            to: 0,
            leaf_version: String::from("1.0.0"),
            schema_fingerprint: [0u8; 16],
            leaf_id_hash: 0,
            kingdom: 0, phylum: 0, class: 0, rank: 0,
            family: 0, genus: 0, species: 0, depth: 0,
            computation_shape: 0, compute_target: 0,
            precision: 1, // FP32
            leaf_type: 0,
            n_upstream: 0,
            ki: 0, ko: 0, domain: 0,
            n_rows: 0, n_cols: 0, n_bins: 0,
            n_cadences: 0, n_tickers: 0,
            bytes_data: 0, bytes_file: 0,
            ts_first_ns: 0, ts_last_ns: 0, ts_range_ns: 0,
            market_open_ns: 0, market_close_ns: 0,
            total_nulls: 0, total_nans: 0, total_infs: 0,
            null_ppm: 0,
            effective_rank: 0.0, compression_ratio: 0.0,
            data_checksum: 0,
            write_timestamp_ns: 0, write_duration_ms: 0,
            compute_duration_ms: 0, rewrite_count: 0,
            engine_version: 0, source_data_hash: 0,
            compute_host: String::new(),
            cuda_version: 0, driver_version: 0,
            dir_offset: 0, dir_entries: 0,
            meta_offset: 0, meta_size: 0, data_start: 0,
            exchange_code: 0, asset_class: 0, universe_tier: 0,
            tick_count: 0, market_cap_tier: 0, avg_spread_e8: 0,
            upstream: Vec::new(),
            global_mean: f64::NAN, global_std: f64::NAN,
            global_min: f64::NAN, global_max: f64::NAN,
            global_median: f64::NAN, global_skew: f64::NAN,
            global_kurtosis: f64::NAN, global_entropy: f64::NAN,
            spatial_dims: 0, atlas_version: 0,
            coordinates: [0.0; 8],
            is_complete: false, is_dirty: false,
            columns: Vec::new(),
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Null-pad a string to fixed width.
fn pad(s: &str, size: usize) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut out = vec![0u8; size];
    let copy_len = bytes.len().min(size);
    out[..copy_len].copy_from_slice(&bytes[..copy_len]);
    out
}

/// Decode null-padded bytes to string.
fn unpad(b: &[u8]) -> String {
    let end = b.iter().position(|&x| x == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

/// Round up to next alignment boundary.
pub fn align(offset: u64, boundary: u64) -> u64 {
    (offset + boundary - 1) & !(boundary - 1)
}

/// SHA-256 first 4 bytes as u32 LE.
pub fn compute_leaf_id_hash(leaf_id: &str) -> u32 {
    let hash = Sha256::digest(leaf_id.as_bytes());
    u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]])
}

/// SHA-256 of sorted (name, dtype_code) pairs, first 16 bytes.
pub fn compute_schema_fingerprint(columns: &[(String, u8)]) -> [u8; 16] {
    let mut sorted: Vec<_> = columns.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hasher = Sha256::new();
    for (name, dtype_code) in &sorted {
        hasher.update(name.as_bytes());
        hasher.update([*dtype_code]);
    }
    let hash = hasher.finalize();
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash[..16]);
    out
}

// ── Little-endian write helpers ──────────────────────────────────

fn put_u8(buf: &mut [u8], off: usize, v: u8) { buf[off] = v; }
fn put_u16(buf: &mut [u8], off: usize, v: u16) { buf[off..off+2].copy_from_slice(&v.to_le_bytes()); }
fn put_u32(buf: &mut [u8], off: usize, v: u32) { buf[off..off+4].copy_from_slice(&v.to_le_bytes()); }
fn put_u64(buf: &mut [u8], off: usize, v: u64) { buf[off..off+8].copy_from_slice(&v.to_le_bytes()); }
fn put_i64(buf: &mut [u8], off: usize, v: i64) { buf[off..off+8].copy_from_slice(&v.to_le_bytes()); }
fn put_f64(buf: &mut [u8], off: usize, v: f64) { buf[off..off+8].copy_from_slice(&v.to_le_bytes()); }

fn get_u8(buf: &[u8], off: usize) -> u8 { buf[off] }
fn get_u16(buf: &[u8], off: usize) -> u16 { u16::from_le_bytes(buf[off..off+2].try_into().unwrap()) }
fn get_u32(buf: &[u8], off: usize) -> u32 { u32::from_le_bytes(buf[off..off+4].try_into().unwrap()) }
fn get_u64(buf: &[u8], off: usize) -> u64 { u64::from_le_bytes(buf[off..off+8].try_into().unwrap()) }
fn get_i64(buf: &[u8], off: usize) -> i64 { i64::from_le_bytes(buf[off..off+8].try_into().unwrap()) }
fn get_f64(buf: &[u8], off: usize) -> f64 { f64::from_le_bytes(buf[off..off+8].try_into().unwrap()) }

fn put_bytes(buf: &mut [u8], off: usize, src: &[u8]) {
    buf[off..off + src.len()].copy_from_slice(src);
}

// ── Pack Block 0 ─────────────────────────────────────────────────

/// Pack MktfHeader into 4096-byte Block 0. Byte-identical to Python pack_block0.
pub fn pack_block0(h: &MktfHeader) -> [u8; BLOCK_SIZE] {
    let mut buf = [0u8; BLOCK_SIZE];

    // FORMAT [0:16)
    put_bytes(&mut buf, 0, MAGIC);
    put_u16(&mut buf, 4, h.format_version);
    put_u32(&mut buf, 6, h.flags);
    put_u16(&mut buf, 10, h.alignment);
    put_u16(&mut buf, 12, h.header_blocks);
    // [14:16) reserved

    // IDENTITY [16:128)
    let o = OFF_IDENTITY;
    put_bytes(&mut buf, o, &pad(&h.leaf_id, 32));
    put_bytes(&mut buf, o + 32, &pad(&h.ticker, 16));
    put_bytes(&mut buf, o + 48, &pad(&h.day, 10));
    put_u8(&mut buf, o + 58, h.ti);
    put_u8(&mut buf, o + 59, h.to);
    put_bytes(&mut buf, o + 60, &pad(&h.leaf_version, 16));
    put_bytes(&mut buf, o + 76, &h.schema_fingerprint);
    put_u32(&mut buf, o + 92, h.leaf_id_hash);
    // [108:128) reserved

    // TREE [128:176)
    let o = OFF_TREE;
    put_u8(&mut buf, o, h.kingdom);
    put_u8(&mut buf, o + 1, h.phylum);
    put_u8(&mut buf, o + 2, h.class);
    put_u8(&mut buf, o + 3, h.rank);
    put_u8(&mut buf, o + 4, h.family);
    put_u8(&mut buf, o + 5, h.genus);
    put_u8(&mut buf, o + 6, h.species);
    put_u8(&mut buf, o + 7, h.depth);
    put_u8(&mut buf, o + 8, h.computation_shape);
    put_u8(&mut buf, o + 9, h.compute_target);
    put_u8(&mut buf, o + 10, h.precision);
    put_u8(&mut buf, o + 11, h.leaf_type);
    put_u32(&mut buf, o + 12, h.n_upstream);
    put_u8(&mut buf, o + 16, h.ki);
    put_u8(&mut buf, o + 17, h.ko);
    put_u8(&mut buf, o + 18, h.domain);
    // [147:176) reserved

    // DIMENSIONS [176:240)
    let o = OFF_DIMENSIONS;
    put_u64(&mut buf, o, h.n_rows);
    put_u16(&mut buf, o + 8, h.n_cols);
    put_u32(&mut buf, o + 10, h.n_bins);
    put_u32(&mut buf, o + 14, h.n_cadences);
    put_u32(&mut buf, o + 18, h.n_tickers);
    put_u64(&mut buf, o + 22, h.bytes_data);
    put_u64(&mut buf, o + 30, h.bytes_file);
    // [214:240) reserved

    // TEMPORAL [240:288)
    let o = OFF_TEMPORAL;
    put_i64(&mut buf, o, h.ts_first_ns);
    put_i64(&mut buf, o + 8, h.ts_last_ns);
    put_i64(&mut buf, o + 16, h.ts_range_ns);
    put_i64(&mut buf, o + 24, h.market_open_ns);
    put_i64(&mut buf, o + 32, h.market_close_ns);
    // [280:288) reserved

    // QUALITY [288:368)
    let o = OFF_QUALITY;
    put_u64(&mut buf, o, h.total_nulls);
    put_u64(&mut buf, o + 8, h.total_nans);
    put_u64(&mut buf, o + 16, h.total_infs);
    put_u32(&mut buf, o + 24, h.null_ppm);
    put_f64(&mut buf, o + 28, h.effective_rank);
    put_f64(&mut buf, o + 36, h.compression_ratio);
    put_u64(&mut buf, o + 44, h.data_checksum);
    // [340:368) reserved

    // PROVENANCE [368:448)
    let o = OFF_PROVENANCE;
    put_i64(&mut buf, o, h.write_timestamp_ns);
    put_u32(&mut buf, o + 8, h.write_duration_ms);
    put_u32(&mut buf, o + 12, h.compute_duration_ms);
    put_u32(&mut buf, o + 16, h.rewrite_count);
    put_u32(&mut buf, o + 20, h.engine_version);
    put_u64(&mut buf, o + 24, h.source_data_hash);
    put_bytes(&mut buf, o + 32, &pad(&h.compute_host, 16));
    put_u32(&mut buf, o + 48, h.cuda_version);
    put_u32(&mut buf, o + 52, h.driver_version);
    // [424:448) reserved

    // LAYOUT [448:512)
    let o = OFF_LAYOUT;
    put_u64(&mut buf, o, h.dir_offset);
    put_u64(&mut buf, o + 8, h.dir_entries);
    put_u64(&mut buf, o + 16, h.meta_offset);
    put_u64(&mut buf, o + 24, h.meta_size);
    put_u64(&mut buf, o + 32, h.data_start);
    // [488:512) reserved

    // ASSET [512:576)
    let o = OFF_ASSET;
    put_u16(&mut buf, o, h.exchange_code);
    put_u8(&mut buf, o + 2, h.asset_class);
    put_u8(&mut buf, o + 3, h.universe_tier);
    put_u32(&mut buf, o + 4, h.tick_count);
    put_u32(&mut buf, o + 8, h.market_cap_tier);
    put_i64(&mut buf, o + 12, h.avg_spread_e8);
    // [532:576) reserved

    // UPSTREAM FINGERPRINTS [576:1600) — 16 slots x 64 bytes
    for (i, up) in h.upstream.iter().take(MAX_UPSTREAM).enumerate() {
        let base = OFF_UPSTREAM + i * UPSTREAM_ENTRY_SIZE;
        put_bytes(&mut buf, base, &pad(&up.leaf_id, 32));
        put_i64(&mut buf, base + 32, up.write_ts_ns);
        put_u64(&mut buf, base + 40, up.data_hash);
        put_u8(&mut buf, base + 48, up.ti);
        put_u8(&mut buf, base + 49, up.to);
        // [+50:+64) reserved
    }

    // STATISTICS [1600:1728)
    let o = OFF_STATISTICS;
    put_f64(&mut buf, o, h.global_mean);
    put_f64(&mut buf, o + 8, h.global_std);
    put_f64(&mut buf, o + 16, h.global_min);
    put_f64(&mut buf, o + 24, h.global_max);
    put_f64(&mut buf, o + 32, h.global_median);
    put_f64(&mut buf, o + 40, h.global_skew);
    put_f64(&mut buf, o + 48, h.global_kurtosis);
    put_f64(&mut buf, o + 56, h.global_entropy);
    // [1664:1728) reserved

    // SPATIAL [1728:1856)
    let o = OFF_SPATIAL;
    put_u32(&mut buf, o, h.spatial_dims);
    put_u32(&mut buf, o + 4, h.atlas_version);
    for j in 0..8 {
        put_f64(&mut buf, o + 8 + j * 8, h.coordinates[j]);
    }
    // [1800:1856) reserved

    // STATUS [4094:4096)
    buf[4094] = if h.is_complete { 1 } else { 0 };
    buf[4095] = if h.is_dirty { 1 } else { 0 };

    buf
}

// ── Unpack Block 0 ───────────────────────────────────────────────

/// Unpack 4096-byte Block 0 into MktfHeader.
pub fn unpack_block0(buf: &[u8]) -> Result<MktfHeader, String> {
    if buf.len() < BLOCK_SIZE {
        return Err(format!("Buffer too small: {} < {}", buf.len(), BLOCK_SIZE));
    }

    if &buf[0..4] != MAGIC {
        return Err(format!("Bad magic: {:?}", &buf[0..4]));
    }

    let mut h = MktfHeader::default();

    // FORMAT
    h.format_version = get_u16(buf, 4);
    h.flags = get_u32(buf, 6);
    h.alignment = get_u16(buf, 10);
    h.header_blocks = get_u16(buf, 12);

    // IDENTITY
    let o = OFF_IDENTITY;
    h.leaf_id = unpad(&buf[o..o + 32]);
    h.ticker = unpad(&buf[o + 32..o + 48]);
    h.day = unpad(&buf[o + 48..o + 58]);
    h.ti = get_u8(buf, o + 58);
    h.to = get_u8(buf, o + 59);
    h.leaf_version = unpad(&buf[o + 60..o + 76]);
    h.schema_fingerprint.copy_from_slice(&buf[o + 76..o + 92]);
    h.leaf_id_hash = get_u32(buf, o + 92);

    // TREE
    let o = OFF_TREE;
    h.kingdom = get_u8(buf, o);
    h.phylum = get_u8(buf, o + 1);
    h.class = get_u8(buf, o + 2);
    h.rank = get_u8(buf, o + 3);
    h.family = get_u8(buf, o + 4);
    h.genus = get_u8(buf, o + 5);
    h.species = get_u8(buf, o + 6);
    h.depth = get_u8(buf, o + 7);
    h.computation_shape = get_u8(buf, o + 8);
    h.compute_target = get_u8(buf, o + 9);
    h.precision = get_u8(buf, o + 10);
    h.leaf_type = get_u8(buf, o + 11);
    h.n_upstream = get_u32(buf, o + 12);
    h.ki = get_u8(buf, o + 16);
    h.ko = get_u8(buf, o + 17);
    h.domain = get_u8(buf, o + 18);

    // DIMENSIONS
    let o = OFF_DIMENSIONS;
    h.n_rows = get_u64(buf, o);
    h.n_cols = get_u16(buf, o + 8);
    h.n_bins = get_u32(buf, o + 10);
    h.n_cadences = get_u32(buf, o + 14);
    h.n_tickers = get_u32(buf, o + 18);
    h.bytes_data = get_u64(buf, o + 22);
    h.bytes_file = get_u64(buf, o + 30);

    // TEMPORAL
    let o = OFF_TEMPORAL;
    h.ts_first_ns = get_i64(buf, o);
    h.ts_last_ns = get_i64(buf, o + 8);
    h.ts_range_ns = get_i64(buf, o + 16);
    h.market_open_ns = get_i64(buf, o + 24);
    h.market_close_ns = get_i64(buf, o + 32);

    // QUALITY
    let o = OFF_QUALITY;
    h.total_nulls = get_u64(buf, o);
    h.total_nans = get_u64(buf, o + 8);
    h.total_infs = get_u64(buf, o + 16);
    h.null_ppm = get_u32(buf, o + 24);
    h.effective_rank = get_f64(buf, o + 28);
    h.compression_ratio = get_f64(buf, o + 36);
    h.data_checksum = get_u64(buf, o + 44);

    // PROVENANCE
    let o = OFF_PROVENANCE;
    h.write_timestamp_ns = get_i64(buf, o);
    h.write_duration_ms = get_u32(buf, o + 8);
    h.compute_duration_ms = get_u32(buf, o + 12);
    h.rewrite_count = get_u32(buf, o + 16);
    h.engine_version = get_u32(buf, o + 20);
    h.source_data_hash = get_u64(buf, o + 24);
    h.compute_host = unpad(&buf[o + 32..o + 48]);
    h.cuda_version = get_u32(buf, o + 48);
    h.driver_version = get_u32(buf, o + 52);

    // LAYOUT
    let o = OFF_LAYOUT;
    h.dir_offset = get_u64(buf, o);
    h.dir_entries = get_u64(buf, o + 8);
    h.meta_offset = get_u64(buf, o + 16);
    h.meta_size = get_u64(buf, o + 24);
    h.data_start = get_u64(buf, o + 32);

    // ASSET
    let o = OFF_ASSET;
    h.exchange_code = get_u16(buf, o);
    h.asset_class = get_u8(buf, o + 2);
    h.universe_tier = get_u8(buf, o + 3);
    h.tick_count = get_u32(buf, o + 4);
    h.market_cap_tier = get_u32(buf, o + 8);
    h.avg_spread_e8 = get_i64(buf, o + 12);

    // UPSTREAM
    let n = (h.n_upstream as usize).min(MAX_UPSTREAM);
    for i in 0..n {
        let base = OFF_UPSTREAM + i * UPSTREAM_ENTRY_SIZE;
        h.upstream.push(UpstreamFingerprint {
            leaf_id: unpad(&buf[base..base + 32]),
            write_ts_ns: get_i64(buf, base + 32),
            data_hash: get_u64(buf, base + 40),
            ti: get_u8(buf, base + 48),
            to: get_u8(buf, base + 49),
        });
    }

    // STATISTICS
    let o = OFF_STATISTICS;
    h.global_mean = get_f64(buf, o);
    h.global_std = get_f64(buf, o + 8);
    h.global_min = get_f64(buf, o + 16);
    h.global_max = get_f64(buf, o + 24);
    h.global_median = get_f64(buf, o + 32);
    h.global_skew = get_f64(buf, o + 40);
    h.global_kurtosis = get_f64(buf, o + 48);
    h.global_entropy = get_f64(buf, o + 56);

    // SPATIAL
    let o = OFF_SPATIAL;
    h.spatial_dims = get_u32(buf, o);
    h.atlas_version = get_u32(buf, o + 4);
    for j in 0..8 {
        h.coordinates[j] = get_f64(buf, o + 8 + j * 8);
    }

    // STATUS
    h.is_complete = buf[4094] != 0;
    h.is_dirty = buf[4095] != 0;

    Ok(h)
}

// ── Pack / Unpack Byte Entry ─────────────────────────────────────

/// Pack one ByteEntry into 128 bytes. Byte-identical to Python pack_byte_entry.
pub fn pack_byte_entry(col: &ByteEntry) -> [u8; ENTRY_SIZE] {
    let mut buf = [0u8; ENTRY_SIZE];

    // [0:32) name
    let name_bytes = pad(&col.name, 32);
    buf[..32].copy_from_slice(&name_bytes);

    // [32] dtype_code
    buf[32] = col.dtype_code;
    // [33:40) reserved (7 bytes)

    // [40:112) numeric fields
    put_u64(&mut buf, 40, col.n_elements);
    put_u64(&mut buf, 48, col.data_offset);
    put_u64(&mut buf, 56, col.data_nbytes);
    put_u64(&mut buf, 64, col.null_count);
    put_f64(&mut buf, 72, col.min_value);
    put_f64(&mut buf, 80, col.max_value);
    put_f64(&mut buf, 88, col.mean_value);
    put_f64(&mut buf, 96, col.scale_factor);
    put_f64(&mut buf, 104, col.sentinel_value);

    // [112:128) compression descriptor
    buf[112] = col.compression_algo;
    buf[113] = col.pre_filter;
    buf[114] = col.typesize;
    buf[115] = col.filter_param;
    put_u64(&mut buf, 116, col.compressed_size);
    // [124:128) reserved

    buf
}

/// Unpack one ByteEntry from 128 bytes at the given offset.
pub fn unpack_byte_entry(buf: &[u8], offset: usize) -> ByteEntry {
    let b = &buf[offset..offset + ENTRY_SIZE];
    ByteEntry {
        name: unpad(&b[..32]),
        dtype_code: b[32],
        n_elements: get_u64(b, 40),
        data_offset: get_u64(b, 48),
        data_nbytes: get_u64(b, 56),
        null_count: get_u64(b, 64),
        min_value: get_f64(b, 72),
        max_value: get_f64(b, 80),
        mean_value: get_f64(b, 88),
        scale_factor: get_f64(b, 96),
        sentinel_value: get_f64(b, 104),
        compression_algo: b[112],
        pre_filter: b[113],
        typesize: b[114],
        filter_param: b[115],
        compressed_size: get_u64(b, 116),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block0_roundtrip() {
        let mut h = MktfHeader::default();
        h.leaf_id = "K01P01.TI00TO00".into();
        h.ticker = "AAPL".into();
        h.day = "2026-03-28".into();
        h.alignment = 64;
        h.n_rows = 598057;
        h.n_cols = 5;
        h.bytes_data = 12_000_000;
        h.bytes_file = 12_004_096;
        h.data_checksum = 0xDEADBEEF_CAFEBABE;
        h.is_complete = true;

        let packed = pack_block0(&h);
        assert_eq!(packed.len(), BLOCK_SIZE);
        assert_eq!(&packed[0..4], MAGIC);

        let h2 = unpack_block0(&packed).unwrap();
        assert_eq!(h2.leaf_id, "K01P01.TI00TO00");
        assert_eq!(h2.ticker, "AAPL");
        assert_eq!(h2.day, "2026-03-28");
        assert_eq!(h2.alignment, 64);
        assert_eq!(h2.n_rows, 598057);
        assert_eq!(h2.n_cols, 5);
        assert_eq!(h2.bytes_data, 12_000_000);
        assert_eq!(h2.bytes_file, 12_004_096);
        assert_eq!(h2.data_checksum, 0xDEADBEEF_CAFEBABE);
        assert!(h2.is_complete);
        assert!(!h2.is_dirty);
    }

    #[test]
    fn byte_entry_roundtrip() {
        let e = ByteEntry {
            name: "K01P01.DI01DO01".into(),
            dtype_code: DTYPE_FLOAT32,
            n_elements: 598057,
            data_offset: 8192,
            data_nbytes: 2392228,
            null_count: 0,
            min_value: 100.5,
            max_value: 250.75,
            mean_value: 175.3,
            scale_factor: 1.0,
            sentinel_value: f64::NAN,
            compression_algo: COMPRESS_NONE,
            pre_filter: FILTER_NONE,
            typesize: 4,
            filter_param: 0,
            compressed_size: 0,
        };

        let packed = pack_byte_entry(&e);
        assert_eq!(packed.len(), ENTRY_SIZE);

        let e2 = unpack_byte_entry(&packed, 0);
        assert_eq!(e2.name, "K01P01.DI01DO01");
        assert_eq!(e2.dtype_code, DTYPE_FLOAT32);
        assert_eq!(e2.n_elements, 598057);
        assert_eq!(e2.data_offset, 8192);
        assert_eq!(e2.data_nbytes, 2392228);
        assert_eq!(e2.null_count, 0);
        assert_eq!(e2.min_value, 100.5);
        assert_eq!(e2.max_value, 250.75);
        assert_eq!(e2.mean_value, 175.3);
    }

    #[test]
    fn alignment_math() {
        assert_eq!(align(0, 4096), 0);
        assert_eq!(align(1, 4096), 4096);
        assert_eq!(align(4096, 4096), 4096);
        assert_eq!(align(4097, 4096), 8192);
        assert_eq!(align(100, 64), 128);
        assert_eq!(align(64, 64), 64);
    }

    #[test]
    fn leaf_id_hash_consistency() {
        let h1 = compute_leaf_id_hash("K01P01.TI00TO00");
        let h2 = compute_leaf_id_hash("K01P01.TI00TO00");
        assert_eq!(h1, h2);
        assert_ne!(h1, compute_leaf_id_hash("K02P01C01.TI00TO05"));
    }

    #[test]
    fn schema_fingerprint_consistency() {
        let cols = vec![
            ("price".into(), DTYPE_FLOAT32),
            ("volume".into(), DTYPE_FLOAT32),
        ];
        let fp1 = compute_schema_fingerprint(&cols);
        let fp2 = compute_schema_fingerprint(&cols);
        assert_eq!(fp1, fp2);

        // Order independent
        let cols_rev = vec![
            ("volume".into(), DTYPE_FLOAT32),
            ("price".into(), DTYPE_FLOAT32),
        ];
        assert_eq!(fp1, compute_schema_fingerprint(&cols_rev));
    }
}
