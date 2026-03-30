//! Persistent GPU store with provenance tracking and cost-aware LRU eviction.
//!
//! The store is the sharing optimizer's memory. It tracks GPU-resident buffers
//! by provenance hash and implements the zero-translation cache:
//!
//!   result === cache entry === consumer input (pointer handoff, no copy)
//!
//! The compiler's execution plan is a pointer routing graph through this store:
//!   - Valid pointer → route it (zero computation, 865x case)
//!   - No pointer → compute → store → route (the fallback)
//!
//! Eviction is cost-aware: when VRAM is over budget, the store evicts
//! buffers that are cheapest to recompute per byte. The caller is responsible
//! for actually freeing the GPU memory (the store tracks metadata, not memory).

use std::collections::HashMap;

use crate::header::{BufferHeader, BufferPtr, EvictedEntry, Location, now_ns};
use crate::world::{ProvenanceCache, DirtyBitmap, ResidencyMap};

/// Sentinel value for "no link" in the LRU list.
const NONE: u32 = u32::MAX;

/// An entry in the store's arena.
struct StoreEntry {
    header: BufferHeader,
    ptr: BufferPtr,
    /// LRU doubly-linked list: index of more-recently-used entry.
    lru_prev: u32,
    /// LRU doubly-linked list: index of less-recently-used entry.
    lru_next: u32,
}

/// Store statistics.
#[derive(Clone, Debug, Default)]
pub struct StoreStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub entries: u32,
    pub used_bytes: u64,
    pub capacity_bytes: u64,
}

impl StoreStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

/// Persistent GPU store with provenance tracking.
///
/// Manages buffer metadata and provenance lookups. Does NOT own GPU memory —
/// the caller allocates and frees device memory. The store tells you:
/// - "I have this computation at this pointer" (hit)
/// - "I don't have it" (miss)
/// - "Evict these entries to make room" (pressure)
pub struct GpuStore {
    /// Provenance hash → entry index in the arena.
    index: HashMap<[u8; 16], u32>,

    /// Arena of store entries.
    entries: Vec<Option<StoreEntry>>,

    /// Free indices in the arena.
    free_list: Vec<u32>,

    /// VRAM budget and usage tracking.
    capacity_bytes: u64,
    used_bytes: u64,

    /// LRU list head (most recently used).
    lru_head: u32,
    /// LRU list tail (least recently used — eviction candidates).
    lru_tail: u32,

    /// Statistics.
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl GpuStore {
    /// Create a new store with the given VRAM budget in bytes.
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            index: HashMap::new(),
            entries: Vec::new(),
            free_list: Vec::new(),
            capacity_bytes,
            used_bytes: 0,
            lru_head: NONE,
            lru_tail: NONE,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Look up a provenance hash. Returns the buffer pointer if found.
    ///
    /// On hit: updates access count, last access time, moves to LRU head.
    /// Target: 1μs hit latency (HashMap lookup + LRU touch).
    pub fn lookup(&mut self, provenance: &[u8; 16]) -> Option<BufferPtr> {
        match self.index.get(provenance).copied() {
            Some(idx) => {
                let entry = self.entries[idx as usize].as_mut()?;
                entry.header.access_count += 1;
                entry.header.last_access_ns = now_ns();
                let ptr = entry.ptr;
                self.hits += 1;
                self.lru_touch(idx);
                Some(ptr)
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Register a new buffer in the store.
    ///
    /// If the provenance already exists, updates the pointer (idempotent).
    /// If over budget, returns entries that should be evicted — the caller
    /// frees the GPU memory, or spills to the next tier.
    pub fn register(
        &mut self,
        header: BufferHeader,
        ptr: BufferPtr,
    ) -> Vec<EvictedEntry> {
        // If already registered, update in place
        if let Some(&idx) = self.index.get(&header.provenance) {
            if let Some(entry) = self.entries[idx as usize].as_mut() {
                entry.ptr = ptr;
                entry.header.last_access_ns = now_ns();
                self.lru_touch(idx);
                return Vec::new();
            }
        }

        // Evict until we have room
        let mut evicted = Vec::new();
        while self.used_bytes + header.byte_size > self.capacity_bytes {
            if let Some(victim) = self.evict_one() {
                evicted.push(victim);
            } else {
                break; // Nothing left to evict
            }
        }

        // Allocate entry
        let idx = self.alloc_entry();
        let provenance = header.provenance;
        self.used_bytes += header.byte_size;
        self.entries[idx as usize] = Some(StoreEntry {
            header,
            ptr,
            lru_prev: NONE,
            lru_next: NONE,
        });
        self.index.insert(provenance, idx);
        self.lru_push_front(idx);

        evicted
    }

    /// Remove a specific entry by provenance.
    pub fn remove(&mut self, provenance: &[u8; 16]) -> Option<EvictedEntry> {
        let idx = self.index.remove(provenance)?;
        let entry = self.entries[idx as usize].take()?;
        self.used_bytes -= entry.header.byte_size;
        self.lru_unlink(idx);
        self.free_list.push(idx);
        Some(EvictedEntry {
            provenance: entry.header.provenance,
            ptr: entry.ptr,
            cost_us: entry.header.cost_us,
            location: entry.header.location,
        })
    }

    /// Get store statistics.
    pub fn stats(&self) -> StoreStats {
        StoreStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            entries: self.index.len() as u32,
            used_bytes: self.used_bytes,
            capacity_bytes: self.capacity_bytes,
        }
    }

    /// Current VRAM usage in bytes.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// VRAM budget in bytes.
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// Number of entries in the store.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    // ── Arena allocation ──────────────────────────────────────

    fn alloc_entry(&mut self) -> u32 {
        if let Some(idx) = self.free_list.pop() {
            idx
        } else {
            let idx = self.entries.len() as u32;
            self.entries.push(None);
            idx
        }
    }

    // ── Cost-aware LRU eviction ───────────────────────────────
    //
    // Pure LRU would evict from the tail. Cost-aware scans the last
    // K entries from the tail and picks the one with the lowest
    // "retention score" = cost_us * access_count / byte_size.
    //
    // Low retention score = cheap to recompute, infrequently used, large.
    // These are the best eviction candidates.

    /// Number of tail entries to consider for cost-aware eviction.
    const EVICTION_WINDOW: usize = 8;

    fn evict_one(&mut self) -> Option<EvictedEntry> {
        if self.lru_tail == NONE {
            return None;
        }

        // Scan the last K entries from tail, find the lowest retention score
        let mut best_idx = self.lru_tail;
        let mut best_score = self.retention_score(self.lru_tail);
        let mut cursor = self.lru_tail;

        for _ in 1..Self::EVICTION_WINDOW {
            let entry = self.entries[cursor as usize].as_ref()?;
            let prev = entry.lru_prev;
            if prev == NONE {
                break;
            }
            cursor = prev;
            let score = self.retention_score(cursor);
            if score < best_score {
                best_score = score;
                best_idx = cursor;
            }
        }

        // Evict the chosen entry
        let entry = self.entries[best_idx as usize].take()?;
        self.index.remove(&entry.header.provenance);
        self.used_bytes -= entry.header.byte_size;
        self.lru_unlink(best_idx);
        self.free_list.push(best_idx);
        self.evictions += 1;

        Some(EvictedEntry {
            provenance: entry.header.provenance,
            ptr: entry.ptr,
            cost_us: entry.header.cost_us,
            location: entry.header.location,
        })
    }

    /// Retention score: higher = keep longer.
    /// cost_us * (1 + access_count) / byte_size
    fn retention_score(&self, idx: u32) -> f64 {
        match &self.entries[idx as usize] {
            Some(entry) => {
                let cost = entry.header.cost_us as f64;
                let accesses = (1 + entry.header.access_count) as f64;
                let bytes = entry.header.byte_size.max(1) as f64;
                cost * accesses / bytes
            }
            None => 0.0,
        }
    }

    // ── Intrusive doubly-linked LRU list ──────────────────────
    //
    // Head = most recently used. Tail = least recently used.
    // O(1) touch (move to head), O(1) unlink, O(1) push front.

    fn lru_touch(&mut self, idx: u32) {
        if self.lru_head == idx {
            return; // Already at head
        }
        self.lru_unlink(idx);
        self.lru_push_front(idx);
    }

    fn lru_push_front(&mut self, idx: u32) {
        if let Some(entry) = self.entries[idx as usize].as_mut() {
            entry.lru_prev = NONE;
            entry.lru_next = self.lru_head;
        }
        if self.lru_head != NONE {
            if let Some(old_head) = self.entries[self.lru_head as usize].as_mut() {
                old_head.lru_prev = idx;
            }
        }
        self.lru_head = idx;
        if self.lru_tail == NONE {
            self.lru_tail = idx;
        }
    }

    fn lru_unlink(&mut self, idx: u32) {
        let (prev, next) = match self.entries[idx as usize].as_ref() {
            Some(entry) => (entry.lru_prev, entry.lru_next),
            None => return,
        };

        // Patch prev's next
        if prev != NONE {
            if let Some(p) = self.entries[prev as usize].as_mut() {
                p.lru_next = next;
            }
        } else {
            self.lru_head = next;
        }

        // Patch next's prev
        if next != NONE {
            if let Some(n) = self.entries[next as usize].as_mut() {
                n.lru_prev = prev;
            }
        } else {
            self.lru_tail = prev;
        }

        // Clear the node's links
        if let Some(entry) = self.entries[idx as usize].as_mut() {
            entry.lru_prev = NONE;
            entry.lru_next = NONE;
        }
    }
}

// ────────────────────────────────────────────────────────────
// WorldState trait implementations for GpuStore
// ────────────────────────────────────────────────────────────

impl ProvenanceCache for GpuStore {
    fn provenance_get(&mut self, provenance: &[u8; 16]) -> Option<BufferPtr> {
        self.lookup(provenance)
    }

    fn provenance_put(&mut self, provenance: [u8; 16], ptr: BufferPtr, cost_us: f32) {
        let header = BufferHeader {
            provenance,
            cost_us,
            access_count: 0,
            location: Location::Gpu,
            dtype: crate::header::DType::F32,
            ndim: 1,
            flags: 0,
            _align: [0; 4],
            len: ptr.byte_size / 4, // assume f32
            byte_size: ptr.byte_size,
            created_ns: now_ns(),
            last_access_ns: now_ns(),
        };
        self.register(header, ptr);
    }
}

impl DirtyBitmap for GpuStore {
    fn is_clean(&self, provenance: &[u8; 16]) -> bool {
        // A buffer is clean if it exists in the store — its provenance
        // hash encodes its inputs, so if the inputs changed, the caller
        // would be looking up a different provenance hash.
        self.index.contains_key(provenance)
    }
}

impl ResidencyMap for GpuStore {
    fn is_resident(&self, provenance: &[u8; 16]) -> bool {
        if let Some(&idx) = self.index.get(provenance) {
            if let Some(entry) = &self.entries[idx as usize] {
                return entry.header.location == Location::Gpu;
            }
        }
        false
    }

    fn resident_pointer(&self, provenance: &[u8; 16]) -> Option<BufferPtr> {
        let &idx = self.index.get(provenance)?;
        let entry = self.entries[idx as usize].as_ref()?;
        if entry.header.location == Location::Gpu {
            Some(entry.ptr)
        } else {
            None
        }
    }
}
