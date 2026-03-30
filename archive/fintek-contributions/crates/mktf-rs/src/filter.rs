//! Pre-filters for MKTF compression pipeline.
//!
//! Filters are applied BEFORE compression to improve compression ratios,
//! and reversed AFTER decompression to recover original data.
//!
//! Byte shuffle (Blosc-style): groups byte 0 of all elements together,
//! then byte 1, etc. For float32 time series, this groups all sign+exponent
//! bytes (which are similar) together, giving LZ4 much better runs.
//!
//! Example for 3 float32 values (typesize=4, 12 bytes):
//!   Input:  [a0 a1 a2 a3] [b0 b1 b2 b3] [c0 c1 c2 c3]
//!   Output: [a0 b0 c0] [a1 b1 c1] [a2 b2 c2] [a3 b3 c3]

/// Byte shuffle: reorder bytes so byte `k` of each element is contiguous.
///
/// `typesize` is the element size in bytes (e.g., 4 for f32, 8 for f64).
pub fn shuffle(input: &[u8], typesize: usize) -> Vec<u8> {
    if typesize <= 1 || input.is_empty() {
        return input.to_vec();
    }

    let n_elements = input.len() / typesize;
    let remainder = input.len() % typesize;
    let mut output = vec![0u8; input.len()];

    // Shuffle complete elements
    for i in 0..n_elements {
        for k in 0..typesize {
            output[k * n_elements + i] = input[i * typesize + k];
        }
    }

    // Copy any remainder bytes as-is at the end
    if remainder > 0 {
        let shuffled_len = n_elements * typesize;
        output[shuffled_len..].copy_from_slice(&input[shuffled_len..]);
    }

    output
}

/// Reverse byte shuffle.
pub fn unshuffle(input: &[u8], typesize: usize) -> Vec<u8> {
    if typesize <= 1 || input.is_empty() {
        return input.to_vec();
    }

    let n_elements = input.len() / typesize;
    let remainder = input.len() % typesize;
    let mut output = vec![0u8; input.len()];

    // Unshuffle complete elements
    for i in 0..n_elements {
        for k in 0..typesize {
            output[i * typesize + k] = input[k * n_elements + i];
        }
    }

    // Copy remainder
    if remainder > 0 {
        let shuffled_len = n_elements * typesize;
        output[shuffled_len..].copy_from_slice(&input[shuffled_len..]);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shuffle_unshuffle_roundtrip() {
        let data: Vec<u8> = (0..120).collect(); // 30 float32 values
        let shuffled = shuffle(&data, 4);
        let recovered = unshuffle(&shuffled, 4);
        assert_eq!(data, recovered);
    }

    #[test]
    fn shuffle_identity_for_typesize_1() {
        let data = vec![1, 2, 3, 4, 5];
        assert_eq!(shuffle(&data, 1), data);
        assert_eq!(unshuffle(&data, 1), data);
    }

    #[test]
    fn shuffle_known_pattern() {
        // 3 elements of typesize=2: [a0 a1] [b0 b1] [c0 c1]
        let data = vec![10, 11, 20, 21, 30, 31];
        let shuffled = shuffle(&data, 2);
        // Expected: [a0 b0 c0] [a1 b1 c1] = [10 20 30 11 21 31]
        assert_eq!(shuffled, vec![10, 20, 30, 11, 21, 31]);
    }

    #[test]
    fn shuffle_f32_improves_compression() {
        // Slowly increasing float32 values — high byte similarity
        let values: Vec<f32> = (0..1000).map(|i| 100.0 + i as f32 * 0.01).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let raw_compressed = lz4_flex::compress_prepend_size(&bytes);
        let shuffled = shuffle(&bytes, 4);
        let shuf_compressed = lz4_flex::compress_prepend_size(&shuffled);

        // Shuffled should compress significantly better
        assert!(
            shuf_compressed.len() < raw_compressed.len(),
            "shuffle+lz4 ({}) should beat raw lz4 ({})",
            shuf_compressed.len(),
            raw_compressed.len(),
        );
    }

    #[test]
    fn shuffle_empty() {
        assert_eq!(shuffle(&[], 4), Vec::<u8>::new());
        assert_eq!(unshuffle(&[], 4), Vec::<u8>::new());
    }
}
