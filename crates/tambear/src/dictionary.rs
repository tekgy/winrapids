//! Dictionary encoding for string columns.
//!
//! Strings are encoded to u32 integer codes at ingestion time.
//! The dictionary (code → string) lives in the `.tb` file header and in
//! the [`ColumnEncoding::Dictionary`] variant of each string column.
//!
//! GroupBy on a string column = GroupBy on its code column = hash scatter on u32.
//! Join on a string column = direct-index gather on u32.
//! No string comparison ever happens inside the engine.
//! Strings are decoded back to human-readable values ONLY at output.
//!
//! "Tam doesn't do strings. Tam knows the dictionary."
//!
//! ## .tb header layout for dictionaries
//!
//! Each string column has a dictionary section in the header:
//! ```text
//! [u32: n_entries]
//! [u32: entry_0_len][u8×n: entry_0_bytes]
//! [u32: entry_1_len][u8×n: entry_1_bytes]
//! ...
//! ```
//! Codes are assigned in the order strings are first seen during ingestion.
//! The code for string s = position of s in this list.
//!
//! ## String→code encoding
//!
//! At ingestion:
//! 1. Scan all values in the string column
//! 2. Assign codes in encounter order (deterministic for reproducibility)
//! 3. Store codes as u32 in the column's GPU buffer
//! 4. Store the dictionary in the `.tb` header
//!
//! For join operations across files: dictionaries must be unified at join time
//! (both files use the same code space). This is O(dict_size), not O(n_rows).

// TODO (observer): implement DictionaryEncoder for use at ingestion (.tb read)

/// In-memory dictionary: maps string values to u32 codes and back.
pub struct Dictionary {
    /// String → code. Lookup during encoding.
    pub str_to_code: std::collections::HashMap<String, u32>,
    /// Code → string. Lookup during decoding (output only).
    pub code_to_str: Vec<String>,
}

impl Dictionary {
    pub fn new() -> Self {
        Self {
            str_to_code: std::collections::HashMap::new(),
            code_to_str: Vec::new(),
        }
    }

    /// Encode a string, assigning a new code if not yet seen.
    pub fn encode(&mut self, s: &str) -> u32 {
        if let Some(&code) = self.str_to_code.get(s) {
            return code;
        }
        let code = self.code_to_str.len() as u32;
        self.str_to_code.insert(s.to_string(), code);
        self.code_to_str.push(s.to_string());
        code
    }

    /// Decode a code back to its string. Panics if out of range.
    pub fn decode(&self, code: u32) -> &str {
        &self.code_to_str[code as usize]
    }

    pub fn n_entries(&self) -> usize {
        self.code_to_str.len()
    }
}

impl Default for Dictionary {
    fn default() -> Self { Self::new() }
}
