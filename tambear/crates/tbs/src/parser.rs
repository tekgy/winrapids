//! Parser for the `.tbs` chain scripting language.
//!
//! ## Grammar
//!
//! ```text
//! chain    = step ('.' step)*
//! step     = name '(' arglist? ')'
//! name     = IDENT ('.' IDENT)*
//! arglist  = arg (',' arg)*
//! arg      = (IDENT '=')? value
//! value    = '"' [^"]* '"'
//!           | '-'? DIGIT+ ('.' DIGIT+)?
//!           | 'true' | 'false'
//! ```
//!
//! ## Example
//!
//! ```
//! use tambear_tbs::TbsChain;
//!
//! let chain = TbsChain::parse(r#"
//!     normalize()
//!       .discover_clusters(epsilon=0.5, min_samples=2)
//!       .train.linear(target="price")
//! "#).unwrap();
//!
//! assert_eq!(chain.steps.len(), 3);
//! assert_eq!(chain.steps[0].name.to_string(), "normalize");
//! assert_eq!(chain.steps[1].name.to_string(), "discover_clusters");
//! assert_eq!(chain.steps[2].name.to_string(), "train.linear");
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

/// A parsed `.tbs` chain: a sequence of steps joined by `.`.
#[derive(Debug, Clone, PartialEq)]
pub struct TbsChain {
    pub steps: Vec<TbsStep>,
}

/// One step in a chain: a name followed by a parenthesised argument list.
#[derive(Debug, Clone, PartialEq)]
pub struct TbsStep {
    pub name: TbsName,
    pub args: Vec<TbsArg>,
}

/// Step name — either `"normalize"` or `"train.linear"` (dotted for namespaced ops).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TbsName {
    Simple(String),
    Dotted(String, String),
}

impl TbsName {
    pub fn as_str(&self) -> (&str, Option<&str>) {
        match self {
            TbsName::Simple(s)   => (s.as_str(), None),
            TbsName::Dotted(a, b) => (a.as_str(), Some(b.as_str())),
        }
    }
}

impl fmt::Display for TbsName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TbsName::Simple(s)    => write!(f, "{s}"),
            TbsName::Dotted(a, b) => write!(f, "{a}.{b}"),
        }
    }
}

/// One argument: positional or named (`key=value`).
#[derive(Debug, Clone, PartialEq)]
pub enum TbsArg {
    Positional(TbsValue),
    Named { key: String, value: TbsValue },
}

impl TbsArg {
    pub fn value(&self) -> &TbsValue {
        match self {
            TbsArg::Positional(v)  => v,
            TbsArg::Named { value, .. } => value,
        }
    }
}

/// A literal value.
#[derive(Debug, Clone, PartialEq)]
pub enum TbsValue {
    Str(String),
    Float(f64),
    Int(i64),
    Bool(bool),
}

impl TbsValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            TbsValue::Float(f) => Some(*f),
            TbsValue::Int(i)   => Some(*i as f64),
            _                  => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            TbsValue::Int(i) if *i >= 0 => Some(*i as usize),
            _                            => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            TbsValue::Str(s) => Some(s.as_str()),
            _                => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Display (canonical round-trip)
// ---------------------------------------------------------------------------

impl fmt::Display for TbsChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, step) in self.steps.iter().enumerate() {
            if i > 0 { write!(f, "\n  .")?; }
            write!(f, "{step}")?;
        }
        Ok(())
    }
}

impl fmt::Display for TbsStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.name)?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{arg}")?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for TbsArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TbsArg::Positional(v)           => write!(f, "{v}"),
            TbsArg::Named { key, value }    => write!(f, "{key}={value}"),
        }
    }
}

impl fmt::Display for TbsValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TbsValue::Str(s)   => write!(f, "\"{s}\""),
            TbsValue::Float(v) => write!(f, "{v}"),
            TbsValue::Int(i)   => write!(f, "{i}"),
            TbsValue::Bool(b)  => write!(f, "{b}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Parse error
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct TbsParseError {
    pub pos: usize,
    pub message: String,
}

impl fmt::Display for TbsParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at position {}: {}", self.pos, self.message)
    }
}

impl std::error::Error for TbsParseError {}

// ---------------------------------------------------------------------------
// Parser state
// ---------------------------------------------------------------------------

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self { input: src.as_bytes(), pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.input.len() - self.pos
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.input.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek() {
            match ch {
                b' ' | b'\t' | b'\r' | b'\n' => { self.pos += 1; }
                b'/' if self.peek2() == Some(b'/') => {
                    // line comment
                    while let Some(c) = self.advance() {
                        if c == b'\n' { break; }
                    }
                }
                _ => break,
            }
        }
    }

    fn expect(&mut self, ch: u8) -> Result<(), TbsParseError> {
        self.skip_ws();
        match self.peek() {
            Some(c) if c == ch => { self.pos += 1; Ok(()) }
            other => Err(TbsParseError {
                pos: self.pos,
                message: format!(
                    "expected '{}', got {}",
                    ch as char,
                    other.map(|c| format!("'{}'", c as char))
                        .unwrap_or_else(|| "end of input".into())
                ),
            }),
        }
    }

    fn parse_ident(&mut self) -> Result<String, TbsParseError> {
        self.skip_ws();
        let start = self.pos;
        match self.peek() {
            Some(c) if c.is_ascii_alphabetic() || c == b'_' => { self.pos += 1; }
            other => return Err(TbsParseError {
                pos: self.pos,
                message: format!(
                    "expected identifier, got {}",
                    other.map(|c| format!("'{}'", c as char))
                        .unwrap_or_else(|| "end of input".into())
                ),
            }),
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' { self.pos += 1; } else { break; }
        }
        Ok(std::str::from_utf8(&self.input[start..self.pos]).unwrap().to_owned())
    }

    fn parse_string(&mut self) -> Result<TbsValue, TbsParseError> {
        self.expect(b'"')?;
        let start = self.pos;
        loop {
            match self.advance() {
                None    => return Err(TbsParseError { pos: self.pos, message: "unterminated string".into() }),
                Some(b'"') => break,
                Some(_) => {}
            }
        }
        let s = std::str::from_utf8(&self.input[start..self.pos - 1]).unwrap().to_owned();
        Ok(TbsValue::Str(s))
    }

    fn parse_number(&mut self) -> Result<TbsValue, TbsParseError> {
        let start = self.pos;
        if self.peek() == Some(b'-') { self.pos += 1; }
        while self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) { self.pos += 1; }
        let is_float = self.peek() == Some(b'.') && self.peek2().map(|c| c.is_ascii_digit()).unwrap_or(false);
        if is_float {
            self.pos += 1; // consume '.'
            while self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) { self.pos += 1; }
        }
        let s = std::str::from_utf8(&self.input[start..self.pos]).unwrap();
        if is_float {
            Ok(TbsValue::Float(s.parse().unwrap()))
        } else {
            Ok(TbsValue::Int(s.parse().unwrap()))
        }
    }

    fn parse_value(&mut self) -> Result<TbsValue, TbsParseError> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => self.parse_string(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(b't') => {
                let id = self.parse_ident()?;
                if id == "true" { Ok(TbsValue::Bool(true)) }
                else { Err(TbsParseError { pos: self.pos, message: format!("unknown value: {id}") }) }
            }
            Some(b'f') => {
                let id = self.parse_ident()?;
                if id == "false" { Ok(TbsValue::Bool(false)) }
                else { Err(TbsParseError { pos: self.pos, message: format!("unknown value: {id}") }) }
            }
            other => Err(TbsParseError {
                pos: self.pos,
                message: format!(
                    "expected value, got {}",
                    other.map(|c| format!("'{}'", c as char))
                        .unwrap_or_else(|| "end of input".into())
                ),
            }),
        }
    }

    /// Parse one argument. Looks ahead to decide named vs positional.
    ///
    /// Strategy: try to read `IDENT '='`. If we see `IDENT` followed immediately
    /// by `=`, it's named. Otherwise fall back to positional `value`.
    fn parse_arg(&mut self) -> Result<TbsArg, TbsParseError> {
        self.skip_ws();
        // Lookahead: is this IDENT=value?
        let saved = self.pos;
        if let Ok(key) = self.parse_ident() {
            self.skip_ws();
            if self.peek() == Some(b'=') {
                self.pos += 1; // consume '='
                let value = self.parse_value()?;
                return Ok(TbsArg::Named { key, value });
            }
        }
        // Not named — reset and parse as value
        self.pos = saved;
        Ok(TbsArg::Positional(self.parse_value()?))
    }

    /// Parse the name part: IDENT ('.' IDENT)?
    ///
    /// Careful: the step separator is also `.`. We only consume a second IDENT
    /// after `.` if what follows is NOT `(` — a dotted name like `train.linear`
    /// has `linear(` not just `.` followed by whitespace.
    fn parse_name(&mut self) -> Result<TbsName, TbsParseError> {
        let first = self.parse_ident()?;
        // Peek: is the next non-ws char '.' followed by an ident (not just ws/'(')?
        let saved = self.pos;
        self.skip_ws();
        if self.peek() == Some(b'.') {
            // could be step separator or dotted name — look one more ahead
            let dot_pos = self.pos;
            self.pos += 1; // consume '.'
            self.skip_ws();
            // If the next thing is an ident AND it's followed by '(', this is a dotted name
            if self.peek().map(|c| c.is_ascii_alphabetic() || c == b'_').unwrap_or(false) {
                let ident_start = self.pos;
                let second = self.parse_ident()?;
                self.skip_ws();
                if self.peek() == Some(b'(') {
                    // Confirmed: train.linear(...)
                    return Ok(TbsName::Dotted(first, second));
                }
                // Not a dotted name — it's a step separator. Restore.
                let _ = (dot_pos, ident_start); // suppress warnings
                self.pos = saved;
            } else {
                self.pos = saved;
            }
        } else {
            self.pos = saved;
        }
        Ok(TbsName::Simple(first))
    }

    fn parse_step(&mut self) -> Result<TbsStep, TbsParseError> {
        self.skip_ws();
        let name = self.parse_name()?;
        self.expect(b'(')?;
        let mut args = Vec::new();
        self.skip_ws();
        if self.peek() != Some(b')') {
            args.push(self.parse_arg()?);
            loop {
                self.skip_ws();
                if self.peek() != Some(b',') { break; }
                self.pos += 1; // consume ','
                self.skip_ws();
                if self.peek() == Some(b')') { break; } // trailing comma OK
                args.push(self.parse_arg()?);
            }
        }
        self.expect(b')')?;
        Ok(TbsStep { name, args })
    }

    fn parse_chain(&mut self) -> Result<TbsChain, TbsParseError> {
        let mut steps = Vec::new();
        steps.push(self.parse_step()?);
        loop {
            self.skip_ws();
            if self.peek() != Some(b'.') { break; }
            self.pos += 1; // consume '.'
            steps.push(self.parse_step()?);
        }
        self.skip_ws();
        if self.remaining() > 0 {
            return Err(TbsParseError {
                pos: self.pos,
                message: format!("unexpected input after chain: '{}'",
                    std::str::from_utf8(&self.input[self.pos..]).unwrap_or("?")),
            });
        }
        Ok(TbsChain { steps })
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl TbsChain {
    /// Parse a `.tbs` chain from text.
    pub fn parse(src: &str) -> Result<Self, TbsParseError> {
        Parser::new(src.trim()).parse_chain()
    }
}

// Convenience: look up the first named arg matching a key, or the n-th positional.
impl TbsStep {
    /// Get a named argument by key, or the n-th positional argument.
    pub fn get_arg(&self, key: &str, positional_index: usize) -> Option<&TbsValue> {
        for arg in &self.args {
            if let TbsArg::Named { key: k, value } = arg {
                if k == key { return Some(value); }
            }
        }
        let mut idx = 0;
        for arg in &self.args {
            if let TbsArg::Positional(v) = arg {
                if idx == positional_index { return Some(v); }
                idx += 1;
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_step_no_args() {
        let c = TbsChain::parse("normalize()").unwrap();
        assert_eq!(c.steps.len(), 1);
        assert_eq!(c.steps[0].name, TbsName::Simple("normalize".into()));
        assert!(c.steps[0].args.is_empty());
    }

    #[test]
    fn parse_named_args() {
        let c = TbsChain::parse("discover_clusters(epsilon=0.5, min_samples=2)").unwrap();
        let step = &c.steps[0];
        assert_eq!(step.name.to_string(), "discover_clusters");
        assert_eq!(step.get_arg("epsilon", 0).and_then(|v| v.as_f64()), Some(0.5));
        assert_eq!(step.get_arg("min_samples", 1).and_then(|v| v.as_usize()), Some(2));
    }

    #[test]
    fn parse_dotted_name() {
        let c = TbsChain::parse(r#"train.linear(target="price")"#).unwrap();
        assert_eq!(c.steps[0].name, TbsName::Dotted("train".into(), "linear".into()));
        assert_eq!(c.steps[0].get_arg("target", 0).and_then(|v| v.as_str()), Some("price"));
    }

    #[test]
    fn parse_multi_step_chain() {
        let src = r#"normalize()
  .discover_clusters(epsilon=0.5, min_samples=2)
  .train.linear(target="price")"#;
        let c = TbsChain::parse(src).unwrap();
        assert_eq!(c.steps.len(), 3);
        assert_eq!(c.steps[0].name.to_string(), "normalize");
        assert_eq!(c.steps[1].name.to_string(), "discover_clusters");
        assert_eq!(c.steps[2].name.to_string(), "train.linear");
    }

    #[test]
    fn parse_positional_args() {
        let c = TbsChain::parse("discover_clusters(1.5, 2)").unwrap();
        let step = &c.steps[0];
        assert_eq!(step.get_arg("epsilon", 0).and_then(|v| v.as_f64()), Some(1.5));
        assert_eq!(step.get_arg("min_samples", 1).and_then(|v| v.as_usize()), Some(2));
    }

    #[test]
    fn parse_bool_value() {
        let c = TbsChain::parse("sample(stratified=true)").unwrap();
        assert_eq!(
            c.steps[0].get_arg("stratified", 0),
            Some(&TbsValue::Bool(true))
        );
    }

    #[test]
    fn parse_negative_number() {
        let c = TbsChain::parse("clip(min=-1.0, max=1.0)").unwrap();
        assert_eq!(c.steps[0].get_arg("min", 0).and_then(|v| v.as_f64()), Some(-1.0));
        assert_eq!(c.steps[0].get_arg("max", 1).and_then(|v| v.as_f64()), Some(1.0));
    }

    #[test]
    fn parse_comment() {
        let src = "normalize() // z-score per column\n  .discover_clusters(epsilon=1.0, min_samples=2)";
        let c = TbsChain::parse(src).unwrap();
        assert_eq!(c.steps.len(), 2);
    }

    #[test]
    fn round_trip() {
        let src = r#"normalize()
  .discover_clusters(epsilon=0.5, min_samples=2)
  .train.linear(target="price")"#;
        let chain1 = TbsChain::parse(src).unwrap();
        let displayed = chain1.to_string();
        let chain2 = TbsChain::parse(&displayed).unwrap();
        assert_eq!(chain1, chain2, "round-trip failed:\noriginal: {src}\ndisplayed: {displayed}");
    }

    #[test]
    fn parse_full_vocabulary_sample() {
        // Exercise all vocabulary categories from the .tbs sketch
        let chains = [
            r#"normalize()"#,
            r#"discover_clusters(epsilon=0.5, min_samples=2)"#,
            r#"kmeans(k=5)"#,
            r#"train.linear(target="price")"#,
            r#"filter(predicate="v > 0.0")"#,
            r#"window(size=10)"#,
            r#"knn(k=5, metric="l2sq")"#,
        ];
        for src in chains {
            TbsChain::parse(src).unwrap_or_else(|e| panic!("failed to parse '{src}': {e}"));
        }
    }

    #[test]
    fn error_on_missing_paren() {
        assert!(TbsChain::parse("normalize").is_err());
    }

    #[test]
    fn error_on_unterminated_string() {
        assert!(TbsChain::parse(r#"train.linear(target="price)"#).is_err());
    }

    #[test]
    fn error_on_trailing_junk() {
        assert!(TbsChain::parse("normalize() garbage").is_err());
    }
}
