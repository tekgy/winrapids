//! Evaluate CUDA-like φ expressions in pure Rust.
//!
//! Used by [`CpuBackend`](crate::CpuBackend) to evaluate scatter/map phi
//! expressions without NVRTC.  Supports the same expression subset used by
//! `scatter_jit.rs`:
//!
//! - Variables: `v`, `r`, `g`, `a`, `b`
//! - Binary: `+`, `-`, `*`, `/`
//! - Comparison: `>`, `<`, `>=`, `<=`, `==`, `!=`
//! - Ternary: `cond ? then : else`
//! - Functions: `exp`, `log`, `sqrt`, `abs`, `fabs`, `fmin`, `fmax`, `pow`
//! - Unary minus
//! - Parentheses
//! - Literal numbers (integer and float, including scientific notation)
//!
//! Well-known phi expressions are fast-pathed to avoid parsing overhead.

/// Variable bindings for a φ expression.
#[derive(Clone, Debug)]
pub struct PhiCtx {
    pub v: f64,
    pub r: f64,
    pub g: f64,
    pub a: f64,
    pub b: f64,
}

impl PhiCtx {
    pub fn scatter(v: f64, r: f64, g: i32) -> Self {
        Self { v, r, g: g as f64, a: 0.0, b: 0.0 }
    }
    pub fn map1(v: f64) -> Self {
        Self { v, r: 0.0, g: 0.0, a: 0.0, b: 0.0 }
    }
    pub fn map2(a: f64, b: f64) -> Self {
        Self { v: 0.0, r: 0.0, g: 0.0, a, b }
    }
}

/// Evaluate a phi expression with the given variable context.
///
/// Fast-paths well-known expressions; falls back to parsing.
pub fn eval_phi(expr: &str, ctx: &PhiCtx) -> f64 {
    // Fast path for the 7 most common expressions
    match expr {
        "v"                     => return ctx.v,
        "v * v"                 => return ctx.v * ctx.v,
        "1.0"                   => return 1.0,
        "v - r"                 => return ctx.v - ctx.r,
        "(v - r) * (v - r)"    => return (ctx.v - ctx.r) * (ctx.v - ctx.r),
        "a * b"                 => return ctx.a * ctx.b,
        "a + b"                 => return ctx.a + ctx.b,
        _                       => {}
    }

    // General path: tokenize → parse → evaluate
    let tokens = tokenize(expr);
    let mut p = Parser { tokens: &tokens, pos: 0, ctx };
    p.ternary()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tokenizer
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Plus, Minus, Star, Slash,
    LParen, RParen, Comma,
    Gt, Lt, GtEq, LtEq, Eq2, NotEq,
    Question, Colon,
}

fn tokenize(s: &str) -> Vec<Tok> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();

    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' | b'\n' | b'\r' => { i += 1; }
            b'+' => { out.push(Tok::Plus);   i += 1; }
            b'-' => { out.push(Tok::Minus);  i += 1; }
            b'*' => { out.push(Tok::Star);   i += 1; }
            b'/' => { out.push(Tok::Slash);  i += 1; }
            b'(' => { out.push(Tok::LParen); i += 1; }
            b')' => { out.push(Tok::RParen); i += 1; }
            b'?' => { out.push(Tok::Question); i += 1; }
            b':' => { out.push(Tok::Colon);  i += 1; }
            b',' => { out.push(Tok::Comma);  i += 1; }
            b'>' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    out.push(Tok::GtEq); i += 2;
                } else {
                    out.push(Tok::Gt); i += 1;
                }
            }
            b'<' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    out.push(Tok::LtEq); i += 2;
                } else {
                    out.push(Tok::Lt); i += 1;
                }
            }
            b'=' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    out.push(Tok::Eq2); i += 2;
                } else {
                    i += 1; // skip lone '='
                }
            }
            b'!' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    out.push(Tok::NotEq); i += 2;
                } else {
                    i += 1;
                }
            }
            b'0'..=b'9' | b'.' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                // Handle scientific notation: 1.23e-10, 3.14E+5
                if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                    i += 1;
                    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                        i += 1;
                    }
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let num_str = &s[start..i];
                let val: f64 = num_str.parse().unwrap_or(f64::NAN);
                out.push(Tok::Num(val));
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                out.push(Tok::Ident(s[start..i].to_string()));
            }
            _ => { i += 1; } // skip unknown chars
        }
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// Recursive-descent parser/evaluator
// ═══════════════════════════════════════════════════════════════════════════

struct Parser<'a> {
    tokens: &'a [Tok],
    pos: usize,
    ctx: &'a PhiCtx,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Tok> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() { self.pos += 1; }
        tok
    }

    fn eat(&mut self, expected: &Tok) -> bool {
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    // ternary = compare ('?' expr ':' expr)?
    fn ternary(&mut self) -> f64 {
        let cond = self.compare();
        if self.eat(&Tok::Question) {
            let then_val = self.ternary();
            let _ = self.eat(&Tok::Colon);
            let else_val = self.ternary();
            if cond != 0.0 { then_val } else { else_val }
        } else {
            cond
        }
    }

    // compare = additive (('>' | '<' | '>=' | '<=' | '==' | '!=') additive)?
    fn compare(&mut self) -> f64 {
        let left = self.additive();
        match self.peek() {
            Some(Tok::Gt)    => { self.advance(); let r = self.additive(); if left > r { 1.0 } else { 0.0 } }
            Some(Tok::Lt)    => { self.advance(); let r = self.additive(); if left < r { 1.0 } else { 0.0 } }
            Some(Tok::GtEq)  => { self.advance(); let r = self.additive(); if left >= r { 1.0 } else { 0.0 } }
            Some(Tok::LtEq)  => { self.advance(); let r = self.additive(); if left <= r { 1.0 } else { 0.0 } }
            Some(Tok::Eq2)   => { self.advance(); let r = self.additive(); if (left - r).abs() < 1e-15 { 1.0 } else { 0.0 } }
            Some(Tok::NotEq) => { self.advance(); let r = self.additive(); if (left - r).abs() >= 1e-15 { 1.0 } else { 0.0 } }
            _ => left,
        }
    }

    // additive = multiplicative (('+' | '-') multiplicative)*
    fn additive(&mut self) -> f64 {
        let mut val = self.multiplicative();
        loop {
            match self.peek() {
                Some(Tok::Plus)  => { self.advance(); val += self.multiplicative(); }
                Some(Tok::Minus) => { self.advance(); val -= self.multiplicative(); }
                _ => break,
            }
        }
        val
    }

    // multiplicative = unary (('*' | '/') unary)*
    fn multiplicative(&mut self) -> f64 {
        let mut val = self.unary();
        loop {
            match self.peek() {
                Some(Tok::Star)  => { self.advance(); val *= self.unary(); }
                Some(Tok::Slash) => { self.advance(); val /= self.unary(); }
                _ => break,
            }
        }
        val
    }

    // unary = '-' unary | primary
    fn unary(&mut self) -> f64 {
        if self.peek() == Some(&Tok::Minus) {
            self.advance();
            -self.unary()
        } else {
            self.primary()
        }
    }

    // primary = NUMBER | IDENT | IDENT '(' args ')' | '(' expr ')'
    fn primary(&mut self) -> f64 {
        match self.advance().cloned() {
            Some(Tok::Num(n)) => n,
            Some(Tok::Ident(name)) => {
                if self.eat(&Tok::LParen) {
                    // Function call
                    self.func_call(&name)
                } else {
                    // Variable lookup
                    match name.as_str() {
                        "v" => self.ctx.v,
                        "r" => self.ctx.r,
                        "g" => self.ctx.g,
                        "a" => self.ctx.a,
                        "b" => self.ctx.b,
                        _ => f64::NAN,
                    }
                }
            }
            Some(Tok::LParen) => {
                let val = self.ternary();
                let _ = self.eat(&Tok::RParen);
                val
            }
            _ => f64::NAN,
        }
    }

    fn func_call(&mut self, name: &str) -> f64 {
        let arg1 = self.ternary();
        let result = if self.eat(&Tok::Comma) {
            // Two-argument function
            let arg2 = self.ternary();
            match name {
                "fmin" | "min" => arg1.min(arg2),
                "fmax" | "max" => arg1.max(arg2),
                "pow"          => arg1.powf(arg2),
                _              => f64::NAN,
            }
        } else {
            // Single-argument function
            match name {
                "exp"   => arg1.exp(),
                "log"   => arg1.ln(),
                "sqrt"  => arg1.sqrt(),
                "abs" | "fabs" => arg1.abs(),
                "sin"   => arg1.sin(),
                "cos"   => arg1.cos(),
                "ceil"  => arg1.ceil(),
                "floor" => arg1.floor(),
                "round" => arg1.round(),
                _       => f64::NAN,
            }
        };
        let _ = self.eat(&Tok::RParen);
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(v: f64, r: f64) -> PhiCtx {
        PhiCtx::scatter(v, r, 0)
    }

    fn ctx2(a: f64, b: f64) -> PhiCtx {
        PhiCtx::map2(a, b)
    }

    // ── Fast-path well-known expressions ─────────────────────────────────

    #[test]
    fn phi_sum() {
        assert_eq!(eval_phi("v", &ctx(3.5, 0.0)), 3.5);
    }

    #[test]
    fn phi_sum_sq() {
        assert_eq!(eval_phi("v * v", &ctx(3.0, 0.0)), 9.0);
    }

    #[test]
    fn phi_count() {
        assert_eq!(eval_phi("1.0", &ctx(999.0, 0.0)), 1.0);
    }

    #[test]
    fn phi_centered_sum() {
        assert_eq!(eval_phi("v - r", &ctx(5.0, 2.0)), 3.0);
    }

    #[test]
    fn phi_centered_sum_sq() {
        assert_eq!(eval_phi("(v - r) * (v - r)", &ctx(5.0, 2.0)), 9.0);
    }

    #[test]
    fn phi_multiply_ab() {
        assert_eq!(eval_phi("a * b", &ctx2(3.0, 4.0)), 12.0);
    }

    #[test]
    fn phi_add_ab() {
        assert_eq!(eval_phi("a + b", &ctx2(3.0, 4.0)), 7.0);
    }

    // ── General parser tests ─────────────────────────────────────────────

    #[test]
    fn general_arithmetic() {
        assert!((eval_phi("v * v + 1.0", &ctx(3.0, 0.0)) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn general_parentheses() {
        assert!((eval_phi("(v + 1.0) * (v - 1.0)", &ctx(3.0, 0.0)) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn general_unary_minus() {
        assert!((eval_phi("-v", &ctx(3.0, 0.0)) - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn general_exp() {
        assert!((eval_phi("exp(v)", &ctx(0.0, 0.0)) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn general_log() {
        assert!((eval_phi("log(v)", &ctx(1.0, 0.0)) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn general_sqrt() {
        assert!((eval_phi("sqrt(v)", &ctx(4.0, 0.0)) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn general_baked_constants() {
        // Simulates: exp(v - MAX) / SUM_EXP with baked values
        // v=3.0, MAX=3.0 → exp(0) = 1.0; SUM_EXP=e → result = 1/e ≈ 0.3679
        let expr = "exp(v - 3.00000000000000000) / 2.71828182845904500";
        let result = eval_phi(expr, &ctx(3.0, 0.0));
        let expected = 1.0 / std::f64::consts::E;
        assert!((result - expected).abs() < 1e-10, "got {result}, expected {expected}");
    }

    #[test]
    fn general_ternary() {
        assert_eq!(eval_phi("b > 0.0 ? a : 0.0", &ctx2(5.0, 1.0)), 5.0);
        assert_eq!(eval_phi("b > 0.0 ? a : 0.0", &ctx2(5.0, -1.0)), 0.0);
    }

    #[test]
    fn general_sgd_update() {
        // a - 0.1 * b = 1.0 - 0.1 * 0.5 = 0.95
        let expr = "a - 0.10000000000000001 * b";
        let result = eval_phi(expr, &ctx2(1.0, 0.5));
        assert!((result - 0.95).abs() < 1e-10, "got {result}");
    }

    #[test]
    fn general_fmin_fmax() {
        assert_eq!(eval_phi("fmin(a, b)", &ctx2(3.0, 5.0)), 3.0);
        assert_eq!(eval_phi("fmax(a, b)", &ctx2(3.0, 5.0)), 5.0);
    }

    #[test]
    fn general_nested_functions() {
        assert!((eval_phi("sqrt(abs(-4.0))", &ctx(0.0, 0.0)) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn general_scientific_notation() {
        assert!((eval_phi("1.5e2 + v", &ctx(1.0, 0.0)) - 151.0).abs() < 1e-10);
    }

    #[test]
    fn general_comparison_ops() {
        assert_eq!(eval_phi("v >= 3.0 ? 1.0 : 0.0", &ctx(3.0, 0.0)), 1.0);
        assert_eq!(eval_phi("v >= 3.0 ? 1.0 : 0.0", &ctx(2.0, 0.0)), 0.0);
        assert_eq!(eval_phi("v < 3.0 ? 1.0 : 0.0", &ctx(2.0, 0.0)), 1.0);
    }
}
