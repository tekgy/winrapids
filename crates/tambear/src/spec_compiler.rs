//! # .spec Formula Compiler
//!
//! Compiles typed mathematical expressions into tambear accumulate+gather calls.
//!
//! Input language examples:
//! ```text
//! vwap = Σ(p*v) / Σ(v)
//! rsi  = 100 - 100/(1 + Σ(max(Δx,0))/Σ(max(-Δx,0)))
//! ema  = scan(α*x + (1-α)*acc)
//! bb_upper = mean + 2*√(Σ(x²)/N - (Σ(x)/N)²)
//! ```
//!
//! The compiler:
//! 1. Parses formulas into a `SpecExpr` AST
//! 2. Compiles the AST to an `ExecutionPlan` of `PlanStep` primitives
//! 3. Identifies shared sub-expressions for fusion (e.g., `Σ(x)` used twice)
//! 4. Reports total passes needed
//!
//! The compilation target is `AccumulateStep` from `proof.rs`, extended with
//! `GatherStep` for offset operations like `x[t-1]` and `Δx`.

use std::collections::HashMap;
use std::fmt;

use crate::proof::{AccumulateStep, BinOp, GroupingTag, UnOp};

// ═══════════════════════════════════════════════════════════════════════════
// AST
// ═══════════════════════════════════════════════════════════════════════════

/// A node in a spec formula expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum SpecExpr {
    /// Literal number.
    Lit(f64),
    /// Variable reference (column name).
    Var(String),
    /// Variable with offset: `x[t-1]`, `p[t+3]`.
    Gather { var: String, offset: i64 },
    /// Difference: `Δx` = `x[t] - x[t-1]`.
    Delta(String),
    /// Binary operation.
    BinOp(BinOp, Box<SpecExpr>, Box<SpecExpr>),
    /// Unary operation.
    UnOp(UnOp, Box<SpecExpr>),
    /// Accumulate: `Σ(expr)`, `Π(expr)`, `max(expr)`, `min(expr)`.
    Accumulate { op: BinOp, inner: Box<SpecExpr> },
    /// Prefix scan: `scan(expr)` where `acc` is the accumulator variable.
    Scan(Box<SpecExpr>),
    /// Special variable `N` — count of elements.
    Count,
}

/// A parsed formula: `name = expr`.
#[derive(Debug, Clone)]
pub struct SpecFormula {
    pub name: String,
    pub expr: SpecExpr,
}

impl fmt::Display for SpecExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpecExpr::Lit(v) => write!(f, "{v}"),
            SpecExpr::Var(s) => write!(f, "{s}"),
            SpecExpr::Gather { var, offset } => {
                if *offset >= 0 { write!(f, "{var}[t+{offset}]") }
                else { write!(f, "{var}[t{offset}]") }
            }
            SpecExpr::Delta(s) => write!(f, "Δ{s}"),
            SpecExpr::BinOp(op, a, b) => {
                match op {
                    BinOp::Max => write!(f, "max({a}, {b})"),
                    BinOp::Min => write!(f, "min({a}, {b})"),
                    _ => {
                        let sym = match op {
                            BinOp::Add => "+", BinOp::Sub => "-",
                            BinOp::Mul => "*", BinOp::Div => "/",
                            _ => "?",
                        };
                        write!(f, "({a} {sym} {b})")
                    }
                }
            }
            SpecExpr::UnOp(op, x) => {
                match op {
                    UnOp::Neg => write!(f, "(-{x})"),
                    UnOp::Sq => write!(f, "{x}²"),
                    UnOp::Sqrt => write!(f, "√({x})"),
                    UnOp::Log => write!(f, "ln({x})"),
                    UnOp::Exp => write!(f, "exp({x})"),
                    UnOp::Abs => write!(f, "|{x}|"),
                    UnOp::One => write!(f, "1"),
                }
            }
            SpecExpr::Accumulate { op, inner } => {
                let name = match op {
                    BinOp::Add => "Σ", BinOp::Mul => "Π",
                    BinOp::Max => "max", BinOp::Min => "min",
                    _ => "acc",
                };
                write!(f, "{name}({inner})")
            }
            SpecExpr::Scan(inner) => write!(f, "scan({inner})"),
            SpecExpr::Count => write!(f, "N"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Parser
// ═══════════════════════════════════════════════════════════════════════════

/// Parse error with position.
#[derive(Debug)]
pub struct SpecParseError {
    pub pos: usize,
    pub msg: String,
}

impl fmt::Display for SpecParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at position {}: {}", self.pos, self.msg)
    }
}

impl std::error::Error for SpecParseError {}

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Parser { input: input.as_bytes(), pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.input.get(self.pos).copied()?;
        self.pos += 1;
        Some(ch)
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == b' ' || ch == b'\t' || ch == b'\r' || ch == b'\n' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, ch: u8) -> Result<(), SpecParseError> {
        self.skip_ws();
        match self.advance() {
            Some(c) if c == ch => Ok(()),
            Some(c) => Err(self.err(format!("expected '{}', got '{}'", ch as char, c as char))),
            None => Err(self.err(format!("expected '{}', got EOF", ch as char))),
        }
    }

    fn err(&self, msg: String) -> SpecParseError {
        SpecParseError { pos: self.pos, msg }
    }

    fn at_end(&self) -> bool {
        let mut p = self.pos;
        while p < self.input.len() {
            if self.input[p] != b' ' && self.input[p] != b'\t'
                && self.input[p] != b'\r' && self.input[p] != b'\n'
            {
                return false;
            }
            p += 1;
        }
        true
    }

    // ── Tokenization ─────────────────────────────────────────────────

    fn parse_ident(&mut self) -> Result<String, SpecParseError> {
        self.skip_ws();
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(self.err("expected identifier".into()));
        }
        Ok(String::from_utf8_lossy(&self.input[start..self.pos]).to_string())
    }

    fn parse_number(&mut self) -> Result<f64, SpecParseError> {
        self.skip_ws();
        let start = self.pos;
        // Optional negative sign (only if not preceded by something that makes it a subtraction)
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == b'.' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(self.err("expected number".into()));
        }
        let s = String::from_utf8_lossy(&self.input[start..self.pos]);
        s.parse::<f64>().map_err(|_| self.err(format!("invalid number: {s}")))
    }

    // ── Grammar ──────────────────────────────────────────────────────

    /// Parse a formula: `name = expr`
    fn parse_formula(&mut self) -> Result<SpecFormula, SpecParseError> {
        let name = self.parse_ident()?;
        self.expect(b'=')?;
        let expr = self.parse_expr()?;
        Ok(SpecFormula { name, expr })
    }

    /// expr = term (('+' | '-') term)*
    fn parse_expr(&mut self) -> Result<SpecExpr, SpecParseError> {
        let mut left = self.parse_term()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => {
                    self.advance();
                    let right = self.parse_term()?;
                    left = SpecExpr::BinOp(BinOp::Add, Box::new(left), Box::new(right));
                }
                Some(b'-') => {
                    // Disambiguate: is this subtraction or a negative number?
                    self.advance();
                    let right = self.parse_term()?;
                    left = SpecExpr::BinOp(BinOp::Sub, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// term = unary (('*' | '/') unary)*
    fn parse_term(&mut self) -> Result<SpecExpr, SpecParseError> {
        let mut left = self.parse_unary()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'*') => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = SpecExpr::BinOp(BinOp::Mul, Box::new(left), Box::new(right));
                }
                Some(b'/') => {
                    self.advance();
                    let right = self.parse_unary()?;
                    left = SpecExpr::BinOp(BinOp::Div, Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }
        Ok(left)
    }

    /// unary = '-' unary | atom ('^' number | '²')?
    fn parse_unary(&mut self) -> Result<SpecExpr, SpecParseError> {
        self.skip_ws();
        // Unary minus
        if self.peek() == Some(b'-') {
            // Check it's not the start of a number following a binary op
            // By this point, we're at the unary level, so this IS negation
            self.advance();
            let inner = self.parse_unary()?;
            return Ok(SpecExpr::UnOp(UnOp::Neg, Box::new(inner)));
        }
        let mut result = self.parse_atom()?;
        self.skip_ws();
        // Postfix: ^2 or ²
        if self.peek() == Some(b'^') {
            self.advance();
            let exp = self.parse_number()?;
            if (exp - 2.0).abs() < 1e-10 {
                result = SpecExpr::UnOp(UnOp::Sq, Box::new(result));
            } else {
                // General power: express as exp(n * ln(x))
                result = SpecExpr::UnOp(
                    UnOp::Exp,
                    Box::new(SpecExpr::BinOp(
                        BinOp::Mul,
                        Box::new(SpecExpr::Lit(exp)),
                        Box::new(SpecExpr::UnOp(UnOp::Log, Box::new(result))),
                    )),
                );
            }
        }
        // UTF-8 ² (0xC2 0xB2)
        if self.pos + 1 < self.input.len()
            && self.input[self.pos] == 0xC2
            && self.input[self.pos + 1] == 0xB2
        {
            self.pos += 2;
            result = SpecExpr::UnOp(UnOp::Sq, Box::new(result));
        }
        Ok(result)
    }

    /// atom = number | variable | gather | delta | accumulate | scan | sqrt | abs | '(' expr ')'
    fn parse_atom(&mut self) -> Result<SpecExpr, SpecParseError> {
        self.skip_ws();
        let ch = self.peek().ok_or_else(|| self.err("unexpected EOF".into()))?;

        // Number
        if ch.is_ascii_digit() || (ch == b'.' && self.input.get(self.pos + 1).map_or(false, |c| c.is_ascii_digit())) {
            let v = self.parse_number()?;
            return Ok(SpecExpr::Lit(v));
        }

        // Parenthesized expression
        if ch == b'(' {
            self.advance();
            let inner = self.parse_expr()?;
            self.expect(b')')?;
            return Ok(inner);
        }

        // Absolute value: |expr|
        if ch == b'|' {
            self.advance();
            let inner = self.parse_expr()?;
            self.expect(b'|')?;
            return Ok(SpecExpr::UnOp(UnOp::Abs, Box::new(inner)));
        }

        // UTF-8 Σ (0xCE 0xA3)
        if self.pos + 1 < self.input.len()
            && self.input[self.pos] == 0xCE
            && self.input[self.pos + 1] == 0xA3
        {
            self.pos += 2;
            self.expect(b'(')?;
            let inner = self.parse_expr()?;
            self.expect(b')')?;
            return Ok(SpecExpr::Accumulate { op: BinOp::Add, inner: Box::new(inner) });
        }

        // UTF-8 Π (0xCE 0xA0)
        if self.pos + 1 < self.input.len()
            && self.input[self.pos] == 0xCE
            && self.input[self.pos + 1] == 0xA0
        {
            self.pos += 2;
            self.expect(b'(')?;
            let inner = self.parse_expr()?;
            self.expect(b')')?;
            return Ok(SpecExpr::Accumulate { op: BinOp::Mul, inner: Box::new(inner) });
        }

        // UTF-8 Δ (0xCE 0x94)
        if self.pos + 1 < self.input.len()
            && self.input[self.pos] == 0xCE
            && self.input[self.pos + 1] == 0x94
        {
            self.pos += 2;
            let name = self.parse_ident()?;
            return Ok(SpecExpr::Delta(name));
        }

        // UTF-8 √ (0xE2 0x88 0x9A)
        if self.pos + 2 < self.input.len()
            && self.input[self.pos] == 0xE2
            && self.input[self.pos + 1] == 0x88
            && self.input[self.pos + 2] == 0x9A
        {
            self.pos += 3;
            self.expect(b'(')?;
            let inner = self.parse_expr()?;
            self.expect(b')')?;
            return Ok(SpecExpr::UnOp(UnOp::Sqrt, Box::new(inner)));
        }

        // Identifier-based: keywords or variable
        if ch.is_ascii_alphabetic() || ch == b'_' {
            let ident = self.parse_ident()?;
            return self.parse_ident_continuation(&ident);
        }

        Err(self.err(format!("unexpected character '{}'", ch as char)))
    }

    /// After parsing an identifier, decide what it is.
    fn parse_ident_continuation(&mut self, ident: &str) -> Result<SpecExpr, SpecParseError> {
        self.skip_ws();

        match ident {
            // Accumulate keywords with parens
            "sum" | "Sum" => {
                self.expect(b'(')?;
                let inner = self.parse_expr()?;
                self.expect(b')')?;
                Ok(SpecExpr::Accumulate { op: BinOp::Add, inner: Box::new(inner) })
            }
            "prod" | "Prod" => {
                self.expect(b'(')?;
                let inner = self.parse_expr()?;
                self.expect(b')')?;
                Ok(SpecExpr::Accumulate { op: BinOp::Mul, inner: Box::new(inner) })
            }
            "max" | "Max" => {
                self.expect(b'(')?;
                let first = self.parse_expr()?;
                self.skip_ws();
                if self.peek() == Some(b',') {
                    // Two-argument form: element-wise max(a, b)
                    self.advance();
                    let second = self.parse_expr()?;
                    self.expect(b')')?;
                    Ok(SpecExpr::BinOp(BinOp::Max, Box::new(first), Box::new(second)))
                } else {
                    // Single-argument form: accumulate max over all elements
                    self.expect(b')')?;
                    Ok(SpecExpr::Accumulate { op: BinOp::Max, inner: Box::new(first) })
                }
            }
            "min" | "Min" => {
                self.expect(b'(')?;
                let first = self.parse_expr()?;
                self.skip_ws();
                if self.peek() == Some(b',') {
                    // Two-argument form: element-wise min(a, b)
                    self.advance();
                    let second = self.parse_expr()?;
                    self.expect(b')')?;
                    Ok(SpecExpr::BinOp(BinOp::Min, Box::new(first), Box::new(second)))
                } else {
                    // Single-argument form: accumulate min over all elements
                    self.expect(b')')?;
                    Ok(SpecExpr::Accumulate { op: BinOp::Min, inner: Box::new(first) })
                }
            }
            // Unary functions
            "sqrt" => {
                self.expect(b'(')?;
                let inner = self.parse_expr()?;
                self.expect(b')')?;
                Ok(SpecExpr::UnOp(UnOp::Sqrt, Box::new(inner)))
            }
            "ln" | "log" => {
                self.expect(b'(')?;
                let inner = self.parse_expr()?;
                self.expect(b')')?;
                Ok(SpecExpr::UnOp(UnOp::Log, Box::new(inner)))
            }
            "exp" => {
                self.expect(b'(')?;
                let inner = self.parse_expr()?;
                self.expect(b')')?;
                Ok(SpecExpr::UnOp(UnOp::Exp, Box::new(inner)))
            }
            "abs" => {
                self.expect(b'(')?;
                let inner = self.parse_expr()?;
                self.expect(b')')?;
                Ok(SpecExpr::UnOp(UnOp::Abs, Box::new(inner)))
            }
            // Prefix scan
            "scan" => {
                self.expect(b'(')?;
                let inner = self.parse_expr()?;
                self.expect(b')')?;
                Ok(SpecExpr::Scan(Box::new(inner)))
            }
            // Count of elements
            "N" | "n" => Ok(SpecExpr::Count),
            // Binary function: max(a, b) and min(a, b) with two args
            // (already handled above as accumulate — two-arg case not needed)

            // Variable — possibly with offset: x[t-1]
            _ => {
                if self.peek() == Some(b'[') {
                    self.advance();
                    let offset = self.parse_offset()?;
                    self.expect(b']')?;
                    Ok(SpecExpr::Gather { var: ident.to_string(), offset })
                } else {
                    Ok(SpecExpr::Var(ident.to_string()))
                }
            }
        }
    }

    /// Parse an offset expression inside `[]`: `t-1`, `t+3`, `-1`, `1`
    fn parse_offset(&mut self) -> Result<i64, SpecParseError> {
        self.skip_ws();
        // Optional 't' prefix
        if self.peek() == Some(b't') {
            self.advance();
            self.skip_ws();
        }
        let neg = if self.peek() == Some(b'-') {
            self.advance();
            true
        } else if self.peek() == Some(b'+') {
            self.advance();
            false
        } else {
            return Ok(0);
        };
        self.skip_ws();
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() { self.pos += 1; } else { break; }
        }
        let s = String::from_utf8_lossy(&self.input[start..self.pos]);
        let v: i64 = s.parse().map_err(|_| self.err("expected integer offset".into()))?;
        Ok(if neg { -v } else { v })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public parsing API
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a formula string: `"vwap = Σ(p*v) / Σ(v)"`.
pub fn parse_formula(src: &str) -> Result<SpecFormula, SpecParseError> {
    let mut p = Parser::new(src);
    let f = p.parse_formula()?;
    if !p.at_end() {
        return Err(p.err("unexpected trailing input".into()));
    }
    Ok(f)
}

/// Parse just an expression (no `name =` prefix).
pub fn parse_expr(src: &str) -> Result<SpecExpr, SpecParseError> {
    let mut p = Parser::new(src);
    let e = p.parse_expr()?;
    if !p.at_end() {
        return Err(p.err("unexpected trailing input".into()));
    }
    Ok(e)
}

/// Parse multiple formulas separated by newlines.
pub fn parse_spec(src: &str) -> Result<Vec<SpecFormula>, SpecParseError> {
    let mut formulas = Vec::new();
    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        formulas.push(parse_formula(trimmed)?);
    }
    Ok(formulas)
}

// ═══════════════════════════════════════════════════════════════════════════
// Execution plan
// ═══════════════════════════════════════════════════════════════════════════

/// A gather (offset) step in the execution plan.
#[derive(Debug, Clone)]
pub struct GatherStep {
    /// Input column name.
    pub input: String,
    /// Temporal offset (negative = look back).
    pub offset: i64,
    /// Output name for the gathered column.
    pub output: String,
}

/// A step in the compiled execution plan.
#[derive(Debug, Clone)]
pub enum PlanStep {
    /// Gather: read column with offset.
    Gather(GatherStep),
    /// Accumulate: `output = accumulate(grouping, phi, op, input_columns)`.
    Accumulate(AccumulateStep),
    /// Arithmetic combination of intermediate results.
    /// `expr` is a postfix expression using intermediate names.
    Arithmetic {
        expr: String,
        output: String,
    },
}

/// Complete execution plan for a set of formulas.
#[derive(Debug)]
pub struct ExecutionPlan {
    /// The original formulas.
    pub formulas: Vec<SpecFormula>,
    /// Ordered steps to execute.
    pub steps: Vec<PlanStep>,
    /// How many data passes are needed.
    /// Pure accumulate formulas need 1 pass (fused).
    /// Formulas with gathers need 2 (gather then accumulate).
    pub n_passes: usize,
    /// Accumulate sub-expressions shared across formulas.
    pub shared: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Compiler
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a set of spec formulas into an execution plan.
pub fn compile(formulas: &[SpecFormula]) -> ExecutionPlan {
    let mut compiler = Compiler::new();
    for f in formulas {
        compiler.compile_formula(f);
    }
    compiler.build_plan(formulas)
}

/// Compile a single formula into an execution plan.
pub fn compile_one(formula: &SpecFormula) -> ExecutionPlan {
    compile(&[formula.clone()])
}

struct Compiler {
    /// Accumulate steps (deduplicated by canonical key).
    accumulates: Vec<AccumulateStep>,
    /// Map from canonical accumulate key → output name.
    acc_names: HashMap<String, String>,
    /// Gather steps.
    gathers: Vec<GatherStep>,
    /// Map from gather key → output name.
    gather_names: HashMap<String, String>,
    /// Counter for generating unique names.
    counter: usize,
    /// Has any gather or delta appeared?
    needs_gather_pass: bool,
    /// Has any scan appeared?
    needs_scan: bool,
    /// Final arithmetic expressions per formula.
    formula_exprs: Vec<(String, String)>,
}

impl Compiler {
    fn new() -> Self {
        Compiler {
            accumulates: Vec::new(),
            acc_names: HashMap::new(),
            gathers: Vec::new(),
            gather_names: HashMap::new(),
            counter: 0,
            needs_gather_pass: false,
            needs_scan: false,
            formula_exprs: Vec::new(),
        }
    }

    fn fresh_name(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{}_{}", prefix, self.counter)
    }

    /// Compile a formula, returning the name of the final result.
    fn compile_formula(&mut self, formula: &SpecFormula) {
        let result = self.compile_expr(&formula.expr);
        self.formula_exprs.push((formula.name.clone(), result));
    }

    /// Compile an expression, returning the intermediate name holding the result.
    fn compile_expr(&mut self, expr: &SpecExpr) -> String {
        match expr {
            SpecExpr::Lit(v) => format!("{v}"),
            SpecExpr::Var(name) => name.clone(),
            SpecExpr::Count => "N".to_string(),

            SpecExpr::Gather { var, offset } => {
                let key = format!("gather({var},{offset})");
                if let Some(name) = self.gather_names.get(&key) {
                    return name.clone();
                }
                self.needs_gather_pass = true;
                let out = self.fresh_name("g");
                self.gathers.push(GatherStep {
                    input: var.clone(),
                    offset: *offset,
                    output: out.clone(),
                });
                self.gather_names.insert(key, out.clone());
                out
            }

            SpecExpr::Delta(var) => {
                // Δx = x[t] - x[t-1]
                // Compile as gather with offset -1, then subtract
                let key = format!("gather({var},-1)");
                let prev = if let Some(name) = self.gather_names.get(&key) {
                    name.clone()
                } else {
                    self.needs_gather_pass = true;
                    let out = self.fresh_name("g");
                    self.gathers.push(GatherStep {
                        input: var.clone(),
                        offset: -1,
                        output: out.clone(),
                    });
                    self.gather_names.insert(key, out.clone());
                    out
                };
                format!("({var} - {prev})")
            }

            SpecExpr::BinOp(op, a, b) => {
                let la = self.compile_expr(a);
                let lb = self.compile_expr(b);
                match op {
                    BinOp::Max => format!("fmax({la}, {lb})"),
                    BinOp::Min => format!("fmin({la}, {lb})"),
                    _ => {
                        let sym = match op {
                            BinOp::Add => "+", BinOp::Sub => "-",
                            BinOp::Mul => "*", BinOp::Div => "/",
                            _ => "?",
                        };
                        format!("({la} {sym} {lb})")
                    }
                }
            }

            SpecExpr::UnOp(op, inner) => {
                let li = self.compile_expr(inner);
                match op {
                    UnOp::Neg => format!("(-{li})"),
                    UnOp::Sq => format!("({li} * {li})"),
                    UnOp::Sqrt => format!("sqrt({li})"),
                    UnOp::Log => format!("ln({li})"),
                    UnOp::Exp => format!("exp({li})"),
                    UnOp::Abs => format!("abs({li})"),
                    UnOp::One => "1.0".to_string(),
                }
            }

            SpecExpr::Accumulate { op, inner } => {
                // Compile the inner expression — this registers gathers for
                // Δx, x[t-1] etc. so they appear as separate steps.
                let phi = self.compile_expr(inner);
                let canonical = format!("acc({op:?},{phi})");

                // Deduplicate: if we've seen this exact accumulate, reuse it
                if let Some(name) = self.acc_names.get(&canonical) {
                    return name.clone();
                }

                let out = self.fresh_name("a");
                let description = format!("{} over {inner}", match op {
                    BinOp::Add => "Σ", BinOp::Mul => "Π",
                    BinOp::Max => "max", BinOp::Min => "min",
                    _ => "acc",
                });
                self.accumulates.push(AccumulateStep {
                    grouping: GroupingTag::All,
                    phi_expr: phi,
                    op: *op,
                    description,
                });
                self.acc_names.insert(canonical, out.clone());
                out
            }

            SpecExpr::Scan(inner) => {
                self.needs_scan = true;
                let phi = self.expr_to_phi(inner);
                let out = self.fresh_name("s");
                self.accumulates.push(AccumulateStep {
                    grouping: GroupingTag::Prefix,
                    phi_expr: phi,
                    op: BinOp::Add, // scan implies additive by default
                    description: format!("scan({inner})"),
                });
                self.acc_names.insert(format!("scan({inner})"), out.clone());
                out
            }
        }
    }

    /// Convert a SpecExpr to a phi expression string (for codegen).
    fn expr_to_phi(&self, expr: &SpecExpr) -> String {
        match expr {
            SpecExpr::Lit(v) => format!("{v}"),
            SpecExpr::Var(name) => name.clone(),
            SpecExpr::Count => "N".to_string(),
            SpecExpr::Delta(var) => format!("({var} - {var}_prev)"),
            SpecExpr::Gather { var, offset } => format!("{var}[{offset}]"),
            SpecExpr::BinOp(op, a, b) => {
                let la = self.expr_to_phi(a);
                let lb = self.expr_to_phi(b);
                match op {
                    BinOp::Max => format!("fmax({la}, {lb})"),
                    BinOp::Min => format!("fmin({la}, {lb})"),
                    _ => {
                        let sym = match op {
                            BinOp::Add => "+", BinOp::Sub => "-",
                            BinOp::Mul => "*", BinOp::Div => "/",
                            _ => "?",
                        };
                        format!("({la} {sym} {lb})")
                    }
                }
            }
            SpecExpr::UnOp(op, inner) => {
                let li = self.expr_to_phi(inner);
                match op {
                    UnOp::Neg => format!("(-{li})"),
                    UnOp::Sq => format!("({li} * {li})"),
                    UnOp::Sqrt => format!("sqrt({li})"),
                    UnOp::Log => format!("ln({li})"),
                    UnOp::Exp => format!("exp({li})"),
                    UnOp::Abs => format!("abs({li})"),
                    UnOp::One => "1.0".to_string(),
                }
            }
            // Nested accumulate in phi — just inline the description
            SpecExpr::Accumulate { op, inner } => {
                let name = match op {
                    BinOp::Add => "Σ", BinOp::Mul => "Π",
                    BinOp::Max => "max", BinOp::Min => "min",
                    _ => "acc",
                };
                format!("{name}({})", self.expr_to_phi(inner))
            }
            SpecExpr::Scan(inner) => format!("scan({})", self.expr_to_phi(inner)),
        }
    }

    fn build_plan(self, formulas: &[SpecFormula]) -> ExecutionPlan {
        let mut steps = Vec::new();

        // 1. Gather passes first
        for g in &self.gathers {
            steps.push(PlanStep::Gather(g.clone()));
        }

        // 2. Accumulate steps
        for a in &self.accumulates {
            steps.push(PlanStep::Accumulate(a.clone()));
        }

        // 3. Final arithmetic for each formula
        for (name, expr) in &self.formula_exprs {
            steps.push(PlanStep::Arithmetic {
                expr: expr.clone(),
                output: name.clone(),
            });
        }

        // Count passes
        let n_passes = if self.needs_gather_pass && !self.accumulates.is_empty() {
            2 // gather pass + accumulate pass
        } else if self.needs_scan {
            2 // scan is a full pass
        } else if !self.accumulates.is_empty() {
            1 // pure accumulates can fuse into one pass
        } else {
            1 // pure arithmetic
        };

        // Find shared accumulates
        let shared: Vec<String> = self.acc_names.values()
            .filter(|name| {
                // Count how many formulas reference this accumulate
                self.formula_exprs.iter()
                    .filter(|(_, expr)| expr.contains(name.as_str()))
                    .count() > 1
            })
            .cloned()
            .collect();

        ExecutionPlan {
            formulas: formulas.to_vec(),
            steps,
            n_passes,
            shared,
        }
    }
}

impl fmt::Display for ExecutionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Execution Plan ({} passes, {} steps)", self.n_passes, self.steps.len())?;
        if !self.shared.is_empty() {
            writeln!(f, "  Shared sub-expressions: {:?}", self.shared)?;
        }
        for (i, step) in self.steps.iter().enumerate() {
            match step {
                PlanStep::Gather(g) => {
                    writeln!(f, "  [{i}] GATHER {}.offset({}) → {}", g.input, g.offset, g.output)?;
                }
                PlanStep::Accumulate(a) => {
                    writeln!(f, "  [{i}] ACCUMULATE {:?} phi=\"{}\" op={:?} // {}", a.grouping, a.phi_expr, a.op, a.description)?;
                }
                PlanStep::Arithmetic { expr, output } => {
                    writeln!(f, "  [{i}] ARITH {output} = {expr}")?;
                }
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Parsing ──────────────────────────────────────────────────────

    #[test]
    fn parse_simple_sum() {
        let f = parse_formula("total = sum(x)").unwrap();
        assert_eq!(f.name, "total");
        assert!(matches!(f.expr, SpecExpr::Accumulate { op: BinOp::Add, .. }));
    }

    #[test]
    fn parse_unicode_sigma() {
        let f = parse_formula("total = Σ(x)").unwrap();
        assert_eq!(f.name, "total");
        assert!(matches!(f.expr, SpecExpr::Accumulate { op: BinOp::Add, .. }));
    }

    #[test]
    fn parse_unicode_delta() {
        let e = parse_expr("Δx").unwrap();
        assert!(matches!(e, SpecExpr::Delta(ref s) if s == "x"));
    }

    #[test]
    fn parse_gather_offset() {
        let e = parse_expr("x[t-1]").unwrap();
        assert!(matches!(e, SpecExpr::Gather { ref var, offset: -1 } if var == "x"));
    }

    #[test]
    fn parse_vwap() {
        let f = parse_formula("vwap = Σ(p*v) / Σ(v)").unwrap();
        assert_eq!(f.name, "vwap");
        // Top level should be division
        assert!(matches!(f.expr, SpecExpr::BinOp(BinOp::Div, _, _)));
    }

    #[test]
    fn parse_scan() {
        let f = parse_formula("ema = scan(a*x + (1 - a)*acc)").unwrap();
        assert_eq!(f.name, "ema");
        assert!(matches!(f.expr, SpecExpr::Scan(_)));
    }

    #[test]
    fn parse_sqrt_function() {
        let e = parse_expr("sqrt(x)").unwrap();
        assert!(matches!(e, SpecExpr::UnOp(UnOp::Sqrt, _)));
    }

    #[test]
    fn parse_unicode_sqrt() {
        let e = parse_expr("√(x)").unwrap();
        assert!(matches!(e, SpecExpr::UnOp(UnOp::Sqrt, _)));
    }

    #[test]
    fn parse_absolute_value() {
        let e = parse_expr("|x|").unwrap();
        assert!(matches!(e, SpecExpr::UnOp(UnOp::Abs, _)));
    }

    #[test]
    fn parse_count() {
        let e = parse_expr("Σ(x) / N").unwrap();
        assert!(matches!(e, SpecExpr::BinOp(BinOp::Div, _, _)));
    }

    #[test]
    fn parse_multi_formula_spec() {
        let src = "mean = Σ(x) / N\nvar = Σ(x^2) / N - (Σ(x) / N)^2";
        let formulas = parse_spec(src).unwrap();
        assert_eq!(formulas.len(), 2);
        assert_eq!(formulas[0].name, "mean");
        assert_eq!(formulas[1].name, "var");
    }

    #[test]
    fn parse_nested_max() {
        let e = parse_expr("Σ(max(Δx, 0))").unwrap();
        // Σ of something — the inner should involve max
        assert!(matches!(e, SpecExpr::Accumulate { op: BinOp::Add, .. }));
    }

    // ── Compilation ──────────────────────────────────────────────────

    #[test]
    fn compile_vwap_one_pass() {
        let f = parse_formula("vwap = Σ(p*v) / Σ(v)").unwrap();
        let plan = compile_one(&f);

        // VWAP needs: Σ(p*v), Σ(v), then division
        assert_eq!(plan.n_passes, 1, "VWAP should be 1 pass");
        let acc_count = plan.steps.iter()
            .filter(|s| matches!(s, PlanStep::Accumulate(_)))
            .count();
        assert_eq!(acc_count, 2, "VWAP needs 2 accumulates: Σ(p*v) and Σ(v)");
    }

    #[test]
    fn compile_mean_one_pass() {
        let f = parse_formula("mean = Σ(x) / N").unwrap();
        let plan = compile_one(&f);
        assert_eq!(plan.n_passes, 1);
    }

    #[test]
    fn compile_variance_shared_subexpr() {
        let src = "mean = Σ(x) / N\nvar = Σ(x^2) / N - (Σ(x) / N)^2";
        let formulas = parse_spec(src).unwrap();
        let plan = compile(&formulas);

        // Σ(x) should be shared between mean and variance
        let acc_count = plan.steps.iter()
            .filter(|s| matches!(s, PlanStep::Accumulate(_)))
            .count();
        // Should be 2, not 3: Σ(x) is shared, Σ(x²) is unique
        assert_eq!(acc_count, 2, "Σ(x) should be shared: {} accumulates", acc_count);
        assert_eq!(plan.n_passes, 1);
    }

    #[test]
    fn compile_rsi_needs_gather() {
        // Simplified RSI
        let f = parse_formula("rsi = 100 - 100 / (1 + Σ(max(Δx, 0)) / Σ(max(-Δx, 0)))").unwrap();
        let plan = compile_one(&f);

        // RSI needs a gather pass (for Δx) then accumulate pass
        assert_eq!(plan.n_passes, 2, "RSI needs 2 passes (gather + accumulate)");

        let gather_count = plan.steps.iter()
            .filter(|s| matches!(s, PlanStep::Gather(_)))
            .count();
        assert!(gather_count >= 1, "RSI needs at least 1 gather for Δx");
    }

    #[test]
    fn compile_ema_scan() {
        let f = parse_formula("ema = scan(a*x + (1 - a)*acc)").unwrap();
        let plan = compile_one(&f);

        let has_prefix = plan.steps.iter().any(|s| {
            matches!(s, PlanStep::Accumulate(a) if a.grouping == GroupingTag::Prefix)
        });
        assert!(has_prefix, "EMA should compile to a prefix scan");
    }

    #[test]
    fn compile_bollinger_shared() {
        let src = "\
            mean = Σ(x) / N\n\
            std = sqrt(Σ(x^2) / N - (Σ(x) / N)^2)\n\
            bb_upper = Σ(x) / N + 2 * sqrt(Σ(x^2) / N - (Σ(x) / N)^2)";
        let formulas = parse_spec(src).unwrap();
        let plan = compile(&formulas);

        // All three formulas share Σ(x), mean and bb_upper also share it
        let acc_count = plan.steps.iter()
            .filter(|s| matches!(s, PlanStep::Accumulate(_)))
            .count();
        // Σ(x) + Σ(x²) = 2 unique accumulates
        assert_eq!(acc_count, 2, "Bollinger should have 2 unique accumulates, got {}", acc_count);
        assert_eq!(plan.n_passes, 1);
    }

    #[test]
    fn compile_display_plan() {
        let f = parse_formula("vwap = Σ(p*v) / Σ(v)").unwrap();
        let plan = compile_one(&f);
        let display = format!("{plan}");
        assert!(display.contains("Execution Plan"));
        assert!(display.contains("ACCUMULATE"));
        assert!(display.contains("ARITH"));
    }

    // ── Round-trip: parse → compile → verify structure ──────────────

    #[test]
    fn roundtrip_max_min() {
        let src = "range = max(x) - min(x)";
        let f = parse_formula(src).unwrap();
        let plan = compile_one(&f);

        let has_max = plan.steps.iter().any(|s| {
            matches!(s, PlanStep::Accumulate(a) if a.op == BinOp::Max)
        });
        let has_min = plan.steps.iter().any(|s| {
            matches!(s, PlanStep::Accumulate(a) if a.op == BinOp::Min)
        });
        assert!(has_max, "range needs max accumulate");
        assert!(has_min, "range needs min accumulate");
        assert_eq!(plan.n_passes, 1);
    }

    #[test]
    fn parse_rsi_full() {
        // Full RSI with max(Δx, 0) — the inner comma-separated max
        // Actually our max() is an accumulate (over all elements).
        // For RSI, max(Δx, 0) is an element-wise clamp, not a reduction.
        // We handle it as max applied to each element.
        let f = parse_formula("up = Σ(max(Δx, 0))").unwrap();
        assert_eq!(f.name, "up");
    }
}
