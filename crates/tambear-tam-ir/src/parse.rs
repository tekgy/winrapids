//! Text parser: `.tam` text → AST.
//!
//! Implements a hand-written recursive descent parser. No parser combinator
//! libraries, no proc-macros, no deps — just `std`.
//!
//! ## Input format
//!
//! - Lines are whitespace-separated tokens.
//! - Comments begin with `;` and run to end of line.
//! - The header must appear first (`.tam`, `.target`).
//! - Functions come before kernels.
//!
//! ## f64 literal formats accepted
//!
//! The printer emits bit-exact hex (`0x<16 hex digits>`). The parser also
//! accepts human-readable forms for hand-written `.tam` files:
//! - `0x3ff0000000000000` — bit-exact hex (16 lowercase hex digits)
//! - `1.0`, `-0.5`, `3.14e7` — standard decimal float
//! - `inf`, `-inf` — infinities
//! - `nan` — canonical NaN
//!
//! ## Error handling
//!
//! `ParseError` carries a line number and a description. There is no recovery;
//! the first error stops parsing.

use crate::ast::*;

// ═══════════════════════════════════════════════════════════════════
// Error type
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl ParseError {
    fn new(line: usize, msg: impl Into<String>) -> Self {
        ParseError { line, message: msg.into() }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

type PResult<T> = Result<T, ParseError>;

// ═══════════════════════════════════════════════════════════════════
// Lexer / token stream
// ═══════════════════════════════════════════════════════════════════

/// A stream of non-comment, non-empty lines, each tokenized.
struct Lexer {
    /// (original_line_number, tokens)
    lines: Vec<(usize, Vec<String>)>,
    pos: usize,
}

impl Lexer {
    fn new(src: &str) -> Self {
        let lines = src
            .lines()
            .enumerate()
            .filter_map(|(i, raw)| {
                // Strip comments
                let without_comment = if let Some(pos) = raw.find(';') {
                    &raw[..pos]
                } else {
                    raw
                };
                let tokens: Vec<String> = without_comment
                    .split_whitespace()
                    .map(String::from)
                    .collect();
                if tokens.is_empty() {
                    None
                } else {
                    Some((i + 1, tokens)) // 1-indexed lines
                }
            })
            .collect();
        Lexer { lines, pos: 0 }
    }

    fn is_done(&self) -> bool {
        self.pos >= self.lines.len()
    }

    fn peek_line(&self) -> Option<(usize, &[String])> {
        self.lines.get(self.pos).map(|(ln, toks)| (*ln, toks.as_slice()))
    }

    #[allow(dead_code)]
    fn current_lineno(&self) -> usize {
        self.lines.get(self.pos).map(|(ln, _)| *ln).unwrap_or(0)
    }

    fn consume_line(&mut self) -> Option<(usize, Vec<String>)> {
        if self.pos < self.lines.len() {
            let line = self.lines[self.pos].clone();
            self.pos += 1;
            Some(line)
        } else {
            None
        }
    }

    /// Consume and return the next line, error if done.
    fn expect_line(&mut self, context: &str) -> PResult<(usize, Vec<String>)> {
        self.consume_line()
            .ok_or_else(|| ParseError::new(0, format!("unexpected end of file in {}", context)))
    }
}

// ═══════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════

/// Parse a `.tam` source string into a `Program`.
pub fn parse_program(src: &str) -> PResult<Program> {
    let mut lex = Lexer::new(src);
    let (version, target) = parse_header(&mut lex)?;
    let mut funcs = Vec::new();
    let mut kernels = Vec::new();

    while !lex.is_done() {
        let (ln, toks) = lex.peek_line().unwrap();
        match toks.first().map(String::as_str) {
            Some("func") => funcs.push(parse_func(&mut lex)?),
            Some("kernel") => kernels.push(parse_kernel(&mut lex)?),
            _ => {
                return Err(ParseError::new(
                    ln,
                    format!("expected 'func' or 'kernel', got {:?}", toks),
                ))
            }
        }
    }

    Ok(Program { version, target, funcs, kernels })
}

// ═══════════════════════════════════════════════════════════════════
// Header
// ═══════════════════════════════════════════════════════════════════

fn parse_header(lex: &mut Lexer) -> PResult<(TamVersion, Target)> {
    // Line 1: `.tam <version>`
    let (ln, toks) = lex.expect_line("header")?;
    if toks.len() < 2 || toks[0] != ".tam" {
        return Err(ParseError::new(ln, "expected '.tam <version>' as first line"));
    }
    let version = parse_version(ln, &toks[1])?;

    // Line 2: `.target <target>`
    let (ln2, toks2) = lex.expect_line("header .target")?;
    if toks2.len() < 2 || toks2[0] != ".target" {
        return Err(ParseError::new(ln2, "expected '.target <target>'"));
    }
    let target = match toks2[1].as_str() {
        "cross" => Target::Cross,
        other => Target::Other(other.to_string()),
    };

    Ok((version, target))
}

fn parse_version(ln: usize, s: &str) -> PResult<TamVersion> {
    let parts: Vec<&str> = s.splitn(2, '.').collect();
    if parts.len() != 2 {
        return Err(ParseError::new(ln, format!("bad version format: {s:?}")));
    }
    let major = parts[0].parse::<u32>()
        .map_err(|_| ParseError::new(ln, format!("bad major version: {}", parts[0])))?;
    let minor = parts[1].parse::<u32>()
        .map_err(|_| ParseError::new(ln, format!("bad minor version: {}", parts[1])))?;
    Ok(TamVersion { major, minor })
}

// ═══════════════════════════════════════════════════════════════════
// Functions
// ═══════════════════════════════════════════════════════════════════

fn parse_func(lex: &mut Lexer) -> PResult<FuncDef> {
    // `func <name>(<params>) -> f64 {`
    let (ln, toks) = lex.expect_line("func header")?;
    // Rejoin to handle the full signature in one string
    let line = toks.join(" ");
    let (name, params) = parse_func_signature(ln, &line)?;

    // `entry:`
    parse_entry_label(lex)?;

    // Body: ops until `}`
    let body = parse_func_body(lex, ln)?;

    Ok(FuncDef { name, params, body })
}

fn parse_func_signature(ln: usize, line: &str) -> PResult<(String, Vec<FuncParam>)> {
    // "func <name>( f64 %a , f64 %b ) -> f64 {"
    let rest = line.strip_prefix("func ").ok_or_else(|| ParseError::new(ln, "expected 'func'"))?;
    let paren_open = rest.find('(').ok_or_else(|| ParseError::new(ln, "missing '(' in func signature"))?;
    let name = rest[..paren_open].trim().to_string();
    let paren_close = rest.rfind(')').ok_or_else(|| ParseError::new(ln, "missing ')' in func signature"))?;
    let params_str = &rest[paren_open + 1..paren_close];
    let params = if params_str.trim().is_empty() {
        vec![]
    } else {
        params_str.split(',')
            .map(|p| parse_func_param(ln, p.trim()))
            .collect::<PResult<_>>()?
    };
    Ok((name, params))
}

fn parse_func_param(ln: usize, s: &str) -> PResult<FuncParam> {
    // "f64 %acc"
    let toks: Vec<&str> = s.split_whitespace().collect();
    if toks.len() != 2 {
        return Err(ParseError::new(ln, format!("bad func param: {s:?}")));
    }
    if toks[0] != "f64" {
        return Err(ParseError::new(ln, format!("func params must be f64, got {}", toks[0])));
    }
    Ok(FuncParam { reg: parse_reg(ln, toks[1])? })
}

fn parse_func_body(lex: &mut Lexer, _start_ln: usize) -> PResult<Vec<Op>> {
    let mut ops = Vec::new();
    loop {
        let (ln, toks) = lex.expect_line("func body")?;
        if toks.len() == 1 && toks[0] == "}" {
            break;
        }
        let op = parse_op(ln, &toks)?;
        ops.push(op);
    }
    Ok(ops)
}

// ═══════════════════════════════════════════════════════════════════
// Kernels
// ═══════════════════════════════════════════════════════════════════

fn parse_kernel(lex: &mut Lexer) -> PResult<KernelDef> {
    let (ln, toks) = lex.expect_line("kernel header")?;
    let line = toks.join(" ");
    let (name, params) = parse_kernel_signature(ln, &line)?;

    parse_entry_label(lex)?;

    let body = parse_kernel_body(lex, ln)?;

    Ok(KernelDef { name, params, attrs: vec![], body })
}

fn parse_kernel_signature(ln: usize, line: &str) -> PResult<(String, Vec<KernelParam>)> {
    let rest = line.strip_prefix("kernel ")
        .ok_or_else(|| ParseError::new(ln, "expected 'kernel'"))?;
    let paren_open = rest.find('(')
        .ok_or_else(|| ParseError::new(ln, "missing '(' in kernel signature"))?;
    let name = rest[..paren_open].trim().to_string();
    let paren_close = rest.rfind(')')
        .ok_or_else(|| ParseError::new(ln, "missing ')' in kernel signature"))?;
    let params_str = &rest[paren_open + 1..paren_close];
    let params = if params_str.trim().is_empty() {
        vec![]
    } else {
        params_str.split(',')
            .map(|p| parse_kernel_param(ln, p.trim()))
            .collect::<PResult<_>>()?
    };
    Ok((name, params))
}

fn parse_kernel_param(ln: usize, s: &str) -> PResult<KernelParam> {
    // "buf<f64> %data" or "i32 %n"
    let toks: Vec<&str> = s.splitn(2, ' ').collect();
    if toks.len() != 2 {
        return Err(ParseError::new(ln, format!("bad kernel param: {s:?}")));
    }
    let ty = match toks[0].trim() {
        "buf<f64>" => Ty::BufF64,
        "i32" => Ty::I32,
        "i64" => Ty::I64,
        "f64" => Ty::F64,
        other => return Err(ParseError::new(ln, format!("unknown kernel param type: {other}"))),
    };
    let reg = parse_reg(ln, toks[1].trim())?;
    Ok(KernelParam { ty, reg })
}

fn parse_kernel_body(lex: &mut Lexer, _start_ln: usize) -> PResult<Vec<Stmt>> {
    let mut stmts = Vec::new();
    loop {
        let (ln, toks) = lex.expect_line("kernel body")?;
        if toks.len() == 1 && toks[0] == "}" {
            break;
        }
        // Loop?
        if toks[0] == "loop_grid_stride" {
            let lp = parse_loop(lex, ln, &toks)?;
            stmts.push(Stmt::Loop(lp));
        } else {
            let op = parse_op(ln, &toks)?;
            stmts.push(Stmt::Op(op));
        }
    }
    Ok(stmts)
}

// ═══════════════════════════════════════════════════════════════════
// Loop
// ═══════════════════════════════════════════════════════════════════

fn parse_loop(lex: &mut Lexer, ln: usize, toks: &[String]) -> PResult<LoopGridStride> {
    // `loop_grid_stride %i in [0, %n) {`
    // Tokens after joining: "loop_grid_stride", "%i", "in", "[0,", "%n)", "{"
    // We need to extract induction var and limit.
    if toks.len() < 4 {
        return Err(ParseError::new(ln, format!("malformed loop_grid_stride: {:?}", toks)));
    }
    let induction = parse_reg(ln, &toks[1])?;

    // Find the limit: it appears as "%name)" somewhere in the tokens.
    // Rejoin to find the pattern [0, %limit)
    let line = toks.join(" ");
    let limit = parse_loop_limit(ln, &line)?;

    // Now consume body until `}`
    let mut body = Vec::new();
    loop {
        let (bln, btoks) = lex.expect_line("loop body")?;
        if btoks.len() == 1 && btoks[0] == "}" {
            break;
        }
        body.push(parse_op(bln, &btoks)?);
    }

    Ok(LoopGridStride { induction, limit, body })
}

fn parse_loop_limit(ln: usize, line: &str) -> PResult<Reg> {
    // Pattern: "[0, %name)" or "[0, %name' )"
    // Find the opening `[0,` and closing `)`
    let bracket = line.find("[0,")
        .ok_or_else(|| ParseError::new(ln, "loop range must start with [0,"))?;
    let rest = &line[bracket + 3..]; // after "[0,"
    let paren = rest.find(')')
        .ok_or_else(|| ParseError::new(ln, "loop range missing ')'"))?;
    let limit_str = rest[..paren].trim();
    parse_reg(ln, limit_str)
}

// ═══════════════════════════════════════════════════════════════════
// Entry label
// ═══════════════════════════════════════════════════════════════════

fn parse_entry_label(lex: &mut Lexer) -> PResult<()> {
    let (ln, toks) = lex.expect_line("entry label")?;
    if toks.len() == 1 && toks[0] == "entry:" {
        Ok(())
    } else {
        Err(ParseError::new(ln, format!("expected 'entry:', got {:?}", toks)))
    }
}

// ═══════════════════════════════════════════════════════════════════
// Op parsing
// ═══════════════════════════════════════════════════════════════════

fn parse_op(ln: usize, toks: &[String]) -> PResult<Op> {
    // Two forms:
    //   %dst = <mnemonic> <operands...>    (value-producing ops)
    //   <mnemonic> <operands...>           (void ops: store, reduce, ret)
    if toks.is_empty() {
        return Err(ParseError::new(ln, "empty op line"));
    }

    // Void ops (no `%dst =` prefix)
    match toks[0].as_str() {
        "store.f64" => return parse_store_f64(ln, toks),
        "reduce_block_add.f64" => return parse_reduce_block_add(ln, toks),
        "ret.f64" => return parse_ret_f64(ln, toks),
        _ => {}
    }

    // Value-producing: `%dst = <mnemonic> <operands...>`
    if toks.len() < 3 || toks[1] != "=" {
        return Err(ParseError::new(ln, format!("expected '%dst = <op> ...', got {:?}", toks)));
    }
    let dst = parse_reg(ln, &toks[0])?;
    let mnemonic = &toks[2];
    let operands = &toks[3..];

    match mnemonic.as_str() {
        "const.f64" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "const.f64 takes one literal"));
            }
            Ok(Op::ConstF64 { dst, value: parse_f64(ln, &operands[0])? })
        }
        "const.i32" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "const.i32 takes one literal"));
            }
            Ok(Op::ConstI32 { dst, value: parse_i32(ln, &operands[0])? })
        }
        "const.i64" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "const.i64 takes one literal"));
            }
            Ok(Op::ConstI64 { dst, value: parse_i64(ln, &operands[0])? })
        }
        "bufsize" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "bufsize takes one register"));
            }
            Ok(Op::BufSize { dst, buf: parse_reg(ln, &operands[0])? })
        }
        "load.f64" => {
            // "load.f64 %buf, %idx"
            let (buf, idx) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::LoadF64 { dst, buf, idx })
        }
        "fadd.f64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::FAdd { dst, a, b })
        }
        "fsub.f64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::FSub { dst, a, b })
        }
        "fmul.f64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::FMul { dst, a, b })
        }
        "fdiv.f64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::FDiv { dst, a, b })
        }
        "fsqrt.f64" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "fsqrt.f64 takes one register"));
            }
            Ok(Op::FSqrt { dst, a: parse_reg(ln, &operands[0])? })
        }
        "fneg.f64" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "fneg.f64 takes one register"));
            }
            Ok(Op::FNeg { dst, a: parse_reg(ln, &operands[0])? })
        }
        "fabs.f64" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "fabs.f64 takes one register"));
            }
            Ok(Op::FAbs { dst, a: parse_reg(ln, &operands[0])? })
        }
        "iadd.i32" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::IAdd { dst, a, b })
        }
        "isub.i32" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::ISub { dst, a, b })
        }
        "imul.i32" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::IMul { dst, a, b })
        }
        "icmp_lt" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::ICmpLt { dst, a, b })
        }
        "iadd.i64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::IAdd64 { dst, a, b })
        }
        "isub.i64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::ISub64 { dst, a, b })
        }
        "and.i64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::AndI64 { dst, a, b })
        }
        "or.i64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::OrI64 { dst, a, b })
        }
        "xor.i64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::XorI64 { dst, a, b })
        }
        "shl.i64" => {
            let (a, shift) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::ShlI64 { dst, a, shift })
        }
        "shr.i64" => {
            let (a, shift) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::ShrI64 { dst, a, shift })
        }
        "ldexp.f64" => {
            let (mantissa, exp) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::LdExpF64 { dst, mantissa, exp })
        }
        "f64_to_i32_rn" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "f64_to_i32_rn takes one register"));
            }
            Ok(Op::F64ToI32Rn { dst, a: parse_reg(ln, &operands[0])? })
        }
        "bitcast.f64.i64" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "bitcast.f64.i64 takes one register"));
            }
            Ok(Op::BitcastF64ToI64 { dst, a: parse_reg(ln, &operands[0])? })
        }
        "bitcast.i64.f64" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "bitcast.i64.f64 takes one register"));
            }
            Ok(Op::BitcastI64ToF64 { dst, a: parse_reg(ln, &operands[0])? })
        }
        "fcmp_gt.f64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::FCmpGt { dst, a, b })
        }
        "fcmp_lt.f64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::FCmpLt { dst, a, b })
        }
        "fcmp_eq.f64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::FCmpEq { dst, a, b })
        }
        "select.f64" => {
            let (pred, on_true, on_false) = parse_three_regs_comma(ln, operands)?;
            Ok(Op::SelectF64 { dst, pred, on_true, on_false })
        }
        "select.i32" => {
            let (pred, on_true, on_false) = parse_three_regs_comma(ln, operands)?;
            Ok(Op::SelectI32 { dst, pred, on_true, on_false })
        }
        "tam_exp.f64" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "tam_exp.f64 takes one register"));
            }
            Ok(Op::TamExp { dst, a: parse_reg(ln, &operands[0])? })
        }
        "tam_ln.f64" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "tam_ln.f64 takes one register"));
            }
            Ok(Op::TamLn { dst, a: parse_reg(ln, &operands[0])? })
        }
        "tam_sin.f64" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "tam_sin.f64 takes one register"));
            }
            Ok(Op::TamSin { dst, a: parse_reg(ln, &operands[0])? })
        }
        "tam_cos.f64" => {
            if operands.len() != 1 {
                return Err(ParseError::new(ln, "tam_cos.f64 takes one register"));
            }
            Ok(Op::TamCos { dst, a: parse_reg(ln, &operands[0])? })
        }
        "tam_pow.f64" => {
            let (a, b) = parse_two_regs_comma(ln, operands)?;
            Ok(Op::TamPow { dst, a, b })
        }
        other => Err(ParseError::new(ln, format!("unknown mnemonic: {other:?}"))),
    }
}

fn parse_store_f64(ln: usize, toks: &[String]) -> PResult<Op> {
    // "store.f64 %buf, %idx, %val"
    if toks.len() < 4 {
        return Err(ParseError::new(ln, "store.f64 requires: %buf, %idx, %val"));
    }
    let operands = &toks[1..];
    let (buf, idx, val) = parse_three_regs_comma(ln, operands)?;
    Ok(Op::StoreF64 { buf, idx, val })
}

fn parse_reduce_block_add(ln: usize, toks: &[String]) -> PResult<Op> {
    // "reduce_block_add.f64 %out_buf, %slot_idx, %val @order(<strategy>)"
    // The three register operands come first (comma-separated), then the
    // optional @order(...) attribute token. If @order is absent, we error:
    // BackendDefault is rejected at parse time to give a useful message.
    if toks.len() < 4 {
        return Err(ParseError::new(ln, "reduce_block_add.f64 requires: %out, %slot, %val @order(...)"));
    }
    // Find the @order(...) token. The three registers may have trailing commas.
    // Strategy: find the token starting with "@order" and treat everything
    // before it (after the mnemonic) as the three register operands.
    let at_order_pos = toks.iter().position(|t| t.starts_with("@order"));
    let (reg_toks, order_tok) = match at_order_pos {
        Some(pos) => (&toks[1..pos], Some(&toks[pos])),
        None => (&toks[1..], None),
    };
    let (out_buf, slot_idx, val) = parse_three_regs_comma(ln, reg_toks)?;
    let order = match order_tok {
        None => return Err(ParseError::new(ln,
            "reduce_block_add.f64 requires an @order(...) attribute: \
             @order(sequential_left) or @order(tree_fixed_fanout(N)). \
             BackendDefault is not permitted.")),
        Some(tok) => parse_order_strategy(ln, tok)?,
    };
    Ok(Op::ReduceBlockAdd { out_buf, slot_idx, val, order })
}

fn parse_order_strategy(ln: usize, tok: &str) -> PResult<OrderStrategy> {
    // Accepted forms:
    //   @order(sequential_left)
    //   @order(tree_fixed_fanout(N))
    //   @order(backend_default)    — parsed but rejected by verifier
    let inner = tok
        .strip_prefix("@order(")
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| ParseError::new(ln, format!("malformed @order attribute: {tok:?}")))?;
    if inner == "sequential_left" {
        Ok(OrderStrategy::SequentialLeft)
    } else if inner == "backend_default" {
        Ok(OrderStrategy::BackendDefault)
    } else if let Some(rest) = inner.strip_prefix("tree_fixed_fanout(") {
        let n_str = rest.strip_suffix(')')
            .ok_or_else(|| ParseError::new(ln, format!("malformed tree_fixed_fanout: {inner:?}")))?;
        let n = n_str.parse::<u32>()
            .map_err(|_| ParseError::new(ln, format!("tree_fixed_fanout fanout must be u32, got {n_str:?}")))?;
        Ok(OrderStrategy::TreeFixedFanout(n))
    } else {
        Err(ParseError::new(ln, format!("unknown order strategy: {inner:?}")))
    }
}

fn parse_ret_f64(ln: usize, toks: &[String]) -> PResult<Op> {
    if toks.len() != 2 {
        return Err(ParseError::new(ln, "ret.f64 takes one register"));
    }
    Ok(Op::RetF64 { val: parse_reg(ln, &toks[1])? })
}

// ═══════════════════════════════════════════════════════════════════
// Register parsing
// ═══════════════════════════════════════════════════════════════════

fn parse_reg(ln: usize, s: &str) -> PResult<Reg> {
    // Strip trailing comma if present (comes from comma-separated lists)
    let s = s.trim_end_matches(',');
    if !s.starts_with('%') {
        return Err(ParseError::new(ln, format!("expected register (starts with %), got {s:?}")));
    }
    let rest = &s[1..]; // drop `%`
    if rest.ends_with('\'') {
        let name = &rest[..rest.len() - 1];
        validate_ident(ln, name)?;
        Ok(Reg { name: name.to_string(), prime: true })
    } else {
        validate_ident(ln, rest)?;
        Ok(Reg { name: rest.to_string(), prime: false })
    }
}

fn validate_ident(ln: usize, s: &str) -> PResult<()> {
    if s.is_empty() {
        return Err(ParseError::new(ln, "empty identifier"));
    }
    let first = s.chars().next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return Err(ParseError::new(ln, format!("identifier must start with letter or _, got {s:?}")));
    }
    if !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(ParseError::new(ln, format!("identifier contains invalid chars: {s:?}")));
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// Literal parsing
// ═══════════════════════════════════════════════════════════════════

/// Parse an f64 literal.
///
/// Accepts:
/// - `0x<16 hex digits>` — bit-exact (printer format)
/// - `inf`, `-inf` — infinities
/// - `nan` — canonical NaN
/// - Any decimal float that Rust's f64::from_str accepts
fn parse_f64(ln: usize, s: &str) -> PResult<f64> {
    if let Some(hex) = s.strip_prefix("0x") {
        // Bit-exact hex: 16 lowercase hex digits
        if hex.len() != 16 {
            return Err(ParseError::new(ln, format!("hex f64 must be exactly 16 hex digits, got {hex:?}")));
        }
        let bits = u64::from_str_radix(hex, 16)
            .map_err(|_| ParseError::new(ln, format!("invalid hex f64: {s:?}")))?;
        Ok(f64::from_bits(bits))
    } else {
        match s {
            "inf"  => Ok(f64::INFINITY),
            "-inf" => Ok(f64::NEG_INFINITY),
            "nan"  => Ok(f64::NAN),
            other  => other.parse::<f64>()
                .map_err(|_| ParseError::new(ln, format!("invalid f64 literal: {other:?}"))),
        }
    }
}

fn parse_i32(ln: usize, s: &str) -> PResult<i32> {
    s.parse::<i32>()
        .map_err(|_| ParseError::new(ln, format!("invalid i32 literal: {s:?}")))
}

fn parse_i64(ln: usize, s: &str) -> PResult<i64> {
    s.parse::<i64>()
        .map_err(|_| ParseError::new(ln, format!("invalid i64 literal: {s:?}")))
}

// ═══════════════════════════════════════════════════════════════════
// Multi-register helpers (strip trailing commas)
// ═══════════════════════════════════════════════════════════════════

fn parse_two_regs_comma(ln: usize, toks: &[String]) -> PResult<(Reg, Reg)> {
    // Input: ["%a,", "%b"] or ["%a", "%b"] or ["%a,", "%b,"]
    let joined = toks.join(" ");
    let parts: Vec<&str> = joined.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if parts.len() < 2 {
        return Err(ParseError::new(ln, format!("expected 2 register operands, got: {:?}", toks)));
    }
    Ok((parse_reg(ln, parts[0])?, parse_reg(ln, parts[1])?))
}

fn parse_three_regs_comma(ln: usize, toks: &[String]) -> PResult<(Reg, Reg, Reg)> {
    let joined = toks.join(" ");
    let parts: Vec<&str> = joined.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if parts.len() < 3 {
        return Err(ParseError::new(ln, format!("expected 3 register operands, got: {:?}", toks)));
    }
    Ok((parse_reg(ln, parts[0])?, parse_reg(ln, parts[1])?, parse_reg(ln, parts[2])?))
}

// ═══════════════════════════════════════════════════════════════════
// Tests (campsite 1.7 acceptance criteria)
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::variance_pass_program;
    use crate::print::print_program;

    #[test]
    fn parse_header_only() {
        let src = ".tam 0.1\n.target cross\n";
        let prog = parse_program(src).unwrap();
        assert_eq!(prog.version, TamVersion::PHASE1);
        assert_eq!(prog.target, Target::Cross);
        assert!(prog.kernels.is_empty());
    }

    #[test]
    fn parse_error_bad_header() {
        let src = "not a tam file\n";
        assert!(parse_program(src).is_err());
    }

    #[test]
    fn parse_const_f64_hex() {
        // 1.0 = 0x3ff0000000000000
        let src = ".tam 0.1\n.target cross\nkernel k() {\nentry:\n  %x = const.f64 0x3ff0000000000000\n}\n";
        let prog = parse_program(src).unwrap();
        let kernel = prog.kernel("k").unwrap();
        match &kernel.body[0] {
            Stmt::Op(Op::ConstF64 { value, .. }) => {
                assert_eq!(value.to_bits(), 1.0f64.to_bits());
            }
            other => panic!("expected ConstF64, got {:?}", other),
        }
    }

    #[test]
    fn parse_const_f64_decimal() {
        let src = ".tam 0.1\n.target cross\nkernel k() {\nentry:\n  %x = const.f64 1.0\n}\n";
        let prog = parse_program(src).unwrap();
        match &prog.kernel("k").unwrap().body[0] {
            Stmt::Op(Op::ConstF64 { value, .. }) => assert_eq!(*value, 1.0),
            other => panic!("{:?}", other),
        }
    }

    #[test]
    fn roundtrip_variance_pass() {
        // Build AST → print → parse → compare
        let original = variance_pass_program();
        let text = print_program(&original);
        let parsed = parse_program(&text)
            .unwrap_or_else(|e| panic!("parse failed: {}\n\ntext was:\n{}", e, text));
        assert_programs_eq(&original, &parsed);
    }

    /// Compare two programs structurally, using bit-equality for f64 constants.
    fn assert_programs_eq(a: &Program, b: &Program) {
        assert_eq!(a.version, b.version);
        assert_eq!(a.target, b.target);
        assert_eq!(a.funcs.len(), b.funcs.len(), "func count mismatch");
        assert_eq!(a.kernels.len(), b.kernels.len(), "kernel count mismatch");
        for (ka, kb) in a.kernels.iter().zip(b.kernels.iter()) {
            assert_eq!(ka.name, kb.name);
            assert_eq!(ka.params, kb.params);
            assert_eq!(ka.body.len(), kb.body.len(), "body length mismatch in {}", ka.name);
            for (sa, sb) in ka.body.iter().zip(kb.body.iter()) {
                assert_stmts_eq(sa, sb);
            }
        }
    }

    fn assert_stmts_eq(a: &Stmt, b: &Stmt) {
        match (a, b) {
            (Stmt::Op(oa), Stmt::Op(ob)) => assert_ops_eq(oa, ob),
            (Stmt::Loop(la), Stmt::Loop(lb)) => {
                assert_eq!(la.induction, lb.induction);
                assert_eq!(la.limit, lb.limit);
                for (oa, ob) in la.body.iter().zip(lb.body.iter()) {
                    assert_ops_eq(oa, ob);
                }
            }
            _ => panic!("stmt type mismatch: {:?} vs {:?}", a, b),
        }
    }

    fn assert_ops_eq(a: &Op, b: &Op) {
        // For ConstF64, use bit equality to handle NaN/subnormal correctly.
        match (a, b) {
            (Op::ConstF64 { dst: da, value: va }, Op::ConstF64 { dst: db, value: vb }) => {
                assert_eq!(da, db);
                assert_eq!(va.to_bits(), vb.to_bits(), "f64 constant bit mismatch");
            }
            _ => assert_eq!(a, b),
        }
    }

    #[test]
    fn parse_sum_all_add_from_file_content() {
        let src = include_str!("../../../campsites/expedition/20260411120000-the-bit-exact-trek/peak1-tam-ir/programs/sum_all_add.tam");
        let prog = parse_program(src)
            .unwrap_or_else(|e| panic!("parse failed: {}", e));
        assert_eq!(prog.kernel("sum_all_add").unwrap().name, "sum_all_add");
    }

    #[test]
    fn parse_variance_pass_from_file_content() {
        let src = include_str!("../../../campsites/expedition/20260411120000-the-bit-exact-trek/peak1-tam-ir/programs/variance_pass.tam");
        let prog = parse_program(src)
            .unwrap_or_else(|e| panic!("parse failed: {}", e));
        assert_eq!(prog.kernel("variance_pass").unwrap().params.len(), 2);
    }

    #[test]
    fn parse_pearson_r_from_file_content() {
        let src = include_str!("../../../campsites/expedition/20260411120000-the-bit-exact-trek/peak1-tam-ir/programs/pearson_r_pass.tam");
        let prog = parse_program(src)
            .unwrap_or_else(|e| panic!("parse failed: {}", e));
        assert_eq!(prog.kernel("pearson_r_pass").unwrap().params.len(), 3);
    }

    // ── New op parse tests (added with i64 / bitcast / ldexp / f64_to_i32_rn ops) ──

    #[test]
    fn parse_bitcast_f64_to_i64() {
        let src = ".tam 0.1\n.target cross\nkernel k(buf<f64> %d) {\nentry:\n  %n = bufsize %d\n  %c = const.f64 1.0\n  %b = bitcast.f64.i64 %c\n}\n";
        let prog = parse_program(src).unwrap();
        match &prog.kernel("k").unwrap().body[2] {
            Stmt::Op(Op::BitcastF64ToI64 { dst, a }) => {
                assert_eq!(dst.name, "b");
                assert_eq!(a.name, "c");
            }
            other => panic!("expected BitcastF64ToI64, got {:?}", other),
        }
    }

    #[test]
    fn parse_bitcast_i64_to_f64() {
        let src = ".tam 0.1\n.target cross\nkernel k() {\nentry:\n  %x = const.i64 4607182418800017408\n  %f = bitcast.i64.f64 %x\n}\n";
        let prog = parse_program(src).unwrap();
        match &prog.kernel("k").unwrap().body[1] {
            Stmt::Op(Op::BitcastI64ToF64 { dst, a }) => {
                assert_eq!(dst.name, "f");
                assert_eq!(a.name, "x");
            }
            other => panic!("expected BitcastI64ToF64, got {:?}", other),
        }
    }

    #[test]
    fn parse_f64_to_i32_rn() {
        let src = ".tam 0.1\n.target cross\nfunc f(f64 %x) -> f64 {\nentry:\n  %n = f64_to_i32_rn %x\n  %c = const.f64 0.0\n  ret.f64 %c\n}\n";
        let prog = parse_program(src).unwrap();
        match &prog.funcs[0].body[0] {
            Op::F64ToI32Rn { dst, a } => {
                assert_eq!(dst.name, "n");
                assert_eq!(a.name, "x");
            }
            other => panic!("expected F64ToI32Rn, got {:?}", other),
        }
    }

    #[test]
    fn parse_ldexp_f64() {
        let src = ".tam 0.1\n.target cross\nfunc f(f64 %m) -> f64 {\nentry:\n  %e = const.i32 3\n  %r = ldexp.f64 %m, %e\n  ret.f64 %r\n}\n";
        let prog = parse_program(src).unwrap();
        match &prog.funcs[0].body[1] {
            Op::LdExpF64 { dst, mantissa, exp } => {
                assert_eq!(dst.name, "r");
                assert_eq!(mantissa.name, "m");
                assert_eq!(exp.name, "e");
            }
            other => panic!("expected LdExpF64, got {:?}", other),
        }
    }

    #[test]
    fn parse_i64_arithmetic() {
        let src = ".tam 0.1\n.target cross\nkernel k() {\nentry:\n  %a = const.i64 100\n  %b = const.i64 200\n  %c = iadd.i64 %a, %b\n  %d = isub.i64 %c, %a\n  %e = and.i64 %a, %b\n  %f = or.i64 %a, %b\n  %g = xor.i64 %a, %b\n}\n";
        let prog = parse_program(src).unwrap();
        let body = &prog.kernel("k").unwrap().body;
        assert!(matches!(body[0], Stmt::Op(Op::ConstI64 { .. })));
        assert!(matches!(body[2], Stmt::Op(Op::IAdd64 { .. })));
        assert!(matches!(body[3], Stmt::Op(Op::ISub64 { .. })));
        assert!(matches!(body[4], Stmt::Op(Op::AndI64 { .. })));
        assert!(matches!(body[5], Stmt::Op(Op::OrI64  { .. })));
        assert!(matches!(body[6], Stmt::Op(Op::XorI64 { .. })));
    }

    #[test]
    fn parse_i64_shifts() {
        let src = ".tam 0.1\n.target cross\nkernel k() {\nentry:\n  %a = const.i64 1\n  %n = const.i32 5\n  %b = shl.i64 %a, %n\n  %c = shr.i64 %b, %n\n}\n";
        let prog = parse_program(src).unwrap();
        let body = &prog.kernel("k").unwrap().body;
        assert!(matches!(body[2], Stmt::Op(Op::ShlI64 { .. })));
        assert!(matches!(body[3], Stmt::Op(Op::ShrI64 { .. })));
    }

    /// Roundtrip test: build a program with new ops by-hand, print it, re-parse it.
    #[test]
    fn roundtrip_new_ops_bitcast_and_conversion() {
        // Build a function that uses BitcastF64ToI64, BitcastI64ToF64, F64ToI32Rn
        let prog = Program {
            version: TamVersion::PHASE1,
            target: Target::Cross,
            funcs: vec![FuncDef {
                name: "probe_new_ops".into(),
                params: vec![FuncParam { reg: Reg::new("x") }],
                body: vec![
                    Op::BitcastF64ToI64 { dst: Reg::new("xi"), a: Reg::new("x") },
                    Op::BitcastI64ToF64 { dst: Reg::new("xf"), a: Reg::new("xi") },
                    Op::F64ToI32Rn { dst: Reg::new("n"), a: Reg::new("x") },
                    Op::RetF64 { val: Reg::new("xf") },
                ],
            }],
            kernels: vec![],
        };
        use crate::print::print_program;
        let text = print_program(&prog);
        let parsed = parse_program(&text)
            .unwrap_or_else(|e| panic!("parse failed: {}\n\ntext:\n{}", e, text));
        assert_eq!(parsed.funcs[0].body.len(), 4);
        assert!(matches!(parsed.funcs[0].body[0], Op::BitcastF64ToI64 { .. }));
        assert!(matches!(parsed.funcs[0].body[1], Op::BitcastI64ToF64 { .. }));
        assert!(matches!(parsed.funcs[0].body[2], Op::F64ToI32Rn { .. }));
    }
}
