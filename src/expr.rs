//! Cepstral's target cost: a Lisp expression tree stored in `voice_u.dat`.
//!
//! Cons cells are `cst_val` type 0x37, which packs two 24-bit self-relative
//! offsets into 8 bytes with the high byte biased by 0x20 (swift.dll @0x1d5f4
//! for car, @0x1d634 for cdr):
//!
//! ```text
//! car = (((b[4] - 0x20) << 8) + b[3] << 8) + b[2]      0 means nil
//! cdr = (((b[7] - 0x20) << 8) + b[6] << 8) + b[5]
//! ptr = val_offset + that,  in 32-bit signed arithmetic
//! ```
//!
//! `cl_target_cost` (swift.dll @0x21420) builds an environment with three
//! bindings per entry in `unit_name_params/unit_features`:
//!
//! ```text
//! targ.<name> = ffeature_int(target_item, name)   -- from the utterance
//! cand.<name> = unit_feats[unit][k]               -- from voice_d.dat
//! <name>      = |targ.<name> - cand.<name>|       -- @0x2146d, integer abs
//! ```
//!
//! then evaluates the expression and truncates the f32 result to int (@0x21617).
//! Builtins, all returning 1.0 or 0.0: `=` `<` `>` `not` `zeroone`, plus `and`,
//! `cond`, and n-ary `*` and `+`.

use crate::val::{cstr, f32le, i32le, u16le};

const T_INT: u16 = 0x01;
const T_FLOAT: u16 = 0x03;
const T_STRING: u16 = 0x33;
const T_CONS: u16 = 0x37;

fn packed(b: &[u8], o: usize, lo: usize) -> Option<usize> {
    let hi = b[o + lo + 2] as i32 - 0x20;
    let v = ((hi << 8) + b[o + lo + 1] as i32) << 8;
    let v = v + b[o + lo] as i32;
    if v == 0 {
        None
    } else {
        Some((o as i64 + v as i64) as usize)
    }
}

pub fn car(b: &[u8], o: usize) -> Option<usize> {
    packed(b, o, 2)
}

pub fn cdr(b: &[u8], o: usize) -> Option<usize> {
    packed(b, o, 5)
}

/// Collect a cons list into the offsets of its elements.
pub fn list_items(b: &[u8], o: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut n = Some(o);
    let mut guard = 0;
    while let Some(p) = n {
        if u16le(b, p) != T_CONS || guard > 100_000 {
            break;
        }
        if let Some(c) = car(b, p) {
            out.push(c);
        }
        n = cdr(b, p);
        guard += 1;
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Mul,
    Add,
    Eq,
    Lt,
    Gt,
    Not,
    And,
    Or,
    Cond,
    ZeroOne,
    /// `(phonedist a b)`: not an operator but a registered function, and the
    /// only one these voices use. See [`crate::phonedist`].
    PhoneDist,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Num(f32),
    /// Index into the environment slots.
    Var(usize),
    /// A name that is not in `unit_features`; evaluates to 0.
    Missing(String),
    Call(Op, Vec<Expr>),
    /// `(test value)` clauses for `cond`.
    Clause(Box<Expr>, Box<Expr>),
}

/// Environment layout: three slots per feature.
#[derive(Debug, Clone, Copy)]
pub enum Slot {
    Diff(usize),
    Targ(usize),
    Cand(usize),
}

pub struct TargetCost {
    pub features: Vec<String>,
    pub expr: Expr,
    /// Parallel to the env vector: what each slot means.
    pub slots: Vec<Slot>,
    pub beam_width: usize,
    /// Names referenced by the expression that are not in `unit_features`.
    pub missing: Vec<String>,
}

fn op_of(name: &str) -> Option<Op> {
    Some(match name {
        "*" => Op::Mul,
        "+" => Op::Add,
        "=" => Op::Eq,
        "<" => Op::Lt,
        ">" => Op::Gt,
        "not" => Op::Not,
        "and" => Op::And,
        "or" => Op::Or,
        "cond" => Op::Cond,
        "zeroone" => Op::ZeroOne,
        "phonedist" => Op::PhoneDist,
        _ => return None,
    })
}

struct Builder<'a> {
    b: &'a [u8],
    index: std::collections::HashMap<String, usize>,
    missing: Vec<String>,
}

impl<'a> Builder<'a> {
    fn slot_for(&mut self, name: &str) -> Expr {
        if let Some(&i) = self.index.get(name) {
            return Expr::Var(i);
        }
        if !self.missing.iter().any(|m| m == name) {
            self.missing.push(name.to_string());
        }
        Expr::Missing(name.to_string())
    }

    fn build(&mut self, o: usize, depth: u32) -> Expr {
        if depth > 64 {
            return Expr::Num(0.0);
        }
        match u16le(self.b, o) {
            T_INT => Expr::Num(i32le(self.b, o + 4) as f32),
            T_FLOAT => Expr::Num(f32le(self.b, o + 4)),
            T_STRING => {
                let s = cstr(self.b, (o as i64 + i32le(self.b, o + 4) as i64) as usize);
                s.parse::<f32>().map(Expr::Num).unwrap_or_else(|_| self.slot_for(s))
            }
            T_CONS => {
                let items = list_items(self.b, o);
                if items.is_empty() {
                    return Expr::Num(0.0);
                }
                let head_is_op = u16le(self.b, items[0]) == T_STRING
                    && op_of(cstr(self.b,
                        (items[0] as i64 + i32le(self.b, items[0] + 4) as i64) as usize))
                        .is_some();
                if head_is_op {
                    let name = cstr(self.b,
                        (items[0] as i64 + i32le(self.b, items[0] + 4) as i64) as usize);
                    let op = op_of(name).unwrap();
                    let args: Vec<Expr> = if op == Op::Cond {
                        items[1..].iter().map(|&i| self.clause(i, depth + 1)).collect()
                    } else {
                        items[1..].iter().map(|&i| self.build(i, depth + 1)).collect()
                    };
                    Expr::Call(op, args)
                } else if items.len() == 1 {
                    // a one-element wrapper list, e.g. the outermost target_cost
                    self.build(items[0], depth + 1)
                } else {
                    // bare list used as a cond clause
                    self.clause(o, depth + 1)
                }
            }
            _ => Expr::Num(0.0),
        }
    }

    fn clause(&mut self, o: usize, depth: u32) -> Expr {
        let items = list_items(self.b, o);
        if items.len() >= 2 {
            let t = self.build(items[0], depth + 1);
            let v = self.build(items[1], depth + 1);
            Expr::Clause(Box::new(t), Box::new(v))
        } else if items.len() == 1 {
            let t = self.build(items[0], depth + 1);
            Expr::Clause(Box::new(t.clone()), Box::new(t))
        } else {
            Expr::Clause(Box::new(Expr::Num(0.0)), Box::new(Expr::Num(0.0)))
        }
    }
}

/// How well this build's target model reproduces a voice's own feature table.
///
/// Scoring a unit against its *own* stored features must cost nothing. Target
/// and candidate are the same vector here, so every difference term cancels and
/// anything left over is a term the expression reads raw -- which means we are
/// evaluating the expression wrongly, not building the target wrongly.
///
/// That distinction cost some time. Swift 4 and 5 voices scored 255000 and this
/// was read as the missing `lisp_durms` and `lisp_zscoredur_norm`, but
/// [`crate::target::build`] is never called from here, so no feature this probe
/// sees is ever computed. The real cause was `(phonedist a b)`, parsed as a bare
/// `cond` clause because `phonedist` is a registered function rather than an
/// operator: the clause dropped the second argument and returned the first, a
/// raw phone id around 50, which the surrounding weights scaled to six figures.
/// See [`crate::phonedist`].
///
/// So this checks the expression, and the duration features are checked against
/// captured engine calls instead -- `tests/oldvoices.rs`.
///
/// Cheap enough to run at load: it touches `probe` units and nothing else.
pub struct SelfCost {
    pub probed: usize,
    pub worst: i32,
    /// Features this build has no direct implementation for. Some are fine --
    /// a categorical one is learned from the unit database -- so this is only
    /// diagnostic context for a non-zero `worst`.
    pub unimplemented: Vec<String>,
}

impl SelfCost {
    pub fn is_exact(&self) -> bool {
        self.worst == 0
    }
}

impl TargetCost {
    /// Parse from the `unit_name_params` sub-table of `voice_u.dat`.
    pub fn parse(image: &[u8], unit_name_params: usize) -> Option<TargetCost> {
        let table = crate::val::walk(image, unit_name_params);
        let find = |k: &str| table.iter().find(|(n, _)| *n == k).map(|(_, o)| *o);

        let feats_off = find("unit_features")?;
        let mut features = Vec::new();
        for item in list_items(image, feats_off) {
            if u16le(image, item) == T_STRING {
                features.push(
                    cstr(image, (item as i64 + i32le(image, item + 4) as i64) as usize).to_string(),
                );
            }
        }

        let mut index = std::collections::HashMap::new();
        let mut slots = Vec::with_capacity(features.len() * 3);
        for (k, f) in features.iter().enumerate() {
            index.insert(f.clone(), slots.len());
            slots.push(Slot::Diff(k));
            index.insert(format!("targ.{f}"), slots.len());
            slots.push(Slot::Targ(k));
            index.insert(format!("cand.{f}"), slots.len());
            slots.push(Slot::Cand(k));
        }

        let cost_off = find("target_cost")?;
        let mut b = Builder { b: image, index, missing: Vec::new() };
        let expr = b.build(cost_off, 0);
        let missing = b.missing.clone();

        let beam_width = find("cand_beam_width")
            .and_then(|o| list_items(image, o).first().copied())
            .map(|i| match u16le(image, i) {
                T_INT => i32le(image, i + 4) as usize,
                T_FLOAT => f32le(image, i + 4) as usize,
                _ => 0,
            })
            .unwrap_or(0);

        Some(TargetCost { features, expr, slots, beam_width, missing })
    }

    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Check this build's target model against the voice, cheaply. See
    /// [`SelfCost`].
    pub fn self_cost(&self, v: &crate::db::Voice, probe: usize) -> SelfCost {
        let mut env = vec![0.0f32; self.num_slots()];
        let mut worst = 0i32;
        let n = v.num_units.max(1);
        let step = (n / probe.max(1)).max(1);
        let mut probed = 0usize;
        for u in (0..n).step_by(step) {
            let own: Vec<i32> = v.unit_feats(u).iter().map(|&x| x as i32).collect();
            worst = worst.max(self.score(&mut env, &own, v.unit_feats(u)));
            probed += 1;
        }
        SelfCost {
            probed,
            worst,
            unimplemented: self
                .features
                .iter()
                .filter(|f| !crate::target::is_implemented(f))
                .cloned()
                .collect(),
        }
    }

    /// Fill the environment from a target feature vector and a candidate's
    /// `voice_d.dat` row, then evaluate.
    pub fn score(&self, env: &mut [f32], targ: &[i32], cand: &[u8]) -> i32 {
        for (i, s) in self.slots.iter().enumerate() {
            env[i] = match *s {
                Slot::Targ(k) => targ.get(k).copied().unwrap_or(0) as f32,
                Slot::Cand(k) => cand.get(k).copied().unwrap_or(0) as f32,
                Slot::Diff(k) => {
                    let t = targ.get(k).copied().unwrap_or(0);
                    let c = cand.get(k).copied().unwrap_or(0) as i32;
                    (t - c).abs() as f32
                }
            };
        }
        eval(&self.expr, env) as i32 // cvttss2si
    }
}

pub fn eval(e: &Expr, env: &[f32]) -> f32 {
    match e {
        Expr::Num(v) => *v,
        Expr::Var(i) => env[*i],
        Expr::Missing(_) => 0.0,
        Expr::Clause(_, v) => eval(v, env),
        Expr::Call(op, args) => match op {
            Op::Mul => args.iter().fold(1.0, |a, x| a * eval(x, env)),
            Op::Add => args.iter().fold(0.0, |a, x| a + eval(x, env)),
            Op::Eq => b2f(args.len() == 2 && eval(&args[0], env) == eval(&args[1], env)),
            Op::Lt => b2f(args.len() == 2 && eval(&args[0], env) < eval(&args[1], env)),
            Op::Gt => b2f(args.len() == 2 && eval(&args[0], env) > eval(&args[1], env)),
            Op::Not => b2f(args.first().map(|a| eval(a, env)).unwrap_or(0.0) == 0.0),
            Op::And => b2f(args.iter().all(|a| eval(a, env) != 0.0)),
            Op::Or => b2f(args.iter().any(|a| eval(a, env) != 0.0)),
            Op::ZeroOne => b2f(args.first().map(|a| eval(a, env)).unwrap_or(0.0) != 0.0),
            Op::PhoneDist => {
                if args.len() != 2 {
                    return 0.0;
                }
                crate::phonedist::dist(eval(&args[0], env) as i32, eval(&args[1], env) as i32)
            }
            Op::Cond => {
                for c in args {
                    if let Expr::Clause(t, v) = c {
                        if eval(t, env) != 0.0 {
                            return eval(v, env);
                        }
                    }
                }
                0.0
            }
        },
    }
}

fn b2f(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}
