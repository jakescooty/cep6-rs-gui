//! Flite CARTs stored in `voice_u.dat`, and the duration model built on one.
//!
//! Blob header, three int32, **relative to the start of the blob** (not to
//! their own field -- getting that wrong shifts the feature table by one entry
//! and silently mispairs every test in the tree):
//!
//! ```text
//! +0  rule_table   the node array
//! +4  feat_table   int32[] of self-relative offsets to feature-name strings
//! +8  unused here
//! ```
//!
//! Node, 8 bytes: `u8 feat; u8 op; u16 no_node; i32 val_self_rel`, where the
//! val offset is relative to the val field. The yes-branch is the next node;
//! the no-branch is `no_node`. `op` 255 marks a leaf.
//!
//! Duration follows flite `cart_duration` (`cst_synth.c:396`):
//! `dur = stretch * (zdur * stddev + mean)`, accumulated into each segment's
//! `end`, which is what `concat_units` consumes as `target_end`.

use crate::val::{cstr, f32le, i32le, u16le};

pub const OP_IS: u8 = 0;
pub const OP_IN: u8 = 1;
pub const OP_LESS: u8 = 2;
pub const OP_GREATER: u8 = 3;
pub const OP_MATCHES: u8 = 4;
pub const OP_EQUALS: u8 = 5;
pub const OP_LEAF: u8 = 255;

/// A feature value as the tree sees it.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatVal {
    Str(String),
    Num(f32),
    None,
}

impl FeatVal {
    pub fn as_num(&self) -> f32 {
        match self {
            FeatVal::Num(v) => *v,
            FeatVal::Str(s) => s.parse().unwrap_or(0.0),
            FeatVal::None => 0.0,
        }
    }

    fn eq_val(&self, other: &FeatVal) -> bool {
        match (self, other) {
            (FeatVal::Str(a), FeatVal::Str(b)) => a == b,
            (FeatVal::None, FeatVal::None) => true,
            (FeatVal::None, _) | (_, FeatVal::None) => false,
            _ => self.as_num() == other.as_num(),
        }
    }
}

/// Anything that can answer feature questions about the item being classified.
pub trait Features {
    fn get(&self, name: &str) -> FeatVal;
}

impl<F: Fn(&str) -> FeatVal> Features for F {
    fn get(&self, name: &str) -> FeatVal {
        self(name)
    }
}

pub struct Cart<'a> {
    image: &'a [u8],
    rule_table: usize,
    pub feats: Vec<String>,
    pub num_nodes: usize,
}

impl<'a> Cart<'a> {
    pub fn parse(image: &'a [u8], blob: usize) -> Cart<'a> {
        let rule_table = (blob as i64 + i32le(image, blob) as i64) as usize;
        let feat_table = (blob as i64 + i32le(image, blob + 4) as i64) as usize;

        let mut feats = Vec::new();
        let mut o = feat_table;
        while o + 4 <= rule_table {
            let t = (o as i64 + i32le(image, o) as i64) as usize;
            if t == 0 || t >= image.len() {
                break;
            }
            feats.push(cstr(image, t).to_string());
            o += 4;
        }

        let mut num_nodes = 0usize;
        let mut o = rule_table;
        while o + 8 <= image.len() && num_nodes < 1_000_000 {
            if image[o] == 0 && image[o + 1] == 0 && i32le(image, o + 4) == 0 {
                break;
            }
            num_nodes += 1;
            o += 8;
        }

        Cart { image, rule_table, feats, num_nodes }
    }

    fn node(&self, i: usize) -> (u8, u8, usize, usize) {
        let o = self.rule_table + i * 8;
        let val = (o as i64 + 4 + i32le(self.image, o + 4) as i64) as usize;
        (self.image[o], self.image[o + 1], u16le(self.image, o + 2) as usize, val)
    }

    fn val(&self, off: usize) -> FeatVal {
        match u16le(self.image, off) {
            0x01 => FeatVal::Num(i32le(self.image, off + 4) as f32),
            0x03 => FeatVal::Num(f32le(self.image, off + 4)),
            0x33 => FeatVal::Str(
                cstr(self.image, (off as i64 + i32le(self.image, off + 4) as i64) as usize)
                    .to_string(),
            ),
            _ => FeatVal::None,
        }
    }

    /// Walk the tree and return the leaf value.
    pub fn interpret(&self, f: &dyn Features) -> FeatVal {
        let mut n = 0usize;
        let mut guard = 0;
        loop {
            let (feat, op, no, valoff) = self.node(n);
            if op == OP_LEAF {
                return self.val(valoff);
            }
            guard += 1;
            if guard > self.num_nodes + 8 {
                return FeatVal::None; // malformed tree; do not spin
            }
            let name = self.feats.get(feat as usize).map(|s| s.as_str()).unwrap_or("");
            let got = f.get(name);
            let want = self.val(valoff);
            let yes = match op {
                OP_IS | OP_EQUALS => got.eq_val(&want),
                OP_LESS => got.as_num() < want.as_num(),
                OP_GREATER => got.as_num() > want.as_num(),
                OP_IN | OP_MATCHES => got.eq_val(&want),
                _ => false,
            };
            n = if yes { n + 1 } else { no };
            if n >= self.num_nodes {
                return FeatVal::None;
            }
        }
    }
}

/// Per-phone duration statistics, in seconds.
#[derive(Debug, Clone)]
pub struct DurStat {
    pub phone: String,
    pub mean: f32,
    pub stddev: f32,
}

pub struct DurStats {
    pub stats: Vec<DurStat>,
}

impl DurStats {
    /// The table is an array of self-relative offsets to 12-byte records
    /// `{i32 name_self_rel; f32 mean; f32 stddev}`.
    pub fn parse(image: &[u8], blob: usize, limit: usize) -> DurStats {
        let mut stats = Vec::new();
        let mut o = blob;
        for _ in 0..limit {
            if o + 4 > image.len() {
                break;
            }
            let t = (o as i64 + i32le(image, o) as i64) as usize;
            if t == 0 || t + 12 > image.len() {
                break;
            }
            let nm = (t as i64 + i32le(image, t) as i64) as usize;
            if nm >= image.len() {
                break;
            }
            let phone = cstr(image, nm).to_string();
            let mean = f32le(image, t + 4);
            let stddev = f32le(image, t + 8);
            if phone.is_empty() || !phone.bytes().all(|c| c.is_ascii_graphic())
                || !mean.is_finite() || !stddev.is_finite()
                || !(0.0..2.0).contains(&mean) || !(0.0..2.0).contains(&stddev)
            {
                break;
            }
            stats.push(DurStat { phone, mean, stddev });
            o += 4;
        }
        DurStats { stats }
    }

    pub fn get(&self, phone: &str) -> Option<&DurStat> {
        self.stats.iter().find(|s| s.phone == phone)
    }
}

/// flite `cart_duration`: seconds for one segment.
pub fn segment_duration(zdur: f32, stat: &DurStat, stretch: f32) -> f32 {
    stretch * (zdur * stat.stddev + stat.mean)
}
