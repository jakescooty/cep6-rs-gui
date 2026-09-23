//! Flite `feat_undump` image reader: the container used by `voice_u.dat`.
//!
//! Key table entries are `<name>\0`, padded to 4 bytes, then an int32 that is
//! relative to its own position. Values are 32-byte `cst_val`s.

pub const T_INT: u16 = 0x01;
pub const T_FLOAT: u16 = 0x03;
pub const T_STRING: u16 = 0x33;
pub const T_BLOB: u16 = 0x35;
pub const T_CONS: u16 = 0x37;

#[derive(Debug, Clone, PartialEq)]
pub enum Val<'a> {
    Int(i32),
    Float(f32),
    Str(&'a str),
    /// Absolute offset of the blob's payload within the image.
    Blob(usize),
    Other(u16, usize),
}

pub fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

pub fn i32le(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

pub fn f32le(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

pub fn cstr(b: &[u8], o: usize) -> &str {
    let end = b[o..].iter().position(|&c| c == 0).map_or(b.len(), |n| o + n);
    core::str::from_utf8(&b[o..end]).unwrap_or("")
}

/// Resolve the value at `off`.
pub fn val(b: &[u8], off: usize) -> Val<'_> {
    let t = u16le(b, off);
    let rel = i32le(b, off + 4);
    match t {
        T_INT => Val::Int(rel),
        T_FLOAT => Val::Float(f32le(b, off + 4)),
        T_STRING => Val::Str(cstr(b, (off as i64 + rel as i64) as usize)),
        T_BLOB => Val::Blob((off as i64 + rel as i64) as usize),
        _ => Val::Other(t, (off as i64 + rel as i64) as usize),
    }
}

/// Walk a feat table starting at `start`, yielding `(name, value_offset)`.
pub fn walk(b: &[u8], start: usize) -> Vec<(&str, usize)> {
    let mut out = Vec::new();
    let mut o = start;
    while o < b.len() && b[o] != 0 {
        let name = cstr(b, o);
        let p = (o + name.len() + 1 + 3) & !3;
        if p + 4 > b.len() {
            break;
        }
        out.push((name, (p as i64 + i32le(b, p) as i64) as usize));
        o = p + 4;
    }
    out
}

pub struct Params<'a> {
    pub entries: Vec<(&'a str, Val<'a>)>,
}

impl<'a> Params<'a> {
    pub fn parse(b: &'a [u8], start: usize) -> Self {
        Params {
            entries: walk(b, start).into_iter().map(|(n, o)| (n, val(b, o))).collect(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Val<'a>> {
        self.entries.iter().find(|(n, _)| *n == key).map(|(_, v)| v)
    }

    pub fn int(&self, key: &str) -> Option<i32> {
        match self.get(key) {
            Some(Val::Int(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn float(&self, key: &str) -> Option<f32> {
        match self.get(key) {
            Some(Val::Float(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn blob(&self, key: &str) -> Option<usize> {
        match self.get(key) {
            Some(Val::Blob(o)) => Some(*o),
            _ => None,
        }
    }

    pub fn string(&self, key: &str) -> Option<&'a str> {
        match self.get(key) {
            Some(Val::Str(s)) => Some(s),
            _ => None,
        }
    }
}
