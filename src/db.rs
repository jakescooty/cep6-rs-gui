//! Voice database: the six Cepstral Swift files, and the accessors that
//! mirror `get_sts_frame` / `get_frame_size` / `get_sts_residual`.

use crate::val::{cstr, i32le, u16le, Params};
use std::io;
use std::path::Path;

pub const UNIT_NONE: i32 = 6_553_500;

#[derive(Debug, Clone)]
pub struct UnitType {
    pub name: String,
    pub start: i32,
    pub count: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct Unit {
    pub type_id: u16,
    pub tree_id: u16,
    pub start: i32,
    pub end: i32,
    pub prev: i32,
    pub next: i32,
}

pub struct Voice {
    image: Vec<u8>,
    idx: Vec<u8>,
    fdat: Vec<u8>,
    cdat: Vec<u8>,
    adat: Vec<u8>,
    ddat: Vec<u8>,

    pub num_sts: usize,
    pub order: usize,
    pub mcep_ch: usize,
    pub fold: i32,
    pub sps: u32,
    pub coeff_min: f32,
    pub coeff_range: f32,
    /// The LPC filter's coefficient scalings, precomputed as swift.dll does at
    /// @0x278a3: the filter is integer, and these are the only form of
    /// `coeff_min` and `coeff_range` it ever sees.
    ///
    /// ```text
    /// coeff_min_scaled   = (int)((double)coeff_min   *  32768.0)
    /// coeff_range_scaled = 1 - (int)((double)coeff_range * -1024.0)
    /// ```
    ///
    /// Both truncate toward zero, and the `1 -` on the second is not a rounding
    /// convenience -- it is what the DLL computes, and dropping it moves the
    /// coefficients by one and the waveform with them.
    pub coeff_min_scaled: i32,
    pub coeff_range_scaled: i32,
    pub mcep_min: f32,
    pub mcep_range: f32,

    pub types: Vec<UnitType>,
    pub num_units: usize,
    units_off: usize,
    pub unit_feat_width: usize,
    pub join_weights: Vec<f32>,
}

fn read(p: &Path) -> io::Result<Vec<u8>> {
    std::fs::read(p)
}

impl Voice {
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Voice> {
        Voice::open_named(dir, "voice")
    }

    pub fn open_named(dir: impl AsRef<Path>, base: &str) -> io::Result<Voice> {
        let d = dir.as_ref();
        let f = |suf: &str| d.join(format!("{base}{suf}"));
        let image = read(&f("_u.dat"))?;
        let idx = read(&f(".idx"))?;
        let fdat = read(&f("_f.dat"))?;
        let cdat = read(&f("_c.dat"))?;
        let adat = read(&f("_a.dat"))?;
        let ddat = read(&f("_d.dat")).unwrap_or_default();
        Voice::from_parts(image, idx, fdat, cdat, adat, ddat)
    }

    pub fn from_parts(
        image: Vec<u8>,
        idx: Vec<u8>,
        fdat: Vec<u8>,
        cdat: Vec<u8>,
        adat: Vec<u8>,
        ddat: Vec<u8>,
    ) -> io::Result<Voice> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        let p = Params::parse(&image, 0);

        let num_sts = p.int("sts_num_sts").ok_or_else(|| bad("sts_num_sts"))? as usize;
        let order = p.int("sts_num_channels").ok_or_else(|| bad("sts_num_channels"))? as usize;
        let mcep_ch = p.int("mcep_num_channels").unwrap_or(0) as usize;
        let fold = p.int("sts_residual_fold").unwrap_or(1);
        let sps = p.int("sts_sample_rate").ok_or_else(|| bad("sts_sample_rate"))? as u32;
        let coeff_min = p.float("sts_coeff_min").unwrap_or(0.0);
        let coeff_range = p.float("sts_coeff_range").unwrap_or(0.0);
        let coeff_min_scaled = (coeff_min as f64 * 32768.0) as i32;
        let coeff_range_scaled = 1 - (coeff_range as f64 * -1024.0) as i32;
        let mcep_min = p.float("mcep_coeff_min").unwrap_or(0.0);
        let mcep_range = p.float("mcep_coeff_range").unwrap_or(0.0);
        let num_types = p.int("num_types").ok_or_else(|| bad("num_types"))? as usize;
        let num_units = p.int("big_num_units").ok_or_else(|| bad("big_num_units"))? as usize;
        let types_off = p.blob("types_big").ok_or_else(|| bad("types_big"))?;
        let units_off = p.blob("units_big").ok_or_else(|| bad("units_big"))?;

        if idx.len() != (num_sts + 1) * 4 {
            return Err(bad("voice.idx size does not match sts_num_sts"));
        }
        if fdat.len() != num_sts * order * 2 {
            return Err(bad("voice_f.dat size does not match num_sts * channels"));
        }
        if !cdat.is_empty() && cdat.len() != num_sts * mcep_ch {
            return Err(bad("voice_c.dat size does not match num_sts * mcep channels"));
        }

        let mut types = Vec::with_capacity(num_types);
        for i in 0..num_types {
            let o = types_off + i * 12;
            let rel = i32le(&image, o) as i64;
            types.push(UnitType {
                name: cstr(&image, (o as i64 + rel) as usize).to_string(),
                start: i32le(&image, o + 4),
                count: i32le(&image, o + 8),
            });
        }

        let mut join_weights = Vec::new();
        if let Some(jo) = p.blob("join_weights") {
            for k in 0..mcep_ch {
                join_weights.push(i32le(&image, jo + k * 4) as f32 / 65536.0);
            }
        }

        let unit_feat_width = if ddat.is_empty() || num_units == 0 {
            0
        } else {
            ddat.len() / num_units
        };

        drop(p);
        Ok(Voice {
            image,
            idx,
            fdat,
            cdat,
            adat,
            ddat,
            num_sts,
            order,
            mcep_ch,
            fold,
            sps,
            coeff_min,
            coeff_range,
            coeff_min_scaled,
            coeff_range_scaled,
            mcep_min,
            mcep_range,
            types,
            num_units,
            units_off,
            unit_feat_width,
            join_weights,
        })
    }

    pub fn dur_cart(&self) -> Option<usize> {
        self.params().blob("dur_cart")
    }

    pub fn dur_stats(&self) -> Option<usize> {
        self.params().blob("dur_stats")
    }

    /// Offset of the nested unit_name_params table, if present.
    pub fn unit_name_params(&self) -> Option<usize> {
        self.params().blob("unit_name_params")
    }

    pub fn image(&self) -> &[u8] {
        &self.image
    }

    pub fn params(&self) -> Params<'_> {
        Params::parse(&self.image, 0)
    }

    pub fn name(&self) -> &str {
        self.params().string("name").unwrap_or("voice")
    }

    // -- sts accessors ----------------------------------------------------

    #[inline]
    pub fn res_offset(&self, i: usize) -> i32 {
        i32le(&self.idx, i * 4)
    }

    /// `get_frame_size`: stored bytes times the fold factor.
    #[inline]
    pub fn frame_size(&self, i: usize) -> usize {
        ((self.res_offset(i + 1) - self.res_offset(i)) * self.fold) as usize
    }

    /// Raw quantised LPC coefficients for frame `i`.
    #[inline]
    pub fn frame(&self, i: usize, out: &mut [u16]) {
        let base = i * self.order * 2;
        for k in 0..self.order {
            out[k] = u16le(&self.fdat, base + k * 2);
        }
    }

    /// Dequantised mel-cepstrum for frame `i`.
    pub fn mcep(&self, i: usize, out: &mut [f32]) {
        let base = i * self.mcep_ch;
        for k in 0..self.mcep_ch {
            out[k] = self.mcep_min + (self.cdat[base + k] as f32 / 255.0) * self.mcep_range;
        }
    }

    pub fn mcep_raw(&self, i: usize) -> &[u8] {
        &self.cdat[i * self.mcep_ch..(i + 1) * self.mcep_ch]
    }

    /// `get_sts_residual`: mu-law decode then expand by `fold` with a
    /// zero-order hold, exactly as swift.dll @0x28c40.
    pub fn residual(&self, i: usize, out: &mut Vec<i16>) {
        let a = self.res_offset(i) as usize;
        let b = self.res_offset(i + 1) as usize;
        let n = (b - a) * self.fold as usize;
        out.clear();
        out.reserve(n);
        if self.fold <= 1 {
            for &byte in &self.adat[a..b] {
                out.push(crate::ulaw::DECODE[byte as usize]);
            }
            return;
        }
        for &byte in &self.adat[a..b] {
            let v = crate::ulaw::DECODE[byte as usize];
            for _ in 0..self.fold {
                out.push(v);
            }
        }
    }

    // -- unit catalogue ---------------------------------------------------

    pub fn unit(&self, i: usize) -> Unit {
        let o = self.units_off + i * 20;
        let packed = i32le(&self.image, o) as u32;
        Unit {
            type_id: (packed & 0xFFFF) as u16,
            tree_id: (packed >> 16) as u16,
            start: i32le(&self.image, o + 4),
            end: i32le(&self.image, o + 8),
            prev: i32le(&self.image, o + 12),
            next: i32le(&self.image, o + 16),
        }
    }

    pub fn type_id(&self, name: &str) -> Option<usize> {
        self.types.iter().position(|t| t.name == name)
    }

    pub fn candidates(&self, type_id: usize) -> std::ops::Range<usize> {
        let t = &self.types[type_id];
        t.start as usize..(t.start + t.count) as usize
    }

    pub fn unit_feats(&self, i: usize) -> &[u8] {
        if self.unit_feat_width == 0 {
            return &[];
        }
        let w = self.unit_feat_width;
        &self.ddat[i * w..(i + 1) * w]
    }

    /// Follow the recorded successor links from `first`.
    pub fn chain(&self, first: usize, limit: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut u = first as i32;
        while u != UNIT_NONE && (u as usize) < self.num_units && out.len() < limit {
            out.push(u as usize);
            u = self.unit(u as usize).next;
        }
        out
    }

    pub fn total_bytes(&self) -> usize {
        self.image.len() + self.idx.len() + self.fdat.len() + self.cdat.len()
            + self.adat.len() + self.ddat.len()
    }
}
