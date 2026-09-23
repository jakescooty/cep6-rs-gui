//! A plain-text "score": the input the front end will eventually produce.
//!
//! ```text
//! # comments with #
//! lead_f0 120            # optional, defaults to 120
//! f0   0.00  180         # F0 contour point: time (s), Hz
//! f0   0.35  150
//! unit paustart  0.10        # type name -> first candidate, end time (s)
//! unit #12345    0.28        # or a literal unit index
//! unit heh       0.28  0.8   # optional local_rescale
//! ```
//!
//! Units are emitted in order; `end` is the segment end time, which becomes
//! `target_end` in samples.

use crate::db::Voice;
use crate::prosody::{gain_from_rescale, F0Target, TargetUnit, DEFAULT_LEAD_F0};
use crate::synth::GAIN_UNITY;

pub struct Score {
    pub units: Vec<TargetUnit>,
    pub names: Vec<String>,
    pub targets: Vec<F0Target>,
    pub lead_f0: f32,
    /// Type ids in order, for re-running selection over the same score.
    pub types: Vec<usize>,
    /// Segment end times in seconds.
    pub ends: Vec<f32>,
    /// True when every unit was named by type rather than pinned with `#index`.
    pub selectable: bool,
}

pub fn parse(v: &Voice, text: &str) -> Result<Score, String> {
    let mut s = Score {
        units: Vec::new(),
        names: Vec::new(),
        targets: Vec::new(),
        lead_f0: DEFAULT_LEAD_F0,
        types: Vec::new(),
        ends: Vec::new(),
        selectable: true,
    };

    for (lno, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let err = |m: &str| format!("line {}: {m}: {raw}", lno + 1);

        match f[0] {
            "lead_f0" if f.len() == 2 => {
                s.lead_f0 = f[1].parse().map_err(|_| err("bad f0"))?;
            }
            "f0" if f.len() == 3 => {
                s.targets.push(F0Target {
                    pos: f[1].parse().map_err(|_| err("bad time"))?,
                    f0: f[2].parse().map_err(|_| err("bad f0"))?,
                });
            }
            "unit" if f.len() >= 3 => {
                let idx = if let Some(n) = f[1].strip_prefix('#') {
                    s.selectable = false;
                    n.parse::<usize>().map_err(|_| err("bad unit index"))?
                } else {
                    let t = v.type_id(f[1]).ok_or_else(|| err("unknown unit type"))?;
                    v.candidates(t).start
                };
                if idx >= v.num_units {
                    return Err(err("unit index out of range"));
                }
                let end: f32 = f[2].parse().map_err(|_| err("bad end time"))?;
                let gain = if f.len() > 3 {
                    gain_from_rescale(f[3].parse().map_err(|_| err("bad rescale"))?)
                } else {
                    GAIN_UNITY
                };
                let u = v.unit(idx);
                s.types.push(u.type_id as usize);
                s.ends.push(end);
                s.units.push(TargetUnit {
                    start: u.start,
                    end: u.end,
                    target_end: (end * v.sps as f32) as i32,
                    gain,
                });
                s.names.push(format!("{}#{}", v.types[u.type_id as usize].name, idx));
            }
            _ => return Err(err("unrecognised directive")),
        }
    }

    if s.units.is_empty() {
        return Err("score contains no units".into());
    }
    Ok(s)
}

/// Build a score from bare type names at a fixed rate, for quick tests:
/// every unit gets `secs_per_unit`, and F0 runs flat at `f0`.
pub fn flat(v: &Voice, names: &[&str], secs_per_unit: f32, f0: f32) -> Result<Score, String> {
    let mut lines = String::new();
    let total = secs_per_unit * names.len() as f32;
    lines.push_str(&format!("f0 0.0 {f0}\nf0 {total} {f0}\n"));
    for (i, n) in names.iter().enumerate() {
        lines.push_str(&format!("unit {n} {}\n", secs_per_unit * (i + 1) as f32));
    }
    parse(v, &lines)
}
