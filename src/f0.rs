//! F0 targets.
//!
//! All four of these voices are built with `PROSODY "none"` in settings.txt and
//! carry no F0 model: `voice_u.dat` has no `f0_trees` or `num_f0_types` keys
//! (those belong to the SPS voice format, read only by
//! `swift_voice_undump_sps`). What runs instead is `flat_prosody`
//! (swift.dll @0x14830), which is a two-point linear declination and nothing
//! more:
//!
//! ```text
//! mean   = param("int_f0_target_mean",   100.0)
//! shift  = param("f0_shift",               1.0)
//! stddev = param("int_f0_target_stddev",  12.0)
//! m = mean * shift
//! Target[0] = { pos: 0.0,              f0: m + stddev }
//! Target[1] = { pos: last segment end, f0: m - stddev }
//! ```
//!
//! Nothing in `swift.dll` ever *sets* `int_f0_target_mean`; it is only read,
//! with that default. Measuring the SDK's own 31.5 s William sample gives a
//! median of 97 Hz and an implied mean of ~100 Hz, so for that voice the engine
//! is indeed running on the built-in default.
//!
//! That default is speaker-independent, which matters: Allison was recorded
//! near 250 Hz, so driving her at 100 Hz would stretch every pitch period by
//! about 2.5x, and `concat_units` pads with silence rather than resampling.
//! `natural_f0` estimates a voice's own F0 from its recorded pitch periods so a
//! caller can set `int_f0_target_mean` sensibly per voice.

use crate::db::Voice;
use crate::prosody::F0Target;

pub const DEFAULT_F0_MEAN: f32 = 100.0;
pub const DEFAULT_F0_STDDEV: f32 = 12.0;
pub const DEFAULT_F0_SHIFT: f32 = 1.0;

#[derive(Debug, Clone, Copy)]
pub struct FlatProsody {
    pub mean: f32,
    pub stddev: f32,
    pub shift: f32,
}

impl Default for FlatProsody {
    fn default() -> Self {
        FlatProsody {
            mean: DEFAULT_F0_MEAN,
            stddev: DEFAULT_F0_STDDEV,
            shift: DEFAULT_F0_SHIFT,
        }
    }
}

impl FlatProsody {
    /// Centre the declination on the voice's own recorded F0 instead of the
    /// engine's speaker-independent 100 Hz.
    pub fn for_voice(v: &Voice) -> FlatProsody {
        FlatProsody { mean: natural_f0(v, 20_000), ..FlatProsody::default() }
    }

    /// swift.dll @0x14830. `utterance_end` is the last segment's `end`, which
    /// is what the duration model accumulates.
    pub fn targets(&self, utterance_end: f32) -> Vec<F0Target> {
        let m = self.mean * self.shift;
        vec![
            F0Target { pos: 0.0, f0: m + self.stddev },
            F0Target { pos: utterance_end, f0: m - self.stddev },
        ]
    }
}

/// Median recorded F0, from the pitch periods in the database.
///
/// Every sts frame is one pitch period, so `sample_rate / frame_size` is the
/// F0 that period was recorded at. Unvoiced frames land outside a plausible
/// speech range and are dropped.
pub fn natural_f0(v: &Voice, sample_count: usize) -> f32 {
    let n = v.num_sts;
    if n == 0 {
        return DEFAULT_F0_MEAN;
    }
    let step = (n / sample_count.max(1)).max(1);
    let mut hz: Vec<f32> = Vec::new();
    let mut i = 0;
    while i < n {
        let sz = v.frame_size(i);
        if sz > 0 {
            let f = v.sps as f32 / sz as f32;
            if (50.0..500.0).contains(&f) {
                hz.push(f);
            }
        }
        i += step;
    }
    if hz.is_empty() {
        return DEFAULT_F0_MEAN;
    }
    hz.sort_by(|a, b| a.partial_cmp(b).unwrap());
    hz[hz.len() / 2]
}
