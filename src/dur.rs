//! The duration model Swift 4 and 5 voices score their target cost against.
//!
//! 6.2 does not use this: its 37 features hold no duration at all. Swift 4 and 5
//! carry two, and both come from one prediction:
//!
//! ```text
//! zdur       = dur_cart applied to the target segment
//! seconds    = zdur * stddev(phone) + mean(phone)          dur_stats
//!
//! lisp_durms           = (int)(seconds * 1000),  0 below 0, 255 above 0.254
//! lisp_zscoredur_norm  = (int)((zdur + 3) / 6 * 254),  0 below -3, 255 above 3
//! ```
//!
//! The two clamp functions are swift.dll 4.2 @0x14530 and @0x146f0 and the
//! constants are read out of the image, not fitted. Swift 5.2 @0x145a0 and
//! @0x14760 are the same code against the same constants -- 0.254, 0, 1000 and
//! -3, 3, 6, 254 -- so one implementation covers both. The chain checks out against
//! the engine's own calls: for `h` in "Hello there." Swift returns `lisp_durms`
//! 93, and the voice's own stats -- mean 0.0900, sd 0.0287 -- with the z that
//! `lisp_zscoredur_norm` quantised give 0.093389 s, which truncates to 93.
//!
//! What this cannot reproduce exactly is `accented` and `next_accent`. Cepstral
//! runs an intonation model to fill them and we have none, so both read 0 here.
//! They are 2 of the duration tree's 22 features; see `_out/v42cap` for the
//! captured values to check against.

use crate::cart::{Cart, DurStats, FeatVal};
use crate::db::Voice;
use crate::rules::Phone;
use crate::target::Pos;

pub struct DurModel<'a> {
    cart: Cart<'a>,
    stats: DurStats,
}

/// One target segment's predicted duration.
#[derive(Clone, Copy, Debug, Default)]
pub struct Predicted {
    /// The tree's own output, a z-score against this phone's stats.
    pub z: f32,
    pub seconds: f32,
    pub durms: i32,
    pub zdurnorm: i32,
}

/// `lisp_durms`, swift.dll 4.2 @0x14530.
pub fn durms(seconds: f32) -> i32 {
    if seconds > 0.254 {
        255
    } else if seconds < 0.0 {
        0
    } else {
        (seconds * 1000.0) as i32
    }
}

/// `lisp_zscoredur_norm`, swift.dll 4.2 @0x146f0.
pub fn zdurnorm(z: f32) -> i32 {
    if z < -3.0 {
        0
    } else if z > 3.0 {
        255
    } else {
        ((z + 3.0) / 6.0 * 254.0) as i32
    }
}

impl<'a> DurModel<'a> {
    pub fn open(v: &'a Voice) -> Option<DurModel<'a>> {
        Some(DurModel {
            cart: Cart::parse(v.image(), v.dur_cart()?),
            stats: DurStats::parse(v.image(), v.dur_stats()?, 256),
        })
    }

    pub fn predict(&self, phones: &[Phone], pos: &[Pos], syl_stress: &[i32]) -> Vec<Predicted> {
        (0..phones.len())
            .map(|i| {
                let f = |n: &str| feature(phones, pos, syl_stress, i, n);
                let z = self.cart.interpret(&f).as_num();
                let seconds = match self.stats.get(&phones[i].phone) {
                    Some(st) => crate::cart::segment_duration(z, st, 1.0),
                    None => 0.0,
                };
                Predicted { z, seconds, durms: durms(seconds), zdurnorm: zdurnorm(z) }
            })
            .collect()
    }
}

/// Everything outside the utterance reads as a pause, which is what the engine
/// sees too: its segment relation is bracketed by them.
fn at(phones: &[Phone], i: isize) -> &str {
    if i < 0 || i as usize >= phones.len() {
        "pau"
    } else {
        phones[i as usize].phone.as_str()
    }
}

fn onsetcoda(pos: &[Pos], i: usize) -> &'static str {
    if pos.get(i).is_some_and(|p| p.in_onset) {
        "onset"
    } else {
        "coda"
    }
}

/// The 22 names in a Swift 4 `dur_cart`. Anything else, and anything we have no
/// model for, reads 0 -- which is the tree's own default branch.
fn feature(phones: &[Phone], pos: &[Pos], syl_stress: &[i32], i: usize, name: &str) -> FeatVal {
    let s = |x: &str| FeatVal::Str(x.into());
    let n = |x: i32| FeatVal::Num(x as f32);
    let k = i as isize;
    match name {
        "name" => s(at(phones, k)),
        "p.name" => s(at(phones, k - 1)),
        "n.name" => s(at(phones, k + 1)),
        "pp.name" => s(at(phones, k - 2)),

        "ph_cvox" => s(crate::phoneset::cvox(at(phones, k))),
        "p.ph_cvox" => s(crate::phoneset::cvox(at(phones, k - 1))),
        "ph_ctype" => s(crate::phoneset::ctype(at(phones, k))),
        "n.ph_ctype" => s(crate::phoneset::ctype(at(phones, k + 1))),
        "pp.ph_ctype" => s(crate::phoneset::ctype(at(phones, k - 2))),
        "ph_vc" => s(crate::phoneset::vc(at(phones, k))),
        "pp.ph_vc" => s(crate::phoneset::vc(at(phones, k - 2))),
        "ph_vlng" => s(crate::phoneset::vlng(at(phones, k))),
        "pp.ph_vlng" => s(crate::phoneset::vlng(at(phones, k - 2))),

        "seg_onsetcoda" => s(onsetcoda(pos, i)),
        "n.seg_onsetcoda" => s(onsetcoda(pos, i + 1)),
        "seg_coda_nasal" => {
            // a nasal anywhere from this phone's syllable vowel to its end
            let p = &pos[i];
            let found = (0..phones.len()).any(|j| {
                pos[j].syl == p.syl
                    && !pos[j].in_onset
                    && crate::phoneset::ctype(&phones[j].phone) == "n"
            });
            n(found as i32)
        }

        "R:SylStructure.parent.stress" => n(syl_stress.get(pos[i].syl).copied().unwrap_or(0)),
        "R:SylStructure.parent.parent.word_numsyls" => n(pos[i].word_numsyls),
        "R:SylStructure.parent.syl_break" => n(pos[i].syl_break),
        "R:SylStructure.parent.R:Syllable.p.syl_break" => n(pos[i].prev_syl_break),
        "R:SylStructure.parent.position_type" => s(position_type(pos, i)),
        "p.R:SylStructure.parent.parent.pbreak" => s(pbreak(pos, i as isize - 1)),

        // no intonation model, so no accents to report
        "R:SylStructure.parent.next_accent"
        | "R:SylStructure.parent.R:Syllable.p.accented"
        | "accented" => n(0),

        _ => FeatVal::Num(0.0),
    }
}

fn position_type(pos: &[Pos], i: usize) -> &'static str {
    let p = &pos[i];
    let single = p.word_numsyls <= 1;
    let last = p.syl_in_word + 1 >= p.word_numsyls;
    match (single, p.syl_in_word == 0, last) {
        (true, _, _) => "single",
        (_, true, _) => "initial",
        (_, _, true) => "final",
        _ => "mid",
    }
}

/// `pbreak` on a word: `BB` at a phrase end, `B` at a lesser one, `NB` inside.
fn pbreak(pos: &[Pos], i: isize) -> &'static str {
    if i < 0 || i as usize >= pos.len() {
        return "BB";
    }
    match pos[i as usize].syl_break {
        4 => "BB",
        3 => "B",
        _ => "NB",
    }
}
