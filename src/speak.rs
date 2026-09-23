//! Text to samples, the way the engine runs it.
//!
//! The one structural thing this adds over calling the pieces by hand: the
//! engine works **one utterance at a time**. Each sentence gets its own
//! `cst_utterance`, its own clunits search and its own `lpc_state`, so the
//! search beam does not see across a sentence boundary and the filter history
//! starts from zero at each one. Running a single search over a whole paragraph
//! picks the same units on short input and drifts on long input, and it puts the
//! streamed piece boundaries in the wrong places either way.

use crate::ceplex::CepLex;
use crate::db::Voice;
use crate::expr::TargetCost;
use crate::norm::DigitStyle;
use crate::prosody::{join_units_streamed, TargetUnit};
use crate::rules::{NameRules, Phone};
use crate::select::{SelectParams, Selector};
use crate::synth::GAIN_UNITY;
use crate::target::PhoneTable;

/// Split a phone sequence at its sentence boundaries.
///
/// A boundary is two pauses in a row -- `pauend` closing one utterance and
/// `paustart` opening the next -- so the cut goes between them and each piece
/// keeps a pause at both ends, which is what the engine's Segment relations
/// look like.
pub fn utterances(phones: &[Phone]) -> Vec<Vec<Phone>> {
    let mut out: Vec<Vec<Phone>> = Vec::new();
    let mut cur: Vec<Phone> = Vec::new();
    for (i, p) in phones.iter().enumerate() {
        cur.push(p.clone());
        let boundary = p.is_pause()
            && phones.get(i + 1).is_some_and(|q| q.is_pause())
            && cur.len() > 1;
        if boundary {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// What one utterance produced, for callers that want more than the samples.
pub struct Said {
    pub phones: Vec<Phone>,
    pub names: Vec<String>,
    pub units: Vec<usize>,
    /// Units per streamed piece; see `Selector::last_pieces`.
    pub pieces: Vec<usize>,
    pub pcm: Vec<i16>,
}

/// Synthesise `text`, utterance by utterance.
pub fn say(v: &Voice, cep: &CepLex, text: &str, style: DigitStyle) -> Vec<Said> {
    say_tagged(v, cep, None, text, style)
}

/// The same with the part-of-speech tagger in front of the lexicon.
///
/// Only utterances holding one of its 237 listed homographs are tagged at all,
/// so everything else comes out exactly as `say` produces it.
pub fn say_tagged(
    v: &Voice,
    cep: &CepLex,
    pos: Option<&crate::postag::PosTag>,
    text: &str,
    style: DigitStyle,
) -> Vec<Said> {
    let (seq, _, _) = crate::lex::text_to_phones_tagged(cep, pos, text, style);
    say_phones(v, &seq)
}

/// The same from a phone sequence that has already been through the front end.
pub fn say_phones(v: &Voice, seq: &[Phone]) -> Vec<Said> {
    let mut out = Vec::new();
    say_phones_each(v, seq, |said| out.push(said));
    out
}

/// The same, handing each utterance over as it is finished.
///
/// Whole-book input is the reason this exists: `say_phones` holds every
/// utterance's audio at once, which for a long file is gigabytes of PCM, while a
/// caller writing each one straight to disk never holds more than one.
pub fn say_phones_each<F: FnMut(Said)>(v: &Voice, seq: &[Phone], mut each: F) {
    let unp = match v.unit_name_params() {
        Some(u) => u,
        None => return,
    };
    let nr = NameRules::parse(v.image(), unp);
    let Some(tc) = TargetCost::parse(v.image(), unp) else { return };
    let table = PhoneTable::learn(v, &tc);

    utterances(seq)
        .into_iter()
        .for_each(|phones| {
            let names = nr.name_utterance(&phones, |n| v.type_id(n).is_some());
            let types: Vec<usize> = names.iter().filter_map(|n| v.type_id(n)).collect();
            let tf = crate::target::build(v, &table, &nr, &tc, &phones);
            let mut sel = Selector::new(v, SelectParams::from_voice(v));
            let chosen = sel.select_with_targets(&types, Some(&tf));
            let units: Vec<TargetUnit> = chosen.iter().map(|c| TargetUnit {
                start: c.start, end: c.end, target_end: 0, gain: GAIN_UNITY,
            }).collect();
            let pcm = join_units_streamed(v, &units, &sel.last_pieces);
            each(Said {
                phones,
                names,
                units: chosen.iter().map(|c| c.unit).collect(),
                pieces: sel.last_pieces.clone(),
                pcm,
            });
        });
}

/// Just the samples.
pub fn say_pcm(v: &Voice, cep: &CepLex, text: &str, style: DigitStyle) -> Vec<i16> {
    say(v, cep, text, style).into_iter().flat_map(|s| s.pcm).collect()
}
