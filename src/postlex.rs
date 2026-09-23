//! `ceplex_us_postlex`: the reductions Cepstral applies after lexicon lookup.
//!
//! ceplex_us.dll @0x1dac runs three passes. The first (@0x18dc) handles the
//! clitics `'s`, `'ve`, `'ll` and `'d`, which our tokenizer never produces as
//! separate words; the other two are here.
//!
//! Without them the front end says `dh i1` for "the" and `ey1` for "a" wherever
//! the engine says `dh ah0` and `ah0`, and since `lisp_phone_nameid` and the
//! syllable stress both change, the error spreads into six target-cost columns
//! and the two syllables on either side.

use crate::rules::{syl_boundary, Phone};

/// Words whose last syllable loses its stress, ceplex_us.dll RVA 0x368d0.
const DESTRESS: [&str; 12] = [
    "a", "an", "of", "from", "at", "for", "to", "and", "as", "but", "than", "that",
];

/// The subset that also reduces its vowel to `ah`, RVA 0x36940.
const REDUCE: [&str; 8] = ["a", "to", "at", "an", "from", "and", "as", "than"];

fn word_of(p: &Phone) -> Option<&str> {
    p.word.as_deref()
}

/// The `[start, end)` spans of each word, and of each pause, in phone indices.
fn word_spans(phones: &[Phone]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < phones.len() {
        let mut j = i + 1;
        if !phones[i].is_pause() {
            while j < phones.len()
                && !phones[j].is_pause()
                && word_of(&phones[j]) == word_of(&phones[i])
                && word_of(&phones[i]).is_some()
            {
                j += 1;
            }
        }
        out.push((i, j));
        i = j;
    }
    out
}

/// Apply the two reductions in place.
pub fn apply(phones: &mut Vec<Phone>) {
    reduce_the(phones);
    destress_function_words(phones);
}

/// ceplex_us.dll @0x1adc. The vowel of "the" is `i` in the lexicon; it becomes
/// `ah` with stress 0 when the next segment exists, is not a pause, and is not
/// itself a vowel. That is why "the outage" keeps `i1` -- `aw` is a vowel -- and
/// "the civil" does not.
fn reduce_the(phones: &mut Vec<Phone>) {
    for i in 0..phones.len() {
        if phones[i].phone != "i" {
            continue;
        }
        if word_of(&phones[i]).map(|w| w.to_ascii_lowercase()) != Some("the".into()) {
            continue;
        }
        let Some(next) = phones.get(i + 1) else { continue };
        if next.is_pause() || crate::phoneset::is_vowel(&next.phone) {
            continue;
        }
        phones[i].phone = "ah".to_string();
        phones[i].stress = 0;
    }
}

/// ceplex_us.dll @0x1c4c. A word in `DESTRESS` that is neither the first nor the
/// last word of its phrase (@0x1bf0 tests `item_next` and `item_prev` in the
/// Phrase relation) loses the stress on its last syllable; a word in `REDUCE`
/// also has its first vowel-followed-by-a-non-vowel renamed to `ah`.
///
/// The `n.ph_vc` test crosses the word boundary, so "a telephone" reduces and
/// "a emergency" would not.
fn destress_function_words(phones: &mut Vec<Phone>) {
    let spans = word_spans(phones);
    for (k, &(a, b)) in spans.iter().enumerate() {
        if phones[a].is_pause() {
            continue;
        }
        let Some(w) = word_of(&phones[a]).map(|w| w.to_ascii_lowercase()) else { continue };
        if !DESTRESS.contains(&w.as_str()) {
            continue;
        }
        // phrase-internal only: a pause on either side, or the end of the
        // utterance, disqualifies the word
        let prev_ok = k > 0 && !phones[spans[k - 1].0].is_pause();
        let next_ok = k + 1 < spans.len() && !phones[spans[k + 1].0].is_pause();
        if !prev_ok || !next_ok {
            continue;
        }

        if REDUCE.contains(&w.as_str()) {
            for i in a..b {
                if !crate::phoneset::is_vowel(&phones[i].phone) {
                    continue;
                }
                let follower_is_vowel = phones
                    .get(i + 1)
                    .map(|p| crate::phoneset::is_vowel(&p.phone))
                    .unwrap_or(false);
                if !follower_is_vowel {
                    phones[i].phone = "ah".to_string();
                    break;
                }
            }
        }

        // R:SylStructure.daughtern.stress = 0
        let names: Vec<&str> = phones[a..b].iter().map(|p| p.phone.as_str()).collect();
        let mut last_start = 0usize;
        for i in 0..names.len() {
            if i + 1 < names.len() && syl_boundary(&names, i, last_start) {
                last_start = i + 1;
            }
        }
        for i in a + last_start..b {
            phones[i].stress = 0;
        }
    }
}
