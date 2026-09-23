//! Building the target feature vectors that drive Cepstral's target cost.
//!
//! `unit_features` names 39 columns of `voice_d.dat`, and `target_cost` scores a
//! candidate by comparing its recorded columns against the target's. Selection is
//! therefore the only place prosody enters these voices: `PROSODY "none"` means no
//! pitch is ever modified, so a falling phrase end exists only if the search picks
//! units that were *recorded* falling.
//!
//! The database says exactly that. Grouping Allison's units by `lisp_finalityp`:
//!
//! ```text
//! value   units   next_pau   median vowel F0
//!   0     90862      0%          204.2 Hz
//!   1      6648      0%          177.8 Hz
//!   2      7101    100%          151.0 Hz
//! ```
//!
//! and `lisp_finalityp` carries the heaviest weight in the whole expression --
//! 4589 for vowels, 4759 for consonants, roughly ten times any other term. Set it
//! correctly and the terminal fall follows; leave it at zero, as an unweighted
//! selector does, and the search is free to end an utterance on a 250 Hz unit.
//!
//! Most of the remaining columns are intrinsic to the phone rather than to the
//! utterance, so this module *learns* them from the voice instead of hardcoding
//! English phonology -- which also keeps it honest on Jean-Pierre, whose inventory
//! is built differently. A column is taken as intrinsic only when one value covers
//! `PURITY` of the units sharing a phone; anything more context-dependent than
//! that stays at zero, which is its modal value in every case here.

use crate::db::Voice;
use crate::expr::TargetCost;
use crate::rules::{self, NameRules, Phone};
use std::collections::BTreeMap;

/// A column counts as phone-intrinsic when its modal value covers this much of
/// the units sharing a phone.
const PURITY: f64 = 0.80;

/// A phone pair needs at least this many examples before its modal value is
/// trusted; rare pairs otherwise memorise one recording's accidents.
const MIN_PAIR_SAMPLES: usize = 8;

/// `lisp_finalityp` / `lisp_phone_break` for the phone immediately before a pause.
const FINAL_PRE_PAUSE: i32 = 2;
const BREAK_PRE_PAUSE: i32 = 6;

/// `lisp_finalityp` for the phrase's coda: every phone from the last vowel on,
/// except the last one, which takes `FINAL_PRE_PAUSE`.
///
/// Read off the engine's own `ffeature_int` calls, captured by `_out/run_cost.py`.
/// "…the outage." ends `aw1 t ih0 jh` and scores `0 0 1 2`; "…normally." ends
/// `l i0` and scores `0 2`; "…emergency." ends `n s i0` and scores `0 0 2`. So
/// the level is not "one phone before the end" -- it starts at the final vowel,
/// which is why finality-1 units slightly outnumber finality-2 units (6% against
/// 5% in Allison) even though only one phone per phrase can take the 2.
const FINAL_APPROACHING: i32 = 1;

/// `lisp_phone_break` on a pause itself. The engine reports `p.lisp_phone_break`
/// 7 for the first phone of every phrase, in both captured sentences and after
/// the sentence-internal pause as well as at the start.
const BREAK_PAUSE: i32 = 7;
/// `lisp_phone_break` at a word boundary, and at a syllable boundary inside a word.
///
/// Pinned against the type names, which carry ground truth: a name ending in a
/// function word (`dhthe`, `ah0the`, `vof`) is a phone *inside* that word, so the
/// word's phone count says how many of its units are word-final. Over Allison:
///
/// ```text
/// break   units   word-final   next_pau   finality>0
///   0     63589      6.3%         0%         10%
///   2     15929      0.0%         0%          0%
///   3     17998     19.0%         0%          0%
///   6      7092      0.7%       100%        100%
///   7         3      0.0%       100%        100%
/// ```
///
/// Value 2 never once falls on a word-final unit across 15,929 of them, so it can
/// only mark a word-internal boundary. Value 3 is enriched threefold in both
/// directions -- 19.0% of break-3 units are word-final against 6.3% of break-0
/// units, and 45.9% of word-final units take break 3 against 15.0% of the rest.
///
/// Only 47% of function-word units take break 3, which at first looks like noise
/// but is not. A function word's units cover *all* its phones, and only the last
/// one is word-final, so the expected rate is 1/n for an n-phone word. Against the
/// lexicon:
///
/// ```text
/// word   phones      n   predicted   observed
/// and    ae n d      3      33%         33%
/// for    f ao r      3      33%         33%
/// the    dh ah       2      50%         54%
/// of     ah v        2      50%         49%
/// to     t uw        2      50%         48%
/// an     ae n        2      50%         50%
/// are    aa r        2      50%         46%
/// ```
///
/// Both three-phone words land exactly on 33% and the two-phone words cluster on
/// 50%. That is a much tighter confirmation that break 3 marks a word boundary
/// than the aggregate enrichment alone, and it rules out the reading that the
/// split reflects phrasing variation across takes.
const BREAK_WORD: i32 = 3;
const BREAK_SYL: i32 = 2;

/// Which utterance-structure column a feature name refers to, if any. Everything
/// else is learned from the database.
#[derive(Clone, Copy, PartialEq)]
enum Slot {
    NameId,
    NextNameId,
    PrevNameId,
    Stress,
    PrevStress,
    NextStress,
    Uptalk,
    Finality,
    SylNumPhones,
    WordNumSyls,
    Break,
    PrevBreak,
    PrevPau,
    NextPau,
    VowNextVoiced,
    PrevRnd,
    PrevHighV,
    ApPrevApvow,
    ApNextApvow,
    AlvNnNasap,
    // Swift 4 and 5 only. 6.2 carries none of these.
    NextNextNameId,
    PrevPrevNameId,
    NextBreak,
    PrevPrevBreak,
    OnsetP,
    SylOnsetSize,
    SylCodaSize,
    SylBreak,
    PrevSylBreak,
    DurMs,
    ZDurNorm,
}

/// Whether this build computes `name` directly. A `false` is not automatically
/// wrong: `PhoneTable::learn` recovers a categorical feature from the unit
/// database. It is wrong when the feature is continuous, which is what makes a
/// Swift 5 voice's `lisp_durms` unrecoverable -- see [`crate::expr::SelfCost`].
pub fn is_implemented(name: &str) -> bool {
    slot_of(name).is_some()
}

fn slot_of(name: &str) -> Option<Slot> {
    Some(match name {
        "lisp_phone_nameid" => Slot::NameId,
        "n.lisp_phone_nameid" => Slot::NextNameId,
        "p.lisp_phone_nameid" => Slot::PrevNameId,
        "R:SylStructure.parent.stress" => Slot::Stress,
        "R:SylStructure.parent.R:Syllable.p.stress" => Slot::PrevStress,
        "R:SylStructure.parent.R:Syllable.n.stress" => Slot::NextStress,
        "lisp_uptalk" => Slot::Uptalk,
        "lisp_finalityp" => Slot::Finality,
        "R:SylStructure.parent.syl_numphones" => Slot::SylNumPhones,
        "R:SylStructure.parent.parent.word_numsyls" => Slot::WordNumSyls,
        "lisp_phone_break" => Slot::Break,
        "p.lisp_phone_break" => Slot::PrevBreak,
        "lisp_prev_pau" => Slot::PrevPau,
        "lisp_next_pau" => Slot::NextPau,
        "lisp_vow_next_voiced" => Slot::VowNextVoiced,
        "lisp_prev_rnd" => Slot::PrevRnd,
        "lisp_prev_high_v" => Slot::PrevHighV,
        "lisp_ap_prev_apvow" => Slot::ApPrevApvow,
        "lisp_ap_next_apvow" => Slot::ApNextApvow,
        "lisp_alv_nn_nasap" => Slot::AlvNnNasap,
        "n.n.lisp_phone_nameid" => Slot::NextNextNameId,
        "p.p.lisp_phone_nameid" => Slot::PrevPrevNameId,
        "n.lisp_phone_break" => Slot::NextBreak,
        "p.p.lisp_phone_break" => Slot::PrevPrevBreak,
        "lisp_onsetp" => Slot::OnsetP,
        "R:SylStructure.parent.syl_onsetsize" => Slot::SylOnsetSize,
        "R:SylStructure.parent.syl_codasize" => Slot::SylCodaSize,
        "R:SylStructure.parent.syl_break" => Slot::SylBreak,
        "R:SylStructure.parent.R:Syllable.p.syl_break" => Slot::PrevSylBreak,
        "lisp_durms" => Slot::DurMs,
        "lisp_zscoredur_norm" => Slot::ZDurNorm,
        _ => return None,
    })
}

/// `lisp_ctype_appr`, swift.dll @0x10640: the phoneset's `ctype` is `ap`.
fn is_approximant(phone: &str) -> bool {
    crate::phoneset::ctype(phone) == "ap"
}

/// `lisp_ap_prev_apvow` (@0x11740) and `lisp_ap_next_apvow` (@0x11660).
///
/// An approximant reports the phone id of its neighbour when that neighbour is
/// itself an approximant or a vowel -- but only when the `lisp_phone_break`
/// between them is 0, so nothing survives a syllable boundary. That last test is
/// what keeps the `r` of "authority" at 0: its `ao1` is syllable-final, break 2.
fn ap_apvow(
    phones: &[Phone],
    table: &PhoneTable,
    i: usize,
    j: Option<usize>,
    gate_break: i32,
) -> i32 {
    if !is_approximant(&phones[i].phone) || gate_break != 0 {
        return 0;
    }
    let Some(j) = j else { return 0 };
    let Some(p) = phones.get(j) else { return 0 };
    if !is_approximant(&p.phone) && !crate::phoneset::is_vowel(&p.phone) {
        return 0;
    }
    table.id(&p.phone).unwrap_or(0) as i32
}

/// `lisp_prev_rnd`, swift.dll @0x11410: a literal `cst_streq` of `p.name`
/// against three phones and nothing else.
const PREV_RND: [&str; 3] = ["aw", "ow", "uw"];

/// `lisp_prev_high_v`, swift.dll @0x11490, the same shape.
const PREV_HIGH_V: [&str; 4] = ["i", "ay", "oy", "ey"];

/// `lisp_vow_next_voiced`, swift.dll @0x10d20.
///
/// A stressed vowel scores 1 when the voicing runs unbroken to the next vowel or
/// to a word or phrase break. The function is written out longhand for six
/// phones of lookahead: at each step it returns 1 if the phone is a vowel, 1 if
/// `lisp_phone_break` on the phone before it is 3 or 6, and 0 if `cvox` is not
/// `+`. That is why "you for" scores 1 -- the word break stops the walk before
/// the unvoiced `f` -- while "Thank you" scores 0, because `k` is reached first.
fn vow_next_voiced(phones: &[Phone], i: usize, stress: i32, brk: &dyn Fn(usize) -> i32) -> i32 {
    if !crate::phoneset::is_vowel(&phones[i].phone) || stress != 1 {
        return 0;
    }
    for k in 0..6 {
        let j = i + 1 + k;
        let Some(p) = phones.get(j) else { return 0 };
        if crate::phoneset::is_vowel(&p.phone) {
            return 1;
        }
        let b = brk(j - 1);
        if b == BREAK_WORD || b == BREAK_PRE_PAUSE {
            return 1;
        }
        if !crate::phoneset::is_voiced_consonant(&p.phone) {
            return 0;
        }
    }
    0
}

/// Which phone a learned column describes: the unit's own, its predecessor, or
/// its successor. Read off the name, since Cepstral is consistent about it.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Ref {
    Own,
    Prev,
    Next,
}

fn ref_of(name: &str) -> Ref {
    if name.starts_with("p.") || name.contains("_prev_") || name.contains("_ap_prev") {
        Ref::Prev
    } else if name.starts_with("n.") || name.contains("_next_") || name.contains("_ap_next") {
        Ref::Next
    } else {
        Ref::Own
    }
}

/// Everything learned from one voice: the phone inventory and the per-phone value
/// of every column that turns out to depend only on phone identity.
pub struct PhoneTable {
    /// phone name -> `lisp_phone_nameid`
    pub ids: BTreeMap<String, u8>,
    /// nameid -> phone name
    pub names: BTreeMap<u8, String>,
    /// `learned[col][nameid]`, empty where the column is not phone-intrinsic
    learned: Vec<BTreeMap<u8, u8>>,
    /// `learned_pair[col][(own, other)]` for columns a single phone cannot fix.
    /// `lisp_prev_rnd` is the clear case: it depends on this phone *and* the one
    /// before it, so neither decides it alone. This also covers voices that
    /// simply lack a column others have -- William has no `lisp_voiced` at all,
    /// so nothing could be derived from it there.
    learned_pair: Vec<BTreeMap<(u8, u8), u8>>,
    /// `learned_trip[col][(own, other, stress)]`, for the columns that depend on
    /// stress as well. `lisp_vow_next_voiced` is one: the engine scores `ih1`
    /// before `v` as 1 and `ih0` before `jh` as 0, and `lisp_phone_nameid` does
    /// not carry stress, so the pair alone cannot tell them apart. The stress is
    /// read from the units' own `R:SylStructure.parent.stress` column, so this
    /// still learns from the voice rather than from English phonology.
    learned_trip: Vec<BTreeMap<(u8, u8, u8), u8>>,
    refs: Vec<Ref>,
    slots: Vec<Option<Slot>>,
    cols: BTreeMap<String, usize>,
    n_cols: usize,
}

impl PhoneTable {
    /// `lisp_phone_nameid` is intrinsic, so all unit types built on one phone share
    /// it, and the common prefix of each group names the phone. That avoids any
    /// hardcoded phone list and works on every voice.
    pub fn learn(v: &Voice, tc: &TargetCost) -> PhoneTable {
        let n_cols = tc.features.len();
        let id_col = tc.features.iter().position(|f| f == "lisp_phone_nameid");
        let next_col = tc.features.iter().position(|f| f == "n.lisp_phone_nameid");
        let prev_col = tc.features.iter().position(|f| f == "p.lisp_phone_nameid");

        let mut ids = BTreeMap::new();
        let mut names = BTreeMap::new();
        if let Some(idc) = id_col {
            let mut groups: BTreeMap<u8, Vec<&str>> = BTreeMap::new();
            for t in 0..v.types.len() {
                let r = v.candidates(t);
                if r.start >= r.end {
                    continue;
                }
                let mut h: BTreeMap<u8, usize> = BTreeMap::new();
                for u in r {
                    *h.entry(v.unit_feats(u)[idc]).or_insert(0) += 1;
                }
                if let Some((&modal, _)) = h.iter().max_by_key(|(_, &n)| n) {
                    groups.entry(modal).or_default().push(&v.types[t].name);
                }
            }
            for (id, members) in groups {
                let first = members[0];
                let pre = members.iter().fold(first.len(), |acc, m| {
                    acc.min(first.bytes().zip(m.bytes()).take_while(|(a, b)| a == b).count())
                });
                let phone = first[..pre.max(1)].to_string();
                ids.insert(phone.clone(), id);
                names.insert(id, phone);
            }
        }

        // Learn each column against whichever nameid column it is keyed by. A
        // "p." column describes the predecessor, so group by p.lisp_phone_nameid.
        let refs: Vec<Ref> = tc.features.iter().map(|f| ref_of(f)).collect();
        let slots: Vec<Option<Slot>> = tc.features.iter().map(|f| slot_of(f)).collect();
        let mut learned = vec![BTreeMap::new(); n_cols];
        let mut learned_pair = vec![BTreeMap::new(); n_cols];
        let mut learned_trip = vec![BTreeMap::new(); n_cols];
        let stress_col = tc.features.iter().position(|f| f == "R:SylStructure.parent.stress");

        for col in 0..n_cols {
            if slots[col].is_some() {
                continue;
            }
            let key_col = match refs[col] {
                Ref::Own => id_col,
                Ref::Prev => prev_col,
                Ref::Next => next_col,
            };
            let Some(kc) = key_col else { continue };

            let mut counts: BTreeMap<u8, BTreeMap<u8, usize>> = BTreeMap::new();
            for u in 0..v.num_units {
                let f = v.unit_feats(u);
                *counts.entry(f[kc]).or_default().entry(f[col]).or_insert(0) += 1;
            }
            for (phone_id, h) in counts {
                let total: usize = h.values().sum();
                if let Some((&modal, &n)) = h.iter().max_by_key(|(_, &n)| n) {
                    if total > 0 && n as f64 / total as f64 >= PURITY {
                        learned[col].insert(phone_id, modal);
                    }
                }
            }

            // The pair, and then the pair with stress. The partner is the phone
            // the column's own name points at, so a "next" column keys on
            // (this, next).
            //
            // These are built unconditionally. An earlier version skipped them
            // when the single key already predicted 98% of rows, which sounds
            // like a reasonable economy and is not: a column that is 99% zeros
            // reaches 99% by predicting zero everywhere, so the gate fired
            // exactly on the columns that needed the finer key. `lisp_prev_rnd`
            // is 1% ones and came out identically zero because of it.
            let Some(idc) = id_col else { continue };
            let other_col = match refs[col] {
                Ref::Prev => prev_col,
                _ => next_col,
            };
            let Some(oc) = other_col else { continue };

            let mut pairs: BTreeMap<(u8, u8), BTreeMap<u8, usize>> = BTreeMap::new();
            let mut trips: BTreeMap<(u8, u8, u8), BTreeMap<u8, usize>> = BTreeMap::new();
            for u in 0..v.num_units {
                let f = v.unit_feats(u);
                *pairs.entry((f[idc], f[oc])).or_default().entry(f[col]).or_insert(0) += 1;
                if let Some(sc) = stress_col {
                    *trips.entry((f[idc], f[oc], f[sc])).or_default()
                        .entry(f[col]).or_insert(0) += 1;
                }
            }
            for (key, h) in pairs {
                let total: usize = h.values().sum();
                if total < MIN_PAIR_SAMPLES {
                    continue;
                }
                if let Some((&modal, &n)) = h.iter().max_by_key(|(_, &n)| n) {
                    if n as f64 / total as f64 >= PURITY {
                        learned_pair[col].insert(key, modal);
                    }
                }
            }
            for (key, h) in trips {
                let total: usize = h.values().sum();
                if total < MIN_PAIR_SAMPLES {
                    continue;
                }
                if let Some((&modal, &n)) = h.iter().max_by_key(|(_, &n)| n) {
                    if n as f64 / total as f64 >= PURITY {
                        learned_trip[col].insert(key, modal);
                    }
                }
            }
        }

        let cols = tc.features.iter().enumerate()
            .map(|(i, f)| (f.clone(), i))
            .collect();
        PhoneTable { ids, names, learned, learned_pair, learned_trip, refs, slots, cols, n_cols }
    }

    pub fn id(&self, phone: &str) -> Option<u8> {
        self.ids.get(phone).copied()
    }

    /// The learned value of column `col` for the phone with id `nameid`.
    pub fn value(&self, col: usize, nameid: u8) -> Option<u8> {
        self.learned.get(col)?.get(&nameid).copied()
    }

    pub fn column(&self, name: &str) -> Option<usize> {
        self.cols.get(name).copied()
    }

    /// How many of the learned columns resolved for this phone. Diagnostic.
    pub fn coverage(&self, phone: &str) -> (usize, usize) {
        let Some(id) = self.id(phone) else { return (0, self.n_cols) };
        let mut have = 0;
        let mut total = 0;
        for col in 0..self.n_cols {
            if self.slots[col].is_some() {
                continue;
            }
            total += 1;
            if self.learned[col].contains_key(&id) {
                have += 1;
            }
        }
        (have, total)
    }
}

/// Where one phone sits in the utterance. Derived by `plan`, not supplied.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pos {
    /// Index of the syllable this phone belongs to.
    pub syl: usize,
    pub is_pause: bool,
    /// Distance in phones to the next pause; `u32::MAX` if none follows.
    pub to_pause: u32,
    pub syl_numphones: i32,
    pub word_numsyls: i32,
    /// This phone ends a syllable, and separately, a word.
    pub syl_final: bool,
    pub word_final: bool,
    /// Phones in this phone's syllable before its first vowel, and after its
    /// last. `syl_onsetsize` @0x13860 and `syl_codasize` @0x137d0 walk the
    /// syllable's daughters from each end, counting until `ph_vc` is `+`.
    pub syl_onsetsize: i32,
    pub syl_codasize: i32,
    /// `seg_onsetcoda` is `onset`: this phone comes before its syllable's vowel.
    /// The vowel itself counts as coda, which is what holds `lisp_onsetp` to 35%
    /// of units against a mean onset of 1.0 phone in a 2.9-phone syllable.
    pub in_onset: bool,
    /// The break *after* this phone's syllable: 0 inside a word, 1 at a word
    /// boundary, 3 at a phrase break, 4 at the end. `syl_break` @0x13700 returns
    /// 0 unless the syllable is its word's last.
    pub syl_break: i32,
    /// The same for the syllable before this one, skipping pauses.
    pub prev_syl_break: i32,
    /// Index of this phone's syllable within its word, for `position_type`.
    pub syl_in_word: i32,
}

/// Group a phone sequence into syllables and words so the structural columns can
/// be filled. Words are delimited by `Phone::word` where it is set, and by pauses
/// otherwise; syllables come from `rules::syl_boundary`, which is the engine's
/// own. `_rules` is unused now that the phoneset answers `is_vowel`, and is kept
/// so callers do not have to change.
pub fn plan(_rules: &NameRules, phones: &[Phone]) -> Vec<Pos> {
    let n = phones.len();
    let mut pos = vec![Pos::default(); n];

    // A syllable never spans two words, because the engine syllabifies one
    // lexicon entry at a time. Inside a word, `rules::syl_boundary` is
    // ceplex_us.dll's own maximal-onset test.
    // Identity is the word occurrence, not the spelling: "one one" is two
    // words. See `Phone::word_id`.
    let same_word = |a: usize, b: usize| match (&phones[a].word, &phones[b].word) {
        (Some(_), Some(_)) => phones[a].word_id == phones[b].word_id,
        _ => phones[a].phone != "pau" && phones[b].phone != "pau",
    };
    let names: Vec<&str> = phones.iter().map(|p| p.phone.as_str()).collect();
    let mut syl_start = 0usize;
    let mut syls: Vec<(usize, usize)> = Vec::new();
    let mut word_start = 0usize;
    for i in 0..n {
        if i > 0 && !same_word(i - 1, i) {
            word_start = i;
        }
        let ends = if let Some(e) = phones[i].syl_end {
            // a lexicon that records syllables (the festival pack does) wins
            e || i + 1 == n
        } else {
            i + 1 == n
                || phones[i].phone == "pau"
                || phones[i + 1].phone == "pau"
                || !same_word(i, i + 1)
                || {
                    let mut end = i + 1;
                    while end < n && same_word(i, end) {
                        end += 1;
                    }
                    rules::syl_boundary(
                        &names[word_start..end],
                        i - word_start,
                        syl_start.max(word_start) - word_start,
                    )
                }
        };
        if ends {
            syls.push((syl_start, i + 1));
            syl_start = i + 1;
        }
    }
    if syl_start < n {
        syls.push((syl_start, n));
    }

    for (k, &(a, b)) in syls.iter().enumerate() {
        for i in a..b {
            pos[i].syl = k;
            pos[i].syl_numphones = (b - a) as i32;
            pos[i].syl_final = i + 1 == b;
        }
    }

    // words: explicit if given, else a run between pauses
    let mut word_syls: Vec<usize> = vec![1; n];
    {
        let mut a = 0usize;
        while a < n {
            let mut b = a;
            let same = |x: &Phone, y: &Phone| match (&x.word, &y.word) {
                (Some(_), Some(_)) => x.word_id == y.word_id,
                _ => x.phone != "pau" && y.phone != "pau",
            };
            while b + 1 < n && same(&phones[a], &phones[b + 1]) && phones[b + 1].phone != "pau" {
                b += 1;
            }
            let nsyl = syls.iter().filter(|&&(s, e)| s >= a && e <= b + 1).count().max(1);
            for i in a..=b {
                word_syls[i] = nsyl;
                pos[i].word_final = i == b;
            }
            a = b + 1;
        }
    }
    for i in 0..n {
        pos[i].word_numsyls = word_syls[i] as i32;
        pos[i].is_pause = phones[i].phone == "pau";
    }

    // distance to the next pause
    let mut d = u32::MAX;
    for i in (0..n).rev() {
        if phones[i].phone == "pau" {
            d = 0;
        } else if d != u32::MAX {
            d = d.saturating_add(1);
        }
        pos[i].to_pause = d;
    }

    // onset and coda, measured from each end of the syllable to its vowel
    for &(a, b) in &syls {
        let first_vowel = (a..b).find(|&k| crate::phoneset::is_vowel(&phones[k].phone));
        let last_vowel = (a..b).rev().find(|&k| crate::phoneset::is_vowel(&phones[k].phone));
        let onset = first_vowel.map(|v| v - a).unwrap_or(b - a) as i32;
        let coda = last_vowel.map(|v| b - 1 - v).unwrap_or(0) as i32;
        for i in a..b {
            pos[i].syl_onsetsize = onset;
            pos[i].syl_codasize = coda;
            pos[i].in_onset = first_vowel.is_some_and(|v| i < v);
        }
    }

    // A pause is not a syllable, so it never carries a break of its own and is
    // skipped when looking backwards for the previous one.
    let syl_brk = |k: usize| -> i32 {
        let (_, b) = syls[k];
        let last = b - 1;
        if pos[last].is_pause || !pos[last].word_final {
            return 0;
        }
        match phones.get(last + 1) {
            Some(p) if p.phone == "pau" => {
                if last + 2 == n {
                    4
                } else {
                    3
                }
            }
            None => 4,
            _ => 1,
        }
    };
    let brks: Vec<i32> = (0..syls.len()).map(syl_brk).collect();
    let mut prev_real: Option<usize> = None;
    let mut prev_of = vec![0i32; syls.len()];
    for k in 0..syls.len() {
        let (a, b) = syls[k];
        if let Some(p) = prev_real {
            prev_of[k] = brks[p];
        }
        if !(a..b).all(|i| pos[i].is_pause) {
            prev_real = Some(k);
        }
    }
    for (k, &(a, b)) in syls.iter().enumerate() {
        for i in a..b {
            pos[i].syl_break = brks[k];
            pos[i].prev_syl_break = prev_of[k];
        }
    }

    // where the syllable sits in its word
    {
        let mut a = 0usize;
        while a < syls.len() {
            let w = syls[a].0;
            let mut b = a;
            while b + 1 < syls.len()
                && !pos[syls[b].1 - 1].word_final
                && !pos[syls[b].1 - 1].is_pause
            {
                b += 1;
            }
            for (j, k) in (a..=b).enumerate() {
                let (s, e) = syls[k];
                for i in s..e {
                    pos[i].syl_in_word = j as i32;
                }
            }
            let _ = w;
            a = b + 1;
        }
    }
    pos
}

/// Build one target feature vector per phone, in `unit_features` order.
pub fn build(
    v: &Voice,
    table: &PhoneTable,
    rules: &NameRules,
    tc: &TargetCost,
    phones: &[Phone],
) -> Vec<Vec<i32>> {
    let pos = plan(rules, phones);
    let n = phones.len();

    // one stress per syllable, taken from its most prominent phone
    let nsyl = pos.iter().map(|p| p.syl).max().map(|m| m + 1).unwrap_or(0);
    let mut syl_stress = vec![0i32; nsyl];
    for (i, p) in phones.iter().enumerate() {
        let k = pos[i].syl;
        if k < nsyl {
            syl_stress[k] = syl_stress[k].max(p.stress as i32);
        }
    }

    // A pause is not a syllable. `R:SylStructure.parent.R:Syllable.p` walks the
    // Syllable relation, which holds no pauses, so the syllable before the first
    // "one" of "nine <break> one <break> one" is "nine" and carries its stress.
    let syl_is_pause: Vec<bool> = {
        let mut v = vec![true; nsyl];
        for (i, p) in phones.iter().enumerate() {
            if !p.is_pause() && pos[i].syl < nsyl {
                v[pos[i].syl] = false;
            }
        }
        v
    };
    let syl_step = |from: usize, back: bool| -> i32 {
        let mut k = from;
        let mut skipped = 0usize;
        loop {
            k = if back {
                if k == 0 { return 0 }
                k - 1
            } else {
                if k + 1 >= nsyl { return 0 }
                k + 1
            };
            if !syl_is_pause[k] {
                return syl_stress[k];
            }
            // One pause is a phrase break and the relation runs through it; two
            // is a sentence boundary, which is a different utterance and has no
            // neighbouring syllable at all.
            skipped += 1;
            if skipped > 1 {
                return 0;
            }
        }
    };

    // The last phone of each syllable, for `lisp_uptalk`.
    let syl_last: Vec<usize> = {
        let mut v = vec![0usize; n];
        let mut end = n;
        for i in (0..n).rev() {
            if pos[i].syl_final || i + 1 == n {
                end = i;
            }
            v[i] = end;
        }
        v
    };

    let idf = |i: usize| -> u8 {
        phones.get(i).and_then(|p| table.id(&p.phone)).unwrap_or(0)
    };

    /// swift.dll's `lisp_uptalk` @0x11250 with its helper @0x10b50.
    ///
    /// 1 when a voiced phone sits in a syllable that a *phrase-internal* pause
    /// follows -- the rise before a comma, not the fall before a full stop. The
    /// helper classifies the phone first: "FP" if a pause comes next, "FS" if it
    /// is a vowel or one of `n m ng l r` in the coda. An onset consonant is
    /// neither, which is why the `n` opening "nine" scores 0 while its vowel and
    /// its coda `n` score 1.
    fn uptalk(phones: &[Phone], pos: &[Pos], syl_last: &[usize], i: usize) -> i32 {
        let name = phones[i].phone.as_str();
        if !(crate::phoneset::is_vowel(name) || crate::phoneset::is_voiced_consonant(name)) {
            return 0;
        }
        let next_is_pau = phones.get(i + 1).is_some_and(|p| p.is_pause());
        let vowel_initial = name.starts_with(['a', 'e', 'i', 'o', 'u']);
        let syllabic = matches!(name, "n" | "m" | "ng" | "l" | "r");
        // `seg_onsetcoda` is "coda" from the syllable's vowel onwards
        let start = (0..=i).rev().find(|&k| k == 0 || pos[k - 1].syl_final).unwrap_or(0);
        let in_coda = (start..i).any(|k| crate::phoneset::is_vowel(&phones[k].phone));
        if !(next_is_pau || vowel_initial || (syllabic && in_coda)) {
            return 0;
        }
        let last = syl_last[i];
        if !phones.get(last + 1).is_some_and(|p| p.is_pause()) {
            return 0;
        }
        match phones.get(last + 2) {
            Some(p) if !p.is_pause() => 1,
            _ => 0,
        }
    }

    // For each phone, the index of the last vowel in its phrase. `lisp_finalityp`
    // takes 1 from there up to the phone before the pause, which takes 2.
    let mut last_vowel = vec![usize::MAX; n];
    {
        let mut i = 0usize;
        while i < n {
            if phones[i].phone == "pau" {
                i += 1;
                continue;
            }
            let start = i;
            while i < n && phones[i].phone != "pau" {
                i += 1;
            }
            let lv = (start..i)
                .rev()
                .find(|&k| crate::phoneset::is_vowel(&phones[k].phone));
            if let Some(lv) = lv {
                for k in start..i {
                    last_vowel[k] = lv;
                }
            }
        }
    }

    // Only Swift 4 and 5 ask for durations, and only they pay for the tree walk.
    let dur = tc
        .features
        .iter()
        .any(|f| f == "lisp_durms" || f == "lisp_zscoredur_norm")
        .then(|| crate::dur::DurModel::open(v))
        .flatten()
        .map(|m| m.predict(phones, &pos, &syl_stress));

    (0..n)
        .map(|i| {
            let me = idf(i);
            let prev = if i > 0 { idf(i - 1) } else { 0 };
            let next = if i + 1 < n { idf(i + 1) } else { 0 };
            let next2 = if i + 2 < n { idf(i + 2) } else { 0 };
            let prev2 = if i >= 2 { idf(i - 2) } else { 0 };
            let d = dur.as_ref().map(|v| v[i]).unwrap_or_default();

            // to_pause counts phones until the next pause: 1 means the next phone
            // is the pause, so this is the last real phone of the phrase.
            let finality = if pos[i].is_pause {
                0
            } else if pos[i].to_pause == 1 {
                FINAL_PRE_PAUSE
            } else if pos[i].to_pause != u32::MAX && i >= last_vowel[i] {
                FINAL_APPROACHING
            } else {
                0
            };
            let brk_of = |k: usize| -> i32 {
                if pos[k].is_pause {
                    BREAK_PAUSE
                } else if pos[k].to_pause == 1 {
                    BREAK_PRE_PAUSE
                } else if pos[k].word_final {
                    BREAK_WORD
                } else if pos[k].syl_final {
                    BREAK_SYL
                } else {
                    0
                }
            };
            let brk = brk_of(i);
            let prev_brk = if i == 0 { 0 } else { brk_of(i - 1) };

            (0..tc.features.len())
                .map(|col| {
                    if let Some(s) = table.slots[col] {
                        return match s {
                            Slot::NameId => me as i32,
                            Slot::NextNameId => next as i32,
                            Slot::PrevNameId => prev as i32,
                            // the parent SYLLABLE's stress, so an onset or a coda
                            // carries the stress of the vowel it belongs to. The
                            // engine reports 1 for the `s` and the `th` in
                            // "civil" and "authority", where the phone's own
                            // stress is 0
                            Slot::Stress => {
                                syl_stress.get(pos[i].syl).copied().unwrap_or(0)
                            }
                            // these are the neighbouring SYLLABLES' stress, not
                            // the neighbouring phones' -- inside a syllable every
                            // phone shares one stress, so using the phone before
                            // simply repeats this phone's own value
                            Slot::PrevStress => syl_step(pos[i].syl, true),
                            Slot::NextStress => syl_step(pos[i].syl, false),
                            Slot::Uptalk => uptalk(phones, &pos, &syl_last, i),
                            Slot::Finality => finality,
                            Slot::SylNumPhones => pos[i].syl_numphones,
                            Slot::WordNumSyls => pos[i].word_numsyls,
                            Slot::Break => brk,
                            Slot::PrevBreak => prev_brk,
                            Slot::NextNextNameId => next2 as i32,
                            Slot::PrevPrevNameId => prev2 as i32,
                            Slot::NextBreak => {
                                if i + 1 < n { brk_of(i + 1) } else { 0 }
                            }
                            Slot::PrevPrevBreak => {
                                if i >= 2 { brk_of(i - 2) } else { 0 }
                            }
                            Slot::OnsetP => pos[i].in_onset as i32,
                            Slot::SylOnsetSize => pos[i].syl_onsetsize,
                            Slot::SylCodaSize => pos[i].syl_codasize,
                            Slot::SylBreak => pos[i].syl_break,
                            Slot::PrevSylBreak => pos[i].prev_syl_break,
                            Slot::DurMs => d.durms,
                            Slot::ZDurNorm => d.zdurnorm,
                            Slot::PrevPau => {
                                (i > 0 && phones[i - 1].phone == "pau") as i32
                            }
                            Slot::NextPau => {
                                (i + 1 < n && phones[i + 1].phone == "pau") as i32
                            }
                            Slot::VowNextVoiced => vow_next_voiced(
                                phones,
                                i,
                                syl_stress.get(pos[i].syl).copied().unwrap_or(0),
                                &brk_of,
                            ),
                            Slot::PrevRnd => (i > 0
                                && PREV_RND.contains(&phones[i - 1].phone.as_str()))
                                as i32,
                            Slot::PrevHighV => (i > 0
                                && PREV_HIGH_V.contains(&phones[i - 1].phone.as_str()))
                                as i32,
                            Slot::ApPrevApvow => ap_apvow(
                                phones,
                                table,
                                i,
                                i.checked_sub(1),
                                if i > 0 { brk_of(i - 1) } else { -1 },
                            ),
                            Slot::ApNextApvow => {
                                ap_apvow(phones, table, i, Some(i + 1), brk)
                            }
                            // swift.dll @0x11940: an alveolar stop between two
                            // vowels, syllable-initial after a syllable boundary,
                            // reports the phone id two ahead when that is a nasal
                            // or an approximant. The `t` of "routing" scores the
                            // id of its `ng`.
                            Slot::AlvNnNasap => {
                                let ok = i > 0
                                    && brk == 0
                                    && brk_of(i - 1) == BREAK_SYL
                                    && i + 1 < n
                                    && brk_of(i + 1) == 0
                                    && crate::phoneset::ctype(&phones[i].phone) == "s"
                                    && crate::phoneset::cplace(&phones[i].phone) == "a"
                                    && crate::phoneset::is_vowel(&phones[i + 1].phone)
                                    && crate::phoneset::is_vowel(&phones[i - 1].phone);
                                match phones.get(i + 2).filter(|_| ok) {
                                    Some(q)
                                        if crate::phoneset::ctype(&q.phone) == "n"
                                            || is_approximant(&q.phone) =>
                                    {
                                        table.id(&q.phone).unwrap_or(0) as i32
                                    }
                                    _ => 0,
                                }
                            }
                        };
                    }
                    let key = match table.refs[col] {
                        Ref::Own => me,
                        Ref::Prev => prev,
                        Ref::Next => next,
                    };
                    // most specific key first: (phone, partner, stress), then the
                    // pair, then the phone alone
                    let other = match table.refs[col] {
                        Ref::Prev => prev,
                        _ => next,
                    };
                    let st = syl_stress.get(pos[i].syl).copied().unwrap_or(0) as u8;
                    if let Some(x) = table.learned_trip[col].get(&(me, other, st)) {
                        return *x as i32;
                    }
                    if let Some(x) = table.learned_pair[col].get(&(me, other)) {
                        return *x as i32;
                    }
                    table.learned[col].get(&key).copied().unwrap_or(0) as i32
                })
                .collect()
        })
        .collect()
}
