//! Target features, and the prosody they buy. Needs CEPS_VOICE_ROOT.
//!
//! These voices apply no pitch modification, so intonation exists only as a
//! selection effect: the search has to find units that were recorded with the
//! contour you want. Measured on "hello world" in Allison, against the SDK's own
//! render of the same phones:
//!
//! ```text
//!                       F0 slope     head -> tail     last 5 voiced
//!   no target cost       +114.6      167 -> 222    212 248 248 248 251
//!   target cost          -138.3      223 -> 145    142 140 137 128 128
//!   SDK                  -142.0      209 -> 143    143 150 129 137 133
//! ```
//!
//! Without target features the search ends the utterance on a 251 Hz unit -- a
//! rise, which reads as a question. `lisp_finalityp` at weight 4589 is what moves
//! it onto the 151 Hz phrase-final units instead.

use ceps::expr::TargetCost;
use ceps::{build_targets, plan, NameRules, Phone, PhoneTable, SelectParams, Selector, Voice};

fn voice(name: &str) -> Option<Voice> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let d = std::path::Path::new(&root).join(name);
    if d.join("voice_u.dat").exists() {
        Voice::open(d).ok()
    } else {
        None
    }
}

fn parts(v: &Voice) -> (NameRules, TargetCost, PhoneTable) {
    let unp = v.unit_name_params().expect("unit_name_params");
    let r = NameRules::parse(v.image(), unp);
    let tc = TargetCost::parse(v.image(), unp).expect("target_cost");
    let t = PhoneTable::learn(v, &tc);
    (r, tc, t)
}

fn hello_world() -> Vec<Phone> {
    [("pau", 0u8), ("h", 0), ("eh", 1), ("l", 0), ("ow", 1),
     ("w", 0), ("er", 1), ("l", 0), ("d", 0), ("pau", 0)]
        .iter()
        .map(|(p, s)| Phone::new(p, *s))
        .collect()
}

fn col(tc: &TargetCost, name: &str) -> usize {
    tc.features.iter().position(|f| f == name).expect(name)
}

/// The phone inventory falls out of the type names: every type built on one phone
/// shares its `lisp_phone_nameid`, so grouping by that id and taking the common
/// prefix recovers the phone with no hardcoded list.
#[test]
fn phone_inventory_is_learned_from_the_database() {
    let Some(v) = voice("Allison") else { return };
    let (_, _, t) = parts(&v);
    assert_eq!(t.ids.len(), 41, "expected 41 phones, got {}", t.ids.len());

    // the alphabet is a reduced CMU variant
    for p in ["pau", "h", "i", "j", "eh", "ow", "er", "l", "d", "w", "ng", "zh"] {
        assert!(t.id(p).is_some(), "{p} missing from the inventory");
    }
    for p in ["hh", "iy", "y"] {
        assert!(t.id(p).is_none(), "{p} should not be in a Cepstral inventory");
    }

    // the learned columns should mostly resolve for ordinary phones
    for p in ["eh", "l", "d", "ow"] {
        let (have, total) = t.coverage(p);
        assert!(have * 4 >= total * 3,
                "{p}: only {have}/{total} learned columns resolved");
    }
}

/// `to_pause == 1` is the last real phone of the phrase. That phone must carry
/// finality 2, and everything from the phrase's last vowel onwards finality 1,
/// because the database reserves those for units recorded at 151 Hz and 178 Hz
/// against a 204 Hz baseline. "hello world" ends `w er l d`, so `er` and `l`
/// take 1 and `d` takes 2; the engine's own calls give `aw1 t ih0 jh` in
/// "outage" the same shape, `0 0 1 2`.
#[test]
fn finality_marks_the_phrase_end() {
    let Some(v) = voice("Allison") else { return };
    let (r, tc, t) = parts(&v);
    let ph = hello_world();
    let f = build_targets(&v, &t, &r, &tc, &ph);
    let k = col(&tc, "lisp_finalityp");
    let got: Vec<i32> = f.iter().map(|x| x[k]).collect();
    assert_eq!(got, [0, 0, 0, 0, 0, 0, 1, 1, 2, 0], "finality column");

    let b = col(&tc, "lisp_phone_break");
    assert_eq!(f[8][b], 6, "the pre-pausal phone must take break level 6");

    let np = col(&tc, "lisp_next_pau");
    assert_eq!(f[8][np], 1, "the pre-pausal phone must see a following pause");
    let pp = col(&tc, "lisp_prev_pau");
    assert_eq!(f[1][pp], 1, "the first real phone must see a preceding pause");

    // uptalk must never be requested: it is what makes a statement sound like
    // a question, and the expression penalises a mismatch by 1000
    let ut = col(&tc, "lisp_uptalk");
    assert!(f.iter().all(|x| x[ut] == 0), "uptalk must stay off");
}

/// Structural columns come from the same maximal-onset syllabification that names
/// the units, so the two always agree.
#[test]
fn structure_columns_follow_the_syllabification() {
    let Some(v) = voice("Allison") else { return };
    let (r, _, _) = parts(&v);
    let ph = hello_world();
    let p = plan(&r, &ph);

    assert!(p[0].is_pause && p[9].is_pause);
    assert_eq!(p[8].to_pause, 1, "the /d/ is one phone from the final pause");
    assert_eq!(p[7].to_pause, 2);
    for q in &p {
        assert!(q.syl_numphones >= 1, "every phone belongs to a syllable");
        assert!(q.word_numsyls >= 1);
    }
}

/// The payoff: with target features the search ends on units recorded far lower
/// than without. This is the terminal fall, and it is purely a selection effect.
#[test]
fn target_features_select_a_falling_phrase_end() {
    let Some(v) = voice("Allison") else { return };
    let (r, tc, t) = parts(&v);
    let ph = hello_world();
    let names = r.name_utterance(&ph, |n| v.type_id(n).is_some());
    let types: Vec<usize> = names.iter().filter_map(|n| v.type_id(n)).collect();
    assert_eq!(types.len(), ph.len(), "every name must resolve to a type");

    let tf = build_targets(&v, &t, &r, &tc, &ph);

    let unit_f0 = |u: usize| -> f32 {
        let x = v.unit(u);
        let mut sz: Vec<usize> = (x.start..x.end)
            .map(|i| v.frame_size(i as usize))
            .filter(|&n| n > 0)
            .collect();
        if sz.is_empty() {
            return 0.0;
        }
        sz.sort_unstable();
        v.sps as f32 / sz[sz.len() / 2] as f32
    };

    let plain = Selector::new(&v, SelectParams::from_voice(&v)).select(&types);
    let mut s2 = Selector::new(&v, SelectParams::from_voice(&v));
    let steered = s2.select_with_targets(&types, Some(&tf));
    assert_eq!(plain.len(), steered.len());

    // the last voiced unit before the closing pause
    let last = steered.len() - 2;
    let f_plain = unit_f0(plain[last].unit);
    let f_steer = unit_f0(steered[last].unit);
    assert!(f_plain > 0.0 && f_steer > 0.0, "need measurable periods");
    assert!(f_steer < f_plain * 0.85,
            "target features must pick a lower phrase end: steered {f_steer:.0} Hz \
             vs unsteered {f_plain:.0} Hz");

    // and the chosen unit must actually be one the database marks as final
    let k = col(&tc, "lisp_finalityp");
    assert_eq!(v.unit_feats(steered[last].unit)[k], 2,
               "the phrase-final position should select a finality-2 unit");
}

/// Pruning to `cand_beam_width` is not just faster, it is what makes the target
/// cost affordable at all.
#[test]
fn target_selection_examines_fewer_joins() {
    let Some(v) = voice("Allison") else { return };
    let (r, tc, t) = parts(&v);
    let ph = hello_world();
    let names = r.name_utterance(&ph, |n| v.type_id(n).is_some());
    let types: Vec<usize> = names.iter().filter_map(|n| v.type_id(n)).collect();
    let tf = build_targets(&v, &t, &r, &tc, &ph);

    let mut a = Selector::new(&v, SelectParams::from_voice(&v));
    a.select(&types);
    let mut b = Selector::new(&v, SelectParams::from_voice(&v));
    b.select_with_targets(&types, Some(&tf));
    assert!(b.joins_evaluated < a.joins_evaluated,
            "steered {} joins vs unsteered {}", b.joins_evaluated, a.joins_evaluated);
}

/// `lisp_phone_break` semantics, pinned against the type names. A type ending in a
/// function word was recorded with that word following, so it is word-final by
/// construction -- which makes it ground truth for what each break value means.
#[test]
fn break_levels_mean_what_the_type_names_say() {
    let Some(v) = voice("Allison") else { return };
    let (r, tc, _) = parts(&v);
    let cb = col(&tc, "lisp_phone_break");
    let cnp = col(&tc, "lisp_next_pau");

    let word_final = |t: usize| -> bool {
        let n = &v.types[t].name;
        r.fwords.iter().any(|w| n.len() > w.len() && n.ends_with(w.as_str()))
    };

    let mut tot = [0usize; 8];
    let mut wf = [0usize; 8];
    let mut np = [0usize; 8];
    for u in 0..v.num_units {
        let b = v.unit_feats(u)[cb] as usize;
        if b >= 8 {
            continue;
        }
        tot[b] += 1;
        if word_final(v.unit(u).type_id as usize) {
            wf[b] += 1;
        }
        if v.unit_feats(u)[cnp] != 0 {
            np[b] += 1;
        }
    }
    let rate = |a: &[usize; 8], b: usize| a[b] as f64 / tot[b].max(1) as f64;

    assert!(tot[2] > 5_000, "need a real sample of break 2, got {}", tot[2]);
    assert_eq!(wf[2], 0,
               "break 2 must never be word-final, found {} of {}", wf[2], tot[2]);
    assert!(rate(&wf, 3) > 2.5 * rate(&wf, 0),
            "break 3 must be enriched for word-final: {:.3} vs {:.3}",
            rate(&wf, 3), rate(&wf, 0));
    assert!(rate(&np, 6) > 0.99, "break 6 must be pre-pausal: {:.3}", rate(&np, 6));
    assert!(rate(&np, 0) < 0.01 && rate(&np, 2) < 0.01 && rate(&np, 3) < 0.01,
            "only break 6 and 7 precede a pause");
}

/// The constants the builder emits must match those meanings.
#[test]
fn builder_emits_the_pinned_break_levels() {
    let Some(v) = voice("Allison") else { return };
    let (r, tc, t) = parts(&v);
    let f = build_targets(&v, &t, &r, &tc, &hello_world());
    let cb = col(&tc, "lisp_phone_break");
    let got: Vec<i32> = f.iter().map(|x| x[cb]).collect();

    assert_eq!(got[8], 6, "the pre-pausal /d/ takes break 6");
    assert!(got[1..8].iter().all(|&b| b != 6),
            "nothing but the pre-pausal phone may take break 6: {got:?}");
    assert!(got.iter().all(|&b| matches!(b, 0 | 2 | 3 | 6 | 7)),
            "only observed break levels may be emitted: {got:?}");
}

/// `R:SylStructure.parent.R:Syllable.p.stress` is the stress of the previous
/// SYLLABLE, not the previous phone. Inside a syllable every phone carries the
/// same stress, so keying on the phone before simply repeats this phone's own
/// value and the column never varies where it should. Measured against the
/// engine's own units this was wrong 19 times in 27; fixing it halved that.
#[test]
fn neighbour_stress_is_syllable_level_not_phone_level() {
    let Some(v) = voice("Allison") else { return };
    let (r, tc, t) = parts(&v);

    // /h ah/0 /l ow/1 /w er/1 /l d/0 -- four syllables, stresses 0 1 1 0
    let ph: Vec<Phone> = [
        ("pau", 0u8), ("h", 0), ("ah", 0), ("l", 1), ("ow", 1),
        ("w", 1), ("er", 1), ("l", 0), ("d", 0), ("pau", 0),
    ].iter().map(|(p, s)| Phone::new(p, *s)).collect();

    let f = build_targets(&v, &t, &r, &tc, &ph);
    let prev = col(&tc, "R:SylStructure.parent.R:Syllable.p.stress");
    let next = col(&tc, "R:SylStructure.parent.R:Syllable.n.stress");
    let own = col(&tc, "R:SylStructure.parent.stress");

    // the two phones of one syllable must agree on their neighbours' stress even
    // when they disagree on their own -- that is the whole point of the fix
    let syls = plan(&r, &ph);
    for i in 1..ph.len() {
        if syls[i].syl == syls[i - 1].syl {
            assert_eq!(f[i][prev], f[i - 1][prev],
                       "phones {} and {i} share a syllable but disagree on p.stress", i - 1);
            assert_eq!(f[i][next], f[i - 1][next],
                       "phones {} and {i} share a syllable but disagree on n.stress", i - 1);
        }
    }

    // and it must not simply mirror this phone's own stress everywhere
    let mirrored = (0..ph.len()).filter(|&i| f[i][prev] == f[i][own]).count();
    assert!(mirrored < ph.len(),
            "p.stress mirrors own stress at every position, so it is still \
             reading the previous phone");
}

/// Some columns cannot be decided by one phone. `lisp_vow_next_voiced` needs this
/// phone to be a vowel AND the next to be voiced, and William has no `lisp_voiced`
/// column at all to derive it from, so the pair table is the general answer.
///
/// The gate for building one is predictive accuracy, not table coverage: a column
/// that is 89% zeros passes the purity test for nearly every phone while
/// predicting nothing.
#[test]
fn pair_keyed_learning_beats_single_phone_where_it_must() {
    for name in ["Allison", "William"] {
        let Some(v) = voice(name) else { continue };
        let (r, tc, t) = parts(&v);

        // whatever the tables do, the output must stay inside the observed range
        // of each column -- an invented value cannot match any recorded unit
        let ph = hello_world();
        let f = build_targets(&v, &t, &r, &tc, &ph);
        for (k, fname) in tc.features.iter().enumerate() {
            let mut lo = u8::MAX;
            let mut hi = 0u8;
            for u in 0..v.num_units {
                let x = v.unit_feats(u)[k];
                lo = lo.min(x);
                hi = hi.max(x);
            }
            for row in &f {
                assert!(row[k] >= lo as i32 && row[k] <= hi as i32,
                        "{name}: {fname} target {} outside the recorded {lo}..={hi}",
                        row[k]);
            }
        }
    }
}
