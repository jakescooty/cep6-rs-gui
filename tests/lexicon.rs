//! The pronunciation lexicon and the text front end.
//!
//! Needs CEPS_LEXICON (or the default pack at ..\packs\cmulex.bin), built by
//! `_out/pack_lex.py` from festival's `lib/dicts/cmu/cmudict-0.4.out`. Tests skip
//! silently when it is absent.

use ceps::{tokenize, text_to_phones, Lexicon, NameRules, Token, Voice};

fn lex() -> Option<Lexicon> {
    let p = std::env::var("CEPS_LEXICON")
        .unwrap_or_else(|_| r"..\packs\cmulex.bin".to_string());
    Lexicon::open(p).ok()
}

fn voice(name: &str) -> Option<Voice> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let d = std::path::Path::new(&root).join(name);
    if d.join("voice_u.dat").exists() {
        Voice::open(d).ok()
    } else {
        None
    }
}

/// The pack round-trips, and the alphabet is Cepstral's rather than festival's.
#[test]
fn lexicon_loads_and_speaks_cepstral_phones() {
    let Some(l) = lex() else { return };
    assert!(l.len() > 100_000, "expected the full dictionary, got {}", l.len());

    let names = l.phone_names();
    for p in ["h", "i", "j", "ah"] {
        assert!(names.iter().any(|x| x == p), "{p} missing from the pack");
    }
    for p in ["hh", "iy", "y", "ax"] {
        assert!(!names.iter().any(|x| x == p),
                "{p} is a festival symbol and must have been folded");
    }
}

/// The fold from festival's alphabet is not a guess: the engine's own naming of
/// "telephone", recovered by inverse-filtering an SDK render, was
/// `teh eh1 lah ah0 f ow0 n`, and festival gives /t eh/1 /l ax/0 /f ow n/1.
/// Those agree exactly once ax folds to ah.
#[test]
fn telephone_matches_the_engines_own_pronunciation() {
    let Some(l) = lex() else { return };
    let syls = l.lookup("telephone").expect("telephone");
    assert_eq!(syls.len(), 3);
    assert_eq!(syls[0].phones, ["t", "eh"]);
    assert_eq!(syls[0].stress, 1);
    assert_eq!(syls[1].phones, ["l", "ah"]);
    assert_eq!(syls[1].stress, 0);
    assert_eq!(syls[2].phones, ["f", "ow", "n"]);
}

/// Dictionary syllable boundaries must override the maximal-onset guess, and the
/// resulting names must all be real types.
#[test]
fn lexicon_syllables_drive_the_naming() {
    let (Some(l), Some(v)) = (lex(), voice("Allison")) else { return };
    let unp = v.unit_name_params().expect("unit_name_params");
    let r = NameRules::parse(v.image(), unp);

    let ph = l.word_phones("telephone").expect("telephone");
    assert_eq!(ph.len(), 7);
    // syl_end marks the last phone of each syllable: t eh | l ah | f ow n
    let ends: Vec<bool> = ph.iter().map(|p| p.syl_end.unwrap()).collect();
    assert_eq!(ends, [false, true, false, true, false, false, true]);

    let names = r.name_utterance(&ph, |n| v.type_id(n).is_some());
    for n in &names {
        assert!(v.type_id(n).is_some(), "{n} is not a type in this voice");
    }
    assert_eq!(names[0], "teh", "the onset must fuse with its nucleus");
}

#[test]
fn tokenizer_keeps_contractions_and_breaks_on_punctuation() {
    let t = tokenize("Don't stop; really, don't. OK?");
    let words: Vec<&str> = t.iter().filter_map(|x| match x {
        Token::Word(w) => Some(w.as_str()),
        _ => None,
    }).collect();
    assert_eq!(words, ["don't", "stop", "really", "don't", "ok"]);

    let hard: Vec<bool> = t.iter().filter_map(|x| match x {
        Token::Break(h, _) => Some(*h),
        _ => None,
    }).collect();
    assert_eq!(hard, [false, false, true, true], "sentence ends must be hard breaks");
}

/// Text in, phones out, with pauses at both ends and at every break.
#[test]
fn text_to_phones_brackets_the_utterance_with_pauses() {
    let Some(l) = lex() else { return };
    let (ph, missing) = text_to_phones(&l, "Hello world. This is a test.");
    assert!(missing.is_empty(), "unexpected gaps: {missing:?}");
    assert!(ph.first().unwrap().is_pause(), "must open on a pause");
    assert!(ph.last().unwrap().is_pause(), "must close on a pause");
    assert!(ph.iter().filter(|p| p.is_pause()).count() >= 3,
            "the sentence break must produce a pause");
    // no two pauses in a row
    assert!(ph.windows(2).all(|w| !(w[0].is_pause() && w[1].is_pause())),
            "pauses must not double up");

    let real: Vec<&str> = ph.iter().filter(|p| !p.is_pause())
        .map(|p| p.phone.as_str()).collect();
    assert!(real.len() > 15, "expected a real phone string, got {real:?}");
}

/// An unpronounceable run of letters is spelled out, which is what the engine
/// does and what `us_aswd` is for: "zzqxwv" has no legal English onset, so it
/// becomes six letter names rather than a word or a hole in the sentence.
#[test]
fn unpronounceable_words_are_spelled_out() {
    let Some(l) = lex() else { return };
    let (ph, missing) = text_to_phones(&l, "hello zzqxwv world");
    assert!(missing.is_empty(), "nothing should be dropped: {missing:?}");
    assert!(!ph.is_empty(), "the known words must still be spoken");
    assert!(l.lookup("zzqxwv").is_none());
    assert!(!ceps::aswd::is_word("zzqxwv"));
    assert!(ceps::aswd::is_word("nasa"), "a word-shaped string is not spelled out");
}

/// Break level 3 marks a word boundary. A function word's units span all of its
/// phones and only the last is word-final, so the share taking break 3 should be
/// 1/n for an n-phone word -- a far tighter prediction than aggregate enrichment.
#[test]
fn function_word_break_rate_tracks_phone_count() {
    let (Some(l), Some(v)) = (lex(), voice("Allison")) else { return };
    let unp = v.unit_name_params().expect("unit_name_params");
    let r = NameRules::parse(v.image(), unp);
    let tc = ceps::expr::TargetCost::parse(v.image(), unp).expect("target_cost");
    let cb = tc.features.iter().position(|f| f == "lisp_phone_break").unwrap();

    let mut checked = 0usize;
    for w in &r.fwords {
        let Some(syls) = l.lookup(w) else { continue };
        let n: usize = syls.iter().map(|s| s.phones.len()).sum();
        if n < 2 {
            continue;
        }

        let (mut total, mut three) = (0usize, 0usize);
        for t in 0..v.types.len() {
            let name = &v.types[t].name;
            if !(name.len() > w.len() && name.ends_with(w.as_str())) {
                continue;
            }
            for u in v.candidates(t) {
                total += 1;
                if v.unit_feats(u)[cb] == 3 {
                    three += 1;
                }
            }
        }
        if total < 100 {
            continue;
        }
        let rate = three as f64 / total as f64;
        let want = 1.0 / n as f64;
        assert!((rate - want).abs() < 0.15,
                "{w} ({n} phones): break-3 rate {rate:.2}, expected ~{want:.2}");
        checked += 1;
    }
    assert!(checked >= 5, "only {checked} function words exercised");
}
