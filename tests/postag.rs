//! The part-of-speech tagger, against the engine's own tagging.
//!
//! `_out/pos_tags.txt` and `_out/pos_tags2.txt` are captures of
//! `lex_lookup(word, pos)` flattened by `_out/gen_tag_calls.py`: one utterance a
//! line, its words, then the Penn tag the engine gave each of them, or `-` for
//! an utterance the engine did not tag at all.
//!
//! Both halves are checked. The gate -- does the tagger run -- has to be exact,
//! because tagging a sentence the engine left alone would change output that is
//! currently bit-identical. The model is a statistical classifier and is scored
//! against a floor instead.
//!
//! Needs CEPS_VOICE_ROOT and the captures; skips silently otherwise.

use ceps::PosTag;

fn postag() -> Option<PosTag> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    PosTag::open(std::path::Path::new(&root).join("pos.bin")).ok()
}

/// `(sentence, words, tags)` per utterance; `None` where the engine did not tag.
type Row = (String, Vec<String>, Vec<Option<String>>);

fn capture(name: &str) -> Option<Vec<Row>> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let p = std::path::Path::new(&root).join("_out").join(name);
    let raw = std::fs::read_to_string(p).ok()?;
    let mut out = Vec::new();
    for line in raw.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 3 {
            continue;
        }
        out.push((
            f[0].to_string(),
            f[1].split_whitespace().map(str::to_string).collect(),
            f[2].split_whitespace()
                .map(|t| if t == "-" { None } else { Some(t.to_string()) })
                .collect(),
        ));
    }
    Some(out)
}

/// Every capture: homographs in noun and verb frames, the hand-written
/// paragraph, and the frames that put a comma against the homograph.
fn all() -> Option<Vec<Row>> {
    let mut rows = capture("pos_tags.txt")?;
    rows.extend(capture("pos_tags2.txt")?);
    rows.extend(capture("pos_tags3.txt")?);
    Some(rows)
}

fn ceplex() -> Option<ceps::CepLex> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    ceps::CepLex::open(std::path::Path::new(&root).join("ceplex.bin")).ok()
}

/// The model is 45 Penn outcomes and the gate is 237 words.
#[test]
fn the_pack_holds_what_it_should() {
    let Some(p) = postag() else { return };
    assert_eq!(p.outcomes().len(), 45, "the full WSJ tagset");
    assert_eq!(p.homographs().len(), 237);
    for w in ["record", "evening", "wind", "lead", "live", "close", "address"] {
        assert!(p.is_homograph(w), "{w} should be on the list");
    }
    // these carry two POS entries with different phones and still do not trigger
    for w in ["axes", "laden", "agape", "aged", "a", "export", "read", "airport"] {
        assert!(!p.is_homograph(w), "{w} should not be on the list");
    }
}

/// Whether the tagger runs at all has to match exactly: an utterance the engine
/// left untagged takes the lexicon's `'0'` default, and tagging it anyway would
/// move output that is currently bit-identical to the engine's.
#[test]
fn the_gate_matches_the_engine_exactly() {
    let Some(p) = postag() else { return };
    let Some(rows) = all() else { return };
    assert!(rows.len() > 250, "only {} utterances captured", rows.len());

    let mut bad = Vec::new();
    for (_, words, tags) in &rows {
        let engine_tagged = tags.iter().any(|t| t.is_some());
        let ours_tagged = p.tag_utterance(words).is_some();
        if engine_tagged != ours_tagged {
            bad.push(format!("{:?} engine {engine_tagged} ours {ours_tagged}", words.join(" ")));
        }
    }
    assert!(bad.is_empty(), "{} of {} utterances gated wrongly: {}",
            bad.len(), rows.len(), bad.join(" | "));
}

/// The model, through the front end that feeds it -- which is what decides
/// whether the tagger sees the sentence's commas.
///
/// These captures are homographs in frames built to force the decision both
/// ways, so this is an adversarial floor and not a typical rate.
#[test]
fn tags_agree_with_the_engine() {
    let Some(p) = postag() else { return };
    let Some(cep) = ceplex() else { return };
    let Some(rows) = all() else { return };
    let style = ceps::DigitStyle::default();

    let (mut ok, mut n, mut skew) = (0usize, 0usize, 0usize);
    let mut bad: Vec<String> = Vec::new();
    for (sentence, words, tags) in &rows {
        let ours = ceps::lex::tags_for_text(&cep, &p, sentence, style);
        if ours.len() != words.len() {
            skew += 1;
            continue;
        }
        for ((w, got), (want, ww)) in ours.iter().zip(tags.iter().zip(words)) {
            assert_eq!(w, ww, "front end and capture disagree on the words");
            let Some(want) = want else { continue };
            n += 1;
            match got {
                Some(g) if want.eq_ignore_ascii_case(g) => ok += 1,
                Some(g) => bad.push(format!("{w} engine {want} ours {g}")),
                None => bad.push(format!("{w} engine {want} ours untagged")),
            }
        }
    }
    assert!(skew < 12, "{skew} sentences tokenised differently from the capture");
    assert!(n > 1300, "only {n} tagged words checked");
    let pct = 100.0 * ok as f64 / n as f64;
    println!("{ok}/{n} tags agree ({pct:.1}%), {skew} sentences skipped");
    assert!(pct >= 94.0, "only {ok}/{n} tags agree ({pct:.1}%): {}",
            bad.iter().take(12).cloned().collect::<Vec<_>>().join(" | "));
}

/// The comma probe on its own. Punctuation reaches the tagger or it does not,
/// and this is the capture that can tell: on the same sentences, dropping the
/// marks costs 14 points.
#[test]
fn punctuation_reaches_the_tagger() {
    let Some(p) = postag() else { return };
    let Some(cep) = ceplex() else { return };
    let Some(rows) = capture("pos_tags3.txt") else { return };
    let style = ceps::DigitStyle::default();

    let (mut with, mut without, mut n) = (0usize, 0usize, 0usize);
    for (sentence, words, tags) in &rows {
        let ours = ceps::lex::tags_for_text(&cep, &p, sentence, style);
        let bare = p.tag_utterance(words);
        if ours.len() != words.len() {
            continue;
        }
        for (i, (want, (_, got))) in tags.iter().zip(&ours).enumerate() {
            let Some(want) = want else { continue };
            n += 1;
            with += usize::from(got.as_deref().is_some_and(|g| want.eq_ignore_ascii_case(g)));
            without += usize::from(
                bare.as_ref().and_then(|b| b.get(i)).is_some_and(|g| want.eq_ignore_ascii_case(g)),
            );
        }
    }
    assert!(n > 600, "only {n} words checked");
    println!("with punctuation {with}/{n}, without {without}/{n}");
    assert!(with > without + n / 20,
            "punctuation should be worth more than 5 points: {with} vs {without} of {n}");
}

/// A sentence with no listed homograph must come out of the front end exactly as
/// it did before the tagger existed. This is the regression that protects the
/// bit-exact passage in `tests/engineparity.rs`.
#[test]
fn untagged_text_is_untouched() {
    let root = std::env::var("CEPS_VOICE_ROOT").ok();
    let Some(root) = root else { return };
    let Some(p) = postag() else { return };
    let Ok(cep) = ceps::CepLex::open(std::path::Path::new(&root).join("ceplex.bin")) else {
        return;
    };
    let style = ceps::DigitStyle::default();
    let long = std::fs::read_to_string(std::path::Path::new(&root).join("_out").join("long1.txt"));

    let mut texts = vec![
        "The public will be alerted when the phone system returns to normal operations."
            .to_string(),
        "The airport is closed. The overall plan is fine.".to_string(),
    ];
    if let Ok(t) = long {
        texts.push(t);
    }
    for text in &texts {
        let (plain, _, _) = ceps::lex::text_to_phones_cepstral(&cep, text, style);
        let (tagged, _, _) = ceps::lex::text_to_phones_tagged(&cep, Some(&p), text, style);
        assert_eq!(plain.len(), tagged.len(), "phone count moved on untagged text");
        for (a, b) in plain.iter().zip(&tagged) {
            assert_eq!((&a.phone, a.stress), (&b.phone, b.stress),
                       "untagged text changed: {}", &text[..text.len().min(48)]);
        }
    }
}

/// And a sentence that does hold one must change, in the direction the engine
/// changed it: "record" the verb keeps its second-syllable stress.
#[test]
fn a_homograph_sentence_takes_the_tagged_entry() {
    let root = std::env::var("CEPS_VOICE_ROOT").ok();
    let Some(root) = root else { return };
    let Some(p) = postag() else { return };
    let Ok(cep) = ceps::CepLex::open(std::path::Path::new(&root).join("ceplex.bin")) else {
        return;
    };
    let tags = p.tag_utterance(&["we", "record", "the", "record", "now"]).expect("triggered");
    assert_eq!(tags.len(), 5);
    assert!(tags[1].starts_with("VB"), "first 'record' is the verb, got {}", tags[1]);
    assert_eq!(tags[3], "NN", "second 'record' is the noun, got {}", tags[3]);

    assert_eq!(cep.lookup_tag("record", Some(&tags[1])).unwrap(),
               ["r", "ih0", "k", "ao1", "r", "d"]);
    assert_eq!(cep.lookup_tag("record", Some(&tags[3])).unwrap(),
               ["r", "eh1", "k", "er0", "d"]);
}

/// "evening" is the case this whole line of work started from, and it lands
/// right for a reason worth recording: we tag it VBG where the engine says NN,
/// and it does not matter. The lexicon keys on the tag's first character, the
/// `v` entry and the second `n` entry are the same phones, so both tags select
/// `i1 v ah0 n ih0 ng` -- the engine's own answer. A wrong tag only costs
/// something when it reaches a different entry.
#[test]
fn evening_is_right_even_though_the_tag_is_not() {
    let Some(p) = postag() else { return };
    let Some(cep) = ceplex() else { return };
    let tags = p.tag_utterance(&["this", "evening", "the", "evening", "news", "ran"])
        .expect("evening is on the homograph list");
    let want = ["i1", "v", "ah0", "n", "ih0", "ng"];
    for (i, slot) in [1usize, 3].into_iter().enumerate() {
        let got = cep.lookup_tag("evening", Some(&tags[slot])).expect("evening");
        assert_eq!(got, want, "occurrence {i} tagged {} gave {got:?}", tags[slot]);
    }
    // and the plain '0' entry is the one that was wrong before the tagger
    assert_eq!(cep.lookup("evening").unwrap(), ["i1", "v", "n", "ih0", "ng"]);
}

/// The feature set is only right if the weights it pulls are the right ones, so
/// spot-check the two the layout argument turned on: `W0_the` is DT, not NNP.
#[test]
fn the_weights_are_attached_to_the_right_tag() {
    let Some(p) = postag() else { return };
    let the = p.feature_weights("W0_the");
    let best = the.iter().max_by(|a, b| a.1.total_cmp(&b.1)).expect("W0_the");
    assert_eq!(best.0, "DT", "W0_the should be DT; NNP means the records are off by one");
    assert!(p.feature_weights("SUF7_ransfer").iter().any(|(t, _)| *t == "VB"));
    assert!(p.feature_weights("this_is_not_a_feature").is_empty());
}
