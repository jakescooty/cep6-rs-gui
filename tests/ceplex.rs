//! Cepstral's own lexicon and letter-to-sound, lifted from ceplex_us.dll.
//!
//! Needs the pack at CEPS_CEPLEX (default ..\packs\ceplex.bin), built by
//! `_out/pack_ceplex.py`. Skips silently without it.
//!
//! Every expected value here was captured from the running engine by hooking
//! `lex_lookup` and `lts_apply` (`_out/run_lex.py`), so these are equality tests
//! against the real thing rather than plausibility checks.

use ceps::{CepLex, DigitStyle, NameRules, Voice};

fn cep() -> Option<CepLex> {
    let p = std::env::var("CEPS_CEPLEX")
        .unwrap_or_else(|_| r"..\packs\ceplex.bin".to_string());
    CepLex::open(p).ok()
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

#[test]
fn pack_has_the_expected_shape() {
    let Some(c) = cep() else { return };
    assert_eq!(c.len(), 13_387, "entry count");
    assert_eq!(c.nodes(), 33_059, "LTS nodes");
    // the letter table is latin-1, and decoding it as UTF-8 silently mangles the
    // accented tail into replacement characters
    let l = c.letters();
    assert!(l.starts_with("abcdefghijklmnopqrstuvwxyz"), "letters: {l:?}");
    assert!(l.contains('é') && l.contains('ü'), "accented letters lost: {l:?}");
    assert!(!l.contains('\u{fffd}'), "letters were decoded as UTF-8: {l:?}");
}

/// The dictionary holds what the LTS gets wrong. `colonel` is the clean case:
/// letter-to-sound says `k aa1 er1 n ah0 l`, which is not a word, so the
/// dictionary carries `k er1 n ah0 l` instead.
#[test]
fn dictionary_holds_what_lts_gets_wrong() {
    let Some(c) = cep() else { return };

    let (ph, from_lts) = c.phones("colonel");
    assert!(!from_lts, "colonel must come from the dictionary");
    assert_eq!(ph.join(" "), "k er1 n ah0 l");
    assert_ne!(c.lts("colonel").join(" "), "k er1 n ah0 l",
               "if LTS already got it right the entry would be redundant");

    // and the common words are absent precisely because LTS handles them
    for w in ["the", "of", "and", "to", "one", "said"] {
        assert!(c.lookup(w).is_none(), "{w} unexpectedly in the dictionary");
        assert!(!c.lts(w).is_empty(), "LTS gave up on {w}");
    }
}

/// Homographs are filed by part of speech and the engine returns the last match.
#[test]
fn hello_returns_the_entry_the_engine_returns() {
    let Some(c) = cep() else { return };
    let (ph, from_lts) = c.phones("hello");
    assert!(!from_lts);
    assert_eq!(ph.join(" "), "h eh1 l ow0",
               "the engine returns the second of hello's two entries");
}

/// Letter-to-sound, against captured `lts_apply` output.
#[test]
fn lts_matches_the_engine() {
    let Some(c) = cep() else { return };
    let cases = [
        ("world", "w er1 l d"),
        ("telephone", "t eh1 l ah0 f ow0 n"),
        ("authority", "ao0 th ao1 r ih0 t i0"),
        ("emergency", "ih0 m er1 jh ih0 n s i0"),
        ("counties", "k aw1 n t i0 z"),
        ("following", "f aa1 l ow0 ih0 ng"),
        ("outage", "aw1 t ih0 jh"),
        ("areas", "eh1 r i0 ah0 z"),
        ("has", "h ae1 z"),
        ("issued", "ih1 sh j uw0 d"),
        ("mescalero", "m eh1 s k ey1 l eh1 r ow0"),
        ("otero", "ow1 t eh1 r ow0"),
        ("zyzzyva", "z ih1 z ih0 v ah0"),
        ("strengths", "s t r eh1 ng th s"),
    ];
    for (w, want) in cases {
        assert_eq!(c.lts(w).join(" "), want, "{w}");
    }
}

/// Stress rides on the phone name here, and this lexicon records no syllable
/// structure, so naming must fall back to maximal onset.
#[test]
fn phones_carry_stress_and_leave_syllables_open() {
    let Some(c) = cep() else { return };
    let ph = c.word_phones("telephone").expect("telephone");
    let names: Vec<String> = ph.iter().map(|p| p.phone.clone()).collect();
    assert_eq!(names, ["t", "eh", "l", "ah", "f", "ow", "n"]);
    assert_eq!(ph[1].stress, 1, "eh1");
    assert_eq!(ph[3].stress, 0, "ah0");
    assert!(ph.iter().all(|p| p.syl_end.is_none()),
            "this lexicon has no syllable boundaries to report");
}

/// Everything the front end produces must name a real unit type, or selection
/// fails at runtime.
#[test]
fn every_phone_reaches_a_real_unit_type() {
    let (Some(c), Some(v)) = (cep(), voice("William")) else { return };
    let unp = v.unit_name_params().expect("unit_name_params");
    let r = NameRules::parse(v.image(), unp);

    let text = "A civil authority has issued a nine one one telephone outage \
                emergency for the following counties or areas in Otero, \
                including Mescalero.";
    let (seq, _, missing) = ceps::text_to_phones_cepstral(&c, text, DigitStyle::Digits);
    assert!(missing.is_empty(), "unpronounceable: {missing:?}");
    assert!(seq.len() > 40, "expected a long phone string, got {}", seq.len());

    let names = r.name_utterance(&seq, |n| v.type_id(n).is_some());
    for n in &names {
        assert!(v.type_id(n).is_some(), "{n} is not a type in this voice");
    }
}

/// The two front ends genuinely disagree; this is why the Cepstral one exists.
#[test]
fn cepstral_and_festival_front_ends_differ() {
    let Some(c) = cep() else { return };
    let fest = std::env::var("CEPS_LEXICON")
        .unwrap_or_else(|_| r"..\packs\cmulex.bin".to_string());
    let Ok(f) = ceps::Lexicon::open(fest) else { return };

    let mut differ = 0;
    let mut total = 0;
    for w in ["hello", "colonel", "the", "world", "telephone", "authority",
              "emergency", "counties", "following", "areas"] {
        let ours = c.phones(w).0.join(" ");
        let Some(theirs) = f.lookup(w) else { continue };
        let flat: Vec<String> = theirs
            .iter()
            .flat_map(|s| s.phones.iter().map(move |p| format!("{}{}", p, s.stress)))
            .collect();
        total += 1;
        if ours != flat.join(" ") {
            differ += 1;
        }
    }
    assert!(total >= 8, "expected most probes to be in both, got {total}");
    assert!(differ * 2 > total,
            "the two front ends should differ on most words, got {differ}/{total}");
}
