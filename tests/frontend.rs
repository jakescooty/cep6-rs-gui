//! Text normalisation and letter-to-sound.
//!
//! Normalisation needs no data files. LTS needs the pack at CEPS_LTS (default
//! ..\packs\cmults.bin) built by `_out/pack_lts.py`, and skips silently without it.

use ceps::norm::{cardinal, ordinal};
use ceps::{normalize, normalize_with, DigitStyle, Lexicon, Lts, NameRules, Token,
           Voice};

fn words(text: &str, style: DigitStyle) -> Vec<String> {
    normalize(text, style)
        .into_iter()
        .filter_map(|t| match t {
            Token::Word(w) => Some(w),
            Token::Break(..) => None,
        })
        .collect()
}

fn say(text: &str) -> String {
    words(text, DigitStyle::Cardinal).join(" ")
}

fn lts() -> Option<Lts> {
    let p = std::env::var("CEPS_LTS").unwrap_or_else(|_| r"..\packs\cmults.bin".to_string());
    Lts::open(p).ok()
}

fn lex() -> Option<Lexicon> {
    let p = std::env::var("CEPS_LEXICON")
        .unwrap_or_else(|_| r"..\packs\cmulex.bin".to_string());
    Lexicon::open(p).ok()
}

fn ceplex() -> Option<ceps::CepLex> {
    let p = std::env::var("CEPS_CEPLEX")
        .unwrap_or_else(|_| r"..\packs\ceplex.bin".to_string());
    ceps::CepLex::open(p).ok()
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
fn cardinals_read_conventionally() {
    let n = |x: u64| {
        let mut v = Vec::new();
        cardinal(x, &mut v);
        v.join(" ")
    };
    assert_eq!(n(0), "zero");
    assert_eq!(n(7), "seven");
    assert_eq!(n(13), "thirteen");
    assert_eq!(n(20), "twenty");
    assert_eq!(n(21), "twenty-one");
    assert_eq!(n(100), "one hundred");
    assert_eq!(n(911), "nine hundred eleven");
    assert_eq!(n(1200), "one thousand two hundred");
    assert_eq!(n(1_000_000), "one million");
    assert_eq!(n(2_500_017), "two million five hundred thousand seventeen");
}

#[test]
fn ordinals_read_conventionally() {
    let n = |x: u64| {
        let mut v = Vec::new();
        ordinal(x, &mut v);
        v.join(" ")
    };
    assert_eq!(n(1), "first");
    assert_eq!(n(3), "third");
    assert_eq!(n(12), "twelfth");
    assert_eq!(n(20), "twentieth");
    assert_eq!(n(21), "twenty-first");
    assert_eq!(n(100), "one hundredth");
}

/// Dates and times, against the engine rather than against convention.
///
/// Two of these are not the conventional readings and are the engine's all the
/// same, probed in `_out/probe_norm.txt`: a four-digit number from 2000 on is
/// "two thousand seventeen", not "twenty seventeen", because `en_exp_id` falls
/// through to the plain number whenever the second digit is a zero; and a time
/// on the hour simply drops the minutes, so 10:00 PM is "ten p m".
#[test]
fn dates_and_times() {
    assert_eq!(say("DEC 3, 2025"), "december third two thousand twenty five");
    assert_eq!(say("on JAN 1, 1984"), "on january first nineteen eighty four");
    assert_eq!(say("in 1905"), "in nineteen oh five");
    assert_eq!(say("in 1900"), "in nineteen hundred");
    assert_eq!(say("in 2000"), "in two thousand");
    assert_eq!(say("at 12:00 PM"), "at twelve p m");
    assert_eq!(say("at 12:30"), "at twelve thirty");
    assert_eq!(say("at 9:05"), "at nine oh five");
    // outside the year range it is an ordinary quantity
    assert_eq!(say("2025 apples"), "two thousand twenty five apples");
    assert_eq!(say("5000 apples"), "five thousand apples");
}

/// Capitals with no vowel are spelled out unless the dictionary holds them, and
/// a state abbreviation after a place name is the state. Against the engine,
/// `_fe/engine_cases.json`.
#[test]
fn initialisms_are_spelled_out() {
    assert_eq!(say("Otero, NM"), "otero new mexico");
    assert_eq!(say("PM"), "p m");
    assert_eq!(say("the FBI said"), "the f b i said");
    assert_eq!(say("Otero"), "otero");

    // Cepstral's dictionary holds fbi as one word, pronounced spelled, and the
    // engine looks it up whole
    let Some(cep) = ceplex() else { return };
    let with_dict = |t: &str| {
        normalize_with(t, DigitStyle::Cardinal, |w| cep.lookup(w).is_some())
            .into_iter()
            .filter_map(|x| match x {
                Token::Word(w) => Some(w),
                Token::Break(..) => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    assert_eq!(with_dict("NASA said"), "nasa said");
    assert_eq!(with_dict("the FBI said"), "the fbi said");
    assert_eq!(with_dict("Otero, NM"), "otero new mexico");
}

/// Every sentence the engine was captured on, word for word: ceplang_en.dll's
/// whole tokenizer against SAPI + Frida.
#[test]
fn tokenizer_matches_every_engine_capture() {
    let Some(cep) = ceplex() else { return };
    let mut bad = Vec::new();
    let mut n = 0;
    for line in include_str!("engine_words.tsv").lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (text, want) = line.split_once('\t').unwrap();
        let got: Vec<String> =
            normalize_with(text, DigitStyle::Cardinal, |w| cep.lookup(w).is_some())
                .into_iter()
                .filter_map(|t| match t {
                    Token::Word(w) => Some(w),
                    Token::Break(..) => None,
                })
                .collect();
        n += 1;
        if got.join(" ") != want {
            bad.push(format!("{text}\n  engine: {want}\n  ours:   {}", got.join(" ")));
        }
    }
    assert!(n > 300, "the capture table is missing or short: {n} sentences");
    assert!(bad.is_empty(), "{} of {n} differ:\n{}", bad.len(), bad.join("\n"));
}

/// Rules 21 and 22 of the engine's tokenizer, captured in `_fe/engine_cases.json`.
#[test]
fn roman_numerals_follow_the_engine() {
    assert_eq!(say("King Henry VIII and Louis XIV were there."),
               "king henry the eighth and louis the fourteenth were there");
    assert_eq!(say("Part II, Act III, Scene V."), "part two act three scene five");
    assert_eq!(say("Rocky II and World War II."), "rocky i i and world war two");
    assert_eq!(say("Mac OS X is here."), "mac os ten is here");
    // a comma before the numeral spells it, whatever comes before that
    assert_eq!(say("Say IV, then XX now."), "say i v then x x now");
    // no L, C, D or M in the engine's pattern, so these reach the state rule
    assert_eq!(say("We drove to Springfield, IL."), "we drove to springfield illinois");
    assert_eq!(say("We drove to Springfield, MD."), "we drove to springfield maryland");
    // a run of one character is counted
    assert_eq!(say("He said zzzzz loudly."), "he said five z 's loudly");
}

#[test]
fn money_percent_and_decimals() {
    assert_eq!(say("$1200"), "one thousand two hundred dollars");
    assert_eq!(say("$1"), "one dollar");
    assert_eq!(say("3.5%"), "three point five percent");
    assert_eq!(say("100%"), "one hundred percent");
    assert_eq!(say("the 21st"), "the twenty first");
}

/// The one genuinely ambiguous case, and the reason `DigitStyle` exists. A number
/// carrying a unit is a quantity either way.
#[test]
fn digit_style_switches_codes_but_not_quantities() {
    assert_eq!(words("911", DigitStyle::Cardinal).join(" "), "nine hundred eleven");
    assert_eq!(words("911", DigitStyle::Digits).join(" "), "nine one one");
    for s in [DigitStyle::Cardinal, DigitStyle::Digits] {
        assert_eq!(words("100%", s).join(" "), "one hundred percent", "{s:?}");
        assert_eq!(words("$1200", s).join(" "), "one thousand two hundred dollars");
        assert_eq!(words("at 12:00", s).join(" "), "at twelve");
        assert_eq!(words("DEC 3, 2025", s).join(" "),
                   "december third two thousand twenty five");
    }
    // a leading zero was never a quantity
    assert_eq!(words("007", DigitStyle::Cardinal).join(" "), "zero zero seven");
}

/// A sentence-final period must not be swallowed into the number before it.
#[test]
fn sentence_ends_are_not_decimals() {
    let t = normalize("It was 2025. Then 3.5 more.", DigitStyle::Cardinal);
    let s: Vec<String> = t.iter().map(|x| match x {
        Token::Word(w) => w.clone(),
        Token::Break(h, _) => (if *h { "|" } else { "," }).to_string(),
    }).collect();
    assert_eq!(s.join(" "),
               "it was two thousand twenty five | then three point five more");
}

#[test]
fn lts_pronounces_what_the_dictionary_lacks() {
    let Some(l) = lts() else { return };
    assert_eq!(l.letters(), 26);

    let p = l.predict("mescalero");
    assert!(!p.is_empty(), "LTS produced nothing");
    assert!(p.len() >= 7, "expected a full pronunciation, got {p:?}");
    assert!(p.iter().all(|x| x.syl_end.is_none()),
            "LTS predicts phones, not syllable structure");

    // every phone must be one the voices actually have
    if let Some(v) = voice("Allison") {
        let unp = v.unit_name_params().expect("unit_name_params");
        let r = NameRules::parse(v.image(), unp);
        let names = r.name_utterance(&p, |n| v.type_id(n).is_some());
        for n in &names {
            assert!(v.type_id(n).is_some(), "LTS led to a dead type {n}");
        }
    }
}

/// LTS must never be consulted for a word the dictionary holds, and must produce
/// something for ordinary unseen words rather than nothing.
#[test]
fn lts_covers_unseen_words() {
    let (Some(l), Some(lx)) = (lts(), lex()) else { return };
    for w in ["mescalero", "ipawscap", "zzqxwv", "flarnbistle"] {
        assert!(lx.lookup(w).is_none(), "{w} is unexpectedly in the dictionary");
        assert!(!l.predict(w).is_empty(), "LTS gave up on {w}");
    }
    // and it is only a fallback: the dictionary answers first
    let (_, guessed, missing) = ceps::text_to_phones_with(&lx, Some(&l), "hello world");
    assert!(guessed.is_empty(), "known words must not reach LTS: {guessed:?}");
    assert!(missing.is_empty());
}

/// The whole front end on real alerting text: every word must be pronounced.
#[test]
fn eas_text_is_fully_pronounceable() {
    let (Some(l), Some(lx)) = (lts(), lex()) else { return };
    let text = "A civil authority has issued A 911 TELEPHONE OUTAGE EMERGENCY for \
                the following counties or areas: Otero, NM; at 12:00 PM on DEC 3, \
                2025. Effective until 12:00 PM DEC 4, 2025.";
    let (ph, _, missing) =
        ceps::text_to_phones_styled(&lx, Some(&l), text, DigitStyle::Digits);
    assert!(missing.is_empty(), "nothing should be unpronounceable: {missing:?}");
    assert!(ph.len() > 100, "expected a long phone string, got {}", ph.len());
    assert!(ph.first().unwrap().is_pause() && ph.last().unwrap().is_pause());
}

/// A colon after a word is punctuation, not part of a time.
#[test]
fn colons_split_words_but_join_times() {
    assert_eq!(say("counties or areas: Otero"), "counties or areas otero");
    assert_eq!(say("at 12:00"), "at twelve");
    // The engine splits these on the minute's width, not on context: two digits
    // after the colon is a time and anything else is a ratio. Probed as
    // "The ratio 3:1 is fine." / "The ratio 3:15 is fine.", which come back
    // "three to one" and "three fifteen".
    assert_eq!(say("ratio 3:1"), "ratio three to one");
    assert_eq!(say("at 3:15"), "at three fifteen");
}

/// Everything the normaliser emits must be a word the dictionary holds, or the
/// front end quietly falls through to letter-to-sound on its own output.
#[test]
fn normaliser_output_is_all_dictionary_words() {
    let Some(lx) = lex() else { return };
    let text = "A 911 alert for Otero, NM; at 12:00 PM on DEC 3, 2025. \
                It rose 3.5% to $1200 on the 21st, up 100%, in 1905 and 1900.";
    for style in [DigitStyle::Cardinal, DigitStyle::Digits] {
        for t in normalize_with(text, style, |w| lx.lookup(w).is_some()) {
            if let Token::Word(w) = t {
                assert!(lx.lookup(&w).is_some(),
                        "{style:?}: normaliser emitted {w:?}, which is not a word");
            }
        }
    }
}
