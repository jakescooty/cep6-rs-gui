//! The lexicon's part-of-speech side, against the engine's own tagged lookups.
//!
//! swift.dll runs a Penn-tag part-of-speech tagger before it touches the
//! lexicon, and the tag decides the pronunciation of a homograph: `record/vbp`
//! is /r ih0 k ao1 r d/ and `record/nn` is /r eh1 k er0 d/. The tagger itself is
//! a maximum-entropy model of about 400,000 features filling most of
//! ceplang_en.dll, and is not ported -- see HANDOFF's "What is left".
//!
//! What *is* settled is the lookup: given a tag, which entry the engine reads.
//! `_out/pos_calls.txt` is a capture of `lex_lookup(word, pos)` and its return
//! value from `_out/run_lex.py` over a paragraph of homographs in both readings,
//! and this replays every tagged row of it.
//!
//! Needs CEPS_VOICE_ROOT and the capture; skips silently otherwise.

use ceps::CepLex;

fn ceplex() -> Option<CepLex> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let p = std::path::Path::new(&root).join("ceplex.bin");
    CepLex::open(p.to_str()?).ok()
}

/// `word <tab> Penn tag <tab> phones`, one tagged lookup a line, written by
/// `_out/gen_pos_calls.py` from a `run_lex.py` capture.
fn capture() -> Option<Vec<(String, String, Vec<String>)>> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let p = std::path::Path::new(&root).join("_out").join("pos_calls.txt");
    let raw = std::fs::read_to_string(p).ok()?;
    let mut out = Vec::new();
    for line in raw.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 3 {
            continue;
        }
        out.push((f[0].to_string(), f[1].to_string(),
                  f[2].split_whitespace().map(|p| p.to_string()).collect()));
    }
    Some(out)
}

/// Every tagged lookup the engine made, reproduced exactly.
#[test]
fn tagged_lookups_match_the_engine() {
    let Some(c) = ceplex() else { return };
    let Some(rows) = capture() else { return };
    assert!(rows.len() > 70, "only {} tagged rows; is the capture present?", rows.len());

    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();
    for (word, tag, want) in &rows {
        let Some(got) = c.lookup_tag(word, Some(tag)) else {
            // the engine reached the letter-to-sound rules for this one
            continue;
        };
        checked += 1;
        if &got != want {
            bad.push(format!("{word}/{tag}: engine {want:?} ours {got:?}"));
        }
    }
    assert!(checked > 45, "only {checked} rows checked");
    // One known exception, and it is our packing rather than the rule.
    // "protest" has two `v` entries; the engine's binary search lands on the
    // second and ours on the first, because `_out/pack_ceplex.py` re-sorts each
    // word's block by part-of-speech character while the DLL keeps the order its
    // own build emitted. Reproducing it would mean repacking the lexicon in the
    // DLL's index order, which would also move the untagged `'0'` fallback that
    // the bit-exact synthesis tests depend on. Left as it is, deliberately.
    let expected: Vec<&str> = vec!["protest/vbp"];
    let got: Vec<&str> = bad.iter().map(|s| s.split(':').next().unwrap()).collect();
    assert_eq!(got, expected,
               "{} of {checked} tagged lookups differ: {}", bad.len(), bad.join(" | "));
}

/// The tag's first character is the key, so the whole `nn`/`nns` and
/// `vb`/`vbp`/`vbd`/`vbz` families collapse onto one entry each.
#[test]
fn only_the_tags_first_character_matters() {
    let Some(c) = ceplex() else { return };
    let noun = c.lookup_tag("record", Some("nn")).expect("record/nn");
    let verb = c.lookup_tag("record", Some("vbp")).expect("record/vbp");
    assert_ne!(noun, verb, "record is a homograph");
    for t in ["nn", "nns", "nnp", "n"] {
        assert_eq!(c.lookup_tag("record", Some(t)).unwrap(), noun, "{t}");
    }
    for t in ["vb", "vbp", "vbd", "vbz", "vbg"] {
        assert_eq!(c.lookup_tag("record", Some(t)).unwrap(), verb, "{t}");
    }
}

/// A tag the word has no entry for falls back to `'0'`, and so does no tag at
/// all -- which is what the synthesis path passes, having no tagger.
#[test]
fn an_unmatched_tag_falls_back_to_the_default() {
    let Some(c) = ceplex() else { return };
    let plain = c.lookup("abstract").expect("abstract");
    assert_eq!(c.lookup_tag("abstract", Some("jj")).unwrap(), plain,
               "abstract has no j entry");
    assert_eq!(c.lookup_tag("abstract", None).unwrap(), plain);
    assert_eq!(plain, ["ae1", "b", "s", "t", "r", "ae0", "k", "t"],
               "the '0' entry is the noun reading");
}

/// Which of several entries under one character wins is decided by where the
/// binary search lands, not by a first-or-last rule. "evening" has two `n`
/// entries and the engine reads the second; "lead" has two `v` entries and it
/// reads the first. `find_full_match` walking back from the hit and then
/// forward is what produces both.
#[test]
fn the_search_landing_decides_among_duplicates() {
    let Some(c) = ceplex() else { return };
    let ns: Vec<Vec<String>> = c.lookup_all("evening").into_iter()
        .filter(|(p, _)| *p == b'n').map(|(_, ph)| ph).collect();
    assert_eq!(ns.len(), 2, "evening should have two n entries");
    assert_ne!(ns[0], ns[1]);
    assert_eq!(c.lookup_tag("evening", Some("nn")).unwrap(), ns[1],
               "evening takes the second of its two n entries");
    assert_eq!(c.lookup_tag("evening", Some("nn")).unwrap(),
               ["i1", "v", "ah0", "n", "ih0", "ng"]);

    let vs: Vec<Vec<String>> = c.lookup_all("lead").into_iter()
        .filter(|(p, _)| *p == b'v').map(|(_, ph)| ph).collect();
    assert_eq!(vs.len(), 2, "lead should have two v entries");
    assert_eq!(c.lookup_tag("lead", Some("vbp")).unwrap(), vs[0],
               "lead takes the first of its two v entries");
    assert_eq!(c.lookup_tag("lead", Some("vbp")).unwrap(), ["l", "i1", "d"]);
}
