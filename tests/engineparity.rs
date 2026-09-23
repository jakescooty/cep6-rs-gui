//! Parity against the engine's own calls, captured by `_out/run_cost.py`.
//!
//! Frida hooks four functions inside swift.dll while SAPI speaks:
//!
//! ```text
//! 0x21170  cl_target_cost(item, unit, db, params) -> int
//! 0x220a0  frame distance (db, u0, u1)            -> int
//! 0x21f00  short-unit pair penalty                -> int
//! 0x002960 ffeature_int(item, name)               -> int
//! ```
//!
//! The `ffeature_int` rows are the engine's own target feature vector for each
//! position, which is the one thing a reimplementation cannot check any other
//! way: voice_d.dat holds what the *voice build tools* computed, and on six
//! columns that is not what the runtime computes. Everything here is measured,
//! not assumed.
//!
//! Needs CEPS_VOICE_ROOT and the captures in `_out`; skips silently otherwise.

use ceps::{build_targets, NameRules, Phone, PhoneTable, Voice};
use ceps::expr::TargetCost;

fn voice(name: &str) -> Option<Voice> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let d = std::path::Path::new(&root).join(name);
    if d.join("voice_u.dat").exists() {
        Voice::open(d).ok()
    } else {
        None
    }
}

fn capture(name: &str) -> Option<Vec<(String, i64, String, i64, i64, i64)>> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let p = std::path::Path::new(&root).join("_out").join(name);
    let raw = std::fs::read_to_string(p).ok()?;
    let mut rows = Vec::new();
    for line in raw.lines() {
        let t = line.trim().trim_start_matches('[').trim_end_matches(']');
        let f: Vec<&str> = t.split(',').map(|x| x.trim()).collect();
        if f.len() < 6 {
            continue;
        }
        let s = |x: &str| x.trim_matches('"').to_string();
        let n = |x: &str| x.trim_matches('"').parse::<i64>().unwrap_or(0);
        rows.push((s(f[0]), n(f[1]), s(f[2]), n(f[3]), n(f[4]), n(f[5])));
    }
    rows.sort_by_key(|r| r.1);
    Some(rows)
}

fn front_end(text: &str) -> Vec<Phone> {
    let path = std::env::var("CEPS_CEPLEX")
        .unwrap_or_else(|_| r"..\packs\ceplex.bin".to_string());
    let c = ceps::CepLex::open(&path).expect("ceplex pack");
    ceps::text_to_phones_cepstral(&c, text, ceps::DigitStyle::Cardinal).0
}

/// The engine's target feature vectors, recovered from its `ffeature_int` calls.
/// Nested lookups appear in the stream -- `lisp_pal_lat` asks for `phone_id`
/// while it is being evaluated -- so take the names that match the next expected
/// feature and ignore the rest.
fn engine_vectors(rows: &[(String, i64, String, i64, i64, i64)], feats: &[String]) -> Vec<Vec<i32>> {
    let mut out = Vec::new();
    let mut cur: Vec<i32> = Vec::new();
    for r in rows.iter().filter(|r| r.0 == "ffint") {
        if r.2 == feats[0] {
            cur.clear();
            cur.push(r.5 as i32);
            continue;
        }
        let k = cur.len();
        if k > 0 && k < feats.len() && r.2 == feats[k] {
            cur.push(r.5 as i32);
            if cur.len() == feats.len() {
                out.push(std::mem::take(&mut cur));
            }
        }
    }
    out
}

const CASES: [(&str, &str); 3] = [
    ("cost_s1.jsonl", "The civil authority has issued a telephone outage emergency."),
    ("cost_s2.jsonl",
     "Calls are now routing normally. Thank you for your patience during the outage."),
    ("cost_s3.jsonl",
     "Bob sang a long dumb song. The cave man moved his big red van. \
      Mister Abstract displays extra monsters."),
];

/// Every target feature column, on every captured position, exactly.
///
/// This is the test that pins the front end. Before the `lisp_*` functions were
/// read out of swift.dll it passed on 24 of 37 columns; the misses were the
/// syllabifier, `ceplex_us_postlex`, and five features whose rules cannot be
/// inferred from voice_d.dat because the database disagrees with the runtime.
#[test]
fn target_features_match_the_engine_exactly() {
    let Some(v) = voice("William") else { return };
    let unp = v.unit_name_params().expect("unit_name_params");
    let tc = TargetCost::parse(v.image(), unp).expect("target_cost");
    let nr = NameRules::parse(v.image(), unp);
    let table = PhoneTable::learn(&v, &tc);

    let mut checked = 0usize;
    for (file, text) in CASES {
        let Some(rows) = capture(file) else { continue };
        let engine = engine_vectors(&rows, &tc.features);
        if engine.is_empty() {
            continue;
        }
        let ph = front_end(text);
        let names = nr.name_utterance(&ph, |n| v.type_id(n).is_some());
        let ours = build_targets(&v, &table, &nr, &tc, &ph);
        let scored: Vec<usize> = (0..names.len()).filter(|&k| !names[k].contains("pau")).collect();
        assert_eq!(engine.len(), scored.len(),
                   "{file}: {} engine vectors against {} non-pause positions",
                   engine.len(), scored.len());

        for (k, e) in engine.iter().enumerate() {
            let o = &ours[scored[k]];
            for c in 0..tc.features.len() {
                assert_eq!(e[c], o.get(c).copied().unwrap_or(0),
                           "{file} position {k} ({}) column {} `{}`: engine {}, ours {}",
                           ph[scored[k]].phone, c, tc.features[c], e[c],
                           o.get(c).copied().unwrap_or(0));
            }
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} positions checked; are the captures present?");
}

/// The join cost, replayed against the engine's own return values.
///
/// `sub_220a0` is the whole frame-distance path: the pause and consecutive-unit
/// short circuits, the end-of-database guard, and the weighted Manhattan sum
/// scaled by the pair's channel-1 level. The last of those is what flite does
/// not have, and leaving it out made the join cost about seventy times too small
/// against the target cost.
#[test]
fn frame_distance_matches_the_engine_exactly() {
    let Some(v) = voice("William") else { return };
    let mut p = ceps::SelectParams::from_voice(&v);
    // optimal_coupling 2 is the frame distance alone; sub_21f00's short-unit
    // penalty is a separate call in the engine and a separate capture here
    p.optimal_coupling = 2;
    let mut s = ceps::Selector::new(&v, p);

    let mut checked = 0usize;
    for (file, _) in CASES {
        let Some(rows) = capture(file) else { continue };
        for r in rows.iter().filter(|r| r.0 == "fdist") {
            let (u0, u1) = (r.2.parse::<usize>().unwrap_or(0), r.3 as usize);
            assert_eq!(s.join_cost(u0, u1) as i64, r.5,
                       "{file}: frame distance {u0} -> {u1}");
            checked += 1;
        }
    }
    assert!(checked > 1000, "only {checked} joins checked; are the captures present?");
}

/// And the short-unit penalty: 0 when both units are longer than four frames,
/// 200000 when one is four or fewer, 400000 when one is a single frame.
#[test]
fn short_unit_penalty_matches_the_engine_exactly() {
    let Some(v) = voice("William") else { return };
    let mut checked = 0usize;
    for (file, _) in CASES {
        let Some(rows) = capture(file) else { continue };
        for r in rows.iter().filter(|r| r.0 == "shortpen") {
            let u0 = r.2.parse::<usize>().unwrap_or(0);
            let (u1, u1prev) = (r.3 as usize, r.4 as i32);
            let want = r.5;
            let got = if u0 as i32 == u1prev {
                0
            } else {
                let (a, b) = (v.unit(u0), v.unit(u1));
                let (l0, l1) = (a.end - a.start, b.end - b.start);
                if l1 > 4 && l0 > 4 { 0 } else if l1 > 1 && l0 > 1 { 200_000 } else { 400_000 }
            };
            assert_eq!(got as i64, want, "{file}: short-unit penalty {u0} -> {u1}");
            checked += 1;
        }
    }
    assert!(checked > 1000, "only {checked} penalties checked");
}

/// The waveform, against the engine's own rendering of its own periods.
///
/// `_out/frames.txt` is the exact `(frame, size)` sequence the engine emitted
/// while producing `_ref/parity1.wav` in the same run, so replaying it removes
/// selection, naming and duration from the comparison entirely: what is left is
/// the LPC path and only the LPC path.
///
/// Three things have to be right at once for this to pass, and a float filter,
/// a zeroed scratch buffer or a missing piece boundary each break it: the
/// coefficients are int32 in Q14, the filter history keeps the full 32-bit
/// accumulator, and the scratch buffer carries the previous period's output into
/// each sample *until the engine streams a piece and reallocates it*.
/// `_out/pieces.txt` holds those boundaries, captured from the same run.
#[test]
fn the_waveform_matches_the_engines_own_render() {
    let Some(v) = voice("William") else { return };
    let root = std::env::var("CEPS_VOICE_ROOT").unwrap();
    for (fr, pc, rf) in [("frames.txt", "pieces.txt", "parity1.wav"),
                         ("frames2.txt", "pieces2.txt", "parity2.wav")] {
        let out = std::path::Path::new(&root).join("_out");
        let wav = std::path::Path::new(&root).join("_ref").join(rf);
        let (Ok(raw), Ok(pieces), Ok(ref_bytes)) = (std::fs::read_to_string(out.join(fr)),
                                                    std::fs::read_to_string(out.join(pc)),
                                                    std::fs::read(&wav))
        else {
            continue;
        };
        let starts: Vec<usize> = pieces.lines().filter(|l| !l.starts_with('#'))
            .filter_map(|l| l.trim().parse().ok()).collect();
        assert!(!starts.is_empty(), "{pc} has no piece boundaries");

        let mut st = ceps::LpcState::new(&v);
        for (n, line) in raw.lines().filter(|l| l.split_whitespace().count() >= 2).enumerate() {
            let f: Vec<&str> = line.split_whitespace().collect();
            let (frame, size) = (f[0].parse::<usize>().unwrap(), f[1].parse::<usize>().unwrap());
            assert_eq!(v.frame_size(frame), size,
                       "the engine emitted frame {frame} at a size it was not recorded at");
            if starts.contains(&n) {
                st.new_piece();
            }
            st.emit_period(&v, frame, size, ceps::GAIN_UNITY);
        }

        // the reference wav is 16-bit mono; find `data` and read what follows
        let pos = ref_bytes.windows(4).position(|w| w == b"data").expect("data chunk") + 8;
        let want: Vec<i16> = ref_bytes[pos..]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(st.out.len(), want.len(), "{rf}: sample count");

        let first = st.out.iter().zip(&want).position(|(a, b)| a != b);
        assert_eq!(first, None, "{rf}: first differing sample");
    }
}

/// The same claim with nothing held back: text in, the engine's samples out.
///
/// The render test above is handed the engine's periods and its piece
/// boundaries. This one is handed the sentence. Lexicon, letter-to-sound,
/// syllabification, post-lexical rules, naming, target features, the search and
/// the filter all have to agree with swift.dll for the bytes to match.
#[test]
fn the_waveform_is_identical_to_the_engine_from_text() {
    let Some(v) = voice("William") else { return };
    let root = std::env::var("CEPS_VOICE_ROOT").unwrap();
    let ceplex = std::path::Path::new(&root).join("ceplex.bin");
    let Ok(cep) = ceps::CepLex::open(ceplex.to_str().unwrap()) else { return };

    // The third is the one that matters: 47 seconds, seven sentences, and every
    // awkward shape at once -- 911 and 9-1-1, two phone numbers, two clock
    // times with meridiems, two dates, "St. Croix", "VI", a colon and a
    // semicolon. The text lives in _out/long1.txt so the two stay in step.
    let long = std::path::Path::new(&root).join("_out").join("long1.txt");
    let long = std::fs::read_to_string(long).unwrap_or_default();
    for (text, rf) in [
        ("The civil authority has issued a telephone outage emergency.", "parity1.wav"),
        ("Calls are now routing normally. Thank you for your patience during the outage.",
         "parity2.wav"),
        (long.trim(), "long1.wav"),
    ] {
        if text.is_empty() {
            continue;
        }
        let wav = std::path::Path::new(&root).join("_ref").join(rf);
        let Ok(ref_bytes) = std::fs::read(&wav) else { continue };
        let pos = ref_bytes.windows(4).position(|w| w == b"data").expect("data chunk") + 8;
        let want: Vec<i16> = ref_bytes[pos..]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        let pcm = ceps::say_pcm(&v, &cep, text, ceps::DigitStyle::Cardinal);

        assert_eq!(pcm.len(), want.len(), "{rf}: sample count");
        assert_eq!(pcm.iter().zip(&want).position(|(a, b)| a != b), None,
                   "{rf}: first differing sample");
    }
}

/// End to end: text in, the same units out that the engine chose.
///
/// `_out/frames.txt` and `frames2.txt` are `emit_period` captures, so each is
/// the exact sequence of database frames the engine played. Mapping those back
/// onto units gives its unit sequence, and selection from the same text has to
/// reproduce it position for position.
#[test]
fn selection_reproduces_the_engines_units() {
    let Some(v) = voice("William") else { return };
    let root = std::env::var("CEPS_VOICE_ROOT").unwrap();
    let unp = v.unit_name_params().expect("unit_name_params");
    let tc = TargetCost::parse(v.image(), unp).expect("target_cost");
    let nr = NameRules::parse(v.image(), unp);
    let table = PhoneTable::learn(&v, &tc);

    let mut starts: Vec<(i32, usize)> =
        (0..v.num_units).map(|u| (v.unit(u).start, u)).collect();
    starts.sort_unstable();

    let cases: [(&str, &str); 2] = [
        ("frames.txt", "The civil authority has issued a telephone outage emergency."),
        ("frames2.txt",
         "Calls are now routing normally. Thank you for your patience during the outage."),
    ];
    let mut ran = 0usize;
    for (file, text) in cases {
        let p = std::path::Path::new(&root).join("_out").join(file);
        let Ok(raw) = std::fs::read_to_string(&p) else { continue };

        let mut engine: Vec<usize> = Vec::new();
        for line in raw.lines() {
            let Some(f) = line.split_whitespace().next() else { continue };
            let Ok(frame) = f.parse::<i32>() else { continue };
            let i = starts.partition_point(|&(s, _)| s <= frame);
            if i == 0 {
                continue;
            }
            let u = starts[i - 1].1;
            let x = v.unit(u);
            if x.start <= frame && frame < x.end && engine.last() != Some(&u) {
                engine.push(u);
            }
        }
        assert!(engine.len() > 20, "{file}: only {} units recovered", engine.len());

        let ph = front_end(text);
        let names = nr.name_utterance(&ph, |n| v.type_id(n).is_some());
        let types: Vec<usize> = names.iter().filter_map(|n| v.type_id(n)).collect();
        let tf = build_targets(&v, &table, &nr, &tc, &ph);
        let mut sel = ceps::Selector::new(&v, ceps::SelectParams::from_voice(&v));
        let ours = sel.select_with_targets(&types, Some(&tf));

        assert_eq!(ours.len(), engine.len(), "{file}: unit count");
        let same = ours.iter().zip(&engine).filter(|(c, &u)| c.unit == u).count();
        assert_eq!(same, engine.len(),
                   "{file}: {same}/{} units match the engine", engine.len());

        // The streaming boundaries too, which the units alone do not pin down:
        // the pieces file holds the period index at which each `sub_27810` call
        // was seen, and our piece list is in units, so one is converted to the
        // other through the period count of each unit.
        let pf = std::path::Path::new(&root).join("_out")
            .join(if file == "frames.txt" { "pieces.txt" } else { "pieces2.txt" });
        if let Ok(pieces) = std::fs::read_to_string(&pf) {
            let want: Vec<usize> = pieces.lines().filter(|l| !l.starts_with('#'))
                .filter_map(|l| l.trim().parse().ok()).collect();
            let periods = |u: usize| (v.unit(u).end - v.unit(u).start) as usize;
            let mut got = Vec::new();
            let mut at = 0usize;
            let mut n = 0usize;
            for &len in &sel.last_pieces {
                got.push(n);
                for u in &ours[at..at + len] {
                    n += periods(u.unit);
                }
                at += len;
            }
            assert_eq!(got, want, "{file}: streamed piece boundaries, in periods");
        }
        ran += 1;
    }
    assert!(ran > 0, "no frame captures found; run _out/run_hook.py");
}

/// `cl_target_cost` itself: our expression evaluator against the engine's, fed
/// the engine's own feature vector so only the arithmetic is under test.
#[test]
fn target_cost_expression_matches_the_engine_exactly() {
    let Some(v) = voice("William") else { return };
    let unp = v.unit_name_params().expect("unit_name_params");
    let tc = TargetCost::parse(v.image(), unp).expect("target_cost");
    let mut s = ceps::Selector::new(&v, ceps::SelectParams::from_voice(&v));

    let mut checked = 0usize;
    for (file, _) in CASES {
        let Some(rows) = capture(file) else { continue };
        let engine = engine_vectors(&rows, &tc.features);
        if engine.is_empty() {
            continue;
        }
        // tcost rows come in runs of one unit type per position, in the same
        // order as the feature vectors
        let mut groups: Vec<Vec<(usize, i64)>> = Vec::new();
        let mut g: Vec<(usize, i64)> = Vec::new();
        let mut gt: Option<u16> = None;
        for r in rows.iter().filter(|r| r.0 == "tcost") {
            let u = r.2.parse::<usize>().unwrap_or(0);
            if u >= v.num_units {
                continue;
            }
            let t = v.unit(u).type_id;
            if gt.is_some() && gt != Some(t) {
                groups.push(std::mem::take(&mut g));
            }
            gt = Some(t);
            g.push((u, r.5));
        }
        if !g.is_empty() {
            groups.push(g);
        }

        for (i, grp) in groups.iter().enumerate() {
            let Some(tv) = engine.get(i) else { continue };
            for &(u, want) in grp {
                assert_eq!(s.target_cost(tv, u) as i64, want,
                           "{file} position {i}: target cost of unit {u}");
                checked += 1;
            }
        }
    }
    assert!(checked > 10_000, "only {checked} target costs checked");
}
