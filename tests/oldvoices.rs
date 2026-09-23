//! Swift 4 and 5 voices, whose target cost differs from 6.2's.
//!
//! Needs CEPS_VOICE_ROOT pointing at a directory holding `oldwill-v4` and
//! `oldwill-v5`; skips silently otherwise.

use ceps::expr::{Expr, Op};
use ceps::{TargetCost, Voice};

fn voice(name: &str) -> Option<Voice> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let d = std::path::Path::new(&root).join(name);
    if d.join("voice_u.dat").exists() {
        Voice::open(d).ok()
    } else {
        None
    }
}

fn cost(v: &Voice) -> TargetCost {
    let unp = v.unit_name_params().expect("no unit_name_params");
    TargetCost::parse(v.image(), unp).expect("no target cost")
}

/// A distance matrix has to be symmetric and cost nothing to stay put. The zero
/// diagonal is also what lets `self_cost` reach 0 at all: with target and
/// candidate equal, every `phonedist` term indexes the diagonal.
#[test]
fn phonedist_is_symmetric_with_a_zero_diagonal() {
    let n = ceps::phonedist::N as i32;
    for a in 0..n {
        assert_eq!(ceps::phonedist::dist(a, a), 0.0, "diagonal at {a}");
        for b in 0..n {
            assert_eq!(
                ceps::phonedist::dist(a, b),
                ceps::phonedist::dist(b, a),
                "asymmetric at {a},{b}"
            );
        }
    }
    // out of range is the engine reading past its table; we return 0 instead
    assert_eq!(ceps::phonedist::dist(-1, 0), 0.0);
    assert_eq!(ceps::phonedist::dist(0, n), 0.0);
    // and it is not a table of zeroes
    assert!(ceps::phonedist::dist(0, 1) > 0.0);
}

/// `phonedist` is a registered two-argument function, not a feature, so it never
/// appears in `unit_features`. Parsed as a bare `cond` clause it silently
/// evaluated to its first argument -- the target's raw phone id -- and dropped
/// the second argument entirely.
#[test]
fn phonedist_parses_as_a_call_and_not_as_a_clause() {
    for name in ["oldwill-v4", "oldwill-v5"] {
        let Some(v) = voice(name) else { continue };
        let tc = cost(&v);
        assert!(
            !tc.missing.iter().any(|m| m == "phonedist"),
            "{name}: phonedist still parsed as a missing name"
        );

        let mut calls = 0;
        let mut walk = |e: &Expr| {
            let mut stack = vec![e];
            while let Some(x) = stack.pop() {
                match x {
                    Expr::Call(op, args) => {
                        if *op == Op::PhoneDist {
                            assert_eq!(args.len(), 2, "{name}: phonedist arity");
                            calls += 1;
                        }
                        stack.extend(args.iter());
                    }
                    Expr::Clause(t, val) => {
                        stack.push(t);
                        stack.push(val);
                    }
                    _ => {}
                }
            }
        };
        walk(&tc.expr);
        // one per neighbour: n, p, n.n, p.p
        assert_eq!(calls, 4, "{name}: expected four phonedist terms");
    }
}

/// The whole point: a unit scored against its own recorded features must cost
/// nothing, on the old voices as much as on 6.2.
#[test]
fn old_voices_score_a_unit_against_itself_at_zero() {
    let mut seen = 0;
    for name in ["oldwill-v4", "oldwill-v5"] {
        let Some(v) = voice(name) else { continue };
        let sc = cost(&v).self_cost(&v, 64);
        assert!(sc.probed > 0, "{name}: nothing probed");
        assert_eq!(sc.worst, 0, "{name}: self-cost {} over {} units", sc.worst, sc.probed);
        seen += 1;
    }
    if seen == 0 {
        eprintln!("no old voices found; set CEPS_VOICE_ROOT to run this");
    }
}

/// Our predicted durations against Swift 4.2's own, captured by hooking the
/// feature functions inside the VM (`_out/vm/hook_ff42.js`) and aligned to
/// phones by `_out/v42_align.py`.
///
/// This is the only check on the *target* side of the duration features:
/// `self_cost` scores a unit against itself, where the duration is the recorded
/// one, so it never exercises the tree at all.
///
/// The tolerance is not tight. Two of the duration tree's 22 features are
/// `accented` and `next_accent`, which need an intonation model we do not have,
/// so some phones take a different branch and land on a different z.
#[test]
fn predicted_durations_track_the_engines() {
    let Some(v) = voice("oldwill-v4") else { return };
    let root = std::env::var("CEPS_VOICE_ROOT").unwrap();
    let tsv = std::path::Path::new(&root).join("_out/v42cap/ff3.jsonl.tsv");
    let Ok(text) = std::fs::read_to_string(&tsv) else {
        eprintln!("no capture at {}; run _out/v42_align.py", tsv.display());
        return;
    };
    let want: Vec<(String, i32)> = text
        .lines()
        .skip(1)
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Some((f.first()?.to_string(), f.get(1)?.parse().ok()?))
        })
        .collect();
    assert!(want.len() > 50, "only {} captured phones", want.len());

    let sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "Thirty three thieves thought they thrilled the throne.",
    ];
    let cep = ceps::CepLex::open(
        &std::env::var("CEPS_CEPLEX").unwrap_or_else(|_| r"..\packs\ceplex.bin".into()),
    )
    .expect("ceplex pack");

    let mut ours: Vec<(String, i32)> = Vec::new();
    for s in sentences {
        let ph = ceps::text_to_phones_cepstral(&cep, s, ceps::DigitStyle::Cardinal).0;
        let nr = ceps::NameRules::parse(v.image(), v.unit_name_params().unwrap());
        let pos = ceps::target::plan(&nr, &ph);
        let nsyl = pos.iter().map(|p| p.syl).max().map(|m| m + 1).unwrap_or(0);
        let mut stress = vec![0i32; nsyl];
        for (i, p) in ph.iter().enumerate() {
            stress[pos[i].syl] = stress[pos[i].syl].max(p.stress as i32);
        }
        let m = ceps::dur::DurModel::open(&v).expect("dur model");
        for (i, p) in m.predict(&ph, &pos, &stress).iter().enumerate() {
            if ph[i].phone != "pau" {
                ours.push((ph[i].phone.clone(), p.durms));
            }
        }
    }

    assert_eq!(ours.len(), want.len(), "phone count: ours {:?}", &ours[..8.min(ours.len())]);
    let mut exact = 0;
    let mut close = 0;
    let mut worst = (0i32, String::new());
    let mut total = 0i64;
    for (k, (w, o)) in want.iter().zip(ours.iter()).enumerate() {
        assert_eq!(w.0, o.0, "phone {k}: engine {} ours {}", w.0, o.0);
        let e = (w.1 - o.1).abs();
        total += e as i64;
        if e == 0 {
            exact += 1;
        }
        if e <= 25 {
            close += 1;
        }
        if e > worst.0 {
            worst = (e, format!("{} engine {} ours {}", w.0, w.1, o.1));
        }
    }
    let n = want.len();
    let pct = 100.0 * close as f64 / n as f64;
    println!(
        "durms: {exact}/{n} exact, {close}/{n} within 25 ms ({pct:.0}%), \
mean error {:.1} ms, worst {} ({})",
        total as f64 / n as f64,
        worst.0,
        worst.1
    );
    assert!(pct >= 90.0, "only {pct:.0}% of predicted durations are within 25 ms");
}

/// 6.2 does not use `phonedist` at all, so adding it must leave that path alone.
#[test]
fn the_new_operator_does_not_disturb_a_62_voice() {
    let Some(v) = voice("William") else { return };
    let tc = cost(&v);
    assert!(!tc.missing.iter().any(|m| m == "phonedist"));
    assert_eq!(tc.self_cost(&v, 64).worst, 0);
}
