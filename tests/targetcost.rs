//! Cepstral's target cost. Needs CEPS_VOICE_ROOT; skips silently otherwise.

use ceps::db::UNIT_NONE;
use ceps::{SelectParams, Selector, TargetCost, Voice};

const VOICES: [&str; 4] = ["Allison", "David", "William", "Jean-Pierre"];

fn voice(name: &str) -> Option<Voice> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let d = std::path::Path::new(&root).join(name);
    if d.join("voice_u.dat").exists() {
        Voice::open(d).ok()
    } else {
        None
    }
}

fn tcost(v: &Voice) -> TargetCost {
    let unp = v.unit_name_params().expect("unit_name_params");
    TargetCost::parse(v.image(), unp).expect("target_cost parses")
}

/// unit_features names the columns of voice_d.dat, so the counts must agree.
#[test]
fn feature_list_matches_voice_d_width() {
    let mut seen = 0;
    for name in VOICES {
        let Some(v) = voice(name) else { continue };
        let tc = tcost(&v);
        assert_eq!(tc.features.len(), v.unit_feat_width,
                   "{name}: {} features vs voice_d.dat width {}",
                   tc.features.len(), v.unit_feat_width);
        assert_eq!(tc.features[0], "lisp_phone_nameid",
                   "{name}: column 0 should be the phone id");
        assert!(tc.beam_width > 0, "{name}: cand_beam_width missing");
        seen += 1;
    }
    assert!(seen > 0, "no voices found");
}

/// A unit scored against its own recorded features has every difference zero,
/// which must be the minimum the expression can produce.
#[test]
fn self_cost_is_zero_and_minimal() {
    for name in VOICES {
        let Some(v) = voice(name) else { continue };
        let tc = tcost(&v);
        let mut env = vec![0.0f32; tc.num_slots()];
        let mut compared = 0;
        for u in (0..v.num_units).step_by(v.num_units / 40 + 1) {
            let own: Vec<i32> = v.unit_feats(u).iter().map(|&x| x as i32).collect();
            let self_cost = tc.score(&mut env, &own, v.unit_feats(u));
            assert_eq!(self_cost, 0, "{name}: unit {u} self-cost {self_cost}");

            let t = v.unit(u).type_id as usize;
            let r = v.candidates(t);
            if r.len() < 2 {
                continue;
            }
            let other = if r.start == u { r.end - 1 } else { r.start };
            let other_cost = tc.score(&mut env, &own, v.unit_feats(other));
            assert!(other_cost >= 0, "{name}: negative target cost {other_cost}");
            compared += 1;
        }
        assert!(compared > 5, "{name}: only compared {compared} pairs");
    }
}

/// With real target vectors, `cand_beam_width` pruning should let a *narrow*
/// path beam recover a recorded chain -- and do far less work than the
/// placeholder needed at a wide beam.
#[test]
fn target_cost_recovers_a_chain_at_a_narrow_beam() {
    let Some(v) = voice("Allison") else { return };
    let mut truth = Vec::new();
    for u in 0..v.num_units {
        if v.unit(u).prev != UNIT_NONE {
            continue;
        }
        let c = v.chain(u, 62);
        if c.len() > truth.len() {
            truth = c;
        }
        if truth.len() >= 62 {
            break;
        }
    }
    assert!(truth.len() > 20);

    let types: Vec<usize> = truth.iter().map(|&u| v.unit(u).type_id as usize).collect();
    let tvecs: Vec<Vec<i32>> = truth.iter()
        .map(|&u| v.unit_feats(u).iter().map(|&x| x as i32).collect())
        .collect();

    let mut p = SelectParams::from_voice(&v);
    p.beam = 8;
    let mut s = Selector::new(&v, p);
    let got = s.select_with_targets(&types, Some(&tvecs));
    let hit = got.iter().zip(&truth).filter(|(g, &t)| g.unit == t).count();

    assert_eq!(hit, truth.len(), "recovered {hit}/{} at beam 8", truth.len());
    assert_eq!(s.last_score, 0, "the true chain costs zero on both terms");

    // the placeholder needs a far wider beam and much more work for the same result
    let mut p2 = SelectParams::from_voice(&v);
    p2.beam = 8;
    let mut s2 = Selector::new(&v, p2);
    s2.select(&types);
    assert!(s2.joins_evaluated > s.joins_evaluated * 3,
            "expected target-cost pruning to cut join work sharply: {} vs {}",
            s.joins_evaluated, s2.joins_evaluated);
}
