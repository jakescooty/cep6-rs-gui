//! Duration model: dur_cart + dur_stats. Needs CEPS_VOICE_ROOT.

use ceps::cart::segment_duration;
use ceps::{Cart, DurStats, FeatVal, Voice};

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
fn dur_stats_are_plausible_phone_durations() {
    let mut seen = 0;
    for name in ["Allison", "David", "William", "Jean-Pierre"] {
        let Some(v) = voice(name) else { continue };
        let ds = DurStats::parse(v.image(), v.dur_stats().expect("dur_stats"), 256);
        assert!(ds.stats.len() >= 20, "{name}: only {} phones", ds.stats.len());
        for s in &ds.stats {
            assert!(s.mean > 0.005 && s.mean < 0.6, "{name}: {} mean {}", s.phone, s.mean);
            assert!(s.stddev >= 0.0 && s.stddev < 0.5, "{name}: {} sd {}", s.phone, s.stddev);
        }
        assert!(ds.get("pau").is_some(), "{name}: no pau entry");
        seen += 1;
    }
    assert!(seen > 0);
}

/// The 41 entries in dur_stats are the phone inventory, which is also the
/// number of distinct lisp_phone_nameid values in voice_d.dat column 0.
#[test]
fn dur_stats_size_matches_the_phone_inventory() {
    let Some(v) = voice("Allison") else { return };
    let ds = DurStats::parse(v.image(), v.dur_stats().unwrap(), 256);
    let mut ids: Vec<u8> = (0..v.num_units).map(|u| v.unit_feats(u)[0]).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ds.stats.len(), ids.len(),
               "dur_stats has {} phones, voice_d.dat column 0 has {} distinct ids",
               ds.stats.len(), ids.len());
}

#[test]
fn dur_cart_parses_and_classifies() {
    let Some(v) = voice("Allison") else { return };
    let c = Cart::parse(v.image(), v.dur_cart().expect("dur_cart"));
    assert!(c.num_nodes > 1000, "only {} nodes", c.num_nodes);
    assert!(c.feats.iter().any(|f| f == "name"), "feature table missing 'name': {:?}", c.feats);
    assert!(c.feats.iter().any(|f| f == "R:SylStructure.parent.stress"));

    let ds = DurStats::parse(v.image(), v.dur_stats().unwrap(), 256);
    let ask = |phone: &str, pbreak: &str| {
        let f = |n: &str| -> FeatVal {
            match n {
                "name" => FeatVal::Str(phone.into()),
                "p.R:SylStructure.parent.parent.pbreak" => FeatVal::Str(pbreak.into()),
                "R:SylStructure.parent.stress" => FeatVal::Str("0".into()),
                "R:SylStructure.parent.syl_break" => FeatVal::Str("0".into()),
                "n.name" | "p.name" => FeatVal::Str("ax".into()),
                _ => FeatVal::Num(0.0),
            }
        };
        c.interpret(&f).as_num()
    };

    // every phone must reach a finite leaf and a sane duration
    for s in &ds.stats {
        let z = ask(&s.phone, "NB");
        assert!(z.is_finite(), "{}: non-finite z-score", s.phone);
        let d = segment_duration(z, s, 1.0);
        assert!(d > 0.0 && d < 1.0, "{}: duration {d} s (z {z})", s.phone);
    }

    // a pause after a major break must be longer than one after none
    let pau = ds.get("pau").unwrap();
    let big = segment_duration(ask("pau", "BB"), pau, 1.0);
    let none = segment_duration(ask("pau", "NB"), pau, 1.0);
    assert!(big > none, "pause after BB ({big}) should exceed after NB ({none})");
}

#[test]
fn duration_stretch_scales_linearly() {
    let Some(v) = voice("Allison") else { return };
    let ds = DurStats::parse(v.image(), v.dur_stats().unwrap(), 256);
    let s = ds.get("pau").unwrap();
    let a = segment_duration(0.5, s, 1.0);
    let b = segment_duration(0.5, s, 2.0);
    assert!((b - 2.0 * a).abs() < 1e-6, "{a} then {b}");
}
