//! flat_prosody and the per-voice F0 estimate. Needs CEPS_VOICE_ROOT.

use ceps::f0::{DEFAULT_F0_MEAN, DEFAULT_F0_STDDEV};
use ceps::{natural_f0, FlatProsody, Voice};

fn voice(name: &str) -> Option<Voice> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let d = std::path::Path::new(&root).join(name);
    if d.join("voice_u.dat").exists() {
        Voice::open(d).ok()
    } else {
        None
    }
}

/// swift.dll @0x14830: two targets, start one stddev above the mean and end
/// one below, with the mean scaled by f0_shift.
#[test]
fn flat_prosody_is_a_two_point_declination() {
    let fp = FlatProsody::default();
    assert_eq!(fp.mean, DEFAULT_F0_MEAN);
    assert_eq!(fp.stddev, DEFAULT_F0_STDDEV);

    let t = fp.targets(2.5);
    assert_eq!(t.len(), 2);
    assert_eq!(t[0].pos, 0.0);
    assert_eq!(t[1].pos, 2.5);
    assert!((t[0].f0 - 112.0).abs() < 1e-4, "start {}", t[0].f0);
    assert!((t[1].f0 - 88.0).abs() < 1e-4, "end {}", t[1].f0);
    assert!(t[0].f0 > t[1].f0, "F0 must decline");

    let shifted = FlatProsody { shift: 2.0, ..FlatProsody::default() };
    let s = shifted.targets(1.0);
    assert!((s[0].f0 - 212.0).abs() < 1e-4, "shift scales the mean only: {}", s[0].f0);
}

/// Pitch periods in the database are the voice's recorded F0.
#[test]
fn natural_f0_is_in_a_human_range_and_sexes_split() {
    let mut female = 0.0f32;
    let mut males = Vec::new();
    for name in ["Allison", "David", "William", "Jean-Pierre"] {
        let Some(v) = voice(name) else { continue };
        let f = natural_f0(&v, 20_000);
        assert!((60.0..350.0).contains(&f), "{name}: {f} Hz");
        if name == "Allison" {
            female = f;
        } else if name != "Jean-Pierre" {
            males.push(f);
        }
    }
    if female > 0.0 && !males.is_empty() {
        let m = males.iter().sum::<f32>() / males.len() as f32;
        assert!(female > m * 1.5,
                "Allison ({female} Hz) should sit well above the male voices ({m} Hz)");
    }
}

/// All four voices ship `PROSODY "none"` and the SDK's own output confirms the
/// engine honours it: over ref_allison.wav the rendered F0 is 195 Hz median with
/// a 137-248 Hz spread against her recorded 204 Hz / 134-269 Hz, and over ref.wav
/// William renders 96 Hz / 73-124 Hz against his recorded 93 Hz / 60-129 Hz. A
/// flat_prosody declination would pin both into a narrow mean +/- stddev band, so
/// the natural spread is the proof. Nothing sets int_f0_target_mean because
/// nothing sets F0. flat_prosody stays reachable, but it is opt-in.
#[test]
fn flat_prosody_would_misfit_the_female_voice_if_it_were_used() {
    let Some(v) = voice("Allison") else { return };
    let nat = natural_f0(&v, 20_000);
    let ratio = nat / DEFAULT_F0_MEAN;
    assert!(ratio > 1.7, "expected a large mismatch, got {ratio:.2}x ({nat} Hz)");

    let tuned = FlatProsody::for_voice(&v);
    assert!((tuned.mean - nat).abs() < 1e-3);
    assert!((tuned.mean / nat - 1.0).abs() < 0.01, "for_voice must centre on the voice");

    if let Some(w) = voice("William") {
        let r = natural_f0(&w, 20_000) / DEFAULT_F0_MEAN;
        assert!((0.8..1.25).contains(&r),
                "the default should suit William, got {r:.2}x");
    }
}
