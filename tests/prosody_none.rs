//! `PROSODY "none"` is literal: the engine applies no F0 model, and units play
//! back at their recorded pitch periods. Needs CEPS_VOICE_ROOT.
//!
//! Established from the SDK's own output for one fixed sentence. Rendered F0,
//! against each voice's recorded F0 measured from the pitch-period index:
//!
//!     Allison   195.1 Hz median, 137-248 Hz p05-p95   (recorded 204.2, 134-269)
//!     William    95.8 Hz median,  73-124 Hz p05-p95   (recorded  93.0,  60-129)
//!
//! The means alone are suggestive; the spread is the proof. A flat_prosody
//! declination pins output into a narrow mean +/- stddev band, so a 111 Hz-wide
//! rendered spread cannot have come from one. Nothing sets int_f0_target_mean
//! because nothing sets F0 at all.

use ceps::prosody::unit_size;
use ceps::{concat_units, f0_targets_to_pm, join_units_simple, F0Target, TargetUnit, Voice,
           GAIN_UNITY};

fn voice(name: &str) -> Option<Voice> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let d = std::path::Path::new(&root).join(name);
    if d.join("voice_u.dat").exists() {
        Voice::open(d).ok()
    } else {
        None
    }
}

/// A run of consecutive units, long enough to cover a few hundred periods.
fn chain(v: &Voice, want_frames: usize) -> Vec<TargetUnit> {
    let mut out = Vec::new();
    let mut frames = 0usize;
    for i in 0..v.num_units {
        let u = v.unit(i);
        if u.end <= u.start {
            continue;
        }
        frames += (u.end - u.start) as usize;
        out.push(TargetUnit {
            start: u.start,
            end: u.end,
            target_end: 0,
            gain: GAIN_UNITY,
        });
        if frames >= want_frames {
            break;
        }
    }
    let sps = v.sps as f32;
    let mut acc = 0.0f32;
    for t in out.iter_mut() {
        acc += unit_size(v, t.start, t.end) as f32 / sps;
        t.target_end = (acc * sps) as i32;
    }
    out
}

fn spread(sizes: &[usize]) -> (f64, f64) {
    let n = sizes.len() as f64;
    let mean = sizes.iter().map(|&s| s as f64).sum::<f64>() / n;
    let var = sizes.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// The natural path emits every frame at its own recorded length, so the period
/// distribution is the database's. Forcing a flat contour flattens it, and that
/// flattening is audible as the monotone the SDK never produces.
#[test]
fn natural_playback_keeps_the_recorded_f0_spread() {
    let Some(v) = voice("Allison") else { return };
    let units = chain(&v, 400);

    let natural: Vec<usize> = units
        .iter()
        .flat_map(|u| (u.start..u.end).map(|i| v.frame_size(i as usize)))
        .collect();
    assert!(natural.len() > 200, "need a real chain, got {}", natural.len());

    let secs = natural.iter().sum::<usize>() as f32 / v.sps as f32;
    let f0 = 204.0f32;
    let pms = f0_targets_to_pm(&v, &[F0Target { pos: 0.0, f0 },
                                     F0Target { pos: secs, f0 }], f0);
    let forced: Vec<usize> = concat_units(&v, &units, &pms).iter().map(|p| p.size).collect();
    assert!(!forced.is_empty());

    let (nm, nsd) = spread(&natural);
    let (fm, fsd) = spread(&forced);

    // both centre on the same pitch; only the variation differs
    assert!((nm / fm - 1.0).abs() < 0.35,
            "means should be comparable: natural {nm:.1} vs forced {fm:.1}");
    assert!(fsd < 2.0, "a flat contour must give near-constant periods, sd {fsd:.2}");
    assert!(nsd > 8.0 * fsd.max(0.5),
            "recorded periods must vary far more than a forced contour: \
             natural sd {nsd:.2}, forced sd {fsd:.2}");
}

/// Natural playback never inserts or drops a sample: output length is exactly
/// the sum of the recorded frame lengths. The forced path pads, which is where
/// the inserted silence in the robotic renders came from.
#[test]
fn natural_playback_inserts_no_silence() {
    for name in ["Allison", "William"] {
        let Some(v) = voice(name) else { continue };
        let units = chain(&v, 200);
        let want: usize = units
            .iter()
            .flat_map(|u| (u.start..u.end).map(|i| v.frame_size(i as usize)))
            .sum();
        let pcm = join_units_simple(&v, &units);
        assert_eq!(pcm.len(), want, "{name}: natural playback must be sample-for-sample");
    }
}

/// Every voice ships PROSODY "none", so the opt-in flat path must never become
/// the default. This pins the relationship the CLI relies on.
#[test]
fn recorded_f0_is_voice_specific_and_far_from_the_flat_default() {
    let Some(a) = voice("Allison") else { return };
    let Some(w) = voice("William") else { return };
    let fa = ceps::natural_f0(&a, 40_000);
    let fw = ceps::natural_f0(&w, 40_000);
    assert!((180.0..230.0).contains(&fa), "Allison recorded F0 {fa}");
    assert!((80.0..110.0).contains(&fw), "William recorded F0 {fw}");
    assert!(fa > fw * 1.8, "the two voices must not share one default");
}
