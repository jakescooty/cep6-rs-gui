//! Behaviour of f0_targets_to_pm / nearest_pm / concat_units.
//! Needs CEPS_VOICE_ROOT; skips silently otherwise.

use ceps::prosody::{concat_units, f0_targets_to_pm, nearest_pm, unit_size, DEFAULT_LEAD_F0};
use ceps::{synth, F0Target, TargetUnit, Voice, GAIN_UNITY};

fn voice() -> Option<Voice> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    for n in ["Allison", "David", "William", "Jean-Pierre"] {
        let d = std::path::Path::new(&root).join(n);
        if d.join("voice_u.dat").exists() {
            return Voice::open(d).ok();
        }
    }
    None
}

fn flat_targets(secs: f32, f0: f32) -> Vec<F0Target> {
    vec![F0Target { pos: 0.0, f0 }, F0Target { pos: secs, f0 }]
}

#[test]
fn flat_f0_gives_the_requested_period_count() {
    let Some(v) = voice() else { return };
    for f0 in [100.0f32, 150.0, 220.0] {
        let secs = 2.0f32;
        let pms = f0_targets_to_pm(&v, &flat_targets(secs, f0), DEFAULT_LEAD_F0);
        let want = (secs * f0) as usize;
        let got = pms.len();
        assert!(
            got.abs_diff(want) <= 2,
            "f0 {f0}: expected ~{want} pitch marks over {secs}s, got {got}"
        );
        // marks must be monotonic and land inside the span
        assert!(pms.windows(2).all(|w| w[1] >= w[0]), "pitch marks not monotonic");
        let last = *pms.last().unwrap() as f32 / v.sps as f32;
        assert!((last - secs).abs() < 0.05, "last mark at {last}s, expected ~{secs}s");
    }
}

#[test]
fn rising_f0_packs_marks_more_tightly_over_time() {
    let Some(v) = voice() else { return };
    let targets = vec![
        F0Target { pos: 0.0, f0: 100.0 },
        F0Target { pos: 1.0, f0: 300.0 },
    ];
    let pms = f0_targets_to_pm(&v, &targets, 100.0);
    assert!(pms.len() > 150, "expected many marks, got {}", pms.len());
    let gaps: Vec<i32> = pms.windows(2).map(|w| w[1] - w[0]).collect();
    let first = gaps[..10].iter().sum::<i32>() / 10;
    let last = gaps[gaps.len() - 10..].iter().sum::<i32>() / 10;
    assert!(last < first / 2, "gaps should shrink: first {first}, last {last}");
}

#[test]
fn nearest_pm_walks_the_unit() {
    let Some(v) = voice() else { return };
    let u = v.unit(5000);
    let total = unit_size(&v, u.start, u.end);
    assert_eq!(nearest_pm(&v, u.start, u.end, 0), u.start, "index 0 -> first frame");
    assert_eq!(
        nearest_pm(&v, u.start, u.end, total * 10),
        u.end - 1,
        "index past the end -> last frame"
    );
    // monotonic in u_index
    let mut prev = u.start;
    for k in 0..=20 {
        let j = nearest_pm(&v, u.start, u.end, total * k / 20);
        assert!(j >= prev, "nearest_pm went backwards: {prev} then {j}");
        prev = j;
    }
}

#[test]
fn concat_covers_every_pitch_mark_exactly_once() {
    let Some(v) = voice() else { return };
    let secs = 1.0f32;
    let pms = f0_targets_to_pm(&v, &flat_targets(secs, 180.0), DEFAULT_LEAD_F0);
    let n = 8;
    let units: Vec<TargetUnit> = (0..n)
        .map(|i| {
            let u = v.unit(1000 + i);
            TargetUnit {
                start: u.start,
                end: u.end,
                target_end: ((i + 1) as f32 / n as f32 * secs * v.sps as f32) as i32,
                gain: GAIN_UNITY,
            }
        })
        .collect();

    let periods = concat_units(&v, &units, &pms);
    assert_eq!(periods.len(), pms.len(), "one period per pitch mark");

    let total: usize = periods.iter().map(|p| p.size).sum();
    assert_eq!(
        total as i32,
        *pms.last().unwrap(),
        "period sizes must tile up to the last pitch mark"
    );

    // every source frame must lie inside the unit it came from
    let mut k = 0;
    for u in &units {
        while k < periods.len() && (periods[k].src_frame as i32) >= u.start
            && (periods[k].src_frame as i32) < u.end
        {
            k += 1;
        }
    }
    assert_eq!(k, periods.len(), "a period referenced a frame outside its unit");

    let pcm = ceps::synth_periods(&v, &periods);
    assert_eq!(pcm.len(), total, "output length must equal the sum of period sizes");
}

#[test]
fn simple_join_matches_the_asis_path() {
    let Some(v) = voice() else { return };
    let ids = [2000usize, 2001, 2002, 7777];
    let units: Vec<TargetUnit> = ids
        .iter()
        .map(|&i| {
            let u = v.unit(i);
            TargetUnit { start: u.start, end: u.end, target_end: 0, gain: GAIN_UNITY }
        })
        .collect();
    let a = ceps::join_units_simple(&v, &units);
    let b = synth::synth_units(&v, &ids);
    assert_eq!(a, b, "join_units_simple must equal the unmodified unit path");
}
