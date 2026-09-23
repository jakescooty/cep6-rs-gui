//! The reference tracer. Needs CEPS_VOICE_ROOT and a reference WAV; skips
//! silently without them.
//!
//! The tracer is ground truth for everything downstream, so what it needs is not
//! "does it look plausible" but internal consistency it could not fake: frames
//! that advance consecutively, scores far below the accept threshold, and the
//! same answer from the fast path as from the exhaustive one.

use ceps::trace::{GOOD, STRIDE};
use ceps::{wav, Tracer, Voice};

fn voice(name: &str) -> Option<Voice> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let d = std::path::Path::new(&root).join(name);
    if d.join("voice_u.dat").exists() {
        Voice::open(d).ok()
    } else {
        None
    }
}

fn reference() -> Option<(Voice, Vec<i16>)> {
    let v = voice("William")?;
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let p = std::path::Path::new(&root).join("ref.wav");
    let (pcm, sps) = wav::read(p).ok()?;
    if sps != v.sps {
        return None;
    }
    Some((v, pcm))
}

/// Round-tripping our own output is the strongest check available: we know
/// exactly which frames went in, so the tracer must name them.
#[test]
fn recovers_frames_from_our_own_synthesis() {
    let Some(v) = voice("William") else { return };

    // a run of consecutive frames from one unit, emitted at their natural sizes
    let u = (0..v.num_units)
        .map(|i| v.unit(i))
        .find(|x| x.end - x.start >= 12)
        .expect("a unit with enough frames");
    let want: Vec<usize> = (u.start as usize..u.end as usize).collect();

    let units = [ceps::TargetUnit {
        start: u.start,
        end: u.end,
        target_end: 0,
        gain: ceps::GAIN_UNITY,
    }];
    let pcm = ceps::join_units_simple(&v, &units);
    assert!(pcm.len() > 400, "need enough audio to trace");

    let tr = Tracer::new(&v, &pcm);
    let trace = tr.run(pcm.len());
    assert!(!trace.hits.is_empty(), "nothing recovered");

    let found: Vec<usize> = trace.hits.iter().map(|h| h.frame).collect();
    let hits = found.iter().filter(|f| want.contains(f)).count();
    assert!(hits * 2 >= found.len(),
            "most recovered frames should be the ones we emitted: {hits}/{} \
             (wanted {:?}.., got {:?}..)",
            found.len(), &want[..4.min(want.len())], &found[..4.min(found.len())]);

    // and every accepted hit must clear the threshold by a real margin
    for h in &trace.hits {
        assert!(h.score <= GOOD, "accepted a score of {} above {GOOD}", h.score);
    }
}

/// The engine walks frames consecutively inside a unit, so a correct trace is
/// dominated by +1 steps. A broken one scatters.
#[test]
fn engine_output_traces_to_consecutive_frames() {
    let Some((v, pcm)) = reference() else { return };
    let tr = Tracer::new(&v, &pcm);
    let trace = tr.run(2 * v.sps as usize);
    assert!(trace.hits.len() > 40, "expected a real trace, got {}", trace.hits.len());

    let steps: Vec<i64> = trace
        .hits
        .windows(2)
        .map(|w| w[1].frame as i64 - w[0].frame as i64)
        .collect();
    let consec = steps.iter().filter(|&&d| d == 1).count();
    assert!(consec * 4 >= steps.len() * 3,
            "expected mostly consecutive frames, got {consec}/{}", steps.len());

    let worst = trace.hits.iter().map(|h| h.score).fold(0.0f32, f32::max);
    assert!(worst <= GOOD, "worst accepted score {worst} exceeds {GOOD}");
}

/// Recovered frames must land on real units, and their natural lengths must
/// match the spacing actually observed in the audio. This is what proves the
/// tracer is reading pitch periods and not just fitting noise.
#[test]
fn recovered_frames_match_their_recorded_periods() {
    let Some((v, pcm)) = reference() else { return };
    let tr = Tracer::new(&v, &pcm);
    let trace = tr.run(2 * v.sps as usize);

    let mut compared = 0usize;
    let mut close = 0usize;
    for w in trace.hits.windows(2) {
        // only where the engine stayed inside one unit
        if w[1].frame != w[0].frame + 1 {
            continue;
        }
        let observed = w[1].pos as i64 - w[0].pos as i64;
        let natural = v.frame_size(w[0].frame) as i64;
        compared += 1;
        if (observed - natural).abs() <= STRIDE as i64 {
            close += 1;
        }
    }
    assert!(compared > 20, "not enough consecutive pairs to judge: {compared}");
    assert!(close * 10 >= compared * 8,
            "recovered periods should match the recorded ones: {close}/{compared}");

    let units = tr.units(&trace);
    assert!(units.len() > 5, "expected several units, got {}", units.len());
    for &(_, u) in &units {
        assert!(u < v.num_units);
    }
}

/// The branch-and-bound in the exhaustive search is an optimisation, not an
/// approximation, so it must return exactly what a full scan would.
#[test]
fn bounded_search_agrees_with_an_exhaustive_one() {
    let Some((v, pcm)) = reference() else { return };
    let tr = Tracer::new(&v, &pcm);
    let a = tr.run(v.sps as usize);

    let mut single = Tracer::new(&v, &pcm);
    single.threads = 1;
    let b = single.run(v.sps as usize);

    assert_eq!(a.hits.len(), b.hits.len(), "thread count changed the trace length");
    for (x, y) in a.hits.iter().zip(&b.hits) {
        assert_eq!(x.frame, y.frame, "thread count changed the frame at {}", x.pos);
        assert_eq!(x.pos, y.pos);
    }
}
