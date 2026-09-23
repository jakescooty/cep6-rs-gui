//! A regression lock on the LPC path, nothing more.
//!
//! These hashes only say the output has not changed since they were taken. The
//! claim that it matches the *engine* is made by `tests/engineparity.rs`, which
//! replays the engine's own emitted periods and compares against the wav the
//! engine produced from them. An earlier version of this file asserted
//! bit-exactness against a float32 reference we wrote ourselves, which locked in
//! a filter the engine does not use -- see the module comment in `src/synth.rs`.
//!
//! Regenerate with `target\release\ceps-hash.exe %CEPS_VOICE_ROOT%\<voice>`.
//!
//! Needs the voice data. Point CEPS_VOICE_ROOT at the directory holding the
//! per-voice folders; the test skips silently if it is unset.

use ceps::{synth, Voice};

fn fnv1a(pcm: &[i16]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &s in pcm {
        for b in s.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

type Case = (usize, usize, usize, u64); // first, count, samples, hash

const GOLDEN: &[(&str, &[Case])] = &[
    ("Allison", &[
        (0, 200, 20594, 0xb4f9178bfa60cd6e),
        (50_000, 120, 14522, 0x6ea4c9b264c27a3e),
        (400_000, 300, 31048, 0x21e77eb13cc17deb),
        (600_000, 120, 10584, 0x52cbff4bdc4bc3b2),
    ]),
    ("David", &[
        (0, 200, 47800, 0x231dcc6e35cf6ca7),
        (50_000, 120, 30238, 0xddd5678e343844de),
        (400_000, 300, 65242, 0x3f17e8be5afb065b),
        (600_000, 120, 30482, 0xc8b0565bc97c5851),
    ]),
    ("William", &[
        (0, 200, 32160, 0x32b27b069685ba4a),
        (50_000, 120, 20556, 0x8f989ca6b01015f5),
        (400_000, 300, 58126, 0xb13d38f4a71c27f2),
        (600_000, 120, 20866, 0x55f5ebdb9a4f79df),
    ]),
    // Jean-Pierre is the only one of the four at LPC order 14, so it is the only
    // one that takes the unrolled filter path. These moved when that path stopped
    // carrying the scratch slot in -- see `src/synth.rs`. The other three are
    // order 10 and their hashes are unchanged, which is the check that the fix
    // touched only what it should.
    ("Jean-Pierre", &[
        (0, 200, 26250, 0x79daabc3f33d51e9),
        (50_000, 120, 15014, 0x0c40cd6cb686b240),
        (400_000, 300, 36706, 0x1d4a9885faed166d),
        (600_000, 120, 15438, 0xa067d35f1e759ba1),
    ]),
];

#[test]
fn lpc_output_is_bit_exact() {
    let root = match std::env::var("CEPS_VOICE_ROOT") {
        Ok(r) => r,
        Err(_) => {
            eprintln!("CEPS_VOICE_ROOT unset, skipping");
            return;
        }
    };
    let mut checked = 0;
    for (name, cases) in GOLDEN {
        let dir = std::path::Path::new(&root).join(name);
        if !dir.join("voice_u.dat").exists() {
            continue;
        }
        let v = Voice::open(&dir).expect("open voice");
        for &(first, count, samples, hash) in *cases {
            let pcm = synth::synth_frames(&v, first, first + count);
            assert_eq!(pcm.len(), samples, "{name} frames {first}..+{count}: sample count");
            assert_eq!(fnv1a(&pcm), hash, "{name} frames {first}..+{count}: output changed");
            checked += 1;
        }
    }
    assert!(checked > 0, "no voices found under {root}");
    eprintln!("verified {checked} cases");
}

#[test]
fn catalogue_invariants() {
    let root = match std::env::var("CEPS_VOICE_ROOT") {
        Ok(r) => r,
        Err(_) => return,
    };
    for (name, _) in GOLDEN {
        let dir = std::path::Path::new(&root).join(name);
        if !dir.join("voice_u.dat").exists() {
            continue;
        }
        let v = Voice::open(&dir).unwrap();
        assert_eq!(
            v.types.iter().map(|t| t.count as usize).sum::<usize>(),
            v.num_units,
            "{name}: type counts must partition the unit table"
        );
        let mut linked = 0;
        for i in 0..v.num_units {
            let u = v.unit(i);
            if u.next == ceps::UNIT_NONE {
                continue;
            }
            let n = v.unit(u.next as usize);
            assert_eq!(u.end, n.start, "{name}: unit {i} end != next start");
            assert_eq!(n.prev, i as i32, "{name}: unit {i} next/prev not mutual");
            linked += 1;
        }
        assert!(linked > 0);
        eprintln!("{name}: {linked} linked unit pairs verified");
    }
}
