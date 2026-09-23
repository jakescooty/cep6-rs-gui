//! Print FNV-1a hashes of synthesised frame ranges, for the regression test.
//!
//!   ceps-hash <voicedir>

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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let v = Voice::open(&a[1]).expect("open voice");
    for &(first, count) in &[(0usize, 200usize), (50_000, 120), (400_000, 300), (600_000, 120)] {
        if first + count > v.num_sts {
            continue;
        }
        let pcm = synth::synth_frames(&v, first, first + count);
        println!("({first}, {count}, {}, 0x{:016x}),", pcm.len(), fnv1a(&pcm));
    }
}
