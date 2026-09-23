//! The `phonedist` matrix Swift 4 and 5 voices score context against.
//!
//! Unlike everything else the target cost reads, this is not voice data. Swift
//! registers it as a two-argument expression function:
//!
//! ```text
//! val_eval_register_func(ctx, "phonedist", 0x012e40, 2)
//!
//! float phonedist(float a, float b) {
//!     return (signed char) TABLE[(int)a * 122 + (int)b];
//! }
//! ```
//!
//! so a name that appears in no `unit_features` list is still a real term, and a
//! parser that treats `(phonedist a b)` as a bare `cond` clause silently returns
//! its first argument instead -- the target's raw phone id, around 50, which the
//! surrounding weights multiply into a six-figure constant.
//!
//! What it costs is the *neighbour* match: the v4 expression calls it on
//! `n.`, `p.`, `n.n.` and `p.p.lisp_phone_nameid`, comparing the phones around
//! the target with the phones that were around the candidate when it was
//! recorded. Stub it out and a wrong-context unit is free, so the search takes
//! whichever candidate is shortest and the voice runs fast.
//!
//! The table is symmetric with a zero diagonal and holds 11 distinct values from
//! 0 to 100. It came out of Swift 4.2's `swift.dll` at RVA `0xaac48`; Swift
//! 5.2's copy, at `0xb8050`, is **byte-identical**, so this one embedded table
//! serves both. Checked by `_out/v52_phonedist_cmp.py` rather than assumed --
//! a difference would not show up in [`crate::expr::SelfCost`], since a zero
//! diagonal makes any such table score a unit against itself at 0.

/// Phone ids index a fixed-size square; the engine does no bounds check.
pub const N: usize = 122;

static TABLE: &[u8; N * N] = include_bytes!("../packs/phonedist.bin");

/// `TABLE[a][b]` as the engine reads it: a signed byte, out-of-range as 0.
///
/// Swift truncates both arguments toward zero before indexing, which is what
/// `as i32` on an `f32` already does.
pub fn dist(a: i32, b: i32) -> f32 {
    if a < 0 || b < 0 || a as usize >= N || b as usize >= N {
        return 0.0;
    }
    TABLE[a as usize * N + b as usize] as i8 as f32
}
