//! Swift's residual-excited LPC synthesis, sample-identical to the engine.
//!
//! Traced from swift.dll:
//!   0x2ab40  emit one pitch period: get_frame_size -> get_sts_frame ->
//!            get_sts_residual -> add_residual -> optional gain -> filter
//!   0x2a900  add_residual: centre crop or centre pad, on int16
//!   0x27350  coefficient dequantisation
//!   0x27440  the all-pole filter
//!   0x277b8  history save
//!
//! **The filter is integer, not float.** An earlier version of this file ran it
//! in f32 with the same operation order, which tracks the engine for a while and
//! then walks away from it: the filter is recursive, so a sub-ULP difference in
//! any coefficient compounds. Replaying the engine's own emitted periods gave
//! 34.7 dB SNR and only the first 1785 samples identical. In integers it is
//! sample-for-sample identical over a whole utterance.
//!
//! Four details carry the difference:
//!
//! * The coefficients are int32 in Q14, decoded by three truncating integer
//!   divisions (@0x27350), not by `raw * (1/65536) * range + min` in float.
//! * The filter history keeps the **full 32-bit** accumulator (@0x27581),
//!   while only its low 16 bits are emitted (@0x27590). A value that overflows
//!   an i16 stays intact in the recursion and wraps only on the way out.
//! * The scratch buffer is **persistent and never re-zeroed**, and the general
//!   filter path accumulates into it with `add dword ptr [r9], ecx` (@0x2776b)
//!   without clearing the slot first. So each sample's accumulator starts at the
//!   *previous period's* output for the same index. `sub_27810` allocates it
//!   once per utterance and stores it at `[ctx+0xc8]`; `sub_272f0` only frees it
//!   when that field is null, which it is not.
//!
//!   This is a bug in Cepstral's code and it is audible in their output, so it
//!   has to be reproduced. It is also why the error looked like a rounding
//!   difference for a long time: the carried-in value is usually small enough
//!   not to change `acc / 16384`, so the two agree for hundreds of samples and
//!   then differ by one. Measured on the engine's own filter calls, our
//!   accumulator was short by exactly the previous period's sample at that
//!   index, twice out of two.
//!
//!   **Only the general path does this.** The order-14 and order-12 paths are
//!   unrolled, and their first tap `mov`s where the general loop `add`s, so the
//!   slot's old contents are overwritten rather than carried in. 6.2's William
//!   is order 10 and takes the general loop; Swift 4 and 5 voices are order 14
//!   and do not.
//!
//!   Swift 4.2 is built the same way -- `0x47a13` is its 14-tap loop (taps -4
//!   through -0x38), `0x47bed` its 12-tap one, and `0x47d8f` the general nested
//!   loop whose `add ecx, [edx]` at `0x47deb` carries in from the first tap.
//!   5.2 matches. `_out/v42_imul_density.py` finds these without needing the
//!   RVAs: a linear sweep desynchronises on padding, but the `0f af` opcode
//!   does not move.
//!
//! * The carry-over stops at a **streamed piece boundary**. The engine commits a
//!   prefix of the utterance as soon as its whole search beam agrees on it and
//!   synthesises that prefix on its own, and `sub_29ba0`'s epilogue (0x29fbe,
//!   under `mode == 2`) calls `sub_27810` for each, which frees the scratch
//!   buffer and takes a zeroed one from `cst_safe_alloc`. `LpcState::new_piece`
//!   is that call; `Selector::last_pieces` says where the boundaries fall. The
//!   filter *history* at `[ctx+0xb8]` is allocated once in the constructor and
//!   is not touched, so it runs straight through a boundary.
//!
//! With all four, synthesis from text is sample-for-sample identical to the
//! engine's own output over whole sentences.

use crate::db::Voice;

/// Unity gain in the 1.15 fixed-point `local_rescale` used at 0x2abe2.
pub const GAIN_UNITY: i32 = 0x8000;

/// The filter's fixed-point shift: the coefficient sum is divided by 2^14
/// before the residual is added (@0x2757e).
const COEFF_SHIFT: u32 = 14;

pub struct LpcState {
    /// y[-order ..= y[-1]] as the filter keeps them: full int32, carried across
    /// frames and across units.
    hist: Vec<i32>,
    coeffs: Vec<i32>,
    raw: Vec<u16>,
    src: Vec<i16>,
    dst: Vec<i16>,
    work: Vec<i32>,
    pub out: Vec<i16>,
}

/// Integer division truncating toward zero, which is what `cdq` + `and` + `sar`
/// computes at 0x2737b, 0x27387 and 0x2757e. Rust's `/` already truncates
/// toward zero, so this is only here to name the intent.
#[inline]
fn div_trunc(x: i32, shift: u32) -> i32 {
    x / (1 << shift)
}

impl LpcState {
    pub fn new(v: &Voice) -> LpcState {
        LpcState {
            hist: vec![0; v.order],
            coeffs: vec![0; v.order],
            raw: vec![0; v.order],
            src: Vec::new(),
            dst: Vec::new(),
            work: Vec::new(),
            out: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.hist.iter_mut().for_each(|x| *x = 0);
        self.work.clear();
        self.out.clear();
    }

    /// Start a new streamed piece. `sub_27810` frees the scratch buffer and
    /// takes a fresh one from `cst_safe_alloc`, which zeroes, so the carry-over
    /// described above starts again from nothing. The filter *history* at
    /// `[ctx+0xb8]` is allocated once in the constructor and survives, so it is
    /// deliberately left alone here.
    pub fn new_piece(&mut self) {
        self.work.clear();
    }

    /// swift.dll @0x2a900. `targ` samples are produced from `src`.
    fn add_residual(src: &[i16], targ: usize, dst: &mut Vec<i16>) {
        dst.clear();
        dst.resize(targ, 0);
        let unit = src.len();
        if unit < targ {
            let off = (targ - unit) / 2;
            dst[off..off + unit].copy_from_slice(src);
        } else {
            let off = (unit - targ) / 2;
            dst.copy_from_slice(&src[off..off + targ]);
        }
    }

    /// swift.dll @0x27350. `raw` is the stored u16; everything else is int32.
    #[inline]
    fn dequantise(v: &Voice, raw: u16) -> i32 {
        let a = (raw as i32).wrapping_mul(v.coeff_range_scaled);
        let a = div_trunc(a, 1);
        let a = div_trunc(a, 10);
        div_trunc(a.wrapping_add(v.coeff_min_scaled), 1)
    }

    /// Emit one pitch period of `targ_size` samples using source frame
    /// `src_frame`. `gain` is 1.15 fixed point; pass `GAIN_UNITY` for none.
    pub fn emit_period(&mut self, v: &Voice, src_frame: usize, targ_size: usize, gain: i32) {
        v.residual(src_frame, &mut self.src);
        Self::add_residual(&self.src, targ_size, &mut self.dst);

        if gain != GAIN_UNITY {
            for s in self.dst.iter_mut() {
                // 0x2abe2: (x * g + (x<0 ? 0x7fff : 0)) >> 15, i.e. C's / 32768
                let x = *s as i32 * gain;
                let adj = if x < 0 { 0x7fff } else { 0 };
                *s = ((x + adj) >> 15) as i16;
            }
        }

        v.frame(src_frame, &mut self.raw);
        for k in 0..v.order {
            self.coeffs[k] = Self::dequantise(v, self.raw[k]);
        }

        let order = v.order;
        if self.work.len() < order + targ_size {
            self.work.resize(order + targ_size, 0);
        }
        self.work[..order].copy_from_slice(&self.hist);

        // Which filter path this order takes, and so whether the scratch slot
        // carries in. See the module comment.
        let unrolled = order == 14 || order == 12;

        for i in 0..targ_size {
            let o = order + i;
            // 32-bit wrapping throughout: the DLL's `imul r32` keeps the low
            // half and never checks, and voiced frames do overflow.
            // On the general path `acc` starts at whatever this slot held last
            // period, because the engine never clears it; the unrolled paths
            // store their first tap instead of adding it, so they start at 0.
            let mut acc: i32 = if unrolled { 0 } else { self.work[o] };
            for k in 0..order {
                acc = acc.wrapping_add(self.coeffs[k].wrapping_mul(self.work[o - 1 - k]));
            }
            acc = div_trunc(acc, COEFF_SHIFT).wrapping_add(self.dst[i] as i32);
            self.work[o] = acc;
            // 0x27590: `movzx eax, word ptr [...]` then a 16-bit store -- the
            // low half only, so the history keeps a value the output cannot
            self.out.push(acc as u16 as i16);
        }

        self.hist.copy_from_slice(&self.work[targ_size..targ_size + order]);
    }
}

/// Synthesise frames `[first, last)` at their recorded pitch periods, i.e.
/// Flite's `asis_to_pm` path with no prosodic modification.
pub fn synth_frames(v: &Voice, first: usize, last: usize) -> Vec<i16> {
    let mut st = LpcState::new(v);
    for i in first..last {
        let n = v.frame_size(i);
        st.emit_period(v, i, n, GAIN_UNITY);
    }
    st.out
}

/// Same, for a list of units.
pub fn synth_units(v: &Voice, units: &[usize]) -> Vec<i16> {
    let mut st = LpcState::new(v);
    for &u in units {
        let un = v.unit(u);
        for i in un.start as usize..un.end as usize {
            let n = v.frame_size(i);
            st.emit_period(v, i, n, GAIN_UNITY);
        }
    }
    st.out
}
