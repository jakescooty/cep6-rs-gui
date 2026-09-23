//! Unit selection: candidate list, join cost, Viterbi.
//!
//! Transcribed from swift.dll rather than from flite. Cepstral's
//! `..\..\source\wavesynth\cst_clunits.c` (the path is in the DLL's own debug
//! strings) has diverged from stock flite enough that the differences decide
//! the answer, so every number below is either read from the voice or cited to
//! the instruction in swift.dll that holds it.
//!
//! Where the two disagree:
//!
//! * **There is no sliding optimal coupling.** flite's `optimal_couple` slides
//!   a window over `u0` and `u1->prev` looking for the cheapest frame pair, and
//!   returns `30000 + best`. Cepstral's frame-distance function @0x21930 has
//!   exactly one caller, inside @0x220a0, which passes `u0.end` and `u1.start`
//!   and nothing else. No search, no `30000`, and no `u0_move`/`u1_move`: units
//!   are used whole, which is also what the `emit_period` traces show.
//! * **The frame distance is scaled by the pair's channel-1 level**, @0x21b81.
//!   flite has no such factor. It averages ~72 on William, so leaving it out
//!   makes the join cost roughly seventy times too small against the target
//!   cost -- which is what the old hand-fitted `target_scale` of 0.1 was
//!   standing in for.
//! * **`continuity_weight` and `different_prev_pen` do not exist.** Neither
//!   string appears anywhere in swift.dll. The path score @0x235e9 is a plain
//!   integer sum: `target + join + previous`.
//! * **A short unit is penalised instead**, @0x21f00, and pause units join for
//!   free, @0x21860.
//! * **The beam keeps one path per unit and has a hole bug.** `sub_15ea0`'s
//!   sorted insert removes an older, worse path to the same unit and closes the
//!   gap by nulling the last slot, and the next insert takes that null slot
//!   before it compares scores. So the tenth-best path can be far worse than one
//!   the search was offered and rejected. `beam_insert` reproduces it, because
//!   the beam's exact contents decide where the engine streams -- see below.
//! * **The search streams.** `clunits_synth` @0x23780 asks `sub_160c0` after
//!   every step for the deepest node the whole beam still shares, and when that
//!   node moves it synthesises everything before it as its own utterance. That
//!   cannot change the chosen path, but each piece gets a fresh LPC scratch
//!   buffer, so it changes the samples. `Selector::last_pieces` carries the
//!   boundaries for `prosody::join_units_streamed`.
//!
//! Cepstral's target cost lives in `expr.rs` and is used by
//! `select_with_targets` when given per-position target feature vectors. With
//! no targets the selector falls back to flite's positional weight, which is
//! unusable on these voices -- see `TARGET_WEIGHT`.

use crate::db::{Voice, UNIT_NONE};
use crate::expr::TargetCost;
use crate::prosody::TargetUnit;
use crate::synth::GAIN_UNITY;

/// Flite's `clunits_target_weight` is 70, but that assumes a cluster CART has
/// already narrowed the candidates to a handful. These voices have degenerate
/// trees -- every unit of the type is a candidate -- so `70 * position` reaches
/// 476,490 for a type like `n` with 6807 units and swamps the join costs.
/// Measured on a 62-unit recorded chain: weight 70 recovers 0/62 units, weight
/// 0 recovers 62/62 at the optimal cost of zero. It is only reachable through
/// `select()` with no target vectors, which the engine never does.
pub const TARGET_WEIGHT: i32 = 0;

/// The Viterbi beam. swift.dll reads it in `clunits_synth` @0x23673 as
/// `voice/search_n_best` with a default of 10; it is a swift runtime parameter,
/// not a voice file field, so 10 is what every voice gets unless the host calls
/// `swift_port_set_param`. None of the four voices' `settings.txt` sets it.
pub const SEARCH_N_BEST: usize = 10;

/// Returned instead of a frame distance when `u0` ends on the last frame in the
/// database, so there is no following frame to measure against. swift.dll
/// @0x221cc.
const AT_END_OF_DB: i32 = 999;

/// The short-unit pair penalty, swift.dll @0x21f00. Both units longer than four
/// frames costs nothing; one of them four or fewer costs `SHORT_UNIT`; one of
/// them a single frame costs `VERY_SHORT_UNIT`. Compiled into the DLL, so the
/// same for every voice.
const SHORT_UNIT: i32 = 200_000;
const VERY_SHORT_UNIT: i32 = 400_000;
const SHORT_FRAMES: i32 = 4;

/// The frame distance's early-out ceiling, passed by @0x22270 and compared
/// against the accumulator scaled by 256 @0x2199d.
const JOIN_CEILING: i32 = 500_000;

/// Kept so the parity harness can price a path with the target cost turned up
/// or down. The engine has no such scale: `cl_target_cost` @0x21617 truncates
/// the expression to an int and `sub_21740` stores it as the candidate score
/// unchanged, and the path callback adds it to the join cost as-is.
pub const DEFAULT_TARGET_SCALE: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct SelectParams {
    pub optimal_coupling: i32,
    pub extend_selections: usize,
    /// Read from the voice and carried for completeness. swift.dll stores it at
    /// `clunit_db+0x48` (@0x449ee) and never reads it back: the frame distance
    /// @0x21930 takes the weights and the channel count and no F0 term at all.
    pub f0_weight: i32,
    pub join_weights: Vec<i32>,
    pub target_weight: i32,
    /// Surviving paths per position. 0 means keep every candidate.
    pub beam: usize,
    /// Cap on candidates examined per position. 0 means no cap. Ours, for
    /// benchmarking; the engine has no equivalent.
    pub max_candidates: usize,
    /// Diagnostic weight on the target cost. 1.0 is the engine.
    pub target_scale: f64,
}

impl SelectParams {
    pub fn from_voice(v: &Voice) -> SelectParams {
        let p = v.params();
        SelectParams {
            optimal_coupling: p.int("optimal_coupling").unwrap_or(0),
            extend_selections: p.int("extend_selections").unwrap_or(0).max(0) as usize,
            f0_weight: p.int("f0_weight").unwrap_or(0),
            join_weights: raw_join_weights(v),
            target_weight: TARGET_WEIGHT,
            beam: SEARCH_N_BEST,
            max_candidates: 0,
            target_scale: DEFAULT_TARGET_SCALE,
        }
    }
}

fn raw_join_weights(v: &Voice) -> Vec<i32> {
    // join_weights are stored as 16.16 fixed point; the cost functions want them raw
    v.join_weights.iter().map(|w| (w * 65536.0).round() as i32).collect()
}

#[derive(Debug, Clone, Copy)]
pub struct Choice {
    pub unit: usize,
    pub start: i32,
    pub end: i32,
}

/// The candidate list for one position, as swift.dll's `sub_21740` builds it.
///
/// Sorted descending by target cost, so the head is the *worst* candidate and
/// the trim drops it. The count lives on the head node and is reset to the beam
/// width after every trim (@0x21830), which leaves the list one entry shorter
/// than the beam: `cand_beam_width` 200 keeps 199 candidates. Modelled rather
/// than corrected, because it decides which of a type's 6807 units survive.
struct CandList {
    items: Vec<(i32, usize)>,
    stored: usize,
    beam: usize,
}

impl CandList {
    fn new(beam: usize) -> CandList {
        CandList { items: Vec::new(), stored: 0, beam }
    }

    fn contains(&self, unit: usize) -> bool {
        self.items.iter().any(|&(_, u)| u == unit)
    }

    fn push(&mut self, score: i32, unit: usize) {
        if self.beam == 0 {
            self.items.insert(0, (score, unit));
            return;
        }
        let at = self.items.partition_point(|&(s, _)| s > score);
        self.items.insert(at, (score, unit));
        self.stored += 1;
        if self.stored + 1 > self.beam {
            self.items.remove(0);
            self.stored = self.beam;
        }
    }
}

pub struct Selector<'a> {
    v: &'a Voice,
    pub p: SelectParams,
    /// Per type: does its name contain "pau"? swift.dll @0x21910 uses `strstr`,
    /// so `pauend` and `paustart` count as well as `pau`.
    pause_type: Vec<bool>,
    pub joins_evaluated: u64,
    /// Total path score of the last select() call.
    pub last_score: i64,
    /// Units per streamed piece from the last select() call. See
    /// `select_with_targets` for what decides the boundaries.
    pub last_pieces: Vec<usize>,
    /// The common-prefix depth after each search step, as `sub_160c0` reports
    /// it. Diagnostic: `last_pieces` is its increments.
    pub last_prefix: Vec<i64>,
    /// Per position, the surviving beam as (unit, score). Diagnostic only.
    pub last_beam: Vec<Vec<(usize, i64)>>,
    /// Per position, the candidate list in the order the search walks it:
    /// worst target cost first. Diagnostic only.
    pub last_cands: Vec<Vec<(usize, i32)>>,
    tcost: Option<TargetCost>,
    env: Vec<f32>,
}

impl<'a> Selector<'a> {
    pub fn new(v: &'a Voice, p: SelectParams) -> Selector<'a> {
        let pause_type = v.types.iter().map(|t| t.name.contains("pau")).collect();
        let tcost = v.unit_name_params().and_then(|o| TargetCost::parse(v.image(), o));
        Selector {
            v, p, pause_type, joins_evaluated: 0, last_score: 0,
            last_pieces: Vec::new(), last_prefix: Vec::new(), last_beam: Vec::new(),
            last_cands: Vec::new(), tcost, env: Vec::new(),
        }
    }

    #[inline]
    fn is_pause(&self, unit: usize) -> bool {
        self.pause_type[self.v.unit(unit).type_id as usize]
    }

    /// swift.dll @0x21930. Weighted Manhattan over the byte mel-cepstra, scaled
    /// by the two frames' channel-1 level.
    ///
    /// Both divisions truncate: `/ 4` on the channel-1 sum @0x21b81 and `/ 256`
    /// on the accumulator @0x21b89. The second is what our old
    /// `diff * 256 * w / 65536` reproduced. The first was missing, and channel 1
    /// runs 133..255 on William, so the join cost came out about seventy times
    /// too small.
    ///
    /// The early-out is the DLL's too: the unrolled nine-channel path tests the
    /// accumulator only after channel 0 (@0x2199d) and jumps straight to the
    /// return, so a bail-out still gets scaled. With c0's weight at 1.5 the
    /// ceiling is unreachable on these voices, but a partial sum is observable
    /// and the generic path below nine channels does test every channel.
    #[inline]
    fn frame_distance(&self, a: usize, b: usize) -> i32 {
        let av = self.v.mcep_raw(a);
        let bv = self.v.mcep_raw(b);
        let limit = JOIN_CEILING.saturating_mul(256);
        let mut r = 0i32;
        if av.len() == 9 {
            for i in 0..9 {
                r += (av[i] as i32 - bv[i] as i32).abs() * self.p.join_weights[i];
                if i == 0 && r >= limit {
                    break;
                }
            }
        } else {
            for i in 0..av.len() {
                r += (av[i] as i32 - bv[i] as i32).abs() * self.p.join_weights[i];
                if r > limit {
                    break;
                }
            }
        }
        let level = (av[1] as i32 + bv[1] as i32) / 4;
        level * (r / 256)
    }

    /// The whole join cost, as the Viterbi path callback assembles it @0x23528:
    /// `sub_220a0` for the frame distance and `sub_21f00` for the short-unit
    /// penalty, added as plain integers.
    ///
    /// `optimal_coupling` is 1 in all four voices, which takes both terms; 2
    /// takes the frame distance alone and 0 takes neither.
    pub fn join_cost(&mut self, u0: usize, u1: usize) -> i32 {
        if self.p.optimal_coupling != 1 && self.p.optimal_coupling != 2 {
            return 0;
        }
        if self.is_pause(u0) || self.is_pause(u1) {
            return 0;
        }
        let un1 = self.v.unit(u1);
        if un1.prev == u0 as i32 {
            return 0; // consecutive recordings join for free
        }
        let un0 = self.v.unit(u0);
        self.joins_evaluated += 1;
        let mut cost = if un0.end as usize >= self.v.num_sts || un1.start < 0 {
            AT_END_OF_DB
        } else {
            self.frame_distance(un0.end as usize, un1.start as usize)
        };
        if self.p.optimal_coupling == 1 {
            let l1 = un1.end - un1.start;
            let l0 = un0.end - un0.start;
            cost += if l1 > SHORT_FRAMES && l0 > SHORT_FRAMES {
                0
            } else if l1 > 1 && l0 > 1 {
                SHORT_UNIT
            } else {
                VERY_SHORT_UNIT
            };
        }
        cost
    }

    /// Cepstral's target cost for one candidate against a target feature vector.
    pub fn target_cost(&mut self, targ: &[i32], unit: usize) -> i32 {
        match &self.tcost {
            None => 0,
            Some(tc) => {
                if self.env.len() != tc.num_slots() {
                    self.env = vec![0.0; tc.num_slots()];
                }
                tc.score(&mut self.env, targ, self.v.unit_feats(unit))
            }
        }
    }

    /// The database units that naturally follow `prev`, where they are of the
    /// right type. swift.dll @0x22e25, gated on `extend_selections` (10 here).
    ///
    /// It runs *after* the candidate loop, on the already-pruned list, and each
    /// continuation goes through the same beam-limited insertion. That is the
    /// whole point of it: a unit that continues the previous one stays a
    /// candidate however poorly its target features score. The loop counts
    /// successful additions, not attempts (@0x23043).
    fn continuations(&self, type_id: usize, prev: &[usize]) -> Vec<usize> {
        let mut out = Vec::new();
        if self.p.extend_selections == 0 || prev.is_empty() {
            return out;
        }
        for &lc in prev {
            if out.len() >= self.p.extend_selections {
                break;
            }
            let nu = self.v.unit(lc).next;
            if nu == UNIT_NONE || nu < 0 || nu as usize >= self.v.num_units {
                continue;
            }
            let nu = nu as usize;
            if self.v.unit(nu).type_id as usize == type_id && !out.contains(&nu) {
                out.push(nu);
            }
        }
        out
    }

    /// Total cost of a fixed unit path under this selector's own cost function.
    ///
    /// The point is diagnostic. Price the engine's actual choices with our cost
    /// and compare against what our search returned: if the engine's path scores
    /// *lower*, our search is leaving something on the table and the beam or the
    /// pruning is at fault; if it scores *higher*, our search is doing its job and
    /// the cost function itself is what differs. Those need opposite fixes, and
    /// nothing else distinguishes them.
    pub fn path_cost(&mut self, path: &[usize], targets: Option<&[Vec<i32>]>) -> (i64, i64, i64) {
        let mut join = 0i64;
        let mut targ = 0i64;
        for (i, &u) in path.iter().enumerate() {
            if let Some(t) = targets.and_then(|t| t.get(i)) {
                targ += self.target_cost(t, u) as i64;
            }
            if i > 0 {
                join += self.join_cost(path[i - 1], u) as i64;
            }
        }
        (join + targ, join, targ)
    }

    pub fn select(&mut self, type_seq: &[usize]) -> Vec<Choice> {
        self.select_with_targets(type_seq, None)
    }

    /// `targets[i]` is the target feature vector for position `i`, in the order
    /// given by `unit_features`. When supplied, Cepstral's target cost replaces
    /// the placeholder and `cand_beam_width` prunes the candidate list.
    pub fn select_with_targets(
        &mut self,
        type_seq: &[usize],
        targets: Option<&[Vec<i32>]>,
    ) -> Vec<Choice> {
        #[derive(Clone)]
        struct Path {
            score: i64,
            unit: usize,
            from: usize,
        }

        /// swift.dll @0x15ea0, the sorted insert into a point's path array.
        ///
        /// Two things a straight "keep the n cheapest" would not do. It holds
        /// **one path per unit**: a better path for a unit already in the list
        /// removes the older one and closes the gap, and a worse path for a unit
        /// already present is dropped where it stands (@0x15f3c). And the gap it
        /// closes is filled with a hole at the end (@0x15ffa), which the next
        /// insert takes unconditionally at @0x15f00 -- *before* comparing
        /// scores. So a path far worse than everything in the list can land in
        /// the last slot and stay there. That is how the engine's tenth-best
        /// path at position 3 of the parity sentence costs 777024 when a 257568
        /// one was offered and rejected.
        fn beam_insert(arr: &mut Vec<Option<Path>>, np: Path) {
            let nb = arr.len();
            for i in 0..nb {
                let (score, unit) = match &arr[i] {
                    Some(p) => (p.score, p.unit),
                    None => {
                        arr[i] = Some(np);
                        return;
                    }
                };
                if np.score < score {
                    let id = np.unit;
                    arr.pop();
                    arr.insert(i, Some(np));
                    let mut j = i + 1;
                    while j < nb {
                        if arr[j].as_ref().is_some_and(|p| p.unit == id) {
                            arr.remove(j);
                            arr.push(None);
                        }
                        // no second look at the entry that just shifted into j:
                        // the engine's loop increments regardless
                        j += 1;
                    }
                    return;
                }
                if unit == np.unit {
                    return;
                }
            }
        }

        let cand_beam = self.tcost.as_ref().map(|t| t.beam_width).unwrap_or(0);
        let nb = if self.p.beam > 0 { self.p.beam } else { 1 };
        let mut lattice: Vec<Vec<Option<Path>>> = Vec::with_capacity(type_seq.len());
        let mut prev_units: Vec<usize> = Vec::new();
        let mut pieces: Vec<usize> = Vec::new();
        let mut prefix: Vec<i64> = Vec::new();
        let mut beams: Vec<Vec<(usize, i64)>> = Vec::new();
        let mut cands: Vec<Vec<(usize, i32)>> = Vec::new();
        let mut emitted: i64 = 0;

        for (pos, &t) in type_seq.iter().enumerate() {
            let targ = targets.and_then(|ts| ts.get(pos));
            let r = self.v.candidates(t);
            let step = if self.p.max_candidates > 0 && r.len() > self.p.max_candidates {
                r.len() / self.p.max_candidates + 1
            } else {
                1
            };

            let mut list = CandList::new(if targ.is_some() { cand_beam } else { 0 });
            for c in r.clone().step_by(step) {
                let score = targ.map(|tv| self.target_cost(tv, c)).unwrap_or(0);
                list.push(score, c);
            }
            for nu in self.continuations(t, &prev_units) {
                if !list.contains(nu) {
                    let score = targ.map(|tv| self.target_cost(tv, nu)).unwrap_or(0);
                    list.push(score, nu);
                }
            }

            cands.push(list.items.iter().map(|&(s, u)| (u, s)).collect());

            let scale = self.p.target_scale;
            let tw = self.p.target_weight;
            let scored = targ.is_some();
            let target_of = move |tc: i32, k: usize| -> i64 {
                if scored {
                    (tc as f64 * scale) as i64
                } else {
                    (tw * (k as i32 + 1)) as i64
                }
            };

            // The first point is the exception: `sub_15ea0` sends it down the
            // per-candidate branch at 0x16032, where the array has one slot per
            // candidate indexed by the candidate's own position, not n_best
            // slots ordered by score. In practice an utterance opens on
            // `paustart`, which has a single candidate, so the two agree.
            let mut layer: Vec<Option<Path>> = if pos == 0 {
                list.items.iter().enumerate()
                    .map(|(k, &(tc, c))| {
                        Some(Path { score: target_of(tc, k), unit: c, from: usize::MAX })
                    })
                    .collect()
            } else {
                vec![None; nb]
            };

            if pos > 0 {
                for (pi, prev) in lattice[pos - 1].iter().enumerate() {
                    let (pscore, punit) = match prev {
                        Some(p) => (p.score, p.unit),
                        None => continue,
                    };
                    // @0x1648d: a predecessor already worse than the last slot
                    // cannot produce anything better, so it is skipped whole.
                    if let Some(w) = layer[nb - 1].as_ref() {
                        if pscore >= w.score {
                            continue;
                        }
                    }
                    for (k, &(tc, c)) in list.items.iter().enumerate() {
                        let target = target_of(tc, k);
                        // @0x1650f: the same test on the target cost alone,
                        // which is a lower bound on the extended path.
                        if let Some(w) = layer[nb - 1].as_ref() {
                            if pscore + target >= w.score {
                                continue;
                            }
                        }
                        let s = pscore + self.join_cost(punit, c) as i64 + target;
                        beam_insert(&mut layer, Path { score: s, unit: c, from: pi });
                    }
                }
            }

            prev_units = layer.iter().flatten().map(|p| p.unit).collect();
            beams.push(layer.iter().flatten().map(|p| (p.unit, p.score)).collect());
            lattice.push(layer);

            // The engine streams. After every search step `clunits_synth`
            // (@0x23780) asks `sub_160c0` for the deepest node all surviving
            // paths still share, and when that node moves it emits everything
            // up to it as a separate utterance: its own Unit relation, its own
            // `join_units` call, its own LPC scratch buffer. The chosen path is
            // unchanged -- committing a prefix the whole beam agrees on cannot
            // alter the answer -- but the filter's carry-over is reset at each
            // boundary, so the samples differ. `sub_160c0` walks all live heads
            // back in lockstep until every one names the same candidate.
            let mut heads: Vec<Option<usize>> = (0..lattice[pos].len())
                .map(|k| lattice[pos][k].as_ref().map(|_| k))
                .collect();
            let mut level = pos;
            let mut agree: i64 = -1;
            loop {
                let mut live = heads.iter().flatten();
                match live.next() {
                    Some(&first) if live.all(|&k| k == first) => {
                        agree = level as i64;
                        break;
                    }
                    Some(_) => {}
                    None => break,
                }
                if level == 0 {
                    break;
                }
                for h in heads.iter_mut() {
                    *h = h.and_then(|k| {
                        let f = lattice[level][k].as_ref().unwrap().from;
                        if f == usize::MAX { None } else { Some(f) }
                    });
                }
                level -= 1;
            }
            prefix.push(agree);
            // The commit node itself is not emitted: `sub_20010` takes the
            // previous commit and the new one and fills in what lies between,
            // so the first fourteen steps -- which all agree only on the very
            // first unit -- emit nothing at all.
            if agree > emitted {
                pieces.push((agree - emitted) as usize);
                emitted = agree;
            }
        }

        // The final flush at 0x239f1 passes the best complete path in place of
        // the beam, so everything the commit point never reached goes out here.
        if emitted < lattice.len() as i64 {
            pieces.push((lattice.len() as i64 - emitted) as usize);
        }
        self.last_pieces = pieces;
        self.last_prefix = prefix;
        self.last_beam = beams;
        self.last_cands = cands;

        // @0x239a5 asks the search for its best complete path rather than
        // assuming slot 0: the hole-filling above can leave the array unsorted.
        let last = match lattice.last() {
            Some(l) => l,
            None => return Vec::new(),
        };
        let mut idx = 0usize;
        let mut best = i64::MAX;
        for (i, p) in last.iter().enumerate() {
            if let Some(p) = p {
                if p.score < best {
                    best = p.score;
                    idx = i;
                }
            }
        }
        self.last_score = if best == i64::MAX { 0 } else { best };

        let mut chain: Vec<usize> = Vec::with_capacity(lattice.len());
        for pos in (0..lattice.len()).rev() {
            let p = lattice[pos][idx].as_ref().unwrap();
            chain.push(p.unit);
            if p.from == usize::MAX {
                break;
            }
            idx = p.from;
        }
        chain.reverse();

        // no optimal-coupling move exists, so every unit is used whole
        chain
            .into_iter()
            .map(|u| {
                let x = self.v.unit(u);
                Choice { unit: u, start: x.start, end: x.end }
            })
            .collect()
    }
}

/// Convenience: select by type name, then build TargetUnits with given segment
/// end times in seconds.
pub fn select_to_targets(
    v: &Voice,
    p: SelectParams,
    types: &[usize],
    ends: &[f32],
) -> (Vec<Choice>, Vec<TargetUnit>) {
    let mut s = Selector::new(v, p);
    let chosen = s.select(types);
    let tu = chosen
        .iter()
        .zip(ends)
        .map(|(c, &e)| TargetUnit {
            start: c.start,
            end: c.end,
            target_end: (e * v.sps as f32) as i32,
            gain: GAIN_UNITY,
        })
        .collect();
    (chosen, tu)
}
