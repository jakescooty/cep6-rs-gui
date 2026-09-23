//! Unit selection behaviour. Needs CEPS_VOICE_ROOT; skips silently otherwise.

use ceps::db::UNIT_NONE;
use ceps::{SelectParams, Selector, Voice};

fn voice(name: &str) -> Option<Voice> {
    let root = std::env::var("CEPS_VOICE_ROOT").ok()?;
    let d = std::path::Path::new(&root).join(name);
    if d.join("voice_u.dat").exists() {
        Voice::open(d).ok()
    } else {
        None
    }
}

fn longest_chain(v: &Voice, cap: usize) -> Vec<usize> {
    let mut best = Vec::new();
    for u in 0..v.num_units {
        if v.unit(u).prev != UNIT_NONE {
            continue;
        }
        let c = v.chain(u, cap);
        if c.len() > best.len() {
            best = c;
        }
        if best.len() >= cap {
            break;
        }
    }
    best
}

/// A recorded chain joins at zero cost, so selection given only its type
/// sequence must find its way back to it. This exercises the join cost,
/// optimal coupling, candidate extension and the Viterbi together.
#[test]
fn recovers_a_recorded_chain() {
    let Some(v) = voice("Allison") else { return };
    let truth = longest_chain(&v, 62);
    assert!(truth.len() > 20, "need a decent chain, got {}", truth.len());
    let types: Vec<usize> = truth.iter().map(|&u| v.unit(u).type_id as usize).collect();

    let mut p = SelectParams::from_voice(&v);
    p.beam = 32;
    let mut s = Selector::new(&v, p);
    let got = s.select(&types);

    let hit = got.iter().zip(&truth).filter(|(g, &t)| g.unit == t).count();
    assert_eq!(hit, truth.len(), "recovered only {hit}/{}", truth.len());
    assert_eq!(s.last_score, 0, "the recorded chain costs zero to join");
}

/// Join costs are non-negative, so a wider beam can never find a worse path.
/// This is the regression test for the i32 overflow in frame_distanceb, which
/// produced negative costs and made beam=128 *worse* than beam=32.
#[test]
fn wider_beam_never_costs_more() {
    let Some(v) = voice("Allison") else { return };
    let truth = longest_chain(&v, 24);
    let types: Vec<usize> = truth.iter().map(|&u| v.unit(u).type_id as usize).collect();

    let mut last = i64::MAX;
    for beam in [4usize, 16, 64] {
        let mut p = SelectParams::from_voice(&v);
        p.beam = beam;
        let mut s = Selector::new(&v, p);
        s.select(&types);
        assert!(s.last_score >= 0, "beam {beam}: negative path cost {}", s.last_score);
        assert!(s.last_score <= last, "beam {beam} scored {} vs {last} for a narrower beam",
                s.last_score);
        last = s.last_score;
    }
}

/// Direct check on the cost function: the c0 join weight is 1.5 in 16.16, and
/// byte mceps differ by up to 255, so the naive `abs(diff)*w` in i32 wraps.
#[test]
fn join_cost_is_never_negative() {
    for name in ["Allison", "David", "William", "Jean-Pierre"] {
        let Some(v) = voice(name) else { continue };
        let p = SelectParams::from_voice(&v);
        let mut s = Selector::new(&v, p);
        // deterministic spread over the catalogue, including the extremes
        let n = v.num_units;
        let mut checked = 0;
        for k in 0..400 {
            let a = (k * 7919) % n;
            let b = (k * 104_729 + 12345) % n;
            let cost = s.join_cost(a, b);
            assert!(cost >= 0, "{name}: join_cost({a},{b}) = {cost}");
            checked += 1;
        }
        assert!(checked > 0);
    }
}

/// Consecutive recordings must join for free -- this is what `extend_selections`
/// and the whole contiguity mechanism rely on.
#[test]
fn consecutive_units_join_for_free() {
    let Some(v) = voice("Allison") else { return };
    let p = SelectParams::from_voice(&v);
    let mut s = Selector::new(&v, p);
    let mut checked = 0;
    for u in (0..v.num_units).step_by(997) {
        let n = v.unit(u).next;
        if n == UNIT_NONE {
            continue;
        }
        let cost = s.join_cost(u, n as usize);
        assert_eq!(cost, 0, "unit {u} -> its own successor {n} should cost 0");
        checked += 1;
    }
    assert!(checked > 10, "only checked {checked} pairs");
}

/// The engine adds the target cost to the join cost unscaled.
///
/// `cl_target_cost` @0x21617 truncates the expression to an int, `sub_21740`
/// stores it as the candidate score untouched, and the path callback @0x235e9
/// computes `target + join + previous` in integers. Neither `continuity_weight`
/// nor `different_prev_pen` appears anywhere in swift.dll. An earlier default of
/// 0.1 here was standing in for the channel-1 factor missing from the join cost,
/// which is the thing to fix rather than to weight around.
#[test]
fn the_target_cost_is_added_unscaled() {
    assert_eq!(ceps::select::DEFAULT_TARGET_SCALE, 1.0);
    let Some(v) = voice("William") else { return };
    assert_eq!(ceps::SelectParams::from_voice(&v).target_scale, 1.0);
}

/// The frame distance is scaled by the pair's channel-1 level.
///
/// swift.dll @0x21b81 multiplies the weighted Manhattan sum by
/// `(a[1] + b[1]) / 4` before returning it. Nothing in flite does this, and
/// leaving it out is what made the join cost too small to compete with the
/// target cost: channel 1 runs 133..255 on William, so the factor averages
/// around 70.
#[test]
fn the_join_cost_carries_the_channel_one_factor() {
    let Some(v) = voice("William") else { return };
    let p = SelectParams::from_voice(&v);
    let w: Vec<i64> = v.join_weights.iter().map(|x| (x * 65536.0).round() as i64).collect();
    let mut s = Selector::new(&v, p);

    let mut checked = 0;
    for k in 0..200usize {
        let a = (k * 7919) % v.num_units;
        let b = (k * 104_729 + 12345) % v.num_units;
        let cost = s.join_cost(a, b);
        let (ua, ub) = (v.unit(a), v.unit(b));
        if cost == 0 || ub.prev == a as i32 || ua.end as usize >= v.num_sts {
            continue;
        }
        let (fa, fb) = (ua.end as usize, ub.start as usize);
        let (ma, mb) = (v.mcep_raw(fa), v.mcep_raw(fb));
        let raw: i64 = (0..ma.len())
            .map(|i| (ma[i] as i64 - mb[i] as i64).abs() * w[i])
            .sum();
        let level = (ma[1] as i64 + mb[1] as i64) / 4;
        let pen = cost as i64 - level * (raw / 256);
        assert!(pen == 0 || pen == 200_000 || pen == 400_000,
                "join_cost({a},{b}) = {cost}, expected {} plus a short-unit penalty, got {pen}",
                level * (raw / 256));
        assert!(level > 20, "channel 1 should be a real level, got {level}");
        checked += 1;
    }
    assert!(checked > 20, "only checked {checked} pairs");
}

/// `extend_selections` exists to survive pruning, not to widen an unpruned list.
///
/// The full candidate list is already every unit of the type, so adding
/// continuations before `cand_beam_width` adds nothing -- the original code's
/// containment test could never fire. The unit that needs putting back is the
/// one whose target cost ranks below the beam, and doing so moved consecutive
/// joins on a captured sentence from 18/51 to 25/51 against the engine's 26/51.
#[test]
fn extend_selections_survives_the_candidate_prune() {
    let Some(v) = voice("William") else { return };
    let p = ceps::SelectParams::from_voice(&v);
    assert_eq!(p.extend_selections, 10,
               "the voice data sets extend_selections; read it, do not default it");
    assert_eq!(p.optimal_coupling, 1);
    assert_eq!(p.f0_weight, 100);

    // a chain of consecutive database units must be recoverable from its own
    // type sequence even when the target cost is switched off entirely
    let start = (0..v.num_units)
        .find(|&u| {
            let x = v.unit(u);
            x.end > x.start && x.next > 0 && (x.next as usize) < v.num_units
        })
        .expect("a unit with a successor");
    let mut chain = vec![start];
    for _ in 0..5 {
        let n = v.unit(*chain.last().unwrap()).next;
        if n <= 0 || (n as usize) >= v.num_units {
            break;
        }
        chain.push(n as usize);
    }
    if chain.len() < 4 {
        return;
    }
    let types: Vec<usize> = chain.iter().map(|&u| v.unit(u).type_id as usize).collect();

    let mut sp = ceps::SelectParams::from_voice(&v);
    sp.target_scale = 0.0;
    let mut sel = ceps::Selector::new(&v, sp);
    let got = sel.select(&types);
    let consec = got
        .windows(2)
        .filter(|w| v.unit(w[1].unit).start == v.unit(w[0].unit).end)
        .count();
    assert!(consec + 1 >= chain.len() / 2,
            "a recorded chain should come back mostly consecutive: {consec}/{}",
            chain.len() - 1);
}
