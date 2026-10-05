//! Deterministic predicate search maximizing a cross-validated, KL-anchored objective.
//!
//! Discovery (slice 00) is split by ROW PARITY into a TRAIN half (even rows) and a
//! VALIDATION half (odd rows):
//!
//!   * token sets are seeded from the weight's top-KL positions on the TRAIN half (unsigned
//!     KL mass, not signed |dce|) and may only name token ids that occur at least twice among
//!     those positions, so a set can never be a one-off coincidence of the discovery text;
//!   * greedy growth and the removal pass maximise the TRAIN objective; every candidate is
//!     reported with both halves and the final ordering is by the VALIDATION objective;
//!   * the objective is
//!     `min(clip(rec), 1 - clip(inv), kl_recall) - lambda * complexity - mu * cov`,
//!     where rec/inv are the signed local-effect ratios of the half and `kl_recall` is the
//!     fraction of the weight's KL mass (profile `kl.npy`) the predicate covers, and the
//!     `mu * cov` term prices the fired positions themselves.  Without it the first three
//!     terms saturate at ~1 for any predicate covering the weight's dce/KL mass, and since
//!     kl_recall is monotone in coverage the *broadest* candidate under the 0.5 cap wins
//!     (measured: near-cap `punct_only` windows outranking `digit_only` for the paper's
//!     neuron-1863 weights).  mu defaults to 0.5: half the objective range per unit of
//!     coverage, i.e. a candidate that fires on 40% of the slice must be ~0.2 better than a
//!     tight one to win, while a 2.9%-coverage predicate pays only 0.015.
//!     Clipping alone does NOT stop sign-splitting: a set that captures only the positive-dce
//!     positions still clips to 1.  `kl_recall` is the real anchor -- the predicate must fire
//!     where the weight actually matters (its KL mass), which split sets cannot fake.
//!   * a candidate that is a strict superset of an equally scoring (by the lambda-free
//!     objective) candidate is dropped: firing wherever an equally good predicate fires, and
//!     more, is redundant.
//!
//! Search families (CONTRACT grammar, no LLM):
//!   (a) greedy token set at every offset 0..-8, seeded from top-KL ids (count >= 2);
//!   (b) the best single top-KL id at every offset;
//!   (c) every vocab class at every offset (in particular all of 0..-3);
//!   (d) every window class over the (lo, hi) grid (-8 <= lo < hi <= -1), masses accumulated
//!       incrementally and the chain cut as soon as coverage passes 0.5;
//!   (d') the two best greedy sets applied over every window (win_in);
//!   (e) AND of pairs among the best ~10 diverse clauses, (f) OR of the best two.
//!
//! Only the validation half is accumulated during generation (the train/full metrics are
//! reported for the emitted top-M, computed from their fired sets afterwards).  Coverage is
//! over ALL slice positions (precomputed bitsets).  Candidates must satisfy 0 < cov <= 0.5
//! and a positive validation objective; the top-M distinct-by-spec are kept (ties: score_val
//! desc, cov asc, complexity asc, canonical key asc).

use crate::bitset::Bits;
use crate::index::{off_index, PosIndex, NOFF};
use crate::spec::Spec;
use crate::vocab::Vocab;
use std::cmp::Ordering;

/// Default complexity penalty: the objective lives in [0, 1] and interpretability needs
/// score >= 0.75, so 0.01 per extra named token costs at most 0.1 for a 10-id set.  That is
/// enough to prefer a class/window (or a shorter token set) over an equal-quality longer one,
/// yet small enough that the KL-recall anchor always dominates.  Raise it to trade recall for
/// fewer named ids; it is off the critical path of generalization (kl_recall does that work).
pub const DEFAULT_LAMBDA: f64 = 0.01;

/// Default coverage penalty (per unit of `cov`, see the module docs).
pub const DEFAULT_COV_PENALTY: f64 = 0.5;

/// Objective tolerance for "same objective" in the domination prune: two predicates that
/// differ by less than this are interchangeable, so the tighter one wins.
const M_TOL: f64 = 1e-9;

#[derive(Clone)]
pub struct Cand {
    pub spec: Spec,
    pub key: String,
    /// number of distinct decoded token strings the predicate names (class/window = 1)
    pub size: usize,
    pub rec: f64,
    pub inv: f64,
    pub cov: f64,
    /// full-slice CONTRACT proxy `min(rec, 1 - inv)` (kept for compatibility)
    pub score: f64,
    pub score_train: f64,
    pub score_val: f64,
    pub kl_recall_val: f64,
    /// number of fired positions over the whole slice (cov * n_pos, exact)
    pub cnt: u64,
    /// lambda- and mu-free quality `min(clip(rec), 1 - clip(inv), kl_recall)`
    pub m: f64,
    /// fingerprint of the fired-position set (diversifies the AND/OR pool)
    pub hash: u64,
}

/// Validation-half masses of one candidate: position count over ALL slice positions plus the
/// captured dce / KL mass over the VALIDATION half's kept positions.
#[derive(Clone, Copy, Default)]
struct ValMass {
    cnt: u64,
    dce: f64,
    kl: f64,
}

/// Both halves, for the emitted candidates' reported metrics.
#[derive(Clone, Copy, Default)]
struct Masses {
    cnt: u64,
    dce_tr: f64,
    dce_val: f64,
    kl_tr: f64,
    kl_val: f64,
}

/// Slice-wide half denominators + degenerate-half fallbacks.
struct Ctx {
    lambda: f64,
    mu: f64,
    n_pos: usize,
    dce_total: f64,
    tot_tr: f64,
    tot_val: f64,
    ok_tr: bool,
    ok_val: bool,
    kl_all: f64,
    kl_tr: f64,
    kl_val: f64,
    kl_ok_tr: bool,
    kl_ok_val: bool,
}

impl Ctx {
    /// (rec, inv) of a half; when the half's signed dce denominator collapses (all the mass
    /// sits in the other half, or the two signs cancel) fall back to the full-slice ratio.
    fn rec_inv(&self, in_dce: f64, val: bool) -> (f64, f64) {
        let (tot, ok) = if val { (self.tot_val, self.ok_val) } else { (self.tot_tr, self.ok_tr) };
        let rec = if ok { in_dce / tot } else { in_dce / self.dce_total };
        (rec, 1.0 - rec)
    }

    /// KL-mass recall of a half; a weight with no KL mass at all constrains nothing (1.0),
    /// and a half with no KL mass falls back to the full-slice recall.
    fn kl_recall(&self, in_kl: f64, val: bool) -> f64 {
        if self.kl_all <= 0.0 {
            return 1.0;
        }
        let (tot, ok) = if val { (self.kl_val, self.kl_ok_val) } else { (self.kl_tr, self.kl_ok_tr) };
        if ok {
            in_kl / tot
        } else {
            in_kl / self.kl_all
        }
    }

    fn objective_tr(&self, in_dce: f64, in_kl: f64, size: usize, cov: f64) -> f64 {
        let (rec, inv) = self.rec_inv(in_dce, false);
        objective(rec, inv, self.kl_recall(in_kl, false), size, cov, self.lambda, self.mu)
    }
}

/// `min(clip(rec), 1 - clip(inv), kl_recall) - lambda * complexity - mu * cov`.
fn objective(rec: f64, inv: f64, kl_recall: f64, size: usize, cov: f64, lambda: f64,
             mu: f64) -> f64 {
    let a = rec.clamp(0.0, 1.0);
    let b = 1.0 - inv.clamp(0.0, 1.0);
    a.min(b).min(kl_recall) - lambda * size as f64 - mu * cov
}

/// Lambda-free, mu-free quality `min(clip(rec), 1 - clip(inv), kl_recall)`: the domination
/// prune and the AND/OR pool ordering compare *quality*, not the priced objective.
fn quality(rec: f64, inv: f64, kl_recall: f64) -> f64 {
    let a = rec.clamp(0.0, 1.0);
    let b = 1.0 - inv.clamp(0.0, 1.0);
    a.min(b).min(kl_recall)
}

/// Complexity: the number of distinct decoded token strings a spec names.  Decoded-string
/// equivalence groups therefore count once, which is exactly what the emitted Python shows
/// (`_set_lit` dedupes reprs), and a class or a window counts 1 regardless of its span.
fn complexity(spec: &Spec, vocab: &Vocab) -> usize {
    match spec {
        Spec::TokIn { ids, .. } | Spec::WinIn { ids, .. } => {
            let mut v: Vec<&str> = ids
                .iter()
                .filter(|i| (**i as usize) < vocab.tokens.len())
                .map(|i| vocab.tokens[*i as usize].as_str())
                .collect();
            v.sort_unstable();
            v.dedup();
            v.len()
        }
        Spec::Cls { .. } | Spec::WinCls { .. } => 1,
        Spec::And(a, b) | Spec::Or(a, b) => complexity(a, vocab) + complexity(b, vocab),
    }
}

/// FNV-1a of one bitset word (folded into the mass loops so no separate pass is needed).
#[inline]
fn hash_word(h: &mut u64, w: u64) {
    *h ^= w;
    *h = h.wrapping_mul(0x100000001b3);
}

/// `b` strictly inside `a` (b subset of a, b != a) given the two popcounts.
fn strict_subset(b: &Bits, a: &Bits, cnt_b: u64, cnt_a: u64) -> bool {
    if cnt_b >= cnt_a {
        return false;
    }
    for i in 0..a.w.len() {
        if b.w[i] & !a.w[i] != 0 {
            return false;
        }
    }
    true
}

/// Validation masses of a fired set.
fn val_masses(b: &Bits, kept_val: &Bits, dense: &[[f32; 2]]) -> ValMass {
    let mut m = ValMass::default();
    for i in 0..b.w.len() {
        let w = b.w[i];
        m.cnt += w.count_ones() as u64;
        let mut x = w & kept_val.w[i];
        while x != 0 {
            let j = x.trailing_zeros() as usize;
            let v = dense[(i << 6) + j];
            m.dce += v[0] as f64;
            m.kl += v[1] as f64;
            x &= x - 1;
        }
    }
    m
}

/// FNV-1a over the bitset words: a cheap fingerprint of a clause's fired set, used to keep
/// the AND/OR pool diverse.  Computed lazily for the top candidates only.
fn bits_hash(b: &Bits) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for w in &b.w {
        hash_word(&mut h, *w);
    }
    h
}

/// Both-half masses of a fired set (only for the emitted candidates).
fn full_masses(b: &Bits, kept_tr: &Bits, kept_val: &Bits, dense: &[[f32; 2]]) -> Masses {
    let mut m = Masses::default();
    for i in 0..b.w.len() {
        let w = b.w[i];
        m.cnt += w.count_ones() as u64;
        let mut x = w & kept_tr.w[i];
        while x != 0 {
            let j = x.trailing_zeros() as usize;
            let v = dense[(i << 6) + j];
            m.dce_tr += v[0] as f64;
            m.kl_tr += v[1] as f64;
            x &= x - 1;
        }
        let mut x = w & kept_val.w[i];
        while x != 0 {
            let j = x.trailing_zeros() as usize;
            let v = dense[(i << 6) + j];
            m.dce_val += v[0] as f64;
            m.kl_val += v[1] as f64;
            x &= x - 1;
        }
    }
    m
}

fn make_cand(spec: Spec, vm: ValMass, ctx: &Ctx, vocab: &Vocab) -> Cand {
    let cov = vm.cnt as f64 / ctx.n_pos as f64;
    let size = complexity(&spec, vocab);
    let (rec_val, inv_val) = ctx.rec_inv(vm.dce, true);
    let kl_val = ctx.kl_recall(vm.kl, true);
    Cand {
        spec,
        key: String::new(),
        size,
        rec: 0.0,
        inv: 0.0,
        cov,
        score: 0.0,
        score_train: 0.0,
        score_val: objective(rec_val, inv_val, kl_val, size, cov, ctx.lambda, ctx.mu),
        kl_recall_val: kl_val,
        cnt: vm.cnt,
        m: quality(rec_val, inv_val, kl_val),
        hash: 0,
    }
}

/// Fill the reported full-slice / train metrics of an emitted candidate.
fn fill_reported(c: &mut Cand, b: &Bits, kept_tr: &Bits, kept_val: &Bits, dense: &[[f32; 2]],
                 ctx: &Ctx) {
    let m = full_masses(b, kept_tr, kept_val, dense);
    let in_all = m.dce_tr + m.dce_val;
    c.rec = in_all / ctx.dce_total;
    c.inv = 1.0 - c.rec;
    c.score = c.rec.min(1.0 - c.inv);
    let (rec_tr, inv_tr) = ctx.rec_inv(m.dce_tr, false);
    c.score_train = objective(rec_tr, inv_tr, ctx.kl_recall(m.kl_tr, false), c.size, c.cov,
                              ctx.lambda, ctx.mu);
}

/// Final ordering: validation objective desc, then the tightest coverage, then the simplest
/// spec, then the canonical key.
fn rank_final(a: &Cand, b: &Cand) -> Ordering {
    b.score_val
        .partial_cmp(&a.score_val)
        .unwrap_or(Ordering::Equal)
        .then(a.cov.partial_cmp(&b.cov).unwrap_or(Ordering::Equal))
        .then(a.size.cmp(&b.size))
        .then(a.key.cmp(&b.key))
}

/// Domination ordering: lambda-free validation objective desc, then the tightest coverage
/// (so a tight predicate is seen before the broad predicates it dominates).
fn rank_dom(a: &Cand, b: &Cand) -> Ordering {
    b.m.partial_cmp(&a.m)
        .unwrap_or(Ordering::Equal)
        .then(a.cov.partial_cmp(&b.cov).unwrap_or(Ordering::Equal))
        .then(a.size.cmp(&b.size))
        .then(a.key.cmp(&b.key))
}

#[allow(clippy::too_many_arguments)]
pub fn search_weight(
    idx: &PosIndex,
    vocab: &Vocab,
    top_m: usize,
    lambda: f64,
    mu: f64,
    dce_total: f64,
    dce_rest: f64,
    pos: &[i32],
    dce: &[f32],
    kl: &[f32],
) -> Vec<Cand> {
    let n_pos = idx.n_pos;
    if dce_total.abs() < 1e-6 * n_pos as f64 {
        return Vec::new();
    }
    let n = pos.len();
    let t = idx.t;
    let n_seq = n_pos / t;

    // ---- kept-position dense views + the train/validation split ----------------------
    // dce and kl interleaved: one cache line touch per fired kept position.
    let mut dense = vec![[0f32; 2]; n_pos];
    let mut kept = Bits::new(n_pos);
    let mut train = Bits::new(n_pos);
    let mut n_tr = 0usize;
    let mut n_val = 0usize;
    for r in 0..n_seq {
        for c in 0..t {
            let p = r * t + c;
            if r % 2 == 0 {
                train.set(p);
                n_tr += 1;
            } else {
                n_val += 1;
            }
        }
    }
    let mut kd_all = 0f64;
    let mut kd_tr = 0f64;
    let mut kk_all = 0f64;
    let mut kk_tr = 0f64;
    for i in 0..n {
        let p = pos[i] as usize;
        if p >= n_pos {
            continue;
        }
        let d = dce[i];
        let k = if kl[i] > 0.0 { kl[i] } else { 0.0 };
        dense[p] = [d, k];
        kept.set(p);
        kd_all += d as f64;
        kk_all += k as f64;
        if train.get(p) {
            kd_tr += d as f64;
            kk_tr += k as f64;
        }
    }
    let mut kept_tr = kept.clone();
    kept_tr.and_with(&train);
    let mut kept_val = kept.clone();
    for i in 0..kept_val.w.len() {
        kept_val.w[i] &= !train.w[i];
    }
    let n_kept_tr = kept_tr.count() as usize;
    let n_kept_val = kept_val.count() as usize;
    let n_rest = (n_tr - n_kept_tr) + (n_val - n_kept_val);
    // positions outside the kept set carry no per-position dce/kl; split the shard's
    // `dce_rest` between the halves in proportion to their non-kept position counts.
    let rest_tr = if n_rest > 0 {
        dce_rest * ((n_tr - n_kept_tr) as f64) / n_rest as f64
    } else {
        0.0
    };
    let kd_val = kd_all - kd_tr;
    let ctx = Ctx {
        lambda,
        mu,
        n_pos,
        dce_total,
        tot_tr: kd_tr + rest_tr,
        tot_val: kd_val + (dce_rest - rest_tr),
        ok_tr: (kd_tr + rest_tr).abs() > 1e-6 * dce_total.abs(),
        ok_val: (kd_val + (dce_rest - rest_tr)).abs() > 1e-6 * dce_total.abs(),
        kl_all: kk_all,
        kl_tr: kk_tr,
        kl_val: kk_all - kk_tr,
        kl_ok_tr: kk_all > 0.0 && kk_tr > 1e-9 * kk_all,
        kl_ok_val: kk_all > 0.0 && (kk_all - kk_tr) > 1e-9 * kk_all,
    };

    // ---- per-(decoded-string group, offset) TRAIN aggregates (greedy decisions) ------
    // Grouped by representative id: at a fixed offset the groups partition the positions,
    // so a set's mass is the plain sum over its groups.
    let mut rep_dce_tr = vec![0f64; idx.n_ids * NOFF];
    let mut rep_kl_tr = vec![0f64; idx.n_ids * NOFF];
    let mut rep_cnt_kl_tr = vec![0u64; idx.n_ids * NOFF];
    for i in 0..n {
        let p = pos[i] as usize;
        if p >= n_pos || !train.get(p) {
            continue;
        }
        let r = p / t;
        let tt = p % t;
        let d = dce[i] as f64;
        let k = dense[p][1] as f64;
        for o in 0..NOFF {
            let ts = tt as i32 + (o as i32 - 8);
            if ts < 0 {
                continue;
            }
            let id = idx.tok[r * t + ts as usize] as usize;
            let j = vocab.rep[id] as usize * NOFF + o;
            rep_dce_tr[j] += d;
            rep_kl_tr[j] += k;
            if k > 0.0 {
                rep_cnt_kl_tr[j] += 1;
            }
        }
    }

    let cap = n_pos as u64 / 2; // cov <= 0.5
    let mut cands: Vec<(Cand, Bits)> = Vec::new();
    let mut gset: Vec<Option<(Vec<u16>, f64)>> = vec![None; NOFF];

    // (a)/(b) greedy token set per offset, seeded from the top-KL ids of the train half;
    // only ids occurring >= 2 times among those positions may enter a set.
    for o in 0..NOFF {
        let mut allowed: Vec<u16> = (0..idx.n_ids as u16)
            .filter(|id| rep_cnt_kl_tr[*id as usize * NOFF + o] >= 2)
            .collect();
        if allowed.is_empty() {
            continue;
        }
        allowed.sort_by(|a, b| {
            rep_kl_tr[*b as usize * NOFF + o]
                .partial_cmp(&rep_kl_tr[*a as usize * NOFF + o])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(b))
        });
        let mut sel: Vec<u16> = Vec::new();
        let (mut s_dce, mut s_kl, mut s_cnt) = (0f64, 0f64, 0u64);
        let mut cur = ctx.objective_tr(0.0, 0.0, 0, 0.0);
        for id in &allowed {
            let j = *id as usize * NOFF + o;
            let c = idx.rep_count[j];
            if s_cnt + c > cap {
                continue;
            }
            let n_obj = ctx.objective_tr(
                s_dce + rep_dce_tr[j],
                s_kl + rep_kl_tr[j],
                sel.len() + 1,
                (s_cnt + c) as f64 / n_pos as f64,
            );
            if n_obj > cur + 1e-12 {
                sel.push(*id);
                s_dce += rep_dce_tr[j];
                s_kl += rep_kl_tr[j];
                s_cnt += c;
                cur = n_obj;
            }
        }
        // removal pass: drop members whose removal does not measurably lower the objective
        let mut i = 0;
        while i < sel.len() {
            let j = sel[i] as usize * NOFF + o;
            let obj_wo = ctx.objective_tr(
                s_dce - rep_dce_tr[j],
                s_kl - rep_kl_tr[j],
                sel.len() - 1,
                (s_cnt - idx.rep_count[j]) as f64 / n_pos as f64,
            );
            if obj_wo >= cur - 1e-12 {
                sel.remove(i);
                s_dce -= rep_dce_tr[j];
                s_kl -= rep_kl_tr[j];
                s_cnt -= idx.rep_count[j];
                cur = obj_wo;
            } else {
                i += 1;
            }
        }
        if !sel.is_empty() {
            let ids = vocab.expand_ids(&sel);
            let b = idx.ids_bits(&ids, o as i32 - 8);
            let vm = val_masses(&b, &kept_val, &dense);
            let mut c = make_cand(Spec::TokIn { off: o as i32 - 8, ids: ids.clone() }, vm, &ctx,
                                  vocab);
            c.key = c.spec.key();
            gset[o] = Some((ids, cur));
            cands.push((c, b));
        }
        if let Some(best) = allowed.first() {
            let ids = vocab.expand_ids(&[*best]);
            let b = idx.ids_bits(&ids, o as i32 - 8);
            let vm = val_masses(&b, &kept_val, &dense);
            let mut c = make_cand(Spec::TokIn { off: o as i32 - 8, ids }, vm, &ctx, vocab);
            c.key = c.spec.key();
            cands.push((c, b));
        }
    }

    // (c) every class at every offset (in particular all of 0..-3)
    for ci in 0..idx.n_cls {
        for o in 0..NOFF {
            let b = idx.cls_bits[ci * NOFF + o].clone();
            let cnt = b.count();
            if cnt == 0 || cnt > cap {
                continue;
            }
            let vm = val_masses(&b, &kept_val, &dense);
            let mut c = make_cand(
                Spec::Cls {
                    off: o as i32 - 8,
                    cls: vocab.class_names[ci].clone(),
                },
                vm,
                &ctx,
                vocab,
            );
            c.key = c.spec.key();
            cands.push((c, b));
        }
    }

    // (d) window classes over the (lo, hi) grid: masses accumulated as the window grows, and
    // the chain cut once coverage passes the cap (extending lo only adds positions).
    for ci in 0..idx.n_cls {
        for hi in (-8..=-1).rev() {
            let mut b = Bits::new(n_pos);
            let mut vm = ValMass::default();
            for lo in (-8..=hi).rev() {
                let cb = &idx.cls_bits[ci * NOFF + off_index(lo)];
                for w in 0..b.w.len() {
                    let add = cb.w[w] & !b.w[w];
                    if add == 0 {
                        continue;
                    }
                    vm.cnt += add.count_ones() as u64;
                    let mut x = add & kept_val.w[w];
                    while x != 0 {
                        let j = x.trailing_zeros() as usize;
                        let v = dense[(w << 6) + j];
                        vm.dce += v[0] as f64;
                        vm.kl += v[1] as f64;
                        x &= x - 1;
                    }
                    b.w[w] |= add;
                }
                if lo == hi {
                    continue; // single-offset window == cls at that offset (family (c))
                }
                if vm.cnt > cap {
                    break; // extending lo only grows the window
                }
                let mut c = make_cand(
                    Spec::WinCls {
                        lo,
                        hi,
                        cls: vocab.class_names[ci].clone(),
                    },
                    vm,
                    &ctx,
                    vocab,
                );
                c.key = c.spec.key();
                cands.push((c, b.clone()));
            }
        }
    }

    // (d') the two best greedy sets (by train objective) applied over every window
    let mut top: Vec<usize> = (0..NOFF).filter(|o| gset[*o].is_some()).collect();
    top.sort_by(|a, b| {
        gset[*b]
            .as_ref()
            .unwrap()
            .1
            .partial_cmp(&gset[*a].as_ref().unwrap().1)
            .unwrap_or(Ordering::Equal)
            .then(a.cmp(b))
    });
    top.truncate(2);
    for o in top {
        let ids = gset[o].as_ref().unwrap().0.clone();
        let sb: Vec<Bits> = (-8..=0).map(|j| idx.ids_bits(&ids, j)).collect();
        for hi in (-8..=-1).rev() {
            let mut b = Bits::new(n_pos);
            let mut vm = ValMass::default();
            for lo in (-8..=hi).rev() {
                let cb = &sb[off_index(lo)];
                for w in 0..b.w.len() {
                    let add = cb.w[w] & !b.w[w];
                    if add == 0 {
                        continue;
                    }
                    vm.cnt += add.count_ones() as u64;
                    let mut x = add & kept_val.w[w];
                    while x != 0 {
                        let j = x.trailing_zeros() as usize;
                        let v = dense[(w << 6) + j];
                        vm.dce += v[0] as f64;
                        vm.kl += v[1] as f64;
                        x &= x - 1;
                    }
                    b.w[w] |= add;
                }
                if lo == hi {
                    continue;
                }
                if vm.cnt > cap {
                    break;
                }
                let mut c = make_cand(
                    Spec::WinIn {
                        lo,
                        hi,
                        ids: ids.clone(),
                    },
                    vm,
                    &ctx,
                    vocab,
                );
                c.key = c.spec.key();
                cands.push((c, b.clone()));
            }
        }
    }

    // (e)/(f) AND of pairs among the best ~10 diverse clauses, OR of the best two.
    // The pool is diversified by fired-position set so that semantically identical clauses
    // (e.g. cls bracket_open@-1 and tok_in@-1 ['(']) do not crowd out distinct ones; the
    // fingerprints are computed for the leading candidates only.
    cands.retain(|(c, _)| c.cov > 0.0 && c.cov <= 0.5 && c.m > 0.0);
    cands.sort_by(|a, b| rank_dom(&a.0, &b.0));
    for k in 0..cands.len().min(64) {
        let h = bits_hash(&cands[k].1);
        cands[k].0.hash = h;
    }
    let mut pool: Vec<usize> = Vec::new();
    let mut pool_hash: Vec<u64> = Vec::new();
    for i in 0..cands.len().min(64) {
        if pool.len() >= 10 {
            break;
        }
        if !pool_hash.contains(&cands[i].0.hash) {
            pool_hash.push(cands[i].0.hash);
            pool.push(i);
        }
    }
    let mut composites: Vec<(Cand, Bits)> = Vec::new();
    for i in 0..pool.len() {
        for j in (i + 1)..pool.len() {
            let b = Bits::and2(&cands[pool[i]].1, &cands[pool[j]].1);
            let cnt = b.count();
            if cnt == 0 || cnt > cap {
                continue;
            }
            let vm = val_masses(&b, &kept_val, &dense);
            let mut c = make_cand(
                Spec::And(
                    Box::new(cands[pool[i]].0.spec.clone()),
                    Box::new(cands[pool[j]].0.spec.clone()),
                ),
                vm,
                &ctx,
                vocab,
            );
            c.key = c.spec.key();
            composites.push((c, b));
        }
    }
    if pool.len() >= 2 {
        let b = Bits::or2(&cands[pool[0]].1, &cands[pool[1]].1);
        let cnt = b.count();
        if cnt > 0 && cnt <= cap {
            let vm = val_masses(&b, &kept_val, &dense);
            let mut c = make_cand(
                Spec::Or(
                    Box::new(cands[pool[0]].0.spec.clone()),
                    Box::new(cands[pool[1]].0.spec.clone()),
                ),
                vm,
                &ctx,
                vocab,
            );
            c.key = c.spec.key();
            composites.push((c, b));
        }
    }
    cands.extend(composites);

    // ---- domination prune ------------------------------------------------------------
    // A candidate is dropped only when another candidate with (essentially) the same
    // penalty-free quality fires strictly inside it: firing wherever an equally good
    // predicate fires, and more, is redundant.  Candidates are visited in
    // (m desc, cov asc) order, so an equal-m dominator is always seen first and the
    // backwards scan can stop as soon as the objective drops below the tolerance.
    cands.sort_by(|a, b| rank_dom(&a.0, &b.0));
    let mut kept_idx: Vec<usize> = Vec::new();
    for i in 0..cands.len() {
        let (ci_cand, ci_bits) = &cands[i];
        let mut dominated = false;
        for &j in kept_idx.iter().rev() {
            let cj = &cands[j].0;
            if cj.m < ci_cand.m - M_TOL {
                break;
            }
            if cj.cnt >= ci_cand.cnt {
                continue;
            }
            if strict_subset(&cands[j].1, ci_bits, cj.cnt, ci_cand.cnt) {
                dominated = true;
                break;
            }
        }
        if !dominated {
            kept_idx.push(i);
        }
    }

    // ---- final ordering by the validation objective ---------------------------------
    let mut out: Vec<Cand> = Vec::new();
    let mut survivors: Vec<usize> = kept_idx;
    survivors.sort_by(|a, b| rank_final(&cands[*a].0, &cands[*b].0));
    let mut seen: Vec<&str> = Vec::new();
    for &i in &survivors {
        let (c, b) = &cands[i];
        if seen.contains(&c.key.as_str()) {
            continue;
        }
        seen.push(c.key.as_str());
        let mut c = c.clone();
        fill_reported(&mut c, b, &kept_tr, &kept_val, &dense, &ctx);
        out.push(c);
        if out.len() == top_m {
            break;
        }
    }
    out
}
