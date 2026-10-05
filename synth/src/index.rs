//! Position bitset index: for every token id x offset and class x offset, the set of
//! slice positions where that token/class occurs at that offset (out-of-row -> false).

use crate::bitset::Bits;
use crate::vocab::Vocab;

pub const NOFF: usize = 9; // offsets -8..=0

pub struct PosIndex {
    pub t: usize,
    pub n_pos: usize,
    pub n_ids: usize,
    pub n_cls: usize,
    /// tokens of the input window, [n_seq * t] (row-major, t = 256)
    pub tok: Vec<u16>,
    /// [id * NOFF + o] with o = off + 8
    pub id_bits: Vec<Bits>,
    /// Σ over the decoded-string group of the id popcount, indexed by group representative
    pub rep_count: Vec<u64>,
    /// [ci * NOFF + o]
    pub cls_bits: Vec<Bits>,
}

pub fn off_index(off: i32) -> usize {
    (off + 8) as usize
}

impl PosIndex {
    pub fn build(tokens: &[u16], n_seq: usize, cols: usize, t: usize, vocab: &Vocab) -> Self {
        let n_pos = n_seq * t;
        let n_ids = vocab.tokens.len();
        let n_cls = vocab.class_names.len();
        let mut tok = vec![0u16; n_pos];
        for r in 0..n_seq {
            for c in 0..t {
                tok[r * t + c] = tokens[r * cols + c];
            }
        }
        let mut id_bits: Vec<Bits> = (0..n_ids * NOFF).map(|_| Bits::new(n_pos)).collect();
        for r in 0..n_seq {
            for ts in 0..t {
                let id = tok[r * t + ts] as usize;
                if id >= n_ids {
                    continue;
                }
                for o in 0..NOFF {
                    let k = o as i32 - 8;
                    let td = ts as i32 - k;
                    if td >= 0 && (td as usize) < t {
                        id_bits[id * NOFF + o].set(r * t + td as usize);
                    }
                }
            }
        }
        // counts aggregated over decoded-string groups, indexed by the group representative
        let mut rep_count = vec![0u64; n_ids * NOFF];
        for id in 0..n_ids {
            let rid = vocab.rep[id] as usize;
            let n = id_bits[id * NOFF..(id + 1) * NOFF]
                .iter()
                .map(|b| b.count())
                .collect::<Vec<u64>>();
            for o in 0..NOFF {
                rep_count[rid * NOFF + o] += n[o];
            }
        }
        let mut cls_bits: Vec<Bits> = Vec::with_capacity(n_cls * NOFF);
        for ci in 0..n_cls {
            for o in 0..NOFF {
                let mut b = Bits::new(n_pos);
                for id in &vocab.class_ids[ci] {
                    b.or_with(&id_bits[*id as usize * NOFF + o]);
                }
                cls_bits.push(b);
            }
        }
        PosIndex {
            t,
            n_pos,
            n_ids,
            n_cls,
            tok,
            id_bits,
            rep_count,
            cls_bits,
        }
    }

    /// OR of the id bitsets for `ids` at offset `off`.
    pub fn ids_bits(&self, ids: &[u16], off: i32) -> Bits {
        let o = off_index(off);
        let mut b = Bits::new(self.n_pos);
        for id in ids {
            b.or_with(&self.id_bits[*id as usize * NOFF + o]);
        }
        b
    }

}
