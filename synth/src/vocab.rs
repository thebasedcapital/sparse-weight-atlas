//! runs/<model>/vocab.json loader + decoded-string equivalence classes.
//!
//! 128 token ids decode to the identical string '\ufffd'; specs that name a strict
//! subset of such a group cannot be expressed by a Python predicate over decoded
//! strings (see engine/swa/predicate.py emission), so every emitted id set is closed
//! under the group (expanded to all ids sharing that decoded string).

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

pub struct Vocab {
    pub tokens: Vec<String>,
    pub class_names: Vec<String>,
    pub class_ids: Vec<Vec<u16>>,
    /// id -> minimum id of its decoded-string equivalence group
    pub rep: Vec<u16>,
    group_of: Vec<u32>,
    groups: Vec<Vec<u16>>,
}

impl Vocab {
    pub fn load(path: &Path) -> Result<Self> {
        let txt = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let v: Value = serde_json::from_str(&txt).with_context(|| format!("parse {}", path.display()))?;
        let toks = v
            .get("tokens")
            .and_then(|t| t.as_array())
            .with_context(|| format!("{}: missing 'tokens'", path.display()))?;
        let tokens: Vec<String> = toks
            .iter()
            .map(|t| t.as_str().unwrap_or("").to_string())
            .collect();
        if tokens.len() != 2048 {
            anyhow::bail!("{}: expected 2048 tokens, got {}", path.display(), tokens.len());
        }
        let mut class_names: Vec<String> = Vec::new();
        let mut class_ids: Vec<Vec<u16>> = Vec::new();
        if let Some(cls) = v.get("classes").and_then(|c| c.as_object()) {
            for (name, ids) in cls.iter() {
                class_names.push(name.clone());
                class_ids.push(
                    ids.as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_i64())
                                .filter(|x| *x >= 0 && (*x as usize) < tokens.len())
                                .map(|x| x as u16)
                                .collect()
                        })
                        .unwrap_or_default(),
                );
            }
        }
        // deterministic class order
        let mut order: Vec<usize> = (0..class_names.len()).collect();
        order.sort_by(|a, b| class_names[*a].cmp(&class_names[*b]));
        let class_names: Vec<String> = order.iter().map(|i| class_names[*i].clone()).collect();
        let class_ids: Vec<Vec<u16>> = order
            .iter()
            .map(|i| {
                let mut v = class_ids[*i].clone();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();

        // decoded-string equivalence groups
        let mut by_str: HashMap<&str, Vec<u16>> = HashMap::new();
        for (i, s) in tokens.iter().enumerate() {
            by_str.entry(s.as_str()).or_default().push(i as u16);
        }
        let mut group_of = vec![0u32; tokens.len()];
        let mut groups: Vec<Vec<u16>> = Vec::new();
        let mut sorted: Vec<(&str, Vec<u16>)> = by_str.into_iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(b.0));
        for (_, ids) in sorted {
            let gi = groups.len() as u32;
            for id in &ids {
                group_of[*id as usize] = gi;
            }
            groups.push(ids);
        }
        let mut rep = vec![0u16; tokens.len()];
        for g in &groups {
            let r = *g.iter().min().unwrap();
            for id in g {
                rep[*id as usize] = r;
            }
        }
        Ok(Vocab {
            tokens,
            class_names,
            class_ids,
            rep,
            group_of,
            groups,
        })
    }

    /// Close an id set under decoded-string equivalence (sorted, deduped).
    pub fn expand_ids(&self, ids: &[u16]) -> Vec<u16> {
        let mut out: Vec<u16> = Vec::with_capacity(ids.len());
        for id in ids {
            let gi = self.group_of[*id as usize] as usize;
            out.extend_from_slice(&self.groups[gi]);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn class_index(&self, name: &str) -> Option<usize> {
        self.class_names.iter().position(|n| n == name)
    }
}
