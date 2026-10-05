//! swa-synth: zero-LLM predicate synthesis for weight-sparse transformer weights.
//!
//! For every weight of every profile shard under runs/<model>/profiles/, search the
//! CONTRACT predicate grammar for specs maximizing the local-effect proxy score under
//! 0 < cov <= 0.5 and write runs/<model>/candidates/<shard>.jsonl (top-M distinct).

mod bitset;
mod emit;
mod index;
mod npy;
mod search;
mod spec;
mod vocab;

use anyhow::{bail, Context, Result};
use clap::Parser;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::index::PosIndex;
use crate::search::{search_weight, Cand};
use crate::vocab::Vocab;

#[derive(Parser, Debug)]
#[command(
    name = "swa-synth",
    about = "Synthesise weight predicates (CONTRACT proxy) for profile shards"
)]
struct Args {
    /// model name (runs/<model>, data/<model>)
    #[arg(long, default_value = "csp_yolo1")]
    model: String,
    /// repository root holding runs/ and data/
    #[arg(long, default_value = ".")]
    root: PathBuf,
    /// comma-separated shard names (default: every complete shard under
    /// runs/<model>/profiles/); a partial profile (complete=false) is processed only when
    /// named here, and then only its profiled wids get candidates
    #[arg(long)]
    shards: Option<String>,
    /// number of candidates kept per weight
    #[arg(long = "top-m", default_value_t = 8)]
    top_m: usize,
    /// complexity penalty per named token string in the objective
    /// `min(clip(rec), 1 - clip(inv), kl_recall) - lambda * complexity - mu * cov`
    /// (default 0.01: small against the [0, 1] objective and the 0.75 interpretability
    /// threshold, so it only breaks near-ties toward the simpler predicate)
    #[arg(long, default_value_t = search::DEFAULT_LAMBDA)]
    lambda: f64,
    /// coverage penalty `mu` in the same objective (default 0.5).  The clipped terms
    /// saturate at ~1 for any predicate covering the weight's dce/KL mass and kl_recall is
    /// monotone in coverage, so without a coverage price the broadest candidate under the
    /// 0.5 cap wins (near-cap `punct_only` windows outranked `digit_only` for the paper's
    /// neuron-1863 weights).  At 0.5 a candidate firing on 40% of the slice must be ~0.2
    /// better than a tight one to win, while a 2.9%-coverage predicate pays only 0.015
    #[arg(long = "cov-penalty", default_value_t = search::DEFAULT_COV_PENALTY)]
    cov_penalty: f64,
    /// worker threads (default: available parallelism)
    #[arg(long)]
    threads: Option<usize>,
    /// re-synthesise shards whose candidates/<shard>.jsonl already exists
    #[arg(long)]
    force: bool,
    /// print emitter self-test lines (per-class Python expression + set literal over the
    /// whole vocab) and exit; used by synth/fixtures/check.py for parity with predicate.py
    #[arg(long = "selftest-emit")]
    selftest_emit: bool,
}

/// Every shard directory that holds a profile (``wid.npy``), complete or not.
fn all_shards(profiles: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    for e in std::fs::read_dir(profiles)
        .with_context(|| format!("read {}", profiles.display()))?
    {
        let e = e?;
        if !e.file_type()?.is_dir() {
            continue;
        }
        if !e.path().join("wid.npy").exists() {
            continue;
        }
        names.push(e.file_name().to_string_lossy().to_string());
    }
    names.sort();
    Ok(names)
}

/// Default synthesis set: shards with a complete profile.  ``meta.json complete=false`` marks
/// a partial ``swa.profile --wids-file`` run; those shards are synthesised only when named
/// explicitly with ``--shards`` (then only their profiled wids get candidates).
fn list_shards(profiles: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    for name in all_shards(profiles)? {
        let meta = profiles.join(&name).join("meta.json");
        if let Ok(txt) = std::fs::read_to_string(&meta) {
            if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                if v.get("complete").and_then(|c| c.as_bool()) == Some(false) {
                    eprintln!(
                        "[synth] skip {} (meta.json complete=false; name it with --shards to synthesise its profiled wids)",
                        name
                    );
                    continue;
                }
            }
        }
        names.push(name);
    }
    Ok(names)
}

fn read_i32(path: &Path) -> Result<Vec<i32>> {
    Ok(npy::read_npy(path)?.as_i32().to_vec())
}

fn read_i64(path: &Path) -> Result<Vec<i64>> {
    Ok(npy::read_npy(path)?.as_i64().to_vec())
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    Ok(npy::read_npy(path)?.as_f32().to_vec())
}

fn cand_json(c: &Cand, vocab: &Vocab) -> Value {
    json!({
        "spec": c.spec.to_json(),
        "py": emit::spec_to_python(&c.spec, vocab),
        "desc": emit::describe(&c.spec, vocab),
        "proxy": {
            "rec": c.rec,
            "inv": c.inv,
            "cov": c.cov,
            "score": c.score,
            "score_train": c.score_train,
            "score_val": c.score_val,
            "kl_recall_val": c.kl_recall_val,
        },
    })
}

fn run_shard(
    shard: &str,
    root: &Path,
    model: &str,
    idx: &PosIndex,
    vocab: &Vocab,
    top_m: usize,
    lambda: f64,
    mu: f64,
    pool: &rayon::ThreadPool,
) -> Result<usize> {
    let dir = root.join("runs").join(model).join("profiles").join(shard);
    let out_path = root.join("runs").join(model).join("candidates").join(format!("{shard}.jsonl"));
    let wids = read_i32(&dir.join("wid.npy"))?;
    let indptr = read_i64(&dir.join("indptr.npy"))?;
    let pos = read_i32(&dir.join("pos.npy"))?;
    let dce = read_f32(&dir.join("dce.npy"))?;
    let kl = read_f32(&dir.join("kl.npy"))?;
    let dce_total = read_f32(&dir.join("dce_total.npy"))?;
    let dce_rest = read_f32(&dir.join("dce_rest.npy"))?;
    let n = wids.len();
    if indptr.len() != n + 1 {
        bail!(
            "{}: indptr has {} entries, expected {}",
            shard,
            indptr.len(),
            n + 1
        );
    }
    if dce_total.len() != n {
        bail!("{}: dce_total has {} entries, expected {}", shard, dce_total.len(), n);
    }
    if kl.len() != dce.len() {
        bail!("{}: kl has {} entries, expected {} (same as dce)", shard, kl.len(), dce.len());
    }
    if dce_rest.len() != n {
        bail!("{}: dce_rest has {} entries, expected {}", shard, dce_rest.len(), n);
    }
    if *indptr.last().unwrap_or(&0) as usize != pos.len() || pos.len() != dce.len() {
        bail!(
            "{}: CSR mismatch (pos={}, dce={}, indptr_last={})",
            shard,
            pos.len(),
            dce.len(),
            indptr.last().unwrap_or(&0)
        );
    }

    let t0 = Instant::now();
    let results: Vec<Vec<Cand>> = pool.install(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let a = indptr[i] as usize;
                let b = indptr[i + 1] as usize;
                search_weight(
                    idx,
                    vocab,
                    top_m,
                    lambda,
                    mu,
                    dce_total[i] as f64,
                    dce_rest[i] as f64,
                    &pos[a..b],
                    &dce[a..b],
                    &kl[a..b],
                )
            })
            .collect()
    });

    let mut buf = String::new();
    for (i, cands) in results.iter().enumerate() {
        let obj = json!({
            "wid": wids[i],
            "cands": cands.iter().map(|c| cand_json(c, vocab)).collect::<Vec<Value>>(),
        });
        buf.push_str(&serde_json::to_string(&obj)?);
        buf.push('\n');
    }
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = out_path.with_extension("jsonl.tmp");
    std::fs::write(&tmp, buf.as_bytes())?;
    std::fs::rename(&tmp, &out_path)?;

    let dt = t0.elapsed().as_secs_f64();
    let with_cands = results.iter().filter(|c| !c.is_empty()).count();
    eprintln!(
        "[synth] {shard}: {n} weights ({with_cands} with candidates) in {dt:.1}s ({:.1} weights/s) -> {}",
        n as f64 / dt.max(1e-9),
        out_path.display()
    );
    Ok(n)
}

fn selftest_emit(root: &Path, model: &str) -> Result<()> {
    let vocab = Vocab::load(&root.join("runs").join(model).join("vocab.json"))?;
    for (ci, name) in vocab.class_names.iter().enumerate() {
        let expr = emit::class_py_expr(name, "tokens[pos]", &vocab, &vocab.class_ids[ci]);
        println!("CLASS\t{name}\t{expr}");
    }
    println!("SETLIT\t{}", emit::set_lit(&vocab.tokens));
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = &args.root;
    if args.selftest_emit {
        return selftest_emit(root, &args.model);
    }
    let profiles = root.join("runs").join(&args.model).join("profiles");
    if !profiles.is_dir() {
        bail!("no profile shards at {}", profiles.display());
    }
    let all = all_shards(&profiles)?;
    if all.is_empty() {
        bail!("no shard directories (with wid.npy) under {}", profiles.display());
    }
    let shards: Vec<String> = match &args.shards {
        Some(s) => {
            let want: Vec<String> = s
                .split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect();
            for w in &want {
                // explicitly named shards may be partial (meta.json complete=false)
                if !all.contains(w) {
                    bail!("shard {w:?} not found under {}", profiles.display());
                }
                let meta = profiles.join(w).join("meta.json");
                if let Ok(txt) = std::fs::read_to_string(&meta) {
                    if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                        if v.get("complete").and_then(|c| c.as_bool()) == Some(false) {
                            eprintln!(
                                "[synth] {w}: partial profile (complete=false); synthesising its profiled wids"
                            );
                        }
                    }
                }
            }
            want
        }
        None => list_shards(&profiles)?,
    };

    let slice_path = root.join("data").join(&args.model).join("slice_00.npy");
    let slice = npy::read_npy(&slice_path)?;
    if slice.shape.len() != 2 || slice.shape[1] < 257 {
        bail!("{}: expected uint16 [n_seq, 257]", slice_path.display());
    }
    let (n_seq, cols) = (slice.shape[0], slice.shape[1]);
    let vocab = Vocab::load(&root.join("runs").join(&args.model).join("vocab.json"))?;
    let idx = PosIndex::build(slice.as_u16(), n_seq, cols, 256, &vocab);

    let nthreads = args
        .threads
        .or_else(|| std::thread::available_parallelism().ok().map(|v| v.get()))
        .unwrap_or(1)
        .max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .context("build rayon pool")?;

    eprintln!(
        "[synth] model={} n_seq={} n_pos={} ids={} classes={} shards={} top_m={} lambda={} mu={} threads={}",
        args.model,
        n_seq,
        idx.n_pos,
        idx.n_ids,
        idx.n_cls,
        shards.len(),
        args.top_m,
        args.lambda,
        args.cov_penalty,
        nthreads
    );

    let mut total = 0usize;
    let mut processed = 0usize;
    for shard in &shards {
        let out = root
            .join("runs")
            .join(&args.model)
            .join("candidates")
            .join(format!("{shard}.jsonl"));
        if out.exists() && !args.force {
            eprintln!("[synth] {shard}: skip (exists, use --force)");
            continue;
        }
        total += run_shard(shard, root, &args.model, &idx, &vocab, args.top_m, args.lambda,
                           args.cov_penalty, &pool)?;
        processed += 1;
    }
    eprintln!(
        "[synth] done: {processed} shards processed, {total} weights total"
    );
    Ok(())
}
