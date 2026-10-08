/* Sparse Weight Atlas — vanilla JS, no build step. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const state = {
  model: null,
  summary: null,
  layers: new Map(),      // L -> parsed layer_LL.json
  buckets: new Map(),     // bucket -> parsed w/{b}.json
  observers: [],
  loadT0: performance.now(),
  loadReported: false,
};

const COLS = 256; // cells per canvas row

const COLORS = {
  unscored: [26, 30, 36],
  no_candidate: [127, 29, 29],
  noise_floor: [75, 85, 99],
  candidate: [92, 72, 140], // unverified synthesized rule (purple)
};
// Legacy exports used `pass` for the strict rule. Only the shared three-field
// schema identifies it as cone; missing rule fields must not become a false pass.
function hasRuleFields(record) {
  return typeof record.pass === "boolean" && typeof record.pass_null_mean === "boolean" &&
    typeof record.pass_strict === "boolean";
}

function ruleVerdicts(record) {
  if (hasRuleFields(record)) {
    return {
      cone: record.pass,
      nullMean: record.pass_null_mean,
      strict: record.pass_strict,
      derived: false,
    };
  }
  const r = record.robust;
  if (state.summary?.fixture && r &&
      [r.mean, r.sem, r.null_mean, r.null_sd].every(Number.isFinite)) {
    const lower = r.mean - 2 * r.sem;
    const gate = r.mean >= 0.75;
    return {
      cone: gate && lower > 0,
      nullMean: gate && lower > 0 && lower > r.null_mean,
      strict: gate && lower > 0 && r.pass === true,
      derived: true,
    };
  }
  return { cone: null, nullMean: null, strict: null, derived: false };
}

function verdictLabel(value) {
  return value === true ? "PASS" : value === false ? "FAIL" : "UNAVAILABLE";
}

function cellColor(row) {
  if (hasRuleFields(row) && row.pass) { // green marks only the cone headline rule
    const s = row.score_robust_mean ?? 0.75;
    const v = 120 + Math.round(90 * Math.min(1, Math.max(0, (s - 0.5) * 2)));
    return [40, v, 90];
  }
  if (row.status === "scored") { // blue
    const s = row.score_robust_mean;
    if (s == null) return [56, 189, 248];
    const v = 90 + Math.round(120 * Math.min(1, Math.max(0, s)));
    return [30, v, 200];
  }
  return COLORS[row.status] || COLORS.unscored;
}

async function jfetch(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

/* ---------- header ---------- */

function fmt(n) { return n == null ? "–" : n.toLocaleString("en-US"); }
function fmtPct(n) { return Number.isFinite(n) ? `${(n * 100).toFixed(2)}%` : "–"; }

function renderSummary(s) {
  state.summary = s;
  const c = s.counts;
  $("#st-total").textContent = fmt(c.total);
  $("#st-scored").textContent = fmt(c.scored);
  const hasRules = Number.isFinite(c.robust_beats_null_mean) && Number.isFinite(c.robust_strict);
  const rate = hasRules ? s.rates.robust : null;
  const nullRate = hasRules ? s.rates.robust_beats_null_mean : null;
  const strictRate = hasRules ? s.rates.robust_strict : null;
  $("#st-robust").textContent = fmt(hasRules ? c.robust_pass : null);
  $("#st-robust-rate").textContent = fmtPct(rate);
  $("#st-robust-pct").textContent = Number.isFinite(s.paper?.robust)
    ? `vs paper ${(s.paper.robust * 100).toFixed(1)}%` : "";
  $("#st-nullmean").textContent = fmt(hasRules ? c.robust_beats_null_mean : null);
  $("#st-nullmean-rate").textContent = fmtPct(nullRate);
  $("#st-strict").textContent = fmt(hasRules ? c.robust_strict : null);
  $("#st-strict-rate").textContent = fmtPct(strictRate);
  $("#st-noise").textContent = fmt(c.noise_floor);
  if ($("#st-cand")) $("#st-cand").textContent = fmt(c.candidate);
  const verdicts = c.scored + c.noise_floor + c.no_candidate;
  $("#data-note").textContent = s.fixture
    ? "SYNTHETIC FIXTURE · not measured model results. " +
      (hasRules ? "Rates describe synthetic verdicts only." :
        "Legacy aggregate/grid fields do not identify the three rules; unavailable outcomes are not passes. Card rules are derived from synthetic numeric stats.")
    : `Combined exported verdict rates · ${fmt(verdicts)} verdicts ` +
      `(scored + noise floor + no candidate); ${fmt(c.candidate ?? 0)} unverified candidate rules and ${fmt(c.unscored)} unscored weights excluded. ` +
      "Pilot + selected case study, not a random sample or the uniform-pilot headline comparison." +
      (hasRules ? "" : " Three-rule fields unavailable in this older export.");
  $("#foot-note").textContent =
    `model ${s.model} · ${fmt(c.total)} weights · cone ${fmtPct(rate)} · ` +
    `beats null mean ${fmtPct(nullRate)} · strict ${fmtPct(strictRate)}` +
    (s.fixture ? " · SYNTHETIC FIXTURE" : " · combined exported verdicts, not uniform-pilot rates");
}

/* ---------- layer grids ---------- */

function makePane(layer, matrix, rows) {
  const pane = document.createElement("div");
  pane.className = "pane";
  const label = document.createElement("div");
  label.className = "pane-label";
  label.textContent = `${matrix === 0 ? "c_fc" : "c_proj"} — ${rows.length} weights`;
  const cv = document.createElement("canvas");
  const n = rows.length;
  const canvasRows = Math.max(1, Math.ceil(n / COLS));
  cv.width = COLS; cv.height = canvasRows;
  cv.dataset.matrix = matrix;
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(COLS, canvasRows);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = cellColor(rows[i]);
    img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  pane.append(label, cv);
  return { pane, cv };
}

function attachHover(cv, rows) {
  const tip = $("#tooltip");
  cv.addEventListener("mousemove", (e) => {
    const rect = cv.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / rect.width * COLS);
    const y = Math.floor((e.clientY - rect.top) / rect.height * cv.height);
    const i = y * COLS + x;
    const row = rows[i];
    if (!row) { tip.classList.add("hidden"); return; }
    const sc = row.score_robust_mean != null ? ` · s̄=${row.score_robust_mean.toFixed(3)}` : "";
    const rules = ruleVerdicts(row);
    tip.innerHTML =
      `<b>wid ${row.wid}</b> · L${rowLayer(cv)} ${row.matrix === 0 ? "c_fc" : "c_proj"}` +
      `[${row.row},${row.col}] · v=${row.value.toExponential(2)}<br>` +
      `${row.status}${sc}` +
      `<br>Cone: ${verdictLabel(rules.cone)} · Null mean: ${verdictLabel(rules.nullMean)}` +
      `<br>Strict (null mean + 2 SD): ${verdictLabel(rules.strict)}` +
      (state.summary.fixture ? "<br>SYNTHETIC FIXTURE · not measured" : "") +
      (row.dce != null ? `<br>ΔCE=${row.dce.toExponential(3)}` : "") +
      (row.desc ? `<br><span class="tt-desc">${escapeHtml(row.desc)}</span>` : "");
    tip.classList.remove("hidden");
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tw - 8) + "px";
    tip.style.top = Math.min(e.clientY + 14, window.innerHeight - th - 8) + "px";
  });
  cv.addEventListener("mouseleave", () => $("#tooltip").classList.add("hidden"));
  cv.addEventListener("click", (e) => {
    const rect = cv.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / rect.width * COLS);
    const y = Math.floor((e.clientY - rect.top) / rect.height * cv.height);
    const row = rows[y * COLS + x];
    if (row) openCard(row.wid);
  });
}

function rowLayer(cv) { return cv.closest(".layer").dataset.layer; }

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

async function renderLayer(L) {
  if (state.layers.has(L)) return;
  state.layers.set(L, null); // in-flight guard
  const model = state.model;
  let data;
  try {
    data = await jfetch(`data/${model}/layer_${String(L).padStart(2, "0")}.json`);
  } catch (err) {
    console.warn(err);
    if (state.model === model) state.layers.delete(L);
    return;
  }
  if (state.model !== model) return; // model switched mid-flight; discard
  const sec = document.createElement("section");
  sec.className = "layer";
  sec.dataset.layer = L;
  sec.style.order = L;
  const scored = data.w.filter((r) => r.status === "scored").length;
  const knownRules = data.w.some(hasRuleFields);
  const passed = data.w.filter((r) => hasRuleFields(r) && r.pass).length;
  const head = document.createElement("div");
  head.className = "layer-head";
  head.innerHTML = `<b>layer ${L}</b><span>${data.n} weights</span>` +
    `<span>${scored} scored</span><span>${knownRules ? `${passed} cone passes` : "cone outcomes unavailable"}</span>` +
    (state.summary.fixture ? "<span>synthetic fixture</span>" : "");
  const panes = document.createElement("div");
  panes.className = "panes";
  const byMatrix = [[], []];
  for (const r of data.w) byMatrix[r.matrix].push(r);
  const fcPane = makePane(L, 0, byMatrix[0]);
  const projPane = makePane(L, 1, byMatrix[1]);
  attachHover(fcPane.cv, byMatrix[0]);
  attachHover(projPane.cv, byMatrix[1]);
  panes.append(fcPane.pane, projPane.pane);
  sec.append(head, panes);
  $(`#layers .layer[data-layer="${L}"]`)?.remove();
  $("#layers").appendChild(sec);
  state.layers.set(L, data);
  maybeReportLoad();
}

function setupLazyLayers(nLayers) {
  $("#layers").innerHTML = "";
  state.observers.forEach((o) => o.disconnect());
  state.observers = [];
  const io = new IntersectionObserver((entries) => {
    for (const en of entries) {
      if (en.isIntersecting) { renderLayer(Number(en.target.dataset.layer)); io.unobserve(en.target); }
    }
  }, { rootMargin: "400px" });
  for (let L = 0; L < nLayers; L++) {
    const ph = document.createElement("div");
    ph.className = "layer";
    ph.dataset.layer = L;
    ph.style.order = L;
    ph.style.minHeight = "60px";
    $("#layers").appendChild(ph);
    io.observe(ph);
  }
  state.observers = [io];
}

function maybeReportLoad() {
  if (state.loadReported || !state.summary) return;
  for (let L = 0; L < 3; L++) if (!state.layers.get(L)) return;
  state.loadReported = true;
  const ms = Math.round(performance.now() - state.loadT0);
  window.__atlasLoadMs = ms;
  console.log(`[atlas] ${state.summary.fixture ? "synthetic fixture" : "exported"} model ready in ${ms} ms (summary + first layers rendered)`);
}

/* ---------- weight card ---------- */

// Tokenize the raw source first, then escape each piece; highlighting already-escaped HTML
// let the number pass rewrite entities (&#39; -> &#<span>39</span>;).
const PY_TOKEN = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(def|return|any|all|for|in|not|and|or|is|None|True|False|range)\b|\b(\d+(?:\.\d+)?)\b/g;
function highlightPy(src) {
  let out = "", last = 0, m;
  PY_TOKEN.lastIndex = 0;
  while ((m = PY_TOKEN.exec(src))) {
    out += escapeHtml(src.slice(last, m.index));
    const cls = m[1] ? "s" : m[2] ? "k" : "n";
    out += `<span class="${cls}">${escapeHtml(m[0])}</span>`;
    last = m.index + m[0].length;
  }
  return out + escapeHtml(src.slice(last));
}

function scoreRow(name, blk, cls) {
  if (!blk) return "";
  const score = blk.score;
  const s = Math.max(0, Math.min(1, score ?? 0)); // clip the bar, never the displayed score
  const nums = `rec ${fmt3(blk.rec)} · inv ${fmt3(blk.inv)} · cov ${fmt3(blk.cov)} · marker = T 0.75`;
  return `<div class="score-row">
    <div class="sr-head"><span>${name}</span><span>score ${fmt3(score)}</span></div>
    <div class="bar"><div class="fill ${cls}" style="width:${(s * 100).toFixed(1)}%"></div>
      <div class="mark" style="left:75%"></div></div>
    <div class="sr-nums">${nums}</div>
  </div>`;
}

function fmt3(x) { return x == null ? "–" : Number(x).toFixed(3); }
function fmtRaw(x) { return Number.isFinite(x) ? String(x) : "–"; }

function robustHtml(w, rules) {
  const r = w.robust;
  if (!r && !state.summary.fixture) {
    return `<section class="robust-rules"><h3>Three robust rules · held-out</h3>
      <p class="sr-nums">Not tested: no held-out scores for this weight, so no rule can pass or fail.</p></section>`;
  }
  const rule = (name, value, formula) => `<div class="rule-row">
    <div class="rule-title"><strong>${name}</strong><span class="badge ${value === true ? "pass" : ""}">${verdictLabel(value)}</span></div>
    <code>${formula}</code></div>`;
  const lower = r && Number.isFinite(r.mean) && Number.isFinite(r.sem) ? r.mean - 2 * r.sem : null;
  const nullLimit = r && Number.isFinite(r.null_mean) && Number.isFinite(r.null_sd)
    ? r.null_mean + 2 * r.null_sd : null;
  return `<section class="robust-rules">
    <h3>Three robust rules · held-out${r?.scores ? ` · K=${r.scores.length}` : ""}</h3>
    ${state.summary.fixture ? '<p class="fixture-note">SYNTHETIC FIXTURE · not measured model evidence.' +
      (rules.derived ? " Outcomes derived from synthetic numeric stats." : " Rule fields unavailable.") + "</p>" : ""}
    ${rule("Cone · headline", rules.cone, "mean ≥ 0.75 and mean − 2·SEM &gt; 0")}
    ${rule("Beats null mean", rules.nullMean, "cone PASS and mean − 2·SEM &gt; mean(null)")}
    ${rule("Strict · null mean + 2 SD", rules.strict, "cone PASS and stored strict PASS (mean − 2·SEM &gt; mean(null) + 2·SD(null))")}
    ${r ? `<div class="robust-stats">
      <span>mean = ${fmtRaw(r.mean)}</span><span>SEM = ${fmtRaw(r.sem)}</span>
      <span>mean(null) = ${fmtRaw(r.null_mean)}</span><span>SD(null) = ${fmtRaw(r.null_sd)}</span>
      <span>mean − 2·SEM = ${fmtRaw(lower)}</span><span>mean(null) + 2·SD(null) = ${fmtRaw(nullLimit)}</span>
      <span>stored robust.pass (strict) = ${typeof r.pass === "boolean" ? String(r.pass) : "unavailable"}</span>
    </div>` : '<p class="sr-nums">No robust numeric stats: this weight was not robust-scored.</p>'}
  </section>`;
}

function tokenHtml(t, isFocus) {
  const vis = t === "" ? "∅" : /^[ ]+$/.test(t) ? "·".repeat(t.length)
    : t === "\n" ? "↵" : /^\s+$/.test(t) ? "␣" : t;
  const cls = isFocus ? "tok focus" : (/^\s*$/.test(t) ? "tok ws" : "tok");
  return `<span class="${cls}" title=${JSON.stringify(t)}>${escapeHtml(vis)}</span>`;
}

function contextHtml(ctx, i) {
  const toks = ctx.tokens.map((t, j) => tokenHtml(t, j === ctx.focus)).join("");
  return `<div class="ctx">#${i + 1} ${toks}
    <div class="meta">focus dce=${ctx.dce.toExponential(2)} · kl=${ctx.kl.toExponential(2)}</div></div>`;
}

async function openCard(wid) {
  const bucket = Math.floor(wid / 1000);
  const backdrop = $("#card-backdrop");
  const card = $("#card");
  card.innerHTML = "<p style='color:var(--dim);font-family:var(--mono)'>loading…</p>";
  backdrop.classList.remove("hidden");
  let w;
  try {
    const model = state.model;
    let b = state.buckets.get(bucket);
    if (!b) { b = await jfetch(`data/${model}/w/${bucket}.json`); if (state.model === model) state.buckets.set(bucket, b); }
    w = b.w[wid];
  } catch (err) {
    card.innerHTML = `<p>failed to load card: ${escapeHtml(String(err))}</p>`;
    return;
  }
  if (!w) { card.innerHTML = `<p>wid ${wid} not found in bucket ${bucket}</p>`; return; }
  const rules = ruleVerdicts(w);
  card.innerHTML = `
    <button class="card-close" id="card-close" aria-label="close">×</button>
    <h2>wid ${w.wid}</h2>
    <div class="cid">layer ${w.layer} · ${w.matrix_name}[${w.row}, ${w.col}] ·
      value ${w.value.toExponential(4)}
      ${w.dce != null ? `· ΔCE ${w.dce.toExponential(3)}` : ""}
      · <span class="badge ${w.status}">${w.status}</span>
      ${rules.cone === true ? '<span class="badge pass">cone robust pass</span>' : ""}</div>
    ${w.status === "candidate" ? `<p class="method-note unverified">UNVERIFIED: proposed by the Rust synthesizer from this weight's slice-00 ablation profile and never tested on held-out text${w.proxy ? ` · search proxy: coverage ${(w.proxy.cov * 100).toFixed(1)}%, validation-half score ${Number(w.proxy.score_val ?? w.proxy.score).toFixed(2)}` : ""}. ΔCE is the exact single-weight ablation on slice 00.</p>` : ""}
    ${w.desc ? `<div class="desc">${escapeHtml(w.desc)}</div>` : ""}
    ${robustHtml(w, rules)}
    ${w.py ? `<h3>predicate</h3><pre class="py">${highlightPy(w.py)}</pre>` : ""}
    ${w.spec ? `<details><summary class="cid">spec JSON</summary><pre class="py">${escapeHtml(JSON.stringify(w.spec, null, 1))}</pre></details>` : ""}
    ${scoreRow("InterpA · discovery, directly searched (inflated)", w.A, "blue")}
    ${scoreRow("InterpB (held-out slice 01)", w.B, "blue")}
    ${w.A ? '<p class="method-note">InterpA is optimized on discovery slice 00, so selection inflates it. It is not held-out evidence.</p>' : ""}
    ${w.contexts && w.contexts.length ? `<h3>top contexts (discovery slice, ±8 tokens)</h3>` : ""}
    ${(w.contexts || []).map(contextHtml).join("")}
  `;
  $("#card-close").addEventListener("click", () => backdrop.classList.add("hidden"));
}

$("#card-backdrop").addEventListener("click", (e) => {
  if (e.target.id === "card-backdrop") e.target.classList.add("hidden");
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("#card-backdrop").classList.add("hidden");
});

/* ---------- search ---------- */

const RE_IDX = /^L(\d{1,2})\s+(fc|c_fc|proj|c_proj)\s+(\d+)\s+(\d+)$/i;

async function ensureLayer(L) {
  if (!state.layers.get(L)) await renderLayer(L);
  return state.layers.get(L);
}

function nLayers() {
  return state.summary ? state.summary.per_layer.length : 12;
}

function layerOfWid(wid) {
  // per-layer totals from summary are equal for this model family; fall back to scan
  const pl = state.summary ? state.summary.per_layer : null;
  if (pl && pl.length && pl.every((p) => p.total === pl[0].total)) {
    return Math.min(Math.floor(wid / pl[0].total), pl.length - 1);
  }
  let acc = 0;
  for (let L = 0; L < (pl ? pl.length : 0); L++) {
    acc += pl[L].total;
    if (wid < acc) return L;
  }
  return 0;
}

async function findLayerOfWid(wid) {
  return ensureLayer(layerOfWid(wid));
}

async function runSearch(q) {
  const box = $("#search-results");
  const query = q.trim();
  if (!query) { box.classList.add("hidden"); return; }
  let hits = [];
  let m;
  if (/^\d+$/.test(query)) {
    const wid = Number(query);
    const data = await findLayerOfWid(wid);
    const row = data && data.w.find((r) => r.wid === wid);
    if (row) hits.push(row);
  } else if ((m = query.match(RE_IDX))) {
    const L = Number(m[1]);
    const wantM = /fc/i.test(m[2]) ? 0 : 1;
    const r = Number(m[3]), c = Number(m[4]);
    const data = await ensureLayer(L);
    const row = data && data.w.find((x) => x.matrix === wantM && x.row === r && x.col === c);
    if (row) hits.push(row);
  } else {
    const ql = query.toLowerCase();
    for (let L = 0; L < nLayers(); L++) {
      const data = await ensureLayer(L);
      if (!data) continue;
      for (const row of data.w) {
        if (row.desc && row.desc.toLowerCase().includes(ql)) {
          hits.push(row);
          if (hits.length >= 50) break;
        }
      }
      if (hits.length >= 50) break;
    }
  }
  box.innerHTML = hits.length
    ? hits.map((r) => `<div class="res" data-wid="${r.wid}">
        <span>wid ${r.wid}</span><span>L${layerOfWid(r.wid)} ${r.matrix === 0 ? "fc" : "proj"}[${r.row},${r.col}]</span>
        <span class="d">${escapeHtml(r.desc || r.status)}</span></div>`).join("")
    : `<div class="res"><span class="d">no matches</span></div>`;
  box.classList.remove("hidden");
  box.querySelectorAll(".res[data-wid]").forEach((el) =>
    el.addEventListener("click", () => {
      box.classList.add("hidden");
      openCard(Number(el.dataset.wid));
    }));
}

/* ---------- model switcher ---------- */

async function loadModel(id) {
  state.model = id;
  state.layers = new Map();
  state.buckets = new Map();
  state.loadReported = false;
  state.loadT0 = performance.now();
  $("#layers").innerHTML = "";
  let s;
  try {
    s = await jfetch(`data/${id}/summary.json`);
  } catch (err) {
    console.error(err);
    $("#st-total").textContent = "–";
    $("#foot-note").textContent = `failed to load model ${id}: ${err}`;
    return;
  }
  renderSummary(s);
  setupLazyLayers(s.per_layer.length);
}

async function init() {
  const sel = $("#model-select");
  let models = [];
  try {
    models = (await jfetch("data/models.json")).models;
  } catch (err) {
    console.warn("no data/models.json:", err);
  }
  if (!models.length) {
    models = [{ id: "fixture", summary: "data/fixture/summary.json" }];
  }
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.fixture || m.id === "fixture" ? `${m.id} (synthetic)` : m.id;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => loadModel(sel.value));
  sel.value = (models.find((m) => !m.fixture && m.id !== "fixture") || models[0]).id;
  await loadModel(sel.value);

  let deb;
  $("#search").addEventListener("input", (e) => {
    clearTimeout(deb);
    deb = setTimeout(() => runSearch(e.target.value), 250);
  });
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".searchbar")) $("#search-results").classList.add("hidden");
  });
}

init();
