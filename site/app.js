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
};

function cellColor(row) {
  if (row.pass) { // green, brighter with higher robust score
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

function renderSummary(s) {
  state.summary = s;
  const c = s.counts;
  $("#st-total").textContent = fmt(c.total);
  $("#st-scored").textContent = fmt(c.scored);
  $("#st-robust").textContent = fmt(c.robust_pass);
  const rate = s.rates.robust;
  $("#st-robust-pct").textContent =
    rate != null ? `(${(rate * 100).toFixed(2)}% vs paper ${(s.paper.robust * 100).toFixed(1)}%)` : "";
  $("#st-noise").textContent = fmt(c.noise_floor);
  $("#foot-note").textContent =
    `model ${s.model} · ${fmt(c.total)} weights · robust ${(rate * 100).toFixed(2)}% ` +
    `(paper ${(s.paper.robust * 100).toFixed(1)}%) · T=${s.paper.T}, c=${s.paper.c}` +
    (s.fixture ? " · SYNTHETIC FIXTURE" : "");
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
    tip.innerHTML =
      `<b>wid ${row.wid}</b> · L${rowLayer(cv)} ${row.matrix === 0 ? "c_fc" : "c_proj"}` +
      `[${row.row},${row.col}] · v=${row.value.toExponential(2)}<br>` +
      `${row.status}${sc}${row.pass ? " · ROBUST PASS" : ""}` +
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
  const passed = data.w.filter((r) => r.pass).length;
  const head = document.createElement("div");
  head.className = "layer-head";
  head.innerHTML = `<b>layer ${L}</b><span>${data.n} weights</span>` +
    `<span>${scored} scored</span><span>${passed} robust pass</span>`;
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
  console.log(`[atlas] fixture model ready in ${ms} ms (summary + first layers rendered)`);
}

/* ---------- weight card ---------- */

function highlightPy(src) {
  const esc = escapeHtml(src);
  return esc
    .replace(/(&quot;[^&]*?&quot;|&#39;[^&]*?&#39;)/g, '<span class="s">$1</span>')
    .replace(/\b(def|return|any|all|for|in|not|and|or|is|None|True|False|range)\b/g, '<span class="k">$1</span>')
    .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="n">$1</span>');
}

function scoreRow(name, blk, cls, scoreOverride) {
  if (!blk) return "";
  const s = Math.max(0, Math.min(1, scoreOverride ?? blk.score ?? 0));
  const nums = blk.rec != null
    ? `rec ${fmt3(blk.rec)} · inv ${fmt3(blk.inv)} · cov ${fmt3(blk.cov)} · marker = T 0.75`
    : `null ${fmt3(blk.null_mean)} ± ${fmt3(blk.null_sd)} sd · marker = T 0.75`;
  return `<div class="score-row">
    <div class="sr-head"><span>${name}</span><span>score ${s.toFixed(3)}</span></div>
    <div class="bar"><div class="fill ${cls}" style="width:${(s * 100).toFixed(1)}%"></div>
      <div class="mark" style="left:75%"></div></div>
    <div class="sr-nums">${nums}</div>
  </div>`;
}

function fmt3(x) { return x == null ? "–" : Number(x).toFixed(3); }

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
  const rob = w.robust;
  const robLine = rob
    ? scoreRow(`robust (K=10 held-out)${rob.pass ? " — PASS" : ""}`, rob, "", rob.mean) +
      `<div class="sr-nums">mean ${fmt3(rob.mean)} ± ${fmt3(rob.sem)} sem</div>`
    : "";
  card.innerHTML = `
    <button class="card-close" id="card-close" aria-label="close">×</button>
    <h2>wid ${w.wid}</h2>
    <div class="cid">layer ${w.layer} · ${w.matrix_name}[${w.row}, ${w.col}] ·
      value ${w.value.toExponential(4)}
      ${w.dce != null ? `· ΔCE ${w.dce.toExponential(3)}` : ""}
      · <span class="badge ${w.status}">${w.status}</span>
      ${w.pass ? '<span class="badge pass">robust pass</span>' : ""}</div>
    ${w.desc ? `<div class="desc">${escapeHtml(w.desc)}</div>` : ""}
    ${w.py ? `<h3>predicate</h3><pre class="py">${highlightPy(w.py)}</pre>` : ""}
    ${w.spec ? `<details><summary class="cid">spec JSON</summary><pre class="py">${escapeHtml(JSON.stringify(w.spec, null, 1))}</pre></details>` : ""}
    ${scoreRow("InterpA (discovery slice 00)", w.A, "blue")}
    ${scoreRow("InterpB (held-out slice 01)", w.B, "blue")}
    ${robLine}
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
    opt.textContent = m.fixture && m.id !== "fixture" ? `${m.id} (synthetic)` : m.id;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => loadModel(sel.value));
  await loadModel(models[0].id);

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
