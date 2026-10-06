#!/usr/bin/env python3
"""Build evidence-backed launch art and a browser tour; never perform model compute.

Run from any directory: .venv/bin/python scripts/make_launch.py
Requires numpy, Pillow, matplotlib, playwright (Chromium), and ffmpeg for the tour.
Absent inputs skip only the affected assets. Existing pipeline inputs are read-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
from swa.export_site import ProfileIndex, load_scores, robust_rules

BG = "#0c1119"
PANEL = "#151e2a"
TEXT = "#edf4ff"
DIM = "#a6b7ca"
GREEN = "#67e8a5"
BLUE = "#6bcaff"
RED = "#f29b9b"
RULES = (("robust", "paper null cone", GREEN),
         ("robust_beats_null_mean", "random null mean", BLUE),
         ("robust_strict", "strict mean + 2 SD", RED))
GALLERY_MAX_COV = 0.1
BROAD_COV = 0.3


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def skip(name, reason):
    print(f"[launch] SKIP {name}: {reason}", flush=True)


def font(size, mono=False, bold=False):
    name = ("LiberationMono" if mono else "LiberationSans") + ("-Bold.ttf" if bold else "-Regular.ttf")
    locations = [Path("/usr/share/fonts/liberation") / name,
                 Path("/usr/share/fonts/truetype/liberation2") / name]
    for path in locations:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def txt(draw, xy, value, size=26, color=TEXT, mono=False, bold=False):
    draw.text(xy, str(value), font=font(size, mono, bold), fill=color)


def source_lines(source, width):
    # The visual wrap changes no characters; indentation is retained at each source line.
    out = []
    for line in source.rstrip().splitlines():
        out.extend(textwrap.wrap(line, width=width, replace_whitespace=False,
                                 drop_whitespace=False, subsequent_indent="    ") or [""])
    return out


def save(image, out, name):
    path = out / name
    image.save(path)
    print(f"[launch] wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} "
          f"({image.width}x{image.height}; {path.stat().st_size:,} bytes)", flush=True)


def rate_label(block):
    if block.get("rate") is None:
        return "not available"
    return f"{100 * block['rate']:.1f}% ± {100 * block['sem']:.1f}% SEM"


def paper_label(report, key):
    label = {"interpA": "InterpA", "interpB": "InterpB", "robust": "Robust"}[key]
    uncertainty = re.search(rf"{label} [\d.]+ \+/- ([\d.]+)%", report["paper"].get("source", ""))
    sem = f" ± {uncertainty.group(1)}%" if uncertainty else ""
    return f"{100 * report['paper'][key]:.1f}%{sem}"


def broad_fraction(records):
    passing = [r for r in records.values() if robust_rules(r)["pass"]]
    broad = sum((r.get("A") or {}).get("cov", 0) > BROAD_COV for r in passing)
    return broad, len(passing)


def headline(report, corpus, out, pilot_records):
    rates = report["rates"]["over_verdicts"]
    if any(rates[k].get("rate") is None for k in ("interpA", "interpB", *(k for k, _, _ in RULES))):
        skip("headline.png", "pilot has no complete three-rule verdict-rate estimates")
        return
    image = Image.new("RGB", (1600, 900), BG)
    d = ImageDraw.Draw(image)
    txt(d, (65, 38), "THE ROBUST RATE DEPENDS ON THE NULL.", 57, bold=True)
    txt(d, (68, 118), f"{report['model']} · n = {rates['robust']['n']:,} uniform weights · no LLM calls", 32, GREEN)
    txt(d, (68, 180), "same executable predicates. same exact held-out scores. three different tests.", 28, DIM)
    for i, (key, label, color) in enumerate(RULES):
        x, y = 65 + i * 500, 245
        d.rounded_rectangle((x, y, x + 470, y + 210), radius=16, fill=PANEL, outline=color, width=2)
        txt(d, (x + 22, y + 18), label, 31, color, bold=True)
        txt(d, (x + 22, y + 63), f"{100 * rates[key]['rate']:.1f}%", 70, color, bold=True)
        txt(d, (x + 230, y + 96), f"± {100 * rates[key]['sem']:.1f}% SEM", 26, DIM)
        txt(d, (x + 25, y + 164), f"{rates[key]['k']} / {rates[key]['n']:,} passing", 26)
    txt(d, (68, 488), f"paper Table 1 · Gao model: robust {paper_label(report, 'robust')}", 33, BLUE, bold=True)
    for i, (key, label) in enumerate((("interpA", "InterpA"), ("interpB", "InterpB"))):
        y = 554 + i * 76
        txt(d, (68, y), label, 35, bold=True)
        txt(d, (255, y), rate_label(rates[key]), 33, GREEN)
        txt(d, (880, y), f"paper {paper_label(report, key)}", 33, BLUE)
    txt(d, (68, 722), "InterpA is inflated: the synthesizer directly searches discovery slice 00.", 27, RED)
    broad, n = broad_fraction(pilot_records)
    if n:
        txt(d, (68, 766), f"{100 * broad / n:.1f}% of cone passers have coverage > {100 * BROAD_COV:.0f}%: mostly broad class/window gates.", 25, DIM)
    txt(d, (68, 812), f"corpus: {corpus} · all verdicts, including noise-floor weights; ± binomial SEM", 24, DIM)
    txt(d, (68, 849), "Fig. 5 null-cone reading is not a matched replication; random-null rules expose the tension.", 24, DIM)
    save(image, out, "headline.png")


def layers_bins(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": DIM,
                         "ytick.color": DIM, "font.size": 15})
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=100, facecolor=BG,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for ax, key, label in zip(axes, ("per_layer", "per_bin"), ("layer", "bin")):
        rows = report[key]
        ax.set_facecolor(BG)
        x = np.arange(len(rows))
        for j, (rule, name, color) in enumerate(RULES):
            values = [100 * r[rule]["rate"] if r[rule]["rate"] is not None else np.nan for r in rows]
            errors = [100 * r[rule]["sem"] if r[rule]["sem"] is not None else 0 for r in rows]
            ax.bar(x + (j - 1) * .25, values, yerr=errors, color=color, capsize=3,
                   error_kw={"ecolor": DIM}, width=.23, label=name)
        ax.set_xticks(x, [str(r[label]) for r in rows], rotation=25 if label == "bin" else 0)
        high = max((100 * (r[k]["rate"] + r[k]["sem"]) for r in rows for k, _, _ in RULES
                    if r[k]["rate"] is not None), default=1)
        ax.set_ylim(0, high * 1.45)
        ax.set_ylabel("passing weights (%) ± SEM")
        ax.set_title("by layer" if label == "layer" else "by |ΔCE| per token (paper bins)", color=TEXT, pad=24)
        for i, r in enumerate(rows):
            ax.text(i, high * 1.25, f"n={r['n']}", rotation=90 if label == "layer" else 0,
                    fontsize=11, ha="center", va="bottom", color=DIM)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle(f"{report['model']} · same scores, three robust rules", fontsize=26, y=.95)
    fig.text(.065, .035, "denominators: uniform-pilot per-group n; error bars: binomial SEM; no LLM calls", color=DIM)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .895), ncol=3, frameon=False)
    fig.subplots_adjust(left=.07, right=.97, bottom=.15, top=.77, wspace=.3)
    path = out / "layers_bins.png"
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(f"[launch] wrote {path} (1600x900; {path.stat().st_size:,} bytes)")


def interp_pass(block):
    return bool(block and block.get("score") is not None and block.get("cov") is not None
                and block["score"] >= .75 and 0 < block["cov"] <= .5)


def neuron_records(records, weights, neuron=1863):
    return [r for wid, r in sorted(records.items()) if weights["layer"][wid] == 0
            and ((weights["matrix"][wid] == 0 and weights["row"][wid] == neuron)
                 or (weights["matrix"][wid] == 1 and weights["col"][wid] == neuron))]


def case_study(records, weights, out):
    rows = neuron_records(records, weights)
    if not rows:
        skip("casestudy_1863.png", "no layer-0 neuron-1863 scores / weight metadata")
        return rows
    sources = list(dict.fromkeys(r["py"] for r in rows if r.get("py")))
    source_id = {s: f"p{i + 1}" for i, s in enumerate(sources)}
    columns = [[r for r in rows if weights["matrix"][r["wid"]] == m] for m in (0, 1)]
    table_bottom = 440 + max(map(len, columns)) * 94
    source_blocks = [(source_id[s], source_lines(s, 166)) for s in sources]
    h = table_bottom + 130 + sum(56 + 30 * len(lines) for _, lines in source_blocks) + 110
    image = Image.new("RGB", (2400, h), BG)
    d = ImageDraw.Draw(image)
    passed = sum(bool(r.get("robust", {}).get("pass")) for r in rows)
    txt(d, (70, 48), "NEURON 1863: THE DIGIT DETECTOR", 57, bold=True)
    txt(d, (73, 125), f"csp_yolo1 · layer 0 · {len(rows)} scored weights · {passed} strict-null passes · no LLM calls", 32, GREEN)
    # Small orientation diagram; counts refer only to the displayed, measured weights.
    for x, label in ((150, f"{len(columns[0])} input weights"), (970, "neuron 1863"), (1790, f"{len(columns[1])} output weights")):
        d.rounded_rectangle((x, 215, x + 430, 320), radius=18, fill=PANEL, outline=BLUE, width=2)
        txt(d, (x + 35, 247), label, 35, bold=True)
    for start, end in ((580, 970), (1400, 1790)):
        d.line((start, 267, end, 267), fill=GREEN, width=5)
        d.polygon(((end, 267), (end - 20, 256), (end - 20, 278)), fill=GREEN)
    txt(d, (70, 350), "cov = discovery coverage; A/B = scores (P/F); R = mean ± SEM; PASS/FAIL = strict random-null rule", 29, DIM)
    for col, records_col in enumerate(columns):
        x = 70 + col * 1180
        txt(d, (x, 400), "c_fc[1863, input]" if col == 0 else "c_proj[output, 1863]", 33, BLUE, mono=True)
        for i, r in enumerate(records_col):
            y = 457 + i * 94
            wid = r["wid"]
            rob = r.get("robust") or {}
            passed = bool(rob.get("pass"))
            d.rounded_rectangle((x, y, x + 1110, y + 83), radius=10, fill=PANEL)
            idx = int(weights["col" if col == 0 else "row"][wid])
            txt(d, (x + 15, y + 5), f"{'in' if col == 0 else 'out'} {idx:<4} | wid {wid:<5} | {source_id.get(r.get('py'), 'no source')} | cov {100 * r['A']['cov']:.2f}%", 28, mono=True)
            txt(d, (x + 15, y + 44), f"A {r['A']['score']:.3f} {'P' if interp_pass(r['A']) else 'F'}  B {r['B']['score']:.3f} {'P' if interp_pass(r['B']) else 'F'}  R {rob['mean']:.3f} ±{rob['sem']:.3f} {'PASS' if passed else 'FAIL'}", 27, GREEN if passed else DIM, mono=True)
    y = table_bottom + 70
    txt(d, (70, y), "EXECUTABLE PREDICATE SOURCES · references above are exact, deduplicated source", 33, BLUE, bold=True)
    y += 65
    for name, lines in source_blocks:
        txt(d, (70, y), name, 28, GREEN, mono=True, bold=True)
        y += 38
        for line in lines:
            txt(d, (120, y), line, 21, mono=True)
            y += 30
        y += 18
    txt(d, (70, y + 18), "raw scores are not clipped; large means can still fail the coverage-matched robust null", 28, DIM)
    save(image, out, "casestudy_1863.png")
    return rows


def draw_context(draw, ctx, x, y, width):
    f = font(21, mono=True)
    left = x
    for i, token in enumerate(ctx["tokens"]):
        visible = token.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r") or "∅"
        focus = i == ctx["focus"]
        # Wrap long decoded tokens too, without discarding any text.
        for chunk in textwrap.wrap(visible, width=48, replace_whitespace=False, drop_whitespace=False) or [" "]:
            tw = draw.textlength(chunk, font=f)
            if x + tw > left + width and x != left:
                x, y = left, y + 30
            if focus:
                draw.rounded_rectangle((x - 2, y - 2, x + tw + 3, y + 27), radius=4, fill=GREEN)
            draw.text((x, y), chunk, font=f, fill=BG if focus else TEXT)
            x += tw
    return y + 36


def predicate_gallery(records, weights, runs, corpus_path, out, n):
    selected = sorted((r for r in records.values() if robust_rules(r)["pass"]
                       and 0 < (r.get("A") or {}).get("cov", 0) <= GALLERY_MAX_COV),
                      key=lambda r: (-abs(r.get("dce") or 0), r["wid"]))[:n]
    if not selected:
        skip("predicate_gallery.png", f"no cone-pass records with coverage <= {GALLERY_MAX_COV}")
        return
    vocab = json_read(runs / "vocab.json")
    if vocab is None or not corpus_path.is_file() or not (runs / "profiles").is_dir():
        skip("predicate_gallery.png", "decoded contexts need vocab.json, slice_00.npy, and profiles")
        return
    profile = ProfileIndex(runs)
    inputs = np.load(corpus_path, mmap_mode="r")[:, :256]
    cards = []
    for r in selected:
        contexts = profile.top_contexts(r["wid"], inputs, vocab["tokens"], k=1)
        if not contexts:
            print(f"[launch] no profile context for wid {r['wid']}; omitting gallery card")
            continue
        lines = source_lines(r["py"], 49)
        cards.append((r, lines, contexts[0], max(465, 275 + len(lines) * 29 + 150)))
    if not cards:
        skip("predicate_gallery.png", "no measured contexts for robust-passing weights")
        return
    row_heights = [max(c[3] for c in cards[i:i + 2]) for i in range(0, len(cards), 2)]
    image = Image.new("RGB", (1600, 215 + sum(h + 25 for h in row_heights)), BG)
    d = ImageDraw.Draw(image)
    txt(d, (55, 40), "PREDICATES YOU CAN RUN", 53, bold=True)
    txt(d, (57, 115), f"top {len(cards)} narrow cone passes · cov ≤ {100 * GALLERY_MAX_COV:.0f}% · sorted by |ΔCE| · green focus token", 25, GREEN)
    txt(d, (57, 149), "selected narrow exceptions, not the distribution of cone passers; raw scores are not clipped", 21, DIM)
    y = 183
    for pair, height in zip(range(0, len(cards), 2), row_heights):
        for col, (r, lines, ctx, _) in enumerate(cards[pair:pair + 2]):
            x = 55 + col * 765
            d.rounded_rectangle((x, y, x + 735, y + height), radius=15, fill=PANEL)
            wid = r["wid"]
            identity = f"L{int(weights['layer'][wid])} {'fc' if weights['matrix'][wid] == 0 else 'proj'}[{int(weights['row'][wid])},{int(weights['col'][wid])}] · wid {wid}"
            txt(d, (x + 20, y + 18), identity, 26, BLUE, mono=True, bold=True)
            rob = r["robust"]
            txt(d, (x + 20, y + 62), f"held-out mean {rob['mean']:.3f} ± {rob['sem']:.3f} SEM", 25, GREEN)
            txt(d, (x + 20, y + 99), f"A {r['A']['score']:.3f} · B {r['B']['score']:.3f} · cov {100 * r['A']['cov']:.2f}%", 24, DIM)
            verdicts = robust_rules(r)
            txt(d, (x + 20, y + 137), " · ".join(f"{name} {'PASS' if verdicts[key] else 'FAIL'}"
                for key, name in (("pass", "cone"), ("pass_null_mean", "null mean"), ("pass_strict", "strict"))), 23, BLUE)
            yy = y + 185
            for line in lines:
                txt(d, (x + 20, yy), line, 22, mono=True)
                yy += 29
            txt(d, (x + 20, yy + 18), "top KL context · discovery slice · ±8 tokens", 22, BLUE)
            yy = draw_context(d, ctx, x + 22, yy + 58, 690)
            txt(d, (x + 20, yy + 14), f"focus ΔCE {ctx['dce']:.2e} · KL {ctx['kl']:.2e}", 22, DIM)
        y += height + 25
    save(image, out, "predicate_gallery.png")


def thread(report, pilot_records, neuron, receipt, manifest, out):
    posts = []
    if report:
        rates = report["rates"]["over_verdicts"]
        posts.append(f"zero-llm search matches the paper's headline under our reading of its test: "
                     f"{100 * rates['robust']['rate']:.1f}% vs {100 * report['paper']['robust']:.1f}%. "
                     f"but against a coverage-matched random null: {100 * rates['robust_beats_null_mean']['rate']:.1f}%; "
                     f"strict null: {100 * rates['robust_strict']['rate']:.1f}%. that tension is the result.")
        posts.append(f"receipt: {receipt['mlp weights']:,} nonzero mlp weights in {report['model']}. "
                     f"uniform pilot: {report['n_sampled']:,} weights. cone rate {rate_label(rates['robust']).lower()}; "
                     f"paper {paper_label(report, 'robust')}. this is not a matched replication.")
    else:
        posts.append("a weight atlas, not llm explanations. no pilot report yet; no whole-model rate claimed. executable predicates, exact tests, measured case study below.")
    posts.append("method: ablate → synthesize → exact held-out test. the predicate controls where a single weight is restored. "
                 "rust enumerative search proposes code; exact conditional-zero cross-entropy tests what it preserves and removes. no llm calls.")
    if report:
        posts.append(f"interpa {rate_label(rates['interpA']).lower()} vs paper {100 * report['paper']['interpA']:.1f}%; "
                     f"interpb {rate_label(rates['interpB']).lower()} vs {100 * report['paper']['interpB']:.1f}%. "
                     "interpa is inflated: the synthesizer directly searches the discovery slice. held-out tests matter.")
        broad, cone_n = broad_fraction(pilot_records)
        if cone_n:
            posts.append(f"{broad}/{cone_n} cone-passing predicates ({100 * broad / cone_n:.1f}%) have coverage > {100 * BROAD_COV:.0f}%. "
                         "most passes are broad class/window gates, like a recent token being alphabetic. the narrow predicate gallery is a selected exception, not the distribution.")
        posts.append(f"cone: mean ≥ {report['paper']['T']} and mean − 2·sem > 0. "
                     "random-null mean: the lower bound also beats a coverage-matched random mask's mean. "
                     "strict null: it beats that mean + 2·sd. same scores, different robustness claims.")
        high = next(r for r in report["per_bin"] if r["bin"] == ">=1e-4")
        posts.append(f"high-impact bin |δce| {high['bin']}: {high['robust']['k']}/{high['n']} cone passes "
                     f"({100 * high['robust']['rate']:.1f}%). the paper calls this roughly a quarter. "
                     "a local high-impact result still does not establish narrow explanations for most weights.")
    if neuron:
        passed = sum(bool((r.get("robust") or {}).get("pass")) for r in neuron)
        posts.append(f"the paper's digit detector, neuron {neuron[0]['_neuron']}: {len(neuron)} scored input/output weights, "
                     f"{passed} strict-null passes. those passing predicates test the current token for digits. "
                     "the image keeps executable source, coverage and exact scores beside each weight.")
    posts.append(f"corpus: {manifest['source']}. repo-exclusive slices; one file per repo; "
                 f"at most {manifest['max_doc_tokens']} tokens per document; {manifest['n_seq'] * manifest['T']:,} positions per slice. "
                 "different corpus budget from the paper.")
    posts.append("honest limits: fixed causal grammar, not llm hypotheses; discovery train/validation split; coverage-penalized candidate selection, "
                 "not plain argmax. null-cone reading and both coverage-matched alternatives are explicit. deviations are in the readme.")
    posts.append("browse predicates, scores and contexts: {site_url}\nreproduce it: {repo_url}")
    if report:
        quote = (f"@Arnauya @sheimersheim zero-llm search reaches {100 * rates['robust']['rate']:.1f}% vs the paper's "
                 f"{100 * report['paper']['robust']:.1f}% under its null-cone test. but most predicates are broad; "
                 f"random-null mean {100 * rates['robust_beats_null_mean']['rate']:.1f}%, strict null "
                 f"{100 * rates['robust_strict']['rate']:.1f}%. executable code + exact tests: {{repo_url}}")
    else:
        quote = "@Arnauya @sheimersheim executable single-weight interpretations: rust predicate search, exact held-out tests, no llm calls. measured digit-neuron case study, not a whole-model rate. {repo_url}"
    for i, post in enumerate(posts):
        if len(post) > 280:
            raise ValueError(f"post {i + 1} exceeds 280 characters ({len(post)})")
        if post != post.lower():
            raise ValueError(f"post {i + 1} is not lowercase")
    if len(quote) > 280:
        raise ValueError("quote text exceeds 280 characters")
    sections = ["# x thread draft", "draft only; link placeholders must be replaced before posting."]
    for i, post in enumerate(posts, 1):
        sections.append(f"## post {i} ({len(post)} chars)\n\n{post}")
    sections.append("## quote tweet\n\nsource: https://x.com/Arnauya/status/2106229708438254025\n\n" + quote)
    (out / "thread.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"[launch] wrote thread.md ({len(posts)} posts; max {max(map(len, posts))} chars; quote {len(quote)} chars)")


def site_tour(site, out, preferred_model):
    manifest = json_read(site / "data/models.json")
    entries = (manifest or {}).get("models", [])
    choices = []
    for entry in entries:
        summary = json_read(site / entry["summary"])
        if summary:
            fixture = bool(entry.get("fixture") or summary.get("fixture"))
            # Historical unmarked synthetic exports must not be mistaken for measured data.
            if not fixture and (summary.get("provenance") or {}).get("kind") != "pipeline_export":
                print(f"[launch] ignoring unprovenanced site model {entry['id']}")
                continue
            choices.append((entry, summary, fixture))
    choices.sort(key=lambda c: (c[2], c[0]["id"] != preferred_model))
    selected = None
    for entry, summary, fixture in choices:
        layer0 = json_read(site / "data" / entry["id"] / "layer_00.json")
        if not layer0:
            continue
        passing = [r for r in layer0["w"] if r.get("pass")]
        if passing:
            # Prefer the paper's named neuron-1863 input weight, if it is exported and passing.
            weight = next((r for r in passing if r["matrix"] == 0 and r["row"] == 1863 and r["col"] == 464), passing[0])
            selected = (entry, summary, fixture, weight, layer0)
            break
    if selected is None:
        skip("site_tour.mp4", "no provenanced real export or labeled fixture with a layer-0 robust pass")
        return
    if not shutil.which("ffmpeg"):
        skip("site_tour.mp4", "ffmpeg is not installed")
        return
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        skip("site_tour.mp4", "install playwright and its Chromium browser")
        return
    entry, summary, fixture, weight, layer0 = selected
    filename = "site_tour_fixture.mp4" if fixture else "site_tour.mp4"
    server = subprocess.Popen(
        [sys.executable, "-u", "-m", "http.server", "0", "--bind", "127.0.0.1",
         "--directory", str(site)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        serving = server.stdout.readline()
        port_match = re.search(r"port (\d+)", serving)
        if port_match is None:
            raise RuntimeError(f"http.server did not start: {serving.strip()}")
        port = int(port_match.group(1))
        with tempfile.TemporaryDirectory(prefix="swa-launch-video-") as tmp, sync_playwright() as pw:
            bundled = Path(pw.chromium.executable_path)
            executable = str(bundled) if bundled.is_file() else shutil.which("chromium") or shutil.which("google-chrome")
            if executable is None:
                skip(filename, "no Chromium executable; run .venv/bin/playwright install chromium")
                return
            browser = pw.chromium.launch(executable_path=executable, headless=True, args=["--disable-gpu"])
            context = browser.new_context(viewport={"width": 1600, "height": 900},
                                          record_video_dir=tmp, record_video_size={"width": 1600, "height": 900})
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle")
            page.locator("#model-select").select_option(entry["id"])
            page.wait_for_function("document.querySelector('#st-total').textContent !== '–'")
            page.wait_for_selector('.layer[data-layer="0"] canvas')
            print(f"[launch] tour hero: {page.locator('#stats').inner_text().replace(chr(10), ' / ')}")
            # These dwells are intentional footage, not action-readiness sleeps.
            page.wait_for_timeout(3500)
            page.evaluate("document.body.style.zoom='1.25'")
            page.locator('.layer[data-layer="0"]').scroll_into_view_if_needed()
            page.wait_for_timeout(3500)
            cv = page.locator(f'.layer[data-layer="0"] canvas[data-matrix="{weight["matrix"]}"]')
            siblings = [r for r in layer0["w"] if r["matrix"] == weight["matrix"]]
            idx = next(i for i, r in enumerate(siblings) if r["wid"] == weight["wid"])
            box = cv.bounding_box()
            canvas_height = cv.evaluate("el => el.height")
            page.mouse.click(box["x"] + ((idx % 256) + .5) / 256 * box["width"],
                             box["y"] + ((idx // 256) + .5) / canvas_height * box["height"])
            page.wait_for_function("wid => document.querySelector('#card h2')?.textContent === `wid ${wid}`", arg=weight["wid"])
            page.wait_for_function("document.querySelector('#card').textContent.includes('PASS')")
            page.evaluate("document.body.style.zoom='1'")
            page.wait_for_timeout(6000)
            page.screenshot(path=str(out / "site_tour_card.png"))
            page.locator("#card-close").click()
            page.evaluate("window.scrollTo(0, 0)")
            page.locator("#search").fill("digit")
            page.wait_for_selector("#search-results .res[data-wid]")
            page.wait_for_timeout(4500)
            video = page.video
            context.close()
            browser.close()
            source = video.path()
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", source,
                            "-t", "29", "-vf", "scale=1600:900,setsar=1", "-r", "30", "-c:v", "libx264",
                            "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2",
                            "-movflags", "+faststart", "-an", str(out / filename)], check=True)
            print(f"[launch] wrote {filename} (model {entry['id']}; wid {weight['wid']}; fixture={fixture})")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server.stdout.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-report", type=Path, default=ROOT / "runs/csp_yolo1/pilot/report.json")
    ap.add_argument("--scores-dirs", nargs="+", type=Path, default=None,
                    help="merge JSONL scores from these directories; later dirs win duplicate wids")
    ap.add_argument("--model", default="csp_yolo1")
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "assets/launch")
    ap.add_argument("--skip-tour", action="store_true", help="regenerate stills/text without a browser recording")
    args = ap.parse_args()
    if args.top_n < 1:
        ap.error("--top-n must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    report = json_read(args.pilot_report)
    runs = ROOT / "runs" / args.model
    corpus_root = ROOT / "data" / args.model
    manifest = json_read(corpus_root / "manifest.json")
    pilot_records = load_scores(runs, Path(report["scores_dir"])) if report else {}
    if report:
        corpus = manifest["source"] if manifest else "corpus manifest unavailable"
        headline(report, corpus, args.out, pilot_records)
        layers_bins(report, args.out)
    else:
        for name in ("headline.png", "layers_bins.png"):
            skip(name, f"pilot report does not exist: {args.pilot_report}")
    directories = args.scores_dirs or [runs / "scores_casestudy", runs / "scores",
                                      Path(report["scores_dir"]) if report else runs / "scores_pilot"]
    records = {}
    for directory in directories:
        if directory.is_dir():
            records.update(load_scores(runs, directory))
        else:
            print(f"[launch] scores directory absent: {directory}")
    weights_path = runs / "weights.npz"
    neuron = []
    if weights_path.is_file():
        with np.load(weights_path) as weights:
            case_records = load_scores(runs, runs / "scores_casestudy")
            neuron = case_study(case_records, weights, args.out)
            for r in neuron:
                r["_neuron"] = int(weights["row" if weights["matrix"][r["wid"]] == 0 else "col"][r["wid"]])
            gallery_records = records if args.scores_dirs else pilot_records or records
            predicate_gallery(gallery_records, weights, runs, corpus_root / "slice_00.npy", args.out, args.top_n)
    else:
        skip("casestudy_1863.png / predicate_gallery.png", f"weight metadata absent: {weights_path}")
    if manifest:
        thread(report, pilot_records, neuron, json_read(ROOT / "assets" / f"{args.model}_param_receipt.json"), manifest, args.out)
    else:
        skip("thread.md", "corpus manifest absent; cannot substantiate model/corpus claims")
    if not args.skip_tour:
        site_tour(ROOT / "site", args.out, args.model)


if __name__ == "__main__":
    main()
