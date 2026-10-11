"""Build report.html from results_1b.json and results/cue_results.csv.

Run:  python model/report.py
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results_1b.json").read_text(encoding="utf-8"))

BASE = R["baseline_mae_cv"]
BAR = R["success_bar_15pct"]
SETS = R["feature_sets"]
PRIMARY_MAE = SETS["primary"]["mae"]


def _t(text: str) -> str:
    return f"<title>{escape(text)}</title>"


def dotplot(items, *, ref=None, ref_label="", h=27, w=760, pad_left=168,
            color=None):
    """Dot plot with error bars. items: [(label, value, err, tooltip)]."""
    lo = min(v - e for _, v, e, _ in items)
    hi_ = max(v + e for _, v, e, _ in items)
    if ref is not None:
        lo, hi_ = min(lo, ref), max(hi_, ref)
    pad = (hi_ - lo) * 0.12 or 0.1
    lo, hi_ = lo - pad, hi_ + pad
    plot = w - pad_left - 78
    X = lambda v: pad_left + plot * (v - lo) / (hi_ - lo)
    H = len(items) * h
    marks = []
    for i, (label, v, e, tip) in enumerate(items):
        y = i * h + h / 2
        c = color(label) if color else "var(--s1)"
        marks.append(
            f'<text x="{pad_left - 10}" y="{y + 4}" class="lbl" '
            f'text-anchor="end">{escape(label)}</text>'
            f'<line x1="{X(v-e):.1f}" y1="{y}" x2="{X(v+e):.1f}" y2="{y}" '
            f'stroke="{c}" stroke-width="2" opacity=".42"/>'
            f'<circle cx="{X(v):.1f}" cy="{y}" r="5" fill="{c}" '
            f'stroke="var(--surface)" stroke-width="2">{_t(tip)}</circle>'
            f'<text x="{w - 72}" y="{y + 4}" class="val">{v:.3f}</text>')
    refline = ""
    if ref is not None:
        rx = X(ref)
        refline = (f'<line x1="{rx:.1f}" y1="0" x2="{rx:.1f}" y2="{H}" '
                   f'class="ref"/>'
                   f'<text x="{rx:.1f}" y="{H + 14}" class="axlbl" '
                   f'text-anchor="middle">{escape(ref_label)}</text>')
    ticks = "".join(
        f'<line x1="{X(t/100):.1f}" y1="0" x2="{X(t/100):.1f}" y2="{H}" '
        f'class="grid"/>'
        f'<text x="{X(t/100):.1f}" y="{H + 14}" class="axlbl" '
        f'text-anchor="middle">{t/100:.2f}</text>'
        for t in range(int(lo * 100) + (5 - int(lo * 100) % 5), int(hi_ * 100), 5)
        if ref is None or abs(X(t / 100) - X(ref)) > 26)
    return (f'<svg viewBox="0 0 {w} {H + 24}" class="chart" role="img">'
            f'{ticks}{"".join(marks)}{refline}</svg>')


def forest(items, w=760, h=34, pad_left=168):
    """CI forest plot. items: [(label, est, lo, hi, excludes_zero, tooltip)]."""
    lo_all = min(min(lo, 0) for _, _, lo, _, _, _ in items)
    hi_all = max(hi for _, _, _, hi, _, _ in items)
    span = (hi_all - lo_all) or 1
    plot = w - pad_left - 110
    x = lambda v: pad_left + plot * (v - lo_all) / span
    rows = []
    for i, (label, est, lo, hi, ok, tip) in enumerate(items):
        y = i * h + h / 2
        col = "var(--good)" if ok else "var(--muted)"
        rows.append(
            f'<text x="{pad_left - 10}" y="{y + 4}" class="lbl" '
            f'text-anchor="end">{escape(label)}</text>'
            f'<line x1="{x(lo):.1f}" y1="{y}" x2="{x(hi):.1f}" y2="{y}" '
            f'stroke="{col}" stroke-width="2"/>'
            f'<line x1="{x(lo):.1f}" y1="{y-5}" x2="{x(lo):.1f}" y2="{y+5}" '
            f'stroke="{col}" stroke-width="2"/>'
            f'<line x1="{x(hi):.1f}" y1="{y-5}" x2="{x(hi):.1f}" y2="{y+5}" '
            f'stroke="{col}" stroke-width="2"/>'
            f'<circle cx="{x(est):.1f}" cy="{y}" r="5" fill="{col}" '
            f'stroke="var(--surface)" stroke-width="2">{_t(tip)}</circle>'
            f'<text x="{w - 104}" y="{y + 4}" class="val">'
            f'{est:+.3f} [{lo:+.3f}, {hi:+.3f}]</text>')
    zx = x(0)
    return (f'<svg viewBox="0 0 {w} {len(items)*h + 20}" class="chart" role="img">'
            f'<line x1="{zx:.1f}" y1="0" x2="{zx:.1f}" y2="{len(items)*h}" '
            f'class="ref"/>'
            f'<text x="{zx:.1f}" y="{len(items)*h + 13}" class="axlbl" '
            f'text-anchor="middle">no improvement</text>{"".join(rows)}</svg>')


def divbar(items, *, band=None, w=760, h=28, pad_left=140):
    """Diverging bars around zero. items: [(label, value, tooltip)]."""
    m = max(abs(v) for _, v, _ in items) * 1.25
    if band:
        m = max(m, band * 1.25)
    plot = w - pad_left - 150
    mid = pad_left + plot / 2
    rows = []
    if band:
        bw = plot * band / (2 * m)
        rows.append(f'<rect x="{mid - bw:.1f}" y="0" width="{2*bw:.1f}" '
                    f'height="{len(items)*h}" class="band"/>'
                    f'<text x="{mid:.1f}" y="{len(items)*h + 13}" class="axlbl" '
                    f'text-anchor="middle">+/- 1 fold sd = indistinguishable</text>')
    for i, (label, v, tip) in enumerate(items):
        y = i * h
        bw = plot * abs(v) / (2 * m)
        x0 = mid if v >= 0 else mid - bw
        col = "var(--s1)" if v >= 0 else "var(--s2)"
        rows.append(
            f'<text x="{pad_left - 10}" y="{y + h/2 + 4}" class="lbl" '
            f'text-anchor="end">{escape(label)}</text>'
            f'<rect class="bar" x="{x0:.1f}" y="{y + 5}" width="{max(bw,1.5):.1f}" '
            f'height="{h - 12}" rx="3" fill="{col}">{_t(tip)}</rect>'
            f'<text x="{pad_left + plot + 10}" y="{y + h/2 + 4}" class="val">'
            f'{v:+.3f}</text>')
    return (f'<svg viewBox="0 0 {w} {len(items)*h + 20}" class="chart" role="img">'
            f'<line x1="{mid:.1f}" y1="0" x2="{mid:.1f}" y2="{len(items)*h}" '
            f'class="ref"/>{"".join(rows)}</svg>')


def hist(null, real, p, *, w=360, h=130, bins=26):
    lo, hi = min(min(null), real), max(max(null), real)
    span = (hi - lo) or 1
    counts = [0] * bins
    for v in null:
        counts[min(bins - 1, int(bins * (v - lo) / span))] += 1
    cmax = max(counts) or 1
    bw = (w - 44) / bins
    bars = "".join(
        f'<rect x="{30 + i*bw:.1f}" y="{h - 26 - (h-44)*c/cmax:.1f}" '
        f'width="{bw - 1.4:.1f}" height="{(h-44)*c/cmax:.1f}" fill="var(--muted)" '
        f'opacity=".5">{_t(f"{c} of {len(null)} shuffles")}</rect>'
        for i, c in enumerate(counts) if c)
    rx = 30 + (w - 44) * (real - lo) / span
    col = "var(--good)" if p < 0.05 else "var(--crit)"
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{bars}'
            f'<line x1="{rx:.1f}" y1="8" x2="{rx:.1f}" y2="{h-26}" stroke="{col}" '
            f'stroke-width="2">{_t(f"real MAE {real:.3f}")}</line>'
            f'<text x="{rx:.1f}" y="{h-13}" class="axlbl" text-anchor="middle" '
            f'fill="{col}">real {real:.3f}</text>'
            f'<text x="{w-6}" y="14" class="axlbl" text-anchor="end">'
            f'shuffled labels</text></svg>')


def scatter(points, w=560, h=380, pad=52):
    xs = [p["actual"] for p in points] + [p["predicted"] for p in points]
    lo, hi = min(xs) - 0.4, max(xs) + 0.4
    span = hi - lo
    X = lambda v: pad + (w - pad - 18) * (v - lo) / span
    Y = lambda v: (h - pad) - (h - pad - 18) * (v - lo) / span
    varieties = sorted({p["variety"] for p in points})
    col = {v: f"var(--s{i+1})" for i, v in enumerate(varieties)}
    ticks = "".join(
        f'<line x1="{X(t)}" y1="{Y(lo)}" x2="{X(t)}" y2="{Y(hi)}" class="grid"/>'
        f'<line x1="{X(lo)}" y1="{Y(t)}" x2="{X(hi)}" y2="{Y(t)}" class="grid"/>'
        f'<text x="{X(t)}" y="{h-28}" class="axlbl" text-anchor="middle">{t}</text>'
        f'<text x="{pad-8}" y="{Y(t)+4}" class="axlbl" text-anchor="end">{t}</text>'
        for t in range(int(lo) + 1, int(hi) + 1))
    dot = []
    for p in points:
        tip = ("melon {id} ({v}): actual {a:.2f}, predicted {q:.2f}"
               .format(id=p["melon_id"], v=p["variety"], a=p["actual"],
                       q=p["predicted"]))
        dot.append(
            '<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{c}" '
            'stroke="var(--surface)" stroke-width="2">{t}</circle>'
            .format(cx=X(p["actual"]), cy=Y(p["predicted"]),
                    c=col[p["variety"]], t=_t(tip)))
    dots = "".join(dot)
    legend = "".join(
        f'<circle cx="{pad + 10 + i*110}" cy="18" r="5" fill="{col[v]}"/>'
        f'<text x="{pad + 22 + i*110}" y="22" class="axlbl">{escape(v)}</text>'
        for i, v in enumerate(varieties))
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{ticks}'
            f'<line x1="{X(lo)}" y1="{Y(lo)}" x2="{X(hi)}" y2="{Y(hi)}" '
            f'class="ref"/>{dots}{legend}'
            f'<text x="{(w+pad)/2}" y="{h-6}" class="axlbl" text-anchor="middle">'
            f'actual degBrix</text>'
            f'<text x="14" y="{h/2}" class="axlbl" text-anchor="middle" '
            f'transform="rotate(-90 14 {h/2})">predicted degBrix</text></svg>')


def linechart(pts, w=560, h=260, pad=54):
    xs = [p["n"] for p in pts]
    ys = [p["mae"] + p["sd"] for p in pts] + [p["mae"] - p["sd"] for p in pts]
    xlo, xhi = min(xs) - 1, max(xs) + 1
    ylo, yhi = min(ys) - 0.04, max(ys) + 0.04
    X = lambda v: pad + (w - pad - 18) * (v - xlo) / (xhi - xlo)
    Y = lambda v: (h - pad) - (h - pad - 18) * (v - ylo) / (yhi - ylo)
    band = ("M " + " L ".join(f"{X(p['n']):.1f} {Y(p['mae']+p['sd']):.1f}" for p in pts)
            + " L " + " L ".join(f"{X(p['n']):.1f} {Y(p['mae']-p['sd']):.1f}"
                                 for p in reversed(pts)) + " Z")
    line = "M " + " L ".join(f"{X(p['n']):.1f} {Y(p['mae']):.1f}" for p in pts)
    dot = []
    for p in pts:
        tip = "n={n}: MAE {m:.3f} +- {s:.3f}".format(n=p["n"], m=p["mae"],
                                                     s=p["sd"])
        dot.append(
            '<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="var(--s1)" '
            'stroke="var(--surface)" stroke-width="2">{t}</circle>'
            '<text x="{cx:.1f}" y="{ty}" class="axlbl" text-anchor="middle">'
            '{n}</text>'.format(cx=X(p["n"]), cy=Y(p["mae"]), t=_t(tip),
                                ty=h - 28, n=p["n"]))
    dots = "".join(dot)
    bl = BASE
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">'
            f'<line x1="{X(xlo)}" y1="{Y(bl)}" x2="{X(xhi)}" y2="{Y(bl)}" '
            f'class="ref"/>'
            f'<text x="{X(xhi)}" y="{Y(bl)-6}" class="axlbl" text-anchor="end">'
            f'baseline {bl:.3f}</text>'
            f'<path d="{band}" fill="var(--s1)" opacity=".13"/>'
            f'<path d="{line}" fill="none" stroke="var(--s1)" stroke-width="2"/>'
            f'{dots}<text x="{(w+pad)/2}" y="{h-6}" class="axlbl" '
            f'text-anchor="middle">melons used</text></svg>')


def gbar(groups, series, *, w=760, unit="", bh=20, gap=12, pad_left=150):
    """Grouped bars. groups: [(glabel, [(slabel, value)])]."""
    vmax = max(v for _, ss in groups for _, v in ss) * 1.14
    plot = w - pad_left - 70
    scol = {s: f"var(--s{i+1})" for i, s in enumerate(series)}
    out, y = [], 0
    for glabel, ss in groups:
        out.append(f'<text x="{pad_left - 10}" y="{y + len(ss)*bh/2 + 4}" '
                   f'class="lbl" text-anchor="end">{escape(glabel)}</text>')
        for slabel, v in ss:
            bw = max(2.0, plot * v / vmax)
            out.append(
                f'<rect class="bar" x="{pad_left}" y="{y + 2}" width="{bw:.1f}" '
                f'height="{bh - 4}" rx="3" fill="{scol[slabel]}">'
                f'{_t(f"{glabel} / {slabel}: {v:.3f}{unit}")}</rect>'
                f'<text x="{pad_left + bw + 7}" y="{y + bh/2 + 4}" class="val">'
                f'{v:.3f}</text>')
            y += bh
        y += gap
    legend = "".join(
        f'<rect x="{pad_left + i*118}" y="{y}" width="11" height="11" rx="2" '
        f'fill="{scol[s]}"/><text x="{pad_left + 17 + i*118}" y="{y+10}" '
        f'class="axlbl">{escape(s)}</text>' for i, s in enumerate(series))
    return (f'<svg viewBox="0 0 {w} {y + 20}" class="chart" role="img">'
            f'{"".join(out)}{legend}</svg>')


def heatmap(cols, matrix):
    head = "".join(f"<th>{escape(c)}</th>" for c in cols)
    rows = []
    for i, c in enumerate(cols):
        cells = []
        for j, v in enumerate(matrix[i]):
            if i == j:
                cells.append('<td class="diag">1</td>')
                continue
            a = min(abs(v), 1) * 0.8
            hue = "var(--s1)" if v > 0 else "var(--s2)"
            cells.append(
                f'<td style="--a:{a:.3f};--c:{hue}" class="hm" '
                f'title="{escape(c)} vs {escape(cols[j])}: rho {v:+.3f}">'
                f'{v:+.2f}</td>')
        rows.append(f"<tr><th>{escape(c)}</th>{''.join(cells)}</tr>")
    return (f'<table class="hmtab"><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def table(headers, rows, cls=""):
    h = "".join(f"<th>{escape(str(x))}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>"
                for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


DEFS = R["set_definitions"]

# Sets with identical column lists are collapsed to one row: visual_only and
# minus_sound are the same five columns, because sound is the only non-visual
# cue, and showing both implied two findings where there is one.
DUPES = {a: b for i, a in enumerate(DEFS) for b in list(DEFS)[i + 1:]
         if DEFS[a] and DEFS[a] == DEFS[b]}
ORDER = [k for k in sorted(SETS, key=lambda k: SETS[k]["mae"])
         if k not in DUPES.values()]
ALIAS = {a: f"{a}  (= {b})" for a, b in DUPES.items()}

SET_WHY = {
    "baseline_mean": "Floor. No features at all &mdash; predicts the training "
                     "fold&rsquo;s mean Brix for every melon.",
    "primary": "The cues the proposal named primary. Cue 2 (weight) has no "
               "column, so this is 3 cues in 4 columns.",
    "all_cues": "One pre-committed column per cue. The headline combined model.",
    "visual_only": "Everything you can see. <b>Identical to minus_sound</b>, "
                   "because sound is the only non-visual cue.",
    "audio_only": "The knock alone &mdash; pitch and decay.",
    "all_measured": "Every complete column in the table, including three that "
                    "are not any cue&rsquo;s representative. Asks whether "
                    "boiling each cue down to one number lost signal.",
}
for _c in R["cue_columns"]:
    SET_WHY[f"minus_{_c}"] = (f"Ablation: all_cues with <b>{escape(_c)}</b> "
                              f"removed, to see what it was contributing.")

defs_table = table(
    ["feature set", "n", "columns", "what it is"],
    [[f"<b>{escape(ALIAS.get(k, k))}</b>", len(DEFS[k]),
      ", ".join(f"<code>{escape(c)}</code>" for c in DEFS[k]) or "&mdash;",
      SET_WHY.get(k, "")] for k in ORDER])

extra = [c for c in DEFS["all_measured"] if c not in DEFS["all_cues"]]
ABL = {a["cue"]: a for a in R["ablation_per_cue"]}
BOOT = R["bootstrap_ci"]
PERM = {p["set"]: p for p in R["permutation_null"]}

crit_mae = PRIMARY_MAE <= BAR
crit_ci = BOOT["primary"]["excludes_zero"]
crit_perm = PERM["primary"]["p"] < 0.05

set_bars = dotplot(
    [(ALIAS.get(k, k), SETS[k]["mae"], SETS[k]["mae_sd"],
      f"{k}: MAE {SETS[k]['mae']:.3f} +- {SETS[k]['mae_sd']:.3f} "
      f"over 100 folds, {SETS[k]['n_feat']} features") for k in ORDER],
    ref=BAR, ref_label=f"15% bar {BAR:.3f}",
    color=lambda k: ("var(--s2)" if k == "baseline_mean"
                     else "var(--s3)" if k.startswith("minus_")
                     else "var(--s1)"))

set_table = table(
    ["feature set", "features", "MAE", "fold sd", "RMSE", "rho OOF", "vs baseline"],
    [[escape(ALIAS.get(k, k)), SETS[k]["n_feat"], f"{SETS[k]['mae']:.3f}",
      f"{SETS[k]['mae_sd']:.3f}",
      f"{SETS[k]['rmse']:.3f}",
      "&ndash;" if SETS[k]["rho_oof"] is None else f"{SETS[k]['rho_oof']:+.3f}",
      f"{100*(SETS[k]['mae']-BASE)/BASE:+.1f}%"] for k in ORDER], "num")

forest_svg = forest([
    (k, b["improvement"], b["ci_lo"], b["ci_hi"], b["excludes_zero"],
     f"{k}: {b['improvement']:+.3f} degBrix ({b['pct_of_baseline']:+.1f}% of "
     f"baseline), 95% CI [{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}]")
    for k, b in BOOT.items()])

perm_cards = "".join(
    f'<div class="sub"><h4>{escape(p["set"])} '
    f'<span class="pill {"ok" if p["p"]<0.05 else "no"}">p = {p["p"]:.3f}</span></h4>'
    f'{hist(p["null"], p["real_mae"], p["p"])}'
    f'<p class="cap">Real MAE {p["real_mae"]:.3f} against a null median of '
    f'{p["null_median"]:.3f}. {escape(p["reading"])}.</p></div>'
    for p in R["permutation_null"])

abl_svg = divbar(
    [(c, ABL[c]["delta_mae"],
      f"removing {c}: MAE changes by {ABL[c]['delta_mae']:+.3f} "
      f"({ABL[c]['reading']})")
     for c in sorted(ABL, key=lambda c: -ABL[c]["delta_mae"])],
    band=R["ablation_fold_sd"])

lovo = gbar([(t["set"], sorted(t["per_variety"].items()))
             for t in R["leave_one_variety_out"]],
            sorted(R["leave_one_variety_out"][0]["per_variety"]), unit=" degBrix")

fam = gbar([(f["model"], [("primary", f["primary"]), ("all_cues", f["all_cues"])])
            for f in R["model_families"]], ["primary", "all_cues"])

pm = sorted(R["per_melon"], key=lambda p: -abs(p["error"]))
per_melon_tab = table(
    ["melon", "variety", "actual", "predicted", "error"],
    [[p["melon_id"], escape(p["variety"]), f"{p['actual']:.2f}",
      f"{p['predicted']:.2f}",
      f'<span class="{"bad" if abs(p["error"])>1.2 else ""}">{p["error"]:+.2f}</span>']
     for p in pm], "num")

shrink_tab = table(
    ["feature set", "alpha chosen", "reading"],
    [[k, f"{v['alpha']:.3f}",
      "heavy shrinkage &mdash; near the mean baseline" if v["alpha"] > 100
      else "moderate &mdash; the features are genuinely used"]
     for k, v in R["shrinkage"].items()], "num")

vt = R["variety_test"]
variety_tab = table(
    ["model", "MAE", "reading"],
    [["variety label alone", f"{vt['variety_only']:.3f}",
      "no better than the baseline &mdash; variety alone predicts nothing"],
     ["all_cues", f"{vt['all_cues']:.3f}", "the cues, without the label"],
     ["all_cues + variety", f"{vt['all_cues_plus_variety']:.3f}",
      "adding the label makes it worse"]], "num")

redundant = R["cue_correlation"]["redundant_pairs"]
red_txt = ("".join(f"<li><code>{escape(a)}</code> &harr; <code>{escape(b)}</code> "
                   f"&rho; = {r:+.3f}</li>" for a, b, r in redundant)
           or "<li>No pair exceeds |&rho;| 0.5.</li>")

cue_rows = "".join(
    f"<li><b>{escape(c)}</b> &rarr; <code>{', '.join(escape(x) for x in cols)}</code>"
    + (' <span class="pill ok">primary</span>' if c in R["primary_cues"] else "")
    + "</li>" for c, cols in R["cue_columns"].items())
absent_rows = "".join(f"<li><b>{escape(c)}</b> &mdash; {escape(w)}</li>"
                      for c, w in R["absent_cues"].items())

with open(HERE.parent / "results" / "cue_results.csv", newline="", encoding="utf-8") as f:
    A1 = list(csv.DictReader(f))
NUMCOLS = ("spearman_rho", "p_single", "p_best_of_7", "partial_rho",
           "partial_p_single", "partial_p_best_of_7", "rho_sonya",
           "rho_kinnaree", "rho_vs_order")
for r in A1:
    for k in NUMCOLS:
        r[k] = float(r[k])
C1A = json.loads((HERE.parent / "results" / "analysis_1a.json").read_text(
    encoding="utf-8"))["confound"]

# the preregistered statistic is the partial one, so that is what is plotted
a1_svg = divbar(
    [(r["column"], r["partial_rho"],
      f"{r['column']}: partial rho {r['partial_rho']:+.3f} (plain "
      f"{r['spearman_rho']:+.3f}), p best-of-7 {r['partial_p_best_of_7']:.3f}, "
      f"{r['verdict']}")
     for r in sorted(A1, key=lambda r: -abs(r["partial_rho"]))])

A1_PILL = {"supported": "ok", "suggestive": "", "no evidence": "no"}
a1_tab = table(
    ["column", "cue", "plain rho", "partial rho", "p best of 7", "sonya",
     "kinnaree", "verdict", "folklore"],
    [[f"<code>{escape(r['column'])}</code>", escape(r["cue"]),
      f"{r['spearman_rho']:+.3f}", f"<b>{r['partial_rho']:+.3f}</b>",
      f"{r['partial_p_best_of_7']:.3f}",
      f"{r['rho_sonya']:+.3f}", f"{r['rho_kinnaree']:+.3f}",
      f'<span class="pill {A1_PILL[r["verdict"]]}">{escape(r["verdict"])}</span>'
      + ("" if r["verdict"] == r["verdict_plain"] else
         f' <span class="cap">was &ldquo;{escape(r["verdict_plain"])}&rdquo; '
         f'without the control</span>'),
      escape(r["folk"])] for r in A1], "num")

order_tab = table(
    ["cue column", "rho vs shooting order", "cue &harr; Brix controlling for order"],
    [[f"<code>{escape(c)}</code>", f"{v:+.3f}",
      f"{C1A['cue_brix_controlling_order'][c]:+.3f}"]
     for c, v in sorted(C1A["cue_vs_order"].items(), key=lambda kv: -abs(kv[1]))],
    "num")

blocks = " &middot; ".join(
    f"<b>{escape(v)}</b> melons {min(ids)}&ndash;{max(ids)}"
    for v, ids in C1A["variety_blocks"].items())

RANK = R["ranking"]
rank_tab = table(
    ["scorer", "kind", "pairwise", "best-of-3 gain", "best-of-3 hit",
     "&rho; OOF", "within sonya", "within kinnaree"],
    [[f"<code>{escape(r['name'])}</code>", escape(r["kind"]),
      f"<b>{r['pairwise']:.3f}</b>", f"{r['best_of_3_gain']:+.3f}",
      f"{r['best_of_3_hit']:.3f}",
      "&mdash;" if r["rho_oof"] is None else f"{r['rho_oof']:+.3f}",
      f"{r['pairwise_sonya']:.3f}", f"{r['pairwise_kinnaree']:.3f}"]
     for r in sorted(RANK, key=lambda r: -r["pairwise_best"])], "num")

rank_svg = dotplot(
    [(r["name"], r["pairwise"], 0.0,
      f"{r['name']}: pairwise {r['pairwise']:.3f}, best-of-3 gain "
      f"{r['best_of_3_gain']:+.3f} degBrix, picks the sweetest of 3 "
      f"{100*r['best_of_3_hit']:.0f}% of the time")
     for r in sorted(RANK, key=lambda r: -r["pairwise"])],
    ref=0.5, ref_label="coin flip 0.500",
    color=lambda k: ("var(--s2)" if k == "blind pick"
                     else "var(--s1)" if k in ("primary", "all_cues",
                                               "all_measured")
                     else "var(--s3)"))

best_cue = max((r for r in RANK if r["kind"] == "cue"),
               key=lambda r: r["pairwise_best"])

VERDICT = ("MET" if (crit_mae and crit_ci) else "NOT MET")

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Watermelon Phases 1A and 1B</title>
<style>
:root{{color-scheme:light;
 --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
 --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --line:rgba(11,11,11,.10);
 --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
 --good:#0ca30c; --crit:#d03b3b; --warn:#fab219;}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme=light])){{
 color-scheme:dark;
 --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
 --muted:#898781; --grid:#2c2c2a; --axis:#383835; --line:rgba(255,255,255,.10);
 --s1:#3987e5; --s2:#d95926; --s3:#199e70;}}}}
:root[data-theme=dark]{{color-scheme:dark;
 --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
 --muted:#898781; --grid:#2c2c2a; --axis:#383835; --line:rgba(255,255,255,.10);
 --s1:#3987e5; --s2:#d95926; --s3:#199e70;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--page);color:var(--ink);
 font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:900px;margin:0 auto;padding:40px 16px 80px}}
h1{{font-size:28px;margin:0 0 4px;letter-spacing:-.02em}}
h2{{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}}
h4{{font-size:14px;margin:0 0 6px;font-weight:600}}
.sub-head{{color:var(--ink2);margin:0 0 32px;font-size:14px}}
section{{background:var(--surface);border:1px solid var(--line);border-radius:12px;
 padding:22px 24px;margin:0 0 18px}}
.lede{{color:var(--ink2);font-size:14px;margin:2px 0 18px}}
.chart{{width:100%;height:auto;display:block;margin:4px 0 6px;overflow:visible}}
.lbl{{font-size:12px;fill:var(--ink2)}}
.val{{font-size:12px;fill:var(--ink2);font-variant-numeric:tabular-nums}}
.axlbl{{font-size:11px;fill:var(--muted)}}
.ref{{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 3}}
.grid{{stroke:var(--grid);stroke-width:1}}
.band{{fill:var(--muted);opacity:.11}}
.bar{{transition:opacity .12s}} .bar:hover{{opacity:.78}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0 2px}}
th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line)}}
th{{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;
 letter-spacing:.04em}}
.num td+td,.num th+th{{text-align:right;font-variant-numeric:tabular-nums}}
.verdict{{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 2px}}
.vcard{{flex:1;min-width:190px;border:1px solid var(--line);border-radius:10px;
 padding:14px 16px}}
.vcard .k{{font-size:11px;color:var(--muted);text-transform:uppercase;
 letter-spacing:.04em}}
.vcard .v{{font-size:24px;font-weight:650;margin:3px 0;letter-spacing:-.02em}}
.vcard .w{{font-size:12px;color:var(--ink2)}}
.pill{{display:inline-block;font-size:11px;padding:1px 7px;border-radius:20px;
 border:1px solid var(--line);vertical-align:1px;font-weight:600}}
.pill.ok{{color:var(--good);border-color:var(--good)}}
.pill.no{{color:var(--crit);border-color:var(--crit)}}
.hero{{font-size:34px;font-weight:700;letter-spacing:-.03em;margin:0}}
.hero.no{{color:var(--crit)}} .hero.ok{{color:var(--good)}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
 gap:18px}}
.cap{{font-size:12px;color:var(--ink2);margin:4px 0 0}}
code{{font-size:12px;background:var(--page);padding:1px 5px;border-radius:4px;
 border:1px solid var(--line)}}
ul{{margin:6px 0;padding-left:20px}} li{{margin:3px 0;font-size:13px}}
.bad{{color:var(--crit);font-weight:600}}
.hmtab td,.hmtab th{{font-size:11px;padding:4px 5px;text-align:center}}
.hmtab th{{text-transform:none;letter-spacing:0}}
.hm{{background:color-mix(in oklab,var(--c) calc(var(--a)*100%),var(--surface));
 font-variant-numeric:tabular-nums}}
.diag{{color:var(--muted)}}
.scroll{{overflow-x:auto}}
footer{{color:var(--muted);font-size:12px;text-align:center;margin-top:26px}}
</style></head><body><div class="wrap">

<h1>Evaluating Folk Heuristics for Choosing a Watermelon</h1>
<p class="sub-head">Phase 1A &mdash; does each cue matter on its own? &nbsp;&middot;&nbsp;
Phase 1B &mdash; does combining the cues predict sweetness better than guessing
the average? &nbsp;&middot;&nbsp; n = {R['n_melons']} melons
&nbsp;&middot;&nbsp; generated {datetime.now():%Y-%m-%d %H:%M}</p>

<section>
  <h2>Can it help you CHOOSE a melon?</h2>
  <p class="lede">MAE asks how close the predicted number is. A shopper never
  needs the number &mdash; they need to pick the better melon out of the few in
  front of them. That is a ranking question, and it is the one the project is
  actually about.</p>
  <ul>
    <li><b>Pairwise</b> &mdash; over all 300 melon pairs, how often the score
      puts the sweeter one first. Coin flip = 0.500.</li>
    <li><b>Best-of-3 gain</b> &mdash; pick 3 melons at random, choose one by the
      score; how many &deg;Brix sweeter than a blind pick. Chance = 0.000.</li>
    <li><b>Best-of-3 hit</b> &mdash; how often that choice is the actual
      sweetest of the three. Chance = 0.333.</li>
    <li><b>Within sonya / kinnaree</b> &mdash; the same pairwise question among
      melons of one variety only, which is what you face at one stall.</li>
  </ul>
  {rank_svg}
  <div class="scroll">{rank_tab}</div>
  <p class="cap">Single cues follow the <b>folk</b> direction, so below 0.500
  means following the folk rule is worse than guessing &mdash; the rule is
  backwards. <code>rind_gloss</code> at
  {[r for r in RANK if r['name']=='rind_gloss'][0]['pairwise']:.3f} is the clearest
  case: invert it and it becomes
  {1-[r for r in RANK if r['name']=='rind_gloss'][0]['pairwise']:.3f}.</p>
  <p class="cap"><b>The models do not win here.</b>
  <code>{escape(best_cue['name'])}</code> alone reaches
  {best_cue['pairwise_best']:.3f} pairwise and picks the sweetest of three
  {100*best_cue['best_of_3_hit']:.0f}% of the time, against 33% for a blind
  pick. Ridge shrinks predictions toward the mean, which lowers squared error
  but compresses the ordering &mdash; so optimising MAE actively costs you on
  the task that matters.</p>
</section>

<section>
  <h2>Phase 1A: does each cue matter on its own?</h2>
  <p class="lede">Spearman correlation of each cue column with Brix, and a
  permutation test: Brix shuffled 10,000 times. <i>p single</i> is how often chance
  beats this cue alone. <i>p best of 7</i> is how often the strongest of all 7 cues
  beats it by chance, which allows for having tested 7. Supported = p best of 7
  below 0.05; suggestive = only p single below 0.05.</p>
  {a1_svg}
  <div class="scroll">{a1_tab}</div>
  <p class="cap"><b>The verdict comes from the partial correlation</b>, because
  that is what the proposal preregistered: &ldquo;partial Spearman &rho; with
  &deg;Brix, controlling for variety&rdquo;. Plain &rho; is kept beside it
  because the gap between the two measures how much of an apparent effect is
  really a between-cultivar difference &mdash; drop it and a reader cannot tell
  whether variety mattered. The null permutes Brix <i>within</i> each variety,
  so it tests &ldquo;does this cue do anything beyond telling you which cultivar
  you bought?&rdquo;</p>
  <p class="cap">Applying the control costs <code>rind_gloss</code> its
  supported verdict (p best-of-7 0.036 &rarr; 0.089), so <b>no cue is supported
  under the preregistered analysis</b>. Note also that gloss nearly vanishes
  within kinnaree (+0.222 vs +0.682 in sonya), while
  <code>decay_20db_s</code> barely moves under the control and keeps its sign in
  both varieties. Source: <code>results/cue_results.csv</code>, written by
  <code>model/analysis_1a.py</code>.</p>
</section>

<section>
  <h2>Is gloss measuring the melon, or the lighting?</h2>
  <p class="lede"><code>rind_gloss</code> counts specular-highlight pixels,
  which depend on lamp angle and exposure as well as on the fruit &mdash; and
  the photos had no colour card and the lighting was not fixed. Photo filenames
  are sequential iPhone numbers, so <b>melon number is the shooting order</b>
  and the worry is directly testable.</p>
  <div class="scroll">{order_tab}</div>
  <p class="cap"><b>Cleared.</b> No cue drifts strongly with shooting order
  (worst |&rho;| = {max(abs(v) for v in C1A['cue_vs_order'].values()):.3f}),
  Brix itself does not drift
  (&rho; = {C1A['brix_vs_order']['rho']:+.3f}, p =
  {C1A['brix_vs_order']['p']:.3f}), and gloss&harr;Brix survives controlling for
  order at
  {C1A['cue_brix_controlling_order']['rind_gloss']:+.3f}, barely moved from
  {[r for r in A1 if r['column']=='rind_gloss'][0]['spearman_rho']:+.3f}. The
  &ldquo;lighting drifted and manufactured the correlation&rdquo; story is not
  supported.</p>
  <p class="cap"><b>But the design has a real weakness:</b> {blocks}. The two
  varieties were bought and photographed in blocks, so any session-level effect
  is inseparable from variety by construction. Interleave the two varieties
  across collection days next time.</p>
</section>

<section>
  <h2>Phase 1B: verdict</h2>
  <p class="lede">Both preregistered criteria had to be met.</p>
  <p class="hero {'ok' if VERDICT=='MET' else 'no'}">Success bar {VERDICT}</p>
  <div class="verdict">
    <div class="vcard"><div class="k">MAE vs 15% bar</div>
      <div class="v">{PRIMARY_MAE:.3f}</div>
      <div class="w">bar is {BAR:.3f} &middot;
      <span class="pill {'ok' if crit_mae else 'no'}">
      {'pass' if crit_mae else 'miss'}</span></div></div>
    <div class="vcard"><div class="k">95% CI on improvement</div>
      <div class="v">{BOOT['primary']['improvement']:+.3f}</div>
      <div class="w">[{BOOT['primary']['ci_lo']:+.3f},
      {BOOT['primary']['ci_hi']:+.3f}] &middot;
      <span class="pill {'ok' if crit_ci else 'no'}">
      {'excludes 0' if crit_ci else 'includes 0'}</span></div></div>
    <div class="vcard"><div class="k">Permutation test</div>
      <div class="v">p = {PERM['primary']['p']:.3f}</div>
      <div class="w">not part of the bar &middot;
      <span class="pill {'ok' if crit_perm else 'no'}">
      {'signal' if crit_perm else 'no signal'}</span></div></div>
  </div>
  <p class="cap"><b>Reading:</b> the shuffled-label test says the model finds
  something real, but the melon-level confidence interval cannot rule out zero
  improvement and the error does not reach the 15% bar. At n={R['n_melons']} that
  is what a modest true effect looks like. Baseline MAE is {BASE:.3f} &deg;Brix;
  Brix itself ranges {R['brix']['min']:.1f}&ndash;{R['brix']['max']:.1f}
  (sd {R['brix']['sd']:.2f}).</p>
</section>

<section>
  <h2>What was measured</h2>
  <p class="lede">One pre-committed column per cue, fixed before any result was
  read. The sound cue is two features because pitch and decay are different
  physics &mdash; they correlate &rho; = +0.01 with each other.</p>
  <div class="grid2">
    <div><h4>Cues tested &mdash; {len(R['cue_columns'])} cues,
      {sum(len(v) for v in R['cue_columns'].values())} columns</h4>
      <ul>{cue_rows}</ul></div>
    <div><h4>Cues with no column</h4><ul>{absent_rows}</ul></div>
  </div>
</section>

<section>
  <h2>What is in each feature set</h2>
  <p class="lede">Every set is a fixed list of columns, chosen before any result
  was read. Nothing here was selected by how well it scored.</p>
  <div class="scroll">{defs_table}</div>
  <p class="cap"><b>all_cues vs all_measured:</b> <code>all_cues</code> is one
  pre-committed column per cue. <code>all_measured</code> adds the
  {len(extra)} complete columns that are not any cue&rsquo;s representative
  &mdash; {', '.join(f'<code>{escape(c)}</code>' for c in extra)} &mdash; so it
  tests whether reducing each cue to a single number lost anything.</p>
</section>

<section>
  <h2>Feature sets</h2>
  <p class="lede">Lower is better. The dashed line is the 15% success bar.
  Orange is the baseline; green are ablation variants, which are diagnostics and
  not candidate headline models.</p>
  {set_bars}
  <div class="scroll">{set_table}</div>
  <p class="cap">Dots, not bars: every value sits between 0.93 and 1.12, and
  bars &mdash; which have to start at zero &mdash; would render twelve
  near-identical lengths. The whisker is &plusmn;1 fold sd, and it is the point:
  most of the gaps between these sets are smaller than their own spread.</p>
</section>

<section>
  <h2>Improvement over the baseline, with 95% CI</h2>
  <p class="lede"><b>What is being sampled: melons.</b> For each of the
  {R['n_melons']} melons we have two numbers &mdash; how far the model missed,
  and how far the baseline missed. Subtract them and you get that
  melon&rsquo;s improvement. Then draw {R['n_melons']} melons <i>with
  replacement</i> from those {R['n_melons']} (some appear twice, some not at
  all), average their improvements, and repeat 5,000 times. The middle 95% of
  those averages is the bar below.</p>
  <p class="lede">It answers: <b>if we had bought a different
  {R['n_melons']} melons from the same market, would the model still have beaten
  guessing?</b> A bar crossing the dashed line means no &mdash; not reliably.</p>
  {forest_svg}
  <p class="cap">Averaging over 20 splits before resampling matters: built on a
  single split this same test reported +22.7% for <code>primary</code> while the
  100-fold table said &minus;13.0%, because that one split happened to be kind.
  Positive numbers mean the model beat the baseline by that many &deg;Brix.</p>
</section>

<section>
  <h2>Permutation null</h2>
  <p class="lede"><b>What is being shuffled: the sweetness values.</b> Take the
  {R['n_melons']} Brix readings and randomly reassign them to melons &mdash;
  melon 1 gets melon 17&rsquo;s sweetness. The cue measurements stay with their
  own melon; only the answers move. Now there is, by construction, no real
  relationship left. Refit, cross-validate, record the error. Repeat
  {len(R['permutation_null'][0]['null'])} times.</p>
  <p class="lede">That builds a picture of <b>what &ldquo;nothing&rdquo; scores
  on your own data</b>, rather than assuming a formula. The grey histogram is
  those shuffled runs; the coloured line is the real model. <code>p</code> is
  the fraction of shuffles that did at least as well.</p>
  <div class="grid2">{perm_cards}</div>
  <p class="cap"><b>Why this and the bootstrap disagree:</b> they ask different
  questions. The permutation asks <i>&ldquo;could this be nothing at
  all?&rdquo;</i> &mdash; and for <code>primary</code>, probably not (p =
  {PERM['primary']['p']:.3f}). The bootstrap asks <i>&ldquo;how big is it, and
  is it reliably above zero from melon to melon?&rdquo;</i> &mdash; and there
  the answer is that it cannot be pinned above zero. A small but real effect
  produces exactly this pattern.</p>
</section>

<section>
  <h2>Per-cue ablation</h2>
  <p class="lede">Each cue removed from the full set in turn. Positive means
  removing it made predictions worse, so it was helping. Dropping the sound cue
  removes both its columns &mdash; the question is what the knock contributes,
  not what pitch adds given decay.</p>
  {abl_svg}
  <p class="cap">The shaded band is &plusmn;1 fold sd
  ({R['ablation_fold_sd']:.3f}). Every cue falls inside it, so all six read
  &ldquo;cannot distinguish&rdquo; &mdash; which is not the same as
  &ldquo;does not matter&rdquo;.</p>
</section>

<section>
  <h2>Does it work on an unseen variety?</h2>
  <p class="lede">Trained on one variety, tested on the other. This stands in for
  the preregistered leave-one-vendor-out, since no vendor was recorded.</p>
  {lovo}
  <p class="cap">The means sit at or above the {BASE:.3f} baseline: shown a
  variety it has never seen, the model is no better than guessing. Only two
  groups, so this is coarse &mdash; but it is the clearest limitation in the
  result.</p>
</section>

<section>
  <h2>Predicted vs actual</h2>
  <p class="lede">Out-of-fold predictions from <code>primary</code>, averaged over
  20 splits. The dashed line is perfect prediction.</p>
  {scatter(R['per_melon'])}
  <p class="cap">The cloud is flatter than the diagonal &mdash; the model pulls
  predictions toward the mean, which is what shrinkage does when the features are
  only weakly informative.</p>
</section>

<section>
  <h2>Would more melons help?</h2>
  <p class="lede">Subsamples of the data, re-evaluated. The band is &plusmn;1 sd
  across 40 draws.</p>
  {linechart(R['learning_curve'])}
  <p class="cap">Still falling at n={R['n_melons']}, so more melons would buy real
  accuracy rather than just narrower error bars. This is the single highest-value
  thing left to do.</p>
</section>

<section>
  <h2>Is ridge the bottleneck?</h2>
  <p class="lede">Six model families, same cross-validation budget as the headline
  table.</p>
  {fam}
  <p class="cap">At n={R['n_melons']} the algorithm is not the limiting factor,
  the data is. Tree models lose to a shrunken linear model, as expected at this
  size. PLS edges ahead &mdash; it is the standard method in fruit
  near-infrared work &mdash; but it was not the preregistered model, so promoting
  it now would be selection after seeing results.</p>
</section>

<section>
  <h2>Checks against the obvious objections</h2>
  <div class="grid2">
    <div><h4>Is it really just variety?</h4>{variety_tab}
      <p class="cap">No. The label alone predicts nothing and adding it hurts.
      Transfer still fails across varieties, which points to the cue&ndash;Brix
      relationship differing by cultivar rather than the cues proxying the
      label.</p></div>
    <div><h4>How hard is ridge shrinking?</h4>{shrink_tab}
      <p class="cap">Ridge at &alpha;&nbsp;&rarr;&nbsp;&infin; <i>is</i> the mean
      baseline, so a moderate &alpha; says cross-validation chose to actually use
      the features.</p></div>
  </div>
</section>

<section>
  <h2>Are the cues measuring the same thing?</h2>
  <p class="lede">Spearman between cue columns. This is the diagnostic that
  explains any gap between a cue&rsquo;s own correlation and its ablation.</p>
  <div class="scroll">{heatmap(R['cue_correlation']['columns'],
                               R['cue_correlation']['matrix'])}</div>
  <h4>Pairs above |&rho;| 0.5</h4><ul>{red_txt}</ul>
</section>

<section>
  <h2>Per-melon predictions</h2>
  <p class="lede">Sorted by absolute error, worst first. Useful for the live demo
  &mdash; and for spotting whether the misses are systematic.</p>
  <div class="scroll">{per_melon_tab}</div>
</section>

<section>
  <h2>Caveats</h2>
  <ul>{''.join(f'<li>{escape(c)}</li>' for c in R['caveats'])}</ul>
</section>

<footer>Built by report.py from results_1b.json and results/cue_results.csv &middot;
rerun: <code>python train_1b.py &amp;&amp; python report.py</code></footer>
</div></body></html>"""

out = HERE / "report.html"
out.write_text(HTML, encoding="utf-8")
print(f"wrote {out}  ({len(HTML)/1024:.0f} KB)")
print(f"verdict: success bar {VERDICT}  "
      f"(MAE {'pass' if crit_mae else 'miss'}, "
      f"CI {'excludes 0' if crit_ci else 'includes 0'}, "
      f"permutation p = {PERM['primary']['p']:.3f})")
print("\nserve it:  python -m http.server 8777 --directory .")
