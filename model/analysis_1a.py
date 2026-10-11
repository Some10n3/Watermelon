"""Phase 1A: does each cue matter on its own?

Supersedes notebook step 26 as the writer of results/cue_results.csv. Two
additions over that cell:

1. PARTIAL Spearman controlling for variety. The proposal specifies "partial
   Spearman rho with Brix, controlling for variety"; the notebook reported plain
   Spearman. The partial version is the preregistered one, so it carries the
   verdict. Plain is kept beside it, because the gap between them measures how
   much of an apparent effect is really a between-cultivar difference - delete it
   and a reader cannot tell whether variety mattered at all.

2. An acquisition-order check. rind_gloss counts specular-highlight pixels,
   which depend on lamp angle and exposure as well as on the fruit; the photos
   had no colour card and the lighting was not fixed. Filenames are sequential
   iPhone numbers, so melon_id IS the shooting order and the confound is
   directly testable.

The null for the partial test permutes Brix WITHIN each variety. That destroys
the cue-Brix link inside a cultivar while leaving the cultivar-Brix difference
intact, which is exactly the hypothesis "this cue does nothing beyond telling
you which variety you bought".

Run:  python model/analysis_1a.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

N_PERM = 10_000
SEED = 42

cues = json.loads((ROOT / "cues.json").read_text(encoding="utf-8"))
data = pd.read_csv(ROOT / "watermelon_features.csv")
Y = "Brix sum"

FEATS, FOLK, CUE_OF = [], {}, {}
for cue, spec in cues["cues"].items():
    for col in spec["columns"]:
        FEATS.append(col)
        FOLK[col] = spec["folk_direction"]
        CUE_OF[col] = cue

d = data[FEATS + [Y, "type", "no"]].dropna().sort_values("no").reset_index(drop=True)
y = d[Y].to_numpy(float)
variety = d["type"].to_numpy()
order = d["no"].to_numpy(float)
z = (variety == "sonya").astype(float)
n, k = len(d), len(FEATS)


def unit(a: np.ndarray) -> np.ndarray:
    """Centre and scale to unit norm, column-wise, so a dot product is r."""
    a = a - a.mean(axis=0)
    return a / np.linalg.norm(a, axis=0)


def residual_maker(*covars) -> np.ndarray:
    A = np.c_[np.ones(n), *covars]
    return np.eye(n) - A @ np.linalg.pinv(A)


Rf = stats.rankdata(d[FEATS].to_numpy(float), axis=0)
ry = stats.rankdata(y)

# plain: Spearman is Pearson on ranks
plain = unit(ry) @ unit(Rf)

# partial: the same, after linearly removing the variety rank from both sides
P_var = residual_maker(stats.rankdata(z))
partial = unit(P_var @ ry) @ unit(P_var @ Rf)

# acquisition order, for the confound check
P_ord = residual_maker(stats.rankdata(order))
vs_order = unit(stats.rankdata(order)) @ unit(Rf)
partial_order = unit(P_ord @ ry) @ unit(P_ord @ Rf)

rng = np.random.default_rng(SEED)
strata = [np.flatnonzero(variety == v) for v in np.unique(variety)]

null_plain = np.empty((N_PERM, k))
null_part = np.empty((N_PERM, k))
Uf_plain, Uf_part = unit(Rf), unit(P_var @ Rf)
for i in range(N_PERM):
    null_plain[i] = unit(rng.permutation(ry)) @ Uf_plain
    # restricted permutation: shuffle Brix inside each variety only
    s = ry.copy()
    for ix in strata:
        s[ix] = rng.permutation(s[ix])
    null_part[i] = unit(P_var @ s) @ Uf_part

best_plain = np.abs(null_plain).max(axis=1)
best_part = np.abs(null_part).max(axis=1)


def verdict(p_family: float, p_single: float) -> str:
    if p_family < 0.05:
        return "supported"
    if p_single < 0.05:
        return "suggestive"
    return "no evidence"


rows = []
for j, f in enumerate(FEATS):
    ps_pl = float((np.abs(null_plain[:, j]) >= abs(plain[j])).mean())
    pf_pl = float((best_plain >= abs(plain[j])).mean())
    ps_pa = float((np.abs(null_part[:, j]) >= abs(partial[j])).mean())
    pf_pa = float((best_part >= abs(partial[j])).mean())
    sub = {v: float(stats.spearmanr(g[f], g[Y]).statistic)
           for v, g in d.groupby("type")}
    rows.append(dict(
        column=f, cue=CUE_OF[f],
        spearman_rho=round(float(plain[j]), 4),
        p_single=round(ps_pl, 4), p_best_of_7=round(pf_pl, 4),
        partial_rho=round(float(partial[j]), 4),
        partial_p_single=round(ps_pa, 4), partial_p_best_of_7=round(pf_pa, 4),
        rho_sonya=round(sub.get("sonya", np.nan), 4),
        rho_kinnaree=round(sub.get("kinnaree", np.nan), 4),
        rho_vs_order=round(float(vs_order[j]), 4),
        verdict_plain=verdict(pf_pl, ps_pl),
        verdict=verdict(pf_pa, ps_pa),            # preregistered: partial decides
        folk=("flat" if abs(partial[j]) < 0.1 else
              "as claimed" if (partial[j] > 0) == (FOLK[f] > 0) else "OPPOSITE"),
    ))

res = pd.DataFrame(rows).sort_values("partial_p_single").reset_index(drop=True)
res.to_csv(OUT / "cue_results.csv", index=False)

print(f"{n} melons, {k} cue columns, {N_PERM} permutations\n")
print(f"{'column':22s} {'plain':>7s} {'partial':>8s} {'p_fam':>7s} "
      f"{'sonya':>7s} {'kinn':>7s} {'order':>7s}  verdict")
print("-" * 92)
for _, r in res.iterrows():
    flag = "  <-- changed" if r.verdict != r.verdict_plain else ""
    print(f"{r.column:22s} {r.spearman_rho:+7.3f} {r.partial_rho:+8.3f} "
          f"{r.partial_p_best_of_7:7.3f} {r.rho_sonya:+7.3f} "
          f"{r.rho_kinnaree:+7.3f} {r.rho_vs_order:+7.3f}  "
          f"{r.verdict}{flag}")

changed = res[res.verdict != res.verdict_plain]
print(f"\nverdict changed by the variety control for {len(changed)} of {k} cues"
      + (": " + ", ".join(changed.column) if len(changed) else ""))

# ---------------------------------------------------------------- confound
brix_order = stats.spearmanr(order, y)
print(f"\n{'=' * 70}\nACQUISITION-ORDER CHECK\n{'=' * 70}")
print("melon_id is the shooting order (sequential iPhone filenames).")
for v in np.unique(variety):
    ids = sorted(d.loc[variety == v, "no"].astype(int))
    print(f"  {v:9s} n={len(ids):2d}  ids {ids}")
print(f"\n  Brix vs order          rho {brix_order.statistic:+.3f}  "
      f"p {brix_order.pvalue:.4f}")
for j, f in enumerate(FEATS):
    print(f"  {f:22s} vs order rho {vs_order[j]:+.3f}")
print(f"\n  gloss<->Brix controlling for order: "
      f"rho {partial_order[FEATS.index('rind_gloss')]:+.3f}")

confound = dict(
    brix_vs_order=dict(rho=round(float(brix_order.statistic), 4),
                       p=round(float(brix_order.pvalue), 4)),
    cue_vs_order={f: round(float(vs_order[j]), 4) for j, f in enumerate(FEATS)},
    cue_brix_controlling_order={f: round(float(partial_order[j]), 4)
                                for j, f in enumerate(FEATS)},
    variety_blocks={v: [int(x) for x in sorted(d.loc[variety == v, "no"])]
                    for v in np.unique(variety)},
    note=("Photo filenames are sequential iPhone numbers, so melon_id is the "
          "shooting order. The two varieties were bought and shot in blocks, so "
          "any session-level effect is inseparable from variety by design."),
)
(OUT / "analysis_1a.json").write_text(json.dumps(
    dict(n_melons=n, n_perm=N_PERM, cues=rows, confound=confound), indent=2),
    encoding="utf-8")

print(f"\nwrote {OUT / 'cue_results.csv'}")
print(f"wrote {OUT / 'analysis_1a.json'}")
print("\nNOTE: this file now writes results/cue_results.csv. Notebook step 26")
print("writes the same path with plain Spearman only - disable that cell, or")
print("whichever ran last silently wins.")
