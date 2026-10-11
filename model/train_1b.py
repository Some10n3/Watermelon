"""Phase 1B: does combining the cues predict Brix better than guessing the mean?

Run:  python model/train_1b.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import (KFold, LeaveOneGroupOut, RepeatedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SEED = 0
N_PERM = 1000
N_BOOT = 5000
ALPHAS = np.logspace(-2, 4, 30)

SEARCH = list(dict.fromkeys(
    [HERE, HERE.parent, HERE.parent / "model", HERE / "model"]))


def locate(name: str) -> Path:
    """Find a data file beside this script, at the repo root, or in model/; newest wins."""
    found = [d / name for d in SEARCH if (d / name).is_file()]
    if not found:
        raise FileNotFoundError(
            f"{name} not found. Looked in:\n  "
            + "\n  ".join(str(d) for d in SEARCH)
            + "\nRe-run the last cells of watermelon_features.ipynb to create it.")
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(found) > 1:
        print(f"WARNING: {len(found)} copies of {name}; using the newest")
        for p in found:
            print(f"  {'USING ' if p is found[0] else '      '}{p}")
    return found[0]


cues = json.loads(locate("cues.json").read_text(encoding="utf-8"))
df = pd.read_csv(locate("watermelon_training.csv"))
TARGET, ID = cues["target"], cues["id_column"]

assert len(df) == df[ID].nunique(), "one row per melon broken"
y = df[TARGET].to_numpy(float)

CUE_COLS = {k: v["columns"] for k, v in cues["cues"].items()}
PRIMARY = [k for k, v in cues["cues"].items() if v["primary"]]
ALL_CUE_COLS = [c for cols in CUE_COLS.values() for c in cols]
COMPLETE = cues["all_complete_columns"]

variety = (pd.read_csv(locate("watermelon_features.csv"))[["no", "type"]]
           .rename(columns={"no": ID}).set_index(ID)["type"].reindex(df[ID]).to_numpy())

SETS: dict[str, list[str]] = {
    "baseline_mean": [],
    "primary":       [c for k in PRIMARY for c in CUE_COLS[k]],
    "all_cues":      ALL_CUE_COLS,
    "visual_only":   [c for k, cols in CUE_COLS.items() if k != "sound" for c in cols],
    "audio_only":    CUE_COLS["sound"],
    "all_measured":  COMPLETE,
}
SETS |= {f"minus_{k}": [c for k2, cols in CUE_COLS.items() if k2 != k for c in cols]
         for k in CUE_COLS}

OUTER = RepeatedKFold(n_splits=5, n_repeats=20, random_state=SEED)
OOF_CV = KFold(5, shuffle=True, random_state=SEED)


def model_for(cols: list[str]):
    """Scaled ridge; the scaler is inside the pipeline so each fold scales on its own training data."""
    if not cols:
        return DummyRegressor(strategy="mean")
    return Pipeline([("scale", StandardScaler()),
                     ("ridge", RidgeCV(alphas=ALPHAS))])


def design(cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(float) if cols else np.zeros((len(df), 1))


def oof_predict(cols: list[str]) -> np.ndarray:
    return cross_val_predict(model_for(cols), design(cols), y, cv=OOF_CV)


def evaluate(cols: list[str]) -> dict:
    X, m = design(cols), model_for(cols)
    s = cross_validate(m, X, y, cv=OUTER, scoring=["neg_mean_absolute_error",
                                                   "neg_root_mean_squared_error"])
    mae = -s["test_neg_mean_absolute_error"]
    if cols:
        rho = spearmanr(oof_predict(cols), y).statistic
    else:
        rho = np.nan
    return dict(n_feat=len(cols), mae=mae.mean(), mae_sd=mae.std(),
                rmse=-s["test_neg_root_mean_squared_error"].mean(), rho_oof=rho,
                mae_folds=mae.tolist())


def permutation_p(cols: list[str], real_mae: float) -> tuple[float, float]:
    """Shuffle the labels and rerun the whole pipeline to get a null MAE distribution."""
    X, m = design(cols), model_for(cols)
    rng = np.random.default_rng(SEED)
    null = np.array([
        -cross_validate(m, X, rng.permutation(y), cv=OOF_CV,
                        scoring="neg_mean_absolute_error")["test_score"].mean()
        for _ in range(N_PERM)])
    return null, float((null <= real_mae).mean())


def repeated_oof(cols: list[str], n_rep: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Per-melon out-of-fold prediction and absolute error, averaged over n_rep 5-fold splits."""
    preds = np.zeros((n_rep, len(y)))
    for r in range(n_rep):
        preds[r] = cross_val_predict(model_for(cols), design(cols), y,
                                     cv=KFold(5, shuffle=True, random_state=SEED + r))
    mean_pred = preds.mean(axis=0)
    mean_err = np.abs(y[None, :] - preds).mean(axis=0)
    return mean_pred, mean_err


def bootstrap_improvement(cols: list[str], base_err: np.ndarray) -> dict:
    """PREREGISTERED: the success bar requires 'a CI that excludes no improvement'."""
    err = repeated_oof(cols)[1]
    diff = base_err - err
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(improvement=float(diff.mean()), ci_lo=float(lo), ci_hi=float(hi),
                excludes_zero=bool(lo > 0),
                pct_of_baseline=float(100 * diff.mean() / base_err.mean()))


print(f"Phase 1B - {len(df)} melons, target {TARGET}")
print(f"{len(CUE_COLS)} cues in {len(ALL_CUE_COLS)} columns; "
      f"primary = {', '.join(PRIMARY)}")
for cue, why in cues["absent_cues"].items():
    print(f"  absent cue '{cue}': {why}")

rows = {name: evaluate(cols) for name, cols in SETS.items()}
base = rows["baseline_mean"]["mae"]
# 15% bar from the cross-validated baseline, not the in-sample one in cues.json
target = 0.85 * base
print(f"\nbaseline: cross-validated {base:.3f}, in-sample "
      f"{cues['mean_baseline_mae']:.3f} degBrix")
print(f"15% success bar: {target:.3f} (CV-consistent); "
      f"cues.json stated {cues['target_mae_15pct']:.3f} from the in-sample figure")

res = (pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "mae_folds"}
                     for k, v in rows.items()}).T
       .assign(vs_base_pct=lambda d: 100 * (d.mae - base) / base)
       .sort_values("mae"))

print(f"\n{'=' * 74}\nFEATURE SETS   (baseline MAE {base:.3f}, "
      f"15% target {target:.3f} degBrix)\n{'=' * 74}")
print(f"{'set':16s} {'n':>2s} {'MAE':>7s} {'+-sd':>6s} {'RMSE':>6s} "
      f"{'rho_oof':>7s} {'vs base':>8s}")
print("-" * 74)
for name, r in res.iterrows():
    hit = " <= TARGET" if r.mae <= target and name != "baseline_mean" else ""
    print(f"{name:16s} {int(r.n_feat):2d} {r.mae:7.3f} {r.mae_sd:6.3f} "
          f"{r.rmse:6.3f} {r.rho_oof:+7.3f} {r.vs_base_pct:+7.1f}%{hit}")

print(f"\n{'=' * 74}\nPER-CUE ABLATION   (vs all_cues MAE "
      f"{rows['all_cues']['mae']:.3f})\n{'=' * 74}")
full, full_sd = rows["all_cues"]["mae"], rows["all_cues"]["mae_sd"]
abl = []
for cue in CUE_COLS:
    d = rows[f"minus_{cue}"]["mae"] - full
    abl.append((cue, d, "contributes" if d > 2 * full_sd
                else "cannot distinguish" if abs(d) < full_sd
                else "unclear"))
print(f"{'cue removed':14s} {'dMAE':>7s}   reading      (fold sd {full_sd:.3f})")
print("-" * 74)
for cue, d, verdict in sorted(abl, key=lambda t: -t[1]):
    print(f"{cue:14s} {d:+7.3f}   {verdict}")
print("\ndMAE > 0 means removing the cue made predictions worse, i.e. it helped.")
print("At n=25 most rows land in 'cannot distinguish' - that is the honest")
print("outcome, and it is not the same as 'the cue does not matter'.")

print(f"\n{'=' * 74}\nBOOTSTRAP 95% CI ON THE IMPROVEMENT   ({N_BOOT} resamples)")
print(f"preregistered: the bar needs a CI that excludes no improvement\n{'=' * 74}")
base_err = repeated_oof([])[1]
boot = {}
for name in ("primary", "all_cues", "visual_only", "audio_only", "all_measured"):
    b = bootstrap_improvement(SETS[name], base_err)
    boot[name] = b
    mark = "EXCLUDES 0" if b["excludes_zero"] else "includes 0"
    print(f"{name:14s} {b['improvement']:+.3f} degBrix "
          f"[{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}]  "
          f"({b['pct_of_baseline']:+.1f}%)  {mark}")

print(f"\n{'=' * 74}\nPERMUTATION NULL   ({N_PERM} shuffles)\n{'=' * 74}")
perm = []
for name in ("primary", "all_cues", "audio_only", "visual_only"):
    null, p = permutation_p(SETS[name], rows[name]["mae"])
    flag = "signal" if p < 0.05 else "indistinguishable from noise"
    perm.append(dict(set=name, real_mae=rows[name]["mae"],
                     null_median=float(np.median(null)), p=p, reading=flag,
                     null=null.tolist()))
    print(f"{name:14s} real {rows[name]['mae']:.3f}   null median "
          f"{np.median(null):.3f}   p = {p:.3f}   -> {flag}")

print(f"\n{'=' * 74}\nLEAVE-ONE-VARIETY-OUT   (stands in for leave-one-vendor-out;\n"
      f"no vendor was recorded)\n{'=' * 74}")
transfer = []
for name in ("primary", "all_cues", "visual_only"):
    X, m = design(SETS[name]), model_for(SETS[name])
    s = cross_validate(m, X, y, groups=variety, cv=LeaveOneGroupOut(),
                       scoring="neg_mean_absolute_error")
    mae = -s["test_score"]
    per_v = dict(zip(sorted(set(variety)), (round(float(e), 4) for e in mae)))
    transfer.append(dict(set=name, per_variety=per_v, mean=float(mae.mean())))
    print(f"{name:14s} " + "  ".join(f"test={v}: {e:.3f}" for v, e in per_v.items())
          + f"   mean {mae.mean():.3f}")
print("\nTrained on one variety, tested on the other. If this collapses while")
print("5-fold looked fine, the model learned the variety rather than the melon.")

print(f"\n{'=' * 74}\nMODEL FAMILY CHECK   (is ridge the bottleneck?)\n{'=' * 74}")
FAMILIES = {
    "ridge":        lambda: RidgeCV(alphas=ALPHAS),
    "lasso":        lambda: LassoCV(n_alphas=50, max_iter=20000, random_state=SEED),
    "elasticnet":   lambda: ElasticNetCV(n_alphas=50, max_iter=20000, random_state=SEED),
    "pls2":         lambda: PLSRegression(n_components=2),
    "randomforest": lambda: RandomForestRegressor(n_estimators=300, random_state=SEED),
    "gradboost":    lambda: GradientBoostingRegressor(random_state=SEED),
}
families = []
for fname, mk in FAMILIES.items():
    row = {"model": fname}
    for sname in ("primary", "all_cues"):
        pipe = Pipeline([("scale", StandardScaler()), ("est", mk())])
        mae = -cross_validate(pipe, design(SETS[sname]), y, cv=OUTER,
                              scoring="neg_mean_absolute_error")["test_score"]
        row[sname] = float(mae.mean())
    families.append(row)
    print(f"{fname:14s} primary {row['primary']:.3f}   all_cues {row['all_cues']:.3f}")
print("\nAt n=25 the algorithm is not the limiting factor; the data is. Tree")
print("models should lose to a shrunken linear model at this sample size.")

print(f"\n{'=' * 74}\nREGULARISATION STRENGTH\n{'=' * 74}")
shrink = {}
for name in ("primary", "all_cues", "all_measured"):
    p = model_for(SETS[name]).fit(design(SETS[name]), y)
    a = float(p.named_steps["ridge"].alpha_)
    coefs = dict(zip(SETS[name], (round(float(c), 4)
                                  for c in p.named_steps["ridge"].coef_)))
    shrink[name] = dict(alpha=a, coefs=coefs)
    print(f"{name:14s} alpha = {a:9.3f}   "
          f"({'heavy shrinkage - near the mean baseline' if a > 100 else 'moderate'})")
print("\nRidge at alpha -> infinity IS the mean baseline, so a large alpha says")
print("cross-validation chose to trust the features only weakly.")

print(f"\n{'=' * 74}\nDOES VARIETY EXPLAIN IT?\n{'=' * 74}")
is_sonya = (variety == "sonya").astype(float)[:, None]
var_only = -cross_validate(DummyRegressor(strategy="mean"), is_sonya, y, cv=OUTER,
                           scoring="neg_mean_absolute_error")["test_score"].mean()
pipe_v = Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])
mae_v = -cross_validate(pipe_v, is_sonya, y, cv=OUTER,
                        scoring="neg_mean_absolute_error")["test_score"].mean()
X_plus = np.hstack([design(SETS["all_cues"]), is_sonya])
mae_plus = -cross_validate(pipe_v, X_plus, y, cv=OUTER,
                           scoring="neg_mean_absolute_error")["test_score"].mean()
vtest = dict(variety_only=float(mae_v), all_cues=float(rows["all_cues"]["mae"]),
             all_cues_plus_variety=float(mae_plus))
print(f"variety label alone      MAE {mae_v:.3f}")
print(f"all_cues                 MAE {rows['all_cues']['mae']:.3f}")
print(f"all_cues + variety       MAE {mae_plus:.3f}")
print("\nIf the variety label alone rivals the cues, the cues may be measuring")
print("which cultivar you bought rather than how ripe the fruit is.")

print(f"\n{'=' * 74}\nLEARNING CURVE   (would more melons help?)\n{'=' * 74}")
rng = np.random.default_rng(SEED)
curve = []
for n in (12, 15, 18, 21, 25):
    vals = []
    for _ in range(40):
        sub = rng.choice(len(y), size=n, replace=False)
        Xs, ys = design(SETS["primary"])[sub], y[sub]
        try:
            vals.append(-cross_validate(
                model_for(SETS["primary"]), Xs, ys,
                cv=KFold(5, shuffle=True, random_state=int(rng.integers(1e6))),
                scoring="neg_mean_absolute_error")["test_score"].mean())
        except Exception:
            pass
    curve.append(dict(n=n, mae=float(np.mean(vals)), sd=float(np.std(vals))))
    print(f"n = {n:2d}   MAE {np.mean(vals):.3f} +- {np.std(vals):.3f}")
print("\nA curve still falling at n=25 means more melons would buy real accuracy.")

oof_primary = repeated_oof(SETS["primary"])[0]
per_melon = [dict(melon_id=int(m), variety=str(v), actual=float(a),
                  predicted=float(p), error=float(p - a))
             for m, v, a, p in zip(df[ID], variety, y, oof_primary)]

# MAE asks "how close is the number". A shopper never needs the number - they
# need to pick the better melon out of the few in front of them, which is a
# ranking question. The two come apart: ridge shrinks predictions toward the
# mean, which lowers squared error but compresses the ordering.
#
# Single cues are oriented by their FOLK direction, not by the sign we observed,
# so these answer "does following this folk rule help?". Below 0.5 means
# following it is worse than guessing - the rule is backwards.
import itertools

PAIRS = list(itertools.combinations(range(len(y)), 2))


def pairwise(score: np.ndarray, mask=None) -> float:
    """Concordance. A tie in the score scores 0.5, the usual convention - without
    it an all-ties scorer lands wherever the index order happens to fall rather
    than on 0.500."""
    ok = tot = 0.0
    for i, j in PAIRS:
        if y[i] == y[j] or (mask is not None and not (mask[i] and mask[j])):
            continue
        tot += 1
        if score[i] == score[j]:
            ok += 0.5
        else:
            ok += (score[i] > score[j]) == (y[i] > y[j])
    return ok / tot if tot else float("nan")


def best_of_k(score: np.ndarray, k: int = 3, trials: int = 20_000):
    r = np.random.default_rng(SEED)
    gain, hit = [], 0
    for _ in range(trials):
        ix = r.choice(len(y), size=k, replace=False)
        pick = ix[np.argmax(score[ix])]
        gain.append(y[pick] - y[ix].mean())
        hit += int(y[pick] == y[ix].max())
    return float(np.mean(gain)), hit / trials


def rank_row(name, score, kind, rho=None):
    pw = pairwise(score)
    g, h = best_of_k(score)
    w = {v: pairwise(score, variety == v) for v in np.unique(variety)}
    return dict(name=name, kind=kind, pairwise=pw, best_of_3_gain=g,
                best_of_3_hit=h, rho_oof=rho,
                pairwise_sonya=w.get("sonya"), pairwise_kinnaree=w.get("kinnaree"),
                direction=("follow the folk rule" if pw >= 0.5
                           else "do the OPPOSITE of the folk rule"),
                pairwise_best=max(pw, 1 - pw))


ranking = []
for sname in ("primary", "all_cues", "all_measured"):
    pred = repeated_oof(SETS[sname])[0]
    ranking.append(rank_row(sname, pred, "model",
                            rho=float(spearmanr(pred, y).statistic)))
for _cue, _spec in cues["cues"].items():
    for _col in _spec["columns"]:
        ranking.append(rank_row(_col, _spec["folk_direction"]
                                * df[_col].to_numpy(float), "cue"))
ranking.append(rank_row("blind pick", np.zeros(len(y)), "baseline"))

print(f"\n{'=' * 78}\nRANKING - can it help someone CHOOSE?\n{'=' * 78}")
print(f"{'scorer':22s} {'kind':9s} {'pairwise':>9s} {'bo3 gain':>9s} "
      f"{'bo3 hit':>8s} {'rho_oof':>8s} {'sonya':>7s} {'kinn':>7s}")
print("-" * 78)
for r in sorted(ranking, key=lambda r: -r["pairwise_best"]):
    rho = f"{r['rho_oof']:+8.3f}" if r["rho_oof"] is not None else " " * 8
    print(f"{r['name']:22s} {r['kind']:9s} {r['pairwise']:9.3f} "
          f"{r['best_of_3_gain']:+9.3f} {r['best_of_3_hit']:8.3f} {rho} "
          f"{r['pairwise_sonya']:7.3f} {r['pairwise_kinnaree']:7.3f}")
print("\nchance: pairwise 0.500, gain 0.000, hit 0.333 (best of 3)")
print("Cues follow the FOLK direction, so below 0.500 means the rule is backwards.")
print("sonya/kinn columns ask whether it still works among melons of ONE variety,")
print("which is the question a shopper at one stall actually faces.")

corr = df[ALL_CUE_COLS].corr(method="spearman").round(3)
print(f"\n{'=' * 74}\nCUE REDUNDANCY   (|rho| > 0.5 between cues)\n{'=' * 74}")
pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(corr.columns)
         for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) > 0.5]
for a, b, r in sorted(pairs, key=lambda t: -abs(t[2])):
    print(f"  {a} <-> {b}   rho = {r:+.3f}")
if not pairs:
    print("  none - the cues carry largely independent information")

res.to_csv(HERE / "results_1b.csv")
payload = {
    "n_melons": int(len(df)),
    "target": TARGET,
    "cue_columns": CUE_COLS,
    "primary_cues": PRIMARY,
    "absent_cues": cues["absent_cues"],
    "baseline_mae_cv": round(float(base), 4),
    "baseline_mae_in_sample": cues["mean_baseline_mae"],
    "success_bar_15pct": round(float(target), 4),
    "feature_sets": {k: {kk: (None if (isinstance(vv, float) and np.isnan(vv))
                              else vv)
                         for kk, vv in v.items()} for k, v in rows.items()},
    "set_definitions": SETS,
    "ablation_per_cue": [dict(cue=c, delta_mae=round(float(d), 4), reading=r)
                         for c, d, r in abl],
    "ablation_fold_sd": round(float(full_sd), 4),
    "bootstrap_ci": boot,
    "permutation_null": perm,
    "leave_one_variety_out": transfer,
    "model_families": families,
    "shrinkage": shrink,
    "variety_test": vtest,
    "learning_curve": curve,
    "per_melon": per_melon,
    "ranking": ranking,
    "cue_correlation": {"columns": list(corr.columns),
                        "matrix": corr.values.tolist(),
                        "redundant_pairs": pairs},
    "brix": {"min": float(y.min()), "max": float(y.max()),
             "mean": float(y.mean()), "sd": float(y.std(ddof=1))},
    "caveats": [
        "n=25: any difference smaller than the fold sd is not a result.",
        "The success bar is recomputed from the CROSS-VALIDATED baseline; "
        "cues.json's target_mae_15pct came from the in-sample deviation and is "
        "an unfairly hard bar for a cross-validated model.",
        "The only set under the bar is an ablation variant (minus_shape). "
        "Promoting it would be selection after seeing results.",
        "No noise floor is available: the three Brix readings are top/middle/"
        "bottom of the cut face, so their spread is the radial gradient, not "
        "refractometer error.",
        "Leave-one-variety-out has only 2 groups, so it is coarse.",
        "Cue 2 (weight) and cue 6 (stripe width) were never testable.",
    ],
}
(HERE / "results_1b.json").write_text(
    json.dumps(payload, indent=2, default=float), encoding="utf-8")
print(f"\nwrote {HERE / 'results_1b.csv'}")
print(f"wrote {HERE / 'results_1b.json'}")
print("\nnext:  python report.py    (builds report.html from that JSON)")
