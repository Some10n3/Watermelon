# Watermelon sweetness: do the folk cues work?

Folk advice says you can pick a sweet watermelon by its field spot, webbing, rind gloss, shape, bee stings and knock sound. We measured these cues on 25 melons (11 sonya, 14 kinnaree) from photos and knock recordings, cut every melon open, and read its sweetness (°Brix) with a refractometer at the top, middle and bottom. The target is the mean of the three readings.

* **1A: does each cue matter on its own?** Partial Spearman controlling for variety, with a permutation test, in `model/analysis_1a.py`.
* **1B: do the cues together predict sweetness?** A ridge regression judged by cross-validation against always guessing the mean, in `model/train_1b.py`.
* **Can either help you choose?** Ranking metrics — pairwise accuracy and best-of-3 — because a shopper needs to pick the better melon, not to know its °Brix.

## Folders

| path | what it is |
| --- | --- |
| `watermelon_features.ipynb` | the main notebook: photo features (steps 1–15), knock-sound features (16–21), training table (22), cue list (23), 1A tests (24–26) |
| `photos/` | 173 photos, one ID photo plus up to 6 side views per melon, named `NN_vK.jpg` / `NN_id.jpg` |
| `audio/` | 25 knock recordings, one per melon (`New Recording 35.m4a` is melon 1, through 59 for melon 25) |
| `prepared/` | rotated and resized copies of the photos, written by notebook step 2 |
| `cues.json` | which column stands for each folk cue, and which way folklore says it should go. Written by step 23 and read by both 1A and 1B |
| `watermelon_features.csv` | every measurement per melon, for reading |
| `results/` | 1A results: `cue_results.csv` and `analysis_1a.json`, written by `model/analysis_1a.py`, plus `cue_scatter.png` from notebook step 26 |
| `model/` | `analysis_1a.py` (1A), `train_1b.py` (1B), the training table, results (`results_1b.csv`, `results_1b.json`) and `report.py` / `report.html` |
| `photo-preprocess/` | `build_photos.py` and `photo_map.csv`, to rebuild `photos/` from the original iPhone HEIC files |

## How to run

```bash
pip install -r requirements.txt
```

Run these in order from the repo root. Each step reads what the previous one wrote.

**First time, from a clean checkout**, run 1 → 2 → 3 → 4, then re-run the notebook's last two cells (steps 25–26) to draw the cue scatter. Those two cells read the 1A results that step 2 produces, so on a first pass they print a "not found" note and skip instead of failing. Everything afterwards is just 1 → 2 → 3 → 4, or only 2 → 3 → 4 if the photos and recordings have not changed.

1. **Features.** Open `watermelon_features.ipynb` and run all cells. It reads `photos/` and `audio/` and writes `watermelon_features.csv`, `model/watermelon_training.csv` and `cues.json`. Processing the photos takes a few minutes.
2. **1A — each cue on its own.** Spearman and a 10,000-shuffle permutation test, both plain and partial (controlling for variety), with a best-of-7 family correction and an acquisition-order confound check:
   ```bash
   python model/analysis_1a.py
   ```
   Writes `results/cue_results.csv` and `results/analysis_1a.json`. **This script is the only writer of `cue_results.csv`** — the notebook reads it.
3. **1B — the cues together.** Ridge, cross-validated against guessing the mean, plus ablations, bootstrap CI, permutation null, variety transfer and ranking metrics:
   ```bash
   python model/train_1b.py
   ```
   Writes `model/results_1b.csv` and `model/results_1b.json`.
4. **The report page**, `model/report.html`, built from the outputs of steps 2 and 3:
   ```bash
   python model/report.py
   ```
   View it with `python -m http.server 8777 --directory model` and open <http://localhost:8777/report.html>.

Steps 2–4 are deterministic: every random seed is fixed, so re-running gives byte-identical results.

`photo-preprocess/build_photos.py` is only needed to rebuild `photos/` from the original HEIC download:

```bash
python photo-preprocess/build_photos.py "<unzipped photo folder>"
```

## Limits

* 25 melons is a small sample, so every result here is a pattern to test on more melons, not a proven rule.
* The photos had no colour card and the lighting was not fixed, so colour and gloss are only roughly comparable. An acquisition-order check (step 2) finds no drift, but the risk cannot be ruled out by design.
* No weight was recorded, so "heavy for its size" could not be tested. Stripe width exists only for the striped sonya melons. **Two of the eight folk cues therefore have no result at all.**
* The two varieties were bought and photographed in blocks (sonya = melons 1–10 and 25, kinnaree = 11–24), so any session-level effect is inseparable from variety. Interleave varieties across collection days next time.
* The three Brix readings are top, middle and bottom of the cut face, so their spread is the fruit's own radial gradient, not refractometer error. **There is no measurement of instrument repeatability**, so the irreducible error cannot be bounded. Take two readings at the same spot on the next batch.
* Under the preregistered analysis (partial Spearman, variety controlled, best-of-7 correction) **no cue reaches "supported"**. The strongest, `rind_gloss`, is "suggestive" and reverses the folk claim.
