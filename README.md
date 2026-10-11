# Watermelon sweetness: do the folk cues work?

Folk advice says you can pick a sweet watermelon by its field spot, webbing, rind gloss, shape, bee stings and knock sound. We measured these cues on 25 melons (11 sonya, 14 kinnaree) from photos and knock recordings, cut every melon open, and read its sweetness (°Brix) with a refractometer at the top, middle and bottom. The target is the mean of the three readings.

* **1A: does each cue matter on its own?** Spearman correlation and a permutation test, in the notebook.
* **1B: do the cues together predict sweetness?** A ridge regression judged by cross-validation against always guessing the mean, in `model/`.

## Folders

| path | what it is |
| --- | --- |
| `watermelon_features.ipynb` | the main notebook: photo features (steps 1–15), knock-sound features (16–21), training table (22), cue list (23), 1A tests (24–26) |
| `photos/` | 173 photos, one ID photo plus up to 6 side views per melon, named `NN_vK.jpg` / `NN_id.jpg` |
| `audio/` | 25 knock recordings, one per melon (`New Recording 35.m4a` is melon 1, through 59 for melon 25) |
| `prepared/` | rotated and resized copies of the photos, written by notebook step 2 |
| `cues.json` | which column stands for each folk cue, and which way folklore says it should go. Written by step 23 and read by both 1A and 1B |
| `watermelon_features.csv` | every measurement per melon, for reading |
| `results/` | 1A results: `cue_results.csv` and `cue_scatter.png` |
| `model/` | 1B: `train_1b.py`, its training table, results (`results_1b.csv`, `results_1b.json`) and `report.html` |
| `photo-preprocess/` | `build_photos.py` and `photo_map.csv`, to rebuild `photos/` from the original iPhone HEIC files |

## How to run

```bash
pip install -r requirements.txt
```

1. Open `watermelon_features.ipynb` in Jupyter or VS Code and run all cells from the repo root. It reads `photos/` and `audio/` and writes `watermelon_features.csv`, `watermelon_training.csv`, `cues.json` and `results/`. Processing the photos takes a few minutes.
2. Train and evaluate the 1B model:
   ```bash
   python model/train_1b.py
   ```
3. Build the 1B report page, `model/report.html`:
   ```bash
   python model/report.py
   ```

`photo-preprocess/build_photos.py` is only needed to rebuild `photos/` from the original HEIC download:

```bash
python photo-preprocess/build_photos.py "<unzipped photo folder>"
```

## Limits

* 25 melons is a small sample, so every result here is a pattern to test on more melons, not a proven rule.
* The photos had no colour card and the lighting was not fixed, so colour and gloss are only roughly comparable.
* No weight was recorded, so "heavy for its size" could not be tested. Stripe width exists only for the striped sonya melons.
