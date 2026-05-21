# AnyMatch — AllianceChicago Setup

Local-fork notes for running [Jantory/anymatch](https://github.com/Jantory/anymatch) on AllianceChicago patient pairs. The authors do **not** publish pretrained weights, so we train our own on the 9 public EM datasets and apply that checkpoint zero-shot to the patient data.

## Local edits already applied

| File | Change |
|---|---|
| `loo.py` | New `--save_model_path` flag + `--leaved_dataset_name none` branch (train on all 9 datasets). |
| `utils/train_eval.py` | Added `predict()` — returns per-row label + match probability without needing ground truth. |
| `utils/data_utils.py` | Dropped unused top-level `from autogluon.tabular import TabularPredictor` (it was hanging local imports for several minutes). Also capped `one_pos_two_neg` negative sampling at `len(neg_pairs)` so near-balanced datasets like WDC don't crash. |
| `predict_alliance.py` | New CLI that loads a saved checkpoint, runs `predict()` on a pairs CSV, and writes `pred` + `match_prob` columns. |
| `anymatch_colab.ipynb` | Colab Pro A100 notebook: install, train, infer. |
| `anymatch_inference.ipynb` | Local Mac inference notebook: loads the downloaded checkpoint and scores synthetic / real patient pairs. |
| `data/synthetic/alliance_pairs_synthetic.csv` | 18 hand-crafted patient pairs (10 matches, 8 non-matches) used by the inference notebook for sanity checking. |

## Workflow

1. **Raw data — already downloaded** into `data/raw/` (~25 MB total):
   - 8 Magellan datasets (`abt`, `amgo`, `beer`, `dbac`, `dbgo`, `foza`, `itam`, `waam`) — direct HTTP zips from [deepmatcher Datasets.md](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md), unzipped and flattened so files sit at `data/raw/<name>/{tableA,tableB,train,valid,test}.csv`.
   - `wdc` — built from the [WDC Products 80pair](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) benchmark (Peeters & Bizer, 2023). The `wdcproducts80cc20rnd000un_{train_large, valid_large, gs}.json.gz` files were converted to `train/valid/test.pkl.gz` with the 7 columns `preprocess.ipynb` expects (`title/price/priceCurrency` × `_left/_right` + `label`). The AnyMatch paper doesn't specify which WDC variant they used; 80cc/large is a reasonable default. To swap variants later, re-run the JSON→pickle conversion against another `wdc_products` zip.
2. **Run `data/preprocess.ipynb`** to produce `data/prepared/<dataset>/{train,valid,test}.csv` and `attr_*.csv` files. The notebook has been patched to be runnable end-to-end against the data in `data/raw/`:
   - Cell 3: `autogluon` import is now optional.
   - Cell 12: WDC valid-set price merge bug fixed.
   - Cell 19: MatchGPT test-set override is now skipped (those pickle files aren't shipped with the Magellan zips).
   - Cell 22–23: WDC test prep rewritten to match the WDC train/valid schema (`title_l, price_l, label, title_r, price_r`) and enabled.
   - Cell 30: `prepare_all_attribute_pairs(...)` enabled (required for `--train_data attr+row`).
   - Cell 34: AutoML still commented out — only needed if you want `--row_sample_func automl_filter` (paper's best). The default training command in `anymatch_colab.ipynb` uses `one_pos_two_neg` to skip the heavy autogluon install.
3. **Upload to Drive:** zip this folder → `MyDrive/AnyMatch.zip`. Upload `data/prepared/` → `MyDrive/AnyMatch/prepared/`.
4. **Train on Colab Pro (A100):** open `anymatch_colab.ipynb`, run cells top-to-bottom. ~30–60 min. Checkpoint backed up to `MyDrive/AnyMatch/checkpoints/anymatch_all9_gpt2/`.
5. **Patient inference — two options:**
   - **Colab:** upload pairs CSV to `MyDrive/AnyMatch/alliance/pairs.csv`; the Colab notebook's Section 5 cell produces `predictions.csv` with `pred` and `match_prob`.
   - **Local Mac:** download `MyDrive/AnyMatch/checkpoints/anymatch_all9_gpt2/` into `AnyMatch/saved_models/anymatch_all9_gpt2/`, launch Jupyter / VS Code from the `AnyMatch/` folder, and run `anymatch_inference.ipynb` top-to-bottom. It runs on the synthetic CSV by default; swap `INPUT_CSV` to your real pairs file when ready. CPU-only is fine for small batches.

## Pairs CSV format

Every patient attribute appears twice — `<attr>_l` and `<attr>_r`. A `label` column is optional (placeholder `0` added if missing).

```
first_name_l,last_name_l,dob_l,ssn4_l,first_name_r,last_name_r,dob_r,ssn4_r
John,Smith,1980-05-12,1234,Jon,Smith,1980-05-12,1234
```

Normalize case, whitespace, and date format upstream — AnyMatch is a string matcher and is sensitive to formatting drift.

## PHI

Do training/inference on Colab Pro or a HIPAA-approved tier, not on free Colab. Never paste raw patient rows into chat-based tools.

## Sanity check

After training, run the notebook's Section 6 cell to score the `abt_buy` test set with the trained model. The paper reports F1 in the 0.6–0.9 range across the 9 datasets — a comparable score there means the pipeline is wired correctly.
