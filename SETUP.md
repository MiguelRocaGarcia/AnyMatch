# AnyMatch — AllianceChicago Setup

Local-fork notes for running [Jantory/anymatch](https://github.com/Jantory/anymatch) on AllianceChicago patient pairs. The authors do **not** publish pretrained weights, so we train our own on the 9 public EM datasets and apply that checkpoint zero-shot to the patient data.

## Local edits already applied

| File | Change |
|---|---|
| `loo.py` | New `--save_model_path` flag + `--leaved_dataset_name none` branch (train on all 9 datasets). |
| `utils/train_eval.py` | Added `predict()` — returns per-row label + match probability without needing ground truth. |
| `utils/data_utils.py` | Dropped unused top-level `from autogluon.tabular import TabularPredictor` (it was hanging local imports for several minutes). Also capped `one_pos_two_neg` negative sampling at `len(neg_pairs)` so near-balanced datasets like WDC don't crash. |
| `predict_alliance.py` | New CLI that loads a saved checkpoint, runs `predict()` on a pairs CSV, and writes `pred` + `match_prob` columns. |
| `anymatch_training.ipynb` | Colab Pro A100 notebook: install, train (`--serialization_mode mode4` → `saved_models/anymatch_all9_gpt2_mode4/`), back up to Drive, sanity-check on a public test set. |
| `anymatch_synthetic_inference.ipynb` | Local Mac inference on the 18-pair synthetic CSV. Uses `--serialization_mode mode4` and the mode4 checkpoint. |
| `anymatch_alliance_inference.ipynb` | Local Mac inference on real candidate pairs. Joins blocking output with cleaned MDM population, applies a `FEATURE_RENAMES` map (MDM column names → semantic English like `dob`, `ssn`, `zip`), then calls `predict_alliance.py --serialization_mode mode4`. |
| `data/synthetic/alliance_pairs_synthetic.csv` | 18 hand-crafted patient pairs (10 matches, 8 non-matches) used by the inference notebook for sanity checking. |
| `synthetic_data_generation/extract_mdm_stats.py` | Aggregate-stats extractor (k-anon; no raw rows) → `synthetic_data_stats.json`. |
| `synthetic_data_generation/build_pools.py` | Offline-first vocabulary-pool builder → `pools/*.json`. |
| `synthetic_data_generation/generate_synthetic.py` | Synthetic fine-tuning corpus generator (deterministic; see "Synthetic fine-tuning corpus" below). |
| `synthetic_data_generation/qa_checks.py` | Asserts the spec §12 sanity checks on generated output. |

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
4. **Train on Colab Pro (A100):** open `anymatch_training.ipynb`, run cells top-to-bottom. ~30–60 min. Trains with `--serialization_mode mode4`; checkpoint backed up to `MyDrive/AnyMatch/checkpoints/anymatch_all9_gpt2_mode4/`. The earlier mode1 checkpoint at `MyDrive/AnyMatch/checkpoints/anymatch_all9_gpt2/` is kept for A/B comparison — leave it in place.
5. **Patient inference — two options:**
   - **Colab:** upload pairs CSV to `MyDrive/AnyMatch/alliance/pairs.csv`; the Colab training notebook's Section 5 cell (uses `--serialization_mode mode4` against the mode4 checkpoint) produces `predictions.csv` with `pred` and `match_prob`.
   - **Local Mac:** download `MyDrive/AnyMatch/checkpoints/anymatch_all9_gpt2_mode4/` into `AnyMatch/saved_models/anymatch_all9_gpt2_mode4/`, launch Jupyter / VS Code from the `AnyMatch/` folder, and run `anymatch_synthetic_inference.ipynb` (18 hand-crafted pairs, has ground-truth labels) or `anymatch_alliance_inference.ipynb` (real candidate pairs from blocking). Both notebooks default to the mode4 checkpoint and pass `--serialization_mode mode4`. CPU-only is fine for small batches; full 212k-pair runs should go to Colab GPU.

## Synthetic fine-tuning corpus

A synthetic, AllianceChicago-shaped pair corpus for fine-tuning (and for testing blocking / deterministic rules / Fellegi–Sunter). Design + rationale: `synthetic_data_generation/Synthetic-Dataset-Spec.md`. Everything below runs **locally, offline, no PHI, no GPU** — it works from committed aggregate stats, not raw rows.

1. **(One-time, PHI-local) extract aggregate stats** from the cleaned MDM file:
   ```sh
   python synthetic_data_generation/extract_mdm_stats.py \
       --input /path/to/MDM_Population_cleaned_v1.csv \
       --output synthetic_data_generation/synthetic_data_stats.json
   ```
   Emits aggregates only (k-anon ≥20 top-N, missingness, histograms, within-cluster field agreement, geo-joint, missingness-pattern joint). The JSON is safe to commit; it's already in the repo.
2. **Build the vocabulary pools** (offline; bootstraps from the stats + curated supplements):
   ```sh
   python synthetic_data_generation/build_pools.py
   ```
   Writes `synthetic_data_generation/pools/{first_names,last_names,streets,nicknames,initial_expansion}.json`.
3. **Generate the corpus** (deterministic from `--seed`; ~4 s for the full 40k+10k build):
   ```sh
   python synthetic_data_generation/generate_synthetic.py --seed 42 --version 2
   ```
   Writes **two** files to `data/synthetic/` at `v2` (v0.5 hybrid design): `synthetic_train` (balanced ~1:1.5; realistic entity-first bulk + budgeted hard-scenario overlay + dirty tail) and `synthetic_test` (realistic prevalence, blocking-survivor-like hard negatives only, entity-disjoint from train). Ground-truth provenance (`entity_id_a/b`, `case_type`, `corruptions_applied`) is carried inline — no separate manifest files. Use `--smoke` for a quick tiny run.
4. **Validate**:
   ```sh
   python synthetic_data_generation/qa_checks.py --version 2
   ```
   Asserts the §12 checks: schema/mode4 parity, realistic missingness (±3pp), the positive multi-corruption heavy-tail, 100% key-sharing test negatives, entity disjointness, and the structural SSN/phone/token rules.

The fine-tune pair CSVs are drop-in for `loo.py` (model reads `*_l`/`*_r` + `label`; provenance columns are skipped by `df_serializer`). Note: the synthetic schema passes `first_name`/`middle_name`/`last_name` as **three separate fields** (spec §2), which supersedes the single derived `name` the current alliance inference notebook uses — update its `FEATURE_RENAMES` to match before scoring with a fine-tuned checkpoint.

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
