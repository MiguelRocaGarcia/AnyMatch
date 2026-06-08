# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **patched fork** of [Jantory/anymatch](https://github.com/Jantory/anymatch) (Zhang et al. 2024, arXiv 2409.04073) — a GPT-2-based zero-shot entity matcher — adapted for the AllianceChicago patient entity-resolution capstone. The upstream authors **do not publish pretrained weights**, so this fork's primary purpose is to (a) train a checkpoint on the 9 public EM datasets and (b) score AllianceChicago patient candidate pairs with the resulting model.

`SETUP.md` is the runbook (raw-data prep, Drive upload, Colab training, local inference). Read it before making process changes. This file is for orientation and gotchas only.

## Workflow at a glance

1. **Data prep** (local): `data/preprocess.ipynb` consumes `data/raw/` → produces `data/prepared/<dataset>/{train,valid,test}.csv` + `attr_*.csv`.
2. **Train** (Colab Pro A100): `anymatch_training.ipynb` runs `loo.py --leaved_dataset_name none` on all 9 datasets; ~30–60 min; checkpoint backed up to Drive at `MyDrive/AnyMatch/checkpoints/anymatch_all9_gpt2_mode4/`. (The previous `anymatch_all9_gpt2/` mode1 checkpoint is kept around for A/B comparison.)
3. **Sanity-check inference** (local): `anymatch_synthetic_inference.ipynb` runs `predict_alliance.py` on `data/synthetic/alliance_pairs_synthetic.csv` (18 hand-crafted patient pairs).
4. **Real inference** (local for small batches, Colab for full 212k): `anymatch_alliance_inference.ipynb` joins the blocking output `data/alliance/candidate_pairs_*.parquet` with `data/alliance/MDM_Population_cleaned_v1.csv`, filters `valid_record=True`, and scores via `predict_alliance.py`.

## Patched files — do not revert

| File | Patch | Why |
|---|---|---|
| `loo.py` | `--leaved_dataset_name none` trains on all 9 datasets (upstream always excludes one). Added `--save_model_path`; wired `save_model=True` into `train()`. | Patient records are out-of-distribution from all public EM datasets — no point holding one out. Without `save_model_path`, the checkpoint was discarded after each run. |
| `utils/train_eval.py` | Added `predict()` returning `(preds, probs)` for unlabeled inference. | Upstream `inference()` returns only F1/accuracy — useless for production scoring. |
| `utils/data_utils.py` | Dropped unused top-level `from autogluon.tabular import TabularPredictor`. Also capped `one_pos_two_neg` negative sampling at `len(neg_pairs)`. | The autogluon import pulled in TensorFlow/abseil and hung local Python for ~8 minutes (visible as `[mutex.cc : 452] RAW: Lock blocking`). The sampling cap fixes `Cannot take a larger sample than population` on near-balanced datasets like WDC. |
| `data/preprocess.ipynb` | autogluon import made optional; WDC valid-set merge bug fixed; MatchGPT pickle override skipped; WDC test prep enabled; attr-pair prep enabled. | See `SETUP.md` for cell-level detail. |

## New files (not in upstream)

- `predict_alliance.py` — inference CLI. Reads any CSV with `*_l` / `*_r` columns + optional `label`, writes the same CSV + `pred` / `match_prob`. Other columns (e.g. `PATID_A`, `PATID_B`) ride through untouched because `df_serializer` only consumes `_l` / `_r`-suffixed columns.
- `anymatch_training.ipynb`, `anymatch_synthetic_inference.ipynb`, `anymatch_alliance_inference.ipynb` — the three end-to-end notebooks.
- `data/synthetic/alliance_pairs_synthetic.csv` — 18 hand-crafted pairs (10 matches, 8 non-matches) for sanity-checking before real data.
- `docs/Data-Cleaning-Guide.md` — field-by-field cleaning rules applied to MDM_Population to produce `*_clean` columns + `valid_record` + derived `full_name_tokens` / `Phones_set` / `Address_normalized`. Any synthetic data we generate must conform to these conventions.
- `synthetic_data_generation/` — design + tooling for a domain-shift fine-tuning corpus (see next section).

## Synthetic fine-tuning corpus (built)

The zero-shot mode4 checkpoint underperforms on FQHC patient pairs in two specific ways: it doesn't treat matching SSN as decisive, and it treats `FirstNM` / `MiddleNM` / `LastNM` as independent (they aren't — clerical entry routinely shuffles tokens across the three fields). Fix is to fine-tune on a synthetic, AllianceChicago-shaped pair corpus.

- `synthetic_data_generation/Synthetic-Dataset-Spec.md` — the design doc (now v0.4, build-complete). Purpose, MDM-cleaned drop-in schema, **case-first** fine-tune assembly + **entity-first** realistic-eval, full scenario catalog, corruption rates calibrated from real within-SSN-cluster agreement (§5.8), entity-disjoint split, output layout, §12 sanity checks. Read it before changing the generator.
- `synthetic_data_generation/extract_mdm_stats.py` — aggregate-stats extractor (k-anon top-N, missingness, histograms, **within-cluster field agreement**, geo-joint, missingness-pattern joint) from `MDM_Population_cleaned_v1.csv` → `synthetic_data_stats.json`. **Aggregates only — no raw rows.** Run locally; JSON is safe to commit.
- `synthetic_data_generation/build_pools.py` — **offline-first** vocabulary-pool builder. Bootstraps `pools/*.json` (first/last names, streets, nicknames, initial-expansion) from `synthetic_data_stats.json` + curated supplements; no network/Ollama needed.
- `synthetic_data_generation/generate_synthetic.py` — the generator. Correlated entity sampler (geo/missingness joints + pediatric coupling), independent-marginal corruptions calibrated from §5.8, case-first scenario registry (all §8 buckets), entity-first realistic-eval, entity-disjoint 15% split. Writes the four §11 outputs to `data/synthetic/` with `_v{N}` versioning. **Byte-reproducible from `--seed`.** Run: `python synthetic_data_generation/generate_synthetic.py --seed 42 --version 1` (~4 s; `--smoke` for a tiny test).
- `synthetic_data_generation/qa_checks.py` — asserts every §12 check; run after generation: `python synthetic_data_generation/qa_checks.py --version 1`.
- Outputs (4 files): `finetune_{train,test}_v1.csv` (balanced 1:1.5, case-first, entity-disjoint), `realistic_eval_v1.csv` (1:9, entity-first), `blocking_eval_v1.csv` (record-level, full cleaning schema + `entity_id`). Provenance (`entity_id`, `case_type`, `corruptions_applied`, split) rides inline in these files — no separate manifest files.

Gotchas: pair CSVs use `_l`/`_r` for model columns and `_A`/`_B` (+ `entity_id_a/b`, `case_type`, `corruptions_applied`) for provenance that `df_serializer` skips. `Address_normalized` is emitted **null** to match the real cleaned file (libpostal wasn't run there). The fine-tune three-name-field schema (§2) supersedes the old single derived `name` — the alliance inference `FEATURE_RENAMES` must be updated to match before scoring with a fine-tuned checkpoint.

## Model mechanics (important when changing the prompt or features)

- Base: `GPT2ForSequenceClassification` (GPT-2 124M + `Linear(768, 2)` head). **Not** a generator — the classification head reads the hidden state of the last non-padding token. `softmax(logits, dim=-1)[:, 1]` = `match_prob`.
- Serialization (`utils/data_utils.py::df_serializer`). The production checkpoint is trained with **mode4**: each pair becomes `Given the attributes of two records, are they the same? Record A is name: <v>, dob: <v>, ssn: <v>, .... Record B is name: <v>, dob: <v>, ssn: <v>, ....`. Attribute names come from the dataframe column names (sans `_l`/`_r`), so the alliance inference notebook applies a `FEATURE_RENAMES` map to convert technical MDM column names (`BirthDT_clean`, `ZipCD_clean_base`) to clean lowercase English (`dob`, `zip`) — that's what the model saw during training and any drift hurts accuracy. The old **mode1** checkpoint (positional `COL v1, COL v2, ...` template, no attribute names) is kept at `saved_models/anymatch_all9_gpt2/` for A/B. Inference and training serialization mode must always match the checkpoint.
- Missing values are replaced with the literal string `'N/A'` via `.fillna('N/A')` — the model has seen this thousands of times in training and treats both-sides-`N/A` as neutral.
- GPT-2 context cap is **1024 tokens**. With ~10 feature columns per side, real patient pairs fit comfortably; check token length with `tokenizer.encode(text)` if you add many columns.

## Train / infer commands

Train (run from the AnyMatch folder, A100 strongly recommended):

```sh
python loo.py --seed 42 --base_model gpt2 --leaved_dataset_name none \
    --serialization_mode mode4 --train_data attr+row \
    --row_sample_func one_pos_two_neg --patience_start 20 \
    --save_model_path saved_models/anymatch_all9_gpt2_mode4
```

`one_pos_two_neg` skips the autogluon dependency (~1 GB). For the paper's best result use `automl_filter` instead, which requires running the AutoML cell in `data/preprocess.ipynb` and installing autogluon.

Predict on an unlabeled pairs CSV:

```sh
python predict_alliance.py \
    --model_path saved_models/anymatch_all9_gpt2_mode4 \
    --base_model gpt2 --serialization_mode mode4 \
    --input_csv <pairs>.csv --output_csv <predictions>.csv --batch_size 32
```

`predict_alliance.py`'s `--serialization_mode` defaults to `mode1` for backward compatibility — always pass `--serialization_mode mode4` explicitly when using the mode4 checkpoint.

## Gotchas (learned the hard way)

- **CSV dtype trap.** When reading any cleaned MDM file, force-cast ID columns to `'string'` via `pd.read_csv(..., dtype={'SSN_clean': 'string', 'last_4_SSN': 'string', 'ZipCD_clean_base': 'string', 'PrimaryPhoneNBR_clean': 'string', ...})`. Without this, pandas infers float64 because of NaN, prints `358467965.0` in the prompt, and strips leading zeros — destroys exact-match signal.
- **Per-side date conversion bug.** Do `pd.to_datetime(...).dt.strftime(...)` **once on the source DataFrame** before joining, not separately on each `_l` / `_r` slice. The latter triggers Series index alignment and silently nulls one side.
- **`predict_alliance.py` requires the AnyMatch directory as cwd** (relative imports of `data`, `utils`, `model`). Notebooks assert `os.path.exists('loo.py')` to enforce this.
- **No MPS on Apple Silicon.** `utils/train_eval.py::predict` only checks `torch.cuda.is_available()`. Inference on Mac falls back to CPU (~700 ms/pair). Don't waste time looking for an MPS path — at the scale this notebook runs, it doesn't matter; full 212k-pair runs should go to Colab GPU.
- **PHI.** Real patient data only on Colab Pro / HIPAA-tier compute, never raw rows in chat tools. The `valid_record=True` filter must run upstream — the model produces confident garbage on rows the cleaning step flagged invalid.

## Layout sketch

```
loo.py                 # training entry point (patched)
predict_alliance.py    # inference CLI (new)
inference.py           # upstream evaluation script (kept for reference)
model.py, data.py      # model + dataset class loaders (unchanged)
utils/
  data_utils.py        # df_serializer (mode1..4), sampling funcs (patched)
  train_eval.py        # train(), evaluate(), inference(), predict() (patched)
data/
  raw/<dataset>/       # 8 Magellan deepmatcher zips + WDC
  prepared/<dataset>/  # output of preprocess.ipynb
  synthetic/           # alliance_pairs_synthetic.csv + (future) finetune_train/test, realistic_eval
  alliance/            # candidate_pairs_*.parquet, MDM_Population_cleaned_v1.csv
docs/
  Data-Cleaning-Guide.md   # MDM cleaning rules (drives synthetic-data conventions)
synthetic_data_generation/
  Synthetic-Dataset-Spec.md # scenario catalog + generation plan (v0.1 scaffold)
  extract_mdm_stats.py      # aggregate-stats extractor for spec [TBD]s
saved_models/          # trained checkpoints (download from Drive)
```
