# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **patched fork** of [Jantory/anymatch](https://github.com/Jantory/anymatch) (Zhang et al. 2024, arXiv 2409.04073) — a GPT-2-based zero-shot entity matcher — adapted for the AllianceChicago patient entity-resolution capstone. The upstream authors **do not publish pretrained weights**, so this fork's primary purpose is to (a) train a checkpoint on the 9 public EM datasets and (b) score AllianceChicago patient candidate pairs with the resulting model.

`SETUP.md` is the runbook (raw-data prep, Drive upload, Colab training, local inference). Read it before making process changes. This file is for orientation and gotchas only.

## Workflow at a glance

1. **Data prep** (local): `data/preprocess.ipynb` consumes `data/raw/` → produces `data/prepared/<dataset>/{train,valid,test}.csv` + `attr_*.csv`.
2. **Train** (Colab Pro A100): `anymatch_training.ipynb` runs `loo.py --leaved_dataset_name none` on all 9 datasets; ~30–60 min; checkpoint backed up to Drive at `MyDrive/AnyMatch/checkpoints/anymatch_all9_gpt2_mode4/`. (The previous `anymatch_all9_gpt2/` mode1 checkpoint is kept around for A/B comparison.)
3. **Fine-tune** (Colab Pro A100): `anymatch_finetuning.ipynb` runs `finetune_alliance.py`, which **resumes from** the all9 mode4 checkpoint and continues training on the synthetic corpus (`data/synthetic/synthetic_train_v2.csv`, early-stopped on `synthetic_test_v2.csv`); ~10–25 min; checkpoint backed up to Drive at `MyDrive/AnyMatch/checkpoints/anymatch_alliance_ft_v2/`. This is the *sequential* (not joint-mix) fine-tune — gentle LR + early stop on the entity-disjoint test.
4. **Sanity-check inference** (local): `anymatch_synthetic_inference.ipynb` renames the generated `data/synthetic/synthetic_test_v2.csv` (the realistic-distribution evaluator) to the friendly schema (`utils/alliance_schema.prep_paired_df`) and scores it **inline** (the fine-tuned checkpoint ships only `tokenizer.json`, which the slow `GPT2Tokenizer` in `predict_alliance.py` can't load).
5. **Real inference** — two paths, both **inline** (not via `predict_alliance.py`):
   - *Small / sanity batches* (local notebook): `anymatch_alliance_inference.ipynb` joins the blocking output `data/alliance/candidate_pairs_v4_2026_06_11.parquet` with the per-record `data/alliance/MDM_Population_cleaned_v3_2026_06_11.parquet` (**both parquet now**), filters `valid_record=True`, scores `N_PAIRS` pairs inline, and shows diagnostics.
   - *Full population* (CLI, multi-day): `predict_alliance_full.py` — resumable, batched, parquet in → incremental single CSV out, terminal progress logs. This is the production scoring path.

## Patched files — do not revert

| File | Patch | Why |
|---|---|---|
| `loo.py` | `--leaved_dataset_name none` trains on all 9 datasets (upstream always excludes one). Added `--save_model_path`; wired `save_model=True` into `train()`. | Patient records are out-of-distribution from all public EM datasets — no point holding one out. Without `save_model_path`, the checkpoint was discarded after each run. **Note:** `loo.py` can only train a fresh GPT-2 base on the 9 prepared dirs — it cannot resume from a checkpoint or read flat pair CSVs, which is why the fine-tune lives in `finetune_alliance.py`. |
| `utils/train_eval.py` | Added `predict()` returning `(preds, probs)` for unlabeled inference. | Upstream `inference()` returns only F1/accuracy — useless for production scoring. |
| `utils/data_utils.py` | Dropped unused top-level `from autogluon.tabular import TabularPredictor`. Also capped `one_pos_two_neg` negative sampling at `len(neg_pairs)`. | The autogluon import pulled in TensorFlow/abseil and hung local Python for ~8 minutes (visible as `[mutex.cc : 452] RAW: Lock blocking`). The sampling cap fixes `Cannot take a larger sample than population` on near-balanced datasets like WDC. |
| `data/preprocess.ipynb` | autogluon import made optional; WDC valid-set merge bug fixed; MatchGPT pickle override skipped; WDC test prep enabled; attr-pair prep enabled. | See `SETUP.md` for cell-level detail. |

## New files (not in upstream)

- `predict_alliance.py` — **legacy** inference CLI. Reads any CSV with `*_l` / `*_r` columns + optional `label`, writes the same CSV + `pred` / `match_prob`. Other columns (e.g. `PATID_A`, `PATID_B`) ride through untouched because `df_serializer` only consumes `_l` / `_r`-suffixed columns. Uses the **slow** `GPT2Tokenizer`, so it **cannot load the fine-tuned checkpoint** (which ships only `tokenizer.json`) — use it only with the base mode4 checkpoint / legacy CSVs.
- `predict_alliance_full.py` — **production scoring CLI** for the full population. Reads the per-record MDM **parquet** + the candidate-pairs **parquet** (point each at the *file*, not a directory — pyarrow scans directories as datasets and chokes on stray files), filters `valid_record=True` on both sides, sorts deterministically, and scores in chunks. **Resumable:** after each `--chunk_size` chunk it appends to the one `--output_csv` and `fsync`s, so a crash/restart (it's a multi-day run) resumes from exactly the next unscored pair — just re-run the same command. Output columns: `PATID_A, PATID_B, pred, match_prob` (lean; join back to records on PATID). Scores **inline** with `GPT2TokenizerFast` (falls back to base `gpt2` BPE), device `auto` (CUDA→MPS→CPU; pass `--device cuda` on a GPU VM). Progress logs go to **stdout only** (no log file): per-chunk `done/total (%)`, remaining, rate, ETA, finish estimate.
- `finetune_alliance.py` — **sequential fine-tune** CLI. Resumes from a checkpoint dir (`GPT2ForSequenceClassification.from_pretrained(base_checkpoint)`), reads the synthetic `_l`/`_r` pair CSVs, renames technical→friendly via `utils/alliance_schema`, serializes mode4, trains via the existing `train()`, early-stops on the entity-disjoint test, and reports a final realistic-eval (1:9) F1. Gentle defaults: `--lr 1e-5 --epochs 10 --patience 3`.
- `utils/alliance_schema.py` — **single source of truth** for the AllianceChicago feature schema. `CANONICAL_RENAMES` (technical MDM column → clean English attribute name; full spec §2: three separate name fields + `suffix`, `address2`, `ssn`, `ssn4`, `email`), plus `id_str_dtypes`, `serialize_set_field`, `prep_record_df` (per-record path for alliance inference), `prep_paired_df` (already-paired path for synthetic inference + fine-tune training). **Both inference notebooks and `finetune_alliance.py` import this** so train and serve serialize identical attribute names — drift here silently hurts accuracy.
- `anymatch_training.ipynb`, `anymatch_finetuning.ipynb`, `anymatch_synthetic_inference.ipynb`, `anymatch_alliance_inference.ipynb` — the four end-to-end notebooks.
- `docs/Data-Cleaning-Guide.md` — field-by-field cleaning rules applied to MDM_Population to produce `*_clean` columns + `valid_record` + derived `full_name_tokens` / `Phones_set` / `Address_normalized`. Any synthetic data we generate must conform to these conventions.
- `synthetic_data_generation/` — design + tooling for a domain-shift fine-tuning corpus (see next section).

## Synthetic fine-tuning corpus (built)

The zero-shot mode4 checkpoint underperforms on FQHC patient pairs in two specific ways: it doesn't treat matching SSN as decisive, and it treats `FirstNM` / `MiddleNM` / `LastNM` as independent (they aren't — clerical entry routinely shuffles tokens across the three fields). Fix is to fine-tune on a synthetic, AllianceChicago-shaped pair corpus.

- `synthetic_data_generation/Synthetic-Dataset-Spec.md` — the design doc (now **v0.5**, the single source of truth). **Hybrid** assembly: a realistic entity-first *bulk* (real missingness joint + multi-field §5.8 corruptions + a correlated **dirty tail**) plus a budgeted minority of hard-scenario *overlays* that never force field presence; **two outputs only** (`synthetic_train` balanced ~1:1.5; `synthetic_test` realistic-prevalence honest evaluator with **blocking-survivor-like hard negatives**); entity-disjoint by construction; mode4-oriented; §12 sanity checks. Read it before changing the generator.
- `synthetic_data_generation/extract_mdm_stats.py` — aggregate-stats extractor (k-anon top-N, missingness, histograms, **within-cluster field agreement**, geo-joint, missingness-pattern joint) from `MDM_Population_cleaned_v1.csv` → `synthetic_data_stats.json`. **Aggregates only — no raw rows.** Run locally; JSON is safe to commit.
- `synthetic_data_generation/build_pools.py` — **offline-first** vocabulary-pool builder. Bootstraps `pools/*.json` (first/last names, streets, nicknames, initial-expansion) from `synthetic_data_stats.json` + curated supplements; no network/Ollama needed.
- `synthetic_data_generation/generate_synthetic.py` — the generator (v0.5 hybrid). Correlated entity sampler (geo/missingness joints + pediatric coupling) → realistic **bulk** positives (`apply_calibrated_corruptions`, with an optional `messiness` multiplier for the dirty tail) and **hard negatives** (`make_hard_negative` / `force_shared_key` — two distinct people forced to share one strong key) + a budgeted **overlay** from the §8 `ScenarioLib` (`run_overlay`, with SSN-presence reconciliation on non-SSN-led match scenarios). `build_train` / `build_test` assemble the two files; train and test draw their own fresh entities so they are entity-disjoint by construction. Writes **two** §11 outputs with `_v{N}` versioning. **Byte-reproducible from `--seed`.** Run: `python synthetic_data_generation/generate_synthetic.py --seed 42 --version 2` (~4 s; `--smoke` for a tiny test).
- `synthetic_data_generation/qa_checks.py` — asserts the §12 checks (incl. realistic-missingness ±3pp, positive multi-corruption heavy-tail, 100% key-sharing test negatives); run after generation: `python synthetic_data_generation/qa_checks.py --version 2`.
- Outputs (2 files): `synthetic_train_v2.csv` (balanced ~1:1.5, hybrid bulk+overlay) and `synthetic_test_v2.csv` (realistic prevalence, hard negatives only, held-out entities). Provenance (`entity_id_a/b`, `case_type`, `corruptions_applied`) rides inline — no manifest files.

Gotchas: pair CSVs use `_l`/`_r` for model columns and `_A`/`_B` (+ `entity_id_a/b`, `case_type`, `corruptions_applied`) for provenance that `df_serializer` skips. `Address_normalized` is emitted **null** to match the real cleaned file (libpostal wasn't run there). Both inference notebooks and `finetune_alliance.py` import the three-name `CANONICAL_RENAMES` from `utils/alliance_schema.py`, so train/serve serialize identical attribute names. The generated `synthetic_{train,test}` CSVs carry **technical** column names (`FirstNM_clean_l`, …); the friendly mode4 rename happens at load time via `prep_paired_df`.

## Model mechanics (important when changing the prompt or features)

- Base: `GPT2ForSequenceClassification` (GPT-2 124M + `Linear(768, 2)` head). **Not** a generator — the classification head reads the hidden state of the last non-padding token. `softmax(logits, dim=-1)[:, 1]` = `match_prob`.
- Serialization (`utils/data_utils.py::df_serializer`). The production checkpoint is trained with **mode4**: each pair becomes `Given the attributes of two records, are they the same? Record A is first_name: <v>, middle_name: <v>, last_name: <v>, suffix: <v>, dob: <v>, ssn: <v>, ssn4: <v>, .... Record B is ....`. Attribute names come from the dataframe column names (sans `_l`/`_r`), so the friendly schema is centralized in `utils/alliance_schema.py::CANONICAL_RENAMES` (imported by both inference notebooks **and** `finetune_alliance.py`) to convert technical MDM column names (`FirstNM_clean`, `BirthDT_clean`, `ZipCD_clean_base`) to clean English (`first_name`, `dob`, `zip`). Any drift between train and serve hurts accuracy — change the schema in that one module, never inline. The old **mode1** checkpoint (positional `COL v1, COL v2, ...` template, no attribute names) is kept at `saved_models/anymatch_all9_gpt2/` for A/B. Inference and training serialization mode must always match the checkpoint.
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

Fine-tune the all9 checkpoint on the synthetic corpus (sequential; A100 recommended):

```sh
python finetune_alliance.py \
    --base_checkpoint saved_models/anymatch_all9_gpt2_mode4 \
    --train_csv data/synthetic/synthetic_train_v2.csv \
    --valid_csv data/synthetic/synthetic_test_v2.csv \
    --eval_csv  data/synthetic/synthetic_test_v2.csv \
    --save_model_path saved_models/anymatch_alliance_ft_v2 \
    --base_model gpt2 --serialization_mode mode4 \
    --lr 1e-5 --epochs 10 --patience 3
```

`--serialization_mode` must match the base checkpoint (mode4). `--valid_csv` is the entity-disjoint `synthetic_test` (drives early stopping); `--eval_csv` (same realistic-distribution test) is reported at the end. Lower `--lr` (5e-6) or `--patience` if the §7 regression check shows public-EM F1 dropping (catastrophic forgetting).

Predict on an unlabeled pairs CSV:

```sh
python predict_alliance.py \
    --model_path saved_models/anymatch_all9_gpt2_mode4 \
    --base_model gpt2 --serialization_mode mode4 \
    --input_csv <pairs>.csv --output_csv <predictions>.csv --batch_size 32
```

`predict_alliance.py`'s `--serialization_mode` defaults to `mode1` for backward compatibility — always pass `--serialization_mode mode4` explicitly when using the mode4 checkpoint. **Note:** `predict_alliance.py` can't load the fine-tuned checkpoint (slow tokenizer vs `tokenizer.json`) — for the fine-tuned model use the inline notebook or `predict_alliance_full.py`.

Score the **full population** with the fine-tuned checkpoint (resumable, parquet in, multi-day):

```sh
python predict_alliance_full.py \
    --records_parquet data/alliance/MDM_Population_cleaned_v3_2026_06_11.parquet \
    --pairs_parquet   data/alliance/candidate_pairs_v4_2026_06_11.parquet \
    --ckpt_dir        saved_models/anymatch_alliance_ft_v2 \
    --output_csv      data/alliance/anymatch_predictions_full.csv \
    --chunk_size 2000 --batch_size 32 --device cuda
```

Re-run the **same command** to resume after a crash (it counts rows already in `--output_csv`). Drop `--device cuda` to auto-pick (CPU is ~15 h for the v4 drop). Logs to stdout only.

## Gotchas (learned the hard way)

- **ID dtype trap.** When reading any cleaned MDM file, force ID columns to `'string'`. For CSV use `pd.read_csv(..., dtype=id_str_dtypes(header))`; for **parquet** (no `dtype=` arg) cast after read: `for col, dt in id_str_dtypes(df.columns).items(): df[col] = df[col].astype(dt)`. Without this, pandas infers float64 because of NaN, prints `358467965.0` in the prompt, and strips leading zeros — destroys exact-match signal. `utils/alliance_schema.id_str_dtypes(columns)` returns the dtype map for whichever ID columns are present.
- **Parquet path must be a file, not a directory.** `pd.read_parquet(dir)` makes pyarrow treat the directory as a dataset and read *every* file in it, failing on any non-parquet sibling (`Parquet magic bytes not found in footer`). Point `--records_parquet` / `--pairs_parquet` (and the notebook paths) at the exact `.parquet` file.
- **OpenMP duplicate-runtime crash (Windows).** On the AllianceChicago Windows VM, torch's `libiomp5md.dll` + LLVM's `libomp.dll` (from tokenizers/transformers) both load and abort the process with `OMP: Error #15 … ExitCode 3` (kernel dies on the inference cell). Fix: set `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'` **before** the first torch/pandas/numpy import. Already baked into `predict_alliance_full.py` and both inference notebooks (top of cell 1).
- **Per-side date conversion bug.** Do `pd.to_datetime(...).dt.strftime(...)` **once on the source DataFrame** before joining, not separately on each `_l` / `_r` slice. The latter triggers Series index alignment and silently nulls one side.
- **`predict_alliance.py` requires the AnyMatch directory as cwd** (relative imports of `data`, `utils`, `model`). Notebooks assert `os.path.exists('loo.py')` to enforce this.
- **Device selection.** The shared `utils/train_eval.py::predict` only checks `torch.cuda.is_available()` (CUDA-or-CPU). The newer inline scorers — both inference notebooks and `predict_alliance_full.py` — add their own `pick_device()` (CUDA → Apple-Silicon MPS → CPU, with a CPU fallback if an MPS op is unimplemented), so don't route through the shared `predict()` if you want MPS. CPU is ~270 ms/pair; the full population (~205k valid pairs in the v4 drop) is ~15 h on CPU, minutes-to-hours on a GPU — pass `--device cuda` when a GPU is available.
- **PHI.** Real patient data only on Colab Pro / HIPAA-tier compute, never raw rows in chat tools. The `valid_record=True` filter must run upstream — the model produces confident garbage on rows the cleaning step flagged invalid.

## Layout sketch

```
loo.py                 # base training entry point (patched; trains GPT-2 on the 9 datasets)
finetune_alliance.py   # sequential fine-tune CLI (new; resumes from checkpoint, synthetic corpus)
predict_alliance.py    # legacy CSV inference CLI (new; base checkpoint only — slow tokenizer)
predict_alliance_full.py # production scoring CLI (new; resumable, batched, parquet in -> incremental CSV)
inference.py           # upstream evaluation script (kept for reference)
model.py, data.py      # model + dataset class loaders (unchanged)
utils/
  data_utils.py        # df_serializer (mode1..4), sampling funcs (patched)
  train_eval.py        # train(), evaluate(), inference(), predict() (patched)
  alliance_schema.py   # CANONICAL_RENAMES + prep_record_df/prep_paired_df (new; shared by train+infer)
data/
  raw/<dataset>/       # 8 Magellan deepmatcher zips + WDC
  prepared/<dataset>/  # output of preprocess.ipynb
  synthetic/           # synthetic_{train,test}_v2 (generated; v0.5 hybrid)
  alliance/            # candidate_pairs_v4_*.parquet, MDM_Population_cleaned_v3_*.parquet (both parquet)
docs/
  Data-Cleaning-Guide.md   # MDM cleaning rules (drives synthetic-data conventions)
synthetic_data_generation/
  Synthetic-Dataset-Spec.md # scenario catalog + generation plan (v0.4, build-complete)
  extract_mdm_stats.py      # aggregate-stats extractor
  build_pools.py            # offline-first vocab-pool builder
  generate_synthetic.py     # the generator (byte-reproducible from --seed)
  qa_checks.py              # asserts the §12 structural checks
anymatch_training.ipynb / anymatch_finetuning.ipynb            # Colab: base train, then fine-tune
anymatch_synthetic_inference.ipynb / anymatch_alliance_inference.ipynb  # local inline inference (synthetic eval / small real batches)
predict_alliance_full.py  # full-population production scoring (see above)
saved_models/          # trained checkpoints (download from Drive)
```
