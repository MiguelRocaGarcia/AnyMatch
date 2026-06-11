# AnyMatch for AllianceChicago — Patient Entity Resolution

A **patched fork** of [Jantory/anymatch](https://github.com/Jantory/anymatch)
(Zhang et al. 2024, *AnyMatch – Efficient Zero-Shot Entity Matching with a Small
Language Model*, arXiv:2409.04073), adapted for the UChicago × AllianceChicago
capstone on patient **entity resolution** (a.k.a. master patient index / record
linkage).

> Upstream README (the original paper's instructions) is preserved at the bottom
> of this file under **"Upstream AnyMatch"**. Everything above it is the
> AllianceChicago adaptation.

---

## 1. What this project does

AllianceChicago is a network of Federally Qualified Health Centers (FQHCs). Their
master patient index contains many records that actually refer to the **same
person** — created by typos, name shuffling, partial SSNs, address changes,
missing fields, etc. The goal is **entity resolution**: decide, for a pair of
candidate records, whether they are the *same patient*.

The pipeline is:

1. **Blocking** (done upstream of this repo) reduces the ~N² possible pairs to a
   manageable set of *candidate pairs* (`candidate_pairs_*.parquet`, just
   `PATID_A` / `PATID_B`).
2. **This repo scores each candidate pair** with a GPT-2-based classifier that
   outputs `match_prob` ∈ [0, 1] and a 0/1 `pred`. Pairs above a chosen
   threshold are treated as the same patient.

### Why a custom model

The AnyMatch authors **do not publish pretrained weights**, so we:

1. **Train** a base checkpoint on the 9 public entity-matching datasets
   (`anymatch_all9_gpt2_mode4`).
2. **Fine-tune** it on a **synthetic, AllianceChicago-shaped** pair corpus
   (`anymatch_alliance_ft_v2`) to fix two domain gaps the zero-shot model has on
   patient data: it didn't treat a matching **SSN** as decisive, and it treated
   the three name fields (first / middle / last) as independent when clerical
   entry routinely **shuffles tokens across them**.
3. **Score** real AllianceChicago candidate pairs with the fine-tuned model.

The model is `GPT2ForSequenceClassification` (GPT-2 124M + a 2-class head). Each
pair is serialized into one natural-language prompt
(*"Given the attributes of two records, are they the same? Record A is
first_name: …, … . Record B is … ."*) and the head's `softmax(logits)[:, 1]` is
`match_prob`. See `CLAUDE.md → Model mechanics` for details.

---

## 2. Setup

```sh
conda env create -f environment.yml      # or use the existing `llm_env`
conda activate anymatch                  # (the AllianceChicago VM uses `llm_env`)
```

**Always run scripts/notebooks from the `AnyMatch/` directory** (relative imports
of `data`, `utils`, `model`; scripts/notebooks assert `loo.py` exists in cwd).

**Windows / conda note (OpenMP crash):** on Windows the env can link two OpenMP
runtimes and abort with `OMP: Error #15 … ExitCode 3`. Both `predict_alliance_full.py`
and the notebooks set `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch to avoid
this. If you write a new script, set it at the very top before any
torch/pandas/numpy import.

**PHI:** real patient data only runs on policy-compliant compute (the HIPAA-tier
VM or Colab Pro). Never paste raw patient rows into chat tools. The
`valid_record=True` filter must be applied — the model produces confident garbage
on rows the cleaning step flagged invalid.

---

## 3. Data layout

```
data/
  raw/<dataset>/        # 8 Magellan deepmatcher zips + WDC (public EM datasets)
  prepared/<dataset>/   # output of data/preprocess.ipynb
  synthetic/            # synthetic_{train,test}_v2.csv (fine-tune corpus)
  alliance/             # candidate_pairs_v4_*.parquet + MDM_Population_cleaned_v3_*.parquet
saved_models/           # trained checkpoints (download from Drive)
```

Two AllianceChicago inputs (both **parquet**, latest drop `2026_06_11`):

- `MDM_Population_cleaned_v3_2026_06_11.parquet` — per-record cleaned population
  (one row per `PATID`, with `*_clean` fields, derived `Phones_set`, and a
  `valid_record` flag).
- `candidate_pairs_v4_2026_06_11.parquet` — blocking output, `PATID_A` / `PATID_B`.

---

## 4. How to run each thing

All commands run from `AnyMatch/`. On the AllianceChicago VM, replace `python`
with the env's interpreter if needed (`~/.conda/envs/llm_env/python.exe`).

### 4.1 Prepare the public EM datasets (only needed to (re)train the base)

Run `data/preprocess.ipynb` — consumes `data/raw/` → writes
`data/prepared/<dataset>/{train,valid,test}.csv` + `attr_*.csv`.

### 4.2 Generate the synthetic fine-tuning corpus

```sh
python synthetic_data_generation/generate_synthetic.py --seed 42 --version 2
python synthetic_data_generation/qa_checks.py --version 2     # asserts structural checks
```
Byte-reproducible from `--seed`. Produces `data/synthetic/synthetic_train_v2.csv`
(balanced ~1:1.5) and `synthetic_test_v2.csv` (realistic 1:9, entity-disjoint).
See `synthetic_data_generation/Synthetic-Dataset-Spec.md` (v0.5) for the design.

### 4.3 Train the base checkpoint (Colab Pro A100 recommended; `anymatch_training.ipynb`)

```sh
python loo.py --seed 42 --base_model gpt2 --leaved_dataset_name none \
    --serialization_mode mode4 --train_data attr+row \
    --row_sample_func one_pos_two_neg --patience_start 20 \
    --save_model_path saved_models/anymatch_all9_gpt2_mode4
```
`--leaved_dataset_name none` trains on **all 9** datasets (patient data is
out-of-distribution, so we hold none out). `one_pos_two_neg` avoids the ~1 GB
autogluon dependency. ~30–60 min on A100.

### 4.4 Fine-tune on the synthetic corpus (Colab Pro A100; `anymatch_finetuning.ipynb`)

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
Resumes from the base checkpoint and continues training (sequential, gentle LR,
early-stopped on the entity-disjoint test). ~10–25 min on A100. `--serialization_mode`
must match the base (mode4).

### 4.5 Evaluate the fine-tuned model on the synthetic test set (`anymatch_synthetic_inference.ipynb`)

Local diagnostic notebook. Scores `synthetic_test_v2.csv` (10k pairs, 1:9) with
the fine-tuned checkpoint and reports the full report: PR-AUC / ROC-AUC / MCC,
confusion matrix, threshold sweep + operating points, difficulty breakdown,
error analysis, and calibration. Use it to pick the production `match_prob`
threshold. Set `CKPT_DIR = saved_models/anymatch_all9_gpt2_mode4` to A/B against
the zero-shot base.

### 4.6 Score real AllianceChicago pairs — small / sanity batches (`anymatch_alliance_inference.ipynb`)

Local notebook. Joins the pairs parquet with the records parquet, filters
`valid_record=True`, scores `N_PAIRS` pairs **inline** (the fine-tuned checkpoint
ships only `tokenizer.json`, which the `predict_alliance.py` CLI can't load, so
the notebook scores in-process), and shows prediction diagnostics. Edit the
config cell: `N_PAIRS = 100` for a sanity check, `-1` for everything (but for the
full run prefer the CLI below).

### 4.7 Score ALL real pairs — full multi-day run (`predict_alliance_full.py`) ⭐

Standalone, **resumable**, batched CLI built for the full population. This is the
production scoring path.

```sh
python predict_alliance_full.py \
    --records_parquet data/alliance/MDM_Population_cleaned_v3_2026_06_11.parquet \
    --pairs_parquet   data/alliance/candidate_pairs_v4_2026_06_11.parquet \
    --ckpt_dir        saved_models/anymatch_alliance_ft_v2 \
    --output_csv      data/alliance/anymatch_predictions_full.csv \
    --chunk_size 2000 --batch_size 32
```
(PowerShell: use a backtick `` ` `` for line continuation, not `\`.)

What it does:
- Reads both parquet inputs, filters to pairs where **both** sides are valid,
  sorts deterministically (so resume is exact).
- Scores in chunks; after each chunk it **appends to the one output CSV and
  fsyncs** — so a crash/restart resumes from exactly where it stopped. Just
  re-run the same command.
- **Output columns:** `PATID_A, PATID_B, pred, match_prob` (join back to the
  records parquet on PATID for the underlying fields).
- **Logs progress to the terminal only** (no log file): a startup banner, then
  one line per chunk with `done/total (%)`, remaining, throughput, ETA, and an
  estimated finish time.

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--records_parquet` | — | Cleaned per-record MDM parquet (must point at the **file**, not the folder). |
| `--pairs_parquet` | — | Blocking-output parquet with `PATID_A`/`PATID_B` (point at the **file**). |
| `--ckpt_dir` | `saved_models/anymatch_alliance_ft_v2` | Checkpoint to score with. |
| `--output_csv` | — | Incremental, resumable predictions file. |
| `--chunk_size` | `2000` | Pairs per disk-flush + log line (progress granularity). |
| `--batch_size` | `32` | Model forward-pass size; raise on GPU. |
| `--device` | `auto` | `auto` picks CUDA → MPS → CPU. **Pass `--device cuda` on a GPU VM** — CPU is ~270 ms/pair (hours-to-days). |
| `--serialization_mode` | `mode4` | Must match the checkpoint. |

> **Gotcha:** point `--pairs_parquet` / `--records_parquet` at the exact `.parquet`
> *file*. If you pass a directory, pyarrow scans every file in it and fails on any
> non-parquet file (`Parquet magic bytes not found`).

### 4.8 (Reference) the upstream prediction CLI

`predict_alliance.py` scores a flat pairs **CSV** that already has friendly
`_l`/`_r` columns. It uses the slow `GPT2Tokenizer`, so it **cannot load the
fine-tuned checkpoint** (which ships only `tokenizer.json`). Use it only with the
base mode4 checkpoint or for legacy CSV inputs; for production use 4.7.

```sh
python predict_alliance.py \
    --model_path saved_models/anymatch_all9_gpt2_mode4 \
    --base_model gpt2 --serialization_mode mode4 \
    --input_csv <pairs>.csv --output_csv <predictions>.csv --batch_size 32
```

---

## 5. Picking a decision threshold

`pred` uses 0.5 by default. In patient ER a **false merge** (combining two
people) is the dangerous error, so for auto-merge lean toward a high-precision
operating point (e.g. `match_prob ≥ 0.99`) and route the mid band to human
review. Use the threshold sweep in `anymatch_synthetic_inference.ipynb` (and,
once available, a hand-labeled real holdout) to choose it.

---

## 6. Key files

| File | What |
|---|---|
| `predict_alliance_full.py` | **Production scoring CLI** — resumable, batched, parquet in, incremental CSV out, terminal progress logs. |
| `anymatch_alliance_inference.ipynb` | Local notebook for small/sanity real-pair batches (inline scoring). |
| `anymatch_synthetic_inference.ipynb` | Fine-tuned-model diagnostics on the synthetic test set. |
| `anymatch_training.ipynb` / `anymatch_finetuning.ipynb` | Colab: base train, then fine-tune. |
| `finetune_alliance.py` | Sequential fine-tune CLI (resumes from base checkpoint on synthetic corpus). |
| `loo.py` | Base training entry point (patched to train on all 9 datasets + save). |
| `predict_alliance.py` | Legacy CSV inference CLI (base checkpoint only). |
| `utils/alliance_schema.py` | **Single source of truth** for the feature schema (technical→friendly rename) shared by train + serve. |
| `utils/data_utils.py` | `df_serializer` (mode1–4) + sampling funcs. |
| `synthetic_data_generation/` | Synthetic corpus generator, stats extractor, pool builder, QA checks, design spec. |
| `docs/Data-Cleaning-Guide.md` | MDM cleaning rules (drive the synthetic-data conventions). |
| `CLAUDE.md` | Orientation + gotchas for working in this repo. |

---

## Upstream AnyMatch

The original paper repository and its instructions:
<https://github.com/Jantory/anymatch> — Zhang et al. 2024, arXiv:2409.04073.
The upstream `loo.py`/`throughput.py` experiments (Sections 6.1–6.3) still run;
note that this fork's `loo.py` adds `--leaved_dataset_name none` and
`--save_model_path`. Baselines (StringSim, ZeroER, Ditto, Jellyfish, MatchGPT)
are referenced in the upstream README history.
