# python predict_alliance_full.py \
#       --records_parquet data/alliance/MDM_Population_cleaned_v3_2026_06_11.parquet \
#       --pairs_parquet   data/alliance/candidate_pairs_v4_2026_06_11.parquet \
#       --ckpt_dir        saved_models/anymatch_alliance_ft_v2 \
#       --output_csv      data/alliance/anymatch_predictions_full.csv \
#       --chunk_size 2000 --batch_size 32

# Windows-VM fix: this conda env links two OpenMP runtimes (PyTorch's
# libiomp5md.dll + LLVM's libomp.dll from tokenizers/transformers). Without this
# the process aborts with "OMP: Error #15 ... ExitCode 3". MUST be set before any
# import that pulls in OpenMP (numpy/pandas/torch).


import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import csv
import time
import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2ForSequenceClassification

from utils.alliance_schema import id_str_dtypes, prep_record_df, FEATURE_COLS_SRC, FEATURE_COLS
from utils.data_utils import df_serializer
from data import GPTDataset

# ----------------------------------------------------------------------------
# Standalone, RESUMABLE, batched inference over ALL AllianceChicago candidate
# pairs. Designed for a multi-day run: writes predictions incrementally to ONE
# CSV (append + fsync per chunk) so a crash/restart picks up where it left off,
# and logs progress (% done, rate, ETA, remaining) to stdout + a log file.
#
# Output columns: PATID_A, PATID_B, pred, match_prob.  (Join back to the records
# parquet on PATID for the underlying fields — keeps the incremental file lean.)
#
# Example:
#   python predict_alliance_full.py \
#       --records_parquet data/alliance/MDM_Population_cleaned_v3_2026_06_11.parquet \
#       --pairs_parquet   data/alliance/candidate_pairs_v4_2026_06_11.parquet \
#       --ckpt_dir        saved_models/anymatch_alliance_ft_v2 \
#       --output_csv      data/alliance/anymatch_predictions_full.csv \
#       --chunk_size 2000 --batch_size 32
# ----------------------------------------------------------------------------

OUTPUT_COLS = ['PATID_A', 'PATID_B', 'pred', 'match_prob']


def parse_args():
    p = argparse.ArgumentParser(description='Resumable batched AnyMatch inference over all candidate pairs.')
    p.add_argument('--records_parquet', required=True, help='Cleaned MDM population parquet (per-record).')
    p.add_argument('--pairs_parquet', required=True, help='Blocking output parquet with PATID_A / PATID_B.')
    p.add_argument('--ckpt_dir', default='saved_models/anymatch_alliance_ft_v2', help='Trained checkpoint dir.')
    p.add_argument('--output_csv', required=True, help='Incremental predictions CSV (appended to; resumable).')
    p.add_argument('--log_file', default=None, help='Log file path (default: <output_csv>.log).')
    p.add_argument('--chunk_size', type=int, default=2000, help='Pairs scored + flushed to disk per chunk.')
    p.add_argument('--batch_size', type=int, default=32, help='Model forward-pass batch size.')
    p.add_argument('--max_len', type=int, default=1024, help='GPT-2 context cap; nothing should be filtered.')
    p.add_argument('--serialization_mode', default='mode4', help='Must match the checkpoint (mode4).')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    return p.parse_args()


def setup_logging(log_file):
    logger = logging.getLogger('predict_full')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_file); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def pick_device(choice):
    if choice != 'auto':
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available() and mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')


def fmt_eta(seconds):
    if seconds is None or seconds != seconds or seconds == float('inf'):
        return 'unknown'
    return str(timedelta(seconds=int(seconds)))


@torch.no_grad()
def score_chunk(model, dataset, batch_size, device):
    """Return (preds, probs) for one chunk's GPTDataset."""
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    preds, probs = [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.detach().float()
        probs += torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
        preds += logits.argmax(dim=-1).cpu().numpy().flatten().tolist()
    return preds, probs


def main():
    args = parse_args()
    assert os.path.exists('loo.py'), (
        f'Run from the AnyMatch/ directory (cwd={os.getcwd()!r}).')
    log_file = args.log_file or (args.output_csv + '.log')
    log = setup_logging(log_file)

    log.info('=== AnyMatch full batched inference ===')
    log.info(f'checkpoint={args.ckpt_dir}  mode={args.serialization_mode}  '
             f'chunk_size={args.chunk_size}  batch_size={args.batch_size}')

    # ---- 1. Records: load parquet, filter valid, rename->friendly, index by PATID ----
    log.info(f'Loading records parquet: {args.records_parquet}')
    records = pd.read_parquet(args.records_parquet)
    # Defensive: force ID-like columns to pandas 'string' so we never get "60640.0"
    # / stripped leading zeros (parquet usually preserves dtype, but be safe).
    for col, dt in id_str_dtypes(records.columns).items():
        records[col] = records[col].astype(dt)
    if 'valid_record' not in records.columns:
        raise KeyError("records parquet has no 'valid_record' column.")
    records_valid = records[records['valid_record']].copy()
    log.info(f'{len(records):,} records -> {len(records_valid):,} with valid_record=True')

    missing_cols = [c for c in FEATURE_COLS_SRC if c not in records_valid.columns]
    if missing_cols:
        raise KeyError(f'FEATURE_COLS_SRC not found in records: {missing_cols}')

    records_valid = prep_record_df(records_valid).set_index('PATID')
    if not records_valid.index.is_unique:
        dup = records_valid.index[records_valid.index.duplicated()].unique()[:5]
        raise ValueError(f'PATID not unique in records. Examples: {list(dup)}')

    # ---- 2. Pairs: load parquet, keep both-valid, DETERMINISTIC order (for resume) ----
    log.info(f'Loading pairs parquet: {args.pairs_parquet}')
    pairs = pd.read_parquet(args.pairs_parquet)[['PATID_A', 'PATID_B']]
    n_raw = len(pairs)
    valid_ids = set(records_valid.index)
    pairs = pairs[pairs['PATID_A'].isin(valid_ids) & pairs['PATID_B'].isin(valid_ids)]
    # Stable sort so the row order is identical across runs -> resume is correct.
    pairs = pairs.sort_values(['PATID_A', 'PATID_B'], kind='stable').reset_index(drop=True)
    total = len(pairs)
    log.info(f'{n_raw:,} candidate pairs -> {total:,} with both sides valid (will be scored)')
    if total == 0:
        log.warning('No pairs to score. Exiting.'); return

    # ---- 3. Resume: how many already written? ----
    done = 0
    file_exists = os.path.exists(args.output_csv) and os.path.getsize(args.output_csv) > 0
    if file_exists:
        with open(args.output_csv, 'r', newline='') as f:
            done = max(sum(1 for _ in f) - 1, 0)  # minus header
        if done >= total:
            log.info(f'Output already has {done:,} rows >= {total:,} pairs. Nothing to do.')
            return
        log.info(f'Resuming: {done:,} pairs already in {args.output_csv}; '
                 f'{total - done:,} remaining ({done/total*100:.2f}% done).')
    else:
        log.info(f'Fresh run -> {args.output_csv}')

    # ---- 4. Model + tokenizer ----
    log.info('Loading tokenizer + model ...')
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(args.ckpt_dir)
        log.info('Tokenizer: fast, from checkpoint')
    except Exception as e:
        log.info(f'Fast tokenizer from checkpoint failed ({type(e).__name__}); falling back to base gpt2 BPE.')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2ForSequenceClassification.from_pretrained(args.ckpt_dir)
    model.config.pad_token_id = model.config.eos_token_id
    device = pick_device(args.device)
    model.to(device); model.eval()
    log.info(f'Scoring on {device.type.upper()}')

    # ---- 5. Chunk loop with incremental append + progress logging ----
    out_f = open(args.output_csv, 'a', newline='')
    writer = csv.writer(out_f)
    if not file_exists:
        writer.writerow(OUTPUT_COLS); out_f.flush(); os.fsync(out_f.fileno())

    session_start = time.perf_counter()
    session_done = 0
    try:
        for start in range(done, total, args.chunk_size):
            end = min(start + args.chunk_size, total)
            chunk = pairs.iloc[start:end].reset_index(drop=True)

            # Build the _l/_r wide table for this chunk (per-side .loc lookup).
            wide = chunk.copy()
            for side, patid_col in [('l', 'PATID_A'), ('r', 'PATID_B')]:
                side_df = (records_valid.loc[chunk[patid_col].values, FEATURE_COLS]
                           .reset_index(drop=True).add_suffix(f'_{side}'))
                wide = pd.concat([wide, side_df], axis=1)

            serial_src = wide.copy()
            serial_src['label'] = 0  # df_serializer requires it; unused for inference
            serialized = df_serializer(serial_src, args.serialization_mode)
            dataset = GPTDataset(tokenizer, serialized, max_len=args.max_len)
            if len(dataset) != len(chunk):
                raise RuntimeError(
                    f'{len(chunk) - len(dataset)} row(s) in chunk [{start}:{end}] exceeded '
                    f'max_len={args.max_len} and were filtered -> would misalign predictions. '
                    'Raise --max_len.')

            preds, probs = score_chunk(model, dataset, args.batch_size, device)

            for pa, pb, pr, pp in zip(chunk['PATID_A'], chunk['PATID_B'], preds, probs):
                writer.writerow([pa, pb, int(pr), f'{pp:.6f}'])
            out_f.flush(); os.fsync(out_f.fileno())  # durable before we count it done

            session_done += (end - start)
            total_done = end
            elapsed = time.perf_counter() - session_start
            rate = session_done / elapsed if elapsed > 0 else 0.0  # pairs/sec this session
            remaining = total - total_done
            eta = remaining / rate if rate > 0 else None
            log.info(
                f'{total_done:,}/{total:,} ({total_done/total*100:.2f}%) | '
                f'remaining={remaining:,} | rate={rate:.1f} pairs/s | '
                f'chunk={end-start} | ETA={fmt_eta(eta)} | '
                f'finish~{(datetime.now()+timedelta(seconds=eta)).strftime("%Y-%m-%d %H:%M") if eta else "unknown"}')
    finally:
        out_f.flush(); os.fsync(out_f.fileno()); out_f.close()

    log.info(f'DONE. {total:,} pairs scored. Predictions -> {args.output_csv}')


if __name__ == '__main__':
    main()
