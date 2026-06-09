"""Fine-tune a trained AnyMatch checkpoint on the synthetic AllianceChicago corpus.

This is the sequential-fine-tune step: it RESUMES from the zero-shot mode4
checkpoint (trained on the 9 public EM datasets by loo.py) and continues training
on the synthetic FQHC pair corpus, so the model learns the two failure modes the
zero-shot model misses (SSN as decisive identity; cross-name-field token shuffles)
without forgetting general EM ability.

It differs from loo.py in exactly the two ways loo.py can't handle:
  1. It loads weights from an existing checkpoint directory, not a fresh GPT-2 base.
  2. It reads flat `_l`/`_r` pair CSVs (the synthetic finetune_{train,test} files),
     renames technical -> friendly columns via utils.alliance_schema (the SAME
     schema both inference notebooks use, guaranteeing train/serve consistency),
     and serializes with df_serializer.

Defaults are the "gentle" fine-tune: low LR + early stopping on the entity-disjoint
held-out test, to minimise catastrophic forgetting.

Example (run from the AnyMatch/ directory):
    python finetune_alliance.py \\
        --base_checkpoint saved_models/anymatch_all9_gpt2_mode4 \\
        --train_csv data/synthetic/finetune_train_v1.csv \\
        --valid_csv data/synthetic/finetune_test_v1.csv \\
        --save_model_path saved_models/anymatch_alliance_ft_v1 \\
        --serialization_mode mode4
"""
import argparse
import os

import pandas as pd
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

from data import GPTDataset
from utils.alliance_schema import id_str_dtypes, prep_paired_df
from utils.data_utils import df_serializer
from utils.train_eval import train, inference


def load_pairs(csv_path, serialization_mode):
    """Read a synthetic pair CSV, map technical -> friendly columns, serialize.

    Returns a [text, label] DataFrame ready for GPTDataset.
    """
    header = pd.read_csv(csv_path, nrows=0).columns
    raw = pd.read_csv(csv_path, dtype=id_str_dtypes(header), low_memory=False)
    if 'label' not in raw.columns:
        raise KeyError(f'{csv_path} has no `label` column — required for fine-tuning.')
    # Rename to the canonical friendly schema and drop off-schema _l/_r columns so
    # the prompt attribute names match inference exactly. df_serializer then turns
    # the friendly _l/_r columns into the mode4 prompt.
    friendly = prep_paired_df(raw)
    friendly['label'] = raw['label'].astype(int).values
    serialized = df_serializer(friendly.copy(), serialization_mode)
    return serialized


def main():
    p = argparse.ArgumentParser(description='Sequential fine-tune of AnyMatch on the synthetic corpus.')
    p.add_argument('--base_checkpoint', required=True,
                   help='Directory of the trained checkpoint to resume from (e.g. the all9 mode4 model).')
    p.add_argument('--train_csv', required=True, help='Synthetic finetune_train CSV (_l/_r + label).')
    p.add_argument('--valid_csv', required=True,
                   help='Held-out CSV for early stopping — use the entity-disjoint finetune_test.')
    p.add_argument('--save_model_path', required=True, help='Where to write the fine-tuned checkpoint.')
    p.add_argument('--base_model', default='gpt2', choices=['gpt2'],
                   help='Only gpt2 is supported (matches the production mode4 checkpoint).')
    p.add_argument('--serialization_mode', default='mode4',
                   choices=['mode1', 'mode2', 'mode3', 'mode4'],
                   help='Must match how the base checkpoint was trained. Production is mode4.')
    p.add_argument('--eval_csv', default='',
                   help='Optional realistic-eval CSV (1:9) for a final report-only metric.')
    # Gentle fine-tune defaults.
    p.add_argument('--lr', type=float, default=1e-5, help='Low LR to limit forgetting (base train used 2e-5).')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3, help='Early-stop after N epochs without valid-F1 gain.')
    p.add_argument('--patience_start', type=int, default=0, help='Earliest epoch early stopping can fire.')
    p.add_argument('--train_batch_size', type=int, default=64)
    p.add_argument('--valid_batch_size', type=int, default=128)
    p.add_argument('--max_len', type=int, default=350, help='Token cap per row; rows above are filtered.')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    assert os.path.exists('loo.py'), (
        f'Run from the AnyMatch/ directory (cwd={os.getcwd()!r}); relative imports need it.')
    for f in ['config.json', 'vocab.json', 'merges.txt']:
        if not os.path.exists(os.path.join(args.base_checkpoint, f)):
            raise FileNotFoundError(f'{args.base_checkpoint} is missing {f} — not a valid checkpoint dir.')
    os.makedirs(args.save_model_path, exist_ok=True)

    print(f'Resuming from checkpoint: {args.base_checkpoint}', flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_checkpoint)
    model = GPT2ForSequenceClassification.from_pretrained(args.base_checkpoint)
    model.config.pad_token_id = model.config.eos_token_id

    print('Serializing train / valid ...', flush=True)
    train_df = load_pairs(args.train_csv, args.serialization_mode)
    valid_df = load_pairs(args.valid_csv, args.serialization_mode)
    train_d = GPTDataset(tokenizer, train_df, max_len=args.max_len)
    valid_d = GPTDataset(tokenizer, valid_df, max_len=args.max_len)
    print(f'Train pairs: {len(train_d)} | Valid pairs: {len(valid_d)}', flush=True)
    print(f'Config — lr: {args.lr} | epochs: {args.epochs} | patience: {args.patience} '
          f'(start {args.patience_start}) | batch: {args.train_batch_size} | '
          f'mode: {args.serialization_mode} | max_len: {args.max_len}', flush=True)

    result_prefix = os.path.join(args.save_model_path, 'finetune')
    best_model = train(
        tokenizer, model, train_d, valid_d,
        seed=args.seed, patient=True, save_model=True, patience_start=args.patience_start,
        lr=args.lr, epochs=args.epochs, base_model=args.base_model,
        train_batch_size=args.train_batch_size, valid_batch_size=args.valid_batch_size,
        save_freq=50, patience=args.patience,
        save_model_path=args.save_model_path, save_result_prefix=result_prefix,
    )
    # train() saves the model weights; persist the tokenizer alongside so the
    # inference scripts can load both from one directory.
    tokenizer.save_pretrained(args.save_model_path)
    print(f'Best fine-tuned checkpoint saved to {args.save_model_path}', flush=True)

    print('\n--- Held-out test (early-stopping set) ---', flush=True)
    f1, acc = inference(tokenizer, best_model, valid_d, batch_size=args.valid_batch_size,
                        base_model=args.base_model)
    print(f'finetune_test  acc={acc*100:.2f}  f1={f1*100:.2f}', flush=True)

    if args.eval_csv:
        print('\n--- Realistic-eval (1:9, report only) ---', flush=True)
        eval_df = load_pairs(args.eval_csv, args.serialization_mode)
        eval_d = GPTDataset(tokenizer, eval_df, max_len=args.max_len)
        f1e, acce = inference(tokenizer, best_model, eval_d, batch_size=args.valid_batch_size,
                              base_model=args.base_model)
        print(f'realistic_eval  acc={acce*100:.2f}  f1={f1e*100:.2f}', flush=True)


if __name__ == '__main__':
    main()
