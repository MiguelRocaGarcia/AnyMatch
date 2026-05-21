"""Run a trained AnyMatch checkpoint on an unlabeled candidate-pair CSV.

Input CSV: any number of attribute columns suffixed `_l` (left record) and the
same set suffixed `_r` (right record). A `label` column is optional; if absent,
a placeholder 0 column is added so the upstream dataset class is happy.

Output CSV: the original columns plus `pred` (0/1) and `match_prob` (P(label==1)).
Row order is preserved.

Example:
    python predict_alliance.py \\
        --model_path saved_models/anymatch_all9_gpt2 \\
        --base_model gpt2 \\
        --serialization_mode mode1 \\
        --input_csv data/prepared/alliance/pairs.csv \\
        --output_csv alliance_predictions.csv
"""
import argparse

import pandas as pd
from transformers import (BertForSequenceClassification, BertTokenizer,
                          GPT2ForSequenceClassification, GPT2Tokenizer,
                          T5Tokenizer, AutoModelForSeq2SeqLM)

from data import BertDataset, GPTDataset, T5Dataset
from utils.data_utils import df_serializer
from utils.train_eval import predict


def load_checkpoint(model_path, base_model):
    if 'gpt' in base_model:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2ForSequenceClassification.from_pretrained(model_path)
        model.config.pad_token_id = model.config.eos_token_id
        return model, tokenizer, GPTDataset
    if 'bert' in base_model:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        return model, tokenizer, BertDataset
    if 't5' in base_model:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return model, tokenizer, T5Dataset
    raise ValueError(f'Unsupported base_model: {base_model}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True,
                        help='Directory containing the fine-tuned checkpoint (output of loo.py --save_model_path).')
    parser.add_argument('--base_model', default='gpt2', choices=['gpt2', 'bert-base', 't5-base'])
    parser.add_argument('--serialization_mode', default='mode1', choices=['mode1', 'mode2', 'mode3', 'mode4'])
    parser.add_argument('--input_csv', required=True, help='Pairs CSV with _l / _r columns.')
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=10000,
                        help='Token cap per row. Default is intentionally huge to keep every row; '
                             'lower it only if you hit OOM.')
    args = parser.parse_args()

    raw = pd.read_csv(args.input_csv)
    if 'label' not in raw.columns:
        raw['label'] = 0

    # df_serializer mutates and projects to [text, label], so work on a copy and merge back by index.
    serialized = df_serializer(raw.copy(), args.serialization_mode)

    model, tokenizer, DatasetClass = load_checkpoint(args.model_path, args.base_model)
    dataset = DatasetClass(tokenizer, serialized, max_len=args.max_len)
    if len(dataset) != len(raw):
        raise RuntimeError(
            f'Row count mismatch after tokenization: input had {len(raw)} rows but dataset has {len(dataset)}. '
            f'Increase --max_len or shorten input records.')

    preds, probs = predict(tokenizer, model, dataset, batch_size=args.batch_size, base_model=args.base_model)

    out = raw.copy()
    out['pred'] = preds
    out['match_prob'] = probs
    out.to_csv(args.output_csv, index=False)
    print(f'Wrote {len(out)} predictions to {args.output_csv}', flush=True)


if __name__ == '__main__':
    main()
