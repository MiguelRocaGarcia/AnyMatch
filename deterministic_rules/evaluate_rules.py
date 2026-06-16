"""Evaluate the deterministic rule engine on the synthetic test set and compare it
head-to-head with the GPT-2 model.

The rules emit three classes (match / non_match / review) while the model is
binary, so the comparison is **decided-subset + queue report**:
  * Rules are scored only on the pairs they auto-decide (match/non_match); we also
    report the size and true-label composition of the review queue.
  * If model predictions are supplied (--model_pred_csv), the model is scored on
    (a) all pairs and (b) the rules' decided subset -- the apples-to-apples view.

Also:
  * a per-case_type decision table (design validation -- confirms M-SSN -> match,
    NM-SSN-03 -> non_match, NM-HH-TWIN/POL-AMBIG -> review, ...);
  * a --calibrate mode that prints comparator distributions for corrupted positives
    vs hard negatives, justifying the thresholds in rules_alliance.py.

Run from the AnyMatch repo root:
    python deterministic_rules/evaluate_rules.py
    python deterministic_rules/evaluate_rules.py --model_pred_csv model_preds.csv
    python deterministic_rules/evaluate_rules.py --calibrate data/synthetic/synthetic_train_v3.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import rules_alliance as R  # noqa: E402  (deterministic_rules is on sys.path as this file's dir)

DEFAULT_TEST = os.path.join(_REPO_ROOT, "data", "synthetic", "synthetic_test_v3.csv")


# ==============================================================================
# Reporting helpers
# ==============================================================================
def _binary_report(title, y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{title}")
    print(f"  n={len(y_true)}  precision={p:.4f}  recall={r:.4f}  f1={f1:.4f}  mcc={mcc:.4f}")
    print(f"  confusion [rows=true 0/1, cols=pred 0/1]:  TN={tn} FP={fp} | FN={fn} TP={tp}")
    return dict(n=len(y_true), precision=p, recall=r, f1=f1, mcc=mcc)


def evaluate(test_csv: str, model_pred_csv: str | None):
    df = R.load_paired_csv(test_csv)
    if "label" not in df.columns:
        raise SystemExit("test CSV has no `label` column -- can't evaluate.")
    scored = R.classify_df(df)
    scored["label"] = scored["label"].astype(int)
    n = len(scored)

    # ---------- decision distribution ----------
    print("=" * 78)
    print(f"DETERMINISTIC RULES   test={os.path.basename(test_csv)}")
    print("=" * 78)
    dist = scored["rule_pred"].value_counts()
    print(f"\nDecision distribution over {n} pairs:")
    for k in (R.MATCH, R.NON_MATCH, R.REVIEW):
        c = int(dist.get(k, 0))
        print(f"  {k:9s}: {c:6d} ({c / n:6.1%})")

    # ---------- decided-subset metrics ----------
    decided = scored[scored["rule_pred"] != R.REVIEW].copy()
    decided["pred_bin"] = (decided["rule_pred"] == R.MATCH).astype(int)
    coverage = len(decided) / n
    print(f"\nAuto-decision coverage: {len(decided)}/{n} = {coverage:.1%}")
    rule_metrics = _binary_report("RULES on auto-decided subset:", decided["label"], decided["pred_bin"])

    # ---------- review-queue report ----------
    review = scored[scored["rule_pred"] == R.REVIEW]
    rv_pos = int((review["label"] == 1).sum())
    rv_neg = int((review["label"] == 0).sum())
    print(f"\nReview queue: {len(review)} pairs ({len(review)/n:.1%} of all)")
    if len(review):
        print(f"  true matches deferred: {rv_pos} ({rv_pos/len(review):.1%})   "
              f"true non-matches deferred: {rv_neg} ({rv_neg/len(review):.1%})")
    # recall accounting against the full positive population
    total_pos = int((scored["label"] == 1).sum())
    auto_tp = int(((decided["label"] == 1) & (decided["pred_bin"] == 1)).sum())
    print(f"\nPositive accounting (of {total_pos} true matches):")
    print(f"  auto-matched : {auto_tp} ({auto_tp/total_pos:.1%})")
    print(f"  sent to review: {rv_pos} ({rv_pos/total_pos:.1%})")
    auto_fn = int(((decided["label"] == 1) & (decided["pred_bin"] == 0)).sum())
    print(f"  auto-non-matched (missed): {auto_fn} ({auto_fn/total_pos:.1%})")

    # ---------- per-case_type decision table ----------
    if "case_type" in scored.columns:
        print("\nPer-case_type decision table (design validation):")
        tab = (scored.groupby(["case_type", "rule_pred"]).size()
               .unstack(fill_value=0).reindex(columns=[R.MATCH, R.NON_MATCH, R.REVIEW], fill_value=0))
        tab["label"] = scored.groupby("case_type")["label"].first()
        tab = tab.sort_index()
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(tab.to_string())

    # ---------- rule_id firing counts ----------
    print("\nRule firing counts:")
    print(scored["rule_id"].value_counts().to_string())

    # ---------- head-to-head vs model ----------
    if model_pred_csv:
        _model_comparison(scored, decided, model_pred_csv)
    else:
        print("\n[no --model_pred_csv supplied] To compare against the model, export a "
              "PATID_A,PATID_B,pred,match_prob CSV from anymatch_synthetic_inference.ipynb "
              "and re-run with --model_pred_csv.")

    return rule_metrics


def _model_comparison(scored: pd.DataFrame, decided: pd.DataFrame, model_pred_csv: str):
    print("\n" + "=" * 78)
    print("HEAD-TO-HEAD vs MODEL")
    print("=" * 78)
    mp = pd.read_csv(model_pred_csv, dtype={"PATID_A": "string", "PATID_B": "string"})
    if "pred" not in mp.columns:
        raise SystemExit("model_pred_csv needs at least PATID_A,PATID_B,pred columns.")
    keyed = scored.merge(mp[["PATID_A", "PATID_B", "pred"]], on=["PATID_A", "PATID_B"], how="left")
    missing = keyed["pred"].isna().sum()
    if missing:
        print(f"  warning: {missing} pairs had no model prediction (dropped from model metrics).")
    have = keyed.dropna(subset=["pred"]).copy()
    have["pred"] = have["pred"].astype(int)
    _binary_report("MODEL on ALL pairs:", have["label"], have["pred"])

    dec_keys = set(zip(decided["PATID_A"], decided["PATID_B"]))
    mask = [(a, b) in dec_keys for a, b in zip(have["PATID_A"], have["PATID_B"])]
    sub = have[mask]
    if len(sub):
        _binary_report("MODEL on the RULES' decided subset (apples-to-apples):", sub["label"], sub["pred"])
    print("\n(Compare 'RULES on auto-decided subset' vs 'MODEL on the RULES' decided subset'.)")


# ==============================================================================
# Calibration mode
# ==============================================================================
def calibrate(train_csv: str):
    """Print comparator distributions for corrupted positives vs hard negatives so
    the thresholds in rules_alliance.py are justified, not guessed."""
    df = R.load_paired_csv(train_csv)
    df["label"] = df["label"].astype(int)
    print("=" * 78)
    print(f"CALIBRATION on {os.path.basename(train_csv)}  (n={len(df)})")
    print("=" * 78)

    jw_first_pos, jw_first_neg = [], []
    for row in df.itertuples(index=False):
        rd = row._asdict()
        fl, fr = R._s(rd.get("first_name_l")), R._s(rd.get("first_name_r"))
        ll, lr = R._s(rd.get("last_name_l")), R._s(rd.get("last_name_r"))
        if fl and fr and fl != fr:
            (jw_first_pos if rd["label"] == 1 else jw_first_neg).append(R._jw(fl, fr))
        if ll and lr and ll != lr:
            (jw_first_pos if rd["label"] == 1 else jw_first_neg).append(R._jw(ll, lr))

    def _pctl(vals, label):
        if not vals:
            print(f"  {label}: (no samples)")
            return
        a = np.array(vals)
        qs = np.percentile(a, [10, 25, 50, 75, 90])
        print(f"  {label}: n={len(a)}  p10={qs[0]:.3f} p25={qs[1]:.3f} "
              f"p50={qs[2]:.3f} p75={qs[3]:.3f} p90={qs[4]:.3f}")

    print("\nJaro-Winkler on DIFFERING first/last name tokens:")
    _pctl(jw_first_pos, "positives (true matches, name corrupted)")
    _pctl(jw_first_neg, "negatives (different people)")
    print(f"\n  Current TAU_JW = {R.TAU_JW}  (tokens at/above this are treated as the same name).")
    print("  A good threshold sits above the negatives' p90 and below the positives' p25-ish mass.")

    # name_level distribution by label
    print("\nname_level distribution by label:")
    rows = []
    for row in df.itertuples(index=False):
        rd = row._asdict()
        rec_l = {f: rd.get(f"{f}_l") for f in R._FRIENDLY}
        rec_r = {f: rd.get(f"{f}_r") for f in R._FRIENDLY}
        rows.append((rd["label"], R.name_level(rec_l, rec_r)))
    nd = pd.DataFrame(rows, columns=["label", "name_level"])
    print(pd.crosstab(nd["name_level"], nd["label"]).to_string())


# ==============================================================================
# CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate deterministic rules and compare to the model.")
    ap.add_argument("--test_csv", default=DEFAULT_TEST)
    ap.add_argument("--model_pred_csv", default=None,
                    help="Optional PATID_A,PATID_B,pred[,match_prob] CSV from the model.")
    ap.add_argument("--calibrate", default=None, metavar="TRAIN_CSV",
                    help="Run calibration on a training CSV instead of evaluating.")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.calibrate)
    else:
        evaluate(args.test_csv, args.model_pred_csv)


if __name__ == "__main__":
    main()
