"""Sanity checks for the synthetic dataset (Synthetic-Dataset-Spec.md §12).

Run after generate_synthetic.py:
  python synthetic_data_generation/qa_checks.py --version 1

Structural checks hard-fail; distribution checks (§12.7) warn if outside ±5pp
(they only hold at full scale). Pass --real-header <path> to additionally assert
exact column-name/order parity of blocking_eval against a real cleaned file.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from generate_synthetic import (
    RECORD_SCHEMA, PAIR_MODEL_COLS, INVALID_SUBSTRINGS, INVALID_EXACT,
)

HERE = Path(__file__).resolve().parent
STR_COLS = {c: "string" for c in
            ["SSN_clean", "last_4_SSN", "ZipCD_clean_base", "ZipCD_clean_ext",
             "PrimaryPhoneNBR_clean", "Phone01NBR_clean", "Phone02NBR_clean",
             "Phone03NBR_clean", "PATID", "SSN_raw", "ZipCD_raw"]}

PASS, FAIL, WARN = "  ok ", "FAIL ", "warn "
_n_fail = 0


def check(cond, msg):
    global _n_fail
    print((PASS if cond else FAIL) + msg)
    if not cond:
        _n_fail += 1


def warn(cond, msg):
    print((PASS if cond else WARN) + msg)


# --------------------------------------------------------------------------- #
def valid_ssn(s):
    if not re.fullmatch(r"\d{9}", str(s)):
        return False
    area, group, serial = int(s[:3]), int(s[3:5]), int(s[5:])
    return area not in (0, 666) and not (900 <= area <= 999) and group != 0 and serial != 0


def valid_phone(p):
    if not re.fullmatch(r"\d{10}", str(p)):
        return False
    npa, nxx, line = p[:3], p[3:6], p[6:]
    if npa[0] in "01" or npa in ("555",) or npa[1:] == "11":
        return False
    if nxx[0] in "01" or nxx[:1] == "5" and nxx[1:] == "55" and line[:1] in ("0", "1"):
        return False
    return True


def has_filtered_token(s):
    u = str(s).strip().upper()
    if u in INVALID_EXACT:
        return True
    return any(sub in u for sub in INVALID_SUBSTRINGS)


def load_pairs(path):
    suffixed = {f"{c}_{s}": "string" for c in STR_COLS for s in ("l", "r")}
    return pd.read_csv(path, dtype=suffixed, keep_default_na=False, na_values=[""])


def run(out: Path, v: int, stats: dict, real_header: Path | None):
    rec = pd.read_csv(out / f"blocking_eval_v{v}.csv", dtype=STR_COLS,
                      keep_default_na=False, na_values=[""])
    train = load_pairs(out / f"finetune_train_v{v}.csv")
    test = load_pairs(out / f"finetune_test_v{v}.csv")
    realistic = load_pairs(out / f"realistic_eval_v{v}.csv")

    print("\n== §12.1 schema parity ==")
    check(list(rec.columns) == RECORD_SCHEMA, "blocking_eval columns == RECORD_SCHEMA, in order")
    if real_header:
        real_cols = list(pd.read_csv(real_header, nrows=0).columns)
        expected = [c for c in RECORD_SCHEMA if c != "entity_id"]
        check(list(rec.columns)[:-1] == real_cols,
              "blocking_eval (minus entity_id) == real cleaned header")
    for name, df in (("train", train), ("test", test), ("realistic", realistic)):
        unsuffixed = {c[:-2] for c in df.columns if c.endswith(("_l", "_r"))}
        check(unsuffixed == set(PAIR_MODEL_COLS), f"{name} _l/_r model columns == PAIR_MODEL_COLS")

    print("\n== §12.2 value conventions ==")
    names = pd.concat([rec["FirstNM_clean"], rec["MiddleNM_clean"], rec["LastNM_clean"]]).dropna()
    check(names.map(lambda s: s == s.upper()).all(), "all names uppercase")
    check(names.map(lambda s: s.isascii()).all(), "all names ASCII")
    zips = rec["ZipCD_clean_base"].dropna()
    check(zips.map(lambda z: bool(re.fullmatch(r"\d{5}", z))).all(), "ZIPs are 5-digit strings")

    print("\n== §12.3 valid_record ==")
    check(rec["valid_record"].astype(str).isin(["True", "true"]).all() or rec["valid_record"].all(),
          "every emitted record valid_record=True")

    print("\n== §12.4 label / case agreement ==")
    allpairs = pd.concat([train, test, realistic])
    m = allpairs[allpairs["case_type"].str.startswith("M-")]
    nm = allpairs[allpairs["case_type"].str.startswith("NM-")]
    check((m["label"] == 1).all(), "all M-* pairs label=1")
    check((nm["label"] == 0).all(), "all NM-* pairs label=0")

    print("\n== §12.5 entity disjointness (fine-tune) ==")
    tr_e = set(train["entity_id_a"]) | set(train["entity_id_b"])
    te_e = set(test["entity_id_a"]) | set(test["entity_id_b"])
    check(tr_e.isdisjoint(te_e), "no entity in both train and test")

    print("\n== §12.9 ZIP3 -> State consistency ==")
    z2s = {}
    ok = True
    for _, r in rec[["ZipCD_clean_base", "StateCD_clean"]].dropna().iterrows():
        z3 = r["ZipCD_clean_base"][:3]
        if z3 in z2s and z2s[z3] != r["StateCD_clean"]:
            ok = False
            break
        z2s.setdefault(z3, r["StateCD_clean"])
    check(ok, "each ZIP3 maps to a single state")

    print("\n== §12.10 SSN structural validity ==")
    ssns = rec["SSN_clean"].dropna()
    check(ssns.map(valid_ssn).all(), f"all {len(ssns)} full SSNs structurally valid")
    l4 = rec["last_4_SSN"].dropna()
    check(l4.map(lambda x: bool(re.fullmatch(r"\d{4}", x)) and x != "0000").all(),
          "all last_4_SSN are 4 digits != 0000")

    print("\n== §12.11 SSN <-> last-4 coupling ==")
    both = rec[rec["SSN_clean"].notna() & rec["last_4_SSN"].notna()]
    check((both["SSN_clean"].str[-4:] == both["last_4_SSN"]).all(),
          f"last_4_SSN == SSN[-4:] on all {len(both)} dual-present records")
    check((rec["SSN_clean"].notna() & rec["last_4_SSN"].isna()).sum() == 0,
          "no record has full SSN but null last-4")

    print("\n== §12.12 NANP-valid phones ==")
    allphones = pd.concat([rec[c] for c in
                           ["PrimaryPhoneNBR_clean", "Phone01NBR_clean",
                            "Phone02NBR_clean", "Phone03NBR_clean"]]).dropna()
    check(allphones.map(valid_phone).all(), f"all {len(allphones)} phones NANP-valid")

    print("\n== §12.13 no cleaning-filtered tokens ==")
    namefields = pd.concat([rec["FirstNM_clean"], rec["MiddleNM_clean"],
                            rec["LastNM_clean"], rec["AddressLine1_clean"]]).dropna()
    check(~namefields.map(has_filtered_token).any(),
          "no name/address contains a cleaning-invalid token")

    print("\n== §12.8 name-derivation invariance (M-NAME) ==")
    # Pure token shuffles must preserve full_name_tokens (order-invariant set).
    shuffles = allpairs[allpairs["case_type"].isin(["M-NAME-02", "M-NAME-03", "M-NAME-05"])]
    tok_inv = shuffles.apply(
        lambda r: set(str(r["full_name_tokens_l"]).split()) == set(str(r["full_name_tokens_r"]).split()),
        axis=1)
    check(tok_inv.all() if len(shuffles) else True,
          f"full_name_tokens equal for {len(shuffles)} pure-shuffle pairs (M-NAME-02/03/05)")
    # Hyphenation (M-NAME-01) collapses to one token in the ANNEMARIE form, so it
    # is compact-invariant rather than token-invariant.
    hyphen = allpairs[allpairs["case_type"] == "M-NAME-01"]
    comp_inv = hyphen.apply(
        lambda r: r["full_name_compact_l"] == r["full_name_compact_r"], axis=1)
    check(comp_inv.all() if len(hyphen) else True,
          f"full_name_compact equal for {len(hyphen)} hyphenation pairs (M-NAME-01)")

    print("\n== §12.7 distribution sanity (realistic records, ±5pp) ==")
    def present(col):
        return rec[col].notna().mean()
    targets = {
        "SSN_clean": 0.214, "last_4_SSN": 0.357, "MiddleNM_clean": 0.194,
        "AddressLine1_clean": 0.963, "Email_clean": 0.311, "SexAtBirthDSC_clean": 0.791,
    }
    for col, tgt in targets.items():
        got = present(col)
        warn(abs(got - tgt) <= 0.05, f"{col} present {got:.1%} (target {tgt:.1%})")

    print(f"\n{'='*50}\n{'ALL STRUCTURAL CHECKS PASSED' if _n_fail==0 else f'{_n_fail} CHECK(S) FAILED'}")
    return _n_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--stats", default=str(HERE / "synthetic_data_stats.json"))
    ap.add_argument("--real-header", default=None)
    args = ap.parse_args()
    stats = json.loads(Path(args.stats).read_text())
    rh = Path(args.real_header) if args.real_header else None
    raise SystemExit(1 if run(Path(args.out_dir), args.version, stats, rh) else 0)


if __name__ == "__main__":
    main()
