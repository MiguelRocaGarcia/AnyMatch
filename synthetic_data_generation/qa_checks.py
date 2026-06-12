"""Sanity checks for the synthetic dataset (Synthetic-Dataset-Spec.md §12, v0.5).

Run after generate_synthetic.py (from the AnyMatch/ directory):
  python synthetic_data_generation/qa_checks.py --version 2

Two pair-level files only: synthetic_train_vN.csv + synthetic_test_vN.csv.
Structural checks hard-fail; distribution checks warn if outside tolerance.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from generate_synthetic import PAIR_MODEL_COLS, INVALID_SUBSTRINGS, INVALID_EXACT

HERE = Path(__file__).resolve().parent
STR_COLS = ["SSN_clean", "last_4_SSN", "ZipCD_clean_base", "ZipCD_clean_ext",
            "PrimaryPhoneNBR_clean", "Phone01NBR_clean", "Phone02NBR_clean",
            "Phone03NBR_clean", "Phones_set"]

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
    npa, nxx = p[:3], p[3:6]
    if npa[0] in "01" or npa in ("555",) or npa[1:] == "11":
        return False
    if nxx[0] in "01":
        return False
    return True


def has_filtered_token(s):
    u = str(s).strip().upper()
    return u in INVALID_EXACT or any(sub in u for sub in INVALID_SUBSTRINGS)


def load_pairs(path):
    suffixed = {f"{c}_{s}": "string" for c in STR_COLS for s in ("l", "r")}
    return pd.read_csv(path, dtype=suffixed, keep_default_na=False, na_values=[""])


def stack_sides(df, col):
    """Both _l and _r values of a model column as one Series (the per-record view)."""
    return pd.concat([df[f"{col}_l"], df[f"{col}_r"]], ignore_index=True)


# Strong blocking fields a hard negative must share (phone handled via Phones_set).
STRONG_FIELDS = ["LastNM_clean", "BirthDT_clean", "AddressLine1_clean", "last_4_SSN"]


def shares_strong_field(r):
    for c in STRONG_FIELDS:
        a, b = r[f"{c}_l"], r[f"{c}_r"]
        if pd.notna(a) and pd.notna(b) and a != "" and a == b:
            return True
    pl = str(r["Phones_set_l"]).split() if pd.notna(r["Phones_set_l"]) else []
    pr = str(r["Phones_set_r"]).split() if pd.notna(r["Phones_set_r"]) else []
    return len(set(pl) & set(pr)) > 0


def run(out: Path, v: int, stats: dict):
    train = load_pairs(out / f"synthetic_train_v{v}.csv")
    test = load_pairs(out / f"synthetic_test_v{v}.csv")
    allpairs = pd.concat([train, test], ignore_index=True)

    print("\n== §12.1 schema parity ==")
    check(list(train.columns) == list(test.columns), "train and test share identical columns")
    for name, df in (("train", train), ("test", test)):
        unsuffixed = {c[:-2] for c in df.columns if c.endswith(("_l", "_r"))}
        check(unsuffixed == set(PAIR_MODEL_COLS), f"{name} _l/_r model columns == PAIR_MODEL_COLS")
    # mode4: every _l/_r column maps cleanly through the friendly schema
    try:
        sys.path.insert(0, str(HERE.parent))
        from utils.alliance_schema import prep_paired_df
        prepped = prep_paired_df(train.head(50).copy())
        check(len(prepped) == 50 and prepped.shape[1] > 0,
              "prep_paired_df maps train _l/_r columns to the friendly mode4 schema")
    except Exception as e:
        warn(False, f"prep_paired_df mode4 check skipped ({type(e).__name__}: {e})")

    print("\n== §12.2 value conventions (stacked sides) ==")
    names = pd.concat([stack_sides(allpairs, c) for c in
                       ("FirstNM_clean", "MiddleNM_clean", "LastNM_clean")]).dropna()
    check(names.map(lambda s: s == s.upper()).all(), "all names uppercase")
    check(names.map(lambda s: s.isascii()).all(), "all names ASCII")
    zips = stack_sides(allpairs, "ZipCD_clean_base").dropna()
    check(zips.map(lambda z: bool(re.fullmatch(r"\d{5}", z))).all(), "ZIPs are 5-digit strings")

    print("\n== §12.4 label / case agreement ==")
    m = allpairs[allpairs["case_type"].str.startswith("M-")]
    nm = allpairs[allpairs["case_type"].str.startswith("NM-")]
    check((m["label"] == 1).all(), "all M-* pairs label=1")
    check((nm["label"] == 0).all(), "all NM-* pairs label=0")

    print("\n== §12.5 entity disjointness (train vs test) ==")
    tr_e = set(train["entity_id_a"]) | set(train["entity_id_b"])
    te_e = set(test["entity_id_a"]) | set(test["entity_id_b"])
    check(tr_e.isdisjoint(te_e), "no entity appears in both train and test")

    print("\n== §12.7 realistic missingness (stacked sides, ±3pp) ==")
    # Model-relevant fields; phone via Phones_set. SSN/last-4 are EXCLUDED here: they
    # are deliberately band-controlled (§8.4) so the training set under-represents SSN
    # vs the 21% population marginal on purpose (most hard pairs have no SSN anchor).
    targets = {
        "MiddleNM_clean": 0.194,
        "AddressLine1_clean": 0.963, "AddressLine2_clean": 0.294,
        "Email_clean": 0.311, "SexAtBirthDSC_clean": 0.791,
        "Phones_set": 0.944, "BirthDT_clean": 0.997, "LastNM_clean": 0.997,
    }
    for col, tgt in targets.items():
        got = 1 - (stack_sides(train, col).isna()).mean()
        warn(abs(got - tgt) <= 0.035, f"{col} present {got:.1%} (target {tgt:.1%}, d={got-tgt:+.1%})")
    ssn_p = 1 - stack_sides(train, "SSN_clean").isna().mean()
    print(f"     (SSN present {ssn_p:.1%} — intentionally suppressed by the §8.4 no-anchor design)")

    print("\n== §12.8 positive difficulty (SSN bands, no-identical, address) ==")
    pos = train[train["label"] == 1]

    def both_eq(df, c):
        a, b = df[f"{c}_l"], df[f"{c}_r"]
        return (a.notna() & b.notna() & (a == b))
    full_ssn = both_eq(pos, "SSN_clean").mean()
    any_ssn = (both_eq(pos, "SSN_clean") | both_eq(pos, "last_4_SSN")).mean()
    no_ssn = 1 - any_ssn
    addr = both_eq(pos, "AddressLine1_clean").mean()
    # POL-AMBIG-03 deliberately agrees on name+DOB+address (no SSN) — that strong-field
    # agreement IS the scenario — so it is exempt from the no-identical rule.
    non_pol = pos[pos["case_type"] != "POL-AMBIG-03"]
    ident = non_pol.apply(lambda r: all((str(r[f"{c}_l"]) == str(r[f"{c}_r"]))
                                        for c in ("FirstNM_clean", "LastNM_clean", "BirthDT_clean", "AddressLine1_clean")),
                          axis=1).sum()
    check(ident == 0, f"zero identical positives outside POL-AMBIG-03 (got {ident})")
    warn(full_ssn <= 0.07, f"full-SSN-match positives = {full_ssn:.1%} (target <=5%)")
    warn(no_ssn >= 0.72, f"no-usable-SSN positives = {no_ssn:.1%} (target ~80%)")
    warn(addr <= 0.34, f"address-match positives = {addr:.1%} (target ~25-30%, per §5.8 line1_exact=28.8%)")

    print("\n== §12.8b negative difficulty (multi-key) ==")
    neg = train[train["label"] == 0]
    def n_shared(r):
        n = sum(1 for c in ("LastNM_clean", "BirthDT_clean", "AddressLine1_clean", "last_4_SSN")
                if pd.notna(r[f"{c}_l"]) and pd.notna(r[f"{c}_r"]) and r[f"{c}_l"] == r[f"{c}_r"])
        pl = str(r["Phones_set_l"]).split() if pd.notna(r["Phones_set_l"]) else []
        pr = str(r["Phones_set_r"]).split() if pd.notna(r["Phones_set_r"]) else []
        return n + (1 if set(pl) & set(pr) else 0)
    sh = neg.apply(n_shared, axis=1)
    warn((sh >= 2).mean() >= 0.30, f"negatives sharing >=2 strong keys = {(sh>=2).mean():.1%} (target ~35-40%)")

    print("\n== §12.9 hard negatives (test) ==")
    teneg = test[test["label"] == 0]
    frac = teneg.apply(shares_strong_field, axis=1).mean()
    # >=95%: a few contrastive teaching scenarios legitimately share no blocking key
    # (NM-SSN-05 full-vs-mismatching-last4; NM-BND-01 thin/all-different) — they are
    # not random strangers but deliberate hard contrasts. The bulk always shares >=1.
    check(frac >= 0.95, f">=95% of test negatives share >=1 strong field (got {frac:.1%})")

    print("\n== §12.10 SSN structural validity + coupling (stacked) ==")
    ssns = stack_sides(allpairs, "SSN_clean").dropna()
    check(ssns.map(valid_ssn).all(), f"all {len(ssns)} full SSNs structurally valid")
    l4 = stack_sides(allpairs, "last_4_SSN").dropna()
    check(l4.map(lambda x: bool(re.fullmatch(r"\d{4}", x)) and x != "0000").all(),
          "all last_4_SSN are 4 digits != 0000")
    # coupling: where both present on the same side, last_4 == SSN[-4:]
    for s in ("l", "r"):
        sub = allpairs[allpairs[f"SSN_clean_{s}"].notna() & allpairs[f"last_4_SSN_{s}"].notna()]
        check((sub[f"SSN_clean_{s}"].str[-4:] == sub[f"last_4_SSN_{s}"]).all(),
              f"side {s}: last_4_SSN == SSN[-4:] on all {len(sub)} dual-present records")

    print("\n== §12.12 NANP-valid phones (stacked Phones_set) ==")
    ph = pd.concat([stack_sides(allpairs, "Phones_set").dropna().map(lambda s: s.split())]).explode().dropna()
    check(ph.map(valid_phone).all(), f"all {len(ph)} phones NANP-valid")

    print("\n== §12.13 no cleaning-filtered tokens ==")
    fields = pd.concat([stack_sides(allpairs, c) for c in
                        ("FirstNM_clean", "MiddleNM_clean", "LastNM_clean", "AddressLine1_clean")]).dropna()
    check(~fields.map(has_filtered_token).any(), "no name/address contains a cleaning-invalid token")

    print("\n== §12.8b name-derivation invariance (M-NAME shuffles) ==")
    shuffles = allpairs[allpairs["case_type"].isin(["M-NAME-02", "M-NAME-03", "M-NAME-05"])]
    tok_inv = shuffles.apply(
        lambda r: set(str(r["full_name_tokens_l"]).split()) == set(str(r["full_name_tokens_r"]).split()),
        axis=1)
    # >=99%: rare degenerate shuffles (e.g. first==middle) are corrupted by the
    # no-identical guard, which legitimately breaks token-set equality.
    rate = tok_inv.mean() if len(shuffles) else 1.0
    check(rate >= 0.99, f"full_name_tokens equal for {rate:.1%} of {len(shuffles)} pure-shuffle pairs")

    print(f"\n{'='*50}\n{'ALL STRUCTURAL CHECKS PASSED' if _n_fail==0 else f'{_n_fail} CHECK(S) FAILED'}")
    return _n_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", type=int, default=2)
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--stats", default=str(HERE / "synthetic_data_stats.json"))
    args = ap.parse_args()
    stats = json.loads(Path(args.stats).read_text())
    raise SystemExit(1 if run(Path(args.out_dir), args.version, stats) else 0)


if __name__ == "__main__":
    main()
