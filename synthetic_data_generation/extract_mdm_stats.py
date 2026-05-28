"""Extract aggregate statistics from MDM_Population_cleaned_v1.csv for use in
designing the synthetic dataset (synthetic_data_generation/Synthetic-Dataset-Spec.md).

Outputs only aggregates (counts, rates, histograms, top-N value lists with a
k-anonymity threshold). No raw rows, no PII strings appearing below the
threshold count.

Usage:
    python synthetic_data_generation/extract_mdm_stats.py \
        --input  /path/to/MDM_Population_cleaned_v1.csv \
        --output synthetic_data_generation/synthetic_data_stats.json

Run from the AnyMatch/ directory. The CSV is expected to live outside the repo
for PHI reasons; pass an absolute path.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Fields treated as strings to preserve leading zeros / avoid float coercion.
STRING_DTYPES = {
    "SSN_clean": "string",
    "SSN_raw": "string",
    "last_4_SSN": "string",
    "ZipCD_clean_base": "string",
    "ZipCD_clean_ext": "string",
    "ZipCD_raw": "string",
    "PrimaryPhoneNBR_clean": "string",
    "Phone01NBR_clean": "string",
    "Phone02NBR_clean": "string",
    "Phone03NBR_clean": "string",
    "PATID": "string",
}

# Below this count, value-frequency entries are aggregated into a single
# "<below_threshold>" bucket so no rare value (and thus no near-identifying
# string) escapes the stats file.
K_ANON_THRESHOLD = 20

# How many top values to keep per categorical field (after applying k-anon).
TOP_N = 50

# Fields whose top-N values we want for synthetic generation (drawn from the
# model-facing schema plus a few raw/derived columns that influence generation).
CATEGORICAL_FIELDS = [
    "FirstNM_clean",
    "MiddleNM_clean",
    "LastNM_clean",
    "SuffixNM_clean",
    "CityNM_clean",
    "StateCD_clean",
    "ZipCD_clean_base",
    "SexAtBirthDSC_clean",
    "CountryNM",
]

# Numeric-ish fields we want length histograms for (to model realistic value
# lengths during synthesis).
LENGTH_FIELDS = [
    "FirstNM_clean",
    "MiddleNM_clean",
    "LastNM_clean",
    "AddressLine1_clean",
    "AddressLine2_clean",
    "Email_clean",
]

# Boolean / status columns we want simple value-counts on.
BOOL_FIELDS = ["valid_record"]


def safe_value_counts(series: pd.Series, top_n: int = TOP_N) -> dict:
    """Top-N value counts, with a k-anonymity floor and a tail bucket."""
    counts = series.dropna().astype(str).value_counts()
    counts = counts[counts >= K_ANON_THRESHOLD]
    head = counts.head(top_n)
    below_threshold = int((series.dropna().shape[0]) - int(head.sum()))
    return {
        "top": {str(k): int(v) for k, v in head.items()},
        "below_threshold_total": below_threshold,
        "n_distinct_above_threshold": int(counts.shape[0]),
    }


def length_histogram(series: pd.Series) -> dict:
    lengths = series.dropna().astype(str).str.len()
    if lengths.empty:
        return {}
    return {
        "n": int(lengths.shape[0]),
        "mean": float(lengths.mean()),
        "std": float(lengths.std(ddof=0)),
        "min": int(lengths.min()),
        "p25": int(lengths.quantile(0.25)),
        "p50": int(lengths.quantile(0.50)),
        "p75": int(lengths.quantile(0.75)),
        "p95": int(lengths.quantile(0.95)),
        "max": int(lengths.max()),
    }


def token_stats(series: pd.Series) -> dict:
    """Whitespace-token statistics: how often a name field has >1 token, etc."""
    s = series.dropna().astype(str)
    if s.empty:
        return {}
    token_counts = s.str.split().str.len()
    has_hyphen = s.str.contains("-").mean()
    has_apostrophe = s.str.contains("'").mean()
    return {
        "n": int(s.shape[0]),
        "mean_tokens": float(token_counts.mean()),
        "pct_one_token": float((token_counts == 1).mean()),
        "pct_two_tokens": float((token_counts == 2).mean()),
        "pct_three_plus_tokens": float((token_counts >= 3).mean()),
        "pct_with_hyphen": float(has_hyphen),
        "pct_with_apostrophe": float(has_apostrophe),
    }


def missingness_per_field(df: pd.DataFrame) -> dict:
    """Field-level missingness rate, restricted to `*_clean` and derived cols."""
    cols = [c for c in df.columns if c.endswith("_clean") or c in {
        "last_4_SSN", "ZipCD_clean_base", "ZipCD_clean_ext",
        "full_name_tokens", "full_name_compact", "Phones_set",
        "Address_normalized", "valid_record",
    }]
    out = {}
    n = len(df)
    for c in cols:
        nulls = df[c].isna().sum()
        out[c] = {
            "n_total": int(n),
            "n_missing": int(nulls),
            "pct_missing": float(nulls / n) if n else None,
        }
    return out


def co_missingness(df: pd.DataFrame) -> dict:
    """Joint-missing patterns relevant to synthetic generation."""
    out = {}

    def has(col):
        return col in df.columns

    if has("SSN_clean") and has("last_4_SSN"):
        no_ssn_no_l4 = df["SSN_clean"].isna() & df["last_4_SSN"].isna()
        out["no_ssn_no_last4_pct"] = float(no_ssn_no_l4.mean())
        no_full_yes_l4 = df["SSN_clean"].isna() & df["last_4_SSN"].notna()
        out["no_full_ssn_but_last4_present_pct"] = float(no_full_yes_l4.mean())

    if has("FirstNM_clean") and has("LastNM_clean"):
        only_first = df["FirstNM_clean"].notna() & df["LastNM_clean"].isna()
        only_last = df["FirstNM_clean"].isna() & df["LastNM_clean"].notna()
        neither = df["FirstNM_clean"].isna() & df["LastNM_clean"].isna()
        out["only_first_name_pct"] = float(only_first.mean())
        out["only_last_name_pct"] = float(only_last.mean())
        out["neither_first_nor_last_pct"] = float(neither.mean())

    addr_cols = [c for c in [
        "AddressLine1_clean", "CityNM_clean", "StateCD_clean", "ZipCD_clean_base",
    ] if has(c)]
    if addr_cols:
        no_addr = df[addr_cols].isna().all(axis=1)
        out["no_address_at_all_pct"] = float(no_addr.mean())

    phone_cols = [c for c in [
        "PrimaryPhoneNBR_clean", "Phone01NBR_clean",
        "Phone02NBR_clean", "Phone03NBR_clean",
    ] if has(c)]
    if phone_cols:
        no_phone = df[phone_cols].isna().all(axis=1)
        out["no_phone_at_all_pct"] = float(no_phone.mean())
        phones_present = df[phone_cols].notna().sum(axis=1)
        out["phones_per_record"] = {
            f"n={k}": int((phones_present == k).sum()) for k in range(len(phone_cols) + 1)
        }

    return out


def dob_stats(df: pd.DataFrame) -> dict:
    if "BirthDT_clean" not in df.columns:
        return {}
    dt = pd.to_datetime(df["BirthDT_clean"], errors="coerce")
    nonnull = dt.dropna()
    if nonnull.empty:
        return {"n_nonnull": 0}
    # Decade-bin years for k-anon-friendly reporting.
    decade = (nonnull.dt.year // 10) * 10
    decade_counts = decade.value_counts().sort_index()
    month_counts = nonnull.dt.month.value_counts().sort_index()
    day_counts = nonnull.dt.day.value_counts().sort_index()
    return {
        "n_nonnull": int(nonnull.shape[0]),
        "min_year": int(nonnull.dt.year.min()),
        "max_year": int(nonnull.dt.year.max()),
        "pct_year_2000_plus": float((nonnull.dt.year >= 2000).mean()),
        "decade_histogram": {str(int(k)): int(v) for k, v in decade_counts.items()},
        "month_histogram": {str(int(k)): int(v) for k, v in month_counts.items()},
        "day_of_month_histogram": {str(int(k)): int(v) for k, v in day_counts.items()},
    }


def ssn_stats(df: pd.DataFrame) -> dict:
    if "SSN_clean" not in df.columns:
        return {}
    s = df["SSN_clean"].dropna().astype(str)
    out = {
        "n_total": int(len(df)),
        "n_nonnull_full_ssn": int(s.shape[0]),
        "pct_nonnull_full_ssn": float(s.shape[0] / len(df)) if len(df) else None,
    }
    if "last_4_SSN" in df.columns:
        l4 = df["last_4_SSN"].dropna().astype(str)
        out["n_nonnull_last4"] = int(l4.shape[0])
        out["pct_nonnull_last4"] = float(l4.shape[0] / len(df)) if len(df) else None
    # Cluster size: how many records share a non-null full SSN?
    if not s.empty:
        cluster_sizes = s.value_counts()
        cs_hist = cluster_sizes.value_counts().sort_index()
        out["records_per_ssn_histogram"] = {
            str(int(k)): int(v) for k, v in cs_hist.head(20).items()
        }
        out["pct_ssns_with_multiple_records"] = float(
            (cluster_sizes > 1).sum() / cluster_sizes.shape[0]
        )
    return out


def phone_stats(df: pd.DataFrame) -> dict:
    cols = [c for c in [
        "PrimaryPhoneNBR_clean", "Phone01NBR_clean",
        "Phone02NBR_clean", "Phone03NBR_clean",
    ] if c in df.columns]
    out = {}
    for c in cols:
        s = df[c].dropna().astype(str)
        out[c] = {
            "n_nonnull": int(s.shape[0]),
            "pct_nonnull": float(s.shape[0] / len(df)) if len(df) else None,
        }
        if not s.empty:
            area_codes = s.str[:3].value_counts()
            area_codes = area_codes[area_codes >= K_ANON_THRESHOLD].head(20)
            out[c]["top_area_codes"] = {
                str(k): int(v) for k, v in area_codes.items()
            }
    return out


def email_stats(df: pd.DataFrame) -> dict:
    if "Email_clean" not in df.columns:
        return {}
    s = df["Email_clean"].dropna().astype(str)
    out = {
        "n_nonnull": int(s.shape[0]),
        "pct_nonnull": float(s.shape[0] / len(df)) if len(df) else None,
    }
    if not s.empty:
        domains = s.str.split("@").str[-1]
        dc = domains.value_counts()
        dc = dc[dc >= K_ANON_THRESHOLD].head(30)
        out["top_domains"] = {str(k): int(v) for k, v in dc.items()}
    return out


def zip_state_agreement(df: pd.DataFrame) -> dict:
    """ZIP-prefix-to-state agreement rate (proxy for data-quality, not a rule)."""
    if "ZipCD_clean_base" not in df.columns or "StateCD_clean" not in df.columns:
        return {}
    sub = df[["ZipCD_clean_base", "StateCD_clean"]].dropna()
    if sub.empty:
        return {}
    # Most-common state per 3-digit ZIP prefix; report agreement rate.
    sub = sub.copy()
    sub["zip3"] = sub["ZipCD_clean_base"].astype(str).str[:3]
    mode_state_per_zip3 = sub.groupby("zip3")["StateCD_clean"].agg(
        lambda x: x.value_counts().index[0]
    )
    sub["mode_state"] = sub["zip3"].map(mode_state_per_zip3)
    agree = (sub["mode_state"] == sub["StateCD_clean"]).mean()
    return {
        "zip3_to_state_agreement_rate": float(agree),
        "n_unique_zip3": int(sub["zip3"].nunique()),
    }


def address_stats(df: pd.DataFrame) -> dict:
    out = {}
    if "AddressLine1_clean" in df.columns:
        s = df["AddressLine1_clean"].dropna().astype(str)
        out["addrline1_length"] = length_histogram(s)
        # USPS suffix prevalence (looking at the last whitespace token).
        last_token = s.str.split().str[-1]
        suffix_counts = last_token.value_counts()
        suffix_counts = suffix_counts[suffix_counts >= K_ANON_THRESHOLD].head(30)
        out["addrline1_last_token_top"] = {
            str(k): int(v) for k, v in suffix_counts.items()
        }
        # PO Box prevalence.
        po_box = s.str.match(r"^(PO|P O|POST OFFICE) ?BOX", case=False)
        out["addrline1_pct_po_box"] = float(po_box.mean())
    if "AddressLine2_clean" in df.columns:
        s2 = df["AddressLine2_clean"].dropna().astype(str)
        out["addrline2_nonnull_pct"] = float(s2.shape[0] / len(df)) if len(df) else None
        first_token = s2.str.split().str[0]
        unit_counts = first_token.value_counts()
        unit_counts = unit_counts[unit_counts >= K_ANON_THRESHOLD].head(20)
        out["addrline2_first_token_top"] = {
            str(k): int(v) for k, v in unit_counts.items()
        }
    return out


def cluster_stats_by_key(df: pd.DataFrame, keys: list[str], label: str) -> dict:
    """How many distinct records share a key — proxy for true-positive cluster
    sizes in the population (e.g., same SSN + same DOB)."""
    cols = [c for c in keys if c in df.columns]
    if len(cols) != len(keys):
        return {"note": f"missing one of {keys}"}
    sub = df[cols].dropna()
    if sub.empty:
        return {"n_keys": 0}
    sizes = sub.groupby(cols).size()
    size_hist = sizes.value_counts().sort_index()
    return {
        "label": label,
        "n_keys": int(sizes.shape[0]),
        "n_keys_multi_record": int((sizes > 1).sum()),
        "pct_keys_multi_record": float((sizes > 1).mean()),
        "cluster_size_histogram_head": {
            str(int(k)): int(v) for k, v in size_hist.head(20).items()
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to MDM_Population_cleaned_v1.csv")
    p.add_argument(
        "--output",
        default="synthetic_data_generation/synthetic_data_stats.json",
        help="Where to write the aggregate stats JSON",
    )
    p.add_argument(
        "--k-anon",
        type=int,
        default=K_ANON_THRESHOLD,
        help="Minimum count for a value to appear in top-N lists",
    )
    return p.parse_args()


def main() -> None:
    global K_ANON_THRESHOLD
    args = parse_args()
    K_ANON_THRESHOLD = args.k_anon

    print(f"Reading {args.input} ...", flush=True)
    df = pd.read_csv(args.input, dtype=STRING_DTYPES, low_memory=False)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.", flush=True)

    stats = {
        "source": str(args.input),
        "n_rows": int(len(df)),
        "k_anon_threshold": K_ANON_THRESHOLD,
        "columns": list(df.columns),
        "missingness": missingness_per_field(df),
        "co_missingness": co_missingness(df),
        "valid_record_rate": float(df["valid_record"].mean())
            if "valid_record" in df.columns else None,
        "categorical_top": {
            c: safe_value_counts(df[c]) for c in CATEGORICAL_FIELDS if c in df.columns
        },
        "name_tokens": {
            c: token_stats(df[c]) for c in [
                "FirstNM_clean", "MiddleNM_clean", "LastNM_clean",
            ] if c in df.columns
        },
        "field_lengths": {
            c: length_histogram(df[c]) for c in LENGTH_FIELDS if c in df.columns
        },
        "dob": dob_stats(df),
        "ssn": ssn_stats(df),
        "phones": phone_stats(df),
        "email": email_stats(df),
        "address": address_stats(df),
        "zip_state_agreement": zip_state_agreement(df),
        "clusters": {
            "by_ssn": cluster_stats_by_key(df, ["SSN_clean"], "SSN_clean"),
            "by_ssn_dob": cluster_stats_by_key(
                df, ["SSN_clean", "BirthDT_clean"], "SSN_clean + BirthDT_clean"
            ),
            "by_last4_dob": cluster_stats_by_key(
                df, ["last_4_SSN", "BirthDT_clean"], "last_4_SSN + BirthDT_clean"
            ),
            "by_namecompact_dob": cluster_stats_by_key(
                df, ["full_name_compact", "BirthDT_clean"],
                "full_name_compact + BirthDT_clean",
            ),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2, default=str))
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes).", flush=True)


if __name__ == "__main__":
    main()
