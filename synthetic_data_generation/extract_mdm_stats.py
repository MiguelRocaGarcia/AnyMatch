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
import itertools
import json
import random
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


# ---------------------------------------------------------------------------
# Within-cluster field agreement (true-positive proxy) and joint distributions.
# These calibrate the synthetic corruption model (Spec §7) and the correlated
# entity sampling (Spec §6) from real data. All outputs are aggregate rates /
# counts only -- no raw values escape, so they remain k-anon safe.
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str, max_d: int = 4) -> int:
    """Plain Levenshtein edit distance with an early-exit length gate."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_d:
        return max_d + 1
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def _dob_relation(a, b) -> str:
    """Classify the relationship between two parsed DOB timestamps."""
    if pd.isna(a) or pd.isna(b):
        return "missing"
    if a == b:
        return "exact"
    # Month/day transposition (e.g. 1985-01-15 vs 1985-10-15), valid only when
    # both day and month are <= 12.
    if a.year == b.year and a.month == b.day and a.day == b.month:
        return "month_day_transpose"
    if a.month == b.month and a.day == b.day and abs(a.year - b.year) == 1:
        return "off_by_one_year"
    if a.year == b.year and a.month == b.month and abs(a.day - b.day) == 1:
        return "off_by_one_day"
    return "other"


def _val(d: dict, idx):
    """Fetch a cell from a column-as-dict, returning None for NaN/missing."""
    v = d.get(idx)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return v


def within_cluster_agreement(
    df: pd.DataFrame, key_cols: list[str], label: str,
    seed: int = 0, max_pairs: int = 300_000,
) -> dict:
    """Field-agreement rates over within-cluster record *pairs*, where a cluster
    is a group sharing `key_cols` (a same-person proxy -- strongest for full
    `SSN_clean`). This is the empirical corruption model that Spec §7 budgets
    should be calibrated against (Spec §5.9).

    Rates are conditional: each metric's denominator is the number of pairs
    where the relevant field is present on *both* sides.
    """
    cols = [c for c in key_cols if c in df.columns]
    if len(cols) != len(key_cols):
        return {"note": f"missing one of {key_cols}"}
    sub = df.dropna(subset=cols)
    if sub.empty:
        return {"label": label, "n_pairs": 0}

    rng = random.Random(seed)
    pairs: list[tuple] = []
    n_clusters_multi = 0
    for _, idx in sub.groupby(cols).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        n_clusters_multi += 1
        pairs.extend(itertools.combinations(idx, 2))
    n_pairs_total = len(pairs)
    if n_pairs_total == 0:
        return {"label": label, "n_pairs": 0, "n_clusters_multi_record": 0}
    sampled = n_pairs_total > max_pairs
    if sampled:
        pairs = rng.sample(pairs, max_pairs)

    fields = [
        "FirstNM_clean", "LastNM_clean", "MiddleNM_clean",
        "full_name_compact", "full_name_tokens",
        "AddressLine1_clean", "CityNM_clean", "StateCD_clean",
        "ZipCD_clean_base", "Phones_set", "Email_clean", "SexAtBirthDSC_clean",
    ]
    data = {f: df[f].to_dict() for f in fields if f in df.columns}
    dob = (
        pd.to_datetime(df["BirthDT_clean"], errors="coerce").to_dict()
        if "BirthDT_clean" in df.columns else {}
    )

    agree: Counter = Counter()   # numerator: # pairs that agree on metric
    denom: Counter = Counter()   # denominator: # pairs where metric is defined

    def both(field, i, j):
        a = _val(data.get(field, {}), i)
        b = _val(data.get(field, {}), j)
        return a, b

    for i, j in pairs:
        # --- name fields ---
        for fld, key in (("FirstNM_clean", "first_exact"),
                         ("LastNM_clean", "last_exact")):
            a, b = both(fld, i, j)
            if a is not None and b is not None:
                denom[key] += 1
                agree[key] += int(a == b)

        a, b = both("full_name_compact", i, j)
        if a is not None and b is not None:
            denom["name_compact_exact"] += 1
            agree["name_compact_exact"] += int(a == b)
            d = _levenshtein(str(a), str(b))
            denom["name_compact_editdist_le1"] += 1
            agree["name_compact_editdist_le1"] += int(d <= 1)
            denom["name_compact_editdist_le2"] += 1
            agree["name_compact_editdist_le2"] += int(d <= 2)

        a, b = both("full_name_tokens", i, j)
        if a is not None and b is not None:
            denom["name_tokens_set_equal"] += 1
            agree["name_tokens_set_equal"] += int(set(str(a).split()) == set(str(b).split()))

        # middle-name behaviour: agreement when both present; one-missing rate.
        a, b = both("MiddleNM_clean", i, j)
        denom["middle_pairs"] += 1
        if a is not None and b is not None:
            denom["middle_both_present"] += 1
            agree["middle_both_present_equal"] += int(a == b)
        if (a is None) != (b is None):
            agree["middle_exactly_one_missing"] += 1

        # --- DOB ---
        if dob:
            rel = _dob_relation(dob.get(i), dob.get(j))
            if rel != "missing":
                denom["dob_present"] += 1
                agree[f"dob_{rel}"] += 1

        # --- address / geo ---
        for fld, key in (("AddressLine1_clean", "address1_exact"),
                         ("CityNM_clean", "city_exact"),
                         ("StateCD_clean", "state_exact"),
                         ("ZipCD_clean_base", "zip_exact")):
            a, b = both(fld, i, j)
            if a is not None and b is not None:
                denom[key] += 1
                agree[key] += int(a == b)

        # --- phones (set overlap) ---
        a, b = both("Phones_set", i, j)
        if a is not None and b is not None:
            sa, sb = set(str(a).split()), set(str(b).split())
            if sa and sb:
                denom["phone_overlap"] += 1
                agree["phone_overlap_ge1"] += int(len(sa & sb) >= 1)

        # --- email / sex ---
        for fld, key in (("Email_clean", "email_exact"),
                         ("SexAtBirthDSC_clean", "sex_exact")):
            a, b = both(fld, i, j)
            if a is not None and b is not None:
                denom[key] += 1
                agree[key] += int(a == b)

    def rate(num_key, den_key):
        d = denom.get(den_key, 0)
        return {
            "n": int(agree.get(num_key, 0)),
            "denom": int(d),
            "rate": float(agree.get(num_key, 0) / d) if d else None,
        }

    n = len(pairs)
    return {
        "label": label,
        "n_clusters_multi_record": int(n_clusters_multi),
        "n_pairs_total": int(n_pairs_total),
        "n_pairs_sampled": int(n),
        "sampled": bool(sampled),
        "name": {
            "first_exact": rate("first_exact", "first_exact"),
            "last_exact": rate("last_exact", "last_exact"),
            "compact_exact": rate("name_compact_exact", "name_compact_exact"),
            "compact_editdist_le1": rate("name_compact_editdist_le1", "name_compact_editdist_le1"),
            "compact_editdist_le2": rate("name_compact_editdist_le2", "name_compact_editdist_le2"),
            "tokens_set_equal": rate("name_tokens_set_equal", "name_tokens_set_equal"),
        },
        "middle": {
            "both_present_equal": rate("middle_both_present_equal", "middle_both_present"),
            "exactly_one_missing_rate": float(
                agree.get("middle_exactly_one_missing", 0) / denom["middle_pairs"]
            ) if denom.get("middle_pairs") else None,
        },
        "dob": {
            rel: rate(f"dob_{rel}", "dob_present")
            for rel in ("exact", "off_by_one_day", "off_by_one_year",
                        "month_day_transpose", "other")
        },
        "address": {
            "line1_exact": rate("address1_exact", "address1_exact"),
            "city_exact": rate("city_exact", "city_exact"),
            "state_exact": rate("state_exact", "state_exact"),
            "zip_exact": rate("zip_exact", "zip_exact"),
        },
        "phone_overlap_ge1": rate("phone_overlap_ge1", "phone_overlap"),
        "email_exact": rate("email_exact", "email_exact"),
        "sex_exact": rate("sex_exact", "sex_exact"),
    }


def geo_joint(df: pd.DataFrame) -> dict:
    """Joint (City, State, ZIP3) distribution -- lets §6 sample geography as a
    correlated block instead of three independent marginals."""
    cols = ["CityNM_clean", "StateCD_clean", "ZipCD_clean_base"]
    if not all(c in df.columns for c in cols):
        return {}
    sub = df[cols].dropna().copy()
    if sub.empty:
        return {}
    sub["zip3"] = sub["ZipCD_clean_base"].astype(str).str[:3]
    combo = sub["CityNM_clean"].astype(str) + "|" + sub["StateCD_clean"].astype(str) + "|" + sub["zip3"]
    vc = combo.value_counts()
    head = vc[vc >= K_ANON_THRESHOLD].head(TOP_N)
    return {
        "format": "City|State|ZIP3",
        "top": {str(k): int(v) for k, v in head.items()},
        "below_threshold_total": int(vc.sum() - head.sum()),
        "n_distinct_above_threshold": int((vc >= K_ANON_THRESHOLD).sum()),
    }


def missingness_patterns(df: pd.DataFrame) -> dict:
    """Joint present/absent pattern over the identifier-bearing fields, so §6
    can sample realistic *co-missingness* (thin transient records miss SSN +
    address + phone together) rather than nulling each field independently."""
    # Single-column bits.
    key_map = {
        "ssn_full": "SSN_clean",
        "ssn_last4": "last_4_SSN",
        "middle": "MiddleNM_clean",
        "email": "Email_clean",
        "sex": "SexAtBirthDSC_clean",
    }
    # Composite bits: "address" = any address field present (matches the
    # "no address at all" 2.4% in co_missingness, i.e. step-6 present-rate
    # ~0.976); "phone" = any phone slot present (matches the phones-per-record
    # 0-bucket ~5.6%, i.e. step-7 present-rate ~0.944). Keying these on a single
    # column (AddressLine1 / PrimaryPhoneNBR) would understate presence and
    # desync the pattern from the §6 step marginals.
    addr_cols = [c for c in ["AddressLine1_clean", "CityNM_clean",
                             "StateCD_clean", "ZipCD_clean_base"] if c in df.columns]
    phone_cols = [c for c in ["PrimaryPhoneNBR_clean", "Phone01NBR_clean",
                              "Phone02NBR_clean", "Phone03NBR_clean"] if c in df.columns]

    bits = {name: df[col].notna() for name, col in key_map.items() if col in df.columns}
    if addr_cols:
        bits["address"] = df[addr_cols].notna().any(axis=1)
    if phone_cols:
        bits["phone"] = df[phone_cols].notna().any(axis=1)
    if not bits:
        return {}
    # Stable, readable field order.
    desired = ["ssn_full", "ssn_last4", "middle", "address", "email", "phone", "sex"]
    order = [name for name in desired if name in bits]
    present = pd.DataFrame({name: bits[name] for name in order})
    pat = present[order].apply(lambda r: "".join("1" if r[c] else "0" for c in order), axis=1)
    vc = pat.value_counts()
    head = vc[vc >= K_ANON_THRESHOLD].head(TOP_N)
    return {
        "fields_order": order,
        "pattern_legend": "each char is 1=present / 0=missing, in fields_order",
        "top_patterns": {str(k): int(v) for k, v in head.items()},
        "below_threshold_total": int(vc.sum() - head.sum()),
        "n_distinct_above_threshold": int((vc >= K_ANON_THRESHOLD).sum()),
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
        # True-positive proxy: field agreement over within-cluster record pairs.
        # `by_ssn` is the highest-purity same-person signal; calibrates §7
        # corruption budgets. `by_last4_dob` is lower-purity (last-4 collisions)
        # and is informative mostly for negative/collision calibration.
        "within_cluster_agreement": {
            "by_ssn": within_cluster_agreement(df, ["SSN_clean"], "SSN_clean"),
            "by_ssn_dob": within_cluster_agreement(
                df, ["SSN_clean", "BirthDT_clean"], "SSN_clean + BirthDT_clean"
            ),
            "by_last4_dob": within_cluster_agreement(
                df, ["last_4_SSN", "BirthDT_clean"], "last_4_SSN + BirthDT_clean"
            ),
        },
        # Joint distributions for correlated entity sampling (§6).
        "geo_joint": geo_joint(df),
        "missingness_patterns": missingness_patterns(df),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2, default=str))
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes).", flush=True)


if __name__ == "__main__":
    main()
