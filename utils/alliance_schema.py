"""Canonical AllianceChicago feature schema — single source of truth.

The fine-tune training serialization and BOTH inference notebooks
(anymatch_alliance_inference, anymatch_synthetic_inference) must serialize the
same attributes under the same names, or the GPT-2 mode4 checkpoint sees prompt
drift between train and serve and accuracy degrades.

`CANONICAL_RENAMES` maps the technical MDM-cleaned column names (what the
`MDM_Population_cleaned_v1.csv` file and the synthetic generator emit) to the
clean lowercase-English attribute names the model is trained on. This is the
full spec §2 schema: three separate name fields (so the model can learn how
tokens move *between* First/Middle/Last — Hispanic two-surname swaps,
middle-name promotion/demotion, name-order swaps), suffix (Jr/Sr signal),
AddressLine2 (apartment-level disambiguation), full SSN + last-4 as separate
signals, and email.

Dict order is positional — it defines the order attributes appear in the mode4
prompt. Keep training and inference importing this one module.
"""
import ast

import pandas as pd

# Technical MDM-cleaned column name -> clean English attribute name (mode4 prompt).
CANONICAL_RENAMES = {
    'FirstNM_clean':       'first_name',
    'MiddleNM_clean':      'middle_name',
    'LastNM_clean':        'last_name',
    'SuffixNM_clean':      'suffix',
    'BirthDT_clean':       'dob',
    'SSN_clean':           'ssn',      # full 9-digit; missing for ~64% of records
    'last_4_SSN':          'ssn4',     # backup signal where full SSN is absent
    'SexAtBirthDSC_clean': 'sex',
    'AddressLine1_clean':  'address',
    'AddressLine2_clean':  'address2',
    'CityNM_clean':        'city',
    'StateCD_clean':       'state',
    'ZipCD_clean_base':    'zip',
    'Phones_set':          'phone',    # set-on-disk -> flattened by serialize_set_field
    'Email_clean':         'email',
}

FEATURE_COLS_SRC = list(CANONICAL_RENAMES.keys())    # technical column names
FEATURE_COLS     = list(CANONICAL_RENAMES.values())  # friendly attribute names (positional)

# Friendly fields needing special per-record handling before serialization.
SET_FIELDS  = {'phone'}   # stored as a Python set literal on disk
DATE_FIELDS = {'dob'}     # normalize to YYYY-MM-DD

# Columns to force-read as strings so pandas doesn't infer float64 (which prints
# "358467965.0" and strips leading zeros, wrecking exact-match signal). Covers
# both the per-record MDM file and the _l/_r-suffixed paired synthetic files.
_STR_BASES = ['PATID', 'SSN_clean', 'last_4_SSN', 'ZipCD_clean_base', 'ZipCD_clean_ext',
              'PrimaryPhoneNBR_clean', 'Phone01NBR_clean', 'Phone02NBR_clean', 'Phone03NBR_clean']


def id_str_dtypes(columns):
    """dtype= map forcing ID-like columns (and their _l/_r variants) to 'string'."""
    cols = set(columns)
    out = {}
    for base in _STR_BASES:
        for name in (base, f'{base}_l', f'{base}_r'):
            if name in cols:
                out[name] = 'string'
    return out


def serialize_set_field(x):
    """Flatten a Python set-on-disk into a sorted, space-joined string.

    Handles a live set/list/tuple, a stringified literal like "{'3125551234'}"
    (what pandas writes when a set round-trips through CSV), and plain strings /
    NaN. Sorting makes the output canonical so two records with the same tokens
    in different order serialize identically.
    """
    if isinstance(x, (set, list, tuple)):
        return ' '.join(sorted(str(t) for t in x if t and str(t).lower() != 'nan'))
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if not s or s.lower() == 'nan':
        return ''
    if s.startswith(('{', '[', '(')):
        try:
            parsed = ast.literal_eval(s)
            return ' '.join(sorted(str(t) for t in parsed if t and str(t).lower() != 'nan'))
        except (ValueError, SyntaxError):
            return s
    return s


def prep_record_df(records):
    """Per-record path (alliance inference): rename technical -> friendly and apply
    set/date transforms ONCE on the record table before the per-side join.

    Returns a copy with friendly-named feature columns. Non-feature columns ride
    through untouched. Date conversion is done here (not per-side) to avoid the
    pandas index-alignment trap that silently nulls one side.
    """
    present = [c for c in FEATURE_COLS_SRC if c in records.columns]
    out = records.rename(columns={c: CANONICAL_RENAMES[c] for c in present}).copy()
    for f in SET_FIELDS:
        if f in out.columns:
            out[f] = out[f].apply(serialize_set_field)
    for f in DATE_FIELDS:
        if f in out.columns:
            out[f] = pd.to_datetime(out[f], errors='coerce').dt.strftime('%Y-%m-%d')
    return out


def prep_paired_df(df, sides=('l', 'r')):
    """Already-paired path (synthetic inference): take a CSV whose model columns
    are technical names suffixed `_l` / `_r` (e.g. FirstNM_clean_l) plus
    provenance columns, and emit a wide df with friendly `_l` / `_r` feature
    columns ready for predict_alliance.py.

    Every `_l`/`_r` column NOT in the canonical schema is dropped so it can't leak
    into the prompt; provenance columns (PATID_A/B, label, entity_id_a/b,
    case_type, ...) that don't carry an `_l`/`_r` suffix ride through.
    """
    passthrough = [c for c in df.columns
                   if not (c.endswith('_l') or c.endswith('_r'))]
    out = df[passthrough].copy()
    for side in sides:
        for src, friendly in CANONICAL_RENAMES.items():
            col = f'{src}_{side}'
            s = df[col] if col in df.columns else pd.Series([pd.NA] * len(df), index=df.index)
            if friendly in SET_FIELDS:
                s = s.apply(serialize_set_field)
            elif friendly in DATE_FIELDS:
                s = pd.to_datetime(s, errors='coerce').dt.strftime('%Y-%m-%d')
            out[f'{friendly}_{side}'] = s.values
    return out
