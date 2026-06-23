"""Shared helpers for silver-label triage and gold-label manual review.

Both the silver-label EDA (`silver_labels/silver_labels_eda.ipynb`) and the
manual gold-labeling notebook (`gold_labeling/gold_labeling.ipynb`) import this
module so they compare fields with *identical* semantics — there is exactly one
definition of "the two records agree on this field".

Responsibilities:
  * `load_records`  — read the cleaned MDM parquet, filter `valid_record`, apply
    the friendly schema (`utils.alliance_schema.prep_record_df`), de-dup on PATID
    and index by it.
  * `merge_pairs`   — attach `<field>_a` / `<field>_b` columns to a PATID pair
    table (silver labels) by reindexing the records frame.
  * `field_status`  — the single comparison primitive returning one of
    {'same','diff','one_missing','both_missing'} for one field of one pair.
  * `status_frame`  — vectorized per-field status for the whole pair table.
  * `render_pair_html` / `legend_html` — colored stacked A-over-B HTML view.

Color convention (used everywhere):
    green  = both present and equal
    red    = both present and different
    yellow = exactly one side missing
    grey   = both sides missing
"""
from __future__ import annotations

import pandas as pd

from utils.alliance_schema import id_str_dtypes, prep_record_df

# Friendly feature columns, in the order they should be displayed / compared.
FIELDS = [
    'first_name', 'middle_name', 'last_name', 'suffix', 'dob', 'ssn', 'ssn4',
    'sex', 'address', 'address2', 'city', 'state', 'zip', 'phone', 'email',
]

# Fields whose "agreement" is set-overlap rather than string equality.
SET_FIELDS = {'phone'}

STATUS_COLORS = {
    'same':         '#b7e1a1',  # green
    'diff':         '#f4a6a6',  # red
    'one_missing':  '#f7e59b',  # yellow
    'both_missing': '#d9d9d9',  # grey
}
STATUS_LABELS = {
    'same':         'both equal',
    'diff':         'different',
    'one_missing':  'one missing',
    'both_missing': 'both missing',
}

_MISSING = {'', 'nan', 'none', 'na', 'n/a', '<na>', 'nat'}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _norm(v) -> str:
    """Stripped uppercase string; '' for any missing sentinel."""
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return '' if s.lower() in _MISSING else s.upper()


def _phone_tokens(v) -> set[str]:
    return {t for t in _norm(v).split() if t}


def _norm_series(s: pd.Series) -> pd.Series:
    """Vectorized `_norm`: returns a 'string' Series with <NA> for missing."""
    out = s.astype('string').str.strip()
    low = out.str.lower()
    out = out.mask(out.isna() | low.isin(_MISSING))
    return out.str.upper()


# ---------------------------------------------------------------------------
# Single-pair comparison (used by the renderer)
# ---------------------------------------------------------------------------
def field_status(a, b, field: str) -> str:
    """One of {'same','diff','one_missing','both_missing'} for one field."""
    if field in SET_FIELDS:
        sa, sb = _phone_tokens(a), _phone_tokens(b)
        ap, bp = bool(sa), bool(sb)
        if not ap and not bp:
            return 'both_missing'
        if not ap or not bp:
            return 'one_missing'
        return 'same' if (sa & sb) else 'diff'
    na, nb = _norm(a), _norm(b)
    ap, bp = bool(na), bool(nb)
    if not ap and not bp:
        return 'both_missing'
    if not ap or not bp:
        return 'one_missing'
    return 'same' if na == nb else 'diff'


# ---------------------------------------------------------------------------
# Vectorized comparison for the whole table (used by the EDA)
# ---------------------------------------------------------------------------
def _phone_status_series(a: pd.Series, b: pd.Series) -> pd.Series:
    sa = [_phone_tokens(x) for x in a]
    sb = [_phone_tokens(x) for x in b]
    out = []
    for x, y in zip(sa, sb):
        if not x and not y:
            out.append('both_missing')
        elif not x or not y:
            out.append('one_missing')
        else:
            out.append('same' if (x & y) else 'diff')
    return pd.Series(out, index=a.index, dtype=object)


def field_status_series(a: pd.Series, b: pd.Series, field: str) -> pd.Series:
    if field in SET_FIELDS:
        return _phone_status_series(a, b)
    na, nb = _norm_series(a), _norm_series(b)
    ap, bp = na.notna(), nb.notna()
    status = pd.Series('both_missing', index=a.index, dtype=object)
    status[ap ^ bp] = 'one_missing'
    both = ap & bp
    status[both & (na == nb)] = 'same'
    status[both & (na != nb)] = 'diff'
    return status


def status_frame(pairs: pd.DataFrame, fields=FIELDS) -> pd.DataFrame:
    """DataFrame of per-field status strings (one column per present field)."""
    cols = {}
    for f in fields:
        ca, cb = f'{f}_a', f'{f}_b'
        if ca in pairs.columns and cb in pairs.columns:
            cols[f] = field_status_series(pairs[ca], pairs[cb], f)
    return pd.DataFrame(cols, index=pairs.index)


# ---------------------------------------------------------------------------
# Data loading / merging
# ---------------------------------------------------------------------------
def load_records(records_path: str, fields=FIELDS) -> pd.DataFrame:
    """Read the cleaned MDM parquet, keep valid records, friendly-rename, index by PATID."""
    records = pd.read_parquet(records_path)
    for col, dt in id_str_dtypes(records.columns).items():
        records[col] = records[col].astype(dt)
    if 'valid_record' in records.columns:
        records = records[records['valid_record']].copy()
    records = prep_record_df(records)
    keep = [f for f in fields if f in records.columns]
    records = records[['PATID'] + keep].drop_duplicates('PATID').set_index('PATID')
    return records


def merge_pairs(pairs_df: pd.DataFrame, records: pd.DataFrame, fields=FIELDS) -> pd.DataFrame:
    """Attach `<field>_a` / `<field>_b` columns to a PATID_A/PATID_B pair table.

    Rows whose PATID is absent from `records` (filtered out as invalid, or simply
    missing) get NaN on that side — flag them with `joinable_mask`.
    """
    feats = [f for f in fields if f in records.columns]
    left = (records[feats].add_suffix('_a')
            .reindex(pairs_df['PATID_A'].values).reset_index(drop=True))
    right = (records[feats].add_suffix('_b')
             .reindex(pairs_df['PATID_B'].values).reset_index(drop=True))
    out = pd.concat([pairs_df.reset_index(drop=True), left, right], axis=1)
    return out


def joinable_mask(pairs: pd.DataFrame, fields=FIELDS) -> pd.Series:
    """True where both sides were joined to a valid record (any feature present)."""
    feats = [f for f in fields if f'{f}_a' in pairs.columns]
    a_ok = pairs[[f'{f}_a' for f in feats]].notna().any(axis=1)
    b_ok = pairs[[f'{f}_b' for f in feats]].notna().any(axis=1)
    return a_ok & b_ok


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _disp(v) -> str:
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    s = str(v)
    return '' if s.lower() in _MISSING else s


def legend_html() -> str:
    chips = ''.join(
        f'<span style="display:inline-block;padding:2px 10px;margin-right:6px;'
        f'border:1px solid #999;border-radius:3px;background:{STATUS_COLORS[k]};">'
        f'{STATUS_LABELS[k]}</span>'
        for k in ('same', 'diff', 'one_missing', 'both_missing')
    )
    return f'<div style="font:13px sans-serif;margin:4px 0;">{chips}</div>'


def render_pair_html(row: pd.Series, fields=FIELDS, title: str | None = None,
                     id_col_a='PATID_A', id_col_b='PATID_B') -> str:
    """Colored stacked HTML for one merged pair row: record A above, B below,
    one column per field, each value cell colored by `field_status`."""
    feats = [f for f in fields if f'{f}_a' in row.index]
    statuses = {f: field_status(row.get(f'{f}_a'), row.get(f'{f}_b'), f) for f in feats}

    base_td = ('padding:3px 8px;border:1px solid #ccc;font:12px monospace;'
               'white-space:nowrap;')
    head_th = ('padding:3px 8px;border:1px solid #ccc;font:11px sans-serif;'
               'background:#f0f0f0;text-align:left;')
    row_lbl = ('padding:3px 8px;border:1px solid #ccc;font:11px sans-serif;'
               'background:#fafafa;font-weight:bold;white-space:nowrap;')

    header = '<th style="%s"></th>' % head_th + ''.join(
        f'<th style="{head_th}">{f}</th>' for f in feats)

    def value_row(side):
        cells = []
        for f in feats:
            bg = STATUS_COLORS[statuses[f]]
            cells.append(f'<td style="{base_td}background:{bg};">'
                         f'{_disp(row.get(f"{f}_{side}"))}</td>')
        return ''.join(cells)

    pa = _disp(row.get(id_col_a))
    pb = _disp(row.get(id_col_b))
    silver = row.get('silver_label', '')
    gold = row.get('gold_label', '')
    cap = title if title is not None else (
        f'silver_label=<b>{_disp(silver)}</b>'
        + (f' &nbsp; gold_label=<b>{_disp(gold)}</b>' if 'gold_label' in row.index else ''))

    return (
        f'<div style="margin:10px 0;">'
        f'<div style="font:12px sans-serif;margin-bottom:2px;">{cap}</div>'
        f'<table style="border-collapse:collapse;">'
        f'<tr>{header}</tr>'
        f'<tr><td style="{row_lbl}">A · {pa}</td>{value_row("a")}</tr>'
        f'<tr><td style="{row_lbl}">B · {pb}</td>{value_row("b")}</tr>'
        f'</table></div>'
    )
