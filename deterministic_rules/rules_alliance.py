"""Deterministic entity-resolution rule engine for AllianceChicago patient pairs.

Given two patient records, returns one of three decisions:
    match | non_match | review

This is a hand-authored, fully deterministic cascade -- NOT Fellegi-Sunter (no
weight summation, no learned thresholds). Every decision carries a `rule_id` that
maps 1:1 to an entry in `deterministic_rules/RULES.md`, so any label is traceable
back to the rule that produced it.

Design is precision-first: when no rule can decide a pair with high confidence it
returns `review` rather than guessing. See RULES.md for the empirical grounding.

Reuses `utils.alliance_schema.prep_paired_df` (friendly column names, train/serve
parity) and the generator's nickname / pools so name-equivalence matches how the
synthetic positives were created.

CLI (run from the AnyMatch repo root):
    python deterministic_rules/rules_alliance.py \
        --input_csv data/synthetic/synthetic_test_v3.csv \
        --output_csv /tmp/rules_out.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import pandas as pd

# --- make the AnyMatch repo root importable regardless of cwd -----------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.alliance_schema import id_str_dtypes, prep_paired_df  # noqa: E402

try:
    import jellyfish
except ImportError as exc:  # pragma: no cover
    raise ImportError("rules_alliance.py needs the `jellyfish` package: pip install jellyfish") from exc


# ==============================================================================
# Decisions
# ==============================================================================
MATCH = "match"
NON_MATCH = "non_match"
REVIEW = "review"


# ==============================================================================
# Tunable thresholds (justified by `evaluate_rules.py --calibrate`)
# ==============================================================================
TAU_JW = 0.92      # Jaro-Winkler cut for single-token typo equivalence
LEV_MAX = 1        # max edit distance for typo / email-domain equivalence
PREFIX_MIN = 4     # min token length for truncation/prefix equivalence


# ==============================================================================
# Nickname / name-equivalence resources (generator parity)
# ==============================================================================
_POOLS_DIR = os.path.join(_REPO_ROOT, "synthetic_data_generation", "pools")


def _load_nickname_groups():
    """Build symmetric nickname equivalence groups from the generator's pool.

    Returns a list of frozensets, each a set of mutually-equivalent name tokens
    (the legal name plus all its nicknames).
    """
    path = os.path.join(_POOLS_DIR, "nicknames.json")
    groups = []
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
        for legal, nicks in raw.items():
            grp = {legal.upper()} | {n.upper() for n in nicks}
            groups.append(frozenset(grp))
    return groups


_NICK_GROUPS = _load_nickname_groups()
# token -> set of group indices it belongs to (for O(1) equivalence lookup)
_NICK_INDEX: dict[str, set[int]] = {}
for _i, _grp in enumerate(_NICK_GROUPS):
    for _tok in _grp:
        _NICK_INDEX.setdefault(_tok, set()).add(_i)


# ==============================================================================
# Small value helpers
# ==============================================================================
def _s(v) -> str:
    """Normalize any cell to a stripped uppercase string ('' when missing)."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    if s.lower() in ("", "nan", "none", "na", "n/a", "<na>", "nat"):
        return ""
    return s.upper()


def _digits(v) -> str:
    return "".join(ch for ch in _s(v) if ch.isdigit())


def _levenshtein(a: str, b: str) -> int:
    return jellyfish.levenshtein_distance(a, b)


def _damerau(a: str, b: str) -> int:
    """Edit distance counting a single adjacent transposition as one edit, so a
    common typo like MARY<->AMRY or DIAZ<->IDAZ is distance 1 (Levenshtein gives 2)."""
    return jellyfish.damerau_levenshtein_distance(a, b)


def _jw(a: str, b: str) -> float:
    return jellyfish.jaro_winkler_similarity(a, b)


# ==============================================================================
# C1 -- Name
# ==============================================================================
def name_tokens(first: str, middle: str, last: str) -> list[str]:
    """Union of name tokens, split on whitespace and hyphen (mirrors
    docs/Data-Cleaning-Guide.md::full_name_tokens). Apostrophes are not split."""
    toks: list[str] = []
    for field in (first, middle, last):
        s = _s(field)
        if not s:
            continue
        for chunk in s.replace("-", " ").split():
            if chunk:
                toks.append(chunk)
    return toks


def name_compact(first: str, middle: str, last: str) -> str:
    """Letters-only concatenation of the name fields (order: first, middle, last)."""
    s = "".join((_s(first), _s(middle), _s(last)))
    return "".join(ch for ch in s if ch.isalpha())


def _nick_equiv(a: str, b: str) -> bool:
    ga, gb = _NICK_INDEX.get(a), _NICK_INDEX.get(b)
    return bool(ga and gb and (ga & gb))


def tokens_equiv(a: str, b: str) -> bool:
    """True if two single name tokens are plausibly the same name token."""
    if a == b:
        return True
    if _nick_equiv(a, b):
        return True
    # initial: one is a single letter prefixing the other (M <-> MICHAEL)
    if len(a) == 1 and b.startswith(a):
        return True
    if len(b) == 1 and a.startswith(b):
        return True
    lo, hi = sorted((a, b), key=len)
    # truncation: prefix relationship, both reasonably long (CHRISTO <-> CHRISTOPHER)
    if len(lo) >= PREFIX_MIN and hi.startswith(lo):
        return True
    # single-edit typo (incl. one transposition) on longer tokens
    if len(hi) >= 4 and (_damerau(a, b) <= LEV_MAX or _jw(a, b) >= TAU_JW):
        return True
    return False


def _greedy_match_count(small: list[str], large: list[str]) -> int:
    """Greedy count of tokens in `small` that have an equivalent (unused) token in `large`."""
    used = [False] * len(large)
    matched = 0
    for t in small:
        for j, u in enumerate(large):
            if not used[j] and tokens_equiv(t, u):
                used[j] = True
                matched += 1
                break
    return matched


def name_level(rec_l: dict, rec_r: dict) -> str:
    tl = name_tokens(rec_l["first_name"], rec_l["middle_name"], rec_l["last_name"])
    tr = name_tokens(rec_r["first_name"], rec_r["middle_name"], rec_r["last_name"])
    if not tl or not tr:
        return "weak"  # can't compare -> defer, never auto-reject on missing name

    sl, sr = set(tl), set(tr)
    if sl == sr:
        return "exact_tokens"
    cl = name_compact(rec_l["first_name"], rec_l["middle_name"], rec_l["last_name"])
    cr = name_compact(rec_r["first_name"], rec_r["middle_name"], rec_r["last_name"])
    if cl and cl == cr:
        return "exact_tokens"

    small, large = (tl, tr) if len(tl) <= len(tr) else (tr, tl)
    matched = _greedy_match_count(small, large)
    if matched == len(small):          # smaller set fully covered under equivalence
        return "strong"
    if matched >= 2:                   # partial overlap, conflicting extras both sides
        return "weak"
    return "disagree"                  # <=1 token shared


# ==============================================================================
# C2 -- DOB
# ==============================================================================
def _parse_date(v):
    s = _s(v)
    if not s:
        return None
    s = s[:10]
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        ts = pd.to_datetime(s, errors="coerce")
        return None if pd.isna(ts) else ts.date()


def dob_level(rec_l: dict, rec_r: dict) -> str:
    a, b = _parse_date(rec_l["dob"]), _parse_date(rec_r["dob"])
    if a is None or b is None:
        return "missing"
    if a == b:
        return "exact"
    # off-by-one year (same month/day)
    if a.month == b.month and a.day == b.day and abs(a.year - b.year) == 1:
        return "near"
    # off-by-one day (same year/month)
    if a.year == b.year and a.month == b.month and abs(a.day - b.day) == 1:
        return "near"
    # month <-> day transposition
    if a.year == b.year and a.month == b.day and a.day == b.month:
        return "near"
    return "disagree"


# ==============================================================================
# C3 -- SSN / last-4
# ==============================================================================
def _full_ssn(rec: dict) -> str:
    d = _digits(rec["ssn"])
    return d if len(d) == 9 else ""


def _last4(rec: dict) -> str:
    full = _full_ssn(rec)
    if full:
        return full[-4:]
    d = _digits(rec["ssn4"])
    return d if len(d) == 4 else ""


# ==============================================================================
# C4 -- Address
# ==============================================================================
def addr_level(rec_l: dict, rec_r: dict) -> str:
    al, ar = _s(rec_l["address"]), _s(rec_r["address"])
    if not al or not ar:
        return "missing"
    if al == ar:
        return "same_line1"
    zl, zr = _digits(rec_l["zip"])[:5], _digits(rec_r["zip"])[:5]
    if zl and zl == zr:
        return "same_zip_diff_street"
    return "diff"


# ==============================================================================
# C5 -- Phone
# ==============================================================================
def _phone_set(rec: dict) -> set[str]:
    return {p for p in _s(rec["phone"]).split() if len(_digits(p)) == 10}


def phone_overlap(rec_l: dict, rec_r: dict) -> bool:
    return bool(_phone_set(rec_l) & _phone_set(rec_r))


# ==============================================================================
# C6 -- Email
# ==============================================================================
def email_equal(rec_l: dict, rec_r: dict) -> bool:
    el, er = _s(rec_l["email"]), _s(rec_r["email"])
    return bool(el) and el == er


def email_domain_typo(rec_l: dict, rec_r: dict) -> bool:
    el, er = _s(rec_l["email"]), _s(rec_r["email"])
    if not el or not er or "@" not in el or "@" not in er:
        return False
    ll, dl = el.rsplit("@", 1)
    lr, dr = er.rsplit("@", 1)
    return ll == lr and dl != dr and _levenshtein(dl, dr) <= LEV_MAX


# ==============================================================================
# C7 -- Discriminators
# ==============================================================================
def suffix_conflict(rec_l: dict, rec_r: dict) -> bool:
    sl, sr = _s(rec_l["suffix"]), _s(rec_r["suffix"])
    return bool(sl) and bool(sr) and sl != sr


def demographic_evidence(rec_l: dict, rec_r: dict):
    """Tally corroboration vs. contradiction for a demographic (no-strong-ID) pair.

    Returns (contradictions, link, addr_level) where:
      * `contradictions` = number of identity fields present on BOTH sides that
        independently disagree, among {address (different street), phone (no
        overlap), email (different)}. A true match has at most ONE (the single
        corrupted dimension); a forced hard-negative disagrees on several.
      * `link` = True if a phone or email is shared -- a positive signal two
        different same-named people are unlikely to share.
    """
    addr = addr_level(rec_l, rec_r)

    def both(f):
        return bool(_s(rec_l[f])) and bool(_s(rec_r[f]))

    addr_diff = both("address") and addr == "diff"
    phone_link = phone_overlap(rec_l, rec_r)
    phone_diff = both("phone") and not phone_link
    email_link = email_equal(rec_l, rec_r) or email_domain_typo(rec_l, rec_r)
    email_diff = both("email") and not email_link
    contradictions = int(addr_diff) + int(phone_diff) + int(email_diff)
    return contradictions, (phone_link or email_link), addr


# ==============================================================================
# Decision cascade
# ==============================================================================
def classify_pair(rec_l: dict, rec_r: dict):
    """Return (decision, rule_id, reason). First matching rule wins."""
    nm = name_level(rec_l, rec_r)
    db = dob_level(rec_l, rec_r)
    name_ok = nm != "disagree"
    dob_ok = db != "disagree"
    name_strong = nm in ("exact_tokens", "strong")

    full_l, full_r = _full_ssn(rec_l), _full_ssn(rec_r)
    l4_l, l4_r = _last4(rec_l), _last4(rec_r)

    # ---- SSN tier ----
    if full_l and full_r:
        if full_l == full_r:
            if name_ok or dob_ok:
                return MATCH, "R-SSN-MATCH", f"full SSN equal; name={nm}, dob={db}"
            return NON_MATCH, "R-SSN-COLLISION", "full SSN equal but name & dob both disagree (collision)"
        # full SSNs present and unequal
        if nm == "exact_tokens" and db == "exact":
            return REVIEW, "R-SSN-CONFLICT", "full SSNs differ yet name & dob identical (possible SSN typo)"
        return NON_MATCH, "R-SSN-CONFLICT", "full SSNs present and differ"

    # one full + one last-4 (cross comparison)
    if (full_l and l4_r) or (full_r and l4_l):
        fside = full_l or full_r
        oside = l4_r if full_l else l4_l
        if fside[-4:] == oside:
            if name_strong and db in ("exact", "near"):
                return MATCH, "R-SSN-L4-MATCH", f"full[-4:]==last4; name={nm}, dob={db}"
            if nm == "disagree":
                return NON_MATCH, "R-L4-COLLISION", "last-4 matches but name disagrees (collision)"
            return REVIEW, "R-L4-REVIEW", f"last-4 matches; name={nm}, dob={db}"
        # negative coupling: full SSN vs mismatching last-4
        if name_strong and db == "exact":
            return REVIEW, "R-SSN-L4-CONFLICT", "full SSN vs mismatching last-4 but name & dob agree"
        return NON_MATCH, "R-SSN-L4-CONFLICT", "full SSN vs mismatching last-4"

    # ---- Last-4 tier (both sides have a last-4) ----
    if l4_l and l4_r:
        if l4_l == l4_r:
            if name_strong and db in ("exact", "near"):
                return MATCH, "R-L4-MATCH", f"last-4 equal; name={nm}, dob={db}"
            if nm == "disagree":
                return NON_MATCH, "R-L4-COLLISION", "last-4 equal but name disagrees (collision)"
            return REVIEW, "R-L4-REVIEW", f"last-4 equal; name={nm}, dob={db}"
        # last-4 present on both and differ
        if nm == "exact_tokens" and db == "exact":
            return REVIEW, "R-L4-CONFLICT", "last-4 differs yet name & dob identical"
        return NON_MATCH, "R-L4-CONFLICT", "last-4 present on both sides and differs"

    # ---- Demographic tier (no usable strong-ID pair) ----
    # Exploit the scenario structure: a true match agrees on name + DOB and
    # disagrees on at most ONE other dimension (the single corrupted field); a
    # hard negative either disagrees on the name, has a large DOB gap, or
    # disagrees on several independent fields at once.
    if nm == "disagree":
        return NON_MATCH, "R-DEMO-NAMEDIFF", "no strong ID and names disagree"
    if suffix_conflict(rec_l, rec_r):
        return NON_MATCH, "R-DEMO-JRSR", f"generational suffix conflict; name={nm}"
    if db == "disagree":
        # Real matches only ever drift DOB by ~1 (off-by-one / transpose => 'near').
        # A large gap with an agreeing name is a different person (NM-COMMON, NM-HARD).
        return NON_MATCH, "R-DEMO-DOBCONTRA", "name agrees but DOB gap too large for a single corruption"

    contradictions, link, addr = demographic_evidence(rec_l, rec_r)

    if nm == "weak":
        # name itself is uncertain (e.g. shared first+last but conflicting extra
        # tokens -- sibling/twin look-alike). Only truly ambiguous when DOB agrees.
        return REVIEW, "R-DEMO-AMBIG", "partial name match, no strong ID (sibling/twin vs same person)"

    # nm in {exact_tokens, strong}, dob in {exact, near, missing}.
    # A shared phone or email positively links the two records -- evidence two
    # different same-named people are unlikely to produce. (A shared *address* is
    # NOT a safe link: households / twins share it -- see the ambiguous case below.)
    if link:
        return MATCH, "R-DEMO-MATCH", "name+dob agree and a shared phone/email links the records"

    if contradictions == 0:
        if addr == "same_line1":
            # name + dob + address all identical with nothing else to tell them
            # apart: POL-AMBIG-03 (a match) vs a twin/triplet (a non-match) are
            # constructed to be indistinguishable here -> the one truly ambiguous case.
            return REVIEW, "R-DEMO-AMBIG", "name+dob+address identical, no distinguishing signal (POL-AMBIG vs twin)"
        # thin record: name+dob agree and no present field contradicts.
        return MATCH, "R-DEMO-MATCH", "name+dob agree, no present field contradicts"

    # name+dob agree but a contact field differs with nothing to confirm identity:
    # a person who moved / changed contact vs. a same-name+dob coincidence are
    # genuinely indistinguishable without a strong ID -> defer to a human.
    return REVIEW, "R-DEMO-AMBIG", "name+dob agree but contact info differs and nothing links them (mover vs namesake)"


# ==============================================================================
# Batch
# ==============================================================================
_FRIENDLY = [
    "first_name", "middle_name", "last_name", "suffix", "dob", "ssn", "ssn4",
    "sex", "address", "address2", "city", "state", "zip", "phone", "email",
]


def classify_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run the cascade over a prepped paired frame; appends rule_pred / rule_id /
    rule_reason. Expects friendly `<field>_l` / `<field>_r` columns (the output of
    `utils.alliance_schema.prep_paired_df`)."""
    out = df.copy()
    preds, ids, reasons = [], [], []
    cols_l = {f: f"{f}_l" for f in _FRIENDLY}
    cols_r = {f: f"{f}_r" for f in _FRIENDLY}
    for row in df.itertuples(index=False):
        rd = row._asdict()
        rec_l = {f: rd.get(cols_l[f]) for f in _FRIENDLY}
        rec_r = {f: rd.get(cols_r[f]) for f in _FRIENDLY}
        decision, rid, reason = classify_pair(rec_l, rec_r)
        preds.append(decision)
        ids.append(rid)
        reasons.append(reason)
    out["rule_pred"] = preds
    out["rule_id"] = ids
    out["rule_reason"] = reasons
    return out


def load_paired_csv(path: str) -> pd.DataFrame:
    """Read a pairs CSV (technical `_l`/`_r` columns) and prep friendly columns."""
    header = pd.read_csv(path, nrows=0).columns
    df = pd.read_csv(path, dtype=id_str_dtypes(header))
    return prep_paired_df(df)


# ==============================================================================
# CLI
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="Deterministic patient-pair classifier (match/non_match/review).")
    ap.add_argument("--input_csv", required=True, help="Pairs CSV with technical _l/_r columns.")
    ap.add_argument("--output_csv", required=True, help="Where to write input + rule_pred/rule_id/rule_reason.")
    args = ap.parse_args()

    df = load_paired_csv(args.input_csv)
    scored = classify_df(df)
    scored.to_csv(args.output_csv, index=False)

    counts = scored["rule_pred"].value_counts()
    n = len(scored)
    print(f"Scored {n} pairs -> {args.output_csv}")
    for k in (MATCH, NON_MATCH, REVIEW):
        c = int(counts.get(k, 0))
        print(f"  {k:9s}: {c:6d} ({c / n:6.1%})")


if __name__ == "__main__":
    main()
