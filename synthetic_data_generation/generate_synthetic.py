"""Synthetic FQHC patient-pair generator for AnyMatch fine-tuning.

Implements Synthetic-Dataset-Spec.md §6-§11 (v0.5 hybrid design). Deterministic from --seed.

Outputs (data/synthetic/, vN-versioned):
  synthetic_train_vN.csv   pair-level, balanced ~1:1.5, realistic bulk + hard-scenario overlay
  synthetic_test_vN.csv    pair-level, realistic prevalence, blocking-survivor-like hard negatives

Both share the identical column layout; train/test are entity-disjoint by construction.

Run from the AnyMatch/ directory:
  python synthetic_data_generation/generate_synthetic.py --seed 42 --version 2
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Cleaning-rule constants (docs/Data-Cleaning-Guide.md) — the generator must
# never emit a value the cleaning step would have flagged invalid (§4.13).
# --------------------------------------------------------------------------- #
INVALID_SUBSTRINGS = {
    "BABYBOY", "BABY BOY", "BABYGIRL", "BABY GIRL", "DUPLICATE", "DONOTUSE",
    "DO NOT USE", "DONT USE", "DON'T USE", "DO NOT", "MEDICARE",
    "DOUBLE ACCOUNT", "DUPLICATE ACCOUNT", "ACCOUNT", "<MRG>",
}
INVALID_EXACT = {
    "TEST", "BABY", "UNKNOWN", "NULL", "NAN", "NONE", "N/A", "NA", "NMI", "-",
    "ID", "",
}
NA_STRING = "N/A"  # df_serializer fills missing with this literal


def is_clean_token(value) -> bool:
    """True if `value` would survive the cleaning invalid-strings guard for a
    name/address field (so we may emit it)."""
    if value is None:
        return False
    s = str(value).strip().upper()
    if s in INVALID_EXACT:
        return False
    if any(sub in s for sub in INVALID_SUBSTRINGS):
        return False
    if re.fullmatch(r"\d+", s) or re.fullmatch(r"ID\d+", s):
        return False
    return True


# Per-field P(differs) targets for the entity-first MATCH population, calibrated
# from §5.8 (locked per-field bound in §7): name/address/phone/email from
# `by_ssn`; sex from `by_ssn_dob`; DOB from `by_ssn`.
DIFFER_RATES = {
    "last_name": 0.28,
    "first_name": 0.12,
    "middle_change": 0.16,       # given both present
    "middle_one_missing": 0.22,  # one side has middle, other doesn't
    "dob": 0.13,                 # of which off-by-one/transpose ~0.001 (handled separately)
    "address_move": 0.70,
    "city": 0.13,
    "state": 0.01,
    "phone_no_overlap": 0.65,
    "email": 0.43,               # given both present
    "sex": 0.015,
}

# Phone area codes by geographic region (NANP-valid). Used to keep phone area
# codes consistent with the drawn city (§6 step 7).
AREA_CODES_BY_REGION = {
    "CHICAGO": ["773", "312", "872", "224", "847", "708", "630", "815", "331", "464"],
    "IL_OTHER": ["847", "708", "630", "815", "224", "331"],
    "KY": ["859", "502", "606", "270"],
    "HI": ["808"],
    "WY": ["307"],
    "MD": ["410", "443", "240", "301"],
    "NY": ["646", "347", "718", "212", "607", "315", "917"],
    "IN": ["219", "765", "574", "317"],
    "CA": ["530", "916", "415", "510"],
}
DEFAULT_AREA_CODES = ["773", "312", "708", "847", "630"]

EMAIL_DOMAIN_TYPOS = {
    "gmail.com": ["gamil.com", "gmai.com", "gnail.com", "gmail.co"],
    "yahoo.com": ["yaho.com", "yahooo.com", "yahoo.co"],
    "hotmail.com": ["hotmial.com", "hotmai.com"],
}

# --------------------------------------------------------------------------- #
# Record-level full schema (blocking_eval) and model-input subset.
# --------------------------------------------------------------------------- #
RAW_COLS = [
    "FirstNM_raw", "LastNM_raw", "MiddleNM_raw", "SuffixNM_raw", "BirthDT_raw",
    "SSN_raw", "AddressLine1_raw", "AddressLine2_raw", "CityNM_raw", "ZipCD_raw",
    "StateCD_raw", "CountryNM", "PrimaryPhoneNBR_raw", "Phone01NBR_raw",
    "Phone02NBR_raw", "Phone03NBR_raw", "Email_raw", "SexAtBirthDSC_raw",
]
CLEAN_COLS = [
    "FirstNM_clean", "LastNM_clean", "MiddleNM_clean", "SuffixNM_clean",
    "BirthDT_clean", "SSN_clean", "last_4_SSN", "AddressLine1_clean",
    "AddressLine2_clean", "CityNM_clean", "ZipCD_clean_base", "ZipCD_clean_ext",
    "StateCD_clean", "PrimaryPhoneNBR_clean", "Phone01NBR_clean",
    "Phone02NBR_clean", "Phone03NBR_clean", "Email_clean", "SexAtBirthDSC_clean",
]
DERIVED_COLS = ["full_name_tokens", "full_name_compact", "Phones_set", "Address_normalized"]
RECORD_SCHEMA = ["PATID"] + RAW_COLS + CLEAN_COLS + DERIVED_COLS + ["valid_record", "entity_id"]

# Model-input columns that get _l/_r suffixes in the pair CSV (§11). Phones_set
# is the serialized 'phone' field; the three name fields ride raw (§2).
PAIR_MODEL_COLS = [
    "FirstNM_clean", "MiddleNM_clean", "LastNM_clean", "SuffixNM_clean",
    "BirthDT_clean", "SSN_clean", "last_4_SSN", "AddressLine1_clean",
    "AddressLine2_clean", "CityNM_clean", "StateCD_clean", "ZipCD_clean_base",
    "PrimaryPhoneNBR_clean", "Phone01NBR_clean", "Phone02NBR_clean",
    "Email_clean", "SexAtBirthDSC_clean",
    "full_name_tokens", "full_name_compact", "Phones_set", "Address_normalized",
]

PHONE_SLOTS = ["PrimaryPhoneNBR_clean", "Phone01NBR_clean", "Phone02NBR_clean", "Phone03NBR_clean"]


# --------------------------------------------------------------------------- #
# Pools + stats loading
# --------------------------------------------------------------------------- #
@dataclass
class Pools:
    first_head: list           # [(name, weight), ...]
    first_tail: list
    last_head: list
    last_tail: list
    streets: list
    nicknames: dict            # canonical -> [variants]; used symmetrically
    nickname_lookup: dict      # any form -> canonical-group list
    initials: dict             # letter -> [full names]

    @classmethod
    def load(cls, pools_dir: Path) -> "Pools":
        def rd(name):
            return json.loads((pools_dir / name).read_text())
        f, l = rd("first_names.json"), rd("last_names.json")
        s = rd("streets.json")
        nick = rd("nicknames.json")
        # Drop any pool name the cleaning step would have flagged invalid (e.g. the
        # surname "NA", which is on the text-null invalid list) so we never emit it.
        def _clean_weighted(items):
            return [(n, w) for n, w in items if is_clean_token(n)]
        def _clean_flat(items):
            return [n for n in items if is_clean_token(n)]
        f["weighted_head"] = _clean_weighted(f["weighted_head"]); f["tail"] = _clean_flat(f["tail"])
        l["weighted_head"] = _clean_weighted(l["weighted_head"]); l["tail"] = _clean_flat(l["tail"])
        # Build a symmetric lookup: every name form maps to the set of all its
        # equivalents (including the canonical).
        lookup: dict[str, list] = {}
        for canon, variants in nick.items():
            group = [canon] + list(variants)
            for form in group:
                lookup[form] = group
        return cls(
            first_head=[(n, w) for n, w in f["weighted_head"]],
            first_tail=list(f["tail"]),
            last_head=[(n, w) for n, w in l["weighted_head"]],
            last_tail=list(l["tail"]),
            streets=list(s["names"]),
            nicknames=nick,
            nickname_lookup=lookup,
            initials=rd("initial_expansion.json"),
        )


@dataclass
class Stats:
    raw: dict

    @classmethod
    def load(cls, path: Path) -> "Stats":
        return cls(json.loads(path.read_text()))

    # convenience accessors built lazily in __post_init__-like helpers
    def geo_joint(self):
        g = self.raw["geo_joint"]["top"]
        keys, weights = [], []
        for combo, cnt in g.items():
            city, state, zip3 = combo.split("|")
            keys.append((city, state, zip3))
            weights.append(cnt)
        return keys, weights

    def missingness_patterns(self):
        mp = self.raw["missingness_patterns"]
        order = mp["fields_order"]
        pats, weights = [], []
        for pat, cnt in mp["top_patterns"].items():
            pats.append(pat)
            weights.append(cnt)
        return order, pats, weights

    def dob_decade_weights(self):
        h = self.raw["dob"]["decade_histogram"]
        decades = [int(d) for d in h]
        return decades, [h[str(d)] for d in decades]

    def sex_value_weights(self):
        t = self.raw["categorical_top"]["SexAtBirthDSC_clean"]["top"]
        return list(t.keys()), list(t.values())

    def zip5_by_zip3(self):
        """Map zip3 -> [(zip5, weight), ...] from the measured ZIP top-N."""
        out: dict[str, list] = {}
        for z5, cnt in self.raw["categorical_top"]["ZipCD_clean_base"]["top"].items():
            out.setdefault(z5[:3], []).append((z5, cnt))
        return out

    def street_suffix_weights(self):
        toks = self.raw["address"]["addrline1_last_token_top"]
        suffixes = {"AVE", "ST", "RD", "DR", "PL", "CT", "BLVD", "LN", "WAY",
                    "CIR", "HWY", "PKWY", "TER"}
        items = [(t, c) for t, c in toks.items() if t in suffixes]
        return [t for t, _ in items], [c for _, c in items]

    def email_domain_weights(self):
        d = self.raw["email"]["top_domains"]
        # keep real consumer domains; skip the junk-ish edu/typo for the base pool
        keep = ["gmail.com", "yahoo.com", "hotmail.com", "icloud.com", "aol.com",
                "outlook.com", "live.com", "comcast.net", "att.net", "sbcglobal.net"]
        items = [(k, d[k]) for k in keep if k in d]
        return [k for k, _ in items], [c for _, c in items]


# --------------------------------------------------------------------------- #
# Region helper
# --------------------------------------------------------------------------- #
def region_for(city: str, state: str) -> str:
    if state == "IL":
        return "CHICAGO" if city == "CHICAGO" else "IL_OTHER"
    return state if state in AREA_CODES_BY_REGION else "IL_OTHER"


# --------------------------------------------------------------------------- #
# Entity + Record
# --------------------------------------------------------------------------- #
@dataclass
class Entity:
    entity_id: str
    pediatric: bool
    present: dict                 # field-group -> bool (from missingness pattern)
    canonical: dict               # canonical clean field values (the "true" person)


class Generator:
    def __init__(self, seed: int, stats: Stats, pools: Pools):
        self.rng = random.Random(seed)
        self.stats = stats
        self.pools = pools
        self.entity_counter = 0  # deterministic entity ids (no uuid4 -> reproducible)
        # precompute
        self.geo_keys, self.geo_w = stats.geo_joint()
        self.mp_order, self.mp_pats, self.mp_w = stats.missingness_patterns()
        self.dec, self.dec_w = stats.dob_decade_weights()
        self.sex_vals, self.sex_w = stats.sex_value_weights()
        self.zip5_map = stats.zip5_by_zip3()
        self.suf, self.suf_w = stats.street_suffix_weights()
        self.edom, self.edom_w = stats.email_domain_weights()

    # -- low-level samplers --------------------------------------------------
    def _choice(self, seq):
        return self.rng.choice(seq)

    def _weighted(self, seq, weights):
        return self.rng.choices(seq, weights=weights, k=1)[0]

    def sample_first(self, year=None):
        # by_year SSA enrichment is not loaded offline; flat measured draw
        # already reflects this population's age mix (§14.4 note).
        if self.rng.random() < 0.83 and self.pools.first_head:
            names = [n for n, _ in self.pools.first_head]
            w = [c for _, c in self.pools.first_head]
            return self._weighted(names, w)
        return self._choice(self.pools.first_tail or [n for n, _ in self.pools.first_head])

    def sample_last(self):
        if self.rng.random() < 0.85 and self.pools.last_head:
            names = [n for n, _ in self.pools.last_head]
            w = [c for _, c in self.pools.last_head]
            return self._weighted(names, w)
        return self._choice(self.pools.last_tail or [n for n, _ in self.pools.last_head])

    def sample_dob(self):
        decade = self._weighted(self.dec, self.dec_w)
        year = decade + self.rng.randint(0, 9)
        year = min(max(year, 1900), 2026)
        month = self.rng.randint(1, 12)
        day = self.rng.randint(1, 28)
        return date(year, month, day)

    def sample_geo(self):
        city, state, zip3 = self._weighted(self.geo_keys, self.geo_w)
        z5_candidates = self.zip5_map.get(zip3)
        if z5_candidates:
            zips = [z for z, _ in z5_candidates]
            w = [c for _, c in z5_candidates]
            zip5 = self._weighted(zips, w)
        else:
            zip5 = zip3 + f"{self.rng.randint(0, 99):02d}"
        return city, state, zip5

    def sample_street_address(self):
        # House number: Chicago-style grid (mostly 3-5 digits, capped ~14000).
        num = self.rng.randint(1, 99) if self.rng.random() < 0.05 else self.rng.randint(100, 13999)
        street = self._choice(self.pools.streets)
        suffix = self._weighted(self.suf, self.suf_w) if self.suf else "ST"
        return f"{num} {street} {suffix}"

    def sample_apt(self):
        prefix = self._weighted(
            ["APT", "UNIT", "BSMT", "FL", "1ST", "2ND", "3RD"],
            [64, 5, 3, 3, 2, 2, 2],
        )
        if prefix in ("BSMT",):
            return prefix
        unit = self.rng.choice([str(self.rng.randint(1, 30)),
                                f"{self.rng.randint(1, 9)}{self.rng.choice('ABCD')}"])
        return f"{prefix} {unit}"

    def gen_phone(self, region: str):
        ac = self._choice(AREA_CODES_BY_REGION.get(region, DEFAULT_AREA_CODES))
        # NXX: first digit 2-9, not N11, not 555x
        while True:
            nxx = f"{self.rng.randint(2,9)}{self.rng.randint(0,9)}{self.rng.randint(0,9)}"
            if nxx[1:] == "11" or nxx in ("555",):
                continue
            break
        line = f"{self.rng.randint(0,9999):04d}"
        if line in ("0100",):  # avoid 555-style fiction handled by NXX; keep simple
            line = "0123"
        return ac + nxx + line

    def gen_ssn(self):
        while True:
            area = self.rng.randint(1, 899)
            if area in (0, 666) or 900 <= area <= 999:
                continue
            break
        group = self.rng.randint(1, 99)
        serial = self.rng.randint(1, 9999)
        return f"{area:03d}{group:02d}{serial:04d}"

    def gen_last4(self):
        v = self.rng.randint(1, 9999)
        return f"{v:04d}"

    def sample_email(self, first, last):
        domain = self._weighted(self.edom, self.edom_w) if self.edom else "gmail.com"
        style = self.rng.random()
        f, l = first.lower(), last.lower().replace(" ", "")
        if style < 0.4:
            local = f"{f}.{l}"
        elif style < 0.7:
            local = f"{f}{l}{self.rng.randint(1, 99)}"
        elif style < 0.9:
            local = f"{f[0]}{l}"
        else:
            local = f"{f}{self.rng.randint(1, 9999)}"
        return f"{local}@{domain}"

    # -- entity construction -------------------------------------------------
    def make_entity(self, force_pediatric=None, force_full_ssn=None) -> Entity:
        dob = self.sample_dob()
        pediatric = (dob.year >= 2010) if force_pediatric is None else force_pediatric
        if force_pediatric:
            dob = self.sample_dob()
            while dob.year < 2010:
                dob = self.sample_dob()

        pattern = self._weighted(self.mp_pats, self.mp_w)
        bits = {self.mp_order[i]: pattern[i] == "1" for i in range(len(self.mp_order))}
        # Pediatric coupling (§6): SSN/email skew absent.
        if pediatric:
            if self.rng.random() < 0.85:
                bits["ssn_full"] = False
            if self.rng.random() < 0.6:
                bits["ssn_last4"] = False
            if self.rng.random() < 0.7:
                bits["email"] = False
        if force_full_ssn:
            bits["ssn_full"] = True
            bits["ssn_last4"] = True

        # names
        first = self.sample_first(dob.year)
        last = self.sample_last()
        if self.rng.random() < 0.07:  # p_two_surname
            last = f"{last} {self.sample_last()}"
        elif self.rng.random() < 0.018:  # p_hyphen_last
            last = f"{last}-{self.sample_last()}"
        if self.rng.random() < 0.02:  # p_compound_first
            first = f"{first} {self.sample_first(dob.year)}"

        middle = None
        if bits.get("middle"):
            if self.rng.random() < 0.97:
                middle = self.rng.choice("ABCDEFGHIJKLMNOPRSTV")
            else:
                middle = self.sample_first(dob.year)

        sex = self._weighted(self.sex_vals, self.sex_w) if bits.get("sex") else None

        # ssn
        ssn = last4 = None
        if bits.get("ssn_full"):
            ssn = self.gen_ssn()
            last4 = ssn[-4:]
        elif bits.get("ssn_last4"):
            last4 = self.gen_last4()

        # geography + address
        city = state = zip5 = line1 = line2 = None
        if bits.get("address"):
            city, state, zip5 = self.sample_geo()
            line1 = self.sample_street_address()
            if self.rng.random() < 0.294:
                line2 = self.sample_apt()
            elif self.rng.random() < 0.021:
                line1 = f"PO BOX {self.rng.randint(1, 99999)}"
        region = region_for(city or "CHICAGO", state or "IL")

        # phones
        phones = []
        if bits.get("phone"):
            n = self._weighted([1, 2, 3, 4], [0.124, 0.556, 0.252, 0.012])
            phones = [self.gen_phone(region) for _ in range(n)]

        email = self.sample_email(first.split()[0], last.split()[0]) if bits.get("email") else None

        canonical = {
            "FirstNM_clean": first, "MiddleNM_clean": middle, "LastNM_clean": last,
            "SuffixNM_clean": None,
            "BirthDT_clean": dob.isoformat(),
            "SSN_clean": ssn, "last_4_SSN": last4,
            "AddressLine1_clean": line1, "AddressLine2_clean": line2,
            "CityNM_clean": city, "StateCD_clean": state,
            "ZipCD_clean_base": zip5, "ZipCD_clean_ext": None,
            "SexAtBirthDSC_clean": sex,
            "Email_clean": email,
            "_phones": phones, "_region": region,
        }
        eid = f"E{self.entity_counter:09d}"
        self.entity_counter += 1
        return Entity(eid, pediatric, bits, canonical)


# --------------------------------------------------------------------------- #
# Record materialization + derivations
# --------------------------------------------------------------------------- #
def split_name_tokens(*fields):
    toks = []
    for f in fields:
        if f:
            toks.extend(re.split(r"[\s\-]+", str(f)))
    return sorted({t for t in toks if t})


def name_compact(first, middle, last):
    parts = [p for p in (first, middle, last) if p]
    return re.sub(r"[^A-Za-z]", "", "".join(parts)).upper() or None


def phones_set(phones):
    clean = sorted({p for p in phones if p})
    return " ".join(clean) if clean else None


def record_from_canonical(canonical: dict, entity_id: str, patid: str) -> dict:
    """Materialize a full record-schema dict from a canonical/variant value dict.
    Phones live in `_phones`; everything else is a clean column."""
    phones = canonical.get("_phones", [])
    rec = {c: None for c in RECORD_SCHEMA}
    rec["PATID"] = patid
    rec["entity_id"] = entity_id
    rec["valid_record"] = True
    rec["CountryNM"] = "US"
    for c in CLEAN_COLS:
        if c in canonical:
            rec[c] = canonical[c]
    # phone slots
    for i, slot in enumerate(PHONE_SLOTS):
        rec[slot] = phones[i] if i < len(phones) else None
    # derived
    rec["full_name_tokens"] = " ".join(split_name_tokens(
        rec["FirstNM_clean"], rec["MiddleNM_clean"], rec["LastNM_clean"])) or None
    rec["full_name_compact"] = name_compact(
        rec["FirstNM_clean"], rec["MiddleNM_clean"], rec["LastNM_clean"])
    rec["Phones_set"] = phones_set(phones)
    rec["Address_normalized"] = None  # libpostal not run -> 100% null in real file
    # raw mirrors clean (synthetic has no upstream noise; §11.1)
    rec["FirstNM_raw"] = rec["FirstNM_clean"]
    rec["LastNM_raw"] = rec["LastNM_clean"]
    rec["MiddleNM_raw"] = rec["MiddleNM_clean"]
    rec["SuffixNM_raw"] = rec["SuffixNM_clean"]
    rec["BirthDT_raw"] = rec["BirthDT_clean"]
    rec["SSN_raw"] = rec["SSN_clean"]
    rec["AddressLine1_raw"] = rec["AddressLine1_clean"]
    rec["AddressLine2_raw"] = rec["AddressLine2_clean"]
    rec["CityNM_raw"] = rec["CityNM_clean"]
    rec["ZipCD_raw"] = rec["ZipCD_clean_base"]
    rec["StateCD_raw"] = rec["StateCD_clean"]
    for slot in PHONE_SLOTS:
        rec[slot.replace("_clean", "_raw")] = rec[slot]
    rec["Email_raw"] = rec["Email_clean"]
    rec["SexAtBirthDSC_raw"] = rec["SexAtBirthDSC_clean"]
    return rec


# --------------------------------------------------------------------------- #
# Corruptions (§7) — each takes (gen, canonical_copy) and mutates in place,
# returning a short corruption name (or None if it could not apply).
# --------------------------------------------------------------------------- #
def _typo(rng: random.Random, s: str) -> str:
    if not s or len(s) < 2:
        return s
    i = rng.randrange(len(s))
    op = rng.choice(["sub", "del", "ins", "trans"])
    if op == "sub":
        out = s[:i] + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + s[i + 1:]
    elif op == "del":
        out = s[:i] + s[i + 1:]
    elif op == "ins":
        out = s[:i] + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + s[i:]
    elif op == "trans" and i < len(s) - 1:
        out = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    else:
        out = s
    # A typo must not produce a token the cleaning step would flag invalid
    # (e.g. deleting "ANA" -> "NA"); fall back to the original in that case.
    return out if is_clean_token(out) else s


class Corruptions:
    def __init__(self, gen: Generator):
        self.gen = gen
        self.rng = gen.rng

    def typo_last(self, c):
        if not c.get("LastNM_clean"):
            return None
        c["LastNM_clean"] = _typo(self.rng, c["LastNM_clean"])
        return "typo_last"

    def typo_first(self, c):
        if not c.get("FirstNM_clean"):
            return None
        c["FirstNM_clean"] = _typo(self.rng, c["FirstNM_clean"])
        return "typo_first"

    def replace_last(self, c):
        c["LastNM_clean"] = self.gen.sample_last()
        return "replace_last"  # maiden<->married

    def drop_middle(self, c):
        if not c.get("MiddleNM_clean"):
            return None
        c["MiddleNM_clean"] = None
        return "drop_middle"

    def middle_to_initial(self, c):
        m = c.get("MiddleNM_clean")
        if not m or len(m) == 1:
            return None
        c["MiddleNM_clean"] = m[0]
        return "middle_to_initial"

    def expand_initial(self, c):
        m = c.get("MiddleNM_clean")
        if not m or len(m) != 1:
            return None
        opts = self.gen.pools.initials.get(m)
        if not opts:
            return None
        c["MiddleNM_clean"] = self.rng.choice(opts)
        return "expand_initial"

    def swap_first_middle(self, c):
        if not c.get("MiddleNM_clean"):
            return None
        c["FirstNM_clean"], c["MiddleNM_clean"] = c["MiddleNM_clean"], c["FirstNM_clean"]
        return "swap_first_middle"

    def nickname(self, c):
        f = c.get("FirstNM_clean")
        grp = self.gen.pools.nickname_lookup.get(f) if f else None
        if not grp:
            return None
        alt = [x for x in grp if x != f]
        if not alt:
            return None
        c["FirstNM_clean"] = self.rng.choice(alt)
        return "nickname"

    def drop_one_surname(self, c):
        last = c.get("LastNM_clean")
        if not last or " " not in last:
            return None
        parts = last.split()
        c["LastNM_clean"] = self.rng.choice(parts)
        return "drop_one_surname"

    def dob_drift(self, c):
        d = c.get("BirthDT_clean")
        if not d:
            return None
        y, m, day = map(int, d.split("-"))
        kind = self.rng.random()
        try:
            if kind < 0.4:
                nd = date(y, m, day) + timedelta(days=self.rng.choice([-1, 1]))
            elif kind < 0.7:
                nd = date(y + self.rng.choice([-1, 1]), m, day)
            elif kind < 0.85 and m <= 12 and day <= 12:
                nd = date(y, day, m)
            else:
                nd = date(y + self.rng.choice([-3, -2, 2, 3, 5]), m, min(day, 28))
        except ValueError:
            return None
        c["BirthDT_clean"] = nd.isoformat()
        return "dob_drift"

    def address_move(self, c):
        if not c.get("AddressLine1_clean"):
            return None
        # ~13% also change city, ~1% state; otherwise same city/state, new street/zip
        r = self.rng.random()
        if r < 0.01:
            city, state, zip5 = self.gen.sample_geo()
            c["CityNM_clean"], c["StateCD_clean"], c["ZipCD_clean_base"] = city, state, zip5
        elif r < 0.14 and c.get("StateCD_clean") == "IL":
            # different IL city, new zip
            city, state, zip5 = self.gen.sample_geo()
            if state == "IL":
                c["CityNM_clean"], c["ZipCD_clean_base"] = city, zip5
        c["AddressLine1_clean"] = self.gen.sample_street_address()
        if self.rng.random() < 0.5:
            c["ZipCD_clean_base"] = self._new_zip(c.get("CityNM_clean"), c.get("ZipCD_clean_base"))
        c["AddressLine2_clean"] = self.gen.sample_apt() if self.rng.random() < 0.294 else None
        return "address_move"

    def _new_zip(self, city, current):
        # pick another zip in the same zip3 if possible
        if current:
            z3 = current[:3]
            cands = [z for z, _ in self.gen.zip5_map.get(z3, []) if z != current]
            if cands:
                return self.rng.choice(cands)
        return current

    def change_apt(self, c):
        if not c.get("AddressLine1_clean"):
            return None
        c["AddressLine2_clean"] = self.gen.sample_apt()
        return "change_apt"

    def phone_replace(self, c):
        phones = list(c.get("_phones", []))
        region = c.get("_region", "CHICAGO")
        if not phones:
            c["_phones"] = [self.gen.gen_phone(region)]
            return "phone_add"
        # replace all -> no overlap
        n = max(1, len(phones))
        c["_phones"] = [self.gen.gen_phone(region) for _ in range(n)]
        return "phone_replace"

    def email_change(self, c):
        if not c.get("Email_clean"):
            return None
        f = (c.get("FirstNM_clean") or "x").split()[0]
        l = (c.get("LastNM_clean") or "x").split()[0]
        c["Email_clean"] = self.gen.sample_email(f, l)
        return "email_change"

    def email_domain_typo(self, c):
        e = c.get("Email_clean")
        if not e or "@" not in e:
            return None
        local, dom = e.rsplit("@", 1)
        opts = EMAIL_DOMAIN_TYPOS.get(dom)
        if not opts:
            return None
        c["Email_clean"] = f"{local}@{self.rng.choice(opts)}"
        return "email_domain_typo"

    def sex_flip(self, c):
        s = c.get("SexAtBirthDSC_clean")
        if s not in ("MALE", "FEMALE"):
            return None
        c["SexAtBirthDSC_clean"] = "FEMALE" if s == "MALE" else "MALE"
        return "sex_flip"

    # -- structural name transforms (§8.4 additions) ----------------------- #
    def first_to_initial(self, c):
        f = c.get("FirstNM_clean")
        if not f or len(f.split()[0]) < 2:
            return None
        c["FirstNM_clean"] = f.split()[0][0]
        return "first_to_initial"

    def truncate_name(self, c):
        f = c.get("LastNM_clean")
        if not f or len(f) < 7:
            return None
        c["LastNM_clean"] = f[: max(5, len(f) - self.rng.randint(1, 3))]
        return "truncate_name"

    def concat_spaces(self, c):
        for key in ("LastNM_clean", "FirstNM_clean"):
            v = c.get(key)
            if v and " " in v:
                c[key] = v.replace(" ", "")
                return "concat_spaces"
        return None

    def cross_lang_variant(self, c):
        # reuse the nickname/equivalence pool (seeded with GUILLERMO<->WILLIAM etc.)
        return self.nickname(c) and "cross_lang_variant"

    def name_order_swap(self, c):
        f, l = c.get("FirstNM_clean"), c.get("LastNM_clean")
        if not f or not l:
            return None
        c["FirstNM_clean"], c["LastNM_clean"] = l, f
        return "name_order_swap"

    def move_within_zip(self, c):
        if not c.get("AddressLine1_clean"):
            return None
        c["AddressLine1_clean"] = self.gen.sample_street_address()
        return "move_within_zip"

    def directional_expand(self, c):
        a = c.get("AddressLine1_clean")
        if not a:
            return None
        repl = {" N ": " NORTH ", " S ": " SOUTH ", " E ": " EAST ", " W ": " WEST ",
                " ST": " STREET", " AVE": " AVENUE", " RD": " ROAD", " DR": " DRIVE"}
        out = a
        for k, v in repl.items():
            out = out.replace(k, v)
        if out == a:
            return None
        c["AddressLine1_clean"] = out
        return "directional_expand"


# --------------------------------------------------------------------------- #
# Variant generation (entity-first) + scenario construction (case-first)
# --------------------------------------------------------------------------- #
def clone(canonical: dict) -> dict:
    c = dict(canonical)
    c["_phones"] = list(canonical.get("_phones", []))
    return c


def apply_calibrated_corruptions(gen: Generator, base: dict, messiness: float = 1.0) -> tuple[dict, list]:
    """Produce one corrupted variant of `base`, applying each field's corruption
    independently at the §7 calibrated marginal (the locked application model).

    `messiness > 1.0` is the §7 dirty-tail multiplier: it scales every field's
    P(differs) up *together* (capped near 1.0), so corruptions correlate and many
    fields drift at once — the "everything is a mess" tail the marginals alone miss.
    """
    corr = Corruptions(gen)
    c = clone(base)
    applied = []
    rng = gen.rng

    def p(field):  # calibrated marginal, amplified by the messiness multiplier
        return min(DIFFER_RATES[field] * messiness, 0.97)

    # Name: last-name change dominates; pick change type when it fires.
    if rng.random() < p("last_name"):
        roll = rng.random()
        fn = (corr.replace_last if roll < 0.45 else
              corr.drop_one_surname if roll < 0.6 else corr.typo_last)
        applied.append(fn(c) or corr.typo_last(c))
    if rng.random() < p("first_name"):
        fn = corr.nickname if rng.random() < 0.4 else corr.typo_first
        applied.append(fn(c) or corr.typo_first(c))
    # Middle
    if c.get("MiddleNM_clean") and rng.random() < p("middle_change"):
        fn = rng.choice([corr.middle_to_initial, corr.expand_initial])
        applied.append(fn(c))
    if rng.random() < p("middle_one_missing"):
        applied.append(corr.drop_middle(c))
    # DOB (rare; off-by-one inside dob_drift is itself rare)
    if rng.random() < p("dob"):
        applied.append(corr.dob_drift(c))
    # Address
    if rng.random() < p("address_move"):
        applied.append(corr.address_move(c))
    elif rng.random() < 0.2:
        applied.append(corr.change_apt(c))
    # Phone
    if rng.random() < p("phone_no_overlap"):
        applied.append(corr.phone_replace(c))
    # Email
    if c.get("Email_clean") and rng.random() < p("email"):
        fn = corr.email_domain_typo if rng.random() < 0.15 else corr.email_change
        applied.append(fn(c) or corr.email_change(c))
    # Sex (very rare)
    if rng.random() < p("sex"):
        applied.append(corr.sex_flip(c))

    return c, [a for a in applied if a]


# Cluster-size histograms by identifier band (§5.7) -> K distribution.
K_HISTOGRAMS = {
    "full_ssn":  ([1, 2, 3, 4, 5], [0.874, 0.111, 0.013, 0.0015, 0.0005]),
    "last4":     ([1, 2, 3, 4, 5], [0.816, 0.143, 0.031, 0.0085, 0.0015]),
    "no_ssn":    ([1, 2, 3, 4, 5], [0.838, 0.130, 0.025, 0.005, 0.002]),
}


def band_of(entity: Entity) -> str:
    if entity.present.get("ssn_full"):
        return "full_ssn"
    if entity.present.get("ssn_last4"):
        return "last4"
    return "no_ssn"


# --------------------------------------------------------------------------- #
# Pair assembly
# --------------------------------------------------------------------------- #
class PairBuilder:
    def __init__(self, gen: Generator):
        self.gen = gen
        self.patid_counter = 0

    def new_patid(self):
        self.patid_counter += 1
        return f"S{self.patid_counter:09d}"

    def emit_record(self, canonical, entity_id):
        return record_from_canonical(canonical, entity_id, self.new_patid())

    # ---- entity-first ----
    def entity_first_records(self, n_entities: int):
        """Produce (records, entity_to_records) drawing realistic K."""
        records = []
        ent_recs: dict[str, list] = {}
        entities: dict[str, Entity] = {}
        for _ in range(n_entities):
            e = self.gen.make_entity()
            entities[e.entity_id] = e
            ks, kw = K_HISTOGRAMS[band_of(e)]
            k = self.gen._weighted(ks, kw)
            recs = []
            base_rec = self.emit_record(e.canonical, e.entity_id)
            recs.append(base_rec)
            for _ in range(k - 1):
                variant, _appl = apply_calibrated_corruptions(self.gen, e.canonical)
                recs.append(self.emit_record(variant, e.entity_id))
            records.extend(recs)
            ent_recs[e.entity_id] = recs
        return records, ent_recs, entities


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def pair_row(rec_a: dict, rec_b: dict, label: int, case_type: str, corruptions: list) -> dict:
    row = {"PATID_A": rec_a["PATID"], "PATID_B": rec_b["PATID"]}
    for col in PAIR_MODEL_COLS:
        row[f"{col}_l"] = rec_a.get(col)
        row[f"{col}_r"] = rec_b.get(col)
    row["label"] = label
    row["case_type"] = case_type
    row["corruptions_applied"] = json.dumps(corruptions)
    row["entity_id_a"] = rec_a["entity_id"]
    row["entity_id_b"] = rec_b["entity_id"]
    return row


# --------------------------------------------------------------------------- #
# Case-first scenario library (§8). Each scenario method returns:
#   (canonA, eidA, canonB, eidB, label, case_type, corruptions)
# For match scenarios eidA == eidB (constructed from one entity); for non-match
# the two entities are distinct people.
# --------------------------------------------------------------------------- #
def _copy(src, dst, fields):
    for f in fields:
        dst[f] = src[f]


ADDR_FIELDS = ["AddressLine1_clean", "AddressLine2_clean", "CityNM_clean",
               "StateCD_clean", "ZipCD_clean_base"]


class ScenarioLib:
    def __init__(self, gen: Generator):
        self.gen = gen
        self.rng = gen.rng
        self.corr = Corruptions(gen)

    # -- entity helpers ------------------------------------------------------
    def _ent(self, **force):
        return self.gen.make_entity(**force)

    def ensure_full_ssn(self, e):
        if not e.canonical["SSN_clean"]:
            ssn = self.gen.gen_ssn()
            e.canonical["SSN_clean"] = ssn
            e.canonical["last_4_SSN"] = ssn[-4:]

    def ensure_no_ssn(self, e):
        e.canonical["SSN_clean"] = None
        e.canonical["last_4_SSN"] = None

    def ensure_last4_only(self, e):
        if not e.canonical["last_4_SSN"]:
            e.canonical["last_4_SSN"] = self.gen.gen_last4()
        e.canonical["SSN_clean"] = None

    def ensure_address(self, e):
        if not e.canonical["AddressLine1_clean"]:
            city, state, zip5 = self.gen.sample_geo()
            e.canonical["CityNM_clean"] = city
            e.canonical["StateCD_clean"] = state
            e.canonical["ZipCD_clean_base"] = zip5
            e.canonical["AddressLine1_clean"] = self.gen.sample_street_address()
            e.canonical["_region"] = region_for(city, state)

    def ensure_phone(self, e):
        if not e.canonical.get("_phones"):
            e.canonical["_phones"] = [self.gen.gen_phone(e.canonical.get("_region", "CHICAGO"))]

    def ensure_email(self, e):
        if not e.canonical["Email_clean"]:
            f = e.canonical["FirstNM_clean"].split()[0]
            l = e.canonical["LastNM_clean"].split()[0]
            e.canonical["Email_clean"] = self.gen.sample_email(f, l)

    def ensure_middle(self, e):
        if not e.canonical["MiddleNM_clean"]:
            e.canonical["MiddleNM_clean"] = self.rng.choice("ABCDEFGHIJKLMNRSTV")

    def ensure_sex(self, e):
        if not e.canonical["SexAtBirthDSC_clean"]:
            e.canonical["SexAtBirthDSC_clean"] = self.rng.choice(["MALE", "FEMALE"])

    def diff_first(self, avoid):
        f = self.gen.sample_first()
        while f == avoid:
            f = self.gen.sample_first()
        return f

    # ======================= MATCH scenarios (label=1) ==================== #
    # ---- No-SSN-led ----
    def m_nossn_control(self):
        e = self._ent(); self.ensure_no_ssn(e); self.ensure_address(e); self.ensure_sex(e)
        return clone(e.canonical), e.entity_id, clone(e.canonical), e.entity_id, 1, "M-NOSSN-01", []

    def m_nossn_addr_moved(self):
        e = self._ent(); self.ensure_no_ssn(e); self.ensure_address(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.address_move(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-NOSSN-02", ["address_move"]

    def m_nossn_phone_overlap(self):
        e = self._ent(); self.ensure_no_ssn(e); self.ensure_address(e); self.ensure_phone(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.address_move(B)  # phones kept -> overlap
        return A, e.entity_id, B, e.entity_id, 1, "M-NOSSN-03", ["address_move"]

    def m_nossn_thin(self):
        e = self._ent(); self.ensure_no_ssn(e); self.ensure_sex(e)
        for f in ADDR_FIELDS:
            e.canonical[f] = None
        e.canonical["_phones"] = []; e.canonical["Email_clean"] = None
        return clone(e.canonical), e.entity_id, clone(e.canonical), e.entity_id, 1, "M-NOSSN-04", []

    def m_nossn_name_typo(self):
        e = self._ent(); self.ensure_no_ssn(e)
        A, B = clone(e.canonical), clone(e.canonical)
        c = self.corr.typo_first(B) if self.rng.random() < 0.5 else self.corr.typo_last(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-NOSSN-05", [c]

    # ---- SSN-led ----
    def m_ssn_identical(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e); self.ensure_address(e)
        return clone(e.canonical), e.entity_id, clone(e.canonical), e.entity_id, 1, "M-SSN-01", []

    def m_ssn_name_typo(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.typo_last(B); self.corr.typo_first(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-SSN-02", ["typo_last", "typo_first"]

    def m_ssn_missing_middle(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e); self.ensure_middle(e)
        A, B = clone(e.canonical), clone(e.canonical)
        B["MiddleNM_clean"] = None
        return A, e.entity_id, B, e.entity_id, 1, "M-SSN-03", ["drop_middle"]

    def m_ssn_maiden_married(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        A, B = clone(e.canonical), clone(e.canonical)
        B["LastNM_clean"] = self.gen.sample_last()
        return A, e.entity_id, B, e.entity_id, 1, "M-SSN-04", ["replace_last"]

    def m_ssn_moved(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e); self.ensure_address(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.address_move(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-SSN-05", ["address_move"]

    def m_ssn_moved_oos(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e); self.ensure_address(e)
        A, B = clone(e.canonical), clone(e.canonical)
        city, state, zip5 = self.gen.sample_geo()
        B["CityNM_clean"], B["StateCD_clean"], B["ZipCD_clean_base"] = city, state, zip5
        B["AddressLine1_clean"] = self.gen.sample_street_address()
        return A, e.entity_id, B, e.entity_id, 1, "M-SSN-06", ["address_move_oos"]

    def m_ssn_diff_contact(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e); self.ensure_phone(e); self.ensure_email(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.phone_replace(B); self.corr.email_change(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-SSN-07", ["phone_replace", "email_change"]

    def m_ssn_dob_drift(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.dob_drift(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-SSN-08", ["dob_drift"]

    def m_ssn_full_vs_last4(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        A, B = clone(e.canonical), clone(e.canonical)
        B["SSN_clean"] = None  # B keeps only the matching last-4
        return A, e.entity_id, B, e.entity_id, 1, "M-SSN-09", ["drop_full_ssn"]

    def m_ssn_vs_nossn(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e); self.ensure_address(e)
        A, B = clone(e.canonical), clone(e.canonical)
        B["SSN_clean"] = None; B["last_4_SSN"] = None
        return A, e.entity_id, B, e.entity_id, 1, "M-SSN-10", ["drop_ssn"]

    def m_ssn_heavy_drift(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e); self.ensure_address(e); self.ensure_middle(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.typo_last(B); B["MiddleNM_clean"] = None
        self.corr.dob_drift(B); self.corr.address_move(B); self.corr.phone_replace(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-SSN-11", ["typo_last", "drop_middle", "dob_drift", "address_move", "phone_replace"]

    # ---- Last-4-led ----
    def m_l4_control(self):
        e = self._ent(); self.ensure_last4_only(e); self.ensure_address(e)
        return clone(e.canonical), e.entity_id, clone(e.canonical), e.entity_id, 1, "M-L4-01", []

    def m_l4_name_typo(self):
        e = self._ent(); self.ensure_last4_only(e)
        A, B = clone(e.canonical), clone(e.canonical)
        c = self.corr.typo_last(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-L4-02", [c]

    def m_l4_dob_drift(self):
        e = self._ent(); self.ensure_last4_only(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.dob_drift(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-L4-03", ["dob_drift"]

    def m_l4_asym(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        A, B = clone(e.canonical), clone(e.canonical)
        B["SSN_clean"] = None  # B only last-4 (== A's last-4)
        return A, e.entity_id, B, e.entity_id, 1, "M-L4-04", ["drop_full_ssn"]

    def m_l4_asym_namedrift(self):
        a, eid, b, _, lab, _, corr = self.m_l4_asym()
        self.corr.typo_last(b)
        return a, eid, b, eid, 1, "M-L4-05", corr + ["typo_last"]

    # ---- Name-coupling-led ----
    def m_name_hyphen(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        e.canonical["FirstNM_clean"] = "ANNE-MARIE"
        A, B = clone(e.canonical), clone(e.canonical)
        B["FirstNM_clean"] = self.rng.choice(["ANNE MARIE", "ANNEMARIE"])
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-01", ["hyphen_variant"]

    def m_name_first_middle_swap(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e); self.ensure_middle(e)
        e.canonical["MiddleNM_clean"] = self.gen.sample_first()
        A, B = clone(e.canonical), clone(e.canonical)
        B["FirstNM_clean"], B["MiddleNM_clean"] = B["MiddleNM_clean"], B["FirstNM_clean"]
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-02", ["swap_first_middle"]

    def m_name_two_surname_shuffle(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        s1, s2 = self.gen.sample_last(), self.gen.sample_last()
        A = clone(e.canonical); A["MiddleNM_clean"] = s1; A["LastNM_clean"] = s2
        B = clone(e.canonical); B["MiddleNM_clean"] = None; B["LastNM_clean"] = f"{s1} {s2}"
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-03", ["two_surname_shuffle"]

    def m_name_two_surname_collapse(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        s1, s2 = self.gen.sample_last(), self.gen.sample_last()
        A = clone(e.canonical); A["LastNM_clean"] = f"{s1} {s2}"
        B = clone(e.canonical); B["LastNM_clean"] = self.rng.choice([s1, s2])
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-04", ["drop_one_surname"]

    def m_name_vietnamese_swap(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        given, fam = "NGUYEN", "THI MAI"
        A = clone(e.canonical); A["FirstNM_clean"] = given; A["LastNM_clean"] = fam; A["MiddleNM_clean"] = None
        B = clone(e.canonical); B["FirstNM_clean"] = fam; B["LastNM_clean"] = given; B["MiddleNM_clean"] = None
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-05", ["name_order_swap"]

    def m_name_middle_initial(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        full = self.gen.sample_first()
        A = clone(e.canonical); A["MiddleNM_clean"] = full
        B = clone(e.canonical); B["MiddleNM_clean"] = full[0]
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-06", ["middle_to_initial"]

    def m_name_compound_first_dropped(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        f1, f2 = self.gen.sample_first(), self.gen.sample_first()
        A = clone(e.canonical); A["FirstNM_clean"] = f"{f1} {f2}"
        B = clone(e.canonical); B["FirstNM_clean"] = f1
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-07", ["compound_first_dropped"]

    def m_name_suffix(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        A = clone(e.canonical); A["LastNM_clean"] = f'{A["LastNM_clean"]} JR'
        B = clone(e.canonical); B["SuffixNM_clean"] = "JR"
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-08", ["suffix_variant"]

    def m_name_suffix_wrong_slot(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        A = clone(e.canonical); A["MiddleNM_clean"] = self.rng.choice(["JR", "SR", "III"])
        B = clone(e.canonical); B["SuffixNM_clean"] = A["MiddleNM_clean"]; B["MiddleNM_clean"] = None
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-08b", ["suffix_wrong_slot"]

    def m_name_nickname(self):
        canon = self.rng.choice(list(self.gen.pools.nicknames.keys()))
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        e.canonical["FirstNM_clean"] = canon
        A, B = clone(e.canonical), clone(e.canonical)
        B["FirstNM_clean"] = self.rng.choice(self.gen.pools.nicknames[canon])
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-09", ["nickname"]

    def _m_name_typo(self, case):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        A, B = clone(e.canonical), clone(e.canonical)
        c = self.corr.typo_last(B) or self.corr.typo_first(B)
        return A, e.entity_id, B, e.entity_id, 1, case, [c]

    def m_name_typo_sub(self):
        return self._m_name_typo("M-NAME-10")

    def m_name_typo_trans(self):
        return self._m_name_typo("M-NAME-11")

    def m_name_typo_insdel(self):
        return self._m_name_typo("M-NAME-12")

    # ---- DOB / address / phone / email / sex / pediatric drift ----
    def m_addr_apt_toggle(self):
        e = self._ent(); self.ensure_address(e)
        e.canonical["AddressLine2_clean"] = self.gen.sample_apt()
        A, B = clone(e.canonical), clone(e.canonical)
        B["AddressLine2_clean"] = None
        return A, e.entity_id, B, e.entity_id, 1, "M-ADDR-01", ["drop_apt"]

    def m_addr_apt_change(self):
        e = self._ent(); self.ensure_address(e)
        e.canonical["AddressLine2_clean"] = self.gen.sample_apt()
        A, B = clone(e.canonical), clone(e.canonical)
        B["AddressLine2_clean"] = self.gen.sample_apt()
        return A, e.entity_id, B, e.entity_id, 1, "M-ADDR-02", ["change_apt"]

    def m_addr_line2_absorb(self):
        e = self._ent(); self.ensure_address(e)
        apt = self.gen.sample_apt()
        A = clone(e.canonical); A["AddressLine2_clean"] = apt
        B = clone(e.canonical); B["AddressLine1_clean"] = f'{e.canonical["AddressLine1_clean"]} {apt}'; B["AddressLine2_clean"] = None
        return A, e.entity_id, B, e.entity_id, 1, "M-ADDR-03", ["line2_absorb"]

    def m_addr_house_typo(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e); self.ensure_address(e)
        A, B = clone(e.canonical), clone(e.canonical)
        B["AddressLine1_clean"] = _typo(self.rng, B["AddressLine1_clean"])
        return A, e.entity_id, B, e.entity_id, 1, "M-ADDR-04", ["house_typo"]

    def m_dob_transpose(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        y, m, d = map(int, e.canonical["BirthDT_clean"].split("-"))
        if m <= 12 and d <= 12:
            A, B = clone(e.canonical), clone(e.canonical)
            try:
                B["BirthDT_clean"] = date(y, d, m).isoformat()
            except ValueError:
                pass
            return A, e.entity_id, B, e.entity_id, 1, "M-DOB-02", ["dob_transpose"]
        return self.m_dob_off_year()

    def m_dob_off_year(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        y, m, d = map(int, e.canonical["BirthDT_clean"].split("-"))
        A, B = clone(e.canonical), clone(e.canonical)
        B["BirthDT_clean"] = date(y + self.rng.choice([-1, 1]), m, d).isoformat()
        return A, e.entity_id, B, e.entity_id, 1, "M-DOB-03", ["dob_off_year"]

    def m_dob_off_day(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        A, B = clone(e.canonical), clone(e.canonical)
        nd = date(*map(int, e.canonical["BirthDT_clean"].split("-"))) + timedelta(days=self.rng.choice([-1, 1]))
        B["BirthDT_clean"] = nd.isoformat()
        return A, e.entity_id, B, e.entity_id, 1, "M-DOB-04", ["dob_off_day"]

    def m_dob_null(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        A, B = clone(e.canonical), clone(e.canonical)
        B["BirthDT_clean"] = None
        return A, e.entity_id, B, e.entity_id, 1, "M-DOB-05", ["dob_null"]

    def m_phone_partial(self):
        e = self._ent(); self.ensure_address(e)
        e.canonical["_phones"] = [self.gen.gen_phone(e.canonical["_region"]) for _ in range(2)]
        A = clone(e.canonical)
        B = clone(e.canonical)
        B["_phones"] = [e.canonical["_phones"][0], self.gen.gen_phone(e.canonical["_region"])]
        return A, e.entity_id, B, e.entity_id, 1, "M-PHONE-01", ["phone_partial"]

    def m_phone_disjoint(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e); self.ensure_phone(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.phone_replace(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-PHONE-02", ["phone_replace"]

    def m_email_change(self):
        e = self._ent(); self.ensure_email(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.email_change(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-EMAIL-01", ["email_change"]

    def m_email_domain_typo(self):
        e = self._ent(); self.ensure_email(e)
        e.canonical["Email_clean"] = e.canonical["Email_clean"].split("@")[0] + "@gmail.com"
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.email_domain_typo(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-EMAIL-02", ["email_domain_typo"]

    def m_sex_other(self):
        e = self._ent(force_full_ssn=True); self.ensure_full_ssn(e)
        A = clone(e.canonical); A["SexAtBirthDSC_clean"] = "OTHER"
        B = clone(e.canonical); B["SexAtBirthDSC_clean"] = self.rng.choice(["MALE", "FEMALE"])
        return A, e.entity_id, B, e.entity_id, 1, "M-SEX-02", ["sex_other"]

    def m_ped_thin(self):
        e = self._ent(force_pediatric=True); self.ensure_no_ssn(e); self.ensure_address(e); self.ensure_sex(e)
        e.canonical["Email_clean"] = None
        return clone(e.canonical), e.entity_id, clone(e.canonical), e.entity_id, 1, "M-PED-01", []

    def m_ped_last4(self):
        e = self._ent(force_pediatric=True); self.ensure_last4_only(e); self.ensure_address(e)
        return clone(e.canonical), e.entity_id, clone(e.canonical), e.entity_id, 1, "M-PED-02", []

    def m_ped_name_drift(self):
        e = self._ent(force_pediatric=True); self.ensure_no_ssn(e); self.ensure_address(e)
        A, B = clone(e.canonical), clone(e.canonical)
        c = self.corr.drop_middle(B) or self.corr.typo_first(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-PED-03", [c]

    # ---- Mixed ----
    def m_mix_two(self):
        e = self._ent(); self.ensure_last4_only(e); self.ensure_address(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.typo_last(B); self.corr.address_move(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-MIX-01", ["typo_last", "address_move"]

    def m_mix_three(self):
        e = self._ent(); self.ensure_last4_only(e); self.ensure_address(e); self.ensure_phone(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.typo_last(B); self.corr.address_move(B); self.corr.phone_replace(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-MIX-02", ["typo_last", "address_move", "phone_replace"]

    def m_mix_thin(self):
        return self.m_nossn_thin()[:5] + ("M-MIX-03", [])

    # ---- §8.4 additional structural hard MATCH cases ---- #
    def m_name_first_initial(self):
        e = self._ent()
        A, B = clone(e.canonical), clone(e.canonical)
        c = self.corr.first_to_initial(B) or self.corr.typo_first(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-13", [c]

    def m_name_truncate(self):
        e = self._ent()
        e.canonical["LastNM_clean"] = self.rng.choice(
            ["MASSIMILIANO", "HERNANDEZHERNANDEZ", "GUTIERREZRAMIREZ", "WASHINGTON"])
        A, B = clone(e.canonical), clone(e.canonical)
        c = self.corr.truncate_name(B) or self.corr.typo_last(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-14", [c]

    def m_name_crosslang(self):
        canon = self.rng.choice(list(self.gen.pools.nicknames.keys()))
        e = self._ent()
        e.canonical["FirstNM_clean"] = canon
        A, B = clone(e.canonical), clone(e.canonical)
        B["FirstNM_clean"] = self.rng.choice(self.gen.pools.nicknames[canon])
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-15", ["cross_lang_variant"]

    def m_name_concat(self):
        e = self._ent()
        e.canonical["LastNM_clean"] = self.rng.choice(
            ["DE LA CRUZ", "DE LEON", "SAN MIGUEL", "MARY ANN"])
        A, B = clone(e.canonical), clone(e.canonical)
        c = self.corr.concat_spaces(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-NAME-16", [c or "concat_spaces"]

    def m_addr_within_zip(self):
        e = self._ent(); self.ensure_address(e)
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.move_within_zip(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-ADDR-05", ["move_within_zip"]

    def m_addr_directional(self):
        e = self._ent(); self.ensure_address(e)
        e.canonical["AddressLine1_clean"] = f"{self.rng.randint(100,9999)} N MAIN ST"
        A, B = clone(e.canonical), clone(e.canonical)
        self.corr.directional_expand(B)
        return A, e.entity_id, B, e.entity_id, 1, "M-ADDR-06", ["directional_expand"]

    def m_zip_drift(self):
        e = self._ent(); self.ensure_address(e)
        A, B = clone(e.canonical), clone(e.canonical)
        B["ZipCD_clean_base"] = self.corr._new_zip(B.get("CityNM_clean"), B.get("ZipCD_clean_base"))
        return A, e.entity_id, B, e.entity_id, 1, "M-ZIP-01", ["zip_drift"]

    # ======================= NON-MATCH scenarios (label=0) ================ #
    def _two(self, **force):
        return self._ent(**force), self._ent(**force)

    def nm_random(self):
        a, b = self._two()
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-EASY-01", []

    def nm_same_state(self):
        a, b = self._two(); self.ensure_address(a); self.ensure_address(b)
        b.canonical["StateCD_clean"] = a.canonical["StateCD_clean"]
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-EASY-02", []

    def _household(self, case, same_last=True, same_dob=False, dob_gap=None, share_phone=False, diff_last=False):
        a, b = self._two(); self.ensure_address(a)
        _copy(a.canonical, b.canonical, ADDR_FIELDS)
        b.canonical["_region"] = a.canonical["_region"]
        if same_last:
            b.canonical["LastNM_clean"] = a.canonical["LastNM_clean"]
        if diff_last:
            b.canonical["LastNM_clean"] = self.diff_last(a.canonical["LastNM_clean"])
        b.canonical["FirstNM_clean"] = self.diff_first(a.canonical["FirstNM_clean"])
        if same_dob:
            b.canonical["BirthDT_clean"] = a.canonical["BirthDT_clean"]
        elif dob_gap is not None:
            y, m, d = map(int, a.canonical["BirthDT_clean"].split("-"))
            ny = min(max(y + self.rng.randint(*dob_gap), 1900), 2026)
            b.canonical["BirthDT_clean"] = date(ny, m, d).isoformat()
        # different identities -> distinct SSN
        b.canonical["SSN_clean"] = None; b.canonical["last_4_SSN"] = None
        if share_phone and a.canonical.get("_phones"):
            b.canonical["_phones"] = [a.canonical["_phones"][0]]
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, case, []

    def diff_last(self, avoid):
        l = self.gen.sample_last()
        while l == avoid:
            l = self.gen.sample_last()
        return l

    def nm_twin(self):
        return self._household("NM-HH-TWIN", same_last=True, same_dob=True)

    def nm_triplet_like(self):
        return self._household("NM-HH-TRIPLET-LIKE", same_last=False, diff_last=True, same_dob=True)

    def nm_jr_sr(self):
        a, eid_a, b, eid_b, lab, _, _ = self._household("NM-HH-JR-SR", same_last=True, dob_gap=(20, 40))
        b["FirstNM_clean"] = a["FirstNM_clean"]  # same given name (Jr/Sr)
        b["SuffixNM_clean"] = "JR"
        return a, eid_a, b, eid_b, 0, "NM-HH-JR-SR", []

    def nm_sibling(self):
        return self._household("NM-HH-SIBLING", same_last=True, dob_gap=(-10, 10))

    def nm_parent_child(self):
        return self._household("NM-HH-PARENT-CHILD", same_last=True, dob_gap=(15, 40))

    def nm_spouse(self):
        return self._household("NM-HH-SPOUSE", same_last=False, diff_last=True, dob_gap=(-5, 5), share_phone=True)

    def nm_roommate(self):
        return self._household("NM-HH-ROOMMATE", same_last=False, diff_last=True, dob_gap=(-20, 20))

    def _common_name(self, case, same_zip=False, two_surname=False):
        a, b = self._two()
        if two_surname:
            shared = f"{self.gen.sample_last()} {self.gen.sample_last()}"
        else:
            shared = self.gen.sample_last()
        shared_first = self.gen.sample_first()
        for e in (a, b):
            e.canonical["FirstNM_clean"] = shared_first
            e.canonical["LastNM_clean"] = shared
            self.ensure_address(e)
            e.canonical["CityNM_clean"] = "CHICAGO"; e.canonical["StateCD_clean"] = "IL"
        # different DOB (>=5y) and different SSN
        y = int(a.canonical["BirthDT_clean"][:4])
        b.canonical["BirthDT_clean"] = a.canonical["BirthDT_clean"].replace(
            str(y), str(min(max(y + self.rng.choice([-9, -7, 7, 9]), 1900), 2026)), 1)
        b.canonical["SSN_clean"] = None; b.canonical["last_4_SSN"] = None
        if same_zip:
            b.canonical["ZipCD_clean_base"] = a.canonical["ZipCD_clean_base"]
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, case, []

    def nm_common_name_city(self):
        return self._common_name("NM-COMMON-01")

    def nm_common_name_zip(self):
        return self._common_name("NM-COMMON-02", same_zip=True)

    def nm_hispanic_surname(self):
        return self._common_name("NM-COMMON-03", two_surname=True)

    def nm_top_zip_collision(self):
        a, eid_a, b, eid_b, lab, _, _ = self._common_name("NM-COMMON-04", same_zip=True)
        a["ZipCD_clean_base"] = b["ZipCD_clean_base"] = self.rng.choice(["60639", "60625", "60640", "60651", "60647"])
        return a, eid_a, b, eid_b, 0, "NM-COMMON-04", []

    def nm_areacode(self):
        a, b = self._two()
        shared_first, shared_last = self.gen.sample_first(), self.gen.sample_last()
        ac = self.rng.choice(["773", "312", "708"])
        for e in (a, b):
            e.canonical["FirstNM_clean"] = shared_first; e.canonical["LastNM_clean"] = shared_last
            e.canonical["_phones"] = [ac + self.gen.gen_phone("CHICAGO")[3:]]
        b.canonical["SSN_clean"] = None; b.canonical["last_4_SSN"] = None
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-COMMON-05", []

    def nm_last4_collision(self):
        a, b = self._two()
        self.ensure_last4_only(a);
        b.canonical["SSN_clean"] = None; b.canonical["last_4_SSN"] = a.canonical["last_4_SSN"]
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-SSN-01", []

    def nm_last4_first_letter(self):
        a, eid_a, b, eid_b, lab, _, _ = self.nm_last4_collision()
        b["FirstNM_clean"] = a["FirstNM_clean"][0] + b["FirstNM_clean"][1:] if len(b["FirstNM_clean"]) > 1 else a["FirstNM_clean"]
        return a, eid_a, b, eid_b, 0, "NM-SSN-02", []

    def nm_ssn_typo_collision(self):
        a, b = self._two(force_full_ssn=True)
        self.ensure_full_ssn(a); self.ensure_full_ssn(b)
        b.canonical["SSN_clean"] = a.canonical["SSN_clean"]  # collision via typo
        b.canonical["last_4_SSN"] = a.canonical["SSN_clean"][-4:]
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-SSN-03", []

    def nm_ssn_opposite_sex(self):
        a, eid_a, b, eid_b, *_ = self.nm_ssn_typo_collision()
        a["SexAtBirthDSC_clean"], b["SexAtBirthDSC_clean"] = "MALE", "FEMALE"
        return a, eid_a, b, eid_b, 0, "NM-SSN-04", []

    def nm_full_vs_mismatch_last4(self):
        a, b = self._two(force_full_ssn=True)
        self.ensure_full_ssn(a)
        b.canonical["SSN_clean"] = None
        # last-4 that deliberately does NOT match A's full SSN tail
        l4 = self.gen.gen_last4()
        while l4 == a.canonical["SSN_clean"][-4:]:
            l4 = self.gen.gen_last4()
        b.canonical["last_4_SSN"] = l4
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-SSN-05", []

    def nm_shelter_addr(self):
        a, b = self._two(); self.ensure_address(a)
        _copy(a.canonical, b.canonical, ADDR_FIELDS)
        b.canonical["SSN_clean"] = None; b.canonical["last_4_SSN"] = None
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-IDF-01", []

    def nm_family_phone(self):
        a, b = self._two(); self.ensure_phone(a)
        b.canonical["_phones"] = [a.canonical["_phones"][0]]
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-IDF-02", []

    def nm_family_email(self):
        a, b = self._two(); self.ensure_email(a)
        b.canonical["Email_clean"] = a.canonical["Email_clean"]
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-IDF-03", []

    def nm_addr_and_phone(self):
        a, b = self._two(); self.ensure_address(a); self.ensure_phone(a)
        _copy(a.canonical, b.canonical, ADDR_FIELDS)
        b.canonical["_phones"] = [a.canonical["_phones"][0]]
        b.canonical["SSN_clean"] = None; b.canonical["last_4_SSN"] = None
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-IDF-04", []

    def nm_ped_siblings(self):
        a, b = self._two(force_pediatric=True); self.ensure_address(a); self.ensure_phone(a)
        _copy(a.canonical, b.canonical, ADDR_FIELDS)
        b.canonical["LastNM_clean"] = a.canonical["LastNM_clean"]
        b.canonical["FirstNM_clean"] = self.diff_first(a.canonical["FirstNM_clean"])
        b.canonical["_phones"] = [a.canonical["_phones"][0]]
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-PED-01", []

    def nm_ped_same_dob(self):
        a, b = self._two(force_pediatric=True); self.ensure_address(a)
        _copy(a.canonical, b.canonical, ADDR_FIELDS)
        b.canonical["BirthDT_clean"] = a.canonical["BirthDT_clean"]
        b.canonical["LastNM_clean"] = self.diff_last(a.canonical["LastNM_clean"])
        b.canonical["FirstNM_clean"] = self.diff_first(a.canonical["FirstNM_clean"])
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-PED-02", []

    def nm_thin_diff_name(self):
        a, b = self._two()
        for e in (a, b):
            for f in ADDR_FIELDS:
                e.canonical[f] = None
            e.canonical["_phones"] = []; e.canonical["Email_clean"] = None
            e.canonical["SSN_clean"] = None; e.canonical["last_4_SSN"] = None
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-BND-01", []

    def nm_thin_diff_dob(self):
        a, eid_a, b, eid_b, *_ = self.nm_thin_diff_name()
        b["FirstNM_clean"] = a["FirstNM_clean"]; b["LastNM_clean"] = a["LastNM_clean"]
        y = int(a["BirthDT_clean"][:4])
        b["BirthDT_clean"] = a["BirthDT_clean"].replace(str(y), str(max(y - 8, 1900)), 1)
        return a, eid_a, b, eid_b, 0, "NM-BND-02", []

    # ---- §8.4 additional hard NON-MATCH cases ---- #
    def nm_common_adjacent_dob(self):
        a, b = self._two()
        f, l = self.gen.sample_first(), self.gen.sample_last()
        for e in (a, b):
            e.canonical["FirstNM_clean"] = f; e.canonical["LastNM_clean"] = l
        y, m, d = map(int, a.canonical["BirthDT_clean"].split("-"))
        try:
            b.canonical["BirthDT_clean"] = date(y, m, d).replace(
                day=min(max(d + self.rng.choice([-2, -1, 1, 2]), 1), 28)).isoformat()
        except ValueError:
            pass
        b.canonical["SSN_clean"] = b.canonical["last_4_SSN"] = None
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-COMMON-06", []

    def nm_cousin(self):
        a, b = self._two(); self.ensure_address(a); self.ensure_address(b)
        shared = self.gen.sample_last()
        a.canonical["LastNM_clean"] = b.canonical["LastNM_clean"] = shared
        b.canonical["CityNM_clean"] = a.canonical["CityNM_clean"]
        b.canonical["ZipCD_clean_base"] = a.canonical["ZipCD_clean_base"]
        b.canonical["SSN_clean"] = b.canonical["last_4_SSN"] = None
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-HH-COUSIN", []

    def nm_last4_dob(self):
        a, b = self._two(); self.ensure_last4_only(a)
        b.canonical["SSN_clean"] = None
        b.canonical["last_4_SSN"] = a.canonical["last_4_SSN"]
        b.canonical["BirthDT_clean"] = a.canonical["BirthDT_clean"]
        return clone(a.canonical), a.entity_id, clone(b.canonical), b.entity_id, 0, "NM-SSN-06", []


# Bucket -> (weight-of-side, [scenario method names]). Flag-gated scenarios
# (§8.3) excluded by default.
MATCH_BUCKETS = {
    "NOSSN": (0.30, ["m_nossn_control", "m_nossn_addr_moved", "m_nossn_phone_overlap",
                     "m_nossn_thin", "m_nossn_name_typo"]),
    "NAME":  (0.20, ["m_name_hyphen", "m_name_first_middle_swap", "m_name_two_surname_shuffle",
                     "m_name_two_surname_collapse", "m_name_vietnamese_swap", "m_name_middle_initial",
                     "m_name_compound_first_dropped", "m_name_suffix", "m_name_suffix_wrong_slot",
                     "m_name_nickname", "m_name_typo_sub", "m_name_typo_trans", "m_name_typo_insdel"]),
    "SSN":   (0.15, ["m_ssn_identical", "m_ssn_name_typo", "m_ssn_missing_middle", "m_ssn_maiden_married",
                     "m_ssn_moved", "m_ssn_moved_oos", "m_ssn_diff_contact", "m_ssn_dob_drift",
                     "m_ssn_full_vs_last4", "m_ssn_vs_nossn", "m_ssn_heavy_drift"]),
    "L4":    (0.08, ["m_l4_control", "m_l4_name_typo", "m_l4_dob_drift", "m_l4_asym", "m_l4_asym_namedrift"]),
    "DRIFT": (0.20, ["m_addr_apt_toggle", "m_addr_apt_change", "m_addr_line2_absorb", "m_addr_house_typo",
                     "m_dob_transpose", "m_dob_off_year", "m_dob_off_day", "m_dob_null",
                     "m_phone_partial", "m_phone_disjoint", "m_email_change", "m_email_domain_typo",
                     "m_sex_other", "m_ped_thin", "m_ped_last4", "m_ped_name_drift"]),
    "MIX":   (0.07, ["m_mix_two", "m_mix_three", "m_mix_thin"]),
}
NM_BUCKETS = {
    "EASY":   (0.15, ["nm_random", "nm_same_state"]),
    "HH":     (0.30, ["nm_twin", "nm_triplet_like", "nm_jr_sr", "nm_sibling",
                      "nm_parent_child", "nm_spouse", "nm_roommate"]),
    "COMMON": (0.22, ["nm_common_name_city", "nm_common_name_zip", "nm_hispanic_surname",
                      "nm_top_zip_collision", "nm_areacode"]),
    "SSN":    (0.13, ["nm_last4_collision", "nm_last4_first_letter", "nm_ssn_typo_collision",
                      "nm_ssn_opposite_sex", "nm_full_vs_mismatch_last4"]),
    "IDF":    (0.12, ["nm_shelter_addr", "nm_family_phone", "nm_family_email", "nm_addr_and_phone"]),
    "PED":    (0.05, ["nm_ped_siblings", "nm_ped_same_dob"]),
    "BND":    (0.03, ["nm_thin_diff_name", "nm_thin_diff_dob"]),
}


def allocate(buckets: dict, total: int) -> list:
    """Round-robin scenario calls to hit per-bucket budgets."""
    plan = []
    for _bucket, (weight, methods) in buckets.items():
        n = round(weight * total)
        per = max(1, n // len(methods))
        for m in methods:
            plan.append((m, per))
    return plan


# --------------------------------------------------------------------------- #
# Assembly, split, writers
# --------------------------------------------------------------------------- #
def test_entities(entity_ids, seed, frac=0.15):
    out = set()
    for eid in entity_ids:
        if random.Random(f"{seed}:{eid}").random() < frac:
            out.add(eid)
    return out


# --------------------------------------------------------------------------- #
# Hybrid assembly (v0.5): realistic entity-first BULK + budgeted hard-scenario
# OVERLAY. Train and test draw their own freshly-sampled entities, so the two
# files are entity-disjoint by construction (every make_entity mints a new id).
# --------------------------------------------------------------------------- #
def recompute_derived(rec: dict) -> None:
    """Recompute the derived name/phone columns after a key-share mutation so the
    serialized name fields and their token/compact/phone derivations stay consistent."""
    rec["full_name_tokens"] = " ".join(split_name_tokens(
        rec["FirstNM_clean"], rec["MiddleNM_clean"], rec["LastNM_clean"])) or None
    rec["full_name_compact"] = name_compact(
        rec["FirstNM_clean"], rec["MiddleNM_clean"], rec["LastNM_clean"])
    phones = rec["Phones_set"].split() if rec.get("Phones_set") else []
    rec["Phones_set"] = phones_set(phones)


# --------------------------------------------------------------------------- #
# Difficulty design (v0.5, §8.4): SSN bands + no-identical + hard (key-sharing) NM.
# Coverage methods guarantee EVERY named scenario appears many times; the bulk
# supplies realistic volume. enforce_positive() applies the SSN band + address
# rule to *every* positive (coverage or bulk) so the bands hold globally.
# --------------------------------------------------------------------------- #
SSN_COVER = ["m_ssn_name_typo", "m_ssn_missing_middle", "m_ssn_maiden_married",
             "m_ssn_moved", "m_ssn_moved_oos", "m_ssn_diff_contact",
             "m_ssn_dob_drift", "m_ssn_heavy_drift"]
L4_COVER = ["m_l4_name_typo", "m_l4_dob_drift", "m_l4_asym", "m_l4_asym_namedrift",
            "m_ssn_full_vs_last4"]
HARD_COVER = [
    "m_nossn_addr_moved", "m_nossn_phone_overlap", "m_nossn_name_typo",
    "m_name_hyphen", "m_name_first_middle_swap", "m_name_two_surname_shuffle",
    "m_name_two_surname_collapse", "m_name_vietnamese_swap", "m_name_middle_initial",
    "m_name_compound_first_dropped", "m_name_suffix", "m_name_suffix_wrong_slot",
    "m_name_nickname", "m_name_typo_sub", "m_name_typo_trans", "m_name_typo_insdel",
    "m_name_first_initial", "m_name_truncate", "m_name_crosslang", "m_name_concat",
    "m_addr_apt_toggle", "m_addr_apt_change", "m_addr_line2_absorb", "m_addr_house_typo",
    "m_addr_within_zip", "m_addr_directional", "m_zip_drift",
    "m_dob_transpose", "m_dob_off_year", "m_dob_off_day", "m_dob_null",
    "m_phone_partial", "m_phone_disjoint", "m_email_change", "m_email_domain_typo",
    "m_sex_other", "m_ped_name_drift", "m_mix_two", "m_mix_three"]

NM_EASY_COVER = ["nm_random", "nm_same_state"]
NM_HARD_COVER = [
    "nm_twin", "nm_triplet_like", "nm_jr_sr", "nm_sibling", "nm_parent_child",
    "nm_spouse", "nm_roommate", "nm_common_name_city", "nm_common_name_zip",
    "nm_hispanic_surname", "nm_top_zip_collision", "nm_areacode",
    "nm_common_adjacent_dob", "nm_cousin", "nm_last4_collision", "nm_last4_first_letter",
    "nm_ssn_typo_collision", "nm_ssn_opposite_sex", "nm_full_vs_mismatch_last4",
    "nm_last4_dob", "nm_shelter_addr", "nm_family_phone", "nm_family_email",
    "nm_addr_and_phone", "nm_ped_siblings", "nm_ped_same_dob",
    "nm_thin_diff_name", "nm_thin_diff_dob"]

_NAME_DOB_ADDR = ["FirstNM_clean", "LastNM_clean", "BirthDT_clean", "AddressLine1_clean"]


def _identical(recA, recB):
    return all((recA.get(c) or "") == (recB.get(c) or "") for c in _NAME_DOB_ADDR)


def enforce_positive(gen: Generator, recA: dict, recB: dict, band: str) -> None:
    """Apply the §8.4 SSN band + address rule to a positive pair, and guarantee
    >=1 difference (no identical pairs)."""
    if band == "easy":
        ssn = recA.get("SSN_clean") or recB.get("SSN_clean") or gen.gen_ssn()
        for r in (recA, recB):
            r["SSN_clean"], r["last_4_SSN"] = ssn, ssn[-4:]
    elif band == "last4":
        l4 = recA.get("last_4_SSN") or recB.get("last_4_SSN") or gen.gen_last4()
        for r in (recA, recB):
            r["SSN_clean"], r["last_4_SSN"] = None, l4
    else:  # hard: no usable SSN match on >=1 side
        for r in (recA, recB):
            t = gen.rng.random()
            if t < 0.75:
                r["SSN_clean"] = r["last_4_SSN"] = None        # neither
            elif t < 0.90:
                r["SSN_clean"] = None                          # last-4 only
        if recA.get("SSN_clean") and recA["SSN_clean"] == recB.get("SSN_clean"):
            recB["SSN_clean"] = None
        if recA.get("last_4_SSN") and recA["last_4_SSN"] == recB.get("last_4_SSN"):
            recB["last_4_SSN"] = None
        # high-mobility population: address rarely matches between true pairs
        if (recA.get("AddressLine1_clean") and recA["AddressLine1_clean"] == recB.get("AddressLine1_clean")
                and gen.rng.random() < 0.85):
            recB["AddressLine1_clean"] = gen.sample_street_address()
    # never emit an identical pair: guarantee >=1 difference on a name/dob field
    if _identical(recA, recB):
        corr = Corruptions(gen)
        for fn in (corr.typo_last, corr.typo_first, corr.nickname):
            fn(recB)
            if not _identical(recA, recB):
                break
        else:
            recB["LastNM_clean"] = gen.sample_last()  # last-resort guaranteed change (maiden-style)
    recompute_derived(recB)


def _band_plan(n, easy=0.05, last4=0.15):
    n_easy = round(n * easy)
    n_l4 = round(n * last4)
    return n_easy, n_l4, n - n_easy - n_l4


def _run_methods(gen, pb, lib, methods, count, enforce_band=None):
    """Round-robin `methods` to produce `count` pairs; if enforce_band is set the
    pairs are positives and get the SSN-band rule applied."""
    rows = []
    i = 0
    while len(rows) < count:
        fn = getattr(lib, methods[i % len(methods)])
        i += 1
        cA, eidA, cB, eidB, label, case, corr = fn()
        recA, recB = pb.emit_record(cA, eidA), pb.emit_record(cB, eidB)
        if enforce_band is not None:
            enforce_positive(gen, recA, recB, enforce_band)
        rows.append(pair_row(recA, recB, label, case, corr))
    return rows


def make_positive(gen: Generator, pb: "PairBuilder", band: str, dirty_frac: float) -> dict:
    """Bulk within-entity match pair for a given SSN band, corruption scaled by band."""
    e = gen.make_entity()
    recA = pb.emit_record(e.canonical, e.entity_id)
    if band == "easy":
        messiness, case = gen.rng.uniform(1.0, 1.6), "M-BULK-EASY"
    elif band == "last4":
        messiness, case = gen.rng.uniform(1.3, 2.2), "M-BULK-L4"
    else:
        messiness = gen.rng.uniform(2.0, 3.5) if gen.rng.random() < dirty_frac else gen.rng.uniform(1.5, 2.6)
        case = "M-BULK-HARD"
    variant, applied = apply_calibrated_corruptions(gen, e.canonical, messiness=messiness)
    recB = pb.emit_record(variant, e.entity_id)
    enforce_positive(gen, recA, recB, band)
    return pair_row(recA, recB, 1, case, applied)


def force_shared_keys(gen: Generator, recA: dict, recB: dict, n_keys: int) -> str:
    """Copy `n_keys` distinct strong fields from A onto B so a cross-entity NON-match
    looks like a blocking survivor. Returns a case tag naming the shared keys."""
    # Only structural blocking keys count (a shared common first name is not a real
    # blocking key); first name may ride along as an extra when n_keys is high.
    avail = []
    if recA["LastNM_clean"]:        avail.append("lastname")
    if recA["BirthDT_clean"]:       avail.append("dob")
    if recA["AddressLine1_clean"]:  avail.append("addr")
    if recA["last_4_SSN"]:          avail.append("last4")
    if recA["Phones_set"]:          avail.append("phone")
    if not avail:
        avail = ["lastname"]
    keys = gen.rng.sample(avail, min(n_keys, len(avail)))
    if n_keys >= 3 and recA["FirstNM_clean"] and "lastname" in keys:
        keys = keys + ["firstname"]   # common full-name collision
    for key in keys:
        if key == "lastname":
            recB["LastNM_clean"] = recA["LastNM_clean"]
        elif key == "firstname":
            recB["FirstNM_clean"] = recA["FirstNM_clean"]
        elif key == "dob":
            recB["BirthDT_clean"] = recA["BirthDT_clean"]
        elif key == "addr":
            for f in ADDR_FIELDS:
                recB[f] = recA[f]
        elif key == "last4":
            recB["last_4_SSN"] = recA["last_4_SSN"]; recB["SSN_clean"] = None
        elif key == "phone":
            existing = recB["Phones_set"].split() if recB["Phones_set"] else []
            recB["Phones_set"] = phones_set([recA["Phones_set"].split()[0]] + existing)
    recompute_derived(recB)
    return "NM-HARD-" + "+".join(sorted(keys)).upper()


def make_hard_negative(gen: Generator, pb: "PairBuilder") -> dict:
    """Two distinct, realistically-corrupted people forced to share 1-3 strong keys."""
    eA, eB = gen.make_entity(), gen.make_entity()
    A, _ = apply_calibrated_corruptions(gen, eA.canonical)
    B, _ = apply_calibrated_corruptions(gen, eB.canonical)
    recA, recB = pb.emit_record(A, eA.entity_id), pb.emit_record(B, eB.entity_id)
    n_keys = gen._weighted([1, 2, 3], [0.58, 0.30, 0.12])   # ~42% share >=2 keys
    case = force_shared_keys(gen, recA, recB, n_keys)
    return pair_row(recA, recB, 0, case, [])


def _assemble(gen, pb, n_match, n_non, cover_frac, dirty_frac, easy_neg_frac):
    """Shared assembly for train and test. Positives follow the §8.4 SSN bands
    (5/15/80); negatives are `easy_neg_frac` easy + the rest hard (1-3 shared keys).
    `cover_frac` of each band/group is filled from the enumerated catalog so EVERY
    named scenario appears many times; the rest is realistic bulk. Train and test use
    this same construction (so the test measures the same hard cases); they differ in
    match prevalence, the disjoint entity pool, and easy-negative fraction (the test
    is all-hard so precision is measured honestly)."""
    lib = ScenarioLib(gen)
    rows = []
    # ---- positives: SSN bands ----
    n_easy, n_l4, n_hard = _band_plan(n_match)
    for band, n, cover in (("easy", n_easy, SSN_COVER),
                           ("last4", n_l4, L4_COVER),
                           ("hard", n_hard, HARD_COVER)):
        n_cov = round(n * cover_frac)
        rows += _run_methods(gen, pb, lib, cover, n_cov, enforce_band=band)
        for _ in range(n - n_cov):
            rows.append(make_positive(gen, pb, band, dirty_frac))
    # ---- negatives: easy (train anchor only) / hard ----
    n_neasy = round(n_non * easy_neg_frac)
    n_nhard = n_non - n_neasy
    if n_neasy:
        rows += _run_methods(gen, pb, lib, NM_EASY_COVER, n_neasy)
    n_nh_cov = round(n_nhard * cover_frac)
    rows += _run_methods(gen, pb, lib, NM_HARD_COVER, n_nh_cov)
    for _ in range(n_nhard - n_nh_cov):
        rows.append(make_hard_negative(gen, pb))
    gen.rng.shuffle(rows)
    return rows


def build_train(gen, pb, n_pairs, ratio=1.5, cover_frac=0.35, dirty_frac=0.25):
    """Balanced ~1:ratio training corpus (§8.4 difficulty); keeps a 3% easy-negative anchor."""
    n_match = round(n_pairs * (1.0 / (1.0 + ratio)))
    return _assemble(gen, pb, n_match, n_pairs - n_match, cover_frac, dirty_frac, easy_neg_frac=0.03)


def build_test(gen, pb, n_pairs, prevalence=0.10, cover_frac=0.35, dirty_frac=0.25):
    """Honest evaluator: realistic prevalence, *identical difficulty construction to
    train* (all named hard cases + SSN bands), on held-out entities, with **all-hard
    negatives** (no random strangers) so precision reflects production."""
    n_match = round(n_pairs * prevalence)
    return _assemble(gen, pb, n_match, n_pairs - n_match, cover_frac, dirty_frac, easy_neg_frac=0.0)


def write_pairs(rows, path):
    df = pd.DataFrame(rows)
    # model + provenance column order
    front = ["PATID_A", "PATID_B"]
    model = [f"{c}_{s}" for c in PAIR_MODEL_COLS for s in ("l", "r")]
    tail = ["label", "case_type", "corruptions_applied", "entity_id_a", "entity_id_b"]
    df = df[front + model + tail]
    df.to_csv(path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--version", type=int, default=2)
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--pools", default=str(HERE / "pools"))
    ap.add_argument("--stats", default=str(HERE / "synthetic_data_stats.json"))
    ap.add_argument("--train-pairs", type=int, default=40000)
    ap.add_argument("--train-ratio", type=float, default=1.5,
                    help="non-match:match ratio in the training set (1:ratio)")
    ap.add_argument("--test-pairs", type=int, default=10000)
    ap.add_argument("--test-prevalence", type=float, default=0.20,
                    help="positive fraction in the test set (1:4 -> enough positives for stable per-case recall)")
    ap.add_argument("--overlay-frac", type=float, default=0.20,
                    help="share of each split drawn from the hard-scenario overlay (rest = realistic bulk)")
    ap.add_argument("--dirty-frac", type=float, default=0.20,
                    help="share of positives drawn with the dirty-tail messiness multiplier")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.train_pairs, args.test_pairs = 4000, 1000

    stats = Stats.load(Path(args.stats))
    pools = Pools.load(Path(args.pools))
    gen = Generator(args.seed, stats, pools)
    pb = PairBuilder(gen)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    v = args.version

    # Train and test draw their own fresh entities -> entity-disjoint by construction.
    train_rows = build_train(gen, pb, args.train_pairs, ratio=args.train_ratio,
                             dirty_frac=args.dirty_frac)
    test_rows = build_test(gen, pb, args.test_pairs, prevalence=args.test_prevalence,
                           dirty_frac=args.dirty_frac)
    write_pairs(train_rows, out / f"synthetic_train_v{v}.csv")
    write_pairs(test_rows, out / f"synthetic_test_v{v}.csv")

    def summ(rows):
        m = sum(r["label"] for r in rows)
        return f"{len(rows)} pairs ({m} match / {len(rows) - m} non-match)"
    print(f"synthetic_train_v{v}: {summ(train_rows)}")
    print(f"synthetic_test_v{v}:  {summ(test_rows)}")
    print(f"wrote 2 files to {out}/ at v{v}")


if __name__ == "__main__":
    main()
