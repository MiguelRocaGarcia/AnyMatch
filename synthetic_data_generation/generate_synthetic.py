"""Synthetic FQHC patient-pair generator for AnyMatch fine-tuning.

Implements Synthetic-Dataset-Spec.md §6-§11. Deterministic from --seed.

Outputs (data/synthetic/, vN-versioned):
  finetune_train_vN.csv / finetune_test_vN.csv   pair-level, balanced corpus (case-first)
  realistic_eval_vN.csv                          pair-level, realistic prevalence (entity-first)
  blocking_eval_vN.csv                           record-level, full cleaning schema + entity_id

Run from the AnyMatch/ directory:
  python synthetic_data_generation/generate_synthetic.py --seed 42 --version 1
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
        return s[:i] + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + s[i + 1:]
    if op == "del":
        return s[:i] + s[i + 1:]
    if op == "ins":
        return s[:i] + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + s[i:]
    if op == "trans" and i < len(s) - 1:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s


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


# --------------------------------------------------------------------------- #
# Variant generation (entity-first) + scenario construction (case-first)
# --------------------------------------------------------------------------- #
def clone(canonical: dict) -> dict:
    c = dict(canonical)
    c["_phones"] = list(canonical.get("_phones", []))
    return c


def apply_calibrated_corruptions(gen: Generator, base: dict) -> tuple[dict, list]:
    """Produce one corrupted variant of `base`, applying each field's corruption
    independently at the §7 calibrated marginal (the locked application model)."""
    corr = Corruptions(gen)
    c = clone(base)
    applied = []
    rng = gen.rng

    # Name: last-name change dominates; pick change type when it fires.
    if rng.random() < DIFFER_RATES["last_name"]:
        roll = rng.random()
        fn = (corr.replace_last if roll < 0.45 else
              corr.drop_one_surname if roll < 0.6 else corr.typo_last)
        applied.append(fn(c) or corr.typo_last(c))
    if rng.random() < DIFFER_RATES["first_name"]:
        fn = corr.nickname if rng.random() < 0.4 else corr.typo_first
        applied.append(fn(c) or corr.typo_first(c))
    # Middle
    if c.get("MiddleNM_clean") and rng.random() < DIFFER_RATES["middle_change"]:
        fn = rng.choice([corr.middle_to_initial, corr.expand_initial])
        applied.append(fn(c))
    if rng.random() < DIFFER_RATES["middle_one_missing"]:
        applied.append(corr.drop_middle(c))
    # DOB (rare; off-by-one inside dob_drift is itself rare)
    if rng.random() < DIFFER_RATES["dob"]:
        applied.append(corr.dob_drift(c))
    # Address
    if rng.random() < DIFFER_RATES["address_move"]:
        applied.append(corr.address_move(c))
    elif rng.random() < 0.2:
        applied.append(corr.change_apt(c))
    # Phone
    if rng.random() < DIFFER_RATES["phone_no_overlap"]:
        applied.append(corr.phone_replace(c))
    # Email
    if c.get("Email_clean") and rng.random() < DIFFER_RATES["email"]:
        fn = corr.email_domain_typo if rng.random() < 0.15 else corr.email_change
        applied.append(fn(c) or corr.email_change(c))
    # Sex (very rare)
    if rng.random() < DIFFER_RATES["sex"]:
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


def build_finetune(gen, pb, n_match, n_nonmatch):
    lib = ScenarioLib(gen)
    rows = []
    for buckets, total in ((MATCH_BUCKETS, n_match), (NM_BUCKETS, n_nonmatch)):
        for method, count in allocate(buckets, total):
            fn = getattr(lib, method)
            for _ in range(count):
                cA, eidA, cB, eidB, label, case, corr = fn()
                recA = pb.emit_record(cA, eidA)
                recB = pb.emit_record(cB, eidB)
                rows.append(pair_row(recA, recB, label, case, corr))
    return rows


def coarse_nm_case(recA, recB):
    if recA["AddressLine1_clean"] and recA["AddressLine1_clean"] == recB["AddressLine1_clean"]:
        return "NM-HH/IDF"
    if recA["full_name_compact"] == recB["full_name_compact"]:
        return "NM-COMMON"
    return "NM-EASY"


def build_realistic(gen, pb, n_entities, target_pairs, pos_frac=0.10):
    records, ent_recs, entities = pb.entity_first_records(n_entities)
    match_rows, nonmatch_rows = [], []
    # match pairs from multi-record entities
    for eid, recs in ent_recs.items():
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                match_rows.append(pair_row(recs[i], recs[j], 1, "M-REALISTIC", []))
    # subsample matches to the positive budget, then draw 1:9 non-matches
    n_match = min(len(match_rows), round(target_pairs * pos_frac))
    if len(match_rows) > n_match:
        match_rows = gen.rng.sample(match_rows, n_match)
    target_nm = round(n_match * (1 - pos_frac) / pos_frac)
    seen = set()
    tries = 0
    while len(nonmatch_rows) < target_nm and tries < target_nm * 40:
        tries += 1
        a, b = gen.rng.sample(records, 2)
        if a["entity_id"] == b["entity_id"]:
            continue
        key = (a["PATID"], b["PATID"])
        if key in seen:
            continue
        seen.add(key)
        nonmatch_rows.append(pair_row(a, b, 0, coarse_nm_case(a, b), []))
    return records, entities, match_rows + nonmatch_rows


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
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--pools", default=str(HERE / "pools"))
    ap.add_argument("--stats", default=str(HERE / "synthetic_data_stats.json"))
    ap.add_argument("--finetune-match", type=int, default=16000)
    ap.add_argument("--finetune-nonmatch", type=int, default=24000)
    ap.add_argument("--realistic-pairs", type=int, default=10000)
    ap.add_argument("--realistic-entities", type=int, default=14000,
                    help="record population behind the realistic-eval / blocking-eval")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        (args.finetune_match, args.finetune_nonmatch,
         args.realistic_entities, args.realistic_pairs) = 1600, 2400, 2000, 1000

    stats = Stats.load(Path(args.stats))
    pools = Pools.load(Path(args.pools))
    gen = Generator(args.seed, stats, pools)
    pb = PairBuilder(gen)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    v = args.version

    # ---- fine-tune corpus (case-first) ----
    ft_rows = build_finetune(gen, pb, args.finetune_match, args.finetune_nonmatch)
    ft_eids = {r["entity_id_a"] for r in ft_rows} | {r["entity_id_b"] for r in ft_rows}
    test_set = test_entities(sorted(ft_eids), args.seed, args.test_frac)
    # A pair goes to test iff either side's entity is held out (§10).
    def is_test(r):
        return r["entity_id_a"] in test_set or r["entity_id_b"] in test_set
    train_rows = [r for r in ft_rows if not is_test(r)]
    test_rows = [r for r in ft_rows if is_test(r)]
    write_pairs(train_rows, out / f"finetune_train_v{v}.csv")
    write_pairs(test_rows, out / f"finetune_test_v{v}.csv")

    # ---- realistic-eval (entity-first) + record-level blocking-eval ----
    rec_list, _rl_entities, rl_rows = build_realistic(
        gen, pb, args.realistic_entities, args.realistic_pairs)
    write_pairs(rl_rows, out / f"realistic_eval_v{v}.csv")
    rec_df = pd.DataFrame(rec_list)[RECORD_SCHEMA]
    rec_df.to_csv(out / f"blocking_eval_v{v}.csv", index=False)

    print(f"fine-tune: {len(train_rows)} train + {len(test_rows)} test "
          f"({sum(r['label'] for r in ft_rows)} match / {len(ft_rows)} total)")
    print(f"realistic-eval: {len(rl_rows)} pairs ({sum(r['label'] for r in rl_rows)} match), "
          f"{len(rec_list)} records")
    print(f"wrote 4 files to {out}/ at v{v}")


if __name__ == "__main__":
    main()
