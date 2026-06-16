# Deterministic Entity-Resolution Rules

A hand-authored, **deterministic** classifier for patient-record pairs. Given two
records it returns one of three decisions:

- **`match`** — the same patient (high precision; safe to auto-merge / silver-label positive).
- **`non_match`** — different patients (safe to auto-label negative).
- **`review`** — genuinely ambiguous; route to a human. No deterministic rule can
  decide it without information we don't have.

This is **not** Fellegi–Sunter. There is no weight summation and no learned
threshold — just an ordered cascade of explicit rules. Each rule carries a
`rule_id`; the engine (`rules_alliance.py`) emits that same `rule_id` so every
decision is traceable back to this document. **This file and the code must stay
in sync via `rule_id`.**

The design is **precision-first**: when in doubt we prefer `review` over a wrong
auto-decision. The review queue is the safety valve.

---

## Why these rules look the way they do (empirical grounding)

Measured on `data/synthetic/synthetic_test_v3.csv` (10,000 pairs; 2,000 match / 8,000 non-match):

1. **Only ~510 / 2,000 positives carry a surviving strong ID** (100 a full 9-digit
   SSN, 410 a derivable last-4). The other **~1,590 positives are demographic-only**
   (name + DOB anchored, one field corrupted). Rather than defer all of these, the
   demographic tier separates them from look-alikes by **counting contradictions**
   (a true match disagrees on ≤1 field; a forced negative on ≥2).
2. **An equal full SSN is NOT unconditionally decisive.** 186 negatives
   (`NM-SSN-03` typo-collision, `NM-SSN-04` junk) share an equal full SSN, and in
   every one **name and DOB both strongly disagree** (0/186 share DOB). So: equal
   SSN decides `match` only when name OR DOB corroborates; equal SSN with both
   disagreeing is a collision → `non_match`.
3. **A large DOB gap is a hard contradiction.** Real matches drift DOB by at most ≈1
   (off-by-one / transposition ⇒ `near`); a *large* DOB gap with an agreeing name is
   a different person (`NM-COMMON`, `NM-HARD-…+FIRSTNAME+LASTNAME`) → `non_match`.
4. **Only true look-alikes are ambiguous.** `POL-AMBIG-03` (same name+DOB+address,
   no SSN, a *match*) and `NM-HH-TWIN/TRIPLET` (same name+DOB+address, a *non-match*)
   are constructed to be indistinguishable without a strong ID — **these, and only
   these, route to `review`.**

---

## Field schema

Records are read through `utils.alliance_schema.prep_paired_df`, which yields the
friendly fields (per side, `_l` / `_r`): `first_name, middle_name, last_name,
suffix, dob` (YYYY-MM-DD), `ssn` (9-digit), `ssn4` (last-4), `sex, address,
address2, city, state, zip, phone` (space-joined phone set), `email`. Missing = empty.

---

## Comparators

Each comparator is symmetric and returns a small ordered label. Targets cite the
synthetic `case_type` the comparator is built to handle.

### C1 — Name (`name_level ∈ {exact_tokens, strong, weak, disagree}`)
Names are compared as **token sets** (union of first/middle/last, split on
whitespace and hyphen) plus a letters-only **compact** form — not field-by-field —
because clerical entry shuffles tokens across the three name fields.

Two tokens are **equivalent** if any of:
- identical;
- **nickname** equivalence from `synthetic_data_generation/pools/nicknames.json`
  (ROBERT↔BOB, JOSE↔PEPE, …) — same pool the generator used (train/serve parity);
- **initial**: one is a single letter that prefixes the other (M ↔ MICHAEL);
- **truncation**: one is a prefix of the other, both length ≥ 4 (CHRISTOPHER ↔ CHRISTO);
- **typo**: longer token ≥ 4 chars and `damerau_levenshtein ≤ 1` (one edit incl. a transposition, so MARY↔AMRY, DIAZ↔IDAZ count) or `jaro_winkler ≥ τ_jw` (τ_jw = 0.92).

Greedily match each token of the smaller set to an equivalent token of the larger.

- **`exact_tokens`** — token sets equal, or compact forms equal. Covers first↔middle
  swap (`M-NAME-02`), two-surname shuffle/collapse (`M-NAME-03/04`), Vietnamese
  order swap (`M-NAME-05`), hyphen/space/concat variants (`M-NAME-01/16`).
- **`strong`** — every token of the smaller set matched (subset under equivalence).
  Covers dropped middle / compound-first drop (`M-NAME-07`), nickname (`M-NAME-09`),
  initial↔full (`M-NAME-06/13`), truncation (`M-NAME-14`), single typo (`M-SSN-02`, `M-L4-02`).
- **`weak`** — ≥ 2 tokens matched but conflicting extra tokens on both sides
  (e.g. twin: same first+last, different middle initial — `NM-HH-TWIN`).
- **`disagree`** — ≤ 1 token shared (only a surname, or nothing). Covers siblings,
  `NM-HARD-LASTNAME`, common-name first-name conflicts.

### C2 — DOB (`dob_level ∈ {exact, near, disagree, missing}`)
- `exact` — equal. `near` — off-by-one day / off-by-one year / month-day
  transposition (`M-DOB-02/03/04`; rare but present). `disagree` — any larger gap.
  `missing` — absent on a side.

### C3 — SSN (full + last-4)
Per side: `full` = 9-digit `ssn` if present; `l4` = `ssn[-4:]` if full present else `ssn4`.
- `both_full` and equal/unequal.
- cross/last-4: `l4_equal` / `l4_conflict`, with a flag for the **negative-coupling**
  case where one side has a full SSN and the other a *mismatching* last-4 (`NM-SSN-05`).

### C4 — Address (`addr_level ∈ {same_line1, same_zip_diff_street, diff, missing}`)
Street fields are already normalized in cleaning. `same_line1` is the strong corroborator.

### C5 — Phone (`phone_overlap` boolean)
Intersection of the full 10-digit phone sets is non-empty. Area-code-only overlap
does **not** count (`NM-COMMON-05`).

### C6 — Email (`email_equal`, `email_domain_typo`)
`email_domain_typo` = identical local part, domain `levenshtein ≤ 1` (`M-EMAIL-02`).

### C7 — Discriminators
- `suffix_conflict` — both suffixes present and different (JR vs SR; `NM-HH-JR-SR`).
  Present-vs-absent is **neutral** (`M-NAME-08` is a true match).
- `middle_conflict`, `sex_conflict` — informational; never used to auto-reject
  (sex flips in ~1.5–2% of true matches).

---

## Decision cascade (ordered; first hit wins)

### SSN tier
| rule_id | condition | decision | scenarios |
|---|---|---|---|
| `R-SSN-MATCH` | both full SSN present & **equal** & (`name_level≠disagree` OR `dob_level≠disagree`) | **match** | M-SSN-01..11 |
| `R-SSN-COLLISION` | both full SSN present & equal & name `disagree` & dob `disagree` | **non_match** | NM-SSN-03, NM-SSN-04 |
| `R-SSN-CONFLICT` | both full SSN present & **unequal** | **non_match**, but → **review** if name `exact_tokens` & dob `exact` | — |
| `R-SSN-L4-MATCH` | one side full, other last-4, `full[-4:] == last4`, name∈{exact,strong}, dob∈{exact,near} | **match** | M-SSN-09, M-L4-04 |
| `R-SSN-L4-CONFLICT` | one side full, other last-4, `full[-4:] ≠ last4` | **non_match**, but → **review** if name `strong`+ & dob `exact` | NM-SSN-05 |

### Last-4 tier (both sides have a last-4, neither pair handled above)
| rule_id | condition | decision | scenarios |
|---|---|---|---|
| `R-L4-MATCH` | `l4_equal` & name∈{exact_tokens,strong} & dob∈{exact,near} | **match** | M-L4-01..04 |
| `R-L4-COLLISION` | `l4_equal` & name `disagree` | **non_match** | NM-SSN-01/02, NM-HARD-LAST4 |
| `R-L4-REVIEW` | `l4_equal` & (name `weak` OR dob `disagree`/`missing`) | **review** | borderline last-4 |
| `R-L4-CONFLICT` | `l4_conflict` | **non_match**, but → **review** if name `exact_tokens` & dob `exact` | — |

### Demographic tier (no usable strong-ID pair — the hard zone)

**Key insight from the scenario design:** a *true match* agrees on name + DOB; its
corruption is one dimension at a time, so it almost always keeps a **shared
phone/email** or has *no* contradicting field. A *hard negative* that happens to
share a name either disagrees on the name, has a **large DOB gap** (real matches
only ever drift DOB by ≈1), or shares *only* name+DOB while every contact field
differs. We exploit exactly that.

For a pair that reaches this tier we tally `demographic_evidence`:
- **`link`** = a shared **phone or email** — a positive identity signal two
  different same-named people rarely produce. A shared **address is NOT** a link:
  households and twins share it (that's the ambiguous case below).
- **`contradictions`** = number of fields present on *both* sides that disagree,
  among `{address (different street), phone (no overlap), email (different)}`.
  Note address+zip move together (one relocation) — counted once, via `address`.

| rule_id | condition | decision | scenarios |
|---|---|---|---|
| `R-DEMO-NAMEDIFF` | name `disagree` | **non_match** | NM-HARD-LASTNAME/DOB/ADDR/PHONE, siblings, spouses |
| `R-DEMO-JRSR` | `suffix_conflict` (JR vs SR …) | **non_match** | NM-HH-JR-SR |
| `R-DEMO-DOBCONTRA` | name agrees but dob `disagree` (large gap) | **non_match** | NM-COMMON, NM-HARD-…+FIRSTNAME+LASTNAME |
| `R-DEMO-MATCH` | name∈{exact,strong} & dob∈{exact,near} & (`link` **or** `contradictions == 0` with no identical-address case) | **match** | M-NOSSN, M-NAME, M-ADDR, M-PHONE, M-EMAIL, M-DOB, M-SEX, M-ZIP, M-PED |
| `R-DEMO-AMBIG` | name `weak`; **or** name+dob agree & `contradictions == 0` & **address identical** (no link); **or** name+dob agree but a contact field differs with **no** link | **review** | POL-AMBIG-03 vs NM-HH-TWIN (name+dob+address identical); mover/changed-contact vs namesake |

The pairs sent to review are the genuinely undecidable ones: **identical name+DOB
with no strong ID and either an identical address (POL-AMBIG-03 match vs a twin
non-match — constructed to be indistinguishable) or all contact info differing
(a person who moved/changed contact vs a same-name+DOB coincidence).**

---

## Thresholds (named constants in `rules_alliance.py`)
- `TAU_JW = 0.92` — Jaro-Winkler cut for single-token typo equivalence.
- `LEV_MAX = 1` — max (Damerau-)edit distance for typo / email-domain equivalence.
- `PREFIX_MIN = 4` — min length for truncation/prefix equivalence.

Each is justified by `evaluate_rules.py --calibrate`, which prints the comparator
distributions for corrupted positives vs. hard negatives on the training file.
