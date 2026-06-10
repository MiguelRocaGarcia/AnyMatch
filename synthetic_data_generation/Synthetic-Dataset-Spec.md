# Synthetic Dataset Spec for AnyMatch Fine-Tuning

## Generated files at a glance

Running `python synthetic_data_generation/generate_synthetic.py --seed 42 --version 2` writes **two** CSVs to `data/synthetic/` (all `vN`-versioned). The generator first builds a population of synthetic patient *entities* (each a real-world person with a stable `entity_id`), emits one or more records per entity with realistic clerical corruptions (typos, name-token shuffles, missing SSN, address moves, …), and then assembles those records into the files below. All ground-truth provenance (`entity_id`, `case_type`, `corruptions_applied`) rides **inside** these files — there are no separate manifest files.

| File | Level | Match:non-match | Purpose |
|---|---|---|---|
| `synthetic_train_v2.csv` | pair | ~1:1.5 (balanced) | **Training set** for fine-tuning the mode4 checkpoint. Built **hybrid** (§9): a realistic entity-first *bulk* (real missingness joint + multi-field corruptions, plus a dirty tail) with a budgeted minority of hard-scenario *overlays* (§8) so the model sees both the natural distribution and the specific failure modes — SSN as decisive identity, cross-name-field token shuffles, household false-matches. Drawn from the **train** entity partition. |
| `synthetic_test_v2.csv` | pair | `--test-prevalence`, default 1:4 | **Honest evaluator.** Realistic-ish match prevalence + **hard negatives only** (no random strangers), the **same difficulty construction as train** (SSN bands + every named hard case + key-sharing negatives), realistic missingness + multi-corruption. **Entity-disjoint** from train (§10). 1:4 (not 1:9) keeps enough positives for stable per-`case_type` recall. |

> **Design note (v0.5, 2026-06-09).** This supersedes the earlier four-file, case-first-dominant design. Two findings drove the change: (1) the old training corpus was unrealistically clean — 70% of positives had exactly one corruption and field presence was distorted by forced `ensure_*`/`force_full_ssn` calls — so the model learned brittle single-signal rules; (2) the old `realistic_eval` measured precision against uniformly-random strangers (0% shared DOB/address/phone/SSN), not the hard near-duplicates a matcher faces. The redesign makes realistic entity-first generation the backbone, demotes hard scenarios to budgeted overlays that **never** force field presence, adds a correlated **dirty tail**, and collapses output to the two files above. Goal: an AnyMatch **mode4** model robust enough on very dirty, sparse pairs to emit **silver labels** for unlabeled data.

**Column conventions (pair files).** Model-input columns carry `_l`/`_r` suffixes and mirror the MDM-cleaned schema (consumed by `df_serializer` after `FEATURE_RENAMES`). Provenance columns use other naming so `df_serializer` skips them: `PATID_A`/`PATID_B`, `entity_id_a`/`entity_id_b` (ground-truth cluster ids), `case_type` (e.g. `M-SSN-04`), `corruptions_applied` (JSON list), and `label`. See §11 for the full layout.

---

**Status:** v0.4 draft — review-pass revisions 2026-05-28: (a) corruption budgets will be **calibrated from real within-cluster field agreement** (new `within_cluster_agreement` stats block, §5.8) instead of hand-guessed; (b) **correlated entity sampling** — geo and missingness sampled as joint blocks, first name conditioned on DOB year (§5.9, §6); (c) **`Phones_set`/`full_name_tokens` emitted sorted** for reproducibility (§3); (d) **NM-SSN-05** added to teach the SSN↔last-4 coupling negatively (§8.2); (e) **name pools now from public reference data** (Census surnames, SSA given-names-by-year), LLM reserved for the two curated tables (§14); (f) realistic-eval / blocking-eval clarified as **deliberately blocking-agnostic** (§9.2). **Stats re-run landed 2026-05-28:** §5.8/§5.9 now carry measured numbers, and §7 corruption budgets are recalibrated from them — headline finding is that **name change (~48%) and address move (~70%) are the *norm* in true pairs, while tidy off-by-one DOB corruptions are essentially nonexistent (<0.1%)**; framing of §7 updated accordingly. **Pre-build decisions locked 2026-05-28:** (1) K-per-entity keyed by identifier band (§7); (2) fine-tune corpus built **case-first** to exact per-scenario budgets, realistic-eval + M-MIX built **entity-first** (§9); (3) corruptions applied as **independent per-field Bernoulli at the §5.8 marginals** (§7); (4) per-field calibration bound fixed — name/address/phone/email from `by_ssn`, sex from `by_ssn_dob`, DOB from `by_ssn` (§7); (5) pools sourced from **US Census surnames + SSA names-by-year + Chicago street names** via `build_pools.py` (§14). Spec is build-ready. **Generator built 2026-06-08:** `build_pools.py` (offline-first) + `generate_synthetic.py` + `qa_checks.py` are implemented and verified — full 40k fine-tune + 10k realistic-eval run is ~4 s, byte-reproducible from `--seed`, with all §12 structural checks passing (§14.5, §13).

**Status (v0.3):** §5 filled from `synthetic_data_generation/synthetic_data_stats.json` (n=163,364, k-anon ≥20); core design decisions locked 2026-05-28. Decisions locked this revision: (1) **input schema** — pass the three name fields raw, include `AddressLine2` (§2); (2) **scale** — 40,000-pair fine-tune corpus at 1:1.5 match:non-match + 10,000-pair realistic-eval at 1:9 (§9), with concrete per-scenario budgets; (3) **split** — 15% entity-disjoint hold-out (§10); (4) **generation approach** — code-primary generator with a local LLM (Ollama) seeding static vocabulary pools only (§14); (5) **stakeholder-gated scenarios** excluded by default behind a flag (§8.3). Remaining open items (§13) are stakeholder sign-offs and the build itself (generator + QA notebook), not design or stats gaps.

Companion docs:
- `docs/Data-Cleaning-Guide.md` — the cleaning pipeline whose outputs this dataset must mimic.
- `synthetic_data_generation/extract_mdm_stats.py` — produces aggregate statistics from `MDM_Population_cleaned_v1.csv` (PHI stays local; only aggregates land in the repo).

---

## 1. Purpose

Generate a synthetic dataset of FQHC patient record pairs to **fine-tune** the AnyMatch GPT-2 checkpoint (currently zero-shot from the 9 public EM datasets, mode4) for the AllianceChicago entity-resolution task.

The fine-tuning target is two specific behaviors the zero-shot model fails on:

1. **Field meaning.** The zero-shot model has no semantic understanding of *what each field is*. It treats SSN, email, phone, and ZIP as roughly interchangeable categorical strings. In FQHC patient ER, fields are deeply *not* interchangeable: a matching valid SSN is essentially proof of identity, while a matching ZIP code is barely a signal at all. The model must learn each field's meaning and weight — that a non-null, structurally valid, non-junk SSN equality decides the pair (maiden→married, moves, clerical drift do not change this); that `last_4_SSN` is a meaningful backup when full SSN is missing; that shared address or shared phone is a weak signal explainable by household co-residence (shelters, group homes, families); that twin / Jr-Sr / common-name pairs are *not* matches even when many fields align.
2. **Field independence.** The model treats all fields as conditionally independent given the match label. They aren't:
    - **Name fields are interchangeable buckets.** `FirstNM`, `MiddleNM`, and `LastNM` are routinely shuffled across each other by clerical entry — the same human name lands in different slots between records (Hispanic two-surname shuffling, Vietnamese name-order swap, middle vs initial, hyphenation, compound names). Per-field agreement is the wrong unit; the cross-field token union is.
    - **Address fields are physically correlated.** `AddressLine1` / `City` / `State` / `Zip` move together; disagreement on one but agreement on the others typically indicates a move, a typo, or shared housing — not three independent disagreements.
    - **Phones are unordered.** The four phone slots (`PrimaryPhoneNBR`, `Phone01–03NBR`) carry no semantic difference; order is noise. Set overlap is the signal.
    - **SSN ↔ last_4_SSN are not independent.** Same person's `last_4_SSN` is the last 4 digits of their `SSN`. The model should treat these as one identifier with two presentations, not two features.
    - **DOB clerical errors (transposed digits, off-by-one) shouldn't break matches when other strong identifiers agree** — i.e., DOB equality is not a hard prerequisite for match when name and SSN are aligned.

## 2. What the model actually sees (input schema)

**Decision (locked 2026-05-28):** the three name buckets (`FirstNM`, `MiddleNM`, `LastNM`) are passed through *as separate fields* (not collapsed into a derived token-set), so the fine-tuning corpus must teach the model how to combine them. `AddressLine2` **is included** in the model schema (29.4% present per §5.1; it carries apt-level signal useful for household disambiguation). This supersedes the prior FEATURE_RENAMES that used a single derived `name` from `full_name_tokens`.

| Model attr | Source column (MDM-cleaned) | Notes |
|---|---|---|
| `first_name` | `FirstNM_clean` | Passed as its own field so the model learns cross-name-field coupling. |
| `middle_name` | `MiddleNM_clean` | Same. |
| `last_name` | `LastNM_clean` | Same. |
| `dob` | `BirthDT_clean` | Standardized date. |
| `sex` | `SexAtBirthDSC_clean` | MALE / FEMALE / OTHER / null. |
| `ssn` | `SSN_clean` | Full 9-digit. Missing in **78.6%** of records (only 21.4% have full SSN). |
| `ssn last 4` | `last_4_SSN` | Backup signal when full SSN is missing. Missing in **64.3%** of records (35.7% have last-4); **14.3% of records have last-4 but no full SSN** (the meaningful backup-only population). |
| `address line 1` | `AddressLine1_clean` | Street line. |
| `address line 2` | `AddressLine2_clean` | Apt/unit line. Included (locked 2026-05-28). 29.4% present; mostly `APT`. |
| `city` | `CityNM_clean` |  |
| `state` | `StateCD_clean` | 2-letter USPS. |
| `zip` | `ZipCD_clean_base` | 5-digit primary. |
| `phone` | `Phones_set` | Derived: whitespace-joined set of all non-null cleaned phone numbers. **Emitted sorted** (see §3) so the same set always serializes to the same string. |
| `email` | `Email_clean` |  |

Missing values arrive at the prompt as the literal string `'N/A'` (per `df_serializer`).

**Implications for synthesis:** since the model sees the three name fields independently, the fine-tuning data must explicitly demonstrate how tokens move *between* `FirstNM_clean`, `MiddleNM_clean`, and `LastNM_clean` for the same human (Hispanic two-surname swaps, Vietnamese order swaps, middle-name promotion/demotion, etc.). This drives the M-NAME-* scenarios in §8. If the schema later switches back to a token-set derivation, those scenarios still hold — they just become a sanity check on the derivation rather than a teaching signal.

## 3. Generation schema

The synthetic dataset is generated at the **MDM-cleaned column level**, drop-in compatible with `MDM_Population_cleaned_v1.csv`. The same FEATURE_RENAMES + derivations (`full_name_tokens`, `full_name_compact`, `Phones_set`, `Address_normalized`) are then applied to produce model inputs. This keeps the synthetic data interchangeable with real cleaned data and lets us validate the whole pipeline end-to-end on synthetic inputs.

Per-record columns we generate (clean values only — `_raw` is not generated for the **pair-level** matcher files since synthesis has no upstream noise to preserve and the model never reads `_raw`). The **record-level** blocking-eval file (§11.1) is the one exception: it carries the full cleaning-output schema including `_raw` columns (populated by copying `_clean`) for drop-in schema parity with the real patient table.

```
FirstNM_clean, MiddleNM_clean, LastNM_clean, SuffixNM_clean
BirthDT_clean
SSN_clean, last_4_SSN
AddressLine1_clean, AddressLine2_clean
CityNM_clean, StateCD_clean, ZipCD_clean_base, ZipCD_clean_ext
PrimaryPhoneNBR_clean, Phone01NBR_clean, Phone02NBR_clean, Phone03NBR_clean
Email_clean
SexAtBirthDSC_clean
```

Derived columns produced post-generation (same logic as Data-Cleaning-Guide §"Global Cross-Field Transformations"):

```
full_name_tokens, full_name_compact, Phones_set, Address_normalized
```

**`Phones_set` must be emitted as a deterministically *sorted* whitespace-joined string** (e.g. `" ".join(sorted(phones))`), not a raw Python `set`. A `set`'s iteration order is not stable across processes, so the same two phone numbers could otherwise serialize to different strings in different records — injecting noise into the exact-overlap signal §1 is trying to teach and breaking byte-for-byte `--seed` reproducibility. The same applies to any other set-derived field (`full_name_tokens`): sort before joining.

**Fields deliberately NOT generated for the model input:**
- **`PATID`** — uniqueness identifier; carries no matching signal. The pair CSV carries `PATID_A` / `PATID_B` for provenance (note: capital `_A`/`_B` so `df_serializer` skips them — it only consumes `_l`/`_r`-suffixed columns).
- **`valid_record`** — by §4 principle 2 this is always `True` for synthetic data; including it in the prompt would just consume a token per record telling the model something it can already assume. Omitted from per-record columns and from the pair CSV.

## 4. Design principles

1. **Cleaned-output convention.** Synthetic values follow the conventions in `docs/Data-Cleaning-Guide.md`: uppercase names, ASCII only, standardized USPS suffixes, etc. We do *not* generate dirty inputs and re-clean them; we generate post-cleaned values directly. (Exception: we *do* simulate the few classes of corruption that survive cleaning — e.g., one-digit SSN typos, name-field swaps, DOB transpositions.)
2. **`valid_record=True` only.** Per `CLAUDE.md`, downstream inference filters to `valid_record=True`. The model never sees invalid records, so we never generate them.
3. **Entity-first for the bulk.** Sample N synthetic persons (entities) from realistic distributions; per entity produce K record variants by applying transformations; form match pairs within an entity and non-match pairs across entities. This naturally generates realistic mixed-corruption patterns.
4. **Case-first construction for the fine-tune corpus.** The training corpus is built case-first: each §8 scenario is constructed to its exact §9.1 budget (twins, Jr-Sr, junk-SSN last-4 collisions, name-order swaps would be too rare to learn from emergent entity-first sampling). Entity-first is used for the realistic-eval and the M-MIX bulk (§9 assembly mechanism).
5. **Realistic distributions cited, not invented.** Every distribution choice (name top-N, DOB year mix, missingness rate per field, ZIP/state mix) cites `synthetic_data_stats.json`. Where we deliberately diverge from real (e.g., oversampling SSN-match-everything-else-disagrees), the spec records the deviation explicitly.
6. **Two-stage output.** §11 — a balanced/oversampled fine-tune corpus *and* a realistic-distribution holdout eval.
7. **Entity-disjoint split.** No synthetic entity appears in both train and test. Prevents the model from memorizing identities. §10.
8. **Deterministic seeds.** Generation accepts a `--seed`. All randomness derives from it.
9. **Auditable provenance.** Every emitted pair carries `entity_id_a`, `entity_id_b`, `case_type` (e.g. `M-SSN-04`), and `corruptions_applied` (list of transformation names). This is shed before training but kept for inspection.
10. **No-SSN is the normal path, not the edge case.** Per §5.1, 64.3% of real records have no SSN at all. Scenario weighting (§8 preamble) and entity generation (§6) must reflect this — SSN-led matching is a strong-signal *subset*, not the default.
11. **Pediatric records are isolated as their own bucket.** DOB-2010s+ patients have a constrained identifier set (no SSN typically; parent-shared phone/address). Pediatric scenarios (M-PED-* / NM-PED-*) live in their own bucket because their missingness shape and household-collision pattern differ qualitatively from adults.
12. **ZIP3 ↔ State is treated as a hard constraint.** Real agreement is 99.99% — the generator never emits inconsistent (ZIP, State) pairs. Any "cross-state move" scenario updates both fields together.
13. **Synthetic records must survive `valid_record=True`.** The generator never emits values that the cleaning rules would flag (`docs/Data-Cleaning-Guide.md`): no `BABY BOY`/`BABY GIRL`/`DUPLICATE`/`DO NOT USE`/`MEDICARE`/`<MRG>`/etc. tokens in name or address fields, no junk SSN patterns, no junk emails, no NANP-invalid phones. Scenarios that *only* exist inside the invalid bucket (e.g., `MiddleNM=DO NOT USE`) are out of scope and must be fixed upstream in cleaning if they matter operationally.

## 5. Real-data statistics (measured)

Source: `synthetic_data_generation/synthetic_data_stats.json` (`n_rows=163,364`, k-anon threshold 20, `valid_record_rate=97.6%`). Numbers below are measured, not estimated. Anything still labeled **[TBD]** in this section is a design choice (e.g., per-scenario budget) that the stats don't decide on their own.

### 5.1 Per-field presence

Rates of `*_clean` being non-null. Drives marginal P(field present) per entity in §6.

| Field | Present | Notes |
|---|---|---|
| `FirstNM_clean` | 99.3% | Effectively always present. |
| `LastNM_clean` | 99.7% | Effectively always present. |
| `MiddleNM_clean` | **19.4%** | Sparse — and when present, mostly a **single-letter initial** (`p50=1, p75=3` chars). Implication: M-NAME-06 (initial vs full middle) is far more central than the spec originally suggested. |
| `SuffixNM_clean` | **0.21%** | Essentially never recorded as its own field. Real Jr/Sr signal lives in name fields (45 `JR` in MiddleNM, suffix tokens leak into LastNM). The M-NAME-08 *"suffix in wrong slot"* recipe is more important than M-NAME-08 *"suffix field set"*. |
| `BirthDT_clean` | 99.7% | Effectively always present. |
| `SSN_clean` (full 9-digit) | **21.4%** | Minority. The model must default to no-SSN matching as the *normal* path. |
| `last_4_SSN` | 35.7% | Of those, **14.3% of all records** have last-4 but no full SSN (the backup-only band). |
| (no full SSN, no last-4) | **64.3%** | Majority population. M-NAME-* and address/phone-led scenarios dominate. |
| `AddressLine1_clean` | 96.3% |  |
| `AddressLine2_clean` | 29.4% | When present, mostly `APT` (63.7% of fills), `UNIT` (4.7%), `BSMT` (3.1%). |
| `CityNM_clean` | 97.4% |  |
| `StateCD_clean` | 97.2% |  |
| `ZipCD_clean_base` | 97.2% |  |
| `ZipCD_clean_ext` | 6.1% | Excluded from §2 model schema (too sparse). |
| `PrimaryPhoneNBR_clean` | 81.8% |  |
| `Phone01NBR_clean` | 76.9% |  |
| `Phone02NBR_clean` | 43.3% |  |
| `Phone03NBR_clean` | 2.0% | Almost never present. |
| Phones per record | 0:5.6%, 1:12.4%, 2:**55.6%**, 3:25.2%, 4:1.2% | Most records carry exactly 2 phones. Drives §6 step 7 directly. |
| `Email_clean` | 31.1% |  |
| `SexAtBirthDSC_clean` | 79.1% | Among present: F 55.5%, M 44.2%, OTHER 0.3%. |
| No address at all | 2.4% |  |
| No phone at all | 5.6% |  |

### 5.2 Name distributions

- **`FirstNM_clean` top 5:** MARIA, JOSE, MICHAEL, DAVID, JUAN. 1,167 names appear ≥20 times; the long tail (`below_threshold_total=135,800`) is 83% of the column. Strong Hispanic over-representation.
- **`LastNM_clean` top 5:** GARCIA, WILLIAMS, HERNANDEZ, MARTINEZ, JOHNSON. 958 names ≥20 times.
- **`MiddleNM_clean` top values:** initials A, M, L, J, D… (the alphabet dominates because most middle names are recorded as initials). Junk tokens visible in the raw stats (`DO NOT USE`, `DUPLICATE`, `BABY GIRL`, `MEDICARE`, …) all live inside the **invalid-record bucket** (per `docs/Data-Cleaning-Guide.md` they trigger `valid_record=False`) and never reach the model at inference time, so they are **out of scope for synthesis**. The exception is **`JR`** (45 records in `MiddleNM_clean`): not on the invalid-strings list, so those records pass through as `valid_record=True` — these are the suffix-slot-leak signal that drives M-NAME-08b.
- **Compound first name (2-token):** 1.9% → `p_compound_first ≈ 0.02`.
- **Compound last name (2-token):** 6.9% → `p_two_surname ≈ 0.07`.
- **Hyphenated last name:** 1.8% (separate from compound; some overlap).
- **Apostrophe:** ~0.3% on first, ~0.1% on last (`O'NEILL`, `D'ANGELO`, etc.).
- **Field-length distributions:** first p50=6, last p50=7, middle p50=**1** (initial-dominated).

### 5.3 DOB

- 99.7% present, span 1900–2026.
- Decade histogram (% of non-null): 1900s 0.004%, 1910s 0.05%, 1920s 0.18%, 1930s 0.74%, 1940s 2.4%, 1950s 7.2%, **1960s 11.2%, 1970s 13.3%, 1980s 18.2%, 1990s 19.7%, 2000s 14.5%, 2010s 10.0%**, 2020s 2.4%. 27% are year ≥2000 (pediatric + young adult).
- Month roughly uniform; day-of-month near-uniform with the expected dip at 29–31.

### 5.4 Geography

- **Cities:** CHICAGO **63.8%**, LEXINGTON 3.1%, CICERO 1.4%, BALTIMORE 1.3%, EVANSTON 1.2%, LOUISVILLE 1.1%, WAUKEGAN/SKOKIE ~1% each. Plus a Hawaii cluster (WAILUKU, KAHULUI, KAMUELA, HONOKAA, LAHAINA) and Wyoming (CHEYENNE, CASPER, LARAMIE).
- **States:** IL **78.8%**, KY 4.8%, HI 3.3%, NY 2.7%, WY 2.3%, MD 2.2%, IN 1.1%, CA 0.85%. Tail of 21 more.
- **ZIPs:** top-10 are all Chicago 606xx; 60639 is largest at 4.5%. ZIP3 → State agreement **99.99%** (essentially perfect; can be treated as a hard constraint).
- **Address last token (street suffix):** AVE 30.6%, ST 15.2%, RD 4.5%, DR 3.8%, PL 2.7%, CT 1.7%, BLVD 1.6%, LN 1.3%. PO Box rate 2.1%.

### 5.5 Phones

- Top area codes match the geography: 773 (Chicago, 42.7% of primaries), 312 (Chicago, 14.6%), 708 (Chicago suburbs, 6.4%), 859 (Lexington KY, 3.3%), 808 (HI, 3.1%), 872/224/847/630/815 (more Chicago overlays), 307 (WY), 502 (Louisville).
- Phone02 area-code mix mirrors Primary closely → phones really are unordered (validates §1 point about phone-slot interchangeability).

### 5.6 Email

- 31.1% present.
- Top domains: gmail.com **66.2%**, yahoo.com 17.6%, hotmail.com 4.5%, icloud.com 3.3%, aol.com 0.9%, outlook.com 0.6%. Heavy gmail concentration → top-3 covers 88%.
- **Real typo domains in the data:** `gamil.com`, `gmai.com`, `parkwestmed.ez`. Useful as an *M-EMAIL-02* one-edit-typo scenario (currently not in catalog — see §13 follow-up).

### 5.7 Clusters (true-positive cluster-size signal)

Used to calibrate K records per synthetic entity in §7.

- **By `SSN_clean`** (n_keys=30,452): singletons **87.4%**, 2-record **11.1%**, 3-record 1.3%, 4-record 0.2%, tail ≤0.05%. Of SSNs with any duplicates, ~88% are doubletons.
- **By `SSN_clean + BirthDT_clean`**: tighter (88.7% singletons, 10.0% double) — most multi-SSN groups also share DOB, validating the SSN-trumping intuition.
- **By `last_4_SSN + BirthDT_clean`** (n_keys=47,116): **81.6% singletons, 14.3% double, 3.1% triple, 0.75% quad+** — much more collision-y than full SSN. Calibrates NM-SSN-01 (last-4 collisions) prevalence.
- **By `full_name_compact + BirthDT_clean`** (n_keys=135,532): 83.8% singletons, 13.0% double, 2.5% triple, 0.5% quad — ~16% of name+DOB blocks contain duplicates. Sets a floor on M-NAME-* / M-MIX-03 weight.

### 5.8 Within-cluster field agreement (true-positive proxy) — corruption calibration

Measured 2026-05-28. Records sharing a key are a same-person proxy; field-agreement rates over within-cluster **record pairs** are the empirical corruption model that drives §7. Two cluster keys matter: **`by_ssn`** (5,696 pairs sharing a full SSN — the primary calibration; but a few are SSN-entry-error pairs, which inflates disagreement) and **`by_ssn_dob`** (4,499 pairs sharing SSN *and* DOB — higher-purity same-person, so a tighter *upper* bound on agreement). `by_last4_dob` (14,340 pairs) is collision-heavy and used for negative calibration, not corruption rates.

| Field agreement | `by_ssn` | `by_ssn_dob` | Reading |
|---|---|---|---|
| `first_exact` | **82.8%** | 93.9% | First name differs in ~6–17% of same-person pairs. |
| `last_exact` | **69.3%** | 78.9% | **Last name differs in ~21–31%** — surname change (maiden↔married, dropped second surname) is common, not rare. Validates M-SSN-04. |
| `compact_exact` | **51.8%** | 59.2% | **Only ~half of same-person pairs share an identical full name.** Name corruption is a *majority* event, not an edge case. |
| `compact_editdist_le1` | 65.9% | 75.1% | Of the name differences, only ~⅓ are single-char-typo-scale; the rest (edit-dist ≥2) are surname swaps / token changes — *bigger* than typos. |
| `tokens_set_equal` | **51.4%** | 58.7% | Cross-field token set unchanged ~half the time → token shuffles + real token changes are both large. |
| middle `both_present_equal` | 83.5% | 83.4% | When both have a middle, it differs ~16%. |
| middle `exactly_one_missing` | 21.5% | 23.6% | One side has a middle, the other doesn't, ~22% of the time. |
| DOB `exact` | **86.9%** | 100%* | DOB matches ~87%; *(`by_ssn_dob` is 100% by construction).* |
| DOB `off_by_one_day`/`year`/`transpose` | **~0.04% each** | — | **The "neat" clerical DOB corruptions are essentially nonexistent in real data.** |
| DOB `other` (larger diff) | **12.9%** | — | When DOB disagrees, it is almost always a *substantial* difference, not off-by-one. |
| `line1_exact` | **28.8%** | 33.1% | **Address moves are the norm — only ~29–33% share a street.** |
| `city_exact` | 86.6% | 88.2% | Most moves stay in-city. |
| `state_exact` | **99.0%** | 98.8% | Cross-state moves are rare. |
| `zip_exact` | 47.0% | 52.4% | ZIP differs about half the time (intra-city moves). |
| `phone_overlap_ge1` | **34.3%** | 39.4% | **Phone overlaps on only ~⅓ of matches** → phone is a weak corroborator, not expected on most true pairs. |
| `email_exact` | 56.3% | 57.4% | (small denom ~500) Email differs ~43% when both present. |
| `sex_exact` | **91.5%** | 98.5% | Sex disagrees ~8.5% in `by_ssn` but only ~1.5% once DOB is also pinned → most of that 8.5% is SSN-error pairs; **use ~1.5–2% as the true same-person sex-flip rate.** |

**Headline implications (these reshape §7 and some §8 weights):**
1. **Name instability is the dominant real corruption**, not typos. ~48% of same-person pairs have a non-identical full name, mostly from surname change / token movement rather than single-char typos. §7 should corrupt names on a large fraction of match variants, weighted toward surname-change and token-shuffle over typos.
2. **Off-by-one / transposition DOB corruptions are a teaching device, not a realistic frequency** (<0.1% each). Keep them in the fine-tune corpus for tolerance training but make them rare in the realistic-eval; the realistic DOB-disagreement mode is exact (87%) or a larger "other" gap (13%).
3. **Address moves dominate** (~70% change street, ~half change ZIP), almost always within state. The matcher cannot lean on address agreement for true pairs.
4. **Phone overlap (~34%) and email agreement are weak positive signals** — most true pairs do *not* share them.
5. The `by_last4_dob` name-agreement collapse (`compact_exact` 22.6%, `tokens_set_equal` 22.4%) confirms **last-4 + DOB is heavily collision-prone** (~77% of such pairs are different names) — calibrates NM-SSN-01/02 as genuinely hard and frequent.

All rates are conditional (denominator = pairs where the field is present on both sides) and aggregate-only (k-anon safe).

### 5.9 Joint distributions for correlated entity sampling

Measured 2026-05-28.

**`geo_joint`** — 282 `City|State|ZIP3` combos above k-anon (plus a 19,051-record tail). Heavily concentrated: **`CHICAGO|IL|606` is 63.0%** of records; the next tier is `LEXINGTON|KY|405` (3.1%), `CICERO|IL|608`, `BALTIMORE|MD|212`, `EVANSTON|IL|602`, `LOUISVILLE|KY|402`, plus the Hawaii (`…|HI|967`) and Wyoming (`…|WY|820/826`) clusters. §6 draws `(City, State, ZIP3)` as **one** joint sample from this table, which makes ZIP3↔state consistency automatic (§4.12) and prevents implausible geography (a Chicago resident on an 808 area code). The street, full ZIP, and area code are then filled conditional on the drawn geography.

**`missingness_patterns`** — 51 joint present/absent patterns above k-anon over `{ssn_full, ssn_last4, middle, address, email, phone, sex}` (only 204 records in the tail). §6 draws a **whole pattern** per entity, so co-missingness is realistic. The shape is dominated by no-SSN records:

| Pattern (ssn_full,ssn_last4,middle,address,email,phone,sex) | Count | Share | Meaning |
|---|---|---|---|
| `0001011` | 48,524 | **29.7%** | No SSN/last-4/middle/email; has address + phone + sex. **The canonical FQHC record.** |
| `0001111` | 18,802 | 11.5% | …same, plus email. |
| `0001010` | 12,424 | 7.6% | No SSN/last-4/middle/email/sex; has address + phone. |
| `1101011` | 10,863 | 6.6% | Full SSN + last-4, no middle, address + phone + sex (no email). |
| `0011011` | 8,490 | 5.2% | Last-4 only (no full SSN), address + phone + sex. |
| `0101111` | 7,929 | 4.9% | Last-4 only, with middle + email. |
| `1111111` | 4,520 | 2.8% | Everything present. |
| `0000000` | 1,008 | 0.6% | Everything missing (still `valid_record=True` on name+DOB). |

(Full 51-row table in `synthetic_data_stats.json`.) Sampling these jointly reproduces the §5.1 marginals automatically and is the single source of truth for field presence (§6).

### 5.10 Statistics we still do **not** have

- Realistic prevalence of named edge cases (twin, Jr-Sr, shelter-address) in blocking output — use heuristics in §8 for now; refine when blocking results land.
- Operator-policy ground truth for POL-AMBIG-* cases — must be supplied by stakeholders.

## 6. Entity generation

A synthetic *entity* represents a single ground-truth person. For each entity:

**Correlated sampling (added 2026-05-28).** Fields are **not** all drawn from independent marginals — that produces marginally-realistic but jointly-implausible records (e.g. a Chicago resident with a Hawaii area code) and decorrelated missingness (too many "exactly one field missing," too few genuinely thin records). Three blocks are sampled jointly from measured distributions, superseding the independent draws for those fields:

- **Geography** is one draw of `(City, State, ZIP3)` from `geo_joint` (§5.9), then the street/full-ZIP detail is filled in (step 6). ZIP3↔state stays a hard constraint (§4.12) as a consequence.
- **Missingness** is one draw of a present/absent *pattern* over `{ssn_full, ssn_last4, middle, address, email, phone, sex}` from `missingness_patterns` (§5.9). **This pattern is the single source of truth for the *presence* of every field in that set.** The per-field presence/null probabilities historically quoted in the steps below (`p_middle_present`, `p_full_ssn`/`p_last4_only`/`p_no_ssn`, address-null 0.024, the phone 0-bucket, `p_email_present`, sex-null 0.209) are **not drawn independently** — they are the marginals this joint pattern reproduces, kept only as the targets §12.7 validates. The steps below therefore govern **only value generation and sub-structure given a field is present** (e.g. *given* SSN-bearing, is it full-9 or last-4-only; *given* ≥1 phone, how many; *given* an address, apt/PO-box). The one coupling the pattern can't express on its own — `last_4_SSN == SSN_clean[-4:]` when full SSN is present — is enforced in step 5.

  *Note:* because real `last_4_SSN` is the tail of `SSN_clean`, the pattern's two SSN bits already encode the three SSN states (full⇒(1,1) 21.4%, last-4-only⇒(0,1) 14.3%, none⇒(0,0) 64.3%); `full=1,last4=0` effectively never occurs and the generator must not emit it.
- **Pediatric coupling:** DOB is drawn **first** (step 2); if it lands in the 2010s/2020s the entity is flagged pediatric and the missingness pattern is drawn from the **pediatric-conditioned** subset (so SSN/email skew absent and phone/address are parent-shared, per §4.11), keeping the pediatric bucket distinct from the adult shape. Precedence is therefore: **DOB → pediatric flag → missingness pattern → per-field values.**

1. **Sex:** *presence per the missingness pattern.* When present, value sampled from `categorical_top.SexAtBirthDSC_clean` — among present: F **55.5%**, M **44.2%**, OTHER **0.3%**. (Marginal present-rate ≈ 0.791; validated, not re-drawn.)
2. **DOB:** sample year from `dob.decade_histogram` (weights: 1940s 2.4%, 1950s 7.2%, 1960s 11.2%, 1970s 13.3%, 1980s 18.2%, 1990s 19.7%, 2000s 14.5%, 2010s 10.0%, 2020s 2.4%, pre-1940 <1%). Month uniform 1–12; day uniform 1–28 initially (lift to true day-of-month distribution once date arithmetic is verified). DOB is effectively always present (99.7%); not part of the missingness-pattern set, so the rare null is drawn here with probability **0.003**.
3. **Names:**
    - Sample first/last from the measured top-N pools weighted by their counts; for the long tail (`below_threshold_total`) draw from the public reference pools (§14.4: Census surnames, SSA given names). **First name is intended to be drawn conditioned on the entity's DOB year** (SSA-by-year), so the name is age-appropriate (a 2020s record gets a 2020s-popular name). *(Build status: DOB-conditioning is **deferred** in v1 — the offline pool build leaves `by_year` empty, so the generator falls back to the flat measured first-name distribution, which already reflects this population's real age mix. Enable via the optional SSA network enrichment in `build_pools.py`; see §14.4.)*
    - `p_compound_first = 0.02` (probability of 2-token first name, e.g. `MARIA CARMEN`).
    - `p_two_surname = 0.07` (probability of 2-token last name; Hispanic paternal+maternal convention). When triggered, set last = `<paternal> <maternal>` drawn jointly from top last-name pool.
    - `p_hyphen_last = 0.018` (independent of compound).
    - **Middle name:** *presence per the missingness pattern* (marginal ≈ 0.194, not re-drawn). When present: `p_middle_initial = 0.97` (single letter, drawn weighted by first-letter frequency from measured distribution), else full middle name from top-N pool.
4. **Suffix:** `p_suffix_field_set = 0.002` (essentially never set as its own field; Jr/Sr-ness is mostly carried in name slots — see scenario M-NAME-08 and NM-HH-JR-SR). When set, top values: `JR` (77%), `III` (11%), `SR` (8%).
5. **SSN:** *the three SSN states (full / last-4-only / none) are decided by the missingness pattern's two SSN bits* (≈ 21.4% / 14.3% / 64.3%; not re-drawn here). This step only generates the **values** for the chosen state and enforces the coupling:
    - **Full present:** 9-digit SSN with `last_4_SSN = SSN[-4:]`. Structural rules: area not in `000/666/900-999`, group not `00`, serial not `0000`.
    - **Last-4-only:** generate `last_4_SSN` only (full null). Last-4 must obey "not `0000`".
    - **None:** both null.
6. **Address:** *presence per the missingness pattern; the `(city, state, ZIP3)` triple comes from the geography draw above (`geo_joint`), not a separate sample here.* When present, generate `AddressLine1_clean` as `<NNN[N[N]]> <STREET-NAME> <SUFFIX>` where suffix is sampled from {AVE: 0.45, ST: 0.22, RD: 0.07, DR: 0.06, PL: 0.04, CT: 0.025, BLVD: 0.024, LN: 0.019, …} (re-normalized from §5.4). `p_apt = 0.294` and `p_po_box = 0.021` are *sub-structure given an address is present*; when apt set, prefix uniform from {APT: 0.64, UNIT: 0.05, BSMT: 0.03, FL: 0.03, 1ST/2ND/3RD: 0.05, …}. (Marginal address present-rate ≈ 0.976, validated.)
7. **Phones:** *presence per the missingness pattern* (the pattern's `phone` bit = the histogram's 0-bucket, 5.6%). When present, draw the **count** N ∈ {1,2,3,4} from the histogram `{1:0.124, 2:0.556, 3:0.252, 4:0.012}` *renormalized to exclude 0*. Each phone's area code is drawn **consistent with the entity's geography** (§5.9) where the mapping is unambiguous — Chicago → {773,312,872,224,847,708,630,815,…}, Lexington KY → 859, Hawaii → 808, Wyoming → 307, Louisville → 502, Baltimore → {410,443}, NY → {646,347,607,…} — falling back to the global measured primary area-code distribution otherwise. NXX and line generated to NANP-valid form (NXX ∈ 200–999, no `N11`). The first phone fills `PrimaryPhoneNBR_clean`, then `Phone01..03NBR_clean` in order.
8. **Email:** *presence per the missingness pattern* (marginal ≈ 0.311, not re-drawn). When present, domain sampled from `{gmail.com: 0.66, yahoo.com: 0.18, hotmail.com: 0.045, icloud.com: 0.033, aol.com: 0.009, outlook.com: 0.006, …}`; local part derived from name with a configurable corruption.

Each entity gets a stable `entity_id` (deterministic `E%09d` counter); each emitted record gets a fresh `PATID` solely for bookkeeping / cross-referencing rows by their `entity_id` provenance columns. `PATID` is never serialized into the model prompt (§3, §11).

## 7. Variant (record) generation per entity

**K (records per entity), keyed by identifier type (locked 2026-05-28).** The realistic cluster-size distribution depends on what identifier the entity carries, so K is drawn from the matching measured histogram in §5.7 — *not* the SSN histogram for everyone (64% of entities have no SSN and would otherwise all be forced to K=1):

| Entity identifier band | K histogram source (§5.7) | Singleton / 2 / 3 / 4+ |
|---|---|---|
| Full SSN | `clusters.by_ssn` | 87.4% / 11.1% / 1.3% / 0.2% |
| Last-4 only | `clusters.by_last4_dob` | 81.6% / 14.3% / 3.1% / 0.95% |
| No SSN at all | `clusters.by_namecompact_dob` | 83.8% / 13.0% / 2.5% / 0.6% |

This is the **realistic-eval / entity-first** K distribution. The **fine-tune corpus** is built case-first (§9.1), so there K is whatever the scenario recipe needs (≥2 for any match scenario); the realistic singleton majority only governs the realistic-eval and the entity-first M-MIX bulk. Singletons (K=1) contribute only non-match pairs.

For each pair of variants drawn from the same entity, apply corruptions. The corruption pool maps directly to the scenario catalog (§8).

**Calibrated corruption rates (from §5.8 `by_ssn` / `by_ssn_dob`).** The earlier "most pairs get 1–2 corruptions" assumption was wrong: in real same-person pairs, **corruption is the norm**. The per-field *differ* rates below are the targets the entity-first match population must reproduce (so the realistic-eval looks like reality); `by_ssn` is the working target, with `by_ssn_dob` as the lower-disagreement bound:

| Field | P(differs) target | Notes for the corruptor |
|---|---|---|
| Full name (compact) | **~0.45** (0.41–0.48) | The headline: name changes on ~half of true pairs. Compose from the field-level rates below. |
| Last name | **~0.28** (0.21–0.31) | Dominant single source — weight surname-change (maiden↔married, drop one of two surnames) heavily, not just typos. |
| First name | ~0.12 (0.06–0.17) | |
| Name-change *type*, given a change | ~⅓ typo-scale (edit ≤1), ~⅔ larger (surname swap / token move) | Don't make every name diff a single-char typo. |
| Middle (both present) | ~0.16 differ | |
| Middle one-side-missing | ~0.22 of all pairs | Drop-middle / add-middle is frequent. |
| DOB | **~0.13** differ, of which **off-by-one/transpose ≈ 0.001**, rest a larger "other" gap | Off-by-one & month-day-transpose are kept for *teaching tolerance* but must be **rare** in the realistic-eval; the realistic DOB-disagreement is a bigger jump. |
| Address line1 (move) | **~0.70** (0.67–0.71) | Address moves are the default; ~half also change ZIP, ~13% change city, **~1% change state**. |
| Phone (no overlap) | **~0.65** | Overlap on only ~⅓ of true pairs — phone is a weak corroborator. |
| Email (both present) | ~0.43 differ | |
| Sex flip | **~0.015–0.02** | Use the `by_ssn_dob` 1.5% (the `by_ssn` 8.5% is mostly SSN-error pairs, not true sex flips). |

**Which bound per field (locked 2026-05-28).** `by_ssn` and `by_ssn_dob` disagree because `by_ssn` includes a slice of SSN-entry-error pairs (different people, same typed SSN). The rule:
- **Name (first/last/compact/tokens), address, phone, email → use `by_ssn`** (the higher-drift figures). In entity resolution, *recall* matters — we want the model to tolerate the full range of same-person drift, and `by_ssn_dob` understates it by conditioning on a clean DOB match.
- **Sex → use `by_ssn_dob` (~1.5%).** The `by_ssn` 8.5% is almost entirely the SSN-error artifact, not genuine same-person sex flips.
- **DOB → use `by_ssn` (~13% differ).** `by_ssn_dob` is 100%-exact by construction and unusable here; within that 13%, off-by-one/transpose stay at their measured <0.1% (teaching-only, see §8 DOB note).

**Corruption application model (locked 2026-05-28): independent per-field Bernoulli at the calibrated marginals.** Each field is corrupted independently at its P(differs) target above. We deliberately do **not** synthesize a field-to-field corruption *correlation*, because we have no measured joint-corruption distribution to calibrate one against and inventing it would distort the validated marginals (§12.7). Real "everything is a mess" co-occurrence is instead supplied explicitly by the heavy-drift scenarios (M-MIX-02, M-SSN-11) so the tail is represented without biasing the bulk. **v0.5 update — the messiness multiplier is now ON (dirty tail).** The deferred "messiness" knob is enabled: a budgeted fraction (~15–25%) of positives draw a shared multiplier `m>1` that scales every field's P(differs) up *together*, so corruptions correlate and name+address+phone+DOB drift at once (5+ differing fields). The independent-Bernoulli marginals still govern the un-multiplied bulk; the multiplier only adds the heavy co-occurrence tail (validated by §12 check 8). SSN/last-4 are stable within a true cluster by construction (they're the cluster key), so SSN is never corrupted in match pairs except in the explicit M-SSN-08 / M-L4-03 drift scenarios.

Corruption types (see §8 for the scenarios that combine these):
- **Name corruptions:**
    - Single-character typo (insert/delete/substitute/transpose) on first or last.
    - Drop middle entirely (null it).
    - Collapse middle to initial (`ANNE → A`) — most common middle-name transformation per §5.1.
    - Expand initial to full (`A → ANNE`, drawn from a small curated table) — reverse direction.
    - Swap first↔middle.
    - Swap last↔middle.
    - Nickname substitution (`ROBERT↔BOB`) — limited curated table.
    - Hyphenation/spacing variant (`ANNE-MARIE↔ANNE MARIE↔ANNEMARIE`).
    - Append/drop suffix in `SuffixNM_clean`.
    - **Suffix-into-wrong-slot:** move `JR`/`SR`/`III` from `SuffixNM_clean` to the end of `MiddleNM_clean` or `LastNM_clean` (and null the suffix field). Required by M-NAME-08b.
    - Drop one surname from two-surname last.
- **DOB corruptions:** off-by-one day, off-by-one month, off-by-one year, month-day transposition (only meaningful when day ≤ 12), null out.
- **SSN corruptions:** drop full SSN (keep `last_4_SSN`), null both, one-digit typo on full SSN (used in non-match-collision scenarios), digit transposition. **`last_4`-only corruption:** drop full but preserve a *correct* last-4 (required by M-L4-* scenarios); also a *malformed* last-4 (random 4 digits) to feed NM-SSN-01.
- **Address corruptions:** move (entirely new address), drop AddressLine2, change apt only, change house-number digit, null street, change city/state/ZIP consistently (always together — ZIP3↔state agreement is hard per §5.4).
- **Phone corruptions:** drop one slot, drop all, add new number, replace a number, **area-code-preserving local-number change** (used in NM-COMMON-05 to teach that area-code overlap is not phone overlap).
- **Email corruptions:**
    - Drop entirely.
    - Replace with new local part.
    - Replace with new domain.
    - **One-character domain typo** (`gmail.com → gamil.com` / `gmai.com` / `gnail.com`). Required by M-EMAIL-02. Stays a match.
- **Sex corruptions:** flip (rare — clerical), null, or **set to `OTHER`** on one side (required by M-SEX-02).

## 8. Scenario catalog

Naming: `<M|NM|EDGE>-<bucket>-<NN>`. Each scenario records:
- **Teaches:** which behavior this case targets.
- **Recipe:** how to construct it from an entity (M) or two entities (NM).
- **Label:** match=1, non-match=0.
- **Fine-tune oversample target:** see the per-bucket budget tables in §9.1 (locked 2026-05-28).
- **Realistic-eval prevalence:** drawn to match natural frequency; eval set sizing in §9.2.

**Weighting principle (post-stats):** the fine-tune corpus rebalances around the measured population shape. **No-SSN matching is the *normal* path** (64.3% of records have no SSN at all); SSN-led matching is a *strong-signal subset* (21.4% full SSN; 14.3% last-4-only). Per-bucket budget hints below reflect this — they are first-cut targets to be set explicitly in §9.

| Bucket | Population basis | First-cut weight |
|---|---|---|
| No-SSN-led (M-NOSSN-*) | 64.3% no-SSN population | **High** (≈30% of M-* budget) |
| Name-coupling (M-NAME-*) | All populations; teaches cross-field token shuffling | **High** (≈20%) |
| SSN-led (M-SSN-*) | 21.4% full-SSN population, but high-signal | **High** (≈15%) — fewer per-scenario, more scenarios |
| Last-4-led (M-L4-*) | 14.3% backup-only band | Medium (≈8%) |
| Address / DOB / Phone / Email / Sex / Pediatric drift | Mixed | Medium (≈20% combined) |
| Mixed / heavy-drift (M-MIX-*) | Realistic worst-case | Medium (≈7%) |

### 8.1 Match scenarios (label = 1)

#### No-SSN-led — teach: "without SSN, name-token-union + DOB + corroborating field is the match signal"

These dominate the real population. The model must comfortably match on name+DOB-led signals alone.

- **M-NOSSN-01 — No-SSN control.** Both sides have identical first/middle/last/DOB/sex/address, no SSN, no last-4. Sanity-check positive for the no-SSN path.
- **M-NOSSN-02 — No-SSN, address moved.** Same name/DOB/sex; different address (intra-city or cross-city); both sides have no SSN.
- **M-NOSSN-03 — No-SSN, name-token-stable + phone overlap.** Same name+DOB, no SSN, address differs, phone sets overlap by ≥1.
- **M-NOSSN-04 — Thin no-SSN record.** Both sides only have first+last+DOB+sex (no address, no phone, no email, no SSN). Agree on those. Sparse but matchable — this is FQHC reality for transient patients.
- **M-NOSSN-05 — No-SSN with name corruption.** No SSN on either side; DOB matches; one side has a single-character typo on first or last name.
- **M-NOSSN-06 — No-SSN, DOB clerical drift.** No SSN on either side; name and address agree exactly; DOB off-by-one day or year. Tests the model's tolerance for DOB noise when the *only* anchor is name+address. *Borderline; oversample modestly and verify policy with stakeholders.*

#### SSN-led — strong-signal subset; teach: "valid full-SSN equality ⇒ match (other fields may drift freely)"

- **M-SSN-01 — Identical record (control).** All 11 model fields agree. Sanity-check positive. Small budget; mostly a calibration anchor.
- **M-SSN-02 — SSN match, name typos.** Same SSN/DOB. Names corrupted by 1–2 single-character typos on one side.
- **M-SSN-03 — SSN match, missing middle.** Same SSN/DOB. One side has `MiddleNM_clean=null`, other has middle present.
- **M-SSN-04 — Maiden ↔ married surname.** Same SSN/DOB/first/middle. Last name on side B is a completely different last name drawn from top-N. *Critical for SSN-trumping behavior.*
- **M-SSN-05 — Moved (different street).** Same SSN/DOB/name. Side B has entirely different address (same city) or different city.
- **M-SSN-06 — Moved out of state.** Same SSN/DOB/name. Side B in different state + ZIP.
- **M-SSN-07 — Different phone, different email.** Same SSN/DOB/name/address. Side B has new phone set + new email.
- **M-SSN-08 — DOB clerical drift.** Same SSN/name. DOB on B has off-by-one day or year. Tests "SSN trumps minor DOB disagreement".
- **M-SSN-09 — Full SSN ↔ last_4 only.** Side A has full 9-digit SSN; side B has only `last_4_SSN` (full is null). last_4 agrees. Name + DOB agree.
- **M-SSN-10 — SSN ↔ no SSN at all.** Side A has full SSN; side B has neither. Name + DOB + address agree. Verifies model doesn't *require* SSN to match.
- **M-SSN-11 — Heavy drift, SSN anchors.** Same SSN. Name has typo + missing middle, DOB has off-by-one, address moved, phone/email changed. Realistic worst-case match.

#### Last-4-led — teach: "last_4_SSN + DOB + name is decisive when full SSN is absent on both sides"

This bucket targets the 14.3% backup-only band where neither side has a full SSN but both have `last_4_SSN`. The naive `last_4` collision rate is 1/10,000 by chance, but conditioned on a name+DOB match it is near-decisive.

- **M-L4-01 — Last-4 + name + DOB match, no full SSN on either side.** The control case for the backup-only band.
- **M-L4-02 — Last-4 + DOB match, name has typo.** Name has a single-character typo on first or last; last-4 + DOB agree exactly. Match.
- **M-L4-03 — Last-4 + name match, DOB off-by-one.** Last-4 + name agree; DOB differs by one day/year. Match (last-4 + name is strong; small DOB drift is plausible clerical noise).
- **M-L4-04 — Asymmetric: full SSN one side, last-4 only the other.** Side A has 9-digit `SSN_clean`; side B has only `last_4_SSN` (full null). `SSN_clean[-4:] == last_4_SSN` on B. Name + DOB agree. *(Pairs that look like this are common when one site collects full SSN and another only collects the last four.)*
- **M-L4-05 — Asymmetric with name drift.** As M-L4-04 plus one name corruption. Last-4 still anchors.

#### Name-coupling-led — teach: "name field assignment is noise; tokens are signal"

- **M-NAME-01 — Hyphenation variant.** `ANNE-MARIE` ↔ `ANNE MARIE` ↔ `ANNEMARIE` across the first-name slot. Same person.
- **M-NAME-02 — First/middle swap.** Side A: `(FirstNM=MARIA, MiddleNM=CARMEN)`; side B: `(FirstNM=CARMEN, MiddleNM=MARIA)`. Clerical reassignment.
- **M-NAME-03 — Two-surname shuffle (Hispanic).** Side A: `(MiddleNM=GARCIA, LastNM=LOPEZ)`; side B: `(MiddleNM=null, LastNM=GARCIA LOPEZ)`. Common when Hispanic patient registers at different clinics.
- **M-NAME-04 — Two-surname collapse to one.** Side A: `LastNM=GARCIA LOPEZ`; side B: `LastNM=LOPEZ` (or `GARCIA`). Same person dropped a surname.
- **M-NAME-05 — Vietnamese name-order swap.** Side A: `(FirstNM=NGUYEN, LastNM=THI MAI)`; side B: `(FirstNM=THI MAI, LastNM=NGUYEN)`. Native vs US order.
- **M-NAME-06 — Middle-initial only.** Side A: `MiddleNM=ANNE`; side B: `MiddleNM=A`. Same person.
- **M-NAME-07 — Compound first dropped.** Side A: `FirstNM=MARIA CARMEN`; side B: `FirstNM=MARIA`. Same person, lazy entry.
- **M-NAME-08 — Generational suffix appears/absent.** Side A: `LastNM=SMITH JR` (post-cleaning collapse); side B: `LastNM=SMITH, SuffixNM=JR`. Or one side has no suffix at all.
- **M-NAME-08b — Suffix in wrong slot.** Real data has 45 records with `JR` in `MiddleNM_clean` while `SuffixNM_clean` is null (the LastNM-trailing-` JR`/`SR` form is removed by the cleaning step at the source, so it never appears post-clean). Side A: `MiddleNM=JR` (or `SR`/`III`); side B: `SuffixNM=JR` with middle clean (or null). Same SSN/DOB/first/last token. Same person; clerical-slot misplacement that survived cleaning.
- **M-NAME-09 — Nickname.** `ROBERT↔BOB`, `WILLIAM↔BILL`, `ELIZABETH↔BETH/LIZ`, `MICHAEL↔MIKE`, `JOSEPH↔JOE`, `JOSE↔PEPE` (Spanish), `FRANCISCO↔PACO`. Limited curated mapping table — same person.
- **M-NAME-10 — Typo: single-char substitute.** `ROCA` ↔ `ROCQ`. Same SSN/DOB.
- **M-NAME-11 — Typo: transposition.** `SMITH` ↔ `SMIHT`. Same SSN/DOB.
- **M-NAME-12 — Typo: insertion/deletion.** `JOHNSON` ↔ `JOHNSOON` / `JONSON`. Same SSN/DOB.
- **M-NAME-13 — Diacritical loss (already cleaned).** Pre-cleaning, side A was `MUÑOZ` and B was `MUNOZ`; both post-clean to `MUNOZ`. Included as a control: should appear identical at our level. (Tests our cleaning fidelity, not the model.)

#### DOB-led

> **Calibration note (§5.8):** in real same-person pairs DOB matches exactly ~87% of the time, and when it differs it is almost always a *substantial* gap, not a tidy off-by-one (off-by-one-day/year and month-day-transpose are each <0.1%). M-DOB-02/03/04 below are therefore **teaching devices for tolerance**, not realistic frequencies — oversample them in the fine-tune corpus but keep them near their true (tiny) rate in the realistic-eval.

- **M-DOB-01 — Identical DOB, otherwise typical.** Control.
- **M-DOB-02 — Month-day transposition.** `1985-01-15 ↔ 1985-10-15` (only valid when day ≤ 12). Same SSN/name.
- **M-DOB-03 — Off-by-one year.** `1985-04-12 ↔ 1986-04-12`. Same SSN/name.
- **M-DOB-04 — Off-by-one day.** Same SSN/name.
- **M-DOB-05 — DOB null on one side.** Same SSN/name; side B `BirthDT_clean=null`.

#### Address-led (still same entity)

- **M-ADDR-01 — Apartment added/removed.** Same building. `AddressLine2_clean` toggles between null and `APT 4B`.
- **M-ADDR-02 — Apartment changed.** Same street; apt differs (moved within building).
- **M-ADDR-03 — Address-line2 absorbed into line1.** Side A: `123 MAIN ST` + `APT 5`; side B: `123 MAIN ST APT 5` + null line2. Same address.
- **M-ADDR-04 — House-number typo.** `12345 MAIN ST ↔ 12354 MAIN ST`. Same SSN.
- **M-ADDR-05 — Street-suffix variants.** Pre-cleaning `MAIN STREET` vs `MAIN ST`; both should normalize to `MAIN ST`. Control case.

#### Phone / email / sex drift

- **M-PHONE-01 — Phone overlap, partial.** Side A has 2 phones, side B has 2 phones; 1 overlaps. Same person.
- **M-PHONE-02 — Phones disjoint, new number.** Same SSN/name/DOB; phones entirely different.
- **M-EMAIL-01 — Email changed.** New email, otherwise same.
- **M-EMAIL-02 — One-char domain typo.** Real-data typos observed: `gmail.com ↔ gamil.com`, `gmail.com ↔ gmai.com`, `parkwestmed.com ↔ parkwestmed.ez`. Local part identical, domain has a one-character insertion / deletion / substitution / transposition. Otherwise identical record. Teach the model to treat this as the same email.
- **M-SEX-01 — Sex disagreement (clerical).** Same SSN/DOB/name/address; sex flipped on one side. Rare in real data but it happens; label remains match because SSN is decisive.
- **M-SEX-02 — OTHER vs male/female.** Real data has 374 records (`OTHER`, 0.3%). Side A `SexAtBirthDSC=OTHER`, side B `MALE` or `FEMALE`. Same SSN/name/DOB. Match. Specifically teaches the model that the `OTHER` value exists and is a valid pairing across sites.

#### Pediatric — teach: "constrained-identifier pediatric records still match on the few fields they have"

DOB years 2010s+2020s account for ~12% of records. Pediatric patients (especially under 5) have a constrained identifier set: typically no SSN, no email of their own, phone is the parent's, address is the parent's. Treat as its own bucket because the *missingness shape* is different from adults.

- **M-PED-01 — Pediatric thin match.** Both sides: same first+middle+last, same DOB (in 2010s or 2020s), same sex, same parent address; no SSN, no last-4, no email. Phone may be present (parent's) and overlap. Match.
- **M-PED-02 — Pediatric with last-4.** As M-PED-01 plus `last_4_SSN` set and equal on both sides. Match.
- **M-PED-03 — Pediatric name-token drift.** Pediatric pair (DOB 2010s+). Same DOB+address+sex; name has 1 token-shuffle (e.g., MiddleNM promoted to FirstNM compound, or one side has only FirstNM+LastNM with MiddleNM null). Same person. Match.
- *(Newborn/placeholder scenarios — `BABY GIRL <LASTNM>` and similar — are excluded: they are flagged `valid_record=False` by the cleaning rules in `docs/Data-Cleaning-Guide.md` and never reach the model at inference time. If FQHCs need to resolve newborn-placeholder pairs, that has to be addressed upstream in cleaning, not by the model.)*

#### Mixed / realistic

- **M-MIX-01 — Two corruptions.** Random pair of one name + one address corruption applied. Bulk of the entity-first match population.
- **M-MIX-02 — Three corruptions.** Heavier drift; SSN or last_4 still anchors.
- **M-MIX-03 — Thin records.** Both sides have only first+last+DOB+sex. Agree on all. Sparse but matchable.

### 8.2 Non-match scenarios (label = 0)

#### Easy negatives

- **NM-EASY-01 — Fully random.** Two entities, no shared field beyond ambient base rates.
- **NM-EASY-02 — Same state only.** Both IL, otherwise unrelated.

#### Hard household negatives

- **NM-HH-TWIN — Twin (same DOB, same address, same last name).** Different first name, different SSN, same sex or different sex.
- **NM-HH-TRIPLET-LIKE — Same DOB + same address but different last name.** Cohabiting unrelated friends born same day. Edge case.
- **NM-HH-JR-SR — Jr / Sr same household.** Side A: `FirstNM=ROBERT`, `LastNM=SMITH`, `SuffixNM=null`, parent DOB. Side B: `FirstNM=ROBERT`, `LastNM=SMITH`, `SuffixNM=JR`, child DOB. Same address. Different SSN. **Hardest non-match category.**
- **NM-HH-SIBLING — Siblings.** Same `LastNM_clean`, same address, similar DOBs (within ±10 years), different first name, different SSN.
- **NM-HH-PARENT-CHILD — Parent / child.** Same `LastNM_clean`, same address, different first name, DOBs differ by 15–40 years.
- **NM-HH-SPOUSE — Spouses / partners.** Same address, overlapping phones, different last name, different DOB, different SSN.
- **NM-HH-ROOMMATE — Roommates.** Same address only; nothing else in common.

#### Common-name collisions (Chicago is full of these)

- **NM-COMMON-01 — Same name, same city, different DOB.** `(JOHN SMITH, CHICAGO)` × 2. Different SSN, different DOB by years, different address.
- **NM-COMMON-02 — Same name, same ZIP.** Tighter geo but still different person.
- **NM-COMMON-03 — Hispanic surname collision.** Top-N two-surname pairs are not unique. `(MARIA GARCIA LOPEZ)` × 2, different DOB/SSN.
- **NM-COMMON-04 — Top-ZIP common name collision.** Exploits the Chicago ZIP concentration (`60639`/`60625`/`60640`/`60651`/`60647` each hold 4–7k records). Same `FirstNM_clean` + same `LastNM_clean` + same `ZipCD_clean_base` (one of the top Chicago ZIPs). Different DOB (by ≥5 years) + different SSN + different street. Far more common in production than the looser NM-COMMON-02 — and the model needs to weight ZIP correctly (a populous ZIP is a weak signal).
- **NM-COMMON-05 — Same name + same Chicago area code.** Same first+last, both have a `773`/`312`/`708` primary phone but the *full number* differs. Different DOB + SSN. Teach: area-code overlap is not phone overlap.

#### SSN-related hard negatives (critical: don't false-match on shared data-quality artifacts)

- **NM-SSN-01 — Last-4 collision.** Side A and B have the same `last_4_SSN` (1/10000 by chance), full SSN missing on both. Different name, different DOB. **Teach: last-4 alone isn't enough.**
- **NM-SSN-02 — Last-4 collision + same first letter of name.** Slightly harder version of the above.
- **NM-SSN-03 — Single-digit SSN-typo collision.** Side A's clerk typed an extra '0' into the SSN, which happens to equal side B's real SSN. Different name/DOB. Tests: don't blindly trust SSN equality if name+DOB are wildly different. (Counterweight to M-SSN-* — model must learn SSN is *strong*, not infallible.)
- **NM-SSN-04 — Same SSN, opposite sex, wildly different DOB.** Likely data-entry error in SSN on one side. Different person. Label = 0. Rare; include modest count.
- **NM-SSN-05 — Full SSN vs *mismatching* last-4.** Side A has full 9-digit `SSN_clean`; side B has only `last_4_SSN`, and `SSN_clean[-4:] != last_4_SSN`. Name + DOB also disagree. **Teach the SSN↔last-4 coupling in the negative direction**: a last-4 that contradicts the other side's full SSN *kills* the SSN signal rather than being ignored. Counterpart to the positive M-L4-04 (full SSN ↔ agreeing last-4 ⇒ match); without this the model only ever learns last-4 *agreement* helps, never that last-4 *disagreement* against a full SSN hurts.

#### Identity-fragment overlap

- **NM-IDF-01 — Shared shelter / group-home address.** Same `AddressLine1_clean`, same city/state/ZIP; different name + DOB + SSN.
- **NM-IDF-02 — Shared family phone.** Same primary phone, different name + DOB + SSN.
- **NM-IDF-03 — Shared family email.** `family@gmail.com` style — different name + DOB.
- **NM-IDF-04 — Shared address + shared phone.** Two unrelated tenants who use one landline. Stronger lure, still NM.

#### Pediatric non-matches

- **NM-PED-01 — Pediatric siblings.** Both DOBs in 2010s+/2020s, within ±5 years of each other. Same `LastNM_clean`, same parent address, same primary phone. Different first name + different SSN/last-4 (when present). Different person — hardest pediatric NM.
- **NM-PED-02 — Pediatric same-DOB unrelated.** Two unrelated children with the same DOB (same daycare, same school) at the same address (multi-family building). Different last name. Different SSN/last-4. Different person.

#### Boundary cases

- **NM-BND-01 — Thin records, disagree on name.** Both sides only have first+last+DOB+sex. First or last name differs by more than a typo. Different person. Tests: don't fall back to "almost-empty = match".
- **NM-BND-02 — Thin records, disagree on DOB.** Both sides only have first+last+DOB+sex. Names agree exactly (common-name pair); DOB differs by years. Different person. Counter-balance to M-NOSSN-04.

### 8.3 Edge / policy cases (label requires explicit decision)

These are scenarios where the *correct* label is ambiguous or depends on the operator's policy. Decisions logged here; cases included only if and how the policy says.

- **POL-AMBIG-01 — Twins with same SSN entered.** Same DOB, same address, same SSN, different first name. In real data this happens when a clerk reuses one twin's SSN. We **exclude** from training (we cannot teach an unambiguous label).
- **POL-AMBIG-02 — Same SSN + same DOB + completely different name + different sex.** Could be a data-entry error in SSN, could be a real person we've conflated. We label as **match=1** for the fine-tune corpus (SSN-trumping policy is the goal of the project) but record this decision explicitly so we can audit later.
- **POL-AMBIG-03 — Same name + same DOB + same address + SSN missing on both.** Could be a household with an unusual coincidence or could be one person registered twice. Default label: **match=1**, with a low oversample budget. Revisit with stakeholders.

> **Policy register.** Every POL-* decision must be approved before generation. Default to excluding any case the team hasn't reviewed.

### 8.4 Additional hard cases (v0.5)

**Difficulty contract (v0.5).** Every named scenario below and in §8.1–§8.2 is filled to a **per-scenario quota** by the enumerated overlay, so the corpus contains **many pairs of each** (hundreds at the 40k scale), not one-offs. On top of that, the bulk supplies the realistic volume. Two global rules now govern *all* positives: (1) **no identical pairs** — every positive has ≥1 corruption; (2) **SSN-band quota** — ≤5% have matching full SSN on both sides (easy), ~15% match on last-4 only (full SSN absent), and **~80% have no usable SSN match** (at least one of full-SSN / last-4 missing or non-matching), forcing resolution on name+DOB+address+phone. Address line1 matches in only ~12–15% of positives (high-mobility FQHC population; cf. §5.8 line1-exact 28.8% measured on the SSN-having stable subpopulation).

**Additional MATCH hard cases (added v0.5):**
- **M-NAME-13 — First name to initial.** `JOHN` ↔ `J` (clerk recorded an initial). Counterpart to the middle-initial case.
- **M-NAME-14 — Name truncation.** Long name cut by a field-length limit (`MASSIMILIANO` → `MASSIMILIAN`).
- **M-NAME-15 — Cross-language given-name variant.** `GUILLERMO` ↔ `WILLIAM`, `JOSE` ↔ `JOSEPH`, `JESUS` ↔ `JESSE` (bilingual records); seeded via the nickname/equivalence pool.
- **M-NAME-16 — Double-space / concatenation.** `DE LA CRUZ` ↔ `DELACRUZ`, `MARY ANN` ↔ `MARYANN`.
- **M-ADDR-05 — Move within ZIP.** New street, same ZIP/city (very common churn).
- **M-ADDR-06 — Directional / abbreviation drift.** `123 N MAIN ST` ↔ `123 NORTH MAIN STREET`.
- **M-ZIP-01 — ZIP-only drift.** Same street, ZIP changed (data-entry or boundary correction).

**Additional NON-MATCH hard cases (added v0.5):**
- **NM-COMMON-06 — Same name + adjacent DOB.** Same first+last, DOB off by 1–3 days/years (looks like a DOB typo but is a different person).
- **NM-HH-COUSIN — Cousins.** Same surname, same city/ZIP, different DOB and address.
- **NM-SSN-06 — Same last-4 + same DOB, different name.** The collision-heavy `by_last4_dob` cluster (§5.7) — shared backup-SSN fragment is *not* identity.

**Test set mirrors train (v0.5).** `synthetic_test` uses the **identical difficulty construction** as `synthetic_train` — the same SSN bands, the same enumerated hard-scenario coverage (so every named case above is *measured*, not just trained), and the same hard (key-sharing) negatives — drawn from **held-out entities** and at **`--test-prevalence` (default 1:4)**. The only differences from train are prevalence, the all-hard negatives (no easy anchor), and the disjoint entity pool. This makes the test both realistic (production-like prevalence + hard negatives) and diagnostic (per-`case_type` recall is readable because each scenario is present).

**Default-exclude flag (locked 2026-05-28).** The first generation run gates stakeholder-sensitive scenarios behind `--include-policy-cases` (default **off**):

- **Excluded by default** (flip the flag to include): **M-NOSSN-06** (DOB drift on a name-only anchor — borderline label) and **M-SEX-01** (sex disagreement, SSN-anchored — possibly ambiguous).
- **Always excluded:** **POL-AMBIG-01** (twins sharing one SSN — no unambiguous teachable label), regardless of the flag.
- **Included by default but tagged:** **POL-AMBIG-02** (same SSN, different name/sex → match=1) and **POL-AMBIG-03** (same name+DOB+address, no SSN → match=1). SSN-trumping and household-coincidence-as-match are explicit project goals, so these ship in the corpus; their `case_type` tag lets us audit or pull them after stakeholder review.

## 9. Pair assembly (hybrid, v0.5)

One entity universe (§6), one corruption engine (§7). The universe is partitioned **entity-disjoint**
into a train pool and a test pool (§10) up front; train pairs are built only from train entities and
test pairs only from test entities. Both files are deterministic from `--seed`.

**Core rule (applies everywhere).** Field *presence* is always drawn from the real missingness joint
(`missingness_patterns.top_patterns`, §5) and is **never** forced present — except for the single field a
hard scenario is intrinsically about, and those scenarios are a budgeted minority. Field
*agreement/corruption* is realistic and multi-field by default: every pair, positive **and** negative,
passes through the §7 corruption engine so each side is independently messy, and a scenario only *adds*
its one defining transformation on top. This is the fix for the old corpus where forced `ensure_*` /
`force_full_ssn` calls and single-transformation scenarios made records too clean and field-rich.

**Two pair sources, mixed in both files:**
- **Bulk (entity-first, ~75–85%).** Sample entities at realistic K / missingness / geo (§6).
  *Positives* = within-entity variant pairs from the §7 engine (independent per-field corruptions at the
  §5.8 marginals → naturally 2–3 simultaneous differences). *Negatives* = cross-entity pairs that
  **share ≥1 strong blocking field** (surname / DOB / address / phone / last-4), each side independently
  corrupted — i.e. blocking-survivor-like, never random strangers.
- **Overlay (hard scenarios, ~15–25%).** The §8 catalog, refactored: start from a naturally-missing,
  already-corrupted variant pair; apply only the scenario's defining transformation; `ensure_<field>`
  only for that scenario's own field. Guarantees rare structural cases (name-order swaps, suffix-slot,
  Jr/Sr, last-4 collisions, full-vs-last-4 SSN asymmetry) appear at a learnable count.

**Dirty tail (§7).** A budgeted fraction of positives (~15–25%) get a shared **messiness multiplier**
that *correlates* corruptions so name + address + phone + DOB drift together (5+ differing fields),
plus thin `N/A`-heavy records — the "everything is a mess at once" tail the marginals alone miss.

### 9.1 Training set — `synthetic_train_vN.csv`

Balanced **~1:1.5** match:non-match, hybrid bulk + overlay, drawn from the **train** entity partition.
Scale is a parameter (`--train-pairs`, default ~40k); generation is cheap and deterministic, so the
workflow stays **generate → fine-tune → read per-`case_type` test metrics → re-weight weak buckets**.
The §8 bucket weights (No-SSN, Name-coupling, SSN, Last-4, drift, Mix for matches; Easy, Household,
Common-name, SSN, Identity-fragment, Pediatric, Boundary for non-matches) govern the **overlay** slice;
the bulk supplies the realistic backbone. Easy random negatives are a small minority — the non-match
budget is dominated by hard, key-sharing negatives (the precise failure modes of an SSN-trumping model).

### 9.2 Test set — `synthetic_test_vN.csv` (honest evaluator)

Test prevalence (`--test-prevalence`, default 0.2 → 1:4), drawn from the **held-out** entity
partition, **never used to train**. Negatives are **blocking-survivor-like only**: every non-match shares
≥1 strong field (QA asserts this), so precision is measured against genuine near-duplicates rather than
random strangers. Positives carry the same realistic missingness + multi-corruption + dirty tail as
train. Each pair keeps its `case_type` tag for per-bucket metrics.

## 10. Train / test split

- **Entity-disjoint.** A single entity universe is partitioned by a deterministic per-entity hash
  (`test_entities`, seeded from `--seed`); no person appears in both files. Default hold-out fraction 15%.
- Because train and test draw from disjoint entity pools, the test set measures generalization, not memorization.
- Deterministic and byte-reproducible from `--seed`.

## 11. Output format

Two pair-level CSVs under `data/synthetic/`, `vN`-versioned:

```
data/synthetic/synthetic_train_vN.csv    # balanced ~1:1.5 hybrid corpus, train entity pool   (pair-level)
data/synthetic/synthetic_test_vN.csv     # realistic prevalence, hard negatives, held-out pool (pair-level)
```

Both share the **identical column layout** (unchanged from prior versions): `PATID_A`/`PATID_B`, the
model-input `_l`/`_r` columns mirroring the MDM-cleaned schema (consumed by `df_serializer` after the
`utils/alliance_schema` rename → friendly mode4 attribute names), then provenance columns
`label`, `case_type`, `corruptions_applied`, `entity_id_a`/`entity_id_b` (which `df_serializer` skips).
No separate manifest or record-level/blocking files are emitted in v0.5.

Pair-CSV column layout. Every column intended for the model has `_l`/`_r` suffix and mirrors the MDM-cleaned schema so the same `FEATURE_RENAMES + derive` pipeline runs. Provenance / bookkeeping columns use other naming to stay out of `df_serializer`:

```
# Provenance (capital _A/_B — not consumed by df_serializer)
PATID_A, PATID_B,

# Model-input columns (lowercase _l/_r — consumed by df_serializer after FEATURE_RENAMES)
FirstNM_clean_l,  FirstNM_clean_r,
MiddleNM_clean_l, MiddleNM_clean_r,
LastNM_clean_l,   LastNM_clean_r,
SuffixNM_clean_l, SuffixNM_clean_r,
BirthDT_clean_l,  BirthDT_clean_r,
SSN_clean_l,      SSN_clean_r,
last_4_SSN_l,     last_4_SSN_r,
AddressLine1_clean_l, AddressLine1_clean_r,
AddressLine2_clean_l, AddressLine2_clean_r,
CityNM_clean_l,   CityNM_clean_r,
StateCD_clean_l,  StateCD_clean_r,
ZipCD_clean_base_l, ZipCD_clean_base_r,
PrimaryPhoneNBR_clean_l, PrimaryPhoneNBR_clean_r,
Phone01NBR_clean_l,      Phone01NBR_clean_r,
Phone02NBR_clean_l,      Phone02NBR_clean_r,
Email_clean_l,    Email_clean_r,
SexAtBirthDSC_clean_l, SexAtBirthDSC_clean_r,
full_name_tokens_l, full_name_tokens_r,       # derived; kept for blocking/QA, may be excluded via FEATURE_RENAMES
full_name_compact_l, full_name_compact_r,     # derived; kept for blocking/QA
Phones_set_l, Phones_set_r,                   # derived; this is the one the model actually reads as 'phone'
Address_normalized_l, Address_normalized_r,   # derived; kept for QA

# Label + provenance (no _l/_r — not consumed by df_serializer)
label,
case_type,                # e.g. "M-SSN-04"
corruptions_applied,      # JSON list, e.g. ["replace_last","address_move"]
entity_id_a, entity_id_b  # ground-truth cluster ids (§4 principle 9); drive the §10 split
```

**Deliberately omitted from the pair CSV:** `PATID_l/_r` (would enter prompt as `patid: <UUID>`), `valid_record_l/_r` (always True by §4 principle 2; pure waste of tokens), `ZipCD_clean_ext_l/_r` (93.9% null per §5.1; excluded from §2 model schema). `case_type`, `corruptions_applied`, and `entity_id_a/_b` are dropped before passing to `loo.py` / `predict_alliance.py` — the model reads only `*_l`, `*_r`, `label`. (`entity_id_a/_b` end in `_a`/`_b`, not `_l`/`_r`, so `df_serializer` skips them even if left in.) `corruptions_applied` is emitted as a JSON-encoded string for clean CSV round-tripping.

## 12. Sanity checks (run after generation)

Assertions the generator must satisfy and a notebook verifies:

1. **Schema parity.** Both `synthetic_train_vN.csv` and `synthetic_test_vN.csv` carry the identical pair-CSV column layout (§11); model columns match the `MDM_Population_cleaned_v1.csv` schema after `_l`/`_r` un-suffixing, and every `_l`/`_r` column maps to a friendly mode4 attribute via `utils/alliance_schema.prep_paired_df` (no column falls outside the schema).
2. **Value conventions.** All names uppercase ASCII; no diacritics; ZIPs are 5-digit strings; SSNs are 9-digit strings with no junk patterns.
3. **`valid_record=True`** on every emitted record.
4. **Label / case agreement.** Every `case_type` in `M-*` has `label=1`; every `NM-*` has `label=0`.
5. **Entity disjointness.** No `entity_id` appears in both `synthetic_train` and `synthetic_test`.
6. **No PHI leakage.** No row matches anything in real `MDM_Population_cleaned_v1.csv` (a hash-set check on a few high-cardinality fields). The point of synthesis is that real persons cannot be recovered.
7. **Realistic missingness (v0.5 — the key fix).** Per-field **presence**, measured by stacking both `_l` and `_r` sides of every pair, falls within **±3** absolute percentage points of the measured §5 marginals for each field below. This is the check the old corpus failed (forced `ensure_*` over-populated SSN/middle/email).
8. **Multi-corruption profile (v0.5 — the other key fix).** Among positives, the distribution of *differing-field count* must have a real heavy tail — mean ≈ 2–3 differing fields, **not** a 70% one-corruption spike — with a non-trivial ≥5-field "dirty tail" share. Validate against the §5.8 within-cluster differ profile.
9. **Hard negatives only in test.** **100%** of `synthetic_test` non-match pairs share ≥1 strong blocking field (surname / DOB / address / phone / last-4). No random-stranger negatives in the test set.
10. **Distribution sanity vs `synthetic_data_stats.json`.** Per-record marginals (stacked sides) fall within ±3 absolute percentage points of the measured §5 values for each of:
    - `FirstNM_clean` / `LastNM_clean` / `BirthDT_clean` presence (~99% expected).
    - `MiddleNM_clean` presence (~19%) and pct-single-initial (~97% of present).
    - `SSN_clean` presence (~21%), `last_4_SSN` presence (~36%), `no_full_ssn_but_last4_present_pct` (~14%).
    - `AddressLine1_clean` presence (~96%), `AddressLine2_clean` presence (~29%).
    - Phones-per-record histogram (0/1/2/3/4: 5.6/12.4/55.6/25.2/1.2%).
    - `Email_clean` presence (~31%); gmail dominance (~66% of present).
    - State distribution (IL ~79%, KY ~5%, HI ~3%, NY ~3%, …).
    - Sex among non-null (F ~56%, M ~44%, OTHER ~0.3%).
    - DOB decade histogram (year-2000+ rate ~27%).
8. **Token-set invariance.** For M-NAME-* scenarios where the recipe intends `full_name_tokens` to be equal on both sides (most of them), assert that.
9. **ZIP3 ↔ State consistency.** Every emitted record satisfies the ZIP3 → State mapping (per §4 principle 12).
10. **SSN structural validity.** Every non-null `SSN_clean` satisfies the cleaning-guide rules (area ∉ {`000`, `666`, `900-999`}, group ≠ `00`, serial ≠ `0000`). Every non-null `last_4_SSN` is 4 digits and ≠ `0000`.
11. **SSN ↔ last-4 coupling.** When both `SSN_clean` and `last_4_SSN` are present on a record, `last_4_SSN == SSN_clean[-4:]` (no drift between the two presentations).
12. **NANP-valid phones.** Every emitted phone is a 10-digit string with area code in the measured top-N pool (or in the long-tail valid set), NXX in 200–999 and not an `N11` (211, 311, …, 911).
13. **No cleaning-filtered tokens.** No emitted name (first/middle/last/suffix) or address field contains any string from the `docs/Data-Cleaning-Guide.md` invalid-strings lists (`BABY BOY`, `BABY GIRL`, `DUPLICATE`, `DO NOT USE`, `DON'T USE`, `MEDICARE`, `DOUBLE ACCOUNT`, `DUPLICATE ACCOUNT`, `ACCOUNT`, `<MRG>`, `TEST`, `BABY`, `HOMELESS`, `TRANSIENT`, etc.). Same for the standard text-null tokens (`UNKNOWN`, `NULL`, `NAN`, `NONE`, `N/A`, `NA`). Junk-equality-as-match-signal is therefore a *non-issue at inference time* — no need to teach it.

## 13. Open questions / TODO

**Resolved by stats run + v0.2 spec edits (2026-05-27):**
- [x] Run `extract_mdm_stats.py` and commit `synthetic_data_generation/synthetic_data_stats.json`.
- [x] Replace per-field missingness placeholders in §2 and §5.
- [x] Confirm `p_compound_first` (0.02) and `p_two_surname` (0.07) from the MiddleNM/LastNM token stats — both lower than initially hand-guessed.
- [x] Rebalance §8 around no-SSN-at-all being the dominant population (64.3%). Added No-SSN-led bucket (M-NOSSN-01..06) as the lead, with a scenario-weight table in the §8 preamble.
- [x] Add Last-4-led bucket (M-L4-01..05) for the 14.3% backup-only band.
- [x] Add suffix-in-wrong-slot sibling (M-NAME-08b) — captures the 45 `JR`-in-`MiddleNM` reality.
- [x] Add M-EMAIL-02 (one-char domain typo) — covers real-data `gamil.com`/`gmai.com` typos.
- [x] Add M-SEX-02 for `OTHER` sex value.
- [x] Add NM-COMMON-04 (Chicago top-ZIP name collision) and NM-COMMON-05 (area-code overlap NM).
- [x] Add pediatric bucket (M-PED-01..03) and pediatric NM (NM-PED-01..02) for the 12% DOB-2010s+ population.
- [x] Extend §7 corruption pool with: middle-to-initial / initial-to-full, suffix-into-wrong-slot, area-code-preserving phone change, one-char email-domain typo, set-to-`OTHER` sex.
- [x] Tighten §4 design principles (no-SSN normal path, pediatric isolation, ZIP3↔state hardness, `valid_record=True` generator guard).
- [x] Sharpen §12 sanity checks with concrete §5 distribution targets.
- [x] **Reconciliation against `docs/Data-Cleaning-Guide.md`**: removed scenarios whose inputs would have been filtered out at cleaning time — M-NAME-14 (junk middle), M-PED-04 (newborn placeholder `BABY GIRL …`), NM-DQ-01 (shared junk middle), NM-DQ-02 (shared `LastNM=ID`). The model never sees these because `valid_record=False`.

**Flagged upstream to the cleaning team (not the model's job):**
- [x] **`LastNM_clean=ID` (519 records) will be invalidated upstream.** Decision (2026-05-27): the cleaning team will add `ID` to the LastNM invalid-strings list in `docs/Data-Cleaning-Guide.md`, so those records will be `valid_record=False` and never reach the model. No synthetic scenario needed.
- [x] **Newborn-placeholder pairs (`BABY GIRL <LASTNM>` etc.) are out of scope.** These are real newborns registered before being named, but the cleaning rule in `docs/Data-Cleaning-Guide.md` treats `BABY GIRL`/`BABY BOY` as `valid_record=False`. Decision (2026-05-27): keep the cleaning rule as-is; placeholder→resolved-name linkage is not AnyMatch's job. No synthetic scenario; no spec change.

**Resolved by design decisions (2026-05-28):**
- [x] **Input schema locked** (§2): three name fields passed raw; `AddressLine2` included.
- [x] **Scale locked** (§9): 40,000-pair fine-tune corpus at 1:1.5 + 10,000-pair realistic-eval at 1:9, with concrete per-bucket / per-scenario budgets. Treated as a first cut to be re-weighted after the first fine-tune run.
- [x] **Record-level blocking-eval file added** (§11.1): `blocking_eval_v1.csv` — realistic-distribution individual records (not pairs) with full cleaning-output schema parity (incl. `_raw` copied from `_clean`) plus an `entity_id` ground-truth column, for testing the blocking strategy end-to-end on synthetic data.
- [x] **`N_scenario` budget set** (§9.1 tables) from the §8 bucket weights.
- [x] **Entity hold-out locked** at 15% (§10).
- [x] **Generation approach locked** (§14): code-primary generator + local LLM (Ollama) seeding static vocabulary pools only.
- [x] **Nickname + initial-expansion tables**: approach locked — LLM-seeded then hand-verified, committed under `synthetic_data_generation/pools/` (§14.4). Contents finalized during the build.
- [x] **Stakeholder-gated scenarios** routed behind `--include-policy-cases` (§8.3): M-NOSSN-06 / M-SEX-01 excluded by default; POL-AMBIG-01 always excluded; POL-AMBIG-02/03 included-but-tagged.

**Resolved by review pass (2026-05-28, v0.4):**
- [x] **Calibrate corruption budgets from real data.** Extended `extract_mdm_stats.py` with `within_cluster_agreement` (true-positive proxy over SSN clusters), `geo_joint`, and `missingness_patterns`. Script re-run 2026-05-28; §5.8/§5.9 filled and §7 recalibrated from §5.8.
- [x] **Correlated entity sampling** (§6): geo as a joint `(City,State,ZIP3)` draw, missingness as a joint pattern draw, first name conditioned on DOB year, pediatric identifier coupling.
- [x] **Deterministic set serialization** (§3): `Phones_set` / `full_name_tokens` emitted sorted.
- [x] **NM-SSN-05** added (§8.2): full SSN vs mismatching last-4 ⇒ non-match (negative side of the SSN↔last-4 coupling).
- [x] **Name pools from public reference data** (§14): Census surnames + SSA given-names-by-birth-year; LLM reserved for nickname / initial-expansion tables.
- [x] **Realistic-eval / blocking-eval are blocking-agnostic by design** (§9.2) so blocking, deterministic rules, and Fellegi–Sunter models can be evaluated against them independently.

**Resolved by stats re-run (2026-05-28):**
- [x] **Re-ran `extract_mdm_stats.py`**; `synthetic_data_stats.json` updated. Filled §5.8 (within-cluster agreement) and §5.9 (geo_joint + missingness_patterns) with measured numbers; recalibrated §7 corruption rates and added the §8 DOB-led calibration note.

**Resolved by pre-build decisions (2026-05-28):**
- [x] **K-per-entity** keyed by identifier band (full-SSN→`by_ssn`, last-4→`by_last4_dob`, no-SSN→`by_namecompact_dob`) so no-SSN entities aren't all singletons (§7).
- [x] **Assembly mechanism**: fine-tune corpus is **case-first** to exact §9.1 budgets (clean labels, guaranteed coverage); realistic-eval + M-MIX bulk are **entity-first** at realistic distributions (§9).
- [x] **Corruption application**: independent per-field Bernoulli at the §5.8 marginals; no invented field-correlation; heavy co-corruption supplied by M-MIX-02 / M-SSN-11; messiness multiplier deferred to v2 (§7).
- [x] **Per-field calibration bound**: name/address/phone/email→`by_ssn`; sex→`by_ssn_dob`; DOB→`by_ssn` (§7).
- [x] **Pool sources** locked + `build_pools.py` step: US Census surnames, SSA names-by-year, Chicago street names (§14.4).

**Resolved by build (2026-06-08):**
- [x] **`build_pools.py`** built (offline-first; bootstraps pools from stats + curated supplements) and `pools/*.json` generated.
- [x] **`generate_synthetic.py`** built and verified: entity sampler with correlated geo/missingness/pediatric coupling (§6); independent-marginal corruptions calibrated from §5.8 (§7); full case-first scenario registry covering all §8 buckets (80 distinct `case_type`s fire); entity-first realistic-eval at 1:9; entity-disjoint 15% split; all four §11 outputs (provenance carried inline; no separate manifest files). Full 40k+10k run is ~4 s and **byte-reproducible from `--seed`** (verified via md5 across runs).
- [x] **`qa_checks.py`** built — asserts every §12 structural check (schema parity, value conventions, valid_record, label/case agreement, entity disjointness, ZIP3↔state, SSN structural + last-4 coupling, NANP phones, no filtered tokens, M-NAME derivation invariance) plus the §12.7 distribution-sanity warnings; **all structural checks pass** on the v1 output.

**Still open:**
- [ ] Stakeholder confirmation of the §8.3 default-exclude policy (M-NOSSN-06, M-SEX-01, POL-AMBIG-*). Defaults stand until they say otherwise; the `--include-policy-cases` flag is specced but not yet wired into `generate_synthetic.py`.
- [ ] *(optional)* Network enrichment in `build_pools.py` (Census/SSA/Chicago) for long-tail name diversity + true first-name DOB-conditioning (§6 step 3 currently falls back to the flat measured distribution).
- [ ] *(optional)* `llm_pools.py` (Ollama wrapper) — pools are hand-authored, so not required.
- [ ] *(optional)* Wrap `qa_checks.py` in `synthetic_dataset_qa.ipynb` if a notebook view is wanted.
- [ ] Finer `case_type` tagging for the entity-first realistic-eval (currently coarse: `M-REALISTIC` / `NM-COMMON` / `NM-EASY`).
- [ ] Decide which scenarios (if any) belong to a "test-only" case bucket.

## 14. Generation implementation (code + local LLM)

**Decision (locked 2026-05-28; name sources revised 2026-05-28): the generator is code-primary; vocabulary pools come primarily from public reference datasets, with a local LLM (Ollama) reserved for the small curated tables only.** All per-pair logic — distributions, corruptions, labels, splits — is deterministic Python.

**Name pools come from public reference data, not the LLM.** Using an LLM to "list common surnames" yields a narrow, low-diversity set (the very thing we *don't* want — see §14.1) and isn't defensibly representative. Public frequency lists are larger, citable, and more diverse:

- **Last names:** US Census Bureau surname frequency file (`Frequently Occurring Surnames`, ~160k surnames with counts). Reweight toward the measured AllianceChicago `LastNM_clean` top-N (Hispanic / Chicago character) but draw the long tail from the full Census list for realistic diversity.
- **First names:** SSA `National Data on the relative frequency of given names` (first-name counts **by birth year**). This is a bonus for §6's correlated sampling — sampling first name *conditioned on the entity's DOB year* makes names age-appropriate for free (a 2020s pediatric record draws from 2020s-popular names, not 1950s ones), reweighted toward the measured AllianceChicago first-name top-N.

The LLM never sees real data and never touches per-pair construction, so it cannot introduce label ambiguity or distributional drift. Ollama runs locally, keeping everything on-machine (consistent with the project PHI posture), though no PHI is involved at this stage regardless.

### 14.1 Why code, not LLM-per-pair

- **Exact distribution control** (SSN 21.4%, phone-per-record histogram, DOB decades). §12.7 requires the realistic-eval marginals within ±5pp — an LLM only approximates and would force rejection-sampling against a checker.
- **Exact corruption recipes** (off-by-one DOB, digit transposition, last-4 collisions) and reliable `corruptions_applied` tagging. An LLM asked to "make a typo" produces an unpredictable edit and breaks `full_name_tokens` invariance (§12.8).
- **Labels known by construction** — the entire value of synthetic data. Letting an LLM judge similarity would re-introduce the labeling ambiguity we are escaping.
- **Determinism from `--seed`** — required for the entity-disjoint split (§10) and for the "iterate and scale weak buckets" workflow (§9.1). LLMs are not reproducible across runs.
- **Cost / speed** — 40k pairs is instant in code; an LLM would be 80k+ record generations per run.
- **`valid_record=True` guarantee** (§4.13, §12.13) is enforced by code guards, not hoped for.

### 14.2 Pipeline stages (one command; `--seed` controls all randomness)

0. `ensure_pools()` — Ollama runs **only if** `pools/*.json` are missing or `--regenerate-pools` is passed; output is validated, deduped, normalized, then written to disk (committed).
1. Sample entities (§6).
2. Corrupt → variants (§7).
3. Assemble pairs (§9): fine-tune corpus case-first to the §9.1 budgets; realistic-eval + M-MIX bulk entity-first.
4. Entity-disjoint split (§10).
5. Write CSVs (§11) — provenance carried inline in the pair / record files.

"Run it all at once" holds: the first invocation does everything end-to-end. Reproducibility also holds: once pools are cached + committed, every later run with the same `--seed` is byte-identical — no LLM nondeterminism enters the per-pair stage. **The `--seed` governs sampling; the committed pool files govern vocabulary.** A teammate or CI without Ollama can regenerate the corpus because the pools are committed.

### 14.3 Ollama integration

- Wrapped in a single `llm_pools.py`; the rest of the generator never imports Ollama. **Scope is now narrow:** name/street pools come from public reference data (§14.4), so the LLM only seeds the small `nicknames.json` / `initial_expansion.json` tables (and even those can be hand-authored).
- Model configurable via `--ollama-model` (TBD; small models like `llama3.2` / `qwen2.5` suffice). The generator is model-agnostic.
- Structured `format=json` output.
- **Validation gate:** every LLM-produced token passes the §4.13 invalid-strings filter + ASCII-uppercase normalization + dedup *before* entering a pool. The LLM cannot poison the corpus with `BABY GIRL`, diacritics, or junk.
- **Missing-pool behavior:** if any pool is absent *and* it cannot be produced (public file missing, or Ollama unreachable for the curated tables), **hard-error with a clear message** — no silent low-diversity fallback (so we never ship a degraded corpus unknowingly).

### 14.4 Pools (validated, committed)

**Sources locked 2026-05-28.** A `build_pools.py` helper fetches/normalizes the public files into `pools/*.json` (run once; outputs committed so the generator and CI need no network).

**From public reference data (not the LLM):**
- `pools/last_names.json` — **US Census Bureau "Frequently Occurring Surnames from the 2010 Census"** (`Names_2010Census.csv`, ~162k surnames with counts; public domain). Reweight the head toward the measured `LastNM_clean` top-N (Hispanic/Chicago character); keep the Census tail for diversity.
- `pools/first_names.json` — **SSA "National data on the relative frequency of given names"** (`names.zip` → `yobYYYY.txt` per birth year; public domain). Stored keyed by birth year so §6 draws age-appropriate names; head reweighted toward the measured `FirstNM_clean` top-N.
- `pools/streets.json` — **City of Chicago street names** (Chicago Data Portal / Census TIGER ROADS for Cook + the other resident counties; public). ~1–2k street stems; `AddressLine1` house numbers + USPS suffix are generated procedurally per §6 step 6 using the measured suffix distribution (§5.4).

**LLM-seeded, then hand-verified (correctness > diversity; small tables):**
- `pools/nicknames.json` — bidirectional nickname map (M-NAME-09); weight the Spanish pairs given the population.
- `pools/initial_expansion.json` — initial↔full-name table for the §7 expand-initial corruption.

The LLM (§14.3) is optional: it only seeds the two curated tables, and even those can be hand-authored.

**Build status (implemented 2026-06-08): `build_pools.py` is offline-first.** Rather than depend on network at build time, it bootstraps every pool from two always-available, k-anon-safe inputs and the curated supplements above:
- name/last-name/street pools are seeded from the measured top-N in `synthetic_data_stats.json` (so synthetic names track the *real* AllianceChicago population directly — more faithful to this dataset than a national list) plus curated long-tail supplements weighted to the population's Hispanic/Chicago character;
- the nickname and initial-expansion tables are authored verbatim in `build_pools.py` (no LLM needed for v1).

This makes the whole pipeline reproducible with no network and no Ollama. **First-name DOB-conditioning (§6 step 3) is deferred:** the offline build leaves `by_year` empty and the generator falls back to the flat measured first-name distribution, which already reflects this population's real age mix. Optional network enrichment (Census 2010 surnames / SSA names-by-year / Chicago street file) can be layered behind a flag later to add long-tail diversity and enable true DOB-conditioning.

### 14.5 File layout

```
synthetic_data_generation/
  pools/                       # committed; built by build_pools.py
    first_names.json           # measured top-N + curated tail (by_year empty offline)
    last_names.json            # measured top-N + curated tail
    streets.json               # measured street stems + curated Chicago streets
    nicknames.json             # authored in build_pools.py + hand-verified
    initial_expansion.json     # authored in build_pools.py + hand-verified
  build_pools.py               # offline-first pool builder (bootstraps from stats); run once
  generate_synthetic.py        # IMPLEMENTED — entity sampler, corruptions, case-first
                               #   scenario registry (§8), entity-first realistic-eval,
                               #   entity-disjoint split, all §11 outputs (provenance inline)
  qa_checks.py                 # IMPLEMENTED — asserts every §12 check (importable by a notebook)
  llm_pools.py                 # OPTIONAL / not built — Ollama wrapper for curated tables
```

**Generator CLI.** `python synthetic_data_generation/generate_synthetic.py --seed 42 --version 1`
(full: 40k fine-tune at 1:1.5 + 10k realistic-eval at 1:9 + record-level blocking-eval, ~4 s, byte-reproducible from `--seed`). `--smoke` runs a tiny version for testing. Validate with `python synthetic_data_generation/qa_checks.py --version 1` (add `--real-header <path>` where the real cleaned CSV is available to assert exact blocking-eval schema parity).

Note: §13 / earlier drafts referenced `scripts/generate_synthetic.py`; the file lives in `synthetic_data_generation/` to sit beside its pools and stats.
