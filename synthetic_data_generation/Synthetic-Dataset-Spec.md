# Synthetic Dataset Spec for AnyMatch Fine-Tuning

**Status:** v0.1 draft — scaffold + first scenario catalog. To be refined slowly. Sections marked **[TBD]** depend on `synthetic_data_generation/synthetic_data_stats.json` (produced by `synthetic_data_generation/extract_mdm_stats.py` on the real MDM data).

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

## 2. What the model actually sees (input schema) — **[TBD]**

The exact set of attributes shipped to the model is still being decided. As of this draft, the working assumption is that the three name buckets (`FirstNM`, `MiddleNM`, `LastNM`) are passed through *as separate fields* rather than collapsed into a single derived token-set — so the fine-tuning corpus must teach the model how to combine them. The current prior FEATURE_RENAMES (which used a derived `name` from `full_name_tokens`) is being revisited; this table reflects the working v0.2 schema, not a committed design.

| Model attr (working) | Source column (MDM-cleaned) | Notes |
|---|---|---|
| `first_name` | `FirstNM_clean` | Provisional — passed as its own field so the model can learn cross-name-field coupling on its own. |
| `middle_name` | `MiddleNM_clean` | Provisional — same. |
| `last_name` | `LastNM_clean` | Provisional — same. |
| `dob` | `BirthDT_clean` | Standardized date. |
| `sex` | `SexAtBirthDSC_clean` | MALE / FEMALE / OTHER / null. |
| `ssn` | `SSN_clean` | Full 9-digit; missingness rate to be measured by `extract_mdm_stats.py`. |
| `ssn last 4` | `last_4_SSN` | Backup signal when full SSN is missing. |
| `address` | `AddressLine1_clean` | Street line only. AddressLine2 inclusion is **[TBD]** pending design call. |
| `city` | `CityNM_clean` |  |
| `state` | `StateCD_clean` | 2-letter USPS. |
| `zip` | `ZipCD_clean_base` | 5-digit primary. |
| `phone` | `Phones_set` | Derived: whitespace-joined set of all non-null cleaned phone numbers. |
| `email` | `Email_clean` |  |

Missing values arrive at the prompt as the literal string `'N/A'` (per `df_serializer`).

**Implications for synthesis:** since the model sees the three name fields independently, the fine-tuning data must explicitly demonstrate how tokens move *between* `FirstNM_clean`, `MiddleNM_clean`, and `LastNM_clean` for the same human (Hispanic two-surname swaps, Vietnamese order swaps, middle-name promotion/demotion, etc.). This drives the M-NAME-* scenarios in §8. If the schema later switches back to a token-set derivation, those scenarios still hold — they just become a sanity check on the derivation rather than a teaching signal.

## 3. Generation schema

The synthetic dataset is generated at the **MDM-cleaned column level**, drop-in compatible with `MDM_Population_cleaned_v1.csv`. The same FEATURE_RENAMES + derivations (`full_name_tokens`, `full_name_compact`, `Phones_set`, `Address_normalized`) are then applied to produce model inputs. This keeps the synthetic data interchangeable with real cleaned data and lets us validate the whole pipeline end-to-end on synthetic inputs.

Per-record columns we generate (raw + clean, but generator writes only `_clean` initially — `_raw` is set equal to `_clean` since we have no upstream noise to preserve):

```
PATID
FirstNM_clean, MiddleNM_clean, LastNM_clean, SuffixNM_clean
BirthDT_clean
SSN_clean, last_4_SSN
AddressLine1_clean, AddressLine2_clean
CityNM_clean, StateCD_clean, ZipCD_clean_base, ZipCD_clean_ext
PrimaryPhoneNBR_clean, Phone01NBR_clean, Phone02NBR_clean, Phone03NBR_clean
Email_clean
SexAtBirthDSC_clean
valid_record  (set to True for every generated record — see §10)
```

Derived columns produced post-generation (same logic as Data-Cleaning-Guide §"Global Cross-Field Transformations"):

```
full_name_tokens, full_name_compact, Phones_set, Address_normalized
```

## 4. Design principles

1. **Cleaned-output convention.** Synthetic values follow the conventions in `docs/Data-Cleaning-Guide.md`: uppercase names, ASCII only, standardized USPS suffixes, etc. We do *not* generate dirty inputs and re-clean them; we generate post-cleaned values directly. (Exception: we *do* simulate the few classes of corruption that survive cleaning — e.g., one-digit SSN typos, name-field swaps, DOB transpositions.)
2. **`valid_record=True` only.** Per `CLAUDE.md`, downstream inference filters to `valid_record=True`. The model never sees invalid records, so we never generate them.
3. **Entity-first for the bulk.** Sample N synthetic persons (entities) from realistic distributions; per entity produce K record variants by applying transformations; form match pairs within an entity and non-match pairs across entities. This naturally generates realistic mixed-corruption patterns.
4. **Case-first top-up for edge cases.** Some scenarios (twins, Jr-Sr, junk-SSN last-4 collisions, name-order swaps) are too rare in entity-first sampling to teach the model reliably. We enumerate them explicitly and append.
5. **Realistic distributions cited, not invented.** Every distribution choice (name top-N, DOB year mix, missingness rate per field, ZIP/state mix) cites `synthetic_data_stats.json`. Where we deliberately diverge from real (e.g., oversampling SSN-match-everything-else-disagrees), the spec records the deviation explicitly.
6. **Two-stage output.** §11 — a balanced/oversampled fine-tune corpus *and* a realistic-distribution holdout eval.
7. **Entity-disjoint split.** No synthetic entity appears in both train and test. Prevents the model from memorizing identities. §10.
8. **Deterministic seeds.** Generation accepts a `--seed`. All randomness derives from it.
9. **Auditable provenance.** Every emitted pair carries `entity_id_a`, `entity_id_b`, `case_type` (e.g. `M-SSN-04`), and `corruptions_applied` (list of transformation names). This is shed before training but kept for inspection.

## 5. Real-data statistics needed (consumed by generator)

Stats extracted by `synthetic_data_generation/extract_mdm_stats.py` and written to `synthetic_data_generation/synthetic_data_stats.json`. The generator reads this file; values below are placeholders.

| Statistic | Used to set | Status |
|---|---|---|
| Per-field missingness rate (clean cols) | Marginal P(field present) for each entity | **[TBD]** |
| `no_full_ssn_but_last4_present_pct` | Probability mass: full SSN vs last_4-only vs no SSN | **[TBD]** |
| `FirstNM_clean` top-N freq | Name sampling pool | **[TBD]** |
| `LastNM_clean` top-N freq | Name sampling pool | **[TBD]** |
| `MiddleNM_clean` fill rate + token stats | Whether to emit middle, whether to make it 2-token (Hispanic surname) | **[TBD]** |
| Name token counts (% hyphen, % apostrophe, % multi-token) | Compound-name generation rates | **[TBD]** |
| `BirthDT_clean.decade_histogram` | DOB year sampling | **[TBD]** |
| `CityNM_clean`, `StateCD_clean`, `ZipCD_clean_base` top-N | Address sampling (heavy Chicago / IL skew expected) | **[TBD]** |
| `zip3_to_state_agreement_rate` | Cross-field validity bound | **[TBD]** |
| Phone fill rate per slot + area-code distribution | Phone generation | **[TBD]** |
| Email fill rate + top domains | Email generation | **[TBD]** |
| Sex distribution | Sex sampling | **[TBD]** |
| `clusters.by_ssn`, `by_namecompact_dob` | Estimate true-positive cluster sizes — informs K records per entity | **[TBD]** |
| `valid_record_rate` | Sanity check only (we generate all valid) | **[TBD]** |

**Statistics we do NOT have from records-only stats** (deferred or estimated):
- Field-agreement rates on *known true-positive* pairs (no labeled pairs in scope here).
- The realistic prevalence of edge cases (twin, Jr-Sr, etc.) in candidate pairs — we use heuristics for now; can refine when blocking output is available.

## 6. Entity generation

A synthetic *entity* represents a single ground-truth person. For each entity:

1. **Sex:** sample from `categorical_top.SexAtBirthDSC_clean`.
2. **DOB:** sample year from `dob.decade_histogram`; month uniform 1–12; day uniform 1–28 (avoid month-end edge cases initially; we can lift this once we verify date arithmetic).
3. **Names:** sample first/middle/last from the top-N pools weighted by frequency. With probability *p_compound_first* (Hispanic-leaning population — calibrate from token stats), make first name a 2-token compound (e.g., `MARIA CARMEN`). With probability *p_two_surname* (Hispanic two-surname convention), set middle = paternal surname, last = maternal surname.
4. **Suffix:** sample with probability *p_suffix* (per stats) from {`JR`, `SR`, `II`, `III`, `IV`, `V`}.
5. **SSN:** with probability *p_full_ssn* generate a 9-digit SSN respecting the structural rules from the cleaning guide (area not in `000/666/900-999`, group not `00`, serial not `0000`). Else with probability *p_last4_only* generate only `last_4_SSN`. Else neither.
6. **Address:** sample (city, state, ZIP3) from top-N joint distribution; generate a synthetic street line that conforms to the post-cleaning conventions (`NNN <NAME> ST/AVE/...` with USPS suffix). Apartment with probability *p_apt*.
7. **Phones:** sample N ∈ {0..4} from `co_missingness.phones_per_record`; each phone has area code drawn from real area-code distribution; NXX and line generated to NANP-valid form.
8. **Email:** with probability *p_email_present* generate `<local>@<top-domain>`; local part derived from name with a configurable corruption.

Each entity gets a stable `entity_id` (UUIDv4); each emitted record gets a fresh `PATID`.

## 7. Variant (record) generation per entity

For each entity, decide K (number of records to emit) from the SSN-cluster-size histogram. Most entities get K=1; the upper tail gets K∈{2..6}.

For each pair of variants drawn from the same entity, apply 0..M corruptions. The corruption pool maps directly to the scenario catalog (§8). A *corruption budget* per pair is sampled — most pairs get 1–2 corruptions, a long tail gets more (the synthetic analog of "this record is a mess").

Corruption types (see §8 for the scenarios that combine these):
- **Name corruptions:** single-character typo (insert/delete/substitute/transpose), drop middle, drop middle to initial, swap first↔middle, swap last↔middle, nickname substitution (`ROBERT↔BOB`), hyphenation/spacing variant (`ANNE-MARIE↔ANNE MARIE↔ANNEMARIE`), append/drop suffix, drop one surname from two-surname last.
- **DOB corruptions:** off-by-one day, off-by-one month, off-by-one year, month-day transposition (only meaningful when day ≤ 12), null out.
- **SSN corruptions:** drop full SSN (keep last_4), null both, one-digit typo (only in non-match-collision scenarios), digit transposition.
- **Address corruptions:** move (entirely new address), drop AddressLine2, change apt only, change house-number digit, null street, change city/state/ZIP consistently.
- **Phone corruptions:** drop one slot, drop all, add new number, replace a number.
- **Email corruptions:** drop, replace with new local part, replace with new domain.
- **Sex corruptions:** flip (rare — clerical) or null.

## 8. Scenario catalog

Naming: `<M|NM|EDGE>-<bucket>-<NN>`. Each scenario records:
- **Teaches:** which behavior this case targets.
- **Recipe:** how to construct it from an entity (M) or two entities (NM).
- **Label:** match=1, non-match=0.
- **Fine-tune oversample target:** **[TBD]** — set once stats land.
- **Realistic-eval prevalence:** **[TBD]** — drawn to match natural frequency.

### 8.1 Match scenarios (label = 1)

#### SSN-led — teach: "SSN equality + structurally valid + non-junk ⇒ match"

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

#### Name-coupling-led — teach: "name field assignment is noise; tokens are signal"

- **M-NAME-01 — Hyphenation variant.** `ANNE-MARIE` ↔ `ANNE MARIE` ↔ `ANNEMARIE` across the first-name slot. Same person.
- **M-NAME-02 — First/middle swap.** Side A: `(FirstNM=MARIA, MiddleNM=CARMEN)`; side B: `(FirstNM=CARMEN, MiddleNM=MARIA)`. Clerical reassignment.
- **M-NAME-03 — Two-surname shuffle (Hispanic).** Side A: `(MiddleNM=GARCIA, LastNM=LOPEZ)`; side B: `(MiddleNM=null, LastNM=GARCIA LOPEZ)`. Common when Hispanic patient registers at different clinics.
- **M-NAME-04 — Two-surname collapse to one.** Side A: `LastNM=GARCIA LOPEZ`; side B: `LastNM=LOPEZ` (or `GARCIA`). Same person dropped a surname.
- **M-NAME-05 — Vietnamese name-order swap.** Side A: `(FirstNM=NGUYEN, LastNM=THI MAI)`; side B: `(FirstNM=THI MAI, LastNM=NGUYEN)`. Native vs US order.
- **M-NAME-06 — Middle-initial only.** Side A: `MiddleNM=ANNE`; side B: `MiddleNM=A`. Same person.
- **M-NAME-07 — Compound first dropped.** Side A: `FirstNM=MARIA CARMEN`; side B: `FirstNM=MARIA`. Same person, lazy entry.
- **M-NAME-08 — Generational suffix appears/absent.** Side A: `LastNM=SMITH JR` (post-cleaning collapse); side B: `LastNM=SMITH, SuffixNM=JR`. Or one side has no suffix at all.
- **M-NAME-09 — Nickname.** `ROBERT↔BOB`, `WILLIAM↔BILL`, `ELIZABETH↔BETH/LIZ`, `MICHAEL↔MIKE`, `JOSEPH↔JOE`, `JOSE↔PEPE` (Spanish), `FRANCISCO↔PACO`. Limited curated mapping table — same person.
- **M-NAME-10 — Typo: single-char substitute.** `ROCA` ↔ `ROCQ`. Same SSN/DOB.
- **M-NAME-11 — Typo: transposition.** `SMITH` ↔ `SMIHT`. Same SSN/DOB.
- **M-NAME-12 — Typo: insertion/deletion.** `JOHNSON` ↔ `JOHNSOON` / `JONSON`. Same SSN/DOB.
- **M-NAME-13 — Diacritical loss (already cleaned).** Pre-cleaning, side A was `MUÑOZ` and B was `MUNOZ`; both post-clean to `MUNOZ`. Included as a control: should appear identical at our level. (Tests our cleaning fidelity, not the model.)

#### DOB-led

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
- **M-SEX-01 — Sex disagreement (clerical).** Same SSN/DOB/name/address; sex flipped on one side. Rare in real data but it happens; label remains match because SSN is decisive.

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

#### SSN-related hard negatives (critical: don't false-match on shared data-quality artifacts)

- **NM-SSN-01 — Last-4 collision.** Side A and B have the same `last_4_SSN` (1/10000 by chance), full SSN missing on both. Different name, different DOB. **Teach: last-4 alone isn't enough.**
- **NM-SSN-02 — Last-4 collision + same first letter of name.** Slightly harder version of the above.
- **NM-SSN-03 — Single-digit SSN-typo collision.** Side A's clerk typed an extra '0' into the SSN, which happens to equal side B's real SSN. Different name/DOB. Tests: don't blindly trust SSN equality if name+DOB are wildly different. (Counterweight to M-SSN-* — model must learn SSN is *strong*, not infallible.)
- **NM-SSN-04 — Same SSN, opposite sex, wildly different DOB.** Likely data-entry error in SSN on one side. Different person. Label = 0. Rare; include modest count.

#### Identity-fragment overlap

- **NM-IDF-01 — Shared shelter / group-home address.** Same `AddressLine1_clean`, same city/state/ZIP; different name + DOB + SSN.
- **NM-IDF-02 — Shared family phone.** Same primary phone, different name + DOB + SSN.
- **NM-IDF-03 — Shared family email.** `family@gmail.com` style — different name + DOB.
- **NM-IDF-04 — Shared address + shared phone.** Two unrelated tenants who use one landline. Stronger lure, still NM.

#### Boundary cases

- **NM-BND-01 — Thin records, disagree on name.** Both sides only have first+last+DOB+sex. First or last name differs by more than a typo. Different person. Tests: don't fall back to "almost-empty = match".

### 8.3 Edge / policy cases (label requires explicit decision)

These are scenarios where the *correct* label is ambiguous or depends on the operator's policy. Decisions logged here; cases included only if and how the policy says.

- **POL-AMBIG-01 — Twins with same SSN entered.** Same DOB, same address, same SSN, different first name. In real data this happens when a clerk reuses one twin's SSN. We **exclude** from training (we cannot teach an unambiguous label).
- **POL-AMBIG-02 — Same SSN + same DOB + completely different name + different sex.** Could be a data-entry error in SSN, could be a real person we've conflated. We label as **match=1** for the fine-tune corpus (SSN-trumping policy is the goal of the project) but record this decision explicitly so we can audit later.
- **POL-AMBIG-03 — Same name + same DOB + same address + SSN missing on both.** Could be a household with an unusual coincidence or could be one person registered twice. Default label: **match=1**, with a low oversample budget. Revisit with stakeholders.

> **Policy register.** Every POL-* decision must be approved before generation. Default to excluding any case the team hasn't reviewed.

## 9. Pair assembly

Two-stage build, sharing the same entity / variant generator.

### 9.1 Fine-tune corpus (balanced / oversampled)

Goal: each scenario above has enough examples for the model to internalize it.

- For every M-* and NM-* scenario, target N_scenario examples in the corpus. **[TBD: set N_scenario based on stats + GPU budget; first cut ~500–2000 per scenario.]**
- Within entity-first generation, run match-pair construction; tag each emitted pair with the dominant scenario triggered.
- Top-up: for scenarios whose entity-first counts fall below target, run a case-first generator that constructs pairs deterministically against the spec.
- Net match:non-match ratio in the fine-tune corpus: **~1:1 to 1:3** (closer to 1:3 because hard negatives need volume). **[TBD]**

### 9.2 Realistic-eval set

Goal: measure model performance as it will be observed in production.

- Sample from the *same* entity / variant generator but with prevalence drawn from `synthetic_data_stats.json` and rough blocking-output assumptions.
- Match:non-match ratio: skewed heavily negative — candidate-pair blocking typically returns ~5–20% positives. **[TBD]**
- Each scenario tagged; per-scenario metrics reported separately (so we see if SSN-trumping works on the realistic mix even if it's rare).

## 10. Train / test split

- **Entity-disjoint** random split. Hold out *X*% of entities to *test*; all pairs derived from any held-out entity (within-entity or cross-entity) move to test if either side belongs to a held-out entity. **[TBD: X ≈ 15%.]**
- Both the fine-tune corpus and the realistic-eval set use the same entity hold-out so the same persons appear in test across both.
- Splits are deterministic from `--seed`.

## 11. Output format

CSV files under `data/synthetic/`. Filenames versioned: `vN` suffix to match the cleaning-output versioning convention.

```
data/synthetic/finetune_train_v1.csv     # balanced corpus, train side of entity split
data/synthetic/finetune_test_v1.csv      # balanced corpus, test side
data/synthetic/realistic_eval_v1.csv     # held-out realistic distribution
data/synthetic/entity_manifest_v1.csv    # per-entity ground truth + which split it landed in
data/synthetic/pair_manifest_v1.csv      # per-pair provenance: case_type, corruptions_applied
```

Pair-CSV column layout (every field has `_l` and `_r` suffix; columns mirror the MDM-cleaned schema so the same `FEATURE_RENAMES + derive` pipeline runs):

```
PATID_l, PATID_r,
FirstNM_clean_l, FirstNM_clean_r,
... (all generation-schema columns),
full_name_tokens_l, full_name_tokens_r,
full_name_compact_l, full_name_compact_r,
Phones_set_l, Phones_set_r,
Address_normalized_l, Address_normalized_r,
label,
case_type,           # e.g. "M-SSN-04"
corruptions_applied  # e.g. ["last_name_replace","address_move"]
```

`case_type` and `corruptions_applied` are dropped before passing to `loo.py` / `predict_alliance.py` (the model reads only `*_l`, `*_r`, `label`).

## 12. Sanity checks (run after generation)

Assertions the generator must satisfy and a notebook verifies:

1. **Schema parity.** Synthetic pair-CSV columns match `MDM_Population_cleaned_v1.csv` schema after `_l`/`_r` un-suffixing.
2. **Value conventions.** All names uppercase ASCII; no diacritics; ZIPs are 5-digit strings; SSNs are 9-digit strings with no junk patterns.
3. **`valid_record=True`** on every emitted record.
4. **Label / case agreement.** Every `case_type` in `M-*` has `label=1`; every `NM-*` has `label=0`.
5. **Entity disjointness.** No `entity_id` appears in both train and test.
6. **No PHI leakage.** No row matches anything in real `MDM_Population_cleaned_v1.csv` (a hash-set check on a few high-cardinality fields). The point of synthesis is that real persons cannot be recovered.
7. **Distribution sanity vs `synthetic_data_stats.json`.** Per-field missingness and top-N frequencies in the *realistic-eval* set fall within ±X% of the real stats. **[TBD: X ≈ 5%.]**
8. **Token-set invariance.** For M-NAME-* scenarios where the recipe intends `full_name_tokens` to be equal on both sides (most of them), assert that.

## 13. Open questions / TODO

- [ ] Run `extract_mdm_stats.py` and commit `synthetic_data_generation/synthetic_data_stats.json`. Replace every **[TBD]** with the measured number.
- [ ] Confirm `p_compound_first` and `p_two_surname` rates with a quick manual look at the top-N MiddleNM / LastNM lists (Hispanic-naming proxy).
- [ ] Decide N_scenario budget per scenario; total fine-tune-corpus size = sum(N_scenario).
- [ ] Decide whether `M-SEX-01` (sex disagreement, SSN-anchored) should be included or held back as ambiguous.
- [ ] Confirm `POL-AMBIG-*` decisions with stakeholders.
- [ ] Settle the nickname mapping table (M-NAME-09) — start from a small curated list, expand only with evidence.
- [ ] Decide the **scale**: total pairs in fine-tune corpus + total in realistic-eval. Will fall out of N_scenario × number-of-scenarios + held-out percentage.
- [ ] Build the generator (`scripts/generate_synthetic.py` or similar) — separate work item, downstream of this spec.
- [ ] Build the sanity-check notebook (`synthetic_data_generation/synthetic_dataset_qa.ipynb`).
- [ ] Decide which scenarios (if any) belong to a "test-only" case bucket (we previously rejected case-stratified holdout, but worth revisiting once we see fine-tune-corpus performance).
