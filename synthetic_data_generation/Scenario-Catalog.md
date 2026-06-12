# Scenario Catalog — full reference

Every pair scenario the generator emits, with a **complete per-field map** so you can see exactly what
each case does to all 15 model fields. Plain-English companion to `Synthetic-Dataset-Spec.md` (design
rationale + statistics). **Source of truth:** `generate_synthetic.py::ScenarioLib` — if this disagrees,
the code wins.

## How to read this

Every pair is two records, side **A** and side **B**. Each scenario lists **all 15 model fields** (Spec
§2) in this fixed order, with an abbreviation and a one-symbol state:

| abbr | field | abbr | field | abbr | field |
|---|---|---|---|---|---|
| `fst` | first_name | `dob` | dob | `ad1` | address line 1 |
| `mid` | middle_name | `sex` | sex | `ad2` | address line 2 |
| `lst` | last_name | `ssn` | full 9-digit SSN | `cty` | city |
| `sfx` | suffix | `l4` | last-4 SSN | `st` | state |
| | | | | `zip` | zip |
| | | | | `phn` | phone (set) |
| | | | | `eml` | email |

**Per-field state symbols — this is the key to "missing on both vs one":**

| symbol | meaning |
|---|---|
| `=` | **identical** on both sides — same value, *or* both empty via realistic missingness (not forced) |
| `≠` | **present on both, but different** — the `↳` note says how |
| `∅` | **forced empty on BOTH** records (the scenario removes/omits it) |
| `∅A` / `∅B` | **forced empty on that one side** (present on the other) — the asymmetric/one-sided case |

**Defaults (because of how pairs are built):**
- **MATCH (`M-*`, label 1)** — A and B start as **clones of one person**, so every field is `=` unless
  the scenario changes it. The fields marked `≠`/`∅` are the *only* edits; the **Anchor** is the
  surviving strong signal that still proves identity.
- **NON-MATCH (`NM-*`, label 0)** — A and B are **two different people**, so any field not forced equal
  is genuinely different. To read NM maps: `=` marks the **forced collision** (the blocking trap); all
  other identity fields differ by construction. SSN is nulled on B in most NM cases (B is "the other
  person we have less data on"); shown as `∅B` (A keeps whatever its entity sampled).

Derived columns (`full_name_tokens`, `full_name_compact`, `Phones_set`, `Address_normalized`) are
recomputed from the 15 fields and not shown separately. **SSN band** (the spine of the match catalog):

| Band | `ssn` / `l4` state | Share of matches |
|---|---|---|
| **No-SSN** | `ssn ∅` · `l4 ∅` (both empty) | bulk — the FQHC norm |
| **Last-4 only** | `ssn ∅` · `l4 =` (last-4 on both, no full) | medium |
| **Full SSN** | `ssn =` · `l4 =` (`l4 = ssn[-4:]`) | strong-signal minority |

---

# MATCH scenarios (label = 1) — same person

## No-SSN-led — anchor: name + DOB (no SSN anywhere)
*The dominant real population. The model must match comfortably on name+DOB plus one corroborator.*

**M-NOSSN-01 — No-SSN control** · *anchor: name+DOB+address*
`fst= mid= lst= sfx= dob= sex= ssn∅ l4∅ ad1= ad2= cty= st= zip= phn= eml=`
*Identical no-SSN record — the sanity-check positive for the no-SSN path.*

**M-NOSSN-02 — Moved, no SSN** · *anchor: name+DOB*
`fst= mid= lst= sfx= dob= sex= ssn∅ l4∅ ad1≠ ad2= cty= st= zip≠ phn= eml=`
*Same person, new home. ↳ `ad1` new street (~half also new `zip`; usually same city/state).*

**M-NOSSN-03 — Moved but shares a phone** · *anchor: name+DOB+phone overlap*
`fst= mid= lst= sfx= dob= sex= ssn∅ l4∅ ad1≠ ad2= cty= st= zip≠ phn= eml=`
*Address changed but a phone number carries over. ↳ `ad1`/`zip` changed; `phn` overlaps ≥1.*

**M-NOSSN-04 — Thin transient record** · *anchor: name+DOB+sex (all there is)*
`fst= mid= lst≈ sfx= dob= sex= ssn∅ l4∅ ad1∅ ad2∅ cty∅ st∅ zip∅ phn∅ eml∅`
*Only name + DOB + sex on both sides (sparse FQHC reality). ↳ address, phone, email forced empty on both;
a 1-char `lst` (or `fst`) difference is enforced so the pair isn't identical. This is the **thin positive**
that balances the thin negatives NM-BND-01/02 — the model must not learn "sparse ⇒ non-match."*

**M-NOSSN-05 — No-SSN with a name typo** · *anchor: DOB + most of name*
`fst≈ mid= lst≈ sfx= dob= sex= ssn∅ l4∅ ad1= ad2= cty= st= zip= phn= eml=`
*↳ exactly one of `fst`/`lst` has a single-character typo on B (the other stays `=`).*

**POL-AMBIG-03 — Household-coincidence / duplicate** · *anchor: name+DOB+address (weak, collidable)* · **low-weight (~2%), tagged**
`fst= mid= lst= sfx= dob= sex= ssn∅ l4∅ ad1= ad2= cty= st= zip= phn≠ eml=`
*Same name + DOB + address, **no SSN on either side** — genuinely ambiguous (a household coincidence vs the
same patient registered twice). Project policy labels it **match=1** at a deliberately low weight, tagged
`POL-AMBIG-03` so its recall/precision can be tracked separately. ↳ only a weak field (`phn`) drifts; the
strong fields agree (that agreement is the ambiguity). The strong NM-HH-* household negatives teach the
boundary so this isn't a blanket "same address ⇒ match." Emitted without the no-identical rule.*

## SSN-led — anchor: full SSN (everything else may drift freely)
*A valid full-SSN equality decides the pair; name/address/DOB/contact may all change.*

**M-SSN-01 — Identical record (control)** · *anchor: full SSN*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*All fields agree — calibration anchor for the full-SSN band.*

**M-SSN-02 — SSN match, name typos** · *anchor: full SSN*
`fst≠ mid= lst≠ sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `fst` and `lst` each get a 1-char typo on B; SSN proves it's the same person.*

**M-SSN-03 — SSN match, missing middle** · *anchor: full SSN*
`fst= mid∅B lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `mid` present on A, forced empty on B (one-sided drop).*

**M-SSN-04 — Maiden ↔ married surname** · *anchor: full SSN*
`fst= mid= lst≠ sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*Surname changed by marriage/divorce. ↳ `lst` is an entirely different surname on B.*

**M-SSN-05 — Moved (new street)** · *anchor: full SSN*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1≠ ad2= cty= st= zip≠ phn= eml=`
*↳ `ad1` new street, usually same city (~half new `zip`).*

**M-SSN-06 — Moved out of state** · *anchor: full SSN*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1≠ ad2= cty≠ st≠ zip≠ phn= eml=`
*↳ whole address block changes together (new city + state + zip + street).*

**M-SSN-07 — New phone & email** · *anchor: full SSN*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn≠ eml≠`
*↳ `phn` fully replaced, `eml` changed; identity fields untouched.*

**M-SSN-08 — DOB clerical drift** · *anchor: full SSN*
`fst= mid= lst= sfx= dob≠ sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `dob` off by one day or year — SSN trumps the disagreement.*

**M-SSN-09 — Full SSN ↔ last-4 only** · *anchor: SSN/last-4*
`fst= mid= lst= sfx= dob= sex= ssn∅B l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `ssn` present on A, forced empty on B; B's `l4` still equals A's tail.*

**M-SSN-10 — SSN ↔ no SSN at all** · *anchor: name+DOB+address*
`fst= mid= lst= sfx= dob= sex= ssn∅B l4∅B ad1= ad2= cty= st= zip= phn= eml=`
*Verifies the model doesn't *require* SSN. ↳ A has full SSN; B has neither `ssn` nor `l4`.*

**M-SSN-11 — Heavy drift, SSN anchors** · *anchor: full SSN*
`fst= mid∅B lst≠ sfx= dob≠ sex= ssn= l4= ad1≠ ad2= cty= st= zip≠ phn≠ eml=`
*Realistic worst case. ↳ last-name typo + middle dropped on B + DOB drift + move + new phone; SSN holds.*

## Last-4-led — anchor: last-4 + name/DOB (no full SSN on either side)

**M-L4-01 — Last-4 control** · *anchor: last-4 + name + DOB*
`fst= mid= lst= sfx= dob= sex= ssn∅ l4= ad1= ad2= cty= st= zip= phn= eml=`
*Control for the backup-only band — last-4 + name + DOB, no full SSN.*

**M-L4-02 — Last-4 + DOB, name typo** · *anchor: last-4 + DOB*
`fst= mid= lst≠ sfx= dob= sex= ssn∅ l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `lst` single-char typo; last-4 + DOB anchor.*

**M-L4-03 — Last-4 + name, DOB off-by-one** · *anchor: last-4 + name*
`fst= mid= lst= sfx= dob≠ sex= ssn∅ l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `dob` drifts one day/year; last-4 + name anchor.*

**M-L4-04 — Asymmetric full-vs-last-4 + name drift** · *anchor: SSN/last-4*
`fst= mid= lst≠ sfx= dob= sex= ssn∅B l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ A full SSN, B last-4 only (B's `l4` = A's tail), plus a `lst` typo on B. The clean no-drift
asymmetric case is **M-SSN-09** — the two were merged so this one always carries drift.*

## Name-coupling — teach "name-field assignment is noise, tokens are signal"
*Unless noted these force a full SSN so the anchor is unambiguous; M-NAME-13..16 use a default entity
(SSN mirrored per person → anchor is name+DOB).*

**M-NAME-01 — Hyphenation variant** · *anchor: SSN+DOB*
`fst≠ mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `fst` formatting: `ANNE-MARIE ↔ ANNE MARIE ↔ ANNEMARIE`.*

**M-NAME-02 — First ↔ middle swap** · *anchor: SSN+DOB*
`fst≠ mid≠ lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `fst` and `mid` swapped between the two records.*

**M-NAME-03 — Two-surname shuffle (Hispanic)** · *anchor: SSN+DOB*
`fst= mid∅B lst≠ sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ A `(mid=S1, lst=S2)` vs B `(mid empty, lst="S1 S2")` — same tokens, different slots.*

**M-NAME-04 — Two-surname collapse** · *anchor: SSN+DOB*
`fst= mid= lst≠ sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ A `lst="S1 S2"` vs B `lst=S1` (one surname dropped).*

**M-NAME-05 — Vietnamese name-order swap** · *anchor: SSN+DOB*
`fst≠ mid∅ lst≠ sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ given/family swapped between `fst` and `lst`; `mid` empty on both.*

**M-NAME-06 — Middle full ↔ initial** · *anchor: SSN+DOB*
`fst= mid≠ lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `mid` is `ANNE` on A, `A` on B.*

**M-NAME-07 — Compound first dropped** · *anchor: SSN+DOB*
`fst≠ mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `fst` `"MARIA CARMEN" ↔ "MARIA"`.*

**M-NAME-08 — Generational suffix appears/absent** · *anchor: SSN+DOB*
`fst= mid= lst≠ sfx∅A dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ A `lst="SMITH JR"`, B `lst="SMITH"` + `sfx="JR"` (suffix present on B only).*

**M-NAME-08b — Suffix in wrong slot** · *anchor: SSN+DOB*
`fst= mid∅B lst= sfx∅A dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ A carries `JR` in `mid`; B carries it in `sfx` (each present on one side only).*

**M-NAME-09 — Nickname / cross-language variant** · *anchor: SSN+DOB*
`fst≠ mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `fst` swaps to a curated equivalent: nickname (`ROBERT ↔ BOB`) or cross-language (`GUILLERMO ↔ WILLIAM`, `JOSE ↔ JOSEPH`). Merged with the former M-NAME-15.*

**M-NAME-10 / 11 / 12 — Name typos** · *anchor: SSN+DOB*
`fst≈ mid= lst≈ sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ a `lst` (or `fst`) typo on B — substitution (10), transposition (11), insertion/deletion (12).*

**M-NAME-13 — First name → initial** · *anchor: name+DOB*
`fst≈ mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `fst` `JOHN ↔ J` (or a typo fallback). SSN/`l4` mirrored per entity.*

**M-NAME-14 — Long last name truncated** · *anchor: name+DOB*
`fst= mid= lst≠ sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `lst` `HERNANDEZHERNANDEZ → HERNANDEZ` (field-length cut).*

**M-NAME-16 — Spacing / concatenation** · *anchor: name+DOB*
`fst= mid= lst≠ sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `lst` `DE LA CRUZ ↔ DELACRUZ`.*

**M-NAME-17 — Conflicting middle name** · *anchor: full SSN*
`fst= mid≠ lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*Both sides have a middle name/initial, but they disagree. ↳ `mid` differs — teaches a conflicting middle should *weaken*, not *break*, a match when a decisive identifier (SSN) agrees.*

## Drift — single-field tolerance (address / DOB / phone / email / sex / pediatric)

**M-ADDR-01 — Apartment toggled** · *anchor: name+DOB+street*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2∅B cty= st= zip= phn= eml=`
*↳ `ad2` present on A (`APT 4B`), empty on B — same building.*

**M-ADDR-02 — Apartment changed** · *anchor: name+DOB+street*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2≠ cty= st= zip= phn= eml=`
*↳ `ad2` differs (moved within building).*

**M-ADDR-03 — Line2 absorbed into line1** · *anchor: name+DOB*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1≠ ad2∅B cty= st= zip= phn= eml=`
*↳ A `"123 MAIN" + ad2 "APT 5"` vs B `"123 MAIN APT 5" + ad2 empty` — same address, different parsing.*

**M-ADDR-04 — House-number typo** · *anchor: full SSN*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1≠ ad2= cty= st= zip= phn= eml=`
*↳ `ad1` house number off by a digit.*

**M-ADDR-05 — Move within ZIP** · *anchor: name+DOB*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1≠ ad2= cty= st= zip= phn= eml=`
*↳ `ad1` new street, same `zip`/`cty` (common churn).*

**M-ADDR-06 — Directional / abbreviation drift** · *anchor: name+DOB*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1≠ ad2= cty= st= zip= phn= eml=`
*↳ `ad1` `N MAIN ST ↔ NORTH MAIN STREET`.*

**M-ZIP-01 — ZIP-only drift** · *anchor: name+DOB+street*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip≠ phn= eml=`
*↳ `zip` changes to another ZIP in the same city (data-entry / boundary correction).*

**M-DOB-02 — Month-day transposition** · *anchor: full SSN*
`fst= mid= lst= sfx= dob≠ sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `dob` `01-15 ↔ 10-15` (only when day ≤ 12).*

**M-DOB-03 — Off-by-one year** · *anchor: full SSN*
`fst= mid= lst= sfx= dob≠ sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `dob` year ±1.*

**M-DOB-04 — Off-by-one day** · *anchor: full SSN*
`fst= mid= lst= sfx= dob≠ sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `dob` day ±1.*

**M-DOB-05 — DOB null on one side** · *anchor: full SSN*
`fst= mid= lst= sfx= dob∅B sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `dob` present on A, forced empty on B.*

**M-PHONE-01 — Partial phone overlap** · *anchor: name+DOB+address*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn≈ eml=`
*↳ each side has 2 phones, exactly 1 in common (`phn≈` = overlapping-but-not-equal sets).*

**M-PHONE-02 — Phones disjoint** · *anchor: full SSN*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn≠ eml=`
*↳ `phn` fully replaced (no overlap).*

**M-EMAIL-01 — Email changed** · *anchor: name+DOB*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml≠`
*↳ `eml` new local part / address.*

**M-EMAIL-02 — Email domain typo** · *anchor: name+DOB*
`fst= mid= lst= sfx= dob= sex= ssn= l4= ad1= ad2= cty= st= zip= phn= eml≠`
*↳ `eml` domain typo `gmail.com → gamil.com`; still the same person.*

**M-SEX-02 — OTHER ↔ male/female** · *anchor: full SSN*
`fst= mid= lst= sfx= dob= sex≠ ssn= l4= ad1= ad2= cty= st= zip= phn= eml=`
*↳ `sex` `OTHER` on A vs `MALE`/`FEMALE` on B.*

**M-PED-01 — Pediatric thin** · *anchor: name+DOB+address*
`fst= mid= lst= sfx= dob= sex= ssn∅ l4∅ ad1= ad2= cty= st= zip= phn= eml∅`
*Child (DOB 2010s+): no SSN, no own email, parent address/phone. ↳ `eml` empty on both.*

**M-PED-02 — Pediatric with last-4** · *anchor: last-4 + name + DOB*
`fst= mid= lst= sfx= dob= sex= ssn∅ l4= ad1= ad2= cty= st= zip= phn= eml=`
*Child with a last-4 recorded; otherwise identical.*

**M-PED-03 — Pediatric name drift** · *anchor: name(mostly)+DOB+address*
`fst≈ mid∅B lst= sfx= dob= sex= ssn∅ l4∅ ad1= ad2= cty= st= zip= phn= eml=`
*↳ middle dropped on B *or* a first-name typo (one of the two).*

## Mixed — heavy realistic drift (last-4/SSN still anchors)

**M-MIX-01 — Two corruptions** · *anchor: last-4 + DOB*
`fst= mid= lst≠ sfx= dob= sex= ssn∅ l4= ad1≠ ad2= cty= st= zip≠ phn= eml=`
*↳ `lst` typo + address move.*

**M-MIX-02 — Three corruptions** · *anchor: last-4 + DOB*
`fst= mid= lst≠ sfx= dob= sex= ssn∅ l4= ad1≠ ad2= cty= st= zip≠ phn≠ eml=`
*↳ `lst` typo + address move + phone replace.*

*(M-MIX-03 removed — it duplicated M-NOSSN-04; thin records live there.)*

---

# NON-MATCH scenarios (label = 0) — different people who collide

> Read NM maps as: `=` is the **forced collision** (why a matcher might be fooled); every other identity
> field genuinely differs because A and B are different people. `∅B` marks SSN deliberately removed on B.

## Easy — anchor only (small minority, train only)

**NM-EASY-01 — Random strangers**
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn≠ l4≠ ad1≠ ad2≠ cty≠ st≠ zip≠ phn≠ eml≠`
*No shared field beyond ambient base rates.*

**NM-EASY-02 — Same state only**
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn≠ l4≠ ad1≠ ad2≠ cty≠ st= zip≠ phn≠ eml≠`
*↳ `st` shared, otherwise unrelated.*

## Household — same address, different person *(the hardest precision pressure)*

**NM-HH-TWIN** — *collide: address + last + DOB*
`fst≠ mid≠ lst= sfx= dob= sex≠ ssn∅B l4∅B ad1= ad2= cty= st= zip= phn≠ eml≠`
*Twins: same surname, same DOB, same home; different first name.*

**NM-HH-TRIPLET-LIKE** — *collide: address + DOB*
`fst≠ mid≠ lst≠ sfx= dob= sex≠ ssn∅B l4∅B ad1= ad2= cty= st= zip= phn≠ eml≠`
*Same DOB and home, different surname (cohabiting unrelated, born same day).*

**NM-HH-JR-SR** — *collide: address + last + first*
`fst= mid≠ lst= sfx∅A dob≠ sex= ssn∅B l4∅B ad1= ad2= cty= st= zip= phn≠ eml≠`
*Parent/child same name; B has `sfx=JR`, DOBs 20–40y apart. **Hardest non-match.***

**NM-HH-SIBLING** — *collide: address + last*
`fst≠ mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4∅B ad1= ad2= cty= st= zip= phn≠ eml≠`
*Same surname + home; DOBs within ±10y; different first name.*

**NM-HH-PARENT-CHILD** — *collide: address + last*
`fst≠ mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4∅B ad1= ad2= cty= st= zip= phn≠ eml≠`
*Same surname + home; DOBs 15–40y apart.*

**NM-HH-SPOUSE** — *collide: address + phone*
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn∅B l4∅B ad1= ad2= cty= st= zip= phn= eml≠`
*Different last name, shared home + phone, DOB ±5y.*

**NM-HH-ROOMMATE** — *collide: address*
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn∅B l4∅B ad1= ad2= cty= st= zip= phn≠ eml≠`
*Same home only; nothing else in common.*

**NM-HH-COUSIN** — *collide: last + city + ZIP*
`fst≠ mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4≠ ad1≠ ad2≠ cty= st= zip= phn≠ eml≠`
*Same surname, same city/ZIP, different street + DOB.*

## Common-name — popular name collisions

**NM-COMMON-01 — Same name, same city, diff DOB**
`fst= mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4∅B ad1≠ ad2≠ cty= st= zip≠ phn≠ eml≠`
*`(JOHN SMITH, CHICAGO)` × 2; DOB ≥5y apart.*

**NM-COMMON-02 — Same name, same ZIP**
`fst= mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4∅B ad1≠ ad2≠ cty= st= zip= phn≠ eml≠`
*↳ tighter geo (`zip` collides) but still different person.*

**NM-COMMON-03 — Hispanic two-surname collision**
`fst= mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4∅B ad1≠ ad2≠ cty= st= zip≠ phn≠ eml≠`
*Shared two-surname full name (`MARIA GARCIA LOPEZ` × 2), different DOB.*

**NM-COMMON-04 — Top-ZIP common-name collision**
`fst= mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4∅B ad1≠ ad2≠ cty= st= zip= phn≠ eml≠`
*Same name + one of the populous Chicago ZIPs — teaches a crowded ZIP is a weak signal.*

**NM-COMMON-05 — Same name, same area code**
`fst= mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4∅B ad1≠ ad2≠ cty≠ st≠ zip≠ phn≈ eml≠`
*↳ both phones share a `773`/`312`/`708` area code but the full numbers differ (`phn≈` = area-code overlap only).*

**NM-COMMON-06 — Same name, adjacent DOB**
`fst= mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4∅B ad1≠ ad2≠ cty≠ st≠ zip≠ phn≠ eml≠`
*↳ `dob` only ±1–2 days apart — looks like a DOB typo but is a different person.*

**NM-COMMON-07 — Nickname "false friend"**
`fst≈ mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4∅B ad1≠ ad2≠ cty= st= zip≠ phn≠ eml≠`
*↳ same surname + a near-variant first name (`fst≈`, looks like the M-NAME-09 nickname case), but `dob` ≥5y apart and no shared SSN — name similarity alone is **not** identity.*

## SSN / identity-fragment collisions *(don't false-match on shared data artifacts)*

**NM-SSN-01 — Last-4 collision**
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn∅ l4= ad1≠ ad2≠ cty≠ st≠ zip≠ phn≠ eml≠`
*↳ same `l4` by chance, no full SSN; different name + DOB. Last-4 alone isn't identity.*

**NM-SSN-02 — Last-4 collision + same first initial**
`fst≈ mid≠ lst≠ sfx= dob≠ sex≠ ssn∅ l4= ad1≠ ad2≠ cty≠ st≠ zip≠ phn≠ eml≠`
*↳ as NM-SSN-01 plus a shared first initial (slightly harder).*

**NM-SSN-03 — Full-SSN typo collision**
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn= l4= ad1≠ ad2≠ cty≠ st≠ zip≠ phn≠ eml≠`
*↳ B's clerk typed A's SSN; name + DOB wildly different. **SSN is strong, not infallible.***

**NM-SSN-04 — Same SSN, opposite sex**
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn= l4= ad1≠ ad2≠ cty≠ st≠ zip≠ phn≠ eml≠`
*↳ same typed SSN, `sex` forced MALE/FEMALE opposite — clearly two people.*

**NM-SSN-05 — Full SSN vs *mismatching* last-4**
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn∅B l4≠ ad1≠ ad2≠ cty≠ st≠ zip≠ phn≠ eml≠`
*↳ A full SSN; B's `l4` deliberately ≠ A's tail. A contradicting last-4 *kills* the SSN signal.*

**NM-SSN-06 — Last-4 + DOB collision**
`fst≠ mid≠ lst≠ sfx= dob= sex≠ ssn∅ l4= ad1≠ ad2≠ cty≠ st≠ zip≠ phn≠ eml≠`
*↳ shared `l4` *and* `dob` (the collision-heavy `by_last4_dob` cluster), different name.*

**NM-IDF-01 — Shared shelter/group-home address**
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn∅B l4∅B ad1= ad2= cty= st= zip= phn≠ eml≠`
*↳ whole address shared by unrelated residents.*

**NM-IDF-02 — Shared family phone**
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn≠ l4≠ ad1≠ ad2≠ cty≠ st≠ zip≠ phn= eml≠`
*↳ one phone number shared, nothing else.*

**NM-IDF-03 — Shared family email**
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn≠ l4≠ ad1≠ ad2≠ cty≠ st≠ zip≠ phn≠ eml=`
*↳ `family@gmail.com`-style shared inbox.*

**NM-IDF-04 — Shared address + phone**
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn∅B l4∅B ad1= ad2= cty= st= zip= phn= eml≠`
*↳ unrelated tenants sharing a landline at one address (stronger lure, still NM).*

**NM-IDF-05 — Shared email domain only**
`fst≠ mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4∅B ad1≠ ad2≠ cty≠ st≠ zip≠ phn≠ eml≈`
*↳ same surname + same email *provider* (`eml≈` = same domain, different local part), different person. A shared `@gmail.com` is not a signal (66% of the population). Surname collision keeps it blocking-survivor-like.*

## Pediatric / boundary

**NM-PED-01 — Pediatric siblings** — *collide: address + last + phone*
`fst≠ mid≠ lst= sfx= dob≠ sex≠ ssn∅B l4∅B ad1= ad2= cty= st= zip= phn= eml≠`
*Both children, same surname/home/phone, different first name + DOB.*

**NM-PED-02 — Pediatric same-DOB unrelated** — *collide: address + DOB*
`fst≠ mid≠ lst≠ sfx= dob= sex≠ ssn∅B l4∅B ad1= ad2= cty= st= zip= phn≠ eml≠`
*Two unrelated kids, same DOB + building, different name.*

**NM-BND-01 — Thin records, disagree on name**
`fst≠ mid≠ lst≠ sfx= dob≠ sex≠ ssn∅ l4∅ ad1∅ ad2∅ cty∅ st∅ zip∅ phn∅ eml∅`
*↳ only name + DOB present, and they differ — "almost empty" must not mean match.*

**NM-BND-02 — Thin records, disagree on DOB**
`fst= mid≠ lst= sfx= dob≠ sex≠ ssn∅ l4∅ ad1∅ ad2∅ cty∅ st∅ zip∅ phn∅ eml∅`
*↳ names agree (common-name pair) but `dob` differs by years — counterweight to M-NOSSN-04.*

## Bulk hard negatives (`NM-HARD-*`) — combinatorial, not named

Generated by `make_hard_negative`: two **different** people forced to share **1–3** strong blocking
fields chosen from {`lst`, `dob`, address-block, `phn`, `l4`} (first name may ride along when ≥3 keys),
then each side independently corrupted. The `case_type` records which keys collided
(`NM-HARD-ADDR+LASTNAME`, `NM-HARD-DOB+PHONE`, …). They are correct by construction (different people),
so any field map reduces to: the named keys are `=`, everything else differs. These are the bulk of the
negative budget — the realistic "survived blocking but isn't a match" population.

**Same-person-signature guard (policy invariant).** A distinct person can still *coincidentally* match an
un-forced identity field — most often a shared **common first name** on top of forced last+DOB+address. If
that lands on **all four** of `fst`+`lst`+`dob`+`ad1`, the negative becomes byte-identical to the
name+DOB+address positives (M-NOSSN-*, M-ADDR-*, POL-AMBIG-03), so labeling it `0` is direct **label noise**.
Per project policy *first+last+DOB+address agree ⇒ match*, so `break_identity_collision` runs after every
negative is built (inside `force_shared_keys` and, defensively, in `_run_methods` for any `label=0`
scenario) and forces **one non-forced** identity field on B to differ (priority `fst`→`lst`→`dob`→`ad1`,
never undoing a deliberately-shared blocking key). Enforced as a hard QA invariant: **zero negatives may
agree on all of first+last+DOB+address** (`qa_checks.py` §12.8b).

---

# Generation proportions (what actually drives sampling)

`_assemble` builds each file from these shares (no entity-first bulk positives since v0.6). Within a
band/source, scenarios are **round-robined** to roughly equal counts.

**Positives — split by SSN band** (`_band_plan`), then scenarios drawn from that band's cover list:

| Band | Share of positives | Scenarios drawn |
|---|---|---|
| Full SSN | **5%** | `SSN_COVER`: M-SSN-02/03/04/05/06/07/08/11 |
| Last-4 only | **15%** | `L4_COVER`: M-L4-02/03/04, M-SSN-09 |
| No usable SSN | **80%** | `HARD_COVER` (~40 scenarios): the no-SSN / name-coupling / address / DOB / phone / email / sex / pediatric / mix cases |

Within the no-usable-SSN band, **~2% of all positives** are carved out for the ambiguous **POL-AMBIG-03**
household-duplicate match (`POL_AMBIG_FRAC`), emitted *without* the no-identical rule (its name+DOB+address
agreement is the point) and tagged for separate auditing.

**Negatives — overwhelmingly hard, key-sharing:**

| Source | Share of negatives | What |
|---|---|---|
| Easy (train only) | **~3%** | NM-EASY-01/02 (random / same-state) |
| Named hard NM | **~34%** | `NM_HARD_COVER`: household, common-name, SSN-collision, identity-fragment, pediatric, boundary |
| Combinatorial hard NM | **~63%** | `NM-HARD-*` (`make_hard_negative`) — two different people sharing 1–3 forced strong keys |

**Not emitted (controls).** Pure controls that are *identical on name + DOB + address* are not written
out — the §8.4 "no identical positives" rule would corrupt them anyway, so they remain conceptual
anchors only: **M-SSN-01, M-NOSSN-01, M-L4-01, M-PED-01/02, M-SSN-10**. (M-NOSSN-04 is the one thin
control that *is* emitted, because a single enforced 1-char name diff makes it a meaningful sparse match.)

