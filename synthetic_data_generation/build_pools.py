"""Build the vocabulary pools the synthetic generator draws from
(Synthetic-Dataset-Spec.md §14.4).

Design decision (2026-05-28): this script is **offline-first**. The spec locks
US Census surnames / SSA given-names-by-year / Chicago street names as the
ideal sources, but those require network access and are not reproducible in CI.
So the default build bootstraps every pool from two always-available, k-anon-safe
inputs:

  1. `synthetic_data_stats.json` — the measured top-N first/last names (with
     counts), street last-tokens, and suffix/apt distributions. This makes the
     synthetic names track the *real* AllianceChicago population directly, which
     is more faithful to this dataset than a national name list.
  2. Curated long-tail supplements (below) — to add diversity beyond the
     k-anon top-N head, weighted toward the measured Hispanic / Chicago character.

The curated nickname and initial-expansion tables are authored here (correctness
matters more than diversity; §14.4) and written verbatim.

Optional network enrichment (Census/SSA/Chicago portal) can be layered on later
behind a flag; the offline build is the committed default.

Usage:
    python synthetic_data_generation/build_pools.py \
        --stats synthetic_data_generation/synthetic_data_stats.json \
        --out   synthetic_data_generation/pools
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# USPS street-type suffix tokens we must NOT mistake for street *names* when
# harvesting stems from the measured address last-token distribution.
STREET_SUFFIX_TOKENS = {
    "AVE", "ST", "RD", "DR", "PL", "CT", "BLVD", "LN", "WAY", "CIR", "HWY",
    "PKWY", "TER", "PARK", "SQ", "PT", "ROW", "RUN", "TRL", "XING", "PATH",
    "LOOP", "WALK", "PLZ", "ALY", "BND", "CRES",
}

# Curated long-tail name supplements (uppercase ASCII, no junk). Weighted toward
# the measured population character (heavy Hispanic + Chicago mix). These extend
# diversity beyond the k-anon top-N head from the stats file.
CURATED_FIRST_NAMES = [
    # Hispanic
    "GUADALUPE", "ESPERANZA", "ROCIO", "XIMENA", "ALEJANDRO", "SANTIAGO",
    "MATEO", "EMILIANO", "VALENTINA", "RENATA", "CAMILA", "ISABELA", "GABRIELA",
    "FERNANDA", "LUCIA", "MARTA", "PILAR", "CONSUELO", "RAMON", "ARMANDO",
    "ROBERTO", "FERNANDO", "EDUARDO", "RICARDO", "ALBERTO", "ENRIQUE", "RAFAEL",
    "SERGIO", "JAVIER", "GERARDO", "LORENA", "ADRIANA", "BEATRIZ", "DOLORES",
    # Anglo
    "DOROTHY", "MARGARET", "HAROLD", "WALTER", "EUGENE", "GERALD", "FRANCES",
    "GLORIA", "RALPH", "HOWARD", "EDITH", "CLARENCE", "BERNICE", "FLOYD",
    "MABEL", "WILLARD", "AGNES", "DELORES", "EVELYN", "GERTRUDE", "LEONARD",
    # African-American common
    "DEANDRE", "JAMAL", "DARNELL", "LATOYA", "TYRONE", "KEISHA", "MALIK",
    "TANISHA", "MARQUIS", "DESHAWN", "SHANICE", "TREVON",
    # Vietnamese / SE-Asian / other (for name-order-swap scenarios)
    "NGUYEN", "THI", "MAI", "MINH", "TUAN", "HUONG", "LINH", "PHUONG",
    # Hawaiian (HI cluster)
    "KEANU", "LEILANI", "KAI", "MALIA", "NOA", "ALANA",
]
CURATED_LAST_NAMES = [
    "VELASQUEZ", "QUINTANILLA", "MALDONADO", "CONTRERAS", "ZAMORA", "CARRILLO",
    "GUZMAN", "ESPINOZA", "OROZCO", "MONTOYA", "AGUILAR", "FUENTES", "MEDINA",
    "VARGAS", "CASTILLO", "ROMERO", "SANTIAGO", "MARQUEZ", "CERVANTES",
    "PENA", "GALLEGOS", "VILLANUEVA", "BARAJAS", "ZAVALA", "CISNEROS",
    "OCONNELL", "FITZGERALD", "KOWALSKI", "NOWAK", "WOJCIK", "KAMINSKI",
    "SCHNEIDER", "OBRIEN", "MURPHY", "KELLY", "RYAN", "SULLIVAN", "GALLAGHER",
    "WASHINGTON", "JEFFERSON", "FREEMAN", "DAWSON", "COLEMAN", "BANKS",
    "NGUYEN", "TRAN", "PHAM", "KAIMAKANI", "KAHELE", "KALANI",
]
CURATED_STREETS = [
    "SANGAMON", "LAWRENCE", "SHERIDAN", "SPAULDING", "SPRINGFIELD", "KEDZIE",
    "ALBANY", "LAWNDALE", "AVERS", "MONTICELLO", "WESTERN", "ASHLAND",
    "DAMEN", "HALSTED", "PULASKI", "CICERO", "CENTRAL", "CRAWFORD", "KOSTNER",
    "KARLOV", "KILDARE", "KOMENSKY", "TRIPP", "KEELER", "HAMLIN", "RIDGEWAY",
    "CALIFORNIA", "MOZART", "RICHMOND", "FRANCISCO", "SACRAMENTO", "WHIPPLE",
    "DIVISION", "CHICAGO", "GRAND", "HURON", "ERIE", "ONTARIO", "OHIO",
    "ARMITAGE", "FULLERTON", "DIVERSEY", "BELMONT", "ADDISON", "IRVING PARK",
    "MONTROSE", "WILSON", "FOSTER", "BRYN MAWR", "DEVON", "PETERSON", "TOUHY",
    "HOWARD", "PRATT", "MORSE", "ROGERS", "JARVIS", "GREENVIEW", "PAULINA",
    "WOLCOTT", "WINCHESTER", "MARSHFIELD", "RACINE", "THROOP", "LOOMIS",
    "MORGAN", "CARPENTER", "ABERDEEN", "MAY", "BISHOP", "LAFLIN", "ELSTON",
    "CLYBOURN", "MILWAUKEE", "OGDEN", "ARCHER", "CERMAK", "ROOSEVELT",
    "MADISON", "MONROE", "ADAMS", "JACKSON", "VAN BUREN", "CONGRESS",
    "HARRISON", "POLK", "TAYLOR", "ROOSEVELT", "18TH", "26TH", "31ST", "35TH",
    "47TH", "51ST", "55TH", "63RD", "71ST", "79TH", "87TH", "95TH",
]

# Bidirectional nickname map (M-NAME-09). Spanish pairs weighted given the
# population. Authored + hand-verified; the generator treats it symmetrically.
NICKNAMES = {
    "ROBERT": ["BOB", "ROB", "BOBBY", "ROBBIE"],
    "WILLIAM": ["BILL", "WILL", "BILLY", "WILLIE"],
    "ELIZABETH": ["BETH", "LIZ", "LIZZIE", "BETTY", "ELIZA"],
    "MICHAEL": ["MIKE", "MICKEY", "MICK"],
    "JOSEPH": ["JOE", "JOEY"],
    "JAMES": ["JIM", "JIMMY", "JAMIE"],
    "RICHARD": ["RICK", "DICK", "RICH", "RICKY"],
    "CHARLES": ["CHARLIE", "CHUCK", "CHAS"],
    "THOMAS": ["TOM", "TOMMY"],
    "CHRISTOPHER": ["CHRIS", "TOPHER"],
    "DANIEL": ["DAN", "DANNY"],
    "MATTHEW": ["MATT"],
    "ANTHONY": ["TONY"],
    "DAVID": ["DAVE", "DAVEY"],
    "JENNIFER": ["JEN", "JENNY", "JENNIE"],
    "JESSICA": ["JESS", "JESSIE"],
    "PATRICIA": ["PAT", "PATTY", "TRICIA"],
    "MARGARET": ["MAGGIE", "PEGGY", "MARGE"],
    "KATHERINE": ["KATE", "KATIE", "KATHY", "KAT"],
    "STEPHANIE": ["STEPH"],
    # Spanish
    "JOSE": ["PEPE", "CHE"],
    "FRANCISCO": ["PACO", "PANCHO", "FRANK"],
    "JESUS": ["CHUY", "CHUS"],
    "GUADALUPE": ["LUPE", "LUPITA"],
    "ROSARIO": ["CHARO", "CHAYO"],
    "DOLORES": ["LOLA", "LOLITA"],
    "MERCEDES": ["MECHE"],
    "ANTONIO": ["TONO", "TONY"],
    "MANUEL": ["MANNY", "MELO", "LOLO"],
    "EDUARDO": ["LALO", "EDDIE"],
    "ALEJANDRO": ["ALEX", "JANDRO"],
    "ROBERTO": ["BETO", "ROB"],
    "IGNACIO": ["NACHO"],
    "REFUGIO": ["CUCO"],
    "CONCEPCION": ["CONCHA", "CONCHITA"],
    "MARIA": ["MARI", "MIA"],
    "ROSA": ["ROSITA", "ROSIE"],
    "TERESA": ["TERE", "TERRY"],
}

# Initial -> plausible full first names (M-NAME-06 / §7 expand-initial). Authored;
# weighted toward names common in this population.
INITIAL_EXPANSION = {
    "A": ["ANNE", "ANNA", "ALICE", "ANGEL", "ANTONIO", "ANA", "ALEJANDRO"],
    "B": ["BARBARA", "BRIAN", "BEATRIZ", "BRENDA"],
    "C": ["CARLOS", "CARMEN", "CHRISTOPHER", "CAROLINA"],
    "D": ["DAVID", "DANIEL", "DIANA", "DOLORES"],
    "E": ["ELIZABETH", "EDUARDO", "ESPERANZA", "ERIC"],
    "F": ["FRANCISCO", "FERNANDO", "FRANCES", "FELIPE"],
    "G": ["GUADALUPE", "GABRIEL", "GLORIA", "GERARDO"],
    "H": ["HECTOR", "HELEN", "HUGO"],
    "I": ["ISABEL", "IVAN", "IRENE"],
    "J": ["JOSE", "JUAN", "JAMES", "JENNIFER", "JESUS", "JORGE"],
    "K": ["KEVIN", "KAREN", "KIMBERLY"],
    "L": ["LUIS", "LAURA", "LUCIA", "LEONARDO"],
    "M": ["MARIA", "MICHAEL", "MIGUEL", "MARGARET", "MANUEL"],
    "N": ["NANCY", "NICOLE", "NELSON"],
    "O": ["OSCAR", "OLIVIA", "OMAR"],
    "P": ["PATRICIA", "PEDRO", "PABLO", "PAULA"],
    "R": ["RICARDO", "ROSA", "ROBERT", "RAFAEL", "RAMON"],
    "S": ["SANDRA", "SERGIO", "SOFIA", "SAMUEL"],
    "T": ["TERESA", "THOMAS", "TANIA"],
    "V": ["VICTOR", "VERONICA", "VANESSA"],
    "W": ["WILLIAM", "WALTER", "WENDY"],
}


def _weighted_from_stats(stats: dict, field: str) -> list[list]:
    """Return [[VALUE, count], ...] from the measured top-N for a field,
    dropping any value that would trip the cleaning invalid-strings guard."""
    from generate_synthetic import is_clean_token  # local import to share rules

    top = stats.get("categorical_top", {}).get(field, {}).get("top", {})
    out = []
    for value, count in top.items():
        if is_clean_token(value):
            out.append([value, int(count)])
    return out


def build(stats_path: Path, out_dir: Path) -> None:
    stats = json.loads(stats_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    first_weighted = _weighted_from_stats(stats, "FirstNM_clean")
    last_weighted = _weighted_from_stats(stats, "LastNM_clean")

    first = {
        "weighted_head": first_weighted,
        "tail": sorted(set(CURATED_FIRST_NAMES)),
        # by_year is populated only by the optional SSA enrichment; empty here.
        # The generator falls back to the flat weighted+tail draw, which already
        # reflects this population's real age mix.
        "by_year": {},
    }
    last = {
        "weighted_head": last_weighted,
        "tail": sorted(set(CURATED_LAST_NAMES)),
    }

    # Street stems: measured non-suffix last-tokens (real Chicago street names)
    # + curated supplement.
    measured_tokens = stats.get("address", {}).get("addrline1_last_token_top", {})
    street_stems = {
        tok for tok in measured_tokens
        if tok not in STREET_SUFFIX_TOKENS and not re.fullmatch(r"\d+", tok)
    }
    street_stems |= set(CURATED_STREETS)
    streets = {"names": sorted(street_stems)}

    _write(out_dir / "first_names.json", first)
    _write(out_dir / "last_names.json", last)
    _write(out_dir / "streets.json", streets)
    _write(out_dir / "nicknames.json", {k: v for k, v in sorted(NICKNAMES.items())})
    _write(out_dir / "initial_expansion.json",
           {k: v for k, v in sorted(INITIAL_EXPANSION.items())})

    print(f"Built pools in {out_dir}:")
    print(f"  first_names : {len(first_weighted)} head + {len(first['tail'])} tail")
    print(f"  last_names  : {len(last_weighted)} head + {len(last['tail'])} tail")
    print(f"  streets     : {len(streets['names'])} stems")
    print(f"  nicknames   : {len(NICKNAMES)} canonical keys")
    print(f"  initials    : {len(INITIAL_EXPANSION)} letters")


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    p.add_argument("--stats", default=str(here / "synthetic_data_stats.json"))
    p.add_argument("--out", default=str(here / "pools"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build(Path(args.stats), Path(args.out))


if __name__ == "__main__":
    main()
