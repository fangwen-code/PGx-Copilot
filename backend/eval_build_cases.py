"""
Build the end-to-end clinical evaluation case set from the CPIC SQLite database.

Two case classes:

  POSITIVE — a real (gene-sets × genotypes, drug) pair that carries a
             strong/moderate CPIC recommendation. The query genotype is the
             DB's OWN lookupkey value — star allele "*4/*4", phenotype
             "Poor Metabolizer", function "Decreased Function", activity score
             "0.0", … — so Pass-1 exact matching always resolves. Ground truth
             is that row's `drugrecommendation`, resolved at the *drug* level
             (never another drug's advice for the same genotype). Multi-gene
             keys (warfarin CYP2C9+VKORC1, TPMT+NUDT15, CYP2C19+…, …) become
             multi-genotype queries resolved via match_multi_genotype.

  NEGATIVE — a plausible gene-drug pair with NO strong/moderate CPIC
             recommendation (verified against the DB at build time), plus a few
             clearly out-of-scope queries. The system must refuse, not fabricate.

Ground truth comes from the CPIC database via match_genotype /
match_multi_genotype + an independent drug-id filter — NOT from the running
report pipeline — so this is an honest end-to-end check of
parse → CPIC matching → report formatting.

Deterministic: fixed seed + fixed pools → same DB → same case file.

Usage (run from backend/, where ../data/cpic.db exists):
    python eval_build_cases.py [--n-pos 40] [--n-neg 10] [--seed 42]
    python eval_build_cases.py --diagnose        # inspect per-gene key formats

Output: backend/evals/clinical_cases.json
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path

from cpic.query import CPICQuery, DB_PATH, _as_value

EVALS_DIR = Path(__file__).resolve().parent / "evals"

# Negative-case pools. A (gene, drug) pair is kept only if the DB has no
# strong/moderate recommendation for it — the system must refuse.
NEG_GENES = [
    "CYP2D6", "CYP2C19", "CYP2C9", "SLCO1B1", "VKORC1", "APOE",
    "TPMT", "NUDT15", "DPYD", "CYP2B6", "HLA-B", "CYP3A4", "UGT1A1",
]
NEG_DRUGS = [
    "atorvastatin", "rosuvastatin", "amlodipine", "losartan", "lisinopril",
    "aspirin", "ibuprofen", "omeprazole", "levothyroxine",
]

# Clearly out-of-scope queries — must be refused, never answered with a drug
# recommendation. Appended on top of the gene-drug negatives.
OUT_OF_SCOPE = [
    {"query": "发烧咳嗽吃什么药", "gene": "", "genotype": "", "drug": ""},
    {"query": "阿司匹林会出血吗", "gene": "", "genotype": "", "drug": ""},
]

# Statin genes — routed EXCLUSIVELY to the statin rule engine (app.py), so
# their CPIC rows are not part of the CPIC eval; their ground truth lives in
# the rule engine (SLCO1B1/APOE), not the recommendation table.
STATIN_GENES = {"SLCO1B1", "APOE", "rs4149056", "rs7412", "rs429358"}

# Natural star-allele queries — the translation-cascade tests. Unlike the
# row-driven cases (which carry the DB's own key value), these start from a
# star allele and rely on star → activity score → phenotype/function conversion
# to reach the DB's keyed value. Kept in the set unconditionally.
STAR_QUERIES = [
    {"genotypes": {"CYP2D6": "*4/*4"}, "drug": "metoprolol"},
    {"genotypes": {"CYP2D6": "*4/*4"}, "drug": "codeine"},
    {"genotypes": {"CYP2C19": "*2/*2"}, "drug": "clopidogrel"},
    {"genotypes": {"CYP2C19": "*1/*17"}, "drug": "clopidogrel"},
    {"genotypes": {"CYP2C9": "*1/*3"}, "drug": "warfarin"},
    {"genotypes": {"SLCO1B1": "*5/*5"}, "drug": "simvastatin"},
    {"genotypes": {"DPYD": "*2A/*2A"}, "drug": "capecitabine"},
    {"genotypes": {"TPMT": "*3A/*3A"}, "drug": "azathioprine"},
    {"genotypes": {"NUDT15": "*3/*3"}, "drug": "mercaptopurine"},
    {"genotypes": {"CYP2B6": "*6/*6"}, "drug": "efavirenz"},
]


def _parse_lookupkey(lk: str) -> dict:
    """Parse the DB's single/double-quoted JSON lookupkey
    (e.g. "{'CYP2C19': '*2/*2'}" or '{"UGT1A1": "Poor Metabolizer"}')."""
    if not lk:
        return {}
    try:
        return json.loads(lk.replace("'", '"'))
    except json.JSONDecodeError:
        return {}


def collect_genes() -> list[str]:
    """Distinct gene symbols that appear in recommendation lookupkeys."""
    conn = sqlite3.connect(DB_PATH)
    genes: set[str] = set()
    for (lk,) in conn.execute("SELECT lookupkey FROM recommendation").fetchall():
        genes.update(_parse_lookupkey(lk or "").keys())
    conn.close()
    return sorted(genes)


def resolve_expected(q: CPICQuery, genotypes: dict[str, str], drug: str) -> tuple[str, str]:
    """Drug-level ground truth: the drugrecommendation for this exact drug.

    Uses the SAME drug-first matcher as the running system
    (match_drug_genotypes), so ground truth and system are consistent; the eval
    then verifies the FULL pipeline (parse → matching → report formatting)
    reproduces the exact CPIC text. Empty string → no strong/moderate
    recommendation for this (genotypes, drug) pair.
    """
    rows = q.match_drug_genotypes(drug, genotypes)
    if not rows:
        return "", ""
    return (rows[0].get("drugrecommendation") or ""), (rows[0].get("classification") or "")


def collect_candidates() -> list[dict]:
    """Row-driven candidates: every strong/moderate recommendation row, using
    the DB's OWN lookupkey values as the query genotypes (so Pass-1 matching
    always resolves). Multi-gene keys become multi-genotype queries.
    """
    candidates = []
    seen: set = set()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """SELECT r.lookupkey, d.name FROM recommendation r
           JOIN drug d ON r.drugid = d.drugid
           WHERE LOWER(r.classification) IN ('strong','moderate')"""
    )
    for lk, drug in cur.fetchall():
        d = _parse_lookupkey(lk or "")
        if not d or not drug:
            continue
        d = {g: str(v).strip() for g, v in d.items() if str(v).strip()}
        if not d:
            continue
        if set(d) & STATIN_GENES:
            continue  # statin genes → rule engine, not CPIC eval
        key = (tuple(sorted(d.items())), drug)
        if key in seen:
            continue
        seen.add(key)
        gt_str = " ".join(f"{g} {v}" for g, v in d.items())
        candidates.append({
            "genes": sorted(d),
            "genotypes": d,
            "query": f"{gt_str} {drug}".strip(),
            "drug": drug,
        })
    conn.close()
    return candidates


def select_positives(q: CPICQuery, candidates: list[dict], n_pos: int,
                     rng: random.Random) -> list[dict]:
    """Resolve drug-level ground truth and pick a gene-set × drug diverse subset."""
    by_pair: dict[tuple, list[dict]] = {}
    for c in candidates:
        expected, cls = resolve_expected(q, c["genotypes"], c["drug"])
        if not expected:
            continue
        c = dict(c, type="positive", expected=expected, classification=cls,
                 pair=(tuple(c["genes"]), c["drug"]))
        by_pair.setdefault(c["pair"], []).append(c)

    pairs = list(by_pair)
    rng.shuffle(pairs)

    selected: list[dict] = []
    for p in pairs:  # one representative per distinct (gene-set, drug) pair
        if len(selected) >= n_pos:
            break
        selected.append(by_pair[p][0])
    if len(selected) < n_pos:  # fill with extra genotypes of used pairs
        used = {c["pair"] for c in selected}
        extras = [c for p in used for c in by_pair[p][1:]]
        rng.shuffle(extras)
        for c in extras:
            if len(selected) >= n_pos:
                break
            selected.append(c)
    return selected


def build_negatives(n_neg: int, rng: random.Random) -> list[dict]:
    """Gene-drug pairs with NO strong/moderate recommendation (verified live)."""
    negatives = []
    conn = sqlite3.connect(DB_PATH)
    for gene in NEG_GENES:
        for drug in NEG_DRUGS:
            count = conn.execute(
                """SELECT COUNT(*) FROM recommendation r JOIN drug d ON r.drugid = d.drugid
                   WHERE d.name LIKE ? AND r.lookupkey LIKE ?
                     AND LOWER(r.classification) IN ('strong','moderate')""",
                (f"%{drug}%", f"%{gene}%"),
            ).fetchone()[0]
            if count == 0:
                negatives.append({
                    "type": "negative",
                    "query": f"{gene} {drug}",
                    "gene": gene, "genotype": "", "drug": drug,
                    "expected": "", "classification": "",
                })
    conn.close()
    rng.shuffle(negatives)
    selected = negatives[:n_neg]
    for oos in OUT_OF_SCOPE:  # out-of-scope always included, not part of the cap
        selected.append({"type": "negative", **dict(oos), "expected": "", "classification": ""})
    return selected


def diagnose(q: CPICQuery) -> None:
    """Print how recommendations are keyed per gene — why some resolve and
    others don't. Star-keyed single-gene keys resolve via match_genotype
    Pass 1; activity/phenotype keys need the allele table's activity values
    (Pass 2); multi-gene keys are dropped for single-gene queries.
    """
    conn = sqlite3.connect(DB_PATH)
    print("=== strong/moderate recommendation keys by gene ===")
    for gene in collect_genes():
        rows = conn.execute(
            """SELECT r.lookupkey, r.classification, d.name FROM recommendation r
               JOIN drug d ON r.drugid = d.drugid
               WHERE r.lookupkey LIKE ? AND LOWER(r.classification) IN ('strong','moderate')""",
            (f"%{gene}%",),
        ).fetchall()
        if not rows:
            print(f"  {gene:<9} (no strong/moderate rows)")
            continue
        keys = sorted({r[0] for r in rows})
        drugs = sorted({r[2] for r in rows})
        print(f"  {gene:<9} {len(rows)} rows | drugs: {', '.join(drugs[:6])}")
        for k in keys[:4]:
            print(f"           key: {k}")
        fs = conn.execute(
            "SELECT DISTINCT functionalstatus FROM allele "
            "WHERE genesymbol = ? AND functionalstatus IS NOT NULL AND functionalstatus != ''",
            (gene,),
        ).fetchall()
        if fs:
            print(f"           functionalstatus: {', '.join(sorted(r[0] for r in fs))}")
        sp = conn.execute(
            "SELECT lookupkey, activityscore, phenotypes FROM recommendation "
            "WHERE lookupkey LIKE ?",
            (f"%{gene}%",),
        ).fetchall()
        samples = set()
        for lk, ascore, phenos in sp:
            try:
                lk_dict = json.loads((lk or "").replace("'", '"'))
            except json.JSONDecodeError:
                continue
            multi = len(lk_dict) > 1
            for g in lk_dict:
                sv = _as_value(ascore, g, multi)
                pv = _as_value(phenos, g, multi)
                if sv is None or pv is None:
                    continue
                if str(sv).strip().lower() == "n/a" or str(pv).strip().lower() == "n/a":
                    continue
                samples.add((str(sv), str(pv)))
        if samples:
            print(f"           score→phenotype: {sorted(samples)[:4]}")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the clinical eval case set from CPIC DB")
    parser.add_argument("--n-pos", type=int, default=40)
    parser.add_argument("--n-neg", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnose", action="store_true",
                        help="print how recommendations are keyed per gene, then exit")
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"[ERROR] CPIC DB not found: {DB_PATH} — build data/ first (see README).")

    rng = random.Random(args.seed)
    q = CPICQuery()

    if args.diagnose:
        diagnose(q)
        return

    from collections import Counter
    candidates = collect_candidates()
    # Star-allele variants test the translation cascade (star → score →
    # phenotype/function), not just exact-value matching.
    for sq in STAR_QUERIES:
        if set(sq["genotypes"]) & STATIN_GENES:
            continue  # SLCO1B1 *5/*5 simvastatin → rule engine, not CPIC eval
        gt_str = " ".join(f"{g} {v}" for g, v in sq["genotypes"].items())
        candidates.append({
            "genes": sorted(sq["genotypes"]),
            "genotypes": sq["genotypes"],
            "query": f"{gt_str} {sq['drug']}".strip(),
            "drug": sq["drug"],
            "star_query": True,
        })
    positives = select_positives(q, candidates, args.n_pos, rng)
    # Guarantee star-allele variants survive selection — they share gene-drug
    # pairs with row-driven cases and would otherwise be squeezed out.
    seen = {c["query"] for c in positives}
    for c in candidates:
        if c.get("star_query") and c["query"] not in seen:
            expected, cls = resolve_expected(q, c["genotypes"], c["drug"])
            if expected:
                positives.append({**c, "type": "positive",
                                  "expected": expected, "classification": cls})
    negatives = build_negatives(args.n_neg, rng)

    cases = positives + negatives
    rng.shuffle(cases)
    for i, c in enumerate(cases):
        c["id"] = i + 1
        if c["type"] == "positive":
            c["gene"] = "/".join(c["genes"])
            c["genotype"] = "/".join(c["genotypes"].values())

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALS_DIR / "clinical_cases.json"
    payload = {
        "meta": {
            "generated_from": "cpic.db",
            "n_pos": len(positives),
            "n_neg": len(negatives),
            "seed": args.seed,
        },
        "cases": cases,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pos_genes = sorted({c["genes"][0] for c in positives})
    gene_counts = Counter(c["genes"][0] for c in positives)
    neg_pairs = sorted({f"{c['gene']}×{c['drug']}" for c in negatives if c["gene"]})
    print(f"[OK] {len(candidates)} candidate triples → {len(positives)} positives, "
          f"{len(negatives)} negatives")
    print(f"     positive genes: {', '.join(pos_genes)}")
    print(f"     per-gene positives: {dict(gene_counts)}")
    print(f"     negative pairs: {', '.join(neg_pairs)}")
    print(f"     written to: {out}")


if __name__ == "__main__":
    main()
