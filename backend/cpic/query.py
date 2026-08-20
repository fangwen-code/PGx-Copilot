"""
CPIC structured query interface.

Provides exact lookup of genotype -> phenotype -> recommendation
from the CPIC SQLite database (built by parser.py).
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

from config import DATA_DIR

DB_PATH = DATA_DIR / "cpic.db"


class CPICQuery:
    """Query the CPIC SQLite database for structured drug-gene information."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = str(db_path)
        self._score_pheno_map: dict | None = None

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def get_drug_info(self, drug_name: str) -> list[dict]:
        """Look up a drug by name."""
        conn = self._connect()
        cur = conn.execute(
            "SELECT * FROM drug WHERE name LIKE ?", (f"%{drug_name}%",)
        )
        rows = cur.fetchall()
        conn.close()
        return [self._row_to_dict("drug", r) for r in rows]

    def get_gene_info(self, gene_symbol: str) -> list[dict]:
        """Look up a gene by symbol."""
        conn = self._connect()
        cur = conn.execute(
            "SELECT * FROM gene WHERE symbol LIKE ?", (f"%{gene_symbol}%",)
        )
        rows = cur.fetchall()
        conn.close()
        return [self._row_to_dict("gene", r) for r in rows]

    def get_alleles(self, gene_symbol: str) -> list[dict]:
        """Get all alleles for a gene."""
        conn = self._connect()
        cur = conn.execute(
            "SELECT * FROM allele WHERE genesymbol = ?", (gene_symbol,)
        )
        rows = cur.fetchall()
        conn.close()
        return [self._row_to_dict("allele", r) for r in rows]

    def get_recommendation(
        self, drug_name: Optional[str] = None, gene_symbol: Optional[str] = None
    ) -> list[dict]:
        """
        Get recommendations filtered by drug name and/or gene.

        Since the recommendation table stores genotypes as a JSON lookupkey
        (e.g. {'CYP2D6': '*4/*4'}), exact genotype matching is done in application code.
        This method returns all relevant recommendations for further filtering.
        """
        conn = self._connect()
        if drug_name and gene_symbol:
            cur = conn.execute(
                """SELECT r.* FROM recommendation r
                   JOIN drug d ON r.drugid = d.drugid
                   WHERE d.name LIKE ? AND r.lookupkey LIKE ?""",
                (f"%{drug_name}%", f"%{gene_symbol}%"),
            )
        elif drug_name:
            cur = conn.execute(
                """SELECT r.* FROM recommendation r
                   JOIN drug d ON r.drugid = d.drugid
                   WHERE d.name LIKE ?""",
                (f"%{drug_name}%",),
            )
        elif gene_symbol:
            cur = conn.execute(
                "SELECT * FROM recommendation WHERE lookupkey LIKE ?",
                (f"%{gene_symbol}%",),
            )
        else:
            cur = conn.execute("SELECT * FROM recommendation LIMIT 50")
        rows = cur.fetchall()
        conn.close()
        return [self._row_to_dict("recommendation", r) for r in rows]

    _CLASS_RANK = {"strong": 2, "moderate": 1, "no recommendation": 0, "optional": 0}

    def match_genotype(
        self, gene: str, genotype: str, min_classification: str = "moderate",
        known_genes: set[str] | None = None,
    ) -> list[dict]:
        """
        Find recommendations matching a specific gene + genotype.

        Filters to Level A/B (classification >= moderate) by default.

        Pass 1 — exact allele-diplotype match on lookupkey (e.g. warfarin's
        lookupkey {'CYP2C9': '*1/*3'}).
        Pass 2 — guidelines keyed by ACTIVITY SCORE / PHENOTYPE instead of the
        allele (e.g. CYP2D6-metoprolol stores lookupkey {'CYP2D6': '1.0'} and
        phenotypes 'Intermediate Metabolizer'). The diplotype is translated to
        an activity score / phenotype and matched again. Rows that need extra
        genes the user did not provide (e.g. {'CYP2D6': '0.0', 'CYP2C19':
        'Normal Metabolizer'} for a CYP2D6-only query) are ranked last.

        Example: match_genotype("CYP2D6", "*4/*4")
        """
        accepted = self._classification_levels(min_classification)
        known = known_genes or {gene}
        conn = self._connect()

        # Pass 1: exact allele diplotype.
        pattern = f"%{gene}%{genotype}%"
        cur = conn.execute(
            "SELECT * FROM recommendation WHERE lookupkey LIKE ?",
            (pattern,),
        )
        results = [self._row_to_dict("recommendation", r) for r in cur.fetchall()]
        filtered = self._filter_exact_genotype(results, gene, genotype, accepted)

        # Pass 2: activity-score / phenotype keyed recommendations.
        if not filtered:
            score = self._diplotype_activity_score(gene, genotype)
            pheno = self._score_to_phenotype(gene, score) if score is not None else ""
            if score is not None or pheno:
                cur = conn.execute(
                    "SELECT * FROM recommendation WHERE lookupkey LIKE ? OR phenotypes LIKE ?",
                    (f"%{gene}%", f"%{pheno}%" if pheno else "%%"),
                )
                results = [self._row_to_dict("recommendation", r) for r in cur.fetchall()]
                filtered = self._filter_score_or_phenotype(results, gene, score, pheno, known)

        # Pass 3: FUNCTION fallback — the last rung before declaring no match.
        # CPIC function-based genes (SLCO1B1 / ABCG2 statin rows keyed by
        # "Decreased Function" etc.) are unreachable by star allele or activity
        # score, so derive the diplotype's functional label from the alleles'
        # functional status and match those rows directly. If this fails, there
        # is truly no recommendation for this genotype.
        if not filtered:
            func = self._diplotype_function(gene, genotype)
            if func:
                cur = conn.execute(
                    "SELECT * FROM recommendation WHERE lookupkey LIKE ?",
                    (f"%{gene}%",),
                )
                results = [self._row_to_dict("recommendation", r) for r in cur.fetchall()]
                filtered = self._filter_function(results, gene, func, known)
        conn.close()
        return filtered

    def _filter_exact_genotype(self, results, gene, genotype, accepted):
        """Pass-1 filter: the lookupkey's gene value equals the allele diplotype."""
        filtered = []
        for r in results:
            if r.get("classification", "").lower() not in accepted:
                continue
            lk = r.get("lookupkey", "")
            if not lk:
                continue
            try:
                lk_dict = json.loads(lk.replace("'", '"'))
                if lk_dict.get(gene, "").lower() == genotype.lower():
                    filtered.append(r)
            except json.JSONDecodeError:
                if genotype.lower() in lk.lower():
                    filtered.append(r)
        return filtered

    def _diplotype_activity_score(self, gene, diplotype):
        """Sum the two alleles' activity values (CPIC activity-score model)."""
        parts = [p.strip() for p in str(diplotype).split("/")]
        if len(parts) != 2:
            return None
        total, found = 0.0, 0
        for p in parts:
            val = self._allele_activity(gene, p)
            if val is not None:
                total += val
                found += 1
        return total if found == 2 else None

    def _allele_activity(self, gene, allele_name):
        """Return an allele's activity value, matching by star-allele name."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT name, activityvalue FROM allele WHERE genesymbol = ?",
            (gene,),
        ).fetchall()
        conn.close()
        target = str(allele_name).strip().upper()
        for name, value in rows:
            n = str(name or "").upper()
            if "*" in n and gene.upper() in n.split("*")[0]:
                n = "*" + n.split("*", 1)[1]
            if n == target or n.lstrip("*") == target.lstrip("*"):
                return float(value) if value is not None else None
        return None

    def _load_score_phenotype_map(self) -> dict:
        """Extract gene → {activity-score: phenotype} straight from the DB.

        `activityscore` / `phenotypes` are per-gene JSON dicts (single- AND
        multi-gene rows), so every gene on every row contributes its own
        score→phenotype pair. Reading the columns directly (no hardcoded
        thresholds) gives the dump's own assignments, e.g. CYP2C19 1.0 →
        "Intermediate Metabolizer" from a clopidogrel row. Cached after first
        use.
        """
        if self._score_pheno_map is not None:
            return self._score_pheno_map
        mapping: dict[str, dict[str, str]] = {}
        conn = self._connect()
        rows = conn.execute(
            "SELECT lookupkey, activityscore, phenotypes FROM recommendation"
        ).fetchall()
        conn.close()
        for lk, ascore, phenos in rows:
            if not lk:
                continue
            try:
                lk_dict = json.loads(lk.replace("'", '"'))
            except json.JSONDecodeError:
                continue
            multi = len(lk_dict) > 1
            for gene in lk_dict:
                score_val = _as_value(ascore, gene, multi)
                pheno_val = _as_value(phenos, gene, multi)
                if score_val is None or pheno_val is None:
                    continue
                if (str(score_val).strip().lower() == "n/a"
                        or str(pheno_val).strip().lower() == "n/a"):
                    continue
                mapping.setdefault(gene, {})[str(score_val)] = str(pheno_val)
        self._score_pheno_map = mapping
        return mapping

    def _score_to_phenotype(self, gene, score):
        """Map an activity score to a CPIC phenotype keyword.

        Pure lookup into the DB's own (activityscore, phenotypes) columns. The
        score is one "format" the input genotype was translated into; this
        yields the phenotype format so Pass 2 can compare it against the DB's
        fixed lookupkey value (case-insensitively). "" when the gene has no
        score→phenotype rows — the cascade falls through to function / no-match.
        """
        if score is None:
            return ""
        return self._load_score_phenotype_map().get(gene, {}).get(str(score), "")

    def _filter_score_or_phenotype(self, results, gene, score, pheno, known):
        """Pass-2 filter: match lookupkey's activity score or phenotypes field.

        Rows that require a gene the user did NOT provide are dropped entirely
        (e.g. {'CYP2D6': '0.0', 'CYP2C19': 'Normal Metabolizer'} for a
        CYP2D6-only query). Remaining rows are sorted by classification
        strength so the strongest recommendation wins.
        """
        matched = []
        for r in results:
            lk = r.get("lookupkey", "")
            ph = (r.get("phenotypes", "") or "").lower()
            hit = False
            if score is not None and lk:
                try:
                    lk_dict = json.loads(lk.replace("'", '"'))
                    hit = self._scores_equal(lk_dict.get(gene, ""), score)
                except (json.JSONDecodeError, TypeError):
                    hit = str(score) in lk
            if not hit and pheno:
                hit = pheno.lower() in ph or pheno.lower() in lk.lower()
            if hit:
                try:
                    lk_dict = json.loads((lk or "").replace("'", '"'))
                    extra = set(lk_dict) - set(known)
                    if extra and not all(
                        str(lk_dict[g]).lower() in ("indeterminate", "no result")
                        for g in extra
                    ):
                        # requires a non-wildcard gene the user didn't provide
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                matched.append(r)
        matched.sort(
            key=lambda r: -self._CLASS_RANK.get(r.get("classification", "").lower(), -1)
        )
        return matched

    def _allele_function(self, gene: str, allele_name: str) -> str | None:
        """Return an allele's CPIC functional status (e.g. "Decreased function")."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT name, functionalstatus FROM allele WHERE genesymbol = ?",
            (gene,),
        ).fetchall()
        conn.close()
        target = str(allele_name).strip().upper()
        for name, status in rows:
            n = str(name or "").upper()
            if "*" in n and gene.upper() in n.split("*")[0]:
                n = "*" + n.split("*", 1)[1]
            if n == target or n.lstrip("*") == target.lstrip("*"):
                return str(status or "").strip() or None
        return None

    def _diplotype_function(self, gene: str, diplotype: str) -> str | None:
        """Derive a diplotype's functional label from its two alleles' status.

        CPIC precedence: any "No function" → No Function; else any "Decreased
        function" → Decreased Function; else any "Increased function" →
        Increased Function; else Normal Function. None when the alleles'
        functional status is unavailable (function fallback cannot apply).
        """
        parts = [p.strip() for p in str(diplotype).split("/")]
        if len(parts) != 2:
            return None
        statuses = [self._allele_function(gene, p) for p in parts]
        statuses = [s for s in statuses if s]
        if len(statuses) != 2:
            return None
        joined = " ".join(s.lower() for s in statuses)
        if "no function" in joined:
            return "No Function"
        if "decreased function" in joined:
            return "Decreased Function"
        if "increased function" in joined:
            return "Increased Function"
        return "Normal Function"

    def _filter_function(self, results, gene, func: str, known: set[str]) -> list[dict]:
        """Pass-3 filter: rows whose lookupkey value for `gene` equals the
        derived functional label. Rows requiring a gene the user did not provide
        are kept only when that extra gene's value is a wildcard
        ("Indeterminate" / "No Result") — otherwise they'd need a genotype the
        query didn't supply.
        """
        matched = []
        accepted = self._classification_levels("moderate")
        for r in results:
            if r.get("classification", "").lower() not in accepted:
                continue
            lk = r.get("lookupkey", "")
            if not lk:
                continue
            try:
                lk_dict = json.loads(lk.replace("'", '"'))
            except json.JSONDecodeError:
                continue
            if (lk_dict.get(gene, "") or "").lower() != func.lower():
                continue
            extra = set(lk_dict) - set(known)
            if extra:
                if not all(str(lk_dict[g]).lower() in ("indeterminate", "no result")
                           for g in extra):
                    continue
            matched.append(r)
        matched.sort(
            key=lambda r: -self._CLASS_RANK.get(r.get("classification", "").lower(), -1)
        )
        return matched

    def _scores_equal(self, val, score):
        try:
            return abs(float(val) - float(score)) < 1e-6
        except (TypeError, ValueError):
            return str(val).strip() == str(score).strip()

    def match_multi_genotype(self, genotypes: dict[str, str], min_classification: str = "moderate") -> list[dict]:
        """
        Match recommendations where ALL gene:genotype pairs match simultaneously.
        Filters to Level A/B (classification >= moderate) by default.
        """
        accepted = self._classification_levels(min_classification)
        conn = self._connect()
        cur = conn.execute("SELECT * FROM recommendation")
        rows = cur.fetchall()
        conn.close()

        matched = []
        for row in rows:
            r = self._row_to_dict("recommendation", row)
            if r.get("classification", "").lower() not in accepted:
                continue
            lk = r.get("lookupkey", "")
            if not lk:
                continue
            try:
                lk_dict = json.loads(lk.replace("'", '"'))
                all_match = all(
                    lk_dict.get(gene, "").lower() == gt.lower()
                    for gene, gt in genotypes.items()
                )
                if all_match:
                    matched.append(r)
            except json.JSONDecodeError:
                pass

        return matched

    def match_drug_genotypes(self, drug: str, genotypes: dict[str, str],
                             min_classification: str = "moderate") -> list[dict]:
        """Drug-first matching — the primary CPIC lookup path.

        Inspect the drug's OWN recommendation keys, classify each key value's
        type (star allele / activity score / phenotype / function / descriptive),
        convert the input genotypes to that type, and match. Working from the
        drug's actual key structure handles multi-gene keys naturally (genes
        the user did not provide don't block a match) and only ever converts to
        the format the DB really uses. Returns rows for this exact drug.
        """
        accepted = self._classification_levels(min_classification)
        matched = []
        for r in self.get_recommendation(drug_name=drug):
            if r.get("classification", "").lower() not in accepted:
                continue
            lk = r.get("lookupkey", "")
            if not lk:
                continue
            try:
                lk_dict = json.loads(lk.replace("'", '"'))
            except json.JSONDecodeError:
                continue
            ok = True
            for gene, gt in genotypes.items():
                row_val = lk_dict.get(gene)
                if row_val is None:
                    ok = False
                    break
                conv = self._convert_to_type(gene, gt, _value_type(str(row_val)))
                if conv is None or str(conv).lower() != str(row_val).lower():
                    ok = False
                    break
            if ok:
                matched.append(r)
        matched.sort(
            key=lambda r: -self._CLASS_RANK.get(r.get("classification", "").lower(), -1)
        )
        return matched

    def _convert_to_type(self, gene: str, input_gt, target_type: str) -> str | None:
        """Convert an input genotype into the DB key's value type (or None)."""
        val = self._input_representations(gene, input_gt).get(target_type)
        return val if val not in (None, "") else None

    def _input_representations(self, gene: str, genotype) -> dict[str, str | None]:
        """All formats of an input genotype: star / score / phenotype / function.

        Which formats get populated depends on what the input is — a star
        allele derives score + phenotype + function; a score derives phenotype;
        a phenotype/function/descriptive string is used as-is. The drug-first
        matcher picks the representation matching the DB key's value type.
        """
        g = str(genotype).strip()
        reps: dict[str, str | None] = {}
        vtype = _value_type(g)
        if vtype == "score":
            score = float(g)
            reps["score"] = g
            reps["phenotype"] = self._score_to_phenotype(gene, score)
        elif vtype == "star":
            reps["star"] = g
            score = self._diplotype_activity_score(gene, g)
            reps["score"] = str(score) if score is not None else None
            reps["phenotype"] = self._score_to_phenotype(gene, score) if score is not None else ""
            reps["function"] = self._diplotype_function(gene, g)
        else:  # phenotype / function / descriptive string — used as-is
            reps["phenotype"] = g
            reps["function"] = g
            reps["descriptive"] = g
        return reps

    def get_gene_result(self, gene_symbol: str) -> list[dict]:
        """Get consultation text for gene results."""
        conn = self._connect()
        cur = conn.execute(
            "SELECT * FROM gene_result WHERE genesymbol = ?", (gene_symbol,)
        )
        rows = cur.fetchall()
        conn.close()
        return [self._row_to_dict("gene_result", r) for r in rows]

    def _classification_levels(self, min_level: str = "moderate") -> set[str]:
        """Return accepted classification values at or above the given level."""
        hierarchy = {"optional": 0, "moderate": 1, "strong": 2}
        return {k for k, v in hierarchy.items() if v >= hierarchy.get(min_level, 0)}

    def get_cited_pmid_entries(self) -> list[dict]:
        """
        Return all gene-PMID pairs cited across the CPIC pair table.

        PMIDs come from `pair.citations` (a PG array like '{23486447,27997040}').
        All levels are included — the abstracts serve as background knowledge,
        not clinical decision input, so no level filtering is applied.
        Removed associations are excluded (they may be deprecated).
        """
        conn = self._connect()
        cur = conn.execute("""
            SELECT p.genesymbol, p.citations, p.cpiclevel, d.name
            FROM pair p
            LEFT JOIN drug d ON p.drugid = d.drugid
            WHERE (p.removed IS NULL OR p.removed = '' OR p.removed = 'f')
        """)
        rows = cur.fetchall()
        conn.close()

        entries = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            gene = row[0] or ""
            citations = row[1] or ""
            level = row[2] or ""
            drug = row[3] or ""

            if not gene:
                continue

            # Parse PG array format: "{23486447,27997040}" or "\N" (NULL)
            pmids = _parse_pg_array(citations)
            if not pmids:
                continue

            for pmid in pmids:
                key = (pmid, gene)
                if key not in seen:
                    seen.add(key)
                    entries.append({
                        "pmid": pmid,
                        "gene": gene,
                        "drug": drug,
                        "cpic_level": level,
                    })

        return entries

    def find_drugs_by_gene(self, gene_symbol: str) -> list[str]:
        """Find all drugs associated with a gene via recommendations."""
        conn = self._connect()
        cur = conn.execute(
            """SELECT DISTINCT d.name FROM recommendation r
               JOIN drug d ON r.drugid = d.drugid
               WHERE r.lookupkey LIKE ?""",
            (f"%{gene_symbol}%",),
        )
        return [r[0] for r in cur.fetchall() if r[0]]

    # ── helpers ────────────────────────────────────

    def _row_to_dict(self, table: str, row) -> dict:
        columns = _SCHEMAS.get(table, [])
        return {c: (v if v is not None else "") for c, v in zip(columns, row)}

    def close(self):
        pass


def _value_type(value: str) -> str:
    """Classify a CPIC key value: star | score | phenotype | function | descriptive.

    Drives the drug-first matcher's conversion target. Heuristic over the
    value's shape: decimal → activity score; `*N` / `N/N` → star allele;
    "Metabolizer"/"Deficient" → phenotype; "Function"/"Indeterminate"/
    "No Result"/"Uncertain" → function; everything else (HLA "*31:01 negative",
    CFTR "ivacaftor responsive in CF patients", …) → descriptive.
    """
    v = str(value).strip()
    if re.fullmatch(r"\d+\.\d+", v):
        return "score"
    if re.fullmatch(r"\*?\d+(?:\s*/\s*\*?\d+)?", v):
        return "star"
    if re.search(r"metabolizer|deficient", v, re.IGNORECASE):
        return "phenotype"
    if re.search(r"function", v, re.IGNORECASE) or v.lower() in (
        "indeterminate", "no result", "uncertain", "uncertain susceptibility"):
        return "function"
    return "descriptive"


def _as_value(cell, gene: str, multi: bool = False):
    """Read a recommendation column cell as the value for `gene`.

    `activityscore` / `phenotypes` are per-gene JSON dicts on multi-gene rows
    (e.g. {"CYP2C19": "1.0"}) and may be plain scalars on single-gene rows.
    Returns the per-gene value or the scalar; a scalar on a multi-gene row is
    ambiguous, so it returns None there. None also when unparseable/absent.
    """
    if not cell:
        return None
    s = str(cell).strip()
    if s.startswith("{"):
        try:
            d = json.loads(s.replace("'", '"'))
            if isinstance(d, dict):
                return d.get(gene)
        except json.JSONDecodeError:
            return None
    return None if multi else s


def _parse_pg_array(value: str) -> list[str]:
    """Parse a PostgreSQL array literal like '{23486447,27997040}' into a list.

    Returns empty list for NULL, empty string, or non-array values.
    """
    if not value or value == r"\N":
        return []
    value = value.strip()
    if not (value.startswith("{") and value.endswith("}")):
        return []
    inner = value[1:-1].strip()
    if not inner:
        return []
    # Split on commas; strip quotes and whitespace
    return [part.strip().strip('"').strip("'") for part in inner.split(",") if part.strip()]


_SCHEMAS = {
    "drug": ["drugid", "name", "pharmgkbid"],
    "gene": ["symbol", "chr", "lookupmethod"],
    "allele": ["id", "genesymbol", "name", "functionalstatus",
               "clinicalfunctionalstatus", "activityvalue", "citations"],
    "recommendation": ["id", "guidelineid", "drugid", "lookupkey",
                       "implications", "drugrecommendation", "classification",
                       "phenotypes", "activityscore", "allelestatus",
                       "population", "comments"],
    "guideline": ["id", "name", "url", "pmid", "source"],
    "gene_result": ["id", "genesymbol", "result", "activityscore",
                    "ehrpriority", "consultationtext", "frequency"],
}
