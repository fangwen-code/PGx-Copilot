"""
CPIC structured query interface.

Provides exact lookup of genotype -> phenotype -> recommendation
from the CPIC SQLite database (built by parser.py).
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from config import DATA_DIR

DB_PATH = DATA_DIR / "cpic.db"


class CPICQuery:
    """Query the CPIC SQLite database for structured drug-gene information."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = str(db_path)

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

    def match_genotype(
        self, gene: str, genotype: str
    ) -> list[dict]:
        """
        Find recommendations matching a specific gene + genotype.

        Example: match_genotype("CYP2D6", "*4/*4")
        Returns recommendations whose lookupkey contains this gene+genotype.
        """
        pattern = f"%{gene}%{genotype}%"
        conn = self._connect()
        cur = conn.execute(
            "SELECT * FROM recommendation WHERE lookupkey LIKE ?",
            (pattern,),
        )
        rows = cur.fetchall()
        conn.close()
        results = [self._row_to_dict("recommendation", r) for r in rows]

        # Filter for exact genotype match in the JSON
        filtered = []
        for r in results:
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

    def get_gene_result(self, gene_symbol: str) -> list[dict]:
        """Get consultation text for gene results."""
        conn = self._connect()
        cur = conn.execute(
            "SELECT * FROM gene_result WHERE genesymbol = ?", (gene_symbol,)
        )
        rows = cur.fetchall()
        conn.close()
        return [self._row_to_dict("gene_result", r) for r in rows]

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
