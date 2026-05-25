"""
CPIC data coverage inventory.

Answers: "What drugs and genes are covered by your system?"
Lists all drugs, genes, chunk counts, and uncovered query patterns.
"""

import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DATA_DIR / "cpic.db"


def show_coverage():
    """Print data coverage report."""
    if not DB_PATH.exists():
        print(f"[WARN] CPIC DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))

    print("=" * 60)
    print("CPIC Data Coverage Report")
    print("=" * 60)

    # 1. Drugs with recommendations
    print("\n1. Drugs with CPIC recommendations:")
    cur = conn.execute("""
        SELECT DISTINCT d.name, COUNT(r.id) as rec_count
        FROM drug d
        JOIN recommendation r ON r.drugid = d.drugid
        GROUP BY d.name
        ORDER BY d.name
    """)
    drugs = cur.fetchall()
    for name, count in drugs:
        print(f"   {name:30s} ({count} recommendations)")

    # 2. Genes with recommendations
    print("\n2. Genes in recommendation lookup keys:")
    cur = conn.execute("""
        SELECT DISTINCT r.lookupkey
        FROM recommendation r
        WHERE r.lookupkey IS NOT NULL
    """)
    genes = set()
    import json
    for row in cur.fetchall():
        try:
            d = json.loads(row[0].replace("'", '"'))
            genes.update(d.keys())
        except Exception:
            pass
    for g in sorted(genes):
        print(f"   {g}")

    # 3. Summary
    print("\n3. Summary:")
    cur = conn.execute("SELECT COUNT(DISTINCT drugid) FROM recommendation")
    drug_count = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM recommendation")
    rec_count = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(DISTINCT symbol) FROM gene")
    gene_count = cur.fetchone()[0]
    print(f"   Drugs with recommendations: {drug_count}")
    print(f"   Total recommendations:      {rec_count}")
    print(f"   Genes in gene table:        {gene_count}")

    conn.close()
    print(f"\n   DB location: {DB_PATH}")


if __name__ == "__main__":
    show_coverage()
