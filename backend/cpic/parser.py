"""
CPIC database dump parser.

Parses PostgreSQL COPY-format dump into:
  1. SQLite database for structured queries (genotype -> phenotype -> recommendation)
  2. RAG text chunks with hierarchical heading context for semantic retrieval

Chunk design (hierarchical):
  Each chunk carries its full context path as metadata + heading hierarchy,
  ensuring it's a complete, self-contained information unit.

Usage:
    python -m cpic.parser --dump path/to/cpic_db_dump-v1.53.2.sql
    python -m cpic.parser --dump path/to/dump.sql --reset
"""

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from config import DATA_DIR
from rag.vector_store import VectorStore

DB_PATH = DATA_DIR / "cpic.db"


# ── SQL COPY Parser ──────────────────────────────────────────────────

def parse_copy_data(sql_path: str, validate_only: bool = False) -> dict[str, dict]:
    pat = re.compile(r"COPY cpic\.(\w+)\s+\(([^)]+)\)\s+FROM stdin;")
    tables = {}
    current_table = None
    parse_errors = []
    with open(sql_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            m = pat.search(line)
            if m:
                current_table = m.group(1)
                cols = [c.strip() for c in m.group(2).split(",")]
                tables[current_table] = {"columns": cols, "rows": []}
                continue
            if current_table and line.strip() == r"\.":
                current_table = None
                continue
            if current_table and line.strip():
                try:
                    tables[current_table]["rows"].append(
                        _parse_copy_line(line, len(tables[current_table]["columns"]))
                    )
                except Exception as e:
                    parse_errors.append(f"  line {line_no} ({current_table}): {e}")
                    if not validate_only:
                        # skip bad row, continue
                        pass

    print(f"Parsed {len(tables)} tables:")
    for name, t in tables.items():
        print(f"  {name}: {len(t['rows'])} rows")
    if parse_errors:
        print(f"\nParse errors ({len(parse_errors)}):")
        for err in parse_errors[:20]:
            print(err)
        if len(parse_errors) > 20:
            print(f"  ... and {len(parse_errors) - 20} more")
    return tables


def _parse_copy_line(line: str, num_cols: int) -> list:
    vals = []
    i = 0
    while i < len(line) and len(vals) < num_cols:
        if line[i] == "\t" or (line[i] == "\n" and vals):
            if not vals:
                vals.append("")
            if line[i] == "\n":
                break
            i += 1
            continue
        if line[i] == "\n":
            if not vals:
                vals.append("")
            break
        if line[i:i+2] == r"\N":
            vals.append(None)
            i += 2
            if i < len(line) and line[i] == "\t":
                i += 1
            continue
        val = ""
        while i < len(line) and line[i] not in ("\t", "\n"):
            if line[i] == "\\":
                i += 1
                val += {"t": "\t", "n": "\n"}.get(line[i], line[i]) if i < len(line) else ""
                i += 1
            else:
                val += line[i]
                i += 1
        vals.append(val)
        if i < len(line) and line[i] == "\t":
            i += 1
    while len(vals) < num_cols:
        vals.append(None)
    return vals[:num_cols]


# ── SQLite Builder ───────────────────────────────────────────────────

def build_sqlite(tables: dict):
    """Build a normalized SQLite database from parsed CPIC tables."""
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    schema = {
        "drug": ["drugid TEXT PRIMARY KEY", "name TEXT", "pharmgkbid TEXT"],
        "gene": ["symbol TEXT PRIMARY KEY", "chr TEXT", "lookupmethod TEXT"],
        "allele": ["id INTEGER PRIMARY KEY", "genesymbol TEXT", "name TEXT",
                   "functionalstatus TEXT", "clinicalfunctionalstatus TEXT",
                   "activityvalue REAL", "citations TEXT"],
        "recommendation": ["id INTEGER PRIMARY KEY", "guidelineid INTEGER",
                           "drugid TEXT", "lookupkey TEXT", "implications TEXT",
                           "drugrecommendation TEXT", "classification TEXT",
                           "phenotypes TEXT", "activityscore TEXT",
                           "allelestatus TEXT", "population TEXT", "comments TEXT"],
        "guideline": ["id INTEGER PRIMARY KEY", "name TEXT", "url TEXT", "pmid TEXT",
                      "source TEXT"],
        "gene_result": ["id INTEGER PRIMARY KEY", "genesymbol TEXT", "result TEXT",
                        "activityscore TEXT", "ehrpriority TEXT",
                        "consultationtext TEXT", "frequency TEXT"],
    }

    for table, columns in schema.items():
        if table not in tables:
            print(f"  [SKIP] {table} not in dump")
            continue
        col_defs = ", ".join(columns)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({col_defs})")
        col_names = [c.split()[0] for c in columns]
        parsed_cols = tables[table]["columns"]
        placeholders = ",".join(["?"] * len(col_names))
        insert_sql = f"INSERT OR IGNORE INTO {table} ({','.join(col_names)}) VALUES ({placeholders})"
        count = 0
        for row in tables[table]["rows"]:
            mapped = []
            for cn in col_names:
                if cn in parsed_cols:
                    val = row[parsed_cols.index(cn)]
                    if isinstance(val, str) and val.startswith("{"):
                        try:
                            val = json.loads(val.replace("'", '"'))
                            val = json.dumps(val, ensure_ascii=False)
                        except json.JSONDecodeError:
                            pass
                    mapped.append(val)
                else:
                    mapped.append(None)
            try:
                conn.execute(insert_sql, mapped)
                count += 1
            except Exception:
                pass
        conn.commit()
        print(f"  {table}: {count} rows inserted")
    conn.close()
    print(f"\nSQLite: {DB_PATH} ({DB_PATH.stat().st_size / 1024:.0f} KB)")


# ── Hierarchical RAG Chunk Builder ───────────────────────────────────

def build_rag_chunks(tables: dict) -> list[dict]:
    """
    Build RAG chunks with hierarchical heading context.

    Hierarchy: CPIC > Guideline > Drug > Gene > Classification > Recommendation

    Each chunk is a self-contained information unit with its full context path
    stored as heading_hierarchy metadata. This enables:
      - Semantic retrieval that understands chunk context
      - Filtering by any level of the hierarchy (drug, gene, source)
      - Building answer with full provenance path
    """
    chunks = []
    drug_map = _build_drug_map(tables)
    gene_drug_map = _build_gene_drug_map(tables)
    pmid_map = _build_pmid_map(tables)
    guideline_map = _build_guideline_map(tables)

    # 1. Recommendations: the richest textual content
    rec = tables.get("recommendation")
    if rec:
        for row in rec["rows"]:
            d = dict(zip(rec["columns"], row))
            drug_id = d.get("drugid", "")
            drug_name = drug_map.get(drug_id, "")
            guideline_name = guideline_map.get(drug_id, "")
            guid_pmid = pmid_map.get(drug_id, "")
            genes = _extract_genes_from_lookupkey(d.get("lookupkey", ""))
            cls = d.get("classification", "")
            phenotype = _fmt_str(d.get("phenotypes", ""))

            # Build hierarchical heading path
            h_path = ["CPIC"]
            if guideline_name:
                h_path.append(guideline_name)
            if drug_name:
                h_path.append(f"Drug: {drug_name}")
            if genes:
                h_path.append(f"Gene: {genes}")
            if cls:
                h_path.append(f"Class: {cls}")
            if phenotype:
                h_path.append(f"Phenotype: {phenotype}")

            heading_full = " > ".join(h_path)
            heading_h1 = h_path[0] if len(h_path) > 0 else "CPIC"
            heading_h2 = h_path[1] if len(h_path) > 1 else ""
            heading_h3 = h_path[2] if len(h_path) > 2 else ""
            heading_h4 = h_path[3] if len(h_path) > 3 else ""

            text = _build_rec_chunk(d, drug_name)
            if text:
                # Ensure chunk is a complete info unit (add context prefix if needed)
                prefix = f"[{heading_full}]\n"
                full_content = prefix + text

                chunks.append({
                    "content": full_content,
                    "source": "CPIC",
                    "drug": drug_name,
                    "gene": genes,
                    "pmid": guid_pmid,
                    "classification": cls,
                    "phenotype": phenotype,
                    "heading_h1": heading_h1,
                    "heading_h2": heading_h2,
                    "heading_h3": heading_h3,
                    "heading_h4": heading_h4,
                    "heading_hierarchy": heading_full,
                    "chunk_type": "recommendation",
                })

                # If implications is long, split into separate chunk for better granularity
                imp = d.get("implications", "")
                if imp and len(imp) > 200:
                    imp_heading = heading_full + " > Implications"
                    chunks.append({
                        "content": f"[{imp_heading}]\nGene: {genes or drug_name}\nDrug: {drug_name}\n{_fmt_str(imp)}",
                        "source": "CPIC",
                        "drug": drug_name,
                        "gene": genes,
                        "pmid": guid_pmid,
                        "classification": cls,
                        "phenotype": phenotype,
                        "heading_h1": heading_h1,
                        "heading_h2": heading_h2,
                        "heading_h3": heading_h3,
                        "heading_h4": heading_h4,
                        "heading_hierarchy": imp_heading,
                        "chunk_type": "implications",
                    })

    # 2. Gene results: consultation text with phenotype mapping
    gr = tables.get("gene_result")
    if gr:
        for row in gr["rows"]:
            d = dict(zip(gr["columns"], row))
            gene = d.get("genesymbol", "")
            result = d.get("result", "")
            text = d.get("consultationtext", "")
            freq = d.get("frequency", "")
            if not text or not gene:
                continue

            h_path = ["CPIC", f"Gene: {gene}", f"Phenotype: {result}"]
            heading_full = " > ".join(h_path)
            drugs_for_gene = gene_drug_map.get(gene, "")
            full = (
                f"[{heading_full}]\n"
                f"Gene: {gene}\n"
                f"Phenotype: {result}\n"
            )
            if drugs_for_gene:
                full += f"Associated Drugs: {drugs_for_gene}\n"
            full += text
            if freq:
                full += f"\n\nPopulation Frequencies: {freq}"

            chunks.append({
                "content": full,
                "source": "CPIC",
                "drug": drugs_for_gene,
                "gene": gene,
                "pmid": "",
                "heading_h1": "CPIC",
                "heading_h2": f"Gene: {gene}",
                "heading_h3": f"Phenotype: {result}",
                "heading_h4": "",
                "heading_hierarchy": heading_full,
                "chunk_type": "gene_result",
            })

    # 3. Allele functional descriptions
    allele = tables.get("allele")
    if allele:
        for row in allele["rows"]:
            d = dict(zip(allele["columns"], row))
            gene = d.get("genesymbol", "")
            name = d.get("name", "")
            func = d.get("functionalstatus", "")
            clin = d.get("clinicalfunctionalstatus", "")
            activity = d.get("activityvalue", "")
            findings = d.get("findings", "")
            citations = d.get("citations", "")
            if not gene or not name:
                continue

            h_path = ["CPIC", f"Gene: {gene}", f"Allele: {name}"]
            heading_full = " > ".join(h_path)
            lines = [f"[{heading_full}]"]
            if func:
                lines.append(f"Functional Status: {func}")
            if clin:
                lines.append(f"Clinical Status: {clin}")
            if activity is not None:
                lines.append(f"Activity Score: {activity}")
            if findings:
                lines.append(f"Details: {findings}")
            if citations:
                lines.append(f"Citations: {citations}")

            chunks.append({
                "content": "\n".join(lines),
                "source": "CPIC",
                "drug": "",
                "gene": gene,
                "pmid": citations if citations else "",
                "heading_h1": "CPIC",
                "heading_h2": f"Gene: {gene}",
                "heading_h3": f"Allele: {name}",
                "heading_h4": func or clin or "",
                "heading_hierarchy": heading_full,
                "chunk_type": "allele",
            })

    print(f"\nBuilt {len(chunks)} hierarchical RAG chunks")
    if chunks:
        print(f"  Sample heading: {chunks[0].get('heading_hierarchy', '')}")
        print(f"  Sample content: {chunks[0]['content'][:120]}...")
    return chunks


# ── Helpers ──────────────────────────────────────────────────────────

def _build_drug_map(tables: dict) -> dict[str, str]:
    d = tables.get("drug")
    return {row[0]: row[1] or "" for row in d["rows"]} if d else {}


def _build_guideline_map(tables: dict) -> dict[str, str]:
    """Build {drugid -> guideline_name}."""
    gl = tables.get("guideline")
    d = tables.get("drug")
    if not gl or not d:
        return {}
    gl_map = {row[0]: row[1] or "" for row in gl["rows"]}  # id -> name
    guid_idx = _col_idx(d["columns"], "guidelineid")
    result = {}
    for row in d["rows"]:
        drug_id = row[0]
        if guid_idx is not None and len(row) > guid_idx:
            gid = row[guid_idx]
            if gid and gid in gl_map:
                result[drug_id] = gl_map[gid]
    return result


def _build_pmid_map(tables: dict) -> dict[str, str]:
    """Build {drugid -> pmid}."""
    gl = tables.get("guideline")
    d = tables.get("drug")
    if not gl or not d:
        return {}
    gl_map = {row[0]: row for row in gl["rows"]}
    pmid_idx = _col_idx(gl["columns"], "pmid")
    guid_idx = _col_idx(d["columns"], "guidelineid")
    result = {}
    for row in d["rows"]:
        drug_id = row[0]
        if guid_idx is not None and len(row) > guid_idx:
            gid = row[guid_idx]
            if gid and gid in gl_map and pmid_idx is not None:
                pmid = gl_map[gid][pmid_idx]
                if pmid:
                    result[drug_id] = pmid
    return result


def _build_gene_drug_map(tables: dict) -> dict[str, str]:
    """Build {gene -> comma-separated drug_names} from recommendation + drug."""
    rec = tables.get("recommendation")
    d = tables.get("drug")
    if not rec or not d:
        return {}
    drug_map = _build_drug_map(tables)
    result = {}
    for row in rec["rows"]:
        rd = dict(zip(rec["columns"], row))
        drug_name = drug_map.get(rd.get("drugid", ""), "")
        genes = _extract_genes_from_lookupkey(rd.get("lookupkey", ""))
        for g in genes.split(", "):
            if g:
                existing = result.get(g, "")
                if drug_name and drug_name not in existing:
                    result[g] = (existing + ", " + drug_name) if existing else drug_name
    return result


def _col_idx(columns: list[str], name: str) -> int | None:
    try:
        return columns.index(name)
    except ValueError:
        return None


def _build_rec_chunk(d: dict, drug_name: str) -> str:
    parts = []
    if drug_name:
        parts.append(f"Drug: {drug_name}")
    lk = d.get("lookupkey", "")
    if lk:
        parts.append(f"Genotype: {lk}")
    cls = d.get("classification", "")
    if cls:
        parts.append(f"Classification: {cls}")
    ph = d.get("phenotypes", "")
    if ph:
        parts.append(f"Phenotype: {_fmt_str(ph)}")
    imp = d.get("implications", "")
    if imp:
        parts.append(f"Implications: {_fmt_str(imp)}")
    rec_text = d.get("drugrecommendation", "")
    if rec_text:
        parts.append(f"Recommendation: {rec_text}")
    pop = d.get("population", "")
    if pop:
        parts.append(f"Population: {pop}")
    comments = d.get("comments", "")
    if comments:
        parts.append(f"Comments: {comments}")
    return "\n".join(parts)


def _extract_genes_from_lookupkey(lookupkey: str) -> str:
    if not lookupkey:
        return ""
    try:
        d = json.loads(lookupkey.replace("'", '"'))
        return ", ".join(d.keys())
    except (json.JSONDecodeError, ValueError):
        return ""


def _fmt_str(val: str) -> str:
    if not val:
        return ""
    try:
        return json.dumps(json.loads(val.replace("'", '"')), ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return val[:800]


# ── Ingest to ChromaDB ──────────────────────────────────────────────

def ingest_chunks(chunks: list[dict], reset: bool = False):
    if not chunks:
        print("No chunks to ingest.")
        return
    store = VectorStore()
    if reset:
        print("Resetting ChromaDB collection ...")
        store.delete_collection()

    texts = [c["content"] for c in chunks]
    metadatas = [{
        "source": c.get("source", "CPIC"),
        "drug": c.get("drug", ""),
        "gene": c.get("gene", ""),
        "pmid": c.get("pmid", ""),
        "classification": c.get("classification", ""),
        "phenotype": c.get("phenotype", ""),
        "heading_h1": c.get("heading_h1", ""),
        "heading_h2": c.get("heading_h2", ""),
        "heading_h3": c.get("heading_h3", ""),
        "heading_h4": c.get("heading_h4", ""),
        "heading_hierarchy": c.get("heading_hierarchy", ""),
        "chunk_type": c.get("chunk_type", ""),
    } for c in chunks]
    ids = [f"cpic:{i}:{hashlib.md5(c['content'][:80].encode()).hexdigest()[:8]}"
           for i, c in enumerate(chunks)]

    store.add_chunks(texts, metadatas, ids)
    print(f"Ingested {len(chunks)} chunks (total: {store.count()})")


# ── Main ─────────────────────────────────────────────────────────────

def run(dump_path: str, reset: bool = False, validate_only: bool = False):
    print(f"Parsing CPIC dump: {dump_path}")
    tables = parse_copy_data(dump_path, validate_only=validate_only)

    if validate_only:
        print("\nValidation complete. No data written.")
        return

    print("\n--- Building SQLite ---")
    build_sqlite(tables)

    print("\n--- Building hierarchical RAG chunks ---")
    chunks = build_rag_chunks(tables)

    print("\n--- Ingesting into ChromaDB ---")
    ingest_chunks(chunks, reset=reset)
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse CPIC SQL dump")
    parser.add_argument("--dump", required=True, help="Path to cpic_db_dump-v1.53.2.sql")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--validate-only", action="store_true",
                        help="Parse and validate the dump file without writing to DB")
    args = parser.parse_args()
    run(dump_path=args.dump, reset=args.reset, validate_only=args.validate_only)
