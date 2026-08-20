"""
Fetch PubMed abstracts cited in the CPIC pair table → build RAG index.

Pipeline:
  1. Read cpic.db → parse pair.citations for all (pmid, gene) pairs
  2. Fetch each abstract via NCBI Entrez E-utilities
  3. Log PMIDs without abstracts to data/pubmed_missing.jsonl
  4. Scan data/sources/pubmed_manual/ for PMID.pdf full-texts to fill gaps
  5. Chunk and store in ChromaDB with source="pubmed"

Usage:
    python -m tools.fetch_pubmed                    # fetch all missing
    python -m tools.fetch_pubmed --reset            # reset and re-fetch
    python -m tools.fetch_pubmed --pmid 12345678    # fetch a single PMID
"""

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import DATA_DIR, NCBI_API_KEY, NCBI_EMAIL, SOURCES_DIR
from cpic.query import CPICQuery, DB_PATH
from rag.vector_store import VectorStore

# Manual full-text PDFs go here, named <PMID>.pdf
MANUAL_PDF_DIR = SOURCES_DIR / "pubmed_manual"
MISSING_LOG = DATA_DIR / "pubmed_missing.jsonl"


# ── PubMed E-utilities ──────────────────────────────────────────────

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# With API key: 10 req/s → 0.12s delay. Without: 3 req/s → 0.35s delay.
DELAY = 0.12 if NCBI_API_KEY else 0.35
TOOL = "PGx-Copilot"


def _efetch_url(pmid: str) -> str:
    """Build the EFetch URL with API key / contact info if configured."""
    params = [
        "db=pubmed",
        f"id={pmid}",
        "retmode=xml",
        "rettype=abstract",
        f"tool={TOOL}",
    ]
    if NCBI_EMAIL:
        params.append(f"email={NCBI_EMAIL}")
    if NCBI_API_KEY:
        params.append(f"api_key={NCBI_API_KEY}")
    return f"{NCBI_BASE}/efetch.fcgi?" + "&".join(params)


def fetch_pubmed_record(pmid: str, retries: int = 3) -> dict | None:
    """Fetch title + abstract for a PMID using EFetch XML.

    Retries transient HTTP errors (502/503/504 Bad Gateway) up to `retries`
    times with a backoff delay. Returns {"title", "abstract"} or None.
    """
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_efetch_url(pmid), timeout=15) as resp:
                xml = resp.read().decode("utf-8")
            break  # success
        except Exception as e:
            last_err = e
            # Only retry on transient server errors / network issues
            if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 404):
                print(f"  [WARN] PMID {pmid} permanent error {e.code}")
                return None
            if attempt < retries - 1:
                wait = 1.5 * (attempt + 1)
                print(f"  [RETRY] PMID {pmid} attempt {attempt + 1} failed ({e}), retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"  [WARN] PMID {pmid} fetch failed after {retries} attempts: {last_err}")
                return None

    # Title
    title_match = re.search(r"<ArticleTitle>([^<]+)", xml)
    title = title_match.group(1).strip() if title_match else ""

    # Abstract — join all <AbstractText> elements
    abstract_texts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", xml, re.DOTALL)
    if not abstract_texts:
        return {"title": title, "abstract": ""}

    # Strip inner XML tags and collapse whitespace
    abstract = " ".join(re.sub(r"<[^>]+>", "", t).strip() for t in abstract_texts)
    abstract = re.sub(r"\s+", " ", abstract).strip()

    return {"title": title, "abstract": abstract}

    # Title
    title_match = re.search(r"<ArticleTitle>([^<]+)", xml)
    title = title_match.group(1).strip() if title_match else ""

    # Abstract — join all <AbstractText> elements
    abstract_texts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", xml, re.DOTALL)
    if not abstract_texts:
        return {"title": title, "abstract": ""}

    # Strip inner XML tags and collapse whitespace
    abstract = " ".join(re.sub(r"<[^>]+>", "", t).strip() for t in abstract_texts)
    abstract = re.sub(r"\s+", " ", abstract).strip()

    return {"title": title, "abstract": abstract}


# ── Chunking ────────────────────────────────────────────────────────


def chunk_abstract(abstract: str, max_chars: int = 600) -> list[str]:
    """Split an abstract into smaller chunks at sentence boundaries."""
    if len(abstract) <= max_chars:
        return [abstract]

    chunks = []
    sentences = re.split(r"(?<=[.!?])\s+", abstract)

    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent) if current else sent
    if current:
        chunks.append(current.strip())

    return chunks


# ── Missing abstract logging ────────────────────────────────────────


def log_missing_abstracts(missing: list[dict]):
    """Append PMIDs without abstracts to the missing log for manual follow-up."""
    if not missing:
        return
    MISSING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(MISSING_LOG, "a", encoding="utf-8") as f:
        for entry in missing:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n[MISSING] {len(missing)} PMIDs without abstracts → logged to {MISSING_LOG.name}")
    print("  Place full-text PDFs (named <PMID>.pdf) in:")
    print(f"    {MANUAL_PDF_DIR}")


# ── Manual full-text PDF fallback ───────────────────────────────────


def ingest_manual_pdfs(store: VectorStore) -> int:
    """
    Scan data/sources/pubmed_manual/ for <PMID>.pdf and index full text.

    Fills the gap left by PMIDs whose abstracts could not be fetched
    from the NCBI API (paywalled / no-abstract papers). Each PDF is
    named after its PMID so metadata can be recovered.

    Returns number of PDFs ingested.
    """
    if not MANUAL_PDF_DIR.exists():
        return 0

    pdfs = sorted(MANUAL_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        return 0

    import pymupdf

    print(f"\n[MANUAL] Found {len(pdfs)} full-text PDF(s) in {MANUAL_PDF_DIR.name}")
    total_chunks = 0

    for pdf_path in pdfs:
        pmid = pdf_path.stem.strip()
        if not pmid.isdigit():
            print(f"  [SKIP] {pdf_path.name} — filename must be <PMID>.pdf")
            continue

        try:
            doc = pymupdf.open(str(pdf_path))
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as e:
            print(f"  [WARN] {pdf_path.name} parse failed: {e}")
            continue

        if len(text.strip()) < 100:
            print(f"  [SKIP] {pdf_path.name} — extracted text too short")
            continue

        # Semantic chunking: split by paragraph (author's own boundaries).
        # Only very long paragraphs (> 1500 chars) are split at sentence
        # boundaries — never cut mid-sentence.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        parts = []
        for para in paragraphs:
            if len(para) > 1500:
                sentences = re.split(r"(?<=[.!?。！？])\s+", para)
                merged = ""
                for sent in sentences:
                    if len(merged) + len(sent) < 1500:
                        merged = (merged + " " + sent) if merged else sent
                    else:
                        if merged:
                            parts.append(merged)
                        merged = sent
                if merged:
                    parts.append(merged)
            else:
                parts.append(para)

        chunks_to_add = []
        for ci, part in enumerate(parts):
            chunk_text = f"Full text: {part}"
            metadata = {
                "source": "pubmed",
                "pmid": pmid,
                "gene": "",
                "drug": "",
                "title": f"Manual full-text PMID {pmid}",
                "chunk_index": ci,
                "fulltext": True,
            }
            chunks_to_add.append((chunk_text, metadata))

        if chunks_to_add:
            texts = [c[0] for c in chunks_to_add]
            metadatas = [c[1] for c in chunks_to_add]
            ids = [f"pubmed:{pmid}:manual:{i}:{hashlib.md5(texts[i][:80].encode()).hexdigest()[:8]}"
                   for i in range(len(texts))]
            added = store.add_chunks(texts, metadatas, ids)
            total_chunks += added
            print(f"  [OK] {pdf_path.name} → {added} chunks")

    print(f"[MANUAL] Total {total_chunks} chunks indexed from manual PDFs")
    return total_chunks


# ── Main pipeline ───────────────────────────────────────────────────


def run(reset: bool = False, single_pmid: str | None = None):
    """Main pipeline: read PMIDs, fetch abstracts, store in ChromaDB."""
    store = VectorStore()
    if reset:
        print("Resetting ChromaDB collection ...")
        store.delete_collection()

    if not store.available:
        print("[ERROR] Vector store not available. Cannot index PubMed abstracts.")
        return

    # Step 1: Get Level A/B entries from CPIC
    db_path = DB_PATH
    if not Path(str(db_path)).exists():
        print(f"[ERROR] CPIC database not found: {db_path}")
        print("  Run: python -m cpic.parser --dump ../data/sources/cpic_db_dump-v1.53.2.sql")
        return

    q = CPICQuery()
    entries = q.get_cited_pmid_entries()
    q.close()

    if not entries:
        print("No cited PMIDs found in CPIC pair table.")
        return

    # Deduplicate by PMID
    pmids_seen: set[str] = set()
    if single_pmid:
        entries = [e for e in entries if e["pmid"] == single_pmid]

    for e in entries:
        pmids_seen.add(e["pmid"])

    print(f"Found {len(entries)} gene-PMID entries ({len(pmids_seen)} unique PMIDs)")

    # Step 2: Fetch each abstract
    chunks_to_add: list[tuple[str, dict]] = []  # (text, metadata)
    fetched_count = 0
    missing: list[dict] = []  # PMIDs without abstracts

    for i, pmid in enumerate(sorted(pmids_seen), 1):
        print(f"  [{i}/{len(pmids_seen)}] PMID {pmid} ... ", end="", flush=True)

        record = fetch_pubmed_record(pmid)
        time.sleep(DELAY)

        # Get all (gene, drug) pairs associated with this PMID
        related_entries = [e for e in entries if e["pmid"] == pmid]
        gene_drug_pairs = list({
            (e["gene"], e["drug"]) for e in related_entries if e.get("gene")
        })

        if not record or not record.get("abstract"):
            print("skip (no abstract)")
            missing.append({
                "pmid": pmid,
                "genes": list(set(g for g, _ in gene_drug_pairs)),
                "reason": "no abstract available",
            })
            continue

        title = record["title"]
        abstract = record["abstract"]

        # Chunk the abstract. One chunk per (gene, drug) pair so metadata
        # values stay scalars (ChromaDB metadata must be str/int/float/bool).
        # A PMID cited for multiple genes produces one indexed copy per pair.
        parts = chunk_abstract(abstract)
        for (gene, drug) in gene_drug_pairs:
            for ci, part in enumerate(parts):
                chunk_text = f"Title: {title}\nAbstract: {part}"
                metadata = {
                    "source": "pubmed",
                    "pmid": pmid,
                    "gene": gene,
                    "drug": drug,
                    "title": title,
                    "chunk_index": ci,
                }
                chunks_to_add.append((chunk_text, metadata))

        fetched_count += 1
        pair_str = ", ".join(f"{g}/{d}" for g, d in gene_drug_pairs[:3])
        print(f"ok ({len(parts)} chunks x {len(gene_drug_pairs)} pairs, pairs={pair_str})")

    # Log PMIDs that couldn't be fetched from API
    log_missing_abstracts(missing)

    # Step 3: Store fetched abstracts in ChromaDB
    if chunks_to_add:
        texts = [c[0] for c in chunks_to_add]
        metadatas = [c[1] for c in chunks_to_add]
        ids = [
            f"pubmed:{metadatas[i].get('pmid', '')}:{metadatas[i].get('gene', '')}:{metadatas[i].get('drug', '')}:chunk:{metadatas[i].get('chunk_index', 0)}:{hashlib.md5(texts[i][:80].encode()).hexdigest()[:8]}"
            for i in range(len(texts))
        ]

        added = store.add_chunks(texts, metadatas, ids)
        print(f"\nDone! {added} chunks indexed from API abstracts ({fetched_count} abstracts)")
    else:
        print("\nNo abstracts fetched from API.")

    # Step 4: Manual full-text PDF fallback
    ingest_manual_pdfs(store)

    print(f"Total in collection: {store.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts → ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Clear and re-index")
    parser.add_argument("--pmid", type=str, help="Fetch only a specific PMID")
    args = parser.parse_args()
    run(reset=args.reset, single_pmid=args.pmid)
