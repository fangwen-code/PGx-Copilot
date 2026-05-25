"""
RAG ingestion pipeline: PDF -> chunk -> embed -> ChromaDB.

Metadata is read from data/sources.json -- a manually curated file
that maps each PDF to its source info (drug, gene, year, source type).
A content-based fallback extracts clues from the PDF text.

Usage:
    python -m rag.ingest                          # ingest all PDFs
    python -m rag.ingest --reset                  # drop and re-ingest
    python -m rag.ingest --file path/to/pdf       # ingest a single file
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import pymupdf

from config import DATA_DIR, SOURCES_DIR
from rag.vector_store import VectorStore

METADATA_FILE = DATA_DIR / "sources.json"


# -- Metadata ----------------------------------------------------------

def load_metadata() -> dict[str, dict]:
    """Load sources.json; returns {filename: {source, year, drug, gene, ...}}."""
    if not METADATA_FILE.exists():
        print(f"[WARN] {METADATA_FILE} not found -- all metadata will be empty.")
        return {}
    with open(METADATA_FILE, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["filename"]: {k: v for k, v in e.items() if k != "filename"} for e in entries}


def _extract_genes(text: str) -> str:
    """Extract gene names from text using standard naming patterns."""
    # CYP family: CYP2D6, CYP2C9, CYP2C19, CYP3A4, etc.
    cyp = re.findall(r'\bCYP\d[A-Z]\d\w*\b', text)
    # HLA family: HLA-B*57:01, HLA-A*31:01, etc.
    hla = re.findall(r'\bHLA-[A-Z]\*?\d+[\*:]?\d*\b', text)
    # Common PGx genes written as standalone (TPMT, NUDT15, SLCO1B1, etc.)
    common = re.findall(r'\b(NAT2|TPMT|NUDT15|SLCO1B1|APOE|VKORC1|DPYD|UGT1A1|BCHE|G6PD)\b', text)
    # Other gene-like patterns: COMT, ADRB1, AGTR1, ACE, etc.
    other = re.findall(r'\b(ADRB[12]|ADRA2C|GRK[45]|AGTR1|ACE|NPPA|CACNA1C|CACNB2|NEDD4L|YEATS4)\b', text)

    all_genes = set(cyp + hla + common + other)
    return ", ".join(sorted(all_genes)) if all_genes else ""


def _extract_drugs(text: str) -> str:
    """Extract drug names from text using naming patterns."""
    # -olol: beta blockers (metoprolol, atenolol, etc.)
    olol = re.findall(r'\b([a-z]+olol)\b', text, re.IGNORECASE)
    # -sartan: ARBs (losartan, valsartan, etc.)
    sartan = re.findall(r'\b([a-z]+sartan)\b', text, re.IGNORECASE)
    # -pril: ACE inhibitors (lisinopril, enalapril, etc.)
    pril = re.findall(r'\b([a-z]+pril)\b', text, re.IGNORECASE)
    # -pine: CCBs (amlodipine, nifedipine, etc.)
    pine = re.findall(r'\b([a-z]+pine)\b', text, re.IGNORECASE)
    # -thiazide, -mide: diuretics
    thiazide = re.findall(r'\b([a-z]+thiazide)\b', text, re.IGNORECASE)
    mide = re.findall(r'\b([a-z]+mide)\b', text, re.IGNORECASE)
    # -statin
    statin = re.findall(r'\b([a-z]+statin)\b', text, re.IGNORECASE)
    # Other common PGx-relevant drugs
    other = re.findall(r'\b(hydralazine|warfarin|clopidogrel|abacavir|azathioprine|mercaptopurine|irinotecan|simvastatin)\b', text, re.IGNORECASE)

    all_drugs = set(
        [x.lower() for x in olol + sartan + pril + pine + thiazide + mide + statin + other]
    )
    return ", ".join(sorted(all_drugs)) if all_drugs else ""


def _extract_source(text: str) -> str:
    """Determine source type from text clues."""
    if re.search(r'Clinical Pharmacogenetics Implementation Consortium|CPIC guideline', text):
        return "CPIC"
    if re.search(r'Food and Drug Administration|FDA label|FDA-approved', text[:2000]):
        return "FDA"
    if re.search(r'PharmGKB|PharmGRB', text[:2000]):
        return "PharmGKB"
    if re.search(r'hypertension guideline|高血压.*指南|AHA|ACC|ESC.*guideline', text[:3000], re.IGNORECASE):
        return "Clinical Guideline"
    if re.search(r'PubMed|https://doi\.org/10\.', text[:1000]):
        return "PubMed"
    return ""


def _extract_year(text: str) -> str:
    """Extract the first 4-digit year from the first portion of text."""
    match = re.search(r'(?:19|20)\d{2}', text[:800])
    return match.group() if match else ""


def metadata_from_content(doc: pymupdf.Document) -> dict:
    """
    Fallback: scan first few pages for metadata clues.
    Used only when sources.json has no entry for this file.
    """
    text = ""
    for i in range(min(5, len(doc))):
        text += doc[i].get_text()

    return {
        "source": _extract_source(text),
        "year": _extract_year(text),
        "drug": _extract_drugs(text),
        "gene": _extract_genes(text),
    }


# -- Chunking ----------------------------------------------------------

def extract_sections(doc: pymupdf.Document) -> list[dict]:
    """Extract text sections by heading structure."""
    sections = []
    current_heading = "前言"
    current_text = []

    for page in doc:
        blocks = page.get_text("blocks")
        for block in sorted(blocks, key=lambda b: (b[1], b[0])):
            text = (block[4] or "").strip()
            if not text:
                continue
            font_size = _detect_font_size(block)
            is_heading = (
                (font_size and font_size > 11)
                or (len(text) < 80 and text.isupper())
                or bool(re.match(r'^[一二三四五六七八九十]+[、.．]\s*\S', text))
            )
            if is_heading and current_text:
                sections.append({"heading": current_heading, "text": "\n".join(current_text).strip()})
                current_heading = text
                current_text = []
            elif is_heading:
                current_heading = text
            else:
                current_text.append(text)

    if current_text:
        sections.append({"heading": current_heading, "text": "\n".join(current_text).strip()})
    return sections


def _detect_font_size(block) -> float | None:
    try:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                return span.get("size")
    except Exception:
        return None


def chunk_sections(sections: list[dict], max_chars: int = 800, overlap_chars: int = 150) -> list[dict]:
    """Split sections into chunks with overlap, respecting paragraph boundaries."""
    chunks = []
    for sec in sections:
        heading, text = sec["heading"], sec["text"]
        paragraphs = re.split(r'\n\s*\n', text)
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) > max_chars and current:
                chunks.append({"content": current.strip(), "heading": heading})
                overlap = current[-overlap_chars:] if len(current) > overlap_chars else current
                current = overlap + "\n\n" + para
            else:
                current = (current + "\n\n" + para) if current else para
        if current:
            chunks.append({"content": current.strip(), "heading": heading})
    return chunks


# -- Ingest ------------------------------------------------------------

def _chunk_id(filename: str, idx: int) -> str:
    return hashlib.md5(f"{filename}:chunk:{idx}".encode()).hexdigest()[:16]


def ingest_pdf(pdf_path: Path, store: VectorStore, meta_lookup: dict) -> int:
    """Parse a single PDF and ingest into ChromaDB."""
    print(f"  [{pdf_path.name}] ", end="", flush=True)

    doc = pymupdf.open(str(pdf_path))
    sections = extract_sections(doc)
    chunks = chunk_sections(sections)

    if pdf_path.name in meta_lookup:
        meta_base = dict(meta_lookup[pdf_path.name])
    else:
        meta_base = metadata_from_content(doc)
    doc.close()

    meta_base["filename"] = pdf_path.name
    chunk_texts, metadatas, ids = [], [], []
    for i, c in enumerate(chunks):
        chunk_texts.append(c["content"])
        metadatas.append({**meta_base, "heading": c["heading"], "chunk_index": i})
        ids.append(_chunk_id(pdf_path.name, i))

    count = store.add_chunks(chunk_texts, metadatas, ids)
    meta_str = f"(source={meta_base.get('source','?')}, drug={meta_base.get('drug','?')})"
    print(f"ok, {count} chunks {meta_str}")
    return count


def ingest_all(reset: bool = False, single_file: str | None = None):
    """Ingest all PDFs from sources directory."""
    store = VectorStore()
    if reset:
        print("Resetting collection ...")
        store.delete_collection()

    meta_lookup = load_metadata()

    if single_file:
        pdfs = [Path(single_file)]
    else:
        pdfs = sorted(SOURCES_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"[WARN] No PDFs found in {SOURCES_DIR}")
        print(f"  Place PDFs in: {SOURCES_DIR}")
        print(f"  Add metadata in: {METADATA_FILE}")
        return

    total = 0
    for pdf in pdfs:
        total += ingest_pdf(pdf, store, meta_lookup)

    print(f"\nDone! Total chunks: {total}  (collection: {store.count()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    ingest_all(reset=args.reset, single_file=args.file)
