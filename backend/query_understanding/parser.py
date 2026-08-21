"""
Query Understanding: parse user input into structured intent.

Uses DeepSeek LLM to extract gene, genotype, drug, symptom, disease, etc.
from free-text user input.
"""

import json
import os
import re
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

_SYSTEM_PROMPT = """You are a query parser for a pharmacogenomics (PGx) decision support system.
Extract structured information from the user's question.

Return ONLY a JSON object with these optional fields:
- "genes": list of gene symbols mentioned (e.g. ["CYP2D6", "SLCO1B1"])
- "genotypes": dict mapping gene to genotype. If a metabolizer phenotype or function is EXPLICITLY stated, use its ENGLISH name as the value (e.g. "中间代谢" → "Intermediate Metabolizer", "poor metabolizer" → "Poor Metabolizer") — English values match the CPIC key directly. Values MUST be in English. Only use a star allele (e.g. "*4/*4") when no phenotype/function is stated.
- "drug": the SPECIFIC drug name (e.g. "metoprolol", "simvastatin"). Only set when a specific drug is named.
- "drug_class": drug class if mentioned (e.g. "CCB", "beta blocker", "statin")

IMPORTANT: generic class terms such as 他汀 / statin / CCB / ARB are DRUG CLASSES — return them in "drug_class", NOT as a specific "drug". Never guess a specific drug from a class term.
- "disease": disease or condition (e.g. "hypertension")
- "symptom": symptom description (e.g. "cough", "muscle pain")
- "condition": clinical condition (e.g. "CKD eGFR<30")
- "intent": one of ["ADR", "efficacy", "safety", "drug_choice", "alternative", "general"]

If the input is unclear or contains nothing PGx-related, return {"intent": "unclear"}.

Examples:
{"input": "吃美托洛尔咳嗽"} -> {"drug": "metoprolol", "symptom": "cough", "intent": "ADR"}
{"input": "CYP2D6 *4/*4"} -> {"genes": ["CYP2D6"], "genotypes": {"CYP2D6": "*4/*4"}, "intent": "general"}
{"input": "高血压CCB效果不好"} -> {"drug_class": "CCB", "disease": "hypertension", "intent": "alternative"}
{"input": "SLCO1B1 TC APOE E3/E4 他汀安全吗"} -> {"genes": ["SLCO1B1", "APOE"], "genotypes": {"SLCO1B1": "TC", "APOE": "E3/E4"}, "drug_class": "statin", "intent": "safety"}
{"input": "我头晕"} -> {"symptom": "dizziness", "intent": "general"}
"""


def parse_query(user_input: str) -> dict:
    """Parse user input into structured intent."""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_api_key_here":
        return _fallback_parse(user_input)

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content
        parsed = json.loads(text)
        # Backfill genes/genotypes the LLM missed — see _merge_fallback.
        return _merge_fallback(parsed, _fallback_parse(user_input))
    except Exception as e:
        print(f"[WARN] LLM parsing failed: {e}, using fallback")
        return _fallback_parse(user_input)


def _merge_fallback(llm: dict, fb: dict) -> dict:
    """Backfill genes/genotypes the LLM dropped using the deterministic fallback.

    Query understanding is the single input gate: if it loses a gene, every
    downstream layer (CPIC match, rule engine, RAG) silently loses it too, and
    the query degrades to a drug-only gap (no recommendation). LLMs are
    occasionally unreliable here, so the cheap deterministic regex doubles as a
    safety net — it only ADDS what the LLM missed, never removes anything.
    """
    llm = llm or {}
    genes = set(llm.get("genes") or []) | set(fb.get("genes") or [])
    genotypes = dict(llm.get("genotypes") or {})
    for g in fb.get("genotypes", {}):
        if g not in genotypes:
            genotypes[g] = fb["genotypes"][g]
    if genes:
        llm["genes"] = sorted(genes)
    if genotypes:
        llm["genotypes"] = genotypes
    return llm


# ── Fallback lexicon: data-driven from the CPIC DB (not hand-maintained) ──

_BASE_GENE_PATTERNS = {
    "CYP2D6": r"cyp\s*2\s*d\s*6|2d6",
    "CYP2C9": r"cyp\s*2\s*c\s*9|2c9",
    "CYP2C19": r"cyp\s*2\s*c\s*19|2c19",
    "SLCO1B1": r"slco1b1",
    "APOE": r"apoe",
    "NAT2": r"nat2",
    "CYP3A4": r"cyp\s*3\s*a\s*4|3a4",
}

# Built-in drug aliases: only the ones the DB can't provide (Chinese names).
# English drug names come from the CPIC drug table (see _load_drug_names).
_BASE_DRUG_ALIASES = {
    "metoprolol": ["美托洛尔", "metoprolol"],
    "losartan": ["氯沙坦", "losartan"],
    "simvastatin": ["辛伐他汀", "simvastatin"],
    "amlodipine": ["氨氯地平", "amlodipine"],
    "lisinopril": ["赖诺普利", "lisinopril"],
}

_gene_pattern_cache: dict | None = None
_drug_name_cache: list[str] | None = None


def _load_gene_patterns() -> dict:
    """Load gene mention regexes from the CPIC DB (data-driven, not hand-kept).

    Every gene in the `gene` table gets a case-insensitive pattern, plus a
    derived abbreviated form for CYP genes (e.g. "2d6" from "CYP2D6"). Falls
    back to the built-in set when the DB is unavailable.
    """
    global _gene_pattern_cache
    if _gene_pattern_cache is not None:
        return _gene_pattern_cache
    patterns = dict(_BASE_GENE_PATTERNS)
    try:
        import sqlite3
        from config import DATA_DIR
        conn = sqlite3.connect(DATA_DIR / "cpic.db")
        rows = conn.execute("SELECT symbol FROM gene").fetchall()
        conn.close()
    except Exception:
        _gene_pattern_cache = patterns
        return patterns
    for (sym,) in rows:
        s = (sym or "").strip()
        if not s:
            continue
        key = s.upper()
        if key in patterns:
            continue
        spaced = r"\s*".join(s.lower())
        pattern = spaced
        m = re.match(r"CYP(\d[A-Z]\d)", s.upper())
        if m:
            pattern += "|" + r"\s*".join(m.group(1).lower())
        patterns[key] = pattern
    _gene_pattern_cache = patterns
    return patterns


def _load_drug_names() -> list[str]:
    """Load drug names from the CPIC drug table (data-driven fallback lexicon)."""
    global _drug_name_cache
    if _drug_name_cache is not None:
        return _drug_name_cache
    names: list[str] = []
    try:
        import sqlite3
        from config import DATA_DIR
        conn = sqlite3.connect(DATA_DIR / "cpic.db")
        rows = conn.execute("SELECT DISTINCT name FROM drug").fetchall()
        conn.close()
    except Exception:
        _drug_name_cache = names
        return names
    for (n,) in rows:
        n = (n or "").strip().lower()
        if n:
            names.append(n)
    _drug_name_cache = names
    return names


def _parse_lookupkey(lk: str) -> dict:
    """Parse the DB's single/double-quoted JSON lookupkey
    (e.g. "{'CYP2C19': '*2/*2'}" or '{"UGT1A1": "Poor Metabolizer"}')."""
    if not lk:
        return {}
    try:
        return json.loads(lk.replace("'", '"'))
    except json.JSONDecodeError:
        return {}


_known_values_cache: dict[str, set[str]] | None = None


def _load_known_values() -> dict[str, set[str]]:
    """Load per-gene known genotype values from the recommendation lookupkeys.

    Direct extraction source: the DB already stores the values a genotype can
    take (phenotype / function / activity score / star allele). The fallback
    parser looks these up in the user text instead of regex-matching shapes.
    """
    global _known_values_cache
    if _known_values_cache is not None:
        return _known_values_cache
    mapping: dict[str, set[str]] = {}
    try:
        import sqlite3
        from config import DATA_DIR
        conn = sqlite3.connect(DATA_DIR / "cpic.db")
        rows = conn.execute("SELECT lookupkey FROM recommendation").fetchall()
        conn.close()
    except Exception:
        _known_values_cache = mapping
        return mapping
    for (lk,) in rows:
        for gene, val in _parse_lookupkey(lk or "").items():
            v = str(val).strip().lower()
            if v:
                mapping.setdefault(str(gene).upper(), set()).add(v)
    _known_values_cache = mapping
    return mapping


def _extract_genotype(tail: str, gene: str) -> str | None:
    """Directly extract a genotype value following a gene mention.

    Preferred: the longest known DB value for the gene (phenotype / function /
    activity score) present in the tail — a direct lookup, no shape-matching.
    Fallback: a star-allele diplotype / score token right after the mention
    (star alleles are pattern-shaped, not enumerable).
    """
    best = ""
    for v in _load_known_values().get(gene.upper(), ()):
        if v and v in tail and len(v) > len(best):
            best = v
    m = re.match(
        r"[\s,;:]*((?:≥|>)?\d+\.\d+|\*?\d+(?:\s*/\s*\*?\d+)?|\d+)", tail
    )
    if m and len(m.group(1)) > len(best):
        best = m.group(1)
    return best or None


def _fallback_parse(user_input: str) -> dict:
    """Simple fallback when LLM is unavailable."""
    result = {"intent": "general"}
    text = user_input.lower()

    gene_patterns = _load_gene_patterns()
    genes = []
    genotypes = {}
    for gene, pattern in gene_patterns.items():
        import re
        m = re.search(pattern, text)
        if m:
            genes.append(gene)
            # Extract the genotype directly: the longest known DB value
            # (phenotype / function / activity score) appearing after the
            # mention, else a star-allele diplotype right after it.
            gt = _extract_genotype(text[m.end():], gene)
            if gt:
                genotypes[gene] = gt

    # Drug class vs specific drug: generic class terms (他汀/CCB/ARB...) are
    # NOT mapped to a specific drug.
    if any(w in text for w in ["他汀", "他汀类", "statin"]):
        result["drug_class"] = "statin"
    # Specific drug patterns — built-in aliases (incl. Chinese names, which the
    # DB can't provide) first, then every drug name loaded from the CPIC table.
    drug = ""
    for name, aliases in _BASE_DRUG_ALIASES.items():
        if any(a in text for a in aliases):
            drug = name
            break
    if not drug:
        for name in _load_drug_names():
            if name in text:
                drug = name
                break

    # Disease patterns
    if any(w in text for w in ["高血压", "hypertension", "血压高"]):
        result["disease"] = "hypertension"
    if any(w in text for w in ["咳嗽", "cough"]):
        result["symptom"] = "cough"
    if any(w in text for w in ["eGFR", "ckd", "kidney", "肾"]):
        result["condition"] = "CKD"

    # Intent based on keywords
    if any(w in text for w in ["安全", "风险", "不良反应", "咳嗽", "副作用"]):
        result["intent"] = "safety"
    elif any(w in text for w in ["效果", "疗效", "不好", "不佳"]):
        result["intent"] = "efficacy"
    elif any(w in text for w in ["替代", "换", "其他选择"]):
        result["intent"] = "alternative"
    elif any(w in text for w in ["选", "推荐", "建议"]):
        result["intent"] = "drug_choice"

    if genes:
        result["genes"] = genes
    if genotypes:
        result["genotypes"] = genotypes
    if drug:
        result["drug"] = drug

    return result
