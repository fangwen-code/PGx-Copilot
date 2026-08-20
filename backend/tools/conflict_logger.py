"""
Fallback Audit Logger — records queries that fell through to RAG.

When neither Layer 1 (CPIC SQL) nor Layer 2 (Rule Engine) can answer
a query, Layer 3 (RAG) serves as fallback. This module logs those
fallback invocations for audit and coverage gap analysis.

The audit trail helps answer:
  - What types of queries can't be handled by structured data?
  - Is RAG providing adequate answers for these gaps?
  - Should new rules or data be added to reduce reliance on RAG?

Logs are stored as JSONL in data/fallback_log/.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import DATA_DIR

FALLBACK_DIR = DATA_DIR / "fallback_log"
FALLBACK_DIR.mkdir(parents=True, exist_ok=True)

# ── Conflict Detection ───────────────────────────────────────────────


def _extract_sql_recommendation(sql_results: list[dict]) -> dict:
    """Extract the key clinical recommendation from SQL results."""
    if not sql_results:
        return {}

    r = sql_results[0]
    return {
        "drugrecommendation": r.get("drugrecommendation", ""),
        "classification": r.get("classification", ""),
        "implications": r.get("implications", ""),
        "phenotypes": r.get("phenotypes", ""),
        "lookupkey": r.get("lookupkey", ""),
    }


def _extract_rag_claims(rag_results: list[dict]) -> list[str]:
    """Extract key claims from RAG chunks for comparison."""
    claims = []
    for r in rag_results[:10]:
        content = r.get("content", "")
        # Extract sentences with key clinical terms
        meta = r.get("metadata", {})
        source = meta.get("source", "unknown")
        heading = meta.get("heading_hierarchy", "")

        # Find dosage-related sentences
        dosages = re.findall(
            r'(?:dose|dosage|recommend|dosing|建议|剂量|推荐|服用|减量)\s*[：:]*\s*[^。\n]{10,100}[。\n]',
            content,
            re.IGNORECASE,
        )
        for d in dosages:
            claims.append(f"[{source}] {d.strip()}")

        # Find risk-related sentences
        risks = re.findall(
            r'(?:risk|contraindication|caution|avoid|注意|风险|禁忌|避免|谨慎)\s*[：:]*\s*[^。\n]{10,100}[。\n]',
            content,
            re.IGNORECASE,
        )
        for rsk in risks:
            claims.append(f"[{source}] {rsk.strip()}")

    return claims


def detect_conflicts(
    sql_results: list[dict],
    rag_results: list[dict],
) -> list[dict]:
    """
    Compare SQL and RAG results, identifying specific discrepancies.

    Returns a list of conflict records, each with:
      - type: "dosage" | "classification" | "recommendation" | "general"
      - sql_claim: what SQL says
      - rag_claim: what RAG retrieved
      - severity: "high" | "medium" | "low"
    """
    conflicts = []

    if not sql_results or not rag_results:
        return conflicts

    sql = _extract_sql_recommendation(sql_results)
    sql_recommendation = sql.get("drugrecommendation", "").lower()
    sql_classification = sql.get("classification", "").lower()

    rag_claims = _extract_rag_claims(rag_results)

    if not sql_recommendation and not rag_claims:
        return conflicts

    # Check each RAG claim against the SQL recommendation
    for claim in rag_claims:
        claim_lower = claim.lower()

        # Skip if RAG claim aligns with SQL
        if sql_recommendation and any(
            phrase in claim_lower for phrase in sql_recommendation.split()
        ):
            continue

        # Dosage conflict: RAG suggests a different dosage
        if sql_recommendation and not sql_recommendation in claim_lower:
            dosage_conflict = _check_dosage_conflict(sql_recommendation, claim_lower)
            if dosage_conflict:
                conflicts.append({
                    "type": "dosage",
                    "sql_claim": _truncate(sql_recommendation, 150),
                    "rag_claim": _truncate(claim, 300),
                    "detail": dosage_conflict,
                    "severity": "high",
                })
                continue

        # Classification conflict
        if sql_classification and sql_classification not in claim_lower:
            if any(t in claim_lower for t in ["contraindicated", "avoid", "caution",
                                                "moderate", "major", "禁忌", "避免"]):
                conflicts.append({
                    "type": "classification",
                    "sql_claim": _truncate(sql_classification, 150),
                    "rag_claim": _truncate(claim, 300),
                    "detail": "RAG suggests a different risk/evidence level than CPIC SQL",
                    "severity": "medium",
                })

    return conflicts


def _check_dosage_conflict(sql_text: str, rag_text: str) -> Optional[str]:
    """Check if dosage values differ between SQL and RAG."""
    # Extract numbers followed by mg/mcg/g units
    sql_doses = re.findall(r'(\d+)\s*(mg|mcg|µg|g|μg)', sql_text, re.IGNORECASE)
    rag_doses = re.findall(r'(\d+)\s*(mg|mcg|µg|g|μg)', rag_text, re.IGNORECASE)

    if not sql_doses or not rag_doses:
        return None

    # Normalize to mg for comparison
    def to_mg(dose, unit):
        unit = unit.lower()
        if unit in ("mcg", "µg", "μg"):
            return float(dose) / 1000
        if unit == "g":
            return float(dose) * 1000
        return float(dose)

    sql_vals = {to_mg(*d) for d in sql_doses}
    rag_vals = {to_mg(*d) for d in rag_doses}

    # Check if RAG mentions a dosage NOT in SQL
    extra_doses = rag_vals - sql_vals
    if extra_doses:
        return f"RAG mentions dosage(s) {extra_doses} not in SQL recommendation"
    return None


# ── Logging ──────────────────────────────────────────────────────────


def log_gap(
    query: str,
    parsed_intent: dict,
    rag_results: list[dict],
    confidence: str = "low",
    top_score: float = 0.0,
) -> dict:
    """
    Log a query that couldn't be answered by any online layer.

    This creates an audit trail of:
      - What queries couldn't be answered by SQL/rule engine/knowledge base
      - Used to identify knowledge gaps for offline analysis
      - Drives knowledge base iteration via RAG-assisted extraction

    Args:
        query: original user query
        parsed_intent: parsed query intent
        rag_results: results from RAG retrieval
        confidence: RAG confidence level
        top_score: top RAG relevance score

    Returns:
        record dict that was appended to the log.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "parsed_intent": parsed_intent,
        "rag_confidence": confidence,
        "rag_top_score": round(top_score, 3),
        "rag_result_count": len(rag_results),
        "has_relevant_results": any(
            r.get("rerank_score", 0) or (1.0 - (r.get("distance") or 1.0)) > 0.5
            for r in rag_results[:3]
        ),
    }

    log_path = FALLBACK_DIR / "fallback_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    status = "adequate" if record["has_relevant_results"] else "low_confidence"
    print(f"[Fallback] [{status}] {query[:50]} → RAG confidence={confidence}, top_score={top_score:.2f}")

    return record


# ── Query / Review ──────────────────────────────────────────────────


def get_recent_fallbacks(limit: int = 20) -> list[dict]:
    """Return the most recent fallback log records."""
    log_path = FALLBACK_DIR / "fallback_log.jsonl"
    if not log_path.exists():
        return []

    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return records[-limit:]


def get_fallback_summary() -> dict:
    """Return aggregate stats about logged fallback queries."""
    records = get_recent_fallbacks(limit=5000)
    if not records:
        return {"total": 0, "adequate": 0, "low_confidence": 0}

    total = len(records)
    adequate = sum(1 for r in records if r.get("has_relevant_results"))
    low_conf = total - adequate

    # Count by confidence level
    by_confidence: dict[str, int] = {}
    for r in records:
        c = r.get("rag_confidence", "unknown")
        by_confidence[c] = by_confidence.get(c, 0) + 1

    return {
        "total": total,
        "adequate": adequate,
        "low_confidence": low_conf,
        "adequate_rate": round(adequate / total * 100, 1) if total else 0,
        "by_confidence": dict(sorted(by_confidence.items())),
    }


# ── Helpers ──────────────────────────────────────────────────────────


def _truncate(text: str, max_len: int = 200) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


# ── Quick CLI review ────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Review RAG fallback audit log")
    parser.add_argument("--summary", action="store_true", help="Show aggregate stats")
    parser.add_argument("--recent", type=int, default=10, help="Show N most recent entries")
    args = parser.parse_args()

    if args.summary:
        stats = get_fallback_summary()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    else:
        records = get_recent_fallbacks(limit=args.recent)
        print(f"\nUnanswered queries / knowledge gaps ({len(records)} records):\n")
        for r in records:
            status = "✅" if r.get("has_relevant_results") else "⚠️"
            print(f"  {status} [{r['timestamp'][:19]}] {r['query'][:60]}")
            print(f"     confidence={r.get('rag_confidence','?')}, "
                  f"top_score={r.get('rag_top_score', 0):.3f}, "
                  f"results={r.get('rag_result_count', 0)}")
            print()
