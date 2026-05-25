"""
LLM report generator with Self-RAG faithfulness validation.

Pipeline:
  1. Build evidence context from structured + RAG results
  2. Generate clinical report via LLM (constrained to evidence)
  3. Self-RAG check: verify faithfulness of generated report against evidence
  4. If unfaithful claims found, regenerate or flag them
"""

import json
import hashlib
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, SELF_RAG_ENABLED

_cache: dict[str, dict] = {}


_SYSTEM_PROMPT = """You are a clinical pharmacogenomics decision support system.

Generate a structured clinical report based ONLY on the provided evidence below.
Do NOT add medical knowledge from your training.

Rules:
1. Every claim MUST be supported by at least one cited source in the evidence.
2. If evidence is insufficient for any section, state "No sufficient evidence" — do NOT guess.
3. Keep content clean — do NOT add inline citations like [Source: xxx] or [PMID: xxx] in the content text. Use the citations list instead.
4. Structure: Risk Assessment, Therapeutic Recommendation, Alternative Options, Cautions.

Output JSON:
{
  "sections": [{"title": str, "content": str, "citations": [str]}],
  "confidence": "high"|"medium"|"low",
  "disclaimer": str,
  "has_sufficient_evidence": bool
}
"""


def generate_report(
    user_query: str,
    parsed_intent: dict,
    structured_results: list[dict] | None,
    rag_results: list[dict],
    rule_engine_result: str | None = None,
) -> dict:
    """Generate and self-validate a clinical report."""
    cache_key = hashlib.md5(
        json.dumps({"q": user_query, "r": len(rag_results)}, sort_keys=True).encode()
    ).hexdigest()
    if cache_key in _cache:
        return _cache[cache_key]

    # Build evidence
    evidence_parts = []
    if structured_results:
        evidence_parts.append("=== CPIC Structured Data (exact match) ===")
        for r in structured_results[:5]:
            lines = [f"{k}: {v}" for k, v in r.items() if v and k in
                     ("drugrecommendation", "implications", "classification",
                      "phenotypes", "comments", "lookupkey")]
            if lines:
                evidence_parts.extend([*lines, "---"])

    if rag_results:
        evidence_parts.append("=== Retrieved Evidence (semantic match) ===")
        for r in rag_results[:5]:
            evidence_parts.append(r.get("content", ""))
            meta = r.get("metadata", {})
            if meta.get("pmid"):
                evidence_parts.append(f"[PMID: {meta['pmid']}]")
            if meta.get("source"):
                evidence_parts.append(f"[Source: {meta['source']}]")
            evidence_parts.append("---")

    if rule_engine_result:
        evidence_parts.append("=== Rule Engine (statin PGx) ===")
        evidence_parts.append(rule_engine_result)

    evidence_text = "\n".join(evidence_parts)

    if not evidence_text.strip():
        return _no_evidence_response()

    # Confidence
    if structured_results:
        confidence = "high"
    elif rag_results:
        score = rag_results[0].get("rerank_score") or (1.0 - (rag_results[0].get("distance") or 1.0))
        if score >= 0.75:
            confidence = "high"
        elif score >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        confidence = "low"

    # Generate
    report = _call_llm(user_query, parsed_intent, evidence_text, confidence)

    # Self-RAG: faithfulness check
    if SELF_RAG_ENABLED and report.get("sections"):
        from rag.self_rag import SelfRAG
        checker = SelfRAG()
        check = checker.check_faithfulness(report, evidence_text)
        if not check.get("faithful", True):
            unsupported = check.get("unsupported_claims", [])
            print(f"[Self-RAG] Found {len(unsupported)} unsupported claim(s): {unsupported[:2]}")
            report["self_rag_warning"] = (
                f"The following claims could not be verified against the provided evidence: "
                f"{'; '.join(unsupported[:3])}"
            )
            report["confidence"] = "low"

    # Only override confidence if unset — let LLM judgment stand
    if not report.get("confidence"):
        report["confidence"] = confidence

    has_structured = bool(structured_results)
    has_good_rag = bool(rag_results) and (rag_results[0].get("rerank_score", 0) or 0) > 0.3
    report["has_sufficient_evidence"] = has_structured or has_good_rag
    _cache[cache_key] = report
    return report


def _call_llm(user_query, parsed_intent, evidence_text, confidence) -> dict:
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_api_key_here":
        return _fallback_report(user_query, None, None, None)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"User query: {user_query}\n\n"
            f"Parsed intent: {json.dumps(parsed_intent, ensure_ascii=False)}\n\n"
            f"Available evidence:\n{evidence_text}\n\n"
            f"Confidence: {confidence}\n\nGenerate JSON report."
        )},
    ]
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL, messages=messages,
            temperature=0.1, response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[WARN] LLM report failed: {e}")
        return _fallback_report(user_query, None, None, None)


def _no_evidence_response() -> dict:
    return {
        "sections": [{
            "title": "无法评估",
            "content": "当前知识库中没有匹配的临床证据，无法生成报告。",
            "citations": [],
        }],
        "confidence": "low",
        "disclaimer": "无匹配证据。",
        "has_sufficient_evidence": False,
    }


def _fallback_report(user_query, struct_results, rag_results, rule_result) -> dict:
    sections = []
    if struct_results:
        for r in struct_results[:3]:
            dr = r.get("drugrecommendation", "")
            if dr:
                sections.append({"title": "用药建议", "content": dr, "citations": ["CPIC"]})
                break
    if rag_results:
        for r in rag_results[:2]:
            sections.append({
                "title": "相关临床证据",
                "content": r.get("content", "")[:300],
                "citations": [r.get("metadata", {}).get("source", "RAG")],
            })
    if rule_result:
        sections.insert(0, {"title": "他汀类药物评估（规则引擎）",
                           "content": rule_result, "citations": ["CPIC SLCO1B1 + APOE"]})
    if not sections:
        sections.append({"title": "无法评估", "content": "当前知识库中没有匹配的临床证据。", "citations": []})
    return {"sections": sections, "confidence": "low", "disclaimer": "自动生成，仅供参考。",
            "has_sufficient_evidence": bool(struct_results or rule_result)}
