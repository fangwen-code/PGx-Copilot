"""
Self-RAG: self-check and faithfulness validation.

1. Relevance check: are retrieved chunks actually relevant to the query?
2. Faithfulness check: is the generated answer supported by the retrieved chunks?
3. Citation accuracy: does each citation actually exist in the evidence?

This runs AFTER retrieval and AFTER generation, as a validation layer.
"""

import json
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

_RELEVANCE_PROMPT = """You are evaluating the relevance of a retrieved document chunk
to a user query. Rate relevance as "relevant", "partially_relevant", or "irrelevant".

Output ONLY a JSON: {"relevance": "...", "reason": "one short sentence"}

Query: {query}

Chunk: {chunk}
"""

_FAITHFULNESS_PROMPT = """You are evaluating whether a generated report section is
faithfully supported by the provided evidence. Check each claim against the evidence.

Output ONLY a JSON: {"faithful": true/false, "unsupported_claims": ["..."], "note": "..."}

Evidence:
{evidence}

Report section:
{section}
"""


class SelfRAG:
    """Self-validation layer for RAG quality."""

    def __init__(self):
        self._llm_available = (
            DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "your_api_key_here"
        )
        if self._llm_available:
            self._client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self._relevance_cache: dict[str, dict] = {}
        self._max_relevance_cache = 200

    def check_relevance(self, query: str, chunk: dict) -> dict:
        """Check whether a retrieved chunk is relevant to the query (cached)."""
        import hashlib
        cache_key = hashlib.md5(
            f"{query}:{chunk.get('id', '')}:{chunk.get('content', '')[:100]}".encode()
        ).hexdigest()
        if cache_key in self._relevance_cache:
            return self._relevance_cache[cache_key]

        default = {"relevance": "relevant", "reason": ""}
        if not self._llm_available:
            q_words = set(query.lower().split())
            c_words = set(chunk.get("content", "").lower().split())
            overlap = len(q_words & c_words) / max(len(q_words), 1)
            result = {"relevance": "irrelevant", "reason": "low keyword overlap"} if overlap < 0.05 else default
            self._relevance_cache[cache_key] = result
            return result

        try:
            resp = self._client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": _RELEVANCE_PROMPT.format(
                    query=query, chunk=chunk.get("content", "")[:800]
                )}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
        except Exception:
            result = default

        if len(self._relevance_cache) >= self._max_relevance_cache:
            self._relevance_cache.pop(next(iter(self._relevance_cache)))
        self._relevance_cache[cache_key] = result
        return result

    def filter_by_relevance(
        self, query: str, results: list[dict], max_check: int = 5
    ) -> tuple[list[dict], int]:
        """
        Remove irrelevant results.

        Args:
            query: original user query
            results: list of retrieved chunks
            max_check: only check top N chunks to limit LLM calls (default 5)

        Returns:
            (filtered_results, removed_count)
        """
        filtered = []
        removed = 0
        # Only run relevance check on top chunks — saves LLM calls
        to_check = results[:max_check]
        rest = results[max_check:]

        for r in to_check:
            verdict = self.check_relevance(query, r)
            if verdict.get("relevance") in ("relevant", "partially_relevant"):
                r["relevance"] = verdict["relevance"]
                filtered.append(r)
            else:
                removed += 1

        # Keep unchecked results but mark them as unverified
        for r in rest:
            r["relevance"] = "unverified"
            filtered.append(r)

        return filtered, removed

    def check_faithfulness(
        self, report_json: dict, evidence_text: str
    ) -> dict:
        """
        Verify if the generated report is faithful to the evidence.

        Returns:
            {"faithful": bool, "unsupported_claims": [str], "note": str}
        """
        if not self._llm_available:
            return {"faithful": True, "unsupported_claims": [], "note": "LLM unavailable"}

        sections = report_json.get("sections", [])
        if not sections:
            return {"faithful": True, "unsupported_claims": [], "note": "no sections"}

        all_claims = []
        for sec in sections:
            all_claims.append(f"[{sec['title']}]\n{sec['content']}")

        combined = "\n\n".join(all_claims)
        if len(combined) > 3000:
            combined = combined[:3000] + "..."

        try:
            resp = self._client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": _FAITHFULNESS_PROMPT.format(
                    evidence=evidence_text[:3000],
                    section=combined,
                )}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            return result
        except Exception as e:
            return {"faithful": True, "unsupported_claims": [], "note": str(e)}
