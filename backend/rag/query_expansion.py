"""
Query expansion: generates alternative phrasings of the user's query
to improve recall in dense retrieval.

Uses DeepSeek LLM to produce N variants, then runs retrieval on all of them
and merges results.
Results are cached (LRU, 100 entries) to avoid repeated LLM calls for the same query.
"""

import hashlib
import json
from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    QUERY_EXPANSION_NUM,
)

_SYSTEM_PROMPT = """You are a query expansion assistant for a pharmacogenomics (PGx) system.
Given a user query about drugs, genes, or medical conditions, generate {num} different
phrasings that would help a search engine find relevant clinical evidence.

Rules:
- Keep the medical meaning identical.
- Use different terminology where possible (e.g., brand vs generic names, synonyms).
- Write each variant on its own line.
- Do NOT invent facts. Only rephrase what is given.
- Output ONLY the variants, one per line, no numbering or extra text.

Example:
Input: "CYP2D6 *4/*4 metoprolol adverse reaction"
Output:
CYP2D6 poor metabolizer metoprolol side effects
*4/*4 genotype metoprolol tolerability
CYP2D6 *4 homozygote metoprolol toxicity risk
"""

_expansion_cache: dict[str, list[str]] = {}
_MAX_CACHE = 100


def expand_query(user_query: str, num: int = QUERY_EXPANSION_NUM) -> list[str]:
    """Generate expanded queries using LLM (cached)."""
    cache_key = hashlib.md5(f"{user_query}:{num}".encode()).hexdigest()
    if cache_key in _expansion_cache:
        return _expansion_cache[cache_key]

    expanded = [user_query]  # include original

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_api_key_here":
        return expanded

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT.format(num=num)},
                {"role": "user", "content": user_query},
            ],
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        for line in text.split("\n"):
            line = line.strip().strip("- ").strip("* ")
            if line and len(line) > 5 and line not in expanded:
                expanded.append(line)
    except Exception as e:
        print(f"[WARN] Query expansion failed: {e}")

    result = expanded[: num + 1]  # original + up to N variants
    # LRU cache: evict oldest if full
    if len(_expansion_cache) >= _MAX_CACHE:
        _expansion_cache.pop(next(iter(_expansion_cache)))
    _expansion_cache[cache_key] = result
    return result
