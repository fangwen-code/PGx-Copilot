"""
HyDE (Hypothetical Document Embedding).

Generates a hypothetical document that would answer the query,
embeds that document instead of the query, and uses it for retrieval.

This bridges the gap between short user queries and long document chunks.
Results are cached (LRU, 100 entries).
"""

import hashlib
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

_SYSTEM_PROMPT = """You are a clinical pharmacogenomics expert.
Write a short paragraph (3-5 sentences) that would be the perfect document
answering the user's question. Write it as if it were an excerpt from
a CPIC guideline or clinical reference.

Do NOT begin with "Here is a hypothetical document" — just write the content.

Include relevant drug names, gene names, and clinical implications.
If you're unsure of specific numbers or recommendations, use general clinical language.
"""

_hyde_cache: dict[str, str | None] = {}
_MAX_CACHE = 100


def generate_hypothetical_document(user_query: str) -> str | None:
    """Generate a hypothetical document that would answer the query (cached)."""
    cache_key = hashlib.md5(user_query.encode()).hexdigest()
    if cache_key in _hyde_cache:
        return _hyde_cache[cache_key]

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_api_key_here":
        return None

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        result = text if len(text) > 20 else None
    except Exception as e:
        print(f"[WARN] HyDE generation failed: {e}")
        result = None

    if len(_hyde_cache) >= _MAX_CACHE:
        _hyde_cache.pop(next(iter(_hyde_cache)))
    _hyde_cache[cache_key] = result
    return result
