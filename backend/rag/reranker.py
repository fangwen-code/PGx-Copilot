"""
Cross-encoder reranker for improving retrieval precision.

Takes initial candidate list from ChromaDB (dense retrieval),
re-scores each pair (query, candidate) with a cross-encoder.
Scores are normalized to [0, 1] via sigmoid.
"""

import math
from sentence_transformers import CrossEncoder

from config import CROSS_ENCODER_MODEL, RAG_TOP_K_RERANK


class Reranker:
    """Cross-encoder reranker."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self.model = None
        self.available = False
        if model_name:
            try:
                self.model = CrossEncoder(model_name)
                self.available = True
            except Exception as e:
                print(f"[WARN] Cross-encoder '{model_name}' failed to load: {e}")
                print("[WARN] Rerank disabled. Using distance-based scoring.")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = RAG_TOP_K_RERANK,
    ) -> list[dict]:
        """
        Rerank candidates by cross-encoder score.

        Args:
            query: original user query
            candidates: list of {"content": str, ...} from ChromaDB
            top_k: number to return after reranking

        Returns:
            same list sorted by cross-encoder score descending,
            each item gets a "rerank_score" field.
        """
        if not candidates:
            return []

        # No reranker available, fall back to original distance order
        if not self.available or self.model is None:
            for i, c in enumerate(candidates):
                # Normalize distance-based score to [0, 1]
                dist = c.get("distance") or 0.0
                c["rerank_score"] = max(0.0, min(1.0, 1.0 - dist))
                c["rerank_rank"] = i
            return candidates[:top_k]

        try:
            pairs = [(query, c["content"]) for c in candidates]
            scores = self.model.predict(pairs)
            for i, c in enumerate(candidates):
                # Sigmoid normalize raw logit to [0, 1]
                logit = float(scores[i])
                c["rerank_score"] = 1.0 / (1.0 + math.exp(-logit))
                c["rerank_rank"] = i
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            print(f"[WARN] Rerank prediction failed: {e}")
            # Fall back to distance-based ordering
            for i, c in enumerate(candidates):
                dist = c.get("distance") or 0.0
                c["rerank_score"] = max(0.0, min(1.0, 1.0 - dist))
                c["rerank_rank"] = i
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]
