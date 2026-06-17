"""
RAG retrieval pipeline.

Orchestrates:
  1. Query expansion (LLM-based alternative phrasings)
  2. HyDE (hypothetical document embedding)
  3. Dense retrieval (ChromaDB)
  4. Cross-encoder rerank
  5. Evidence-based relevance filter
  6. Confidence scoring
"""

import json
from pathlib import Path

from config import (
    CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD,
    HYDE_ENABLED, QUERY_EXPANSION_ENABLED, DATA_DIR,
)
from rag.vector_store import VectorStore
from rag.reranker import Reranker
from rag.query_expansion import expand_query
from rag.hyde import generate_hypothetical_document
from rag.evidence_verifier import EvidenceVerifier


class Retriever:
    """Full retrieval pipeline with expansion, HyDE, rerank, and evidence check."""

    def __init__(self):
        self.store = VectorStore()
        self.reranker = Reranker()
        self.evidence_checker = EvidenceVerifier()

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_filter: str | None = None,
        metadata_filter: dict | None = None,
        use_hyde: bool = HYDE_ENABLED,
        use_expansion: bool = QUERY_EXPANSION_ENABLED,
        use_rerank: bool = True,
        use_evidence_check: bool = True,
    ) -> dict:
        """
        Full retrieval pipeline.

        Args:
            query: user query
            top_k: number of results to return
            source_filter: restrict to specific source (e.g. "CPIC")
            metadata_filter: restrict by metadata fields (e.g. {"drug": "metoprolol"})
            ...
        """
        where = {"source": source_filter} if source_filter else {}
        if metadata_filter:
            where.update(metadata_filter)
        where = where or None

        # Step 1: Query expansion
        queries = [query]
        if use_expansion:
            expanded = expand_query(query)
            if len(expanded) > 1:
                queries = expanded

        # Step 2: HyDE
        hyde_doc = None
        if use_hyde:
            hyde_doc = generate_hypothetical_document(query)

        # Step 3: Retrieve for each query variant
        seen = {}
        for q in queries:
            candidates = self.store.search(q, top_k=top_k, where=where)
            # Also search with HyDE document if available
            for c in candidates:
                if c["id"] not in seen:
                    seen[c["id"]] = c

        if hyde_doc:
            hyde_candidates = self.store.search(hyde_doc, top_k=top_k, where=where)
            for c in hyde_candidates:
                if c["id"] not in seen:
                    seen[c["id"]] = c

        candidates = list(seen.values())

        if not candidates:
            return {"results": [], "confidence": "low", "top_score": 0.0,
                    "total_retrieved": 0, "expansions_used": len(queries) - 1,
                    "expanded_queries": queries, "hyde_used": hyde_doc is not None,
                    "irrelevant_removed": 0}

        # Step 4: Cross-encoder rerank
        if use_rerank:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)

        # Step 5: Evidence-based relevance filter
        irrelevant_removed = 0
        if use_evidence_check and self.evidence_checker._llm_available:
            candidates, irrelevant_removed = self.evidence_checker.filter_by_relevance(query, candidates)

        # Step 6: Confidence scoring (based on rerank score or distance)
        if candidates:
            if candidates[0].get("rerank_score") is not None:
                top_score = candidates[0]["rerank_score"]
            else:
                top_score = 1.0 - (candidates[0].get("distance") or 0.0)

            if top_score >= (1.0 - CONFIDENCE_HIGH_THRESHOLD):
                confidence = "high"
            elif top_score >= (1.0 - CONFIDENCE_MEDIUM_THRESHOLD):
                confidence = "medium"
            else:
                confidence = "low"
        else:
            top_score = 0.0
            confidence = "low"

        return {
            "results": candidates[:top_k],
            "confidence": confidence,
            "top_score": top_score,
            "total_retrieved": len(candidates),
            "expansions_used": len(queries) - 1,
            "expanded_queries": queries,
            "hyde_used": hyde_doc is not None,
            "irrelevant_removed": irrelevant_removed,
        }

    def search_by_intent(
        self,
        query: str,
        gene: str | None = None,
        drug: str | None = None,
        top_k: int = 10,
    ) -> dict:
        """Intent-aware search with metadata fallback chain.

        Strategy:
          1. If drug or gene is known, try metadata-filtered search first
          2. If filtered search returns nothing, fall back to unfiltered semantic search
        """
        if gene and drug:
            targeted = f"{drug} {gene} PGx pharmacogenomics clinical recommendation"
        elif drug:
            targeted = f"{drug} pharmacogenomics PGx dosing recommendation"
        elif gene:
            targeted = f"{gene} pharmacogenomics allele phenotype recommendation"
        else:
            targeted = query

        # Try metadata-filtered search first (exact match on drug/gene)
        metadata_filter = {}
        if drug:
            metadata_filter["drug"] = drug
        if gene and not drug:
            metadata_filter["gene"] = gene

        if metadata_filter:
            result = self.search(targeted, top_k=top_k, metadata_filter=metadata_filter)
            if result.get("results"):
                return result
            # Fall back to unfiltered search — and log the coverage gap
            _log_coverage_gap(query, metadata_filter)

        return self.search(targeted, top_k=top_k)


_COVERAGE_LOG = DATA_DIR / "coverage_gaps.jsonl"


def _log_coverage_gap(query: str, metadata_filter: dict):
    """Log unmatched metadata filters to a JSONL file for coverage analysis."""
    try:
        _COVERAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {"query": query, "filter": metadata_filter}
        with open(str(_COVERAGE_LOG), "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[Coverage] Logged gap: {metadata_filter}")
    except Exception as e:
        pass  # best-effort logging
