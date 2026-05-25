"""
RAG retrieval pipeline.

Orchestrates:
  1. Query expansion (LLM-based alternative phrasings)
  2. HyDE (hypothetical document embedding)
  3. Dense retrieval (ChromaDB)
  4. Cross-encoder rerank
  5. Self-RAG relevance filter
  6. Confidence scoring
"""

from config import (
    CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD,
    HYDE_ENABLED, QUERY_EXPANSION_ENABLED,
)
from rag.vector_store import VectorStore
from rag.reranker import Reranker
from rag.query_expansion import expand_query
from rag.hyde import generate_hypothetical_document
from rag.self_rag import SelfRAG


class Retriever:
    """Full retrieval pipeline with expansion, HyDE, rerank, and self-check."""

    def __init__(self):
        self.store = VectorStore()
        self.reranker = Reranker()
        self.self_rag = SelfRAG()

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_filter: str | None = None,
        metadata_filter: dict | None = None,
        use_hyde: bool = HYDE_ENABLED,
        use_expansion: bool = QUERY_EXPANSION_ENABLED,
        use_rerank: bool = True,
        use_self_rag: bool = True,
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
                    "hyde_used": hyde_doc is not None, "irrelevant_removed": 0}

        # Step 4: Cross-encoder rerank
        if use_rerank:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)

        # Step 5: Self-RAG relevance filter
        irrelevant_removed = 0
        if use_self_rag and self.self_rag._llm_available:
            candidates, irrelevant_removed = self.self_rag.filter_by_relevance(query, candidates)

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
            # Fall back to unfiltered search
            print(f"[Retriever] Filtered search returned 0 results for {metadata_filter}, falling back to semantic search")

        return self.search(targeted, top_k=top_k)
