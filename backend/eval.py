"""
RAG retrieval evaluation — compares baseline vs HyDE vs query expansion.

Tests retrieval quality across curated queries, measuring:
  - recall@k for sources, genes, drugs
  - precision impact of different strategies
  - Confidence distribution

Usage:
    python eval.py --top_k 5                     # default evaluation
    python eval.py --strategy hyde               # compare strategies
    python eval.py --strategy all                # run all strategies and compare
"""

import argparse
import time

from rag.vector_store import VectorStore
from rag.retriever import Retriever

TEST_QUERIES = [
    {"input": "CYP2D6 *4/*4 metoprolol adverse reaction", "expected_sources": ["CPIC"], "expected_genes": ["CYP2D6"], "expected_drugs": ["metoprolol"]},
    {"input": "SLCO1B1 TC simvastatin safety", "expected_sources": ["CPIC"], "expected_genes": ["SLCO1B1"], "expected_drugs": ["simvastatin"]},
    {"input": "hypertension CCB alternative treatment", "expected_drugs": ["amlodipine"]},
    {"input": "CYP2C9 losartan efficacy", "expected_genes": ["CYP2C9"], "expected_drugs": ["losartan"]},
    {"input": "APOE E4 statin efficacy", "expected_genes": ["APOE"], "expected_drugs": ["simvastatin"]},
    {"input": "NAT2 hydralazine lupus risk", "expected_genes": ["NAT2"], "expected_drugs": ["hydralazine"]},
    {"input": "warfarin CYP2C9 VKORC1 dosing", "expected_genes": ["CYP2C9", "VKORC1"], "expected_drugs": ["warfarin"]},
    {"input": "clopidogrel CYP2C19 poor metabolizer", "expected_genes": ["CYP2C19"], "expected_drugs": ["clopidogrel"]},
    {"input": "thiopurine TPMT NUDT15 intermediate metabolizer", "expected_genes": ["TPMT", "NUDT15"]},
    {"input": "UGT1A1 irinotecan toxicity", "expected_genes": ["UGT1A1"], "expected_drugs": ["irinotecan"]},
    {"input": "metoprolol cough side effect", "expected_drugs": ["metoprolol"]},
    {"input": "losartan poor metabolizer CYP2C9 *1/*3", "expected_genes": ["CYP2C9"], "expected_drugs": ["losartan"]},
]


def evaluate(strategy: str = "default", top_k: int = 5):
    """Run evaluation for a given retrieval strategy."""
    retriever = Retriever()
    store = VectorStore()

    print(f"\n{'='*60}")
    print(f"Strategy: {strategy.upper()}  (top_k={top_k})")
    print(f"{'='*60}")

    total = len(TEST_QUERIES)
    src_ok = gene_ok = drug_ok = 0
    src_t = gene_t = drug_t = 0
    conf_dist = {"high": 0, "medium": 0, "low": 0}
    total_time = 0

    for q in TEST_QUERIES:
        t0 = time.time()

        if strategy == "baseline":
            # Simple vector search only
            results = store.search(q["input"], top_k=top_k)
            rag_resp = {"results": results, "confidence": "medium", "top_score": 0.5}
        else:
            # Full retriever with expansion, HyDE, rerank
            rag_resp = retriever.search(
                q["input"], top_k=top_k,
                use_hyde=(strategy in ("hyde", "all", "default")),
                use_expansion=(strategy in ("expansion", "all", "default")),
                use_self_rag=False,
            )

        elapsed = time.time() - t0
        total_time += elapsed

        results = rag_resp.get("results", [])
        conf = rag_resp.get("confidence", "low")
        conf_dist[conf] = conf_dist.get(conf, 0) + 1

        retrieved_text = " ".join([r.get("content", "") for r in results]).lower()

        s_ok = all(s.lower() in retrieved_text for s in q.get("expected_sources", [])) if q.get("expected_sources") else True
        g_ok = all(g.lower() in retrieved_text for g in q.get("expected_genes", [])) if q.get("expected_genes") else True
        d_ok = all(d.lower() in retrieved_text for d in q.get("expected_drugs", [])) if q.get("expected_drugs") else True

        if q.get("expected_sources"):
            src_ok += 1 if s_ok else 0; src_t += 1
        if q.get("expected_genes"):
            gene_ok += 1 if g_ok else 0; gene_t += 1
        if q.get("expected_drugs"):
            drug_ok += 1 if d_ok else 0; drug_t += 1

        status = "OK" if (s_ok and g_ok and d_ok) else "MISS"
        top_score = rag_resp.get("top_score", 0)
        print(f"  [{status}] {q['input'][:45]:45s} {conf:6s} score={top_score:.3f}  ({elapsed:.1f}s)")

    # Summary
    print(f"\n--- Results [{strategy}] ---")
    print(f"  Source recall: {src_ok}/{src_t} = {src_ok/src_t*100:.0f}%" if src_t else "")
    print(f"  Gene recall:   {gene_ok}/{gene_t} = {gene_ok/gene_t*100:.0f}%" if gene_t else "")
    print(f"  Drug recall:   {drug_ok}/{drug_t} = {drug_ok/drug_t*100:.0f}%" if drug_t else "")
    print(f"  Confidence:    high={conf_dist['high']} medium={conf_dist['medium']} low={conf_dist['low']}")
    print(f"  Avg time:      {total_time/total:.1f}s/query")
    print(f"  Collection:    {store.count()} chunks")

    return {"src_recall": src_ok/src_t if src_t else 0,
            "gene_recall": gene_ok/gene_t if gene_t else 0,
            "drug_recall": drug_ok/drug_t if drug_t else 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--strategy", default="default",
                        choices=["default", "baseline", "hyde", "expansion", "all"])
    args = parser.parse_args()

    if args.strategy == "all":
        results = {}
        for s in ["baseline", "expansion", "hyde"]:
            results[s] = evaluate(s, args.top_k)
        print("\n\n=== COMPARISON ===")
        print(f"{'Strategy':<15} {'Source Recall':>15} {'Gene Recall':>15} {'Drug Recall':>15}")
        print("-" * 60)
        for s, r in results.items():
            print(f"{s:<15} {r['src_recall']:>13.0%} {r['gene_recall']:>13.0%} {r['drug_recall']:>13.0%}")
    else:
        evaluate(args.strategy, args.top_k)
