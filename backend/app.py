"""PGx-Copilot: FastAPI backend — wired with CPIC query + RAG + report generation."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import HOST, PORT
from models.schemas import QueryRequest, ReportResponse, SourceRef, QueryIntent
from rule_engine.registry import evaluate_all
from query_understanding.parser import parse_query
from rag.retriever import Retriever
from report_generator.generator import generate_report

app = FastAPI(title="PGx-Copilot", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=ReportResponse)
async def query(req: QueryRequest):
    """
    Main endpoint:
    1. Parse user intent
    2. Route to CPIC structured query and/or RAG
    3. Generate report
    """
    try:
        return _handle_query(req)
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return ReportResponse(
            query=req.text,
            parsed_intent=QueryIntent(),
            rule_engine_result=None,
            report_text="系统处理请求时出现错误，请稍后重试。",
            sources=[],
        )


def _handle_query(req: QueryRequest) -> ReportResponse:
    """Synchronous query handler with structured error handling."""
    # Step 1: Query Understanding
    intent = parse_query(req.text)
    parsed = QueryIntent(**intent)

    # Step 2: Determine routes
    has_exact_genotype = bool(parsed.genotypes and parsed.drug)

    # CPIC structured query (exact genotype match)
    structured_results = None
    if has_exact_genotype:
        try:
            from cpic.query import CPICQuery
            q = CPICQuery()
            for gene, gt in parsed.genotypes.items():
                structured_results = q.match_genotype(gene, gt)
            q.close()
        except Exception as e:
            print(f"[WARN] CPIC query failed: {e}")

    # RAG retrieval (for all inputs)
    rag_response = {"results": [], "confidence": "low", "top_score": 0.0}
    if retriever.store.available:
        rag_response = retriever.search_by_intent(
            query=req.text,
            gene=parsed.genes[0] if parsed.genes else None,
            drug=parsed.drug,
            top_k=req.top_k or 5,
        )
    else:
        print("[WARN] Vector store unavailable, skipping RAG retrieval")

    # Rule engine (registered engines matching input genotypes)
    rule_results = evaluate_all(parsed.genotypes or {})
    rule_result = next(iter(rule_results.values())) if rule_results else None

    # Step 2b: Early rejection for unrelated queries (no structured data, poor RAG scores)
    rag_results_list = rag_response.get("results", [])
    if not structured_results and not rule_result:
        top_score = rag_response.get("top_score", 0)
        if top_score <= 0 or not rag_results_list:
            return ReportResponse(
                query=req.text,
                parsed_intent=parsed,
                rule_engine_result=None,
                report_text=(
                    "## 无法评估\n\n"
                    "当前知识库中没有与您描述相关的药物基因组学信息。\n\n"
                    "请提供更具体的药物、基因或基因型信息以便进一步查询。"
                ),
                sources=[],
            )

    # Step 3: Generate report
    report = generate_report(
        user_query=req.text,
        parsed_intent=intent,
        structured_results=structured_results,
        rag_results=rag_response.get("results", []),
        rule_engine_result=rule_result,
    )

    # Step 4: Build source references
    sources = _build_sources(rag_response.get("results", []), structured_results)

    return ReportResponse(
        query=req.text,
        parsed_intent=parsed,
        rule_engine_result=rule_result,
        report_text=_format_report(report),
        sources=sources,
    )


def _build_sources(
    rag_results: list[dict],
    structured_results: list[dict] | None,
) -> list[SourceRef]:
    """Build source list. Only include high relevance results (>= 0.7)."""
    sources = []
    seen = set()

    for r in rag_results[:5]:
        meta = r.get("metadata", {})
        src_key = f"{meta.get('source', '')}-{meta.get('pmid', '')}-{meta.get('drug', '')}"
        if src_key in seen:
            continue
        seen.add(src_key)

        # Prefer rerank_score; fall back to distance; clamp to >= 0
        score = r.get("rerank_score") or (1.0 - (r.get("distance") or 0))
        score = max(0.0, float(score))

        # Only show high-confidence sources to users
        if score < 0.7:
            continue

        sources.append(SourceRef(
            id=len(sources) + 1,
            content=_clean_chunk_content(r["content"]),
            source=meta.get("source", "CPIC"),
            year=meta.get("year", ""),
            drug=meta.get("drug", ""),
            gene=meta.get("gene", ""),
            relevance=round(float(score), 3),
        ))

    return sources


def _clean_chunk_content(content: str) -> str:
    """Strip technical metadata lines from chunk content for user display."""
    lines = content.split("\n")
    cleaned = []
    for line in lines:
        if line.startswith("[") and line.strip().endswith("]"):
            continue
        stripped = line.strip()
        if stripped.startswith("Drug:") or stripped.startswith("Genotype:") or \
           stripped.startswith("Classification:") or stripped.startswith("Phenotype:"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()[:500]


def _format_report(report: dict) -> str:
    """Convert structured report dict to formatted text."""
    if not report.get("sections"):
        return "（无法生成报告）"

    lines = []

    # Self-RAG warning (if any)
    if report.get("self_rag_warning"):
        lines.append(f"> ⚠️ **自查警告**: {report['self_rag_warning']}\n")

    for sec in report["sections"]:
        lines.append(f"## {sec['title']}")
        lines.append(sec["content"])
        lines.append("")

    if report.get("disclaimer"):
        lines.append(f"*{report['disclaimer']}*")

    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
