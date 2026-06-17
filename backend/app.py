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

    # Debug info collector
    debug = {"steps": []}
    parsed_data = {"原始查询": req.text, "intent": intent.get("intent", "")}
    if intent.get("genes"):
        parsed_data["基因"] = intent["genes"]
    if intent.get("genotypes"):
        parsed_data["基因型"] = intent["genotypes"]
    if intent.get("drug"):
        parsed_data["药物"] = intent["drug"]
    if intent.get("symptom"):
        parsed_data["症状"] = intent["symptom"]
    debug["steps"].append({"name": "查询解析", "data": parsed_data})

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

    # RAG retrieval (only if no structured results — avoid noise)
    rag_response = {"results": [], "confidence": "low", "top_score": 0.0}
    if not structured_results and retriever.store.available:
        rag_response = retriever.search_by_intent(
            query=req.text,
            gene=parsed.genes[0] if parsed.genes else None,
            drug=parsed.drug,
            top_k=req.top_k or 5,
        )
    elif not retriever.store.available:
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
                    "## 无充分证据\n\n"
                    "当前 CPIC 指南中没有与您描述相关的药物基因组学证据。\n\n"
                    "请提供更具体的药物、基因或基因型信息以便进一步查询。\n\n"
                    "> ⚠️ **本结论仅基于 CPIC 指南数据，不排除其他临床证据来源。**"
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

    # Build debug info
    # -- routing decision: show only the PRIMARY route
    routes = []
    if rule_result:
        routes.append(f"✅ 规则引擎（{', '.join(rule_results.keys())}）")
    elif structured_results:
        routes.append(f"✅ CPIC 结构化查询（精确基因型匹配）")
    elif rag_response.get("results"):
        routes.append(f"✅ RAG 语义检索（top_score={rag_response.get('top_score', 0):.2f}）")
    else:
        routes.append("❌ 知识库无匹配 → 返回无充分证据")
    debug["steps"].append({"name": "路由决策", "data": {"路由": routes}})

    # -- RAG details (only when RAG was the PRIMARY source of answer)
    if rag_response.get("results") and not rule_result:
        import hashlib
        seen_content = set()
        rerank_scores = []
        dup_count = 0
        for r in rag_response.get("results", []):
            content_key = hashlib.md5(r.get("content", "")[:80].encode()).hexdigest()
            score = r.get("rerank_score") or (1.0 - (r.get("distance") or 0))
            meta = r.get("metadata", {})
            entry = {
                "score": round(float(score), 3),
                "source": meta.get("source", ""),
                "drug": meta.get("drug", ""),
                "gene": meta.get("gene", ""),
            }
            if content_key in seen_content:
                dup_count += 1
                continue
            seen_content.add(content_key)
            rerank_scores.append(entry)
        dedup_note = ""
        if dup_count > 0:
            dedup_note = f"（原始 {len(rerank_scores) + dup_count} 条，去重合并 {dup_count} 条相似内容）"
        rag_detail = {
            "策略": "语义向量搜索（ChromaDB + bge-base-en-v1.5）",
            "查询改写": rag_response.get("expanded_queries", [req.text]),
            "总检索数": rag_response.get("total_retrieved", 0),
            "rerank 后取 top": len(rerank_scores),
            "证据校验删除": rag_response.get("irrelevant_removed", 0),
            "rerank 得分": rerank_scores,
            "去重说明": dedup_note,
        }
        if rag_response.get("hyde_used"):
            rag_detail["HyDE"] = "已启用"
        debug["steps"].append({"name": "RAG 检索", "data": rag_detail})

    # -- report quality
    debug["steps"].append({
        "name": "报告质量", "data": {
            "置信度": report.get("confidence", "low"),
            "章节数": len(report.get("sections", [])),
            "证据校验": "通过" if not report.get("evidence_warning") else report.get("evidence_warning"),
        }
    })

    return ReportResponse(
        query=req.text,
        parsed_intent=parsed,
        rule_engine_result=rule_result,
        report_text=_format_report(report),
        sources=sources,
        debug_info=debug,
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

    # Evidence check warning (if any)
    if report.get("evidence_warning"):
        lines.append(f"> ⚠️ **证据校验警告**: {report['evidence_warning']}\n")

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
