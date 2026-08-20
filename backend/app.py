"""PGx-Copilot: FastAPI backend — multi-source PGx knowledge retrieval and exploration tool."""

import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import HOST, PORT, CPIC_VERSION, CHINESE_DOC_TITLES
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

_STATIN_GENES = {"SLCO1B1", "APOE", "rs4149056", "rs7412", "rs429358"}
_STATIN_DRUGS = {"simvastatin", "atorvastatin", "rosuvastatin", "pravastatin",
                 "pitavastatin", "fluvastatin", "lovastatin"}
# Chinese statin terms — the LLM parser often returns "他汀" / "辛伐他汀" as the
# drug value, which the English-only _STATIN_DRUGS set would miss.
_STATIN_TERMS = ("他汀", "statin", "辛伐他汀", "阿托伐他汀", "瑞舒伐他汀",
                 "普伐他汀", "氟伐他汀", "洛伐他汀", "匹伐他汀")


def _has_statin_content(genes: list[str] | None, drug: str | None,
                        drug_class: str | None = None) -> bool:
    """Check if query involves statin-related genes, drugs, or drug class."""
    if genes and _STATIN_GENES & set(genes):
        return True
    if drug:
        d = drug.lower()
        if d in _STATIN_DRUGS or any(t in d for t in _STATIN_TERMS):
            return True
    if drug_class and drug_class.lower() in ("statin", "他汀", "他汀类"):
        return True
    return False


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=ReportResponse)
async def query(req: QueryRequest):
    """
    Main endpoint:
    1. Parse user intent
    2. Multi-source retrieval:
       - CPIC SQL: structured guideline lookup
       - Rule engine: statin risk assessment
       - RAG (PubMed): gene/drug mechanism background
       - RAG (Chinese guideline): local consensus context
    3. Synthesize results into an exploration report
    """
    try:
        return _handle_query(req)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Query failed: {e}")
        return ReportResponse(
            query=req.text,
            parsed_intent=QueryIntent(),
            rule_engine_result=None,
            report_text="系统处理请求时出现错误，请稍后重试。",
            sources=[],
        )


def _handle_query(req: QueryRequest) -> ReportResponse:
    """Synchronous query handler with dual-channel routing."""
    # Step 1: Query Understanding
    intent = parse_query(req.text)
    parsed = QueryIntent(**intent)

    has_exact_genotype = bool(parsed.genotypes and parsed.drug)
    has_gene_or_drug = bool(parsed.genes or parsed.drug or parsed.genotypes)
    has_statin = _has_statin_content(parsed.genes, parsed.drug, parsed.drug_class)

    # Debug info
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

    # ── CPIC Channel ──
    # Layer 1: drug-first exact match (Level A/B only). A recommendation exists
    # for a (gene, genotype) × drug pair, so matching starts from the parsed
    # DRUG, inspects its own recommendation keys, converts the input genotypes
    # to each key's value type (star / score / phenotype / function), and
    # matches — multi-gene keys handled by construction. Both a gene/genotype
    # AND a drug are required to enter this channel (has_exact_genotype);
    # missing either → no recommendation.
    #
    # Statin-related genes (SLCO1B1/APOE/rs...) are routed EXCLUSIVELY to the
    # statin rule engine — their CPIC rows are function-keyed multi-gene and add
    # nothing the rule engine doesn't decide more directly. Other genes + a
    # statin drug (e.g. CYP2C9-fluvastatin) still go through CPIC.
    statin_genes = _STATIN_GENES & set(parsed.genotypes or {})
    structured_results = None
    if has_exact_genotype and not statin_genes:
        try:
            from cpic.query import CPICQuery
            q = CPICQuery()
            structured_results = q.match_drug_genotypes(parsed.drug, parsed.genotypes)
            # Fallback: some guidelines key recommendations by PHENOTYPE, not
            # allele diplotype (e.g. CYP2D6-metoprolol stores 'PM' not '*4/*4').
            # When the drug-first match finds nothing, match by drug + gene so
            # the CPIC drugrecommendation is still surfaced.
            if not structured_results:
                for gene in (parsed.genes or [])[:2]:
                    result = q.get_recommendation(drug_name=parsed.drug, gene_symbol=gene)
                    if result:
                        structured_results = result
                        break
            q.close()
        except Exception as e:
            print(f"[WARN] CPIC query failed: {e}")

    # CPIC RAG: PubMed abstracts (gene/drug background knowledge).
    # Statin queries are covered by the rule engine + Chinese guideline,
    # so the generic PubMed background is skipped for them (avoids a
    # redundant "RAG: CPIC 摘要" route and CPIC background section).
    cpic_rag = {"results": [], "top_score": 0.0}
    if has_gene_or_drug and not has_statin and retriever.store.available:
        try:
            cpic_rag = retriever.search_by_intent(
                query=req.text,
                gene=parsed.genes[0] if parsed.genes else None,
                drug=parsed.drug,
                top_k=3,
                source_filter="pubmed",
            )
        except Exception as e:
            print(f"[WARN] CPIC RAG failed: {e}")

    # ── Statin Channel ──
    # Rule engine (SLCO1B1/APOE)
    rule_results = evaluate_all(parsed.genotypes or {})
    rule_result = next(iter(rule_results.values())) if rule_results else None

    # Statin RAG: Chinese guideline PDFs.
    # (The root cause of the earlier Chinese hybrid regression — BM25 branch
    # missing the gene/drug metadata filter — is fixed in hybrid.py, so hybrid
    # is used uniformly across sources.)
    statin_rag = {"results": [], "top_score": 0.0}
    if has_statin and retriever.store.available:
        try:
            statin_rag = retriever.search_by_intent(
                query=req.text,
                gene=parsed.genes[0] if parsed.genes else None,
                drug=parsed.drug,
                top_k=3,
                source_filter="chinese_guideline",
            )
        except Exception as e:
            print(f"[WARN] Statin RAG failed: {e}")

    # ── Gene / drug background (CPIC) ──
    # Gene-function / drug background from the CPIC database so the report can
    # always write a Gene Summary / Drug Summary, even when RAG has no hits.
    gene_background: dict[str, dict] = {}
    drug_info: list[dict] = []
    if parsed.genes or parsed.drug:
        try:
            from cpic.query import CPICQuery
            q = CPICQuery()
            for gene in (parsed.genes or [])[:2]:
                gb = {
                    "info": q.get_gene_info(gene),
                    "alleles": q.get_alleles(gene),
                    "results": q.get_gene_result(gene),
                }
                if any(gb.values()):
                    gene_background[gene] = gb
            if parsed.drug:
                drug_info = q.get_drug_info(parsed.drug) or []
            q.close()
        except Exception as e:
            print(f"[WARN] CPIC background fetch failed: {e}")

    # ── Gap logging + early rejection ──
    has_any_answer = bool(
        structured_results or rule_result
        or cpic_rag.get("results") or statin_rag.get("results")
        or gene_background or drug_info
    )
    if not has_any_answer:
        try:
            from tools.conflict_logger import log_gap
            log_gap(query=req.text, parsed_intent=intent,
                    rag_results=[], confidence="no_source", top_score=0.0)
        except Exception as e:
            print(f"[WARN] Gap logging failed: {e}")
        return ReportResponse(
            query=req.text, parsed_intent=parsed, rule_engine_result=None,
            report_text="## 无充分证据\n\n当前知识库中没有与您描述相关的药物基因组学证据。\n\n请提供更具体的药物、基因或基因型信息以便进一步查询。\n\n> ⚠️ **本结果仅基于现有知识库数据，不排除其他来源存在相关信息。**",
            sources=[],
        )

    # ── Report generation ──
    # RAG hits are passed as evidence so the LLM can write Gene Summary /
    # Clinical Relevance from the actual guideline content, not only from
    # CPIC structured data + the rule engine.
    # A failure here must NOT take down the whole request: the deterministic
    # recommendation (below) is the medically-important part. Degrade to an
    # empty-report fallback instead, and log the full traceback.
    try:
        report = generate_report(
            user_query=req.text,
            parsed_intent=intent,
            structured_results=structured_results,
            rule_engine_result=rule_result,
            rag_results={
                "chinese_guideline": statin_rag.get("results", []),
                "pubmed": cpic_rag.get("results", []),
            },
            gene_background=gene_background,
            drug_info=drug_info,
            language=("chinese" if has_statin else "english"),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WARN] Report generation failed, degrading: {e}")
        report = {
            "sections": [],
            "confidence": "low",
            "disclaimer": "",
            "has_sufficient_evidence": False,
        }
    sources = _build_sources(structured_results)

    # Deterministic recommendation — shown verbatim as the report headline.
    # Statin → rule engine; other exact CPIC matches → CPIC recommendation.
    recommendation = None
    recommendation_src = ""
    if rule_result:
        recommendation = rule_result
        recommendation_src = "规则引擎（SLCO1B1/APOE 基因多态性检测）"
    elif structured_results:
        for r in structured_results[:3]:
            dr = (r.get("drugrecommendation") or "").strip()
            if dr:
                recommendation = dr
                recommendation_src = "CPIC Guideline"
                break

    # Raw "Retrieved Evidence" lives in the technical-analysis panel, not the
    # polished main report. The source line at the end keeps the PMIDs /
    # guideline sources visible for traceability.
    # Medical accuracy first: a recommendation is shown ONLY when a
    # deterministic source (rule engine / CPIC structured) produced one. Any
    # LLM-generated "Recommendation" section is always dropped.
    # Gene / Drug Summary sections are omitted when the query has no gene /
    # drug, so the report only shows the sections the user's input calls for.
    has_gene_in_query = bool(parsed.genes or parsed.genotypes)
    has_drug_in_query = bool(parsed.drug or parsed.drug_class)
    main_sections = []
    evidence_for_debug = ""
    for sec in report.get("sections", []):
        t = (sec.get("title", "") or "").lower()
        if "retrieved evidence" in t or "evidence" in t or "证据" in t:
            evidence_for_debug = _italicize_genes(sec.get("content", "") or "")
        elif "reference" in t or "参考" in t:
            continue
        elif "recommendation" in t or "推荐" in t:
            continue
        elif ("gene summary" in t or "基因摘要" in t or "基因概述" in t
              or "基因简介" in t or "基因总结" in t) and not has_gene_in_query:
            continue
        elif ("drug summary" in t or "药物摘要" in t or "药物概述" in t
              or "药物简介" in t or "药物总结" in t) and not has_drug_in_query:
            continue
        else:
            main_sections.append(sec)
    if main_sections:
        report = {**report, "sections": main_sections}

    # ── Debug info ──
    routes = []
    if structured_results:
        routes.append("CPIC SQL: 精确基因型匹配")
    if rule_result:
        routes.append(f"规则引擎: 他汀评估")
    if cpic_rag.get("results"):
        routes.append(f"RAG: CPIC 摘要（{cpic_rag.get('top_score', 0):.2f}）")
    if statin_rag.get("results"):
        routes.append(f"RAG: 他汀指南（{statin_rag.get('top_score', 0):.2f}）")
    if not routes:
        routes.append("Gap Log")
    debug["steps"].append({"name": "路由决策", "data": {"路由": routes}})

    # RAG retrieval detail — makes the rerank / evidence-filter layer visible.
    for tag, rag in (("他汀指南（中国共识）", statin_rag), ("PubMed/CPIC 背景", cpic_rag)):
        if rag.get("results"):
            debug["steps"].append({"name": "RAG 检索", "data": _rag_debug_data(tag, rag)})

    if evidence_for_debug:
        debug["steps"].append({"name": "检索证据", "data": {"证据": evidence_for_debug}})

    # CPIC structured sources — the "检索依据" moved into the technical panel.
    if sources:
        debug["steps"].append({"name": "检索依据", "data": {"依据": [
            {"label": f"CPIC — {s.drug}", "content": s.content[:300]} for s in sources
        ]}})

    return ReportResponse(
        query=req.text, parsed_intent=parsed,
        rule_engine_result=rule_result,
        report_text=_format_report(report, recommendation=recommendation, recommendation_src=recommendation_src, is_statin=has_statin),
        sources=sources, debug_info=debug,
    )


def _build_sources(structured_results: list[dict] | None) -> list[SourceRef]:
    if not structured_results:
        return []
    sources = []
    for i, r in enumerate(structured_results[:5]):
        sources.append(SourceRef(
            id=i + 1, content=r.get("drugrecommendation", "")[:300],
            source="CPIC", drug=r.get("lookupkey", ""), relevance=1.0,
        ))
    return sources


# Gene symbols are italicized in report markdown (standard PGx convention).
# HLA is excluded — allele symbols carry `*` (e.g. HLA-B*57:01) which would
# break markdown italics.
_GENE_PATTERN = re.compile(
    r"(?<![*A-Za-z])"
    r"(?:CYP\d[A-Z]\d\w*"
    r"|NAT2|TPMT|NUDT15|SLCO1B1|OATP1B1|APOE|VKORC1|DPYD|UGT1A1|BCHE|G6PD"
    r"|ADRB[12]|ADRA2C|GRK[45]|AGTR1|ACE|NPPA|CACNA1C|CACNB2|NEDD4L|YEATS4)"
    r"(?![\w*])",
    re.IGNORECASE,
)


def _italicize_genes(text: str) -> str:
    """Wrap gene symbols in italic markdown (*gene*); skip already-italicized."""
    return _GENE_PATTERN.sub(lambda m: f"*{m.group(0)}*", text)


def _rag_debug_data(label: str, rag: dict) -> dict:
    """Format a RAG retrieval result for the technical-analysis panel.

    The frontend's "RAG 检索" step renders the cross-encoder rerank scores,
    query expansion, and evidence-filter stats — the retrieval-quality layer.
    Each item carries a `ref` so the retrieved source is verifiable (PMID for
    PubMed, document title for the Chinese guidelines).
    """
    results = rag.get("results", []) or []
    scores = []
    for r in results[:5]:
        meta = r.get("metadata", {}) or {}
        score = r.get("rerank_score")
        if score is None and r.get("distance") is not None:
            score = 1.0 - float(r["distance"])
        if score is None:
            score = r.get("bm25_score") or 0.0
        ref = ""
        if meta.get("pmid"):
            ref = f"PMID {meta.get('pmid')}"
        elif meta.get("filename"):
            ref = CHINESE_DOC_TITLES.get(str(meta.get("filename")), str(meta.get("filename")))
        elif meta.get("title"):
            ref = str(meta.get("title"))
        scores.append({
            "score": max(0.0, min(1.0, float(score))),
            "source": meta.get("source", ""),
            "drug": meta.get("drug", ""),
            "gene": meta.get("gene", ""),
            "ref": ref,
            "snippet": (r.get("content", "") or "")[:120],
        })
    return {
        "策略": label,
        "查询改写": rag.get("expanded_queries", []),
        "总检索数": rag.get("total_retrieved", 0),
        "rerank 后取 top": len(results),
        "证据校验删除": rag.get("irrelevant_removed", 0),
        "rerank 得分": scores,
    }


def _format_report(
    report: dict,
    recommendation: str | None = None,
    recommendation_src: str = "",
    is_statin: bool = False,
) -> str:
    """Convert report dict to formatted text.

    A deterministic "Recommendation" (rule engine / CPIC) — or an explicit
    "cannot provide a recommendation" note when none exists — headlines the
    report, then the LLM-written sections (Gene Summary / Drug Summary /
    Clinical Relevance) provide the analysis, each with its own source
    annotation.

    Heading hierarchy (rendered by the frontend under "检索分析报告"):
      ###  Recommendation (deterministic, or an explicit "cannot provide" note)
      ###  Gene Summary / Drug Summary / Clinical Relevance
    """
    if not report.get("sections"):
        return "（无法生成报告）"

    lines = []
    if report.get("evidence_warning"):
        warn_label = "⚠️ Evidence Check Warning" if not is_statin else "⚠️ 证据校验警告"
        lines.append(f"> {warn_label}: {report['evidence_warning']}\n")

    # Recommendation headline — deterministic only. When no deterministic
    # recommendation exists, state it explicitly (medical accuracy first:
    # never fabricate advice, but don't stay silent either).
    rec_label = "Recommendation" if not is_statin else "推荐建议"
    lines.append(f"### {rec_label}")
    src_word = "Sources" if not is_statin else "来源"
    if recommendation:
        lines.append(recommendation)
        if recommendation_src:
            lines.append(f"*{src_word}: {recommendation_src}*")
    else:
        no_rec = ("Based on the available information, an accurate medication "
                  "recommendation cannot be provided."
                  if not is_statin else
                  "基于已有的信息，无法获取准确的推荐建议。")
        lines.append(no_rec)
    lines.append("")

    def _norm_src(s: str) -> str:
        s = (s or "").strip()
        if s.lower() in ("cpic", "cpic guideline"):
            return "CPIC Guideline"
        if s.lower() in ("pubmed", "pubmed abstract"):
            return "PubMed"
        return s

    for sec in report["sections"]:
        lines.append(f"### {sec['title']}")
        lines.append(_italicize_genes(sec["content"]))
        cites = [_norm_src(c) for c in (sec.get("citations") or []) if _norm_src(c)]
        if cites:
            lines.append(f"*{src_word}: {', '.join(cites)}*")
        lines.append("")

    # One disclaimer, in the user's language per query type.
    disclaimer = (
        "*本报告为科研探索工具生成的结果，仅供研究参考，不构成临床建议。*"
        if is_statin else
        "*This report is generated by a research tool for research reference only and does not constitute medical advice.*"
    )
    lines.append("")
    lines.append("---")
    lines.append(disclaimer)
    cpic_note = f"基于 CPIC {CPIC_VERSION} 指南" if is_statin else f"Based on CPIC {CPIC_VERSION} guidelines"
    lines.append(f"*{cpic_note}*")
    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
