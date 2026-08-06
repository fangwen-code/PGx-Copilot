# PGx-Copilot · 药物基因组学科研探索工具

在降脂药物规则引擎基础上升级为覆盖 **CPIC 多基因多药物**的智能检索系统。基于 CPIC 指南构建检索管线，输入基因型 / 药物 / 症状，自动生成带来源溯源的基因-药物关联分析报告。

> ⚠️ 科研探索工具，结果仅供参考，不构成医疗建议。

## 核心设计：三层确定性架构

```
输入(基因型/药物/症状)
  │
  ├─ 查询理解(LLM 解析基因/基因型/药物/意图)
  │
  ├─ Layer 1  CPIC SQL 精确匹配   ── 确定性结论(推荐建议)
  ├─ Layer 2  他汀规则引擎         ── 确定性结论(SLCO1B1/APOE 评估)
  ├─ Layer 3  RAG 背景增强         ── PubMed 英文摘要 + 中文指南(仅背景)
  │
  └─ 报告生成:确定性推荐 + LLM 背景/解读 + 每节来源标注
```

**关键决策**：推荐建议（用药建议）只由确定性层（规则引擎 / CPIC 结构化表）逐字输出，LLM 在 prompt 中被禁止生成医疗结论；证据不足时明确拒答。RAG 检索仅提供背景证据，不参与医疗决策——从架构上杜绝幻觉。

## 亮点

- **架构升级**：从纯规则引擎升级为“CPIC 精确匹配 + 他汀类规则引擎 + RAG 背景”三层架构，兼顾准确性与覆盖度，从架构层面规避医疗结论幻觉风险。  
- **幻觉控制**：实现 Writer-Reviewer 报告审核机制，减少LLM幻觉。  
- **质量监控**：设计Coverage Gap 主动监控机制，驱动知识库持续迭代。   

## 项目结构

```
├── backend/
│   ├── app.py                 # FastAPI 主入口(路由/报告组装/技术面板)
│   ├── config.py              # 配置(env)
│   ├── eval.py                # 可复现检索评测(Hit@K / MRR / NDCG)
│   ├── cpic/                  # CPIC SQLite 解析与查询(多键回退匹配)
│   ├── rule_engine/           # 他汀 SLCO1B1/APOE 规则引擎
│   ├── rag/                   # 检索管线(expansion/hybrid/rerank/证据过滤/入库)
│   ├── query_understanding/   # 查询解析(基因/药物类别/意图)
│   ├── report_generator/      # LLM 报告生成 + Writer-Reviewer 证据核验
│   ├── tools/                 # PubMed 抓取、Coverage Gap 日志
│   └── tests/                 # 单元测试(多键回退/drug_class/截断)
├── frontend/app.py            # Streamlit 前端
├── data/                      # 数据(cpic.db、chroma_db、PDF 源)——不入库
├── Dockerfile.backend / Dockerfile.frontend / docker-compose.yml
└── .env.example               # 环境变量模板
```

## 快速开始(本地)

### 1. 环境

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
cd backend
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp ../.env.example ../.env && nano ../.env          # 填 DEEPSEEK_API_KEY
```

### 2. 构建数据

```bash
cd backend
# a) 解析 CPIC dump → SQLite(需要 data/sources/cpic_db_dump-v1.53.2.sql)
python -m cpic.parser --dump ../data/sources/cpic_db_dump-v1.53.2.sql

# b) 嵌入 PDF(中文指南)→ ChromaDB
python -m rag.ingest

# c) 可选:抓取 PubMed 摘要 → ChromaDB
python -m tools.fetch_pubmed
```

### 3. 启动

```bash
# 后端(FastAPI, :8000)
cd backend && python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 前端(Streamlit, :8501)
cd frontend && streamlit run app.py --server.port 8501
```

访问 `http://127.0.0.1:8501`。

## Docker 部署

前置：宿主机已有构建好的 `data/`(cpic.db + chroma_db)和 `.env`。

```bash
# 1. 确保数据已构建(见上文"构建数据")
ls data/cpic.db data/chroma_db

# 2. 构建并启动
docker compose up --build

# 3. 访问
#   前端 http://127.0.0.1:8501
#   后端 http://127.0.0.1:8000
```

- 后端镜像在构建时下载 embedding 模型(`Dockerfile.backend`)，请保持 `.env` 的 `EMBEDDING_MODEL` 与其一致。
- `data/` 通过 volume 挂载进容器，镜像内不含数据。
- 如需单独构建：`docker build -f Dockerfile.backend -t pgx-backend .` / `docker build -f Dockerfile.frontend -t pgx-frontend .`

## 检索评测

```bash
cd backend
python eval.py                          # 生产配置(hybrid)逐来源指标
python eval.py --compare-strategies     # baseline/expansion/hybrid/HyDE 对比
python eval.py --compare-strategies --hard   # 不依赖元数据过滤的难查询
python -m pytest tests/ -v              # 单元测试
```

## 环境变量(见 `.env.example`)

| 变量 | 说明 | 默认 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API key(必填) | - |
| `DEEPSEEK_MODEL` | LLM 模型 | `deepseek-chat` |
| `EMBEDDING_MODEL` | 嵌入模型(bge-m3 多语言 / bge-base-en-v1.5 英文) | `BAAI/bge-m3` |
| `QUERY_EXPANSION_ENABLED` | 查询扩展 | `true` |
| `HYBRID_SEARCH_ENABLED` | Dense+BM25 混合检索 | `true` |
| `HYDE_ENABLED` | HyDE(默认关闭,延迟收益不划算) | `false` |
| `EVIDENCE_CHECK_ENABLED` | Writer-Reviewer 证据核验 | `true` |
| `CHROMA_COLLECTION` | ChromaDB 集合名 | `pgx_knowledge` |


## 常见问题

- **向量库为空 / 检索无结果**：确认跑过 `cpic.parser`、`rag.ingest`、`tools.fetch_pubmed`。
- **embedding 模型加载失败**：确认 `.env` 的 `EMBEDDING_MODEL` 已在本地缓存(`local_files_only=True`)，或先 `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('<model>')"` 下载。
- **Docker 内检索无结果**：确认 `data/` 已构建并挂载成功(`docker compose exec backend ls /app/data`)。

## 后续规划  
1. 建更大规模的相关性评测集；
2. 生成层用更强的LLM当裁判，对 faithfulness（是否忠于证据）和helpfulness（回答是否有用）批量自动化打分；
3. 端到端找临床药师做盲评，不告诉他们是系统生成的还是人工写的，让他们打分，看报告是否准确、可用。
