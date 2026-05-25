"""PGx-Copilot: Streamlit frontend."""

import streamlit as st
import requests
import json

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="PGx-Copilot", page_icon="🧬", layout="wide")
st.title("PGx-Copilot — 药物基因组学科研探索工具")

st.markdown(
    "输入基因型、药物、症状或临床描述，系统将基于 CPIC 指南数据 + RAG 检索生成参考分析报告。\n\n"
    "> ⚠️ **本工具为科研探索用途，结果仅供参考，不构成医疗建议。所有用药方案请咨询执业医师。**"
)

# --- Sidebar with example queries ---
with st.sidebar:
    st.header("示例查询")
    examples = [
        "🧬 CYP2D6 *4/*4 吃美托洛尔",
        "🧬 SLCO1B1 TC APOE E3/E4 他汀安全吗",
        "🧬 CYP2C9 *1/*3 氯沙坦",
        "🧬 氯吡格雷 CYP2C19 中间代谢",
        "🧬 NAT2 肼屈嗪风险",
        "💊 warfarin 剂量",
        "⚠️ 我头晕",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, type="secondary"):
            # Strip emoji prefix for actual query text
            query = ex[2:] if ex.startswith("🧬") or ex.startswith("💊") or ex.startswith("⚠️") else ex
            st.session_state["query_text"] = query

# --- Input ---
default_query = st.session_state.get("query_text", "")

input_mode = st.radio("输入方式", ["自由文本", "结构化表单"], horizontal=True)
query_text = ""

if input_mode == "自由文本":
    query_text = st.text_area(
        "描述您的情况（基因、药物、症状、疾病均可）",
        value=default_query,
        placeholder='例如：\n'
        '  "CYP2D6 *4/*4，吃美托洛尔效果不好"\n'
        '  "SLCO1B1 TC，APOE E3/E4，用他汀安全吗"\n'
        '  "CYP2C9 *1/*3 氯沙坦"\n'
        '  "我头晕"',
        height=100,
    )
else:
    with st.form("structured_form"):
        col1, col2 = st.columns(2)
        with col1:
            gene = st.text_input("基因", placeholder="如 CYP2D6、SLCO1B1、CYP2C19")
            genotype = st.text_input("基因型", placeholder="如 *4/*4, TT, TC, CC")
        with col2:
            drug = st.text_input("药物", placeholder="如 美托洛尔、氯沙坦、辛伐他汀")
            symptom = st.text_input("症状/疾病", placeholder="如 高血压、咳嗽")
        submitted = st.form_submit_button("提交", use_container_width=True)
        if submitted:
            parts = []
            if gene and genotype:
                parts.append(f"{gene} {genotype}")
            elif gene:
                parts.append(gene)
            if drug:
                parts.append(drug)
            if symptom:
                parts.append(symptom)
            query_text = "，".join(parts)

# --- Submit ---
col1, col2 = st.columns([6, 1])
with col1:
    submit = st.button("生成报告", type="primary", use_container_width=True,
                       disabled=not query_text.strip())

if submit or (query_text.strip() and st.session_state.get("auto_submit")):
    st.session_state["auto_submit"] = False
    with st.spinner("正在分析并检索临床证据..."):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json={"text": query_text.strip(), "top_k": 5},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.divider()
                st.subheader("检索分析报告")

                # Intent display
                intent_map = {
                    "ADR": "不良反应", "efficacy": "疗效", "safety": "安全性",
                    "drug_choice": "用药选择", "alternative": "替代方案",
                    "dosing": "剂量", "unclear": "",
                }
                if data.get("parsed_intent"):
                    intent = data["parsed_intent"]
                    tags = []
                    if intent.get("genes"):
                        tags.append(f"🧬 {', '.join(intent['genes'])}")
                    if intent.get("drug"):
                        tags.append(f"💊 {intent['drug']}")
                    if intent.get("symptom"):
                        tags.append(intent['symptom'])
                    if intent.get("disease"):
                        tags.append(intent['disease'])
                    if intent.get("intent"):
                        label = intent_map.get(intent["intent"], "")
                        if label:
                            tags.append(label)
                    if tags:
                        st.info(" · ".join(tags))

                # Rule engine result
                if data.get("rule_engine_result"):
                    with st.expander("规则引擎结果（他汀类药物）", expanded=False):
                        st.success(data["rule_engine_result"])

                # Report text
                report = data.get("report_text", "")
                if report:
                    st.markdown(report)

                # Sources
                if data.get("sources"):
                    with st.expander(f"检索依据 ({len(data['sources'])} 条)", expanded=False):
                        for i, src in enumerate(data["sources"], 1):
                            with st.container():
                                parts = []
                                if src.get('drug'):
                                    parts.append(f"💊 {src['drug']}")
                                if src.get('gene'):
                                    parts.append(f"🧬 {src['gene']}")
                                meta = " · ".join(parts) if parts else ""
                                label = src['source']
                                if meta:
                                    label += f" — {meta}"
                                st.markdown(f"**{i}. {label}**")
                                st.caption(src["content"][:400])
                                if i < len(data["sources"]):
                                    st.markdown("---")
            else:
                st.error(f"API 错误: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("无法连接到后端服务。请确认 FastAPI 已启动:\n"
                     f"  cd backend && uvicorn app:app --reload")
        except Exception as e:
            st.error(f"请求失败: {e}")
elif query_text.strip():
    st.info("点击「生成报告」提交查询")
else:
    st.info("输入查询内容后点击「生成报告」")
