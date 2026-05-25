"""
Query Understanding: parse user input into structured intent.

Uses DeepSeek LLM to extract gene, genotype, drug, symptom, disease, etc.
from free-text user input.
"""

import json
import os
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

_SYSTEM_PROMPT = """You are a query parser for a pharmacogenomics (PGx) decision support system.
Extract structured information from the user's question.

Return ONLY a JSON object with these optional fields:
- "genes": list of gene symbols mentioned (e.g. ["CYP2D6", "SLCO1B1"])
- "genotypes": dict mapping gene to genotype (e.g. {"CYP2D6": "*4/*4", "SLCO1B1": "TC"})
- "drug": the drug name (e.g. "metoprolol", "simvastatin")
- "drug_class": drug class if mentioned (e.g. "CCB", "beta blocker", "statin")
- "disease": disease or condition (e.g. "hypertension")
- "symptom": symptom description (e.g. "cough", "muscle pain")
- "condition": clinical condition (e.g. "CKD eGFR<30")
- "intent": one of ["ADR", "efficacy", "safety", "drug_choice", "alternative", "general"]

If the input is unclear or contains nothing PGx-related, return {"intent": "unclear"}.

Examples:
{"input": "吃美托洛尔咳嗽"} -> {"drug": "metoprolol", "symptom": "cough", "intent": "ADR"}
{"input": "CYP2D6 *4/*4"} -> {"genes": ["CYP2D6"], "genotypes": {"CYP2D6": "*4/*4"}, "intent": "general"}
{"input": "高血压CCB效果不好"} -> {"drug_class": "CCB", "disease": "hypertension", "intent": "alternative"}
{"input": "SLCO1B1 TC APOE E3/E4 他汀安全吗"} -> {"genes": ["SLCO1B1", "APOE"], "genotypes": {"SLCO1B1": "TC", "APOE": "E3/E4"}, "drug": "simvastatin", "intent": "safety"}
{"input": "我头晕"} -> {"symptom": "dizziness", "intent": "general"}
"""


def parse_query(user_input: str) -> dict:
    """Parse user input into structured intent."""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_api_key_here":
        return _fallback_parse(user_input)

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content
        return json.loads(text)
    except Exception as e:
        print(f"[WARN] LLM parsing failed: {e}, using fallback")
        return _fallback_parse(user_input)


def _fallback_parse(user_input: str) -> dict:
    """Simple fallback when LLM is unavailable."""
    result = {"intent": "general"}
    text = user_input.lower()

    # Gene patterns
    gene_patterns = {
        "CYP2D6": r"cyp\s*2\s*d\s*6|2d6",
        "CYP2C9": r"cyp\s*2\s*c\s*9|2c9",
        "CYP2C19": r"cyp\s*2\s*c\s*19|2c19",
        "SLCO1B1": r"slco1b1",
        "APOE": r"apoe",
        "NAT2": r"nat2",
        "CYP3A4": r"cyp\s*3\s*a\s*4|3a4",
    }
    genes = []
    genotypes = {}
    for gene, pattern in gene_patterns.items():
        import re
        m = re.search(pattern, text)
        if m:
            genes.append(gene)
            # Try to extract genotype after gene mention
            # Simple: look for *number pattern
            gt_match = re.search(rf"{pattern}[\s,;]*(\*?\d+[\*\/]\*?\d+)", text)
            if gt_match:
                genotypes[gene] = gt_match.group(1).strip()

    # Drug patterns
    drug_map = {
        "metoprolol": ["美托洛尔", "metoprolol"],
        "losartan": ["氯沙坦", "losartan"],
        "simvastatin": ["辛伐他汀", "simvastatin", "他汀"],
        "amlodipine": ["氨氯地平", "amlodipine"],
        "lisinopril": ["赖诺普利", "lisinopril"],
    }
    drug = ""
    for name, aliases in drug_map.items():
        if any(a in text for a in aliases):
            drug = name
            break

    # Disease patterns
    if any(w in text for w in ["高血压", "hypertension", "血压高"]):
        result["disease"] = "hypertension"
    if any(w in text for w in ["咳嗽", "cough"]):
        result["symptom"] = "cough"
    if any(w in text for w in ["eGFR", "ckd", "kidney", "肾"]):
        result["condition"] = "CKD"

    # Intent based on keywords
    if any(w in text for w in ["安全", "风险", "不良反应", "咳嗽", "副作用"]):
        result["intent"] = "safety"
    elif any(w in text for w in ["效果", "疗效", "不好", "不佳"]):
        result["intent"] = "efficacy"
    elif any(w in text for w in ["替代", "换", "其他选择"]):
        result["intent"] = "alternative"
    elif any(w in text for w in ["选", "推荐", "建议"]):
        result["intent"] = "drug_choice"

    if genes:
        result["genes"] = genes
    if genotypes:
        result["genotypes"] = genotypes
    if drug:
        result["drug"] = drug

    return result
