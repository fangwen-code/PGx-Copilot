"""
Statin PGx rule engine — ported from FlaskVersion3/suggestion.py

Covers SLCO1B1 (rs4149056) and APOE (rs7412, rs429358) for statin
safety and efficacy recommendations.
"""


# APOE diplotypes lookup: (rs7412, rs429358) -> diplotype
_APOE_GT = {
    ("TT", "TT"): "E2/E2",
    ("CT", "TT"): "E2/E3",
    ("CC", "TT"): "E3/E3",
    ("CT", "TC"): "E2/E4",
    ("CC", "TC"): "E3/E4",
    ("CC", "CC"): "E4/E4",
}

# SLCO1B1 reduced-function star alleles: all carry the rs4149056 C allele.
# *1 carries the normal T allele.
_SLCO1B1_REDUCED = {"*5", "*15", "*17"}


def slco1b1_star_to_rs(diplotype: str) -> str | None:
    """Convert an SLCO1B1 star-allele diplotype to rs4149056 (TT/TC/CC).

    Two reduced-function alleles (*5/*15/*17, the rs4149056 C carriers) → CC,
    one reduced + *1 → TC, *1/*1 → TT. Returns None when not a star diplotype.
    """
    parts = [p.strip() for p in str(diplotype).split("/")]
    if len(parts) != 2 or "*" not in diplotype:
        return None
    a, b = parts
    a_red, b_red = a in _SLCO1B1_REDUCED, b in _SLCO1B1_REDUCED
    if a_red and b_red:
        return "CC"
    if a_red or b_red:
        return "TC"
    return "TT"


def apoegt(rs7412: str, rs429358: str) -> str:
    """Resolve APOE diplotype from rs7412 and rs429358 genotypes."""
    return _APOE_GT.get((rs7412, rs429358), "未知")


def suginfo(slco1b1: str = "", rs7412: str = "", rs429358: str = "") -> str:
    """Return statin medication suggestion based on SLCO1B1 + APOE.

    Handles partial genotypes:
      - Both SLCO1B1 and APOE → full combined assessment
      - Only SLCO1B1 → risk assessment with APOE-missing notice
      - Only APOE → efficacy reference with SLCO1B1-missing notice
    """
    has_slco1b1 = bool(slco1b1) and slco1b1 not in ("", "未知")
    has_apoe = bool(rs7412 and rs429358) and "未知" not in (rs7412, rs429358)

    if not has_slco1b1 and not has_apoe:
        return "未检测到 SLCO1B1 和 APOE 基因型信息，无法进行他汀风险评估。"

    # Case: only APOE provided
    if not has_slco1b1 and has_apoe:
        apoe = apoegt(rs7412, rs429358)
        if apoe in ("E3/E4", "E4/E4"):
            return (
                f"APOE 表型：{apoe}，他汀类药物降脂疗效可能欠佳。"
                "⛔ 缺少 SLCO1B1 基因型，无法评估肌病风险。建议补充检测。"
            )
        else:
            return (
                f"APOE 表型：{apoe}，他汀类药物疗效预期良好。"
                "⛔ 缺少 SLCO1B1 基因型，无法评估肌病风险。建议补充检测。"
            )

    # Case: only SLCO1B1 provided
    if has_slco1b1 and not has_apoe:
        if slco1b1 == "TT":
            return (
                "SLCO1B1 TT 型：肌病风险较低（正常转运功能）。"
                "⛔ 缺少 APOE 基因型，无法评估他汀降脂疗效。建议补充检测。"
            )
        elif slco1b1 in ("TC", "CC"):
            risk = "中度升高" if slco1b1 == "TC" else "显著升高"
            return (
                f"SLCO1B1 {slco1b1} 型：肌病风险{risk}，建议从低剂量起始，"
                "定期监测肌酸激酶。"
                "⛔ 缺少 APOE 基因型，无法评估他汀降脂疗效。建议补充检测。"
            )

    # Case: both provided
    apoe = apoegt(rs7412, rs429358)
    good_response = apoe in ("E2/E2", "E2/E3", "E3/E3", "E2/E4")
    poor_response = apoe in ("E3/E4", "E4/E4")

    if slco1b1 == "TT" and good_response:
        return "发生不良反应风险较低，药效较好，使用各类他汀药物均可。"
    if slco1b1 == "TT" and poor_response:
        return "发生不良反应风险较低，但使用他汀类药物疗效欠佳。"
    if slco1b1 in ("TC", "CC") and good_response:
        return (
            "使用他汀类药物疗效均可，但需注意调整服药剂量，"
            "以降低发生不良反应的风险。若在服用他汀类药物期间"
            "如发生肌病问题，建议立即就医。"
        )
    if slco1b1 in ("TC", "CC") and poor_response:
        return (
            "使用他汀类药物疗效欠佳，且需谨慎不良反应，建议换药，"
            "如普罗布考。具体情况请以医生处方为准，若在服用他汀类药物期间"
            "如发生肌病问题，建议立即就医。"
        )

    return "非常抱歉，您提供的基因型组合不在规则引擎覆盖范围内。"


from .registry import rule_engine


def get_advice(genotypes: dict[str, str]) -> dict:
    """
    Unified interface for the rule engine.

    Args:
        genotypes: {"SLCO1B1": "TT", "APOE_rs7412": "CC", "APOE_rs429358": "TT"}
                   or {"rs4149056": "TT", "rs7412": "CC", "rs429358": "TT"}
                   or {"SLCO1B1": "TC", "APOE": "E3/E4"}  (diplotype format)

    Returns:
        {"engine": "rule", "drug_class": "statin",
         "genotypes": ..., "apoe": ..., "suggestion": ...}
    """
    slco1b1 = genotypes.get("SLCO1B1") or genotypes.get("rs4149056", "")
    # Star-allele input (e.g. "*5/*5") → rs4149056 genotype so the rule covers it.
    if slco1b1 and "*" in str(slco1b1):
        converted = slco1b1_star_to_rs(str(slco1b1))
        if converted:
            slco1b1 = converted
    rs7412 = genotypes.get("APOE_rs7412") or genotypes.get("rs7412", "")
    rs429358 = genotypes.get("APOE_rs429358") or genotypes.get("rs429358", "")

    # Accept APOE diplotype (e.g. "E3/E4") as alternative to raw SNPs
    apoe_diplotype = genotypes.get("APOE_diplotype") or genotypes.get("APOE", "")
    if apoe_diplotype and not (rs7412 and rs429358):
        _DIPLO_TO_SNP = {v: k for k, v in _APOE_GT.items()}
        snps = _DIPLO_TO_SNP.get(apoe_diplotype)
        if snps:
            rs7412, rs429358 = snps

    apoe = apoegt(rs7412, rs429358)
    suggestion = suginfo(slco1b1, rs7412, rs429358)

    return {
        "engine": "rule",
        "drug_class": "statin",
        "genotypes": {
            "SLCO1B1": slco1b1,
            "APOE_diplotype": apoe,
        },
        "suggestion": suggestion,
    }


@rule_engine("statin", ["SLCO1B1", "APOE", "rs4149056", "rs7412", "rs429358"])
def _statin_advice(genotypes: dict[str, str]) -> str | None:
    """Registered wrapper for the rule engine registry."""
    result = get_advice(genotypes)
    return result.get("suggestion")
