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


def apoegt(rs7412: str, rs429358: str) -> str:
    """Resolve APOE diplotype from rs7412 and rs429358 genotypes."""
    return _APOE_GT.get((rs7412, rs429358), "未知")


def suginfo(slco1b1: str, rs7412: str, rs429358: str) -> str:
    """Return statin medication suggestion based on SLCO1B1 + APOE."""
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
    if not slco1b1 or slco1b1 == "未知":
        return "未检测到 SLCO1B1 基因型信息，无法进行他汀风险评估。"
    if "未知" in [rs7412, rs429358] or (not rs7412 and not rs429358):
        return "需要 APOE (rs7412, rs429358) 基因型信息才能完成完整的他汀类药物风险评估。当前仅 SLCO1B1 结果可供 CPIC 参考。"
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
