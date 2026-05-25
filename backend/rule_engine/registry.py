"""
Rule engine registry.

Each rule engine registers itself via the @rule_engine decorator, declaring
which genes it handles. app.py calls evaluate_all() to run all matching engines.

Usage:
    from rule_engine.registry import rule_engine, evaluate_all

    @rule_engine("statin", ["SLCO1B1", "APOE", "rs4149056"])
    def my_engine(genotypes: dict[str, str]) -> str | None:
        ...  # return advice text or None
"""

_rule_engines: list[dict] = []


def rule_engine(name: str, genes: list[str]):
    """Decorator to register a rule engine.

    Args:
        name: engine name (e.g. "statin")
        genes: list of gene symbols this engine handles

    Usage:
        @rule_engine("statin", ["SLCO1B1", "APOE", "rs4149056"])
        def my_engine(genotypes: dict[str, str]) -> str | None:
            ...
    """
    def wrapper(func):
        _rule_engines.append({"name": name, "genes": genes, "func": func})
        return func
    return wrapper


def evaluate_all(genotypes: dict[str, str]) -> dict[str, str | None]:
    """Run all registered engines whose genes match the input.

    Returns:
        {engine_name: advice_text_or_None}
    """
    results = {}
    if not genotypes:
        return results
    for engine in _rule_engines:
        if any(g in genotypes for g in engine["genes"]):
            try:
                result = engine["func"](genotypes)
                if result:
                    results[engine["name"]] = result
            except Exception as e:
                print(f"[WARN] Rule engine '{engine['name']}' failed: {e}")
    return results
