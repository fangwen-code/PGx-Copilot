"""
End-to-end clinical evaluation: run real user queries through the FULL
pipeline (parse → CPIC matching → rule engine → RAG → report formatting) and
score the deterministic output against CPIC ground truth.

Metrics
  recommendation_accuracy  POSITIVE cases where the report reproduces the exact
                           CPIC drugrecommendation (normalized substring match).
                           This is the drug-scoped recommendation — if the
                           report surfaces a different drug's advice, it FAILS.
  refusal_accuracy         NEGATIVE cases where the system refuses
                           (无充分证据 / "cannot be provided") instead of
                           fabricating a recommendation.
  hallucination_rate       1 - refusal_accuracy on negative cases.
  miss / mismatch          Positive cases that wrongly refused, or surfaced a
                           different recommendation than ground truth.
  latency                  mean / p50 / p95 wall time per query.

Usage (run from backend/):
    python eval_clinical.py                    # run all cases
    python eval_clinical.py --limit 5          # quick smoke run
    python eval_clinical.py --print 3          # dump full report for case #3
    python eval_clinical.py --sleep 0.5        # seconds between cases
"""

import argparse
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from models.schemas import QueryRequest
from app import _handle_query

EVALS_DIR = Path(__file__).resolve().parent / "evals"

REFUSAL_MARKERS = [
    "无充分证据",
    "cannot be provided",
    "无法获取准确的推荐建议",
    "无法生成报告",
    "无法提供",
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _contains(expected: str, report_text: str) -> bool:
    return bool(expected) and _norm(expected) in _norm(report_text)


def _is_refusal(report_text: str) -> bool:
    t = _norm(report_text)
    return any(m.lower() in t for m in REFUSAL_MARKERS)


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def evaluate_case(case: dict) -> dict:
    """Run one case through the real pipeline; returns status + latency."""
    t0 = time.time()
    try:
        resp = _handle_query(QueryRequest(text=case["query"]))
        report_text = resp.report_text
    except Exception as e:
        return {
            "id": case.get("id"), "type": case["type"], "query": case["query"],
            "status": "ERROR", "latency": time.time() - t0, "report": f"ERROR: {e}",
        }
    elapsed = time.time() - t0

    if case["type"] == "positive":
        if _contains(case["expected"], report_text):
            status = "PASS"
        elif _is_refusal(report_text):
            status = "REFUSED"  # should have matched — wrongly refused
        else:
            status = "MISMATCH"  # surfaced a different recommendation
    else:
        status = "PASS" if _is_refusal(report_text) else "HALLUCINATED"

    return {
        "id": case.get("id"), "type": case["type"], "query": case["query"],
        "gene": case.get("gene", ""), "drug": case.get("drug", ""),
        "expected": case.get("expected", ""),
        "status": status, "latency": elapsed, "report": report_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end clinical evaluation")
    parser.add_argument("--cases", type=Path, default=EVALS_DIR / "clinical_cases.json")
    parser.add_argument("--limit", type=int, default=0, help="run only first N cases (smoke test)")
    parser.add_argument("--print", type=int, metavar="ID", help="print full report for one case, then exit")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds between cases")
    args = parser.parse_args()

    if not args.cases.exists():
        raise SystemExit(f"[ERROR] case file not found: {args.cases} — run eval_build_cases.py first.")

    data = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = data["cases"]
    if args.limit:
        cases = cases[: args.limit]

    if args.print:
        target = next((c for c in cases if c.get("id") == args.print), None)
        if not target:
            raise SystemExit(f"[ERROR] no case with id={args.print}")
        r = evaluate_case(target)
        print(f"===== case {r['id']} [{r['type']}] → {r['status']} ({r['latency']:.2f}s) =====")
        print(f"query:    {r['query']}")
        if r.get("expected"):
            print(f"expected: {r['expected']}")
        print(f"--- report ---\n{r['report']}")
        return

    print(f"\n{'='*100}")
    print(f"CLINICAL EVAL — {len(cases)} cases (from {args.cases.name})")
    print(f"{'='*100}")
    print(f"{'id':>3} {'type':<9} {'status':<12} {'lat(s)':>7}  gene/drug / query")
    print("-" * 100)

    results = []
    for case in cases:
        r = evaluate_case(case)
        results.append(r)
        label = f"{r.get('gene','')}/{r.get('drug','')}" if r.get("gene") else "-"
        print(f"{r['id']:>3} {r['type']:<9} {r['status']:<12} {r['latency']:>7.2f}  "
              f"{label}  {r['query'][:60]}")
        if args.sleep:
            time.sleep(args.sleep)

    pos = [r for r in results if r["type"] == "positive"]
    neg = [r for r in results if r["type"] == "negative"]
    lats = sorted(r["latency"] for r in results)

    n_pos_ok = sum(1 for r in pos if r["status"] == "PASS")
    n_refused_ok = sum(1 for r in neg if r["status"] == "PASS")
    n_errors = sum(1 for r in results if r["status"] == "ERROR")
    n_refused_miss = sum(1 for r in pos if r["status"] == "REFUSED")
    n_mismatch = sum(1 for r in pos if r["status"] == "MISMATCH")
    n_halluc = sum(1 for r in neg if r["status"] == "HALLUCINATED")

    print("-" * 100)
    print(f"\n=== SUMMARY ===")
    print(f"  Positives:           {len(pos)}   recommendation accuracy = "
          f"{n_pos_ok}/{len(pos)} = {n_pos_ok/len(pos):.0%}"
          f"   (REFUSED wrongly: {n_refused_miss}, MISMATCH: {n_mismatch})")
    if neg:
        print(f"  Negatives:           {len(neg)}   refusal accuracy = "
              f"{n_refused_ok}/{len(neg)} = {n_refused_ok/len(neg):.0%}")
        print(f"  HALLUCINATION RATE:  {n_halluc}/{len(neg)} = {n_halluc/len(neg):.0%}")
    if n_errors:
        print(f"  Errors:              {n_errors}")
    print(f"  Latency (s):         mean={statistics.mean(lats):.2f}  "
          f"p50={_pct(lats, 0.5):.2f}  p95={_pct(lats, 0.95):.2f}")

    out = EVALS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  detail written to: {out}")


if __name__ == "__main__":
    main()
