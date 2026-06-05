import json
import sys
from pathlib import Path
from typing import List

from eval.metrics import aggregate
from token_optimizer.engine import optimize


def run_harness(repo: str, golden_path: str = "eval/golden.jsonl") -> List[dict]:
    gf = Path(golden_path)
    if not gf.exists():
        print(f"ERROR: golden file not found: {golden_path}", file=sys.stderr)
        sys.exit(1)

    cases = []
    with open(gf) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    print(f"Running {len(cases)} eval cases against repo '{repo}' …")

    results = []
    for case in cases:
        task_type = case["task_type"]
        user_input = case["input"]
        expect_contains: List[str] = case.get("expect_contains", [])

        try:
            result = optimize(task_type, repo, user_input)
            out_lower = result.output.lower()
            hits = [kw for kw in expect_contains if kw.lower() in out_lower]
            quality_pass = len(hits) == len(expect_contains) if expect_contains else True

            results.append({
                "id": case.get("id", "?"),
                "task_type": task_type,
                "savings_pct": result.token_report.savings_pct,
                "baseline_tokens": result.token_report.baseline_tokens,
                "context_tokens": result.token_report.context_tokens,
                "input_tokens": result.token_report.input_tokens,
                "output_tokens": result.token_report.output_tokens,
                "quality_pass": quality_pass,
                "hits": f"{len(hits)}/{len(expect_contains)}",
                "model": result.token_report.model_used,
                "error": None,
            })

        except Exception as exc:
            results.append({
                "id": case.get("id", "?"),
                "task_type": task_type,
                "savings_pct": 0.0,
                "baseline_tokens": 0,
                "context_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "quality_pass": False,
                "hits": f"0/{len(expect_contains)}",
                "model": "N/A",
                "error": str(exc),
            })

    _print_table(results)
    return results


def _print_table(results: List[dict]) -> None:
    stats = aggregate(results)

    W = 90
    print("\n" + "=" * W)
    print(
        f"{'ID':<4} {'TYPE':<10} {'SAVINGS%':>9} {'BASELINE':>10} "
        f"{'CTX':>8} {'BILLED_IN':>10} {'STATUS':<18} MODEL"
    )
    print("-" * W)
    for r in results:
        if r["error"]:
            status = f"ERROR: {r['error'][:20]}"
        else:
            status = ("PASS" if r["quality_pass"] else "FAIL") + f" ({r['hits']})"
        print(
            f"{str(r['id']):<4} {r['task_type']:<10} {r['savings_pct']:>8.1f}% "
            f"{r['baseline_tokens']:>10,} {r['context_tokens']:>8,} "
            f"{r['input_tokens']:>10,} {status:<18} {r['model']}"
        )
    print("=" * W)
    print(
        f"Avg savings: {stats['avg_savings_pct']:.1f}%  |  "
        f"Pass: {stats['passing']}/{stats['total']}  |  "
        f"CI: {'PASS ✓' if stats['ci_pass'] else 'FAIL ✗'}"
    )
    print()
