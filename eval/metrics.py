from typing import Dict, List


def token_savings_pct(baseline: int, optimized: int) -> float:
    if baseline == 0:
        return 0.0
    return max(0.0, (baseline - optimized) / baseline * 100)


def aggregate(results: List[Dict]) -> Dict:
    valid = [r for r in results if not r.get("error")]
    passing = [r for r in valid if r.get("quality_pass")]
    return {
        "total": len(results),
        "valid": len(valid),
        "passing": len(passing),
        "avg_savings_pct": (
            sum(r["savings_pct"] for r in valid) / len(valid) if valid else 0.0
        ),
        "quality_pass_rate": len(passing) / len(valid) if valid else 0.0,
        "ci_pass": len(passing) == len(results),
    }
