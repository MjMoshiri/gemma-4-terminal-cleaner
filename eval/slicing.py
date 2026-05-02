"""Group eval results by various keys for failure-mode analysis."""
from typing import Callable


def length_bucket(n_tokens: int) -> str:
    if n_tokens < 512:
        return "<512"
    if n_tokens < 2048:
        return "512-2k"
    return "2k-8k"


def slice_results(records: list[dict], key_fn: Callable[[dict], str]) -> dict[str, dict]:
    """Group records by key_fn(record) and return per-slice aggregates."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = key_fn(r)
        groups.setdefault(key, []).append(r)
    out = {}
    for key, recs in groups.items():
        n = len(recs)
        n_passed = sum(1 for r in recs if r.get("passed", False))
        metrics = [r.get("metric", 0.0) for r in recs]
        out[key] = {
            "count": n,
            "pass_rate": n_passed / n if n else 0.0,
            "metric_p50": sorted(metrics)[n // 2] if n else 0.0,
            "metric_mean": sum(metrics) / n if n else 0.0,
        }
    return out
