"""Quality metrics for the cleanup task."""
import re


def exact_match(pred: str, gold: str) -> bool:
    return pred == gold


_WS_RUN_RE = re.compile(r"[ \t]+")


def _normalize(s: str) -> str:
    return "\n".join(
        _WS_RUN_RE.sub(" ", line).rstrip()
        for line in s.splitlines()
    )


def normalized_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def token_reduction_pct(n_input: int, n_output: int) -> float:
    if n_input == 0:
        return 0.0
    return 1.0 - (n_output / n_input)
