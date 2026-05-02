"""Run full evaluation: lossless guard, quality metrics, slice analysis, threshold check.

Usage:
    uv run python -m eval.run --adapter models/adapter \
        --eval-sets val,eval_real,eval_passthrough

Loads the base model + (optional) LoRA adapter, runs each requested eval set,
and writes a JSON report to ``eval/reports/<timestamp>.json``. Exits with code
0 if all configured acceptance thresholds pass, 1 otherwise.

This module monkey-patches ``mlx_lm`` (via ``train.patch_mlx_lm``) before any
``mlx_lm`` import — required for the Gemma 4 E2B kv-shared-layers checkpoint.
"""
from __future__ import annotations

# Patch MUST be imported before any mlx_lm import.
import train.patch_mlx_lm  # noqa: F401

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

from mlx_lm import generate, load
from tqdm import tqdm

from eval.lossless_guard import lossless_check
from eval.metrics import exact_match, normalized_match, token_reduction_pct
from eval.slicing import length_bucket, slice_results
from infer.ansi_strip import strip
from train.prompt_template import INSTRUCTION


# --- Acceptance thresholds (spec §7.5) -----------------------------------

THRESHOLDS: dict[str, dict[str, float]] = {
    "eval_real": {
        "guard_pass_rate": 0.995,
        "normalized_match_rate": 0.80,
        "median_token_reduction": 0.30,
    },
    "eval_passthrough": {
        "guard_pass_rate": 1.0,
        "max_abs_median_token_reduction": 0.02,
    },
    # `val` has no formal threshold — we report metrics only.
}

DEFAULT_DATA_PATHS = {
    "val": "data/val.jsonl",
    "eval_real": "data/eval_real.jsonl",
    "eval_passthrough": "data/eval_passthrough.jsonl",
}


# --- Prompt construction -------------------------------------------------


def _build_user_content(dirty: str) -> str:
    """User-turn content; matches train/prompt_template.format_chat exactly."""
    return f"{INSTRUCTION}\n\n---\n{dirty}\n---"


def _apply_chat_template(tokenizer, dirty: str) -> str:
    """Apply the tokenizer's chat template with ``add_generation_prompt=True``.

    The model is fine-tuned on chat-templated examples (see
    ``train/format_dataset.py``); raw-text prompting will not match training
    distribution and produces poor outputs.
    """
    messages = [{"role": "user", "content": _build_user_content(dirty)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# --- Per-record evaluation -----------------------------------------------


def _evaluate_set(
    model,
    tokenizer,
    path: Path,
    *,
    max_tokens: int = 4096,
    limit: int | None = None,
    generate_fn: Callable[..., str] = generate,
) -> list[dict]:
    """Generate predictions for every record in ``path`` and compute per-record metrics.

    The ``generate_fn`` parameter is for tests — the default is ``mlx_lm.generate``.
    """
    with path.open(encoding="utf-8") as f:
        lines = list(f)
    if limit is not None:
        lines = lines[:limit]

    records: list[dict] = []
    for line in tqdm(lines, desc=path.name):
        rec = json.loads(line)
        dirty = rec["input"]
        gold = rec["output"]

        # Pre-step: deterministic ANSI strip (matches inference path).
        pre = strip(dirty)
        prompt = _apply_chat_template(tokenizer, pre)

        pred = generate_fn(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
        )
        pred_clean = strip(pred).strip()

        # Token counts (post-strip pred for fair reduction calc against original input).
        n_in = len(tokenizer.encode(dirty))
        n_out = len(tokenizer.encode(pred_clean)) if pred_clean else 0

        guard = lossless_check(dirty, pred_clean)
        records.append(
            {
                "input": dirty,
                "gold": gold,
                "pred": pred_clean,
                "guard_passed": guard.passed,
                "missing_atoms": list(guard.missing_atoms),
                "exact": exact_match(pred_clean, gold),
                "normalized": normalized_match(pred_clean, gold),
                "n_in": n_in,
                "n_out": n_out,
                "tok_red": token_reduction_pct(n_in, n_out),
                "meta": rec.get("meta", {}),
            }
        )
    return records


# --- Aggregation ---------------------------------------------------------


def _summarize(records: list[dict]) -> dict[str, Any]:
    """Aggregate per-record metrics into per-set summary stats."""
    n = len(records)
    if n == 0:
        return {
            "count": 0,
            "guard_pass_rate": 0.0,
            "exact_match_rate": 0.0,
            "normalized_match_rate": 0.0,
            "median_token_reduction": 0.0,
            "mean_token_reduction": 0.0,
        }
    guard_passed = sum(1 for r in records if r["guard_passed"])
    exact_passed = sum(1 for r in records if r["exact"])
    norm_passed = sum(1 for r in records if r["normalized"])
    tok_reds = [r["tok_red"] for r in records]
    return {
        "count": n,
        "guard_pass_rate": guard_passed / n,
        "exact_match_rate": exact_passed / n,
        "normalized_match_rate": norm_passed / n,
        "median_token_reduction": statistics.median(tok_reds),
        "mean_token_reduction": statistics.fmean(tok_reds),
    }


def _slices_for_set(records: list[dict]) -> dict[str, dict]:
    """Group records by recipe / archetype / length-bucket for failure-mode analysis."""
    # Adapt records into the shape eval.slicing.slice_results expects:
    # {"passed": bool, "metric": float, ...key fields...}
    shaped = [
        {
            **r,
            "passed": r["guard_passed"],
            "metric": float(r["normalized"]),
            "_recipe": r.get("meta", {}).get("recipe")
            or r.get("meta", {}).get("command")
            or "unknown",
            "_archetype": r.get("meta", {}).get("archetype", "unknown"),
            "_len_bucket": length_bucket(r["n_in"]),
        }
        for r in records
    ]
    return {
        "by_recipe": slice_results(shaped, key_fn=lambda r: r["_recipe"]),
        "by_archetype": slice_results(shaped, key_fn=lambda r: r["_archetype"]),
        "by_length": slice_results(shaped, key_fn=lambda r: r["_len_bucket"]),
    }


# --- Threshold check -----------------------------------------------------


def check_thresholds(per_set_summary: dict[str, dict]) -> tuple[bool, list[dict]]:
    """Return ``(overall_pass, per-check details)``.

    ``per_set_summary`` maps set-name → summary dict (output of ``_summarize``).
    """
    results: list[dict] = []
    overall = True
    for set_name, thresholds in THRESHOLDS.items():
        summary = per_set_summary.get(set_name)
        if summary is None or summary.get("count", 0) == 0:
            # No data for this set — skip silently rather than fail.
            continue
        for metric_name, target in thresholds.items():
            if metric_name == "max_abs_median_token_reduction":
                actual = abs(summary["median_token_reduction"])
                ok = actual <= target
                results.append(
                    {
                        "set": set_name,
                        "metric": metric_name,
                        "target": f"<= {target}",
                        "actual": actual,
                        "ok": ok,
                    }
                )
            else:
                # All other thresholds are minimums.
                summary_key = (
                    "median_token_reduction"
                    if metric_name == "median_token_reduction"
                    else metric_name
                )
                actual = summary[summary_key]
                ok = actual >= target
                results.append(
                    {
                        "set": set_name,
                        "metric": metric_name,
                        "target": f">= {target}",
                        "actual": actual,
                        "ok": ok,
                    }
                )
            if not results[-1]["ok"]:
                overall = False
    return overall, results


# --- Reporting -----------------------------------------------------------


def _print_report(
    per_set_summary: dict[str, dict],
    per_set_slices: dict[str, dict],
    threshold_results: list[dict],
    overall_pass: bool,
) -> None:
    print()
    print("=" * 72)
    print("EVAL REPORT")
    print("=" * 72)
    for name, summary in per_set_summary.items():
        print(f"\n[{name}]  n={summary['count']}")
        print(f"  guard_pass_rate        = {summary['guard_pass_rate']:.4f}")
        print(f"  exact_match_rate       = {summary['exact_match_rate']:.4f}")
        print(f"  normalized_match_rate  = {summary['normalized_match_rate']:.4f}")
        print(f"  median_token_reduction = {summary['median_token_reduction']:+.4f}")
        print(f"  mean_token_reduction   = {summary['mean_token_reduction']:+.4f}")
        slices = per_set_slices.get(name, {})
        for slice_name in ("by_recipe", "by_archetype", "by_length"):
            buckets = slices.get(slice_name, {})
            if not buckets:
                continue
            print(f"  {slice_name}:")
            for key, agg in sorted(buckets.items()):
                print(
                    f"    {key:<28} n={agg['count']:<5} "
                    f"guard_pass={agg['pass_rate']:.3f}  "
                    f"normalized_p50={agg['metric_p50']:.3f}"
                )
    print("\n" + "-" * 72)
    print("THRESHOLD CHECK (spec §7.5)")
    print("-" * 72)
    if not threshold_results:
        print("  (no thresholded sets evaluated)")
    for r in threshold_results:
        marker = "PASS" if r["ok"] else "FAIL"
        print(f"  [{marker}] {r['set']}.{r['metric']}: actual={r['actual']:.4f} "
              f"target={r['target']}")
    print("-" * 72)
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 72)


# --- CLI -----------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval.run",
        description="Run lossless / quality / slice eval on the cleaner model.",
    )
    p.add_argument(
        "--base",
        type=str,
        default="models/base/gemma-4-E2B-it-4bit",
        help="Path to the base 4-bit model (default: models/base/gemma-4-E2B-it-4bit).",
    )
    p.add_argument(
        "--adapter",
        type=str,
        default="",
        help="Path to LoRA adapter dir. Empty/omitted = baseline (base model only).",
    )
    p.add_argument(
        "--eval-sets",
        type=str,
        default="val,eval_real,eval_passthrough",
        help="Comma-separated list of eval sets (val, eval_real, eval_passthrough).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the eval jsonl files.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens per generation. Default 4096 (matches training cap).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap records-per-set for smoke testing. Default: no limit.",
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("eval/reports"),
        help="Where to write the JSON report.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    eval_sets = [s.strip() for s in args.eval_sets.split(",") if s.strip()]
    unknown = [s for s in eval_sets if s not in DEFAULT_DATA_PATHS]
    if unknown:
        print(f"ERROR: unknown eval set(s): {unknown}. "
              f"Known: {list(DEFAULT_DATA_PATHS)}", file=sys.stderr)
        return 2

    adapter_arg = args.adapter if args.adapter else None
    print(f"Loading model from {args.base}"
          f"{' + adapter ' + adapter_arg if adapter_arg else ' (no adapter — baseline)'}")
    model, tokenizer = load(args.base, adapter_path=adapter_arg)

    per_set_records: dict[str, list[dict]] = {}
    per_set_summary: dict[str, dict] = {}
    per_set_slices: dict[str, dict] = {}

    for set_name in eval_sets:
        # Allow --data-dir override of the default filename.
        path = args.data_dir / Path(DEFAULT_DATA_PATHS[set_name]).name
        if not path.exists():
            print(f"WARN: {path} does not exist; skipping '{set_name}'", file=sys.stderr)
            continue
        records = _evaluate_set(
            model,
            tokenizer,
            path,
            max_tokens=args.max_tokens,
            limit=args.limit,
        )
        per_set_records[set_name] = records
        per_set_summary[set_name] = _summarize(records)
        per_set_slices[set_name] = _slices_for_set(records)

    overall_pass, threshold_results = check_thresholds(per_set_summary)
    _print_report(per_set_summary, per_set_slices, threshold_results, overall_pass)

    # Write JSON report.
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    suffix = "baseline" if adapter_arg is None else "adapter"
    report_path = args.reports_dir / f"{timestamp}-{suffix}.json"
    payload = {
        "timestamp": timestamp,
        "base": args.base,
        "adapter": adapter_arg,
        "eval_sets": eval_sets,
        "limit": args.limit,
        "summary": per_set_summary,
        "slices": per_set_slices,
        "thresholds": threshold_results,
        "overall_pass": overall_pass,
        # Record-level dumps live under records.<set> so consumers can post-hoc analyze.
        "records": per_set_records,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nReport written to {report_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
