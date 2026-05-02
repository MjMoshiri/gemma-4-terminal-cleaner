"""Lossless terminal-output cleaner. Text-in / text-out with safe fallback.

Loads the trained Gemma 4 E2B model (or base+adapter, or base alone) lazily on
first ``clean()`` call, runs inference with the same channel-marker prompt
structure used in training (see ``train/format_for_cloud.py``), then verifies
the output preserves every information atom from the input via the lossless
guard (see ``eval/lossless_guard.py``). On guard failure we fall back to the
deterministic ANSI strip — never silently lose data.

CLI:
    cat dirty.log | uv run python -m infer.clean > clean.log
"""
from __future__ import annotations

# Patch MUST be imported before any mlx_lm import (kv-shared layers fix for the
# base 4-bit checkpoint). Harmless for the merged trained model.
import train.patch_mlx_lm  # noqa: F401

import logging
import os
from pathlib import Path
from typing import Optional

from eval.lossless_guard import lossless_check
from infer.ansi_strip import strip
from train.format_for_cloud import CHANNEL_CLOSE, CHANNEL_OPEN
from train.prompt_template import INSTRUCTION

_logger = logging.getLogger("infer.clean")
_model_cache: Optional[tuple] = None


def _resolve_model_paths() -> tuple[str, Optional[str]]:
    """Return ``(model_path, adapter_or_None)``.

    Resolution order:
    1. ``CLEAN_MODEL_PATH`` (default ``models/trained``) if it exists and is
       non-empty — a merged + re-quantized 4-bit MLX model from cloud_train.
    2. Otherwise ``CLEAN_MODEL_BASE`` + ``CLEAN_MODEL_ADAPTER`` if the adapter
       dir exists and is non-empty — base model with LoRA adapter overlay.
    3. Otherwise ``CLEAN_MODEL_BASE`` alone — smoke / pre-training mode.
    """
    merged = os.environ.get("CLEAN_MODEL_PATH", "models/trained")
    merged_p = Path(merged)
    if merged_p.exists() and any(merged_p.iterdir()):
        return merged, None
    base = os.environ.get("CLEAN_MODEL_BASE", "models/base/gemma-4-E2B-it-4bit")
    adapter = os.environ.get("CLEAN_MODEL_ADAPTER", "models/adapter")
    adapter_p = Path(adapter)
    if adapter_p.exists() and any(adapter_p.iterdir()):
        return base, adapter
    return base, None


def _load_model():
    """Lazy-load the model + tokenizer. Mockable in tests."""
    from mlx_lm import load

    model_path, adapter = _resolve_model_paths()
    if adapter is None:
        return load(model_path)
    return load(model_path, adapter_path=adapter)


def _generate(model, tokenizer, prompt: str, max_tokens: int = 4096) -> str:
    """Mockable generation wrapper."""
    from mlx_lm import generate

    return generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
    )


def _build_prompt(tokenizer, dirty_pre_stripped: str) -> str:
    """Construct the chat-templated prompt with channel-marker prefill.

    Mirrors the structure used in training (see ``train/prompt_template.py``
    and ``train/format_for_cloud.py``) so the trained model continues
    naturally into the final-channel content.
    """
    user_content = f"{INSTRUCTION}\n\n---\n{dirty_pre_stripped}\n---"
    messages = [{"role": "user", "content": user_content}]
    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return base_prompt + CHANNEL_OPEN


def _postprocess(raw_pred: str) -> str:
    """Strip the close marker and any surrounding whitespace.

    Also passes the result through ``ansi_strip.strip`` defensively in case
    the model emitted any escape sequences.
    """
    answer = raw_pred.split(CHANNEL_CLOSE, 1)[0]
    return strip(answer).strip()


def clean(dirty: str) -> str:
    """Lossless terminal-output cleanup.

    Returns the model's cleaned output if it passes the lossless guard
    (every information atom from the input is preserved); otherwise returns
    the deterministically pre-stripped input as a safe fallback.

    Empty input returns empty string.
    """
    global _model_cache
    if not dirty:
        return dirty
    if _model_cache is None:
        _model_cache = _load_model()
    model, tokenizer = _model_cache

    pre = strip(dirty)
    prompt = _build_prompt(tokenizer, pre)
    raw_pred = _generate(model, tokenizer, prompt)
    pred = _postprocess(raw_pred)

    guard = lossless_check(dirty, pred)
    if guard.passed:
        return pred
    _logger.warning(
        "lossless guard failed; falling back to deterministic strip. "
        "missing atoms (first 10): %s",
        sorted(guard.missing_atoms)[:10],
    )
    return pre


def main() -> None:
    """CLI: stdin -> clean() -> stdout."""
    import sys

    dirty = sys.stdin.read()
    sys.stdout.write(clean(dirty))


if __name__ == "__main__":
    main()
