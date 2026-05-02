"""Wrap assistant turns in Gemma 4 channel markers for cloud SFT.

Read MLX-format chat jsonl from ``--in`` (expects ``train.jsonl`` and
``valid.jsonl`` — falls back to ``val.jsonl``) and write the same jsonl into
``--out`` with each assistant message's content wrapped as::

    <|channel>final\\n{original}<channel|>

Why: PoC found Gemma 4's base model defaults to the *thought* channel
(``<|channel>thought\\n...``) when given a bare assistant turn. Baking the
final-channel structure into training data teaches the trained model to emit
``<|channel>final\\n{answer}<channel|>`` directly. The eval/inference paths
then prefill ``<|channel>final\\n`` after ``add_generation_prompt`` and strip
output at the first ``<channel|>``.

Usage::

    uv run python -m train.format_for_cloud --in data/mlx --out data/cloud
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CHANNEL_OPEN = "<|channel>final\n"
CHANNEL_CLOSE = "<channel|>"


def wrap_assistant_content(content: str) -> str:
    """Wrap a single assistant message string with the final-channel markers.

    Idempotent: if ``content`` is already wrapped (starts with the open
    marker AND ends with the close marker), returns it unchanged. This keeps
    re-runs of the formatter safe.
    """
    if content.startswith(CHANNEL_OPEN) and content.endswith(CHANNEL_CLOSE):
        return content
    return f"{CHANNEL_OPEN}{content}{CHANNEL_CLOSE}"


def transform_record(rec: dict) -> dict:
    """Return a new record with the assistant turn (last message) wrapped.

    Validates the basic shape: ``rec["messages"]`` is a non-empty list whose
    last element has ``role == "assistant"``. Raises ``ValueError`` otherwise
    so format drift is loud.
    """
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError(f"record missing non-empty 'messages' list: {rec!r}")
    last = msgs[-1]
    if last.get("role") != "assistant":
        raise ValueError(
            f"last message must be role=assistant, got role={last.get('role')!r}"
        )
    if not isinstance(last.get("content"), str):
        raise ValueError(
            f"assistant content must be string, got {type(last.get('content')).__name__}"
        )
    new_msgs = list(msgs[:-1]) + [
        {**last, "content": wrap_assistant_content(last["content"])}
    ]
    return {**rec, "messages": new_msgs}


def transform_file(src: Path, dst: Path, limit: int | None = None) -> tuple[int, int]:
    """Transform every line of ``src`` into ``dst``. Return (n_in, n_out).

    If ``limit`` is set, stop after writing that many records.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with src.open(encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if limit is not None and n_out >= limit:
                break
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            out_rec = transform_record(rec)
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_out += 1
    return n_in, n_out


def _resolve_pair(src_dir: Path) -> list[tuple[Path, str]]:
    """Resolve input file names. We accept both ``valid.jsonl`` (MLX-LM
    convention) and ``val.jsonl``. Output filename is normalized to
    ``train.jsonl`` / ``val.jsonl`` per spec."""
    pairs: list[tuple[Path, str]] = []
    train = src_dir / "train.jsonl"
    if not train.exists():
        raise FileNotFoundError(f"missing {train}")
    pairs.append((train, "train.jsonl"))

    val_candidates = [src_dir / "val.jsonl", src_dir / "valid.jsonl"]
    val_src = next((p for p in val_candidates if p.exists()), None)
    if val_src is None:
        raise FileNotFoundError(
            f"missing val/valid jsonl under {src_dir}: tried {val_candidates}"
        )
    pairs.append((val_src, "val.jsonl"))
    return pairs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="train.format_for_cloud",
        description="Wrap assistant turns with Gemma 4 final-channel markers.",
    )
    p.add_argument(
        "--in",
        dest="src",
        type=Path,
        default=Path("data/mlx"),
        help="Source dir containing train.jsonl + val.jsonl/valid.jsonl.",
    )
    p.add_argument(
        "--out",
        dest="dst",
        type=Path,
        default=Path("data/cloud"),
        help="Destination dir; train.jsonl + val.jsonl will be written here.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap records per file for smoke testing (default: no limit).",
    )
    args = p.parse_args(argv)

    args.dst.mkdir(parents=True, exist_ok=True)
    for src, out_name in _resolve_pair(args.src):
        dst = args.dst / out_name
        n_in, n_out = transform_file(src, dst, limit=args.limit)
        print(f"  {src.name} -> {dst}: in={n_in} out={n_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
