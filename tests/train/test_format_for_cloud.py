"""Unit tests for train/format_for_cloud.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from train import format_for_cloud as ffc


# --- wrap_assistant_content ---------------------------------------------


def test_wrap_assistant_content_adds_markers():
    out = ffc.wrap_assistant_content("answer body")
    assert out == "<|channel>final\nanswer body<channel|>"


def test_wrap_assistant_content_preserves_internal_newlines():
    out = ffc.wrap_assistant_content("line1\nline2\nline3")
    assert out == "<|channel>final\nline1\nline2\nline3<channel|>"


def test_wrap_assistant_content_idempotent_when_already_wrapped():
    once = ffc.wrap_assistant_content("body")
    twice = ffc.wrap_assistant_content(once)
    assert once == twice


def test_wrap_assistant_content_empty_string():
    """Edge: empty assistant content should still get markers (caller's choice)."""
    assert ffc.wrap_assistant_content("") == "<|channel>final\n<channel|>"


# --- transform_record ----------------------------------------------------


def test_transform_record_wraps_only_last_message():
    rec = {
        "messages": [
            {"role": "user", "content": "USER MSG"},
            {"role": "assistant", "content": "ASSISTANT MSG"},
        ]
    }
    out = ffc.transform_record(rec)
    assert out["messages"][0] == {"role": "user", "content": "USER MSG"}
    assert out["messages"][1]["role"] == "assistant"
    assert out["messages"][1]["content"] == "<|channel>final\nASSISTANT MSG<channel|>"


def test_transform_record_does_not_mutate_input():
    rec = {
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    }
    snapshot = json.dumps(rec, sort_keys=True)
    _ = ffc.transform_record(rec)
    assert json.dumps(rec, sort_keys=True) == snapshot


def test_transform_record_preserves_extra_fields():
    rec = {
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "weight": 1.0},
        ],
        "meta": {"src": "synth"},
    }
    out = ffc.transform_record(rec)
    assert out["meta"] == {"src": "synth"}
    assert out["messages"][1].get("weight") == 1.0  # extra fields kept


def test_transform_record_rejects_missing_messages():
    with pytest.raises(ValueError, match="messages"):
        ffc.transform_record({})


def test_transform_record_rejects_empty_messages():
    with pytest.raises(ValueError, match="messages"):
        ffc.transform_record({"messages": []})


def test_transform_record_rejects_non_assistant_last():
    with pytest.raises(ValueError, match="assistant"):
        ffc.transform_record(
            {"messages": [{"role": "user", "content": "no answer"}]}
        )


def test_transform_record_rejects_non_string_content():
    with pytest.raises(ValueError, match="string"):
        ffc.transform_record(
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": ["not", "a", "string"]},
                ]
            }
        )


# --- transform_file (file IO end-to-end) --------------------------------


def _make_record(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def test_transform_file_roundtrip(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    records = [
        _make_record("u1", "a1"),
        _make_record("u2", "multi\nline\nresponse"),
        _make_record("u3", ""),
    ]
    src.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    )
    n_in, n_out = ffc.transform_file(src, dst)
    assert n_in == n_out == 3

    out_lines = dst.read_text().splitlines()
    assert len(out_lines) == 3
    parsed = [json.loads(line) for line in out_lines]

    # Each line must be valid JSON with wrapped assistant content.
    for orig, got in zip(records, parsed):
        # User passes through unchanged.
        assert got["messages"][0] == orig["messages"][0]
        # Assistant content is wrapped with markers.
        wrapped = got["messages"][1]["content"]
        assert wrapped.startswith("<|channel>final\n")
        assert wrapped.endswith("<channel|>")
        # Round-trip: stripping the markers recovers the original content.
        inner = wrapped[len("<|channel>final\n"):-len("<channel|>")]
        assert inner == orig["messages"][1]["content"]


def test_transform_file_skips_blank_lines(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    src.write_text(
        json.dumps(_make_record("u", "a")) + "\n\n   \n"
        + json.dumps(_make_record("u2", "a2")) + "\n"
    )
    n_in, n_out = ffc.transform_file(src, dst)
    assert n_in == 2 and n_out == 2


def test_transform_file_creates_destination_parent(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(_make_record("u", "a")) + "\n")
    dst = tmp_path / "nested" / "deeper" / "out.jsonl"
    ffc.transform_file(src, dst)
    assert dst.exists()


# --- _resolve_pair (input layout flexibility) ----------------------------


def test_resolve_pair_accepts_valid_jsonl(tmp_path: Path):
    (tmp_path / "train.jsonl").write_text("")
    (tmp_path / "valid.jsonl").write_text("")
    pairs = ffc._resolve_pair(tmp_path)
    assert [p[1] for p in pairs] == ["train.jsonl", "val.jsonl"]
    assert pairs[1][0].name == "valid.jsonl"


def test_resolve_pair_accepts_val_jsonl(tmp_path: Path):
    (tmp_path / "train.jsonl").write_text("")
    (tmp_path / "val.jsonl").write_text("")
    pairs = ffc._resolve_pair(tmp_path)
    assert pairs[1][0].name == "val.jsonl"


def test_resolve_pair_missing_train(tmp_path: Path):
    (tmp_path / "valid.jsonl").write_text("")
    with pytest.raises(FileNotFoundError, match="train"):
        ffc._resolve_pair(tmp_path)


def test_resolve_pair_missing_val(tmp_path: Path):
    (tmp_path / "train.jsonl").write_text("")
    with pytest.raises(FileNotFoundError, match="val"):
        ffc._resolve_pair(tmp_path)


# --- CLI smoke ----------------------------------------------------------


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "train.format_for_cloud", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert "--in" in result.stdout
    assert "--out" in result.stdout


def test_cli_end_to_end(tmp_path: Path):
    src_dir = tmp_path / "mlx"
    src_dir.mkdir()
    (src_dir / "train.jsonl").write_text(
        json.dumps(_make_record("u-train", "a-train")) + "\n"
    )
    (src_dir / "valid.jsonl").write_text(
        json.dumps(_make_record("u-val", "a-val")) + "\n"
    )
    out_dir = tmp_path / "cloud"

    rc = ffc.main(["--in", str(src_dir), "--out", str(out_dir)])
    assert rc == 0

    train_out = (out_dir / "train.jsonl").read_text().strip()
    val_out = (out_dir / "val.jsonl").read_text().strip()
    train_rec = json.loads(train_out)
    val_rec = json.loads(val_out)
    assert train_rec["messages"][1]["content"] == "<|channel>final\na-train<channel|>"
    assert val_rec["messages"][1]["content"] == "<|channel>final\na-val<channel|>"


def test_cli_limit_truncates_per_file(tmp_path: Path):
    src_dir = tmp_path / "mlx"
    src_dir.mkdir()
    train_lines = [json.dumps(_make_record(f"u{i}", f"a{i}")) for i in range(5)]
    val_lines = [json.dumps(_make_record(f"vu{i}", f"va{i}")) for i in range(5)]
    (src_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n")
    (src_dir / "val.jsonl").write_text("\n".join(val_lines) + "\n")
    out_dir = tmp_path / "cloud"

    rc = ffc.main(
        ["--in", str(src_dir), "--out", str(out_dir), "--limit", "2"]
    )
    assert rc == 0
    assert (
        len((out_dir / "train.jsonl").read_text().strip().splitlines()) == 2
    )
    assert (
        len((out_dir / "val.jsonl").read_text().strip().splitlines()) == 2
    )
