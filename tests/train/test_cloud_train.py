"""Smoke tests for train/cloud_train.py — imports + pure helpers only.

We do NOT invoke Modal (`modal run …`) here. Instead we verify:
  - the module imports cleanly when Modal is available;
  - the pure helpers work on a tiny synthetic dataset;
  - constants reflect the documented model id resolution.

The whole file is skipped if ``modal`` isn't installed (it's an optional
extra; install with ``uv pip install -e .[cloud]``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

modal = pytest.importorskip("modal")  # noqa: F841 — used to gate the file


def _import_module():
    """Import lazily so the importorskip guard runs first."""
    from train import cloud_train

    return cloud_train


def test_module_imports_without_error():
    cloud_train = _import_module()
    # Sanity: the Modal app + functions exist and have the expected names.
    assert cloud_train.APP_NAME == "gemma-4-terminal-cleaner"
    assert hasattr(cloud_train, "train")
    assert hasattr(cloud_train, "upload_data_local")
    assert hasattr(cloud_train, "download_output")
    assert hasattr(cloud_train, "main")


def test_model_ids_documented():
    """The two canonical Gemma 4 ids should be present in the constants."""
    cloud_train = _import_module()
    assert cloud_train.MODEL_ID_HF == "google/gemma-4-E2B-it"
    assert cloud_train.MODEL_ID_UNSLOTH == "unsloth/gemma-4-E2B-it"
    # HF fallback list must include the canonical id first.
    assert cloud_train.HF_FALLBACK_CANDIDATES[0] == cloud_train.MODEL_ID_HF


def test_load_jsonl_with_limit(tmp_path: Path):
    cloud_train = _import_module()
    src = tmp_path / "x.jsonl"
    src.write_text("\n".join(json.dumps({"i": i}) for i in range(5)) + "\n")
    rows = cloud_train.load_jsonl(src, limit=3)
    assert [r["i"] for r in rows] == [0, 1, 2]


def test_load_jsonl_skips_blank_lines(tmp_path: Path):
    cloud_train = _import_module()
    src = tmp_path / "x.jsonl"
    src.write_text(
        json.dumps({"a": 1}) + "\n\n   \n" + json.dumps({"a": 2}) + "\n"
    )
    rows = cloud_train.load_jsonl(src)
    assert rows == [{"a": 1}, {"a": 2}]


def test_render_chat_for_training_uses_apply_chat_template():
    cloud_train = _import_module()

    class _Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            # Critical: training-time templating must NOT add a generation prompt
            # because we're emitting full chat including the assistant turn.
            assert add_generation_prompt is False
            return f"|{messages[0]['content']}::{messages[1]['content']}|"

    rec = {
        "messages": [
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "<|channel>final\nA<channel|>"},
        ]
    }
    out = cloud_train.render_chat_for_training(_Tok(), rec)
    assert out == "|U::<|channel>final\nA<channel|>|"


def test_build_lora_config_targets_qkvo():
    """Cloud LoRA targets all four projections."""
    pytest.importorskip("peft")
    cloud_train = _import_module()
    cfg = cloud_train.build_lora_config()
    assert set(cfg.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05


def test_remote_paths_are_absolute():
    cloud_train = _import_module()
    assert cloud_train.REMOTE_DATA_DIR.is_absolute()
    assert cloud_train.REMOTE_INPUT_DIR.is_absolute()
    assert cloud_train.REMOTE_OUTPUT_DIR.is_absolute()


def test_main_entrypoint_exists():
    """``modal run train/cloud_train.py::main --dry-run`` requires this exists."""
    cloud_train = _import_module()
    assert callable(cloud_train.main)
