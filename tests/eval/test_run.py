"""Smoke tests for eval/run.py — no model weights touched."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Importing eval.run pulls in mlx_lm; that's fine in the dev env (it's a project dep).
from eval import run as evalrun


# --- CLI smoke -----------------------------------------------------------


def test_help_runs_without_error():
    """`python -m eval.run --help` exits cleanly with a usage banner."""
    result = subprocess.run(
        [sys.executable, "-m", "eval.run", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "eval.run" in result.stdout or "usage" in result.stdout.lower()
    assert "--adapter" in result.stdout
    assert "--eval-sets" in result.stdout


def test_unknown_eval_set_returns_nonzero(tmp_path):
    """Passing an unknown set name should fail fast (rc=2) BEFORE loading any model."""
    # We invoke main() directly; load() would be called only for known sets, but
    # we error out earlier here so it's safe.
    rc = evalrun.main(["--eval-sets", "bogus_set", "--data-dir", str(tmp_path)])
    assert rc == 2


# --- Threshold check logic ------------------------------------------------


def test_check_thresholds_all_pass():
    summary = {
        "eval_real": {
            "count": 100,
            "guard_pass_rate": 1.0,
            "normalized_match_rate": 0.95,
            "median_token_reduction": 0.5,
            "exact_match_rate": 0.5,
            "mean_token_reduction": 0.5,
        },
        "eval_passthrough": {
            "count": 50,
            "guard_pass_rate": 1.0,
            "normalized_match_rate": 1.0,
            "median_token_reduction": 0.0,
            "exact_match_rate": 1.0,
            "mean_token_reduction": 0.0,
        },
    }
    overall, results = evalrun.check_thresholds(summary)
    assert overall is True
    # Every check must have ok=True.
    assert all(r["ok"] for r in results), [r for r in results if not r["ok"]]


def test_check_thresholds_fails_on_low_guard():
    summary = {
        "eval_real": {
            "count": 100,
            "guard_pass_rate": 0.50,  # below 0.995
            "normalized_match_rate": 0.95,
            "median_token_reduction": 0.5,
            "exact_match_rate": 0.5,
            "mean_token_reduction": 0.5,
        },
    }
    overall, results = evalrun.check_thresholds(summary)
    assert overall is False
    failed = [r for r in results if not r["ok"]]
    assert any(r["metric"] == "guard_pass_rate" for r in failed)


def test_check_thresholds_fails_on_passthrough_oversize_reduction():
    summary = {
        "eval_passthrough": {
            "count": 50,
            "guard_pass_rate": 1.0,
            "normalized_match_rate": 1.0,
            # 5% reduction, exceeds the 2% bound (in absolute value).
            "median_token_reduction": 0.05,
            "exact_match_rate": 1.0,
            "mean_token_reduction": 0.05,
        },
    }
    overall, results = evalrun.check_thresholds(summary)
    assert overall is False
    failed = [r for r in results if not r["ok"]]
    assert any(r["metric"] == "max_abs_median_token_reduction" for r in failed)


def test_check_thresholds_skips_set_with_no_records():
    """Sets that weren't evaluated (count=0 or missing) shouldn't fail the run."""
    overall, results = evalrun.check_thresholds({})
    assert overall is True
    assert results == []


# --- _evaluate_set with mocked model/tokenizer/generate -------------------


class _StubTokenizer:
    """Minimal tokenizer stub: no real chat template, fake encode."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Return the user content; preserve the user-turn payload for assertion.
        assert add_generation_prompt is True, "must request generation prompt"
        assert tokenize is False
        return f"<USER>{messages[0]['content']}</USER>"

    def encode(self, text):
        # Token count proxy: one token per whitespace-separated chunk.
        return text.split() if text else []


def _stub_generate_returns_input(model, tokenizer, prompt, max_tokens, verbose):
    """Pretend the model perfectly echoes the input portion of the prompt."""
    # Extract back the user content (everything between the markers).
    start = prompt.index("<USER>") + len("<USER>")
    end = prompt.index("</USER>")
    user_content = prompt[start:end]
    # Strip the INSTRUCTION + framing so we approximate "the dirty payload".
    if "---\n" in user_content:
        payload = user_content.split("---\n", 1)[1].rsplit("\n---", 1)[0]
        return payload
    return user_content


def test_evaluate_set_shape(tmp_path):
    """`_evaluate_set` produces a list of dicts with the expected keys."""
    src = tmp_path / "tiny.jsonl"
    src.write_text(
        "\n".join(
            json.dumps(rec)
            for rec in [
                {
                    "input": "hello world foo",
                    "output": "hello world foo",
                    "meta": {"recipe": "passthrough", "archetype": "log"},
                },
                {
                    "input": "ls -la 100 200",
                    "output": "ls -la 100 200",
                    "meta": {"command": "ls", "archetype": "table"},
                },
            ]
        )
        + "\n"
    )

    records = evalrun._evaluate_set(
        model=None,
        tokenizer=_StubTokenizer(),
        path=src,
        max_tokens=128,
        limit=None,
        generate_fn=_stub_generate_returns_input,
    )

    assert len(records) == 2
    expected_keys = {
        "input", "gold", "pred", "guard_passed", "missing_atoms",
        "exact", "normalized", "n_in", "n_out", "tok_red", "meta",
    }
    for r in records:
        assert expected_keys.issubset(r.keys()), r.keys()
        # Stub echoed the dirty payload, so guard MUST pass.
        assert r["guard_passed"] is True
        assert r["missing_atoms"] == []


def test_evaluate_set_respects_limit(tmp_path):
    src = tmp_path / "tiny.jsonl"
    src.write_text(
        "\n".join(
            json.dumps({"input": f"line {i}", "output": f"line {i}"})
            for i in range(5)
        )
        + "\n"
    )
    records = evalrun._evaluate_set(
        model=None,
        tokenizer=_StubTokenizer(),
        path=src,
        max_tokens=64,
        limit=2,
        generate_fn=_stub_generate_returns_input,
    )
    assert len(records) == 2


# --- Aggregation / slicing ------------------------------------------------


def test_summarize_basic():
    records = [
        {"guard_passed": True, "exact": True, "normalized": True, "n_in": 100,
         "n_out": 50, "tok_red": 0.5},
        {"guard_passed": False, "exact": False, "normalized": True, "n_in": 100,
         "n_out": 60, "tok_red": 0.4},
    ]
    s = evalrun._summarize(records)
    assert s["count"] == 2
    assert s["guard_pass_rate"] == 0.5
    assert s["normalized_match_rate"] == 1.0
    assert s["exact_match_rate"] == 0.5
    assert s["median_token_reduction"] == pytest.approx(0.45)


def test_summarize_empty():
    s = evalrun._summarize([])
    assert s["count"] == 0
    assert s["guard_pass_rate"] == 0.0


def test_slices_for_set_groups_by_recipe():
    records = [
        {"guard_passed": True, "normalized": True, "n_in": 100,
         "meta": {"recipe": "passthrough"}},
        {"guard_passed": False, "normalized": False, "n_in": 100,
         "meta": {"recipe": "passthrough"}},
        {"guard_passed": True, "normalized": True, "n_in": 5000,
         "meta": {"recipe": "noisy_logs"}},
    ]
    slices = evalrun._slices_for_set(records)
    assert "passthrough" in slices["by_recipe"]
    assert slices["by_recipe"]["passthrough"]["count"] == 2
    assert slices["by_recipe"]["passthrough"]["pass_rate"] == 0.5
    assert "2k-8k" in slices["by_length"]
