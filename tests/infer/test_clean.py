"""Unit tests for ``infer.clean``.

These tests must NEVER load real model weights. We patch ``_load_model`` and
``_generate`` so they exercise only the prompt / postprocess / guard logic.
"""
from unittest.mock import MagicMock, patch

import pytest

from infer import clean as clean_mod


def _mock_tokenizer():
    """Tokenizer mock supporting ``apply_chat_template``."""
    tok = MagicMock()
    tok.apply_chat_template.return_value = "<chat>USER<|turn>model\n"
    return tok


def _mock_load(*args, **kwargs):
    return MagicMock(), _mock_tokenizer()


@pytest.fixture(autouse=True)
def _clear_cache():
    clean_mod._model_cache = None
    yield
    clean_mod._model_cache = None


def test_clean_uses_model_output_when_guard_passes():
    with patch.object(clean_mod, "_load_model", side_effect=_mock_load), patch.object(
        clean_mod, "_generate", return_value="hello world foo<channel|>"
    ):
        out = clean_mod.clean("\x1b[31mhello world foo\x1b[0m")
        assert out == "hello world foo"


def test_clean_fallback_when_guard_fails():
    with patch.object(clean_mod, "_load_model", side_effect=_mock_load), patch.object(
        clean_mod, "_generate", return_value="some unrelated text<channel|>"
    ):
        out = clean_mod.clean("\x1b[31mimportant_id_42 path /etc/hosts\x1b[0m")
        # Falls back to the deterministically ANSI-stripped input.
        assert "important_id_42" in out
        assert "/etc/hosts" in out


def test_clean_close_marker_postprocess():
    """If the model's output contains the close marker, anything after it is dropped."""
    with patch.object(clean_mod, "_load_model", side_effect=_mock_load), patch.object(
        clean_mod, "_generate", return_value="hello<channel|> trailing junk"
    ):
        out = clean_mod.clean("hello")
        assert out == "hello"


def test_clean_caches_model():
    """Model is loaded once and reused across calls."""
    load_calls = {"n": 0}

    def _counted_load(*a, **k):
        load_calls["n"] += 1
        return MagicMock(), _mock_tokenizer()

    with patch.object(clean_mod, "_load_model", side_effect=_counted_load), patch.object(
        clean_mod, "_generate", return_value="x<channel|>"
    ):
        clean_mod.clean("a")
        clean_mod.clean("b")
    assert load_calls["n"] == 1


def test_clean_empty_input_returns_empty():
    # Should not even try to load the model.
    with patch.object(clean_mod, "_load_model", side_effect=AssertionError("must not load")):
        assert clean_mod.clean("") == ""


def test_clean_no_close_marker_emitted():
    """If the model never emits ``<channel|>``, treat the whole output as the answer."""
    with patch.object(clean_mod, "_load_model", side_effect=_mock_load), patch.object(
        clean_mod, "_generate", return_value="hello world foo"
    ):
        out = clean_mod.clean("\x1b[31mhello world foo\x1b[0m")
        assert out == "hello world foo"


def test_clean_prompt_uses_chat_template_and_prefill():
    """Verify the prompt is constructed via ``apply_chat_template`` + ``CHANNEL_OPEN``."""
    captured = {}

    def _capture_generate(model, tokenizer, prompt, **kwargs):
        captured["prompt"] = prompt
        return "out<channel|>"

    with patch.object(clean_mod, "_load_model", side_effect=_mock_load), patch.object(
        clean_mod, "_generate", side_effect=_capture_generate
    ):
        clean_mod.clean("anything")

    assert captured["prompt"].endswith(clean_mod.CHANNEL_OPEN)
    # Tokenizer's chat-template output is prefixed.
    assert captured["prompt"].startswith("<chat>USER<|turn>model\n")


def test_clean_user_content_includes_instruction_and_pre_stripped_input():
    """The user-side message should include INSTRUCTION + the ANSI-stripped input."""
    captured = {}
    tok = _mock_tokenizer()

    def _capture_apply(messages, tokenize, add_generation_prompt):
        captured["messages"] = messages
        return "<chat>USER<|turn>model\n"

    tok.apply_chat_template.side_effect = _capture_apply

    def _load(*a, **k):
        return MagicMock(), tok

    with patch.object(clean_mod, "_load_model", side_effect=_load), patch.object(
        clean_mod, "_generate", return_value="x<channel|>"
    ):
        clean_mod.clean("\x1b[31mhello\x1b[0m")

    assert len(captured["messages"]) == 1
    msg = captured["messages"][0]
    assert msg["role"] == "user"
    # INSTRUCTION + pre-stripped (ANSI removed) input.
    from train.prompt_template import INSTRUCTION

    assert INSTRUCTION in msg["content"]
    assert "hello" in msg["content"]
    assert "\x1b[31m" not in msg["content"]


def test_resolve_model_paths_prefers_trained(tmp_path, monkeypatch):
    """When ``models/trained`` (override) exists & is non-empty, return it as the merged path."""
    trained = tmp_path / "trained"
    trained.mkdir()
    (trained / "config.json").write_text("{}")
    monkeypatch.setenv("CLEAN_MODEL_PATH", str(trained))

    model_path, adapter = clean_mod._resolve_model_paths()
    assert model_path == str(trained)
    assert adapter is None


def test_resolve_model_paths_falls_back_to_adapter(tmp_path, monkeypatch):
    """When merged path is missing, fall back to base + adapter if adapter exists."""
    missing = tmp_path / "trained"  # never created
    base = tmp_path / "base"
    base.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")

    monkeypatch.setenv("CLEAN_MODEL_PATH", str(missing))
    monkeypatch.setenv("CLEAN_MODEL_BASE", str(base))
    monkeypatch.setenv("CLEAN_MODEL_ADAPTER", str(adapter))

    model_path, adapter_out = clean_mod._resolve_model_paths()
    assert model_path == str(base)
    assert adapter_out == str(adapter)


def test_resolve_model_paths_falls_back_to_base_only(tmp_path, monkeypatch):
    """When neither merged nor adapter exist, return base alone."""
    missing_trained = tmp_path / "trained"
    base = tmp_path / "base"
    base.mkdir()
    missing_adapter = tmp_path / "adapter"  # never created

    monkeypatch.setenv("CLEAN_MODEL_PATH", str(missing_trained))
    monkeypatch.setenv("CLEAN_MODEL_BASE", str(base))
    monkeypatch.setenv("CLEAN_MODEL_ADAPTER", str(missing_adapter))

    model_path, adapter_out = clean_mod._resolve_model_paths()
    assert model_path == str(base)
    assert adapter_out is None
