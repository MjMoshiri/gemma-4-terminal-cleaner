"""Tests for the Gemma 4 E2B kv-shared sanitize monkey-patch.

These tests must NOT load the real Gemma 4 weights — they fake `self.args`
and stub out the original `sanitize` so we only verify our filter logic.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest


@pytest.fixture
def patch_mod():
    """Import the patch module fresh and yield it.

    We don't reload between tests because `apply()` is supposed to be
    idempotent, and importing once mirrors real-world usage.
    """
    import train.patch_mlx_lm as patch_mlx_lm

    return patch_mlx_lm


def test_import_swaps_in_patched_sanitize(patch_mod):
    """Importing train.patch_mlx_lm should replace Model.sanitize."""
    from mlx_lm.models import gemma4_text as g4t

    assert getattr(g4t.Model.sanitize, "_gemma4_kv_patched", False) is True
    assert g4t.Model.sanitize is patch_mod._patched_sanitize


def _fake_self(num_hidden_layers: int = 35, num_kv_shared_layers: int = 20):
    return SimpleNamespace(
        args=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            num_kv_shared_layers=num_kv_shared_layers,
        )
    )


def test_patched_sanitize_drops_kv_for_shared_layers(patch_mod):
    """Layers >= first_kv_shared (=15) lose k_proj/v_proj/k_norm weights."""
    weights = {
        # Layer 0: kept (below first_kv_shared)
        "language_model.model.layers.0.self_attn.k_proj.weight": "k0",
        "language_model.model.layers.0.self_attn.v_proj.weight": "v0",
        "language_model.model.layers.0.self_attn.k_norm.weight": "kn0",
        # Layer 14: kept (last layer below first_kv_shared)
        "language_model.model.layers.14.self_attn.v_proj.weight": "v14",
        # Layer 15: dropped (first kv-shared layer)
        "language_model.model.layers.15.self_attn.k_proj.weight": "k15",
        "language_model.model.layers.15.self_attn.v_proj.weight": "v15",
        "language_model.model.layers.15.self_attn.k_norm.weight": "kn15",
        # Layer 34: dropped
        "language_model.model.layers.34.self_attn.k_proj.weight": "k34",
        # Quantized suffixes also dropped (regex unanchored on right)
        "language_model.model.layers.20.self_attn.k_proj.scales": "s20",
        "language_model.model.layers.20.self_attn.v_proj.biases": "b20",
        # Unrelated weight: always kept
        "language_model.model.layers.15.self_attn.q_proj.weight": "q15",
        "language_model.model.embed_tokens.weight": "embed",
    }

    fake_self = _fake_self()
    # Stub the original sanitize so we only test our filter behavior.
    with mock.patch.object(
        patch_mod, "_orig_sanitize", side_effect=lambda self, w: w
    ) as orig:
        result = patch_mod._patched_sanitize(fake_self, weights)

    # Verify exactly which keys reached _orig_sanitize.
    forwarded = orig.call_args.args[1]
    assert set(forwarded.keys()) == {
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
        "language_model.model.layers.0.self_attn.k_norm.weight",
        "language_model.model.layers.14.self_attn.v_proj.weight",
        "language_model.model.layers.15.self_attn.q_proj.weight",
        "language_model.model.embed_tokens.weight",
    }
    # And the patched sanitize returns whatever orig returned.
    assert result is forwarded


def test_patched_sanitize_delegates_to_orig(patch_mod):
    """The patched sanitize must call the original with the filtered dict."""
    fake_self = _fake_self()
    sentinel = object()
    with mock.patch.object(
        patch_mod, "_orig_sanitize", return_value=sentinel
    ) as orig:
        out = patch_mod._patched_sanitize(fake_self, {})

    orig.assert_called_once()
    assert out is sentinel


def test_apply_is_idempotent(patch_mod):
    """Calling apply() twice should not double-wrap or change state."""
    from mlx_lm.models import gemma4_text as g4t

    first = g4t.Model.sanitize
    patch_mod.apply()
    patch_mod.apply()
    assert g4t.Model.sanitize is first
    assert getattr(g4t.Model.sanitize, "_gemma4_kv_patched", False) is True


def test_first_kv_shared_boundary_with_different_args(patch_mod):
    """Boundary should be computed from self.args, not hard-coded."""
    # E.g. a hypothetical model with num_hidden_layers=10, shared=4 => first=6
    fake_self = _fake_self(num_hidden_layers=10, num_kv_shared_layers=4)
    weights = {
        "language_model.model.layers.5.self_attn.k_proj.weight": "k5",  # keep
        "language_model.model.layers.6.self_attn.k_proj.weight": "k6",  # drop
    }
    with mock.patch.object(
        patch_mod, "_orig_sanitize", side_effect=lambda self, w: w
    ) as orig:
        patch_mod._patched_sanitize(fake_self, weights)
    forwarded = orig.call_args.args[1]
    assert "language_model.model.layers.5.self_attn.k_proj.weight" in forwarded
    assert "language_model.model.layers.6.self_attn.k_proj.weight" not in forwarded
