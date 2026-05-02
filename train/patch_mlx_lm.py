"""Runtime monkey-patch for mlx_lm to handle Gemma 4 E2B's kv-shared layers.

The mlx-community/gemma-4-E2B-it-4bit checkpoint includes k_proj/v_proj/k_norm
weights for all transformer layers, but mlx-lm 0.31.3's gemma4_text.Model
only allocates these for layers below `num_hidden_layers - num_kv_shared_layers`.
We pre-filter the weights dict before mlx-lm's strict load checks them.

Import this module before importing mlx_lm.lora or mlx_lm.utils.load.
"""

from __future__ import annotations

import re

from mlx_lm.models import gemma4_text as _g4t

# Unanchored on the right so it matches keys with any suffix
# (e.g. ``.weight``, ``.scales``, ``.biases`` for quantized checkpoints).
_KV_ONLY_PAT = re.compile(
    r"language_model\.model\.layers\.(\d+)\.self_attn\.(?:k_proj|v_proj|k_norm)\."
)

_orig_sanitize = _g4t.Model.sanitize


def _patched_sanitize(self, weights):
    first_kv_shared = (
        self.args.num_hidden_layers - self.args.num_kv_shared_layers
    )
    filtered = {}
    for k, v in weights.items():
        m = _KV_ONLY_PAT.search(k)
        if m and int(m.group(1)) >= first_kv_shared:
            continue
        filtered[k] = v
    return _orig_sanitize(self, filtered)


_patched_sanitize._gemma4_kv_patched = True  # type: ignore[attr-defined]


def apply() -> None:
    """Apply the monkey-patch. Idempotent."""
    if getattr(_g4t.Model.sanitize, "_gemma4_kv_patched", False):
        return
    _g4t.Model.sanitize = _patched_sanitize  # type: ignore[assignment]


# Auto-apply on import
apply()
