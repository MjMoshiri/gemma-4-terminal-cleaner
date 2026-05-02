"""Entrypoint that applies the Gemma 4 E2B sanitize patch then runs mlx_lm.lora."""

from __future__ import annotations

import train.patch_mlx_lm  # noqa: F401  # apply() runs on import


def main() -> None:
    from mlx_lm.lora import main as lora_main
    lora_main()


if __name__ == "__main__":
    main()
