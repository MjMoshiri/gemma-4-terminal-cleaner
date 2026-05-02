#!/usr/bin/env bash
set -euo pipefail

CONFIG="train/config.yaml"
ARGS=(
    --train
    --config "$CONFIG"
)

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Would run: caffeinate -i uv run python -m train.run_lora ${ARGS[*]}"
    exit 0
fi

if [[ "${1:-}" == "--steps" ]]; then
    # Override iters for smoke test
    OVERRIDE_ITERS="$2"
    echo "Smoke-test: $OVERRIDE_ITERS steps"
    ARGS+=( --iters "$OVERRIDE_ITERS" )
fi

# Ensure data prepared
if [[ ! -f data/mlx/train.jsonl ]]; then
    echo "Formatting dataset..."
    uv run python -m train.format_dataset
fi

mkdir -p models/adapter

# Caffeinate keeps the Mac awake during long runs
exec caffeinate -i uv run python -m train.run_lora "${ARGS[@]}"
