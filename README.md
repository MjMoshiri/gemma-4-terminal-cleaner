# Gemma 4 E2B Terminal Output Cleaner

Fine-tuned Gemma 4 E2B for lossless cleaning of dirty terminal output.

See [design spec](docs/superpowers/specs/2026-05-02-gemma-terminal-cleaner-design.md) for the full architecture.

## Quickstart

```bash
uv sync --extra dev
uv run pytest -q

# Download base model
uv run hf download mlx-community/gemma-4-E2B-it-4bit \
    --local-dir models/base/gemma-4-E2B-it-4bit

# Generate corpus + training data (Tasks 4 + 9)
uv run python -m corpus.generate_corpus
uv run python -m dirtifier.generate

# Train (Task 16)
bash train/run.sh

# Eval (Task 19)
uv run python -m eval.run --adapter models/adapter/

# Use it
echo $'\x1b[31mhello\x1b[0m' | uv run python -m infer.clean
```
