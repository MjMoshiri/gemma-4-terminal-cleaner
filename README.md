# Gemma 4 E2B Terminal Output Cleaner — PoC

Fine-tuning a small (2.3B effective params) open model to clean up dirty terminal output without losing data. Strip ANSI escapes, collapse `\r` progress-bar overwrites, dedupe repeated lines, normalize whitespace. No summarization, no dropped tokens.

Why bother: agent harnesses burn a lot of context on shell output that's mostly noise. A cheap on-device model that drops noise without touching data is a useful primitive. This repo asks a narrower question: can a small Gemma model, fine-tuned cheaply on synthetic data, learn this?

## TL;DR

The pipeline works end-to-end: synthetic corpus → 60M-token training set → Modal H100 LoRA fine-tune → MLX 4-bit conversion → local inference.

This was a 500-step run, ~$5 of GPU. The model learned the genre (output looks like clean terminal output) but didn't learn faithful copy yet — long inputs trigger repetition loops.

Honest read: base Gemma 4 E2B out of the box is already a decent baseline. The pipeline is reproducible, the model is undertrained, and a 5–10K-step run (~$50) is the obvious next step.

## Before / After

Three samples on identical prompts. **Base** is `mlx-community/gemma-4-E2B-it-4bit` out of the box. **Fine-tuned** is the 500-step LoRA-merged + MLX-quantized checkpoint from this repo. Full inputs/outputs in [`eval/reports/sample_before_after.json`](eval/reports/sample_before_after.json).

### 1. `pip install` progress bar (ANSI + `\r`-overwrites)

**Input** (raw bytes, escapes visible):

```
Collecting requests
  Downloading requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)\r
Downloading requests-2.31.0-py3-none-any.whl (62 kB)
   \x1b[91m━━━…━━━\x1b[0m \x1b[32m20.1/62.1 kB\x1b[0m \x1b[31m1.2 MB/s\x1b[0m eta \x1b[36m0:00:00\x1b[0m\r
   \x1b[91m━━━…━━━\x1b[0m \x1b[32m62.1/62.1 kB\x1b[0m \x1b[31m1.5 MB/s\x1b[0m eta \x1b[36m0:00:00\x1b[0m
Installing collected packages: requests
Successfully installed requests-2.31.0
```

**Base:** near-perfect. Strips ANSI, keeps both progress states, preserves install lines.

**Fine-tuned (500 steps):** repetition loop — `Downloading requests-2.31.0-py3-none-any.whl (4.6 kB)` repeated until cutoff. Classic undertraining: model knows what pip output looks like, doesn't know to mirror this specific input.

### 2. Build error with ANSI

**Input:** `ERROR: build failed` + `src/main.rs:42:5` + `error[E0308]: mismatched types` + `--> expected 'String', found '&str'`, all wrapped in red/yellow/cyan ANSI.

**Base:** ANSI stripped, all four lines preserved verbatim.

**Fine-tuned:** ANSI stripped, but dropped `error[E0308]: mismatched types`. Real lossless violation.

### 3. Spinner + completion

**Input:** three `\r`-overwritten spinner frames followed by `✓ Compiled in 3.4s`.

**Base:** over-collapses to just `Compiled in 3.4s`, drops the "Compiling project..." context.

**Fine-tuned:** keeps both `Compiling project...` and `✓ Compiled in 3.4s` on separate lines.

### What this shows

The fine-tuned model shifted the output distribution (formatting, how aggressively to collapse) but at 500 steps it hasn't learned copy-with-edits. Loss floor was ~1.5 (knows the type of output), well above the ~0.3-0.5 you'd want for faithful copying. Consistent with seeing only ~22K of ~352K training records (0.045 epochs).

## How it works

```
                       ┌─────────────────────────┐
clean source text ──►  │ corpus/   (synthetic)   │
                       │ ls, tree, diff, log,    │
                       │ code, JSON, …            │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │ dirtifier/ transforms   │
                       │ ANSI, \r-overwrite,     │
                       │ repeat, pad, locale, …  │
                       └────────────┬────────────┘
                                    │  (clean, dirty) pairs
                       ┌────────────▼────────────┐
                       │ data/train.jsonl  (60M+ tok) │
                       │ data/eval_real.jsonl  (real) │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │ train/cloud_train.py    │
                       │ Modal H100 + Unsloth    │
                       │ LoRA r=16 on q,k,v,o    │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │ merge → MLX 4-bit local │
                       │ infer/clean.py + guard  │
                       └─────────────────────────┘
```

Design choices:

- **Synthetic-first data.** Hand-writing 60M tokens of clean terminal output is insane. Programmatically rendering tables/trees/diffs/etc. and applying randomized ANSI/whitespace/CR transforms is cheap and gives unlimited paired data.
- **Lossless guard.** Every prediction goes through an atom-set check (`eval/lossless_guard.py`). If any non-noise token from the input is missing from the output, fall back to deterministic ANSI stripping. Model can't silently lose data.
- **Cloud LoRA, local inference.** Training on a Modal H100 (~$3/hr), merged model quantized to MLX 4-bit and runs on the user's Mac.
- **Channel-marker prompt.** Gemma 4's chat template defaults to the reasoning channel. Prefilling `<|channel>final\n` skips the chain-of-thought preamble and emits cleaned output directly.

## Repo layout

| Dir | What |
|---|---|
| `corpus/` | Synthetic clean-text generators (tables, trees, diffs, logs, code, JSON) |
| `dirtifier/` | Composable noise-injection transforms + recipe DSL |
| `train/` | Modal-based cloud training (`cloud_train.py`) + dataset formatter |
| `eval/` | Lossless guard, ANSI strip, metrics, slice reports |
| `infer/` | `clean(dirty: str) -> str` with guard fallback |
| `docs/superpowers/specs/` | Full design doc |

The MLX 4-bit weights (~2.4 GB) and 60M-token training set aren't committed. Recipe below regenerates both deterministically.

## Reproduce

```bash
uv sync --extra dev
uv pip install -e .[cloud]   # adds modal

# 1) Generate ~60M tokens of synthetic training data
uv run python -m corpus.generate_corpus
uv run python -m dirtifier.generate

# 2) Cloud train on H100 (requires Modal account + HF token)
modal secret create huggingface-token HF_TOKEN=hf_...
modal run train/cloud_train.py::main

# 3) Pull the merged HF model + convert to MLX 4-bit locally
modal volume get gemma-4-terminal-cleaner-data /output/merged_hf models/cloud_merged_hf/
uv run python -m mlx_lm.convert \
    --hf-path models/cloud_merged_hf/merged_hf \
    --mlx-path models/trained \
    -q

# 4) Evaluate
uv run python -m eval.run --base models/trained --adapter ""

# 5) Use it
echo $'\x1b[31mhello\x1b[0m' | uv run python -m infer.clean
```

## Cost

| Phase | Cost (this run) |
|---|---|
| Corpus + dirtifier (local) | $0 |
| Modal H100, 500 steps + merge | ~$5 |
| MLX convert (local) | $0 |

A real V1 (5–10K steps) lands around **$50–80** end-to-end.

## What I'd do differently

1. **Train longer.** 500 steps was bypass-the-merge-bug-and-validate-pipeline, not ship-a-model. Loss curve was still descending normally.
2. **Mix in real data earlier.** Synthetic data has tells (sterile timestamps, no rare commands). A few hundred hand-curated real (dirty, clean) pairs would help generalization.
3. **Smaller initial corpus, faster iteration.** 60M tokens is overkill for a PoC. 5M tokens × 5K steps would have produced a more useful model and let me iterate.
4. **Two-stage pipeline.** Deterministic ANSI strip + repetition collapse first, model only on the residual ambiguous cases. Base Gemma 4 E2B already handles 80% of the easy stuff; the model should specialize on the hard stuff.

## License

Code: MIT. Model adapter: derived from `google/gemma-4-E2B-it`, see Gemma license at <https://ai.google.dev/gemma/terms>.
