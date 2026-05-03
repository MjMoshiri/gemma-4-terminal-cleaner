# Gemma 4 E2B Terminal Output Cleaner — PoC

A proof-of-concept for fine-tuning a **small (2.3B-effective-params) open model**
to do **lossless cleanup** of dirty terminal output — strip ANSI escapes,
collapse `\r` progress-bar overwrites, dedupe repeated lines, normalize
whitespace, etc. — without losing a single information atom.

The motivation: agent harnesses spend a lot of context on shell-tool output
that's mostly noise. A cheap on-device model that compresses noise without
summarizing data is a useful primitive. This repo answers a smaller question:
**can a small Gemma model, fine-tuned cheaply on synthetic data, learn this
function?**

## TL;DR

- ✅ End-to-end recipe works: synthetic corpus → 60M-token training set →
  Modal H100 LoRA fine-tune → MLX 4-bit conversion → local inference.
- ⚠️ This run was a **500-step PoC** (~$5 of GPU). The model learned the
  *genre* (output looks like clean terminal output) but didn't reach
  *faithful copy* yet — long inputs trigger repetition loops.
- 💡 Out-of-the-box Gemma 4 E2B is already a surprisingly strong baseline.
  The honest takeaway: **the pipeline is reproducible, the model is
  undertrained, and a 5–10K-step run (~$50) is the obvious next step.**

## Before / After

Three small samples, run on identical prompts. **Base** is `mlx-community/gemma-4-E2B-it-4bit`
out of the box. **Fine-tuned** is the 500-step LoRA-merged + MLX-quantized
checkpoint produced by this repo (regenerate via the steps below). Full inputs and outputs in
[`eval/reports/sample_before_after.json`](eval/reports/sample_before_after.json).

### 1. `pip install` progress bar (ANSI + `\r`-overwrites)

**Input** (raw bytes shown with escapes visible):

```
Collecting requests
  Downloading requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)\r
Downloading requests-2.31.0-py3-none-any.whl (62 kB)
   \x1b[91m━━━…━━━\x1b[0m \x1b[32m20.1/62.1 kB\x1b[0m \x1b[31m1.2 MB/s\x1b[0m eta \x1b[36m0:00:00\x1b[0m\r
   \x1b[91m━━━…━━━\x1b[0m \x1b[32m62.1/62.1 kB\x1b[0m \x1b[31m1.5 MB/s\x1b[0m eta \x1b[36m0:00:00\x1b[0m
Installing collected packages: requests
Successfully installed requests-2.31.0
```

**Base (Gemma 4 E2B, 4-bit, no fine-tune):** ✅ near-perfect — strips ANSI,
keeps both progress states, preserves the install lines.

**Fine-tuned (500 steps):** ❌ enters a repetition loop —
`Downloading requests-2.31.0-py3-none-any.whl (4.6 kB)` repeated until cutoff.
Classic undertraining symptom: model has learned what pip output looks like,
hasn't learned to mirror this *specific* input.

### 2. Build error with ANSI

**Input:** `ERROR: build failed` + `src/main.rs:42:5` + `error[E0308]: mismatched types` + `--> expected 'String', found '&str'`, all wrapped in red/yellow/cyan ANSI.

**Base:** ✅ ANSI stripped, all four lines preserved verbatim.

**Fine-tuned:** ⚠️ ANSI stripped, but **dropped** `error[E0308]: mismatched types`
— a real lossless violation.

### 3. Spinner + completion

**Input:** three `\r`-overwritten spinner frames followed by `✓ Compiled in 3.4s`.

**Base:** ⚠️ over-collapses to just `Compiled in 3.4s` — drops the
"Compiling project..." context.

**Fine-tuned:** ✅ keeps both `Compiling project...` and `✓ Compiled in 3.4s`
on separate lines.

### What this shows

The fine-tuned model **changed the output distribution** (formatting decisions,
how aggressively to collapse) but at 500 steps it hasn't yet learned the
**copy-with-edits** behavior the data is teaching. Loss floor was ~1.5 (model
"knows the type of output"), well above the ~0.3-0.5 you'd need for faithful
copying. This is consistent with seeing only ~22K of ~352K training records
(0.045 epochs).

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

Key design choices:

- **Synthetic-first data.** Writing 60M tokens of clean terminal output by
  hand is insane; programmatically rendering tables/trees/diffs/etc. and then
  applying randomized ANSI/whitespace/CR transforms is cheap and gives
  unlimited paired data.
- **Lossless guard.** Every prediction passes an atom-set check
  (`eval/lossless_guard.py`) — if any non-noise token from the input is
  missing from the output, we fall back to deterministic ANSI stripping.
  The model can never silently lose data.
- **Cloud LoRA, local inference.** Training runs on a Modal H100 (~$3/hr);
  the merged model is quantized to MLX 4-bit and runs on the user's Mac.
- **Channel-marker prompt.** Gemma 4's chat template defaults to the
  reasoning channel. We prefill `<|channel>final\n` to skip the chain-of-thought
  preamble and emit the cleaned output directly.

## Repo layout

| Dir | What |
|---|---|
| `corpus/` | Synthetic clean-text generators (tables, trees, diffs, logs, code, JSON) |
| `dirtifier/` | Composable noise-injection transforms + recipe DSL |
| `train/` | Modal-based cloud training (`cloud_train.py`) + dataset formatter |
| `eval/` | Lossless guard, ANSI strip, metrics, slice reports |
| `infer/` | `clean(dirty: str) -> str` with guard fallback |
| `docs/superpowers/specs/` | Full design doc |

The trained MLX 4-bit weights (~2.4 GB) and 60M-token training set are not
committed — the recipe below regenerates both deterministically.

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

A real V1 (5–10K steps) would land around **$50–80** end-to-end.

## What I'd do differently

1. **Train longer.** 500 steps was bypass-the-merge-bug-and-validate-pipeline,
   not "ship a model." Loss curve was still descending normally.
2. **Mix in real data earlier.** Synthetic data has tells (sterile timestamps,
   no rare commands). A few hundred hand-curated real (dirty, clean) pairs
   would help the model generalize.
3. **Smaller initial corpus, faster iteration.** 60M tokens is overkill for a
   PoC. 5M tokens × 5K steps would have produced a more useful model and let
   me iterate.
4. **Two-stage pipeline.** Deterministic ANSI strip + repetition collapse
   first, model only on the residual ambiguous cases. The base Gemma 4 E2B
   is already 80% of the way there on the easy stuff; the model should
   specialize on the hard stuff.

## License

Code: MIT. Model adapter: derived from `google/gemma-4-E2B-it`, see Gemma
license at <https://ai.google.dev/gemma/terms>.
