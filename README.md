# gemma-4-terminal-cleaner

My first fine-tuning attempt. Took Gemma 4 E2B (2.3B params) and trained it on synthetic dirty/clean terminal-output pairs to act as a lossless cleaner — strip ANSI escapes, collapse `\r` progress-bar overwrites, drop repeated lines, normalize whitespace, without dropping any actual data.

The motivation was making agent shell-tool output cheaper to keep in context. A small on-device model that compresses noise but never summarizes data felt like a useful primitive to try.

## What the model picked up

After 500 steps on a Modal H100 (~$5), the LoRA already shifted real behavior. The model:

- Skips Gemma 4's chain-of-thought preamble and emits cleaned output directly, after a `<|channel>final\n` prefill.
- Strips ANSI escapes consistently across all samples I tried.
- Picks up on terminal-output structure — outputs look like `ls`, `pip`, build logs, etc., not freeform prose.
- Handles short inputs cleanly. On a spinner-then-done sample, the base model over-collapses to just the final line; the fine-tuned model keeps both the work-in-progress line and the completion:

```
input:  ⠋ Compiling project...\r⠙ Compiling project...\r⠹ Compiling project...\r✓ Compiled in 3.4s
base:   Compiled in 3.4s
ft:     Compiling project...
        ✓ Compiled in 3.4s
```

So at 500 steps the model has clearly learned the *genre* of clean terminal output. The next step is teaching it faithful copy on longer inputs — at this step count, long pip-install logs still trigger a repetition loop. Loss was ~1.5 and still descending. 5–10K steps (~$50) is the next run.

Three before/after samples in `eval/reports/sample_before_after.json`.

## Repro

Modal account + HF token required.

```bash
uv sync --extra dev
uv pip install -e .[cloud]
uv run python -m corpus.generate_corpus
uv run python -m dirtifier.generate
modal secret create huggingface-token HF_TOKEN=hf_...
modal run train/cloud_train.py::main
modal volume get gemma-4-terminal-cleaner-data /output/merged_hf models/cloud_merged_hf/
uv run python -m mlx_lm.convert --hf-path models/cloud_merged_hf/merged_hf --mlx-path models/trained -q
echo $'\x1b[31mhello\x1b[0m' | uv run python -m infer.clean
```

`corpus/` generates clean source text, `dirtifier/` adds noise transforms (ANSI, CR-overwrite, repeats, padding, locale), `train/` runs the LoRA on Modal, `eval/lossless_guard.py` checks no atoms get lost, `infer/clean.py` runs inference and falls back to a deterministic ANSI strip if the model output drops anything.
