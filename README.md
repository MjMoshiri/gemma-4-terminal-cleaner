# gemma-4-terminal-cleaner

PoC: fine-tune Gemma 4 E2B (2.3B) to clean dirty terminal output without losing data — strip ANSI, collapse `\r`-overwrites, dedupe repeats. The idea was a cheap on-device model that compresses agent shell-tool output without summarizing.

It didn't really work. 500 steps on a Modal H100 (~$5), pipeline runs end-to-end, but the model learned the shape of clean terminal output without learning to mirror the actual input. Long pip-install logs trigger repetition loops. Base Gemma 4 E2B out of the box is already fine for most easy cases.

Sample (one of three in `eval/reports/sample_before_after.json`):

```
input:  Collecting requests
          Downloading requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)\r
        Downloading requests-2.31.0-py3-none-any.whl (62 kB)
           \x1b[91m━━━…\x1b[0m \x1b[32m20.1/62.1 kB\x1b[0m ... eta \x1b[36m0:00:00\x1b[0m\r
           \x1b[91m━━━…\x1b[0m \x1b[32m62.1/62.1 kB\x1b[0m ... eta \x1b[36m0:00:00\x1b[0m
        Successfully installed requests-2.31.0

base:   ANSI stripped, both progress states kept, install line preserved. fine.
ft:     repetition loop — "Downloading requests-2.31.0-py3-none-any.whl (4.6 kB)" until cutoff.
```

Loss floor was ~1.5 after 500 steps, ~0.045 epochs. Should have just paid for 5–10K steps (~$50).

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

Design doc with the full architecture is in `docs/superpowers/specs/`.
