# Gemma 4 E2B Terminal Output Cleaner — Design Spec

**Date:** 2026-05-02
**Status:** Approved by user, ready for implementation planning
**Author:** Claude (brainstorming session)

---

## 1. Goal

Fine-tune **Gemma 4 E2B** to act as a **lossless terminal-output cleaner**. The model is a pure text-in / text-out function: given dirty terminal output (ANSI escapes, progress-bar overwrites, repeated lines, whitespace noise), it returns the same content with mechanical noise removed and **no information lost**.

The intended consumer is an agent harness that wants to reduce the token cost of shell-tool outputs without surrendering data. The model itself is *independent of the command line* — it never sees commands, only their output strings.

### Success in one line

A v1 model that, on a hand-curated real eval set, **strips terminal noise with ≥99.5% lossless-guard pass rate and ≥30% median token reduction on dirty inputs**, while leaving already-clean inputs essentially unchanged.

---

## 2. Constraints

| | |
|---|---|
| **Training hardware** | MacBook Air M4, 24 GB unified memory, 10-core (4P+6E), fanless |
| **Model** | Gemma 4 E2B — 2.3B effective / 5.1B with embeddings, 128K context, Apache 2.0 |
| **Training method** | 4-bit QLoRA via `mlx-lm` (only realistic path on this hardware) |
| **Dataset budget** | 100 MB – 1 GB of curated training pairs |
| **Cleanup style** | **Strict lossless** — no truncation, no semantic summarization |
| **Deployment** | Local Python function (CLI / import / optional HTTP); no shell hooks in v1 |

---

## 3. Non-Goals (explicit out-of-scope for v1)

- **Lossy compression / summarization.** No "47k files in node_modules → omitted" behavior. Every datum preserved.
- **Shell hook system (RTK-style PreToolUse rewriting).** Integration with agent shell-tool calls is downstream of this project.
- **Multimodal input/output.** Even though Gemma 4 E2B supports image/video/audio, we use text only.
- **Real-time / streaming inference.** Synchronous batch only.
- **Multi-tenant serving, auth, fine-tuning UI.** Single-user local tool.
- **Real command capture as the primary data source.** We use synthetic generation as the bulk strategy (see Section 5).

---

## 4. Architecture

The project decomposes into **four isolated units** plus eval/infer harnesses, communicating through file-on-disk artifacts.

```
gemma_4/
├── corpus/                 # (1) Clean source text — ground truth pool
│   ├── tables/             #     column tables (ls -l, ps, du, df, docker ps...)
│   ├── trees/              #     tree-shaped output (tree, find)
│   ├── lists/              #     flat path lists, env, history, key=val
│   ├── diffs/              #     unified-diff samples
│   ├── logs/               #     log lines, build output, test results
│   ├── code/               #     source file content (cat-like)
│   └── json/               #     JSON, YAML pretty-print
│
├── dirtifier/              # (2) Composable transforms: clean → dirty
│   ├── transforms/         #     ansi.py, progress.py, cr_overwrite.py,
│   │                       #     repeat.py, pad.py, box.py, locale.py, ...
│   ├── pipeline.py         #     random-stack composer
│   └── generate.py         #     CLI: corpus + recipe → jsonl pairs
│
├── data/                   # (3) Generated training data (jsonl, gitignored)
│   ├── train.jsonl         #     ~180k pairs
│   ├── val.jsonl           #     ~10k pairs
│   └── eval_real/          #     ~500 hand-curated real (dirty, clean) pairs
│
├── train/                  # (4) MLX-LM QLoRA training
│   ├── config.yaml         #     model, lora rank, lr, seq_len, batch
│   ├── prompt_template.py  #     wraps each pair into instruction format
│   └── run.sh              #     mlx_lm.lora ... entry point
│
├── eval/                   # Eval harness
│   ├── lossless_guard.py   #     verifies no info loss (atom-set check)
│   ├── metrics.py          #     token-reduction %, exact-match %, etc.
│   └── run.py              #     loads adapter, runs eval set, reports
│
└── infer/                  # Minimal text-in/text-out wrapper
    └── clean.py            #     load 4-bit base + LoRA adapter, generate
```

### Boundary rules

- `corpus/` is **read-only ground truth**. Never modified by training.
- `dirtifier/` is **pure functions**: `clean: str → dirty: str`. Reproducible from a seed.
- `data/` is **regenerated** from `corpus/` + `dirtifier/`. Cheap to rebuild.
- `train/` and `eval/` only consume jsonl from `data/`, never read `corpus/` directly.
- `corpus/` and `dirtifier/` are **independently versioned** — swap dirt distributions without touching the corpus, expand the corpus without rewriting transforms.

---

## 5. Data generation pipeline

The data strategy is **purely synthetic with a small real eval set**: we generate `(dirty, clean)` pairs by starting from clean text and applying randomized "dirtifier" transforms. Losslessness is **provable by construction**.

### 5.1 Clean corpus

Sources (~150–250 MB total clean text):
- **Synthetic templates** — programmatically render tables, trees, diffs, logs, JSON, code listings from random structured data.
- **Real artifacts (small, license-clean)** — sampled from open-source repos for code/diffs, public log datasets (e.g., LogHub) for system logs.

Organized by **structural archetype** (table / tree / list / diff / log / code / json), not by command — the model is command-agnostic.

### 5.2 The dirtifier

Each transform implements a uniform interface:

```python
class Transform:
    def apply(self, clean: str, rng: Random) -> str: ...
    def applicability(self, clean: str) -> float: ...  # 0..1
```

**v1 transform set (the dirt taxonomy):**

| Transform | What it does | Real-world source |
|---|---|---|
| `AnsiColor` | wrap tokens/lines in random CSI color codes | every `--color=auto` tool |
| `AnsiBold` / `AnsiUnderline` | SGR styling | bash builtins, less |
| `AnsiOsc` | OSC sequences (terminal title etc.) | many TUIs |
| `ProgressBar` | series of `\r`-overwriting bar lines, ending on the original line | npm, pip, cargo, wget |
| `Spinner` | `\r`-overwriting spinner frames | docker, pnpm |
| `CursorMovement` | cursor up / clear-line escape codes | top, htop |
| `BoxDrawing` | wrap content in Unicode box chars (├ ─ │ └) | tree, some tables |
| `RepeatedLines` | duplicate identical lines N times (N=2..50) | warning floods |
| `WhitespacePadding` | pad columns with extra spaces / tabs | column-aligned tools |
| `TrailingWhitespace` | trailing spaces, blank lines | many tools |
| `WindowsLineEndings` | mixed `\r\n` and `\n` | cross-platform tools |
| `LocaleVariants` | thousand separators, date formats | LANG-affected tools |
| `Timestamps` | ISO timestamp prefix per line | log lines |
| `MixedStreams` | interleave fake stderr lines into stdout | curl -v, build tools |
| `BellChars` / `NulBytes` | random `\a`, `\0` | edge cases |
| `Hyperlink` | OSC-8 hyperlink wrapping | modern tools |

~15–20 transforms for v1. Easy to add more.

### 5.3 Recipe-based composition

Rather than purely random stacking, **named recipes** mirror real dirt distributions:

| Recipe | Composition |
|---|---|
| `cli_colored_table` | `AnsiColor + WhitespacePadding + maybe TrailingWhitespace` |
| `install_with_progress` | `ProgressBar + Spinner + AnsiColor + AnsiBold + RepeatedLines(warnings)` |
| `tui_redraw` | `CursorMovement + AnsiColor + BoxDrawing` |
| `noisy_logs` | `Timestamps + RepeatedLines + AnsiColor` |
| `passthrough` | no transforms (model must learn to leave clean text alone) |

Each clean sample is dirtied 1–3 times under different recipes.

### 5.4 Output format

```json
{"input": "<dirty>", "output": "<clean>", "meta": {"recipe": "...", "transforms": [...], "src": "..."}}
```

`meta` is for debugging/eval slicing only — not fed to the model.

### 5.5 Sequence length policy

- ~90% of pairs target **<2k tokens combined** (input+output).
- Long-tail **5–10%** up to **8k tokens** so the model handles big outputs.
- Anything longer is rare; v1 caps at 8k.

### 5.6 Generation volumes

| Set | Pairs | Size | Notes |
|---|---|---|---|
| `train.jsonl` | ~180k | ~400 MB | |
| `val.jsonl` | ~10k | ~25 MB | held-out clean sources + recipes excluded from training |
| `eval_real/` | ~500 | ~5 MB | hand-curated real captures |
| `eval_passthrough.jsonl` | ~200 | ~1 MB | already-clean inputs; output should equal input |
| **Total** | | **~430 MB** | within v1 budget |

---

## 6. Training pipeline

### 6.1 Framework

`mlx-lm` (Apple Silicon native, Metal-accelerated). Hugging Face PEFT/transformers via MPS is ~3–5× slower for this exact case. Unsloth doesn't yet target Apple Silicon for training.

### 6.2 Model + quantization

- **Base**: `mlx-community/gemma-4-E2B-it-4bit` (community 4-bit MLX checkpoint)
- **Adapter**: LoRA in bf16

### 6.3 LoRA config (starting point — tune live)

| Knob | Value | Reasoning |
|---|---|---|
| `lora_layers` | 16 (last 16) | Enough for narrow style transfer |
| `lora_rank` | 16 | Sweet spot for narrow tasks |
| `lora_alpha` | 32 | 2× rank, standard |
| `lora_dropout` | 0.05 | Light regularization |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj` | Attention only for v1 |

Trainable parameters: ~6–10 M.

### 6.4 Prompt template (instruction format)

```
<start_of_turn>user
Clean the following terminal output. Preserve all information losslessly.
Strip ANSI codes, collapse progress-bar overwrites to their final state,
deduplicate identical repeated lines using [Nx] prefix, normalize whitespace.

---
<DIRTY_INPUT>
---<end_of_turn>
<start_of_turn>model
<CLEAN_OUTPUT><end_of_turn>
```

Loss computed only on model-turn tokens (standard SFT masking).

### 6.5 Hyperparameters (initial)

| Knob | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-4 (warmup-then-cosine-decay) |
| Warmup steps | 100 |
| Batch size | 1 |
| Gradient accumulation | 8 (effective batch = 8) |
| Max seq length (train) | 4096 |
| Epochs | 2–3 |
| Validation cadence | every 250 steps |
| Checkpoint cadence | every 500 steps; keep best-3 by val loss |

### 6.6 Estimated runtime on M4 24 GB

- ~180k pairs × 2 epochs / effective batch 8 = ~45k steps
- Estimated **0.6–1.2 sec/step** on M4 → **~16–30 hours total** for 2 epochs
- Run as overnight sessions with frequent checkpointing (thermal/sleep tolerance)

### 6.7 Memory budget (24 GB ceiling)

| Item | ~MB |
|---|---|
| 4-bit Gemma 4 E2B weights | ~1,800 |
| Activations @ seq 4096, batch 1 | ~2,500 |
| LoRA params + AdamW state | ~120 |
| Gradient accumulation buffer | ~1,000 |
| MLX/Metal runtime overhead | ~1,500 |
| **Total** | **~7 GB** |

Comfortable headroom for OS + browser etc.

### 6.8 Failure-mode plan

If training is too slow / unstable:
1. Drop `seq_len` to 2048 → halves per-step time
2. Drop `lora_layers` to 8 → less memory + faster
3. Skip `o_proj` from target modules → smaller adapter

If quality is poor:
1. Bump `lora_rank` to 32, then add MLP modules
2. Add 3rd epoch
3. Inspect failures, add missing dirtifier patterns, regenerate data

---

## 7. Evaluation harness

The eval is **load-bearing**. With strict-lossless cleanup, "looks clean" isn't enough — we must hard-guarantee no silent data loss.

### 7.1 Layer 1 — Lossless guard (deterministic, must pass)

Before any quality metric, every model output is checked against a deterministic information-preservation rule.

The guard extracts **information atoms** from both the model's input (after deterministic ANSI-strip) and output, then checks the output's atom-set is a superset of the input's.

**Atoms include:**
- File paths (regex `[\w./~-]+\.[\w]+` plus directory-looking tokens)
- All numbers (ints, floats, sizes like `1.2K`, `4.5MB`, durations)
- All identifiers ≥ 3 chars (function names, package names, error codes)
- All quoted strings
- All URLs / IPs / emails

If `atoms(input) ⊄ atoms(output)`, the example **fails**. No partial credit.

A configurable **whitelist of removable atoms** covers dirtifier-injected content (run-length-encoded repeated lines, ANSI codes, progress-bar percentages) — these are *expected* to disappear, identifiable because we own the dirtifier.

### 7.2 Layer 2 — Quality metrics (graded, reported)

Per example, after passing Layer 1:
- **`exact_match`** (binary) — output equals ground-truth byte-for-byte
- **`normalized_exact_match`** (binary) — equal after whitespace-run collapse + trailing-whitespace strip per line
- **`token_reduction_pct`** = `1 - len_tokens(output) / len_tokens(input)`
- **`char_diff`** — Levenshtein ratio (diagnostic only)

Aggregated as p50 / p95 across the eval set.

### 7.3 Layer 3 — Slice analysis

Eval set sliced by:
- Recipe (`install_with_progress`, `tui_redraw`, `passthrough`, ...)
- Input length bucket (`<512`, `512-2k`, `2k-8k`)
- Source archetype (table, tree, list, diff, log, code, json)
- Real vs synthetic

Per-slice metrics localize where to add training data.

### 7.4 Eval sets

**`val.jsonl`** — 10k synthetic, held-out; tests recipe-level generalization (some recipes excluded from training).

**`eval_real.jsonl`** — 500 hand-curated pairs from real captured outputs:
- 50 each from: `ls -la --color`, `find /usr | head -1000`, `git diff --color`, `pytest -v`, `npm install`, `cargo build`, `docker ps`, `kubectl get pods`, `cat <random_source_file>`, `tree`
- Captured by running commands in real shells; labels hand-cleaned with cross-check.
- The **only** dataset that touches real command output during training/eval — keeps the synthetic-vs-real distribution gap visible.

**`eval_passthrough.jsonl`** — 200 pairs with already-clean input; output must equal input. Stress-tests the model against over-correcting on clean input. Critical: a model that "always cleans something" is worse than no model.

### 7.5 v1 acceptance criteria

| Metric | Threshold |
|---|---|
| Layer 1 lossless guard pass rate (`eval_real`) | **≥ 99.5%** |
| Layer 1 lossless guard pass rate (`eval_passthrough`) | **100%** |
| `normalized_exact_match` on `eval_real` | **≥ 80%** |
| Median `token_reduction_pct` on `eval_real` (dirty inputs) | **≥ 30%** |
| `token_reduction_pct` on `eval_passthrough` | **0% ± 2%** |

If we fall short, slice analysis indicates which recipe / archetype to strengthen. Regenerate, retrain.

---

## 8. Inference & deployment

### 8.1 The deliverable surface

```python
def clean(dirty: str) -> str:
    """
    Lossless terminal-output cleaner.
    Returns either the model's cleaned output (if it passes the lossless guard)
    or a deterministic ANSI-stripped version of the input (safe fallback).
    """
```

### 8.2 Behavior

1. Load 4-bit base + LoRA adapter once, cache in process.
2. Apply deterministic ANSI-strip + `\r`-collapse to input as a pre-step (mechanical, risk-free).
3. Run the pre-stripped input through the model with the training prompt template.
4. Run the **lossless guard** on `(input → output)`.
5. If guard passes → return model output.
6. If guard fails → log failure case + return the pre-stripped (deterministic) version.

This makes the system **strictly safe**: at worst, the user sees the dirty (well, ANSI-stripped) original. Never silently truncated data.

### 8.3 Three usage modes (all built on `clean()`)

| Mode | Use case | Implementation |
|---|---|---|
| **CLI** | `cat dirty.log \| python -m infer.clean` | `infer/__main__.py` reads stdin, writes stdout |
| **Python import** | `from infer import clean` | Standard Python API |
| **Local HTTP (optional)** | persistent process, agents POST raw output | `infer/server.py` ~30 LOC FastAPI route |

The HTTP server matters only to amortize the 3–5 sec model-load time.

### 8.4 Performance expectations on M4 24 GB

- Model load: ~3–5 sec (one-time per process)
- Generation: ~25–50 tok/sec
- 2k-token output: ~40–80 sec
- 500-token output: ~10–20 sec

**This is not real-time inline cleaning.** It's batch / async cleanup. If real-time is needed later, the levers are: smaller distilled model, draft-model speculative decoding, or faster hardware.

---

## 9. Risks & open questions

| Risk | Mitigation |
|---|---|
| Synthetic→real distribution gap | 500-pair real eval set, slice analysis, expand recipes to close gaps |
| Model silently drops data | Lossless guard at eval AND inference; safe deterministic fallback |
| MacBook Air thermal throttling on multi-day runs | Frequent checkpointing; tolerate 0.5–2× variance in step time |
| Model over-cleans `passthrough` inputs | Dedicated `passthrough` recipe in training; `eval_passthrough.jsonl` with 100% threshold |
| Training instability at lr 2e-4 | Conservative warmup; drop to 1e-4 if val loss diverges |
| MLX-LM API churn | Pin `mlx-lm` version in `requirements.txt` |

### Deferred decisions (revisit at v2)

- Whether to add a streaming inference mode
- Whether to distill into a smaller model for real-time use
- Whether to add a shell-hook integration (RTK-style)
- Whether to retrain on multi-recipe stacking beyond depth 3

---

## 10. Acceptance for handoff to implementation planning

This spec is approved when all of the following are true:
- [x] Architecture and unit boundaries agreed
- [x] Data generation strategy (synthetic + real eval) agreed
- [x] Training framework (MLX-LM 4-bit QLoRA) and starting hyperparameters agreed
- [x] Eval thresholds agreed (99.5% lossless guard, 30% median token reduction, 0% ± 2% on passthrough)
- [x] Inference behavior with safe fallback agreed
- [x] Non-goals explicitly enumerated
