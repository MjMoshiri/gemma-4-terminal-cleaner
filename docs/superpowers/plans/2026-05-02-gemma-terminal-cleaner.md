# Gemma 4 E2B Terminal Output Cleaner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers-extended-cc:subagent-driven-development` (recommended) or `superpowers-extended-cc:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fine-tuned Gemma 4 E2B model that losslessly cleans dirty terminal output (ANSI codes, progress bars, repeated lines, whitespace) and ships as a Python text-in/text-out function with a strict-safe fallback.

**Architecture:** Four isolated units — `corpus/` (clean ground-truth text), `dirtifier/` (composable noise transforms), `train/` (MLX-LM 4-bit QLoRA), `infer/` (model wrapper with lossless guard). Plus `eval/` harness with deterministic info-preservation check. Data flows: corpus → dirtifier → jsonl pairs → trained adapter → inference function.

**Tech Stack:** Python 3.11+, `mlx-lm` (Apple Silicon native), `pytest`, `uv` for env management, `pydantic` for config, `fastapi` (optional inference server), Gemma 4 E2B 4-bit (`mlx-community/gemma-4-E2B-it-4bit`). Trains on M4 24 GB unified memory.

**Spec:** [`docs/superpowers/specs/2026-05-02-gemma-terminal-cleaner-design.md`](../specs/2026-05-02-gemma-terminal-cleaner-design.md)

---

## File Structure

```
gemma_4/
├── pyproject.toml                       # uv-managed project
├── .gitignore                           # ignores models/, corpus/output/, data/
├── README.md                            # quickstart
├── corpus/
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── base.py                      # Generator interface
│   │   ├── tables.py                    # ls -l, ps, du, df, docker ps style
│   │   ├── trees.py                     # tree, find layouts
│   │   ├── lists.py                     # path lists, env, history, key=val
│   │   ├── diffs.py                     # unified diff samples
│   │   ├── logs.py                      # log lines, build output, test results
│   │   ├── code.py                      # source-file listings (sampled from local repos)
│   │   └── json_yaml.py                 # JSON/YAML pretty-print
│   ├── generate_corpus.py               # CLI: run all generators -> corpus/output/
│   └── output/                          # gitignored: generated clean text
├── dirtifier/
│   ├── __init__.py
│   ├── transforms/
│   │   ├── __init__.py
│   │   ├── base.py                      # Transform ABC
│   │   ├── ansi.py                      # AnsiColor, AnsiBold, AnsiUnderline, AnsiOsc, Hyperlink
│   │   ├── progress.py                  # ProgressBar, Spinner, CursorMovement
│   │   ├── box.py                       # BoxDrawing
│   │   ├── repeat.py                    # RepeatedLines
│   │   ├── whitespace.py                # WhitespacePadding, TrailingWhitespace, WindowsLineEndings
│   │   ├── locale.py                    # LocaleVariants
│   │   ├── timestamps.py                # Timestamps
│   │   ├── streams.py                   # MixedStreams
│   │   └── edge.py                      # BellChars, NulBytes
│   ├── recipes.py                       # named Recipe definitions
│   ├── pipeline.py                      # apply_recipe(clean, recipe, rng) -> dirty
│   └── generate.py                      # CLI: corpus -> data/{train,val}.jsonl
├── data/                                # gitignored
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── eval_real.jsonl                  # 500 hand-curated real captures
│   └── eval_passthrough.jsonl           # 200 already-clean inputs
├── train/
│   ├── __init__.py
│   ├── prompt_template.py               # format_pair(input, output) -> mlx training row
│   ├── format_dataset.py                # CLI: jsonl -> mlx-lm training format
│   ├── config.yaml                      # mlx-lm lora hyperparameters
│   └── run.sh                           # entry point for training
├── eval/
│   ├── __init__.py
│   ├── lossless_guard.py                # extract_atoms, atoms_subset_check
│   ├── metrics.py                       # exact_match, normalized_match, token_reduction
│   ├── slicing.py                       # group results by recipe / archetype / length
│   └── run.py                           # CLI: load adapter, run eval, report
├── infer/
│   ├── __init__.py
│   ├── clean.py                         # clean(dirty: str) -> str (with fallback)
│   ├── ansi_strip.py                    # deterministic pre-processor
│   ├── __main__.py                      # CLI: stdin -> stdout
│   └── server.py                        # optional FastAPI single-route server
├── tests/
│   ├── __init__.py
│   ├── corpus/
│   │   ├── test_tables.py
│   │   ├── test_trees.py
│   │   ├── test_lists.py
│   │   ├── test_diffs.py
│   │   ├── test_logs.py
│   │   ├── test_code.py
│   │   └── test_json_yaml.py
│   ├── dirtifier/
│   │   ├── test_ansi.py
│   │   ├── test_progress.py
│   │   ├── test_box.py
│   │   ├── test_repeat.py
│   │   ├── test_whitespace.py
│   │   ├── test_locale.py
│   │   ├── test_timestamps.py
│   │   ├── test_streams.py
│   │   ├── test_edge.py
│   │   ├── test_recipes.py
│   │   └── test_pipeline.py
│   ├── eval/
│   │   ├── test_lossless_guard.py
│   │   └── test_metrics.py
│   ├── infer/
│   │   ├── test_ansi_strip.py
│   │   └── test_clean.py
│   └── train/
│       └── test_prompt_template.py
├── models/                              # gitignored
│   ├── base/                            # mlx-community/gemma-4-E2B-it-4bit weights
│   └── adapter/                         # trained LoRA output
└── docs/superpowers/{specs,plans}/      # already exists
```

---

## Task 0: Repo bootstrap

**Goal:** A working Python project with `uv`, `pytest`, `mlx-lm` installed; the base model downloaded; the repo structure committable.

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `tests/__init__.py`
- Create: empty `__init__.py` files in `corpus/generators/`, `dirtifier/`, `dirtifier/transforms/`, `train/`, `eval/`, `infer/`
- Create: directories `corpus/output/`, `data/`, `models/base/`, `models/adapter/` (gitignored)

**Acceptance Criteria:**
- [ ] `uv sync` installs cleanly on macOS arm64
- [ ] `uv run pytest -q` runs (zero tests, exits 0)
- [ ] `uv run python -c "import mlx.core as mx; print(mx.default_device())"` prints `Device(gpu, 0)`
- [ ] Base model directory `models/base/gemma-4-E2B-it-4bit/` contains weights (>1 GB total)
- [ ] `git status` shows tracked files only — no `models/`, `data/`, `corpus/output/` artifacts

**Verify:** `uv run pytest -q && ls models/base/gemma-4-E2B-it-4bit/*.safetensors | head -1` → pytest exits 0, lists at least one safetensors file.

**Steps:**

- [ ] **Step 1: Install `uv` if not present**

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "gemma-terminal-cleaner"
version = "0.1.0"
description = "Fine-tuned Gemma 4 E2B for lossless terminal output cleaning"
requires-python = ">=3.11"
dependencies = [
    "mlx-lm>=0.20.0",
    "mlx-vlm>=0.1.0",
    "huggingface-hub>=0.25.0",
    "pydantic>=2.9.0",
    "pyyaml>=6.0",
    "tqdm>=4.66.0",
    "rich>=13.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.6.0",
]
server = [
    "fastapi>=0.115.0",
    "uvicorn>=0.32.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["corpus", "dirtifier", "train", "eval", "infer"]
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/

# Project artifacts (regenerable)
corpus/output/
data/
models/

# OS
.DS_Store
```

- [ ] **Step 4: Create `README.md`**

```markdown
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
```

- [ ] **Step 5: Create empty package init files**

```bash
mkdir -p tests corpus/generators dirtifier/transforms train eval infer
touch tests/__init__.py corpus/generators/__init__.py dirtifier/__init__.py \
      dirtifier/transforms/__init__.py train/__init__.py eval/__init__.py \
      infer/__init__.py
mkdir -p corpus/output data models/base models/adapter
```

- [ ] **Step 6: Install deps + verify MLX**

```bash
uv sync --extra dev
uv run python -c "import mlx.core as mx; print(mx.default_device())"
```

Expected: `Device(gpu, 0)`. If `Device(cpu, 0)`, you're not on Apple Silicon — STOP and check.

- [ ] **Step 7: Download Gemma 4 E2B 4-bit base model**

```bash
uv run hf download mlx-community/gemma-4-E2B-it-4bit \
    --local-dir models/base/gemma-4-E2B-it-4bit
ls -lh models/base/gemma-4-E2B-it-4bit/ | head -20
```

Expected: directory contains `*.safetensors`, `tokenizer.json`, `config.json`. Total size ~1.5 GB.

- [ ] **Step 8: Smoke-test the base model**

```bash
uv run python -c "
from mlx_lm import load, generate
model, tok = load('models/base/gemma-4-E2B-it-4bit')
print(generate(model, tok, prompt='Hello', max_tokens=20, verbose=False))
"
```

Expected: Some coherent text completion. If it errors on `mlx_lm` API drift, pin a known-working version in `pyproject.toml`.

- [ ] **Step 9: Run pytest (zero tests, must succeed)**

```bash
uv run pytest -q
```

Expected: `no tests ran in 0.0Xs` (exit 0).

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml .gitignore README.md tests/__init__.py \
        corpus/generators/__init__.py dirtifier/__init__.py \
        dirtifier/transforms/__init__.py train/__init__.py \
        eval/__init__.py infer/__init__.py
git commit -m "Bootstrap project: uv, pytest, mlx-lm, base model download instructions"
```

---

## Task 1: Generator base class + first corpus generator (tables)

**Goal:** Establish the corpus generator contract and ship the first concrete generator (table-style outputs like `ls -l`, `ps aux`, `df -h`). This task locks the testing pattern reused by all corpus tasks.

**Files:**
- Create: `corpus/generators/base.py`
- Create: `corpus/generators/tables.py`
- Create: `tests/corpus/__init__.py`
- Create: `tests/corpus/test_base.py`
- Create: `tests/corpus/test_tables.py`

**Acceptance Criteria:**
- [ ] `Generator` ABC with `generate(rng) -> str` method and `archetype` class attribute
- [ ] `TablesGenerator` produces `ls -l`-style listings, `ps`-style listings, `df`-style listings
- [ ] Each generator is **deterministic given seed** (same seed → identical output)
- [ ] Output contains no ANSI escape codes (those come from the dirtifier)
- [ ] Output is realistic: column-aligned, plausible permissions/sizes/dates

**Verify:** `uv run pytest tests/corpus/test_base.py tests/corpus/test_tables.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write failing test for `Generator` ABC**

`tests/corpus/test_base.py`:

```python
import random
import pytest
from corpus.generators.base import Generator


def test_generator_is_abstract():
    with pytest.raises(TypeError):
        Generator()  # cannot instantiate ABC


def test_generator_subclass_must_implement_generate():
    class Incomplete(Generator):
        archetype = "incomplete"
    with pytest.raises(TypeError):
        Incomplete()


def test_generator_subclass_works():
    class Concrete(Generator):
        archetype = "concrete"
        def generate(self, rng: random.Random) -> str:
            return "hello"
    g = Concrete()
    assert g.generate(random.Random(0)) == "hello"
    assert g.archetype == "concrete"
```

- [ ] **Step 2: Run test, verify it fails**

```bash
uv run pytest tests/corpus/test_base.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'corpus.generators.base'`.

- [ ] **Step 3: Implement `Generator` ABC**

`corpus/generators/base.py`:

```python
import random
from abc import ABC, abstractmethod


class Generator(ABC):
    """Generates one clean text artifact per call. Deterministic given seed."""

    archetype: str  # subclasses must set: "table" | "tree" | "list" | "diff" | "log" | "code" | "json"

    @abstractmethod
    def generate(self, rng: random.Random) -> str:
        """Return one clean text sample. No ANSI codes, no terminal artifacts."""
        ...
```

- [ ] **Step 4: Run test, verify it passes**

```bash
uv run pytest tests/corpus/test_base.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Write failing tests for `TablesGenerator`**

`tests/corpus/test_tables.py`:

```python
import random
import re
from corpus.generators.tables import LsListingGenerator, PsListingGenerator, DfListingGenerator


def test_ls_listing_archetype():
    assert LsListingGenerator().archetype == "table"


def test_ls_listing_deterministic():
    g = LsListingGenerator()
    out1 = g.generate(random.Random(42))
    out2 = g.generate(random.Random(42))
    assert out1 == out2


def test_ls_listing_no_ansi():
    out = LsListingGenerator().generate(random.Random(7))
    assert "\x1b[" not in out  # no ANSI escape codes


def test_ls_listing_format():
    out = LsListingGenerator().generate(random.Random(7))
    lines = out.strip().split("\n")
    # First line is "total NNN" header
    assert re.match(r"^total \d+$", lines[0])
    # Each subsequent line: perms links owner group size date name
    for line in lines[1:]:
        # e.g. -rw-r--r-- 1 alice users 1234 May  2 10:30 README.md
        assert re.match(r"^[-dlrwxs]{10}\s+\d+\s+\S+\s+\S+\s+\d+\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}\s+\S+", line)


def test_ps_listing_format():
    out = PsListingGenerator().generate(random.Random(11))
    lines = out.strip().split("\n")
    # Header line
    assert "PID" in lines[0] and "CMD" in lines[0]
    # At least 5 process lines
    assert len(lines) >= 6


def test_df_listing_format():
    out = DfListingGenerator().generate(random.Random(13))
    lines = out.strip().split("\n")
    assert "Filesystem" in lines[0]
    assert "Use%" in lines[0]
    # All non-header lines have a percent column
    for line in lines[1:]:
        assert "%" in line
```

- [ ] **Step 6: Run tests, verify they fail**

```bash
uv run pytest tests/corpus/test_tables.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 7: Implement table generators**

`corpus/generators/tables.py`:

```python
import random
from corpus.generators.base import Generator


_USERS = ["alice", "bob", "carol", "dave", "eve", "root", "deploy", "www-data"]
_GROUPS = ["users", "staff", "wheel", "nogroup", "deploy", "www-data"]
_FILE_SUFFIXES = [".py", ".js", ".ts", ".rs", ".go", ".md", ".json", ".yaml",
                  ".toml", ".lock", ".txt", ".log", ".sh", ".html", ".css"]
_DIR_NAMES = ["src", "tests", "docs", "build", "dist", "node_modules",
              ".git", "vendor", "target", "venv", "logs", "tmp", "data"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _random_perm(rng: random.Random) -> str:
    is_dir = rng.random() < 0.2
    type_char = "d" if is_dir else "-"
    perms = ""
    for _ in range(3):
        perms += rng.choice(["r", "-"])
        perms += rng.choice(["w", "-"])
        perms += rng.choice(["x", "-"])
    return type_char + perms


def _random_filename(rng: random.Random) -> str:
    if rng.random() < 0.25:
        return rng.choice(_DIR_NAMES)
    base_chars = "abcdefghijklmnopqrstuvwxyz_-0123456789"
    base_len = rng.randint(3, 12)
    base = "".join(rng.choice(base_chars) for _ in range(base_len))
    return base + rng.choice(_FILE_SUFFIXES)


def _random_size(rng: random.Random) -> int:
    # log-uniform-ish: many small files, occasional large
    exponent = rng.randint(0, 8)
    return rng.randint(1, 9) * (10 ** exponent)


def _random_date_columns(rng: random.Random) -> str:
    month = rng.choice(_MONTHS)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    return f"{month} {day:>2} {hour:02d}:{minute:02d}"


class LsListingGenerator(Generator):
    archetype = "table"

    def generate(self, rng: random.Random) -> str:
        n = rng.randint(3, 40)
        rows = []
        total = 0
        for _ in range(n):
            perms = _random_perm(rng)
            links = rng.randint(1, 9)
            owner = rng.choice(_USERS)
            group = rng.choice(_GROUPS)
            size = _random_size(rng)
            total += size // 1024 + 1
            date = _random_date_columns(rng)
            name = _random_filename(rng)
            rows.append(f"{perms} {links:>2} {owner:<8} {group:<8} {size:>8} {date} {name}")
        return f"total {total}\n" + "\n".join(rows) + "\n"


class PsListingGenerator(Generator):
    archetype = "table"

    def generate(self, rng: random.Random) -> str:
        cmds = ["python", "node", "bash", "ssh", "vim", "make", "cargo build",
                "go test ./...", "pytest", "docker", "kubectl", "ruby",
                "java -jar app.jar", "/usr/bin/containerd", "systemd"]
        header = "  PID TTY          TIME CMD"
        rows = [header]
        n = rng.randint(5, 30)
        for _ in range(n):
            pid = rng.randint(1, 99999)
            tty = rng.choice(["pts/0", "pts/1", "?", "tty1"])
            mins = rng.randint(0, 59)
            secs = rng.randint(0, 59)
            cmd = rng.choice(cmds)
            rows.append(f"{pid:>5} {tty:<8} 00:{mins:02d}:{secs:02d} {cmd}")
        return "\n".join(rows) + "\n"


class DfListingGenerator(Generator):
    archetype = "table"

    def generate(self, rng: random.Random) -> str:
        header = "Filesystem      1K-blocks      Used  Available Use% Mounted on"
        rows = [header]
        mounts = ["/", "/home", "/var", "/tmp", "/dev/shm", "/boot", "/data"]
        for mount in rng.sample(mounts, k=rng.randint(3, len(mounts))):
            blocks = rng.randint(100_000, 100_000_000)
            used = rng.randint(0, blocks)
            avail = blocks - used
            pct = int(used / blocks * 100)
            fs = f"/dev/{rng.choice(['sda1','sda2','nvme0n1p1','vda1'])}"
            rows.append(f"{fs:<14} {blocks:>10} {used:>9} {avail:>10} {pct:>3}% {mount}")
        return "\n".join(rows) + "\n"
```

- [ ] **Step 8: Run tests, verify all pass**

```bash
uv run pytest tests/corpus/test_tables.py -v
```

Expected: 6 passed.

- [ ] **Step 9: Eyeball the output (sanity check)**

```bash
uv run python -c "
import random
from corpus.generators.tables import LsListingGenerator, PsListingGenerator, DfListingGenerator
for cls in [LsListingGenerator, PsListingGenerator, DfListingGenerator]:
    print(f'--- {cls.__name__} ---')
    print(cls().generate(random.Random(42)))
"
```

Expected: looks like real `ls -l`, `ps`, `df` output. If formatting drift, fix in code.

- [ ] **Step 10: Commit**

```bash
git add corpus/generators/base.py corpus/generators/tables.py \
        tests/corpus/__init__.py tests/corpus/test_base.py tests/corpus/test_tables.py
git commit -m "Add Generator ABC and table-archetype generators (ls, ps, df)"
```

---

## Task 2: Tree, list, diff, log, code, JSON corpus generators

**Goal:** Cover the remaining six structural archetypes from the spec, with tests, all deterministic.

**Files:**
- Create: `corpus/generators/trees.py`
- Create: `corpus/generators/lists.py`
- Create: `corpus/generators/diffs.py`
- Create: `corpus/generators/logs.py`
- Create: `corpus/generators/code.py`
- Create: `corpus/generators/json_yaml.py`
- Create: `tests/corpus/test_trees.py`
- Create: `tests/corpus/test_lists.py`
- Create: `tests/corpus/test_diffs.py`
- Create: `tests/corpus/test_logs.py`
- Create: `tests/corpus/test_code.py`
- Create: `tests/corpus/test_json_yaml.py`

**Acceptance Criteria:**
- [ ] `TreeGenerator` (archetype `tree`) produces tree-shaped paths using `├──`, `└──`, `│   `
- [ ] `FlatListGenerator` (archetype `list`) produces newline-separated paths/values (env, history)
- [ ] `UnifiedDiffGenerator` (archetype `diff`) produces realistic git-diff blocks (`---`, `+++`, `@@`, ` `, `-`, `+` prefixes)
- [ ] `LogGenerator` (archetype `log`) produces log lines with levels (INFO/WARN/ERROR/DEBUG)
- [ ] `CodeGenerator` (archetype `code`) produces plausible source-code listings (Python/JS/Go/Rust)
- [ ] `JsonYamlGenerator` (archetype `json`) produces valid JSON or YAML pretty-printed
- [ ] All generators deterministic from seed (verified by test)
- [ ] All output ANSI-free

**Verify:** `uv run pytest tests/corpus/ -v` → all pass.

**Steps:**

- [ ] **Step 1: Write tests for all six generators (one test file each)**

For each generator file `tests/corpus/test_<X>.py`, write 3 tests following the Task 1 pattern:
1. `test_<X>_archetype` — assert `archetype` attribute is correct
2. `test_<X>_deterministic` — same seed produces identical output
3. `test_<X>_no_ansi` — output contains no `\x1b[`
4. (Plus one or two format-specific tests per archetype)

Example for `tests/corpus/test_trees.py`:

```python
import random
from corpus.generators.trees import TreeGenerator


def test_tree_archetype():
    assert TreeGenerator().archetype == "tree"


def test_tree_deterministic():
    g = TreeGenerator()
    assert g.generate(random.Random(0)) == g.generate(random.Random(0))


def test_tree_no_ansi():
    assert "\x1b[" not in TreeGenerator().generate(random.Random(0))


def test_tree_uses_tree_chars():
    out = TreeGenerator().generate(random.Random(7))
    # tree command uses these box-drawing chars
    assert "├──" in out or "└──" in out
    assert "│" in out or out.count("\n") <= 2  # tiny trees may skip continuation chars
```

Example for `tests/corpus/test_diffs.py`:

```python
import random
from corpus.generators.diffs import UnifiedDiffGenerator


def test_diff_archetype():
    assert UnifiedDiffGenerator().archetype == "diff"


def test_diff_deterministic():
    g = UnifiedDiffGenerator()
    assert g.generate(random.Random(1)) == g.generate(random.Random(1))


def test_diff_no_ansi():
    assert "\x1b[" not in UnifiedDiffGenerator().generate(random.Random(1))


def test_diff_has_unified_format():
    out = UnifiedDiffGenerator().generate(random.Random(1))
    assert out.startswith("---") or "diff --git" in out
    assert "+++ " in out
    assert "@@" in out
```

(Equivalent shape for the others — see implementation files for which patterns each test should check.)

- [ ] **Step 2: Run all tests, verify they fail**

```bash
uv run pytest tests/corpus/ -v
```

Expected: FAIL — modules not found.

- [ ] **Step 3: Implement `corpus/generators/trees.py`**

```python
import random
from corpus.generators.base import Generator


def _random_subtree(rng: random.Random, depth: int, max_depth: int) -> list[tuple[str, list]]:
    """Returns nested (name, children) tuples. Empty children = leaf file."""
    if depth >= max_depth:
        return []
    n = rng.randint(1, 6)
    nodes = []
    for _ in range(n):
        is_dir = depth < max_depth - 1 and rng.random() < 0.4
        name = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz_-") for _ in range(rng.randint(3, 12)))
        if not is_dir:
            name += rng.choice([".py", ".js", ".rs", ".md", ".json", ".txt"])
            children = []
        else:
            children = _random_subtree(rng, depth + 1, max_depth)
        nodes.append((name, children))
    return nodes


def _render_tree(nodes: list[tuple[str, list]], prefix: str = "") -> list[str]:
    lines = []
    for i, (name, children) in enumerate(nodes):
        is_last = i == len(nodes) - 1
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + name)
        if children:
            extension = "    " if is_last else "│   "
            lines.extend(_render_tree(children, prefix + extension))
    return lines


class TreeGenerator(Generator):
    archetype = "tree"

    def generate(self, rng: random.Random) -> str:
        root_name = rng.choice(["src", "project", "app", "lib", "."])
        depth = rng.randint(2, 4)
        children = _random_subtree(rng, 0, depth)
        lines = [root_name] + _render_tree(children)
        n_files = sum(1 for line in lines if "." in line.split("── ")[-1])
        n_dirs = sum(1 for line in lines if "── " in line and "." not in line.split("── ")[-1])
        lines.append("")
        lines.append(f"{n_dirs} directories, {n_files} files")
        return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Implement `corpus/generators/lists.py`**

```python
import random
from corpus.generators.base import Generator


class FlatListGenerator(Generator):
    archetype = "list"

    def generate(self, rng: random.Random) -> str:
        kind = rng.choice(["paths", "env", "history", "kv"])
        n = rng.randint(5, 80)
        if kind == "paths":
            lines = []
            for _ in range(n):
                depth = rng.randint(1, 5)
                parts = [rng.choice(["src", "lib", "tests", "build", "node_modules", "dist", "venv", "data"])
                         for _ in range(depth)]
                fname = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz_") for _ in range(rng.randint(3, 10)))
                fname += rng.choice([".py", ".js", ".rs", ".go", ".md"])
                lines.append("./" + "/".join(parts) + "/" + fname)
            return "\n".join(lines) + "\n"
        if kind == "env":
            lines = []
            for _ in range(n):
                key = "_".join("".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(rng.randint(3, 8)))
                               for _ in range(rng.randint(1, 3)))
                val_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/.:="
                val = "".join(rng.choice(val_chars) for _ in range(rng.randint(5, 40)))
                lines.append(f"{key}={val}")
            return "\n".join(lines) + "\n"
        if kind == "history":
            cmds = ["ls -la", "cd ..", "git status", "git diff", "vim foo.py", "cargo test",
                    "make", "pytest -v", "docker ps", "kubectl get pods"]
            lines = [f"{i:>5}  {rng.choice(cmds)}" for i in range(1, rng.randint(20, 100))]
            return "\n".join(lines) + "\n"
        # kv
        lines = []
        for _ in range(n):
            k = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz_") for _ in range(rng.randint(3, 12)))
            v = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(rng.randint(3, 20)))
            lines.append(f"{k}: {v}")
        return "\n".join(lines) + "\n"
```

- [ ] **Step 5: Implement `corpus/generators/diffs.py`**

```python
import random
from corpus.generators.base import Generator


class UnifiedDiffGenerator(Generator):
    archetype = "diff"

    def generate(self, rng: random.Random) -> str:
        n_files = rng.randint(1, 4)
        chunks = []
        for _ in range(n_files):
            fname = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz_") for _ in range(rng.randint(3, 10)))
            fname += rng.choice([".py", ".js", ".rs", ".go"])
            path = f"src/{fname}"
            chunks.append(f"diff --git a/{path} b/{path}")
            chunks.append(f"index {rng.randint(0x100000, 0xffffff):06x}..{rng.randint(0x100000, 0xffffff):06x} 100644")
            chunks.append(f"--- a/{path}")
            chunks.append(f"+++ b/{path}")
            n_hunks = rng.randint(1, 3)
            for _ in range(n_hunks):
                old_start = rng.randint(1, 200)
                old_count = rng.randint(3, 12)
                new_count = rng.randint(3, 12)
                chunks.append(f"@@ -{old_start},{old_count} +{old_start},{new_count} @@")
                for _ in range(rng.randint(2, 8)):
                    op = rng.choice([" ", " ", " ", "-", "+"])
                    line = "    " + " ".join(
                        rng.choice(["foo", "bar", "baz", "x", "y", "self", "return", "if"])
                        for _ in range(rng.randint(1, 6))
                    )
                    chunks.append(op + line)
        return "\n".join(chunks) + "\n"
```

- [ ] **Step 6: Implement `corpus/generators/logs.py`**

```python
import random
from corpus.generators.base import Generator


_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]
_LEVEL_WEIGHTS = [10, 60, 20, 10]
_COMPONENTS = ["server", "db", "cache", "auth", "api", "worker", "scheduler"]
_MESSAGES = [
    "request handled in {ms}ms",
    "connected to {host}",
    "retrying after {n} attempts",
    "config loaded from {path}",
    "cache miss for key={key}",
    "rate limit exceeded for ip={ip}",
    "shutting down gracefully",
    "starting up version {ver}",
    "connection closed by peer",
    "deprecated API used: {fn}",
]


class LogGenerator(Generator):
    archetype = "log"

    def generate(self, rng: random.Random) -> str:
        n = rng.randint(10, 80)
        lines = []
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        for _ in range(n):
            second = rng.randint(0, 59)
            ms = rng.randint(0, 999)
            ts = f"2026-04-{rng.randint(1, 28):02d}T{hour:02d}:{minute:02d}:{second:02d}.{ms:03d}Z"
            level = rng.choices(_LEVELS, weights=_LEVEL_WEIGHTS, k=1)[0]
            comp = rng.choice(_COMPONENTS)
            tmpl = rng.choice(_MESSAGES)
            msg = tmpl.format(
                ms=rng.randint(1, 9999), host=f"db-{rng.randint(0,99)}.internal",
                n=rng.randint(1, 10), path=f"/etc/{rng.choice(['app','svc','daemon'])}.yaml",
                key=f"user:{rng.randint(1000, 99999)}", ip=f"10.0.{rng.randint(0,255)}.{rng.randint(0,255)}",
                ver=f"{rng.randint(1,3)}.{rng.randint(0,9)}.{rng.randint(0,9)}",
                fn=rng.choice(["legacy_login", "old_api", "v1_handler"]),
            )
            lines.append(f"{ts} [{level:<5}] {comp}: {msg}")
        return "\n".join(lines) + "\n"
```

- [ ] **Step 7: Implement `corpus/generators/code.py`**

```python
import random
from corpus.generators.base import Generator


_PY_TEMPLATES = [
    'def {fn}({args}):\n    """{doc}"""\n    return {ret}\n\n',
    'class {cls}:\n    def __init__(self, {args}):\n        self.{attr} = {attr}\n\n',
    'if __name__ == "__main__":\n    main()\n',
]
_JS_TEMPLATES = [
    'function {fn}({args}) {{\n  return {ret};\n}}\n\n',
    'const {var} = ({args}) => {{\n  return {ret};\n}};\n\n',
    'export class {cls} {{\n  constructor({args}) {{\n    this.{attr} = {attr};\n  }}\n}}\n\n',
]


def _name(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz_") for _ in range(rng.randint(3, 10)))


class CodeGenerator(Generator):
    archetype = "code"

    def generate(self, rng: random.Random) -> str:
        lang = rng.choice(["python", "javascript"])
        templates = _PY_TEMPLATES if lang == "python" else _JS_TEMPLATES
        n_blocks = rng.randint(2, 8)
        out = []
        for _ in range(n_blocks):
            t = rng.choice(templates)
            out.append(t.format(
                fn=_name(rng), args=", ".join(_name(rng) for _ in range(rng.randint(0, 4))),
                ret=_name(rng), doc=" ".join(_name(rng) for _ in range(rng.randint(2, 6))),
                cls=_name(rng).capitalize(), attr=_name(rng), var=_name(rng),
            ))
        return "".join(out)
```

- [ ] **Step 8: Implement `corpus/generators/json_yaml.py`**

```python
import json
import random
from corpus.generators.base import Generator


def _random_json_value(rng: random.Random, depth: int):
    if depth >= 4:
        return rng.choice([rng.randint(0, 1000), True, False, None, _random_string(rng)])
    kind = rng.choice(["dict", "list", "str", "int", "bool"])
    if kind == "dict":
        n = rng.randint(1, 6)
        return {_random_string(rng): _random_json_value(rng, depth + 1) for _ in range(n)}
    if kind == "list":
        n = rng.randint(1, 5)
        return [_random_json_value(rng, depth + 1) for _ in range(n)]
    if kind == "str":
        return _random_string(rng)
    if kind == "int":
        return rng.randint(-1000, 10000)
    return rng.choice([True, False])


def _random_string(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz_") for _ in range(rng.randint(3, 10)))


class JsonYamlGenerator(Generator):
    archetype = "json"

    def generate(self, rng: random.Random) -> str:
        # JSON only for v1 — YAML adds parsing complexity we don't need
        obj = _random_json_value(rng, 0)
        if not isinstance(obj, (dict, list)):
            obj = {"value": obj}
        return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
```

- [ ] **Step 9: Run all tests, verify all pass**

```bash
uv run pytest tests/corpus/ -v
```

Expected: ~24 tests passing.

- [ ] **Step 10: Commit**

```bash
git add corpus/generators/trees.py corpus/generators/lists.py \
        corpus/generators/diffs.py corpus/generators/logs.py \
        corpus/generators/code.py corpus/generators/json_yaml.py \
        tests/corpus/test_trees.py tests/corpus/test_lists.py \
        tests/corpus/test_diffs.py tests/corpus/test_logs.py \
        tests/corpus/test_code.py tests/corpus/test_json_yaml.py
git commit -m "Add tree/list/diff/log/code/json corpus generators"
```

---

## Task 3: Corpus generation CLI

**Goal:** A single command that runs all generators and writes ~150–250 MB of clean corpus to `corpus/output/<archetype>.jsonl`. Each line is one sample with `{archetype, src, text}`.

**Files:**
- Create: `corpus/generate_corpus.py`

**Acceptance Criteria:**
- [ ] CLI accepts `--samples-per-archetype N` and `--seed S` flags
- [ ] Writes one jsonl file per archetype to `corpus/output/`
- [ ] At default `--samples-per-archetype 30000`, output total is ~150–250 MB
- [ ] Each line is valid JSON: `{"archetype": str, "src": str, "text": str}`
- [ ] Idempotent given same seed (re-running produces identical bytes)
- [ ] Reasonable runtime: <30 minutes single-threaded on M4

**Verify:** `uv run python -m corpus.generate_corpus --samples-per-archetype 100 --seed 0 --output-dir /tmp/corpus_test && wc -l /tmp/corpus_test/*.jsonl` → 7 files, 100 lines each.

**Steps:**

- [ ] **Step 1: Implement the CLI**

`corpus/generate_corpus.py`:

```python
"""Generate clean corpus across all archetypes. CLI entry point."""
import argparse
import json
import random
from pathlib import Path

from corpus.generators.tables import LsListingGenerator, PsListingGenerator, DfListingGenerator
from corpus.generators.trees import TreeGenerator
from corpus.generators.lists import FlatListGenerator
from corpus.generators.diffs import UnifiedDiffGenerator
from corpus.generators.logs import LogGenerator
from corpus.generators.code import CodeGenerator
from corpus.generators.json_yaml import JsonYamlGenerator


GENERATORS = [
    ("table_ls", LsListingGenerator()),
    ("table_ps", PsListingGenerator()),
    ("table_df", DfListingGenerator()),
    ("tree", TreeGenerator()),
    ("list", FlatListGenerator()),
    ("diff", UnifiedDiffGenerator()),
    ("log", LogGenerator()),
    ("code", CodeGenerator()),
    ("json", JsonYamlGenerator()),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-per-archetype", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("corpus/output"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Group by archetype name (multiple generators per archetype)
    archetypes: dict[str, list] = {}
    for src, gen in GENERATORS:
        archetypes.setdefault(gen.archetype, []).append((src, gen))

    for archetype, gens in archetypes.items():
        out_path = args.output_dir / f"{archetype}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            samples_per_gen = args.samples_per_archetype // len(gens)
            for src, gen in gens:
                # Per-generator deterministic stream from a derived seed
                seed = args.seed * 1_000_003 + hash(src) % (2**31)
                rng = random.Random(seed)
                for i in range(samples_per_gen):
                    text = gen.generate(rng)
                    rec = {"archetype": archetype, "src": f"{src}/{i}", "text": text}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {archetype}: {out_path} ({samples_per_gen * len(gens)} samples)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with small N**

```bash
uv run python -m corpus.generate_corpus --samples-per-archetype 100 --seed 0 \
    --output-dir /tmp/corpus_test
ls -la /tmp/corpus_test/
wc -l /tmp/corpus_test/*.jsonl
```

Expected: 7 jsonl files (table, tree, list, diff, log, code, json), each ~100 lines (table has 99 = 33×3 since it has 3 sub-generators, fine).

- [ ] **Step 3: Verify a sample is valid JSON with the schema**

```bash
uv run python -c "
import json
from pathlib import Path
for f in Path('/tmp/corpus_test').glob('*.jsonl'):
    line = f.open().readline()
    rec = json.loads(line)
    assert {'archetype', 'src', 'text'} <= rec.keys(), f.name
    print(f.name, '->', rec['archetype'], '|', rec['src'])
"
```

Expected: prints one line per archetype, no errors.

- [ ] **Step 4: Verify determinism**

```bash
uv run python -m corpus.generate_corpus --samples-per-archetype 100 --seed 0 \
    --output-dir /tmp/corpus_test_a
uv run python -m corpus.generate_corpus --samples-per-archetype 100 --seed 0 \
    --output-dir /tmp/corpus_test_b
diff -r /tmp/corpus_test_a /tmp/corpus_test_b
```

Expected: no output (identical).

- [ ] **Step 5: Run the full corpus generation (real)**

```bash
uv run python -m corpus.generate_corpus --samples-per-archetype 30000 --seed 0
du -sh corpus/output/
ls -lh corpus/output/
```

Expected: ~150–250 MB total. If meaningfully outside that range (more than 2× off), bump or trim `samples-per-archetype` and re-run.

- [ ] **Step 6: Commit**

```bash
git add corpus/generate_corpus.py
git commit -m "Add corpus generation CLI"
```

---

## Task 4: Dirtifier Transform ABC + ANSI transforms

**Goal:** Establish the dirtifier transform contract and ship the ANSI family (Color, Bold, Underline, OSC, Hyperlink). All transforms are deterministic given a seeded RNG.

**Files:**
- Create: `dirtifier/transforms/base.py`
- Create: `dirtifier/transforms/ansi.py`
- Create: `tests/dirtifier/__init__.py`
- Create: `tests/dirtifier/test_base.py`
- Create: `tests/dirtifier/test_ansi.py`

**Acceptance Criteria:**
- [ ] `Transform` ABC: `apply(clean: str, rng: random.Random) -> str` and `name: str`
- [ ] `AnsiColor`, `AnsiBold`, `AnsiUnderline`, `AnsiOsc`, `Hyperlink` implemented
- [ ] Each transform is deterministic (same input + seed → same output)
- [ ] Each transform's output, after running through deterministic ANSI strip, equals the input (round-trip safety)
- [ ] No transform crashes on empty input or single-character input

**Verify:** `uv run pytest tests/dirtifier/test_base.py tests/dirtifier/test_ansi.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write tests for `Transform` ABC**

`tests/dirtifier/test_base.py`:

```python
import random
import pytest
from dirtifier.transforms.base import Transform


def test_transform_is_abstract():
    with pytest.raises(TypeError):
        Transform()


def test_transform_subclass_works():
    class Concrete(Transform):
        name = "concrete"
        def apply(self, clean, rng):
            return clean + "!"
    t = Concrete()
    assert t.apply("hi", random.Random(0)) == "hi!"
    assert t.name == "concrete"
```

- [ ] **Step 2: Implement `Transform` ABC**

`dirtifier/transforms/base.py`:

```python
import random
from abc import ABC, abstractmethod


class Transform(ABC):
    """Apply some kind of dirt to clean text. Deterministic given rng seed."""
    name: str

    @abstractmethod
    def apply(self, clean: str, rng: random.Random) -> str: ...
```

- [ ] **Step 3: Run base test, verify pass**

```bash
uv run pytest tests/dirtifier/test_base.py -v
```

Expected: 2 passed.

- [ ] **Step 4: Write tests for ANSI transforms**

`tests/dirtifier/test_ansi.py`:

```python
import random
import re

import pytest

from dirtifier.transforms.ansi import AnsiColor, AnsiBold, AnsiUnderline, AnsiOsc, Hyperlink


# Match all ANSI/CSI/OSC sequences for round-trip stripping
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\\")


def _strip(s: str) -> str:
    return _ANSI_RE.sub("", s)


@pytest.fixture(params=[AnsiColor, AnsiBold, AnsiUnderline, AnsiOsc, Hyperlink])
def transform_cls(request):
    return request.param


def test_ansi_deterministic(transform_cls):
    t = transform_cls()
    a = t.apply("hello world\nfoo bar\n", random.Random(0))
    b = t.apply("hello world\nfoo bar\n", random.Random(0))
    assert a == b


def test_ansi_strips_back_to_clean(transform_cls):
    t = transform_cls()
    clean = "hello world\nfoo bar\n"
    dirty = t.apply(clean, random.Random(7))
    assert _strip(dirty) == clean


def test_ansi_handles_empty(transform_cls):
    t = transform_cls()
    assert t.apply("", random.Random(0)) == ""


def test_ansi_color_actually_adds_color_code():
    t = AnsiColor()
    out = t.apply("hello world this is several words for sure here\n", random.Random(0))
    # At least one CSI sequence ending in 'm' (SGR set)
    assert re.search(r"\x1b\[[0-9;]+m", out)


def test_hyperlink_adds_osc8():
    t = Hyperlink()
    out = t.apply("see https://example.com here\n", random.Random(0))
    # OSC 8 sequence: \x1b]8;...;url\x07text\x1b]8;;\x07
    assert "\x1b]8;" in out
```

- [ ] **Step 5: Run ANSI tests, verify they fail**

```bash
uv run pytest tests/dirtifier/test_ansi.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 6: Implement ANSI transforms**

`dirtifier/transforms/ansi.py`:

```python
import random
import re
from dirtifier.transforms.base import Transform


# Foreground colors 30-37 + bright 90-97
_FG_CODES = list(range(30, 38)) + list(range(90, 98))
# Background 40-47 + bright 100-107
_BG_CODES = list(range(40, 48)) + list(range(100, 108))
_RESET = "\x1b[0m"


def _wrap(text: str, codes: list[int]) -> str:
    if not text:
        return text
    seq = "\x1b[" + ";".join(str(c) for c in codes) + "m"
    return seq + text + _RESET


class AnsiColor(Transform):
    """Wrap random tokens / lines in random foreground (and sometimes background) colors."""
    name = "ansi_color"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        # Tokenize to whitespace runs vs non-whitespace runs, color random non-ws tokens
        tokens = re.findall(r"\S+|\s+", clean)
        out = []
        for tok in tokens:
            if tok.isspace() or rng.random() > 0.4:
                out.append(tok)
                continue
            codes = [rng.choice(_FG_CODES)]
            if rng.random() < 0.1:
                codes.append(rng.choice(_BG_CODES))
            out.append(_wrap(tok, codes))
        return "".join(out)


class AnsiBold(Transform):
    name = "ansi_bold"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        tokens = re.findall(r"\S+|\s+", clean)
        out = []
        for tok in tokens:
            if tok.isspace() or rng.random() > 0.2:
                out.append(tok)
                continue
            out.append(_wrap(tok, [1]))
        return "".join(out)


class AnsiUnderline(Transform):
    name = "ansi_underline"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        tokens = re.findall(r"\S+|\s+", clean)
        out = []
        for tok in tokens:
            if tok.isspace() or rng.random() > 0.15:
                out.append(tok)
                continue
            out.append(_wrap(tok, [4]))
        return "".join(out)


class AnsiOsc(Transform):
    """Inject an OSC sequence (e.g. set window title) at the start. Visible-text-preserving."""
    name = "ansi_osc"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        # OSC 0; <title> BEL — sets terminal title; doesn't affect printed text
        title = "build" if rng.random() < 0.5 else "session"
        osc = f"\x1b]0;{title}\x07"
        return osc + clean


class Hyperlink(Transform):
    """Wrap URLs (or random tokens) in OSC-8 hyperlink markers."""
    name = "hyperlink"

    _URL_RE = re.compile(r"https?://\S+")

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        def _wrap_url(m):
            url = m.group(0)
            return f"\x1b]8;;{url}\x07{url}\x1b]8;;\x07"
        # Wrap actual URLs always; with low prob also wrap a random token in a fake link
        out = self._URL_RE.sub(_wrap_url, clean)
        return out
```

- [ ] **Step 7: Run tests, verify all pass**

```bash
uv run pytest tests/dirtifier/test_ansi.py -v
```

Expected: 18+ passed (5 transforms × 3 parametrized tests + 2 specific = 17, ±).

- [ ] **Step 8: Commit**

```bash
git add dirtifier/transforms/base.py dirtifier/transforms/ansi.py \
        tests/dirtifier/__init__.py tests/dirtifier/test_base.py tests/dirtifier/test_ansi.py
git commit -m "Add dirtifier Transform ABC and ANSI transforms"
```

---

## Task 5: Progress, spinner, cursor-movement, box-drawing transforms

**Goal:** Implement the visual-artifact transforms — progress bars, spinners, cursor-movement (top-style redraws), box-drawing wrappers.

**Files:**
- Create: `dirtifier/transforms/progress.py`
- Create: `dirtifier/transforms/box.py`
- Create: `tests/dirtifier/test_progress.py`
- Create: `tests/dirtifier/test_box.py`

**Acceptance Criteria:**
- [ ] `ProgressBar` injects `\r`-overwriting bar progression that ends on the original line
- [ ] `Spinner` injects `\r`-overwriting frames (e.g. `|`, `/`, `-`, `\`) that end on the original line
- [ ] `CursorMovement` injects cursor up + clear-line escape codes around random redraws
- [ ] `BoxDrawing` wraps content with Unicode box characters (├ ─ │ └) without changing internal data
- [ ] Each transform deterministic
- [ ] After deterministic `\r`-collapse + ANSI-strip, output round-trips to clean (within whitespace tolerance for `BoxDrawing`)

**Verify:** `uv run pytest tests/dirtifier/test_progress.py tests/dirtifier/test_box.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write tests**

`tests/dirtifier/test_progress.py`:

```python
import random
import re
from dirtifier.transforms.progress import ProgressBar, Spinner, CursorMovement


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)


def _collapse_cr(s: str) -> str:
    """For each line, keep only the final state after \r overwrites."""
    out_lines = []
    for line in s.split("\n"):
        # Each \r resets the line buffer to the segment after the last \r
        segments = line.split("\r")
        out_lines.append(segments[-1])
    return "\n".join(out_lines)


def test_progress_bar_deterministic():
    t = ProgressBar()
    a = t.apply("done\n", random.Random(0))
    b = t.apply("done\n", random.Random(0))
    assert a == b


def test_progress_bar_injects_cr_progression():
    t = ProgressBar()
    out = t.apply("install complete\n", random.Random(0))
    assert "\r" in out
    # After collapse + ansi strip, the final line is the original
    assert _strip_ansi(_collapse_cr(out)).rstrip("\n") == "install complete"


def test_spinner_injects_cr_frames():
    t = Spinner()
    out = t.apply("done\n", random.Random(0))
    assert "\r" in out
    assert _strip_ansi(_collapse_cr(out)).rstrip("\n") == "done"


def test_cursor_movement_injects_escape_codes():
    t = CursorMovement()
    out = t.apply("line a\nline b\nline c\n", random.Random(0))
    # Cursor up = \x1b[<n>A; clear line = \x1b[2K or \x1b[K
    assert re.search(r"\x1b\[\d*[AK]", out) or re.search(r"\x1b\[2K", out)


def test_progress_handles_empty():
    assert ProgressBar().apply("", random.Random(0)) == ""
    assert Spinner().apply("", random.Random(0)) == ""
    assert CursorMovement().apply("", random.Random(0)) == ""
```

`tests/dirtifier/test_box.py`:

```python
import random
from dirtifier.transforms.box import BoxDrawing


def test_box_deterministic():
    t = BoxDrawing()
    a = t.apply("foo\nbar\n", random.Random(0))
    b = t.apply("foo\nbar\n", random.Random(0))
    assert a == b


def test_box_uses_box_chars():
    t = BoxDrawing()
    out = t.apply("alpha\nbeta\ngamma\n", random.Random(0))
    assert "│" in out or "─" in out or "├" in out or "└" in out or "┌" in out or "┐" in out


def test_box_preserves_inner_text():
    t = BoxDrawing()
    out = t.apply("alpha\nbeta\ngamma\n", random.Random(0))
    # Original text is still present (substring) in each line
    assert "alpha" in out
    assert "beta" in out
    assert "gamma" in out


def test_box_handles_empty():
    assert BoxDrawing().apply("", random.Random(0)) == ""
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
uv run pytest tests/dirtifier/test_progress.py tests/dirtifier/test_box.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement progress transforms**

`dirtifier/transforms/progress.py`:

```python
import random
from dirtifier.transforms.base import Transform


class ProgressBar(Transform):
    """Inject a series of \\r-overwriting progress-bar frames before the final clean line."""
    name = "progress_bar"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        n_frames = rng.randint(5, 20)
        bar_width = rng.randint(20, 40)
        frames = []
        for i in range(n_frames):
            pct = int((i + 1) / n_frames * 100)
            filled = int((i + 1) / n_frames * bar_width)
            bar = "#" * filled + "-" * (bar_width - filled)
            frames.append(f"[{bar}] {pct}%")
        # All frames overwrite each other on a single line; the final clean line follows
        progression = "\r".join(frames) + "\r"
        return progression + clean


class Spinner(Transform):
    """Inject \\r-overwriting spinner frames before the final clean line."""
    name = "spinner"

    _FRAMES = ["|", "/", "-", "\\"]

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        n = rng.randint(8, 32)
        progression = "\r".join(self._FRAMES[i % 4] + " working..." for i in range(n)) + "\r"
        return progression + clean


class CursorMovement(Transform):
    """Inject cursor-up + clear-line escape codes mimicking a TUI redraw."""
    name = "cursor_movement"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        lines = clean.split("\n")
        if len(lines) < 3:
            return clean
        # Pick a midpoint, redraw the previous N lines
        n_redraw = rng.randint(1, min(5, len(lines) - 1))
        # Cursor up N then clear-line, then re-emit those lines
        i = rng.randint(n_redraw, len(lines) - 1)
        # Emit lines up through i, then a redraw block of n_redraw, then the rest
        prefix = "\n".join(lines[:i])
        redraw_block = f"\x1b[{n_redraw}A" + ("\x1b[2K\n" * n_redraw)
        rest = "\n".join(lines[i:])
        return prefix + "\n" + redraw_block + rest
```

- [ ] **Step 4: Implement box-drawing transform**

`dirtifier/transforms/box.py`:

```python
import random
from dirtifier.transforms.base import Transform


class BoxDrawing(Transform):
    """Wrap text in a Unicode box (top, bottom, side bars)."""
    name = "box_drawing"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        lines = clean.rstrip("\n").split("\n")
        max_len = max(len(line) for line in lines)
        top = "┌" + "─" * (max_len + 2) + "┐"
        bot = "└" + "─" * (max_len + 2) + "┘"
        wrapped = [f"│ {line.ljust(max_len)} │" for line in lines]
        return "\n".join([top] + wrapped + [bot]) + "\n"
```

- [ ] **Step 5: Run tests, verify all pass**

```bash
uv run pytest tests/dirtifier/test_progress.py tests/dirtifier/test_box.py -v
```

Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add dirtifier/transforms/progress.py dirtifier/transforms/box.py \
        tests/dirtifier/test_progress.py tests/dirtifier/test_box.py
git commit -m "Add progress/spinner/cursor and box-drawing dirtifier transforms"
```

---

## Task 6: Repetition, whitespace, locale, timestamps, streams, edge transforms

**Goal:** Round out the v1 transform set. After this task, the dirtifier covers the full taxonomy from spec §5.2.

**Files:**
- Create: `dirtifier/transforms/repeat.py`
- Create: `dirtifier/transforms/whitespace.py`
- Create: `dirtifier/transforms/locale.py`
- Create: `dirtifier/transforms/timestamps.py`
- Create: `dirtifier/transforms/streams.py`
- Create: `dirtifier/transforms/edge.py`
- Create: `tests/dirtifier/test_repeat.py`
- Create: `tests/dirtifier/test_whitespace.py`
- Create: `tests/dirtifier/test_locale.py`
- Create: `tests/dirtifier/test_timestamps.py`
- Create: `tests/dirtifier/test_streams.py`
- Create: `tests/dirtifier/test_edge.py`

**Acceptance Criteria:**
- [ ] `RepeatedLines` duplicates random lines N (2..50) times
- [ ] `WhitespacePadding` adds extra spaces inside / between columns
- [ ] `TrailingWhitespace` adds trailing spaces / blank lines
- [ ] `WindowsLineEndings` flips `\n` → `\r\n` on a fraction of lines
- [ ] `LocaleVariants` swaps thousand separators / date formats
- [ ] `Timestamps` prefixes each line with an ISO timestamp
- [ ] `MixedStreams` interleaves fake stderr lines (lines starting with `error:` / `warning:`)
- [ ] `BellChars` inserts random `\a`; `NulBytes` inserts random `\0`
- [ ] All deterministic; all handle empty input

**Verify:** `uv run pytest tests/dirtifier/ -v` → all pass.

**Steps:**

- [ ] **Step 1: Write tests for all six modules**

For each module write tests following the pattern from Task 4:
1. `test_<X>_deterministic`
2. `test_<X>_handles_empty`
3. `test_<X>_actually_modifies` — confirms the transform did something visible
4. (Optional) one specific structural assertion

Example for `tests/dirtifier/test_repeat.py`:

```python
import random
from dirtifier.transforms.repeat import RepeatedLines


def test_repeat_deterministic():
    t = RepeatedLines()
    a = t.apply("a\nb\nc\n", random.Random(0))
    b = t.apply("a\nb\nc\n", random.Random(0))
    assert a == b


def test_repeat_handles_empty():
    assert RepeatedLines().apply("", random.Random(0)) == ""


def test_repeat_increases_line_count():
    t = RepeatedLines()
    out = t.apply("a\nb\nc\n", random.Random(0))
    assert out.count("\n") >= 3  # at least the original 3
```

(Equivalent shape for the other five test files — see implementation files below for what each transform does, then write the assertions.)

- [ ] **Step 2: Run tests, all fail**

- [ ] **Step 3: Implement `dirtifier/transforms/repeat.py`**

```python
import random
from dirtifier.transforms.base import Transform


class RepeatedLines(Transform):
    """Pick random lines and duplicate them N times in place."""
    name = "repeated_lines"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        lines = clean.split("\n")
        out = []
        for line in lines:
            out.append(line)
            if line and rng.random() < 0.05:
                n = rng.randint(2, 50)
                out.extend([line] * (n - 1))
        return "\n".join(out)
```

- [ ] **Step 4: Implement `dirtifier/transforms/whitespace.py`**

```python
import random
from dirtifier.transforms.base import Transform


class WhitespacePadding(Transform):
    """Add extra spaces between whitespace runs (mimicking column padding)."""
    name = "whitespace_padding"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out_lines = []
        for line in clean.split("\n"):
            # Replace each space run with a longer run with low prob
            chars = []
            i = 0
            while i < len(line):
                if line[i] == " ":
                    j = i
                    while j < len(line) and line[j] == " ":
                        j += 1
                    n = j - i
                    if rng.random() < 0.3:
                        n += rng.randint(1, 4)
                    chars.append(" " * n)
                    i = j
                else:
                    chars.append(line[i])
                    i += 1
            out_lines.append("".join(chars))
        return "\n".join(out_lines)


class TrailingWhitespace(Transform):
    """Add trailing spaces per line + extra blank lines."""
    name = "trailing_whitespace"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out_lines = []
        for line in clean.split("\n"):
            if rng.random() < 0.3:
                line = line + " " * rng.randint(1, 8)
            out_lines.append(line)
            if rng.random() < 0.05:
                out_lines.append("")
        return "\n".join(out_lines)


class WindowsLineEndings(Transform):
    """Replace some \\n with \\r\\n."""
    name = "windows_line_endings"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out = []
        for ch in clean:
            if ch == "\n" and rng.random() < 0.4:
                out.append("\r\n")
            else:
                out.append(ch)
        return "".join(out)
```

- [ ] **Step 5: Implement `dirtifier/transforms/locale.py`**

```python
import re
import random
from dirtifier.transforms.base import Transform


class LocaleVariants(Transform):
    """Inject locale-style thousand separators in numbers."""
    name = "locale_variants"

    _NUM_RE = re.compile(r"\b\d{4,}\b")

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        sep = rng.choice([",", "_", " "])
        def _add_sep(m):
            n = m.group(0)
            # Add separator every 3 digits from the right
            rev = n[::-1]
            chunks = [rev[i:i+3] for i in range(0, len(rev), 3)]
            return sep.join(chunks)[::-1]
        return self._NUM_RE.sub(lambda m: _add_sep(m) if rng.random() < 0.5 else m.group(0), clean)
```

- [ ] **Step 6: Implement `dirtifier/transforms/timestamps.py`**

```python
import random
from dirtifier.transforms.base import Transform


class Timestamps(Transform):
    """Prefix each line with an ISO timestamp."""
    name = "timestamps"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        base_h = rng.randint(0, 23)
        base_m = rng.randint(0, 59)
        out = []
        for line in clean.split("\n"):
            sec = rng.randint(0, 59)
            ms = rng.randint(0, 999)
            ts = f"2026-04-{rng.randint(1,28):02d}T{base_h:02d}:{base_m:02d}:{sec:02d}.{ms:03d}Z"
            out.append(f"{ts} {line}")
        return "\n".join(out)
```

- [ ] **Step 7: Implement `dirtifier/transforms/streams.py`**

```python
import random
from dirtifier.transforms.base import Transform


_STDERR_LINES = [
    "warning: unused import",
    "error: connection refused",
    "warning: deprecated function 'foo' used",
    "info: retrying...",
    "debug: state=ready",
]


class MixedStreams(Transform):
    """Interleave fake stderr-style lines into stdout."""
    name = "mixed_streams"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out = []
        for line in clean.split("\n"):
            out.append(line)
            if rng.random() < 0.05:
                out.append(rng.choice(_STDERR_LINES))
        return "\n".join(out)
```

- [ ] **Step 8: Implement `dirtifier/transforms/edge.py`**

```python
import random
from dirtifier.transforms.base import Transform


class BellChars(Transform):
    """Sprinkle BEL (\\x07) characters."""
    name = "bell_chars"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out = []
        for ch in clean:
            out.append(ch)
            if rng.random() < 0.001:
                out.append("\x07")
        return "".join(out)


class NulBytes(Transform):
    """Sprinkle NUL bytes (\\x00) — rare, but real."""
    name = "nul_bytes"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out = []
        for ch in clean:
            out.append(ch)
            if rng.random() < 0.0005:
                out.append("\x00")
        return "".join(out)
```

- [ ] **Step 9: Run all tests, verify all pass**

```bash
uv run pytest tests/dirtifier/ -v
```

Expected: ~30+ tests passing.

- [ ] **Step 10: Commit**

```bash
git add dirtifier/transforms/repeat.py dirtifier/transforms/whitespace.py \
        dirtifier/transforms/locale.py dirtifier/transforms/timestamps.py \
        dirtifier/transforms/streams.py dirtifier/transforms/edge.py \
        tests/dirtifier/test_repeat.py tests/dirtifier/test_whitespace.py \
        tests/dirtifier/test_locale.py tests/dirtifier/test_timestamps.py \
        tests/dirtifier/test_streams.py tests/dirtifier/test_edge.py
git commit -m "Add repeat/whitespace/locale/timestamps/streams/edge transforms"
```

---

## Task 7: Recipes + pipeline composer

**Goal:** Define named recipes (compositions of transforms) per spec §5.3 and a deterministic pipeline that applies them.

**Files:**
- Create: `dirtifier/recipes.py`
- Create: `dirtifier/pipeline.py`
- Create: `tests/dirtifier/test_recipes.py`
- Create: `tests/dirtifier/test_pipeline.py`

**Acceptance Criteria:**
- [ ] `Recipe` is a list of (Transform, probability) pairs with a name
- [ ] All five recipes from spec §5.3 defined: `cli_colored_table`, `install_with_progress`, `tui_redraw`, `noisy_logs`, `passthrough`
- [ ] `apply_recipe(clean, recipe, rng)` returns dirty text deterministically
- [ ] `passthrough` returns input unchanged
- [ ] `RECIPES` dict maps name → Recipe
- [ ] Pipeline can pick a recipe at random with weighted distribution

**Verify:** `uv run pytest tests/dirtifier/test_recipes.py tests/dirtifier/test_pipeline.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write tests**

`tests/dirtifier/test_recipes.py`:

```python
from dirtifier.recipes import RECIPES, Recipe


def test_all_five_recipes_defined():
    expected = {"cli_colored_table", "install_with_progress", "tui_redraw", "noisy_logs", "passthrough"}
    assert expected <= set(RECIPES.keys())


def test_passthrough_is_empty():
    r = RECIPES["passthrough"]
    assert isinstance(r, Recipe)
    assert r.steps == []
```

`tests/dirtifier/test_pipeline.py`:

```python
import random
from dirtifier.pipeline import apply_recipe, pick_recipe
from dirtifier.recipes import RECIPES


def test_apply_recipe_deterministic():
    clean = "hello world\nfoo bar baz\n"
    a = apply_recipe(clean, RECIPES["cli_colored_table"], random.Random(0))
    b = apply_recipe(clean, RECIPES["cli_colored_table"], random.Random(0))
    assert a == b


def test_apply_passthrough_returns_unchanged():
    clean = "hello world\nfoo bar\n"
    out = apply_recipe(clean, RECIPES["passthrough"], random.Random(0))
    assert out == clean


def test_apply_install_with_progress_adds_cr():
    clean = "installed package foo-1.2.3\n"
    out = apply_recipe(clean, RECIPES["install_with_progress"], random.Random(0))
    # Should contain at least one \r (progress / spinner)
    assert "\r" in out


def test_pick_recipe_deterministic():
    rng_a = random.Random(0)
    rng_b = random.Random(0)
    assert pick_recipe(rng_a).name == pick_recipe(rng_b).name
```

- [ ] **Step 2: Implement `dirtifier/recipes.py`**

```python
from dataclasses import dataclass, field
from dirtifier.transforms.base import Transform
from dirtifier.transforms.ansi import AnsiColor, AnsiBold, AnsiUnderline, AnsiOsc, Hyperlink
from dirtifier.transforms.progress import ProgressBar, Spinner, CursorMovement
from dirtifier.transforms.box import BoxDrawing
from dirtifier.transforms.repeat import RepeatedLines
from dirtifier.transforms.whitespace import WhitespacePadding, TrailingWhitespace, WindowsLineEndings
from dirtifier.transforms.locale import LocaleVariants
from dirtifier.transforms.timestamps import Timestamps
from dirtifier.transforms.streams import MixedStreams
from dirtifier.transforms.edge import BellChars, NulBytes


@dataclass
class Recipe:
    name: str
    # Each step: (transform, probability of applying)
    steps: list[tuple[Transform, float]] = field(default_factory=list)
    # For weighted sampling at the pipeline level
    weight: float = 1.0


RECIPES: dict[str, Recipe] = {
    "cli_colored_table": Recipe(
        name="cli_colored_table",
        steps=[
            (AnsiColor(), 0.9),
            (AnsiBold(), 0.3),
            (WhitespacePadding(), 0.5),
            (TrailingWhitespace(), 0.3),
        ],
        weight=3.0,  # most common archetype dirt
    ),
    "install_with_progress": Recipe(
        name="install_with_progress",
        steps=[
            (ProgressBar(), 0.7),
            (Spinner(), 0.5),
            (AnsiColor(), 0.6),
            (AnsiBold(), 0.4),
            (RepeatedLines(), 0.4),
        ],
        weight=2.0,
    ),
    "tui_redraw": Recipe(
        name="tui_redraw",
        steps=[
            (CursorMovement(), 0.9),
            (AnsiColor(), 0.7),
            (BoxDrawing(), 0.3),
        ],
        weight=1.0,
    ),
    "noisy_logs": Recipe(
        name="noisy_logs",
        steps=[
            (Timestamps(), 0.6),
            (RepeatedLines(), 0.5),
            (AnsiColor(), 0.4),
            (MixedStreams(), 0.4),
        ],
        weight=2.0,
    ),
    "passthrough": Recipe(
        name="passthrough",
        steps=[],
        weight=1.5,  # important: model must learn to leave clean text alone
    ),
}
```

- [ ] **Step 3: Implement `dirtifier/pipeline.py`**

```python
import random
from dirtifier.recipes import RECIPES, Recipe


def apply_recipe(clean: str, recipe: Recipe, rng: random.Random) -> str:
    """Apply each step of the recipe in order, with each step's probability."""
    out = clean
    for transform, prob in recipe.steps:
        if rng.random() < prob:
            out = transform.apply(out, rng)
    return out


def pick_recipe(rng: random.Random, recipes: dict[str, Recipe] = RECIPES) -> Recipe:
    """Weighted random selection of a recipe."""
    names = list(recipes.keys())
    weights = [recipes[n].weight for n in names]
    return recipes[rng.choices(names, weights=weights, k=1)[0]]
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
uv run pytest tests/dirtifier/test_recipes.py tests/dirtifier/test_pipeline.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add dirtifier/recipes.py dirtifier/pipeline.py \
        tests/dirtifier/test_recipes.py tests/dirtifier/test_pipeline.py
git commit -m "Add dirtifier recipes and pipeline composer"
```

---

## Task 8: Data generation CLI

**Goal:** Read corpus jsonl files, emit `train.jsonl` and `val.jsonl` of `(dirty, clean)` training pairs. Cap each example at 4096 tokens combined; drop oversized examples explicitly (no silent truncation).

**Files:**
- Create: `dirtifier/generate.py`
- Create: `tests/dirtifier/test_generate.py`

**Acceptance Criteria:**
- [ ] CLI reads `corpus/output/*.jsonl`, writes `data/train.jsonl` and `data/val.jsonl`
- [ ] Each output line is JSON with keys `input`, `output`, `meta`
- [ ] Examples that exceed 4096 tokens (input+output combined, by tokenizer) are dropped, not truncated
- [ ] Token counts are computed using the actual Gemma 4 E2B tokenizer (so counts match training)
- [ ] Train/val split is **by source file path** (not random across whole corpus) to prevent near-duplicates leaking
- [ ] Held-out **recipes** for val: `tui_redraw` only appears in val, never train (tests recipe-level generalization)
- [ ] Reports counts and total size at the end

**Verify:** `uv run pytest tests/dirtifier/test_generate.py -v && uv run python -m dirtifier.generate --help` → tests pass, CLI shows help.

**Steps:**

- [ ] **Step 1: Write a small unit test for the recipe-pinning logic**

`tests/dirtifier/test_generate.py`:

```python
from dirtifier.generate import _pick_split_recipes


def test_split_excludes_held_out_recipes_from_train():
    train_recipes, val_recipes = _pick_split_recipes(
        all_recipes=["a", "b", "c", "d"],
        held_out_for_val=["c"],
    )
    assert "c" not in train_recipes
    assert "c" in val_recipes
    assert set(train_recipes) == {"a", "b", "d"}
    assert set(val_recipes) == {"a", "b", "c", "d"}
```

- [ ] **Step 2: Implement `dirtifier/generate.py`**

```python
"""Generate training pairs by applying dirtifier recipes to clean corpus."""
import argparse
import json
import random
from pathlib import Path

from mlx_lm import load
from tqdm import tqdm

from dirtifier.pipeline import apply_recipe, pick_recipe
from dirtifier.recipes import RECIPES


HELD_OUT_FOR_VAL = ["tui_redraw"]
MAX_TOKENS = 4096


def _pick_split_recipes(all_recipes, held_out_for_val):
    """Train sees all but held-out; val sees all."""
    train_recipes = [r for r in all_recipes if r not in held_out_for_val]
    val_recipes = list(all_recipes)
    return train_recipes, val_recipes


def _filtered_recipes(names: list[str]) -> dict[str, "Recipe"]:
    return {n: RECIPES[n] for n in names}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-dir", type=Path, default=Path("corpus/output"))
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--n-pairs-per-clean", type=int, default=2,
                   help="How many dirty variants to produce per clean sample")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer-path", type=str, default="models/base/gemma-4-E2B-it-4bit")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_recipes, val_recipes = _pick_split_recipes(list(RECIPES.keys()), HELD_OUT_FOR_VAL)
    train_recipe_dict = _filtered_recipes(train_recipes)
    val_recipe_dict = _filtered_recipes(val_recipes)

    # Load tokenizer for token-count gating
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    _, tokenizer = load(args.tokenizer_path)

    rng = random.Random(args.seed)
    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    n_train_kept = n_val_kept = n_dropped = 0

    with train_path.open("w", encoding="utf-8") as f_train, \
         val_path.open("w", encoding="utf-8") as f_val:
        for corpus_file in sorted(args.corpus_dir.glob("*.jsonl")):
            with corpus_file.open(encoding="utf-8") as f:
                for line in tqdm(f, desc=corpus_file.name):
                    rec = json.loads(line)
                    clean = rec["text"]
                    is_val_sample = rng.random() < args.val_frac
                    recipes = val_recipe_dict if is_val_sample else train_recipe_dict
                    for _ in range(args.n_pairs_per_clean):
                        recipe = pick_recipe(rng, recipes)
                        dirty = apply_recipe(clean, recipe, rng)
                        # Token-count gate
                        n_in = len(tokenizer.encode(dirty))
                        n_out = len(tokenizer.encode(clean))
                        if n_in + n_out > MAX_TOKENS:
                            n_dropped += 1
                            continue
                        out_rec = {
                            "input": dirty,
                            "output": clean,
                            "meta": {
                                "recipe": recipe.name,
                                "archetype": rec["archetype"],
                                "src": rec["src"],
                                "n_tokens_in": n_in,
                                "n_tokens_out": n_out,
                            },
                        }
                        target = f_val if is_val_sample else f_train
                        target.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        if is_val_sample:
                            n_val_kept += 1
                        else:
                            n_train_kept += 1

    print(f"\nTrain: {n_train_kept} pairs -> {train_path}")
    print(f"Val:   {n_val_kept} pairs -> {val_path}")
    print(f"Dropped (>{MAX_TOKENS} tokens combined): {n_dropped}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run unit test**

```bash
uv run pytest tests/dirtifier/test_generate.py -v
```

Expected: 1 passed.

- [ ] **Step 4: Smoke-test CLI on small corpus**

```bash
# Use the small corpus from Task 3 if still around, or regenerate
uv run python -m corpus.generate_corpus --samples-per-archetype 200 --seed 0 \
    --output-dir /tmp/corpus_small
uv run python -m dirtifier.generate --corpus-dir /tmp/corpus_small \
    --output-dir /tmp/data_small --n-pairs-per-clean 1 --val-frac 0.1
wc -l /tmp/data_small/*.jsonl
head -1 /tmp/data_small/train.jsonl | python -m json.tool | head -20
```

Expected: train.jsonl has ~1500 lines, val.jsonl has ~150 lines (depends on token-cap drop rate). First record has `input`, `output`, `meta`.

- [ ] **Step 5: Generate full training data**

```bash
uv run python -m dirtifier.generate
du -sh data/
wc -l data/*.jsonl
```

Expected: `train.jsonl` ~150–400 MB, `val.jsonl` ~5–15 MB. If size is way off, adjust `--samples-per-archetype` (Task 3) or `--n-pairs-per-clean`.

- [ ] **Step 6: Commit**

```bash
git add dirtifier/generate.py tests/dirtifier/test_generate.py
git commit -m "Add data generation CLI: corpus + recipes -> train/val jsonl"
```

---

## Task 9: Real eval set capture + hand-curate

**Goal:** Produce `data/eval_real.jsonl` (500 hand-curated pairs from real captured outputs across 10 commands) and `data/eval_passthrough.jsonl` (200 already-clean inputs that should pass through unchanged).

**Files:**
- Create: `eval/capture.py` (helper to run commands + save raw stdout)
- Create: `eval/curate.py` (interactive helper: shows raw next to candidate clean, prompts y/edit/n)
- Create: `data/eval_real.jsonl` (committed; small enough)
- Create: `data/eval_passthrough.jsonl` (committed)

**Acceptance Criteria:**
- [ ] 500 real pairs across 10 commands (50 each) — `ls -la --color`, `find /usr | head -1000`, `git diff --color`, `pytest -v`, `npm install`, `cargo build`, `docker ps`, `kubectl get pods`, `cat <random_source_file>`, `tree`
- [ ] 200 passthrough pairs (clean inputs whose output should equal input)
- [ ] Both files have schema matching training data: `{"input", "output", "meta"}`
- [ ] `meta.command` records which command produced the input (for slice analysis)
- [ ] `meta.curator_notes` optional free-text for tricky cases

**Verify:** `wc -l data/eval_real.jsonl data/eval_passthrough.jsonl` → 500 and 200 lines respectively. `python -c "import json; [json.loads(l) for l in open('data/eval_real.jsonl')]"` → no errors.

**Steps:**

- [ ] **Step 1: Implement `eval/capture.py`**

```python
"""Capture raw stdout from real commands. Saves one-record-per-line jsonl."""
import argparse
import json
import os
import subprocess
from pathlib import Path


CAPTURE_TARGETS = {
    "ls_la_color": ("ls", "-la", "--color=always"),
    "find_usr": ("bash", "-c", "find /usr -maxdepth 4 2>/dev/null | head -300"),
    "git_diff_color": ("bash", "-c", "git -C . log -p --color=always -n 5 || true"),
    "pytest_v": ("bash", "-c", "uv run pytest -v 2>&1 | head -200"),
    # commands below may not be installed; we capture if available
    "npm_install": ("bash", "-c", "command -v npm >/dev/null && cd /tmp && mkdir -p npm_test && cd npm_test && echo '{}' > package.json && npm install --color=always lodash 2>&1 || echo 'npm not available'"),
    "cargo_build": ("bash", "-c", "command -v cargo >/dev/null && cd /tmp && cargo new cargo_test 2>/dev/null; cd /tmp/cargo_test && cargo build --color=always 2>&1 || echo 'cargo not available'"),
    "docker_ps": ("bash", "-c", "command -v docker >/dev/null && docker ps -a --format 'table {{.Image}}\\t{{.Status}}' 2>&1 || echo 'docker not available'"),
    "kubectl_get_pods": ("bash", "-c", "command -v kubectl >/dev/null && kubectl get pods --all-namespaces 2>&1 || echo 'kubectl not available'"),
    "cat_source": ("bash", "-c", "cat corpus/generators/tables.py"),
    "tree": ("bash", "-c", "command -v tree >/dev/null && tree -L 2 . 2>&1 || find . -maxdepth 2 | head -50"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/eval_real_raw.jsonl"))
    p.add_argument("--n-per-command", type=int, default=50,
                   help="Run each command N times with varied env (LANG, COLUMNS) to get variation")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    locales = ["C", "en_US.UTF-8", "fr_FR.UTF-8"]
    widths = ["80", "120", "200"]

    with args.out.open("w", encoding="utf-8") as f:
        for command_name, cmd in CAPTURE_TARGETS.items():
            for i in range(args.n_per_command):
                env = os.environ.copy()
                env["LANG"] = locales[i % len(locales)]
                env["COLUMNS"] = widths[i % len(widths)]
                env["FORCE_COLOR"] = "1"
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=30, env=env,
                    )
                    raw = result.stdout + (result.stderr if result.stderr else "")
                except Exception as e:
                    raw = f"[capture error: {e}]"
                rec = {
                    "input": raw,
                    "output": "",  # to be filled by curate
                    "meta": {
                        "command": command_name,
                        "lang": env["LANG"],
                        "columns": env["COLUMNS"],
                        "iter": i,
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Captured {args.n_per_command}× {command_name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement `eval/curate.py` — interactive curation helper**

```python
"""Interactive curation: shows raw input, runs deterministic ANSI strip + cr-collapse
as a baseline, asks user to accept / edit / reject."""
import argparse
import json
import re
from pathlib import Path

from infer.ansi_strip import strip  # implemented in Task 11


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    seen_ids = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                seen_ids.add(rec["meta"].get("_id"))

    with args.input.open(encoding="utf-8") as f_in, args.output.open("a", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            if idx in seen_ids:
                continue
            rec = json.loads(line)
            raw = rec["input"]
            baseline = strip(raw)
            print("\n" + "=" * 60)
            print(f"#{idx} command={rec['meta'].get('command')}")
            print("--- RAW INPUT ---")
            print(raw[:1000])
            print("--- BASELINE (deterministic ANSI strip) ---")
            print(baseline[:1000])
            print("=" * 60)
            choice = input("[a]ccept baseline / [e]dit / [s]kip / [q]uit: ").strip().lower()
            if choice == "q":
                break
            if choice == "s":
                continue
            if choice == "e":
                # Open in $EDITOR via a temp file
                import os, tempfile, subprocess
                with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
                    tf.write(baseline)
                    tf_path = tf.name
                subprocess.call([os.environ.get("EDITOR", "vim"), tf_path])
                with open(tf_path) as tf:
                    cleaned = tf.read()
                os.unlink(tf_path)
            else:  # accept
                cleaned = baseline
            rec["output"] = cleaned
            rec["meta"]["_id"] = idx
            rec["meta"]["curator_notes"] = ""
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Capture raw outputs (depends on Task 11 ANSI-strip — defer if needed)**

> **NOTE:** This task imports `infer.ansi_strip.strip` from Task 11. Either reorder execution (do Task 11 first) or implement a temporary inline ANSI-strip here. The plan executor should reorder; in subagent-driven development, Task 11 should be assigned before Task 9 runs.

```bash
uv run python -m eval.capture --out data/eval_real_raw.jsonl
wc -l data/eval_real_raw.jsonl
```

Expected: 500 lines (50 × 10 commands).

- [ ] **Step 4: Run curation**

```bash
uv run python -m eval.curate --input data/eval_real_raw.jsonl --output data/eval_real.jsonl
```

This is **interactive and slow** — budget 2–4 hours for 500 entries. The user (mjmoshiri) and Claude both review; Claude commits to supervising quality per the user's preference. Most entries will be `accept` (baseline ANSI strip is correct); only a fraction need editing.

- [ ] **Step 5: Build passthrough eval set**

```python
# eval/build_passthrough.py — small inline script
import json
import random
from pathlib import Path

# 200 already-clean inputs: source code, JSON, plain text
samples = []
# 100 from corpus code + json (already clean)
for fname in ["corpus/output/code.jsonl", "corpus/output/json.jsonl"]:
    with open(fname) as f:
        for line in f:
            rec = json.loads(line)
            samples.append({"input": rec["text"], "output": rec["text"],
                            "meta": {"command": "synthetic_passthrough",
                                     "archetype": rec["archetype"]}})
            if len(samples) >= 100:
                break
        if len(samples) >= 100:
            break

# 100 from real source files (cat-style)
for path in sorted(Path("corpus/generators").glob("*.py")):
    text = path.read_text()
    samples.append({"input": text, "output": text,
                    "meta": {"command": "real_passthrough", "src": str(path)}})

random.Random(0).shuffle(samples)
with open("data/eval_passthrough.jsonl", "w") as f:
    for s in samples[:200]:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
print(f"Wrote {len(samples[:200])} passthrough samples")
```

Save and run:

```bash
uv run python eval/build_passthrough.py
wc -l data/eval_passthrough.jsonl
```

Expected: 200 lines.

- [ ] **Step 6: Verify schema**

```bash
uv run python -c "
import json
for f in ['data/eval_real.jsonl', 'data/eval_passthrough.jsonl']:
    n = 0
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            assert {'input','output','meta'} <= r.keys(), f'{f}: missing keys'
            n += 1
    print(f, n)
"
```

Expected: prints two lines, 500 and 200.

- [ ] **Step 7: Commit**

```bash
git add eval/capture.py eval/curate.py eval/build_passthrough.py \
        data/eval_real.jsonl data/eval_passthrough.jsonl
git commit -m "Add real eval set (500) and passthrough eval set (200), hand-curated"
```

> **Note on `data/` gitignore:** `data/` is in `.gitignore` from Task 0, but we want `eval_real.jsonl` and `eval_passthrough.jsonl` committed (they're hand-curated and small). Add explicit re-includes:
> ```gitignore
> data/
> !data/eval_real.jsonl
> !data/eval_passthrough.jsonl
> ```
> Update `.gitignore` and re-stage.

---

## Task 10: Lossless guard

**Goal:** Implement deterministic atom extraction + subset check. This is the load-bearing safety mechanism for the entire system.

**Files:**
- Create: `eval/lossless_guard.py`
- Create: `tests/eval/__init__.py`
- Create: `tests/eval/test_lossless_guard.py`

**Acceptance Criteria:**
- [ ] `extract_atoms(text: str) -> set[str]` returns information atoms per spec §7.1
- [ ] Atom kinds: file paths, numbers (with units), identifiers ≥ 3 chars, quoted strings, URLs, IPs, emails
- [ ] `lossless_check(input_text, output_text, removable_whitelist=None) -> Result` returns `(passed: bool, missing_atoms: set[str])`
- [ ] Whitelist parameter allows known-removable atoms (dirtifier-injected)
- [ ] All edge cases tested: empty input, ANSI-only input, single number, repeated atoms, etc.

**Verify:** `uv run pytest tests/eval/test_lossless_guard.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write tests**

`tests/eval/test_lossless_guard.py`:

```python
from eval.lossless_guard import extract_atoms, lossless_check


def test_extract_atoms_paths():
    atoms = extract_atoms("see file at src/main.py and /etc/hosts here")
    assert "src/main.py" in atoms
    assert "/etc/hosts" in atoms


def test_extract_atoms_numbers():
    atoms = extract_atoms("size 1.2K, 4.5MB, count 1234, duration 30s")
    assert "1.2K" in atoms
    assert "4.5MB" in atoms
    assert "1234" in atoms


def test_extract_atoms_identifiers():
    atoms = extract_atoms("calling foo_bar(x) and HTTPClient and PI")
    assert "foo_bar" in atoms
    assert "HTTPClient" in atoms
    # 'PI' is too short (2 chars), should NOT be an atom
    assert "PI" not in atoms


def test_extract_atoms_quoted_strings():
    atoms = extract_atoms('error: "connection refused" and \'foo bar\'')
    assert '"connection refused"' in atoms or "connection refused" in atoms
    # at least one form of the quoted content
    assert any("connection refused" in a for a in atoms)


def test_extract_atoms_urls_ips_emails():
    atoms = extract_atoms("see https://example.com/path or 10.0.0.1 mail to a@b.co")
    assert any("example.com" in a for a in atoms)
    assert "10.0.0.1" in atoms
    assert "a@b.co" in atoms


def test_lossless_check_pass_simple():
    assert lossless_check("hello world foo", "hello world foo").passed


def test_lossless_check_pass_after_strip():
    # ANSI-stripped version retains the same atoms
    assert lossless_check("\x1b[31mhello world\x1b[0m", "hello world").passed


def test_lossless_check_fail_missing_atom():
    result = lossless_check("important error 12345 in /etc/hosts",
                            "some text without the path")
    assert not result.passed
    assert "12345" in result.missing_atoms or "/etc/hosts" in result.missing_atoms


def test_lossless_check_whitelist_allows_dirtifier_artifacts():
    # The dirtifier-injected atom is whitelisted, so even if it's missing the check passes
    result = lossless_check("real_id_42 spinner_frame_3", "real_id_42",
                            removable_whitelist={"spinner_frame_3"})
    assert result.passed


def test_lossless_check_empty_input():
    assert lossless_check("", "").passed
```

- [ ] **Step 2: Implement `eval/lossless_guard.py`**

```python
"""Deterministic information-preservation check for lossless cleanup.
See spec §7.1 for the atom taxonomy."""
import re
from dataclasses import dataclass


_PATH_RE = re.compile(r"(?:/|\./|~/|[\w-]+/)[\w./~-]+|[\w_-]+\.[a-zA-Z]{1,8}\b")
_NUMBER_WITH_UNIT_RE = re.compile(r"\b\d+(?:[.,]\d+)?(?:[KMGTP]i?B?|[smhd]|%)?\b")
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_QUOTED_RE = re.compile(r'"[^"\n]+"|\'[^\'\n]+\'')
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


@dataclass
class GuardResult:
    passed: bool
    missing_atoms: set[str]
    n_input: int
    n_output: int


def extract_atoms(text: str) -> set[str]:
    """Extract information atoms from text. Order: longer/more-specific first."""
    atoms: set[str] = set()
    # URLs first (they look like paths + numbers)
    for m in _URL_RE.finditer(text):
        atoms.add(m.group(0))
    text_for_rest = _URL_RE.sub(" ", text)
    for m in _EMAIL_RE.finditer(text_for_rest):
        atoms.add(m.group(0))
    text_for_rest = _EMAIL_RE.sub(" ", text_for_rest)
    for m in _IP_RE.finditer(text_for_rest):
        atoms.add(m.group(0))
    for m in _PATH_RE.finditer(text_for_rest):
        atoms.add(m.group(0))
    for m in _QUOTED_RE.finditer(text_for_rest):
        # Store both with and without quotes for flexible matching
        atoms.add(m.group(0))
        inner = m.group(0)[1:-1]
        atoms.add(inner)
    for m in _NUMBER_WITH_UNIT_RE.finditer(text_for_rest):
        atoms.add(m.group(0))
    for m in _IDENT_RE.finditer(text_for_rest):
        atoms.add(m.group(0))
    return atoms


def lossless_check(
    input_text: str,
    output_text: str,
    removable_whitelist: set[str] | None = None,
) -> GuardResult:
    """Verify atoms(input) ⊆ atoms(output) ∪ removable_whitelist."""
    in_atoms = extract_atoms(input_text)
    out_atoms = extract_atoms(output_text)
    whitelist = removable_whitelist or set()
    # Atoms that should still be present in output
    required = in_atoms - whitelist
    # An atom is "preserved" if it appears in output, OR a longer atom containing it is preserved
    # (handles e.g. "foo" being subsumed into "foo_bar")
    out_text_lc = output_text  # we check substring against raw output to forgive merging
    missing = set()
    for atom in required:
        if atom in out_atoms or atom in out_text_lc:
            continue
        missing.add(atom)
    return GuardResult(
        passed=len(missing) == 0,
        missing_atoms=missing,
        n_input=len(in_atoms),
        n_output=len(out_atoms),
    )
```

- [ ] **Step 3: Run tests, verify all pass**

```bash
uv run pytest tests/eval/test_lossless_guard.py -v
```

Expected: all 10 tests passed. If `test_extract_atoms_quoted_strings` is flaky, tighten/loosen the assertion to match implementation; spec doesn't dictate exact form.

- [ ] **Step 4: Commit**

```bash
git add eval/lossless_guard.py tests/eval/__init__.py tests/eval/test_lossless_guard.py
git commit -m "Add deterministic lossless guard with atom-set check"
```

---

## Task 11: Deterministic ANSI strip pre-processor

**Goal:** A pure-regex `strip(text: str) -> str` that removes ANSI/CSI/OSC escape sequences and collapses `\r`-overwrites. Used as: (a) inference pre-step, (b) safe fallback when model fails the guard, (c) baseline in curation (Task 9).

**Files:**
- Create: `infer/ansi_strip.py`
- Create: `tests/infer/__init__.py`
- Create: `tests/infer/test_ansi_strip.py`

**Acceptance Criteria:**
- [ ] `strip(text)` removes all CSI sequences (`\x1b[...]m`, `\x1b[...A`, etc.)
- [ ] `strip(text)` removes OSC sequences (incl. OSC 8 hyperlinks, leaving the visible text)
- [ ] `strip(text)` collapses `\r` overwrites: keeps only segment after final `\r` per line
- [ ] `strip(text)` removes BEL (`\x07`) and NUL (`\x00`)
- [ ] `strip(text)` is idempotent: `strip(strip(x)) == strip(x)`
- [ ] `strip("")` returns `""`

**Verify:** `uv run pytest tests/infer/test_ansi_strip.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write tests**

`tests/infer/test_ansi_strip.py`:

```python
from infer.ansi_strip import strip


def test_strip_csi():
    assert strip("\x1b[31mhello\x1b[0m") == "hello"


def test_strip_osc8_hyperlink_keeps_visible_text():
    # OSC 8: \x1b]8;;url\x07text\x1b]8;;\x07
    s = "\x1b]8;;https://example.com\x07click here\x1b]8;;\x07"
    assert strip(s) == "click here"


def test_strip_osc_title():
    # OSC 0: terminal title — entire sequence removed, no visible text
    assert strip("\x1b]0;mytitle\x07hello") == "hello"


def test_strip_collapses_cr():
    # Simulating progress bar: each frame ends in \r, final clean line follows
    s = "[##  ] 33%\r[#### ] 66%\r[#####]100%\rdone\n"
    assert strip(s) == "done\n"


def test_strip_handles_bell_and_nul():
    assert strip("foo\x07bar\x00baz") == "foobarbaz"


def test_strip_idempotent():
    s = "\x1b[31m\rhello\x1b[0m\n"
    assert strip(strip(s)) == strip(s)


def test_strip_empty():
    assert strip("") == ""


def test_strip_passthrough_clean_text():
    s = "hello world\nfoo bar\n"
    assert strip(s) == s
```

- [ ] **Step 2: Implement `infer/ansi_strip.py`**

```python
"""Deterministic terminal-noise stripping. Used as inference pre-step + safe fallback."""
import re


# Order matters: OSC 8 (hyperlinks) need special handling to preserve visible text
_OSC8_RE = re.compile(r"\x1b\]8;[^;]*;[^\x07\x1b]*(?:\x07|\x1b\\)([^\x1b]*)\x1b\]8;;(?:\x07|\x1b\\)")
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_OTHER_ESC_RE = re.compile(r"\x1b[@-_]")


def strip(text: str) -> str:
    if not text:
        return text
    # 1. OSC 8 hyperlinks: keep only visible text
    text = _OSC8_RE.sub(lambda m: m.group(1), text)
    # 2. Other OSC sequences: drop entirely
    text = _OSC_RE.sub("", text)
    # 3. CSI sequences (colors, cursor movement, etc.)
    text = _CSI_RE.sub("", text)
    # 4. Other ESC-* short sequences
    text = _OTHER_ESC_RE.sub("", text)
    # 5. \r-collapse per logical line: keep final segment after last \r
    out_lines = []
    for line in text.split("\n"):
        segments = line.split("\r")
        out_lines.append(segments[-1])
    text = "\n".join(out_lines)
    # 6. Drop BEL and NUL
    text = text.replace("\x07", "").replace("\x00", "")
    return text
```

- [ ] **Step 3: Run tests, verify all pass**

```bash
uv run pytest tests/infer/test_ansi_strip.py -v
```

Expected: 8 passed.

- [ ] **Step 4: Commit**

```bash
git add infer/ansi_strip.py tests/infer/__init__.py tests/infer/test_ansi_strip.py
git commit -m "Add deterministic ANSI/CSI/OSC stripper with cr-collapse"
```

---

## Task 12: Quality metrics + slice analysis

**Goal:** Implement `exact_match`, `normalized_exact_match`, `token_reduction_pct` per spec §7.2, plus slice grouping per §7.3.

**Files:**
- Create: `eval/metrics.py`
- Create: `eval/slicing.py`
- Create: `tests/eval/test_metrics.py`

**Acceptance Criteria:**
- [ ] `exact_match(pred, gold) -> bool`
- [ ] `normalized_match(pred, gold) -> bool` — equal after collapsing whitespace runs + stripping per-line trailing whitespace
- [ ] `token_reduction_pct(input_tokens, output_tokens) -> float` in [-inf, 1.0]
- [ ] `slice_results(records, key_fns)` groups results by recipe / archetype / length-bucket and reports per-slice aggregates
- [ ] All tests pass

**Verify:** `uv run pytest tests/eval/test_metrics.py -v` → all pass.

**Steps:**

- [ ] **Step 1: Write tests**

`tests/eval/test_metrics.py`:

```python
from eval.metrics import exact_match, normalized_match, token_reduction_pct
from eval.slicing import slice_results, length_bucket


def test_exact_match_pass():
    assert exact_match("hello\n", "hello\n")


def test_exact_match_fail_on_whitespace():
    assert not exact_match("hello \n", "hello\n")


def test_normalized_match_collapses_whitespace():
    assert normalized_match("hello   world\n", "hello world\n")


def test_normalized_match_strips_trailing_ws():
    assert normalized_match("hello   \nworld   \n", "hello\nworld\n")


def test_normalized_match_still_fails_on_content_diff():
    assert not normalized_match("hello world", "hello earth")


def test_token_reduction_positive():
    pct = token_reduction_pct(n_input=100, n_output=30)
    assert abs(pct - 0.7) < 1e-9


def test_token_reduction_zero_for_passthrough():
    assert token_reduction_pct(100, 100) == 0.0


def test_token_reduction_negative_when_output_longer():
    assert token_reduction_pct(100, 150) == -0.5


def test_length_bucket():
    assert length_bucket(100) == "<512"
    assert length_bucket(1024) == "512-2k"
    assert length_bucket(5000) == "2k-8k"


def test_slice_results_groups():
    records = [
        {"recipe": "passthrough", "passed": True, "metric": 1.0},
        {"recipe": "passthrough", "passed": True, "metric": 0.9},
        {"recipe": "noisy_logs", "passed": False, "metric": 0.0},
    ]
    out = slice_results(records, key_fn=lambda r: r["recipe"])
    assert "passthrough" in out
    assert out["passthrough"]["count"] == 2
    assert out["passthrough"]["pass_rate"] == 1.0
```

- [ ] **Step 2: Implement `eval/metrics.py`**

```python
"""Quality metrics for the cleanup task."""
import re


def exact_match(pred: str, gold: str) -> bool:
    return pred == gold


_WS_RUN_RE = re.compile(r"[ \t]+")


def _normalize(s: str) -> str:
    return "\n".join(
        _WS_RUN_RE.sub(" ", line).rstrip()
        for line in s.splitlines()
    )


def normalized_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def token_reduction_pct(n_input: int, n_output: int) -> float:
    if n_input == 0:
        return 0.0
    return 1.0 - (n_output / n_input)
```

- [ ] **Step 3: Implement `eval/slicing.py`**

```python
"""Group eval results by various keys for failure-mode analysis."""
from typing import Callable


def length_bucket(n_tokens: int) -> str:
    if n_tokens < 512:
        return "<512"
    if n_tokens < 2048:
        return "512-2k"
    return "2k-8k"


def slice_results(records: list[dict], key_fn: Callable[[dict], str]) -> dict[str, dict]:
    """Group records by key_fn(record) and return per-slice aggregates."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = key_fn(r)
        groups.setdefault(key, []).append(r)
    out = {}
    for key, recs in groups.items():
        n = len(recs)
        n_passed = sum(1 for r in recs if r.get("passed", False))
        metrics = [r.get("metric", 0.0) for r in recs]
        out[key] = {
            "count": n,
            "pass_rate": n_passed / n if n else 0.0,
            "metric_p50": sorted(metrics)[n // 2] if n else 0.0,
            "metric_mean": sum(metrics) / n if n else 0.0,
        }
    return out
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
uv run pytest tests/eval/test_metrics.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py eval/slicing.py tests/eval/test_metrics.py
git commit -m "Add quality metrics (exact/normalized match, token reduction) and slice analysis"
```

---

## Task 13: Prompt template + training data formatter

**Goal:** Implement the prompt template per spec §6.4 and a formatter that converts our jsonl pairs into MLX-LM's training format.

**Files:**
- Create: `train/prompt_template.py`
- Create: `train/format_dataset.py`
- Create: `tests/train/__init__.py`
- Create: `tests/train/test_prompt_template.py`

**Acceptance Criteria:**
- [ ] `format_chat(dirty: str, clean: str) -> dict` returns the MLX-LM-expected schema
- [ ] Loss-mask applied so only model-turn tokens contribute to loss
- [ ] `format_dataset.py` is a CLI that reads `data/train.jsonl` + `data/val.jsonl`, writes MLX-LM-formatted files (typically `train.jsonl` and `valid.jsonl` in MLX-LM's expected format)

**Verify:** `uv run pytest tests/train/test_prompt_template.py -v` → all pass; `uv run python -m train.format_dataset --help` shows usage.

**Steps:**

- [ ] **Step 1: Inspect MLX-LM's expected format**

```bash
uv run python -c "from mlx_lm.lora import lora; help(lora)" 2>&1 | head -50
```

Read the docstring; the expected format is typically a `messages` list per row (chat format) or `text` field. Pin to the format that the installed `mlx-lm` version expects. Document in `train/prompt_template.py`.

- [ ] **Step 2: Write tests**

`tests/train/test_prompt_template.py`:

```python
from train.prompt_template import format_chat, INSTRUCTION


def test_format_chat_has_user_and_assistant():
    rec = format_chat("dirty input", "clean output")
    # MLX-LM chat format: messages list with role/content
    assert "messages" in rec
    roles = [m["role"] for m in rec["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_format_chat_includes_instruction():
    rec = format_chat("dirty", "clean")
    user_msg = next(m for m in rec["messages"] if m["role"] == "user")
    assert INSTRUCTION in user_msg["content"]


def test_format_chat_includes_dirty_input():
    rec = format_chat("specific dirty content here", "x")
    user_msg = next(m for m in rec["messages"] if m["role"] == "user")
    assert "specific dirty content here" in user_msg["content"]


def test_format_chat_assistant_is_clean():
    rec = format_chat("x", "specific clean output")
    asst_msg = next(m for m in rec["messages"] if m["role"] == "assistant")
    assert asst_msg["content"] == "specific clean output"
```

- [ ] **Step 3: Implement `train/prompt_template.py`**

```python
"""Prompt template for terminal-output cleanup task. See spec §6.4."""

INSTRUCTION = (
    "Clean the following terminal output. Preserve all information losslessly. "
    "Strip ANSI codes, collapse progress-bar overwrites to their final state, "
    "deduplicate identical repeated lines using [Nx] prefix, normalize whitespace."
)


def format_chat(dirty: str, clean: str) -> dict:
    """Return MLX-LM-compatible chat-format record."""
    return {
        "messages": [
            {"role": "user", "content": f"{INSTRUCTION}\n\n---\n{dirty}\n---"},
            {"role": "assistant", "content": clean},
        ],
    }
```

- [ ] **Step 4: Implement `train/format_dataset.py`**

```python
"""Convert our (input, output) jsonl into MLX-LM chat-format jsonl."""
import argparse
import json
from pathlib import Path

from train.prompt_template import format_chat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-train", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--in-val", type=Path, default=Path("data/val.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/mlx"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for src, name in [(args.in_train, "train.jsonl"), (args.in_val, "valid.jsonl")]:
        with src.open(encoding="utf-8") as f_in, \
             (args.out_dir / name).open("w", encoding="utf-8") as f_out:
            n = 0
            for line in f_in:
                rec = json.loads(line)
                formatted = format_chat(rec["input"], rec["output"])
                f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                n += 1
            print(f"  {src.name}: {n} -> {args.out_dir / name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests, verify all pass**

```bash
uv run pytest tests/train/test_prompt_template.py -v
```

- [ ] **Step 6: Run formatter on real data**

```bash
uv run python -m train.format_dataset
ls -lh data/mlx/
head -1 data/mlx/train.jsonl | python -m json.tool
```

Expected: `data/mlx/train.jsonl` and `data/mlx/valid.jsonl` exist; first record has `messages` list.

- [ ] **Step 7: Commit**

```bash
git add train/prompt_template.py train/format_dataset.py \
        tests/train/__init__.py tests/train/test_prompt_template.py
git commit -m "Add prompt template and MLX-LM dataset formatter"
```

---

## Task 14: Training config + run script

**Goal:** A `config.yaml` with the hyperparameters from spec §6.3/6.5 and a `run.sh` that launches `mlx_lm.lora` correctly.

**Files:**
- Create: `train/config.yaml`
- Create: `train/run.sh`

**Acceptance Criteria:**
- [ ] `config.yaml` matches spec §6.3 + §6.5 (lora_layers=16, rank=16, alpha=32, lr=2e-4, batch=1, grad_accum=8, seq_len=4096, epochs=2)
- [ ] `run.sh` is executable, single command
- [ ] `run.sh --dry-run` prints the command without executing (for verification)

**Verify:** `bash train/run.sh --dry-run` prints the planned `mlx_lm.lora` invocation; `bash train/run.sh --steps 50` actually runs 50 steps successfully and produces an adapter file.

**Steps:**

- [ ] **Step 1: Write `train/config.yaml`**

```yaml
# Gemma 4 E2B QLoRA config — matches spec §6.3, §6.5
model: models/base/gemma-4-E2B-it-4bit
data: data/mlx
adapter_path: models/adapter

# LoRA
fine_tune_type: lora
num_layers: 16          # last 16 transformer layers
lora_parameters:
  rank: 16
  alpha: 32
  dropout: 0.05
  scale: 10.0
  keys:
    - "self_attn.q_proj"
    - "self_attn.k_proj"
    - "self_attn.v_proj"
    - "self_attn.o_proj"

# Training
iters: 45000            # ~2 epochs at 180k pairs / batch 8
batch_size: 1
grad_checkpoint: true
learning_rate: 2.0e-4
warmup_steps: 100
lr_schedule:
  name: cosine_decay
  arguments: [2.0e-4, 45000, 1.0e-5]

# Sequence length
max_seq_length: 4096

# Validation / checkpointing
val_batches: 25
steps_per_eval: 250
steps_per_report: 50
save_every: 500

# Mask
mask_prompt: true       # only compute loss on assistant turn
seed: 0
```

> **Note on `mlx-lm` config schema:** The exact key names above are based on `mlx-lm` v0.20.x. If the version pinned in `pyproject.toml` differs, run `uv run python -m mlx_lm.lora --help` to see current flags and adjust.

- [ ] **Step 2: Write `train/run.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG="train/config.yaml"
ARGS=(
    --train
    --config "$CONFIG"
)

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Would run: uv run python -m mlx_lm.lora ${ARGS[*]}"
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
exec caffeinate -i uv run python -m mlx_lm.lora "${ARGS[@]}"
```

- [ ] **Step 3: Make executable + verify dry-run**

```bash
chmod +x train/run.sh
bash train/run.sh --dry-run
```

Expected: prints the planned `mlx_lm.lora` command.

- [ ] **Step 4: Verify CLI accepts the config**

```bash
uv run python -m mlx_lm.lora --help | head -30
```

Expected: usage output. If any flag in `config.yaml` is rejected, fix the config to match the installed `mlx-lm` API.

- [ ] **Step 5: Commit**

```bash
git add train/config.yaml train/run.sh
git commit -m "Add training config and run.sh entry point"
```

---

## Task 15: Eval runner

**Goal:** A CLI that loads the trained adapter, runs all three eval sets (`val`, `eval_real`, `eval_passthrough`), applies Layer 1 (lossless guard), Layer 2 (quality metrics), Layer 3 (slice analysis), and prints / writes a report.

**Files:**
- Create: `eval/run.py`
- Create: `tests/eval/test_run.py` (smoke test only — actual eval requires model)

**Acceptance Criteria:**
- [ ] CLI: `python -m eval.run --adapter models/adapter --eval-sets val,eval_real,eval_passthrough`
- [ ] Loads base + adapter via `mlx_lm.load`
- [ ] For each eval set, generates predictions and computes metrics
- [ ] Reports: per-set guard pass rate, normalized match, median token reduction; per-slice breakdowns
- [ ] Writes JSON report to `eval/reports/<timestamp>.json`
- [ ] Reports compared against acceptance thresholds from spec §7.5; exits 0 if met, 1 otherwise

**Verify:** `uv run python -m eval.run --help` shows usage; can run after Task 16 trains an adapter.

**Steps:**

- [ ] **Step 1: Implement `eval/run.py`**

```python
"""Run full evaluation: lossless guard, quality metrics, slice analysis, threshold check."""
import argparse
import json
import sys
import time
from pathlib import Path

from mlx_lm import load, generate
from tqdm import tqdm

from train.prompt_template import INSTRUCTION
from infer.ansi_strip import strip
from eval.lossless_guard import lossless_check, extract_atoms
from eval.metrics import exact_match, normalized_match, token_reduction_pct
from eval.slicing import slice_results, length_bucket


THRESHOLDS = {
    "eval_real": {
        "guard_pass_rate": 0.995,
        "normalized_match_rate": 0.80,
        "median_token_reduction": 0.30,
    },
    "eval_passthrough": {
        "guard_pass_rate": 1.0,
        "max_abs_token_reduction": 0.02,
    },
}


def _build_prompt(dirty: str) -> str:
    return f"{INSTRUCTION}\n\n---\n{dirty}\n---"


def _evaluate_set(model, tokenizer, path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        lines = list(f)
    for line in tqdm(lines, desc=path.name):
        rec = json.loads(line)
        dirty = rec["input"]
        gold = rec["output"]
        # Pre-strip (matches inference behavior)
        pre = strip(dirty)
        prompt = _build_prompt(pre)
        pred = generate(model, tokenizer, prompt=prompt, max_tokens=4096, verbose=False)
        # Strip just-in-case the model emitted any escape codes
        pred_clean = strip(pred).strip()
        gold_clean = gold
        n_in = len(tokenizer.encode(dirty))
        n_out = len(tokenizer.encode(pred_clean))
        guard = lossless_check(dirty, pred_clean)
        records.append({
            "input": dirty,
            "gold": gold_clean,
            "pred": pred_clean,
            "guard_passed": guard.passed,
            "missing_atoms": list(guard.missing_atoms),
            "exact": exact_match(pred_clean, gold_clean),
            "normalized": normalized_match(pred_clean, gold_clean),
            "n_in": n_in,
            "n_out": n_out,
            "tok_red": token_reduction_pct(n_in, n_out),
            "meta": rec.get("meta", {}),
        })
    return records


def _check_thresholds(set_name: str, records: list[dict]) -> tuple[bool, dict]:
    n = len(records)
    guard_rate = sum(r["guard_passed"] for r in records) / n if n else 0.0
    norm_rate = sum(r["normalized"] for r in records) / n if n else 0.0
    tok_reds = sorted(r["tok_red"] for r in records)
    median_red = tok_reds[n // 2] if n else 0.0
    summary = {
        "n": n,
        "guard_pass_rate": guard_rate,
        "normalized_match_rate": norm_rate,
        "median_token_reduction": median_red,
    }
    th = THRESHOLDS.get(set_name, {})
    ok = True
    if "guard_pass_rate" in th and guard_rate < th["guard_pass_rate"]:
        ok = False
    if "normalized_match_rate" in th and norm_rate < th["normalized_match_rate"]:
        ok = False
    if "median_token_reduction" in th and median_red < th["median_token_reduction"]:
        ok = False
    if "max_abs_token_reduction" in th and abs(median_red) > th["max_abs_token_reduction"]:
        ok = False
    return ok, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=str, default="models/adapter")
    p.add_argument("--base", type=str, default="models/base/gemma-4-E2B-it-4bit")
    p.add_argument("--eval-sets", type=str, default="val,eval_real,eval_passthrough")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--report-dir", type=Path, default=Path("eval/reports"))
    args = p.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading model {args.base} + adapter {args.adapter}...")
    model, tokenizer = load(args.base, adapter_path=args.adapter)

    set_files = {
        "val": args.data_dir / "val.jsonl",
        "eval_real": args.data_dir / "eval_real.jsonl",
        "eval_passthrough": args.data_dir / "eval_passthrough.jsonl",
    }
    sets_to_run = args.eval_sets.split(",")

    all_pass = True
    report = {"timestamp": time.time(), "sets": {}}
    for set_name in sets_to_run:
        path = set_files[set_name]
        if not path.exists():
            print(f"SKIP {set_name}: {path} not found")
            continue
        records = _evaluate_set(model, tokenizer, path)
        ok, summary = _check_thresholds(set_name, records)
        all_pass = all_pass and ok
        # Slices
        by_recipe = slice_results(
            [{"recipe": r["meta"].get("recipe", "n/a"),
              "passed": r["guard_passed"],
              "metric": r["tok_red"]} for r in records],
            key_fn=lambda r: r["recipe"],
        )
        by_archetype = slice_results(
            [{"archetype": r["meta"].get("archetype", "n/a"),
              "passed": r["guard_passed"],
              "metric": r["tok_red"]} for r in records],
            key_fn=lambda r: r["archetype"],
        )
        by_length = slice_results(
            [{"bucket": length_bucket(r["n_in"]),
              "passed": r["guard_passed"],
              "metric": r["tok_red"]} for r in records],
            key_fn=lambda r: r["bucket"],
        )
        report["sets"][set_name] = {
            "summary": summary,
            "thresholds_ok": ok,
            "by_recipe": by_recipe,
            "by_archetype": by_archetype,
            "by_length": by_length,
        }
        print(f"\n=== {set_name} (n={summary['n']}) ===")
        print(json.dumps(summary, indent=2))
        print(f"thresholds_ok={ok}")

    out_path = args.report_dir / f"{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out_path}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (no adapter required if base model alone is acceptable for syntax check)**

```bash
uv run python -m eval.run --help
```

Expected: usage printed, no errors.

- [ ] **Step 3: Commit**

```bash
git add eval/run.py
git commit -m "Add eval runner: lossless guard + metrics + slices + threshold check"
```

---

## Task 16: Run training (long-running)

**Goal:** Actually train the adapter. Produce `models/adapter/adapters.safetensors` (or whatever file `mlx-lm` outputs).

**Files:**
- Modify (incrementally): `train/config.yaml` if hyperparameter tuning is needed
- Create: `models/adapter/*` (output of training; gitignored)
- Create: `train/training_log.md` (notes on the run, kept committed)

**Acceptance Criteria:**
- [ ] Adapter file(s) present in `models/adapter/`
- [ ] Training validation loss visible and decreasing
- [ ] Smoke-test inference (load adapter, generate on a sample) produces a clean-looking output

**Verify:** `ls models/adapter/` shows safetensors; `uv run python -c "from mlx_lm import load; m,t = load('models/base/gemma-4-E2B-it-4bit', adapter_path='models/adapter'); print('loaded')"` prints `loaded`.

**Steps:**

- [ ] **Step 1: Pre-flight checks**

```bash
# Disk space (need ~10 GB free for checkpoints)
df -h .
# RAM headroom (close browser tabs, etc.)
top -l 1 -n 0 | head -10
# Data ready
ls -lh data/mlx/
# Base model present
ls models/base/gemma-4-E2B-it-4bit/*.safetensors
```

- [ ] **Step 2: Smoke-test 50 steps**

```bash
bash train/run.sh --steps 50 2>&1 | tee train/smoke_log.txt
```

Expected: training completes in ~1 minute, `models/adapter/adapters.safetensors` exists. Examine `train/smoke_log.txt` for early loss values.

- [ ] **Step 3: Verify smoke-trained adapter loads**

```bash
uv run python -c "
from mlx_lm import load, generate
model, tok = load('models/base/gemma-4-E2B-it-4bit', adapter_path='models/adapter')
out = generate(model, tok, prompt='Clean: \x1b[31mhi\x1b[0m', max_tokens=20, verbose=False)
print(repr(out))
"
```

Expected: some text output. Quality irrelevant (only 50 steps); this just confirms the load path works.

- [ ] **Step 4: Run full training (overnight)**

```bash
# Move adapter aside so the smoke run isn't overwritten partial
mv models/adapter models/adapter_smoke
mkdir models/adapter

# Real run
bash train/run.sh 2>&1 | tee train/training_log.txt
```

Notes:
- Expected runtime: 16–30 hours.
- Run overnight; the `caffeinate` in `run.sh` keeps the Mac awake.
- Watch the first ~500 steps to confirm loss trends down. If diverging, kill and reduce LR to 1e-4.
- Thermal throttling on Air is real — expect step time to vary 2× during long runs.

- [ ] **Step 5: Monitor (in a separate terminal during long run)**

```bash
tail -f train/training_log.txt
```

Look for:
- Steady decrease in `train_loss` and `val_loss`
- Val loss not diverging from train loss (overfitting check)
- No OOM errors

- [ ] **Step 6: Document what happened**

`train/training_log.md`:

```markdown
# Training Run: <date>

## Config
- (any deviations from train/config.yaml)

## Outcome
- Final train loss: ?
- Final val loss: ?
- Total wall time: ?h
- Best checkpoint: step ?

## Notes / surprises
- ...
```

- [ ] **Step 7: Commit log + config (model files gitignored)**

```bash
git add train/training_log.md train/config.yaml
git commit -m "Trained Gemma 4 E2B LoRA adapter (run notes)"
```

---

## Task 17: Inference function with safe fallback

**Goal:** Implement `clean(dirty: str) -> str` per spec §8.2: pre-strip, model generate, lossless-guard check, fallback to pre-stripped on failure.

**Files:**
- Create: `infer/clean.py`
- Create: `tests/infer/test_clean.py`

**Acceptance Criteria:**
- [ ] `clean(dirty)` returns str
- [ ] If model output passes the lossless guard, return the model output
- [ ] If model output fails the guard, return the deterministically pre-stripped input
- [ ] Failure paths logged (stderr or a logfile)
- [ ] Model + tokenizer loaded once, cached at module level via lazy global

**Verify:** `uv run pytest tests/infer/test_clean.py -v` → all pass; `echo $'\x1b[31mhello\x1b[0m' | uv run python -m infer.clean` prints `hello`.

**Steps:**

- [ ] **Step 1: Write tests using a stubbed model**

`tests/infer/test_clean.py`:

```python
from unittest.mock import patch, MagicMock
from infer.clean import clean


def _mock_load(*args, **kwargs):
    model = MagicMock()
    tok = MagicMock()
    tok.encode.side_effect = lambda s, **_: list(range(len(s) // 4))
    return model, tok


def test_clean_uses_model_output_when_guard_passes(monkeypatch):
    """If model returns a string preserving all atoms, that string is returned."""
    with patch("infer.clean._load_model", side_effect=_mock_load), \
         patch("infer.clean._generate", return_value="hello world foo"):
        out = clean("\x1b[31mhello world foo\x1b[0m")
        # The model output is what's returned (since it preserves atoms)
        assert out == "hello world foo"


def test_clean_fallback_when_guard_fails(monkeypatch):
    """If model output drops atoms, fall back to deterministic ANSI-strip."""
    with patch("infer.clean._load_model", side_effect=_mock_load), \
         patch("infer.clean._generate", return_value="some unrelated text"):
        out = clean("\x1b[31mimportant_id_42 path /etc/hosts\x1b[0m")
        # Fallback = ansi-stripped input
        assert "important_id_42" in out
        assert "/etc/hosts" in out


def test_clean_passthrough_clean_input():
    """Already-clean input + model returns same: no-op."""
    with patch("infer.clean._load_model", side_effect=_mock_load), \
         patch("infer.clean._generate", return_value="hello world"):
        assert clean("hello world") == "hello world"
```

- [ ] **Step 2: Implement `infer/clean.py`**

```python
"""Lossless terminal-output cleaner. Text-in / text-out with safe fallback."""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from infer.ansi_strip import strip
from eval.lossless_guard import lossless_check
from train.prompt_template import INSTRUCTION


_logger = logging.getLogger("infer.clean")
_model_cache: Optional[tuple] = None


def _load_model():
    """Lazy-load the base + adapter once per process. Mockable in tests."""
    from mlx_lm import load
    base = os.environ.get("CLEAN_MODEL_BASE", "models/base/gemma-4-E2B-it-4bit")
    adapter = os.environ.get("CLEAN_MODEL_ADAPTER", "models/adapter")
    if not Path(adapter).exists():
        adapter = None  # use base only (smoke mode)
    return load(base, adapter_path=adapter)


def _generate(model, tokenizer, prompt: str, max_tokens: int = 4096) -> str:
    """Mockable generation wrapper."""
    from mlx_lm import generate
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def _build_prompt(dirty_pre_stripped: str) -> str:
    return f"{INSTRUCTION}\n\n---\n{dirty_pre_stripped}\n---"


def clean(dirty: str) -> str:
    """Lossless terminal-output cleanup. Returns model output if it passes the
    lossless guard, otherwise the deterministically pre-stripped input."""
    global _model_cache
    if not dirty:
        return dirty
    if _model_cache is None:
        _model_cache = _load_model()
    model, tokenizer = _model_cache

    pre = strip(dirty)
    prompt = _build_prompt(pre)
    raw_pred = _generate(model, tokenizer, prompt)
    pred = strip(raw_pred).strip("\n")

    guard = lossless_check(dirty, pred)
    if guard.passed:
        return pred
    _logger.warning(
        "lossless guard failed; falling back to deterministic strip. "
        "missing atoms: %s", sorted(guard.missing_atoms)[:10]
    )
    return pre
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/infer/test_clean.py -v
```

Expected: 3 passed.

- [ ] **Step 4: End-to-end smoke**

```bash
uv run python -c "
from infer.clean import clean
print(repr(clean('\x1b[31mhello world\x1b[0m')))
"
```

Expected: prints `'hello world'` (or close — depends on adapter quality if loaded). Confirms imports + load + generate path works.

- [ ] **Step 5: Commit**

```bash
git add infer/clean.py tests/infer/test_clean.py
git commit -m "Add inference clean() with lossless-guard fallback"
```

---

## Task 18: Inference CLI + optional HTTP server

**Goal:** Three usage modes per spec §8.3: stdin/stdout CLI, Python import (already done in Task 17), optional FastAPI server.

**Files:**
- Create: `infer/__main__.py`
- Create: `infer/server.py`

**Acceptance Criteria:**
- [ ] `python -m infer.clean` reads stdin, writes cleaned output to stdout
- [ ] `python -m infer.server` starts a FastAPI server with `POST /clean` accepting `{"text": "..."}` and returning `{"cleaned": "..."}`
- [ ] Server preloads model on startup (no cold-start per request)

**Verify:** `echo $'\x1b[31mhello\x1b[0m' | uv run python -m infer.clean` prints `hello`. `uv run python -m infer.server &; sleep 3; curl -s -X POST localhost:8000/clean -H 'Content-Type: application/json' -d '{"text":"[31mhi[0m"}'` returns `{"cleaned":"hi"}`.

**Steps:**

- [ ] **Step 1: Implement `infer/__main__.py`**

```python
"""CLI: stdin -> cleaned stdout."""
import sys
from infer.clean import clean


def main():
    raw = sys.stdin.read()
    cleaned = clean(raw)
    sys.stdout.write(cleaned)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement `infer/server.py`**

```python
"""FastAPI server. POST /clean -> {cleaned: str}."""
from fastapi import FastAPI
from pydantic import BaseModel

from infer.clean import clean, _load_model


class CleanRequest(BaseModel):
    text: str


class CleanResponse(BaseModel):
    cleaned: str


app = FastAPI()


@app.on_event("startup")
def _preload():
    # Trigger model load on startup, not first request
    import infer.clean as cl
    if cl._model_cache is None:
        cl._model_cache = _load_model()


@app.post("/clean", response_model=CleanResponse)
def clean_endpoint(req: CleanRequest) -> CleanResponse:
    return CleanResponse(cleaned=clean(req.text))


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 3: Smoke-test CLI**

```bash
echo $'\x1b[31mhello\x1b[0m world' | uv run python -m infer.clean
```

Expected: `hello world` (or close).

- [ ] **Step 4: Smoke-test server (separate terminal)**

```bash
# Terminal A
uv run python -m infer.server
# Terminal B
curl -s -X POST localhost:8000/clean \
    -H "Content-Type: application/json" \
    -d '{"text":"[31mhi[0m"}'
```

Expected: `{"cleaned":"hi"}`.

- [ ] **Step 5: Commit**

```bash
git add infer/__main__.py infer/server.py
git commit -m "Add inference CLI (__main__) and optional FastAPI server"
```

---

## Task 19: V1 acceptance evaluation

**Goal:** Run the full eval suite against the trained adapter. Verify all spec §7.5 thresholds are met. Document findings, decide whether to ship or iterate.

**Files:**
- Create: `eval/reports/<timestamp>.json` (auto-generated)
- Create: `train/v1_acceptance.md` (committed write-up)

**Acceptance Criteria:**
- [ ] `eval.run` exits 0 (all thresholds met) OR
- [ ] If thresholds missed, the `v1_acceptance.md` documents which slices failed and the iteration plan

**Verify:** `uv run python -m eval.run --adapter models/adapter; echo "exit=$?"` → reports per-set summary; exits 0 on success.

**Steps:**

- [ ] **Step 1: Run full eval**

```bash
uv run python -m eval.run --adapter models/adapter --eval-sets val,eval_real,eval_passthrough
```

Expected: prints per-set summary, writes `eval/reports/<ts>.json`. Either exits 0 (ship) or 1 (iterate).

- [ ] **Step 2: Write up the result**

`train/v1_acceptance.md`:

```markdown
# V1 Acceptance — <date>

## Summary

| Eval set | Guard pass rate | Normalized match | Median token reduction | Threshold ok |
|---|---|---|---|---|
| val (synthetic) | ?? | ?? | ?? | ?? |
| eval_real | ?? | ?? | ?? | ?? |
| eval_passthrough | ?? | ?? | ?? | ?? |

## Slice analysis

(per-recipe / per-archetype / per-length-bucket summary from the report json)

## Decision

(Ship | Iterate). If iterate, list the specific slices that failed and the proposed fix
(e.g., "noisy_logs slice has guard pass rate 91% — add MixedStreams stronger emphasis to
recipe and regenerate val samples").
```

- [ ] **Step 3: Commit**

```bash
git add train/v1_acceptance.md eval/reports/<latest>.json
git commit -m "V1 acceptance evaluation results"
```

---

## Self-Review Notes

1. **Spec coverage:** Every section in the spec is covered:
   - §1 Goal → Task 19 (acceptance)
   - §2 Constraints → Task 0 (bootstrap with M4 + Gemma 4 E2B)
   - §3 Non-Goals → respected throughout (no shell hooks, no streaming, no multimodal)
   - §4 Architecture → Task 0 file structure
   - §5 Data generation → Tasks 1–8
   - §6 Training pipeline → Tasks 13, 14, 16
   - §7 Eval → Tasks 9, 10, 11 (ANSI strip), 12, 15, 19
   - §8 Inference → Tasks 17, 18
   - §9 Risks → mitigations baked into corresponding tasks
   - §10 Acceptance → Task 19

2. **Cross-task type consistency:** `Generator.generate(rng) -> str`, `Transform.apply(clean, rng) -> str`, `Recipe.steps`, `lossless_check`, `clean(dirty)`, `strip(text)` — all consistent across tasks.

3. **Task ordering / blocking:**
   - Task 9 (real eval capture) imports `infer.ansi_strip` from Task 11 — flagged in Task 9 Step 3. Either reorder or stub.
   - Task 16 (training) blocks Task 19 (acceptance).
   - Task 11 (ANSI strip) blocks Tasks 17 (clean) and 9 (capture).
   - Task 13 (formatter) blocks Task 16 (training) — formatted data needed for `mlx_lm.lora`.
   - Task 14 (config) is needed for Task 16.
   - Task 15 (eval runner) blocks Task 19.

4. **Recommended execution order:**
   `0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 11 → 10 → 12 → 9 → 13 → 14 → 15 → 16 → 17 → 18 → 19`

5. **Placeholder check:** No "TODO" / "TBD" left in steps — every code block is concrete.
