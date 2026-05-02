"""Build a 200-pair passthrough eval set from already-clean inputs.

Sources:
  - Up to 100 entries from corpus/output/code.jsonl and corpus/output/json.jsonl
  - Remaining entries from real source files in corpus/generators/*.py
Output: data/eval_passthrough.jsonl (200 lines, schema {"input", "output", "meta"})
"""
import json
import random
from pathlib import Path


TARGET = 200
# All available corpus output files to draw synthetic passthrough entries from
CORPUS_FILES = [
    "corpus/output/code.jsonl",
    "corpus/output/json.jsonl",
    "corpus/output/log.jsonl",
    "corpus/output/list.jsonl",
    "corpus/output/table.jsonl",
    "corpus/output/tree.jsonl",
    "corpus/output/diff.jsonl",
]


def main():
    # 200 already-clean inputs: source code, JSON, plain text
    samples = []

    # Pull from real source files first (cat-style) — exhausts quickly (~9 files)
    for path in sorted(Path("corpus/generators").glob("*.py")):
        text = path.read_text()
        samples.append({"input": text, "output": text,
                        "meta": {"command": "real_passthrough", "src": str(path)}})

    # Fill the rest (up to TARGET) from corpus synthetic outputs
    for fname in CORPUS_FILES:
        if len(samples) >= TARGET:
            break
        try:
            with open(fname) as f:
                for line in f:
                    if len(samples) >= TARGET:
                        break
                    rec = json.loads(line)
                    samples.append({"input": rec["text"], "output": rec["text"],
                                    "meta": {"command": "synthetic_passthrough",
                                             "archetype": rec["archetype"]}})
        except FileNotFoundError:
            continue  # skip missing files gracefully

    random.Random(0).shuffle(samples)
    out_path = Path("data/eval_passthrough.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in samples[:TARGET]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples[:TARGET])} passthrough samples")


if __name__ == "__main__":
    main()
