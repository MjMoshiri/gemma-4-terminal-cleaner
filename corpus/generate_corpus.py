"""Generate clean corpus across all archetypes. CLI entry point."""
import argparse
import json
import random
import zlib
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
                # Per-generator deterministic seed derived with stable zlib.crc32
                # (avoids Python's randomized hash() which varies per-process)
                seed = args.seed * 1_000_003 + zlib.crc32(src.encode("utf-8"))
                rng = random.Random(seed)
                for i in range(samples_per_gen):
                    text = gen.generate(rng)
                    rec = {"archetype": archetype, "src": f"{src}/{i}", "text": text}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {archetype}: {out_path} ({samples_per_gen * len(gens)} samples)")


if __name__ == "__main__":
    main()
