"""Generate training pairs by applying dirtifier recipes to clean corpus."""
import argparse
import json
import random
from pathlib import Path

from mlx_lm.tokenizer_utils import load as load_tokenizer
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
    p = argparse.ArgumentParser(
        description="Generate dirty/clean training pairs from corpus JSONL files."
    )
    p.add_argument("--corpus-dir", type=Path, default=Path("corpus/output"))
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument(
        "--n-pairs-per-clean",
        type=int,
        default=2,
        help="How many dirty variants to produce per clean sample",
    )
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tokenizer-path",
        type=str,
        default="models/base/gemma-4-E2B-it-4bit",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_recipes, val_recipes = _pick_split_recipes(list(RECIPES.keys()), HELD_OUT_FOR_VAL)
    train_recipe_dict = _filtered_recipes(train_recipes)
    val_recipe_dict = _filtered_recipes(val_recipes)

    # Load tokenizer for token-count gating.
    # Use mlx_lm.tokenizer_utils.load (takes a Path) to avoid the strict model
    # weight loading in mlx_lm.load which fails with the 4-bit KV-shared weights.
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = load_tokenizer(Path(args.tokenizer_path))
    print("Tokenizer loaded.")

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
                        # Token-count gate: drop oversized examples, never truncate.
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
