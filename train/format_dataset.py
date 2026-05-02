"""Convert our (input, output) jsonl into MLX-LM chat-format jsonl with token-cap drop."""
import argparse
import json
from pathlib import Path

from mlx_lm.tokenizer_utils import load as load_tokenizer
from tqdm import tqdm

from train.prompt_template import format_chat


def _count_templated_tokens(tokenizer, formatted_record):
    """Count tokens in the templated form. Prefer apply_chat_template for accuracy."""
    try:
        text = tokenizer.apply_chat_template(formatted_record["messages"], tokenize=False)
        return len(tokenizer.encode(text))
    except Exception:
        # Fallback: approximate with raw content concatenation + 30-token reserve
        msg = formatted_record["messages"]
        concat = msg[0]["content"] + msg[1]["content"]
        return len(tokenizer.encode(concat)) + 30


def main():
    p = argparse.ArgumentParser(
        description="Convert (input, output) jsonl pairs into MLX-LM chat-format jsonl. "
                    "Drops entries whose templated token count exceeds --max-tokens."
    )
    p.add_argument("--in-train", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--in-val", type=Path, default=Path("data/val.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/mlx"))
    p.add_argument(
        "--tokenizer-path",
        type=str,
        default="models/base/gemma-4-E2B-it-4bit",
        help="Path to the tokenizer used for post-template token counting.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum token count (post-template) per entry. Entries exceeding this "
             "are DROPPED (never truncated) to preserve the lossless contract. "
             "Matches MLX-LM trainer max_seq_length (spec §6.5).",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = load_tokenizer(Path(args.tokenizer_path))

    for src, name in [(args.in_train, "train.jsonl"), (args.in_val, "valid.jsonl")]:
        n_in = n_out = n_dropped = 0
        with src.open(encoding="utf-8") as f_in, \
             (args.out_dir / name).open("w", encoding="utf-8") as f_out:
            for line in tqdm(f_in, desc=src.name):
                rec = json.loads(line)
                formatted = format_chat(rec["input"], rec["output"])
                n_in += 1
                n_tok = _count_templated_tokens(tokenizer, formatted)
                if n_tok > args.max_tokens:
                    n_dropped += 1
                    continue
                f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                n_out += 1
        print(
            f"  {src.name}: in={n_in}, out={n_out}, "
            f"dropped={n_dropped} (>{args.max_tokens} tok)"
        )


if __name__ == "__main__":
    main()
