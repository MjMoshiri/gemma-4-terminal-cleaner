"""Auto-curate captured real outputs using the deterministic strip + lossless guard.

For each raw entry:
  baseline = strip(raw)
  if lossless_check(raw, baseline).passed: accept baseline (output to eval_real.jsonl)
  else: flag for human review (output to eval_real_flagged.jsonl with missing_atoms)
"""
import argparse
import json
from pathlib import Path

from infer.ansi_strip import strip
from eval.lossless_guard import lossless_check


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/eval_real_raw.jsonl"))
    p.add_argument("--accepted-out", type=Path, default=Path("data/eval_real.jsonl"))
    p.add_argument("--flagged-out", type=Path, default=Path("data/eval_real_flagged.jsonl"))
    args = p.parse_args()

    n_total = n_accepted = n_flagged = 0
    args.accepted_out.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open(encoding="utf-8") as f_in, \
         args.accepted_out.open("w", encoding="utf-8") as f_acc, \
         args.flagged_out.open("w", encoding="utf-8") as f_flag:
        for idx, line in enumerate(f_in):
            rec = json.loads(line)
            raw = rec["input"]
            baseline = strip(raw)
            result = lossless_check(raw, baseline)
            n_total += 1
            rec["output"] = baseline
            rec["meta"]["_id"] = idx
            if result.passed:
                rec["meta"]["curator"] = "auto"
                f_acc.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_accepted += 1
            else:
                rec["meta"]["curator"] = "flagged"
                rec["meta"]["missing_atoms"] = sorted(result.missing_atoms)[:20]  # cap to 20 for readability
                rec["meta"]["n_input_atoms"] = result.n_input
                rec["meta"]["n_output_atoms"] = result.n_output
                f_flag.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_flagged += 1

    print(f"Total: {n_total}, accepted: {n_accepted}, flagged: {n_flagged}")
    print(f"Accepted: {args.accepted_out}")
    print(f"Flagged:  {args.flagged_out}")


if __name__ == "__main__":
    main()
