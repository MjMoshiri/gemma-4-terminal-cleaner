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
