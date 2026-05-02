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
