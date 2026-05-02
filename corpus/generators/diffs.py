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
