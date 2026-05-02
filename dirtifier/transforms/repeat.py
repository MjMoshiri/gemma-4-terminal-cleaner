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
