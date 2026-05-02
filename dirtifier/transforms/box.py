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
