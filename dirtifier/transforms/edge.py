import random
from dirtifier.transforms.base import Transform


class BellChars(Transform):
    """Sprinkle BEL (\\x07) characters."""
    name = "bell_chars"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out = []
        for ch in clean:
            out.append(ch)
            if rng.random() < 0.001:
                out.append("\x07")
        return "".join(out)


class NulBytes(Transform):
    """Sprinkle NUL bytes (\\x00) — rare, but real."""
    name = "nul_bytes"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out = []
        for ch in clean:
            out.append(ch)
            if rng.random() < 0.0005:
                out.append("\x00")
        return "".join(out)
