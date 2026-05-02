import random
from dirtifier.transforms.base import Transform


_STDERR_LINES = [
    "warning: unused import",
    "error: connection refused",
    "warning: deprecated function 'foo' used",
    "info: retrying...",
    "debug: state=ready",
]


class MixedStreams(Transform):
    """Interleave fake stderr-style lines into stdout."""
    name = "mixed_streams"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        out = []
        for line in clean.split("\n"):
            out.append(line)
            if rng.random() < 0.05:
                out.append(rng.choice(_STDERR_LINES))
        return "\n".join(out)
