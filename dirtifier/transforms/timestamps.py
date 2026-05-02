import random
from dirtifier.transforms.base import Transform


class Timestamps(Transform):
    """Prefix each line with an ISO timestamp."""
    name = "timestamps"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        base_h = rng.randint(0, 23)
        base_m = rng.randint(0, 59)
        out = []
        for line in clean.split("\n"):
            sec = rng.randint(0, 59)
            ms = rng.randint(0, 999)
            ts = f"2026-04-{rng.randint(1,28):02d}T{base_h:02d}:{base_m:02d}:{sec:02d}.{ms:03d}Z"
            out.append(f"{ts} {line}")
        return "\n".join(out)
