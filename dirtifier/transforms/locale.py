import re
import random
from dirtifier.transforms.base import Transform


class LocaleVariants(Transform):
    """Inject locale-style thousand separators in numbers."""
    name = "locale_variants"

    _NUM_RE = re.compile(r"\b\d{4,}\b")

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        sep = rng.choice([",", "_", " "])
        def _add_sep(m):
            n = m.group(0)
            # Add separator every 3 digits from the right
            rev = n[::-1]
            chunks = [rev[i:i+3] for i in range(0, len(rev), 3)]
            return sep.join(chunks)[::-1]
        return self._NUM_RE.sub(lambda m: _add_sep(m) if rng.random() < 0.5 else m.group(0), clean)
