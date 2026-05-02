import random
from abc import ABC, abstractmethod
from typing import ClassVar


class Transform(ABC):
    """Apply some kind of dirt to clean text. Deterministic given rng seed."""
    name: ClassVar[str]

    @abstractmethod
    def apply(self, clean: str, rng: random.Random) -> str: ...
