import random
from abc import ABC, abstractmethod


class Generator(ABC):
    """Generates one clean text artifact per call. Deterministic given seed."""

    archetype: str  # subclasses must set: "table" | "tree" | "list" | "diff" | "log" | "code" | "json"

    @abstractmethod
    def generate(self, rng: random.Random) -> str:
        """Return one clean text sample. No ANSI codes, no terminal artifacts."""
        ...
