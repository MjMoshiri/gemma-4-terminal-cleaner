import random
from dirtifier.transforms.base import Transform


class ProgressBar(Transform):
    """Inject a series of \r-overwriting progress-bar frames before the final clean line."""
    name = "progress_bar"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        n_frames = rng.randint(5, 20)
        bar_width = rng.randint(20, 40)
        frames = []
        for i in range(n_frames):
            pct = int((i + 1) / n_frames * 100)
            filled = int((i + 1) / n_frames * bar_width)
            bar = "#" * filled + "-" * (bar_width - filled)
            frames.append(f"[{bar}] {pct}%")
        # All frames overwrite each other on a single line; the final clean line follows
        progression = "\r".join(frames) + "\r"
        return progression + clean


class Spinner(Transform):
    """Inject \r-overwriting spinner frames before the final clean line."""
    name = "spinner"

    _FRAMES = ["|", "/", "-", "\\"]

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        n = rng.randint(8, 32)
        progression = "\r".join(self._FRAMES[i % 4] + " working..." for i in range(n)) + "\r"
        return progression + clean


class CursorMovement(Transform):
    """Inject cursor-up + clear-line escape codes mimicking a TUI redraw."""
    name = "cursor_movement"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        lines = clean.split("\n")
        if len(lines) < 3:
            return clean
        # Pick a midpoint, redraw the previous N lines
        n_redraw = rng.randint(1, min(5, len(lines) - 1))
        # Cursor up N then clear-line, then re-emit those lines
        i = rng.randint(n_redraw, len(lines) - 1)
        # Emit lines up through i, then a redraw block of n_redraw, then the rest
        prefix = "\n".join(lines[:i])
        redraw_block = f"\x1b[{n_redraw}A" + ("\x1b[2K\n" * n_redraw)
        rest = "\n".join(lines[i:])
        return prefix + "\n" + redraw_block + rest
