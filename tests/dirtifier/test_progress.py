import random
import re
from dirtifier.transforms.progress import ProgressBar, Spinner, CursorMovement


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)


def _collapse_cr(s: str) -> str:
    """For each line, keep only the final state after \r overwrites."""
    out_lines = []
    for line in s.split("\n"):
        # Each \r resets the line buffer to the segment after the last \r
        segments = line.split("\r")
        out_lines.append(segments[-1])
    return "\n".join(out_lines)


def test_progress_bar_deterministic():
    t = ProgressBar()
    a = t.apply("done\n", random.Random(0))
    b = t.apply("done\n", random.Random(0))
    assert a == b


def test_progress_bar_injects_cr_progression():
    t = ProgressBar()
    out = t.apply("install complete\n", random.Random(0))
    assert "\r" in out
    # After collapse + ansi strip, the final line is the original
    assert _strip_ansi(_collapse_cr(out)).rstrip("\n") == "install complete"


def test_spinner_injects_cr_frames():
    t = Spinner()
    out = t.apply("done\n", random.Random(0))
    assert "\r" in out
    assert _strip_ansi(_collapse_cr(out)).rstrip("\n") == "done"


def test_cursor_movement_injects_escape_codes():
    t = CursorMovement()
    out = t.apply("line a\nline b\nline c\n", random.Random(0))
    # Cursor up = \x1b[<n>A; clear line = \x1b[2K or \x1b[K
    assert re.search(r"\x1b\[\d*[AK]", out) or re.search(r"\x1b\[2K", out)


def test_progress_handles_empty():
    assert ProgressBar().apply("", random.Random(0)) == ""
    assert Spinner().apply("", random.Random(0)) == ""
    assert CursorMovement().apply("", random.Random(0)) == ""
