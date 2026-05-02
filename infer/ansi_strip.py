"""Deterministic terminal-noise stripping. Used as inference pre-step + safe fallback."""
import re


# Order matters: OSC 8 (hyperlinks) need special handling to preserve visible text
_OSC8_RE = re.compile(r"\x1b\]8;[^;]*;[^\x07\x1b]*(?:\x07|\x1b\\)([^\x1b]*)\x1b\]8;;(?:\x07|\x1b\\)")
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_OTHER_ESC_RE = re.compile(r"\x1b[@-_]")


def strip(text: str) -> str:
    if not text:
        return text
    # 1. OSC 8 hyperlinks: keep only visible text
    text = _OSC8_RE.sub(lambda m: m.group(1), text)
    # 2. Other OSC sequences: drop entirely
    text = _OSC_RE.sub("", text)
    # 3. CSI sequences (colors, cursor movement, etc.)
    text = _CSI_RE.sub("", text)
    # 4. Other ESC-* short sequences
    text = _OTHER_ESC_RE.sub("", text)
    # 5. \r-collapse per logical line: keep final segment after last \r
    text = text.replace("\r\n", "\n")  # CRLF → LF before \r-collapse so Windows line endings don't lose content
    out_lines = []
    for line in text.split("\n"):
        segments = line.split("\r")
        out_lines.append(segments[-1])
    text = "\n".join(out_lines)
    # 6. Drop BEL and NUL
    text = text.replace("\x07", "").replace("\x00", "")
    return text
