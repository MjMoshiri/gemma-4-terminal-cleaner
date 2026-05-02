import random
import re
from dirtifier.transforms.base import Transform


# Foreground colors 30-37 + bright 90-97
_FG_CODES = list(range(30, 38)) + list(range(90, 98))
# Background 40-47 + bright 100-107
_BG_CODES = list(range(40, 48)) + list(range(100, 108))
_RESET = "\x1b[0m"


def _wrap(text: str, codes: list[int]) -> str:
    if not text:
        return text
    seq = "\x1b[" + ";".join(str(c) for c in codes) + "m"
    return seq + text + _RESET


class AnsiColor(Transform):
    """Wrap random tokens / lines in random foreground (and sometimes background) colors."""
    name = "ansi_color"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        # Tokenize to whitespace runs vs non-whitespace runs, color random non-ws tokens
        tokens = re.findall(r"\S+|\s+", clean)
        out = []
        for tok in tokens:
            if tok.isspace() or rng.random() > 0.4:
                out.append(tok)
                continue
            codes = [rng.choice(_FG_CODES)]
            if rng.random() < 0.1:
                codes.append(rng.choice(_BG_CODES))
            out.append(_wrap(tok, codes))
        return "".join(out)


class AnsiBold(Transform):
    name = "ansi_bold"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        tokens = re.findall(r"\S+|\s+", clean)
        out = []
        for tok in tokens:
            if tok.isspace() or rng.random() > 0.2:
                out.append(tok)
                continue
            out.append(_wrap(tok, [1]))
        return "".join(out)


class AnsiUnderline(Transform):
    name = "ansi_underline"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        tokens = re.findall(r"\S+|\s+", clean)
        out = []
        for tok in tokens:
            if tok.isspace() or rng.random() > 0.15:
                out.append(tok)
                continue
            out.append(_wrap(tok, [4]))
        return "".join(out)


class AnsiOsc(Transform):
    """Inject an OSC sequence (e.g. set window title) at the start. Visible-text-preserving."""
    name = "ansi_osc"

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        # OSC 0; <title> BEL — sets terminal title; doesn't affect printed text
        title = "build" if rng.random() < 0.5 else "session"
        osc = f"\x1b]0;{title}\x07"
        return osc + clean


class Hyperlink(Transform):
    """Wrap URLs (or random tokens) in OSC-8 hyperlink markers."""
    name = "hyperlink"

    _URL_RE = re.compile(r"https?://\S+")

    def apply(self, clean: str, rng: random.Random) -> str:
        if not clean:
            return clean
        def _wrap_url(m):
            url = m.group(0)
            return f"\x1b]8;;{url}\x07{url}\x1b]8;;\x07"
        # Wrap actual URLs always; with low prob also wrap a random token in a fake link
        out = self._URL_RE.sub(_wrap_url, clean)
        return out
