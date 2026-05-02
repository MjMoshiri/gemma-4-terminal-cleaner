import random
import re

import pytest

from dirtifier.transforms.ansi import AnsiColor, AnsiBold, AnsiUnderline, AnsiOsc, Hyperlink


# Match all ANSI/CSI/OSC sequences for round-trip stripping
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\\")


def _strip(s: str) -> str:
    return _ANSI_RE.sub("", s)


@pytest.fixture(params=[AnsiColor, AnsiBold, AnsiUnderline, AnsiOsc, Hyperlink])
def transform_cls(request):
    return request.param


def test_ansi_deterministic(transform_cls):
    t = transform_cls()
    a = t.apply("hello world\nfoo bar\n", random.Random(0))
    b = t.apply("hello world\nfoo bar\n", random.Random(0))
    assert a == b


def test_ansi_strips_back_to_clean(transform_cls):
    t = transform_cls()
    clean = "hello world\nfoo bar\n"
    dirty = t.apply(clean, random.Random(7))
    assert _strip(dirty) == clean


def test_ansi_handles_empty(transform_cls):
    t = transform_cls()
    assert t.apply("", random.Random(0)) == ""


def test_ansi_color_actually_adds_color_code():
    t = AnsiColor()
    out = t.apply("hello world this is several words for sure here\n", random.Random(0))
    assert re.search(r"\x1b\[[0-9;]+m", out)


def test_hyperlink_adds_osc8():
    t = Hyperlink()
    out = t.apply("see https://example.com here\n", random.Random(0))
    assert "\x1b]8;" in out
