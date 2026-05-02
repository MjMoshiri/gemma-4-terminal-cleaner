import random
from dirtifier.transforms.edge import BellChars, NulBytes

_LONG_INPUT = "x" * 5000


def test_bell_deterministic():
    t = BellChars()
    a = t.apply(_LONG_INPUT, random.Random(13))
    b = t.apply(_LONG_INPUT, random.Random(13))
    assert a == b


def test_bell_handles_empty():
    assert BellChars().apply("", random.Random(0)) == ""


def test_bell_name():
    assert BellChars.name == "bell_chars"


def test_nul_deterministic():
    t = NulBytes()
    a = t.apply(_LONG_INPUT, random.Random(17))
    b = t.apply(_LONG_INPUT, random.Random(17))
    assert a == b


def test_nul_handles_empty():
    assert NulBytes().apply("", random.Random(0)) == ""


def test_nul_name():
    assert NulBytes.name == "nul_bytes"
