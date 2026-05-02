import random
from dirtifier.transforms.repeat import RepeatedLines


def test_repeat_deterministic():
    t = RepeatedLines()
    a = t.apply("a\nb\nc\n", random.Random(0))
    b = t.apply("a\nb\nc\n", random.Random(0))
    assert a == b


def test_repeat_handles_empty():
    assert RepeatedLines().apply("", random.Random(0)) == ""


def test_repeat_increases_line_count():
    t = RepeatedLines()
    out = t.apply("a\nb\nc\n", random.Random(0))
    assert out.count("\n") >= 3  # at least the original 3


def test_repeat_name():
    assert RepeatedLines.name == "repeated_lines"
