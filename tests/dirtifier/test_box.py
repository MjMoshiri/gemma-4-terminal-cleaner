import random
from dirtifier.transforms.box import BoxDrawing


def test_box_deterministic():
    t = BoxDrawing()
    a = t.apply("foo\nbar\n", random.Random(0))
    b = t.apply("foo\nbar\n", random.Random(0))
    assert a == b


def test_box_uses_box_chars():
    t = BoxDrawing()
    out = t.apply("alpha\nbeta\ngamma\n", random.Random(0))
    assert "│" in out or "─" in out or "├" in out or "└" in out or "┌" in out or "┐" in out


def test_box_preserves_inner_text():
    t = BoxDrawing()
    out = t.apply("alpha\nbeta\ngamma\n", random.Random(0))
    # Original text is still present (substring) in each line
    assert "alpha" in out
    assert "beta" in out
    assert "gamma" in out


def test_box_handles_empty():
    assert BoxDrawing().apply("", random.Random(0)) == ""
