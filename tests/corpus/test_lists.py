import random
from corpus.generators.lists import FlatListGenerator


def test_list_archetype():
    assert FlatListGenerator().archetype == "list"


def test_list_deterministic():
    g = FlatListGenerator()
    assert g.generate(random.Random(0)) == g.generate(random.Random(0))


def test_list_no_ansi():
    assert "\x1b[" not in FlatListGenerator().generate(random.Random(0))


def test_list_nonempty_lines():
    out = FlatListGenerator().generate(random.Random(5))
    lines = [l for l in out.split("\n") if l]
    assert len(lines) >= 5


def test_list_multiple_seeds_vary():
    g = FlatListGenerator()
    out1 = g.generate(random.Random(1))
    out2 = g.generate(random.Random(2))
    # Different seeds should (almost certainly) produce different output
    assert out1 != out2
