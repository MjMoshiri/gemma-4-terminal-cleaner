import random
from corpus.generators.trees import TreeGenerator


def test_tree_archetype():
    assert TreeGenerator().archetype == "tree"


def test_tree_deterministic():
    g = TreeGenerator()
    assert g.generate(random.Random(0)) == g.generate(random.Random(0))


def test_tree_no_ansi():
    assert "\x1b[" not in TreeGenerator().generate(random.Random(0))


def test_tree_uses_tree_chars():
    out = TreeGenerator().generate(random.Random(7))
    # tree command uses these box-drawing chars
    assert "├──" in out or "└──" in out
    assert "│" in out or out.count("\n") <= 2  # tiny trees may skip continuation chars


def test_tree_has_summary_line():
    out = TreeGenerator().generate(random.Random(3))
    # last non-empty line should be "N directories, M files"
    lines = [l for l in out.strip().split("\n") if l]
    assert "directories" in lines[-1] and "files" in lines[-1]
