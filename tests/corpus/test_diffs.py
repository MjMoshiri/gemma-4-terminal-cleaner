import random
from corpus.generators.diffs import UnifiedDiffGenerator


def test_diff_archetype():
    assert UnifiedDiffGenerator().archetype == "diff"


def test_diff_deterministic():
    g = UnifiedDiffGenerator()
    assert g.generate(random.Random(1)) == g.generate(random.Random(1))


def test_diff_no_ansi():
    assert "\x1b[" not in UnifiedDiffGenerator().generate(random.Random(1))


def test_diff_has_unified_format():
    out = UnifiedDiffGenerator().generate(random.Random(1))
    assert out.startswith("---") or "diff --git" in out
    assert "+++ " in out
    assert "@@" in out


def test_diff_has_changed_lines():
    out = UnifiedDiffGenerator().generate(random.Random(2))
    lines = out.split("\n")
    added = [l for l in lines if l.startswith("+") and not l.startswith("+++")]
    removed = [l for l in lines if l.startswith("-") and not l.startswith("---")]
    assert len(added) > 0 or len(removed) > 0
