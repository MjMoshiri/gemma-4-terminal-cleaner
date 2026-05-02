import random
from corpus.generators.code import CodeGenerator


def test_code_archetype():
    assert CodeGenerator().archetype == "code"


def test_code_deterministic():
    g = CodeGenerator()
    assert g.generate(random.Random(0)) == g.generate(random.Random(0))


def test_code_no_ansi():
    assert "\x1b[" not in CodeGenerator().generate(random.Random(0))


def test_code_has_code_keyword():
    # Run several seeds to ensure we see Python or JS output
    found = False
    for seed in range(10):
        out = CodeGenerator().generate(random.Random(seed))
        if "def " in out or "function" in out or "=>" in out:
            found = True
            break
    assert found


def test_code_nonempty():
    out = CodeGenerator().generate(random.Random(0))
    assert len(out.strip()) > 0
