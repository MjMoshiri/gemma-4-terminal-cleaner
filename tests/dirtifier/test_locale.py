import random
from dirtifier.transforms.locale import LocaleVariants


def test_locale_deterministic():
    t = LocaleVariants()
    inp = "processed 12345 of 67890 records in 1000000 ms"
    a = t.apply(inp, random.Random(5))
    b = t.apply(inp, random.Random(5))
    assert a == b


def test_locale_handles_empty():
    assert LocaleVariants().apply("", random.Random(0)) == ""


def test_locale_digits_preserved():
    t = LocaleVariants()
    inp = "processed 12345 of 67890 records in 1000000 ms"
    out = t.apply(inp, random.Random(5))
    # All original digits must still appear somewhere in the output
    for d in "123456789":
        assert d in out, f"digit {d!r} lost after locale transform"


def test_locale_name():
    assert LocaleVariants.name == "locale_variants"
