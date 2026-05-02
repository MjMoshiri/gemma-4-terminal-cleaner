import random
from dirtifier.transforms.whitespace import WhitespacePadding, TrailingWhitespace, WindowsLineEndings


# -- WhitespacePadding --

def test_whitespace_padding_deterministic():
    t = WhitespacePadding()
    a = t.apply("foo   bar   baz\nqux  quux\n", random.Random(7))
    b = t.apply("foo   bar   baz\nqux  quux\n", random.Random(7))
    assert a == b


def test_whitespace_padding_handles_empty():
    assert WhitespacePadding().apply("", random.Random(0)) == ""


def test_whitespace_padding_names():
    assert WhitespacePadding.name == "whitespace_padding"


def test_whitespace_padding_output_length():
    t = WhitespacePadding()
    inp = "col1   col2   col3\n" * 20
    out = t.apply(inp, random.Random(42))
    # With prob 0.3 per space-run, the output should be at least as long
    assert len(out) >= len(inp)


# -- TrailingWhitespace --

def test_trailing_whitespace_deterministic():
    t = TrailingWhitespace()
    a = t.apply("hello\nworld\n", random.Random(1))
    b = t.apply("hello\nworld\n", random.Random(1))
    assert a == b


def test_trailing_whitespace_handles_empty():
    assert TrailingWhitespace().apply("", random.Random(0)) == ""


def test_trailing_whitespace_output_length():
    t = TrailingWhitespace()
    inp = "hello\nworld\n"
    out = t.apply(inp, random.Random(1))
    assert len(out) >= len(inp)


def test_trailing_whitespace_name():
    assert TrailingWhitespace.name == "trailing_whitespace"


# -- WindowsLineEndings --

def test_windows_line_endings_deterministic():
    t = WindowsLineEndings()
    a = t.apply("line1\nline2\nline3\n", random.Random(3))
    b = t.apply("line1\nline2\nline3\n", random.Random(3))
    assert a == b


def test_windows_line_endings_handles_empty():
    assert WindowsLineEndings().apply("", random.Random(0)) == ""


def test_windows_line_endings_inserts_crlf():
    # With 40% probability per newline and many seeds, at least one should flip
    t = WindowsLineEndings()
    inp = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n"
    found = False
    for seed in range(20):
        out = t.apply(inp, random.Random(seed))
        if "\r\n" in out:
            found = True
            break
    assert found, "Expected at least one seed to produce \\r\\n"


def test_windows_line_endings_name():
    assert WindowsLineEndings.name == "windows_line_endings"
