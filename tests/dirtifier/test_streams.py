import random
from dirtifier.transforms.streams import MixedStreams


def test_streams_deterministic():
    t = MixedStreams()
    inp = "\n".join([f"line {i}" for i in range(50)]) + "\n"
    a = t.apply(inp, random.Random(11))
    b = t.apply(inp, random.Random(11))
    assert a == b


def test_streams_handles_empty():
    assert MixedStreams().apply("", random.Random(0)) == ""


def test_streams_line_count_nondecreasing():
    t = MixedStreams()
    inp = "\n".join([f"line {i}" for i in range(50)]) + "\n"
    out = t.apply(inp, random.Random(11))
    assert out.count("\n") >= inp.count("\n")


def test_streams_name():
    assert MixedStreams.name == "mixed_streams"
