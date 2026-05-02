import random
from dirtifier.transforms.timestamps import Timestamps


def test_timestamps_deterministic():
    t = Timestamps()
    a = t.apply("step one\nstep two\nstep three\n", random.Random(9))
    b = t.apply("step one\nstep two\nstep three\n", random.Random(9))
    assert a == b


def test_timestamps_handles_empty():
    assert Timestamps().apply("", random.Random(0)) == ""


def test_timestamps_prefix_year():
    t = Timestamps()
    out = t.apply("step one\nstep two\nstep three\n", random.Random(9))
    for line in out.split("\n"):
        assert line.startswith("2026-"), f"line does not start with '2026-': {line!r}"


def test_timestamps_name():
    assert Timestamps.name == "timestamps"
