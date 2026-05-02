import random
import re
from corpus.generators.logs import LogGenerator


def test_log_archetype():
    assert LogGenerator().archetype == "log"


def test_log_deterministic():
    g = LogGenerator()
    assert g.generate(random.Random(0)) == g.generate(random.Random(0))


def test_log_no_ansi():
    assert "\x1b[" not in LogGenerator().generate(random.Random(0))


def test_log_has_level():
    out = LogGenerator().generate(random.Random(4))
    assert re.search(r"\[(DEBUG|INFO |WARN |ERROR)\]", out)


def test_log_has_timestamps():
    out = LogGenerator().generate(random.Random(4))
    # Timestamps look like 2026-04-DD
    assert re.search(r"2026-04-\d{2}T\d{2}:\d{2}:\d{2}", out)
