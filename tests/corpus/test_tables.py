import random
import re
from corpus.generators.tables import LsListingGenerator, PsListingGenerator, DfListingGenerator


def test_ls_listing_archetype():
    assert LsListingGenerator().archetype == "table"


def test_ls_listing_deterministic():
    g = LsListingGenerator()
    out1 = g.generate(random.Random(42))
    out2 = g.generate(random.Random(42))
    assert out1 == out2


def test_ls_listing_no_ansi():
    out = LsListingGenerator().generate(random.Random(7))
    assert "\x1b[" not in out  # no ANSI escape codes


def test_ls_listing_format():
    out = LsListingGenerator().generate(random.Random(7))
    lines = out.strip().split("\n")
    # First line is "total NNN" header
    assert re.match(r"^total \d+$", lines[0])
    # Each subsequent line: perms links owner group size date name
    for line in lines[1:]:
        # e.g. -rw-r--r-- 1 alice users 1234 May  2 10:30 README.md
        assert re.match(r"^[-dlrwxs]{10}\s+\d+\s+\S+\s+\S+\s+\d+\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}\s+\S+", line)


def test_ps_listing_deterministic():
    g = PsListingGenerator()
    assert g.generate(random.Random(11)) == g.generate(random.Random(11))


def test_ps_listing_no_ansi():
    assert "\x1b[" not in PsListingGenerator().generate(random.Random(11))


def test_df_listing_deterministic():
    g = DfListingGenerator()
    assert g.generate(random.Random(13)) == g.generate(random.Random(13))


def test_df_listing_no_ansi():
    assert "\x1b[" not in DfListingGenerator().generate(random.Random(13))


def test_ps_listing_format():
    out = PsListingGenerator().generate(random.Random(11))
    lines = out.strip().split("\n")
    # Header line
    assert "PID" in lines[0] and "CMD" in lines[0]
    # At least 5 process lines
    assert len(lines) >= 6


def test_df_listing_format():
    out = DfListingGenerator().generate(random.Random(13))
    lines = out.strip().split("\n")
    assert "Filesystem" in lines[0]
    assert "Use%" in lines[0]
    # All non-header lines have a percent column
    for line in lines[1:]:
        assert "%" in line
