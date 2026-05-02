from infer.ansi_strip import strip


def test_strip_csi():
    assert strip("\x1b[31mhello\x1b[0m") == "hello"


def test_strip_osc8_hyperlink_keeps_visible_text():
    # OSC 8: \x1b]8;;url\x07text\x1b]8;;\x07
    s = "\x1b]8;;https://example.com\x07click here\x1b]8;;\x07"
    assert strip(s) == "click here"


def test_strip_osc_title():
    # OSC 0: terminal title — entire sequence removed, no visible text
    assert strip("\x1b]0;mytitle\x07hello") == "hello"


def test_strip_collapses_cr():
    # Simulating progress bar: each frame ends in \r, final clean line follows
    s = "[##  ] 33%\r[#### ] 66%\r[#####]100%\rdone\n"
    assert strip(s) == "done\n"


def test_strip_handles_bell_and_nul():
    assert strip("foo\x07bar\x00baz") == "foobarbaz"


def test_strip_idempotent():
    s = "\x1b[31m\rhello\x1b[0m\n"
    assert strip(strip(s)) == strip(s)


def test_strip_empty():
    assert strip("") == ""


def test_strip_passthrough_clean_text():
    s = "hello world\nfoo bar\n"
    assert strip(s) == s


def test_strip_windows_line_endings():
    assert strip("foo\r\nbar\r\n") == "foo\nbar\n"


def test_strip_mixed_cr_and_crlf():
    # Progress bar followed by Windows newline
    assert strip("[#] 50%\r[##]100%\rdone\r\nnext\n") == "done\nnext\n"
