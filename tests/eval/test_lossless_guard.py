from eval.lossless_guard import extract_atoms, lossless_check


def test_extract_atoms_paths():
    atoms = extract_atoms("see file at src/main.py and /etc/hosts here")
    assert "src/main.py" in atoms
    assert "/etc/hosts" in atoms


def test_extract_atoms_numbers():
    atoms = extract_atoms("size 1.2K, 4.5MB, count 1234, duration 30s")
    assert "1.2K" in atoms
    assert "4.5MB" in atoms
    assert "1234" in atoms


def test_extract_atoms_identifiers():
    atoms = extract_atoms("calling foo_bar(x) and HTTPClient and PI")
    assert "foo_bar" in atoms
    assert "HTTPClient" in atoms
    # 'PI' is too short (2 chars), should NOT be an atom
    assert "PI" not in atoms


def test_extract_atoms_quoted_strings():
    atoms = extract_atoms('error: "connection refused" and \'foo bar\'')
    assert '"connection refused"' in atoms or "connection refused" in atoms
    # at least one form of the quoted content
    assert any("connection refused" in a for a in atoms)


def test_extract_atoms_urls_ips_emails():
    atoms = extract_atoms("see https://example.com/path or 10.0.0.1 mail to a@b.co")
    assert any("example.com" in a for a in atoms)
    assert "10.0.0.1" in atoms
    assert "a@b.co" in atoms


def test_lossless_check_pass_simple():
    assert lossless_check("hello world foo", "hello world foo").passed


def test_lossless_check_pass_after_strip():
    # ANSI-stripped version retains the same atoms
    assert lossless_check("\x1b[31mhello world\x1b[0m", "hello world").passed


def test_lossless_check_fail_missing_atom():
    result = lossless_check("important error 12345 in /etc/hosts",
                            "some text without the path")
    assert not result.passed
    assert "12345" in result.missing_atoms or "/etc/hosts" in result.missing_atoms


def test_lossless_check_whitelist_allows_dirtifier_artifacts():
    # The dirtifier-injected atom is whitelisted, so even if it's missing the check passes
    result = lossless_check("real_id_42 spinner_frame_3", "real_id_42",
                            removable_whitelist={"spinner_frame_3"})
    assert result.passed


def test_lossless_check_empty_input():
    assert lossless_check("", "").passed
