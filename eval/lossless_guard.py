"""Deterministic information-preservation check for lossless cleanup.
See spec §7.1 for the atom taxonomy."""
import re
from dataclasses import dataclass

import infer.ansi_strip as _ansi


_PATH_RE = re.compile(r"(?:/|\./|~/|[\w-]+/)[\w./~-]+|[\w_-]+\.[a-zA-Z]{1,8}\b")
_NUMBER_WITH_UNIT_RE = re.compile(r"\b\d+(?:[.,]\d+)?(?:[KMGTP]i?B?|[smhd]|%)?\b")
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_QUOTED_RE = re.compile(r'"[^"\n]+"|\'[^\'\n]+\'')
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


@dataclass
class GuardResult:
    passed: bool
    missing_atoms: set[str]
    n_input: int
    n_output: int


def extract_atoms(text: str) -> set[str]:
    """Extract information atoms from text. Order: longer/more-specific first."""
    # Strip ANSI/terminal codes before extracting so escape sequences don't
    # produce spurious atoms (e.g. "0m" from "\x1b[0m").
    text = _ansi.strip(text)
    atoms: set[str] = set()
    # URLs first (they look like paths + numbers)
    for m in _URL_RE.finditer(text):
        atoms.add(m.group(0))
    text_for_rest = _URL_RE.sub(" ", text)
    for m in _EMAIL_RE.finditer(text_for_rest):
        atoms.add(m.group(0))
    text_for_rest = _EMAIL_RE.sub(" ", text_for_rest)
    for m in _IP_RE.finditer(text_for_rest):
        atoms.add(m.group(0))
    for m in _PATH_RE.finditer(text_for_rest):
        atoms.add(m.group(0))
    for m in _QUOTED_RE.finditer(text_for_rest):
        # Store both with and without quotes for flexible matching
        atoms.add(m.group(0))
        inner = m.group(0)[1:-1]
        atoms.add(inner)
    for m in _NUMBER_WITH_UNIT_RE.finditer(text_for_rest):
        atoms.add(m.group(0))
    for m in _IDENT_RE.finditer(text_for_rest):
        atoms.add(m.group(0))
    return atoms


def lossless_check(
    input_text: str,
    output_text: str,
    removable_whitelist: set[str] | None = None,
) -> GuardResult:
    """Verify atoms(input) ⊆ atoms(output) ∪ removable_whitelist."""
    in_atoms = extract_atoms(input_text)
    out_atoms = extract_atoms(output_text)
    whitelist = removable_whitelist or set()
    # Atoms that should still be present in output
    required = in_atoms - whitelist
    # An atom is "preserved" if it appears in output, OR a longer atom containing it is preserved
    # (handles e.g. "foo" being subsumed into "foo_bar")
    out_text_lc = output_text  # we check substring against raw output to forgive merging
    missing = set()
    for atom in required:
        if atom in out_atoms or atom in out_text_lc:
            continue
        missing.add(atom)
    return GuardResult(
        passed=len(missing) == 0,
        missing_atoms=missing,
        n_input=len(in_atoms),
        n_output=len(out_atoms),
    )
