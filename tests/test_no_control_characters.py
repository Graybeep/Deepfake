"""No source file may contain a literal control character.

This exists because of a bug that cost real time and was invisible to every
tool used to look for it. `tests/test_landing_page.py` had:

    re.sub(r"<style\x08.*?</style>", " ", text, flags=re.S)

A `\b`, intended as a regex word boundary, had been interpreted as the
BACKSPACE escape on its way through a shell heredoc. The pattern then required
a literal 0x08 byte after "style", so it never matched, so the page's CSS was
never stripped, so a fixture documented as returning "only what a reader sees"
was returning the whole file including stylesheets and scripts.

What makes it worth a permanent guard: `grep`, `sed` and a Read all render 0x08
as nothing, so the line looks exactly like the correct one. It was found only by
`inspect.getsource()` on the compiled function and printing `repr()` of each
line. Confidence in reading the file was misplaced -- the file did not contain
what it appeared to contain.

The same escape has now been mangled three times in this codebase's history --
here, in `scripts/mutate.py`'s witness regexes, and in a `print("\n...")` that
became a real newline mid-string. Two of those three were silent in the
permissive direction: the check still ran and still passed.

Tabs and newlines are excluded, being legitimate. Everything else in the C0
range is a mistake.

# In-process, static. No live probe needed: this is a property of the bytes in
# the repository.
"""
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWED = {0x09, 0x0A, 0x0D}          # tab, newline, carriage return

SOURCES = sorted(
    p for p in list(ROOT.glob("tests/*.py"))
    + list(ROOT.glob("scripts/*.py"))
    + list(ROOT.rglob("src/**/*.py"))
    + list(ROOT.glob("web/*.html"))
    if ".venv" not in p.parts
)


def test_the_source_list_is_not_empty():
    """A glob that matches nothing turns every case below into a no-op, which
    is the failure mode this whole file exists to prevent."""
    assert len(SOURCES) > 20, f"only found {len(SOURCES)} files; the globs are wrong"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_literal_control_characters(path: pathlib.Path):
    text = path.read_text(encoding="utf-8")

    found = [
        (i, hex(ord(c)))
        for i, c in enumerate(text)
        if ord(c) < 0x20 and ord(c) not in ALLOWED
    ]

    assert not found, (
        f"{path.relative_to(ROOT)} contains {len(found)} literal control "
        f"character(s) at {found[:5]} -- almost certainly an escape sequence "
        f"that was interpreted instead of written literally. These are "
        f"invisible to grep and to reading the file."
    )
