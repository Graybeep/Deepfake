"""Fail if a tracked document states a test count that is not the real one.

The docs rule in CLAUDE.md says the solution document regenerates in the same
commit as any schema or test-count change. That rule has now lost to a human
forgetting twice -- first still claiming a 5-day sprint and 114 tests, then 139
against an actual 147 -- and both times a person caught it, not the process. A
tracked document is worse than an untracked one when it is stale, because being
in the repo is itself a claim that it is current.

So this is mechanical. Run standalone or from .githooks/pre-commit:

    python scripts/check_docs_current.py

It deliberately checks only what can be checked without judgement: the number of
tests each document claims versus the number pytest actually collects. It cannot
tell whether the prose still describes the system -- that still needs a person --
but the counted claims are exactly the ones that drifted both times.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Every tracked document that commits to a number of tests, with the patterns
# that express the claim in that document.
#
# README.md is here because scoping this check to docs/ was itself the bug. The
# rule it enforces is "a tracked document that is stale is worse than no
# document", and that applies to every tracked document -- but the checker
# watched one of them. README.md drifted to 106 against an actual 157 while
# docs/ stayed correct, and nothing noticed, because the guard for exactly this
# failure was pointed somewhere else. Any new document making a counted claim
# belongs in this list on the day it starts making it.
DOCUMENTS = [
    (
        pathlib.Path("docs") / "solution-overview.html",
        [
            re.compile(r"(\d+)\s+unit/integration tests"),
            re.compile(r"(\d+)\s+automated tests passing"),
        ],
    ),
    (
        pathlib.Path("README.md"),
        [
            re.compile(r"(\d+)\s+tests, no infrastructure required"),
        ],
    ),
    (
        # The landing page states the count twice -- once in the data-count
        # attribute the counter animates toward, once as the text that renders
        # with JS off. Both are checked, by two single-group patterns rather
        # than one two-group pattern, because findall() with several groups
        # yields tuples and the loop below int()s whatever it gets. Two claims
        # in one element is precisely how a number gets half-updated.
        pathlib.Path("web") / "landing.html",
        [
            re.compile(r'<b data-count="(\d+)">\d+</b><span>Tests passing'),
            re.compile(r'<b data-count="\d+">(\d+)</b><span>Tests passing'),
        ],
    ),
]


def collected_test_count() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings"],
        cwd=ROOT, capture_output=True, text=True,
    )

    # A COLLECTION ERROR MUST NOT PRODUCE A NUMBER.
    #
    # If a test module fails to import, pytest still prints "N tests collected"
    # for everything else and adds "1 error". Taking that N treats an incomplete
    # collection as authoritative -- and it fails in the dangerous direction: the
    # count drops, so a document claiming the LOWER number validates, and the
    # guard against stale documents goes green because the suite is broken.
    #
    # Found 2026-08-30, when the hook ran under a different interpreter that had
    # pytest but not opencv. tests/test_face_extraction.py failed to import, the
    # count fell from 241 to 233, and the checker reported that as fact. It
    # blocked a correct commit that time; the same mechanism would have passed a
    # stale one.
    if re.search(r"\d+\s+error", proc.stdout) or "ERROR" in proc.stdout:
        print("pytest could not collect the whole suite, so the count is not "
              "trustworthy. Fix collection first -- an incomplete count is worse "
              "than none, because it silently lowers the bar a document has to "
              "clear.")
        print(proc.stdout[-1200:])
        raise SystemExit(2)

    # pytest -q --collect-only prints either a "N tests collected" summary or a
    # per-file "path: N" listing, depending on version. Handle both rather than
    # letting the check quietly stop working after an upgrade -- a broken
    # checker that exits 0 is worse than no checker.
    m = re.search(r"(\d+)\s+tests? collected", proc.stdout)
    if m:
        return int(m.group(1))

    per_file = re.findall(r"^\S+\.py:\s*(\d+)\s*$", proc.stdout, re.M)
    if per_file:
        return sum(int(n) for n in per_file)

    print("could not determine the collected test count; pytest said:\n"
          f"{proc.stdout[-500:]}\n{proc.stderr[-500:]}")
    raise SystemExit(2)


def main() -> int:
    actual = collected_test_count()

    stale: list[str] = []
    silent: list[str] = []
    missing: list[str] = []

    for relpath, patterns in DOCUMENTS:
        doc = ROOT / relpath
        if not doc.exists():
            missing.append(str(relpath))
            continue

        text = doc.read_text(encoding="utf-8")
        found_any = False
        for pattern in patterns:
            for claimed in pattern.findall(text):
                found_any = True
                if int(claimed) != actual:
                    stale.append(f"  {relpath}: says {claimed}, pytest collects {actual}")

        # Per document, not across all of them. A document whose wording drifted
        # out of every one of its patterns is no longer being checked at all,
        # and another document still matching would otherwise hide that.
        if not found_any:
            silent.append(str(relpath))

    if missing:
        print("missing:")
        for path in missing:
            print(f"  {path}")
        return 1

    if silent:
        print("no test-count claim found in:")
        for path in silent:
            print(f"  {path}")
        print(
            "\nThe check has silently stopped checking that document. Either "
            "restore the claim or update its patterns in DOCUMENTS."
        )
        return 1

    if stale:
        print("stale test-count claims:")
        print("\n".join(stale))
        print(
            "\nUpdate every document above in this same commit, per CLAUDE.md.\n"
            "For docs/solution-overview.html the PDF regenerates with it:\n"
            "  1. edit docs/solution-overview.html\n"
            "  2. chrome --headless --disable-gpu --no-pdf-header-footer \\\n"
            "       --print-to-pdf=docs/Deepfake-Detection-Solution.pdf \\\n"
            "       file:///.../docs/solution-overview.html\n"
            "  3. git add docs/ README.md"
        )
        return 1

    checked = ", ".join(str(relpath) for relpath, _ in DOCUMENTS)
    print(f"test count current ({actual}) in: {checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
