"""Fail if docs/ states a test count that is not the real one.

The docs rule in CLAUDE.md says the solution document regenerates in the same
commit as any schema or test-count change. That rule has now lost to a human
forgetting twice -- first still claiming a 5-day sprint and 114 tests, then 139
against an actual 147 -- and both times a person caught it, not the process. A
tracked document is worse than an untracked one when it is stale, because being
in the repo is itself a claim that it is current.

So this is mechanical. Run standalone or from .githooks/pre-commit:

    python scripts/check_docs_current.py

It deliberately checks only what can be checked without judgement: the number of
tests the document claims versus the number pytest actually collects. It cannot
tell whether the prose still describes the system -- that still needs a person --
but the counted claims are exactly the ones that drifted both times.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "solution-overview.html"

# Every place the document commits to a number of tests.
CLAIM_PATTERNS = [
    re.compile(r"(\d+)\s+unit/integration tests"),
    re.compile(r"(\d+)\s+automated tests passing"),
]


def collected_test_count() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings"],
        cwd=ROOT, capture_output=True, text=True,
    )
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
    if not DOC.exists():
        print(f"{DOC.relative_to(ROOT)} is missing")
        return 1

    actual = collected_test_count()
    text = DOC.read_text(encoding="utf-8")

    stale: list[str] = []
    found_any = False
    for pattern in CLAIM_PATTERNS:
        for claimed in pattern.findall(text):
            found_any = True
            if int(claimed) != actual:
                stale.append(f"  document says {claimed}, pytest collects {actual}")

    if not found_any:
        print("no test-count claim found in the document -- the check has silently "
              "stopped checking anything. Update CLAIM_PATTERNS to match the doc.")
        return 1

    if stale:
        print("docs/ is stale:")
        print("\n".join(stale))
        print(
            "\nRegenerate in this same commit, per CLAUDE.md:\n"
            "  1. edit docs/solution-overview.html\n"
            "  2. chrome --headless --disable-gpu --no-pdf-header-footer \\\n"
            "       --print-to-pdf=docs/Deepfake-Detection-Solution.pdf \\\n"
            "       file:///.../docs/solution-overview.html\n"
            "  3. git add docs/"
        )
        return 1

    print(f"docs/ test count current ({actual})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
