"""Mutation harness that verifies the mutation actually mutated something.

CLAUDE.md requires any bug-specific regression test to be shown RED before the
fix is in place, by mutating production source. That practice had no check of
its own, and it failed exactly the way everything else in this codebase has
failed: silently, in the permissive direction.

The concrete miss: a mutation written as

    return [] or ["UNVERIFIED MODEL: ..."]

edits the source, compiles, looks like a mutation in the diff -- and evaluates
to the original list. The test stayed green, which was reported as "the test is
vacuous". The test was fine. The mutation did nothing. A no-op mutation and a
vacuous test produce byte-identical output, so RED/GREEN alone cannot tell them
apart, and the only reason it was caught was that the result looked surprising
enough to re-read. That is not a control.

So: before running the test, evaluate a WITNESS -- a snippet that calls the
mutated code on a known input and prints the result -- in a fresh interpreter,
once against the original source and once against the mutated source. If the
two outputs match, the mutation changed no behaviour and the run reports NO-OP
rather than a verdict. Only a mutation that demonstrably changes behaviour is
allowed to produce RED or GREEN.

The witness runs in a subprocess on purpose: the mutated module must be
imported fresh, and an already-imported module in this process would not be.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Mutation:
    label: str
    file: str          # repo-relative
    old: str
    new: str
    witness: str       # python that prints observable behaviour of the target
    test: str          # pytest target


class Result:
    NO_OP = "NO-OP (mutation changed nothing -- verdict withheld)"
    RED = "RED (test caught it)"
    GREEN = "GREEN (test did NOT catch it)"


def _run_witness(code: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT, capture_output=True, text=True,
        env={**_env(), "PYTHONPATH": str(ROOT / "src")},
    )
    return (proc.stdout + proc.stderr).strip()


def _env() -> dict:
    import os

    return dict(os.environ)


def run(mutation: Mutation) -> str:
    target = ROOT / mutation.file
    original = target.read_text(encoding="utf-8")
    if mutation.old not in original:
        raise SystemExit(f"anchor not found in {mutation.file}: {mutation.old[:60]!r}")

    try:
        before = _run_witness(mutation.witness)
        target.write_text(original.replace(mutation.old, mutation.new, 1), encoding="utf-8")
        after = _run_witness(mutation.witness)

        if before == after:
            # The whole point. Do not run the test, do not report a verdict:
            # a no-op mutation tells you nothing about the test either way.
            return f"{Result.NO_OP}\n      witness unchanged: {before[:120]}"

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", mutation.test, "-p", "no:warnings", "-q"],
            cwd=ROOT, capture_output=True, text=True,
        )
        return Result.RED if proc.returncode else Result.GREEN
    finally:
        target.write_text(original, encoding="utf-8")


def run_all(mutations: list[Mutation]) -> int:
    failures = 0
    for m in mutations:
        verdict = run(m)
        print(f"{verdict.splitlines()[0]:<46} {m.label}")
        for extra in verdict.splitlines()[1:]:
            print(extra)
        if not verdict.startswith("RED"):
            failures += 1
    return failures


# --- the advisory suite, including the mutation that fooled the old harness ---

ADVISORY_WITNESS = (
    "from df.gateway.app import _validation_advisories as a;"
    "print(a({'model_validation': None, 'model_version_id': 'face-effnet-abc'}),"
    "      a({'model_validation': 'research-checkpoint', 'model_version_id': 'face-effnet-abc'}))"
)

ADVISORY_MUTATIONS = [
    Mutation(
        label="revert to the fail-open substring check",
        file="src/df/gateway/app.py",
        old='    level = job.get("model_validation")',
        new='    level = VALIDATION_PLACEHOLDER if "stub" in (job.get("model_version_id") or "")'
            ' else VALIDATION_PRODUCTION',
        witness=ADVISORY_WITNESS,
        test="tests/test_validation_advisory.py",
    ),
    Mutation(
        label="unknown/NULL level returns no caveat",
        file="src/df/gateway/app.py",
        old="    # NULL, or anything unrecognised.\n    return [",
        new="    # NULL, or anything unrecognised.\n    return []\n    return [",
        witness=ADVISORY_WITNESS,
        test="tests/test_validation_advisory.py",
    ),
    Mutation(
        label="research checkpoint treated as validated",
        file="src/df/gateway/app.py",
        old="    if level == VALIDATION_RESEARCH:",
        new="    if level == VALIDATION_RESEARCH:\n        return []\n    if False:",
        witness=ADVISORY_WITNESS,
        test="tests/test_validation_advisory.py",
    ),
    Mutation(
        # Kept deliberately. This is the exact edit that fooled the old harness
        # into reporting the test vacuous; it must now report NO-OP.
        label="the no-op that fooled the old harness (`[] or [...]`)",
        file="src/df/gateway/app.py",
        old="    # NULL, or anything unrecognised.\n    return [",
        new="    # NULL, or anything unrecognised.\n    return [] or [",
        witness=ADVISORY_WITNESS,
        test="tests/test_validation_advisory.py",
    ),
]


if __name__ == "__main__":
    print("mutating production source; witness-checked before any verdict\n")
    bad = run_all(ADVISORY_MUTATIONS)
    print(
        f"\n{len(ADVISORY_MUTATIONS)} mutation(s), {bad} did not produce RED "
        "(NO-OP is expected for the last one)"
    )
