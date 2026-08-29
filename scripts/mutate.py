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
import shutil
import subprocess
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Bytecode cache for every subprocess this harness starts. Outside the repo
# tree on purpose: a cache inside it outlives the run and is inherited.
PYCACHE_DIR = ROOT / ".mutate-pycache"


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
        env=_isolated_env(),
    )
    return (proc.stdout + proc.stderr).strip()


def _env() -> dict:
    import os

    return dict(os.environ)


def _isolated_env() -> dict:
    """Env for a subprocess that must read the source as it is on disk NOW.

    CPython validates a cached .pyc against the source's (mtime_seconds, size).
    A mutation that preserves file size and is written and restored inside the
    same filesystem tick produces a .pyc whose header matches the RESTORED
    source exactly -- so the mutated bytecode is treated as current and reused
    indefinitely.

    That is not hypothetical and it is not a small bug. It was found here on
    2026-08-29 by `floors = {"image": 1` -> `{"image": 3`, a one-character,
    identical-length edit. Afterwards `grep` showed 1 in the source and every
    fresh interpreter reported 3, until __pycache__ was deleted by hand.

    Two consequences, and the second is the dangerous one:
      * the harness compared two poisoned witness runs and reported NO-OP for a
        mutation that genuinely changes behaviour -- a false verdict from the
        component whose entire job is to stop false verdicts;
      * pytest afterwards runs against MUTATED bytecode with clean source on
        disk. A suite that goes green then proves nothing, and one that goes red
        indicts code that is fine.

    `-B` does not fix this: it stops Python writing .pyc files, not reading the
    poisoned one that already exists. PYTHONPYCACHEPREFIX does, by sending every
    cache read and write to a scratch directory this harness controls and
    empties between runs, and by keeping the repo tree free of caches the next
    process could inherit.

    `measured: yes` -- the reproduction above, and REPORTING_MUTATIONS keeps the
    same-length mutation as a permanent regression case: if the harness ever
    reports NO-OP for it again, the isolation has regressed.
    """
    _purge_pycache()
    return {
        **_env(),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONPYCACHEPREFIX": str(PYCACHE_DIR),
    }


def _purge_pycache() -> None:
    """Empty the scratch cache, and any __pycache__ already in the tree.

    The in-tree sweep matters on the first run after this fix: a cache poisoned
    by an earlier harness version is still sitting there, and it would be read
    before the prefix ever takes effect.
    """
    shutil.rmtree(PYCACHE_DIR, ignore_errors=True)
    PYCACHE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in ROOT.rglob("__pycache__"):
        if ".venv" not in stale.parts:
            shutil.rmtree(stale, ignore_errors=True)


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
            # Same isolation as the witness. Without it pytest can import the
            # bytecode left behind by a previous mutation instead of the source
            # it is being pointed at.
            env=_isolated_env(),
        )
        return Result.RED if proc.returncode else Result.GREEN
    finally:
        target.write_text(original, encoding="utf-8")
        # The restored file can collide with the mutated file's cache entry on
        # (mtime_seconds, size). Leaving that behind would hand the next
        # process -- a later mutation, a plain pytest run, the developer --
        # bytecode for source that no longer exists anywhere.
        _purge_pycache()


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


# --- the DB-layer suite -----------------------------------------------------
#
# These mutate db.py back into the two shapes that actually shipped broken. Both
# were invisible to pytest because FakeDb replaces Db wholesale, so no test ever
# executed a line of it. tests/test_db_api_shape.py runs the real bodies against
# a psycopg-shaped connection; these mutations are what show it has teeth.
#
# No SQL is executed either way -- these prove the API shape and the column
# list, not that the statements are valid. verify_attribution.py and
# verify_retention.py remain the only evidence of that.

_SHAPED_CONN = (
    "import sys; sys.path.insert(0, 'tests');"
    "import psycopg;"
    "from psycopg_shape import RecordingConnection;"
    "from df.db import Db;"
)

INSERT_ITEMS_WITNESS = (
    _SHAPED_CONN
    + "conn = RecordingConnection();"
    + "psycopg.connect = lambda *a, **k: conn;"
    + "item = {'item_index': 0, 'item_kind': 'face', 'face_index': 0, 'score': 1.0,"
      " 'confidence': 0.9, 'object_key': 'k', 'model_version_id': 'm',"
      " 'model_validation': 'placeholder'};"
    + "\ntry:\n"
      "    Db(dsn='postgresql://unused').insert_items('j', [item])\n"
      "    print('completed', conn._kinds())\n"
      "except Exception as e:\n"
      "    print('raised', type(e).__name__, e)\n"
)

GET_ITEMS_WITNESS = (
    _SHAPED_CONN
    + "conn = RecordingConnection(rows=[[]]);"
    + "psycopg.connect = lambda *a, **k: conn;"
    + "Db(dsn='postgresql://unused').get_items('j');"
    + "print(conn._statements()[0])"
)

DB_MUTATIONS = [
    Mutation(
        # The bug that dead-lettered every job against a real database while
        # the whole suite stayed green. psycopg3 puts executemany on Cursor
        # only; Connection has execute().
        label="insert_items calls executemany on a Connection",
        file="src/df/db.py",
        old="        with self.conn() as c, c.transaction(), c.cursor() as cur:\n"
            "            cur.executemany(",
        new="        with self.conn() as c, c.transaction():\n"
            "            c.executemany(",
        witness=INSERT_ITEMS_WITNESS,
        test="tests/test_db_api_shape.py",
    ),
    Mutation(
        # The second divergence: get_items stopped selecting model_version_id
        # while FakeDb kept returning it, so attribution read fine in pytest
        # and came back NULL in Postgres.
        label="get_items stops selecting model_version_id",
        file="src/df/db.py",
        old="                SELECT item_index, item_kind, face_index, score, confidence,\n"
            "                       object_key, model_version_id, model_validation",
        new="                SELECT item_index, item_kind, face_index, score, confidence,\n"
            "                       object_key, model_validation",
        witness=GET_ITEMS_WITNESS,
        test="tests/test_db_api_shape.py",
    ),
]


# --- coverage and per-face evidence -----------------------------------------
#
# Both are reporting contracts rather than policies, which makes them easy to
# break silently: a wrong coverage number still looks like a number, and a
# mis-ranked face list still looks like a list. Two mutations per value where a
# value is involved, per CLAUDE.md -- dropping the write and writing a wrong
# constant fail differently, and a check that only proves the field exists
# cannot see the second.

COVERAGE_WITNESS = (
    "from df.aggregation import AggregationParams, ScoredItem, aggregate;"
    "items = [ScoredItem(index=i, score=50.0, confidence=0.9) for i in range(4)]"
    " + [ScoredItem(index=9+i, score=50.0, confidence=0.1) for i in range(6)];"
    "r = aggregate(items, AggregationParams.for_media('video'));"
    "print(r.coverage, r.items_total, r.items_used)"
)

FLOOR_WITNESS = (
    "from df.aggregation import AggregationParams as P;"
    "print(P.for_media('image').min_items_for_score,"
    "      P.for_media('video').min_items_for_score)"
)

# More faces than MAX_REPORTED_FACES, deliberately. The first version of this
# witness used two, so the cap could not affect it and the harness correctly
# reported NO-OP for a mutation of the cap -- a weak witness, not a weak test.
EVIDENCE_WITNESS = (
    "from df.rollup import face_evidence;"
    "rows = [{'item_index': i, 'face_index': 0, 'score': float(i), 'confidence': 0.8}"
    "        for i in range(25)];"
    "ev = face_evidence(rows);"
    "print(ev['faces_total'], ev['faces_reported'], [f['score'] for f in ev['top_faces']])"
)

REPORTING_MUTATIONS = [
    Mutation(
        # Drop the measurement entirely.
        label="coverage never reported (always None)",
        file="src/df/aggregation.py",
        old="        if not self.items_total:\n            return None",
        new="        if True:\n            return None",
        witness=COVERAGE_WITNESS,
        test="tests/test_coverage_and_evidence.py",
    ),
    Mutation(
        # Report a plausible wrong constant. A test that only checks the field
        # is present, or is not None, passes this.
        label="coverage always reports full (1.0)",
        file="src/df/aggregation.py",
        old="        return round(self.items_used / self.items_total, 4)",
        new="        return 1.0",
        witness=COVERAGE_WITNESS,
        test="tests/test_coverage_and_evidence.py",
    ),
    Mutation(
        # Revert the modality split: the image floor goes back to the video one.
        label="image inherits the video floor again",
        file="src/df/aggregation.py",
        old='        floors = {"image": 1, "video": 3, "audio": 3}',
        new='        floors = {"image": 3, "video": 3, "audio": 3}',
        witness=FLOOR_WITNESS,
        test="tests/test_coverage_and_evidence.py",
    ),
    Mutation(
        # Rank the wrong way: the face that set the label falls off the cap.
        label="face evidence ranked ascending",
        file="src/df/rollup.py",
        old='    ranked = sorted(faces, key=lambda r: r["score"], reverse=True)',
        new='    ranked = sorted(faces, key=lambda r: r["score"])',
        witness=EVIDENCE_WITNESS,
        test="tests/test_coverage_and_evidence.py",
    ),
    Mutation(
        # Report only what fits the cap, losing "3 of 47".
        label="faces_total counts only the reported faces",
        file="src/df/rollup.py",
        old='        "faces_total": len(faces),',
        new='        "faces_total": min(len(faces), limit),',
        witness=EVIDENCE_WITNESS,
        test="tests/test_coverage_and_evidence.py",
    ),
]


# --- the torch-backend seam -------------------------------------------------
#
# Both of these shipped broken and were invisible until a real checkpoint ran:
# the stub extractors emitted a sha256 digest labelled image/png, and the
# published checkpoint needs unwrapping plus de-prefixing before it loads.

STUB_PNG_WITNESS = (
    "from df.pipelines.extract import _stub_png;"
    "a = _stub_png(b'one'); b = _stub_png(b'two');"
    "print(len(a), a[:8] == bytes([137,80,78,71,13,10,26,10]), a == b)"
)

STATE_DICT_WITNESS = (
    "from df.inference.efficientnet import _state_dict_from;"
    "ck = {'epoch': 37, 'state_dict': {'module.fc.weight': 1,"
    "                                  'encoder.module.block.weight': 2}};"
    "print(sorted(_state_dict_from(ck).items()));"
    "print(sorted(_state_dict_from({'fc.weight': 9}).items()))"
)

WEIGHTS_MUTATIONS = [
    Mutation(
        # Back to the shape that dead-lettered every job the moment a detector
        # that decodes its input was wired in.
        label="stub extractors emit a bare digest again",
        file="src/df/pipelines/extract.py",
        old="    stream = bytearray()\n    seed = payload",
        new="    return hashlib.sha256(payload).digest()\n"
            "    stream = bytearray()\n    seed = payload",
        witness=STUB_PNG_WITNESS,
        test="tests/test_weights_loading.py",
    ),
    Mutation(
        # A constant image is still deterministic, and would make every job
        # score identically -- which is why the determinism check needs its
        # differs-by-input sibling.
        label="stub png ignores its input (constant image)",
        file="src/df/pipelines/extract.py",
        old="    seed = payload\n",
        new="    seed = b'constant'\n",
        witness=STUB_PNG_WITNESS,
        test="tests/test_weights_loading.py",
    ),
    Mutation(
        label="checkpoint wrapper is not unwrapped",
        file="src/df/inference/efficientnet.py",
        old='    if isinstance(sd, dict) and "state_dict" in sd:\n        sd = sd["state_dict"]',
        new='    if False:\n        sd = sd["state_dict"]',
        witness=STATE_DICT_WITNESS,
        test="tests/test_weights_loading.py",
    ),
    Mutation(
        # The subtler one: strip `module.` anywhere instead of only at the
        # front, corrupting a legitimate nested key.
        label="module prefix stripped anywhere, not just leading",
        file="src/df/inference/efficientnet.py",
        old='        (k[len("module."):] if k.startswith("module.") else k): v',
        new='        k.replace("module.", ""): v',
        witness=STATE_DICT_WITNESS,
        test="tests/test_weights_loading.py",
    ),
]


if __name__ == "__main__":
    print("mutating production source; witness-checked before any verdict\n")

    print("advisory suite")
    bad = run_all(ADVISORY_MUTATIONS)
    print(
        f"  {len(ADVISORY_MUTATIONS)} mutation(s), {bad} did not produce RED "
        "(NO-OP is expected for the last one)"
    )

    print("\nDB-layer suite")
    db_bad = run_all(DB_MUTATIONS)
    print(f"  {len(DB_MUTATIONS)} mutation(s), {db_bad} did not produce RED")

    print("\nreporting suite (coverage, per-face evidence)")
    rep_bad = run_all(REPORTING_MUTATIONS)
    print(f"  {len(REPORTING_MUTATIONS)} mutation(s), {rep_bad} did not produce RED")

    print("\ntorch-backend seam (stub PNGs, checkpoint key rewriting)")
    w_bad = run_all(WEIGHTS_MUTATIONS)
    print(f"  {len(WEIGHTS_MUTATIONS)} mutation(s), {w_bad} did not produce RED")
