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


# --- calibration ------------------------------------------------------------
#
# The fitter has no real data to be checked against, so its only guard is the
# synthetic recovery test -- which makes mutating it more important, not less.
# The scheme mutations restore the exact false claim that was shipped: a
# hardcoded "launch-snapshot" on results where nothing had been fitted.

FIT_WITNESS = (
    "import math, random;"
    "from df.inference.calibration import fit_temperature;"
    "rng = random.Random(42);"
    "z = [rng.gauss(0.0, 2.5) for _ in range(4000)];"
    "y = [1 if rng.random() < 1/(1+math.exp(-v)) else 0 for v in z];"
    "print(round(fit_temperature([v/2.0 for v in z], y, fitted_on='w').value, 3))"
)

SCHEME_WITNESS = (
    "from df.inference.calibration import Temperature;"
    "print(Temperature(1.0, fitted_on='x', fitted=False).scheme,"
    "      Temperature(1.7, fitted_on='x', fitted=True).scheme,"
    # The discriminating case, and the whole reason `fitted` is a field: a real
    # fit that landed on 1.0. Without it the witness cannot see a mutation that
    # infers fitted-ness from the value, and the harness correctly reports NO-OP.
    "      Temperature(1.0, fitted_on='x', fitted=True).scheme)"
)

ADVISORY_CAL_WITNESS = (
    "from df.gateway.app import _calibration_advisories as a;"
    "print(a({'model_validation': 'research-checkpoint',"
    "         'calibration': 'temperature.v1:unfitted'}));"
    "print(a({'model_validation': 'research-checkpoint'}))"
)

# Must actually attempt a one-class fit, or removing the guard changes nothing
# observable and the harness withholds a verdict.
ONE_CLASS_WITNESS = (
    # No try/except and no newlines on purpose. The witness captures stderr
    # as well as stdout, so an uncaught raise IS observable output: unmutated
    # this prints a ValueError traceback, mutated it prints a number. That is
    # a cleaner discriminator than a caught exception, and it avoids embedding
    # newlines in a single-line -c payload.
    "from df.inference.calibration import fit_temperature;"
    "print('returned', round(fit_temperature("
    "[1.0, -2.0, 0.5, 3.0], [1, 1, 1, 1], fitted_on='w').value, 4))"
)


CALIBRATION_MUTATIONS = [
    Mutation(
        # The shipped bug: assert a snapshot regardless of whether one happened.
        label="scheme hardcoded to launch-snapshot again",
        file="src/df/inference/calibration.py",
        old="        return SCHEME_LAUNCH_SNAPSHOT if self.fitted else SCHEME_UNFITTED",
        new="        return SCHEME_LAUNCH_SNAPSHOT",
        witness=SCHEME_WITNESS,
        test="tests/test_calibration_fit.py",
    ),
    Mutation(
        # The subtler version: infer "fitted" from the value, which reports a
        # genuine fit that landed on 1.0 as though it never happened.
        label="fitted inferred from value != 1.0",
        file="src/df/inference/calibration.py",
        old="        return SCHEME_LAUNCH_SNAPSHOT if self.fitted else SCHEME_UNFITTED",
        new="        return SCHEME_UNFITTED if self.value == 1.0 else SCHEME_LAUNCH_SNAPSHOT",
        witness=SCHEME_WITNESS,
        test="tests/test_calibration_fit.py",
    ),
    Mutation(
        # Fit on the wrong axis. T is not convex; the search can settle in the
        # wrong place, and the returned number still looks like a temperature.
        label="fitter searches T instead of 1/T",
        file="src/df/inference/calibration.py",
        old="    return Temperature(value=1.0 / ((lo + hi) / 2.0), fitted_on=fitted_on, fitted=True)",
        new="    return Temperature(value=(lo + hi) / 2.0, fitted_on=fitted_on, fitted=True)",
        witness=FIT_WITNESS,
        test="tests/test_calibration_fit.py",
    ),
    Mutation(
        # Accept a one-class set. NLL is then minimised at a search bound and
        # the boundary artefact is returned as a confident fit.
        label="one-class set accepted rather than refused",
        file="src/df/inference/calibration.py",
        old="    if len(set(labels)) < 2:",
        new="    if False:",
        witness=ONE_CLASS_WITNESS,
        test="tests/test_calibration_fit.py",
    ),
    Mutation(
        # Fail open: say nothing when the calibration is unrecorded.
        label="calibration advisory silent on unknown scheme",
        file="src/df/gateway/app.py",
        old='    # NULL, or a scheme this function does not know about. Same rule as the\n'
            '    # validation advisory: unrecognised provenance warns rather than stays quiet.\n'
            '    return [',
        new='    # NULL, or a scheme this function does not know about. Same rule as the\n'
            '    # validation advisory: unrecognised provenance warns rather than stays quiet.\n'
            '    return []\n    return [',
        witness=ADVISORY_CAL_WITNESS,
        test="tests/test_calibration_fit.py",
    ),
]


# --- calibration-set extraction ---------------------------------------------
#
# This script decides what a temperature gets fitted on, so a silent fault here
# corrupts the calibration rather than crashing. Recording `score` instead of
# `logit` is the dangerous one: it produces a full, plausible file and a
# meaningless fit.

EXTRACT_WITNESS = (
    "import sys; sys.path.insert(0, 'scripts');"
    "from extract_logits import rows_for_video;"
    "from df.inference.base import Prediction;"
    "from df.pipelines.extract import FaceCrop, Frame;"
    "S = type('S', (), {'sample': lambda self, b: "
    "    [Frame(index=i, data=b'f', timestamp_s=0.0) for i in range(6)]})();"
    # TWO faces per frame, against max_faces=3. With one per frame the
    # loop's own break already stops at the budget and the slice never
    # bites, so a cap mutation is genuinely a no-op -- as the harness
    # correctly reported. Overshooting the budget within a single frame
    # is the only shape where the slice does work.
    "E = type('E', (), {'extract': lambda self, b, frame_index=0: "
    "    [FaceCrop(frame_index=frame_index, face_index=i, data=b'c', confidence=0.7)"
    "     for i in range(2)]})();"
    "D = type('D', (), {'predict_batch': lambda self, xs: "
    "    [Prediction(score=90.0, confidence=1.0, logit=-1.75) for _ in xs]})();"
    "r = rows_for_video(b'v', 1, 'a.mp4', sampler=S, extractor=E, detector=D, max_faces=3);"
    "print(len(r), sorted(r[0].keys()), r[0].get('logit'), r[0].get('label'))"
)

EXTRACT_MUTATIONS = [
    Mutation(
        # The quiet catastrophe: write the already-scaled score. The file looks
        # right, the fit runs, and the temperature it produces means nothing.
        label="records score instead of logit",
        file="scripts/extract_logits.py",
        old='            "logit": pred.logit,',
        new='            "logit": pred.score,',
        witness=EXTRACT_WITNESS,
        test="tests/test_extract_logits.py",
    ),
    Mutation(
        # Drop the per-video cap, letting one long clip dominate the fit.
        label="face cap ignored",
        file="scripts/extract_logits.py",
        old="    crops = crops[:max_faces]",
        new="    crops = crops[:]",
        witness=EXTRACT_WITNESS,
        test="tests/test_extract_logits.py",
    ),
    Mutation(
        # Accept a detector with no logit -- i.e. calibrate on the stub's hash.
        label="missing logit accepted instead of refused",
        file="scripts/extract_logits.py",
        old="        if pred.logit is None:",
        new="        if False:",
        witness=(
            "import sys; sys.path.insert(0, 'scripts');"
            "from extract_logits import rows_for_video;"
            "from df.inference.base import Prediction;"
            "from df.pipelines.extract import FaceCrop, Frame;"
            "S = type('S', (), {'sample': lambda self, b: "
            "    [Frame(index=0, data=b'f', timestamp_s=0.0)]})();"
            "E = type('E', (), {'extract': lambda self, b, frame_index=0: "
            "    [FaceCrop(frame_index=0, face_index=0, data=b'c', confidence=0.7)]})();"
            "D = type('D', (), {'predict_batch': lambda self, xs: "
            "    [Prediction(score=90.0, confidence=1.0) for _ in xs]})();"
            "print(rows_for_video(b'v', 1, 'a.mp4', sampler=S, extractor=E,"
            "                     detector=D, max_faces=3))"
        ),
        test="tests/test_extract_logits.py",
    ),
]


# --- the real face extractor ------------------------------------------------
#
# This path had no test at all until 2026-08-30, which is how it came to raise
# on every detected face. The aspect-ratio mutation is the one worth having:
# squaring the crop does not fail, it silently defeats the detector's
# isotropic-resize-and-pad and changes what the model sees.

EXTRACT_FACE_WITNESS = (
    "import numpy as np, cv2;"
    "from df.pipelines.extract import OpenCVFaceExtractor;"
    "img = np.random.default_rng(0).integers(0, 255, (300, 400, 3)).astype(np.uint8);"
    "png = cv2.imencode('.png', img)[1].tobytes();"
    "C = type('C', (), {'detectMultiScale3': lambda self, *a, **k: "
    "    (np.array([[10, 20, 120, 60], [200, 40, 50, 50]]), None, np.array([9.0, 0.5]))});"
    "cv2.CascadeClassifier = lambda *a, **k: C();"
    "cs = OpenCVFaceExtractor().extract(png);"
    "print(len(cs), [c.bbox for c in cs], [round(c.confidence, 3) for c in cs],"
    "      [cv2.imdecode(np.frombuffer(c.data, np.uint8), cv2.IMREAD_COLOR).shape[:2] for c in cs])"
)

RATELIMIT_WITNESS = (
    "import os; os.environ['DF_TRUSTED_PROXY_HOPS'] = '1';"
    "import df.gateway.app as g;"
    "from df.config import Settings; g.settings = Settings();"
    "R = lambda peer, xff=None: type('R', (), {"
    "    'client': type('C', (), {'host': peer})(),"
    "    'headers': ({'x-forwarded-for': xff} if xff else {})})();"
    # At one hop: same client via rotating proxies shares a key, two clients
    # do not, a spoofed prefix is ignored, a malformed header falls back, and
    # an ipv4-mapped address does not get a second bucket.
    "print(g.identity_of(R('100.64.0.2', '198.51.100.22')),"
    "      g.identity_of(R('100.64.0.19', '198.51.100.22')),"
    "      g.identity_of(R('100.64.0.2', '203.0.113.77')),"
    "      g.identity_of(R('100.64.0.2', 'evil, 198.51.100.22')),"
    "      g.identity_of(R('203.0.113.9', 'not-an-ip')),"
    "      g.identity_of(R('10.0.0.7', '::ffff:198.51.100.22')));"
    # Then TWO hops with a one-entry header. The "fewer entries than hops"
    # guard is unreachable at one hop -- an empty header returns earlier --
    # so a witness probing only hops=1 reports NO-OP for a mutation of it,
    # which is exactly what happened the first time. No try/except needed:
    # without the guard this indexes past the list and the traceback is the
    # observable difference.
    "os.environ['DF_TRUSTED_PROXY_HOPS'] = '2'; g.settings = Settings();"
    "print('two-hop short header ->', g.identity_of(R('203.0.113.9', '1.2.3.4')))"
)

RATELIMIT_MUTATIONS = [
    Mutation(
        # The shipped bug: key on the socket peer behind a rotating proxy
        # pool, so nothing ever accumulates and limiting does not happen.
        label="key on the socket peer again (rotating proxies defeat it)",
        file="src/df/gateway/app.py",
        old="    if hops <= 0:",
        new="    if True:",
        witness=RATELIMIT_WITNESS,
        test="tests/test_rate_limit_identity.py",
    ),
    Mutation(
        # The dangerous "fix": trust the LEFTMOST entry, letting any client
        # choose its own bucket by sending a header.
        label="leftmost XFF entry trusted (client picks its own bucket)",
        file="src/df/gateway/app.py",
        old="    candidate = parts[-hops]",
        new="    candidate = parts[0]",
        witness=RATELIMIT_WITNESS,
        test="tests/test_rate_limit_identity.py",
    ),
    Mutation(
        # Drop the unmapping: ::ffff:1.2.3.4 gets its own bucket, doubling
        # an allowance for anyone who sends the mapped form.
        label="ipv4-mapped address not unmapped (two buckets per client)",
        file="src/df/gateway/app.py",
        old='    mapped = getattr(addr, "ipv4_mapped", None)',
        new="    mapped = None",
        witness=RATELIMIT_WITNESS,
        test="tests/test_rate_limit_identity.py",
    ),
    Mutation(
        # Fail toward a shared constant instead of the peer: every caller
        # lands in one bucket and the service 429s the world.
        label="malformed header falls back to a shared constant",
        file="src/df/gateway/app.py",
        old="    if len(parts) < hops:",
        new="    if False:",
        witness=RATELIMIT_WITNESS,
        test="tests/test_rate_limit_identity.py",
    ),
]

DECODE_WITNESS = (
    "import cv2, numpy as np;"
    "from df.pipelines.extract import decode_image, sniff_format;"
    "from df.gateway.app import _decode_advisories as adv;"
    "img = np.zeros((40, 60, 3), np.uint8);"
    "jpg = cv2.imencode('.jpg', img)[1].tobytes();"
    "heic = bytes(4) + b'ftypheic' + bytes(16);"
    "print(sniff_format(heic), sniff_format(jpg),"
    "      decode_image(jpg) is not None, decode_image(b'') is None,"
    "      len(adv({'decodable': False, 'media_format': 'heic'})),"
    "      len(adv({'decodable': True})), len(adv({})))"
)

DECODE_MUTATIONS = [
    Mutation(
        # HEIC misidentified -- the reason a judge would be told "no face".
        label="ftyp brands not recognised (heic reads as unknown)",
        file="src/df/pipelines/extract.py",
        old='    if raw[4:8] == b"ftyp":',
        new="    if False:",
        witness=DECODE_WITNESS,
        test="tests/test_media_decoding.py",
    ),
    Mutation(
        # The only thing standing between an empty upload and a dead worker.
        # An explicit empty-buffer guard used to sit alongside this; it was
        # removed precisely because two mechanisms guarding one property made
        # the property untestable -- every single-line mutation was a no-op
        # and the harness rightly refused a verdict.
        label="cv2.error not handled (empty upload crashes the worker)",
        file="src/df/pipelines/extract.py",
        old="    except cv2.error:",
        new="    except ZeroDivisionError:",
        witness=DECODE_WITNESS,
        test="tests/test_media_decoding.py",
    ),
    Mutation(
        # Fail open: an undecodable upload reports nothing, so the result
        # reads as "no face present" rather than "not analysed".
        label="undecodable upload reported silently",
        file="src/df/gateway/app.py",
        old='    if preprocess.get("decodable") is not False:',
        new="    if True:",
        witness=DECODE_WITNESS,
        test="tests/test_media_decoding.py",
    ),
    Mutation(
        # Absent treated as False: every pre-gate job gets a false warning.
        label="missing decodability treated as undecodable",
        file="src/df/gateway/app.py",
        old='    if preprocess.get("decodable") is not False:',
        new='    if preprocess.get("decodable"):',
        witness=DECODE_WITNESS,
        test="tests/test_media_decoding.py",
    ),
]

GATE_WITNESS = (
    # Three probes, each discriminating a different mutation:
    #  A: the measured portrait -- does the artefact gate at all?
    #  B: a LONE detection at 0.30, deliberately BELOW the 0.4 ratio value.
    #     Relative keeps it (ratio 1.0); an absolute floor gates it and
    #     empties the set. A lone 0.44 cannot tell those apart, which is why
    #     the first version of this witness reported NO-OP.
    #  C: two STRONG detections. Needed to catch "only the best survives",
    #     which is invisible whenever the input has one clear winner.
    "import os; os.environ['DF_DETECTION_CONFIDENCE_RATIO'] = '0.4';"
    "from df.pipelines.extract import FaceCrop;"
    "import df.workers.cpu_preprocess as cp;"
    "from df.config import Settings; cp.settings = Settings();"
    "c = lambda i, k: FaceCrop(frame_index=0, face_index=i, data=b'x',"
    "                          confidence=k, bbox=(1, 1, 120, 120));"
    "dA = [];"
    "A = cp._gate_detections([c(0, 0.968), c(1, 0.316), c(2, 0.075)], dA, frame_index=0);"
    "B = cp._gate_detections([c(9, 0.30)], [], frame_index=0);"
    "C = cp._gate_detections([c(3, 0.95), c(4, 0.80)], [], frame_index=0);"
    "print([x.face_index for x in A], len(dA),"
    "      [x.face_index for x in B], [x.face_index for x in C])"
)

GATE_MUTATIONS = [
    Mutation(
        label="detection gate disabled",
        file="src/df/workers/cpu_preprocess.py",
        old="        if rel < ratio:",
        new="        if False:",
        witness=GATE_WITNESS,
        test="tests/test_detection_gate.py",
    ),
    Mutation(
        # Gate, but drop silently -- the 2026-08-30 bug reintroduced.
        label="gated detections not recorded",
        file="src/df/workers/cpu_preprocess.py",
        old="            discarded.append({",
        new="            _ = ({",
        witness=GATE_WITNESS,
        test="tests/test_detection_gate.py",
    ),
    Mutation(
        # The whole point of the relative form: an ABSOLUTE floor can empty
        # the set, which is what leaves a lone marginal face undetermined.
        label="absolute floor instead of a ratio (can empty the set)",
        file="src/df/workers/cpu_preprocess.py",
        old="        rel = crop.confidence / best",
        new="        rel = crop.confidence",
        witness=GATE_WITNESS,
        test="tests/test_detection_gate.py",
    ),
    Mutation(
        # Keep only the single best detection, discarding a genuine second
        # face -- the case worst-case rollup exists for.
        label="only the best detection survives",
        file="src/df/workers/cpu_preprocess.py",
        old="        if rel < ratio:",
        new="        if rel < 1.0:",
        witness=GATE_WITNESS,
        test="tests/test_detection_gate.py",
    ),
]

ZERO_ITEM_WITNESS = (
    # Drives the real router handler with a job that has no item rows and
    # prints what lands on the audit row. Behavioural, not source-inspecting:
    # a witness that greps the source would agree with any mutation keeping
    # the same text, which is the opposite of the point.
    "import sys; sys.path.insert(0, 'tests');"
    "from fakes import FakeDb;"
    "from df.queue import Message;"
    "from df.storage import InMemoryStorage;"
    "import df.workers.router as router;"
    "S = type('S', (), {'publish': lambda self, *a, **k: None})();"
    "db = FakeDb(); db.add_job('j1', 'video', status='queued');"
    "router.handle(Message(id='m1', topic='aggregate', payload={"
    "    'job_id': 'j1', 'media_type': 'video',"
    "    'model_version_id': 'face-from-the-queue-message'}, attempts=0),"
    "    db=db, storage=InMemoryStorage(), status=S);"
    "print(repr(db.jobs['j1']['model_version_id']))"
)

ZERO_ITEM_MUTATIONS = [
    Mutation(
        # The behaviour that shipped: with no rows, take the queue message's
        # claim and stamp it on an undetermined row where that model never ran.
        label="zero-item job credited to the queue message model",
        file="src/df/workers/router.py",
        old="    if items:",
        new="    if True:",
        witness=ZERO_ITEM_WITNESS,
        test="tests/test_end_to_end.py",
    ),
]

FACE_EXTRACT_MUTATIONS = [
    Mutation(
        # Back to squaring the crop. Does not raise now that the constant is a
        # tuple again -- it just quietly destroys the aspect ratio and makes the
        # detector's careful resize a no-op.
        label="crop squashed to a square before the detector sees it",
        file="src/df/pipelines/extract.py",
        old="            crop = arr[y : y + h, x : x + w]",
        new="            crop = cv2.resize(arr[y : y + h, x : x + w], (380, 380))",
        witness=EXTRACT_FACE_WITNESS,
        test="tests/test_face_extraction.py",
    ),
    Mutation(
        # Reinstate the extractor-level confidence filter, which drops weak
        # detections with no row, no count and nothing for a dispute to read.
        label="low-confidence faces silently dropped again",
        file="src/df/pipelines/extract.py",
        old="            crops.append(\n                FaceCrop(",
        new="            if _haar_confidence(weight) < 0.3:\n                continue\n"
            "            crops.append(\n                FaceCrop(",
        witness=EXTRACT_FACE_WITNESS,
        test="tests/test_face_extraction.py",
    ),
    Mutation(
        # Unbounded confidence. Aggregation multiplies by this, so a value above
        # 1 does not fail -- it reweights the mean in favour of one detection.
        label="haar confidence not clamped to 0-1",
        file="src/df/pipelines/extract.py",
        old="    return float(min(1.0, max(0.0, weight / 10.0)))",
        new="    return float(weight / 10.0)",
        witness=(
            "from df.pipelines.extract import _haar_confidence as c;"
            "print(c(-5.0), c(3.0), c(1000.0))"
        ),
        test="tests/test_face_extraction.py",
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

    print("\ncalibration suite (fitter, scheme honesty, advisory)")
    c_bad = run_all(CALIBRATION_MUTATIONS)
    print(f"  {len(CALIBRATION_MUTATIONS)} mutation(s), {c_bad} did not produce RED")

    print("\ncalibration-set extraction")
    e_bad = run_all(EXTRACT_MUTATIONS)
    print(f"  {len(EXTRACT_MUTATIONS)} mutation(s), {e_bad} did not produce RED")

    print("\nreal face extractor")
    f_bad = run_all(FACE_EXTRACT_MUTATIONS)
    print(f"  {len(FACE_EXTRACT_MUTATIONS)} mutation(s), {f_bad} did not produce RED")

    print("\nzero-item attribution")
    z_bad = run_all(ZERO_ITEM_MUTATIONS)
    print(f"  {len(ZERO_ITEM_MUTATIONS)} mutation(s), {z_bad} did not produce RED")

    print("\nrate-limit identity")
    rl_bad = run_all(RATELIMIT_MUTATIONS)
    print(f"  {len(RATELIMIT_MUTATIONS)} mutation(s), {rl_bad} did not produce RED")

    print("\nmedia decoding")
    d_bad = run_all(DECODE_MUTATIONS)
    print(f"  {len(DECODE_MUTATIONS)} mutation(s), {d_bad} did not produce RED")

    print("\ndetection gate")
    g_bad = run_all(GATE_MUTATIONS)
    print(f"  {len(GATE_MUTATIONS)} mutation(s), {g_bad} did not produce RED")
