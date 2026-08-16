"""Which jobs the sweeper is willing to look at.

The hold gate was already covered across every delete trigger
(test_retention_hold_gate.py). This covers the step before it: whether a job
is ever *offered* to that gate at all. A delete path that is never reached is
indistinguishable from one that does not exist, and that is exactly how
dead-lettered jobs kept their media.

Found by inspecting a live stack: a job that dead-lettered left its raw upload
and eight face crops in the bucket with no expiry. The sweeper selected on
`completed_at IS NOT NULL`, and a dead-lettered job never gets completed_at
set -- while the completion-triggered delete never fires for it either,
because it never completed. Neither Tier 1 delete path covered it.

Failures are also correlated: one bad deploy dead-letters every job that
arrives during it, so this leaks in bulk exactly when nobody is watching.
"""
from __future__ import annotations

import datetime as dt
import inspect
import pathlib
import re

import pytest

from df.db import Db

from df.retention import (
    DeleteOutcome,
    delete_media_for_job,
    sweep_abandoned_uploads,
    sweep_stalled_jobs,
    sweep_undeleted,
)
from df.storage import InMemoryStorage
from tests.fakes import FakeDb

NOW = dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)
LONG_AGO = NOW - dt.timedelta(days=3)


@pytest.fixture
def db() -> FakeDb:
    return FakeDb()


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


def seed(db: FakeDb, storage: InMemoryStorage, job_id: str, *, status: str,
         hold: bool = False, created_at: dt.datetime = LONG_AGO,
         completed: bool = False) -> tuple[str, str]:
    """A job holding real bytes: one raw upload and three face crops."""
    raw = f"raw/{job_id}/original"
    derived = f"derived/{job_id}/"
    storage.put_bytes(raw, b"raw-video-bytes")
    for i in range(3):
        storage.put_bytes(f"{derived}f{i:05d}_x0.png", f"crop-{i}".encode())
    db.add_job(job_id, "video", retention_hold=hold, status=status, created_at=created_at)
    if completed:
        db.jobs[job_id]["completed_at"] = created_at
    return raw, derived


def media_present(storage: InMemoryStorage, raw: str, derived: str) -> bool:
    return storage.exists(raw) or storage.list_prefix(derived) != []


# --- terminal-state sweep ---------------------------------------------------


@pytest.mark.parametrize("status", ["dead_letter", "failed"])
def test_failed_jobs_media_is_swept(db, storage, status):
    """The regression. Neither delete path reached these before."""
    raw, derived = seed(db, storage, f"job-{status}", status=status)

    reports = sweep_undeleted(db, storage)

    assert [r.outcome for r in reports] == [DeleteOutcome.DELETED]
    assert not media_present(storage, raw, derived), "failed job kept its media"


def test_completed_jobs_are_still_swept(db, storage):
    """Widening the predicate must not lose the case it already covered."""
    raw, derived = seed(db, storage, "job-done", status="complete", completed=True)

    reports = sweep_undeleted(db, storage)

    assert [r.outcome for r in reports] == [DeleteOutcome.DELETED]
    assert not media_present(storage, raw, derived)


def test_hold_still_blocks_a_dead_lettered_job(db, storage):
    """The new predicate must not become a way around the hold flag.

    Two independent layers, asserted separately. The query excludes held jobs
    so they are never offered up; the gate inside delete_media_for_job refuses
    them even when they are. Either alone would protect the media today, but
    the query filter is the one a future widening could drop.
    """
    raw, derived = seed(db, storage, "job-held-dl", status="dead_letter", hold=True)

    # layer 1: never even selected
    assert sweep_undeleted(db, storage) == []
    assert storage.exists(raw), "hold did not protect a dead-lettered job"
    assert len(storage.list_prefix(derived)) == 3

    # layer 2: and refused if handed to the delete path directly
    report = delete_media_for_job("job-held-dl", db, storage)
    assert report.outcome is DeleteOutcome.SKIPPED_HOLD
    assert storage.exists(raw)
    assert "retention.delete_skipped" in db.event_names("job-held-dl")


def test_in_flight_jobs_are_left_alone(db, storage):
    """A job still being processed is not terminal. Deleting its media
    mid-pipeline would fail the job it was about to score."""
    raw, derived = seed(db, storage, "job-running", status="inference")

    assert sweep_undeleted(db, storage) == []
    assert media_present(storage, raw, derived)


# --- stalled in-flight sweep ------------------------------------------------


@pytest.mark.parametrize(
    "status", ["queued", "preprocessing", "inference", "aggregating"]
)
def test_stalled_in_flight_jobs_are_failed_and_cleared(db, storage, status):
    """The third instance of this bug, found by inspection rather than by leak.

    The gateway and workers both commit the status change before pushing to
    Redis. A crash in between -- or a push lost in Redis's AOF everysec window
    -- leaves a row in an in-flight state with no message to advance it. It
    never completes and never dead-letters, so no other sweep covers it.
    """
    raw, derived = seed(db, storage, f"job-{status}", status=status,
                        created_at=LONG_AGO)
    db.jobs[f"job-{status}"]["updated_at"] = LONG_AGO

    reports = sweep_stalled_jobs(db, storage, older_than_hours=6, now=NOW)

    assert [r.outcome for r in reports] == [DeleteOutcome.DELETED]
    assert not media_present(storage, raw, derived)
    assert db.jobs[f"job-{status}"]["status"] == "failed", "row left non-terminal"
    assert "job.stalled" in db.event_names(f"job-{status}")


def test_a_recently_active_job_is_not_stalled(db, storage):
    """Deleting media under a job that is merely slow would fail work that was
    about to succeed. The threshold is the only thing separating the two."""
    raw, derived = seed(db, storage, "job-busy", status="inference")
    db.jobs["job-busy"]["updated_at"] = NOW - dt.timedelta(minutes=2)

    assert sweep_stalled_jobs(db, storage, older_than_hours=6, now=NOW) == []
    assert media_present(storage, raw, derived)
    assert db.jobs["job-busy"]["status"] == "inference"


def test_hold_blocks_the_stalled_sweep(db, storage):
    """A held job may be marked failed -- that is a status change, not a
    delete -- but its media must survive."""
    raw, derived = seed(db, storage, "job-held-stall", status="inference", hold=True)
    db.jobs["job-held-stall"]["updated_at"] = LONG_AGO

    reports = sweep_stalled_jobs(db, storage, older_than_hours=6, now=NOW)

    assert [r.outcome for r in reports] == [DeleteOutcome.SKIPPED_HOLD]
    assert storage.exists(raw), "hold did not protect a stalled job"
    assert len(storage.list_prefix(derived)) == 3


def test_abandoned_upload_row_becomes_terminal(db, storage):
    """Clearing bytes without moving the status leaves a row that grows the
    table forever and that nothing can tell apart from a live job."""
    seed(db, storage, "job-ghost2", status="awaiting_upload")

    sweep_abandoned_uploads(db, storage, older_than_hours=24, now=NOW)

    assert db.jobs["job-ghost2"]["status"] == "failed"


# --- the enumeration itself -------------------------------------------------


def schema_statuses() -> set[str]:
    """Every value the migration's CHECK constraint permits."""
    sql = pathlib.Path(__file__).resolve().parents[1] / "migrations" / "001_init.sql"
    block = re.search(r"CHECK \(status IN \((.*?)\)\)", sql.read_text(encoding="utf-8"), re.S)
    assert block, "could not find the status CHECK constraint"
    return set(re.findall(r"'([a-z_]+)'", block.group(1)))


def status_literals_in(fn) -> set[str]:
    """Status values named in the SQL this function actually executes.

    Read out of the live source rather than restated here. A test that carries
    its own list of what the sweeps 'should' cover is a second description of
    the code that can drift from it -- the same failure that let FakeDb diverge
    from psycopg3 and hide a broken write path behind a green suite.
    """
    src = inspect.getsource(fn)
    found: set[str] = set()
    for m in re.finditer(r"status\s+IN\s*\(([^)]*)\)", src, re.I):
        found |= set(re.findall(r"'([a-z_]+)'", m.group(1)))
    found |= set(re.findall(r"status\s*=\s*'([a-z_]+)'", src, re.I))
    return found


# The finders whose predicates decide what each sweep will ever see.
FINDERS = {
    "sweep_undeleted": Db.find_undeleted_terminal,
    "sweep_stalled_jobs": Db.find_stalled_in_flight,
    "sweep_abandoned_uploads": Db.find_abandoned_uploads,
}


def test_the_real_sql_predicates_cover_every_status_the_schema_allows():
    """Reads both sides from code: the constraint from the migration, the
    covered set from the SQL in db.py.

    Deleting a status from a predicate makes this fail, which a maintained
    list in the test file would not. Manual auditing missed a status three
    times running, so the check has to look at what executes.
    """
    covered: set[str] = set()
    for name, fn in FINDERS.items():
        named = status_literals_in(fn)
        assert named, f"{name}: no status literal found in {fn.__name__} -- extraction broke"
        covered |= named

    # 'complete' is reached by completed_at IS NOT NULL, not by naming the
    # status. Asserted against the source so removing that clause fails here.
    terminal_src = inspect.getsource(Db.find_undeleted_terminal)
    assert re.search(r"completed_at\s+IS\s+NOT\s+NULL", terminal_src, re.I), (
        "find_undeleted_terminal no longer selects on completed_at, so nothing "
        "reaches 'complete'"
    )
    covered.add("complete")

    declared = schema_statuses()
    assert declared == covered, (
        f"statuses with no sweep predicate: {sorted(declared - covered)}; "
        f"predicates naming a status the schema forbids: {sorted(covered - declared)}"
    )


@pytest.mark.parametrize("status", sorted(schema_statuses()))
def test_every_schema_status_is_claimed_by_exactly_one_sweep(db, storage, status):
    """The behavioural counterpart, run through the fakes.

    The test above proves the real SQL names every status. This proves a job
    actually holding each one gets picked up -- and by exactly one sweep, so
    no status is double-owned or silently owned by nobody.
    """
    job_id = f"job-{status}"
    raw, derived = seed(db, storage, job_id, status=status,
                        completed=(status == "complete"))
    db.jobs[job_id]["updated_at"] = LONG_AGO

    claimed = {
        "terminal": [r.job_id for r in sweep_undeleted(db, storage)],
        "stalled": [r.job_id for r in sweep_stalled_jobs(
            db, storage, older_than_hours=6, now=NOW)],
        "abandoned": [r.job_id for r in sweep_abandoned_uploads(
            db, storage, older_than_hours=24, now=NOW)],
    }
    owners = [name for name, ids in claimed.items() if job_id in ids]

    assert owners, f"status {status!r} is claimed by no sweep -- its media is never deleted"
    assert len(owners) == 1, f"status {status!r} claimed by more than one sweep: {owners}"
    assert not media_present(storage, raw, derived)


# --- abandoned upload sweep -------------------------------------------------


def test_abandoned_upload_media_is_cleared(db, storage):
    """Granted an upload URL, uploaded, never notified. The job never enters
    the pipeline, so no terminal state is ever reached."""
    raw, derived = seed(db, storage, "job-ghost", status="awaiting_upload")

    reports = sweep_abandoned_uploads(db, storage, older_than_hours=24, now=NOW)

    assert [r.outcome for r in reports] == [DeleteOutcome.DELETED]
    assert not media_present(storage, raw, derived)


def test_a_recent_upload_grant_is_not_abandoned(db, storage):
    """Deleting under a client still legitimately uploading would be worse
    than the leak this sweep exists to close."""
    raw, derived = seed(db, storage, "job-fresh", status="awaiting_upload",
                        created_at=NOW - dt.timedelta(minutes=5))

    assert sweep_abandoned_uploads(db, storage, older_than_hours=24, now=NOW) == []
    assert media_present(storage, raw, derived)


def test_hold_blocks_the_abandoned_sweep(db, storage):
    """Same two layers as the terminal sweep."""
    raw, _ = seed(db, storage, "job-held-ghost", status="awaiting_upload", hold=True)

    assert sweep_abandoned_uploads(db, storage, older_than_hours=24, now=NOW) == []
    assert storage.exists(raw)

    report = delete_media_for_job("job-held-ghost", db, storage)
    assert report.outcome is DeleteOutcome.SKIPPED_HOLD
    assert storage.exists(raw)


def test_the_two_sweeps_do_not_overlap(db, storage):
    """Each job is owned by exactly one sweep, so neither double-deletes nor
    assumes the other will handle it."""
    dl_raw, dl_derived = seed(db, storage, "job-dl", status="dead_letter")
    gh_raw, gh_derived = seed(db, storage, "job-gh", status="awaiting_upload")

    terminal = [r.job_id for r in sweep_undeleted(db, storage)]
    abandoned = [r.job_id for r in sweep_abandoned_uploads(db, storage,
                                                           older_than_hours=24, now=NOW)]

    assert terminal == ["job-dl"]
    assert abandoned == ["job-gh"]
    assert not media_present(storage, dl_raw, dl_derived)
    assert not media_present(storage, gh_raw, gh_derived)
