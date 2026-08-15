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

import pytest

from df.retention import (
    DeleteOutcome,
    delete_media_for_job,
    sweep_abandoned_uploads,
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
