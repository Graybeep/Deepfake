"""The hold-flag gate, across every delete trigger that can touch flagged media.

CLAUDE.md's stated testing gap: it is not enough to show the hold flag blocks the
primary completion-triggered delete, and not enough to show the job row was left
alone. What has to hold is that the *preserved media itself* -- the face crops
that drove a flagged score -- survives every code path capable of deleting it,
while a job with no hold behaves normally on that same path.

The three delete triggers:
  1. delete_media_for_job()      -- completion
  2. sweep_undeleted()           -- crash recovery
  3. expire_extended_retention() -- cold-storage timer expiry

Each is tested twice: held (media survives) and unheld (media goes). A test that
only checked the held case would pass against a delete path that was simply
broken.
"""
from __future__ import annotations

import datetime as dt

import pytest

from df.retention import (
    DeleteOutcome,
    delete_media_for_job,
    expire_extended_retention,
    open_extended_retention_window,
    sweep_expired_windows,
    sweep_undeleted,
)

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
AFTER_EXPIRY = NOW + dt.timedelta(days=31)


@pytest.fixture
def flagged(store, storage, seeded_job):
    """A >80 job: window open, driving crops preserved in cold storage."""

    def _make(job_id: str = "job-flagged", *, hold: bool = False):
        _, raw, derived = seeded_job(job_id, hold=hold)
        driving = storage.list_prefix(derived)[:2]
        _, preserved = open_extended_retention_window(
            job_id, store, storage, driving_keys=driving, days=30, now=NOW
        )
        assert preserved, "fixture failed to preserve anything"
        return job_id, raw, derived, preserved

    return _make


# --- trigger 1: the completion-triggered delete -----------------------------


def test_hold_blocks_completion_delete_of_preserved_media(store, storage, flagged):
    job_id, raw, _, preserved = flagged("job-hold-1", hold=True)

    report = delete_media_for_job(job_id, store, storage)

    assert report.outcome is DeleteOutcome.SKIPPED_HOLD
    assert all(storage.exists(k) for k in preserved), "preserved crops were deleted"
    assert storage.exists(raw)


def test_without_hold_completion_delete_clears_working_copies(store, storage, flagged):
    job_id, raw, derived, preserved = flagged("job-nohold-1", hold=False)

    report = delete_media_for_job(job_id, store, storage)

    assert report.outcome is DeleteOutcome.DELETED
    assert not storage.exists(raw)
    assert storage.list_prefix(derived) == []
    # The window's whole purpose: these outlive the Tier 1 delete.
    assert all(storage.exists(k) for k in preserved)


# --- trigger 2: the crash-recovery sweeper ----------------------------------


def test_hold_blocks_sweeper_delete_of_preserved_media(store, storage, flagged):
    job_id, raw, _, preserved = flagged("job-hold-2", hold=True)
    store.completed_undeleted = [job_id]

    reports = sweep_undeleted(store, storage)

    assert [r.outcome for r in reports] == [DeleteOutcome.SKIPPED_HOLD]
    assert all(storage.exists(k) for k in preserved), "sweeper deleted preserved crops"
    assert storage.exists(raw)


def test_without_hold_sweeper_clears_working_copies(store, storage, flagged):
    job_id, raw, derived, preserved = flagged("job-nohold-2", hold=False)
    store.completed_undeleted = [job_id]

    reports = sweep_undeleted(store, storage)

    assert [r.outcome for r in reports] == [DeleteOutcome.DELETED]
    assert not storage.exists(raw)
    assert all(storage.exists(k) for k in preserved)


# --- trigger 3: cold-storage expiry -----------------------------------------


def test_hold_blocks_window_expiry_delete_of_preserved_media(store, storage, flagged):
    """The flag has to outrank the timer. If expiry could delete held media,
    a hold set during a dispute would evaporate on day 30 anyway."""
    job_id, _, _, preserved = flagged("job-hold-3", hold=True)

    report = expire_extended_retention(job_id, store, storage, now=AFTER_EXPIRY)

    assert report.outcome is DeleteOutcome.SKIPPED_HOLD
    assert all(storage.exists(k) for k in preserved), "expiry deleted held media"
    assert "retention.cold_delete_skipped" in store.event_names(job_id)


def test_without_hold_expiry_deletes_preserved_media(store, storage, flagged):
    job_id, _, _, preserved = flagged("job-nohold-3", hold=False)

    report = expire_extended_retention(job_id, store, storage, now=AFTER_EXPIRY)

    assert report.outcome is DeleteOutcome.DELETED
    assert report.cold_objects_deleted == len(preserved)
    assert not any(storage.exists(k) for k in preserved)
    assert storage.list_prefix(f"cold/{job_id}/") == []


def test_hold_blocks_expiry_via_the_sweeper_too(store, storage, flagged):
    """Same gate, reached through the scheduled sweep rather than directly."""
    held, _, _, held_keys = flagged("job-hold-4", hold=True)
    free, _, _, free_keys = flagged("job-nohold-4", hold=False)

    reports = sweep_expired_windows(store, storage, now=AFTER_EXPIRY)
    outcomes = {r.job_id: r.outcome for r in reports}

    assert outcomes[held] is DeleteOutcome.SKIPPED_HOLD
    assert outcomes[free] is DeleteOutcome.DELETED
    assert all(storage.exists(k) for k in held_keys)
    assert not any(storage.exists(k) for k in free_keys)


def test_window_does_not_expire_early(store, storage, flagged):
    """A fixed timer that fires early is just data loss."""
    job_id, _, _, preserved = flagged("job-early", hold=False)

    report = expire_extended_retention(
        job_id, store, storage, now=NOW + dt.timedelta(days=29)
    )

    assert report.outcome is DeleteOutcome.NOT_EXPIRED
    assert all(storage.exists(k) for k in preserved)


def test_expiry_is_idempotent(store, storage, flagged):
    job_id, _, _, _ = flagged("job-twice", hold=False)

    first = expire_extended_retention(job_id, store, storage, now=AFTER_EXPIRY)
    second = expire_extended_retention(job_id, store, storage, now=AFTER_EXPIRY)

    assert first.outcome is DeleteOutcome.DELETED
    assert second.outcome is DeleteOutcome.ALREADY_DELETED


def test_expiry_of_one_job_leaves_another_jobs_preserved_media(store, storage, flagged):
    a, _, _, a_keys = flagged("job-a", hold=False)
    b, _, _, b_keys = flagged("job-b", hold=False)

    expire_extended_retention(a, store, storage, now=AFTER_EXPIRY)

    assert not any(storage.exists(k) for k in a_keys)
    assert all(storage.exists(k) for k in b_keys), "expiry crossed job boundaries"


def test_every_delete_trigger_records_why_it_skipped(store, storage, flagged):
    """A held job that was never deleted must be able to prove the gate fired."""
    job_id, _, _, _ = flagged("job-audit", hold=True)
    store.completed_undeleted = [job_id]

    delete_media_for_job(job_id, store, storage)
    sweep_undeleted(store, storage)
    expire_extended_retention(job_id, store, storage, now=AFTER_EXPIRY)

    names = store.event_names(job_id)
    assert names.count("retention.delete_skipped") == 2   # completion + sweeper
    assert names.count("retention.cold_delete_skipped") == 1
