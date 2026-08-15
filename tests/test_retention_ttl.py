"""TTL deletion tests.

CLAUDE.md names this as the gap to close before Tier 1 can be called done: the
original plan load-tested the rate limiter but never asserted that TTL deletion
actually deletes.

These tests assert on the STORAGE BACKEND, not on a mock's call log. A mock that
records `delete_object(...)` was called proves the code made a call; it does not
prove the bytes are gone. InMemoryStorage is queried directly for the absence of
every key.
"""
from __future__ import annotations

import datetime as dt

from df.retention import (
    DeleteOutcome,
    delete_media_for_job,
    open_extended_retention_window,
    sweep_undeleted,
)


def test_raw_media_and_face_crops_are_actually_gone(store, storage, seeded_job):
    job_id, raw, derived = seeded_job()

    assert storage.exists(raw)
    assert len(storage.list_prefix(derived)) == 3

    report = delete_media_for_job(job_id, store, storage)

    assert report.outcome is DeleteOutcome.DELETED
    # The actual assertion that matters: the bytes are not in the bucket.
    assert not storage.exists(raw), "raw upload survived TTL delete"
    assert storage.list_prefix(derived) == [], "face crops survived TTL delete"
    assert report.raw_deleted is True
    assert report.derived_objects_deleted == 3


def test_delete_marks_the_row_and_records_an_event(store, storage, seeded_job):
    job_id, _, _ = seeded_job()

    delete_media_for_job(job_id, store, storage)

    row = store.get_retention_row(job_id)
    assert row.raw_deleted_at is not None
    assert row.derived_deleted_at is not None
    # A deleted job must still be able to prove it was deleted.
    assert "retention.deleted" in store.event_names(job_id)


def test_hold_flag_blocks_deletion(store, storage, seeded_job):
    job_id, raw, derived = seeded_job("job-held", hold=True)

    report = delete_media_for_job(job_id, store, storage)

    assert report.outcome is DeleteOutcome.SKIPPED_HOLD
    assert storage.exists(raw), "held job's raw media was deleted"
    assert len(storage.list_prefix(derived)) == 3, "held job's face crops were deleted"
    assert store.get_retention_row(job_id).raw_deleted_at is None
    assert "retention.delete_skipped" in store.event_names(job_id)


def test_hold_flag_is_checked_before_any_storage_call(store, storage, seeded_job):
    """The hold check must gate the delete, not merely accompany it."""
    job_id, _, _ = seeded_job("job-held-2", hold=True)

    calls: list[str] = []
    original_delete_object = storage.delete_object
    original_delete_prefix = storage.delete_prefix
    storage.delete_object = lambda k: (calls.append(f"object:{k}"), original_delete_object(k))[1]
    storage.delete_prefix = lambda p: (calls.append(f"prefix:{p}"), original_delete_prefix(p))[1]

    delete_media_for_job(job_id, store, storage)

    assert calls == [], f"storage was touched despite the hold flag: {calls}"


def test_delete_is_idempotent(store, storage, seeded_job):
    job_id, _, _ = seeded_job()

    first = delete_media_for_job(job_id, store, storage)
    second = delete_media_for_job(job_id, store, storage)

    assert first.outcome is DeleteOutcome.DELETED
    assert second.outcome is DeleteOutcome.ALREADY_DELETED


def test_missing_job_does_not_delete_anything(store, storage, seeded_job):
    job_id, raw, _ = seeded_job()

    report = delete_media_for_job("no-such-job", store, storage)

    assert report.outcome is DeleteOutcome.NOT_FOUND
    assert storage.exists(raw), "unrelated job's media was deleted"


def test_delete_only_touches_its_own_job(store, storage, seeded_job):
    job_a, raw_a, derived_a = seeded_job("job-a")
    job_b, raw_b, derived_b = seeded_job("job-b")

    delete_media_for_job(job_a, store, storage)

    assert not storage.exists(raw_a)
    assert storage.exists(raw_b), "deleting job-a removed job-b's raw media"
    assert len(storage.list_prefix(derived_b)) == 3


def test_sweeper_deletes_media_the_completion_path_missed(store, storage, seeded_job):
    """Simulates a router crash between writing the verdict and deleting bytes."""
    job_id, raw, derived = seeded_job("job-orphan")
    store.terminal_undeleted = [job_id]

    reports = sweep_undeleted(store, storage)

    assert [r.outcome for r in reports] == [DeleteOutcome.DELETED]
    assert not storage.exists(raw)
    assert storage.list_prefix(derived) == []


def test_sweeper_respects_the_hold_flag(store, storage, seeded_job):
    job_id, raw, _ = seeded_job("job-orphan-held", hold=True)
    store.terminal_undeleted = [job_id]

    reports = sweep_undeleted(store, storage)

    assert [r.outcome for r in reports] == [DeleteOutcome.SKIPPED_HOLD]
    assert storage.exists(raw)


def test_extended_retention_window_is_a_fixed_timer_not_a_hold(store, storage, seeded_job):
    """Tier 2 wording and semantics: fixed 30-day timer, auto-expires."""
    job_id, _, derived = seeded_job("job-hi")
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    until, _ = open_extended_retention_window(
        job_id, store, storage, driving_keys=storage.list_prefix(derived), days=30, now=now
    )

    assert until == now + dt.timedelta(days=30)
    # Opening the window must NOT set the hold flag -- they are different things.
    assert not store.rows[job_id].retention_hold

    _, event, detail = store.events[-1]
    assert event == "retention.extended_window_opened"
    assert "not a legal hold" in detail["note"]


def test_window_preserves_the_driving_crops_but_not_the_raw_source(store, storage, seeded_job):
    """CLAUDE.md: the window protects the flagged media itself -- specifically
    the face crops that drove the score, NOT the full raw source.

    A window that only re-protected the job row would leave the >80 branch with
    nothing a dispute could actually use.
    """
    job_id, raw, derived = seeded_job("job-highband")
    driving = storage.list_prefix(derived)[:2]   # the crops that set the score

    _, preserved = open_extended_retention_window(
        job_id, store, storage, driving_keys=driving, days=30
    )
    report = delete_media_for_job(job_id, store, storage)

    assert report.outcome is DeleteOutcome.DELETED
    # Tier 1 still wins over the raw source and the working copies.
    assert not storage.exists(raw), "raw source must still be deleted"
    assert storage.list_prefix(derived) == [], "working derived copies must still be deleted"
    # ...but the evidence survives.
    assert len(preserved) == 2
    assert all(storage.exists(k) for k in preserved), "driving crops were not preserved"


def test_window_preserves_only_the_crops_that_drove_the_score(store, storage, seeded_job):
    job_id, _, derived = seeded_job("job-selective")
    all_crops = storage.list_prefix(derived)
    driving = [all_crops[1]]

    _, preserved = open_extended_retention_window(
        job_id, store, storage, driving_keys=driving, days=30
    )
    delete_media_for_job(job_id, store, storage)

    cold = storage.list_prefix(f"cold/{job_id}/")
    assert len(cold) == 1, f"expected only the driving crop in cold storage, got {cold}"
