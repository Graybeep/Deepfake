"""Retention and deletion.

Tier 1 (non-negotiable): raw video/image/audio and every derived face crop are
deleted when inference completes. Every delete call reads the hold flag first.

Tier 2: high-band results get an "extended retention window" -- a FIXED 30-day
timer. It is NOT a legal hold. It auto-expires on its own, including in the
middle of an active dispute, which is exactly why it must never be called or
labelled a hold anywhere in this codebase or in user-facing copy.

The window protects THE FLAGGED MEDIA ITSELF -- specifically the face crops (or
spectrogram chunks) that drove the score -- not just the job row. The row and
per-item scores are already retained for every job regardless of band, so a
window that only re-protected those would leave the >80 branch with nothing a
dispute could actually use. The full raw source is NOT preserved: Tier 1 deletes
it on completion for every band.

This is the engineering default. It still needs real legal sign-off before being
relied on for an actual dispute.

THREE delete paths can touch this media, and the Tier 1 hold-flag check gates
every one of them:
  * delete_media_for_job()      -- the completion-triggered delete
  * sweep_undeleted()           -- crash recovery, delegates to the above
  * expire_extended_retention() -- cold-storage expiry once the timer runs out
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from df import storage as storage_mod
from df.config import settings

log = logging.getLogger("df.retention")


class DeleteOutcome(str, Enum):
    DELETED = "deleted"
    SKIPPED_HOLD = "skipped_retention_hold"
    ALREADY_DELETED = "already_deleted"
    NOT_FOUND = "job_not_found"
    # Cold-storage expiry only: the fixed timer has not run out yet.
    NOT_EXPIRED = "window_not_expired"


@dataclass
class RetentionRow:
    """The subset of the job row the delete paths need."""

    job_id: str
    retention_hold: bool
    raw_object_key: str | None
    derived_prefix: str | None
    raw_deleted_at: dt.datetime | None
    derived_deleted_at: dt.datetime | None
    # Cold storage holding the preserved driving media, if the window is open.
    cold_prefix: str | None = None
    cold_deleted_at: dt.datetime | None = None
    extended_retention_until: dt.datetime | None = None


class RetentionStore(Protocol):
    """Persistence the delete paths depend on. Postgres in prod, fake in tests."""

    def get_retention_row(self, job_id: str) -> RetentionRow | None: ...
    def mark_media_deleted(
        self, job_id: str, *, raw: bool, derived: bool, at: dt.datetime
    ) -> None: ...
    def mark_cold_deleted(self, job_id: str, at: dt.datetime) -> None: ...
    def set_extended_retention(
        self, job_id: str, until: dt.datetime, *, cold_prefix: str
    ) -> None: ...
    def record_event(self, job_id: str, event: str, detail: dict) -> None: ...
    def find_undeleted_terminal(self, limit: int = 100) -> list[str]: ...
    def find_abandoned_uploads(
        self, older_than: dt.datetime, limit: int = 100
    ) -> list[str]: ...
    def find_stalled_in_flight(
        self, older_than: dt.datetime, limit: int = 100
    ) -> list[str]: ...
    def mark_job_failed(self, job_id: str, error: str) -> None: ...
    def find_expired_extended_retention(
        self, now: dt.datetime, limit: int = 100
    ) -> list[str]: ...


@dataclass
class DeleteReport:
    job_id: str
    outcome: DeleteOutcome
    raw_deleted: bool = False
    derived_objects_deleted: int = 0
    cold_objects_deleted: int = 0


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def delete_media_for_job(
    job_id: str,
    store: RetentionStore,
    storage: storage_mod.Storage,
    *,
    now: dt.datetime | None = None,
) -> DeleteReport:
    """Delete raw media and derived face crops for one job.

    THE HOLD CHECK IS FIRST AND UNCONDITIONAL. Do not add a fast path, a cache,
    or a "we already know this job" shortcut above it -- the whole point of the
    flag is that no delete can happen without reading it.
    """
    now = now or _now()

    row = store.get_retention_row(job_id)
    if row is None:
        log.warning("retention: job %s not found, nothing to delete", job_id)
        return DeleteReport(job_id, DeleteOutcome.NOT_FOUND)

    # --- hold flag check: gate on every delete call ---
    if row.retention_hold:
        store.record_event(
            job_id,
            "retention.delete_skipped",
            {"reason": "retention_hold set", "checked_at": now.isoformat()},
        )
        log.info("retention: job %s held, skipping delete", job_id)
        return DeleteReport(job_id, DeleteOutcome.SKIPPED_HOLD)

    if row.raw_deleted_at is not None and row.derived_deleted_at is not None:
        return DeleteReport(job_id, DeleteOutcome.ALREADY_DELETED)

    raw_deleted = False
    if row.raw_object_key and row.raw_deleted_at is None:
        raw_deleted = storage.delete_object(row.raw_object_key)

    derived_deleted = 0
    if row.derived_prefix and row.derived_deleted_at is None:
        derived_deleted = storage.delete_prefix(row.derived_prefix)

    store.mark_media_deleted(job_id, raw=True, derived=True, at=now)
    store.record_event(
        job_id,
        "retention.deleted",
        {
            "raw_object_key": row.raw_object_key,
            "raw_deleted": raw_deleted,
            "derived_prefix": row.derived_prefix,
            "derived_objects_deleted": derived_deleted,
            "deleted_at": now.isoformat(),
        },
    )
    log.info(
        "retention: job %s deleted raw=%s derived=%d", job_id, raw_deleted, derived_deleted
    )
    return DeleteReport(job_id, DeleteOutcome.DELETED, raw_deleted, derived_deleted)


def open_extended_retention_window(
    job_id: str,
    store: RetentionStore,
    storage: storage_mod.Storage,
    *,
    driving_keys: list[str],
    days: int | None = None,
    now: dt.datetime | None = None,
) -> tuple[dt.datetime, list[str]]:
    """Start the Tier 2 window and preserve the media that drove the score.

    `driving_keys` are the derived objects that actually produced the aggregated
    score -- the face crops (or spectrogram chunks) that survived confidence
    dropping and trimming. They are COPIED into cold storage before the Tier 1
    delete removes derived/, so the window has something a dispute could use.

    The full raw source is deliberately not preserved: Tier 1 deletes it on
    completion for every band, and the driving crops are the evidence, not the
    whole upload.

    MUST be called before delete_media_for_job(), which removes derived/.

    Reminder: this timer expires by itself, including mid-dispute. It is not a
    legal hold and must not be presented as one.
    """
    now = now or _now()
    days = days if days is not None else settings.extended_retention_days
    until = now + dt.timedelta(days=days)
    cold = storage_mod.cold_prefix(job_id)

    preserved: list[str] = []
    for key in driving_keys:
        dst = f"{cold}{key.rsplit('/', 1)[-1]}"
        if storage.copy_object(key, dst):
            preserved.append(dst)

    store.set_extended_retention(job_id, until, cold_prefix=cold)
    store.record_event(
        job_id,
        "retention.extended_window_opened",
        {
            "until": until.isoformat(),
            "days": days,
            "scope": "driving_media",
            "cold_prefix": cold,
            "objects_preserved": len(preserved),
            "driving_keys_requested": len(driving_keys),
            "note": "fixed timer, auto-expires; not a legal hold",
        },
    )
    if len(preserved) != len(driving_keys):
        # The window is meant to hold evidence. If a copy silently failed, the
        # >80 branch has less than it claims to -- say so loudly.
        log.error(
            "retention: job %s preserved %d/%d driving objects",
            job_id, len(preserved), len(driving_keys),
        )
    return until, preserved


def expire_extended_retention(
    job_id: str,
    store: RetentionStore,
    storage: storage_mod.Storage,
    *,
    now: dt.datetime | None = None,
) -> DeleteReport:
    """Delete cold-storage media once the fixed timer has run out.

    This is the THIRD delete path that can touch retained media, so it carries
    the same unconditional hold-flag check as the other two. A job under hold
    keeps its preserved crops past window expiry -- that is the entire point of
    the flag.
    """
    now = now or _now()

    row = store.get_retention_row(job_id)
    if row is None:
        return DeleteReport(job_id, DeleteOutcome.NOT_FOUND)

    # --- hold flag check: gate on every delete call ---
    if row.retention_hold:
        store.record_event(
            job_id,
            "retention.cold_delete_skipped",
            {"reason": "retention_hold set", "checked_at": now.isoformat()},
        )
        log.info("retention: job %s held, keeping preserved media past expiry", job_id)
        return DeleteReport(job_id, DeleteOutcome.SKIPPED_HOLD)

    if row.cold_deleted_at is not None or not row.cold_prefix:
        return DeleteReport(job_id, DeleteOutcome.ALREADY_DELETED)

    if row.extended_retention_until and now < row.extended_retention_until:
        return DeleteReport(job_id, DeleteOutcome.NOT_EXPIRED)

    deleted = storage.delete_prefix(row.cold_prefix)
    store.mark_cold_deleted(job_id, now)
    store.record_event(
        job_id,
        "retention.extended_window_expired",
        {
            "cold_prefix": row.cold_prefix,
            "objects_deleted": deleted,
            "expired_at": now.isoformat(),
        },
    )
    log.info("retention: job %s window expired, deleted %d cold object(s)", job_id, deleted)
    return DeleteReport(job_id, DeleteOutcome.DELETED, cold_objects_deleted=deleted)


def sweep_undeleted(
    store: RetentionStore,
    storage: storage_mod.Storage,
    *,
    limit: int = 100,
) -> list[DeleteReport]:
    """Catch jobs that reached a terminal state but still hold media.

    The completion-triggered delete is the primary path; this is the backstop
    for a worker that died between writing the result and deleting the bytes.
    Without it, a crash silently turns "deleted on completion" into a lie.

    Terminal includes dead-lettered and failed jobs, not just completed ones.
    A job that dead-letters never fires the completion delete (it never
    completed) and never sets completed_at, so it is exactly the case with no
    other delete path -- and a failing pipeline produces these in bulk.
    """
    reports: list[DeleteReport] = []
    for job_id in store.find_undeleted_terminal(limit=limit):
        reports.append(delete_media_for_job(job_id, store, storage))
    return reports


def sweep_abandoned_uploads(
    store: RetentionStore,
    storage: storage_mod.Storage,
    *,
    older_than_hours: int | None = None,
    limit: int = 100,
    now: dt.datetime | None = None,
) -> list[DeleteReport]:
    """Clear media from jobs that were granted an upload and never notified.

    Such a job never enters the pipeline, so it never completes and never
    dead-letters -- it simply sits in awaiting_upload holding whatever the
    client pushed to object storage. Neither the completion delete nor the
    terminal sweep will ever look at it.

    Deletion still goes through delete_media_for_job, so the hold flag gates
    this path exactly like every other one.
    """
    now = now or _now()
    hours = older_than_hours if older_than_hours is not None else settings.abandoned_upload_hours
    cutoff = now - dt.timedelta(hours=hours)

    reports: list[DeleteReport] = []
    for job_id in store.find_abandoned_uploads(cutoff, limit=limit):
        report = delete_media_for_job(job_id, store, storage, now=now)
        if report.outcome is not DeleteOutcome.SKIPPED_HOLD:
            # Terminal, so the row stops sitting in awaiting_upload forever.
            # Clearing the bytes without moving the status leaves a table that
            # only grows -- a smaller problem than retained media, but still a
            # dead row nothing can distinguish from a live one.
            store.mark_job_failed(job_id, "upload grant abandoned; client never notified")
        reports.append(report)
    return reports


def sweep_stalled_jobs(
    store: RetentionStore,
    storage: storage_mod.Storage,
    *,
    older_than_hours: int | None = None,
    limit: int = 100,
    now: dt.datetime | None = None,
) -> list[DeleteReport]:
    """Fail jobs stuck mid-pipeline, then clear their media.

    The in-flight states (queued, preprocessing, inference, aggregating) are
    transient only while their queue message exists. Both the gateway and the
    workers commit the status change before pushing to Redis, so a crash in
    between -- or a push lost inside Redis's AOF everysec window -- strands the
    row with no message to advance it. It will never complete and never
    dead-letter, so no other sweep covers it while it holds a raw upload.

    The threshold is a proxy for "no message exists", because the job row
    cannot tell us that directly. Set it far above real processing time: the
    cost of being wrong is failing a job that was merely slow, and the client
    can resubmit, but the cost of never sweeping is media retained forever.

    Marking failed first is what makes the media reachable: the job then falls
    under the ordinary terminal delete path, hold flag and all.
    """
    now = now or _now()
    hours = older_than_hours if older_than_hours is not None else settings.stalled_job_hours
    cutoff = now - dt.timedelta(hours=hours)

    reports: list[DeleteReport] = []
    for job_id in store.find_stalled_in_flight(cutoff, limit=limit):
        store.mark_job_failed(
            job_id, f"stalled in-flight with no queue message for over {hours}h"
        )
        store.record_event(
            job_id,
            "job.stalled",
            {"detected_at": now.isoformat(), "threshold_hours": hours},
        )
        reports.append(delete_media_for_job(job_id, store, storage, now=now))
    return reports


def sweep_expired_windows(
    store: RetentionStore,
    storage: storage_mod.Storage,
    *,
    limit: int = 100,
    now: dt.datetime | None = None,
) -> list[DeleteReport]:
    """Delete preserved media whose extended retention window has run out.

    The window is a fixed timer, so something has to actually enforce expiry --
    otherwise "30 days" quietly becomes "forever", which is its own liability.
    """
    now = now or _now()
    return [
        expire_extended_retention(job_id, store, storage, now=now)
        for job_id in store.find_expired_extended_retention(now, limit=limit)
    ]
