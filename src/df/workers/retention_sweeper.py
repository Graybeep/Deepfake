"""Retention sweeper. Four passes, all hold-flag gated.

Between them these cover every value the status CHECK constraint allows, which
is the property worth preserving: two leaks were found by noticing media in the
bucket, and both were a status the sweep query did not know to look for. Adding
a status without giving it a path here recreates that bug.

Passes run in this order deliberately; 1 hands work to 2.

1. Stalled in-flight jobs -- queued/preprocessing/inference/aggregating rows
   whose queue message no longer exists. Both the gateway and the workers commit
   the status change before pushing to Redis, so a crash in between strands the
   row with nothing to advance it. It never completes and never dead-letters, so
   pass 2 cannot see it while it is in-flight. This pass ONLY marks such jobs
   failed -- it does not delete. Its finder does not filter on the hold flag (a
   held job that stalled should still stop looking in-flight rather than sitting
   invisible to every sweep forever), so deleting here would leave
   delete_media_for_job's gate as the single protection for held media.
   Marking, then letting pass 2 delete, gives two independent layers.

2. Backstop for the completion-triggered delete, and the owner of deletion for
   every terminal state. Without it, a router worker that dies between
   committing a verdict and deleting the bytes leaves raw video and face crops
   in the bucket indefinitely -- and "deleted on inference completion" quietly
   stops being true. Covers dead-lettered and failed jobs, including the ones
   pass 1 just marked: those never fire the completion delete at all, so this is
   their only path to deletion. Its finder excludes held jobs.

3. Abandoned uploads -- a client granted an upload URL that never notified.
   Those jobs never enter the pipeline, so no terminal state is ever reached and
   pass 1 would never see them, while the media sits in the bucket. Also marked
   terminal, so the row does not sit in awaiting_upload forever.

4. Enforcement of the extended retention window's fixed timer. A 30-day timer
   that nothing enforces silently becomes "retained forever", which is its own
   liability.
"""
from __future__ import annotations

import logging
import time

from df import storage as storage_mod
from df.db import Db
from df.retention import (
    DeleteOutcome,
    sweep_abandoned_uploads,
    sweep_expired_windows,
    sweep_stalled_jobs,
    sweep_undeleted,
)

log = logging.getLogger("df.retention.sweeper")

INTERVAL_SECONDS = 300


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    db, storage = Db(), storage_mod.build_storage()
    log.info("retention sweeper up, interval=%ds", INTERVAL_SECONDS)

    while True:
        # Stalled FIRST. It only marks jobs failed; sweep_undeleted below owns
        # the deletion, and its finder excludes held jobs, which is the second
        # layer the stalled path would otherwise lack. Running it first means
        # the handoff completes in this pass rather than the next one.
        try:
            stalled = sweep_stalled_jobs(db, limit=200)
            if stalled:
                log.warning(
                    "sweeper marked %d stalled job(s) failed -- stuck in an in-flight "
                    "state with no queue message; media is cleared by the terminal "
                    "sweep below, which excludes held jobs",
                    len(stalled),
                )
        except Exception:  # noqa: BLE001
            log.exception("stalled job sweep failed, retrying next interval")

        try:
            reports = sweep_undeleted(db, storage, limit=200)
            deleted = [r for r in reports if r.outcome is DeleteOutcome.DELETED]
            held = [r for r in reports if r.outcome is DeleteOutcome.SKIPPED_HOLD]
            if reports:
                log.warning(
                    "sweeper cleaned %d job(s) missed by the completion path "
                    "(%d held, %d total examined)",
                    len(deleted), len(held), len(reports),
                )
        except Exception:  # noqa: BLE001 - sweeper must survive a bad round
            log.exception("undeleted sweep failed, retrying next interval")

        try:
            abandoned = sweep_abandoned_uploads(db, storage, limit=200)
            cleared = [r for r in abandoned if r.outcome is DeleteOutcome.DELETED]
            if abandoned:
                log.warning(
                    "sweeper cleared media from %d abandoned upload(s) "
                    "(%d examined) -- granted an upload, never notified",
                    len(cleared), len(abandoned),
                )
        except Exception:  # noqa: BLE001
            log.exception("abandoned upload sweep failed, retrying next interval")

        try:
            expired = sweep_expired_windows(db, storage, limit=200)
            gone = [r for r in expired if r.outcome is DeleteOutcome.DELETED]
            kept = [r for r in expired if r.outcome is DeleteOutcome.SKIPPED_HOLD]
            if expired:
                log.info(
                    "extended retention: %d window(s) expired and cleared, %d held",
                    len(gone), len(kept),
                )
        except Exception:  # noqa: BLE001
            log.exception("window expiry sweep failed, retrying next interval")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
