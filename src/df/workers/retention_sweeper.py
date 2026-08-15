"""Retention sweeper. Three passes, all hold-flag gated.

1. Backstop for the completion-triggered delete. Without it, a router worker
   that dies between committing a verdict and deleting the bytes leaves raw
   video and face crops in the bucket indefinitely -- and "deleted on inference
   completion" quietly stops being true. Covers every terminal state, including
   dead-lettered and failed jobs: those never fire the completion delete at all,
   so this is their only path to deletion.

2. Abandoned uploads -- a client granted an upload URL that never notified.
   Those jobs never enter the pipeline, so no terminal state is ever reached and
   pass 1 would never see them, while the media sits in the bucket.

3. Enforcement of the extended retention window's fixed timer. A 30-day timer
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
