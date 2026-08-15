"""Shared worker loop: pop, handle, ack; on exception retry then dead-letter."""
from __future__ import annotations

import logging
import signal
import sys
from typing import Callable

from df.db import Db
from df.jobstatus import JobStatus
from df.queue import Message, RedisQueue

log = logging.getLogger("df.worker")

_running = True


def _stop(*_args) -> None:
    global _running
    log.info("shutdown requested, finishing current message")
    _running = False


def run_worker(
    topic: str,
    handler: Callable[[Message], None],
    *,
    queue: RedisQueue | None = None,
    db: Db | None = None,
    status: JobStatus | None = None,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    queue = queue or RedisQueue()
    db = db or Db()
    status = status or JobStatus()

    from df.jobstatus import assert_persistence_enabled

    assert_persistence_enabled(status.r)

    # A worker killed mid-message leaves it stranded in the processing list.
    moved = queue.requeue_stale_processing(topic)
    if moved:
        log.warning("requeued %d stale message(s) on %s", moved, topic)

    log.info("worker up on topic=%s", topic)
    while _running:
        msg = queue.pop(topic, timeout=5)
        if msg is None:
            continue

        job_id = msg.payload.get("job_id")
        try:
            handler(msg)
            queue.ack(msg)
        except Exception as exc:  # noqa: BLE001 - worker must not die on one bad job
            log.exception("handler failed topic=%s job=%s", topic, job_id)
            retried = queue.fail(msg, f"{type(exc).__name__}: {exc}")
            if not retried and job_id:
                db.set_status(job_id, "dead_letter", error=f"{type(exc).__name__}: {exc}")
                status.publish(job_id, "dead_letter", error=str(exc))
                db.record_event(job_id, "job.dead_lettered", {"error": str(exc), "topic": topic})

    log.info("worker stopped on topic=%s", topic)
    sys.exit(0)
