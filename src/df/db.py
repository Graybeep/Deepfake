"""Postgres access. The job row is the audit trail, so every write that changes
a verdict also writes hash / model_version_id / aggregation method + params.
"""
from __future__ import annotations

import datetime as dt
import json
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from df.config import settings
from df.retention import RetentionRow


class Db:
    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.pg_dsn

    @contextmanager
    def conn(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.dsn, row_factory=dict_row) as c:
            yield c

    # --- job lifecycle -----------------------------------------------------

    def create_job(
        self, *, media_type: str, raw_object_key: str, derived_prefix: str, submitted_by: str
    ) -> str:
        with self.conn() as c, c.transaction():
            row = c.execute(
                """
                INSERT INTO jobs (media_type, status, raw_object_key, derived_prefix, submitted_by)
                VALUES (%s, 'awaiting_upload', %s, %s, %s)
                RETURNING id
                """,
                (media_type, raw_object_key, derived_prefix, submitted_by),
            ).fetchone()
            return str(row["id"])

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.conn() as c:
            return c.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()

    def set_status(self, job_id: str, status: str, *, error: str | None = None) -> None:
        with self.conn() as c, c.transaction():
            c.execute(
                """
                UPDATE jobs
                   SET status = %s,
                       error = COALESCE(%s, error),
                       updated_at = now()
                 WHERE id = %s
                """,
                (status, error, job_id),
            )

    def set_content_hash(self, job_id: str, content_hash: str) -> None:
        with self.conn() as c, c.transaction():
            c.execute(
                "UPDATE jobs SET content_hash = %s, updated_at = now() WHERE id = %s",
                (content_hash, job_id),
            )

    def bump_attempts(self, job_id: str) -> int:
        with self.conn() as c, c.transaction():
            row = c.execute(
                "UPDATE jobs SET attempts = attempts + 1, updated_at = now() "
                "WHERE id = %s RETURNING attempts",
                (job_id,),
            ).fetchone()
            return int(row["attempts"]) if row else 0

    def insert_items(self, job_id: str, items: list[dict[str, Any]]) -> None:
        """Write per-item scores. Idempotent by natural key.

        Streams reclaim can hand a slow consumer's message to a second one, so
        the same item can legitimately be scored twice. DO NOTHING keeps the
        first write rather than counting the item twice in aggregation --
        re-scoring the same bytes with the same model_version_id is
        deterministic, so the discarded row would have been identical anyway.
        The unique index (migration 002) is what makes this safe; without it
        the clause is a no-op that silently permits the duplicate.
        """
        if not items:
            return
        # executemany is a cursor method in psycopg3; Connection only has
        # execute(). Every other method here happens to use execute(), which is
        # why this was the one path that broke against a real database.
        with self.conn() as c, c.transaction(), c.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO job_items
                    (job_id, item_index, item_kind, face_index, score, confidence,
                     object_key, model_version_id, model_validation, face_w, face_h,
                     calibration)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id, item_index, face_index) DO NOTHING
                """,
                [
                    (
                        job_id,
                        i["item_index"],
                        i["item_kind"],
                        i.get("face_index"),
                        i["score"],
                        i["confidence"],
                        i.get("object_key"),
                        i.get("model_version_id"),
                        i.get("model_validation"),
                        i.get("face_w"),
                        i.get("face_h"),
                        i.get("calibration"),
                    )
                    for i in items
                ],
            )

    def item_model_versions(self, job_id: str) -> list[str]:
        """Which models actually produced this job's surviving rows.

        The router attributes the job's score to this rather than to the queue
        message, so the recorded model is the one whose numbers were used.
        More than one value means the job was scored by two different models --
        reachable during a rolling deploy when duplicate delivery lets each
        consumer win some items.
        """
        with self.conn() as c:
            rows = c.execute(
                "SELECT DISTINCT model_version_id FROM job_items "
                "WHERE job_id = %s AND model_version_id IS NOT NULL "
                "ORDER BY model_version_id",
                (job_id,),
            ).fetchall()
        return [r["model_version_id"] for r in rows]

    def item_model_validations(self, job_id: str) -> list[str]:
        """Validation levels recorded on this job's surviving rows.

        Derived from the rows for the same reason model_version_id is: the
        queue message is hearsay, the rows are what produced the score.
        """
        with self.conn() as c:
            rows = c.execute(
                "SELECT DISTINCT model_validation FROM job_items "
                "WHERE job_id = %s AND model_validation IS NOT NULL "
                "ORDER BY model_validation",
                (job_id,),
            ).fetchall()
        return [r["model_validation"] for r in rows]

    def item_calibrations(self, job_id: str) -> list[str]:
        """Calibrations recorded on this job's surviving rows.

        Derived from the rows for the same reason model_version_id is. This one
        matters even when the model is unchanged: model_version_id is keyed on
        the weights hash, so refitting the temperature moves the score without
        moving the id. Two rows scored under different temperatures are not
        comparable, and this is the only column that shows it.
        """
        with self.conn() as c:
            rows = c.execute(
                "SELECT DISTINCT calibration FROM job_items "
                "WHERE job_id = %s AND calibration IS NOT NULL "
                "ORDER BY calibration",
                (job_id,),
            ).fetchall()
        return [r["calibration"] for r in rows]

    def get_discarded_detections(self, job_id: str) -> dict[str, Any]:
        """What the detection-confidence floor rejected, from the audit trail.

        Read from job_events rather than job_items because these were never
        scored: job_items.score is NOT NULL, so a row would have to invent a
        score for a crop the model never saw. The event is the permanent record.

        Returns {} when nothing was recorded -- a pre-gate job, which is not the
        same claim as "nothing was discarded".
        """
        with self.conn() as c:
            row = c.execute(
                "SELECT detail FROM job_events "
                "WHERE job_id = %s AND event = 'preprocess.complete' "
                "ORDER BY created_at DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        return (row or {}).get("detail") or {}

    def get_items(self, job_id: str) -> list[dict[str, Any]]:
        with self.conn() as c:
            return c.execute(
                """
                SELECT item_index, item_kind, face_index, score, confidence,
                       object_key, model_version_id, model_validation,
                       face_w, face_h, calibration
                  FROM job_items WHERE job_id = %s ORDER BY item_index, face_index
                """,
                (job_id,),
            ).fetchall()

    def write_result(
        self,
        job_id: str,
        *,
        result_class: str,
        band: str,
        aggregate_score: float | None,
        model_version_id: str,
        aggregation_method: str,
        aggregation_params: dict,
        item_count: int,
        items_total: int,
        face_count: int | None,
        calibration: str | None,
        items_unattributed: int,
        model_validation: str | None,
    ) -> None:
        """Single write that makes a job's verdict reproducible.

        Score, model version, and aggregation params land together -- a row with
        a score but no model_version_id is an unauditable result.

        items_unattributed lands here too, on the audit row rather than only in
        review_flags. The flag is operational and gets attention now; this row
        is what a dispute reads months later, and it must not assert full
        attribution when only part of the evidence carried a producer.
        """
        with self.conn() as c, c.transaction():
            c.execute(
                """
                UPDATE jobs
                   SET result_class = %s,
                       band = %s,
                       aggregate_score = %s,
                       model_version_id = %s,
                       aggregation_method = %s,
                       aggregation_params = %s,
                       item_count = %s,
                       items_total = %s,
                       face_count = %s,
                       items_unattributed = %s,
                       model_validation = %s,
                       calibration = %s,
                       status = 'complete',
                       completed_at = now(),
                       updated_at = now()
                 WHERE id = %s
                """,
                (
                    result_class,
                    band,
                    aggregate_score,
                    model_version_id,
                    aggregation_method,
                    json.dumps(aggregation_params),
                    item_count,
                    items_total,
                    face_count,
                    items_unattributed,
                    model_validation,
                    calibration,
                    job_id,
                ),
            )

    def flag_for_review(self, job_id: str, reason: str, urgency: str = "normal") -> None:
        """Tier 3 substitute for the human-in-the-loop dashboard."""
        with self.conn() as c, c.transaction():
            c.execute(
                "INSERT INTO review_flags (job_id, reason, urgency) VALUES (%s, %s, %s)",
                (job_id, reason, urgency),
            )

    # --- RetentionStore ----------------------------------------------------

    def get_retention_row(self, job_id: str) -> RetentionRow | None:
        with self.conn() as c:
            r = c.execute(
                """
                SELECT id, retention_hold, raw_object_key, derived_prefix,
                       raw_deleted_at, derived_deleted_at,
                       cold_prefix, cold_deleted_at, extended_retention_until
                  FROM jobs WHERE id = %s
                """,
                (job_id,),
            ).fetchone()
        if r is None:
            return None
        return RetentionRow(
            job_id=str(r["id"]),
            retention_hold=r["retention_hold"],
            raw_object_key=r["raw_object_key"],
            derived_prefix=r["derived_prefix"],
            raw_deleted_at=r["raw_deleted_at"],
            derived_deleted_at=r["derived_deleted_at"],
            cold_prefix=r["cold_prefix"],
            cold_deleted_at=r["cold_deleted_at"],
            extended_retention_until=r["extended_retention_until"],
        )

    def mark_media_deleted(
        self, job_id: str, *, raw: bool, derived: bool, at: dt.datetime
    ) -> None:
        with self.conn() as c, c.transaction():
            c.execute(
                """
                UPDATE jobs
                   SET raw_deleted_at     = CASE WHEN %s THEN COALESCE(raw_deleted_at, %s)
                                                 ELSE raw_deleted_at END,
                       derived_deleted_at = CASE WHEN %s THEN COALESCE(derived_deleted_at, %s)
                                                 ELSE derived_deleted_at END,
                       updated_at = now()
                 WHERE id = %s
                """,
                (raw, at, derived, at, job_id),
            )

    def mark_cold_deleted(self, job_id: str, at: dt.datetime) -> None:
        with self.conn() as c, c.transaction():
            c.execute(
                "UPDATE jobs SET cold_deleted_at = COALESCE(cold_deleted_at, %s), "
                "updated_at = now() WHERE id = %s",
                (at, job_id),
            )

    def set_extended_retention(
        self, job_id: str, until: dt.datetime, *, cold_prefix: str
    ) -> None:
        with self.conn() as c, c.transaction():
            c.execute(
                """
                UPDATE jobs
                   SET extended_retention_until = %s, cold_prefix = %s, updated_at = now()
                 WHERE id = %s
                """,
                (until, cold_prefix, job_id),
            )

    def set_retention_hold(self, job_id: str, *, reason: str) -> None:
        """Set the hold flag. Nothing in the MVP calls this automatically --
        the flag exists and is checked before any code path can set it."""
        with self.conn() as c, c.transaction():
            c.execute(
                """
                UPDATE jobs
                   SET retention_hold = TRUE, hold_set_at = now(),
                       hold_reason = %s, updated_at = now()
                 WHERE id = %s
                """,
                (reason, job_id),
            )

    def record_event(self, job_id: str, event: str, detail: dict) -> None:
        with self.conn() as c, c.transaction():
            c.execute(
                "INSERT INTO job_events (job_id, event, detail) VALUES (%s, %s, %s)",
                (job_id, event, json.dumps(detail)),
            )

    def find_undeleted_terminal(self, limit: int = 100) -> list[str]:
        """Jobs that reached a terminal state but still hold media.

        Terminal means completed OR dead-lettered OR failed -- not merely
        completed. A dead-lettered job never gets completed_at set (see
        workers/loop.py), so keying this on completed_at alone left failed
        jobs' raw uploads and face crops in the bucket with no expiry at all.
        The completion-triggered delete cannot cover them either, because they
        never completed. Failures also cluster: one bad deploy dead-letters
        every job that arrives during it, and each one leaks its media.
        """
        with self.conn() as c:
            rows = c.execute(
                """
                SELECT id FROM jobs
                 WHERE (completed_at IS NOT NULL
                        OR status IN ('dead_letter', 'failed'))
                   AND retention_hold = FALSE
                   AND (raw_deleted_at IS NULL OR derived_deleted_at IS NULL)
                 ORDER BY COALESCE(completed_at, updated_at)
                 LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [str(r["id"]) for r in rows]

    def find_stalled_in_flight(self, older_than: dt.datetime, limit: int = 100) -> list[str]:
        """Jobs stuck mid-pipeline with nothing left to move them along.

        The in-flight states are transient only while the queue message that
        drives them exists. It can stop existing: the gateway and the workers
        both commit the status change to Postgres BEFORE pushing to Redis, so a
        crash between the two -- or Redis losing the push inside its AOF
        everysec window -- leaves a row claiming 'queued' with no message
        behind it. Neither queue backend can recover that: Streams reclaim and
        the list backend's requeue both recover a message that exists but was
        abandoned, not one that was never written.

        Such a job never completes and never dead-letters, so no other sweep
        will ever look at it while it holds a raw upload.

        Held jobs are deliberately NOT excluded here. This marks a job failed;
        it does not delete anything. The hold flag protects media, and the
        delete path that follows applies it as usual.
        """
        with self.conn() as c:
            rows = c.execute(
                """
                SELECT id FROM jobs
                 WHERE status IN ('queued', 'preprocessing', 'inference', 'aggregating')
                   AND updated_at < %s
                 ORDER BY updated_at
                 LIMIT %s
                """,
                (older_than, limit),
            ).fetchall()
        return [str(r["id"]) for r in rows]

    def mark_job_failed(self, job_id: str, error: str) -> None:
        """Move a job to a terminal state so it stops being invisible.

        Without this, a swept job keeps its non-terminal status forever: the
        media is gone but the row accumulates, and nothing downstream can tell
        a dead job from a live one.
        """
        with self.conn() as c, c.transaction():
            c.execute(
                "UPDATE jobs SET status = 'failed', error = %s, updated_at = now() "
                "WHERE id = %s",
                (error, job_id),
            )

    def find_abandoned_uploads(self, older_than: dt.datetime, limit: int = 100) -> list[str]:
        """Jobs that got an upload grant and never came back.

        A client can PUT/POST to its presigned URL and simply never call the
        notify endpoint. The job then sits in awaiting_upload forever, holding
        real media that no terminal-state sweep will ever look at. The presign
        TTL is minutes, so anything older than the threshold can no longer be
        uploaded to and is safe to clear.
        """
        with self.conn() as c:
            rows = c.execute(
                """
                SELECT id FROM jobs
                 WHERE status = 'awaiting_upload'
                   AND retention_hold = FALSE
                   AND created_at < %s
                   AND (raw_deleted_at IS NULL OR derived_deleted_at IS NULL)
                 ORDER BY created_at
                 LIMIT %s
                """,
                (older_than, limit),
            ).fetchall()
        return [str(r["id"]) for r in rows]

    def find_expired_extended_retention(
        self, now: dt.datetime, limit: int = 100
    ) -> list[str]:
        """Jobs whose fixed window has run out and whose cold media is still there.

        Held jobs are excluded here as well as inside expire_extended_retention()
        -- belt and braces, because this is the query that decides what gets
        offered up for deletion in the first place.
        """
        with self.conn() as c:
            rows = c.execute(
                """
                SELECT id FROM jobs
                 WHERE cold_prefix IS NOT NULL
                   AND cold_deleted_at IS NULL
                   AND retention_hold = FALSE
                   AND extended_retention_until IS NOT NULL
                   AND extended_retention_until <= %s
                 ORDER BY extended_retention_until
                 LIMIT %s
                """,
                (now, limit),
            ).fetchall()
        return [str(r["id"]) for r in rows]
