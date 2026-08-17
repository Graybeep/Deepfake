"""In-memory stand-ins for Postgres and Redis.

These exist so the worker handlers can be driven end to end without
infrastructure. They implement the same surface the real Db / JobStatus expose
to the handlers, and they store real state -- a handler that forgets to write a
result or delete a key fails the test rather than passing against a mock that
records the call and shrugs.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from df.retention import RetentionRow


class FakeDb:
    """Stands in for df.db.Db across all four workers."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.items: dict[str, list[dict[str, Any]]] = {}
        self.events: list[tuple[str, str, dict]] = []
        self.review_flags: list[tuple[str, str]] = []

    # --- seeding ---
    def add_job(
        self,
        job_id: str,
        media_type: str,
        *,
        retention_hold: bool = False,
        raw_object_key: str | None = None,
        derived_prefix: str | None = None,
        status: str = "queued",
        created_at: dt.datetime | None = None,
        updated_at: dt.datetime | None = None,
    ) -> None:
        self.jobs[job_id] = {
            "id": job_id,
            "media_type": media_type,
            "status": status,
            # Both NOT NULL DEFAULT now() in the real schema. The abandoned
            # sweep keys off created_at and the stalled sweep off updated_at,
            # so the fake has to carry both rather than a simplification.
            "created_at": created_at or dt.datetime.now(dt.timezone.utc),
            "updated_at": updated_at or created_at or dt.datetime.now(dt.timezone.utc),
            "attempts": 0,
            "content_hash": None,
            "retention_hold": retention_hold,
            "raw_object_key": raw_object_key or f"raw/{job_id}/original",
            "derived_prefix": derived_prefix or f"derived/{job_id}/",
            "raw_deleted_at": None,
            "derived_deleted_at": None,
            "extended_retention_until": None,
            "cold_prefix": None,
            "cold_deleted_at": None,
            "result_class": None,
            "band": None,
            "aggregate_score": None,
            "model_version_id": None,
            "aggregation_method": None,
            "aggregation_params": None,
            "item_count": None,
            "face_count": None,
            "completed_at": None,
        }
        self.items[job_id] = []

    # --- job lifecycle ---
    def set_status(self, job_id: str, status: str, *, error: str | None = None) -> None:
        self.jobs[job_id]["status"] = status
        # The real UPDATE sets updated_at = now(); the stalled sweep reads it.
        self.jobs[job_id]["updated_at"] = dt.datetime.now(dt.timezone.utc)
        if error:
            self.jobs[job_id]["error"] = error

    def set_content_hash(self, job_id: str, content_hash: str) -> None:
        self.jobs[job_id]["content_hash"] = content_hash

    def bump_attempts(self, job_id: str) -> int:
        self.jobs[job_id]["attempts"] += 1
        return self.jobs[job_id]["attempts"]

    def insert_items(self, job_id: str, items: list[dict[str, Any]]) -> None:
        """Mirrors the unique index from migration 002 (ON CONFLICT DO NOTHING).

        Without this the fake happily accepts duplicates the real database
        rejects, and a test asserting that a redelivered message does not skew
        aggregation would pass for the wrong reason.
        """
        existing = self.items.setdefault(job_id, [])
        seen = {(r["item_index"], r.get("face_index")) for r in existing}
        for row in items:
            key = (row["item_index"], row.get("face_index"))
            if key in seen:
                continue
            seen.add(key)
            existing.append(row)

    def item_model_validations(self, job_id: str) -> list[str]:
        """Mirrors the real DISTINCT query, NULLs dropped."""
        return sorted({
            r["model_validation"] for r in self.items.get(job_id, [])
            if r.get("model_validation") is not None
        })

    def item_model_versions(self, job_id: str) -> list[str]:
        """Mirrors the real DISTINCT query, including dropping NULLs.

        Rows written before migration 003 have no recorded producer; counting
        them as a distinct version would make every pre-migration job look
        mixed.
        """
        return sorted({
            r["model_version_id"] for r in self.items.get(job_id, [])
            if r.get("model_version_id") is not None
        })

    def get_items(self, job_id: str) -> list[dict[str, Any]]:
        return sorted(
            self.items.get(job_id, []),
            key=lambda r: (r["item_index"], r["face_index"] if r["face_index"] is not None else -1),
        )

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
        face_count: int | None,
        items_unattributed: int,
        model_validation: str | None,
    ) -> None:
        self.jobs[job_id].update(
            result_class=result_class,
            band=band,
            aggregate_score=aggregate_score,
            model_version_id=model_version_id,
            aggregation_method=aggregation_method,
            aggregation_params=aggregation_params,
            item_count=item_count,
            face_count=face_count,
            items_unattributed=items_unattributed,
            model_validation=model_validation,
            status="complete",
            completed_at=dt.datetime.now(dt.timezone.utc),
        )

    def flag_for_review(self, job_id: str, reason: str, urgency: str = "normal") -> None:
        self.review_flags.append((job_id, reason, urgency))

    # --- RetentionStore ---
    def get_retention_row(self, job_id: str) -> RetentionRow | None:
        j = self.jobs.get(job_id)
        if j is None:
            return None
        return RetentionRow(
            job_id=job_id,
            retention_hold=j["retention_hold"],
            raw_object_key=j["raw_object_key"],
            derived_prefix=j["derived_prefix"],
            raw_deleted_at=j["raw_deleted_at"],
            derived_deleted_at=j["derived_deleted_at"],
            cold_prefix=j["cold_prefix"],
            cold_deleted_at=j["cold_deleted_at"],
            extended_retention_until=j["extended_retention_until"],
        )

    def mark_media_deleted(self, job_id: str, *, raw: bool, derived: bool, at) -> None:
        j = self.jobs[job_id]
        if raw and j["raw_deleted_at"] is None:
            j["raw_deleted_at"] = at
        if derived and j["derived_deleted_at"] is None:
            j["derived_deleted_at"] = at

    def mark_cold_deleted(self, job_id: str, at) -> None:
        j = self.jobs[job_id]
        if j["cold_deleted_at"] is None:
            j["cold_deleted_at"] = at

    def set_extended_retention(self, job_id: str, until, *, cold_prefix: str) -> None:
        self.jobs[job_id]["extended_retention_until"] = until
        self.jobs[job_id]["cold_prefix"] = cold_prefix

    def find_expired_extended_retention(self, now, limit: int = 100) -> list[str]:
        return [
            jid for jid, j in self.jobs.items()
            if j["cold_prefix"]
            and j["cold_deleted_at"] is None
            and j["extended_retention_until"] is not None
            and j["extended_retention_until"] <= now
        ][:limit]

    def record_event(self, job_id: str, event: str, detail: dict) -> None:
        self.events.append((job_id, event, detail))

    def find_undeleted_terminal(self, limit: int = 100) -> list[str]:
        """Mirrors the real predicate: terminal means completed OR failed.

        A dead-lettered job never gets completed_at, so keying on that alone
        left its media with no delete path at all.
        """
        return [
            jid for jid, j in self.jobs.items()
            if (j["completed_at"] is not None
                or j.get("status") in {"dead_letter", "failed"})
            and not j["retention_hold"]
            and (j["raw_deleted_at"] is None or j["derived_deleted_at"] is None)
        ][:limit]

    def find_stalled_in_flight(self, older_than, limit: int = 100) -> list[str]:
        """Held jobs are NOT excluded -- this marks failed, it does not delete."""
        return [
            jid for jid, j in self.jobs.items()
            if j.get("status") in {"queued", "preprocessing", "inference", "aggregating"}
            and j.get("updated_at") is not None
            and j["updated_at"] < older_than
        ][:limit]

    def mark_job_failed(self, job_id: str, error: str) -> None:
        self.jobs[job_id]["status"] = "failed"
        self.jobs[job_id]["error"] = error

    def find_abandoned_uploads(self, older_than, limit: int = 100) -> list[str]:
        return [
            jid for jid, j in self.jobs.items()
            if j.get("status") == "awaiting_upload"
            and not j["retention_hold"]
            and j.get("created_at") is not None
            and j["created_at"] < older_than
            and (j["raw_deleted_at"] is None or j["derived_deleted_at"] is None)
        ][:limit]

    # --- helpers ---
    def event_names(self, job_id: str) -> list[str]:
        return [e for j, e, _ in self.events if j == job_id]


class FakeJobStatus:
    """Stands in for df.jobstatus.JobStatus. Keeps the full push history so
    tests can assert clients saw every transition, not just the final state."""

    def __init__(self) -> None:
        self.published: list[dict] = []

    def publish(self, job_id: str, status: str, **extra: Any) -> dict:
        doc = {"job_id": job_id, "status": status, **extra}
        self.published.append(doc)
        return doc

    def read(self, job_id: str) -> dict | None:
        for doc in reversed(self.published):
            if doc["job_id"] == job_id:
                return doc
        return None

    def statuses(self, job_id: str) -> list[str]:
        return [d["status"] for d in self.published if d["job_id"] == job_id]
