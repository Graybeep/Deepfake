from __future__ import annotations

import datetime as dt
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from df.retention import RetentionRow  # noqa: E402
from df.storage import InMemoryStorage  # noqa: E402


class FakeRetentionStore:
    """In-memory RetentionStore. Records every event so tests can assert on the
    audit trail as well as on the bytes."""

    def __init__(self) -> None:
        self.rows: dict[str, RetentionRow] = {}
        self.events: list[tuple[str, str, dict]] = []
        self.extended: dict[str, dt.datetime] = {}
        # Jobs that reached a terminal state (complete, dead_letter, failed)
        # and still hold media.
        self.terminal_undeleted: list[str] = []
        # job_id -> created_at, for jobs still sitting in awaiting_upload.
        self.awaiting_upload: dict[str, dt.datetime] = {}
        self.cold_deleted: dict[str, dt.datetime] = {}

    def add(
        self,
        job_id: str,
        *,
        retention_hold: bool = False,
        raw_object_key: str | None = None,
        derived_prefix: str | None = None,
    ) -> None:
        self.rows[job_id] = RetentionRow(
            job_id=job_id,
            retention_hold=retention_hold,
            raw_object_key=raw_object_key,
            derived_prefix=derived_prefix,
            raw_deleted_at=None,
            derived_deleted_at=None,
        )

    # --- RetentionStore protocol ---
    def get_retention_row(self, job_id: str) -> RetentionRow | None:
        return self.rows.get(job_id)

    def mark_media_deleted(self, job_id: str, *, raw: bool, derived: bool, at) -> None:
        row = self.rows[job_id]
        if raw and row.raw_deleted_at is None:
            row.raw_deleted_at = at
        if derived and row.derived_deleted_at is None:
            row.derived_deleted_at = at

    def mark_cold_deleted(self, job_id: str, at) -> None:
        self.cold_deleted[job_id] = at
        self.rows[job_id].cold_deleted_at = at

    def set_extended_retention(self, job_id: str, until, *, cold_prefix: str) -> None:
        self.extended[job_id] = until
        row = self.rows[job_id]
        row.extended_retention_until = until
        row.cold_prefix = cold_prefix

    def find_expired_extended_retention(self, now, limit: int = 100) -> list[str]:
        return [
            jid for jid, row in self.rows.items()
            if row.cold_prefix
            and row.cold_deleted_at is None
            and row.extended_retention_until is not None
            and row.extended_retention_until <= now
        ][:limit]

    def record_event(self, job_id: str, event: str, detail: dict) -> None:
        self.events.append((job_id, event, detail))

    def find_undeleted_terminal(self, limit: int = 100) -> list[str]:
        return self.terminal_undeleted[:limit]

    def find_abandoned_uploads(self, older_than, limit: int = 100) -> list[str]:
        return [
            jid for jid, created in self.awaiting_upload.items()
            if created < older_than
            and not self.rows[jid].retention_hold
            and self.rows[jid].raw_deleted_at is None
        ][:limit]

    # --- helpers ---
    def event_names(self, job_id: str) -> list[str]:
        return [e for j, e, _ in self.events if j == job_id]


@pytest.fixture
def store() -> FakeRetentionStore:
    return FakeRetentionStore()


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
def seeded_job(store: FakeRetentionStore, storage: InMemoryStorage):
    """A completed job with raw media and three face crops on disk."""

    def _seed(job_id: str = "job-1", *, hold: bool = False):
        raw = f"raw/{job_id}/original"
        derived = f"derived/{job_id}/"
        storage.put_bytes(raw, b"raw-video-bytes")
        for i in range(3):
            storage.put_bytes(f"{derived}items/f{i:05d}_x0.png", f"face-crop-{i}".encode())
        store.add(job_id, retention_hold=hold, raw_object_key=raw, derived_prefix=derived)
        return job_id, raw, derived

    return _seed
