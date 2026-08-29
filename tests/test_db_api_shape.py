"""Execute the real `Db` bodies against a psycopg-shaped connection.

Closes the first of the two gaps CLAUDE.md lists under "Testing status": every
test runs against `FakeDb`, which does not mirror psycopg3's Connection/Cursor
split, so `insert_items` shipped calling `executemany` on a Connection and
dead-lettered every job against a real database while the suite stayed green.

Scope, stated plainly: this proves the psycopg API *shape* every `Db` method
touches is real. It executes no SQL. The statements themselves are only proven
by the live probes -- `scripts/verify_attribution.py`,
`scripts/verify_retention.py`, `scripts/smoke_compose.py`. A green run here is
not "the DB layer works", and the second gap (storage enforcing the upload size
cap) is untouched by it.
"""
from __future__ import annotations

import datetime as dt
import inspect

import psycopg
import pytest

from df.db import Db
from psycopg_shape import RecordingConnection, RecordingCursor

NOW = dt.datetime(2026, 8, 29, tzinfo=dt.timezone.utc)


def _public(obj: type) -> set[str]:
    return {n for n in dir(obj) if not n.startswith("_")}


# --- 1. the fake cannot be more permissive than the library -----------------


def test_fake_surface_is_a_subset_of_real_psycopg():
    """The stand-in may expose less than psycopg. It must never expose more.

    CLAUDE.md records three divergences between a fake and the real thing, and
    every one had the fake as the permissive side, so the fake-only check
    stayed green. The allowed surface is therefore read off the installed
    library here rather than written down anywhere.
    """
    pairs = (
        (RecordingConnection, psycopg.Connection),
        (RecordingCursor, psycopg.Cursor),
    )
    for fake, real in pairs:
        extra = _public(fake) - _public(real)
        assert not extra, (
            f"{fake.__name__} exposes {sorted(extra)}, which {real.__name__} "
            "does not. A fake wider than the library is how the DB layer broke "
            "before."
        )


def test_executemany_is_cursor_only__with_a_positive_control():
    """The absence claim, and its sibling presence claim in the same setup.

    CLAUDE.md: any block asserting "X is absent" needs one asserting "X is
    present", or it is only evidence that the setup is broken in the convenient
    direction. Both halves are checked against real psycopg AND the stand-in,
    so the two cannot drift apart.
    """
    # Absent on Connection -- this is the bug that shipped.
    assert not hasattr(psycopg.Connection, "executemany")
    assert not hasattr(RecordingConnection, "executemany")

    # Present on Cursor -- the positive control. Without it, the two assertions
    # above would pass just as happily against a misspelled attribute name.
    assert hasattr(psycopg.Cursor, "executemany")
    assert hasattr(RecordingCursor, "executemany")


# --- 2. every Db method runs its real body ----------------------------------


CASES: list[tuple[str, list, dict]] = [
    ("create_job", [], dict(media_type="video", raw_object_key="raw/j/o",
                            derived_prefix="derived/j/", submitted_by="key:abc")),
    ("get_job", ["j"], {}),
    ("set_status", ["j", "queued"], {}),
    ("set_content_hash", ["j", "deadbeef"], {}),
    ("bump_attempts", ["j"], {}),
    ("insert_items", ["j", [{"item_index": 0, "item_kind": "face", "face_index": 0,
                             "score": 12.5, "confidence": 0.9, "object_key": "k",
                             "model_version_id": "m", "face_w": 80, "face_h": 90,
                             "calibration": "temperature.v1:unfitted",
                             "model_validation": "placeholder"}]], {}),
    ("item_model_versions", ["j"], {}),
    ("item_model_validations", ["j"], {}),
    ("item_calibrations", ["j"], {}),
    ("get_items", ["j"], {}),
    ("write_result", ["j"], dict(result_class="authentic", band="likely_authentic",
                                 aggregate_score=11.0, model_version_id="m",
                                 aggregation_method="weighted_trimmed_mean.v1",
                                 aggregation_params={"trim_frac": 0.1}, item_count=3,
                                 items_total=4, face_count=1, items_unattributed=0,
                                 calibration="temperature.v1:unfitted",
                                 model_validation="placeholder")),
    ("flag_for_review", ["j", "uncertain band"], {}),
    ("get_retention_row", ["j"], {}),
    ("mark_media_deleted", ["j"], dict(raw=True, derived=True, at=NOW)),
    ("mark_cold_deleted", ["j", NOW], {}),
    ("set_extended_retention", ["j", NOW], dict(cold_prefix="cold/j/")),
    ("set_retention_hold", ["j"], dict(reason="dispute")),
    ("record_event", ["j", "media_deleted", {"raw": True}], {}),
    ("find_undeleted_terminal", [], {}),
    ("find_stalled_in_flight", [NOW], {}),
    ("mark_job_failed", ["j", "stalled"], {}),
    ("find_abandoned_uploads", [NOW], {}),
    ("find_expired_extended_retention", [NOW], {}),
]

# Rows a method needs handed back so its own post-processing runs rather than
# short-circuiting on an empty result.
ROWS: dict[str, list[list[dict]]] = {
    "create_job": [[{"id": "job-1"}]],
    "get_job": [[{"id": "job-1", "status": "queued"}]],
    "bump_attempts": [[{"attempts": 2}]],
    "item_model_versions": [[{"model_version_id": "face-efficientnet_b4-abc"}]],
    "item_model_validations": [[{"model_validation": "research-checkpoint"}]],
    "item_calibrations": [[{"calibration": "temperature.v1:unfitted"}]],
    "get_items": [[{"item_index": 0, "item_kind": "face", "face_index": 0,
                    "score": 9.0, "confidence": 0.9, "object_key": "k",
                    "model_version_id": "m", "model_validation": "placeholder"}]],
    "get_retention_row": [[{"id": "job-1", "retention_hold": False,
                            "raw_object_key": "raw/j/o",
                            "derived_prefix": "derived/j/",
                            "raw_deleted_at": None, "derived_deleted_at": None,
                            "cold_prefix": None, "cold_deleted_at": None,
                            "extended_retention_until": None}]],
    "find_undeleted_terminal": [[{"id": "job-1"}]],
    "find_stalled_in_flight": [[{"id": "job-1"}]],
    "find_abandoned_uploads": [[{"id": "job-1"}]],
    "find_expired_extended_retention": [[{"id": "job-1"}]],
}


def test_case_table_covers_every_public_db_method():
    """A coverage table that silently stops covering something is the failure
    mode CLAUDE.md calls "silently stopped checking". Adding a method to Db
    without adding it here fails this, rather than passing quietly."""
    covered = {name for name, _, _ in CASES}
    public = {
        n for n, _ in inspect.getmembers(Db, predicate=inspect.isfunction)
        if not n.startswith("_") and n != "conn"
    }
    assert not public - covered, f"uncovered Db methods: {sorted(public - covered)}"
    assert not covered - public, f"CASES names absent methods: {sorted(covered - public)}"


@pytest.mark.parametrize("name,args,kwargs", CASES, ids=[c[0] for c in CASES])
def test_db_method_touches_only_real_psycopg_api(monkeypatch, name, args, kwargs):
    """Runs the actual method body. An AttributeError here is the shipped bug.

    psycopg-shape only: no SQL is executed. The statements are proven by
    scripts/verify_attribution.py and scripts/verify_retention.py.
    """
    conn = RecordingConnection(rows=[list(r) for r in ROWS.get(name, [])])
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: conn)

    getattr(Db(dsn="postgresql://unused"), name)(*args, **kwargs)

    assert conn.calls, f"{name} opened a connection but issued no statement"


# --- 3. regressions for the two divergences that actually shipped -----------


def test_insert_items_routes_executemany_through_a_cursor(monkeypatch):
    """The bug: `c.executemany(...)` on a Connection. psycopg3 has no such
    method, so it raised against a real database and dead-lettered every job.

    Shown RED before being trusted: `scripts/mutate.py` carries this as
    `insert_items_executemany_on_connection`, which reverts db.py to the
    Connection call and must report RED.
    """
    conn = RecordingConnection()
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: conn)

    Db(dsn="postgresql://unused").insert_items("j", [
        {"item_index": 0, "item_kind": "face", "face_index": 0, "score": 1.0,
         "confidence": 0.9, "object_key": "k", "model_version_id": "m",
         "model_validation": "placeholder"},
    ])

    assert "executemany" in conn._kinds(), "insert_items must batch, not loop"
    assert "transaction" in conn._kinds(), "the batch must be transactional"


def test_get_items_selects_the_columns_its_callers_read(monkeypatch):
    """The second divergence CLAUDE.md records: `get_items` stopped selecting
    `model_version_id` while FakeDb kept returning it, so attribution looked
    fine in pytest and was NULL in Postgres. The column list is load-bearing --
    the router derives model_version_id and model_validation from these rows,
    and a missing one degrades silently to the strongest caveat rather than
    raising.
    """
    conn = RecordingConnection(rows=[[]])
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: conn)

    Db(dsn="postgresql://unused").get_items("j")

    sql = conn._statements()[0]
    for column in ("item_index", "item_kind", "face_index", "score", "confidence",
                   "object_key", "model_version_id", "model_validation",
                   # Added by migration 006. face_evidence in the API reads
                   # these; dropping them from the SELECT would silently make
                   # every face report "no geometry recorded" rather than fail.
                   "face_w", "face_h",
                   # Migration 007. Dropping this would make every result read
                   # as "calibration not recorded" instead of failing.
                   "calibration"):
        assert column in sql, f"get_items stopped selecting {column}"
