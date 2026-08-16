"""Prove model attribution against real Postgres and a real router worker.

Runs INSIDE a container:

    docker compose exec -T gateway python scripts/verify_attribution.py

Why this exists: the unit tests for attribution were mutation-verified against
FakeDb, which proves only that they distinguish the fake's broken mode from the
fake's fixed mode. The refusal branch had never been observed against real
Postgres, a real DISTINCT query, or the real router process -- the same trap the
queue work named, applied to the technique meant to prevent it rather than to a
bug.

Nothing here is simulated except the cause. The rows go into real Postgres under
the real unique index, the aggregate message goes onto the real Redis stream,
and the refusal is performed by the router CONTAINER, not by an in-process call.
The dead-letter is then read back out of Postgres.

Manufacturing the mixed-model state honestly: a single insert_items call is
atomic, so two consumers writing the SAME item set can never split -- the first
wins all of them. They split only when the two runs produce different item
sets, which is what a rolling deploy that changed frame sampling looks like.
Consumer v1 writes items 0-3, consumer v2 writes 0-7: v2's 0-3 lose the
ON CONFLICT, its 4-7 land, and the job now genuinely holds rows from two models.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/app/src")

from df.db import Db  # noqa: E402
from df.queue import TOPIC_AGGREGATE, build_queue  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def rows_for(model: str, lo: int, hi: int) -> list[dict]:
    return [
        {
            "item_index": i, "item_kind": "frame", "face_index": 0,
            "score": 10.0 if model.endswith("v1") else 90.0,
            "confidence": 0.9, "object_key": f"derived/x/f{i}.png",
            "model_version_id": model,
        }
        for i in range(lo, hi)
    ]


def wait_for_status(db: Db, job_id: str, wanted: set[str], seconds: int = 60) -> str:
    deadline = time.time() + seconds
    status = ""
    while time.time() < deadline:
        job = db.get_job(job_id)
        status = job["status"] if job else "?"
        if status in wanted:
            return status
        time.sleep(1)
    return status


def main() -> int:
    db = Db()
    queue = build_queue()

    # --- 1. a genuinely mixed job, refused by the live router ---------------
    mixed = db.create_job(
        media_type="video", raw_object_key="", derived_prefix="", submitted_by="probe"
    )
    db.set_status(mixed, "queued")

    db.insert_items(mixed, rows_for("face-probe-v1", 0, 4))
    db.insert_items(mixed, rows_for("face-probe-v2", 0, 8))

    observed = db.item_model_versions(mixed)
    check("real Postgres reports both producers", observed == ["face-probe-v1", "face-probe-v2"],
          str(observed))

    stored = db.get_items(mixed)
    v1 = sum(1 for r in stored if r["model_version_id"] == "face-probe-v1")
    v2 = sum(1 for r in stored if r["model_version_id"] == "face-probe-v2")
    check("the unique index split the rows between them", (v1, v2) == (4, 4),
          f"v1={v1} v2={v2} -- v2's first four lost ON CONFLICT, its last four landed")

    # Hand it to the live router the way a real duplicate delivery would.
    queue.push(TOPIC_AGGREGATE, {
        "job_id": mixed, "media_type": "video",
        "model_version_id": "face-probe-v2",   # hearsay: the message's claim
    })
    status = wait_for_status(db, mixed, {"dead_letter", "complete", "failed"})
    check("the router container refused it", status == "dead_letter",
          f"status={status}")

    job = db.get_job(mixed)
    check("refusal names the cause", "multiple model versions" in (job["error"] or ""),
          (job["error"] or "")[:90])
    check("no verdict was stored", job["result_class"] is None and job["band"] is None,
          f"class={job['result_class']} band={job['band']}")
    check("no model version was invented", job["model_version_id"] is None,
          str(job["model_version_id"]))

    # --- 2. the migration straddle: some rows NULL, some recorded ----------
    straddle = db.create_job(
        media_type="video", raw_object_key="", derived_prefix="", submitted_by="probe"
    )
    db.set_status(straddle, "queued")

    pre = rows_for("ignored", 0, 3)
    for r in pre:
        r["model_version_id"] = None          # written before migration 003
    db.insert_items(straddle, pre)
    db.insert_items(straddle, rows_for("face-probe-v1", 3, 6))

    observed = db.item_model_versions(straddle)
    check("NULL producers are excluded, not counted as a second model",
          observed == ["face-probe-v1"], str(observed))

    queue.push(TOPIC_AGGREGATE, {
        "job_id": straddle, "media_type": "video",
        "model_version_id": "face-probe-v1",
    })
    status = wait_for_status(db, straddle, {"complete", "dead_letter", "failed"})
    check("a straddling job is NOT refused", status == "complete", f"status={status}")

    job = db.get_job(straddle)
    check("attributed to the one recorded producer",
          job["model_version_id"] == "face-probe-v1", str(job["model_version_id"]))

    with db.conn() as c:
        events = [
            r["event"] for r in c.execute(
                "SELECT event FROM job_events WHERE job_id = %s", (straddle,)
            ).fetchall()
        ]
    check("partial provenance is recorded, not silently treated as full",
          "router.partial_provenance" in events, str(events))

    print("\n" + ("all attribution checks passed" if not failures
                  else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
