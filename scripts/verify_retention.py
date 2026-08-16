"""Retention against real Postgres and real object storage.

Runs INSIDE a container (Postgres is internal-only):

    docker compose exec -T gateway python scripts/verify_retention.py

Two parts, hold gate first.

PART 1 -- the hold flag actually BLOCKS deletion. This is the Tier 1 guarantee
CLAUDE.md is most emphatic about ("no delete can happen without reading it"),
and until now it was proven only against FakeDb and InMemoryStorage. The compose
smoke test proves deletion HAPPENS against a real bucket; nothing proved the gate
STOPS it. Those are different claims and only one of them was tested.

PART 2 -- the sweep predicates select the right jobs against real SQL. The fake
counterparts are hand-written Python; the real ones are raw SQL. That pair has
diverged three times in this codebase (executemany, get_items missing a column,
insert_items accepting duplicates the unique index rejects), and every time the
fake was the more permissive side, so a fake-only check stayed green.

Both parts assert on the BYTES -- objects present or absent in the bucket -- not
on a return value or a row flag. A delete path that reports success while leaving
media behind is the failure mode that matters.
"""
from __future__ import annotations

import datetime as dt
import sys

sys.path.insert(0, "/app/src")

from df import storage as storage_mod  # noqa: E402
from df.db import Db  # noqa: E402
from df.retention import (  # noqa: E402
    DeleteOutcome,
    delete_media_for_job,
    expire_extended_retention,
    open_extended_retention_window,
    sweep_abandoned_uploads,
    sweep_stalled_jobs,
    sweep_undeleted,
)

failures: list[str] = []
NOW = dt.datetime.now(dt.timezone.utc)


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def seed(db: Db, storage, *, status: str, hold: bool = False,
         age_hours: int = 0, completed: bool = False) -> tuple[str, str, str]:
    """A job holding real bytes in the real bucket."""
    job_id = db.create_job(
        media_type="video", raw_object_key="", derived_prefix="", submitted_by="probe"
    )
    raw = storage_mod.raw_key(job_id)
    derived = storage_mod.derived_prefix(job_id)

    storage.put_bytes(raw, b"probe-raw-video-bytes")
    for i in range(3):
        storage.put_bytes(f"{derived}f{i:05d}_x0.png", f"probe-crop-{i}".encode())

    with db.conn() as c, c.transaction():
        c.execute(
            """
            UPDATE jobs
               SET raw_object_key = %s, derived_prefix = %s, status = %s,
                   retention_hold = %s,
                   hold_reason = CASE WHEN %s THEN 'probe hold' ELSE NULL END,
                   created_at = now() - make_interval(hours => %s),
                   updated_at = now() - make_interval(hours => %s),
                   completed_at = CASE WHEN %s
                                       THEN now() - make_interval(hours => %s)
                                       ELSE NULL END
             WHERE id = %s
            """,
            (raw, derived, status, hold, hold, age_hours, age_hours,
             completed, age_hours, job_id),
        )
    return job_id, raw, derived


def media_present(storage, raw: str, derived: str) -> bool:
    return storage.exists(raw) or storage.list_prefix(derived) != []


def release_hold(db: Db, job_id: str) -> None:
    with db.conn() as c, c.transaction():
        c.execute("UPDATE jobs SET retention_hold = FALSE WHERE id = %s", (job_id,))


def main() -> int:
    db = Db()
    storage = storage_mod.build_storage()
    print(f"storage backend: {type(storage).__name__}\n")

    # ================= PART 1: the hold flag blocks deletion ================
    print("--- hold gate, every delete trigger, real Postgres + real bucket ---")

    # Trigger 1: the completion-triggered delete.
    job, raw, derived = seed(db, storage, status="complete", hold=True, completed=True)
    report = delete_media_for_job(job, db, storage)
    check("completion delete refuses a held job", report.outcome is DeleteOutcome.SKIPPED_HOLD,
          str(report.outcome))
    check("held media survives the completion delete", media_present(storage, raw, derived),
          "bytes gone from the bucket despite the hold")

    # The same job, hold released: the delete must actually work, or the check
    # above proves nothing except that the path is broken.
    release_hold(db, job)
    report = delete_media_for_job(job, db, storage)
    check("and deletes once the hold is released", report.outcome is DeleteOutcome.DELETED,
          str(report.outcome))
    check("media really gone from the bucket", not media_present(storage, raw, derived))

    # Trigger 2: the crash-recovery sweeper.
    job, raw, derived = seed(db, storage, status="complete", hold=True, completed=True)
    sweep_undeleted(db, storage, limit=200)
    check("crash-recovery sweep leaves held media alone", media_present(storage, raw, derived))

    # Trigger 2b/2c: the sweeps added for failed and abandoned jobs.
    job_dl, raw_dl, der_dl = seed(db, storage, status="dead_letter", hold=True)
    sweep_undeleted(db, storage, limit=200)
    check("terminal sweep leaves a held dead-lettered job alone",
          media_present(storage, raw_dl, der_dl))

    job_ab, raw_ab, der_ab = seed(db, storage, status="awaiting_upload", hold=True, age_hours=48)
    sweep_abandoned_uploads(db, storage, older_than_hours=24, limit=200)
    check("abandoned sweep leaves held media alone", media_present(storage, raw_ab, der_ab))

    job_st, raw_st, der_st = seed(db, storage, status="queued", hold=True, age_hours=48)
    sweep_stalled_jobs(db, older_than_hours=6, limit=200)
    check("stalled sweep does not delete at all", media_present(storage, raw_st, der_st))
    check("but it does mark the held job failed, so it stops hiding in-flight",
          db.get_job(job_st)["status"] == "failed", db.get_job(job_st)["status"])

    # The handoff is the point of routing deletion through the terminal sweep:
    # that finder excludes held jobs, so a formerly-stalled held job is now
    # protected twice -- by the query AND by the gate -- where before it had
    # only the gate. This scenario did not exist until that change.
    sweep_undeleted(db, storage, limit=200)
    check("held job survives the stalled -> terminal handoff",
          media_present(storage, raw_st, der_st),
          "two layers: terminal finder excludes it, gate refuses it")

    # Trigger 3: cold-storage expiry. The flag has to outrank the timer, or a
    # hold set during a dispute evaporates on day 30 anyway.
    job_cold, raw_c, der_c = seed(db, storage, status="complete", completed=True)
    driving = storage.list_prefix(der_c)[:2]
    open_extended_retention_window(job_cold, db, storage, driving_keys=driving, days=30)
    cold_keys = storage.list_prefix(storage_mod.cold_prefix(job_cold))
    check("window preserved the driving crops", len(cold_keys) == 2, str(len(cold_keys)))

    with db.conn() as c, c.transaction():
        c.execute(
            "UPDATE jobs SET retention_hold = TRUE, extended_retention_until = "
            "now() - interval '1 day' WHERE id = %s", (job_cold,)
        )
    report = expire_extended_retention(job_cold, db, storage, now=NOW)
    check("cold expiry refuses a held job", report.outcome is DeleteOutcome.SKIPPED_HOLD,
          str(report.outcome))
    check("held cold media survives its own expiry",
          all(storage.exists(k) for k in cold_keys))

    # ================= PART 2: sweep predicates on real SQL ================
    print("\n--- sweep predicates, real SQL ---")

    job, raw, derived = seed(db, storage, status="dead_letter")
    sweep_undeleted(db, storage, limit=200)
    check("dead-lettered job is swept", not media_present(storage, raw, derived),
          "the leak this sweep was added for")

    job, raw, derived = seed(db, storage, status="failed")
    sweep_undeleted(db, storage, limit=200)
    check("failed job is swept", not media_present(storage, raw, derived))

    job, raw, derived = seed(db, storage, status="awaiting_upload", age_hours=48)
    sweep_abandoned_uploads(db, storage, older_than_hours=24, limit=200)
    check("abandoned upload is swept", not media_present(storage, raw, derived))
    check("and its row is moved to a terminal state",
          db.get_job(job)["status"] == "failed", db.get_job(job)["status"])

    job, raw, derived = seed(db, storage, status="awaiting_upload", age_hours=1)
    sweep_abandoned_uploads(db, storage, older_than_hours=24, limit=200)
    check("a recent upload grant is NOT swept", media_present(storage, raw, derived),
          "deleting under a client still uploading is worse than the leak")

    job, raw, derived = seed(db, storage, status="queued", age_hours=48)
    sweep_stalled_jobs(db, older_than_hours=6, limit=200)
    check("stalled sweep marks but does not delete", media_present(storage, raw, derived))
    sweep_undeleted(db, storage, limit=200)
    check("terminal sweep then clears it", not media_present(storage, raw, derived))
    check("and is marked failed with a reason",
          "stalled" in (db.get_job(job)["error"] or ""), (db.get_job(job)["error"] or "")[:60])

    job, raw, derived = seed(db, storage, status="inference", age_hours=0)
    sweep_stalled_jobs(db, older_than_hours=6, limit=200)
    check("a job still in flight is NOT swept", media_present(storage, raw, derived),
          "deleting mid-pipeline would fail work about to succeed")

    print("\n" + ("all retention checks passed" if not failures
                  else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
