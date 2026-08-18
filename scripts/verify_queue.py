"""Verify the Streams queue against a real Redis.

Runs INSIDE a container, because Redis sits on the internal network with no
host port and publishing one to test would weaken the isolation being tested:

    docker compose exec -T -e DF_QUEUE_RECLAIM_MS=1000 gateway \
        python scripts/verify_queue.py

Why this is a script and not a pytest: the behaviour that matters here is
Redis's, not ours. Consumer-group ownership, the pending entries list and
XAUTOCLAIM's idle accounting are the things being relied on, and a fake that
reimplemented them would be asserting that our idea of Streams is
self-consistent -- the same trap as FakeDb accepting executemany on a
Connection while real psycopg3 refused it.

The reclaim threshold is lowered by env for the probe. That changes only how
long this test waits, not what it proves: the property is that an idle,
unacked message becomes claimable by another consumer, not that it does so at
exactly 120 seconds.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/app/src")

from df.config import settings  # noqa: E402
from df.queue import TOPIC_PREPROCESS, Message, RedisStreamQueue  # noqa: E402

TOPIC = "verify-probe"
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main() -> int:
    print(f"backend={settings.queue_backend} reclaim_ms={settings.queue_reclaim_ms}\n")

    a = RedisStreamQueue(consumer="probe-a")
    b = RedisStreamQueue(consumer="probe-b")

    # Start from a clean stream so a rerun cannot pass on leftovers.
    a.r.delete(a._stream(TOPIC), a._dlq(TOPIC))

    # --- ordinary delivery ---
    a.push(TOPIC, {"job_id": "probe-1"})
    msg = a.pop(TOPIC, timeout=2)
    check("message delivered", msg is not None and msg.payload["job_id"] == "probe-1")
    check("delivery carries a stream entry id", bool(msg and msg.entry_id), str(msg.entry_id))

    a.ack(msg)
    pending = a.r.xpending(a._stream(TOPIC), a.GROUP)
    check("ack clears the pending entry", pending["pending"] == 0, str(pending["pending"]))

    # --- the capability lists did not have ---
    a.push(TOPIC, {"job_id": "probe-2"})
    taken = a.pop(TOPIC, timeout=2)
    check("consumer A took the message", taken is not None)

    # Owned by A and not yet idle: B must not be able to steal live work.
    stolen = b.pop(TOPIC, timeout=1)
    check(
        "another consumer cannot take work still owned and fresh",
        stolen is None,
        f"got {stolen.payload if stolen else None}",
    )

    # A "dies" here -- it never acks. Wait for the entry to go idle.
    time.sleep(settings.queue_reclaim_ms / 1000 + 0.5)

    reclaimed = b.pop(TOPIC, timeout=2)
    check(
        "an idle unacked message is reclaimed by another consumer",
        reclaimed is not None and reclaimed.payload["job_id"] == "probe-2",
        "no worker restart was involved",
    )
    if reclaimed:
        b.ack(reclaimed)

    pending = b.r.xpending(b._stream(TOPIC), b.GROUP)
    check("nothing left pending after reclaim + ack", pending["pending"] == 0,
          str(pending["pending"]))

    # --- retry/DLQ semantics must match the list backend exactly ---
    a.push(TOPIC, {"job_id": "probe-3"})
    msg = a.pop(TOPIC, timeout=2)
    retries = 0
    while a.fail(msg, "boom"):
        retries += 1
        msg = a.pop(TOPIC, timeout=2)
        if msg is None:
            break
    check("retries up to the limit then dead-letters",
          retries == settings.max_attempts - 1, f"{retries} retries")
    check("dead letter recorded", a.dead_letter_size(TOPIC) == 1,
          str(a.dead_letter_size(TOPIC)))
    check("stream drained after dead-lettering", a.pop(TOPIC, timeout=1) is None)

    # --- the group vanishing under a live consumer ---
    # An operator deletes the stream, or Redis returns from a snapshot older
    # than the group. The consumer must heal rather than wedge on NOGROUP
    # until someone restarts it -- restart-only recovery is the weakness this
    # backend exists to remove.
    a.r.delete(a._stream(TOPIC), a._dlq(TOPIC))

    # Plant the message BEFORE destroying the group, and destroy only the group
    # so the entry survives. Pushing afterwards instead would prove nothing:
    # a brand-new message is delivered whether the group was recreated at 0 or
    # at $, so that version of this test passes even if the recreate silently
    # skips everything already in the stream.
    a.push(TOPIC, {"job_id": "probe-regroup"})
    a.r.xgroup_destroy(a._stream(TOPIC), a.GROUP)

    healed = a.pop(TOPIC, timeout=2)
    check(
        "a message already in the stream survives losing its group",
        healed is not None and healed.payload["job_id"] == "probe-regroup",
        "recreate must start at 0, not $ -- $ would abandon it",
    )
    if healed:
        a.ack(healed)

    # --- poison message: kills its consumer before fail() can run ---
    a.r.delete(a._stream(TOPIC), a._dlq(TOPIC))
    a.push(TOPIC, {"job_id": "probe-poison"})
    seen = 0
    for _ in range(settings.max_attempts + 3):
        got = a.pop(TOPIC, timeout=1)      # take it, never ack: consumer "dies"
        if got is None:
            break
        seen += 1
        time.sleep(settings.queue_reclaim_ms / 1000 + 0.2)
    check(
        "a message that always kills its consumer is eventually dead-lettered",
        a.dead_letter_size(TOPIC) == 1,
        f"claimed {seen} times, dlq={a.dead_letter_size(TOPIC)}",
    )

    # --- has_message_for: the signal sweep_stalled_jobs trusts ---
    # It decides whether a job is stranded or merely waiting behind a slow
    # worker. A wrong False gets a healthy job swept and its upload deleted, so
    # all three shapes are checked here rather than against a fake, which
    # cannot reproduce XRANGE paging at all.
    #
    # These use a REAL topic. has_message_for only scans ALL_TOPICS, so running
    # them on the probe topic made every answer False and the negative cases
    # pass for no reason at all -- caught by the positive cases failing.
    probe_topic = TOPIC_PREPROCESS
    a.r.delete(a._stream(probe_topic))

    check(
        "no message means no message (the write/push race this defends)",
        a.has_message_for("job-never-pushed") is False,
        "Redis reachable, XADD never issued",
    )

    a.push(probe_topic, {"job_id": "job-in-pel"})
    taken = a.pop(probe_topic, timeout=2)
    check(
        "a taken but unacked message still counts as alive",
        a.has_message_for("job-in-pel") is True,
        "entries are XDELed on ack, so one still present is still work",
    )
    if taken:
        a.ack(taken)

    a.r.delete(a._stream(probe_topic))
    for i in range(1500):
        a.push(probe_topic, {"job_id": "bulk-%d" % i})
    check(
        "a message past the first page is still found",
        a.has_message_for("bulk-1499") is True,
        "returned False when the scan stopped at 1000, and it failed under deep "
        "backlog, the exact condition this check exists for",
    )
    check(
        "and an absent job is still absent after a full scan",
        a.has_message_for("job-not-here") is False,
    )
    a.r.delete(a._stream(probe_topic))

    a.r.delete(a._stream(TOPIC), a._dlq(TOPIC))
    print("\n" + ("all queue checks passed" if not failures else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
