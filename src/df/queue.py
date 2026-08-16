"""Job queue with retry-limit + dead-letter (Tier 2).

Two Redis backends behind one protocol, chosen by DF_QUEUE_BACKEND:

  streams (default) -- Streams + consumer groups. A message a worker took but
      never acked sits in that consumer's pending list, and ANY live consumer
      can claim it back once it goes idle (XAUTOCLAIM). Recovery no longer
      depends on a worker restarting, and more than one consumer can serve a
      topic, which is what makes the GPU worker horizontally scalable.

  lists -- the original BRPOPLPUSH handoff. Kept as a rollback path for a
      component on the critical data path. Recovery there needs a worker
      restart (requeue_stale_processing), so a message stranded while all
      workers stay up is invisible until the retention sweeper catches the job
      hours later. That is the concrete gap Streams closes.

Neither backend can recover a message that was never written -- the gateway and
the workers commit the job's status to Postgres before pushing here, so a crash
in between still strands the row. That is what sweep_stalled_jobs is for.

DLQ policy is Tier 2 either way: retry limit, dead-letter list, one log line.
No PagerDuty. Both backends write the same q:<topic>:dead list, so there is one
place to look regardless of which is running.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from df.config import settings

log = logging.getLogger("df.queue")

TOPIC_PREPROCESS = "preprocess"
TOPIC_INFERENCE = "inference"
TOPIC_AGGREGATE = "aggregate"


@dataclass
class Message:
    topic: str
    payload: dict[str, Any]
    attempts: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Stream entry id, set only by the Streams backend. Transport-level, so it
    # is deliberately not part of encode() -- it identifies this delivery, not
    # this message, and must not survive a requeue.
    entry_id: str | None = None

    def encode(self) -> str:
        return json.dumps(
            {"id": self.id, "topic": self.topic, "payload": self.payload, "attempts": self.attempts}
        )

    @staticmethod
    def decode(raw: str) -> "Message":
        d = json.loads(raw)
        return Message(topic=d["topic"], payload=d["payload"], attempts=d.get("attempts", 0), id=d["id"])


class Queue(Protocol):
    def push(self, topic: str, payload: dict[str, Any]) -> str: ...
    def pop(self, topic: str, timeout: int = 5) -> Message | None: ...
    def ack(self, msg: Message) -> None: ...
    def fail(self, msg: Message, error: str) -> bool: ...
    def dead_letter_size(self, topic: str) -> int: ...


class RedisQueue:
    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import redis

            client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.r = client

    @staticmethod
    def _key(topic: str) -> str:
        return f"q:{topic}"

    @staticmethod
    def _processing(topic: str) -> str:
        return f"q:{topic}:processing"

    @staticmethod
    def _dlq(topic: str) -> str:
        return f"q:{topic}:dead"

    def push(self, topic: str, payload: dict[str, Any]) -> str:
        msg = Message(topic=topic, payload=payload)
        self.r.lpush(self._key(topic), msg.encode())
        return msg.id

    def pop(self, topic: str, timeout: int = 5) -> Message | None:
        raw = self.r.brpoplpush(self._key(topic), self._processing(topic), timeout)
        if raw is None:
            return None
        return Message.decode(raw)

    def ack(self, msg: Message) -> None:
        self.r.lrem(self._processing(msg.topic), 1, msg.encode())

    def fail(self, msg: Message, error: str) -> bool:
        """Requeue, or dead-letter once the retry limit is hit.

        Returns True if the message was retried, False if it was dead-lettered.
        """
        self.ack(msg)
        msg.attempts += 1
        if msg.attempts >= settings.max_attempts:
            self.r.lpush(
                self._dlq(msg.topic),
                json.dumps(
                    {
                        "id": msg.id,
                        "topic": msg.topic,
                        "payload": msg.payload,
                        "attempts": msg.attempts,
                        "error": error,
                        "dead_lettered_at": time.time(),
                    }
                ),
            )
            # Tier 2: the log line IS the alert. No pager.
            log.error(
                "DLQ topic=%s msg=%s job=%s attempts=%d error=%s",
                msg.topic,
                msg.id,
                msg.payload.get("job_id"),
                msg.attempts,
                error,
            )
            return False

        self.r.lpush(self._key(msg.topic), msg.encode())
        log.warning(
            "retry topic=%s msg=%s attempt=%d error=%s", msg.topic, msg.id, msg.attempts, error
        )
        return True

    def dead_letter_size(self, topic: str) -> int:
        return int(self.r.llen(self._dlq(topic)))

    def requeue_stale_processing(self, topic: str) -> int:
        """Move anything stranded in the processing list back onto the queue.

        Run at worker startup: a worker killed mid-message leaves its message in
        the processing list forever otherwise.
        """
        moved = 0
        while True:
            raw = self.r.rpoplpush(self._processing(topic), self._key(topic))
            if raw is None:
                return moved
            moved += 1


class RedisStreamQueue:
    """Streams + consumer groups. Same protocol as RedisQueue.

    Retry accounting stays in the payload (Message.attempts) rather than moving
    to the PEL delivery counter, so both backends dead-letter after exactly the
    same number of handler failures. The delivery counter is used for a
    different job: catching a message that keeps killing whatever consumer
    claims it, which the payload counter cannot see because the handler never
    got far enough to call fail().
    """

    GROUP = "df"

    def __init__(self, client: Any | None = None, consumer: str | None = None) -> None:
        if client is None:
            import redis

            client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.r = client
        # Must be unique per process: two consumers sharing a name share a
        # pending list, and each would think the other's work was its own.
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"
        self._groups: set[str] = set()

    @staticmethod
    def _stream(topic: str) -> str:
        return f"s:{topic}"

    @staticmethod
    def _dlq(topic: str) -> str:
        return f"q:{topic}:dead"

    def _ensure_group(self, topic: str) -> None:
        if topic in self._groups:
            return
        import redis

        try:
            self.r.xgroup_create(self._stream(topic), self.GROUP, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._groups.add(topic)

    def push(self, topic: str, payload: dict[str, Any]) -> str:
        msg = Message(topic=topic, payload=payload)
        self._ensure_group(topic)
        self.r.xadd(self._stream(topic), {"data": msg.encode()})
        return msg.id

    def pop(self, topic: str, timeout: int = 5) -> Message | None:
        self._ensure_group(topic)

        # Abandoned work first: a message whose consumer died is more urgent
        # than a new one, and leaving it means a job silently stops moving.
        reclaimed = self._reclaim(topic)
        if reclaimed is not None:
            return reclaimed

        resp = self._read_new(topic, timeout)
        if not resp:
            return None
        _stream, entries = resp[0]
        if not entries:
            return None
        entry_id, fields = entries[0]
        msg = Message.decode(fields["data"])
        msg.entry_id = entry_id
        return msg

    def _read_new(self, topic: str, timeout: int) -> Any:
        """XREADGROUP, recreating the group if it has gone missing.

        The group can vanish under a running consumer: an operator deletes the
        stream, or Redis comes back from a snapshot taken before the group was
        created. Without this the consumer raises NOGROUP on every poll and
        stays wedged until the process restarts -- which is the same
        "recovery requires a restart" weakness that motivated moving off lists.
        """
        import redis

        args = (self.GROUP, self.consumer, {self._stream(topic): ">"})
        kwargs = {"count": 1, "block": max(1, timeout) * 1000}
        try:
            return self.r.xreadgroup(*args, **kwargs)
        except redis.ResponseError as exc:
            if "NOGROUP" not in str(exc):
                raise
            log.warning("consumer group missing on topic=%s, recreating", topic)
            self._groups.discard(topic)
            self._ensure_group(topic)
            return self.r.xreadgroup(*args, **kwargs)

    def _reclaim(self, topic: str) -> Message | None:
        """Take back a message whose consumer stopped without acking it.

        This is the capability lists did not have. There, a stranded message
        waited for a worker restart; here any live consumer picks it up once it
        has been idle long enough.
        """
        import redis

        stream = self._stream(topic)
        try:
            result = self.r.xautoclaim(
                stream, self.GROUP, self.consumer,
                min_idle_time=settings.queue_reclaim_ms, count=1,
            )
        except redis.ResponseError:
            return None

        # redis-py returns (cursor, entries) or (cursor, entries, deleted).
        entries = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else []
        if not entries:
            return None
        entry_id, fields = entries[0]
        if not fields or "data" not in fields:
            # Entry was deleted underneath us; drop the dangling PEL record.
            self.r.xack(stream, self.GROUP, entry_id)
            return None

        msg = Message.decode(fields["data"])
        msg.entry_id = entry_id

        # A message that keeps killing its consumer never reaches fail(), so the
        # payload counter stays put and it would be reclaimed forever.
        delivered = self._times_delivered(stream, entry_id)
        if delivered > settings.max_attempts:
            self._dead_letter(
                msg,
                f"reclaimed {delivered} times without completing; "
                "consumer keeps dying before it can fail cleanly",
            )
            self._drop(stream, entry_id)
            return None

        log.warning(
            "reclaimed topic=%s msg=%s job=%s delivery=%d -- previous consumer "
            "stopped without acking",
            topic, msg.id, msg.payload.get("job_id"), delivered,
        )
        return msg

    def _times_delivered(self, stream: str, entry_id: str) -> int:
        try:
            pending = self.r.xpending_range(
                stream, self.GROUP, min=entry_id, max=entry_id, count=1
            )
        except Exception:  # noqa: BLE001 - never let bookkeeping kill the worker
            return 1
        if not pending:
            return 1
        return int(pending[0].get("times_delivered", 1))

    def _drop(self, stream: str, entry_id: str | None) -> None:
        if entry_id is None:
            return
        self.r.xack(stream, self.GROUP, entry_id)
        # Acked entries still occupy the stream; delete so it stays bounded.
        self.r.xdel(stream, entry_id)

    def ack(self, msg: Message) -> None:
        self._drop(self._stream(msg.topic), msg.entry_id)

    def fail(self, msg: Message, error: str) -> bool:
        """Requeue, or dead-letter once the retry limit is hit.

        Returns True if the message was retried, False if it was dead-lettered.
        """
        self._drop(self._stream(msg.topic), msg.entry_id)
        msg.attempts += 1
        if msg.attempts >= settings.max_attempts:
            self._dead_letter(msg, error)
            return False

        msg.entry_id = None  # this delivery is over; the requeue is a new one
        self.r.xadd(self._stream(msg.topic), {"data": msg.encode()})
        log.warning(
            "retry topic=%s msg=%s attempt=%d error=%s",
            msg.topic, msg.id, msg.attempts, error,
        )
        return True

    def _dead_letter(self, msg: Message, error: str) -> None:
        self.r.lpush(
            self._dlq(msg.topic),
            json.dumps(
                {
                    "id": msg.id,
                    "topic": msg.topic,
                    "payload": msg.payload,
                    "attempts": msg.attempts,
                    "error": error,
                    "dead_lettered_at": time.time(),
                }
            ),
        )
        # Tier 2: the log line IS the alert. No pager.
        log.error(
            "DLQ topic=%s msg=%s job=%s attempts=%d error=%s",
            msg.topic, msg.id, msg.payload.get("job_id"), msg.attempts, error,
        )

    def dead_letter_size(self, topic: str) -> int:
        return int(self.r.llen(self._dlq(topic)))

    def requeue_stale_processing(self, topic: str) -> int:
        """No-op: reclaiming is continuous here, not a startup step.

        Kept so run_worker() can call it against either backend.
        """
        return 0


def build_queue() -> Queue:
    """Pick a backend. Streams unless explicitly rolled back to lists."""
    if settings.redis_url.startswith("memory://"):
        return InMemoryQueue()
    if settings.queue_backend == "lists":
        return RedisQueue()
    return RedisStreamQueue()


class InMemoryQueue:
    """Test backend with the same retry/DLQ semantics."""

    def __init__(self) -> None:
        self.queues: dict[str, list[str]] = {}
        self.dead: dict[str, list[dict]] = {}

    def push(self, topic: str, payload: dict[str, Any]) -> str:
        msg = Message(topic=topic, payload=payload)
        self.queues.setdefault(topic, []).insert(0, msg.encode())
        return msg.id

    def pop(self, topic: str, timeout: int = 5) -> Message | None:
        q = self.queues.get(topic, [])
        if not q:
            return None
        return Message.decode(q.pop())

    def ack(self, msg: Message) -> None:  # nothing in flight to clear
        return None

    def fail(self, msg: Message, error: str) -> bool:
        msg.attempts += 1
        if msg.attempts >= settings.max_attempts:
            self.dead.setdefault(msg.topic, []).append(
                {"payload": msg.payload, "error": error, "attempts": msg.attempts}
            )
            log.error("DLQ topic=%s job=%s error=%s", msg.topic, msg.payload.get("job_id"), error)
            return False
        self.queues.setdefault(msg.topic, []).insert(0, msg.encode())
        return True

    def dead_letter_size(self, topic: str) -> int:
        return len(self.dead.get(topic, []))
