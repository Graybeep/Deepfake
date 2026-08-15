"""Job queue on Redis lists, with retry-limit + dead-letter (Tier 2).

Deliberately lists, not Streams: CLAUDE.md puts Redis Streams consumer groups in
week 2+. Lists with a processing-list handoff (BRPOPLPUSH) give at-least-once
delivery, which is enough for the MVP as long as workers are idempotent.

DLQ policy is Tier 2: retry limit, dead-letter list, one log line. No PagerDuty.
"""
from __future__ import annotations

import json
import logging
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
