from __future__ import annotations

from df.config import settings
from df.queue import InMemoryQueue


def test_push_then_pop_round_trips_the_payload():
    q = InMemoryQueue()
    q.push("preprocess", {"job_id": "j1", "media_type": "video"})

    msg = q.pop("preprocess")

    assert msg.payload["job_id"] == "j1"
    assert msg.topic == "preprocess"


def test_empty_queue_returns_none():
    assert InMemoryQueue().pop("preprocess") is None


def test_failures_retry_up_to_the_limit_then_dead_letter():
    q = InMemoryQueue()
    q.push("preprocess", {"job_id": "j1"})

    msg = q.pop("preprocess")
    retries = 0
    while q.fail(msg, "boom"):
        retries += 1
        msg = q.pop("preprocess")
        assert msg is not None

    assert retries == settings.max_attempts - 1
    assert q.dead_letter_size("preprocess") == 1
    assert q.pop("preprocess") is None


def test_dead_letter_entry_keeps_the_payload_and_error():
    q = InMemoryQueue()
    q.push("inference", {"job_id": "j2"})

    msg = q.pop("inference")
    while q.fail(msg, "gpu oom"):
        msg = q.pop("inference")

    entry = q.dead["inference"][0]
    assert entry["payload"]["job_id"] == "j2"
    assert entry["error"] == "gpu oom"


def test_topics_are_isolated():
    q = InMemoryQueue()
    q.push("preprocess", {"job_id": "a"})

    assert q.pop("inference") is None
    assert q.pop("preprocess").payload["job_id"] == "a"
