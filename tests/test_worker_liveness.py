"""Health check that reports the WORKERS, not just the gateway.

The failure this exists for: the job flow is asynchronous, so nothing blocks on
the model warming and a fast 200 is correct for the gateway. That same property
makes a dead inference worker invisible from the health endpoint. The platform
sees a healthy service, the gateway accepts the upload, the job lands in Redis,
and nothing picks it up. No error, no failed status -- a client watching a
spinner forever, looking exactly like a model thinking hard.

There are two layers and they cover different things:

  * df.deploy's supervisor catches a worker PROCESS exiting, in ~2s. Verified
    live: SIGKILL on the inference worker produced "gpu-inference exited with
    -9; shutting down the container".
  * this heartbeat catches a worker that is alive but not working -- wedged,
    or having lost Redis. Verified live with SIGSTOP: healthz went 503 with
    degraded=["inference"] after the TTL, and back to 200 on SIGCONT.

# In-process here. The live counterpart is those two container experiments.
"""
from __future__ import annotations

import pytest

from df.gateway.app import REQUIRED_WORKER_TOPICS, WORKER_GRACE_SECONDS


class FakeStatus:
    def __init__(self, alive: dict[str, bool], raises: bool = False) -> None:
        self._alive, self._raises = alive, raises

    def worker_alive(self, topic: str) -> bool:
        if self._raises:
            raise ConnectionError("redis down")
        return self._alive.get(topic, False)


def _call(monkeypatch, status, elapsed: float):
    """Invoke healthz with a controlled clock, so the grace window is testable
    without sleeping through it."""
    import df.gateway.app as app_mod

    monkeypatch.setattr(app_mod, "_status", status)
    monkeypatch.setattr(app_mod, "_STARTED_AT", 0.0)
    monkeypatch.setattr(app_mod.time, "monotonic", lambda: elapsed)
    import json as _json

    resp = app_mod.healthz()
    return resp.status_code, _json.loads(bytes(resp.body).decode())


def test_all_workers_alive_is_healthy(monkeypatch):
    status = FakeStatus({t: True for t in REQUIRED_WORKER_TOPICS})

    code, body = _call(monkeypatch, status, WORKER_GRACE_SECONDS + 10)

    assert code == 200
    assert body["ok"] is True
    assert body["workers"] == {t: True for t in REQUIRED_WORKER_TOPICS}
    assert "degraded" not in body


def test_a_dead_worker_fails_the_check_after_the_grace(monkeypatch):
    """The point. A platform restarts the container on this instead of routing
    uploads into a queue nothing is reading."""
    alive = {t: True for t in REQUIRED_WORKER_TOPICS}
    alive["inference"] = False

    code, body = _call(monkeypatch, FakeStatus(alive), WORKER_GRACE_SECONDS + 10)

    assert code == 503
    assert body["ok"] is False
    assert body["degraded"] == ["inference"]
    assert body["warming"] is False


def test_a_dead_worker_is_tolerated_DURING_the_grace(monkeypatch):
    """Workers take seconds to boot and the inference worker loads a 254MB model
    first (6-16s measured). Without a grace the first health check fails and the
    deploy is rolled back before anything had a chance to start.

    Still reported as degraded in the body, so a slow boot can be told apart
    from a stuck one."""
    alive = {t: True for t in REQUIRED_WORKER_TOPICS}
    alive["inference"] = False

    code, body = _call(monkeypatch, FakeStatus(alive), WORKER_GRACE_SECONDS - 1)

    assert code == 200
    assert body["degraded"] == ["inference"]
    assert body["warming"] is True


def test_redis_unreachable_is_unhealthy_not_assumed_healthy(monkeypatch):
    """Failing closed. The queue lives in Redis, so if it cannot be reached the
    service genuinely cannot process a job -- guessing 'probably fine' here is
    how the spinner-forever failure comes back by another route."""
    code, body = _call(monkeypatch, FakeStatus({}, raises=True), WORKER_GRACE_SECONDS + 10)

    assert code == 503
    assert sorted(body["degraded"]) == sorted(REQUIRED_WORKER_TOPICS)


def test_the_retention_sweeper_cannot_fail_the_check():
    """It runs on a timer; its absence delays cleanup rather than stranding a
    job. Including it would let a non-critical worker roll back a deploy."""
    assert "retention" not in " ".join(REQUIRED_WORKER_TOPICS)
    assert set(REQUIRED_WORKER_TOPICS) == {"preprocess", "inference", "aggregate"}
