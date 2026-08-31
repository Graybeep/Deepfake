"""Job status: Redis key + pub/sub push, with polling as the reconnect fallback.

Both halves are required (Tier 1). The WebSocket carries live updates; the Redis
key is the source of truth a client reads after a dropped connection, because a
client that was disconnected during a transition never saw the message.

Redis persistence must be ON. The compose file starts redis with
`--appendonly yes --appendfsync everysec`; assert_persistence_enabled() fails
loudly at worker/gateway startup if someone points this at a default in-memory
Redis, which would drop every in-flight job on restart.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from df.config import settings

log = logging.getLogger("df.jobstatus")

# Status survives long enough for a client to reconnect and catch up; the
# durable record is Postgres, not this key.
STATUS_TTL_SECONDS = 24 * 3600


def status_key(job_id: str) -> str:
    return f"job:{job_id}:status"


def channel(job_id: str) -> str:
    return f"job:{job_id}:events"


class JobStatus:
    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import redis

            client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.r = client

    # --- worker liveness ---------------------------------------------------
    #
    # /healthz answers as soon as uvicorn binds, which is correct for the
    # gateway: the job flow is asynchronous and nothing blocks on the model
    # warming. But it means a green health check says the GATEWAY is alive and
    # says nothing whatsoever about the workers behind it.
    #
    # That combination is the worst failure this service can have on a demo. If
    # the inference worker dies at boot -- OOM-killed on a small tier, say --
    # the platform sees a healthy service, the gateway accepts the upload, the
    # job lands in Redis, and nothing ever picks it up. No error, no failed
    # status, no red anything. The client watches a spinner forever and it looks
    # exactly like a model thinking hard.
    #
    # So each worker beats a key with a TTL, and /healthz reports a worker as
    # dead once its key expires. A platform health check then restarts the
    # container instead of routing traffic into a black hole.

    HEARTBEAT_TTL_SECONDS = 45

    @staticmethod
    def _heartbeat_key(topic: str) -> str:
        return f"df:worker:heartbeat:{topic}"

    def beat(self, topic: str) -> None:
        """Refresh this worker's liveness key. Cheap enough to call every poll.

        TTL is ~9x the 5s poll interval, so a worker busy with one slow
        inference (B7 on CPU is ~0.3-3s per face, and a batch of eight took
        5.1s) is not reported dead for being slow. Liveness, not latency.
        """
        try:
            self.r.set(
                self._heartbeat_key(topic), str(time.time()),
                ex=self.HEARTBEAT_TTL_SECONDS,
            )
        except Exception:
            # A heartbeat failure must never take down a worker that is
            # otherwise fine. Redis being unreachable will surface through the
            # queue on the next poll anyway, which is the honest signal.
            log.warning("heartbeat failed for topic=%s", topic, exc_info=True)

    def worker_alive(self, topic: str) -> bool:
        return bool(self.r.exists(self._heartbeat_key(topic)))

    def publish(self, job_id: str, status: str, **extra: Any) -> dict:
        """Write the key, then publish. Order matters: a client that gets the
        push and immediately polls must not read a staler value than the push.
        """
        doc = {"job_id": job_id, "status": status, "updated_at": time.time(), **extra}
        payload = json.dumps(doc)
        pipe = self.r.pipeline()
        pipe.set(status_key(job_id), payload, ex=STATUS_TTL_SECONDS)
        pipe.publish(channel(job_id), payload)
        pipe.execute()
        return doc

    def read(self, job_id: str) -> dict | None:
        raw = self.r.get(status_key(job_id))
        return json.loads(raw) if raw else None

    def subscribe(self, job_id: str):
        pubsub = self.r.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(channel(job_id))
        return pubsub


def assert_persistence_enabled(client: Any) -> None:
    """Refuse to run against a Redis that would lose in-flight job state.

    A default `redis:7` with no flags keeps everything in memory. Job status and
    the queue both live here, so a restart would strand every running job with
    no way for a client to learn what happened.
    """
    try:
        cfg = client.config_get("appendonly")
        aof_on = str(cfg.get("appendonly", "no")).lower() == "yes"
        save_cfg = client.config_get("save").get("save", "")
        rdb_on = bool(save_cfg and save_cfg.strip())
    except Exception as exc:  # CONFIG disabled on managed Redis
        log.warning("could not verify redis persistence (%s) -- verify manually", exc)
        return

    if not (aof_on or rdb_on):
        raise RuntimeError(
            "Redis has neither AOF nor RDB persistence enabled. In-flight job "
            "state would be lost on restart. Start redis with "
            "`--appendonly yes --appendfsync everysec` (see docker-compose.yml)."
        )
    log.info("redis persistence ok (aof=%s rdb=%s)", aof_on, rdb_on)
