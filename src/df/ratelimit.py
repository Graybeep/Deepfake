"""Ingress rate limiting (Tier 1).

Token bucket in Redis, evaluated in a Lua script so check-and-consume is atomic
-- a read-then-write version lets concurrent requests both see the same token
and both pass.

Applied to job creation, which is the expensive path: every accepted job costs
GPU time. Status reads are cheap and get a looser bucket.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from df.config import settings

# KEYS[1] = bucket key
# ARGV = capacity, refill_per_sec, now, cost, ttl
_SCRIPT = """
local key       = KEYS[1]
local capacity  = tonumber(ARGV[1])
local refill    = tonumber(ARGV[2])
local now       = tonumber(ARGV[3])
local cost      = tonumber(ARGV[4])
local ttl       = tonumber(ARGV[5])

local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local ts     = tonumber(bucket[2])

if tokens == nil then
  tokens = capacity
  ts = now
end

local elapsed = math.max(0, now - ts)
tokens = math.min(capacity, tokens + elapsed * refill)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, ttl)

local retry_after = 0
if allowed == 0 and refill > 0 then
  retry_after = (cost - tokens) / refill
end

return {allowed, tostring(tokens), tostring(retry_after)}
"""


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    tokens_remaining: float
    retry_after_seconds: float


class RateLimiter:
    def __init__(
        self,
        client: Any | None = None,
        *,
        capacity: int | None = None,
        refill_per_sec: float | None = None,
    ) -> None:
        if client is None:
            import redis

            client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.r = client
        self.capacity = capacity if capacity is not None else settings.ratelimit_capacity
        self.refill = (
            refill_per_sec if refill_per_sec is not None else settings.ratelimit_refill_per_sec
        )
        self._script = self.r.register_script(_SCRIPT)

    def check(self, identity: str, *, cost: float = 1.0) -> RateLimitDecision:
        # Bucket must refill fully within its own TTL or a returning client
        # would find an expired (i.e. full) bucket sooner than it earned one.
        ttl = int(self.capacity / self.refill) + 60 if self.refill > 0 else 3600
        allowed, tokens, retry_after = self._script(
            keys=[f"rl:{identity}"],
            args=[self.capacity, self.refill, time.time(), cost, ttl],
        )
        return RateLimitDecision(
            allowed=bool(int(allowed)),
            tokens_remaining=float(tokens),
            retry_after_seconds=float(retry_after),
        )


class InMemoryRateLimiter:
    """Same token-bucket maths without Redis, for tests and single-process dev."""

    def __init__(self, *, capacity: int = 30, refill_per_sec: float = 0.5) -> None:
        self.capacity = capacity
        self.refill = refill_per_sec
        self._buckets: dict[str, tuple[float, float]] = {}

    def check(self, identity: str, *, cost: float = 1.0, now: float | None = None) -> RateLimitDecision:
        now = now if now is not None else time.time()
        tokens, ts = self._buckets.get(identity, (float(self.capacity), now))
        tokens = min(self.capacity, tokens + max(0.0, now - ts) * self.refill)

        if tokens >= cost:
            tokens -= cost
            self._buckets[identity] = (tokens, now)
            return RateLimitDecision(True, tokens, 0.0)

        self._buckets[identity] = (tokens, now)
        retry_after = (cost - tokens) / self.refill if self.refill > 0 else 3600.0
        return RateLimitDecision(False, tokens, retry_after)
