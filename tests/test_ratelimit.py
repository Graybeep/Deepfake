from __future__ import annotations

from df.ratelimit import InMemoryRateLimiter


def test_bucket_allows_up_to_capacity_then_blocks():
    rl = InMemoryRateLimiter(capacity=5, refill_per_sec=0.0)

    assert all(rl.check("client", now=100.0).allowed for _ in range(5))
    assert rl.check("client", now=100.0).allowed is False


def test_tokens_refill_over_time():
    rl = InMemoryRateLimiter(capacity=5, refill_per_sec=1.0)
    for _ in range(5):
        rl.check("client", now=100.0)

    assert rl.check("client", now=100.5).allowed is False
    assert rl.check("client", now=102.0).allowed is True


def test_refill_is_capped_at_capacity():
    rl = InMemoryRateLimiter(capacity=3, refill_per_sec=1.0)
    rl.check("client", now=100.0)

    # An hour idle must not bank an hour's worth of tokens.
    for _ in range(3):
        assert rl.check("client", now=3700.0).allowed is True
    assert rl.check("client", now=3700.0).allowed is False


def test_identities_have_separate_buckets():
    rl = InMemoryRateLimiter(capacity=2, refill_per_sec=0.0)
    for _ in range(2):
        rl.check("a", now=100.0)

    assert rl.check("a", now=100.0).allowed is False
    assert rl.check("b", now=100.0).allowed is True


def test_retry_after_is_reported_when_blocked():
    rl = InMemoryRateLimiter(capacity=1, refill_per_sec=0.5)
    rl.check("client", now=100.0)

    decision = rl.check("client", now=100.0)

    assert decision.allowed is False
    assert decision.retry_after_seconds == 2.0


def test_cheap_reads_can_use_fractional_cost():
    """Status polling shares the limiter but must not exhaust the ingest budget."""
    rl = InMemoryRateLimiter(capacity=1, refill_per_sec=0.0)

    assert all(rl.check("client", cost=0.1, now=100.0).allowed for _ in range(10))
    assert rl.check("client", cost=0.1, now=100.0).allowed is False
