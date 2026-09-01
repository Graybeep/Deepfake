"""Which address the rate limiter buckets on, behind a proxy.

The bug this closes, `measured: yes` 2026-09-01 on the deployed service: 45
rapid POST /v1/jobs returned 45x 201 and no 429. The limiter keyed on the socket
peer, which behind a platform proxy is the proxy — and the pool ROTATES. The
gateway observed 20+ distinct source IPs (100.64.0.2-.22), so requests spread
across many near-fresh buckets and nothing ever accumulated. Not coarse
limiting; none.

The dangerous fix is "just read X-Forwarded-For". That header is client-supplied
and only its rightmost entries are trustworthy, because each proxy APPENDS the
peer it received from. Taking the leftmost entry lets any caller pick its own
bucket, which is worse than no limiting because it looks like protection.

So: N trusted hops, and the client is the Nth entry from the right. Every
failure path falls back to the socket peer — this code runs on every request,
and a parsing bug that collapsed everyone into one bucket would 429 the world.

# In-process. The live counterpart is a burst against the deployed URL with
# DF_TRUSTED_PROXY_HOPS set, checked for 429.
"""
from __future__ import annotations

import pytest

from df.config import Settings
from df.gateway.app import _client_ip, identity_of


class FakeRequest:
    def __init__(self, peer: str | None, headers: dict[str, str] | None = None):
        self.client = type("C", (), {"host": peer})() if peer else None
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}


@pytest.fixture
def hops(monkeypatch):
    def _set(n: int):
        monkeypatch.setenv("DF_TRUSTED_PROXY_HOPS", str(n))
        monkeypatch.setattr("df.gateway.app.settings", Settings())
    return _set


# --- the default: trust nothing --------------------------------------------


def test_zero_hops_ignores_the_header_entirely(hops):
    """Correct on a bare host, and safe everywhere. A header present but
    untrusted must not influence the bucket at all."""
    hops(0)
    req = FakeRequest("203.0.113.9", {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"})

    assert _client_ip(req) == "203.0.113.9"


# --- the trusted case -------------------------------------------------------


def test_one_hop_takes_the_rightmost_entry(hops):
    """With a single trusted proxy, that proxy appended the address it saw, so
    the rightmost entry is the real client."""
    hops(1)
    req = FakeRequest("10.0.0.7", {"X-Forwarded-For": "1.2.3.4, 198.51.100.22"})

    assert _client_ip(req) == "198.51.100.22"


def test_two_hops_takes_the_second_from_the_right(hops):
    hops(2)
    req = FakeRequest("10.0.0.7", {"X-Forwarded-For": "9.9.9.9, 198.51.100.22, 10.0.0.3"})

    assert _client_ip(req) == "198.51.100.22"


def test_a_spoofed_prefix_cannot_choose_its_own_bucket(hops):
    """The attack the rightmost rule defends against. A client sending its own
    X-Forwarded-For gets those values pushed LEFT as real proxies append, so
    they never land in the trusted position."""
    hops(1)
    spoofed = FakeRequest(
        "10.0.0.7",
        # Everything before the last entry is attacker-supplied.
        {"X-Forwarded-For": "evil-1, evil-2, 198.51.100.22"},
    )

    assert _client_ip(spoofed) == "198.51.100.22"


def test_two_requests_from_one_client_share_a_bucket_across_rotating_proxies(hops):
    """The actual production failure, in miniature. Same client, two different
    proxy IPs — they must land on the SAME key, or the limiter never
    accumulates."""
    hops(1)
    a = FakeRequest("100.64.0.2", {"X-Forwarded-For": "198.51.100.22"})
    b = FakeRequest("100.64.0.19", {"X-Forwarded-For": "198.51.100.22"})

    assert identity_of(a) == identity_of(b) == "ip:198.51.100.22"


def test_different_clients_behind_one_proxy_get_different_buckets(hops):
    """The positive control. A fix that collapsed everyone onto one key would
    pass the test above and then 429 every user at once."""
    hops(1)
    a = FakeRequest("100.64.0.2", {"X-Forwarded-For": "198.51.100.22"})
    b = FakeRequest("100.64.0.2", {"X-Forwarded-For": "203.0.113.77"})

    assert identity_of(a) != identity_of(b)


# --- every failure path falls back, never to a shared constant --------------


@pytest.mark.parametrize("headers", [
    {},                                          # header absent
    {"X-Forwarded-For": ""},                     # header empty
    {"X-Forwarded-For": "   "},                  # whitespace only
    {"X-Forwarded-For": "not-an-ip"},            # unparseable
    {"X-Forwarded-For": "1.2.3.4"},              # fewer entries than hops (needs 2)
])
def test_malformed_or_missing_header_falls_back_to_the_peer(hops, headers):
    """Falling back is at worst the behaviour we already had. Returning a
    constant would put all traffic in one bucket and 429 everyone — this runs on
    every request, so the failure mode has to be the harmless one."""
    hops(2)
    req = FakeRequest("203.0.113.9", headers)

    assert _client_ip(req) == "203.0.113.9"


def test_ipv6_mapped_and_plain_share_one_bucket(hops):
    """Normalised, so ::ffff:1.2.3.4 and 1.2.3.4 do not get separate buckets —
    which would be a one-header way to double an allowance."""
    hops(1)
    mapped = FakeRequest("10.0.0.7", {"X-Forwarded-For": "::ffff:198.51.100.22"})
    plain = FakeRequest("10.0.0.7", {"X-Forwarded-For": "198.51.100.22"})

    assert _client_ip(mapped) == _client_ip(plain)


def test_no_client_at_all_still_returns_a_key(hops):
    """ASGI can present request.client as None. It must not raise on a path that
    every request goes through."""
    hops(0)

    assert _client_ip(FakeRequest(None)) == "unknown"


# --- api key still wins -----------------------------------------------------


def test_an_api_key_identifies_a_caller_across_addresses(hops):
    """A key follows the caller between networks, which an IP cannot."""
    hops(1)
    a = FakeRequest("100.64.0.2", {"X-Forwarded-For": "1.1.1.1", "x-api-key": "abc"})
    b = FakeRequest("100.64.0.9", {"X-Forwarded-For": "2.2.2.2", "x-api-key": "abc"})

    assert identity_of(a) == identity_of(b) == "key:abc"
